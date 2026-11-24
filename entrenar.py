from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import numpy as np
from motor import normalizar, validar_respuestas

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def cargar_datos(dataset_dir, modo):
    directory = dataset_dir / 'tensores' / modo
    needed = ('datos.npz', 'tokenizador.json', 'etiquetas.json', 'respuestas.json', 'informe.json')
    missing = [name for name in needed if not (directory / name).is_file()]
    if missing:
        raise ValueError(f'Faltan archivos en {directory}: {missing}. Ejecuta preparar_tensores.py.')
    tokenizer = read(directory / 'tokenizador.json')
    tags = read(directory / 'etiquetas.json')['tags']
    responses = read(directory / 'respuestas.json')
    original = read(directory / 'informe.json')
    source = dataset_dir / original['entrada']
    if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != original['sha256_entrada']:
        raise ValueError('Los intents cambiaron después de crear los tensores. Ejecuta preparar_tensores.py y vuelve a entrenar.')
    if len(tags) < 2 or len(tags) != len(set(tags)):
        raise ValueError('Las etiquetas o las respuestas no son válidas.')
    validar_respuestas(tags, responses)
    if tokenizer['tokens'][:2] != ['<PAD>', '<UNK>'] or tokenizer['pad_id'] != 0 or tokenizer['unk_id'] != 1:
        raise ValueError('El tokenizador debe usar relleno 0 y desconocido 1.')
    with np.load(directory / 'datos.npz', allow_pickle=False) as file:
        arrays = {k: file[k].copy() for k in file.files}
    seen = set()
    examples = {}
    for split in ('train', 'validation', 'test'):
        X, y, mask = (arrays.get(f'{k}_{split}') for k in ('X', 'y', 'mask'))
        if X is None or y is None or mask is None:
            raise ValueError('Faltan matrices o máscaras en datos.npz.')
        if X.ndim != 2 or X.shape[1] != tokenizer['max_length'] or y.shape != (len(X),):
            raise ValueError(f'Las dimensiones de {split} son incorrectas.')
        if X.dtype.kind not in 'iu' or y.dtype.kind not in 'iu':
            raise ValueError('Las entradas y etiquetas deben ser enteros.')
        if X.size and (X.min() < 0 or X.max() >= len(tokenizer['tokens'])):
            raise ValueError('Hay caracteres fuera del vocabulario.')
        if y.size and (y.min() < 0 or y.max() >= len(tags)):
            raise ValueError('Hay etiquetas fuera del catálogo.')
        if mask.shape != X.shape or not np.array_equal(mask, X != 0):
            raise ValueError('Las máscaras no corresponden a las matrices.')
        jsonl = directory / f'{split}.jsonl'
        if not jsonl.is_file():
            raise ValueError(f'Falta {jsonl.name}; regenera los tensores para verificar su correspondencia.')
        rows = [json.loads(line) for line in jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != len(X):
            raise ValueError(f'El número de textos no coincide con {split}.')
        keys = set()
        mapa = {c: i for i, c in enumerate(tokenizer['tokens'])}
        for i, row in enumerate(rows):
            text = normalizar(row['text'])
            ids = [mapa.get(c, 1) for c in text]
            if len(ids) > X.shape[1]:
                raise ValueError('Hay frases truncadas. Amplía la longitud de los tensores antes de entrenar.')
            expected = np.zeros(X.shape[1], dtype=np.int32)
            expected[:len(ids)] = ids
            if not np.array_equal(expected, X[i]) or row['tag'] != tags[int(y[i])]:
                raise ValueError(f'El texto o etiqueta de la fila {i} en {split} no corresponde a sus números.')
            keys.add(text)
        if seen & keys:
            raise ValueError('Hay textos duplicados entre las particiones de entrenamiento y evaluación.')
        seen.update(keys)
        examples[split] = rows
    if not len(arrays['y_train']) or set(map(int, arrays['y_train'])) != set(range(len(tags))):
        raise ValueError('Cada etiqueta debe tener por lo menos un ejemplo de entrenamiento.')
    return directory, arrays, tokenizer, tags, responses, examples


def crear_modelo(tf, length, vocabulary, classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(length,), dtype='int32', name='caracteres'),
        tf.keras.layers.Embedding(vocabulary, 32, mask_zero=False, name='representacion'),
        tf.keras.layers.Conv1D(96, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(96, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(classes, activation='softmax', name='intencion'),
    ], name='clasificador_nahuatl')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def entrenar_modo(tf, args, modo):
    directory, a, tokenizer, tags, responses, examples = cargar_datos(args.dataset_dir, modo)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    output = args.output_dir / modo
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{modo}_entrenando_', dir=args.output_dir))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1

    def data(split, shuffle=False):
        d = tf.data.Dataset.from_tensor_slices((a[f'X_{split}'], a[f'y_{split}']))
        if shuffle:
            d = d.shuffle(len(a['y_train']), seed=args.seed, reshuffle_each_iteration=True)
        return d.batch(args.batch_size).with_options(options).prefetch(1)

    has_val = bool(len(a['y_validation']))
    monitor = 'val_loss' if has_val else 'loss'
    model = crear_modelo(tf, tokenizer['max_length'], len(tokenizer['tokens']), len(tags))

    class Progreso(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                detail = ', '.join(f'{k}={float(v):.4f}' for k, v in (logs or {}).items())
                print(f'[{modo}] época {epoch+1}/{args.epochs}: {detail}', flush=True)

    print(f'\nEntrenando {modo}: {len(tags)} clases, {len(a["y_train"])} ejemplos; salida {output}', flush=True)
    callbacks = [Progreso(),
                 tf.keras.callbacks.CSVLogger(str(stage / 'historial.csv')),
                 tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=args.patience,
                                                  restore_best_weights=True, min_delta=0.001),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                                      patience=max(4, args.patience // 2), min_lr=0.00005)]
    history = model.fit(data('train', True), validation_data=data('validation') if has_val else None,
                        epochs=args.epochs, callbacks=callbacks, verbose=0)
    metrics = {}
    for split in ('train', 'validation', 'test'):
        metrics[split] = model.evaluate(data(split), verbose=0, return_dict=True) if len(a[f'y_{split}']) else None
    model.save(stage / 'modelo.h5', include_optimizer=True)
    reloaded = tf.keras.models.load_model(stage / 'modelo.h5', compile=False)
    probe = a['X_test'][:8] if len(a['X_test']) else a['X_train'][:8]
    before = np.asarray(model(probe, training=False))
    after = np.asarray(reloaded(probe, training=False))
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
    for name in ('tokenizador.json', 'etiquetas.json', 'respuestas.json'):
        shutil.copy2(directory / name, stage / name)
    report = {
        'modo': modo, 'tensorflow': tf.__version__, 'numpy': np.__version__,
        'seed': args.seed, 'epocas_ejecutadas': len(history.history['loss']),
        'metricas': metrics, 'parametros': model.count_params(),
        'filas': {s: len(a[f'y_{s}']) for s in ('train', 'validation', 'test')},
        'historial': {k: [float(x) for x in v] for k, v in history.history.items()},
        'h5_recargado_y_verificado': True,
        'diferencia_maxima_h5': float(np.max(np.abs(before-after))),
        'limitacion': 'Clasificación de un catálogo pequeño. Los splits educativos comparten expresiones base y difieren en plantillas. No mide traducción general ni calidad lingüística.' if has_val else
                      'Solo se pudo evaluar entrenamiento: no hay validación ni prueba independiente para conversaciones.',
        'umbral': 'Heurístico, ajustable; no está calibrado con datos fuera del dominio.',
    }
    if len(a['X_test']):
        probabilities = np.asarray(model.predict(data('test'), verbose=0))
        predictions = probabilities.argmax(axis=1)
        errores = []
        for i, row in enumerate(examples['test']):
            if predictions[i] != a['y_test'][i]:
                errores.append({'texto': row['text'], 'esperada': row['tag'],
                                'predicha': tags[int(predictions[i])],
                                'puntuacion': float(probabilities[i, predictions[i]])})
        write(stage / 'errores_test.json', errores)
    write(stage / 'reporte_entrenamiento.json', report)
    hashes = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
              for name in ('modelo.h5', 'tokenizador.json', 'etiquetas.json', 'respuestas.json')}
    write(stage / 'config.json', {
        'version': 1, 'modo': modo, 'fecha_entrenamiento': datetime.now(timezone.utc).isoformat(),
        'umbral': args.umbral, 'margen': args.margen, 'sha256': hashes,
        'test_accuracy': metrics['test']['accuracy'] if metrics['test'] else None,
    })
    if output.exists():
        backup = output.with_name(modo + '_anterior_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.rename(backup)
        print(f'Versión anterior conservada: {backup}', flush=True)
    stage.rename(output)
    print(f'Guardado y verificado: {output / "modelo.h5"}', flush=True)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modo', choices=['aprendizaje', 'conversaciones', 'ambos'], default='ambos')
    parser.add_argument('--dataset-dir', type=Path, default=BASE / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'models')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--umbral', type=float, default=0.60)
    parser.add_argument('--margen', type=float, default=0.15)
    parser.add_argument('--solo-verificar', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error('epochs, patience y batch-size deben ser positivos.')
    if not 0 <= args.umbral <= 1 or not 0 <= args.margen <= 1:
        parser.error('umbral y margen deben estar entre 0 y 1.')
    modes = ('aprendizaje', 'conversaciones') if args.modo == 'ambos' else (args.modo,)
    for mode in modes:
        cargar_datos(args.dataset_dir, mode)
        print(f'Datos de {mode}: verificados.', flush=True)
    if args.solo_verificar:
        return
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    for mode in modes:
        entrenar_modo(tf, args, mode)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
