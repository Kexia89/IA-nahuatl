from pathlib import Path
import hashlib
import json
import os
import threading
import unicodedata

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')


def leer_json(path):
    return Path(path).read_text(encoding='utf-8-sig')


def normalizar(texto):
    return ' '.join(unicodedata.normalize('NFC', texto).casefold().split())


def validar_respuestas(tags, responses):
    if not isinstance(responses, dict) or any(
        not isinstance(responses.get(tag), list) or not responses[tag]
        or any(not isinstance(s, str) or not s.strip() for s in responses[tag])
        for tag in tags
    ):
        raise ValueError('Cada etiqueta debe tener una lista de respuestas de texto no vacías.')


def codificar(texto, tokenizador):
    import numpy as np
    texto = normalizar(texto)
    longitud = tokenizador['max_length']
    if not texto or not any(c.isalnum() for c in texto):
        raise ValueError('Escribe una palabra o una pregunta para practicar.')
    if len(texto) > longitud:
        raise ValueError(f'Usa una frase más corta: este modelo admite hasta {longitud} caracteres normalizados.')
    mapa = {char: i for i, char in enumerate(tokenizador['tokens'])}
    ids = [mapa.get(char, tokenizador['unk_id']) for char in texto]
    X = np.full((1, longitud), tokenizador['pad_id'], dtype=np.int32)
    X[0, :len(ids)] = ids
    desconocidos = sum(n == tokenizador['unk_id'] for n in ids) / len(ids)
    return X, desconocidos


class Predictor:
    def __init__(self, model_dir):
        self.directory = Path(model_dir).resolve()
        needed = ('modelo.h5', 'tokenizador.json', 'etiquetas.json', 'respuestas.json', 'config.json')
        missing = [name for name in needed if not (self.directory / name).is_file()]
        if missing:
            raise FileNotFoundError('Faltan archivos del modelo: ' + ', '.join(missing) + '. Ejecuta entrenar.py primero.')
        self.config = json.loads(leer_json(self.directory / 'config.json'))
        for name, digest in self.config.get('sha256', {}).items():
            if name not in needed:
                raise ValueError('El manifiesto del modelo contiene un archivo inesperado.')
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{name} no corresponde a esta versión del modelo. Vuelve a entrenar o restaura su carpeta completa.')
        self.tokenizador = json.loads(leer_json(self.directory / 'tokenizador.json'))
        self.tags = json.loads(leer_json(self.directory / 'etiquetas.json'))['tags']
        self.responses = json.loads(leer_json(self.directory / 'respuestas.json'))
        validar_respuestas(self.tags, self.responses)
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError('Falta TensorFlow. Instala las dependencias de requirements.txt.') from exc
        self.model = tf.keras.models.load_model(self.directory / 'modelo.h5', compile=False)
        if self.model.input_shape != (None, self.tokenizador['max_length']):
            raise ValueError('El tokenizador no coincide con la entrada del modelo.')
        if self.model.output_shape[-1] != len(self.tags):
            raise ValueError('Las etiquetas no coinciden con la salida del modelo.')
        self._lock = threading.Lock()

    def info(self):
        return {
            'mode': self.config.get('modo'), 'classes': len(self.tags),
            'max_length': self.tokenizador['max_length'],
            'threshold': self.config['umbral'], 'margin': self.config['margen'],
            'trained_at': self.config.get('fecha_entrenamiento'),
            'test_accuracy': self.config.get('test_accuracy'),
            'score_note': 'Puntuación del clasificador; no mide exactitud de la lengua ni está calibrada.',
        }

    def predict(self, text):
        import numpy as np
        if not isinstance(text, str):
            raise ValueError('El mensaje debe ser texto.')
        X, unknown = codificar(text, self.tokenizador)
        if unknown > 0.30:
            return {'answer': 'No reconozco suficientes caracteres de esta entrada. Prueba con uno de los ejemplos.',
                    'tag': None, 'confidence': 0.0, 'accepted': False, 'source': 'fuera_del_vocabulario'}
        with self._lock:
            scores = np.asarray(self.model(X, training=False))[0]
        if scores.shape != (len(self.tags),) or not np.all(np.isfinite(scores)):
            raise RuntimeError('El modelo produjo una salida inválida.')
        ranked = np.argsort(scores)[::-1]
        best = int(ranked[0])
        confidence = float(scores[best])
        margin = confidence - float(scores[ranked[1]]) if len(ranked) > 1 else confidence
        accepted = confidence >= self.config['umbral'] and margin >= self.config['margen']
        return {
            'answer': self.responses[self.tags[best]][0] if accepted else
                      'No encuentro una intención suficientemente clara. Prueba una frase del catálogo o reformula la pregunta.',
            'tag': self.tags[best] if accepted else None,
            'candidate_tag': self.tags[best], 'confidence': confidence,
            'accepted': accepted, 'source': 'modelo_h5', 'margin': margin,
        }
