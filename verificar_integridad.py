from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
MODOS = ('aprendizaje', 'conversaciones')
CASOS = {
    'aprendizaje': [
        ('¿Qué significa «Piyali» en español?', 'hola'),
        ('¿Cómo se dice «buenos días» en náhuatl de la Huasteca?', 'Mah cualli tonalli / Tonalti'),
    ],
    'conversaciones': [
        ('Piyali', 'Piyali'),
        ('Kenihkatzah tiitztok?', 'Kualli niitztok'),
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cargar_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verificar_carpeta(mode: str, inferir: bool) -> dict:
    directory = BASE / 'models' / mode
    config = cargar_json(directory / 'config.json')
    report = cargar_json(directory / 'reporte_entrenamiento.json')
    tokenizer = cargar_json(directory / 'tokenizador.json')
    tags = cargar_json(directory / 'etiquetas.json')['tags']
    responses = cargar_json(directory / 'respuestas.json')
    assert config['modo'] == mode
    assert report['modo'] == mode
    assert set(tags) == set(responses)
    hashes = {}
    for name, expected in config['sha256'].items():
        actual = sha256(directory / name)
        assert actual == expected, f'{mode}/{name}: esperado {expected}, actual {actual}'
        hashes[name] = actual
        if name != 'modelo.h5':
            dataset_file = BASE / 'dataset' / 'tensores' / mode / name
            assert sha256(dataset_file) == actual, f'{mode}/{name}: models y dataset no coinciden'
    test_metric = report['metricas']['test']
    reported_accuracy = test_metric['accuracy'] if test_metric else None
    assert config.get('test_accuracy') == reported_accuracy
    result = {
        'modo': mode,
        'hashes': hashes,
        'clases': len(tags),
        'longitud': tokenizer['max_length'],
        'parametros': report['parametros'],
        'test_accuracy': reported_accuracy,
        'inferencia': [],
    }
    if inferir:
        from motor import Predictor
        predictor = Predictor(directory)
        assert predictor.model.input_shape == (None, tokenizer['max_length'])
        assert predictor.model.output_shape[-1] == len(tags)
        assert predictor.model.count_params() == report['parametros']
        for text, expected_answer in CASOS[mode]:
            prediction = predictor.predict(text)
            assert prediction['accepted'], (mode, text, prediction)
            assert prediction['answer'] == expected_answer, (mode, text, prediction)
            result['inferencia'].append({
                'entrada': text,
                'respuesta': prediction['answer'],
                'puntuacion': prediction['confidence'],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inferir', action='store_true')
    args = parser.parse_args()
    result = [verificar_carpeta(mode, args.inferir) for mode in MODOS]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
