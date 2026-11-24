from pathlib import Path
import json
import tempfile
import threading
import unittest

import numpy as np

from entrenar import cargar_datos
from motor import Predictor, codificar, normalizar


BASE = Path(__file__).resolve().parent
DATASET = BASE.parent / 'dataset'


def escribir_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class TokenizacionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizador = {
            'max_length': 12, 'pad_id': 0, 'unk_id': 1,
            'tokens': ['<PAD>', '<UNK>', ' ', 'á', 'h', "'", 'n', 'a'],
        }

    def test_unicode_espacios_y_signos_con_significado_se_conservan(self):
        self.assertEqual(normalizar(" \tA\u0301H\n N'A  "), "áh n'a")
        X, unknown = codificar(" \tA\u0301H\n N'A  ", self.tokenizador)
        np.testing.assert_array_equal(X[0], [3, 4, 2, 6, 5, 7, 0, 0, 0, 0, 0, 0])
        self.assertEqual(X.dtype, np.int32)
        self.assertEqual(unknown, 0)

    def test_desconocidos_y_relleno_son_distintos(self):
        X, unknown = codificar('ax', self.tokenizador)
        np.testing.assert_array_equal(X[0, :3], [7, 1, 0])
        self.assertEqual(unknown, 0.5)

    def test_rechaza_vacio_y_solo_puntuacion(self):
        for text in ('', '  \n\t', '¿?!', '🙂'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                codificar(text, self.tokenizador)

    def test_limite_es_sobre_texto_normalizado_y_no_trunca(self):
        X, _ = codificar('  ' + 'a' * 12 + '  ', self.tokenizador)
        self.assertEqual(X.shape, (1, 12))
        self.assertTrue(np.all(X == 7))
        with self.assertRaisesRegex(ValueError, 'más corta'):
            codificar('a' * 13, self.tokenizador)


class DatosRealesTests(unittest.TestCase):
    def test_ambos_catalogos_tienen_mapeo_identico_en_entrenamiento_e_inferencia(self):
        for modo in ('aprendizaje', 'conversaciones'):
            with self.subTest(modo=modo):
                _, arrays, tokenizer, tags, responses, examples = cargar_datos(DATASET, modo)
                self.assertGreater(len(tags), 1)
                self.assertEqual(set(arrays['y_train']), set(range(len(tags))))
                for split, rows in examples.items():
                    for i, row in enumerate(rows):
                        encoded, _ = codificar(row['text'], tokenizer)
                        np.testing.assert_array_equal(encoded[0], arrays[f'X_{split}'][i])
                        self.assertEqual(tags[int(arrays[f'y_{split}'][i])], row['tag'])
                        self.assertIsInstance(responses[row['tag']][0], str)
                if modo == 'conversaciones':
                    self.assertEqual(len(arrays['y_validation']), 0)
                    self.assertEqual(len(arrays['y_test']), 0)
                else:
                    self.assertGreater(len(arrays['y_validation']), 0)
                    self.assertGreater(len(arrays['y_test']), 0)


class ValidacionDatosTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nahuatl_tests_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'tensores' / 'prueba'
        self.directory.mkdir(parents=True)
        self.tokenizer = {
            'max_length': 4, 'pad_id': 0, 'unk_id': 1,
            'tokens': ['<PAD>', '<UNK>', 'a', 'b'],
        }
        escribir_json(self.directory / 'tokenizador.json', self.tokenizer)
        escribir_json(self.directory / 'etiquetas.json', {'tags': ['saludo', 'despedida']})
        escribir_json(self.directory / 'respuestas.json', {'saludo': ['Hola'], 'despedida': ['Adiós']})
        escribir_json(self.directory / 'informe.json', {'entrada': 'intents.json', 'sha256_entrada': 'unused'})
        self.rows = {
            'train': [{'text': 'ab', 'tag': 'saludo'}, {'text': 'ba', 'tag': 'despedida'}],
            'validation': [{'text': 'aa', 'tag': 'saludo'}],
            'test': [{'text': 'bb', 'tag': 'despedida'}],
        }
        self.arrays = {}
        for split, rows in self.rows.items():
            X = np.concatenate([codificar(row['text'], self.tokenizer)[0] for row in rows])
            self.arrays[f'X_{split}'] = X
            self.arrays[f'y_{split}'] = np.array([0 if row['tag'] == 'saludo' else 1 for row in rows], dtype=np.int32)
            self.arrays[f'mask_{split}'] = X != 0
            self.escribir_filas(split)
        self.guardar_npz()

    def escribir_filas(self, split):
        path = self.directory / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in self.rows[split]), encoding='utf-8')

    def guardar_npz(self):
        np.savez_compressed(self.directory / 'datos.npz', **self.arrays)

    def cargar(self):
        return cargar_datos(self.root, 'prueba')

    def test_fixture_valido(self):
        self.assertEqual(len(self.cargar()[3]), 2)

    def test_archivo_faltante(self):
        (self.directory / 'tokenizador.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Faltan archivos'):
            self.cargar()

    def test_json_invalido(self):
        (self.directory / 'etiquetas.json').write_text('{malformado', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            self.cargar()

    def test_json_bom_se_admite(self):
        (self.directory / 'tokenizador.json').write_text(json.dumps(self.tokenizer), encoding='utf-8-sig')
        self.assertEqual(self.cargar()[2], self.tokenizer)

    def test_intents_modificados_requieren_regeneracion(self):
        escribir_json(self.root / 'intents.json', {'intents': []})
        with self.assertRaisesRegex(ValueError, 'intents cambiaron'):
            self.cargar()

    def test_mascara_corrupta(self):
        self.arrays['mask_train'][0, 0] = False
        self.guardar_npz()
        with self.assertRaisesRegex(ValueError, 'máscaras'):
            self.cargar()

    def test_ids_fuera_del_vocabulario(self):
        self.arrays['X_train'][0, 0] = 99
        self.guardar_npz()
        with self.assertRaisesRegex(ValueError, 'vocabulario'):
            self.cargar()

    def test_etiquetas_fuera_del_catalogo(self):
        self.arrays['y_train'][0] = 99
        self.guardar_npz()
        with self.assertRaisesRegex(ValueError, 'fuera del catálogo'):
            self.cargar()

    def test_flotantes_no_se_aceptan_como_ids(self):
        self.arrays['X_train'] = self.arrays['X_train'].astype(np.float32)
        self.guardar_npz()
        with self.assertRaisesRegex(ValueError, 'enteros'):
            self.cargar()

    def test_matriz_faltante(self):
        del self.arrays['mask_test']
        self.guardar_npz()
        with self.assertRaisesRegex(ValueError, 'Faltan matrices'):
            self.cargar()

    def test_texto_jsonl_no_coincide_con_sus_numeros(self):
        self.rows['train'][0]['text'] = 'aa'
        self.escribir_filas('train')
        with self.assertRaisesRegex(ValueError, 'no corresponde'):
            self.cargar()

    def test_no_admite_contaminacion_entre_train_y_test(self):
        self.rows['test'] = [self.rows['train'][0].copy()]
        self.arrays['X_test'] = self.arrays['X_train'][:1].copy()
        self.arrays['y_test'] = self.arrays['y_train'][:1].copy()
        self.arrays['mask_test'] = self.arrays['mask_train'][:1].copy()
        self.escribir_filas('test')
        self.guardar_npz()
        with self.assertRaisesRegex(ValueError, 'duplicados entre las particiones'):
            self.cargar()


class ModeloSimulado:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def __call__(self, X, training):
        self.calls += 1
        if training is not False:
            raise AssertionError('La inferencia debe desactivar dropout.')
        return np.asarray([self.scores], dtype=np.float32)


class InferenciaTests(unittest.TestCase):
    def predictor(self, scores=(0.85, 0.15)):
        predictor = Predictor.__new__(Predictor)
        predictor.tokenizador = {
            'max_length': 12, 'pad_id': 0, 'unk_id': 1,
            'tokens': ['<PAD>', '<UNK>', 'a', 'b'],
        }
        predictor.tags = ['saludo', 'despedida']
        predictor.responses = {'saludo': ['Piyali'], 'despedida': ['Timoittaseh']}
        predictor.config = {'umbral': 0.6, 'margen': 0.15}
        predictor.model = ModeloSimulado(scores)
        predictor._lock = threading.Lock()
        return predictor

    def test_respuesta_sigue_el_mapa_de_etiquetas(self):
        result = self.predictor((0.1, 0.9)).predict('ab')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['tag'], 'despedida')
        self.assertEqual(result['answer'], 'Timoittaseh')
        self.assertAlmostEqual(result['confidence'], 0.9, places=6)

    def test_baja_puntuacion_no_devuelve_respuesta_del_candidato(self):
        result = self.predictor((0.55, 0.45)).predict('ab')
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['tag'])
        self.assertEqual(result['candidate_tag'], 'saludo')
        self.assertNotEqual(result['answer'], 'Piyali')

    def test_margen_puede_rechazar_aunque_se_supere_umbral(self):
        predictor = self.predictor((0.55, 0.45))
        predictor.config['umbral'] = 0.5
        self.assertFalse(predictor.predict('ab')['accepted'])

    def test_fuera_de_vocabulario_no_ejecuta_el_modelo(self):
        predictor = self.predictor()
        result = predictor.predict('漢字')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['source'], 'fuera_del_vocabulario')
        self.assertEqual(predictor.model.calls, 0)

    def test_tipo_de_entrada_incorrecto(self):
        for text in (None, 12, {}, ['ab']):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.predictor().predict(text)

    def test_salida_no_finita_o_de_dimension_incorrecta(self):
        for scores in ((float('nan'), 0.1), (float('inf'), 0.1), (1.0,)):
            with self.subTest(scores=scores), self.assertRaisesRegex(RuntimeError, 'salida inválida'):
                self.predictor(scores).predict('ab')

    def test_carpeta_sin_modelo_indica_como_entrenarlo(self):
        with tempfile.TemporaryDirectory(prefix='nahuatl_modelo_') as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'Ejecuta entrenar.py primero'):
                Predictor(directory)

    def test_manifiesto_modificado_se_rechaza_antes_de_cargar_tensorflow(self):
        with tempfile.TemporaryDirectory(prefix='nahuatl_modelo_') as directory:
            root = Path(directory)
            for name in ('modelo.h5', 'tokenizador.json', 'etiquetas.json', 'respuestas.json'):
                (root / name).write_bytes(b'changed')
            escribir_json(root / 'config.json', {'sha256': {'modelo.h5': 'digest incorrecto'}})
            with self.assertRaisesRegex(ValueError, 'no corresponde a esta versión'):
                Predictor(root)


if __name__ == '__main__':
    unittest.main()
