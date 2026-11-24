# Intents de náhuatl de la Huasteca

Dos conjuntos separados con la misma estructura del archivo de referencia:

```json
{"intents": [{"tag": "etiqueta", "patterns": ["entrada"], "responses": ["respuesta"]}]}
```

## Qué contiene esta entrega

| Archivo | Contenido | Cantidad |
|---|---|---:|
| `intents_conversaciones_nahuatl.json` | Entradas y respuestas en náhuatl; diálogos breves de práctica | 29 intents, 42 patrones |
| `intents_aprender_nahuatl.json` | Preguntas en español para aprender expresiones y consultar sus significados | 345 intents, 2,412 patrones |
| `traducciones_conversaciones_es.json` | Glosas de cada patrón y respuesta, con la misma etiqueta y orden que el archivo conversacional | 29 intents |
| `frases_documentadas.json` | Pares bilingües seleccionados, texto de origen y cambios | 174 pares distintos |
| `procedencia.json` | Fuentes y trazabilidad por etiqueta | Ambos conjuntos |
| `perfiles/` | Conversaciones separadas por fuente y escritura | UV, INALI y Axolotl occidental |
| `tensores/` | Matrices `.npz`, etiquetas, tokenizadores y particiones `.jsonl` | Dos conjuntos independientes |
| `auditoria.json` | Conteos y controles estructurales | Sin patrones idénticos entre etiquetas de un mismo archivo |

**El conjunto es una base documentada de práctica, no un corpus certificado por hablantes.**
El mayor volumen corresponde a preguntas didácticas creadas con plantillas españolas. Hay 174 pares bilingües, no 2,454 conversaciones independientes. Los intercambios en náhuatl son breves y su cobertura es limitada.

## Fidelidad y procedencia

Se consultó el [prontuario de INALI](https://site.inali.gob.mx/Micrositios/Prontuarios/nahuatlV.html), las [conversaciones de la Universidad Veracruzana](https://www.uv.mx/celulaode/alfabeto-nahuatl/tema3.html) y la distribución de [Axolotl en Py-Elotl](https://github.com/ElotlMX/py-elotl). Se conservaron expresiones documentadas; no se utilizó traducción automática para crear frases nuevas en náhuatl.

En Axolotl se localizaron 1,596 registros etiquetados con variantes de la Huasteca dentro de 16,111 registros totales. Se hizo una selección explícita de expresiones occidentales y se excluyeron desalineaciones evidentes, fragmentos dudosos y textos jurídicos. `descartes_ejemplos.json` muestra algunos problemas detectados. **Esto es selección editorial y cotejo textual, no validación lingüística exhaustiva.** Una traducción publicada también puede contener errores, depender del contexto o emplear una grafía local.

En conversaciones se emparejaron expresiones para situaciones de práctica. Ese emparejamiento es editorial y está señalado en `procedencia.json`; no debe atribuirse como diálogo literal a las fuentes. El personaje Juana y las situaciones de estudio o desplazamiento son escenarios de práctica, no identidad ni experiencias reales de un asistente.

Las fuentes de UV e INALI dicen «Huasteca» sin identificar una localidad concreta. Axolotl etiqueta la selección como **náhuatl de la Huasteca occidental, `nhw`**. Se conserva esa atribución; no se inventaron códigos ISO para las otras dos fuentes. Los archivos en `perfiles/` permiten trabajar con una sola escritura. No hay una ortografía unificada artificialmente: formas como `Piyali`, `Tlazohcamati` y `tlaskamati` mantienen su procedencia.

Los préstamos presentes en las fuentes se conservaron. Por ejemplo, no se sustituyeron `karpinteros`, `Bueno` ni otros préstamos por supuestas palabras «más puras». Algunas glosas son contextuales: no enseñan todos los significados posibles de una palabra.

## Antes de ampliar o entrenar

1. Elegir la comunidad y el perfil de escritura del modelo. La etiqueta regional Huasteca abarca diversidad local.
2. Revisar las equivalencias, la naturalidad de cada intercambio y el grado de cortesía con un hablante de esa comunidad. `validacion_hablante` está en `false` porque esa revisión aún no se realizó.
3. Recoger nuevas formulaciones reales por cada intención, con autorización de quienes aporten sus datos. No multiplicar ejemplos solo cambiando mayúsculas, comas o nombres.
4. Mantener las traducciones españolas fuera del entrenamiento conversacional si se desea responder exclusivamente en náhuatl.

Un ejemplo contiene el imperativo directo «¡Vete!» en un contexto entre personas cercanas; no es una fórmula general de cortesía. Eliminar esa intención si el asistente será de atención al público. Las notas por etiqueta describen este tipo de límites.

## Qué aprende un modelo con estos archivos

La estructura `tag/patterns/responses` es apropiada para **clasificar una entrada y escoger una respuesta asociada**. No describe turnos con memoria, participantes ni historia completa. Entrenar un clasificador con ella no enseña por sí solo traducción libre ni conversación general en náhuatl.

Cada patrón de una etiqueta admite la respuesta de esa etiqueta. Las traducciones equivalentes de fuentes diferentes se agruparon para evitar asignar etiquetas incompatibles a la misma consulta. En aprendizaje algunas respuestas presentan alternativas separadas por `/`; su procedencia queda en los metadatos.

No se añadió un `fallback` con entradas artificiales. Un clasificador tiende a asignar incluso entradas desconocidas a alguna clase. Para desplegarlo hace falta evaluar ejemplos fuera del dominio y definir un criterio de rechazo; no basta con fijar un umbral arbitrario.

## Usar los datos numéricos

Los `.npz` ya están preparados. `X_*` contiene IDs de caracteres, `y_*` IDs enteros de etiquetas y `mask_*` identifica las posiciones que contienen texto. Son matrices NumPy, listas para convertirse a tensores del framework elegido. No contienen pesos de un modelo entrenado.

```python
from pathlib import Path
import numpy as np
import tensorflow as tf

base = Path("nahuatl_huasteca_dataset")  # Ajustar a la carpeta extraída.
d = np.load(base / "tensores/aprendizaje/datos.npz", allow_pickle=False)
X_train = tf.convert_to_tensor(d["X_train"], dtype=tf.int32)
y_train = tf.convert_to_tensor(d["y_train"], dtype=tf.int32)
mask_train = tf.convert_to_tensor(d["mask_train"], dtype=tf.bool)
```

Para PyTorch, `torch.from_numpy(d["X_train"]).long()` y `torch.from_numpy(d["y_train"]).long()` producen tensores de índices. Cada conjunto tiene su propio tokenizador y mapa de etiquetas: no intercambiarlos.

`0` es relleno y `1` es carácter desconocido. El tokenizador conserva acentos, apóstrofos y letras; no aplica un stemmer español. Una capa Embedding debe considerar el relleno, por ejemplo con `mask_zero=True` en Keras. Para objetivos de clasificación las etiquetas son enteras, compatibles con una pérdida categórica dispersa. Las respuestas se mantienen en `respuestas.json` para la recuperación posterior.

La conversión NumPy fue ejecutada y verificada. Los fragmentos de TensorFlow y PyTorch son ejemplos de integración; esos frameworks no se instalaron ni se entrenó una red en esta entrega. Referencias: [conversión a tensores](https://www.tensorflow.org/guide/tensor) y [máscaras de Keras](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding).

Para regenerar las matrices después de editar los JSON:

```console
python -m pip install numpy
python preparar_tensores.py
```

Para un perfil particular:

```console
python preparar_tensores.py --input perfiles/intents_conversaciones_uv.json --output tensores/uv
```

## Particiones y evaluación

- **Aprendizaje:** separación por familias de plantillas en español. El vocabulario y la longitud máxima se ajustan solo con train. Los informes indican caracteres desconocidos y truncamientos. Las mismas clases y expresiones base existen en todas las particiones: se evalúa clasificación cerrada de estas consultas, no generalización a frases nuevas en náhuatl. Las plantillas siguen siendo parecidas; una buena puntuación no demuestra competencia lingüística.
- **Conversaciones:** solo train. Muchas clases tienen una sola formulación y no existe un test independiente suficiente. Se dejaron validation/test vacíos para no fabricar una evaluación engañosa.
- Los archivos `train.jsonl`, `validation.jsonl` y `test.jsonl` permiten inspeccionar la correspondencia texto/etiqueta de cada fila de las matrices.

## Fuentes, derechos y reproducción

Las fuentes públicas no equivalen a una licencia universal para redistribuir todas sus obras. `procedencia.json` describe lo que se pudo comprobar. Se acompaña la licencia MPL-2.0 publicada por Py-Elotl, sin afirmar que sustituya los derechos de cada texto compilado. Para un producto redistribuido, comprobar las condiciones de los materiales originales seleccionados. INALI y UV se emplean para expresiones funcionales breves; no se redistribuyen sus lecciones completas.

El ZIP no incluye el CSV completo de Axolotl ni los registros no seleccionados. El archivo original de consulta permanece en `fuentes/axolotl.csv` dentro del proyecto local y **no es un conjunto listo para entrenamiento**. `construir_dataset.py` requiere ese CSV y su hash documentado; después de descomprimir el ZIP, puede obtenerse del enlace de `procedencia.json`. No usar una versión distinta sin volver a revisar los números de registro y la selección.

El manual del ILV se localizó en su catálogo, pero la descarga devolvió HTTP 403. No se usó su contenido como fuente de frases ni se afirma haberlo revisado.
