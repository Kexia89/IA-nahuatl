# Informe de integridad de artefactos

Fecha de revisión: 24 de noviembre de 2026.

## Causa raíz

El entrenamiento se realizó en Windows y los JSON de cada modelo se guardaron con saltos de línea CRLF. `entrenar.py` calculó correctamente las firmas SHA-256 sobre esos bytes. El equipo tenía `core.autocrlf=true` y el repositorio no incluía `.gitattributes`; al añadir los archivos, Git normalizó los JSON a LF. Los H5 no cambiaron, pero los JSON publicados dejaron de coincidir con el manifiesto de `config.json`.

En el directorio de trabajo de Windows, Git reconstruía CRLF al hacer checkout y los hashes coincidían. En GitHub y Streamlit Community Cloud se descargaban los blobs LF del commit, por lo que `motor.py` rechazaba `tokenizador.json` antes de cargar el modelo. No hubo BOM, cambio de UTF-8, corrupción del H5 ni mezcla entre ejecuciones de entrenamiento.

## Recuperación

Los artefactos originales estaban disponibles en `entrenador_nahuatl/modelos/`. Se compararon byte a byte con `GitHub/models/`: todos los archivos originales coinciden con las firmas registradas. Esos bytes originales se volvieron a añadir al índice de Git sin conversión de texto. No se reentrenaron modelos y no se regeneraron las firmas de `config.json`.

## Prevención

`.gitattributes` define `models/** -text` y `dataset/** -text`. Git conserva esos árboles byte a byte en Windows, GitHub y Linux. Los formatos H5, NPZ e imágenes también se marcan como binarios. El código y la documentación ordinarios se mantienen como texto LF.

`verificar_integridad.py` comprueba los cuatro hashes de cada modelo, la correspondencia entre modo, etiquetas y respuestas, la exactitud publicada y, con `--inferir`, las dimensiones del H5 y cuatro expresiones documentadas. `motor.py` mantiene la validación estricta y ahora indica cuando una discrepancia corresponde exactamente a una conversión LF/CRLF; nunca acepta el archivo modificado.

## Firmas confirmadas

| Modo | Archivo | SHA-256 |
|---|---|---|
| aprendizaje | modelo.h5 | `0c4497e81121c6ebf0a0724f61156559a8a1c984748ff85af14a6ed5315fee5b` |
| aprendizaje | tokenizador.json | `9eb221fc5b34c882491c3d5ec48770327cac41f473ae9d72c1f464d8d34430fa` |
| aprendizaje | etiquetas.json | `a706116881cf4100bb5a4476ece0a9c78fe6423b7eade149c5a8db79e7d2c7cf` |
| aprendizaje | respuestas.json | `16243922f1ceb6d2388f1f6bf8c4f19c0615b566116c60e221344982b265b3b2` |
| conversaciones | modelo.h5 | `abb257af35cffced09fe47857b5fa2f963fce4dd544bf23f3f93683360ee2a69` |
| conversaciones | tokenizador.json | `c723433d240bd1cdaad60f28e2db2979a286acaa5619dd999076035e8177f9a3` |
| conversaciones | etiquetas.json | `79874dcb4cefa18d412ca689846192ea5ca3db5cf3979552adca5d7b2790e131` |
| conversaciones | respuestas.json | `99dc1242339244680aed0fa7bf5b7c3f8fe168188f8086fed60ae9554a31f805` |

## Métricas vinculadas

El modelo de aprendizaje tiene 345 salidas, entrada de 128 caracteres y `test_accuracy=0.815891444683075`, igual en `config.json` y `reporte_entrenamiento.json`. El modelo conversacional tiene 29 salidas y entrada de 64 caracteres; no tiene test independiente y ambos archivos registran `null`. No se modificaron las métricas publicadas.

## Pruebas ejecutadas

- 25 pruebas unitarias: correctas.
- Verificación de los ocho hashes en el directorio de trabajo: correcta.
- Verificación de los ocho blobs preparados en el índice de Git: correcta.
- Coincidencia byte a byte entre tokenizadores, etiquetas y respuestas de `models/` y `dataset/tensores/`: correcta.
- TensorFlow 2.21.0, NumPy 2.4.6 y Streamlit 1.51.0 instalados en un entorno virtual limpio: correctos.
- Inferencia de aprendizaje: `Piyali` → `hola` (0.996374) y `buenos días` → `Mah cualli tonalli / Tonalti` (0.980889).
- Inferencia conversacional: `Piyali` → `Piyali` (0.999989) y `Kenihkatzah tiitztok?` → `Kualli niitztok` (0.999952).
- Pruebas HTTP con modelos reales, entradas inválidas y fuera de vocabulario: correctas.
- Inicio de Streamlit, endpoint de salud y página principal: `ok`, HTTP 200.

## Nota sobre el manifiesto del dataset

Tres scripts del dataset habían sido modificados intencionalmente para retirar comentarios antes de esta revisión. Se actualizaron únicamente sus tres entradas en `dataset/SHA256SUMS.txt` para describir el código actual. Los corpus, tensores y artefactos lingüísticos no se modificaron y conservan sus firmas anteriores.
