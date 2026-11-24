<p align="center">
  <img src="assets/hero-nahuatl.png" alt="Libro abierto del que nacen palabras, rodeado por un paisaje verde de la Huasteca" width="100%">
</p>

<h1 align="center">Tlahtolli · Chatbot educativo de náhuatl</h1>

<p align="center">
  Un prototipo abierto para conversar, aprender y explorar cómo la tecnología puede apoyar la preservación, transmisión y visibilidad de lenguas indígenas y nativas de México.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow 2.21" src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Licencia MIT" src="https://img.shields.io/badge/código-MIT-0E6258">
</p>

> [!IMPORTANT]
> Este proyecto es un clasificador de intenciones con respuestas de un catálogo documentado. No es un traductor libre, no genera náhuatl nuevo y no sustituye a hablantes, docentes ni procesos comunitarios de validación.

## La idea

**Tlahtolli** reúne dos experiencias en una interfaz Streamlit:

- **Aprender:** consulta equivalencias y expresiones en español y náhuatl de la Huasteca.
- **Conversar:** practica intercambios breves que el catálogo reconoce.

La inferencia utiliza modelos TensorFlow locales y no requiere una API externa. El repositorio incluye modelos H5, tokenizadores, etiquetas, respuestas, reportes, código de entrenamiento, pruebas y el conjunto preparado para reproducir el experimento.

## Resultados actuales

| Modo | Clases | Train | Validación | Test | Resultado |
|---|---:|---:|---:|---:|---|
| Aprendizaje | 345 | 1,380 | 516 | 516 | **81.59% accuracy en test** |
| Conversaciones | 29 | 42 | 0 | 0 | 100% en train; **sin evaluación independiente** |

El primer resultado es prometedor dentro del catálogo. No mide traducción general. El segundo indica memorización de un conjunto muy pequeño y todavía no permite afirmar que el bot generalice. Lee el [análisis completo](docs/EVALUACION.md).

## Probarlo en local

Requiere Python 3.12 de 64 bits.

```bash
git clone URL_DE_TU_REPOSITORIO
cd NOMBRE_DEL_REPOSITORIO
python -m venv .venv
```

En Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

En macOS o Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`. La primera inferencia tarda un poco más porque carga TensorFlow y el modelo elegido.

## Publicar en GitHub y Streamlit Community Cloud

1. Crea un repositorio vacío en GitHub y reemplaza `URL_DE_TU_REPOSITORIO` en este README.
2. Desde esta carpeta ejecuta:

   ```bash
   git init
   git add .
   git commit -m "Publica Tlahtolli"
   git branch -M main
   git remote add origin URL_DE_TU_REPOSITORIO.git
   git push -u origin main
   ```

3. En [share.streamlit.io](https://share.streamlit.io), selecciona el repositorio, la rama `main` y `app.py` como archivo principal.
4. La app no necesita secretos. Streamlit instalará `requirements.txt` y utilizará `runtime.txt`.

Los dos modelos ocupan menos de 3 MB en conjunto, por lo que caben en GitHub sin Git LFS.

## Reentrenar

El conjunto preparado vive en `dataset/` y los resultados se escriben en `models/`.

```bash
python entrenar.py --solo-verificar
python entrenar.py --modo ambos
```

También puedes entrenar un modo o cambiar el máximo de épocas:

```bash
python entrenar.py --modo aprendizaje --epochs 120
python entrenar.py --modo conversaciones --epochs 160
```

La arquitectura procesa caracteres: `Embedding(32) → Conv1D(96) → Conv1D(96) → GlobalMaxPooling → Dense(192) → Dropout → Softmax`. El guardado incluye arquitectura y pesos en H5, además de manifiestos SHA-256 que evitan mezclar archivos de versiones distintas.

## Estructura

```text
.
├── app.py                    # interfaz Streamlit
├── motor.py                  # tokenización, carga e inferencia
├── entrenar.py               # entrenamiento y evaluación
├── interfaz.py               # servidor web local alternativo
├── models/                   # modelos H5 y metadatos
├── dataset/                  # datos, tensores, procedencia y scripts
├── tests/                    # pruebas de integridad e inferencia
├── docs/                     # evaluación, fuentes y ética
├── web/                      # interfaz local alternativa
└── assets/                   # imagen del proyecto
```

## Pruebas

```bash
python -m unittest discover -s tests -v
```

Las pruebas rápidas validan normalización, codificación, particiones, etiquetas, umbrales y manejo de entradas fuera de vocabulario. `entrenar.py` también vuelve a cargar cada H5 y compara sus predicciones antes de dar el entrenamiento por terminado.

## Hoja de ruta

- Co-diseñar el proyecto con hablantes y organizaciones de la variante representada.
- Ampliar el modo conversacional con datos consentidos, revisados y diversos.
- Crear un test realmente independiente y reportar macro-F1, cobertura y falsas aceptaciones.
- Mostrar fuente, variante, ortografía y estado de revisión con cada respuesta.
- Añadir audio revisado por hablantes y ejercicios pedagógicos progresivos.
- Comparar el clasificador neuronal con recuperación explicable y búsqueda difusa.

## Fuentes, ética y licencia

Las fuentes registradas incluyen materiales del INALI, la Universidad Veracruzana y el corpus Axolotl distribuido en Py-Elotl. Hay diferencias de variante, grafía y licenciamiento que deben conservarse visibles. Consulta [Fuentes, licencias y uso responsable](docs/FUENTES_Y_ETICA.md) y los manifiestos de `dataset/` antes de reutilizar los datos.

El software original se publica bajo licencia [MIT](LICENSE). Los datos y materiales de terceros mantienen sus propios términos; la licencia del código no los reemplaza.

---

<p align="center"><strong>Una lengua vive en sus hablantes.</strong><br>La tecnología debe acompañar sus decisiones, su enseñanza y su futuro.</p>
