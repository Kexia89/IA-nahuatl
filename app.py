from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from motor import Predictor


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
HERO = BASE_DIR / "assets" / "hero-nahuatl.png"

MODES = {
    "aprendizaje": {
        "label": "Aprender",
        "description": "Consulta expresiones del catálogo en español y náhuatl de la Huasteca.",
        "examples": [
            "¿Cómo se dice buenos días en náhuatl de la Huasteca?",
            "¿Qué significa Piyali en español?",
            "Traduce al náhuatl de la Huasteca: muchas gracias",
        ],
    },
    "conversaciones": {
        "label": "Conversar",
        "description": "Practica intercambios breves reconocidos por el catálogo conversacional.",
        "examples": ["Piyali", "Tlen motoka?", "Kenihkatzah tiitztok?"],
    },
}


st.set_page_config(
    page_title="Tlahtolli · Náhuatl",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --jade:#0e6258; --maize:#e7ad37; --ink:#173b37; --cream:#fffaf0; }
      .stApp { background: linear-gradient(180deg, #fffdf8 0%, #f3f8f5 100%); }
      [data-testid="stHeader"] { background: rgba(255,253,248,.82); }
      .hero-title { color:var(--ink); font-size:clamp(2.1rem,5vw,4rem); font-weight:800;
                    letter-spacing:-.04em; margin:.25rem 0 0; }
      .hero-subtitle { color:#3e625c; font-size:1.08rem; max-width:760px; margin:0 0 1.2rem; }
      .purpose { border-left:4px solid var(--maize); background:#fff7df; border-radius:0 12px 12px 0;
                 padding:.8rem 1rem; color:#533e13; margin:.7rem 0 1.2rem; }
      [data-testid="stChatMessage"] { border:1px solid #dfe9e5; border-radius:16px; background:#fff; }
      [data-testid="stMetric"] { border:1px solid #dbe7e2; border-radius:14px; padding:12px;
                                  background:rgba(255,255,255,.8); }
      .small-note { color:#5d736e; font-size:.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor(mode: str) -> Predictor:
    return Predictor(MODELS_DIR / mode)


def model_available(mode: str) -> bool:
    return (MODELS_DIR / mode / "modelo.h5").is_file()


def run_prediction(mode: str, text: str) -> dict[str, Any]:
    with st.spinner("Escuchando la frase…"):
        return load_predictor(mode).predict(text)


if HERO.is_file():
    st.image(str(HERO), use_container_width=True)

st.markdown('<p class="hero-title">Tlahtolli</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Un prototipo abierto para practicar náhuatl de la Huasteca '
    'y explorar cómo la tecnología puede apoyar lenguas indígenas y nativas de México.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="purpose"><strong>Propósito:</strong> apoyar el aprendizaje y la visibilidad de '
    'lenguas vivas. Este proyecto no sustituye a docentes, hablantes ni procesos comunitarios de '
    'documentación y validación.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Modo de práctica")
    mode = st.radio(
        "Elige una experiencia",
        options=list(MODES),
        format_func=lambda value: MODES[value]["label"],
        label_visibility="collapsed",
    )
    st.caption(MODES[mode]["description"])
    st.divider()
    st.subheader("Ideas para comenzar")
    selected_example = None
    for index, example in enumerate(MODES[mode]["examples"]):
        if st.button(example, key=f"example-{mode}-{index}", use_container_width=True):
            selected_example = example
    st.divider()
    st.caption("Modelo local · sin API externa · respuestas de un catálogo documentado")

history_key = f"messages_{mode}"
if history_key not in st.session_state:
    st.session_state[history_key] = []

header, action = st.columns([5, 1])
with header:
    st.subheader(f"{MODES[mode]['label']} náhuatl")
with action:
    if st.button("Limpiar", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun()

if not st.session_state[history_key]:
    st.info("Selecciona un ejemplo o escribe una frase para comenzar.")

for message in st.session_state[history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Escribe en español o náhuatl…", max_chars=2000)
prompt = selected_example or prompt

if prompt:
    st.session_state[history_key].append({"role": "user", "content": prompt})
    try:
        result = run_prediction(mode, prompt)
        score = result.get("confidence", 0.0)
        meta = f"Puntuación del clasificador: {score:.1%}"
        if result.get("accepted"):
            meta += f" · intención: {result.get('tag', '—')}"
        else:
            meta += " · respuesta rechazada por umbral o margen"
        st.session_state[history_key].append(
            {"role": "assistant", "content": result["answer"], "meta": meta}
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.session_state[history_key].append(
            {"role": "assistant", "content": f"No pude procesar la frase: {exc}", "meta": "Error local"}
        )
    st.rerun()

st.divider()
st.subheader("Qué puede hacer este prototipo")
col1, col2, col3 = st.columns(3)
col1.metric("Exactitud en test", "81.59%", help="Solo para el catálogo de aprendizaje: 516 ejemplos de test.")
col2.metric("Intenciones educativas", "345", help="Clases del modo de aprendizaje.")
col3.metric("Intenciones de diálogo", "29", help="Entrenadas con solo 42 ejemplos; sin test independiente.")

with st.expander("Alcance, privacidad y límites"):
    st.markdown(
        """
        - Es un **clasificador de intenciones con respuestas recuperadas**, no un modelo generativo ni un traductor libre.
        - La puntuación mostrada no está calibrada como probabilidad de corrección lingüística.
        - El modo conversacional no tiene un conjunto de prueba independiente y puede memorizar sus ejemplos.
        - Las fuentes y expresiones requieren revisión continua por hablantes y especialistas de la variante.
        - La inferencia se ejecuta en el servidor que aloja la app; no necesita enviar el texto a una API de IA.
        """
    )

with st.expander("Estado de los modelos"):
    for key, details in MODES.items():
        status = "disponible" if model_available(key) else "no encontrado"
        st.write(f"**{details['label']}:** {status}")

st.markdown(
    '<p class="small-note">Construido con TensorFlow y Streamlit. Revisa el README del repositorio '
    'para conocer fuentes, metodología y formas responsables de contribuir.</p>',
    unsafe_allow_html=True,
)
