"use strict";

const ui = Object.fromEntries([
  "mode", "mode-description", "model-status", "status-dot", "refresh", "clear",
  "notice", "chat", "welcome", "welcome-description", "examples", "chat-form",
  "message", "send", "counter",
].map((id) => [id, document.getElementById(id)]));

const descriptions = {
  aprendizaje: "Pregunta en español por palabras y expresiones en náhuatl.",
  conversaciones: "Escribe una expresión en náhuatl para practicar una respuesta breve.",
};
const examples = {
  aprendizaje: ["¿Cómo se dice «buenos días» en náhuatl de la Huasteca?", "¿Qué significa «Piyali» en español?"],
  conversaciones: ["Piyali", "Tlen motoka?", "Kenihkatzah tiitztok?"],
};
let availableModels = new Map();
let busy = false;

function setNotice(message, isError = false) {
  ui.notice.textContent = message;
  ui.notice.hidden = !message;
  ui.notice.classList.toggle("error", isError);
}

function renderModelStatus() {
  const model = availableModels.get(ui.mode.value);
  const ready = Boolean(model?.available);
  ui["model-status"].textContent = ready ? "Modelo disponible" : "Modelo pendiente de entrenamiento";
  ui["status-dot"].classList.toggle("ready", ready);
  ui["status-dot"].classList.remove("error");
  if (ready) {
    setNotice("");
  } else {
    setNotice(`Primero entrena el modelo de ${ui.mode.value}. En la terminal, dentro de la carpeta del proyecto, ejecuta:\npython entrenar.py --modo ${ui.mode.value}\nDespués pulsa «Actualizar estado».`);
  }
  ui.send.disabled = busy || !ready;
}

async function refreshStatus() {
  ui.refresh.disabled = true;
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("No se pudo consultar el estado.");
    const data = await response.json();
    availableModels = new Map(data.models.map((model) => [model.mode, model]));
    renderModelStatus();
  } catch (error) {
    ui["model-status"].textContent = "Sin conexión con el servidor";
    ui["status-dot"].classList.remove("ready");
    ui["status-dot"].classList.add("error");
    ui.send.disabled = true;
    setNotice("No se pudo conectar. Comprueba que python interfaz.py sigue en ejecución y actualiza el estado.", true);
  } finally {
    ui.refresh.disabled = false;
  }
}

function configureMode() {
  ui["mode-description"].textContent = descriptions[ui.mode.value];
  ui["welcome-description"].textContent = ui.mode.value === "aprendizaje"
    ? "Escribe en español para consultar una palabra o expresión. El modelo buscará una intención entre las que aprendió."
    : "Escribe una expresión en náhuatl de tu conjunto de conversaciones. Cada mensaje se interpreta por separado.";
  ui.message.placeholder = ui.mode.value === "aprendizaje" ? "Escribe tu pregunta…" : "Escribe una expresión en náhuatl…";
  ui.examples.replaceChildren();
  for (const example of examples[ui.mode.value]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example-button";
    button.textContent = example;
    button.addEventListener("click", () => {
      ui.message.value = example;
      updateCounter();
      ui.message.focus();
    });
    ui.examples.append(button);
  }
}

function clearChat() {
  ui.chat.querySelectorAll(".message").forEach((message) => message.remove());
  ui.welcome.hidden = false;
}

function updateCounter() {
  ui.counter.textContent = `${ui.message.value.length} / 2000`;
}

function appendMessage(text, kind, result = null) {
  ui.welcome.hidden = true;
  const article = document.createElement("article");
  article.className = `message ${kind}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = kind === "user" ? "Tú" : kind === "error" ? "Aviso" : "Respuesta del modelo";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.append(label, bubble);

  if (result) {
    const details = document.createElement("details");
    details.className = "message-details";
    const summary = document.createElement("summary");
    const confidence = Number(result.confidence);
    const score = Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)} %` : "sin puntuación";
    summary.textContent = `Coincidencia: ${score} · ${result.accepted ? "intención aceptada" : "sin coincidencia suficiente"}`;
    const tag = document.createElement("p");
    tag.textContent = `Etiqueta: ${result.tag || "sin etiqueta"}. Esta puntuación no mide certeza lingüística.`;
    details.append(summary, tag);
    if (result.source) {
      const source = document.createElement("p");
      const origin = result.source === "modelo_h5" ? "modelo entrenado" : typeof result.source === "string" ? result.source : JSON.stringify(result.source);
      source.textContent = `Origen de la respuesta: ${origin}`;
      details.append(source);
    }
    article.append(details);
  }
  ui.chat.append(article);
  ui.chat.scrollTop = ui.chat.scrollHeight;
  return article;
}

async function sendMessage(event) {
  event.preventDefault();
  const message = ui.message.value.trim();
  if (busy || !message || !availableModels.get(ui.mode.value)?.available) return;
  const mode = ui.mode.value;
  busy = true;
  ui.send.disabled = true;
  ui.mode.disabled = true;
  ui.clear.disabled = true;
  ui.send.textContent = "Buscando…";
  appendMessage(message, "user");
  ui.message.value = "";
  updateCounter();
  const waiting = appendMessage("Buscando una respuesta… La primera consulta puede tardar mientras se carga el modelo.", "assistant");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode, message}),
    });
    const result = await response.json();
    waiting.remove();
    if (!response.ok) {
      appendMessage(result.error || "No se pudo procesar el mensaje.", "error");
      if (result.code === "model_missing") await refreshStatus();
    } else {
      appendMessage(result.answer || "El modelo no devolvió una respuesta.", "assistant", result);
    }
  } catch (error) {
    waiting.remove();
    appendMessage("No se pudo conectar con el servidor. Comprueba que sigue abierto y vuelve a intentar.", "error");
  } finally {
    busy = false;
    ui.mode.disabled = false;
    ui.clear.disabled = false;
    ui.send.textContent = "Enviar ↑";
    ui.send.disabled = !availableModels.get(ui.mode.value)?.available;
    ui.message.focus();
  }
}

ui["chat-form"].addEventListener("submit", sendMessage);
ui.message.addEventListener("input", updateCounter);
ui.message.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    ui["chat-form"].requestSubmit();
  }
});
ui.mode.addEventListener("change", () => { clearChat(); configureMode(); renderModelStatus(); });
ui.clear.addEventListener("click", () => { clearChat(); ui.message.focus(); });
ui.refresh.addEventListener("click", refreshStatus);
configureMode();
refreshStatus();
