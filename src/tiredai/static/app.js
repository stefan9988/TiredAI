import { createStatusQueue, errorMessage, escapeHtml, parseSSE, renderMarkdown } from "./lib.mjs";

const messages = document.querySelector("#messages");
const empty = document.querySelector("#empty");
const form = document.querySelector("#composer");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const meta = document.querySelector("#meta");

// Kept in memory only: reloading the page starts a new chat.
let conversationId = null;

fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    const catalog =
      h.vector_store.status === "ok" ? `${h.vector_store.points.toLocaleString()} tires` : "catalog unavailable";
    meta.textContent = `${h.model} · ${catalog}`;
    meta.classList.toggle("warn", h.vector_store.status !== "ok");
  })
  .catch(() => {
    meta.textContent = "API unreachable";
    meta.classList.add("warn");
  });

function nearBottom() {
  return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 120;
}

function scrollDown(force = false) {
  if (force || nearBottom()) messages.scrollTop = messages.scrollHeight;
}

function addUserMessage(text) {
  const el = document.createElement("article");
  el.className = "message user";
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messages.append(el);
  scrollDown(true);
}

// An assistant reply: one status line that changes as the agent works, then the streamed answer.
function addAssistantMessage() {
  const el = document.createElement("article");
  el.className = "message assistant";
  el.innerHTML = `<div class="status" role="status" hidden></div><div class="answer"></div>`;
  messages.append(el);
  const line = el.querySelector(".status");
  const answer = el.querySelector(".answer");
  let text = "";

  const statuses = createStatusQueue({
    show({ stage, text: label }) {
      line.hidden = false;
      line.dataset.stage = stage;
      line.textContent = label;
      line.classList.remove("enter");
      void line.offsetWidth; // restart the fade-in for the new text
      line.classList.add("enter");
      scrollDown();
    },
    hide() {
      line.hidden = true;
    },
  });

  return {
    status(data) {
      statuses.push(data);
    },
    token(chunk) {
      statuses.finish(); // the answer is being written
      text += chunk;
      answer.innerHTML = renderMarkdown(text);
      scrollDown();
    },
    fail(message) {
      statuses.finish();
      const error = document.createElement("p");
      error.className = "error";
      error.textContent = message;
      answer.append(error);
      scrollDown();
    },
    finish() {
      statuses.finish();
    },
  };
}

async function sendMessage(text) {
  empty?.remove();
  addUserMessage(text);
  const reply = addAssistantMessage();
  send.disabled = true;

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, conversation_id: conversationId ?? undefined }),
    });
    if (!response.ok) {
      throw new Error(errorMessage(await response.json().catch(() => null), response.status));
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSSE(buffer);
      buffer = parsed.rest;
      for (const { event, data } of parsed.events) {
        if (event === "start") conversationId = data.conversation_id;
        else if (event === "status") reply.status(data);
        else if (event === "token") reply.token(data.text);
        else if (event === "error") reply.fail(data.message);
      }
    }
  } catch (err) {
    reply.fail(err.message || "Something went wrong.");
  } finally {
    reply.finish();
    send.disabled = false;
    input.focus();
  }
}

function resize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || send.disabled) return;
  input.value = "";
  resize();
  sendMessage(text);
});

input.addEventListener("input", resize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".example").forEach((button) =>
  button.addEventListener("click", () => {
    input.value = button.textContent;
    form.requestSubmit();
  }),
);

input.focus();
