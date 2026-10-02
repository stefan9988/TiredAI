import {
  chatIdFromUrl,
  chatUrl,
  createStatusQueue,
  errorMessage,
  escapeHtml,
  parseSSE,
  renderMarkdown,
} from "./lib.mjs";

const messages = document.querySelector("#messages");
const empty = document.querySelector("#empty");
const form = document.querySelector("#composer");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const meta = document.querySelector("#meta");
const chats = document.querySelector("#chats");
const newChat = document.querySelector("#new-chat");

// The open chat, also kept in the URL. null for a new chat until its first reply starts.
let conversationId = null;
let chatList = [];
// While a reply streams or a chat loads, sending and switching chats wait.
let busy = false;

function setBusy(value) {
  busy = value;
  send.disabled = value;
  newChat.disabled = value;
  chats.classList.toggle("busy", value);
}

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

function addSavedAnswer(text) {
  const el = document.createElement("article");
  el.className = "message assistant";
  el.innerHTML = `<div class="answer">${renderMarkdown(text)}</div>`;
  messages.append(el);
}

function addError(message) {
  const el = document.createElement("article");
  el.className = "message assistant";
  el.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
  messages.append(el);
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

function renderChats() {
  if (!chatList.length) {
    chats.innerHTML = '<p class="no-chats">No chats yet</p>';
    return;
  }
  chats.replaceChildren(
    ...chatList.map((chat) => {
      const link = document.createElement("a");
      link.className = "chat";
      link.href = chatUrl(chat.id);
      link.dataset.id = chat.id;
      link.textContent = link.title = chat.title;
      if (chat.id === conversationId) link.setAttribute("aria-current", "page");
      return link;
    }),
  );
}

async function loadChats() {
  try {
    const response = await fetch("/conversations");
    if (response.ok) chatList = await response.json();
  } catch {
    // Keep the list as it was; chatting still works.
  }
  renderChats();
}

// Shows a saved chat, or the empty state for a new one (id null).
async function openChat(id) {
  conversationId = id;
  renderChats();
  if (!id) {
    messages.replaceChildren(empty);
    input.focus();
    return;
  }
  messages.replaceChildren();
  setBusy(true);
  try {
    const response = await fetch(`/conversations/${encodeURIComponent(id)}/messages`);
    if (!response.ok) {
      throw new Error(errorMessage(await response.json().catch(() => null), response.status));
    }
    for (const message of await response.json()) {
      if (message.role === "user") addUserMessage(message.content);
      else addSavedAnswer(message.content);
    }
    scrollDown(true);
  } catch (err) {
    // A stale link: show why, and let the next message start a new chat.
    conversationId = null;
    history.replaceState(null, "", chatUrl(null));
    renderChats();
    addError(`Could not open this chat: ${err.message || "something went wrong."}`);
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function sendMessage(text) {
  empty.remove();
  addUserMessage(text);
  const reply = addAssistantMessage();
  setBusy(true);

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
        if (event === "start") {
          if (conversationId !== data.conversation_id) {
            conversationId = data.conversation_id;
            history.replaceState(null, "", chatUrl(conversationId));
          }
          loadChats(); // the chat is listed (or moved to the top) once its reply starts
        }
        else if (event === "status") reply.status(data);
        else if (event === "token") reply.token(data.text);
        else if (event === "error") reply.fail(data.message);
      }
    }
  } catch (err) {
    reply.fail(err.message || "Something went wrong.");
  } finally {
    reply.finish();
    setBusy(false);
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

chats.addEventListener("click", (e) => {
  const link = e.target.closest("a.chat");
  // Modified clicks open the chat in a new tab or window as usual.
  if (!link || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  if (busy || link.dataset.id === conversationId) return;
  history.pushState(null, "", chatUrl(link.dataset.id));
  openChat(link.dataset.id);
});

newChat.addEventListener("click", () => {
  if (busy) return;
  if (conversationId) history.pushState(null, "", chatUrl(null));
  openChat(null);
});

window.addEventListener("popstate", () => {
  // Back/forward can't interrupt a streaming reply, so stay on the open chat.
  if (busy) history.pushState(null, "", chatUrl(conversationId));
  else openChat(chatIdFromUrl(location.search));
});

loadChats();
openChat(chatIdFromUrl(location.search));
