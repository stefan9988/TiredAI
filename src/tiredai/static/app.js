import {
  BENCHMARKS_URL,
  chatIdFromUrl,
  chatUrl,
  createStatusQueue,
  errorMessage,
  escapeHtml,
  isBenchmarksUrl,
  nextSort,
  parseSSE,
  renderBenchmarks,
  renderGuardrailNote,
  renderMarkdown,
  renderToolCalls,
} from "./lib.mjs";

const messages = document.querySelector("#messages");
const empty = document.querySelector("#empty");
const form = document.querySelector("#composer");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const meta = document.querySelector("#meta");
const chats = document.querySelector("#chats");
const newChat = document.querySelector("#new-chat");
const details = document.querySelector("#details");
const detailsTitle = document.querySelector("#details-title");
const detailsBody = document.querySelector("#details-body");
const detailsClose = document.querySelector("#details-close");
const benchmarksLink = document.querySelector("#benchmarks");
const benchmarksView = document.querySelector("#benchmarks-view");
const bottom = document.querySelector("#bottom");
const guardrail = document.querySelector("#guardrail");

// The open chat, also kept in the URL. null for a new chat until its first reply starts.
let conversationId = null;
let chatList = [];
// While a reply streams or a chat loads, sending and switching chats wait.
let busy = false;
// The tool calls behind each answer (with the data the model got), by the button that shows them.
const searchesOf = new WeakMap();
// The button whose searches the side panel shows; null while the panel is closed.
let detailsFor = null;
// The benchmark results are shown instead of a chat.
let showingBenchmarks = false;
// Counts requests for the results, so only the latest one is shown.
let benchmarksRequest = 0;
// The results shown, and how each table is sorted ({agent: {key, direction}, ...}), kept while the page is open.
let benchmarks = null;
const benchmarkSort = {};
// The guardrail switch is on unless this browser turned it off; every message sends its state.
const GUARDRAIL_KEY = "tiredai.guardrail";

const SEARCH_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2.2"/>' +
  '<path d="m15.5 15.5 5 5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>';

try {
  guardrail.checked = localStorage.getItem(GUARDRAIL_KEY) !== "off";
} catch {
  // No storage (private window, blocked site data): the switch starts on.
}

guardrail.addEventListener("change", () => {
  try {
    localStorage.setItem(GUARDRAIL_KEY, guardrail.checked ? "on" : "off");
  } catch {
    // The choice still holds until the page is reloaded.
  }
});

function setBusy(value) {
  busy = value;
  send.disabled = value;
  newChat.disabled = value;
  benchmarksLink.classList.toggle("busy", value);
  chats.classList.toggle("busy", value);
}

fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    const catalog =
      h.vector_store.status === "ok" ? `${h.vector_store.points.toLocaleString()} tires` : "catalog unavailable";
    meta.textContent = `${h.model} · ${catalog}`;
    meta.classList.toggle("warn", h.vector_store.status !== "ok");
    const toggle = guardrail.closest(".toggle");
    if (h.guardrail.available) {
      toggle.title =
        `${h.guardrail.model} checks each message before the chat model sees it, and answers the off-topic ` +
        `and adversarial ones itself (block score ${h.guardrail.threshold} or more). Off: messages go straight to the chat model.`;
    } else {
      guardrail.disabled = true;
      toggle.title = "The guardrail is unavailable: it needs OPENROUTER_API_KEY.";
    }
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

function addSavedAnswer(text, toolCalls, decision) {
  const el = document.createElement("article");
  el.className = "message assistant";
  el.innerHTML = `<div class="answer">${renderMarkdown(text)}</div>`;
  if (decision) el.insertAdjacentHTML("beforeend", renderGuardrailNote(decision));
  messages.append(el);
  for (const call of toolCalls) addSearch(el, call);
}

// Adds a tool call to the small button under an answer that shows them in the side panel; the
// first one creates the button. A streaming reply's open panel updates as its searches finish.
function addSearch(article, call) {
  let button = article.querySelector(":scope > button.searches");
  if (!button) {
    button = document.createElement("button");
    button.type = "button";
    button.className = "searches";
    button.setAttribute("aria-controls", "details");
    button.setAttribute("aria-expanded", "false");
    article.append(button);
    searchesOf.set(button, []);
  }
  const calls = searchesOf.get(button);
  calls.push(call);
  button.innerHTML = `${SEARCH_ICON}<span>${calls.length}</span>`;
  button.title = `Show ${calls.length === 1 ? "the search" : `the ${calls.length} searches`} behind this answer`;
  button.setAttribute("aria-label", button.title);
  if (detailsFor === button) renderDetails();
}

function renderDetails() {
  const calls = searchesOf.get(detailsFor);
  detailsTitle.textContent =
    calls.length === 1 ? "The search behind this answer" : `The ${calls.length} searches behind this answer`;
  detailsBody.innerHTML = renderToolCalls(calls);
}

function openDetails(button) {
  detailsFor?.setAttribute("aria-expanded", "false");
  detailsFor = button;
  button.setAttribute("aria-expanded", "true");
  renderDetails();
  details.hidden = false;
  detailsBody.scrollTop = 0;
  detailsClose.focus();
}

function closeDetails() {
  if (!detailsFor) return;
  const button = detailsFor;
  const focusWasInside = details.contains(document.activeElement);
  button.setAttribute("aria-expanded", "false");
  detailsFor = null;
  details.hidden = true;
  if (focusWasInside) button.focus();
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
    toolCall(data) {
      addSearch(el, data);
      scrollDown();
    },
    guardrail(decision) {
      el.insertAdjacentHTML("beforeend", renderGuardrailNote(decision));
      scrollDown();
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

function showBenchmarksView(show) {
  showingBenchmarks = show;
  messages.hidden = bottom.hidden = show;
  benchmarksView.hidden = !show;
  if (show) benchmarksLink.setAttribute("aria-current", "page");
  else benchmarksLink.removeAttribute("aria-current");
}

// Shows the latest benchmark results per model in place of the chat; opening it again reloads them.
async function openBenchmarks() {
  closeDetails();
  conversationId = null;
  renderChats();
  showBenchmarksView(true);
  const request = ++benchmarksRequest;
  const current = () => request === benchmarksRequest && showingBenchmarks;
  benchmarksView.innerHTML = '<p class="loading">Loading benchmark results…</p>';
  try {
    const response = await fetch("/benchmarks");
    if (!response.ok) {
      throw new Error(errorMessage(await response.json().catch(() => null), response.status));
    }
    const data = await response.json();
    if (current()) {
      benchmarks = data;
      benchmarksView.innerHTML = renderBenchmarks(benchmarks, benchmarkSort);
    }
  } catch (err) {
    if (current()) {
      const reason = escapeHtml(err.message || "something went wrong.");
      benchmarksView.innerHTML = `<p class="error">Could not load the benchmark results: ${reason}</p>`;
    }
  }
}

// Shows a saved chat, or the empty state for a new one (id null).
async function openChat(id) {
  closeDetails();
  showBenchmarksView(false);
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
      else addSavedAnswer(message.content, message.tool_calls ?? [], message.guardrail);
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
      body: JSON.stringify({
        message: text,
        conversation_id: conversationId ?? undefined,
        guardrail: guardrail.checked && !guardrail.disabled,
      }),
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
        else if (event === "tool_call") reply.toolCall(data);
        else if (event === "guardrail") reply.guardrail(data);
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

messages.addEventListener("click", (e) => {
  const button = e.target.closest("button.searches");
  if (!button) return;
  if (button === detailsFor) closeDetails();
  else openDetails(button);
});

detailsClose.addEventListener("click", closeDetails);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && detailsFor) closeDetails();
});

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
  if (conversationId || showingBenchmarks) history.pushState(null, "", chatUrl(null));
  openChat(null);
});

benchmarksLink.addEventListener("click", (e) => {
  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  if (busy) return;
  if (!showingBenchmarks) history.pushState(null, "", BENCHMARKS_URL);
  openBenchmarks();
});

// A column header sorts its table; the header keeps the focus, so the keyboard can sort again.
benchmarksView.addEventListener("click", (e) => {
  const button = e.target.closest("button[data-sort-key]");
  if (!button || !benchmarks) return;
  const { sortKind: kind, sortKey: key, sortFirst: first } = button.dataset;
  benchmarkSort[kind] = nextSort(benchmarkSort[kind], key, first);
  benchmarksView.innerHTML = renderBenchmarks(benchmarks, benchmarkSort);
  benchmarksView.querySelector(`button[data-sort-kind="${kind}"][data-sort-key="${CSS.escape(key)}"]`)?.focus();
});

// The page URL says what to show: the benchmarks, a saved chat, or a new chat.
function openFromUrl() {
  if (isBenchmarksUrl(location.search)) openBenchmarks();
  else openChat(chatIdFromUrl(location.search));
}

window.addEventListener("popstate", () => {
  // Back/forward can't interrupt a streaming reply, so stay on the open chat.
  if (busy) history.pushState(null, "", chatUrl(conversationId));
  else openFromUrl();
});

loadChats();
openFromUrl();
