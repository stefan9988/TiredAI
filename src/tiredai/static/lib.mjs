// Pure helpers for the chat page. No DOM access, so they can be tested with node (tests/ui).

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

// Inline formatting on already-escaped text: `code`, **bold**, *italic*.
function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>");
}

function tableRow(line, cell) {
  const cells = line.trim().replace(/^\||\|$/g, "").split("|");
  return `<tr>${cells.map((c) => `<${cell}>${inline(c.trim())}</${cell}>`).join("")}</tr>`;
}

// The small subset of Markdown the assistant uses: paragraphs, headings, lists, tables, rules
// (`---`, which also separates the text written before and after a tool call), inline styles.
// All text is HTML-escaped before formatting, so model output can never inject markup.
export function renderMarkdown(text) {
  const lines = text.replace(/\r\n/g, "\n").trim().split("\n");
  const html = [];
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (list) html.push(`</${list}>`);
    list = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const item = line.match(/^(\s*)(?:([-*•])|\d+[.)])\s+(.*)$/);
    const heading = line.match(/^#{1,6}\s+(.*)$/);

    if (line.trim().startsWith("|") && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1] ?? "")) {
      flushParagraph();
      closeList();
      const rows = [`<thead>${tableRow(line, "th")}</thead><tbody>`];
      for (i += 2; i < lines.length && lines[i].trim().startsWith("|"); i++) rows.push(tableRow(lines[i], "td"));
      i--;
      html.push(`<table>${rows.join("")}</tbody></table>`);
    } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flushParagraph();
      closeList();
      html.push("<hr>");
    } else if (item) {
      flushParagraph();
      const type = item[2] ? "ul" : "ol";
      if (list !== type) {
        closeList();
        html.push(`<${type}>`);
        list = type;
      }
      const nested = item[1].length >= 2 ? ' class="sub"' : "";
      html.push(`<li${nested}>${inline(item[3])}</li>`);
    } else if (heading) {
      flushParagraph();
      closeList();
      html.push(`<h3>${inline(heading[1])}</h3>`);
    } else if (!line.trim()) {
      flushParagraph();
      closeList();
    } else {
      closeList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  closeList();
  return html.join("");
}

// Splits a Server-Sent Events buffer into complete events; `rest` is an unfinished tail to keep.
export function parseSSE(buffer) {
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop();
  const events = [];
  for (const block of blocks) {
    let event = "message";
    const data = [];
    for (const line of block.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue; // comments are keep-alives
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join("\n")) });
  }
  return { events, rest };
}

// Readable text for an API error body ({detail: "..."} or FastAPI's validation list).
export function errorMessage(body, status) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail.map((d) => d.msg).join("; ");
  return `Request failed (HTTP ${status})`;
}

// Shows agent status messages one at a time on a single line. Each stays visible for at least
// `minMs` so quick steps don't flash by; after `finish()` the line hides once the last message
// has had its time. `show(status)` and `hide()` do the drawing.
export function createStatusQueue({ show, hide, minMs = 700 }) {
  const queue = [];
  let timer = null;
  let finished = false;

  const display = (status) => {
    show(status);
    timer = setTimeout(advance, minMs);
  };
  const advance = () => {
    timer = null;
    if (queue.length) display(queue.shift());
    else if (finished) hide();
  };

  return {
    push(status) {
      finished = false;
      if (timer === null && !queue.length) display(status);
      else queue.push(status);
    },
    // The agent is done working (its answer is streaming, or the turn ended). A queued "thinking"
    // step is no longer true, so it is dropped.
    finish() {
      finished = true;
      for (let i = queue.length - 1; i >= 0; i--) if (queue[i].stage === "thinking") queue.splice(i, 1);
      if (timer === null && !queue.length) hide();
    },
  };
}

// The open chat lives in the page URL (/?c=<id>), so reloading or sharing the link reopens it.
export function chatUrl(conversationId) {
  return conversationId ? `/?c=${encodeURIComponent(conversationId)}` : "/";
}

export function chatIdFromUrl(search) {
  return new URLSearchParams(search).get("c") || null;
}

// A value from a tool call as text: lists are joined, and a range shows the ends that are set
// ({min: null, max: 60} -> "max 60"). Everything else as is, so values match the data exactly.
export function formatValue(value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.map(formatValue).join(", ");
  if (typeof value === "object") {
    return Object.entries(value)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `${k} ${formatValue(v)}`)
      .join(", ");
  }
  return String(value);
}

const LEADING_COLUMNS = ["name", "price"];

// Columns of the product table: name and price first, then every other field in the order the products list them.
export function productColumns(products) {
  const keys = [...new Set(products.flatMap((p) => Object.keys(p)))];
  const leading = LEADING_COLUMNS.filter((k) => keys.includes(k));
  return [...leading, ...keys.filter((k) => !leading.includes(k))];
}

function fields(values, none) {
  const entries = Object.entries(values ?? {});
  if (!entries.length) return `<p class="none">${none}</p>`;
  return `<dl>${entries.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(formatValue(v))}</dd>`).join("")}</dl>`;
}

function productTable(products) {
  const columns = productColumns(products);
  const head = columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const rows = products.map((p) => `<tr>${columns.map((c) => `<td>${escapeHtml(formatValue(p[c]))}</td>`).join("")}</tr>`);
  return `<div class="table-scroll"><table class="products"><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function searchResult(result) {
  const total = Number(result.total_matching ?? 0).toLocaleString("en-US");
  const html = [
    "<h4>Filters applied</h4>",
    fields(result.filters, "None: the whole catalog was searched"),
    "<h4>Products the model got</h4>",
    `<p>${result.products.length} of ${total} matching · order: ${escapeHtml(formatValue(result.order))}</p>`,
  ];
  if (result.note) html.push(`<p class="none">${escapeHtml(result.note)}</p>`);
  if (result.products.length) html.push(productTable(result.products));
  return html.join("");
}

function toolCall(call, title) {
  const html = [`<h3>${escapeHtml(title)}</h3>`, "<h4>The model asked for</h4>"];
  // Arguments that could not be parsed are kept as the raw text the model sent.
  html.push(typeof call.args === "string" ? `<pre>${escapeHtml(call.args)}</pre>` : fields(call.args, "No arguments"));
  if (call.error) {
    html.push("<h4>The model got an error</h4>", `<p class="error">${escapeHtml(call.error)}</p>`);
  } else if (Array.isArray(call.result?.products)) {
    html.push(searchResult(call.result));
  } else {
    const output = typeof call.result === "string" ? call.result : JSON.stringify(call.result, null, 2);
    html.push("<h4>The model got</h4>", `<pre>${escapeHtml(output ?? "")}</pre>`);
  }
  return `<section class="tool-call">${html.join("")}</section>`;
}

// The side panel's view of the tool calls behind an answer: for each, what the model asked for and
// the error or data it got back, so the answer can be checked against it. All text is escaped.
export function renderToolCalls(calls) {
  return calls
    .map((call, i) => {
      const label = call.name === "search_tires" ? "Search" : call.name;
      return toolCall(call, calls.length > 1 ? `${label} ${i + 1} of ${calls.length}` : label);
    })
    .join("");
}
