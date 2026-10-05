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

// The benchmark results replace the chat at /?view=benchmarks.
export const BENCHMARKS_URL = "/?view=benchmarks";

export function isBenchmarksUrl(search) {
  return new URLSearchParams(search).get("view") === "benchmarks";
}

function percent(value) {
  return value === null || value === undefined ? "–" : `${(value * 100).toFixed(1)}%`;
}

function detail(name, value) {
  if (value === null || value === undefined) return "–";
  if (name.includes("tokens")) return value >= 1000 ? `${Math.round(value / 1000).toLocaleString("en-US")}k` : String(Math.round(value));
  return value < 1 ? value.toFixed(2) : value.toFixed(1);
}

// "2026-10-05T10:00:01Z" -> "2026-10-05 10:00 UTC"
function when(iso) {
  return iso ? `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC` : "";
}

// The highest value of each score over the rows, where some row is lower: every row it ties wins,
// but a score all rows share marks nothing.
export function bestScores(rows, names) {
  const best = {};
  for (const name of names) {
    const values = rows.map((r) => r.scores[name]).filter((v) => v !== null && v !== undefined);
    if (values.length && Math.max(...values) > Math.min(...values)) best[name] = Math.max(...values);
  }
  return best;
}

const RANKING_LABELS = { hybrid: "Hybrid (app)", dense: "Dense", sparse: "BM25" };
const RANKING_ORDER = ["hybrid", "dense", "sparse"];

// A row's value in a column: the model name, the ranking's place (the app's first), or a score or detail.
function sortValue(row, key) {
  if (key === "model") return row.model.toLowerCase();
  if (key === "ranking") {
    const place = RANKING_ORDER.indexOf(row.ranking);
    return place < 0 ? RANKING_ORDER.length : place;
  }
  return row.scores[key] ?? row.details[key] ?? null;
}

// The rows ordered by a column, sort = {key, direction: "asc" | "desc"}. Rows without a value go last
// either way, and equal values keep their order. Without a sort, the order the API sent.
export function sortRows(rows, sort) {
  if (!sort) return rows;
  const sign = sort.direction === "asc" ? 1 : -1;
  const missing = (v) => v === null || v === undefined;
  return [...rows].sort((a, b) => {
    const x = sortValue(a, sort.key);
    const y = sortValue(b, sort.key);
    if (missing(x) || missing(y)) return missing(x) - missing(y);
    return x < y ? -sign : x > y ? sign : 0;
  });
}

// Clicking a column sorts it `first` ("desc" for scores, "asc" for times, tokens and names); clicking
// the sorted column again turns it around.
export function nextSort(current, key, first) {
  if (current?.key === key) return { key, direction: current.direction === "asc" ? "desc" : "asc" };
  return { key, direction: first };
}

function sortHeader(kind, key, label, { first, sort, numeric = false, description = "" }) {
  const active = sort?.key === key;
  const direction = active ? (sort.direction === "asc" ? "ascending" : "descending") : "none";
  const arrow = active ? (sort.direction === "asc" ? "▲" : "▼") : "";
  const title = description ? ` title="${escapeHtml(description)}"` : "";
  return (
    `<th class="sortable${numeric ? " num" : ""}" aria-sort="${direction}"${title}>` +
    `<button type="button" data-sort-kind="${kind}" data-sort-key="${escapeHtml(key)}" data-sort-first="${first}">` +
    `${escapeHtml(label)}<span class="arrow" aria-hidden="true">${arrow}</span></button></th>`
  );
}

function benchmarkTable(kind, results, sort) {
  const names = results.scores.map((s) => s.name);
  const best = bestScores(results.rows, names);
  const metric = (first) => (m) => sortHeader(kind, m.name, m.label, { first, sort, numeric: true, description: m.description });
  const head = [
    sortHeader(kind, "model", kind === "agent" ? "Chat model" : "Embedding model", { first: "asc", sort }),
    kind === "retrieval" ? sortHeader(kind, "ranking", "Ranking", { first: "asc", sort }) : "",
    ...results.scores.map(metric("desc")), // higher is better
    ...results.details.map(metric("asc")), // time and tokens: lower is better
    "<th>Langfuse</th>",
  ].join("");

  const rows = sortRows(results.rows, sort).map((row) => {
    const notes = [`${row.items} ${kind === "agent" ? "conversations" : "queries"}`, when(row.finished_at)];
    if (row.runs > 1) notes.push(`mean of ${row.runs} runs`);
    const lines = [notes.join(" · ")];
    // The index the agent searched, without its provider: "embeddings qwen/qwen3-embedding-8b".
    if (kind === "agent" && row.embedding_model) lines.push(`embeddings ${row.embedding_model.replace(/^[a-z]+:/, "")}`);
    if (row.failed) lines.push(`<span class="failed">${row.failed} failed</span>`);
    const scores = names.map((name) => {
      const value = row.scores[name];
      const top = value !== null && value !== undefined && value === best[name];
      return `<td class="num${top ? " best" : ""}">${percent(value)}</td>`;
    });
    const details = results.details.map((d) => `<td class="num">${detail(d.name, row.details[d.name])}</td>`);
    const links = row.urls.map((url, i) => {
      const label = row.urls.length > 1 ? `run ${i + 1}` : "open";
      return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${label}</a>`;
    });
    return [
      "<tr>",
      `<td class="model"><strong>${escapeHtml(row.model)}</strong>${lines.map((l) => (l.startsWith("<span") ? l : `<span>${escapeHtml(l)}</span>`)).join("")}</td>`,
      kind === "retrieval" ? `<td>${escapeHtml(RANKING_LABELS[row.ranking] ?? row.ranking ?? "")}</td>` : "",
      ...scores,
      ...details,
      `<td>${links.join(" ") || "–"}</td>`,
      "</tr>",
    ].join("");
  });
  return `<div class="table-scroll"><table class="bench-table"><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

const BENCHMARK_SECTIONS = {
  agent: {
    title: "Agent: chat models",
    about: "Whole conversations through the agent, every turn checked against the catalog: size searches, product " +
      "inquiries, education, follow-ups and off-topic requests.",
    command: "uv run python scripts/benchmark_agent.py --llm-model MODEL",
  },
  retrieval: {
    title: "Retrieval: embedding models",
    about: "Shopper queries through the catalog search: named products (hit@k, MRR) and needs like “mud tires " +
      "for my jeep” (P@10, nDCG@10).",
    command: "uv run python scripts/benchmark_retrieval.py --embedding-model MODEL",
  },
};

// The benchmarks view: a table per benchmark with the latest result of each model, the best value
// of each score in bold. Column headers explain their score on hover and sort the table when
// clicked; `sort` holds each table's sort ({agent: {key, direction}, ...}). All text is escaped.
export function renderBenchmarks(data, sort = {}) {
  const sections = Object.entries(BENCHMARK_SECTIONS).map(([kind, section]) => {
    const results = data[kind];
    const body = results.rows.length
      ? benchmarkTable(kind, results, sort[kind])
      : `<p class="none">No results yet. Run <code>${escapeHtml(section.command)}</code>.</p>`;
    return `<section class="bench"><h2>${escapeHtml(section.title)}</h2><p class="about">${escapeHtml(section.about)}</p>${body}</section>`;
  });
  return `<h1>Benchmark results</h1>${sections.join("")}<p class="about">Scores are averages over the cases. Hover a column name for what it measures; click it to sort.</p>`;
}
