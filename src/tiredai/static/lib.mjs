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

// A page's title as a link that opens in a new tab; only web addresses become links.
function pageLink(page) {
  const title = escapeHtml(page.title || page.url || "");
  if (!/^https?:\/\//i.test(page.url ?? "")) return title;
  return `<a href="${escapeHtml(page.url)}" target="_blank" rel="noopener noreferrer">${title}</a>`;
}

function lookupResult(result) {
  const html = [
    "<h4>Pages the model got</h4>",
    `<p>${result.pages.length} for the ${escapeHtml(result.vehicle ?? "vehicle")} from ${escapeHtml(formatValue(result.sites))}</p>`,
  ];
  if (result.note) html.push(`<p class="none">${escapeHtml(result.note)}</p>`);
  for (const page of result.pages) {
    html.push(`<h5>${pageLink(page)} <span class="site">${escapeHtml(page.site ?? "")}</span></h5>`, `<pre>${escapeHtml(page.excerpt ?? "")}</pre>`);
  }
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
  } else if (Array.isArray(call.result?.pages)) {
    html.push(lookupResult(call.result));
  } else {
    const output = typeof call.result === "string" ? call.result : JSON.stringify(call.result, null, 2);
    html.push("<h4>The model got</h4>", `<pre>${escapeHtml(output ?? "")}</pre>`);
  }
  return `<section class="tool-call">${html.join("")}</section>`;
}

const TOOL_LABELS = { search_tires: "Search", find_vehicle_tire_sizes: "Vehicle lookup" };

// The side panel's view of the tool calls behind an answer: for each, what the model asked for and
// the error or data it got back, so the answer can be checked against it. All text is escaped.
export function renderToolCalls(calls) {
  return calls
    .map((call, i) => {
      const label = TOOL_LABELS[call.name] ?? call.name;
      return toolCall(call, calls.length > 1 ? `${label} ${i + 1} of ${calls.length}` : label);
    })
    .join("");
}

// Why the guardrail answered instead of the chat model, by the reason the API gives.
const GUARDRAIL_REASONS = { off_topic: "off-topic", manipulation: "prompt injection", harmful: "harmful request" };

const SHIELD_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 5 6v5c0 4.4 3 8.3 7 9.5 4-1.2 7-5.1 7-9.5V6l-7-3Z" ' +
  'fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';

const score = (value) => (typeof value === "number" ? value.toFixed(2) : "?");

// The note under an answer the guardrail gave: its reason, and on hover the scores behind it.
export function renderGuardrailNote(decision) {
  const reason = GUARDRAIL_REASONS[decision.reason] ?? decision.reason ?? "blocked";
  const probabilities = Object.entries(decision.probabilities ?? {}).map(([name, p]) => `${name.replace("_", " ")} ${score(p)}`);
  const details = [`Block score ${score(decision.block_score)} (blocks at ${score(decision.threshold)})`, ...probabilities];
  if (decision.model) details.push(decision.model);
  return (
    `<p class="guardrail-note" title="${escapeHtml(details.join(" · "))}">` +
    `${SHIELD_ICON}<span>Blocked by the guardrail: ${escapeHtml(reason)}</span></p>`
  );
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
  if (name.startsWith("cost")) return `$${value.toFixed(value < 0.01 ? 4 : 2)}`;
  if (name === "auc" || name.endsWith("threshold")) return value.toFixed(2);
  return value < 1 ? value.toFixed(2) : value.toFixed(1);
}

// "2026-10-05T10:00:01Z" -> "2026-10-05 10:00 UTC"
function when(iso) {
  return iso ? `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC` : "";
}

// The best value of each score over the rows (the highest, or the lowest where `better` is "lower"),
// where some row is worse: every row it ties wins, but a score all rows share marks nothing.
export function bestScores(rows, scores) {
  const best = {};
  for (const { name, better } of scores) {
    if (better !== "higher" && better !== "lower") continue;
    const values = rows.map((r) => r.scores[name]).filter((v) => v !== null && v !== undefined);
    if (!values.length || Math.max(...values) === Math.min(...values)) continue;
    best[name] = better === "lower" ? Math.min(...values) : Math.max(...values);
  }
  return best;
}

const RANKING_LABELS = { hybrid: "Hybrid (app)", dense: "Dense", sparse: "BM25" };
const RANKING_ORDER = ["hybrid", "dense", "sparse"];
const VARIANT_LABELS = { message: "Message only", recent: "Recent (agent's window)", full: "Whole conversation" };
const VARIANT_ORDER = ["message", "recent", "full"];

// The index an agent row searched, without its provider: "qwen/qwen3-embedding-8b".
function embeddingName(row) {
  return row.embedding_model ? row.embedding_model.replace(/^[a-z]+:/, "") : null;
}

function place(order, value) {
  const i = order.indexOf(value);
  return i < 0 ? order.length : i;
}

// The columns after the model that say what else a row ran with: the cell's text, and the value
// it sorts by (names A–Z; rankings and variants in their own order, the app's ranking first).
const SETUP_COLUMNS = {
  agent: [
    {
      key: "embedding_model", label: "Embeddings", description: "The embedding model of the index the agent searched",
      text: embeddingName, sort: (row) => embeddingName(row)?.toLowerCase() ?? null,
    },
  ],
  retrieval: [
    { key: "ranking", label: "Ranking", text: (row) => RANKING_LABELS[row.ranking] ?? row.ranking, sort: (row) => place(RANKING_ORDER, row.ranking) },
  ],
  guardrail: [
    {
      key: "variant", label: "Conversation", description: "How much of the conversation before the message the guard saw",
      text: (row) => VARIANT_LABELS[row.variant] ?? row.variant, sort: (row) => place(VARIANT_ORDER, row.variant),
    },
    {
      key: "threshold", label: "Threshold", description: "Messages were blocked at this block score or above", numeric: true,
      text: (row) => detail("threshold", row.threshold), sort: (row) => row.threshold ?? null,
    },
  ],
};
const SETUP_SORT = Object.fromEntries(Object.values(SETUP_COLUMNS).flat().map((c) => [c.key, c.sort]));

// A row's value in a column: the model name, a setup column's value, or a score or detail.
function sortValue(row, key) {
  if (key === "model") return row.model.toLowerCase();
  if (key in SETUP_SORT) return SETUP_SORT[key](row);
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

// Clicking a column sorts it `first` (the better values first: "desc" for most scores, "asc" for
// times, tokens, false blocks and names); clicking the sorted column again turns it around.
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
  const section = BENCHMARK_SECTIONS[kind];
  const setup = SETUP_COLUMNS[kind] ?? [];
  const best = bestScores(results.rows, results.scores);
  const metric = (m) =>
    sortHeader(kind, m.name, m.label, { first: m.better === "higher" ? "desc" : "asc", sort, numeric: true, description: m.description });
  const head = [
    sortHeader(kind, "model", section.model, { first: "asc", sort }),
    ...setup.map((c) => sortHeader(kind, c.key, c.label, { first: "asc", sort, numeric: c.numeric, description: c.description })),
    ...results.scores.map(metric),
    ...results.details.map(metric),
    "<th>Langfuse</th>",
  ].join("");

  const rows = sortRows(results.rows, sort).map((row) => {
    const notes = [`${row.items} ${section.items}`, when(row.finished_at)];
    if (row.runs > 1) notes.push(`mean of ${row.runs} runs`);
    const lines = [notes.join(" · ")];
    if (row.failed) lines.push(`<span class="failed">${row.failed} failed</span>`);
    const scores = results.scores.map(({ name }) => {
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
      ...setup.map((c) => `<td${c.numeric ? ' class="num"' : ""}>${escapeHtml(c.text(row) ?? "–")}</td>`),
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
    model: "Chat model",
    items: "conversations",
    about: "Whole conversations through the agent, every turn checked against the catalog: size searches, product " +
      "inquiries, education, follow-ups and off-topic requests.",
    command: "uv run python scripts/benchmark_agent.py --llm-model MODEL",
  },
  retrieval: {
    title: "Retrieval: embedding models",
    model: "Embedding model",
    items: "queries",
    about: "Shopper queries through the catalog search: named products (hit@k, MRR) and needs like “mud tires " +
      "for my jeep” (P@10, nDCG@10).",
    command: "uv run python scripts/benchmark_retrieval.py --embedding-model MODEL",
  },
  guardrail: {
    title: "Guardrail: guard models",
    model: "Guard model",
    items: "messages",
    about: "Shopper messages the assistant should handle or stop (off-topic requests, prompt injections, harmful " +
      "asks), judged with none, some or all of the conversation before them.",
    command: "uv run python scripts/benchmark_guardrail.py --model MODEL",
  },
};

// The benchmarks view: a table per benchmark with the latest result of each setup, the best value
// of each score highlighted. Column headers explain their score on hover and sort the table when
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
