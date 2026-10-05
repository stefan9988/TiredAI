// Run by tests/test_ui.py, or directly: node --test tests/ui/lib.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BENCHMARKS_URL,
  bestScores,
  chatIdFromUrl,
  chatUrl,
  createStatusQueue,
  errorMessage,
  formatValue,
  isBenchmarksUrl,
  nextSort,
  parseSSE,
  productColumns,
  renderBenchmarks,
  renderGuardrailNote,
  renderMarkdown,
  renderToolCalls,
  sortRows,
} from "../../src/tiredai/static/lib.mjs";

test("model output is escaped, so it cannot inject markup", () => {
  assert.equal(renderMarkdown('<img src=x onerror="alert(1)">'), "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt;</p>");
});

test("paragraphs, line breaks and inline styles", () => {
  assert.equal(
    renderMarkdown("\n\nThe **TBB TP-16** is *quiet*.\nUse `205/60R15`.\n\nSecond paragraph."),
    "<p>The <strong>TBB TP-16</strong> is <em>quiet</em>.<br>Use <code>205/60R15</code>.</p><p>Second paragraph.</p>",
  );
});

test("prices and sizes are not mistaken for italics", () => {
  assert.equal(renderMarkdown("2 * $56.91 for 205/60R15"), "<p>2 * $56.91 for 205/60R15</p>");
});

test("bulleted, numbered and nested lists", () => {
  assert.equal(
    renderMarkdown("- **TBB** — $56.91\n  - 50,000 mi warranty\n* Laufenn\n\n1. First\n2. Second"),
    '<ul><li><strong>TBB</strong> — $56.91</li><li class="sub">50,000 mi warranty</li><li>Laufenn</li></ul>' +
      "<ol><li>First</li><li>Second</li></ol>",
  );
});

test("rules separate the text before and after a search", () => {
  assert.equal(renderMarkdown("Let me check.\n\n---\n\nOne tire fits."), "<p>Let me check.</p><hr><p>One tire fits.</p>");
  assert.equal(renderMarkdown("- TBB\n***\n_ _ _\nDone."), "<ul><li>TBB</li></ul><hr><hr><p>Done.</p>");
  assert.equal(renderMarkdown("-- not a rule"), "<p>-- not a rule</p>");
});

test("headings", () => {
  assert.equal(renderMarkdown("## Best value\nLaufenn"), "<h3>Best value</h3><p>Laufenn</p>");
});

test("tables", () => {
  assert.equal(
    renderMarkdown("| Tire | Price |\n|---|---:|\n| TBB | $56.91 |\n| Laufenn | $77.99 |\n\nDone."),
    "<table><thead><tr><th>Tire</th><th>Price</th></tr></thead><tbody>" +
      "<tr><td>TBB</td><td>$56.91</td></tr><tr><td>Laufenn</td><td>$77.99</td></tr></tbody></table><p>Done.</p>",
  );
});

test("SSE: complete events are parsed and an unfinished tail is kept", () => {
  const { events, rest } = parseSSE(
    'event: start\ndata: {"conversation_id": "abc"}\n\n: keep-alive\n\nevent: token\ndata: {"text": "Hi"}\n\nevent: tok',
  );
  assert.deepEqual(events, [
    { event: "start", data: { conversation_id: "abc" } },
    { event: "token", data: { text: "Hi" } },
  ]);
  assert.equal(rest, "event: tok");
});

test("SSE: CRLF line endings and split data lines", () => {
  const { events } = parseSSE('event: status\r\ndata: {"stage": "thinking",\r\ndata: "text": "Thinking…"}\r\n\r\n');
  assert.deepEqual(events, [{ event: "status", data: { stage: "thinking", text: "Thinking…" } }]);
});

test("API error bodies become readable messages", () => {
  assert.equal(errorMessage({ detail: "Unknown conversation 'x'." }, 404), "Unknown conversation 'x'.");
  assert.equal(errorMessage({ detail: [{ msg: "message must not be blank" }] }, 422), "message must not be blank");
  assert.equal(errorMessage(null, 500), "Request failed (HTTP 500)");
});

function statusLine() {
  const log = [];
  const queue = createStatusQueue({ show: (s) => log.push(s.text), hide: () => log.push("<hidden>") });
  return { log, queue };
}

const thinking = { stage: "thinking", text: "Thinking…" };
const searching = { stage: "searching", text: "Searching the catalog: 205/60R15" };
const found = { stage: "results", text: "Found 6 tires" };

test("status: the first message shows at once, later ones replace it after the minimum time", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { log, queue } = statusLine();

  queue.push(thinking);
  queue.push(searching);
  queue.push(found);
  assert.deepEqual(log, ["Thinking…"]);

  t.mock.timers.tick(699);
  assert.deepEqual(log, ["Thinking…"]);
  t.mock.timers.tick(1);
  assert.deepEqual(log, ["Thinking…", "Searching the catalog: 205/60R15"]);
  t.mock.timers.tick(700);
  assert.deepEqual(log, ["Thinking…", "Searching the catalog: 205/60R15", "Found 6 tires"]);
});

test("status: a message that arrives after the minimum time shows at once", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { log, queue } = statusLine();

  queue.push(thinking);
  t.mock.timers.tick(2000);
  queue.push(searching);

  assert.deepEqual(log, ["Thinking…", "Searching the catalog: 205/60R15"]);
});

test("status: the last message hides on finish, but not before it was visible long enough", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { log, queue } = statusLine();

  queue.push(thinking);
  t.mock.timers.tick(300);
  queue.finish();
  assert.deepEqual(log, ["Thinking…"]);
  t.mock.timers.tick(400);
  assert.deepEqual(log, ["Thinking…", "<hidden>"]);
});

test("status: finishing after the minimum time hides at once", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { log, queue } = statusLine();

  queue.push(found);
  t.mock.timers.tick(1000);
  queue.finish();

  assert.deepEqual(log, ["Found 6 tires", "<hidden>"]);
});

test("status: queued steps still show after finish, except a stale thinking step", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { log, queue } = statusLine();

  queue.push(thinking);
  queue.push(searching);
  queue.push(found);
  queue.push(thinking);
  queue.finish(); // the answer started streaming
  for (let i = 0; i < 4; i++) t.mock.timers.tick(700); // fake timers run one due timer per tick

  assert.deepEqual(log, ["Thinking…", "Searching the catalog: 205/60R15", "Found 6 tires", "<hidden>"]);
});

test("status: a new step after finish shows again", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { log, queue } = statusLine();

  queue.push(thinking);
  queue.finish();
  t.mock.timers.tick(700);
  queue.push(searching); // the model wrote a few words, then decided to search

  assert.deepEqual(log, ["Thinking…", "<hidden>", "Searching the catalog: 205/60R15"]);
});

test("the open chat is kept in the URL", () => {
  assert.equal(chatUrl(null), "/");
  assert.equal(chatUrl("3f2a-b9"), "/?c=3f2a-b9");
  assert.equal(chatIdFromUrl("?c=3f2a-b9"), "3f2a-b9");
  assert.equal(chatIdFromUrl(""), null);
  assert.equal(chatIdFromUrl("?c="), null);
  assert.equal(chatIdFromUrl(new URL(chatUrl("a&b=c d"), "http://x").search), "a&b=c d");
});

test("tool call values are shown as they are, lists joined and ranges by their set ends", () => {
  assert.equal(formatValue("255/50R19"), "255/50R19");
  assert.equal(formatValue(59.93), "59.93");
  assert.equal(formatValue(false), "false");
  assert.equal(formatValue(["205/55R16", "205/55ZR16"]), "205/55R16, 205/55ZR16");
  assert.equal(formatValue({ min: null, max: 60 }), "max 60");
  assert.equal(formatValue({ min: 50, max: 100 }), "min 50, max 100");
  assert.equal(formatValue(null), "");
});

test("product table: name and price first, then every field in the order the products have them", () => {
  const products = [
    { sku: "N1", name: "Goodyear Eagle F1", size: "255/50R19", price: 241.99 },
    { sku: "N2", name: "Accelera Phi-R", size: "255/50R19", price: 99.5, runFlat: true },
  ];
  assert.deepEqual(productColumns(products), ["name", "price", "sku", "size", "runFlat"]);
  assert.deepEqual(productColumns([]), []);
});

const search = {
  id: "call-1",
  name: "search_tires",
  args: { size: "255 50 19", brand: "goodyear", max_price: 250 },
  error: null,
  result: {
    total_matching: 1234,
    returned: 2,
    order: "relevance",
    filters: { size: "255/50R19", brand: "Goodyear", price: { min: null, max: 250 } },
    products: [
      { sku: "N1", name: "Goodyear Eagle F1", price: 241.99, runFlat: false },
      { sku: "N2", name: "Goodyear Assurance", price: 180, loadRange: "XL" },
    ],
  },
};

test("a search shows what the model asked for, the filters applied and every product it got", () => {
  const html = renderToolCalls([search]);

  assert.match(html, /^<section class="tool-call"><h3>Search<\/h3>/);
  assert.ok(html.includes("<h4>The model asked for</h4><dl><dt>size</dt><dd>255 50 19</dd><dt>brand</dt><dd>goodyear</dd><dt>max_price</dt><dd>250</dd></dl>"));
  assert.ok(html.includes("<h4>Filters applied</h4><dl><dt>size</dt><dd>255/50R19</dd><dt>brand</dt><dd>Goodyear</dd><dt>price</dt><dd>max 250</dd></dl>"));
  assert.ok(html.includes("<p>2 of 1,234 matching · order: relevance</p>"));
  assert.ok(html.includes("<thead><tr><th>name</th><th>price</th><th>sku</th><th>runFlat</th><th>loadRange</th></tr></thead>"));
  assert.ok(html.includes("<tr><td>Goodyear Eagle F1</td><td>241.99</td><td>N1</td><td>false</td><td></td></tr>"));
  assert.ok(html.includes("<tr><td>Goodyear Assurance</td><td>180</td><td>N2</td><td></td><td>XL</td></tr>"));
});

test("several searches are numbered; a search without filters or products says so", () => {
  const empty = { ...search, args: {}, result: { total_matching: 0, returned: 0, order: "price_asc", filters: {}, products: [], note: "No products in the catalog match these filters." } };
  const html = renderToolCalls([search, empty]);

  assert.ok(html.includes("<h3>Search 1 of 2</h3>") && html.includes("<h3>Search 2 of 2</h3>"));
  assert.ok(html.includes('<h4>The model asked for</h4><p class="none">No arguments</p>'));
  assert.ok(html.includes('<p class="none">None: the whole catalog was searched</p>'));
  assert.ok(html.includes('<p>0 of 0 matching · order: price_asc</p><p class="none">No products in the catalog match these filters.</p></section>'));
  assert.equal(html.match(/<table/g).length, 1);
});

test("failed calls show the error the model got, and unparsable arguments as the raw text", () => {
  const html = renderToolCalls([
    { id: "bad", name: "search_tires", args: 'size: "205/55R15"', error: "Error: could not be parsed", result: null },
  ]);

  assert.ok(html.includes("<pre>size: &quot;205/55R15&quot;</pre>"));
  assert.ok(html.includes('<h4>The model got an error</h4><p class="error">Error: could not be parsed</p>'));
  assert.ok(!html.includes("<table"));
});

test("tool call details are escaped, so data cannot inject markup", () => {
  const html = renderToolCalls([
    {
      ...search,
      args: { query: "<b>quiet</b>" },
      result: { ...search.result, filters: { "<i>": "x" }, products: [{ name: "<img src=x onerror=alert(1)>" }] },
    },
  ]);

  assert.ok(!html.includes("<b>") && !html.includes("<i>") && !html.includes("<img"));
  assert.ok(html.includes("&lt;b&gt;quiet&lt;/b&gt;") && html.includes("&lt;img src=x onerror=alert(1)&gt;"));
});

test("other tools' output is shown as text", () => {
  const html = renderToolCalls([{ id: "c", name: "lookup", args: { q: 1 }, error: null, result: { ok: true } }]);

  assert.ok(html.includes("<h3>lookup</h3>") && html.includes('<h4>The model got</h4><pre>{\n  &quot;ok&quot;: true\n}</pre>'));
});

test("the benchmarks view has its own URL", () => {
  assert.equal(BENCHMARKS_URL, "/?view=benchmarks");
  assert.ok(isBenchmarksUrl("?view=benchmarks"));
  assert.ok(!isBenchmarksUrl("?c=abc") && !isBenchmarksUrl(""));
  assert.equal(chatIdFromUrl("?view=benchmarks"), null);
});

const METRICS = {
  agent: {
    scores: [
      { name: "passed", label: "Passed", description: "Every check passed", better: "higher" },
      { name: "groundedness", label: "Grounded", description: "Facts match <the data>", better: "higher" },
    ],
    details: [
      { name: "seconds_per_turn", label: "s / turn", description: "Mean time", better: "lower" },
      { name: "input_tokens", label: "Tokens in", description: "Prompt tokens", better: "lower" },
    ],
  },
  retrieval: {
    scores: [{ name: "reciprocal_rank", label: "MRR", description: "Mean reciprocal rank", better: "higher" }],
    details: [],
  },
  guardrail: {
    scores: [
      { name: "correct", label: "Correct", description: "Judged right", better: "higher" },
      { name: "false_block", label: "False blocks", description: "Blocked by mistake", better: "lower" },
    ],
    details: [
      { name: "auc", label: "AUC", description: "ROC AUC", better: "higher" },
      { name: "best_threshold", label: "Best threshold", description: "Best balanced accuracy", better: null },
      { name: "cost_usd", label: "Cost", description: "What the run cost", better: "lower" },
    ],
  },
};

// The API's response with these rows per benchmark, the others empty.
function benchmarkData(rows = {}) {
  return Object.fromEntries(Object.entries(METRICS).map(([kind, metrics]) => [kind, { ...metrics, rows: rows[kind] ?? [] }]));
}

function agentRow(model, passed, extra = {}) {
  return {
    model, embedding_model: "openrouter:qwen", ranking: null, finished_at: "2026-10-05T10:00:01Z", runs: 1, items: 27,
    failed: 0, scores: { passed, groundedness: 1 }, details: { seconds_per_turn: 4.577, input_tokens: 20565 },
    urls: ["https://langfuse.example/run?a=1&b=2"], ...extra,
  };
}

test("best scores per column: ties with the best win, a score every row shares marks nothing", () => {
  const rows = [agentRow("a", 0.5), agentRow("b", 0.9), agentRow("c", 0.9)];
  const scores = [...METRICS.agent.scores, { name: "missing", better: "higher" }];
  assert.deepEqual(bestScores(rows, scores), { passed: 0.9 });
});

test("best scores: the lowest wins where lower is better, and nothing where neither is", () => {
  const rows = [{ scores: { false_block: 0.1, best_threshold: 0.4 } }, { scores: { false_block: 0, best_threshold: 0.7 } }];
  const scores = [{ name: "false_block", better: "lower" }, { name: "best_threshold", better: null }];
  assert.deepEqual(bestScores(rows, scores), { false_block: 0 });
});

test("benchmark tables: a row per model, best scores marked, everything escaped", () => {
  const html = renderBenchmarks(benchmarkData({ agent: [agentRow("<b>ling</b>", 0.9), agentRow("qwen", 0.5, { failed: 2, runs: 3 })] }));

  assert.ok(html.startsWith("<h1>Benchmark results</h1>"));
  assert.ok(html.includes("<strong>&lt;b&gt;ling&lt;/b&gt;</strong>"));
  assert.ok(html.includes('<td class="num best">90.0%</td><td class="num">100.0%</td>')); // all 100%: no winner
  assert.ok(html.includes('<td class="num">50.0%</td>'));
  assert.ok(html.includes('title="Facts match &lt;the data&gt;"'));
  assert.ok(html.includes("<span>27 conversations · 2026-10-05 10:00 UTC</span></td><td>qwen</td>")); // without the provider
  assert.ok(html.includes("mean of 3 runs") && html.includes('<span class="failed">2 failed</span>'));
  assert.ok(html.includes('<td class="num">4.6</td><td class="num">21k</td>'));
  assert.ok(html.includes('<a href="https://langfuse.example/run?a=1&amp;b=2" target="_blank" rel="noopener">open</a>'));
  assert.ok(html.includes("No results yet. Run <code>uv run python scripts/benchmark_retrieval.py"));
});

test("retrieval rows show their ranking", () => {
  const row = { ...agentRow("BM25", null), ranking: "sparse", items: 188, embedding_model: null, scores: { reciprocal_rank: 0.985 }, urls: [] };
  const html = renderBenchmarks(benchmarkData({ retrieval: [row] }));

  assert.ok(html.includes("<td>BM25</td>") && html.includes("188 queries"));
  assert.ok(html.includes('<td class="num">98.5%</td>') && html.includes("<td>–</td>")); // one row: nothing to beat
});

test("a chat model benchmarked with two embedding models has a row for each", () => {
  const rows = [agentRow("ling", 0.9), agentRow("ling", 0.8, { embedding_model: "openrouter:openai/text-embedding-3-small" })];
  const html = renderBenchmarks(benchmarkData({ agent: rows }));

  assert.ok(html.includes('data-sort-key="embedding_model" data-sort-first="asc">Embeddings'));
  assert.ok(html.includes("</td><td>qwen</td>") && html.includes("</td><td>openai/text-embedding-3-small</td>"));
});

test("guardrail rows show what the guard saw and its threshold; fewest false blocks marked", () => {
  const row = (variant, correct, falseBlock, extra = {}) => ({
    ...agentRow("typesafe/jev-1.13", null), embedding_model: null, variant, threshold: 0.5, items: 116,
    scores: { correct, false_block: falseBlock }, details: { auc: 1, best_threshold: 0.41, cost_usd: 0.00385 }, ...extra,
  });
  const html = renderBenchmarks(benchmarkData({ guardrail: [row("message", 0.991, 0.02), row("recent", 1, 0)] }));

  assert.ok(html.includes('data-sort-key="model" data-sort-first="asc">Guard model'));
  assert.ok(html.includes("<span>116 messages · 2026-10-05 10:00 UTC</span></td><td>Message only</td><td class=\"num\">0.50</td>"));
  assert.ok(html.includes("<td>Recent (agent&#39;s window)</td>"));
  assert.ok(html.includes('<td class="num best">100.0%</td><td class="num best">0.0%</td>')); // most correct, fewest false blocks
  assert.ok(html.includes('<td class="num">1.00</td><td class="num">0.41</td><td class="num">$0.0039</td>'));
  assert.ok(html.includes('data-sort-key="false_block" data-sort-first="asc"') && html.includes('data-sort-key="auc" data-sort-first="desc"'));
  assert.ok(html.includes('data-sort-key="best_threshold" data-sort-first="asc"') && html.includes('data-sort-key="variant" data-sort-first="asc"'));
  assert.ok(html.includes("No results yet. Run <code>uv run python scripts/benchmark_agent.py"));
});

test("rows sort by a score, a detail, the model or a setup column; missing values go last", () => {
  const rows = [
    { model: "b", embedding_model: "openrouter:Zeta", ranking: "sparse", variant: "full", threshold: 0.5, scores: { passed: 0.5 }, details: { seconds_per_turn: 9 } },
    { model: "A", embedding_model: null, ranking: "hybrid", variant: "recent", threshold: null, scores: { passed: null }, details: { seconds_per_turn: 4 } },
    { model: "c", embedding_model: "fastembed:alpha", ranking: "dense", variant: "message", threshold: 0.3, scores: { passed: 0.9 }, details: {} },
  ];
  const order = (sort) => sortRows(rows, sort).map((r) => r.model);

  assert.deepEqual(order({ key: "passed", direction: "desc" }), ["c", "b", "A"]);
  assert.deepEqual(order({ key: "passed", direction: "asc" }), ["b", "c", "A"]);
  assert.deepEqual(order({ key: "seconds_per_turn", direction: "asc" }), ["A", "b", "c"]);
  assert.deepEqual(order({ key: "model", direction: "asc" }), ["A", "b", "c"]);
  assert.deepEqual(order({ key: "ranking", direction: "asc" }), ["A", "c", "b"]); // hybrid, dense, BM25
  assert.deepEqual(order({ key: "embedding_model", direction: "asc" }), ["c", "b", "A"]); // by name, not provider
  assert.deepEqual(order({ key: "variant", direction: "asc" }), ["c", "A", "b"]); // message, recent, full
  assert.deepEqual(order({ key: "threshold", direction: "desc" }), ["b", "c", "A"]);
  assert.equal(sortRows(rows, undefined), rows); // no sort: the API's order
});

test("a first click sorts the useful way round, a second click turns it around", () => {
  const scores = nextSort(undefined, "passed", "desc");
  assert.deepEqual(scores, { key: "passed", direction: "desc" });
  assert.deepEqual(nextSort(scores, "passed", "desc"), { key: "passed", direction: "asc" });
  assert.deepEqual(nextSort(scores, "seconds_per_turn", "asc"), { key: "seconds_per_turn", direction: "asc" });
});

test("sortable headers say how their table is sorted", () => {
  const html = renderBenchmarks(benchmarkData({ agent: [agentRow("ling", 0.9), agentRow("qwen", 0.5)] }), { agent: { key: "passed", direction: "asc" } });

  assert.ok(html.indexOf("<strong>qwen</strong>") < html.indexOf("<strong>ling</strong>"));
  assert.ok(html.includes('<th class="sortable num" aria-sort="ascending" title="Every check passed">'));
  assert.ok(html.includes('data-sort-kind="agent" data-sort-key="passed" data-sort-first="desc">Passed<span class="arrow" aria-hidden="true">▲</span>'));
  assert.ok(html.includes('data-sort-key="seconds_per_turn" data-sort-first="asc"'));
  assert.ok(html.includes('aria-sort="none"><button type="button" data-sort-kind="agent" data-sort-key="model" data-sort-first="asc">Chat model'));
  assert.ok(html.includes("<th>Langfuse</th>")); // links don't sort
});

test("an answer from the guardrail says why, with the scores on hover", () => {
  const note = renderGuardrailNote({
    blocked: true,
    reason: "off_topic",
    block_score: 0.96,
    threshold: 0.7,
    probabilities: { in_scope: 0.04, manipulation: 0.02, harmful: 0.01 },
    model: "typesafe/jev-1.13-20260917",
  });

  assert.match(note, /<span>Answered by the guardrail: off-topic<\/span>/);
  assert.match(
    note,
    /title="Block score 0\.96 \(blocks at 0\.70\) · in scope 0\.04 · manipulation 0\.02 · harmful 0\.01 · typesafe\/jev-1\.13-20260917"/,
  );
});

test("guardrail notes escape what they show and name every reason", () => {
  assert.match(renderGuardrailNote({ reason: "manipulation", threshold: 0.7 }), /tries to change the rules/);
  assert.match(renderGuardrailNote({ reason: "harmful", threshold: 0.7 }), /harmful request/);
  const odd = renderGuardrailNote({ reason: "<b>new</b>", block_score: null, model: '"><script>' });
  assert.ok(!odd.includes("<b>") && !odd.includes("<script>"));
  assert.match(odd, /Block score \? /);
});
