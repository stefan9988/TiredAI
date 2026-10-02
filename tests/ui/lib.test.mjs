// Run by tests/test_ui.py, or directly: node --test tests/ui/lib.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  chatIdFromUrl,
  chatUrl,
  createStatusQueue,
  errorMessage,
  formatValue,
  parseSSE,
  productColumns,
  renderMarkdown,
  renderToolCalls,
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
