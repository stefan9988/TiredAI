// Run by tests/test_ui.py, or directly: node --test tests/ui/lib.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { createStatusQueue, errorMessage, parseSSE, renderMarkdown } from "../../src/tiredai/static/lib.mjs";

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
