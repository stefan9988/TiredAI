# TiredAI

A RAG chatbot that helps shoppers find and understand tires from a catalog of about 10,000 products. It answers four kinds of requests:

| Request | Example | What the assistant does |
| --- | --- | --- |
| Size search | "I need all-season tires in 205/60R15" | Matches the size exactly, asks for a missing size or preferences, recommends only matching, in-stock tires |
| Product inquiry | "Goodyear Eagle F1 Asymmetric SUV-4X 255/50R19 103W" | Finds that exact product and answers from its data; says clearly when it isn't in the catalog instead of swapping in another tire |
| Education | "What does UTQG mean?" | Answers from general tire knowledge without searching the catalog |
| Vehicle lookup | "What tires fit my 2019 Toyota RAV4 LE?" | Looks the car up on tire-size websites, then searches the catalog in its size, or lists the sizes per trim and asks which one |

It can run entirely on free tiers: an open model on OpenRouter, free or local embeddings, and a local Qdrant vector store. Only the optional vehicle lookup is paid, about $0.007 per web search.

![The chat page in dark mode: a new chat with the four example questions](docs/chat-page-dark.png)

## Architecture

```
                 preprocess_dataset.py             build_index.py
 tires CSV ───────────────────────────▶ tires.parquet ───────────────▶ Qdrant collection
 (raw text)      lossless normalization,              embed + load      one point per product:
                 verified round trip                                    dense + BM25 vectors,
                                                                        product payload
                                                                               ▲
                                                                               │ filters + hybrid ranking
 Browser (chat UI) ──▶ FastAPI ──▶ LangChain agent ──▶ search_tires tool ──────┘
 Terminal (chat.py) ─┘    │         (guardrail: Jev,
                          │          OpenRouter LLM,  ──▶ find_vehicle_tire_sizes tool ──▶ OpenRouter web search
                          │          system prompt)       (cached in SQLite)               (four tire-size sites)
                          ▼
                 SQLite: conversation history
                 (LangGraph checkpoints + chat list)
```

| Module | Role |
| --- | --- |
| `src/tiredai/preprocessing.py` | Normalizes the raw CSV into typed columns and verifies every value against the raw text |
| `src/tiredai/documents.py` | Builds the text that gets embedded and the payload stored with each product |
| `src/tiredai/embeddings.py` | Dense embeddings (local fastembed or the OpenRouter API, cached on disk) and BM25 sparse vectors |
| `src/tiredai/vectorstore.py` | Creates the Qdrant collection, loads products and verifies every stored payload |
| `src/tiredai/search.py` | The `search_tires` tool: hard filters, hybrid ranking, sorting, and what the agent sees of each product |
| `src/tiredai/vehicles.py` | The `find_vehicle_tire_sizes` tool: a web search for a vehicle's tire sizes on a few tire-size sites, cached |
| `src/tiredai/agent.py` | The agent: OpenRouter chat model, system prompt, middleware for the guardrail, limits and tool errors, streaming |
| `src/tiredai/guardrail.py` | The guardrail: asks Jev (TypeSafe's decision model) whether a message is in scope, manipulative or harmful |
| `src/tiredai/api.py` | FastAPI server: chat (JSON and streamed), saved conversations, health, the chat page |
| `src/tiredai/tracing.py` | Langfuse tracing: one trace per message, grouped by conversation |
| `src/tiredai/static/` | The chat UI: plain HTML, CSS and JavaScript, with no build step |
| `prompts/system.md` | System prompt: the four request types and the rules for product facts, searching and the conversation |

### How a message is answered

1. The chat page sends the shopper's message to `POST /chat/stream`.
2. With the Guardrail button above the message box on (the default), Jev checks the message in the context of the recent conversation, in about half a second. If it is off-topic, manipulative or harmful, the shopper gets a fixed reply and the turn ends there: the chat model never sees the message.
3. The agent sees the system prompt and the recent conversation, and decides what kind of request it is. Education questions are answered directly.
4. For product questions it calls `search_tires`. Every constraint the shopper has given (size, budget, season, brand, ...) becomes a filter, and the product name or description becomes the query. A shopper who doesn't know the size but names the car first gets a vehicle lookup (`find_vehicle_tire_sizes`), which finds the factory sizes on the web.
5. Qdrant applies the filters and ranks the matching products. The tool returns the number of matches and up to 20 products as JSON.
6. The model writes its answer from those products only. The server streams status updates ("Searching the catalog: 205/60R15 · up to $80", "Found 4 tires"), each finished search, and the answer text as Server-Sent Events.
7. The whole turn, including tool calls and results (or the guardrail's decision), is saved, so the conversation continues after a restart.

## Key technical decisions

### Data: lossless preprocessing

The raw CSV is all text, with `N/A` and empty cells meaning "no data". Preprocessing applies only conversions that provably keep every value: prices and diameters become numbers, `runFlat` becomes a boolean, `10/32` becomes `treadDepth32nds = 10`, `50,000 miles` becomes `mileageWarrantyMiles = 50000`. Missing markers become null, and a lowercase `x` in sizes becomes `X`. The Parquet file is read back and every value is compared with the raw text. If one doesn't match, nothing is written.

The catalog has no stock or rating information, so preprocessing adds two generated columns:

- `available`: in stock (about 80% of products).
- `recommendations`: the store's recommendation level from 1 to 5. It is a whole number drawn from a normal distribution around 4 (spread 0.6): about 20% score 3, 60% score 4 and 20% score 5.

Both are derived from a hash of each product's SKU, so every rebuild produces the same values, and the verification step checks them too.

### Hard constraints are filters, never ranking hints

Every constraint the shopper states is a Qdrant filter: size, price range, season, brand, car type, performance category, run-flat, minimum speed rating and minimum recommendation level. Every returned product is guaranteed to satisfy all of them; the free-text query only orders the results. The tool's answer includes `total_matching`, so "nothing matches" is a fact the model can report, not a guess.

The filters handle how shoppers actually write things:

- **Sizes** are normalized before matching: `205 55 16`, `205/55/16`, `P205/55ZR16` and `205/55R16` are the same size. Number formatting is ignored (`5.20-13` matches `5.2-13`). An `LT` prefix also restricts results to light-truck tires.
- **Speed ratings** use the real speed order, which is not alphabetical: H < V < Z < W < Y.
- **Unknown values** are reported back to the model instead of being dropped. A size that isn't in the catalog is reported as such, an unknown season, car type or performance category gets the list of allowed values, and a misspelled brand gets close matches ("Did you mean: Goodyear?"). The model asks the shopper rather than silently searching for something else.

### Hybrid search

Each product gets a dense embedding and a BM25 keyword vector of a short description: the product name, season, performance category, car type, plus "Run-flat" and the sidewall style only when they carry information. Qdrant takes the top 50 candidates from each and fuses them with reciprocal rank fusion. Dense vectors catch descriptive queries ("quiet touring tire"), while BM25 catches the exact codes in product names (`SUV-4X`, `103W`, `KO2`) that embeddings tend to blur. Numbers and codes such as price, warranty and UTQG stay in the payload for filtering instead of being embedded.

Without a query, results are sorted by price. The model can also sort by price in either direction or by recommendation level; with a query, that sort applies to the 50 most relevant products.

### What the model sees of each product

The tool returns product data exactly as stored, with three adjustments that came from observed mistakes:

- **`model` is hidden.** In 21% of rows the model text doesn't appear in the product name: 13% of all rows hold a number or size instead, and 3% hold another brand's name. The full product name is reliable and is what the assistant uses. The field stays in the stored payload.
- **Units are sent as text.** Given a bare number, the model misreported a tread depth of 8/32" as "8,000". The tool now sends `treadDepth: "8/32 in"`, `mileageWarranty: "50,000 miles"` and `recommendations: "4/5"`. The stored values stay numeric.
- **Out-of-stock tires are returned, flagged** with `available: false`, instead of filtered out. That way, asking about a specific out-of-stock tire gets "it's out of stock" rather than "it doesn't exist". The system prompt tells the model to recommend only available tires.

### Grounded answers

The system prompt requires every price, specification and SKU to come from search results in the conversation, and forbids estimating them. Constraints carry over between turns ("something cheaper" keeps the size and lowers the price). The prompt also tells the model never to loosen a constraint without asking.

To check an answer against its data, every answer that used the catalog has a small search button in the chat UI. It opens a side panel with each search's arguments as the model sent them, the filters actually applied, the number of matches, and the full table of products the model received, or the error it got instead. This works for saved conversations too, because tool calls and results are stored with the history.

### Two tools, with guard rails around them

The agent has two tools, `search_tires` and `find_vehicle_tire_sizes` (below), and a system prompt that defines the four request types; the model decides on every message which one applies. Around them:

- **History window:** the model sees only the latest messages (`AGENT_HISTORY_MESSAGES`, default 10), cut at whole turns so a tool call is never separated from its result. The saved conversation keeps everything.
- **Tool call limit:** at most `AGENT_MAX_TOOL_CALLS` searches per message (default 5). Further calls get an error result and the model answers with what it has.
- **Tool failures reach the model:** a search that raises, an unknown argument (e.g. `speed_rating` instead of `min_speed_rating`), or arguments that aren't valid JSON all come back as error results the model can act on. If all of the model's calls were unparsable, it is asked again up to two times. Every tool call in the history keeps a result, which providers require.

### Looking a car up when the shopper doesn't know the size

Many shoppers know their car but not their tire size. `find_vehicle_tire_sizes` takes the year, make and model; the agent asks for whichever is missing and never guesses them. It searches four tire-size sites with OpenRouter's web search (the web plugin with the Exa engine) and returns the pages it found: site, address, title and the excerpt the search engine read.

- **Four sites, chosen by testing:** tiresize.com, firestonecompleteautocare.com, mavis.com and goodyear.com returned the right car's page, with every trim's sizes in the excerpt, for each test car. Tire Rack and Discount Tire build their pages in the browser, so the search gets no sizes from them. Car review sites list one trim per page, and new-cars.com returned the wrong years. The list is `VEHICLE_LOOKUP_SITES`.
- **The chat model reads the pages itself.** There is no second model to extract sizes. The system prompt tells it to use only pages about that exact year, make and model. A fitment is one size, or a pair of different front and rear sizes. If the shopper's trim has one fitment, the agent searches the catalog with it right away (one search per size of a pair). If it has several, the agent lists them by trim with the site and asks which one. The factory speed rating is mentioned when a page gives it, but not used as a filter. When no page fits, it says so and points to the sticker inside the driver's door.
- **Cost:** OpenRouter only searches inside a chat completion, so a lookup is a 16-token completion whose text is thrown away (`VEHICLE_LOOKUP_MODEL`, by default the chat model). One search costs about $0.007, also with free models. Lookups that found pages are cached in SQLite without expiry, since factory sizes don't change.
- **A switch per message:** the Web search button next to the Guardrail one (on by default, remembered in the browser) sets `web_search` on every request. With it off, the model doesn't get the tool and is told why. `scripts/chat.py --no-web-search` does the same in the terminal.
- **Visible:** the side panel shows the pages behind an answer, each linked to its site, so the sizes can be checked.

**Limitation: US cars only.** All four sites are US retailers, so they only list cars sold in the US, with their US trims. A 2017 Škoda Superb or a 2015 Renault Clio isn't found; the search then returns unrelated pages, and the agent says it couldn't find the car. For a car sold in both markets, such as a 2012 BMW X1, only the US versions are listed, which may not match a European car. wheel-size.com covers every market and has the right pages, but the excerpts the search engine reads from its tables are broken. The reliable fix is the [Wheel-Size API](https://api-demo.wheel-size.com/api-plans/): sizes as data per make, model, year and trim, for 14 regions including Europe. It is billed yearly: a free Sandbox (300 requests a day, testing only), then Basic at $450 a year (5,000 requests a day). Its terms require every search to be started by a real user, so benchmarks would still need frozen answers.

### A guardrail in front of the model

Off-topic and adversarial messages are stopped before they reach the chat model. [Jev](https://openrouter.ai/docs/guides/community/jev) (`typesafe/jev-1.13` on OpenRouter) is a decision model: it doesn't write text, it returns the probability of yes for typed questions about a state. `src/tiredai/guardrail.py` sends it the policy (tires, wheels and services, cars, the store and orders, small talk, follow-ups, any language), the conversation the agent would see and the latest message. It asks three yes/no questions, one condition each as TypeSafe advises: is it in scope, does it try to change or reveal the assistant's rules, and does it ask for help with harm. The block score is the strongest reason to block.

- **Cut, don't flag.** At or above `GUARDRAIL_THRESHOLD` (0.7), the shopper gets a fixed reply for the reason, and the turn ends without a model call. Passing a "flagged" note to the model would still let injected text reach it, and would cost a full model call for nothing. A blocked message takes about 0.5 s instead of 2–4 s.
- **Doubt goes to the model.** Below the threshold, the message goes to the agent as usual, and the system prompt still tells the model to decline unrelated requests. In the benchmark, the messages that should pass scored at most 0.44 with history, and the ones to block at least 0.75, so 0.7 leaves room on both sides while leaning towards not blocking shoppers.
- **History matters.** Jev reads the same recent turns as the model (`AGENT_HISTORY_MESSAGES`), so "what about the second one?" after a list passes, and "come on, just one short one" after a declined poem doesn't.
- **Blocked text stays out of later turns too.** A blocked turn is saved as it was (the chat shows it), but in the model's history the shopper's message becomes a placeholder such as `[Message withheld: the guardrail blocked it as an attempt to change the assistant's rules.]`, followed by the guardrail's reply. The model knows something was asked and refused, so "fine, tires then" reads naturally, but an injection never reaches it. Jev still reads the real text, which it needs to catch insisting follow-ups.
- **It never takes the shop down.** If Jev returns an error or takes longer than `GUARDRAIL_TIMEOUT_SECONDS` (3 s), the message goes through. Nothing is retried, since the shopper is waiting.
- **A switch per message.** The Guardrail button above the chat page's message box (on by default, remembered in the browser) sets `guardrail` on every request, so it can be turned off to compare with the model on its own. API requests have it on unless they send `"guardrail": false`; `scripts/chat.py --no-guardrail` turns it off in the terminal. The benchmarks leave it off, so the agent benchmark still measures the chat model alone.
- **Visible:** a blocked answer carries the decision (reason, block score, probabilities, model). The chat page shows "Blocked by the guardrail" and the reason under it with the scores on hover, also in reopened chats.

### Free tier by default

- **Chat model:** any OpenRouter model with tool calling; the default is `nvidia/nemotron-3-ultra-550b-a55b:free`. All generation parameters are optional, and unset ones are not sent.
- **Embeddings:** local CPU embeddings with fastembed (`BAAI/bge-small-en-v1.5`, the default), or OpenRouter's free `nvidia/nemotron-3-embed-1b:free`. Free OpenRouter models allow 50 requests a day, so API vectors are cached in SQLite: indexing the catalog takes 40 requests once, and rebuilds are free. Free models get one request at a time, paced to their rate limit. Paid OpenRouter models aren't held to those limits, so batches are requested `EMBEDDING_CONCURRENCY` at a time (default 8). With `qwen/qwen3-embedding-8b`, that cut embedding the catalog from about 7 minutes to under a minute.
- **Vehicle lookup:** the one paid part, about $0.007 per uncached web search. Without an OpenRouter key the agent simply doesn't have the tool.
- **Vector store:** Qdrant instead of Pinecone. It runs as a local on-disk store with no account or server, supports dense and sparse vectors with server-side fusion, payload filters and ordering, and can point to a Qdrant server or Qdrant Cloud through `QDRANT_URL`.

## Observability

Tracing uses [Langfuse](https://langfuse.com) through its LangChain integration (Python SDK 4.16). To turn it on, set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`; `LANGFUSE_BASE_URL` picks the region or a self-hosted server. Without keys nothing is recorded or sent. The server log says at startup whether tracing is on, and warns if the keys are rejected or the server can't be reached.

Each shopper message is one trace, and the traces of a conversation form one Langfuse session, so a whole chat can be replayed in the Sessions view. A turn with a search looks like this:

```
answer-shopper-message            span        input: the shopper's message, output: the answer
└─ tire_agent                     agent       the agent loop
   ├─ model                       chain
   │  └─ ChatOpenRouter           generation  full prompt, answer or tool calls, reasoning, model, tokens
   ├─ tools                       chain
   │  └─ search_tires             tool        arguments as the model sent them, the JSON it got back
   │     └─ retrieve-products     retriever   filters as applied, number of matches, ranking with scores
   │        └─ embed-query        embedding   the query text and the embedding model
   └─ model                       chain
      └─ ChatOpenRouter           generation  the answer written from the search results
```

A vehicle lookup appears as a `find_vehicle_tire_sizes` tool with a `look-up-vehicle-sizes` retriever inside it: the vehicle, the search query and sites, the pages found, whether they came from the cache and what the search cost.

With the guardrail on, the trace also holds a `check-message` observation of type guardrail: the state Jev read and its decision (probabilities, block score, reason). For a blocked message, the agent span has no model call and the trace's output is the guardrail's reply.

The model calls and the catalog lookup are separate observations, so you can inspect retrieval and generation on their own. You can see what the model was asked, what the search applied and returned, and what the model answered from it. Each trace also carries:

- **Tags:** where the message came from: `chat-stream` (the chat page), `chat` (the JSON endpoint) or `cli`.
- **Metadata:** the chat model, the embedding model, the agent limits and whether the guardrail and the web search were on, to compare setups.
- **Environment:** from `LANGFUSE_TRACING_ENVIRONMENT`, so development traces stay apart from others.
- **Errors:** a failed turn or query embedding is marked as an error, with the message. When searches go over the limit, the step that blocked them stays in the trace with the error results; otherwise that bookkeeping step is left out.

Token usage is recorded for every generation. Langfuse calculates cost only for models in its price list, so to see cost for an OpenRouter model, add a model definition with its prices under Project Settings > Models in Langfuse. API keys are never part of a trace.

## Benchmarks

Two benchmarks show whether the system works and compare models, and a third measures the guardrail. They take the models as flags, so any embedding model or chat model can be benchmarked without editing `.env`. Each run is a Langfuse experiment.

| Benchmark | What it runs | Scores |
| --- | --- | --- |
| Retrieval (`scripts/benchmark_retrieval.py`) | 188 queries through the catalog search, per embedding model and ranking (hybrid as the app uses it, dense alone, BM25 alone) | Product queries: hit@1, hit@3, hit@10, reciprocal rank (mean = MRR). Descriptive queries: precision@10, nDCG@10 |
| Agent (`scripts/benchmark_agent.py`) | 32 conversations (39 turns) through the real agent, per chat model | Intent accuracy, retrieval hit@3, filters applied, constraint correctness, groundedness, answer checks, passed; also latency per turn and tokens |
| Guardrail (`scripts/benchmark_guardrail.py`) | 118 shopper messages to allow or block, 34 of them follow-ups after a real conversation, through TypeSafe's Jev decision model, each with no history, the agent's history window and the full history | Accuracy, false-block rate, share of bad messages caught, ROC AUC, best threshold; also latency, tokens and cost |

```bash
uv run python scripts/benchmark_retrieval.py \
  --embedding-model openrouter:qwen/qwen3-embedding-8b --embedding-model fastembed:BAAI/bge-small-en-v1.5
uv run python scripts/benchmark_agent.py \
  --llm-model inclusionai/ling-3.0-flash-vl --llm-model qwen/qwen3-30b-a3b-instruct-2507 [--repeat 3]
uv run python scripts/benchmark_guardrail.py [--variant message|recent|full] [--threshold 0.5]
```

Without flags, both use the models in `.env`. `--check` only validates the cases against the catalog, `--local` sends nothing to Langfuse, and `--case ID` runs some agent cases only. Each embedding model gets its own in-memory index, so the app's `data/vectorstore/` is never touched and a running server doesn't get in the way. Vectors come from the embedding cache, so only a model's first run calls its API.

**The cases** are files in `benchmarks/`, reviewed like code:

- `retrieval_products.yaml`: 40 products sampled across car types (generated by `scripts/generate_retrieval_queries.py`, seeded), each asked for in four styles: the catalog name, how shoppers type it (`accelera phi-r 205 55 15`), with a typo, and the tread line without a size.
- `retrieval_descriptive.yaml`: 30 needs such as "mud tires for my jeep". Relevance comes from catalog attributes (Mud Terrain, Truck/SUV or Light Truck).
- `guardrail_cases.yaml`: messages the guardrail must allow (tires, wheels and services, cars, the store, small talk, other languages, a tire request mixed with an off-topic one, alarming but legitimate wording) or block (off-topic, writing only themed on tires, prompt injection, harm and fraud), and follow-ups such as "what about the second one?" after a list. The conversations before the follow-ups were played through the agent once and frozen in `guardrail_conversations.yaml` (`scripts/capture_guardrail_conversations.py`), so every run shows Jev the same real answers.
- `agent_cases.yaml`: size searches (including loose formats, an LT size, a misspelled brand and incomplete sizes), product inquiries (including typos and products that don't exist), education questions, follow-ups that must keep their constraints ("something cheaper"), vehicle lookups, and off-topic and prompt-injection requests. Every product, size and price is a real catalog row, and `--check` verifies they still are.
- `vehicle_pages.yaml`: the pages the vehicle lookup found for the cars in the agent cases: a 2016 Ford Focus (pick the trim, or the S with one size), a 2016 Corvette Stingray (a front and rear pair), a 2019 RAV4 (the year is missing at first) and a 2021 Ford Focus, which the sites don't have. `scripts/capture_vehicle_pages.py` searches once for each new car and freezes the pages, so the agent benchmark never searches the web and every run reads the same pages.

**Deterministic scoring.** The scores are checks against the catalog, not an LLM judge. Intent is read from what the agent did: education and off-topic questions must not search, a product inquiry must search for that product, and a size search must filter by the size (or ask first, where the case allows it). A vehicle lookup must look the car up and then search every size of its one fitment, or ask which trim, or not search at all when the car wasn't found; with the year missing, it must ask instead of looking up. After a lookup, every tire size in the answer must be on its pages. Every product the answer names must meet the turn's constraints and be in stock. Every price, SKU and spec it states must match the products the search returned. The details are in the docstrings of `src/tiredai/benchmarks/conversations.py` and `answers.py`; the comment on each score says which turn failed and why.

**In Langfuse,** the case files are synced to the datasets `tiredai-retrieval`, `tiredai-agent` and `tiredai-guardrail`: changed cases are updated and removed ones archived. Every run is a dataset run named after its model, with the git commit, the system prompt's hash and the agent settings in its metadata. Under the dataset's Experiments tab, runs of different models can be compared score by score. Each case is a trace with its scores. In the agent benchmark that trace holds the usual turn traces (tagged `benchmark`), so a failing turn can be inspected down to the search. The summary is also printed and saved to `benchmarks/results/` as Markdown, with every case's output and scores in a JSON file next to it.

**On the chat page,** the Benchmark results button under New chat shows them: one table for chat models with the embedding model their searches used, one for embedding models with each ranking, and one for guard models with how much of the conversation they saw and the threshold. Each row is the latest run of that setup, with repeats averaged. The best value of each score is highlighted, and each row links to its Langfuse run. Hovering a column name explains the score, and clicking it sorts the table: scores highest first, times and tokens lowest first, and a second click reverses the order. Benchmarking a setup again replaces only its row, so a chat model run with two embedding models has a row for each. Docker mounts `./benchmarks`, so the page there shows new results without a rebuild. Where lower is better (false blocks, times, tokens, cost), the lowest is highlighted and sorts first.

**Cost.** One agent run is roughly 100 chat requests. `:free` models run one conversation at a time, and the 50 requests a day of a free account won't cover a full run. One retrieval run per model and ranking sends about 1,900 observations and scores to Langfuse; `--local` skips that.

### Results

The runs of 2026-10-05. The full reports are in [`benchmarks/results/`](benchmarks/results/), with every case's output and the reason for each failed check, and in the Langfuse datasets `tiredai-agent` and `tiredai-retrieval`. Each model ran once; LLM answers vary between runs, so `--repeat 3` gives a steadier comparison.

**Agent** ([report](benchmarks/results/20261005T112306Z-agent.md); 27 conversations, 32 turns; commit `a246a3f`, embeddings `qwen/qwen3-embedding-8b`):

| Chat model | Passed | Intent | Hit@3 | Filters | Constraints | Grounded | Answers | s / turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inclusionai/ling-3.0-flash-vl` (current) | **96.3%** | **100%** | 100% | 100% | 100% | 99.6% | **100%** | 3.9 |
| `qwen/qwen3-30b-a3b-instruct-2507` | 92.6% | 98.1% | 100% | 100% | 100% | **100%** | 93.8% | **3.8** |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 85.2% | 87.0% | 100% | 100% | 100% | **100%** | **100%** | 17 |

- **ling-3.0-flash-vl** failed one conversation: it said the Milestar costs "$1.31 more" than the GT Radial ($83.95 vs. $82.71, so $1.24 more).
- **qwen3-30b** failed two. It answered "show me something cheaper" from the earlier results without searching again, though the system prompt says to search again. It also said the nonexistent Michelin Pilot Sport 9 "is not available in 245/40R18", which suggests it exists in other sizes, instead of saying it isn't in the catalog.
- **nemotron-3-ultra:free** failed four conversations, all on `503 Service temporarily overloaded` from Nvidia's free endpoint after three attempts. 15 of its 27 conversations needed a retry. Every conversation that completed passed. It is also about four times slower per turn.
- No model recommended an out-of-stock tire, missed a hard constraint in its searches, or stated a price or spec the search didn't return. The one exception is ling's arithmetic slip.
- **With the vehicle lookup** ([report](benchmarks/results/20261005T150559Z-agent.md); 32 conversations, 39 turns), `inclusionai/ling-3.0-flash-vl` passed all 32. Run three times, the five vehicle conversations always took the right route (lookup, then search, ask or decline). Two of the 15 failed a check: one answer left out the door sticker, and one stated a recommendation level the check attributed to the wrong product.

**Retrieval** ([report](benchmarks/results/20261005T104820Z-retrieval.md); 158 product queries and 30 descriptive ones; commit `c5f5148`):

| Embedding model | Ranking | Hit@1 | MRR | P@10 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| `nvidia/nemotron-3-embed-1b:free` | hybrid | 98.7% | 99.4% | 86.3% | 90.2% |
| `openai/text-embedding-3-small` | hybrid | 95.6% | 97.7% | 89.7% | 92.8% |
| `qwen/qwen3-embedding-8b` (current) | hybrid | 93.7% | 96.7% | 86.7% | 90.4% |
| `BAAI/bge-small-en-v1.5` (local) | hybrid | 89.2% | 94.4% | 84.3% | 89.6% |
| `nvidia/nemotron-3-embed-1b:free` | dense | **99.4%** | **99.7%** | 90.7% | 93.7% |
| `openai/text-embedding-3-small` | dense | 90.5% | 94.3% | **93.0%** | **96.7%** |
| `qwen/qwen3-embedding-8b` | dense | 93.0% | 95.8% | 88.3% | 92.2% |
| `BAAI/bge-small-en-v1.5` | dense | 70.3% | 78.7% | 84.3% | 91.7% |
| BM25 alone | sparse | 97.5% | 98.5% | 83.7% | 86.4% |

Hit@3 and hit@10 are left out here: they are 97.5–100% for every run except bge-small dense (85.4% and 96.8%).

- **Named products are nearly solved.** BM25 alone ranks the right product first 97.5% of the time, because product names are mostly distinctive words and sizes. These queries separate the embedding models only a little.
- **Hybrid ranking protects against a weak embedding model.** bge-small alone reaches an MRR of 78.7%, and with BM25 it reaches 94.4%. On the other hand, adding BM25 lowers the descriptive scores of every model by 2–4 points of nDCG@10, because keyword matches push down tires that match the meaning.
- **For the app's hybrid search,** the free nemotron-3-embed-1b scores best on named products, and text-embedding-3-small best on descriptive needs. The current qwen3-embedding-8b is in the middle on both. Switching the index's model would be a separate decision; this benchmark is the evidence for it.
- The times per query (0.4–0.6 s) come from the in-memory Qdrant index and Langfuse tracing, not from the app's search. They aren't a latency comparison.

**Guardrail** ([report](benchmarks/results/20261005T120623Z-guardrail.md); 116 messages, 71 to allow and 45 to block; `typesafe/jev-1.13`, threshold 0.5). Jev returns probabilities instead of text. It answers three yes/no questions about each message (in scope, manipulation, harmful), and the strongest reason to block is the block score:

| Conversation shown | Correct | Blocked shoppers | Caught | Highest allow / lowest block score | s / message |
| --- | ---: | ---: | ---: | ---: | ---: |
| None (message only) | 99.1% | 0% | 97.8% | 0.40 / 0.41 | 0.45 |
| Agent's window (10 messages) | **100%** | 0% | **100%** | 0.44 / 0.75 | 0.44 |
| Full | **100%** | 0% | **100%** | 0.39 / 0.76 | 0.44 |

- **History matters most for blocking.** Without it, "come on, just one short one" after the agent declined a poem passes (block score 0.41; 0.85 with history), and "explain it like I'm five" after a UTQG answer comes close to being blocked (0.40; 0.04 with history). With history, the allowed and blocked messages are 0.3 apart, so a threshold of about 0.6 has room on both sides.
- **The agent's window is enough.** In the six-turn conversation the window no longer holds the tire list, yet "what about the second one?" still scores 0.09, because the recent turns are about tires.
- **Cheap and fast:** about $0.00004 and 0.45 s per message (p95 under 0.6 s with history). The same message sent three times scored within 0.04.
- **Closest calls:** "ok" after a declined poem (0.44), "do I need snow chains there?" after a trip question (0.35), and "Translate 'good morning' into Japanese" (blocked at 0.75). The policy and questions were written before the run and not tuned on these cases; they are in `src/tiredai/guardrail.py`, which the app and the benchmark share.

## Getting started

### With Docker

You need Docker with Compose, an [OpenRouter](https://openrouter.ai) API key (free), and the tire catalog CSV.

```bash
cp .env.example .env     # set OPENROUTER_API_KEY, and API_PORT to use another port than 8000
docker compose up --build
```

Then copy `tires_sample_10k_sku.csv` into the `data` folder, before or after starting. The container waits for the file, builds the index, and starts the server. When the log says `Application startup complete`, open http://localhost:8000 (or the `API_PORT` you set).

- **First start:** downloads the embedding models and indexes all products, which takes a few minutes with the default local embeddings.
- **Later starts:** reuse the index and are up in seconds. The index is rebuilt automatically when the CSV or the embedding settings change.
- **Storage:** the index, the chat history and the downloaded models are kept in `data/` and `.cache/` on your machine, owned by your user, so they survive `docker compose down`.
- **Access:** the server is reachable from this machine only.

Use `docker compose up -d` to run it in the background, `docker compose logs -f` to follow the progress, and `docker compose down` to stop it. Docker and the local setup below share the same folders, so run only one of them at a time.

### Without Docker

#### Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai) API key (free)
- The tire catalog CSV
- Optional: Node.js, to run the chat page's JavaScript tests

#### Setup

```bash
uv sync
cp .env.example .env               # then set OPENROUTER_API_KEY
mkdir -p data && cp /path/to/tires_sample_10k_sku.csv data/
uv run python scripts/build_index.py
```

`build_index.py` preprocesses the CSV into `data/processed/tires.parquet`, embeds every product, loads it into Qdrant (`data/vectorstore/`), and checks every stored payload. With the default local embeddings, the models are downloaded on the first run. Use `--skip-preprocess` to reuse the existing Parquet file, or `--if-changed` to rebuild only when the data or embedding settings changed since the last build.

To inspect the data first:

```bash
uv run python scripts/analyze_dataset.py       # data-quality report for the raw CSV
uv run python scripts/preprocess_dataset.py    # normalization summary and generated-column shares
```

#### Run

```bash
uv run python scripts/serve.py                 # chat UI at http://localhost:8000
uv run python scripts/chat.py                  # or chat in the terminal (/new starts over)
uv run python scripts/chat.py "What does UTQG mean?"
```

The chat page lists every saved conversation in a sidebar, and the open chat is kept in the URL, so reloading reopens it. The local Qdrant store allows one process at a time, so stop the server before running `build_index.py`.

### Configuration

All settings live in `.env`; `.env.example` lists every option with comments. The main ones:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | | Required for the chat model, and for OpenRouter embeddings |
| `LLM_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Any OpenRouter model that supports tool calling |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, ... | unset | Optional generation parameters; unset ones are not sent |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` (local) or `openrouter`; rebuild the index after changing it |
| `EMBEDDING_CONCURRENCY` | `8` | Requests sent at once when indexing with a paid OpenRouter model (`:free` models always send one at a time) |
| `QDRANT_PATH` / `QDRANT_URL` | `data/vectorstore` | Local store, or a Qdrant server URL |
| `AGENT_HISTORY_MESSAGES` | `10` | Chat messages the model sees |
| `AGENT_MAX_TOOL_CALLS` | `5` | Searches per shopper message |
| `AGENT_MAX_SEARCH_RESULTS` | `20` | Products returned by each search |
| `GUARDRAIL_MODEL` | `typesafe/jev-1.13` | The OpenRouter Decisions model that checks messages |
| `GUARDRAIL_THRESHOLD` | `0.7` | Block score at or above which a message gets the guardrail's reply |
| `GUARDRAIL_TIMEOUT_SECONDS` | `3` | A slower check lets the message through |
| `VEHICLE_LOOKUP_SITES` | tiresize.com, firestonecompleteautocare.com, mavis.com, goodyear.com | The sites the vehicle lookup searches (comma-separated domains) |
| `VEHICLE_LOOKUP_MAX_RESULTS` | `5` | Pages per lookup |
| `VEHICLE_LOOKUP_MODEL` | the chat model | Model of the small completion that carries the search; set a cheap one if `LLM_MODEL` is expensive |
| `VEHICLE_LOOKUP_TIMEOUT_SECONDS` | `20` | A slower search gives the agent an error result |
| `VEHICLE_LOOKUP_CACHE_PATH` | `.cache/vehicle_pages.sqlite` | Cache of lookups that found pages |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Server address; with Docker, `API_PORT` is the port opened on your machine |
| `SYSTEM_PROMPT_PATH` | `prompts/system.md` | The system prompt |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | | Langfuse project keys; tracing is off without them |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse region (US: `https://us.cloud.langfuse.com`) or self-hosted server |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` (in `.env.example`) | Environment name the traces are filed under |

## API

The server is a local demo without authentication. Interactive docs are at `/docs`.

| Endpoint | Description |
| --- | --- |
| `POST /chat` | Send `{"message": ..., "conversation_id": ..., "guardrail": true, "web_search": true}` and get the whole reply. Omit `conversation_id` to start a new conversation; `"guardrail": false` skips the guardrail, and `"web_search": false` takes the vehicle lookup away for that message. When the guardrail answered, `guardrail` in the reply holds its decision. |
| `POST /chat/stream` | The same, streamed as Server-Sent Events: `start`, then `status`, `tool_call` and `token` events as the agent works, then `end` (or `error`). A blocked message gets its reply as one `token`, then a `guardrail` event with the decision. |
| `GET /conversations` | Every saved conversation, most recently active first; the title is its first message |
| `GET /conversations/{id}/messages` | A conversation's messages; each answer lists its tool calls with the data the model received (products, or a vehicle lookup's pages), and the guardrail's decision if it answered |
| `GET /benchmarks` | The latest benchmark result of each model, from `benchmarks/results/`, with what each score measures |
| `GET /health` | Model name, vector store status, and whether the guardrail and the vehicle lookup are available (both need `OPENROUTER_API_KEY`) |

## Tests

```bash
uv run pytest
```

The tests run offline: scripted chat models, a deterministic fake embedder and fake OpenRouter transports (embeddings, chat completions with web search results, and Jev's Decisions API) stand in for every external service, and `OPENROUTER_API_KEY` is blanked so nothing can reach OpenRouter by accident. They cover preprocessing round trips, size normalization and every search filter, the agent's limits and error handling, the guardrail (blocking, letting through, failing open, turning it off), the vehicle lookup (which pages are kept, caching, errors, the web search switch), the API and streaming, conversation storage, and the benchmarks' metrics, answer checks and Langfuse dataset sync. The chat page's JavaScript helpers are tested with `node --test tests/ui/lib.test.mjs`, which pytest also runs when Node.js is installed.

## Project layout

```
prompts/system.md         system prompt
scripts/                  analyze_dataset, preprocess_dataset, build_index, chat, serve, start (Docker entrypoint),
                          benchmark_retrieval, benchmark_agent, benchmark_guardrail,
                          generate_retrieval_queries, capture_guardrail_conversations, capture_vehicle_pages
src/tiredai/              preprocessing, documents, embeddings, vectorstore, search, vehicles, guardrail, agent, api, conversations, config,
                          startup, tracing
src/tiredai/static/       chat UI (index.html, app.js, lib.mjs, style.css)
src/tiredai/benchmarks/   benchmark cases, scoring and Langfuse experiments
benchmarks/               benchmark cases (YAML) and results/
tests/                    pytest suite; tests/ui/ holds the JavaScript tests
data/                     raw CSV, processed Parquet, vector store and conversation history (not in git)
docs/                     screenshot of the chat page
Dockerfile, docker-compose.yml
```
