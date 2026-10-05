# TiredAI

A RAG chatbot that helps shoppers find and understand tires from a catalog of about 10,000 products. It answers three kinds of requests:

| Request | Example | What the assistant does |
| --- | --- | --- |
| Size search | "I need all-season tires in 205/60R15" | Matches the size exactly, asks for a missing size or preferences, recommends only matching, in-stock tires |
| Product inquiry | "Goodyear Eagle F1 Asymmetric SUV-4X 255/50R19 103W" | Finds that exact product and answers from its data; says clearly when it isn't in the catalog instead of swapping in another tire |
| Education | "What does UTQG mean?" | Answers from general tire knowledge without searching the catalog |

It runs entirely on free tiers: an open model on OpenRouter, free or local embeddings, and a local Qdrant vector store.

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
 Terminal (chat.py) ─┘    │         (OpenRouter LLM,
                          │          system prompt)
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
| `src/tiredai/agent.py` | The agent: OpenRouter chat model, system prompt, middleware for limits and tool errors, streaming |
| `src/tiredai/api.py` | FastAPI server: chat (JSON and streamed), saved conversations, health, the chat page |
| `src/tiredai/tracing.py` | Langfuse tracing: one trace per message, grouped by conversation |
| `src/tiredai/static/` | The chat UI: plain HTML, CSS and JavaScript, with no build step |
| `prompts/system.md` | System prompt: the three request types and the rules for product facts, searching and the conversation |

### How a message is answered

1. The chat page sends the shopper's message to `POST /chat/stream`.
2. The agent sees the system prompt and the recent conversation, and decides what kind of request it is. Education questions are answered directly.
3. For product questions it calls `search_tires`. Every constraint the shopper has given (size, budget, season, brand, ...) becomes a filter, and the product name or description becomes the query.
4. Qdrant applies the filters and ranks the matching products. The tool returns the number of matches and up to 20 products as JSON.
5. The model writes its answer from those products only. The server streams status updates ("Searching the catalog: 205/60R15 · up to $80", "Found 4 tires"), each finished search, and the answer text as Server-Sent Events.
6. The whole turn, including tool calls and results, is saved, so the conversation continues after a restart.

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

### A single tool, with guard rails around it

The agent has one tool, `search_tires`, and a system prompt that defines the three request types; the model decides on every message which one applies. Around it:

- **History window:** the model sees only the latest messages (`AGENT_HISTORY_MESSAGES`, default 10), cut at whole turns so a tool call is never separated from its result. The saved conversation keeps everything.
- **Tool call limit:** at most `AGENT_MAX_TOOL_CALLS` searches per message (default 5). Further calls get an error result and the model answers with what it has.
- **Tool failures reach the model:** a search that raises, an unknown argument (e.g. `speed_rating` instead of `min_speed_rating`), or arguments that aren't valid JSON all come back as error results the model can act on. If all of the model's calls were unparsable, it is asked again up to two times. Every tool call in the history keeps a result, which providers require.

### Free tier only

- **Chat model:** any OpenRouter model with tool calling; the default is `nvidia/nemotron-3-ultra-550b-a55b:free`. All generation parameters are optional, and unset ones are not sent.
- **Embeddings:** local CPU embeddings with fastembed (`BAAI/bge-small-en-v1.5`, the default), or OpenRouter's free `nvidia/nemotron-3-embed-1b:free`. Free OpenRouter models allow 50 requests a day, so API vectors are cached in SQLite: indexing the catalog takes 40 requests once, and rebuilds are free. Free models get one request at a time, paced to their rate limit. Paid OpenRouter models aren't held to those limits, so batches are requested `EMBEDDING_CONCURRENCY` at a time (default 8). With `qwen/qwen3-embedding-8b`, that cut embedding the catalog from about 7 minutes to under a minute.
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

The model calls and the catalog lookup are separate observations, so you can inspect retrieval and generation on their own. You can see what the model was asked, what the search applied and returned, and what the model answered from it. Each trace also carries:

- **Tags:** where the message came from: `chat-stream` (the chat page), `chat` (the JSON endpoint) or `cli`.
- **Metadata:** the chat model, the embedding model and the agent limits, to compare setups.
- **Environment:** from `LANGFUSE_TRACING_ENVIRONMENT`, so development traces stay apart from others.
- **Errors:** a failed turn or query embedding is marked as an error, with the message. When searches go over the limit, the step that blocked them stays in the trace with the error results; otherwise that bookkeeping step is left out.

Token usage is recorded for every generation. Langfuse calculates cost only for models in its price list, so to see cost for an OpenRouter model, add a model definition with its prices under Project Settings > Models in Langfuse. API keys are never part of a trace.

## Benchmarks

Two benchmarks show whether the system works and compare models. Both take the models as flags, so any embedding model or chat model can be benchmarked without editing `.env`. Each run is a Langfuse experiment.

| Benchmark | What it runs | Scores |
| --- | --- | --- |
| Retrieval (`scripts/benchmark_retrieval.py`) | 188 queries through the catalog search, per embedding model and ranking (hybrid as the app uses it, dense alone, BM25 alone) | Product queries: hit@1, hit@3, hit@10, reciprocal rank (mean = MRR). Descriptive queries: precision@10, nDCG@10 |
| Agent (`scripts/benchmark_agent.py`) | 27 conversations (32 turns) through the real agent, per chat model | Intent accuracy, retrieval hit@3, filters applied, constraint correctness, groundedness, answer checks, passed; also latency per turn and tokens |

```bash
uv run python scripts/benchmark_retrieval.py \
  --embedding-model openrouter:qwen/qwen3-embedding-8b --embedding-model fastembed:BAAI/bge-small-en-v1.5
uv run python scripts/benchmark_agent.py \
  --llm-model inclusionai/ling-3.0-flash-vl --llm-model qwen/qwen3-30b-a3b-instruct-2507 [--repeat 3]
```

Without flags, both use the models in `.env`. `--check` only validates the cases against the catalog, `--local` sends nothing to Langfuse, and `--case ID` runs some agent cases only. Each embedding model gets its own in-memory index, so the app's `data/vectorstore/` is never touched and a running server doesn't get in the way. Vectors come from the embedding cache, so only a model's first run calls its API.

**The cases** are files in `benchmarks/`, reviewed like code:

- `retrieval_products.yaml`: 40 products sampled across car types (generated by `scripts/generate_retrieval_queries.py`, seeded), each asked for in four styles: the catalog name, how shoppers type it (`accelera phi-r 205 55 15`), with a typo, and the tread line without a size.
- `retrieval_descriptive.yaml`: 30 needs such as "mud tires for my jeep". Relevance comes from catalog attributes (Mud Terrain, Truck/SUV or Light Truck).
- `agent_cases.yaml`: size searches (including loose formats, an LT size, a misspelled brand and incomplete sizes), product inquiries (including typos and products that don't exist), education questions, follow-ups that must keep their constraints ("something cheaper"), and off-topic and prompt-injection requests. Every product, size and price is a real catalog row, and `--check` verifies they still are.

**Deterministic scoring.** The scores are checks against the catalog, not an LLM judge. Intent is read from what the agent did: education and off-topic questions must not search, a product inquiry must search for that product, and a size search must filter by the size (or ask first, where the case allows it). Every product the answer names must meet the turn's constraints and be in stock. Every price, SKU and spec it states must match the products the search returned. The details are in the docstrings of `src/tiredai/benchmarks/conversations.py` and `answers.py`; the comment on each score says which turn failed and why.

**In Langfuse,** the case files are synced to the datasets `tiredai-retrieval` and `tiredai-agent`: changed cases are updated and removed ones archived. Every run is a dataset run named after its model, with the git commit, the system prompt's hash and the agent settings in its metadata. Under the dataset's Experiments tab, runs of different models can be compared score by score. Each case is a trace with its scores. In the agent benchmark that trace holds the usual turn traces (tagged `benchmark`), so a failing turn can be inspected down to the search. The summary is also printed and saved to `benchmarks/results/` as Markdown, with every case's output and scores in a JSON file next to it.

**On the chat page,** the Benchmarks button under New chat shows those results: one table for chat models and one for embedding models with each ranking. Each row is a model's latest run, with repeats averaged. The best value of each score is highlighted, and each row links to its Langfuse run. Hovering a column name explains the score, and clicking it sorts the table: scores highest first, times and tokens lowest first, and a second click reverses the order. Benchmarking one model again replaces only its row. Docker mounts `./benchmarks`, so the page there shows new results without a rebuild.

**Cost.** One agent run is roughly 100 chat requests. `:free` models run one conversation at a time, and the 50 requests a day of a free account won't cover a full run. One retrieval run per model and ranking sends about 1,900 observations and scores to Langfuse; `--local` skips that.

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
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Server address; with Docker, `API_PORT` is the port opened on your machine |
| `SYSTEM_PROMPT_PATH` | `prompts/system.md` | The system prompt |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | | Langfuse project keys; tracing is off without them |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse region (US: `https://us.cloud.langfuse.com`) or self-hosted server |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` (in `.env.example`) | Environment name the traces are filed under |

## API

The server is a local demo without authentication. Interactive docs are at `/docs`.

| Endpoint | Description |
| --- | --- |
| `POST /chat` | Send `{"message": ..., "conversation_id": ...}` and get the whole reply. Omit `conversation_id` to start a new conversation. |
| `POST /chat/stream` | The same, streamed as Server-Sent Events: `start`, then `status`, `tool_call` and `token` events as the agent works, then `end` (or `error`) |
| `GET /conversations` | Every saved conversation, most recently active first; the title is its first message |
| `GET /conversations/{id}/messages` | A conversation's messages; each answer lists its tool calls with the data the model received |
| `GET /benchmarks` | The latest benchmark result of each model, from `benchmarks/results/`, with what each score measures |
| `GET /health` | Model name and vector store status |

## Tests

```bash
uv run pytest
```

The tests run offline: scripted chat models, a deterministic fake embedder and a fake OpenRouter transport stand in for every external service. They cover preprocessing round trips, size normalization and every search filter, the agent's limits and error handling, the API and streaming, conversation storage, and the benchmarks' metrics, answer checks and Langfuse dataset sync. The chat page's JavaScript helpers are tested with `node --test tests/ui/lib.test.mjs`, which pytest also runs when Node.js is installed.

## Project layout

```
prompts/system.md         system prompt
scripts/                  analyze_dataset, preprocess_dataset, build_index, chat, serve, start (Docker entrypoint),
                          benchmark_retrieval, benchmark_agent, generate_retrieval_queries
src/tiredai/              preprocessing, documents, embeddings, vectorstore, search, agent, api, conversations, config, startup, tracing
src/tiredai/static/       chat UI (index.html, app.js, lib.mjs, style.css)
src/tiredai/benchmarks/   benchmark cases, scoring and Langfuse experiments
benchmarks/               benchmark cases (YAML) and results/
tests/                    pytest suite; tests/ui/ holds the JavaScript tests
data/                     raw CSV, processed Parquet, vector store and conversation history (not in git)
Dockerfile, docker-compose.yml
```
