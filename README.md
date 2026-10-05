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

## API

The server is a local demo without authentication. Interactive docs are at `/docs`.

| Endpoint | Description |
| --- | --- |
| `POST /chat` | Send `{"message": ..., "conversation_id": ...}` and get the whole reply. Omit `conversation_id` to start a new conversation. |
| `POST /chat/stream` | The same, streamed as Server-Sent Events: `start`, then `status`, `tool_call` and `token` events as the agent works, then `end` (or `error`) |
| `GET /conversations` | Every saved conversation, most recently active first; the title is its first message |
| `GET /conversations/{id}/messages` | A conversation's messages; each answer lists its tool calls with the data the model received |
| `GET /health` | Model name and vector store status |

## Tests

```bash
uv run pytest
```

The tests run offline: scripted chat models, a deterministic fake embedder and a fake OpenRouter transport stand in for every external service. They cover preprocessing round trips, size normalization and every search filter, the agent's limits and error handling, the API and streaming, and conversation storage. The chat page's JavaScript helpers are tested with `node --test tests/ui/lib.test.mjs`, which pytest also runs when Node.js is installed.

## Project layout

```
prompts/system.md         system prompt
scripts/                  analyze_dataset, preprocess_dataset, build_index, chat, serve, start (Docker entrypoint)
src/tiredai/              preprocessing, documents, embeddings, vectorstore, search, agent, api, conversations, config, startup
src/tiredai/static/       chat UI (index.html, app.js, lib.mjs, style.css)
tests/                    pytest suite; tests/ui/ holds the JavaScript tests
data/                     raw CSV, processed Parquet, vector store and conversation history (not in git)
Dockerfile, docker-compose.yml
```
