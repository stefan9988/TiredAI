"""Project settings, read from environment variables and an optional .env file (see .env.example)."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: str) -> Path:
    # Relative paths are resolved against the project root, so scripts work from any directory.
    path = Path(os.getenv(name) or default)
    return path if path.is_absolute() else ROOT / path


def _parsed(env: Mapping[str, str], name: str, parse, kind: str):
    """Parse an optional variable; empty or unset means 'not configured'."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return None
    try:
        return parse(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {kind}") from exc


def _bool(raw: str) -> bool:
    values = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    if raw.lower() not in values:
        raise ValueError(raw)
    return values[raw.lower()]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise ValueError(raw)
    return value


def _json_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(raw)
    return value


@dataclass(frozen=True)
class LLMSettings:
    """Chat model settings. Parameters left unset are not sent, so the provider's defaults apply."""

    model: str
    base_url: str | None
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None
    reasoning_effort: str | None
    extra_params: dict[str, Any]
    timeout_seconds: float | None
    max_retries: int
    streaming: bool
    system_prompt_path: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "LLMSettings":
        prompt = Path(env.get("SYSTEM_PROMPT_PATH") or "prompts/system.md")
        max_retries = _parsed(env, "LLM_MAX_RETRIES", int, "integer")
        return cls(
            model=env.get("LLM_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free",
            base_url=env.get("LLM_BASE_URL") or None,
            temperature=_parsed(env, "LLM_TEMPERATURE", float, "number"),
            top_p=_parsed(env, "LLM_TOP_P", float, "number"),
            max_tokens=_parsed(env, "LLM_MAX_TOKENS", int, "integer"),
            seed=_parsed(env, "LLM_SEED", int, "integer"),
            frequency_penalty=_parsed(env, "LLM_FREQUENCY_PENALTY", float, "number"),
            presence_penalty=_parsed(env, "LLM_PRESENCE_PENALTY", float, "number"),
            reasoning_effort=env.get("LLM_REASONING_EFFORT") or None,
            extra_params=_parsed(env, "LLM_EXTRA_PARAMS", _json_object, "JSON object") or {},
            timeout_seconds=_parsed(env, "LLM_TIMEOUT_SECONDS", float, "number"),
            max_retries=2 if max_retries is None else max_retries,
            streaming=_parsed(env, "LLM_STREAMING", _bool, "boolean (true/false)") is not False,
            system_prompt_path=prompt if prompt.is_absolute() else ROOT / prompt,
        )


@dataclass(frozen=True)
class AgentSettings:
    """Limits that keep each model request small."""

    history_messages: int  # earlier chat messages the model sees, counted in whole turns
    max_tool_calls: int  # per shopper message
    max_search_results: int  # products per search

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AgentSettings":
        def limit(name: str, default: int) -> int:
            value = _parsed(env, name, _positive_int, "positive integer")
            return default if value is None else value

        return cls(
            history_messages=limit("AGENT_HISTORY_MESSAGES", 10),
            max_tool_calls=limit("AGENT_MAX_TOOL_CALLS", 5),
            max_search_results=limit("AGENT_MAX_SEARCH_RESULTS", 20),
        )


@dataclass(frozen=True)
class Settings:
    raw_data_path: Path
    processed_data_path: Path
    qdrant_path: Path
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    embedding_provider: str
    embedding_model: str
    sparse_model: str
    model_cache_dir: Path
    embedding_cache_path: Path
    embedding_concurrency: int  # parallel requests to a paid OpenRouter embedding model
    openrouter_api_key: str | None
    llm: LLMSettings
    agent: AgentSettings
    conversations_path: Path
    api_host: str
    api_port: int

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        return cls(
            raw_data_path=_path("RAW_DATA_PATH", "data/tires_sample_10k_sku.csv"),
            processed_data_path=_path("PROCESSED_DATA_PATH", "data/processed/tires.parquet"),
            qdrant_path=_path("QDRANT_PATH", "data/vectorstore"),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv("QDRANT_COLLECTION") or "tires",
            embedding_provider=os.getenv("EMBEDDING_PROVIDER") or "fastembed",
            embedding_model=os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5",
            sparse_model=os.getenv("SPARSE_MODEL") or "Qdrant/bm25",
            model_cache_dir=_path("MODEL_CACHE_DIR", ".cache/fastembed"),
            embedding_cache_path=_path("EMBEDDING_CACHE_PATH", ".cache/embeddings.sqlite"),
            embedding_concurrency=_parsed(os.environ, "EMBEDDING_CONCURRENCY", _positive_int, "positive integer") or 8,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            llm=LLMSettings.from_env(os.environ),
            agent=AgentSettings.from_env(os.environ),
            conversations_path=_path("CONVERSATIONS_DB_PATH", "data/conversations.sqlite"),
            api_host=os.getenv("API_HOST") or "127.0.0.1",
            api_port=_parsed(os.environ, "API_PORT", int, "integer") or 8000,
        )
