"""Project settings, read from environment variables and an optional .env file (see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: str) -> Path:
    # Relative paths are resolved against the project root, so scripts work from any directory.
    path = Path(os.getenv(name) or default)
    return path if path.is_absolute() else ROOT / path


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
    openrouter_api_key: str | None

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
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        )
