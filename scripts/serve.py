"""Run the tire assistant API (local demo, no auth). Host and port come from .env (API_HOST, API_PORT).

Interactive docs are served at /docs. The API opens the local vector store, so stop it before
running scripts/build_index.py.

Usage:
    uv run python scripts/serve.py [--reload]
"""

import argparse

import uvicorn

from tiredai.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reload", action="store_true", help="restart when source files change")
    args = parser.parse_args()
    settings = Settings.load()

    uvicorn.run(
        "tiredai.api:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
