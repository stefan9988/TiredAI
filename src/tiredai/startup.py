"""Startup steps for the Docker container (scripts/start.py)."""

import time
from collections.abc import Callable
from pathlib import Path


def wait_for_file(
    path: Path,
    *,
    settle_seconds: float = 2.0,
    poll_seconds: float = 1.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> None:
    """Return once `path` exists and hasn't changed for `settle_seconds`, so a copy in progress isn't read."""
    announced = False
    while True:
        if path.is_file() and clock() - path.stat().st_mtime >= settle_seconds:
            return
        if not announced and not path.is_file():
            log(f"Waiting for {path.name}: copy the tire catalog CSV into the data folder to continue.")
            announced = True
        sleep(poll_seconds)
