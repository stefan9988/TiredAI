"""Start the app in its Docker container: wait for the catalog CSV, build the index if needed, then serve.

The index is rebuilt only when the CSV, the preprocessing or the embedding settings changed (see
build_index.py --if-changed). The container runs as the owner of the mounted data folder, so the
files it writes there and in .cache belong to whoever cloned the repo, and local runs can use them.

Usage (the image's command):
    python scripts/start.py
"""

import os
import subprocess
import sys
from pathlib import Path

from tiredai.config import ROOT, Settings
from tiredai.startup import wait_for_file


def run_as_owner_of(data_dir: Path, writable: list[Path]) -> None:
    """When started as root, switch to the user who owns data_dir, handing them the writable folders first."""
    if os.getuid() != 0:
        return
    owner = data_dir.stat()
    if owner.st_uid == 0:
        return
    for path in writable:
        path.mkdir(parents=True, exist_ok=True)
        if path.stat().st_uid == 0:  # created by Docker for a missing bind mount
            os.chown(path, owner.st_uid, owner.st_gid)
    os.setgroups([])
    os.setgid(owner.st_gid)
    os.setuid(owner.st_uid)


def main() -> None:
    settings = Settings.load()
    data_dir = settings.raw_data_path.parent
    run_as_owner_of(data_dir, [ROOT / ".cache"])

    if not settings.openrouter_api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Add it to .env (see .env.example) and start the container again.")

    wait_for_file(settings.raw_data_path, log=lambda line: print(line, flush=True))
    build = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_index.py"), "--if-changed"])
    if build.returncode:
        sys.exit(build.returncode)

    # The port the browser uses: the container always listens on API_PORT inside, PUBLIC_PORT is mapped to it.
    port = os.getenv("PUBLIC_PORT") or settings.api_port
    print(f"Starting the chat server: open http://localhost:{port} once startup is complete.", flush=True)
    os.execv(sys.executable, [sys.executable, str(ROOT / "scripts" / "serve.py")])


if __name__ == "__main__":
    main()
