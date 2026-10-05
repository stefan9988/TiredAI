"""Search the tire-size pages of the agent benchmark's vehicles once and freeze them.

The agent benchmark answers its vehicle lookups (find_vehicle_tire_sizes) from
benchmarks/vehicle_pages.yaml instead of searching the web, so runs are repeatable and free. This
script runs the live lookup (OpenRouter web search on VEHICLE_LOOKUP_SITES, about $0.007 each) for
every vehicle in benchmarks/agent_cases.yaml that has no captured pages yet, or for all of them when
the sites or the result count changed. Read the new pages before committing them.

Usage:
    uv run python scripts/capture_vehicle_pages.py [--all] [--check]
"""

import argparse
import sys

import httpx

from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.cases import load_cases, vehicles
from tiredai.config import Settings
from tiredai.vehicles import FrozenPages, SiteSearch, VehicleLookupError

CASES = ex.BENCHMARKS_DIR / "agent_cases.yaml"
CAPTURED = ex.BENCHMARKS_DIR / "vehicle_pages.yaml"
HEADER = ("Tire-size pages of the vehicles in agent_cases.yaml, as the vehicle lookup found them; written by\n"
          "# scripts/capture_vehicle_pages.py. Don't edit by hand; capture again after adding a vehicle.")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="capture every vehicle again")
    parser.add_argument("--check", action="store_true", help="only list the vehicles that need capturing")
    args = parser.parse_args()
    settings = Settings.load()
    config = settings.vehicle_lookup

    wanted = vehicles(load_cases(CASES))
    captured = FrozenPages.load(CAPTURED)
    same_search = captured.sites == config.sites and captured.max_results == config.max_results
    stale = [v for v in wanted if args.all or not same_search or captured.row(v) is None]
    if args.check:
        print("\n".join(f"needs capturing: {v}" for v in stale) or "Every vehicle is captured.")
        sys.exit(1 if stale else 0)
    if not stale:
        print("Every vehicle is captured with the current sites.")
        return
    if not settings.openrouter_api_key:
        sys.exit("Capturing searches the web through OpenRouter: set OPENROUTER_API_KEY in .env")

    search = SiteSearch(settings.openrouter_api_key, config.model or settings.llm.model, config.sites,
                        max_results=config.max_results, client=httpx.Client(timeout=60))  # fmt: skip
    fresh = {}
    for vehicle in stale:
        try:
            found = search.pages(vehicle)
        except VehicleLookupError as exc:
            sys.exit(f"Looking up the {vehicle} failed: {exc}")
        print(f"{vehicle}: {len(found.pages)} pages ({', '.join(p['site'] for p in found.pages) or 'none'})")
        fresh[str(vehicle).lower()] = {
            "vehicle": {"year": vehicle.year, "make": vehicle.make, "model": vehicle.model},
            "captured_at": ex.timestamp(),
            "pages": found.pages,
        }
    # In the order of the cases; vehicles no case uses any more are dropped.
    rows = [fresh.get(str(vehicle).lower()) or captured.row(vehicle) for vehicle in wanted]
    FrozenPages(config.sites, config.max_results, rows).write(CAPTURED, HEADER)
    print(f"Saved {CAPTURED.relative_to(ex.BENCHMARKS_DIR.parent)}: {len(fresh)} captured, {len(rows)} vehicles.")


if __name__ == "__main__":
    main()
