"""Frozen forecast for one task from one service, written to results/.

Nothing runs on a GPU. The service adapter (adapters/) asks its service for
the pre-run forecast of the task and normalises it to the stand's fields; the
forecast is written to results/<service>/<task>_<UTC time>.json and never
edited afterwards: its timestamp and version prove it predates any measured
run. The passport (reference_sec, min_vram_gb) and the ladder (compute_ratio
per GPU) are printed as pairs reference / service.

    python run.py [task] [--service krauncher]   # task = module in tasks/, default bert_imdb
"""

import argparse
import asyncio
from pathlib import Path

from adapters import ADAPTERS
from models import Forecast, Task

ROOT = Path(__file__).parent


def compare(fc: Forecast, task: Task) -> dict:
    """Reference / service pairs for the passport and for every GPU of the ladder."""
    ref, got = task.reference, fc.fields
    return {
        "passport": {
            "reference_sec": [ref.get("reference_sec"), got.get("reference_sec")],
            "min_vram_gb": [ref.get("vram_gb"), got.get("min_vram_gb")],
        },
        "compute_ratio": {g: [r, (got.get("compute_ratio") or {}).get(g)]
                          for g, r in ref.get("compute_ratio", {}).items()},
    }


async def main(name: str, service: str) -> None:
    task = Task.load(name)
    adapter = ADAPTERS[service]
    fc = await adapter.forecast(task)
    out = adapter.save(fc)
    cmp = compare(fc, task)
    print(f"{out.relative_to(ROOT)}: {service} {fc.version}")
    print(f"{'':<22}{'reference':>10}{service:>10}")
    for key, (r, a) in cmp["passport"].items():
        print(f"{key:<22}{'-' if r is None else r:>10}{'-' if a is None else a:>10}")
    print("compute_ratio")
    for g, (r, a) in cmp["compute_ratio"].items():
        print(f"  {g:<20}{r:>10.3f}{'-' if a is None else f'{a:.3f}':>10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", default="bert_imdb")
    ap.add_argument("--service", default="krauncher", choices=sorted(ADAPTERS))
    args = ap.parse_args()
    asyncio.run(main(args.task, args.service))
