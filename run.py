"""Frozen forecast for one task from one service, written to results/.

Nothing runs on a GPU. The service adapter (adapters/) asks its service for
the pre-run forecast of the task and normalises it to the stand's fields; the
forecast is written to results/<service>/<task>_<UTC time>.json and never
edited afterwards: its timestamp and version prove it predates any measured
run. The task's description (tasks/<task>.json) holds the expected
forecast: classification / assay fields are printed expected / service with
ok / DIFF, the passport (reference_sec, min_vram_gb) and the ladder
(compute_ratio per GPU) as pairs reference / service.

    python run.py [task] [--service krauncher]   # task = module in tasks/, default bert_imdb
"""

import argparse
import asyncio
from pathlib import Path

from adapters import ADAPTERS
from models import Forecast, Task

ROOT = Path(__file__).parent


# Fields compared as values (pass / fail); the rest — measured times and the
# ladder — are shown as pairs, their error is the report's business.
PARAMS_TOL = 0.25                         # params_billions within +/-25 %
_NORM = {"precision": lambda v: v or "fp32"}  # no precision in the code is fp32


def matches(field: str, want, got) -> bool:
    if field == "params_billions":
        return got is not None and abs(got - want) <= PARAMS_TOL * want
    n = _NORM.get(field, lambda v: v)
    return n(want) == n(got)


def compare(fc: Forecast, task: Task, caps: set[str]) -> dict:
    """Expected / service pairs over the expected fields the service answers."""
    exp, got = task.expected, fc.fields
    pairs = {f: [exp[f], got.get(f)] for f in exp
             if f in caps and f not in ("reference_sec", "min_vram_gb", "compute_ratio")}
    return {
        "fields": {f: [w, g, matches(f, w, g)] for f, (w, g) in pairs.items()},
        "passport": {f: [exp.get(f), got.get(f)] for f in ("reference_sec", "min_vram_gb") if f in caps},
        "compute_ratio": {g: [r, (got.get("compute_ratio") or {}).get(g)]
                          for g, r in exp.get("compute_ratio", {}).items()},
    }


async def main(name: str, service: str) -> None:
    task = Task.load(name)
    adapter = ADAPTERS[service]
    fc = await adapter.forecast(task)
    out = adapter.save(fc)
    cmp = compare(fc, task, adapter.capabilities())
    print(f"{out.relative_to(ROOT)}: {service} {fc.version}")
    print(f"{'':<22}{'expected':>16}{service:>16}")
    for key, (w, g, ok) in cmp["fields"].items():
        print(f"{key:<22}{str(w):>16}{str(g):>16}  {'ok' if ok else 'DIFF'}")
    print(f"{'':<22}{'reference':>16}{service:>16}")
    for key, (r, a) in cmp["passport"].items():
        print(f"{key:<22}{'-' if r is None else r:>16}{'-' if a is None else a:>16}")
    print("compute_ratio")
    for g, (r, a) in cmp["compute_ratio"].items():
        print(f"  {g:<20}{r:>16.3f}{'-' if a is None else f'{a:.3f}':>16}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", default="bert_imdb")
    ap.add_argument("--service", default="krauncher", choices=sorted(ADAPTERS))
    args = ap.parse_args()
    asyncio.run(main(args.task, args.service))
