"""Frozen forecast for one task from one service, written to results/.

Nothing runs on a GPU. The service adapter (adapters/) asks its service for
the pre-run forecast of the task and normalises it to the stand's fields; the
forecast is written to results/<service>/<task>_<UTC time>.json and never
edited afterwards: its timestamp and version prove it predates any measured
run. The task's description (tasks/<source>/<task>.json) holds the expected
forecast: classification / assay fields are printed expected / service with
ok / DIFF, the passport (reference_sec, min_vram_gb) and the ladder
(compute_ratio per GPU) as pairs reference / service.

    python run.py [task] [--service krauncher]   # task = <source>/<task>, e.g. krauncher_tutorials/bert_imdb
"""

import argparse
import asyncio
from pathlib import Path

from adapters import ADAPTERS
from models import Forecast, Task
from scoring import MEASURED, ladder_pairs, matches

ROOT = Path(__file__).parent


def compare(fc: Forecast, task: Task, caps: set[str], mid: str | None = None) -> dict:
    """Expected / service pairs over the expected fields the service answers,
    and reference / service pairs for the measurement set `mid` (first if None)."""
    exp, got, meas = task.expected, fc.fields, task.measured(mid)
    pairs = {f: [exp[f], got.get(f)] for f in exp
             if f in caps and f not in MEASURED}
    return {
        "fields": {f: [w, g, matches(f, w, g)] for f, (w, g) in pairs.items()},
        "passport": {f: [meas.get(f), got.get(f)] for f in ("reference_sec", "min_vram_gb") if f in caps},
        "compute_ratio": {g: list(v) for g, v in ladder_pairs(meas, got.get("compute_ratio")).items()},
    }


async def main(name: str, service: str, mid: str | None) -> None:
    task = Task.load(name)
    adapter = ADAPTERS[service]
    fc = await adapter.timed_forecast(task)
    out = adapter.save(fc)
    cmp = compare(fc, task, adapter.capabilities(), mid)
    print(f"{out.relative_to(ROOT)}: {service} {fc.version}")
    print("timing, s: " + ", ".join(f"{k} {v}" for k, v in fc.timing.items()))
    print(f"{'':<22}{'expected':>16}{service:>16}")
    for key, (w, g, ok) in cmp["fields"].items():
        print(f"{key:<22}{str(w):>16}{str(g):>16}  {'ok' if ok else 'DIFF'}")
    print(f"{'measurement: ' + (task.measured(mid).get('id') or '-'):<22}{'reference':>16}{service:>16}")
    for key, (r, a) in cmp["passport"].items():
        print(f"{key:<22}{'-' if r is None else r:>16}{'-' if a is None else a:>16}")
    anchor = task.measured(mid).get("anchor_gpu")
    print("compute_ratio" + (f" (relative to {anchor})" if anchor else ""))
    for g, (r, a) in cmp["compute_ratio"].items():
        print(f"  {g:<20}{r:>16.3f}{'-' if a is None else f'{a:.3f}':>16}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", default="krauncher_tutorials/bert_imdb")
    ap.add_argument("--service", default="krauncher", choices=sorted(ADAPTERS))
    ap.add_argument("--measurement", help="measurement set id of the description (default: the first)")
    args = ap.parse_args()
    asyncio.run(main(args.task, args.service, args.measurement))
