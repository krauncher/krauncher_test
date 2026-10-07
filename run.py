"""Frozen forecast for one task: assay v1 and GPU ladder, written to results/.

Nothing runs on a GPU. The client analyzes the exact source it would submit,
fetches the assay of that analysis and the ladder for the assay, and writes
both to results/<task>_<UTC time>.json. A result file is never edited
afterwards: its timestamp and calibration_id prove the forecast predates
any measured run. Both are compared with the task's REFERENCE: the passport
(reference_sec, min_vram_gb) and the ladder (compute_ratio per GPU), each as a
pair reference / analyzer.

    python run.py [task]        # task = module in tasks/, default bert_imdb
"""

import asyncio
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from krauncher import KrauncherClient, KrauncherError

ROOT = Path(__file__).parent


def ensure_data_sources(client: KrauncherClient, sources: list[dict]) -> None:
    """Register the task's data sources on this account when missing."""
    for ds in sources:
        try:
            client.data_source(ds["name"]).info
        except KrauncherError:
            client.data_source(**ds)


def compare(assay: dict, ladder: dict, ref: dict) -> dict:
    """Reference / analyzer pairs for the passport and for every GPU of the ladder."""
    got = {r["gpu_id"]: r["compute_ratio"] for r in ladder["rows"]}
    return {
        "passport": {
            "reference_sec": [ref["reference_sec"], assay["work"]["reference_sec"]],
            "min_vram_gb": [ref["vram_gb"], assay["requirements"]["min_vram_gb"]],
        },
        "compute_ratio": {g: [r, got.get(g)] for g, r in ref["compute_ratio"].items()},
    }


async def main(name: str) -> None:
    task = importlib.import_module(f"tasks.{name}")

    client = KrauncherClient(estimate_only=True)
    ensure_data_sources(client, getattr(task, "DATA_SOURCES", []))
    handle = await client.task(**task.OPTIONS)(task.FUNC)(**task.KWARGS)
    c = handle.classification

    assay = await client.analyzer().assay(c.analyzer_job_id)
    ladder = await client.ladder(assay)

    now = datetime.now(timezone.utc)
    record = {
        "task": name,
        "created_utc": now.isoformat(timespec="seconds"),
        "calibration_id": ladder["meta"]["calibration_id"],
        "client_commit": subprocess.run(
            ["git", "-C", str(ROOT / "client"), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "task_options": task.OPTIONS,
        "task_kwargs": task.KWARGS,
        "assay": assay,
        "ladder": ladder,
        "comparison": compare(assay, ladder, task.REFERENCE),
    }
    out = ROOT / "results" / f"{name}_{now:%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    cmp = record["comparison"]
    print(f"{out.relative_to(ROOT)}: calibration {record['calibration_id']}")
    print(f"{'':<22}{'reference':>10}{'analyzer':>10}")
    for name, (r, a) in cmp["passport"].items():
        print(f"{name:<22}{'-' if r is None else r:>10}{a:>10}")
    print("compute_ratio")
    for g, (r, a) in cmp["compute_ratio"].items():
        print(f"  {g:<20}{r:>10.3f}{'-' if a is None else f'{a:.3f}':>10}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "bert_imdb"))
