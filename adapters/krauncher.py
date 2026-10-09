"""Krauncher: estimate-only submission through the krauncher client, assay v1
and the GPU ladder, read into the stand's fields."""

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from krauncher import KrauncherClient, KrauncherError

from interface import ServiceInterface
from models import Forecast, Task

CLIENT = Path(__file__).resolve().parent.parent / "client"


def _ensure_data_sources(client: KrauncherClient, sources: list[dict]) -> None:
    """Register the task's data sources on this account when missing."""
    for ds in sources:
        try:
            client.data_source(ds["name"]).info
        except KrauncherError:
            client.data_source(**ds)


def stand_fields(assay: dict, ladder: dict) -> dict:
    """Assay v1 + ladder/2 -> stand fields."""
    wl, model, knobs = assay["workload"], assay["workload"]["model"], assay["workload"]["knobs"]
    return {
        "workload_type": wl["type"],
        "mode": wl["mode"],
        "framework": wl["framework"],
        "precision": model["precision"],
        "model_name": model["name"],
        "params_billions": model["params_billions"],
        "batch_size": knobs["batch_size"],
        "epochs": knobs["epochs"],
        "dataset_samples": knobs["dataset_samples"],
        "seq_len": knobs["seq_len"],
        "cpu_only": assay["requirements"]["cpu_only"],
        "min_vram_gb": assay["requirements"]["min_vram_gb"],
        "reference_sec": assay["work"]["reference_sec"],
        "spread_factor": assay["work"]["spread"]["factor"],
        "compute_ratio": {r["gpu_id"]: r["compute_ratio"] for r in ladder["rows"]},
    }


class KrauncherAdapter(ServiceInterface):
    name = "krauncher"

    def capabilities(self) -> set[str]:
        return {"workload_type", "mode", "framework", "precision", "model_name",
                "params_billions", "batch_size", "epochs", "dataset_samples", "seq_len",
                "cpu_only", "min_vram_gb", "reference_sec", "spread_factor", "compute_ratio"}

    async def forecast(self, task: Task) -> Forecast:
        mod = task.module
        client = KrauncherClient(estimate_only=True)
        _ensure_data_sources(client, getattr(mod, "DATA_SOURCES", []))
        handle = await client.task(**mod.OPTIONS)(mod.FUNC)(**mod.KWARGS)
        t0 = time.monotonic()
        assay = await client.analyzer().assay(handle.classification.analyzer_job_id)
        ladder = await client.ladder(assay)
        t_assay_ladder = time.monotonic() - t0
        return Forecast(
            service=self.name,
            task=task.name,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            version={
                "calibration_id": ladder["meta"]["calibration_id"],
                "client_commit": subprocess.run(
                    ["git", "-C", str(CLIENT), "rev-parse", "HEAD"],
                    capture_output=True, text=True).stdout.strip(),
            },
            fields=stand_fields(assay, ladder),
            raw={"assay": assay, "ladder": ladder},
            request={"options": mod.OPTIONS, "kwargs": mod.KWARGS},
            timing={
                # code analysis as the client measured it (submit -> classification)
                "analysis_sec": round(handle.classification.analyzer_time or 0.0, 2),
                "assay_ladder_sec": round(t_assay_ladder, 2),
            },
        )
