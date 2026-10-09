"""Normalised data of the stand: a task, and a service's forecast of it.

Every service adapter reads its own answer into a Forecast; the stand compares
Forecast.fields with the task's expected fields in one way for all services.
"""

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


TASKS = Path(__file__).parent / "tasks"


@dataclass
class Task:
    """A stand task — labelled data: its neutral code (tasks/<name>.py, a
    plain script runnable on a GPU machine) and its description
    (tasks/<name>.json): `fields`, the expected forecast in the stand's
    fields (classification / assay), `sources`, where they come from, and
    `measurements`, sets of measured values (reference_sec, min_vram_gb,
    compute_ratio), each with its id, source, date and whether it is
    independent of the service's calibration. The form a service takes is a
    prepared file, variants/<service>/<name>.py."""

    name: str
    code: Path
    expected: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    measurements: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, name: str) -> "Task":
        desc = json.loads((TASKS / f"{name}.json").read_text())
        return cls(name=name, code=TASKS / f"{name}.py", expected=desc["fields"],
                   sources=desc.get("sources", {}), measurements=desc.get("measurements", []))

    def measured(self, mid: str | None = None) -> dict[str, Any]:
        """The measurement set `mid` (the first one when None); {} if absent."""
        for m in self.measurements:
            if mid is None or m["id"] == mid:
                return m
        return {}

    def variant(self, service: str) -> ModuleType:
        """The task in the form `service` takes (variants/<service>/<name>.py)."""
        return importlib.import_module(f"variants.{service}.{self.name}")


@dataclass
class Forecast:
    """A service's pre-run forecast of a task, normalised to the stand's fields.

    fields — stand fields the service answered (see ServiceInterface.capabilities);
    compute_ratio is {gpu_id: compute on that GPU / compute on the reference card}.
    raw — the service's answer as received, kept for audit.
    version — what identifies the forecasting model (Krauncher: calibration_id
    and client commit).
    """

    service: str
    task: str
    created_utc: str
    version: dict[str, Any]
    fields: dict[str, Any]
    raw: dict[str, Any]
    request: dict[str, Any] = field(default_factory=dict)
    # Wall-clock seconds. forecast_sec — measured by the stand around the
    # service call (ServiceInterface.timed_forecast), the same way for every
    # service; other keys — the service's own parts, where it reports them.
    timing: dict[str, float] = field(default_factory=dict)
