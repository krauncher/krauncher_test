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
    """A stand task: its code (tasks/<name>.py, today in the Krauncher form —
    FUNC, OPTIONS, KWARGS, DATA_SOURCES; see README) and its description
    (tasks/<name>.json): `fields`, the expected forecast in the stand's
    fields, and `sources`, where each expected value comes from."""

    name: str
    module: ModuleType
    expected: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str) -> "Task":
        desc = json.loads((TASKS / f"{name}.json").read_text())
        return cls(name=name, module=importlib.import_module(f"tasks.{name}"),
                   expected=desc["fields"], sources=desc.get("sources", {}))


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
