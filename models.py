"""Normalised data of the stand: a task, and a service's forecast of it.

Every service adapter reads its own answer into a Forecast; the stand compares
Forecast.fields with the task's REFERENCE (and, later, EXPECTED) in one way
for all services.
"""

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any


@dataclass
class Task:
    """A stand task. Today a task module is in the Krauncher form (FUNC,
    OPTIONS, KWARGS, REFERENCE, DATA_SOURCES; see README)."""

    name: str
    module: ModuleType
    reference: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str) -> "Task":
        mod = importlib.import_module(f"tasks.{name}")
        return cls(name=name, module=mod, reference=mod.REFERENCE,
                   expected=getattr(mod, "EXPECTED", {}))


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
