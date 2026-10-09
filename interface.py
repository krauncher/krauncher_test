"""ServiceInterface — abstract base class for the services the stand tests.

The same scheme as the GPU-provider adapters of Krauncher
(cas-provider ProviderInterface): each adapter hides one service's request
format and answer behind these methods, and the stand handles every service
alike. See doc/extension_plan.md.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from models import Forecast, Task

RESULTS = Path(__file__).parent / "results"


class ServiceInterface(ABC):
    """One service under test: how to ask it about a stand task, and how to
    read its answer into the stand's fields."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique service name ('krauncher', ...)."""

    @abstractmethod
    def capabilities(self) -> set[str]:
        """Stand fields this service answers. The report scores only these;
        a field outside them is 'not provided', not 'wrong'."""

    @abstractmethod
    async def forecast(self, task: Task) -> Forecast:
        """The service's pre-run forecast of `task`, normalised."""

    def save(self, fc: Forecast) -> Path:
        """Write the forecast to results/<service>/<task>_<UTC>.json. A result
        file is never edited afterwards: its time and version prove the
        forecast predates any measurement it is compared with."""
        stamp = datetime.fromisoformat(fc.created_utc).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS / fc.service / f"{fc.task}_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fc.__dict__, indent=2, ensure_ascii=False) + "\n")
        return out
