# Extension plan: a service-neutral test stand

Agreed 2026-10-09. The stand validates Krauncher's forecasts and compares
them with competing products, so the Krauncher analyzer is one interface
among several. The validation project is a project of its own; towards
third parties (e.g. an independent tester) it is a tool offered for
independent testing.

## What exists

- `tasks/<name>.py`: `FUNC` (the task), `OPTIONS` (arguments of
  `@client.task`), `KWARGS` (call arguments), `REFERENCE` (`reference_sec`,
  `vram_gb`, `compute_ratio` per GPU).
- `run.py`: estimate-only submission → assay v1 + GPU ladder →
  `results/<task>_<UTC>.json` with `calibration_id` and client commit, never
  edited afterwards (the forecast predates any measurement); compares the
  passport and the ladder with `REFERENCE`.
- 10 tasks. Their `REFERENCE` comes from Krauncher's calibration runs: not an
  independent check (stated in each task).

## Gaps

- The task format is Krauncher's: `FUNC` reads `/data/...` (the `hf://`
  bridge), data and model come through `OPTIONS["data_urls"]`; it does not
  run locally without Krauncher.
- No expected classification / assay fields (workload_type, mode, cpu_only,
  precision, model, ...).
- References have no per-value source and date; no independent references.
- No report across tasks and levels; one console print per run.

## Design

The same scheme as the GPU-provider adapters in CaS
(`cas-provider/src/provider/`: `interface.py` `ProviderInterface(ABC)`,
`models.py` normalised `HostOffer` / `GpuCandidate`, `adapters/<provider>.py`,
`registry.py`). Interaction formats and answers differ between services and
are hidden behind the abstract class.

| GPU providers | Stand |
|---|---|
| `ProviderInterface(ABC)`: abstract `name`, `sync_offers`, `health_check`, `terminate_host`; defaults such as `provision_host` | `ServiceInterface(ABC)`: abstract `name`, `capabilities()`, `forecast(task)`; defaults such as writing the result file and the time stamp |
| `models.py`: `HostOffer`, `GpuCandidate` | `models.py`: `Task` (neutral task), `Forecast` (normalised forecast + raw answer) |
| `adapters/<provider>.py`, shared `cloud_base.py` | `adapters/krauncher.py` first, then one module per competitor; each submits a prepared variant (`variants/<service>/`) and normalises the answer |
| `registry.py` | adapter chosen by name: `run.py --service krauncher`; `report.py` compares services on the same tasks |

```python
class ServiceInterface(ABC):
    """One service under test: how to ask it about a stand task, and how to
    read its answer into the stand's fields."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def capabilities(self) -> set[str]:
        """Stand fields this service can answer (workload_type, reference_sec,
        min_vram_gb, compute_ratio, price_usd, ...). The report scores only
        these; the rest is 'not provided', not 'wrong'."""

    @abstractmethod
    async def forecast(self, task: "Task") -> "Forecast":
        """The service's pre-run forecast of `task`, normalised."""


@dataclass
class Forecast:
    service: str
    created_utc: str
    version: str               # Krauncher: calibration_id + client commit
    fields: dict[str, Any]     # stand fields; compute_ratio: {gpu_id: float}
    raw: dict                  # the answer as received, kept for audit
```

- `Task`: labelled data — the task's code and its description
  `tasks/<name>.json`: `fields`, the expected forecast in the stand's fields
  (classification / assay fields read from the code, `reference_sec`,
  `min_vram_gb`, `compute_ratio`), and `sources` per value.
- Service variants are prepared files, not built by adapter code:
  `variants/<service>/<task>.py` holds the task in the form that service
  takes (Krauncher: `FUNC` / `OPTIONS` with the `/data` bridge, the installed
  decorator form). How a variant is made — by hand, by a script, or by an LLM
  from the neutral task — does not matter to the stand; it is reviewed and
  committed, so what was sent to a service is fixed and auditable. The
  adapter only submits a prepared variant and normalises the answer.
- Result files keep the normalised `Forecast` and the raw answer; never
  edited.
- The report compares `Forecast.fields` with `EXPECTED` / `REFERENCE` over
  the service's `capabilities()`.

## Stand levels reported

1. Classification / assay fields (expected analyzer emission).
2. Reference-card time.
3. Per-card ladder: ratio, card order, cheapest-card choice; regret in
   $/task from a wrong ranking.
4. Issued spread: share of runs inside the interval.

## Steps

1. `interface.py`, `models.py`, `adapters/krauncher.py` reproducing today's
   `run.py` (assay + ladder → stand fields); `run.py` picks the adapter by
   name; current tasks are read as they are. *Done 2026-10-09.*
2. The description file per task (`tasks/<name>.json`, `REFERENCE` moved
   out of the code), scored field by field (same fields as the CaS
   classification stand, `cas-analyzer/research/classification_eval.py`).
   *Done 2026-10-09.*
3. `report.py` over the normalised results, per service. *Done 2026-10-09;
   regret in $/task waits for prices.*
4. Neutral task form (a plain script runnable locally on a GPU, public data
   ids) in `tasks/`; today's Krauncher-form tasks move to
   `variants/krauncher/`. Variants for other services are prepared files
   (by hand, script or LLM), reviewed and committed.
5. Several measurements per value with `source` / `date` / host in the
   description; new tasks outside the calibration corpus.

Open: task sources for the independent corpus; budget for measurement runs.
