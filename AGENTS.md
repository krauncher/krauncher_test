# krauncher_test — reference for coding agents

**This stand measures the quality of Krauncher's forecast. It does not execute
code.** Tasks are never submitted to a GPU, and the stand is not a way to run
them: it only checks the analyzer's pre-run forecast of a task against
reference values. For a task it asks the analyzer for two answers and compares
each with a reference value:

1. **Assay (passport)** — the work on the reference card, RTX PRO 6000
   Blackwell: `work.reference_sec` (whole task, seconds) and
   `requirements.min_vram_gb`.
2. **Ladder** — for every GPU, `compute_ratio`: the compute phase on that GPU
   over the compute phase on the reference card. Other phases (warmup,
   download, setup) do not depend on the GPU and are not in the ladder.

The analyzer reads the task's source code and the call arguments; it does not
execute them. The task code therefore does not need to run anywhere, its data
need not be downloaded, and no GPU time is spent. Do not add code that submits
or executes the task.

Client API reference: https://github.com/krauncher/krauncher/blob/main/AGENTS.md

---

## Deploy

```bash
git clone https://github.com/krauncher/krauncher_test.git
cd krauncher_test
./setup.sh                 # clones the krauncher client into client/, installs it into .venv
cp .env.example .env       # then set KRAUNCHER_API_KEY=cas_...
```

- Python 3.11+ and `git`.
- The API key comes from https://krauncher.com → Account → API Keys. Ask the
  user for it; never invent or search for one.
- The ladder needs **ladder access** on the account. It is not public: the user
  requests it from admin@krauncher.com with the account's email. Without it a
  run stops at the ladder with
  `KrauncherError: Ladder access is not enabled for this account.`
- `./setup.sh <ref>` pins the client to a branch, tag or commit.

## Run

```bash
.venv/bin/python run.py <task> [--service krauncher]   # <task> = module name in tasks/, without .py
```

Output: reference / service pairs on stdout, and
`results/<service>/<task>_<UTC time>.json` with the normalised forecast:
`version` (Krauncher: `calibration_id`, client commit), `fields` (the stand's
fields: workload_type, mode, precision, ..., reference_sec, min_vram_gb,
compute_ratio per GPU), `raw` (the service's answer as received: assay,
ladder) and `request` (task options and arguments).
**Never edit or delete a result file**: its UTC time and `calibration_id` prove
the forecast predates the runs it is compared with. Run again for a new one.
`results/` is not tracked by git; do not commit result files.

Errors:

| Message | Meaning |
|---|---|
| `Missing API key` | `.env` has no `KRAUNCHER_API_KEY` |
| `Ladder access is not enabled for this account.` | ladder not granted (see Deploy) |
| `AssayOutdated` | the analyzer was recalibrated between assay and ladder: run again |
| `Analyzer failed and CU estimation is unavailable` | the analyzer could not read the task |

---

## Services

A service under test is an adapter in `adapters/` implementing
`ServiceInterface` (`interface.py`): `name`, `capabilities()` (the stand
fields it answers) and `forecast(task)` (its pre-run forecast, normalised to
a `Forecast`, `models.py`). `adapters/__init__.py` registers adapters by name
for `run.py --service`. Plan: `doc/extension_plan.md`.

## Wrapping your own example as a task

Create `tasks/<name>.py`. `run.py` imports it and needs four names:

```python
"""<What the task is>.

FUNC is the task, OPTIONS the arguments of its @client.task decorator,
KWARGS the arguments of the call. REFERENCE holds the values the forecast
is checked against.
"""

OPTIONS = dict(
    timeout=3600,
    data_urls=["hf://datasets/stanfordnlp/imdb", "hf://models/google-bert/bert-base-uncased"],
    pip=["datasets"],
    dataset_size=84,          # MB of input data, when data_urls do not say it
    disk_gb=20,
)

KWARGS = dict(num_epochs=3, batch_size=16)

REFERENCE = dict(
    reference_sec=None,       # whole task, seconds, measured on RTX PRO 6000 Blackwell
    vram_gb=None,             # measured peak VRAM, GB
    compute_ratio={},         # gpu_id -> compute on that GPU / compute on the reference card
)


def train(num_epochs: int = 3, batch_size: int = 16):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification

    ds = load_dataset("/data/stanfordnlp__imdb")
    model = AutoModelForSequenceClassification.from_pretrained(
        "/data/google-bert__bert-base-uncased", num_labels=2).cuda()
    ...
    return {"loss": 0.1}


FUNC = train
```

### FUNC — the task function

- The code the user wants checked, as close to the original as possible: the
  analyzer reads the model, the dataset, batch size, epochs and steps off the
  source. A simplified rewrite is a different task.
- Defined at module top level, **self-contained**: imports and helper
  functions go inside it; no module-level globals or closures.
- Parameters the run depends on (epochs, batch size, sample count, sequence
  length, tokens to generate) should be function parameters: their values
  from `KWARGS` are sent to the analyzer.

### OPTIONS — the `@client.task` arguments

Parameters of the decorator (full table in the client's AGENTS.md):
`timeout`, `data_urls`, `data`, `pip`, `dataset_size`, `disk_gb`,
`stream_stderr`.

- **Do not set `vram_gb`.** It overrides the analyzer's VRAM requirement, and
  that requirement is one of the things under test.
- Do not set `gpu_name` / `gpu_arch`: the ladder covers every GPU.
- Keep the rest of the original example's decorator as it is.

### KWARGS — the call arguments

Keyword arguments of the call, `dict()` for the defaults. Only scalar values
(int, float, bool, str) reach the analyzer.

### REFERENCE — the values the forecast is checked against

| Key | Compared with | How to obtain |
|---|---|---|
| `reference_sec` | `assay.work.reference_sec` | whole task, seconds, on RTX PRO 6000 Blackwell |
| `vram_gb` | `assay.requirements.min_vram_gb` | peak VRAM of the run, GB |
| `compute_ratio` | ladder `compute_ratio` per `gpu_id` | compute on the GPU / compute on the reference card, where compute = whole task − download − setup |

Unknown values stay `None` (or `{}` for `compute_ratio`); the run prints `-`
for them. `gpu_id` keys are the ladder's own: run the task once and take them
from `ladder.rows[].gpu_id` in the result file. State in a comment where the
reference values come from (own measurement, hardware, date).

### DATA_SOURCES — optional

A task that names a registered data source (`OPTIONS["data"]`) lists it so
`run.py` registers it on the account when missing; the analyzer gets the
dataset size from it:

```python
DATA_SOURCES = [
    dict(name="food-101", urls=["http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"], size_gb=5.0),
]
```

### Check

```bash
.venv/bin/python -c "from tasks import <name> as t; print(t.FUNC.__name__, t.OPTIONS, t.KWARGS)"
.venv/bin/python run.py <name>
```

Then read `results/krauncher/<name>_<UTC time>.json`: `raw.assay.workload` shows what the
analyzer recognized (model, precision, knobs); a `null` in `knobs` for a value
the code sets means the analyzer did not read it — a finding to report, not a
reason to rewrite the task.

---

## Existing tasks

`tasks/` holds ten examples with reference values from Krauncher's own
calibration runs (the data the analyzer is calibrated on, so not an independent
check): `bert_imdb`, `qwen25_7b_lora_alpaca`, `qwen25_7b_gsm8k`,
`qwen25_7b_batched`, `qwen25_7b_long`, `phi3_inference`, `qwen15_inference`,
`bert_batch_inference`, `vit_batch_inference`, `resnet152_food101`. Use them as
templates.

Do not commit `.env`, `.venv/` or `client/`.
