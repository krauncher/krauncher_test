# Adding a sample

A sample is a task the stand asks services to forecast before it runs, with
the answer it expects. This document is for anyone who adds one: your own
workload, or a published benchmark. Services under test never execute the
sample on the stand; they forecast it from the code.

## What a sample is

Three files under the sample's name `<name>` (lowercase, `_` between words):

| File | What | Who writes it |
|---|---|---|
| `tasks/<name>.py` | the neutral task: a plain script, runnable on a machine with a GPU | you |
| `tasks/<name>.json` | the description: the expected forecast and the measurements | you |
| `variants/<service>/<name>.py` | the same task in the form one service takes | you, or anyone adding that service; by hand, a script or an LLM from the neutral task — then reviewed |

## 1. The neutral task — `tasks/<name>.py`

- One task function and the call arguments `KWARGS`; `python tasks/<name>.py`
  runs it:

  ```python
  """<What the task is> (<source, if published>).

  Neutral form of the task: a plain script, runnable on any machine with a GPU
  (`python tasks/<name>.py`); models and datasets by their public ids. KWARGS
  are the call arguments. The expected forecast and its sources are in
  <name>.json; the form a service takes is in variants/<service>/<name>.py.
  """

  KWARGS = {"batch_size": 32, "num_epochs": 1}


  def train(batch_size: int = 32, num_epochs: int = 1):
      import torch
      from datasets import load_dataset
      from transformers import AutoModelForSequenceClassification
      ...


  if __name__ == "__main__":
      print(train(**KWARGS))
  ```

- Models and datasets by public ids (Hugging Face `org/name`) or a public URL
  the script downloads when missing. No local paths, no credentials.
- Imports inside the function; scalar values (int, float, bool, str) in
  `KWARGS`, as the call that is measured.
- The script is the workload as measured: if a measurement used 3 epochs on
  a 25 000-sample split, the script does exactly that.
- A published benchmark: the benchmark's own script or its smallest faithful
  reduction to one file, with its settings (model, batch size, precision,
  sequence length / image size, number of iterations). Name the source and
  its version (repository, commit or release) in the docstring.

## 2. The description — `tasks/<name>.json`

```json
{
  "task": "<name>",
  "fields": {
    "workload_type": "ai_training",
    "mode": "training",
    "framework": "pytorch",
    "precision": "fp16",
    "params_billions": 0.11,
    "batch_size": 32,
    "epochs": 1,
    "dataset_samples": 25000,
    "seq_len": 256,
    "cpu_only": false
  },
  "sources": {
    "classification": "labelled from the task code, <date>, <who>"
  },
  "measurements": [
    {
      "id": "<source>-<year>",
      "independent": true,
      "source": "<who measured, where it is published: URL, table, version>",
      "date": "<when measured or published>",
      "hardware": "<GPUs, host, software stack: driver, CUDA, framework versions>",
      "reference_sec": 143,
      "reference_sec_note": "<how>",
      "min_vram_gb": 3.9,
      "min_vram_gb_note": "<how>",
      "anchor_gpu": "a100_sxm_80",
      "compute_ratio": {"a100_sxm_80": 1.0, "h100_sxm": 0.58},
      "compute_ratio_note": "<how>"
    }
  ]
}
```

### `fields` — what the task is

Read them from the code, as a careful engineer would — never copy a
service's answer into them (that would test the service against itself).
Leave out a field you cannot state.

| Field | Values / meaning |
|---|---|
| `workload_type` | `llm_inference`, `batch_inference`, `diffusion_inference`, `cv_training`, `ai_training`, `lora_training`, `object_detection`, `segmentation`, `graph_nn`, `gan`, `tts`, `rl`, `diffusion_training`, `3d_render`; `training` / `inference` when the model fits none of them (e.g. an MLP or an RNN); `unknown` when the file runs no task |
| `mode` | `training`, `inference`, or `null` for non-ML compute |
| `framework` | `pytorch`, `tensorflow`, `jax`, ... |
| `precision` | `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`, ... — what the code computes in (no setting = `fp32`) |
| `params_billions` | model size, billions of parameters (scored within ±25 %) |
| `batch_size`, `epochs`, `dataset_samples`, `seq_len` | as the code runs them; `batch_size` 1 for one prompt per `generate` call |
| `cpu_only` | `true` when the code does not use a GPU |

### `measurements` — what was measured

One set per source of measurements; a task may carry several (your runs, a
published table, a validator's runs). Every value keeps its note: how it was
obtained.

| Key | Meaning |
|---|---|
| `id` | unique per task, e.g. `mlperf-inference-v5.0`, `lambda-2025`, `own-2026-11` |
| `independent` | `true` unless the values come from the tested service's own calibration data |
| `source`, `date`, `hardware` | where the values are published or who measured, when, on what (GPUs, host, driver, CUDA, framework versions) |
| `reference_sec` | whole task, seconds, on the reference card (RTX PRO 6000 Blackwell). Leave out if the reference card was not measured |
| `min_vram_gb` | peak GPU memory of the run, GB |
| `compute_ratio` | per GPU: compute time on that GPU / compute time on the anchor GPU. Compute = whole task minus data download and environment setup |
| `anchor_gpu` | the GPU the ratios are relative to; leave out when it is the reference card. The stand divides each service's forecast by its forecast for the anchor |

GPU keys are the stand's `gpu_id`s (`rtx_6000_blackwell`, `h100_sxm`,
`a100_sxm_80`, `rtx_4090`, ...): run the sample once and take them from
`raw.ladder.rows[].gpu_id` in the result file.

### From a published benchmark

- Throughput (samples/s, tokens/s, images/s) for the same workload on
  several GPUs: `compute_ratio[g] = throughput[anchor] / throughput[g]`.
- Time per step or per run: `compute_ratio[g] = time[g] / time[anchor]`.
- Use only rows with the same settings (model, batch, precision, sequence
  length, software version) across GPUs; a different setting is a different
  sample.
- A single-GPU number per card: one row per card. Multi-GPU or multi-node
  results do not fit a single-GPU sample — leave them out.
- Cite the exact table: URL, version / round, row identifiers.

## 3. A service variant — `variants/<service>/<name>.py`

The task in the form that service takes. For Krauncher: the task function
with the `@client.task` arguments and the call arguments,

```python
OPTIONS = dict(timeout=3600, data_urls=["hf://datasets/<org>/<name>", "hf://models/<org>/<name>"])
KWARGS = dict(batch_size=32, num_epochs=1)
def train(batch_size: int = 32, num_epochs: int = 1): ...   # reads /data/<org>__<name>
FUNC = train
```

(see `AGENTS.md`, "Wrapping your own example as a task"). The variant must
be the same workload as the neutral task: same model, data, settings,
iterations. A variant made by an LLM or a script is reviewed against the
neutral task before it is committed.

## 4. Check and submit

```bash
.venv/bin/python -m py_compile tasks/<name>.py variants/krauncher/<name>.py
.venv/bin/python run.py <name>                 # one forecast per service
.venv/bin/python report.py --measurement <id>  # the sample in the report
```

- `run.py` prints the expected fields with `ok` / `DIFF` and the measured
  values against the forecast. A `DIFF` is a finding about the service, not
  a reason to change the label — re-check the label against the code, and
  keep it if the code says so.
- Do not edit or delete result files; they prove the forecast predates the
  measurement.
- Submit the three files with the measurement sources; add the sample to the
  table in `README.md`.
