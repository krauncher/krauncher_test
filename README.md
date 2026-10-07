# krauncher_test

Test stand for independent validation of krauncher forecasts. For each task
it records, before anything runs, the analyzer's assay v1 (the passport: the
work on the reference card) and the GPU ladder (each GPU's compute coefficient
against the reference card), stamped with the UTC time and the analyzer
`calibration_id`, and compares both with the task's reference values.

**The stand measures the quality of Krauncher's forecast; it does not execute
code.** Tasks are never submitted to a GPU: the analyzer reads their source
code and call arguments, no data is downloaded and no GPU time is spent.

## How to run the test

### 1. Requirements

- Python 3.11 or newer, `git`.

### 2. Download

    git clone https://github.com/krauncher/krauncher_test.git
    cd krauncher_test
    ./setup.sh

`setup.sh` clones the current krauncher client from git into `client/` and
installs it into a local `.venv`. `./setup.sh <ref>` pins the client to a
branch, tag or commit.

### 3. Get a Krauncher API key

1. Sign up at https://krauncher.com.
2. Create a key: Account → API Keys.
3. Put it into `.env`:

       cp .env.example .env
       # edit .env: KRAUNCHER_API_KEY=cas_...

### 4. Request access to the GPU ladder

The assay is available to every account. The GPU ladder is not publicly
available: access is granted per account on personal request. Write to
admin@krauncher.com with the email of your Krauncher account.

Without ladder access a run stops with
`Ladder access is not enabled for this account.`

### 5. Run

One task:

    .venv/bin/python run.py bert_imdb

All tasks:

    for t in tasks/[a-z]*.py; do .venv/bin/python run.py "$(basename "$t" .py)"; done

`resnet152_food101` names the public Food-101 dataset through a data source
`food-101`, from which the analyzer takes the dataset size; `run.py` registers
it on your account if it is missing.

### 6. Read the result

Each run prints reference / analyzer pairs for the passport (`reference_sec`,
`min_vram_gb`) and for every GPU of the ladder (`compute_ratio`), and writes
`results/<task>_<UTC time>.json`: task options and arguments, client commit,
`calibration_id`, assay, ladder and the reference / analyzer pairs. Result
files are not edited after they are written: the UTC time and
`calibration_id` show that the forecast was made before any run it is
compared with. `results/` is not tracked by git: the files stay with whoever
runs the stand; keep them, or hand them over together with the measurements.

## Tasks

| Module | Task |
|---|---|
| `bert_imdb` | BERT fine-tuning on IMDB |
| `qwen25_7b_lora_alpaca` | Qwen2.5-7B LoRA fine-tuning on Alpaca |
| `qwen25_7b_gsm8k` | Qwen2.5-7B-Instruct inference on GSM8K |
| `qwen25_7b_batched` | Qwen2.5-7B-Instruct batched inference |
| `qwen25_7b_long` | Qwen2.5-7B-Instruct long generation |
| `phi3_inference` | Phi-3-mini inference on GSM8K |
| `qwen15_inference` | Qwen2.5-1.5B-Instruct inference on GSM8K |
| `bert_batch_inference` | BERT batch inference |
| `vit_batch_inference` | ViT batch inference |
| `resnet152_food101` | ResNet-152 training on Food-101 |

## Adding a task

A module in `tasks/` defines the task function (`FUNC`), the arguments of
its `@client.task` decorator (`OPTIONS`), of the call (`KWARGS`) and the
reference values (`REFERENCE`: `reference_sec`, `vram_gb`, `compute_ratio` per GPU).
A task that reads a registered data source (`OPTIONS["data"]`) lists it in
`DATA_SOURCES`; `run.py` registers it on the account when missing.
