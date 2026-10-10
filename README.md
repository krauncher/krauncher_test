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

    .venv/bin/python run.py krauncher_tutorials/bert_imdb

`--service <name>` picks the service under test (default `krauncher`; the
services are adapters in `adapters/`, see `doc/extension_plan.md`).

All tasks:

    for t in tasks/*/*.json; do t=${t#tasks/}; .venv/bin/python run.py "${t%.json}"; done

`resnet152_food101` names the public Food-101 dataset through a data source
`food-101`, from which the analyzer takes the dataset size; `run.py` registers
it on your account if it is missing.

### 6. Read the result

Each run prints reference / service pairs for the passport (`reference_sec`,
`min_vram_gb`) and for every GPU of the ladder (`compute_ratio`), and writes
`results/<service>/<task>_<UTC time>.json`: the forecast normalised to the
stand's fields, the service's answer as received (for Krauncher: assay and
ladder), its version (`calibration_id`, client commit) and the task options
and arguments. Result
files are not edited after they are written: the UTC time and
`calibration_id` show that the forecast was made before any run it is
compared with. `results/` is not tracked by git: the files stay with whoever
runs the stand; keep them, or hand them over together with the measurements.

### 7. Report

    .venv/bin/python report.py [--service krauncher] [--version <calibration_id>] [--out DIR]

Writes `report.json` (machine-readable), `report.md` and `report.html` (for
viewing: every value marked accurate / miss / problem) to `DIR` (default
`results/report/`), over the latest result of every task, per service:
forecast time
(request to normalised forecast, measured by the stand), classification /
assay fields (match per field), reference-card time (centre, typical
deviation, inside the issued spread), VRAM (centre, forecasts below the
measured peak) and the ladder (error per GPU, rank correlation of the card
order, time regret of the GPU picked as fastest).

## Tasks

Krauncher client tutorials (`tasks/krauncher_tutorials/`, measured on Krauncher's calibration runs — not independent):

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

Published benchmarks (`tasks/<source>/`, independent measurements):

| Task | Source |
|---|---|
| `lambdalabs/resnet50_amp_bs1280` | ResNet-50 training, AMP, batch 1280, synthetic data; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/resnet50_amp_bs928` | ResNet-50 training, AMP, batch 928, synthetic data; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/resnet50_amp_bs448` | ResNet-50 training, AMP, batch 448, synthetic data; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/bert_base_squad_amp_bs320` | BERT-base fine-tuning on SQuAD v1.1, mixed precision, batch 320, 101 steps; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/bert_base_squad_amp_bs192` | BERT-base fine-tuning on SQuAD v1.1, mixed precision, batch 192, 101 steps; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/bert_base_squad_amp_bs96` | BERT-base fine-tuning on SQuAD v1.1, mixed precision, batch 96, 101 steps; Lambda GPU benchmark (RTX 4090, RTX 3090, 24 GB) |
| `lambdalabs/bert_large_squad_amp_bs112` | BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 112, 101 steps; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/bert_large_squad_amp_bs64` | BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 64, 101 steps; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/ncf_ml20m_bs10m` | NCF (NeuMF) training on MovieLens-20M, batch 10 000 000, 2 epochs, fp32; Lambda GPU benchmark (A100 / H100 80 GB, RTX 6000 Ada, RTX A6000, Quadro RTX 8000) |
| `lambdalabs/ncf_ml20m_bs4278184` | NCF (NeuMF) training on MovieLens-20M, batch 4 278 184, 2 epochs, fp32; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/waveglow_ljs625_bs48` | WaveGlow training on LJSpeech (625 clips, 8000-sample segments), batch 48, 2 epochs, fp32; model and data code verbatim from NVIDIA DeepLearningExamples (BSD 3-Clause); Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/waveglow_ljs625_bs32` | the same WaveGlow training, batch 32; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000) |
| `lambdalabs/waveglow_ljs625_bs18` | the same WaveGlow training, batch 18; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/tacotron2_ljs625_bs256` | Tacotron 2 training on LJSpeech (625 clips), batch 256, 3 epochs, fp32; model, data and text code verbatim from NVIDIA DeepLearningExamples (BSD 3-Clause; text front end MIT, Apache 2.0); Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/tacotron2_ljs625_bs148` | the same Tacotron 2 training, batch 148, 1 epoch; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000) |
| `lambdalabs/tacotron2_ljs625_bs88` | the same Tacotron 2 training, batch 88, 2 epochs; Lambda GPU benchmark (RTX 4090, RTX 3090) |
| `lambdalabs/transformer_xl_base_wt103_bs104` | Transformer-XL base training on WikiText-103, batch 104, 400 steps, fp16; two phases (data preparation, training); model, optimizer and data code verbatim from NVIDIA DeepLearningExamples (Apache 2.0); Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/transformer_xl_base_wt103_bs64`, `transformer_xl_base_wt103_bs24` | the same, batch 64, 40 steps (RTX 6000 Ada, RTX A6000, Quadro RTX 8000); batch 24, 400 steps (RTX 4090, RTX 3090, A10) |
| `lambdalabs/transformer_xl_large_wt103_bs48` | Transformer-XL large training on WikiText-103, batch 48, 400 steps, fp16; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/transformer_xl_large_wt103_bs32`, `transformer_xl_large_wt103_bs8` | the same, batch 32, 40 steps (RTX 6000 Ada, RTX A6000, Quadro RTX 8000); batch 8, 400 steps (RTX 4090, RTX 3090, A10) |
| `lambdalabs/bert_large_squad_amp_bs28` | BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 28, 101 steps; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |

## Adding a task

The procedure, for your own workload or a published benchmark:
[doc/adding_samples.md](doc/adding_samples.md).

A task is its neutral code `tasks/<name>.py` — a plain script runnable on a
GPU machine (the task function, the call arguments `KWARGS`, models and
datasets by public ids) — and its description `tasks/<name>.json`: the
expected classification and assay fields (read from the code) and one or more
measurement sets (`reference_sec`, `min_vram_gb`, `compute_ratio` per GPU), each
with its source, date and whether it is independent of the service's
calibration. A service takes the task in its own form, a
prepared file `variants/<service>/<name>.py`. The Krauncher variant defines
the task function (`FUNC`) and the arguments of its `@client.task` decorator
(`OPTIONS`) and of the call (`KWARGS`); a task that reads a registered data
source (`OPTIONS["data"]`) lists it in `DATA_SOURCES`, and the run registers
it on the account when missing.
