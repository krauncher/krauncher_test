"""Qwen2.5-7B-Instruct inference on GSM8K — tutorial 22.

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
REFERENCE holds measured values the forecast is checked against.
"""

OPTIONS = dict(
    timeout=36000,
    data_urls=['hf://datasets/openai/gsm8k', 'hf://models/Qwen/Qwen2.5-7B-Instruct'],
    pip=['datasets'],
    dataset_size=3,
    disk_gb=40,
)

KWARGS = dict()

# Reference values. Source: Krauncher calibration runs on rented hosts
# (March-September 2026), the same data the analyzer's calibrator is fitted on,
# so this is not an independent check. Geometric mean over runs.
REFERENCE = dict(
    # Passport (assay), reference card = RTX PRO 6000 Blackwell.
    reference_sec=711,  # whole task measured on the reference card
    vram_gb=15.7,  # median vram_peak_mb (16091 MB) over completed tasks with this entry_point,
    # all configurations mixed; device memory from nvidia-smi, polled every 10 s
    # Ladder: measured compute (exec - io - setup) on a GPU, on its rented hosts,
    # over the same on the reference card brought to the ideal host with the
    # analyzer's host model (576.0 s), per ladder gpu_id.
    compute_ratio={
        "a100_sxm_80": 1.774, "h100_sxm": 1.180, "l4": 4.911, "l40": 1.893,
        "rtx_4090": 2.345, "rtx_4500_blackwell": 1.837, "rtx_5090": 1.444, "rtx_6000_ada": 1.698,
        "rtx_6000_blackwell": 1.000, "rtx_6000s": 1.086, "rtx_a5000": 2.186, "rtx_a6000": 2.396,
    },
)


def qwen_inference_gsm8k(
    num_samples: int = 200,
    max_new_tokens: int = 256,
):
    """Generate solutions for GSM8K math problems with Qwen2.5-7B-Instruct."""
    # Print before imports so the user sees the task is alive even while
    # torch/transformers are loading (~15-25s on a cold container).
    print("Task started. Importing torch / transformers (~15-25s)...",
          flush=True)
    import re
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s. "
          f"Loading tokenizer...", flush=True)

    t0 = time.monotonic()

    model_path = "/data/Qwen__Qwen2.5-7B-Instruct"
    dataset_path = "/data/openai__gsm8k"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading model weights (fp16, ~14 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded in {time.monotonic() - t1:.1f}s. "
          f"Switching to eval mode.", flush=True)
    model.eval()

    print(f"Loading dataset from {dataset_path}...", flush=True)
    ds_full = load_dataset(dataset_path, "main", split="test")
    ds = ds_full.select(range(min(num_samples, len(ds_full))))

    print(f"Running inference on {len(ds)} samples, "
          f"max_new_tokens={max_new_tokens}", flush=True)

    correct = 0
    total = 0
    num_re = re.compile(r"-?\d+\.?\d*")
    last_log = time.monotonic()
    HEARTBEAT_SEC = 50

    for i, sample in enumerate(ds):
        messages = [
            {
                "role": "user",
                "content": (
                    f"Solve this step by step. Put the final numeric answer "
                    f"on the last line.\n\nProblem: {sample['question']}"
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        gt_raw = sample["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gt = float(gt_raw)
        except ValueError:
            continue

        preds = num_re.findall(response.replace(",", ""))
        if preds:
            try:
                pred = float(preds[-1])
                if abs(pred - gt) < 1e-3:
                    correct += 1
            except ValueError:
                pass

        total += 1

        now = time.monotonic()
        if (i + 1) % 20 == 0 or (now - last_log) >= HEARTBEAT_SEC:
            elapsed = now - t0
            print(f"  [{i + 1}/{len(ds)}] running accuracy: "
                  f"{correct / total:.3f} (elapsed {elapsed:.0f}s)", flush=True)
            last_log = now

    accuracy = correct / total if total else 0.0

    return {
        "samples": len(ds),
        "evaluated": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "max_new_tokens": max_new_tokens,
    }


FUNC = qwen_inference_gsm8k
