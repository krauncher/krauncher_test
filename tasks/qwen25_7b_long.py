"""Qwen2.5-7B-Instruct long generation — tutorial 31.

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
REFERENCE holds measured values the forecast is checked against.
"""

OPTIONS = dict(
    timeout=43200,
    data_urls=['hf://datasets/openai/gsm8k', 'hf://models/Qwen/Qwen2.5-7B-Instruct'],
    pip=['datasets'],
    dataset_size=3,
    disk_gb=40,
    stream_stderr=True,
)

KWARGS = dict()

# Reference values. Source: Krauncher calibration runs on rented hosts
# (March-September 2026), the same data the analyzer's calibrator is fitted on,
# so this is not an independent check. Geometric mean over runs.
REFERENCE = dict(
    # Passport (assay), reference card = RTX PRO 6000 Blackwell.
    reference_sec=204,  # whole task measured on the reference card
    vram_gb=15.8,  # median vram_peak_mb (16182 MB) over completed tasks with this entry_point,
    # all configurations mixed; device memory from nvidia-smi, polled every 10 s
    # Ladder: measured compute (exec - io - setup) on a GPU, on its rented hosts,
    # over the same on the reference card brought to the ideal host with the
    # analyzer's host model (141.8 s), per ladder gpu_id.
    compute_ratio={
        "a100_sxm_80": 1.898, "h100_sxm": 1.341, "l40": 1.977, "rtx_5090": 1.251,
        "rtx_6000_ada": 1.855, "rtx_6000_blackwell": 1.000, "rtx_6000s": 1.330, "rtx_a6000": 2.605,
    },
)


def qwen7b_long_inference(num_samples: int = 30, max_new_tokens: int = 2000):
    """Long-form chain-of-thought generation, pushes per-task token total ~60k."""
    print("Task started. Importing torch / transformers (~15-25s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/Qwen__Qwen2.5-7B-Instruct"
    dataset_path = "/data/openai__gsm8k"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading model weights (fp16, ~14 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    print(f"Model loaded in {time.monotonic() - t1:.1f}s.", flush=True)

    ds_full = load_dataset(dataset_path, "main", split="test")
    ds = ds_full.select(range(min(num_samples, len(ds_full))))
    print(f"Running inference on {len(ds)} samples, max_new_tokens={max_new_tokens}",
          flush=True)

    last_log = time.monotonic()
    HEARTBEAT_SEC = 50

    for i, sample in enumerate(ds):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": (
                f"Solve this carefully, showing all reasoning steps in detail. "
                f"Problem: {sample['question']}"
            )}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        now = time.monotonic()
        if (i + 1) % 5 == 0 or (now - last_log) >= HEARTBEAT_SEC:
            print(f"  [{i + 1}/{len(ds)}] elapsed {now - t0:.0f}s", flush=True)
            last_log = now

    return {
        "samples": len(ds),
        "max_new_tokens": max_new_tokens,
        "total_tokens": len(ds) * max_new_tokens,
    }


FUNC = qwen7b_long_inference
