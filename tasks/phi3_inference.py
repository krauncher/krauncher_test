"""Phi-3-mini inference on GSM8K — tutorial 32.

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
REFERENCE holds measured values the forecast is checked against.
"""

OPTIONS = dict(
    timeout=9600,
    data_urls=['hf://datasets/openai/gsm8k', 'hf://models/microsoft/Phi-3-mini-4k-instruct'],
    pip=['datasets'],
    dataset_size=3,
    disk_gb=25,
    stream_stderr=True,
)

KWARGS = dict()

# Reference values. Source: Krauncher calibration runs on rented hosts
# (March-September 2026), the same data the analyzer's calibrator is fitted on,
# so this is not an independent check. Geometric mean over runs.
REFERENCE = dict(
    # Passport (assay), reference card = RTX PRO 6000 Blackwell.
    reference_sec=133,  # whole task measured on the reference card
    vram_gb=8.8,  # median vram_peak_mb (8986 MB) over completed tasks with this entry_point,
    # all configurations mixed; device memory from nvidia-smi, polled every 10 s
    # Ladder: measured compute (exec - io - setup) on a GPU, on its rented hosts,
    # over the same on the reference card brought to the ideal host with the
    # analyzer's host model (107.1 s), per ladder gpu_id.
    compute_ratio={
        "a100_sxm_80": 2.891, "h100_sxm": 2.082, "l4": 4.222, "l40": 2.695,
        "rtx_5080": 2.910, "rtx_5090": 1.931, "rtx_6000_ada": 2.702, "rtx_6000_blackwell": 1.000,
        "rtx_6000s": 1.516, "rtx_a6000": 3.715,
    },
)


def phi3_inference(num_samples: int = 50, max_new_tokens: int = 500):
    """Mid-size LLM Q/A generation."""
    print("Task started. Importing torch / transformers (~12-20s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/microsoft__Phi-3-mini-4k-instruct"
    dataset_path = "/data/openai__gsm8k"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading model weights (fp16, ~7.6 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded in {time.monotonic() - t1:.1f}s.", flush=True)

    ds_full = load_dataset(dataset_path, "main", split="test")
    ds = ds_full.select(range(min(num_samples, len(ds_full))))
    print(f"Running inference on {len(ds)} samples, max_new_tokens={max_new_tokens}",
          flush=True)

    last_log = time.monotonic()
    HEARTBEAT_SEC = 40

    for i, sample in enumerate(ds):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve step by step: {sample['question']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        now = time.monotonic()
        if (i + 1) % 10 == 0 or (now - last_log) >= HEARTBEAT_SEC:
            print(f"  [{i + 1}/{len(ds)}] elapsed {now - t0:.0f}s", flush=True)
            last_log = now

    return {
        "samples": len(ds),
        "max_new_tokens": max_new_tokens,
        "total_tokens": len(ds) * max_new_tokens,
    }


FUNC = phi3_inference
