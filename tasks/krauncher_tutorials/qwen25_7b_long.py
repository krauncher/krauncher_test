"""Qwen2.5-7B-Instruct long generation — tutorial 31.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/krauncher_tutorials/qwen25_7b_long.py`); models and datasets by their public Hugging Face
ids. KWARGS are the call arguments. The expected forecast and its sources are
in qwen25_7b_long.json; the form a service takes is in variants/<service>/krauncher_tutorials/qwen25_7b_long.py.
"""

KWARGS = {}


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
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path = "openai/gsm8k"

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


if __name__ == "__main__":
    print(qwen7b_long_inference(**KWARGS))
