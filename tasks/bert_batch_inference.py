"""BERT batch inference — tutorial 35.

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
REFERENCE holds measured values the forecast is checked against.
"""

OPTIONS = dict(
    timeout=900,
    data_urls=['hf://models/google-bert/bert-base-uncased'],
    dataset_size=1,
    disk_gb=10,
    stream_stderr=True,
)

KWARGS = dict()

# Reference values. Source: Krauncher calibration runs on rented hosts
# (March-September 2026), the same data the analyzer's calibrator is fitted on,
# so this is not an independent check. Geometric mean over runs.
REFERENCE = dict(
    # Passport (assay), reference card = RTX PRO 6000 Blackwell.
    reference_sec=13,  # whole task measured on the reference card
    vram_gb=0.8,  # median vram_peak_mb (815 MB) over completed tasks with this entry_point,
    # all configurations mixed; device memory from nvidia-smi, polled every 10 s
    # Ladder: measured compute (exec - io - setup) on a GPU, on its rented hosts,
    # over the same on the reference card brought to the ideal host with the
    # analyzer's host model (4.2 s), per ladder gpu_id.
    compute_ratio={
        "a100_sxm_80": 1.761, "h100_sxm": 1.458, "l4": 2.531, "l40": 1.610,
        "rtx_5080": 2.184, "rtx_5090": 2.288, "rtx_6000_ada": 1.758, "rtx_6000_blackwell": 1.000,
        "rtx_6000s": 1.451, "rtx_a6000": 2.067,
    },
)


def bert_batch_inference(batch_size: int = 64, max_length: int = 128):
    """Single-batch BERT classification — no autoregressive loop."""
    print("Task started. Importing torch / transformers (~10s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/google-bert__bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Tokenizer + model loaded in {time.monotonic() - t0:.1f}s.", flush=True)

    # Synthetic short reviews — deterministic, no dataset download needed.
    texts = [
        f"This product number {i} was {'excellent' if i % 2 else 'terrible'} "
        f"with battery life {(i % 10) + 1} hours."
        for i in range(batch_size)
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    print(f"Running inference: batch={batch_size}, max_len={max_length}", flush=True)
    t1 = time.monotonic()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
    infer_sec = time.monotonic() - t1
    print(f"Inference completed in {infer_sec:.3f}s "
          f"({batch_size / infer_sec:.1f} samples/s)", flush=True)

    return {
        "batch_size": batch_size,
        "max_length": max_length,
        "infer_sec": round(infer_sec, 4),
        "predicted_positive": int((preds == 1).sum().item()),
    }


FUNC = bert_batch_inference
