"""BERT batch inference — tutorial 35.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/krauncher_tutorials/bert_batch_inference.py`); models and datasets by their public Hugging Face
ids. KWARGS are the call arguments. The expected forecast and its sources are
in bert_batch_inference.json; the form a service takes is in variants/<service>/krauncher_tutorials/bert_batch_inference.py.
"""

KWARGS = {}


def bert_batch_inference(batch_size: int = 64, max_length: int = 128):
    """Single-batch BERT classification — no autoregressive loop."""
    print("Task started. Importing torch / transformers (~10s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "google-bert/bert-base-uncased"

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


if __name__ == "__main__":
    print(bert_batch_inference(**KWARGS))
