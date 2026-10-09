"""ViT batch inference — tutorial 36.

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
Krauncher variant of tasks/krauncher_tutorials/vit_batch_inference.py (the neutral form); the expected
forecast and its sources are in tasks/krauncher_tutorials/vit_batch_inference.json.
"""

OPTIONS = dict(
    timeout=900,
    data_urls=['hf://models/google/vit-base-patch16-224'],
    dataset_size=1,
    disk_gb=10,
    stream_stderr=True,
)

KWARGS = dict()


def vit_batch_inference(batch_size: int = 128, image_size: int = 224):
    """Single-batch ViT forward — synthetic random images, no dataset needed."""
    print("Task started. Importing torch / transformers (~10s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/google__vit-base-patch16-224"

    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(
        model_path, dtype=torch.float16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s.", flush=True)

    # Synthetic batch: random pixel values in [0, 255], same shape as a real
    # processed batch. Use the processor's normalization for realism.
    import numpy as np
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, size=(batch_size, image_size, image_size, 3),
                       dtype=np.uint8)
    images = [raw[i] for i in range(batch_size)]
    inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)

    print(f"Running inference: batch={batch_size}, image={image_size}x{image_size}",
          flush=True)
    t1 = time.monotonic()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
    infer_sec = time.monotonic() - t1
    print(f"Inference completed in {infer_sec:.3f}s "
          f"({batch_size / infer_sec:.1f} images/s)", flush=True)

    return {
        "batch_size": batch_size,
        "image_size": image_size,
        "infer_sec": round(infer_sec, 4),
        "unique_classes": int(preds.unique().numel()),
    }


FUNC = vit_batch_inference
