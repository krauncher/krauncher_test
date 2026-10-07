"""ViT batch inference — tutorial 36.

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
REFERENCE holds measured values the forecast is checked against.
"""

OPTIONS = dict(
    timeout=900,
    data_urls=['hf://models/google/vit-base-patch16-224'],
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
    reference_sec=9,  # whole task measured on the reference card
    vram_gb=0.8,  # median vram_peak_mb (834 MB) over completed tasks with this entry_point,
    # all configurations mixed; device memory from nvidia-smi, polled every 10 s
    # Ladder: measured compute (exec - io - setup) on a GPU, on its rented hosts,
    # over the same on the reference card brought to the ideal host with the
    # analyzer's host model (4.6 s), per ladder gpu_id.
    compute_ratio={
        "a100_sxm_40": 3.137, "a100_sxm_80": 5.174, "h100_sxm": 1.538, "l4": 3.468,
        "l40": 1.584, "rtx_2000_ada": 1.680, "rtx_5080": 1.601, "rtx_5090": 1.713,
        "rtx_6000_ada": 1.914, "rtx_6000_blackwell": 1.000, "rtx_6000s": 1.564, "rtx_a6000": 2.700,
    },
)


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
