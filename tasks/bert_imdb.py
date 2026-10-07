"""BERT fine-tuning on IMDB — tutorial 20 of the krauncher client.

The function below is the tutorial's task, unchanged. FUNC is the task,
OPTIONS the arguments of its @client.task decorator, KWARGS the arguments
of the call. REFERENCE holds measured values the forecast is checked against.
"""

OPTIONS = dict(
    timeout=1800,
    data_urls=["hf://datasets/stanfordnlp/imdb", "hf://models/google-bert/bert-base-uncased"],
    dataset_size=84,  # IMDB dataset ~84 MB
)

KWARGS = dict(num_epochs=3, batch_size=16, lr=2e-5)

# Reference values. Source: Krauncher calibration runs on rented hosts
# (March-September 2026), the same data the analyzer's calibrator is fitted on,
# so this is not an independent check. Geometric mean over runs.
REFERENCE = dict(
    # Passport (assay), reference card = RTX PRO 6000 Blackwell.
    reference_sec=143,  # whole task measured on the reference card
    vram_gb=3.9,  # median vram_peak_mb (3979 MB) over completed tasks with this entry_point,
    # all configurations mixed; device memory from nvidia-smi, polled every 10 s
    # Ladder: measured compute (exec - io - setup) on a GPU, on its rented hosts,
    # over the same on the reference card brought to the ideal host with the
    # analyzer's host model (136.0 s measured -> 136.6 s), per ladder gpu_id.
    compute_ratio={
        "a100_pcie_80": 2.051, "a100_sxm_80": 1.796, "a40": 3.783, "b200": 0.667,
        "h100_pcie": 1.902, "h100_sxm": 1.660, "l4": 5.574, "l40": 2.152, "l40s": 2.327,
        "qrtx_6000": 5.156, "rtx_2000_ada": 7.302, "rtx_3090": 3.475, "rtx_4060ti": 5.998,
        "rtx_4090": 1.994, "rtx_4500_ada": 3.496, "rtx_4500_blackwell": 2.165,
        "rtx_5060ti": 4.599, "rtx_5070ti": 4.775, "rtx_5080": 2.516, "rtx_5090": 1.912,
        "rtx_6000_ada": 2.097, "rtx_6000_blackwell": 1.000, "rtx_6000s": 1.285,
        "rtx_a4000": 5.556, "rtx_a4500": 4.021, "rtx_a5000": 3.756, "rtx_a6000": 3.318,
    },
)


def finetune_bert_imdb(num_epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    """Fine-tune BERT on IMDB sentiment classification (positive/negative)."""
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    print("Task started. Waiting for result (download + training, ~15-20 min)...")

    # Load from pre-downloaded local paths (hf:// data bridge)
    model_path = "/data/google-bert__bert-base-uncased"
    dataset_path = "/data/stanfordnlp__imdb"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2,
    )

    import os
    _ds_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(dataset_path) for f in fn) / (1 << 20)
    print(f"Loading dataset ({_ds_mb:.0f} MB)...")
    ds = load_dataset(dataset_path)

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    print("Tokenizing...")
    ds = ds.map(tokenize, batched=True, batch_size=1000)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Training arguments — standard HF Trainer config
    training_args = TrainingArguments(
        output_dir="/tmp/bert-imdb",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": float(accuracy)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    train_result = trainer.train()

    print("Evaluating...")
    eval_result = trainer.evaluate()

    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Eval accuracy: {eval_result['eval_accuracy']:.4f}")

    return {
        "train_loss": round(train_result.training_loss, 4),
        "eval_accuracy": round(eval_result["eval_accuracy"], 4),
        "eval_loss": round(eval_result["eval_loss"], 4),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_runtime_sec": round(train_result.metrics["train_runtime"], 1),
    }


FUNC = finetune_bert_imdb
