"""BERT fine-tuning on IMDB — tutorial 20 of the krauncher client.

The function below is the tutorial's task, unchanged. FUNC is the task,
OPTIONS the arguments of its @client.task decorator, KWARGS the arguments
of the call.
Krauncher variant of tasks/krauncher_tutorials/bert_imdb.py (the neutral form); the expected
forecast and its sources are in tasks/krauncher_tutorials/bert_imdb.json.
"""

OPTIONS = dict(
    timeout=1800,
    data_urls=["hf://datasets/stanfordnlp/imdb", "hf://models/google-bert/bert-base-uncased"],
    dataset_size=84,  # IMDB dataset ~84 MB
)

KWARGS = dict(num_epochs=3, batch_size=16, lr=2e-5)


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
