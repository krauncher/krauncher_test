"""ResNet-152 training on Food-101, 3 epochs x 150 batches — calibration task 18 (fast).

The function below is the task, unchanged. FUNC is the task, OPTIONS the
arguments of its @client.task decorator, KWARGS the arguments of the call.
The expected forecast and its sources are in resnet152_food101.json.
"""

OPTIONS = dict(
    timeout=3600,
    pip=[],
    data='food-101',
)

# Data sources the task reads (OPTIONS["data"]), registered on the account
# by run.py when missing. Food-101, ETH Zurich (public).
DATA_SOURCES = [
    dict(name="food-101", urls=["http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"], size_gb=5.0),
]

KWARGS = dict(epochs=3, batch_size=64, lr=0.01, max_batches=150)


def train_resnet152(epochs: int, batch_size: int, lr: float, max_batches: int = 0):
    """Train ResNet-152 on Food-101 from scratch."""
    import os
    import tarfile
    import time

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet152

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Unpack dataset ──
    data_root = "/data/food-101"
    archive = "/data/food-101.tar.gz"
    if not os.path.isdir(data_root) and os.path.isfile(archive):
        print("Extracting dataset...")
        t0 = time.time()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall("/data")
        print(f"Extracted in {time.time() - t0:.1f}s")

    # ── Build train split from meta/train.txt ──
    train_dir = os.path.join(data_root, "images")
    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    n_batches = len(train_loader) if max_batches <= 0 else min(max_batches, len(train_loader))
    print(f"Training samples: {len(train_ds)}, batches/epoch: {n_batches}")

    # ── Model ──
    model = resnet152(num_classes=101).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader, 1):
            if i > n_batches:
                break
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if i % 50 == 0 or i == n_batches:
                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"batch {i}/{n_batches}  "
                    f"loss={running_loss / i:.4f}  "
                    f"acc={correct / total:.3f}",
                    flush=True,
                )

        avg_loss = running_loss / n_batches
        accuracy = correct / total
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})
        print(
            f"Epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  acc={accuracy:.3f}",
            flush=True,
        )

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_vram:.2f} GB")

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "final_loss": round(history[-1]["loss"], 4),
        "final_accuracy": round(history[-1]["accuracy"], 4),
    }


FUNC = train_resnet152
