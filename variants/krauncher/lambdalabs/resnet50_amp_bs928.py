"""Krauncher variant of tasks/lambdalabs/resnet50_amp_bs928.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/lambdalabs/resnet50_amp_bs928.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call. Synthetic data: nothing to download.
"""

OPTIONS = dict(
    timeout=1800,
)

KWARGS = dict(batch_size=928, epochs=1, iterations_per_epoch=50)


def train_resnet50_amp(batch_size: int = 928, epochs: int = 1, iterations_per_epoch: int = 50):
    import time

    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    device = torch.device("cuda")
    model = resnet50(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=256, growth_interval=1_000_000_000)

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        for i in range(iterations_per_epoch):
            with torch.cuda.amp.autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_value = loss.item()
        print(f"epoch {epoch + 1}/{epochs}: loss {loss_value:.3f}")
    elapsed = time.time() - t0
    steps = epochs * iterations_per_epoch
    return {"steps": steps, "elapsed_sec": round(elapsed, 2),
            "images_per_sec": round(steps * batch_size / elapsed, 1)}



FUNC = train_resnet50_amp
