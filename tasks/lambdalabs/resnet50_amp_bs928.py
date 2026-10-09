"""ResNet-50 training, AMP, batch 928, synthetic data — Lambda GPU benchmark.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/resnet50_amp_bs928.py`). KWARGS are the call
arguments. The expected forecast and its sources are in
resnet50_amp_bs928.json; the form a service takes is in
variants/<service>/lambdalabs/resnet50_amp_bs928.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6
  (2025-12-17), pytorch/scripts/config_v1/config_pytorch_48GB.sh,
  PyTorch_resnet50_AMP_PARAMS: Classification/ConvNets main.py --arch resnet50
  --amp --static-loss-scale 256 --epochs 1 --prof 50 --batch-size 928
  --training-only --data-backend synthetic --workers 64.
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, commit 667536cc (2025-12-10), PyTorch/Classification/ConvNets.

This file is the smallest faithful reduction of that run to one file; every
choice below follows the source (paths relative to ConvNets/ at 667536cc):
- model: ResNet-50 v1.5 (image_classification/models/resnet.py:410-418 —
  layers [3, 4, 6, 3], widths 64..512, expansion 4, stride on the 3x3 conv),
  the same topology as torchvision.models.resnet50;
- data: one synthetic batch, randn(batch, 3, 224, 224) on the GPU and random
  labels, yielded again every iteration (image_classification/dataloaders.py
  SynteticDataLoader, :517-546; image size 224 = the arch default);
- iterations: --prof N runs N iterations per epoch (image_classification/
  training.py:244), so epochs x prof = 1 x 50 = 50 training steps; no
  evaluation (--training-only);
- step: forward and loss under torch.cuda.amp.autocast, scaler.scale(loss)
  .backward(), scaler.step, scaler.update, zero_grad (training.py:91-95,
  154-178); GradScaler(init_scale=256, growth_interval=1e9) = a static loss
  scale (main.py:482-488); loss.item() every step, as the per-step log
  (training.py:225-240);
- optimizer and loss: SGD lr 0.1, momentum 0.9, weight decay 1e-4, no
  Nesterov; CrossEntropyLoss, no label smoothing (main.py defaults);
  memory format NCHW (main.py default).

Differences from the source, none of which changes the step's compute:
torchvision's ResNet-50 instead of DLE's builder (same layers; weight init
differs); weight decay applied to BatchNorm parameters too (DLE excludes them
by default); no 5 s sleep at the end of each epoch (training.py:245; outside
the measured throughput); single process, no multiproc launcher (one GPU).
"""

KWARGS = {"batch_size": 928, "epochs": 1, "iterations_per_epoch": 50}


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


if __name__ == "__main__":
    print(train_resnet50_amp(**KWARGS))
