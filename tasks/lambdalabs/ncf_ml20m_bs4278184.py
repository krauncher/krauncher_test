"""NCF (NeuMF) training on MovieLens-20M, batch 4 278 184, 2 epochs — Lambda GPU benchmark.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/ncf_ml20m_bs4278184.py`); the dataset is downloaded from
its public URL into ./data when missing. KWARGS are the call arguments. The
expected forecast and its sources are in ncf_ml20m_bs4278184.json; the form a
service takes is in variants/<service>/lambdalabs/ncf_ml20m_bs4278184.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6
  (2025-12-17), pytorch/scripts/config_v1/config_pytorch_24GB.sh,
  PyTorch_ncf_FP16_PARAMS: ncf.py --data /data/ncf/cache/ml-20m --epochs 2
  --batch_size 4278184
  (scripts/benchmark_pytorch.sh, benchmark_pytorch_ncf). The "FP16" set does
  not pass --amp, so the run computes in fp32.
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, commit 667536cc (2025-12-10), PyTorch/Recommendation/NCF.

This file is the smallest faithful reduction of that run to one file; every
choice follows the source (paths relative to NCF/ at 667536cc):
- model: NeuMF — MF and MLP embeddings for users and items, MLP layers
  [256, 256, 128, 64] with ReLU and dropout 0.5, MF factors 64, a final linear
  layer over [MF product, MLP output] (neumf.py:38-98; ncf.py defaults
  --factors 64 --layers 256 256 128 64 --dropout 0.5);
- data: MovieLens-20M ratings, ids made contiguous, each user's latest
  rating held out for the test (convert.py:175-183); per epoch every positive
  is followed by 4 negatives with random items (dataloading.py:208-236,
  --negative_samples 4), the epoch is shuffled and split into batches, the
  last partial batch dropped (dataloading.py:240-250);
- step: logits, BCEWithLogitsLoss mean, backward, optimizer step, gradients
  dropped (ncf.py:300-321); Adam lr 0.0045, betas (0.25, 0.5), eps 1e-8
  (ncf.py:75-87, 248-249);
- validation after every epoch: each test user's positive and 100 negatives
  scored in batches of 2**20, hit rate in the top 10 (ncf.py:126-173, 332;
  convert.py --valid_negative 100);
- 2 epochs; the default hit-rate threshold 1.0 never stops the run early
  (ncf.py:81, 364-367).

Differences from the source, none of which changes the measured work:
torch.optim.Adam instead of apex FusedAdam; the loss is called directly
instead of through torch.jit.trace; the test negatives are random items, not
filtered against the user's rated items (convert.py's sampler filters them —
this changes the reported hit rate, not the compute); the cached tensors of
convert.py are built in memory.
"""

KWARGS = {"batch_size": 4_278_184, "epochs": 2}


def train_ncf_ml20m(batch_size: int = 4_278_184, epochs: int = 2):
    import os
    import time
    import urllib.request
    import zipfile

    import pandas as pd
    import torch
    import torch.nn as nn

    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    archive, csv_path = "data/ml-20m.zip", "data/ml-20m/ratings.csv"
    if not os.path.isfile(csv_path):
        os.makedirs("data", exist_ok=True)
        if not os.path.isfile(archive):
            print("Downloading MovieLens-20M (~190 MB)...")
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extract("ml-20m/ratings.csv", "data")

    df = pd.read_csv(csv_path, usecols=["userId", "movieId", "timestamp"])
    df["userId"] = pd.factorize(df["userId"])[0]
    df["movieId"] = pd.factorize(df["movieId"])[0]
    nb_users, nb_items = int(df["userId"].max()) + 1, int(df["movieId"].max()) + 1
    df = df.sort_values("timestamp")
    test = df.groupby("userId", group_keys=False).tail(1).sort_values("userId")
    train = df.drop(test.index)

    device = torch.device("cuda")
    train_users = torch.tensor(train["userId"].values, device=device)
    train_items = torch.tensor(train["movieId"].values, device=device)
    valid_negative, negative_samples = 100, 4
    test_users = torch.tensor(test["userId"].values, device=device).repeat_interleave(valid_negative + 1)
    test_items = torch.cat((torch.tensor(test["movieId"].values, device=device).view(-1, 1),
                            torch.randint(0, nb_items, (len(test), valid_negative), device=device)), dim=1).view(-1)

    class NeuMF(nn.Module):
        def __init__(self, mf_dim=64, layers=(256, 256, 128, 64), dropout=0.5):
            super().__init__()
            self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
            self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
            self.mlp_user_embed = nn.Embedding(nb_users, layers[0] // 2)
            self.mlp_item_embed = nn.Embedding(nb_items, layers[0] // 2)
            self.mlp = nn.ModuleList(nn.Linear(layers[i - 1], layers[i]) for i in range(1, len(layers)))
            self.final = nn.Linear(layers[-1] + mf_dim, 1)
            self.dropout = dropout

        def forward(self, user, item):
            xmf = self.mf_user_embed(user) * self.mf_item_embed(item)
            x = torch.cat((self.mlp_user_embed(user), self.mlp_item_embed(item)), dim=1)
            for layer in self.mlp:
                x = nn.functional.dropout(nn.functional.relu(layer(x)), p=self.dropout, training=self.training)
            return self.final(torch.cat((xmf, x), dim=1))

    model = NeuMF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0045, betas=(0.25, 0.5), eps=1e-8)
    criterion = nn.BCEWithLogitsLoss()

    throughputs, hr = [], 0.0
    for epoch in range(epochs):
        model.train()
        begin = time.time()
        users = torch.cat((train_users, train_users.repeat(negative_samples)))
        items = torch.cat((train_items, torch.randint(0, nb_items, (len(train_items) * negative_samples,), device=device)))
        labels = torch.cat((torch.ones(len(train_items), device=device),
                            torch.zeros(len(train_items) * negative_samples, device=device)))
        order = torch.randperm(len(users), device=device)
        for idx in order.split(batch_size)[:-1]:
            loss = criterion(model(users[idx], items[idx]).view(-1), labels[idx])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        throughputs.append(len(users) / (time.time() - begin))
        del users, items, labels, order

        model.eval()
        with torch.no_grad():
            scores = torch.cat([model(u, i) for u, i in zip(test_users.split(2 ** 20), test_items.split(2 ** 20))])
            top = torch.topk(scores.view(-1, valid_negative + 1), 10)[1]
            hr = (top == 0).any(dim=1).float().mean().item()
        print(f"epoch {epoch + 1}/{epochs}: {throughputs[-1]:.3e} samples/s, hr@10 {hr:.4f}, loss {loss.item():.4f}")
    return {"train_samples": len(train_items), "epochs": epochs,
            "mean_train_samples_per_sec": round(sum(throughputs) / len(throughputs), 1), "hr@10": round(hr, 4)}


if __name__ == "__main__":
    print(train_ncf_ml20m(**KWARGS))
