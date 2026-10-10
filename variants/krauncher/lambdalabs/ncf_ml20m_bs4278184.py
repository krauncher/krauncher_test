"""Krauncher variant of tasks/lambdalabs/ncf_ml20m_bs4278184.py (the neutral form, with the
source and the reduction it documents); the expected forecast and its
sources are in tasks/lambdalabs/ncf_ml20m_bs4278184.json.

The task function is the neutral one; the MovieLens-20M archive comes through
a registered data source (/data/ml-20m.zip) instead of a download. FUNC is the
task, OPTIONS the arguments of its @client.task decorator, KWARGS the
arguments of the call.
"""

OPTIONS = dict(
    timeout=3600,
    data="ml-20m",
)

# Data source the task reads (OPTIONS["data"]), registered on the account when
# missing. MovieLens-20M, GroupLens (public).
DATA_SOURCES = [
    dict(name="ml-20m", urls=["https://files.grouplens.org/datasets/movielens/ml-20m.zip"], size_gb=0.2),
]

KWARGS = dict(batch_size=4_278_184, epochs=2)


def train_ncf_ml20m(batch_size: int = 4_278_184, epochs: int = 2):
    import os
    import time
    import zipfile

    import pandas as pd
    import torch
    import torch.nn as nn

    archive, csv_path = "/data/ml-20m.zip", "/data/ml-20m/ratings.csv"
    if not os.path.isfile(csv_path):
        with zipfile.ZipFile(archive) as z:
            z.extract("ml-20m/ratings.csv", "/data")

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



FUNC = train_ncf_ml20m
