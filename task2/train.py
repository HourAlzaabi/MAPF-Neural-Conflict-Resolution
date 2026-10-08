"""train.py - 4-layer CNN behavioral cloning. Run: python task2/train.py [dummy|real]"""
import sys
import numpy as np
import torch
import torch.nn as nn


class PolicyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 128), nn.ReLU(),
            nn.Linear(128, 5),               # logits; softmax applied in the loss
        )

    def forward(self, x):
        return self.net(x)


def load(mode):
    if mode == "dummy":
        X = np.random.rand(2000, 4, 11, 11).astype(np.float32)
        y = np.random.randint(0, 5, 2000)
    else:
        d = np.load("task2/dataset.npz")
        X, y = d["X"], d["y"]
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)


def main(mode="dummy", epochs=20):
    X, y = load(mode)
    n = int(0.8 * len(X))
    perm = torch.randperm(len(X))
    tr, va = perm[:n], perm[n:]
    model = PolicyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        order = tr[torch.randperm(len(tr))]
        for b in range(0, len(order), 64):
            idx = order[b:b + 64]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X[va]).argmax(1) == y[va]).float().mean().item()
        print(f"epoch {ep + 1:2d}  loss {loss.item():.3f}  val acc {acc:.3f}")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dummy")
