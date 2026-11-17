# train_v3.py  -  Task 2, version 3.
#
# Builds on train_v2.py. Two changes that target the rare-wait problem:
#   1. Trains on the main dataset PLUS the dense/conflict-rich dataset, so the
#      model sees many more real yields (~4x more, measured).
#   2. Uses a CLASS-WEIGHTED loss: rare actions (especially Wait) count more,
#      so the model is punished for ignoring them instead of coasting on the
#      99% of easy "keep going" moves.
#
# Still validates on whole scenarios the model never saw (no leakage), and
# still reports per-class accuracy + a confusion matrix so the wait number is
# never hidden behind the overall number.
#
# Run from the repo root:
#     python task2/train_v3.py
#
# If the dense set does not exist yet, generate it first:
#     cd task1 && python generate_dense_dataset.py && cd ..
#     (the warehouse set already mixes flavours, so one folder is enough)

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidance import scenario_to_samples

NAMES = ["N", "S", "E", "W", "Wait"]
# the datasets to learn from; warehouse mix already includes the conflict-rich cluster maps
FOLDERS = ["task1/wh_dataset"]   # warehouse mix (spread+cluster+debris), with parking legs


class PolicyCNN(nn.Module):
    """Same 4-layer CNN as before (unchanged, so model.pt stays compatible)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 128), nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


def build(folders):
    """Load every scenario from the folders; remember which scenario each
    sample came from (g) so we can split by scenario, not by sample."""
    X, y, g = [], [], []
    scenario_no = 0
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"  (skipping missing folder: {folder})")
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(folder, name)) as f:
                # keep_goal_wait low: drop most "parked on goal" waits so the
                # Wait class is mostly REAL yields, which is what we weight for
                inp, lab = scenario_to_samples(json.load(f), keep_goal_wait=0.05)
            X += inp
            y += lab
            g += [scenario_no] * len(lab)
            scenario_no += 1
    return np.stack(X), np.array(y), np.array(g), scenario_no


def main():
    X_np, y_np, g, n_scen = build(FOLDERS)
    X = torch.tensor(X_np)
    y = torch.tensor(y_np, dtype=torch.long)
    print(f"Built {len(y)} samples from {n_scen} scenarios.")
    print("label counts:", dict(zip(NAMES, np.bincount(y_np, minlength=5))))

    # split by scenario: 80% train, 20% validation (unseen maps)
    np.random.seed(0)
    torch.manual_seed(0)
    scenarios = np.unique(g)
    np.random.shuffle(scenarios)
    val_scenarios = scenarios[int(0.8 * len(scenarios)):]
    is_val = np.isin(g, val_scenarios)
    train_idx = torch.tensor(np.where(~is_val)[0])
    val_idx = torch.tensor(np.where(is_val)[0])

    # --- class weights: balanced, so rare actions (Wait) matter more ---
    counts = np.bincount(y_np[~is_val], minlength=5).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)       # inverse-frequency
    weights = torch.tensor(weights, dtype=torch.float32)
    print("class weights:", {NAMES[i]: round(float(weights[i]), 2) for i in range(5)})

    model = PolicyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss(weight=weights)    # <-- the key change

    for epoch in range(1, 21):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        total, batches = 0.0, 0
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            optimizer.zero_grad()
            loss = loss_function(model(X[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1

        model.eval()
        with torch.no_grad():
            guess = model(X[val_idx]).argmax(1)
        truth = y[val_idx]
        acc = (guess == truth).float().mean().item()
        wait = truth == 4
        wait_recall = (guess[wait] == 4).float().mean().item() if wait.any() else 0.0
        move_acc = (guess[~wait] == truth[~wait]).float().mean().item()
        print(f"epoch {epoch:2d}  loss {total/batches:.3f}  acc {acc:.3f}"
              f"  WAIT recall {wait_recall:.3f}  move acc {move_acc:.3f}")

    # confusion matrix on the validation set
    table = np.zeros((5, 5), dtype=int)
    for t, p in zip(truth.numpy(), guess.numpy()):
        table[t][p] += 1
    print("\nConfusion matrix (row = CBS, column = model)")
    print("      ", NAMES)
    for i in range(5):
        print(NAMES[i].ljust(5), table[i])

    torch.save(model.state_dict(), "model.pt")
    print("\nSaved model.pt")


if __name__ == "__main__":
    main()
