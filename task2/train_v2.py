# Version 1
# Wait accuracy is still low on real yields (only 33 in the dataset).
# The high overall accuracy mostly comes from easy moves and parked waits.
# Next: denser dataset with more conflicts, then retrain with class weights.

# train_v2.py
# Trains the CNN. Validation uses scenarios the model never saw.
# Run: python task2/train_v2.py

import numpy as np
import torch
import torch.nn as nn

names = ["N", "S", "E", "W", "Wait"]

# the CNN: 4 convolution layers, then 2 normal layers
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
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)

# load the data
data = np.load("task2/dataset_groups.npz")
X = torch.tensor(data["X"])
y = torch.tensor(data["y"], dtype=torch.long)
g = data["g"]

# split by scenario: 80% train, 20% validation
np.random.seed(0)
torch.manual_seed(0)
scenarios = np.unique(g)
np.random.shuffle(scenarios)
cut = int(0.8 * len(scenarios))
val_scenarios = scenarios[cut:]

is_val = np.isin(g, val_scenarios)
train_idx = torch.tensor(np.where(~is_val)[0])
val_idx = torch.tensor(np.where(is_val)[0])
print("train samples:", len(train_idx), " validation samples:", len(val_idx))

model = PolicyCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

for epoch in range(1, 21):
    # training
    model.train()
    mixed = train_idx[torch.randperm(len(train_idx))]
    total_loss = 0
    batches = 0
    for start in range(0, len(mixed), 64):
        batch = mixed[start:start + 64]
        optimizer.zero_grad()
        loss = loss_function(model(X[batch]), y[batch])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1

    # checking on the validation scenarios
    model.eval()
    with torch.no_grad():
        guess = model(X[val_idx]).argmax(1)
    truth = y[val_idx]

    acc = (guess == truth).float().mean().item()
    wait = truth == 4
    wait_acc = (guess[wait] == 4).float().mean().item()
    move_acc = (guess[~wait] == truth[~wait]).float().mean().item()

    print("epoch", epoch,
          "loss", round(total_loss / batches, 3),
          "acc", round(acc, 3),
          "wait acc", round(wait_acc, 3),
          "move acc", round(move_acc, 3))

# confusion matrix: row = what CBS did, column = what the model guessed
table = np.zeros((5, 5), dtype=int)
for t, p in zip(truth.numpy(), guess.numpy()):
    table[t][p] += 1

print("\nConfusion matrix (row = CBS, column = model)")
print("      ", names)
for i in range(5):
    print(names[i].ljust(5), table[i])

torch.save(model.state_dict(), "model.pt")
