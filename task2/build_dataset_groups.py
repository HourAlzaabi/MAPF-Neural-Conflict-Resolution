# build_dataset_groups.py
# Makes the training data, like guidance.build_dataset,
# but also remembers which scenario each sample came from.
# Run: python task2/build_dataset_groups.py task1/dataset

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidance import scenario_to_samples

folder = sys.argv[1] if len(sys.argv) > 1 else "task1/dataset"

X = []   # the inputs (4 x 11 x 11)
y = []   # the answers (0 to 4)
g = []   # the scenario number of each sample

files = sorted(os.listdir(folder))
number = 0
for name in files:
    if not name.endswith(".json"):
        continue
    with open(os.path.join(folder, name)) as f:
        scenario = json.load(f)
    inputs, labels = scenario_to_samples(scenario)
    X += inputs
    y += labels
    g += [number] * len(labels)
    number += 1

X = np.stack(X)
y = np.array(y)
g = np.array(g)

np.savez_compressed("task2/dataset_groups.npz", X=X, y=y, g=g)
print("Saved", X.shape, "from", number, "scenarios")
print("label counts:", np.bincount(y, minlength=5))
