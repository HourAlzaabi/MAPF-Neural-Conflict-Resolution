"""
make_holdout.py  -  carve out the TRUE held-out test set.

The model (task2/train_v3.py) trains on 80% of the scenarios and sets aside 20%
as unseen validation, chosen with numpy seed 0 over the scenario files in sorted
order. But task4/evaluate.py, run on the full wh_dataset, scores the policy on
ALL 1,500 scenarios - including the 80% the model trained on. That makes the
model's success rate optimistically biased (data leakage).

This script reproduces the EXACT same split and copies only the 20% held-out
scenarios into a new folder, so the model can be evaluated on maps it has never
seen. Greedy, CBS and the shield are unaffected by leakage, but we evaluate them
on the same held-out set too, so every number comes from the same maps.

Run from the repo root:
    python make_holdout.py
Then evaluate on the held-out folder, e.g.:
    cd task4
    python evaluate.py --dataset ..\\task1\\wh_holdout --policy model  --model ..\\model.pt
    python evaluate.py --dataset ..\\task1\\wh_holdout --policy model  --model ..\\model.pt --shield
    python evaluate.py --dataset ..\\task1\\wh_holdout --policy greedy
    python evaluate.py --dataset ..\\task1\\wh_holdout --policy cbs
"""
import os
import shutil

import numpy as np

SRC = os.path.join("task1", "wh_dataset")
DST = os.path.join("task1", "wh_holdout")
VAL_FRACTION = 0.20          # must match train_v3.py (last 20% after the shuffle)
SEED = 0                     # must match train_v3.py's np.random.seed(0)


def main():
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".json"))
    n = len(files)
    if n == 0:
        raise SystemExit(f"No scenarios found in {SRC}")

    # Reproduce train_v3.py's split EXACTLY:
    #   scenarios = np.unique(g)          -> array([0, 1, ..., n-1])
    #   np.random.seed(0); np.random.shuffle(scenarios)
    #   val = scenarios[int(0.8 * n):]
    scenarios = np.arange(n)
    np.random.seed(SEED)
    np.random.shuffle(scenarios)
    val_idx = sorted(scenarios[int((1 - VAL_FRACTION) * n):].tolist())

    if os.path.isdir(DST):
        shutil.rmtree(DST, ignore_errors=True)
    os.makedirs(DST, exist_ok=True)
    for i in val_idx:
        shutil.copy(os.path.join(SRC, files[i]), os.path.join(DST, files[i]))

    print(f"Total scenarios in {SRC} : {n}")
    print(f"Held-out (unseen) copied  : {len(val_idx)}  ->  {DST}")
    print("These are the exact maps the model never trained on.")
    print("Now evaluate --policy model on this folder for the honest numbers.")


if __name__ == "__main__":
    main()
