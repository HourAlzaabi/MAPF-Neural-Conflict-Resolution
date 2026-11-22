# Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning

**Senior Design Project I — Group COSC 11** · Supervisor: Dr. Khaled Al-Wahedi
Hour Alzaabi (100062976) · Hoor Almuhsen (100062910) · Aisha Al Naqbi (100063035) · Meera Alshehhi (100062797)

Route several warehouse robots to their shelves with **no collisions**. An optimal
but slow planner (**CBS**) teaches a fast **neural network** to imitate it, and a
**safety shield** guarantees collision-free execution on top of the learned policy.

```
  CBS expert  ──teaches──►  CNN policy  ──proposes──►  Safety shield  ──►  robots execute
   (Task 1)                  (Task 2)                    (Task 3)          (Task 4: visualize + evaluate)
```

---

## 1. Quick start (one file does everything)

**Double-click `run_all.bat`** in this folder and pick an option from the menu:

```
 [1] Full pipeline     rebuild dataset - retrain - evaluate   (slow)
 [2] Evaluate only     honest held-out numbers
 [3] Visualizer only   watch the robots
 [4] Demo              held-out evaluation, then the visualizer
 [5] Shield tests      run the safety unit tests
```

That's all you need. If you'd rather run one specific step without the menu, use one of the shortcuts below.

### Or use a specific shortcut (double-click any of these)

| File | What it does |
|---|---|
| `run_all.bat` | **Menu** — pick any of the steps below from one screen. |
| `demo.bat` | **For the live demo.** Runs the held-out evaluation (greedy / CNN / CNN+shield / CBS), then launches the visualizer. |
| `show_visualizer.bat` | Opens the visualizer only — watch the robots, nothing else. |
| `run_pipeline.bat` | Full rebuild: regenerate dataset → retrain the model → evaluate on held-out maps. Slow. |

All four **auto-detect Anaconda Python** — they look for `python.exe` under your own user folder (`%USERPROFILE%\anaconda3`, then `miniconda3`, then a system-wide install), so they work on any machine, not just one person's. If none is found they print a short message telling you to run from **Anaconda Prompt** or to point the file at your `python.exe`. Nothing is hardcoded to a specific user.

---

## 2. Requirements

- **Python 3** via Anaconda (this project was developed on Anaconda).
- Packages: `numpy`, `torch` (PyTorch), `pygame`. Install once:
  ```
  pip install numpy torch pygame
  ```
- On Windows, if `python` prints *"Python was not found"*, that's the Microsoft Store stub — use **Anaconda Prompt** instead, or run one of the `.bat` files, which find Anaconda for you automatically.

---

## 3. Repository layout

| Path | What it is |
|---|---|
| `task1/warehouse_generator.py` | Random warehouse maps (spread / cluster / debris); shelf-face goals; non-blocking parking. |
| `task1/cbs.py` | Conflict-Based Search expert (A* low-level); solves start → goal → park optimally. |
| `task1/generate_warehouse_dataset.py` | Builds the dataset (generate + solve + save). |
| `task1/wh_dataset/` | 1,500 solved scenarios as JSON (reproducible; **seed = 42**). |
| `task2/guidance.py` | 4-channel 11×11 observation tensors + distance fields. |
| `task2/train_v3.py` | 4-layer CNN trained by behavioural cloning (class-weighted); saves `model.pt`. |
| `task3/shield.py` | Priority-reservation safety shield (`SafetyShield`, `shield_step`, `check_step`). |
| `task3/test_shield.py` | Shield test suite (adversarial cases + 12,000 fuzz steps + speed test). |
| `task4/evaluate.py` | Evaluation harness: greedy / model / cbs, with `--shield`. |
| `task4/visualize.py` | 2-D visualizer (smooth motion, controls, light/dark, sound). |
| `make_holdout.py` | Carves out the 300 held-out test maps (leak-free evaluation). |
| `model.pt` | Trained CNN weights. |
| `holdout_scenario_ids.txt` | The exact 300 held-out scenario IDs (reproduces the reported numbers). |
| `run_all.bat` | **Menu that runs everything.** |
| `demo.bat` / `show_visualizer.bat` / `run_pipeline.bat` | One-click shortcuts (see §1). Portable — auto-detect Anaconda Python. |

---

## 4. Running it manually (what `run_all.bat` does)

```bash
# 1) Build the dataset (reproducible, seed 42)
cd task1
python generate_warehouse_dataset.py --n 1500 --out wh_dataset --seed 42
cd ..

# 2) Train the policy  ->  saves model.pt
python task2/train_v3.py

# 3) Evaluate on the HELD-OUT test set (maps the model never trained on)
python make_holdout.py                 # creates task1/wh_holdout (300 maps)
cd task4
python evaluate.py --dataset ../task1/wh_holdout --policy greedy
python evaluate.py --dataset ../task1/wh_holdout --policy model --model ../model.pt
python evaluate.py --dataset ../task1/wh_holdout --policy model --model ../model.pt --shield
python evaluate.py --dataset ../task1/wh_holdout --policy cbs

# 4) Safety tests for the shield
python -m pytest ../task3 -q

# 5) Watch it
python visualize.py ../task1/wh_dataset
```

**Visualizer controls:** `N`/`P` next/prev scenario · `SPACE` play/pause · `←`/`→` step · `T` theme · `M` mute · `+`/`−` speed · `Q` quit.

---

## 5. Results (held-out test set — maps the model never trained on)

| Policy | Fleet success | Collisions | Flowtime vs CBS |
|---|---|---|---|
| Greedy (naive baseline) | 77.0% | 100 | 1.00× |
| Neural network (CNN) | 82.7% | 69 | 1.00× |
| **Neural + Safety shield** | **90.0%** | **0** | 1.28× |
| CBS (optimal expert) | 100% | 0 | 1.00× |

Training quality: **98.5% move accuracy, 99.3% Wait recall**.

**Why "held-out":** the model trains on 80% of the scenarios and 20% are held out. The
model numbers above are measured only on the held-out 300 maps (`make_holdout.py`
reproduces the seed-0 split; the exact IDs are in `holdout_scenario_ids.txt`), so there
is **no data leakage**. Greedy and CBS have no training, so they are leakage-free either
way, and the shield's zero-collision guarantee holds on any map.

---

## 6. Reproducibility
- Dataset is seeded (`SEED = 42`) — everyone gets the identical 1,500 maps.
- The held-out split is seeded (numpy seed 0, 80/20 by scenario) — the reported numbers
  reproduce to the decimal. The exact 300 held-out IDs are committed in
  `holdout_scenario_ids.txt`.

## 7. Status
Task 1 (dataset) · Task 2 (neural policy) · Task 3 (safety shield) · Task 4 (visualizer + evaluation) — **complete**.
See `CHANGES.md` for the full change log.
