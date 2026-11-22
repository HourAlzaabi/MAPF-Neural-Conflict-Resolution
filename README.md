# Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning

Group COSC 11 · Senior Design Project I · Supervisor: Dr. Khaled Al-Wahedi

Route several warehouse robots to their shelves with **no collisions**. An optimal
but slow planner (**CBS**) teaches a fast **neural network** to imitate it; a
**safety shield** then guarantees collision-free execution.

```
CBS expert  ──teaches──►  CNN policy  ──proposes──►  Safety shield  ──►  robots execute
 (Task 1)                  (Task 2)                    (Task 3)          (Task 4: visualize + evaluate)
```

## Repository layout
| Path | What it is |
|---|---|
| `task1/warehouse_generator.py` | Random warehouse maps (spread / cluster / debris); shelf-face goals; non-blocking parking. |
| `task1/cbs.py` | Conflict-Based Search expert (A* low-level), solves start → goal → park optimally. |
| `task1/generate_warehouse_dataset.py` | Builds the dataset: generate + solve + save. |
| `task1/wh_dataset/` | 1,500 solved scenarios as JSON (reproducible; **seed = 42**). |
| `task2/guidance.py`, `task2/train_v3.py` | 4-channel observation tensors; 4-layer CNN trained by behavioural cloning (class-weighted). |
| `task3/shield.py`, `task3/test_shield.py` | Priority-reservation safety shield + test suite. |
| `task4/evaluate.py` | Evaluation harness (greedy / model / cbs, with `--shield`). |
| `task4/visualize.py` | 2-D visualizer (smooth motion, controls, light/dark, sound). |
| `make_holdout.py` | Carves out the 300 held-out test maps (leak-free evaluation). |
| `model.pt` | Trained CNN weights. |
| `holdout_scenario_ids.txt` | The exact 300 held-out scenario IDs (reproduces the reported numbers). |

## How to run
```bash
# 1. Build the dataset (reproducible, seed 42)
cd task1
python generate_warehouse_dataset.py --n 1500 --out wh_dataset --seed 42
cd ..

# 2. Train the policy
python task2/train_v3.py            # saves model.pt

# 3. Evaluate on the HELD-OUT test set (maps the model never trained on)
python make_holdout.py              # creates task1/wh_holdout (300 maps)
cd task4
python evaluate.py --dataset ../task1/wh_holdout --policy greedy
python evaluate.py --dataset ../task1/wh_holdout --policy model  --model ../model.pt
python evaluate.py --dataset ../task1/wh_holdout --policy model  --model ../model.pt --shield
python evaluate.py --dataset ../task1/wh_holdout --policy cbs

# 4. See it
python visualize.py ../task1/wh_dataset
```
Windows one-click: `run_pipeline.bat` (full pipeline) · `demo.bat` (held-out eval + visualizer) · `show_visualizer.bat`.

## Results (held-out test set — maps the model never trained on)
| Policy | Fleet success | Collisions | Flowtime vs CBS |
|---|---|---|---|
| Greedy (baseline) | 77.0% | 100 | 1.00× |
| Neural network (CNN) | 82.7% | 69 | 1.00× |
| Neural + Safety shield | **90.0%** | **0** | 1.28× |
| CBS (optimal expert) | 100% | 0 | 1.00× |

Training: 98.5% move accuracy, 99.3% Wait recall. Model numbers are reported on the
300 held-out maps (seed-0 80/20 split) to avoid data leakage; greedy and CBS are
leakage-free and the shield's 0-collision guarantee holds on any map.

## Status
Task 1 (dataset), Task 2 (policy), Task 3 (shield), Task 4 (visualizer + evaluation) — complete.
