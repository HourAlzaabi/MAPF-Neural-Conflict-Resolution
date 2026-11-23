# MAPF Project — Summary of Changes

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Owner of these changes:** Hour (Task 1) · **Updated:** 23 Nov 2026

A plain-English rundown of what was built and why.

## Evaluation methodology (important)
The model trains on 80% of the scenarios; 20% are held out. All **model** success
numbers below are measured on the **held-out 300 maps the model never trained on**,
to avoid data leakage. Greedy and CBS have no training, so they are leakage-free
either way; the shield's zero-collision guarantee holds on any map.

## Final results (held-out test set)
| Policy | Fleet success | Collisions | Flowtime vs CBS |
|---|---|---|---|
| Greedy (naive baseline) | 77.0% | 100 | 1.00× |
| Neural network (CNN) | 82.7% | 69 | 1.00× |
| Neural + Safety shield | 90.0% | 0 | 1.28× |
| CBS (optimal expert) | 100% | 0 | 1.00× |

Training quality: **98.5% move accuracy, 99.3% Wait recall** (on held-out maps).
Shield: ~0.05 ms/step; deadlock rate 10% (measured and reported).
Dataset: 1,500 scenarios, seed 42, density 0.185, ~26% with conflicts; 300 held out for evaluation.

## Task 1 — Dataset generator
- Every robot backs off after picking and parks in open floor (never a chokepoint, never an aisle mouth), so it can't block a lane.
- Goals are shelf pick-faces (a cell touching a shelf), not random floor.
- Reproducible via seed 42; every map checked for density, connectivity and solvability (CBS) before acceptance.

## Task 2 — Neural policy
- 4-layer CNN; 4-channel 11×11 local view (walls / ego / others / guidance).
- Behavioural cloning of CBS; class-weighted loss (Wait ≈ 2.8×) for the rare yield.
- Scenario-level 80/20 split (no leakage); retrained on the final dataset.

## Task 3 — Safety shield
- Priority-reservation shield; vetoes vertex/swap conflicts, substitutes a safe move.
- Guarantees 0 collisions for any policy; ~0.05 ms/step; 12,000 fuzz tests pass.
- Merged into main.

## Task 4 — Visualizer & evaluation
- Visualizer: smooth motion, heading arrows, carried crate, legend, controls, light/dark, sound.
- evaluate.py: four policies, with a `--shield` option; `make_holdout.py` carves out the held-out test set.
- `demo.bat` runs the held-out evaluation then the visualizer.

## Reproducing results
Run `run_pipeline.bat` (regenerate → retrain → evaluate) or `demo.bat` (held-out eval + visualizer).
