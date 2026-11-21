# MAPF Project — Summary of Changes

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Owner of these changes:** Hour (Task 1) · **Date:** 22 Nov 2026

A plain-English rundown of what was updated across the project and *why*, so everyone is on the same page.

---

## Task 1 — Dataset generator (`task1/warehouse_generator.py`)

**What changed**
- **Robots now back off after picking.** When a robot reaches its shelf (its goal) and "picks", it no longer freezes there — it steps off the shelf face to an open-floor staging cell and parks.
- **Goals are now shelf pick-faces**, not random open floor. Every goal is a cell that touches a shelf, so each task reads as "drive to this shelf and pick something."
- **Parking cells are guaranteed not to block anyone.** A rest spot is never a chokepoint (the only link between two areas) and is never at an aisle entrance (next to a 1-wide corridor). Robots always rest in genuinely open floor.

**Why**
- Supervisor feedback: a finished robot must not sit in an aisle and block the others. Now it clears the pick-face and parks out of the way, becoming a static obstacle the fleet simply routes around.
- Random goals looked arbitrary. Shelf-face goals model real warehouse picking and make robots converge on shelves, which creates more realistic conflicts for the network to learn from.
- "Open on all four sides" alone wasn't enough — a cell can still seal an aisle mouth. The stricter rule removes that case entirely.

**Result:** 1,500 maps, density 0.185 (within the 10–25% spec), ~26% with conflicts. Validated: **0 parks on a chokepoint, 0 at an aisle mouth**, and CBS solves **100%**.

---

## Task 2 — Neural policy (`task2/train_v3.py`, `task2/guidance.py`)

**What changed**
- The observation tensors switch the robot's target from **goal → parking cell** per step, so the network learns the back-off leg correctly.
- **Class-weighted loss** (the rare "Wait" action weighted ~2.8×) so the model doesn't ignore yielding.
- **Retrained** on the regenerated dataset.

**Why**
- The new dataset changed the trajectories (added the back-off legs), so the model had to be retrained to match.
- Without class weighting the model coasts on the 99% of easy "keep going" moves and never learns to wait — the weighting fixes that.

**Result:** **98.5% move accuracy, 99.3% Wait recall** (the model almost never misses a required yield).

---

## Task 3 — Safety shield (merged from branch `hoormuhsen-patch-1`)

**What changed**
- Merged the safety-shield branch into `main` (brings `task3/` and the `--shield` option in `evaluate.py`).
- Re-ran the shield on the new dataset.

**Why**
- The shield is the safety net: it vetoes any move that would cause a collision and substitutes a safe one, guaranteeing collision-free execution on top of the neural policy.

**Result:** **0 collisions** on all 1,500 maps, ~0.04 ms per step (real-time).

---

## Task 4 — Visualizer (`task4/visualize.py`)

**What changed**
- Rebuilt to look polished and professional (clean board, raised shelves, hazard-tile debris, goal rings, start markers).
- **Smooth gliding motion**, **heading arrows** (clear direction), a **crate carried after picking**, moving shadows, anti-aliasing.
- **Fading trails** that melt into the floor.
- Side panel with **live fleet status, statistics, and a legend**.
- **Clickable control bar** + crisp single-press stepping.
- **Light / dark mode**, a **speed stepper**, and **sound effects** (mute with M).
- Launches a **random sample of 20 scenarios** each run (N / P to flip between them).

**Why**
- It needs to be clear and demo-ready. It plays back the real dataset so it visibly shows the full behavior: drive to shelf → pick → back off → park, and the shield making robots yield.

---

## Pipeline helper (`run_pipeline.bat`)

- One file that runs the whole thing in the correct order — **generate dataset → train policy → benchmark** — using the right Python. Avoids order/path mistakes.

---

## Final results (1,500 warehouse maps)

| Policy | Fleet success | Collisions | Flowtime vs CBS |
|---|---|---|---|
| Greedy (baseline) | 73.9% | 558 | 1.00× |
| Neural policy (CNN) | **92.8%** | 228 | 1.00× |
| Neural + Safety Shield | **95.1%** | **0** | 1.14× |
| CBS (expert) | 100% | 0 | 1.00× |

**Takeaway:** the neural policy nearly matches the perfect expert and beats the naive baseline by ~19 points, and the shield guarantees **zero collisions** (while even nudging success up).

---

## Still to do

- **Push:** Task 1 files (generator + `wh_dataset` + retrained `model.pt`) — Hour. `task4/visualize.py` — Aisha.
- Final report + slide deck (in progress).
