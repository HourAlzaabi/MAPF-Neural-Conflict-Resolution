# Member Contributions — Group COSC 11

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Supervisor:** Dr. Khaled Al-Wahedi

Grades for this project are awarded individually. This document records each member's own contribution to the **system (code)** and to the **report**, consistent with the commit history on GitHub (commit author shown in brackets).

*The report and project-record documents are the whole team's collaborative work. Where these shared documents were committed and pushed through Hour's account, that reflects Hour acting as the submitting representative on the team's behalf, not sole authorship; each member's own task code is committed under their own GitHub account.*

## Summary

| Member | ID | GitHub | Task owned | System / code built | Report sections written |
|---|---|---|---|---|---|
| **Hour Alzaabi** | 100062976 | `HourAlzaabi` | Task 1 — Environment & dataset | Warehouse generator, CBS expert, dataset pipeline, non-blocking parking, held-out evaluation set, integration | Abstract; Ch. 1 (incl. §1.2 literature & patent review, §1.4); Ch. 3 (theoretical model); §4.1–4.5 & §5.1 (Task 1); §6.1–6.2; Ch. 7 (conclusions & future work) |
| **Meera Alshehhi** | 100062797 | `Meera` | Task 2 — Neural policy | CNN architecture, behavioural-cloning training, class-weighted loss | §4.6 (policy design), §5.2 (training implementation) |
| **Hoor Almuhsen** | 100062910 | `hoormuhsen` | Task 3 — Safety shield | Priority-reservation shield, veto/substitute logic, test & fuzz suite | §4.7 (shield design), §5.3 (shield implementation) |
| **Aisha Al Naqbi** | 100063035 | `EngAisha11` | Task 4 — Visualizer & evaluation | Visualizer (rebuilt for the demo), evaluation harness, results analysis | §4.8 (design), §5.4 (implementation), §6.3 (evaluation) |

## Detailed contributions (with commit evidence)

### Hour Alzaabi — Task 1 (Environment & dataset) · lead integrator
- `warehouse_generator.py` — random warehouse maps (spread / cluster / debris), shelf pick-face goals, A* parking routes, non-blocking parking (never a chokepoint, never an aisle mouth).
- `cbs.py` — Conflict-Based Search expert with A* low-level search.
- `generate_warehouse_dataset.py` + `SCHEMA` — generate + solve + save the 1,500-scenario dataset (seed 42).
- `make_holdout.py` — leak-free held-out test set (300 maps) and the committed held-out IDs.
- Integrated `train_v3.py` (class-weighted loss) and merged the team's branches into `main`.
- Report: Abstract, Chapter 1 (incl. literature review), Chapter 3, Task 1 chapters, verification/validation, Conclusions, Future Work; submission package (forms, title page, README, portable run scripts).
- *Commits (`HourAlzaabi`):* map generator / CBS / dataset builder (4 Oct), README + dataset (9 Oct), `train_v3` (14 Nov), map flavours + A* parking (17 Nov), shelf-face goals + non-blocking parking + regenerate, branch merges (21 Nov), final model.pt results + demo files (22 Nov).

### Meera Alshehhi — Task 2 (Neural policy)
- `guidance.py` — 4-channel 11×11 observation tensors + distance fields.
- Task 2 CNN training scripts and the trained model — a 4-layer CNN, behavioural cloning of CBS, class-weighted loss (Wait ≈ 2.83), scenario-level 80/20 split.
- Report: policy design (§4.6) and training implementation (§5.2).
- *Commits (`Meera`):* task2 guidance + training scripts (8 Oct), Task 2 CNN training scripts and model (10 Nov).

### Hoor Almuhsen — Task 3 (Safety shield)
- `shield.py` — priority-reservation shield; vertex + swap conflict checks; veto-and-substitute with Wait fallback.
- `test_shield.py` — adversarial unit tests + 12,000 fuzz steps + speed test.
- Added the `--shield` option to `evaluate.py`; merged branch `hoormuhsen-patch-1` into `main`.
- Report: shield design (§4.7) and implementation (§5.3).
- *Commits (`hoormuhsen`):* safety shield + tests + README (18 Nov), `--shield` option on evaluate.py (18 Nov).

### Aisha Al Naqbi — Task 4 (Visualizer & evaluation)
- `visualize.py` — 2-D visualizer, **rebuilt for the live demo**: smooth eased motion, heading arrows, a carried crate on pick, a strengthened fading trail, a plain uncluttered floor, a side panel with per-robot status and a legend, clickable + keyboard controls with crisp single-press stepping, adjustable speed, light/dark mode, sound effects, 20-scenario browsing, and a `--gif` export for figures.
- `evaluate.py` — evaluation harness over four policies (greedy / model / cbs) with a `--shield` option; the four metrics (success / collisions / flowtime / makespan) and the CBS-replay sanity check.
- Report: visualizer + evaluation design/implementation (§4.8, §5.4) and the evaluation results write-up (§6.3).
- *Commits (`EngAisha11`):* evaluate.py (16 Nov), visualize.py + README (16 Nov), visualizer update (22 Nov).

Each member's commits are under their own GitHub account, so the repository's contribution graph independently backs up this table.
