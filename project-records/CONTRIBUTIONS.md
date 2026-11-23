# Member Contributions — Group COSC 11

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Supervisor:** Dr. Khaled Al-Wahedi

Each member owned one task and contributed to the shared report. The breakdown below reflects the work as built, and matches the GitHub commit history (commit author shown in brackets).

## Summary

| Member | ID | GitHub | Task owned | Main deliverables |
|---|---|---|---|---|
| **Hour Alzaabi** | 100062976 | `HourAlzaabi` | Task 1 — Environment & dataset | Warehouse generator, CBS expert, dataset pipeline, parking logic, held-out evaluation set, integration |
| **Meera Alshehhi** | 100062797 | `Meera` | Task 2 — Neural policy | CNN architecture, behavioural-cloning training, guidance tensors |
| **Hoor Almuhsen** | 100062910 | `hoormuhsen` | Task 3 — Safety shield | Priority-reservation shield, veto/substitute logic, test + fuzz suite |
| **Aisha Al Naqbi** | 100063035 | `EngAisha11` | Task 4 — Visualizer & evaluation | Visualizer, evaluation harness, results tables/charts |

## Detailed contributions (with commit evidence)

### Hour Alzaabi — Task 1 (Environment & dataset) · lead integrator
- `warehouse_generator.py` — random warehouse maps (spread / cluster / debris), shelf pick-face goals, A* parking routes, non-blocking parking (never a chokepoint, never an aisle mouth).
- `cbs.py` — Conflict-Based Search expert with A* low-level search.
- `generate_warehouse_dataset.py` + `SCHEMA` — generate + solve + save the 1,500-scenario dataset (seed 42).
- `make_holdout.py` — leak-free held-out test set (300 maps) and the committed held-out IDs.
- Integrated `train_v3.py` (class-weighted loss) and merged the team's branches into `main`.
- Report: Task 1 chapters, Conclusions, Future Work, shared Abstract/Introduction; submission package (forms, title page, README, portable run scripts, master reference).
- *Commits (`HourAlzaabi`):* map generator / CBS / dataset builder (4 Oct), README + dataset (9 Oct), `train_v3` (14 Nov), map flavours + A* parking (17 Nov), shelf-face goals + non-blocking parking + regenerate, branch merges (21 Nov), final model.pt results + demo files (22 Nov).

### Meera Alshehhi — Task 2 (Neural policy)
- `guidance.py` — 4-channel 11×11 observation tensors + distance fields.
- Task 2 CNN training scripts and the first trained model — a 4-layer CNN, behavioural cloning of CBS, class-weighted loss for the rare "Wait", scenario-level 80/20 split.
- Report: policy design (Ch. 4) + training implementation (Ch. 5).
- *Commits (`Meera`):* task2 guidance + training scripts (8 Oct), Task 2 v1 CNN training scripts and model (10 Nov).

### Hoor Almuhsen — Task 3 (Safety shield)
- `shield.py` — priority-reservation shield; vertex + swap conflict checks; veto-and-substitute with Wait fallback.
- `test_shield.py` — adversarial unit tests + 12,000 fuzz steps + speed test.
- Added the `--shield` option to `evaluate.py` (policy → shield → execute); merged branch `hoormuhsen-patch-1` into `main`.
- Report: shield design (Ch. 4) + implementation (Ch. 5).
- *Commits (`hoormuhsen`):* safety shield + tests + README (18 Nov), `--shield` option on evaluate.py (18 Nov).

### Aisha Al Naqbi — Task 4 (Visualizer & evaluation)
- `visualize.py` — 2-D visualizer, later rebuilt (smooth motion, heading arrows, carried crate, legend, clickable controls, light/dark, sound).
- `evaluate.py` — evaluation harness over four policies (greedy / model / cbs) with a `--shield` option.
- Report: visualizer + evaluation design/implementation (Ch. 4–5) and the Evaluation results write-up (Ch. 6.3).
- *Commits (`EngAisha11`):* evaluate.py (16 Nov), visualize.py + README (16 Nov), visualizer update (22 Nov).

---

*Each member's commits are under their own GitHub account, so the repository's contribution graph independently backs up this table.*
