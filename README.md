# MAPF — Conflict Resolution via Neural Imitation Learning

Senior Design Project (SDPI), Khalifa University.
Supervisor: Dr. Khaled Al-Wahedi.

## What this is

Three robots move across a warehouse grid (15×15–20×20, with walls). Each must
reach its goal without colliding. The system combines a classical optimal solver
with a learned policy:

- **Expert (offline):** Conflict-Based Search (CBS) computes optimal,
  collision-free paths.
- **Policy (real-time):** a small CNN is trained by behavioural cloning to imitate
  CBS, deciding each robot's next move (N / S / E / W / Wait) from a local view.
- **Safety shield:** a rule-checker blocks any move that would cause a collision.

## Repository layout

| Path | What it is |
|---|---|
| `SCHEMA.md` | The agreed data schema (JSON scenarios + tensor format). **Read this first.** |
| `task1/grid_generator.py` | Random maps with a connectivity check. |
| `task1/cbs.py` | Conflict-Based Search solver. |
| `task1/generate_dataset.py` | Builds the dataset of solved scenarios. |
| `task1/check_dataset.py` | Validates every scenario (collisions, actions, goals). |
| `task1/actions.py` | Turns a path into per-step action labels (N/S/E/W/Wait). |
| `task1/view_scenario.py` | Prints a scenario as a text animation. |
| `task1/dataset/` | 1,500 solved scenarios as JSON (regenerable; seed = 42). |

## Running Task 1

Requires Python 3.10+ (standard library only — no packages to install).

```
cd task1
python generate_dataset.py    # builds task1/dataset (~1500 files)
python check_dataset.py        # should print "ALL GOOD"
python view_scenario.py map_0586
```

The generator is seeded (`SEED = 42`), so everyone gets the identical dataset.

## Tasks

1. Map generator, CBS solver, dataset — **complete**.
2. Distance fields, tensor extraction, CNN training.
3. Safety shield (vertex + swap conflict checks, fallback).
4. Visualizer and evaluation metrics.

## Use of AI

AI assistance (Anthropic's Claude) was used as a tool during this project to help
refine code. All AI-assisted work was reviewed, tested, and verified by the team, who take full
responsibility for the contents of this repository. AI use is documented in line
with the course's academic-integrity requirements; see the project report's
AI-use disclosure for the full account.
