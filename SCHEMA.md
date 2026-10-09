# MAPF Project — Data Schema (agreed)

This is the single source of truth for the data formats every task builds
against. It is the schema produced by the Task 1 code, adopted by the team as
the agreed project schema. Do not change it without a team decision; if it
changes, update this file in the same commit.

---

## 1. Coordinate convention

Every cell is written `[x, y]`:

- `x` = column, counted from the **left** (0 = left edge)
- `y` = row, counted from the **top** (0 = top edge)

So `[0, 0]` is the **top-left** corner. Moving North makes `y` smaller.

This convention is used everywhere: obstacles, starts, goals, and trajectories.

---

## 2. JSON scenario format

One solved scenario per file, named `map_XXXX.json` (zero-padded, e.g.
`map_0000.json`). Produced by `generate_dataset.py`, verified by
`check_dataset.py`.

```json
{
  "scenario_id": "map_0586",
  "grid": {
    "width": 15,
    "height": 15,
    "obstacle_density": 0.133,
    "obstacles": [[0, 2], [0, 5], [2, 3]]
  },
  "agents": [
    {"id": 0, "start": [11, 2], "goal": [11, 12]},
    {"id": 1, "start": [14, 12], "goal": [8, 3]},
    {"id": 2, "start": [10, 8], "goal": [14, 11]}
  ],
  "trajectories": {
    "0": [[11, 2], [11, 3], [12, 3]],
    "1": [[14, 12], [13, 12], [12, 12]],
    "2": [[10, 8], [11, 8], [11, 9]]
  },
  "actions": {
    "0": ["S", "E"],
    "1": ["W", "W"],
    "2": ["E", "S"]
  },
  "makespan": 15,
  "sum_of_costs": 37,
  "had_conflict": true,
  "solver": "CBS",
  "optimal": true
}
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `scenario_id` | string | File identifier, matches the filename stem. |
| `grid.width`, `grid.height` | int | Grid size in cells (square: 15–20). |
| `grid.obstacle_density` | float | Fraction of cells that are walls (rounded). |
| `grid.obstacles` | list of `[x, y]` | Wall cells. |
| `agents` | list | One object per robot. |
| `agents[].id` | int | Robot id, `0`-based, matches the keys in `trajectories`. |
| `agents[].start` | `[x, y]` | Starting cell. |
| `agents[].goal` | `[x, y]` | Goal cell. |
| `trajectories` | object | Key = agent id **as a string**; value = list of `[x, y]` cells, one per time step. |
| `actions` | object | Key = agent id as a string; value = list of actions (`N`/`S`/`E`/`W`/`Wait`), one per step. This is the CNN's training label: the move CBS took at each step, including waiting to resolve a conflict. Length = trajectory length − 1 = `makespan`. |
| `makespan` | int | Steps until the last robot arrives (= trajectory length − 1). |
| `sum_of_costs` | int | Sum over robots of (trajectory length − 1); CBS minimises this. |
| `had_conflict` | bool | True if CBS had to resolve a conflict between robots. |
| `solver` | string | Always `"CBS"` for this dataset. |
| `optimal` | bool | True — CBS solutions are optimal by sum of costs. |

### Invariants (checked by `check_dataset.py`)

1. Every robot starts on its `start` and ends on its `goal`.
2. All `trajectories` lists have the **same length**. A robot that arrives
   early **repeats its goal cell** to pad to full length.
3. Consecutive cells differ by at most one step (up/down/left/right) or are
   equal (a wait). No diagonal moves.
4. No cell is a wall, and no cell is outside the grid.
5. No **vertex conflict**: two robots never occupy the same cell at the same
   time step.
6. No **swap conflict**: two robots never exchange cells between two
   consecutive steps.
7. `actions` reconstruct `trajectories` exactly: applying each action to a
   robot's cell reproduces its next cell.

### Action labels (how a conflict is resolved)

The resolution of a conflict is encoded in `actions`: a robot that yields
shows up as a `Wait` at that step. For example, if robot 2 waited one step to
let robot 1 pass, `actions["2"]` contains a `Wait` at that index while the
other robots move. The action order used by the CNN output is fixed:

```
N, S, E, W, Wait
```

where, in `[x, y]` coordinates, `N` = y−1, `S` = y+1, `E` = x+1, `W` = x−1.

**Note for training:** once a robot reaches its goal it stays there, so its
trailing actions are `Wait`. These "parked" waits are different from a yield;
mask out steps after a robot has arrived if you do not want the model to learn
them. The helper `actions.py` (`trajectory_to_actions`) produces these labels
from a path and is used by both `generate_dataset.py` and `check_dataset.py`.

---

## 3. Tensor format (CNN input, Task 2)

For each agent at each time step, the CNN receives a `(4, 11, 11)` tensor: four
stacked `11 x 11` grids, centred on that agent. Cells outside the map are
treated as walls.

| Channel | Contents |
|---|---|
| 0 | Local obstacle mask (walls; off-grid = wall). |
| 1 | Ego-agent position (the centre pixel). |
| 2 | Other agents inside the window. |
| 3 | Distance-field guidance toward this agent's goal. |

**CNN output:** 5 action probabilities, in this order — `N, S, E, W, Wait`.

North/South/East/West follow the coordinate convention above (North = `y`
decreases). The training label for each example is the action CBS took at that
step.

---

## 4. Open points (to confirm, then delete from this list)

- Padding rule once more than 3 agents are used (same repeat-goal rule assumed).
- Whether obstacle density stays in the 0.10–0.20 range if maps are made denser
  to produce more conflicts.

_Last updated: 2026-10-08._
