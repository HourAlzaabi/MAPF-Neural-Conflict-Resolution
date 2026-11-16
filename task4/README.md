# Task 4 — Visualizer & Evaluation

Two deliverables, both runnable against the Task 1 dataset. Task 4 does **not**
depend on Task 3 — it can be built and committed on its own.

## Files

| File | What it does |
|---|---|
| `visualize.py` | 2D playback of a scenario: grid, walls, robot trails, start/goal markers, 11×11 FOV, wait-highlights, collision flashes. Can also save an animated GIF. |
| `evaluate.py` | Runs a policy over the dataset and reports fleet success rate, makespan, flowtime, and collisions vs the CBS optimum. |

## Setup

```
pip install pygame numpy imageio      # imageio only for GIF export; numpy/torch for the model policy
```

## Visualizer

```
python visualize.py ../task1/dataset/map_0032.json           # play the CBS solution
python visualize.py ../task1/dataset/map_0032.json --fov 0   # show robot 0's 11×11 view
python visualize.py ../task1/dataset/map_0032.json --policy greedy   # watch the baseline collide
python visualize.py ../task1/dataset/map_0032.json --gif run.gif     # save a GIF for the report/slides
```

Controls: **SPACE** play/pause · **→ / ←** step · **R** restart · **+ / −** speed · **Q** quit.

What you see: fading **trails** show where each robot has been; a hollow **square**
marks each start, a **ring** marks each goal (filled once reached); an **amber ring**
means a robot is **waiting** (yielding); a **red flash** marks a collision.

## Evaluation

```
python evaluate.py --dataset ../task1/dataset --policy cbs      # sanity check
python evaluate.py --dataset ../task1/dataset --policy greedy   # no-coordination baseline
python evaluate.py --dataset ../task1/dataset --policy model --model ../model.pt
```

Expected on a correct build:

- **`cbs`** replays the stored optimal paths → **100% success, 0 collisions, 1.00× makespan/flowtime.** If this isn't exactly 1.00×/100%, the harness has a bug.
- **`greedy`** rolls each robot downhill ignoring the others → it reaches goals but **collides** (~20–25% of scenarios fail). That gap is what the CNN (Task 2) and the safety shield (Task 3) exist to close.
- **`model`** evaluates Meera's CNN. It loads `model.pt` (a state_dict → rebuilds `PolicyCNN`) and builds the input tensor with **Task 2's `guidance.build_observation`**, so the model sees exactly what it was trained on.

## Metrics (from the proposal)

- **Fleet Success Rate** — every robot reaches its goal with no collision.
- **Makespan** — steps until the last robot arrives (reported as a ratio to CBS).
- **Flowtime** — sum over robots of steps taken (ratio to CBS `sum_of_costs`).
- **Collisions** — vertex + swap conflicts the policy would have caused.

## Later: plugging in Task 3

When the safety shield exists, it wraps any policy: the policy proposes a move,
the shield vetoes it if it would collide and substitutes a safe one. `evaluate.py`
can then gain a `--shield` flag to measure the shielded policy. No change needed
to the visualizer.
