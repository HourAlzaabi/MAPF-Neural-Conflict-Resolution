# Task 3 — Safety Shield

> The shield is a reactive rule layer that guarantees collision-free execution by
> vetoing any proposed move that would cause a vertex or swap conflict and
> replacing it with a safe fallback.

```
CBS (Task 1) -> teaches -> CNN policy (Task 2) -> proposes moves
                                   |
                         +-------------------+
                         |  SAFETY SHIELD    |  <- Task 3
                         | (veto + fallback) |
                         +-------------------+
                                   |  guaranteed-safe moves
                                   v
                            robots execute
```

The shield can wrap any policy: the CNN, the greedy baseline or CBS-replay.

## Files

| File | What it is |
|---|---|
| `shield.py` | `SafetyShield` class + `shield_step()` (Section 3 contract) + `check_step()` independent collision checker |
| `../task4/evaluate.py --shield` | Integration: runs policy → shield → execute and reports success, collisions, deadlocks, makespan, flowtime and shield time |
| `test_shield.py` | Hand-made adversarial cases + randomized fuzz + speed test |

## Interface (contract)

```python
from task3.shield import SafetyShield, shield_step

safe = shield_step(positions, proposed, walls, width, height)
# positions: {robot_id: (x, y)}     proposed: {robot_id: "N"|"S"|"E"|"W"|"Wait" or 0..4}
# walls: set of (x, y) — MUST include debris
# returns {robot_id: action}, guaranteed collision-free for step t -> t+1
```

Conventions match Tasks 1/2: `N=(0,-1) S=(0,1) E=(1,0) W=(-1,0) Wait=(0,0)`,
indices `0=N 1=S 2=E 3=W 4=Wait`, `(x, y)` = (column, row), `(0,0)` top-left.
A robot with no proposal (e.g. parked) is treated as `Wait` and its cell stays
an obstacle for the others.

Options: `SafetyShield(walls, w, h, priority="id"|"rotate"|"distance"|"random", fallback="wait"|"alternative")`.

## Safety invariants (guaranteed every step)

1. **No wall/boundary violation:** no robot moves into an obstacle (including debris) or off the grid.
2. **No vertex conflict:** no two robots occupy the same cell at t+1.
3. **No swap conflict:** no two robots trade cells between t and t+1.
4. **Always returns:** every robot gets a move, and `Wait` is the universal fallback.

On a 4-connected grid, vertex and swap conflicts are the only two collision types (Stern et al., 2019).

## Algorithm: priority-reservation shield

All decisions are made against the snapshot at time t; positions are never
mutated while deciding.

1. **Pass 1:** every robot tentatively stays, so each current cell is reserved.
2. **Pass 2:** go through the robots in priority order (by `robot_id`). Robot *i* takes its proposed move only if the target cell is in-bounds, not a wall or debris, and **not reserved**. If it moves, it releases its old cell and reserves the new one. Pass 2 is **repeated until nothing changes**. This way a follower checked before the robot ahead of it gets a second chance once that robot has moved off (the "chase" case). Without the repeat, the shield needlessly blocked 14% of the perfectly safe CBS replays.
3. **Fallback:** with `fallback="alternative"` (the default), a vetoed robot may take another legal step that does not move it further from its current target (goal, then parking cell). Otherwise it Waits.

**Why it is safe (proof sketch)**
- *Vertex:* a robot only enters an unreserved cell and reserves it, so each cell is held by at most one robot. No two robots end on the same cell.
- *Swap:* for *i* to enter *j*'s cell, *j* must already have released it by moving to a cell that was unreserved at that time. *i*'s own cell was still reserved by *i* then, so *j* did not move into it. No trade is possible.
- *Walls/boundary:* explicitly rejected (`is_free`).
- *Always returns:* every robot starts as `Wait`, and staying put is safe because its cell stays reserved until it moves.

Cost: O(robots²) worst case per step for the repeated sweep. That is trivial for 3 robots, and there is no search.

## Results on the real Task 1 dataset

These numbers come from `task1/wh_dataset`: 1,500 maps, seed 42, start → goal → parking. They were produced with `task4/evaluate.py`:

```bash
cd task4
python evaluate.py --policy greedy                 # unshielded baseline
python evaluate.py --policy greedy --shield        # R1
python evaluate.py --policy cbs --shield           # shield must not hurt a safe policy
python evaluate.py --policy model --model ../model.pt --shield   # R2
```

| Policy | Success | Collisions | Deadlock | Makespan vs CBS | Flowtime vs CBS | Shield time/step |
|---|---|---|---|---|---|---|
| Greedy, no shield | 64.7% | **729** | – | 1.00× | 1.00× | – |
| Greedy + shield (id / wait) | 70.1% | **0** | 29.9% | 1.95× | 1.75× | 0.005 ms |
| **Greedy + shield (id / alternative)** (default) | **79.4%** | **0** | 20.6% | 1.65× | 1.50× | 0.009 ms |
| Greedy + shield (rotate / alternative) | 79.1% | **0** | 20.9% | 1.66× | 1.51× | 0.010 ms |
| CBS replay, no shield | 100% | 0 | – | 1.00× | 1.00× | – |
| CBS replay + shield (any setting) | 100% | 0 | 0.0% | 1.00× | 1.00× | ≤ 0.005 ms |
| CNN model + shield | *to run* | | | | | |

How to read the table:
- **R1 is met.** Greedy collides 729 times on its own and **0 times** with the shield, across all 1,500 maps and every shield setting.
- **R6 is met.** Success goes from 64.7% to 79.4% with the shield. The cost is longer paths (flowtime 1.50× CBS), because blocked robots wait or detour.
- **The shield adds nothing to a safe policy.** Wrapping the collision-free CBS replay changes nothing: there are 0 vetoes, 100% success and 1.00× cost.
- **R5 is met.** The worst scenario averages under 0.1 ms per step, against a sub-millisecond target.
- **R7, deadlocks.** These are counted when no robot moves for 10 steps. The 20.6% for greedy comes mostly from greedy itself: it only ever walks straight downhill, so two greedy robots that want each other's corridor freeze. The shield keeps them from colliding, but cannot make them plan. This is reported, not hidden.
- **R2 is still to run.** It needs PyTorch and `model.pt`; run the command above.

## Known limitations

- **Deadlock / livelock:** the shield is conservative. A cycle (A→B, B→C, C→A) is frozen because every target cell is occupied. Greedy deadlocks often because it never steps sideways. Rotating the priority order and using the alternative-move fallback both reduce this. The deadlock rate is **measured and reported**, not hidden.
- **Follow moves:** these are allowed thanks to the repeated sweep, but a whole "train" may only close up if its front robot actually moves.
- **Parked robots are static obstacles:** a parking cell in a narrow aisle could deadlock other robots. Task 1 places parking cells in open floor, but the shield does not assume this.
- **Throughput cost:** forced waits can lengthen paths. That is why makespan and flowtime are reported with and without the shield.
- **Safety, not liveness:** the shield is reactive and never re-plans. Finishing is the policy's job; not colliding is the shield's job.

## Tests

```bash
python -m pytest task3 -q      # or: python task3/test_shield.py
```

The tests cover head-on (corridor and gap), cross, chase/follow (including a 3-robot train), corner swap, walls/boundary, parked robot, a 2×2 rotation deadlock, the alternative fallback, index actions, and 12,000 randomized fuzz steps across all priority/fallback modes. Every fuzz step is checked by the independent `check_step`. A speed test asserts under 1 ms per step.
