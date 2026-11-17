"""guidance.py - Student 2: distance fields + 4x11x11 observation tensors.

UPDATED for the parking schema (Task 1): a robot's route is start -> goal ->
parking cell. Its single-agent guidance must therefore point at whatever it is
currently heading for: the GOAL until it has reached it, then the PARKING cell.
If the guidance kept pointing at the goal during the parking leg, channel 3 would
disagree with the expert action and confuse the network. We switch the distance
field per time step accordingly. Scenarios with no "park" field fall back to the
old behaviour (target = goal throughout).
"""
import json
import os
from collections import deque

import numpy as np

FOV = 11
HALF = FOV // 2  # 5
# Same order as Student 1's STEPS: N, S, E, W  (x = column, y = row from top)
MOVES = [(0, -1), (0, 1), (1, 0), (-1, 0)]
UNREACHABLE = -1


def distance_field(goal, width, height, obstacles):
    """Backward BFS from a target. Returns int array D[y, x]; walls/unreachable = -1."""
    D = np.full((height, width), UNREACHABLE, dtype=np.int32)
    gx, gy = goal
    D[gy, gx] = 0
    q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height
                    and (nx, ny) not in obstacles and D[ny, nx] == UNREACHABLE):
                D[ny, nx] = D[y, x] + 1
                q.append((nx, ny))
    return D


def action_label(cur, nxt):
    """0=N, 1=S, 2=E, 3=W, 4=Wait."""
    d = (nxt[0] - cur[0], nxt[1] - cur[1])
    if d == (0, 0):
        return 4
    return MOVES.index(d)


def build_observation(i, positions, D, width, height, obstacles):
    """4x11x11 tensor for agent i. positions = list of (x,y) at time t.
    D is the distance field toward agent i's CURRENT target (goal or parking cell)."""
    obs = np.zeros((4, FOV, FOV), dtype=np.float32)
    cx, cy = positions[i]
    d_ego = D[cy, cx]
    for row in range(FOV):
        for col in range(FOV):
            x, y = cx + col - HALF, cy + row - HALF
            if not (0 <= x < width and 0 <= y < height) or (x, y) in obstacles:
                obs[0, row, col] = 1.0              # wall / debris / outside map
            elif D[y, x] != UNREACHABLE:
                # positive = closer to target than ego; scaled to [-1, 1]
                obs[3, row, col] = np.clip(d_ego - D[y, x], -5, 5) / 5.0
    obs[1, HALF, HALF] = 1.0                        # ego
    for j, (x, y) in enumerate(positions):
        if j != i and abs(x - cx) <= HALF and abs(y - cy) <= HALF:
            obs[2, y - cy + HALF, x - cx + HALF] = 1.0
    return obs


def _targets(agent):
    """(goal, park) for an agent; park defaults to goal if the scenario has none."""
    goal = tuple(agent["goal"])
    park = tuple(agent["park"]) if agent.get("park") is not None else goal
    return goal, park


def scenario_to_samples(s, keep_rest_wait=0.1, rng=None, keep_goal_wait=None):
    """Turn one scenario dict into (X, y) arrays.

    keep_rest_wait: fraction of the trailing 'parked and waiting' samples to keep
    (the robot rests on its parking cell forever, so most of those waits are
    uninformative). keep_goal_wait is accepted as an old alias.
    """
    if keep_goal_wait is not None:
        keep_rest_wait = keep_goal_wait
    rng = rng or np.random.default_rng(0)
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}   # includes debris (they are obstacles)
    ids = sorted(int(k) for k in s["trajectories"])
    traj = {i: [tuple(c) for c in s["trajectories"][str(i)]] for i in ids}
    ag = {a["id"]: a for a in s["agents"]}

    goal = {}; park = {}; f_goal = {}; f_park = {}; reached_t = {}
    for i in ids:
        goal[i], park[i] = _targets(ag[i])
        f_goal[i] = distance_field(goal[i], w, h, walls)
        f_park[i] = f_goal[i] if park[i] == goal[i] else distance_field(park[i], w, h, walls)
        # first time step the robot stands on its goal (task done -> head to park)
        rt = next((t for t in range(len(traj[i])) if traj[i][t] == goal[i]), len(traj[i]))
        reached_t[i] = rt

    X, Y = [], []
    for t in range(len(traj[ids[0]]) - 1):
        pos = [traj[i][t] for i in ids]
        for k, i in enumerate(ids):
            field = f_park[i] if t >= reached_t[i] else f_goal[i]
            label = action_label(traj[i][t], traj[i][t + 1])
            # robot parked on its final cell and just waiting: keep only a few
            if pos[k] == park[i] and label == 4 and rng.random() > keep_rest_wait:
                continue
            X.append(build_observation(k, pos, field, w, h, walls))
            Y.append(label)
    return X, Y


def build_dataset(folder, out_file="task2/dataset.npz"):
    X, Y = [], []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name)) as f:
                x, y = scenario_to_samples(json.load(f))
            X += x
            Y += y
    X, Y = np.stack(X), np.array(Y, dtype=np.int64)
    np.savez_compressed(out_file, X=X, y=Y)
    print("Saved", X.shape, "labels:", np.bincount(Y, minlength=5))


if __name__ == "__main__":
    import sys, time
    if len(sys.argv) > 1:                      # python task2/guidance.py wh_dataset
        build_dataset(sys.argv[1])
    else:                                      # timing test (goal: < 10 ms)
        walls = {(5, y) for y in range(3, 18)}
        t0 = time.perf_counter()
        for _ in range(100):
            distance_field((19, 19), 20, 20, walls)
        print(f"BFS 20x20: {(time.perf_counter() - t0) * 10:.3f} ms per call")
