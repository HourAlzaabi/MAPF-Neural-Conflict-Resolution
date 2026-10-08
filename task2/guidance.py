"""guidance.py - Student 2: distance fields + 4x11x11 observation tensors."""
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
    """Backward BFS from goal. Returns int array D[y, x]; walls/unreachable = -1."""
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
    """4x11x11 tensor for agent i. positions = list of (x,y) at time t."""
    obs = np.zeros((4, FOV, FOV), dtype=np.float32)
    cx, cy = positions[i]
    d_ego = D[cy, cx]
    for row in range(FOV):
        for col in range(FOV):
            x, y = cx + col - HALF, cy + row - HALF
            if not (0 <= x < width and 0 <= y < height) or (x, y) in obstacles:
                obs[0, row, col] = 1.0              # wall / outside map
            elif D[y, x] != UNREACHABLE:
                # positive = closer to goal than ego; scaled to [-1, 1]
                obs[3, row, col] = np.clip(d_ego - D[y, x], -5, 5) / 5.0
    obs[1, HALF, HALF] = 1.0                        # ego
    for j, (x, y) in enumerate(positions):
        if j != i and abs(x - cx) <= HALF and abs(y - cy) <= HALF:
            obs[2, y - cy + HALF, x - cx + HALF] = 1.0
    return obs


def scenario_to_samples(s, keep_goal_wait=0.1, rng=None):
    """Turn one scenario dict into (X, y) arrays."""
    rng = rng or np.random.default_rng(0)
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}
    ids = sorted(int(k) for k in s["trajectories"])
    traj = {i: [tuple(c) for c in s["trajectories"][str(i)]] for i in ids}
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    fields = {i: distance_field(goals[i], w, h, walls) for i in ids}
    X, Y = [], []
    for t in range(len(traj[ids[0]]) - 1):
        pos = [traj[i][t] for i in ids]
        for k, i in enumerate(ids):
            label = action_label(traj[i][t], traj[i][t + 1])
            # agent parked on its goal: keep only a few "Wait" samples
            if pos[k] == goals[i] and label == 4 and rng.random() > keep_goal_wait:
                continue
            X.append(build_observation(k, pos, fields[i], w, h, walls))
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
    if len(sys.argv) > 1:                      # python task2/guidance.py task1/dataset
        build_dataset(sys.argv[1])
    else:                                      # timing test (goal: < 10 ms)
        walls = {(5, y) for y in range(3, 18)}
        t0 = time.perf_counter()
        for _ in range(100):
            distance_field((19, 19), 20, 20, walls)
        print(f"BFS 20x20: {(time.perf_counter() - t0) * 10:.3f} ms per call")
