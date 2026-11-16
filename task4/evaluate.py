"""
evaluate.py  -  Task 4 evaluation harness.

Runs a POLICY over a dataset and measures it against the optimal CBS solution
recorded in each file: fleet success rate, makespan, flowtime, collisions.

    python evaluate.py --dataset ../task1/dataset --policy cbs      # sanity: 100%, 1.00x
    python evaluate.py --dataset ../task1/dataset --policy greedy   # no-coordination baseline
    python evaluate.py --dataset ../task1/dataset --policy model --model ../model.pt

Policies:
    cbs     - replays the stored optimal paths (proves the harness: 100%, 1.00x).
    greedy  - rolls downhill ignoring others; reaches goals but COLLIDES.
    model   - Meera's trained CNN. Loads her model and builds the tensor with her
              task2/guidance.py, so the input matches training exactly.
"""

import argparse
import json
import os
import sys
from collections import deque

ACTIONS = ["N", "S", "E", "W", "Wait"]
STEP = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "Wait": (0, 0)}


def load_scenario(path):
    with open(path) as f:
        return json.load(f)


def walls_of(s):
    return {tuple(c) for c in s["grid"]["obstacles"]}


def distance_field(goal, walls, w, h):
    dist = {tuple(goal): 0}
    q = deque([tuple(goal)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in walls and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


class CBSReplayPolicy:
    name = "cbs"

    def reset(self, s):
        self.paths = {int(k): [tuple(c) for c in v] for k, v in s["trajectories"].items()}

    def act(self, s, state, aid, t):
        path = self.paths[aid]
        cur = tuple(state[aid])
        nxt = path[t + 1] if t + 1 < len(path) else path[-1]
        d = (nxt[0] - cur[0], nxt[1] - cur[1])
        for a, st in STEP.items():
            if st == d:
                return a
        return "Wait"


class GreedyPolicy:
    name = "greedy"

    def reset(self, s):
        w, h = s["grid"]["width"], s["grid"]["height"]
        walls = walls_of(s)
        self.df = {a["id"]: distance_field(a["goal"], walls, w, h) for a in s["agents"]}
        self.walls, self.w, self.h = walls, w, h

    def act(self, s, state, aid, t):
        x, y = state[aid]
        best, best_d = "Wait", self.df[aid].get((x, y), 10 ** 9)
        for a, (dx, dy) in STEP.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.walls:
                d = self.df[aid].get((nx, ny))
                if d is not None and d < best_d:
                    best, best_d = a, d
        return best


class ModelPolicy:
    """Loads Meera's CNN (a state_dict) and builds the tensor with her guidance.py."""
    name = "model"

    def __init__(self, model_path):
        import torch
        import torch.nn as nn
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, "..", "task2"), os.path.join(here, "task2")):
            if os.path.isdir(cand):
                sys.path.insert(0, cand)
        import guidance
        self.guidance = guidance

        class PolicyCNN(nn.Module):               # must match train_v*.py
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(32 * 11 * 11, 128), nn.ReLU(),
                    nn.Linear(128, 5),
                )

            def forward(self, x):
                return self.net(x)

        self.torch = torch
        self.model = PolicyCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def reset(self, s):
        self.w, self.h = s["grid"]["width"], s["grid"]["height"]
        self.walls = walls_of(s)
        self.goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
        self.ids = sorted(self.goals)
        self.fields = {i: self.guidance.distance_field(self.goals[i], self.w, self.h, self.walls)
                       for i in self.ids}

    def act(self, s, state, aid, t):
        positions = [state[i] for i in self.ids]
        k = self.ids.index(aid)
        obs = self.guidance.build_observation(k, positions, self.fields[aid],
                                              self.w, self.h, self.walls)
        x = self.torch.tensor(obs).unsqueeze(0)
        with self.torch.no_grad():
            logits = self.model(x)
        return ACTIONS[int(logits.argmax())]


def rollout(s, policy):
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = walls_of(s)
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    state = {a["id"]: tuple(a["start"]) for a in s["agents"]}
    ids = sorted(state)
    policy.reset(s)
    horizon = max(2 * (w + h), 4 * s["makespan"] + 5)
    arrive = {i: 0 for i in ids if state[i] == goals[i]}
    collisions = 0
    for t in range(horizon):
        if all(state[i] == goals[i] for i in ids):
            break
        prev = dict(state)
        moves = {}
        for i in ids:
            dx, dy = STEP[policy.act(s, state, i, t)]
            nx, ny = state[i][0] + dx, state[i][1] + dy
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in walls:
                nx, ny = state[i]
            moves[i] = (nx, ny)
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                i, j = ids[a_i], ids[b_i]
                if moves[i] == moves[j] or (moves[i] == prev[j] and moves[j] == prev[i]):
                    collisions += 1
        state = moves
        for i in ids:
            if state[i] == goals[i] and i not in arrive:
                arrive[i] = t + 1
    reached = all(state[i] == goals[i] for i in ids)
    makespan = max(arrive.values()) if reached else horizon
    flowtime = sum(arrive.get(i, horizon) for i in ids)
    return {"success": reached and collisions == 0, "reached": reached,
            "collisions": collisions, "makespan": makespan, "flowtime": flowtime}


def make_policy(kind, model_path=None):
    if kind == "cbs":
        return CBSReplayPolicy()
    if kind == "greedy":
        return GreedyPolicy()
    if kind == "model":
        if not model_path:
            raise SystemExit("--policy model needs --model path/to/model.pt")
        return ModelPolicy(model_path)
    raise SystemExit("unknown policy: " + kind)


def main():
    ap = argparse.ArgumentParser(description="Task 4 evaluation of a MAPF policy.")
    ap.add_argument("--dataset", default="../task1/dataset")
    ap.add_argument("--policy", default="greedy", choices=["cbs", "greedy", "model"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dataset) if f.endswith(".json"))
    if args.limit:
        files = files[:args.limit]
    policy = make_policy(args.policy, args.model)
    n = succ = reached = coll = 0
    mk = ft = 0.0
    for name in files:
        s = load_scenario(os.path.join(args.dataset, name))
        r = rollout(s, policy)
        n += 1
        succ += r["success"]
        reached += r["reached"]
        coll += r["collisions"]
        mk += r["makespan"] / max(1, s["makespan"])
        ft += r["flowtime"] / max(1, s["sum_of_costs"])
    print(f"\nPolicy: {args.policy}   Scenarios: {n}")
    print("-" * 44)
    print(f"Fleet success rate     : {100*succ/n:5.1f}%   ({succ}/{n})")
    print(f"All robots reached goal: {100*reached/n:5.1f}%   ({reached}/{n})")
    print(f"Collisions (total)     : {coll}")
    print(f"Makespan vs CBS (avg)  : {mk/n:5.2f}x")
    print(f"Flowtime vs CBS (avg)  : {ft/n:5.2f}x")


if __name__ == "__main__":
    main()
