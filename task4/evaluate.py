"""
evaluate.py  -  Task 4 evaluation harness.

Runs a POLICY over a dataset and measures it against the optimal CBS solution
recorded in each file: fleet success rate, makespan, flowtime, collisions.

    python evaluate.py --dataset ../task1/wh_dataset --policy cbs      # sanity: 100%, 1.00x
    python evaluate.py --dataset ../task1/wh_dataset --policy greedy   # no-coordination baseline
    python evaluate.py --dataset ../task1/wh_dataset --policy model --model ../model.pt
    python evaluate.py --dataset ../task1/wh_dataset --policy greedy --shield   # Task 3

Policies:
    cbs     - replays the stored optimal paths (proves the harness: 100%, 1.00x).
    greedy  - rolls downhill ignoring others; reaches its spots but COLLIDES.
    model   - Meera's trained CNN. Loads her model and builds the tensor with her
              task2/guidance.py, so the input matches training exactly.

PARKING SCHEMA: a robot's route is start -> goal -> parking cell. "Done" means it
has reached its parking cell (having passed through its goal). The greedy and model
policies therefore steer toward the goal first, then toward the parking cell.
Flowtime/makespan are measured to the parking cell, matching the stored CBS cost
(which also includes the parking leg) so the ratios stay apples-to-apples.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))

ACTIONS = ["N", "S", "E", "W", "Wait"]
STEP = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "Wait": (0, 0)}


def load_scenario(path):
    with open(path) as f:
        return json.load(f)


def walls_of(s):
    return {tuple(c) for c in s["grid"]["obstacles"]}


def goal_of(a):
    return tuple(a["goal"])


def park_of(a):
    return tuple(a["park"]) if a.get("park") is not None else tuple(a["goal"])


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
    """Rolls downhill toward its CURRENT target: goal first, then parking cell."""
    name = "greedy"

    def reset(self, s):
        w, h = s["grid"]["width"], s["grid"]["height"]
        walls = walls_of(s)
        self.goal = {a["id"]: goal_of(a) for a in s["agents"]}
        self.park = {a["id"]: park_of(a) for a in s["agents"]}
        self.f_goal = {i: distance_field(self.goal[i], walls, w, h) for i in self.goal}
        self.f_park = {i: distance_field(self.park[i], walls, w, h) for i in self.park}
        self.visited = set()
        self.walls, self.w, self.h = walls, w, h

    def _field(self, aid, pos):
        if pos == self.goal[aid]:
            self.visited.add(aid)
        return self.f_park[aid] if aid in self.visited else self.f_goal[aid]

    def act(self, s, state, aid, t):
        x, y = state[aid]
        df = self._field(aid, (x, y))
        best, best_d = "Wait", df.get((x, y), 10 ** 9)
        for a, (dx, dy) in STEP.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.walls:
                d = df.get((nx, ny))
                if d is not None and d < best_d:
                    best, best_d = a, d
        return best


class ModelPolicy:
    """Loads Meera's CNN (a state_dict) and builds the tensor with her guidance.py.
    Steers toward the goal until reached, then toward the parking cell - same target
    switch the training data uses."""
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
        self.goal = {a["id"]: goal_of(a) for a in s["agents"]}
        self.park = {a["id"]: park_of(a) for a in s["agents"]}
        self.ids = sorted(self.goal)
        self.f_goal = {i: self.guidance.distance_field(self.goal[i], self.w, self.h, self.walls)
                       for i in self.ids}
        self.f_park = {i: (self.f_goal[i] if self.park[i] == self.goal[i]
                           else self.guidance.distance_field(self.park[i], self.w, self.h, self.walls))
                       for i in self.ids}
        self.visited = set()

    def act(self, s, state, aid, t):
        positions = [state[i] for i in self.ids]
        k = self.ids.index(aid)
        if tuple(state[aid]) == self.goal[aid]:
            self.visited.add(aid)
        field = self.f_park[aid] if aid in self.visited else self.f_goal[aid]
        obs = self.guidance.build_observation(k, positions, field,
                                              self.w, self.h, self.walls)
        x = self.torch.tensor(obs).unsqueeze(0)
        with self.torch.no_grad():
            logits = self.model(x)
        return ACTIONS[int(logits.argmax())]


def _load_shield_class():
    """Task 3's safety shield (task3/shield.py)."""
    for cand in (os.path.join(_HERE, "..", "task3"), os.path.join(_HERE, "task3")):
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    from shield import SafetyShield
    return SafetyShield


def rollout(s, policy, shield=None, deadlock_window=10):
    """shield = None (raw policy) or a dict of SafetyShield options, e.g.
    {"priority": "rotate", "fallback": "alternative"}: then every step runs
    policy -> shield -> execute."""
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = walls_of(s)
    goal = {a["id"]: goal_of(a) for a in s["agents"]}
    park = {a["id"]: park_of(a) for a in s["agents"]}      # final resting cell
    state = {a["id"]: tuple(a["start"]) for a in s["agents"]}
    ids = sorted(state)
    policy.reset(s)
    horizon = max(2 * (w + h), 4 * s["makespan"] + 5)
    visited_goal = {i for i in ids if state[i] == goal[i]}
    done = {i: 0 for i in ids if state[i] == park[i] and i in visited_goal}
    collisions = 0
    sh = None
    if shield is not None:
        sh = _load_shield_class()(walls, w, h, **shield)
        fields = {}

        def dist_fn(cell, tgt):
            if tgt not in fields:
                fields[tgt] = distance_field(tgt, walls, w, h)
            return fields[tgt].get(cell, 10 ** 9)
    shield_time = 0.0
    steps = 0
    still = 0
    deadlock = False

    def all_done():
        return all(state[i] == park[i] and i in visited_goal for i in ids)

    for t in range(horizon):
        if all_done():
            break
        prev = dict(state)
        proposed = {i: policy.act(s, state, i, t) for i in ids}
        if sh is not None:
            targets = {i: (park[i] if i in visited_goal else goal[i]) for i in ids}
            t0 = time.perf_counter()
            proposed = sh.filter(state, proposed, targets=targets, dist_fn=dist_fn)
            shield_time += time.perf_counter() - t0
        steps += 1
        moves = {}
        for i in ids:
            dx, dy = STEP[proposed[i]]
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
            if state[i] == goal[i]:
                visited_goal.add(i)
            if state[i] == park[i] and i in visited_goal and i not in done:
                done[i] = t + 1
        # Deadlock: nobody has moved for `deadlock_window` steps, not finished.
        still = 0 if state != prev else still + 1
        if sh is not None and still >= deadlock_window and not all_done():
            deadlock = True
            break
    reached = all_done()
    makespan = max(done.values()) if reached else horizon
    flowtime = sum(done.get(i, horizon) for i in ids)
    return {"success": reached and collisions == 0, "reached": reached,
            "collisions": collisions, "makespan": makespan, "flowtime": flowtime,
            "deadlock": deadlock, "steps": steps, "shield_time": shield_time,
            "vetoes": sh.stats["vetoes"] if sh is not None else 0}


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
    ap.add_argument("--dataset", default="../task1/wh_dataset")
    ap.add_argument("--policy", default="greedy", choices=["cbs", "greedy", "model"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shield", action="store_true",
                    help="wrap the policy in Task 3's safety shield")
    ap.add_argument("--shield-priority", default="id",
                    choices=["id", "rotate", "distance", "random"])
    ap.add_argument("--shield-fallback", default="alternative",
                    choices=["wait", "alternative"])
    args = ap.parse_args()
    shield = ({"priority": args.shield_priority, "fallback": args.shield_fallback}
              if args.shield else None)

    files = sorted(f for f in os.listdir(args.dataset) if f.endswith(".json"))
    if args.limit:
        files = files[:args.limit]
    policy = make_policy(args.policy, args.model)
    n = succ = reached = coll = dead = steps = vetoes = 0
    mk = ft = sh_time = 0.0
    worst_ms = 0.0
    for name in files:
        s = load_scenario(os.path.join(args.dataset, name))
        r = rollout(s, policy, shield)
        n += 1
        dead += r["deadlock"]
        steps += r["steps"]
        vetoes += r["vetoes"]
        sh_time += r["shield_time"]
        worst_ms = max(worst_ms, 1000 * r["shield_time"] / max(1, r["steps"]))
        succ += r["success"]
        reached += r["reached"]
        coll += r["collisions"]
        mk += r["makespan"] / max(1, s["makespan"])
        ft += r["flowtime"] / max(1, s["sum_of_costs"])
    label = args.policy
    if shield:
        label += f" + shield ({args.shield_priority} / {args.shield_fallback})"
    print(f"\nPolicy: {label}   Scenarios: {n}")
    print("-" * 44)
    print(f"Fleet success rate     : {100*succ/n:5.1f}%   ({succ}/{n})")
    print(f"All robots parked clear: {100*reached/n:5.1f}%   ({reached}/{n})")
    print(f"Collisions (total)     : {coll}")
    print(f"Makespan vs CBS (avg)  : {mk/n:5.2f}x")
    print(f"Flowtime vs CBS (avg)  : {ft/n:5.2f}x   (flowtime = sum-of-costs, the headline)")
    if shield:
        print(f"Deadlocks              : {100*dead/n:5.1f}%   ({dead}/{n})")
        print(f"Shield vetoes (total)  : {vetoes}")
        print(f"Shield time per step   : {1000*sh_time/max(1, steps):.4f} ms avg, "
              f"{worst_ms:.4f} ms worst scenario")


if __name__ == "__main__":
    main()
