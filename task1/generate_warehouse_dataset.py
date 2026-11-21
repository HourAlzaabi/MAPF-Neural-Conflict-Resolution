"""
generate_warehouse_dataset.py  -  build the Task 1 training set.

Writes N solved warehouse scenarios (default 1,500) as JSON, mixing the three
map flavours so the CNN sees variety:  spread / cluster / debris  (see
warehouse_generator.py).  Every robot drives start -> goal -> parking cell, solved
jointly and collision-free by CBS (cbs.py).  Sum-of-costs (flowtime) is the
headline cost in each file; makespan is kept as a secondary number.

Usage:
    python generate_warehouse_dataset.py                  # 1500 -> ./wh_dataset
    python generate_warehouse_dataset.py --n 60 --out smoke
    python generate_warehouse_dataset.py --n 1500 --seed 42
"""
import argparse
import json
import os
import random
import re
import time

import cbs
from actions import trajectory_to_actions
from warehouse_generator import make_scenario

# How often each flavour appears. Mostly spread (the base case), a solid chunk
# of tight clusters, a fifth with fallen boxes.
FLAVOUR_MIX = [("spread", 0.50), ("cluster", 0.30), ("debris", 0.20)]


def _pick_flavour(rng):
    r = rng.random()
    c = 0.0
    for name, p in FLAVOUR_MIX:
        c += p
        if r <= c:
            return name
    return FLAVOUR_MIX[-1][0]


def _pad(paths):
    L = max(len(p) for p in paths)
    return [p + [p[-1]] * (L - len(p)) for p in paths]


def _compact(d):
    # keep each [x, y] on one line for readability
    return re.sub(r"\[\s+(-?\d+),\s+(-?\d+)\s+\]", r"[\1, \2]", json.dumps(d, indent=2))


def build_record(sid, size, flavour, walls, debris, agents, paths, info):
    padded = _pad([[tuple(c) for c in p] for p in paths])
    soc = sum(len(p) - 1 for p in paths)
    makespan = max(len(p) - 1 for p in paths)
    return {
        "scenario_id": sid,
        "flavour": flavour,
        "grid": {
            "width": size, "height": size,
            "obstacle_density": round(len(walls) / (size * size), 3),
            "obstacles": [list(c) for c in sorted(walls)],
            "debris": [list(c) for c in sorted(debris)],
        },
        "agents": [{"id": a["id"], "start": list(a["start"]),
                    "goal": list(a["goal"]),
                    "park": (list(a["park"]) if a.get("park") is not None else None)}
                   for a in agents],
        "trajectories": {str(i): [list(c) for c in padded[i]]
                         for i in range(len(padded))},
        "actions": {str(i): trajectory_to_actions(padded[i])
                    for i in range(len(padded))},
        "sum_of_costs": soc,     # headline cost (flowtime)
        "flowtime": soc,         # alias, same number
        "makespan": makespan,    # secondary
        "had_conflict": info["had_conflict"],
        "solver": "CBS", "optimal": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="wh_dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-size", type=int, default=15)
    ap.add_argument("--max-size", type=int, default=20)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    began = time.time()

    made = attempts = 0
    stats = {f: 0 for f, _ in FLAVOUR_MIX}
    dsum = csum = soc_sum = mk_sum = wait_sum = 0.0
    while made < args.n:
        attempts += 1
        if attempts > args.n * 40:
            print("  stopping early: too many rejected attempts")
            break
        size = rng.randint(args.min_size, args.max_size)
        flavour = _pick_flavour(rng)
        out = make_scenario(rng, size, size, args.agents, flavour)
        if out is None:
            continue
        walls, debris, agents = out
        paths, info = cbs.solve(size, size, walls, agents, timeout=args.timeout)
        if paths is None:
            continue

        sid = f"wh_{made:04d}"
        rec = build_record(sid, size, flavour, walls, debris, agents, paths, info)
        with open(os.path.join(args.out, sid + ".json"), "w") as f:
            f.write(_compact(rec))

        made += 1
        stats[flavour] += 1
        dsum += rec["grid"]["obstacle_density"]
        csum += 1 if info["had_conflict"] else 0
        soc_sum += rec["sum_of_costs"]
        mk_sum += rec["makespan"]
        wait_sum += sum(1 for p in paths for t in range(1, len(p)) if p[t] == p[t - 1])
        if made % 100 == 0:
            print(f"  {made}/{args.n}  ({time.time()-began:.0f}s)")

    print(f"\nDone: {made} scenarios in {args.out}/  "
          f"({attempts} attempts, {time.time()-began:.0f}s)")
    print("Flavour mix:   " + "  ".join(f"{f}={stats[f]}" for f, _ in FLAVOUR_MIX))
    if made:
        print(f"Avg density:   {dsum/made:.3f}  (spec 0.10-0.25)")
        print(f"With conflict: {100*csum/made:.1f}%")
        print(f"Avg flowtime (sum-of-costs): {soc_sum/made:.1f}   "
              f"avg makespan: {mk_sum/made:.1f}")
        print(f"Avg yields/scenario: {wait_sum/made:.2f}")


if __name__ == "__main__":
    main()
