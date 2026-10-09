"""
generate_dataset.py  -  RUN THIS FILE to build the dataset.

It makes random maps, solves each one with CBS, and saves every solved
scenario as one JSON file inside a folder called "dataset".

How to run (in the VS Code terminal):
    python generate_dataset.py
"""

import json
import os
import random
import re
import time

import cbs
from actions import trajectory_to_actions
from grid_generator import make_map

# ----------------------------------------------------------------------
# SETTINGS  - change these numbers if the team decides something different
# ----------------------------------------------------------------------
NUM_SCENARIOS = 1500        # how many solved scenarios to save
NUM_AGENTS = 3              # robots per map
MIN_SIZE, MAX_SIZE = 15, 20 # maps are square, from 15x15 up to 20x20
MIN_DENSITY, MAX_DENSITY = 0.10, 0.20   # share of cells that are walls
TIMEOUT_SECONDS = 30        # give up on a map if CBS takes longer than this
SEED = 42                   # same seed -> exactly the same dataset every time
OUTPUT_FOLDER = "dataset"
# ----------------------------------------------------------------------


def pad(paths):
    """Make every path the same length by repeating the last cell
    (a robot that arrived early just waits on its goal)."""
    longest = max(len(p) for p in paths)
    return [p + [p[-1]] * (longest - len(p)) for p in paths]


def to_json_text(data):
    """Normal JSON, but keeps each [x, y] pair on one line so it is readable."""
    text = json.dumps(data, indent=2)
    return re.sub(r"\[\s+(-?\d+),\s+(-?\d+)\s+\]", r"[\1, \2]", text)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, OUTPUT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(SEED)
    began = time.time()
    saved = 0
    not_connected = 0
    unsolved = 0
    with_conflict = 0

    print(f"Making {NUM_SCENARIOS} scenarios in: {out_dir}")

    while saved < NUM_SCENARIOS:
        size = rng.randint(MIN_SIZE, MAX_SIZE)
        density = rng.uniform(MIN_DENSITY, MAX_DENSITY)

        made = make_map(rng, size, size, density, NUM_AGENTS)
        if made is None:
            not_connected += 1      # some area was walled off -> try again
            continue
        obstacles, agents = made

        paths, info = cbs.solve(size, size, obstacles, agents,
                                timeout=TIMEOUT_SECONDS)
        if paths is None:
            unsolved += 1           # CBS ran out of time -> discard this map
            continue

        if info["had_conflict"]:
            with_conflict += 1

        sum_of_costs = sum(len(p) - 1 for p in paths)
        paths = pad(paths)
        scenario_id = f"map_{saved:04d}"

        scenario = {
            "scenario_id": scenario_id,
            "grid": {
                "width": size,
                "height": size,
                "obstacle_density": round(len(obstacles) / (size * size), 3),
                "obstacles": [list(c) for c in sorted(obstacles)],
            },
            "agents": [{"id": a["id"],
                        "start": list(a["start"]),
                        "goal": list(a["goal"])} for a in agents],
            "trajectories": {str(i): [list(c) for c in paths[i]]
                             for i in range(len(paths))},
            "actions": {str(i): trajectory_to_actions(paths[i])
                        for i in range(len(paths))},
            "makespan": len(paths[0]) - 1,
            "sum_of_costs": sum_of_costs,
            "had_conflict": info["had_conflict"],
            "solver": "CBS",
            "optimal": True,
        }

        file_path = os.path.join(out_dir, scenario_id + ".json")
        with open(file_path, "w") as f:
            f.write(to_json_text(scenario))

        saved += 1
        if saved % 100 == 0:
            print(f"  {saved} / {NUM_SCENARIOS} done")

    seconds = time.time() - began
    print()
    print("FINISHED")
    print(f"  Saved scenarios            : {saved}")
    print(f"  ...where robots had to avoid each other : {with_conflict}")
    print(f"  Maps thrown away (walled-off areas)     : {not_connected}")
    print(f"  Maps thrown away (CBS timed out)        : {unsolved}")
    print(f"  Time taken                 : {seconds:.1f} seconds")


if __name__ == "__main__":
    main()
