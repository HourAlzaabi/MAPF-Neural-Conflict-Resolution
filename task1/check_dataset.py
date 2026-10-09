"""
check_dataset.py  -  proves the dataset is correct.

It opens every JSON file in the "dataset" folder and checks:
  - every robot starts on its start and ends on its goal
  - robots never leave the map or walk into a wall
  - robots only move one cell (up/down/left/right) or wait
  - no two robots are ever in the same cell   (vertex conflict)
  - no two robots ever swap places            (swap conflict)
  - all trajectories have the same length

How to run:
    python check_dataset.py
"""

import json
import os

from actions import trajectory_to_actions

FOLDER = "dataset"


def check(s):
    """Return a list of problems found in one scenario (empty = all good)."""
    problems = []
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}
    paths = {}

    for agent in s["agents"]:
        aid = str(agent["id"])
        path = [tuple(c) for c in s["trajectories"][aid]]
        paths[agent["id"]] = path
        if path[0] != tuple(agent["start"]):
            problems.append(f"robot {agent['id']} does not begin on its start")
        if path[-1] != tuple(agent["goal"]):
            problems.append(f"robot {agent['id']} does not end on its goal")
        # the action labels must match the path exactly
        if "actions" in s:
            if trajectory_to_actions(path) != s["actions"].get(aid):
                problems.append(f"robot {agent['id']} actions do not match its path")
        for (x, y) in path:
            if not (0 <= x < w and 0 <= y < h):
                problems.append(f"robot {agent['id']} left the map")
            if (x, y) in walls:
                problems.append(f"robot {agent['id']} walked into a wall")
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            if abs(x1 - x2) + abs(y1 - y2) > 1:
                problems.append(f"robot {agent['id']} jumped more than 1 cell")

    lengths = {len(p) for p in paths.values()}
    if len(lengths) != 1:
        problems.append("trajectories have different lengths")
        return problems
    if s["makespan"] != lengths.pop() - 1:
        problems.append("makespan number is wrong")

    ids = sorted(paths)
    for t in range(len(paths[ids[0]])):
        for i in ids:
            for j in ids:
                if i >= j:
                    continue
                if paths[i][t] == paths[j][t]:
                    problems.append(f"robots {i},{j} same cell at t={t}")
                if t > 0 and (paths[i][t] == paths[j][t - 1] and
                              paths[i][t - 1] == paths[j][t]):
                    problems.append(f"robots {i},{j} swapped at t={t}")
    return problems


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(here, FOLDER)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))

    bad = 0
    for name in files:
        with open(os.path.join(folder, name)) as f:
            problems = check(json.load(f))
        if problems:
            bad += 1
            print(name, "->", problems[0])

    print(f"Checked {len(files)} files. Files with problems: {bad}")
    if bad == 0:
        print("ALL GOOD - no collisions, no wall hits, all goals reached.")


if __name__ == "__main__":
    main()
