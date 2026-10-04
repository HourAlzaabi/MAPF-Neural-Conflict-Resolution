"""
view_scenario.py  -  shows one scenario as a text picture, step by step.

How to run:
    python view_scenario.py                (shows map_0000)
    python view_scenario.py map_0172       (shows the one you name)

Legend:   #  wall      .  empty      0 1 2  robots      a b c  their goals
"""

import json
import os
import sys


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "map_0000"
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "dataset", name + ".json")) as f:
        s = json.load(f)

    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}
    goals = {tuple(a["goal"]): "abcdefghij"[a["id"]] for a in s["agents"]}
    paths = s["trajectories"]

    print(f"{name}: {w}x{h}, makespan {s['makespan']}, "
          f"conflict resolved: {s['had_conflict']}")

    for t in range(s["makespan"] + 1):
        robots = {tuple(paths[k][t]): k for k in paths}
        print(f"\nTime step {t}")
        for y in range(h):                 # y = row, top to bottom
            row = ""
            for x in range(w):             # x = column, left to right
                if (x, y) in robots:
                    row += robots[(x, y)]
                elif (x, y) in walls:
                    row += "#"
                elif (x, y) in goals:
                    row += goals[(x, y)]
                else:
                    row += "."
                row += " "
            print(row)


if __name__ == "__main__":
    main()
