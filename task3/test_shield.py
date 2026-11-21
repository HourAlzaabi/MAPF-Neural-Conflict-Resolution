"""Tests for the Task 3 safety shield.  Run:  python -m pytest task3 -q
(or simply:  python task3/test_shield.py)"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shield import SafetyShield, check_step, shield_step, apply, normalize_action  # noqa: E402


def run(positions, proposed, walls=(), w=5, h=5, **kw):
    s = SafetyShield(walls, w, h, **kw)
    safe = s.filter(positions, proposed)
    assert set(safe) == set(positions)
    assert check_step(positions, safe, walls, w, h) == []
    return safe


# ---------------------------------------------------------------- Section 9
def test_head_on_corridor():
    # 1-wide corridor on row 1; robots facing each other, adjacent
    walls = {(x, 0) for x in range(5)} | {(x, 2) for x in range(5)}
    safe = run({0: (1, 1), 1: (2, 1)}, {0: "E", 1: "W"}, walls, 5, 3)
    assert "Wait" in safe.values()          # swap prevented


def test_head_on_gap_one_cell():
    # one empty cell between them, both step into it -> vertex
    safe = run({0: (0, 1), 1: (2, 1)}, {0: "E", 1: "W"}, w=3, h=3)
    assert list(safe.values()).count("Wait") == 1


def test_cross():
    # A goes E into (1,1), B goes S into (1,1)
    safe = run({0: (0, 1), 1: (1, 0)}, {0: "E", 1: "S"}, w=3, h=3)
    assert list(safe.values()).count("Wait") == 1


def test_chase_conservative():
    # B follows A into A's current cell. Conservative rule: A's cell is
    # still reserved when B is checked *only* if B has higher priority.
    # With id order (A=0 first) A moves away first and B may follow.
    safe = run({0: (2, 1), 1: (1, 1)}, {0: "E", 1: "E"}, w=5, h=3)
    assert safe == {0: "E", 1: "E"}
    # Follower has higher priority (id 0 behind id 1): on the first sweep it
    # sees the leader's cell still reserved; the repeated sweep lets it follow
    # once the leader has released the cell.
    safe = run({0: (1, 1), 1: (2, 1)}, {0: "E", 1: "E"}, w=5, h=3)
    assert safe == {0: "E", 1: "E"}
    # A 3-robot train in a corridor all moves together.
    safe = run({0: (0, 1), 1: (1, 1), 2: (2, 1)}, {0: "E", 1: "E", 2: "E"}, w=5, h=3)
    assert safe == {0: "E", 1: "E", 2: "E"}
    # ...but not if the front robot is blocked: then nobody may close up.
    safe = run({0: (0, 1), 1: (1, 1), 2: (2, 1)}, {0: "E", 1: "E", 2: "Wait"}, w=5, h=3)
    assert safe == {0: "Wait", 1: "Wait", 2: "Wait"}


def test_corner_swap():
    safe = run({0: (0, 0), 1: (1, 0)}, {0: "E", 1: "W"}, w=2, h=2)
    assert safe == {0: "Wait", 1: "Wait"}


def test_wall_and_boundary():
    safe = run({0: (0, 0), 1: (2, 2)}, {0: "N", 1: "E"}, walls={(3, 2)}, w=4, h=4)
    assert safe == {0: "Wait", 1: "Wait"}


def test_parked_robot_is_obstacle():
    # robot 1 parked (no proposal) — robot 0 may not enter its cell
    safe = run({0: (0, 0), 1: (1, 0)}, {0: "E"}, w=3, h=3)
    assert safe == {0: "Wait", 1: "Wait"}


def test_rotation_deadlock_is_safe():
    # 2x2 rotation: all target an occupied cell -> all wait (known limitation)
    pos = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    prop = {0: "E", 1: "S", 2: "W", 3: "N"}
    safe = run(pos, prop, w=2, h=2)
    assert all(a == "Wait" for a in safe.values())


def test_alternative_fallback():
    # robot 0 (higher priority) goes S into (1,1); robot 1 wants E into the
    # same cell -> vetoed, but S also gets it closer to (2,2), so take that.
    dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    s = SafetyShield((), 3, 3, fallback="alternative")
    pos = {0: (1, 0), 1: (0, 1)}
    safe = s.filter(pos, {0: "S", 1: "E"}, targets={0: (1, 2), 1: (2, 2)}, dist_fn=dist)
    assert check_step(pos, safe, (), 3, 3) == []
    assert safe == {0: "S", 1: "S"}
    assert s.stats["substitutions"] == 1


def test_index_actions_accepted():
    assert normalize_action(0) == "N" and normalize_action(4) == "Wait"
    assert normalize_action("wait") == "Wait"


# --------------------------------------------- randomized fuzz (R1/R3/R4)
def test_fuzz_random_proposals():
    rng = random.Random(0)
    acts = ["N", "S", "E", "W", "Wait"]
    for prio in ("id", "rotate", "random"):
        for fb in ("wait", "alternative"):
            for _ in range(2000):
                w, h = rng.randint(2, 8), rng.randint(2, 8)
                cells = [(x, y) for x in range(w) for y in range(h)]
                rng.shuffle(cells)
                n = rng.randint(1, min(6, len(cells)))
                walls = set(cells[n:n + rng.randint(0, len(cells) - n)])
                pos = {i: cells[i] for i in range(n)}
                prop = {i: rng.choice(acts) for i in range(n)}
                s = SafetyShield(walls, w, h, priority=prio, fallback=fb)
                safe = s.filter(pos, prop)
                assert check_step(pos, safe, walls, w, h) == [], (pos, prop, safe)


def test_speed_sub_millisecond():
    s = SafetyShield(set(), 20, 20)
    pos = {0: (1, 1), 1: (5, 5), 2: (10, 10)}
    prop = {0: "E", 1: "S", 2: "W"}
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        s.filter(pos, prop)
    per = (time.perf_counter() - t0) / n
    assert per < 1e-3, per


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
