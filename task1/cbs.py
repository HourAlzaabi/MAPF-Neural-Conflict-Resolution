"""
cbs.py  -  Conflict-Based Search (CBS), the "expert" our neural net copies.

THE IDEA IN PLAIN WORDS
    1. Plan a shortest route for every robot on its own, ignoring the others.
    2. Look for the first crash between two robots.
    3. If there is none -> finished.
    4. If there is one, make two copies of the plan:
         copy A: robot 1 is NOT allowed to be there at that time
         copy B: robot 2 is NOT allowed to be there at that time
       Re-plan only the robot that got the new rule. Go back to step 2,
       always continuing with the cheapest plan found so far.

    Because it always continues from the cheapest plan, the first crash-free
    plan it finds has the smallest possible total number of steps
    (this total is called "sum of costs", a.k.a. flowtime). That is why CBS
    is called optimal.

TWO LEVELS
    High level = the loop above (decides the rules, called "constraints").
    Low level  = A* search for ONE robot that obeys its constraints.

A* (LOW LEVEL), IN PLAIN WORDS
    A* is shortest-path search that is told, for every cell, roughly how far
    the goal still is (the "heuristic"). It always expands the option that
    looks cheapest overall = steps already taken + estimated steps left, so it
    heads straight for the goal instead of searching blindly in all directions.
    Our heuristic is EXACT: a backward BFS from the goal gives the true walking
    distance from every cell (see `distance_to_goal`). BFS here is only the
    pre-computation that feeds A*; the actual route search is A*.

WAYPOINTS / PARKING  (so a robot never freezes in an aisle)
    A robot's route is start -> goal -> parking cell. It drives to its goal
    (does its task), then clears to a parking cell that sits in open floor, and
    rests there. The low level is therefore a multi-waypoint A*: it searches
    through space AND time AND which leg of the trip it is on. CBS coordinates
    every robot across BOTH legs at once, so the parking moves are collision-
    free too. A robot with no parking cell just rests on its goal (old behaviour).

TWO KINDS OF CRASH
    Vertex conflict: two robots in the same cell at the same time.
    Swap conflict:   two robots trade places (they pass through each other).
"""

import heapq
import time
from collections import deque

from grid_generator import neighbours


def distance_to_goal(goal, width, height, obstacles):
    """Backward BFS from a cell: the real walking distance from every cell to
    `goal`. A* uses this as its exact "how far is left" guess (the heuristic)."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(cell, width, height, obstacles):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def plan_one_robot(start, waypoints, width, height, obstacles, dists,
                   banned_cells, banned_moves):
    """
    LOW LEVEL: multi-waypoint A* search through space AND time for one robot.

    waypoints    : ordered list of cells to visit, e.g. [goal, parking_cell].
                   The robot must reach them in order and finally rest on the
                   LAST one. A single-element list = plain "go to goal and park
                   on it" (the classic one-shot behaviour).
    dists        : list of distance fields, dists[k] = distance_to_goal for
                   waypoints[k]. Used as the A* heuristic for each leg.
    banned_cells : set of (cell, t)          -> may not be in `cell` at time t
    banned_moves : set of (from, to, t)      -> may not step from->to arriving at t
    Returns the path as a list of cells (index = time step), or None.
    """
    last = len(waypoints) - 1
    final = waypoints[last]

    if start not in dists[0]:
        return None  # first waypoint cannot be reached at all

    # Cost still to travel AFTER finishing leg k, walking the remaining
    # waypoints in order: dists[k+1][waypoints[k]] + dists[k+2][waypoints[k+1]] ...
    suffix = [0] * len(waypoints)
    for k in range(last - 1, -1, -1):
        hop = dists[k + 1].get(waypoints[k])
        if hop is None:
            return None  # a later waypoint is unreachable from an earlier one
        suffix[k] = hop + suffix[k + 1]

    def h(cell, phase):
        d = dists[phase].get(cell)
        if d is None:
            return None
        return d + suffix[phase]

    def advance(cell, phase):
        """If we are standing on the waypoint(s) for this leg, move on to the
        next leg(s) for free."""
        while phase < last and cell == waypoints[phase]:
            phase += 1
        return phase

    # The robot parks on `final` forever, so it may only "finish" after the
    # last time it is banned from standing on `final`.
    banned_final_times = [t for (cell, t) in banned_cells if cell == final]
    earliest_finish = max(banned_final_times) + 1 if banned_final_times else 0

    latest_rule = max([t for (_, t) in banned_cells] +
                      [t for (_, _, t) in banned_moves] + [0])
    # Enough time to walk every leg plus wait out every rule.
    max_time = len(waypoints) * width * height + latest_rule

    start_phase = advance(start, 0)
    counter = 0
    h0 = h(start, start_phase)
    if h0 is None:
        return None
    # Each entry: (f = steps so far + steps left, steps left, tie, cell, t, phase)
    open_list = [(h0, h0, counter, start, 0, start_phase)]
    came_from = {(start, 0, start_phase): None}

    while open_list:
        _, _, _, cell, t, phase = heapq.heappop(open_list)

        if phase == last and cell == final and t >= earliest_finish:
            path = []
            state = (cell, t, phase)
            while state is not None:
                path.append(state[0])
                state = came_from[state]
            return path[::-1]

        if t >= max_time:
            continue

        # The robot can wait in place or step to a free neighbour.
        for nxt in [cell] + neighbours(cell, width, height, obstacles):
            nt = t + 1
            if (nxt, nt) in banned_cells:
                continue
            if (cell, nxt, nt) in banned_moves:
                continue
            nphase = advance(nxt, phase)
            key = (nxt, nt, nphase)
            if key in came_from:
                continue
            est = h(nxt, nphase)
            if est is None:
                continue
            came_from[key] = (cell, t, phase)
            counter += 1
            heapq.heappush(open_list, (nt + est, est, counter, nxt, nt, nphase))
    return None


def where(path, t):
    """Where the robot is at time t (it stays on its parking cell after arriving)."""
    return path[t] if t < len(path) else path[-1]


def first_conflict(paths):
    """Find the earliest crash between any two robots, or None."""
    longest = max(len(p) for p in paths)
    for t in range(longest):
        for a in range(len(paths)):
            for b in range(a + 1, len(paths)):
                # Vertex conflict: same cell, same time.
                if where(paths[a], t) == where(paths[b], t):
                    return {"type": "vertex", "a": a, "b": b,
                            "cell": where(paths[a], t), "t": t}
                # Swap conflict: they trade places between t and t+1.
                if (where(paths[a], t) == where(paths[b], t + 1) and
                        where(paths[a], t + 1) == where(paths[b], t)):
                    return {"type": "swap", "a": a, "b": b,
                            "from_a": where(paths[a], t),
                            "from_b": where(paths[b], t), "t": t + 1}
    return None


def _waypoints_of(agent):
    """A robot's ordered cells to visit: [goal] or [goal, parking cell]."""
    wp = [tuple(agent["goal"])]
    if agent.get("park") is not None:
        wp.append(tuple(agent["park"]))
    return wp


def solve(width, height, obstacles, agents, timeout=30.0):
    """
    HIGH LEVEL of CBS.
    Returns (paths, info) or (None, info) if no answer within `timeout` seconds.
    paths[i] is the list of cells for robot i (start -> goal -> parking cell).
    info["sum_of_costs"] (flowtime) is the headline cost; makespan is secondary.
    """
    began = time.time()
    n = len(agents)
    waypoints = [_waypoints_of(a) for a in agents]
    # One distance field per waypoint (feeds the A* heuristic for each leg).
    dists = [[distance_to_goal(wp, width, height, obstacles) for wp in wps]
             for wps in waypoints]

    def plan(i, cells, moves):
        return plan_one_robot(tuple(agents[i]["start"]), waypoints[i],
                              width, height, obstacles, dists[i], cells, moves)

    def total_cost(paths):
        return sum(len(p) - 1 for p in paths)      # sum of costs = flowtime

    root_cells = [set() for _ in range(n)]
    root_moves = [set() for _ in range(n)]
    root_paths = [plan(i, root_cells[i], root_moves[i]) for i in range(n)]
    info = {"plans_checked": 0, "had_conflict": False,
            "sum_of_costs": None, "makespan": None}
    if any(p is None for p in root_paths):
        return None, info

    counter = 0
    open_list = [(total_cost(root_paths), counter,
                  root_paths, root_cells, root_moves)]

    while open_list:
        if time.time() - began > timeout:
            return None, info

        _, _, paths, cells, moves = heapq.heappop(open_list)
        info["plans_checked"] += 1

        conflict = first_conflict(paths)
        if conflict is None:
            info["sum_of_costs"] = total_cost(paths)        # headline (flowtime)
            info["makespan"] = max(len(p) - 1 for p in paths)  # secondary
            return paths, info
        info["had_conflict"] = True

        for robot in (conflict["a"], conflict["b"]):
            new_cells = [set(s) for s in cells]
            new_moves = [set(s) for s in moves]
            t = conflict["t"]
            if conflict["type"] == "vertex":
                new_cells[robot].add((conflict["cell"], t))
            elif robot == conflict["a"]:
                new_moves[robot].add((conflict["from_a"], conflict["from_b"], t))
            else:
                new_moves[robot].add((conflict["from_b"], conflict["from_a"], t))

            new_path = plan(robot, new_cells[robot], new_moves[robot])
            if new_path is None:
                continue
            new_paths = list(paths)
            new_paths[robot] = new_path
            counter += 1
            heapq.heappush(open_list, (total_cost(new_paths), counter,
                                       new_paths, new_cells, new_moves))
    return None, info
