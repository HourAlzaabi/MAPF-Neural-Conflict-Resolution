"""
cbs.py  -  Conflict-Based Search (CBS), the "expert" our neural net copies.

THE IDEA IN PLAIN WORDS
    1. Plan a shortest path for every robot on its own, ignoring the others.
    2. Look for the first crash between two robots.
    3. If there is none -> finished.
    4. If there is one, make two copies of the plan:
         copy A: robot 1 is NOT allowed to be there at that time
         copy B: robot 2 is NOT allowed to be there at that time
       Re-plan only the robot that got the new rule. Go back to step 2,
       always continuing with the cheapest plan found so far.

    Because it always continues from the cheapest plan, the first crash-free
    plan it finds has the smallest possible total number of steps
    (this total is called "sum of costs"). That is why CBS is called optimal.

TWO LEVELS
    High level = the loop above (decides the rules, called "constraints").
    Low level  = A* search for ONE robot that obeys its constraints.

TWO KINDS OF CRASH
    Vertex conflict: two robots in the same cell at the same time.
    Swap conflict:   two robots trade places (they pass through each other).
"""

import heapq
import time
from collections import deque

from grid_generator import neighbours


def distance_to_goal(goal, width, height, obstacles):
    """BFS from the goal: real walking distance from every cell to the goal.
    A* uses this as its "how far is left" guess (the heuristic)."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(cell, width, height, obstacles):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def plan_one_robot(start, goal, width, height, obstacles, dist,
                   banned_cells, banned_moves):
    """
    LOW LEVEL: A* search through space AND time for one robot.

    banned_cells : set of (cell, t)          -> may not be in `cell` at time t
    banned_moves : set of (from, to, t)      -> may not step from->to arriving at t
    Returns the path as a list of cells (index = time step), or None.
    """
    if start not in dist:
        return None  # goal cannot be reached at all

    # The robot parks on its goal forever, so it may only "finish" after the
    # last time it is banned from standing on the goal.
    banned_goal_times = [t for (cell, t) in banned_cells if cell == goal]
    earliest_finish = max(banned_goal_times) + 1 if banned_goal_times else 0

    # Safety limit so the search always stops.
    latest_rule = max([t for (_, t) in banned_cells] +
                      [t for (_, _, t) in banned_moves] + [0])
    max_time = width * height + latest_rule

    # Each entry: (f = steps so far + steps left, steps left, tie-breaker, cell, t)
    counter = 0
    open_list = [(dist[start], dist[start], counter, start, 0)]
    came_from = {(start, 0): None}

    while open_list:
        _, _, _, cell, t = heapq.heappop(open_list)

        if cell == goal and t >= earliest_finish:
            # Walk backwards to rebuild the path.
            path = []
            state = (cell, t)
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
            if (nxt, nt) in came_from:
                continue  # already reached this cell at this time
            came_from[(nxt, nt)] = (cell, t)
            counter += 1
            heapq.heappush(open_list,
                           (nt + dist[nxt], dist[nxt], counter, nxt, nt))
    return None


def where(path, t):
    """Where the robot is at time t (it stays on its goal after arriving)."""
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


def solve(width, height, obstacles, agents, timeout=30.0):
    """
    HIGH LEVEL of CBS.
    Returns (paths, info) or (None, info) if no answer within `timeout` seconds.
    paths[i] is the list of cells for robot i.
    """
    began = time.time()
    n = len(agents)
    dists = [distance_to_goal(a["goal"], width, height, obstacles)
             for a in agents]

    def plan(i, cells, moves):
        return plan_one_robot(agents[i]["start"], agents[i]["goal"],
                              width, height, obstacles, dists[i],
                              cells, moves)

    def total_cost(paths):
        return sum(len(p) - 1 for p in paths)

    # Root plan: everybody takes their own shortest path, no rules yet.
    root_cells = [set() for _ in range(n)]
    root_moves = [set() for _ in range(n)]
    root_paths = [plan(i, root_cells[i], root_moves[i]) for i in range(n)]
    info = {"plans_checked": 0, "had_conflict": False}
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
            return paths, info          # no crashes -> this is the answer
        info["had_conflict"] = True

        # Make two children: one where robot a gets the new rule,
        # one where robot b gets it.
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
                continue                # this choice is a dead end
            new_paths = list(paths)
            new_paths[robot] = new_path
            counter += 1
            heapq.heappush(open_list, (total_cost(new_paths), counter,
                                       new_paths, new_cells, new_moves))
    return None, info
