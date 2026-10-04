"""
grid_generator.py  -  makes random warehouse maps.

COORDINATES (the whole team must use the same rule!):
    Every cell is written [x, y].
    x = column, counted from the LEFT   (0 is the left edge)
    y = row,    counted from the TOP    (0 is the top edge)
    So [0, 0] is the top-left corner. Moving "North" makes y smaller.
"""

import random
from collections import deque

# The 4 directions a robot can step: North, South, East, West
STEPS = [(0, -1), (0, 1), (1, 0), (-1, 0)]


def neighbours(cell, width, height, obstacles):
    """Free cells you can step to from `cell` (up, down, left, right)."""
    x, y = cell
    result = []
    for dx, dy in STEPS:
        nx, ny = x + dx, y + dy
        inside = 0 <= nx < width and 0 <= ny < height
        if inside and (nx, ny) not in obstacles:
            result.append((nx, ny))
    return result


def all_free_cells_connected(width, height, obstacles):
    """
    Connectivity check.
    Start at one free cell and "flood" outward (this is BFS).
    If the flood reaches every free cell, no part of the map is walled off,
    so every robot can always reach its goal.
    """
    free = [(x, y) for x in range(width) for y in range(height)
            if (x, y) not in obstacles]
    if not free:
        return False

    seen = {free[0]}
    queue = deque([free[0]])
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(cell, width, height, obstacles):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(free)


def make_map(rng, width, height, density, num_agents):
    """
    Build one random map.
    Returns (obstacles, agents) or None if this random try was not usable
    (the caller then simply tries again).

    obstacles : a set of (x, y) wall cells
    agents    : a list like [{"id": 0, "start": (1, 1), "goal": (18, 18)}, ...]
    """
    all_cells = [(x, y) for x in range(width) for y in range(height)]

    # 1. Pick the wall cells at random.
    num_obstacles = round(density * width * height)
    obstacles = set(rng.sample(all_cells, num_obstacles))

    # 2. Throw the map away if some area is walled off.
    if not all_free_cells_connected(width, height, obstacles):
        return None

    # 3. Pick a start and a goal for every robot. All of them are different
    #    cells, so no two robots start or finish on the same square.
    free = [c for c in all_cells if c not in obstacles]
    picked = rng.sample(free, 2 * num_agents)
    agents = []
    for i in range(num_agents):
        agents.append({"id": i,
                       "start": picked[2 * i],
                       "goal": picked[2 * i + 1]})
    return obstacles, agents
