"""
warehouse_generator.py  -  randomised warehouse maps, within the 10-25% density
spec, in three flavours so the network sees variety (not one memorised map):

  spread   : rack grid with aisles BOTH ways; one aisle direction is 1-wide (a
             bottleneck that forces a yield) while the other is a wide cross-aisle.
  cluster  : the SAME racks packed tight into one zone (1-wide aisles) with open
             floor beside it - a dense storage block + an open staging area, like
             a real warehouse. Same density band, just concentrated.
  debris   : a spread or cluster map plus a few random "fallen boxes" (e.g. a box
             that dropped off a shelf). Debris are treated as ordinary STATIC
             obstacles (CBS and the CNN see them as walls); they are also tagged
             separately so the visualiser can colour them.

Every map is checked: density inside 10-25%, all free cells connected, solvable.
Each robot is given a PARKING cell in open floor (never a 1-wide aisle) so that
after it reaches its goal it clears the aisle and rests out of the way instead of
freezing in a lane. See cbs.py for how the start -> goal -> parking route is solved.
"""
from collections import deque

FLAVOURS = ["spread", "cluster", "debris"]


def _connected(free):
    fs = set(free)
    if not fs:
        return False
    s = next(iter(fs))
    seen = {s}
    q = deque([s])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n in fs and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(fs)


def _free_degree(cell, walls, w, h):
    x, y = cell
    d = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in walls:
            d += 1
    return d


def _spread_walls(rng, w, h):
    walls = set()
    narrow = rng.choice(['v', 'h'])        # which aisle direction is the 1-wide bottleneck
    bw = rng.choice([1, 2])                # rack block width
    bh = rng.choice([2, 3])                # rack block height
    wide = rng.choice([2, 3])              # the cross-aisle (wide) width
    av = 1 if narrow == 'v' else wide
    ah = 1 if narrow == 'h' else wide
    ox, oy = rng.randint(1, 2), rng.randint(1, 2)
    y = oy
    while y + bh <= h - 1:
        x = ox
        while x + bw <= w - 1:
            for yy in range(y, y + bh):
                for xx in range(x, x + bw):
                    walls.add((xx, yy))
            x += bw + av
        y += bh + ah
    return walls


def _cluster_walls(rng, w, h):
    """Racks packed tight (1-wide aisles) into ONE zone; open floor elsewhere."""
    walls = set()
    zw = rng.randint(int(w * 0.5), int(w * 0.72))
    zh = rng.randint(int(h * 0.5), int(h * 0.72))
    zx = rng.choice([1, w - 1 - zw])
    zy = rng.choice([1, h - 1 - zh])
    zx = max(1, min(zx, w - 1 - zw))
    zy = max(1, min(zy, h - 1 - zh))
    bw = rng.choice([2, 3])
    bh = rng.choice([2, 3])
    y = zy
    while y + bh <= zy + zh:
        x = zx
        while x + bw <= zx + zw:
            for yy in range(y, y + bh):
                for xx in range(x, x + bw):
                    walls.add((xx, yy))
            x += bw + 1                     # 1-wide aisle
        y += bh + 1
    return walls


def _pick_parking(rng, free_set, walls, w, h, taken, n):
    """n distinct OPEN-floor cells (degree >= 3) to park in, avoiding `taken`
    (all starts/goals) and each other. Falls back to degree >= 2 if needed."""
    for min_deg in (3, 2):
        cand = [c for c in free_set
                if c not in taken and _free_degree(c, walls, w, h) >= min_deg]
        if len(cand) >= n:
            return rng.sample(cand, n)
    return None


def make_scenario(rng, w, h, n_agents, flavour, tries=80):
    """
    Build one random warehouse scenario of the given flavour.
    Returns (walls, debris, agents) or None.
      walls  : set of (x,y) obstacle cells (INCLUDING debris - they are obstacles)
      debris : set of (x,y) cells that are fallen boxes (subset of walls; for colour)
      agents : [{"id", "start", "goal", "park"}, ...]  park is an open-floor cell
    """
    for _ in range(tries):
        base = flavour if flavour != "debris" else rng.choice(["spread", "cluster"])
        walls = _spread_walls(rng, w, h) if base == "spread" else _cluster_walls(rng, w, h)
        debris = set()

        free = [(x, y) for y in range(h) for x in range(w) if (x, y) not in walls]
        if len(free) < 3 * n_agents or not _connected(set(free)):
            continue

        # Fallen boxes: a few random free cells become obstacles (kept in spec).
        if flavour == "debris":
            budget = int(0.25 * w * h) - len(walls)
            k = min(rng.randint(3, 8), max(0, budget))
            if k:
                cand = rng.sample(free, min(len(free), 40))
                for c in cand:
                    if len(debris) >= k:
                        break
                    trial = walls | debris | {c}
                    rest = [f for f in free if f not in debris and f != c]
                    if _connected(rest):
                        debris.add(c)
            walls = walls | debris
            free = [f for f in free if f not in debris]

        dens = len(walls) / (w * h)
        if not (0.10 <= dens <= 0.25):
            continue
        if len(free) < 3 * n_agents or not _connected(set(free)):
            continue

        # starts + goals (all distinct), then open-floor parking cells.
        pk = rng.sample(free, 2 * n_agents)
        starts = [pk[2 * i] for i in range(n_agents)]
        goals = [pk[2 * i + 1] for i in range(n_agents)]
        taken = set(starts) | set(goals)
        parks = _pick_parking(rng, set(free), walls, w, h, taken, n_agents)
        if parks is None:
            continue
        agents = [{"id": i, "start": starts[i], "goal": goals[i], "park": parks[i]}
                  for i in range(n_agents)]
        return walls, debris, agents
    return None


# Backwards-compatible helper (spread flavour, parks on goal) for old callers.
def make_warehouse_map(rng, w, h, n_agents, tries=60):
    out = make_scenario(rng, w, h, n_agents, "spread", tries)
    if out is None:
        return None
    walls, _debris, agents = out
    for a in agents:
        a.pop("park", None)
    return walls, agents
