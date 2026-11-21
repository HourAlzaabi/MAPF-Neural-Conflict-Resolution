"""
Task 3 — Safety Shield (priority-reservation shield).

The policy proposes one action per robot; the shield checks the whole set
jointly against the snapshot at time t and returns a move per robot that is
guaranteed collision-free for the step t -> t+1.

Guarantees (for every step):
  1. No wall / boundary violation (debris counts as a wall).
  2. No vertex conflict  (no two robots in the same cell at t+1).
  3. No swap/edge conflict (no two robots trade cells between t and t+1).
  4. Always returns a move for every robot ("Wait" is the universal fallback).

Conventions (match Task 1 / Task 2):
  N=(0,-1)  S=(0,1)  E=(1,0)  W=(-1,0)  Wait=(0,0)
  indices   0=N 1=S 2=E 3=W 4=Wait
  coordinates (x, y): x = column, y = row, (0,0) top-left, North decreases y.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]
RobotId = Hashable

ACTIONS: List[str] = ["N", "S", "E", "W", "Wait"]
DELTAS: Dict[str, Cell] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
    "Wait": (0, 0),
}
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def normalize_action(a) -> str:
    """Accept either a string ("N", "wait", ...) or an index 0..4."""
    if isinstance(a, int):
        return ACTIONS[a]
    s = str(a).strip()
    if s.lower() == "wait":
        return "Wait"
    s = s.upper()
    if s not in DELTAS:
        raise ValueError(f"Unknown action: {a!r}")
    return s


def apply(cell: Cell, action: str) -> Cell:
    dx, dy = DELTAS[action]
    return (cell[0] + dx, cell[1] + dy)


class SafetyShield:
    """
    Priority-reservation shield.

    Parameters
    ----------
    walls, width, height : static map (walls must include debris).
    priority : "id"       -> fixed order by robot id (deterministic)
               "rotate"   -> rotate the order every step (anti-deadlock)
               "distance" -> fewest steps-to-target first (needs dist_fn + targets)
               "random"   -> random order each step (seeded)
    fallback : "wait"        -> blocked robots Wait (spec baseline)
               "alternative" -> try other legal moves (closest to target first)
                                before Wait (spec's optional upgrade; needs
                                dist_fn + targets for the ordering, otherwise
                                uses a fixed N,S,E,W order)
    """

    def __init__(
        self,
        walls: Iterable[Cell],
        width: int,
        height: int,
        priority: str = "id",
        fallback: str = "wait",
        seed: int = 42,
    ):
        self.walls: Set[Cell] = {tuple(w) for w in walls}
        self.width = int(width)
        self.height = int(height)
        if priority not in {"id", "rotate", "distance", "random"}:
            raise ValueError(f"bad priority: {priority}")
        if fallback not in {"wait", "alternative"}:
            raise ValueError(f"bad fallback: {fallback}")
        self.priority = priority
        self.fallback = fallback
        self._step = 0
        self._rng = random.Random(seed)
        # statistics
        self.stats = {"steps": 0, "vetoes": 0, "substitutions": 0}

    # ------------------------------------------------------------------ utils
    def is_free(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.walls

    def _order(
        self,
        ids: List[RobotId],
        positions: Dict[RobotId, Cell],
        targets: Optional[Dict[RobotId, Cell]],
        dist_fn: Optional[Callable[[Cell, Cell], float]],
    ) -> List[RobotId]:
        ids = sorted(ids, key=lambda r: (str(type(r)), r))
        if self.priority == "id":
            return ids
        if self.priority == "rotate":
            k = self._step % len(ids) if ids else 0
            return ids[k:] + ids[:k]
        if self.priority == "random":
            ids = ids[:]
            self._rng.shuffle(ids)
            return ids
        # distance: fewest steps-to-target first; robots without info go last
        def key(r):
            if targets is None or dist_fn is None or r not in targets:
                return (float("inf"), r)
            return (dist_fn(positions[r], targets[r]), r)

        return sorted(ids, key=key)

    def _candidates(
        self,
        rid: RobotId,
        pos: Cell,
        proposed: str,
        targets: Optional[Dict[RobotId, Cell]],
        dist_fn: Optional[Callable[[Cell, Cell], float]],
    ) -> List[str]:
        """Proposed action first, then (optionally) alternatives. Wait is never
        in this list — it is the implicit final fallback."""
        out = [proposed] if proposed != "Wait" else []
        if self.fallback == "alternative" and proposed != "Wait":
            alts = [a for a in ("N", "S", "E", "W") if a != proposed]
            if targets is not None and dist_fn is not None and rid in targets:
                here = dist_fn(pos, targets[rid])
                scored = [(dist_fn(apply(pos, a), targets[rid]), i, a)
                          for i, a in enumerate(alts) if self.is_free(apply(pos, a))]
                # only alternatives that do not take the robot further than
                # staying put would (prevents pointless wandering)
                alts = [a for d, _, a in sorted(scored) if d <= here]
            out += alts
        return out

    # ------------------------------------------------------------------- core
    def filter(
        self,
        positions: Dict[RobotId, Cell],
        proposed: Dict[RobotId, object],
        targets: Optional[Dict[RobotId, Cell]] = None,
        dist_fn: Optional[Callable[[Cell, Cell], float]] = None,
    ) -> Dict[RobotId, str]:
        """Return a guaranteed-safe action for every robot in `positions`.

        Robots missing from `proposed` (e.g. parked / finished) are treated as
        "Wait": they keep occupying their cell as an obstacle.
        """
        positions = {r: tuple(p) for r, p in positions.items()}  # snapshot at t
        ids = list(positions.keys())

        # Pass 1: everybody tentatively stays — reserve every current cell.
        reserved: Dict[Cell, RobotId] = {}
        for r in ids:
            reserved[positions[r]] = r
        safe: Dict[RobotId, str] = {r: "Wait" for r in ids}

        # Pass 2: upgrade to the proposed move in priority order. Repeated until
        # nothing changes, so a follower blocked because the robot ahead had not
        # been checked yet gets a second chance once that robot has released
        # its cell (the "chase" case). Each upgrade still needs an unreserved
        # target cell, so the safety argument is unchanged.
        order = self._order(ids, positions, targets, dist_fn)
        want = {r: normalize_action(proposed.get(r, "Wait")) for r in ids}
        pending = [r for r in order if want[r] != "Wait"]
        changed = True
        while changed and pending:
            changed = False
            for r in list(pending):
                cand = apply(positions[r], want[r])
                if self.is_free(cand) and cand not in reserved:
                    del reserved[positions[r]]
                    reserved[cand] = r
                    safe[r] = want[r]
                    pending.remove(r)
                    changed = True

        # Pass 3 (optional): robots whose proposal was vetoed may take another
        # legal step that does not move them further from their target.
        if self.fallback == "alternative":
            for r in pending:
                pos = positions[r]
                for a in self._candidates(r, pos, want[r], targets, dist_fn)[1:]:
                    cand = apply(pos, a)
                    if self.is_free(cand) and cand not in reserved:
                        del reserved[pos]
                        reserved[cand] = r
                        safe[r] = a
                        break

        for r in ids:
            if safe[r] != want[r]:
                self.stats["vetoes"] += 1
                if safe[r] != "Wait":
                    self.stats["substitutions"] += 1

        self._step += 1
        self.stats["steps"] += 1
        return safe

    __call__ = filter


# ---------------------------------------------------------------- functional
def shield_step(
    positions: Dict[RobotId, Cell],
    proposed: Dict[RobotId, object],
    walls: Iterable[Cell],
    width: int,
    height: int,
) -> Dict[RobotId, str]:
    """Stateless one-shot version of the Section 3 contract (priority = id,
    fallback = Wait)."""
    return SafetyShield(walls, width, height).filter(positions, proposed)


def check_step(
    positions: Dict[RobotId, Cell],
    actions: Dict[RobotId, object],
    walls: Iterable[Cell],
    width: int,
    height: int,
) -> List[str]:
    """Independent checker: returns a list of violations for one step
    (empty list == safe). Used by the tests and the evaluator."""
    walls = {tuple(w) for w in walls}
    errs = []
    nxt = {}
    for r, p in positions.items():
        a = normalize_action(actions[r])
        c = apply(tuple(p), a)
        x, y = c
        if not (0 <= x < width and 0 <= y < height):
            errs.append(f"boundary: robot {r} -> {c}")
        elif c in walls:
            errs.append(f"wall: robot {r} -> {c}")
        nxt[r] = c
    seen = {}
    for r, c in nxt.items():
        if c in seen:
            errs.append(f"vertex: robots {seen[c]} & {r} at {c}")
        seen[c] = r
    ids = list(positions)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (nxt[a] == tuple(positions[b]) and nxt[b] == tuple(positions[a])
                    and nxt[a] != nxt[b]):
                errs.append(f"swap: robots {a} & {b}")
    return errs
