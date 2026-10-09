"""
actions.py  -  turns a robot's path into the per-step action labels
               the neural network has to learn (Task 2's "answer").

ACTIONS is the fixed output order of the CNN: N, S, E, W, Wait.
Coordinates are [x, y]: x = column from the left, y = row from the top,
so North makes y smaller (see SCHEMA.md).
"""

# The order the CNN outputs its 5 probabilities in. Do not reorder.
ACTIONS = ["N", "S", "E", "W", "Wait"]

# How each action changes a cell [x, y].
STEP = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "Wait": (0, 0)}

# The reverse lookup: a change in position -> the action that caused it.
_FROM_STEP = {change: action for action, change in STEP.items()}


def trajectory_to_actions(path):
    """
    path = [[x, y], [x, y], ...]  (one cell per time step)
    returns ['E', 'N', 'Wait', ...] of length len(path) - 1.

    Each action is the move that takes the robot from one cell to the next.
    A repeated cell (the robot staying put) becomes 'Wait' - this is how a
    robot that yielded in a conflict shows up.
    """
    acts = []
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        change = (x2 - x1, y2 - y1)
        if change not in _FROM_STEP:
            raise ValueError(f"illegal move {change} in path {path}")
        acts.append(_FROM_STEP[change])
    return acts
