"""
visualize.py  -  Task 4 2D visualizer (polished).

Plays a scenario as an animation: the grid, walls, each robot's fading trail,
start markers, goal rings, a highlight when a robot waits (yields), and a red
flash on any collision. By default it plays the optimal CBS paths stored in the
file; with --policy it simulates a policy step-by-step instead.

Run on your laptop:
    pip install pygame
    python visualize.py ../task1/dataset/map_0032.json
    python visualize.py ../task1/dataset/map_0032.json --fov 0
    python visualize.py ../task1/dataset/map_0032.json --policy greedy

Controls:  SPACE play/pause   -> / <- step   R restart   + / - speed   Q quit

No screen, or making a figure for the report/slides:
    python visualize.py ../task1/dataset/map_0032.json --gif run.gif
    python visualize.py ../task1/dataset/map_0032.json --export frames/
"""

import argparse
import json
import os
import sys

import pygame

STEP = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "Wait": (0, 0)}
AGENT = [(229, 72, 77), (47, 125, 246), (31, 157, 107),
         (232, 161, 58), (139, 134, 255), (0, 173, 181)]

BG = (247, 248, 251)
CARD = (255, 255, 255)
GRID = (226, 230, 238)
WALL = (46, 52, 72)
INK = (26, 29, 41)
MUTED = (124, 132, 150)
AMBER = (232, 161, 58)
CRASH = (229, 72, 77)

CELL = 38
PAD = 22
HEADER = 60
FOOTER = 44


def load(path):
    with open(path) as f:
        return json.load(f)


def frames_from_trajectories(s):
    traj = {int(k): [tuple(c) for c in v] for k, v in s["trajectories"].items()}
    T = max(len(p) for p in traj.values())
    return [{i: traj[i][min(t, len(traj[i]) - 1)] for i in traj} for t in range(T)]


def frames_from_policy(s, kind, model_path=None):
    import evaluate
    pol = evaluate.make_policy(kind, model_path)
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    state = {a["id"]: tuple(a["start"]) for a in s["agents"]}
    ids = sorted(state)
    pol.reset(s)
    out = [dict(state)]
    for t in range(max(2 * (w + h), 4 * s["makespan"] + 5)):
        if all(state[i] == goals[i] for i in ids):
            break
        nxt = {}
        for i in ids:
            dx, dy = STEP[pol.act(s, state, i, t)]
            nx, ny = state[i][0] + dx, state[i][1] + dy
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in walls:
                nx, ny = state[i]
            nxt[i] = (nx, ny)
        state = nxt
        out.append(dict(state))
    return out


def lighten(c, f):
    return tuple(int(v + (255 - v) * f) for v in c)


def draw(surf, s, frames, t, fonts, fov=None):
    f_title, f_small = fonts
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    starts = {a["id"]: tuple(a["start"]) for a in s["agents"]}
    frame, prev = frames[t], frames[t - 1] if t else frames[t]
    surf.fill(BG)
    ox, oy = PAD, HEADER

    def px(cx, cy):
        return ox + cx * CELL, oy + cy * CELL

    # board backing card
    pygame.draw.rect(surf, CARD, pygame.Rect(ox - 6, oy - 6, w * CELL + 12, h * CELL + 12),
                     border_radius=12)
    # cells + walls
    for y in range(h):
        for x in range(w):
            r = pygame.Rect(*px(x, y), CELL - 2, CELL - 2)
            if (x, y) in walls:
                pygame.draw.rect(surf, WALL, r, border_radius=5)
            else:
                pygame.draw.rect(surf, GRID, r, 1, border_radius=5)

    # trails (fading dots on visited cells)
    for i in sorted(frame):
        c = AGENT[i % len(AGENT)]
        seen = []
        for k in range(t + 1):
            p = frames[k][i]
            if p not in seen:
                seen.append(p)
        for j, p in enumerate(seen[:-1]):
            fade = 0.35 + 0.5 * (j / max(1, len(seen)))
            cx, cy = px(*p)
            pygame.draw.circle(surf, lighten(c, 1 - fade),
                               (cx + CELL // 2, cy + CELL // 2), 4)

    # start markers (hollow squares) + goals (rings, filled when reached)
    for i, st in starts.items():
        c = AGENT[i % len(AGENT)]
        sx, sy = px(*st)
        pygame.draw.rect(surf, c, pygame.Rect(sx + CELL // 2 - 6, sy + CELL // 2 - 6, 12, 12),
                         2, border_radius=3)
    for i, g in goals.items():
        c = AGENT[i % len(AGENT)]
        gx, gy = px(*g)
        ctr = (gx + CELL // 2, gy + CELL // 2)
        reached = frame[i] == g
        pygame.draw.circle(surf, c, ctr, CELL // 2 - 5, 0 if reached else 3)

    # collisions this step (vertex / swap)
    crashed = set()
    ids = sorted(frame)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if frame[i] == frame[j] or (frame[i] == prev[j] and frame[j] == prev[i]):
                crashed.add(frame[i])
                crashed.add(frame[j])

    # robots
    waiting_now = []
    for i in sorted(frame):
        c = AGENT[i % len(AGENT)]
        p = frame[i]
        rx, ry = px(*p)
        ctr = (rx + CELL // 2, ry + CELL // 2)
        rad = CELL // 2 - 5
        is_wait = t > 0 and p == prev[i] and p != goals[i]
        if is_wait:
            waiting_now.append(i)
            pygame.draw.circle(surf, AMBER, ctr, rad + 4, 3)
        if p in crashed:
            pygame.draw.circle(surf, CRASH, ctr, rad + 6, 4)
        pygame.draw.circle(surf, (20, 20, 30), (ctr[0] + 1, ctr[1] + 2), rad)   # shadow
        pygame.draw.circle(surf, c, ctr, rad)
        lbl = f_small.render(str(i), True, (255, 255, 255))
        surf.blit(lbl, (ctr[0] - lbl.get_width() // 2, ctr[1] - lbl.get_height() // 2))

    # header
    surf.blit(f_title.render(s["scenario_id"], True, INK), (PAD, 14))
    meta = f"{len(goals)} robots  ·  {w}x{h}  ·  {'conflict' if s['had_conflict'] else 'no conflict'}"
    surf.blit(f_small.render(meta, True, MUTED), (PAD, 40))
    # legend (right side of header)
    lx = ox + w * CELL - 1
    for i in sorted(goals, reverse=True):
        txt = f_small.render(f"R{i}", True, AGENT[i % len(AGENT)])
        lx -= txt.get_width() + 8
        surf.blit(txt, (lx, 20))
        lx -= 16
        pygame.draw.circle(surf, AGENT[i % len(AGENT)], (lx + 6, 26), 6)

    # FOV box
    if fov is not None and fov in frame:
        cx, cy = frame[fov]
        x0, y0 = px(cx - 5, cy - 5)
        pygame.draw.rect(surf, AGENT[fov % len(AGENT)],
                         pygame.Rect(x0, y0, 11 * CELL, 11 * CELL), 2, border_radius=8)

    # footer status
    fy = oy + h * CELL + 12
    if crashed:
        msg, col = "Collision!", CRASH
    elif waiting_now:
        msg, col = f"Robot {', '.join(map(str, waiting_now))} waiting (yield)", AMBER
    elif all(frame[i] == goals[i] for i in frame):
        msg, col = "All robots reached their goals", (31, 157, 107)
    else:
        msg, col = f"step {t} / {len(frames) - 1}", MUTED
    surf.blit(f_small.render(msg, True, col), (PAD, fy))
    hint = f_small.render("SPACE play · arrows step · R restart · +/- speed · Q quit", True, MUTED)
    surf.blit(hint, (ox + w * CELL - hint.get_width(), fy))


def build_surface_size(s):
    w, h = s["grid"]["width"], s["grid"]["height"]
    return PAD * 2 + w * CELL, HEADER + h * CELL + FOOTER


def main():
    ap = argparse.ArgumentParser(description="Task 4 MAPF visualizer.")
    ap.add_argument("scenario")
    ap.add_argument("--fov", type=int, default=None)
    ap.add_argument("--policy", default=None, choices=["cbs", "greedy", "model"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--export", default=None, help="save PNG frames to this dir")
    ap.add_argument("--gif", default=None, help="save an animated GIF to this file")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    s = load(args.scenario)
    frames = frames_from_policy(s, args.policy, args.model) if args.policy \
        else frames_from_trajectories(s)

    if args.export or args.gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    fonts = (pygame.font.SysFont("Arial", 20, bold=True), pygame.font.SysFont("Arial", 14))
    W, H = build_surface_size(s)

    if args.export or args.gif:
        surf = pygame.Surface((W, H))
        images = []
        for t in range(len(frames)):
            draw(surf, s, frames, t, fonts, args.fov)
            if args.export:
                os.makedirs(args.export, exist_ok=True)
                pygame.image.save(surf, os.path.join(args.export, f"frame_{t:03d}.png"))
            if args.gif:
                arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
                images.append(arr)
        if args.gif:
            import imageio
            images += [images[-1]] * 4          # hold the final frame
            imageio.mimsave(args.gif, images, duration=1.0 / args.fps, loop=0)
            print(f"Saved GIF: {args.gif}  ({len(frames)} frames)")
        if args.export:
            print(f"Saved {len(frames)} frames to {args.export}/")
        return

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MAPF visualizer - " + s["scenario_id"])
    clock = pygame.time.Clock()
    t, playing, running, fps = 0, True, True, args.fps
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_SPACE:
                    playing = not playing
                elif e.key == pygame.K_RIGHT:
                    t, playing = min(t + 1, len(frames) - 1), False
                elif e.key == pygame.K_LEFT:
                    t, playing = max(t - 1, 0), False
                elif e.key == pygame.K_r:
                    t, playing = 0, True
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(fps + 1, 12)
                elif e.key == pygame.K_MINUS:
                    fps = max(fps - 1, 1)
        draw(screen, s, frames, t, fonts, args.fov)
        pygame.display.flip()
        if playing and t < len(frames) - 1:
            t += 1
        elif playing:
            playing = False
        clock.tick(fps)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
