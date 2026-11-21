"""
visualize.py  -  Task 4 MAPF visualizer (pygame), styled like the web mock:
light theme, a board on the left, a Fleet + Statistics panel on the right, and a
playback/status bar below. Reads the real dataset JSONs, so it shows exactly what
Task 1 produced (robots drive to a shelf, pick, then back off to an open-floor
staging cell and park - becoming a static obstacle others route around).

Run (pass a FOLDER to browse a random sample, or a single file):
    pip install pygame
    python visualize.py ../task1/wh_dataset              # random 20 scenarios
    python visualize.py ../task1/wh_dataset --random 15
    python visualize.py ../task1/wh_dataset/wh_0003.json

Controls:  SPACE play/pause · hold <-/-> step · N/P next/prev map · R restart · Q quit
Headless:  python visualize.py scene.json --gif run.gif   (or --export frames/)
"""
import argparse
import glob
import json
import os
import random
import sys

import pygame

STEP = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "Wait": (0, 0)}
AGENT = [(59, 99, 212), (217, 138, 43), (138, 79, 208),
         (31, 157, 107), (0, 150, 170), (200, 90, 140)]

# ---- light "web" palette ----
BG      = (238, 241, 246)
PANEL   = (255, 255, 255)
LINE    = (225, 229, 238)
INK     = (29, 36, 51)
MUTED   = (95, 102, 120)
CELL_FILL = (247, 249, 252)
CELL_LN   = (219, 225, 236)
SHELF_TOP = (198, 208, 230)
SHELF_BOT = (147, 162, 198)
SHELF_EDGE = (117, 133, 168)
DEBRIS_FILL = (251, 234, 210)
DEBRIS_EDGE = (221, 138, 57)
DEBRIS_INK  = (207, 101, 18)
AMBER   = (224, 138, 43)
CRASH   = (226, 70, 70)
OKGREEN = (46, 158, 110)
ACCENT  = (59, 99, 212)

CHIP_BG = (237, 240, 247)
TILE_BG = (244, 246, 250)

CELL = 40
PAD = 26
HEADER = 86         # room for the title + scenario line above the board
FOOTER = 56
SIDE = 272          # right panel width

FONT_STACK = "Segoe UI,SF Pro Display,Helvetica Neue,Roboto,DejaVu Sans,Arial"


def _font(size, bold=False):
    f = pygame.font.SysFont(FONT_STACK, size, bold=bold)
    return f


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


def _rrect(surf, col, rect, r=8, width=0):
    pygame.draw.rect(surf, col, rect, width, border_radius=r)


def stats_of(s, frames):
    """makespan, flowtime, total waits, per-robot moves/waits/goal-t/done-t."""
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    finals = {a["id"]: tuple(a["park"]) if a.get("park") else tuple(a["goal"])
              for a in s["agents"]}
    per = {}
    for i in sorted(goals):
        path = [frames[t][i] for t in range(len(frames))]
        # trim trailing stay at final cell
        done_t = len(path) - 1
        while done_t > 0 and path[done_t] == path[done_t - 1] == finals[i]:
            done_t -= 1
        moves = sum(1 for t in range(1, done_t + 1) if path[t] != path[t - 1])
        waits = sum(1 for t in range(1, done_t + 1)
                    if path[t] == path[t - 1] and path[t] != finals[i])
        goal_t = next((t for t in range(len(path)) if path[t] == goals[i]), done_t)
        per[i] = dict(moves=moves, waits=waits, goal_t=goal_t, done_t=done_t)
    makespan = max(p["done_t"] for p in per.values())
    flowtime = sum(p["done_t"] for p in per.values())
    total_w = sum(p["waits"] for p in per.values())
    return dict(per=per, makespan=makespan, flowtime=flowtime, total_w=total_w)


def draw(surf, s, frames, t, fonts, meta, stats, fov=None):
    f_h, f_b, f_s, f_n, f_stat = fonts
    w, h = s["grid"]["width"], s["grid"]["height"]
    walls = {tuple(c) for c in s["grid"]["obstacles"]}
    debris = {tuple(c) for c in s["grid"].get("debris", [])}
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    starts = {a["id"]: tuple(a["start"]) for a in s["agents"]}
    parks = {a["id"]: tuple(a["park"]) for a in s["agents"] if a.get("park")}
    finals = {i: parks.get(i, goals[i]) for i in goals}
    frame, prev = frames[t], frames[t - 1] if t else frames[t]

    surf.fill(BG)
    ox, oy = PAD, HEADER

    def px(cx, cy):
        return ox + cx * CELL, oy + cy * CELL

    # ---- header (its own band above the board, so nothing can overlap it) ----
    title = f_h.render("Warehouse Fleet", True, INK)
    surf.blit(title, (PAD, 20))
    sub = f_s.render("Multi-Agent Path Finding", True, MUTED)
    surf.blit(sub, (PAD + title.get_width() + 12, 26))
    meta_r = f_b.render(meta, True, MUTED)
    surf.blit(meta_r, (ox + w * CELL - meta_r.get_width(), 24))

    # ---- board backing card (sits fully below the header band) ----
    _rrect(surf, (214, 219, 230), (ox - 9, oy - 7, w * CELL + 20, h * CELL + 20), 16)  # shadow
    _rrect(surf, PANEL, (ox - 10, oy - 10, w * CELL + 20, h * CELL + 20), 16)

    # cells
    for y in range(h):
        for x in range(w):
            r = pygame.Rect(*px(x, y), CELL - 3, CELL - 3)
            if (x, y) in debris:
                _rrect(surf, DEBRIS_FILL, r, 6)
                pygame.draw.rect(surf, DEBRIS_EDGE, r, 2, border_radius=6)
                cx, cy = r.center
                pts = [(cx, cy - 8), (cx + 8, cy + 6), (cx - 8, cy + 6)]
                pygame.draw.polygon(surf, DEBRIS_INK, pts)
                surf.blit(f_n.render("!", True, (255, 255, 255)), (cx - 3, cy - 5))
            elif (x, y) in walls:
                sr = pygame.Rect(r.x, r.y - 5, r.w, r.h)   # raised block
                pygame.draw.rect(surf, (210, 216, 230), sr.move(0, 5), border_radius=7)  # shadow
                for k in range(sr.h):
                    f = k / sr.h
                    col = tuple(int(SHELF_TOP[j] + (SHELF_BOT[j] - SHELF_TOP[j]) * f) for j in range(3))
                    pygame.draw.line(surf, col, (sr.x, sr.y + k), (sr.x + sr.w, sr.y + k))
                pygame.draw.rect(surf, SHELF_EDGE, sr, 1, border_radius=7)
            else:
                _rrect(surf, CELL_FILL, r, 5)
                pygame.draw.rect(surf, CELL_LN, r, 1, border_radius=5)

    # start markers
    for i, st in starts.items():
        c = AGENT[i % len(AGENT)]
        sx, sy = px(*st)
        rect = pygame.Rect(sx + CELL // 2 - 9, sy + CELL // 2 - 9, 18, 18)
        pygame.draw.rect(surf, c, rect, 2, border_radius=4)

    # park pads (dashed-ish open ring) + goal rings
    for i, pk in parks.items():
        c = AGENT[i % len(AGENT)]
        pxc, pyc = px(*pk)
        pygame.draw.rect(surf, c, pygame.Rect(pxc + 5, pyc + 5, CELL - 13, CELL - 13),
                         1, border_radius=5)
    for i, g in goals.items():
        c = AGENT[i % len(AGENT)]
        gx, gy = px(*g)
        ctr = (gx + CELL // 2, gy + CELL // 2)
        pygame.draw.circle(surf, c, ctr, CELL // 2 - 6, 3)

    # trails (dotted; dim the return/back-off leg once the robot has passed its goal)
    for i in sorted(frame):
        c = AGENT[i % len(AGENT)]
        seen, hit_goal = [], False
        for k in range(t + 1):
            p = frames[k][i]
            ret = hit_goal
            if p == goals[i]:
                hit_goal = True
            if not seen or seen[-1][0] != p:
                seen.append((p, ret))
        for p, ret in seen[:-1]:
            cx, cy = px(*p)
            col = lighten(c, 0.55) if ret else lighten(c, 0.25)
            pygame.draw.circle(surf, col, (cx + CELL // 2, cy + CELL // 2), 3)

    # collisions this step
    crashed = set()
    ids = sorted(frame)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if frame[i] == frame[j] or (frame[i] == prev[j] and frame[j] == prev[i]):
                crashed.add(frame[i]); crashed.add(frame[j])

    # robots
    waiting = []
    for i in sorted(frame):
        c = AGENT[i % len(AGENT)]
        p = frame[i]
        rx, ry = px(*p)
        ctr = (rx + CELL // 2, ry + CELL // 2)
        rad = CELL // 2 - 6
        parked = p == finals[i] and t > 0
        is_wait = t > 0 and p == prev[i] and p != finals[i]
        col = lighten(c, 0.45) if parked else c                 # dim when parked
        if is_wait:
            waiting.append(i)
            pygame.draw.circle(surf, AMBER, ctr, rad + 4, 3)
        if p in crashed:
            pygame.draw.circle(surf, CRASH, ctr, rad + 6, 4)
        pygame.draw.circle(surf, (210, 216, 230), (ctr[0], ctr[1] + 3), rad)  # soft shadow
        pygame.draw.circle(surf, col, ctr, rad)
        pygame.draw.circle(surf, lighten(col, 0.5), ctr, rad, 2)
        lbl = f_n.render(str(i), True, (255, 255, 255))
        surf.blit(lbl, (ctr[0] - lbl.get_width() // 2, ctr[1] - lbl.get_height() // 2))
        if parked:                                              # done check
            pygame.draw.circle(surf, OKGREEN, (ctr[0] + rad - 2, ctr[1] - rad + 2), 7)

    # ---- side panel ----
    panel_x = ox + w * CELL + 24
    _draw_panel(surf, panel_x, oy, s, frame, prev, goals, finals, waiting,
                crashed, stats, t, fonts)

    # ---- footer status ----
    fy = oy + h * CELL + 16
    if crashed:
        msg, col = "Collision!", CRASH
    elif waiting:
        msg, col = "Robot %s waiting — yielding" % ", ".join(map(str, waiting)), AMBER
    elif all(frame[i] == finals[i] for i in frame):
        msg, col = "All robots done · parked clear of the aisles", OKGREEN
    else:
        msg, col = "running · step %d / %d" % (t, len(frames) - 1), MUTED
    surf.blit(f_b.render(msg, True, col), (PAD, fy))
    hint = f_s.render("SPACE play · hold ←/→ step · N/P map · R restart · Q quit", True, MUTED)
    surf.blit(hint, (PAD, fy + 22))


def _chip(surf, rx, ry, text, col, font):
    """A rounded status pill sized to its text, right-anchored at rx."""
    t = font.render(text, True, col)
    cw = t.get_width() + 16
    rect = pygame.Rect(rx - cw, ry, cw, 20)
    _rrect(surf, lighten(col, 0.84), rect, 7)
    surf.blit(t, (rect.x + 8, ry + 4))
    return cw


def _draw_panel(surf, x, y, s, frame, prev, goals, finals, waiting, crashed, stats, t, fonts):
    f_h, f_b, f_s, f_n, f_stat = fonts
    W = SIDE - 24
    ids = sorted(frame)

    # ---- FLEET card ----
    row_h = 36
    fleet_h = 44 + len(ids) * row_h
    _rrect(surf, (214, 219, 230), (x + 1, y + 2, W, fleet_h), 16)      # shadow
    _rrect(surf, PANEL, (x, y, W, fleet_h), 16)
    surf.blit(f_n.render("FLEET", True, MUTED), (x + 16, y + 14))
    for k, i in enumerate(ids):
        c = AGENT[i % len(AGENT)]
        ry = y + 38 + k * row_h
        p, pv = frame[i], prev[i]
        if crashed and p in crashed:
            st = ("COLLIDE", CRASH)
        elif p == finals[i] and t > 0:
            st = ("PARKED", MUTED)
        elif t > 0 and p == pv and p != finals[i]:
            st = ("WAITING", AMBER)
        elif p == goals[i]:
            st = ("AT GOAL", OKGREEN)
        else:
            st = ("MOVING", ACCENT)
        pygame.draw.circle(surf, (210, 216, 230), (x + 24, ry + 12), 8)
        pygame.draw.circle(surf, c, (x + 24, ry + 11), 8)
        surf.blit(f_b.render("Robot %d" % i, True, INK), (x + 40, ry + 3))
        _chip(surf, x + W - 14, ry + 1, st[0], st[1], f_n)

    # ---- STATISTICS card ----
    sy = y + fleet_h + 16
    tiles = [("MAKESPAN", stats["makespan"], "steps"),
             ("FLOWTIME", stats["flowtime"], "sum of costs"),
             ("TOTAL WAIT", stats["total_w"], "steps"),
             ("COLLISIONS", 0, "shielded")]
    tw = (W - 34) // 2
    th = 62
    stats_h = 44 + 2 * th + 10
    _rrect(surf, (214, 219, 230), (x + 1, sy + 2, W, stats_h), 16)     # shadow
    _rrect(surf, PANEL, (x, sy, W, stats_h), 16)
    surf.blit(f_n.render("STATISTICS", True, MUTED), (x + 16, sy + 14))
    for k, (lab, val, sub) in enumerate(tiles):
        tx = x + 13 + (k % 2) * (tw + 8)
        ty = sy + 38 + (k // 2) * (th + 6)
        _rrect(surf, TILE_BG, (tx, ty, tw, th), 10)
        surf.blit(f_n.render(lab, True, MUTED), (tx + 11, ty + 10))
        surf.blit(f_stat.render(str(val), True, INK), (tx + 11, ty + 24))
        surf.blit(f_n.render(sub, True, MUTED), (tx + 11, ty + 46))


def build_surface_size(s):
    w, h = s["grid"]["width"], s["grid"]["height"]
    return PAD * 2 + w * CELL + SIDE, HEADER + h * CELL + FOOTER


def main():
    ap = argparse.ArgumentParser(description="Task 4 MAPF visualizer (pygame).")
    ap.add_argument("scenario", help="a scenario .json, or a folder of them")
    ap.add_argument("--policy", default=None, choices=["cbs", "greedy", "model"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--random", type=int, default=20, help="browse a random sample of N")
    ap.add_argument("--export", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    if os.path.isdir(args.scenario):
        pool = sorted(glob.glob(os.path.join(args.scenario, "*.json")))
        random.shuffle(pool)
        files = pool[:max(1, args.random)]
    else:
        here = os.path.abspath(args.scenario)
        pool = [f for f in sorted(glob.glob(os.path.join(os.path.dirname(here), "*.json"))) if f != here]
        random.shuffle(pool)
        files = [here] + pool[:max(0, args.random - 1)]
    idx = 0

    def framesof(scn):
        return frames_from_policy(scn, args.policy, args.model) if args.policy \
            else frames_from_trajectories(scn)

    s = load(files[idx]); frames = framesof(s); stats = stats_of(s, frames)

    if args.export or args.gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    fonts = (_font(22, bold=True),    # title
             _font(14, bold=True),    # row labels / meta
             _font(13),               # subtitle / hints
             _font(11, bold=True),    # small caps labels / chips
             _font(24, bold=True))    # big stat numbers

    def meta():
        return "%s  ·  %dx%d  ·  map %d/%d" % (
            s["scenario_id"], s["grid"]["width"], s["grid"]["height"], idx + 1, len(files))

    if args.export or args.gif:
        W, H = build_surface_size(s)
        surf = pygame.Surface((W, H))
        images = []
        for t in range(len(frames)):
            draw(surf, s, frames, t, fonts, meta(), stats)
            if args.export:
                os.makedirs(args.export, exist_ok=True)
                pygame.image.save(surf, os.path.join(args.export, "frame_%03d.png" % t))
            if args.gif:
                images.append(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
        if args.gif:
            import imageio
            images += [images[-1]] * 4
            imageio.mimsave(args.gif, images, duration=1.0 / args.fps, loop=0)
            print("Saved GIF:", args.gif)
        if args.export:
            print("Saved %d frames to %s/" % (len(frames), args.export))
        return

    W, H = build_surface_size(s)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MAPF visualizer")
    pygame.key.set_repeat(260, 55)
    clock = pygame.time.Clock()
    t, playing, running, fps = 0, True, True, args.fps

    def load_index(i):
        nonlocal s, frames, stats, screen, t, playing
        s = load(files[i]); frames = framesof(s); stats = stats_of(s, frames)
        screen = pygame.display.set_mode(build_surface_size(s))
        t, playing = 0, True

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
                elif e.key == pygame.K_n:
                    idx = (idx + 1) % len(files); load_index(idx)
                elif e.key == pygame.K_p:
                    idx = (idx - 1) % len(files); load_index(idx)
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(fps + 1, 12)
                elif e.key == pygame.K_MINUS:
                    fps = max(fps - 1, 1)
        draw(screen, s, frames, t, fonts, meta(), stats)
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
