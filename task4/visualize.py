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
import math
import os
import random
import sys

import pygame
import pygame.gfxdraw

STEP = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "Wait": (0, 0)}
AGENT = [(59, 99, 212), (217, 138, 43), (138, 79, 208),
         (31, 157, 107), (0, 150, 170), (200, 90, 140)]

def apply_theme(dark):
    """Set every palette global for light or dark mode. Drawing code just reads
    these names, so one call reskins the whole visualizer."""
    global BG, PANEL, LINE, INK, MUTED, CELL_FILL, CELL_LN
    global SHELF_TOP, SHELF_BOT, SHELF_EDGE, DEBRIS_FILL, DEBRIS_EDGE, DEBRIS_INK
    global AMBER, CRASH, OKGREEN, ACCENT, CHIP_BG, TILE_BG
    global SHADOW, SHADOW2, ROBOT_SH, TIMELINE_BG, BTN_BG, BTN_BORDER
    if not dark:
        BG = (238, 241, 246); PANEL = (255, 255, 255); LINE = (225, 229, 238)
        INK = (29, 36, 51); MUTED = (95, 102, 120)
        CELL_FILL = (247, 249, 252); CELL_LN = (219, 225, 236)
        SHELF_TOP = (198, 208, 230); SHELF_BOT = (147, 162, 198); SHELF_EDGE = (117, 133, 168)
        DEBRIS_FILL = (251, 234, 210); DEBRIS_EDGE = (221, 138, 57); DEBRIS_INK = (207, 101, 18)
        AMBER = (224, 138, 43); CRASH = (226, 70, 70); OKGREEN = (46, 158, 110); ACCENT = (59, 99, 212)
        CHIP_BG = (237, 240, 247); TILE_BG = (244, 246, 250)
        SHADOW = (214, 219, 230); SHADOW2 = (210, 216, 230); ROBOT_SH = (206, 212, 226)
        TIMELINE_BG = (225, 230, 240); BTN_BG = (234, 238, 246); BTN_BORDER = (214, 220, 232)
    else:
        BG = (16, 19, 26); PANEL = (26, 31, 41); LINE = (44, 51, 66)
        INK = (232, 235, 242); MUTED = (156, 165, 183)
        CELL_FILL = (31, 37, 49); CELL_LN = (46, 54, 70)
        SHELF_TOP = (74, 86, 116); SHELF_BOT = (50, 60, 86); SHELF_EDGE = (96, 110, 146)
        DEBRIS_FILL = (78, 60, 38); DEBRIS_EDGE = (201, 132, 60); DEBRIS_INK = (240, 174, 96)
        AMBER = (228, 164, 92); CRASH = (233, 112, 112); OKGREEN = (76, 194, 142); ACCENT = (125, 155, 255)
        CHIP_BG = (40, 47, 64); TILE_BG = (32, 38, 50)
        SHADOW = (10, 12, 18); SHADOW2 = (10, 12, 18); ROBOT_SH = (8, 10, 15)
        TIMELINE_BG = (42, 49, 64); BTN_BG = (40, 47, 64); BTN_BORDER = (56, 64, 82)


apply_theme(False)

CELL = 40
PAD = 26
HEADER = 86         # room for the title + scenario line above the board
FOOTER = 104        # status line + clickable control bar + timeline + hint
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


def _toward_bg(c, f):
    """Blend colour c toward the floor colour by fraction f (1 = invisible)."""
    f = max(0.0, min(1.0, f))
    return tuple(int(c[k] + (CELL_FILL[k] - c[k]) * f) for k in range(3))


def _ease(t):
    """smootherstep — gentle accelerate/decelerate between cells."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _aa_disc(surf, cx, cy, r, col):
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    if r <= 0:
        return
    pygame.gfxdraw.filled_circle(surf, cx, cy, r, col)
    pygame.gfxdraw.aacircle(surf, cx, cy, r, col)


def _aa_ring(surf, cx, cy, r, col, width=2):
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    for rr in range(r, max(0, r - width), -1):
        pygame.gfxdraw.aacircle(surf, cx, cy, rr, col)


def _darken(c, f=0.55):
    return tuple(int(v * f) for v in c)


def build_sounds():
    """Generate short tones in-memory (no sound files needed). Returns a dict of
    pygame Sounds, or None if audio can't start (e.g. headless / no numpy)."""
    try:
        import numpy as np
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
    except Exception:
        return None
    sr = 22050

    def tone(segs, vol=0.3):
        import numpy as np
        buf = []
        for freq, ms in segs:               # freq=None → noise (collision)
            n = int(sr * ms / 1000)
            tt = np.linspace(0, ms / 1000, n, False)
            w = (np.random.uniform(-1, 1, n) if freq is None
                 else np.sin(2 * np.pi * freq * tt))
            env = np.ones(n)
            a = max(1, int(n * 0.08)); d = max(1, int(n * 0.35))
            env[:a] = np.linspace(0, 1, a); env[-d:] = np.linspace(1, 0, d)
            buf.append(w * env)
        w = np.concatenate(buf) * vol
        a16 = np.int16(np.clip(w, -1, 1) * 32767)
        return pygame.sndarray.make_sound(np.column_stack([a16, a16]).copy())

    try:
        return {
            "step":  tone([(190, 30)], 0.045),
            "pick":  tone([(660, 90), (880, 110)], 0.16),
            "park":  tone([(523, 150)], 0.14),
            "yield": tone([(400, 70)], 0.12),
            "crash": tone([(None, 180)], 0.18),
        }
    except Exception:
        return None


def step_sound(frames, t, goals, finals):
    """Which single sound best describes the transition into step t."""
    if t <= 0:
        return None
    cur, pre = frames[t], frames[t - 1]
    ids = sorted(cur)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if cur[i] == cur[j] or (cur[i] == pre[j] and cur[j] == pre[i]):
                return "crash"
    if any(cur[i] == goals[i] and pre[i] != goals[i] for i in ids):
        return "pick"
    if any(cur[i] == finals[i] and pre[i] != finals[i] for i in ids):
        return "park"
    if any(cur[i] == pre[i] and cur[i] != finals[i] for i in ids):
        return "yield"
    if any(cur[i] != pre[i] for i in ids):
        return "step"
    return None


def _dir_arrow(surf, cx, cy, dx, dy, rad, col):
    """A bold arrowhead sticking out of the robot in its heading direction —
    an unmistakable 'which way am I going' cue, not a dot."""
    n = (dx * dx + dy * dy) ** 0.5 or 1
    ux, uy = dx / n, dy / n
    perpx, perpy = -uy, ux
    tip = (cx + ux * (rad + 10), cy + uy * (rad + 10))
    baseL = (cx + ux * (rad + 1) + perpx * 7, cy + uy * (rad + 1) + perpy * 7)
    baseR = (cx + ux * (rad + 1) - perpx * 7, cy + uy * (rad + 1) - perpy * 7)
    pts = [(int(round(a)), int(round(b))) for a, b in (tip, baseL, baseR)]
    pygame.gfxdraw.filled_polygon(surf, pts, col)
    pygame.gfxdraw.aapolygon(surf, pts, (255, 255, 255))


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


def outcome_of(s, frames):
    """Classify a run from its frames: SUCCESS (all parked, no crash),
    COLLISION (a vertex/swap clash happened), or DEADLOCK (nobody crashed but
    not everyone reached their parking cell). Matches evaluate.py's rules."""
    goals = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
    finals = {a["id"]: tuple(a["park"]) if a.get("park") else tuple(a["goal"])
              for a in s["agents"]}
    ids = sorted(goals)
    collisions = 0
    for t in range(1, len(frames)):
        cur, pre = frames[t], frames[t - 1]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                if cur[i] == cur[j] or (cur[i] == pre[j] and cur[j] == pre[i]):
                    collisions += 1
    reached = all(frames[-1][i] == finals[i] for i in ids)
    result = "COLLISION" if collisions else ("SUCCESS" if reached else "DEADLOCK")
    return dict(collisions=collisions, reached=reached, result=result)


def full_stats(s, frames, label):
    """stats_of + outcome_of + the policy label, in one dict for the HUD."""
    d = stats_of(s, frames)
    d.update(outcome_of(s, frames))
    d["label"] = label
    return d


def draw(surf, s, frames, t, fonts, meta, stats, fov=None, frac=0.0,
         playing=True, speed=3.0, muted=False, controls=False, dark=False):
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
    surf.blit(meta_r, (ox + w * CELL - meta_r.get_width(), 10))
    # play state · speed · sound  (small status line under the scenario name)
    status = ("%s  ·  speed x%d  ·  sound %s" %
              ("PLAYING" if playing else "PAUSED", int(speed), "off" if muted else "on"))
    sc = ACCENT if playing else AMBER
    st_r = f_s.render(status, True, sc)
    surf.blit(st_r, (ox + w * CELL - st_r.get_width(), 30))

    # ---- policy label + outcome badge (left, below the title) ----
    label = stats.get("label", "")
    result = stats.get("result", "")
    if label:
        lab_r = f_b.render(label, True, INK)
        surf.blit(lab_r, (PAD, 52))
        if result:
            rc = {"SUCCESS": OKGREEN, "COLLISION": CRASH, "DEADLOCK": AMBER}.get(result, MUTED)
            dotx = PAD + lab_r.get_width() + 16
            doty = 52 + lab_r.get_height() // 2
            pygame.gfxdraw.filled_circle(surf, dotx, doty, 5, rc)
            pygame.gfxdraw.aacircle(surf, dotx, doty, 5, rc)
            res_r = f_b.render(result, True, rc)
            surf.blit(res_r, (dotx + 10, 52))

    # ---- board backing card (sits fully below the header band) ----
    _rrect(surf, SHADOW, (ox - 9, oy - 7, w * CELL + 20, h * CELL + 20), 16)  # shadow
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
                pygame.draw.rect(surf, SHADOW2, sr.move(0, 5), border_radius=7)  # shadow
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

    # trails — a smooth line that fades into the past, brightest near the robot
    nxt_t = frames[min(t + 1, len(frames) - 1)]
    e_t = _ease(frac)
    for i in sorted(frame):
        c = AGENT[i % len(AGENT)]
        pts = []
        for k in range(t + 1):
            cx, cy = px(*frames[k][i])
            pts.append((cx + CELL / 2, cy + CELL / 2))
        # extend to the robot's interpolated head so the trail stays attached
        p, pn = frame[i], nxt_t[i]
        hx = ox + (p[0] + (pn[0] - p[0]) * e_t) * CELL + CELL / 2
        hy = oy + (p[1] + (pn[1] - p[1]) * e_t) * CELL + CELL / 2
        pts.append((hx, hy))
        # dedupe consecutive identical points
        line = [pts[0]]
        for q in pts[1:]:
            if q != line[-1]:
                line.append(q)
        m = len(line)
        for j in range(1, m):
            age = j / m                      # 0 = oldest, 1 = newest
            # fade strongly toward the board background: old segments nearly
            # vanish, only the stretch near the robot stays vivid
            col = _toward_bg(c, 0.90 - 0.80 * age)
            wdt = 2 if age < 0.45 else 3
            pygame.draw.line(surf, col, line[j - 1], line[j], wdt)

    # collisions this step
    crashed = set()
    ids = sorted(frame)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if frame[i] == frame[j] or (frame[i] == prev[j] and frame[j] == prev[i]):
                crashed.add(frame[i]); crashed.add(frame[j])

    # robots — smoothly interpolated between step t and t+1
    nxt = frames[min(t + 1, len(frames) - 1)]
    e = _ease(frac)
    now = t + frac
    rad = CELL // 2 - 6
    waiting = []
    for i in sorted(frame):
        c = AGENT[i % len(AGENT)]
        p, pn = frame[i], nxt[i]
        pick_step = next((k for k in range(len(frames)) if frames[k][i] == goals[i]), None)
        picked = pick_step is not None and now >= pick_step
        fx = p[0] + (pn[0] - p[0]) * e
        fy = p[1] + (pn[1] - p[1]) * e
        cxp = ox + fx * CELL + CELL / 2
        cyp = oy + fy * CELL + CELL / 2
        parked = p == finals[i] and pn == finals[i] and t > 0
        is_wait = t > 0 and p == prev[i] and p == pn and p != finals[i]
        moving = pn != p
        col = lighten(c, 0.45) if parked else c                 # dim when parked

        # pick pulse — an expanding ring the moment it grabs its item
        if pick_step is not None and 0 <= now - pick_step < 1.2:
            pr = now - pick_step
            gx, gy = px(*goals[i])
            _aa_ring(surf, gx + CELL / 2, gy + CELL / 2, rad + 3 + pr * 16,
                     lighten(OKGREEN, pr * 0.8), 2)
        # yield halo — expanding amber ring while a robot holds back (shield at work)
        if is_wait:
            waiting.append(i)
            _aa_ring(surf, cxp, cyp, rad + 4 + (now % 1) * 7,
                     lighten(AMBER, (now % 1) * 0.6), 2)
            _aa_ring(surf, cxp, cyp, rad + 5, AMBER, 3)
        if p in crashed:
            _aa_ring(surf, cxp, cyp, rad + 7, CRASH, 4)

        # heading: travel direction if moving, else face the current target
        if moving:
            hx, hy = pn[0] - p[0], pn[1] - p[1]
        else:
            tgt = finals[i] if picked else goals[i]
            hx, hy = tgt[0] - p[0], tgt[1] - p[1]

        _aa_disc(surf, cxp, cyp + 4, rad, ROBOT_SH)      # soft moving shadow
        if not parked and (hx or hy):                           # clear direction arrow
            _dir_arrow(surf, cxp, cyp, hx, hy, rad, _darken(c))
        _aa_disc(surf, cxp, cyp, rad, col)                      # body (anti-aliased)
        _aa_disc(surf, cxp - rad * 0.3, cyp - rad * 0.3, rad * 0.5,
                 lighten(col, 0.4))                             # gloss highlight
        _aa_ring(surf, cxp, cyp, rad, lighten(col, 0.55), 2)
        if picked:                                              # carried crate
            bs = 9
            br = pygame.Rect(int(cxp - bs / 2), int(cyp - rad - bs + 5), bs, bs)
            pygame.draw.rect(surf, DEBRIS_FILL, br, border_radius=2)
            pygame.draw.rect(surf, DEBRIS_EDGE, br, 1, border_radius=2)
        lbl = f_n.render(str(i), True, (255, 255, 255))
        surf.blit(lbl, (cxp - lbl.get_width() / 2, cyp - lbl.get_height() / 2))
        if parked:
            _aa_disc(surf, cxp + rad - 2, cyp - rad + 2, 7, OKGREEN)
            chk = f_n.render("✓", True, (255, 255, 255))
            surf.blit(chk, (cxp + rad - 2 - chk.get_width() / 2,
                            cyp - rad + 2 - chk.get_height() / 2))

    # ---- side panel ----
    panel_x = ox + w * CELL + 24
    _draw_panel(surf, panel_x, oy, s, frame, prev, goals, finals, waiting,
                crashed, stats, t, fonts)

    # ---- footer status ----
    fy = oy + h * CELL + 12
    if crashed:
        msg, col = "Collision!", CRASH
    elif waiting:
        msg, col = "Robot %s waiting — yielding" % ", ".join(map(str, waiting)), AMBER
    elif all(frame[i] == finals[i] for i in frame):
        if stats.get("result") == "COLLISION":
            msg, col = "Parked — but a collision happened en route · not a valid solution", CRASH
        else:
            msg, col = "All robots done · parked clear of the aisles", OKGREEN
    elif stats.get("result") == "DEADLOCK" and t >= len(frames) - 1:
        msg, col = "Deadlock — robots stuck waiting · not everyone reached their spot", AMBER
    else:
        msg, col = "running · step %d / %d" % (t, len(frames) - 1), MUTED
    surf.blit(f_b.render(msg, True, col), (PAD, fy))
    # clickable control bar
    ctrl = _draw_controls(surf, fonts, PAD, fy + 22, playing, speed, muted, dark) if controls else {}
    # progress timeline
    tl_y, tl_w = fy + (56 if controls else 26), w * CELL
    _rrect(surf, TIMELINE_BG, (PAD, tl_y, tl_w, 6), 3)
    prog = (t + frac) / max(1, len(frames) - 1)
    _rrect(surf, ACCENT, (PAD, tl_y, int(tl_w * min(1.0, prog)), 6), 3)
    hint = f_s.render("Click a button above, or keys: SPACE play · ←/→ step · N/P map · +/− speed · M mute · T theme · Q quit",
                      True, MUTED)
    surf.blit(hint, (PAD, tl_y + 12))
    return ctrl


def _draw_controls(surf, fonts, x0, y, playing, speed, muted, dark):
    """A row of clickable buttons. Returns {id: Rect} for hit-testing."""
    f_h, f_b, f_s, f_n, f_stat = fonts
    specs = [("map_prev", "|< Map"), ("step_prev", "< Step"),
             ("play", "|| Pause" if playing else "> Play"),
             ("step_next", "Step >"), ("map_next", "Map >|"),
             ("slower", "–"), ("speed", "x%d" % int(speed)), ("faster", "+"),
             ("mute", "Sound: off" if muted else "Sound: on"),
             ("theme", "Light" if dark else "Dark")]
    rects = {}
    bx = x0
    for bid, label in specs:
        tw = max(27, f_s.size(label)[0] + 18)
        rect = pygame.Rect(bx, y, tw, 27)
        hot = (bid == "play")
        info = (bid == "speed")                       # non-button display chip
        _rrect(surf, ACCENT if hot else (TILE_BG if info else BTN_BG), rect, 8)
        if not hot and not info:
            pygame.draw.rect(surf, BTN_BORDER, rect, 1, border_radius=8)
        tcol = (255, 255, 255) if hot else (MUTED if info else INK)
        tx = bx + (tw - f_s.size(label)[0]) // 2
        surf.blit(f_s.render(label, True, tcol), (tx, y + 6))
        rects[bid] = rect
        bx += tw + 6
    return rects


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
    _rrect(surf, SHADOW, (x + 1, y + 2, W, fleet_h), 16)      # shadow
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
        pygame.draw.circle(surf, SHADOW2, (x + 24, ry + 12), 8)
        pygame.draw.circle(surf, c, (x + 24, ry + 11), 8)
        surf.blit(f_b.render("Robot %d" % i, True, INK), (x + 40, ry + 3))
        _chip(surf, x + W - 14, ry + 1, st[0], st[1], f_n)

    # ---- STATISTICS card ----
    sy = y + fleet_h + 16
    _coll = stats.get("collisions", 0)
    _csub = "shielded" if "Shield" in stats.get("label", "") else ("none" if not _coll else "unshielded")
    tiles = [("MAKESPAN", stats["makespan"], "steps"),
             ("FLOWTIME", stats["flowtime"], "sum of costs"),
             ("TOTAL WAIT", stats["total_w"], "steps"),
             ("COLLISIONS", _coll, _csub)]
    tw = (W - 34) // 2
    th = 62
    stats_h = 44 + 2 * th + 10
    _rrect(surf, SHADOW, (x + 1, sy + 2, W, stats_h), 16)     # shadow
    _rrect(surf, PANEL, (x, sy, W, stats_h), 16)
    surf.blit(f_n.render("STATISTICS", True, MUTED), (x + 16, sy + 14))
    for k, (lab, val, sub) in enumerate(tiles):
        tx = x + 13 + (k % 2) * (tw + 8)
        ty = sy + 38 + (k // 2) * (th + 6)
        _rrect(surf, TILE_BG, (tx, ty, tw, th), 10)
        vcol = CRASH if (lab == "COLLISIONS" and val) else INK
        surf.blit(f_n.render(lab, True, MUTED), (tx + 11, ty + 10))
        surf.blit(f_stat.render(str(val), True, vcol), (tx + 11, ty + 24))
        surf.blit(f_n.render(sub, True, MUTED), (tx + 11, ty + 46))

    # ---- LEGEND card ----
    ly = sy + stats_h + 16
    rows = [("arrow", ACCENT, "arrow = which way it's heading"),
            ("crate", DEBRIS_EDGE, "crate = carrying its item"),
            ("ring", ACCENT, "ring = its shelf (goal)"),
            ("square", ACCENT, "square = start point"),
            ("amber", AMBER, "amber halo = yielding (shield)"),
            ("check", OKGREEN, "green check = parked, done")]
    leg_h = 40 + len(rows) * 22
    _rrect(surf, SHADOW, (x + 1, ly + 2, W, leg_h), 16)
    _rrect(surf, PANEL, (x, ly, W, leg_h), 16)
    surf.blit(f_n.render("LEGEND", True, MUTED), (x + 16, ly + 14))
    for k, (kind, col, text) in enumerate(rows):
        ry = ly + 38 + k * 22
        gx, gy = x + 24, ry + 7
        if kind == "arrow":
            pygame.gfxdraw.filled_polygon(surf, [(gx + 7, gy), (gx - 4, gy - 5), (gx - 4, gy + 5)], col)
        elif kind == "crate":
            pygame.draw.rect(surf, DEBRIS_FILL, (gx - 5, gy - 5, 10, 10), border_radius=2)
            pygame.draw.rect(surf, col, (gx - 5, gy - 5, 10, 10), 1, border_radius=2)
        elif kind == "ring":
            pygame.draw.circle(surf, col, (gx, gy), 6, 2)
        elif kind == "square":
            pygame.draw.rect(surf, col, (gx - 5, gy - 5, 10, 10), 2, border_radius=2)
        elif kind == "amber":
            pygame.draw.circle(surf, col, (gx, gy), 6, 2)
        elif kind == "check":
            pygame.draw.circle(surf, col, (gx, gy), 6)
        surf.blit(f_s.render(text, True, MUTED), (x + 40, ry - 1))


def build_surface_size(s):
    w, h = s["grid"]["width"], s["grid"]["height"]
    return PAD * 2 + w * CELL + SIDE, HEADER + h * CELL + FOOTER


def main():
    ap = argparse.ArgumentParser(description="Task 4 MAPF visualizer (pygame).")
    ap.add_argument("scenario", help="a scenario .json, or a folder of them")
    ap.add_argument("--policy", default=None, choices=["cbs", "greedy", "model"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--shield", action="store_true",
                    help="wrap the policy with Task 3's safety shield (expect 0 collisions)")
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

    LABEL = ("CNN + Shield" if (args.policy == "model" and args.shield)
             else "CNN (no shield)" if args.policy == "model"
             else "Greedy + Shield" if (args.policy == "greedy" and args.shield)
             else "Greedy (no coordination)" if args.policy == "greedy"
             else "CBS (optimal expert)" if args.policy == "cbs"
             else "CBS (optimal, replayed)")

    _pol = {}

    def framesof(scn):
        # No policy -> replay the stored optimal (CBS) paths.
        # With a policy -> run it live through evaluate.rollout, which also
        # applies Task 3's shield and the goal->park schema, so the animation
        # matches the evaluation numbers exactly.
        if not args.policy:
            return frames_from_trajectories(scn)
        import evaluate
        if "p" not in _pol:
            _pol["p"] = evaluate.make_policy(args.policy, args.model)
        res = evaluate.rollout(scn, _pol["p"], shield=({} if args.shield else None))
        return res["frames"]

    s = load(files[idx]); frames = framesof(s); stats = full_stats(s, frames, LABEL)

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
        SUB = 6                       # sub-frames per logical step → smooth glide
        n = 0
        for t in range(len(frames) - 1):
            for k in range(SUB):
                draw(surf, s, frames, t, fonts, meta(), stats, frac=k / SUB)
                if args.export:
                    os.makedirs(args.export, exist_ok=True)
                    pygame.image.save(surf, os.path.join(args.export, "frame_%04d.png" % n))
                if args.gif:
                    images.append(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
                n += 1
        draw(surf, s, frames, len(frames) - 1, fonts, meta(), stats, frac=0.0)  # final rest
        if args.export:
            pygame.image.save(surf, os.path.join(args.export, "frame_%04d.png" % n))
        if args.gif:
            images.append(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
            import imageio
            images += [images[-1]] * 6
            imageio.mimsave(args.gif, images, duration=1.0 / (args.fps * SUB), loop=0)
            print("Saved GIF:", args.gif)
        if args.export:
            print("Saved %d frames to %s/" % (n + 1, args.export))
        return

    W, H = build_surface_size(s)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MAPF visualizer")
    clock = pygame.time.Clock()
    ft, playing, running = 0.0, True, True
    speed = float(max(1, args.fps))          # logical steps per second
    sounds = build_sounds()                  # None if audio unavailable
    muted = sounds is None
    last_step = 0

    def targets():
        g = {a["id"]: tuple(a["goal"]) for a in s["agents"]}
        f = {a["id"]: tuple(a["park"]) if a.get("park") else tuple(a["goal"])
             for a in s["agents"]}
        return g, f

    def load_index(i):
        nonlocal s, frames, stats, screen, ft, playing, last_step
        s = load(files[i]); frames = framesof(s); stats = full_stats(s, frames, LABEL)
        screen = pygame.display.set_mode(build_surface_size(s))
        ft, playing, last_step = 0.0, True, 0

    ctrl_rects = {}
    dark = False
    SPEEDS = [1, 2, 3, 5, 8, 12]

    def do(action):
        nonlocal ft, playing, speed, muted, idx, last_step, dark
        last = len(frames) - 1
        base = int(round(ft))                      # snap to the nearest whole step
        if action == "play":
            playing = not playing
        elif action == "step_next":
            ft, playing = min(base + 1, last), False
        elif action == "step_prev":
            ft, playing = max(base - 1, 0), False
        elif action == "map_next":
            idx = (idx + 1) % len(files); load_index(idx)
        elif action == "map_prev":
            idx = (idx - 1) % len(files); load_index(idx)
        elif action == "faster":
            speed = min(speed + 1, 16)
        elif action == "slower":
            speed = max(speed - 1, 1)
        elif action == "speed":                    # click the x3 chip to cycle presets
            nxt = next((v for v in SPEEDS if v > speed), SPEEDS[0])
            speed = float(nxt)
        elif action == "mute" and sounds:
            muted = not muted
        elif action == "theme":
            dark = not dark; apply_theme(dark)
        elif action == "restart":
            ft, playing, last_step = 0.0, True, 0

    while running:
        dt = clock.tick(60) / 1000.0
        last = len(frames) - 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for bid, rect in ctrl_rects.items():
                    if rect.collidepoint(e.pos):
                        do(bid); break
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_SPACE:
                    do("play")
                elif e.key == pygame.K_RIGHT:
                    do("step_next")
                elif e.key == pygame.K_LEFT:
                    do("step_prev")
                elif e.key == pygame.K_r:
                    do("restart")
                elif e.key == pygame.K_n:
                    do("map_next")
                elif e.key == pygame.K_p:
                    do("map_prev")
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    do("faster")
                elif e.key == pygame.K_MINUS:
                    do("slower")
                elif e.key == pygame.K_m:
                    do("mute")
                elif e.key == pygame.K_t:
                    do("theme")
        t = min(int(ft), last)
        frac = 0.0 if t >= last else ft - t
        # sound: play once per newly-entered logical step
        if sounds and not muted and t > last_step:
            g, f = targets()
            key = step_sound(frames, t, g, f)
            if key and key in sounds:
                sounds[key].play()
        last_step = t
        ctrl_rects = draw(screen, s, frames, t, fonts, meta(), stats,
                          frac=frac, playing=playing, speed=speed,
                          muted=muted, controls=True, dark=dark)
        pygame.display.flip()
        if playing:
            ft = min(ft + speed * dt, last)
            if ft >= last:
                playing = False
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
