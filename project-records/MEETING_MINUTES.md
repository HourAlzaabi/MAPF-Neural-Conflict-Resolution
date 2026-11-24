# Meeting Minutes — Group COSC 11

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Supervisor:** Dr. Khaled Al-Wahedi (Assistant Professor, Computer Science, Khalifa University)
**Members:** Hour Alzaabi (100062976) · Hoor Almuhsen (100062910) · Aisha Al Naqbi (100063035) · Meera Alshehhi (100062797)

*Compiled from the team's working notes, Progress Report 1 (9 October 2026), the GitHub commit history, and supervisor email correspondence. From the progress-report period the team held **regular weekly working sessions (typically on Thursdays), with extra sessions in the final week**, plus two meetings with the supervisor, Dr. Khaled Al-Wahedi (kick-off on 10 September 2026 and a progress meeting on 16 November 2026).*

*These minutes, the other project records, and the report are the collaborative work of the whole team. Where documents were committed and pushed through Hour's GitHub account, that reflects Hour acting as the submitting representative on the team's behalf, not sole authorship — the per-task code commits are under each member's own account (see `CONTRIBUTIONS.md`).*

---

## Meeting 1 — Supervisor kick-off and task allocation
**Thursday 10 September 2026, 4:00 pm · Dr. Khaled's office** · Present: Hour, Hoor, Aisha, Meera, Dr. Khaled
*(Arranged by Dr. Khaled's email "SDP Team Allocation and Request for Kick-off Meeting", 9 Sept 2026.)*

**Decisions agreed:**
- Three-stage system: CBS expert (teacher) → CNN policy (apprentice) → safety shield (supervisor).
- **Supervisor's direction:** a warehouse setting with random variation; **flowtime (sum-of-costs)** as the headline cost.
- **Task allocation:** Task 1 — Hour; Task 2 — Meera; Task 3 — Hoor; **Task 4 (visualizer + evaluation) — Aisha**.
- Shared GitHub repository; each member commits their own task.

**Actions:** Hour to build the generator + CBS; all to set up Anaconda and the repo.

---

## Meeting 2 — Proposal, data schema and repository setup
**Thursday 1 October 2026** · Present: Hour, Hoor, Aisha, Meera

**Decisions agreed:**
- **Scenario format = JSON**, one file per scenario (human-readable, diff-able in Git), with an agreed coordinate convention ((x, y), (0,0) top-left, North decreases y) documented in `SCHEMA.md`.
- GitHub repository created; teammates added as collaborators; README with an open "Use of AI" disclosure.
- **Aisha:** confirmed the visualizer and evaluator will both read this JSON directly and **share the same policy code**, so the animation always shows exactly the behaviour that was measured.

**Actions:** Hour — finish the generator + CBS; all — prepare material for Progress Report 1.

---

## Meeting 3 — Finalise Progress Report 1
**Thursday 8 October 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:** reviewed and finalised Progress Report 1 (submitted 9 Oct 2026, Week 7): Task 1 complete; Task 2 started; the design decisions and the Weeks 1–15 plan. Agreed the weekly working-session cadence going forward.

**Actions:** Meera — begin CNN training scripts; Aisha — begin the evaluator and visualizer skeleton.

---

## Meeting 4 — Dataset, interfaces and the density decision
**Thursday 15 October 2026** · Present: Hour, Hoor, Aisha, Meera

**Decisions agreed (with reasons):**
- **Policy input = a local 11×11 window, 4 channels** (walls / this robot / other robots / guidance) — enough to see a neighbour and a route around an obstacle, small enough for one network for any map size.
- **1,500 scenarios, fixed seed 42**; actions N/S/E/W/Wait; **scenario-level 80/20 split** to avoid leakage.
- **Obstacle density:** the proposal specified 10–25%, but higher densities too often produced walled-off maps that the connectivity check rejects, so we **used 0.10–0.20 for reliably-connected maps**, and agreed to **include maps of varying density — keeping a denser variant as an option — to observe how crowding affects the conflict rate and the policy's behaviour**.

**Actions:** Hour — finalise the pipeline; Meera — training scripts; Aisha — evaluator.

---

## Meeting 5 — Task 2 design and visualizer planning (Aisha)
**Thursday 22 October 2026** · Present: Hour, Hoor, Aisha, Meera

**Decisions agreed:**
- **Behavioural cloning over RL** (CBS gives optimal labels); **a CNN** (grid-shaped input); **a greedy baseline** as the no-coordination floor; **four metrics** (success, collisions, flowtime, makespan).
- **Aisha presented the visualizer plan:** build it in **Python + pygame** (reusing the policy code, reading the Task 1 JSON directly). The visualizer must let a viewer answer three questions at a glance — *where is each robot going, which robot is giving way, did anything go wrong* — with a distinct symbol for each.

**Actions:** Meera — prototype the CNN; Aisha — build the evaluation harness and a first visualizer.

---

## Meeting 6 — Class imbalance and shield kickoff
**Thursday 29 October 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:** dataset analysis showed the rare **"Wait"** action is heavily outnumbered (class imbalance), so a naively trained CNN would under-learn the give-way. Agreed a **class-weighted loss** plus dropping redundant "already-parked, keep waiting" samples. Hoor to begin the **safety-shield design**.

**Actions:** Meera — class-weighted training; Hoor — shield design; Aisha — continue the harness and visualizer.

---

## Meeting 7 — First visualizer prototype review (Aisha)
**Thursday 5 November 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed:** Aisha demonstrated the first working visualizer and the evaluator; the team gave feedback on readability.

**Decisions / choices agreed (Aisha, Task 4):**
- **Direction must be shown by a clear heading arrow on each robot**, not a small direction dot — the dot was too hard to read at a glance.
- **Each robot keeps one fixed colour** for the whole run so it is easy to follow.
- Agreed the training setup for Meera's model: **Adam, lr 0.001, batch 64, 20 epochs, seed 0** for repeatability.

**Actions:** Aisha — refine the visualizer (arrows, colours, side panel); Meera — finish `train_v3`; Hoor — implement the shield.

---

## Meeting 8 — Shield design and first integration
**Thursday 12 November 2026** · Present: Hour, Hoor, Aisha, Meera

**Decisions agreed:**
- Adopt **`train_v3`** (class-weighted loss, Wait ≈ 2.83).
- **Shield = priority-reservation, veto-and-substitute**, checking only vertex and swap conflicts; it must wrap *any* policy and be proven on the greedy baseline.
- First end-to-end greedy-vs-CNN runs through Aisha's harness.

**Actions:** Hoor — implement `shield.py` + tests; Hour — map flavours + A* parking; Aisha — integrate the shield option into the evaluator.

---

## Meeting 9 — Supervisor progress review
**Monday 16 November 2026, 2:00 pm · Dr. Khaled's office** · Present: Hour, Hoor, Aisha, Meera, Dr. Khaled
*(Arranged by Dr. Khaled's email "SDP Project Progress Meeting", 11 Nov 2026.)*

**Supervisor's guidance:** a finished robot **must not block an aisle**; the final **report must be readable by a non-specialist** — every term explained, each choice justified — should **read as a continuous narrative**, and follow the **Handbook's phase-based structure**.

**Actions:** Hour — non-blocking parking + shelf-face goals; all — write their own report sections.

---

## Meeting 10 — Shield merge, parking fix, and visualizer rebuild kickoff (Aisha)
**Thursday 19 November 2026** · Present: Hour, Hoor, Aisha, Meera

**Decisions agreed:**
- Merge the shield to `main`; **repeat Pass 2 until stable** (safe following); keep the **fixed priority order**.
- **Parking rule v3:** park on open floor that is never a chokepoint and never an aisle mouth; **goals moved to shelf pick-faces**.
- Regenerate the dataset (seed 42) and **retrain** `model.pt`.
- **Aisha proposed rebuilding the visualizer for the demo** — the current version worked but looked plain. Agreed to make it polished and presentation-ready.

**Actions:** Hoor — merge; Hour — regenerate + retrain; Aisha — rebuild the visualizer.

---

## Meeting 11 — Visualizer demo polish (Aisha) and data-leakage discovery
**Saturday 22 November 2026** · Present: Hour, Hoor, Aisha, Meera

A weekend working session near the deadline, covering Aisha's rebuilt visualizer and a problem found while preparing results.

**Decisions / choices agreed — visualizer (Aisha, Task 4):**
- **Smooth, interpolated motion with an easing curve**; **a heading arrow** and **a carried crate** on pick; **a fading trail** with the fade **strengthened** so recent paths stand out; **reverted an earlier floor restyle** back to a plain, uncluttered floor.
- **A side panel** (per-robot status, statistics, legend); **clickable + keyboard controls** with **crisp single-press stepping**; **adjustable speed**; **light/dark mode**; **sound effects** (pick / park / wait / collision); **20-scenario browsing** and a **`--gif` export** for figures.
- Status markers: waiting = orange ring; parked = lighten + green tick; collision = red ring + "Collision!".

**Decisions / choices agreed — evaluation:**
- **Data-leakage issue identified:** evaluation had been run on all 1,500 maps, including training maps. Agreed to evaluate the model **only on the 300 held-out maps** it never trained on. `make_holdout.py` written and the 300 held-out scenario IDs committed the same day.

**Actions:** Aisha — finalise and push the rebuilt visualizer; Hour — confirm the held-out numbers and sync the change log.

---

## Meeting 12 — Honest results, records and submission prep
**Sunday 23 November 2026** · Present: Hour, Hoor, Aisha, Meera

**Decisions agreed:**
- Held-out results confirmed and reported everywhere: **Greedy 77.0% → CNN 82.7% → CNN + Shield 90.0% (0 collisions) → CBS 100%**; the change log (`CHANGES.md`) synced to these honest numbers.
- Keep the project records under `project-records/`; each member writes and commits their own report sections for individual grading.
- Submission plan: one report (**`COSC11.SDP1 Final Report`**) submitted by the Week-14 deadline (27 Nov); forms (Declaration + Health & Safety) prepared; rehearse the live demo; each member to book a 1:1 with the supervisor before the Week-15 presentation.

**Actions:** each member — finalise and commit their own sections; Hour — finalise and submit the records on the team's behalf; all — rehearse the demo.

---

## Key design decisions and rationale (team log)

| # | Decision | Why |
|---|---|---|
| 1 | Imitation learning (CBS → CNN), not RL | CBS gives optimal labels; supervised imitation is stable and fast |
| 2 | 11×11, 4-channel local view | Enough to see neighbours/obstacles; one network for any map size |
| 3 | Scenario-level 80/20 split + 300 held-out | Prevents data leakage; honest, reproducible results |
| 4 | Class-weighted loss (Wait ≈ 2.83) | The rare give-way is safety-critical and must not be ignored |
| 5 | Obstacle density 0.10–0.20, with a denser variant kept as an option | Keeps maps reliably connected while producing conflicts; varied density lets us study how crowding affects behaviour |
| 6 | Rule-based shield (priority-reservation) | Provable zero collisions, which a learned model cannot guarantee |
| 7 | Repeat Pass 2; fixed priority order | Allows safe "following"; simplest deterministic order, no worse than alternatives |
| 8 | Parking off-aisle (non-chokepoint, non-mouth) | A finished robot can never block a lane (supervisor requirement) |
| 9 | Shelf pick-face goals | Realistic "go to shelf and pick"; creates genuine conflicts to learn from |
| 10 | Flowtime as headline cost | Measures total fleet work (supervisor's steer) |
| 11 | Seed 42 (dataset), seed 0 (training split) | Full reproducibility of the maps and the reported numbers |
| 12 | Visualizer (Aisha): heading arrows, carried crate, strengthened fading trail, plain floor, legend + side panel, clickable + keyboard controls, adjustable speed, light/dark mode, sound; shared policy code with the evaluator | Makes direction, give-way and outcomes readable at a glance; the animation shows exactly what was measured |
