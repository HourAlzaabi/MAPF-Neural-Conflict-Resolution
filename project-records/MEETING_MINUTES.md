# Meeting Minutes — Group COSC 11

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Supervisor:** Dr. Khaled Al-Wahedi (Assistant Professor, Computer Science, Khalifa University)
**Members:** Hour Alzaabi (100062976) · Hoor Almuhsen (100062910) · Aisha Al Naqbi (100063035) · Meera Alshehhi (100062797)

*Compiled from the team's notes, the GitHub commit history, and supervisor email correspondence (23 Nov 2026). The team met with the supervisor, Dr. Khaled Al-Wahedi, on two occasions — the kick-off on 10 September 2026 and a progress meeting on 16 November 2026 — and held its own working sessions at project milestones in between. Where a discussion point is reconstructed from the project record rather than a verbatim transcript, it is phrased as the team's understanding.*

---

## 1. Supervisor meeting — Project kick-off and task allocation
**Thursday 10 September 2026, 4:00 pm · Dr. Khaled's office** · Present: Hour, Hoor, Aisha, Meera, Dr. Khaled
*(Arranged by Dr. Khaled's email "SDP Team Allocation and Request for Kick-off Meeting", 9 Sept 2026.)*

**Discussed / agreed:**
- Project scope confirmed: an optimal planner (CBS) teaches a neural network to resolve MAPF conflicts, with a safety shield enforcing collision-free execution.
- The team's understanding of the supervisor's direction: use a **warehouse setting with random variation** (not one fixed map), and treat **flowtime (sum-of-costs)** as the headline cost measure.
- **Task allocation:** Task 1 — Hour (environment + CBS + dataset); Task 2 — Meera (CNN policy); Task 3 — Hoor (safety shield); Task 4 — Aisha (visualizer + evaluation).
- Set up a shared GitHub repository; each member commits their own task.

**Actions:** Hour to build the generator + CBS so there is data to train on; all members to set up Anaconda and the repository.

---

## 2. Team working session — Dataset format and interfaces
**Early October 2026** (around the first commits, 4–9 Oct) · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:**
- The JSON scenario format that connects all four tasks (map + starts + goals + expert moves).
- The policy input: a **4-channel 11×11 local view** (walls / this robot / other robots / guidance).
- Reproducibility: fix the dataset seed so everyone generates identical maps — **1,500 scenarios, seed 42**, three map flavours (spread / cluster / debris); actions N/S/E/W/Wait; a scenario-level **80/20 train–validation split** to avoid leakage.

**Actions:** Hour to finalise the dataset pipeline (committed 4–9 Oct); Meera to start the Task 2 training scripts against this format.

---

## 3. Team working session — Task 2 design and Task 4 planning
**Mid October 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:**
- Why a **CNN** and **behavioural cloning** (imitate CBS) rather than reinforcement learning — optimal labels make supervised imitation stable and fast.
- Task 4 plan: a greedy baseline as the "no-coordination" floor, and the four metrics (success / makespan / flowtime / collisions).

**Actions:** Meera — prototype the CNN; Aisha — start the evaluation harness and 2-D visualizer.

---

## 4. Team working session — Integration of CNN, evaluator and visualizer
**Early–mid November 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:**
- Meera's first CNN was committed (10 Nov). The rare **"Wait"** action was being under-learned (class imbalance); agreed to address it with a **class-weighted loss**, integrated as `train_v3.py` (14 Nov).
- Aisha's `evaluate.py` and `visualize.py` landed (16 Nov) — the first end-to-end greedy-vs-CNN runs.
- Hoor to begin the safety-shield design in parallel.

**Actions:** Hoor — shield design and implementation; Hour — warehouse map flavours and the A* parking route.

---

## 5. Supervisor meeting — Progress review
**Monday 16 November 2026, 2:00 pm · Dr. Khaled's office** · Present: Hour, Hoor, Aisha, Meera, Dr. Khaled
*(Arranged by Dr. Khaled's email "SDP Project Progress Meeting", 11 Nov 2026.)*

**Discussed:**
- Progress review across the four tasks.
- The team's understanding of the supervisor's guidance: a finished robot **must not block an aisle**; the final **report must be readable by a non-specialist** — every term explained (seed, Prim, CBS, A*, flowtime) and each choice justified (e.g. why seed 42) — should **read as a continuous narrative**, and should follow the **Handbook's phase-based structure**.

**Actions:** Hour — implement the non-blocking parking rule and shelf-face goals; all — write their own report sections to the Handbook structure.

---

## 6. Team working session — Parking fix, shield merge and dataset regeneration
**17–21 November 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:**
- Hoor's **safety shield + tests committed** (18 Nov); `--shield` added to `evaluate.py`; branch `hoormuhsen-patch-1` merged into `main`.
- Parking rule finalised (**v3**): robots back off after picking and park where they can never block an aisle (not a chokepoint, not an aisle mouth); goals moved to **shelf pick-faces** (17–21 Nov).
- Dataset regenerated (seed 42) and `model.pt` **retrained** on it (21 Nov).

**Actions:** Aisha — push the rebuilt visualizer; Hour — regenerate, retrain and verify.

---

## 7. Team working session — Data-leakage fix, honest results and submission prep
**22–23 November 2026** · Present: Hour, Hoor, Aisha, Meera

**Discussed / agreed:**
- A **data-leakage issue** was identified during review: the evaluation had been run on all 1,500 maps, including maps the model trained on. Agreed to evaluate the model **only on the 300 held-out maps** it never trained on, for honest figures (`make_holdout.py`; the exact 300 IDs committed).
- Held-out results finalised: **Greedy 77.0% → CNN 82.7% → CNN + Shield 90.0% (0 collisions) → CBS 100%.**
- Remaining submission items reviewed: report sections, forms, title page, references, and the project records (minutes, contributions, AI-prompt log).

**Actions:** each member — complete their report sections and commit their own work; all — finalise the project records and rehearse the live demo.

---

*Further working sessions will be added here as they are held.*
