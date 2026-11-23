# AI Prompt Log — Group COSC 11

**Project:** Conflict Resolution in Multi-Agent Path Finding via Neural Imitation Learning
**Supervisor:** Dr. Khaled Al-Wahedi

As required by the Handbook, this log discloses where AI tools were used during the project. The Handbook permits AI-assisted code generation **with disclosure**, and this log provides that disclosure honestly.

**How AI was used.** AI tools (a large-language-model assistant) were used to help **draft and debug code**, to **explain background concepts**, and to **proof-read and tighten the report**. The team directed the work, made every design decision, and **reviewed, tested, understood and owns every line** of the submitted code. All experiments were run by the team, and every reported figure is reproducible from the committed dataset and the held-out scenario IDs — no result was taken from an AI.

| # | Member | Where AI was used | What the team did with it |
|---|---|---|---|
| 1 | Hour | Help drafting and debugging the warehouse generator and the back-off/parking logic (non-blocking parking) | Set the design; reviewed and tested the code; verified 0 parks land on a chokepoint or aisle mouth across all maps |
| 2 | Hour | Explanation of *data leakage* after a review comment, and help adding a held-out split | Understood the issue; built and ran `make_holdout.py`; re-ran the evaluation and reported the honest, lower numbers |
| 3 | Meera | Help drafting the CNN training loop and understanding class-weighted loss for a rare class | Chose and tuned the Wait weight (≈2.83); reviewed and tested the training code; confirmed the effect on the held-out confusion matrix |
| 4 | Hoor | Help implementing and debugging the shield (vertex/swap checks, fallback) and the test harness | Designed the priority-reservation scheme; reviewed every line; verified 0 collisions over 12,000 fuzz steps |
| 5 | Aisha | Help with pygame rendering details and structuring the evaluation harness | Built both tools; reviewed and tested them; confirmed the CBS-replay sanity check (100% / 0 collisions / 1.00×) |
| 6 | Team | Proof-reading and tightening the report, and checking it reads clearly for a non-specialist (per Dr. Khaled's guidance) | Wrote the content; used feedback to improve clarity, not to generate claims or results |

## Responsible-use statement
- AI assistance was at the level of a tutor, a pair-programmer and a proof-reader, used under the team's direction and with full disclosure.
- The team is responsible for, understands, and owns all submitted code and writing; all algorithm design, experiments and results are the team's own.
- Nothing in the report is fabricated: every figure is reproducible via `run_pipeline.bat` / `demo.bat` on the committed dataset.

*Members may attach a few representative verbatim prompts of their own below; only real prompts should be added.*
