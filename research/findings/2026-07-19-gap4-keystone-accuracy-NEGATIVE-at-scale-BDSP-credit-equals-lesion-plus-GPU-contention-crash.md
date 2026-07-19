# gap#4 keystone accuracy — NEGATIVE at ep300/hidden128 (BDSP credit == lesion, held 0.420 < chance) + 3 arms CRASHED (my GPU over-parallelization). The credit-DIRECTION wall stands; the KP fix is UNRESOLVED (crashed).

**2026-07-19.** The 4 gap#4 keystone-accuracy arms (`_d1_onbridge_learn_to_accuracy --microcircuit`, emerge1, hidden=128,
seed 42) completed: 1 with a result, 3 crashed empty. Honest result + a self-caused-crash lesson.

## The one arm that finished (graded+KP ep300) — GO=false, accuracy NOT achieved
- **BDSP held-out 0.420 == LESION 0.420 == wrong-sign 0.420**, all BELOW chance 0.549. oracle (2-layer backprop) 0.983,
  numpy single-layer floor 0.510 ≈ chance → the task GENUINELY needs a hidden layer + correct deep credit.
- **The MECHANISM is validated** (both plastic pathways move under credit: in→hid dw 1973, hid→out 166; the P0 moat holds:
  apical-lesion hidden-dw 0.000 ≪ credit 1973; NO weight transport). But the **BDSP credit produces NO accuracy gain over
  the lesion** — it moves weights in a direction that does NOT help the task (BDSP == lesion == wrong-sign). ⇒ this is the
  credit-**DIRECTION** wall (the D2/D3 finding: graded credit fixes the moat but not the direction; feedback alignment's
  generic partiality). The keystone accuracy (held ≥0.75 on-bridge) is NOT achieved at this config.

## 3 arms CRASHED (fixed ep300, KP ep600, measured-B ep300) — GPU over-parallelization (my error)
- All 3 have 0-byte logs (mtime = launch time 01:42, NEVER written) and died in the same 90s window. No OOM in dmesg, RAM
  fine (38Gi free). **The cause is GPU-memory contention:** the gap#4 arms run on-bridge (CuPy/GPU), and I ran MULTIPLE
  concurrent n_ca3=2000 SWR runs (the sweep + validate6 + localize, ~each a large bridge) on the same 24GB GPU → a CUDA OOM
  crashed 3 arms (the stderr traceback was lost to the stdout-only redirect). **LESSON (silent-failure class): the gap#4
  arms USE the GPU; do NOT stack concurrent GPU-heavy runs on top of them. Watch `nvidia-smi` memory before launching; the
  gap#4 arms should run ISOLATED (or the SWR runs serialized).** The fixed-vs-KP A/B (the credit-DIRECTION test — does
  Kolen-Pollack learned feedback fix the direction?) is therefore UNRESOLVED.

## Status (per THE LAW — the negative names the next mechanism)
- **gap#4 keystone accuracy = NOT achieved at ep300/hidden128** — the BDSP fixed-feedback credit doesn't produce
  accuracy-useful hidden-layer learning (== lesion). The mechanism/wiring/moat are all correct; the credit DIRECTION is the wall.
- **NEXT (re-run, GPU-ISOLATED this time):** the KP learned-feedback A/B at the ACCURACY-tuned config (the runner's verdict
  says accuracy needs "width + epochs + drive tuning" the smoke omits) — does Kolen-Pollack learned feedback lift held-out
  over fixed? Run ONE arm at a time (or 2 max) with `nvidia-smi` memory headroom, NOT concurrent with n_ca3=2000 SWR runs.
  If KP still == lesion → the credit-direction needs a genuinely different mechanism (per 2026-07-14; the deep-research gate
  fires). If KP lifts → 6-seed + anti-cheats → the gap#4 accuracy milestone.
- **Context:** this session CLOSED gap#5 (i) SWR readout specificity (6/6 GO) + de-risked gap#5 (ii) emergent-DG selection
  (6-seed GO); the gap#4 keystone accuracy remains the open credit-direction problem, now with the KP A/B still to run cleanly.
