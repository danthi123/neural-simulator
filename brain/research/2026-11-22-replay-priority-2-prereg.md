# Pre-registration: D2-PR2, tempered loss-prioritised replay (stock self-reply leak), 1.5B, DEV screen

Written 2026-11-22, after the D2-PR DEV read and before any D2-PR2 code or life.
Capability: lifelong learning (sleep that strengthens what is weakly learned without drilling the stock replies).

**Record brief.**
- D2-PR (`brain/research/2026-11-21-replay-priority-prereg.md`) was read on DEV s319-321
  (`brain/experiments/results/2026-11-22-d2pr-dev-read-1.5b.md`; 5 refuters).
  - PR-l (w_e = L_e, alpha 1) cut the stock self-reply leak by 43.7% pooled, and the cut is robust over nights 1-15.
    Its numeric guards passed.
  - But its engagement bar failed. On s321 night 5 the top-6 share was 0.596, against 0.439 for the strength-only
    weighting of the same night; one episode took 0.144.
  - PR-x (strength x loss) failed engagement on 6/6.
  - Lever count on this defect: 2 (PR-x, PR-l). Both are verdicts on alpha = 1.
- What is new: the prioritised-replay literature never uses alpha = 1. Schaul et al. 2016 (prioritized experience
  replay, proportional variant) sample with P(i) proportional to p_i^alpha, using alpha = 0.6 so that a few
  high-error items cannot take the whole replay. Mattar & Daw 2018 weight replay by gain, which is bounded per item.
  alpha is taken from that paper as written, not tuned here. No other value is tried in this screen.
- Prerequisites: all met (the same as D2-PR). Scale: 1.5B, the reference size. Host decision: none (one exponent from
  the literature, applied to the cortex's own per-episode error).
- Expected: the permanent share stays well below the strength-only weighting. Permanent episodes are the most
  memorised, with L_e 0.15-0.41 against 1.0-1.8 for once-told facts, and alpha 0.6 keeps that order. The top-6 and the
  largest single share move toward the strength-only values or below them. The leak cut is smaller than PR-l's but
  still present. If the leak cut needs the concentration, it shrinks toward B; that outcome is a NARROWED read.

## 1. Method

Arm **PR-a**: `Brain(replay_priority="loss_a")`. Everything is as in PR-l except w_e = L_e ** 0.6
(`REPLAY_ALPHA = 0.6`, a named constant citing Schaul et al. 2016). The default stays None, byte-identical.
The night record keeps the same `replay_priority` block, so engagement is read exactly as in D2-PR.

Before any life, run an offline engagement check on the saved night-15 PR-l trunk homes.
- Report the permanent share, the top-6 share and the max share under strength-only, alpha 1 and alpha 0.6.
- This is not evidence and does not change the method.

## 2. Screen (DEV only)

- Seeds: DEV 319, 320 and 321, against `lifeS1F_s<S>` (B) and `lifePRl_s<S>` (PR-l, as a reference).
- Command: the D2-PR command with `--replay-priority loss_a`, 15 days, the same flags as B.
- Compute: CPU fp32 for every life, on the same pool node as each seed's B if possible.
  - B s319: pool41 node-dl4g243. B s320: pool42 node-dl5d243. B s321: pool40 node-mxl1462drl.
  - Record the host (JSON `host`).
  - No GPU or fp16 life is compared to a CPU B. This closes the PR-x confound.

## 3. Read-outs and bars

The bars are those of D2-PR §3, unchanged, with two clarifications fixed now.
- **3.1 Engagement.** On nights 5 and 10, the permanent share and the top-6 share are both below the within-night
  strength-only reference (`strength_permanent_share`, `strength_top6_share`; B's JSON keeps no per-episode weights,
  and its class shares were shown equal to the PR records' on all 12 seed-nights). This must hold on 3/3 seeds,
  each night, or the arm is UNDEFINED.
- **3.2 Primary.** The stock self-reply leak (`brain/eval/life_leak.py`), nights 10 and 15, both arms pooled per seed.
  It must be below B on 3/3 seeds, and at least 30% lower pooled.
- **3.3 Guards (any fail = KILL).**
  - H hippocampus-off, nights 5-15: not lower than B by more than 0.5/6 per seed.
  - Direct hippocampus-off recall of day-1-2 facts, read over **nights 5-15** (the D2-PR read's window): not lower
    by more than 1/6 per seed. Nights 1-4 are reported beside it, not as a bar.
  - Collapse no worse than B: the seeds-ever count per probe arm is at most B's.
  - Language-guard KL at night 15: at most 1.25x B per seed.
  - The whole-transcript §7 blind read (`brain/eval/life_read.py`, 2 sonnet readers, week 3) finds no new defect class
    (identity loss, name overwrite, marker leak, stuck tail absent in B).
- **3.4 Report only.**
  - The leak over all nights 1-15, paired with a sign-flip test.
  - First-person echo (`brain/eval/life_echo.py`).
  - Reader repetition counts.
  - Adapter norm.
  - Everything listed in D2-PR §3.4.

## 4. Decision

- **Passes 3.1, 3.2 and 3.3 on DEV:** a registered six-seed test on a FRESH TEST split (design §9), then the default
  only on that result.
- **3.1 and 3.3 hold, 3.2 fails:** NARROWED. The leak cut needed the concentration, and the leak lever moves elsewhere.
- **3.3 fails:** KILLED.
- **3.1 fails:** UNDEFINED, and the replay-priority lever stops there (lever 3). The next step is a research round,
  not a fourth weighting.

Timebox: 2 days from launch.
