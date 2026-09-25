---
type: finding
status: partial
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: load-dependent sleep renormalization (webapp/sleep_replay_capture.py r3, BRAIN_SLEEP_LOAD_RENORM, default OFF) -- the night's downscaling amplitude is the measured fraction dW / W of the store's strength that the preceding wake added, applied with the r2 reactivation protection, so a weak, never re-mentioned fact fades as a function of how much else is learned after it
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
seeds: [42]
verdict: seed-42 de-risk smoke, not a gate row. grade_seed_fi reads all seven FI gates and every UNDEFINED rule as holding at this seed. The six pre-registered gate rows are not run yet (1 of 6 seeds, smoke only).
artifacts:
  - research/findings/raw/_sleep_forgetting_interference_smoke/seed42.json
  - research/findings/raw/_sleep_forgetting_interference_smoke/seed42/*.json
  - research/findings/raw/_sleep_forgetting_interference/design_fake_substrate.json
---

# Sleep renormalization set by the day's learning: seed-42 smoke of the fi family (1 of 6 seeds)

seed-waiver: this is the seed-42 de-risk smoke that Amendment 6 of the prereg schedules after its own commit. It is
not one of the six gate rows, and it makes no 6-seed claim. The pool lines for the six rows are at the end.

## What ran

`--family fi --seed 42 --ltm off --workers 5` at revision `2def39c76` (Amendment 6 committed, code `cbeccb54c`), under
`bash tools/memcap.sh 8` after `bash tools/mem_ok.sh 8`, numpy CPU with single-threaded BLAS. The worktree had the five
corpus files linked in, and no arm log shows the cross-edge build failing. Nine arms ran in 69 minutes of wall time,
26 to 36 minutes each. Each arm process held about 0.7 GB. Graded by the registered `grade_seed_fi`:
`research/findings/raw/_sleep_forgetting_interference_smoke/seed42.json`, zero arm errors, and every told sentence
stored as one new block (6 of 6 on the one-a-day arm, 18 of 18 on each three-a-day arm, 20 of 20 on the re-mention arm).

## The gates, as registered

| gate | result at seed 42 |
|---|---|
| G0 null rebuild (`fih_lr_a` = `fih_lr_b`: daily outcomes, recalled triples, ledger, blocks, sleep record) | holds |
| P1 immediate recall (`neu_imm_fi`) | holds |
| I1 seven epochs, records as armed, the registered dose delivered and counted by the load read | holds (gated arms and `fih_shy`) |
| I2 load lesion held (applied delta 0, every scale 1, the read above 0 on nights 2-7) | holds |
| I3 the dose arms identical through the first morning | holds |
| gamma identical across arms | holds |
| FI1 nothing else learned -> kept all seven mornings (`fiv_lr`) | holds |
| FI2 three facts a day -> lost by the seventh morning (`fih_lr_a`) | holds |
| FI3 ratio at the seventh morning: vacuum > one a day > three a day | holds |
| FI4 load edge cut, same dose -> kept all seven mornings (`fih_lr_lesion`) | holds |
| FI5 salient telling, same dose -> kept (`fis_lr`) | holds |
| FI6 re-mentioned after nights 1 and 2, same dose -> kept (`fir_lr`) | holds |
| FI7 no confab on any morning of any arm | holds |

## Trajectories (the fact block's increment-to-baseline ratio each morning; the read is magnitude-invariant)

| arm | mornings recalled | ratio, morning 1 | morning 3 | morning 7 | delta night 1 / night 2 |
|---|---|---|---|---|---|
| `fiv_lr`, 0 a day | 1-7 | 1.305007 | 1.305007 | 1.305007 | 0.166166 / 0.0 |
| `fil_lr`, 1 a day | 1-7 | 1.305007 | 1.015466 | 0.699117 | 0.166166 / 0.191344 |
| `fih_lr_a`, 3 a day | 1-3 (lost from 4) | 1.305007 | 0.750572 | 0.416856 | 0.166166 / 0.413120 |
| `fih_lr_lesion`, 3 a day, edge cut | 1-7 | 1.461247 | 1.461247 | 1.461247 | 0.0 / 0.0 applied |
| `fih_shy`, 3 a day, r2 constant | 1-6 (lost on 7) | 1.292000 | 0.997789 | 0.53719 | constant 0.18 |
| `fis_lr`, salient, 3 a day | 1-7 | 2.521142 | 1.628356 | 1.068687 | 0.264830 / 0.417652 |
| `fir_lr`, re-mentioned, 3 a day | 1-7 | 1.305007 | 0.703708 | 0.406652 | 0.166166 / 0.444924 |

- **Nothing else learned, nothing fades.** On `fiv_lr` the telling night read dW 1.048037 over W 6.307156, an amplitude
  of 0.166166. Every later night read dW 0 and applied no depression (all scales 1.0). The fact kept the same ratio
  for six nights.
- **Later learning erases it, in dose order.** At three facts a day the night-2 amplitude was 0.413120, the old
  fact's reactivation read fell from 0.356535 on night 1 to 0.000790 on night 7, and the fact was not recalled from
  the fourth morning. It was recalled at ratio 0.750572 and lost at 0.623906. At one fact a day it was still
  recalled at 0.699117 on the seventh morning.
- **The edge carries it.** With the load edge cut the same 18 facts were told and stored and the load was read
  (night 2 read 0.410659), but nothing was applied. The fact kept ratio 1.461247 and was recalled every morning. So
  the told facts do not erase it by crowding the store or by any other route in the arm.
- **r2's constant ignores the dose.** `fih_shy` followed the r2 horizon arm's trajectory under three facts a day: recalled
  at 0.638139 on morning 6 and lost at 0.53719 on morning 7, as seed 42's `d10w_shy` did with no later learning at all
  (`research/findings/raw/_sleep_replay_capture_r2_horizon_smoke/seed42/d10w_shy.json`).
- **Salience and re-mention protect.** The salient fact was still at 1.068687 on the seventh morning. On `fir_lr` the
  original block fell to 0.406652, as on `fih_lr_a`, but the recall matched the day-3 re-mention block, which held at
  ratio 0.786238. <!--derived-->

## Against the predictions stated in Amendment 6

- FI1-FI7 hold, as predicted.
- The telling night's amplitude was 0.166166. The fake had read 0.188877693, and Amendment 6 predicted a value close to
  0.19. It sits close to de Vivo's 0.18, and nothing was set to reach either.
- **Missed: the heavy dose erased the fact earlier than predicted.** The fake predicted the first loss on night 5 or
  6. The brain lost it on morning 4. Two measured differences account for it. First, the told facts' learned
  strength was larger on the brain (dW 4.647059 for three facts on day 2), so the night-2 amplitude was 0.413120
  against the fake's 0.339328. Second, the brain's recall boundary on this arm (lost at 0.623906) sits higher than the
  midpoint the fake used.
- `fil_lr` was kept through seven mornings, and `fih_shy` was lost on night 7, both as predicted.

## What this does and does not show

- On one seed, the fade of a weak, unrehearsed fact follows how much the brain learns after it: none, one a day, three
  a day. It goes through the renormalization edge, and it spares the salient and the re-mentioned telling. Every weak
  arm still recalled the fact on the third morning. That is the human three-day anchor of Amendment 6 (Rivera-Lares
  et al. 2022). The three-a-day arm lost it before a week. How a model dose maps onto a human day is not claimed.
- One seed. The six gate rows decide the family's verdict.
- The told facts are dissimilar to the target, so similarity-dependent interference is not tested. The store grows by
  one block per fact, so W grows with knowledge, which Amendment 6 declares.
- The read-back R of the old fact on the three-a-day arm falls to about zero by night 7. The fact then pays the full
  amplitude each night: the competitive down-selection in the mechanism. Whether that is right is a question about
  the R_i protection, which is inherited from r2, not tested here.

## The six gate rows (pool lines, NOT queued)

Provision the pinned revision first, from `/home/dant123/Projects/sim`:

    POOL_PROVISION_ALLOW_STALE=1 bash tools/pool_provision.sh --revision 2def39c76cb4d81ca1b907399d3e9482f62bfe42 --isolated pool1 pool2 pool41 pool42

Then one line per seed, for N in 42 43 44 100 101 102:

    bash tools/pool_queue.sh add 'mem_gb=3 && cd ~/derisk-pool/revisions/2def39c76cb4d81ca1b907399d3e9482f62bfe42 && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --family fi --seed N --ltm off --workers 3 --out research/findings/raw/_sleep_forgetting_interference' --checked 'prereg research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md (Amendment 6, branch research/sleep-forgetting-interference @ 2def39c76); load-dependent renormalization family fi 6-seed; seed-42 smoke holds every gate; mem_gb=3'

`2def39c76cb4d81ca1b907399d3e9482f62bfe42` is the Amendment-6 commit. It contains the amendment and the code it governs (`cbeccb54c`), and the
smoke ran at it. Nothing after it changes code. With `--workers 3` a seed takes about two hours on a pool node, and three arms at about 0.7 GB fit in 3
GB. Combine: `--family fi --aggregate research/findings/raw/_sleep_forgetting_interference`.
