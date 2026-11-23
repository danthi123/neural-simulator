# Capability: Lifelong learning
# Record: brain/research/2026-11-21-sleep-language-keep-prereg.md (K-P, A1/A2); brain/experiments/results/2026-11-22-kp-dev-read-1.5b.md (costs); brain/research/2026-11-23-stage2-test2-prereg.md (G7, G8, §0)
# Expected-fail: yes, named. (1) Assistant-turn positions carried much of K-P's held-out repair (B s319 held-out KL A 3.75 vs U 1.96), so masking them may give back the language win. (2) The self-name and refusal costs may sit in the weights (replay strength, adapter shape), not in the keep set, and then masking leaves G7/G8 where K-P has them. (3) One night from a B home may not show the K-P cost at all (it grew over a life), so the gate's G7/G8 read can come out uninformative.

# K-P-U: a sleep keep term held only on the prior's own User turns, against K-P's self-name and refusal costs (prereg, 2026-11-23, written before any K-P-U night)

## 0. Record brief and hypothesis
- K-P (`Brain(keep_bank="prior")`) fixed the language guard at DEV: held-out KL at night 15 was 0.118x B on 3/3 seeds.
  It brought two costs. (a) Self-name: on the 30 "nova" rows per seed, K-P was wrong on 26/26/17 vs B's 12/3/7, and it
  said "My name is Dan/Alex/Philippa/...". (b) Stock refusals: whole probe nights answered "I'm sorry, I don't know what
  you mean", 59 of 1158 rows vs B's 0.
- Bank audit: 290-311 of the 512 texts have a non-User turn, with labels Assistant 184, Bot 118, AI 69 and others (s319).
  About 1-2% of texts state a self-name explicitly, "nova" appears 0/512 times, and refusal-like texts are 6-14/512.
  A content filter (K-P-f, test2 §0) would remove about 2% of the bank. It would also be a hand-written English list
  (host templating). It is dropped.
- **H-role.** The keep KL over whole chat texts pins the cortex's Assistant-role distribution to the prior's generic
  persona. Identity is not in the weights: it is retrieved from permanent hippocampal episodes. The pinned persona
  outvotes the retrieved "My name is Nova", and stock refusals win whole nights. The mechanism is persona-wide, not a
  content leak. The record has no test of it.
- Prior evidence: Assistant positions drift more than User positions (langdrift n15: 2.91 vs 1.81). Literature (from memory, not verified): LwF (Li & Hoiem 2016); pseudo-rehearsal (Robins 1995); generative replay
  (Shin et al. 2017); role- or position-masked distillation, a standard lever when the new behaviour is local to one
  role; CLS interleaving (McClelland et al. 1995).
- Prerequisites: nightly sleep, the frozen prior and the K-P bank sampler are MET. Self and identity is PARTIAL: identity
  is host-seeded, so a pass here means only "retrieved identity survives sleep", not learned self-modelling. Masking
  needs no new prerequisite: `sleep.py` already slices the KL by position for guard pairs (`g_from`, `[k:]`).

## 1. Method
**Primary, K-P-U:** `Brain(keep_bank="prior_u")`.
- The bank is the K-P bank, byte for byte: same sampler, same seed (`sleep_seed + KEEP_BANK_SEED_OFFSET`), same five
  numbers, and the same nightly subset draw.
- The only change: the keep KL counts only positions that predict a token inside a frame turn.
- **Turn rule** (structural, no content words). A line that opens with a speaker label (the start of the text, or right
  after "\n", then at most 3 words and ":") opens a turn. A line with no label continues the turn that is open.
- A turn is a frame turn when its label equals `KEEP_BANK_FRAME` ("User").
- Masked: every other labelled turn (Assistant, Bot, AI, ...), its label tokens and its continuation. Kept: the first
  turn is always "User:", so every text keeps its opening turn, and prose running on inside a User turn.

**Secondary, K-P-0 (Step 0 only, unless §4 promotes it):** `keep_bank="prior_0"`.
- The same sampler and numbers, with no frame: sampled from the end-of-text start state, so it is the cortex's
  unconditioned prior.
- Whole-text KL, as K-P. It tests whether chat-framed sampling is what pins the persona.
- Risk: it drifts to prose and leaves the chat format uncovered.

**Code (default OFF; None and "prior" stay byte-identical):**
1. `brain/brain.py`: accept `keep_bank in (None, "prior", "prior_u", "prior_0")`, stored in meta, a resume that
   switches it refused (as "prior"). "prior_u" reuses `_keep_bank()` unchanged and passes `(text, keep_positions)` pairs
   as `generic`; "prior_0" samples with frame "" from the end-of-text state (bank file records `frame: ""`).
2. `brain/keep_mask.py` (new, with the three headers): `frame_turn_positions(ids, decode, frame)` implements the turn
   rule, using the same character-offset walk as `drift_read.roles()`.
3. `brain/sleep.py`: a `generic` item may be a str (unchanged path) or a `(text, positions)` pair; for a pair, reference
   and student log-softmax are indexed by `positions` and `batchmean` runs over kept positions. Replay RNG unchanged.
4. `brain/eval/life.py`: `--keep-bank` choices gain `prior_u` and `prior_0`. The night record adds `kept_pos_share`.
5. `brain/SCAFFOLDS.md`: add one row for the turn rule (cap: label of at most 3 words, frame label only) and one row
   for the K-P-0 empty frame.
6. Tests (0.1B, CPU):
   - `keep_bank=None` and `"prior"`: 3-night homes byte-identical to head (adapters, meta, `keep_bank.json`);
   - the "prior_u" bank file equals the "prior" bank for the same seed;
   - the mask on hand-made texts: a User/Assistant text, a User/Bot text, a single-newline turn, unlabelled prose;
   - a pair whose positions cover every position gives the same keep loss as the str path.

## 2. Step 0: one-night gate (before any life)
- **Homes:** the B night-15 sources of the K-P gate replications: `_scratch/kpgate_w/src_s319`, `src_s320` (W), and the
  s321 source used for `_scratch/kpgate_p40` (copied to W first).
- **Arms:** B, K-P, K-P-U and K-P-0, one night each from the same snapshot and seed, through `one_night()` and
  `gate_night_g.py` (arm names extended).
- **Device:** all four arms on one device.
  - Preferred: the W GPU after the reboot, at `Cortex(device="cuda", cuda_precision="fp32")` (TF32 off). B and K-P are
    then re-run on it.
  - Fallback: W CPU, reusing the existing CPU B and K-P after-homes for s319 and s320, and re-running s321 on W CPU.
  - K-P-U copies `keep_bank.json` from the matching K-P after-home, which is the same bank by construction, so it skips
    the 2.4 h sampling. The copy is hash-checked.
- **Reads on every after-home:**
  - (a) `drift_read.py` on the snapshot with the after-adapters as extras: held-out KL and NLL to pretrained,
    `role_kl` U and A on HELDOUT, prose KL, and replayed-episode NLL.
  - (b) Self-name. `life.probe()` on the after-home, arms intact and hippocampus-off. The questions are
    `battery(seed, 16, ledger)`, with the ledger from the life's own schedule and the brain kwargs from the B life JSON.
    There are 6 replicates: 3 as asked, and 3 with the nova row's text replaced by its existing `reworded` form ("What
    should I call you?"). Each replicate gets its own `probe(seed=S + 1000*r)` reseed. That gives 12 nova rows per arm
    per seed, 36 pooled.
    - G7 wrong = `ok` False, the existing strict scorer for item "nova".
    - The friend-name rate uses the test2 §5 regex (report only).
  - (c) Refusal. On the same batteries, the G8a rate on answerable rows (H not dan/nova, rotating, neardup, `told` not
    False). It uses the test2 §5 G8 pattern verbatim: the scorer, not a template.
- **Gate bars (pooled over 3 seeds; the direction per seed is reported):**
  1. Language. Let ΔKL = held-out KL to pretrained after the night minus the snapshot's. The variant's ΔKL must be at
     most K-P's ΔKL + 0.30 nats. The margin is about 20% of K-P's s316 repair of 1.54. The variant's ΔKL must also be
     below B's on 3/3 seeds.
  2. Self-name. The variant's G7 wrong rate must be at most B's + 0.10, and at most K-P's.
  3. The night writes. The variant's replayed-episode NLL must be at most B's + 0.2 (one-sided, as A1).
  4. Refusal. The variant's G8a rate must be at most B's + 0.03, and at most K-P's.
  5. Engagement (else UNDEFINED). The bank passes the K-P §3.1 bars. For K-P-U, `kept_pos_share` is between 0.3 and
     0.95, and the mean keep KL is greater than 0.
- **Validity of 2 and 4.** If K-P itself is not worse than B (K-P G7 minus B G7 below 0.10, and K-P G8a at most B's +
  0.03), one night does not show the cost. Bars 2 and 4 are then UNINFORMATIVE. In that case the gate passes on 1, 3
  and 5 plus the mechanism read: K-P-U's HELDOUT A-position KL must be at least 1.25x K-P's, the persona unpinned. If
  K-P-U's A-position KL is within 1.25x of K-P's, the mask did not engage the mechanism. Report it UNDEFINED and do not
  launch.

## 3. Step 1: DEV screen (only if the gate passes)
- **Lives:** DEV seeds 319, 320 and 321, one life each, with the K-P DEV command (K-P prereg §2) and
  `--keep-bank prior_u` (or `prior_0` if §4 promotes it):
  `python -m brain.eval.life --model 1.5b --seed S --days 15 --weekend --split dev --full-probe-nights 1,5,10,15
  --frozen-probe --span --transfer --pruned-a --cue-ladder --interview --lang-guard --keep-bank prior_u --device cuda
  --homes ~/brain-lane/workLIFEkpu_sS --out ~/brain-lane/outLIFE/lifeKPU_sS`
- **Device:** AWS g5 (A10G), CUDA fp16 (default `--cuda-precision`), the device of the DEV K-P lives and the planned
  TEST. The lives are read against B-fp16 `lifeBg_s<S>` (same device) and against K-P `lifeKP_s<S>`. If `lifeBg` is not
  complete, the comparison is CPU `lifeS1F_s<S>`, labelled device-confounded. About 6.5 h per life, about $20 for the
  three, within the $50/day cap.
- **Bars (pooled over 3 seeds, each on 3/3 seeds unless stated):**
  - K-P §3.1 engagement, plus `kept_pos_share` on every night.
  - K-P §3.2 primary: n15 KL at most 0.5x B, below B 3/3; the NLL midpoint 3/3.
  - K-P §3.3 guards: H hippocampus-off, d12 recall (nights 5-15 and nights 1-4, -1.0 of 6), collapse, leak on all rows
    and on non-refusal rows, and no new defect class in the transcript read.
  - G7 (test2 §5): wrong rate at most B + 0.25 per seed.
  - G8 (test2 §5): (a) at most B + 0.03 on each probe arm; (b) no seed-arm-night at or above 0.6 unless B is there.
  - Report only: n15 KL vs K-P, G10, G11, the friend-name rate, and `role_kl` U and A at n15.
- The `life_costs.py` scorers from the test2 draft are used once written and hash-frozen. Until then, the same regexes
  are copied verbatim into the reader and must reproduce the test2 §5 DEV validation numbers.

## 4. Decisions
- **Gate:**
  - K-P-U passes: launch the DEV screen with K-P-U.
  - K-P-U fails and K-P-0 passes: launch with K-P-0. K-P-0 is then the lever.
  - Both fail: no lives; report which bar failed.
  - Language passes, self-name fails (bar 2 informative): H-role weakened at one night; recorded, no lives.
- **DEV:**
  - **KEEP:** every §3 bar holds. The variant replaces K-P-f in test2 §0 and becomes the TEST arm; K-P is not run.
  - **H-role FALSIFIED:** primary and engagement hold, A-position KL is unpinned (at least 1.25x K-P's at n15), and G7
    and G8 are still at K-P's level (within 0.10). The keep set is not the cause. The cost is in the weights or in
    retrieval, and is judged against the Self-and-identity prerequisite (PARTIAL) before a lever 2.
  - **NARROWED:** G7 or G8 passes but not both, or 2 of 3 seeds pass. Report it as a lead. K-P runs at TEST with its
    Expected-fail, and the variant does not enter.
  - **KILLED:** the primary fails, meaning masking gives the language win back. K-P runs at TEST unchanged.
- **Test2 arm list:** exactly one of K-P / K-P-U / K-P-0, fixed at the freeze by this read. If there is no KEEP by the
  timebox, K-P runs unchanged and its G7/G8 Expected-fail stands.
- **Lever count:** lever 1 on K-P's identity/refusal cost; K-P-0, if promoted, is lever 1b (same hypothesis and read).
  Lever 2 requires `before-build` (2nd-fix check); after it fails, judge against scale and prerequisites before a third.
- **Timebox: 2 days.** Gate on 11-23 and 11-24, DEV lives launched by 11-24 and read by 11-25, inside the test2 §0
  window (prereg 11-25, read 12-01). No second mask rule, label cap, bank size or keep weight is tried in this window.
- One life per arm per seed at DEV: no generalisation claim (six seeds come only at TEST).

## Amendment A1 (2026-11-23 06:15 UTC, before any gate result exists)
- **Device:** the W GPU and W CPU are held by the owner, so the gate runs on the pool CPUs, fp32: s319 on pool41, s320
  on pool42, s321 on pool40. Each seed re-runs ALL four arms (B, K-P, K-P-U, K-P-0) on its own node, so every
  within-seed comparison stays on one device; the W-CPU B / K-P reuse in "Fallback" is not used. Seeds are not
  compared across nodes.
- **Inputs:** each node's `~/brain-lane/kpgate/src_s<S>` is checked against the W source by the sha256 of
  `cortex_adapters.pt` before the first night (all three matched). Code: brain/ at 9fd474dde, `_scratch/kpid/gate.py`
  and `drift_read.py` (committed with this amendment, exactly as shipped to the nodes); runner `_scratch/kpid/pool/run_pool_gate.sh`, launcher `_scratch/kpid/pool/launch.sh`.
- Bars, reads and verdict rules are unchanged.
