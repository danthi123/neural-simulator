# Shortcut #3 fold — `OneBrainComposer(integrated_loop=True)` de-risk: the spiking K-way sequencer routing is ANSWER-IDENTICAL to the host `_scan` (2026-06-21)

**Type:** WIRING de-risk (compose already-de-risked pieces — the deep-research gate does NOT fire; the K=32 capability
+ moat were de-risked GO at the production threshold, `2026-06-21-shortcut3-K32-capability-surpass.md`). This validates
the FOLD at the COMPOSER-API level: the production `OneBrainComposer`'s `query_patient` / `ask_yes_no` /
`update_on_mismatch` answers are IDENTICAL with `integrated_loop` ON (the spiking K-way sequencer routes the
(agent, action) cue-match) vs OFF (the host first-match oracle), the no-confab moat holds 0-false-accept, at production
scale, multi-seed — with the anti-cheats (sequencer-LESION fails safe, permuted-rule inverts, the NO-DIVNORM raw control
fails). Plan: `docs/plans/2026-06-21-shortcut3-fold-host-scan-to-spiking-sequencer-plan.md` (commit 94a5e237).

**Op-point (the validated production op-point, NEVER re-tuned to mask a failure):** `match_thresh=0.06`, `gain=0.11`,
`sigma=1.0`, `input_gain=1.0`, `retreat=divnorm`.

**NO `sim/` edit** (reuse-by-import: the S0 K-way sequencer fabric + the S5 divnorm score bridge + the S2 decoded-line
drive, all already shipped). Runner: `research/runners/_phaseB_onebrain_integrated_loop_fold_derisk.py`.

---

## What the runner asserts (per seed, per K)

Two composers on the SAME facts/codes: `c_host = OneBrainComposer(integrated_loop=False)` (the oracle) and `c_seq =
OneBrainComposer(integrated_loop=True, sequencer_match_thresh=0.06)`. The full who/what + moat battery (the K=32 fact
set: 32 distinct facts, unique (agent, action) cues, the 8 actions each shared by 4 facts — the maximal shared-action
stress), through the production composer API:

- **ANSWER-IDENTITY:** `c_seq.query_patient(a,x) == c_host.query_patient(a,x)` AND `c_seq.ask_yes_no(...) ==
  c_host.ask_yes_no(...)` for every present cue (each answers ITS block — the scan reaches the LAST block) AND every
  abstention (an unstored cue → `query_patient is None`; an absent-agent / absent-action / cross cue → `is None`; a
  never-stored SVO → `ask_yes_no == "unknown"`).
- **MOAT (HARD, never traded):** `fa_total == 0` — no absent/cross cue selects a block on the `integrated_loop` path.
- **Reconsolidation abstain:** a never-stored cue → `update_on_mismatch(...)["action"] == "abstain"` via the routed
  `_find_cued_block` (the spiking decision) == host.
- **Anti-cheats** (on the composer's OWN built sequencer fabric, the same `sb`/`meta`/`drives` `_seq_block` uses):
  sequencer-LESION fails safe (sever the result→op drive → abstain), permuted-rule INVERTS (the decision follows the
  cyclic-shift rule), the NO-DIVNORM raw control FAILS (the divnorm is load-bearing).

---

## Results — numpy-CPU, V=72, D=128, seeds 42/43/44, K ∈ {2,4,8} — **OVERALL GO**

Every gate green at every K, all 3 seeds (`OVERALL: GO  (K in [2, 4, 8], 3 seeds, match_thresh=0.06, V=72)`):

| K | ==host | moat (FA_total) | recon | lesion-safe | permuted | raw-fails | verdict |
|---|---|---|---|---|---|---|---|
| 2 | 3/3 | 3/3 (0) | 3/3 | 3/3 | 3/3 | 3/3 | **GO** |
| 4 | 3/3 | 3/3 (0) | 3/3 | 3/3 | 3/3 | 3/3 | **GO** |
| 8 | 3/3 | 3/3 (0) | 3/3 | 3/3 | 3/3 | 3/3 | **GO** |

Per-seed (every row): `==host  moat-OK  recon-OK  lesion-SAFE  perm-inverts  raw-fails`. The `integrated_loop=True`
composer is answer-identical to the host-`_scan` oracle on the full who/what + abstention matrix, the moat holds
0-false-accept at the validated `match_thresh=0.06`, and all three anti-cheats behaved (LESION→abstain,
permuted→cyclic-shift, NO-DIVNORM raw→fails, so the divnorm is load-bearing).

## Results — 320-scale (V=320, K=32, GPU, `SIM_BACKEND=cupy`), seeds 42/43/44

_PENDING (the production-tier confirmation — task 6)._

---

## Verdict

_PENDING (filled when the multi-seed K∈{2,4,8} + the K=32 320-scale runs land)._

The build-1 honest scope: the (agent, action) hot-path sites (`query_patient` / `ask_yes_no` / `_find_cued_block` →
reconsolidation/reason_chain) route through spikes; `query_agent` (action,patient) + agent-only `render_fact`/`describe`
stay on the host read (named bounded follow-ons — a swapped-cue + a 1-role cascade — still abstaining via the oracle).
