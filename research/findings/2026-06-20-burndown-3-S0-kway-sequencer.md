# Burndown #3 Stage S0 — the on-substrate sequencer generalized K=2 → K-way (the K-way priority WTA + K-way rule): 6-seed GO (2026-06-20)

**Type:** implementation (Stage S0 of the approved #3 plan). CPU/numpy. Strict TDD, PATHSPEC. NO `sim/` edit. The
no-confab moat held 0 false-accepts throughout (the HARD gate).

**Plan:** `research/findings/2026-06-20-burndown-3-production-sequencer-scoping.md` (Stage S0 = "generalize the
sequencer builder K=2 → K (the K-way priority WTA + the K-way production rule), CPU"). #3 retires the production host
`_scan` (the Python `for/if/return` cue-match + answer/abstain ROUTING under `query_patient`/`query_agent`/`ask_yes_no`/
`render_fact`/`query_chain`) by scaling the proven Phase-B spiking sequencer (gated-disinhibition match cascade + BG
production rule) from K=2 to production K. S0 is the first stage: make the sequencer BUILDER a clean parameter K.

---

## TL;DR (the verdict)

**S0 GO, 6 seeds (42–47), CPU/numpy, D=64.** A K-way generalization of the K=2 sequencer (`build_sequencerK_bridge` +
`run_sequencerK`, the K-way priority WTA + the K-way first-match production rule) is `== the host _scan` control
decision at **K ∈ {2, 4, 8, 16}**, with the no-confab **moat 0 false-accepts at every K, every seed**:

| K | ==host (block selection) | moat 0-FA | lesion-fails-safe | permuted-inverts |
|---|---|---|---|---|
| 2 (parity guard) | 6/6 | 6/6 | 6/6 | 6/6 |
| 4 | 6/6 | 6/6 | 6/6 | 6/6 |
| 8 | 6/6 | 6/6 | 6/6 | 6/6 |
| 16 | 6/6 | 6/6 | 6/6 | 6/6 |
| per-block PRIORITY (degenerate multi-match) | — | — | — | **6/6** |

- **K=2 PARITY (regression guard):** the K-way sequencer reproduces the committed K=2 GO EXACTLY (== host `_scan`, the
  same answer/abstain, the same per-query block decisions). The committed K=2 path (`_phaseB_onebrain_sequencer_derisk.py`)
  is UNTOUCHED and still GO 3/3.
- **K-way generalization:** == host `_scan` for who/what (the right block answers; absent/cross cues abstain) at K=4,
  K=8, and K=16 (past the {4,8} target), 6 seeds.
- **Anti-cheats:** sequencer-LESION fails SAFE (sever the result→op conditioning → abstain, never a wrong block);
  permuted-rule INVERTS (the decision follows the cyclic-shift match→answer map, not a fixed scan order); per-block
  PRIORITY correct (a degenerate two-block-match cue answers the LOWER block == the host first-match).
- **NO `sim/` edit** (reuse-by-import, as the plan predicted). **The moat was never the negotiable axis.**

**Honest scope:** this is S0's CPU/numpy exact-algebra parity oracle at D=64 (the sequencer is small). The PRODUCTION
K=32 + D=128 margin sweep (the one place #3 could hit a substrate boundary — the 1-of-K=32 match-cascade
discrimination) is **Stage S2**, not S0. S0 demonstrates the K-generalization is correct and the mechanism scales
cleanly to K=16 on CPU.

---

## The K-generalization mechanism (what changed K=2 → K)

The K=2 builder (`_phaseB_onebrain_sequencer_derisk.py:build_sequencer_bridge`) hard-codes K=2: explicit `m0`/`m1`
match pools, `ans0`/`ans1` answer channels, `inh0`/`inh1` interneurons, a `blocks_scores[:2]` truncation, a 2-element
production-rule dict, and a block-0-priority chain (`ans0→inh0→{ans1,abstain}`, `ans1→inh1→abstain`). S0 replaces
every K=2 hard-coding with a loop over `range(K)` (new file `research/runners/_phaseB_onebrain_sequencerK_derisk.py`,
reusing the composer-side cleanup reader `block_cleanup_scores` + `scores_to_drive` VERBATIM by import — only the
sequencer CONTROL fabric is generalized; the FHRR cleanup is unchanged):

1. **K match cascades** (`build_sequencerK_bridge`): for `b in range(K)`, the per-block gated-disinhibition match —
   decoded word-line `d{b}{role}_w` → `mw{b}{role}_w` through a transmission gate `g{b}{role}_w` that the CUE word-line
   `cue{role}_w` opens (so `mw` fires iff the decoded word == the cue word); a role OR-pool `m{role}{b}`; a block AND
   `m{b} ← mX{b}` gated by `mA{b}` (action match passes iff agent ALSO matched). Identical to K=2, replicated K times.
   The gated-disinhibition primitive (`couple_gate_to_pool`, registered by `wire_sequencerK_couplings`) is the proven
   Phase-B match — no new selection mechanism.
2. **K-way first-match priority WTA + abstain**: each `m{b}` drives its answer channel `ans{b}`; first-match priority
   (== the host `_scan`'s "return the FIRST matching block") = block i inhibits every block j>i AND abstain, via an
   inhibitory interneuron `inh{b}` (`ans{b}→inh{b}→{ans{j>b}} ∪ {abstain}`). The `abstain` channel is the tonic default
   suppressed by ANY match (the K-way OR into abstain's inhibition — the canonical BG default-suppression). For the
   common case (facts distinct, a unique cue matches one block) a plain WTA suffices; the priority chain only
   disambiguates the rare degenerate multi-match, preserving the host first-match semantics exactly. This is the BG
   disinhibition motif (feed-forward, the `g11_bg_runner` selection template, catalog A.04), NOT a recurrent soft-WTA
   (the retreat the plan §3.1 named to avoid the Rutishauser α>1 instability).
3. **The K-way production rule** (`run_sequencerK`): the decision is read from the K spiking match pools `m{0..K-1}`
   (the K=2 precedent reads `m0`/`m1` + applies the rule in Python — the production rule OVER the spiking match result,
   the legitimate body read): the LOWEST-index block with `m{b} > match_thresh` answers (first-match priority); none →
   abstain (the moat). `permute` cyclically shifts the match→answer map (`m{b} → ans{(b+1)%K}`) — the anti-cheat.

---

## Two K-scale tunings (both per-query housekeeping, NOT new control mechanisms, NOT moat-relevant)

The K=2 operating point did NOT transfer verbatim to larger K — two issues surfaced at K=8 / K=16 and at more seeds.
Both were diagnosed to a clean root cause and fixed within the EXISTING reset/score→drive discipline (no new
mechanism, no `sim/` edit). Crucially, **NEITHER is a moat-relevant axis** — the moat held 0-FA at every intermediate
state, and an ABSENT cue matches NO block at any setting:

1. **Inter-query state leak (the drain).** At K=8 a PRIOR query that matched block b (firing `mA{b}`, opening `gblk{b}`
   via its EMA, driving the `inh{b}` priority chain) left delayed/recurrent activity that a single membrane clear did
   not fully drain; a borderline near-tie `m{b}` (~0.16, just over the 0.15 threshold) then leaked into the NEXT query
   and first-match priority picked the stale lower block (diagnosed: K=8 seed-42 blk5 read `m4=0.159` carried over from
   the blk4 query; in isolation the same cue read `m4=0.000` and answered correctly). **Fix:** `reset_sequencerK_state`
   runs a `drain_steps=20` blank-input settle (the prior recurrent/delayed state decays to rest) BEFORE the
   EMA/gate/membrane clear, so every query starts from the same resting state. This is exactly the per-query
   housekeeping the K=2 reset comment already stated as its job ("consecutive queries on the SAME persistent bridge
   don't leak") — at K=2 the small fabric drained within the reset; at K=8 the larger inhibitory fabric needs the
   explicit drain.
2. **Decoded-line over-inclusion at the score→drive (the `drive_frac`).** The host `_scan` matches on the cleanup
   ARGMAX (a single winner per role); the K=2 default `frac=0.5` (in the imported `scores_to_drive`) also lit the
   RUNNER-UP when its score was ≥50% of the peak — harmless at K=2, but at imperfect-fidelity K a near-tie runner-up
   (measured WORST ratio 0.81 across 48 role/block/seed cleanups at D=64) spuriously matched another cue and first-match
   priority picked the wrong block (diagnosed: seed-45 block-0 agent decoded `dog@1.00` AND `bird@0.54`, so cue
   `(bird,go)` spuriously matched block 0). **Fix:** `run_sequencerK(drive_frac=0.9)` (above the 0.81 worst runner-up)
   lights ONLY the argmax winner — the faithful spiking realization of the cleanup's own decision (which word this block
   decodes to), == the host argmax. `scores_to_drive(frac=...)` is a parameter on the imported helper, so this is a
   call-site argument, NO edit to the shared K=2 helper. NOT moat-relevant (an absent cue matches no block at any frac);
   it only removes the wrong-PRESENT-block leak.

---

## A methodology note (control-decision parity, not host-patient re-decode)

S0 generalizes the host `_scan`'s **cue-match + first-match ROUTING** — the CONTROL flow (which block matches →
answer/abstain). The parity comparison is therefore **block selection** (`host_scan_block` = the index of the first
block whose decoded agent+action match the cue, == `query_patient`'s internal `for i, got: if got['agent']==a and
got['action']==x: return i`), with the emitted patient LABEL read from kb on BOTH sides (`patient_of`). This deliberately
does NOT compare the host's separately-re-decoded patient, which at small D=64 occasionally mis-decodes a patient even
when the agent+action cue-match is clean (observed: seed-46 fact-1 patient `river`→`go` — the host cleanup's patient
readback fidelity, a DOWNSTREAM data op, NOT the control op the sequencer replaces; the SUBSTRATE selects the right
block there too). Conflating the two would mismeasure the sequencer's control decision against the host cleanup's
patient-readback fidelity. S2 runs the production D=128 where the cleanup fidelity is higher.

---

## Reproduce

```bash
# the S0 deliverable run (6 seeds, K in {2,4,8,16}, D=64, CPU)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_derisk \
    --seeds 42,43,44,45,46,47 --dim 64 --ks 2,4,8,16 \
    --out research/findings/raw/_phaseB_onebrain_sequencerK_6seed.json
# -> K=2/4/8/16 all: ==host 6/6  moat 6/6  lesion-fails-safe 6/6  permuted-inverts 6/6;  PRIORITY 6/6;  OVERALL GO

# the committed K=2 path is UNTOUCHED + still GO (regression guard)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencer_derisk --seeds 42,43,44 --dim 64
# -> ==host 3/3  moat 3/3  lesion-fails-safe 3/3  permuted-inverts 3/3  -> GO
```

---

## Files

- **NEW:** `research/runners/_phaseB_onebrain_sequencerK_derisk.py` — the K-way sequencer (the K-way priority WTA +
  K-way rule), self-validating against the host `_scan` control decision + the moat + the anti-cheats, K ∈ {2,4,8,16}.
  Reuses `OneBrainComposer` + `SimulationBridge` + `couple_gate_to_pool` + the public `set_transmission_gate` /
  `cp_external_input_current` + (by import) the K=2 `block_cleanup_scores` / `scores_to_drive`. NO `sim/` edit.
- **raw:** `research/findings/raw/_phaseB_onebrain_sequencerK_6seed.json` (+ `.log`) — the 6-seed result.

## What's next (NOT S0)

- **S1** — wire the validated on-bridge S5 divisive-norm into the loop (close the residual host read), CPU.
- **S2** — the K ∈ {2,8,16,32} margin sweep at the production D=128 (THE BOUNDARY TEST: the 1-of-K=32 match-cascade
  discrimination, the worst no-match leak vs threshold vs true-match as K grows), CPU → GPU. The S0 drain + `drive_frac`
  operating point carries forward as the starting config; the divnorm `gain` re-confirm at scale is folded into S2/S4.
- **S3/S4** — fold into `OneBrainComposer` (opt-in, default-off byte-identical) + the 320-scale production GO, GPU.
