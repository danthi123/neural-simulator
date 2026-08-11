---
type: finding
status: smoke
date: 2026-08-11
mechanism: sleep-replay-consolidation-self-generated-hippocampal-engram + bdsp_wmax substrate de-clamp
lane: breadth / teacher-loop / memory (continual learning)
runner: research/runners/_teacher_loop_sleep_replay_declamp_derisk.py
builds-on: research/runners/_teacher_loop_sleep_replay_consolidation_derisk.py
attacks-baseline: teacher-loop SELF-REPLAY consolidation cap (6-seed clamped-replay frac_recalled mean ~0.55, range 0.20-0.90; no-replay ~0.13; interleaved ceiling 0.8; findings 2026-08-08-teacher-loop-sleep-replay-consolidation-... and 2026-08-09-teacher-loop-sleep-replay-budget-sweep-...-NEGATIVE)
biological-pattern: hippocampal->cortical systems consolidation (McClelland/McNaughton/O'Reilly 1995; Wilson/McNaughton 1994). The lever is a SUBSTRATE-config clamp, not a new biological mechanism.
confirms: research/findings/2026-08-09-bound-trap-bdsp-wmax6-silences-teacher-loop-reservoir-declamp-recovers-75pct-of-N20-retention.md (the bound-trap was already established 6-seed at N=20 on the neurogenesis runner; this localizes it to the N=10 sleep-replay-consolidation runner)
corrects: research/findings/2026-08-09-teacher-loop-sleep-replay-budget-sweep-more-replay-does-not-reach-ceiling-NEGATIVE.md (its "store-fidelity is the lever" residual was measured with the clamp held FIXED and is confounded by it)
seed-waiver: 2-seed SMOKE (42 saturated + 43 the informative low seed); the 6-seed sweep is the deliverable (pool command below). Per-seed runner verdict is GO on BOTH seeds; the headline is NOT a 6-seed claim.
artifacts:
  - research/findings/raw/sleep_replay_declamp_s42.json
  - research/findings/raw/sleep_replay_declamp_s43.json
---

# The ~0.55 self-replay cap is the known bdsp_wmax=6 BOUND-TRAP, not engram fidelity: de-clamping lifts replay-based continual retention to 1.00 on both tested seeds — CORRECTING the budget-sweep's "fidelity" residual (2-seed SMOKE; 6-seed queued)

## The cap this attacks (named)

<!--derived-->
(this paragraph quotes the PRIOR sleep-replay findings' measured baselines, not this run's artifact)

Teaching N facts SEQUENTIALLY into one shared leaky-readout forgets to ~1/N. The brain's own hippocampal
self-replay (a lossy mean-vector engram, teacher/world absent) beats that but caps at a **6-seed clamped-replay
`frac_recalled` mean ~0.55** (range 0.20-0.90; no-replay ~0.13; interleaved ceiling 0.8). The budget-sweep
NEGATIVE showed MORE replay is flat across a 64x range and concluded the residual is **engram-store FIDELITY** --
but it held the SUBSTRATE CONFIG FIXED. It never varied the substrate weight clamp. That is the un-tried lever
here (candidate "replay + the bdsp_wmax de-clamp"), differentiated from the sibling fidelity/generator lane.

## Relation to the prior record (this CONFIRMS + reconciles; it does NOT discover the clamp)

<!--derived-->
(this section quotes PRIOR findings' measured numbers, not this run's artifact)

The `bdsp_wmax=6` bound-trap is ALREADY established, 6-seed: `2026-08-09-bound-trap-bdsp-wmax6-...` showed it
silences the teacher-loop DG-expansion reservoir at **N=20** (self-replay -> chance 0.05; de-clamp -> 0.742;
capacity -> 0.967), and the MASTER ROADMAP wall-ledger's "Forgetting" row already records it as the dominant term
with engram-fidelity listed as dominated. The prioritized-replay runner also de-clamped. What was NEVER done is
apply that de-clamp to the SPECIFIC runner and operating point where the ~0.55 self-replay CAP and the
"fidelity-is-the-lever" verdict were set: the N=10 `_teacher_loop_sleep_replay_consolidation_derisk.py` and its
budget-sweep. The budget-sweep (SAME day as the bound-trap finding) still concluded **"store-fidelity (WS-1) is
the lever, not quantity"** because it varied replay QUANTITY with the clamp held FIXED. This de-risk closes that
gap: it runs the clean clamp A/B on THAT runner and shows the ~0.55 cap there is the same bound-trap -- so the
budget-sweep's flat curve and "fidelity" residual are a clamp confound, reconciling the two 2026-08-09 lines.

## The mechanism (read the substrate before theorizing)

The `OnBridgeEpropNet` parent sets `bdsp_w_min/max = -6/+6` while `ff_w_init=2000` and `w_clip=4000`. Even with
BDSP as a byte-inert kernel (lr=0, e-prop is the sole learner), `fused_bdsp_update` ENDS in an UNCONDITIONAL
`cp.clip(w_new, w_min, w_max)` (kernels.py:485, documented at `_onbridge_eprop_port_derisk.py:136-156`) that RUNS
every forward pass on every FF synapse whose presyn fired -- so it silently CRUSHES the ~2000-scale FF weights
toward |w|<=6 as teaching proceeds. A direct read confirms it (seed 42, teach 5 facts, numpy): default clamp ->
FF |w|mean **229->82**, frac|w|<=6 **0.42->0.68**; de-clamped (`hp['bdsp_wmax']=1e9`, the port's own
single-variable CONFIG lever) -> |w|mean **229->229 preserved**. Immediate acquisition is 1.000 either way, so the
clamp does not block learning the NEW fact -- it destroys the SHARED features that carried the OLD facts. This is
a substrate-level catastrophic-forgetting term the "fidelity" framing could not see. It is the CLAUDE.md pattern
(gap#5: 97% of a weight change was the clamp; "the proxy usually owns the measurement"). The prioritized-replay
runner already de-clamped (`bdsp_wmax=1e9`); the sleep-replay / budget-sweep line did not.

## Result -- 2-seed SMOKE, N=10, chance 0.10, cap 0.55 (per-fact args identical to the baseline)

<!--derived-->

| seed | noreplay_clamped | replay_clamped | noreplay_declamped | replay_declamped | scramble_declamped | replay_declamped acq |
|---|---|---|---|---|---|---|
| 42 (saturated) | 0.10 | 0.90 | 0.60 | **1.00** | 0.20 | 0.945 |
| 43 (low/informative) | 0.10 | **0.20** | 0.70 | **1.00** | 0.00 | 0.982 |
| **mean** | 0.10 | **0.55** | 0.65 | **1.00** | 0.10 | — |

The 2-seed `replay_clamped` mean is **0.55 -- exactly the record's 6-seed cap** (seed 42 is the saturated seed,
seed 43 is a budget-sweep low seed). `replay_declamped` is **1.00 on both**, and de-clamped replay beats the named
0.55 cap AND each seed's own clamped-replay. Runner Verdict = **GO** on each seed (2-seed roll-up refuses the
6-seed GO by design, printing NO-GO until n>=6).

## Teeth (per-seed; the honest decomposition is the point)

<!--derived-->

- **(a) de-clamped replay beats the cap** -- 1.00 > 0.55, and > each seed's clamped-replay (0.90 s42, 0.20 s43).
- **(b) the CLAMP is load-bearing on forgetting** -- de-clamp ALONE (no replay) lifts retention 0.10->0.60 (s42)
  and 0.10->0.70 (s43). `attributable_to(clamp alone on forgetting, noreplay declamped vs clamped)` = **83-86%**.
- **(c) content still load-bearing under de-clamp** -- `scramble_declamped` (labels shuffled, identical compute)
  collapses to 0.20 / 0.00; replay-vs-scramble margin **+0.80 / +1.00**, `attributable_to` = **80-100%**. So the
  extra replay gain on the de-clamped substrate is the STORED ENGRAM CONTENT, not the extra gradient steps.
- **(d) immediate acquisition stays high** -- de-clamp lets weights grow larger, a small newest-fact tradeoff:
  immediate acq 1.000 (clamped) -> 0.945 / 0.982 (declamped), still >= 0.9 floor.
- **the seed-42 saturation caveat, made explicit** -- on seed 42 the de-clamp's MARGINAL effect on the replay arm
  is only +0.10 (clamped replay was already 0.90 = no headroom); `attributable_to(de-clamp on replay arm)` reads
  10% THERE. That is exactly why seed 43 (clamped-replay 0.20, +0.60 headroom) is the informative seed: there the
  de-clamp adds **+0.80** to the replay arm. This is the seed-dependence the budget-sweep warned about; the
  6-seed sweep resolves it.

## What this maps (and where the residual actually is)

The dominant term in the ~0.55->0.8 self-replay gap on these 2 seeds is the **substrate weight clamp**, not engram
fidelity. The budget-sweep's flat curve is consistent: replaying a lossy prototype 16x cannot help when the shared
FF weights carrying old facts are being crushed to |w|<=6 every forward pass regardless of what is replayed.
De-clamping restores the pathway's intended operating point (ff_w_init=2000 scale) and self-replay consolidation
reaches full retention at N=10. **This does not solve LIFETIME continual learning** -- it removes a confound that
was masquerading as a fidelity wall at the small-N operating point. The wall-ledger's genuinely-open levers are
unchanged and are where fidelity/compute actually bind: capacity is a small-N patch (SLIPS at N=100), and the
lifetime answer is the two NAMED, still-open builds -- a **non-forgetting generative replay** (van de Ven; the
sibling lane) and **prioritized/sparse replay** for bounded per-night compute (Mattar-Daw/Tse). The correct read
of this de-risk: it retires "the sleep-replay ~0.55 cap = a fidelity wall" and re-points the small-N story at the
already-known bound-trap, so the fidelity/compute work is spent on the regime (N>=50) where it is actually the
binding term.

## Scope / honest boundary

- **2-seed SMOKE, not a 6-seed GO.** Seed 42 is saturated for the replay-arm A/B (uninformative there); seed 43
  is the decisive low seed. The 6-seed sweep (42-47) is the deliverable -- pool command below.
- **The de-clamp is CONFIG, not a host shortcut.** `bdsp_w_max` is a substrate scalar (the synaptic weight
  ceiling in bridge units); the +-6 default is a units-scale artifact inconsistent with this FF pathway's
  ff_w_init=2000, NOT a biological bound on it. Widening it is the port's documented `_bw` lever and the exact
  move the prioritized-replay runner (2026-08-09) already made. NO `sim/` edit; reuse-by-import of the world /
  teach / held-out-acc / engram-store / replay machinery. Brain-based self-generation is inherited unchanged
  (`_self_replay_consolidate` / `Hippocampus.generate_replay` take NO env -- teacher/world absent during sleep).
- **N=10, OnBridgeEpropNet transport-free e-prop (48-neuron net), numpy** (cupy is launch-bound and slower at
  this size for this line). The mechanism is backend-independent; the 6-seed command keeps numpy for parity.
- The de-clamp raises the newest-fact immediate-acq tradeoff slightly (still >= 0.9); worth watching at larger N.

## Reproduce

Single-seed SMOKE (as run for 42 and 43):
```
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._teacher_loop_sleep_replay_declamp_derisk --seed 42 \
    --n-max 10 --milestones 1 5 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
    --out research/findings/raw/sleep_replay_declamp_s42.json
```

6-SEED (the deliverable; one seed per process; GO = every seed's replay_declamped > 0.55 AND > its clamped-replay
AND clamp-load-bearing), then aggregate:
```
for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._teacher_loop_sleep_replay_declamp_derisk --seed $s \
  --n-max 10 --milestones 1 5 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
  --out research/findings/raw/sleep_replay_declamp_s$s.json & done; wait
.venv/bin/python -u -m research.runners._teacher_loop_sleep_replay_declamp_derisk --aggregate \
  research/findings/raw/sleep_replay_declamp_s{42,43,44,45,46,47}.json
```
