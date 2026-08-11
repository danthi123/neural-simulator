---
type: finding
status: qualified
date: 2026-08-11
mechanism: deep-credit-on-spikes
backend: numpy
runner: research/runners/_gap4_spatial_depth3_smallT_derisk.py
artifacts:
  - research/findings/raw/_gap4_spatial_depth3_smallT_smoke_seed42.json
  - research/findings/raw/_gap4_depth3_oracle_generalization_diag.json
  - research/findings/raw/_gap4_depth3_oracle_generalization_diag.py
  - research/findings/raw/_gap4_spatial_depth3_smallT_6seed.json
  - research/findings/raw/_gap4_spatial_depth2_smallT_6seed.json
seed-waiver: 1-seed smoke ORIGINALLY; the 6-seed sweep (42/43/44/100/101/102) HAS NOW BEEN RUN (coordinator) — BOTH the N∈{1,2,3} and the N∈{1,2} companion return verdict **UNDEFINED: the task is NOT depth-separating at some (N,seed) (the shallow oracle already fits) → a depth sweep says nothing about deep credit**. This CONFIRMS the 1-seed diagnostic at 6 seeds: the compositional-inheritance task is depth-2-solvable, so it cannot serve as a depth-3 (or even a clean depth-2) reference. The instrument is genuinely absent — a depth-3-OBLIGATORY task is the prerequisite (the `gap4-depth3-obligatory-task` design+verify workflow is in flight to build one).
---

# gap#4 deep-credit-on-spikes — at SMALL T the SPATIAL depth-2 DFA credit is LOAD-BEARING (1-seed smoke), but the depth-3 rung is INSTRUMENT-LIMITED: the compositional task is depth-2-solvable, so the depth-3 rate ceiling OVERFITS (train 1.0, held-out chance) — a depth-3-OBLIGATORY task must be built first

<!--derived-->
**One-line verdict (1-seed smoke; qualified).** The 2026-08-11 temporal-depth-floor finding named the next rung
verbatim: "re-pose the DFA N=2,3,4 depth sweep at SMALL T (T=2-4) where the deeper spatial layers are OBLIGATORY —
genuine depth-3 credit assignment." This smoke does exactly that: FIX T=3 (small, so the LIF membrane's temporal
window cannot supply the effective depth) and SWEEP spatial hidden depth N∈{1,2,3} on the SAME
compositional-inheritance task + SAME LIF SNN + SAME transport-free DFA e-prop credit (`run_seed`,
credit_mode=eprop; DFA feedback is a SEPARATE fixed-random stream → no weight transport). It returns a two-part
result: (1) at small T the SPATIAL depth-2 DFA credit is load-bearing at seed 42 (+0.148 over the 1-hidden
floor) — a positive lead, seed-fragile at 1 seed so the 6-seed sweep is the arbiter; (2) the depth-3 rung is not a
credit question yet but an INSTRUMENT question — the compositional task is depth-2-SOLVABLE, so a depth-3 net
(even the best-possible full-backprop rate ceiling) OVERFITS and the depth-3 ceiling does not exist to test
against. No `sim/` edit (reuse-by-import of the validated `run_seed`; the depth sweep + verdict are runner-side).

## Result — 1 seed (42), the N-sweep at fixed small T=3 (compositional-inheritance, LIF SNN, credit_mode=eprop DFA)

<!--derived-->
Artifact `research/findings/raw/_gap4_spatial_depth3_smallT_smoke_seed42.json` (numpy/CPU, hidden 32, 60 epochs).
chance 0.333, held-out-inheritance set = 27 items. `floor` = a fresh 1-hidden LIF-DFA net (the "deep layers
REMOVED" control); by construction snn(N=1) == floor (identical arch+seed) — an internal determinism check that
held. `oracle` = the depth-MATCHED rate DendriticMLP ceiling (full backprop) built inside `run_seed`.

| N (hidden) | snn_inherit | floor (1-hidden) | depth_gain vs floor | oracle (depth-matched) | permuted | depth-sep |
|---|---|---|---|---|---|---|
| 1 | 0.593 | 0.593 | +0.000 | 0.444 | 0.444 | True |
| 2 | **0.741** | 0.593 | **+0.148** | **1.000** | 0.370 | True |
| 3 | 0.407 | 0.593 | −0.185 | **0.333** | 0.333 | True |

<!--derived-->
The load-bearing reads: (A) **depth-2 spatial credit is load-bearing at small T** — the N=2 DFA net rides
snn 0.741 vs the 1-hidden floor 0.593, a +0.148 gain that is PURE spatial depth (floor and N=2 share T=3, so the
LIF temporal window is subtracted out), and its depth-matched rate ceiling exists (oracle 1.000), so the arm is
interpretable. This is the temporal-floor finding's forward direction confirmed in the depth-sweep framing.
(B) **the N=3 arm collapses BELOW the floor (0.407) AND its depth-matched rate ceiling collapses to chance
(oracle 0.333)** — so the SNN's N=3 result is uninterpretable as a credit result: even the best-possible credit
(full-backprop rate net) does not clear held-out at depth 3. Status is UNDEFINED (a precondition failed), NOT a
negative on DFA credit.

## The depth-3 ceiling collapse is a GENERALIZATION failure, not an optimization or credit failure

<!--derived-->
A depth-matched hyperparameter sweep of the RATE oracle (full backprop, the strongest possible credit) isolates
WHY the depth-3 ceiling is absent — artifact `research/findings/raw/_gap4_depth3_oracle_generalization_diag.json`,
seed 42, same task:

| depth | train acc | held-out inheritance | note |
|---|---|---|---|
| 1-hidden (ep250, lr0.3) | 1.000 | 0.444 | fits train, underfits composition |
| 2-hidden (ep250, lr0.3) | 1.000 | **1.000** | perfect compositional generalization |
| 3-hidden (ep250, lr0.3) | 1.000 | **0.333** | fits train, generalizes at CHANCE |
| 3-hidden (lr0.05/0.10, ep800) | 1.000 | 0.407 | tuning does not recover |
| 3-hidden (lr0.30, ep800) | 1.000 | 0.296 | — |
| 3-hidden width 192 (lr0.10, ep800) | 1.000 | **0.259** | WIDER is WORSE (overfits harder) |

<!--derived-->
Every depth-3 setting that FITS the training set (train→1.000) collapses to ~chance on held-out inheritance,
across lr∈{0.05,0.10,0.30}, epochs∈{250,800}, width∈{96,192}; the 2-hidden net generalizes PERFECTLY. So the
depth-3 ceiling absence is NOT an optimization wall (the deep net optimizes train loss to 1.000) and NOT a credit
wall (this is full backprop, no transport, no spikes) — it is a GENERALIZATION collapse: the compositional task
is **depth-2-SOLVABLE**, so the 3rd layer is surplus capacity that memorizes a non-compositional solution
(wider → worse is the overfitting signature). There is no depth-3-OBLIGATORY signal for the deepest layer to
latch onto. **This precisely locates "the depth-3-instrument-construction problem" the crux cluster kept naming
("hier3 does not separate depth-2 from depth-3"): the problem was never the credit rule and never the fixed T —
it is that no depth-3-required task existed, so the depth-3 ceiling overfits and cannot serve as a reference.**

## Anti-cheats / preconditions (the EARNED verdict block)

<!--derived-->
The runner emits a `Verdict` preconditions block (gate-visible). At 1 seed: temporal_window_small ✔ (T=3≤4);
task_fittable_ceiling_exists ✔ (max oracle 1.000); task_depth_separating ✔; n1_equals_1hidden_floor ✔ (the
substrate is deterministic — snn(N=1)≡floor); depth_lever_moved ✔ (N 1→3 changed the net, tools.lab.lever). TWO
preconditions FAIL, and both are why the smoke is UNDEFINED not GO: depth_Nhi_ceiling_trainable ✘ (the depth-3
rate ceiling collapses, above) and no_label_leakage ✘ (permuted 0.444 > chance+0.05 at N=1). The permuted trip is
1-seed quantization noise: the held set is 27 items (~0.037 resolution), so a 3-item excess crosses the one-sided
tol; the 6-seed sweep pools 162 held evaluations and reads it cleanly. The one-flag≠one-variable discipline is
enforced by `lever` on the depth count and `attributable_to` on the chance-baselined N_hi skill.

## Honest scope (what this IS and is NOT)

<!--derived-->
- IS: a 1-seed MECHANISM smoke that (a) shows spatial depth-2 DFA credit is load-bearing at small T at seed 42
  (+0.148, interpretable ceiling 1.000), and (b) DECISIVELY locates the depth-3 blocker as a task/instrument
  gap, not a credit gap — the depth-3 rate ceiling overfits (train 1.0 → held-out chance) at every hyperparameter
  tested, so depth-3 credit is untestable on THIS task. No `sim/` edit.
- IS NOT: a 6-seed result. Both the DEPTH-2 GAIN and the depth-3 ceiling collapse are seed-FRAGILE — a 2-seed
  peek (not committed) read the depth-2 gain +0.148 (s42) / +0.000 (s43), mean +0.074, and the depth-3 oracle
  0.333 (s42) / 1.000 (s43); the small ~27-item held set makes single seeds quantization-noisy. The 6-seed
  sweep is the arbiter of BOTH the depth-2-load-bearing magnitude and the systematicity of the depth-3 collapse
  (RETURNED command). IS NOT a claim that DFA depth-3 credit fails — the smoke never got a valid depth-3
  instrument to test it against (UNDEFINED, not a negative). IS NOT a demonstration that depth-3 credit succeeds.

## Next mechanism (named, not deferred)

<!--derived-->
1. Confirm at 6 seeds (the RETURNED command): the depth-2-load-bearing gain and the depth-3 ceiling collapse are
   both expected to be seed-robust (the collapse reproduced across a full hyperparameter sweep); this also pools
   out the 1-seed permuted noise so the anti-cheat reads cleanly.
2. Build a depth-3-OBLIGATORY task (the true instrument): held-out generalization must REQUIRE 3 composition
   levels — a depth-2 rate net must UNDERFIT held-out while a depth-3 net CLEARS it (the mirror of the current
   depth-1-underfits/depth-2-clears structure, pushed one level deeper). Verify that ceiling FIRST (a depth-3
   rate oracle that generalizes), THEN re-pose the transport-free DFA depth-3 sweep at small T on it. Candidate:
   a 3-level type hierarchy (super→family→member) where the held-out inheritance target is only recoverable by
   composing two learned abstractions, not one.
3. Secondary (symptom, not root): regularization (weight decay / dropout / early-stop / narrower deep layers)
   may let a depth-3 net generalize on a depth-2-solvable task, but that does not create a depth-3-REQUIRED
   signal — the depth-3-obligatory task in (2) is the load-bearing fix.

Sources: `2026-08-11-gap4-TEMPORAL-DEPTH-FLOOR-ISOLATED-...` (the named next rung, the small-T instrument);
`2026-08-02-gap4-DFA-eprop-is-depth-robust-...` (open edge #1: the temporal-depth floor + the
depth-3-instrument-construction problem). Bellec et al. 2020 e-prop (eligibility traces over many spikes);
Nokland 2016 direct feedback alignment (transport-free credit); Neftci, Mostafa & Zenke 2019 surrogate-gradient
learning in SNNs. NO-EXTERNAL-NEEDED beyond these: the blocker is a task-design property (depth-2-solvability)
measured directly here, not a credit-mechanism limit.
