# gap#1 ROOT CAUSE — the checkpoint was trained in the WRONG recurrence mode; retraining fixes the catastrophe

Found by READING THE WKV forward source (the control-first / read-the-source discipline that works), after many runs
of config archaeology that did not.

## The root cause

`_emerge_wkv_lm_derisk`'s WKV model has multiple recurrence modes: `--recurrence wkv` (DEFAULT — classic normalized
WKV, read-out `Wo` D->D) vs `--recurrence ssm --dual-nonneg` (the `[ap; an]` two-positive-integrator linear SSM,
read-out **`Wo_sp` 2D->D**). **The on-bridge runners realize the `ssm/dual-nonneg` state** (they build `[ap; an]` and
apply `Wo_sp`). **My regenerated checkpoint was trained with the DEFAULT `wkv` mode**, so `Wo_sp` was never trained on
the `[ap; an]` state -> the on-bridge read-out produced near-uniform garbage (NLL 6.7 ~= ln(1000)) DESPITE a
byte-exact state (corr 1.000). The M1 finding's own text names it: "corr(cp_ssm_state, numpy **dual-nonneg** SSM
state) = 1.000".

## The fix confirms it — retrain in the right mode

Retrained with `--recurrence ssm --dual-nonneg` (same V=1000/d=128/n-sent 80000/ep 10):

| checkpoint | off-bridge train | M1 on-bridge (canonical: --ssm-state --use-ssm-readout, corr 1.000) |
|---|---|---|
| WRONG (`wkv` default) | (n/a) | **-3.013** (near-uniform garbage) |
| RIGHT (`ssm --dual-nonneg`) | **+0.429 GO** | **-0.958** |

**-3.013 -> -0.958 confirms the root cause.** The read-out now works (onbridge NLL 4.647, well above uniform), and
the state is byte-exact.

## The RESIDUAL — a ~1.4-nat on-bridge gap remains (precisely characterized, not yet closed)

Off-bridge the model is **+0.429**; on-bridge with a BYTE-EXACT state (corr 1.000) and the SSM's OWN read-out it is
**-0.958** — a ~1.4-nat gap that should NOT exist if the state is exact and the read-out is identical. Candidate
causes (for the next diagnosis, control-first): (a) the on-bridge runner reconstructs a DIFFERENT eval stream
(ev_ids) than the off-bridge eval used, so the two numbers are on different data; (b) a residual state-scaling or
ON/OFF-ordering difference between the on-bridge `[ap; an]` and the read-out's expected input that survives a high
correlation but shifts the logits; (c) the trigram baseline differs between the two evals. The decisive next check:
compare the on-bridge read-out LOGITS against the off-bridge model's OWN logits for the SAME token sequence — if they
match, the gap is the eval stream; if not, it is the state/read-out application.

## Status

- **Primary root cause: FOUND + FIXED** (wrong recurrence mode; retrain -3.013 -> -0.958). The regenerated-checkpoint
  catastrophe is explained and resolved.
- **M1 does NOT yet cleanly reproduce** its off-bridge +0.429 on-bridge (residual -0.958), so the M1 control still
  does not pass, and **the token-SDR encode test remains blocked** until the residual is closed.
- **NEXT (well-specified, control-first):** close the ~1.4-nat residual by comparing on-bridge vs off-bridge logits
  on identical tokens; once M1 on-bridge reproduces ~+0.4, run NEF (must reproduce ~-0.03) then tokensdr, with
  write-fidelity measured on the DEPLOYED accumulated state.
- **The correctly-trained checkpoint** `bridges/wkv_ckpt/wkv_ssm_v1000_d128_seed42.npz` is the usable artifact going
  forward (the `wkv`-mode one is not).

## The lesson, reinforced

Reading the WKV forward source pinned the root cause in ONE step; the preceding config archaeology (n-sentences,
vocab rebuild, firing-rate-vs-graded mode, own-vs-refit read-out) was all downstream noise. The day's standing
critique — READ the source in depth rather than sweep configs — applied here would have saved the entire archaeology.

---

## DECISIVE — NEF vs tokensdr on the CORRECT checkpoint: token-SDR is REFUTED (deployed-worse than NEF)

With the correctly-trained (`ssm/dual-nonneg`) checkpoint, ran all three paths:

| path | verify corr(state, ref) | deep d10-99 vs-trigram |
|---|---|---|
| M1 exact state (--ssm-state --use-ssm-readout) | **1.000** | -0.958 |
| NEF encode (regression) | 0.630 | -3.421 |
| **tokensdr encode (selection)** | **0.524** | **-3.723** |

**Token-SDR (corr 0.524) is DEPLOYED-WORSE than NEF (corr 0.630)** — the deep-NLL is worse too (-3.723 vs -3.421).
This CONFIRMS the earlier retraction on a valid checkpoint: **the gate's "selection beats regression" escape does NOT
hold in deployment.** My standalone write-fidelity 0.906 (> M2's 0.786) was measuring a non-deployed quantity; the
deployed accumulated-state fidelity REVERSES the ordering. **The token-SDR mechanism is REFUTED for gap#1's
spiking-input problem** — it does not beat the NEF regression it was designed to replace.

## Two honest negatives, both now on a VALID checkpoint

1. **The M1-realization residual is real** (~1.4 nats: off-bridge +0.429 vs on-bridge exact-state -0.958). Even a
   byte-exact input state + the own read-out does not recover the off-bridge deep-NLL, so there is a floor the encode
   cannot beat. Cause still uncharacterized (logit comparison is the next check) — but it means NO encode can recover
   a positive deep-NLL until this residual closes.
2. **The token-SDR encode is refuted** (deployed-worse than NEF), so even if the M1 residual were closed, this
   particular escape is not the fix.

## Net gap#1 state

- **The spiking-INPUT path is NOT solved by token-SDR selection.** The gate's #1 candidate is refuted in deployment.
- **An additional M1-realization residual** (exact state -> -0.958, not +0.429) is uncharacterized and is a
  prerequisite blocker.
- **M2's NEF regression (-0.030 in its finding, -3.421 here)** — the discrepancy is the M1-realization residual: M2's
  -0.030 was relative to a working M1 baseline; here even M1 is -0.958, so M2's -3.421 is consistent (both degraded by
  the same ~1.4-nat realization floor plus the encode loss).
- **HONEST OPEN:** the spiking-input recurrent-state problem remains genuinely open. The graded-state M1 result
  (off-bridge) stands; realizing it on-bridge with a spiking input has an unclosed realization residual AND the
  token-SDR escape is refuted. This is a real wall on the spiking-INPUT half, honestly documented — the next
  mechanism must both close the M1-realization residual and encode the input without the accumulated-state fidelity
  loss that sinks both NEF (0.63) and tokensdr (0.52).

## The session's gap#1 tally, honestly

Started from a gate reframe (encode is the wall — stands), built a token-SDR de-risk that looked like it beat M2
(0.906, RETRACTED as non-deployed), chased an invalid harness through four self-corrections, found the root cause by
reading the source (wrong recurrence mode — fixed), and on the valid checkpoint REFUTED the token-SDR (deployed-worse
than NEF) while surfacing a real M1-realization residual. **No positive mechanism result; a rigorous set of honest
negatives + a precisely-located open problem.**
