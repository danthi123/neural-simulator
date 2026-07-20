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
