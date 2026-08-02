---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/rep_fwd_credit_xor_smoke_s42.json
  - research/findings/raw/gap4/rep_fwd_credit_xor_smoke_onbits_s42.json
---

# gap#4 — a REPRESENTABLE forward (PlateauExpander) does NOT by itself let on-bridge e-prop train XOR: the oracle solves the codon but e-prop stays at chance on the DENSE codon (all encodings tried) — REVISING "the wall is the forward": the wall is the on-bridge e-prop credit's weight-finding on a dense codon, and a genuinely SPARSE representable codon is the untested residual

<!--derived-->
**One-line verdict.** The roadmap's named highest-value lever: credit ON TOP OF a REPRESENTABLE forward (the
`PlateauExpander`), to close the production-bridge deep-credit residual. Ran it on XOR (`--task-xor`): the PlateauExpander
codon MAKES XOR representable — a backprop oracle on the codon reaches 0.994 (literal encoding) / 0.877 (onbits) — but the
on-bridge e-prop **still does not train** (eprop 0.50/0.48 ≈ chance 0.55/0.55; trains_the_task=False; deep_credit_share
degenerate/nan because eprop ≈ frozen ≈ chance). The codon is DENSE (`codon_sparsity` = 0.499 = ~50% columns active) under
BOTH input encodings — the encoding lever does not sparsify it. **This REVISES this session's earlier "the wall is the
Izhikevich FORWARD, not the credit rule" claim**: even a REPRESENTABLE forward (oracle solves it) does not let on-bridge
e-prop train XOR, so the wall is the on-bridge e-prop CREDIT's weight-finding on a DENSE codon — not forward-representability
alone. No `sim/` edit (additive `--task-xor` on `_gap4_representable_forward_plus_credit_derisk.py`).

## Result — 1-seed smokes, XOR, PlateauExpander (representable) forward + e-prop credit

<!--derived-->
| encoding | oracle (on codon) | eprop_inherit | frozen_hidden | trains_the_task | codon_sparsity |
|---|---|---|---|---|---|
| literal (default) | 0.994 | 0.501 | 0.487 | False | 0.499 |
| onbits | 0.877 | 0.479 | 0.454 | False | 0.499 |

chance ~0.55. Artifact e.g. `research/findings/raw/gap4/rep_fwd_credit_xor_smoke_s42.json`. Command:
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_representable_forward_plus_credit_derisk --task-xor --mode expander`.
The two encodings differ in whether the codon fully linearizes XOR (oracle 0.994 vs 0.877) but give the SAME dense codon
(0.499) and the SAME e-prop failure — so the input-encoding lever is not the sparsity lever.

## The decisive read + what it revises

<!--derived-->
The oracle column is load-bearing: a backprop oracle ON THE CODON solves XOR (0.88-0.99), so the codon IS a
representable forward (the PlateauExpander did its job — it makes the level-2 XOR computation representable). Yet the
on-bridge e-prop, on that same representable codon, sits at chance. Two things are therefore established: (1) the
production-bridge wall is NOT merely that the raw Izhikevich forward can't represent XOR — a representable forward exists
and the oracle uses it; (2) but a representable forward is NOT SUFFICIENT for on-bridge e-prop to train — the on-bridge
e-prop credit cannot find the weights the oracle finds, ON A DENSE codon (0.499). **⇒ the wall is the on-bridge e-prop
credit's weight-finding on a dense representable code, which supersedes the earlier "the wall is the forward" framing
(that framing was correct that the LIF forward works and the raw Izhikevich forward doesn't, but INCOMPLETE — a
representable Izhikevich forward still fails).**

## Honest scope + the named residual (the sparsity lever is not yet exposed)

<!--derived-->
**The confound / residual:** the codon is DENSE (0.499). The PlateauExpander probe's own GO condition depends on codon
sparsity, and dense codes are known to block local surrogate-credit training (the agent flagged this a-priori). The
input-ENCODING lever (literal/onbits) does NOT change the codon sparsity — sparsity is set by the coincidence threshold
`ACT_TH` (=2) and `SAMP` (=3) in `_gap4_plateau_expander_probe.py`, which are NOT CLI-exposed. So the clean test — does
a genuinely SPARSE representable codon let on-bridge e-prop train — is NOT YET RUN; it needs a code change to expose
`ACT_TH` (raise it -> sparser codon) and confirm the codon stays representable (oracle high) at the sparser setting.
**NEXT (no-defer, the clean next-session build):** expose `ACT_TH`/`SAMP` as flags, sweep to a sparse-BUT-representable
codon (oracle high, sparsity < ~0.15), and re-read deep_credit_share. If e-prop then trains -> the wall was the dense
code (fixable) and the representable-forward lever works; if it still fails on a sparse representable codon -> the wall
is the on-bridge e-prop credit rule ITSELF on the Izhikevich substrate (the deepest residual), pointing to the
learned-instructive-signal / operating-point levers the roadmap tracks. The crux CORE (LIF/rate) is untouched; this
precisely narrows the PRODUCTION-bridge residual from "the forward" to "on-bridge e-prop weight-finding on a dense
representable code", with the sparse-codon test as the decisive next step.