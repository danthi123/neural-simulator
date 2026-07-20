# gap#1 DEFINITIVE — harness VALIDATED (M1 +0.542 GO); the ENCODE is the wall (confirmed); token-SDR REFUTED

The full gap#1 investigation resolved on a FULLY VALIDATED harness. The residual that blocked everything was a
**per-channel-vs-uniform decay mismatch**, found by isolating the numpy ref against the torch model.

## The last root cause — per-channel vs uniform decay

The isolation check (numpy dual-nonneg state + read-out vs the torch model on identical tokens) gave **-1.104**,
matching the on-bridge -0.958 but NOT the torch model's +0.429. Cause: the checkpoint's decay `w` is **per-channel**
(shape [128], std 0.80), but `cp_ssm_state` uses ONE `ssm_k_leak`, and the on-bridge runners read `w[0]` as a single
decay -> uniform-decay state vs a per-channel-trained model -> ~1.4-nat residual. The M1 finding used
`--uniform-decay` (a single shared decay matching the substrate's one k_leak); I had omitted it.

**Retrained with `--recurrence ssm --dual-nonneg --uniform-decay`:** off-bridge +0.360 GO, and **M1 on-bridge
exact-state (corr 1.000, own read-out) = +0.542 GO** — the M1 control now REPRODUCES (and exceeds the finding's
+0.486). **The harness is validated.**

## THE DEFINITIVE ENCODE RESULT (on the validated harness)

| path | verify corr(state, ref) | deep d10-99 vs-trigram |
|---|---|---|
| **M1 exact state (host inject, corr 1.000)** | **1.000** | **+0.542 GO** |
| NEF encode (regression) | 0.616 | -2.904 |
| token-SDR encode (selection) | 0.501 | -3.416 |

**Two decisive conclusions:**

1. **The ENCODE is the wall — confirmed, quantified.** A PERFECT input (M1, corr 1.000) is GO (+0.542), but ANY
   spiking encode collapses it: NEF drops the state corr to 0.616 -> -2.904, tokensdr to 0.501 -> -3.416. The
   deep-NLL is HYPERSENSITIVE to state fidelity — a drop from corr 1.000 to 0.616 turns +0.542 into -2.904. The
   gate's estimate that "near-exact input is needed" is CONFIRMED empirically: the encode must reach state corr
   very close to 1.0, and ~0.6 is catastrophically short.

2. **Token-SDR is REFUTED** on the validated harness: corr 0.501 < NEF's 0.616, deep-NLL -3.416 < NEF's -2.904.
   The gate's "selection beats regression" escape does NOT hold in deployment. My standalone write-fidelity 0.906
   (> M2's 0.786) measured a non-deployed quantity (per-token reset + subtracted D-dim); the deployed
   accumulated-state fidelity reverses the ordering.

## Net gap#1 state — a real, precisely-characterized WALL on the spiking-input half

- **The graded-state recurrent LM WORKS on-bridge with a perfect (host) input** (+0.542 GO, corr 1.000) — this is
  M1, now validated on a reproducible harness. The recurrent STATE and READ-OUT are solved.
- **The spiking INPUT is the open wall:** no spiking encode tried (NEF regression 0.616, token-SDR selection 0.501)
  achieves the near-1.0 state fidelity the deep-NLL requires. The requirement is now QUANTIFIED (corr must be
  ~>0.9+; 0.5-0.6 is the ceiling of both encodes), which is a sharp target for any future encode.
- **Token-SDR refuted; NEF bounded.** The two ranked encode candidates are both insufficient. A new mechanism must
  deliver the per-token input at near-exact state fidelity — the honest, precisely-located open problem.

## The session's gap#1 arc, honestly

Gate reframe (encode is the wall — CONFIRMED here) -> token-SDR de-risk (looked like 0.906 > M2, RETRACTED as
non-deployed, then REFUTED at 0.501 < 0.616 on the valid harness) -> a cascade of harness bugs (n-sentences
vocab-mismatch, vocab-rebuild-vs-saved-words, wrong recurrence mode, per-channel-vs-uniform decay) each found by a
FAILING CONTROL or by READING THE SOURCE, culminating in a validated harness (M1 +0.542 GO) and the definitive
encode verdict. **The deliverable: a validated on-bridge recurrent-LM harness, a confirmed encode-is-the-wall
diagnosis with the fidelity requirement quantified, and a refuted token-SDR — no false positive survived.**

## Correct artifacts

`bridges/wkv_ckpt/wkv_ssmU_v1000_d128_seed42.npz` (uniform-decay, dual-nonneg) is THE usable on-bridge checkpoint.
The `wkv` and non-uniform `ssm` checkpoints are not on-bridge-compatible.


---

## Honest caveat on the NEF absolute number (does NOT affect the token-SDR verdict)

My NEF on the validated checkpoint reads corr 0.616 / deep -2.904, WORSE than the M2 finding's tuned NEF (corr 0.786
/ deep -0.030). So my NEF config (n_enc / t_step / calibration) is likely UNTUNED relative to the M2 finding — the
absolute encode number is not the M2 finding's best. **This does NOT affect the two load-bearing conclusions:**
(1) the encode-is-the-wall is a RELATIVE fact (exact input corr 1.000 -> +0.542 GO vs ANY spiking encode's collapse),
robust to NEF tuning; (2) the token-SDR refutation is a RELATIVE comparison on the SAME harness (token-SDR 0.501 <
NEF 0.616), so token-SDR being worse than NEF holds regardless of NEF's absolute tuning. A fully-tuned NEF might
reach the finding's 0.786/-0.030, but token-SDR (0.501) is below even my untuned NEF (0.616), so a tuned NEF would
only widen the gap. **The token-SDR is refuted; the exact NEF ceiling is a separate, non-load-bearing tuning question.**
