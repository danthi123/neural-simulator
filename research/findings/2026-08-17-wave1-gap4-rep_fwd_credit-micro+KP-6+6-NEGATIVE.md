---
type: finding
status: negative
date: 2026-08-17
mechanism: wave1-banking
---
## gap4 rep_fwd_credit micro+KP (6+6): deep credit on spikes stays at chance — NEGATIVE (0/12)

Two 6-seed sets, probe `gap4_representable_forward_plus_credit` (n_prop=3, expander, act_th=3, 2 hidden layers, held-out XOR).
- **KP set** (`rep_fwd_credit_xor_kp_local_s{42,43,44,100,101,102}`): credit=`kp_learned_feedback` (Kolen-Pollack learned feedback, kp_lr=0.1). **0/6 SIGNAL, all NOT-GO, trains_the_task=false.**
- **"micro" set** (`rep_fwd_xor_actth3_s{...}`): parent-labeled "micro" but is the PLAIN e-prop baseline (no `credit` field, microcircuit/learned_feedback absent). **0/6 SIGNAL, trains_the_task=false.** The real microcircuit-selfpredict rule is smoke-only (single-seed s42).

Result: the forward is fine — `oracle_inherit` 0.92-0.96 (linear readout on the frozen codon solves XOR) — but the on-spikes credit rule (KP OR plain e-prop) leaves held-out accuracy at chance (~0.45-0.53). `frozen_hidden_inherit ≈ eprop_inherit`, so any above-chance is the random reservoir, not learned deep credit.

Load-bearing (verify-go): controls clean — permuted & shuffle-DFA at chance, `codon_reproducibility=1.0`; `deep_credit_share` correctly UNDEFINED (NaN), not 0; `eprop_ff_weight_moved`~1.1M proves weights move yet carry no task credit.

Wall context (deep credit on spikes): `onbridge_eprop_XOR_K8` = 1/6 (only s102 trains); `rep_fwd_fa_convergence_izh` n_converges=0, FA cos_top_final -0.11..-0.14 (feedback alignment never converges on Izhikevich).

Residual (no overclaim): negative for THESE configs; per project memory plausibly an Izhikevich read-regime / per-arm-lr hyperparameter artifact, NOT a proof of impossibility.

Banked artifacts (this branch): KP set `research/findings/raw/gap4/rep_fwd_credit_xor_kp_local_s*.json` (+.prov.json); e-prop baseline `research/findings/raw/gap4/rep_fwd_xor_actth3_s*.json` (+.prov.json), seeds 42/43/44/100/101/102 each.

Note: the KP-set raw JSONs are historical pre-gate artifacts (no `preconditions` block) retained at their origin path (cited above); the e-prop-baseline actth3 arm JSONs are on main and this finding carries the verdict evidence.
