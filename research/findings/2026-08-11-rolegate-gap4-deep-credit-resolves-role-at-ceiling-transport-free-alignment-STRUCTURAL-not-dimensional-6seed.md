---
type: finding
status: contributing
date: 2026-08-11
mechanism: ROLE-GATE x gap#4 DEEP CREDIT — replace the role-gate's PLAIN REINFORCE credit with an e-prop forward eligibility trace (fixed slow-NMDA-hold time-constant) + a transport-free learning signal from the DISTAL verb-prediction error (three-factor pre x post x DA)
lane: emergence engine / working memory x gap#4 / deep-credit (the convergent unblock)
verdict: 6-SEED real-slot at L=3 (the aligned credit-fidelity ceiling) + 1-seed smoke + slot-free 6-seed calibration. The credit assignment WAS the role-induction residual: gap#4 deep credit with a credit-fidelity CEILING feedback (aligned = R^T, a labelled TRANSPORT shortcut, mirroring gap#4's BPTT ceiling) reaches the MARKER CEILING RELIABLY on all 6 seeds (held-out branch(verb) near ceiling, token-identity gap large, fires pos0 high / pos>0 near-zero on every seed) where plain REINFORCE is HIGH-VARIANCE (a decent mean but its worst seed fails entirely) and the banked HOST position-oracle failed. The eligibility TRACE is the load-bearing ingredient (no-trace collapses); homeostasis is a SEED-MARGINAL reliability aid (one seed collapses without it here, NOT a load-bearing mechanism); permuted-reward collapses; the identity gate fails the crux. THE RESIDUAL, precisely (CORRECTED after adversarial verify — the earlier "F=4 too low-dim" claim was FALSIFIED): the TRANSPORT-FREE feedback does NOT reach role RELIABLY, but NOT because of readout dimension. The non-canonical KP arm (frozen readout R=I, NO weight decay, B-only) has no alignment attractor so B ANTI-aligns (cos(B,I)→−0.60, identical at F=8); a CANONICAL co-adapting KP (Akrout 2019 — co-adapt forward R + feedback B + weight decay) RECOVERS alignment at F=4 (cos(B,Rᵀ) +0.96, 6-seed) → the feedback-alignment residual is STRUCTURAL, not dimensional. And canonical KP PARTIALLY induces role TRANSPORT-FREE (0.637, gap +0.52) — genuine transport-free role learning — but HIGH-VARIANCE (min 0.144), NOT the aligned ceiling (1.000 [min 1.000]). So the residual is now RELIABILITY, not alignment; the strict brain-based role-GO (reliable transport-free role induction) is NOT yet met. See the body for the per-arm numbers.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_var_bind_rolegate_gap4_credit_derisk.py
artifacts:
  - research/findings/raw/_var_bind_rolegate_gap4_credit/rolegate_gap4_credit_6seed_L3.json
  - research/findings/raw/_var_bind_rolegate_gap4_credit/smoke1.json
instrument: reuse-by-import of `_var_bind_role_gate_derisk` (the SAME-POOL positional agreement stream, the SpikingSlot eval on the D3 slow-NMDA HOLD slot, positional_fire = the token-identity crux, the marker/HTM/n-gram baselines + memory teeth) + RUNG6c barcodes. The e-prop eligibility trace + the transport-free learning signal + the homeostatic companion are RUNNER-side host math (their on-substrate spiking DA-gated realisation is the named next rung). SIM_BACKEND=numpy device=cpu; cfg.seed set via SpikingSlot (build_persistent_slot sets cfg.seed = int(seed), the substrate IS seeded). NO sim/ edit.
---
<!--derived-->

# Role-gate x gap#4 deep credit — the credit assignment WAS the residual: deep credit reaches the role ceiling reliably (6-seed) where plain REINFORCE could not; the transport-free feedback-alignment is STRUCTURAL-recoverable (canonical KP, cos +0.94) but NOT sufficient — the corrected residual

The banked role-gate 6-seed HONEST NEGATIVE
(`2026-08-11-role-based-write-gate-same-pool-positional-grammar-recurrent-latch-learns-role-1seed-SMOKE.md`) left a
DECISIVELY-named residual: on a same-pool positional grammar (subject = position 0; distractors = the SAME noun pool at
positions 1..L; the verb agrees with the subject's feature), a reward-driven recurrent-latch write-gate trained by PLAIN
REINFORCE reaches only held-out 0.602 (chance 0.250, token-identity gap +0.45) — and, decisively, even a HOST
position-oracle fails (0.265) with plain REINFORCE. So the residual is the CREDIT ASSIGNMENT (gap#4), not the positional
signal. This de-risk applies the gap#4 deep-credit surpass
(`2026-08-11-gap4-learned-feedback-KP-reaches-the-3rd-hidden-layer-...`,
`2026-08-11-gap4-onspikes-KP-learned-feedback-ALIGNS-...`) to that residual and asks: does deep/eligibility-based credit
let the recurrent role-gate learn role RELIABLY where plain REINFORCE could not?

## The credit-assignment problem, precisely

<!--derived-->
The reward (verb-prediction match) depends causally on the LAST feature left in the WM. Plain REINFORCE accumulates ONE
eligibility (sum over tokens of (a_t - p_t) code_t) and multiplies it by ONE scalar advantage (reward - baseline): EVERY
gate-decision in the sentence receives the SAME credit. It therefore cannot separate "LOAD pos0 was good" from "LOAD a
distractor was bad" — the two decisions are lumped, and a mixed episode muddles them. This is the classic high-variance
temporal-credit failure that e-prop / three-factor plasticity exist to fix.

## The mechanism — gap#4 deep credit, ONLY the training credit rule changes

<!--derived-->
The deep-credit gate is the SAME recurrent-latch policy with the SAME deployment (the REAL spiking D3 slot at eval), so
any difference is attributable to the credit mechanism. It subclasses the banked `PolicyGate('recurrent')` — `p_load` and
`decide` (the deployed policy) are byte-identical to the REINFORCE baseline; only `train` differs. Three ingredients,
each from the gap#4 biology:

1. **A forward ELIGIBILITY TRACE (e-prop, Bellec 2020).** Model the write as a gated memory m_t = (1-p_t) m_{t-1} + p_t
   v_t (v_t = onehot(feat(token_t)); the hard limit is the slot's clear-then-load overwrite). The trace
   E_t = d m_t / d(gate params) is computed FORWARD:
   E_t = a_leak * E_{t-1} + gain*p_t*(1-p_t) * (v_t - m_{t-1}) (x) x_t, where a_leak is the FIXED slow-NMDA-hold
   time-constant (NOT the state-dependent forward retention (1-p): conflating them washes distal credit at init when
   p~0.5 — the corrected leak is the intrinsic hold tau, ~1). A decision whose loaded content SURVIVES to the readout
   keeps a strong eligibility; one that is overwritten decays. THIS is the credit through the intervening fillers plain
   REINFORCE lacks.
2. **A TRANSPORT-FREE learning signal from the DISTAL verb-prediction error.** The readout maps m_T -> verb logits by the
   FIXED agreement lexicon R (feature f -> verb N+f, a host scaffold exactly as verb_of was). delta = softmax(scale*m_T)
   - onehot(feat(subj)) is the verb-prediction error (the SAME reward signal REINFORCE uses — the verb is IN the stream;
   feat(subj) is NEVER read as a role label, only via the verb). The learning signal on m is ell = scale * B @ delta with
   B a SEPARATE feedback matrix (NOT R^T copied). grad = ell . E_T. Three-factor: pre[code eligibility] x post[memory
   sensitivity (v-m), surrogate p(1-p)] x DA[verb-prediction error]. Feedback arms mirror gap#4: fixed = frozen random B
   (transport-free DFA baseline); kp = B co-adapts toward R^T by the matched transposed readout delta (Kolen-Pollack,
   transport-free — the brain-based candidate); aligned = B == R^T == I (weight-TRANSPORT; the credit-fidelity CEILING,
   labelled a shortcut, mirroring gap#4's BPTT ceiling).
3. **Intrinsic firing-rate HOMEOSTASIS (Turrigiano) — an optional companion.** 1 subject vs L distractors biases the
   credit toward "don't fire", which collapsed the gate silent on SOME inits in a low-budget slot-free calibration
   (60 episodes). A slow homeostatic nudge on the bias toward a target mean-rate was added as the companion the standing
   lesson predicts ("what else does the real system run alongside this, that we replaced with a constant?"). HONESTY
   NOTE: homeostasis's necessity is SEED-MARGINAL — in the agent's committed run no-homeo reached 1.000 (homeostasis not
   needed), but in the regenerated run one seed collapses to 0.000 without it (no-homeo 0.833 [min 0.000]); the flip is
   RNG-dependent. It is kept default-on as a robustness aid, NOT claimed as a load-bearing mechanism.

## The DECISIVE result — 6-seed on the REAL spiking slot at L=3 (`research/findings/raw/_var_bind_rolegate_gap4_credit/rolegate_gap4_credit_6seed_L3.json`)

<!--derived-->
N=12 shared nouns, F=4 features (chance 0.250), L=3 (dependency distance 4, 1728 distractor paths, held-out NOVEL
distractor tuples), seeds 42/43/44/100/101/102, matched 80-episode budget. Held-out branch(verb) mean [min over seeds] |
fire pos0 / pos>0 | token-identity gap [min]:

(These numbers are from the REGENERATED 6-seed that adds the canonical-KP arm + the corrected verdict string. It is a
fresh reproduction of the agent's committed run — every CONCLUSION holds and the decimals move only by seed noise: the
agent's run had aligned 0.983 [min 0.896] / REINFORCE@80 0.760 [min 0.271] / no-homeo 1.000, which the 5-lens
adversarial verification reproduced byte-for-byte.)

- **eprop_ALIGNED (deep credit + transport CEILING = Bᵀ): 1.000 [min 1.000] | gap +1.00 [min +1.00] | cos +1.00** —
  RELIABLE across ALL 6 seeds; reaches the marker ceiling; the SAME nouns are LOADed at position 0 and IGNORED as
  distractors. The headline: deep credit makes role induction RELIABLE.
- **REINFORCE @80 (matched budget): 0.748 [min 0.233] | gap +0.67 [min +0.00]** — HIGH-VARIANCE; a decent mean pulled up
  by good seeds while the WORST seed collapses (min 0.233, gap +0.00). The win over REINFORCE is RELIABILITY (min across
  seeds), not the mean — exactly the residual the banked finding named.
- **REINFORCE @8 (banked native budget): 0.606 [min 0.233] | gap +0.52 [min +0.00]** — the high-variance REINFORCE
  continuity check (banked 6-seed was 0.602).
- **eprop_KP (transport-free, NON-canonical: frozen R=I, no weight decay, B-only): 0.189 [min 0.100] | gap -0.02 |
  cos(B,I) -0.03 → -0.60** — does NOT reach role; the feedback ANTI-aligns. **CORRECTED (adversarial verify): this is NOT
  a dimension limit — it is a NON-CANONICAL KP with no alignment attractor. At F=8 it is IDENTICAL (cos → -0.67),
  falsifying "reaches it only at higher dim".**
- **eprop_KP_CANON (transport-free, CANONICAL co-adapting KP: co-adapt forward R + feedback B + weight decay — Akrout
  2019): 0.637 [min 0.144] | gap +0.52 [min -0.15] | cos(B,Rᵀ) +0.10 → +0.96** — the decisive confirm. Canonical KP (i)
  RECOVERS feedback alignment at F=4 (cos +0.96), settling that the alignment residual is STRUCTURAL not dimensional, and
  (ii) PARTIALLY induces role TRANSPORT-FREE (mean 0.637, positive token-identity gap +0.52) — genuine transport-free
  role learning, well above the non-canonical KP. BUT it is HIGH-VARIANCE (min 0.144, gap min -0.15) — comparable to
  REINFORCE, NOT the aligned ceiling (1.000 [min 1.000]). (The seed-42 smoke's 0.167 was a pessimistic outlier.) So the
  transport-free residual is now **RELIABILITY, not alignment**: aligned feedback is recoverable and partially sufficient,
  but not yet reliable role induction.
- **eprop_fixed (transport-free fixed-random DFA): 0.283 [min 0.144] | gap +0.04 | cos -0.08** — fails (single-layer,
  no hidden layer for chained-FA).
- **identity gate (code-only): 0.285 [min 0.222] | gap +0.00** — fails the crux by construction.
- **marker ceiling 1.000; HTM 0.006; chance 0.250; lesion-the-hold 0.000.**

Levers (all EXECUTED via tools.lab / tools.verdict, at 6 seeds): **NO-TRACE** (eligibility leak 0 → only the last
decision credited) **0.254** [min 0.000], gap +0.17 — the eligibility TRACE is DECISIVELY load-bearing (aligned 1.000 vs
0.254). **PERMUTED-REWARD** (verb target shuffled per sentence) **0.106** [min 0.067], gap -0.07 — the learning signal
carries no signal → no learning. **NO-HOMEO** (homeostasis + bias-init OFF) **0.833** [min 0.000], gap +0.83 —
homeostasis is a SEED-MARGINAL reliability aid: one seed collapses to 0.000 without it here (the agent's committed run
had no-homeo 1.000; the necessity flips with RNG), so it is kept default-on as a robustness aid, NOT claimed as a
load-bearing mechanism. All 7 Verdict validity preconditions hold (generalisation defined; marker ceiling exists;
HTM/n-gram at chance; task positional; hold zero-input; deep-credit differs from REINFORCE). The strict brain-based
role-GO (`role_go`) is FALSE — it requires the transport-free feedback to reach role RELIABLY, which is the residual.

## 1-seed real-slot smoke + slot-free 6-seed calibration (supporting)

<!--derived-->
The 1-seed real-slot smoke (`research/findings/raw/_var_bind_rolegate_gap4_credit/smoke1.json`, seed 42) showed the
aligned gate at held-out 1.000, gap +1.00 (fires pos0 1.00 / pos>0 0.00) while REINFORCE@80 collapsed to fire-everything
(0.250, gap +0.00) — confirming deep credit transfers to the real spiking slot. A slot-free surrogate calibration over
the 6 seeds located the operating point (eligibility
leak 1.0, eprop-lr 0.08, readout-scale 3.0) and showed the aligned gate reliable while the transport-free KP/fixed arms
under-reach — reproduced decisively above on the real slot.

## Scope / honesty — what is de-risked, what is the residual (brain-based-only)

<!--derived-->
NO-EXTERNAL-NEEDED: grounded in our OWN verified components (the banked role-gate stream + D3 hold + RUNG6c barcodes +
the gap#4 e-prop/KP machinery). This is a method-positive-with-a-named-residual, not a capability wall.

- **DE-RISKED (the convergent unblock landed).** The credit assignment WAS the role-induction residual: gap#4 deep credit
  (e-prop eligibility trace with the slow-hold time-constant + a distal verb-prediction learning signal) reaches the role
  ceiling RELIABLY across 6 seeds (held-out 1.000 [min 1.000], token-identity gap +1.00 [min +1.00]) where plain
  REINFORCE is high-variance (mean 0.748 but min 0.233 / gap min +0.00 — some seeds fail entirely; banked 0.602) and even
  the banked host position-oracle (0.265) could NOT. The eligibility TRACE is decisively load-bearing (no-trace 0.254);
  the identity gate fails the crux; permuted-reward collapses. Homeostasis is a SEED-MARGINAL robustness aid (one seed
  collapses without it in the regenerated run), NOT claimed load-bearing.
- **THE RESIDUAL, precisely named — CORRECTED after adversarial verify (the earlier "F=4 too low-dim" claim was an
  INSTRUMENT bug, now fixed).** The reliable result uses the TRANSPORT (aligned = Rᵀ) credit-fidelity CEILING feedback —
  a labelled shortcut, exactly as gap#4's BPTT ceiling uses Wᵀ. The TRANSPORT-FREE feedback does NOT yet reach role, and
  the reason is **STRUCTURAL, not dimensional**:
  1. The original KP arm was **non-canonical** — a *frozen* forward readout (R = readout_scale·I, never updated) and a
     B-only update with **no weight decay**. Canonical Kolen-Pollack (Akrout 2019) co-adapts BOTH forward W and feedback
     B *plus* weight decay; the decay is the alignment ATTRACTOR. With W frozen and no decay there is no attractor at any
     dimension, and the self-term drives B to a negative diagonal — exactly the observed cos(B,I) → −0.67, **identical at
     F=8** (the dimension test falsifies "reaches it only at higher dim").
  2. The **CANONICAL co-adapting KP** arm added here (co-adapt R + B + weight decay) **RECOVERS alignment at F=4**
     (cos(B,Rᵀ) +0.10 → +0.96, 6-seed) — decisively settling that the feedback-alignment residual is STRUCTURAL, not a
     dimension limit.
  3. Canonical KP with that aligned feedback **PARTIALLY induces role transport-free** (6-seed 0.637 [min 0.144], gap
     +0.52 [min -0.15]) — genuine transport-free role learning, well above the non-canonical KP — **but HIGH-VARIANCE,
     not the aligned ceiling** (1.000 [min 1.000]). So aligned feedback is necessary and partially sufficient; the
     remaining residual is **RELIABILITY** (the worst seed still fails), not alignment. The likely cause of the variance
     is that the co-adapting readout can absorb some of the credit pressure the gate needs — a genuinely open sub-problem.
  4. **a-1 citation corrected:** the gap#4 record's transport-free wins are (i) chained multi-hop FA + σ′ clearing the
     depth-2 ceiling, and (ii) KP rescuing MNIST at **depth-4, fixed hidden-dim 128** — a DEPTH + co-adaptation result,
     **NOT a dimension-sweep result**. The prior "go higher readout dim" prescription miscited these.
- **Named next build (dependency-ordered) — REVISED.** (a) ✅ the REAL-slot 6-seed confirms reliability (done, above);
  (b) the transport-free residual is now **feedback-alignment-recoverable but insufficient** — so the next rung is NOT
  "higher dim"; it is **chained multi-hop FA + σ′** (needs a hidden layer this single-layer gate lacks), and/or
  **hold/regularise the readout so it cannot absorb the credit**, and/or route the credit through the emergence-engine's
  own (higher-dim, co-adapting) sequence code — trained WITH this deep-credit rule; then (c) the on-substrate spiking
  DA-gated realisation of the e-prop eligibility trace + three-factor update (HOST math here). Reuse-by-import; NO sim/ edit.

## Reproduce

<!--derived-->
1-seed smoke (FOREGROUND):
`SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_gap4_credit_derisk --seeds 42 --distances 3 --n-test 24`

The decisive 6-seed sweep across L=2,3,4 (fan each seed to its own process for core-parallelism, or run the single
self-aggregating process; point `--out` at a `rolegate_gap4_credit_6seed.json` inside the same
`_var_bind_rolegate_gap4_credit` raw directory as the smoke artifact):
`SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_gap4_credit_derisk --seeds 42 43 44 100 101 102 --distances 2 3 4 --n-test 90 --out rolegate_gap4_credit_6seed.json`
