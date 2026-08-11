---
type: finding
status: contributing
date: 2026-08-11
mechanism: ROLE-GATE x gap#4 DEEP CREDIT — replace the role-gate's PLAIN REINFORCE credit with an e-prop forward eligibility trace (fixed slow-NMDA-hold time-constant) + a transport-free learning signal from the DISTAL verb-prediction error (three-factor pre x post x DA)
lane: emergence engine / working memory x gap#4 / deep-credit (the convergent unblock)
verdict: 6-SEED real-slot at L=3 (the aligned credit-fidelity ceiling) + 1-seed smoke + slot-free 6-seed calibration. The credit assignment WAS the role-induction residual: gap#4 deep credit with a credit-fidelity CEILING feedback (aligned = R^T, a labelled TRANSPORT shortcut, mirroring gap#4's BPTT ceiling) reaches the MARKER CEILING RELIABLY on all 6 seeds (held-out branch(verb) near ceiling, token-identity gap large, fires pos0 high / pos>0 near-zero on every seed) where plain REINFORCE is HIGH-VARIANCE (a decent mean but its worst seed fails entirely) and the banked HOST position-oracle failed. The eligibility TRACE is the load-bearing ingredient (no-trace collapses); homeostasis is an optional low-budget robustness aid (NOT load-bearing at this budget); permuted-reward collapses; the identity gate fails the crux. THE RESIDUAL, precisely: the TRANSPORT-FREE (brain-based) feedback (KP/fixed) does NOT yet reach role at the F=4 readout dim — the SAME feedback-alignment sub-problem the gap#4 lane already carries; the strict brain-based role-GO is therefore NOT yet met. See the body for the per-arm numbers.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_var_bind_rolegate_gap4_credit_derisk.py
artifacts:
  - research/findings/raw/_var_bind_rolegate_gap4_credit/rolegate_gap4_credit_6seed_L3.json
  - research/findings/raw/_var_bind_rolegate_gap4_credit/smoke1.json
instrument: reuse-by-import of `_var_bind_role_gate_derisk` (the SAME-POOL positional agreement stream, the SpikingSlot eval on the D3 slow-NMDA HOLD slot, positional_fire = the token-identity crux, the marker/HTM/n-gram baselines + memory teeth) + RUNG6c barcodes. The e-prop eligibility trace + the transport-free learning signal + the homeostatic companion are RUNNER-side host math (their on-substrate spiking DA-gated realisation is the named next rung). SIM_BACKEND=numpy device=cpu; cfg.seed set via SpikingSlot (build_persistent_slot sets cfg.seed = int(seed), the substrate IS seeded). NO sim/ edit.
---
<!--derived-->

# Role-gate x gap#4 deep credit — the credit assignment WAS the residual: deep credit reaches the role ceiling where plain REINFORCE could not; the transport-free realisation is the named residual (1-seed SMOKE + slot-free 6-seed calibration)

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
   CORRECTION from the 6-seed real-slot run: at the 80-episode budget homeostasis is NOT load-bearing — the eligibility
   trace alone reaches the ceiling reliably (see the no-homeo lever below), and homeostasis marginally lowers one seed.
   It is kept default-on as a low-budget robustness aid, not claimed as a load-bearing mechanism.

## The DECISIVE result — 6-seed on the REAL spiking slot at L=3 (`research/findings/raw/_var_bind_rolegate_gap4_credit/rolegate_gap4_credit_6seed_L3.json`)

<!--derived-->
N=12 shared nouns, F=4 features (chance 0.250), L=3 (dependency distance 4, 1728 distractor paths, held-out NOVEL
distractor tuples), seeds 42/43/44/100/101/102, matched 80-episode budget. Held-out branch(verb) mean [min over seeds] |
fire pos0 / pos>0 | token-identity gap [min]:

- **eprop_ALIGNED (deep credit + transport CEILING): 0.983 [min 0.896] | 0.98 / 0.01 | gap +0.98 [min +0.89]** —
  RELIABLE across ALL 6 seeds; reaches the marker ceiling; the SAME nouns are LOADed at position 0 and IGNORED as
  distractors. This is the headline: deep credit makes role induction RELIABLE.
- **REINFORCE @80 (matched budget): 0.760 [min 0.271] | 0.97 / 0.30 | gap +0.67 [min +0.00], std 0.339** — HIGH-VARIANCE;
  a decent mean pulled up by good seeds while the WORST seed collapses (min 0.271, gap +0.00). The win over REINFORCE is
  RELIABILITY (min across seeds), not the mean — exactly the residual the banked finding named.
- **REINFORCE @8 (the banked native budget): 0.538 [min 0.125] | 0.84 / 0.42 | gap +0.40** — reproduces the banked
  high-variance REINFORCE (the banked 6-seed was 0.602).
- **eprop_KP (transport-free learned feedback): 0.188 [min 0.125] | 0.21 / 0.24 | gap -0.05, cos(B,I) -0.03 -> -0.67** —
  does NOT reach role; the feedback ANTI-aligns at the F=4 readout dim.
- **eprop_fixed (transport-free DFA): 0.267 [min 0.083] | 0.28 / 0.26 | gap +0.01** — fails.
- **identity gate (code-only, existing): 0.288 [min 0.250] | 0.68 / 0.64 | gap +0.00** — fails the crux by construction.
- **marker ceiling 1.000; HTM 0.010; n-gram 0.304; chance 0.250; lesion-the-hold 0.000; permuted-position 0.253.**

Levers (all EXECUTED via tools.lab / tools.verdict, at 6 seeds): **NO-TRACE** (eligibility leak 0 -> only the last
decision credited) **0.194** [min 0.000], gap -0.03 — the eligibility TRACE is DECISIVELY load-bearing (0.983 vs 0.194).
**PERMUTED-REWARD** (verb target shuffled per sentence) **0.056** [min 0.000], gap -0.04 — the learning signal carries no
signal -> no learning. **NO-HOMEO** (homeostasis + bias-init OFF) **1.000** [min 1.000], gap +1.00 — homeostasis is NOT
load-bearing at this budget (the trace alone is sufficient and even cleaner). All 7 Verdict validity preconditions hold
(generalisation defined; marker ceiling exists; HTM/n-gram at chance; task positional; hold zero-input; deep-credit
differs from REINFORCE). The strict brain-based role-GO (`role_go`) is FALSE — it requires the TRANSPORT-FREE feedback to
reach role, which is the residual.

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
  ceiling RELIABLY across 6 seeds (held-out 0.983 [min 0.896], token-identity gap +0.98 [min +0.89]) where plain
  REINFORCE is high-variance (mean 0.760 but min 0.271 / gap min +0.00 — some seeds fail entirely; banked 0.602) and even
  the banked host position-oracle (0.265) could NOT. The eligibility TRACE is decisively load-bearing (no-trace 0.194);
  the identity gate fails the crux; permuted-reward collapses. Homeostasis is an optional low-budget robustness aid, NOT
  load-bearing here (no-homeo also reaches 1.000).
- **THE RESIDUAL, precisely named (brain-based).** The reliable result uses the TRANSPORT (aligned = R^T) credit-fidelity
  CEILING feedback — a labelled shortcut, exactly as gap#4's BPTT ceiling uses W^T. The TRANSPORT-FREE (brain-based)
  feedback (KP-learned / fixed-random DFA) does NOT yet reach role at the F=4 readout dim. This is the SAME
  feedback-alignment sub-problem the gap#4 lane already carries: fixed-DFA cannot align intermediate feedback; KP-learned
  reaches deep credit only at higher dimension. The role-gate's F=4 verb readout is too low-dim for KP to align. This
  directly links the role-gate residual back to the gap#4 lane's own open feedback-alignment rung.
- **Named next build (dependency-ordered).** (a) the REAL-slot 6-seed sweep (the command below) to confirm reliability
  on the spiking substrate; then (b) close the transport-free feedback-alignment residual (a higher-dim / weight-mirror
  learned feedback, or route the credit through the emergence-engine's own sequence code that already carries a
  higher-dim positional signal); then (c) the on-substrate spiking DA-gated realisation of the e-prop eligibility trace +
  three-factor update (the trace + learning signal are HOST math here). Reuse-by-import; NO sim/ edit.

## Reproduce

<!--derived-->
1-seed smoke (FOREGROUND):
`SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_gap4_credit_derisk --seeds 42 --distances 3 --n-test 24`

The decisive 6-seed sweep across L=2,3,4 (fan each seed to its own process for core-parallelism, or run the single
self-aggregating process; point `--out` at a `rolegate_gap4_credit_6seed.json` inside the same
`_var_bind_rolegate_gap4_credit` raw directory as the smoke artifact):
`SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_gap4_credit_derisk --seeds 42 43 44 100 101 102 --distances 2 3 4 --n-test 90 --out rolegate_gap4_credit_6seed.json`
