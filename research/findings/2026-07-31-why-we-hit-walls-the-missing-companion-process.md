---
type: finding
lane: audit
status: live
date: 2026-07-31
claim_check: synthesis
---

# Why we hit walls: the biology is a SYSTEM of processes and we implement ONE, substituting a constant for the rest

**A synthesis, not a new measurement.** Every claim below is evidence from today's session, each already
recorded in its own finding. Written down because it was articulated in conversation, and this project has
established that a conclusion living only in chat does not survive the session.

## 0. Evidence

`research/findings/raw/gap5_density/AGG_clamp_budget.json` · `AGG_laps_dwell.json` ·
`research/findings/raw/laneD_norm/AGG_norm_arms.json` · `research/findings/raw/_affect_state_region_6seed.json`

## 1. The question

Given how much is known about the biology, what actually causes friction in emulating it? Today produced four
answers, all visible in the same session's failures.

## 2. Biology runs interacting processes; we implement one and proxy the rest with a constant

Real cortex runs potentiation AND heterosynaptic depression AND synaptic scaling AND competition AND adaptation
simultaneously, each holding the others in a viable regime. We implement one rule and substitute a static proxy —
usually a hard bound — for everything else. **Then the proxy dominates.**

- gap#5: `w_max=150` against `W0=250` made **97% of the measured weight change the CLAMP**, identical in the
  `lr=0` control. We were measuring a bound standing in for homeostasis, not BTSP.
- lane D: potentiation-only toward a ceiling with no heterosynaptic depression ⇒ ON and OFF converge and the
  signed receptive field cancels — `on_mean` **9.1544** against `|on−off|` **0.0678** (3 seeds,
  `AGG_norm_arms.json`).

> ⛔ **CORRECTION, caught by a parallel agent reading the artifact rather than this document.** This line
> first read "`on_mean` 9.19 vs `off_mean` 9.17". The aggregate holds `on_mean` 9.1544 and **no
> `off_mean` at all** — 9.19/9.17 came from a SINGLE-SEED smoke and were quoted as if they were the
> 3-seed aggregate. The conclusion is unchanged (the channels converge; the difference is 0.7% of the
> mean), but the numbers were not what the cited artifact says. Exactly the class this document is about.
>
> ⚠️ **A SECOND DISCREPANCY, left OPEN rather than resolved by picking one:** `AGG_norm_arms.json` records
> `abs_on_minus_off` **0.0538** for the meansub arm, while the per-seed table in the lane-D finding reports
> **0.0164**. Those are `|mean|` and `mean|·|` — DIFFERENT QUANTITIES, which is failure class 6. Neither is
> wrong; they answer different questions, and the aggregate does not say which it computed. Recorded here
> so nobody reconciles them by choosing the convenient one. <!--derived-->
- affect: an attractor implemented without its eviction mechanism ⇒ a measured RATCHET, mood never returns
  (0.0942 → 0.0962 → 0.0904 → 0.0984 across HIGH → LOW → LOW → silence).

**A clamp is not homeostasis. It is a scalar where biology has a process.**

## 3. Papers give the mechanism; the operating point is implicit in the animal

Bittner specifies plateau-gated potentiation with a seconds-long eligibility kernel. It does not specify firing
rate, tonic drive or weight scale — the animal supplies those. We pick them by tuning, and tuning optimises
whatever the metric rewards. Ours rewarded increment CONCENTRATION, so four honest steps walked `circ_dW`
0.2474 → 0.7050 while walking AWAY from place-specificity (shuffle ratio 1.01, p=0.42).

gap#4 is the same shape: the sparse operating point is biologically correct, and at E≈0.04 the credit term
φ'(E)=E(1−E) vanishes **~1600× over depth**. Real neurons are sparse AND learn, so biology solves this. We
implemented the rule without whatever holds the regime.

## 4. The protocol is part of the mechanism, and nobody writes it down

BTSP is one-shot: one plateau, one field. We ran five laps, which re-potentiates every position and erases the
field — measured, ratio 4.40 → 2.57 → 1.11 for laps 1 → 2 → 5. Bittner never says "do not run five laps" because
no experimenter would. **The constraint lived in the experimental design, not the results section**, and code
inherits the mechanism while silently choosing its own protocol. This is what
`research/biology/<id>.md`'s `constraints_config` exists to capture (`laps: 1`, with its reason, checkable).

## 5. And usually we cannot tell WHICH of the above is happening

The multiplier. Half of 2026-07-31 went into building instruments — raw ON/OFF weights, the sign budget, the
position-shuffled permutation null — and **every one showed the problem was somewhere other than where the record
said**. Lane D was "weight collapse"; it is common-mode convergence. gap#5's tuned point was "the best field"; it
is a clamp artifact. The crux was "needs more epochs"; more epochs was already a banked NO-GO.

**Most walls here were not the biology being hard. They were measuring the wrong quantity and believing it.**

## 6. The operational consequence

Under the no-defer law, a wall usually means a **missing companion process**, not a missing mechanism. The first
question is therefore not *"what biology surpasses this?"* but:

> **"What else does the real system run alongside this, that we replaced with a constant?"**

Because the answer is nearly always a homeostatic or competitive process proxied by a bound.

Second consequence, less comfortable: **the instrument is part of the emulation.** A mechanism you cannot measure
correctly is one you will tune in the wrong direction, confidently, for weeks.
