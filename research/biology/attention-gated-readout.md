---
type: biology
id: attention-gated-readout
mechanism: A per-class, per-trial biased-competition gate on the readout -- a top-down attentional template (the class's own learned discriminant weight magnitude) combines multiplicatively with the bottom-up stimulus drive, then a winner-take-all competition zeroes the losing units before the excitatory/inhibitory sign-split read, so which conjunction units a class population's decision listens to varies both by CLASS (top-down template) and by TRIAL (which units the current stimulus actually drove), instead of every trial reading a single fixed linear combination over the whole population.
status: de-risking
last_verified: 2026-09-09
current_finding: research/findings/2026-09-09-vision-configural-binding-attention-gated-readout-NEXT-MECHANISM-PREREGISTERED.md
current_status: "BUILT (--readout attention-gated in _vision_lindiscrim_readout_derisk.py), byte-identical-off proven at --attn-kwta-frac 1.0, GO gate pre-registered BEFORE the decisive run. Decisive 6-seed run LANDED a REGRESSION versus the --conj-select competitive baseline it is stacked on: PARTIAL-beat2/6-lb3/6 (vs the baseline's beat4/6-lb6/6), not a task GO. Per-seed diagnosis against the ungated linear score on the identical trained discriminant + spike code: the hard k-WTA gate (--attn-kwta-frac 0.5) actively destroys accuracy in 4/6 seeds (RATE_lin_ceiling_held is bit-identical to the baseline, ruling out a front-end/bank confound) -- the lane's C2 code is a fine DISTRIBUTED cosine modulation, and hard elimination of half the population per trial discards exactly the kind of small broadly-distributed contribution that code is made of. BANKED at this operating point/functional form (hard k-WTA); the named next rung is a GRADED/soft attention gain (multiplicative reweighting, no hard elimination -- the Reynolds & Heeger 2009 normalization-model-of-attention form this file's own satdiv machinery already implements one level up for a different population), not a --attn-kwta-frac sweep of the same hard-competition form."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "a second kind of attention, feature attention"
    note: "EXTERNAL. Kandel PNS-6e names FEATURE-BASED attention as distinct from spatial attention -- attending to a FEATURE (a color, an item identity) rather than a retinal location: 'a second kind of attention, feature attention: In your search, you ignore [task-irrelevant items] and attend only to [task-relevant ones].' This mechanism's per-class top-down template (A_c = normalized |w_c|, the class's own learned discriminative weight magnitude over CONJUNCTION UNITS instead of colors) is the same structure -- attention selects FOR a feature dimension, here 'which conjunction units this class finds informative,' not a spatial window."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Another top-down influence is perceptual task"
    note: "EXTERNAL. Grounds the TOP-DOWN (task/goal-driven, not purely stimulus-driven) half of the combination rule: which inputs a downstream population processes is modulated by what the current task/goal is, not fixed by the stimulus alone. Here the 'task' is 'decide class c,' and the top-down template is that class's own already-learned discriminative weight vector -- an already-available signal (no new learning), not a fresh top-down channel."
  - path: research/runners/_vision_lindiscrim_readout_derisk.py
    anchor: "Foldiak 1991 / Kohonen 1982-style winner-relative competitive-learning gate"
    note: "LOCAL. The SAME competitive-learning primitive already established TWICE in this file -- _bcm_learn_s2_templates's competitive_frac (restricts a WEIGHT UPDATE to the top-k-by-current-drive templates) and _select_conjunctions_competitive's per-presentation k-WTA (restricts which CANDIDATE UNITS survive into the bank). This entry's mechanism (_attention_gated_class_read) reuses the IDENTICAL top-k-by-current-drive competition rule a THIRD time, now at READ time: which of the bank's EXISTING units a class's decision listens to on a given trial, rather than which units get a weight update or which units get to exist at all -- the same winner-relative-inhibition arithmetic, moved one more step downstream in the pipeline."
  - path: research/biology/conjunction-competitive-selection.md
    anchor: "population SELF-SELECTS which conjunctions are informative"
    note: "LOCAL. The sibling mechanism this entry is deliberately orthogonal to: conjunction-competitive-selection decides which conjunction units EXIST in the bank (a one-time, training-time, population-wide decision shared by every class); this entry decides which of those EXISTING units a given class's decision listens to on a given trial (a per-class, per-trial, inference-time decision). Both reuse the identical k-WTA primitive at two different points in the pipeline -- structure vs. read-time attention, the same distinction Desimone & Duncan's biased-competition model draws between which neurons populate a representation and which of them win the competition for read-out on a given presentation (Desimone & Duncan 1995, Annu Rev Neurosci 18:193, not locally anchored -- summarized, not quoted)."
implemented_by:
  - research/runners/_vision_lindiscrim_readout_derisk.py
findings:
  - research/findings/2026-09-09-vision-configural-binding-attention-gated-readout-NEXT-MECHANISM-PREREGISTERED.md
---

# The class population's decision does not read the whole bank the same way on every trial

**The wall this answers.** Every readout lever in this lane so far (`_train_linreadout` + `_spiking_class_read`)
has been a SINGLE fixed signed-linear combination `V` over the WHOLE conjunction-unit population, applied
identically to every trial and every class: `net_c = w_c . r + const_c`, with `w_c` frozen after training. The
conjunction-bank levers (`--conj-order triple`, `--conj-select competitive`) all changed which features exist for
that fixed readout to combine; the readout combination rule itself never varied. The competitive-selection sweep's
own closing line named the untried orthogonal axis: reweight which units the READ listens to, per class, per
trial -- not which units exist.

**The mechanism.** For each class `c`, a TOP-DOWN attentional template `A_c = |w_c| / mean(|w_c|)` is read
straight off the ALREADY-FITTED discriminant `w_c` (zero new learning -- this is a pure read-time gate on top of
the same `V`/`b`/`mu`/`sd` every other arm already produces). Per trial, the bottom-up stimulus drive `r` is
combined multiplicatively with this template (`bd = r * A_c`), and a k-WTA competition keeps only the top
`--attn-kwta-frac` fraction of units by `bd`, zeroing the rest. The SURVIVORS' raw (not attention-reweighted)
drive, gain-renormalized by the realized surviving fraction, then goes through the exact same Dale's-law
excitatory-minus-inhibitory sign-split (`w = w+ - w-`) every other arm in this file uses, and the same LIF
class-population port + spiking WTA downstream.

**Why this is not a new primitive.** The k-WTA competitive-gate arithmetic is IDENTICAL to what
`_bcm_learn_s2_templates` (competitive_frac, gating a weight update) and `_select_conjunctions_competitive`
(per-presentation k-WTA, gating which candidates survive into the bank) already established in this file -- only
WHERE it is applied is new: at read time, per class, per trial, instead of at weight-update time or bank-selection
time. The top-down template itself is not a new learned quantity either -- it is read directly off the weight
vector every prior arm in this file already fits.

**Why this could plausibly help.** A single fixed linear combination must find ONE compromise weighting that
works across every trial; a class-specific per-trial gate lets the SAME shared conjunction bank serve every class
differently (class A's decision can lean on a different subset of units than class B's) and lets each trial's
decision ignore whichever units this particular trial's noise/nuisance variation happened to drive spuriously,
provided that noise is not correlated with the class's own top-down template. The honest risk: the template is
derived from the SAME weight vector the readout also uses, so a degenerate case (the gate simply reproduces
whatever the linear readout would have done anyway) is possible and is exactly why `--attn-kwta-frac >= 1.0`
(the no-gating limit) is required to reproduce `--readout linear` byte-for-byte -- the gate's OWN causal
contribution is isolated by how much attn_kwta_frac < 1 changes the result versus that limit, not merely by
`attention-gated` beating `linear`'s absence.

**The honesty boundary.** The top-down template is computed from held-out-blind training data only (it is a
deterministic function of `w_c`, itself fit on `tr_c1`/`tr_cls` only, never held/scrambled data) -- no new
leakage surface versus every prior arm in this file. `--attn-kwta-frac` is an untuned de-risk operating-point
knob (0.5, a single principled default -- half the population competes down to zero per trial), not a
biology-REQUIRED constant; no `constraints_config` is bound, matching this file's other competitive-gate entries.
