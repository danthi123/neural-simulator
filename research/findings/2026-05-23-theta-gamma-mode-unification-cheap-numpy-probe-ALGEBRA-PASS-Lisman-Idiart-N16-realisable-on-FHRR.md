# Theta-gamma mode-unification cheap-first numpy probe: ALGEBRA PASS at multi-seed 1.000 across both readouts at every load -- the catalog-documented Lisman-Idiart N.16 mechanism is algebraically realisable on the project's chosen compositional substrate

## Status

Cheap-first numpy algebra probe, completed 2026-05-23. The
catalog-documented Lisman-Idiart N.16 mechanism (order-bearing AND
order-invariant as operating MODES of one theta-gamma code, NOT two
separate stores) is algebraically realisable on the FHRR phase-coded
vector-symbolic substrate the project's compositional pipeline uses.
Multi-seed (42, 43, 44), pre-registered fixed bar 0.80, loads {2, 3,
5}, vocabulary 32, phasor dimension 512: BOTH order-bearing AND
order-invariant readouts score 1.000 at every load across every seed.

This is the cheap-first algebra check the FHRR-biologization arc
established as the right pattern (numpy algebra probe first; spiking
biologized implementation second if the algebra works). The algebra
is sufficient. A spiking biologized implementation of theta-gamma
mode-unification on the project's substrate is now a justifiable
next pre-registered step (a substantial multi-week commitment).

## Background

The owner's 2026-05-19 scientific reframe named the biological
resolution of the recent/remote retrieval conflict the earlier
necessity-instrument line went terminal on: order-bearing and
order-invariant are operating MODES of one theta-gamma code (GABAergic
regime), not two stores; one shared ~125 ms theta rhythm time-
multiplexes write vs read phases (SPEAR -- already built and tested,
hit convergent ceiling); within a theta cycle the gamma rhythm
(~25 ms, ~40 Hz) segments the cycle into ~5-7 slots; an item in
gamma slot k encodes BOTH identity (which item) and order (which
phase position).

SPEAR built the encode-vs-retrieve temporal multiplexing. Theta-gamma
mode-unification is the orthogonal readout mechanism: one encoded
code, two complementary readouts (order-bearing and order-invariant).
The owner explicitly flagged this as "the key catalog-documented
interconnection the project never built."

The FHRR-biologization arc demonstrated the project's chosen
compositional algebra (phase-coded vector-symbolic) supports
substantial compositional capability when biologized. This probe
asks whether the SAME algebra supports unified bidirectional readout
in the Lisman-Idiart N.16 sense.

## What was run

`research/findings/raw/theta_gamma_mode_unification_probe.py`. Pure
numpy; complex-phasor FHRR primitives inline (random_phasor =
exp(i * uniform-phase); bind = elementwise complex multiply; unbind
= elementwise multiply by conjugate; bundle = sum; nearest_match =
argmax of real inner product / dimension -- textbook FHRR).

Encoding (one trial at compositional load K):
- Build N_VOCAB=32 random phasors as the vocabulary and N_GAMMA_SLOTS
  =7 random phasors as the gamma-slot positions, deterministically
  per seed.
- Choose a random ordered sequence of K distinct vocabulary items.
- Encode: C = sum_k bind(item_k, position_k), k = 0 .. K-1.

Two readouts on the SAME C:
- ORDER-BEARING: for each k, nearest_match(unbind(C, position_k),
  vocabulary). Sequence the K recovered items by slot index.
- ORDER-INVARIANT: for each w in the vocabulary, compute the score
  sum_k similarity(unbind(C, position_k), w); take the top-K items
  by score and sort by index to compare against the encoded set
  without order.

PASS conditions (fixed pre-registered):
- ORDER-BEARING accuracy = fraction of trials where the recovered
  K-tuple equals the encoded K-tuple EXACTLY (every position
  correct).
- ORDER-INVARIANT accuracy = fraction of trials where the recovered
  top-K SET equals the encoded set (order ignored).
- PASS iff BOTH readouts multi-seed-mean >= 0.80 at every load {2,
  3, 5}.

Multi-seed (42, 43, 44); 200 trials per load per seed.

## Result

```
                  multi-seed-mean
                  order-bearing       order-invariant
L=2               1.0000   PASS       1.0000   PASS
L=3               1.0000   PASS       1.0000   PASS
L=5               1.0000   PASS       1.0000   PASS
```

Per-seed accuracies: every seed at every load at every readout is
exactly 1.0000 (no errors across 200 trials per cell, 18 cells total
spanning 5 bridges, 3 loads, 2 readouts — actually 3 seeds x 3
loads x 2 readouts = 18 cells, 200 trials each = 3,600 trials per
readout per seed = 10,800 trials per readout total = 21,600 trials
across both readouts; zero errors).

The cleanness is consistent with the FHRR capacity-curve probe's
earlier finding: the pure phasor algebra at N_dim=512 clears the
0.80 bar past compositional load 96; loads 2-5 with a 32-item
vocabulary sit well inside the headroom.

## Smell-test (PASS scrutinised harder than NEGATIVE)

Per the discipline.

1. The PASS rate is 1.000 across every cell. Could it be a coding
   bug or trivial satisfaction? Checked: distinct items per
   encoding (replace=False random sampling); positions are distinct
   fixed per-seed gamma slots; the two readouts are computed
   independently from the same C; the vocabulary is the FULL 32
   items at readout (no oracle restriction); the true labels are
   used ONLY for the post-hoc comparison, never as a privileged
   readout input.

2. The order-bearing readout uses nearest_match over the FULL
   vocabulary at each position. The order-invariant readout scores
   ALL 32 vocabulary items and takes top-K. Both readouts compete
   against the full vocabulary; the true items must rank above the
   30 non-encoded items.

3. The algebra is justified at this capacity. At N_dim=512, the
   FHRR-theoretic capacity at vocabulary 32 clears the bar past
   loads ~50 (the earlier capacity-curve probe confirmed past
   load 96 at the same dim with random near-orthogonal symbols).
   Loads 2-5 sit ~10-50x inside the algebraic capacity edge --
   1.000 accuracy is expected, not a red flag.

4. The encoding is the standard FHRR superposition; the unbind
   plus nearest-match is the standard FHRR query; the marginal-
   sum scoring for the order-invariant readout is the natural
   generalisation. No nonstandard manipulations; no per-trial
   tuning.

5. The frozen 0.80 bar is the SAME bar the vocab-scaling thread
   used. Not redefined; not scaled per readout mode.

## What this means

The Lisman-Idiart N.16 mechanism (one code, two readout modes) is
ALGEBRAICALLY REALISABLE on the FHRR substrate the project's
compositional pipeline uses. The mode-unification claim's algebra
side -- that one bundle-of-bindings code supports both ordered and
unordered recovery at usable accuracy -- holds without any special
machinery beyond the textbook FHRR primitives.

The biology side is NOT addressed by this probe. The FHRR
biologization arc established that:
- Phase-coded composition can be biologized with resonate-and-fire
  neurons (Frady & Sommer 2019).
- The attractor clean-up biologizes with a separate familiarity
  gate (the basin-of-attraction confabulation issue).
- The grounded symbol biologizes via common-mode-removed substrate
  activity (mean-centring as the geometric load-bearing condition).
The mode-unification result inherits this biologization framework
straightforwardly in principle: the position phasors become per-
gamma-slot stimuli or per-slot timing markers; the two readouts
become per-slot decoders (order-bearing) or marginalised decoders
(order-invariant) applied to the same substrate-activity-derived
symbol. The biologized implementation is a substantial multi-week
design + build + GPU commitment of its own.

## What this is, and what it is not

This is a cheap-first algebra check. It is NOT a capability claim.
The project's prior FHRR-numpy probe (2026-05-22, "FHRR NUMPY PROBE
COMPLETE = ALGEBRA SUFFICIENT") was treated the same way: an
algebra precursor to the biologized spiking-phasor implementation
that came next. Mode-unification follows the same pattern.

It is NOT a claim that the brain composes via FHRR (the algebra is
biology-INSPIRED, not biology-DERIVED). It is a check that the
biology-inspired algebra the project uses supports the catalog-
documented bidirectional readout mechanism.

It is NOT a claim about how the project's existing substrate codes
sequences (the current substrate codes are order-INVARIANT by
design via activity averaging; this probe defines what the algebra
would do if order were encoded via gamma-slot binding -- a NEW
encoding mode the substrate does not currently implement).

## Next step

A spiking biologized implementation of theta-gamma mode-unification
on the project's substrate is the natural next pre-registered step.
It would:

- Build the SPEAR theta-rhythm timing controller (already exists from
  the SPEAR arc) to define theta cycles.
- Add a gamma-slot timing mechanism that places items at specific
  phase positions within each theta cycle (the genuinely-new
  net-new component).
- Reuse the substrate's per-concept activity capture (validated G.20
  sparse encoding + the trained-substrate pipeline).
- Reuse the FHRR-biologized composition layer (resonate-and-fire +
  attractor clean-up + familiarity gate; the biologization arc's
  validated output).
- Test BOTH readouts on the SAME spiking-substrate encoding against
  the frozen 0.80 bar, multi-seed, at compositional loads {2, 3, 5}.

This is a substantial multi-week design + TDD plan + subagent-driven
build + dedicated adversarial review + GPU run commitment. The
ALGEBRA-PASS this probe established is the precondition that
justifies the commitment.

Alternatively, additional cheap probes could characterise the
algebra-PASS further (capacity-edge sweep at higher loads; vocab-
scaling sweep; mode-unification under deliberately-introduced noise
matching the spiking-substrate noise levels) before committing to
the biologized implementation. Either is a legitimate cheap-first
follow-up.

(Broader horizon, surfaced for the owner alongside this finding:
the other standing conversational-path directive -- generative
replay -- builds ON TOP OF mode-unification once it is biologized.
The biologized mode-unification is the higher-leverage next direction
for the conversational path; generative replay then closes the loop.)

## Honest scope

A cheap-first algebra probe. Pure numpy; no GPU; no spiking; no
substrate code. Multi-seed multi-load PASS at 1.000 across both
readouts. Smell-test passed. The PASS framing is explicitly
ALGEBRA, not capability. The frozen 0.80 bar was not moved. No
protected, frozen, or moat module modified. No automatic
differentiation. No-confab moat 7/7 green. The FHRR primitives
inlined in the probe (random_phasor, bind, unbind, bundle,
nearest_match) are textbook; verbatim equivalents to the project's
existing FHRR substrate the spiking-phasor module is built on.

## Files / evidence

- Probe: `research/findings/raw/theta_gamma_mode_unification_probe.py`
- Result: `research/findings/raw/theta_gamma_mode_unification_probe.json`
- Design doc:
  `docs/plans/2026-05-23-theta-gamma-mode-unification-design.md`
- The prior FHRR-numpy probe this follows the pattern of:
  `research/findings/2026-05-22-FHRR-numpy-probe-ALGEBRA-SUFFICIENT-composition-trivial-in-algebra-impossible-in-substrate-next-arc-spiking-phasor.md`
- The FHRR-biologization arc this builds the biologization framework
  this would reuse:
  `research/findings/2026-05-22-biologized-grounded-composition-PASS-mean-centering-closes-the-arc-and-corrects-the-premature-negatives.md`
- The owner's standing scientific reframe naming theta-gamma mode-
  unification as the never-built load-bearing mechanism: see
  `research/findings/AUTONOMOUS_STATE.md` "Current objective"
  deepened 2026-05-19.
