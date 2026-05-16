# G1 Follow-up Branches — Pre-staged Design (executes on the Task-10 verdict)

> **For Claude:** Pre-staged per the autonomous-runs "pre-stage parallel
> branches" discipline so the autonomous run executes the instant the
> Task-10 verdict lands. Continuation of
> `2026-05-16-bidirectional-generative-conversational-agent-design.md`
> (this fleshes out its pre-registered "FAIL => add P" branch with the
> cheaper precursors the *diagnosed* failure mode specifically calls
> for). Same anti-cheat discipline: every branch is a pre-registered,
> permuted-control-gated, honest-negative-or-positive probe. NEVER
> config-crank a failed approach. On verdict: pick the branch, invoke
> superpowers:writing-plans for that branch, then
> superpowers:subagent-driven-development.

## The sharpened diagnosis (why this is more than a binary)

G1 probes Approach B (bare songbird controller over the existing
recognition-only G.20 substrate, RL-trained by self-comprehension).
The observed failure mode is precise, not generic: `mean_reward=0`
across all epochs because **gate-cleared productions decode
confidently but never to the intended first concept** — i.e. the
*integrated single-concept self-comprehension judge, reading the
final residual via argmax, does not distinguish intended order from
scrambled order* (corroborated pre-data by Step-0 calibration: AUC
only 0.775; encoded mean 73.2 barely over control-max 72.0). The
bottleneck is the **order-readout / order-discriminability of the
judge on this substrate**, not (necessarily) the controller idea.
That precision dictates the branch ordering below.

## Decision tree

### Verdict = FAIL (current trend) — falsify-cheaply order: G1.5 -> G1.6 -> P

#### Branch G1.5 — order-sensitive readout (CHEAP, do FIRST)

**Hypothesis:** the order signal exists in the substrate's *dynamics*
during the production but is discarded by taking `argmax` of the
*final* residual.

**Change (readout only, no architecture rewrite):** add an
order-sensitive self-comprehension variant in
`research/runners/song_g1_ignite.py` — instead of one argmax of the
post-sequence residual, decode the *trajectory*: accumulate per-step
concept-pattern firing across the production window and extract the
ordered sequence of transient peak concepts (the path through the
pool), via the already-validated stim-recall accumulation. Score that
ordered trajectory against the intended order with the unmodified
`song_g1_core.score_order`. New gate runner variant
`song_g1_gate.py --readout trajectory` reusing the SAME pre-registered
`g1_verdict`, the SAME sidecar-frozen control-calibrated floor
methodology (re-derive the floor for the trajectory-readout regime
via the SAME control-distribution AUC method — pre-registered,
control-calibrated, never tuned), and the SAME load-bearing
permuted-ORDER control.

**Pre-registered gate (unchanged bar):** trajectory-readout held-out
true-order beats permuted-ORDER control by >= 10% AND clears the
regime-calibrated floor AND >= 0.5 majority. PASS => controller is
viable with a better judge (large win, no P rewrite; proceed to G2
scale). FAIL => the order signal is genuinely absent from the
substrate dynamics, not just the readout => P is justified by
evidence.

**Scope:** small (a readout function + a gate flag + its
control-calibration); reuses `song_hvc`/`song_g1_core`/the trained
checkpoint. Days. Re-run the no-harm probe contract still holds
(write-only ignition unchanged).

#### Branch G1.6 — songbird developmental scaffolding (cold-start fix)

**Hypothesis (if G1.5 also FAILs on cold-start, not on
discriminability):** the failure is sparse-reward starvation — random
babble never once hits `c1` to bootstrap `reinforce` (DA-gated no-op
at reward 0 => W never moves). Real juveniles do NOT babble from
noise into a vacuum: memorized tutor template + innate predisposition
+ subsong->plastic-song developmental stages (Aronov & Fee 2011;
Doupe & Kuhl 1999).

**Change (`research/runners/song_g1_train.py`, curriculum only):** a
developmental schedule — early epochs strongly bias / teacher-force
the chain's first state toward the intended `c1` (the "tutor
template" = the grounded proposition the agent already stores), then
*fade* the scaffold to zero over a pre-registered schedule so the
final evaluated controller is unscaffolded. Self-contained (template
is internal; no external teacher at runtime). The Task-10 held-out
gate is run with the scaffold fully faded (an honest test of the
*learned* controller, not the scaffold).

**Pre-registered gate:** same Task-10 held-out + permuted-ORDER +
fixed bar, scaffold-faded. PASS => self-supervised songbird works
given biologically-faithful developmental scaffolding (real, honest;
proceed to G2). FAIL => cold-start is not the bottleneck => P.

**Anti-cheat note:** the scaffold MUST be fully faded before the
held-out eval and the fade schedule pre-registered before the run;
a residual scaffold at eval would be teacher-forcing the verdict
(forbidden). Permuted-ORDER control still load-bearing.

#### Branch P — predictive-coding top-down (deepest; the design's pre-registered FAIL branch)

Only after G1.5 + G1.6 cheaply exhaust themselves. Add Rao-Ballard
top-down generative + prediction-error pathways to the concept cortex
(its own design doc + plan at execution time): the comprehension
judge becomes a *generative model scoring sequence likelihood* (rich
order-sensitive error at every level, not a static argmax), and each
next ignition is shaped by a learned top-down generative prior
(Bastos 2012 canonical microcircuit; Friston active inference).
Largest scope; the biologically-correct generative architecture
regardless. Re-gate with the SAME pre-registered Task-10 protocol.
If P also FAILs: the honest, decision-relevant conclusion strengthens
to "self-contained generative sequence production is out of reach on
this substrate/hardware; the validated grounded-retrieval +
no-confabulation asset is the deliverable" — propagated honestly,
no spin (the Inc-1/2/3 discipline).

### Verdict = PASS (less likely given trend) — design's scoped G2 -> G3

- **G2:** multi-seed (>=3) G1 + held-out *novel compositional*
  propositions never babbled (generalization, not memorization — the
  Inc-3 held-out lesson) beating permuted-ORDER >= 10%.
- **Cross-bridge:** lift the single-bridge scoping to real
  cross-bridge propositions ("apple in-nouns IS big in-adj") via the
  validated cross-bridge engram path.
- **+P for grammaticality**, then **G3:** multi-turn generated
  conversation with the no-confabulation abstention moat intact +
  CLS no-forgetting check.

### Verdict = BOUNDARY (partial: some held-out pass, weak margin)

Project's established BOUNDARY handling: characterize per-proposition
honestly, multi-seed it, NO overclaim. Route by *which* propositions
showed signal: discriminability-limited subset -> G1.5; cold-start-
limited subset -> G1.6; pervasive weak -> P.

## Why every branch is progress toward conversation

Each branch is a pre-registered falsification that converges on the
architecture that can actually generate. FAIL is not a dead end: it
exhausts the cheap-controller hypothesis and routes us — by evidence,
cheapest-first — to the predictive generative model that is
independently the more biologically-correct substrate for language
production. This is the same systematic-falsification discipline that
produced the honest Inc-1/2/3 negatives and the validated G.20
memory; it is the path, not a detour from it.

## Scientific basis

Rao & Ballard 1999; Friston 2010; Bastos 2012 (predictive coding /
canonical microcircuit — Branch P). Hahnloser 2002; Fee & Goldberg
2011; Aronov & Fee 2011; Doupe & Kuhl 1999 (songbird sequence
production, developmental subsong->plastic-song scaffolding —
Branches G1.5/G1.6). Builds on the project's validated
Pulvermuller/Kanerva/Tonegawa/Marr-McClelland substrate.

## Reuse surface (DRY — already built, do not rebuild)

`sim/song_hvc.py`, `research/runners/song_g1_core.py` (g1_verdict /
score_order / permuted_order_controls — UNMODIFIED bars
_G1_MARGIN=0.10 / _G1_ABS_FLOOR=0.5), `song_g1_ignite.py`
(write-only ignition; G1.5 adds a trajectory-readout variant here),
`song_g1_train.py` (G1.6 adds a faded curriculum here),
`song_g1_gate.py` (sidecar-frozen-floor gate; G1.5 adds a
`--readout` variant), `song_g1_noharm_probe.py` (the no-harm
contract every branch must keep), `sim/train_checkpoint.py`,
the validated G.20 320-sparse bridges + abstention moat.
