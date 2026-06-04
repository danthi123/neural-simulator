# Phasor FHRR is a unified-substrate candidate: 320-concept diversity AND nesting — 2026-06-03

**One line:** The phasor FHRR substrate that the Direction-A nesting work lives on *also* holds
production-scale diversity (320 concepts) with 3-role SVO composition at 1.00 (even at D=1024). So the
substrate split — production diversity on real-Hadamard vs. research nesting on phasor — is **not
necessary**: one phasor FHRR substrate can carry both. This de-risks the capacity question behind a
possible substrate unification.

## The strategic context

The project's validated conversational results split across two substrates:

- **Production diversity** (160/320-concept G.20 sparse, multi-tag retrieval, ~88–93% cross-cue) lives
  on a **real-valued / dense-Hadamard-style** binding. That binding is **non-invertible**, so it
  **cannot nest** — the hierarchical-320 attempt scored 0.000 on structured facts and was retracted;
  the flat-distinct workaround is single-binding-only.
- **Nesting** (this session's Direction A: resonator decoder, multi-modifier, recursive clause, all
  validated in algebra *and* genuine resonate-and-fire spikes) lives on **phasor FHRR**, where binding
  is invertible.

The open question that decides whether these unify: does phasor FHRR also carry production-scale
**diversity** (hundreds of concepts) with composition, or does its capacity bend before 320?

## Result — phasor FHRR carries 320-concept diversity + composition with large headroom

All numbers are random-phasor codes (the standard VSA assumption; the same well-conditioned codes the
resonator probes used), numpy phasor algebra (the fast realization of the validated resonate-and-fire ops).

**320 concepts + a 3-role SVO fact (subject/verb/object), decode each role over all 320:**

| D | per-role decode | full-SVO |
|---|---|---|
| 1024 | 1.00 / 1.00 / 1.00 | **1.00** |
| 2048 | 1.00 | 1.00 |
| 4096 | 1.00 | 1.00 |
| 8192 | 1.00 | 1.00 |

No capacity problem at all — a single SVO fact over a 320-word vocabulary decodes perfectly at D=1024.

**Single-bundle capacity (how many distinct role-bindings one bundle holds), D=1024, 320 vocab:**

| K role-bindings | per-role | full |
|---|---|---|
| 3–24 | 1.000 | 1.00 |
| 32 | 0.999 | 0.97 |
| 48 | 0.986 | 0.50 |
| 64 | 0.930 | 0.00 |
| 96 | 0.734 | 0.00 |

A single bundle holds ~24–32 role-bindings cleanly at D=1024 — **8–10× the 3 roles of an SVO fact**. The
bend at K≈48 and break at K≈64 are the expected VSA bundle-capacity curve (cleanup SNR ~1/√(K−1)); they
move up with D. The production unit (one fact per bundle, a knowledge base = a list of bundles, exactly as
the nesting agent's `self.kb` works) sits deep inside the safe region.

## Grounded-proxy: robust to common-mode inter-code correlation (the easy kind)

Grounded codes are not orthogonal. Injecting a shared component into all 320 codes (D=1024) to raise their
mean pairwise |cosine|:

| shared-component α | mean pairwise \|cos\| | full-SVO |
|---|---|---|
| 0.0 | 0.028 | 1.00 |
| 0.3 | 0.035 | 1.00 |
| 0.6 | 0.102 (≈ production overlap) | 1.00 |
| 1.0 | 0.404 | 1.00 |
| 1.5 | 0.768 | 1.00 |

SVO decode holds at 1.00 even at mean cosine 0.768. A single shared component is *common-mode* correlation
— it shifts every cleanup candidate equally, so it does not move the argmax. The harder, more grounded-faithful
case is *clustered* correlation (semantically related concepts forming mutually-similar subsets):

**Clustered correlation — 16 clusters × 20 concepts, the HARDEST case (S, V, O all drawn from the same
cluster), D=1024:**

| per-cluster component β | within-cluster mean \|cos\| | full-SVO (same-cluster) |
|---|---|---|
| 0.0 | 0.026 | 1.00 |
| 0.5 | 0.067 | 1.00 |
| 1.0 | 0.386 | 1.00 |
| 2.0 | 0.829 | 1.00 |

Still 1.00 even when within-cluster concepts have cosine 0.829 and all three roles are filled from the same
cluster. The reason: each code keeps a distinct per-concept random component on top of the shared cluster
component, and the argmax cleanup latches onto that distinguishing residual. The only thing that breaks this
is *degenerate* codes (two concepts with near-identical codes, cosine → 1.0). So the grounded-capacity
question is de-risked for both common-mode and clustered correlation, provided the grounded encoder gives
each concept a distinguishable code (which any non-degenerate sparse encoding does).

## The unified AGENT (not just the substrate) works at production-diversity scale

The `NestedCompositionAgent` was run at a 120-concept vocabulary (60 nouns + 30 verbs + 30 adjectives,
D=2048) with 40 facts of mixed kind (flat / one-attribute / two-attribute / embedded clause), querying each:

| seed | correct / 40 | abstain-on-unknown |
|---|---|---|
| 42 | 39/40 | ✓ (None) |
| 43 | 38/40 | ✓ |
| 44 | 39/40 | ✓ |

≈ 96% across mixed fact kinds at 120-concept diversity, with correct abstention on an unknown query. The
1–2 misses per seed are the known two-attribute / deeper-nesting boundary cases — consistent with the
per-kind robustness measured separately, not a scaling collapse. So the *agent* — nested composition + the
four auto-detected patient kinds + Q&A + abstention — holds up at production-diversity scale, not only the
underlying substrate algebra.

## The learning question — cheap-first analog RESOLVES

The production system does not construct representations algebraically; it *learns* a map from input
("word") to representation (concept activity) via Hebbian/STDP plasticity. The cheap-first phasor analog: a
single-pass Hebbian heteroassociator `W = (1/N) Σ_c code_c ⊗ cue_c^H` that learns to retrieve each concept's
phasor code from its input cue, then composes the *retrieved* codes.

| D | cue→code retrieval (320 concepts) | compose-from-learned-cues, full-SVO |
|---|---|---|
| 1024 | 1.000 | 1.00 |
| 2048 | 1.000 | 1.00 |
| 4096 | 1.000 | 1.00 |

A one-pass Hebbian rule learns the 320-concept input→code map at 1.000 retrieval (320 < D, within linear
associator capacity), and the *learned* representations compose into an SVO fact at 1.00. **Honest scope:**
this is a *linear* Hebbian associator, not a spiking-STDP network — it shows the map is learnable *in
principle* at this scale; the spiking-STDP realization is a further engineering step. But the cheap-first
learning analog passes, which is real evidence the learning side is tractable, not just the capacity side.

## What this means (and what it does NOT)

**Does mean:** phasor FHRR is a viable **unified-substrate candidate** — it has, in one representation,
(1) the 320-concept diversity the project achieved on real-Hadamard, (2) the multi-factor nesting that
real-Hadamard fundamentally cannot do, and (3) 3-role composition — all with invertible binding and a
validated spiking realization (resonate-and-fire). The capacity objection to unifying on phasor FHRR is
answered: there is no capacity wall at 320.

**Does NOT mean:** that the production system should be migrated today. Important honest caveats:

- These bindings are **algebraic** (constructed by phase arithmetic). The production system **learns**
  the input→representation map via STDP on a spiking network. The cheap-first learning analog (a one-pass
  Hebbian associator, above) passes at 320 scale, so the learning side is plausibly tractable — but the
  full **spiking-STDP** realization is still a real engineering step, not free. This finding de-risks the
  *capacity* question and the *linear-learning* analog; the spiking-STDP implementation remains open.
- The codes here are random phasors. The correlation sweeps (common-mode + clustered, above) cover the
  main ways production **grounded** codes depart from orthogonality, and both hold at 1.00. **Update
  (resolved):** a re-test with the *actual* grounded encoder (`vocab_to_drive_pattern`, real ~10% overlap)
  now also RESOLVES — see `2026-06-03-spiking-STDP-learns-phasor-map-RESOLVES-algorithmic.md` §"Grounded
  word codes". The grounded-code question is no longer open.
- The diversity test stores each fact as its own bundle (as the agent does). It is not a claim about
  superposing hundreds of facts in one vector (that bends at K≈48, as shown).

## Files

- Probes were inline (this note records the tables); they reuse the same phasor algebra validated in
  `_resonator_capacity_probe.py` + `_spiking_resonator_probe.py` (spiking fidelity) and
  `_recursive_clause_probe.py` (nesting).

## Verdict

**Phasor FHRR is a unified-substrate candidate.** Diversity (320 concepts), composition (3-role SVO at
1.00), and nesting (Direction A, spiking-validated) coexist on one invertible-binding substrate with
8–10× single-bundle headroom. The substrate split is not forced by capacity. Whether to unify the
production system onto it is a learning-and-engineering decision for the owner; this finding removes the
capacity uncertainty from that decision.
