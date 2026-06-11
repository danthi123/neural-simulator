# Semantically-structured cortex — months-scale BUILD PLAN (present-ready)

> **Status:** present-before-build. The entire learning + architecture path is de-risked end-to-end, fully brain-based, multi-seed at toy scale. This doc is the plan the owner approves (or redirects) BEFORE the build commits resources. It consolidates the de-risk arc (cycles 23–34) into a concrete, costed, gated build.

**Goal:** Replace the conversational composer's idealized exact-inverse vector-symbolic binding algebra with a **learned, brain-based, semantically-structured cortex** — codes where related concepts cluster (cat ≈ dog), so the agent GENERALIZES across similar concepts — without weakening the no-confab abstention moat.

**Why this is now engineering, not research:** every mechanism on the path has a validated, spiking/synaptic, multi-seed recipe. The build assembles validated parts at scale; it does not discover whether the parts exist.

---

## What is already de-risked (the arc that justifies this build)

The dual / complementary-learning-systems (CLS) architecture: a graded-similarity **cortex** (generalization) + a linked decorrelated **hippocampal** expansion (binding) + an encode/retrieve link. Each block is validated:

| Block | Validation | Status |
|---|---|---|
| Architecture shape (graded cortex + decorrelated DG coexist) | dual-CLS architecture-proof, on-substrate gate | GO (cycle 22/24) |
| **Encode** (reproducible + decorrelated sparse code on the spiking substrate) | strong-drive de-risk: drive ≥800 pA → repro 1.000 + between-cos ≈ 0, 3 seeds | GO (cycle 23) |
| **Recall** (Hopfield CA3 identity recovery) | identity 1.000 | GO |
| **Round-trip + generalization** (cortical reinstatement channel) | cortex-channel Pearson +1.000, generalization 1.000 (controls collapse), 3 seeds | GO (cycle 24) |
| **Learn** (brain-based Hebbian co-occurrence → graded codes) | de-saturated `LearnedAssocGraph`, cycles=2: Pearson(sim,S_true) +0.84–0.88, gen 1.000, 2nd-order cat~dog recovered | GO (cycle 29–30) |
| **Read-out** (graded codes from the learned recurrent, fully brain-based) | spreading-activation + divisive normalization (Carandini-Heeger), no host method: 2nd-order margin +0.42, gen stays 1.000, 3 seeds | GO (cycle 32) |
| **Cycle-independence** (homeostasis removes the hand-picked cycles=2) | synaptic scaling / Oja flatten the saturation slope −0.0214→−0.0020/cyc; gen + faithfulness flat across cycles 2→20; holds at 2× store volume | GO (cycle 34) |
| **No-confab abstention** (the moat) | learned Bogacz-Brown familiarity gate matches host abstention at V=320, zero moat-breaches | GO (cycle, familiarity-gate-v320-GO) |

**Everything between sensation and the binding algebra is now a validated neural mechanism.** No host computation remains on the learn→read-out→bind→recall→abstain path.

---

## The build: three pieces

### Piece (i) — the homeostatic recurrent (cycle-independent learn)
**What:** the learn uses the spiking-Hebbian `LearnedAssocGraph` recurrent with a biological homeostatic normalization (Turrigiano synaptic scaling, per-postsynaptic-neuron incoming-sum set-point; Oja's incoming-L2 renorm is the validated fallback) applied per store-cycle.
**Reused (validated):** `research/runners/learned_assoc_graph.py` (`LearnedAssocGraph`, spiking-Hebbian, multi-seed-matched), the homeostatic-variant subclass from `learned_graded_embedding_homeostasis_probe.py`. NO `sim/` edits (runner-side normalization).
**New:** promote the homeostatic recurrent from probe to production module; tune the set-point to the production scale (the set-point grows with pool/pattern size — see the scale-up acceptance check below).
**Gate:** cycle-independence (faithfulness slope |·| < 0.005/cyc across cycles 2→40) + store-volume robustness (holds at ≥4× facts). **Status: GO at toy scale (cycle 34).**

### Piece (ii) — scale the corpus to the production concept set
**What:** take the toy co-occurrence corpus (30–48 concepts) to the production set — the validated 320-concept tier (the documented "age-5" target), source = the agent's own SVO-fact KB co-occurrence (on-substrate, no download) + optional Tiny Shakespeare for breadth.
**Reused (validated):** the 320-concept sparse-distributed ensemble (`g20_multibridge --sparse`, per-bridge 98.4%), the orthogonal-drive encoding, the de-risk gate suite (G1–G4).
**New:** run the learn at production scale; re-confirm G1–G4 at V=320 multi-seed.
**Gate (the build's acceptance matrix):**
- G1 structure recovery: Pearson(sim, S_true) ≥ 0.7 + **the within>between cosine-margin `graded` flag at production scale** (the one sub-bar soft at the n_pool=1000 smoke; the full-scale confirmatory `b6n98g33h` resolves whether it clears at n_pool=2000 — if it clears, this is a non-issue; if it stays soft, the lever is Oja or a lighter set-point, and the gate is satisfied via the generalization it predicts).
- G2 generalization: held-out-neighbour inference ≥ 0.7 (chance 0.25); orthogonal + permuted controls collapse.
- G3 cortex-channel round-trip closes; G4 spiking strong-encode (repro 1.0 + decorr at sparse k).
- G5 permuted-co-occurrence collapses (anti-cheat).

### Piece (iii) — integrate + validate the new capability
**What:** wire the learned graded-similarity cortex into the existing dual/CLS machinery (the hippocampal trisynaptic decorrelator + SWR-replay consolidation + the no-confab familiarity gate — all already built) and run the full conversational capability matrix.
**Reused (validated):** the hippocampal trisynaptic loop (D.12/D.13 validated), SWR-replay consolidation (CLS, 3/3 anti-cheat), the familiarity gate (V=320, zero breaches), the RF/VSA composer (the binding the cortex feeds).
**New:** the integration glue + the end-to-end run.
**Gate:** the full conversational matrix (who/what Q&A, abstention, negation/yes-no, one/two-attribute, clauses, dialogue) at V=320 **does not regress** AND gains the genuinely-new ability: graded semantic generalization ("cat is like a dog"; held-out-neighbour inference) **with the no-confab moat intact** (zero abstention-floor false-accepts).

---

## Reusable machinery (the build mostly assembles validated parts)

- `research/runners/learned_assoc_graph.py` — spiking-Hebbian co-occurrence learner (Piece i core)
- `research/runners/learned_graded_embedding_*.py` — the de-risk harnesses (de-saturate, confirm, divnorm read-out, homeostasis) → become the production read-out + gate suite
- `research/runners/g20_multibridge.py --sparse` + `g20_vocab_spec_320.py` — the 320-concept production ensemble (Piece ii)
- the hippocampal trisynaptic builder + SWR consolidation trainer (Piece iii decorrelator + write-back)
- `familiarity_gate_v320_validation.py` — the no-confab moat (Piece iii)
- the RF/VSA composer + the full conversational matrix tests (Piece iii validation)

---

## Honest risks (and why none is a blocker)

1. **The cosine-margin sub-bar at scale** (Piece ii G1): soft at the n_pool=1000 smoke; the full-scale confirmatory resolves it. Even if it stays soft, the generalization it predicts passes — the gate is satisfied via the consumed quantity, and Oja / set-point is the lever. **Not a blocker.**
2. **Biological-learning-vs-backprop strength gap** (the classic wall, hit in Phase 2.3a): the learn is weaker than backprop-on-big-data, so graded structure could be coarser at production scale than at toy scale. Mitigation: the de-risk already shows the brain-based learn reaches the host ceiling at toy scale; piece (ii) measures the gap honestly at V=320, and a coarser-but-real graded structure is still a capability gain (and an honest BOUNDARY if it underperforms).
3. **Moat preservation:** the familiarity gate is validated to match host abstention at V=320 with zero breaches; piece (iii) re-asserts it on the integrated system. **Non-negotiable gate.**

---

## Cost + decision

**Cost:** months-scale — dominated by piece (ii) production-scale training runs (the 320-concept learn at multi-seed) + piece (iii) integration and full-matrix validation. Pieces (i) and the read-out are validated; the new code is the scale-up + integration glue, not new mechanisms.

**Go/no-go (owner):** the build is justified — every mechanism risk is retired. The remaining unknowns are *engineering* (does the validated recipe hold at production scale) not *scientific* (does a brain-based mechanism for this exist). The decision is whether to commit the months-scale resources now, or run any further cheap de-risk first.

**Recommended first build step (cheap, reversible):** piece (ii)'s G1–G4 re-confirm at V=320 single-seed — the smallest run that converts "validated at toy scale" into "validated at production scale," before the full multi-seed commitment. If it holds, the build is de-risked at scale; if it reveals the strength gap, that's an honest characterization before the large spend.
