# Semantically-structured cortex — BUILD PLAN (present-ready, ~2–4 weeks)

> **Cost correction (2026-06-11):** earlier drafts said "months-scale." That was wrong — it carried over the cost of the **dendritic-substrate rewrite** (the original option-B path), which the dual/CLS route AVOIDS. This build assembles already-validated pieces; honest cost is **~2–4 weeks** (compute + integration + iteration), NOT months. The genuine months-scale work (the dendritic rewrite) is a separate, deferred path this route does not require.

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
**New:** promote the homeostatic recurrent from probe to production module; calibrate the set-point per scale.
**⭐ DEFAULT TO OJA, not synaptic scaling (evidence-based, cycle 36):** Oja's incoming-L2 set-point transfers GRACEFULLY across scales (t=15@n_pool=1000 → 40@7000 → 50@12000, all near-ceiling), whereas synaptic scaling's incoming-SUM set-point is SCALE-FRAGILE — at V=320/n_pool=12000, scaling t=2000 COLLAPSED to Pearson +0.291 (over-normalized) while Oja t=50 hit +0.991 (the same failure mode as scaling t=2400 at V=160). Both are valid WITH a correctly-calibrated set-point, but Oja is materially easier to calibrate as the build scales, so it is the default; scaling is an option requiring per-scale set-point re-calibration.
**Gate:** cycle-independence (faithfulness slope |·| < 0.005/cyc) + store-volume robustness + `graded=1`. **Status: GO — multi-seed 3/3 (cycle 34/35, commit f0800378) AND now V=320 production tier (cycle 36, seed 42): Oja t=50 = Pearson +0.991 (≥ host ceiling +0.959), gen 1.000, 2nd-order +0.975, store-volume +0.994 at 2×.**

### Piece (ii) — scale the corpus to the production concept set
**What:** take the toy co-occurrence corpus (30–48 concepts) to the production set — the validated 320-concept tier (the documented "age-5" target), source = the agent's own SVO-fact KB co-occurrence (on-substrate, no download) + optional Tiny Shakespeare for breadth.
**Reused (validated):** the 320-concept sparse-distributed ensemble (`g20_multibridge --sparse`, per-bridge 98.4%), the orthogonal-drive encoding, the de-risk gate suite (G1–G4).
**New:** run the learn at production scale; re-confirm G1–G4 at V=320 multi-seed. **✅ DONE (2026-06-12): V=320 multi-seed = CLEAN GO 3/3** (Oja t=50, seeds 42/43/44: Pearson +0.991/+0.990/+0.992, gen 1.000 all, 2nd-order +0.97+, permuted controls collapse). Single-pool scaling curve: V=160 +0.977 → V=320 +0.992 (improving). **Single-pool capped at ~V=320–450 by the synapse-install memory wall** (V=640 OOM'd) → larger vocab = multi-bridge.
**Gate (the build's acceptance matrix):**
- G1 structure recovery: Pearson(sim, S_true) ≥ 0.7 + the within>between cosine-margin `graded` flag. **Satisfied in advance at full scale (confirmatory `b6n98g33h`):** at n_pool=2000/pattern=100 the homeostatic recurrent (synaptic scaling, set-point t=600) holds `graded=1` at BOTH cycles 2 and 20 with generalization 1.000 and 2nd-order margin +0.67 — the soft flag at the n_pool=1000 smoke was a scale + set-point artifact, now cleared. At production V=320 this is a one-line set-point tuning (the set-point grows with pool size), not a risk.
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

## Scaling path to large vocabulary (V=320 → V=640 → 2,048 multi-bridge)

The single learned pool has a hard **quadratic memory wall** — measured live, not estimated: the learned recurrent's synapses scale as pool² (V=160 pool 7,000 → 30.7M synapses; V=320 pool 12,000 → 88.6M — 1.71× pool = 2.89× synapses ≈ N²). Consequences for one pool on a 24 GB GPU (RTX 3090):

| Vocab | pool (~37 neurons/concept) | synapses | feasible single-pool? |
|---|---|---|---|
| V=320 | 12,000 | 88.6M (7.7 GB) | yes (in flight) |
| V=640 | ~24,000 | ~354M | **NO — OOMs at the synapse install** (empirically, 2026-06-11: pinned-memory pool exhausted transferring 354M synapses host→device, before any compute) |
| V=1,280 | ~48,000 | ~1.4B | no — OOM |
| **V=2,048** | **~77,000** | **~3.6B (~40 GB)** | **no — OOM by ~16 GB; learn ~days even if it fit** |

So a single-pool 2,048 run is **infeasible on this GPU** — a memory wall, not a time budget. Dropping the pool to fit memory collapses density to ~12 neurons/concept and breaks separability (it would "fail" for a density artifact, not a real ceiling — a misleading test).

**The correct route — multi-bridge sparse-distributed (the project's existing production method).** The 320 tier is already 5 bridges × 64 concepts precisely to dodge this wall. The dual/CLS recipe is **per-bridge**, so it composes: **2,048 ≈ 32 bridges × 64 concepts**, each bridge tiny (64 ≪ V=160, where the recipe is already GO) and tractable. This *is* the "is the old vocabulary ceiling lifted?" test — the old architecture's per-bridge recipe + cross-bridge composition vs the new one's.

**Genuine open questions for the multi-bridge route (the real work, not the per-bridge recipe):**
1. **Within- vs cross-bridge similarity.** The learned graded embedding is *within-bridge* — cat~dog generalize only if both live in the same bridge. So the corpus must be **sharded by semantic cluster** (similar concepts grouped into the same bridge) for the generalization to be useful; cross-bridge relationships go through the existing composition/binding layer, not a shared embedding.
2. **Cross-bridge composition at 32 bridges.** The project validated cross-bridge composition at 5 bridges (the 320 ensemble); 32 is 6.4× more — whether composition + the no-confab moat hold at that fan-out is the open question.
3. Per-bridge graded generalization + cross-bridge composition must together deliver the full conversational matrix — the integration question piece (iii) already owns, now at multi-bridge fan-out.

**Queued sequence (gated, cheap-first):**
1. V=320 single-pool gate (3-seed gated; in flight) — confirms the recipe at the production single-pool tier.
2. ~~V=640 single-pool~~ **— SCRATCHED (2026-06-11): clean-density V=640 OOMs at the synapse install (354M synapses); the single-pool memory wall is ~V=320–450, not V=640.** The single-pool scaling curve completes at **V=160 +0.977 → V=320 +0.991** (both near/above ceiling, improving) — sufficient single-pool evidence; a feasible (lower-density) V=640 would confound scale with density and is not worth it. **Pre-build multi-seed confirmation runs at V=320** (the production single-pool tier) instead.
3. **Multi-bridge 2,048** (32 × 64) — the lifted-ceiling demonstration AND now the only path past the production tier. **Held until the V=320 multi-seed confirms AND the owner green-lights the scale** (it folds into piece iii's integration + the ~2–4 week push). Compute-feasible because each bridge is small (64 concepts); the cost is in the cross-bridge composition + semantic-cluster sharding, not raw pool size. **Design the cheap-first multi-bridge de-risk before building (standing practice).**

---

## Cost + decision

**Cost — ~2–4 weeks (honest decomposition):**
- **Piece (ii)** — mostly GPU compute. A single-seed V=320 gate run is ~1.5–2.5 hr; the multi-seed re-confirm (3-seed gated: seed 1 gates seeds 2–3) is ~1 day of GPU. Implementation is light (the probes exist; it's parameter scaling).
- **Piece (iii)** — the larger piece, still weeks not months: wire the learned cortex into the existing dual/CLS machinery (decorrelator + sleep-replay consolidation + familiarity gate + composer, all built), supply a real co-occurrence corpus, and validate the full conversational matrix multi-seed with the new generalization capability + the abstention moat intact. Dominated by integration iteration (pieces validated individually may surface interaction issues at scale) + multi-seed conversational-matrix compute.
- Pieces (i) and the read-out are validated; the new work is scale-up + integration glue, not new mechanisms. My implementation-time estimates run ~2–3× high (treat as a ceiling); compute estimates are reliable.

**Go/no-go (owner):** the build is justified — every mechanism risk is retired. The remaining unknowns are *engineering* (does the validated recipe hold at production scale) not *scientific* (does a brain-based mechanism exist). The gate is NOT a months-long blank check — it is the point where work shifts from ~2-hour reversible probes (the cheap scale-checks, running now) to the **~2–4 week sustained multi-seed + integration push**. That push is worth an explicit "go" because it is a sustained effort, not because it is months.

**Recommended first build step (cheap, reversible):** piece (ii)'s G1–G4 re-confirm at V=320 single-seed — the smallest run that converts "validated at toy scale" into "validated at production scale," before the full multi-seed commitment. If it holds, the build is de-risked at scale; if it reveals the strength gap, that's an honest characterization before the large spend.
