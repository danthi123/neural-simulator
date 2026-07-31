---
type: plan
status: live
date: 2026-06-05
---

# Conversational-path cheat/shortcut conversion — full scope + phased plan — 2026-06-05

> Owner directive: "fully scope and plan out phases conversion of all remaining cheats/shortcuts that are
> non-biological and make resolving them the highest priority before moving forward." Backed by three deep-research
> passes (catalog `docs/biology.md` + Kandel 6e + literature): findings `2026-06-05-cheat-A-code-grounding-research.md`,
> `-cheat-BC-spiking-phasor-cleanup-memory-research.md`, `-cheat-D-associative-graph-research.md`.

## Scope — the audit (the FHRR-on-bridge conversational production path)

The production conversational agent (`BrainConversationalAgent` → `RFPhasorComposer` + the Hebbian parser + the dlPFC)
runs on the bridge; the COMPOSITION (bind/unbind/bundle) is genuine resonate-and-fire phasor spiking. **What is NOT
yet biological — four shortcuts**, of which two were CLEARED on the old rate composer and reintroduced by the RF recode:

| # | shortcut | where | the rate composer had it? |
|---|---|---|---|
| **A** | concept/role codes are RANDOM (`rng.uniform`), not grounded/learned | `rf_phasor_composer.py` 66-73 | both used given codes (denoise64 vs random) |
| **B** | cleanup is numpy phase-cosine `argmax` | `rf_phasor_composer.py` 148-151 | CLEARED (NEF cleanup, n=118) — RF reintroduced |
| **C** | memory is a Python `kb` list (numpy composite + label dict) | `rf_phasor_composer.py` 74,172 | CLEARED (Crawford weight-store, n=119) — RF reintroduced |
| **D** | dlPFC association graph is a Python dict from the kb | `brain_conversational_agent.py` _assoc_graph | shortcut in both (the spread is spiking; the GRAPH SOURCE is Python) |

**NOT cheats (verified):** the parser (`BridgeParser`) is genuinely Hebbian-learned on the bridge (6 conjunction →
3 role ensembles, plastic co-firing); the dlPFC SPREAD is genuine spiking content-selection. Residual: vocab at the
validated probe scale (V=16). **Broader sim** (navigation/perception cheats) was addressed in earlier arcs (perception
arc; pure-biology cheat-removal trio n=114) — out of this plan's scope, which is the conversational production path.

## The load-bearing research findings (why this is mostly integration, not new research)

1. **B and C collapse to ONE object — a complex Hopfield / TPAM** (Frady-Sommer 2019). Its weight matrix `W = S S*`
   IS the bridge's existing complex-synapse matvec (`rf_set_complex_weights` + `_rf_advance_one` already compute
   `u = W z`), and it is already implemented + pre-registered-validated in-repo (`resonate_fire_fhrr.py::ResonateFireTPAM`).
   The phasor recode made these MORE substrate-native than the rate composer's ON/OFF mechanisms, not harder.
2. **A is mostly already done** (the 2026-06-04 unified-agent grounding work: V1 Gabor RFs `sim/visual_cortex.py` →
   ventral decorrelation → IT-level codes; measured raw-V1 composition 0% → decorrelated 100%). The novel piece is
   wiring grounded codes into the RF composer + the test; the on-bridge decorrelation is Földiák 1990 (Hebbian +
   anti-Hebbian PV-FS lateral inhibition = local-rule sparse decorrelation).
3. **D is the same mechanism the project already has** — Hebbian co-occurrence → engram-tag store (the
   engram-stim-recall + multitag work, 87.5%/90% validated); CA3 recurrent autoassociation (Marr/Treves-Rolls);
   Garagnani-Pulvermüller emergent spread. The conversion routes the dlPFC's graph weights from the substrate engram
   store instead of the Python recompute.

## Phased plan (de-risk-first parity gate per phase; reuse-by-import; minimal/no `sim/` edits; the agent's FULL suite at parity)

**Phase 1 — (B) spiking phasor cleanup [TRACTABLE, do first].** Replace `_cleanup`'s numpy argmax with a TPAM cleanup:
vocabulary in `W = S S*/D`, magnitude-gated phase-preserving transfer `z = u/|u|·H(|u|−Θ)` (the phasor analogue of
the NEF threshold-rectification + striatal WTA — CA3 pattern completion, Kandel p.1360 / Marr 1971; biology.md
§action-selection WTA), winner = `argmax|S*z|`, on the bridge via `rf_set_complex_weights` + `rf_kick` +
`rf_resonate_steps` (no `sim/` edit). **Gate:** TPAM winner == numpy-argmax winner on the composer's real noisy unbind,
multi-seed 42/43/44; then the agent's full suite at parity. Biology: CA3 completion + BG WTA.

**Phase 2 — (C-A) phasor substrate memory store [TRACTABLE].** Replace the Python `kb` composite with the bound
phasor held in per-fact COMPLEX output weights (`trigger → readout = c·w_gain`; the substrate's weights are already
complex), retrieved by firing the trigger → phase readout (magnitude-invariant — removes the rate store's 0.975-cos
blemish; the linear-glue residual the rate store deferred does NOT exist for phasors). Biology: Hebb cell-assembly
memory-in-weights (Kandel p.1357 verbatim; Tonegawa/Liu 2012) + Marr CA3. **Gate:** per-role substrate-store unbind ==
numpy-store unbind, cleanup held constant, multi-seed; trigger-silence collapse check (genuine read). The fact-dict
LABELS (structure/clause routing) are the residual to minimise — decode structure from the bound vector where possible.

**Phase 3 — (A) grounded concept codes [TRACTABLE for the groundable subset; PARTIAL at 320-scale].** Replace
`rng.uniform` codes with phases from grounded activity: visual concepts via V1 Gabor → ventral decorrelation →
IT-level sparse code; abstract concepts (verbs/function words) via multimodal co-occurrence Hebbian (Pulvermüller);
`φ = angle(P·s)/(2π) mod 1` into `composer.concepts` (the `concepts=` kwarg already exists). On-bridge decorrelation =
Földiák local rules. Biology: Kandel Ch 24 (IT convergence, experience-sharpening); Atick-Redlich (decorrelation);
Olshausen-Field (sparse coding); Quian Quiroga. **Gate:** grounded codes ≥ random-code accuracy on the composer's
queries + abstention preserved, multi-seed. **HONEST boundaries (disclose, do not force):** abstract-from-raw-sensation
is an embodied-cognition limit (best = grounded in the motor/lexical referent); fully-on-bridge decorrelation at 320
concepts is seed-fragile (the labelled-ZCA stand-in may remain a disclosed boundary); the format conversion
([0,1)^D@128 vs [−π,π]@2048) is a known silent-bug class — pin it with a test.

**Phase 4 — (D) substrate associative graph [TRACTABLE for storage; PARTIAL for cue-spreading].** Route the dlPFC's
`c2d` edge weights from the substrate's learned engram-tag co-occurrence store (store co-occurrence as engram tags at
fact-storage; derive each edge from a `stimulate_tag` substrate read) instead of the Python dict; keep the validated
spiking spread unchanged. Biology: CA3 autoassociation (Treves-Rolls/Marr); Garagnani-Pulvermüller; Lerner 2012
attractor spread. **Gate:** `c2d` from engram-store vs Python-dict oracle pick the same direct associates, multi-seed.
**HONEST boundary (already measured):** cue-only associative recall is ~27.5% multi-seed (barely above chance) — the
heteroassociative asymmetry (clean cue completion needs sparse codes); the parity gate is load-bearing; if cue-direction
is the blocker, the principled lever is SWR sleep-replay consolidation, NOT weight hand-tuning.

## Explicitly DEFERRED (honest hard boundaries — documented, not converted now)
- **C-B** (TPAM-as-KB for partial-cue pattern completion, e.g. query-by-agent without the full SVO): dense-phasor
  capacity wall (Frady-Sommer's high capacity is for SPARSE phasors; bundled facts are dense) + needs a fast-weight
  rank-1 path the substrate doesn't expose. Matches the project's prior (B)-options conclusion.
- **Abstract-concept sensory grounding** (Phase 3 residual): embodied-cognition limit.
- **Fully-online STDP autoassociator** (Phase 4 Option B): inherits the v16-compose STDP cold-start hazard.
- **320-scale fully-on-bridge decorrelation** (Phase 3 residual): Földiák local-rule selectivity is seed-fragile at scale.

These are real biology-translatable boundaries (per the top-level goal: honest negatives under strict biology ARE the
deliverable). They are documented as boundaries, not papered over.

## Execution discipline (every phase)
De-risk FIRST (a parity probe with a multi-seed GATE, the A-arc pattern that worked); if GO, integrate as an OPT-IN on
the composer/agent (reuse-by-import; the rate-coded path + the current RF default stay until the opt-in re-validates);
re-validate the agent's FULL existing suite at parity (no test weakened, no-confab moat intact, ZERO regression);
commit + push BOTH remotes; honest finding (GO or BOUNDARY) each step. Protected `sim/` edits only if strictly required
(flagged; expected: none — all four reuse existing bridge mechanisms). Sequencing: Phase 1 (B) → Phase 2 (C-A) →
Phase 3 (A) → Phase 4 (D); B/C are independent of A/D and lowest-risk, so they go first.

## Outcome
On completion: the conversational composition path is FULLY spiking (cleanup + memory on the substrate, no numpy
readout/store), the codes are grounded for the groundable subset, and the association memory is substrate-held — with
the four honest boundaries above explicitly disclosed. That is "all the convertible non-biological shortcuts converted,
the unconvertible ones named as biology limits" — the honest finish line.

---

## ✅ FINAL STATUS (2026-06-05) — arc complete

| Cheat | Result | Evidence |
|---|---|---|
| **B — numpy cleanup** | ✅ **CONVERTED** | matched filter on the complex synapse + spiking Izhikevich WTA; composer == numpy 27/27 multi-seed; agent **8/8 GPU**. `enable_spiking_cleanup`. `2026-06-05-phase1-tpam-cleanup-derisk-GO.md` |
| **C — Python memory** | ✅ **CONVERTED** | bound composites in per-fact synaptic weights (Crawford store), retrieved in spikes; composer == numpy 27/27 multi-seed; agent **9/9 GPU** (both opt-ins). `enable_substrate_store`. `2026-06-05-phase2-substrate-store-derisk-GO.md` |
| **A — random codes** | ⚠️ **PARTIAL (interface + boundary)** | RF composer works on REAL V1-Gabor-grounded codes 6/6 multi-seed; `grounded_codes` opt-in shipped. BOUNDARY: real-image semantic grounding + abstract-concept grounding (embodied limit) is a multi-month arc. `2026-06-05-phase3-grounded-codes-PARTIAL.md` |
| **D — Python assoc graph** | ✅ **RESOLVED (2026-06-05)** | the residual (weights SET not LEARNED) is CLOSED: `LearnedAssocGraph` learns the concept association graph in a sparse Hebbian recurrent (CA3 autoassociator), wired into `BrainConversationalAgent(enable_learned_assoc=True)` — `elaborate()` spreads over the substrate-LEARNED graph. Multi-seed 24/24 edges + 9/9 top; agent GPU-validated; anti-cheat-clean. The deeper 27.5% cue-direction recall is also RESOLVED (learned heteroassociative completion, multi-seed). `2026-06-05-D-cue-recall-RESOLVED-sparse-heteroassoc.md` |

**Net:** the 2 clearable cheats (B+C — the ones the rate composer had cleared) are FULLY converted, agent-validated on
GPU, no-confab moat intact, zero regression, NO `sim/` edits. The 2 harder cheats (A+D) are honestly bounded: A's
grounding interface works (full grounding is the embodied/dataset boundary); D's associative memory is substrate-genuine
(the weight-learning residual is a documented buildable follow-on; cue-direction recall is the measured boundary).

**The one remaining BUILD is now DONE (2026-06-05):** D's Hebbian-learned association graph shipped — `LearnedAssocGraph`
(sparse Hebbian recurrent, CA3 autoassociator) wired into the agent (`enable_learned_assoc`), multi-seed + agent-GPU
validated, anti-cheat-clean. **Net update: B+C+D fully converted; A partial (grounding interface + embodied boundary).**
Everything convertible is converted; the only remaining named limit is A's deep semantic grounding (a real object-image
dataset → V1→IT codes), the embodied-cognition boundary.
