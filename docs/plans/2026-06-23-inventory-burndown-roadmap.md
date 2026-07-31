---
type: plan
status: live
date: 2026-06-23
---

# Inventory burndown roadmap — close out the cheats / integration / optimization / hardware gaps (2026-06-23)

> **Source:** `research/findings/2026-06-23-cheats-shortcuts-integration-inventory.md` (the definitive 4-dimension inventory).
> **Owner directive (2026-06-23):** after the conversation test, close out as much as we can. Order = **quick/easy first → highest-impact → long/true-cortex last**. Each item gets a **deep-research pass** + the **established breakthrough workflows**. **No deferring** — pursue each to resolution. **Parallel where possible.**

## Standing per-item workflow (the breakthrough pattern, applied to EVERY item)
1. **Research gate FIRST** (read-only deep-research subagent): diagnosis → ranked biologically-grounded options → reusable machinery → cheapest-first de-risk → the anti-cheat controls it needs. Present before building. For any item framed as a BOUNDARY/"can't"/"primitive", the **SURPASS round is mandatory** (isolate+quantify the genuine residual, reframe via real biology, rank cheap surpass mechanisms, verdict surpassable-or-precisely-why-not) — accept a boundary only after it SURVIVES.
2. **Cheap-first de-risk** (numpy/CPU probe) — prove/refute the mechanism before GPU/`sim/` effort.
3. **Build** controller-managed (GPU runs as a controller Bash bg, exit-notified — NOT a backgrounding subagent that stalls). `sim/` edits get a byte-level diff review.
4. **Validate** — 6-seed for variable effects, the anti-cheat controls (shuffled/lesion/permuted), trust-but-verify each diff. Honest negatives ARE the deliverable.
5. **Commit both remotes** (origin + gitea), update the inventory doc + `AUTONOMOUS_STATE.md`.
- Never weaken the no-confab moat below "no UNSUPPORTED claim survives"; brain-based-only; both halves stay GPU for real runs.

## Execution model
- **Parallel tracks within a phase** (independent items run concurrently as subagents / controller-managed runs). Dependency chains run in order.
- Phases are roughly sequential by the owner's order, but a later-phase **research gate** (read-only, cheap) may run in parallel with an earlier phase's build to stay ahead.

---

## PHASE 1 — QUICK & EASY (default-flips + cheap engineering; little/no new science)
*Goal: bank the items that are already built-and-validated but defaulted off, plus the cheap storage/codegen wins. Mostly a default-flip + a validation pass.*

- **1A — Conversational spiking default-flip pass** (C-2/C-3/C-4/C-5, H-5/H-6/H-7). One coordinated arc (the flags interact): `enable_spiking_cleanup`, `integrated_loop` (the K-way sequencer), `enable_substrate_store`, `enable_learned_assoc`, the cleanup-codebook local conj. **Known blocker:** the full default-on migration REVERTED (the sequencer over-abstains at small vocab; moat stayed intact). → the research gate targets the over-abstention (threshold/scale), de-risk the **fold+scale**, then flip. Validate: full who/what + describe/yes-no + moat 0-FA at V=320 multi-seed. *Closes ~5 conversational shortcuts + 3 hardware blockers at once.*
- **1B — Nav limbic-core purity flip** (N-1/N-2). Default-on `spiking_reward_us` / `enable_neural_critic` / `perceived_approach_reward` so the deployed reward/value/δ is spiking, not the host sign-formula. **Validate by the signal's FUNCTION** (the Schultz RPE battery — already 6/6), NOT the orient-solvable nav A/B (the GREEN_INERT confound). Confirm nav not regressed + moat preserved (disjoint slices).
- **1C — Drop the float64-CSR storage waste** (H-4, cheap part). Drop the all-zero imaginary CSR (→5.93 GB) + f32 data (→3.95 GB). `sim/` edit, default-off→on byte-review; the dense-fp16 form is Phase 2 (2A). Advances both VRAM and the hardware axis.
- **1D — On-device connection tuple-gen** (O-7). Build `rows/cols/data` with cupy `arange`/broadcast instead of host Python `complex()` generators (`sim/bridge.py:rf_set_complex_weights`). Removes the ~29 ms host-sync on every `store`/reconsolidation. `sim/` edit, byte-review.
- *(stretch)* **1E — exact RMSNorm residual** (H-9): the L1-approx `enable_input_divisive_norm` (+0.037, 49 instances) — a small `√(mean x²)` divisive circuit OR accept the host-RMSNorm read.

**Parallel:** 1A · 1B · 1C · 1D are independent → 4 concurrent tracks. (1A has an internal dependency: the sequencer fold+scale before the flip.)

---

## PHASE 2 — HIGHEST IMPACT
*Goal: the biggest-leverage wins — the cheap-but-huge perf/storage root, and the owner's #1 north-star (functional one-brain integration).*

- **2A — The storage/perf root** (O-2 → O-1 → O-3; ties H-4 dense-fp16). A sequential chain:
  1. **O-2 dense matvec** (sparse complex-CSR → cuBLAS GEMM): bit-exact, ~3600–13000×/shape. Host-forward needs NO `sim/` edit; the on-bridge-purity version is an optional default-off `cfg.rf_dense_weights` + dense `cp_rf_w_dense` (byte-review).
  2. **O-1 on-GPU LLM forward** (keep graded nonlinearities + attention + RoPE on-GPU, kill the ~216 per-linear H↔D copies/token): 97% of the bridge-co-residence wall-clock. → projected prefill ~7200 / gen ~330 tok/s.
  3. **O-3 KV cache**: generation O(context)→O(1)/token.
  *Unlocks usable real-time-local language AND the natural neuromorphic representation in one move.*
- **2B — Functional one-brain integration** (I-4 + I-1 + I-5 + I-7). The owner's "real one brain": nav and conversation INTERACTING via synapses, the brain's own ops handing off as spikes (not host round-trips), the limbic core reaching the cortex. Steps (each research-gated):
  - the synaptic **parser→composer** route (I-5; the `hear_synaptic` precedent exists in the conv-only bridge);
  - the **op-handoff-as-spikes** within the composer (I-1; replace the `to_host`/re-kick glue with a persistent interacting loop);
  - **cross-region nav↔conv** synaptic pathways (I-4; the spoken-instruction `COMMAND_GATE` is the working template — perception during nav writes conversational memory, a parsed command drives the cascade, shared grounded concepts);
  - the **limbic→composer** deep integration (I-7; the read-side DA gate is GO+wired — extend to the encoding hook + the RF-dynamics threading, a sketched `sim/` edit).

**Parallel:** 2A (a perf chain) · 2B (the integration arc) are independent → 2 concurrent tracks. 2B's four steps are partly sequential.

---

## PHASE 3 — DEEP / LONG / TRUE-CORTEX (last)
*Goal: the items we're confident take long — the dendritic substrate and everything it unblocks. 3A is the ENABLER; it goes first within the phase.*

- **3A — The dendritic substrate, Phase 3** (the enabler). The two-compartment apical/basal neuron (D2 Phases 0-2 already built: the numpy gate + the on-bridge two-compartment neuron + the learned graded cortex embedding) → plug into the dual/CLS pipeline + the conversational gates (task #23). **Unblocks 3B + 3C.**
- **3B — FHRR → learned generalizing cortex** (C-1/H-3 + H-2). Replace the exact-inverse binding algebra with a LEARNED spiking-cortical binder that generalizes across similar concepts (the dendrite-gated path) + developmental self-organization of the structure (H-2, the deepest categorical blocker — host-designed→self-organized). The step-3 "true cortex" fork's structured (B) path.
- **3C — The dendritic-frontier nav items** (B-1 place-value δ, B-3 TD temporal-credit). The graded value-read + the TD backup via the two-compartment dendrite (apical=structural place drive, basal=learned value — separating the two quantities a point neuron can't). *Honest: behaviorally inert on the current task → substrate-mapping deliverables; the SURPASS round decides build-vs-accept.*
- **3D — Off-bridge-LLM → on-bridge functional integration** (I-2/H-1). Bridge co-residence is DEMONSTRATED (24-layer Qwen on the live RF substrate, bit-exact, 14 GB local) but co-RESIDENT not INTERACTING → the faculty's fluency functionally gated by the brain's grounding, all on one substrate; on-chip representation is the further frontier.
- **3E — Brain-owns-generation** (I-9 + the b2 seed). Integrate the BPTT-SNN generative loop (DEMONSTRATED+robust standalone) + the b2 generative-replay proposer (GO) into the ONE conversational brain — the brain generates novel grounded discourse, verified by the moat.
- **3F — The SC sustained-orienting loop** (B-2). The deepest genuine nav substrate boundary (the closed neural-reward→critic→actor loop can't sustain nav). The SURPASS pass: the log-polar foveal-render environment-side surpass (in-flight) + the actor cascade N-bias.
- **3G — The neural discourse-planner** (C-6). Replace the host rich-answer assembly (gather/thread/follow-up/stop logic) with neural working-memory-driven content selection over the spiking association memory — substantive-conversation cognition on-substrate.

**Parallel:** 3A first (the enabler) → then 3B · 3C (dendrite-dependent) + 3D · 3E · 3F · 3G (independent) run as concurrent tracks.

---

## Boundaries note (honest)
B-1/B-2/B-3/B-4 are characterized point-neuron limits today. Per the SURPASS directive, each still gets the mandatory surpass research round before we accept it — but several are likely genuine substrate boundaries (the BRAIN-BASED-ONLY deliverable, not a "fix"). 3A (the dendrite) is the named candidate that may convert B-1/B-3 from boundary to closed. B-4's ~16% spiking-decision cost is an irreducible finite-size floor (already default-on).

## Sequencing summary
**Phase 1** (4 parallel tracks, mostly flips+validation) → **Phase 2** (2A perf-chain ∥ 2B integration-arc) → **Phase 3** (3A enabler → 3B/3C ∥ 3D/3E/3F/3G). Phase-3 research gates can pre-run during Phases 1-2 to stay ahead. Every item: research-gate → cheap-first de-risk → build → 6-seed+anti-cheat validate → commit both remotes → update the inventory.
