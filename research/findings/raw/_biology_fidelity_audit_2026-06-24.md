# Biology-fidelity audit of the recent weeks' work (2026-06-24)

**Read-only audit.** No edits, no runs, no webapp. Scope = the ~last month of arcs (CLAUDE.md
"Recent arcs" 2026-06-23 + 2026-06-24, AUTONOMOUS_STATE CYCLEs ~478-536, the headline findings).
Owner's question: are we PROPERLY basing the work on real biology, or just loose functional
approximations, OUTSIDE the few known/deferred shortcuts?

## The bar (3 levels + residual class)
1. **SUBSTRATE** — does the cognitive op run on simulated neurons/synapses (Izhikevich / resonate-and-fire
   on `SimulationBridge`), or is it numpy math standing in for a brain?
2. **MECHANISM-GROUNDING** — cited to real biology (catalog entry / Kandel / paper), or chosen purely to
   get the function?
3. **STRUCTURE** — even if spiking at runtime, is the structure (weights, bind algebra, sampling, the
   decision/orchestration) LEARNED/self-organized, or host-designed/host-orchestrated?
   - (a) LEGITIMATE host — environment + body + pure bookkeeping/routing the BRAIN-BASED-ONLY standard allows
   - (b) RESIDUAL SHORTCUT — host doing cognition (should be converted)
   - (c) KNOWN-DEFERRED — owner has explicitly OK'd deferring (the FHRR bind algebra; LLM-as-fluency; etc.)

Method: every load-bearing claim cross-checked against code (file:line) + the finding + the catalog.
Where I confirmed a claim by reading the code I say so; where I'm taking the finding's word I say so.

---

## Per-mechanism table

| # | Mechanism (finding) | (1) Substrate-spiking? | (2) Biology-cited? | (3) Structure: learned/self-org vs host | Residual class |
|---|---|---|---|---|---|
| M1 | **Stream / PPMI cortex** — learn concept codes from listening (`2026-06-15-on-bridge-hebbian-co-occurrence`; used by develop loop via `_phaseB_onbridge_stream_cortex_derisk.build_stream_bridge`) | **YES** — real `SimulationBridge`, `enable_hebbian_learning=True`, rate-Hebbian co-occurrence; codes read from spiking population | YES — Hebbian co-occurrence / PPMI local-normalization; ATL hub-and-spoke (Patterson-Lambon Ralph); Garagnani-Pulvermüller spiking precedent | **LEARNED** — codes self-organize from the corpus on-substrate; frozen-brain anti-cheat (plasticity-off → learns nothing) confirms. Residual: log-domain read-out double-centring is a host scaffold (I-10) | mostly (a); read-out normalization scaffold = (b) thin |
| M2 | **Limbic / DA core** — δ = r − V in spikes (`2026-06-18-limbic-core-rpe-battery-GO`; deployed via 1B) | **YES** — `reward_us`→`snc` synaptic afferent, `striosome_value` GABA_B/GIRK value subtraction, `dopamine` signed-firing modulator; r, V, δ all neural | YES — Schultz 1998 RPE; Rescorla-Wagner; PPN reward afferent; striosome value critic; GABA_B/GIRK | **LEARNED** — critic learns V via three-factor STDP×DA. Decisive lesions: reward-lesion → burst vanishes, GABA_B-lesion → gap collapses (proves r,V not re-hidden host scalars) | (a) — genuine end-to-end |
| M3 | **Spiking decision (nav action)** default-on (`2026-06-19-spiking-decision-default-on-GO`) | **YES** — Wang-2002 accumulator + Lo-Wang commit-burst; 100% commit-burst, 0 argmax fallback | YES — Wang 2002 NMDA attractor; Lo-Wang; Usher-McClelland leak | **self-organized dynamics** (emergent threshold-crossing). Honest residual: ~16% finite-size/commit-timing cost vs host argmax = irreducible point-neuron floor (B-4) | (a) — the residual is a characterized boundary, not a shortcut |
| M4 | **Cross-region Route A** (language→action) default-ON (`2026-06-24-crossregion-...GO`) | **YES** — parser firing opens synaptic `COMMAND_GATE`; learned word→action route; cascade WTA selects | YES — thalamocortical transmission-gating (Logiaco-Abbott-Escola); the spoken-instruction template | **LEARNED route** + provenance-asserted (no Python value copy); lesion-collapse + scramble-collapse | (a) — load-bearing, anti-cheated |
| M5 | **Cross-region Route B** (perception→compose, host-M closed) (`2026-06-24-crossregion-...GO`) | **YES** — `gen_perception→gen_concept` convergence is plastic+learned; grounding reads `cp_firing_states` (line 175); NMDA on `gen_concept`. SPIKES-ONLY (host matvec `M` retired from default) | YES — ATL crossmodal convergence; NMDA temporal integration past the rate-code wall | **LEARNED** convergence (the LEARN pass grows only `gen_perception→gen_concept`). Residual: the `gen_concept→gen_fact` projection is FIXED block-diagonal (structural prior, host-designed) | (a) for the learned convergence; the fixed fact-tag projection = thin (b)/structural-prior |
| M6 | **One-brain composer — binding ops** (`one_brain_composer.py`, OneBrainComposer) | **YES** — RF resonate-and-fire + complex synapses; bind/unbind/bundle THROUGH complex synapses (Frady-Sommer) | YES — Frady-Sommer 2019 FHRR resonate-and-fire phasor neurons + complex synapses; VSA / Eliasmith SPA | **HOST-DESIGNED ALGEBRA** — exact-inverse FHRR is a clean idealization; the weights/conj structure are `np.conj` host-injected. `local_reciprocal_unbind=True` localizes the unbind (default-on); cleanup-codebook conj stays host | **(c) KNOWN-DEFERRED** — the step-3 "true cortex"; 3 NEGATIVEs incl. the deep-dendritic oracle → characterized point-neuron BOUNDARY (3B) |
| M7 | **One-brain — persistent loop / op hand-offs** (`integrated_loop`, CYCLE 526 / I-1) | **PARTIAL** — flat+clause recall carries the live phasor across unbind→cleanup ON substrate (the only `to_host` is the final body-read); BUT the BETWEEN-op hand-offs (`_compose_phases`→`_write_block`→`_read_blocks`→`_select`) still round-trip through host. I-1-a de-risk proved the 3 round-trips byte-replaceable on-device but not yet wired | YES — persistent attractor loop / Frady-Sommer phasor handoff | host round-trips between ops (host orchestration) | (b) for the BETWEEN-op host glue — but it is "routing/handoff," the lighter end of (b); the de-risk shows it's cheaply convertible |
| M8 | **One-brain — cue-match SCAN + no-confab moat** (C-2, `_scan`/`query_agent`/`ask_yes_no`) | **PARTIAL** — `integrated_loop` (spiking K-way sequencer, K=32 GO) default-ON at the 320 flagship; default-OFF at the small-vocab library (over-abstains = code-margin boundary). The matched FILTER is spiking; the WHICH-fact-answers + abstain-decision is a host Python `==` loop on the library default | partial — the sequencer is biology-shaped (K-way competition); the host scan is not cited | host loop on the library default; spiking at the 320 flagship | **(b)** at the library default — "the single largest live default-on conversational host residual"; CLOSED at the flagship |
| M9 | **One-brain — cleanup SELECTION** (C-3 winner-pick) | **YES (flagship) / host (rf-lib)** — Izhikevich NEF WTA == argmax, ON in the 320 demo + onebrain (1A); host `argmax` on the rf/numpy/test path | YES — Stewart-Tang-Eliasmith 2011 NEF thresholded cleanup (the Spaun cleanup) | spiking WTA structure validated; the residual argmax is host-bookkeeping where it remains | mostly (a) at production; (b)-thin on the rf/test path (intentional oracle) |
| M10 | **Spiking generator (Gen-F → RF)** — "spiking generator generates novel text" (`2026-06-22-genseq-convert-GO`, `-fullblock-rf-integration-GO`, `-rf-distill-GO`) | **MIXED — the weakest substrate claim.** The convert (`_genseq_convert_derisk`) is a **PyTorch rate-coded** conversion (T=32), NOT on the bridge. The "on-bridge" full-block runs the learned matvecs on the **conductance-free RF accumulator** where `Re(Z)=nsteps·(a@W)` is computed **EXACTLY, with NO g·(V−E)** → i.e. the RF substrate reused as an exact linear-matmul accumulator, not dynamical spiking; softmax/GELU/LayerNorm are **host "faithful reads"** (0 spiking) in the integration | YES (the rate-code conversion class: QCFS/MBE; FHRR complex synapses for the matvec) — but the grounding is for the conversion math, not for "neurons compute this" | **HOST-DESIGNED** — the weights are the ANN's distilled/host weights injected; the nonlinearities are host reads | (b)/(c) — host weights = the H-2 deferred structure; the "spiking softmax/GELU/LayerNorm" are validated SEPARATELY (0.96/0.99/0.9998) but NOT in this integration; **flag: "spiking generator on the bridge" overstates — it is host-weight matvecs on the RF accumulator + host nonlinearity reads** |
| M11 | **Spiking softmax / GELU / LayerNorm** (`2026-06-23-spiking-{softmax,gelu,layernorm}-GO`) | **YES** (standalone) — spiking realizations validated 0.96/0.99/0.9998 | YES — rate-code / divisive-normalization biology | spiking ops; structure is the op shape (parameter-free) | (a) standalone; NOTE they are not yet composed into M10's full block |
| M12 | **Grounded-language faculty — the spiking Qwen** (`2026-06-23-grounded-lang-P1b-GO`, `-INTEGRATION-GO`) | **NO (off-bridge)** — full spiking forward of Qwen2.5-0.5B runs in **PyTorch on the 3090**, NOT on `SimulationBridge` (bridge co-residence DEMONSTRATED separately at 14 GB but the integration uses the off-bridge model). Spiking-IN-PYTORCH (rate-coded ops) ≠ on-substrate | YES — the project's own calibrated-graded-read convert (rate code); fluency=LLM is the owner-sanctioned decoupling | **HOST/LLM** — the entire fluency faculty is a 494M trained transformer; not the brain | **(c) KNOWN-DEFERRED + OWNER-TRADE** — "LLM supplies fluency, brain supplies+verifies content"; confirmed fluency-ONLY by the lesion (sever brain proposal → VERIFY rejects every free-gen) |
| M13 | **GATE→CONSTRAIN→VERIFY grounding loop** (`-INTEGRATION-GO`, `brain_chat_tui.ChatBrain`) | **mixed** — GATE = composer recall+abstain (M6/M8 substrate), CONSTRAIN = the off-bridge faculty (M12), VERIFY = re-parse via the brain's content-extractor + `BridgeParser` (substrate) | YES — the no-confab moat / predictive-verification framing | **host orchestration** — the loop control + the `rsvo==gate_svo` `==` comparison + question routing are host Python | (b)-thin (legitimate routing) for the loop glue; the moat-DECISION `==` ties to C-2's host scan. The anti-cheat (caught a real 0.5B role-inversion) is genuine |
| M14 | **Communicable-brain Probe 1** — "what do you think about X" (`2026-06-24-communicable-brain-probe1-GO`) | **mixed** — ASSIMILATE reads the learned PPMI graph (M1, substrate-derived); PROPOSE = the **b2 host-loop sampler** (M15); RENDER = neural serial-order (substrate) + a CPU fluency stub; VERIFY = substrate re-parse | YES — generative replay (G.09); serial-order production (Grossberg CQ, G.07/H.19) | **host-orchestrated turn pipeline** — the 4-step ASSIMILATE/PROPOSE/RENDER/VERIFY sequence is host Python; the load-bearing signals (PPMI graph, parser, renderer) are the brain's | (b) for the turn orchestration; lesion anti-cheat (46/46: sever brain proposal → VERIFY rejects free-gen) genuinely proves the CONTENT is the brain's, not the LLM's |
| M15 | **b2 generative replay proposer** (`2026-06-23-genfrontier-b2-...GO`) | **NO** — confirmed by code: `GenerativeReplayProposer` is a numpy host loop; `_sample_weighted` uses `self.rng.choice(p=...)`; no `SimulationBridge`, no spikes | YES — hippocampal generative replay / constructive imagination (G.09 — confirmed real catalog entry, Kandel 6e Ch 52; Stoianov-Maisto-Pezzulo 2022; Barry/Love 2023) | **host sampling, brain-derived likelihood** — the PLAUSIBILITY signal IS the brain's learned PPMI graph (load-bearing: shuffled-graph collapses to chance); the resampling LOOP + the gates are host code | **(b)** for the sampler (host bookkeeping doing the recombination); fully-spiking SWR-gated CA3 resampler is the named follow-on. The shuffled-graph + lesion + 0-leak anti-cheats are rigorous |
| M16 | **Develop loop** (`2026-06-23-longitudinal-develop-loop-GPU-GO`) | **mixed** — WAKE/LEARN = the REAL on-bridge stream cortex (M1, spiking Hebbian, corr(M,C) 0.894); CONVERSE = the composer (M6-M9); SLEEP/consolidate = host self-replay re-`hear()` + retention-retest (I-8), NOT hippocampal SWR (the conv bridge has no hippocampus) | YES — CLS (McClelland 1995); the WAKE learning is genuinely biological | WAKE learning self-organized; the loop orchestration (day stages, growth, persist) + consolidation stand-in are host | WAKE = (a); consolidation = **OWNER-TRADE** stand-in (the no-forget is load-bearing, 2.25× forget-contrast); loop orchestration = (a) bookkeeping |
| M17 | **Value/salience + learned-talkativeness** (scopings: `2026-06-19-tier2-limbic-to-composer`, Probe-1 §honest-scope) | **n/a — scoping only** — the DA→composer READ-side hook is wired+GO (I-4-a, default-ON) using M2's spiking DA; the value/salience appraisal that would make the brain "choose to speak more" is SCOPED, not built | (read-side) Lisman-Grace VTA-hippocampal gate | read-hook uses the spiking DA (learned); the appraisal mechanism is unbuilt | (a) for the wired read-hook; the broader appraisal is future work, not a current shortcut |

---

## The honest headline (rough fractions)

Counting the ~17 recent mechanisms by their load-bearing core:

- **Genuinely (1)+(2)+(3) — spiking substrate + cited biology + learned/self-organized structure (or a
  characterized boundary, not a shortcut): ~40%.**
  The nav/limbic spine is the strongest: M1 stream cortex (learned codes on-substrate), M2 limbic δ=r−V
  (genuinely spiking + decisive lesions), M3 spiking decision, M4 Route A, M5 Route B's learned
  convergence. These are real computational-neuroscience mechanisms with rigorous lesion/scramble/
  frozen-brain anti-cheats. This is the project's high-fidelity core and it is genuinely impressive.

- **(1)+(2) spiking + cited, but host-DESIGNED/host-ORCHESTRATED structure: ~35%.**
  The conversational composer (M6-M9): the RF ops are spiking + Frady-Sommer-cited, but the bind algebra
  is the host-designed exact-inverse FHRR idealization (KNOWN-DEFERRED, c), the between-op hand-offs round-
  trip through host (M7, b), the cue-match scan is a host loop on the library default (M8, b). The
  generator (M10): the matvecs run on the RF accumulator but with host weights + host nonlinearity reads.

- **Functional approximation / host orchestration standing in for a brain mechanism: ~25%.**
  The b2 proposer (M15, numpy sampler), the communicable-brain turn pipeline (M14, host-orchestrated 4
  steps), the GATE/VERIFY loop control (M13), the off-bridge PyTorch LLM (M12). These produce real,
  anti-cheated RESULTS, but the cognition is host-orchestrated around brain-derived signals — they are
  honest de-risks/scaffolds, not yet on-substrate mechanisms.

**Crucially:** almost every host-orchestration case is HONESTLY LABELLED in its own finding as a de-risk
harness / propositions-not-discourse / off-bridge / stand-in. The project's self-audit
(`2026-06-23-cheats-shortcuts-integration-inventory.md`) is rigorous and matches this audit closely. The
recent work is NOT secretly cheating; it is largely de-risking + scaffolding, with the spiking spine
genuinely built.

---

## Residual shortcuts (class b) BEYOND the known-deferred (class c)

These are the ones a reader skimming "Recent arcs" might gloss over — host doing cognition, default-on or
load-bearing, not the headline owner-OK'd ones:

**Class (b) — host doing cognition (should eventually convert):**
1. **C-2 cue-match SCAN + abstention decision** (M8) — host Python `==` loop on the library default; the
   spiking K-way sequencer exists (320-GO) but the small-vocab default reverted (code-margin boundary).
   *The single largest live default-on conversational host residual.* The moat-decision being a host `==`
   also makes M13's VERIFY decision host.
2. **The b2 generative-replay SAMPLER** (M15) — the resampling loop + gates are numpy host code; only the
   plausibility likelihood is the brain's. The "first novel-composition > 0 from a brain mechanism" claim
   is fair for the SIGNAL but the SAMPLER is host. Fully-spiking SWR-CA3 resampler = the named follow-on.
3. **The communicable-brain turn pipeline** (M14) — the ASSIMILATE/PROPOSE/RENDER/VERIFY orchestration is
   host Python around brain signals. Genuinely anti-cheated (the lesion proves the content is the brain's),
   but the turn-level cognition is host-orchestrated.
4. **Between-op hand-offs in the one-brain composer** (M7 / I-1) — even on the OneBrain path the ops
   sequence via host `to_host` round-trips + re-kicks. Lighter end of (b) (routing), de-risked as
   byte-replaceable, but not yet the "one persistent interacting spiking loop" the owner means.
5. **Rich-answer assembly** (C-6, the `--rich` discourse path) — gather/de-dup/thread/follow-up/breadth-walk
   are host heuristics; the validated dlPFC spiking discourse-planner (3G) is GO but NOT wired onto the
   runtime `--rich` path (only the de-risk runner sets `neural_planner=True`). The "substantive
   conversation" cognition is host.
6. **Read-out normalization scaffold** (M1 / I-10) — the log-domain double-centring read-out is a host
   scaffold; the on-bridge normalization circuit (per-concept feedforward inhibition + per-hub adaptation)
   is scoped, not built.
7. **Generator host weights + host nonlinearity reads** (M10) — overlaps H-2; the matvec runs on the RF
   accumulator but the WEIGHTS are ANN-distilled host weights and softmax/GELU/LayerNorm are host reads in
   the integration de-risk (the spiking versions exist standalone but aren't composed in).

**Structural-prior residuals (b-thin / defensible):**
8. **`gen_concept→gen_fact` fixed block-diagonal projection** (M5) — host-designed structure (the category
   structure is learned; the tag projection is fixed). Thin.
9. **V1 Gabor RF weights** (N-5) — host formula; ruled DEFENSIBLE innate prior (criterion-2 residual);
   target = retinal-wave self-organization. Defensible, not a cheat.

**Class (c) — KNOWN-DEFERRED (owner explicitly OK'd) — listed for completeness, NOT flagged:**
- The exact-inverse FHRR bind algebra → learned cortex (M6 / C-1 / H-3) — 3 NEGATIVEs incl. the deep-
  dendritic oracle → characterized point-neuron BOUNDARY, not a pending build.
- The off-bridge PyTorch Qwen as the fluency faculty (M12 / C-9 / I-2 / H-1) — the LLM-fluency decoupling;
  the moat makes hallucination impossible by construction (proven: a real 0.5B role-inversion was caught).
- The host-DESIGNED structure of every converted spiking op (H-2) — the deepest categorical blocker; target
  = developmental self-organization. The DEV-RANDOM codes are defensible; the structured part is the residual.
- Consolidation = self-replay stand-in (I-8) — full SWR-on-conv-bridge deferred (no hippocampus there).

---

## Fair verdict

**Are we "properly based on real biology"? — Yes, with honest caveats, and notably more rigorously than a
"loose functional approximation" project.** Three things hold up under scrutiny:

1. **The mechanism-grounding (level 2) is consistently real.** Every mechanism I checked cites a specific
   catalog entry / Kandel chapter / paper (I confirmed G.09, G.07, H.19, D.14 exist in the catalog; the
   limbic core cites Schultz/R-W; the decision cites Wang-2002/Lo-Wang; the binding cites Frady-Sommer).
   The deep-research gate is genuinely the standing first move, and it has repeatedly reframed problems
   correctly (the rate-code/whitening family, the missing accumulator, ventral-vs-dorsal nav).

2. **The substrate (level 1) is real where it matters most — the nav/limbic/perception spine — and the
   anti-cheats are rigorous.** Lesion/scramble/frozen-brain/shuffled-graph/6-seed controls are applied
   consistently and have caught the project's own overclaims (the multiple documented retractions; the
   3F false-localization fix; the b2 design-iteration negatives). This is the opposite of self-deception.

3. **The honesty discipline is strong.** The project's own inventory + close-out audit pre-empt almost
   everything in this audit. Findings label off-bridge/host/stand-in explicitly. The BRAIN-BASED-ONLY
   standard distinguishes legitimate host (environment/body/routing) from host-cognition cleanly.

**The biggest fidelity gaps to close (in priority order):**

1. **Functional one-brain integration (I-1/I-4/M7/M14)** — the headline "one brain" is still, at the
   integration level, host-orchestrated hand-offs between spiking ops + co-located-not-interacting halves.
   This is the owner's named #1 and the most important honesty boundary: "spiking ops glued by host Python"
   is not yet "one persistent interacting spiking loop." Probe-1's turn pipeline and the rich-answer
   assembly are host-orchestrated cognition, not just routing.

2. **The conversational composer's host residuals (M6/M8/M10)** — the FHRR algebra (c, boundary) is
   honest; but the cue-match scan (b, default-on) and the generator's host-weights+host-nonlinearity-reads
   (b/c) are where "spiking" most overstates. The "spiking generator on the bridge" claim specifically
   should be read as **host-weight matvecs on the RF accumulator + host nonlinearity reads**, not dynamical
   spiking neurons generating text — the conductance-free RF accumulator computes `a@W` exactly by
   construction.

3. **The off-bridge LLM (M12)** — the single biggest level-1 gap; bridge co-residence is demonstrated but
   the integration runs use PyTorch. Owner-sanctioned, but it is the largest "not on the substrate" piece.

4. **Host-DESIGNED structure (H-2)** — the deepest level-3 gap, cutting across the bind weights, the
   consolidated-ANN weights, the grounding projection. Owner-flagged; the target (developmental
   self-organization) is genuinely far.

**One-line:** the recent work is properly grounded in real biology at the mechanism level and genuinely
spiking on its nav/limbic/perception spine with rigorous anti-cheats; the honest residuals are (a) the
host orchestration GLUE of the "one brain" (the integration is host-sequenced, not yet one spiking loop),
(b) a handful of default-on host-cognition residuals on the conversational path (the cue-match scan, the
rich-answer assembly, the b2 sampler, the turn pipelines), and (c) the explicitly-deferred FHRR algebra +
off-bridge LLM + host-designed structure. The "spiking generator/composer/one-brain" language occasionally
runs ahead of the substrate reality (host-weight RF-accumulator matvecs and host-orchestrated hand-offs),
and that is the gap between the headline and the strict bar — but it is largely self-documented, not hidden.
