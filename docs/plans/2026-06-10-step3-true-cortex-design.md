# Step 3 — the TRUE CORTEX: a LEARNED spiking-cortical binder to replace the composer's exact-inverse VSA idealization

**Status:** DESIGN (read-only deep-research + design pass; the ONLY file written is this doc).
**Date:** 2026-06-10 (overnight Thread, step-3 greenlit by the owner — "complete the functional cortex").
**Author role:** read-only deep-research + design subagent (catalog + Kandel + literature reviewed BEFORE any build, per standing practice).
**Scope:** the project's deepest, highest-variance arc. Replace the conversational composer's clean, exactly-invertible
vector-symbolic algebra (an idealization) with a **learned, lossy, redundant spiking-cortical binder** that (i) matches
the validated capability spec, (ii) keeps the no-confabulation abstention moat, (iii) dissolves the cross-code
rate-vs-phasor wall the functional-integration arc exposes.

---

## 0. Terms (defined once — owner standing requirement; no undefined acronyms)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated spiking neurons stepped by one
  `_run_one_simulation_step` loop. The "brain."
- **composer** — the conversational module that holds facts and answers questions. Two implementations exist, both
  reuse-by-import (no `sim/` edit): `RFPhasorComposer` (`research/runners/rf_phasor_composer.py`, the **production
  default**) and `CoreSimComposer` (`research/runners/core_sim_composition.py`, the legacy rate variant).
- **role-filler binding** — combining a *role* (agent / action / patient / polarity / attribute) with a *filler*
  (a concept word, e.g. "dog") into one composite vector that represents "dog is the agent." A fact "dog go north"
  is the **bundle** (vector sum) of three bound role-filler pairs.
- **bind / unbind** — *bind* makes the composite; *unbind* recovers a filler given a role ("who is the agent?").
- **VSA (vector-symbolic architecture)** — a family of schemes that represent symbols as high-dimensional vectors
  and bind them with an algebraic operation that has a defined inverse (so unbind is exact up to noise). FHRR
  (Fourier Holographic Reduced Representation), the production scheme, binds by **phasor** multiplication: each
  concept is a vector of phases in `[0,1)^D` (a complex unit vector per dimension), bind = element-wise complex
  product, unbind = multiply by the complex conjugate. The composer realises this on the bridge's
  **resonate-and-fire** neurons + **complex synapses** (`NeuronModel.RESONATE_AND_FIRE`, `rf_kick`,
  `rf_set_complex_weights`, `rf_read_phases`).
- **cleanup** — after unbind, the recovered vector is noisy; cleanup snaps it to the nearest stored concept code.
  Today this is a fixed nearest-neighbour `argmax` over the codebook (with an opt-in spiking NEF/WTA variant).
- **the no-confab(ulation) moat / abstention** — the agent returns "I don't know" (Python `None` / `"unknown"`)
  when no stored fact matches the query, instead of inventing an answer. This is the project's hardest validated
  acceptance bar (100% = 20/20 unstored cues abstain, multi-seed, V=320).
- **familiarity / novelty signal** — a scalar (or small population) that reports "have I seen this before?"
  independent of *what* it is. In cortex this is the perirhinal repetition-suppression / recognition-memory signal.
- **mixed selectivity** — single neurons tuned to nonlinear *combinations* of inputs (role AND filler), which makes
  a downstream linear readout able to separate combinations a pure rate code cannot (Rigotti-Fusi 2013; the
  cerebellar expansion-recoding of catalog F.12 is the canonical substrate).
- **systematicity (Fodor-Pylyshyn)** — if a system understands "dog chases cat," it should automatically understand
  "cat chases dog" without separate training. A symbolic algebra has this for free (the operation is the same for
  any operands); learned networks notoriously do not. This is the central risk of step 3 (§7).
- **BPTT spiking cortex** — the project's surrogate-gradient backprop-through-time spiking network (`sim/bptt_snn.py`,
  `sim/bptt_snn_gpu.py`, `sim/surrogate_grad.py`, `sim/char_tokenizer.py`,
  `research/runners/cortex_pretraining.py`). A multi-layer leaky-integrate-and-fire net trained by gradient descent
  with a smooth surrogate for the spike threshold. Validated on a toy ABC sequence task and Tiny Shakespeare
  next-character prediction. **All files are on `main`** (their "path-f-hybrid only" headers are stale — verified
  `git ls-tree main`; corresponding tests `tests/test_bptt_snn*.py`, `tests/test_surrogate_grad.py`,
  `tests/test_char_tokenizer.py` are present).
- **engram tag (Tonegawa, catalog D.14)** — the set of neurons that fired above threshold during a window
  (`start_engram_recording` → run steps → `commit_engram_tag`); `stimulate_tag` later re-drives exactly that
  ensemble (causal recall). A way to store "the pattern that just fired" without any codebook.
- **denoise64 codes** — the project's REAL concept codes: captured + denoised firing of the 16-concept pools on a
  trained bridge, cached at `research/findings/raw/.../denoise64_seed{N}.npz`. These are correlated, lossy, grounded
  in the brain's own activity — exactly the "messy code" a learned cortex must read.

---

## 1. DIAGNOSIS — precisely what the idealization is, and what a learned cortex must provide

### 1.1 What the composer actually is

Both composers implement the **same VSA contract** with two idealizing properties:

**(I-1) An exact-inverse algebra.** Bind has a clean mathematical inverse. In FHRR, `unbind(bind(r, f), r) = f`
*exactly* in the noiseless limit because complex-conjugate multiplication inverts complex multiplication
(`rf_phasor_composer._bind` / `_unbind_phases`). The rate variant's `±1` Hadamard is its own inverse
(`core_sim_composition.hadamard_spiking`, reused for unbind with `role := query`). Either way, the *operation* is a
fixed, hand-specified, perfectly-invertible transform. Nothing is learned: `self.roles` and `self.concepts` are
drawn once from a seeded RNG; the bind/unbind wiring is fixed.

**(I-2) A clean-code demand.** The algebra only stays invertible if the codes are **decorrelated and
full-precision**. FHRR phases must be near-uniform and independent across dimensions; the `±1` scheme wants
near-orthogonal codes (hence the optional `decorrelate=True` ZCA whitening in `CoreSimComposer`, and the whole
"orthogonal-codes / sparsity" lineage in the concept-pool work). When codes are correlated (as the brain's real
`denoise64` codes are), the unbind SNR (signal-to-noise ratio) falls and cleanup mis-resolves. CLAUDE.md already
records this as the **"composer-as-idealization" known limitation**: the binding *operations* are on-substrate
spiking, but the residual idealization is *the exact-inverse algebra + the clean-code demand*.

A third, smaller property worth naming:

**(I-3) A codebook-indexed cleanup.** Cleanup is `argmax` over a *known* codebook (`self.words`). The system must be
*told* the vocabulary; it cannot recover a filler it has no code for, and the abstention logic is a Python
`if no fact matches: return None` *outside* the spiking path — the moat is currently bookkeeping, not a neuron.

### 1.2 What a real cortex does instead (the gap to close)

A real cortex has **learned, lossy, redundant readouts** that learn to read *whatever* code arrives:

- It does **not** invert a hand-specified algebra; it *learns* a transform (synaptic weights grown by a plasticity
  rule) that maps "(role, composite) → filler" and tolerates correlated, low-precision, redundant codes.
- Its cleanup is an **associative attractor** (CA3 autoassociator, catalog D.05/D.13; equivalently a Hopfield
  network with Hebbian outer-product weights) — partial/noisy input relaxes onto the nearest stored pattern. This is
  *learned content-addressable memory*, not an `argmax` over a god's-eye codebook.
- Its "I don't know" is a **separate familiarity/novelty signal** (perirhinal recognition memory, catalog D.04
  CA1 match/mismatch comparator; Bogacz-Brown anti-Hebbian familiarity) — a neuron population that reports
  unfamiliarity, gating the readout to abstain. Abstention is a *computed signal*, not a host `if`.

**So step 3 must deliver three learned, spiking things:** (a) a learned binder/unbinder that reads lossy correlated
codes; (b) a learned associative-memory cleanup (attractor) replacing the fixed `argmax`; (c) a learned spiking
familiarity gate that produces the abstention the moat currently fakes in Python. (a)+(b) dissolve I-1 and I-2;
(c) replaces I-3's host bookkeeping with a neuron — and is the load-bearing, hardest piece (§3).

---

## 2. RANKED biology-grounded options for the learned spiking-cortical binder

Ranking is **cheapest-first** and by how much of the idealization each removes. For each: what it *learns*, the
spiking substrate, how it binds AND unbinds, and the honest risk. The recommendation is **Option A first** (it
removes the load-bearing clean-code demand at the lowest cost and on the most existing machinery), staging toward
Option B, with Option C as the high-variance "full" target that is the right place to *measure* the systematicity
limit, not to bet the arc on.

### Option A (RECOMMENDED FIRST) — keep the VSA bind/unbind *operations*, replace the fixed cleanup with a LEARNED associative-memory attractor, and LEARN the codes

- **What it learns:** (1) the **codebook** — the concept codes become *learned* (or learned-decorrelated) instead of
  random-clean, so the binder reads the brain's real `denoise64` codes; (2) the **cleanup** — a learned associative
  memory (a CA3/Hopfield attractor whose recurrent weights are Hebbian outer products of the stored codes) replaces
  the `argmax`. The bind/unbind *operation* stays the FHRR phasor product (it is already spiking and already on the
  bridge), but it no longer *demands* clean codes because the learned attractor cleans up the lossy result.
- **Spiking substrate:** the existing RF phasor neurons + complex synapses for bind/unbind (unchanged); a new
  **attractor cleanup** built from existing primitives — a recurrent excitatory population (the CA3 `internal_density`
  / `ca3→ca3` pathway primitive, catalog D.05) with Hebbian-imprinted patterns and the slow-NMDA recurrent
  machinery already in the step loop (`bridge.py:5751`, the Wang-2002 graded attractor). The codes can be learned by
  the *existing* `decorrelate=`/ZCA path and/or the concept-pool training the project already ships.
- **Bind:** unchanged FHRR phasor product. **Unbind:** unchanged conjugate phasor product → a noisy estimate → the
  estimate is the partial cue into the **attractor cleanup**, which relaxes to the nearest stored code = the
  recovered filler (catalog D.13 pattern completion). This is exactly the Frady-Sommer "superposition + clean-up
  memory" architecture (the resonator literature notes the clean-up step *is* a Hopfield net with outer-product
  Hebbian weights) — but with the clean-up made the *load-bearing learned* piece.
- **Honest risk:** this is a *hybrid* — the binding operation is still the fixed algebra, only the cleanup and codes
  are learned. It removes I-2 (clean-code demand) and I-3 (god's-eye `argmax`) but **not** I-1 (the exact-inverse
  algebra). So it is the smallest honest step toward "learned cortex," not the whole thing — an owner who wants the
  *binding* itself learned will see this as partial (correctly). Its value: it dissolves the rate-vs-phasor wall's
  *cleanup* half cheaply and de-risks the attractor machinery before the bigger build, and it is the only option
  that can plausibly hit the full V=320 spec on the first try (the operations are already validated at 320).

### Option B — a LEARNED transformation-binding (Vector-Derived Transformation Binding, Gosmann-Eliasmith 2019)

- **What it learns:** the **binding transform itself** — a linear transformation matrix (per role) that binds, learned
  to be efficient/robust rather than the fixed circular-convolution/phasor product. Unbind is the (learned/derived)
  inverse transform. Still needs a cleanup (Option A's attractor).
- **Spiking substrate:** the Neural Engineering Framework realises a learned/derived linear transform as a feedforward
  spiking population (a weight matrix between two ensembles); the project's nearest substrate is a plastic
  `RegionPathway` between two `BrainRegion`s, or the BPTT cortex trained to implement the transform. VTB is explicitly
  designed for spiking implementation and is **more dimension-efficient** than circular convolution (its headline
  claim) — relevant because the embedded-clause capability is the project's one dimension-floor degradation (needs
  D≥256 at V=320), so a more dimension-efficient bind could lift that floor.
- **Bind:** apply the learned per-role transform to the filler vector. **Unbind:** apply the inverse transform, then
  Option-A attractor cleanup.
- **Honest risk:** VTB still has an (approximately) invertible transform, so it only *partially* escapes I-1 — it is
  "learned but still algebra-shaped." It is a real step beyond Option A (the binding is now learned, not fixed) at
  moderate cost, but it inherits the same systematicity *gift* as VSA precisely because it stays transform-shaped, so
  it does **not** test whether a genuinely-learned readout generalizes (Option C does). Treat B as the "learned-but-safe"
  midpoint: more brain-like than A, lower-variance than C.

### Option C — a FULLY LEARNED mixed-selectivity binder + learned readout (the BPTT cortex)

- **What it learns:** *everything* — a high-dimensional **mixed-selectivity expansion** (neurons tuned to nonlinear
  role×filler conjunctions; catalog F.12 cerebellar expansion-recoding / Rigotti-Fusi mixed selectivity) followed by a
  **learned linear/plastic readout** trained (by surrogate-gradient BPTT, or a local three-factor rule) to map
  "(role cue, composite) → filler" and "(composite) → familiarity." No algebra is assumed; the bind is whatever the
  expansion+readout learns.
- **Spiking substrate:** the project's BPTT spiking cortex (`sim/bptt_snn_gpu.py`) as the learned readout, fed by an
  expansion layer (a random sparse projection into a larger LIF population — the `build_mf_gc_codon_layer` the catalog
  flags as the missing F.12 generator, here a few lines of fixed sparse connectivity, *not* a `sim/` edit). Training
  data = role-filler facts generated from the codebook; loss = recover the held-out filler.
- **Bind:** the composite is formed by *co-activating* role and filler into the expansion (their conjunctive
  mixed-selectivity pattern IS the bound representation — biologically faithful: there is no separate "bind operator,"
  binding is the joint population state). **Unbind:** drive the expansion with the composite + the role cue; the
  learned readout emits the filler (and the attractor cleans it). This is the most cortex-like: binding is a learned
  joint code, not an operation.
- **Honest risk (the big one):** **systematicity (Fodor-Pylyshyn).** A learned readout will tend to memorize the
  *trained* role-filler combinations and fail on *novel* ones ("dog go north" trained ⇒ "north go dog" or "cat go
  river" may not unbind). This is the documented, persistent failure mode of learned compositional binders (2025:
  "Fodor and Pylyshyn's Legacy — Still No Human-like Systematic Compositionality"; the Lake-Baroni meta-learning-for-
  compositionality counterexample is contested). Option C is where the arc could go **NEGATIVE**, and that negative is
  itself a scientific deliverable (it maps exactly where a learned substrate stops being systematic). It is also the
  most expensive (training a cortex to V=320 across roles is the heaviest GPU item). Recommend C **only after** A and
  B, and **gate it on the systematicity probe** (§7) so the negative is found cheaply.

### Ranking summary

| | Learns | Bind/Unbind | Removes | Cost | Systematicity risk |
|---|---|---|---|---|---|
| **A (first)** | codes + cleanup | fixed phasor + learned attractor | I-2, I-3 | low (reuses RF + CA3/NMDA + ZCA) | low (algebra kept) |
| **B** | the binding transform | learned transform + attractor | partial I-1, I-2, I-3 | moderate (NEF/plastic pathway) | low (transform-shaped) |
| **C** | everything | learned joint code + readout | I-1, I-2, I-3 | high (BPTT cortex to V=320) | **high — the core risk** |

**Recommendation:** ship **A** (dissolves the clean-code wall on existing machinery, plausibly hits full spec), then
**B** (learned binding, dimension-efficient, may lift the clause floor), and run **C as the explicitly-gated
systematicity experiment** — its honest negative is the deliverable that maps the learned-cortex limit. Every option
shares the §3 familiarity gate and the §1.2(b) attractor cleanup, so build those once (they are the reusable core).

---

## 3. THE NO-CONFAB FAMILIARITY GATE (load-bearing — design it)

The abstention moat is the hardest validated bar (100% = 20/20, multi-seed, V=320) and today it is **host
bookkeeping** (a Python `if no fact matches: return None`). A learned binder *cannot* keep it for free: a learned
readout will always emit *some* nearest filler, and a learned attractor will always relax to *some* stored pattern —
so on an **unknown** cue both would confabulate. The fix is the biological one: a **separate familiarity/novelty
detector** that gates the readout.

### 3.1 The mechanism — Bogacz-Brown anti-Hebbian familiarity (catalog D.04 CA1 match/mismatch)

- **Biology:** perirhinal cortex performs familiarity discrimination by **repetition suppression** — a population
  whose response is *high for novel* input and *suppressed for familiar* input. Bogacz-Brown (2003+) show an
  **anti-Hebbian** learning rule ("cells that fire together wire apart") realises this with **far higher capacity than
  Hebbian familiarity models specifically when inputs are CORRELATED** — which is *exactly* the project's case (the
  real `denoise64` codes are correlated). A 2025 result ("Continual familiarity decoding from recurrent connections in
  spiking networks") confirms familiarity is decodable from a recurrent **spiking** net — so this is buildable on the
  bridge, not just in rate models.
- **What it computes:** a scalar/low-D **novelty energy** N(cue). When a fact (or a concept code) has been stored,
  the gate's response to its cue is *suppressed* (familiar → low N). An unknown cue is *not* suppressed (novel → high
  N). Threshold N → abstain.

### 3.2 How it gates the moat (per query)

For "who/what/yes-no/describe(agent)" the query supplies a cue (the matched role-fillers, or the agent code):

1. Present the cue to the familiarity population (built on the bridge: a recurrent excitatory pool with
   anti-Hebbian-trained recurrent weights, imprinted on each `store`; equivalently the CA1 distal/proximal
   match/mismatch comparator of catalog D.04 — direct EC-III drive vs CA3-recalled pattern).
2. Read the population's novelty signal N. **If N > threshold (novel) → return abstain** (`None`/`"unknown"`) *before*
   running unbind/cleanup. **If N ≤ threshold (familiar) → run the binder** and return the recovered filler.
3. The threshold is set on a held-out calibration set (stored cues must read familiar; a disjoint set of unknown
   cues must read novel) — the same way the project sets every spiking operating point.

This makes abstention a **computed spiking signal**, not a host `if`: the brain *recognises* it has no memory and
stays silent. It also composes with all three binder options (A/B/C) unchanged — the gate is orthogonal to how bind
works.

### 3.3 Why this is the right design (and the honest risk)

- It is the *biologically correct* division of labour: recall (binder) and recognition (familiarity) are **separate
  systems** in cortex (perirhinal familiarity vs hippocampal recollection — the dual-process recognition-memory model).
  The project's own engram work already separates "store the pattern" from "recall the pattern"; this adds "recognise
  whether there is a pattern."
- **Honest risk:** the familiarity threshold is a *graded* signal, so abstention becomes a **detection problem with a
  false-positive/false-negative trade-off** — unlike the current exact `None`. The bar is 100% (20/20), which a graded
  detector must hit with *zero* false "I know it" on unknowns AND zero false "I don't know" on knowns. Anti-Hebbian's
  high capacity on correlated codes is the reason to expect this is reachable, but it is a real, measurable risk and is
  the first thing the de-risk (§5) must check. If the gate cannot hit 20/20 at the operating vocab, that is an honest
  negative that maps a substrate limit on the no-confab moat.

---

## 4. REUSABLE project machinery (what step 3 builds ON, not from scratch)

- **The RF phasor substrate** (`NeuronModel.RESONATE_AND_FIRE` + complex synapses: `rf_kick`,
  `rf_set_complex_weights`, `rf_resonate_steps`, `rf_read_phases`) — the spiking bind/unbind for Options A/B; already
  validated to FHRR parity. The `rf_kick(neuron_mask=)` masked-RF-ops edit (owner-approved, default-off byte-identical)
  lets the binder co-reside on a shared bridge.
- **The BPTT spiking cortex** — `sim/bptt_snn.py` (numpy reference + gradient-check), `sim/bptt_snn_gpu.py` (CuPy,
  validated == numpy at fp32), `sim/surrogate_grad.py` (ATan + fast-sigmoid surrogates), `sim/char_tokenizer.py`,
  `research/runners/cortex_pretraining.py` (`train_abc` + `train_shakespeare` + `save_checkpoint`/`load_checkpoint`).
  This is the learned readout for Option C, ready to retarget from next-char to role-filler recovery. (Headers say
  "path-f-hybrid only" — **stale**; all are on `main` with passing tests.)
- **The CA3 autoassociator / attractor primitives** — `RegionPathway` with `internal_density>0` (a `ca3→ca3`
  recurrent, catalog D.05), plus the slow-NMDA graded-attractor machinery already in the step loop
  (`bridge.py:5751`, Wang 2002). The learned cleanup (§1.2b) is built from these (catalog flags D.13 as "missing —
  no runner builds it yet," so this is also a catalog-closing deliverable).
- **The engram-tag API** (`bridge.py:3048-3225`: `start_engram_recording` / `commit_engram_tag` /
  `stimulate_tag` / `get_engram_tag_indices` / ...) — a *codebook-free* store ("the pattern that fired IS the
  memory"). Two uses: (i) the substrate-held fact store (an alternative to the numpy `kb`, already prototyped as
  `enable_substrate_store`/`enable_spiking_memory`); (ii) the mechanism that lets a *perceived* (grounded) ensemble be
  stored without a phasor code — the bridge that the functional-integration arc (perception→memory) needs and that
  step 3 generalises.
- **The validated REAL codes** — `denoise64_seed{N}.npz` (the brain's own concept-pool activity), `load_concepts()`,
  the `decorrelate=`/ZCA whitening, and the concept-pool training runners — the lossy correlated codes the learned
  cortex must read, plus the existing way to *learn-decorrelate* them.
- **The capability spec + probe harness** — `research/runners/vocab_ceiling_probe.py` (the full capability matrix
  scored pass/fail with the two anti-cheats per cell). Step 3 must pass THIS, verbatim, on the learned binder.
- **The composer's conversational API + the agent** — `BrainConversationalAgent` / `BridgeParser`
  (`research/runners/brain_conversational_agent.py`) delegate all storage/retrieval to a *composer object* via a
  fixed interface (`store`, `query_agent`, `query_patient`, `ask_yes_no`, `render_fact`, `elaborate`). A learned binder
  that implements this interface is a **drop-in** (the parser + dlPFC are composer-agnostic) — so step 3 is "write a
  new composer class," not "rewire the agent."

---

## 5. THE CHEAP-FIRST DE-RISK (the smallest test before any full build)

**The single load-bearing question:** can a LEARNED binder do **one bind → unbind on the project's REAL `denoise64`
concept codes**, AND **abstain on an unknown**, at the operating vocab — before building anything bigger?

**Probe (CPU-cheap, minutes; one seed):**
1. Load REAL codes: `load_concepts(seed=42, proj_dim=…)` → the correlated `denoise64` codes (NOT random-clean). This is
   the crux: the probe must use the *brain's* codes so the clean-code demand is actually stressed.
2. **Option-A core (recommended de-risk):** bind two stored facts with the unchanged FHRR phasor product; unbind one
   role; feed the noisy estimate into a **Hopfield/CA3 attractor cleanup** whose recurrent weights are the Hebbian
   outer product of the stored codes; check the cleanup recovers the correct filler. **Baseline control:** the same
   unbind with the *fixed `argmax` cleanup* — measure the accuracy gap (attractor should match or beat `argmax` on
   correlated codes; if not, Option A is not buying anything and we re-rank). This is a few matrix ops + a short
   relaxation; no BPTT, no GPU.
3. **The familiarity gate, same probe:** store fact F; build the anti-Hebbian familiarity pool; present F's cue
   (familiar) and a **never-stored** cue (novel); confirm N(F-cue) < threshold < N(unknown-cue) — i.e. the gate would
   *answer* on the known and *abstain* on the unknown. One stored + one unknown is enough to see the separation; the
   full 20/20 is the later gate.
4. **The lesion-in-the-probe (anti-cheat seed):** zero the attractor's recurrent weights (or the familiarity pool's
   trained weights) → cleanup degrades / abstention collapses → confirms the behaviour rides the learned weights, not
   leakage.

**Gate to proceed:** on REAL codes, the learned attractor cleanup recovers the filler ≥ the `argmax` baseline AND the
familiarity gate separates known-vs-unknown with a clean threshold. **Only then** build the full learned-binder
composer and run the §6 anti-cheats + the §2-ordered options.

**CPU vs GPU:** the de-risk is **CPU** (`SIM_BACKEND=numpy` is fine — a few hundred-D matrix ops + a short attractor
relaxation + an anti-Hebbian outer product). **GPU (`SIM_BACKEND=cupy`)** is needed only for (i) the full
`vocab_ceiling_probe` at V=320, and (ii) Option C's BPTT cortex training (the heavy item). This mirrors how the
project de-risks every spiking mechanism cheaply before committing GPU (the cleanup-NEF, substrate-store, and
functional-integration arcs all did the CPU-probe-first move).

---

## 6. ANTI-CHEAT controls (the moat preserved; not memorizing; systematicity tested)

1. **The no-confab moat preserved (the primary bar).** Re-run `vocab_ceiling_probe`'s abstention cell **verbatim**:
   ≥ 20/20 unstored cues must abstain, *and* the shuffled-fact permuted control (wrong-queries must abstain) must
   give zero false hits — multi-seed (42–47), at V up to 320. A learned binder that confabulates on unknowns FAILS,
   full stop. This is now carried by the **familiarity gate** (§3), so this control is also the gate's acceptance test.
2. **Not just memorizing facts (held-out test).** Train/imprint the binder on a set of facts; **query a held-out fact
   it was given but with a different surface form / a held-out (agent, action) pair**; the recovered filler must be
   correct. A binder that only echoes the exact training trace (a lookup table) FAILS. (Distinguishes "stored the
   fact" from "can bind/unbind it.")
3. **SYSTEMATICITY on NOVEL role-filler combinations (the deepest control).** Train the binder so that, e.g., "dog,"
   "cat," "go," "north" each appear in *some* training facts, but the specific combination **"cat go north" is NEVER
   trained**. Then store "cat go north" at test time and query it. A *systematic* binder handles it (it composes the
   parts it knows); a *memorizing* learned readout fails. Score the fraction of novel combinations correctly
   unbound vs a chance baseline and vs the algebra (which is 100% systematic by construction). **This is the control
   that exposes the Fodor-Pylyshyn risk** and the one Option C must pass to be more than a lookup table.
4. **Provenance / brain-based audit.** The bind/unbind/cleanup/familiarity must all be neuron firing + synaptic
   current on the bridge; the host is legitimate only for presenting the sentence (the environment) and reading the
   final answer. Grep the composer for any numpy *computation* of the match, the bound vector, the cleanup argmax over
   a god's-eye codebook, or the abstention `if` — each must be replaced by a spiking readout (the project's standing
   "BRAIN-BASED ONLY" bar; the current composer's residual numpy ops are exactly what step 3 removes).
5. **Lesion the learned weights → capability collapses.** Zero the learned readout / attractor / familiarity weights
   and confirm the capability and the moat both collapse — proves the behaviour is carried by the learned substrate,
   not by leftover structure or a host path.

---

## 7. HONEST could-be-NEGATIVE — SYSTEMATICITY is the core known risk

**The arc can go NEGATIVE, and the most likely place is Option C's systematicity (anti-cheat #3).**

- **Why it is the core risk.** The composer's clean algebra has systematicity *for free*: bind/unbind are the same
  operation for any operands, so "cat go north" works whether or not it was ever trained. A **learned** binder gets
  systematicity only if it generalizes to novel combinations — and the 2025 literature is blunt that this is the
  *persistent unsolved* failure of learned compositional networks ("Fodor and Pylyshyn's Legacy — Still No Human-like
  Systematic Compositionality in Neural Networks," 2506.01820). The Lake-Baroni "meta-learning for compositionality"
  result (Nature 2023) is the main counter-claim, but it is contested and required a bespoke meta-training regime — not
  evidence a vanilla learned spiking readout will be systematic. **So the expected honest negative is: Option C
  unbinds trained combinations but degrades on novel ones, while Options A/B (which keep the algebra/transform) stay
  systematic.** That negative is *itself the deliverable* — it maps precisely the boundary "a learned cortical readout
  trades the algebra's free systematicity for the ability to read lossy codes," which is the exact trade CLAUDE.md
  already names ("the algebra buys the no-confab moat + compositional reliability ~free; a learned cortex does not").

- **The cheap test that exposes it EARLY.** Do **not** wait for a full V=320 BPTT train to discover this. The
  systematicity control (#3) is runnable at **toy scale on a handful of facts on CPU**: train on N−1 of the N
  combinations of a tiny role-filler grid, hold one out, test it. If the learned binder fails the held-out combination
  at toy scale, it will fail at full scale — *stop and report the negative* instead of spending GPU on a cortex that
  cannot generalize. Run this control on Option C **before** any large training, immediately after the §5 de-risk
  passes. (Options A/B should pass it trivially because they keep the systematic operation; if they don't, that is a
  bug, not a science result.)

- **Secondary negatives, each a mapped limit, not a surprise:**
  - **The familiarity gate may not reach 20/20** (a graded detector vs an exact `None`; §3.3). Reachable in principle
    (anti-Hebbian's high capacity on *correlated* codes is the whole reason to expect it), but if not, it is an honest
    bound on the no-confab moat under a learned substrate — caught by the §5 de-risk's known-vs-unknown separation.
  - **The learned attractor cleanup may not beat `argmax` on these particular codes** — then Option A buys nothing and
    we re-rank toward B/C (caught by the §5 baseline control).
  - **The embedded-clause dimension floor** (D≥256 at V=320) may move under a learned binder — possibly *better*
    (VTB's dimension efficiency, Option B) or *worse* (a learned readout may need more dimensions for recursion). Either
    way it is the one already-characterized degradation; track it as the clause cell of the spec.

- **Net honest framing.** Options A/B are expected to *work* (they keep a systematic operation and only learn the
  codes/cleanup/transform) and to dissolve the clean-code wall + the cross-code cleanup half on existing machinery.
  Option C — the *genuinely* learned binding — is where the project's deepest open question lives, and the disciplined
  move is to **probe its systematicity at toy scale first** so the likely negative is found in minutes, mapped, and
  reported as the scientific deliverable it is, rather than discovered after a large GPU train. "Complete the
  functional cortex" succeeds either as a learned binder that matches the spec (A/B, maybe C) **or** as the precise,
  controlled negative that says where a learned spiking cortex stops being systematic — and that map is what motivates
  whatever comes after step 3.

---

## 8. SEQUENCING (build order, every step reuse-by-import; any `sim/` edit additive + default-off + byte-reviewed)

1. **§5 CPU de-risk** — learned attractor cleanup + familiarity separation on REAL `denoise64` codes, vs the `argmax`
   and host-`if` baselines. Gate: attractor ≥ argmax; gate separates known/unknown. *(minutes, CPU)*
2. **Build the familiarity gate** (§3) to the 20/20 moat bar at the operating vocab; it is shared by all binders, so
   build it once. *(the load-bearing piece)*
3. **Option A composer** (learned codes + attractor cleanup, fixed phasor bind) implementing the
   `BrainConversationalAgent` composer interface; pass `vocab_ceiling_probe` + the §6 anti-cheats, multi-seed to V=320.
4. **Option B** (learned transformation binding, VTB) — the learned *binding*; check the clause dimension floor.
5. **Option C systematicity probe FIRST (toy, CPU)** — anti-cheat #3 on a tiny grid *before* any BPTT train; if it
   fails, **report the negative** (the systematicity boundary) and stop the C build. If it passes toy-scale, train the
   BPTT-cortex readout and run the full spec. *(GPU only here)*
6. **Drop-in + the functional-integration payoff** — the winning learned binder reads correlated/grounded codes, so
   the perception→memory cross-code wall (the functional-integration arc's deferred (B)) becomes tractable: a
   *perceived* rate/grounded ensemble can be bound into memory by the learned readout. Closing that loop is the
   concrete demonstration that step 3 dissolved the rate-vs-phasor wall.

---

## Sources (literature consulted, beyond the in-repo catalog/code)

- Crawford, Gingerich, Eliasmith — *Biologically Plausible, Human-Scale Knowledge Representation* (Cognitive Science,
  2016): https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12261 — learned, robust VSA-on-spikes at human-scale
  vocabulary; the cleanup is the load-bearing associative memory.
- Gosmann & Eliasmith — *Vector-Derived Transformation Binding* (2019):
  https://compneuro.uwaterloo.ca/files/publications/gosmann.2019b.pdf — a **learned** binding transform for spiking
  nets, more dimension-efficient than circular convolution (Option B).
- Frady, Kent, Olshausen, Sommer — *Resonator Networks* (Neural Computation, 2020):
  https://direct.mit.edu/neco/article/32/12/2332/95653 — superposition + clean-up memory; the clean-up step is a
  Hopfield net with outer-product Hebbian weights (Option A's attractor cleanup; spiking/neuromorphic variants exist).
- Bogacz & Brown — *An Anti-Hebbian Model of Familiarity Discrimination in the Perirhinal Cortex*:
  https://www.mrcbndu.ox.ac.uk/sites/default/files/pdf_files/An%20anti-Hebbian%20model%20of%20familiarity_0.pdf —
  high-capacity familiarity discrimination *when inputs are correlated* (the no-confab gate, §3).
- *Continual familiarity decoding from recurrent connections in spiking networks* (bioRxiv, 2025):
  https://www.biorxiv.org/content/10.1101/2025.01.13.632765v1.full — familiarity is decodable from a recurrent
  **spiking** net (the gate is buildable on the bridge).
- *Fodor and Pylyshyn's Legacy — Still No Human-like Systematic Compositionality in Neural Networks* (arXiv
  2506.01820, 2025): https://arxiv.org/abs/2506.01820 — the live systematicity challenge (§7 core risk).
- Lake & Baroni — *Human-like systematic generalization through a meta-learning neural network* (Nature, 2023):
  https://www.nature.com/articles/s41586-023-06668-3 — the contested counter-claim that learned nets *can* be
  systematic with the right meta-training.
