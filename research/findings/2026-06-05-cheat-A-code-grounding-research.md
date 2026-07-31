---
type: finding
status: contributing
date: 2026-06-05
mechanism: grounding
---

# Cheat-A (RF-composer concept-code grounding) — biology-grounded conversion research — 2026-06-05

**The shortcut.** The FHRR-on-bridge conversational composer (`research/runners/rf_phasor_composer.py`,
`RFPhasorComposer`, lines 66–73) assigns each concept *and* role a **random phase vector**
`rng.uniform(0,1,D)`. The composition on top — bind / unbind / bundle — is now *genuinely spiking* on the
`SimulationBridge` resonate-and-fire substrate (`rf_kick` / `rf_set_complex_weights` / `rf_resonate_steps` /
`rf_read_phases`, `sim/bridge.py:4887–4975`). The remaining non-biological piece is the **codes themselves**:
they are *given*, not grounded in sensory input nor learned from experience. This is cheat-removal **#4** in
`2026-06-04-pure-biology-cheat-removal-backlog.md`, scoped specifically to the RF composer.

**This document's verdict in one line.** The grounding is **TRACTABLE and largely already de-risked** for the
*mechanism*, because the project did most of this work for the rate-coded / numpy-algebra agents in early June.
The honest, novel deliverable here is (a) tying that work to the textbook biology with citations, (b) replacing
the two residual numpy shortcuts inside the *existing* grounding pipeline (a **random** V1→phasor projection and a
**numpy ZCA** decorrelation) with **on-bridge, biologically-local** equivalents, and (c) the smallest test that
proves the RF composer specifically runs on grounded codes. The realistic best outcome is **constructed-parity
composition on grounded codes for the visually/lexically-groundable vocabulary, with a documented abstract-concept
boundary** — not a universal "every concept emerges from raw natural images" claim.

---

## 1. The biology mechanism (cited)

The brain does not assign concept codes randomly. A concept code is the **convergent activity at the top of a
sensory hierarchy, sculpted by experience-dependent plasticity into a sparse, decorrelated, partly-invariant
representation.** Five mechanisms, each citable in the project's own catalog + the textbook + the primary
literature, combine to produce that:

### 1a. Sensory hierarchy → IT-level object codes (the convergence)
- **Catalog:** `docs/biology.md` "How the eye works (vision)", lines 113–165: retina (32×32 ON/OFF) → V1 Gabor
  simple cells → V1 complex (orientation-invariant pooling) → V2 (corners/junctions) → **IT (full object
  recognition; sparse-distributed concept representations)**. The catalog already names the right endpoint and
  the right citation (Quian Quiroga 2005).
- **Textbook (read directly):** Kandel 6e **Ch 24 "High-Level Visual Processing: From Vision to Cognition"**, PDF
  pp. 609–626 (book pp. 564–581). Key facts pulled from the actual pages:
  - The ventral stream is a strict convergent hierarchy V1→V2→V4→IT; "neurons at each synaptic relay receive
    convergent input from the preceding stage" and IT "integrate[s] a large and diverse quantity of visual
    information over a vast region of visual space" (PDF p. 610).
  - IT neurons "encode complex stimulus features", have "large, centrally located receptive fields", and are
    **organized in functionally specialized columns (~400 µm)** that **partially overlap** so that "one stimulus
    can activate multiple columns", forming "distributed networks for encoding objects" (PDF p. 613–614). This is
    *exactly* a sparse-distributed (K-of-N) code — the same form the project's G.20 sparse-pool work uses.
  - **Perceptual constancy** (size / position / form-cue invariance — Fig 24-7, PDF p. 617) and **categorical
    perception** (the "apples in a basket / letter A in many fonts" example; the Freedman–Miller dog-cat morph,
    PDF p. 618) are built *up the hierarchy* — invariance is the hierarchy's product, not an input property.
- **Primary literature:** Quian Quiroga, Reddy, Kreiman, Koch & Fried (2005), *Nature* 435:1102–1107 — single MTL
  neurons with **invariant, sparse, explicit** responses to a concept (the "Jennifer Aniston neuron"); the
  follow-up Waydo/Quian-Quiroga (2006) *J. Neurosci.* 26:10232 quantifies the **sparseness**. This is the
  biological target representation: a sparse code over a high-D population, invariant to the sensory particulars.

### 1b. Codes are LEARNED, not wired (experience-dependent sharpening)
- **Textbook:** Kandel 6e Ch 24, "Implicit Visual Learning Leads to Changes in the Selectivity of Neuronal
  Responses" (PDF p. 618): IT selectivity for complex objects is "highly modifiable by experience"; training on
  novel objects makes IT neurons "selective for those objects", "manifested as a **sharpening of neural
  selectivity** rather than changes in absolute firing rate". "Learning can generate entire areas of functional
  specialization within inferior temporal cortex" (Logothetis wire-form study, p. 618). → **The code for a concept
  is earned by experience-dependent plasticity, which the bridge already models (STDP + homeostatic sharpening).**
- **Catalog:** `docs/biology.md` "How the brain learns (plasticity)", lines 239–328 — STDP (Bi & Poo 1998; Song
  et al. 2000), three-factor reward-gated plasticity (Schultz 1998; Frémaux & Gerstner 2016). These are the local
  rules that sharpen the codes.

### 1c. Sparse coding from natural statistics (why the codes look like Gabors + are sparse)
- **Olshausen & Field (1996),** *Nature* 381:607–609: a learning algorithm that maximizes **sparseness** of a
  linear code for natural images *develops* localized, oriented, bandpass receptive fields — i.e. V1 Gabor RFs are
  not innate magic, they are what an **unsupervised sparse-coding objective** yields from natural image statistics.
  This is the principled reason the project's Gabor V1 bank (`sim/visual_cortex.py`) is the right front-end, and
  the reason a *learned* sparse code (not a fixed random projection) is the biologically-correct way to form the
  higher code.

### 1d. Decorrelation / efficient coding (why the hierarchy whitens — the key to composability)
- **Atick & Redlich (1992),** *Neural Computation* "What does the retina know about natural scenes?": natural
  scenes are highly correlated; the retina/early-vision job is to **decorrelate (whiten)** the signal — produce a
  representation with reduced redundancy and higher statistical independence. Validated experimentally for RGCs
  (Pitkow & Meister 2012, *Nat. Neurosci.* 15:628).
- **Why this is load-bearing here:** the project's own 2026-06-04 finding
  (`2026-06-04-v-multimodal-grounding-decorrelation-unifies.md`) measured that a *single* V1 Gabor layer leaves
  high inter-code coherence (near-duplicate stimuli) that the FHRR resonator **cannot factor** during composition —
  attribute composition collapsed to 0% on raw V1 codes. **Decorrelation (the ventral hierarchy's efficient
  coding) is what turns grounded-but-coherent codes into composition-ready low-coherence codes.** This is the
  single most important biological fact for making grounding *usable* by the composer.

### 1e. The biologically-LOCAL way to get a sparse decorrelated code (the on-bridge recipe)
- **Földiák (1990),** *Biol. Cybernetics* 64:165–170, "Forming sparse representations by local anti-Hebbian
  learning": a layer of **Hebbian feed-forward** units + **anti-Hebbian (decorrelating) lateral inhibition**
  learns a **sparse, decorrelated** code by *purely local rules* — "a biologically plausible alternative to PCA".
  This is the keystone: it means the numpy ZCA whitening step in the project's current pipeline has a **local,
  on-bridge replacement** the bridge *already has the parts for* — STDP feed-forward + PV-FS lateral inhibition
  (the project's "PV-FS lateral inhibition between pools per Vogels 2011" already used in the concept-pool work).
- **Pulvermüller (1999, 2001, 2003)** — distributed action-word **cell assemblies** formed by **Hebbian
  correlation learning**: a word's assembly contains neurons in language cortex *and* in the sensory/motor areas
  for its meaning, recruited because they co-fire during acquisition. Catalog: `docs/biology.md` "How the brain
  learns words", lines 329–378. → For **abstract/non-visual** concepts (verbs, function words) with no canonical
  image, the grounding is **multimodal co-occurrence Hebbian binding** (hear "go" while the motor pool fires),
  not vision. This is the principled answer to the abstract-concept gap.
- **Bellmund, Gärdenfors, Doeller & Behrens (2018),** *Science* 362:eaat6766, "Navigating cognition: spatial codes
  for human thinking": entorhinal **grid codes** generalize to **non-spatial / abstract conceptual spaces**
  (a concept = a position in a learned feature space). This is the biological license for treating *abstract*
  concept codes as points in a learned low-D "concept map" rather than demanding a sensory image — relevant if the
  abstract-vocabulary grounding is pushed beyond co-occurrence.

**Synthesis.** A biology-grounded concept code is: *the sparse, decorrelated, experience-sharpened activity at the
convergent top of a sensory (or multimodal) hierarchy.* Vision grounds visual concepts (V1→…→IT); multimodal
Hebbian co-occurrence grounds abstract ones (Pulvermüller); decorrelation (Atick-Redlich, realized by Földiák's
local Hebbian+anti-Hebbian) makes both composable. Every piece maps to machinery the bridge already has.

---

## 2. Concrete ON-BRIDGE spiking realization

**Reuse, don't reinvent.** The project already built and validated most of this pipeline for the numpy / spiking
*unified* agents. The honest job here is to (i) wire it to the **RF composer specifically**, and (ii) replace the
two residual numpy shortcuts with on-bridge biological steps.

### What already exists (validated, reuse-by-import)
| Piece | File | Status |
|---|---|---|
| Real V1 Gabor bank (retina→V1 simple, Hubel-Wiesel) | `sim/visual_cortex.py` (`build_v1_simple_weights`, `apply_v1_gabor_weights`) | shipped; on-bridge wiring helper exists |
| V1-grounding separability + cleanup probe | `research/runners/_visual_grounding_probe.py` | 12 concepts, mean cos 0.25, 97% cleanup |
| V1 features → phasor (FHRR) codes → bind/unbind/cleanup | `research/runners/_visual_grounded_composition_probe.py` | grounded codes compose **100% clean / 92% corrupted** |
| Ventral-hierarchy **decorrelation** stand-in (ZCA) | `research/runners/unified_agent_visual_grounded.py` (`_decorrelate`) | raw 0% attr → decorrelated 100% attr |
| Multimodal grounding (nouns→V1, verbs/adj→word encoder) + decorrelate → unified codebook | `research/runners/unified_agent_multimodal_grounded.py` | **78/78 = 100%, 2 seeds** |
| Spiking unified agent on V1-grounded codes (genuine spikes) | `research/runners/spiking_unified_agent_grounded.py` | **72/72 = 100%, 2 seeds** |
| Word-encoder (abstract-concept) grounding | `sim/text_embeddings.py` (`vocab_to_drive_pattern` SHA-256 sparse; `orthogonal_drive_pattern`) | the grounded word-cue level, validated |

So the *representation-level* question ("do grounded codes compose as FHRR phasors?") is **already GO**. What is
**not** yet done: feeding grounded codes into `RFPhasorComposer` (which composes on the **bridge's** RF neurons,
not numpy), and removing the two numpy shortcuts inside the grounding pipeline.

### The interface point (exact)
`RFPhasorComposer.__init__` builds `self.concepts = {w: rng.uniform(0,1,D) for w in words}` (line 68) and
`self.roles = {...}` (line 73). Everything downstream consumes `self.concepts[w]` as **phases in [0,1)^D** (then
`_to_phasor` = `exp(2πj·phases)`, line 94). **Grounding = replace that dict** with phases derived from grounded
on-bridge activity. The composer needs **no structural change** beyond accepting an injected codebook (add a
`concepts=` kwarg mirroring `CoreSimComposer`/`BrainConversationalAgent`'s existing `concepts` hook). The role
vectors stay random/fixed (roles are abstract relational slots — agent/action/patient — not sensory concepts;
they have no referent to ground, and randomness is the correct FHRR choice for them).

### The grounding pipeline, on-bridge (proposed, 4 stages)

```
 stage 0  per-concept stimulus
   visual concept  → distinct 32×32 ON/OFF image (sim/visual_cortex render helpers)
   abstract concept → its referent's drive (motor-pool drive for verbs; word-form pixels for lexical;
                      or word-encoder sparse code as the labelled fallback)

 stage 1  SENSORY TRANSDUCTION + FEATURE EXTRACTION  (ON THE BRIDGE, already real)
   image → retina region → V1 Gabor (apply_v1_gabor_weights) → [V2 → IT pools]   ← bridge spikes
   read per-concept population activity (spike counts over an encoding window) = the grounded feature vector g_c

 stage 2  SPARSE DECORRELATED CONCEPT CODE  (ON THE BRIDGE, Földiák-local — replaces numpy ZCA)
   an "IT/concept" pool with Hebbian feed-forward (STDP, lang/V1→pool) + anti-Hebbian PV-FS lateral inhibition
   between concept units → sparse, decorrelated sparse-distributed code  s_c (K-of-N active)   ← Földiák 1990
   (this is the on-bridge replacement for `_decorrelate`'s ZCA; biology = Atick-Redlich efficient coding
    realized by local rules, exactly the project's concept-pool + PV-FS recipe)

 stage 3  ACTIVITY → FHRR PHASE CODE  (the grounded→phasor bridge)
   φ_c[k] = angle( P · s_c ) / (2π)  mod 1      with P a FIXED projection  →  phases in [0,1)^D
   set RFPhasorComposer.concepts[c] = φ_c     (matches the composer's [0,1) convention; see §5)
```

**Stage 2 is the novel on-bridge contribution.** Today the project's *validated* pipeline does stage-1 V1 on the
bridge but stages 2–3 in numpy (ZCA + a random complex projection). The biologically-honest upgrade is to do the
**decorrelation in spikes** via a concept pool with anti-Hebbian lateral inhibition (Földiák) — the bridge already
has STDP and PV-FS lateral inhibition (used throughout the concept-pool / Tier-1 work), so this is **reuse, not a
`sim/` edit**. The output sparse code `s_c` is then projected to phases. This makes the *whole* path
"sensory→sparse-decorrelated→phasor" on-brain, leaving only the final fixed projection as a deterministic readout.

**On the residual "random projection" (stage 3).** Even after stage 2, `P` is a fixed matrix. Two honest options,
in increasing biological fidelity / cost:
1. **Fixed random `P` (cheapest, already validated form).** `P` is a *deterministic readout* (like a fixed
   random hashing of a sparse pattern to phases) — the code is still a *fixed function of grounded activity*, so
   it is grounded, not free. This is what `_visual_grounded_composition_probe.py` does and it composes 100%/92%.
   Disclose `P` as a fixed readout (not learned).
2. **Phase = identity-of-active-unit (no `P` at all).** Assign each of the N concept-pool units a fixed phase
   offset; `φ_c` = the (soft) phase histogram of `s_c`'s active units. Then the phase code is *literally* the
   sparse pattern, zero free parameters. This is the most defensible and worth testing in the de-risk.

### Why this is "genuinely on the bridge"
- Stage 1 (transduction + V1 features) runs in the bridge's own step (retina→V1 regions, real Gabor weights).
- Stage 2 (sparse decorrelation) runs as a bridge region with STDP + PV-FS inhibition (Földiák in spikes).
- Stage 3 is a fixed readout of bridge activity into the phase format the **RF composition substrate already
  consumes on the bridge** (`rf_kick` takes the phasors; bind/unbind/bundle resonate on the bridge's RF neurons).

So the *only* numpy left is the deterministic `angle(P·s)` readout (stage 3 option 1) or nothing (option 2) — a
strict improvement over the current pipeline's numpy ZCA + random complex projection.

---

## 3. Smallest DE-RISK test (prove or falsify, cheaply, before any build)

**Claim to test:** *Sensory/lexically grounded codes, decorrelated on the bridge, drive the **RFPhasorComposer**
(on-bridge RF neurons) to compose at parity with its random codes, and abstain correctly.*

**The de-risk (one probe, `research/findings/raw/_rf_composer_grounded_derisk.py`, `SIM_BACKEND=numpy`):**
1. Take the composer's default vocab (V=17, `DEFAULT_VOCAB` in `rf_phasor_composer.py`). For each **visually-
   groundable** word, render a distinct stimulus → real V1 Gabor (`build_v1_simple_weights`) → V1 feature; for
   each **abstract** word, use the word-encoder sparse code (`vocab_to_drive_pattern`) — exactly the multimodal
   split already validated in `unified_agent_multimodal_grounded.py`.
2. **Decorrelate** (start with the validated ZCA `_decorrelate` to isolate the composer question; the Földiák
   on-bridge version is the *follow-up* once parity is shown — don't conflate the two unknowns).
3. Project to phases `φ_c = angle(P·s_c)/(2π) mod 1` (test stage-3 option 2, the no-`P` phase-histogram form,
   alongside option 1).
4. Inject: construct `RFPhasorComposer(...)` then **overwrite `composer.concepts`** with `{c: φ_c}` (and add the
   `concepts=` kwarg). Run the composer's **own** conversational API against a small frozen fact set:
   `store` SVO facts → `query_agent` / `query_patient` / `ask_yes_no` / `render_fact`, plus an **abstention**
   probe (query an unstored agent → must return `None`).
5. **Pass criterion:** grounded-code accuracy ≥ random-code accuracy on the same facts (constructed parity) on
   the core ops (who/what/yes-no/render) for the groundable subset, **and** abstention preserved (the no-confab
   moat must survive grounding — a grounded code that breaks abstention is a real negative). Multi-seed (≥3).

**Why this is the right minimal test:** it changes *only* `composer.concepts` and runs the **RF composer's actual
on-bridge composition** (`rf_resonate_steps`), so a pass isolates "grounded codes work on the production RF
substrate" from the already-answered numpy-algebra question. It reuses every validated helper; the only new code
is ~40 lines of glue. Expected runtime: minutes on CPU (the composer caches RF bridges per op).

**Falsification:** if grounded codes drop core-op accuracy or break abstention even after decorrelation, the
grounding is *not* a drop-in for the RF composer — a genuine finding (likely cause: the RF phase readout amplifies
residual code coherence that the numpy `vdot` cleanup tolerated; the fix/limit would be characterized, not hidden).

---

## 4. Honest difficulty assessment

**Rating: TRACTABLE for the mechanism + groundable vocab; PARTIAL for "the whole vocabulary from raw sensory
input"; the fully-on-bridge decorrelation is a bounded sub-arc, not a multi-month rewrite.**

Tier-by-tier honesty:

- **Representation-level grounding (V1/word → phasor → compose): already GO.** Not speculative — the project
  measured 100%/92% composition on V1-grounded phasor codes and 100% multimodal-grounded on both numpy and the
  *spiking* unified agent. The de-risk above is expected to **pass** for the groundable subset; if it does, the
  RF-composer grounding is a **days-scale wire-up**, not a months arc.

- **On-bridge decorrelation (Földiák, replacing numpy ZCA): tractable but the real new work.** The bridge has the
  parts (STDP + PV-FS lateral inhibition), and the project has *used* this exact recipe (concept pools), but
  achieving ZCA-quality decorrelation from local anti-Hebbian rules at 320-concept scale is the one place a
  documented negative is plausible (local rules approximate, not equal, ZCA; the project's own P5/concept-pool arc
  showed selectivity from local rules is seed-fragile at scale). Honest expectation: **partial** — local
  decorrelation will *help* (raw→better) but may not fully reach ZCA's composability at 320; the labelled ZCA
  stand-in may remain the disclosed boundary, exactly as the cheat-4 agent-integration finding already disclosed.
  This is a **bounded sub-arc (weeks)**, and a negative here is itself a deliverable.

- **Abstract concepts from raw sensory input: NOT fully achievable — and that's biology, not a bug.** Verbs,
  function words, and most adjectives have **no canonical image** (the embodied-cognition limit; Kandel Ch 24's
  IT is *visual* object recognition). The biologically-correct grounding for them is **multimodal co-occurrence
  Hebbian binding** (Pulvermüller) — "go" grounds in the motor pool firing, not in a picture. The project's
  validated answer (word-encoder for the abstract block + decorrelation to unify) is honest but the word-encoder
  is itself a *given* code, so abstract grounding is **at best "grounded in the motor/lexical referent", not in
  raw sensation.** Claiming otherwise would be overreach.

- **"Genuinely emergent from natural images" (Olshausen-Field end-to-end, learned V1 from natural scenes):
  out of scope / multi-month.** The project uses *pre-tuned* Gabor V1 (the catalog already discloses this skips
  the developmental sparse-coding phase, biology.md lines 141–144). Learning V1 from natural-image statistics on
  the bridge is a separate, large research arc and is **not required** for code grounding — the pre-tuned Gabor
  bank is a legitimate, disclosed simplification of the *same* sparse-coding endpoint (Olshausen-Field is the
  *justification* that Gabors are what sparse coding yields).

**Realistic best outcome:** *the RF composer runs its full conversational capability matrix on grounded codes —
visual concepts via on-bridge V1 (+ on-bridge Földiák decorrelation where it reaches parity, labelled-ZCA where it
doesn't), abstract concepts via multimodal co-occurrence / word-encoder — at constructed parity on the groundable
subset, with the abstract-from-raw-sensation limit and any residual decorrelation gap explicitly disclosed.* That
is a real removal of the "random `rng.uniform` codes" shortcut for the RF composer, with the honest residue named.

**Worst plausible outcome (and it's still a finding):** the RF phase readout proves more coherence-sensitive than
the numpy `vdot` cleanup, so grounded codes need *stronger* decorrelation than the bridge's local rules deliver at
320, and the labelled-ZCA stand-in stays load-bearing. The "fully-on-bridge grounded composition at 320" then
remains the disclosed boundary — characterized, not hidden.

---

## 5. Interface with the FHRR phasor composer (exact, load-bearing)

The composer is interfaced **only through its concept codebook**; everything else (bind/unbind/bundle/cleanup,
the no-confab abstention, dialogue planning) is unchanged and substrate-correct.

1. **Inject point.** Add a `concepts=` kwarg to `RFPhasorComposer.__init__` (mirroring `CoreSimComposer` and
   `BrainConversationalAgent.__init__`, which already accept `concepts`). When provided, set
   `self.concepts = {w: np.asarray(concepts[w], float) for w in words}` instead of `rng.uniform`. **Roles stay
   random** (abstract relational slots, no referent — correct FHRR choice). The agent passes it through:
   `BrainConversationalAgent(..., concepts=grounded_codes)` → `RFPhasorComposer(..., vocab=..., concepts=...)`.

2. **Format contract (the load-bearing detail).** The RF composer represents a code as **phases in [0,1)^D**;
   `_to_phasor(phases) = exp(2πj·phases)` (line 94) and `_cleanup` compares with
   `cos(2π·(rec − concepts[w]))` (line 150). The existing visual-grounding probe instead produces phasors via
   `exp(1j·angle(P·v1_code))` — **angles in [−π, π]**. So the grounded→composer conversion **must** be
   `φ = (angle(P·s) / (2π)) mod 1` to land in [0,1). Getting this wrong silently halves/rotates every phase (a
   "complex-vs-phase format bug" of exactly this kind was already caught once in the cheat-4 agent-integration
   work — see backlog #4). The de-risk must assert `0 ≤ φ < 1`.

3. **Dimension.** The agent constructs the RF composer at **D=128** (`brain_conversational_agent.py:173`); the
   grounding projection `P` (or the phase-histogram readout) must output **D=128** phases. (The standalone probes
   used D=2048; D=128 is the production agent's value — the de-risk should run at D=128 to match.)

4. **What does NOT change.** `store` / `query_agent` / `query_patient` / `ask_yes_no` / `render_fact` / `elaborate`
   are untouched. `_cleanup` still does phase-cosine argmax over `self.concepts` — now over **grounded** codes, so
   cleanup snaps a noisy unbind to the nearest **grounded** concept (the CA3/IT pattern-completion the catalog and
   the cheat-4 finding both invoke). Abstention (`query_*` returns `None` when no stored fact matches) is a
   *structural* property of the KB-match loop, independent of how codes are formed — but the de-risk **must** still
   verify it survives, because a grounded code with higher inter-code coherence could cause a spurious match.

5. **The composition substrate is already the bridge.** Once `self.concepts` holds grounded phases, `_bind` /
   `_bundle` / `_unbind_phases` call `_resonate` → `rf_set_complex_weights` + `rf_kick` + `rf_resonate_steps` +
   `rf_read_phases` on the bridge's resonate-and-fire neurons (`sim/bridge.py:4887–4975`). So **grounded codes +
   on-bridge spiking composition** compose end-to-end with no `sim/` edits — the grounding feeds the *existing*
   on-bridge FHRR machinery exactly as random codes do today.

---

## Files & citations referenced
- Shortcut: `research/runners/rf_phasor_composer.py` (lines 66–73, 94, 150), `research/runners/brain_conversational_agent.py` (151–173).
- On-bridge RF substrate: `sim/bridge.py:4887–4975` (`rf_kick`/`rf_set_complex_weights`/`rf_resonate_steps`/`rf_read_phases`).
- Reuse: `sim/visual_cortex.py`; `sim/text_embeddings.py`; `research/runners/_visual_grounding_probe.py`,
  `_visual_grounded_composition_probe.py`, `unified_agent_visual_grounded.py`, `unified_agent_multimodal_grounded.py`,
  `spiking_unified_agent_grounded.py`.
- Prior findings: `2026-06-04-cheat4-visual-grounding-cheap-first-RESOLVES.md`,
  `2026-06-04-cheat4-visual-grounding-agent-integration.md`,
  `2026-06-04-v-multimodal-grounding-decorrelation-unifies.md`,
  `2026-06-04-spiking-plus-grounding-unification.md`,
  `2026-06-02-input-side-fidelity-grounding-data-efficiency-VALIDATED.md`,
  `2026-06-04-pure-biology-cheat-removal-backlog.md` (#4), `2026-06-05-FHRR-pivot-derisk.md`.
- Catalog: `docs/biology.md` lines 113–165 (vision), 239–328 (plasticity), 329–378 (language).
- Textbook (read directly): Kandel 6e Ch 24 "High-Level Visual Processing", PDF pp. 609–626 (book pp. 564–581) —
  IT convergence (p. 610), IT columns/hypercolumns/partial-overlap distributed codes (pp. 613–614), perceptual
  constancy Fig 24-7 (p. 617), categorical perception + Freedman-Miller (p. 618), implicit-learning selectivity
  sharpening (p. 618). Extracted to `research/findings/raw/_kandel_ch24_it.txt`.
- Papers: Olshausen & Field 1996 *Nature* 381:607; Atick & Redlich 1992 *Neural Comput.*; Földiák 1990
  *Biol. Cybern.* 64:165; Quian Quiroga et al. 2005 *Nature* 435:1102 + Waydo et al. 2006 *J. Neurosci.* 26:10232;
  Pulvermüller 1999/2001/2003 (Hebbian word cell assemblies); Bellmund et al. 2018 *Science* 362:eaat6766.
