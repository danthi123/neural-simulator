# B1 — Self-organizing V1 simple-cell receptive fields on-substrate (scoping, 2026-06-21)

**Type:** read-only deep-research + scoping (this doc is the only write).
**Trigger:** the definitive shortcut inventory (`2026-06-21-shortcut-inventory-definitive.md`, `ddc3b8db`) surfaced a
previously-untracked **criterion-2 (hardware-port) structure residual**: the V1 simple-cell receptive-field (RF)
read-out **weights are host-designed** (`sim/visual_cortex.py:build_v1_simple_weights` — hand-built Gabor filters
computed in Python and injected as fixed synapse weights). The operation (V1 filtering → spikes) runs on-substrate; the
*structure* (the Gabor weights) is host-computed → "spiking at runtime, host-designed structurally," which breaks a clean
neuromorphic-hardware port (a chip would need a host to compute + inject the Gabor bank). Per the owner's standing bar
(fully-spiking end-to-end + hardware-compatible; memory `feedback_spiking_structure_must_self_organize`).

**Definitions (once).** *Receptive field (RF):* the spatial pattern of input weights onto a sensory neuron — what
image pattern makes it fire. *Gabor filter:* an oriented sinusoid under a Gaussian envelope; the standard linear model of a
V1 simple cell's RF (Hubel-Wiesel 1962). *Self-organize:* the weights EMERGE from a local plasticity rule + input
statistics, rather than being computed by a host formula. *STDP:* spike-timing-dependent plasticity (the project's
spike-order Hebbian rule). *Sparse coding:* the hypothesis (Olshausen-Field 1996) that V1 represents natural images with
few active units; Gabor-like RFs fall out as the optimal sparse basis of natural-image statistics. *DEV-RANDOM:* the
inventory's accepted criterion-2 tag for a one-time genome-style random weight draw (`rng.uniform(seed)`), counted as
self-organized / hardware-portable (the feedback-alignment precedent, `sim/dendritic_neuron.py:25`). *HOST-DESIGNED:* weights
computed by a host formula and injected — NOT hardware-free.

---

## (1) The exact host residual — what, where, how big

**File / function:** `sim/visual_cortex.py`
- `gabor_kernel(sigma_x, sigma_y, theta, freq, phase)` (lines 39-73) — the 2D Gabor formula (Gaussian envelope × cosine
  carrier).
- `build_v1_simple_weights(...)` (lines 76-152) — tiles 8 orientations × 4 spatial frequencies of Gabor kernels across a
  16×16 grid of retinotopic positions, sampling retina pixels within a radius-4 RF; splits the bipolar Gabor into ON/OFF
  retina channels (positive lobe → ON, negative lobe → OFF). Returns `(pre_idx, post_idx, weights)`.
- `apply_v1_gabor_weights(bridge, ...)` (lines 223-310) — translates those to global indices and **overwrites** the
  `retina → cortex_v1_simple` pathway via `bridge.set_pathway_weights("retina_to_v1_simple_gabor", ..., add_missing=True)`.

**Wiring (the deployed nav stack), `research/runners/g11_bg_runner.py`:**
- The `retina → cortex_v1_simple` pathway is declared **`plastic=True`** with `plasticity_gate="visual_cortex_v1"`
  (line 2631-2637). The code comment is explicit: *"Plastic so STDP can refine weights from whatever Gabor init we apply
  post-build (or from random init in v1 minimal mode)."*
- Under `--enable-visual-cortex`, `apply_v1_gabor_weights(...)` is called once after `_initialize_simulation_data()`
  (line 4688-4701), replacing the random-init pathway weights with the host Gabor bank.

**Size (measured, default parameters 8 orient × 4 freq × 16×16 pos, retina 32×32×2, RF radius 4):**

| quantity | value |
|---|---|
| V1 simple cells (post) | **8,192** |
| retina inputs (pre) | 2,048 |
| **host-built Gabor synapse weights injected** | **527,543** |
| distinct V1 cells weighted | 8,192 (all) |
| mean synapses / V1 cell | 64.4 |
| weight magnitude range | [0.0104, 1.0000] |
| **distinct RF templates (the true residual)** | **32** (= 8 orientations × 4 frequencies) |

**The genuine residual is small.** Although 527 k synapse *values* are injected, they are generated from just **32 unique
Gabor kernels** (8 orientations × 4 spatial frequencies), tiled translation-invariantly across 256 positions. So the
host-designed information content is **32 oriented-filter templates** — exactly the orientation/frequency tuning a real V1
develops. Everything downstream (`V1_complex → V2 → IT`) is already learned on-substrate (STDP, `plastic=True`). The residual
is precisely "where do the 32 oriented templates come from — a host formula, or local plasticity + input statistics?"

---

## (2) Biology + existing-sim-precedent review (the standing opening move)

**How biology builds V1 RFs (it does NOT inject Gabors).**
- **Hubel-Wiesel (Kandel 6e Ch 22, p ~595-598; catalog E.08):** simple-cell oriented RFs are built from *aligned*
  LGN center-surround inputs — a linear-filter + threshold approximation that *looks* Gabor, but is *assembled by wiring*,
  not specified by a formula.
- **Spontaneous-activity-driven refinement / retinal waves (Kandel 6e Ch 49, p 1218-1222; catalog L.05):** *before* eyes
  open, the retina generates spontaneous *patterned* waves of correlated activity that drive RF/column refinement via
  NMDAR-dependent (Hebbian) rules. "The wave content matters — random noise wouldn't produce ocular-dominance maps;
  coherent waves do. The brain is *self-organizing* its sensory representations before experience arrives." The catalog
  L.05 sim-status note **anticipates this exact build**: *"generate retinal-wave-like input → train sensory→cortex pathway
  during pretraining gate-open phase → freeze → verify cortex develops coherent receptive fields (analogue of orientation
  columns)."*
- **Critical periods (catalog L.04 / L.19):** the V1 ocular-dominance critical period (~P21-P35 mouse) is exactly a
  developmental window of plasticity that then closes — mirrored by the project's `plasticity_gate` freeze-after-window
  pattern.
- **Sparse coding (Olshausen-Field 1996; the canonical computational account):** Gabor-like RFs are the *optimal sparse
  basis* of natural-image statistics. A local rule that maximizes sparsity + decorrelation on natural-image (or
  retinal-wave) input *recovers* oriented, localized, multi-scale filters — i.e. the host Gabor bank's content emerges from
  the right objective + the right input.
- **BCM rule (Bienenstock-Cooper-Munro 1982; catalog Z.* / E references):** the sliding-threshold Hebbian rule whose
  fixed points are orientation-selective; the classic theoretical route from local plasticity to V1 selectivity. The
  project's NMDAR Ca²⁺-amplitude LTP/LTD switch is explicitly noted (catalog, Kandel) as the "BCM-like substrate."

**Existing-sim precedent (literature — this is a SOLVED, well-replicated result).** Spiking networks with local Hebbian/STDP
+ inhibition develop Gabor-like orientation-selective RFs from natural-image patches:
- **SAILnet (Zylberberg-Murphy-DeWeese 2011):** spiking integrate-and-fire neurons + *local* (Foldiak-style) learning rules
  produce "localized and oriented Gabor-like filters" — sparse coding realized purely in spikes with local rules. The most
  directly transferable precedent (no backprop, no host weights).
- **Mirrored-STDP autoencoder (Burbank 2015, PLOS Comp Biol):** STDP in a spiking net develops "compact receptive fields
  that resemble the Gabor filters found in V1 simple cells" on natural-image patches.
- **Voltage-dependent STDP V1 models** + a recent biologically-grounded E/I spiking V1 (2024-2025, PLOS Comp Biol):
  STDP-trained spiking networks "develop localized oriented Gabor-like receptive fields of varying sizes and spatial
  frequencies that closely resemble physiological studies," with sparse coding + decorrelation emerging.
- The general result (King-Zylberberg-DeWeese; Savin et al.): "STDP-based learning rules in spiking networks naturally
  lead to the emergence of orientation-selective Gabor-like features through unsupervised learning on natural images."

**What the project ALREADY has (reusable machinery, no need to reinvent):**
- The `retina → cortex_v1_simple` pathway **already exists, is `plastic=True`, and is gated** (`visual_cortex_v1`) — the
  freeze-after-critical-period control is already wired.
- **STDP** (`fused_stdp_weight_update`, `sim/kernels.py:365`) + **rate-Hebbian** (`enable_hebbian_learning`,
  `hebbian_learning_rate`, `sim/config.py:268`) + **homeostatic threshold adaptation** (`fused_homeostasis_update`,
  `sim/kernels.py:348`) — the three ingredients of a BCM-like local rule.
- **Structural plasticity / activity-dependent synaptogenesis** (`enable_structural_plasticity=True` by default,
  `struct_plast_activity_bias`, distance kernels; `sim/config.py:531-539`) — could grow the *local* RF support rather than
  fixing radius-4.
- **The image-render + retina-drive front end** (`render_gridworld_to_image`, `image_to_retina_drive`) — the input pipe.
- **Plasticity-gate freeze/thaw** (`set_plasticity_gate`) — the critical-period close.
- **Precedent the project relies on for symmetric correlation learning:** CYCLE-95 already established that for
  *symmetric co-occurrence* the **right rule is rate-Hebbian, not STDP** (STDP measured 656 k events / 0 weight change at
  Δt≈0 because symmetric co-occurrence has no pre→post order — `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`).
  This is directly load-bearing here: natural-image RF learning is also a *correlation* problem, so the de-risk should
  prefer the rate-Hebbian/BCM path (or a sparse-coding objective), not naïve symmetric STDP.

---

## (3) Ranked cheap-first on-substrate mechanisms

Each: the emergence target + reusable machinery + cheap-first de-risk + anti-cheat controls + sim/-edit-or-not.

### Mechanism A (RECOMMENDED) — retinal-wave / natural-image pretraining of `retina→V1_simple` with the existing local rule
The catalog-L.05 build, exactly. Open `visual_cortex_v1`, drive the retina with **patterned** input (retinal-wave-like
correlated blobs, or natural-image patches), let the *existing* on-substrate rule (rate-Hebbian + homeostasis, BCM-like;
or voltage/rate-STDP) + lateral inhibition refine `retina→V1_simple` from random init, then **freeze the gate**
(critical-period close). The 32 oriented templates emerge instead of being injected.
- **Reusable machinery:** the pathway is already `plastic=True` + gated; STDP + Hebbian + homeostasis kernels exist;
  the retina-drive pipe exists; `set_plasticity_gate` does the freeze. **Likely NO `sim/` edit** for a first de-risk — it
  reuses `build_v1_simple_weights` only to *measure* (the host Gabor bank becomes the *reference* the learned RFs are
  scored against, not the deployed weights). Lateral inhibition between V1 cells (the sparse-coding ingredient) may need a
  `V1_simple→V1_simple` inhibitory pathway — declarable in the regions framework (no `sim/` edit), or reuse the existing
  FS-interneuron pattern.
- **Cheap-first de-risk:** numpy / CPU, off the bridge first (fastest): a rate-Hebbian/Oja or a sparse-coding (SAILnet-style
  local) update on `retina→V1` from natural-image patches → measure whether oriented, localized RFs emerge. If GO, lift to
  the real spiking bridge (drive retina, run plastic, read learned weights). Multi-seed.
- **Anti-cheat controls (the Gabor-emergence metric is the crux):**
  1. **Orientation/frequency tuning of the LEARNED RFs** — fit each learned RF to a Gabor; report the orientation HWHH
     (target ~30°, catalog E.08) + spatial-frequency tuning. A learned RF bank should span orientations + scales like the
     host bank.
  2. **RSA / structure-preservation vs the host Gabor bank** — correlate the off-diagonal of the learned-RF response
     similarity matrix with the host-bank response matrix on a held-out image set (the load-bearing downstream metric — see
     §4). High r ⇒ the learned front end carries the same similarity structure.
  3. **No-learning control** — random-init `retina→V1` (gate never opened): RFs stay unoriented (no Gabor tuning, low RSA).
     This is the discriminating baseline (cf. Option B's flat-distinct = 0.000).
  4. **Input-content control (L.05's "wave content matters")** — train on UNSTRUCTURED noise instead of patterned
     waves/natural images: oriented RFs should NOT emerge (proves the structure comes from input statistics + the rule, not
     the substrate alone).
  5. **Downstream-functional check** — re-run the Option-B within>between margin (§4) with the LEARNED front end; it should
     reproduce the +0.78 similarity margin the host bank gives.
- **Why ranked first:** it is the *biologically correct* close (retinal waves + Hebbian refinement = how V1 actually
  develops), it is the catalog's anticipated build, the literature shows it robustly works in spiking nets, and it reuses
  near-everything (probably zero `sim/` edits for the de-risk).

### Mechanism B — accept a DEV-RANDOM-then-refined V1 (genome-coarse, experience-refined)
Initialize `retina→V1` with a **structured DEV-RANDOM** draw (random localized oriented blobs from `rng(seed)` — coarse
orientation bias genome-specified), then let the existing plastic rule refine under visual experience. The criterion-2 tag
becomes DEV-RANDOM (accepted, hardware-portable — the feedback-alignment precedent `dendritic_neuron.py:25`) rather than
HOST-DESIGNED.
- **Reusable machinery:** same pathway + gate; a structured-random init helper (small, additive). The refinement is the
  same local rule as A.
- **De-risk:** confirm a structured-random init + short refinement reaches the same downstream RSA/margin as the host bank.
- **Anti-cheat:** same Gabor-tuning + RSA + no-refinement control.
- **sim/ edit:** none required (the init can be a runner-side draw fed through `set_pathway_weights`).
- **Note:** weaker than A biologically (pure random orientation seeds are not how waves work), but it is the *cheapest*
  criterion-2 close if the downstream pipeline only needs *some* oriented similarity structure (see §4) rather than
  experience-tuned Gabors.

### Mechanism C — full sparse-coding objective realized in spikes (SAILnet on-bridge)
Implement the SAILnet local learning rules (Foldiak: Hebbian feedforward + anti-Hebbian recurrent inhibition + homeostatic
threshold) on the `retina→V1_simple` + a `V1_simple→V1_simple` inhibitory pathway, trained on natural-image patches. This is
the *principled* sparse-coding route — the strongest, most faithful close.
- **Reusable machinery:** Hebbian + homeostasis kernels; the inhibitory-pathway declaration. Likely a small `sim/` edit for
  the anti-Hebbian recurrent term IF the existing inhibitory-plasticity path doesn't cover it.
- **De-risk:** same emergence metrics; additionally measure population sparsity (the SAILnet signature).
- **sim/ edit:** possibly minor (anti-Hebbian inhibitory plasticity) — heavier than A.
- **Note:** ranked third only because it is more machinery than A for the same first-order outcome (oriented RFs); reserve
  it if A's simpler BCM-like rule doesn't reach clean orientation tuning, or if population-sparsity faithfulness is wanted.

---

## (4) Honest framing — genuine residual vs developmentally defensible, + the downstream dependency

**Is B1 a genuine residual to close, or already defensible?** Both, with a clean split:

- **For the FULLY-SPIKING / neuromorphic-hardware-port goal, B1 IS a genuine (if small) residual.** It is HOST-DESIGNED
  (criterion 2), which is *weaker than DEV-RANDOM* — a deterministic host formula injected at init, so a chip would need a
  host to compute the Gabor bank. The inventory correctly flags it. The residual's true size is **32 oriented templates**,
  and the project *already has* the machinery + a `plastic=True` pathway to grow them, so the close is *cheap-ish* and
  biologically grounded (Mechanism A). It is therefore worth closing — not because it is large, but because it is the kind
  of "spiking-at-runtime, host-structured" residual the owner's hardware-port lesson explicitly targets, and the close is
  the *correct* biology (retinal waves + Hebbian refinement).

- **It is ALSO developmentally defensible in the weak sense** that real V1 RFs are genome-coarse + experience-refined, and
  Gabor RFs are biologically standard. So if a full Mechanism-A build hits cost, Mechanism B (DEV-RANDOM-then-refined) is a
  legitimate criterion-2 close — it moves the tag from HOST-DESIGNED to DEV-RANDOM (the accepted self-organized bar), which
  is the minimum bar for the hardware-port goal.

**The downstream-dependency note (decisive for how much is needed).** The 2026-06-16 generalization arc used V1 *only for
visual similarity structure*, and quantified exactly what it needs:
- `2026-06-16-generalization-optionB-visual-similarity.md` (GO, 3 seeds): the Gabor/V1 front end's load-bearing output is a
  **similarity-structured perception code** (within-category cosine 0.86 vs between 0.08, margin **+0.78**), and — the
  decisive anti-cheat — **the code's structure tracks the PIXELS, not exact Gabor identity (RSA pixel-provenance r=0.99)**.
  The discriminating baseline is the flat-distinct code = 0.000 margin.
- **Implication:** the downstream pipeline (cross-modal Hebbian unification → category generalization → who/what + moat)
  needs the front end to produce *oriented, localized, similarity-preserving* responses — it does **NOT** require the exact
  host Gabor coefficients. A **learned or DEV-RANDOM-structured RF bank that preserves the pixel-similarity geometry
  (high RSA r, positive within>between margin) is functionally sufficient.** This means the *cheapest* close that survives
  the §3 anti-cheats (RSA vs the host bank ≈ high; within>between margin reproduced; no-learning control collapses) fully
  discharges B1 for every current downstream use — exact Gabor recovery is a *nice-to-have faithfulness*, not a
  requirement. (Caveat to record, from Option B: the absolute +0.78 margin is inflated by deliberately orientation-separable
  bar stimuli; the load-bearing criterion is *relative* — learned ≫ flat, and learned-RSA ≈ host-RSA.)

- **Scope honesty:** this scoping does not build or de-risk anything; it pins the residual (32 host Gabor templates on a
  `plastic=True` pathway), confirms the mechanism is solved in the literature + anticipated by the catalog, and shows the
  downstream only needs similarity structure. The decisive confirmation is the Mechanism-A de-risk (learned RFs reproduce
  the Option-B margin + survive the no-learning + input-content controls).

---

## Recommended cheap-first de-risk (one line)

**Run Mechanism A:** open `visual_cortex_v1`, drive the retina with patterned input (retinal-wave-like correlated blobs or
natural-image patches), let the existing on-substrate **rate-Hebbian + homeostasis** rule (BCM-like; NOT naïve symmetric
STDP, per CYCLE-95) + lateral inhibition refine `retina→V1_simple` from random init, freeze the gate, then score the
LEARNED RFs against the host Gabor bank with the §3 anti-cheats — **Gabor orientation/frequency tuning + RSA-vs-host-bank +
the Option-B within>between margin**, with a **no-learning control** and an **unstructured-noise input control** (L.05's
"wave content matters") as the discriminating baselines. CPU/numpy off-bridge first (SAILnet-style local update on
natural-image patches), then lift to the real spiking bridge. Likely **no `sim/` edit** for the first de-risk (reuse the
existing plastic pathway + kernels; `build_v1_simple_weights` becomes the scoring reference, not the deployed weights).

---

## References

- Kandel 6e Ch 22 (Hubel-Wiesel V1 simple cells, p ~595-598) + Ch 49 (retinal waves / activity-dependent refinement,
  p 1218-1222). Catalog: E.08 (V1 simple cells), E.09 (complex cells), E.10 (cortical columns / ocular dominance), L.04 +
  L.19 (critical periods), **L.05 (spontaneous-activity-driven refinement — the anticipated build)**.
- Olshausen & Field 1996 (sparse coding → Gabor basis of natural images); Bienenstock-Cooper-Munro 1982 (BCM).
- Zylberberg, Murphy & DeWeese 2011, SAILnet (spiking + local Foldiak rules → Gabor-like RFs):
  [PLOS Comp Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1004566) (mirrored-STDP
  autoencoder, Burbank 2015).
- Recent biologically-grounded E/I spiking V1 (STDP → Gabor RFs + sparse coding + decorrelation):
  [bioRxiv 2024.12.05.627100](https://www.biorxiv.org/content/10.1101/2024.12.05.627100v1.full) /
  [PLOS Comp Biol 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013644).
- Bio-inspired spiking CNN with layer-wise sparse coding + STDP:
  [arXiv:1611.03000](https://arxiv.org/pdf/1611.03000).
- Project: `2026-06-16-generalization-optionB-visual-similarity.md` (downstream needs similarity structure, RSA r=0.99);
  `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (rate-Hebbian, not STDP, for symmetric correlation);
  `2026-06-21-shortcut-inventory-definitive.md` (B1 surfaced); `sim/dendritic_neuron.py:25` (DEV-RANDOM precedent).
