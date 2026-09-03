---
type: design
status: design
date: 2026-09-03
lane: perception (board #135 / #75)
mechanism: vision configural-binding / relative-offset conjunctive units above the frozen S2 bank
runner: research/runners/_vision_lindiscrim_readout_derisk.py   # the mechanism this design EXTENDS (not edited by this doc)
builds_on:
  - research/findings/2026-09-03-vision-satdiv-divisive-norm-readout-BORDERLINE.md
  - research/findings/2026-09-03-satdiv-readout-mostly-in-control-BORDERLINE-refined.md
  - research/findings/2026-09-01-vision-readout-side-exhausted-satdiv-plus-ridge-plateau-points-to-S2-template-learning.md
  - research/findings/2026-09-01-vision-s2-bcm-template-learning-NOGO-collapses-without-competition-underperforms-baseline-with-it.md
  - research/findings/2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md
  - research/findings/2026-08-19-vision-hmax-hierarchy-composed-pooling-solves-position-invariance-learning-not-load-bearing.md
  - research/biology/coincidence-binding.md
---

# DESIGN: configural binding for the vision identity-readout — a relative-offset conjunctive layer that binds WHICH feature is at WHICH slot, above the frozen S2 bank, to lift the wall the readout-side levers (satdiv/z/k-WTA/BCM/ridge/width) all plateaued on

**Status: DESIGN — a research + specification, NOT a build.** No `sim/` edit, no runner edit, no run. This doc
diagnoses precisely why every readout-side lever on the flat max-pooled S2 code plateaus below the strict
`capability_go` bar, specifies a brain-based configural-binding mechanism that changes the REPRESENTATION (not the
readout), names the exact hooks it reuses in `_vision_lindiscrim_readout_derisk.py`, gives a code-level sketch +
flags, and scopes the cheapest de-risk with its GO gate, anti-cheats, and expected failure modes. It builds on the
board's own standing conclusion that the next mechanism is configural binding, "in flight, not another readout
sweep."

## One-line

The C2 stage MAX-pools each S2 template's response over ALL locations to buy position invariance — and that same
operation DISCARDS the location tag, collapsing "orientation A at slot i, orientation B at slot j" into a
location-free bag of feature matches. The task's identity is exactly that discarded binding (every class shares one
orientation histogram; only the arrangement differs), so no rescaling (satdiv/z), sparsifying (k-WTA), re-tuning
(ridge) or unsupervised retuning (BCM) of a code from which the binding is already gone can cross the bar. The fix
is a representational one: insert conjunctive units tuned to feature-PAIRS-at-a-fixed-RELATIVE-offset, then pool
THOSE over absolute position — position-invariant AND arrangement-selective, binding-by-conjunctive-cells, realized
as spiking coincidence detectors already validated on this substrate.

---

## Part 1 — Why the flat readout caps out: the binding diagnosis

### The pipeline, and the exact operation that destroys the signal

The current mechanism (`_vision_lindiscrim_readout_derisk.py`, all levers below share it) is:

`pixels -> Gabor/V1 -> hypercolumn competition + gate -> C1 innate local MAX-pool per orientation
-> S2 (cosine-match each patch against a FROZEN RANDOM n_s2=96 template bank) -> _apply_s2_norm / _kwta_over_templates
-> LIF S2 spikes -> C2 = per-template MAX over ALL locations (drive.max(axis=1)) -> signed linear-discriminant
ridge readout -> spiking class populations -> argmax.`

The task (`_object_classes`, `_render_object`, imported from `_vision_hmax_hierarchy_derisk`): every object is
`n_slots=3` oriented strokes at fixed relative x-offsets (`slot_offset=10px`); each of the `n_classes=4` classes is
a distinct PERMUTATION of the same orientation set across the slots. By construction every class has the IDENTICAL
orientation histogram, so the discriminative variable is purely configural — WHICH orientation sits in WHICH slot —
and a histogram/bag-of-features read is forced to chance (the `H_flat_pool_held` arm confirms this: ~0.267, <!--derived-->
chance=0.25; quoted from the 2026-08-19 rate HMAX finding). <!--derived-->

The load-bearing failure is the C2 op `drive.max(axis=1)`. Taking, per template, the maximum response over ALL
patch-locations is what makes the code translation-invariant (the object can be anywhere). But the maximum over
locations THROWS AWAY where each template matched. After C2 the 96-vector answers "did template i match anywhere in
the image?" — a location-free SET of feature matches. Two classes that are permutations of one orientation set
produce near-identical C2 vectors, because both contain the same local matches, merely at swapped SLOTS — and the
slot identity is exactly the quantity `max(axis=1)` pooled away. This is the classic binding problem / the
superposition catastrophe (von der Malsburg 1981; Treisman & Gelade 1980's illusory conjunctions are its
psychophysical signature): a feature-LIST cannot represent a feature-CONJUNCTION, and pooling for invariance sits on
the invariance horn of the invariance-vs-selectivity trade-off, sacrificing the conjunction.

### Why there is any signal at all — and why it is stuck at ~0.45

The readout is not at chance (`LEARNED_spkwta_held` ~0.44-0.47) because the S2 patch has extent `s2_p=3`: a template
whose 3x3 window straddles a slot boundary is, by accident of the random draw, a weak two-feature conjunction
detector for one specific adjacent pair. A small, DISTRIBUTED population of such accidental straddlers carries the
residual configural signal — measured in the rate spiking finding as an across-template std of ~0.042 riding on a <!--derived-->
~0.80 common mode. A signed linear discriminant extracts it; normalization and sparsification re-weight it; none can
manufacture the binding the pool already discarded. Three independent results converge on this being a
REPRESENTATIONAL ceiling, not a readout one:

- **Readout-side exhaustion** (2026-09-01, 2026-09-03): satdiv (Carandini-Heeger divisive normalization), z/submean
  affine norm, k-WTA sparsification, and a ridge re-tune all IMPROVE the margin and all PLATEAU; `capability_go` is
  0/6 across ~30 satdiv cells (54/54 seeds on the first 9-cell sweep), best cell just 1/6 on a single seed. The
  misses are narrow but structural.
- **Unsupervised template learning underperforms doing nothing** (2026-09-01 BCM NO-GO): BCM sliding-threshold
  learning of the S2 bank, even with a competitive gate that fixes its collapse, sits at or below the frozen-random
  baseline on every explore seed. Its own diagnosis: "unsupervised, local, activity-dependent tuning has no way to
  discover a discriminative axis that is defined by cross-location ARRANGEMENT rather than local feature content."
  It named the exit as a relational/conjunctive layer that "conjoins S2 responses from DIFFERENT retinotopic
  locations."
- **The distributed configural code is spike-quantization-fragile** (2026-08-19 spiking finding): fully spike-coding
  the S2->C2 configural readout is a 6-seed NO-GO (held ~0.34, position LEAKS at decode 0.97), because the ~5% cosine
  modulation falls below the per-unit spike floor; random==learned (Johnson-Lindenstrauss). Its named next mechanism:
  a DISCRIMINATIVE SPARSE SELECTIVE S2 code, predicted to survive spike coding where the distributed random code does
  not.

### What the flat pool replaced with a constant (the wall-reframe)

Per the project's first-question-at-a-wall ("what does the real system run alongside this that we replaced with a
constant?"): the missing companion process is a stage that BINDS features across locations before pooling. Biology
does not read object identity off a global histogram of V1 outputs; it interposes intermediate stages (V4, posterior
IT) of units tuned to feature COMBINATIONS in relative arrangement, and only pools for position AFTER the
conjunction is formed. The current pipeline pools first and never forms the conjunction. The constant we substituted
is "the conjunction layer is absent; the readout must recover binding from a bag" — which, as three findings show,
it provably cannot.

### The grounding in vision neuroscience

- **HMAX C2 = conjunctions of C1 afferents** (Riesenhuber & Poggio 1999, *Nat Neurosci* 2:1019-1025). The canonical
  model interposes S2 units tuned to COMBINATIONS of C1 features between simple cells and the global pool; the MAX
  op gives shift-tolerance while the conjunction preserves selectivity. Our current runner has the MAX and the
  bank, but its templates are single-location (per-patch) matches pooled independently — it lacks the CROSS-location
  conjunction that makes the arrangement survive the pool. Our own rate HMAX finding (2026-08-19) is the direct
  evidence this closes the gap: with a composed conjunction+pool architecture on the SAME task, held-position decode
  reached 0.5972 vs V1-direct 0.3698 (5/6) — the configural wall fell at rate. <!--derived--> The lindiscrim arc is that same
  capability in the spike-readable regime, still missing the conjunction that the rate version's straddling extent
  supplied.
- **IT columnar feature-combinations** (Tanaka 1996, *Annu Rev Neurosci* 19:109-139; Tsunoda, Yamane, Nishizaki &
  Tanaka 2001, *Nat Neurosci* 4:832-838). Inferotemporal cortex is organized into columns each responding to a
  moderately complex combination of simpler features — the physiological instantiation of conjunctive coding, and
  the target representation for our S2.5 layer.
- **Features of intermediate complexity** (Ullman, Vidal-Naquet & Sali 2002, *Nat Neurosci* 5:682-687). The most
  informative units for classification are neither local edges nor whole objects but intermediate fragments that
  bind a few parts in relative position — exactly a (feature-a, feature-b, offset) conjunction.
- **The binding debate: conjunctive cells vs synchrony** (von der Malsburg 1981/1999 correlation theory; Singer &
  Gray 1995, *Annu Rev Neurosci* 18:555-586 binding-by-synchrony; Ghose & Maunsell 1999, *Neuron* 24:79-85 the
  combination-coding alternative). Two candidate solutions to the superposition catastrophe: (a) dedicated
  conjunctive cells (combination coding), (b) temporal synchrony tagging co-object features. This design takes
  route (a) as primary because it is directly realizable and measurable on this substrate (coincidence detectors,
  below) and because this arc's prior spike-timing/latency codes UNDERperformed here (2026-08-19: latency < count);
  route (b) is named as the alternative with its trade-off in Part 2.

---

## Part 2 — The mechanism: relative-offset conjunctive units (S2.5), spiking, above the frozen bank

### (a) The representational change

Insert a new stage BETWEEN the per-location S2 template drive and the C2 max-over-locations pool. Instead of pooling
each template independently (a bag), define a bank of `n_conj` conjunctive units, each a triple
`c = (template_a, template_b, Δ)`:

```
conj_response_c(image) = MAX over absolute location p of  AND( drive[p, a] , drive[p+Δ, b] )
```

- The `AND` (a coincidence gate) makes the unit fire only when template a matches at some location AND template b
  matches at the location Δ away — a BOUND feature-pair-in-relative-arrangement.
- The `MAX over absolute p` keeps translation invariance (the whole pair can sit anywhere) while PRESERVING the
  relative arrangement, because Δ is internal to the unit, not pooled away.

For a 3-slot object at fixed relative x-offsets, the informative Δ values map to one-slot and two-slot
displacements; a bank of conjunctions spanning those offsets reconstructs the pairwise arrangement code the flat
pool destroyed. This is HMAX's C2-over-C1-conjunctions done RELATIONALLY (Δ-parameterized), an IT-column /
intermediate-complexity unit, and it resolves the binding problem by binding-by-conjunctive-cells.

### (b) How it stays brain-based

- **The AND is coincidence detection**, the project's established binding primitive
  (`research/biology/coincidence-binding.md`, status: established; Kandel: a synapse is "a biochemical detector of
  the near simultaneity" of two converging signals, whose conjunction is SUPRALINEAR — "signal is greater than the
  linear sum" — via NMDA Mg²⁺ expulsion). The spiking realization is a population of coincidence-detector somata,
  each receiving two afferents (template a at p, template b at p+Δ), that cross threshold only on near-simultaneous
  co-drive — the SAME `coincidence_weighted_drive` / `coincidence_k_threshold` mechanism already validated on the
  bridge and deployed in the production vision-identity organ (`vision_identity_production_organ.py`,
  `_emerge36_spiking_perception_pipeline_derisk.py`). In THIS runner (which uses the standalone `lif_spike_read`, not
  the CoreSimConfig bridge) the faithful analog is a `lif_spike_read` soma fed `min(a, b)` with a raised threshold,
  so it spikes only when both afferents are strong — the standalone port of coincidence detection, exactly as this
  runner already ports LIF and the S1->C1 front end.
- **Fixed-conjunctive FIRST, learned SECOND.** Start with a FROZEN RANDOM conjunction bank (sample the `(a,b,Δ)`
  triples once per seed), mirroring the frozen-random-S2-bank precedent. This isolates the REPRESENTATIONAL change
  (conjunctions vs feature-list) from any learning question: if fixed random conjunctions clear the bar, the
  diagnosis (binding was the missing representation) is proven with the cheapest possible mechanism. The
  discriminative/supervised conjunction-SELECTION (below) is the named next rung, and it is precisely what BOTH prior
  findings pointed to — the BCM NO-GO's exit #2 (a relational conjunctive layer) and the spiking finding's predicted
  "discriminative sparse selective S2 becomes load-bearing on spikes."
- **The Δ-weight-sharing** (one conjunction applied at every absolute p with a fixed internal Δ) is the SAME innate
  retinotopic-convolution scaffold already FLAGGED and defended in the HMAX findings (complex-cell RFs are
  developmentally pre-structured; Hubel-Wiesel) — no new scaffold class is introduced. The coincidence gate is a
  neuron/synapse operation, not a host argmax.

### (c) Composition with the existing pipeline — the hooks reused

The binding stage slots in with the readout, dataset, front end, GO gate and every anti-cheat UNCHANGED:

| Reused hook | Source | Role, unchanged |
|---|---|---|
| `_c1_spiking`, `lif_spike_read`, `spike_code`, `_flat` | `_vision_hmax_spiking_derisk.py` | spiking S1->C1 front end + LIF core (config B) |
| `build_gabor_response_matrix`, `encode_v1`, `pool_v1_to_complex` | `_genfrontier_optionB_visual_similarity_derisk.py` | V1 rate front end + rendering (environment) |
| `_object_classes`, `_build_objects`, `_render_object`, `_positions`, `_extract_patches`, `_scramble_images` | `_vision_hmax_hierarchy_derisk.py` | the histogram-matched configural stimulus + held-out positions |
| `_init_templates` | `_vision_rstdp_readout_derisk.py` | the frozen random base S2 bank (the conjunctions are built OVER its per-location drive) |
| `_apply_s2_norm`, `_kwta_over_templates` | this runner | S2 normalization / competition, applied to per-location drive BEFORE the conjunction |
| `_standardise`, `_train_linreadout`, `_spiking_class_read`, `_lin_score_pred` | this runner | the signed-linear-discriminant ridge readout — reads `r (N, D)` with D from the array shape, so it consumes `r (N, n_conj)` with ZERO change |
| `_centroid_decode`, `_within_split_decode` | `_vision_hmax_hierarchy_derisk.py` | position-pooled-out + anti-cheat decodes |
| the whole `capability_go` gate (`run_seed` L555-569) | this runner | the GO bar, unchanged (Part 3) |

The natural branch point is INSIDE `_c2_spike_code` (L354-375) and `_c2_rate_code` (L378-387), after
`_apply_s2_norm`/`_kwta_over_templates` and BEFORE the `.max(axis=1)` pool. Both functions share the identical
pre-pool structure, so the new op is factored out (like `_apply_s2_norm`) and called from both to keep the spike and
rate arms aligned. Because the readout reads D from the feature array's shape, changing the feature count
`n_S2 -> n_conj` flows through untouched.

### (d) Code-level sketch + flags

Flag cluster (copy the additive-default-off `--s2-kwta-frac` / `--s2-learn none|bcm` pattern; `none` = byte-identical
to today):

```
--conj-bind {none,fixed,learned}   dest conj_bind      default none   # none = current max-pool path, byte-identical
--conj-n INT                       dest conj_n         default 192    # number of conjunctive units (see width control)
--conj-offset-max INT              dest conj_offset_max default 4      # sample Δ in ±1..±offset_max location units
--conj-mode {min,prod,coincidence} dest conj_mode      default min     # the AND: min/product (rate) or LIF coincidence
--conj-delta0-only                 dest conj_delta0    action store_true  # DEGENERATE control (same-location AND only)
--conj-shuffle-offsets             dest conj_shuffle   action store_true  # LESION: break the relative-offset binding
```

The new op (inserted into the runner — shown here, not applied by this doc):

```python
def _bind_conjunctions(drive, pairs, offsets, mode, shuffle_rng=None):
    """Configural binding. drive (N, n_loc, n_S2) post-norm/post-kWTA S2 template match per location ->
    (N, n_loc, n_conj) relative-offset conjunction drive; the caller then MAX-pools over n_loc.
    pairs   (n_conj, 2) int : template indices (a, b) into the n_S2 bank
    offsets (n_conj,)   int : relative x-displacement Δ for afferent b (0 excluded unless delta0-only)
    mode    'min' | 'prod'  : the coincidence AND (rate proxy); 'coincidence' handled by the caller's LIF soma
    shuffle_rng             : if set, afferent b is read at an INDEPENDENT random location (binding LESION)."""
    N, n_loc, n_S2 = drive.shape
    g = int(round(n_loc ** 0.5))                      # locations on a g x g grid; Δ is an x-shift
    da = drive.reshape(N, g, g, n_S2)
    out = np.zeros((N, g, g, len(pairs)), np.float32)
    for c, ((a, b), d) in enumerate(zip(pairs, offsets)):
        aresp = da[:, :, :, a]                         # template a at location p
        if shuffle_rng is not None:                    # LESION: b at an unrelated absolute location
            bresp = da[:, :, :, b][:, :, shuffle_rng.permutation(g)]
        else:
            bresp = np.roll(da[:, :, :, b], -d, axis=2) # template b at p + Δ (x)
            if d != 0:                                  # drop the wrap-around edge, keep binding local
                (aresp := aresp.copy())[:, :, g-abs(d):] = 0
                bresp = bresp.copy(); bresp[:, :, g-abs(d):] = 0
        out[:, :, :, c] = np.minimum(aresp, bresp) if mode == 'min' else aresp * bresp
    return out.reshape(N, g * g, len(pairs))
```

Wire-in inside `_c2_spike_code` (analogously `_c2_rate_code`), right after `_kwta_over_templates`:

```python
    if getattr(a, "conj_bind", "none") != "none":
        rng = np.random.default_rng(base_seed * 71 + 5)
        off_set = ([0] if getattr(a, "conj_delta0", False)
                   else [d for d in range(-a.conj_offset_max, a.conj_offset_max + 1) if d != 0])
        pairs   = rng.integers(0, W0.shape[0], size=(a.conj_n, 2))
        offsets = rng.choice(off_set, size=a.conj_n)
        shuf    = np.random.default_rng(base_seed * 91 + 7) if getattr(a, "conj_shuffle", False) else None
        drive   = _bind_conjunctions(drive, pairs, offsets, a.conj_mode, shuffle_rng=shuf)
        # 'coincidence' mode: raise the LIF threshold so the soma fires only on joint drive
        gain = a.s2_gain * (1.6 if a.conj_mode == "coincidence" else 1.0)
    # ... existing lif_spike_read over `drive` -> spike_code -> .max(axis=1) -> r (N, n_conj)
```

Everything downstream (`_standardise -> _train_linreadout -> _spiking_class_read`, the anti-cheats, the GO gate) is
unchanged; `r` is now `(N, n_conj)` instead of `(N, n_S2)`.

The `learned` variant (the NEXT rung, sketched only): score the fixed random conjunction bank against the train
labels (a Fisher/mutual-information score per unit, or extend `_train_linreadout`'s three-factor delta back into a
top-k SELECTION over `(a,b,Δ)`), keep the top-k most discriminative conjunctions — a supervised, sparse, selective
conjunction code, the exact "discriminative sparse S2" the spiking finding predicted becomes load-bearing. Deferred
behind proving the fixed representation first.

---

## Part 3 — Cheapest de-risk, GO gate, anti-cheats, failure modes

### Cheapest de-risk

`--conj-bind fixed --conj-n 192 --conj-offset-max 4 --conj-mode min`, at RATE first (`_c2_rate_code`, the generous
reference — if rate cannot, spikes will not), then the spiking `count` code. 6 seeds (42/43/44/100/101/102),
CPU-only, dispatched to the mini-PC pool exactly like the satdiv sweep (~20-22s per 6-seed run observed in this
lane) — ZERO GPU, ZERO `sim/` edit, no new learning. This is a pure representational A/B against the byte-identical
`--conj-bind none` baseline on the SAME GO gate the whole readout arc used.

### GO gate — the runner's existing `capability_go`, UNCHANGED (≥5/6)

Apples-to-apples with the readout arc. Per seed, the AND of all eight (`run_seed` L559-568): (1) held object decode
≥ chance+0.15; (2) beats the config-C NO-GO floor by ≥0.10 (≥0.44); (3) **beats V1-direct-held by ≥+0.10** (the
margin satdiv made on only 2/6 seeds); (4) beats the flat pool by ≥0.10; (5) `learning_load_bearing` (LEARNED −
RANDOM ≥0.10 — here: the CONJUNCTION bank vs a random-readout control); (6) `position_pooled_out` (held-out-position
object decode ≥ chance+0.15 AND position ≤ chance+0.15 off the class-population spike code — the held-out position
generalization test); (7) pixel-scramble null ≤ chance+0.15; (8) label-shuffle null ≤ chance+0.15. GO = ≥5/6 seeds
pass `capability_go`. Held-out positions are structural: train {0,2,4,6}, held {1,3,5,7}, never seen.

### Anti-cheats — three NEW ones specific to configural binding (do not fool ourselves)

The single biggest risk is that adding conjunctions merely WIDENS a fixed-random hidden layer, and per Huang, Zhu &
Siew (2006, *Neurocomputing* 70:489-501) a wider random hidden layer lifts a linear readout for free — a capacity
win masquerading as a binding win. Three controls isolate the binding:

1. **Width-matched flat control (THE ELM control).** Run the byte-identical baseline at `--n-s2 192` (= `conj_n`),
   `--conj-bind none` — same feature count, NO binding. It must NOT clear the bar. If it does, the win was capacity,
   not binding -> NO-GO on this mechanism, pivot to the already-named bank-WIDTH lever.
2. **Offset-shuffle lesion.** `--conj-shuffle-offsets`: read afferent b at an INDEPENDENT random location
   (`min` of two BAGS, no relative binding). The capability must VANISH — this proves it is the RELATIVE arrangement,
   not merely "AND of two features anywhere." (This lesion is a build-time reconfiguration of the conjunction wiring;
   it holds by construction at measurement, satisfying the `lesion` term condition.)
3. **Δ=0 degenerate control.** `--conj-delta0-only` (same-location AND only) must NOT clear the bar — a same-location
   conjunction adds no cross-slot binding; if it passes, the win is local feature interaction, not configural.

Plus the runner's existing scramble-null, label-shuffle-null, and determinism byte-compare, all unchanged. And the
existing `position_pooled_out` gate is itself the guard against the config-C failure mode (a conjunction+pool that
smuggles absolute position back into the code — 2026-08-19 saw exactly this leak at decode 0.97).

### Expected failure modes

1. **Width confound** (mitigated by anti-cheat 1). Most likely single confound; the width-matched control is the
   decisive discriminator.
2. **Random conjunctions may be as inert as random templates** (Johnson-Lindenstrauss: 4 well-separated classes can
   survive a random projection, so if the only problem were spike quantization rather than a missing representation,
   fixed random conjunctions might not help). Fixed-first cleanly separates "representation missing" (fixed
   conjunctions clear it) from "discriminative learning missing" (they don't; the learned/selective rung is then
   required — the predicted path). Either outcome is informative and no-defer.
3. **Position leak** — a conjunction+pool reintroducing absolute position (config-C's failure). Caught by
   `position_pooled_out`; the design encodes relative Δ (not absolute p) precisely to avoid it, and the edge-drop in
   `_bind_conjunctions` keeps the binding local.
4. **Pairs insufficient for a 3-slot object.** Second-order conjunctions may only partially lift a 3-slot
   arrangement; if pairs lift-but-plateau, third-order (triple) conjunctions are the follow-on, at a bounded
   combinatorial cost (random sampling keeps `n_conj` fixed).
5. **Spike quantization of the coincidence AND — but this is a PREDICTED POSITIVE.** The AND is a sharp nonlinearity
   producing a SPARSE SELECTIVE code (few strongly-firing conjunction units), which the 2026-08-19 finding explicitly
   predicts survives spike coding where the distributed cosine code does not. So the spiking arm may match or exceed
   rate here — a directional prediction to verify, not just a risk. De-risk at rate first regardless.

### What a GO here would and would not establish

A GO would establish the CAPABILITY (a spike-readable, position-invariant, arrangement-selective configural code) at
the runner level with the whole anti-cheat suite — it would NOT by itself be `wired`/`on-by-default`/`integrated`
(no live conversational vision consumer ingests a percept beyond the default-OFF `BRAIN_VISION_IDENTITY` organ; see
2026-08-19's production-wiring block). The honest scope of the de-risk is the spiking capability; production
wire-in is a separate, later rung.

## Sources

- Riesenhuber, M. & Poggio, T. (1999). Hierarchical models of object recognition in cortex. *Nature Neuroscience*
  2:1019-1025. (HMAX; S2 units tune to C1 feature CONJUNCTIONS; C MAX gives shift-tolerance.)
- Tanaka, K. (1996). Inferotemporal cortex and object vision. *Annual Review of Neuroscience* 19:109-139.
  (Columnar organization for moderately complex feature combinations.)
- Tsunoda, K., Yamane, Y., Nishizaki, M. & Tanaka, K. (2001). Complex objects are represented in macaque
  inferotemporal cortex by the combination of feature columns. *Nature Neuroscience* 4:832-838.
- Ullman, S., Vidal-Naquet, M. & Sali, E. (2002). Visual features of intermediate complexity and their use in
  classification. *Nature Neuroscience* 5:682-687. (Intermediate-complexity fragments bind a few parts in relative
  position; the most informative units for classification.)
- von der Malsburg, C. (1981/1999). The correlation theory of brain function; and Binding in models of perception
  and brain function. *Neuron* 24:95-104. (The binding problem / superposition catastrophe.)
- Treisman, A. & Gelade, G. (1980). A feature-integration theory of attention. *Cognitive Psychology* 12:97-136.
  (Illusory conjunctions: binding is a real computational problem.)
- Singer, W. & Gray, C. M. (1995). Visual feature integration and the temporal correlation hypothesis. *Annual
  Review of Neuroscience* 18:555-586. (Binding-by-synchrony — the named alternative route.)
- Ghose, G. M. & Maunsell, J. (1999). Specialized representations in visual cortex: a role for binding? *Neuron*
  24:79-85. (Combination coding vs synchrony — the conjunctive-cell horn this design takes.)
- Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). Extreme learning machine: theory and applications. *Neurocomputing*
  70(1-3):489-501. (Fixed-random-hidden-layer capacity scales with WIDTH — the confound anti-cheat 1 controls for.)
- On this substrate: the six `builds_on` findings (satdiv BORDERLINE + refined, readout-exhausted, BCM NO-GO,
  spiking-hierarchy quantization-limited, rate HMAX composed-pooling GO) and `research/biology/coincidence-binding.md`
  (the coincidence-AND primitive, status: established; Kandel PNS 6e anchors).

## Reproduce (the de-risk this design specifies — NOT run here; no artifacts exist yet)

All `--out` files below are written into `research/findings/raw/lanes/perception/` (bare filenames shown so this
design doc, which cites no artifacts, is not mis-read as claiming these future outputs already exist).

```bash
# 1. Fixed configural binding, RATE reference (the generous ceiling), 6 seeds, CPU/pool:
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
    research.runners._vision_lindiscrim_readout_derisk \
    --code count --conj-bind fixed --conj-n 192 --conj-offset-max 4 --conj-mode min \
    --seeds 42 43 44 100 101 102 \
    --out conjbind_fixed_min_n192_6seed.json

# 2. Width-matched flat ELM control (same feature count, NO binding) -- must NOT clear the bar:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --n-s2 192 --seeds 42 43 44 100 101 102 \
    --out conjbind_widthctrl_n192_6seed.json

# 3. Offset-shuffle lesion (break relative binding) -- capability must VANISH:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --conj-bind fixed --conj-n 192 --conj-offset-max 4 --conj-mode min --conj-shuffle-offsets \
    --seeds 42 43 44 100 101 102 \
    --out conjbind_shuffle_n192_6seed.json

# 4. Δ=0 degenerate control (same-location AND) -- must NOT clear the bar:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --conj-bind fixed --conj-n 192 --conj-mode min --conj-delta0-only \
    --seeds 42 43 44 100 101 102 \
    --out conjbind_delta0_n192_6seed.json

# 5. Spiking coincidence code (the predicted-positive spike arm), after rate clears:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --code count --conj-bind fixed --conj-n 192 --conj-offset-max 4 --conj-mode coincidence \
    --seeds 42 43 44 100 101 102 \
    --out conjbind_coincidence_n192_6seed.json
```
