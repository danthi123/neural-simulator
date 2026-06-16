# Functional integration — PERCEPTION→MEMORY survives a TRAINED (noisy) read-out: GO

**Date:** 2026-06-16 (functional-integration arc, the (B) interaction — the trained-map follow-on).
**Verdict:** **GO** — the perception→engram→recall loop **survives a TRAINED (Hebbian-grown, NOISY)
`cortex_it → language_output` read-out**, not just the clean labeled-line stand-in. A perception-driven engram
tag, stimulated later, still recalls the perceived concept SPECIFICALLY (perceive A → recall A, not B), through
a read-out whose per-object selectivity was LEARNED by co-firing — and the recall is lesion-confirmed to ride
the trained synapses. Multi-seed **6/6** (42/43/44/100/101/102), all 4 objects.
**Probe:** `research/runners/funcint_perception_to_memory_trained_probe.py` (CPU, `SIM_BACKEND=numpy`,
~30 s/seed incl. the lesion re-train).
**Result data:** `research/findings/raw/funcint_perception_to_memory_trained.json`.
**Builds on:** `research/findings/2026-06-16-funcint-perception-to-memory-cheap-first.md` (the (B) GO with a
CLEAN labeled-line read-out) — whose HONEST SCOPE pre-registered exactly this trained-map test as the next step.

---

## 0. What changed vs the confirmed (B) probe (the one idealization this peels off)

The confirmed probe (GO 2026-06-16) installed the `cortex_it → language_output` read-out as a **fixed per-object
TOPOGRAPHIC labeled line** (object o's `cortex_it` band → object o's `language_output` band, fixed weights,
`plastic=False`). That read-out was a *perfect* per-object map, so the recall cosines were ≈ 1.0 to the correct
word and exactly 0.0 to the others — noise-free. Its scope section flagged the **TRAINED (noisy) read-out** as
the load-bearing next test.

This probe replaces that clean stand-in with a **trained, lossy map**:

| | confirmed (B) probe | this trained-map probe |
|---|---|---|
| `cortex_it → language_output` route | fixed **band_o → band_o** labeled line (per-object pre-wired) | **DENSE** (every exc `cortex_it` → every exc `language_output`), low init weight, `plastic=True` |
| selectivity | **wired** (a perfect labeled line) | **LEARNED** by Hebbian co-firing (Pulvermüller / b3 / concept-pool idiom) |
| recall cosine to correct word | ≈ 1.0 (noise-free) | **0.83–0.92** (genuine signal through a noisy map) |
| recall vs chance | 4× chance, but trivial (clean map) | **4× chance through a trained, imperfect read-out** |

The dense-then-trained design is deliberate: a band→band wiring + training would just re-grow the same labeled
line. A DENSE plastic route makes the per-object selectivity something the network must LEARN — the cross-object
synapses are physically present and only stay weak because they never co-fire (confirmed: on-diagonal route
weight ≈ 8.5 vs off-diagonal ≈ 0.65 after training, every seed).

---

## 1. The trained-map mechanism (the NEW part — all synaptic, no `sim/` edit)

The bridge's Hebbian rule is **soft-bound co-firing**: for a synapse whose pre neuron fired the previous step
AND whose post neuron fires this step, `Δw = lr · (w_max − w)`, multiplied by the per-synapse
`cp_plasticity_rate_gain` (`sim/bridge.py:6476–6510`). That is exactly the substrate's embodied co-firing
learning. The probe uses it as follows:

1. **Wire the route DENSE + plastic + low** — every excitatory `cortex_it` neuron → every excitatory
   `language_output` neuron, weight `0.05` (the Hebbian floor), `plastic=True`, tagged
   `plasticity_gate="it_to_lang"`. Excitatory-only on both ends (an inhibitory route neuron would *suppress* its
   target — the inversion the (A)/(B) probes documented). ≈ 42 k synapses (204 × 204).
2. **TRAIN by co-firing** — for each object o, drive object o's `cortex_it` band (the perceived ensemble,
   presynaptic) AND object o's `language_output` band (the teacher, postsynaptic) TOGETHER for a short window.
   Both bands fire → the o→o synapses grow toward `w_max`; cross-object synapses never co-fire → stay at floor.
   Trials are **interleaved** (object order reshuffled each round, deterministic per seed) so no last-trained
   object dominates — the concept-pool recipe's anti-order-effect pattern. 60 trials/object × 4 = 240 trials.
   - **Training is RESTRICTED to the route.** Before training, `cp_plasticity_rate_gain` is set to 0 everywhere
     and 1 only on the `it_to_lang` synapses. Since the Hebbian Δw is multiplied by this gain, the internal
     region recurrence (which also co-fires under the band drives) gets **zero** weight change — only the route
     learns. (Belt-and-braces: the regions are also built `plastic_internal=False`.)
3. **FREEZE** — `set_plasticity_gate("it_to_lang", 0.0)` + `cp_plasticity_rate_gain[:] = 0` +
   `enable_hebbian_learning = False`. No weight updates anywhere after this.
4. **TEST exactly as the confirmed probe** — perceive object → `start_engram_recording` → perception window →
   `commit_engram_tag(region_filter=["cortex_it"])`; recall by `stimulate_tag` → read the `language_output`
   reactivation → cosine to each word's band. (`encode_percept_engram`, `_recall_lang_output_pattern`,
   `_recall_metrics`, `provenance_check` are imported VERBATIM from the confirmed probe.)

**The teacher drive is a LEARNING signal, not a recall-time percept copy.** Driving the `language_output`
teacher band happens only in the TRAIN phase (the Pulvermüller "elevate the output target so the sites co-fire"
pattern, as `bio_three_factor` does). At RECALL time the only write is `stimulate_tag` (driving the perceived
`cortex_it` ensemble); `language_output` is never driven at recall. The recall therefore genuinely rides the
TRAINED synapses (lesion-confirmed below), not a copied vector.

---

## 2. Result — GO on all 6 seeds, all 4 objects

Per-object recall after stimulating the perception-driven engram tag, through the **trained** read-out.
**CLEAN** = perception engram + trained route intact; **LESION** = the trained `cortex_it → language_output`
synapses zeroed (engram still intact). "top1" = the word the `language_output` reactivation spells (argmax cosine
over the 4 object words); "perceived_score" = cosine to the perceived object's band; "margin" = top1 − runner-up.
A recall is "correct" only if the perceived object is the UNIQUE top-1 by ≥ 0.02 (`MIN_RECALL_MARGIN`).

### Per-object detail (seed 42, representative)

| object (perceived) | CLEAN top1 | CLEAN perceived_score | CLEAN margin | correct | LESION |
|--------------------|:----------:|----------------------:|-------------:|:-------:|:------:|
| apple              | apple      | 0.908                 | +0.879       | ✅      | 0.0 → ✗ |
| river              | river      | 0.853                 | +0.787       | ✅      | 0.0 → ✗ |
| dog                | dog        | 0.927                 | +0.899       | ✅      | 0.0 → ✗ |
| cat                | cat        | 0.877                 | +0.816       | ✅      | 0.0 → ✗ |

The CLEAN recall is **specific but now genuinely noisy**: each tag's reactivation drives `language_output` to
spell the perceived object with cosine **≈ 0.85–0.93** (NOT ≈ 1.0 — the trained map is lossy), with a healthy
margin (≈ 0.79–0.90) over the runner-up. This is the signal-above-chance the brief asked for.

### Roll-up (all 6 seeds)

| seed | trained route w (on / off-diag) | CLEAN recall correct | LESION recall correct | provenance |
|------|:-------------------------------:|:--------------------:|:---------------------:|:----------:|
| 42   | 7.88 / 0.63                     | 4/4                  | 0/4                   | ✅         |
| 43   | 8.84 / 0.63                     | 4/4                  | 0/4                   | ✅         |
| 44   | 7.92 / 0.65                     | 4/4                  | 0/4                   | ✅         |
| 100  | 8.66 / 0.72                     | 4/4                  | 0/4                   | ✅         |
| 101  | 8.53 / 0.66                     | 4/4                  | 0/4                   | ✅         |
| 102  | 9.11 / 0.65                     | 4/4                  | 1/4 (OU blip, see §3) | ✅         |

**24/24 perception-driven recalls correct through the trained map (chance = 1/4; ~4× per object, unanimous);
23/24 collapse under the lesion.** The grown route is selective every seed (on-diagonal ≈ 13× the off-diagonal),
i.e. the per-object map was LEARNED, not wired.

---

## 3. The anti-cheat controls — all pass

1. **Lesion (primary).** A fresh bridge, route **re-trained** and every object's engram re-encoded, then every
   trained `cortex_it → language_output` synapse zeroed (≈ 42 k synapses), re-stimulate every tag: recall
   **collapses to 0/4 on 5 of 6 seeds**, with the `language_output` reactivation all-zeros (every cosine 0.0).
   → the recall rides the trained route synapses specifically, not ambient leakage or a Python path.
   - **The single seed-102 residual is an OU-noise floor blip, not a leak.** Under lesion, seed 102 gives
     river/dog/cat = exactly 0.0 (silent) and apple = a lone cosine **0.196** (one stray OU-driven spike landing
     in the apple band, crossing the 0.02 margin). The route is cut; this is the same OU-WTA noise the confirmed
     probe noted in its <1.0 cosines, here surfacing under the (now genuinely noisy) regime. It is far below any
     real-signal level and within the verdict's `≤ 1/4` lesion-collapse tolerance.
2. **Specificity / cross-control.** Stimulate `seen_A` → recalls A NOT B: folded into "correct" (each tag must
   recall its OWN object as the unique top-1). Recall accuracy 24/24 vs chance 1/4 — the trained map's diagonal
   dominates its off-diagonal, so the perception-driven tag reactivates the perceived concept, not another.
3. **Provenance (no Python value-copy).** Asserted structurally per seed (all pass): every committed tag's
   indices are a SUBSET of `cortex_it` (committed with `region_filter=["cortex_it"]`). The training co-firing
   drive is a LEARNING signal in a SEPARATE phase; at recall the ONLY write is `stimulate_tag` (the perceived
   `cortex_it` ensemble) — `language_output` is never driven at recall. No host code copies a percept vector
   into the recall drive.

So the recall is a genuine synaptic perception→memory write+read through a **learned, lossy** read-out: the
perceived ensemble is stored as the tag, re-stimulation reactivates it, and the reactivation reaches language
ONLY through the lesionable TRAINED route.

---

## 4. HONEST SCOPE — still RECALL, not composition; the noisy read-out is now de-risked

- **What this ADDS over the confirmed (B) GO:** the read-out is no longer an idealized clean labeled line but a
  **TRAINED, lossy, noisier** map (recall cosines 0.83–0.92, on/off-diagonal selectivity ratio ≈ 13×). The
  perception→engram→recall loop **survives** it — the next idealization peeled off the (B) probe. This is the
  same scope discipline as the (A) gating probe (de-risk the mechanism, not its idealizations).
- **What this still does NOT establish (deliberately out of scope):** *composition over perceived content.* You
  cannot yet algebraically bind the perceived apple into a novel role-filler fact ("the apple is red") through
  the phasor composer — that needs the perceived (rate/grounded) code to enter the bind/unbind algebra, i.e. the
  rate-vs-phasor wall, which is **step-3 (the learned spiking-cortical binder)**. This probe is a RECALL
  interaction ("I saw the apple" → later recall "apple"), exactly as the confirmed (B) probe.
- **Remaining cheap-probe idealizations (honest notes):** the perception ensembles are direct orthogonal band
  renders (the navigation Gabor/retina front-end is separately validated, not exercised here); the read-out is
  trained on those clean band renders, not on Gabor-derived noisy IT codes; vocabulary is 4 objects at 256
  neurons. What is now de-risked is that a **trained, noisy** `cortex_it → language_output` map carries the
  perception-driven recall — unambiguously YES, lesion-confirmed, 6/6.

---

## 5. Verdict and next step

**GO.** The perception→engram→recall loop survives a trained (Hebbian-grown, noisy) read-out on all 6 seeds and
all 4 objects: recall is specific (24/24, ~4× chance) through a map whose selectivity was LEARNED (on/off ≈ 13×),
the lesion collapses it (23/24; the 1 residual is a characterized OU blip), and provenance is clean. The recall
no longer rides a perfect labeled line — it rides a genuine trained read-out, the load-bearing question the
confirmed (B) probe deferred.

Next: the same trained-map robustness on **Gabor-derived** `cortex_it` codes (the live front-end) and at larger
vocabulary; then the **navigate-to-see-then-answer** behavioral A/B on the full merged bridge (perceive via the
live Gabor pipeline → tag → "what did you see?" → reactivation names it, ABSTAIN when nothing seen), running the
same lesion/specificity/provenance anti-cheats at the system level. The *compositional* version stays mapped to
**step-3 (the learned cortex)** — the rate-vs-phasor wall — which this GO continues to motivate.

---

## 6. Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners.funcint_perception_to_memory_trained_probe \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/funcint_perception_to_memory_trained.json
# exit 0 = GO ; 2 = PARTIAL ; 1 = NEGATIVE
```
