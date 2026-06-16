# Functional integration cheap-first de-risk — PERCEPTION→MEMORY: GO

**Date:** 2026-06-16 (functional-integration arc, the (B) interaction).
**Verdict:** **GO** — an engram tag DRIVEN BY THE NAVIGATION PERCEPTION region (`cortex_it`, the ventral
object-identity code), when stimulated later, recalls the perceived concept SPECIFICALLY (perceive A → recall
A, not B), and the recall is lesion-confirmed to ride the perception→language synapses. Multi-seed (42/43/44 +
100/101/102 = **6/6**), all 4 objects.
**Probe:** `research/runners/funcint_perception_to_memory_probe.py` (CPU, `SIM_BACKEND=numpy`, ~seconds/seed).
**Result data:** `research/findings/raw/funcint_perception_to_memory_probe.json`.
**Design:** `docs/plans/2026-06-10-functional-integration-one-brain-design.md` §3.3 (the engram-tag mechanism),
§4 (the cheap-first style), §5 (anti-cheats), §6 (the rate-vs-phasor cross-code risk + WHY engram tags
sidestep it).
**Precedent:** mirrors `research/runners/funcint_lang_to_action_probe.py` (the (A) LANGUAGE→ACTION de-risk,
GO 2026-06-10) — same scaffold style (minimal merged bridge, static drive, lesion + provenance anti-cheats,
JSON out, exit codes).

---

## 0. Terms (defined once)

- **navigation perception region (`cortex_it`)** — the ventral "what"-stream object-identity code: per-object
  category ensembles (the navigation agent perceives *what* it is looking at, never coordinates). In the full
  navigation stack it is fed by the retina→V1→V2→IT Gabor pipeline; here it is the engram SOURCE.
- **conversational read-out (`language_output`)** — the Broca-area-like spelling region: its firing pattern,
  read by cosine to each word's code, is the agent's spoken/recalled word. The recall CHANNEL.
- **engram tag** (Tonegawa, catalog D.14) — the set of neurons that fired above a threshold during a window
  (`start_engram_recording` → run steps → `commit_engram_tag`); later `stimulate_tag` re-drives exactly that
  ensemble (causal recall). The tag IS the neurons, not a copied vector.
- **rate code vs phasor code** — the navigation perception is a *rate* code (`cortex_it` Izhikevich firing-rate
  ensembles); the conversational composer stores a *phasor* code (phases in [0,1)^D on resonate-and-fire
  neurons). They are not commensurable (design §6) — the central cross-code-transfer problem.
- **labeled line / topographic route** — a fixed per-object excitatory projection (object o's `cortex_it` band
  → object o's `language_output` band): the structural stand-in for the trained perception→word map, exactly as
  the (A) probe installed the trained `language_input → cortex_X` map structurally.

---

## 1. The load-bearing question (design §3.3 + §6)

Does an engram tag driven by the navigation PERCEPTION region (NOT a `language_input` cue), when stimulated
later, recall the perceived concept SPECIFICALLY (perceive A → recall A, not B), carried by the tagged synapses
(lesion-confirmed)? This shows a synaptic perception→memory write that **sidesteps the rate-vs-phasor wall**:
the tag is the perceived ENSEMBLE; it is never converted to a phasor.

This is the deeper "one brain" interaction — it closes the perception→memory→language loop synaptically — and
it is the pre-registered second build after (A) LANGUAGE→ACTION proved a cross-region synaptic route works on
the merged substrate.

## 2. The mechanism under test (design §3.3, fully synaptic, no `sim/` edit)

- **PERCEIVE object X (the environment presents the percept):** `cortex_it` carries object-identity ensembles.
  "The agent sees object X" = X's distinct `cortex_it` sub-ensemble fires. The probe renders this by driving
  X's orthogonal band of `cortex_it` (a legitimate sensory render — the environment presenting the percept,
  the perception-side analogue of the navigation Gabor/retina pipeline producing an IT object code). The
  load-bearing thing de-risked is the ENGRAM WRITE FROM PERCEPTION, not the (separately-validated) Gabor
  front-end.
- **WRITE the perceived ensemble to memory (the NEW (B) part — catalog D.14):** with X perceived,
  `start_engram_recording("seen_X")` → run the perception window → `commit_engram_tag("seen_X",
  region_filter=["cortex_it"])`. The tag IS the actual perceived `cortex_it` ensemble — **no phasor code, no
  Python copy of a percept vector; the neurons that fired ARE the memory.**
- **RECALL by neural reactivation:** later, `stimulate_tag("seen_X")` re-drives that ensemble; the reactivation
  propagates through `cortex_it → language_output`; the read-out reads which concept word the reactivation
  spells (cosine of the `language_output` firing pattern to each word's orthogonal code). **Recall = neural
  reactivation, NOT a Python lookup.**

### Cheapest faithful substrate (design §4 explicitly allows it)

A fresh brain-region-framework `SimulationBridge` with ONLY the perception region `cortex_it` + the
conversational read-out `language_output` + the `cortex_it → language_output` read-out pathway (the SAME
pathway the navigation builder wires when `enable_visual_cortex` + `enable_text_io`,
`g11_bg_runner.py:2660-2667`). No retina/V1/V2/striatum/parser/dlPFC/RF — none are needed to drive an engram
from a `cortex_it` ensemble and read the `language_output` reactivation. On CPU, seconds per seed.
Reuse-by-import: the navigation perception/read-out region shapes + the bridge engram API +
`sim.text_embeddings.orthogonal_drive_pattern`. **No `sim/` edit.**

The `cortex_it → language_output` selectivity ("apple's IT ensemble" → spells "apple") is what TRAINING grows
in the full stack; this cold probe does not train, so — exactly as the (A) probe installed the trained
`language_input → cortex_X` map as a per-direction topographic labeled line — the read-out is installed as a
per-object topographic labeled line. The thing being de-risked is the ENGRAM WRITE+RECALL, not the learning of
the read-out map.

---

## 3. Result — GO on all 6 seeds, all 4 objects

Per-object recall after stimulating the perception-driven engram tag. **CLEAN** = the perception engram intact;
**LESION** = the `cortex_it → language_output` read-out synapses zeroed (the engram still intact). "top1" = the
word the `language_output` reactivation spells (argmax cosine over the 4 object words); "perceived_score" =
cosine to the perceived object's spelling band; "margin" = top1 − runner-up. A recall is "correct" only if the
perceived object is the UNIQUE top-1 by ≥ 0.02 (`MIN_RECALL_MARGIN`) — so a floating-point tie does not count.

### Per-object detail (seed 42, representative)

| object (perceived) | CLEAN top1 | CLEAN perceived_score | CLEAN margin | correct | LESION top1 | LESION perceived_score | correct |
|--------------------|:----------:|----------------------:|-------------:|:-------:|:-----------:|-----------------------:|:-------:|
| apple              | apple      | 0.9996                | +0.9996      | ✅      | (none, 0.0) | 0.0000                 | —       |
| river              | river      | 1.0000                | +1.0000      | ✅      | (none, 0.0) | 0.0000                 | —       |
| dog                | dog        | 1.0000                | +1.0000      | ✅      | (none, 0.0) | 0.0000                 | —       |
| cat                | cat        | 1.0000                | +1.0000      | ✅      | (none, 0.0) | 0.0000                 | —       |

The CLEAN recall is **perfectly specific**: each tag's reactivation drives `language_output` to spell ONLY the
perceived object (cosine ≈ 1.0 to the correct word, exactly 0.0 to all three others). The LESION recall pattern
is **all zeros** — with the route cut, `language_output` literally never fires, so every cosine is 0.0 (the
top1 defaults to the first word but with margin 0.0 → `correct=False`).

### Roll-up (all 6 seeds)

| seed | CLEAN recall correct | LESION recall correct | provenance (every tag ⊆ cortex_it) |
|------|:--------------------:|:---------------------:|:----------------------------------:|
| 42   | 4/4                  | 0/4                   | ✅                                 |
| 43   | 4/4                  | 0/4                   | ✅                                 |
| 44   | 4/4                  | 0/4                   | ✅                                 |
| 100  | 4/4                  | 0/4                   | ✅                                 |
| 101  | 4/4                  | 0/4                   | ✅                                 |
| 102  | 4/4                  | 0/4                   | ✅                                 |

**24/24 perception-driven recalls correct (chance = 1/4 = 0.25, so ~4× chance per object, and unanimous);
0/24 survive the lesion.** Per-seed CLEAN perceived-scores range 0.997–1.000 (the small <1.0 values are OU-noise
in the `language_output` WTA, not a wrong word). Each tag is 26 `cortex_it` neurons (`top_k=60` after the
`region_filter`, bounded by the ~26-neuron excitatory band).

---

## 4. The anti-cheat controls (design §5) — all pass

1. **Lesion (primary).** A fresh bridge with the perception engram intact but every `cortex_it →
   language_output` read-out synapse zeroed (≈ 2.0–2.2k synapses/seed), then re-stimulate every tag: recall
   **collapses to 0/4 every seed**, and the `language_output` reactivation is **all-zeros** (every cosine 0.0).
   → the recall rides the `cortex_it → language_output` synapses specifically, not ambient leakage and not a
   Python path. With the percept still "stored" (the tag intact) but the read-out cut, recall fails — the
   load-bearing test of design §5.1.
2. **Specificity / cross-control.** Stimulate `seen_A` → recalls A NOT B: folded into "correct" (each tag must
   recall its OWN object as the unique top-1). The full cosine matrix is diagonal — `seen_apple` → apple=1.0,
   {river,dog,cat}=0.0; etc. Recall accuracy 24/24 vs chance 1/4. → the perception-driven tag reactivates the
   perceived concept, not another.
3. **Provenance (no Python value-copy).** ASSERTED STRUCTURALLY: every committed tag's indices are a SUBSET of
   the `cortex_it` region (the probe asserts this per seed; all pass). The tag = the perceived ENSEMBLE (the
   neurons that fired), committed with `region_filter=["cortex_it"]`. The ONLY current writes anywhere in the
   probe are (i) the orthogonal object code into `cortex_it` (the sensory render — the environment presenting
   the percept) and (ii) `stimulate_tag` (which drives the TAGGED neurons = the perceived ensemble, NOT a
   copied percept vector). No host code copies a percept vector into the recall drive.

So the recall is a genuine synaptic perception→memory write+read: the perceived ensemble is stored as the tag,
re-stimulation reactivates it, and the reactivation reaches language ONLY through the lesionable read-out.

---

## 5. HONEST SCOPE — this is a RECALL interaction, NOT composition (the rate-vs-phasor wall is step-3)

Per design §6, what this GO does and does NOT establish:

- **What it establishes:** a real, synaptic, perception→memory **recall** interaction that **sidesteps the
  central cross-code-transfer problem.** The engram tag stores the *perceived rate ensemble itself* and recalls
  it by re-stimulation — it NEVER converts the `cortex_it` rate percept into a phasor, so the rate-vs-phasor
  mismatch (design §6) does not arise. "I saw the apple" → later recall "apple", lesion-confirmed.
- **What it does NOT establish (deliberately out of scope):** *composition over perceived content.* You cannot
  yet algebraically bind the perceived apple into a novel role-filler fact (e.g. "the apple is red") through
  the composer — that genuinely requires the perceived (rate/grounded) code to enter the phasor bind/unbind
  algebra, which is exactly the rate-vs-phasor wall. A naive "wire `cortex_it` → composer role bank" would
  inject a rate pattern the exact-inverse VSA algebra cannot bind (the expected honest negative that maps the
  limit). **That boundary IS the scientific deliverable** — it motivates **step-3** (a learned spiking-cortical
  binder that reads correlated/grounded codes), the principled fix for the wall.
- **Cheap-probe idealizations (the honest engineering notes):** the read-out is a clean per-object topographic
  labeled line (the structural stand-in for the trained `cortex_it → language_output` map), so the CLEAN recall
  cosines are ≈ 1.0/0.0 — noise-free at this 4-object/256-neuron scale. The full navigation stack's read-out is
  a *trained, lossy, noisier* map; the perception ensembles are *Gabor-derived* (not direct band renders). This
  probe de-risks the WRITE+RECALL MECHANISM (does a perception-driven engram reactivate the concept through the
  synaptic route — unambiguously YES, lesion-confirmed), NOT the read-out's robustness under trained-map noise
  or the Gabor front-end (both separately validated; the trained-noisy version is the next build). This is the
  same scope discipline as the (A) probe (which de-risked the GATING, not the learning of the word→action map).

---

## 6. Verdict and next step

**GO.** A perception-driven engram tag recalls the perceived concept specifically and lesion-confirmedly, on
all 6 seeds and all 4 objects, with the lesion collapsing the recall to all-zeros and a clean structural
provenance audit. This is the second demonstrated cross-region synaptic interaction between the navigation
brain and the conversational brain — the perceived ensemble, written as an engram and reactivated, reaches the
conversational read-out by neuron firing + synaptic current alone, **without ever crossing the rate-vs-phasor
wall.**

**Proceed to the navigate-to-see-then-answer task (design §2 follow-on / §7 step 4):** on the full merged
bridge, place an object in the grid; the agent navigates until its `cortex_it` perceives the object (the live
Gabor front-end), at which point `start_engram_recording` → the perception window → `commit_engram_tag`; then,
queried "what did you see?", `stimulate_tag` → the `language_output` reactivation names the perceived object,
and ABSTAINS (the no-confab moat) when nothing was perceived. The 6-seed behavioral A/B then runs the same
anti-cheats at the system level (lesion the read-out → the answer collapses; isolated nav perceives-but-cannot-
report; isolated conv reports-but-never-perceives). The *compositional* version (binding perceived content into
a novel fact) stays mapped to **step-3 (the learned cortex)** — that is where the rate-vs-phasor wall is
climbed, and this GO is the empirical motivation for it.

No banking: this de-risk is GO, so the next action is the navigate-to-see-then-answer task harness, not a stop.

---

## 7. Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners.funcint_perception_to_memory_probe \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/funcint_perception_to_memory_probe.json
# exit 0 = GO ; 2 = PARTIAL ; 1 = NEGATIVE
```
