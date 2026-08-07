---
type: finding
status: boundary
date: 2026-08-07
mechanism: grounded-message-to-word
runner: research/runners/_grounded_retina_percept_naming_derisk.py
artifacts:
  - research/findings/raw/grounded_retina_percept_naming/full_seed42.json
---

# Driving the naming map from the NEURAL RETINA: raw V1 firing grounds the percept but does NOT clear the naming bar — a characterized grounded-perception→naming boundary (single-seed full-battery de-risk)

**One-line verdict.** The on-bridge spiking naming map
(`2026-08-07-grounded-message-to-word-onbridge-spiking-naming-map.md`, 6-seed GO) named the referent from a FIXED
percept code; that finding's own residual was *"the percept assemblies are deterministic rather than emerged from
vision."* This rung drives the naming map's percept from the project's REAL retina→V1 Gabor front end: each object
is rendered to an image, the retina sees it, `cortex_v1_simple` fires through the fixed Gabor RF bank, and that V1
firing — not a host code — drives the word pools through the learned V1→word synapses (decoded from word-pool spike
counts). The result is an **honest characterized boundary**: the percept is genuinely vision and every *structural*
anti-cheat holds, but naming DIRECTLY off raw V1 does **not** clear the accuracy bar — the raw V1 code lacks the
viewpoint-invariant object identity that the ventral stream builds downstream. Single seed 42, full example battery
(24 held-out jittered trials); runner's own verdict UNDEFINED. Artifact:
`research/findings/raw/grounded_retina_percept_naming/full_seed42.json`.

## What holds (the instrument works; the percept IS vision)

The naming input is the firing of real V1 simple cells driven ONLY by the retina — asserted, not assumed:

<!--derived-->
- **Retina-derived.** V1 fires on every inference trial and is non-zero for every object; at inference the ONLY
  externally driven region is the retina (word/V1 never injected). The percept carries object identity from vision.
- **Discriminable.** Mean cross-object V1 cosine 0.255 — the four rendered objects produce distinguishable V1 codes.
- **Innate RF bank frozen.** The fixed Gabor retina→V1 weights are byte-unchanged by teaching (`gabor_frozen=True`):
  the whole substrate is globally gain-frozen and ONLY the V1→word naming synapses open during the teacher event.
- **Render-faithful (1.0).** Over all 24 held-out presentations the body articulates exactly the word the brain
  decoded from spikes (`patient == chosen_word`, one WKV invocation) — no host override of the spoken word.
- **Gate + safety.** The spiking request/silence race routes hungry→request and sated→silence→zero word output;
  lesioning the V1→word pathway emits zero confident decodes (fails safe); a novel untaught object abstains.

## What does NOT hold (the boundary)

<!--derived-->
Single seed 42, full battery (n_ex=12, hold=6 → 24 held-out jittered trials per the values below):

<!--derived-->
| check | value | bar | verdict |
|---|---:|---:|---|
| naming accuracy from raw-V1 percept (held-out jitter) | 0.500 | ≥0.80 | FAIL |
| untrained random V1→word map (chance control) | 0.292 | — | ~chance (0.25) |
| learned separation (treatment − control) | 0.208 | >0.30 | FAIL |
| lesion of the V1→word pathway | 0.417 | <name | weak collapse |
| permutation followed | 0.458 | ≥0.80 | FAIL |
| permuted-map rejects original (sep) | 0.167 | >0.30 | FAIL |
| distinct words named (of 4 objects) | 3 | 4 | 2 objects collapse |

The runner's own verdict is **UNDEFINED** (the near-chance control contaminates the attribution: only 41.7% of the
0.50 is attributable to the learned map). Substantively this is a **weak/partial grounding**: the naming map learns
something above chance from raw V1, but the object-identity signal in the sparse (~40-cell) V1 response to thin
oriented-bar objects, destabilised by viewpoint jitter, is insufficient for the plastic V1→word Hebbian map to cleanly
separate four referents. Two objects collapse to one word; the permutation control genuinely cannot separate because
the held-out V1 percepts overlap too much.

## Why (the missing companion stage) and the biological surpass

The ventral stream does not name off V1. V1 simple cells are orientation/position/phase filters with NO viewpoint or
position invariance; object identity that supports naming is constructed downstream through **V2 → V4 → IT** (Tanaka
1996; Felleman & Van Essen 1991), where receptive fields grow and pool toward invariant object codes. Naming directly
from raw V1 skips that stage — which is exactly what this boundary measures. The project's own precedent agrees:
EMERGE-34 (`2026-07-02-emerge34-perception-grounded-emergence-GO.md`) grounded categories in the SAME Gabor/V1 front
end but only after a **competitive spatial pooler** built a stable column code per object; it did not name off raw V1.

**The surpass (named, not deferred):** insert the ventral invariance stage between V1 and the naming map — the
committed `cortex_it` object path, or the EMERGE-34 competitive pooler — so the naming map reads a viewpoint-invariant
IT/pooled object code (retina→V1→IT/pooler→word) rather than raw V1. The percept stays fully vision-derived; the
missing companion process is the pooling/invariance stage, not the naming rule (which is 6-seed GO on stable codes).
The `2026-08-01-perception-v2it-...` scoping already flags the deployed V2/IT path as inert and standardises on the
V1→pooler codon — so the concrete next rung is naming off the EMERGE-34 pooled code, on the shared bridge.

## Scope / honesty

Single seed, full battery — a de-risk that CHARACTERISES the boundary, not a 6-seed generalization claim. Declared
disabled (as in the on-bridge rung): structural plasticity / threshold homeostasis / synaptic scaling are frozen
around the naming map; the carrier frame and the oriented-bar object render are named host scaffolds; the WKV
articulation is the fixed off-bridge language circuit. The banked method here is *naming off RAW V1*; the boundary is a
verdict on THAT method, and the next method (naming off a pooled/IT invariant code) is named above.

## Next mechanism

1. Insert the EMERGE-34 competitive pooler (or `cortex_it`) between V1 and the naming map; re-run this battery on the
   pooled code. Expect the learned separation + permutation separation to recover if the pooled code is object-stable.
2. If the pooled code clears the bar single-seed, run the 6-seed generalization with
   `research.runners._grounded_retina_percept_naming_derisk --seeds 42 43 44 100 101 102` (this runner is the raw-V1
   baseline the pooled version must beat).
