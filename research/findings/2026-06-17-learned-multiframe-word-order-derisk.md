# Learned multi-frame word order — cheap-first de-risk (the EASY half of productive syntax)

**Date:** 2026-06-17 · **Verdict: GO (6/6 seeds), on the spiking substrate.**
**Runner:** `research/runners/_phaseB_learned_multiframe_word_order_derisk.py` · **Raw:**
`research/findings/raw/_phaseB_learned_multiframe_word_order.json`
**Pre-registration:** `research/findings/2026-06-17-productive-syntax-scoping.md` (Option 1, ranked #1) — bars FROZEN
before data; FRACTIONAL ≥5/6-of-seeds pass bar (`feedback_6seed_validation`). No `sim/` edit; reuse-by-import.

---

## 1. What this de-risks (the precise capability) — and how it differs from the CYCLE-106 precursors

The conversational agent's grammar is a set of **hardcoded templates**: one fixed SVO frame; word order = a
hardcoded primacy tuple (`_phaseB_serial_order_spiking_derisk.PRIMACY_pA`). **Productive word order** means a
grammatical **frame is a LEARNED primacy gradient over the grammatical ROLE slots** (subject/verb/object), not a
hardcoded tuple — so a **new frame's order generalizes to fillers it was never trained on** (the order is over
ROLES, not words), and a **context cue SELECTS the frame** (dlPFC-style). This is the smallest genuinely-productive
step: the agent **generates a sentence in a word-order frame it was NOT given as a template**.

The mechanism composes only **validated** pieces: the rate-coded **competitive-queuing** serial-order generator
(Grossberg 1978 / Bullock-Rhodes 2003; catalog **G.07/H.19**; `neural_serial_order_renderer` /
`_phaseB_serial_order_spiking_derisk` — graded current → spiking-RATE ranking = emission order) + a **per-frame
Hebbian primacy gradient** + a **Hebbian cue→frame selection map**. Nothing is hardcoded: both the per-frame role
order and the cue→frame routing are learned.

**Distinct from the CYCLE-106 precursors** (`_phaseB_serial_order_multiframe[_spiking]_derisk.py`), which tested
frame-CONDITIONED order via a *cross-frame* control only (true 1.000 / cross 0.000–0.005, 6/6). This productivity
de-risk **adds the load-bearing controls those lacked**:

| Element | CYCLE-106 multiframe precursor | THIS de-risk |
|---|---|---|
| Two frames over the same role slots | ✅ (F0=SVO, F1=[2,0,1]) | ✅ **NON-NATIVE 2nd frame = verb-initial VSO=[1,0,2] "ran dog north"** |
| Held-out fillers | train/held split | ✅ disjoint held-out filler tuples; emission tested on them |
| Cross-frame control | ✅ | ✅ (implicit — the two frames differ) |
| **Frame-SELECTION (cue → frame)** | ❌ | ✅ **learned Hebbian cue→frame WTA, scored** |
| **PERMUTED-FRAME control (shuffle frame→gradient)** | ❌ | ✅ **the load-bearing discriminator** |
| **LESION control (remove gradient)** | ❌ | ✅ |
| **No-confab MOAT** | ❌ | ✅ |
| `--seeds` argparse | ❌ | ✅ |

---

## 2. Result — per-seed, all controls (6 seeds, SPIKING substrate, CuPy)

The order read-out is **real spikes**: the selected frame's learned primacy gradient is graded **external current**
into the fact's driven concept pools (`build_pool_bridge`/`pool_rates`), and the **per-pool spiking-rate ranking** is
the emitted order. Frame-selection / permuted-frame / lesion / moat are layered on top.

| Seed | held-out SVO | held-out **VSO** (non-native, learned) | **frame-SELECT** | permuted-FRAME | lesion | moat | seed |
|---|---|---|---|---|---|---|---|
| 42  | 1.000 | 1.000 | 1.000 | 0.333 | 0.333 | 1.000 | **PASS** |
| 43  | 1.000 | 1.000 | 1.000 | 0.333 | 0.417 | 1.000 | **PASS** |
| 44  | 1.000 | 1.000 | 1.000 | 0.333 | 0.222 | 1.000 | **PASS** |
| 100 | 1.000 | 0.944 | 0.958 | 0.333 | 0.306 | 1.000 | **PASS** |
| 101 | 1.000 | 1.000 | 1.000 | 0.333 | 0.500 | 1.000 | **PASS** |
| 102 | 1.000 | 1.000 | 1.000 | 0.333 | 0.250 | 1.000 | **PASS** |
| **mean** | **1.000** | **0.991** | **0.993** | **0.333** | **0.338** | **1.000** | **6/6 GO** |

- **GO bars (frozen):** held-out order ≥ 0.90, frame-selection ≥ 0.90 — both met on **6/6** (≥5/6 required).
- **PERMUTED-FRAME collapse (the discriminator):** shuffle the frame→gradient mapping (the VSO label now points at a
  different frame's gradient) → VSO-order accuracy **collapses to 0.333 ≈ chance** (empirical chance-order baseline
  0.343). This is the load-bearing proof: the produced order is the **LEARNED frame**, not a fixed/native bias. If
  the "order" had been the native SVO bias all along, the shuffle would not have collapsed it.
- **LESION collapse:** remove the learned gradient (equal drive) → **0.338 ≈ chance** — the learned gradient is
  load-bearing.
- **MOAT:** an unfilled role (None) or an unknown (out-of-vocab) filler **abstains (None), never confabulates** —
  1.000 every seed.
- **No native regression:** SVO held-out = 1.000.

**Collapse magnitudes:** permuted-frame **0.991 → 0.333** (−0.658, to chance); lesion **0.991 → 0.338** (−0.653, to
chance). Both controls cleanly destroy the capability, as required.

**Example sentence (seed 42, held-out filler tuple `(15, 6, 1)`, mock-spelled):**

> native **SVO** → `"dog ran north"`  ·  **LEARNED non-native VSO** → `"ran dog north"`

Same fillers, **different learned frame order** — the verb-initial order was never a template; it is produced by the
VSO frame's learned primacy gradient driving the spiking competitive-queuing read-out.

---

## 3. Honest scope / limitation (what GO does and does NOT claim)

- **GO claims:** the *architecture* of productive word order works on the spiking substrate — grammatical roles are
  the order-encoded slots, a frame is a **learned** (not hardcoded) per-frame primacy gradient, the frame is
  **selected from a context cue**, the order is produced by **real spikes** (rate-coded CQ), and the load-bearing
  **permuted-frame** + **lesion** controls collapse to chance while the **moat** holds. This is genuinely-productive
  syntax (the agent emits a frame it was never templated with), staying entirely inside the **single-attribute /
  role-filler** regime already validated on real LIF — it never touches the bundling/SNR walls.
- **GO does NOT claim a hard generalization gradient.** The primacy gradient is over **3 roles** and is learned
  filler-agnostically, so "generalizes to held-out fillers" is true **by construction** (the order does not depend
  on which concept fills a role — which is exactly the point: order is over ROLES, not words). The held-out test
  therefore confirms *order-over-roles*, and the **discriminating** evidence is the permuted-frame collapse +
  frame-selection, not the held-out accuracy alone. A reader should read this as "the learned-frame *mechanism* is
  GO," not "a difficult generalization was overcome."
- **Seed-100 dip (VSO 0.944, select 0.958):** honest spiking-read variance — on one held-out tuple two pools tied
  in spike rate within the read window, mis-ordering one slot. Still well over the 0.90 bar; characterizes the
  rate-read tie risk (the documented dt-bound rank-coding tie), not a mechanism failure.
- **Frame inventory = 2 (SVO + VSO).** A 3rd/4th-frame capacity stress + a richer non-native frame (a 4-slot
  ditransitive S-V-recipient-theme) are the natural follow-ons; the runner is structured to extend `FRAMES`.

---

## 4. The HARD half is explicitly parked (not in scope here)

Per the scoping doc, productive syntax is **two problems**. This de-risk is the **easy half** (novel word-order
frames). The **hard half** — *arbitrary recursion depth* and *non-adjacent agreement* — collides head-on with the
two named point-neuron walls and is **NOT** addressed here:

- arbitrary recursion depth → the **nested-composition / SNR wall** (`2026-06-02-full-320…hierarchical null`);
- non-adjacent agreement / long-range dependency → the **multi-attribute BUNDLING NEGATIVE**
  (`2026-06-16-onsubstrate-learned-binder…bundling-NEGATIVE`).

Those reduce to the *same* superposition-inverse / multiplicative-binding operation the project has four-times found
to be the point-neuron limit; the two routes past them are **(a)** the dendritic multiplicative substrate (Option 4)
or **(b)** the Assembly-Calculus projection-parser's disinhibition recursion control (Option 2). That fork is the
*next* scoping question — **after** this GO localizes the easy half as solved. Do not mis-scope this GO as "all of
syntax."

---

## 5. Next move (the GO routes to)

1. **Promote to the agent (default-off):** wire a **learned multi-frame `render`** into `BrainConversationalAgent`
   (the renderer selects a frame from an utterance-type cue, applies that frame's learned gradient through the
   existing `NeuralSerialOrderRenderer` CQ read-out) behind a default-off flag, preserving the no-confab moat and
   the full conversational matrix — the near-drop-in this de-risk's reuse-by-import structure was built for.
2. **Capacity + richer frames:** extend `FRAMES` to a 3rd/4th frame and a 4-slot ditransitive; re-run to localize
   whether selection or capacity is the next sub-problem (the BOUNDARY branch the runner pre-registers).
3. **Then** open the hard-half fork (deep recursion + agreement: AC disinhibition-control vs. dendritic
   multiplication).

---

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_learned_multiframe_word_order_derisk \
    --seeds 42,43,44,100,101,102
# numpy (cheap-first / CI) path is byte-equivalent for the order logic and runs in ~2s:
SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_multiframe_word_order_derisk --seeds 42,43
```

Frozen bars: held-out order ≥ 0.90, frame-selection ≥ 0.90, permuted-frame + lesion must collapse to chance, native
SVO un-regressed, ≥5/6 seeds. **Result: GO (6/6).**
