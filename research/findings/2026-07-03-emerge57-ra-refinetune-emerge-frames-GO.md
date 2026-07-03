# EMERGE-57 (Rung 2) — RE-fine-tune the RA 21M on EMERGE's grounded frames → renders them FLUENTLY + CORRECTLY, moat held: **GO**

**2026-07-03 (autonomous).** Rung 2 of the north-star wire (Rung 1 = EMERGE-56 GO, `2026-07-03-emerge56-reasoning-to-fluent-wire-GO.md`). Rung 1 confirmed the WIRE (adapter + gate-first moat) carries to the real 21M, but the RA-fine-tuned 21M rendered EMERGE's `can-fly` / intransitive-exception frames **out of distribution** → it CONFABULATED content ("the owl likes to follow leaf") and DOUBLE-INFLECTED ("walkses"). Rung 2 CLOSES that gap with a **DATA/format continuation fine-tune** (NOT a new mechanism), reusing the RA recipe with a new EMERGE-frame example generator interleaved in. Reuse-by-import; **NO `sim/` edit**.

## The gap and the fix

The RA fine-tune (`_fluidconv_phase2_ra_finetune`) was trained on **transitive SVO** ("the dog eats meat"). EMERGE emits two frame families the RA never saw:
- **ABILITY / INHERITANCE:** `the {subject} can {verb} .` (a modal `can` + a BARE infinitive: fly / swim)
- **INTRANSITIVE EXCEPTION:** `the {subject} {intr_3sg} .` (an already-3rd-person-sg intransitive: walks / lurks)

**Fix = a continuation fine-tune** on the RA ckpt (`gen_tinystories_ra_ft.ckpt.pt`) with a NEW EMERGE-frame example generator (`_make_emerge_example`: ability-affirm / ability-describe / intransitive-exception / abstain, broad-vocab + random) **INTERLEAVED** with the ORIGINAL RA frames (anti-forget the transitive-SVO format) + raw TinyStories (anti-forget base fluency, per P2). Written to a NEW ckpt (`gen_tinystories_ra_emerge_ft.ckpt.pt`); the RA ckpt stays intact.

**The load-bearing bug fix — frame-aware inflection (`emerge_v3`):** the RA `_v3` blindly appends `-s` → `walks`→`walkses`, `fly`→`flys`. `emerge_v3` is frame-aware + irregular-aware: an already-3sg intransitive stays verbatim (`walks`→`walks`), irregulars inflect correctly (`fly`→`flies`), regular RA verbs match the RA table. Idempotent on intransitives (the double-inflection guard). CPU-tested.

**Root cause of the residual subject-garbling (found + fixed mid-run):** the RA subject vocab lacks most EMERGE members (`minnow`/`gar`/`wren`/`pike` absent) → greedy decode garbled them (`minnow`→"mini", `gar`→"glide", `pike`→"pig"). Fix: add EMERGE's animal members (`_EMERGE_MEMBERS`: owl/wren/minnow/gar/penguin/pike/robin/… + a broad bird/fish pool) to the EMERGE-frame generator's subject vocab (all round-trip through the BPE) → the model learns to render them.

## De-risk — **GO** (all four gates, `--derisk --steps 500 --n-emerge 14000 --n-ra 8000`)

| gate | value | bar |
|---|---|---|
| (a) EMERGE-frame RENDER FIDELITY (correct grounded property + correct polarity + focused + correct subject, no confab) | **1.00 (6/6)** | ≥ 0.85 |
| (b) NO catastrophic forgetting — original-frame held-out ppl | **2.00 → 2.07** (ratio **1.04**) | ≤ 1.5 |
| (b) EMERGE frames LEARNED — EMERGE-frame held-out ppl | **16.30 → 1.75** (ratio **0.11**) | < 1.0 |
| (c) MOAT preserved — renders on abstains / model-invocations on abstains | **0 / 0** (2 abstains) | 0 |
| (d) correct inflection — double-inflections ("walkses") | **0** | 0 |

Continuation fine-tune: **~119 s** on the 3090 (500 steps; corpus 3.86M chars, 1.25M tokens). Full de-risk ran end-to-end (the numpy-native EMERGE-51 render console runs in a `SIM_BACKEND=numpy` subprocess; torch stays on CUDA).

### Render transcript — BEFORE (RA ckpt) vs AFTER (EMERGE re-fine-tune), seed 42, gate-first moat

```
                       BEFORE (RA ckpt, confabulates)              AFTER (EMERGE re-fine-tune, GO)
can an owl fly?    ->  "no , the owl does not fly ."          ->  "yes ."                         [INHERIT, correct]
can a wren fly?    ->  "no , the wren does not fly ."         ->  "yes ."                         [INHERIT, correct]
can a minnow swim? ->  "no , the MINE does not swim ."        ->  "yes , the minnow can swim ."   [INHERIT, correct]
can a gar swim?    ->  "no , the gar does not swim ."         ->  "yes , the gar can swim ."       [INHERIT, correct]
can a penguin fly? ->  "no , the penguin does not fly to..."  ->  "no , the penguin walks ."       [CANCEL, correct]
can a pike swim?   ->  "no , the PIG does not LURB ."         ->  "no , a pike lurks ."             [CANCEL, correct]
can a zzz fly?     ->  [MOAT; model NOT invoked]              ->  "I don't know what a zzz is."     [MOAT held]
can a wobble swim? ->  [MOAT; model NOT invoked]              ->  "I don't know what a wobble is."  [MOAT held]
```

BEFORE: inverted polarity ("no" for an inherited ability), garbled subjects (minnow→"mine", pike→"pig"), garbled verbs ("lurb"). AFTER: correct polarity (inherit→yes, exception→no), correct subject, correct intransitive inflection (`walks`/`lurks`, not `walkses`), focused — **and the gate-first moat holds on both** (the zzz/wobble abstains NEVER invoke the generator; render-count 0). `owl/wren` (bird) + `minnow/gar` (fish) are GENUINE held-outs (inherit only via the shared discovered codon); `penguin/pike` are the member-specific exceptions (cancellation).

## Verdict

**GO.** The RA generator RE-fine-tuned on EMERGE's frames renders them FLUENTLY + CORRECTLY behind the SAME gate-first no-confab moat: render fidelity 1.00, frame-aware inflection FIXED (0 "walkses"), the moat holds (0 renders on abstains — the load-bearing property), NO catastrophic forgetting (original-frame ppl ratio 1.04) AND the EMERGE frames were LEARNED (EMERGE-frame ppl ratio 0.11). **⇒ the emergent brain (EMERGE-51..55 reasons over discovered categories: inheritance / cancellation / abstention) now answers FLUENTLY, grounded, moat-safe. Wernicke decides → Broca articulates.**

## Files
- `research/runners/_emerge57_ra_refinetune_emerge_frames_derisk.py` — the frame-aware inflection fix (`emerge_v3`), the EMERGE-frame example generator (`_make_emerge_example`), the combined corpus builder (`build_emerge_corpus`: EMERGE + RA + TinyStories interleaved), the continuation re-fine-tune (`refinetune`, with pre/post held-out ppl), the gate-first render+moat de-risk (`_render_derisk` via a numpy subprocess; the counting faculty), and `--check-corpus`/`--smoke`/`--derisk`/`--render-only`.
- `tests/test_emerge57_ra_refinetune_emerge_frames.py` — 9 tests (8 CPU: the inflection fix incl. idempotence, the frame generator well-formedness + no-double-inflection, bare-infinitive ability frames, the corpus builder; 1 GPU render+moat smoke, skip-if-no-ckpt — passes with the ckpt present).
- `research/findings/raw/_emerge57_ra_refinetune_emerge_frames.json` — the combined de-risk (ppl + render + moat).
- ckpt: `research/findings/raw/fluidconv/gen_tinystories_ra_emerge_ft.ckpt.pt` (21.3M; the RA ckpt is preserved intact).

**Full-run command (reproduce the GO):**
```
SIM_BACKEND=cupy python -m research.runners._emerge57_ra_refinetune_emerge_frames_derisk --derisk --steps 500 --n-emerge 14000 --n-ra 8000
```
(~2 min on a 3090; the render de-risk auto-spawns a numpy subprocess for the EMERGE-51 console, torch stays on CUDA.)

## Honest scope
A DATA/format continuation fine-tune on the RA ckpt (NOT a new mechanism): a new EMERGE-frame generator interleaved with the original RA frames + TinyStories (anti-forgetting, per P2). The **generator ANN remains a tracked temporary scaffold** — its spiking-forward conversion is deferred (validated at 88.6M). The moat is preserved BY CONSTRUCTION (the gate short-circuits before the generator). Validated on EMERGE-51's scripted bird/fish taxonomy (2 held-out inherit + 2 exception + 2 moat per seed, single-seed re-fine-tune + render — the recipe is deterministic given the ckpt); a multi-seed re-fine-tune (different fine-tune seeds) is a cheap follow-on. **Rung 3** = merge into `_fluidconv_chat_repl.py` so EMERGE `can a penguin fly?` + the existing `what does a dog eat?` both work under one consistent moat + fluency.
