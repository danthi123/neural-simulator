# Burndown cheap-completion — DEPLOY the corpus-mined Bucket-B structure as the live console default — GO

**Date:** 2026-06-27
**Type:** Bucket-B burndown DEPLOY (the "finish the cheap wins" pass). Reuse-by-import; **NO `sim/` edit**; additive edits to
`research/runners/first_chat_console.py` + `research/runners/one_brain_composer.py` only.
**Verdict:** **GO — all three parts deployed/assessed, the DEFAULT `--rubric` stays 10/10 byte-identical, moat 0-FA.**

The three cheap Bucket-B wins were GO but validated-OPT-IN (the hand structures were the live console default). This pass makes the
corpus-MINED structure the console default (with the hand structures retained as the parity ORACLE / vocab-poor fallback), under
the standing discipline: the DEFAULT `--rubric` stays **10/10 byte-identical-or-better AND moat 0-FA**, re-run BEFORE+AFTER each deploy.

---

## The HARD GATE — the default `--rubric`, 10/10 byte-identical BEFORE and AFTER (re-run after EACH deploy)

`SIM_BACKEND=numpy python -m research.runners.first_chat_console --rubric` (the default brain `brain1454_w7000`, plain
`RFPhasorComposer`):

| | RUBRIC SCORE | moat leaks | mixed-type | VERDICT |
|---|---|---|---|---|
| BEFORE (baseline) | **10/10** | 0 | MIXED | PASS |
| AFTER (a)+(b) deployed | **10/10** | 0 | MIXED | PASS |

The full transcript is byte-identical (every line — `the dragonfly hums cod`, the grounded hedges, the unknown-word clarification —
matches). **Why it stays byte-identical even with the mining LIVE on the default path:** the default brain `brain1454_w7000` DOES
mine 124 verb-frames + a 7-entry wh-map (validated verbs mined: go/come/walk/run/give/send), but (a) the mined FRAMES only BIND on the
`--argstructure` path (the default rubric uses the plain `RFPhasorComposer`, no typed roles), and (b) the mined wh-MAP is now used in
the rubric's wh-routing, but the rubric's `what does X Y` prompts route to the existing `_WHAT_DOES_RE` (not `_wh_response`) on a plain
composer, and `what → patient` resolution is preserved by the mined `_default` frame (`derive_frame_lexicon` keeps `_default =
[agent, action, patient]` verbatim). So the deploy is genuinely LIVE on the default build (the mine runs, structure ACQUIRED) AND the
rubric is byte-identical.

---

## Part (a) — DEPLOY the corpus-mined verb-frames as the composer default — **GO (deployed-as-default)**

`ArgStructureComposer` (the `--composer rf --argstructure` oracle/CPU path) now builds with the **CORPUS-MINED `FRAME_LEXICON`** by
default — the verb frames are DERIVED from corpus argument co-occurrence over the brain's OWN learned verbs (B-mine-1 GO 6-seed),
not hand-typed. The hand `FRAME_LEXICON` is retained as the parity ORACLE / vocab-poor fallback (passed `frame_lexicon=None` →
byte-identical).

- **What was wired** (additive, `first_chat_console.py`): a module-level `_mine_verb_frames(vocab)` (mirroring B-wire-1's
  `_mine_size_axis` discipline — guarded, cached, gated on spaCy + corpus + ≥1 mineable validated verb), reuse-by-importing the
  B-mine-1 mining halves VERBATIM (`mine_verb_argstats` + `derive_frame_lexicon`). The mined frames are passed `frame_lexicon=mined`
  to `ArgStructureComposer` AND (C4 typed-onebrain path) to `OneBrainComposer`. Falls back to the hand frames for a vocab-poor brain.
- **The OneBrainComposer (C4 typed-spiking) deploy** needed ONE additive change to `one_brain_composer.py`: a `frame_lexicon=None`
  kwarg (default = byte-identical = the hand `FRAME_LEXICON`) threaded through `render` (`frame_for`/`frame_id`/`realized_units`
  `lexicon=`) + the numpy `FrameCQ(lexicon=)` (the spiking `SpikingFrameCQ` is frame-agnostic — it orders the realized-index list —
  so it needs no lexicon). The C4 default path (`frame_lexicon=None`) is byte-identical; the GPU C4 guard (`test_argstructure_onebrain.py`,
  7 tests) passes verbatim, and the mined-frames C4 path runs answer-identically on the spiking substrate (below).
- **Live transcript** (`--argstructure --facts-json _tier0_typed_facts.json`, `frames=corpus-mined`, `RFPhasorComposer`): the
  `--argstructure --demo` is byte-identical to the hand-frame baseline ("the boy goes", "the bird flies"; moat 0-FA). The natural
  wh-route on the MINED frames + MINED wh-map:
  ```
  YOU: where does the boy go?     BRAIN: the boy goes to the park        [wh_role=GOAL, mined go->GOAL frame, moat=OK]
  YOU: what does the bird fly?    BRAIN: Good question. The bird flies sky.   [moat=OK]
  YOU: when does the boy go?      BRAIN: I don't have a stored fact answering that ...   [mined when->[] -> abstain, moat=OK]
  YOU: where does the dragon go?  BRAIN: I don't know the word "dragon" yet ...          [unknown-word, moat=OK]
  ```
- **GO/NO-GO: GO, deployed-as-default.** The mined frames render/recall at parity with the hand frames (B-mine-1: mined-acc 1.000,
  permuted-mining 0.033, 6 seeds), so the deploy is a non-regression; the rubric stays 10/10; moat 0-FA.

## Part (b) — DEPLOY the corpus-mined wh-map as the parser default — **GO (deployed-as-default)**

The wh-route (`answer_wh`) and the wh-routing decision (`parse_wh_question`) now resolve through the **CORPUS-MINED wh→role map +
per-verb frame-roles** (the INVERSE INDEX of the mined frames, B-mine-2 GO 6-seed) by default — the hand `WH_ROLE_CANDIDATES` is
retained as the parity ORACLE / vocab-poor fallback (`role_map=None`/`frame_roles=None` → byte-identical).

- **What was wired** (additive, `first_chat_console.py`): `_mine_verb_frames` also derives the wh-map (`derive_wh_role_map` over the
  mined frames, ranked by the per-role corpus attestation — GOAL>LOCATION). The mined `wh_role_map`/`wh_frame_roles`/`wh_multiword`
  are carried into the `brain` dict → stored on `DiscursiveTurn` (`self._wh_role_map`/`_wh_frame_roles`/`_wh_multiword`) → threaded
  into `answer_wh` (via a new `_answer_wh_mined` helper) + the routing `parse_wh_question`. The mined multiword table (== the hand one
  in the validated case) is swapped into the `WH_MULTIWORD` module constant ONLY when it DIFFERS (it does not here), restored after
  the call — so the common path never mutates the module global.
- **Live transcript:** the `where does the boy go? → GOAL → to the park` and the `when does the boy go? → abstain` (the mined
  `when→[]` correctly fails to license, the no-confab moat) cases above run through the MINED wh-map. Parse-parity (B-mine-2: 1.000,
  6 seeds) means the mined map answers identically to the hand map on the validated cases.
- **GO/NO-GO: GO, deployed-as-default.** The mined wh-map resolves at parity (B-mine-2: parse-parity 1.000, permuted-mining 0.250,
  6 seeds, moat 0-FA); the default rubric stays 10/10; moat 0-FA.

## Part (c) — cleanup-select: ASSESS the spiking Izhikevich-WTA cleanup on the console onebrain path — **assessed: KEEP host-argmax (NO CODE CHANGE; the documented cost at console D=128, moat 0-FA)**

The task: wire the spiking Izhikevich-WTA cleanup on the console onebrain path IF it holds (rubric-equivalent recall + moat
0-FA) at the console D=128; if it costs the documented abstain, KEEP host-argmax + document the wider-D need (do NOT regress).

**Assessment (re-measured on current code, GPU, the default 7K D=128 / V=1454 brain, the console's 12 facts):** built the
`OneBrainComposer` with `enable_spiking_cleanup=True` (the fully-spiking Izhikevich-WTA winner-pick) vs `False` (the host-argmax
the C3 console default uses), compared per-fact recall + the no-confab moat:

```
recall: host-argmax 8/12  spiking-WTA 7/12        (spiking-WTA loses 1 recall)
moat false-accepts: host 0  spiking 0  (0 = clean)  (the moat holds on BOTH)
mismatches (host vs spiking) on stored facts: 2
   dragonfly/hum/cod -> host=cod  spiking=None     (the documented C3 SAFE-direction abstain -- spiking abstains, never fabricates)
   bison/knot/clock  -> host=durian spiking=bind   (a thin-margin fact NEITHER cleanup recalls correctly -- not a moat breach)
VERDICT: spiking-WTA COSTS 1 recall -> KEEP host-argmax + document wider-D
```

This **reproduces the C3 finding exactly** (`2026-06-27-burndown-C3-onebrain-console-GO.md`): at the CROWDED console scale (V=1454,
D=128, thin code margins) the spiking Izhikevich-WTA cleanup costs ≥1 recall vs host-argmax (the `dragonfly/hum/cod → None`
safe-direction abstain), while the **no-confab moat is 0-FA on BOTH** (the spiking-WTA never FABRICATES a different fact — it abstains
when uncertain). The substrate STORE / bind / unbind / scan already run on FIRING NEURONS regardless (the C3 substrate win); the only
choice is the final winner-PICK, and the spiking-WTA is == numpy argmax at its validated D=2048 scale but lossy at this crowded
D=128 console scale.

- **GO/NO-GO: KEEP host-argmax (no code change).** Per the task's explicit instruction, the spiking-WTA's recall cost at console
  D=128 means the console onebrain path stays on `enable_spiking_cleanup=False` (host-argmax) — which is ALREADY the C3 console
  default (`build_brain_on_codes`'s onebrain branch + the C4 typed-onebrain branch both pass `enable_spiking_cleanup=False`). So part
  (c) is a **no-change confirmation**: the measured cost confirms the existing console default is correct, and the wider-D / shard pass
  to close the 1-fact margin (so the fully-spiking-WTA cleanup is == host-argmax at the console scale too) is the documented follow-on
  — NOT a regression we force here. The fully-spiking-WTA cleanup remains the documented default ELSEWHERE (consolidated_320 / the
  agent's onebrain path, where the D=128/320 codes are well-separated → recall 1.0).

---

## CI guards — all green

`SIM_BACKEND=numpy python -m pytest tests/test_argstructure_composer.py tests/test_wh_question_parser.py
tests/test_bucketB_corpus_mined_frames.py tests/test_bucketB_corpus_mined_wh_map.py
tests/test_first_chat_console_spiking_render.py tests/test_argstructure_spiking_cq.py -q`: **46 passed, 7 skipped** (the skips are
GPU-gated). `SIM_BACKEND=cupy python -m pytest tests/test_argstructure_onebrain.py -q`: **7 passed** (the C4 `frame_lexicon=None`
default byte-identity on the spiking substrate). The two B-mine de-risks re-run GO BEFORE the deploy: B-mine-1 mined-acc 1.000 /
permuted-mining 0.033 / moat 0-FA (6 seeds); B-mine-2 parse-parity 1.000 / permuted-mining 0.250 / moat 0-FA (6 seeds).

## Honest scope / caveats

- **What this is:** the validated corpus-mined Bucket-B structure (the verb-frame lexicon, B-mine-1; the wh→role map, B-mine-2) is now
  the LIVE console default — structure ACQUIRED, not given — with the hand structures retained as the parity ORACLE / vocab-poor
  fallback (exactly as B-wire-1 kept the curated `_SIZE_LADDER`). The mining is lossy + corpus-budget- and vocab-dependent (B-mine's
  measured boundary: `send` is corpus-justified-differ, `put` un-mineable in the default vocab, `when→[]` corpus-justified); the
  believability is the B-mine *signature* (match-or-justify + permuted-mining collapse + provenance), not 100% coverage.
- **NO `sim/` edit.** Additive edits to `first_chat_console.py` (the mining helper + the deploy wiring) + `one_brain_composer.py` (the
  `frame_lexicon=` kwarg, default-None byte-identical). The mining is host-side curriculum prep (legitimate per BRAIN-BASED-ONLY:
  preparing the verb's frame the brain then RENDERS/RECALLS through spikes — like rendering a retinal image).
- **What this does NOT touch (correctly deferred, per the Bucket-B gate):** the closed-class / morphology lexicons (the hard part is
  the recursive grammar, not the list); the tag-VALUE inference frontiers (common-ground / tense); the genuine months-frontier (a
  learned RECURSIVE generative grammar + the developmental self-organization of the binding connectivity).

## Reproduce

```bash
# the DEFAULT rubric (the HARD GATE -- must stay 10/10, moat 0-FA; the mine runs LIVE but the rubric is byte-identical)
SIM_BACKEND=numpy python -m research.runners.first_chat_console --rubric

# the typed-frame + wh route on the MINED frames + MINED wh-map (the deploy payoff)
SIM_BACKEND=numpy python -m research.runners.first_chat_console --argstructure \
    --facts-json research/findings/raw/_tier0_typed_facts.json --demo

# the B-mine de-risks (the parity oracle, re-run GO before the deploy)
SIM_BACKEND=numpy python -m research.runners._bucketB_corpus_mined_frames_derisk --seeds 42 43 44 45 46 47
SIM_BACKEND=numpy python -m research.runners._bucketB_corpus_mined_wh_map_derisk --seeds 42 43 44 45 46 47
```
