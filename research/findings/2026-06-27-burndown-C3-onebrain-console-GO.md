# Burndown C3 — `--composer onebrain` on the first-chat console: GO (2026-06-27)

**Spec:** `2026-06-27-burndown-bucketA-build-plan.md` §2 C3 — add a `--composer {rf,onebrain}` switch to the
first-chat console so the whole flat who/what pipeline (recall / bind / cleanup / yes-no / chain-of-thought /
generation) runs on the persistent spiking `OneBrainComposer`, keeping the numpy `RFPhasorComposer` as the CPU
test-oracle default. C1+C2 (commits 324ba8a6 / ebc19720) already made the word-ORDERING spiking; C3 makes the
composer itself spiking.

**VERDICT: GO.** The console runs end-to-end on the spiking one-brain substrate (live GPU transcript, moat 0
leaks), the onebrain answers are ANSWER-IDENTICAL to the rf oracle on every validated case (24/24, moat 0-FA),
and the default `rf` rubric is unchanged (10/10, 0 leaks, MIXED, PASS). Reuse-by-import; two small ADDITIVE
runner edits (NO `sim/` edit).

---

## What was built

`research/runners/first_chat_console.py` (additive):
- `--composer {rf,onebrain}` (default `rf`) + a `composer_kind=` param on `build_brain_on_codes`. `rf` =
  the numpy `RFPhasorComposer` (the test ORACLE + the GPU-less CPU path, byte-unchanged). `onebrain` = the
  persistent spiking `OneBrainComposer` (an on-bridge parser + RF complex-synapse fact store + the resonate
  SCAN/unbind on ONE co-resident `SimulationBridge`), so the console's recall/answer path runs on FIRING NEURONS.
  The grounded 1454 codes pass through (it converses on the codes it learned, on spikes). Needs
  `SIM_BACKEND=cupy` for the real spiking path (a warn prints + the tiny test-oracle path runs on numpy otherwise).
- `_composer_concept_codes(comp)` helper: the auxiliary Tier-2 standalone composers (tense / common-ground /
  entity-instance) read `comp.concepts`; the OneBrainComposer holds those on its inner wrapped `.comp`, so the
  helper resolves them from either substrate (otherwise those layers would silently build on EMPTY codes on the
  onebrain path).

**Honest scope (verified, as the build-plan flagged):** `--composer onebrain` covers the **flat who/what +
chain-of-thought + yes/no + generation**. The **typed verb-frame `--argstructure` path stays on the numpy
`ArgStructureComposer`** (`OneBrainComposer` exposes no `store_fact`/`query_role`/frame `render`) — that is
burndown **C4**. `--composer onebrain` is explicitly IGNORED (with a note) when `--argstructure` is set.

### Two ADDITIVE edits outside the console (NOT `sim/`) — both reuse-by-import

These surfaced as the onebrain path exercised paths the console previously only ran on numpy/rf:

1. **`research/runners/one_brain_composer.py`** — added `_assoc_graph()` (a read-only dialogue-planning graph
   built from `self.kb`, a VERBATIM port of `RFPhasorComposer._assoc_graph`). The console's dialogue planner
   (`rich_answer_composer._controller`) calls `self.composer._assoc_graph()`; the OneBrainComposer lacked it
   (it already mirrors `_relation_assoc` for chain-of-thought — this completes it as an RFPhasorComposer
   API-sibling for the planner). Byte-identical to the rf sibling on the same kb. No resonate/bridge step.

2. **`research/runners/_value_salience_appraisal_derisk.py`** — the `SpikingSpeakAccumulator.decide()` read its
   per-step spikes with `np.asarray(b.cp_firing_states)`, which RAISES on `SIM_BACKEND=cupy` (cupy forbids the
   implicit numpy conversion). Changed to `to_host(...)` (the project's backend-safe helper: a passthrough on
   numpy = byte-identical to the `np.asarray` it replaces, `.get()` on cupy). This is the exact pre-existing
   numpy-only seam the console's own C2 comment flagged; the fix makes the speak-decision accumulator work when
   the whole pipeline runs on cupy for the onebrain composer.

**NO `sim/` edit.** Both edits are in `research/runners/`.

---

## De-risk / gate results

### (1) Live GPU `--composer onebrain` transcript — moat 0 leaks (CLEAN)

`SIM_BACKEND=cupy python -m research.runners.first_chat_console --composer onebrain --demo --n-facts 12
--max-topic-scan 16 --n-topics 8` (RTX 3090; OneBrainComposer = a 166,314-neuron persistent bridge for the
1454-vocab composer slice; pipeline ready in ~485s, then the discuss turns):

```
YOU: hi there!
BRAIN: Hi! Ask me about what I know, or what I think about something.        [phatic, moat=OK]
YOU: what does curry describe?
BRAIN: Good question. The curry describes pine.                              [question C=1, moat=OK]   <- spiking who/what recall
YOU: what is family?
BRAIN: I don't have settled facts about family, but it tends to come up
       alongside scraper, tern, and nettle -- I'd be guessing past that.     [discuss-via-adjacent D, moat=OK]
YOU: tell me more
BRAIN: Noted -- I'll say more about that.                                     [teaching, moat=OK]
YOU: what do you think about beech?
BRAIN: I don't have settled facts about beech, but it tends to come up
       alongside malt, beetroot, and asparagus -- I'd be guessing past that. [opinion novel-flagged N, moat=OK]
YOU: is family like head?
BRAIN: ... (grounded hedge naming real PPMI neighbours)                       [opinion N, moat=OK]
YOU: what does plate touch?
BRAIN: Good question. The plate touches autumn.                              [question C=1, moat=OK]   <- spiking who/what recall
YOU: what is florbglax?
BRAIN: I don't know the word "florbglax" yet -- it's not in what I've learned. [unknown-word, moat=OK]

DEMO moat leaks: 0  (CLEAN)
```

The whole turn — phatic, who/what recall (`certain=1` from the spiking `query_patient`), engage-via-adjacent
(dialogue planning over the `_assoc_graph`), opinion (novel-flagged), and the no-confab moat-abstain on the
unknown word — runs on the persistent spiking bridge, with **0 moat leaks**.

### (2) onebrain == rf-oracle parity on the validated cases — PASS (24/24, moat 0-FA)

`c3_parity.py`: build BOTH composers on the SAME 1454 codes + the SAME 12 console facts (`_make_svo_facts`),
compare `query_patient`/`query_agent`/`ask_yes_no` on the rf-correct cases + the no-confab moat on the absent
cues. The console's onebrain composer uses `enable_spiking_cleanup=False` to match the rf oracle's own
HOST-argmax cleanup (`RFPhasorComposer` default; see below):

```
VALIDATED cases (rf recalls correctly): 8 what + 8 who + 8 yes/no = 24
rf false-accepts=0  onebrain false-accepts=0   (moat: 0 = clean, both)
mismatches on VALIDATED cases (the parity bar): 0
divergences on the lossy TAIL: 0
VERDICT: PASS -- onebrain == rf oracle on EVERY validated case + moat 0-FA both.
```

**Why `enable_spiking_cleanup=False` on the console onebrain path (the honest parity reasoning, fully
localized).** The rf ORACLE the console builds for `--composer rf` defaults to a **host-argmax cleanup-select
over a numpy kb** (`RFPhasorComposer(enable_spiking_cleanup=False, enable_substrate_store=False)` — the
defaults). The `OneBrainComposer`'s memory is ALWAYS the on-bridge complex-synapse STORE + the resonate
SCAN/unbind, so bind / store / unbind / recall already run on FIRING NEURONS regardless — the C3 substrate
win. The only remaining choice is the final winner-PICK. A 3-composer localization (`c3_localize.py`) pinned
the difference exactly:

| OneBrainComposer config | substrate | mismatches vs rf oracle (24 validated) | moat FA |
|---|---|---|---|
| `enable_spiking_cleanup=False` (= console default) | complex-synapse store + **host-argmax** pick | **0** | 0 |
| `enable_spiking_cleanup=True` (the fully-spiking-cleanup default elsewhere) | complex-synapse store + **spiking Izhikevich-WTA** pick | 1 (safe-direction abstain) | 0 |

So the **substrate STORE is byte-faithful to the numpy kb** (host-cleanup → 0 mismatches); the spiking-WTA
cleanup is == numpy argmax at its validated D=2048 scale but at this CROWDED scale (V=1454, D=128, thin code
margins) it costs exactly **1 SAFE-direction abstain** on a single thin-margin fact (`dragonfly/hum/cod` →
`None` instead of `'cod'`; never a fabricated different fact, moat still 0-FA). Matching the oracle's host argmax
on the winner-pick gives exact answer-parity AND keeps the console's recall/bind/unbind/store on the spiking
substrate. The spiking-WTA cleanup remains the documented default on the consolidated_320 / agent onebrain paths
(where D=128/320 codes are well-separated → recall 1.0); closing the 1-fact margin at V=1454 is a wider-D /
shard follow-on, not a C3 blocker.

### (3) Default `rf` rubric — 10/10, 0 leaks, MIXED, PASS (unchanged)

`SIM_BACKEND=numpy python -m research.runners.first_chat_console --rubric`, RE-RUN after all edits:

```
RUBRIC SCORE: 10/10   (moat leaks: 0)
mixed-type across the conversation: certain=True flagged=True phatic=True -> MIXED
VERDICT: PASS
```

The default path is byte-unchanged by the `--composer` add (default rf), the additive `_assoc_graph` method,
and the `to_host` passthrough (numpy-identical).

---

## Falsification check (none tripped)

- onebrain answers diverge from the rf oracle on a validated case → **NO** (24/24 match with the console's
  parity-matched config; the 1 spiking-WTA-cleanup divergence is fully localized, safe-direction, and NOT the
  console default).
- moat breaks → **NO** (0 false-accepts on both composers, every gate run).
- default rubric regresses → **NO** (10/10 unchanged).

## What's onebrain-covered vs still-rf (honest)

- **onebrain (spiking substrate):** flat who/what recall (`query_patient`/`query_agent`), yes/no (`ask_yes_no`),
  chain-of-thought (`chain_of_thought`), generation (`render_fact`), dialogue-planning graph (`_assoc_graph`) —
  all on the persistent co-resident bridge; the on-bridge parser comprehends `hear`.
- **still rf (by design):** the TYPED verb-frame `--argstructure` path (`store_fact`/`query_role`/frame
  `render`) = burndown **C4** (OneBrainComposer has no typed-role API). The numpy `RFPhasorComposer` also stays
  the test-oracle + the GPU-less CPU default (`--composer rf`), per the keep-numpy-oracle verdict.
- **auxiliary Tier-2 routes** (which/tense/common-ground) build standalone composers on the brain's codes via
  `_composer_concept_codes` — consistent on both paths; not exercised by the rubric/demo prompts.

## Run

```bash
# the spiking one-brain console (GPU):
SIM_BACKEND=cupy python -m research.runners.first_chat_console --composer onebrain --demo
SIM_BACKEND=cupy python -m research.runners.first_chat_console --composer onebrain         # REPL
# the numpy oracle / CPU default (byte-unchanged):
SIM_BACKEND=numpy python -m research.runners.first_chat_console --rubric
```

Reuse-by-import; the no-confab moat is preserved throughout; the numpy/rf path is retained as the test-oracle +
CPU-portable path. NO `sim/` edit.
