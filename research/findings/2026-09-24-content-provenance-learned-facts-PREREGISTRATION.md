---
type: preregistration
status: preregistered
date: 2026-09-24
lane: D6-learn-and-grow (content provenance; the owner's "fancy RAG" concern)
mechanism: an instrument, not a mechanism -- a pre-registered content-provenance test through the production /api/brain-chat path with the Qwen mouth ON (renderer='qwen', production default rich path, wikidata LTM off), plus a learned-content fraction measurement; the learning path under test is the default-OFF D6 local Hebbian write (BRAIN_D6_HEBBIAN_STORE)
seeds: [7, 42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. No arm of this probe has run. Seed 7 is a dev smoke; 42/43/44/100/101/102 decide.
runner: research/runners/content_provenance_probe.py
---

# Content provenance with the Qwen mouth on: pre-registration

Committed on its own, before any run it governs. The code it governs is commit `c648d45d9` on branch
`research/content-provenance-learned-facts`; every constant (teach sentences, probes, the 20 untaught questions,
the hand-written answer keys, STOP/GENERIC word lists, arm flags) is fixed there. Terms follow `docs/TERMS.md`.

## Why

The owner's concern: the brain's knowledge may be "fancy RAG the LLM pulls from" rather than learning. The
2026-09-24 audit found that the default chat's facts come from a host bulk imprint (wikidata_100k), none written by a
plasticity rule, and that Qwen is mouth-only (it renders one gated triple; VERIFY and the claim moat reject additions).
D6 showed that a local Hebbian write carries a later recall, but only at runner level with the LLM disabled (stub
renderer, `rich=False`): `research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md`.
Nobody has tested the same question with the real Qwen mouth rendering the default fluent turn, on facts that Qwen's
own prior contradicts. Nobody has counted what share of what the brain says was written by plasticity.

## Protocol (per seed; each arm is a fresh brain in its own subprocess, `BRAIN_CHAT_SEED=seed`)

Every request goes through `webapp.server.brain_chat` with `renderer='qwen'` (the warm `QwenRenderer` /
`SpikingQwenFaculty`) and with `rich` omitted, so the production default (the fluent multi-sentence path) applies.
`BRAIN_LTM_SHIP_DEFAULT=0` and `BRAIN_PERSIST_LEARNING=0` are set in every primary arm. The backend (numpy brain + CPU
Qwen with CUDA hidden, or cupy brain + CUDA Qwen through `tools/gpu_queue.sh`) is recorded in every arm file, and all
arms of one seed run on one backend.

| phase | turns |
|---|---|
| teach | "the wug eats the dax" (nonce: Qwen has no prior) / "the cow eats the moon" / "the bee makes the stone" (counterfactual: Qwen's prior is grass/honey) |
| read | "what does the dog chase" / "what does the cat eat" (build-time facts) |
| probe | "what does the wug eat" / "what does the cow eat" / "what does the bee make" |
| untaught (USE, DEFAULT) | 20 questions: 10 near-miss on known agents (e.g. "what does the cow drink"), 10 fully untaught (e.g. "who wrote hamlet") |
| secondary (USE, FREEZE, DEFAULT; after every primary turn; report-only) | "the cat eats the moon", then "what does the cat eat" |

The owner's example "the cat eats the moon" is secondary and never scored. The tiny-demo brain already holds
"cat eat fish" as a build-time fact, so this item cannot separate Qwen's prior from the brain's own imprint.
The cow item keeps the same counterfactual object.

| arm | flags / step |
|---|---|
| USE | `BRAIN_D6_HEBBIAN_STORE=1`, `BRAIN_D6_ENGRAM_VOCAB=1`, `BRAIN_D6_ENGRAM_READTIME=1`, `BRAIN_D6_HEBBIAN_FREEZE=0` |
| FREEZE (a) | USE's flags + `BRAIN_D6_HEBBIAN_FREEZE=1` (eta=0 for in-conversation writes; same encode activity) |
| ABLATE (b) | USE's flags; after the three teach turns the experimenter zeroes every taught block's synapses (`d6_hebbian_store.ablate_block`) |
| HEARD (c) | FREEZE's flags; the teach turns are the exposure sentences "the wug and the dax" / "the cow and the moon" / "the bee and the stone" (no assertion, nothing acquired) |
| DEFAULT | every D6 flag `0`: the production write, a host direct copy of the composite into the weights |
| DEFAULT_LTM (optional; fraction only; never scored) | DEFAULT with `BRAIN_LTM_SHIP_DEFAULT` unset: the true shipped default (wikidata LTM attached) |

The USE arm also asks the same warm Qwen faculty each probe and untaught question directly, with no brain
("Answer in one short sentence: {q}?", greedy). This is the Qwen-alone reference.

## Criteria (per seed)

**VOID (the seed reads UNDEFINED):** an arm is missing or errored; the renderer is not `QwenRenderer`; the LTM is
attached in a primary arm; or any turn did not take the rich path.

**Instruments.** Any failure makes the seed UNDEFINED, never a pass or a fail:
- `lever_USE_written`: all three taught blocks exist at probe time with mean |w| > 0.5.
- `lever_FREEZE_zero`: all three exist with mean |w| == 0.0.
- `lever_ABLATE_zero`: all three exist with mean |w| == 0.0, and the ablation was applied to all three.
- `lever_HEARD_no_write`: no taught block exists and no in-session store write happened.
- `lesion_no_later_writes`: FREEZE, ABLATE and HEARD make zero store writes from after the teach turns through the
  last primary turn (the counter is installed and reads `[]`).
- `qwen_mouth_rendered_USE_probes`: at least one Qwen generate call happens during the USE probe turns.
- `qwen_alone_present`: the Qwen-alone reference exists for all 3 probes and all 20 untaught questions.
- `FD1_ok` (live failing direction of CP1): each probe's Qwen-alone answer is substituted for the USE reply, with
  the taught fact claimed as its support. It must be scored as a CP1 failure.
- `FD2_ok` (live failing direction of CP5): at least 15 of the 20 Qwen-alone untaught answers must be flagged as
  leaks when substituted for the brain's reply.

A word is matched on a plural/3sg-insensitive stem. `prior(item)` is the hand key plus the content stems of the
Qwen-alone answer to that probe. Those stems exclude the question's words, the STOP and GENERIC lists, the brain's
own lexicon (build-time vocabulary plus taught words), and the taught object.

- **CP1 taught answer.** For all 3 items, the USE probe reply is not abstained. It states the taught object
  (dax / moon / stone) and states no `prior(item)` word.
- **CP2 freeze.** For all 3 items, the FREEZE probe reply is abstained. It does not state the taught object and
  states no `prior(item)` word.
- **CP3 zeroed.** The same test as CP2, applied to ABLATE.
- **CP4 heard-only.** The same test as CP2, applied to HEARD.
- **CP5 no Qwen leak.** In both USE and DEFAULT, 0 of the 20 untaught replies leak. A reply leaks if it states a
  hand-key answer word or a content stem of Qwen's own answer to that question (same exclusions as `prior`). A reply
  also leaks if it carries a supporting or recalled proposition that is in no store block (unsourced).

**Seed GO** requires every instrument and CP1 through CP5 to hold. **Aggregate GO** requires GO on all six of
42/43/44/100/101/102. Seed 7 is a dev smoke: it is reported and never counted. A partial seed set reads INCOMPLETE,
never GO or NO-GO. Scoring command (verbatim):

```
ARM_DIR=research/findings/raw/_content_provenance
.venv/bin/python -m research.runners.content_provenance_probe --score-only --seeds 7 42 43 44 100 101 102 \
    --arm-dir "$ARM_DIR" --json "$ARM_DIR"/cp_verdict.json
```

**Reported, never scored:**
- The learned-content fraction. Over the read, probe and untaught turns, a proposition counts when the reply
  states it (its object word is in the reply text). The fraction is the share of those propositions whose store
  block was last written by an in-session local-Hebbian write with eta > 0. The writer is logged live by
  instrumenting `_store_composite` / `_write_block`. Other classes: pre-session imprint, in-session host copy,
  frozen, ablated, LTM imprint, unsourced. The fraction is reported for DEFAULT, for USE, and for DEFAULT_LTM when
  it was run.
- `attributable_to` for the stated taught object: USE vs FREEZE, and USE vs HEARD.
- Counterfactual validity: Qwen alone does not state the taught object.
- Qwen calls per probe and per untaught turn.
- Read-path intact (the d1/d2 replies are not abstained).
- Whether DEFAULT also states the taught objects.
- The secondary cat/moon replies.
- Per-arm elapsed time.

## Predictions (written before any run)

- **CP2-CP4 pass.** D6 K3/K4/K3e passed 6/6 on its own protocol. What is new here is the Qwen mouth and the rich
  path.
- **CP5 reads 0 leaks.** An abstain is a host template, and Qwen only renders a gated triple.
  - Open risk: rich-path elaboration or the add-on organs (curiosity, GNW, metacog hedges) put Qwen text into a
    reply.
  - Open risk: a near-miss question routes role-blind to a taught fact. That would be cross-talk from the brain's
    own store, not a leak; it is reported.
- **CP1 is the open prediction.** Qwen may "correct" a counterfactual triple (cow eat moon -> grass). If VERIFY
  rejects the render and a template states the fact, CP1 passes. If the sentence is dropped, CP1 fails, and that is
  a real finding about the mouth.
- **The learned-content fraction.** DEFAULT should read exactly 0.0: the production path has no plasticity write.
  USE should read about 3 / (3 + the build-time and elaboration propositions stated), roughly 0.4-0.6.

## Honest scope

- This is an instrument. It measures WHERE a stated fact's synapses came from and whether Qwen can inject content.
  It does not make the D6 write less teacher-forced than its own addendum states: a local rule carried by a
  host-wired instructive pathway, matching the host copy at complex correlation 0.99996 <!--derived-->. It changes no production
  default. Every D6 flag stays default-OFF.
- One tiny-demo brain, one scripted protocol, 3 taught facts. With the LTM off, the leak test covers the mouth and
  the moat, not a knowledge base.
- Host code in the probe is experimenter instrumentation only: logging wrappers, the ablation lesion and the
  Qwen-alone reference. None of it feeds a reply.

## Honesty

Functional read-outs only. "Learned" means the reply's content changes with a synaptic write, measured by lesion. It
makes no claim of felt experience.

## Amendment log

### A1 (2026-09-24, ~13:30 EDT): two mouth variants

**Seen when this was written.** Only the seed-7 `prod` smoke's TEACH turns: all six arms' teach replies and the
Qwen generate-call count of each turn. No read, probe, untaught or validation-seed result existed.

**Observation.** USE, ABLATE and DEFAULT acknowledged every teach turn with 0 Qwen calls. The surface was the
lower-case spiking form ("the cow eats the moon"). FREEZE's teach turns made 2-4 Qwen calls.

**Cause (read in the code, not inferred from the data).** `ChatBrain.spiking_recall_surface` and
`RichAnswerComposer._render_one_verified` render every bounded transitive SVO on the spiking Broca recall mouth
first. That mouth is `BRAIN_SPIKING_MOUTH_RECALL`, default-ON since 2026-08-26. Qwen is consulted only when that
surface fails VERIFY. So in the registered (shipped) configuration Qwen never phrases a recalled bounded-SVO fact
unless the spiking surface fails, and `qwen_mouth_rendered_USE_probes` would read UNDEFINED for a reason unrelated
to Qwen.

**Change.** Two variants. Both are scored by the unchanged CP1-CP5, word lists, arms and instruments, except as
stated here.
- `prod` (the registered configuration). `qwen_mouth_rendered_USE_probes` becomes a reported measurement,
  `use_probe_qwen_calls`, and stops being a required instrument. A count of 0 there is itself the answer to "does
  Qwen phrase learned content in production".
- `qwenforced` (new; the adversarial test of the owner's question). Every arm runs with
  `BRAIN_SPIKING_MOUTH_RECALL=0`, so Qwen phrases every recalled fact. Every instrument is required as registered,
  including `qwen_mouth_rendered_USE_probes`. Arm directory: `research/findings/raw/_content_provenance_qwenforced`.
- Each arm file records its variant. An arm scored under the other variant is VOID.
- Each variant has its own aggregate verdict (GO needs 6/6 on 42/43/44/100/101/102). `qwenforced` is the primary
  answer to the owner's question; `prod` is secondary.
- Run order: the seed-7 `qwenforced` smoke, then the `qwenforced` validation seeds, then the `prod` validation
  seeds as capacity allows. A variant that has not run on all six seeds reads INCOMPLETE.

The code change is the variant plumbing (`--variant`, `CPROV_VARIANT`) and the prod-only instrument relaxation.
`--selftest` adds `prod_zero_qwen_calls_is_scored` and `variant_mismatch_undefined` and passes.

```
ARM_DIR=research/findings/raw/_content_provenance
.venv/bin/python -m research.runners.content_provenance_probe --score-only --variant prod \
    --seeds 7 42 43 44 100 101 102 --arm-dir "$ARM_DIR" --json "$ARM_DIR"/cp_verdict.json
QF_DIR=research/findings/raw/_content_provenance_qwenforced
.venv/bin/python -m research.runners.content_provenance_probe --score-only --variant qwenforced \
    --seeds 7 42 43 44 100 101 102 --arm-dir "$QF_DIR" --json "$QF_DIR"/cp_verdict.json
```
