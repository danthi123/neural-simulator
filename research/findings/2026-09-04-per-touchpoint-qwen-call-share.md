---
type: finding
status: measured
date: 2026-09-04
mechanism: per-touchpoint Qwen-vs-substrate call-share measurement over the REAL `/api/brain-chat` entry
  point (one-brain-wiring de-risk #2, research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md
  SS3), unblocked by the 2026-09-04 generator-trace mislabel fix (commit c08ce8fb3,
  research/findings/2026-09-04-generator-trace-mislabel-fix.md)
lane: language (own-voice mouth / retire the Qwen scaffold) + one-brain (substrate consolidation)
seeds: [42]
seed-waiver: A real-traffic measurement soak through the REAL `/api/brain-chat` entry point
  (`webapp.server.brain_chat`, in-process, same pattern as `_open_ended_bundle_moat_safety_soak.py`), not
  a stochastic training run. `seed=42` is used ONLY to draw a reproducible sample of 4 known topics from
  the live store's agent pool and as the fixed internal generation seed every real turn already uses
  (server.py never passes a seed override) — repeating this under the other 5 standard seeds would be
  byte-for-byte identical for the generation itself, not additional evidence; a larger/different SAMPLE is
  the natural next rung, not a 6x seed repeat.
instrument: research/runners/_per_touchpoint_qwen_share_measure.py — two phases (shipped_default /
  open_ended), each run in its own process (so peak RSS resets between them) against
  `webapp.server.brain_chat`, CPU-forced (`CUDA_VISIBLE_DEVICES=""`, `SIM_BACKEND=numpy`). Surface A
  (shipped_default) has no HTTP-level trace naming which engine rendered a given sentence, so this
  instrument installs a pure read-through wrap on the session's already-built
  `ChatBrain.spiking_recall_surface` (counts hit/miss, calls the original and returns its value completely
  unchanged — verified by code inspection, not by an independent hash/exact-compare against an
  uninstrumented run, so "byte-identical" per docs/TERMS.md is NOT asserted here) plus a call-counter on
  `chat.renderer.render_svo`/
  `render_svo_regen` as a cross-check. Surface B (open_ended) needs no instrumentation: it reads the
  now-correct `open_ended.generator` trace field directly from the JSON response.
runner: research/runners/_per_touchpoint_qwen_share_measure.py
external: NO-EXTERNAL-NEEDED — a real-traffic measurement of this repo's own already-shipped mechanism.
verdict: DESCRIPTIVE, not a GO/NO-GO. On this probe battery Touchpoint A (the Surface-A open-prose recall
  fallback, the roadmap's Stage 2 target) never fired once — not because it has been retired, but because the
  direct-recall GATE upstream of it essentially never reaches real LTM-scale content on natural phrasing (a
  gap PRIOR TO and orthogonal to the roadmap's own touchpoint fork). Surface B's Qwen one-shot fallback (the
  roadmap's Stage 1 target) IS the measurably active call site (9/15 = 60% of dispatched turns) — but 100% of
  it lands on `known=False` traffic where the moat already substitutes a fixed honest-hedge string over the
  visible surface regardless; every `known=True` turn (4/4) was already 100% substrate-covered (fact-clause
  fallback) with zero Qwen involvement. The roadmap's own assumption ("Touchpoint A is the larger live share")
  is NOT confirmed and the priority it implies is inverted: neither roadmap stage is the live bottleneck on
  the production-default surface — the direct-recall gate's own coverage of real LTM entities is. See "The
  verdict" (SS5) for the full reasoning.
artifacts:
  - research/findings/raw/_per_touchpoint_qwen_share_shipped_default.json (Surface A, 16 turns, complete)
  - research/findings/raw/_per_touchpoint_qwen_share_open_ended.json (Surface B, 16 turns, complete)
---

# Per-touchpoint Qwen-vs-substrate call share — the roadmap's Touchpoint-A assumption does not hold on real traffic

**One-brain-wiring de-risk #2** (`research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md` SS3),
unblocked by the 2026-09-04 generator-trace mislabel fix. The roadmap named its own next action plainly: "The
design ASSERTS Touchpoint A is 'the LARGER live share of actual Qwen render calls' — verify it against real
traffic, so Stage 1 vs Stage 2 are ordered by measured share, not assumption." This is that verification, run
through the real `/api/brain-chat` entry point on both reply surfaces. **The roadmap's assumption does not
survive contact with real traffic**, but not in the direction anyone expected: Touchpoint A did not merely
lose the race, it never reached the starting line on this probe battery — the direct-recall GATE that would
feed it a sentence to render essentially never matches real LTM-scale content for a natural question, on
either "Tell me about X" or "What is X" phrasing, for any of 4 real sampled store entities across all 4
phrasings tried. That gate-coverage gap, not the render-engine choice, is the dominant finding here.

## 1. The touchpoints, read straight from shipped code

A live `/api/brain-chat` turn has two mutually-exclusive reply surfaces, selected by `BRAIN_OPEN_ENDED`
(default-OFF, `webapp/server.py:4683`) — `webapp/server.py::brain_chat` (request parsing, `:4184`) calls
`brain_reply` (`:4283`, the shared faculty pipeline both the HTTP route and the TUI/OpenAI-shim run).

**Surface A — the strict/rich recall turn (`BRAIN_OPEN_ENDED` unset — the production default).**
`brain_reply` resolves `use_rich = req.rich if req.rich is not None else _brain_rich_default()` (`:5950`,
default True) and calls `RichAnswerComposer.answer(msg)` (`research/runners/rich_answer_composer.py:841`).
`answer()` -> `gather()` (`:493`) runs the no-confab MOAT (`_direct_fact`, `:327`, which calls
`self.chat.gate(question)`) then `render_paragraph(facts)` (`:672`), which — with the default-OFF
`BRAIN_RICH_BATCH_RENDER` — loops `_render_one_verified` (`:797`) once per gathered fact:

```python
spk = self.chat.spiking_recall_surface(a, v, p)      # the bounded-transitive spiking Broca mouth
if spk is not None:
    return spk, True                                   # TOUCHPOINT: spiking-mouth-recall
surface, asserted = self.chat.renderer.render_svo(a, v, p)   # TOUCHPOINT A: Qwen (GPU) / host template (CPU)
if self._verify_rendered(surface, asserted, svo, gated):
    return surface, True
if hasattr(self.chat.renderer, "render_svo_regen"):
    surface2, asserted2 = self.chat.renderer.render_svo_regen(a, v, p)
    ...
```

`spiking_recall_surface` (`research/runners/brain_chat_tui.py:1558`) is gated on
`BRAIN_SPIKING_MOUTH_RECALL` (default-ON since the 2026-08-26 flip, `research/runners/
spiking_mouth_recall_prod.py:75`) and `frame_supported(a, v, p)` (a bounded transitive-SVO frame check); a
miss (open/multi-word/copula/irregular-verb prose the frame can't cover, the flag off, or a verify-miss)
falls straight through to `chat.renderer.render_svo` — **Touchpoint A**, whatever renderer the chat session
picked (Qwen2.5-0.5B on a CUDA host, a host template on a GPU-free host — `_default_brain_renderer()`,
`webapp/server.py:3517`).

**Surface B — the `BRAIN_OPEN_ENDED` free-generation channel.** `webapp/open_ended_chat.py::answer_turn`
(`:567`) dispatches, in order (lines 616-734): (1) the WKV mouth (`wkv_mouth_enabled()`, default-ON) — an
in-vocab prompt tries `_WKV.generate()`, which itself may render via `render_fact_sentence`
(`sentence_fact_used` in the now-fixed `trace` dict -> `generator="spiking_clause"`) or genuine free-gen
(`generator="wkv_mouth"`); (2) the fact-clause fallback (`fact_clause_fallback_enabled()`, default-ON) — a
known topic the WKV mouth did not already handle also tries `render_fact_sentence` directly
(`generator="spiking_clause"`); (3) the gen-time consensus veto (`gen_time_honesty_enabled()`, **default-OFF**)
— skipped entirely in this measurement; (4) the one-shot `OpenEndedGenerator.generate` — **Qwen**
(`generator="qwen"`), reached for out-of-scope prompts, unknown topics, or any exception from (1)/(2). This
`generator` field is what the 2026-09-04 mislabel fix (commit c08ce8fb3) made trustworthy — before that fix
it followed which of `answer_turn`'s two internal try-blocks reached the reply, not which of `generate()`'s
own branches actually wrote it, corrupting exactly this provenance.

## 2. Methodology

**Probe set** (`research/runners/_per_touchpoint_qwen_share_measure.py::build_probes`), extending —
not just reusing — `_open_ended_bundle_moat_safety_soak.py`'s known/unknown/dangerous battery (which is
entirely "Tell me about X" phrasing) with the classes this task asked for: `known_factual` (4 real store
agents sampled `>=2 facts, non-taxonomic`, 4 different natural phrasings), `known_multi_sentence`
("Tell me everything you know about X and explain why it matters"), `known_followup` ("tell me more"),
`unknown` (`_UNKNOWN_ENTITIES[:2]`, the project's canonical made-up-word list), `dangerous`
(`_QWEN_KNOWN_STORE_UNKNOWN[:2]`, the canonical Qwen-known/brain-unknown list), `open_ended_opinion` (3
genuinely topic-less prompts), `greeting_social` (3 conversational turns). 16 turns per phase, the IDENTICAL
list run against both surfaces in one continuous session per phase (reset once, at the start of that
phase).

**Two disclosed, budget-driven deviations** from a literal from-scratch shipped default (both structural —
they change which ENGINE sits behind a touchpoint, never the touchpoint-dispatch logic itself):
1. `BRAIN_LTM_BUNDLE` forced to the SAME candidate-root resolution `_default_ltm_bundle_dir()` uses, pointed
   at `wikidata_core_15k` (17 MB) rather than letting it resolve to the 2026-09-02-flipped true shipped
   default `wikidata_100k` (88 MB, ~5x). A throwaway de-risk probe this session measured a
   `renderer=qwen` + `wikidata_core_15k` + tiny-demo build at 3.95 GB RSS after 2 throwaway turns — ~50 MB
   under this task's own RSS<4GB budget with ZERO real probes run — on a machine already showing 6.3 GB
   free / 13 GB "available" system-wide (`free -h`, this session; the GPU busy with an unrelated scale probe
   per this task's own brief).
2. Surface A's `renderer` left at auto (`_default_brain_renderer()`), which resolves to the GPU-free
   `template-stub` renderer under `SIM_BACKEND=numpy` — confirmed live: `renderer: "template-stub (GPU-free)"`
   in every Surface-A row — rather than the literal Qwen2.5-0.5B a CUDA host would pick, avoiding a SECOND
   ~2 GB model load stacked on top of Surface B's (which genuinely needs Qwen warm regardless, so that cost
   is paid there). This is not a fabricated condition — it is exactly what `_default_brain_renderer()`
   already does on any GPU-absent production host today (its own docstring: "GPU-free hosts get the
   multi-sentence TEMPLATE stub"). It means Surface A's own TOUCHPOINT-DISPATCH share (spiking-mouth-recall
   hit vs renderer-fallback) is measured through the exact same gate/verify logic production uses; only the
   fallback ENGINE's identity (Qwen vs template) is a CPU-host substitution. The two phases ran as SEPARATE
   PROCESSES specifically so Surface B's Qwen load never stacks on top of Surface A's own baseline — Surface
   A's measured peak RSS was 1067.3 MB, nowhere near the budget in the end (see the artifact).

**Instrumentation validity.** The Surface-A wrap is a pure read-through (calls the original
`spiking_recall_surface`, returns its result completely unchanged, only increments a counter) — confirmed by
the fact that every recorded `answer` text is a live, unmodified reply (spot-checked against the raw JSON,
see below); the wrap cannot itself have altered which branch fired.

## 3. Results — Surface A (shipped-default, `BRAIN_OPEN_ENDED` unset)

Full artifact: `research/findings/raw/_per_touchpoint_qwen_share_shipped_default.json` (`complete: true`,
`aborted_for_memory: false`, peak RSS 1067.3 MB, wall 448.2 s).

| class | n | abstained | reached the render fork | sentences: spiking-mouth-recall | sentences: Touchpoint A (renderer) |
|---|---|---|---|---|---|
| known_factual | 4 | 4/4 | 0/4 | 0 | 0 |
| known_multi_sentence | 1 | 1/1 | 0/1 | 0 | 0 |
| known_followup | 1 | 1/1 | 0/1 | 0 | 0 |
| unknown | 2 | 0/2 (see below) | 2/2 | 2 | 0 |
| dangerous | 2 | 0/2 (see below) | 2/2 | 2 | 0 |
| open_ended_opinion | 3 | 2/3 | 1/3 | 3 | 0 |
| greeting_social | 3 | 1/3 | 1/3 (+1 self-initiated, off-fork) | 1 | 0 |
| **total** | **16** | **9/16** | **6/16 turns, 8 sentences** | **8/8 (100%)** | **0/8 (0%)** |

**The headline number: 0 Touchpoint-A (Qwen/renderer-fallback) calls across every turn that reached the
render fork at all.** `render_calls` (the cross-check counter on `chat.renderer.render_svo`/
`render_svo_regen`) reads 0 on every single row in the artifact. Every rendered sentence came from
`spiking_recall_surface`.

**Why — this is NOT "the spiking mouth has already won," it's "the fork was barely ever reached":**

- **The 4 `known_factual` probes (real store agents, 4 different natural phrasings — "Tell me about X.",
  "What do you know about X?", "Can you tell me about X?", "What is X?" — each on a DIFFERENT sampled
  topic) all ABSTAINED**, every one via a curiosity-flavored non-answer (`"Sure — I don't know about that.
  My curiosity is piqued — I haven't learned about angora yet: what can you tell me about angora? — worth
  going further here."` — verbatim, row 0), not the plain "I don't know about that." abstain. Root cause,
  confirmed by reading the code (not just observed behaviour): `ChatBrain.stored_facts`/`agents_set`
  (`research/runners/brain_chat_tui.py:657-659`) are snapshotted from the tiny-demo composer's OWN small
  built-in KB (`comp.kb`) **at ChatBrain construction time**, which happens BEFORE `_build_chat_brain`
  (`webapp/server.py:3849-3858`) wraps the composer in a `TieredFactStore` pointed at the LTM shard — so
  `QuestionRouter.match_fact` (the host fallback `gate()` uses for anything that isn't an exact "what does
  AGENT ACTION?" parse) searches a corpus that **never contains the LTM shard's entities at all**, and the
  ONLY path that does reach the tiered `query_patient` (`_substrate_recall`,
  `research/runners/brain_chat_tui.py:1240`) requires the NEURAL BridgeParser to extract a literal
  `(agent, action)` pair from the question — which "tell me about X" / "what is X" (no verb at all) cannot
  produce. Separately, the curiosity lead-in itself truncates the underscored multi-word slug to its first
  token ("angora_turkey" -> "angora"), so even the ABSTAIN text misreads the topic. **This is a real,
  previously-undocumented gap: on this shipped chat config, a natural question about a real LTM-scale entity
  structurally cannot reach EITHER touchpoint — it abstains upstream of the fork, every time, regardless of
  phrasing.**
- **The 2 `unknown` + 2 `dangerous` probes did NOT abstain, and DID reach the fork (spk_hit=1 each) — but
  this is a SEPARATE, genuine defect, not evidence for either touchpoint.** All four probes share the exact
  phrasing "Tell me about {single fabricated or real bare word}." (`zorplaxian`/`flibberwock`/`paris`/
  `python`), and all four produced garbled, template-shaped prose: `"the tell abouts the zorplaxian — worth
  going further here."`, and on the NEXT turn `"...my mismatch monitor fired: I'd learned that tell about
  zorplaxian. Updated — I'd stored that tell about flibberwock...the tell abouts the flibberwock..."`
  (verbatim, rows 6-9). Reading this: `ChatBrain.gate()`'s in-loop `_maybe_acquire` (IN-LOOP LEARNING,
  `:680-682`) mis-parses the imperative "Tell me about X" as a fresh SVO ASSERTION ("tell"/"about"/X)
  whenever its object is a single bare token (unlike the multi-word underscored slugs above, which do not
  match whatever shape the acquisition parser needs) — TEACHES it into the live substrate, and the surprise/
  reconsolidation "mismatch monitor" then fires on the NEXT such turn because the taught `(tell, about, ·)`
  binding's patient just changed. The freshly-taught fact is a trivial bounded-transitive triple, so
  `spiking_recall_surface` recalls it immediately — a real hit, but of a fabricated fact the harness's OWN
  probe phrasing accidentally taught, not a genuine "the brain already knew this" recall. **Disclosed
  explicitly so these 4 sentences are not miscounted as evidence that spiking-mouth-recall is winning a
  genuine content-recall race** — they are a distinct, real robustness defect in how Surface A handles a
  casual "tell me about X" over an out-of-store bare word, worth its own follow-up, out of this
  measurement's scope.
- **The clean signal:** `open_ended_opinion` probe "Why do you think memory matters?" and `greeting_social`
  probe "how are you?" DID genuinely match content — the tiny-demo's own small built-in fact set (`("brain",
  "use", "spikes")`, `("brain", "store", "memory")`, `("brain", "learn", "words")` — the SAME class of
  simple, single-word-vocabulary, bounded-transitive facts the smoke-test script's own `_SMOKE_FACTS` uses).
  Answer: `"Setting the held thread aside — On memory, then — Sure — the brain stores the memory the brain
  uses the spikes the brain learns the words — worth going further here."` (row 12, 3 sentences) and `"Sure
  — the brain uses the spikes — worth going further here."` (row 14, 1 sentence) — both entirely via
  `spiking_recall_surface`. This is genuine, uncorrupted signal: **every sentence that reached the fork via
  a real (non-mis-taught) recall was covered by the bounded-transitive spiking Broca mouth; Touchpoint A
  never fired.** But the N here is small (4 clean sentences) and the content is drawn entirely from the
  demo's OWN tiny hand-built KB — exactly the shape `spiking_recall_surface`'s bounded frame was purpose-
  built to cover 100% of. It says nothing about what happens when a real LTM-scale, open-prose fact (a
  Wikidata relation like `located_in_time_zone`/`participant_of`) DOES reach the fork, because on this
  measurement none of them ever did.

## 4. Results — Surface B (`BRAIN_OPEN_ENDED=1`)

Full artifact: `research/findings/raw/_per_touchpoint_qwen_share_open_ended.json` (`complete: true`,
`aborted_for_memory: false`, peak RSS 3973.7 MB — under the 4300 MB hard limit throughout; the process's own
`ru_maxrss` is a HIGH-WATER MARK, so this is a peak reached once, not a sustained level — live `ps` RSS was
measured at ~2.5-3.1 GB for most of the run, see SS2's methodology note reproduced in the honest residuals
below; wall 760.3 s).

The SAME 16 turns, IDENTICAL prompts, run against the `BRAIN_OPEN_ENDED` free-generation channel:

| class | n | known | generator: spiking_clause | generator: wkv_mouth | generator: qwen | off-fork |
|---|---|---|---|---|---|---|
| known_factual | 4 | 4/4 | 4 | 0 | 0 | 0 |
| known_multi_sentence | 1 | 0/1 | 0 | 1 | 0 | 0 |
| known_followup | 1 | 0/1 | 0 | 0 | 1 | 0 |
| unknown | 2 | 0/2 | 0 | 0 | 2 | 0 |
| dangerous | 2 | 0/2 | 0 | 0 | 2 | 0 |
| open_ended_opinion | 3 | 0/3 | 0 | 1 | 2 | 0 |
| greeting_social | 3 | 0/3 | 0 | 0 | 2 | 1 |
| **total** | **16** | **4/16** | **4** | **2** | **9** | **1** |

(The 1 "off-fork" row — `"what's on your mind?"` — is the SELF-INITIATED UTTERANCE branch, which fires
BEFORE the `BRAIN_OPEN_ENDED` dispatch on an idle-turn lead-in regardless of the flag; verified identical
verbatim text to Surface A's own row 15, confirming it's a third, distinct code path, not part of either
surface's FORM-generator fork.)

**Of the 15 turns that actually reached the open-ended FORM-generator fork: Qwen wrote 9 (60.0%); the
substrate (spiking_clause + wkv_mouth combined) wrote 6 (40.0%: spiking_clause 4 = 26.7%, wkv_mouth 2 =
13.3%).** But that 60/40 split, read alone, hides the one clean, decisive pattern in this data:

**Every `known=True` turn (4/4, 100%) was written by the substrate (`spiking_clause`) with ZERO Qwen
involvement — the exact same 4 real store entities Surface A's gate could never reach at all.** Read the
actual answers: `"the Angora Turkey is located in the time zone of the Kaliningrad Time"`,
`"the Imperial Roman follows the Res Publica Romana"`, `"the L Quipe de France is associated with the sport
of the Association Football Club"` — genuine, grounded, capitalized, coherent English sentences built
directly from the LTM shard's real Wikidata relations, via the fact-clause fallback
(`fact_clause_fallback_enabled()`, default-ON, `webapp/open_ended_chat.py:382-453`) — the RELATION_LEXICON
covering `located_in_time_zone`/`follows`/`sport`/`country` etc. Qwen's 9 calls are ENTIRELY concentrated on
the 11 `known=False` turns (open-ended opinion, greetings, genuinely-unknown/dangerous entities, and the two
phrasing-miss cases) — i.e., exactly the traffic class where there is no stored fact to speak at all.

**A further qualification that matters for interpreting "60% Qwen": on every `known=False` turn, the FINAL
answer is the SAME fixed honest-hedge template with only the topic substituted** — `"I'm not sure about
{topic} — I don't have anything about it in what I've actually learned, so I'd only be guessing."` (verbatim
across `zorplaxian`/`flibberwock`/`paris`/`python`/`music`/`more`, rows 5-10, 12-14). This is the moat's
`post_filter` honest-abstain path (`_base_post_filter`'s hedge/abstain branch, `webapp/open_ended_chat.py:
192-250`), not organic Qwen prose — this measurement did not capture Qwen's own pre-filter `raw` text
separately (an honest residual, SS6), so it cannot say how much the raw generation itself varied before the
template substitution, only that the SURFACE the user actually sees converges to one fixed safe string
regardless of topic. So "Qwen fires on 60% of dispatched turns" measures how often `OpenEndedGenerator.
generate()` gets CALLED, which is real (the compute/latency cost of loading and running a 0.5B transformer is
paid every time) — but it overstates how much of the visible SURFACE PROSE is actually Qwen-authored content
a retirement would need to replace, since the moat already substitutes a fixed, brain-independent string for
most of it.

## 5. The verdict — which touchpoint's retirement unblocks the most

**The roadmap's own assumption — "Touchpoint A is the LARGER live share of actual Qwen render calls" — is
NOT confirmed by this measurement, and the corrected picture inverts the priority the roadmap assumed.**

1. **On the ACTUAL production-default surface (A), Touchpoint A never fired once (0/8 sentences across every
   turn that reached the render fork).** This is not because Touchpoint A has been retired — the roadmap's own
   ledger still reads `neural-render: BLOCKED, host_scaffold_in_default: "off-bridge Qwen2.5-0.5B transformer
   (or host templates)"` — it is because the DIRECT-RECALL GATE upstream of the fork essentially never matches
   real LTM-scale content for a natural question, regardless of phrasing (SS3). **Retiring Touchpoint A's
   render step today would have ZERO measured effect on real Surface-A traffic**, because that traffic never
   reaches it. The roadmap's Stage 2 ("retire Touchpoint A") targets a code path that is not the live
   bottleneck; the live bottleneck is one step earlier.
2. **On Surface B (not yet the production default), Qwen IS the measurably active call site (60% of
   dispatched turns) — but concentrated entirely on content the store doesn't have (`known=False`), where the
   moat already substitutes a fixed honest-hedge string over most of what Qwen would say.** The genuinely
   informative content path (`known=True`, real facts) is ALREADY 100% substrate-covered by the fact-clause
   fallback, with no Qwen involvement measured at all in this sample. Retiring the Surface-B one-shot fallback
   (the roadmap's Stage 1) would remove a real, measured, currently-executing model call — but on unknown
   topics the VISIBLE reply is already brain-independent (the fixed hedge), so the retirement's practical yield
   is removing wasted compute/latency, not closing a live fabrication or scaffold-dependency gap on the content
   that matters.
3. **The corrected ordering: fix the upstream gate-coverage gap BEFORE either roadmap stage.** The single
   highest-leverage next step this measurement surfaces is NOT named as a stage in the roadmap at all —
   Surface A's direct-recall gate needs the SAME kind of topic-extraction Surface B's `extract_topic()`
   already has (lead-in stripping down to the bare entity, `webapp/open_ended_chat.py:456-480`) reaching the
   TieredFactStore-wrapped LTM shard, not just the tiny-demo's own construction-time `stored_facts` snapshot
   (SS3's root-cause finding). Until that lands, Stage 2 (teach the mouth to word open-prose recall) has
   no live traffic to exercise it on the production-default surface, and Stage 1's yield is mostly a
   compute/latency saving rather than a scaffold-dependency closure, because the moat already keeps Qwen's
   words off the visible surface on the classes it's called for.
4. **A genuine, disclosed uncertainty this measurement cannot resolve on its own:** whether a LARGER,
   differently-phrased probe battery (SS6) would find natural phrasings that DO reach Surface A's gate for
   real LTM content (e.g. if some fraction of real user traffic happens to use the exact "what does AGENT
   ACTION?" form) and, if so, whether Touchpoint A fires there. This measurement's own 4-topic/4-phrasing
   sample never found such a case, but a sample of 4 cannot rule out a rarer path.

## 6. Honest residuals / what this measurement does not do

- **N=16 per phase, one probe list, one seed-drawn topic sample.** This is a real-traffic MEASUREMENT with a
  disclosed, curated probe battery, not an exhaustive census of production traffic — see the seed-waiver.
  A larger, independently-sampled battery (more known topics, more phrasing variety) is the natural next
  rung, not a 6-seed repeat of this one (repeating under a different seed number would not change anything
  the deterministic pipeline computes — see `_open_ended_bundle_moat_safety_soak.py`'s own "seed honesty"
  precedent, reused here).
- **This does NOT fix the two defects it surfaced** (the direct-gate's inability to reach LTM-scale content
  on natural phrasing; the in-loop-acquisition mis-parse of "Tell me about X" over a bare out-of-store word)
  — both are named here as disclosed, out-of-scope findings for a follow-up, not silently absorbed into the
  touchpoint-share numbers.
- **`BRAIN_LTM_BUNDLE=wikidata_core_15k` substitution** (RSS budget, SS2) means the KNOWN-class sample is
  drawn from the smaller of the two shipped bundles; the touchpoint-DISPATCH mechanism measured is identical
  regardless of bundle size (same code path, same gate, same `spiking_recall_surface`/`render_svo` fork),
  but the specific entities sampled would differ under `wikidata_100k`.
- **Surface A's renderer substitution** (template-stub, not literal Qwen — SS2) means the 0 Touchpoint-A
  calls observed cannot distinguish "Qwen would also never have been reached" from "a CUDA host's literal
  Qwen calls would look identical in count to the template's here" — the GATE never reaching the fork is
  the dominant fact regardless of which engine sits behind it.
- **The gen-time consensus veto** (`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`, default-OFF) was left off, per this
  task's own instruction to test the shipped-default vs `BRAIN_OPEN_ENDED=1` configs — it never fires in
  either measured config.
- **Surface B's raw (pre-`post_filter`) Qwen text was not recorded separately from the final `answer`.** The
  harness stores `answer` (the filtered surface), not `open_ended.raw`; SS4's observation that every
  `known=False` reply converges to one fixed hedge string is about the FINAL surface, and this measurement
  cannot say whether Qwen's own `raw` generation varied more underneath before the moat's hedge/abstain
  branch overwrote it. Recording `raw` alongside `answer` is a one-line harness change for a follow-up, not
  done here.
- **RSS reported throughout is `ru_maxrss`, a high-water mark, not a live/current reading** — it can only
  increase within a process, so a transient allocation spike (e.g. mid-import or mid-model-load) that is
  later freed still reports at its peak for the rest of the run. Surface B's own process showed `ps`-reported
  CURRENT RSS around 2.5-3.1 GB for most of its run despite the logged 3973.7 MB ceiling reached once early
  (during warm-up) — both numbers are genuine and are not in conflict; disclosed so the 3973.7 MB figure is
  not read as "the process ran at 3973.7 MB the whole time."
- **A genuine bug was found and fixed in this task's own harness mid-session**, not in any shipped `sim/`/
  `webapp/` file: `_checkpoint()`'s `"complete"` field originally read `not aborted`, which is `True` on
  EVERY per-probe checkpoint (not just the final one) since `aborted` only ever becomes `True` on the
  hard-RSS-abort path — so a partial, still-running artifact falsely claimed completion. Caught directly
  (a Monitor watch for `'"complete": true'` exited after the FIRST checkpoint, not the 16th) and fixed by
  adding a `finished` parameter set only at the true end of `main()`; the CURRENTLY-RUNNING processes that
  produced this finding's two artifacts predate the fix (their own `"complete": true` fields are only
  reliable because they are also the LAST write before process exit, verified against process-liveness
  separately, not trusted from the field alone) — the committed runner file has the fix.

## Provenance

Shipped code read this session (2026-09-04): `webapp/server.py::brain_chat`/`brain_reply` (`:4184-6060`, the
`BRAIN_OPEN_ENDED` dispatch `:4669-4754`, `_brain_rich_default`/`_default_brain_renderer`/
`_resolve_ltm_bundle`/`_default_ltm_bundle_dir`/`_get_rich_composer`/`_build_chat_brain` `:3517-3928`,
`:3730-3897`), `webapp/open_ended_chat.py` (the full `answer_turn` dispatch `:566-753` + the flag family
`:253-453`, `post_filter`/`_base_post_filter` hedge-abstain path `:192-250`, `extract_topic`/`_LEADINS`
`:456-480`), `research/runners/rich_answer_composer.py` (`render_paragraph`/`_render_one_verified`/
`_render_paragraph_batched`/`_batch_render_enabled` `:79-836`), `research/runners/brain_chat_tui.py`
(`ChatBrain.__init__` stored_facts/agents_set `:657-659`, `gate`/`_gate_router_combine` `:666-724`,
`spiking_recall_surface`/`_apply_mouth_mood_tone` `:1546-1603`, `_substrate_recall` `:1240-1290`),
`research/runners/spiking_mouth_recall_prod.py` (`recall_mouth_enabled` default-ON `:75-88`),
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (`neural-render` row `:170-185`,
`host_scaffold_in_default: "off-bridge Qwen2.5-0.5B transformer (or host templates); enable_neural_render=
False on all builders"`). Precedent instrument pattern reused (not re-derived):
`research/runners/_open_ended_bundle_moat_safety_soak.py`. Builds on: the roadmap
(`research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md`), the mislabel fix
(`research/findings/2026-09-04-generator-trace-mislabel-fix.md`, commit c08ce8fb3). De-risk memory probe
(3.95 GB RSS, `wikidata_core_15k`+qwen renderer+tiny-demo, 2 throwaway turns) run this session, not
separately artifacted (a disposable calibration run — its number is quoted directly above from the session's
own log). `data/corpus/tinystories.txt` (gitignored) was symlinked from the primary checkout into this
worktree to satisfy `SpikingQwenFaculty`'s held-out-text dependency — a filesystem convenience for a
git-untracked data file, not a code or git-tracked change.
