---
type: finding
status: partial
date: 2026-09-02
mechanism: R1 (source-provenance-honesty audit residual) rung 1 -- an open-ended (BRAIN_OPEN_ENDED) turn in `webapp/server.py::brain_reply` now writes the D5 episodic store (`d5_episodic_production_organ.get_episodic_organ(cache_key, ...).note_topic(...)`) before returning, so a topic discussed in open-ended mode is no longer un-recallable by a later referential turn. This closes ONE of the session-state-write gaps a completeness audit found in the open-ended EARLY RETURN; the audit's larger finding (the branch also BYPASSES ~20 production faculties by returning before the shared pipeline's affect-drives/ToM/DA-mode/common-ground/prospective-memory/worldview/multiref-WM/discourse/causal/pragmatic/comprehension/surprise/non-contradiction/metacog/curiosity blocks and the rich/single-fact composer) is UNCHANGED by this rung -- see "Staged plan" below.
lane: architecture (production-integration -- the open-ended branch as a shared-pipeline FORM OVERRIDE, not a bypass)
verdict: PARTIAL / rung 1 of N. GO on this rung's own scope (byte-identical off, response-surface byte-identical on, the D5 write now genuinely fires via spiking dendritic-completion) -- see the artifact. The REST of the ~20-faculty skip-list (see below) is untouched; do not read this as "R1 closed."
artifacts:
  - research/findings/raw/2026-09-02-open-ended-episodic-writeback-r1-rung1-verify.json
---

# R1 rung 1: an open-ended turn writes the D5 episodic store (the rest of the skip-list is staged, not closed)

## Context: the R1 residual

A completeness audit (owner-directed) found that `brain_reply`'s open-ended branch
(`webapp/server.py`, `if os.environ.get("BRAIN_OPEN_ENDED", ...)`, default-OFF) **returns early**, before:

1. the shared pipeline's session-state-writing faculties (affect-drives #84, W5 ToM, DA-mode #79, common-ground,
   DA-gated encoding, wandered-thought #86, continuous ideation, prospective memory, D5 episodic recall, E2
   internal worldview, D6 multi-referent WM, activity-silent-WM, D3 discourse register, T1-4 causal, scalar-
   implicature pragmatics, D4 comprehension gate, D2 surprise, board #129 surprise->episodic/source-provenance
   cross-edges, B3 non-contradiction, E1 metacognition, D3 curiosity follow-up), and
2. the rich/single-fact composer (where confidence-forthcomingness #94 and source-provenance-honesty #129/#140
   actually apply their framing).

Concretely, in the current file (`webapp/server.py`), the open-ended block sits at lines 4546-4597 (the `if`
guard through its `return _safe_json_response(_oe_resp, "open_ended")`); the shared pipeline's post-open-ended
section runs lines 4599-6283 (see the section markers: `AFFECT DRIVES THE RESPONSE` 4599, `W5 AFFECTIVE ToM`
4621, `DA-MODE DRIVES THE RESPONSE` 4648, `COMMON GROUND` 4671, `DA-GATED ENCODING` 4695, `WANDERED THOUGHT`
4718, `NOVEL IDEA` (continuous ideation) 4744, `PROSPECTIVE MEMORY` 4771, `EPISODIC RECALL` (D5, Hook A) 4832,
`INTERNAL WORLDVIEW` (E2) 4902, `MULTI-REFERENT WORKING MEMORY` (D6) 4962, `activity-silent-wm` 5077,
`DISCOURSE EVENT REGISTER` (D3) 5119, `CAUSAL WHY/WHAT-IF` (T1-4) 5156, `SCALAR-IMPLICATURE` 5228,
`COMPREHENSION MEASUREMENT` (D4) 5263, `EXPECTATION-VIOLATION/SURPRISE` (D2) 5377, board #129 cross-edges
5450, `NON-CONTRADICTION` (B3) 5486, `SELF-MODEL/METACOGNITION` (E1) 5522, `CURIOSITY follow-up` 5575, the
`RICH path` 5638, the `single-fact path` 5973, W5 ToM again (single-fact variant) 6200, `GNW GLOBAL-STOP` 6249).
An open-ended turn touches NONE of this and writes NO session state except one thing that happens to run
*before* the branch: the Gate-B affect block (lines 4460-4530) already updates `_SESSION_MOOD` via
`_update_session_mood` regardless of the open-ended flag, since that block is unconditional and sits earlier in
the function. Every OTHER session-state store (`_SESSION_WORLDVIEW`, `_SESSION_MULTIREF`, `_SESSION_SILENT_WM`,
`_SESSION_DISCOURSE`, `_SESSION_PMEM`'s write side, and the D5 episodic store) was, before this rung, untouched
by an open-ended turn.

## What this rung closes

The D5 episodic store specifically. `webapp/server.py`'s open-ended block, immediately after building
`_oe_resp` and before its existing `return`, now additionally runs:

```python
try:
    import research.runners.d5_episodic_production_organ as _OE_EP
    if _OE_EP.episodic_enabled() and _oe.get("known") and _oe.get("facts") and _episodic_store_ok():
        _oe_ep_topics = getattr(chat, "agents_set", None) or _brain_vocab(chat)
        _OE_EP.get_episodic_organ(cache_key, 42, _oe_ep_topics).note_topic(_oe["facts"][0][0])
except Exception:
    pass
```

This is the identical call, gating (`episodic_enabled()` + `_episodic_store_ok()`), and topic convention
(`facts[0][0]`, the agent of the first supporting fact) the rich path (`server.py:5720`) and the single-fact
path (`server.py:6084`) already use -- substituting the open-ended generator's own retrieved facts (`_oe["facts"]`)
since an open-ended turn never reaches `chat.gate()`. It reads/mutates nothing in `_oe_resp`, so the JSON surface
returned to the client is unchanged.

## Verification (`research/runners/_open_ended_episodic_writeback_r1_verify.py`, `report` phase)

**Scope, disclosed.** The verify uses a minimal mock `chat` (a `SimpleNamespace` carrying only `agents_set` /
`actions_set` / `patients_set` / a trivial abstaining `.gate`), not a full tiny-demo `MultiTurnAgent` build --
that build was independently timed out (>280s) on this host under concurrent load from other sessions'
CPU-bound runs (confirmed via `ps`/`free -m`: <1.2GB free RAM, ~21GB swap in use from unrelated processes at
verification time), and re-paying that cost here would test infrastructure already covered by every other
faculty's own de-risk, not this change. Every OTHER default-on faculty that would otherwise dereference an
attribute the mock lacks is switched off via env flags for this run only (`BRAIN_AFFECT`, `BRAIN_GNW_2ORGAN`,
`BRAIN_GNW_3ORGAN`, `BRAIN_GNW_BUS`, `BRAIN_VALUE_CHOICE`, `BRAIN_SWAP_DRIVES`, `BRAIN_SELF_INITIATE`,
`BRAIN_VISION_IDENTITY`, `BRAIN_BG_SELECT`, `BRAIN_GNW_MULTISTEP` all `=0`); several others (curiosity,
common-ground-drives, DA-drives, DA-encoding) still fired unprompted through the mock and are visible in the
captured response, unaffected. The heavy warm-Qwen loader and `open_ended_chat.answer_turn` are monkeypatched
(no GPU, no real model) for a synthetic but shape-faithful known-topic reply (`topic="dog"`,
`facts=[["dog","chase","cat"]]`, `"dog"` is a real tiny-demo agent). `report` runs the D5 write itself on the
**cupy** backend specifically (`SIM_BACKEND=cupy`) -- `d5_episodic_production_organ.py`'s own docstring records
the write as "~seconds on cupy but ~510s/topic on numpy@2000" (confirmed directly: a first attempt on numpy took
>90s and was still building the 4028-neuron/1.8M-synapse hippocampal bridge when interrupted), which is exactly
why `_episodic_store_ok()` defers the write on numpy in production; `BRAIN_EPISODIC_STORE=1` (the documented
override) is set so the gate does not defer even though this is a fresh, empty bridge, not the deployed one.

**Method.** `report` orchestrates the full before/after comparison via `git stash push/pop` on `webapp/server.py`
only (this file, and the artifact it writes, are untracked and untouched by the stash), running two independent
subprocesses per phase (current code, then the stashed HEAD original) and diffing their stdout.

**Result (GO, `research/findings/raw/2026-09-02-open-ended-episodic-writeback-r1-rung1-verify.json`):**

- **OFF path (`BRAIN_OPEN_ENDED` unset).** `brain_reply`'s JSON response is **byte-identical**, patched vs
  original `server.py`, through the real single-fact pipeline (curiosity/common-ground/DA-drives all fire on the
  mock). This is not merely structural -- both processes were actually run.
- **ON path, response surface.** Byte-identical, patched vs original (`{"answer": "Dogs chase cats around
  here.", "mode": "open_ended", "known": true, "facts": [["dog","chase","cat"]], ...}` in both).
- **ON path, state write -- the load-bearing before/after.** Reading the D5 episodic organ for the SAME
  `cache_key` after the identical turn: **original code** -> `{"in_memory": false, "reason": "no-store-yet"}`
  (the organ was never even built -- the pre-fix dead end). **Patched code** -> `{"in_memory": true, "reason":
  "spiking-dap-completion"}` -- a genuine spiking dendritic action-potential completion read, not a host flag.

## Skip-list still open (staged, not closed by this rung)

Unchanged by this rung, in the order they'd run in the shared pipeline (see line numbers above): affect-drives
#84's own lead/trace, W5 affective ToM, DA-mode #79's engagement suffix, common-ground/audience-design, DA-gated
encoding (a learning write), the wandered-thought lead, continuous ideation, prospective-memory delivery/latch,
E2 internal worldview forward-model (`_SESSION_WORLDVIEW`), D6 multi-referent WM (`_SESSION_MULTIREF`),
activity-silent-WM (`_SESSION_SILENT_WM`), D3 discourse event register (`_SESSION_DISCOURSE`), T1-4 causal
why/what-if, scalar-implicature pragmatics, D4 comprehension gate, D2 surprise, board #129
surprise->episodic/source-provenance cross-edges, B3 non-contradiction, E1 metacognition, D3 curiosity
follow-up, and the rich/single-fact composer's own confidence-forthcomingness #94 + source-provenance-honesty
#129/#140 framing. An open-ended turn still bypasses all of these.

## Design for the remaining rungs (the seam)

The shared pipeline is one long sequential function. Between the Gate-B affect block (~4460) and the final
answer assembly (~6094-6260, where every `_lead`/`_prefix`/`_suffix` computed along the way is joined onto
`resp["answer"]`), each faculty either (a) computes a decoration variable consumed at the end, or (b) owns a
DISJOINT turn class and returns its own `JSONResponse` early (episodic referential recall, prospective-memory
reminder delivery, D6 multiref disjoint resolution, causal disjoint, comprehension repair, GNW-bus explicit
branch, self-schema, GNW-stop). The open-ended generator should NOT be a blanket top-level override of all of
this -- it should be the FORM CHOICE for the case the pipeline reaches "compute the answer surface" (today: run
`chat.gate()` + render via the composer). The minimal seam is: let the disjoint classes keep firing exactly as
they do today (a more specific mechanism already has a definite, grounded answer for those turns -- open-ended
is irrelevant there), let every decoration variable get computed as today, and at the point the pipeline would
call `chat.gate()` + `chat.render()`/the rich composer, branch on `BRAIN_OPEN_ENDED`: run the open-ended
generator instead of the composer for the SURFACE text, then apply the SAME lead/prefix/suffix assembly code
that already exists (so `affect_drives_lead`, `tom_lead`, `da_drives_suffix`, `pmem_prefix`, `mc_prefix`,
`cu_suffix`, etc. decorate an open-ended answer exactly as they decorate a composer answer). Faculty outputs an
open-ended answer should actively CONSUME (not merely not-skip): affect-drives' tone lead already targets manner
the same way `_oe`'s current valence/arousal seeding does (redundant work today, since Gate-B affect feeds
`_oe_val`/`_oe_aro` while affect-drives independently reads the same ladder); common-ground's audience-design
decision (`introduce` vs. assume-known) should shape whether the open-ended generator's prompt states the topic
as new or already-shared; D6 multi-referent WM should resolve an open-ended turn's own anaphora before topic
extraction.

## Staged plan for the remaining rungs

1. **(this rung, DONE)** D5 episodic write-back for the open-ended branch.
2. Move the position of the `BRAIN_OPEN_ENDED` dispatch itself from "early return right after Gate-B affect" to
   "the point the pipeline decides there is no disjoint short-circuit and is about to call `chat.gate()`" --
   purely a relocation, still returning the SAME `_oe_resp` shape, so every decoration variable computed between
   the old and new position (affect-drives, ToM, DA-mode, common-ground, DA-encoding, wander, ideation, pmem,
   worldview, multiref, silent-wm, discourse, causal, pragmatic, comprehension, surprise, #129 xedge,
   non-contradiction, metacog, curiosity) now runs and writes its session state on an open-ended turn -- but the
   `_oe_resp["answer"]` text stays UNDECORATED for now (a second, deliberately separable increment). Verify
   byte-identical off; verify EACH newly-reachable `_SESSION_*` store now updates (repeat this rung's
   before/after pattern per store) and that a disjoint class (e.g. a referential "you mentioned X" turn) now
   correctly short-circuits to ITS OWN answer instead of open-ended generation on a flagged session -- confirm
   this is the desired behavior change (it is, per the design above) before treating it as done.
3. Apply the decoration assembly (leads/prefixes/suffixes) to the open-ended answer surface, matching the
   rich/single-fact path's own assembly code. Verify the SAME lesion-vanish proofs each decorating faculty
   already has (e.g. `BRAIN_AFFECT_DRIVES_LESION=1` vanishes the lead on an open-ended turn too).
4. Route the open-ended generator's surface call through the confidence-forthcomingness #94 content-volume plan
   and the source-provenance-honesty #129/#140 framing, so an open-ended reply's forthcomingness/framing matches
   what a composer-rendered reply about the same fact would get.
5. Re-run this rung's own before/after verify pattern end-to-end once BRAIN_OPEN_ENDED reflects production
   traffic (a real tiny-demo/developed-brain build, not the scoped mock) -- this rung's mock-chat scope is a
   verified LOWER BOUND on correctness (the wiring is right), not a substitute for an integration soak.

## Honest effort estimate

Rung 1 (this pass): ~1 session (mapping + design + implementation + verify), most of it spent characterizing the
skip-list precisely and working around this host's resource contention for the verify rather than on the code
change itself (the change is 15 lines). Rungs 2-3 (relocate the dispatch + wire decorations) are the delicate
core of R1 -- each disjoint-class interaction needs its own before/after check, so plan on 1 session per rung
minimum, likely 2 for rung 2 given ~8 disjoint short-circuits to individually verify. Rung 4 (confidence-
forthcomingness / source-provenance routing) is smaller, comparable to this rung. Rung 5 (real-brain soak) is
mostly compute time (GPU queue), not engineering time. Total: roughly 4-6 sessions to fully close R1, assuming
no wall is hit in the disjoint-class interaction (untested; this is the actual risk this staging is designed to
surface early rather than discover mid-rewrite).
