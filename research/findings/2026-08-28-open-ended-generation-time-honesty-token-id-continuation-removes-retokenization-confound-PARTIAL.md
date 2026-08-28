---
type: finding
status: contributing
date: 2026-08-28
mechanism: open-ended-generation-time-consensus-veto-tokenid-continuation
lane: E-language-open-ended-honesty
seeds: [42]
seed-waiver: The evidence landed THIS session is (a) a deterministic wiring-sanity battery over a fixed
  fake-tokenizer/fake-model script (no seed dependence at all -- the same reasoning the parent
  2026-08-27-open-ended-generation-time-honesty-PARTIAL.md finding's controlled unit battery already used),
  and (b) an unmodified-logic re-confirmation of that parent finding's own CPU-only pieces. The DECISIVE
  live-mouth multi-seed GPU measurement (does token-id continuation raise the divergence rate over the
  2026-08-27 text-continuation baseline) is explicitly NOT run this session -- it is STAGED on
  research/queue/gpu.queue for the controller, and this finding does not claim its result.
instrument: research/runners/_open_ended_gen_time_consensus_veto_derisk.py (upgraded in place -- see below)
  plus a new research/runners/_open_ended_gen_time_tokenid_continuation_wiring_verify.py (fake
  tokenizer/model, no GPU/Qwen/organs).
runner: research/runners/_open_ended_gen_time_consensus_veto_derisk.py
external: SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching (arXiv:2504.00970) <!--derived-->
  -- an external paper id, not a measurement from this session's own runs -- confirms
  sentence-boundary-triggered token/cache continuation (never decode-then-re-encode already-generated
  text) is an established technique for exactly the retokenization-avoidance property this session's
  token-id continuation implements; this is an orchestration fix inside an already-GO, already-merged
  mechanism (2026-08-27), not a new biology/mechanism proposal.
artifacts:
  - research/findings/raw/_open_ended_gen_time_tokenid_continuation_wiring_verify.json
---
# Token-ID continuation removes the retokenization confound named by the 2026-08-27 PARTIAL finding's own NEXT (mechanism GO on wiring sanity; decisive live-mouth GPU measurement STAGED, not yet run)

Artifact: research/findings/raw/_open_ended_gen_time_tokenid_continuation_wiring_verify.json (20/20 checks
GO on a deterministic fake-tokenizer/fake-model battery exercising the REAL new orchestration code against
the REAL, unmodified `clause_filter_sentence`).

**One line.** `2026-08-27-open-ended-generation-time-honesty-PARTIAL.md` named its own NEXT rung: "token-ID
continuation (drop the retokenization confound) to raise the live-mouth divergence rate across more
topics/seeds." This finding BUILDS that rung inside the same file
(`research/runners/_open_ended_gen_time_consensus_veto_derisk.py`, still behind the same default-OFF
`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` flag, still additive), proves the new orchestration correct with a
bounded CPU-only wiring-sanity battery, and STAGES the decisive GPU multi-seed comparison rather than
claiming its result — the honesty-of-scope discipline the parent finding itself modeled.

## State of the lane (checked before building, per this session's own instructions)

`.venv-rag/bin/python tools/rag/rag_search.py` surfaced the 2026-08-27 PARTIAL finding as the top hit for
"open-ended generation-time honesty" -- reading it in full showed the GENERATION-TIME coupling (the LTM-
exempt organ-B/C consensus veto shaping the mouth's OWN sentence-by-sentence decode, not a post-hoc filter)
was ALREADY BUILT and merged to `main` (`5eec26339`, 2026-08-27 19:06), with the PRIMARY controlled-battery
evidence already load-bearing (vary/lesion 3/3 GO, deterministic) -- so the coupling itself is not "genuinely
pending." What the finding disclosed as PARTIAL, by name, was the SECONDARY live-mouth confirmation: only
1/3 topics showed the ON/LESIONED decodes actually diverge, because the v1 "text continuation" technique
decoded the growing accepted reply to a string and RE-TOKENIZED `prompt + accepted_text` from scratch on
every sentence step -- a text-roundtrip confound, honestly disclosed as NOT a defect in the consensus
mechanism itself. That confound, and the finding's own named fix for it, is the concrete, well-scoped
"genuinely pending" rung this session advances.

## The mechanism

`_open_ended_gen_time_consensus_veto_derisk.py`'s `generate_with_generation_time_veto` gained a
`continuation=` parameter (`"token_id"`, the new default; `"text"`, the original 2026-08-27 path, kept
verbatim for direct A/B). The new `token_id` path (`_generate_tokenid_continuation` /
`_continue_chunk_ids` / `_find_sentence_boundary_ids`) carries the growing reply as TOKEN IDS between
sentence-generation steps instead of a decoded string: a KEPT sentence (nothing for the consensus to
suppress -- the common case) has its OWN model-generated ids appended to the context directly
(`torch.cat`, zero retokenization); only a REPAIRED sentence (an actual text edit -- the store-wrong span
was removed) re-encodes, and only that one repaired span, not the whole accumulated reply. `clause_filter_
sentence`, `sentence_contradicts`, `consensus_facts_for_topic`, and the string safety net are all UNCHANGED
-- only the orchestration of how the off-bridge Qwen mouth is stepped between them. `run_battery` was
extended to run BOTH continuation techniques per topic/seed (same prompt, same seed) so any future run
reports a direct, disclosed A/B rather than a bare claim, and gained a `--seeds` comma-list so a multi-seed
sweep loads the (expensive) Qwen faculty ONCE and reuses it across seeds. No `sim/` edit; no change to
`webapp/open_ended_chat.py` (the call site's signature is unchanged, so the flag-off byte-identical
guarantee and the existing wiring verify are both untouched by construction, not by re-proof).

## Results (this session)

**Wiring sanity (decisive for what it claims, CPU-only, deterministic).**
`_open_ended_gen_time_tokenid_continuation_wiring_verify.py` (new) builds a fake, word-level, deterministic
tokenizer + a fake model that emits a fixed two-sentence script across two `generate()` calls -- the FIRST
sentence is the exact MUST_DROP adversarial sentence `ADVERSARIAL_SENTENCES["canada"]` already uses ("Canada
is bordered by the United States to the south and Mexico to the west.") -- and runs it through the REAL,
unmodified `_generate_tokenid_continuation` and the REAL, unmodified `clause_filter_sentence` (not stubbed).
20/20 checks GO: (A) with `facts=[("borders","united states")]` the repair matches `clause_filter_sentence`'s
own output exactly and Mexico is dropped while United States is kept; (B) the repaired span is a genuine
re-encode (a different, shorter token length than the model's own unedited words); (C) the KEPT second
sentence's text is byte-identical to what the fake model generated (nothing edited, nothing re-encoded); (D)
LESIONED (`facts=[]`) leaves Mexico in place, matching `run_controlled_unit_battery`'s own
`LESIONED_keeps_wrong` property; (E) the final ON text differs from the final LESIONED text -- the
load-bearing vary/lesion property, demonstrated at the orchestration level; (F) generation stops correctly
on EOS / sentence-count limits; (G) a token budget smaller than the full reply truncates cleanly, never
raises. Full log: `research/findings/raw/_open_ended_gen_time_tokenid_continuation_wiring_verify.json`.

**No-regression on the existing, unmodified gates (re-run this session, numpy backend).**
`_open_ended_gen_time_consensus_veto_wiring_verify.py` (the `answer_turn`/flag-routing wiring, which stubs
the mechanism function entirely) -- still **6/6 GO**, confirming the flag-off byte-identical guarantee and
the flag-on routing are untouched by this file's internal orchestration change.
`_open_ended_clause_contradiction_filter_verify.py` (the string moat itself, unmodified by this session) --
still **GO** (10/10 catch, 0 leaks), confirming the safety net this mechanism layers on top of is untouched.

## Honest scope

**Not run this session, by design:** the decisive live-mouth GPU comparison -- does `continuation="token_id"`
actually raise the ON/LESIONED divergence rate on the REAL off-bridge Qwen (cupy backend, real organs) above
the 2026-08-27 baseline's 1/3? `run_battery` now computes both techniques side-by-side and `main` reports
`n_live_diverged` (token_id) vs `n_live_diverged_text` (text) for exactly this comparison, but running it
costs a real Qwen load + generation across 3 topics x 2 techniques x 2 lesion-states x N seeds on the GPU --
out of scope for this session's CPU-only, bounded-memory build pass. **This finding does not claim a result
for that comparison** -- it is staged on `research/queue/gpu.queue` (6 seeds: 42/43/44/100/101/102) for the
controller to run and, if the token_id technique's live-mouth divergence rate is genuinely higher (not
hollow) with no regression on the PRIMARY controlled battery or the safety net, this arc's status can move
from PARTIAL toward GO on the live-mouth rung specifically named by the parent finding.

The CPU-only real-organ re-confirmation of `run_controlled_unit_battery` (the PRIMARY, decisive, already-GO
evidence from 2026-08-27) was started this session on `SIM_BACKEND=numpy` and NOT completed -- real
Izhikevich organs rebuilt per `combine()` call on CPU-only numpy is slow (the runtime's own warning: 10-50x
slower than cupy), and this session's only change to that code path was adding a `"seed"` key to its output
rows (non-functional). The parent finding's 3/3 GO on that battery is unaffected by this session's edits and
is not re-claimed here as a fresh measurement -- cite `2026-08-27-open-ended-generation-time-honesty-
PARTIAL.md` for it.

NEXT: run the staged 6-seed cupy GPU verify; if `n_live_diverged` (token_id) > `n_live_diverged_text` with no
regression, promote the arc toward GO on the live-mouth rung. Remaining NEXT items the parent finding also
named and this session did not attempt: skip-and-continue past a dropped sentence (reach later same-reply
residuals); broaden past capital/continent/borders toward the store's fuller relation set. NO `sim/` edit.
