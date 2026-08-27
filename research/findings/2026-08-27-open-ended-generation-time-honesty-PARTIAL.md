---
type: finding
status: contributing
date: 2026-08-27
mechanism: open-ended-generation-time-consensus-veto
lane: E-language-open-ended-honesty
seeds: [42]
seed-waiver: The PRIMARY, decisive evidence (the controlled unit battery) is deterministic filter logic over
  a fixed adversarial sentence per topic, sourced from a genuinely-spiking but seed-stable LTM-exempt
  consensus verdict (organ_a_recall/committed match across builds at a fixed seed) -- a seed sweep would not
  change the ON/LESIONED outcome, same waiver shape as the 2026-08-21/08-27 filter findings this arc builds
  on. The SECONDARY live-mouth confirmation is genuinely seed-dependent (Qwen's own greedy decode content);
  it is reported as opportunistic, not decisive, and NOT used to gate the GO/PARTIAL call -- see Honest scope.
instrument: research/runners/_open_ended_gen_time_consensus_veto_derisk.py -- a controlled unit battery
  (clause_filter_sentence fed facts sourced from webapp.gnw_two_organ_bus/gnw_three_organ_bus's
  two_organ_combine/three_organ_combine instead of a static table) plus a live sentence-by-sentence
  off-bridge-Qwen generation loop, both with tools.verdict.Verdict; a separate wiring verify
  (research/runners/_open_ended_gen_time_consensus_veto_wiring_verify.py) checks the webapp.open_ended_chat
  integration in isolation (stubbed generator + stubbed mechanism, no GPU/organs).
runner: research/runners/_open_ended_gen_time_consensus_veto_derisk.py
external: NO-EXTERNAL-NEEDED -- reuse-by-import of already-GO, already-production machinery
  (webapp.gnw_two_organ_bus / webapp.gnw_three_organ_bus, default-ON since 2026-08-20/08-21, plus the
  2026-08-27 LTM-exemption flip; the 2026-08-21/08-27 string contradiction/clause filters); no new mechanism
  is proposed, only a new PROVENANCE (live consensus vs static table) for an existing filter's inputs.
artifacts:
  - research/findings/raw/_open_ended_gen_time_consensus_veto_derisk.json
  - research/findings/raw/_open_ended_gen_time_consensus_veto_wiring_verify.json
  - research/findings/raw/_open_ended_clause_contradiction_filter_verify.json
---
# Generation-time honesty: the LTM-exempt organ-B/C spiking consensus suppresses a known-supplement clause AT generation, not after (PARTIAL)

Artifact: research/findings/raw/_open_ended_gen_time_consensus_veto_derisk.json (GO on the controlled,
decisive claim; live-mouth confirmation on 1/3 probed topics, reported honestly, not gating).

**One line.** Honesty moved from a post-hoc STRING filter to a live SPIKING signal that shapes generation:
for a known-topic candidate clause, `webapp.gnw_two_organ_bus.two_organ_combine` /
`gnw_three_organ_bus.three_organ_combine` (the SAME LTM-exempt organ-B/C consensus that already authors the
strict/rich recall path, production default-ON) supplies the ground truth `clause_filter_sentence` checks
against -- not a static python dict -- and a wrong clause is suppressed/repaired BEFORE it is fixed into the
context later sentences are generated from, not after the whole reply is written.

## The mechanism

`research/runners/_open_ended_gen_time_consensus_veto_derisk.py` (new): `build_consensus_chat` builds a
lightweight, genuinely-spiking chat (the existing tiny-demo buffer composer + a fresh `ShardedPhasorStore`
LTM holding canada/france/morocco's capital/continent/borders facts, composed via `TieredFactStore` -- the
identical drop-in the production knowledge-in-chat flip uses). `consensus_facts_for_topic` asks the LIVE
consensus what it COMMITS for each (topic, relation) -- `organb_ltm_exempt=True` genuinely applies
(`recall_source="ltm"`, confirmed for all 3 topics x 3 relations), not vacuously. `generate_with_generation_
time_veto` steps the off-bridge spiking Qwen ONE SENTENCE at a time (greedy continuation from the growing
ACCEPTED text) and runs each candidate through the UNCHANGED, imported `clause_filter_sentence` fed those
consensus-sourced facts; only the accepted (possibly repaired) text shapes what the mouth generates next.
`lesion_coupling=True` -- this wiring's own lesion, distinct from the organs' internal biology levers --
returns `facts=[]` without ever calling the consensus, severing exactly the coupling under test. Wired into
`webapp/open_ended_chat.py` (`answer_turn`'s new `chat=` parameter + `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`, a
SECOND independent flag on top of `BRAIN_OPEN_ENDED`, both default OFF) and `webapp/server.py` (passes the
live, organ-wired `chat` through); the existing string post-filter (`post_filter`) still runs afterward on
whatever this mode emits, unconditionally -- a safety net, never bypassed.

## Results

**PRIMARY (controlled, decisive).** A fixed adversarial sentence per topic carrying the store's own exact
MUST_DROP wrong border (e.g. "Canada is bordered by the United States to the south and Mexico to the west.")
run through `clause_filter_sentence` with `facts` sourced from the live consensus: coupling ON drops the
wrong border and keeps the correct one on **3/3 topics** (canada->mexico dropped/united states kept,
france->italy dropped/spain kept, morocco->algeria dropped/spain kept); coupling LESIONED (facts=[]) leaves
the wrong border in place, unchanged, on **3/3 topics**. This is deterministic (no live generation involved)
and directly traceable to the exact clause class the mission named (recall/known-supplement clauses).

**SECONDARY (live, off-bridge Qwen mouth, opportunistic).** The sentence-by-sentence continuation decode is
NOT byte-identical to one-shot generation (a text-roundtrip retokenization effect -- see Honest scope), so a
given run's ON and LESIONED decodes may or may not diverge. This run they diverged on **1/3** topics: canada
ON ends "...bordered by the United States to the west." while LESIONED continues "...and Europe to the east.
The country has a rich cultural heritage..." -- the consensus caught and repaired a live, spontaneous
fabrication (a wrong continent claim) that this run's incremental decode produced, at the exact point it
would have shaped the next sentence. France and morocco's ON/LESIONED decodes were BYTE-IDENTICAL this run
(nothing for either variant to suppress) -- reported as UNDEFINED for the live vary/lesion check on those 2
topics, not counted as a pass. **No-regression**: the unchanged string safety net leaked **0/3 both ON and
LESIONED** on the live end-to-end output (never less safe than before this file existed), and the pre-existing
`_open_ended_clause_contradiction_filter_verify.py` battery (unmodified) still reads **10/10 catch, 0 leaks,
GO** after this change. **Wiring**: `_open_ended_gen_time_consensus_veto_wiring_verify.py` (stubbed generator
+ stubbed mechanism, no GPU/organs) is **6/6 GO**: flag OFF is byte-identical whether or not a `chat` is
passed (exact string equality, not inferred); flag ON + `chat=None` falls back byte-identically; flag ON + an
unknown topic is untouched; flag ON + a known topic + a live chat routes to the mechanism with the correct
(topic, seed, system, user, chat, generator) arguments; the safety net still runs on the gen-time output;
`webapp/server.py` passes `chat=chat` at the still-singly-imported, still-`BRAIN_OPEN_ENDED`-gated call site
(`git diff --stat` shows a 5-line additive change to server.py, 79-line additive change to
open_ended_chat.py -- no line removed, no existing branch altered).

## Honest scope

**Disabled, named**: a live-mouth vary/lesion demonstration on EVERY probed topic. The sentence-by-sentence
continuation technique re-tokenizes `prompt + accepted_text` on each step rather than continuing token IDs
directly, so it is a genuinely DIFFERENT decode from one-shot generation (confirmed: canada's ON/LESIONED text
differs from the saved one-shot reply from the first divergent sentence on) -- an honest, disclosed property
of this v1 continuation technique, not a defect in the consensus mechanism itself, which the controlled unit
battery isolates and proves cleanly regardless. Only 1/3 topics happened to fabricate live content this run
for either variant to suppress; a next rung (token-ID continuation via `past_key_values`, or a larger topic
sample) would raise that hit rate. **v1 also conservatively STOPS generation on an unrepairable sentence**
(truncates the reply there) rather than skip-and-continue past it -- never fabricates past an unverifiable
point, but a later sentence in the same reply may go untested by a single pass (the string safety net still
covers whatever text is ultimately emitted). Relations checked are capital/continent/borders, matching the
string filter's own structural scope; a bare unsupported number/date is caught by that filter's OWN
facts-independent branch either way, unaffected by (and not attributed to) this file.

NEXT: token-ID continuation (drop the retokenization confound) to raise the live-mouth divergence rate across
more topics/seeds; skip-and-continue past a dropped sentence to reach later same-reply residuals; broaden
past capital/continent/borders toward the store's fuller relation set. NO `sim/` edit.
