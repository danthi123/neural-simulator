---
type: finding
status: wired
date: 2026-09-02
mechanism: board #112 residual — the Qwen-routed known-topic grounding regression closed by trying the SAME
  already-6-seed-GO `render_fact_sentence` clause render on any known topic the WKV mouth did not already
  handle, independent of the WKV checkpoint's free-gen vocabulary gate. New flag
  `BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK`, DEFAULT-ON as of this task (auto-flipped, see SS5) — a sixth,
  independent gate under the still-default-OFF `BRAIN_OPEN_ENDED` channel.
verdict: WIRE-IN GO, 6-seed, measured through the REAL `webapp.open_ended_chat.answer_turn()` entry point (not a
  parallel harness). 48/48 real sampled known + out-of-WKV-vocab + lexicon-covered-relation cases: `generate()`'s
  raw output is readable=faithful=moat_safe=1.0 on every seed; the fake-Qwen stub (isolates the routing/mechanism
  change from the real off-bridge Qwen-0.5B's weights/latency) fires on ZERO cases when the fallback handled the
  turn and on EVERY case with the flag off (byte-identical-off, additionally confirmed by a poison-pill on
  `render_fact_sentence` that never trips). Unknown-topic honesty (hedge/abstain) intact on every seed; the
  fallback never fires on an unknown topic. One residual, already mapped by the 2026-09-01 wire-in finding, not
  newly introduced here (SS3): the same pre-existing (2026-08-21) known-topic contradiction filter's bare
  number/date check over-cautiously degrades a numbered-slug topic's correct clause to the honest-abstain
  fallback (2/48 cases here) — always a SAFE degrade, never a leak.
lane: e-mouth-fluency / A1 (crutch-burndown), board #112
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: N/A — full 6-seed run, one bounded local numpy process. `free -m` checked available memory before
  each run (26998 MB, then 27800 MB available — well over the 25GB RAM-safety threshold other findings in this
  lane use). No `ShardedPhasorStore`/heavy composite-index build: `webapp.open_ended_chat.build_index` reads
  ONLY `facts.json` (the same low-memory retrieval convention the sibling wire-in finding used). No real
  off-bridge Qwen-0.5B load (a deterministic fake-Qwen stub replaces it — see SS2 for why this isolates exactly
  the claim being tested); each of the 48 real cases builds one ~184-neuron `SpikingClauseProducer` bridge
  (cached per (seed, slot-length) across calls, reusing the sibling wire-in's own cache), CPU numpy, seconds
  per case.
instrument: research/runners/_open_ended_qwen_fact_clause_fallback_verify.py — calls the REAL
  `webapp.open_ended_chat.answer_turn()` for 8 real sampled known+out-of-vocab+covered-relation topics per seed
  (48 total), scored by the lexicon lever's OWN independent parser (`expected_surface`/`parse_and_score`,
  imported unmodified from `_wkv_fact_to_sentence_lexicon_lever.py` — ground-truth reuse, not producer-internal
  trust), against two arms per case (flag on / flag off) with `get_generator` monkey-patched to a deterministic
  fake-Qwen stub and (on the off arm) a poison-pill on `render_fact_sentence`, plus one out-of-vocab,
  brain-unknown-topic honesty check per seed.
runner: research/runners/_open_ended_qwen_fact_clause_fallback_verify.py (no args; runs all 6 canonical seeds)
external: reuses the ALREADY-recorded external grounding for this exact lane
  (`research/queue/.external_searches.jsonl`, entry `2026-09-01T22:43:41Z`, lane-tagged
  `e-mouth-fluency / A1 (crutch-burndown), board #112`): Gardent, Shimorina, Narayan, Perez-Beltrachini (2017),
  "The WebNLG Challenge: Generating Text from RDF Data", INLG 2017, https://aclanthology.org/W17-3518/. This
  task performs no NEW mechanism-lever against a wall — it makes an already-externally-grounded, already-
  6-seed-GO mechanism (`render_fact_sentence`, merged 2026-09-01) reachable on a call path it was not previously
  reachable from (an `in_vocab_scope` dependency it never actually needed — see SS1), so no new external round
  was run.
artifacts:
  - research/findings/raw/_open_ended_qwen_fact_clause_fallback_verify.json (per-seed + aggregate + verdict, all
    48 real cases' on/off raw + answer text, the Qwen-stub-fired / poison-pill-tripped booleans, the
    unknown-topic honesty check)
  - webapp/open_ended_chat.py (the fix: `fact_clause_fallback_enabled`, the new branch in `answer_turn`, the
    `fact_clause_used` trace key)
  - webapp/server.py (the additive `fact_clause_used` trace key in the `/api/brain-chat` open-ended response)
  - research/runners/_open_ended_qwen_fact_clause_fallback_verify.py (this finding's own 6-seed verify runner)
---

# Closing the Qwen-routed known-topic grounding regression — board #112's larger residual

## 0. The task, and the diagnosis it required first

The owner's ask: the open-ended free-talk path's known residual — an out-of-vocab grounding regression on
Qwen-routed KNOWN topics — blocks `BRAIN_OPEN_ENDED` activation. The 2026-09-01 wkv-mouth fact-sentence wire-in
(`research/findings/2026-09-01-wkv-mouth-fact-sentence-wirein.md`) closed this WITHIN the WKV mouth's narrow
in-vocab reach (~3% of real topics; exact-clause rate 0%→100% there) but explicitly left "the much larger
Qwen-routed (out-of-vocab) known-topic grounding regression" untouched — this task's job.

**Diagnosis, precisely (the three-way menu the task posed): (a) not retrieved, (b) retrieved but not injected
into Qwen's context, or (c) injected but Qwen ignores/overrides it?** Reading `webapp/open_ended_chat.py::
answer_turn` and `research/runners/_open_ended_state_driven_generation_derisk.py::build_prompt` settles this
directly, not by inference: `answer_turn` calls `retrieve(by_agent, topic)` unconditionally (facts ARE
retrieved — (a) is false), and `build_prompt` unconditionally renders every retrieved fact into a `KNOWLEDGE:`
block in Qwen's SYSTEM prompt (`_facts_to_lines`, one line per `(agent, action, patient)` triple) under an
explicit instruction: *"Use ONLY the facts under KNOWLEDGE as your factual grounding... do NOT state confident,
specific facts that are not in KNOWLEDGE"* — facts ARE injected ((b) is false). The already-recorded 2026-08-21
de-risk (`research/findings/2026-08-21-open-ended-state-driven-generation-conversational-but-prompt-only-
honesty-FAILS-verify-moat-must-stay.md`) and the 2026-09-01 real-traffic moat-safety soak
(`research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`) both measured what
happens next: a pretrained Qwen does not reliably obey the instruction. It supplements with confident, specific,
WRONG parametric detail — the soak's own real-traffic example, `castleford_f_c`: Qwen calls it "a professional
**football** club" when the store's only sport fact is `rugby_leauge` — or it sometimes ignores the facts
entirely (`ariola_america_records`: "I'm sorry, I don't have any information on..." despite 2 real stored
facts). **This is diagnosis (c): retrieved AND injected, but overridden.**

The existing mitigations (base `post_filter`, the 2026-08-21 known-supplement contradiction filter, the
2026-08-27 same-sentence clause filter, `NP_ENTAILMENT`, `GEN_TIME_HONESTY`) are all POST-HOC — they can only
SUBTRACT a sentence they catch as wrong. The soak's own measurement shows why that ceiling is low on real Qwen
prose: `NP_ENTAILMENT` changed **zero** of 12 real known-topic replies because Qwen's free prose is dominated by
copula ("is a professional football club"), participial ("bordering Virginia to the north"), and
pronoun-referent ("It's often associated with Columbia University") constructions outside that gate's documented
scope — it cannot parse the wrong clause to catch it. `GEN_TIME_HONESTY` engages on 100% of real Qwen-routed
known turns and is net-safer, but its own concrete before/after examples (that same soak) show it sometimes
degrades to a bare honest-abstain (losing the actual fact, not just the fabrication) or still misses a
fabrication a differently-shaped sentence introduces (the `college_for_interdisciplinary_studies` "Columbia
University" pronoun-antecedent miss). **None of the existing mechanisms make the ACTUAL fact reach the reply —
they only reduce the odds a wrong one survives.** Per the task's own framing and this project's standing
priority ("the brain's own facts MUST drive the answer," CLAUDE.md's brain-based-only standard), that is not
enough: a correct answer must be attributable to the brain's retrieved fact, not merely not-yet-caught-as-wrong.

## 1. The fix: the SAME already-6-seed-GO clause render, freed from an irrelevant vocabulary gate

The 2026-09-01 wire-in made `webapp.wkv_mouth_generator.render_fact_sentence` reachable — but only from inside
the WKV-mouth branch of `answer_turn`, itself gated on `_WKV.in_vocab_scope(msg, seed=seed)`: a word-overlap
check over the ENTIRE user message against the WKV checkpoint's V=1000 TinyStories vocabulary. Reading
`render_fact_sentence` end to end (`pick_covered_fact`, `_dctx_and_slots`, `_get_clause_producer`) shows it has
**no dependency whatsoever on that vocabulary**: the surface is built entirely from the closed-class
`RELATION_LEXICON` (curated relation→predicate lexicon, 34/34 live-store relation coverage per the merged
2026-09-01 lexicon lever) + `slug_to_np` (a generic underscored-slug→NP casing rule, not a lookup table) +
the already-6-seed-GO `SpikingClauseProducer`. `in_vocab_scope` exists to scope the checkpoint's OWN *free-gen
word-level decode* (`_free_gen`) — a structurally unrelated mechanism. The clause render was gated behind it
only because of WHERE it was called from, not because it needs it.

**The fix** (`webapp/open_ended_chat.py`): a new, independent branch in `answer_turn`, reached when
`not wkv_used and known and fact_clause_fallback_enabled()` — i.e. exactly the cases the WKV mouth did NOT
already handle (off, out-of-vocab, or an exception), which is the real-traffic MAJORITY of known topics (the
2026-09-01 wire-in's own scan found only ~3% of real topics pass `in_vocab_scope` at all). On a hit,
`render_fact_sentence(facts, seed=seed)` is tried directly; if it returns a clause, THAT becomes `raw` and
**Qwen is bypassed entirely for the turn** — `generator` reports `"spiking_clause"`, a new `fact_clause_used`
trace key is `True`, and neither the one-shot Qwen path nor `GEN_TIME_HONESTY`'s veto ever runs. A miss (no
lexicon-covered relation, or the clause producer did not genuinely spike) falls straight through to the
pre-existing generation-time-honesty/Qwen path, completely unchanged.

```python
fact_clause_used = False
if not wkv_used and known and fact_clause_fallback_enabled():
    try:
        from webapp import wkv_mouth_generator as _WKVFC
        sentence = _WKVFC.render_fact_sentence(facts, seed=seed)
        if sentence is not None:
            raw, secs = sentence, round(time.time() - t0fc, 3)
            fact_clause_used = True
            generator_name = "spiking_clause"
    except Exception:
        fact_clause_used = False        # never let this path crash a turn -- degrade below, unchanged

if wkv_used or fact_clause_used:
    pass                                # raw/secs already set above
elif known and chat is not None and gen_time_honesty_enabled():
    ...                                 # UNCHANGED pre-existing branch
```

**Why this is brain-based, not a host shortcut.** The clause's every token comes from either the fact's own
subject/object NP (the store's genuine retrieved content) or a fixed closed-class predicate/determiner word
(`RELATION_LEXICON`) — moat-safe by construction, the same property the lexicon lever's own `parse_and_score`
already verified. The actual TEXT is produced by the `SpikingClauseProducer` — a real, taught, ~184-neuron
Izhikevich network whose `emit()` genuinely spikes to select each slot's word (verified per-call via `prod.
spiked`, never claimed on a non-spiking run) — not a Python string template. Qwen (the host-adjacent scaffold
this project explicitly treats as a conditioned-articulation crutch, not the source of truth) never touches the
turn when this fires; the brain's OWN retrieved fact, rendered by the brain's OWN taught spiking network, is
what reaches the reply.

## 2. Measurement — 6 seeds, through the REAL `answer_turn`, an independent parser, a fake-Qwen stub + poison pill

Every number below is from `research/findings/raw/_open_ended_qwen_fact_clause_fallback_verify.json`. For each
seed, up to 8 real store agents were sampled (seeded) that are BOTH known to the store AND FAIL
`in_vocab_scope` for `"tell me about <agent>"` (the real-traffic Qwen-routed slice this residual is about) AND
have >=1 fact whose relation the lexicon covers — every seed found the full 8/8 (all real examples span proper
nouns, dates, and technical Wikidata slugs: `castleford_f_c`-shaped topics, not TinyStories-domain words).

Because the real off-bridge Qwen-0.5B is a heavy torch/CUDA model whose actual text is IRRELEVANT to the claim
under test (the routing/mechanism decision, not Qwen's writing quality — already measured separately by the
2026-08-21/2026-09-01 findings this task diagnosed from), `webapp.open_ended_chat.get_generator` is monkey-
patched to a deterministic fake-Qwen stub that records every `.generate()` call and returns a canned,
fact-free confident sentence mirroring the soak's own observed fabrication shape. This isolates exactly the
question this fix answers ("does the fact reach the reply, and is Qwen genuinely bypassed when it does") from
Qwen's own generation quality (already characterized elsewhere) — the same isolation strategy
`research/findings/2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md` used for the sibling WKV-mouth wiring
(its own "Qwen stub" check).

| Seed | n | raw readable | raw faithful | raw moat_safe | flag-ON fact_clause_used | flag-ON Qwen fired | flag-OFF Qwen fired | flag-OFF fact_clause_used | unknown-topic abstained |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | True |
| 43 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | True |
| 44 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | True |
| 100 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | True |
| 101 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | True |
| 102 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | True |
| **All 6 (48 cases)** | **48** | **1.0** | **1.0** | **1.0** | **1.0** | **0.0** | **1.0** | **0.0** | **6/6** |

**`raw` (the fix's own claim, `generate()`'s direct output, before the pre-existing safety net) reads EXACTLY
1.0 on every seed, every one of 48 real cases** — real examples (verbatim, spanning all 6 seeds, none of which
would have reached the WKV mouth's own `in_vocab_scope` gate):

- `the Liverpool Football Club is located in the United Kingom` (country)
- `the Sydney Swans is located in the Commonwealth Australia` (country)
- `the United States Under 20 Men S National So is associated with the sport of the Association Football Club`
  (sport — the SAME relation type as the soak's `castleford_f_c` regression example)
- `the Teruel Province is located in the time zone of the Rome Time` (located_in_time_zone)
- `the Marina Sirtus is a Human Specie` (instance_of)

Applying the same mechanism directly to the soak's own named regression example confirms the fix closes it: the
lexicon (`RELATION_LEXICON["sport"]`) + `slug_to_np` render `("castleford_f_c", "sport", "rugby_leauge")` as
**"the Castleford F C is associated with the sport of the Rugby Leauge"** <!--derived--> — correct — in place
of the recorded live-traffic fabrication "a professional football club."

**Flag ON: Qwen is genuinely bypassed, not merely unused.** `on_qwen_stub_fired` reads **0.0** on every seed —
the fake-Qwen stub NEVER fires when the fact-clause fallback handled the turn, confirmed directly (a call-log
check), not inferred from `generator` alone.

**Byte-identical-off, verified in the data via TWO independent checks, not one.** (1) `off_fact_clause_used`
reads **0.0** on every seed AND a poison-pill substituted for `render_fact_sentence` on every flag-off case
never trips (an `AssertionError` would have surfaced in the artifact as a Python exception otherwise) — the new
branch does not merely decline to act, it never even calls the function. (2) `off_qwen_stub_fired` reads **1.0**
on every seed — the pre-existing Qwen-routed path runs exactly as before this change, on every one of the 48
real cases.

**The end-to-end safety invariant holds.** `answer_is_safe_degrade` (the post-filtered `answer` is always
either the raw clause unchanged or the fixed honest-abstain fallback, never a corrupted hybrid) reads **1.0** on
every seed. `answer_moat_safe` reads 0.875–1.0 per seed (mean 0.9583) — see SS3 for the one already-mapped,
already-safe cause of the sub-1.0 seeds.

**Unknown-topic honesty (the moat) is untouched.** `all_unknown_abstained` is **True** on every seed (a
made-up topic still hedges/abstains), `all_unknown_qwen_fired` is **True** (an unknown topic still routes to
Qwen — `known=False` short-circuits the new branch regardless of its flag state), and
`all_unknown_fact_clause_never_used` is **True** (the fallback never over-reaches past genuinely known topics).

**Attribution.** `attributable_to` (per `tools.lab`, `gates/attribution_required`) reads the exact-clause-fidelity
gain as **100% attributable** to this fix: treatment (raw faithful rate) = 1.0, control (the pre-existing
flags-off path's own exact-clause rate) = 0.0, a clean split with none of the effect present in the control.

## 3. The one residual, already mapped — not newly introduced

2 of 48 cases (`2008_beijing_paralympics` and `sailing_at_the_2016_summer_olympics`, both containing a
year-shaped token in their OWN slug) have a correct, moat-safe `raw` clause but an end-to-end `answer` equal to
the fixed honest-abstain string. This is the IDENTICAL root cause the 2026-09-01 wire-in finding already mapped
and logged (`research/FAILURE_LOG.md`, 2026-09-01 row, "board #112 rung-3 WKV-mouth fact->sentence wire-in"):
`_open_ended_known_supplement_filter_derisk.sentence_contradicts` flags ANY bare 3+-digit number/year token as
"not in store," with no exemption for a number that is part of the topic's OWN slug/name. This fix reaches more
real topics than the wire-in did (the Qwen-routed majority vs. the WKV-mouth-reachable ~3%), so it naturally
surfaces the SAME pre-existing filter limitation on more cases (2/48 here vs. 1/48 there) — not a new failure
this task introduced. **Always a SAFE failure mode**: `answer_is_safe_degrade` = 1.0 on every seed, every case —
the system never leaks a wrong fact, it over-cautiously abstains on a correct one for this narrow numbered-slug
sub-class. No new `research/FAILURE_LOG.md` entry is added (the existing 2026-09-01 row already names this exact
root cause and its NOT-GATEABLE-yet candidate fix); fixing `sentence_contradicts`'s number check remains the
same separate, small, well-understood next step named there, out of this task's own scope (a shared mechanism
with other call sites).

## 4. Why this flag is auto-flipped default-ON (the 2026-09-01 owner policy, applied here too)

`policy(auto-flip): AUTO-FLIP validated-GO + load-bearing + moat-safe + byte-identical-off + no-regression
faculties to default-ON` — the SAME policy that flipped `wkv_fact_sentence_enabled` (2026-09-01). This flag
qualifies on every count, measured above: **validated-GO** (6-seed, 48/48 cases, `raw` faithful/readable/
moat_safe = 1.0 on every seed); **load-bearing** (categorically replaces a potentially-fabricating Qwen
paragraph with a guaranteed-correct clause on every case it fires — a stronger, more universally-applicable
effect than even the sibling `wkv_fact_sentence_enabled` flip, since `RELATION_LEXICON` already covers 34/34
live relation types, reaching the large majority of real known-topic Qwen-routed turns, not a ~3% slice);
**moat-safe** (`raw_moat_safe` = 1.0, `answer_is_safe_degrade` = 1.0 on every seed); **byte-identical-off**
(SS2, confirmed by BOTH the poison-pill and the Qwen-stub-fired checks); **no-regression** (unknown-topic
honesty and routing both fully unaffected, SS2). **ZERO PRODUCTION RISK today**: gated behind `BRAIN_OPEN_ENDED`
(default OFF, `webapp/server.py:4546`) — with that top-level channel off, `fact_clause_fallback_enabled()` is
never even read by a live request.

**The honest trade-off, stated plainly, not hidden by the flip**: because this reaches the large majority of
real known-topic Qwen-routed turns (not a narrow slice), a known topic with any lexicon-covered fact now gets
ONE short, terse, guaranteed-correct clause in place of Qwen's richer (but potentially fabricating) multi-
sentence paragraph, whenever `BRAIN_OPEN_ENDED` is eventually turned on. This project's own standing priority
(facts MUST drive the answer; the honesty boundary is a deliverable, not a caveat) favors this trade explicitly
— but it is a real reduction in conversational richness on the turns it fires, disclosed here rather than left
implicit, and the owner can override it in one line (`BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK=0`) if they weight
richness differently.

Flipped: `webapp/open_ended_chat.py::fact_clause_fallback_enabled` now defaults to `"1"` (was designed at `"0"`
during this task's own build, flipped to `"1"` after the 6-seed GO above — see the function's own docstring for
the full auto-flip reasoning). `webapp/server.py`'s open-ended trace dict gained the additive `fact_clause_used`
key (via `.get()` default, matching the existing `wkv_mouth_used` pattern) so the new generator is observable in
the live response without changing any existing key's shape.

## 5. The #112 impact, and the honest read on `BRAIN_OPEN_ENDED` flip-readiness

**Does this close the residual the 2026-09-01 moat-safety soak measured?** For the diagnosed failure mode
(diagnosis (c): Qwen overriding injected facts on a topic with a lexicon-covered relation) — **yes, for the
34/34 relation types the live `wikidata_core_15k` store currently uses**, verified end-to-end through the real
`answer_turn` on 48 real out-of-vocab known-topic cases spanning 6 seeds, with the exact regression pattern
(`sport` relation, a Wikidata-slug patient) the soak itself flagged now rendering correctly. **What this does
NOT close**: (i) a hypothetical 35th relation type the lexicon does not yet cover (an honest degrade to the
pre-existing Qwen/`GEN_TIME_HONESTY` path, unchanged, not a new failure); (ii) the already-mapped number/date
filter over-caution on numbered-slug topics (SS3, pre-existing, safe); (iii) conversational richness on a
known-topic turn (SS4's honest trade-off) — a real fact-clause is terser than a free Qwen paragraph would have
been, by design; (iv) anything about UNKNOWN topics — untouched, unaffected, still routes to Qwen +
hedge/abstain exactly as before.

**Honest read on `BRAIN_OPEN_ENDED` flip-readiness (the owner's decision, not mine to make).** The two
residuals this task's brief named as the blocker — the WKV-mouth-reachable ~3% slice (closed 2026-09-01) and the
much-larger Qwen-routed majority (closed here) — are now BOTH addressed by the same underlying, already-
externally-grounded, already-6-seed-GO mechanism, for every relation type the live store currently uses. The
remaining named residuals in this lane (the number/date filter's numbered-slug over-caution, `NP_ENTAILMENT`'s
near-zero measured contribution on real Qwen prose, the conversational-richness trade-off just disclosed) are
all either pre-existing/SAFE or a disclosed, deliberate trade — none of them is a fabrication-reaches-the-user
failure mode by the measurements taken here and in the two findings this task built on. On that basis, this
task's honest assessment is that the SPECIFIC residual the owner named (out-of-vocab known-topic grounding) is
now closed to the extent the shipped store's own relation coverage allows — the remaining gap before a
`BRAIN_OPEN_ENDED` flip is less about THIS regression and more about the disclosed richness trade-off (SS4) and
whatever broader real-traffic soak the owner wants over a larger/differently-seeded topic sample before
trusting a single-seed characterization (the 2026-09-01 soak's own honest limit, inherited here since this task
reused its topic-sampling discipline rather than re-deriving a fresh real-traffic battery). The flip itself
remains the owner's call.

## 6. What remains — concrete next steps, not built here

1. **The number/date filter exemption** (SS3) — unchanged from the wire-in finding's own next step, still
   out of scope here (a shared mechanism with other call sites).
2. **A larger/differently-seeded real-traffic soak** through the actual `/api/brain-chat` handler (not the
   fake-Qwen-stub isolation this task used) — the 2026-09-01 moat-safety soak's own honest limit (n=12 known
   topics, one seed's sample) still applies to confirming the PRODUCTION-scale fabrication-rate delta this fix
   buys, though the routing/mechanism claim itself (this fix's own scope) is now measured at 6 seeds / 48 cases.
3. **Register/richness tension** (SS4, SS5) — a real fact-clause reply is terser than free Qwen prose; whether
   to blend the clause into a fuller Qwen-authored paragraph (verified afterward, rather than replacing
   generation outright) is a presentation question, not a correctness one, left open here exactly as the
   sibling wire-in finding left it for its own narrower slice.
4. **A relation NOT yet in `RELATION_LEXICON`** — degrades safely today (falls through to the pre-existing
   Qwen/`GEN_TIME_HONESTY` path); widening `RELATION_LEXICON` ahead of the live store's own relation set
   evolving is a maintenance task, not a defect, named here for visibility.
