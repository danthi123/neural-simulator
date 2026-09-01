---
type: finding
status: wired
date: 2026-09-01
mechanism: np-entailment-moat-gate
lane: integration
seeds: [42]
seed-waiver: A deterministic wiring/lesion verify of an entirely non-stochastic pipeline (spiking role
  assignment trained once at a fixed seed, then a pure extraction+entailment CLASSIFICATION over fixed
  sentences) — the evidence is catch/leak booleans and byte-equality checks against a fixed adversarial
  set + real saved Qwen replies, run through the REAL webapp entry point, not a stochastic effect a seed
  sweep would move. BridgeParser's own 2026-08-20 GO and NPHeadBinder's own GO already establish their
  spiking mechanisms independent of this wiring; this verify is scoped to whether the WIRING is
  load-bearing and attributable, which a seed sweep does not bear on.
instrument: research/runners/_np_entailment_moat_gate_wiring_verify.py — runs the WIRED
  `webapp.open_ended_chat.post_filter` (flag on vs off) plus two independent component-level lesions
  (entailment classification forced to always "grounded"; NPHeadBinder-based extraction forced to always
  fail) with tools.verdict.Verdict.
runner: research/runners/_np_entailment_moat_gate_wiring_verify.py
external: NO-EXTERNAL-NEEDED — this wires two already-validated, REUSED-UNCHANGED mechanisms
  (BridgeParser's position x voice spiking role parser, 2026-08-20 GO; NPHeadBinder's spiking NP-boundary
  binding, its own GO de-risk; FactStore/classify_claim entailment, the original open-text moat verifier
  de-risk) into a new live code path. It is a wiring/attribution verify of existing validated mechanisms,
  not a new mechanism claim.
artifacts:
  - research/findings/raw/_np_entailment_moat_gate_wiring_verify.json
---
# NPHeadBinder + entailment wired into the live open-text moat verifier's known-topic path — measured LOAD-BEARING (default-OFF)

Artifact: research/findings/raw/_np_entailment_moat_gate_wiring_verify.json (runner verdict: GO).

**One line.** `webapp/open_ended_chat.py`'s live known-topic post-filter only recognized THREE relation
shapes (borders/continent/capital) plus a bare number/year regex, so a fabricated supplement on ANY OTHER
relation ("Mercury discovered Neptune" against a store holding only mercury/orbits/sun) tripped no branch
and leaked through unedited. A new module, `webapp/np_entailment_moat_gate.py`, wires NPHeadBinder
(spiking NP-boundary binding) + `classify_claim`/`FactStore` entailment (both reused UNCHANGED by import)
into that same post-filter path, gated by a new flag, `BRAIN_OPEN_ENDED_NP_ENTAILMENT` (default OFF). The
new gate catches that whole class of wrong-relation confabulation the gazetteer is structurally blind to
— measured directly, through the real production entry point, not asserted — and the catch vanishes when
either the entailment classifier or the NPHeadBinder-based extractor is independently lesioned, with the
flag left on. Additive: with the flag off, `post_filter` is byte-identical to before this change.

## The gap (why this is not a hollow flip)

The live known-topic filter runs `_clause_filter_sentence` -> `sentence_contradicts`
(`research/runners/_open_ended_known_supplement_filter_derisk.py`), a HOST GAZETTEER: it only has branches
for `borders`, `continent`, `capital`, and a bare number/year regex. Any wrong supplement on a DIFFERENT
relation has no matching branch, so `sentence_contradicts` returns `None` and the sentence is kept
unedited — a confab leak the existing filter cannot see, by construction, regardless of how well-trained
or fluent the generator is. Two other de-risked mechanisms in this arc already solve the GENERAL version
of "does this clause's SVO hold": `NPHeadBinder` (`research/runners/_spiking_np_boundary_extraction_derisk.py`)
extracts an (agent, action, patient) triple for any clause `segment_clause` can close, via spiking,
vocabulary-agnostic (BridgeParser's own words: "the parser is vocabulary-agnostic ... assigns roles by
word position x voice") role assignment; `classify_claim`/`FactStore`
(`research/runners/_open_text_moat_verifier_derisk.py`) is the SAME entailment semantics production's
single-triple `ask_yes_no` already uses. Neither had been wired into the live open-text path — this
change does that.

## The wiring

- `webapp/np_entailment_moat_gate.py` (new file). `gate_sentence(sent, topic, facts, ...)`: splits `sent`
  into clauses (`split_clauses`, reused unchanged), skips hedge/opinion clauses (`is_opinion`, reused
  unchanged), extracts each remaining clause with `extract_svo_npbind` (NPHeadBinder + BridgeParser,
  reused unchanged), skips a clause whose subject does not normalize to the retrieved topic (out of this
  gate's fact-store scope) or whose verb is a copula ("is"/"are"/"was"/"were" — see Scope below), and
  otherwise classifies the extracted triple against a `FactStore` built from the SAME
  `(agent, action, patient)` facts `answer_turn` already retrieved, via `classify_claim` (reused
  unchanged). Returns the sentence unchanged unless a clause is classified `!= "grounded"`, in which case
  the WHOLE sentence is dropped. `_get_spiking_pair()` builds a process-shared `(BridgeParser,
  NPHeadBinder)` pair once, lazily, under a lock (mirrors `open_ended_chat.get_generator`'s pattern).
- `webapp/open_ended_chat.py`: `post_filter`'s known-topic branch now runs, per sentence, the existing
  `_clause_filter_sentence` FIRST, and — only when `np_entailment_enabled()` is truthy AND that sentence
  survived — additionally screens it with `gate_sentence`. Monotonic-only: the new gate can only drop a
  sentence the earlier stage already kept, never restore one. `np_entailment_enabled()` reads
  `BRAIN_OPEN_ENDED_NP_ENTAILMENT` (default OFF), a SECOND, independent gate stacked on the existing
  `BRAIN_OPEN_ENDED` flag — both env-read functions live beside the file's other three flags in the same
  style. `webapp/np_entailment_moat_gate` is imported LAZILY, only inside the flag's truthy branch
  (mirroring the file's existing `wkv_mouth_generator` / gen-time-honesty pattern), so a flag-off run never
  imports it — and therefore never triggers the spiking build or that module's own `SIM_BACKEND` default.
  No change to `webapp/server.py`: `post_filter`'s call signature is unchanged, so the existing
  `/api/brain-chat` -> `answer_turn` -> `post_filter` call path reaches this gate unmodified whenever
  `BRAIN_OPEN_ENDED` is truthy — reachable from the production endpoint on some request, the code
  condition this repo's terms file ties to that word, though the flag stays off by default (see below).

## The load-bearing proof (measured, `_np_entailment_moat_gate_wiring_verify.py`)

Two adversarial cases, each a wrong supplement on a relation the gazetteer cannot see:

- `mercury_discovered` — topic `mercury`, facts `[(mercury, orbits, sun)]`, raw
  `"Mercury orbits the sun. Mercury discovered Neptune."` Flag OFF (same `post_filter` call):
  `"Mercury orbits the sun Mercury discovered Neptune"` — **leaked**. Flag ON: `"Mercury orbits the sun"`
  — **caught**, true content intact.
- `einstein_invented` — topic `einstein`, facts `[(einstein, developed, relativity)]`, raw
  `"Einstein developed relativity. Einstein invented the telephone."` Flag OFF:
  `"Einstein developed relativity Einstein invented the telephone"` — **leaked**. Flag ON:
  `"Einstein developed relativity"` — **caught**, true content intact.

Both cases are `load_bearing: true` in the artifact (leaks flag-off, caught flag-on, true fragment
survives both ways) — the SAME function, SAME inputs, verdict changes ONLY with the flag.

## The component-level lesions (attributes the catch to the two NAMED mechanisms, not "some code path")

With the flag ON, each mechanism was independently lesioned by monkeypatching the SOURCE module it is
imported from (`gate_sentence` re-imports both by name on every call, so the source patch takes effect
immediately):

- **Entailment lesion**: `research.runners._open_text_moat_verifier_derisk.classify_claim` forced to
  always return `"grounded"`. Both catch cases leaked again (`catch_vanished: true` for both) — NPHeadBinder
  still extracted the correct triple, the flag was still on, but with entailment unable to say "no" the
  gate could not drop anything.
- **Extraction lesion**: `research.runners._spiking_np_boundary_extraction_derisk.extract_svo_npbind`
  forced to always return `(None, None)` (nothing parses). Both catch cases leaked again — entailment
  classification was untouched and the flag was still on, but with no triple ever extracted the gate had
  nothing to classify.
- **Restore sanity**: with both lesions reverted (flag still on, nothing monkeypatched), both cases were
  caught again (`leaked: false`) — confirms the lesions above were real and reversible, not a broken
  harness silently always reporting a leak.
- **Attribution** (`tools.lab.attributable_to`, not just both arms sitting side by side): catch rate
  flag-on-unlesioned = 1.0 (2/2), catch rate entailment-lesioned = 0.0 (0/2), catch rate
  extraction-lesioned = 0.0 (0/2) — `attributable_to(..., 1.0, 0.0)` = **1.0** for both lesions: 100% of
  the catch is attributable to each named mechanism, 0% is present in its lesioned control.

This is the two-sided proof the task asked for: varying either named mechanism's state changes the moat
verdict, and each change vanishes when that mechanism specifically is lesioned (with the flag itself,
and the OTHER mechanism, left untouched).

## False-reject safety (the gate's declared monotonic-only scope, measured)

Three cases, each expected UNCHANGED whether the flag is on or off — and measured so:

- `grounded_kept` — `"Mercury orbits the sun."` (a true, non-copula, on-topic claim): kept both ways.
- `offtopic_agent_untouched` — `"Newton discovered gravity."` under topic `einstein`: the extracted
  subject (`newton`) does not match the retrieved topic, so the gate is out of its adjudicable scope
  (the fact store handed to it holds only `einstein`'s facts) — untouched both ways.
- `copula_untouched` — `"Canada is a vast country located in North America."`: the extracted verb
  (`is`) is a copula, excluded by design. Without this exclusion the same clause extracts as
  `(canada, is, "vast country located in north america")`, and strict entailment against a store keyed on
  `isa`/`capital`/`continent`/`borders` FALSE-REJECTS this true, merely elaborated sentence — measured
  directly while building this gate (not hypothesized), which is why copula verbs are explicitly out of
  scope (see `webapp/np_entailment_moat_gate.py`'s docstring, point (d)).

**Real-data regression**: the 3 saved known-topic Qwen replies this arc's own prior wiring verify used
(canada/france/morocco, `_open_ended_verify_postfilter_derisk.json`) are **byte-identical** through
`post_filter` flag-on vs flag-off — the new gate adds zero additional drops on real generated prose, because
their wrong content (wrong borders, an unsupported founding year) already falls inside the pre-existing
gazetteer's own coverage and is caught before this gate ever runs; their true content (multi-word
descriptive predicate nominals — "a vast country located in North America") falls under the copula
exclusion and survives.

## Byte-identical when OFF

`_check_off_path_no_import`: with the flag off, calling `post_filter` never adds
`webapp.np_entailment_moat_gate` to `sys.modules` — the module (and the heavier BridgeParser/NPHeadBinder
build + spiking-backend default it would otherwise pull in) is never imported. `np_entailment_enabled()`
reads `False` when unset and `True` when set (both checked directly, not inferred). Combined with the
real-data regression's byte-equality and the safety cases' unchanged output, `post_filter`'s flag-off
behavior is unchanged from before this file existed.

## Default state, and why it stays off for now

`BRAIN_OPEN_ENDED_NP_ENTAILMENT` defaults OFF, stacked under the ALREADY default-off `BRAIN_OPEN_ENDED`
— so nothing here changes the production default turn. The scope decisions above (skip unparseable
clauses, skip off-topic subjects, skip copula verbs) were sized against a handful of adversarial +
real-saved-reply cases, not live traffic; before proposing a default-on flip the honest next step is
broader coverage measurement against a larger sample of real open-ended replies (mirroring how
`BRAIN_OPEN_ENDED_WKV_MOUTH` earned its later default-on flip only after its own dedicated measurement
rung), not asserting safety from this smaller adversarial set alone.

## Honest scope (named limits, not hidden ones)

- **Coverage, not correctness-everywhere.** `extract_svo_npbind`'s `segment_clause` only closes a clause
  that is exactly 3 content words (fully general over the verb), a recognized copula/passive frame, or a
  longer span whose verb is in its own narrow 3-word lexicon. Most compound, listy, or long descriptive
  sentences fall outside this and the gate is a no-op on them (unparsed -> pass through). This bounds the
  gate's value to the subset of clauses it can confidently parse; it never causes a false reject on what
  it cannot parse, but it also does not extend the gazetteer's blind spot to every possible sentence
  shape.
- **Topic-as-subject only.** The retrieved fact store holds only the ONE topic's facts, so this gate
  cannot adjudicate a claim whose extracted subject is a different entity (a genuine role-swap onto a
  non-topic subject, e.g. "the cat chases the dog" under topic "dog", is out of scope by the same
  agent-match check that keeps `offtopic_agent_untouched` safe — this is a declared boundary, not
  something this wiring claims to catch).
- **Negation and multi-clause antecedent-carry are inherited, not extended.** `extract_svo_npbind` has
  no antecedent-carry across clauses (unlike the more elaborate `_moat_claim_entailment_derisk.py`
  verifier, a different pipeline not reused here) and its negation coverage is whatever `segment_clause`'s
  own NEGATORS handling already provides — neither is modified or re-verified by this wiring; only the
  entailment CLASSIFICATION of whatever DOES extract is newly checked in production.

NEXT: measure false-reject / catch rate against a larger sample of real open-ended replies (beyond the 3
saved known-topic ones on hand) before considering a default-on flip; the copula-object entailment problem
(matching a rich predicate nominal, not a single store object) is the natural next rung, same shape as the
2026-08-21 finding's own named "next rung" (NLI / store-backed entity check). NO `sim/` edit.
