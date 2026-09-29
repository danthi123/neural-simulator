# Claude review of the weekend seam map and the first host-share probe (2026-09-29)

Reviewed on branch `research/host-share-baselines` (from `main` @ f52c6d246). Subject: the local model's work on
`research/draft-seam-map` (base 411782014): `docs/drafts/seam-map/host-decision-seam-map.md`, its `WORKLOG.md`,
`research/runners/_draft_host_share_probe.py` and `tests/test_draft_host_share_probe.py`. Plan context:
`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` sections 1-2. None of the cited source files changed
between 411782014 and f52c6d246 (`git diff --stat` empty for `webapp/server.py`, `brain_chat_tui.py`,
`rich_answer_composer.py`, the six gate wrappers, `value_choice_production_organ.py`, `metacog_production_organ.py`),
so the draft's line numbers apply to current main.

## Verdict in four lines

1. **Seam map: keep it, with corrections.** Every file:line I checked (24 rows) is accurate. The trace order and the
   wrapper-stack analysis are correct, and the stale-attribute finding on the deliberation gate holds in the code.
2. **The map's main error is one of classification, not location.** It reads `composer.kb` / `chat.stored_facts` as
   "the brain's own store". Those are a host-side shadow list of every heard fact, not a read of the substrate. Several
   content rows marked YES or PARTLY-neural are host decisions over that list. It also misses the most basic content
   seam: which stored fact answers a query is a host first-match loop by default.
3. **Probe: honest and runnable, but not yet a host-share measurement.** Its intervention is true by construction.
   It runs on a toy composer, not the chat path. Its degraded panel is confounded (next point).
4. **The owner's question: yes, the degraded panel confounds the reading.** Half of the rows where the two confidence
   reads disagree (8 of 16, dev seeds 1-7) are misrecalls: the brain returned the wrong word. Those rows sit in a
   damaged-store regime the live brain does not visit, and the probe never checks whether an answer is correct.
   Details and numbers are in section 3.

## 1. Seam-map rows verified against the code

"OK" means that file:line and the described operation match the source. Corrections are in the right-hand column.

| row | claim in the draft | check | correction / note |
|---|---|---|---|
| A5 | `what_does` → `composer.query_patient` "IS the spiking recall" | line OK; claim PARTLY wrong | `query_patient` (`one_brain_composer.py:1671-1695`) picks the answering block with a **host first-match loop** over spiking decodes. Default path: the DG-CA3 fact shard is on (`_FACT_SHARD_DEFAULT_ON = True`, line 363), so `_fact_shard_first_match` (1361-1375) walks the shard's blocks in ascending order and returns the first whose decoded agent/action equal the cue. `_seq_block` (1569-1580) is the fallback on an escalation, and it is the same first-match loop while `integrated_loop` is OFF (`webapp/server.py:3757`). The per-role word decode is spiking (`enable_spiking_cleanup` defaults on for onebrain, `brain_conversational_agent.py:248-249`), but the selection among stored facts is host. Dev seed 7: the `_seq_block` wrapper never fired on a default recall, which is how this path was found. **Missing row; now seam K-SELECT below.** |
| A9 | host `QuestionRouter.match_fact` fixes the candidate | OK (`brain_chat_tui.py:852`) | This is the plan's "host fixes the candidate set" pattern: the router picks one fact by keyword overlap, then the spiking recall only checks it exists (`recalled == p`, 857-868). Live only on self/identity and anaphora-miss turns. |
| A10 | "THE commitment seam" | OK (857-868) | On the default turn the covered class is authored by the three-organ bus (`gnw_three_organ_bus.py:320-385`), so A10 runs only on out-of-scope turns. It is not the main commitment seam of a default factual turn. |
| A11 | render + verify | OK (1642-1672) | `_verify` compares the re-parse to the gated SVO. It checks the render matches the host's committed fact, not that the fact answers the question. |
| B6 | topic = patient if it is an agent | OK (664-668) | `agents_set` comes from `self._stored_facts()` = `composer.kb` (host list), not a substrate read. |
| B7 | dedup + cap at `max_sentences` | OK (671-674) | Agree: positional cap is host. |
| B9 | corner turn `own[0]` | OK (503-520) | `_facts_about` reads `composer.kb` (host list, `rich_answer_composer.py:356-359, 421-428`). The draft calls this "the brain's own stored facts"; it is the host shadow list. |
| B10 | dlPFC rank "YES neural" | OK (540) | The concept ranking is spiking; each concept is then turned into a sentence by `_facts_for_concept`, a host lookup in the shadow list. The content is not read out of the substrate. |
| B11 | edge-weight fallback sort | OK (613-623) | Agree. |
| C6 | `candidate_patients` "a faithful read" | OK (`value_choice_production_organ.py:354-361`) | It iterates `chat.stored_facts`, the host shadow list. The candidate set is fixed by host code before the "neural" value choice runs. |
| C7 | engagement = recency + WM referent | OK (364-393) | Agree it is host. Stronger: the value critic reads V at a cue position computed ONLY from this host scalar (module docstring, lines 21-27), so the host ranking decides which candidate wins. **Seam V-CHOICE below.** |
| C9 | value-WTA "YES" | OK (441-445) | The WTA is spiking, but its drive order comes from C7. Also: `_stable_seed` (line 399) uses Python `hash()`, which is randomised per process unless `PYTHONHASHSEED` is set, so the salience seed is not stable across processes. |
| C10-C13 + stale-attribute finding | deliberation gate reads `chat._last_gnw_bus` | OK (`gnw_deliberation.py:179, 237`; `gnw_bus_shadow.py:521-533` is the only writer) | Confirmed in code: with three_organ authoring, bus_shadow's wrapper is not called, so the `authored_by == "bus"` check fails and the gate passes through (reason `out_of_scope_or_no_bus_info`). The new instrument records `chat._last_gnw_delib` on every turn, which gives the live check the draft asked for. |
| C14 | real-vocab membership = organ C veto authority | OK (`gnw_three_organ_bus.py:139-154, 191-195`) | Agree, and it matters more than the draft says: for every proposition whose words are known (every tiny-demo recall), organ C's vote is this host set-membership test; the spiking D4 margin is never read. **Seam ORGAN-C below.** |
| C17/C18 | vote assembly; consensus "YES" | OK (293-299) | The candidate list is `[cand, cand-if-B, cand-if-C]` plus a host-picked decoy: only ONE content candidate (organ A's recall) ever competes. Inside `norgan_hop` (`_gnw_norgan_bus_derisk.py:153-159`) the winner is `np.argmax(rates)` over occupied slots and commit is `rates[winner] >= THR`: a host argmax + threshold read-out of spiking rates. Correct to call it a read-out; it is not a selection among contents. |
| C19 | class dispatch | OK (320-385) | Agree. |
| rows 42, 43, 45 | affect plan → composer budget; reach; confidence cap | OK (`server.py:6407-6410, 6431-6434, 6456-6460`) | Agree. |
| row 35/36, E11 | metacog confidence = host arithmetic by default | OK (`server.py:6291-6305`; `metacog_production_organ.py` `mean_role_confidence`) | Agree. **Seam METACOG below.** The flag-on spiking margin was already tried as a production flip and recorded NO-GO (2026-09-05, overconfident on degraded recall; see the docstring at `metacog_production_organ.py:194-207`). The draft did not cite it. |
| row 38 | curiosity novelty is a host constant | OK (`server.py:6341`) | Agree. |
| D6 (disjoint) | episodic content = host first-match | OK (`server.py:5528-5539`) | Resolves the draft's "unsure": the loop walks `composer.kb` (host shadow list) in store order and renders the host record. It is not a neural ranking. |

**Rows the map should add (content, default turn):**

- **K-SELECT** — host first-match block selection (`_fact_shard_first_match` on the default path, `_seq_block` on an
  escalation; above). The spiking alternative exists in production (`BRAIN_INTEGRATED_LOOP=1`, the K-way sequencer,
  which bypasses the shard), default OFF because it over-abstains at the tiny-demo vocab.
- **Host shadow fact list as a candidate source** — `composer.kb` / `chat.stored_facts` feeds A9, B6, B9, B10, C6, D6.
  The D6 comment at `brain_chat_tui.py:772-774` states the problem: the kb list "records every heard fact whether or
  not a synapse changed". Any content picked from it is taught content, but the substrate did not produce it.

**What the map gets right and should keep:** the execution order, the default-on wrapper stack
(multistep → value_choice → deliberation → three_organ → [two_organ → bus_shadow]), the split between parses on
input (a declared boundary) and decisions over neural output, and the note that the moat abstains are host maps of an
empty neural verdict.

## 2. The probe (`_draft_host_share_probe.py`): what is sound, what is not

**Sound, and reused:** recording the host confidence (`_host_mrc`, which never prefers `margin_spiking`) and the
neural-derived one (`mean_role_confidence` with the flag on) from ONE trace, so both reads see the same neural state.
The new instrument reads the same two fields from the live chat trace.

**Not sound as a host-share measurement:**

1. **The intervention is true by construction.** `_confident(organ, m)` calls `MetacogProductionOrgan.judge(ev)`,
   whose only input is the host's evidence scalar. "Holding the trace fixed" does nothing, because `judge` never reads
   the trace. `host_share_indicator = 1.0` on 35/35 rows follows from the call graph. It is not a measurement. It also
   stops at the organ's boolean, not at the reply.
2. **It is not the live seam.** It builds `OneBrainComposer(seed, trace=True, vocab=VOCAB)` directly: D=128, 5 dev
   facts, no chat path, no three-organ bus, no rich composer. Its own docstring says so. This is fine for a
   mechanism screen, but it does not measure the production seam.
3. **Decoding statistic.** Pearson over 5 clean values per seed. The seed-4 "outlier" (r=0.062) is noise on 5 nearly
   equal points, not a divergence to follow up.
4. **The degraded panel is confounded** (section 3).

## 3. The owner's question: does the degraded panel confound the host-share reading?

**Yes.** What the panel does: it adds complex Gaussian noise (sigma 0-2.5) to every store synapse, re-queries the 5 dev
facts, and compares the host and spiking confidence reads.

I re-ran the draft's panel exactly (same code, same `default_rng(seed)` sequence, same restore; dev seeds 1-7; the
straddle counts match the draft's `dev_degraded_sweep.json` row for row). I added two things the draft does not
record: whether each answer is the taught patient, and whether the store's cached CSR was rebuilt.

| measurement (seeds 1-7 pooled) | value |
|---|---|
| answered rows across the sweep | 240 |
| answered rows whose answer is WRONG (not the taught patient) | 29 |
| rows where host and neural confidence disagree ("straddle") | 16 |
| straddle rows that are wrong answers | **8 of 16** |
| straddles where host hedges and the spiking read is confident | 13 of 16 |
| wrong answers read CONFIDENT by the host read / by the spiking read | 19 of 29 / 23 of 29 |

Examples of straddle rows (want → got): `sky → sink`, `hill → fly`, `pond → spikes`, `sky → spikes`, `sea → stone`. Under
this noise the recall returns another fact's word, and the probe still counts the row as "answered".

**How the confound works:**

- The noise does reach both the answer path and the confidence trace, so there is no split between an intact answer
  and a damaged trace (checked: the store CSR is never rebuilt during a level, yet answers change). Both confidence
  reads see the same damaged scores, so comparing them is internally fair.
- But the rows where the two reads disagree are the rows where the memory is failing. Half are misrecalls. What
  those rows show is "which read stays confident on a wrong answer" (mostly the spiking margin, 13/16). That is the
  already-recorded 2026-09-05 NO-GO (overconfident spiking margin on degraded recall), found again. It is not
  evidence about who owns the commitment.
- The regime is outside what the brain meets: every straddle is at sigma ≥ 1.5, where the answered rate is already
  falling (e.g. seed 1: 5, 4, 3, 0, 1, 4 answered across sigma 1.2-2.5). On the clean panel the two reads never
  disagree (0/35), so the panel shows "the substitution is consequential" only where recall is broken.
- A wider observation for the metacog seam (dev only): under damage, both reads call most misrecalls confident (host
  19/29, spiking 23/29). The margin measures how decisive the decode was, not whether it was correct.

**What to keep:** a perturbation that makes the two reads disagree is useful for SUBSTITUTION, but only on rows whose
answer is checked correct, and preferably with natural ambiguity (competing facts, interference) rather than
synthetic synapse noise. It cannot stand in for INTERVENTION.

Evidence: `research/runners/_host_share_draft_confound_check.py` (re-runs the draft panel from the draft branch and
adds correctness) and `research/findings/raw/_host_share/draft_confound_dev_s1-7.json`. DEV seeds only; not a verdict.

## 4. Live checks from the new instrument's dev runs (dev seed 7, numpy, not a verdict)

- **Deliberation gate (C10-C13) is inert on the traced turns.** `chat._last_gnw_delib.reason` read
  `out_of_scope_or_no_bus_info` on a teach turn and `already_abstained` on a competing-recall turn. It did not act on
  either, and read `out_of_scope_or_no_bus_info` on all three committed recall turns of the recall group (dev seed 7).
  The draft's stale-attribute reading holds live: the gate never acts on the default three-organ path.
- **On a competing recall, the three-organ consensus abstains and value choice supplies the content.** After "the dog
  chase the ball" (reconsolidation off), "what does the dog chase": organ A recalls `cat` (host first-match), organ B
  withholds (`consensus_veto_organ_b_withheld`), the bus abstains, and value choice commits `ball`, the more recent
  fact, from the host candidate list [cat, ball]. Reversing the host engagement order flipped the reply to `cat`.
  The winner on this turn is decided by the host recency ranking.
- **Deliberation, value choice and the three-organ bus all ran; the `_seq_block` wrapper never fired** on a default
  recall. That is how the fact-shard path (row A5) was found.

## 5. What the new instrument does differently

`research/runners/host_share.py` (this branch) measures six seams on the real `/api/brain-chat` handler: K-SELECT,
V-CHOICE, BG-SELECT (the speak/silent pre-filter that step 3 targets), ORGAN-C, METACOG and IGNITION (the consensus
read-out). It follows the load-bearing-fraction discipline: fresh brain per arm, numpy/CPU, stub renderer, no LLM, a
sham arm, and a held-fixed check. The three plan levels stay separate. Events are classified by INTERVENTION outcome
through one importable schema (`research/runners/host_share_schema.py`), so agreement between host and neural rules
is never scored as neural ownership. Seams whose host decision reaches the reply only through host code or a pure
function of it (organ-c, metacog, ignition) are reported as call-graph checks, not pooled. Preregistration:
`research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md`. Execution packet:
`docs/drafts/host-share-baselines/HANDOFF.md`.
