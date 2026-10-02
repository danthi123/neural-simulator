---
type: preregistration
status: preregistration
date: 2026-10-02
packet: stage1-host-share
runner: research/runners/stage1_host_share.py
scorer: research/runners/stage1_host_share.py --aggregate
seeds: [42, 43, 44, 100, 101, 102]
---
# Stage-1 host share: who owns the decision at the C2/C3 seams, by intervention (PRE-REGISTRATION, 2026-10-02)

Written by the cloud Claude session 3 before any gate-seed run, committed alone. HANDOFF:
`docs/drafts/stage1-host-share/HANDOFF.md` (runbook task T22). Owner ruling `host-share-packet` (runbook section 6),
built early with the owner's approval. It answers the Stage-1 exit HANDOFF's NEEDS-CLAUDE #10 ("run the host-share
instrument's intervention arms on the C2/C3 probe turns; report F/S/D beside the criterion verdicts") and the Stage-1
prereg's section 5, which reported the seam audit without computing F/S/D because the instrument was not on main.

This is a MEASUREMENT, not a capability test. Its verdict is MEASUREMENT-VALID or INVALID-OR-INCOMPLETE; the word GO
is not used. It never changes a Stage-1 criterion verdict. It writes one credit line per criterion that the Stage-1
exit report prints beside that criterion's verdict.

## 1. The question

The Stage-1 gate asks whether the integrated brain recalls a new word (C2) and a lived fact (C3) at the reply level,
with a lesion arm per faculty. A PASS there says the faculty is load-bearing. It does not say who chose the answer.
The plan (`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` section 2) asks: hold the neural state at a
seam fixed, set only the host decision to an alternative, and see whether the reply follows the host. That is what
this packet measures, on the exact turns the Stage-1 gate scores.

## 2. What is imported unchanged

Everything below comes from the host-share baseline (`research/runners/host_share.py`,
`research/runners/host_share_schema.py`, prereg `research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md`
sections 2, 4, 5 and 7, merged to this branch from `research/host-share-baselines`): the seam table (SEAM_SPEC), every
wrapper, the per-event measures live / F / f_outcome / S / D, the buckets, the per-seam classes (HOST-OWNED at HS >=
0.8, NEURAL-OWNED at NS >= 0.8, MASKED at MS >= 0.8, MIXED otherwise, UNDEFINED below 4 live seeds), the panel, the
content fractions and the UNDEFINED rules. Nothing is re-derived. An F = 1 event is never counted as neural.

Seams measured here: k-select (content), bg-select (commitment), both MEASURED (a downstream neural path); organ-c,
metacog and ignition, CALL-GRAPH CHECKS (reported, never pooled into the measured panel). v-choice is not measured:
no Stage-1 C2/C3 turn has two competing stored patients for one cue, so it could never be live.

## 3. Probe groups, turns, arms (frozen in `stage1_host_share.py` S1_GROUPS)

Each group replays one Stage-1 session turn for turn (`research/runners/stage1_exit_gate.py` GROUPS, intact turns,
gate revision `c238ff325`). Labels carry a prefix (`s1w_`, `s1l_`); the rest is the Stage-1 label.

| group | criterion | session (messages in order) | target turns (cue) |
|---|---|---|---|
| `s1_word` | C2 | what does the bear chase; the otter eats the clam; the bear chases the otter; what does the bear chase; what does it eat; what does the otter eat; what does the otter chase; what does the clam eat; what does the dog chase | p_bear_chase (bear, chase), p_otter_eat (otter, eat), c_otter_chase (otter, chase), c_clam_eat (clam, eat), c_dog_chase (dog, chase) |
| `s1_lived` | C3 | what does the wolf bite; did we discuss the wolf; the wolf bites the apple; how is the weather today; what is the weather like tomorrow; what does the cat eat; what does the wolf bite; did we discuss the wolf; what does the wolf chase; what does the dog chase | p_wolf_bite (wolf, bite), c_wolf_chase (wolf, chase), c_dog_chase (dog, chase) |

Not targets: the pre-exposure probes, the teach turns, the distractors, the anaphora probe (no named agent; Stage-1
reports it, does not count it) and the disclosure probe `did we discuss the wolf` (section 7). They still run: they
are part of the session the target turns depend on.

Arms per group: intact_a (production, no wrapper), intact_b (sham: every wrapper observes), k-select / bg-select /
organ-c / metacog each `:intervene` and `:substitute`, ignition `:intervene` = 11. Two groups x 11 arms x 6 seeds =
132 pool jobs, each one fresh brain in its own process (one session per group, as the baseline). Gate seeds 42, 43,
44, 100, 101, 102. Dev seed 7 (22 jobs) first, section 8.

Turn resolution (new here): each call is matched to its label by its POSITION in the session and the message must
match, because both sessions repeat a message (the pre-exposure probe and the later probe are the same text). The
baseline's (session, message) lookup would have labelled both turns as the later one. A call that does not resolve is
recorded in `unresolved_turns`; any such shard makes the measurement INVALID-OR-INCOMPLETE.

## 4. Harness (frozen) and its declared departures

Real `webapp.server.brain_chat` handler through the regression battery's worker, in-process, as the baseline; tiny-demo
brain with the LTM tier on (`BRAIN_LTM_SHIP_DEFAULT=1`, expected `source` = `tiny-demo +LTM`); stub renderer; no LLM;
numpy/CPU; one BLAS thread; `BRAIN_CHAT_SEED=<seed>`; `PYTHONHASHSEED=0`. Group `s1_lived` sets `BRAIN_EPISODIC_STORE=1`
in every arm, exactly as the Stage-1 gate's `lived` group (its declared departure (a)).

Declared departures from the Stage-1 gate, identical in every arm including intact_a:

- (a) `rich=True` in the request. The Stage-1 gate omits `rich`; an omitted `rich` takes `_brain_rich_default()`,
  which is ON unless `BRAIN_RICH` is set, and no arm sets it. The two paths are the same code path.
- (b) Seed threading: `BRAIN_CHAT_SEED` is threaded into the value-choice critic, the BG selector and the shared
  salience organ (the baseline's departure (a)). The Stage-1 gate leaves them at production's hardcoded seed 42. So
  this packet's intact_a is a same-turn, same-brain replay, not a byte copy of the Stage-1 gate's intact_a. The
  agreement table (section 6) reports, descriptively, how often the two give the same committed outcome.
- (c) The corpus: every arm records `corpus_present` (`data/corpus/tinystories.txt` at the revision root). The
  corpus-learned production organs degrade silently without it (Stage-1 prereg section 3 (d)); the baseline
  instrument does not record it.

No `sim/` or `webapp/` file is edited. Every wrapper is an in-process monkeypatch in the arm's worker.

## 5. Validity (frozen) and the verdict

V1 to V6 as the baseline prereg section 6 (sham and determinism on at least 5 of 6 seeds; liveness reported per seam;
every live intervention applied; at least one F = 1 event; seam state held fixed; every reply from `tiny-demo +LTM`),
plus:

- **V7 corpus:** `corpus_present` is true in every arm of every seed.
- **Turn resolution:** no shard has an unresolved turn.
- PYTHONHASHSEED=0 and the seed threading confirmed in every arm; no shard missing.

**MEASUREMENT-VALID** if all hold; else **INVALID-OR-INCOMPLETE** (numbers reported and labelled, the credit line
reads UNDEFINED). Scored only by
`python -m research.runners.stage1_host_share --aggregate research/findings/raw/_stage1_host_share/gate --seeds 42,43,44,100,101,102`.
Exit code 0 when MEASUREMENT-VALID, 3 otherwise.

## 6. What is reported, per criterion (frozen)

For C2 (`s1_word`) and C3 (`s1_lived`) separately: per seam the class, HS, NS, MS, SC, DA, n events with F and live
seeds (over that group's target turns only); the measured panel and the call-graph panel; a per-target-turn table of F
per seed per seam; the content fractions (pooled); the sham reply-noise attribution of the criterion's host share.

The credit line (frozen; `credit_line()` in the runner), from the five seam classes of that criterion:

| seam classes | credit line |
|---|---|
| at least one seam HOST-OWNED or HOST-OWNED (replaceable) | `PARTIAL (host decision: <those seams>)` |
| none host-owned, at least one MIXED | `PARTIAL (host share mixed: <those seams>)` |
| none host-owned or mixed, at least one defined | `NO HOST-OWNED SEAM among the defined seams (<seams>)` |
| every seam UNDEFINED | `UNDEFINED (no seam defined)` |

The Stage-1 exit report writes the line beside the criterion verdict, in the RULING `iw2-stage1-credit` form. A
criterion that PASSES Stage-1 with a PARTIAL line here counts as PARTIAL toward the stage exit, as RULING
`t18-credit` treats a host-gated write: the host still decides. A seam classed UNDEFINED is never read as neural.
With `--stage1-dir <the Stage-1 gate shard dir>`, the aggregate adds a descriptive agreement table (committed outcome
on each target turn, this harness's intact_a against the Stage-1 gate's intact_a). It has no bar.

## 7. Out of scope here, with the next method for each (no-defer)

- **The C3 disclosure** (`did we discuss the wolf`): host routing (`is_referential` regex + `extract_referent`) feeds
  the spiking dAP completion. The instrument has no wrapper for it. Next method: a `disclose-route` seam whose
  intervention routes the turn as non-referential (or to a never-discussed topic token) with the dAP state held fixed.
- **C1** (`cat_disc`: the generative route's `_plausible` admission and novelty exclusion) and **C4** (the
  `appraise_text` salience gate and the `feel_readout` composition): candidate admission and scoring seams, also with
  no wrapper yet. Next method: one wrapper per seam in the same schema, then this packet's groups extended.
- **The store side** (`_maybe_acquire` SVO extraction and `lemma_verb`, host parsing that decides what is stored):
  a parse seam; the substrate-parse work (T20 word-role-parse) is the method that moves it.

These are named so the Stage-1 report does not read the C2/C3 lines as covering them.

## 8. Dev step (dev seed 7, no verdict)

22 jobs, the same lines at seed 7 into `research/findings/raw/_stage1_host_share/dev`. Then
`python -m research.runners.stage1_host_share --devcheck research/findings/raw/_stage1_host_share/dev --seeds 7`,
which checks only mechanics: no shard missing, no errored or unresolved turn, every reply from `tiny-demo +LTM`, the
corpus in every arm, PYTHONHASHSEED and seed threading, sham clean, every live intervention applied. It prints the
live-seed count per criterion and seam, reported only. A DEVCHECK FAILED stops the packet (a harness defect,
Claude's). Liveness on the dev seed never changes a bar, a turn, an arm or a seed, and the gate runs whatever the dev
liveness was.

## 9. Predictions (written before the gate; expectations, not bars)

- **k-select:** live on the counted recall probes (p_bear_chase, p_otter_eat, p_wolf_bite, c_dog_chase) as "match"
  events, HOST-OWNED (the host declares the fact absent and nothing downstream re-derives it). Not live on the moats
  (no block matches), so those events are UNDEFINED.
- **bg-select:** HOST-OWNED on every live target turn (the swapped salience pair turns the reply into the HOLD reply,
  as on the baseline's dev seed 7).
- **organ-c:** on the new-word cues (bear, otter) the sub-kind reads `unknown` if the recruited word is not in the
  brain's vocabulary at vote time; HS near 1 regardless (a call-graph check).
- **metacog, ignition:** HS near 1 wherever an alternative is observable; ignition UNDEFINED on the moats (abstain
  with no organ-A candidate).
- **Credit lines:** C2 `PARTIAL (host decision: k-select, bg-select, ...)` and C3 likewise.
- **Agreement with the Stage-1 intact_a:** high but below 100% on the turns the seed-threaded organs touch.

## 10. Cost

132 jobs at mem_gb=7 (memcap 9). Per-arm time from the Stage-1 gate's measurements: `word` about 21 min, `lived`
about 65 min (the forced episodic write), plus the wrappers' observation cost. About 95 job-hours, 25 to 50 h of
wall time on pool41 and pool42 once the lines reach the head of the queue, behind the Stage-1 gate's own 204.

## 11. Amendments

Any change after the first gate job is dispatched is appended here, dated, before the aggregate is read.
