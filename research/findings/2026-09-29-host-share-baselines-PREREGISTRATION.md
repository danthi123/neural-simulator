---
type: finding
status: live
date: 2026-09-29
lane: C-self-workspace
mechanism: PRE-REGISTRATION of the host-decision-share BASELINE (plan step 2, "prove who owns the computation") -- decoding / substitution / intervention at six content + commitment seams of the live /api/brain-chat turn (k-select, v-choice, bg-select, organ-c, metacog, ignition), classified by INTERVENTION outcome with one importable schema, a sham arm, a held-fixed check, and a learned-content fraction reported separately from the decision fractions
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any gate-seed arm was built. No result is claimed here.
runner: research/runners/host_share.py
artifacts:
  - research/runners/host_share_schema.py
  - docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md
  - docs/drafts/host-share-baselines/HANDOFF.md
  - research/findings/raw/_host_share/draft_confound_dev_s1-7.json
---

# Host-decision share baselines: PRE-REGISTRATION (2026-09-29)

**Committed alone, before any gate-seed (42/43/44/100/101/102) arm is built.** Branch
`research/host-share-baselines`. The gate runs at the commit that adds this file, pinned on the pool with
`tools/pool_provision.sh --revision <sha> --isolated`. Plan: `docs/plans/2026-09-25-prove-who-owns-the-computation-
PLAN.md` sections 1, 2 and 7 (step 2: record host-share baselines before any scaffold is retired). This revision
applies the 2026-09-29 mid-flight audit (`.claude/worktrees/_audit_2026-09-29/host-share.md`, 7 confirmed issues) and
the coordinator's four rulings.

This is a BASELINE, not a capability test. Its verdict is BASELINE-VALID or INVALID-OR-INCOMPLETE; the word GO is not
used. The per-seam numbers for bg-select and ignition are the "before" values for step 3 (the selection-seam build,
which imports `research/runners/host_share_schema.py` for its own before/after rather than keeping its own statistic).

## 1. The question and the three levels

At the content and commitment seams of one live chat turn, does the decision that reaches the reply come from the
spiking brain or from host Python? Three levels, never merged (plan section 2):

- **DECODING (screen only):** does the neural-derived decision equal the host decision at the seam?
- **SUBSTITUTION:** replace the host rule with the neural-derived rule on the target turn. Does a reply field
  DOWNSTREAM of the seam change?
- **INTERVENTION (primary):** hold the neural state at the seam fixed and set only the host decision to its
  preregistered alternative on the target turn. Does the reply's outcome at that seam follow the host?

## 2. The six seams (frozen; file:line on `main` f52c6d246)

| seam | category | host decision (default) | neural-derived alternative (substitution) | host alternative (intervention) | downstream neural path |
|---|---|---|---|---|---|
| k-select | content | which stored block answers (agent, action): host first-match over the spiking decodes of the DG-CA3 shard's blocks (`_fact_shard_first_match`, `one_brain_composer.py:1361-1375`; `_seq_block` 1574-1580 on an escalation) | the spiking K-way sequencer (`integrated_loop=True`, production code, default off) | >=2 matches: the LAST decoded match ("order"); 1 match: no match ("match") | YES (organ B / organ C / the consensus read the recalled candidate) |
| v-choice | content | the RAW engagement scalar per competing patient (recency ratio + WM-referent boost, `value_choice_production_organ.py:364-393`), then the shared spiking salience transduction, learned critic, spiking value-WTA | raw engagement = each candidate block's spiking decode margin (`margin_spiking`, patient role), 0 if the block does not decode to that candidate, min-max normalised | reversed raw engagement (1 - e), applied upstream of the spiking transduction | YES (transduction + critic + WTA) |
| bg-select | commitment | the host salience pair and the pre-filter `if not (silent_sal > speak_sal): return None` (`bg_action_selection_production_organ.py:295-297`), which skips the spiking BG race on every content turn | the spiking BG race run on the target turn at equal salience (0.5, 0.5) with no pre-filter | the host salience pair swapped (speak <-> silent) | YES (the race) |
| organ-c | commitment | organ C's vote for a known-word proposition: host set-membership (`gnw_three_organ_bus.py:191-195`) | the spiking D4 sel-pool margin read (the same function's unknown-word path) | the opposite vote | NO (a consensus needs all three votes) |
| metacog | commitment | host arithmetic over matched-filter margins -> evidence (`metacog_production_organ.mean_role_confidence` + `evidence_from_role_conf`) | host band arithmetic on the recall circuit's SPIKING margin (`margin_spiking`) | evidence mirrored about 0.5 (1 - ev) | NO (`judge(ev)` reads only the evidence) |
| ignition | commitment | the consensus read-out: `norgan_hop`'s `np.argmax(rates)` over occupied slots + `rates[winner] >= THR` | none in production (S UNDEFINED by design) | commit -> abstain; abstain -> commit organ A's candidate | NO (the read-out maps straight to the answer) |

Seams with no downstream neural path are **CALL-GRAPH CHECKS**: their F is fixed by the call graph, so it is reported
separately and never pooled into the panel. Every wrapper is an in-process monkeypatch in the arm's worker
subprocess; no `sim/` or `webapp/` file is edited. A manipulation acts only on target turns (and, for k-select,
v-choice, organ-c and ignition, only on calls for the target turn's (agent, action) cue).

## 3. Probe groups, turns, arms (frozen)

Real `webapp.server.brain_chat` handler through the regression battery's worker; `rich=True` (the production default
path); tiny-demo brain with the LTM tier pinned on (`BRAIN_LTM_SHIP_DEFAULT=1`, expected `source` = `tiny-demo +LTM`);
stub renderer; no LLM; numpy/CPU; one thread; `BRAIN_CHAT_SEED=<seed>`; `PYTHONHASHSEED=0`. Each group is ONE session
(one fresh ChatBrain) in its own fresh process per arm.

- **recall** (production defaults): "what does the dog chase" (target, dog/chase); "what does the cat eat" (target,
  cat/eat); "the wolf bites the apple" (setup: in-conversation teach); "what does the wolf bite" (target, wolf/bite).
  Seams: k-select, bg-select, organ-c, metacog, ignition. Arms: intact_a, intact_b + 2 per seam (ignition: intervene
  only) = 11.
- **compete** (`BRAIN_RECONSOLIDATION=0`): "the dog chase the ball" (setup); "what does the dog chase" (target);
  "the cat eat the worm" (setup); "what does the cat eat" (target). Seams: k-select, v-choice, ignition. Arms: 7.

18 fresh-brain arms per seed, 108 in total, one pool job each.

**Declared harness departures, identical in every arm incl. intact_a:** (a) `BRAIN_CHAT_SEED` is threaded into the
value-choice critic, the BG selector and the shared salience organ, which production builds at a hardcoded seed 42
(without it the six seeds would repeat one organ six times; logged in `research/FAILURE_LOG.md`); (b)
`PYTHONHASHSEED=0` (value choice's `_stable_seed` uses Python `hash()`); (c) compete group only,
`BRAIN_RECONSOLIDATION=0`: with reconsolidation on, a contradicting assertion rewrites the stored fact in place, so no
second patient can be taught through chat. It acts only on the two setup assertions.

**Declared consequence of one session per group:** a manipulation on an earlier target turn can carry into a later one
(referent, mood, DA). F and S are read on the seam's outcome and seam-downstream DECISION fields, not the full answer
text. Cost that forced this (dev seed 7, numpy): a build plus first turn ~25 min, a later turn ~5 min.

## 4. Per-event measures (frozen)

An EVENT is one seam x one target turn x one seed.

- **live:** intact_b recorded the seam on the target turn (k-select: a target-cue block selection with >=1 match;
  v-choice: >=2 candidates; bg-select: a salience read; organ-c: a vote on the target cue; metacog: a non-None
  evidence read; ignition: a consensus read-out on the target cue).
- **Seam outcome of a reply:** k-select / v-choice / organ-c / ignition = the committed patient (`recalled_svo[2]`) or
  ABSTAIN; bg-select = HOLD (the reply carries `bg_select`) or SPEAK; metacog = `metacog.confident`.
- **F:** 1 if the intervene arm's seam outcome equals the host alternative, 0 otherwise. UNDEFINED when the host
  alternative equals the intact outcome (not observable), or does not exist (v-choice reversed engagement with no
  unique top; metacog evidence exactly 0.5; no metacog verdict; ignition abstain with no organ-A candidate).
- **f_outcome (F = 0):** `returned_to_neural` if the reply's seam outcome equals the neural-derived decision's outcome
  and that differs from the host alternative; `ltm_fallback` if the reply was answered from the LTM tier; otherwise
  `masked`.
- **S:** 1 if a seam-downstream decision field differs from intact_a. Fields: `abstained`, `recalled_svo`,
  `supporting_facts`, `n_sentences`, `verified`, `metacog.confident`, `hedge` (the metacog hedge text is present),
  `hold`. For metacog only `metacog.confident`, `hedge`, `n_sentences`, `supporting_facts`; a difference in
  `abstained` / `recalled_svo` / `verified` there makes S UNDEFINED ("substitution side effect upstream").
- **D:** 1 if the neural-derived decision equals the host decision at the seam. UNDEFINED when the neural rule made
  no decision (value choice declined; the BG race did not commit). Ignition's D is 1 by construction and is labelled so
  (the plan's own example: the scores trivially predict their argmax).

## 5. The schema: buckets, classes, panel (frozen; `research/runners/host_share_schema.py`)

Buckets, by INTERVENTION outcome: `host_decided` (F=1, S=1), `host_causal_replaceable` (F=1, S=0),
`host_causal_s_undefined` (F=1, S undefined), `neural_decided` (F=0 and `returned_to_neural`), `masked` (F=0
otherwise), `undefined` (F undefined). **An F = 1 event is never counted as neural.**

Per seam (live events): HS = mean F, NS = share neural_decided, MS = share masked (each over events with F defined), SC
= mean S, DA = mean D. Class: **HOST-OWNED** if HS >= 0.8 (written "HOST-OWNED (replaceable)" if SC < 0.2);
**NEURAL-OWNED** if NS >= 0.8; **MASKED** if MS >= 0.8; **MIXED** otherwise; **UNDEFINED** if live on fewer than 4
of 6 seeds or no F defined. k-select is also reported per sub-kind (order / match), organ-c per sub-kind (known /
unknown word).

Panel, over the MEASURED seams only (k-select, v-choice, bg-select): host_decision_share = #(F=1) / #(F defined);
neural_decision_fraction = #neural_decided / #(F defined); masked_fraction likewise; replaceable_share_of_host =
#host_causal_replaceable / #(F=1). The call-graph seams get the same panel, reported separately.

Content (separate from the decision fractions; intact_b target turns with an asserted fact): per asserted fact, who
selected it. `substrate` = a spiking recall (`query_patient`) returned that exact triple this turn; `host_list` = it
came from the host's shadow fact list (`_facts_about` / `_facts_mentioning`) or value choice's host candidate list;
else `unattributed`. **learned_content_fraction** = substrate share; **python_selected_fraction** = host_list share.
The lexical share of fact words vs template words is reported as `fact_word_share`, a wording measure only.

## 6. Validity bars (frozen) and the verdict

- **V1 sham + determinism:** intact_a == intact_b on the full answer text and every decision field of every turn, on
  >= 5 of 6 seeds. A failing turn's events are UNDEFINED.
- **V2 liveness:** reported per seam; fewer than 4 live seeds makes the seam UNDEFINED.
- **V3 intervention applied:** every live event's intervene arm recorded `applied=True`; else that event is UNDEFINED.
- **V4 both outcomes registrable:** at least one F = 1 event, and the sham clean on at least one seed.
- **V5 state held fixed:** in each intervene and substitute arm, the seam's recorded pre-decision state (k-select
  match list; v-choice candidates + raw engagement; bg-select salience pair; organ-c candidate + known flag; metacog
  role margins; ignition candidates + rates) equals intact_b's; else the event is UNDEFINED ("seam state drifted").
- **V6 same brain:** every response in every arm and seed reports `source` = `tiny-demo +LTM`.
- `PYTHONHASHSEED=0` and the seed threading confirmed in every arm; no shard missing.

Verdict: **BASELINE-VALID** if V1, V3, V4, V6, the hash/seed checks hold and no shard is missing; else
**INVALID-OR-INCOMPLETE** (numbers reported and labelled, not used as a baseline). Scored only by
`python -m research.runners.host_share --aggregate research/findings/raw/_host_share/gate --seeds 42,43,44,100,101,102`.

## 7. UNDEFINED handling

A missing or errored arm, a failed sham on that turn, an intervention not applied, drifted seam state, a seam not
live, a host alternative that does not exist or coincides with the intact outcome, a substitution side effect upstream
of the seam, or a neural rule that made no decision makes that measure UNDEFINED with the reason recorded. UNDEFINED
is never counted as 0 or 1. A figure over zero defined events is UNDEFINED.

## 8. Predictions written before the gate (expectations, not bars)

- **v-choice:** HOST-OWNED. Dev seed 7: reversing the engagement flipped the committed patient (ball -> cat). The
  neural-margin alternative may or may not agree (SC unknown).
- **k-select:** "match" events HOST-OWNED (nothing downstream re-derives a fact the host declared absent). "order"
  events in the compete group mostly UNDEFINED, because value choice re-commits the recency winner and the host
  alternative (the last match) equals the intact reply. Descriptively (dev seed 7), the consensus rejected the host's
  first-match pick (organ B withheld `cat`) and accepted the flipped pick (`ball`): a neural veto that the reply
  cannot show.
- **bg-select:** HOST-OWNED: with the saliences swapped the race commits STAY-SILENT (the race follows the host
  salience bias, per its own 2026-08 de-risk). D probably UNDEFINED or 0 at equal salience.
- **organ-c, metacog, ignition (call-graph checks):** HS near 1 wherever an alternative is observable.
- **Content:** learned_content_fraction below 1; the compete turns' committed facts are python_selected.

## 9. What was seen on dev seed 7 before this was written (numpy, not a verdict)

- Sham clean with every wrapper installed (answer text identical to intact_a, compete group first two turns).
- On "what does the dog chase" after "the dog chase the ball": organ A recalled `cat` (host first-match), organ B
  withheld, the consensus abstained, the deliberation gate did not act (`already_abstained`), and value choice
  committed `ball` from the host candidate list [cat, ball]. Reversing the host engagement committed `cat`.
- The k-select order flip (organ A recalls `ball`) made the consensus commit `ball` (3 votes); the reply was
  unchanged (`ball`).
- The `_seq_block` wrapper never fired on a default recall; the production selection path is the fact shard.
- Section 10 lists the verification run on the final code.

## 10. Dev verification on the final code (dev seed 7)

Run 2026-09-29 14:17-14:41 on the final wrapper code (numpy; shards in `research/findings/raw/_host_share/dev/s7/`).
PASS on the HANDOFF section 2 list:

- `research/findings/raw/_host_share/dev/s7/compete__v-choice_substitute.json`: seams recorded bg-select, ignition,
  k-select, metacog, organ-c, v-choice (+ the `_recall` / `_hostlist` recorders); applied v-choice; sources
  `tiny-demo +LTM`; no errored turns; seeds_used 7/7/7.
- `research/findings/raw/_host_share/dev/s7/recall__bg-select_intervene.json`: applied bg-select; sources
  `tiny-demo +LTM`; no errored turns; seeds_used 7/7/7.
- `research/findings/raw/_host_share/dev/s7/recall__intact_b.json`: seams recorded bg-select, ignition, k-select
  (fact-shard path), metacog, organ-c; nothing applied (sham); sources `tiny-demo +LTM`; no errored turns;
  seeds_used 7/7/7.

Observed (dev, not a verdict): swapping the host salience pair turned all three recall targets into the HOLD reply;
the neural-margin engagement ranked the same candidates first as host recency (ball over cat, worm over fish); the
consensus read-out committed with slot rates [0.33, 0, 0, 0]; host evidence was saturated at 1.0 on every recall
target; the deliberation gate reported `out_of_scope_or_no_bus_info` on every committed recall turn. Build + first turn
~10-23 min per arm under load.

## 11. What this does not claim

Six seams on a five-fact tiny-demo brain with the LTM tier, one probe set, stub renderer. Not the Qwen mouth, not the
single-fact path, not the other seams in the map, not a developed brain. HOST-OWNED says who decides there today, not
whether a neural mechanism could. Agreement between the host rule and the neural rule (F=1, S=0) is reported as
host-causal and replaceable, never as neural ownership (`docs/TERMS.md`: not "fully spiking").

## 12. Amendments

Any change after the first gate job is dispatched is appended here, dated, before the aggregate is read.
