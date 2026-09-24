---
type: preregistration
status: preregistered
date: 2026-09-24
lane: load-bearing (open-ended conversation; plan step S16, lane A3)
mechanism: open-ended-gated-turn (BRAIN_OPEN_ENDED_GATED, default OFF) -- the open-ended turn routed through chat.gate (GNW 2/3-organ bus, deliberation, value-choice, multistep), a spiking BG speak/abstain race and the SETTLE-window affect-marker read
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any run it governs; no result is claimed here.
runner: research/runners/_open_ended_gated_turn_gate.py
note: the governed artifacts do not exist yet. The seed-7 dev smoke and the flag-off byte-identity check are NOT governed by this document (dev seed, integrity checks) and are reported in the lane's finding.
---

# Open-ended gated turn — pre-registration (2026-09-24)

## The standing verdict this design does NOT overturn

The 2026-09-23 amendment-3 NO-GO stands:
[`2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md`](2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md).
Seed 100 read UNDEFINED (the noise streams never changed a reply within an arm, so the null was degenerate).
The mean effect over the five DEFINED seeds was 0.09375, under the 0.10 floor
(`research/findings/raw/_load_bearing/_oe_production_turn/a3/default_a3_aggregate.json`, `mean_delta`).
Nothing below re-scores that harvest or reinterprets it. This is a NEW mechanism (the gated routing) measured by a NEW instrument.
If this gate reads GO, the finding must still say that the 2026-09-23 categorical a3 verdict on the ungated turn is NO-GO.

## What is being built (the thing under test)

`BRAIN_OPEN_ENDED_GATED=1` (default OFF; `webapp/open_ended_gated_turn.py`, one-line hooks in `webapp/server.py`).
With the flag on, every single-fact and rich turn keeps the ordinary pipeline and adds three things after the gate:

1. **Gate trace.** An outermost wrapper on `chat.gate` records, per turn, the fresh GNW verdict: routable,
   ignited, n_ignited, abstain reason, who authored it. The wrapped chain is the production one: the N-organ bus,
   the 2-organ bus, the 3-organ bus, deliberation, value-choice, and multistep.
   From the gate result the turn gets a `route`: `grounded` (a stored or acquired SVO committed), `hypothesis`
   (the spiking generative draw volunteered a HypothesisSVO), `withheld` (routable, but the workspace did not
   ignite or organ B vetoed), `grounded_ltm` (the gate abstained, but the long-term store holds facts on the topic),
   or `offkb` (none of these).
2. **A spiking speak/abstain decision.** A dedicated two-channel basal-ganglia selector
   (`bg_action_selection_production_organ.BGActionSelector`, the Gate-A v2 topology, its own bridge instance,
   `cfg.seed` = the brain seed) runs ONE race. Its two saliences come from brain reads:
   `speak = clip(fam + eng, 0, 1)` and `silent = clip(1 - fam - eng, 0, 1)`.
   Here `fam` is 1.0 for grounded or grounded_ltm, 0.75 for hypothesis, and 0.0 for withheld or offkb.
   `eng = min(1, |valence|)`, where `valence = clip(4 x differential, -1, 1)` and the differential is the live
   Gate-B spiking affect ladder read. The race picks SPEAK or STAY_SILENT, or no clean winner.
3. **The reply.** STAY_SILENT gives the hold line. SPEAK on `grounded`/`hypothesis` keeps the pipeline's answer.
   SPEAK on `grounded_ltm` renders the stored fact clause (`render_fact_sentence`) and passes it through the
   known-topic honesty post-filter. SPEAK on `offkb` builds a brain-state prompt from functional read-outs:
   the route, the familiarity band, the valence sign and the SETTLE-window marker register.
   It then generates with the warm Qwen when one is loaded, and runs the unknown-topic honesty post-filter plus a
   phenomenal-claim sentence filter. With no LLM (the battery and this gate), raw is "" and the post-filter returns its
   fixed honest hedge. No commit or `withheld` keeps the pipeline's answer unchanged.
   The SETTLE-window marker is `AffectMarkerWTA(settle=True)`, a separate reader instance; its input is the
   felt mood from affect-drives, and a neutral mood (level 0) is not sent to the circuit.

The row lesion `BRAIN_OPEN_ENDED_GATE_LESION=1` (only read when the flag is on) cuts the AFFERENTS from the
brain reads into the two new spiking competitions.
Both BG channels receive the same baseline salience (0.5, 0.5), and every marker pool receives the same baseline
current (the marker circuit's own `lesion=True` construction).
The organs themselves stay installed and are read as usual.
The lesion writes no measured field: the measured fields are what the two competitions commit when their input carries
no information.

## Declared host residuals (brain-based-only boundary)

- The read-to-salience transduction (`fam` table, `eng` formula, the clip) is HOST code. It is a declared residual of the same kind as
  `bg_action_selection_production_organ.salience()` (a token count) and A10's read-to-pA map.
- `route` is a host categorisation of the gate's result and trace. The decisions under it are spiking:
  the workspace ignition, organ B's corroboration, the draw, and the BG race.
- Topic extraction (`open_ended_chat.extract_topic`) and the long-term-store index lookup (`retrieve`) are host,
  as in the existing open-ended path.
- The BG commit is read by the de-risk's own first-crossing rule (host read-out of a spiking race). The marker
  winner is read by `np.argsort` plus a dead margin (host read-out, shortcut S1 of the SETTLE block).
- The prompt text, the honesty post-filter and the phenomenal-claim filter are host. Qwen is the FORM mouth.
  Qwen writing off-KB CONTENT is outside the ratified articulation-only role; it is the open owner fork S00(b).
  The gate below never loads Qwen (LLM disabled, stub renderer), so it measures decisions, not rendered text.

## Part A — the capability gate (the a3 successor on the gated turn)

**Amended by Amendment 3 (below):** the Part A verdict is the reply-level metric; CONT is a manipulation check.
**Reconciliation (added on this fix round):** the "GO (Part A)" rule stated further down this section (CONT-based)
is SUPERSEDED by Amendment 3, A3.1, and never governed a run -- it is kept verbatim, below, for the audit trail
only. So is the "Default-ON requires... the Part A GO" line under "What a GO would and would not mean": that phrase
means the A3.1 reply-level GO, not the CONT-based rule stated below. Read A3.1 for the rule that actually governs.

Runner: `research/runners/_open_ended_gated_turn_gate.py`, reusing `_lbf_open_ended_production_turn_probe`'s
session worker. It uses the same TEACH world, the same ASK ("what might a dog chase") and the same per-session
noise-stream installer.

- **Mode** `oe_gated`: `BRAIN_OPEN_ENDED_GATED=1`, nothing else changed (numpy, stub renderer, `SIM_DISABLE_LLM=1`).
- **Arms, per seed:** `intact` (`BRAIN_SPIKING_DRAW_LESION=0`), `lesion` (`BRAIN_SPIKING_DRAW_LESION=1`, the
  same host-weight-vector lesion as a3, declared as a HOST-vector lesion), and `intact_rebuild`
  (one session, the same noise stream as intact session 0).
- **Sub-streams:** M = 3 sessions per arm per seed. Each is a fresh process on its own OU noise stream,
  `seed*1000 + {intact:0, lesion:500} + j`. K = 8 asks per session. The session is the unit.
- **CAT (the original a3 categorical metric, reported beside):** session value = mean over asks of
  w(volunteered patient)/w_peak, with ABSTAIN and hold scored 0 (`session_value`, reused unchanged).
  Delta_cat(s) = mean intact - mean lesion. It is scored with the a3 UNDEFINED rules unchanged: degenerate null,
  draw not reached, lesion not applied, stored facts differ, and a non-reproducing rebuild.
- **CONT (the new continuous reply metric):** for each ask, take the spiking competition that produced its reply.
  That is the last `draw_from_weights` call of the ask whose winner equals the volunteered patient; its last
  non-silent `_compete` gives the firing vector fv over that call's candidate list.
  ask value = sum_p (fv_p / sum fv) x w_p / w_peak, the firing-share-weighted likelihood of the competition that
  chose the reply. w is the SAME reference vector CAT uses (`w_ref` = intact session 0's `likelihood_weight`).
  An ask whose reply is ABSTAIN or hold scores 0, as in CAT.
  Session value = mean over asks. Delta_cont(s) = mean intact - mean lesion.
  CONT is read one step upstream of the categorical argmax. It is a read of the reply-selecting spiking
  competition, not of rendered text.
- **CONT UNDEFINED rules (per seed; UNDEFINED is never a pass and never a 0):**
  - any session has no recorded competition on some ask that returned a hypothesis;
  - the null is degenerate: the M intact session values are all identical AND the M lesion session values are
    all identical;
  - the rebuild's per-ask CONT values differ from intact session 0's (not exactly equal);
  - the lesion was not applied: an ablated-draw fraction below 1 in the lesion arm or above 0 in intact;
  - the stored facts differ across sessions.
- **GO (Part A) -- SUPERSEDED by Amendment 3, A3.1; never governed a run (kept for the record):** all 6 seeds
  DEFINED on CONT, AND Delta_cont(s) >= 0.10 on EVERY seed.
  The exact one-sided sign test over the 6 Delta_cont then has p = 1/64 < 0.05, implied by the per-seed floor.
  Any UNDEFINED seed means NOT GO.
- **CAT is reported beside CONT on every seed, never dropped.** It carries its own a3-rule verdict: 6 seeds
  DEFINED, sign test p < 0.05, mean Delta_cat >= 0.10.
  If CONT reads GO and CAT does not, the headline must say both in one line.
  The reply-level categorical claim then stays NO-GO.
- Per-seed exact permutation p (C(6,3) = 20 splits) and `tools.lab.attributable_to` are descriptive, not gates.

## Part B — decision-level load-bearing rows (measured by the LBF instrument)

**Amended by Amendments 2 and 3 (below):** the gnw-drive probe turn, two rows' decision fields, and one row's status.

Module: `research/runners/lbf_rows/open_ended_gated.py` (`EXTRA_LESIONS` / `EXTRA_PROBES`, merged by the
AG-REG import hook).
The rows are measured in tag b2b-caps with `--extra-env BRAIN_OPEN_ENDED_GATED=1`, 6 seeds, using the LBF runner's own
intact / intact-rebuild null / lesion (repeats 2) design and its load-bearing rule.
At production defaults the flag is OFF, the `open_ended_gated` key is absent in both arms, and each row reads
not-exercised. That is reported as opt-in, never as not-load-bearing.

| row | lesion flag | kind | probe turn | decision fields |
|---|---|---|---|---|
| open-ended-turn-faculty-drive | BRAIN_OPEN_ENDED_GATE_LESION=1 | neural-lesion (afferent cut into the BG race + marker WTA) | `unknown` | `open_ended_gated.bg_action`, `open_ended_gated.reply_kind`, `open_ended_gated.marker_level` |
| open-ended-turn-affect-drive | BRAIN_AFFECT_LESION=1 | neural-lesion (Gate-B `affect_out` gate 0) | `emo` | `open_ended_gated.bg_action`, `open_ended_gated.reply_kind` |
| open-ended-turn-gnw-drive | BRAIN_GNW_2ORGAN_WS_LESION=1 | neural-lesion (workspace recurrence 0) | `sw_open` (Amendment 2: `chase`) | `open_ended_gated.route`, `open_ended_gated.bg_action`, `open_ended_gated.reply_kind` |

**Pass-by-construction audit (blocking; recorded in the row module).**
- No lesion writes a measured field. Each lesion sets an input current or a synaptic gate upstream of a spiking
  competition, and the measured field is that competition's committed output or a host label of it.
- `valence_sign` and `familiarity_band` are organ copies that the affect and GNW lesions set directly. They are carried as TRACE fields
  and are NOT decision fields of any row: counting them would be an integrity smoke.
- Each artifact records a cut assertion (`open_ended_gated.cut`). It records the saliences actually applied,
  whether they equal the baseline under the row lesion, the marker lesion flag, `affect.lesioned`, and the 2-organ
  `ws_lesion`. `tools.lab.lever` logs, at read time, that the applied saliences moved off the intact ones.

## Integrity checks (not gates; reported in the finding)

- Flag-off byte identity: the full probe roster, run through `webapp.server.brain_chat` with the flag unset, at
  this branch's head vs at its merge-base with origin/main. The response JSON must be identical.
- Seed-7 dev smoke: the conditioning state under intact, intact-rebuild, the gate lesion, the affect lesion and
  the GNW lesion.

## What a GO would and would not mean

- A Part A GO means that, on the gated turn, the host likelihood vector transmitted through the spiking draw moves
  the competition that selects the reply by at least 0.10 of the peak weight on every seed. It does not show that
  the spiking part is load-bearing (no host-oracle arm), and it does not measure rendered Qwen text.
- A Part B row reading load-bearing means that cutting that input changes the gated turn's committed decision
  fields (not the prose) on the tiny-demo brain, with LLM disabled.
- Default-ON requires all of: the Part A GO (Amendment 3, A3.1: the reply-level GO defined there, not the
  CONT-based rule stated above this section), a SOUND opus review, no fcap0924 drop, and the owner's yes on S00(b).
  Otherwise the flag stays opt-in.

## Sources

No external result is used as a threshold.
<!--derived-->
The external search logged for this lane (the GNW ignition access literature, e.g. Almeida 2022,
doi:10.1016/j.neuropsychologia.2022.108202, and the BG role in speech production, Krýže 2026,
doi:10.1002/ana.78276) motivates the design: report is gated by an ignition and a BG commit. It does not set any number here.

## Amendment 1 (2026-09-24, after the dev-seed-7 smoke, BEFORE any gate-seed run)

No gate seed (42/43/44/100/101/102) has run under this document. Everything below was decided on dev seeds 7, 11 and 13.

**What changes: the engagement afferent.** `eng` becomes the Gate-B affect organ's own graded register,
`|tone_level| / 3`, where `tone_level` is the staircase level the server already reads off the spiking ladder
differential (range -3..3).
It replaces `min(1, |clip(4 x differential)|)`. The fallback to the x4 squash applies only when no `tone_level` is attached.

**Why (calibration, disclosed as such).** The x4 squash was the Qwen-prompt mood mapping and was never calibrated
as a striatal salience.
- On the strongest affective probe (`emo`, seed 7) the Gate-B differential was 0.0375 and the tone level was 2
  (`research/findings/raw/_open_ended_gated/smoke/s7/intact_a.json`, `affect`).
  The x4 squash turned that into a speak salience of 0.15 (`open_ended_gated.salience_speak`, same file).
- The dev-seed psychometric read of the speak/abstain race
  (`research/findings/raw/_open_ended_gated/bg_curve/bg_curve_dev_s7_11_13.json`, `pooled`) gives
  SPEAK on 2 of 24 races at s = 0.15, against 21 of 24 at s = 0.67 (level 2 / 3).
- So under the original formula the affect afferent could not move the race by construction. The measured seed-7
  consequence: the affect lesion changed the emo turn's valence sign and saliences but not the BG action.

**What does NOT change.** The route table, the lesion constructions, the rows, the probe turns, the compared fields
and every Part A rule (M = 3, K = 8, the 0.10 per-seed floor, all 6 DEFINED) stay as registered.
The Part A ask ("what might a dog chase") is affectively neutral, so it is not expected to move.

**Expected single-shot row behaviour, stated before the gate seeds (from the same dev curve; descriptive).**
- `open-ended-turn-faculty-drive` on `unknown`: the intact race is at s = 0 (STAY_SILENT on 23 of 24 dev races) and the
  cut race at s = 0.5 (SPEAK on 12 of 24). One LBF build per arm is a single draw, so the row can read not-load-bearing on
  about half the seeds even if the afferent coupling is real. A 6/6 robust-core result is not expected for this row.
- `open-ended-turn-affect-drive` on `emo`: if the gate seeds' emo tone level is 2, the intact race sits at s = 0.67
  (21 of 24 SPEAK), and the lesioned race at s = 0 (0 of 24 SPEAK).
- `open-ended-turn-gnw-drive` on `sw_open`: the route change does not go through the race's noise.

**Added descriptive read (not a gate).** `python -m research.runners._open_ended_gated_turn_gate --bg-curve` at the
gate seeds gives P(SPEAK) at the intact and at the cut saliences over 8 races per point. It is reported beside the
single-shot rows, so that a row read as not-load-bearing can be told apart from an afferent that carries no information.

## Amendment 2 (2026-09-24, BEFORE any gate-seed run of Part A or Part B)

No gate seed (42/43/44/100/101/102) has run under this document. Everything below was decided on dev seed 7.

**What changes: the probe turn of `open-ended-turn-gnw-drive`, from `sw_open` to `chase`.** Same lesion, same
decision fields (`route`, `bg_action`, `reply_kind`), same scoring.

**Why (an integration defect, disclosed).** The AG-REG row hook merged on main (`338d9ecee`) parks any row whose
probe turn is not in `onebrain_regression_battery.PROBE_TURNS`. `sw_open` is a label-only turn, so as registered
the row would never enter the LBF registry, and Part B would measure 2 of its 3 rows.
`chase` ("what does the dog chase all the way") is in the default roster and asks about the same boot fact.

**Dev-seed evidence for the new turn (seed 7, not governed).** On `chase` the GNW workspace lesion moves the route
grounded -> withheld, the BG action SPEAK -> STAY_SILENT and the reply kind grounded -> withheld_abstain, with the
intact rebuild identical to intact
(`research/findings/raw/_open_ended_gated/smoke_gnwturn/s7/smoke_summary.json`, `changes_vs_intact.gnw_lesion.chase`).
That is the same change `sw_open` showed
(`research/findings/raw/_open_ended_gated/smoke_amend1/s7/smoke_summary.json`, `changes_vs_intact.gnw_lesion.sw_open`).
The other two default-roster candidates read were `question` and `open`. They route off-KB and hold in every arm, so
they cannot carry this row.

**What does NOT change.** Part A, the other two rows, every lesion construction, and every rule.

## Amendment 3 (2026-09-24, after the opus review of `7a39851c`, BEFORE any gate-seed run)

No gate seed (42/43/44/100/101/102) has run under this document. The only new input is the review.
The 42 Part A sessions queued on the pool (pinned to `49b8fb128`) were removed from `research/queue/pool.queue` under
its lock before any of them started. None of them was in `pool.queue.running`, `pool.queue.done` or `pool.queue.claims`.
The exact removed lines are in `research/findings/raw/_open_ended_gated/partA_queue_removed_2026-09-24.tsv`
(6 seeds x intact j = 0..2, lesion j = 0..2, intact_rebuild j = 0).
The corrected sessions are requeued only after this amendment is committed, at a revision that contains it.

### A3.1 Part A: what it tests, and what it does not

**The review's point, accepted: Part A does not exercise the gated routing.** On Part A's ask the gated turn passes the
pipeline's answer through. In the dev smoke every ask routed `hypothesis`, the BG race picked SPEAK, and no reply was
replaced (`research/findings/raw/_open_ended_gated/partA_smoke/oe_gated_s7_intact_n0.json`, `gated_trace`).
So Part A's replies are, by design, the a3 `default` path's replies with the flag on.
Part A is the a3 successor for the SPIKING DRAW under the gated flag. The gated routing is measured by Part B only.

**CONT is re-labelled a MANIPULATION CHECK. It can never produce a GO.** CONT weights the firing shares of the
reply-selecting competition by the same host vector w that the lesion replaces with ones.
A lesion arm whose firing follows its uniform drive therefore reads about mean(w)/peak, whatever the reply does.
Seed-7 smoke, per ask: lesion 0.327 / 0.347 / 0.339, uniform-firing value 0.367, intact 0.609 / 0.548 / 0.564
(recomputed with `session_cont` from the two `partA_smoke` files, `w_ref` = the intact session's `likelihood_weight`).
So CONT can clear 0.10 on a seed where the reply never moves (at a3 seed 100 both arms answered `deer` on all 32 asks).
It shows that the lesion reached the competition that selects the reply. Nothing more.

**The Part A verdict is the reply-level metric.** Per seed it is the registered CAT session value (mean over asks of
w(reply)/peak; ABSTAIN and hold score 0), scored by `score_seed_a3` unchanged.
A seed is DEFINED only when the a3 UNDEFINED rules pass AND the manipulation check is DEFINED.
**GO (Part A) needs ALL of:**
- exactly the registered seed set 42, 43, 44, 100, 101, 102 (any other set reads NOT-GO, whatever it shows);
- all 6 seeds DEFINED;
- the a3 rule: exact one-sided sign test p < 0.05 and mean Delta_reply >= 0.10;
- Delta_reply(s) >= 0.10 on EVERY seed (the per-seed floor Part A originally put on CONT);
- the manipulation check Delta_cont(s) >= 0.10 on every seed.

The aggregate writes one verdict string: `GO`, `NO-GO` (all 6 DEFINED, a rule not met), `NOT-GO (UNDEFINED)`, or
`NOT-GO (WRONG SEED SET)` (`summary.verdict`).

**Expected outcome, stated before the run.** The a3 noise streams `seed*1000 + {0, 500} + j` for j = 0..2 are the same
streams Part A uses. If the pass-through holds, Part A's replies should equal the a3 `default` sessions on them, up to
code drift since `eefdd666a`.
The a3 per-seed Delta read 0.031 / 0.021 / 0.010 / UNDEFINED / 0.146 / 0.260 for 42 / 43 / 44 / 100 / 101 / 102
(`research/findings/raw/_load_bearing/_oe_production_turn/a3/default_a3_aggregate.json`, `summary.delta`).
So Part A is expected to read NO-GO or NOT-GO (UNDEFINED). A GO would contradict the pass-through; it must be checked
against the reads below before anyone believes it.

**Two descriptive reads are added. Neither is a gate.**
- Pass-through, per seed and arm: counts of the recorded route, BG action and reply kind over the asks, and the number
  of asks whose reply the gated turn replaced (hold, conditioned generation, long-term-store clause).
- Reply identity against the a3 `default` session at the same (seed, arm, j): matching asks out of asks. Those sessions
  ran the ungated turn at another code revision, so a mismatch can come from code drift as well as from the gated turn.

**Why the review's other option (gating CONT intact vs the host_oracle arm) was not taken.**
- The host oracle has no spiking competition, so CONT is undefined in that arm. Comparing a firing-share read with a
  proportional probability vector compares two different quantities.
- The host oracle draws with the proposer's own `np.random.default_rng(_gen_seed)`, which is not on the per-session noise
  stream. Its M sessions per seed would be identical, so its null would be degenerate.

Doing it properly needs a new instrument (the host draw on the session stream) and its own dev smoke. It is recorded as
a follow-on and is not part of this amendment.

**A rule implemented as registered (the review's MINOR).** The CONT rule "the lesion was not applied" is a FRACTION: every
lesion session's ablated-draw fraction (`n_ablated_calls / n_calls`) must be exactly 1, and every intact session's
exactly 0. The previous code only checked for a count of 0, so a partially ablated lesion session passed.

**A scorer defect fixed (found by the review; not a design change).** This document already required all 6 seeds.
The code passed `n_required = len(seeds)`, so the requirement never bound: 3 synthetic seeds, or 6 dev seeds, read GO
through `score()`. Both now read `NOT-GO (WRONG SEED SET)`. The runner's `--selftest` now drives the `score()` path with
synthetic session files, including a case where the firing moves and the reply does not (it must read NO-GO).

### A3.2 Part B rows

**`open-ended-turn-faculty-drive`: `marker_level` is removed from the decision fields.** The decision fields are now
`bg_action` and `reply_kind`. Under the row lesion the marker WTA runs with its own `lesion=True`.
Its docstring says the dead-margin check then fails on (almost) every trial and returns None. So on an affective turn
the field changes by construction.
On the registered `unknown` turn it is 0 in every arm, because a neutral mood is never sent to the circuit.
It stays in the trace. The seed-7 "marker 2 -> none" on `emo` is that construction, not an effect.

**`open-ended-turn-faculty-drive` is declared NON-DISCRIMINATING and descriptive only.** The cut replaces the intact
saliences (0, 1) with (0.5, 0.5), so it adds speak drive. One LBF build per arm is one race.
On the dev curve the intact race held on 23 of 24 races and the cut race spoke on 12 of 24
(`research/findings/raw/_open_ended_gated/bg_curve/bg_curve_dev_s7_11_13.json`, `pooled`).
The row's verdict is therefore a per-seed coin flip, whatever the afferent coupling. Amendment 1 disclosed this.
**Revised by Amendment 4, below:** `DESCRIPTIVE_ONLY` alone was not enough -- `load_bearing_fraction.py`'s own
scoring never reads it, only this module's dev-smoke `score_row()` does -- so the row is now also `PARKED`, which
the registry hook honours unconditionally; it can never contribute to a load-bearing fraction number.
The distributional read beside it stays `--bg-curve` at the gate seeds.
A discriminating version needs the race read many times inside the turn's own conditions, for example a Part A arm
whose BG race runs on the per-session stream. It is recorded as a follow-on.

**`open-ended-turn-gnw-drive`: the decision field is now `route` only.** `bg_action` moves only through the host FAM
table (withheld -> fam 0), and `decide()` keeps the pipeline's answer on a withheld route whatever the race commits.
`reply_kind` withheld_abstain is a direct map of the route. So the three dev-seed diffs were one cause.
The reply change on `chase` ("the dog chases the cat ..." -> "I don't know about that. ...", both with the same curiosity
suffix; `research/findings/raw/_open_ended_gated/smoke_gnwturn/s7/smoke_summary.json`, `changes_vs_intact.gnw_lesion.chase.answer`)
is the pipeline's own gate abstain. `decide()` keeps the pipeline's answer on a withheld route, so the flag-on reply is
the one the pipeline produced before the gated turn ran. It is not credited to the gated turn.
A flag-off seed-7 read of `chase` under the same lesion is reported in the lane's finding.
What the row measures: whether the GNW workspace's ignition decides the gated turn's route label on a KB-hit turn.
On the off-KB turns (`question`, `open`, `unknown`, `emo`) the GNW lesion changed nothing at seed 7.
So the plan's S16 GNW criterion is met only on KB-hit turns.

**`open-ended-turn-affect-drive`: unchanged.** It shares `BRAIN_AFFECT_LESION` with the base `affect-coloring` row.
Any fraction that includes both counts one organ lesion, not two (`SHARED_LESION_WITH`).

**Launch condition (revised by Amendment 4, below): two rows, not three.** `open-ended-turn-affect-drive` and
`open-ended-turn-gnw-drive` are generated in a dedicated `tools/lb_shard.py` invocation with `--faculties` limited to
exactly these two keys (`lbf_rows.open_ended_gated.FACULTIES`).
`--extra-env BRAIN_OPEN_ENDED_GATED=1` applies to every job of an invocation. Any other row generated in the same
invocation would be measured with the gated turn on. `open-ended-turn-faculty-drive` is PARKED (Amendment 4): it
never enters the registry, gated flag or not, and is read only via `--bg-curve` and the dev-seed smoke's own
`score_row()`.

### A3.3 Flag-ON readiness (not a gate; the default stays OFF)

The BG selector and the marker reader are process-wide singletons keyed by seed.
Their races now run under one lock (`_RACE_LOCK`), so two requests cannot step one organ at once.
They are still shared across chat sessions: one session's race follows other sessions' earlier races.
Per-session organs are a precondition for any default-ON.
Owner decision (2026-09-24): Qwen stays fact-free. **That is the owner's statement; what follows is this lane's own
inference from it, not a separate owner ruling on this flag:** since an off-KB SPEAK still routes through Qwen
(A3.1 record), and Qwen fact-free rules out that path, this lane infers `BRAIN_OPEN_ENDED_GATED` stays default OFF
whatever Parts A and B read. The lane continues as a measured de-risk.

### A3.4 What does NOT change

The arms, M = 3, K = 8, the noise-stream seeds, the ask and the teach world, the 0.10 floor, the a3 UNDEFINED rules,
every Part B lesion construction and probe turn, and Amendments 1 and 2.

## Amendment 4 (2026-09-24, after a second review of `e79031483`; BEFORE any gate-seed run)

No gate seed (42/43/44/100/101/102) has run under this document. The review found four issues, none touching a
lesion construction, a probe turn or a threshold; all four are registry/documentation corrections.

**1) REQUIRED_ENV confirmed at both settings (main's convention, commit `eed3652a0`, merged into this branch).**
`research/runners/lbf_rows/__init__.py`'s registry-merge hook now honours the `REQUIRED_ENV` this module already
declared. Confirmed by direct import in a fresh process: with `BRAIN_OPEN_ENDED_GATED` unset, none of the three row
keys is in `load_bearing_fraction.FACULTY_LESIONS`; with it set to `1` before import, `open-ended-turn-affect-drive`
and `open-ended-turn-gnw-drive` are (see #2 for the third key). Measured with `tools/lb_shard.py jobs ...
--extra-env BRAIN_OPEN_ENDED_GATED=1 --faculties <rows>`, which sets the env in both arms of every shard.
The lane's finding said "the hook now merges all three rows ... none parked" describing the pre-`eed3652a0` state of
this same branch (the hook did not yet honour `REQUIRED_ENV`, so the rows merged unconditionally at that time); the
finding is corrected to describe the current, post-merge, env-conditioned behaviour.

**2) `open-ended-turn-faculty-drive` is now PARKED, not merely DESCRIPTIVE_ONLY.** The review's point: Part B is
actually measured by `load_bearing_fraction.py`'s own scoring over the merged registry, which does not read this
module's `DESCRIPTIVE_ONLY` dict -- only `score_row()`, the dev-seed smoke's own helper, does. So once
`REQUIRED_ENV` is met, the coin-flip row (#4 below) would enter the real headline load-bearing counts undetected.
The smallest honest fix, taken here: `research/runners/lbf_rows/open_ended_gated.py` now declares a module-level
`PARKED = {"open-ended-turn-faculty-drive": "..."}`, which the registry hook already honours unconditionally
(`research/runners/lbf_rows/__init__.py`, the same mechanism `live_organs.py`'s `self-schema` row uses) --
independent of `REQUIRED_ENV`, so the key never enters `FACULTY_LESIONS`/`FACULTY_PROBES` whether or not the flag
is set. `load_bearing_fraction.py` itself is unmodified. The row keeps its `EXTRA_LESIONS`/`EXTRA_PROBES` entries
(for `score_row()`'s dev-smoke use and the module's own audit trail) and its `DESCRIPTIVE_ONLY` entry (still read by
that same dev-smoke path); it is now measurable only as the `--bg-curve` descriptive read and via the dev-seed
smoke, never through a headline battery. `FACULTIES` (the b2b-caps `--faculties` list) drops to the two keys that
actually enter the registry: `open-ended-turn-affect-drive`, `open-ended-turn-gnw-drive`.

**3) Part A's CONT-based "GO" bullet and the "Default-ON requires... the Part A GO" line are reconciled with
Amendment 3, A3.1, inline, above (see the note opening the Part A section and the annotations on both bullets).**
Neither describes the rule that actually governs; A3.1's reply-level GO does. Nothing in A3.1 itself changes.

**4) Wrong number, fixed.** `lbf_rows.open_ended_gated.DESCRIPTIVE_ONLY["open-ended-turn-faculty-drive"]` read
"intact P(SPEAK) 1/24 at (0, 1)"; the pooled dev bg-curve at s = 0.0 is SPEAK 0 / STAY_SILENT 23 / none 1 of 24
(`research/findings/raw/_open_ended_gated/bg_curve/bg_curve_dev_s7_11_13.json`, `pooled["0.0"]`, re-checked on this
fix round) -- the module docstring's own "STAY_SILENT on 23/24 dev races" was already correct; only the
`DESCRIPTIVE_ONLY` reason string had the stray "1". Now reads "intact P(SPEAK) 0/24". The `PARKED` reason string
added in #2 states the full triple (0 / 23 / 1) so this cannot drift again unnoticed.

**What does NOT change.** Every lesion construction, every probe turn, the a3 standing verdict, Part A's actual
rule (A3.1), the two dev-seed findings' measured numbers other than the one string in #4, and the owner's fact-free
decision on Qwen (only its labelling as owner-statement-vs-lane-inference is corrected, inline, in A3.3 above).
