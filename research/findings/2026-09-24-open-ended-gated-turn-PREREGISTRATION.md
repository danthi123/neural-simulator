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
- **GO (Part A):** all 6 seeds DEFINED on CONT, AND Delta_cont(s) >= 0.10 on EVERY seed.
  The exact one-sided sign test over the 6 Delta_cont then has p = 1/64 < 0.05, implied by the per-seed floor.
  Any UNDEFINED seed means NOT GO.
- **CAT is reported beside CONT on every seed, never dropped.** It carries its own a3-rule verdict: 6 seeds
  DEFINED, sign test p < 0.05, mean Delta_cat >= 0.10.
  If CONT reads GO and CAT does not, the headline must say both in one line.
  The reply-level categorical claim then stays NO-GO.
- Per-seed exact permutation p (C(6,3) = 20 splits) and `tools.lab.attributable_to` are descriptive, not gates.

## Part B — decision-level load-bearing rows (measured by the LBF instrument)

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
- Default-ON requires all of: the Part A GO, a SOUND opus review, no fcap0924 drop, and the owner's yes on S00(b).
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
