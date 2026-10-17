---
type: preregistration
status: preregistered
date: 2026-10-14
mechanism: bg-live-seam-retirement, amendment 1 (the spiking commit read)
amends: research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION AMENDMENT only. No gate seed of the commit build has run when this is committed.
runner: research/runners/host_share_bg_seam.py
---

# Amendment 1 to the BG live-seam retirement: G4 on the spiking commit read

**Filed before any gate seed (42/43/44/100/101/102) of the commit build has run.** The 2026-10-02 run of the 09-29
build stays its record: NO-GO on G4 only (`research/findings/2026-10-02-bg-live-seam-retirement-RESULT.md`; the
fixed-rate latch read changed its committed action on 0.25 of events at threshold x0.5 / x2). This amendment
registers the next method for the same capability: the spiking commit stage (`docs/drafts/spiking-commit/PLAN.md`
section 3) on the BG channels. Worklog with the design check and the DEV table: `docs/drafts/bg-commit/WORKLOG.md`.
Handoff with the exact commands: `docs/drafts/bg-commit/HANDOFF.md`.

bar-sanity: docs/drafts/bg-commit/bar_sanity_record_bg_commit_g4.json

## What is unchanged

The question, the turns (`hsb_empty1` "...", `hsb_q` "what does the dog chase", `hsb_empty2` "?", `hsb_hi` "hi"),
the seven arms and their environments except as stated below, the scorer's 09-29 measures, the bars of G1, G2a, G2b,
G3 and G5, every 09-29 precondition (the sham null on every target event; the membrane in [-150, +100] mV on every
neural-seam trial; at least 6 events per gated measure) and the construction guards of `tests/test_bg_neural_seam.py`.

## What changes

1. **Build.** The three seam arms (`seam_b`, `seam_readout`, `seam_lesion_value`) add `BRAIN_BG_COMMIT=1`
   (`--commit` on the runner; the prod arms are untouched). Organ region `bg-commit`
   (`research/runners/bg_action_selection_production_organ.py`): on the neural-seam organ's own bridge, per channel
   k, motor_k -> commit gate G_k (12 excitatory, self-recurrent, transmission gate `bg_commit_in`), G_k -> GI_k
   (8 inhibitory) -| G_j (gate `bg_commit_excl`), G_k -> latch S_k (20 excitatory, self-holding S_k -> S_k, gate
   `bg_commit_latch`), and the single-slot exclusion GI_k -| S_j (a later commitment replaces an earlier one);
   operating point `BG_COMMIT_GRID[4]` (`BG_COMMIT_OP = 4`), in full: NG 12, NGI 8, NS 20, W_XG 30, W_GG 20,
   W_GGI 60, W_GIG 60, W_GS 12, W_SS 60, W_GIS 60, W_RESET 16 (selector_reset -> G and S, density 0.70, GABA_A),
   READ_STEPS 30, READ_FRAC 0.03; every stage pathway at density 1.0, no jitter, frozen; AMPA excitation. **Frozen at
   the commit that adds this amendment; any change to this row, to `BG_COMMIT_OP`, or to the trial schedule voids the
   amendment** (pinned by `tests/test_bg_commit.py::test_frozen_row_4`, and by the precondition in item 5). The
   shared wiring and read live in `research/runners/spiking_commit.py`. The 09-29 latch stays built and its read is
   recorded from the same trial, never used.
2. **The commit read.** Over the last 30 ms of the 200 ms zero-stimulus hold, S_k is ON iff its spike count reaches
   READ_FRAC x 20 x 30 (READ_FRAC 0.03: 18 spikes). Exactly one ON -> that action; none -> no commit (the turn falls
   through, as before); both -> co-commit (no commit, counted). The same spikes are re-read at READ_FRAC x0.5 and x2.
3. **Organ seed.** Under the flag the organ seed follows `BRAIN_CHAT_SEED`. In the 10-02 run the server built the
   organ at seed 42 for every chat seed, so the first two turns read identical latch rates on all six seeds; the six
   seeds of this run are six organ builds. Flag off, nothing changes (organ seed 42).
4. **G4, amended.** Measure: the fraction of `seam_b` neural-seam events whose committed action under the commit read
   changes at READ_FRAC x0.5 or x2, or that carry no commit record. **Bar: 0 (no event changes).** The 09-29 G4
   measure of the old latch read (now against the old read's own choice) is reported, not gated.
5. **New preconditions (UNDEFINED, never NO-GO).** (a) Every neural-seam event of the three seam arms carries a
   commit record, in shards run with `BRAIN_BG_COMMIT=1` (`commit_engaged` = 1.0). Its "names the event's committed
   action" clause is presence-only: the organ copies the commit winner into the event's winner
   (`bg_action_selection_production_organ.py:546`), so it cannot fail when the record is present. (b) Exactly one
   neural-seam event per seam-arm turn: 12 x 6 = 72 (`commit_event_count_ok`). A turn whose stage raised leaves no
   event, because `webapp/server.py:5699` swallows the exception and the turn falls through to SPEAK; that turn makes
   the run UNDEFINED instead of passing G2b. (c) `commit.op == 4` on every event. (d) `commit.organ_seed` equals the
   shard's chat seed on every event. (e) G4 has at least 6 events.
6. **Scorer.** `host_share_bg_seam.score(..., commit=True)` and `verdict` with `BARS_AMENDED`; the per-event
   measures are `commit_event_measures` and the bar is `g4_commit_holds`. Without `--commit` the scorer reproduces the
   committed 10-02 score exactly (`tests/test_bg_commit.py::test_scorer_without_commit_reproduces_the_1002_score`),
   and it refuses (ValueError) shards that carry this amendment's id or `BRAIN_BG_COMMIT=1`: a commit run is never
   scored as a 09-29 run.

## Operating point (dev seeds 1-8 only)

`research/runners/_bg_commit_derisk.py`, raw `research/findings/raw/_bg_commit/dev_organ/`, aggregates
`research/findings/raw/_bg_commit/dev_organ/summary.json`, selection `research/findings/raw/_bg_commit/dev_organ/selection.json`
(organ level; per row and seed: the live four-turn sequence plus 12 trials per message class on one warm organ, and
12 trials on each lesion organ). Every row reads invariantly on 608/608 trials, intact and lesion (latched at least
65 spikes, unlatched at most 1); the commit and latch lesions commit nothing in 96/96 trials each, in every row.

| row | change from A1's row 0 | empty held | content held | co-commit | agrees with the old read | seeds all_pass |
|---|---|---|---|---|---|---|
| 0 | none (latches coexist) | 77/112 | 0/104 | 96 | 0.7375 | 3/8 |
| 1 | W_SGI 60 (first commitment keeps it) | 94/112 | 8/104 | 7 | 0.7781 | 2/8 |
| 2 | W_SGI 60, W_XG 30 | 95/112 | 8/104 | 6 | 0.7812 | 2/8 |
| 3 | W_GIS 60 (a later commitment replaces it) | 98/112 | 7/104 | 0 | 0.9563 | 1/8 |
| 4 | W_GIS 60, W_XG 30 | 98/112 | 7/104 | 0 | 0.9625 | 1/8 |

The old latch read on the same trials: empty 99/112, content 6/104. **Declared deviation.** The rule written before
the sweep (the first row with all_pass on all eight seeds) selected NONE, because its capability checks test the race
rather than the read: the old latch read on the same trials fails them on 5 of 8 seeds. Row 4 was chosen after the
data by a rule declared here: among rows whose stage checks (read invariance, the commit and latch lesions, membrane
and no carry-over) hold on all eight seeds, the one whose committed action agrees most often with the old read on the
same trials (`selection.json`). The choice was made on dev seeds only; rows 3 and 4 differ from the old read by one
trial in each capability count. **Declared plainly:** this rule favoured the row that agrees most with the old latch
read, and that read had already passed G2a, G2b and G3 on the 10-02 gate data; the selection therefore leans toward
the decisions that passed those gates once.

**Declared plainly (seed 42).** Under the flag the organ seed follows the chat seed, so chat seed 42 builds the organ
at seed 42, and that is the organ all six 10-02 gate seeds used (the server's fixed organ seed). Its race on these
four turns was already seen in the 10-02 gate data; only chat seeds 43, 44, 100, 101 and 102 are organ builds no
gate run has seen.

## DEV expectations (organ level, row 4; stated before any gate seed)

The commit read changes at x0.5 / x2 on 0 of 608 trials. Empty turns held 98/112 (0.875); content turns held 7/104
(0.0673), none of them the live question: on the live sequence `hsb_q` is SPEAK on 8/8 dev seeds
(`research/findings/raw/_bg_commit/dev_organ/summary.json`). With six independent
organ seeds, G2b (0 of 6 question turns held) therefore carries a real chance of failing on race noise alone (about
one in three if the per-trial content rate applied to the question; the live question was never held on dev seeds).
The value lesion with the stage on still commits (0.5-1.0 per class) but no longer by class (held 0.25-0.75 empty,
0.0-0.75 content), so about half the target events should change under the lesion (G3's bar is 0.3). Live dev
canary (seed 7, all seven arms, `research/findings/raw/_bg_live_seam/commit_dev/`): commit engaged on 12/12
neural-seam events, the commit read invariant on 4/4, empties held 2/2, the question not held, the old read agreeing
4/4; the value lesion changed the reply on 1 of 4 events (coin flips matching the intact turn), so G3 is the second
open risk after G2b. Latency: the stage adds 80 neurons and per-step spike counting; the trial schedule is unchanged.

## Reported, not gated (amended run)

The old latch read's choice, its x0.5 / x2 changes, and its agreement with the commit read (`old_read_agree`);
co-commits; the production first-to-12 read's disagreement; lane C's class, S and D; `seam_readout` F; the one-word
hold rate. Organ level (dev only, in the worklog): the exclusion lesion's commits and co-commits; the value lesion.

## Outcome -> verdict

| preconditions | gates | verdict | sentence |
|---|---|---|---|
| hold | G1, G2a, G2b, G3, G4 (amended), G5 all pass | GO | Behind the flags, the live seam's admission, score and commit are neural: the commit is a spiking latch whose read does not change at half or twice its level, and the bus argmax does no work on these turns. |
| hold | G4 (amended) fails | NO-GO | The commit stage's read is not threshold-invariant on the live turn; the read still owns part of the commit. |
| hold | G2a or G2b fails, G4 passes | NO-GO | The commit read is invariant, but the seam's race does not hold empty turns or spares no question; the capability, not the read, fails. |
| hold | G3 fails | NO-GO | The reply does not depend on the value route through the commit stage often enough. |
| hold | G1 or G5 fails | NO-GO | Unchanged from 09-29 (the production host does not own the reply / the bus argmax does work). |
| any fails | - | UNDEFINED | No verdict; `commit stage engaged` failing means the build did not run the stage (do not re-run; NEEDS-CLAUDE). |

A GO does not mean: that the stage or the value routes are learned (they are wired; learned-content fraction 0); that
any flag is on by default; that the organ is on the one-brain substrate (it is its own bridge). Residual host steps
(declared): the word rendering, the trial schedule (drive, hold, read window), the latch read itself (a threshold
whose invariance is the measured bar), and the env flags.

## Run and score (verbatim; the pool runs one seed per line)

```bash
cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 \
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.host_share_bg_seam \
  --run-all research/findings/raw/_bg_live_seam/commit_gate/<P7> --seed <s> --allow-gate-seeds --commit
.venv/bin/python -m research.runners.host_share_bg_seam --score research/findings/raw/_bg_live_seam/commit_gate/<P7> \
  --seeds 42 43 44 100 101 102 --commit
```
