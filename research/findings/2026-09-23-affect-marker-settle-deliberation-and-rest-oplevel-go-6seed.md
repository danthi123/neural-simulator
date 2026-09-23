---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
verdict: GO (circuit-level de-risk, 6-seed); the full-brain ON-vs-OFF contrast verdict is PENDING (re-staged on the pool in the fix round)
---

# Affect-marker: two truncated companion processes (deliberation time, inter-turn rest) caused the dead-zone; restoring them makes the WTA commit on 6/6 seeds at circuit level (2026-09-23)

D1 lane ("grow the robust core past 23"). This is a circuit-level de-risk GO. It does NOT move the robust core.

**What a full-brain GO would and would not mean (fix round, docs/TERMS.md).** `BRAIN_AFFECT_MARKER_SETTLE` is
default-off. A GO on the staged full-brain contrast gate would show that affect-marker is lesion-load-bearing under the
adequate probe with an opt-in, default-off flag. It would not make the faculty on-by-default or production-default, and
it would not grow the production-default robust core. Reported Option-C style, as a pair: the adequate-probe robust core
stays 23/26 at the shipped flag state, and would read 24/26 with SETTLE opt-in. Flipping the default is owner-reserved
and would need its own re-verify against the flipped code. The earlier "robust core 23 to 24" wording is withdrawn.

## Which faculties are outside the robust core (read from the artifacts)
The exercised adequate-probe roster has 26 faculties
(`research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s*.json`). The robust core is 23:
the 20 that were 6/6 there, plus episodic-memory, source-provenance-honesty and prospective-memory (each later made
6/6). That leaves three outside:
- **affect-marker-spiking-wta**: load-bearing on 1/6 seeds under the 2026-09-22 diagnosis read
  (`research/findings/raw/_lbf_borderline_isolated/op_s*.json`). This build targets it. Correction (fix round): that
  read was labelled "RNG-isolated", but it runs the intact and the lesion read on one cached warm reader with OU noise
  off. The per-seed confound it carries is warm-state carry-over between reads, not an RNG draw. The inter-turn rest
  below is the process that addresses it. The code comment in `_lbf_borderline_operating_point.py` is corrected.
- **open-ended-generation**: 0/6 on the single-turn ruler. It is load-bearing distributionally
  (`research/findings/raw/_load_bearing/_followon2_openended_distributional_6seed.json`), but that ruler measures
  a synthetic `_followon2` world, not the production turn. It is not counted here.
- **da-gated-encoding**: 0/6. Its only flip needed a tuned read-damage operating point, which was rejected.

Ranked by how tractable a principled fix is: affect-marker ranks first. Its failure is an intact circuit that does
not commit, which makes it a single-circuit problem with a clean lesion. Open-ended-generation needs a production
ruler, not a mechanism. Da-gated-encoding has no natural drive yet.

## The defect, read off the substrate
<!--derived-->
Source: research/findings/raw/_lbf_borderline_isolated/op_s42.json .. op_s102.json (ladder.mood, intact.margin); boundary = midpoint of MOOD_CENTERS[4:6].
On the 'emo' turn the felt mood is about +0.064 to +0.071, which sits on the +2/+3 register boundary at +0.07125.
On 5 of 6 seeds the INTACT winner-minus-runner-up margin is about 0.033 to 0.044, below `DEAD_MARGIN`=0.05. The
intact circuit therefore emits no marker, the lesion also emits none, and the faculty reads not load-bearing.

We asked what the real circuit runs alongside this that the code had replaced with a constant. Reading the circuit
found two such constants. Neither one is part of the competition's own wiring. Both were first seen in exploratory
probes before the gate was registered, and are reproduced as a committed artifact by `--mechanism-probe`
(research/findings/raw/_affect_marker_settle/mechanism_probe.json):
<!--derived-->
Source: research/findings/raw/_affect_marker_settle/mechanism_probe.json.
1. **Deliberation time (`WARMUP_STEPS`=60 ms).** When two inputs are nearly equal, a lateral-inhibition race resolves
   slowly. Decision time grows as the evidence difference shrinks (Roitman & Shadlen 2002, LIP; Wang 2002, the
   recurrent-inhibition decision circuit). On calibration seed 7, with the mood exactly on the +2/+3 boundary and the
   circuit unchanged: after 60 ms both pools were still firing (+2 pool 0.055556, +3 pool 0.019444, margin 0.036111,
   no winner). After 300 ms the race had resolved (0.1375 vs 0.0). After 500 ms it was 0.131944 vs 0.0. The circuit
   could always choose; the read stopped it before the race finished.
2. **Inter-turn rest (`WASHOUT_STEPS`=40 ms).** The reader stays warm across turns, and the circuit only advances
   while it is being read. Between two turns it relaxed for only 40 ms. On seed 42 at mood +0.0682, three
   consecutive reads on one warm reader gave these margins:
   - 40 ms washout: 0.113889, 0.035417, 0.115278;
   - 200 ms: 0.113889, 0.045139, 0.045139;
   - 1000 ms and 3000 ms: 0.113889 on all three reads.
   The previous read's slow state degrades the next read unless the circuit rests. The brain does not freeze between
   utterances.

## The fix (additive, DEFAULT-OFF `BRAIN_AFFECT_MARKER_SETTLE=1`, no sim/ edit)
`research/runners/_affect_marker_wta_derisk.py`: with the flag set, a read deliberates for `DELIBERATION_MS`=500
and rests for `INTERTURN_REST_MS`=1000 before it starts. Wiring, weights, tuning, drive and `DEAD_MARGIN` are
unchanged. What SETTLE adds is only the clock, meaning how long the circuit runs or rests. The competition that picks
the marker is the spiking race read off `cp_firing_states`. A warm reader is cached under the key `(seed, "settle")`,
so it never mixes with the OFF reader. `load_bearing_fraction` adds the report key `affect_marker_settle_env`, which
records whether the flag reached the run.

## Named host shortcuts on this path (declared in the fix round)
The build round said "no host formula chooses the marker" and "the host sets only the clock". Both overstate it. Two
pre-existing host steps sit on either side of the spiking competition, and SETTLE leaves both in place:
- **S1, readout.** `AffectMarkerWTA._select` names the winner with `np.argsort` over the pools' spike rates plus the
  host `DEAD_MARGIN` threshold on winner minus runner-up. That is an argmax over spike counts, which CLAUDE.md names as
  a shortcut. After a resolved 500 ms race the losing pools sit at rate 0, so the argsort is benign here. The
  replacement target is a downstream spiking read-out pool driven by the marker assemblies.
- **S2, drive.** The felt mood is a neural read of the #81 ladder, but the host turns that float into a Gaussian-tuned
  current per pool (`DRIVE_BASE_PA + DRIVE_GAIN_PA * exp(-(v - c)^2 / 2 sigma^2)`). No synapses from the ladder's
  populations do this. The replacement target is a synaptic projection from the ladder's V+/V- and arousal pools.
- S3, the deliberation and rest durations, is the host clock. This build added it.
- S4 is pre-existing and unchanged: the upstream host `mood_to_level` binning, the emphasis fallback `felt > 0`, and
  the host rendering the winning register's fixed word.
These are declared in the runner docstring and in the SETTLE block of `_affect_marker_wta_derisk.py`.

## Calibration of DELIBERATION_MS, including a disclosed amendment
The criterion was pre-registered in commit a4891aa3e before any calibration ran. It was checked on calibration seeds
7 to 11, which are disjoint from the verification seeds. It has four parts: C1, commit at every register boundary;
C2, choose the correct register at every center; C3, no clean winner under the lesion (baseline-only drive); C4, a
repeated read gives the same answer.
<!--derived-->
Source: research/findings/raw/_affect_marker_settle/calibration.json.
**As registered, no window from 60 to 800 ms passed.** C1 failed in two places:
- at the mood=0 midpoint between the -1 and +1 registers, on 3 or more seeds at every window;
- at the arousal boundary on seed 10, at every window.

The mood=0 point can never reach this circuit in production. In `webapp.affect_drives_chat`, any |mood| below
`_MOOD_L1`=0.010 <!--derived--> becomes level 0, and the neutral gate returns before the circuit is called. The arousal boundary
sets only the emphasis punctuation, not whether a marker appears.

So the criterion was amended after the fact, and the amendment is recorded in the runner and the artifact. The
amended C1' requires a commit at every REACHABLE valence boundary; the arousal-boundary commit is reported, not
gated. The shortest window that passes C1' + C2 + C3 + C4 on all 5 calibration seeds is **500 ms**. The arousal
boundary committed on 4/5 seeds at 300 ms and above.

## Result: circuit-level 6-seed gate (`--verify`)
<!--derived-->
Source: research/findings/raw/_affect_marker_settle/oplevel_verify.json.
All reads use the real ladder mood from the 'emo' turn, through the same `_lbf_borderline_operating_point._affect_marker`
path the committed diagnosis used.
| seed | mood | OFF lead intact/lesion (margin) | SETTLE lead intact/lesion (margin) | load-bearing OFF -> ON |
|---|---|---|---|---|
| 42 | +0.0682 | '' / '' (0.044) | 'Gladly! ' / '' (0.147) | no -> yes |
| 43 | +0.0710 | '' / '' (0.033) | 'Gladly! ' / '' (0.137) | no -> yes |
| 44 | +0.0703 | '' / '' (0.040) | 'Gladly! ' / '' (0.142) | no -> yes |
| 100 | +0.0639 | 'Gladly! ' / '' (0.160) | 'Gladly! ' / '' (0.158) | yes -> yes |
| 101 | +0.0685 | '' / '' (0.038) | 'Gladly! ' / '' (0.138) | no -> yes |
| 102 | +0.0691 | '' / '' (0.040) | 'Gladly! ' / '' (0.146) | no -> yes |

The gate result is **GO** (`tools.verdict.Verdict`, every precondition measured):
- G1: load-bearing with SETTLE on 6/6 seeds, against 1/6 with it off.
- G2: the lesion removes the marker on 6/6; it does not just pick a different one.
- G3: two fresh readers give the same intact lead on 6/6.
- G4: the shuffled drive changes the register on 4/6, exactly at the pre-registered floor of 4.
- G5: with SETTLE off, the intact and lesion reads equal the committed diagnosis dicts exactly (lead, level and float
  margin) on 6/6. In the build round the code compared only the boolean; the fix round makes G5 assert exact dict
  equality and re-ran `--verify`. Every per-seed read and count is unchanged, and G5 is still 6/6 under the exact test.
- Fix-round verdict structure: G1 and G4 are outcomes, so a failure now reads NO-GO. The full seed set, G2, G3 and G5
  are validity checks, so a failure reads UNDEFINED. The re-run still reads GO.

The intact margins with SETTLE on are 0.137 to 0.158, about 3 times `DEAD_MARGIN`.

**Attribution (G6, reported, not gated).** Each arm was run separately:
- deliberation alone is load-bearing on 6/6 seeds;
- rest alone is load-bearing on 5/6 seeds; it fails on s43, which sits exactly on the boundary, with intact margin
  0.0097;
- with neither, 1/6 seeds.
Most of the flip comes from deliberation time. The rest restores reads that the previous read's state had degraded.

## Honest residuals
- **The circuit and the host binning disagree at the boundary.** On s43 and s44 the upstream host `mood_to_level`
  bins the mood as level 3 (mood ≥ 0.070 <!--derived-->), but the settled circuit picks register +2 ('Gladly'). This is the
  documented residual #2 of the 2026-08-28 finding: the topographic boundary (0.07125) is not the host bin edge
  (0.070 <!--derived-->). The marker is still driven by the felt state; it just carves the axis slightly differently from the host
  table.
- **The C1 criterion was amended after the fact** (see above). As registered, calibration had no passing window.
- **Arousal boundary**: the circuit does not commit at the exact arousal boundary on 1 of 5 calibration seeds, at
  any window. There, emphasis falls back to the pre-existing host `felt > 0` rule. That rule is a named host
  shortcut, and this build leaves it unchanged.
- **Latency**: each marker read costs about 1.5 s of simulated time (500 + 1000 + 60 ms). On the numpy backend
  that is sub-second of wall time for a 216-neuron circuit.
- **Scope**: this is a circuit-level de-risk. The opt-in claim (24/26 with SETTLE on) waits on the staged full-brain
  contrast gate. The production-default robust core does not move either way until the owner flips the default.
- **Pre-registration timing**: commit a4891aa3e registered `DELIBERATION_MS`=300. The calibration and the verify then
  ran in the same second from a tree that already held the C1' amendment, so the verify did not wait for the
  calibration that fixed the constant. The review re-checked this independently: at op level, 200 ms and 300 ms also
  give load-bearing 6/6 with rest 1000 ms, so the choice of 500 ms was not selected by the outcome.

## Full-brain smoke, 1 seed (local, numpy, memcap 16): load-bearing at s42 (not a headline)
<!--derived-->
Source: research/findings/raw/_affect_marker_settle/lbf/lbf_settle_s42.json.
Command: `BRAIN_AFFECT_MARKER_SETTLE=1 load_bearing_fraction --only affect-marker-spiking-wta --seed 42 --repeats 2`.
It read `load_bearing=True` (`affect_drives.lead` 'Gladly! ' -> ''), `null_control_clean=True`,
`lesion_reproduced=True`, `deterministic=True`, and `affect_marker_settle_env`='1'. In the adequate 6-seed battery,
s42 read `pass` (not load-bearing).

This is one seed; it counts toward nothing until the 6-seed run lands. It is also a mixed-revision artifact: it
started at 95d9d39ce, and its later worker arms (intact_b and both lesion arms) ran at 56e588d16 after a mid-run merge
that changed `research/runners/_episodic_dap_dialogue_memory.py`. Its `deterministic=True` therefore compares arms built
on different code. Read it as an integrity smoke, not as determinism evidence. Environment caveat, now logged in
FAILURE_LOG: the worktree has no `data/corpus/tinystories.txt`, so the onebrain XEDGE build degraded to standalone
organs. The pool revision dirs have the same gap. The affect-marker path (AffectDrivesWorkspace) does not use that
corpus, and the ON and OFF pool arms share the same environment.

## Staged: the full-brain ON-vs-OFF contrast (fix round, re-staged)
**Superseded staging (build round, revision 56e588d16).** The ON and OFF arms shared one `--out` dir. The
`load_bearing_fraction` intermediate arm files (`intact_a_emo[_sS].json`, `lesion_affect_marker_spiking_wta[_sS].json`)
do not name their arm, so an OFF run could overwrite an ON run's raw arms on the same node. The gate also required only
that OFF rows carry no flag, so it could credit SETTLE even if the OFF control were load-bearing too. In the fix round
the remaining running and queued jobs were killed by PID. The six rows that had finished (ON s42, s43, s44, s101; OFF
s42, s43) were copied unread to `research/findings/raw/_affect_marker_settle/lbf_superseded_rev56e588d/` and do not
enter the verdict.
<!--derived-->
Source: research/findings/raw/_affect_marker_settle/lbf_superseded_rev56e588d/pool41/lbf_settle_on_s42.json and siblings.
Read only after the contrast gate was committed, as an informational cross-check:
- ON s42, s43 and s101 read load-bearing, null-clean, lesion-reproduced and deterministic, with the flag set.
- ON s44 is `arm-build-failed`, so `load_bearing` is null and the row is UNDEFINED. pool41 had 0 GB available at the
  time.
- OFF s42 and s43 read not load-bearing, valid and deterministic, so the contrast holds on those two seeds.
- On s43 the full-brain ON lead is 'Wonderful! ' (register +3), while the op-level read picked 'Gladly! ' (+2). The
  cause was not investigated; a different ladder mood in the full-brain turn is the likely candidate, since s43 sits
  closest to the boundary. Either register counts as load-bearing.
These rows mix raw arm files from different arms in one dir and are one revision behind, so they are not evidence
for the gate.

**Re-staged layout.** Each seed runs as one pool job on one node: the ON arm, then the OFF arm, from the same revision,
serialized by a per-node lock (each full brain is about 8 GB on a 15 GB node). Every arm and seed has its own
directory: `research/findings/raw/_affect_marker_settle/lbf_on/<seed>/lbf_settle_on_<seed>.json` and
`.../lbf_off/<seed>/lbf_settle_off_<seed>.json`. The harvest copies each node into its own subdirectory
(`lbf_on/<node>/<seed>/`), so no two files can collide.
Queued 2026-09-23 at pool revision `56f1abf54` (prereg `ac02c209d` merged with main), one job per seed, provisioned on
pool41 and pool42 with `data/corpus` present. Harvest, for each node N in pool41 and pool42:
`rsync -a N:derisk-pool/revisions/56f1abf54917f271c061815a8ba886ee61fd99f2/research/findings/raw/_affect_marker_settle/lbf_on/ research/findings/raw/_affect_marker_settle/lbf_on/N/`
and the same for `lbf_off/`. Then run the scoring command below.

**The contrast gate (pre-registered in the fix round, `_affect_marker_settle_derisk.py --score-fullbrain`).** The
rule and the amendment log are in the runner docstring, committed before any re-staged row existed.
- SETTLE is credited on a seed only if the ON row is load-bearing, null-clean, lesion-reproduced, deterministic and
  carries `affect_marker_settle_env`=="1", and the same-seed OFF row is a valid measurement that reads not
  load-bearing. A valid OFF row has no flag, is deterministic, has a clean null control and a non-null `load_bearing`.
- **GO**: ON load-bearing on 6/6 and OFF load-bearing on at most 1/6. The allowance of one is the op-level OFF reading
  known before the gate was written (s100 in `oplevel_verify.json`).
- **PARTIAL, a NO-GO for the flag**: ON 6/6 but OFF load-bearing on 2 to 5 seeds. That is reported as load-bearing 6/6
  with the flag, SETTLE-attributable on k/6 only.
- **NO-GO**: ON below 6/6, or OFF load-bearing on 6/6, which makes the flag unattributable.
- **UNDEFINED**: any row missing, ambiguous, non-deterministic, carrying the wrong flag state, or with an unmeasured
  `load_bearing`. UNDEFINED is never a pass.
The selftest drives the scorer in every failing direction (11 cases, `--selftest`).
Scoring: `.venv/bin/python -m research.runners._affect_marker_settle_derisk --score-fullbrain`.
It writes `research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json`.
A GO means load-bearing under the adequate probe with a default-off flag, not a production-default robust core.

## Honesty boundary
This is a functional read-out only. The expression marker is chosen by a spiking competition driven by the
interoceptive felt-state read. No claim of felt or phenomenal experience is made.
