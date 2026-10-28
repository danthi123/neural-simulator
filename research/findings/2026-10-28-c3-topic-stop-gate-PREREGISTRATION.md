---
type: preregistration
status: preregistration
date: 2026-10-28
packet: c3-topic-stop
runner: research/runners/stage2_exit_gate.py (group ask, sixteen arms per seed)
scorer: research/runners/c3_topic_stop_score.py --mode gate
---
# C3 per-topic stop (L+P) at the six gate seeds (PREREGISTRATION, 2026-10-28)

host-decision: scaffold
decides: whether the curiosity follow-up question is asked, through the crave-gain formula clip(0.2, 3.0, 1 + 1.5 (da_eff - 0.5)) applied to the topic-stop ASK-pool want (P method B), the rate-to-da_eff normalization and the two linear afferent encodings; and, for L, which topic slice a topic takes when all 11 slices are in use (least recently presented)
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement)
replaced-by: a per-pathway dopamine synaptic-gain target on the topic -> ASK synapses (a sim/neuromodulators.py scope extension), so the SNc dopamine scales the topic afferents on the substrate and no host formula scales the want
cap: 2 builds

flip: no this gate qualifies L+P for a default-ON flip and flips no default itself; the flip is a separate, later flip prereg or commit that carries its own smoke-record lines (tools/smoke_first.py), made only after a PASS here
bar-sanity: docs/drafts/c3-topic-stop-gate/bar_sanity_record_c3ts_gate.json

Written by Claude on 2026-10-28, before any gate-seed run of this method. No gate seed (42, 43, 44, 100, 101, 102) has been
run with `BRAIN_CURIOSITY_TOPIC_STOP` or `BRAIN_CURIOSITY_PHASIC_DA` set. The header above is the packet's own scaffold
declaration, copied unchanged (`research/coordination/c3-topic-stop-PACKET.md`), so this file joins the packet's scaffold
group; it is the group's first prereg (1 build of the cap of 2).

## 1 The decision this gate makes

Candidate C = L+P = `BRAIN_CURIOSITY_TOPIC_STOP=1 BRAIN_CURIOSITY_PHASIC_DA=1`, merged default OFF at `79c6f5128`
(packet sections 3 and 4). The dev battery of packet section 7 (192 shards, dev seeds 1-12, seed 9 the design seed and never
counted; `research/findings/raw/_c3_topic_stop/dev/`, all at `79c6f5128`) passed D1 to D8 on 11 of 11 counted seeds, and an
independent skeptic re-score agreed. By the packet's own rule ("every bar must hold for the method to go to a gate-seed
preregistration") that is what this file is.

The gate decides one thing: whether L+P **qualifies for a default-ON flip before the 11-21 Stage-2 re-pin**. A PASS does not
flip anything. The flip is a separate, later step with its own prereg or commit (smoke-first records, the escape value, the
default reader), and that step is where `BRAIN_CURIOSITY_TOPIC_STOP` and `BRAIN_CURIOSITY_PHASIC_DA` would become default ON.

## 2 Bars, fixed before any gate-seed data

Each gate bar G1 to G8 is the packet's D1 to D8 (section 7), word for word in substance, with "11/11 counted seeds" read as
"6/6 gate seeds". The packet's text is quoted first, then the gate reading. Every bar needs **6 of 6** gate seeds.

- **G1 = D1.** Packet: "C intact_a and C reversed: every C3 clause of `score_ask` holds, 11/11 counted seeds each." Gate: the
  same, 6/6 gate seeds each.
- **G2 = D2.** Packet: "C intact_b: the per-label asked vector equals C intact_a, 11/11." Gate: the same, 6/6.
- **G3 = D3.** Packet: "C permuted: `score_ask(permuted=True)` discriminates (the asks follow the exposure), 11/11." Gate:
  the same, 6/6.
- **G4 = D4.** Packet: "lesion_L: u_toad_4 asked, and different from C on u_toad_4, on 11/11 in both orders; attributable
  fraction of the extra toad asks against the intact_b null >= 0.5 (`tools.lab.attributable_to`)." Gate: u_toad_4 asked under
  lesion_L and not under C, 6/6 in both orders; the attributable fraction (section 3) >= 0.5.
- **G5 = D5.** Packet: "lesion_P intact_a: lp_learnable false and lp_unlearnable false on >= 9/11 (the phasic afferent is what
  keeps novel topics asked at the DA floor)." Gate: the same clauses on **6/6**. Not looser: 9/11 is a rate of 0.818 <!--derived-->; the
  nearest six-seed count at or above that rate is 5/6 (0.833 <!--derived-->), and 6/6 is stricter still. A brain whose lesion_P leaves one
  seed in six unchanged passes the dev threshold's rate (5/6 >= 9/11) and fails G5 (bar-sanity `null_one_seed_lesionP_inert`,
  pinned by `tests/test_c3_topic_stop_score.py`). No other dev bar had a threshold below 11/11.
- **G6 = D6.** Packet: "lesion_curiosity: 0 asks on every turn, 11/11." Gate: the same, 6/6.
- **G7 = D7.** Packet: "answer rate: `abstained` and `recalled_svo` identical to the same-order flag-off arm on every turn,
  11/11, for every counted arm that has such a comparator: C intact_a, C intact_b, lesion_L and lesion_P intact_a against
  flag-off intact_a; C reversed, lesion_L and lesion_P reversed against flag-off reversed; C permuted against flag-off
  permuted. lesion_curiosity has no same-order flag-off arm and is outside D7 (D6 covers its asks)." Gate: the same eight
  pairs, 6/6.
- **G8 = D8.** Packet: "read-backs: every lesion holds on every probe of its arm (stp_tau_d 200 ms; reward_us_spikes 0;
  weight sum 0)." Gate: the same, on lesion_L (both orders), lesion_P (both orders) and lesion_curiosity, 6/6.
- **G9, the Stage-2 C3 clause.** `score_ask(shard)["all"]` is True on C intact_a and on C reversed, 6/6. This is the
  Stage-2 gate's C3 correctness rule after Amendment A4 (`docs/drafts/stage2-exit/AMENDMENT-A4.md`, A4.2: correct =
  `score_ask(intact_a)["all"]` AND `score_ask(reversed)["all"]`), read on the C arms. It is the same call as G1 on the same
  two arms, so it cannot disagree with G1; it is listed so the retake's own clause is named and read on these seeds.
- **G10, answer rate.** Flag-ON never lowers answers against flag-OFF on any turn. For C intact_a and C intact_b against
  flag-off intact_a, C reversed against flag-off reversed and C permuted against flag-off permuted: no turn where the flag-off
  reply is `answered` and the flag-on reply is not, by `tools.answer_rate.classify`; and
  `tools.answer_rate.rate(<all 19 flag-on replies>)["answered"] >= rate(<all 19 flag-off replies>)["answered"]`; 6/6. These
  are the library calls. The `tools/answer_rate.py` CLI reads UNMEASURED on this layout (no `on_`/`off_` prefix, replies
  nested under `responses`), so it is not the instrument here; the scorer calls `classify` and `rate` directly.
- **DEC.** Every bar G1 to G10 PASS (G4 including its attribution) -> PASS: L+P qualifies for the default-ON flip, made by
  a separate flip prereg or commit before the 11-21 Stage-2 re-pin. Any bar FAIL -> STOP with the data: no flip, no retuning
  and no second flag on gate seeds; the FAIL is a verdict on the method, and section 8's rung is the next method. Otherwise
  UNDEFINED -> section 6.

## 3 The D4 / G4 attribution, defined

The packet names `tools.lab.attributable_to` without its inputs; the dev scorer had to choose them. They are fixed here, as
the dev read used them:

- treatment = the number of toad probes (u_toad_1 to u_toad_4) asked under `lesionL_intact_a` and not asked under
  `C_intact_a`, summed over the six gate seeds;
- control = the number of toad probes whose asked state differs between `C_intact_a` and `C_intact_b`, summed over the same
  seeds;
- fraction = `attributable_to("lesion_L extra toad asks vs the intact_b null", treatment, control)`, which must be >= 0.5.
  A fraction of None (treatment 0) is not a PASS; with treatment 0 the toad half of G4 already fails.

The reversed-order treatment (`lesionL_reversed` against `C_reversed`) is reported beside it; there is no reversed intact_b
null to set against it.

**The D2 / G2 and D4 / G4 nulls are zero by construction.** C intact_b is a fresh process with the same seed and flags as
intact_a, and the build is deterministic: on all 12 dev seeds the 19 intact_b responses compare equal to intact_a's as JSON
(an exact compare, the scorer's `responses_identical`). So G2 and the attribution's control can only differ from zero if the
run is non-deterministic. G2 is a cross-host determinism check (the two arms may run on different pool nodes), not a null
distribution, and the attribution fraction is 1.0 whenever the toad half of G4 holds.

## 4 Arms, lines and instruments

**Arms.** Exactly the packet's section 7, sixteen per gate seed, one fresh process each (96 runs):

| tag | runner arm | env on the line | role |
|---|---|---|---|
| C_intact_a, C_intact_b, C_reversed, C_permuted, C_lesion_curiosity | intact_a, intact_b, reversed, permuted, lesion_curiosity | C | counted |
| lesionL_intact_a, lesionL_reversed | intact_a, reversed | C + `BRAIN_CURIOSITY_TOPIC_STOP_LESION=1` | counted |
| lesionP_intact_a, lesionP_reversed | intact_a, reversed | C + `BRAIN_CURIOSITY_PHASIC_DA_LESION=1` | counted |
| off_intact_a, off_reversed, off_permuted | intact_a, reversed, permuted | none (production defaults) | flag-OFF baseline |
| Lpinned_intact_a, Lpinned_reversed | intact_a, reversed | `BRAIN_CURIOSITY_TOPIC_STOP=1 BRAIN_CURIOSITY_DA_LESION=1` | informational |
| pconst_intact_a, pconst_reversed | intact_a, reversed | C + `BRAIN_CURIOSITY_PHASIC_DA_CONST_SALIENCE=1` | informational |

The three flag-OFF arms are the flip-prereg rule's flag-OFF battery arms on the gate seeds: G7 and G10 compare every flag-on
arm with them at the same revision and seed. `lesion_curiosity`'s `BRAIN_CURIOSITY_LESION=1` is the runner's own arm env.

**Lines.** `research/coordination/c3-topic-stop-gate-lines.txt`, 96 lines (TSV: command, then the `--checked` reason), emitted
by `python -m research.runners.c3_topic_stop_score --emit-lines` with `<sha>` left as the placeholder for F (section 5). Each
command is the dev battery's line byte for byte (the 192 lines queued 2026-10-27 under RULING c3-battery-lines-1027:
`mem_ok.sh 9 1`, `memcap.sh 10`, the one-thread numpy env block, the flags in the dev order, `.venv/bin/python -u -m
research.runners.stage2_exit_gate --shard --seed N --group ask --arm A`) except three things: the seed, the `--out` directory
(`research/findings/raw/_c3_topic_stop/gate/s<N>/ask__<tag>.json`) and the pair `--allow-gate-seed --prereg
research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md`, without which the runner refuses a gate seed. Checked:
`emit_lines` regenerates all 192 dev commands byte for byte once that pair is removed. Each reason names this prereg (`prereg
<path> (F <sha>)`) and ends `mem_gb=9 max_age_h=240`, like the dev lines. `bash -n` passes on three substituted lines, the
queue shape check passes on all 96, and all 96 parse through `stage2_exit_gate.main` (run_shard stubbed;
`tests/test_c3_topic_stop_score.py::test_every_line_parses_in_the_runner`).

**Scorer.** `python -m research.runners.c3_topic_stop_score --mode gate --pin <F, full sha> --out
<gate root>/verdict.json` (default root `research/findings/raw/_c3_topic_stop/gate`, default seeds
the six gate seeds). It prints each cell DEFINED or UNDEFINED with the reason, each bar with its per-seed reads, the
attribution, the informational reads and the DEC line. Its module docstring gives the reading of each bar; it adds no bar and
loosens none. Tests: `tests/test_c3_topic_stop_score.py` (35).

**The scorer reproduces the dev verdict.** `python -m research.runners.c3_topic_stop_score --mode dev` over the 192 dev
shards (pin `79c6f5128`, seed 9 read and not counted, D5 at its packet need of 9) reads 192 of 192 cells DEFINED, D1 to D8 PASS
11/11 each, the attribution treatment 29 against control 0 (fraction 1.0; reversed-order treatment 24, reported), DEC PASS.
Its per-seed reads equal the dev scorer's (`score_c3ts_dev.py`, the read the skeptic re-scored) on every seed: each bar, the
lesion_L extra toad asks in both orders, the null toad differences, the asked vectors, null_count and the answered counts
(0 mismatches). G9 and G10, which the packet did not have, also read 11/11 on the dev shards (reported there, not a dev bar).

**Bar sanity.** `docs/drafts/c3-topic-stop-gate/bar_sanity_spec_c3ts_gate.py`, decide = DEC of the real scorer over synthetic
six-gate-seed sets. The perfect brains are the real dev shards relabelled (a random counted dev seed per gate seed), and the
same with the goat teach acquired. The nulls remove one capability on every seed or on one seed (no stop, the stop in one
order only, the learnable gap not asked, the moats asked, a non-deterministic build, the asks blind to the exposure, lesion_L
inert or inert in one order, lesion_P inert, lesion_curiosity asking, an answer lost, a recalled fact changed, a lesion not
held), or break one validity rule (a missing cell, a sidecar off the pin, a sidecar without the prereg, a lesion arm run
without its flag, a missing engagement trace, a flag-OFF arm that is not off). `random` asks at random. The record's verdict
and rates are in the cited JSON. An informational cell missing (`info_informational_cell_missing`) does not hold DEC; that is
reported, not thresholded.

## 5 Revision

**The dev pin is not the gate pin.** The arms import a static closure of 425 repo files (any import statement, module-level
or inside a function, from the runner, `webapp/server.py`, `webapp/curiosity_topic_stop_chat.py` and the organ;
`python -m research.runners.c3_topic_stop_score --closure`). `--closure-check 79c6f5128 6c6e08ad9` prints DIFFERS on 6:
`research/runners/stage2_exit_gate.py` (the C2 `kb_scramble` arms and `--kb-summary`; infer group only),
`research/runners/compositional_chain_route.py` (the C2 change), `research/runners/emotion_category_organ.py` (the census
moved on-device), `sim/bridge.py`, `sim/config.py` and `sim/kernels.py` (Izhikevich membrane sub-steps, default 1; the
`SIM_STABLE_INTEGRATION` switch, default OFF; apical GABA_A routing, default OFF). Their commits declare them default OFF;
this file does not assert the ask replies are unchanged across them. So the gate is pinned to
**P = `6c6e08ad903dffad002ba0a9ce1859da607634b1`** (origin/main when this file was written), not to `79c6f5128`. The dev
verdict is evidence at `79c6f5128` only; at the gate every comparison (flag-ON against flag-OFF, lesion against intact) is
within one revision and one seed.

**Gate revision F.** The pool's gate-seed guard needs the cited prereg to exist at the line's pinned revision, and the
provisioner refuses a revision that does not contain origin/main, so P itself cannot be the line revision once main has
moved. F is the tip of main, after this branch merges, that (1) contains this file, the scorer and the lines file, (2)
carries `research/runners/c3_topic_stop_score.py` byte-identical to this branch's (its sha256 is recorded below)
and (3) passes `python -m research.runners.c3_topic_stop_score --closure-check 6c6e08ad903dffad002ba0a9ce1859da607634b1 <F>`
with IDENTICAL. If (3) prints DIFFERS, no line is queued: an amendment names the new P and why, before any gate-seed run.
Every line pins F (`~/derisk-pool/revisions/<F>`), and the scorer is run with `--pin <F>`.

Scorer sha256 at this commit: `4812f633de5126696c10caf123686bf5e4589f02f06f509e3e3612a989e59ecb`.

**Before any gate line is queued** (the pool's own guards): a passing pool canary of `research.runners.stage2_exit_gate` at
F; and a CLEAN pre-flight read at F. The pre-flight runs the same sixteen arms on dev seed 7 at F (`--emit-lines <F>
--line-seeds 7 --line-dir research/findings/raw/_c3_topic_stop/preflight`, the same commands with seed 7) and records
`python3 tools/gate_preflight.py record --prereg research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md
--revision <F> -- .venv/bin/python -m research.runners.c3_topic_stop_score --mode gate --root
research/findings/raw/_c3_topic_stop/preflight --seeds 7 --pin <F>`. A dev-seed read is no verdict; it shows the instrument
can read the brain.

## 6 UNDEFINED rules

A cell is UNDEFINED, never a pass or a fail, when any of these holds (the scorer's `cell_problems`):

- the shard or its `.prov.json` sidecar is missing or unreadable;
- the sidecar is not a clean measurement of F: `git_sha` is not F, `git_dirty` is not False, or it is not a verified
  `git_archive` run (`source_manifest_verified_at_start` and `_at_exit` True), or its argv lacks `--allow-gate-seed --prereg
  research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md`, or names another arm, seed or group;
- the sidecar's `BRAIN_CURIOSITY*` env is not the arm's (section 4 table), or the shard's own arm, seed or group differ;
- `stage2_exit_gate.arm_validity` reports a problem (source not `tiny-demo +LTM`, PYTHONHASHSEED, corpus absent, an errored
  turn, an organ error, a turn table that is not the arm's registered one);
- the engagement trace is missing or wrong: with L on, an abstained probe without an applied `curiosity_topic_stop` trace, a
  trace with an `error`, `held.ou_enabled` not False, or (outside lesion_L) `held.stp_tau_d_ms` not 12000; with P on, a trace
  whose phasic read is not applied, or (outside lesion_P) `reward_us_spikes` 0 on a probe with salience > 0; a p_const_salience
  trace without `const_salience`; an L-pinned trace with a phasic read; a flag-OFF arm with any `curiosity_topic_stop` key.

A bar's seed read is UNDEFINED when a cell it reads is UNDEFINED, or when the moat rate is what decides it (Amendment A4.1:
no taught moat acquired). A bar is PASS when its PASS seeds reach 6, FAIL when its FAIL seeds make 6 unreachable, else
UNDEFINED. An UNDEFINED cell is re-run once at F, verbatim, and the re-run is logged (runbook WORKLOG) before its result is
read. A cell still UNDEFINED never counts: the bar stays UNDEFINED and DEC reads STOP with the data, never PASS. An
informational cell (Lpinned, pconst) follows the same re-run rule and is reported as UNDEFINED; it holds no bar.

## 7 Informational reads, reported and not barred

At each gate seed the scorer reports, so that the caveats of section 9 are measured at the gate seeds and not carried from
dev: Lpinned in both orders (`score_ask` all, the asked vector, whether it equals C's in the same order, `da_lesion_held`);
pconst in both orders (the same, and the count of probes read with a constant salience); `null_count` at k = 1 and 2 against
C intact_a over the abstained probes; per applied probe of C and lesion_P, whether asked equals `want_decision_hz >=
threshold` (the host decision, section 9 (f)); the toad ask strings (asks before the stop); the flag-OFF arms' C3 clauses in
both orders (the failure L+P addresses); lesion_P reversed; the tonic session DA at lesion_P's first and last applied probe;
and the answered counts of every G10 pair.

## 8 The next method for the host threshold

Every ask is still decided by host arithmetic at its last step (section 9 (f)). The next method replaces
`want_hz * gain >= threshold` with a spiking decision rung: the ASK pool's output drives an ask-commit population on the
substrate, and the existing spiking omission veto (`research/findings/2026-08-01-curiosity-reward-omission-veto-spiking-circuit-6seed.md`:
reward_us -> rmtg -| omit -> veto, plastic cue -> veto) inhibits it, so the follow-up question is asked when the commit
population fires and the veto does not, with no host comparison. Because the veto learns that a topic yields no reward, this
rung also replaces the count-based stop (L's label) with a learning-progress-sensitive one. The `replaced-by` line of the
header (a per-pathway dopamine synaptic-gain target on the topic -> ASK synapses) retires the gain formula; this rung retires
the threshold. Neither is built; neither is part of this gate.

## 9 What a PASS does and does not show

A PASS shows that, at six seeds never used to design or tune the method, with L+P on, the curiosity follow-up is asked on
the novel gaps and the learnable gap and stops on the unlearnable one in both orders, follows the exposure, is cut by each
lesion as registered, and costs no answer against the flag-OFF brain. It does not show the following, each carried from the
dev read and re-measured at the gate seeds where a read exists (section 7):

- **(a) P is shown necessary only against the gain formula's floor clip, not for the per-topic stop.** L alone with the host
  gain pinned to 1.0 (Lpinned) passes every C3 clause on 11/11 dev seeds in both orders. lesion_P fails D5 because it sends
  `da_eff` to 0, where the formula's clip sets the gain to 0.25. On dev seed 12, lesion_P intact_a still asked all three novel
  gaps while the tonic session DA fell only from 0.64 to 0.17.
- **(b) The phasic drive carries C3, not its topic-locking.** pconst (the salience replaced by a constant, so the topic-locking
  is removed) passes 11/11 in both orders, and its asked vector equals C's on 9/11 (intact_a) and 11/11 (reversed).
- **(c) A host count rule reproduces the decision.** null_count with k = 1 (ask iff the shared pool is curious and the topic
  was presented fewer than once before) equals C's ask decision on 117 of 121 abstained dev probes. L is count-based by
  design; the spiking omission-veto rung (section 8) is unbuilt.
- **(d) The goat moat is UNDEFINED.** `m_goat` is undefined on every dev seed: the goat teach is refused by the default-ON D4
  comprehension gate (the known C2 defect), so the moat rate rests on yak and moose. In the permuted arm `g_crab` is undefined
  for the same reason (the crab teach is refused).
- **(e) The D2 and D4 nulls are zero by construction** (section 3): C intact_b is an exact replay of intact_a, so G2 is a
  cross-host determinism check.
- **(f) Host-decision share.** Every ask or no-ask is still host arithmetic, `want_hz * clip(0.2, 3.0, 1 + 1.5 (DA - 0.5)) >=
  threshold` (`webapp/da_curiosity_drives_chat.py`), over spiking inputs (the topic-stop ASK-pool rate and the SNc read). On
  every applied C probe of the dev read, asked equals `want_decision_hz >= threshold`. P method B is a declared host-arithmetic
  scaffold (header): the gain formula, its clip, the rate normalization and the two linear afferent encodings are host code;
  the tonic afferent still carries the session-global host EMA.
- **(g) The answer-rate instrument.** The `tools/answer_rate.py` CLI reads UNMEASURED on this layout; the dev lens used the
  library `rate()` over all 19 replies (C 7/19 = flag-off 7/19 on every dev seed), and G10 registers the library calls. The
  later flip's smoke records (`tools/smoke_first.py` needs `answer_rate.py paired --json` records) cannot be written from these
  shards without a layout reader; that is owed by the flip step, not by this gate.
- **(h) The D4 attribution inputs were the scorer's choice at dev** (the packet named `attributable_to` without them); section
  3 fixes them before any gate-seed data.

Also not shown: anything about the gain left at its production default with L alone, the afferent-only noise variant (packet
section 3, needs a build), the one-brain residual (the topic slices live on the per-conversation `Context`, packet section
9), or any claim of felt curiosity: the follow-up question is a functional read-out of the organ's state (CLAUDE.md
non-negotiable 5).
