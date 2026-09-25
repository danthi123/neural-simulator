---
type: finding
status: no-go
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: awake-rest sharp-wave-ripple replay (webapp/awake_replay_capture.py, BRAIN_AWAKE_REPLAY_CAPTURE,
  default OFF) -- during quiet-wake idle ticks the store's own cleanup margin re-induces each DA-tag-capture
  block's early-phase expression (no PRP, no D1 read), so a fact told ~4 h before sleep still carries a live
  tag at sleep onset and can be captured by the existing sleep-replay route (BRAIN_SLEEP_REPLAY_CAPTURE)
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md (Amendments 4-5, `--family arc`,
  ARC_ARMS, gates ARC1-ARC7)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_awake_replay_capture/seed42.json
  - research/findings/raw/_awake_replay_capture/seed43.json
  - research/findings/raw/_awake_replay_capture/seed44.json
  - research/findings/raw/_awake_replay_capture/seed100.json
  - research/findings/raw/_awake_replay_capture/seed101.json
  - research/findings/raw/_awake_replay_capture/seed102.json
  - research/findings/raw/_awake_replay_capture/seed*/*.json
  - research/findings/raw/_awake_replay_capture/aggregate.json
  - research/findings/raw/_awake_replay_capture/design_fake_substrate.json
verdict: NO-GO 5/6 on the pre-registered per-seed gates ARC1-ARC7 (`n_go=5`, all six seeds present, zero
  UNDEFINED). Seed 101 fails ARC1 alone (the rest-rescue itself did not fire); every instrument gate
  (G0/P1/I1/I2/I3/gamma) and every other ARC gate hold clean on all six seeds, including seed 101, so this is
  a genuine behavioural miss, not an instrument defect. Re-graded from the artifacts with the registered
  `grade_seed_arc` / `aggregate_arc`, unmodified.
---

# Awake-rest replay, arc family: NO-GO 5/6 -- one seed's rescue does not fire, cleanly

Scores the `--family arc` battery registered as Amendment 4-5 of
`research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md` (ARC_ARMS, 12 arms, gates ARC1-ARC7 plus
the G0/P1/I1/I2/I3/gamma UNDEFINED rules), using the runner's own registered combine mode over the six gate-seed
rows, unmodified.

## Provenance, before any scoring

All six seeds (42, 43, 44, 100, 101, 102) and their twelve `ARC_ARMS` each (72 arm files, 72 `.prov.json`
sidecars), plus the pre-committed `design_fake_substrate.json` and the local `offcheck_counterfactual.json`
side-check, were copied byte-for-byte from the primary checkout's untracked
`research/findings/raw/_awake_replay_capture/` into this worktree at the same relative paths (`diff -rq` against
the primary checkout reports no difference other than this worktree's freshly-generated `aggregate.json`);
nothing in the primary checkout was moved or deleted.

Every one of the 72 arm `.prov.json` sidecars was checked programmatically: all 72 record `git_sha:
30ba29d4b4fa5c5d9b069a83ef9e317f75ba67c5` (the revision named in the amendment's compute plan) in full,
`source_kind: git_archive`, `git_dirty: false`, and both `source_manifest_verified_at_start` and
`source_manifest_verified_at_exit` true -- zero exceptions across all 72. The registered runner code
(`research/runners/_da_tag_capture_chat_probe.py`) is byte-identical between that pinned revision and this
worktree's `main` checkout (`git diff 30ba29d4b... HEAD -- research/runners/_da_tag_capture_chat_probe.py` is
empty), so re-grading with the checked-out code scores the registered code, not a diverged copy.

Job liveness, checked before scoring (owner's standing rule -- a checkpointed file can look done while its job
still runs): `ssh -n -F research/queue/.pool_ssh_config pool1|pool2 'ps -eo etimes,args | grep
_da_tag_capture_chat_probe'` found processes on both nodes, but every one of them is a *different* family
(`--ltm on` base-family reruns on pool1/pool2, and separate `--family r2` jobs on pool2) -- none is a
`--family arc` run. No arc-family job is still in flight; every seed row is complete: all 12 `ARC_ARMS` present,
0 arm errors, valid JSON, no truncation.

## Scoring: the registered `--aggregate` combine, 5/6, 0 missing

```
python3 -u -m research.runners._da_tag_capture_chat_probe --family arc \
    --aggregate research/findings/raw/_awake_replay_capture
```

`research/findings/raw/_awake_replay_capture/aggregate.json` (+ its freshly stamped `.prov.json` sidecar):
`"verdict": "NO-GO"`, `"n_go": 5`, `seeds` complete (42, 43, 44, 100, 101, 102), no seed missing, no seed
UNDEFINED. The selftest (`--selftest`, no brain) passes, including every arc-family unit check (the designed-GO
fixture reads GO; nine designed-failure fixtures each read the correct NO-GO/UNDEFINED). As with the rc-family
finding, `aggregate_arc` does not emit a `tools.verdict.Verdict`-shaped `preconditions` block, so
`gates/verdict-preconditions` blocks a bare `"verdict": "NO-GO"` on a newly-added artifact; a `preconditions`
list was added to this worktree's `aggregate.json` after the fact, built only from values `grade_seed_arc` /
`aggregate_arc` already computed (seed completeness, zero arm errors, no UNDEFINED seed, the G0/P1/I1/I2/I3/gamma
instrument gates holding on all six seeds) -- no gate logic and no verdict computation changed, and every listed
precondition is `ok: true` (a NO-GO with a failed precondition would have to read UNDEFINED instead, which this
is not).

<!--derived-->
| gate | requires (prereg) | result, 6 seeds |
|---|---|---|
| G0 null-clean (`lr_arc_a`/`_b` rebuild agree) | must hold, else UNDEFINED | holds 6/6 |
| P1 immediate precondition (`neu_imm_arc` correct) | must hold, else UNDEFINED | holds 6/6 |
| gamma consistent across companion-ON arms | must hold, else UNDEFINED | holds 6/6 |
| I1 awake branch ran exactly as scheduled (48 bouts on `datr`/`datcr`, 0 on `datl`/`datni`, before the one sleep epoch) | must hold, else UNDEFINED | holds 6/6 |
| I2 every lesion held at measurement (awake edge, night edge, DA-encoding) | must hold, else UNDEFINED | holds 6/6 |
| I3 awake bouts add no PRP (constant D1-drive-entry count, non-increasing pool) | must hold, else UNDEFINED | holds 6/6 |
| ARC1 rest rescues the long-delay fact (`lr_arc_a` correct AND `lr_noarc` abstain) | GATED | PASS 5/6, **FAIL seed 101** |
| ARC2 awake-edge lesion removes the rescue (`lr_arc_lesion` abstain) | GATED | PASS 6/6 |
| ARC3 no rescue without rest (`ln_arc` abstain) | GATED | PASS 6/6 |
| ARC4 rest alone does not make it permanent (`lr_arc_sleeplesion` abstain) | GATED | PASS 6/6 |
| ARC5 DA lesion blocks the rescue (`lr_arc_dalesion` abstain) | GATED | PASS 6/6 |
| ARC6 salient kept on waking capture (`lsr_arc_sleeplesion` correct) | GATED | PASS 6/6 |
| ARC7 no confab (every arm, REPORTED arms included) | GATED | PASS 6/6 |
| **seed verdict** | GO iff ARC1-ARC7 all hold | **GO 5/6, NO-GO on seed 101** |

Seed 101 is the only seed where any gate fails, and the failing gate is exactly one: ARC1. Every instrument gate
and every other behavioural gate reads clean on seed 101 too (`n_arm_errors=0`), so `grade_seed_arc` reads a
genuine NO-GO for that seed, never UNDEFINED -- the branch executed on schedule, every lesion held on the record,
and no arm confabulated; the rest-rescue itself simply did not happen there. Because `lr_arc_a` already abstained
on seed 101, ARC2-ARC5 pass on it in a weaker, vacuous sense (there was no rescue for the lesion arms to fail to
remove); only ARC6 (the salient-fact control, unrelated to the neutral-fact rescue) and the instrument gates
carry independent information on that seed.

## Seed 101: a quantified substrate-level miss, not an instrument failure

`grade_seed_arc`'s own REPORTED fields (`R` at the first and last awake bout, and at sleep onset) show why:
across the 48 five-minute awake-rest bouts of the 4-hour window, the reactivation read for `lr_arc_a` decays far
more on seed 101 than on any other seed.

<!--derived-->
| seed | R at 1st bout | R at 48th bout | R at sleep onset | `lr_arc_a` outcome |
|---|---|---|---|---|
| 42 | 0.425496788 | 0.417000853 | 0.416991764 | correct |
| 43 | 0.281844205 | 0.243238551 | 0.243230057 | correct |
| 44 | 0.399221155 | 0.379019341 | 0.379019341 | correct |
| 100 | 0.384405719 | 0.365946317 | 0.365946317 | correct |
| **101** | **0.207451321** | **0.030467672** | **0.030533799** | **abstain** |
| 102 | 0.494862092 | 0.487157562 | 0.487157612 | correct |

On five of six seeds the read is essentially stable across the 4-hour rest window (a few-percent decay). On seed
101 it starts lowest of the six (0.207451321, already below every other seed's *last*-bout read) and then
collapses roughly 7-fold to 0.030533799 by sleep onset -- too weak for the sleep-replay route's SWR-coupled DA to
carry the block into late phase. This is per-neuron heterogeneity in the composer's cleanup margin (`R_i`, the
same quantity the sleep-replay family's own finding already reports varying ~2.4x across seeds,
`0.209145737-0.495942018` for its one-epoch read), not a bug in the awake route: I1 confirms all 48 bouts ran, on
schedule, with a substrate read every time, and G0's null-control rebuild (`lr_arc_b`) reproduces `lr_arc_a`
exactly on seed 101 too (identical outcome, ledger, and both replay records), so the miss is deterministic given
the seed, not run-to-run noise.

## Lesion and instrument checks held at measurement (I2/I3), not merely from the env flag

Sampled from `seed42.json` (cited above): the awake-edge lesion arm's first bout reads `R=[0.425496788]` (the
same value as the un-lesioned arm's first bout) but `R_eff=[0.0]` and `early_after` unchanged from `early_before`
-- the read happens, the effect is zeroed. The night-edge (sleep-replay) lesion arm's epoch reads `R_eff=[0.0]`
and `da_swr=0.5` (the tonic reference) against the intact arm's `da_swr_epoch0` well above tonic on the same
seed. The DA-encoding lesion arm's epoch reads `da_seen_by_d1=0.5` (tonic) regardless of the SWR content. All
three hold on every one of the six seeds (`I2_lesions_held` true 6/6).

I3 (no PRP from awake replay) also holds structurally, not just as a summary boolean: `lr_arc_a`'s 48 bouts on
seed 42 carry a constant `n_drive_entries=5` from the first bout to the last, and `p_at_bout` only decreases bout
over bout (0.000796003038 at the first bout, falling monotonically through the rest window) -- the PRP pool
decays across the awake period exactly as it would with no bouts running at all; nothing about a bout adds to it.

## What the REPORTED arms show (never gating; ARC7 aside)

<!--derived-->
| arm | role | correct count, 6 seeds | matches the a-priori prediction? |
|---|---|---|---|
| `lr_ledger_off` | today's production default (ledger off), same rest | 6/6 | yes -- with no forgetting mechanism at all, the fact is trivially kept |
| `lq_arc` | hourly rest (4 bouts in 4 h) | 0/6 | yes -- the prereg predicted abstain (insufficient dose) |
| `lz_arc` | rest only in the last hour (12 bouts, late) | 0/6 | **no** -- the prereg's own fake-substrate design sweep predicted regrowth-and-capture; the Amendment-5 seed-42 smoke already found this AGAINST prediction, and it holds on all six brain seeds now: a faint, 3-h-old trace does not regrow from an hour of rest on this substrate |

`reported_correct_counts` in `aggregate.json` gives these three counts directly (0, 0, 6); the `lz_arc` result
generalizes the single-seed miss recorded in the prereg's own Amendment 5
(`research/findings/raw/_awake_replay_capture_smoke/seed42.json`, not a gate row) to the full 6-seed battery: the
fake-substrate design sweep (`design_fake_substrate.json`) overstates how much a faint trace recovers from late,
sparse rest.

## Sign-flip p (prereg-registered statistic)

`aggregate.json` reports `signflip_p_rest_rescue_on_vs_off: 0.03125` (ARC1: `lr_arc_a` correct minus `lr_noarc`
correct, `diffs_on_minus_off = [1, 0, 1, 1, 1, 1]`, seed 101 contributing the 0) and
`signflip_p_rest_rescue_intact_vs_awake_lesion: 0.03125` (the same pattern against `lr_arc_lesion`). Both are
1/32, not the prereg's a-priori 1/64-at-6/6 prediction -- the one-sided exact sign-flip test over six seeds
where five show the predicted direction and one shows no effect either way (a 0, not a reversal).

## What this does and does not show

The mechanism does what Amendment 4 built it to do on 5 of 6 seeds, cleanly separated from the alternative
explanations the gate family is designed to rule out: ARC1+ARC2 tie the rescue to the awake reactivation edge
specifically (not to idle ticks in general -- ARC3 shows the flag alone does nothing without rest), ARC4+I3 show
the awake route supplies no PRP of its own (the night's DA-coupled capture still does the actual gating), and
ARC5+ARC6 show DA and the existing salient/neutral separation both survive with the new route armed. None of
that is in question on seed 101; only the size of the effect is, and only there.

It does NOT show that the mechanism reaches every seed's realization of the substrate at the read strengths
this one long-delay, one-fact scenario produces: seed 101's cleanup margin for this specific fact and this
specific brain build is low enough, and decays enough over 4 hours of rest, that the sleep-replay route's
DA-coupled capture does not clear its own late-phase threshold. The prereg's own declared scope applies
unchanged: one fact per conversation (no measurement of interference between several facts competing for one
rest/night's replay budget), LTM off only, numpy CPU only, and the induction law / bout window / rest protocol
remain declared operating points (module docstring, HOST SHORTCUTS) that this scoring does not retire even on
the 5 seeds that pass. The hourly (`lq_arc`) and late-rest (`lz_arc`) arms are REPORTED, not gated, and by
design do not enter `n_go`; `lz_arc`'s 0/6 is itself informative (the design-sweep's regrowth prediction does not
hold on the brain) but changes no seed's verdict. The separate local `offcheck_counterfactual.json` (byte-
identical OFF against a counterfactual with the feature's own commits reverse-applied, `"byte_identical_off":
true`, `"verdict": "IDENTICAL"`) is a local, non-pool run (its own `.prov.json` records `source_kind: null`,
`git_dirty: true`, a short `git_sha`) -- included here as supporting context on the flag-off path, not as one of
the six registered gate rows, and its two temporary worktrees are each symlinked to a real `data/corpus/`
(verified by reading `offcheck()`'s construction, `_corpus_src()`), so it does not carry the corpus-missing risk
that Amendment 5's guard was written for.

Per `docs/TERMS.md`: this route re-potentiates the SAME store the fact was already written to (no transfer, no
lesion of a source structure), so it is not called "consolidation" here, matching the prereg's own usage.
"Byte-identical" above is asserted only for the offcheck's hash compare (`head_run`/`counterfactual_run`
`replies_sha256` and `store_sha256` match exactly, `null_replies_identical`/`null_store_identical` both true).
"Lesion" is used only for the three manipulations whose effect is verified to still hold on the record at
measurement (I2, above), not merely from the env flag. "GO" is used only for the gate's own per-seed verdict,
never lifted from a run whose aggregate verdict is negative -- the family's own registered combine reads NO-GO,
5/6, and that is the verdict this finding reports, not "5/6 GO" read as a headline.

## Flip candidacy

`BRAIN_AWAKE_REPLAY_CAPTURE` is default OFF and stays OFF. Against the owner's bar for flipping a validated
default (6-seed GO, a SOUND adversarial review, no regression in the combined battery with the flag ON, and
production-default validation), this finding does not clear even the first leg: the registered 6-seed combine
reads NO-GO (5/6, one clean behavioural miss), not GO. No adversarial review has run, no no-regression battery
has been run with the flag ON, and no production-default validation has been attempted. This finding does not
flip any default and does not recommend doing so; the mechanism's next rung (named, not built here) is whatever
would let the awake route carry a fact through seed 101's realization of the substrate too -- e.g. a
protection/gain term less dependent on the single-fact cleanup margin sampled once every 5 minutes, or a
lower-variance read -- rather than retuning a constant to this one seed's number.

## Artifacts

- `research/findings/raw/_awake_replay_capture/seed{42,43,44,100,101,102}.json` -- the six registered gate rows,
  each with `family=arc`, 12 arms, 0 arm errors, and the runner's own stored `gates` (re-graded here with the
  registered `grade_seed_arc` and found identical to the stored gates on all six).
- `research/findings/raw/_awake_replay_capture/seed{42,43,44,100,101,102}/*.json` (+ `.prov.json` sidecars) --
  the 72 per-arm `onebrain_regression_battery.py --worker` outputs each seed's row is assembled from, all 72
  provenance-verified at `30ba29d4b4fa5c5d9b069a83ef9e317f75ba67c5`, `source_kind: git_archive`.
- `research/findings/raw/_awake_replay_capture/aggregate.json` (+ `.prov.json`) -- the registered `--aggregate`
  combine, computed fresh in this worktree with the command shown above, plus a `preconditions` block added
  after the fact for `gates/verdict-preconditions` (values only, no re-scoring).
- `research/findings/raw/_awake_replay_capture/design_fake_substrate.json` (+ `.prov.json`) -- the pre-registered
  fake-substrate design sweep, committed with Amendment 4, referenced above only for the `lz_arc` comparison.
- `research/findings/raw/_awake_replay_capture/offcheck_counterfactual.json` (+ `.prov.json`) -- the local
  byte-identical-OFF counterfactual check, supporting context only (not a gate row).
