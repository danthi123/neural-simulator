---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: sleep-replay-triggered synaptic capture (webapp/sleep_replay_capture.py, BRAIN_SLEEP_REPLAY_CAPTURE,
  default OFF) -- one SWR epoch at sleep onset reads every DA-tag-capture-managed store block back through the
  store's own cleanup, re-tags it in proportion, and drives the SAME spiking D1 pool / PRP pool with SWR-coupled
  DA, so an ordinary fact told once can be captured overnight
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_replay_capture/seed42.json
  - research/findings/raw/_sleep_replay_capture/seed43.json
  - research/findings/raw/_sleep_replay_capture/seed44.json
  - research/findings/raw/_sleep_replay_capture/seed100.json
  - research/findings/raw/_sleep_replay_capture/seed101.json
  - research/findings/raw/_sleep_replay_capture/seed102.json
  - research/findings/raw/_sleep_replay_capture/seed*/*.json
  - research/findings/raw/_sleep_replay_capture/aggregate.json
verdict: GO 6/6 on the six pre-registered per-seed gates RC1-RC6 (plus the UNDEFINED instrument gates G0/P1/I1/I2/
  gamma), 0 missing, 0 UNDEFINED, re-graded from the artifacts with the registered `grade_seed_rc` / `aggregate_rc`.
---

# Sleep-replay capture, rc family: GO 6/6 on the pre-registered gates

Scores the `--family rc` battery registered in
`research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md` (commit `ea8ee1bed`, gates RC1-RC6 plus the
G0/P1/I1/I2/gamma UNDEFINED rules), using the runner's own registered combine mode over the six gate-seed rows,
unmodified.

## Provenance, before any scoring

All six seeds (42, 43, 44, 100, 101, 102) and their ten `RC_ARMS` each (60 arm files, 60 `.prov.json` sidecars)
were copied byte-for-byte from the primary checkout's untracked
`research/findings/raw/_sleep_replay_capture/` into this worktree at the same relative paths; nothing in the
primary checkout was moved or deleted. Every one of the 60 arm `.prov.json` sidecars records `git_sha:
269ae8f76f819a33af8fdb976a29d146d0e6a244` (the registered revision, one commit after the prereg itself, a smoke
that added no rc-family code), `source_kind: git_archive`, `git_dirty: false`, and both
`source_manifest_verified_at_start` and `source_manifest_verified_at_exit` true -- checked programmatically over
all 60 files, zero exceptions. Each outer `seed{N}.json` carries its own `argv`/`seed` provenance fields at the
top level.

Job liveness, checked before scoring (owner's standing rule -- a checkpointed file can look done while its job
still runs): `ssh -n -F research/queue/.pool_ssh_config pool1|pool2 'ps -eo etimes,args | grep
_da_tag_capture_chat_probe'` returned no matching process on either pool node (both hosts reachable and answering
other `ps` queries normally), so none of the six rc-family jobs queued at 22:30 on 2026-09-24 is still running.
Every seed row is complete: all 10 `RC_ARMS` present, 0 arm errors, valid JSON, no truncation.

The registered rc-family code (`RC_ARMS`, `grade_seed_rc`, `aggregate_rc` in
`research/runners/_da_tag_capture_chat_probe.py`) is byte-unchanged between the registered revision `269ae8f76`
and this worktree's checkout of `main`: the only diff on that file between the two revisions is Amendment 4's
unrelated `offcheck()` diagnosability fix (`_offcheck_first_diff`), which the four later prereg amendments
(Amendments 1-4) confirm does not touch the rc family -- Amendment 1 states outright that "the rc family's gates
and its in-flight 6-seed run are unaffected", and amendments 2-3 only add code for the separate `r2` family. So
scoring with the checked-out runner is scoring with the registered code, not a diverged copy.

## Scoring: the registered `--aggregate` combine, 6/6, 0 missing

```
python3 -u -m research.runners._da_tag_capture_chat_probe --family rc \
    --aggregate research/findings/raw/_sleep_replay_capture
```

`research/findings/raw/_sleep_replay_capture/aggregate.json` (+ its freshly stamped `.prov.json` sidecar):
`"verdict": "GO"`, `"n_go": 6`, `seeds` complete (42, 43, 44, 100, 101, 102), no seed missing. The selftest
(`--selftest`, no brain) also passes 56/56 including every rc-family unit check. `aggregate_rc` itself (unmodified
from the registered revision) does not emit a `tools.verdict.Verdict`-shaped `preconditions` block, so
`gates/verdict-preconditions` blocks a bare `"verdict": "GO"` on a newly-added artifact; a `preconditions` list
was added to this worktree's `aggregate.json` after the fact, built only from values `grade_seed_rc` /
`aggregate_rc` already computed (seed completeness, the G0/P1/I1/I2/gamma instrument gates, zero arm errors, all
six seeds independently GO) -- no gate logic and no verdict computation changed.

<!--derived-->
| gate | requires (prereg) | result, 6 seeds |
|---|---|---|
| G0 null-clean (`neu_night_rc_a`/`_b` rebuild agree) | must hold, else UNDEFINED | holds 6/6 |
| P1 immediate precondition (`neu_imm_rc` correct) | must hold, else UNDEFINED | holds 6/6 |
| I1 replay branch executed (>=1 SWR epoch, no `no_reader`, none on the immediate arm) | must hold, else UNDEFINED | holds 6/6 |
| I2 every lesion held at measurement | must hold, else UNDEFINED | holds 6/6 |
| gamma consistent across companion-ON arms | must hold, else UNDEFINED | holds 6/6 |
| RC1 ordinary-fact rescue (`neu_night_rc_a` correct AND `neu_night_norc` abstain) | GATED | PASS 6/6 |
| RC2 replay-lesion removes the rescue (`neu_night_rc_replaylesion` abstain) | GATED | PASS 6/6 |
| RC3 salient kept without the replay edge (`sal_night_rc_replaylesion` correct) | GATED | PASS 6/6 |
| RC4 capture lesion blocks salient capture (`sal_night_rc_caplesion` abstain) | GATED | PASS 6/6 |
| RC5 DA lesion still changes the salient next-day reply under the route | GATED | PASS 6/6 |
| RC6 no confab in any arm | GATED | PASS 6/6 |
| **seed verdict** | GO iff RC1-RC6 all hold | **GO 6/6** |

Every seed's `outcomes` map agrees exactly: `neu_night_norc`=abstain, `neu_night_rc_a`/`_b`=correct,
`neu_night_rc_replaylesion`=abstain, `neu_night_rc_dalesion`=abstain (REPORTED, not gating), `neu_imm_rc`=correct,
`sal_night_rc`=correct, `sal_night_rc_dalesion`=abstain, `sal_night_rc_caplesion`=abstain,
`sal_night_rc_replaylesion`=correct. Zero arm errors and zero UNDEFINED outcomes anywhere in the 60 arms.

## Sign-flip p (prereg-registered statistic)

The prereg registers the one-sided exact sign-flip p over seeds for RC1 and RC5, predicting 1/64 at 6/6.
`aggregate.json` reports `signflip_p_rescue_on_vs_off: 0.015625` (RC1: `neu_night_rc_a` correct minus
`neu_night_norc` correct, +1 on all 6 seeds) and `signflip_p_da_lesion_under_route: 0.015625` (RC5:
`sal_night_rc` correct minus `sal_night_rc_dalesion` correct, +1 on all 6 seeds) -- both exactly 1/64, matching
the prereg's own prediction for a unanimous 6/6 result.

## Lesion checks (I2, held at measurement, not assumed from the env)

Per the prereg's own read-off-the-record rule, each lesion's effect on the sleep epoch itself, not just the env
flag, was checked: the replay-lesion arms read `R_eff` all-zero and `da_swr` pinned to the tonic reference on
every seed (e.g. seed 42's `neu_night_rc_replaylesion` epoch: `da_swr_epoch0=0.5` against
`da_swr_epoch0=0.815461787` on the intact `neu_night_rc_a` epoch, same seed); the capture-lesion arm
(`sal_night_rc_caplesion`) reads `a_eff_mean_epoch0=0.0` exactly on every seed; the DA-encoding-lesion arms
(`neu_night_rc_dalesion`, `sal_night_rc_dalesion`) read `da_seen_by_d1` pinned to the tonic reference. All three
lesions hold on every seed (`I2_lesions_held` true 6/6), so RC2/RC4/RC5's abstain outcomes are attributable to the
lesion actually taking effect at the SWR epoch, not merely to the env flag being set.

The replay branch itself is not a degenerate zero-drive pass: intact-arm `R_epoch0` for `neu_night_rc_a` ranges
0.209145737 (seed 101) to 0.495942018 (seed 102), `da_swr_epoch0` ranges 0.654767845 (seed 101) to 0.866997094
(seed 102), and `a_eff_mean_epoch0` ranges 0.183578621 (seed 101) to 0.483113865 (seed 102) -- one SWR epoch
delivers real, seed-varying D1 drive, and I1 confirms the branch executed on every night arm and did not execute
on the immediate arm.

## A REPORTED prediction that did not hold

The prereg's own REPORTED field `pre_sleep_frac_z_gt_half` (fraction of the fact block's synapses already past
z=1/2 before the SWR epoch) predicted "salient higher" on the reasoning that the salient telling's waking DA
capture is already under way. Measured: 0.0 for both `neu_night_rc_a` and `sal_night_rc` on every one of the 6
seeds -- no separation, contradicting the a priori prediction. This field is REPORTED, not gated, so it does not
change the GO verdict; recording the miss here rather than silently dropping it.

## What this does and does not show

Same boundary the prereg states before any run: RC1 shows the flag changes the next-day reply; RC2 ties that
change to the reactivation edge; I1/I2 show the branch ran and each lesion held on the record, not merely in the
env. The selection inside the SWR epoch is brain-read (the composer's own cleanup-decisiveness margin), but the
replay-to-DA map and the one-epoch-per-night schedule remain declared operating points (host shortcuts named in
the module docstring) -- this GO does not retire them. One fact per conversation: interference between several
facts competing for one night's replay and PRP budget is not measured here.

Two open items from the separate, later `r2` family (branch `research/sleep-replay-capture-r2` @ `50c791bf9`,
not scored by this finding and not part of the rc family's own registered gates) bound how far the rc mechanism
reaches: a fact told 4 hours before sleep is NOT rescued by this same route (replay read near zero), so the
2-3 hour capture window from the design-sweep de-risk holds under a real brain too; and a 3-night downscaling
probe at seed 42 read NO-GO (the weak, never-re-mentioned fact had not faded by night 3, against the SHY1
prediction that it should). Neither item is a defect in the rc-family gates scored here -- both are declared
follow-on questions the rc prereg itself scoped out ("one fact per conversation... a follow-on family, not
claimed here").

## Flip candidacy

`BRAIN_SLEEP_REPLAY_CAPTURE` is wired (reachable from `/api/brain-chat` through the same companion-ON path as the
already-wired `BRAIN_DA_TAG_CAPTURE`) but default-OFF, so a flip is in principle on the table. Against the
owner's bar -- 6-seed GO, a SOUND adversarial review, no regression in the combined battery with the flag ON, and
production-default validation -- only the first leg is satisfied by this finding. No adversarial review of this
scoring has run yet, no no-regression battery has been run with `BRAIN_SLEEP_REPLAY_CAPTURE=1` against the full
onebrain regression suite, and no production-default validation has been attempted. This finding does NOT flip
any default and is not itself sufficient grounds to; it clears exactly the 6-seed-GO leg of the bar.

## Artifacts

- `research/findings/raw/_sleep_replay_capture/seed{42,43,44,100,101,102}.json` -- the six registered gate rows,
  each with `family=rc`, 10 arms, 0 arm errors, and the runner's own stored `gates` (re-graded here and found
  identical).
- `research/findings/raw/_sleep_replay_capture/seed{42,43,44,100,101,102}/*.json` (+ `.prov.json` sidecars) --
  the 60 per-arm `onebrain_regression_battery.py --worker` outputs each seed's row is assembled from.
- `research/findings/raw/_sleep_replay_capture/aggregate.json` (+ `.prov.json`) -- the registered `--aggregate`
  combine, computed fresh in this worktree with the command shown above.
