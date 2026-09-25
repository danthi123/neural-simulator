---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: sleep-replay-triggered synaptic capture, r2 extension (webapp/sleep_replay_capture.py,
  BRAIN_SLEEP_REPLAY_CAPTURE + BRAIN_SLEEP_DOWNSCALING, both default OFF) -- item 1 tests a long-delay (~4 h)
  telling against the same one-epoch route; item 2 adds a nightly synaptic-downscaling pass (de Vivo et al. 2017
  magnitude, R_i-protected) after the route's reactivation; item 3 is a byte-identical-OFF counterfactual for the
  whole feature
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_replay_capture_r2/seed42.json
  - research/findings/raw/_sleep_replay_capture_r2/seed43.json
  - research/findings/raw/_sleep_replay_capture_r2/seed44.json
  - research/findings/raw/_sleep_replay_capture_r2/seed100.json
  - research/findings/raw/_sleep_replay_capture_r2/seed101.json
  - research/findings/raw/_sleep_replay_capture_r2/seed102.json
  - research/findings/raw/_sleep_replay_capture_r2/seed*/*.json
  - research/findings/raw/_sleep_replay_capture_r2/offcheck_counterfactual.json
  - research/findings/raw/_sleep_replay_capture_r2/aggregate.json
verdict: item 2 (GATED, downscaling) NO-GO 0/6 GO (4/6 seeds cleanly NO-GO on SHY1, 2/6 seeds UNDEFINED on an
  unrelated precondition, 0/6 GO), as registered and as scored by the runner's own unmodified `aggregate_r2`.
  Item 1 (REPORTED, long delay) reads NOT-RESCUED 6/6. Item 3 (byte-identical OFF) reads IDENTICAL.
---

# Sleep-replay capture, r2 family: NO-GO 0/6 on the downscaling gate, as registered

Scores the `--family r2` battery registered as Amendment 1 (plus Amendments 2 and 3) of
`research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md`, using the runner's own registered combine
mode over the six gate-seed rows, unmodified. Amendment 6 (committed later, merged `df12ec1cc`, governing the
separate `fi` family) withdraws the three-night downscaling criterion (SHY1) for the `fi` family only, on
biological grounds discovered after seeing the seed-42 r2 smoke's 3-night NO-GO. **The verdict of record below is
r2's rule exactly as registered before its runs ran (SHY1 included).** A second, clearly separated reading applies
Amendment 6's reasoning to r2 as well and is reported for completeness, not as the verdict of record.

## Provenance, before any scoring

All six seeds (42, 43, 44, 100, 101, 102) and their twelve `R2_ARMS` each (72 arm files, 72 `.prov.json`
sidecars), plus `offcheck_counterfactual.json` (+ its `.prov.json`), were copied byte-for-byte from the primary
checkout's untracked `research/findings/raw/_sleep_replay_capture_r2/` into this worktree at the same relative
paths (`diff -rq` against the source confirmed an exact match after copy); nothing in the primary checkout was
moved or deleted. The six outer `seed{N}.json` combine files carry no `.prov.json` sidecar by design (this
runner's seed-level combine step is not itself a `research.runners` provenance-stamped call); every one of the 72
per-arm `.prov.json` sidecars records `git_sha: 50c791bf9215df97404a9bf05e3ce40a9a257606` (the revision named in
the task), `source_kind: git_archive`, `git_dirty: false`, and both `source_manifest_verified_at_start` and
`source_manifest_verified_at_exit` true -- checked programmatically over all 72 files, zero exceptions.

Job liveness, checked before scoring: `ssh -n -F research/queue/.pool_ssh_config pool1|pool2|pool41|pool42
'ps -eo etimes,args | grep _da_tag_capture_chat_probe'` returned no `--family r2` process on any of the four pool
nodes (pool1/pool2: no match at all; pool41/pool42: three unrelated `--family fi` workers for seeds 100/101/102,
a later family, not this one). Every seed row is complete: all 12 `R2_ARMS` present, 0 arm errors, valid JSON, no
truncation.

The registered r2 code (`R2_ARMS`, `grade_seed_r2`, `aggregate_r2` in
`research/runners/_da_tag_capture_chat_probe.py`) is byte-identical between the pinned revision `50c791bf9` and
this worktree's checkout of `main` (`13dfae67f`): extracted and diffed directly (not inferred from a changelog),
0 bytes differ in any of the three. The diff between the two revisions on this file is confined to the later
`arc` and `fi` families (Amendments 4-6), which append new arms/graders and do not touch `R2_ARMS`,
`grade_seed_r2` or `aggregate_r2`. `--selftest` (92 checks, no brain) also passes, including the r2-specific unit
cases -- notably `[PASS] r2 grade: d3w_rc=abstain -> seed_verdict UNDEFINED`, the exact designed case that fires
on two of the six real seeds below.

## Scoring: the registered `--aggregate` combine

```
.venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --family r2 \
    --aggregate research/findings/raw/_sleep_replay_capture_r2
```

`research/findings/raw/_sleep_replay_capture_r2/aggregate.json` (+ its freshly stamped `.prov.json`):
`"verdict": "NO-GO"`, `"n_go": 0`, all six seeds present (`seeds` complete), `"signflip_p_downscaling_fade": 1.0`,
`"long_delay_verdict_counts": {"NOT-RESCUED": 6}`. `aggregate_r2` does not itself emit a `tools.verdict.Verdict`
`preconditions` block, so `gates/verdict-preconditions` blocks a bare verdict on a newly-added artifact; a
`preconditions` list was added to this worktree's `aggregate.json` after the fact, built only from values already
computed (seed completeness, zero arm errors, gamma-isolation consistency, the code-identity check above, the
per-arm provenance check above) -- no gate logic and no verdict computation changed, and the list is checked
(`vp.check(...)` returns `[]`).

<!--derived-->
| seed | P1 | LD_verdict (item 1) | P2 | SHY1 fades | SHY2 salient | SHY3 remention | SHY4 no confab | seed_verdict (item 2) |
|---|---|---|---|---|---|---|---|---|
| 42 | correct | NOT-RESCUED | held | **FAIL** (kept) | pass | pass | pass | **NO-GO** |
| 44 | correct | NOT-RESCUED | held | **FAIL** (kept) | pass | pass | pass | **NO-GO** |
| 100 | correct | NOT-RESCUED | held | **FAIL** (kept) | pass | pass | pass | **NO-GO** |
| 102 | correct | NOT-RESCUED | held | **FAIL** (kept) | pass | pass | pass | **NO-GO** |
| 43 | correct | NOT-RESCUED | **FAILED** (`d3w_rc`=abstain) | n/a | pass | fail | pass | **UNDEFINED** |
| 101 | correct | NOT-RESCUED | **FAILED** (`d3w_rc`=abstain) | n/a | pass | pass | pass | **UNDEFINED** |

Every seed's `G0_null_clean`, `I_LD_sleep_after_waking_and_lesion_held`, `I_SHY_three_nights_and_scaling_as_armed`
and `G_isolation_gamma_consistent` hold (6/6), and `n_arm_errors` is 0 in all 6 -- the instrument ran cleanly; the
UNDEFINED and NO-GO readings are outcome results, not instrument failures.

**On the four seeds where P2 held (42, 44, 100, 102), SHY1 fails cleanly and unanimously**: the weak,
never-re-mentioned fact was still recalled after three nights of `BRAIN_SLEEP_DOWNSCALING`, exactly the pattern
Amendment 1 flagged as uncertain and Amendment 6 later explains biologically (three nights of *no other
learning* is the minimum-interference condition; the de Vivo 2017 constant, applied without a later-learning
companion process, does not erase a trace in that condition). SHY2 (salient survives), SHY3 (re-mention protects,
3/4) and SHY4 (no confab) hold on every defined seed; seed 43 additionally fails SHY3 REPORTED-adjacent (not
gating item 2's UNDEFINED status, since P2 already forces UNDEFINED there, but recorded for completeness -- see
below).

**On seeds 43 and 101, `d3w_rc` (the no-downscaling control) itself reads `abstain`**: the weak fact does not
survive three idle nights even with `BRAIN_SLEEP_DOWNSCALING` off. P2 (`grade_seed_r2`'s own pre-registered
precondition -- "the weak fact must survive three nights WITHOUT downscaling, or a fade under downscaling is not
attributable to it") fails, so `--selftest`'s own designed case (`d3w_rc=abstain -> seed_verdict UNDEFINED`) is
exactly what fires; UNDEFINED here is the instrument correctly refusing to attribute a fade it cannot isolate,
not a failed gate. **This is never scored as a 0 or as a NO-GO** -- `seed_verdicts` in `aggregate.json` records
it distinctly as `"UNDEFINED"`, separate from the four `"NO-GO"` rows.

## Why P2 fails on two seeds (REPORTED, not gated)

<!--derived-->
`d3w_shy_a`'s own increment-to-baseline ratio at recall is *lower* on seeds 43 (0.6337) and 101 (0.6548) than on
any of the four P2-holding seeds (0.7364-0.9978), yet `d3w_rc` (no downscaling at all) still reads `abstain` on
those two while its own ratio is 1.0842 (43) and 1.1488 (101) -- both *above* 1.0, i.e. the fact's synaptic trace
is, if anything, stronger than baseline at recall. The failure is not a weak trace; it is a
readout miss on the composer's cleanup margin (unlike the fake-substrate design sweep's ratio-threshold heuristic
cited in the prereg, the real brain's recall correctness is not a monotone function of stored magnitude alone).
This is a REPORTED observation, outside every item-2 gate, and it does not change any seed's status; it is
recorded so the P2-failure is not mistaken for "the fact decayed."

## Item 1 (long delay, REPORTED): NOT-RESCUED 6/6

`LD_verdict` reads `NOT-RESCUED` on every seed (`long_delay_verdict_counts: {"NOT-RESCUED": 6}`), matching the
prereg's own a-priori prediction. `R_at_sleep_onset` for `ld_rc` ranges 0.004059 (seed 100) to 0.184615 (seed
102); `da_swr` ranges 0.503004 to 0.636615. Seed 42's read, 0.005609, matches the value Amendment 1 and Amendment
4 both already cite from the seed-42 smoke, confirming the 6-seed gate row reproduces the earlier de-risk exactly
on that seed. `ld_ledger_off` (today's production default, ledger off) recalls the fact correctly on every seed
(the baseline this route is trying to beat); `ld_norc` and `ld_rc_replaylesion` abstain on every seed, and the
replay-lesion holds on the record itself, not just the env flag -- e.g. seed 42's `ld_rc_replaylesion` epoch
reads `R_eff: [0.0]`, `da_swr: 0.5` (tonic) against the intact `ld_rc` epoch's `R_eff: [0.005609012]`,
`da_swr: 0.504150669` at the identical `R` read.

## Item 3: byte-identical OFF, counterfactual offcheck -- IDENTICAL

`offcheck_counterfactual.json`: `"verdict": "IDENTICAL"`, `"byte_identical_off": true`, `"null_replies_identical":
true`, `"null_store_identical": true`, `head_run` and `counterfactual_run` `replies_sha256` and `store_sha256`
both equal (`491f6b6b...` and `dc0d9108...`), `n_store_conns: 768` on both sides. The run was local (host
`dant123-wk`, not the pool), at HEAD `24380cbe406a8fe074ce3e75d1f67e703cee7fc2` -- the commit that lands
Amendment 2's offcheck construction, and confirmed here to be an ANCESTOR of the pinned gate-row revision
`50c791bf9` (`git merge-base 50c791bf9... 24380cbe4...` returns `24380cbe4`). The three commits between them
(`27fe92022`, `e349a47c6`, `50c791bf9`) touch only `_da_tag_capture_chat_probe.py`,
`onebrain_regression_battery.py`, tests and the prereg doc -- none touches `webapp/da_tag_capture*.py` or
`webapp/sleep_replay_capture.py` (checked via `git show --stat` on the intervening commit), so the offcheck's
IDENTICAL verdict still applies unchanged to the code the pinned six gate rows ran. Six feature commits were
reverse-applied (five 3-way, one zero-context on `webapp/server.py`), `residual_feature_refs: []`, both
`webapp/da_tag_capture*.py` and `webapp/sleep_replay_capture.py` are absent from the counterfactual tree.

## Amendment 3's REPORTED ten-night horizon (never gates item 2)

`d10w_shy` (downscaling on, ten nights): first night not correct = 7 (seed 42), 5 (seed 44), 4 (seed 100), 6
(seed 102) -- inside the 4-10 range Amendment 3 predicted from the three-night trajectory. Seeds 43 and 101 both
read first-night-not-correct = 1, on *both* `d10w_rc` and `d10w_shy` -- consistent with the same P2-adjacent
readout fragility noted above, not a downscaling-specific effect (the no-downscaling horizon control fails
identically). These arms are REPORTED only and excluded from every item-2 gate, error count and UNDEFINED rule,
per the prereg; they do not change any seed's `seed_verdict`.

## Sign-flip p (prereg-registered statistic)

`aggregate.json` reports `signflip_p_downscaling_fade: 1.0` for the registered diff ("`d3w_rc` kept minus
`d3w_shy_a` kept" per seed): `diffs_fade = [0, 0, 0, 0, 0, 0]`. On the four P2-holding seeds both arms are kept
(diff 0, since SHY1 fails -- `d3w_shy_a` is also kept); on the two UNDEFINED seeds both arms already fail without
downscaling (diff 0). The sign-flip test finds **zero evidence across any seed** that `BRAIN_SLEEP_DOWNSCALING`
removes a fact the flag-off arm would otherwise keep -- p = 1.0 is the correct, unfavourable reading, not an
artifact of a small or noisy sample.

## Disclosure: the reading with the three-night criterion (SHY1) withdrawn

Amendment 6 (2026-09-25, merged `df12ec1cc`) withdraws SHY1 as a requirement **for the `fi` family**, on the
biological grounds that a three-night, no-later-learning protocol is the minimum-interference condition and
predicts little to no forgetting (Wixted 2004; Villarreal et al. 2002; Rivera-Lares et al. 2022). It states
explicitly: *"The r2 rules are not edited: `grade_seed_r2` still computes SHY1, and any r2 row reads what it
read... It does not show a defect, and it is not re-scored as a pass."* **This finding follows that instruction:
the verdict of record above does NOT withdraw SHY1 and does NOT re-score any r2 row.**

For transparency, applying Amendment 6's reasoning to r2's own `core` computation (dropping SHY1, keeping
SHY2-SHY4) as a hypothetical, non-registered alternative: on the four P2-holding seeds (42, 44, 100, 102), SHY2,
SHY3 and SHY4 all hold, so all four would read item-2 GO under that alternative reading. Seeds 43 and 101 remain
UNDEFINED regardless -- their UNDEFINED status comes from P2 (the fact not surviving three nights even WITHOUT
downscaling), which SHY1's withdrawal does not touch. Applying the registered family-level roll-up rule ("GO iff
all six seeds GO") to this alternative per-seed reading, **the 6-seed roll-up is still not GO**: 2 of 6 rows are
UNDEFINED, so `n_go` (4) is not 6 either way. Under both readings -- as registered, and with SHY1 withdrawn -- the
family-level r2 item-2 verdict is not GO. The two readings agree on the headline (not GO) and disagree only on
whether the 4 P2-holding seeds are best read as "clean failures of an as-registered criterion" (verdict of
record) or "clean passes of a corrected criterion, on the seeds where the correction is meaningful" (Amendment
6's logic, unregistered for r2). Both facts are true and are stated here without picking one as "the" verdict for
r2, per the task's request to disclose them separately.

## Lesion checks (I_LD / I_SHY, held at measurement, not assumed from the env)

Item 1's replay lesion (`ld_rc_replaylesion`) holds on the record on every seed: `R_eff` all-zero and `da_swr`
pinned to the tonic reference (0.5) against the intact `ld_rc` arm's non-zero `R_eff` and seed-varying `da_swr`
(e.g. seed 42: lesioned `R_eff: [0.0]`, `da_swr: 0.5` vs intact `R_eff: [0.005609012]`, `da_swr: 0.504150669`, at
an identical read `R: [0.005609012]` on both -- confirming the read still runs and only its downstream effect is
cut). Item 2's `I_SHY_three_nights_and_scaling_as_armed` confirms every three-night arm ran exactly 3 epochs, none
with a `no_reader`, and each carries a `shy_scale` record if and only if `BRAIN_SLEEP_DOWNSCALING` was set -- on
every one of the 6 seeds.

## What this does and does not show

Item 1 shows the same boundary the rc-family finding already reported: a fact told ~4 h before sleep is not
rescued by the one-epoch route, consistent with a 2-3 h capture window (Kandel ch.54). Item 2 shows that adding a
literature-magnitude nightly downscaling pass (de Vivo et al. 2017's ~18%, R_i-protected) does **not** erase a
weak, never-re-mentioned fact within three nights on this substrate, on every seed where the precondition for
measuring that at all was met (4/6) -- and shows **zero** sign-flip evidence for any erasing effect across all 6
seeds. It does **not** show that downscaling does nothing: Amendment 6's own later `fi`-family work (not scored
here) found that the missing companion process is *later learning*, not more nights of the same empty protocol,
and that load-dependent renormalization (a different mechanism, `BRAIN_SLEEP_LOAD_RENORM`) does erase the fact
under that companion process. It does **not** show what causes seeds 43/101's baseline (no-downscaling) failure
to retain the weak fact for three nights -- that is a REPORTED, unexplained source of per-seed variance in this
battery, not a downscaling effect (both `d3w_rc` and `d10w_rc`, downscaling OFF, lose the fact by night 1 on
those two seeds). Item 3 shows the feature is byte-identical off at the code level the pinned gate rows ran, not
that the feature is well-calibrated when on.

## Flip candidacy

`BRAIN_SLEEP_DOWNSCALING` is not a flip candidate from this finding under either reading. As registered, item 2
reads NO-GO 0/6 (this finding's verdict of record). Under the Amendment-6-style alternative reading, the
family-level roll-up is still not GO (2/6 UNDEFINED). Against the owner's bar -- 6-seed GO, a SOUND adversarial
review, no regression in the combined battery with the flag ON, and production-default validation -- **no leg of
the bar is cleared**: there is no 6-seed GO under any reading scored here, no adversarial review of this scoring
has run, no no-regression battery has been run with `BRAIN_SLEEP_DOWNSCALING=1`, and no production-default
validation has been attempted. This finding does not flip any default.

## Artifacts

- `research/findings/raw/_sleep_replay_capture_r2/seed{42,43,44,100,101,102}.json` -- the six registered gate
  rows, each with `family=r2`, 12 arms, 0 arm errors, and the runner's own stored `gates` (re-graded here and
  found identical).
- `research/findings/raw/_sleep_replay_capture_r2/seed{42,43,44,100,101,102}/*.json` (+ `.prov.json` sidecars) --
  the 72 per-arm `onebrain_regression_battery.py --worker` outputs each seed's row is assembled from; every
  sidecar records `git_sha=50c791bf9215df97404a9bf05e3ce40a9a257606`, `source_kind=git_archive`, both manifest
  checks true.
- `research/findings/raw/_sleep_replay_capture_r2/offcheck_counterfactual.json` (+ `.prov.json`) -- the item-3
  byte-identical-OFF counterfactual, run locally at an ancestor of the pinned revision (`24380cbe4`), verdict
  IDENTICAL.
- `research/findings/raw/_sleep_replay_capture_r2/aggregate.json` (+ `.prov.json`) -- the registered `--aggregate`
  combine, computed fresh in this worktree with the command shown above, with a `preconditions` block added for
  `gates/verdict-preconditions` (family-level instrument-integrity checks only; see the field's own
  `preconditions_note`).
