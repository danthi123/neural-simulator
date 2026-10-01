---
type: finding
status: go
claim_check: measured
date: 2026-09-29
lane: load-bearing
mechanism: pattern completion in both replay routes (webapp/replay_completion.py, BRAIN_AWAKE_REPLAY_COMPLETION /
  BRAIN_SLEEP_REPLAY_COMPLETION, both default OFF) -- an assembly-coded, unanimous-ignition item competition
  reinstates a block's stored agent/action/patient from the same substrate read the margin-only route already
  takes, replacing the margin-only rescue (webapp/awake_replay_capture.py, webapp/sleep_replay_capture.py) with
  a completed one wherever the reinstated pattern still resolves
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md (Amendment 8 + Addendum 8a,
  `--family arcc`, ARCC_ARMS, gates ARCC1-ARCC7, REPORTED `completion_load_bearing` /
  `awake_completion_needed` / `night_completion_needed`)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_awake_replay_completion/seed42.json
  - research/findings/raw/_awake_replay_completion/seed43.json
  - research/findings/raw/_awake_replay_completion/seed44.json
  - research/findings/raw/_awake_replay_completion/seed100.json
  - research/findings/raw/_awake_replay_completion/seed101.json
  - research/findings/raw/_awake_replay_completion/seed102.json
  - research/findings/raw/_awake_replay_completion/seed*/*.json
  - research/findings/raw/_awake_replay_completion/aggregate.json
verdict: GO 6/6 on the pre-registered per-seed gates ARCC1-ARCC7 (`n_go=6`, all six seeds present, zero
  UNDEFINED, sign-flip p=0.015625 on both registered contrasts). Re-graded from the raw per-arm records with
  BOTH the registered `grade_seed_arcc` / `aggregate_arcc` AND a second, independently-typed re-derivation that
  does not import the registered aggregator; the two agree in full. CAVEAT (CLAUDE.md non-negotiable 4,
  faithfulness-first but falsifiable): pattern completion was NECESSARY for the GO on only 1 of 6 seeds
  (`n_completion_load_bearing=1`, seed 101; the awake route carried that one seed, the night route was never
  independently necessary, `n_night_completion_needed=0`). On the other 5 seeds the pre-existing margin-only
  route (Amendment 4/5) already rescued the fact, so the completion mechanism ran but changed nothing there.
---

# Awake+night pattern completion, arcc family: GO 6/6 -- but the mechanism is load-bearing on a single seed

Scores the `--family arcc` battery registered as Amendment 8 + Addendum 8a of
`research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md` (15 arms: the 12 `ARC_ARMS` plus 3
attribution arms, gates ARCC1-ARCC7 == ARC1-ARC7 of Amendment 4 on the completion-armed arms, plus I4), using
BOTH the runner's own registered combine mode AND a second, independently-typed scorer built for this task,
reading only the raw per-seed/per-arm JSON (never the registered `grade_seed_arc`/`grade_seed_arcc`/
`aggregate_arc`/`aggregate_arcc` functions, and never the `gates` field the seed files already carry).

## Provenance, before any scoring

All six seeds (42, 43, 44, 100, 101, 102) are present with all 15 arms each (90 arm files, 90 `.prov.json`
sidecars); every sidecar records `git_sha: f7693a36f6b352fe805fb7e7c26fc9d34cd72129` (the commit Addendum 8a
pins), `source_kind: git_archive`, `git_dirty: false`, `source_manifest_verified_at_start`/`_at_exit` both
true -- checked programmatically over all 90, zero exceptions (`pin_mismatches: []`, both scoring passes below).
Zero arm errors across all 90 files. `gamma` reads `32.77353587789152` identically on every gated, ledger-ON arm
of all six seeds (the same numpy reference value Amendment 7 quotes).

## Scoring pass 1: the registered `--aggregate` combine

```
python3 -u -m research.runners._da_tag_capture_chat_probe --family arcc \
    --aggregate research/findings/raw/_awake_replay_completion
```

`aggregate.json`: `"verdict": "GO"`, `"n_go": 6`, seeds complete, `signflip_p_rest_rescue_on_vs_off: 0.015625`,
`signflip_p_rest_rescue_intact_vs_awake_lesion: 0.015625` (both diffs `[1,1,1,1,1,1]` over the six seeds --
1/64, matching the prereg's own a-priori "1/64 at 6/6" statement), `n_completion_load_bearing: 1`,
`n_awake_completion_needed: 1`, `n_night_completion_needed: 0`. `aggregate_arcc` does not itself emit a
`tools.verdict.Verdict`-shaped `preconditions` block, so `gates/verdict-preconditions` would block a bare
`"verdict": "GO"` on a newly-added artifact; a `preconditions` list (12 checks, all `ok: true`) was added to
`aggregate.json` after the fact via `tools.verdict.Verdict`, built only from values already independently
re-measured below (seed completeness, zero arm errors, G0/P1/gamma/I1/I2/I3/I4 holding on all six seeds, the
pin, and agreement with the independent re-derivation) -- no gate logic and no verdict computation changed.
`gates/verdict-preconditions` (`check()` + its own `selftest()`) passes clean against the result.

## Scoring pass 2: an independent re-derivation, not the registered aggregator

A second scorer (`grade_seed` / `seed_signflip_p` re-typed from the prereg text, not imported from
`_da_tag_capture_chat_probe.py`) reads only `res["arms"][k]["recall_outcome"]`, `.../env`, `.../errors`,
`.../awake_replay_at_recall/bouts` and `.../sleep_replay_at_recall/epochs` from the six raw `seed*.json` files,
and recomputes every UNDEFINED precondition (G0 null-rebuild, P1, gamma consistency, I1 schedule, I2 lesion
persistence, I3 no-PRP, I4 completion-record integrity, zero arm errors) and all seven ARCC gates from
scratch.

<!--derived-->
| check | requires | independent result |
|---|---|---|
| G0 null-clean (`lr_arc_a`/`_b` rebuild agree in full) | else UNDEFINED | holds 6/6 |
| P1 immediate precondition (`neu_imm_arc` correct) | else UNDEFINED | holds 6/6 |
| gamma consistent across gated, ledger-ON arms | else UNDEFINED | holds 6/6 |
| I1 awake branch ran exactly as scheduled | else UNDEFINED | holds 6/6 |
| I2 every armed lesion (awake/night/DA) still held at measurement | else UNDEFINED | holds 6/6 |
| I3 awake bouts add no PRP of their own | else UNDEFINED | holds 6/6 |
| I4 completion record present iff armed, on every arm | else UNDEFINED | holds 6/6 |
| arm errors, all 90 files | must be 0 | 0 |
| ARCC1 `lr_arc_a` correct AND `lr_noarc` abstain | GATED | PASS 6/6 |
| ARCC2 `lr_arc_lesion` abstain | GATED | PASS 6/6 |
| ARCC3 `ln_arc` abstain | GATED | PASS 6/6 |
| ARCC4 `lr_arc_sleeplesion` abstain | GATED | PASS 6/6 |
| ARCC5 `lr_arc_dalesion` abstain | GATED | PASS 6/6 |
| ARCC6 `lsr_arc_sleeplesion` correct | GATED | PASS 6/6 |
| ARCC7 no confab, any arm | GATED | PASS 6/6 |
| **seed verdict** | GO iff ARCC1-ARCC7 all hold, else UNDEFINED wins | **GO 6/6** |
| **family verdict** | GO iff 6/6 seeds GO | **GO** |
| signflip p, rest-rescue on vs off | -- | 0.015625, diffs `[1,1,1,1,1,1]` |
| signflip p, intact vs awake-lesion | -- | 0.015625, diffs `[1,1,1,1,1,1]` |
| `n_completion_load_bearing` | -- | 1 |
| `n_awake_completion_needed` | -- | 1 |
| `n_night_completion_needed` | -- | 0 |

Every row agrees with pass 1, including the two counts and both sign-flip p-values, computed by a second,
independently-typed combinatorial sign test (`seed_signflip_p`) over the same six per-seed diffs. Two
implementations built from the same prereg text, reading the same raw files through disjoint code paths,
reach the identical verdict, counts and p-values.

## Which seed carries the completion, and which route

`n_completion_load_bearing=1` names seed 101 -- the SAME seed that alone failed ARC1 in the pure margin-only
`arc` family (`research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md`, NO-GO 5/6). The other
five seeds' margin-only route already rescues the fact; completion runs on every gated arm of every seed
(I4 holds 6/6) but changes the OUTCOME on only this one.

<!--derived-->
| seed | `lr_arc_nocomp` (margin only) | `lr_arc_awakeonly` | `lr_arc_sleeponly` | `lr_arc_a` (both) | completion load-bearing? |
|---|---|---|---|---|---|
| 42 | correct | correct | correct | correct | no |
| 43 | correct | correct | correct | correct | no |
| 44 | correct | correct | correct | correct | no |
| 100 | correct | correct | correct | correct | no |
| **101** | **abstain** | **correct** | **abstain** | **correct** | **yes -- awake route** |
| 102 | correct | correct | correct | correct | no |

Seed 101's own awake-bout trajectory (`lr_arc_a`, `research/findings/raw/_awake_replay_completion/seed101.json`):
`R` (the margin read, unchanged by completion) starts at 0.207451321 and stays there (0.207771172 by the 48th
bout) -- almost flat, because the awake completion re-induces `e <- e + R_c(1-e)` every bout instead of letting
`e` decay under the bare margin law; `R_eff` reads 0.774392016 throughout (the completion's own reinstatement
strength, `DISCRIMINATION_G`-cleared). Contrast `lr_arc_nocomp` on the same seed: `R` collapses from
0.207451321 to 0.030467672 by the 48th bout (identical numbers, to 9 digits, to the pure `arc` family's own
NO-GO finding for `lr_arc_a` there), and the night reads `R_eff=0.030533799` -- too weak to clear the SWR-DA
Go boundary, hence `lr_arc_nocomp` abstains. `lr_arc_sleeponly` (night completion, no awake completion) tracks
`lr_arc_nocomp`'s decayed awake trajectory exactly (same `R` sequence, no completion applied to the bouts) and
then completes only at the one sleep epoch -- too late; it also abstains. `lr_arc_awakeonly` (awake completion,
no night completion) holds `R_eff=0.774392016` through every bout and into the night epoch, where the plain
margin-route DA read (now starting from a held, not decayed, trace) clears the boundary on its own -- correct.
This is exactly the mechanism Addendum 8a's Predictions section named as the main open question ("On seed 101
this is uncertain: the night reads the margin at e~0.98 ... near the dev-estimated capture point"): on the one
gate seed where it mattered, the AWAKE route, not the night route, did the rescuing.

## The caveat this finding is required to carry (CLAUDE.md non-negotiable 4)

A GO 6/6 headline would overstate what changed. Faithfulness to the completion mechanism (assembly coding,
unanimous three-role ignition, the discrimination criterion) is not itself evidence that the mechanism was
NECESSARY for this result: on 5 of 6 gate seeds the margin-only route (`lr_arc_nocomp`, i.e. Amendment 4/5's
pre-existing `arc` mechanism) already reads `correct`, so pattern completion there is faithful but redundant --
it runs (I4 confirms every bout/epoch carries a completion record) and reinstates the exact stored triple
(`items` == `resolved` == the fact on every sampled bout, ARCC7 confirms zero confab anywhere), but the
outcome is identical to what the margin-only route already gave. Only on seed 101 is completion load-bearing,
and there only the awake route is independently necessary (`awake_completion_needed=1/6`); the night route
is never independently necessary on any seed (`night_completion_needed=0/6`) -- night-route completion is
consistent with the mechanism (I4 holds, ARCC4/ARCC6 pass) but this battery gives it no seed where it alone
carries the recall. `n_completion_load_bearing=1/6` is therefore the number that belongs beside the GO, not a
subordinate detail: the lesion-verified load-bearing fraction for this capability, on this battery, is 1/6.

## REPORTED arms (never gating; ARCC7 aside)

<!--derived-->
| arm | role | correct count, 6 seeds |
|---|---|---|
| `lr_ledger_off` | today's production default (ledger off), same rest | 6/6 |
| `lq_arc` | hourly rest (4 bouts in 4 h) | 2/6 |
| `lz_arc` | rest only in the last hour (12 bouts, late) | 0/6 |
| `lr_arc_nocomp` | margin-only route, both flags lesioned | 5/6 (abstains only on seed 101) |
| `lr_arc_awakeonly` | awake completion only | 6/6 |
| `lr_arc_sleeponly` | night completion only | 5/6 (abstains only on seed 101) |

`lq_arc` moved from 0/6 (pure `arc` family) to 2/6 here -- REPORTED only, consistent with the prereg's own
"either can now be correct" prediction for the sparse-rest arms once completion can regrow a faint trace
whenever it still selects its items, and it changes no gate.

## Terms (docs/TERMS.md)

"GO" is used only for the gate's own per-seed and 6-seed verdicts, never lifted from a REPORTED arm. "Lesion"
is used only for `lr_arc_lesion` / `lr_arc_sleeplesion` / `lr_arc_dalesion` / `lr_arc_nocomp`
(`BRAIN_REPLAY_COMPLETION_LESION`), each verified by I2 to still hold on the record at the moment of
measurement (R_eff pinned to 0 or the DA read pinned to the tonic reference), not merely from the env flag.
"Consolidation" is not used: both routes re-potentiate the SAME store the fact was already written to (the
prereg's own usage, restated in Amendment 8). "Load-bearing" is used only in the precise, prereg-defined sense
(`completion_load_bearing` = the seed's own outcome flips from abstain to correct when the mechanism is
lesioned out), matching CLAUDE.md's "lesion-verified load-bearing fraction" usage, never as a synonym for
"the mechanism ran" or "the mechanism is faithful".

## Flip candidacy

`BRAIN_AWAKE_REPLAY_COMPLETION` / `BRAIN_SLEEP_REPLAY_COMPLETION` are default OFF and stay OFF. This finding
establishes the registered 6-seed GO and the load-bearing caveat; it does not itself run an adversarial review,
a no-regression battery with both flags ON, or a production-default validation, so it does not clear the
owner's flip bar on its own. The 1/6 load-bearing fraction is the number a flip decision should weigh: the
capability, on this battery, rescues one gate seed's realization of one fact against one alternative mechanism
(the margin-only route), which is a real but narrow result -- not evidence that completion is broadly necessary
for awake/night replay-driven recall.

## Artifacts

- `research/findings/raw/_awake_replay_completion/seed{42,43,44,100,101,102}.json` -- the six registered gate
  rows, `family=arcc`, 15 arms each, 0 arm errors.
- `research/findings/raw/_awake_replay_completion/seed{42,43,44,100,101,102}/*.json` (+ `.prov.json`) -- the 90
  per-arm outputs, all provenance-verified at `f7693a36f6b352fe805fb7e7c26fc9d34cd72129`, `source_kind:
  git_archive`.
- `research/findings/raw/_awake_replay_completion/aggregate.json` (+ `.prov.json`) -- the registered
  `--aggregate` combine, computed fresh in this worktree, plus the `preconditions` block added after the fact
  for `gates/verdict-preconditions` (values only, no re-scoring) and a `caveat_completion_load_bearing` field
  mirroring this finding's caveat section.
