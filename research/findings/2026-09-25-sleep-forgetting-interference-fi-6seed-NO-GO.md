---
type: finding
status: no-go
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: night-load-dependent synaptic renormalization keyed to LATER LEARNING (webapp/sleep_replay_capture.py,
  BRAIN_SLEEP_LOAD_RENORM, default OFF) -- after the night's sleep-replay-capture reactivation, each managed
  block's increment is scaled by 1 - delta (1 - R_i), where delta = dW/W is the fraction of the store's total
  synaptic strength the preceding wake actually added (not a fixed constant), and R_i is the block's own
  reactivation read (r2's protection), so an unrehearsed weak fact fades as a function of how much else is
  learned afterward rather than of nights alone
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md (Amendment 6, `--family fi`,
  `FI_ARMS`, gates FI1-FI7)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_forgetting_interference/seed42.json
  - research/findings/raw/_sleep_forgetting_interference/seed43.json
  - research/findings/raw/_sleep_forgetting_interference/seed44.json
  - research/findings/raw/_sleep_forgetting_interference/seed100.json
  - research/findings/raw/_sleep_forgetting_interference/seed101.json
  - research/findings/raw/_sleep_forgetting_interference/seed102.json
  - research/findings/raw/_sleep_forgetting_interference/seed*/*.json
  - research/findings/raw/_sleep_forgetting_interference/aggregate.json
verdict: NO-GO 3/6 on the pre-registered per-seed gates FI1-FI7 (`n_go=3`; all six seeds present, zero
  UNDEFINED). Seeds 42, 44 and 102 read seed_verdict GO. Seeds 43 and 101 fail FI1 and FI4 on a night-1
  sleep-replay-capture-margin miss for the WEAK telling (`_DATC_WEAK`, the fact told last) that is present with
  ZERO later learning -- outside the load-renormalization mechanism this family exists to measure. Seed 101 ALSO
  fails FI5: its salient telling IS captured, but that block's own reactivation read collapses across the week
  (0.291 -> 0.010 by night 6) and the `1 - delta(1-R)` protection collapses with it, losing the fact by morning 5.
  Seed 100 fails FI6: the twice-re-mentioned fact is lost by night 4 under the continuing 3-facts/day dose, and
  the failing read matches an older, weaker-margin duplicate block rather than the newest, strongest one.
  Re-graded from the raw artifacts with the registered `grade_seed_fi` / `aggregate_fi`, unmodified.
---

# Sleep forgetting-interference (`fi` family): NO-GO 3/6, as registered

Scores the `--family fi` battery pre-registered as Amendment 6 of
`research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md`, using the runner's own registered combine
mode over the six gate-seed rows, unmodified. Amendment 6 replaced r2's withdrawn three-night downscaling
criterion (SHY1) with a mechanism whose amplitude is measured from the store's own load rather than fixed at de
Vivo et al. 2017's constant, on the grounds that the missing companion process the constant stood in for is
**later learning**, not more idle nights.

## Provenance, before any scoring

All six seeds (42, 43, 44, 100, 101, 102) and their nine `FI_ARMS` each (54 arm files, 54 `.prov.json` sidecars)
were already present, committed on `main` at `fe1066f64` per the task, in the primary checkout's
`research/findings/raw/_sleep_forgetting_interference/`. Checked programmatically over all 54 per-arm sidecars:
every one records `git_sha=2def39c76cb4d81ca1b907399d3e9482f62bfe42` (a single value -- no seed and no arm ran at
a different revision), `source_kind=git_archive`, `git_dirty=false`, and both `source_manifest_verified_at_start`
and `source_manifest_verified_at_exit` true. That revision is `prereg(sleep-replay-capture): Amendment 6 -- the
missing companion is later learning; SHY1 withdrawn; the fi family`, an ancestor of this branch's `main` base
(`ce570ab7b`) and a descendant of the amendment's own governing commit `cbeccb54c`
(`git merge-base --is-ancestor` confirms both). Every one of the 54 raw arm records carries `"errors": []`
(checked directly, not read off a summary field) -- zero arm-level errors across the whole family.

The registered code (`FI_ARMS`, `grade_seed_fi`, `aggregate_fi` in `research/runners/_da_tag_capture_chat_probe.py`)
is byte-identical between the pinned revision and this scoring checkout: `git diff` between the two revisions on
that file returns nothing at all. `research/runners/onebrain_regression_battery.py` is **not** byte-identical as a
whole file between the two revisions -- but the only difference is a 5-line comment added to an unrelated
`FACULTY_PROBES` row (`affect-drives-response`, documenting the 2026-09-25 affect-marker-surface retirement); it
does not touch `_FI_FACTS` (line 338) or the `fiv`/`fil`/`fih`/`fis`/`fir` group/turn definitions (`_fi_group`
calls at line 362), which are unchanged. `aggregate_fi` itself does not import `onebrain_regression_battery.py` at
all -- it re-grades the already-recorded `res["arms"]` of each `seed*.json` with the current `grade_seed_fi` -- so
this comment-only diff cannot have affected the scoring below in any case.

**Undeclared-until-now deviation from the registered command.** Amendment 6 registers `--workers 1` (line 798 of
the prereg); every one of the six seeds' stored `argv` reads `--workers 3` instead (checked directly in each
`seed*.json`). `--workers` only bounds how many of an arm's `ThreadPoolExecutor`-dispatched subprocesses run
concurrently (`run_seed`, `research/runners/_da_tag_capture_chat_probe.py:258`); each of the 9 `FI_ARMS` is still
its own subprocess regardless of the count, and `G0_null_clean` and `I3_identical_through_night1` both hold
byte-for-byte above, so this deviation is not shown to have changed any result -- but it is a real deviation from
the registered command and is declared here, not silently absorbed.

## Scoring: the registered `--aggregate` combine

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --family fi \
    --aggregate research/findings/raw/_sleep_forgetting_interference
```

Re-running this against the existing raw files reproduces, field for field, the aggregate the task described as
already committed-but-blocked: `"verdict": "NO-GO"`, `"n_go": 3`, `"seed_verdicts": {"42": "GO", "43": "NO-GO",
"44": "GO", "100": "NO-GO", "101": "NO-GO", "102": "GO"}`. `aggregate_fi` does not itself emit a
`tools.verdict.Verdict` `preconditions` block, so `gates/verdict-preconditions` blocks a bare verdict on a
newly-added artifact. A `preconditions` list was added to `aggregate.json` after the fact, built only from values
already checked above (seed completeness, zero arm errors, per-seed instrument-integrity flags, the code-identity
check, the per-arm provenance check) -- no gate logic and no verdict computation changed, and the list is checked
clean by `tools/gates/verdict_preconditions.py` directly (`_check_one` returns `[]`).

## Per-seed gate table

<!--derived-->
| seed | FI1 vacuum | FI2 erases by n7 | FI3 dose-ordered | FI4 lesion keeps it | FI5 salient kept | FI6 remention kept | FI7 no confab | seed_verdict |
|---|---|---|---|---|---|---|---|---|
| 42 | pass | pass | pass | pass | pass | pass | pass | **GO** |
| 44 | pass | pass | pass | pass | pass | pass | pass | **GO** |
| 102 | pass | pass | pass | pass | pass | pass | pass | **GO** |
| 43 | **FAIL** | pass | pass | **FAIL** | pass | pass | pass | **NO-GO** |
| 101 | **FAIL** | pass | pass | **FAIL** | **FAIL** | pass | pass | **NO-GO** |
| 100 | pass | pass | pass | pass | pass | **FAIL** | pass | **NO-GO** |

Every seed's `G0_null_clean`, `P1_immediate_precondition`, `G_isolation_gamma_consistent`,
`I1_seven_nights_dose_delivered`, `I2_load_lesion_held` and `I3_identical_through_night1` hold (6/6), and
`n_arm_errors` is 0 in all six -- the instrument ran cleanly on every seed; every FAIL above is an outcome result,
not an instrument failure.

## Why seeds 43 and 101 fail FI1/FI4: a night-1 sleep-replay-capture-margin miss, not encoding, not later-learning erasure

`fiv_lr` (FI1: nothing else is ever learned in this arm) reads `abstain` on **every one of its seven mornings** on
both seeds, starting at night 1 -- before a single later fact has been told. Because I3 holds, `fil_lr`,
`fih_lr_a`, `fih_lr_b` and `fir_lr` are identical to `fiv_lr` through night 1 on these two seeds and so already
read `abstain` at the first morning too. `fih_lr_lesion` is **not** one of I3's checked arms (`same = (fiv_lr,
fil_lr, fih_lr_a, fih_lr_b, fir_lr)`), so I3 says nothing about it -- but it independently shows the identical
night-1 signature checked below (`z_mean` `3.8088625378623516e-10` / `1.0692804587972005e-11`, `R` `0.168587758` /
`0.035599825`, `abstain` at recall 1 on both seeds), because the lesion only changes what a night does to the
*applied* renormalization delta (it sets the applied scale to 1 and reads-but-does-not-apply the load); it does
not touch the SWR-epoch capture step that sets `z`. FI4 fails for the same upstream reason as FI1 -- the
load-renormalization edge genuinely is irrelevant to a fact that was never captured, but that is shown by the
lesion arm's own night-1 record, not inferred from I3.

**Ruling out an encoding miss (for the weak telling's storage, not its immediate recall).** `n_managed_blocks_by_night`
for `fiv_lr` is `1` on every one of the seven nights on both seeds -- the ledger created exactly one managed block
for the fact, and it never disappears. The block's own increment-to-baseline ratio at recall 1 is
`0.9561444027828692` (seed 43) and `0.9920212521528132` (seed 101) -- both close to the fresh-fact range the four
seeds where the fact was captured show at the same point (`1.305007482661105`, `1.0131071147025406`,
`1.0254155897799755`, `1.209882425794491` for seeds 42, 44, 100, 102 respectively). The fact was told and stored
as one block with a substantial weight; it was not "never stored." This does **not** show the block was
immediately recallable: `P1_immediate_precondition` (`neu_imm_fi`, group `datni`) holds on both seeds, but that
arm uses the NEUTRAL telling (`_DATC_NEUTRAL`, `datni_recall`), not the WEAK telling `fiv_lr`/`fil_lr`/`fih_lr_*`/
`fir_lr` use (`_DATC_WEAK`). P1 shows the encoding-and-immediate-recall pathway works at all on these two seeds'
builds; no arm in this family probes immediate recall of the weak telling itself, so "immediately recallable" is
not directly measured here.

**What actually differs at recall: the SWR reactivation margin at night 1, whether the block ever crosses into
late-phase capture, and what that means for what is EXPRESSED by morning.** Every managed block records `z_mean`,
the fraction of the block's synapses that have crossed into the late (protein-synthesis-independent, persistent)
phase. At recall 1, `z_mean` is `1.0000000000508993` (seed 42), `0.999999999981423` (seed 44),
`1.000000000015095` (seed 100) and `1.0000000000483527` (seed 102) -- captured. On seeds 43 and 101 it is
`3.8088625378623516e-10` and `1.0692804587972005e-11` -- effectively zero: the block never left early phase. The
sleep epoch's own reactivation read, `R`, is `0.356535268` / `0.310521057` / `0.320028458` / `0.423728264` on the
four seeds where the fact was captured, against `0.168587758` (seed 43) and `0.035599825` (seed 101) -- roughly
half to a twelfth as large; the resulting SWR-coupled DA (`da_swr`) is correspondingly lower (`0.624754941` and
`0.526343871` against `0.729785582`-`0.813558916` on those four seeds, low end at seed 44 not seed 42), and the
D1-gated capture that Amendment 6's mechanism runs through never crosses whatever threshold turns early-phase
expression into a permanent trace (`a_eff_mean` `0.161502711` / `0.029512006`, `tag_rep_mean` -- the block's own
replay tag -- `0.15702335749144491` / `0.03306745477406034`, both seeds).

Because `z` never flips, the ratio quoted above (`0.956`/`0.992`, `|inc|/|base|` of the *stored* increment) is not
what is expressed at the recall turn. `weight_factor(blk) = e + z * (1 - e)` (`webapp/da_tag_capture.py:493-495`),
with `e = exp(-(t - t_w) / tau_early_h)` and `tau_early_h ≈ 1.5` h: at the ~24 h morning read, `e ≈ 1.1e-7` on a
block with `z ≈ 0`, so the increment's expressed contribution to the stored weight is negligible by morning -- the
early-phase trace itself has decayed away, on top of never having been tagged for late-phase persistence. This is
consistent with the actual chat-turn reply at recall 1 on seed 43: `"(I'd been mulling over dog.) As for it — I
don't know about that."`, `matched_fact_index: null` against `n_facts_scanned: 6`, and the spiking read-out firing
`2/6880` readout neurons (`frac_fired = 0.00029069767441860465`) -- essentially nothing. Seed 101's `fiv_recall1`
reads the same shape. This is a **night-1 sleep-replay-capture failure** (the reactivation read never gets the
block's synapses to cross into persistence) **followed by ordinary early-phase decay** of whatever was written --
not an encoding miss, and not "closest to an immediate-read failure" in the sense of a readout-side margin acting
on an intact trace: the trace itself has decayed in expression by the time recall is asked.

**This is not "seeds 43 and 101 cannot recall the fact"; it is specific to the weak telling's margin.** The
*salient* telling (`fis_lr`, the same fact inside surprising news) reaches `R = [0.388730055]` (seed 43, inside
the four captured seeds' weak-telling night-1 range above) and `R = [0.290963714]` (seed 101, BELOW that range)
at its own night-1 epoch, and both cross into capture (`z_mean = 1.0000000001010352` and `1.0000000000866132`);
`fis_lr` is correct at recall 1 on both seeds. Seed 101's capture despite the lower `R` tracks a larger salient
tag, not a higher reactivation read: the salient block's `tag0_mean` is `2.1228416378885058` against the weak
block's `1.0` on the same seed (seed 43: `2.2027627035027137` vs `1.0`). The failure is a margin specific to the
weak telling's smaller tag on THIS seed's heterogeneous population, not a broken seed: raise the drive (salience,
via a larger tag) and the same mechanism captures normally. It is a property of the *sleep-replay-capture* route
(`BRAIN_SLEEP_REPLAY_CAPTURE`, Amendment 1-5), not of `BRAIN_SLEEP_LOAD_RENORM` (this amendment).

**This same signature was already reported once, for the same two seeds and the same weak telling, in a sibling
family -- and this finding supersedes that reading.** `research/findings/2026-09-25-sleep-replay-capture-r2-NO-GO-6seed.md`
("Why P2 fails on two seeds") found `d3w_rc` (r2's own no-downscaling control, same content, no later learning at
all) reading `abstain` on seeds 43 and 101 while its own increment-to-baseline ratio at recall was *above* 1.0 on
both, and named it "a readout miss on the composer's cleanup margin", explicitly not a decayed trace. `d3w_rc`'s
own `z_mean` at recall is `8.674002122648018e-13` (seed 43) and `1.5650274174830834e-12` (seed 101) -- the same
near-zero capture state measured here, with the same night-1 `R` (`0.168587758` / `0.035599825`, identical to
`fiv_lr`'s, as I3-style identity through night 1 predicts for a shared seed and telling). The `z_mean`/`R`/
`weight_factor` breakdown above contradicts "not a weak trace ... not mistaken for the fact decayed": the trace
genuinely never crosses into persistence and its early-phase expression genuinely decays by the recall turn, so
this finding's account **supersedes** r2's "readout margin" reading rather than merely refining it. Seeds 43 and
101 failing FI1 is a **replication** of an already-reported per-seed fragility in this route's weak-telling
capture, not a new failure mode of the load-renormalization mechanism.

## Why seed 101 also fails FI5: the salient block's own protection collapses across the week

Seed 101's `fis_lr` is correct at recall 1 (above) but is not the seed's only FI failure: it also fails FI5
(salient kept under the dose), the one gate this finding's table shows failing on 101 and not on 43. This is a
distinct mechanism from the night-1 capture-margin miss above -- the block IS captured (`z_mean ≈ 1` from night
1) -- and it plays out entirely within the load-renormalization mechanism this amendment adds, not upstream of
it. The block's own reactivation read collapses across the week: `R = [0.290963714, 0.260448706, 0.14836735,
0.023063055, 0.022094502, 0.009963793, 0.032167699]` for nights 1-7. Because the protection is `1 - delta * (1 -
R)`, a falling `R` weakens the protection every subsequent night regardless of `delta`: on night 2, `shy_scale =
0.692666663` at `delta = 0.415567303`, consistent with `1 - 0.415567303 * (1 - 0.260448706) ≈ 0.693`. Once `R`
collapses (night 4 onward, `R ≈ 0.02`), the block is renormalized at close to the full `delta` every night like an
unprotected trace, and its ratio (`inc_mag/base_mag`) falls accordingly: `2.0236041350102183` (night 1) ->
`1.0611021373623823` (night 3) -> `0.7063632341370527` (morning 5), where the reply first fails (`matched_fact_index: 5`, `verified: false` -> `abstain`) --
matched but not confident enough to pass verification. The same matched-but-unverified pattern repeats at morning
6, and by morning 7 the read finds nothing at all (`matched_fact_index: null`). FI5 was pre-registered as "salient
telling, same dose -> kept"; on this one seed the salience tag captures the block but does not keep its own
reactivation read high enough for the protection formula to hold it through a full week of continuing load -- a
genuine limit of the built protection under this amendment's own mechanism, not a capture-margin artifact.

## Why seed 100 fails FI6: re-mention's own protection erodes under the continuing dose

`fir_lr` re-mentions the fact ("the cat chases the ball") after nights 1 and 2, each time writing a fresh block,
before that day's three later facts. `n_managed_blocks_by_night` for seed 100 is `[1, 5, 9, 12, 15, 18, 21]` --
1 (the original) at night 1, +1 (re-mention) +3 (day-2 facts) = 5 by night 2, +1 (re-mention) +3 (day-3 facts) = 9
by night 3, then +3 a night through night 7, exactly the registered protocol. `daily_outcomes` are `correct,
correct, correct, abstain, abstain, abstain, abstain` -- lost starting the fourth morning, one to two nights
after the second re-mention. The *tracked original* block's own ratio falls every night: `1.0254155897799755`
(night 1) -> `0.7024895478071427` -> `0.49393965992299305` -> `0.40280562188118535` -> `0.34197805377694906` ->
`0.3015797022920384` -> `0.2691548021017118` (night 7), against a per-night renormalization delta of
`0.1440927`, `0.452880123`, `0.307868533`, `0.188656302`, `0.152351566`, `0.121774295`, `0.110228511`.

That original block alone falling below margin is not the whole story: the second re-mention writes its own block
(store position 5 of the night-4 record, i.e. fact index 10 counting the 5 build-time facts), and at morning 4
its ratio is `0.9785813694572846` -- well inside the range seed 100's OTHER arms read as "correct" at a
comparable point (`fil_lr` correct at `0.6875558289374295`, lost the next morning at `0.6085411651256613`;
`fih_lr_a` already lost by `0.5357732936788184`; `fiv_lr` sits at a constant `1.0254155897799755`; `fis_lr`'s
lowest point, night 7, is still correct at `1.0217791765005546`). The morning-4 read does not use that block: it
matches an OLDER duplicate instead (store position 1, fact index 6, ratio `0.6023277961028686`, from the FIRST
re-mention) -- `activity.matched_fact_index: 6`, `activity.abstained: false`, patient `ball` at confidence `1.0`,
but the reply then fails verification (`verified: false`) and the arm abstains. The same match-then-fail-
verification pattern repeats at morning 5 (matched to the same first-re-mention block, index 6, as morning 4 --
not the original block, which is index 0); mornings 6-7 find no match at all (`matched_fact_index: null`). The
loss on this seed therefore depends on WHICH of the fact's duplicate blocks
the read settles on, not on every duplicate uniformly eroding below a shared recall margin -- the strongest
duplicate (the second re-mention) was still comfortably above that margin when the read failed. This is still a
genuine, mechanism-consistent finding, not an instrument defect (`I2_load_lesion_held` and `n_arm_errors=0` both
hold for seed 100): it shows the re-mention protection is not robust to having several duplicate blocks of the
same fact competing for the read once the load edge keeps eroding each of them, a failure mode the pre-registered
gate did not anticipate and this scoring did not set out to characterize in more detail than "FI6 fails".

## On the three seeds where every FI gate holds, the mechanism behaves as pre-registered -- with earlier loss than predicted

On seeds 42, 44 and 102, `fiv_lr` (0 facts/day) is correct all seven mornings, `fih_lr_lesion` (load edge cut) is
correct all seven mornings, `fis_lr` and `fir_lr` are correct at night 7, no arm confabulates, and the dose
ordering holds: on seed 42 at night 7 the fact's own ratio is `1.305007482661105` (`fiv_lr`, no dose) >
`0.6991167688886667` (`fil_lr`, 1/day) > `0.416855717690678` (`fih_lr_a`, 3/day). `fih_lr_a` (FI2, 3 facts/day) is
lost by night 7 on all three, but earlier than Amendment 6's a-priori prediction of "night 5 or 6": seed 42 loses
it at night 4 (`correct_at_night3: true`), seed 44 at night 3 (`correct_at_night3: false`) and seed 102 at night 5
(matching the prediction). The prereg flagged this exact number as uncertain ("the brain's W, its R for new
facts and its DA-gated write gains differ from the fake"); the brain loses the fact somewhat faster under this
dose than the fake-substrate design sweep anticipated on 2 of 3 GO seeds. `fih_shy` (REPORTED only, r2's fixed
18% constant under the same dose) is lost later on every seed that reaches night 7 with it intact (e.g. seed 42:
night 7, `fact_ratio_by_night[6] = 0.5371897300418954`) -- consistent with the constant ignoring the load and
therefore charging every night the same amount regardless of how much was actually learned.

**Other REPORTED, a-priori predictions, checked over all six seeds, not just the three GO seeds.** Amendment 6
predicted `fil_lr` (1 fact/day) recalled on all seven mornings; it is on 2/6 seeds (42, 102) -- on 44 and 100 it is
lost by night 5 (`0.6085411651256613` and below), and on 43/101 it is `abstain` from night 1 for the same
capture-margin reason as `fiv_lr`. It predicted `fih_shy` first lost on night 7 or later; that holds on 1/6 (42)
-- 44 loses it at night 5, 100 at night 4, 102 at night 6 (43/101 again `abstain` from night 1, so the SHY-specific
prediction is not testable on them). It predicted the brain's night-1 `delta` close to `0.19`; the measured
`fiv_lr` night-1 `delta` across all six seeds is `0.166166284` (42), `0.142042173` (43), `0.142236829` (44),
`0.1440927` (100), `0.141486819` (101), `0.151454139` (102). <!--derived--> That is a 0.141-0.166 band (the min
and max of the six values above), consistently below the fake-substrate sweep's `0.19`. None of these three misses changes any gate's pass/fail (`fil_lr`'s outcome is
REPORTED, not gated; `fih_shy` enters no gate but FI7; the delta is REPORTED), but they are registered predictions
that were not previously reported against the full six-seed data.

The aggregate's own `first_night_not_correct` summary (`aggregate.json`, built by the registered `aggregate_fi`)
carries this field for six of the nine arms (`fiv_lr`, `fil_lr`, `fih_lr_a`, `fih_shy`, `fis_lr`, `fir_lr`); it
does not carry `fih_lr_lesion` or `fih_lr_b`, though the prereg asks for "the first-lost night per seed for each
arm". Read directly from each seed's own (registered, unmodified) `grade_seed_fi(...)["reported"]` rather than
from `aggregate.json`: `fih_lr_b`'s first-lost night is identical to `fih_lr_a`'s on every seed (`4, 1, 3, 3, 1, 5`
for seeds 42/43/44/100/101/102), as G0's null-control equality requires. `fih_lr_lesion`'s is `None` (never lost)
on the four seeds where the fact is captured, and night 1 on 43/101 -- the same capture-margin miss as `fiv_lr`,
independent of the lesion, as shown above. Neither omission changes a gate verdict; both are declared here because
`aggregate.json` itself does not carry them.

## Sign-flip p: diluted by the night-1 miss, not evidence the dose effect is absent

`aggregate.json` reports `signflip_p_vacuum_minus_dose: 0.0625` and `signflip_p_lesion_minus_dose: 0.0625`
(`diffs_vacuum_minus_dose` and `diffs_lesion_minus_dose` both `[1, 0, 1, 1, 0, 1]` over seeds `[42, 43, 44, 100,
101, 102]`), not the 1/64 a clean 6/6 would give. <!--derived--> The two zero entries are seeds 43 and 101: on
both, `fiv_lr` and `fih_lr_a` are already identical (`abstain`) by night 1 for the reason above, so the
night-7 diff is mechanically 0 regardless of whether load-renormalization would have separated them had the fact
survived to be renormalized at all -- a floor effect from the capture-margin miss, not a null result on the
dose effect. On the four seeds where the fact was captured at all (42, 44, 100, 102), the vacuum-minus-dose and
lesion-minus-dose diffs are 1 on all four: wherever this measurement is possible, the dose effect and the
lesion-protection effect both point the predicted direction on every seed.

## Relation to the owner's 2026-09-25 "remember what matters" directive

The owner's directive calls for prioritized retention: important facts and key details kept, minor details
allowed to fade, without RAG. On the 3/6 seeds where every FI gate holds, this family's qualitative shape matches
that directive's shape: an unrehearsed, unimportant fact fades as later learning accumulates (FI2), the rate is
ordered by how much is learned (FI3), and two independent importance signals -- telling it saliently (FI5) or
re-mentioning it (FI6, on 5/6 seeds) -- protect it. That is a real, measured piece of evidence in the directive's
direction. Neither the reactivation read itself nor the amplitude that turns it into a night's renormalization is
brain-computed: `webapp/sleep_replay_capture.py`'s own HOST SHORTCUTS declare `R_i` as "the composer's
decisiveness margin, (peak - runner_up)/peak of the cleanup membrane scores: host arithmetic on a substrate read"
-- so the protection read that this whole family turns on is itself a host computation over a substrate read, not
read directly off the brain's own store. The downstream amplitude (the `dW`/`W` sums, their ratio `delta`, and the
multiply `1 - delta(1-R)`) is declared in the same docstring as host arithmetic over synaptic quantities. Both are
documented shortcuts under the brain-based-only standard, not something this finding should imply is fully
synaptic.

The owner's 2026-09-25 ruling was prompted by exactly the defect this family reproduces: it names the sibling r2
route's result as losing the weak telling "2/6, through a single threshold on the replay read... forgetting that
is not driven by what matters." `fi` reproduces that same defect, unchanged, on the same two seeds (43, 101) --
Amendment 6 does not address it, because it targets the renormalization step downstream of capture, not the
capture step itself.

It does **not** show the directive is implemented. The 6-seed verdict is NO-GO: seeds 43 and 101 fail FI1/FI4
before the renormalization mechanism ever gets a chance to act on the weak telling (a capture-margin fragility in
the upstream sleep-replay route, not this amendment's mechanism); seed 101 ALSO fails FI5 through a failure
internal to this amendment's own mechanism (its salient block's reactivation read collapses across the week and
the protection collapses with it); and seed 100 shows the re-mention protection is not robust to a full week of
continuing load once several duplicate blocks of the same fact compete for the read. The mechanism scored here
distinguishes exactly two categories of "important" (told saliently, or repeated) against one unrehearsed control;
it has no notion of graded importance among many concurrently known facts, no interaction with genuine
content-similarity interference (the told facts here are deliberately dissimilar to the target, per the prereg),
and depends on a capture step (sleep-replay-capture) that this same scoring shows is not yet reliable across seeds
for an ordinary weak fact, nor -- on seed 101 -- reliably protective of a fact it did capture. Whatever downstream
feature the memory entry describes as "waiting for this fix" should keep waiting: this is a directionally
consistent building block, re-graded here as NOT YET a 6-seed-robust one.

## What this does and does not show

FI1 and FI2 together, on the seeds where both are measurable (42, 44, 102), tie the loss to later learning: the
same brain, the same telling, the same first night (I3), diverging only once later facts start. FI4 ties it to
the renormalization edge specifically: cutting it keeps the fact all seven nights on those same seeds. FI3 is a
genuine dose-response on a continuous read on the four seeds where the fact is captured (42, 44, 100, 102), not
just a discrete pass/fail. On seeds 43 and 101, FI3 also reads "pass" by the gate's literal wording (`fiv_lr`'s
ratio still exceeds `fil_lr`'s, which still exceeds `fih_lr_a`'s), but all three are `z ≈ 0`, never-expressed
STORED increments on a block whose weight is already, in effect, indistinguishable from baseline at recall -- a
pass with no behavioral meaning on those two seeds, since the ordering plays out entirely inside a quantity the
composer never reads from. It does **not** show that this family's night amplitude corresponds to any particular
human day (declared already in the prereg: the store's W grows by a block per fact told, so this model's delta
falls faster with accumulated knowledge than a renormalize-the-whole-brain system would, biasing toward
retention). It does **not** show that re-mention is a reliable protection at this dose once several duplicate
blocks of the same fact compete for the read (seed 100), nor that salience is a reliable protection across a full
week once captured (seed 101, FI5). It does **not** show that the sleep-replay-capture route reliably captures an
ordinary weak fact at all (seeds 43, 101) -- a pre-existing property of an earlier amendment, not of the mechanism
this amendment adds. Similarity-dependent interference (A-B, A-C) is not tested here, as the prereg states.

## Honest limits

- This is a re-grading of already-produced raw files with the registered, unmodified grader and aggregator; no
  new brain build ran for this finding.
- The night-1 capture-margin miss on seeds 43/101 is described here with more mechanistic detail (`z_mean`, `R`,
  `da_swr` at the exact epoch) than the prior r2 finding gave, but neither finding identifies WHY those two
  seeds' heterogeneous builds put the weak telling's reactivation margin below the effective threshold while
  the salient telling's (larger tag, on the same seeds) does not; that remains open.
- Seed 101's FI5 failure (the salient block's own reactivation read collapsing across the week) is reported here
  for the first time but not root-caused, and it is not unique to seed 101: the salient block (`fis_lr`) is
  captured (`z ≈ 1`) on all six seeds, not four. Why THIS seed's salient block's `R` falls from `0.290963714` to a
  minimum of `0.009963793` (night 6) is not investigated -- and at night 7, `R` reads `0.315911916` (seed 42),
  `0.048512264` (43), `0.2689885` (44), `0.328374885` (100), `0.032167699` (101), `0.424426312` (102): only seeds
  42, 100 and 102 stay clearly above `0.27`, seed 44 ends the week just below it, and seed 43's `R` collapses
  almost as far as 101's (`0.388730055` -> `0.048512264` over nights 1-7, against 101's `0.290963714` ->
  `0.032167699`). Seed 43 does not fail FI5 only because it starts from a higher `R` and its own block's ratio
  (`fact_ratio_by_night`, `inc_mag/base_mag`) does not fall below the recall margin until later: `0.7420857325160359`
  at morning 7, close to the `0.7063632341370527` at which seed 101 first failed (morning 5, above). Whether seed
  101's failure is a build-specific fragility, or seed 43 shows the same general failure mode narrowly escaped, is
  not investigated.
- Whether the store's own `dW/W` read is the right form of "how much was learned", versus a per-synapse or
  per-region measure, is a declared operating point in Amendment 6, not validated by this scoring.

## Next step

The three failure modes point to three different next probes, none of which this finding runs: (1) a fake- or
brain-level sweep of the weak telling's night-1 `R` across more seeds/build variants to characterize where the
capture threshold sits relative to typical `R`, since seeds 43/101 show the capture step itself -- not the
renormalization step Amendment 6 added -- is the more fragile link for an ordinary fact; (2) an `fir`-style arm
with re-mention continuing past night 2 (e.g. one re-mention per subsequent day) to test whether seed 100's night-4
loss is a fixed limit of "re-mention twice" or recoverable with sustained rehearsal, which is closer to what a
person actually does with something they consider worth remembering; (3) a sweep of the salient block's own
across-week `R` trajectory (seed 101 collapses from `0.290963714` to `0.032167699` by night 7; seed 43 collapses
almost as far, from `0.388730055` to `0.048512264`, without failing FI5; seeds 42, 100 and 102 stay clearly above
`0.27` at night 7 (`0.315911916` / `0.328374885` / `0.424426312`) while seed 44 ends the week just below it at
`0.2689885`) to characterize whether seed 101's failure is a build-specific fragility, why seed 43's comparably
large collapse does not cost it FI5, or whether both are one general failure mode of leaving a captured trace's
own protection to decay unrehearsed for a week.

## Flip candidacy

`BRAIN_SLEEP_LOAD_RENORM` is not a flip candidate from this finding. The 6-seed verdict is NO-GO (`n_go=3`); two
independent adversarial re-reviews of this scoring have now run -- one on the initial scoring commit, one on the
fix round that followed it -- and neither changed the NO-GO verdict, each instead finding prose/mechanism errors
in the write-up that this and the prior fix round corrected. No production-default validation has been attempted.
This finding does not flip any default.

## Artifacts

- `research/findings/raw/_sleep_forgetting_interference/seed{42,43,44,100,101,102}.json` -- the six registered
  gate rows, each with `family=fi`, 9 arms, 0 arm errors, and the runner's own stored `gates` (re-graded here and
  found identical).
- `research/findings/raw/_sleep_forgetting_interference/seed{42,43,44,100,101,102}/*.json` (+ `.prov.json`
  sidecars) -- the 54 per-arm `onebrain_regression_battery.py --worker` outputs each seed's row is assembled
  from; every sidecar records `git_sha=2def39c76cb4d81ca1b907399d3e9482f62bfe42`, `source_kind=git_archive`, both
  manifest checks true.
- `research/findings/raw/_sleep_forgetting_interference/aggregate.json` (+ `.prov.json`) -- the registered
  `--aggregate` combine, computed fresh in this worktree with the command shown above, with a `preconditions`
  block added for `gates/verdict-preconditions` (family-level instrument-integrity checks only; see the field's
  own `preconditions_note`).
