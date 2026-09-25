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
  sleep-replay-capture-margin miss for the neutral telling that is present with ZERO later learning -- outside
  the load-renormalization mechanism this family exists to measure. Seed 100 fails FI6: the twice-re-mentioned
  fact is lost by night 4 under the continuing 3-facts/day dose. Re-graded from the raw artifacts with the
  registered `grade_seed_fi` / `aggregate_fi`, unmodified.
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
read `abstain` at the first morning too, which is why FI4 (`fih_lr_lesion`, the load edge cut) also fails: the
load-renormalization edge is irrelevant to a fact that was never captured in the first place.

**Ruling out an encoding miss.** `P1_immediate_precondition` holds on both seeds -- `neu_imm_fi` (a fresh brain,
the same telling, recalled at once with no night) is correct. `n_managed_blocks_by_night` for `fiv_lr` is `1` on
every one of the seven nights on both seeds -- the ledger created exactly one managed block for the fact, and it
never disappears. The block's own increment-to-baseline ratio at recall 1 is `0.9561444027828692` (seed 43) and
`0.9920212521528132` (seed 101) -- both close to the fresh-fact range the four GO seeds show at the same point
(`1.305007482661105`, `1.0131071147025406`, `1.0254155897799755`, `1.209882425794491` for seeds 42, 44, 100, 102
respectively). The fact was told, stored as one block with a substantial weight, and immediately recallable. It
was not "never stored."

**What actually differs: the SWR reactivation margin at night 1, and whether the block crosses into late-phase
capture.** Every managed block records `z_mean`, the fraction of the block's synapses that have crossed into the
late (protein-synthesis-independent, persistent) phase. At recall 1, `z_mean` is `1.0000000000508993` (seed 42),
`0.999999999981423` (seed 44), `1.000000000015095` (seed 100) and `1.0000000000483527` (seed 102) -- captured. On
seeds 43 and 101 it is `3.8088625378623516e-10` and `1.0692804587972005e-11` -- effectively zero: the block never
left early phase. The one thing that differs at the moment of capture is the sleep epoch's own reactivation read,
`R`: `0.356535268` / `0.310521057` / `0.320028458` / `0.423728264` on the four GO seeds against `0.168587758`
(seed 43) and `0.035599825` (seed 101) -- roughly half to a twelfth as large. The resulting SWR-coupled DA
(`da_swr`) is correspondingly lower (`0.624754941` and `0.526343871` against `0.763836098`-`0.813558916` on the
GO seeds), and the D1-gated capture that Amendment 6's mechanism runs through never crosses whatever threshold
turns early-phase expression into a permanent trace. The actual chat-turn reply at recall 1 on seed 43 is
consistent with a genuine cue-matching miss, not silence: `"(I'd been mulling over dog.) As for it — I don't know
about that."`, with `matched_fact_index: null` against `n_facts_scanned: 6` -- the store scanned all six blocks
(five build-time plus the one managed block) and found nothing confident enough to answer with. Seed 101's
`fiv_recall1` reads the same shape (`abstained: true`, `recalled_svo: null`, `matched_fact_index: null`,
`n_facts_scanned: 6`).

**This is not "seeds 43 and 101 cannot recall the fact"; it is specific to the neutral telling's margin.** The
*salient* telling (`fis_lr`, the same fact inside surprising news) reaches `R = [0.388730055]` (seed 43) and
`R = [0.290963714]` (seed 101) at its own night-1 epoch -- inside or above the GO seeds' neutral-telling range --
and both cross into capture (`z_mean = 1.0000000001010352` and `1.0000000000866132`); `fis_lr` is correct at
recall 1 on both seeds. The failure is a margin that sits close to a threshold for THIS content on THIS seed's
heterogeneous population, not a broken seed: raise the drive (salience) and the same mechanism captures normally.
This is the best-supported reading of "encoding miss (never stored) vs. immediate-read failure vs. something
else": it is closest to an immediate-read failure, precisely localized to a capture event that fails to cross
threshold at the very first night, upstream of the load-renormalization mechanism this family is built to
measure. It is a property of the *sleep-replay-capture* route (`BRAIN_SLEEP_REPLAY_CAPTURE`, Amendment 1-5), not
of `BRAIN_SLEEP_LOAD_RENORM` (this amendment).

**This same signature was already reported once, for the same two seeds and the same weak telling, in a sibling
family.** `research/findings/2026-09-25-sleep-replay-capture-r2-NO-GO-6seed.md` ("Why P2 fails on two seeds")
found `d3w_rc` (r2's own no-downscaling control, same content, no later learning at all) reading `abstain` on
seeds 43 and 101 while its own increment-to-baseline ratio at recall was *above* 1.0 on both, and named it "a
readout miss on the composer's cleanup margin", not a decayed trace. The `z_mean`/`R` breakdown above is a more
precise account of the same phenomenon, now traced to a specific step (the sleep epoch's capture-threshold
crossing) rather than only the recall turn's outcome. Seeds 43 and 101 failing FI1 is a **replication** of an
already-reported per-seed fragility in this route's neutral-telling capture, not a new failure mode of the
load-renormalization mechanism.

## Why seed 100 fails FI6: re-mention's own protection erodes under the continuing dose

`fir_lr` re-mentions the fact ("the cat chases the ball") after nights 1 and 2, each time writing a fresh block,
before that day's three later facts. `n_managed_blocks_by_night` for seed 100 is `[1, 5, 9, 12, 15, 18, 21]` --
1 (the original) at night 1, +1 (re-mention) +3 (day-2 facts) = 5 by night 2, +1 (re-mention) +3 (day-3 facts) = 9
by night 3, then +3 a night through night 7, exactly the registered protocol. `daily_outcomes` are `correct,
correct, correct, abstain, abstain, abstain, abstain` -- lost starting the fourth morning, one to two nights
after the second re-mention. The tracked original block's own ratio falls every night: `1.0254155897799755` (night
1) -> `0.7024895478071427` -> `0.49393965992299305` -> `0.40280562188118535` -> `0.34197805377694906` ->
`0.3015797022920384` -> `0.2691548021017118` (night 7), against a per-night renormalization delta of
`0.1440927`, `0.452880123`, `0.307868533`, `0.188656302`, `0.152351566`, `0.121774295`, `0.110228511`. This is a
genuine, monotone erosion under the mechanism the family measures, unlike seeds 43/101's all-or-nothing night-1
miss: the re-mention writes fresh blocks (which reset their own R and start the same load-renormalization clock),
but the continuing 3-facts/day dose renormalizes each of them in turn before their next re-mention, so by night 4
none of the three related blocks (original + two re-mentions) clears the recall margin any longer. FI6 predicted
re-mention would keep the fact through night 7 at this dose; on this one seed it does not, once re-mentioned
blocks are treated by the same load edge as any other block. This is a genuine limit of the built protection, not
an instrument defect (`I2_load_lesion_held` and `n_arm_errors=0` both hold for seed 100).

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
direction, and it is built the way the directive asks: the amplitude is read from the brain's own store (`dW/W`),
not a host-picked importance score.

It does **not** show the directive is implemented. The 6-seed verdict is NO-GO: two of six seeds fail before the
renormalization mechanism ever gets a chance to act (an unrelated capture-margin fragility in the upstream
sleep-replay route), and a third shows the re-mention protection itself is not robust to a full week of continuing
load. The mechanism scored here distinguishes exactly two categories of "important" (told saliently, or repeated)
against one unrehearsed control; it has no notion of graded importance among many concurrently known facts, no
interaction with genuine content-similarity interference (the told facts here are deliberately dissimilar to the
target, per the prereg), and depends on a capture step (sleep-replay-capture) that this same scoring shows is not
yet reliable across seeds for an ordinary neutral fact. Whatever downstream feature the memory entry describes as
"waiting for this fix" should keep waiting: this is a directionally consistent building block, re-graded here as
NOT YET a 6-seed-robust one.

## What this does and does not show

FI1 and FI2 together, on the seeds where both are measurable (42, 44, 102), tie the loss to later learning: the
same brain, the same telling, the same first night (I3), diverging only once later facts start. FI4 ties it to
the renormalization edge specifically: cutting it keeps the fact all seven nights on those same seeds. FI3 is a
genuine dose-response on a continuous read, not just a discrete pass/fail. It does **not** show that this
family's night amplitude corresponds to any particular human day (declared already in the prereg: the store's W
grows by a block per fact told, so this model's delta falls faster with accumulated knowledge than a
renormalize-the-whole-brain system would, biasing toward retention). It does **not** show that re-mention is a
reliable protection at this dose (seed 100). It does **not** show that the sleep-replay-capture route reliably
captures an ordinary neutral fact at all (seeds 43, 101) -- a pre-existing property of an earlier amendment, not
of the mechanism this amendment adds. Similarity-dependent interference (A-B, A-C) is not tested here, as the
prereg states.

## Honest limits

- This is a re-grading of already-produced raw files with the registered, unmodified grader and aggregator; no
  new brain build ran for this finding.
- The night-1 capture-margin miss on seeds 43/101 is described here with more mechanistic detail (`z_mean`, `R`,
  `da_swr` at the exact epoch) than the prior r2 finding gave, but neither finding identifies WHY those two
  seeds' heterogeneous builds put the neutral telling's reactivation margin below the effective threshold while
  the salient telling's does not; that remains open.
- Whether the store's own `dW/W` read is the right form of "how much was learned", versus a per-synapse or
  per-region measure, is a declared operating point in Amendment 6, not validated by this scoring.

## Next step

The two failure modes point to two different next probes, neither of which this finding runs: (1) a fake- or
brain-level sweep of the neutral telling's night-1 `R` across more seeds/build variants to characterize where the
capture threshold sits relative to typical `R`, since seeds 43/101 show the capture step itself -- not the
renormalization step Amendment 6 added -- is the more fragile link for an ordinary fact; (2) an `fir`-style arm
with re-mention continuing past night 2 (e.g. one re-mention per subsequent day) to test whether seed 100's night-4
loss is a fixed limit of "re-mention twice" or recoverable with sustained rehearsal, which is closer to what a
person actually does with something they consider worth remembering.

## Flip candidacy

`BRAIN_SLEEP_LOAD_RENORM` is not a flip candidate from this finding. The 6-seed verdict is NO-GO (`n_go=3`), no
adversarial review of this scoring has run, and no production-default validation has been attempted. This finding
does not flip any default.

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
