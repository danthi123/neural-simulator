# Pre-registration (REGISTERED): oracle reply ceiling on the slept K-P brain (does a right fetch improve the reply at all?)

> **STATUS 11-27: REGISTERED 2026-11-27 after pre-data sonnet review.** Stage 0 is a rescore of banked JSON and was run
> report-only (below). Stage 1 has not been run; no reply has been read. Design: `brain/research/2026-11-27-fable-review-dg-port.md`
> section 3. Capability: Episodic recall. Code: `brain/experiments/memory_ceiling.py` (imports `routing_replay`, edits nothing
> there), tests `brain/tests/test_memory_ceiling.py` (28), node script `brain/experiments/results/2026-11-27-memory-ceiling/run_node.sh`.
>
> **Fixes from the pre-data review (all applied before any Stage 1 run; the pass rule itself is unchanged):**
> 1. `verdict` requires the identical (seed, night) unit set and told row count in OFF, A, ORACLE and SHAM, equal to the set of
>    non-void nights in the `dec_*` files; otherwise INCONCLUSIVE (no comparison across different populations).
> 2. The recall closure raises unless each recall call's cue equals battery row i's cue, and unless the arm made exactly one recall
>    per battery row (forced indices cannot land on the wrong row silently).
> 3. Engagement is recorded: per row `forced_idx`, `forced`, `engaged` (the recalled text is the forced episode's text); per rekeyed
>    episode old and new delta norms (an empty rekey raises). `score` reports per arm told rows forced / engaged / with no stored
>    right episode, so a small ORACLE - OFF cannot hide an arm that did not engage.
> 4. `score` writes `handread_sheet.json` (blind, shuffled labels) and `handread_key.json` (the 40-row hand-read is now delivered).
> 5. A told row with no stored right episode (or no sham candidate) keeps A's fetch in ORACLE / ORACLE-rekey / SHAM (section 4 is now
>    literally true) and is counted (`told_no_right_episode`).
> 6. SHAM excludes the right episode itself as well as the row's item (`i != right_idx`).
> 7. K-REFIT changes A's fetch only on told and fire rows; identity rows keep A's fetch (section 2 reading kept, section 4 "no fetch" applies to those two classes only).
> 8. Fire-row measures are reported per row kind, with a `never` / untold-near-duplicate subtotal for OFF, A and K-REFIT.
> 9. `dec_*` and `u_*` files carry the stage0.json hash and commit; a resume under a different hash or commit stops, and `score`
>    returns INCONCLUSIVE on mixed provenance (use a fresh out dir after any code or stage0 change).
> 10. The user-voice measure calls `voice_split.knows` itself (test pins equality); `uv_told` is reported beside `uv_knows`; the gate
>     (clause H6) stays `uv_knows` ORACLE minus OFF <= 5, as declared in section 5.
> Tests added: mismatched night sets, mixed commits, engagement and fire kinds, recall closure with a fake brain (cue and count mismatch).
> The three operational readings in section 5 are confirmed as written.

## 1. Record brief
- The routing arc (11-23..11-26) is closed with "no 6th lever" (`PREREQUISITES.md`, Memory selection). On the 1.5B K-P slept life the
  banked fetch is right on 44 of 217 told rows, wrong on 136, no-fetch 37, and false fires 239. The right episode ranks first on 168
  under S_all but its strength-weighted score does not clear the bar (screen 2 read: median `tanh(strength)` 0.36 vs 0.85 on rows
  that fired right).
- At fetch level the hippocampus hurts on K-P (told 123/223 with it, 138 without). On the slept gate, fetches 1/18 -> 14/18 moved
  replies +4/36, the lesion as good (11-18). On 11-13, always-reinstate scored 7.67 vs 9.33 hippocampus-off. Oracle reinstatement
  has only been run on unslept cortex (11-13, 11-14), never on the slept brain with the right episode forced.
- **Question.** Does a right fetch improve the reply at all? That decides whether any fetch-side work (DG, write-side separation,
  strength, source tag) is worth building. A measurement, not a method; no brain default changes.
- Expected-fail (declared): the strict told scorer and the user's-voice habit may keep ORACLE near OFF even with the right
  episode in the state; that is the NO-HEADROOM outcome, not a defect.

## 2. Population
Seeds 319/320/321 (the KR lives of screen 2), nights 5-15, the banked intact-frozen battery of each night (`routing_replay.banked_rows`).
Told rows (217 after the s321 night-5 void, as screen 2) plus the fire rows (open, greet, never-told, untold near-duplicate) and the
identity rows (kept at A's fetch in every non-OFF arm). Void rules as `routing_replay`: a night is void if arm A's decision does not
reproduce the banked fetch idx and recalled text on every row, or A's rightness differs from the banked row. Void nights run no arm.

## 3. Stage 0 (no new compute; report only)
Rescore `brain/experiments/results/2026-11-25-routing-screen2/s2diag_s*.json` with the told-blind bar rule of the DG prereg section 5,
for arm K (S_all slept key): the smallest question bar on the grid 0.000..1.000 step 0.005 at which K fires on the pooled fire rows
no more often than A (A-matched) and at half A's fires; statements at `bar * 0.45/0.12`; weighted margin 0.02. Report right, wrong,
no-fetch and fires. A refit winner's rightness is known only where the banked fields decide it (K fired at the registered bar, or the
weighted winner is the raw winner); the rest is reported as `fired_unknown_right`, never guessed. The A-matched decisions are kept for
the K-REFIT arm. Output: `results/2026-11-27-memory-ceiling/stage0.json`.

**Stage 0 result (11-27, CPU, banked files; report-only).** Registered K is reproduced on every row. A fires 239 on 359 fire rows.

| arm (told 217, fire rows 359) | bar | right | wrong | fired, right unknown | no-fetch | fires |
|---|---|---|---|---|---|---|
| A (banked) | - | 44 | 136 | 0 | 37 | 239 |
| K registered | 0.165 | 49 | 40 | 0 | 128 | 67 |
| K-refit, A-matched | 0.000 | 86 | 42 | 43 | 46 | 172 |
| K-refit, half A | 0.115 | 65 | 42 | 30 | 80 | 119 |

The A-matched bar is the grid floor: K at bar 0 still fires only 172 times on the fire rows (the margin blocks the rest), under A's 239.
So A-matched "refit" means "no score bar, margin 0.02 only". Right is a lower bound (43 / 30 fired rows have unknown rightness; the
upper bound is right + unknown = 129 / 95).

## 4. Stage 1 arms (all through the brain's own reinstatement path at shipped `Hippocampus.GAIN`, no host text)
| arm | what |
|---|---|
| OFF | hippocampus lesion (`set_lesions(hippocampus_off=True)`) |
| A | banked fetch (reproduction check against the banked replies) |
| ORACLE | the right episode forced on told rows (A's fetch on every other row); where A already fetched a right episode, that one |
| SHAM | a forced other told episode, not the row's own item, matched on strength tercile (over the told episodes) and nearest age; the no-contrast control |
| ORACLE-rekey | ORACLE with the forced episode's delta re-read through tonight's cortex (separates drift from content; the 184/217 pointer) |
| K-REFIT | the Stage 0 A-matched decision reinstated (no fetch where it does not fire) |

Right episode (ORACLE) = the stored episode that is the row's item's own telling (`life.episode_is_source`); if several, the one with the
highest raw similarity to the cue. Each (seed, night, arm) is the banked battery run as a running conversation through `Brain.hear`
on a fresh copy of the night snapshot WITH deltas, the probe's clock and reseed, FREEZE on (so non-forced rows are as the banked probe).

## 5. Measures and the pass rule (Fable review, verbatim; fixed before data)
- **Measures:** strict told scorer; user-voice rate with the existing voice-split scorer; blind hand-read of 40 rows per arm;
  never-told fire rows for OFF, A and K-refit. Report pooled, per seed, the 89 non-H rows, and the 68 (seed, item) clusters, because
  128 rows come from four items.
- **HEADROOM** needs all of: ORACLE - OFF >= +15 of 217; >= +6 on the 89 non-H rows; positive on 3/3 seeds; cluster sign test with
  wins > losses; ORACLE - SHAM >= +15; user-voice not above OFF by more than 5.
- **NO-HEADROOM** if ORACLE - OFF < +8 or ORACLE - SHAM < +8.
- Anything between is **UNDEFINED**, with no rescue.
- **INCONCLUSIVE** if any control arm (OFF, A, ORACLE, SHAM) is missing or a seed is missing.

Operational readings (for the review to confirm or change before data; the code is `verdict()`): all thresholds are absolute counts
on the surviving rows (not rescaled if a void night lowers N below 217); "ORACLE - OFF" is the difference of strict-ok counts on told
rows; "positive on 3/3 seeds" is ORACLE ok > OFF ok within each seed; the cluster sign test counts a (seed, item) cluster a win if
its ORACLE ok count exceeds OFF's, a loss if lower (ties dropped), and the exact p is reported only; "user-voice" is the count of told
rows whose reply names the told fact (voice_split.knows) in the user's voice (`cascade_screen.own2`), ORACLE minus OFF <= 5.
The blind hand-read and the fire-row measures are reported, not barred. A's reply reproduction is reported (not a gate; the void
rule above is the decision-level gate).

## 6. Compute
Per (seed, night) 6 arms x about 20 battery rows = 120 replies; 33 nights = about 4,000 replies (Fable's ~1,100 counted told rows
only; a running battery cannot skip the other rows without changing the conversation). One seed per node (pool40/41/42), CPU fp32,
under `tools/memcap.sh`. Per-reply time is unmeasured: time night 5 first (`--nights 5-5`), and move to the 3090 fp16 if the
projection exceeds 20 hours (`BRAIN_ALLOW_W3090=1`, idle-only). Per-unit atomic files with resume; a re-run skips finished units.
Host / commit / device are recorded in every output file.

## 7. What each outcome does
- **NO-HEADROOM:** close all fetch-side work, including the DG and source-monitor ports; bank that the slept cortex already carries
  what reinstatement adds. The lever is what sleep stores and replays (both speakers' turns), after test2.
- **HEADROOM:** build the mechanism the diagnosis names (reachable strength: nightly replay strengthens the replayed episode's same
  trace with homeostatic scaling), with its own prereg and six seeds; no DG.
- **UNDEFINED / INCONCLUSIVE:** reported as such; no rescue, no relaxed bar.

## 8. Caveats
Seen seeds, 3 seeds (a ceiling, not a generalisation claim); 1.5B K-P only; the strict scorer misses paraphrase; the 4 items behind
128 rows dominate the pooled count (hence clusters); the two-sided cluster test has low power at 68 clusters; ORACLE-rekey uses
`episode.text` (stored without contrast / completion options, as the lives stored them).
