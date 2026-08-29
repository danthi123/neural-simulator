---
type: finding
status: live
lane: e-language-mouth-read-snr
board: 80
date: 2026-08-28
mechanism: information-limiting-correlations diagnostic (Moreno-Bote et al. 2014) at FIXED trained (head_w) weights
  for the mouth read-power wall (#80/gap#4, rank 3 of the deep-research shortlist), bundled with the mandatory
  recheck of whether the shortlist's own motivating walls survive the 2026-08-27 stale-weight-cache fix.
seeds: [42, 43]
seed-waiver: the diagnostic is a 2-seed alignment-vs-permutation-null measurement (a sequencing gate, not a
  GO/NO-GO claim per CLAUDE.md's own instruction for this rung: "a couple of seeds for robustness" suffices).
  The bundled recheck is pure git-history/artifact-lineage forensics (no seeds; it re-reads existing 6-seed
  artifacts already on disk) plus one already-committed, already-6-seed-confirmed production result
  (`2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md`, 6 seeds).
artifacts:
  - research/findings/raw/_wkv_mouth_read_infolimiting/diag_s4243.json
  - research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json
  - research/findings/raw/_wkv_freshbuild_verify/fb_6seed.json
  - research/findings/raw/_wkv_softmax_confidence/diag_s42.json
external:
  - Moreno-Bote, Beck, Kanitscheider, Pitkow, Latham & Pouget, "Information-limiting correlations", Nat Neurosci
    17:1410-1417 (2014)
runner: research/runners/_wkv_mouth_read_infolimiting_diagnostic.py
---

# Mouth read-power wall (#80): the information-limiting diagnostic says INFORMATION-LIMITING (small effect) -- and the "deep 0.34-0.37 plateau" the shortlist targeted had ALREADY vanished before the shortlist was written

Artifact: `research/findings/raw/_wkv_mouth_read_infolimiting/diag_s4243.json` (this runner, 2 seeds, numpy/CPU).

> **Reads before this one (do not re-derive):**
> `research/findings/2026-08-28-mouth-read-power-wall-deep-research-ranked-shortlist.md` (the ranked shortlist
> this diagnostic was queued from, rank 3) · `research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`
> (source of the "~0.34-0.37 plateau" number) · `research/findings/2026-08-27-mouth-stale-coo-training-fix-PARTIAL.md`
> + `research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md` (the fix + its
> production confirmation) · `research/findings/2026-08-27-decorrelation-read-shared-fidelity-wall-PARTIAL.md`
> (the sibling retraction this finding extends).

## Summary (read this first)

Two results, one diagnostic run and one archival bundle:

1. **THE DIAGNOSTIC (rank 3): INFORMATION-LIMITING, but the effect is small.** At fixed trained (`head_w`)
   weights, the leading eigenvector of the substrate read's trial-to-trial noise covariance aligns with the
   read's signal direction beyond a channel-identity permutation null on the POOLED (canonical, best-powered)
   test at both seeds (z=3.95 seed 42, z=2.47 seed 43) -- same-population pooling/expansion cannot reduce this
   noise's effect on the decision. Per-probe (less powerful, 8 probes/seed) the signal is present but weaker
   (1-2/8 probes individually significant per seed). This independently explains why the ensemble (`--sub-pop`)
   read was inert and why repeated-read averaging left `weight_cosine` flat (both already banked, see below).
2. **THE MANDATORY RECHECK: the shortlist's OWN motivating "deep plateau" wall had ALREADY vanished before the
   shortlist was written, and a second, sibling wall (the structure-selective collapse) had ALREADY been
   officially retracted the day before -- neither fact was threaded through to the shortlist's problem framing.**
   This does not change the diagnostic's verdict (item 1 stands on its own, fixed-weight measurement, immune to
   the bug in question -- see Method note below) but it substantially re-prices the STAKES: the shortlist's
   ranks 1-2 are being funded to close what reads, post-recheck, as a SMALL residual, not the "deep plateau"
   the shortlist's §1 still describes.

## Part 1 -- the diagnostic

### Method (numpy/CPU, no GPU touched)

`research/runners/_wkv_mouth_read_infolimiting_diagnostic.py` (new, reuse-by-import of `BatchedSubstrateReadout`
from the eprop batched-substrate runner, `_positions`/`_load_eval`/`WKVReadout` from the fewspike/eprop-learn
runners). Per seed: fix weights to `ro.head_w` ONCE (`set_weights` called exactly once -- no re-editing between
reads, so this measurement is structurally immune to the stale-weight-cache bug discussed in Part 2, which only
bites when a caller re-edits weights on a reused bridge without invalidation); collect 60 repeated
`batch_margin()` reads of the SAME B=8 probes (each call is a fresh OU/spiking noise draw on the still-running
process, never reseeded between calls -- confirmed genuine, `resid_std` nonzero every seed, CV ~5-6%). Per probe:
restrict to the top-15 "live" word-pool channels by that probe's own trial mean, ROW-CENTER (subtract each
trial's own mean over the full V=1000 channels -- the instrument `_measure_gain`'s own commentary already
established as the metric that predicts what softmax/argmax decoding actually sees, since a shared per-position
additive shift is decision-invisible), compute the noise covariance's leading eigenvector, and the
`head_w`-predicted (noiseless) ideal-linear-map direction restricted to the same channels as the signal
reference. Alignment = |cos(leading noise eigenvector, signal direction)|, tested against a 1000-draw
channel-identity permutation null. A POOLED version repeats this across a shared top-20-channel set built from
all 8 probes together (closer to Moreno-Bote's canonical multi-stimulus signal-covariance-vs-noise-covariance
construction; more statistical power than any single probe). B=8/read-window=64 is the structure-characterization
runner's own established memory-safe CPU operating point (reused verbatim).

An early version of this diagnostic (RAW, non-row-centered) gave misleadingly HIGH permutation-null baselines
(0.5-0.9) because both the noise eigenvector and the signal direction were dominated by a shared cross-channel
"row baseline" -- exactly the artifact `_measure_gain`'s own docstring warns about. Row-centering (matching the
established instrument) dropped the null baseline to a sensible range and is what the reported numbers use;
this is recorded so a future rerun does not re-discover it the hard way.

### Result

2 seeds, `research/findings/raw/_wkv_mouth_read_infolimiting/diag_s4243.json`:

| seed | pooled alignment | pooled null mean | pooled z | pooled verdict | per-probe info-limiting count |
|---|---|---|---|---|---|
| 42 | 0.9244 | 0.6456 | 3.948 | information_limiting | 1/8 |
| 43 | 0.7980 | 0.6569 | 2.465 | information_limiting | 1/8 |

Mean per-probe alignment 0.8265 vs mean per-probe null mean 0.8106 (mean z 0.818 -- weak/mixed at the
single-probe level, only 8 trials-worth of channel structure each); the POOLED test (more samples, the
canonical construction) is the decisive one and reads information-limiting at both seeds with z well above the
z>2 threshold. **Verdict: INFORMATION-LIMITING**, on 2 seeds -- a real but MODEST effect (CV of the noise is only
~5-6% of the signal's own magnitude; this is not a large-noise regime).

### What this explains (cross-checks with already-banked results)

Two independent facts already on record are consistent with -- and now partly explained by -- an
information-limiting (not merely small) noise structure:

- **The ensemble (`--sub-pop`) read is inert by construction** (`2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`):
  word-pool members are deterministic conductance replicas of ONE shared noisy hidden population -> common-mode
  noise -> pooling cancels it exactly. Common-mode noise is a canonical case of noise living along a shared,
  decision-relevant direction.
- **Averaging repeated reads left `weight_cosine` FLAT** (`2026-08-27-mouth-readsnr-decoder-direction-perword-calib-NOGO-read-gradient-already-ideal.md`,
  quoted verbatim, not re-measured here, per-read noise CV was reported ~0.08 <!--derived--> and averaging K=3
  reads per gradient step left `weight_cosine` flat -- "read noise is not the lever" in that finding's own words.
  An information-limiting verdict predicts exactly this:
  small noise magnitude (matches CV ~0.05-0.08 across both measurements) whose EFFECTIVE cost cannot be reduced
  by same-population averaging because it is aligned with the direction the decision depends on.

## Part 2 -- the mandatory recheck: does the wall this shortlist targets survive the stale-weight-cache fix?

This is pure archival/artifact-lineage work (git log timestamps + reading already-committed 6-seed artifacts),
zero new compute, done because CLAUDE.md's own discipline says a comfortable "wall" verdict is the START of
research, never the end -- and because the shortlist's rank-5 text ITSELF already flags that "the cupy
structure-characterization wall vanished... once a stale-COO cache was fixed" without following that thread back
into its own §1 problem statement.

### 2a. The "deep ~0.34-0.37 plateau" the shortlist's §1 cites as a SEPARATE, deeper problem: VANISHED

Exact chronology (`git log`, all times same repo, EDT):

| when | commit | what |
|---|---|---|
| 2026-08-27 07:50 | `3c75d58c0` | ensemble/dendritic-lever finding banks "`sub_learned_recov_mean` ~0.34-0.37", sourced from 2026-08-19 pre-fix runs |
| 2026-08-27 22:00 | `d6c375de5` | `sim/bridge.py::mark_weights_edited()` fix lands: the mouth eprop TRAINING loop's per-gradient-step `set_weights()` was never invalidating the megakernel-v2 transposed-CSR cache, so every substrate-forward read after the first transmitted the FIRST-loaded weights -- **the exact same instrument the training loop's own OWN `set_weights` calls (SAME file, SAME method) that Part 1's diagnostic above deliberately calls only ONCE.** |
| 2026-08-28 02:19 | `e7a9049fb` | full-scale B=48 6-seed CONFIRMATION on the IDENTICAL runner + IDENTICAL decisive config (`n-train-pos 9600`, `epochs`, `batch 48`) the "0.34-0.37" number itself used: `sub_learned_recov_mean = 0.8499`, `sub_copied_recov_mean = 0.9785`, `sub_recov_ratio_mean = 0.8686` (min 0.8399), anti-cheats clean, `go_count 3/6` |
| 2026-08-28 19:34 | `878a31d2d` | the ranked shortlist is written -- AFTER the 02:19 confirmation, yet still frames "the DEEPER ~0.34-0.37 plateau" as a problem SEPARATE from "the narrow residual" (§1: "These are DIFFERENT problems") |

The "0.34-0.37" number and the "0.8499 mean / 3-of-6-miss-narrowly" number are **the SAME metric, on the SAME
runner, at the SAME decisive config**, measured before vs. after the ONE fix that specifically targeted this
exact call path. There is no independent evidence of a second, deeper wall distinct from the narrow
(0.84-0.91-band) residual the pending coverage/epochs run is already addressing -- the "0.34-0.37 deep plateau"
**was** the stale-cache artifact, at production scale, confirmed by the project's own subsequent 6-seed run.
<!--derived--> This MOOTS the shortlist §1/§5 framing of two separate problems: there is one residual, and it
is already down to a ~13-percentage-point ratio gap (0.8686 vs the 1.0 line), not a ~50-point one.

### 2b. The structure-selective-collapse wall (motivates ranks 4-7): ALREADY officially retracted, one day before the shortlist

`research/findings/2026-08-27-mouth-readsnr-structure-characterization-cupy-SUBSTRATE-WALL.md` (the "0/6
recodable... structured direction reads ~0, random reads ~0.31" wall the shortlist's rank-4/5/6 downweighting
language references) is **already listed in `docs/RETRACTED.md`** (PARTIAL, added 2026-08-27), superseded by
`research/findings/2026-08-27-decorrelation-read-shared-fidelity-wall-PARTIAL.md`: on a FRESH build per probe
(the correct instrument), `head_w` decodes 0.9569 vs random 0.9473 -- **structured == random, 6/6 seeds**. The
"structure-selective collapse" premise motivating the whole nonlinear-expansion/sparse-K family (ranks 4-7) is
void, not merely downweighted. The shortlist's rank-5 text already knows this ("the cupy structure-characterization
wall vanished... once a stale-COO cache was fixed") but its own §1/§2 problem framing does not fully carry the
consequence through: the read was never structure-selective to begin with, so "recodability" (ranks 4-7's shared
premise) was never the right question.

### 2c. A previously un-flagged sibling: the softmax-confidence-weightnorm-NOGO finding shares the exact same bug and has NO retraction row -- added here

`research/findings/2026-08-27-mouth-readsnr-softmax-confidence-weightnorm-NOGO.md` (17:04, `SIM_BACKEND=cupy`,
confirmed from its own artifact's `"backend": "cupy"` field) independently reported the identical qualitative
result (`head_w` corr -0.0006 to -0.0028 vs random ~0.9465-0.9483) via the IDENTICAL vulnerable shape
(`_measure_gain` -> `s_batch.set_weights()` called ~11 times reusing ONE bridge, no invalidation available at
that commit -- verified via `git log -p`: the `mark_weights_edited()` line was added to this exact `set_weights`
method at `d6c375de5`, 22:00, ~5 hours AFTER this finding was committed). The superseding
decorrelation-read-shared-fidelity-wall-PARTIAL finding already NAMES this specific finding's failure mode in its
own text (its line 36: "The softmax diagnostic's probes A/B/C were the SAME direction rescaled (all matched the
stale weights, all high) and its order-control reused that direction, so it never caught the bug") -- but
`docs/RETRACTED.md` had no row for it, and `ROADMAP.md`/`2026-08-27-mouth-readsnr-structure-characterization-BACKEND-SEED-CONFOUND.md`
cite it without `⛔`. **Added the missing row to `docs/RETRACTED.md` this session** (verified against
`tools/check_docs.py`, W1/W2 both pass); its adaptive-gain 6-seed NO-GO sub-result (Result 3 of that finding)
is unaffected and survives, since it compares two calibration MODES against each OTHER on the same
(equally-affected) substrate, not against an absolute correlation floor. `ROADMAP.md`'s citation still needs its
own `⛔` -- flagged, left for `sync-documentation` (out of this diagnostic's scope; the citation is a batch-listing
mention, not a load-bearing claim).

## Net interpretation

The mouth read is **not** structurally/architecturally broken -- every "wall" that was actually measured through
the pre-fix instrument (deep plateau, structure-selective collapse, and now its softmax-confidence sibling) has
either directly reversed (0.34-0.37 -> 0.85) or been shown to vanish on a correct read (head_w == random,
~0.95-0.96 both). The ONLY defect that was real is the stale-weight-cache bug itself, now fixed everywhere it has
been checked (training loop + fresh-build read verification). What remains, on top of a substrate read that IS
faithful for structured directions, is the SMALL residual this diagnostic characterizes: a genuine
information-limiting noise component (CV ~5-6%, POOLED alignment z 2.5-3.9 above a permutation null at both
tested seeds) that same-population pooling/expansion cannot reduce -- consistent with, and now mechanistically
explaining, both the ensemble-inert NO-GO and the averaging-flat NO-GO already on record.

## Funding verdict for the shortlist's ranks (decisive, given the above)

- **Ranks 1-2 (Urbanczik-Senn dendritic teacher; variance-weighted regularized read) are STILL the correct CLASS
  of lever** -- an information-limiting noise mode is precisely what an INDEPENDENT teaching/estimation pathway
  can address and same-population pooling cannot (three independent pieces of evidence now converge: the
  ensemble NO-GO, the averaging-flat NO-GO, and this diagnostic's alignment result). Rank 1's staged 6-seed GPU
  run remains worth its queue slot. **But the STAKES are lower than the shortlist stated**: they are being
  funded to close a residual that is now ~0.87 ratio (not the ~0.35-0.37 "deep plateau" framing), and the
  PENDING coverage/epochs run (a pure data/budget lever, cheaper than either) may close much of the remaining
  gap on its own -- if it does, ranks 1-2 become the mechanism for the LAST few points, not the primary unlock.
- **Ranks 4-7 (nonlinear-subunit expansion, sparse-K recoding, ensemble wiring fix, plastic sparse-K) are now
  MORE than "downweighted by the vision-2layer NO-GO"**: (a) their shared "structure-selective collapse" premise
  is void (§2b, already retracted); (b) this diagnostic's information-limiting verdict independently predicts
  same-population expansion/pooling schemes cannot fix an information-limiting noise mode regardless of how the
  population is organized. Recommend these drop further down the queue -- fund only if ranks 1-2 both NO-GO.
- **Rank 3 (this diagnostic) is DONE** at the diagnostic-appropriate bar (2 seeds + permutation null, per the
  task's own instruction). A decisive 6-seed confirmation would be cheap (CPU-only, ~5 min/seed) but is not
  required to act on the verdict above; queue it only if rank 1's decisive run comes back ambiguous and the
  information-limiting question becomes load-bearing again.

## Files

- `research/runners/_wkv_mouth_read_infolimiting_diagnostic.py` -- the diagnostic (fixed-weight repeated-read
  collection, row-centered noise/signal covariance, permutation null, per-probe + pooled). CPU/numpy only,
  reuse-by-import, no `sim/` edit.
- `docs/RETRACTED.md` -- added the missing row for `2026-08-27-mouth-readsnr-softmax-confidence-weightnorm-NOGO.md`.
