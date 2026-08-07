---
type: finding
status: negative
date: 2026-08-07
lane: laneC
mechanism: source-monitor-conjunctive-source-tag
runner: research/runners/_laneC_source_monitor_conjunctive_tag.py
artifacts:
  - research/findings/raw/laneC_source_monitor_conjunctive_tag/smoke_650_651_gain1.0.json
  - research/findings/raw/laneC_source_monitor_conjunctive_tag/smoke_650_651_gain1.0.json.prov.json
---

# Source-monitor conjunctive source-tag NO-GO (smoke): tagging differentiates WHICH CELLS FIRE at encoding, but recall must drive the WHOLE shared pattern, so the rival subsets fire anyway

**Smoke, not a verdict on the capability.** Two calibration seeds (650, 651), numpy, deterministic. This is the
pre-scoped FALLBACK to the heterosynaptic-encoding NO-GO
([2026-08-07-source-monitor-hetero-encoding-NO-GO-...](2026-08-07-source-monitor-hetero-encoding-NO-GO-shared-core-fires-in-every-recall-so-pedestal-depression-redistributes-not-removes-rival-burden.md)),
which reframed the wall as: separation must live in WHICH CELLS FIRE, not in which synapses survive. Mechanism: a
**conjunctive source-tag** (Komorowski-Manns-Eichenbaum 2009 item-in-context conjunctive cells). During `experience`
(ENCODING ONLY) the physical source afferent WEAKLY modulates the overlap layer -- delivered as a weak additive drive
(`source_tag_gain * drive_pA`) to a fixed source-specific random subset of the driven episode cells -- so a
source-specific subset of the shared core fires preferentially per source. One knob `--source-tag-gain`;
`--source-tag-gain 0` adds nothing and is asserted byte-identical to the symmetric-Hebbian overlap NO-GO baseline. No
`sim/` edit (drive/afferent machinery reused by reference).

## Result: NO-GO. min M never clears the floor F=0.15 and never beats the lesion arm, across gain and overlap

Best `min M` anywhere is **-0.0042** (seed 650, overlap 0.2, gain 1.0) -- still NEGATIVE, ~36x below the 0.15 floor,
and never beats the competition-lesion arm `min L`. Swept gain in {0.5, 1.0, 2.0} x overlap in {0.2, 0.4} x seed in
{650, 651}; every cell is a NO-GO. Representative gain=1.0 rows:

| seed | overlap | core | commitment (seen/heard/self) | H | min M (treat) | min L | min M (base, gain=0) | clears 0.15 | beats L |
|------|---------|------|------------------------------|------|---------------|-------|----------------------|-------------|---------|
| 650  | 0.20    | 2    | 2 / 0 / 0                    | 0.00 | -0.0042       | +0.0042 | -0.0092            | no          | no      |
| 651  | 0.20    | 2    | 1 / 0 / 1                    | 0.63 | -0.1308       | -0.0550 | -0.1233            | no          | no      |
| 650  | 0.40    | 5    | 5 / 0 / 0                    | 0.00 | -0.1200       | -0.0825 | -0.1292            | no          | no      |
| 651  | 0.40    | 5    | 2 / 1 / 2                    | 0.96 | -0.1608       | -0.0942 | -0.1292            | no          | no      |

## The structural reason (measured): the tag moves separation into encoding firing, but recall is source-blind by construction

The commitment anti-cheat shows the tag DOES sometimes differentiate the core into source-specific subsets (seed 651
overlap 0.4: commitment spans all three sources, H=0.96). **Yet `min M` still does not rise** -- it is in fact WORSE
than the gain=0 baseline on that row (-0.1608 vs -0.1292). The reason is the honesty constraint itself: recall drives
the WHOLE shared episode pattern with a source-blind drive (it MUST -- a source-conditioned recall drive would be the
exact source-label cheat this gate exists to prevent). So at recall of source s, every committed subset reactivates,
including the subsets that committed to the rivals, and those subsets drive their rival source memories -- the rival
burden persists. Differentiating WHICH cells fire at ENCODING does not change which cells fire at RECALL, because the
recall drive cannot depend on the source being tested. The conjunctive tag therefore relocates the differentiation to
a phase (encoding) where it cannot separate the recall-time rivals, and where it separates them (seed 650: commitment
collapses to a single source, H=0.00) it is a biased break, not a uniform one -- the same symmetry-breaking bias the
hetero lever hit.

## Anti-cheats (ALL reported; the honesty guards hold cleanly -- the NO-GO is real, not an instrument artifact)

- **(a) null control PASSES:** `source_tag_gain=0` is byte-identical to the symmetric-Hebbian overlap NO-GO on every
  row (asserted against the original `_laneC_source_monitor_overlap_sweep.evaluate_overlap`: min M, min L, and learned
  L1 all equal). The lever, not the instrument, is under test.
- **(b) recall is episode-only -- THE cheat guard -- PASSES on every arm:** at recall the source-afferent external
  current is **0.0**, the source-afferent firing is **0.0**, and the episode drive equals `drive_pA` exactly (no tag
  boost leaked into the recall drive). The tag is applied ONLY inside `experience` (`_drive_with_tag`); `recall` uses
  the base `_drive` unchanged. Non-vacuity diagnostic: a tag forcibly injected AT recall of the `seen` pattern MOVES
  the recalled winner away from `seen` (to `self_generated`/`heard`) -- proving the guard excludes a REAL label path,
  not a no-op. The honest measurement never applies it. **The margin is NOT a source-label read.**
- **(c) commitment spans three sources only sometimes** (seed 651: H=0.63 at overlap 0.2, H=0.96 at overlap 0.4; seed
  650: H=0.00, collapses to `seen`) -- the differentiation is seed-dependent and biased, and even where it is near
  uniform the margin does not improve.
- **(d) reliability FAILS on every row:** `all_dominant_correct` is not preserved and/or a source's own recall rate
  drops vs the gain=0 baseline -- the tag's extra encoding drive to one subset comes at the cost of the untagged
  subset's contribution to that source's own recall.
- **(e) zero-learned-weight control stays `strict=False`** on every row (no stepping-history artifact).

## Verdict + the wall this leaves (no-defer: a verdict on the METHOD, not the capability)

The conjunctive source-tag at encoding is a **NO-GO** for the source-monitor weakest-margin criterion, and it sharpens
the wall rather than closing it. Both encoding-side levers now agree, from opposite directions, on one structural fact:
**a source-blind recall drive over a shared core cannot separate co-resident sources**, because whatever
differentiation exists at encoding (in synapses, per hetero; in which cells fire, per this tag) is re-mixed the moment
recall reactivates the whole shared pattern. The residual capability is therefore not "sharpen the encoding" at all --
it is **make the recall reactivation itself source-selective WITHOUT a source label** (e.g. a pattern-completion /
attractor dynamic where the correct source's assembly, cued by the episode alone, competitively recruits its own
encoding subset and suppresses the rivals' subsets -- a recall-side, not encoding-side, mechanism), which is the next
method to isolate and quantify. Runner + the byte-identical null control are retained so that next method has a clean
A/B against both encoding-side arms.

Full-validation commands (calib 650/651 + dev 652/653/654 + held-out 655/656/657) are recorded for the parent to run
orphan-proof; this smoke does not run them.
