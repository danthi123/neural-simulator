---
type: finding
status: qualified
date: 2026-09-04
mechanism: configural-binding width-robustness sweep (--conj-bind none|fixed --conj-mode prod --conj-offset-max 4, n in {1024,1088,1152,1216,1280,1344}) re-testing the 2026-09-03 n1152 6/6 crossing across width
lane: vision (identity readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: the n1152 6/6 crossing is real and reproducible but is NOT a robust plateau — capability_go oscillates 4-6/6 across widths with no monotonic trend; binding's apparent +1-seed edge over the flat control at n1152 is cancelled by a -1 at n1344 (both arms sum to an identical 27/36 over the sweep), and the GO/NO-GO call matches between arms at every width tested
artifacts:
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1024_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1088_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1216_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1280_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1344_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1024_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1088_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1152_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1216_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1280_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1344_6seed.json
---

# The n1152 vision-identity crossing is a fragile peak, not a robust plateau — binding's seed-boost is noise across widths

**Status:** width-robustness re-test (6 widths x 2 arms x 6 seeds = 72 runs) of the 2026-09-03 n1152 6/6 crossing
(`1a3e0bfc`, `2026-09-03-vision-configural-binding-crossing-is-mostly-capacity-anticheat-caught-it.md`). The
crossing itself reproduces exactly, but it does NOT generalize to neighboring widths: neither arm shows a
monotonic or plateaued relationship between width and `capability_go`, and binding's headline +1-seed advantage
over its width-matched flat control at n1152 is exactly cancelled by a -1 at n1344.

## Artifacts

<!--derived-->
All 12 runs of the sweep (6 widths x {FLAT, BINDING}), cited throughout:

| width N | FLAT (`--conj-bind none --n-s2 N`) | BINDING (`--conj-bind fixed --conj-mode prod --conj-n N --conj-offset-max 4`) |
|---|---|---|
| 1024 | research/findings/raw/lanes/perception/conjbind_widthctrl_n1024_6seed.json | research/findings/raw/lanes/perception/conjbind_bindarm_n1024_6seed.json |
| 1088 | research/findings/raw/lanes/perception/conjbind_widthctrl_n1088_6seed.json | research/findings/raw/lanes/perception/conjbind_bindarm_n1088_6seed.json |
| 1152 | research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_6seed.json | research/findings/raw/lanes/perception/conjbind_bindarm_n1152_6seed.json |
| 1216 | research/findings/raw/lanes/perception/conjbind_widthctrl_n1216_6seed.json | research/findings/raw/lanes/perception/conjbind_bindarm_n1216_6seed.json |
| 1280 | research/findings/raw/lanes/perception/conjbind_widthctrl_n1280_6seed.json | research/findings/raw/lanes/perception/conjbind_bindarm_n1280_6seed.json |
| 1344 | research/findings/raw/lanes/perception/conjbind_widthctrl_n1344_6seed.json | research/findings/raw/lanes/perception/conjbind_bindarm_n1344_6seed.json |

The n1024/n1152/n1280 BINDING points reproduce the pre-existing `conjbind_prod_n{1024,1152,1280}_6seed.json`
(same config, identical per-seed outcomes — confirmed by direct comparison) — those three widths are not new
compute, only re-cited here under the sweep's naming.

## Result — the width x arm table

<!--derived-->
Deepest-bar `capability_go` (the `primary_code` = `"count"` entry's `by_code.count.summary.verdict_fracs
.capability_go`, cross-checked against `per_seed_capability_go`; strict GO = >=5/6), FLAT = `--conj-bind none
--n-s2 N` (`conjbind_widthctrl_nN_6seed.json`), BINDING = `--conj-bind fixed --conj-mode prod --conj-n N
--conj-offset-max 4` (`conjbind_bindarm_nN_6seed.json`), 6-seed (42/43/44/100/101/102):

<!--derived-->
| width N | FLAT capability_go | BINDING capability_go | delta (bind-flat) | GO/NO-GO call (>=5/6) |
|---|---|---|---|---|
| 1024 | 5/6 | 5/6 | +0 | GO / GO |
| 1088 | 4/6 | 4/6 | +0 | NO-GO / NO-GO |
| 1152 | 5/6 | **6/6** | **+1** | GO / GO |
| 1216 | 5/6 | 5/6 | +0 | GO / GO |
| 1280 | 4/6 | 4/6 | +0 | NO-GO / NO-GO |
| 1344 | 4/6 | **3/6** | **-1** | NO-GO / NO-GO |
| **sum /36** | **27/36** | **27/36** | **0** | — |

<!--derived-->
Three facts fall out of this table directly. (1) The FLAT control's own strict-GO crossing is not a plateau: it
clears >=5/6 at 3 of 6 widths (1024, 1152, 1216) and misses at the other 3 (1088, 1280, 1344), oscillating
between 4/6 and 5/6 with no trend as width grows — a system sitting on the resolution boundary of a 6-seed
measurement, not one that has settled above it. (2) Binding's own strict-GO pattern crosses at the SAME 3
widths and misses at the SAME 3 widths as flat — the GO/NO-GO call never diverges between arms anywhere in the
sweep. (3) Summed over all 6 widths, FLAT and BINDING pass an IDENTICAL 27 of 36 seed-width cells. Binding is
not, on average, better than its width-matched flat control over this range — it is exactly as good, with more
variance (its single best result, 6/6, and single worst result, 3/6, both belong to binding; flat never leaves
the 4-5/6 band).

## Per-seed decomposition — the pattern is seed identity, not width or binding

<!--derived-->
Reading each arm's `per_seed_capability_go` array by seed INDEX (not by width) shows the oscillation in the
table above is driven by a handful of idiosyncratically hard/easy seeds, not by a smooth function of width or
by the binding mechanism:

<!--derived-->
| seed | FLAT: widths passed (of 6) | BINDING: widths passed (of 6) |
|---|---|---|
| 42 | 6/6 | 2/6 |
| 43 | 5/6 | 5/6 |
| 44 | 4/6 | 5/6 |
| 100 | 6/6 | 6/6 |
| 101 | **0/6** | 3/6 |
| 102 | 6/6 | 6/6 |

<!--derived-->
Seed 101 fails at EVERY width under FLAT (0/6) — width never rescues it, so "5/6 at n1152" for flat is really
"the other 5 seeds pass, 101 never does," and 4/6-vs-5/6 elsewhere is decided by whether one further seed among
{43, 44} also happens to fail at that width. Seed 42 plays the equivalent role for BINDING, failing at 4 of 6
widths (1024, 1088, 1216, 1344) but passing at 1152 AND 1280. The n1152 6/6 for binding is the one point in the
sweep where BOTH of binding's idiosyncratically-hard seeds (42 and 101) happen to clear the bar simultaneously
— not a width effect, a coincidence of which seed draws land where for a fixed-random conjunction bank.

## What this means for the 2026-09-03 crossing claim (confirms / tempers / falsifies)

The 6-seed n1152 result itself is unchanged and reproduces exactly: re-running the identical config
(`conjbind_bindarm_n1152_6seed.json`) against the original `conjbind_prod_n1152_6seed.json` gives the SAME
per-seed outcomes at n1024, n1152 and n1280 (the 3 widths the two sweeps share), so this is not a determinism
or provenance problem — `git_sha 65c5a334` for every file in this sweep. What the width sweep changes is the
INTERPRETATION of that single data point.

**CONFIRMS, and more strongly than before:** the 2026-09-03 finding's "mostly capacity, not binding" reading.
Averaged over the sweep, binding and flat are not just close — they are IDENTICAL (27/36 each). The earlier
"+1-seed marginal boost" language undersold how thin the binding contribution is; a mean of zero over 6 widths
is a cleaner statement than a marginal-but-real boost at one width.

**FALSIFIES:** reading n1152's 6/6 as evidence of a stable, capacity-or-binding-driven plateau. It does not
hold one step in either direction (n1088 = 4/6, n1280 = 4/6 for both arms), and the specific mechanism behind
it — two chronically-hard seeds both clearing at once — is a seed-draw coincidence, not a property of width
1152. A different 6-seed panel at the same width would very plausibly land at 5/6 or 4/6 instead.

**TEMPERS:** the vision-identity `capability_go` bar itself. Across this width range both arms hover at a mean
of 4.5/6 (75%), well inside "close to but not reliably above the strict 5/6 GO bar" — occasional 6/6 and
occasional 3/6 are both within reach of the same underlying system, which is the signature of noise straddling
a threshold rather than a capability cleanly on one side of it.

## Honest verdict + next

The first vision-identity `capability_go` crossing (1a3e0bfc) is real as a measurement and was never in
question; what this sweep retracts is the informal sense that n1152 marked a settled operating point worth
building on. It does not: the crossing is a fragile peak, and configural binding's contribution to it, measured
honestly across width, is statistically indistinguishable from zero on this sweep (+1 at one width, -1 at
another, 0 net). Two honest openings follow. (1) A capability that toggles GO/NO-GO on single-seed draws needs
either more seeds per width (12+ to shrink the +-1-seed resolution floor) or a held-out-position / scramble-null
anti-cheat repeated at every width, not just n1152, before any width is called a stable operating point. (2) The
seed-level decomposition above suggests the fixed-random conjunction bank (and the fixed-random flat S2 bank)
each have a small number of "unlucky" draws baked in at initialization — a LEARNED (not fixed-random) conjunction
selection is the more promising lever than further width, since it could target exactly the seeds a fixed random
draw currently strands.

## Reproduce

```bash
# The 12-run width x arm sweep (each already on disk; shown for reproducibility):
for N in 1024 1088 1152 1216 1280 1344; do
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
      --ridge 0.5 --conj-bind none --n-s2 $N \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/lanes/perception/conjbind_widthctrl_n${N}_6seed.json
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
      --ridge 0.5 --conj-bind fixed --conj-mode prod --conj-n $N --conj-offset-max 4 \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/lanes/perception/conjbind_bindarm_n${N}_6seed.json
done
```
