---
type: finding
status: no-go
date: 2026-09-01
runner: research/runners/_replay_dg_pattern_separation_readfix_cuerelative.py
artifacts:
  - research/findings/raw/sep_readfix/cuerelative_6seed_gain1000.json
  - research/findings/raw/sep_readfix/cuerelative_6seed_gain2000.json
  - research/findings/raw/sep_readfix/cuerelative_6seed_gain300.json
  - research/findings/raw/sep_readfix/cuerelative_writebias_6seed_gain2000.json
---

# Board #91 read-side attempt 2: a per-granule cue-relative excitability bias CONFIRMS the collapse locus is intrinsic membrane heterogeneity (novel-cue anti-cheat 0/6 -> 6/6) but does NOT move `both_win` off 0/6 -- the WRITE still encodes the uncorrected code

**Board #91** — lane-H memory-separator, READ-side frontier. The prior finding
(`2026-09-01-board91-memory-separator-readfix-popstate-reset-NOGO-relocalizes-to-consolidation-induced-generic-capture.md`)
showed the post-consolidation DG read collapses onto the dominant engram for ANY
cue, including a genuinely novel/untaught one (mean Jaccard 0.90 vs 0.44
pre-consolidation, 6/6 seeds), and that resetting the population-set-point's full
host state changes nothing (0/6 either arm). Its explicit handoff: attack the DG
competition's cue-selectivity directly, naming as the cheapest concrete candidate
"normalize each DG granule's effective threshold/gain by its OWN drive under the
CURRENT cue relative to its drive under a broad reference set." This finding builds
and tests exactly that mechanism, in two variants, and reports a **decisive, DEFINED
NO-GO** on the board's `both_win` bar — but with a real, causally-verified positive
sub-result that sharpens the locus further than #90/#91 could.

## Verdict

**NO-GO** for both tested variants. `both_win` (the dg->answer behavioral read
completing correctly for BOTH memories) stays **0/6** at every gain tried
(0 to 12,000 pA, read-only variant) and is actively **worse** when the same bias is
also applied during consolidation (write+read variant: novel-cue fix regresses to
0/6, and one seed's dominant-memory recall flips from win to loss). Both preconditions
hold for the READ-ONLY variant at the reported gain (OFF arm reproduces the #90/#91
residual; no seed regresses on the dominant memory's own probe relative to OFF), so
this is a genuine negative on the board's bar, not an instrument failure — `Verdict`
reports `NO-GO`, not `UNDEFINED`. The write+read variant fails its own precondition
(a seed DOES regress), reported as `UNDEFINED` for that arm's own verdict (a
regression means the manipulation is doing something uncontrolled, not that it is
cleanly negative) and is banked as a tried, refuted extension.

**But two anti-cheats DID move, causally and cleanly, which is the deliverable of
this file**: (1) a genuinely novel/untaught cue no longer spuriously completes to
the dominant engram — `novel_no_spurious_completion` goes from the #91 baseline's
**0/6** to **6/6** (novel->eng0 Jaccard drops from ~0.90-1.00 to 0.20-0.42) at
gain>=1000, confirmed on the held-out seeds (44/101/102) never used to pick the
gain, not just the tune seeds (42/43/100); (2) the DG-level "subordinate cue
reactivates its own engram" diagnostic improves from **0/6** to **3/6**
(`m1_reactivates_own`). Both effects are attributable to the manipulation (0 at
gain=0-300 that don't correct the underlying excitability gap; 6/6 and 3/6 once the
gain is large enough to actually move the ranking — see Results).

## What was built (two variants, no `sim/` edit)

**Wiring check before building** (pure host arithmetic, no sim run): the fixed
random perforant projection (input->dg, `dg_fan_in=12` of `n_input=48`) gives every
granule an almost identical EXPECTED raw overlap with any input assembly (measured
directly from the wiring adjacency for 3 seeds: mean~6, std~1.5 for m0/m1/novel
alike). The wiring alone does not privilege specific granules for specific cues, so
the universal-winner set is not a static wiring artifact.

**Diagnostic that located the real intrinsic property** (new this file, pure
per-neuron array read, no consolidation needed): DG engrams for THREE completely
different cues (m0, m1, and an independently-drawn novel cue), measured on a
FRESH, never-consolidated bridge, all cluster on the SAME low membrane-capacitance
(`cp_izh_C`) subset of granules:

| seed | eng0 mean C | eng1 mean C | novel-engram mean C | non-member mean C | overall mean/std |
|---:|--:|--:|--:|--:|--:|
| 42 | 85.7 | 84.2 | 86.9 | 110.2 | 101.1 / 15.3 |
| 43 | 84.3 | 85.8 | 86.0 | 106.5 | 99.3 / 13.3 |
| 44 | 87.2 | 89.3 | 85.8 | 107.8 | 100.3 / 13.4 |

(`cp_izh_a`, `cp_izh_b`, `cp_izh_d_increment` show no comparable separation.) Lower
capacitance means faster membrane response (dV/dt = I/C) for the SAME synaptic
drive -- a granule with C~85 reaches threshold faster than one with C~110 for
essentially any sufficiently strong input, REGARDLESS of which specific cue drove
it. This is present from build time (the bridge's own "Applied heterogeneity to 4
parameters" step, seeded from `cfg.seed`), before any replay/consolidation runs,
and explains the PRE-consolidation baseline overlap the #91 finding measured
(mean novel->eng0 Jaccard 0.44) as well as why it is CUE-INDEPENDENT rather than
subordinate-cue-specific. It does not, by itself, explain the further amplification
to 0.90 post-consolidation (that additional escalation is presumably the #78
population set-point's cumulative-recruitment/basket-drive integrator interacting
with this pre-existing bias over 28 replay events — not re-derived here, banked as
the likely mechanism for future work).

**Variant A -- READ-ONLY bias** (`--mode read`, the file's default): on the
already-consolidated bridge, before scoring a real cue, drive `n_ref=8`
independently-drawn generic reference patterns (a third RNG stream, disjoint from
m0/m1/novel) through the identical read protocol and record each granule's average
firing count (its measured excitability on THIS substrate instance, whatever caused
it). z-score across granules and inject `bias_g = -gain * z_g` as a STATIC
per-granule current into `dg` for every subsequent read (m0, m1, novel, and the
real behavioral probe). The write (`base._consolidate`) is untouched.

**Variant B -- WRITE+READ bias** (`--mode writebias`): the SAME bias, but measured
on the FRESH pre-consolidation bridge (cheaper -- no post-consolidation run needed,
since the C-based gap is present from t=0) and installed as a persistent
`_run_one_simulation_step` wrapper BEFORE `_consolidate` runs, so the WRITE and
every READ see the same corrected DG code.

Biology: divisive/subtractive gain-control relative to a population reference
(Carandini & Heeger 2012, *Nat Rev Neurosci* 13:51-62), applied per-granule rather
than as the single population-scalar set-point #78/#91 already tried and refuted
— this is "candidate 3, made concrete" from the #91 handoff. The underlying
substrate property it targets (fixed, heterogeneous intrinsic excitability shaping
which cells win a competition, independent of the specific input driving it) is the
well-established biological phenomenon of intrinsic-excitability heterogeneity
(Marder & Goaillard 2006, *Nat Rev Neurosci* 7:563-574; Padmanabhan & Urban 2010,
*Nat Neurosci* 13:1276-1282 — biophysical diversity among otherwise-similar neurons
measurably shapes which cells respond to a given input). Real dentate granule cells
are known to be highly heterogeneous in intrinsic excitability, and healthy DG
pattern separation is understood to depend on strong, FAST, input-locked
feedforward inhibition dominating that heterogeneity (Pouille & Scanziani 2001,
already cited by this lineage) -- this file's result is consistent with that
picture: on THIS substrate, intrinsic heterogeneity is currently strong enough
relative to the feedforward-inhibition timing to decide the k-WTA winner set
largely on its own.

## Results

**Variant A, read-only bias, 6 seeds (42/43/44/100/101/102), tuned on 42/43/100,
verified on held-out 44/101/102** (`cuerelative_6seed_gain1000.json`,
`..._gain300.json`, `..._gain2000.json`):

<!--derived-->
_Numbers from the cited artifacts. `both_win` = the dg->answer behavioral probe
picks the correct answer for BOTH memories. `novel_ok` = novel cue's Jaccard to
BOTH taught engrams stays < 0.5. `m1->own` = m1's isolated DG reactivation Jaccard
favors its own engram over the dominant one._

| gain (pA) | both_win ON | m1_reactivates_own ON | novel_ok ON | m0 regressed vs OFF (per-seed) |
|---:|:--:|:--:|:--:|:--:|
| 0-300   | 0/6 | 0/6 | 0/6 | no |
| 1000    | 0/6 | 3/6 | **6/6** | no |
| 2000    | 0/6 | 3/6 | **6/6** | no |
| 12000   | 0/6 | (not re-measured; novel_ok 3/3 on tune seeds) | 3/3 (tune seeds) | no (tune seeds) |

Pooled at gain=1000 (the reported operating point): `both_win_on_count=0`,
`m1_reactivates_own_on_count=3` (up from OFF's 0), `m0_wins_on_count=4` == 
`m0_wins_off_count=4` (identical per-seed pattern -- no regression), 
`novel_no_spurious_completion_on_count=6` (up from OFF's implicit 0, matching #91).
`attributable_to both_win ON vs OFF` reports **UNDEFINED: both arms are ~0** (both
0/6 -- correctly flagged as a null, not "0% attributable", since there is no effect
to attribute). At every seed where m1 newly reads its own engram (43/44/100), m0
simultaneously STOPS reading its own (42/101/102 keep m0, lose m1) -- the bias
redistributes WHICH SINGLE engram wins per seed; it does not produce two
independently cue-locked engrams.

**Variant B, write+read bias, gain=2000, 6 seeds** (`cuerelative_writebias_6seed_gain2000.json`):
worse on every axis that moved for Variant A: `novel_no_spurious_completion_on_count`
**0/6** (regresses -- novel->eng0 back up to 0.77-1.00), `m1_reactivates_own_on_count`
**1/6**, and one seed (101) flips its OWN dominant-memory probe from win (OFF
m0_sel=+0.12) to loss (ON m0_sel=-0.14) -- a genuine regression that trips the
Verdict's own precondition (reported `UNDEFINED`, not a clean negative). Installing
the correction persistently interacts badly with the #78 population set-point's
own error-integration over the 28-event replay, rather than helping it settle a
cleaner code.

## Why `both_win` never moves even when the DG-level diagnostic partially does

The dissociation is precise and, we think, the most useful output of this file:
`base._consolidate` (the WRITE, in Variant A) runs BEFORE any bias is measured or
applied -- it writes the dg->answer coincidence through the SAME uncorrected,
intrinsically-collapsed DG activity #90/#91 characterized. Variant A's bias then
corrects WHICH granules fire at READ time, but the LEARNED weights still point at
whichever granules fired during the (uncorrected) WRITE. Even on a seed where the
isolated reactivation diagnostic now shows m1 recruiting its own engram (a
DIFFERENT set of granules than fired during m1's own consolidation replay), those
newly-recruited granules were never the ones whose dg->answer synapses got
potentiated -- so the behavioral read still fails, or reads through whatever
residual weight structure exists on the OLD (dominant) granules. Variant B's
attempt to close this gap (correct the write too) instead destabilizes the
population set-point, which was tuned and validated (#78, GO) against the
UNCORRECTED excitability landscape.

## Relation to a prior-arc negative with the identical failure signature

`.venv-rag/bin/python tools/rag/rag_search.py` (`--corpus finding`) surfaces
`2026-08-07-source-monitor-attractor-competition-NO-GO-single-gcomp-knob-wta-does-not-track-cue-joint-storage-separation-knob-needed.md`
— a DIFFERENT arc (source-monitoring attractor competition) hit the SAME failure
shape: a single scalar competition-gain knob amplifies one attractor's dominance
across ALL cues rather than tracking which cue is presented, and that finding's own
conclusion was that a single competition knob cannot serve both joint-storage and
separation; a SEPARATE, cue-tracking knob was needed. This file's own result is
consistent with that lesson at the mechanism-design level: a single per-instance
static bias (however well-targeted at the real underlying variable, here `cp_izh_C`)
is still a SCALAR-PER-GRANULE correction, not a genuinely CUE-CONDITIONED one, and
plateaus for the same structural reason.

## Where the next lever should go (banked, not built here)

1. **A genuine CA3-style completion stage** (the #90/#91 finding's candidate 1, the
   heavier build, now the best-supported remaining option): since a shallow
   excitability-leveling bias demonstrably CAN break the cue-independent capture
   (proven by the 6/6 novel-cue fix) but cannot on its own produce two
   independently-recoverable, WRITE-consistent engrams, a small recurrent
   auto-associative stage that completes toward each memory's OWN private core
   (using the corrected, leveled DG code as its input) is the most direct next
   step -- it would need to run consistently across both replay (write) and
   probe (read), which Variant B's failure shows cannot be done by simply
   installing the SAME bias unconditionally throughout.
2. **Decouple the #78 population set-point's error signal from raw excitability.**
   Variant B's regression suggests the set-point (tuned against the uncorrected
   landscape) needs to track CUMULATIVE RECRUITMENT of a bias-corrected activity
   signal, not raw firing counts, if the two mechanisms are to compose -- worth a
   cheap, separate test (feed the set-point's `nfired` count the bias-corrected
   spike set) before building the heavier completion stage.
3. **A CUE-CONDITIONED (not per-instance-static) version of this file's bias** —
   e.g., recompute the reference-relative z-score continuously against a hebbian
   trace of RECENT cue-driven activity rather than a fixed pre/post-consolidation
   snapshot, so the correction itself can adapt within a single read rather than
   applying one frozen per-granule offset for the whole window.

**GO-gate for any of the above** (unchanged from #91): `both_win` >= 5/6 seeds, WITH
the OFF arm reproducing the #90/#91 residual, no regression on the dominant memory
(OFF-relative, per-seed -- this file corrects a flaw in the naive "6/6 ceiling" bar:
the TRUE baseline is not at ceiling on the behavioral probe metric on every seed to
begin with, so the fair test is "does ON ever lose a seed OFF won", not "does ON
win literally every seed"), and the novel-cue anti-cheat (<0.5 Jaccard to both
taught engrams), on held-out seeds 44/101/102 never used for tuning.

## Reproduce

    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_cuerelative \
        --seeds 42 43 44 100 101 102 --gain 1000 \
        --out research/findings/raw/sep_readfix/cuerelative_6seed_gain1000.json

    # write+read variant (banked negative):
    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_cuerelative \
        --seeds 42 43 44 100 101 102 --gain 2000 --mode writebias \
        --out research/findings/raw/sep_readfix/cuerelative_writebias_6seed_gain2000.json

    # gain tuning sweep (TUNE_SEEDS 42/43/100 only):
    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_cuerelative --tune

(6-seed wall-clock: ~15-30s per run, numpy/CPU; the read-only variant additionally
runs 8 reference-cue reads per bridge instance, the write+read variant runs them
once pre-consolidation.)

## Tracked scaffolds (host, not brain)

Inherited from #71/#78/#90/#91: host-defined input patterns + answer assemblies;
host reinstatement of each memory's input during replay; scheduled down-states; the
WRITE/READ transmission-gate phase; a rate-window Hebbian coactivity write; the
population set-point PI controller (host, unmodified in Variant A; its interaction
with Variant B's bias is measured, not modified). NEW this file: the reference-cue
excitability measurement and the resulting per-granule bias current are a
runner-side host computation + current injection (not a `sim/` edit); the
`cp_izh_C` correlation diagnostic is a read-only inspection of existing per-neuron
arrays, no mechanism change. Every read/probe and the dg->answer write stay
on-substrate spiking.

## Sources

EXTERNAL-SEARCH-RAN: DG/CA3 pattern separation vs completion; intrinsic-excitability
heterogeneity and network coding; population/contrast gain-control normalization
(logged via `tools/before_you_build.sh`, 2026-09-01; RAG corpus search against
`kandel` and `finding` corpora for granule-cell excitability heterogeneity and
winner-take-all/attractor competition).

- Carandini, M., Heeger, D.J. (2012). Normalization as a canonical neural
  computation. *Nat Rev Neurosci* 13:51-62 — the divisive/subtractive gain-control
  framing this file's per-granule reference-relative bias implements.
- Marder, E., Goaillard, J.-M. (2006). Variability, compensation and homeostasis in
  neuron and network function. *Nat Rev Neurosci* 7:563-574 — intrinsic
  excitability heterogeneity as a general, biologically normal property of neural
  circuits (motivates why this substrate's heterogeneity is not itself a bug, only
  its DOMINANCE over cue-specific drive in this particular competition regime).
- Padmanabhan, K., Urban, N.N. (2010). Intrinsic biophysical diversity decorrelates
  neuronal firing while increasing information content. *Nat Neurosci* 13:1276-1282
  — biophysical (e.g. capacitance/conductance) diversity among similar neurons
  measurably shapes population coding, directly relevant to the `cp_izh_C`
  correlation this file measured.
- Pouille, F., Scanziani, M. (2001). *Science* 293:1159-1163 — already cited by
  this lineage; fast, input-locked feedforward inhibition is the biological
  mechanism DG relies on to keep intrinsic heterogeneity from dominating the read,
  which this substrate's current regime does not yet achieve.

Internal: reads and does NOT re-derive
`2026-09-01-board91-memory-separator-readfix-popstate-reset-NOGO-relocalizes-to-consolidation-induced-generic-capture.md`
(#91 popreset — the cue-independent-capture characterization this file's mechanism
targets) and its own #90/#78 lineage. Cross-arc citation (new this file):
`2026-08-07-source-monitor-attractor-competition-NO-GO-single-gcomp-knob-wta-does-not-track-cue-joint-storage-separation-knob-needed.md`
— an independent arc's identical failure signature (a single scalar competition
knob cannot track cue identity), corroborating this file's structural diagnosis
that a scalar-per-granule (not cue-conditioned) correction plateaus for the same
reason. Board ledger: `GAP_CLOSURE_MISSION.md` names #91 "READ-TIME (CA3 completion
onto the private core / clear reactivation persistence)"; this finding banks
candidate 3 (per-cue-relative competition gain) as tried in two forms and
insufficient alone, adds a causally-verified mechanistic diagnosis (intrinsic
`cp_izh_C` heterogeneity) that neither #90 nor #91 identified, and sharpens
candidate 1 (CA3-style completion) as the best-supported remaining lever.
