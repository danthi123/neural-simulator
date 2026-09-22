---
type: finding
status: live
lane: load-bearing
date: 2026-09-21
---

# The four seed-dependent BORDERLINE load-bearing faculties: an operating-point diagnosis (2026-09-21)

The 6-seed adequate load-bearing fraction (`2026-09-21-load-bearing-fraction-6seed-adequate-0.85-robust-core-20.md`)
found a robust core of 20/26 faculties load-bearing in ALL of seeds {42,43,44,100,101,102} and FOUR seed-dependent
BORDERLINE faculties — load-bearing in some but not all seeds: `episodic-memory` (5/6, off@s44),
`affect-marker-spiking-wta` (4/6, off@s42,s43), `source-provenance-honesty` (4/6, off@s44,s102), `prospective-memory`
(5/6, off@s44). Each faculty's INTEGRATED decision field is a boolean derived from a CONTINUOUS spiking read crossing a
FIXED host CONSTANT. This finding MEASURES, per seed, that continuous read + its threshold for the intact AND lesion
arm at the operating-point drive the load-bearing battery uses, so the MARGIN to the decision — and WHY it flips in some
seeds not others — is exposed. Applying the project's deepest lesson: **"what else does the real system run ALONGSIDE
this, that we replaced with a CONSTANT?"**

## Method + artifacts

`research.runners._lbf_borderline_operating_point --seed <s>` builds each of the four organs at seed `s` via its OWN
production constructor (`EpisodicRecallOrgan(seed)` / `ProspectiveMemoryOrgan(seed)` / `AffectDrivesWorkspace(seed)` /
`SourceProvenanceHonestyMonitor(seed)`) — exactly how `webapp/server.py` builds it (each reads `_brain_chat_seed()`) —
drives it with the verbatim battery turn texts, and reads the continuous quantity + threshold for both arms. numpy CPU,
memcap 10G, six seeds. Artifacts: `research/findings/raw/_lbf_borderline/op_s{42,43,44,100,101,102}.json` (per-seed
reads, prov-sidecarred) + `research/findings/raw/_lbf_borderline/diagnosis_verdict.json` (the aggregation + the EARNED
`tools.verdict.Verdict` block: STATUS=GO, 7/7 preconditions). Ground truth = the committed AWS `_adequate6/` +
`_pmem_v2_6seed/` per-seed verdicts. The runner's `--selftest` passes (no brain build). DEFAULT paths untouched — this
is a new read-only diagnostic module; it imports and calls production organ entry points, changes none.

## Per-seed MARGIN to the decision threshold (this box, numpy CPU)

Margin = (continuous read) − (threshold); a POSITIVE margin is the load-bearing side (intact clears, lesion collapses).
All values below are rounded from the exact per-seed reads in the cited `op_s*.json` + `diagnosis_verdict.json` (`stats`).

<!--derived-->
| faculty (quantity vs threshold) | s42 | s43 | s44 | s100 | s101 | s102 |
|---|---|---|---|---|---|---|
| episodic-memory (apical_cue − COMPLETE_MIN 0.20) | +0.80 | +0.30 | +0.69 | +0.69 | +0.80 | +0.58 |
| prospective-memory (rel − FIRE_THR 0.20) | +0.141 | +0.127 | **−0.016** | +0.066 | +0.074 | +0.103 |
| affect-marker-spiking-wta (WTA margin − DEAD_MARGIN 0.05) | −0.006 | −0.017 | −0.010 | +0.110 | −0.013 | −0.010 |
| source-provenance-honesty (intact opponent read d, threshold sign=0) | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |

<!--derived-->
Lesion arms (collapsed control): episodic apical_cue=0.0 @ all seeds; pmem rel≈0.02–0.04; affect WTA margin≈0.0;
provenance d=0.0 @ all seeds.

## The four diagnoses — what was replaced by a constant, and the candidate stabilizer

<!--derived-->
| faculty | what flips it | the constant replacing a companion process | candidate biology-grounded stabilizer |
|---|---|---|---|
| prospective-memory | GENUINE near-threshold fragility: `rel` (held×cue coincidence) dips to 0.184, −0.016 UNDER `FIRE_THR=0.20`, only @s44; +0.07..+0.14 elsewhere. Margin-sign predicts the AWS verdict 6/6. | `FIRE_THR=0.20` is a fixed release threshold; nothing regulates the held BA10 assembly's firing rate toward a reliable set-point across the per-neuron heterogeneity draw. | firing-rate homeostasis (synaptic scaling / intrinsic-excitability set-point, Turrigiano) on the held intention assembly, targeting `rel` to a set-point margin above `FIRE_THR` regardless of seed. |
| episodic-memory | The RECALL is robust: `apical_cue` clears `COMPLETE_MIN` by +0.30..+0.80 at EVERY seed incl. s44 (organ-level store formed at all 6). The AWS off@s44 is the STORE failing to form the assembly in the integrated build context. | `COMPLETE_MIN=0.20` is cleared robustly (not the crux). The real gap: no assembly-SIZE homeostat at the STORE — the CA3 assembly emerges in a NARROW `kthresh=8` firing-threshold window; the DG pattern-separation set-point (`D5_SEP_BIAS`) exists but is DISARMED (`_default_sep_bias()=0.0`). | a BTSP-plateau assembly-size homeostat (per-CA3 intrinsic-excitability set-point) guaranteeing a minimum reliable assembly across seeds/context. (The existing `sep_bias` is shrink-only and soaks 5/6 — a partial, not a fix.) |
| affect-marker-spiking-wta | SEVERE near-threshold fragility: the WTA winner-minus-runner-up margin is ~0.033–0.044, straddling `DEAD_MARGIN=0.05`, at 5/6 seeds (only s100 decisive @0.160). The verdict flips with seed AND build-context RNG AND hardware (AWS thread=2 got winners at 4 seeds; this box, episodic-first context, only s100). Mood/level are robust (level 2–3 at all seeds). | `DEAD_MARGIN=0.05` is a fixed rate-separation bound; nothing drives the lateral-inhibition competition to a DECISIVE winner. Second fragility: the WTA settle steps on OU noise off the process-global RNG and is NOT `_isolated` (unlike the ladder read), so the margin depends on the whole preceding RNG history. | a homeostatic inhibitory-gain set-point on the WTA lateral inhibition (canonical cortical microcircuit decisiveness) to force margin ≫ `DEAD_MARGIN`; PLUS RNG-isolate the read (reuse the ladder's `_isolated`). |
| source-provenance-honesty | NOT a brain operating point. The intact opponent read `d≈1.000` at EVERY seed (perceived pool wins decisively, rate_generated=0). The "borderline" is the LESION CONTROL: learning-off → both pools silent → `d=0` → a genuine tie-break COIN FLIP on the discretized label. `np.random.default_rng(seed)` makes it seed-deterministic, landing on 'perceived' (agreeing with intact) at exactly s44,s102 → spurious "pass". | Not a homeostatic proxy — an INSTRUMENT artifact: the battery compares the discretized SIGN (label), destroying a robust, seed-independent d≈1.0-vs-0.0 separation via a 50/50 tie-break. | INSTRUMENT fix (cheap, no-defer, not biology): compare the continuous `d` (robust every seed), or make the degenerate lesion read deterministic. Flips source-provenance to robustly 6/6 load-bearing. |

## The bounded fix decision — honest characterization, NO fix forced (HARD RULE 2)

Per the timebox (diagnosis for all 4 + at most ONE principled fix) and the no-fabrication rule: **no biology-grounded
stabilizer is CLEARLY earned within this arc**, so none was built. Each candidate above is a real spiking-mechanism
build needing its own de-risk + 6-seed and carries a live risk of becoming a tuned knob (episodic's `sep_bias` is
already known to only shrink assemblies and soak 5/6, not close s44). The one clearly-earned, non-tuned resolution is
source-provenance's, but it is an INSTRUMENT fix (compare the robustly-separated continuous `d`, not the coin-flip
label), not a homeostatic brain stabilizer — a cheap follow-on rung, staged, not built here. Under the no-defer law
these are METHODS banked with named biological surpasses (homeostatic set-points), not abandoned capabilities.

## What this changes

<!--derived-->
- Two of the four "seed-dependent borderline" faculties are GENUINE near-threshold operating-point fragilities
  (**prospective-memory**: `rel` −0.016 under `FIRE_THR` @s44; **affect-marker**: WTA margin straddling `DEAD_MARGIN`
  at 5/6 seeds), each with a named homeostatic surpass. **episodic-memory**'s recall is robust (+0.30..+0.80 every
  seed) — its fragility is the STORE (emergent assembly formation), a different lever. **source-provenance-honesty**
  is NOT fragile at all (intact d≈1.0 every seed) — its borderline is an INSTRUMENT artifact (the lesion control's
  coin-flip), fixable in the instrument. So the honest actionable remainder from the 0.85 finding narrows: 2 homeostatic
  operating-point levers, 1 store-formation lever, 1 instrument fix.
- The instrument is part of the emulation: the affect-marker WTA read's non-`_isolated` RNG makes its verdict
  build-context-dependent, and the source-provenance label comparison manufactured a seed-dependence that the
  underlying robust read never had.

## Next

1. (cheap, no-defer) source-provenance INSTRUMENT fix: a default-off probe comparing the continuous `d`, 6-seed
   re-verify → expect robustly 6/6 load-bearing.
2. prospective-memory + affect-marker homeostatic set-point stabilizers (each its own de-risk + 6-seed; guard against
   the tuned-knob failure mode).
3. episodic STORE assembly-size homeostat (distinct from the recall read, which is robust).
