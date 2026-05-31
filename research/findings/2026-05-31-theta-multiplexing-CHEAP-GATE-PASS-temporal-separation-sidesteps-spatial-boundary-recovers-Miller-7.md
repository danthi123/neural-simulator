# Theta-multiplexing cheap-first gate: PASS (survives scrutiny). Temporal (theta-phase-slot) separation HOLDS N>=4 dense-stable items with positive decode margin while the no-phase superposition control collapses -- AND it holds the OVERLAPPING (between-cos 0.60) codes that this session's spatial DG separation could NOT separate. Under realistic phase jitter the permissive capacity (16) falls to exactly ~7, recovering Miller's working-memory number from theta/gamma ratio + phase precision. This de-risks the owner-preferred next arc and PASSES the hard gate to motivate the spiking build. Named open risk for the spiking build's adversarial review: the cheap model assumes a reader that already knows each item's phase slot; the spiking build must test whether phase-addressing is LEARNABLE and STABLE across encode/recall.

> ## ⚠️ CORRECTION / PARTIAL RETRACTION (2026-05-31, same day, on checking prior in-project work)
> This probe RE-DERIVED already-validated work; the "FIRST positive de-risking result" framing
> is RETRACTED and the "motivates the spiking build" disposition is WRONG. The project had ALREADY
> validated theta-gamma multiplexing at the ALGEBRA level with DECISIVE controls BEFORE this probe:
> - 2026-05-24-direction-E-theta-gamma-multiplexing-ALGEBRA-VALIDATED-controls-decisive.md: PERFECT
>   1.000 at loads {2,3,5,7}; THREE controls decisive (permutation->chance, no-slot-windowing->chance,
>   high-overlap-vocab->robust). N_GAMMA=7 the natural ceiling. Explicitly ALGEBRA-ONLY.
> - 2026-05-23 cheap FHRR probe: "Lisman-Idiart N16 realisable on FHRR"; capacity envelope wide,
>   algebra survives substrate-realistic noise.
> So my RESOLVES + overlap-robustness + Miller-7 is an INDEPENDENT RE-CONFIRMATION of already-validated
> algebra, not a novel result. The only genuinely-additive bit is the Miller-7 capacity-recovery-under-
> phase-jitter (a minor biology-translatable addendum), and even N_GAMMA~7 was already the documented cap.
> CRUCIALLY: the algebra was NEVER the bottleneck. When theta-gamma multiplexing was actually BUILT INTO
> THE SPIKING SUBSTRATE for composition, it hit a DECISIVE 5-architecture CONVERGENT CEILING:
> 2026-05-20-THETA-GAMMA-decisive-honest-negative-...md (GATE=FAIL; per_regime_advantage NEGATIVE at N=5;
> cue-suppression-during-retrieve gives an ANTI-effect, violating the encoding-specificity principle).
> Therefore "cheap gate PASS -> motivate a NEW spiking build" is FALSE: that spiking build exists and
> was ceiling'd. The genuinely-open next direction (per the 2026-05-20 pre-registration) is the 6th
> architecture = generative replay + PFC-held compositional frame WITHOUT cue-suppression, or honest
> closure -- NOT another phase-multiplexing variation. Root-cause of this overclaim: I built the cheap
> probe WITHOUT first reading the extensive 2026-05-19..05-24 in-project theta-gamma work, violating the
> standing "check existing work first" directive. The body below is preserved as written (pre-correction);
> read it through this banner. The scrutiny IN the body (margin/jitter/Miller-7) is sound; the NOVELTY +
> DISPOSITION claims are the retracted part.

**Date:** 2026-05-31
**Status:** Cheap-first CPU/numpy gate PASSED + scrutinized for the theta-phase-multiplexing conversational-holding arc (grounded in the prior note 2026-05-31-theta-multiplexing-...-NEXT-ARC-grounding-...md). This is the FIRST positive de-risking result for the owner's preferred biological conversational direction (2026-05-19 reframe: theta-multiplexing, not static retrieval) after this session's two biological attempts hit boundaries (integrated-loop VOID, DG separation-vs-reliability BOUNDARY). Run during the GPU-busy window while the P4 multi-hop hub-reuse test runs -- CPU-only, zero GPU contention.

## What was tested (faithful, non-rigged Lisman-Idiart model)

Probe `research/findings/raw/_theta_multiplex_holding_probe.py` (throwaway; stdlib+numpy only; no protected import). Theta period = 70 phase bins; each item i = a gamma assembly active over a gaussian window (FIXED biological width, FWHM 10 bins -> ~7 non-overlapping slots) centered at evenly-spaced phase phi_i. Multiplexed buffer x(t) = sum_i env_i(t) v_i + noise (neighbors BLEED in time when slots crowd -- the capacity limit is EMERGENT, not imposed). Readback item i: sample x(phi_i), decode by spatial cosine to stored codes (standard readout). NO-PHASE CONTROL: all phi_i identical -> every readback samples the SAME superposition -> mutual collapse (rate <= 1/N). A phase PASS while the control collapses is a genuine falsification, not trivial.

PRE-REGISTERED FROZEN BAR (set before the run, never tuned): N>=4 dense items all read back >= 0.90 from phase slots AND no-phase control collapses (< 0.50). Three-state RESOLVES / BOUNDARY / DOES-NOT-RESOLVE; instrument-validity checked first (N=1 -> 1.0; codes unit; control superposes).

## Result + the load-bearing scrutiny

Instrument valid (N=1 -> 1.000; control superposes). Multi-seed 42/43/44, 40 trials.

| codes | N | phaseRead | decode margin | ctrlRead |
|---|---|---|---|---|
| near-orthogonal dense | 4 | 1.000 | 0.252 | 0.217 |
| near-orthogonal dense | 7 | 1.000 | 0.222 | 0.121 |
| near-orthogonal dense | 16 | 0.949 | 0.091 | 0.070 |
| OVERLAPPING dense (cos 0.60) | 4 | 0.996 | 0.093 | 0.233 |
| OVERLAPPING dense (cos 0.60) | 7 | 0.989 | 0.074 | 0.118 |
| OVERLAPPING dense (cos 0.60) | 16 | 0.819 | 0.028 | 0.066 |

PRE-REGISTERED VERDICT: **RESOLVES** -- N=4 phaseRead 1.000 (>= 0.90) AND N=4 ctrlRead 0.217 (< 0.50). The hard gate to motivate the spiking build is PASSED.

Three scrutiny checks applied (scrutinize a PASS harder than a FAIL):

1. DECODE MARGIN (is the PASS confident or lucky argmax?): near-orthogonal margin +0.22 to +0.25 at N<=7 -- solidly positive, confident decode. Overlapping codes thin the margin (+0.07 to +0.09) but it stays positive -- overlap costs margin exactly as expected, not a failure.

2. THE BOUNDARY-ESCAPE (the whole point): OVERLAPPING dense codes (between-cos 0.60 -- the SAME stable-but-unseparated regime that this session's spatial DG separation could NOT separate, the 4000-dense point on the boundary curve) are read back at 0.989 (N=7) via phase slots while the spatial-only control collapses (0.118). Temporal separation carries the distinguishability that spatial separation could not. This is the mechanistic demonstration that theta-multiplexing routes AROUND the spatial competitive-k-WTA knob that produced the boundary.

3. CAPACITY REALISM (is capacity=16 real or a permissive artifact?): at zero phase jitter the model holds up to 16 (>> biological ~7) because spatial-cosine decode is robust to temporal bleed. Adding realistic phase jitter (sample slot i at phi_i + N(0, jitter)) collapses capacity toward biology:

| phase jitter (bins) | capacity (max N with readback >= 0.90) |
|---|---|
| 0.0 | 16 |
| 2.0 | **7** |
| 4.0 | 4 |
| 6.0 | 2 |

At jitter ~2 bins (20% of the 10-bin burst width -- modest, realistic phase imprecision) capacity is EXACTLY 7 -- Miller's working-memory number, recovered from the theta/gamma timescale ratio + phase precision. So the no-jitter 16 is confirmed a permissive artifact, AND the model is biologically faithful (it reproduces the ~7 capacity), not merely permissively passing. Biology-translatable insight: WM capacity ~7 is the theta-period / (gamma-burst-width x phase-precision) ratio.

## Honest scope + the named open risk (for the spiking build's adversarial review)

What this de-risks: the CORE claim that temporal (theta-phase) separation can HOLD multiple dense-stable items -- including spatially-OVERLAPPING ones -- distinguishably, with a realistic ~7 capacity, where spatial separation alone could not. This is exactly the boundary-escape the next arc needs.

What this does NOT de-risk (the load-bearing open risk to test in the spiking build): the cheap model assumes a reader that ALREADY KNOWS each item's phase slot (it samples x(phi_i) directly). In the real spiking substrate the downstream reader must LEARN/lock to each item's phase, and phase assignment must be STABLE across encode and recall. The cheap gate says nothing about phase-addressing learnability/stability -- that is the first thing the spiking build (and its adversarial review) must attack. Also: the raw within-item cosine "reliability" metric is noise-norm-dominated (noise norm 3.35 vs signal 1) so it reads ~0.08 despite perfect decodability; the decode-MARGIN is the meaningful reliability proxy and it is healthy. Claim scoped to decodability-with-margin.

## Disposition + next concrete action

The cheap-first HARD GATE for the theta-multiplexing arc is PASSED (RESOLVES), and the PASS survived three scrutiny checks + recovered biology. Per the staged discipline (same gate that correctly KILLED the integrated-loop and denoiser builds at the cheap stage), the spiking build is now MOTIVATED -- but it is the bigger step and must go through: (a) reuse-by-import wiring of the parked integrated-loop theta-timing controller + concept-pool gamma assemblies, no new autograd; (b) a DEDICATED ADVERSARIAL REVIEW of the phase-addressing-learnability risk BEFORE any decisive multi-seed run; (c) a frozen verdict module; (d) controller-only decisive run with the smell-test. The very next concrete step is to write the spiking-build design/plan (brainstorm -> design doc -> TDD plan) for the theta-multiplexing holding mechanism on the real bridge, with phase-addressing learnability as the load-bearing question.

This composes with the in-flight P4 multi-hop hub-reuse verdict independently: whatever that returns, this theta-multiplexing arc is the owner-aligned biological direction and its cheap gate has now passed.

## Discipline

Throwaway CPU probe only; no protected/frozen/moat/sim/runner/builder module touched; stdlib+numpy only; SIM_BACKEND=numpy. Pre-registered bar set before the run and NOT tuned by results. The PASS was scrutinized harder than a FAIL (margin + boundary-escape + capacity-realism), the permissive-capacity artifact was caught and corrected (jitter -> ~7), and the load-bearing open risk (phase-addressing learnability) is named up front for the spiking build's adversarial review rather than buried. The boundary-escape claim is now an EVIDENCED cheap result, not just a hypothesis -- but the spiking instantiation remains to be falsified.
