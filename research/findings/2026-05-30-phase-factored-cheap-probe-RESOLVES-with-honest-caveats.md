# Phase-factored cheap-first falsification probe: verdict RESOLVES (gate met) -- but with honest caveats; the residual-coupling is substrate-dependent and the toy under-represents it, so the spiking build must still test index survival

**Date:** 2026-05-30
**Status:** Task 1 cheap-first gate of the phase-factored integrated-loop plan. Verdict RESOLVES -> the HARD GATE permits proceeding to the spiking build (Task 2+). Recorded honestly: this is a logic de-risk + fatal-flaw screen, NOT empirical proof; one leg (residual coupling) is genuinely measured and surfaced an honest surprise.

## The result

Cheap probe (research/findings/raw/phase_factored_cheap_probe.py), multi-seed (42,43,44):

```
verdict: RESOLVES
means: single_pass_best 0.783 ; two_phase_content_update 0.993 ;
       two_phase_content_noupdate 0.999 ; two_phase_pointer 0.996
coupling_demonstrated: false
measure(42): idx_pointer 1.0 ; idx_content_noupdate 1.0 ;
             idx_content_update 1.0 ; mean_move 0.007 ; sep_gain -0.008
```

- Instrument valid: the single-pass control reproduces the conflict (single_pass_best 0.783 < the frozen 0.90 bar -- one online pass cannot clear both readouts).
- The two-phase factorization clears the bar (content_update mean 0.993 >= 0.90): both readouts pass where the single pass had them mutually exclusive.
- Verdict RESOLVES. Per the plan's HARD GATE, the spiking build (Task 2+) is permitted.

## The honest surprise (why strengthening the probe mattered)

The probe's FIRST version (committed 19ef6f1) computed the residual-coupling from MADE-UP closed forms (move = 0.6*sep; overlap = 1 - move = 0.58), so it returned its conclusion by construction -- a rubber stamp. The controller caught this in the mandatory smell-test and strengthened it (commit 23ae76e): the residual-coupling is now GENUINELY MEASURED -- real concept vectors, the project's own validated cortical separation transform (common-mode / pooled-inhibition removal), and nearest-match index resolution over 200 episodes, with and without consolidation updating the index.

The measurement CONTRADICTS the old closed form: at the toy's 16 near-orthogonal D=64 vectors, the common-mode is tiny (mean of 16 random unit vectors has norm ~0.25; removing 0.6x barely perturbs each rep), so mean_move is only ~0.007 and the order-index survives the drift WITHOUT any update -- idx_content_noupdate measures 1.0, so coupling_demonstrated is FALSE. The probe verified it can still fail: at D=4 with the same transform the coupling reappears (noupdate 0.89 < update 1.0). At the spec's D=64 the coupling is honestly weak, and the probe now says so instead of asserting it.

## What this gate DOES and DOES NOT establish (honest scope)

ESTABLISHES (legitimate cheap de-risk):
- The two-phase factorization LOGIC is sound: separating online order-recording from offline shuffled-selectivity lets both readouts pass where a single online pass cannot (the design's core claim, screened for fatal flaws -- none found).
- The residual-coupling is NOT a fatal flaw at the toy scale: a mild representational drift does not break a nearest-match order-index.

DOES NOT establish:
- The single-pass conflict and the wm(sep) selectivity map are CLOSED-FORM ASSUMPTIONS grounded in the project's already-validated finding that concept selectivity needs shuffled/interleaved presentation (v16 concept-binding; the 2026-05-19 integrated-loop iteration-4 finding). They are not fresh measurements. The conflict itself is established prior knowledge, so assuming it is fair, but the probe does not RE-prove it.
- SUBSTRATE CAVEAT (load-bearing): the toy uses near-orthogonal random vectors with TINY common-mode. The REAL substrate is different -- the Direction-arc geometry diagnostic MEASURED that real concept-pool reps have LARGE common-mode (raw pairwise cosine ~0.18-0.68 before mean-centring). In the real substrate, the offline separation transform (common-mode removal) would therefore move the reps SUBSTANTIALLY (not ~0.007), and the residual-coupling could be genuinely present. The toy under-represents it. The spiking build MUST still test whether the order-index survives consolidation drift on real reps, and should keep the consolidation-updates-the-index path wired as cheap insurance even though the toy says it may not be strictly needed.

## Decision

Gate met (RESOLVES): proceed to the spiking build (Task 2+). Carry the substrate caveat forward as a design constraint: the Task 2 two-phase controller wires the order-preserving index AND the consolidation-updates-index path (insurance), and the decisive run's smell-test must explicitly check whether the order readout survives consolidation drift on the real substrate (the toy could not settle this). The honest framing: the cheap tier screened for fatal flaws (none) and de-risked the factorization logic; the spiking build is the decisive empirical test, and the residual-coupling is the specific thing it must measure on real reps.

## Files

- Probe: research/findings/raw/phase_factored_cheap_probe.py (commits 19ef6f1 then strengthened 23ae76e)
- Tests: tests/test_phase_factored_cheap_probe.py (28 pass), tests/test_phase_factored_loop_grounding.py (6 pass / 3 skip)
- Plan: docs/plans/2026-05-30-phase-factored-integrated-loop-implementation.md
- Design: docs/plans/2026-05-30-phase-factored-integrated-loop-design.md

## Discipline

No bar moved (_PROBE_BAR frozen 0.90). No protected/frozen/moat module touched (byte-empty diff; abstention moat 7/7). stdlib+numpy only; no autograd. The controller caught the circular first version in the smell-test and strengthened the load-bearing leg to a genuine measurement BEFORE passing the gate -- the "scrutinize a PASS harder than a FAIL" discipline working as intended. Honest scope stated, not overclaimed.
