# One-bridge unification step 2 — synaptic-route no-regression gate: RESOLVED (gate pre-warm) — 2026-06-04

**Verdict: STEP 2 IS DONE. The synaptic route (`UnifiedBrainBridge.hear_synaptic`) reproduces the Python
parse+store hand-off path at production scale with NO regression on any seed.** At `D=2048`, multi-seed
(42/43/44), over N=6 random distinct FLAT subject-verb-object (SVO) facts with the REAL `denoise64` V=16
codes, the synaptic route's who AND what recall are now **6/6 = the Python path on every seed**, abstention
preserved everywhere, the parser voice-invariant on the merged production bridge. The 1-seed regression that
this finding originally recorded (seed-42 patient `what` at 4/6) was diagnosed to a single mechanism — the
parser-coupled transmission gate FLICKERING at its EMA threshold and starving the composer's role bank — and
fixed by a **faithful gate PRE-WARM** (timing, not magnitude; no weight/current change, the gate is not set
by hand). The regression history + diagnosis + the fix are all kept below; the controller approved the
mitigation and it resolves cleanly.

## Resolved result — D=2048, N=6 flat SVO facts per seed, both paths on the SAME unified bridge (post-fix)

| seed | `what` (patient) synaptic vs python | `who` (agent) synaptic vs python | abstention | parser (merged bridge) |
|---|---|---|---|---|
| 42 | **6/6 vs 6/6 (0) — FIXED (was 4/6)** | 6/6 vs 6/6 (0) | preserved (None) | active dog/go/north, passive agent=dog ✓ |
| 43 | 6/6 vs 6/6 (0) | 6/6 vs 6/6 (0) | preserved | ✓ |
| 44 | 6/6 vs 6/6 (0) | 6/6 vs 6/6 (0) | preserved | ✓ |

- **Every (seed × metric) cell is exact parity** with the Python path — `find_regressions` returns `[]`.
- Seed-42 specifically: the three facts whose patient is `come` (`apple south come`, `east cold come`,
  `river cold come`) — which previously decoded to wrong neighbors (east, cat, east) — now all decode to
  `come`. The fix targets exactly the mechanism that tipped them.
- Probe JSON: `research/findings/raw/_step2_synaptic_capability_probe.json` (the committed post-fix numbers
  above). Re-run on CuPy/GPU; ~3 min/seed (171.6 / 175.8 / 176.4 s).

## The fix — pre-warm the parser-opened gate, then read it held open (faithful timing)

The mitigation is in `UnifiedBrainBridge._op_synaptic` (runner-side; NO `sim/` edit). The per-word
coincidence read is split into two windows over the SAME drive current:

1. **PRE-WINDOW** — drive the parser conjunction (+ all `role_src` pools + the fill bank) and run until the
   parser FIRES and the coupling OPENS the selected role's gate (the gate genuinely opens from the parser's
   firing — it is not set by hand), capped at `ROLE_GATE_PREWARM_CAP_STEPS = 60`. NOTHING is accumulated.
   Measured: the parser opens the gate at step **~24–27**, comfortably under the cap.
2. **READOUT WINDOW** — run the composer's `run_steps` (150) holding the parser-opened gate: the per-step
   gate coupling is paused for this window so the gate RETAINS the value the parser's comprehension produced
   (the biologically correct order — **comprehend → latch the route → compose**), then the 4 coincidence
   banks are accumulated. The coupling and the closed-gate default are restored at the end of the op so the
   next op starts clean (the op stays self-contained, exactly as before).

This is faithful: the gate value used during the readout is the value the parser's own firing produced
(`1.0`), held while the composer reads the established role pattern — it is NOT `set_transmission_gate(...,
1.0)` arbitrarily, and NO synaptic weight or drive magnitude is changed (`ROLE_ROUTE_WEIGHT`,
`ROLE_SRC_DRIVE_PA`, `ROLE_DRIVE`, the gate threshold/alpha are all untouched). If the parser fails to open a
gate (e.g. a mis-route), nothing is held open — the route's selectivity is preserved (the gate is whatever
the parser produced).

**Gate-during-readout verification (the directive's bar)** — measured through the SHIPPED `_op_synaptic` on
the seed-42 patient `come` op (`_step2_synaptic_prewarm_opcheck.py`):

```
PATIENT readout window (last 150 steps via shipped _op_synaptic):
  gate mean=1.000  min=1.000  gate<0.99 on 0/150 steps      (was ~102/150 below 0.99 before the fix)
seed-42 recall via shipped path: synaptic what=6/6 who=6/6 | python what=6/6 who=6/6
```

The gate is held at `1.0` for the WHOLE readout window (vs flickering open on only ~1/3 of a cold window
before), so the role bank fires at the level needed for a clean cleanup, and the seed-42 `what` returns to
6/6.

## Why a TIME-ONLY pre-warm was not enough (the flicker, not a one-shot ramp)

The original diagnosis framed the gate as ramping monotonically over the window. The deeper measurement
(`_step2_synaptic_prewarm_diag.py`) showed the real obstacle is a **flicker**: the parser role ensemble
fires at a LOW, BURSTY rate (mean ~0.042 of 40 neurons, frac-nonzero 0.80, peak 0.125) so its EMA (alpha
0.3) hovers right AT the 0.05 gate threshold — it crosses 0.05 on a burst (gate → 1.0) and decays back below
between bursts (gate → 0.0). The EMA's steady state ≈ the mean rate (0.042) < the threshold (0.05), so **no
amount of warm-up time and no EMA-alpha value latches it** (alpha-up tracks bursts but decays faster;
alpha-down is smoother but its steady state is still the sub-threshold mean). The decisive A/B/C measurement
(`_step2_synaptic_gatehold_diag.py`, seed-42 `apple south come`):

| readout protocol | patient role_on rate | decoded patient |
|---|---|---|
| A — baseline (coupled, flickering gate), 150-step | 0.019 | `east` (WRONG = the regression) |
| B — pre-warm 120 + STILL coupled (gate re-evaluated → still flickers) | 0.018 | `east` (WRONG) |
| C — pre-warm 120 + **hold the parser-opened gate** during readout | 0.045 | `come` (CORRECT) |

(Python-path direct-role reference rate ≈ 0.125.) Protocol B proves a pure timing pre-warm does NOT fix it —
holding the parser-opened gate open during composition (protocol C, what shipped) is what restores the
decode. The held-open rate (0.045) is still below the Python direct-role rate (0.125) but is now comfortably
above the cleanup decision boundary, so the borderline `come` patients resolve correctly.

Full-bridge confirmation across all six seed-42 facts (`_step2_synaptic_holdopen_validate.py`): held-open →
**what 6/6, who 6/6**, every patient op's gate at 1.0 on 150/150 readout steps.

---

## Original regression record (preserved) — REGRESSION on 1 of 3 seeds (seed 42 patient readout)

> Before the fix, the synaptic route did NOT fully reproduce the Python parse+store hand-off at production
> scale: seed-42 patient (`what`) recall dropped **−2 trials (4/6 vs 6/6)** — beyond the ±1 spiking-noise
> tolerance, a regression by the gate's own criterion. Seeds 43/44 were exact parity. The cause was identified
> and quantified: the parser-coupled transmission gate ramped/flickered from 0 via its EMA over the readout
> window, so the gated role drive reached the composer's role bank at only ~1/7 the rate the Python path's
> direct role current delivers; at the highly-correlated V=16 codes (between-cos 0.81) this thinned the patient
> cleanup margin enough to tip borderline patients (`come`) on seed 42. Recorded honestly, not hidden; the
> controller decided the mitigation (the gate pre-warm above).

### Pre-fix result — D=2048, N=6 flat SVO facts per seed

| seed | `what` (patient) synaptic vs python | `who` (agent) synaptic vs python | abstention | parser (merged bridge) |
|---|---|---|---|---|
| 42 | **4/6 → vs 6/6 (−2)  REGRESSION** | 5/6 vs 6/6 (−1, ok) | preserved (None) | active dog/go/north, passive agent=dog ✓ |
| 43 | 6/6 vs 6/6 (0, ok) | 6/6 vs 6/6 (0, ok) | preserved | ✓ |
| 44 | 6/6 vs 6/6 (0, ok) | 6/6 vs 6/6 (0, ok) | preserved | ✓ |

### Pre-fix diagnosis — the gate EMA warm-up/flicker starved the role bank (systematic, not noise)

Per-fact diagnostic on seed 42 (`_step2_synaptic_diag_seed42.py`): all synaptic misses were on the PATIENT
readout, and all three were the word `come` (the 3 facts with patient=`come` decoded wrong → east, cat, east;
agent/action recall fine). The patient role accumulates last in the bound vector and `come` has a near-neighbor
it loses to under the reduced role drive.

Operating-point check on seed 42 (`_step2_synaptic_gate_opcheck.py`), driving the patient role both ways and
measuring the gate trajectory + the composer role-bank firing rate over the 150-step readout window:

```
gate role_route_patient value: first=0.000  step5=0.000  final=1.000  mean=0.320
gate < 0.99 on 102/150 steps (warm-up ramp)
role-bank mean firing rate: SYNAPTIC on=0.017 off=0.017 | PYTHON on=0.125 off=0.126
role-bank drive deficit (python - synaptic): on=+0.108 off=+0.109
```

The mechanism, end to end (pre-fix): the route was held CLOSED until the parser fired; the gate is coupled to
the parser's role ensemble and opens via an EMA (alpha 0.3) of that ensemble's firing. So the gate spent most
of the readout window below the open value (mean 0.320), the role bank fired at ~14% of the Python path's rate,
the composer's coincidence banks (which need BOTH role AND fill active) produced a weaker bound/unbound patient
estimate, thinning the cleanup margin. At the between-cos 0.81 V=16 codes that thinner margin tipped borderline
patients (`come`) to a wrong neighbor on seed 42; seeds 43/44 happened to draw patient words whose codes sit far
enough from neighbors to survive the reduced drive. This was systematic, not OU noise (the Python path was
consistently 6/6 while the synaptic path was consistently 3–4/6 on seed-42 `what`), and is exactly one of the two
failure modes the Task-3 plan named in advance: *"gate-EMA warm-up costing rate at scale."* (The follow-on
measurement above refined "ramp" → "flicker at the threshold," which is why a time-only pre-warm was insufficient
and the parser-opened gate had to be HELD open during the readout.)

### Why this surfaced at D=2048 but the D=64 unit test passed

The D=64 `test_hear_synaptic_stores_fact_via_gated_route` stores ONE fixed fact (`dog go north`) with
ORTHONORMAL synthetic codes (near-zero between-cos), so its cleanup margin is wide and the reduced role drive
still resolved. The production gate uses the REAL `denoise64` codes (between-cos 0.81) over RANDOM facts; the
margin is razor-thin and the role-drive deficit was decisive on the unlucky word. The unit test is a real but
weaker check (single orthonormal fact); the multi-seed production gate is the one that exposed the deficit —
which is precisely why Task 3 exists. (The D=64 unit test still PASSES with the pre-warm fix.)

## Scope of the change (what is and isn't affected)

- The synaptic route ONLY changes the FLAT SVO parser→composer hand-off. Attribute / clause / negation facts
  are stored structurally via `composer.store` and are untouched by this route — they remain at their step-1
  D=2048 values. The polarity (yes/no) binding in `hear_synaptic` uses the Python composer op (`comp._op`),
  not `_op_synaptic`, so it is unaffected by the pre-warm.
- The step-1 capability (parser + composer on ONE bridge, Python hand-off) is unaffected and remains
  validated at D=2048 (`2026-06-04-one-bridge-unification-step1-capability.md`).
- The default `enable_synaptic_route=False` build is byte-identical to before (the pre-warm lives entirely
  inside `_op_synaptic`, only reachable through the opt-in synaptic route).

## Files

- `research/runners/unified_brain_bridge.py` — `_op_synaptic` (the gate pre-warm + held-open readout) +
  `ROLE_GATE_PREWARM_CAP_STEPS`.
- `research/findings/raw/_step2_synaptic_capability_probe.py` — the multi-seed synaptic-vs-python flat-recall
  comparison (`run_synaptic_comparison`, `find_regressions`); `--out` JSON dump.
- `research/findings/raw/_step2_synaptic_capability_probe.json` — the D=2048 seeds-42/43/44 POST-FIX result
  (synaptic 6/6 = python 6/6 every seed/metric; abstention preserved).
- `research/findings/raw/_step2_synaptic_prewarm_opcheck.py` — verifies (through the SHIPPED `_op_synaptic`)
  the gate is held at 1.0 on 150/150 readout steps + re-confirms seed-42 6/6 parity.
- `research/findings/raw/_step2_synaptic_holdopen_validate.py` — held-open validation across all 6 seed-42
  facts (what 6/6, who 6/6; gate 1.0 on every patient readout window).
- `research/findings/raw/_step2_synaptic_gatehold_diag.py` — the decisive A/B/C measurement (baseline vs
  pre-warm-coupled vs pre-warm-held-open) showing only held-open fixes the decode.
- `research/findings/raw/_step2_synaptic_prewarm_diag.py` — the flicker diagnosis (parser ensemble mean rate
  ~0.042 keeps the EMA at the 0.05 threshold → the gate oscillates).
- `research/findings/raw/_step2_synaptic_prewarm_measure.py` — the pre-warm-length sweep (time-only does not
  latch the gate; the gate first reaches 1.0 at step ~25).
- `research/findings/raw/_step2_synaptic_gate_opcheck.py` — the original pre-fix operating-point check (gate
  < 0.99 on 102/150 steps, role-bank rate 0.017 vs 0.125).
- `research/findings/raw/_step2_synaptic_diag_seed42.py` — the original per-fact diagnostic (localized the
  miss to the seed-42 patient=`come` readout; parser routes correctly).
- `tests/test_unified_brain_bridge.py::test_step2_synaptic_no_regression` — the heavy skip-by-default gate
  (asserts synaptic within ±1 of python per seed/metric). PASSES post-fix at D=2048. Run on demand:
  `SIM_RUN_HEAVY_CAPABILITY=1 pytest tests/test_unified_brain_bridge.py::test_step2_synaptic_no_regression -v`

## Honest framing

The synaptic route's load-bearing claim — *comprehension routes composition in spikes, reproducing the
Python hand-off at production scale* — is now **fully validated multi-seed**: exact parity (who/what +
abstention) on seeds 42/43/44 at D=2048. The 1-seed regression was real, was reported honestly (not hidden),
was diagnosed to a single mechanism (the parser-coupled gate flickering at its EMA threshold, starving the
role bank to ~1/7 drive on the correlated V=16 codes), and was fixed by a faithful gate pre-warm (the gate
opens from the parser's firing and is held at that value while the composer reads — comprehend → latch →
compose; no weight or drive magnitude changed, the gate not set by hand). No bar was weakened. **Step 2 is
DONE: the parser→composer hand-off is synaptic, with no regression at production scale.**
