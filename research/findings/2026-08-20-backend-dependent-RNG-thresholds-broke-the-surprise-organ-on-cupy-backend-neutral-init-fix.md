---
type: finding
status: live
date: 2026-08-20
mechanism: backend-neutral-thresholds
lane: integration
seeds: [42]
seed-waiver: A backend-PARITY fix verified by threshold-hash IDENTITY (numpy vs cupy) + a within-run discrimination read + a GNW GO — not a stochastic effect size. The evidence is determinism/parity (same seed → same thresholds across backends), which a seed population does not measure.
instrument: research/runners/_gnw_two_distinct_organs_derisk.py (cupy vs numpy) + the surprise organ's own confirm/contradict discrimination read + threshold md5 hash across backends
runner: research/runners/_spiking_expectation_rpe_derisk.py
external: NO-EXTERNAL-NEEDED — an in-repo backend-determinism bug + its fix using existing engine flags; the fix is a numerical-conditioning parity fix, not a literature question.
artifacts:
  - research/findings/raw/_gnw_two_distinct_organs/summary.json
---
# A GPU faculty silently mis-behaved because per-neuron thresholds are drawn from the ACTIVE backend's RNG — the surprise organ broke on cupy; `backend_neutral_izh_initialization` fixes it

Artifact: research/findings/raw/_gnw_two_distinct_organs/summary.json

**One line.** The production `SurpriseProductionOrgan` discriminates agreement from conflict on numpy but NOT on cupy —
which blocked wiring the genuinely-distinct second organ into the GNW bus on the production (GPU) server. The root cause
is a GENERAL backend-determinism hazard, not organ-specific, and the fix is an existing engine flag.

## The hazard (general — this can bite ANY cupy faculty sensitive to threshold precision)
<!--derived-->
Per-neuron homeostatic firing THRESHOLDS are drawn from the **active backend's** RNG (`sim/bridge.py`, `cp.random.uniform`
under cupy vs `np.random` under numpy). For the SAME seed the two backends produce **different thresholds** (numpy
threshold-sum −45039.07 vs cupy −45123.98). With `enable_homeostasis=True` (the config default) those thresholds ARE the
spike thresholds. Everything else was byte-identical between backends (learned weights 135241 vs 135243, Izhikevich
params) — ONLY the thresholds differed. This is the same CLASS as the "seed never controlled the substrate" bug
([[2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE-the-deep-credit-arc-was-confounded-by-unseeded-neurons]]) and the
"CPU-validated faculty ships a GPU-only defect" class (2026-08-11): a faculty validated only on numpy can silently
mis-behave on the cupy production path.

## Why it broke the surprise organ specifically (a near-cancellation operating point amplifies the tiny threshold shift)
<!--derived-->
The first hypothesis — catastrophic float32 cancellation in the GABA_A subtractive op `I_syn = g_e*(E_e−v) + g_i*(E_i−v)`
(`sim/kernels.py`) — was FALSIFIED: computing it in float64 and casting back left cupy's confirm rate bit-unchanged
(0.7161). The real chain: the surprise organ sits at a near-cancellation CONFIRM operating point where the FS
`patient_expected` prediction pool must fire hard to cancel the drive. The hardware-dependent thresholds shifted that
stiff FS pool's firing ~3× (**173 Hz numpy → 55 Hz cupy**), so its GABA_A inhibition of the surprise pool collapsed and
CONFIRM fired spuriously (block-4: 0.0 Hz numpy → 3.30 Hz cupy) — the organ stopped discriminating agreement from conflict.

## The fix (existing engine flags, default-on, NO `sim/` edit)
Set `cfg.backend_neutral_izh_initialization=True` (+ optional `_arithmetic`) in `build_expectation_circuit`
(`_spiking_expectation_rpe_derisk.py`, +24 lines). These route threshold init through a **host RNG identical across
backends** + explicit IEEE round-to-nearest ops. **Numerical-conditioning only — the GABA_A subtractive-inhibition biology
is unchanged.** `init` is the load-bearing part (fixes discrimination alone); `arith` adds a per-step strict cupy kernel
(measurable chat latency) and is optional — kept default-on per faithfulness>speed, separately toggleable.

## Verified both directions
<!--derived-->
- **numpy: byte-identical no-op.** Threshold md5 identical with flags on vs off (`a8576dec0659`); confirm/contradict/novel
  bit-identical (`0.19531250 / 5.43981481 / 5.26620370`); 50 backend-neutral + determinism unit tests pass. The host RNG
  reproduces numpy's own draw, so numpy is provably unchanged.
- **cupy: now discriminates.** confirm 0.716→0.203; cupy threshold hash now EQUALS numpy's (`a8576dec0659`); block-4
  3.30→0.0. The **GNW two-organ de-risk is GO on the real RTX 3090** (`summary.json`, `--backend cupy`, seed 42): organ B
  discriminates (agree 0.351 ≪ thr 1.401 ≪ disagree 2.677), coincidence_2hop 1.000, every anti-cheat 0.000. Pre-fix this
  run went inert (`committed=None`).

## Consequence + the broader action
This unblocks the genuinely-distinct second organ on the production GPU backend → the GNW two-organ bus can go
default-on on cupy (production wiring end-to-end re-verification in flight before the flip). **Broader:** any GPU faculty
whose behaviour turns on per-neuron threshold precision should be audited against this — the `backend_neutral_izh_*`
flags are the general parity fix, and a numpy-only validation is not sufficient evidence for the cupy production path.
The one-brain MERGE path (default-off) builds from its own config-superset and would need the same two flags if it ever
goes cupy-production. (Agent-diagnosed — first hypothesis falsified, root cause then precisely localized to the
backend-RNG thresholds; parent verified the GNW cupy GO + the numpy byte-identity from the artifacts.)
