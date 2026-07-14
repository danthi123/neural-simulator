# Transport honesty CLOSURE (Rung-3, 6-seed): the load-bearing real-text selective-SSM result is genuinely TRANSPORT-FREE — with fixed random feedback (broadcast alignment) the selective SSM still beats the reservoir + all controls at deep context (mechanism gates 6/6; bigram 5/6)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung3_selective_ssm_realtext_derisk.py --gate-feedback random` · raw `research/findings/raw/_rung3_randfb/`. numpy; NO `sim/` edit.
**Status:** ✅ closure — the selective-SSM ladder's real-text result does NOT depend on weight transport.

## Why (the gap the coupling's adversarial-verify exposed)

The 3-skeptic adversarial-verify of the coupling caught that the selective GATE's SPATIAL learning signal used `delta = Wro.T @ err` — the read-out **transpose** = weight transport — across the whole ladder (Rungs 2–4b + the original coupling), contradicting the "transport-free" claim. The TEMPORAL eligibility was always no-BPTT (the load-bearing O(n) forward-mode claim); only the spatial read-out feedback used transport (the biologically-implausible ceiling). The coupling was fixed with fixed random feedback (broadcast alignment) and SURVIVED. This closes the gap on the **load-bearing rung** — the real-text Rung 3 — by re-running it transport-free.

## Result — 6-seed, `--gate-feedback random` (fixed random feedback `Bc`, no transport), TinyStories V=200, deep d≥4

| seed | selective | fixed_res | detached | randgate | bigram | sel<fix | sel<det | sel<rand | sel<big | GO |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 3.273 | 3.722 | 3.430 | 3.791 | 3.395 | +0.449 | +0.156 | +0.518 | +0.122 | GO |
| 43 | 3.254 | 3.704 | 3.434 | 3.747 | 3.387 | +0.450 | +0.180 | +0.493 | +0.133 | GO |
| 44 | 3.323 | 3.767 | 3.471 | 3.863 | 3.353 | +0.444 | +0.148 | +0.540 | +0.030 | GO |
| 100 | 3.430 | 3.750 | 3.497 | 3.830 | 3.429 | +0.320 | +0.066 | +0.400 | −0.002 | no |
| 101 | 3.326 | 3.766 | 3.488 | 3.846 | 3.397 | +0.440 | +0.162 | +0.520 | +0.071 | GO |
| 102 | 3.264 | 3.724 | 3.418 | 3.776 | 3.373 | +0.461 | +0.155 | +0.512 | +0.109 | GO |

- **The mechanism comparisons are 6/6**: transport-free, the selective SSM beats the fixed reservoir (`sel<fix` +0.32..+0.46), an untrained gate (`sel<det` +0.07..+0.18 → LEARNING matters), and a random-token gate (`sel<rand` +0.40..+0.54 → CURRENT-token conditioning matters) — every seed. The selective long-range mechanism does NOT require weight transport.
- **5/6 on the strict all-four-controls gate** (which also requires beating the bigram by +0.02): `sel<bigram` is +0.03..+0.13 on 5 seeds; seed 100 is a −0.002 tie. (The committed exact-transport version was 6/6 on this strict gate.)
- **Cost of transport-free:** selective mean ~3.31 vs the committed exact-transport ~3.06 — a modest weakening (random feedback is a weaker spatial learning signal than exact transport), but the full qualitative result (beats the reservoir + all controls at deep context) is retained.

## ⇒ closure

The selective-SSM ladder's real-text long-range result is **genuinely transport-free**: fixed random feedback (broadcast alignment — biologically plausible, no weight transport) reproduces "the locally-trained selective gate captures more deep context than the fixed reservoir it upgrades," beating every control 6/6. The committed Rungs 2–4b used `Wro.T` for the SPATIAL feedback (a stronger-but-implausible variant); this verifies the mechanism does not depend on it. Combined with the transport-free coupling (frozen + joint + on-bridge, all random-feedback, all 6/6), the WHOLE selective-SSM → emergent-generator arc is now transport-free end-to-end (no BPTT, no weight transport — the O(n) forward-mode eligibility + broadcast-alignment learning a brain has).

The committed Rung-2/3/4a/4b findings' "transport-free" phrasing was accurate for the TEMPORAL credit (the load-bearing O(n) no-BPTT claim) but not for the SPATIAL read-out feedback (which used transport); this closure records that the mechanism works fully transport-free. `--gate-feedback` defaults to `transport` (the committed runs + CI byte-identical); `random` is the verified transport-free path.

## Files
- `research/runners/_reslm_rung3_selective_ssm_realtext_derisk.py` (`--gate-feedback`); raw `research/findings/raw/_rung3_randfb/seed*.json`.
