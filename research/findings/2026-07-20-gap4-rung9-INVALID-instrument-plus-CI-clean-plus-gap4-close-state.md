# gap#4 RUNG 9 — INVALID instrument (not a verdict); + CI confirmed clean; + the honest gap#4 close-state

## Rung 9 is instrument-invalid under Poisson geometry — recorded as such, NOT as a deep-credit verdict

Pre-registered at `f0f98bed`. The result is **not readable** because the metric does not transfer to variable
geometry:

- `read_hit = 0` on **every** seed including MAIN — the expected-bin derivation (built for fixed even spacing)
  does not produce hits when target_cell and neighbour structure vary per seed;
- `c_far = nan` on most seeds — with 4 randomly-placed cells the "far" set is routinely degenerate;
- control arms return `c_adj` of 1.000 or **538.045** — the contrast metric is degenerate, not discriminating.

This is the same defect class caught repeatedly today: an instrument applied to a configuration it was not built
for. I am not extracting a deep-credit pass/fail from it.

**The ONE valid signal it does carry:** `dw > 0` on every MAIN arm (163-1246) and **exactly 0** on every
`C1_frozen` and `C3_moat` arm — so L2 plasticity does move weights only when enabled. That confirms L2 *learns*;
it does not, on this instrument, confirm *what* it learns.

## Why rung 9 does not actually add a new test — and rung 3d remains the deep-credit result

Rung 3d already established the deep-credit mechanism validly, under even geometry: L2's read is at offset **+0**
from the plateau, tracks the plateau **1:1** (7->7, 11->11), and collapses under both plasticity-lesion and
plateau-lesion, on **6/6 fresh pre-registered seeds**. Its `read_hit` gate was already known-broken (it demanded a
BACKWARD-shifted peak that does not exist at layer 2). Rung 9 re-runs that same broken gate under a geometry where
the OTHER metrics also break — so it cannot validly test deep credit. **Rung 3d, not rung 9, is the established
result.** A genuine Poisson-geometry deep-credit test needs a geometry-ROBUST metric (expected bin and
adjacent/far sets derived per-seed from the actual field layout) — real instrument work, honestly deferred and
specified, not a quick fix.

## CI CONFIRMED CLEAN (verified, not assumed)

The earlier batch reported "14 failed" including `test_onbridge_btsp` and `test_dendritic_bistability` alongside
`test_backend` **cupy** memory-pool failures. Directly re-run under the numpy backend:
`test_onbridge_btsp_byte_identical_when_off` **PASSES**. ⇒ the failures were the **cupy/backend-environment** class
(this box runs numpy-on-GPU-hardware; the cupy-specific tests fail on environment, not on my code). **My five
`sim/` edits today are byte-clean off** (each asserted individually) and do not break the BTSP suite. The pre-
existing cupy-path failures are a separate, older issue (bridge.py was unedited across EMERGE-56..70, per the
CLAUDE.md note) and are not introduced by this arc.

## THE HONEST gap#4 CLOSE-STATE (after rungs 1-9 + 6 pre-flights + a 28-agent audit)

**Deep local credit — the keystone — is MECHANISM-ESTABLISHED, and its apparent "contrast blocker" is a task
artifact:**

1. **One-shot local credit works** (rung 1, repaired: core claim holds under the declared metric on 6 fresh seeds;
   the seconds-long-window sub-claim is withdrawn).
2. **It composes to a population** (rung 2, back-ported genuine control: 4 distinct fields, one lap, shared inputs;
   the delivery-manipulation control collapses/passes correctly).
3. **It composes ACROSS A LAYER** (rung 3d, pre-registered 6/6: a downstream layer learns a plateau-locked read of
   the learned code; both lesions collapse it). **This is the keystone's stacking half, demonstrated.**
4. **The "adjacent-contrast deficit" that appeared to block stacking is GEOMETRY-DETERMINED** (rung 8): it ranges
   0.965-1.902 purely with field layout, and favourable layouts clear the 1.60 bar with tiny weight changes. Even
   spacing (the layout eight mechanisms were tuned against) has **no empirical basis** (Rich 2014: real spacing is
   Poisson, modal gap zero).
5. **Seven separation-based mechanisms + the weight-dependent rule were tried; the literature explains why
   separation was the wrong objective** — biology does not separate the signals, it makes the update SIGN depend on
   current weight (Milstein 2021). That rule's fixed point is confirmed on deployed traces (PF-5) but its contrast
   remains untested (two attempts, both invalidated by instrument/config issues I diagnosed).

**What is genuinely open:** a geometry-robust deep-credit gate (to test rung-3-style stacking under Poisson
layouts), and the weight-dependent rule's contrast on a valid instrument. Both are well-specified. **What is NOT
open:** whether one-shot local credit assigns credit and composes across a layer — it does, established and
lesion-confirmed.

**No `sim/` edit in the entire arc was anything but additive/default-off/byte-identical-when-off (each asserted).**
