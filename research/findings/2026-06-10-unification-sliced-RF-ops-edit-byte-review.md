---
type: finding
status: live
date: 2026-06-10
mechanism: fhrr
---

# ⚠️ FOR OWNER BYTE-REVIEW — sliced-RF-ops co-residence edit (sim/bridge.py), default-off byte-identical (2026-06-10)

**The protected `sim/` edit for STRICT RF co-residence on one bridge (roadmap step 2).** It is the minimal
edit the de-risks (5a, 5b) pointed to: slice the resonate-and-fire (RF) ops to the RF neurons so the FHRR
composer can run on a slice of the shared navigation bridge without touching the Izhikevich (navigation)
slice — and vice versa. **Default-off (no mask) is byte-identical; please byte-review before I build the merge
on it.**

## The diff (31 insertions, 6 deletions; only two RF-only methods)

`sim/bridge.py`:
- `rf_kick(..., neuron_mask=None)` — new optional bool array (True = RF neuron). When None (default), the v/u
  writes are the SAME two statements as before (just hoisted into named locals `_kick_re`/`_kick_im`). When
  set, the kick is written ONLY at masked positions.
- `_rf_advance_one()` — when `_rf_neuron_mask` is set, the spike-crossing mask is `& mask` and the advanced
  state is written back ONLY at masked positions. When None, the three write-back statements are byte-identical
  to before.

No other code path is touched. The Izhikevich / Hodgkin-Huxley / AdEx dynamics and the global step-loop
dispatch are byte-unchanged by construction — the edit lives entirely inside the RF-only methods, which are
inert unless `rf_kick`/`rf_resonate_steps` are called.

## Why this is the right (minimal) edit — not a core-step-loop dual-dispatch

From the production composer's actual usage (`rf_phasor_composer.py`): the RF composer is STATELESS across ops
(re-`rf_kick`s each op) and stores its memory in COMPLEX synapses (`cp_rf_w_re`/`cp_rf_w_im`, array-disjoint
from `v`/`u` AND the navigation's real-valued `cp_connections`). So the RF slice's `v`/`u` need NOT survive a
navigation `_run_one_simulation_step`; the only real requirement is that a composer op not clobber the
navigation slice's `v`/`u`. Slicing the RF ops achieves exactly that with the lowest possible blast radius.
(Full rationale: `2026-06-10-unification-5b-rf-izh-KILL-and-minimal-edit-approach.md`.)

## Byte-identity proof (default-off == baseline)

1. **The mask-None code path is the prior statements verbatim** (the `if _rf_mask is None:` branches), plus a
   no-op `_rf_neuron_mask = None` assignment and a `getattr(..., None)` lookup — no change to any existing
   computation.
2. **18/18 production conversational tests pass VERBATIM** (`tests/test_core_sim_composition.py` +
   `tests/test_brain_conversational_agent.py`, 359s on GPU) — these build real RF composer bridges and assert
   exact outputs incl. the no-confab `is None` abstention moat. The composer passes no mask → byte-identical.
3. **The 5b RF reference is unchanged** after the edit (|z| = 1.0000 after one RF step; the kill demo
   reproduces identically).

## Edited-version validation (the new masked path is correct)

`research/runners/derisk_unification_5b_edited.py` (GPU) + `tests/test_rf_neuron_mask_coexistence.py` (2 tests,
1.6s):

| Claim | Result |
|---|---|
| RF op on a masked slice of an Izhikevich bridge == a standalone RF bridge | **EXACT** (max\|Δphase\| = 0.000) |
| The co-resident Izhikevich slice's v/u byte-identical across the RF op | **True** (v and u unchanged) |
| The RF op actually ran on the slice (RF slice off-rest) | True |

## Co-residence orchestration (how the merge uses it)

On the merged bridge: navigation runs `_run_one_simulation_step` (harmlessly clobbering the idle RF slice's
`v`/`u` between conversational ops — re-kicked each op); when the agent converses, the composer runs its ops
on the RF slice via `rf_kick(..., neuron_mask=rf_mask)` + `rf_resonate_steps`, leaving navigation's `v`/`u`
intact. Runner-side wiring constraint: the RF neurons carry only complex synapses (no `cp_connections`
out-edges into navigation) — already true of the composer's `connections_per_neuron=0` bridges.

## What I will NOT do until you byte-review

Build the full nav+conv merge (task 12) ON this edit. It is committed default-off (changes nothing for any
current run) and byte-proven, but per the standing discipline I will not RELY on the masked path for the merge
until you have reviewed the diff above. Also note the 5a clip caveat for the merge: frozen conversational
REAL-valued weights (parser role-routes, dlPFC edges) must sit within the shared bridge's clip bounds; the RF
composer's COMPLEX binding weights are immune to that clip.
