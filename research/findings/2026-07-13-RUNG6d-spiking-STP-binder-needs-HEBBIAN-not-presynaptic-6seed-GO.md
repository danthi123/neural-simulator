# RUNG 6d — the spiking-STP realization of the novel-referent binder needs HEBBIAN short-term potentiation, NOT Tsodyks-Markram presynaptic facilitation (6-seed GO; the mechanism the on-substrate build must use)

**Date:** 2026-07-13
**Runner:** `research/runners/_stp_facilitation_binder_derisk.py` (numpy STP-dynamics model; reuse-by-import the RUNG6c task + metric; NO `sim/` edit).
**Verdict:** the mechanism question for the fully-spiking binder is RESOLVED — HEBBIAN (pre×post, WTA-gated) short-term potentiation binds; PRESYNAPTIC (TM) does not. 6-seed GO for the Hebbian rule.

## The question (the spiking rung of RUNG6c)
RUNG6c validated the novel-referent binder with a generic Hebbian outer-product fast weight (6-seed GO). The fully-spiking rung realizes it in short-term plasticity (Mongillo 2008: facilitation `u` decaying with `tau_f` = `sim`'s `cp_stp_u`). **THE MECHANISM QUESTION:** Tsodyks-Markram STP facilitation is PRESYNAPTIC — `u` rises wherever the PREsynaptic barcode fired, regardless of which slot won. Can that do SELECTIVE content-addressable binding (bind barcode_e → ITS slot, not all slots)? Or is HEBBIAN (pre×post) short-term potentiation required (`u` rises only on the WTA-winner slot)?

## The result (numpy STP-dynamics model; the RUNG6c held-out-novel + entity-deref + lesion metric)
Per-synapse facilitation `u[slot, barcode-bit]`; a NORMALIZED (cosine) content-addressable match reads the slot; retrieve if cosine > θ=0.6 else allocate a fresh slot; `u` decays exp(-Δ/`tau_f`) between clauses.
| STP rule (seed 42) | novel-track | collisions |
|---|---|---|
| **presynaptic** (TM-faithful: `u[:,bits]+=`, all slots) | **0.004** | **0.999** (non-selective — all entities collapse together) |
| **hebbian** (WTA-winner only: `u[winner,bits]+=`) | **0.546** | **0.000** |

**Hebbian rule, 6-seed (42/43/44/100/101/102):** **6/6 GO** — novel-track mean **0.531** (= the RUNG6c 0.525 ceiling), binding-penalty **0.000**, collisions **0.000** every seed, merge-lesion **0.000** (identical codes collapse), no-bind chance. The absolute 0.53 inherits the D3 autoregressive-rollout ceiling (RUNG6c's separate axis).

## The two load-bearing findings for the on-substrate build
1. **Presynaptic TM STP is NON-SELECTIVE (a real gap):** because `u` rises on ALL barcode→slot synapses when the barcode fires (presynaptic-only, no postsynaptic gate), every entity's facilitation lands on every slot → 99.9% collisions. So **`sim`'s vanilla STP (`cp_stp_u`, presynaptic) alone does NOT realize the binder.**
2. **HEBBIAN short-term potentiation DOES (6-seed GO):** the facilitation must be gated to the WTA-winner (pre×post) — a postsynaptic/coincidence gate. **On the substrate this is reuse-by-import:** `sim` has a RATE-WINDOW Hebbian rule with a `hebbian_coactivity_decay` (`cp_hebb_coactivity_trace`, `bridge.py:1467`/`6966`) — a decaying pre×post coactivity trace = exactly the Hebbian short-term potentiation WITH the facilitation-fade window — plus FS-WTA lateral inhibition (`sim/regions.py`, the merged-bridge `sel_*` + `sel_FS`) for the winner-selection. The cosine (normalized) read is scale-invariant → robust to BOTH facilitation decay (uniform scale) and overlapping-barcode cross-facilitation.

## ⇒ Next: the on-substrate build (now precisely scoped)
A `SimulationBridge`: barcode-input → K slot-pools synapses, PLASTIC via the rate-window Hebbian (coactivity-decay = the Mongillo window, `tau_f`~1.5s → decay~0.9/clause per the RUNG6c window sweep) + FS-WTA between slot pools. Present a barcode → FS-WTA picks the winner slot → the Hebbian coactivity (barcode × winner) potentiates barcode→winner (decaying = the fast weight); re-present → the potentiated winner fires (retrieve); a novel barcode → FS-WTA opens a fresh slot. Cheap-first on the bridge: present barcode_e twice with distractors between → the same slot fires (via the potentiated coactivity) vs a fresh slot for a novel barcode; held-out-novel + FS-WTA read + merge/no-bind lesions, 6-seed. Likely reuse-by-import (rate-window Hebbian + FS-WTA both exist); a `sim/` edit only if a faithful mechanism needs it.

Reuse-by-import; NO `sim/` edit. Runner: `_stp_facilitation_binder_derisk.py` (`--rule presynaptic|hebbian --theta`).
