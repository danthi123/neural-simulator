# RUNG 6e (in progress) — the on-substrate STP binder: FS-WTA + Hebbian RETRIEVE work on a real bridge; emergent FRESH-SLOT ALLOCATION is the identified sub-problem (familiarity-gated)

**Date:** 2026-07-13
**Runner:** `research/runners/_stp_binder_onbridge_derisk.py` (real `SimulationBridge`: barcode region + K slot pools + FS-WTA, barcode→slot plastic via rate-window Hebbian; numpy-CPU; NO `sim/` edit).
**Status:** IN PROGRESS — WTA + Hebbian-retrieve validated on-substrate; the first-mention fresh-slot allocation is the precise open sub-problem.

## What works on the substrate (concrete progress)
Porting RUNG 6d's mechanism (Hebbian short-term potentiation + FS-WTA) to a real bridge:
- **Slot neurons fire** with ~500 pA (direct-drive 0.175 rate) — healthy.
- **The barcode→slot synaptic conductance is weak** (g_e≈0.033 → ~2 pA effective, ~250× below firing), so a raw synaptic drive won't fire the slots (a0-measured: g_i=0.157 ≫ g_e; weight 2.5→40 didn't scale the slot rate because the FS auto-balances). **Fix = tonic-bias + synaptic-DIFFERENTIAL WTA:** a tonic ~400 pA brings the slots near threshold; the barcode→slot conductance provides the differential → a clear FS-WTA winner (w_bs=40, tonic=400, FS=3 → winner margin 0.028).
- **Hebbian RETRIEVE works (2/2):** presenting a barcode binds a winner (Hebbian potentiates barcode→winner via the rate-window coactivity), and RE-presenting retrieves that same slot. The content-addressable retrieve is realized on the substrate.

## The identified sub-problem: emergent FRESH-SLOT ALLOCATION
`bind={0:0, 1:0}` — both a first entity AND a second (distinct) entity win **slot 0** (`distinct=False`). The barcode→slot differential (weight-jitter 0.3 on 40) is too small vs the tonic bias to make the FIRST-mention winner entity-specific, so every novel barcode falls into the same systematically-favored slot. My numpy binder (RUNG 6c/6d) solved this with an EXPLICIT free-slot counter + a retrieve-vs-allocate THRESHOLD (novel → next free slot). On the substrate this must EMERGE: **a novel barcode (no strong match to any already-potentiated slot) must be detected as NOVEL and routed to a FRESH slot; a familiar barcode retrieves its bound slot.**

## ⇒ The mechanism (familiarity-gated allocation — connects to existing machinery)
This is a **familiarity / novelty-detection gate** — exactly the Bogacz-Brown familiarity mechanism the project already validated (the no-confab moat's familiarity gate) + hippocampal DG novelty/neurogenesis (the barcode mint is already DG pattern-separation). The on-substrate wiring: a familiarity read (max potentiated barcode→slot conductance) gates retrieve-vs-allocate; on NOVEL (low familiarity), a free-slot-selection mechanism (a disinhibited/unpotentiated slot wins) claims a fresh slot. The RETRIEVE half is done; the ALLOCATE half is the next build.

**NEXT CONCRETE ACTION:** a-1 our own record (Bogacz-Brown familiarity gate + DG novelty + any free-slot/winnerless-competition finding) → wire a familiarity-gated fresh-slot allocation onto the bridge binder (a novel barcode → a fresh slot; a familiar → retrieve), then the full held-out-novel + merge/no-bind de-risk (6-seed). The rate/numpy binder (RUNG 6c) + the Hebbian-vs-presynaptic mechanism (RUNG 6d, 6-seed GO) STAND; this characterizes the on-substrate realization (WTA + retrieve GO; allocation = the next rung). NO `sim/` edit.

Runner: `_stp_binder_onbridge_derisk.py` (`--smoke` WTA, `--derisk` bind/retrieve).
