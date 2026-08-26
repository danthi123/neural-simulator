# RUNG B-1b — the role SELECTION is now an ON-BRIDGE spiking WTA (the host `argmax` is removed) — **GO**

**Date:** 2026-07-04
**Runner:** `research/runners/_rungB1b_neural_role_wta_derisk.py`
**Test:** `tests/test_rungB1b_neural_role_wta.py`
**Raw:** `research/findings/raw/_rungB1b_neural_role_wta.json`

## Why (closing the host-`argmax` shortcut — the owner's "one shared substrate, no cheats" directive)

RUNG B-1 made the comprehension→composition HAND-OFF synaptic (the reservoir's roles drive the composer's bind through
the `role_route` gates), but the role SELECTION was still a host `argmax(f @ Ws[k])` — a Python argmax picked the role,
then the corresponding parser conjunction was fired to open its gate. RUNG B-1b removes that host argmax: the reservoir's
per-word role LOGITS drive an on-bridge spiking mutual-inhibition WTA, and the WTA WINNER's firing opens its `role_route`
gate directly. The whole comprehend→select→bind→recall turn runs on ONE `UnifiedBrainBridge`. Nothing in role SELECTION
is host-decided (the host `f @ Ws` read-out matmul remains — that is the last shortcut, closed in B-1c).

## The mechanism (reservoir logits → on-bridge WTA → gate, no host argmax)

3 excitatory role ensembles (P=20 each) + one shared inhibitory pool (INH=30) live on the bridge's `role_wta` slice
(the additive `role_wta_n` allocation; `num_traits=2`, all traits forced excitatory, the inh pool flipped to trait 1 via
`cp_traits[inh]=1`). Wired runner-side, IN-PLACE (`set_pathway_weights(add_missing=True)` — preserves the trained parser,
no re-injection): E→I feedforward (`W_EI=24`), E→E self-excitation (`W_EE=18`), I→E feedback (`W_IE=20`) — genuine biased
competition (Desimone-Duncan / Wang-2002). Each ensemble is coupled to its gate: `couple_gate_to_indices(role_route_<r>,
ens[r], threshold=0.005)`.

Per content word: `logits3 = (f @ Ws[k])[[AGENT,PREDICATE,THEME]]` → `_wta_drive` = `WTA_BASE=150` (uniform, to every
ensemble) `+ WTA_GAIN=120 · normalized_rectified(logits3)` (**no argmax**) → the ensembles compete; the graded bias +
the shared inhibition silence the losers → the WINNER fires → its `role_route` gate opens. Then, EXACTLY as
`_op_synaptic`: SETTLE (40 steps) → prewarm-watch until the FIRST gate opens (the winner) → LATCH by pausing the
couplings so the gates RETAIN the value the competition produced (winner open, losers closed — **never hand-forced**) →
run the composer coincidence readout through the held gate. The latched role is read from the GATE the WTA opened, never
from a host argmax over the logits (`_op_wta` never even references `Ws`).

## The de-risk — **GO** (3 seeds; reuse EMERGE-78/88 reservoir + I5a route + the on-bridge WTA; CPU/numpy; NO `sim/` edit)

Nine anti-cheats (RUNG B-1's six + three WTA-specific), all pass on every seed (42/43/44):

| gate | 3-seed | bar |
|---|---|---|
| **route recall (WTA-selected, synaptic)** | **12/12 all** (mean 1.000) | ≥ 0.80·n |
| **route not worse than the host-argmax dict path** (same WTA-wired substrate) | **True** all (12 == 12) | — |
| **no-confab MOAT** | **0.00** FA all | ≤ 0.05 |
| **provenance-clean** (composer role bank gets ZERO direct current) | **True** all | — |
| **route-lesion collapses** (cut the synaptic route → no gate → starved) | **True** all (0 < 12) | — |
| **reservoir-lesion collapses** (collapse the reservoir → wrong roles) | **True** all | — |
| **provenance-NEURAL-SELECT** (source: `_op_wta` never sees `Ws`, no host argmax picks the gate; runtime: latched role == argmax over ens FIRING) | **True** all | — |
| **WTA-lesion collapses** (zero I→E → competition collapses → multiple gates → superimposed roles) | **True** all (0 < 12; seed 43 4 < 12) | — |
| **Ws-scramble collapses** (permute Ws' 3 role columns → logits misroute) | **True** all (0 < 12) | — |

**The result:** the reservoir's role logits drive an on-bridge spiking WTA whose winner's firing opens the composer's
route — the host `argmax` is gone, and the selection is a genuine neural competition (the route-lesion AND the WTA-lesion
both collapse recall, proving the gate-opening AND the mutual inhibition are load-bearing; the source-check + the
latched==firing-winner runtime check prove no host argmax decides the role). ~87 s/seed, CPU/numpy.

## Honest scope + tuning (self-caught defects, no faking)

- **Drive retuned to a uniform baseline (`WTA_BASE=150` + `WTA_GAIN=120`).** A max-normalized drive alone lets the winner
  win by feedforward, which would make the WTA-lesion NOT load-bearing (the inhibition wouldn't be what silences losers).
  The uniform baseline drives all three ensembles toward firing so the I→E inhibition IS what suppresses the losers —
  intact = 1 gate opens; I→E-lesioned = 2–3 gates open → superimposed roles → collapse. This makes the WTA-lesion genuine.
- **The latch never force-opens a gate.** An earlier draft hard-set the winner's gate to 1.0, which bypassed the
  route-lesion (gate forced open despite the cut). Faithfully mirroring `_op_synaptic` (pause couplings, hold whatever the
  competition produced) makes BOTH the route-lesion and the WTA-lesion collapse correctly.
- **`PROJ_DIM` 128 → 192.** At pd=128 the WTA route missed 1–2 borderline facts on seeds 43/44 — but verified an OU
  codebook-margin artifact, NOT a selection deficiency (the WTA latches the correct role on 0/6 mismatches vs host argmax
  every seed; the SAME miss appears on the host-argmax dict path on the SAME substrate, and it goes both ways — the WTA
  beats dict on seed 42). Widening the codebook (the I5a 64→128 lever) removes the margin so both hit 12/12. The dict
  baseline runs on the SAME WTA-wired substrate, isolating role selection.
- **WTA-lesion collapse** is full (0 < 12) on seeds 42/44, partial (4 < 12) on seed 43 — still collapses.
- Reuse-by-import (EMERGE-78/88 reservoir; I5a route + instruments; the committed `role_wta_n` allocation); NO `sim/` edit.

## The remaining shortcut → B-1c (next)

The host `f @ Ws[k]` read-out matmul is the last host computation in role selection. **B-1c** makes `Ws` real
reservoir→ensemble SYNAPSES with the reservoir (`OnBridgeLSM`) co-resident on the same bridge, so even the read-out is
on-substrate — then NOTHING load-bearing in the comprehend→select→bind turn is host-computed.

## Files
- `research/runners/_rungB1b_neural_role_wta_derisk.py` — the on-bridge WTA role-selection + the nine anti-cheats.
- `tests/test_rungB1b_neural_role_wta.py` — 6 fast structural tests + a slow seed-42 GO gate.
- `research/findings/raw/_rungB1b_neural_role_wta.json` — the 3-seed GO.
