# RUNG 6 — the FULL SPIKING composition: the deployed two-gate SPIKING D3 register feeds its resumed-protagonist who-state into a reservoir GENERATOR's read-out (6-seed GO)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung6_spiking_composition_derisk.py` (reuse-by-import: `SpikingPopGateRegister` + `make_discourse`/`_truth` + a fixed ESN reservoir; numpy; NO `sim/` edit).
**Status:** ✅ 6-seed GO (dev 42/43/44 + blind 100/101/102, all 6 GO).

## What this composes (the upgrade over the cheap-first Rung 6)

The cheap-first Rung 6 (`_reslm_rung6_pushpop_register_derisk`, 12-seed GO) used a MINIMAL structural two-gate register on a synthetic discourse. This upgrades to the DEPLOYED, VALIDATED spiking pieces:

| piece | what it contributes |
|---|---|
| `SpikingPopGateRegister` (D3) | a two-gate who-register whose HELD slot is a **persistent slow-NMDA attractor on a real `SimulationBridge`** — PUSH = clear-then-load, POP = read the attractor's spikes; `who_agent()` after a pop = the RESUMED earlier protagonist |
| `make_discourse` / `_truth` (D3) | REAL discourse statistics — a connective + named subject opens an event (boundary); a connective + a pronoun is a discourse POP (return to the prior protagonist) |
| a fixed ESN reservoir | the emergent generator's fading-memory recurrence (the reslm reservoir class) |

The D3 arc already validated that the spiking register RESUMES the earlier protagonist across a pop (RESUME_spiking vs pop-lesion). **Rung 6 tests the GENERATION value:** does that spiking resumed who-state, fed to a reservoir read-out, let the GENERATOR predict the resumed referent — the register carrying what the reservoir fades?

## Result — 6-seed GO (n_disc=120; chance = 1/6 = 0.167)

| seed | register | reservoir | shuffle | pop-lesion | GO |
|---|---|---|---|---|---|
| 42 | 0.78 | 0.25 | 0.22 | 0.42 | ✓ |
| 43 | 0.67 | 0.17 | 0.14 | 0.33 | ✓ |
| 44 | 0.69 | 0.14 | 0.11 | 0.42 | ✓ |
| 100 | 0.94 | 0.14 | 0.11 | 0.08 | ✓ |
| 101 | 0.58 | 0.08 | 0.14 | 0.08 | ✓ |
| 102 | 0.64 | 0.25 | 0.22 | 0.08 | ✓ |
| **mean** | **0.72** | **0.17** | **0.16** | **0.24** | **6/6** |

GO gate per seed: register > reservoir + 0.15 AND > shuffle + 0.15 AND > pop-lesion + 0.10. **6/6 GO** (dev 3/3 + blind 3/3).

The register-augmented read-out (mean 0.72) beats the faded reservoir (0.17 ≈ chance), a shuffled who-state (0.16), and — the load-bearing D3 control — the register's own **pop-lesion** (0.24, held slot not restored). So the generator's cross-pop prediction rides the SPIKING two-gate register's resumed who-state, produced on a real `SimulationBridge`, on real discourse statistics.

## Honest scope

- register mean 0.72 (not the cheap-first's 1.00) reflects the deployed register's ~0.9 spiking resume accuracy × the finite-sample read-out on real discourse statistics — the honest price of the real spiking substrate + real discourse vs the idealized cheap-first. Still decisive over every control on all 6 seeds (min margin over the strongest control ≥ 0.16).
- The register's per-clause spiking (SimulationBridge + slow-NMDA attractor reads) is the compute cost — the 6-seed is a real background job, not a wall.
- NO `sim/` edit. The reservoir-scale ceiling does not block this — Rung 6 is a DISCOURSE-structure rung (the register carries what the reservoir fades), not a scale rung.

## ⇒ the claim

The emergent generator's next-clause-subject prediction across a discourse POP is improved by a **two-gate who-register realized ON SPIKES on a real `SimulationBridge`** (a persistent slow-NMDA held slot), on REAL discourse statistics — the cheap-first Rung-6 result upgraded to the deployed spiking pieces, wiring validated end-to-end.

## Files

- `research/runners/_reslm_rung6_spiking_composition_derisk.py` (`--smoke` / full); raw `_rung6spk_{dev,blind}.json`.
- Builds on the cheap-first Rung 6 (`2026-07-13-RUNG6-pushpop-register-...`) + the D3 spiking pop-gate register (`_d3_event_popgate_spiking_agent_derisk` / `2026-07-10-D3-*`) + the reslm reservoir class.
