# RUNG 6 (cheap-first, 12-seed GO) — the emergent generator predicts the RESUMED protagonist across a discourse POP via a TWO-GATE push/pop who-register, surpassing the Rung-2 single-latch ceiling

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung6_pushpop_register_derisk.py` (self-contained cheap-first: minimal ESN reservoir + a minimal two-gate push/pop who-register + a synthetic push/pop discourse; numpy; NO `sim/` edit).
**Status:** ✅ CHEAP-FIRST GO (12-seed: standard 42/43/44/100/101/102 6/6 + FRESH 7/8/9/10/11/12 6/6).

## The distinctive new capability over Rung 2

The emergent-generation ladder's Rung 2 validated that a **single non-fading latch** carries a distal discourse referent the fading reservoir forgets. But a single latch holds the MOST-RECENT referent — so across a discourse **POP** (Grosz-Sidner attentional stack: a return marker + a pronoun resumes an EARLIER protagonist), a single latch holds the wrong entity (the interloper). Rung 6's genuinely-new piece is the **two-gate** register (the D3 push/pop event-register semantics): a PUSH gate stores the current protagonist when a new named subject enters; a POP gate restores it when the discourse returns. This probes whether adding that two-gate who-state as a read-out feature lets the reservoir generator predict the RESUMED protagonist, where the reservoir alone (faded) and a single latch (holds the interloper) both fail.

## The task (per trial)

INTRODUCE A → PUSH (a new named subject B enters) → B holds the floor for PUSH_K=8 clauses (A goes DISTAL) → POP (a return marker + a **PRONOUN** surface token — the reservoir is driven by the pronoun, so it never directly encodes A). Predict the referent of the pronoun = the resumed protagonist A. The pop clause's surface being a pronoun (not A's name) is the load-bearing design: only the register's pop gate restores A; the reservoir sees only an ambiguous pronoun over a faded-distal A.

## Result (12-seed; chance = 1/N_ENT = 0.167)

| arm | post-pop referent accuracy (representative) |
|---|---|
| **REGISTER** (reservoir ⊕ two-gate push/pop who-state) | **1.00** |
| LATCH-ONLY (reservoir ⊕ single most-recent-subject latch) — the Rung-2 ceiling | **0.17** (holds the interloper B ≈ chance) |
| RESERVOIR-ONLY (fading) | 0.12–0.18 (≈ chance — A is distal, faded) |
| SHUFFLE-REGISTER (anti-cheat: who-state permuted across trials) | 0.14–0.26 (collapses) |

GO gate: register > latch + 0.15 AND > reservoir + 0.15 AND > shuffle + 0.15. **12/12 GO.**

## Depth-robustness (the register carries the referent UNBOUNDED)

Sweeping the distal depth (`--push-k`, how many clauses the interloper holds before the pop) confirms
the register's advantage is not a shallow-depth artifact — it holds as A goes arbitrarily distal while
the reservoir fades completely:

| PUSH_K (A is k+1 clauses distal) | register | latch | reservoir |
|---|---|---|---|
| 4 (A 5 back) | 1.00 | 0.28 | 0.20 |
| 8 (A 9 back) | 1.00 | 0.20 | 0.15 |
| 16 (A 17 back) | 1.00 | 0.19 | 0.16 |
| 32 (A 33 back) | 1.00 | 0.19 | 0.14 |

The register (non-fading two-gate slot) stays at 1.00 to 33 clauses back; the reservoir fades to chance
by ~5 clauses. This is the fading-memory ceiling (the reservoir's bound) surpassed by the structural
register — exactly the Rung-6 discourse-structure role.

## Why this is the right controls

- **LATCH-ONLY being WRONG post-pop is the point** — it is exactly the Rung-2 single-latch ceiling (it holds B, the most-recent named subject). The two-gate register surpasses it by restoring A.
- **RESERVOIR-ONLY at chance** confirms A is genuinely distal (the fading reservoir has lost it by the pop) — the register is not decorative.
- **SHUFFLE-REGISTER collapsing** confirms the two-gate who-state (not a generic extra feature) carries the answer.

## Honest scope + next

- **Cheap-first + self-contained:** a minimal ESN reservoir + a minimal structural two-gate register on a synthetic push/pop discourse — the CORE hypothesis (the two-gate register surpasses the single latch for generation across a pop) before wiring the FULL validated pieces. The register's gates are structural (driven by the discourse markers) — the same two-gate semantics the D3 event-register arc realized ON SPIKES and learned self-supervised (`2026-07-10-D3-event-*`).
- **Next (the full composition):** wire the validated spiking D3 `SelfSupEventRegister` (push/pop on a real `SimulationBridge`, self-supervised δ) + the reslm reservoir generator (`_emerge_reservoir_lm_derisk`) so the who-state feature is produced ON SPIKES by the deployed register, and run the by-referent-accuracy metric on a real multi-clause discourse stream. The reservoir-scale ceiling does NOT block this — Rung 6 is a DISCOURSE-structure rung (the register carries what the reservoir fades), not a scale rung.
- NO `sim/` edit. Reuse-by-import path identified.

## Files

- `research/runners/_reslm_rung6_pushpop_register_derisk.py`; raw `research/findings/raw/_rung6_{standard,fresh}.json`.
- Builds on the emergent-generation ladder (Rungs 1–5) + the D3 two-gate event register (`2026-07-10-D3-*`).
