# Reconsolidation (prediction-error-gated in-place fact update) — cheap-first de-risk = 6/6 GO

**Date:** 2026-06-17 (CYCLE 146 foreground track — the next *medium* conversational roadmap item)
**Status:** **GO, unanimous 6/6** (seeds 42, 43, 44, 100, 101, 102). A prediction-error-gated, in-place fact
update on the composer memory recovers a corrected fact, is cleanly distinguishable from naive-append /
overwrite-always / do-nothing, with the prediction-error boundary condition real and the no-confab moat intact.
**Scope:** cheap-first numpy/CPU, reuse-by-import, **NO `sim/` edit**. Gate before the production build (Option A).
**Runner:** `research/runners/_phaseB_reconsolidation_update_derisk.py`
**Scope doc:** `research/findings/2026-06-17-reconsolidation-conversational-memory-scoping.md`
**Raw:** `research/findings/raw/_phaseB_reconsolidation_update.json`

## The capability + the gap it fills

The production conversational memory (`RFPhasorComposer`) is **append-only**: `store()` pushes each fact onto
`self.kb`, and every `query_*` returns the **first** matching fact. Tell the agent "the dog went north", then
correct it ("actually, the dog went south"), and today you get **two contradictory facts coexisting**, with the
stale one answered first. **Reconsolidation** is the biology that fixes exactly this (Nader-Schafe-LeDoux 2000;
Osan-Tort-Amaral 2011 mismatch-gated attractor update; Sevenster 2013 prediction-error necessity): a reactivated
memory becomes labile and is **updated in place** — but **only when retrieval carries a prediction error**; a
fully-predicted re-statement re-stabilizes unchanged. That prediction-error gate is the load-bearing distinction:
without it "reconsolidation" is just `dict[key] = value` (last-write-wins), which is neither biological nor a
capability. The conversational behavior unlocked: **the agent can be corrected through dialogue and updates the
fact in place** — a persistent-memory behavior a stateless basic LLM has no analogue for.

## Mechanism (Option A, realized on the existing composer)

`update_on_mismatch(agent, action, new_patient)` on `ReconsolidatingComposer(RFPhasorComposer)`:
1. **Reactivate** — find the fact whose cue roles (agent + action) match, by the substrate unbind + cleanup
   (`self.unbind(comp, "agent"/"action")`). **No matching trace → abstain** (no fact created — the moat).
2. **Prediction error** — `PE = 1 − phase-cos(recovered patient phasor, asserted new code)`, reusing
   `_unbind_phases(comp, "patient")` and `self.concepts[new_patient]`. ~0 when the new filler matches the stored
   one (a re-statement); ~1 when it mismatches (a real correction).
3. **Gate** — if `PE ≥ PE_LABILE` → **rewrite in place** (re-`_encode` the corrected fact, replace the `kb`
   entry — no append). If `PE < PE_LABILE` (a re-statement) → **re-stabilize unchanged** (no write).

`PE_LABILE` is **frozen** at the measured same-vs-different PE midpoint (the `calibrate_threshold` rule —
the data's own separation point, **not** tuned to a downstream probe). The bind / unbind / bundle that compute
the composite and the PE all run on the **real resonate-and-fire spiking substrate** (`_resonate`); only the
`kb` store and the cleanup use the validated numpy fast path.

## Result — 6/6 unanimous, all anti-cheats hold

| check (per seed) | result |
|---|---|
| **RECONSOLIDATE corrects** (query→`south`, exactly **one** `dog go *` fact, untouched `cat run`→`south`) | **6/6** |
| **C1 — prediction-error boundary** (re-stating the SAME fact, PE≈0 → RECONSOLIDATE does **not** write; OVERWRITE-ALWAYS writes anyway = the last-write-wins tell) | **6/6** |
| **C2 — moat** (correcting a NEVER-stored subject → **abstain**, no fact fabricated) | **6/6** |
| **C3 — lesion + permuted** (force PE=0 → update collapses to no-write; wrong-cue correction → abstain, target intact) | **6/6** |
| **four arms cleanly separated** (RECONSOLIDATE passes; NAIVE-APPEND → 2 facts, answers stale `north`; OVERWRITE-ALWAYS fails C1; DO-NOTHING → `north`) | **6/6** |
| baseline recovery (the substrate recovers all 4 stored facts before any correction) | 6/6 |

**Prediction-error separation (6-seed mean):** re-statement PE **0.480** vs correction PE **1.003**, gate at the
measured midpoint **0.741**. The separation is large and clean every seed (gap ≈ 0.52; every re-statement PE
0.41–0.49 sits below the gate, every correction PE 0.96–1.05 above it), so the boundary condition is robust — the
PE is not a clean 0/1 (bundle cross-talk lifts the same-fact PE off 0), but the **same-vs-different separation is
decisive**, which is all the gate needs.

## Reading it

- The headline conversational capability — **correct the agent and it updates the fact in place** — works,
  multi-seed, with **exactly one** fact retained (no contradictory duplicate), and the untouched facts preserved.
- The scientifically load-bearing part — that this is **prediction-error-gated reconsolidation, not
  last-write-wins** — is proven by C1: a re-statement (PE below the gate) leaves the memory unchanged, while the
  OVERWRITE-ALWAYS ablation writes even there. The boundary condition is real.
- The no-confab moat is **respected, not weakened** (owner 2026-06-17: a plus, not a hard gate): updating a
  *reactivated, previously-stored* trace is the feature; *fabricating a never-stored* fact still abstains (C2).
  This is exactly the update-an-existing-trace vs invent-from-nothing distinction.

## Honest scope

Reconsolidation here is realized at the composer's **KB layer** — the project's documented
"composer-as-idealization" host-held fact store (`self.kb`, a list of `(fact, composite_phasor)`). The
reactivation, the PE measurement, and the re-bind all run on the real RF spiking substrate, but the *store* being
rewritten is the idealized layer, not yet the synapse. The **synaptic-literal tier** (Option B — engram-tag
reactivation + a prediction-error-driven `plasticity_window_gate` + `forget`/`consolidate` re-store) is the named
follow-on this GO de-risks, with schema-assimilation (Option C, V_SCHEMA/Tse-2007) as its low-PE partner. This is
the project's standing scaffold→spiking conversion pattern: prove the gate logic cheaply at the composer layer
first, then realize it at the synapse as a separately-validated step.

## Next (the production build, GO-licensed)

Build Option A as a small **default-off additive** `RFPhasorComposer.update_on_mismatch(...)` (the append-only
path bit-preserved when off) + a **correction-turn hook** in `MultiTurnAgentV2` (detect "actually / no, …" or a
contradicting re-assertion → route to the composer update), with the existing conversational test suites
asserting the no-confab moat and the four-arm distinguishability. Reuse-by-import, no `sim/` edit.

## Reproduce
```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_reconsolidation_update_derisk \
    --seeds 42,43,44,100,101,102
```
