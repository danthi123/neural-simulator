# RUNG B-1 — the reservoir's LEARNED roles drive the composer's bind SYNAPTICALLY (the comprehension→composition hand-off) — **GO**

**Date:** 2026-07-04
**Runner:** `research/runners/_rungB1_reservoir_synaptic_handoff_derisk.py`
**Test:** `tests/test_rungB1_reservoir_synaptic_handoff.py`
**Raw:** `research/findings/raw/_rungB1_reservoir_synaptic_handoff.json`

## Why (the FUNCTIONAL bar — the first synaptic hand-off)

The EMERGE-92..95 ladder consolidated the conversational turn onto ONE spiking bridge (the SUBSTRATE bar), but the
EMERGE-95 finding was explicit that the hand-offs were still **host-dict** and called RUNG B (synaptic interaction) "a
genuine multi-week arc, correctly deferred." The RUNG-B research gate re-scoped that: the synaptic parser→composer route
already exists and is validated (`_burndown_I5a_synaptic_parser_composer`, all four anti-cheats GO — a parser-firing-gated
`role_route_<R>` topographic route carries role R's ±1 pattern into the composer role bank, provenance-clean). In I5a the
role per word comes from the HAND parser's positional rule. **RUNG B-1 makes the role come from the RESERVOIR's LEARNED
form→role map instead — so the comprehension→composition hand-off is now SYNAPTIC, not a host `{role:word}` dict.** This is
the first FUNCTIONAL-one-brain rung, cheap-first (reuse-by-import; NO `sim/` edit).

## The mechanism (reservoir role → conjunction → gate → composer, reusing the I5a route unchanged)

Because `_op_synaptic(k)` fires the role ensemble selected by conjunction index `k`, the reservoir's chosen role maps to the
conjunction that fires it (`role2k = {agent:0, action:2, patient:4}`, built at runtime from `parser.role_of`), and the
ENTIRE I5a route + all four anti-cheats are reused unchanged:

```
reservoir final state f --(Ws[k], the EMERGE-78/88 learned read-out)--> role r per content word
   --> fire conj[role2k[r]] --> gate role_route_<r> opens --> composer role bank gets role r's ±1 pattern
   --> the word binds with role r, provenance-clean (the role pattern crosses the region boundary ONLY via the
       gated synapses, never as a host {role:word} current).
```

The role SELECTION is the reservoir's (learned from function-word structure, content abstracted); the HAND-OFF is synaptic
(the gate). A comprehension missing a role (a collapsed/mislabeled parse, e.g. under the reservoir-lesion control) is not
storable → it counts as a recall miss rather than polluting the kb.

## The de-risk — **GO** (reuse EMERGE-78/88 reservoir + I5a route/instruments; CPU/numpy; NO `sim/` edit)

Six anti-cheats, multi-seed (ONE shared corpus/task; per-seed reservoir + bridge RNG — I5a's multi-seed pattern):

| gate | 6-seed (42/43/44/100/101/102) | bar |
|---|---|---|
| **parse** — reservoir maps each transitive to (agent, action, patient) | **1.00** all seeds | ≥ 0.90 |
| **route recall (SYNAPTIC)** — who/what recovered through the gates | **12/12 all seeds** (mean 1.000) | ≥ 0.80·n |
| **route not worse than the host-dict path** — robust to composer OU jitter | **True** all seeds (route 12 ≥ dict 10–12) | — |
| **no-confab MOAT** — unstored (agent, action) → abstain | **0.00** false-accept all seeds | ≤ 0.05 |
| **gated-by-firing** — only the reservoir-selected role's gate opens | **True** all seeds | — |
| **provenance-clean** — composer role bank gets ZERO direct external current | **True** all seeds | — |
| **route-lesion collapses** — cut the synaptic route → recall collapses | **True** all seeds (0 < 12) | — |
| **reservoir-lesion collapses** — collapse the reservoir → recall collapses | **True** all seeds (0 < 12) | — |

**The result (6/6 unanimous, all six anti-cheats):** the reservoir's learned role output drives the composer's bind
through synapses, recovering the facts (route recall 12/12 mean 1.000, always ≥ the host-dict path, which itself varies
10–12 with the composer's OU jitter), gated-by-firing, provenance-clean, with BOTH the synaptic route AND the reservoir
load-bearing (each lesion collapses recall to 0 on every seed). The comprehension→composition hand-off is no longer a
host dict — it is a parser-firing-gated synaptic route driven by the reservoir's comprehension. (~57 s/seed, CPU/numpy.)

## Honest scope (what is synaptic now, and what is not)

- **What became synaptic:** the reservoir's role output → the composer's role bank. The role ±1 pattern crosses the
  region boundary ONLY through the gated `role_route_<R>` synapses (provenance-clean); no host `{role:word}` quantity is
  copied across. This closes the *comprehension→composition* half of the "host-dict hand-off" gap EMERGE-95 named.
- **What is still host (RUNG B-1b, next):** the reservoir STATE `f` and its `argmax(f @ Ws[k])` read-out are still computed
  in host (the rate reservoir, EMERGE-88). The research gate's fuller target — "Ws as a fixed RegionPathway + a 3-way
  spiking WTA over role ensembles" — puts the reservoir ON the bridge (EMERGE-82 `OnBridgeLSM`) and makes the role
  SELECTION itself neural (a WTA over the Ws-projected role ensembles), removing the host argmax. That is the purity
  follow-on; this rung proves the hand-off SYNAPSE while the read-out stays host (matching EMERGE-88's rate reservoir).
- **The OTHER half of RUNG B (still host):** the composer's recalled answer → the producer's `decision` (production
  side). RUNG B-1 is the comprehension→composition hand-off only; the recall→production hand-off is a separate rung.
- Reuse-by-import (EMERGE-78/88 reservoir; I5a route + `lesion_route`/`provenance_role_bank_current` instruments); NO
  `sim/` edit; the reused modules byte-preserved.

## Files
- `research/runners/_rungB1_reservoir_synaptic_handoff_derisk.py` — the reservoir-driven synaptic store + the six anti-cheats.
- `tests/test_rungB1_reservoir_synaptic_handoff.py` — fast structural tests + a slow seed-42 GO gate.
- `research/findings/raw/_rungB1_reservoir_synaptic_handoff.json` — the multi-seed de-risk.
