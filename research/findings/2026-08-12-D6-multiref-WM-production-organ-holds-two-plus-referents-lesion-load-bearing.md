---
type: finding
status: go
date: 2026-08-12
mechanism: D6 MULTI-REFERENT WORKING MEMORY production organ — a co-resident spiking buffer (R banks of the D3 slow-NMDA bistable HOLD on ONE bridge, ONE shared FS) that latches >=2 discourse referents by register and SUSTAINS each across an intervening span; surfaces an honest hold read-out ("I'm holding dog and cat at once") a single-attractor store cannot
lane: working memory / conversation (D6 — hold >=2 discourse referents; wire the 2026-08-11 multi-slot GO to production)
verdict: WIREABLE — the organ reuses-by-import the 6-seed GO `_multi_slot_binding_derisk.MultiSlotHold` (no reimplementation, NO sim/ edit). Standalone verify (6 seeds 42/43/44/100/101/102, numpy-CPU): INTACT holds+recovers k=2,3,4 referents all_recovered=1.000 with hold-alive>0 and external input ASSERTED zero; LESION (recur=0) collapses the >=2 read-back to all_recovered=0.000 (bumps dead, 0.0000) -> load-bearing (Delta>=0.90 every k); the SUPERPOSED single-register control ties (mean recovered <=1.0, the ~2-cap the single-attractor anaphora store hits); FLAG-OFF (BRAIN_MULTIREF=0) + out-of-scope inputs (fewer than 2 referents / no hold-query) return None -> byte-identical; MOAT preserved (output is a read-out of the organ's OWN buffer, never a manufactured fact / flipped abstain; no invented referent). Residual (unchanged, per task + de-risk): the learned SPIKING WRITE-GATE is the open rung (register assignment is a role-by-position host MARKER; the referent PARSE + argmax READ are host).
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_d6_multiref_wm_production_verify.py
organ: research/runners/d6_multiref_wm_production_organ.py
artifacts:
  - research/findings/raw/_d6_multiref_wm/verify.json
  - research/findings/raw/_multi_slot_binding/multi_slot_6seed.json
depends_on: 2026-08-11-multi-slot-variable-binding-working-memory-holds-k-bindings-no-crosstalk-ceiling-k5-6seed-GO.md
---

# D6 multi-referent WM: a co-resident spiking buffer holds >=2 discourse referents across a span (lesion-load-bearing), wireable into production as an honest hold read-out — the single-attractor anaphora store ties on 2+

## The faculty and the verify-first verdict

Conversation needs BOTH referents live at once — "the dog and the cat ... it chased her". The prior single-attractor
anaphora store TIES on 2+: one bump wins the 1-of-K WTA (the superposition ~2-cap). The 2026-08-11 6-seed GO
`2026-08-11-multi-slot-variable-binding-working-memory-holds-k-bindings-no-crosstalk-ceiling-k5-6seed-GO.md` already
built + validated the fix: R disjoint slow-NMDA bistable banks on ONE `SimulationBridge` sharing ONE FS inhibitory
pool, each register latching one bind and SUSTAINING it (lesion-the-hold recur=0 collapses k>=2 to 0.000; superposed
single-slot collides at ~1/k; k=2 held-out ALL-correct 1.000, ceiling k=5). VERIFY-FIRST this session: I re-read the
runner's OWN verdict (not the prose), confirmed the 6-seed artifact (k=2 all=1.000, lesion=0.000, superposed=0.0,
filler-swap=0.000, zero-input asserted), and REPRODUCED it numpy-CPU (1-seed k=1,2,3: all=1.000, lesion 0.0->1.0
[MOVED], superposed 0.0->1.0 [MOVED]). The spiking HOLD is a genuine `cp_firing_states` read and is lesion-load-bearing
— NOT a host-formula, mis-read, or seed-fragile shortcut. WIREABLE.

## The organ (reuse-by-import; NO sim/ edit)

`research/runners/d6_multiref_wm_production_organ.py` — a process-shared `MultiReferentWMOrgan` that imports
`MultiSlotHold` (the spiking core) + the RUNG6c `HebbianBinder`/`_mint_codes` and adds ONLY the production shell:
a host referent parse (a small referent lexicon + capitalized proper names — the declared vocab-ceiling residual),
a role-by-position write MARKER (referent r -> register r — the declared open rung), and an honest hold read-out.
It MAINTAINS the session's live referents in the spiking buffer and, on an explicit "who/what are we talking about /
what are you keeping in mind" query, READS every held referent BACK off the sustained bumps and answers
"I'm holding 2 referents in working memory at once: dog and cat" — the read-back is what a single-attractor store
cannot produce. Default-ON; `BRAIN_MULTIREF=0` -> byte-identical oracle; `BRAIN_MULTIREF_LESION=1` -> recur=0.

## Verify numbers (6 seeds, numpy-CPU — `research/findings/raw/_d6_multiref_wm/verify.json`)

<!--derived-->
| test | k=2 | k=3 | k=4 |
|---|---|---|---|
| INTACT all_recovered | 1.000 | 1.000 | 1.000 |
| INTACT hold_alive_min | 0.0634 | 0.0639 | 0.0588 |
| LESION (recur=0) all_recovered | 0.000 | 0.000 | 0.000 |
| LESION hold_alive_min | 0.0000 | 0.0000 | 0.0000 |

SUPERPOSED single-register recovered (k=2) = 1.000 (a single attractor ties; the multi-register buffer holds 2).
FLAG-OFF + scope byte-identical = True (out-of-scope: 0 referents / 1 referent / pronoun-only -> None). MOAT ok =
True (two held+recovered, no invented referent, hold-query reads both back, output is a read-out not a fact). All
six load-bearing gates PASS.

## Honest residuals (declared)

The learned, emergent, SPIKING multi-register WRITE-GATE is the open rung: register assignment is a role-by-position
host MARKER (`739a8867`: even a host position-ORACLE fails to induce role at 6 seeds -> the residual is CREDIT
ASSIGNMENT, gap#4). The referent EXTRACTION is a host parse bounded by the referent lexicon (the same vocab-ceiling
class the comprehension organ declares); the BIND is the host-numpy RUNG6c binder (capped at _K=6 distinct
referents, ceiling k=5); the register READ is a host argmax over the bank's firing rates (a read-out instrument, as
in the affect/comprehension/metacog organs). CO-RESIDENT: the buffer runs on its own `MultiSlotHold` bridge alongside
the recall composer, riding the one-brain merge (burn-down #1). The LOAD-BEARING spiking contribution — the multi-item
bump-attractor HOLD that carries >=2 referents where a single attractor ties — is real, spiking, and lesion-load-bearing.
