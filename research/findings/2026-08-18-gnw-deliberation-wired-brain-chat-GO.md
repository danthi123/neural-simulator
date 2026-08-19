---
status: go
type: finding
lane: T1-1
date: 2026-08-18
integration_faculty: gnw-deliberation
---
# THE KEYSTONE, WIRED — a confidence/conflict-GATED deliberation on the LIVE `/api/brain-chat`: when the brain has COMPETING candidate answers, the WORKSPACE'S OWN spiking conflict read makes it ABSTAIN instead of committing the shaky first-match (deliberation-until-sure + halt-if-unsure) — GO

**Date:** 2026-08-18 · **Wiring:** [`webapp/gnw_deliberation.py`](../../webapp/gnw_deliberation.py) + the `install_deliberation_gate` call in [`webapp/server.py::brain_chat`](../../webapp/server.py) · **Verify runner:** [`research/runners/_gnw_deliberation_wired_verify.py`](../runners/_gnw_deliberation_wired_verify.py) · **Artifact:** `research/findings/raw/_gnw_deliberation_wired/verify.json` (+ `.prov.json`) · **Scope:** additive, DEFAULT-ON (`BRAIN_GNW_DELIBERATE`), reuse-by-import, `NO sim/ edit` (`git diff sim/` empty). FUNCTIONAL correlate only; NO phenomenal claim.

## Verdict: GO — the keystone's decisive piece (single-hop confidence/conflict-gated abstain) is WIRED into the live default turn; all four gates hold through the REAL ChatBrain + handler

The keystone de-risk ([`2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md`](2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md)) proved the substrate's OWN spiking read (`n_ignited` off `cp_firing_states` + the `nmda_norm` confidence balance off `cp_conductance_g_nmda`) can GATE deliberation. It was default-off, unreachable from production. This wires the DECISIVE piece — **halt-if-unsure** — into the LIVE `/api/brain-chat` turn: after the GNW ignition bus commits, the workspace's own conflict read DECIDES commit-vs-abstain. The "ACT on the conflict/confidence signals we only REPORT" audit item (T1-1 rung d) is now on the default turn.

## The genuinely-ambiguous case (what changes, scoped honestly)

In production the 3 organ reads only ever vote the forward recall `cand_A` or `None`, and `query_patient` is first-match — so the bus never sees a genuine multi-candidate conflict. The real "competing candidate answers" the mission names arise when **≥2 stored facts share the SAME (agent, action) with DIFFERENT patients** (e.g. `dog chase cat` [built-in] + `dog chase bird` [taught]). Today's bus commits the arbitrary FIRST-match (`cat`), silently discarding the competitor — a SHAKY commit: the brain is not actually SURE which single answer to give.

## Mechanism (reuse-by-import; the ONLY new code is production glue)

`install_deliberation_gate` wraps `chat.gate` AFTER `install_bus_gate`. On a bus-authored covered-class recall:
- **PROPOSE** (declared modular-processor boundary, read-only): `all_candidate_patients` enumerates the DISTINCT patients bound under (agent, action) off the live composer (the same `unbind` `query_patient` uses); `last_trace` is saved/restored so the surfaced activity is byte-identical.
- **EVALUATE** (the substrate): the distinct candidates are driven EQUALLY (`IGNITE_PA`) into the P1.2 GNW workspace slots (equally-valid stored answers); mutual-inhibition WTA + the ignition knee settle the competition; `conf` + `n_ignited` are read off the workspace spikes (the keystone `_ignite_and_read_nmda`/`_conf_from_nmda`).
- **ACC GATE** (the keystone `acc_conflict_gate`, reads ONLY spiking `conf`/`n_ignited` + its retry budget; theta self-calibrated via `calibrate_theta`, `theta_hi=0.500`, clean_gap): a single clean winner (`n_ignited==1`, `conf≥theta_hi`) → ADVANCE → the answer stands (byte-identical); ≥2 co-ignited (`n_ignited≥2`) or a single-but-low-conf read → RETRY (deterministic single-hop re-drive) then ABSTAIN → the brain says "I don't know" instead of committing the shaky first-match. <!--derived-->

## THE DECISIVE PIECE + THE NAMED DEFERRED REMAINDER

WIRED: the confidence/conflict-gated **halt-if-unsure** on the single production recall hop. NOT wired (the named deferred rung): the full MULTI-HOP **deliberation-until-sure** — the variable-depth transitive chase whose re-entrant CYCLE COUNT emerges across a CHAIN of inferences — stays the de-risk fixture (a single production recall is ONE hop; on one hop RETRY re-drives the SAME deterministic read, so a sustained conflict resolves to ABSTAIN after the budget). This is an honest scope, not an overclaim: the load-bearing spiking DECISION (commit-vs-abstain) moved to the substrate; the cross-inference cycle-count loop did not.

## GO GATE — through the REAL production ChatBrain + `/api/brain-chat` handler (numpy)

| gate | result | evidence |
|---|---|---|
| (A) ABSTAIN-ON-CONFLICT | **PASS** | pristine bus COMMITS `[dog,chase,cat]`; wired **ABSTAINS** (None) — `n_candidates=2`, workspace `conf=0.0`, `n_ignited=2`. Real handler: default `abstained=True` (`"I don't know about that."`) vs `BRAIN_GNW_DELIBERATE=0` `"The dog chases cat."` + `BRAIN_GNW_DELIBERATE_LESION=1` `"The dog chases cat."` |
| (B) BYTE-IDENTICAL reactive panel | **PASS** | in-process wired gate == pristine gate 12/12 turns (recall/abstain/inconsistent/self + learn + anaphora); real handler md5 default-ON == `BRAIN_GNW_DELIBERATE=0` (separate processes) = `334cc206a5a23981e0e2ae1724501255` |
| (C) LESION-LOAD-BEARING | **PASS** | `BRAIN_GNW_DELIBERATE_LESION=1` (workspace self-recurrence ZEROED) → the conflict cannot co-ignite (`n_ignited=0`) → COMMITS `[dog,chase,cat]` again; flag-off pass-through also commits (== pristine). Dissociation separation intact-abstain 1.0 vs lesion-abstain 0.0 |
| (D) MOAT-SAFE | **PASS** | a single clean answer (`cat eat fish`) is unchanged (pristine==wired); never un-abstains a bus abstain; never invents a fact; only ADDS abstentions on a genuine conflict |

**6-SEED robustness (the one seed-varying claim — does the substrate's ignition/conflict read still separate the cases across substrate seeds?):** on all 6 seeds **42/43/44/100/101/102** the workspace conflict read gives 2-equal-candidates → **ABSTAIN**, 1-candidate → **ADVANCE**, lesioned-2-candidates → **COMMIT** — **6/6**. Attribution of the abstain to the spiking workspace (intact-abstain vs lesion-abstain) = **1.0** (`attributable_to`). The keystone already showed `theta_hi` self-calibrates seed-invariantly; the byte-identical / moat integration facts are deterministic.

## Instrument (validated before wiring)

Intact workspace: 2 equal candidates → ABSTAIN (`conf=0.0`, `n_ignited=2`); 1 candidate → ADVANCE (`conf≈1.0`, `n_ignited=1`). Lesion workspace: 2 candidates → COMMIT (`n_ignited=0`, the co-ignition collapses). theta self-calibrated `theta_hi=0.500`, `clean_gap=True`. The de-risk mechanism underneath is the keystone's committed 6/6 GO; the WIRING is deterministic at the production workspace seed (42). <!--derived-->

## Honest residuals (declared, not faked)

- **Single-hop halt-if-unsure only** — the multi-hop deliberation-until-sure is the named deferred rung (above).
- **PROPOSE (candidate enumeration) is a host modular-processor boundary** — the same boundary the keystone / coincidence integrator declare; the workspace conflict read + the `acc_conflict_gate` carry the DECISION (the lesion proves it — disable the spiking workspace and the abstain is gone).
- **Abstain renders as the generic honest "I don't know"** — a conflict-specific honest hedge ("I have more than one answer") is a cheap named refinement, not built here.
- **Co-resident** on its own P1.2 workspace bridge (rides the one-brain merge burn-down #1).

## Files

Wiring: [`webapp/gnw_deliberation.py`](../../webapp/gnw_deliberation.py); the install call + `_GNW_DELIBERATE_DEFAULT_ON` anchor in [`webapp/server.py`](../../webapp/server.py). Verify: [`research/runners/_gnw_deliberation_wired_verify.py`](../runners/_gnw_deliberation_wired_verify.py). Artifact: `research/findings/raw/_gnw_deliberation_wired/verify.json` (+ `.prov.json`). Ledger row `gnw-deliberation` in [`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`](../../docs/PRODUCTION_INTEGRATION_LEDGER.yaml). Reproduce: `SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_deliberation_wired_verify`.

Cites: the keystone [`2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md`](2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md); the global-STOP companion [`2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md`](2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md); the bus default-flip finding `2026-08-13-gnw-bus-default-flip-substrate-authors-organ-combination`; the T1-1 faculty audit.
