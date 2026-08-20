---
type: finding
status: live
date: 2026-08-20
mechanism: gnw-workspace
lane: integration
integration_faculty: gnw-two-organ-bus
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: The default-on FLIP is a production-integration verification (end-to-end correctness on both backends + zero-regression off-vs-on), not a stochastic effect size — the underlying combine mechanism is the 6-seed GO already banked ([[2026-08-20-gnw-workspace-integrates-two-genuinely-distinct-organs-6seed-GO]]); this finding records that it cleared the promotion gate and is DEFAULT-ON (the substrate authors the covered-recall combination verdict by default; the host gate() extraction + side-effects still run, so the scaffold is not yet retired).
instrument: research/runners/_gnw_two_organ_production_verify.py — the REAL ChatBrain.gate handler, cupy AND numpy, agree→commit / disagree+lesion→abstain / byte-identity-off / safe-inert fallback
runner: research/runners/_gnw_two_organ_production_verify.py
external: NO-EXTERNAL-NEEDED — a production-integration flip of an in-repo de-risked mechanism; the workspace-ignition-combines-convergent-evidence principle is grounded in Dehaene & Changeux 2011 (Sources), already the basis of the 6-seed GO.
artifacts:
  - research/findings/raw/_gnw_two_distinct_organs/summary.json
---
# The GNW two-genuinely-distinct-organs bus is now PRODUCTION DEFAULT-ON, verified load-bearing on both backends

Artifact: research/findings/raw/_gnw_two_distinct_organs/summary.json

**One line.** The spiking Global-Neuronal-Workspace bus that COMBINES two genuinely-distinct organ reads by ignition
(composer recall + the production `SurpriseProductionOrgan`) — a 6-seed GO
([[2026-08-20-gnw-workspace-integrates-two-genuinely-distinct-organs-6seed-GO]]) — is now DEFAULT-ON on the live chat
path, on BOTH backends (the substrate authors the covered-recall combination verdict by default; the host gate()
extraction + side-effects still run, so the scaffold is not yet retired). This is the mission's integration-to-production-default spine: a genuinely-distinct
second spiking organ is now LOAD-BEARING on the conversation by default, not a default-off de-risk beside a host pipeline.

## What changed
<!--derived-->
The host per-turn combine for a covered routable recall is authored by the COINCIDENCE of organ A (composer
`query_patient`) + organ B (the surprise monitor corroborating the candidate against its OWN expectation via a real
`cp_firing_states[surprise]` read): both agree → ignite → commit; organ B contradicts or is lesioned → consensus-veto →
abstain. The master switch (`BRAIN_GNW_2ORGAN`, read in `webapp/server.py` + `webapp/gnw_two_organ_bus.py`) flips
default `""`→`"on"`: unset = ON; `=0`/`off` still disables (escape preserved, byte-identical to the pre-bus path).

## Why it is safe on cupy (the production backend) — the blocker that was fixed first
The surprise organ was inert on cupy because per-neuron thresholds were drawn from the active backend's RNG
([[2026-08-20-backend-dependent-RNG-thresholds-broke-the-surprise-organ-on-cupy-backend-neutral-init-fix]]). With
`backend_neutral_izh_initialization` default-on in `build_expectation_circuit`, the production organ discriminates on
cupy too. The bus's own gate (`_organ_discriminates`) requires that fix and otherwise falls back to a SAFE INERT
(byte-identical) path — it never runs a mis-discriminating organ. (`backend_neutral_izh_arithmetic` was measured FULLY
inert for this ~390-neuron organ: byte-identical discrimination + 0% latency — so only the load-bearing INIT is required.)

## The promotion gate (all green, both backends)
<!--derived-->
- **Production wired bus end-to-end** (real `ChatBrain.gate`): **cupy GO** — 5/5 stored recalls commit the host patient,
  organ-B lesion + workspace lesion both collapse to abstain, moat abstains 3/3, wired discrimination
  **0.434 ≪ thr 1.401 ≪ 2.488 Hz**; **numpy GO** — same, discrimination 0.434 ≪ 1.371 ≪ 2.488 Hz.
- **Real server handler** (fresh `SIM_BACKEND=cupy` subprocess → `brain_chat` + the server hook): HTTP 200 with the bus
  OFF and ON — no cupy crash on the live handler.
- **Default-off byte-identity** re-confirmed True on both backends (HEAD == flag-on-clean == runtime-flip-off, byte-for-
  byte over the 8-turn panel) — the module is not even imported when disabled.
- **No-regression** (numpy chat/phasor/brain suite incl. the conversation aggregator): **263 passed / 57 skipped / 2
  failed, IDENTICAL off vs forced-on** — zero new failures. The 2 failures are PRE-EXISTING + environmental (a missing
  `denoise64_seed42.npz` concept cache in an unrelated subsystem; they fail identically off and on and should be fixed
  to skip-not-fail separately — they do NOT gate this flip).

## Honest scope + residuals
The bus authors the covered ROUTABLE-recall combine (extraction/side-effects run once, mirroring `gate_via_bus`); it
does not replace every host step. The one-brain MERGE path (`BRAIN_ONEBRAIN_MERGE=1`, default-off) builds from its own
config-superset and would need the two `backend_neutral_izh_*` lines to inherit the cupy fix if it ever goes
cupy-production — flagged, not a default blocker today. (Agent-wired + agent-verified; parent confirmed the cupy
`go=True` + the discrimination + byte-identity from the artifacts, then flipped the default.) The raw production-verify
JSONs are now BANKED, per the deferral above: `research/findings/raw/_gnw_two_organ/production_verify_cupy.json` and
`research/findings/raw/_gnw_two_organ/production_verify_numpy.json`, produced by the hardened
`research/runners/_gnw_two_organ_production_verify.py` (a top-level `tools.verdict.Verdict` — 8/8 preconditions PASS on
each backend, none unmeasured, none failed — plus an `attributable_to` call attributing the stored-fact abstain/commit
behaviour to each lesion lever: 100% attributable to the organ-B lesion and 100% to the workspace lesion vs the wired
baseline, on both backends). Both re-verify **GO**, matching the numbers above exactly — cupy discrimination
0.434 ≪ thr 1.401 ≪ 2.488 Hz, numpy 0.434 ≪ thr 1.371 ≪ 2.488 Hz. Hardening the runner surfaced and fixed one real bug
in the verify machinery itself (not the bus): the runner tested "flag OFF" by unsetting `BRAIN_GNW_2ORGAN`, correct
back when the flag was default-off, but `two_organ_enabled()`'s default-on flip (this same finding) made an unset var
mean ON — so the old runner's "HEAD" baseline was silently the two-organ bus comparing itself to itself. The runner now
forces `BRAIN_GNW_2ORGAN` to an explicit `"0"`/`"1"` on every phase instead of relying on unset; the bus mechanism
itself was never in question (every substantive teeth check — (A)/(B.i)/(B.ii)/(C)/(D2) — already read True before the
fix, only the OFF-path self-comparison was vacuous). The flip itself rests on the committed 6-seed GO + this verified
promotion gate.

## Sources
- Dehaene, S. & Changeux, J.-P. (2011). "Experimental and theoretical approaches to conscious processing." *Neuron*
  70(2):200–227 — the Global-Neuronal-Workspace ignition model: a broadcast fires only on CONVERGENT drive, the exact
  AND-over-organs the two-organ coincidence bus realizes (recall + surprise must agree to ignite). Grounds the mechanism
  this finding flips to production default.
