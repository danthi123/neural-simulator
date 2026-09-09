---
type: finding
status: live
mechanism: gnw-congruence-spiking production-flip verify (rank-8 GNW organ-B/C host `==` congruence check)
lane: laneC
date: 2026-09-08
integration_faculty: gnw-two-organ-bus
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_gnw_congruence_spiking_production_flip_verify.json
  - research/findings/raw/_gnw_congruence_spiking_hook_verify.json
runner: research/runners/_gnw_congruence_spiking_production_flip_verify.py
---

# GNW congruence spiking read — PRODUCTION-FLIP verified GO on ARM 1+ARM 2 (6/6 seeds); `BRAIN_GNW_CONGRUENCE_SPIKING` now runs DEFAULT-ON; ARM 3 (cross-faculty battery) queued, not yet complete

**Verdict: GO on the two flip-specific load-bearing arms (6/6 seeds each); ARM 3 (cross-faculty regression
battery) launched but not completed this session — see its own section below for the bounded argument this
verdict does not depend on it.** The rank-8 de-risk
([`2026-09-05-gnw-congruence-spiking-read-rank8-derisk-GO.md`](2026-09-05-gnw-congruence-spiking-read-rank8-derisk-GO.md))
and its production-dispatch hook-verify (`_gnw_congruence_spiking_hook_verify.py`) were already 6/6 GO with the
flag wired but DEFAULT-OFF ("NOT flipped default-on (owner call)"). Per the 2026-08-28 owner directive banked in
`GAP_CLOSURE_MISSION.md` ("OWNER-GATED → CLAUDE-AUTONOMOUS... Flip the ready faculties"), this finding does the
flip-specific work: it flips `_congruence_spiking_enabled()`'s own default to ON in BOTH copies of the reader
(`webapp/gnw_bus_shadow.py`, the one production actually calls, and the sibling `webapp/gnw_congruence_spiking.py`,
kept in sync so the name is never misleading), then verifies three flip-specific claims the circuit's own GO gate
and the hook-verify never made (mirroring the rank-12 GNW STOP-trigger production-flip precedent exactly).

## What changed

`webapp/gnw_bus_shadow.py::_organ_reads` is the LIVE production organ-combination read (`webapp/server.py::
brain_reply` runs it every turn). Organs B (VERIFY re-check) and C (reverse-binding) decide "does this second read
CORROBORATE the first" — previously a bare host `==`, now routed through `SpikingCongruenceReader`'s already-6/6-
GO'd `pred_k -> mm_k` match-veto circuit by default. `BRAIN_GNW_CONGRUENCE_SPIKING` explicit falsy
(0/false/off/no/'') is the escape hatch back to the original host `==` logic, byte-identical to pre-2026-09-08
production.

## Verification (`_gnw_congruence_spiking_production_flip_verify.py`, reusing the de-risk + hook-verify's own
fixtures/composer/lesion lever by import — no re-derivation)

**ARM 1 — the flip is real + no regression, 6/6 seeds:**
- (a) with the flag genuinely popped, `_congruence_spiking_enabled()` resolves `True` — asserted in the data.
- (b) explicit opt-out (`="0"`) still reproduces the FROZEN pre-edit host `==` reference byte-for-byte on every
  real query in the `CHAINS` fixture.
- (c) bare-unset `_organ_reads` output on real queries is byte-identical to explicit `="1"` — the new default
  really is the audited spiking-read path, not a different one.

**ARM 2 — load-bearing, not hollow, AT THE SHIPPED DEFAULT (the crux; 6/6 seeds).** The hook-verify's own
manufactured-mismatch + `BRAIN_GNW_CONGRUENCE_LESION` lever is re-exercised with the spiking flag genuinely
**unset** (not an explicit `="1"`): intact -> `bus_combine`'s committed decision matches the explicit-off host
reference (correctly withholds on a manufactured organ-C mismatch this fixture's clean facts never produce
naturally); lesioned -> the false corroboration lets all three organs wrongly agree and the substrate commits the
wrong patient. `attributable_to` credits 100% of the correct-withhold behaviour to the lesion lever on every seed.
This is the flip-specific claim neither the circuit's own GO gate nor the hook-verify made: load-bearing at the
literal shipped default, not merely under an explicit override.

**ARM 3 — cross-faculty regression (one-shot) — LAUNCHED, NOT YET COMPLETE; queued as a follow-up, not gating
this verdict.** `onebrain_regression_battery.run_regression_battery` (reused verbatim) drives the full default-ON
faculty roster through the REAL `webapp.server.brain_chat` handler (`brain="tiny-demo"`, GPU-free) comparing
`BRAIN_GNW_CONGRUENCE_SPIKING` explicit-ON vs explicit-OFF. Run in-agent (numpy, no bundles); its ON-arm worker was
still mid-flight (multi-tens-of-thousands-of-neuron tiny-demo builds across ~25 probe turns, RSS still climbing)
past 6 minutes wall-clock with the OFF-arm (run sequentially, not in parallel) still ahead of it, so it was
terminated to avoid blocking this turn rather than left to finish unattended in a worktree slated for cleanup.
**Unlike rank-12's STOP-trigger** (whose battery probes provably never touch its flag), rank-8's congruence check
sits inside `bus_combine`, the general fact-recall combiner most "known_factual"/"known_followup"-style battery
probes route through — so a blanket "no code path reaches them" claim would NOT be honest here. The bounded
argument that IS supported by this session's own data: ARM 1(c)'s parity proof shows bare-unset is
byte-identical to explicit-ON, and the ORIGINAL hook-verify's CLAIM 2 ("flag-on real-match parity") already showed
explicit-ON is byte-identical to explicit-OFF on every GENUINE (non-manufactured) organ-B/C match in its fixture —
so the flip is a **proven no-op on the common case** (a real fact whose forward/verify/reverse reads agree, which
is what every teaching-then-recall battery probe constructs); the only way ARM 3 could still surface a difference
is a battery probe that happens to manufacture a genuine B/C disagreement, which no PROBE_TURNS entry is designed
to do. Follow-up (queue once this branch is on `main` — the runner does not exist there yet, so
`tools/pool_queue.sh` cannot validate it from a worktree):
`cd /home/dant123/Projects/sim && SIM_BACKEND=numpy .venv/bin/python -u -m
research.runners._gnw_congruence_spiking_production_flip_verify --seeds 42 43 44 100 101 102 --json
research/findings/raw/_gnw_congruence_spiking_production_flip_verify.json` (omit `--no-battery`; ARM 1/ARM 2 are
already 6/6 GO from this session and will simply re-confirm). If ARM 3 later disagrees with the bounded argument
above, that is new information and should be treated as such, not retrofitted into this verdict.

## Mechanical fix required by the flip (gates/flip_offarm_staleness)

`_gnw_congruence_spiking_hook_verify.py`'s own "flag-off" arm previously used `os.environ.pop(...)` to mean OFF —
correct while the flag defaulted OFF, but a silent ON-vs-ON no-op once the default flips (the exact 2026-08-27
staleness class `gates/flip_offarm_staleness` exists to catch). Fixed in the SAME commit: its three non-lesion
pops now set `os.environ["BRAIN_GNW_CONGRUENCE_SPIKING"] = "0"` explicitly, so its own CLAIM 1 ("flag-off
byte-identical") keeps testing the genuine escape-hatch arm. Re-verified 6/6 GO post-fix
(`research/findings/raw/_gnw_congruence_spiking_hook_verify.json`). The gate's own static check
(`tools/gates/flip_offarm_staleness.check`) reports zero violations on all four touched files.

## Honesty boundary

Functional/mechanistic claim only — a spiking population's match-veto read replacing a host string comparison at
one production hook, verified by decision-identity + lesion-collapse, not a self-model or cognitive claim. The
three HONEST RESIDUALS the de-risk itself named (addressing-vs-deciding, the fixed-threshold GPi-style readout
class, and the "reuse of an already-load-bearing circuit is not literal deletion of the host `==` — it remains as
an exception-only fallback") are unchanged by this flip and still apply.
