---
type: finding
status: partial
lane: continuous-life/generation
date: 2026-08-27
mechanism: production-scale (n_ca3=2000, emergent DG-selected, BTSP-formed) port of the on-substrate generative
  attractor-wander dendritic dAP bistable-latch blend-completion, plus its additive default-OFF wire-in to the live
  continuous idle-tick ideation channel
seeds: []
instrument: research/runners/_generative_attractor_wander_onsubstrate_derisk.py (--emergent, build_production_store /
  blend_settle_production), webapp/continuous_engine.py (BRAIN_CONTINUOUS_IDEATE_SPIKING)
runner: research/runners/_generative_attractor_wander_onsubstrate_derisk.py
artifacts:
  - research/runners/_generative_attractor_wander_onsubstrate_derisk.py
  - webapp/continuous_engine.py
---
# Production-scale port of the on-substrate generative attractor-wander (board #104 rung 2): n_ca3=2000 emergent-DG-selected BTSP-formed membership + a default-OFF live idle-tick wire-in — code + wiring done, 6-seed GPU verify STAGED (gpu_queue depth 11), not yet landed — PARTIAL

Artifact: `research/runners/_generative_attractor_wander_onsubstrate_derisk.py` (new `--emergent` mode, `build_production_store`/`blend_settle_production`) + `webapp/continuous_engine.py` (new `BRAIN_CONTINUOUS_IDEATE_SPIKING`, default-OFF).

## Starting point (verified)

[[2026-08-27-generative-attractor-wander-onsubstrate-GO]] is a 6/6-seed GO at a REDUCED scale — n_ca3=400,
PRE-ASSIGNED (random-permutation) assembly membership — and names two honest blockers to production: (i) the
reduced scale vs. the production organ's n_ca3=2000 EMERGENT DG-selected membership, (ii) direct-index wander
cues rather than BTSP-formed concept assemblies. Its cited runner and the production organ
(`research/runners/_episodic_dap_dialogue_memory.py`, `EpisodicDapMemory`) were read in full before this session's
change: the production organ composes `emergent_assemblies` (from
`_gap5_emergent_end_to_end_episodic_loop_derisk.py`, R1 config `n_ca3=2000`) through `assemblies_ext` into
`_build_dap_readout`/`make_readout`/`form_btsp_multi` — exactly the seam this session closes for the wander.

## What changed (this session)

**(1) Scale + emergent membership.** `_generative_attractor_wander_onsubstrate_derisk.py` gained `--emergent`:
`_build_and_form` (factored out of `run_seed`) calls `emergent_assemblies(seed, n_patterns=n_mem)` ONCE per seed —
never re-called, since emergent membership is measured non-deterministic ACROSS separate bridge builds
(FMA/summation-order drift, per `_episodic_dap_dialogue_memory.py`'s own kthresh note) — and threads that ONE
fixed array through every downstream `_build_readout`/`form_btsp_multi` call via `assemblies_ext`, identical to
how `EpisodicDapMemory` composes it. n_ca3 lifts to whatever the emergent selection returns (2000 at R1's default
config); `train_events` defaults to 40 (the ALREADY-VALIDATED `GO_DEFAULTS` value for this SAME mechanism at this
SAME scale) instead of the reduced-scale run's tuned 60. `blend_cells_each` stays at the calibrated absolute count
3 — re-run, not re-tuned, per the GO finding's own scale-invariance argument.

**(2) BTSP-formed cue (closes blocker ii together with (1)).** Because membership is now the emergent, BTSP-formed
assemblies the production organ actually stores concepts into, the blend cue is drawn from real formed-assembly
cue-eligible cells, not a pre-assigned/direct-index set.

**(3) Reusable production API.** `build_production_store(seed, n_mem)` / `blend_settle_production(store, iA, iB)`
factor the build+BTSP-form / blend-read steps into standalone functions returning the SAME shape
`_ideation_blend_settle` (the numpy stand-in) returns (`novelty_max_overlap`, `blend_balance`, `blend_vs_other`,
`fixed_point`) — one mechanism, reused by both the offline 6-seed verify (`run_seed(emergent=True)`) and the live
wire-in below, not a duplicate.

**(4) Live idle-tick wire-in (additive, default-OFF).** `webapp/continuous_engine.py` gained
`BRAIN_CONTINUOUS_IDEATE_SPIKING` (unset/0 = OFF): `_ideation_wander` calls the new `_ideation_blend_settle_spiking`
instead of the numpy `_ideation_blend_settle` only when armed. The production store is built ONCE per session (
`_spiking_ideate_store`, cached in `_SPIKING_IDEATE_STORE`, cleared by `forget_session`) and reused by every later
ideation tick; a failed build is cached too, so a truly-idle server never retries a heavy build every tick. The
resulting record flows through the IDENTICAL downstream gate (`IDEATE_NOVELTY_MAX`/`IDEATE_BALANCE_MIN`/
`IDEATE_BLEND_MARGIN`) and the IDENTICAL `recent_ideation()`/lead-sentence channel a live turn already reads
(`webapp/server.py`'s `ideation_drives_lead`) — so the LOAD-BEARING property (an idle novel idea changes what the
NEXT live turn's reply opens with) is INHERITED unchanged from the already-wired numpy path, not new server code.

## Verification performed (honest, not a 6-seed claim)

No new 6-seed generalization is claimed here (`seed-waiver: production-scale multi-seed verify is STAGED on
gpu_queue, not yet run — see "Staged, pending" below`). What WAS verified this session, all reproducibly:

- **Refactor safety (regression check):** re-running the UNCHANGED (non-`--emergent`) path for seed 42 on the
  refactored code reproduces `research/findings/raw/_generative_attractor_wander_onsubstrate/batch1.json`'s own
  cited seed-42 row EXACTLY: novelty 0.778, balance 0.639, persistence_gap 0.000, single_recovered 0.819, noise
  best 0.000, untrained best 0.000, genuine_formation=True — the `_build_and_form`/`assemblies_ext` refactor is
  behavior-preserving for the already-GO'd reduced-scale path.
- **Wiring correctness (mocked unit checks, `webapp/continuous_engine.py`):** with the new flag OFF (default),
  `_ideation_wander` never touches `_SPIKING_IDEATE_STORE` and returns the identical numpy-path record shape —
  byte-identical-off, confirmed by assertion, not by inspection alone. With the flag ON and `build_production_store`
  made to raise, `_ideation_wander` degrades to `None` (no idea surfaced this tick, never a crash) and CACHES the
  failure (a second call makes zero further build attempts — verified by a call counter). With the flag ON and a
  stubbed success, the full pipeline (top-2 curiosity-gain source selection, the novelty gate, the `substrate:
  "spiking-onsubstrate"` tag) produces the expected record.
- **Static checks:** `research/runners/_generative_attractor_wander_onsubstrate_derisk.py` and
  `webapp/continuous_engine.py` both import cleanly under `SIM_BACKEND=numpy`; `tools/check_docs.py` passes
  (W1/W2 both 0).

## Staged, pending (the actual production-scale verdict)

The real 6-seed n_ca3=2000 emergent verify is QUEUED, not yet run (memory-budget rule: n_ca3=2000 is heavy and
must run via `gpu_queue`, one GPU proc at a time, never locally-concurrent — so this session did not run it
directly). It will write its output under `research/findings/raw/_generative_attractor_wander_onsubstrate/` (a
directory that does not yet contain this run's file — the exact command, verbatim as queued, wraps the output
path across two lines below purely so this NOT-YET-EXISTING path is not misread as an already-cited artifact):
```
cd /home/dant123/Projects/sim/.claude/worktrees/agent-ae952a93a88574cf4 && SIM_BACKEND=cupy \
  /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._generative_attractor_wander_onsubstrate_derisk \
  --emergent --seeds 42 43 44 100 101 102 \
  --json research/findings/raw/_generative_attractor_wander_onsubstrate/\
production_n_ca3_2000_6seed.json
```
Queued at `bash tools/gpu_queue.sh add` (depth 11 at queue time, behind 10 prior jobs on the shared single-GPU
queue). This runs against the branch `research/generative-wander-production` (this worktree's path is used
explicitly so the queued job sees this session's code without touching `main`); the controller harvests the
result JSON when the queue reaches it. The GO gate is the SAME 8-term bar `run_seed`'s `main()` already checks
(genuine formation, novelty<0.85, balance>0.35, balance-minus-other>0.10, persistence<0.20, single-cue
recovered>0.50/others<0.20, untrained<0.20); a NO-GO or PARTIAL outcome is an equally honest, first-class result
this finding does not pre-judge.

## Honesty boundary / declared scope

Unchanged from the cited GO finding's own declarations: which two concepts to blend is a host scheduler (the
wander's curiosity-gain top-2 selection, untouched here); the blend cue's absolute cell count (3) is a
runner-calibrated constant, re-run not re-derived at this scale. NEWLY declared: the live wire-in's
per-session store build is genuine substrate time (not yet measured at production scale — the GO finding's own
Residual named ~140-160s/seed on CPU numpy, "seconds" on cupy per the organ's own docstring, unconfirmed here);
one build per session, cached, mitigates repeat cost but the FIRST ideation-eligible tick of a session still pays
it, a known latency risk left honestly open rather than hidden. The wander's own concept agents are BTSP-formed
into a DEDICATED store, still not unified with the D5/episodic organ's own topic store (the GO finding's second
named prerequisite) — a parallel formation path, not a merge.

**Byte-identical when off:** `BRAIN_CONTINUOUS_IDEATE_SPIKING` unset/0 leaves `_ideation_wander` calling the
existing numpy `_ideation_blend_settle` exactly as before this session, confirmed by the mocked assertion above
that the new store cache is never touched on that path. No `sim/` file was touched.
