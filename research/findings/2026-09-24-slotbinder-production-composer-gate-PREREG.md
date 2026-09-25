---
type: preregistration
status: preregistered
date: 2026-09-24
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
mechanism: BRAIN_COMPOSER_KIND=slotbinder through the REAL production chat path (webapp.server._build_chat_brain ->
  developed_brain_io.load_developed_brain -> MultiTurnAgent -> BrainConversationalAgent -> SlotBinderComposer)
seeds: [7 (dev), 42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. No cupy production-scale result exists yet.
---

# SlotBinder PRODUCTION composer gate: pre-registration (6-seed, real chat path, SIM_BACKEND=cupy)

## Why this gate, and why it differs from the L3 latency de-risk

`research/findings/2026-09-24-slotbinder-l3-cupy-latency-derisk-PREREG.md` (and its 6-seed GO,
`2026-09-24-slotbinder-l3-gpu-latency-GO-6seed.md`) measured `SlotBinderComposer(...)` constructed DIRECTLY, at 2
facts/seed, on the GPU: ~1 s/query, recall 1.0, moat/mismatch passing. It never called `load_developed_brain` or
`webapp.server._build_chat_brain`, so it never exercised `developed_brain_io._restore_facts`'s REAL production
fact-restore semantics (for `composer_kind='slotbinder'`, `_restore_facts` re-`store()`s -- re-teaches -- EVERY fact
in the loaded bundle, since `SlotBinderComposer` has no `.kb` composite fast-path). A default-OFF production route
already exists: `BRAIN_COMPOSER_KIND=slotbinder` routes `webapp/server.py`'s developed-brain loader through exactly
that path (`research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md`).

This gate runs the SAME production entry point `/api/brain-chat` uses (`_build_chat_brain`) end to end, at the
largest fact count the real corpus actually has (see FACT SCALE below), and requires the gate to be able to FAIL
(an ablation that zeroes the built SlotBinder bridge's own synapses must collapse recall -- a composer answering via
a host shortcut, not the taught synapses, would be unaffected).

Runner: `research/runners/_slotbinder_production_gate.py` (this commit). Read its module docstring for the full
per-arm protocol; this document states only the run command, the fact scale, and the criteria fixed BEFORE any
`SIM_BACKEND=cupy` result exists.

## FACT SCALE (measure-and-state, decided from the runner's own sizing note, not a new claim)

The real day_33 bundle carries K=2020/KF=1195 (404 facts, 788-word vocab) -- fixed, because
`load_developed_brain` always sizes `slotbinder_max_facts=len(facts)` from whatever bundle it is given, so there is
no partial-teach mode against the full bundle. `fanout=32` (this gate's default) pins the effective per-slot
candidate set at 32 regardless of vocab size, so nnz -- and VRAM -- scales with `K=5N` synapses alone
(~14,160 synapses/fact), trivial against a 24 GB consumer GPU at any N this 404-fact corpus can produce. **The
largest fact count that fits is therefore the WHOLE corpus: N=404** -- there is no smaller-than-full VRAM wall to
size against at this topology, so "largest that fits" and "the full real bundle" coincide. If the dev-seed run
below measures otherwise (VRAM pressure, or a wall-clock budget that makes N=404 impractical per seed), this
document is AMENDED with the measured ceiling before the 6-seed battery is queued, not silently substituted.

## Run (fixed)

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_production_gate \
    --seed <SEED> --n-facts 404 --fanout 32 --renderer stub \
    --out research/findings/raw/_slotbinder_production_gate/seed<SEED>.json
```
Dev seed 7 first, un-queued (small enough to run to completion and read by hand). Queued through
`bash tools/gpu_queue.sh add` (one brain-loading GPU process at a time), each invocation wrapped in
`until bash tools/mem_ok.sh 12 4; do sleep 30; done` per the lane brief, for the six evaluation seeds
(42 43 44 100 101 102) once the dev seed reads sane. `--check-flagoff` is passed for the dev seed only (cheap,
once), not for the 6-seed battery.

## Criteria (written before any cupy result)

A seed is **GO** iff every one of `verdict_criteria` in the runner's own output JSON is `true`:
- `recall_ge_fhrr`: SlotBinder's recall accuracy over the sampled facts is `>=` the FHRR (`rf`) reference arm's.
- `parity_1_0`: SlotBinder's answer matches the FHRR arm's answer on every sampled question (`parity_rate == 1.0`).
- `slotbinder_moat_pass` / `fhrr_moat_pass`: a never-taught (agent, action) pair abstains on both composers.
- `slotbinder_mismatch_pass` / `fhrr_mismatch_pass`: cross-fact agent/action pairing does not leak the wrong patient.
- `ablation_falsifies_intact_pass`: zeroing the built SlotBinder bridge's synapses (`cp_connections.data[:] = 0`)
  COLLAPSES its recall accuracy below the intact value -- the falsifiability requirement. A composer that is
  unaffected by this ablation is a NO-GO regardless of every other criterion (it would mean recall is not coming
  from the taught synapses).

The gate's headline verdict ("SlotBinder production composer gate GO N/6") requires all of the above `true` on
every one of the six evaluation seeds (42 43 44 100 101 102); seed 7 is dev-only per `feedback_6seed_validation`.
Per-turn latency (p50/p95 vs the FHRR composer on the same GPU) and the `--check-flagoff` resolved-class check are
reported alongside as measurements, not gating criteria (latency was already gated at L3; flag-off byte-identity is
established by `git diff main` over wiring files, not by this smoke -- see the runner docstring's point 6 and the
2026-09-24 FAILURE_LOG entry on `--check-flagoff`'s known limitation against the tiny synthetic dev-seed bundle).

## Known residuals going in (not gated by this document; carried from the runner's own docstring + FAILURE_LOG)

1. `SlotBinderComposer`'s recall composer uses exact-inverse FHRR bind/unbind arithmetic -- host arithmetic, a
   declared open item from the L1-L3 findings, unchanged by this gate.
2. `research/runners/brain_chat_tui.py`'s `ChatBrain._refresh_facts()` read `comp.kb` unconditionally and crashed
   for `composer_kind='slotbinder'` (no `.kb`, facts live in `.facts`) -- found smoke-testing this gate's script
   logic on CPU/numpy, CLOSED in this commit (`_refresh_facts` now falls back to `comp.facts` when `.kb` is
   absent; a no-op by inspection for every composer kind that already has `.kb`). See the FAILURE_LOG row dated
   2026-09-24.
3. `--check-flagoff` against the tiny synthetic (`n_facts=3`) dev-seed sample bundle raises `KeyError:
   'onebrain_composer'` building the production-default (`onebrain`) arm -- a bundle-structure gap in the ad hoc
   sample bundle, not in the SlotBinder path this gate is about. OPEN, not gating this document's criteria. See
   the FAILURE_LOG row dated 2026-09-24.

## AMENDMENT 1 (2026-09-25 03:10, orchestrator) -- measured wall-clock ceiling, before any 6-seed battery

The dev-seed run registered above (`--seed 7 --n-facts 404 --fanout 32 --check-flagoff`, cupy, GPU queue, worktree
`agent-ad44199e6b8cdcc9b`) started 2026-09-24 21:00:38 and printed `[seed 7] running arm=slotbinder ...` after a 266.9 s
staging build. It was still inside that FIRST arm (of three: slotbinder, the FHRR reference, flag-off) at 03:08, after 6 h
08 min of GPU time, with no further output: the runner prints nothing per query, so per-query latency and progress
were not observable. It was stopped (SIGTERM, rc 143) to free the shared GPU for three queued jobs. No artifact was written.

This is the case the sizing note above anticipates: N=404 is impractical per seed on this path, against the L3 de-risk's
~1 s/query for a directly constructed composer (so the production `load_developed_brain` path costs orders of magnitude more
per query, consistent with 2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency). Before any 6-seed
battery: (1) add per-fact progress and per-query latency lines to the runner; (2) re-run seed 7 at small N (e.g. 8, 32, 128)
to measure the per-query cost on this path; (3) register the largest N whose three arms finish within a stated wall-clock
budget as a further amendment. The gate criteria themselves are unchanged.
