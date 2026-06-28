# Knowledge-scaling arc — BUILT + CPU-construction-validated (GPU learning-validation pending) (2026-06-27)

**Type:** BUILD (CPU-only; NO `sim/` edit; NO GPU job launched — a live develop run holds the GPU). Implements the
two cheap, default-preserving fixes from the scoping `2026-06-27-develop-knowledge-scaling-arc-scoping.md`
(commit fa47b27f) that together unlock ~300-500 high-recall concepts on a 24GB 3090. The DEFAULT path is
byte-identical (validated, below).

**The single decisive insight (from the scoping):** *D-threading fixes RECALL but not VRAM; the WM-kill fixes VRAM
but not recall; you need BOTH, and both are cheap.* This BUILD lands both.

---

## The two edits

### (1) WM-KILL — `use_multiturn` threaded through `develop_gpu` (scoping §3 option b1)
The per-day CONVERSE agent's `MultiTurnAgent` eagerly builds a persistent discourse working-memory loop (a
`SpikingLoopContextBuffer`) sized `wm_n = 80*len(referents)` neurons at `internal_density=0.1` across 2 regions +
2 cross-pathways — `~1080*V^2` synapses, the ~quadratic VRAM that OOMs at 100s of concepts on 24GB (the CYCLE-685
wall). The per-day develop battery (recall / heldout / retain / chain / yes-no / moat) needs **no cross-turn
anaphora** (multi-hop runs on the composer's `query_chain`, not the WM), so the WM loop is dead weight for the probe.

`build_agent` already had a `use_multiturn` param (its `else` branch builds a plain `BrainConversationalAgent` with
NO WM loop). The build threads it from `develop_gpu`:
- `develop_gpu(..., use_multiturn=True)` — **DEFAULT True = byte-identical** to the validated loop (every existing
  caller — `_corpus_develop_probe`, `_self_knowledge_demo`, `develop_loop_supervisor`, the in-file self-tests —
  keeps the multi-turn agent unchanged).
- the `build_agent` call site (`_longitudinal_develop_loop_gpu.py`) now passes `use_multiturn=use_multiturn`
  (was a hardcoded `True`).
- `develop_run`'s `--corpus-curriculum` (scaling) path passes `use_multiturn=False` (so a future bigger-D
  100s-of-concepts run fits VRAM); the hardcoded ~24-concept demo schedule keeps `use_multiturn=True`.

### (2) THREAD-D — a `D` kwarg the whole chain (scoping §3 option a)
The composer's phasor dimension was a hardcoded literal `128` at two sites, unreachable from the develop loop;
worse, `_inject_grounded` DROPS any learned code whose `v.shape[0] != comp.D`, so even raising `StreamCortex`'s D
alone would have made the brain converse on the composer's *random* codes (the learned codes silently dropped).
The fix threads ONE `D` end-to-end so `StreamCortex.D == comp.D`:

```
develop_gpu(D)  ->  build_agent(D)  ->  MultiTurnAgent(D)  ->  BrainConversationalAgent(D)  ->  OneBrainComposer(D=D)
                                     \-> BrainConversationalAgent(D)  (use_multiturn=False)      RFPhasorComposer(D=D)
                 \-> StreamCortex(D)  (already threaded; the grounded codes are length-D phasors)
```

- `BrainConversationalAgent.__init__(..., D=128)` — new kwarg; substitutes the two literal `128`s
  (`OneBrainComposer(D=D)` + `RFPhasorComposer(D=D)`). When an EXTERNAL `composer` is passed, that composer's own D
  wins (D ignored) — unchanged.
- `MultiTurnAgent.__init__(..., D=128)` — new kwarg; passes `D=D` to the inner `BrainConversationalAgent`.
- `build_agent(..., D=128)` — new kwarg; passes `D=D` to both the multi-turn and plain branches.
- `develop_gpu` already takes `D=128` and threads it to `StreamCortex`; it now ALSO passes `D=D` to `build_agent`,
  so the `_inject_grounded` `v.shape[0]==comp.D` guard PASSES and the brain converses on the codes it LEARNED.
- `develop_run` exposes `--develop-D` (default `None` -> uses `--D`); the resolved `develop_D` drives BOTH
  `StreamCortex(D=develop_D)` AND `develop_gpu(D=develop_D)` from ONE value, so they cannot diverge (the exact
  mismatch the scoping warns about). The startup banner now logs `D=<develop_D> multi_turn=<bool>`.

**DEFAULT `D=128` everywhere = byte-identical.** The composers build `self.concepts = {w: rng.uniform(0,1,D)}`, so
`D=128` reproduces the prior 128-length random codes exactly; `_inject_grounded` (with `comp.D==128`) injects the
128-length grounded codes exactly as before.

### Files (all under `E:\Documents\Projects\sim`, runner-only — NO `sim/` edit)
- `research/runners/brain_conversational_agent.py` — `D=128` kwarg + 2 literal substitutions + docstring.
- `research/runners/multi_turn_agent.py` — `D=128` kwarg + pass-through to the inner agent.
- `research/runners/_longitudinal_develop_loop.py` — `build_agent(..., D=128)` + pass-through both branches + docstring.
- `research/runners/_longitudinal_develop_loop_gpu.py` — `develop_gpu(..., use_multiturn=True)` + thread `D` to
  `build_agent` + docstring.
- `research/runners/develop_run.py` — `--develop-D` flag; one `develop_D` value drives StreamCortex + develop_gpu;
  `use_multiturn=False` on the corpus-curriculum path; banner logs D + multi_turn.

---

## CPU construction-smoke (SIM_BACKEND=numpy, NO GPU) — ALL PASS

`scratchpad/knowledge_scaling_construction_smoke.py` (numpy-CPU). Builds the develop agent with
`use_multiturn=False` + `D=256` over a small vocab, then:

- **WM-KILL:** `use_multiturn=False` -> a plain `BrainConversationalAgent` (NOT a `MultiTurnAgent`), no `wm` attribute
  -> the ~quadratic WM loop is not built. **PASS.**
- **THREAD-D:** `comp.D == 256`. A 256-dim grounded code injected via `_inject_grounded` **LANDS** in
  `comp.concepts['dog']` (matches the injected vector; replaces the composer's random default) — i.e. it is **NOT
  dropped**. A deliberately length-mismatched 128-dim code IS correctly dropped against the D=256 composer (the guard
  is intact both ways). **PASS.**
- **end-to-end wiring:** the agent answers one stored fact (`what_does('dog','chase') == 'cat'`) and the no-confab
  moat abstains (`None`) on an unstored cue. **PASS.**
- **DEFAULT byte-identical:** `build_agent(defaults)` -> a `MultiTurnAgent` (use_multiturn=True), composer `D == 128`,
  WM loop built (`wm is not None`); a plain `BrainConversationalAgent(defaults)` also has composer `D == 128`. **PASS.**

(The numpy backend logs "GPU memory: 15.6GB/51.2GB" as part of its standard reporting — it reports system RAM under
the "NumPy backend (CPU)" device; no CUDA/cupy was imported, no GPU job launched.)

### Regression — the existing agent suites pass verbatim on numpy
`SIM_BACKEND=numpy pytest tests/test_multi_turn_agent.py tests/test_brain_conversational_agent.py` -> **10 passed,
5 skipped** (the 5 skips are the pre-existing `denoise64`-cache-dependent tests per CLAUDE.md, unrelated to this
change). Adding the default `D=128` kwarg + the `use_multiturn` thread did NOT change the default behavior.

`research/runners/develop_run.py --help` shows `--develop-D` wired; `--status` runs with no GPU (reports the live
run's day/vocab/facts).

---

## What is validated vs DEFERRED

- **VALIDATED (CPU, this BUILD):** the threading is correct end-to-end — a bigger-D composer now ACCEPTS the
  stream-learned codes (no silent drop), the WM-kill produces a plain agent, the default path is byte-identical, and
  the existing tests pass. The CONSTRUCTION of both fixes is sound.
- **DEFERRED (GPU learning-recall validation):** the actual recall-rises-above-the-72-cap + moat-stays-0-FA at
  V~128/D=256 and V~200-320/D=512 — and the V=320 VRAM-now-fits demonstration — require GPU days and are **NOT run
  here** (the GPU is occupied by the owner's live develop run). The scoping's per-step de-risks (§4 steps 1-2) are
  the pending GPU validation: (a) a GPU day at V=64 confirming the per-day battery is identical with
  `use_multiturn=False`; (b) a GPU day at V=320 confirming it now FITS where the multi-turn agent OOM'd; (c) GPU days
  at V=128/D=256 + V=200/D=512 confirming recall rises + moat 0-FA.

## How to run the scaled develop later (when the GPU frees)
```
SIM_BACKEND=cupy python -m research.runners.develop_run --corpus-curriculum --develop-D 512 \
    --concepts-per-day 24 --max-concepts 320
```
`--corpus-curriculum` auto-engages the WM-kill (`use_multiturn=False`); `--develop-D 512` lifts the composer recall
margin for ~320 concepts and keeps StreamCortex+composer dimensions consistent. Resume/pause/bundle seams unchanged.

## Hard-rule confirmations
- **NO `sim/` edit** (git `sim/` dirty check empty; all 5 changed files are under `research/runners/`).
- **NO GPU job launched** (only `SIM_BACKEND=numpy` smoke + tests + `develop_run --help/--status`).
- **DEFAULT path byte-identical** (D=128 + use_multiturn=True paths unchanged; tests pass verbatim; construction-smoke
  asserts the defaults).
- **GPU learning-validation DEFERRED** (the owner's live develop run holds the GPU).
