---
type: finding
status: live
date: 2026-09-24
lane: latency-and-cost (production GPU chat path)
mechanism: PRE-REGISTRATION of a per-phase wall-clock + memory-pressure timing instrument for the DEFAULT production `/api/brain-chat` turn on the cupy/GPU backend (research/runners/_prod_chat_phase_timing.py, a runner version of tests/test_production_chat_gpu_smoke.py) — a HOST-SIDE measurement instrument, not a cognitive mechanism; it adds no new BRAIN_* flag and changes nothing the brain computes
seeds: ["42 (production default, unset BRAIN_CHAT_SEED)"]
verdict: PRE-REGISTRATION only. No timing result is claimed here.
runner: research/runners/_prod_chat_phase_timing.py
artifacts:
  - research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json
---

# Production chat phase timing: PRE-REGISTRATION (2026-09-24, S01/G1)

Filed on branch `research/g1-prod-chat-phase-timing`, cut from `origin/main`, in its own commit together with the
runner it governs (`research/runners/_prod_chat_phase_timing.py`) and BEFORE either governed artifact exists. The
artifact cited above is PRIOR evidence this prereg builds on (the existing full-brain SETTLE NO-GO, see "Why"
below) — the two artifacts THIS prereg's own run will produce do not exist yet, by construction, and are named
only with a `<mode>` placeholder in the Commands section below so no not-yet-existing path is asserted as cited
evidence (the convention `2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md` also uses, there with
`<seed>`).
`prereg-same-commit: the runner and this prereg are non-evaluative code — no research/findings/raw/** artifact is
staged in this commit, so gates/prereg_before_run has nothing to order here regardless.`

## Why (context this fixes-forward from)

`tests/test_production_chat_gpu_smoke.py` is a class guard: it proves the default chat turn (tiny-demo + the
off-bridge spiking Qwen-0.5B mouth) answers HTTP 200 on the cupy backend instead of 400-crashing, GPU-present
only. It says nothing about HOW LONG any phase takes, nor whether the box is being reclaim-throttled under a
`tools/memcap.sh` cap. Two open questions this instrument exists to answer:

1. Tonight's plan needs a GO/NO-GO-adjacent number for `BRAIN_AFFECT_MARKER_SETTLE` (which lengthens the
   affect-marker circuit's deliberation window 60→500 ms and inter-turn rest 40→1000 ms simulated,
   `research/runners/_affect_marker_wta_derisk.py`) before it can be considered for a default-on flip: does
   turning it on measurably slow down a WARM turn on the real production GPU chat path, not just the isolated
   affect-marker circuit (`_settle_turn_cost_probe.py` already measures that in isolation — this measures the
   same flag through `webapp.server.brain_chat` itself).
2. Whether a `tools/memcap.sh`-capped production chat build gets reclaim-throttled (a rising cgroup
   `memory.events` `high` count) before completing, on a box that is RAM-tight by design (46 GB total, often
   <12 GB free).

## Criterion L (fixed now, before any run)

**SETTLE warm-turn delta ≤ +0.3 s on GPU.** Concretely: given two runs of `_prod_chat_phase_timing.py` at the
same commit, same backend (`SIM_BACKEND=cupy`), same seed (BRAIN_CHAT_SEED unset → production default 42), same
two messages, differing ONLY in `BRAIN_AFFECT_MARKER_SETTLE` (1 vs 0/unset) —

    delta_s = on["warm_turn_total_s"] - off["warm_turn_total_s"]
    PASS iff delta_s <= 0.3

where `warm_turn_total_s` is turn 2's total wall time (cache-hit: no brain build, no Qwen weight load — the
same session's SECOND `brain_chat()` call). This is a **cost measurement, not a capability gate**: it says
whether SETTLE is affordable on the GPU chat path, and carries no opinion on SETTLE's own (separately-gated,
already-NO-GO'd on its full-brain contrast, `research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json`)
functional effect. `_prod_chat_phase_timing.py --compare-on --compare-off` computes this mechanically from the
two run artifacts named in the Commands section; PASS/FAIL is not eyeballed.

**This is de-risk, not a validation-seed claim.** One seed (production default 42), one message pair. A single
run pair says only whether the flag is affordable at all on this box today, not that it generalizes across
seeds/messages — no 6-seed rule applies to a latency measurement (it is not a capability GO/NO-GO), but the
number is reported as "de-risk (1 seed)", never as a generalizable throughput claim.

## What the runner measures (four phases, each stamped with `/proc/loadavg` + this process's own cgroup
`memory.events` `high` count)

1. **brain_build** — constructing the tiny-demo spiking substrate (`webapp.server._build_chat_brain`, minus the
   Qwen sub-span below). Paid once, inside the first `brain_chat()` call.
2. **qwen_weight_load** — the off-bridge Qwen-0.5B model load + the P1b calibration pass + spiking-op install
   (`webapp.server._get_warm_qwen_renderer` → `QwenRenderer.__init__` → `SpikingQwenFaculty.__init__`), a
   process-wide singleton paid once per process. Declared honestly: this number is the WHOLE one-time setup
   inside that function, not model-weight I/O alone — it also includes the calibration forward pass and
   installing the spiking ops (`_grounded_lang_p1b_stepB1_forward_derisk.install_spiking_ops`).
3. **turn_1** — the first answer's wall time MINUS the brain-build + Qwen-load span (so it reads the actual
   gate/compose/render cost of one turn, not construction), plus the Qwen CUDA generation span isolated within
   it (see below).
4. **turn_2** (total / `qwen_cuda_generation` / non-generation) — a second, WARM turn on the same session (no
   build, no weight load): total wall time, the isolated `model.generate()` CUDA span, and the remainder. This
   is `warm_turn_total_s`, the quantity criterion L compares.

Phase separation WRAPS four existing functions for the run's duration only (`webapp.server._build_chat_brain`,
`webapp.server._get_warm_qwen_renderer`, `SpikingQwenFaculty._generate`, `SpikingQwenFaculty._generate_batch` —
the latter two cover both the single-item and the batched-launch render paths, since the production `rich`
default can route through either). **No production module is edited on disk.** This is purely a host-side
instrument around an unmodified call, not a change to what the brain computes — see the brain-based-only
boundary in CLAUDE.md; the only host residual here is the measurement itself (timers + `/proc` reads), which
computes nothing between sensation and action.

## Honest scope / what this does NOT claim

- **Not a capability measurement.** No lesion, no null control, no accuracy/recall check. `turn1_http_status` /
  `abstained` are recorded only as a sanity check that the turn actually answered (matching the class-guard
  test's own 200-not-400 assertion), not as evidence about the faculty being timed.
- **Not a 6-seed validation.** One seed at production default. A generalization claim needs the multi-seed
  battery this plan explicitly reserves for the orchestrator (this runner/lane does **not** run or queue a
  6-seed evaluation).
- **The Qwen-weight-load number conflates model load + calibration + op-install** (declared above); a finer
  split was not built because nothing downstream currently needs it more finely resolved.
- **`memory.events` `high` absence is reported as `null`, never coerced to `0`** (tools.lab's undefined-not-zero
  discipline): a `null` means the cgroup path was not readable (no cgroup v2, not inside a memcap.sh scope,
  permission), a genuine `0` means the cap was never touched.
- **A `TIMED_OUT` artifact is not a NO-GO.** The runner's own watchdog thread force-exits at `--timeout-s`
  (default 1800 s, matching this step's own deadline) and writes `{"status": "TIMED_OUT"}` rather than hanging
  the GPU queue forever; that is an operational safety net, not a measured result.

## Commands (queued, not run ad hoc against the shared 3090 — see `tools/gpu_queue.sh`)

Gate: `bash tools/mem_ok.sh 20 4` (checked ONCE by the queue dispatcher before the job starts; the job itself
runs twice back-to-back under one `memcap.sh` invocation).

(paths below use a `<mode>` placeholder, `<mode>` = `on` then `off`, for the two not-yet-existing output
artifacts — see the note under the frontmatter for why a literal future path is not written here.)

```
bash tools/gpu_queue.sh add 'bash tools/mem_ok.sh 20 4 && \
  SIM_BACKEND=cupy OMP_NUM_THREADS=1 BRAIN_AFFECT_MARKER_SETTLE=1 bash tools/memcap.sh 22 -- \
    .venv/bin/python -u -m research.runners._prod_chat_phase_timing --mode settle_on \
    --out research/findings/raw/_settle_cost/prod_chat_phase_settle_<mode>.json && \
  SIM_BACKEND=cupy OMP_NUM_THREADS=1 BRAIN_AFFECT_MARKER_SETTLE=0 bash tools/memcap.sh 22 -- \
    .venv/bin/python -u -m research.runners._prod_chat_phase_timing --mode settle_off \
    --out research/findings/raw/_settle_cost/prod_chat_phase_settle_<mode>.json'
```

(with the first `<mode>` = `on` and the second `<mode>` = `off`, matching the `--mode` given on each line)

Dev/calibration smoke (CPU, no GPU, seed 7 ONLY — never a validation seed):

```
SIM_BACKEND=numpy BRAIN_CHAT_SEED=7 OMP_NUM_THREADS=1 .venv/bin/python -u \
  -m research.runners._prod_chat_phase_timing --renderer stub --seed 7 --out <scratch_dir>/smoke.json
```

Evaluate criterion L once both artifacts exist:

```
.venv/bin/python -m research.runners._prod_chat_phase_timing \
  --compare-on research/findings/raw/_settle_cost/prod_chat_phase_settle_<mode>.json \
  --compare-off research/findings/raw/_settle_cost/prod_chat_phase_settle_<mode>.json
```

(`<mode>` = `on` for `--compare-on`, `off` for `--compare-off`)

## Amendment log

(none yet)
