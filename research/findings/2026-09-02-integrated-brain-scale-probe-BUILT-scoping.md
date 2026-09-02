---
status: instrument-built (scoping; the full 400-vs-2000 measurement is QUEUED for the GPU, not yet run)
lane: experiment-B de-risk / episodic-ideation organ scale (n_ca3 400 vs 2000) resource cost
type: finding
date: 2026-09-02
seed-waiver: measurement-only resource profile (build wall-time / peak VRAM / peak RSS / per-turn latency are
  not stochastic capability GO/NO-GO verdicts). The correctness SMOKE is seed 42; the full measurement is
  likewise a resource profile, deterministic in scale (n_ca3 is fixed per arm; emergent membership varies
  build-to-build in assembly SIZE only, which the probe records but does not average into a headline claim).
instrument: research/runners/_integrated_brain_scale_probe.py -- reuse-by-import of the EXACT production
  build+BTSP-form path (`_generative_attractor_wander_onsubstrate_derisk.build_production_store` /
  `_build_and_form`, the same call `webapp/continuous_engine.py`'s live ideation wiring makes) + the shared
  gap#5 dendritic-dAP readout the D5 episodic organ composes. Uses the PROCESS backend (does not force one).
runner: research/runners/_integrated_brain_scale_probe.py
artifacts:
  - research/findings/raw/_integrated_brain_scale/smoke.json (numpy CPU correctness smoke, seed 42, two small
    PRE-ASSIGNED arms n_ca3 200 vs 400 -- validates every field is measured + the anti-cheat cross-arm
    CA3-count assert fires; NOT the production measurement)
  - research/runners/_integrated_brain_scale_probe.py (the probe)
  - webapp/server.py (lines 45-80: the production cupy/GPU backend default -- the premise correction below)
  - webapp/continuous_engine.py (lines 931-1007: the n_ca3 400-vs-2000 ideation-organ note + the live
    `build_production_store` call site)
  - research/runners/_episodic_dap_dialogue_memory.py (the production D5 episodic organ; n_ca3=2000 emergent)
---

# Integrated-brain episodic/ideation organ SCALE probe (n_ca3 400 vs 2000) — BUILT + smoke-validated; measurement QUEUED

## What this is

A trustworthy resource probe for owner-requested experiment "B": the cost of running the integrated chat brain's
episodic/ideation spiking organ at the PRODUCTION scale (n_ca3=2000, emergent DG-selected membership) versus the
REDUCED de-risk stand-in scale (n_ca3=400, pre-assigned membership). It does NOT test conversation quality — it
measures build wall-time, peak GPU VRAM, peak CPU RSS, and per-turn latency, so the controller can decide whether
2000 fits the 24 GB card and how much per-turn latency it adds before any quality experiment is scoped.

## STEP 1 — CHARACTERIZATION (read, not assumed)

**The scale knob is NOT a whole-brain neuron count.** `n_ca3` is the CA3 assembly scale of ONE organ family — the
gap#5 dendritic-dAP CA3 store — realised in two live places, both distinct from the main conversational bridge:

- **D5 EPISODIC recall organ** (`d5_episodic_production_organ.get_episodic_organ` -> `_episodic_dap_dialogue_memory.
  EpisodicDapMemory`). Default-ON (`BRAIN_EPISODIC`). It is ALREADY at the production scale: `EpisodicDapMemory`
  always builds membership via `emergent_assemblies` (R1 config, n_ca3=2000). There is no 400 variant of THIS organ
  — the 400 scale is history/stand-in, not a live episodic-recall setting.
- **IDEATION organ** (`webapp/continuous_engine.py`). Default numpy stand-in (`_ideation_blend_settle`, a small
  numpy Hopfield net — not even a CA3 bridge). Its on-substrate port
  (`_generative_attractor_wander_onsubstrate_derisk.build_production_store`, n_ca3=2000 emergent) is behind
  `BRAIN_CONTINUOUS_IDEATE_SPIKING` (default-OFF). The 400 pre-assigned membership is the DE-RISK stand-in scale
  that on-substrate port was validated at (`P["n_ca3"]=400`), before it lifts to 2000 emergent.

So "n_ca3=400 vs 2000" precisely means: the pre-assigned-membership DE-RISK stand-in scale (400) versus the
emergent-DG-selected PRODUCTION scale (2000) of this one dendritic-dAP CA3 store. The rest of the integrated brain
(the `build_one_brain` co-resident bridge + the spiking-Qwen mouth) is scale-INVARIANT in n_ca3, so the 400->2000
delta is entirely in this organ; total@scale = fixed integrated baseline + organ(scale).

**(b) Backend — the premise correction.** My initial guess ("does it fit 24 GB VRAM?") was doubted on the grounds
that the substrate runs on numpy/CPU. That is TRUE only of the `_conversation_turing_test_derisk.py` eval harness,
which deliberately forces `SIM_BACKEND=numpy` (a deterministic small-scale test substrate with the Qwen mouth on
CUDA). The LIVE production `/api/brain-chat` path does the OPPOSITE: `webapp/server.py:45-80` `setdefault`s
`SIM_BACKEND=cupy` before any sim import whenever a CUDA GPU is present (numpy is only the GPU-less fallback or an
explicit CPU override). So in production BOTH the spiking substrate AND the Qwen mouth live on the GPU.

**(c) Which resource the knob stresses.** On the production cupy/GPU backend, scaling the organ 400->2000 grows the
CA3 dendritic-dAP bridge, which lives in GPU VRAM — so the resource is **GPU VRAM** (the genuine "does 2000 + the
mouth fit 24 GB?" question) plus **cupy per-turn wall-clock**. The Qwen mouth's VRAM is fixed w.r.t. n_ca3 but is
likely the largest single consumer, so total-fit depends on mouth + main bridge + organ(2000). CPU RSS is captured
for completeness but is NOT the production headline. (The numpy path is a different regime: there the organ is on
CPU, VRAM is 0, and the codebase already documents the BTSP store as ~510 s/topic on numpy@2000 — `server.py:3404`
`_episodic_store_ok` DEFERS the episodic WRITE on numpy for exactly this reason; on cupy it is ~seconds/topic.)

## STEP 2 — THE PROBE

`research/runners/_integrated_brain_scale_probe.py`. Per arm (`--n-ca3`, an arm==2000 uses the emergent production
path; any other value uses pre-assigned at that n_ca3):

- **organ build** — `build_production_store` (2000, emergent) or `_build_and_form(emergent=False)` (400,
  pre-assigned): times `organ_build_s` (bridge builds + BTSP formation).
- **per-turn read** — one `blend_settle_production` (the live per-idle-tick ideation read); if smoke budgets make
  formation non-genuine so that read short-circuits, a direct fresh-bridge drive/read is timed instead, so the
  substrate read path is always exercised + timed.
- **`--with-integrated`** (GPU) — ALSO builds `build_one_brain` (co-resident bridge) + fm world-model + the
  converted spiking-Qwen mouth and drives ONE real HUMAN_TURN through the turing-test `run_conversation` driver,
  reporting the TOTAL integrated VRAM (cupy substrate pool + torch mouth allocator) and a real per-turn latency.
- **resources** — peak VRAM = this-process cupy pool total + torch reserved (0.0 honestly on the numpy substrate);
  peak RSS via `resource.getrusage` + `/proc/self/status`; nvidia-smi sampled as whole-GPU CONTEXT (other procs).
- **anti-cheat** — each arm reads the ACTUAL substrate CA3 cell count (`len(R.ca3_idx)`) + formed assembly sizes;
  across arms the probe ASSERTS the count grew by the requested factor and prints INVALID (exit 2) if two arms
  report the same n_ca3 (the knob did not engage — no bogus "no difference").

Per-arm JSON fields: `arm_n_ca3, actual_n_ca3, actual_assembly_sizes, organ_bridge_n_neurons, backend_substrate,
backend_mouth, organ_build_s, organ_recall_read_s (/organ_direct_read_s), integrated_build_s, per_turn_latency_s,
peak_rss_mb, cupy_pool_total_mb, torch_reserved_mb, peak_vram_mb, nvidia_smi_used_mb, notes`.

## STEP 3 — SMOKE VALIDATION (numpy CPU, correctness only)

`research/findings/raw/_integrated_brain_scale/smoke.json` (seed 42, `--smoke --n-ca3 200,400 --n-mem 2`, numpy).
Two SMALL pre-assigned arms so the run is seconds on CPU and the cross-arm anti-cheat can fire. Confirmed: every
field is populated; `actual_n_ca3` reads 200 and 400 off the real substrate (`R.ca3_idx`); assembly sizes 36 vs 72
(the pre-assigned 0.18*n_ca3, scaling correctly); `organ_bridge_n_neurons` 1778 vs 2028; the read path is exercised
(`organ_direct_read_s` ~0.5 s vs ~1.1 s, the blend read short-circuited under the reduced smoke budget as designed);
`peak_vram_mb` 0.0 reported HONESTLY (substrate on CPU); `peak_rss_mb` < 750 MB (well under the 4 GB budget); and the
anti-cheat PASSES: "substrate CA3 count grew 200->400 (ratio 2.00 ~ requested 2.00)". The smoke is a PLUMBING check
— reduced BTSP/step budgets, so `genuine_formation=false` and it does NOT reproduce the GO physics; that is expected
and does not affect the resource-instrument validation.

## STEP 4 — THE FULL MEASUREMENT (for the controller to queue; needs the GPU)

Because production is cupy/GPU and the mouth is on CUDA, the production-faithful measurement is a **GPU job** — queue
it on `tools/gpu_queue.sh` (sequential, VRAM-contention-safe), NOT the CPU pool. It loads the brain on the GPU, so it
must respect the one-brain-loading-GPU-proc-at-a-time rule (run it when the current GPU campaign yields, or via the
queue that serialises it).

Command (both arms in one process so the cross-arm anti-cheat runs in-process; the mouth loads once per arm):

    SIM_BACKEND=cupy PYTHONPATH=$PWD .venv/bin/python -m research.runners._integrated_brain_scale_probe \
        --n-ca3 400,2000 --n-mem 3 --with-integrated --seed 42 \
        --out research/findings/raw/_integrated_brain_scale/production_400_vs_2000_s42

The `--out` basename is left extension-less here on purpose (the runner writes the JSON verbatim to whatever path
is given; a concrete `...s42.json` is not written into this scoping doc because that result does not exist yet and
must not read as a citation). The controller may append `.json` when it runs. Output lands in
`research/findings/raw/_integrated_brain_scale/`.

Expected runtime: on cupy the BTSP formation is ~seconds/topic (vs ~510 s/topic on numpy), so each organ arm is
roughly single-digit minutes and the two `--with-integrated` mouth loads add the fixed Qwen build; a total on the
order of ~10-20 minutes is expected (dominated by the emergent-assembly build + the two mouth loads), not hours. If
VRAM contention with the concurrent GPU campaign is a concern, drop `--with-integrated` to measure the organ-scale
VRAM/latency delta alone (the fixed mouth baseline can be measured once separately), or reduce `--n-mem`.

## Honest scope / what this does NOT claim

- It measures RESOURCE cost, not conversation QUALITY — the quality experiment is a separate, later step.
- The live D5 EPISODIC recall organ is already n_ca3=2000; the crisp 400-vs-2000 knob is the IDEATION organ's
  on-substrate stand-in-vs-production scale. The probe measures that same dendritic-dAP CA3 store family, which is
  the shared mechanism, so the VRAM/latency numbers transfer to the episodic organ's 2000 build — but the "400"
  arm is the de-risk stand-in scale, not a currently-live episodic setting. An accurate map beats a forced knob.
- The full 400-vs-2000 numbers are PENDING the queued GPU run; this document reports only the BUILT + smoke-
  validated instrument and the characterization, and cites no un-run result.
