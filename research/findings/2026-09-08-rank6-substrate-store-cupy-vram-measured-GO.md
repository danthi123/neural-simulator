---
type: finding
status: live
date: 2026-09-08
verdict: GO — the substrate-store's cupy VRAM at the real 78,857-fact wikidata_100k scale is ~2.5 GiB, ~20 GiB free on the 24GB 3090; no VRAM-reduction fallback needed; #211's VRAM blocker cleared.
mechanism: knowledge-core-substrate-write
lane: scaffold-retirement
board: scaffold_retirement_backlog#6, vikunja#211
runner: research/runners/_rank6_substrate_store_cupy_vram_derisk.py
seeds: [42]
seed-waiver: this is a resource/structural measurement (a cupy memory-pool byte count, a deterministic
  function of N_facts x per-fact array shapes), not a stochastic capability claim to average over seeds --
  matching this lane's own established convention (the cited 2026-09-05 RSS finding waives its own
  memory/time cost probe the identical way, seed 42 only).
artifacts:
  - research/findings/raw/_rank6_substrate_store_cupy_vram_derisk/full_run.json
  - research/findings/raw/_rank6_substrate_store_cupy_vram_derisk/smoke.json
---

# RANK-6 production-flip's cupy VRAM gate is MEASURED: the substrate-store adds ~2.5 GiB at 78,857-fact scale, comfortably inside the 24GB 3090 budget — the flip's VRAM question, never actually asked before this file, is answered GO

## Verdict

**GO.** `enable_substrate_store=True`'s marginal GPU VRAM cost at the real, actually-deployed 78,857-fact
`wikidata_100k` knowledge core is **measured (cupy memory-pool accounting, not estimated): ~33.2-33.5 KB/fact,
projecting to ~2.5 GiB** — well under the plan doc's own 3.8-5GB *estimate*, and a small fraction of the 24 GiB
single-consumer-RTX-3090 budget this project holds itself to. Combined with the current production
spiking-substrate's own documented worst-case VRAM footprint (~1.34 GiB, `gates/consumer_hardware_reference.py`,
8x-safety-margined, already including the spiking-mouth pools) and the deployed mouth readout's own zero-VRAM
footprint (`webapp/wkv_mouth_generator.py` is pure numpy), the flip lands with **~20 GiB of the 24 GiB budget
still free** even under worst-case brain accounting, or **~18.5 GiB free measured against this box's own actual
observed idle VRAM** (~21 GiB). **None of the plan doc's VRAM-reduction fallbacks (#1 lazy-shard, #2 numpy
worker) are needed to clear this gate.** This resolves Vikunja #211's one remaining named blocker ("the one
remaining check is confirming its memory footprint fits a single 24GB graphics card").

## Why this measurement had never actually been taken (state contradiction, resolved before writing code)

Two documents disagreed. `docs/plans/2026-09-07-rank6-substrate-store-vram-reduction-options.md` says the cupy
VRAM measurement is "queued" and only *estimates* 3.8-5GB. `research/coordination/scaffold_retirement_backlog.md`'s
rank-6 STATUS UPDATE (2026-09-05) reports "MEASURED (not estimated) marginal memory cost: 50.26 KB/fact ...
3.78 GB at the full 78,857-fact core -- affordable." Reading the cited source
(`research/findings/2026-09-05-rank6-knowledge-core-substrate-write-scaled-derisk-mixed.md`, section (b)) in full
shows the 3.78 GB figure is **peak host RSS** (`resource.getrusage(...).ru_maxrss`), produced by
`research/runners/_rank6_knowledge_core_substrate_write_derisk.py`, whose `main()` calls
`assert_backend("numpy", ...)` **unconditionally** (line 405) -- that runner cannot run on cupy at all, by
construction, and never has. Host RAM (this box has ~46GB) and GPU VRAM (the 24GB 3090) are different, physically
separate resources; the backlog's "affordable" verdict answered the RAM question, never the gate's actual VRAM
question. **The plan doc's "queued" was the correct read; the backlog's STATUS UPDATE conflated the two
resources.** No committed artifact anywhere in the repo (searched all rank-6/substrate-store findings) held a
cupy VRAM reading for the substrate-store at any knowledge-core scale before this file.

**Bonus correction surfaced while re-reading `webapp/server.py`'s bundle-resolution code for this task:**
`wikidata_100k` (78,857 facts) — not the 15k core — has been the actual **shipped default LTM bundle since
2026-09-02** (`_default_ltm_bundle_dir()`, board #108: `bundle_order = ["wikidata_100k", "wikidata_core_15k"]`).
So this gate checks the scale that is actually live in production today, not a hypothetical future one. Also
confirmed by direct grep: `enable_substrate_store`/`BRAIN_LTM_SUBSTRATE_STORE` do not appear anywhere in this
checkout's `webapp/server.py` or `webapp/developed_brain_io.py` — the flip genuinely lives only on the
`research/substrate-store-flip` branch Vikunja #211 names, not on `main`, consistent with #211's framing
("approved, pending...") over board #192's own looser "now RETIRED by default" phrasing for the same mechanism.

**The other half of the verify-first check: the "structural gap."** `ShardedPhasorStore.save()`'s
`TypeError: cannot pickle 'mappingproxy'` (named in the 2026-09-05 finding's section (c)) is **already fixed and
merged to `main`** (`fede9d596`, confirmed an ancestor of this checkout's HEAD via
`git merge-base --is-ancestor`), per `research/findings/2026-09-05-rank6-shardedphasorstore-pickle-fix-GO.md`
(GO, regression-tested, a real save/load round-trip re-confirmed against the actual 78,857-fact bundle). It does
not block this flip.

## Method: reuse the existing store path, swap the resource read from RSS to cupy VRAM

New file `research/runners/_rank6_substrate_store_cupy_vram_derisk.py` (additive, no `sim/` edit, no new store
mechanism). It imports `build_store`/`load_real_facts`/`vocab_of` **directly from the existing RSS derisk runner**
and runs the identical `ShardedPhasorStore`/`RFPhasorComposer` construction under `SIM_BACKEND=cupy` instead of
numpy, at the SAME checkpoints (N=500/2,000/8,000) the RSS runner used, for direct comparability. Same
subprocess-per-variant isolation (CUDA needs `spawn`, not `fork`) and same "baseline reading taken AFTER the
fixed CUDA-context+codebook cost, grow ONE store to the largest checkpoint" design as the RSS runner's own
earned methodology, to avoid the identical proxy-dominates trap that runner's docstring documents.

**Instrument finding, earned mid-run (smoke pass, N=10/30/60 first, per this project's own de-risk-cheap-first
discipline):** whole-device `nvidia-smi memory.used` is **too noisy to use directly on this shared, actively-used
box.** The real-scale run's own numbers show why: the numpy-kb (baseline) arm's `nvidia-smi` reading jumped from
2,936 MiB (N=2,000) to 10,816 MiB (N=8,000) — an apparent +8GB — while that SAME arm's own cupy memory-pool
`total_bytes()` stayed at 5.655 MiB across both checkpoints (unchanged). Since the baseline arm never persists
GPU state per fact (`_encode`'s composite comes back via `np.asarray()` on a cupy array, landing as a plain HOST
array — confirmed by reading `_resonate`'s return path), an 8GB jump cannot be this process's own allocation; it
is an **external process on the shared card** (this box runs a live GPU-queue campaign + desktop/other
processes) whose memory footprint changed during this run's ~90-minute wall-clock window. The substrate arm's
`nvidia-smi` reading shows the mirror-image artifact: a **-7,267 MiB** "marginal" between N=2,000 and N=8,000
checkpoints (i.e. the SAME external process releasing memory mid-run). Both are reported in the raw artifact,
not hidden, and the `nvidia_smi` projection in the JSON (a nonsensical -77 GiB) is flagged here explicitly as
**instrument noise, not a measurement** — the fix, used for the headline number, is **cupy's own
`get_default_memory_pool().used_bytes()/total_bytes()`**, which reflects only THIS process's own allocations and
is immune to what else is running on the card.

## The measurement

cupy memory-pool `total_bytes()` (per-process, immune to other GPU tenants), seed 42, real curated
`wikidata_100k` facts:

| N facts | numpy-kb (baseline) pool total | substrate-store (candidate) pool total |
|---|---|---|
| 500   | 3.417 MiB  | 19.526 MiB |
| 2,000 | 5.655 MiB  | 70.253 MiB |
| 8,000 | 5.655 MiB  | 264.580 MiB |

**Baseline (numpy-kb) is flat from N=2,000 to N=8,000** — the per-op resonate bridges are cached by neuron-count
(`RFPhasorComposer._bridge_cache`) and reused across facts of the same shape, so 6,000 additional facts add zero
net pool growth. This is the cupy-VRAM analogue of the RSS finding's own "0 KB/fact measured" baseline result,
now independently confirmed on the GPU-memory axis. Marginal slope N=500->8,000: 0.000298 MiB/fact -> 0.023 GiB
projected at 78,857 facts — negligible, unlike the below.

**Substrate-store (candidate) grows monotonically and near-linearly**, matching the plan doc's own diagnosis
("`_store_substrate` builds a full `SimulationBridge` PER FACT... `ShardedPhasorStore` does NOT pool them"). Three
independent pairwise slopes across the checkpoint range agree within ~4%: N=500->2,000 gives 33.8 KB/fact,
N=2,000->8,000 gives 32.4 KB/fact, N=500->8,000 (the RSS runner's own largest-vs-smallest convention) gives 32.7
KB/fact. **Projected at the real deployed 78,857-fact scale: 2.49-2.52 GiB**, reported here as **~2.5 GiB**.

**Independent cross-check (a completely different method, converging on the same number).**
`gates/consumer_hardware_reference.py`'s own static itemization (`PER_NEURON_BYTES=200`, `PER_SYNAPSE_BYTES=64`,
no safety multiplier — the raw itemized cost, not its 8x-margined worst case) applied to `_store_substrate`'s
actual per-fact bridge shape (confirmed by reading `rf_phasor_composer.py:1461-1474`: a `1+D=129`-neuron bridge
with exactly `D=128` synapses, `conns = [(1+k, 0, ...) for k in range(D)]`) predicts **129*200 + 128*64 = 33,992
bytes/fact = 33.2 KB/fact** — within ~2% of the empirically measured 32.7-33.8 KB/fact band, from a formula that
was written for an unrelated purpose (a different gate, itemized from `sim/bridge.py`'s array declarations, not
tuned to this result). Two independent methods agreeing this closely is strong evidence the ~2.5 GiB projection
is real, not a measurement artifact.

**nvidia-smi peaks, for completeness (whole-device, includes the external-contention noise documented above,
so read as an upper-bound sanity check, not a clean reading):** peaked at 11,271 MiB during the substrate arm's
N=8,000 build — but per the analysis above this reflects the external process's own footprint stacked on top of
this job's, not this job's own marginal cost. No VRAM-related crash or OOM occurred at any point in either arm's
build to N=8,000, a real (if secondary) practical signal that genuine cupy execution at this scale is stable.

## Headroom vs the 24GB 3090 budget

- **Card:** RTX 3090, 24,576 MiB (24 GiB) total (`nvidia-smi`, this session).
- **Current production spiking-substrate, worst case:** ~1.344 GiB <!--derived--> (`gates/consumer_hardware_reference.py`'s own
  live estimate at N=7,002 neurons, 8x safety-margined; this N already includes the ledger's documented spiking-
  mouth-adjacent pools, so no separate mouth line is added on top of it).
- **Deployed default mouth readout:** `webapp/wkv_mouth_generator.py` is pure numpy (confirmed via
  `docs/plans/2026-09-07-mouth-next-token-scale-launch-plan.md`'s own "Deployability" section) — 0 additional
  VRAM at inference for the path actually shipped today.
- **This file's measured substrate-store addition:** ~2.5 GiB.
- **Worst-case brain-only accounting:** 24 - 1.344 - 2.5 = **~20.16 GiB still free** (84% of the budget), i.e. <!--derived-->
  the flip alone would use roughly 16% of a 3090's VRAM.
- **Against this box's own actually-observed idle free VRAM this session** (20.7-21.4 GiB free at multiple
  `nvidia-smi` checks before/after this run, netting out whatever else genuinely runs on this shared machine):
  **~18.5-18.9 GiB would remain free** after the flip — still comfortably clears the separate off-bridge
  Qwen-0.5B open-ended-generation path's own independent ">12GB free" precondition
  (`2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`), with several GB of margin, for the
  (uncommon) case that path is concurrently resident.

## Scope / what this file does and does not settle

This is a **VRAM-only** gate closure. It does not re-run or change the 2026-09-05 finding's own already-6/6-GO
recall+moat parity claim, does not touch the query-time-cost residual (`_can_batch_scan()` still requires
`enable_substrate_store=False`, so the candidate path still falls back to a per-fact loop at query time — a
separate, already-disclosed cost Vikunja #211 itself already treats as accepted, not gating), and does not touch
curation selection (the deeper, separate residual both cited findings name and explicitly do not attempt). It
edits no `sim/` file and flips no production default; `enable_substrate_store` remains `False` everywhere on
`main`. The decision to merge `research/substrate-store-flip` is the owner's, per Vikunja #211's own framing
("approved, pending...") — this file supplies the one measurement that framing was waiting on.

## Reproduce

```
# smoke (fast sanity pass, ~1 min):
tools/gpu_queue.sh add 'cd <checkout> && SIM_BACKEND=cupy <venv>/bin/python -u -m \
    research.runners._rank6_substrate_store_cupy_vram_derisk --smoke \
    --out research/findings/raw/_rank6_substrate_store_cupy_vram_derisk/smoke.json'

# real scale (~90 min, both arms to N=8,000 -- MUST go through the GPU queue, never run directly):
tools/gpu_queue.sh add 'cd <checkout> && SIM_BACKEND=cupy <venv>/bin/python -u -m \
    research.runners._rank6_substrate_store_cupy_vram_derisk \
    --cost-points 500 2000 8000 --target-n 78857 \
    --out research/findings/raw/_rank6_substrate_store_cupy_vram_derisk/full_run.json'
```
