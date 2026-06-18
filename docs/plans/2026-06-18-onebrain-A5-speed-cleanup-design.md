# A5 — make the OneBrainComposer speed-competitive, then make it the default + retire the legacy numpy runtime

**Date:** 2026-06-18. **Status:** design (deep-research-first prep). **Owner of execution:** a FRESH-FOCUS session with a
QUIET GPU — A5 is the deepest core-engine edit of the one-brain arc (a masked-megakernel `sim/` change with a
bit-identity gate), and the project's practice is to do those with fresh focus, not a marathon-tail rush (the same
reasoning that deferred the CYCLE-153 CUDA-graph refactor).

## Goal (the owner's end-state, memory `project_one_brain_integrated_pipeline_and_cleanup`)

Phase 2 (the real one brain) is DONE + feature-complete (the `OneBrainComposer` runs the whole who/what turn on one
persistent bridge, wired into `BrainConversationalAgent(composer_kind="onebrain")`, multi-seed GO, CI-guarded). A5 is
the final phase: (1) make the one-brain path **speed-competitive** with the numpy production default, (2) make it a
documented option / the default, (3) once competitive, **retire the legacy numpy production runtime** — KEEPING numpy as
the **test oracle** (the cleanup rule: numpy out of production, numpy stays in tests).

## The latency problem (why onebrain is slow today)

The `OneBrainComposer` query is a **cue-matching scan**: for each of K stored blocks, reconstruct the composite (fire its
trigger, one ~208-step resonate) then unbind+cleanup (another ~208-step resonate). So a who/what query is **O(K) blocks ×
~2 resonate windows × ~208 steps each**, and each resonate step issues ~15 CuPy kernel launches in a Python loop (the
launch-bound cost the latency profile measured, `2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`: the
resonate loop is ~98% of an op's cost). The rf production default is the speed reference (numpy codes + small per-op
bridges + the batched scan + the adopted megakernel).

**Measure first (the A5 session's step 0):** run `_phaseB_onebrain_latency_probe.py --k 8` to quantify onebrain vs rf
ms/query. The structural gap is known: O(K) reconstruct-per-block × ~2 un-fused 208-step resonate windows per block, vs
the rf path's batched scan + adopted megakernel. The probe re-runs after each lever to track the closing gap.

## The levers (cheap-first; each individually verifiable, answer-identical, the moat preserved)

**Lever 1 — BATCHED SCAN (no `sim/` edit; do FIRST) — DE-RISKED GO (2026-06-18).** Replace reconstruct-per-block with
ONE resonate over a block-diagonal layout: fire ALL K triggers at once → the K readout blocks reconstruct in parallel →
a resident block-diagonal unbind (each block's roles, tiled) → block-diagonal cleanup → read all K×roles. **Result
(`_phaseB_onebrain_batched_scan_derisk.py`, 6/6 at K=8): == the per-block loop == ground truth (answer-identical), 7.3×
faster** (~350 → ~48 ms/fact). 7.3× > the 5.6× onebrain-vs-rf gap, so this lever ALONE makes the one-brain composer
competitive with (≈ faster than) the rf reference. **Integration note:** the unbind + cleanup conns are FIXED (the role
phasors + the codebook don't change per query — only the store/trigger conns change per fact), so the production
integration should PRECOMPUTE the fixed unbind/cleanup conns lists once (avoid the per-query Python list-build, which is
the O(K·V·D) cost that would otherwise offset the win at large K) and the query just fires the triggers + reads. The
per-block scan stays the correctness oracle behind a flag.

**Lever 2 — INDEXED STORE (host-side routing; optional).** A cue→block index so a query reconstructs only the candidate
block(s), O(K)→O(1) reconstructs. NOTE: the FHRR-faithful form is the superposition-search (fire-all, lever 1), so lever
1 likely subsumes this; keep lever 2 as a fallback for very large K (or shard, the validated 320-concept route).

**Lever 3 — MASKED-MEGAKERNEL (a flagged, default-preserving `sim/` edit; do if lever 1 is insufficient).** The adopted
RF megakernel (`enable_rf_cudagraph`, one CUDA `RawKernel`/step) fuses the resonate loop to 1 launch/step — but it BAILS
to the Python loop when a `neuron_mask` is set (`bridge.py:5565-5567`), and the co-resident one-brain bridge ALWAYS masks
(the RF ops are sliced). So A5 needs a **masked-megakernel path**: the fused kernel honors the RF neuron mask (write
`v`/`u` + trackers only for masked neurons — the same masking `_rf_advance_one` already does, ported into the kernel).
ADDITIVE + default-preserving (mask=None → current behaviour, bit-identical); GPU-only. **Gate:** bit-identity to the
masked loop (`tests/test_rf_*` golden + a co-resident golden) + answer-identical end-to-end + a clean speedup. Also flagged
by the production scope (risk C): mask the whole-array RF spike-tracker re-init (`bridge.py:5481-5484`) so a per-op kick
on one register doesn't reset a co-resident register's trackers.

**Stacking:** lever 1 (K×) × lever 3 (the ~17-24× per-resonate megakernel win, on the masked path) plausibly reaches the
profile's real-time target (~tens of ms/turn). Measure after each lever; stop when competitive.

## The cleanup phase (after speed-competitive)

1. Make `composer_kind="onebrain"` a documented option (README / CLAUDE.md / the webapp if surfaced); consider making it
   the default once the speed + the full capability suite (incl. the richer caps + multi-turn) are at rf parity.
2. **Deprecate-then-retire the legacy numpy production runtime** (NOT big-bang): the per-op host orchestration, the
   dual numpy/spiking branches, the per-op bridge caches, the legacy rate composer, the reference-only standalone phasor
   sims. KEEP the numpy versions as the **test oracle** (the correctness-provability the spiking path is validated
   against). The deepest cleanup is architectural — the persistent substrate replaces the orchestration layer; the host
   does only I/O.

## Honest risks / open questions for the A5 session

- Lever 1's block-diagonal reconstruct must stay phase-coherent across K blocks resonating together (the multi-window
  settle is proven to ~5 ops; K-block parallel reconstruct is new — gate on answer-identity, fall back to a per-block
  micro-schedule or shard).
- The masked-megakernel `sim/` edit is the deepest/riskiest piece (a custom CUDA kernel honoring a mask) — byte-review it,
  gate on `tests/test_rf_*` bit-identity, default-off-preserving, revertable by flag/git-revert.
- The richer caps (`render_fact`/`query_chain`) + multi-turn + reconsolidation on the persistent bridge should be at rf
  parity BEFORE onebrain becomes the default (so nothing regresses for users).

## First concrete A5 action (for the fresh session)

Build lever 1 (the batched scan, no `sim/` edit) as a `OneBrainComposer` method + an answer-identical de-risk (== the
per-block scan + the moat, 3 seeds × 2 D) + a re-run of the latency probe to quantify the win. If still not competitive,
scope+build the masked-megakernel (lever 3). Reuse-by-import where possible; byte-review any `sim/` edit; numpy stays the
test oracle.
