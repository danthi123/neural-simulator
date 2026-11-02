---
type: finding
status: resolved
date: 2026-10-31
lane: Vikunja #203 (a synapse marked non-plastic still drifts from use) -- the pi lane
mechanism: the 2026-09-02 fix (`enforce_plastic_mask_in_hebbian` / `BRAIN_ENFORCE_PLASTIC_MASK`, merge 509c7137b)
  gated the HEBBIAN LTP/decay/clip sites only. A grep of every write to `cp_connections.data` in `sim/bridge.py`
  found six further sites that move a synapse declared `plastic=False` / `plastic_internal=False`, all now gated
  behind a NEW additive default-OFF flag `cfg.enforce_plastic_mask_beyond_hebbian` (same env umbrella
  `BRAIN_ENFORCE_PLASTIC_MASK`). No default changed; STDP/BDSP/BTSP already enforced the mask and are untouched.
seeds: [42, 43, 44, 100, 101, 102]
verdict: VERIFIED BOTH DIRECTIONS on 6 seeds x 6 rules. Flag OFF reproduces the drift (positive control, every
  site); flag ON makes every declared-frozen weight bit-unchanged across 30 successive reads while the companion
  plastic population still moves. Flag OFF is SHA-identical to the genuine pre-fix code (36/36 comparisons).
  On the production comprehension organ the flag changes NOTHING at the reply level (6 seeds x 2 arms, 25 battery
  items, every item digest identical) -- so the freeze is real at the write sites and inert in that organ.
  NO capability claim, no default flip: flipping the default is owner-gated (blast radius below).
runner: research.runners.pi203_frozen_synapse_battery (CPU/numpy, 6 neurons, 30 reads x 20 steps per run)
second runner: research.runners.pi203_comprehension_no_regression (the production comprehension organ, CPU/numpy,
  12 runs, ~2.5 min each, 4 concurrent)
artifact: research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json (72 runs: 6 rules x 6 seeds x 2 arms)
byte-identity artifact: research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json (36 comparisons,
  36/36 identical) -- runner research.runners.pi203_prefix_comparison
throughput: 72 runs (6 rules x 6 seeds x 2 arms) in 5.21 s wall, numpy/CPU, no GPU (the artifact records backend,
  device, per-run elapsed and a cost projection); 36 x 3 byte-identity comparisons in 41 s wall, same backend
builds_on:
  - research/findings/2026-09-02-read-isolation-audit-29runner-followup-map-plus-a-bigger-adjacent-bug.md
  - research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md (the flip criteria this residual must satisfy before the default changes)
  - merge 509c7137b (`enforce_plastic_mask_in_hebbian`, `tests/test_enforce_plastic_mask.py`)
---

# The residual: the write sites the Hebbian fix never reached (Vikunja #203)

Board #203 asks for the runtime path that still moves a frozen synapse. The Hebbian hole was closed on 2026-09-02;
this is the rest of it. The instrument is one 6-neuron bridge with two populations wired by `inject_explicit_wiring`:

- **frozen**: all-to-all among neurons 0-3, declared `plastic=False`, and deliberately given **no** named
  `plasticity_gate` -- so only the mask can protect it (a named gate at 0 is a different, already-working mechanism);
- **control**: all-to-all among neurons 4-5, `plastic=True`, same starting weight. If the control population does
  not move, the rule never ran and a zero in the frozen column means nothing.

30 successive reads x 20 steps each, numpy backend, `max|w - w0|` reported for each population.
The full battery is `research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json` (72 runs,
6 rules x 6 seeds x 2 arms, 5.21 s wall on CPU, backend + device + per-run elapsed recorded in the artifact).

## The sites, measured (all six seeds; the artifact holds every run)


| Site | Where | OFF (the bug) | ON (the fix) |
|---|---|---|---|
| Homeostatic synaptic scaling | `sim/bridge.py` block "5b", `data[:] = data * post_scales` | frozen 20.0 -> 5.0, delta **15.0** (6/6 seeds) | delta **0.0**; control still 15.0 |
| Structural-plasticity elimination | `data[eliminate_mask] = 0.0` | frozen 0.02 zeroed, delta **0.02** (6/6) | delta **0.0**; control still eliminated |
| Reward-modulation clip | the `else` branch of the C2 clip, `cp.clip(data, w_min, w_max)` with no named gain array | frozen 100.0 clipped to `hebbian_max_weight`=1.0, delta **99.0** (6/6) | delta **0.0**; control still clipped |
| Mask bookkeeping across a rebuild | `_remap_gate_indices_after_rebuild` / `_compact_synapse_arrays` read the attribute `cp_plastic_mask`, which `sim/` never assigns | after `set_pathway_weights(..., add_missing=True)`, **4** declared (pre,post) pairs are labelled with the wrong plasticity (6/6 seeds) | **0** mislabelled; frozen flat, control moves |
| The reward UPDATE itself | `data[:] += weight_updates`, eligibility-gated but mask-ungated | frozen synapse with a stale eligibility trace is written (shown by the `combined` rule: frozen delta **99.99** OFF) | delta **0.0**; control still updated |
| Newly FORMED synapses | `_add_synapses_to_arrays` -- the mask is grown lazily with `fill=False` | synapses created by structural plasticity/rebuild read as FROZEN (0 of 2 labelled plastic) | 2 of 2 labelled plastic, matching the gain default of 1.0 = open gate |
| Deferred CSR COMPACTION | `_compact_synapse_arrays` -- the 2026-07-31 fix compacted five per-synapse arrays and left the mask out | after elimination + compaction, **4** declared pairs label the wrong synapse (6/6 seeds) and the frozen population drifts 5.0 | **0** mislabelled; frozen delta **0.0**, control still moves 5.0 |

The `combined` rule is the integration case: scaling + elimination + reward modulation on ONE bridge at
once. OFF the frozen population drifts 99.99 (6/6 seeds); ON it is bit-flat (6/6) while the control population
moves 99.88.

Eligibility accumulation in the reward block IS mask-gated (`sim/bridge.py` ~11781); only the final clip/write was not.
STDP (~1276/1356), BDSP, BTSP and inhibitory STDP already consult `cp_synapse_plastic_mask` unconditionally, so the
mask column above is complete for every write to `cp_connections.data` in the step loop.

## What was changed

`sim/config.py`: one new field `enforce_plastic_mask_beyond_hebbian: bool = False`.
`sim/bridge.py`: one shared gate helper `_plastic_mask_enforced(cfg, field)` (env `BRAIN_ENFORCE_PLASTIC_MASK` OR the
field), `_hebbian_plastic_mask_enforced` now delegates to it, and `_live_plastic_mask(cfg, field, nnz)` returns the
nnz-sized mask or `None`. Every one of the four sites runs its masked branch only when that returns a mask: scaling
forces scale=1.0 on frozen synapses and skips the homeostatic clip there; elimination removes frozen synapses from the
weak set; the reward clip writes only the plastic entries; the remap/compaction carry the LIVE mask by coordinate
(new synapses default to plastic, mirroring the gain default of 1.0 = open gate).

When the flag is OFF, `_live_plastic_mask` returns None and every new branch is unreachable -- the code path is
literally the old one.

## Verification (both directions, not asserted)

1. **Flag OFF == pre-fix.** The scenario file was run against the genuine pre-fix source (`git archive` of the
   pre-fix commit `710883be5`, extracted into a scratch dir -- `git worktree` fails on this box, FAILURE_LOG
   2026-10-31) with the flag absent: SHAs over `cp_connections.data` + `v` + `u` match the fixed code's OFF arm
   exactly in **36/36** comparisons (6 rules x 6 seeds), recorded in
   `research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json` (each comparison also carries a
   repeat OFF run of the current tree, so identity is not a single lucky draw). `tests/test_plastic_mask_beyond_hebbian.py`
   re-derives the same comparison live from git history (no hard-coded SHA).
2. **Flag ON freezes.** 6 rules x 6 seeds: frozen max-delta exactly 0.0 in all 36 ON runs, the control population
   moves in all 36, spikes present in all 72 (no undefined comparison). OFF-arm drifts: scaling 15.0, structural
   0.02, reward 99.0, rebuild 15.01-15.04, compaction 5.0, combined 99.99 -- identical on every seed.
3. **The instrument can see the misalignment it claims to fix**: 4 mislabelled pairs OFF after a rebuild, 4 OFF
   after deferred compaction, 0 ON in both cases.

Commands (verbatim, from the repo root):

```
CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy PYTHONPATH=. python -m pytest tests/test_plastic_mask_beyond_hebbian.py -q
CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy PYTHONPATH=. python -m research.runners.pi203_frozen_synapse_battery
CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy PYTHONPATH=. python -m research.runners.pi203_prefix_comparison
CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy PYTHONPATH=. python tests/_plastic_mask_beyond_hebbian_scenario.py scaling --enforce
```

A second-opinion pass on the diff (MiMo) flagged three length hazards, all fixed in the shipped code: the
compaction branch never truncated the mask (`_compact_synapse_arrays` now assigns the compacted slice), the
CSR-rebuild remap could boolean-index a mask shorter than the pre-rebuild synapse list (both the mask and the
rate-gain remap now clamp to `min(len(arr), len(old_positions))`, the idiom the neighbouring STP/eligibility remap
already uses), and `fill=True` in the formation path would have un-frozen existing synapses sitting between the
mask's lazy length and the new ones (now `fill=False` plus an explicit `True` on the new range only). The
remap/compaction is gated on the NEW flag: a run with `enforce_plastic_mask_in_hebbian` ON and this flag OFF keeps
the pre-existing mask-vs-reorder misalignment -- no regression, but both flags must be ON for the mask to be
trustworthy across a re-ordering.

## The reply-level arm (lane E · Language): does the freeze change the REPLY?

A write-site freeze that never reaches the reply is a correctness fix, not a faculty. So the flag was also run
against the organ the original board finding named: the production comprehension organ, 6 seeds x 2 arms
(flag OFF vs `BRAIN_ENFORCE_PLASTIC_MASK=1`), CPU/numpy, fresh process per arm (the organ and its learned lexicons
are process-global singletons keyed by the first build), 25 battery items per arm
(HAND_COVERED / HELD_NOUN / HELD_VERB / JOINT / MOAT).

**Result: zero differences.** Every item's SHA-256 digest matches across the two arms on all six seeds
(`research/findings/raw/_pi203_frozen_synapse_residual/comprehension_no_regression.json`, 12 runs, ~2.5 min each,
4 concurrent, backend/device recorded, provenance sidecar stamped). Read as a negative, not a comfort: in that
organ none of the six gated sites is reached in a way that survives to the read, so the flag is a no-op there.
It does NOT mean the drift is harmless -- the drift battery (above) is what shows the weights move, and the
board's 13.8->56.1 read-driven-drift number came from this same organ. The two facts together say the exposed
weights in the comprehension organ are written by paths OTHER than the six sites gated here (or are never read
into the reply), which is the next thing to pin down, not a reason to stop.

## Follow-up (same day): which paths reach the comprehension organ

The reply arm's zero difference is explained by `research/findings/2026-10-31-write-path-attribution-the-comprehension-organ-s-frozen-drift-is-the-hebbian-loop.md`:
instrumenting every write in that organ shows the six sites above never fire there (its structural / reward /
scaling flags are off); its frozen drift is entirely the Hebbian LTP + decay writes, which the umbrella cfg field
did not arm -- that gap is now closed, and the pooled wave3 organ makes no weight writes at all.

## What this does NOT claim

- It is not a capability GO and no faculty is measured here; it is a correctness fix for a declared freeze.
- The default is NOT flipped. `enforce_plastic_mask_in_hebbian` is still default-OFF and the flip criteria in
  `2026-09-24-plastic-mask-flip-PREREGISTRATION.md` (6-seed drift battery + 6-seed load-bearing no-regression) still
  govern the owner decision. Runners that built ad-hoc gate mitigations on the assumption that everything not
  gate-named is plastic will change behavior under this flag.
- Two honest limits of the fix itself: (a) verification is numpy/CPU only -- the branches are backend-agnostic
  (`cp.where`/boolean indexing on already-materialised arrays), but the cupy path is not SHA-tested here;
  (b) with the flag ON, structural elimination can skip the `cp.random.rand(nnz)` draw when the masked weak set is
  empty, so an ON run has a different RNG trajectory from an OFF run by design. The mask is grown lazily with
  `fill=False` at use sites: any synapse added by a path that does not remap it would read as frozen; the two
  formation paths in the step loop do remap.
