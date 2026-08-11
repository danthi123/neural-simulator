---
type: finding
status: live
date: 2026-08-11
mechanism: per-parameter-heterogeneity-reseed
lane: integration-7-one-brain
seed-waiver: this is a determinism/byte-identity result verified by exact sha256 hash compare and np.array_equal (bit-for-bit), not a stochastic capability claim — seed-count generalisation does not apply to an exact-compare invariant.
---

# Per-parameter heterogeneity reseed — the one-brain append-LAST seam is invariant for the Izhikevich params (additive, default-off `sim/` flag)

Artifact: `research/findings/raw/2026-08-11-per-param-heterogeneity-reseed-verification.json`

## The problem (one-brain merge, INTEGRATION #7 burn-down #1)

The one-brain merge appends the e-prop neuron slices LAST to the conversational bridge. The merge is byte-identical for most parameters, but NOT for the Izhikevich per-neuron heterogeneity. `sim/bridge.py::_apply_parameter_heterogeneity` re-seeds the RNG ONCE (`cp.random.seed(het_seed)`) and then draws each per-neuron parameter (`a`, `b`, `d`, `C`) SEQUENTIALLY from that one stream as `cp.random.<dist>(size=n)`. Appending N neurons grows every `size=n` draw by N, which shifts the stream position for every parameter AFTER the first — so the first-`n_pre` (pre-existing) values of every parameter except the first change. That flips a near-tie arbiter on 2/14 chat turns, so the merged bridge is not byte-identical to the pre-merge bridge for those turns.

## The mechanism (why appending LAST is the trigger)

A single seeded stream is positional: the i-th parameter's draw begins at the offset left by all earlier `size=n` draws. Grow `n` and every downstream offset moves, so the pre-existing prefix drifts. The fix is to give each parameter its OWN reseeded substream drawn from position 0. Then a parameter's first `n_pre` values are a pure prefix of its `size=n` draw, and the CuPy RNG has the prefix property: a freshly-seeded `cp.random.normal/lognormal(size=n)` and `size=n_pre` agree on their first `n_pre` values (verified directly on this 3090/CuPy: max abs diff 0.0 for both normal and lognormal). Independent substreams make the pre-existing values invariant to how many neurons are appended LAST.

## The change (additive, default-OFF, guarded)

- `sim/config.py`: new field `per_parameter_heterogeneity_seed: bool = False` (in the B2 heterogeneity block). Default OFF preserves the legacy single-stream draw.
- `sim/bridge.py::_apply_parameter_heterogeneity`: read `per_param_seed = bool(getattr(cfg, "per_parameter_heterogeneity_seed", False))`; define stride `_HET_PER_PARAM_SEED_STRIDE = 1_000_003` (a large prime, so adjacent parameters' seeds do not neighbour-correlate); enumerate the distributions; and, when the flag is ON, reseed before each drawn parameter — CuPy path `cp.random.seed(het_seed + i*STRIDE)`, backend-neutral path a fresh `_backend_neutral_random_state(resolved_het_seed + i*STRIDE, ...)`. The `het_seed` derivation (`cfg.heterogeneity_seed>=0 else cfg.seed`), the draw ORDER, the clip, the per-region `cp.where` mask, and the end-of-method RNG-state restore are all unchanged. `i` is the stable enumerate index of the distributions dict, so it does not depend on `n`.

Scope note: the flag governs ONLY the heterogeneity parameters drawn in this method (`izh_a/b/C/d`). `cp_neuron_firing_thresholds` are drawn by a SEPARATE path (`cp.random.uniform(size=n)` in the model-init section, outside this method) and are deliberately not altered here — that seam has its own handling in the merge (the current merge byte-identity check is threshold-hash-only). This flag extends the byte-identity guarantee to cover the heterogeneity parameters too.

## Verification (all from the cited artifact JSON)

1. DEFAULT-OFF byte-identity. With the flag omitted/False, a fresh build hashes `cp_neuron_firing_thresholds` + every `cp_izh_*` array to values identical to a pre-edit (826b1dfd) build — the exact sha256 compare matches for all ten arrays (`default_off_byte_identical_vs_pre_edit: true`; e.g. `cp_izh_a` = a5d0581b563be19d, `cp_izh_b` = fafa43167b1d339c on both). The determinism suite `tests/test_determinism.py` reports 9 passed, 0 failed. The flag is inert when off.

2. FLAG-ON append-LAST invariance. Build at n=200, then build at n=280 (80 neurons appended LAST) with the flag ON: the pre-existing 200 neurons' `cp_izh_a/b/d/C` are byte-identical to the n=200 build — `np.array_equal` prefix compare is True with max abs diff 0.0 for all four parameters (`flagon_append_last_all_invariant: true`). This is the whole point: appending LAST no longer perturbs the pre-existing per-neuron parameters.

3. FLAG-ON per-seed determinism. Two flag-ON builds at seed 42 are array-identical for `izh_a/b/d/C`; a seed-7 build differs — a valid, reproducible seeded substrate (`flagon_determinism_ok: true`).

4. Blast radius (flag ON vs OFF, same seed, no append). Only `izh_b/d/C` change; `izh_a` is unchanged because it is the index-0 parameter whose reseed `het_seed + 0*STRIDE` equals the legacy single reseed. `cp_neuron_firing_thresholds` (drawn before), `cp_recovery_variable_u` (computed from the base `b` before jitter), and `cp_ou_current` (drawn after; the end-of-method RNG restore erases the per-parameter reseeds) are all unchanged. The flag's effect is confined exactly to the heterogeneity parameters.

## Status

This is the production enabler for the one-brain merge: with `per_parameter_heterogeneity_seed=True` the merged bridge's Izhikevich heterogeneity becomes append-LAST invariant, so the near-tie arbiter no longer flips on those 2/14 turns. Additive and default-off — no existing run changes. Biology-neutral: the heterogeneity distributions (Marder & Goaillard 2006; Tripathy et al. 2013) are unchanged; only the RNG substream structure changes, and drawing each parameter from an independent substream is arguably more correct.
