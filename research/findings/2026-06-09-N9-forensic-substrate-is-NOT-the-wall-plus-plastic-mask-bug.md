# N9 forensic — the substrate is NOT the wall (deployed bridge bootstraps the critic on CPU); + a separate real `cp_synapse_plastic_mask` bug

**Date:** 2026-06-09
**Type:** systematic-debugging forensic (CPU/numpy, read-only `sim/`) + controller-reproduced verification.
**Supersedes / corrects:** `2026-06-09-N9-warmup-deployed-nav-NEGATIVE.md` — its "substrate limit / the MSN critic can't fire in the deployed 47-region bridge" **conclusion is RETRACTED**. The deployed substrate DOES fire and bootstrap the critic.

## Headline

The premise behind the N9 negative — "the afferent→critic pathway delivers ~20× less current in the
deployed bridge; the critic plateaus at −79.6 mV vs −71 mV in isolation; it's a substrate wall" —
is **FALSIFIED on the CPU path**. The deployed nav bridge (`build_bg_brain_regions`, full flag set,
WITH Gabor visual pre-init) is **current-identical to the isolation probe** and **bootstraps the
critic to robust firing**. The 1800-step CuPy negative is therefore **NOT** a substrate/wiring/
current property. It is a **CuPy-path divergence (or a measurement artifact in the original CuPy
run)**, pending one short GPU confirmation.

## Decisive measured evidence (both bridges, identical warm-up drive, `SIM_BACKEND=numpy`, seed 42)

Controller-reproduced via `research/findings/raw/_n9_critic_current_diag.py`:

| Measure (critic `striosome_value`) | ISOLATION probe | DEPLOYED nav bridge (+ Gabor) |
|---|---|---|
| afferent→critic CSR slice | n=5981, w≈0.2009 | **n=8062, w=0.2005, byte-identical BEFORE vs AFTER Gabor growth** |
| critic g_e (last-half) | 0.0786 | 0.0852 (ratio 0.92) |
| critic membrane V (mean) | −69.40 mV | −69.06 mV (≈ equal; **both ~−69, no −79.6 plateau**) |
| afferent firing | 15.3 Hz | 15.5 Hz (ratio 0.98) |
| **value-leads-reward train (40 trials, deployed knobs, all-goals)** | 0.20 → 0.34 (bootstrapped) | **0.2005 → 3.3117 (bootstrapped; w=0.71 by trial 20; crit 0→39.7 Hz)** |

So the Gabor growth does **not** misalign the afferent→critic synapses (count/weights/targets
unchanged); the afferent fires at the same rate; the critic gets the same g_e and reaches the same
membrane V; and the deployed substrate **bootstraps the critic monotonically** under the warm-up
protocol. The earlier "−71 vs −79.6 mV, afferent 2× weaker" forensic does not reproduce and was a
measurement artifact (stale state / wrong critic slice in that earlier probe).

## What this does to the N9 negative

The committed `…-N9-warmup-deployed-nav-NEGATIVE.md` concluded a **substrate limit**. That is wrong.
The substrate fires the critic. The real situation: the **deployed CuPy 1800-step run** reported the
critic weight byte-frozen at 0.20078 with `striov_rate_log` all-zero, but **CPU with the same
protocol grows it to 3.31 (≥0.71 by 20 trials)**. The discrepancy is CuPy-vs-CPU, the one variable
the CPU forensic cannot eliminate (no GPU available; owner on machine).

**The decisive next step (short, GPU):** run the actual `--critic-warmup-trials` on **CuPy** with
`WARMUP_DEBUG=1` (the `g11_bg_runner.py:4390` hook prints per-trial `crit_spk` / DA / weight). Two
outcomes:
- weight grows on CuPy too → the smoke's "frozen 0.20078" was a **measurement artifact** (likely
  `_mean_critic_weight()` reading the wrong CSR slice on the CuPy/COO layout) → N9 may already work;
  re-measure + run the A/B.
- weight stays 0 on CuPy → a genuine **CuPy numeric/UB divergence** in the warm-up path → debug that.

Either way **N9 is recoverable, not a substrate boundary.**

## Separate, genuine bug pinned (independent of N9): stale `cp_synapse_plastic_mask`

`cp_synapse_plastic_mask` is built once at wiring (`sim/bridge.py:2212`) and **never resized after
`apply_v1_gabor_weights` grows `cp_connections`** (the grep confirms it appears in NO grow/rebuild
path: not in `_grow_synapse_arrays_if_needed` `:726-787`, not in the `set_pathway_weights` CSR-rebuild
`:2732-2765`). It is then read **unguarded** in the STDP update:

```
sim/bridge.py:5957   plastic_here = self.cp_synapse_plastic_mask[stdp_active_indices]
```

where `stdp_active_indices` index the **grown** CSR (up to nnz−1, e.g. 241208) but the mask is stale
(e.g. len 174132). When a visual-stream cell fires in the nav loop (retina/v1/v2/it own the
beyond-mask rows), the gather is **out of bounds** → on **CuPy** this is silent UB (garbage booleans
→ corrupted `cp.where` → mis-frozen/mis-updated visual-cortex STDP); on **numpy** it raises
`IndexError`, caught by the per-step blanket `except` (`:6516`) → `stop_simulation()` → the rest of
that step (incl. reward modulation) silently skipped. **Same bug class** as the gate under-sizing
fix (`512026ee`) and the GABA_B-mask fix (`6f73b5f0`). The sibling reads at the same site already
route through `_ensure_gate_capacity` (e.g. `cp_plasticity_rate_gain` at `:5971` reads a grown array).

**This does NOT explain the N9 freeze** (the warm-up drives only `vs_place_context` + `snc`, both
in-bounds, so the OOB is never triggered during warm-up). It is a real, separate fidelity bug
corrupting **visual-cortex plasticity** during the deployed nav on CuPy (nav stays ~sane because the
BG cascade is unaffected — which is why the smoke produced 2.136 while silently corrupting visual STDP).

### Proposed minimal fix (PROTECTED `sim/` edit — flagged for owner byte-review; NOT made)

One line at the use site, the exact analogue of the existing gate guards (additive; byte-identical
when no growth has occurred, since `_ensure_gate_capacity` returns the array unchanged when
`arr.shape[0] >= n`):

```python
# sim/bridge.py:5956-5957  (inside the `if self.cp_synapse_plastic_mask is not None:` block)
-   if self.cp_synapse_plastic_mask is not None:
-       plastic_here = self.cp_synapse_plastic_mask[stdp_active_indices]
+   if self.cp_synapse_plastic_mask is not None:
+       plastic_here = self._ensure_gate_capacity(
+           "cp_synapse_plastic_mask", self.cp_connections.nnz,
+           fill=False, dtype=cp.bool_)[stdp_active_indices]
```

New (Gabor-grown) entries default `False` = **non-plastic**, the correct default (Gabor synapses are
fixed-weight). Byte-identical when no growth occurred. Anti-cheat: under the flagship NEURAL nav,
(a) nav score preserved (≈2.136 final-quarter); (b) with `bridge.strict_step_errors=True` the
per-step blanket-`except` no longer trips on a CuPy nav run (surfaces any remaining OOB).

## Status

- `sim/` byte-empty; no edits made; no commits by the forensic agent.
- Diagnostic harness: `research/findings/raw/_n9_critic_current_diag.py` (reproduces every number
  above on CPU; controller-rerun confirmed).
- **Decision fork (owner steer):** (1) approve the one-line `cp_synapse_plastic_mask` byte-review fix
  (a real fidelity bug, lands independent of N9); (2) the N9 confirmation needs a short CuPy
  `WARMUP_DEBUG` run (GPU — owner's go / when off the machine) to localize the freeze (artifact vs
  CuPy divergence). N9 is recoverable either way.
