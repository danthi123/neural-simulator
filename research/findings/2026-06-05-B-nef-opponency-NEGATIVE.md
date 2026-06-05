# De-risk (B) — gated NEF signed-value integrator for the opponency → NEGATIVE (but lifts the aggregate read 0.41→0.90) — 2026-06-05

**Verdict: NEGATIVE on the GATE (per-seed unbind recovery == 1.000), with a genuine partial win worth recording.**
The documented next-idea after the in-network-opponency NEGATIVE — a **gated NEF integrator** that represents the
signed value `s = bon − boff` in a population and decodes it linearly (the subtraction happens in the *represented
value* before the lossy fire-rate read, Eliasmith-Anderson NEF) — was built and de-risked. It **substantially lifts
the aggregate signed read** (signed cosine **0.41 → 0.90** vs the simple-accumulator de-risk) but **does NOT reach
unbind parity**: per-role recovery is **0.077** (≈1/13, near-chance) at the production proj_dim=800. The owner's
explicit choice ("Build the NEF integrator") is executed; the result confirms the parallel research synthesis's SNR
prediction (`2026-06-05-spiking-opponency-literature-synthesis.md`): the NEF helps the *aggregate* signed value but
stays SNR-limited for the *per-role* unbind. Pivot is to the research's **Option B (bipolar threshold)** structural fix.

## The crux (the GATE)
After (A) cleared the readout shortcut and (B) cleared the store shortcut, the only remaining numpy in the compute
path is `bind_fact`'s superposition + `onoff(bon − boff)` opponency. The prior de-risk
(`2026-06-05-B-innetwork-superposition-NEGATIVE.md`) showed the **opponency** is the blocker: `bon,boff` are
strongly correlated (cos 0.89, a large common mode), so the signed difference is small, and a rate-coded read of two
separately-summed non-negative channels recovers the signed vector at only **cos 0.41**. The documented next idea
was the NEF integrator: drive a population with `(A+B) − (C+D)` so the subtraction is in the represented value, then
NEF-decode `s' = Ddec · a(s)` (linear decode, sidestepping the divisive-shunting wall). **GATE: per-seed recovery of
the NEF-built bound vector's unbind == the numpy bound vector's unbind == 1.000.**

## Result — NEGATIVE on the GATE, positive on the aggregate read
`research/findings/raw/_b_nef_opponency_probe.py`, seed 42, proj_dim=800, numpy deterministic cleanup held constant
(isolates the opponency), CuPy / RTX 3090. NEF operating point `M=8000, src_drive=500, w_nef=40, E_inh=−80,
gains 1–3, bias 20–80`:

| M (encoder neurons) | recovery (NEF unbind == numpy) | **signed cosine** `s'·(bon−boff)` | recon cosine |
|---|---|---|---|
| 2000 | 0.077 | **0.9031** | 0.9130 |
| 8000 | 0.077 | **0.9026** | 0.9129 |
| 16000 | 0.077 | **0.9038** | 0.9140 |
| seed-42 eval | **0.077** (GATE: 1.000) | **0.9027** | 0.9128 |

vs the simple-accumulator de-risk: signed cosine **0.41**. The NEF **more than doubles** the aggregate signed-read
fidelity (0.41 → 0.90). But the GATE (per-role unbind parity) fails decisively (0.077 vs 1.000).

## Why aggregate 0.90 does NOT give per-role unbind parity (the diagnosis)
Two compounding causes, both consistent with the research synthesis's SNR argument:
1. **Per-role superposition SNR.** The bound vector superposes 3–4 `role⊗filler` bindings; the unbind extracts ONE.
   An aggregate signed cosine of 0.90 means a ~0.19 residual is spread across the bundle — but each role's component
   is only ~1/3–1/4 of the bundle norm, so the *weakest* superposed component the unbind must read sits at or below
   that residual. Aggregate-0.90 ≠ per-component-0.90.
2. **Bias-dominated encoder (`smell_zero_collapse = 1.824 > 1`).** The zero-input encoder mean rate is *higher* than
   the live-input mean rate — i.e. the NEF neurons fire mostly from their bias, and the signed input is a small
   modulation on a large baseline. The offline decoder `Ddec` still recovers the aggregate `s` (cos 0.90), but the
   per-dimension input SNR is low, so the rectify-then-unbind of the weak per-role components fails.
3. **M-invariance is the tell.** Signed cosine is **flat at 0.903 across M=2000/8000/16000** (8× neuron range). If
   this were an averaging-N wall, more neurons would lift it (error ∝ 1/√N). It does not → the wall is
   **representational/calibration**, not sample-count. Confirms the research's "still SNR-limited at heart" call on
   Option C.

## Honest scope / what this does and doesn't change
- **NEGATIVE on the GATE.** The NEF integrator does not make `bind_fact`'s opponency spiking at unbind parity. The
  superposition + opponency therefore **stay numpy** (the disclosed boundary, n=111, is unchanged).
- **Both DEEP shortcuts stay CLEARED.** (A) cleanup (NEF) and (B) store (Crawford weight-store) remain GO and
  unaffected — this is the *third*, linear-glue piece, the one that does not clear.
- **A genuine partial result is recorded, not hidden:** the NEF integrator lifts the aggregate signed read 0.41→0.90.
  A future mechanism that only needs the *aggregate* signed value (not per-role unbind) could use it. The per-role
  unbind is the unmet bar.
- **GPU-only** (the composer's spiking bind is degenerate on the numpy backend); CuPy / RTX 3090.
- **NO `sim/` edits.** `build_bind_bridge` / `CoreSimComposer` reused by import; the NEF bridge is standalone.

## Next (the research-directed pivot)
Per `2026-06-05-spiking-opponency-literature-synthesis.md`, the NEF (Option C) being SNR-limited is the predicted
outcome, and the structural fix is **Option B (bipolar threshold / MAP-B)**: don't represent the value as a small
*graded* difference of correlated rates — collapse it to a per-dimension **sign** via an ON/OFF winner-take-all
(biology's push-pull as a *decision*; cancels the common mode). Cheap-first numpy test queued: does binarizing the
bound vector (`sign(bon − boff)`) still unbind at numpy parity? If yes → build the spiking per-dim WTA. If binarize
kills the VSA → assess Option A (FHRR phasor pivot, surface to owner) vs Option D (honest SNR boundary). Either way,
both deep shortcuts remain cleared and the disclosed linear-glue boundary is unchanged.

## Artifacts
- Probe: `research/findings/raw/_b_nef_opponency_probe.py` (NEF encode/decode + the M-sweep + signed-cosine +
  smell-test diagnostics; reuses `CoreSimComposer` / `numpy_raw_superposition` by import)
- Result: `research/findings/raw/_b_nef_opponency.json` (seed 42, M-sweep)
- Backend: CuPy / RTX 3090. NO `sim/` edits.
