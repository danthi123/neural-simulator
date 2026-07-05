# Biologizing the reservoir→role read-out — a LEARNED (delta-rule) read-out resolves the degraded seed 44 + removes the host ridge shortcut (2026-07-04)

**One-line:** the seed-fragile read-out (CYCLE 918/919) had a residual NON-biological shortcut hiding in plain sight — the
read-out matrix `Ws` was a HOST RIDGE FIT (`np.linalg.solve`). A research gate reframed the fix: replace it with a
per-role **DELTA RULE learned ON the spiking substrate**. De-risk: seed 44 (host-fit 11/18, the degraded draw NO
fixed-circuit mechanism could crack) reaches **18/18**, and the scrambled-label anti-cheat is **0/18** (the learning is
genuinely role-specific). It scaled with training (E4→12/18, E12→18/18 — under-trained, not ceiling-limited). The 6-seed
blind generalization test (the headline claim) is in flight. Strictly CPU/numpy; **NO `sim/` edit**.

## Why the host ridge fit was the problem (measurement-grounded, CYCLE 919)

The read-out reproduces `argmax_r((f·Ws)[r])` on the spiking substrate. `Ws` is fit by a HOST ridge solve on the reservoir
RATE feature. The B-1c arc removed the read *step* shortcut (host `f@Ws`→synapses, host argmax→neural argmax), but the
*weights* stayed host-learned. Measured failure: the host-fit `Ws` delivered as synapses is SEED-FRAGILE — 18/18 on the
dev draws 42/43, but 11/18 on seed 44 and **7/9/5 out of 18 on the unseen 100/101/102** (near chance). Root cause (a
research gate verified by measurement): a **train/deploy objective MISMATCH**. The ridge minimizes `‖f·Ws − Y‖²` (a linear
reconstruction of a rate matrix), but deployment runs a spiking WTA whose winner is set by IGNITION ORDER — a
threshold-nonlinear, dynamics-dependent quantity the ridge never sees. So `Ws` is correct for the linear surrogate and only
COINCIDENTALLY correct for the spiking argmax on the dev draws; on an unseen draw the margin lands on the wrong side of the
WTA ignition inversion. (NOT a sub-1% margin, NOT a degraded feature, NOT the dendritic frontier — all refuted: DRIVE-WRONG
=0/18, isolated ens f-I monotone to 450 pA.)

## The mechanism — a per-role delta rule learned on the spiking substrate

Per training sentence + content slot k (the reservoir is FROZEN; only the `res2ens` synapses are plastic):
```
drive the frozen reservoir → ρ = reservoir firing,  a = ACTUAL ensemble firing (via run_with_ens, the REAL spiking read)
error_r = T_r − a_norm_r                 (T = one-hot on the KNOWN slot-k role label — environmental supervision)
W_k[r, :] += η · error_r · ρ             (Widrow-Hoff / cerebellar PF→Purkinje form; clip ≥ 0, Dale-legal excitatory)
```
The learned `W_k` ARE the read-out (delivered as `res2ens` synapses) — NO host `np.linalg.solve`, NO host `f@Ws`, NO host
argmax (winner = the neural argmax over the ensembles' firing). Three load-bearing properties, each grounded in the
project's own track record:
- **PER-ROLE-LOCAL error, not global scalar.** `(T_r − a_r)` is computed independently at each of the 3 role ensembles —
  the "per-region/per-role error" credit-assignment that passed **3/3** (supervised gradient) where a global DA scalar
  FAILED (sign-only 1/6, magnitude 0/6; `2026-05-05-W-to-A-VERDICT`). Same architecture, only the credit rule differs.
- **`a` is the REAL spiking ensemble firing** — so the f-I nonlinearity + the WTA ignition-order are INSIDE the error
  term. The rule doesn't reproduce a host matmul; it drives the correct ensemble to WIN THE SPIKING COMPETITION on THIS
  draw. Swap the draw, re-run the same local rule, it re-finds the winning weights → **generalizes by construction** (the
  project's own learned-cortex thesis: "learn to read whatever messy code arrives"; Gilra-Gerstner FOLLOW 2017).
- **Rate-Hebbian / delta, NOT spike-timing STDP** — the reservoir-feature × per-role-error co-activation is symmetric
  (Δt≈0), exactly where STDP is measured-NEGATIVE (`2026-06-15-on-bridge-hebbian-co-occurrence`, 656k events / 0 Δw).
- **FREEZE the reservoir** (training the recurrence HURTS: 0.25 vs 0.90, `_fork2_predesign`); the `res2ens` synapses are
  the only plastic pathway. Precedent: the project's own scratch prototype (fixed reservoir + local delta-rule read-out)
  scored **1.000** (`_fork2_predesign_local_credit_prototype.py`).

## Results (de-risk, seed 44 = the degraded draw; CPU/numpy)

| read-out | seed 44 |
|---|---|
| host ridge fit (committed) | 11/18 |
| learned delta rule, E4 (under-trained) | 12/18 |
| **learned delta rule, E12** | **18/18** |
| learned delta rule, E12, SCRAMBLED labels (anti-cheat) | **0/18** (must fail — it does) |

⇒ the mechanism WORKS (learning is real + role-specific: scrambled-label collapses to 0), it SCALES with training (the
E4→E12 climb proves under-training, not a ceiling), and it RESOLVES the degraded seed 44 that host-fit + ~20 fixed-circuit
read mechanisms could not.

## Honest scope / in flight
- **THE headline (6-seed blind generalization) is RUNNING**: the SAME fixed protocol (E12/η0.05/N35, no per-subset tune)
  on 42/43/44 + the unseen 100/101/102 — the GO bar is ≥17/18 on ALL six (host-fit: 18/18/11, 7/9/5). If it holds on the
  unseen seeds, this is a generalizing BIOLOGICAL surpass that also retires the host ridge shortcut.
- Anti-cheats: scrambled-label (shown 0/18); + to run: syn-readout lesion collapses, global-scalar-control near chance,
  source-clean (no ridge/`f@Ws`/host argmax in the learn+select path).
- Promotion: `--mode c3` in `_rungB1c_spiking_reservoir_synaptic_readout_derisk.py` (add `_learn_Ws_spiking` replacing
  `_fit_Ws_spiking`; leave c1/c2 verbatim). Scratchpad: `research/findings/raw/signed_conductance/step5_learned_readout.py`.
- Aligned with the project master goals: everything on the ONE spiking substrate, LEARNED (no host shortcut), the
  learns-and-grows artificial-life direction.
