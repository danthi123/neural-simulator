# Bridge-coresidence PERF lever (scoping #6): the DENSE-MATVEC lever is bit-exact + crushes the CSR matvec (≈9000× lm_head), BUT once the matvec is cheap the host nonlinearities/attention dominate (97%) → end-to-end 8.8 tok/s, GO_WITH_CAVEAT (2026-06-23)

**De-risk #3 (`2026-06-23-bridge-coresidence-DEMONSTRATED.md`) proved the full 24-layer Qwen2.5-0.5B runs on the live
SimulationBridge RF substrate, LOCAL + bit-exact + coherent, but SLOW: 0.786 tok/s prefill, 161 s/generated token,
"launch-/CSR-gather-bound." This de-risk (scoping #6, the PERF LEVER) profiles the RF forward and tests the
dense-matvec + batch levers. Verdict: GO_WITH_CAVEAT — the dense matvec is the proven, bit-exact FIRST lever, but it
is necessary-not-sufficient; the real end-to-end bottleneck SHIFTS to the host graded nonlinearities + attention.**
`research/runners/_bridge_cores_perf_derisk.py`, RTX 3090. NO `sim/` edit (runner-level measurement).

## (1) PROFILE — where the 161 s/token goes
The per-row CSR-RF matvec is dominated by the **resonate** (the cuSPARSE complex-CSR matvec over a DENSE 494M weight),
NOT the per-op launches:
- gate_proj (896→4864): 3.78 ms/row = resonate 91% / kick 7% / read 2%. `down_proj` (4864→896) is **14.6 ms/row** (4×
  the others — its CSR has 4864 post-rows from the big input dim).
- resonate fit (megakernel): **0.42 ms/step + 0.10 ms fixed/call = only 3% launch overhead** → the resonate is
  **compute/gather-bound on the CSR**, not launch-bound. ⇒ the matvec (inside resonate), not the per-token loop, is the
  dominant cost. The megakernel already fused the per-step launches; the wall is the CSR-on-dense gather.

## (2)+(3) DENSE-MATVEC + BATCH levers — bit-exact, ≈3600–9000× faster per shape
The RF read is `Re(Z)/nsteps = a@W` (verified max-err 1.3e-7 vs `a@W`). Since W is **100% dense** (these Qwen layers
are dense), compute it as a dense cuBLAS GEMM `a@W_dense` instead of the cuSPARSE CSR resonate:

| matvec | CSR-RF per-row (S=32) | dense batched GEMM | speedup | err (csr-vs-batch f32) |
|---|---|---|---|---|
| gate_proj | 121 ms | 0.034 ms | **3600×** | 9.5e-7 |
| down_proj | 470 ms | 0.036 ms | **13000×** | 8.6e-7 |
| **lm_head** | **8012 ms** | **0.89 ms** | **≈9000×** | 3.2e-6 |

- **Bit-faithful:** the dense f64 GEMM is the SAME math as `a@W` (max-err **7.5e-15** = roundoff); the f32 dense GEMM
  and the f32 RF membrane read both approximate `a@W` and agree to **3.2e-6** (f32 precision). Bit-exactness is trivial
  (`a@W == a@W`); the speedup is the deliverable. The sparse CSR was simply the WRONG storage for a dense matrix.

## (4) EXTRAPOLATE + the MEASURED end-to-end (the honest number)
- **Matvec-only projection** (pure dense GEMM, all 169 linears, activations on-GPU): prefill **~7200 tok/s**,
  generation **~330 tok/s** (vs CSR 0.786 / 161 s-per-tok). This is the matvec CEILING.
- **MEASURED end-to-end** (de-risk #3's full forward VERBATIM — real Qwen weights, B-1 graded RMSNorm/SiLU/softmax +
  RoPE + attention — with the per-row CSR-RF matvec swapped for the dense GEMM): **8.8 tok/s** (32 tok in 3647 ms),
  only **11× the CSR baseline**, NOT 330×.
- **Why:** breakdown of the end-to-end forward → **dense linears (incl H↔D) = 108 ms = 3%; the host graded
  nonlinearities + attention + RoPE + the ~216 per-linear device↔host copies = 3540 ms = 97%.** Once the matvec is
  cheap, the bottleneck SHIFTS to the host (numpy) nonlinearities/attention and the per-linear H↔D round-trips.

## Verdict: GO_WITH_CAVEAT — two levers, in order
1. **THE DENSE MATVEC (proven here):** bit-exact, ≈3600–9000× faster than the CSR matvec — the CSR-on-dense gather WAS
   the wall and dense storage IS the ANN GEMM speed. **Necessary first move.**
2. **NECESSARY-NOT-SUFFICIENT:** the real end-to-end win is keeping the **whole forward ON-GPU** — cupy graded
   nonlinearities + an on-GPU attention + no per-linear D→H (only the final logits read). That is where 97% of the
   wall-clock now lives. This is the actual usability work.

## `sim/`-edit recommendation
**No `sim/` edit is REQUIRED for either lever** — both are host-forward changes:
- The dense GEMM is already runner-computable WITHOUT a `sim/` edit (the host forward calls cupy `A @ W_dense` directly,
  bypassing the RF matvec for the dense linears).
- The second lever (on-GPU nonlinearities/attention) is also host-forward (the graded ops ported to cupy).
- An **OPTIONAL** guarded `sim/` edit — a dense-weight RF-matvec mode (`cfg.rf_dense_weights` + a stored dense
  `cp_rf_w_dense`, read in `rf_resonate_steps`/`_rf_advance_one` via GEMM; DEFAULT-OFF = the byte-identical CSR path,
  the composer's sparse O(D) bind/unbind unaffected) — buys the same throughput THROUGH the bridge's own RF read, i.e.
  for on-bridge co-residence *purity*, not for the throughput itself. Precisely scoped, default-off, but not on the
  critical path to usability.

## Cloud
Still **LOCAL** (`feedback_long_local_runs_ok_confirm_cloud_cause`): no VRAM wall (14 GB resident << 24 GB). The
bottleneck is compute/host-overhead, which the on-GPU forward (lever 2) fixes locally; an H100 would only lift compute
~3–5×, not the host-round-trip wall. Cloud NOT triggered.

Artifact: `research/findings/raw/_bridge_cores_perf_derisk.json`.
