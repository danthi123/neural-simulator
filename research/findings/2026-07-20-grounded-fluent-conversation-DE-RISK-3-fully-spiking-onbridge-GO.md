# Grounded fluent conversation — DE-RISK 3 (fully-spiking on-bridge) GO: the grounded answer is produced ON SPIKES

**Date:** 2026-07-20 · **Status:** DE-RISK 3 GO — the format-fine-tuned WKV renders the grounded answer ON the
spiking substrate (RF-phase spiking input, and the fully-synaptic no-host-read path), at parity with the off-bridge
numpy answer. The north-star "a brain you COMMUNICATE with" is de-risked end-to-end. NO `sim/` edit.

## The rung

De-risk 0/1/2 established: the spiking WKV is a drop-in grounded-fluent renderer for the fluid console (focused-grounded
0.83, RA-faithful 1.00, gate-first moat verified) — but the WKV forward was run off-bridge (numpy, the CPU-portable
reference). De-risk 3 = run it ON the bridge with gap#1's spiking input, so the fluency is produced ON spikes.

## GO — on-bridge parity across grounded frames (RF-phase + fully-synaptic)

`_emerge_wkv_onbridge_derisk.py --ssm <grounded_ft.npz> --ssm-state --use-ssm-readout --rf-phase-encode` renders the
grounded frame (`the A v3 P <ans>`) on a real `SimulationBridge`: each token's value is delivered through the PHASE of
an RF (resonate-and-fire) spike (gap#1's spiking input), charging the graded `cp_ssm_state`; the SSM's own trained
read-out picks the next word. Greedy (temp 0), seed 42:

| frame | on-bridge render (RF-phase spiking input) | matches off-bridge? |
|---|---|---|
| the dog eats meat `<ans>` | the dog eats meat **`<eos>`** | ✓ (== "the dog eats meat") |
| the fox chases rabbit `<ans>` | the fox chases rabbit **`<eos>`** | ✓ |
| the bee makes honey `<ans>` | the bee makes honey **`<eos>`** | ✓ |

Every on-bridge answer matches the off-bridge numpy answer EXACTLY, with `<eos>` firing at the correct position (the
copy skill + the eos stop both survive the spiking delivery). The runner generates a fixed token budget so a trailing
token appears after `<eos>`; the answer span (before `<eos>`) is the exact grounded restatement.

**FULLY-SYNAPTIC (no host read) — GO:** with `--rf-synaptic --rf-period 500` (the RF spike drives a real slow-NMDA
conductance synapse; the value is read from `g_nmda`, NO host `rf_read_phases` — gap#1's RUNG-4 full-parity path), the
same frame renders `the dog eats meat <eos>` (calib corr 1.0000). ⇒ the grounded fluent answer is produced with a
**fully-synaptic spiking input** — no host read anywhere in the encode.

## Read-out — the north-star ladder is de-risked end-to-end

- **⇒ grounded fluent conversation, fluency produced on the spiking substrate:** type a question → the brain
  (comprehension + grounded-fact retrieval + gate-first no-confab moat) → the answer is rendered as fluent grounded
  prose by the spiking WKV cortex (RF-phase / fully-synaptic input, the SSM's own trained read-out), at parity with
  the off-bridge reference. The ~21M ANN scaffold is retired for the render path.
- **The honest "fully spiking" bar (SpikeGPT-class, stated plainly):** spiking I/O (RF-phase / fully-synaptic input +
  the WKV's own trained head over the substrate) + a graded slow-conductance recurrent state (`cp_ssm_state`) — the
  "value = trained read-out over a fixed spiking substrate" convergence gap#1 names. Not "every op a spike"; do not
  overclaim.
- **Composition note (honest):** the console (De-risk 1) runs the numpy-parity WKV forward (bit-identical to on-bridge,
  CPU-portable) alongside the numpy RF composer; De-risk 3 proves the SAME answer is producible on the cupy spiking
  bridge. A single-process co-execution of the numpy composer + the cupy on-bridge WKV is the one-backend-per-process
  consolidation (the EMERGE-70/71 pattern) — a firming follow-on, not a capability gap.
- **Scope (honest):** the grounded frame inventory is the bounded curriculum SVO (De-risk 4 = open/rich multi-fact
  prose is the field wall, managed by render-per-fact + VERIFY). The format fine-tune is single-seed (seed 42); a
  multi-seed fine-tune firming is the follow-on (the EMERGE-57 pattern). The on-bridge RF-phase parity itself is
  gap#1-6-seed-validated; the render is greedy-deterministic given the checkpoint.

Runner: `_emerge_wkv_onbridge_derisk.py` (`--rf-phase-encode` / `--rf-synaptic`). Ckpt:
`bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz`.
