# 🎉🎉🎉 BRIDGE CO-RESIDENCE DEMONSTRATED — the full 494M spiking Qwen faculty RUNS on the SimulationBridge, LOCAL, bit-exact, coherent (the 'one brain' north star for language, feasibility achieved) (2026-06-23)

**The full 24-layer Qwen2.5-0.5B (all decoder layers + embedding + final RMSNorm + tied lm_head) runs end-to-end on
the live SimulationBridge RF substrate: VRAM 14.05 GB resident (FITS < 24 GB → LOCAL, no cloud), on-bridge ppl 7.041
EXACTLY matches the B-1 off-bridge spiking forward (the per-layer graded-SEM does NOT compound over 24 layers —
RF-vs-B1 logit cos 1.0, argmax-agree 1.0), and the spiking generation is coherent + byte-identical to B-1 ("Once upon
a time, in a land far, far away, there was a"). ⇒ the spiking fluent faculty is CO-RESIDENT on the brain's substrate =
the 'one brain' north star for language, feasibility DEMONSTRATED. NO `sim/` edit (reuse-by-import of the C1 RF
exact-matvec + the de-risk #2 layer + the B-1 graded ops).** `research/runners/_bridge_cores_fullfwd_derisk.py`, the
live bridge on the RTX 3090.

## The de-risk ladder (all GO)
| # | test | result |
|---|---|---|
| 1 | one q_proj on RF | bit-exact (max-err 4.58e-7) |
| 2 | one full decoder layer | cos 1.0 vs B-1, no error accumulation |
| 3 | **full 24-layer forward** | **ppl 7.041 == B-1, coherent gen, 14 GB LOCAL** |

## Result
- **VRAM: 14.05 GB resident** (FITS << 24 GB → LOCAL). Storage = complex CSR (`cp_rf_w_re` + all-zero `cp_rf_w_im`,
  float64 + int32), built VECTORIZED (bit-identical to `rf_set_complex_weights`, `max|·|=0.0` — milliseconds vs the
  intractable 136M-iteration Python list-comp the lm_head would otherwise need; lm_head install 2.33s). The peak
  during install/compare touched ~24 GB; the steady-state resident is 14 GB.
- **ppl 7.041 = B-1's 7.041** (the SEM does NOT compound over 24 layers; vs ANN 5.872 on the 159-token slice).
  RF-vs-B1 logit cos 1.0, argmax 1.0; RF-vs-ANN cos 0.989 = the B-1 ceiling exactly.
- **Generation coherent** (byte-identical to B-1, token-agree 1.0): *"Once upon a time, in a land far, far away, there
  was a"*.
- Anti-cheat lesion: row-permute the lm_head RF weights → logits collapse (cos 0.286, argmax 0.000) while the shuffled
  matvec reproduces `a@W_shuf` exactly (max-abs 5.5e-9) → the RF carries the computation.

## The wall = WALL-CLOCK (the perf lever, NOT cloud)
- **0.786 tok/s prefill** (warm, install-cached); **161 s/token generation** (each generated token = a full re-forward
  over the growing context; 16 tokens = 2580 s). The forward is launch-/CSR-gather-bound (the scoping predicted this).
- **LOCAL** (per `feedback_long_local_runs_ok_confirm_cloud_cause`): no VRAM wall (14 GB << 24 GB) → cloud NOT
  triggered. A 2000-token validation corpus (prefill) ETAs **0.7 h → overnight-viable**. The slow path is GENERATION
  (161 s/token); the first move is the LOCAL perf lever (scoping #6: batch the per-row resonate / dense-on-bridge RF
  matvec), NOT cloud (an H100 lifts no VRAM wall — only ~3-5× compute).

## ⇒ the 'one brain' north star for language — feasibility DEMONSTRATED
The spiking fluent faculty is co-resident on the SimulationBridge (the brain's substrate), bit-exact + coherent,
LOCAL. **HONEST SCOPE:** a FEASIBILITY demonstration — it RUNS but is SLOW (the perf lever is the usability follow-on);
the faculty is co-RESIDENT but not yet INTERACTING with the conversational brain (the full functional 'one brain'
integration — the faculty's fluency gated by the brain's grounding, all on one bridge — is the deeper step). NO `sim/`
edit anywhere in the de-risk ladder.
