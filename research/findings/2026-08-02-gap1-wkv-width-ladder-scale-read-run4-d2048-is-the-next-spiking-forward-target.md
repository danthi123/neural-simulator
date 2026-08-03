---
type: finding
status: contributing
date: 2026-08-02
mechanism: wkv-cortex
artifacts:
  - research/findings/raw/wkv_spiking_forward/wkv_scale_ladder_run4_d2048_summary.json
  - research/findings/raw/wkv_spiking_forward/wkv_scale_ladder_run4_d2048_summary.json.prov.json
  - research/findings/raw/wkv_spiking_forward/run3_rf_6seed.json
  - research/findings/raw/wkv_spiking_forward/run4_rf_2seed_cheap.json
  - research/findings/raw/wkv_spiking_forward/run4_rf_2seed_cheap.json.prov.json
---

# gap#1 WKV scale ladder: width still buys fluency, and run4 d2048 clears the cheap-first RF spiking-forward gate

<!--derived-->
**One-line verdict.** The WKV scale ladder now has a clean CPU-side read from existing training logs: at the matched
1.179B-token point, validation perplexity improves monotonically with width on the identical run3 token shard
(`d1024` 65.89 -> `d1536` 61.03 -> `d2048` 57.14). The mature `run4_d2048` checkpoint reaches best validation NLL
3.8213 / ppl 45.66 at 6.988B tokens, clearly ahead of `run3_d1024` best NLL 3.987 / ppl 53.89. This supports the
board's current "scale the WKV cortex" frontier. These training-log measurements are captured in
`research/findings/raw/wkv_spiking_forward/wkv_scale_ladder_run4_d2048_summary.json`. After the unsandboxed relaunch,
the 267M run4 RF spiking-forward cheap-first gate also cleared 2/2 seeds on the RTX 3090: mean ppl_ratio
0.9999999963, mean logit-fidelity Spearman 0.99999999997, max RF read error < 7.7e-6. The GPU-dependent next is now
the six-seed promotion of the same checkpoint, not another 83M reproduction.

## What changed since the 2026-07-23 launch spec

<!--derived-->
The config-prep artifact treated `run4_d2048` as an early step-2000 checkpoint and warned that the meaningful RF
de-risk should be rerun once the checkpoint converged. That condition is now satisfied enough to act: `run4_d2048`
has reached step 899500 / 7.369B tokens, with best step 853000 / 6.988B tokens and best val NLL 3.8213. The run is no
longer an early wiring-only artifact; it is the strongest available WKV scale checkpoint in the workspace.

<!--derived-->
| run | params | d_model | current tokens | best val NLL | best val ppl | best deep NLL |
|---|---:|---:|---:|---:|---:|---:|
| run3 | 83.17M | 1024 | 7.553B | 3.9870 | 53.89 | 3.9347 |
| run5 | 162.49M | 1536 | 1.180B | 4.0656 | 58.30 | 4.0161 |
| run4 | 266.98M | 2048 | 7.369B | 3.8213 | 45.66 | 3.7630 |

<!--derived-->
Matched at 1.179648B tokens, the width ladder is monotone:

<!--derived-->
| matched token budget | d1024 run3 | d1536 run5 | d2048 run4 |
|---:|---:|---:|---:|
| 1.179648B tokens val ppl | 65.89 | 61.03 | 57.14 |
| 1.179648B tokens deep NLL | 4.1436 | 4.0612 | 3.9923 |

<!--derived-->
At the mature run4 budget, run4 also dominates the same-width-time comparison available from run3:

<!--derived-->
| near 7.368704B tokens | d1024 run3 | d2048 run4 |
|---:|---:|---:|
| val ppl | 55.35 | 47.08 |
| deep NLL | 3.9600 | 3.7899 |

## What is already landed

<!--derived-->
The 83M run3 RF spiking-forward port is already banked and verified: `run3_rf_6seed.json` records mean ppl ratio
0.999999998 and mean logit-fidelity Spearman 0.999999999966 across seeds 42/43/44/100/101/102. The prior seed-43
blowup was fixed as runner-level `id()` reuse cache aliasing, not a substrate limit. A CPU regression guard now pins
that cache discipline (`tests/test_wkv_spiking_forward.py`).

## Run4 RF spiking-forward cheap-first

<!--derived-->
The 267M checkpoint was run through the RF bridge after CUDA access was restored:

```bash
.venv/bin/python -m research.runners._wkv_spiking_forward_derisk --mode full --ckpt bridges/lmtrain/run4_d2048/ckpt/best.pt --backend rf-bridge --seeds 42 43 --n-windows 8 --nsteps 8 --block-size 256 --n-logit-pos 16 --out research/findings/raw/wkv_spiking_forward/run4_rf_2seed_cheap.json
```

<!--derived-->
Result: **GO 2/2**. Seed 42 had ANN ppl 26.810050 vs spiking ppl 26.810050, ppl_ratio
0.9999999984, logit-fidelity Spearman 0.999999999969, RF max read error 7.66e-6. Seed 43 had ANN ppl 38.286565 vs
spiking ppl 38.286564, ppl_ratio 0.9999999941, logit-fidelity Spearman 0.999999999972, RF max read error 7.55e-6.

## Exact next action

<!--derived-->
Promote the existing six-seed command, which is now actively running on the local RTX 3090:

```bash
.venv/bin/python -m research.runners._wkv_spiking_forward_derisk --mode full --ckpt bridges/lmtrain/run4_d2048/ckpt/best.pt --backend rf-bridge --seeds 42 43 44 100 101 102 --n-windows 16 --nsteps 8 --block-size 256 --n-logit-pos 16 --out <run4_rf_6seed_output>
```

<!--derived-->
Resource note: the previous Codex sandbox had no visible NVIDIA driver, but after relaunch `nvidia-smi` and CuPy both
see the RTX 3090 and the RF-bridge runner is using it.

## Honest scope

<!--derived-->
This finding claims only the two-seed cheap-first run4 RF spiking-forward GO. It does not claim six-seed promotion
until the `run4_rf_6seed` artifact exists and validates. The 83M port is already landed; the next real information is
whether the same RF spiking-graded-read conversion holds robustly at 267M/d2048 across six seeds.
