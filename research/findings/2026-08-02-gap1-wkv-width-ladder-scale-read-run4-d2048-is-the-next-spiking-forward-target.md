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
  - research/findings/raw/wkv_spiking_forward/run4_rf_6seed.json
  - research/findings/raw/wkv_spiking_forward/run4_rf_6seed.json.prov.json
---

# gap#1 WKV scale ladder: width buys fluency, and run4 d2048 clears the six-seed RF spiking-forward gate

<!--derived-->
**One-line verdict.** The WKV scale ladder now has a clean CPU-side read from existing training logs: at the matched
1.179B-token point, validation perplexity improves monotonically with width on the identical run3 token shard
(`d1024` 65.89 -> `d1536` 61.03 -> `d2048` 57.14). The mature `run4_d2048` checkpoint reaches best validation NLL
3.8213 / ppl 45.66 at 6.988B tokens, clearly ahead of `run3_d1024` best NLL 3.987 / ppl 53.89. This supports the
board's current "scale the WKV cortex" frontier. These training-log measurements are captured in
`research/findings/raw/wkv_spiking_forward/wkv_scale_ladder_run4_d2048_summary.json`. After the unsandboxed relaunch,
the 267M run4 RF spiking-forward gate cleared the full six-seed promotion on the local RTX 3090: mean ppl_ratio
0.9999999974, mean logit-fidelity Spearman 0.99999999997, max RF read error 7.66e-6. This promotes the 267M/d2048 WKV
checkpoint as RF-spiking-forward faithful under the current parity test.

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

## Run4 RF spiking-forward six-seed promotion

<!--derived-->
The six-seed promotion completed on the local RTX 3090:

```bash
.venv/bin/python -u -m research.runners._wkv_spiking_forward_derisk --mode full \
  --ckpt bridges/lmtrain/run4_d2048/ckpt/best.pt --backend rf-bridge \
  --seeds 42 43 44 100 101 102 --n-windows 16 --nsteps 8 \
  --block-size 256 --n-logit-pos 16 \
  --out research/findings/raw/wkv_spiking_forward/run4_rf_6seed.json
```

<!--derived-->
Result: **GO 6/6**.

<!--derived-->
| metric | result |
|---|---:|
| checkpoint params | 266.98M |
| seeds | 6 |
| windows per seed | 16 |
| mean ppl ratio | 0.9999999974 |
| mean logit-fidelity Spearman | 0.99999999997 |
| max RF read error | 7.66e-6 |
| elapsed | 9166.4 s |

<!--derived-->
Per seed:

| seed | ANN ppl | RF-spiking ppl | ppl ratio | logit Spearman | RF max read error |
|---:|---:|---:|---:|---:|---:|
| 42 | 31.481553 | 31.481553 | 1.0000000010 | 0.999999999969 | 7.60e-6 |
| 43 | 39.232326 | 39.232325 | 0.9999999897 | 0.999999999971 | 7.47e-6 |
| 44 | 35.538272 | 35.538272 | 1.0000000066 | 0.999999999972 | 7.33e-6 |
| 100 | 40.171848 | 40.171848 | 0.9999999966 | 0.999999999971 | 7.42e-6 |
| 101 | 30.749478 | 30.749478 | 0.9999999902 | 0.999999999971 | 7.47e-6 |
| 102 | 35.441583 | 35.441583 | 1.0000000007 | 0.999999999971 | 7.66e-6 |

## Honest scope

<!--derived-->
This finding claims RF-spiking-forward parity for the mature 267M/d2048 WKV checkpoint under the current readout
fidelity test. It does not claim the language system is grounded, conversationally sufficient, or learned by a
biological local rule. The checkpoint is still trained by conventional sequence training; the RF bridge result says the
forward computation can be carried through the RF/spiking read path without measurable degradation.

The next language-frontier step is not another RF parity repeat. It is to use this larger faithful language-circuit
scaffold inside the grounded speech-action plan while continuing to burn down corpus-training and host-side phrasing
scaffolds.
