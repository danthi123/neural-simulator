---
type: finding
status: live
lane: own-voice-fluency
date: 2026-09-20
---

# Scaling NO-GO confirmed 3-seed: raw scale + data do not reach fluency at d384 (2026-09-20)

Confirms the decisive single-seed result (`2026-09-18-scaling-go-no-go-raw-scale-falsified-lever-is-data-do-not-buy-hardware-yet.md`,
seed 42). Seeds 43 + 44 (extended d384, FineWeb-Edu, token points 0.96M/1.92M/3.84M/7.68M passages, 12 epochs) both
land the same verdict.

## Result (artifact: research/findings/raw/_gen_cortex_token_supply_extended_d384_s43_44.json)

**verdict = NO-GO-CAPACITY-SATURATED** (n_seeds=2: 43, 44; + seed 42 from the prior run = 3-seed).
- mean_delta_nll_min_to_max_tokens = **0.0026** — NLL is FLAT across an 8x increase in training tokens (0.96M→7.68M
  passages): more data does not help at this capacity.
- mean_top_point_wkv_deep_nll = **3.8834** — above the fluency band [3.0, 3.69] (still sub-fluent at the top point).
- max_tok_per_active_param_reached = **173.122** — far past the Chinchilla ~20 compute-optimal ratio; the model is
  data-saturated, not data-starved.
- n_still_descending_at_top = 0 / n_margin_grows_with_tokens = 0 — no seed is still improving with more tokens.
- (It does beat trigram + use context at the top point — it learns SOMETHING — but plateaus below fluency.)

## Verdict

The 2026-09-19 owner-ratified arc conclusion holds robustly (3 seeds): the spiking mouth's broad-domain fluency is
NOT reachable by raw scale (params/tokens) at d384 — capacity-saturated. **Do not buy hardware to chase fluency this
way.** Consistent with the permanent-Qwen-mouth decision (the mouth was never the bottleneck).

## Honest bounds
- d384 / vocab-2000 / max-len-48 specific; a much larger d_model is a different (untested, and per the arc,
  deprioritized) point — the arc's decision is that this whole raw-scale axis is the wrong lever, not that one
  d_model failed.
- CPU/GPU per the run's provenance sidecar; deterministic per-seed.

## Next
None on this axis — the arc is closed (falsified). The forward work is the load-bearing / faculty-drive metric
(the brain deciding under the permanent mouth), not the mouth's from-scratch fluency.
