# Nav close-out R1-a — spiking-default nav decision + SC-orienting — GO (2026-06-27)

The deployed merged-gate/demo nav CLI default is flipped from the host-argmax oracle (`--readout-source motor` + host SC tie-break) to the **SPIKING decision** (`spiking_wta` + `urgency 180` + the CYCLE-1B levers: `sel_recurrent_weight 0.3`, `n_sel/commit 40`) **+ spiking SC-orienting** (`--spiking-sc` default-ON with the Burndown-3F validated config: popvector + FIX1 stochastic tie-break + log-polar foveal render). Host (`--readout-source motor --no-spiking-sc`) is retained as the EXPLICIT oracle. ⇒ the shipped nav benchmark runs fully-spiking-on-one-brain by default. Reuse-by-import / runner-default flip; **NO `sim/` edit** (edits to `research/runners/_nav_gate_merged_run.py` + `research/runners/g11_bg_runner.py`).

## Result (grid-32 / 1800 steps, the merged-gate runner; controller-run benchmark)
| run | score (Σ per-phase final-quarter mean distance, lower=better) |
|---|---|
| spiking-default seed 42 | 5.39 |
| spiking-default seed 43 | 5.52 |
| host-oracle seed 42 | 2.86 |

- The spiking-default **NAVIGATES**: ~5.4 cells from goal ≪ the ~21 random-walk baseline; tracks every goal change, reaches within ~1 cell (confirmed in-episode).
- **Honest cost: 1.91×** the host-oracle (5.46 spiking mean / 2.86 oracle) — the brain-based-only deliverable (the SC-orienting on spikes). Better than the ~2.4× the nav-loop research gate predicted; R2 (the SC opponent-axis, FIX3) is the documented margin-SNR remedy to shrink it further.
- The host-oracle flag reproduces the documented benchmark (2.86) unchanged.

## Moat + safety
- The conversational no-confab MOAT is UNAFFECTED: `tests/test_nav_conv_merged_agent.py` + `tests/test_nav_conv_step2b_coresident.py` = **29 passed / 1 xfailed** (pre-existing) — the nav decision is array-disjoint from the parser/composer by construction.

## Honest scope
The 1.91× IS the honest spiking cost (per BRAIN-BASED-ONLY, the cost is the deliverable, not a fail). The nav decision was already the LIBRARY default (CYCLE 1B, 2026-06-19); R1-a flips the deployed merged-gate CLI + makes the spiking-SC the default, retiring the host orienting from the default path while keeping it as the oracle. Process note: the benchmark was run by the controller after the dispatching subagent stalled on a background-GPU-wait (a subagent cannot resume on bg-completion).

## Reproduce
```bash
# spiking-default (the new default): just the bare invocation
SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run --seed 42 --grid-size 32 --n-steps 1800 --out spiking.json
# host oracle (retained benchmark-reproduction path):
SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run --seed 42 --grid-size 32 --n-steps 1800 --readout-source motor --no-spiking-sc --out oracle.json
```
Raw: `research/findings/raw/_navcloseout_R1a_spiking_s42.json`, `_spiking_s43.json`, `_oracle_s42.json`.
