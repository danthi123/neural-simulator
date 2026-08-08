---
type: finding
status: smoke
date: 2026-08-08
mechanism: episodic-cortical-cue-recall
lane: EPISODIC
---

# Episodic B: CA3 recurrent COMPLETION now LOAD-BEARING (within-assembly attractor potentiation) — recall is completion-driven, not feedforward; neural WTA stays an honest negative; pool 6-seed crash fixed (SMOKE, seed 42)

Status: SMOKE (single seed 42; the completion sub-wall CLOSES with a full teeth panel PASSING; the WTA
sub-wall is a sharpened, teeth-backed honest negative). Needs the 6-seed run below before any generalization.
Backend: cupy (RTX 3090). `cfg.seed`-seeded substrate (build-twice threshold hash IDENTICAL, `--verify-seed`).
Runner: `research/runners/_riii_ca3_cortical_episodic_wta_derisk.py`. Artifact:
`research/findings/raw/cortical_episodic_wta/_completion_loadbearing_SMOKE_s42.json`. NO `sim/` edit — all new
mechanism is runner-side region/pathway/weight construction (reuse-by-import); the two new levers are additive
and default-off (proof of byte-identity below).

## The law applied: the prior IGNITION finding banked TWO honest negatives; this closes one of them

The prior smoke (`2026-08-08-episodic-cortical-readout-IGNITION-...`) SURPASSED the Wave-1 silent readout by
readout fan-in, but banked two open sub-walls: (1) the neural WTA was inert (`wta_off_sep == full_sep`) and (2)
CA3 recurrent COMPLETION was inert (`zero_recurrent_winner == full`, `ca3_completion=0.00`) — recall was
FEEDFORWARD heteroassociation from the cued CA3 cells, not recurrent attractor completion. This finding takes a
new method at sub-wall (2) and closes it; sub-wall (1) survives its new method and is re-banked with the
mechanism mapped.

## What moved sub-wall (2): the missing companion process was within-assembly recurrent POTENTIATION

Per CLAUDE.md's wall question ("what does the real circuit run alongside CA3 recurrence that we replaced with a
constant?"): the encode-time plasticity here only ever DEPRESSED cross-assembly ca3→ca3 (the competition step +
`structural_sep` = pattern SEPARATION) and NEVER POTENTIATED the within-assembly recurrents — so an assembly
never became the strong recurrent attractor that Marr/Rolls/Hopfield autoassociative completion requires (a
partial cue completes iff the co-active cells' MUTUAL recurrents are potentiated). Two interacting constants
were the block, isolated by an operating-point sweep (all at n_ca3=1500, assembly 0.10 → 150 cells):

- The **recall dAP coincidence threshold** (`recall_k_thresh`) was 40, far above the within-assembly recurrent
  fan-in a held-out cell receives from a partial cue (~2–4 inputs at density 0.05). Lowering it 40→5 alone moved
  `ca3_completion` only 0.00→0.01 — necessary but not sufficient.
- The **within-assembly recurrent weight was never potentiated** (new lever `attractor_w`) AND the **recurrent
  connectivity was too sparse** (new lever `ca3_density`, 0.05) to put a held-out cell's fan-in above threshold.

Setting `attractor_w=80` (strong within-assembly ca3→ca3), `ca3_density=0.30` (held-out fan-in ≈ cued×density ≈
45×0.30 ≈ 13 > threshold), `recall_k_thresh=5`, and a SMALL cue (`ca3_cue_frac=0.3`, so the cued subset alone
CANNOT ignite the cortical readout and completion of the held-out cells is REQUIRED): the readout ignites via
completion, and zeroing the recurrents makes it SILENT. That is CA3 completion becoming load-bearing.

## Result (SMOKE, seed 42, n_ca3=1500, k=4, assembly 0.10, ca3_cortex_density 1.0, cue 0.3, ca3_density 0.30, attractor_w 80, recall_k_thresh 5)

<!--derived-->
| condition | winner_overall | max_cortex_rate | ca3_completion | reads as |
|---|---|---|---|---|
| full | 0.75 | 0.050 | 0.16 | readout IGNITES via completion; correct WHAT/WHEN wins (chance 0.25) |
| zero_recurrent | 0.25 | 0.000 | 0.00 | ca3→ca3 zeroed → held-out cannot complete → SILENT → chance ✅ COMPLETION LOAD-BEARING |
| permute_cue | 0.00 | 0.050 | 0.00 | wrong-assembly cue → no valid completion → wrong readout (specificity) |
| lesion_real | 0.25 | 0.000 | 0.00 | ablated cue-assembly → silent readout, chance |
| lesion_sham | 0.75 | 0.050 | 0.15 | unrelated ablation → recall PRESERVED |
| untrained | 0.25 | 0.000 | 0.00 | no engram → silent readout, chance |
| wta_off | 0.75 | 0.050 | 0.16 | lateral inhibition off → UNCHANGED (see sub-wall (1) below) |

Attribution (`tools.lab.attributable_to`): completion (full − zero_recurrent) = +0.75 vs +0.25 control, diff
+0.50 → 66.7% attributable to the manipulation (the 33.3% residual is the chance floor 0.25, not leakage).
Verdict (`tools.verdict.Verdict`): preconditions PASS (readout ignites; recall 0.75 > chance 0.25) and every
teeth control passes → GO for ignition+completion. `ca3_completion=0.16` is in the range of the gap#5 CLOSED
attractor's own completion (0.18–0.33), consistent with a lossy cortex→CA3→cortex stack on top of it.

## Sub-wall (1) neural WTA — STILL an honest negative, now with the mechanism mapped

`wta_off` is byte-for-byte identical to `full` (winner 0.75; full sep 0.00729 vs wta_off sep 0.00714); the
attribution gives the WTA 2.0% of the sep, the other 98% present in the control. At this operating point the heteroassociative readout is CLEAN —
only the correct item's cortical cells fire, EVEN WITH THE WTA OFF — so the feedback lateral inhibition has
nothing to suppress. Attempting to manufacture a resolvable competition by raising the baseline (unpotentiated)
CA3→cortex weight FAILS in a diagnostic way: at `ca3_cortex_w`≥8 the dense assembly fan-in drives ALL items
roughly equally (the small potentiated increment is swamped), the winner COLLAPSES to chance, and the single-FS
E%-max WTA — which scales all items down uniformly — cannot break a ~tie toward the correct item (with WTA on,
winner 0.25; off, 0.38 — the inhibition if anything hurts). The tension is structural: an E%-max WTA is
load-bearing only over a graded-but-ORDERED multi-item drive; this readout produces either clean-single (WTA
unneeded) or swamped-uniform (WTA cannot help). The mapped surpass (a scoped next method, not a defer): the
readout must carry a preserved rate-gradient across OVERLAPPING engrams — real memories share features — with
inhibition tuned to the active-item count (de Almeida–Idiart–Lisman sets sparsity by divisive+subtractive
inhibition, which one FS basket does not supply). That is the WTA's own companion process.

## Pool 6-seed crash — ROOT-CAUSED and FIXED

The staged pool run (`--seeds 42 43 / 44 100 / 101 102`, `--out …/episodic_ignition_s*.json`) produced NO
artifacts on the nodes. Cause: the runner's `--seeds` parser did `s.split(",")` only. Space-separated staging
crashed BEFORE any write — `--seeds "42 43"` (one whitespace-joined token) → `int("42 43")` ValueError;
`--seeds 42 43` (two shell words) → argparse "unrecognized arguments: 43" → SystemExit(2). Both reproduced.
Fix: `--seeds` is now `nargs="*"` and `_parse_seeds` re-splits each token on `[,\s]+`, so `42,43` / `42 43` /
`42, 43` all parse identically (unit-checked). (The `episodic_ignition_s*.json` glob was also never a per-seed
template — the runner writes ONE aggregated `--out`; use one process for the 6-seed, command below.)

## Byte-identity of the default-off levers (proven, not asserted)

<!--derived-->
(The numbers in this section are cross-run comparisons quoted from the prior committed IGNITION artifact and a
scratch same-backend re-run, not from this finding's cited artifact — hence marked derived.)

The two new levers are additive and default-off: `attractor_w=None` skips the potentiation block entirely;
`ca3_density=0.05` is the prior literal. Running the runner at the prior committed IGNITION config (cue 0.5, no
attractor, density 0.05) reproduces every DISCRETE decision metric of the committed artifact exactly
(full_winner 0.75, max_cortex 0.0417, zero_recurrent 0.75, ca3_completion 0.00, permute 0.00, lesion_real 0.25,
sham 0.75, untrained 0.25). The only difference is the continuous `sep` margin at the 4th decimal
(0.005655 → 0.005804), which is BACKEND FP: the committed artifact ran on numpy; re-running the PRE-EDIT runner
(`git show HEAD:…`) on cupy gives sep 0.005803571… — identical to the edited runner on cupy. So the code path
is byte-identical when the levers are off; the margin delta is numpy-vs-cupy, not the edit.

## 6-seed command (the generalization gate — run as ONE process, single aggregated artifact)

(A 6-seed run at this config is IN FLIGHT as of this commit; the density-0.30 recurrent field is ~4x the
synapses of the baseline, so each of the 42 condition-runs is ~2-3 min. The aggregate artifact will land at
`_completion_loadbearing_6seed.json` under the same raw dir as the smoke artifact above.)

```bash
OUT=research/findings/raw/cortical_episodic_wta
.venv/bin/python -m research.runners._riii_ca3_cortical_episodic_wta_derisk \
  --seeds 42,43,44,100,101,102 --n-ca3 1500 --k-items 4 --assembly-frac 0.10 --ca3-cortex-density 1.0 \
  --ca3-cue-frac 0.3 --recall-k-thresh 5 --attractor-w 80 --ca3-density 0.30 \
  --out "$OUT"/_completion_loadbearing_6seed.json
```

GO (completion) if, mean over seeds: full ignites (max_cortex > 0) and full_winner > chance AND full_winner >
{zero_recurrent, permute_cue, lesion_real, untrained} AND sham ≈ full. The WTA remains reported as an honest
negative until the overlapping-engram + divisive-inhibition method above is built and tested.
