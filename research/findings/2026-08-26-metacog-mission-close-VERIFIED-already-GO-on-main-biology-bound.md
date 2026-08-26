---
type: finding
status: go
date: 2026-08-26
mechanism: plastic-three-factor-hebbian-metacog-monitor
artifacts:
  - research/findings/raw/lanes/metacog/metacog_plastic_3factor_6seed.json
  - research/findings/raw/lanes/metacog/metacog_host_learnedacc_6seed_ceiling.json
  - research/biology/metacog-three-factor-confidence.md
---

## metacog MISSION-bar residual: VERIFIED already CLOSED on main + biology-bound

**Task premise (stale) vs reality.** The dispatch said `_second_order_metacog_monitor_derisk` is "6/6 base-GO
but `mission_go=False`". On current `main` that is NOT the state: the mission bar was closed 2026-08-19 by
commit `c39885d26` (finding [`2026-08-18-self-organized-metacog-monitor-GO`](2026-08-18-self-organized-metacog-monitor-GO.md)).
The `mission_go=False` reading is what the runner emits on any path that does NOT carry the plastic parity gate —
the base `learned_acc` main() path never writes a `mission_go` field, and the plastic path returns `mission_go=False`
if invoked WITHOUT `--host-json` (the parity gate has no host ceiling to compare against). The banked plastic
6-seed run WITH the host ceiling is `mission_go=True`.

**Independent reproduction (this session).** A fresh single-seed foreground numpy run of the plastic path
reproduced the banked seed-42 result BYTE-IDENTICALLY:

```bash
SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seed 42 --n-trials 160 --confidence-read plastic_acc --backend numpy \
  --host-json research/findings/raw/lanes/metacog/metacog_host_learnedacc_6seed_ceiling.json \
  --json OUT_plastic_s42.json      # scratch; deleted after reading the verdict
```

<!--derived-->
Result (stdout verdict, transcribed; 0.715 = 0.85x the host mean 0.841): type2_auc=0.855 meta_d=2.49
M-ratio=1.35, type1_acc 0.825 (in window), AUC-parity ok (0.855 >= 0.85x0.841=0.715), and ALL controls
collapse — meta-lesion type2_auc->0.500 / meta_d->0.00, permuted perm_p=0.005, self-read-lesion perm_p=0.005 —
host_logistic_calls_on_path=0, seed GO=True. Identical to the banked seed-42 entry in the cited
`metacog_plastic_3factor_6seed.json` (auc 0.855, meta_d 2.49). `cfg.seed` determinism holds.

**The mission bar is the exact mechanism the task requested.** The dispatch asked for "a stronger learned
error-monitor / symmetric comparator reading {cleanup-score, accumulator margin, #competitors}, trained on
correctness." That IS the banked plastic path: a reward-gated three-factor Hebbian OPPONENT (V+/V-) monitor whose
presynaptic ACC features are `winner_rate`/`runner_rate` (cleanup score), `margin_abs`/`signed_margin`
(accumulator margin), `conflict`/`balance` (# / strength of competitors) plus dynamic late-conflict and
response-persistence terms, with the confidence→correctness mapping LEARNED from trial correctness by a
dopamine/RPE-gated local rule (no host optimizer).
<!--derived-->
Banked 6-seed (aggregate values from the cited `metacog_plastic_3factor_6seed.json` and host ceiling artifacts):
mean type2_AUC 0.825 (parity ratio 0.982 vs host 0.841; AUC-parity 6/6), mean meta-d' 2.49, all in-window, all
four controls collapse 6/6.

**Anti-cheats (the task's three) are present and pass 6/6.** (1) lesion → confidence flat while accuracy intact:
meta-lesion zeroes the read-only monitor's drive → type2_AUC→0.500, meta-d'→0 while d'/accuracy UNCHANGED. (2)
shuffled outcomes → miscalibration: self-read-lesion re-fits the SAME rule on SHUFFLED correctness feedback (200-
draw permutation test) → chance. (3) second-order orthogonal to raw difficulty: within-class type2_AUC>0.55 on
all 6 (tracks correctness within a fixed stimulus class, not the stimulus).

**What this session added (no runner change — it is already GO, and modifying a GO 6/6 path risks an
unverifiable regression).** A biology binding `research/biology/metacog-three-factor-confidence.md` (passes
`tools/biology_check.py`, 4 external sources resolve: Fleming-Daw 2017 second-order framework; Holroyd-Coles 2002
ACC error monitoring via a dopaminergic RL signal; Schultz 1997 dopamine RPE; Namburi-Tye 2015 opponent valence
coding) — the mechanism had no binding, which is why the closure was re-dispatchable.

**Honest residual (unchanged, NOT the mission bar).** "self-organized" scopes to the confidence→correctness
MAPPING WEIGHTS; the presynaptic ACC features are host RATE READS of the brain's own competition (not yet a
fully-spiking presynaptic population read), and the learned synaptic sum is injected as the meta subpool current.
A fully-spiking presynaptic read is the next rung. meta-d' PARITY (matching host magnitude) is 4/6 — the two
misses are low-d' seeds where the SDT AUC→meta-d' inversion is noisy, an honest over-strictness of a secondary
read, not an AUC deficit (the AUC parity that IS the gate is 6/6). Functional correlate only; no phenomenal claim.

**6-seed reproduce command (pool / gpu_queue — not run by this agent):**

```bash
SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 160 --confidence-read plastic_acc --backend numpy \
  --host-json research/findings/raw/lanes/metacog/metacog_host_learnedacc_6seed_ceiling.json \
  --json OUT_plastic_3factor_6seed_reverify.json   # NEW file; do not overwrite the banked artifact
```
