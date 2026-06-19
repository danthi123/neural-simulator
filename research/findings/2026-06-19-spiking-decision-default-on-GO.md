# Roadmap #4 (TRUE-ONE-BRAIN) — the fully-spiking nav action-decision is DEFAULT-ON (6-seed GO, 1.16× host, 100% commit-burst) (2026-06-19, CYCLE 235)

**Owner directive (2026-06-19, brain-based purity):** "dispatch sub agents to close the boundaries that only
require engineering, and for the action-decision one, spend some time with deep research, planning, testing... to
see how much we can lower the cost. Once you've determined we've done all we reasonably can, set it to default-on.
The goal here is a fully spiking, single brain as default."

This is the #4 (action-decision) deliverable: the merged "one brain" now navigates with a **fully-spiking decision
by default** — the action EMERGES from the spiking competition (a Wang-2002 NMDA accumulator + a Lo-Wang/superior-
colliculus commit-burst threshold-crossing), retiring the host Python argmax. The host argmax stays available as
the opt-in **oracle**.

## The arc: from BOUNDARY to default-on

**Starting point (CYCLE 216, the #4 boundary):** the fully-spiking read-out was GENUINELY achieved on the merged
bridge (every step decided by the `commit_X` all-or-none burst, the host argmax retired) but it navigated WORSE —
**~1.7× the host argmax** (the documented substrate cost: a clean spiking decision is slower/less selective than a
zero-noise argmax).

**Deep-research plan** (`2026-06-19-spiking-decision-cost-reduction-plan.md`): the ~1.7× cost is **~85% closable,
~15% fundamental**. The entire excess sits in the *post-goal-change* phases (stable phases already reach the host
floor); two named, mechanism-fixable spiking artifacts — **cross-trial NMDA hysteresis** (the τ≈100ms accumulator
lingers on the previous winner when the goal switches) and **weak-drive silent-commit** — over a small irreducible
finite-size noise floor (∝1/√N) no spiking race beats versus a zero-noise argmax (the honest BRAIN-BASED-ONLY
residual).

**Testing (grid-8 ranking → grid-32 deploy gate), anti-cheat enforced every round (the win must come from the
`commit_X` burst = `decision_path=primary`, NOT a sel-lean argmax fallback):**

| round | lever | result |
|---|---|---|
| ROUND 1 | accumulator LEAK (lower `sel_recurrent_weight` = Usher-McClelland forgetting) | grid-8 3-seed: srw 1.0→**+1.28**, 0.3→**+0.44** vs host — the leak cuts the cost **~66%**, 100% commit-burst |
| ROUND 2 | extend leak below 0.3 | optimum is a BASIN ~0.2–0.3; **srw 0.1 COLLAPSES** the attractor (too little recurrence → unreliable commit). grid-8 has run-to-run variance → the 0.2-vs-0.3 distinction is within noise |
| ROUND 3 | N-scaling (paired `n_sel`=`n_commit`, the 1/√N lever) | grid-32: n=20 → 1.37× host; **n=40 → 1.13×** (within 25%); n=80 → 1.16× (40 is the sweet spot) |

**6-seed deploy gate (grid-32/1800, the winning config):**

| seed | score | ratio vs host | commit-burst primary |
|---|---|---|---|
| 42 | 2.451 | 1.23× | 1800/1800 |
| 43 | 2.035 | 1.02× | 1800/1800 |
| 44 | 2.266 | 1.13× | 1800/1800 |
| 100 | 2.664 | 1.33× | 1800/1800 |
| 101 | 2.416 | 1.21× | 1800/1800 |
| 102 | 2.097 | 1.05× | 1800/1800 |
| **mean** | **2.32** | **1.16×** | **100%** |

Host motor/thal = exactly **2.0** on all 6 seeds (the optimal floor, deterministic). The spiking decision is
**within 25% of host (criterion met)** with the decision terminating on the spike **100% of the time** (zero
argmax fallback, zero random) — the host argmax is genuinely retired, not a hidden tie-break.

⇒ the two levers (leak + N-scaling) take the spiking-decision cost **~1.7× → 1.16×** (≈77% of the gap closed). The
remaining ~16% is the irreducible commit-timing / finite-size-noise floor — per the BRAIN-BASED-ONLY standard, this
honest residual IS the deliverable (it maps the cost the point-neuron substrate pays to make the decision itself a
spike). "Done all we reasonably can" → default-on, eating the residual (the owner's directive).

## The deploy (NO `sim/` edit; runner-only default flip)

**Winning config:** `readout_source="spiking_wta"`, `sel_recurrent_weight=0.3`, `n_sel_per_action=n_commit_per_action=40`,
`urgency_max_pA=180.0`, `enable_commit_burst=True`.

`run_moving_goal_episode` (`g11_bg_runner.py`) FUNCTION defaults flipped to the winning config, so the merged "one
brain" (and any caller inheriting the default) navigates fully-spiking by default. **The CLI `--readout-source`
default stays `"motor"`** — the documented standalone benchmarks reproduce unchanged, and `motor`/`thal` are the
opt-in host-argmax oracle. The tuned levers are inert under `motor`/`thal` (the sel/commit layer is only built for
`spiking_wta`). `build_bg_brain_regions`' own defaults are untouched (the episode passes its values down).

**Gates (both GREEN):**
- **nav-not-regressed** = the 6-seed grid-32 table above (1.16× host, within 25%, 100% commit-burst).
- **conversational answer-identity / no-confab moat** = `tests/test_nav_conv_merged_agent.py` (8) +
  `tests/test_nav_conv_step2b_coresident.py` (7) pass with the new default. These build `MergedNavConvAgent`, which
  does NOT route through `run_moving_goal_episode` and builds without the WTA layer — so the spiking read-out is
  array-disjoint from the parser/composer and the moat is preserved by construction (the gate is the confirmation).

## Reproduce
```bash
# the 6-seed deploy gate (the cheap-first grid-8 ranking rounds are in nav_gate_2a/_round*.json):
SIM_BACKEND=cupy python -m research.runners._merged_spiking_readout_navcmp \
  --seeds 42,43,44,100,101,102 --sweep "n_sel_per_action+n_commit_per_action=40" \
  --lever sel_recurrent_weight=0.3 --grid-size 32 --n-steps 1800 --urgency-max-pa 180
# default-on: run_moving_goal_episode now defaults to spiking_wta; the host-argmax oracle is readout_source="motor".
```

## Status: #4 = the cleanly-closable boundary, CLOSED + deployed

The owner's brain-based-purity directive split three ways: **#4 (action-decision)** was the one cleanly closable by
engineering — leak + N-scaling brought it within the deploy bar and it is now DEFAULT-ON (the merged brain's
action-decision is fully spiking). The other two "engineering-only" boundaries turned out to need more: **#5 (place
code)** hit a substrate value-read-out wall (sparsification worked but the all-or-none critic read-out can't grade →
needs a graded-rate read-out or the dendrite; `2026-06-19-place-code-sparsify-default-BOUNDARY.md`), and **#3
(cue-shift consolidation)** reached the numeric bar but failed the lesion anti-cheat (a merged-SNc onset transient;
`2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md`). Both stand as honest, well-localized boundaries with the
brain-based version underneath (the self-org place code; the standalone lesion-clean cue-shift). In tandem, the
**latency CSR cache** landed a 10–19.5× composer speedup (`2026-06-19-latency-csr-cache-GO.md`).
