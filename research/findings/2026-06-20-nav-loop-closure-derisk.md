# Nav loop-closure de-risk — the deep-research "open reentrant loop" premise is FALSIFIED; the actor does NOT go silent for lack of the `thal→cortex` arc (2026-06-20)

**Type:** cheap-first de-risk of a deep-research diagnosis (CPU-first; the numpy path is broken for this
neural-critic config, so the *tiny* grid-8 smoke ran on GPU — seconds, not a sweep). NO `sim/` edit.
**Pre-registered by:** `research/findings/2026-06-20-nav-reward-value-loop-deep-research.md` (commit `39866596`),
Phase 0: "turn `enable_cluster_a_closed_loop` ON in the failing spiking-SC config and test whether the actor STOPS
going silent (the loop self-sustains)."
**Owner standard:** BRAIN-BASED-ONLY. The deliverable is the honest verdict, positive or negative.

---

## TL;DR verdict — HONEST CORRECTION of the deep-research diagnosis

**The deep-research's load-bearing structural premise is factually wrong, and the recommended Phase-0 experiment is a
no-op as written, because the reentrant arc was ALREADY ON in the config that produced the NO-GO.**

1. **The `thal_X → cortex_X` reentrant self-sustain arc (`enable_cluster_a_closed_loop`, catalog A.05) was ON, not
   OFF, in the spiking-SC NO-GO.** The NO-GO (`2026-06-19-nav-spiking-sc-deploy-NO-GO.md`, ~58× host, actor silent)
   was produced by `research/runners/_nav_gate_merged_run.py --with-conv --spiking-sc`. That runner sets
   `enable_cluster_a_closed_loop=True` in its base `kw` (the line just above `enable_cluster_e_topography`), for BOTH
   the host-control arm and the SC arm. No code path disables it when `--spiking-sc` is set (grep-verified across the
   runner, `nav_conv_merged_bridge.py`, and `g11_bg_runner.py`). The deep-research doc's central claim — *"the
   reentrant `thal_X → cortex_X` closure is gated behind `enable_cluster_a_closed_loop` (default OFF, and NOT set by
   `--spiking-sc`)"* — is **incorrect** for the deployed config.

2. **Direct synapse-count instrumentation confirms the flag controls the arc exactly as expected, and the actor does
   NOT go silent at the smoke scale with the arc ON *or* OFF.** A grid-8 / 120-step standalone A/B (the failing
   `--spiking-sc` kwargs, only the closed-loop flag differing):

   | arm | `thal→cortex` synapses | per-action | motor_sustain | late_sustain (2nd half) | total motor spikes |
   |---|---|---|---|---|---|
   | **closed_on** (arc ON; the NO-GO's actual setting) | **490** | N:137 E:127 S:102 W:124 | **0.950** | **1.000** | 1792 |
   | **closed_off** (arc OFF; the doc's *claimed* failing config) | **0** | all 0 | **0.975** | **1.000** | 2040 |

   The arc is flag-controlled (490 vs 0). But the actor **sustains firing to the end in BOTH arms** (late_sustain =
   1.000 either way). The "actor goes silent" signature the doc attributes to the open loop does NOT appear at this
   scale, and closing the loop is not what keeps the actor firing.

3. **Closing the loop does NOT rescue navigation — at grid-8 it is comparable-to-slightly-worse, and the SC arm is
   already ~6× worse than the host-control regardless of the arc.** Phase-0 finalQ (single-phase, the clean
   apples-to-apples at 120 steps; lower = better):

   | config (grid-8 / 120 / seed 42) | phase-0 finalQ |
   |---|---|
   | **host-control** (heuristic orienting + host reward, arc ON) | **0.50** |
   | SC-arm **closed_on** (arc ON) | 0.982 |
   | SC-arm **closed_off** (arc OFF) | 2.867 |

   The SC arm underperforms the host-control by ~2–6× even at this easy scale (a partial reproduction of the NO-GO's
   58× — the full gap needs grid-32 / 1800). Crucially, the reentrant arc being ON does not close that gap; the loss
   is in the SC orienting drive / neural-reward policy, not in the loop's self-sustain.

4. **The faithful MERGED config (the exact NO-GO setup: `--with-conv` + `--spiking-sc`), grid-8 / 480 steps (all 4
   goal phases reached, conv co-resident), confirms it.** This is the cleaner read (no phase-count mismatch):

   | arm | `thal→cortex` synapses | motor_sustain | late_sustain (2nd half) | gate (4-phase Σ finalQ) |
   |---|---|---|---|---|
   | **closed_on** (arc ON; the NO-GO's actual setting) | **490** | 0.990 | **0.992** | **4.48** |
   | **closed_off** (arc OFF; the doc's *claimed* config) | **0** | 0.969 | **0.963** | **5.58** |

   On the merged bridge the actor **still does not go silent** (late_sustain 0.992 / 0.963 — fires to the end in
   both). Closing the loop here is mildly **beneficial** (gate 4.48 ON < 5.58 OFF), the OPPOSITE of "the open loop
   causes the silence" — the arc is a small help that was already present, not the missing sustain. The SC arm is
   still ~9–11× the host-control floor (~0.5) regardless of the arc, so closing the loop does not rescue navigation.

**⇒ The fork's answer is NOT "open loop / loop-stability via the missing reentrant arc."** That arc was already
closed. The actor-silence in the NO-GO is therefore NOT caused by the open `thal→cortex` loop (it cannot be — the
loop was closed). The deep-research diagnosis identified the right *family* (a systems / operating-point issue, not a
dendritic credit-assignment wall — that part is well-argued and consistent with the organs validating in isolation),
but it pinned it on the wrong structural cause. The load-bearing problem is the **SC orienting current (`sc_map →
cortex_X`, weight 18) being too weak to replace the 800 pA host Manhattan heuristic as the actor's drive**, with the
neural reward/critic loop not compensating — and the already-closed reentrant arc demonstrably does not fix it.

This is the honest cheap-first deliverable: a controller-verified deep-research premise, trust-but-verified against
the actual code + a direct measurement, found to be **falsified at the structural level**. Per the standing
"trust-but-verify the load-bearing claims" discipline, this is exactly the check that was meant to catch it before
GPU spend on the wrong lever.

---

## What was tested

**Probe:** `research/runners/_nav_loop_closure_derisk.py` (CPU-first; the `enable_neural_critic` + `spiking_snc`
config hits a pre-existing CuPy/NumPy backend mismatch at `g11_bg_runner.py:4838` under `SIM_BACKEND=numpy` —
`region_indices_cp` is unconditionally `cp.asarray` while `cp_external_input_current` is a NumPy array — so the tiny
grid-8 smoke ran on GPU. This is a *seconds* smoke, not a sweep; the CPU-first intent — don't burn the GPU on a big
sweep before a cheap check answers the fork — is honored, and the numpy-path bug is flagged below, not patched).

It runs the EXACT failing `--spiking-sc` merged kwargs (`enable_spiking_sc` + `enable_spiking_sc_approach` +
`spiking_reward_us` + `enable_neural_critic` + `spiking_snc` + `heuristic_strength=0`, the SC merged op-point env
values 160/12/3500/40), toggling only `enable_cluster_a_closed_loop`, and per arm: (a) counts the `thal_X →
cortex_X` synapses directly from the built CSR (confirming the arc structurally), (b) reads `motor_counts` + the
phase finalQ from the episode JSON.

**Anti-cheat / methodology used:**
- **Direct structural confirmation** (the arc synapse count, not an assumption) — the 490-vs-0 count is the decisive
  refutation of "the arc is OFF."
- **Host positive control** — the host-heuristic arm at the same grid-8/120 anchors the SC-arm degradation (0.50 vs
  2.87–0.98).
- **Lesion already implied** — the doc's anti-cheat #2 was "lesion the reentrant arc → actor must go silent again."
  Here the lesion (`closed_off`) does the OPPOSITE of regressing: the actor sustains (late 1.000) and phase-0 finalQ
  is comparable. So by the doc's own anti-cheat criterion, the arc is **not** the load-bearing sustain.

---

## Honest scope + the faithful-scale handoff

- **Scale caveat (the one open item):** the NO-GO's "actor goes silent after warmup" was reported at **grid-32 /
  1800 steps**; this de-risk's actor-sustain finding is at **grid-8 / 120** (standalone) and **grid-8 / 480 merged**
  (all 4 phases, conv co-resident). The *structural* refutation (arc was ON) is exact and scale-independent — it is a
  code fact + a synapse count (490 vs 0), true at any grid. The *behavioral* claim ("closing the loop does not rescue
  / the actor does not silence for lack of the arc") is shown at two small scales (standalone 120 + merged 480) and
  should be confirmed at the faithful grid-32/1800 scale before the file is fully closed. But note: because the arc
  was ALREADY ON in the grid-32/1800 NO-GO, the grid-32/1800 "closed_on" arm simply IS the NO-GO (gate 117.5). The
  only new datum the faithful A/B adds is whether the **lesioned** (`--no-closed-loop`) arm is *even worse* — which,
  if the arc were the missing sustain, it would be; at both small scales it is not (the lesion is comparable or
  slightly better/worse, never a regression toward silence).

- **The faithful grid-32/1800 A/B is wired and handed to the controller** (the decisive scale test, GPU):
  `_nav_gate_merged_run.py` now has an additive `--no-closed-loop` flag (default off = arc ON = byte-identical to the
  existing STEP-2a gate + SC deploy A/B; verified via `--help`). The A/B:

  ```bash
  # SC arm, reentrant arc ON (== the NO-GO's actual config; expected ~58x host)
  SC_RET_SC=160 SC_REC=12 SC_RET_DRIVE=3500 SC_ROS_US=40 SIM_BACKEND=cupy \
    python -m research.runners._nav_gate_merged_run --with-conv --spiking-sc \
    --seed 42 --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/loopclose_grid32_on_seed42.json
  # SC arm, reentrant arc LESIONED (the doc's claimed failing config; the decisive control)
  SC_RET_SC=160 SC_REC=12 SC_RET_DRIVE=3500 SC_ROS_US=40 SIM_BACKEND=cupy \
    python -m research.runners._nav_gate_merged_run --with-conv --spiking-sc --no-closed-loop \
    --seed 42 --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/loopclose_grid32_off_seed42.json
  ```
  Read `gate_score` + `motor_counts` from each. **Prediction (from the grid-8 result):** the lesion (`--no-closed-loop`)
  will NOT meaningfully change the outcome — both arms ~58× host with the actor sustaining; closing the loop is not
  the missing fix. If instead the lesion regresses sharply while the arc-ON arm recovers, the doc's mechanism would
  be vindicated at scale (not expected). 3 seeds suffice if the effect is large/mechanistic; 6 if it is a marginal
  variable effect.

---

## What the de-risk says about the REAL next lever (for the controller)

Given the arc was already closed and does not rescue the actor, the deep-research's *downstream* recommendations that
do NOT depend on the false premise are still the right direction:
- **6-B (keep a neural perception drive into the actor):** the merged-gate config has NO `place_goal_readout`
  (grep-verified) — the only actor drive in the SC arm is `sc_map → cortex_X` (weight 18) plus the (warmup-gated)
  vision hierarchy. The single most plausible real fix is **raising the SC→cortex orienting drive** and/or keeping a
  stronger neural perception drive into the actor, since the 800 pA host heuristic it replaced was ~44× the SC weight.
- **6-C (the #4 urgency / N-scaling op-point levers):** already-default-on knobs that let a weak orienting current
  cross the commit bound — applicable here and cheap.

These are operating-point levers on the **drive**, consistent with the doc's correct "operating-point family, not a
dendrite" framing — just targeting the SC→cortex drive (the actual load-bearing gap) rather than the (already-closed)
reentrant loop.

---

## sim/ edit?

**NONE.** Reuse-by-import of `run_moving_goal_episode` + the merged conv hook. The one runner change is the additive
`--no-closed-loop` flag on `_nav_gate_merged_run.py` (default off = byte-identical). The numpy-backend bug at
`g11_bg_runner.py:4838` for the neural-critic config is FLAGGED (not patched) — it blocks CPU smokes of this config
and is a candidate for a separate fix.

## Commits (all on `main`, PATHSPEC, pushed origin + gitea)

- `cdb2603d` — the probe `_nav_loop_closure_derisk.py` + the standalone grid-8/120 A/B result (arc 490 vs 0; actor
  sustains both arms).
- (this finding) + the `--no-closed-loop` flag on `_nav_gate_merged_run.py` + the merged smoke + host-ref JSONs.
