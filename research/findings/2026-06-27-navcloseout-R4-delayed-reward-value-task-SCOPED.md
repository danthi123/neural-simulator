# nav close-out R4 — the DELAYED-REWARD task that proves the spiking nav VALUE-critic is load-bearing — BUILT + CPU-SMOKE GREEN (2026-06-27)

**Type:** BUILD + CPU smoke (the runner-side harness + the well-formedness validation). NO `sim/` edit.
**Status:** task + 2×2 factorial harness BUILT; CPU smoke PASS; the GPU eval command + de-risk criteria are
specified below and **FLAGGED "FOR CONTROLLER TO RUN"** (the long GPU eval is NOT run here — the B1/R1-a
stall lesson: a subagent cannot resume on a background GPU run, so it must hand the command off).
**Runner:** `research/runners/_navcloseout_R4_delayed_reward_value.py`
**Gate context:** `2026-06-27-nav-loop-closure-research-gate.md` (SHA 7e66e81a) RANK 4 (R4); the value-load-bearing
design `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` (the catalog F.22/F.23 2×2 logic).

---

## 0. The one-paragraph result

The merged "one brain" nav value-critic is SPIKING-DEFAULT-ON (CYCLE 1B: `enable_neural_critic` +
`spiking_snc` + `spiking_reward_us`; δ=r−V = the SNc firing minus the striosome GABA_B subtraction), but its
LOAD-BEARINGNESS has only ever been shown on IMMEDIATE-reward nav, where the gridworld is orient-solvable and a
value baseline is not strictly necessary — so lesioning the value barely moved the score (the #9 nav deploy was a
QUALIFIED-NEGATIVE for exactly this reason, Δ7.2%). **R4 builds the task that makes the value PROVABLY needed: a
DELAYED-reward nav variant (a temporal GAP between an approach action and its reward delivery) + a 2×2 factorial
{value-critic ON, OFF} × {immediate, delayed}.** The load-bearing signature (validate-by-function): the
spiking value-critic HELPS on the DELAYED arm (it must bridge the reach→reward gap) and is ~NEUTRAL on the
IMMEDIATE arm (where no value baseline is needed) — i.e. a value×delay INTERACTION, not a general boost. Built
with **~95% reuse, NO `sim/` edit**: the GAP is a `homeostatic_hook` closure (run_moving_goal_episode's existing
per-step hook — buffer each step's reward, release it `delay` steps later → the SNc/critic/STDP see the reward in
a CS-free gap); the value lesion is a `prebuilt_post_init_hook` that zeros `cp_gabab_synapse_mask` (the established
`_merged_navcritic_valuetrain.lesion_gabab` route — the spiking critic still fires but supplies no value
baseline); the spiking-default merged nav config is the run_moving_goal_episode library default (CYCLE 1B). The
CPU smoke (6 checks) is GREEN: the delay hook delivers reward late (lag == delay, reward conserved), the immediate
arm is a pure pass-through, the PERMUTED-delay anti-cheat breaks the earned↔delivered contingency, the value-lesion
logic is sound, the 2×2 matrix is well-formed, and the verdict aggregator computes the interaction. The full 2×2
episode is the GPU eval (run_moving_goal_episode is CuPy-only), specified below.

---

## 1. The task + the 2×2 factorial

**Substrate:** `run_moving_goal_episode` (the documented multi-goal moving-goal nav at **grid-32**, NEVER grid-8),
the SPIKING-DEFAULT merged nav limbic core (CYCLE 1B). Score = the nav cost = **mean Manhattan distance to the
goal over the episode** (`mean_distance_overall`, LOWER is better) + `n_steps_at_goal` (HIGHER is better).

**The DELAYED-reward variant (the temporal GAP — zero edit to the episode loop):**
A `make_delay_hook(delay)` closure is passed as `homeostatic_hook` (called every step AFTER the natural reward is
finalized, as `gated_reward, new_goal = hook(reward, x, y, gx, gy, step, dist_after)`). It maintains a FIFO primed
with `delay` zeros: at step t it enqueues the just-earned reward and releases what was earned `delay` steps ago.
So the spiking SNc burst (and the corticostriatal STDP it gates) arrives in a CS-free gap that the value critic +
eligibility trace (`reward_eligibility_tau_ms=500`) must bridge. `delay=0` is a pure pass-through (the immediate
arm). **This is the trace-conditioning logic (catalog F.22) adapted to nav: a reward separated from its predictive
event by a gap, where the only way to learn the right policy is to carry a learned value across the gap.**

**The 2×2 factors:**

| arm | `value_on` | `reward_delay` | mechanism |
|---|---|---|---|
| `value_on_immediate`  | True  | 0  | spiking critic ON; no gap (pass-through hook) |
| `value_off_immediate` | False | 0  | GABA_B→SNc lesioned; no gap |
| `value_on_delayed`    | True  | 12 | spiking critic ON; reward delayed 12 steps |
| `value_off_delayed`   | False | 12 | GABA_B→SNc lesioned; reward delayed 12 steps |
| `value_on_delayed_permuted`  | True  | 12 | **AC_PERMUTE:** delayed + contingency broken |
| `value_off_delayed_permuted` | False | 12 | AC_PERMUTE control |

- **value ON** = the SPIKING striosome critic subtracts V at the SNc via GABA_B (the merged default).
- **value OFF (lesion)** = the SAME spiking critic, but `cp_gabab_synapse_mask` is zeroed in the
  `prebuilt_post_init_hook` (the established value lesion) → the critic fires but supplies NO value baseline.
  The hook records `n_gabab_cut` (a non-zero count is the proof the lesion landed on a real value route).

---

## 2. The de-risk criteria (what makes R4 a GO — validate-by-function)

Score = `mean_distance_overall` (LOWER better), so an IMPROVEMENT = (value-OFF − value-ON) (positive when ON is
better). Computed by `summarize_factorial`.

| Gate | Criterion |
|---|---|
| **(G_HEADLINE) value HELPS on DELAYED** | `improvement_delayed = md(value_off_delayed) − md(value_on_delayed) > 0` (the value is load-bearing across the gap), ≥5/6 seeds. |
| **(G_DISCRIM) value ~NEUTRAL on IMMEDIATE** | `improvement_immediate ≈ 0` (`|imp_immediate| ≤ max(0.5, 0.5·|imp_delayed|)`): the value is NOT needed without a gap — the **direct answer to the orient-solvable confound that sank the #9 nav deploy**, ≥5/6 seeds. |
| **(G_INTERACTION) the value×delay interaction is POSITIVE** | `interaction = improvement_delayed − improvement_immediate > 0`: the critic's help is SPECIFIC to the delayed condition, NOT a general boost, ≥5/6 seeds. |
| **(AC_PERMUTE) the genuine help exceeds the permuted help** | `improvement_delayed > improvement_delayed_permuted`: with the CS→reward contingency broken (the reward delivered after the gap is a randomly-permuted earned magnitude, not the one earned `delay` steps ago), the value cannot bridge anything → the headline help must come from the genuine contingency, not the mere presence of a gap. |
| **(AC_LESION-LANDED) the lesion cut real GABA_B synapses** | `n_gabab_cut > 0` on every value-OFF arm (the lesion silenced a real value route, not a no-op). |

**Pass = G_HEADLINE ∧ G_DISCRIM ∧ G_INTERACTION ∧ AC_PERMUTE (+ AC_LESION-LANDED) at ≥5/6 seeds.** The decisive
pair is G_HEADLINE ∧ G_DISCRIM: the value lesion collapses the DELAYED advantage but NOT the IMMEDIATE score →
the value is provably load-bearing for the function it computes (credit over a gap), and the task is proven to
discriminate "needs the critic" from "immediate-reward-solvable."

**Honest scope / expected failure modes (the deliverable either way):**
- This is the **instrumental** trace variant (the agent ACTS over the gap, not just predicts). The B4/#9 scoping
  flagged that ACT-over-gap is where a real substrate wall MIGHT appear (the spatial actor-critic-credit family,
  3× NEGATIVE on the hidden goal). **If the value does NOT help on delayed (G_HEADLINE fails), that is the honest
  characterized boundary** — it localizes the wall to ACT-over-gap on the point-neuron cascade (distinct from the
  PREDICT-over-gap Pavlovian close, which is a point-neuron GO). It would be the legitimate juncture for the
  deferred dendritic substrate question, NOT a failure of this task.
- The eligibility-trace window (`reward_eligibility_tau_ms=500`, ~one nav step at dt=1ms scale) bounds the
  bridgeable gap. `--reward-delay 12` is the first probe; a **gap sweep {6, 12, 24}** is the natural follow-on
  (the catalog's gap-parameter sweep — at the gap where the value first matters, the interaction should peak).

---

## 3. THE GPU EVAL — **FOR CONTROLLER TO RUN** (do NOT background-and-wait)

`run_moving_goal_episode` is **CuPy-only** (it does `import cupy as cp` internally), so the full 2×2 episode runs
on GPU. Each arm is one ~grid-32/1800-step episode (~minutes on a 3090); the full 6-arm factorial × 6 seeds is
~36 episodes. Run as a controller background task and read the result JSONs — **do not block on it.**

### 3.1 The full 2×2 + permuted control, one seed (the per-seed verdict):

```bash
SIM_BACKEND=cupy python -m research.runners._navcloseout_R4_delayed_reward_value \
    --factorial --seed 42 --grid-size 32 --n-steps 1800 --reward-delay 12 \
    --out research/findings/raw/navcloseout_R4/R4_factorial_seed42.json
```

Prints `improvement_immediate`, `improvement_delayed`, `value_x_delay_interaction`, and the per-seed PASS proxies
(`helps_on_delayed`, `neutral_on_immediate`, `interaction_positive`, `permute_control_ok`). Writes per-arm JSONs
to `research/findings/raw/navcloseout_R4/R4_<arm>_seed42.json` + the factorial summary.

### 3.2 Multi-seed (the gate — 6 seeds for the variable effect, per the standing rule):

```bash
for S in 42 43 44 100 101 102; do
  SIM_BACKEND=cupy python -m research.runners._navcloseout_R4_delayed_reward_value \
      --factorial --seed $S --grid-size 32 --n-steps 1800 --reward-delay 12 \
      --out research/findings/raw/navcloseout_R4/R4_factorial_seed$S.json
done
```

(On Windows PowerShell: `foreach ($S in 42,43,44,100,101,102) { SIM_BACKEND=cupy python -m ... --seed $S --out ...seed$S.json }`.)

**The 6-seed verdict:** GO if, across ≥5/6 seeds, `helps_on_delayed ∧ neutral_on_immediate ∧ interaction_positive
∧ permute_control_ok`. The per-seed JSONs carry `improvement_delayed` / `improvement_immediate` /
`value_x_delay_interaction` / `improvement_delayed_permuted` for a simple aggregate (mean interaction > 0 +
≥5/6 PASS).

### 3.3 (optional) ONE arm in isolation (for a quick GPU sanity before the full factorial):

```bash
SIM_BACKEND=cupy python -m research.runners._navcloseout_R4_delayed_reward_value \
    --arm value_on_delayed --seed 42 --grid-size 32 --n-steps 1800 --reward-delay 12
```

### 3.4 (optional follow-on) the gap sweep (the value should matter MORE as the gap grows past the eligibility window):

Run §3.1 at `--reward-delay 6`, `12`, `24` and compare the interaction (it should be ~0 at a gap the
immediate-reward dynamics already bridge, and rise where the value becomes the only bridge).

---

## 4. The CPU smoke (what was validated here, GREEN)

```
SIM_BACKEND=numpy python -m research.runners._navcloseout_R4_delayed_reward_value --smoke
```
Six checks, all PASS (no bridge built — run_moving_goal_episode is CuPy-only, so the smoke validates the
runner-side LOGIC):
- **(a)** the delay hook delivers reward DELAYED by exactly `delay` steps (`delivered[t] == earned[t−delay]`,
  0 for `t < delay`) + reward conserved over the matured window (in 3.0 == out 3.0).
- **(b)** the immediate arm (`delay=0`) is a pure pass-through (delivered == earned).
- **(c)** the PERMUTED control breaks the earned↔delivered timing (delivered ≠ simply-lagged) while drawing only
  from genuinely-earned reward magnitudes (no fabricated reward).
- **(d)** the value-lesion logic zeros a (mock) `cp_gabab_synapse_mask` + clears `g_gabab` and records
  `n_gabab_cut` (4 of 6 in the mock).
- **(e)** all six 2×2 arms set the value/timing factors correctly + carry the spiking-default nav config
  (multi-goal LIST schedule, `spiking_snc` + `enable_neural_critic` + `spiking_reward_us` +
  `perceived_approach_reward`).
- **(f)** the verdict aggregator computes the value×delay interaction from a synthetic 2×2 (imp_immediate=0.10,
  imp_delayed=2.50, interaction=2.40, permute_ok=True).

---

## 5. Reuse map (~95% reuse-by-import, NO `sim/` edit)

| Reused piece | From | What it provides |
|---|---|---|
| `run_moving_goal_episode` | `research/runners/g11_bg_runner.py` | the whole nav episode + the SPIKING-default merged limbic core (CYCLE 1B defaults) + the score |
| `homeostatic_hook` (per-step) | `run_moving_goal_episode` arg | the zero-edit DELAY lever (buffer→release reward `delay` steps later) |
| `prebuilt_post_init_hook` | `run_moving_goal_episode` arg | the zero-edit VALUE-LESION lever (zero `cp_gabab_synapse_mask` after build) |
| `cp_gabab_synapse_mask` zeroing | `_merged_navcritic_valuetrain.lesion_gabab` | the established value-critic GABA_B lesion (mirrors the in-episode lesion at `g11_bg_runner.py:6279`) |
| the multi-goal schedule | `g11_bg_runner` argparse `--goal-schedule multi` | the documented flagship moving-goal task (rebuilt as the LIST form, scaled to grid+n_steps) |
| `critic_warmup_trials` | `run_moving_goal_episode` arg | latent-learning so the value critic has acquired V BEFORE the test |

**The genuinely-new code:** the `make_delay_hook` FIFO closure (+ the permuted variant), the
`make_value_lesion_hook`, the 2×2 arm matrix + `build_episode_kwargs`, the `summarize_factorial` verdict, and the
CPU smoke. All runner-side.

**`sim/`-edit-or-not: NONE NEEDED.** The DELAY is a pure `homeostatic_hook`; the value lesion is a pure
`prebuilt_post_init_hook` zeroing an existing mask; the spiking critic + GABA_B + eligibility traces + spiking
reward_us all ship and are the library default. No protected-code change was required or made.

---

## 6. Anti-cheat / moat

- **(AC_MOAT)** the conversational no-confab moat is ARRAY-DISJOINT from the nav/limbic critic by construction
  (`cp_rf_w_re/im` separate arrays from `cp_connections`); **this standalone harness builds the NAV bridge only**
  (it calls `run_moving_goal_episode` directly, not the merged conversational agent), so the moat is preserved by
  construction and untouched. *(If a future variant runs this on `MergedNavConvAgent`, re-assert
  `_merged_navcritic_valuetrain.check_moat` — `what_does('dog','go')=='north'` AND `what_does('river','look') is
  None` — FLAGGED here; not needed for the nav-only harness.)*
- **(AC_REGIME)** grid-32 (NEVER grid-8, the documented false-GO scale); the run_moving_goal_episode deterministic
  nav regime; 6 seeds for the variable effect.
- **(AC_PERMUTE + AC_LESION-LANDED)** built into the harness (§2).

---

## 7. Citations

**Project record:** `2026-06-27-nav-loop-closure-research-gate.md` (R4 = the delayed-reward 2×2; ~90% reuse, no
`sim/` edit); `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` (the value-load-bearing design + the
F.22/F.23 2×2 logic; the #9 deploy qualified-NEGATIVE Δ7.2%; the V-A-safe / V-B-genuine substrate-wall
localization); `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md` + `_merged_navcritic_valuetrain.py` (V learned
co-resident ~20×, lesion-confirmed; `lesion_gabab` + `check_moat`); `feedback_validate_signal_by_its_function`
(the validate-by-function standard); the CYCLE-1B spiking-limbic default flip (`g11_bg_runner.py:3346/3480/3506/3523`).

**Code (verified this pass):** `run_moving_goal_episode` (CuPy-only; `homeostatic_hook` `:3358/7676`;
`prebuilt_post_init_hook` `:4239/5038`; the GABA_B lesion handle `cp_gabab_synapse_mask` `:6279`; the spiking-SNc
critic drive `:7851-7892`; `results` dict `:8066`, returned `:8248`).

**Catalog/literature (via the R4 + B4 scoping):** F.22 trace conditioning + the delay-vs-trace × lesion 2×2
factorial; C.29 eligibility traces; C.30 actor-critic; C.22 Schultz RPE; Sutton-Barto 2e (TD/bootstrap/eligibility);
Hesslow-Yeo 2002; the eNeuro-2025 NAc-DA-encodes-the-trace-period result.

_BUILD + CPU smoke. NO `sim/` edit. The long GPU eval is NOT run here — §3 commands are FLAGGED FOR THE CONTROLLER
(the B1/R1-a stall lesson: a subagent cannot resume on a background GPU run)._
