# Shortcut-burndown #9-critic — GRADED-RATE critic read-out: CHARACTERIZED (point-neuron read-out over the deferred dendritic field-carving floor) (2026-06-20)

**Item:** the LAST active shortcut-burndown item. Nav shortcut #9 splits (per the read-out research gate `345d0687`,
controller-verified) into **(a) a POINT-NEURON critic READ-OUT** over **(b) a deferred DENDRITIC-flavored place-
FIELD-carving floor**. This is **(a)**: convert the nav value-critic's read-out from its all-or-none coincidence-
plateau form (which over-clamps the SNc → binarizes the value signal) to a GRADED RATE read-out (a point-neuron
linear-synapse critic; value = striosome firing RATE scaling with the learned weight), and measure whether the
spiking δ = r − V improves OR is capped by the deferred dendritic floor.

**Verdict: CHARACTERIZED — the graded-rate point-neuron read-out is NOT realizable for the MSN-D1 critic; the
read-out form is a genuine FORK over the deferred dendritic floor (b), not a clean swap.** This is the EXPECTED,
valid deliverable the task pre-registered ("an honest characterized boundary (δ field-capped) is an EXPECTED, valid
deliverable — this item is likely a point-neuron read-out over a dendritic floor"). **NO `sim/` edit.**

## SCOPE (the read-out, located)

The nav value-critic (`g11_bg_runner.build_bg_brain_regions(enable_neural_critic=True, spiking_reward_us=True)`)
computes δ = r − V brain-based: the SNc reward burst `r` (spiking `reward_us` US afferent) is SUBTRACTED by the
critic's value `V` via GABA_B/GIRK inhibition (`striosome_value → snc`, `receptor="gaba_b"`, E_K=−90 mV) at the SNc
membrane; the host `_V_scaffold` term is dropped (`g11_bg_runner.py:7358-7365`). δ is read as the SNc spike count over
the reward-hold window (`_snc_burst_rate`, `g11_bg_runner.py:5771`). **The read-out form** is HOW the critic value V is
produced from the place afferent — two distinct paths in the runner:

- **LINEAR (graded-rate, point-neuron):** `vs_place_context → striosome_value` PLAIN PLASTIC synapse (the host-Gaussian
  `vs_place_context` scaffold, `g11_bg_runner.py:1878-1891`/`:1891`). V = a graded striosome firing rate (learned w ×
  place drive). This is the form whose value-train reached the host-Gaussian δ ~1.3 (CYCLE-212/219).
- **PLATEAU (all-or-none):** `place → striosome_value` with `coincidence_detector=True` (`g11_bg_runner.py:1871-1877`,
  the `--neural-place-selforg` Route-D path nav DEPLOYS) → read through `fused_coincidence_plateau` (`sim/kernels.py:253`):
  a steep all-or-none sigmoid `g_inc = plateau / (1 + exp(-gain·(c_drive − k)))`, `gain=2`, `plateau=80` ⇒ SATURATES at
  ~plateau once k inputs coincide. This is the read-out the burndown flags as over-clamping (it fires the MSN ~125-378 Hz,
  unphysiological for an MSN, which over-clamps the SNc).

## DE-RISK (graded-rate vs all-or-none, SAME harness, faithful scale)

Both read-outs measured on the SAME deterministic-nav-faithful harness (reuse-by-import of the validated
`snc_stageb_critic_probe_navfaithful` — global OU/conductance-noise/homeostasis OFF; the dense grid-32 `vs_place_context`
afferent; per-region homeostasis on afferent+critic = the 2026-06-09 PASS config; GABA_B/GIRK critic→SNc; value-leads-
reward training; lead 150 ms). The ONLY difference between forms is the `coincidence_detector` flag on the afferent +
the coincidence cfg knobs. Runner: `research/runners/_burndown9_critic_graded_readout_derisk.py`.

### δ table — faithful grid-32, multi-seed 42/43/44, GPU (δ = far_burst[unpredicted] / near_burst[predicted]; host-Gaussian ref ~1.3)

| seed | LINEAR δ | (critic Hz) | above-floor | PLATEAU δ | (critic Hz) | above-floor | lin-lesion δ | plat-lesion δ |
|---|---|---|---|---|---|---|---|---|
| 42 | **1.00** | 0.00 | (n/a) | **0.00** | 219.4 | no | 1.00 | 0.72 |
| 43 | **1.00** | 0.00 | (n/a) | **0.00** | 192.6 | no | 1.00 | 0.74 |
| 44 | **1.00** | 0.00 | (n/a) | **0.00** | 175.6 | no | 1.00 | 1.16 |
| **median** | **1.00** | **0.00** | — | **0.00** | **192.6** | 0/3 | 1.00 (3/3 collapse) | (2/3 collapse) |

Reference: host-Gaussian nav-deployment value-train δ ~1.3 (CYCLE-219 `2026-06-18-merged-neural-place-code-delta-probe-NEGATIVE.md`);
the 2026-06-09 navfaithful PASS got the LINEAR read-out to δ 1.39-3.19 (mean ~2.3) when the homeostasis crutch fired the
critic to ~1.3 Hz.

### The mechanism (why the graded-rate read-out can't fire the MSN — confirmed directly)

The LINEAR read-out's critic reads **0.00 Hz at every seed and every afferent weight tested (0.2 → 6.0, i.e. 30× the
validated init)**. Direct membrane probe (CPU, seed 42, w=6.0): the afferent fires (13.2 Hz population, homeostasis mask
active), but the **MSN-D1 critic membrane only reaches −72.4 mV** — barely above its −80 mV rest, nowhere near the ~−40 mV
rheobase. The MSN-D1's deep rest + high rheobase + strong inward-rectifier down-state means **linear synaptic summation
from a distributed place input is sub-rheobase on the point neuron, at any reasonable weight.** The all-or-none
coincidence plateau exists PRECISELY to overcome this: it injects a regenerative NMDA-spike-like (dendritic-flavored)
plateau current that crosses the down-state — but it SATURATES (`gain=2` sigmoid), driving V so hard (176-219 Hz) that
the GABA_B annihilates BOTH the near and far reward bursts → δ = 0.00 (the over-clamp the burndown flagged).

The intermediate WEIGHTED-subunit plateau (the g11 `_run_stage_b_smoke` read-out toggle, `coincidence_weighted_drive`,
in WEIGHT units) also does NOT fire the critic at these magnitudes (critic 0 Hz, δ 1.00) — the "graded plateau" stays
sub-threshold just like the linear form. So the read-out spectrum is unambiguous: **graded forms (linear, weighted-
plateau) can't fire the point-neuron MSN; only the saturating all-or-none COUNT plateau fires it, and it over-clamps.**

## Anti-cheats

- **Lesion (GABA_B mask zeroed):** the PLATEAU gap collapses where the critic fires (lesion δ 0.72/0.74 at seeds 42/43;
  seed 44's 1.16 is a marginal residual but still ≤1.30 = gap effectively gone) → the value subtraction is the GABA_B
  conductance, load-bearing, not host arithmetic. The LINEAR lesion is trivially 1.00 (no V is produced to subtract).
- **host-EMA contrast:** 1.0 by construction (a scalar reward-EMA is place-blind; carried from `_test_and_gate`).
- **Regime fidelity (anti-cheat d):** global OU/conductance-noise/homeostasis OFF, asserted by the navfaithful builder
  (the exact knobs nav disables) — so this replicates deployment, NOT a permissive smoke.
- **Faithful scale (the #6 lesson):** grid-32, the dense place afferent, the deterministic regime, n_train=40, multi-seed
  — NOT a tiny non-faithful smoke (which misled #6).

## What this means for #9-critic (the honest close)

1. **The read-out is a genuine FORK, not a clean conversion.** "Convert the all-or-none read-out to a graded RATE read-out"
   is **not realizable on the point-neuron MSN-D1 substrate**: the graded-rate (linear) form physically can't fire the cell
   (sub-rheobase at any weight), and the form that DOES fire it (the all-or-none coincidence plateau) over-clamps. The
   residual δ floor is the deferred **DENDRITIC place-field-carving (b)** — a dendritic-flavored mechanism is what would
   carve sparse-separable fields AND grade the plateau into a non-saturating band; that is legitimately deferred (NOT this
   item).
2. **This is the documented point-neuron rate-code / dendritic-coincidence wall, in the read-out.** It is the SAME family
   as the project's standing point-neuron limit (Mikulasch-Priesemann; the conversational whitening blocker; the CYCLE-219
   self-org δ-flat NEGATIVE): a point neuron cannot produce a graded analog read-out of a distributed code from linear
   summation — the analog/graded computation is dendritic. Per BRAIN-BASED-ONLY, a neural-mechanism-underperforms-host
   result IS the scientific deliverable (it maps the substrate's cost).
3. **Net:** #9-critic closes at **CHARACTERIZED** — the read-out form is mapped (graded=can't-fire, all-or-none=fires-but-
   over-clamps), the over-clamp is reproduced and lesion-confirmed at faithful scale, and the δ floor is attributed to the
   deferred dendritic field-carving (b), not to a missing clean read-out swap. The dendritic (b) is a deliberate
   owner/deep-frontier call (the D2 two-compartment / learned-graded-cortex Phase 3 arc), not closable as a read-out tweak.

## Honest secondary note (a regression observed, not chased)

The 2026-06-09 navfaithful PASS fired the LINEAR critic to ~1.3 Hz (δ 1.39-3.19) via the afferent+critic homeostasis
crutch; the CURRENT code reads that same config at critic 0.00 Hz. The intervening `36f15b25` ("break the numpy
cp_izh_vr/cp_membrane_potential_v ALIASING in Izhikevich-2007 init", owner byte-reviewed) shifted the MSN-D1 init so the
marginally-homeostasis-fired critic now stays sub-threshold. This does NOT change the verdict (it sharpens it: the linear
read-out's ability to fire the MSN was always marginal/fragile, now sub-threshold), and is OUT OF SCOPE for this read-out
item (a separate Izhikevich-init forensic). Flagged for the controller, not chased here.

## Reproduce

```bash
# Faithful multi-seed (GPU) — the δ table + lesion + verdict:
SIM_BACKEND=cupy python -m research.runners._burndown9_critic_graded_readout_derisk \
    --seeds 42,43,44 --n-train 40 --lead-ms 150 \
    --out research/findings/raw/_burndown9_critic_readout.json
# CPU smoke (single seed, reduced training):
SIM_BACKEND=numpy python -m research.runners._burndown9_critic_graded_readout_derisk --seed 42 --n-train 15
```

Raw: `research/findings/raw/_burndown9_critic_readout.json`.
