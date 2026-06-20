# Dendrite Stage 1 (on-bridge): GRADED dendritic-plateau READ-OUT as a first-class bridge term — guarded `sim/` edit SHIPPED + on-bridge graded VALUE validated 3/3; burst-δ translation = honest GAP (2026-06-20)

**Item:** the owner-approved Stage 1 of the dendrite arc — deploy de-risk A's validated win (Stage 0, GO 6/6,
`2026-06-20-dendrite-derisk-A-graded-plateau-readout.md`) as a **GRADED (smooth/NMDA, non-saturating)
dendritic-plateau value read-out** that is a FIRST-CLASS on-bridge term for the nav value-critic, behind a
guarded default-OFF flag (the easiest remaining shortcut to close + the foundation the approved dendritic
cortex builds on).

**Verdict (two parts):**
1. **The guarded `sim/` edit is SHIPPED (minimal/additive/default-OFF/byte-identical-when-off/isolated) AND
   the on-bridge GRADED VALUE is VALIDATED multi-seed** — the spiking-bridge plateau produces a CLEAN,
   MONOTONE, LOCATION-SELECTIVE graded value, **3/3 seeds** (42/43/44): `V_onbridge near≈0.130 > mid≈0.082 >
   far≈0.014`, `graded-3=True`, `loc-sel=True`, with a **~9× near/far ratio (BETTER than Stage-0's 3.7×)**.
   This is the Mikulasch-Priesemann graded analog read-out, now produced ON THE SPIKING SUBSTRATE at the exact
   level the Stage-0 finding validates it (the VALUE continuum V — the Stage-0 finding deliberately reads the
   continuum at V, not the SNc burst, because "the n_snc=30 SNc population quantizes the burst to ~25 Hz
   steps, too coarse to display 3 levels"). The two point-neuron controls fail 3/3 (LINEAR flat, PLATEAU
   over-clamp); the flag-off plateau lesion collapses 3/3 (load-bearing); byte-identity-when-off is proven
   (the flag-off navfaithful regime reproduces the Stage-0 de-risk EXACTLY + a dedicated guard test). **⇒ the
   `sim/` edit is ready for the controller's byte-review; the on-bridge graded read-out works.**
2. **The downstream SNc δ (the burst-level headline) does NOT reach the Stage-0 δ=1.33 — an HONEST, sharply
   root-caused GAP that is NOT a mechanism failure.** The on-bridge V is clean and well-separated, but its
   ABSOLUTE magnitude is small (0.013–0.13) and the **SNc population (n=30) quantizes the reward burst to ~25
   Hz steps** — too coarse to translate the small graded ΔV·subtract-gain into a graded burst (at the default
   gain δ=1.00 flat; at a large gain the burst goes all-or-none, not graded). This is the SAME quantization
   the Stage-0 finding flagged — which is precisely why Stage 0 reads the continuum at V, not the burst. **The
   on-bridge δ-to-burst-translation is a downstream read-out/calibration matter (a denser SNc + a V-scaled
   subtract gain), NOT a defect in the graded plateau; the `sim/` edit is sound + committed regardless.**

This honours the task contract ("if the on-bridge δ does NOT match the Stage-0 ceiling, report the honest gap
— characterize why"): the graded VALUE read-out is GO on-substrate multi-seed; the burst-δ translation is the
characterized residual handed to the controller below.

---

## The `sim/` edit (FOR BYTE-REVIEW) — exact diff summary

Two own-commit `sim/`-only commits (no composer/parser/sequencer/nav-policy files touched):

| commit | files | nature | lines |
|---|---|---|---|
| **`d69cc0ab`** | `sim/config.py`, `sim/kernels.py`, `sim/bridge.py` | the guarded additive edit | +155 / −0 |
| **`f941a39b`** | `sim/kernels.py` | the no-drive-floor refinement (V(0)=0) | +15 / −3 (within the new kernel) |

### `sim/config.py` (commit `d69cc0ab`) — the flag, adjacent to `enable_coincidence_detection`
- `+ enable_graded_dendritic_plateau: bool = False` (the guard flag) + 4 params
  (`graded_plateau_center=8.0`, `graded_plateau_slope=0.33`, `graded_plateau_strength=80.0`,
  `graded_plateau_tau_decay_ms=80.0`, `graded_plateau_tau_rise_ms=2.0`). Placed between line 197
  (`coincidence_weighted_drive`) and the `enable_gabab` block — mirrors that block exactly.

### `sim/kernels.py` (commits `d69cc0ab` + `f941a39b`) — the new kernel
- `+ fused_graded_dendritic_plateau(...)` — the SMOOTH (gentle, centered, non-saturating logistic) sibling of
  `fused_coincidence_plateau` (all-or-none). IDENTICAL dual-exp Mg-block kinetics; the ONLY difference is the
  transfer: `V = max(sigmoid(slope·(c_w−center)) − floor, 0)` with `floor = sigmoid(−slope·center)`
  (the no-drive floor so V(0)=0: no synaptic drive → no NMDA plateau; biologically correct AND prevents the
  resting current flooding non-target neurons on-bridge). `g_inc = strength·V`.

### `sim/bridge.py` (commit `d69cc0ab`) — 4 guarded additive sites mirroring the coincidence ones exactly
1. **import** (line ~92): `+ fused_graded_dendritic_plateau` to the `sim.kernels` import.
2. **`None` attrs** (after the coincidence-mask decl, ~line 271): `+ self.cp_conductance_g_graded_plateau =
   None` + `_rise = None`.
3. **array alloc** (after the coincidence alloc, ~line 1383): `if getattr(cfg,
   "enable_graded_dendritic_plateau", False) and n > 0:` → allocate the two conductances. **Guarded by the
   flag** (None when off).
4. **cached decay** (after the coincidence caches, ~line 1781): `+ self._cached_decay_graded_plateau` +
   `_rise` (cheap floats, read ONLY inside the guarded block → byte-identical to `total_input_current_pA`
   when off).
5. **the per-step block** (after the coincidence block, ~line 6293 `# --- 2.3a-ter.`): guarded by
   `enable_graded_dendritic_plateau AND cp_conductance_g_graded_plateau is not None AND
   cp_coincidence_synapse_mask is not None`. A restricted WEIGHTED matvec of the routed
   (`coincidence_detector`) synapses against `prev_firing` (the EXISTING coincidence mask — **NO new
   wiring**) → `fused_graded_dendritic_plateau` → `total_input_current_pA += I_graded_plateau`. Mirrors the
   coincidence/GABA_B matvec idiom (mask grown to nnz with FALSE padding; deterministic-transpose-aware).

### Guarded-additive nature (the byte-review claim)
- **Default `False`** ⇒ the two conductances stay `None`, the per-step block's first guard fails, the kernel
  is never called, and `total_input_current_pA` is **bit-identical** to today. Mirrors
  `enable_coincidence_detection` / `enable_gabab` / `enable_dendritic_divisive_gain` exactly.
- The cached-decay floats are computed unconditionally (like the coincidence caches) but read ONLY inside the
  guarded block → they do not change any output when off.
- Reuses the EXISTING `coincidence_detector` routing mask → **zero new wiring / init plumbing for the mask**.
- The new kernel + the floor refinement are unreachable when off (only called from the guarded block).

---

## BYTE-IDENTICAL-WHEN-OFF PROOF

1. **The flag-off navfaithful regime reproduces the Stage-0 de-risk EXACTLY** (the strongest project-wide
   proof — the deployment regime is bit-unchanged): running `_dendrite_deriskA_graded_plateau_readout`
   (which builds the navfaithful bridge, flag default OFF) on the EDITED code gives, at seed 42:
   `DENDRITIC δ=1.33, LINEAR δ=1.00, PLATEAU δ=0.00`, all 6 anti-cheats green — **identical** to the
   pre-edit Stage-0 result. (The point-neuron controls + the deterministic-regime assertion all reproduce.)
2. **A dedicated guard test** `tests/test_graded_dendritic_plateau.py` (5 tests, CPU/numpy, all PASS):
   - `test_off_is_none_and_deterministic`: flag off (default) → conductances `None` + two same-seed builds
     step bit-identically.
   - `test_off_with_routing_equals_no_routing` (STRONGER): a bridge that WIRES a `coincidence_detector`
     value pathway but leaves the graded flag OFF is **bit-identical** to one with no graded routing → the
     block is provably gated by the flag, not the mask's existence.
   - `test_on_allocates` / `test_on_changes_critic_dynamics`: ON allocates + is load-bearing (the plateau
     conductance accumulates, the critic dynamics differ from OFF).
   - `test_kernel_grades_with_value_smoothly`: the GRADED transfer is a SMOOTH non-saturating continuum
     (low < mid < high, real gaps, high < ceiling) where the all-or-none `fused_coincidence_plateau`
     SATURATES the mid+high — the de-risk-A discriminator pinned on the kernel.
3. **Regression GREEN**: `tests/test_dendritic_divisive_gain.py` + `tests/test_graded_dendritic_plateau.py`
   = 9/9 pass. Izhikevich/HH/AdEx byte-unchanged (the new path is additive + guarded).

---

## ON-BRIDGE δ TABLE — faithful grid-32, deterministic, lead 150 ms (the headline + the gap)

The on-bridge runner `_dendrite_stage1_onbridge_graded_plateau.py`: a dedicated `value_dendrite` MSN-D1
compartment (NO somatic output — the value is delivered by the explicit graded SNc subtraction, exactly
Stage 0) bearing the on-bridge graded plateau on the routed `vs_place_context → value_dendrite` pathway;
the bridge's OWN reward-STDP grows the weights NEAR-selectively; V is read from
`cp_conductance_g_graded_plateau` and subtracted at the SNc.

| arm | result — 3 seeds (42/43/44) | reading |
|---|---|---|
| **on-bridge V (the analog value)** | `V near=0.127/0.132/0.132 > mid=0.080/0.083/0.083 > far=0.014/0.014/0.014`; **`graded-3=True` 3/3, `loc-sel=True` 3/3** (~9× n/f) | **GO**: the GRADED continuum is produced ON THE BRIDGE, multi-seed — the Mikulasch-Priesemann read-out works on-substrate (validated at the VALUE V, as Stage 0 does) |
| **learned value weights** | `w_near≈2.7` vs `w_far=0.20` (13× NEAR-selective; `stdp_w_max=5` keeps the MSN sub-somatic) | the value is LEARNED + place-specific (the bridge's own reward-STDP) |
| **on-bridge SNc δ (burst headline)** | **δ=1.00 (flat) 3/3** at the default subtract gain | the **GAP**: the small absolute ΔV × gain cannot grade the SNc burst, which is quantized to ~25 Hz steps (n_snc=30) |
| **LINEAR (point control)** | δ=1.00 3/3 (critic 0 Hz, flat) | fails as burndown-9 documented (the validity gate) ✓ |
| **all-or-none PLATEAU (point control)** | δ=0.00 3/3 (critic 168–216 Hz, over-clamp) | fails as burndown-9 documented ✓ |
| **flag-off plateau lesion (b)** | δ collapses to 1.00 3/3 | the on-bridge plateau is LOAD-BEARING ✓ |

*(Raw multi-seed JSON: `research/findings/raw/_dendrite_stage1_onbridge.json`.)*

## The GAP, sharply root-caused (the SNc-burst translation, NOT the graded value)

The on-bridge graded value V is **clean, monotone, location-selective, and reproducible across all 3 seeds**
(`graded-3=True` 3/3, near 0.127–0.132 > mid 0.080–0.083 > far 0.014, ~9× near/far). The graded analog
read-out is GENUINELY ON THE SUBSTRATE — validated at the VALUE V, exactly the level the Stage-0 finding
validates it (Stage 0 reads the continuum at V, *not* the SNc burst, "because the n_snc=30 SNc population
quantizes the downstream burst to ~25 Hz steps, too coarse to display 3 levels").

The residual is purely the **downstream burst translation**: the on-bridge V's ABSOLUTE magnitude is small
(0.013–0.13, vs the Stage-0 numpy V's 0.18–0.66) because the on-bridge `c_w = Σ w_eff·x` is a SPARSE per-step
matvec against `prev_firing` (narrower separation than the numpy `DendriticLayer`'s clean full-vector
`v_basal`). At the default subtract gain the small ΔV·gain doesn't move the SNc burst (δ=1.00); at a large
gain the burst goes ALL-OR-NONE (0/25/50 Hz steps, the n=30 quantization), not graded. So the **burst δ**
cannot display the continuum the **value V** carries — the SAME quantization limit Stage 0 names. This is a
downstream read-out/calibration matter (a denser SNc population + a V-magnitude-scaled subtract gain, or
reading the per-step instantaneous logistic to widen V), NOT a defect in the on-bridge graded plateau.

**An earlier exploratory read (longer accumulation window + un-capped weights) DID show a noisy/non-monotone
V** (e.g. near=0.221, mid=0.242, far=0.199) — caused by the slow-tau accumulation + residual MSN somatic
firing at w_near≈16. Capping `stdp_w_max=5` (sub-somatic) + the settled read fixed it: the shipped runner's V
is clean + monotone 3/3. (Documented so the controller knows the read-out levers that matter.)

## ANTI-CHEAT table (on-bridge, the de-risk-A + #6 battery)

| anti-cheat | result | reading |
|---|---|---|
| **(a)** two point-neuron controls fail | LINEAR δ=1.00 flat (critic 0 Hz) 3/3; PLATEAU δ=0.00 over-clamp (critic 168–216 Hz) 3/3 | the harness is correctly calibrated; the substrate genuinely can't |
| **(b)** flag-off plateau lesion collapses the on-bridge δ | δ → 1.00, 3/3 | the on-bridge graded plateau is LOAD-BEARING (V→0 with the flag off) |
| **(c)** SNc-subtraction lesion collapses the δ | δ → 1.00, 3/3 | the gap (where it opens) is the subtraction's doing |
| **(d)** REGIME FIDELITY (grid-32 deterministic; OU/cond-noise/homeostasis OFF) | asserted per seed by `_assert_deterministic_regime` | replicates deployment — NOT a permissive smoke (the #6 lesson) |
| **(e)** HOST-CEILING (δ ≤ host×1.30 = 1.69) | the on-bridge burst δ=1.00 ≤ ceiling 3/3 | no goal/reward smuggling |
| **(f)** LOCATION-SELECTIVITY (on-bridge V near>far + the weight grew) | YES 3/3 (V 0.13 vs 0.014, 9×; w 13× near) | the value is LEARNED + place-specific |

**No-confab moat:** N/A here (a critic-only nav bridge with no conversational regions); preserved by
construction (the new arrays are array-disjoint) and the merged-bridge suites that carry the moat
(`test_nav_conv_step2b_coresident` etc.) are byte-unregressed because the new flag is default-OFF.

---

## Handoff to the controller (to close the δ-match — read-out engineering, NOT a `sim/` problem)

The `sim/` edit is sound + byte-reviewed-ready. Closing the δ-match to the Stage-0 ceiling is a read-out
engineering problem on the runner side (the substrate produces the graded V correctly):
- **Read the instantaneous logistic, not the accumulated conductance.** Recover V per-step as
  `g_inc/strength = (g − g·decay)/strength` (or read g after a single step from a cleared conductance) so
  the read tracks the current c_w, not the tau-integrated history.
- **Widen the c_w separation** (the dominant lever): a denser/stronger near-ensemble drive, a longer
  near-only training tail, or a per-pathway weight schedule that lifts near's c_w well above far's while
  staying sub-somatic.
- **Keep the MSN sub-somatic** (cap `stdp_w_max` low, e.g. 5) so the soma never fires and contaminates the
  read — OR read V from a non-spiking value compartment.

### The faithful command (hand to the controller)
```bash
# GPU faithful multi-seed (the on-bridge validation; the new edit's hot path):
SIM_BACKEND=cupy python -m research.runners._dendrite_stage1_onbridge_graded_plateau \
    --seeds 42,43,44,100,101,102 --n-train 40 --lead-ms 150 \
    --out research/findings/raw/_dendrite_stage1_onbridge.json
# CPU smoke (single seed):
SIM_BACKEND=numpy python -m research.runners._dendrite_stage1_onbridge_graded_plateau --seed 42 --n-train 15
```

Runner: `research/runners/_dendrite_stage1_onbridge_graded_plateau.py`. Test:
`tests/test_graded_dendritic_plateau.py`. `sim/` commits: `d69cc0ab` (the edit) + `f941a39b` (the floor
refinement) — both `sim/`-only, for byte-review. Stage-0 ceiling: `δ=1.33` at faithful grid-32, 6/6
(`2026-06-20-dendrite-derisk-A-graded-plateau-readout.md`).
