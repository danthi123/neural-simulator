# N9 TD cue-shift — B-2 conductance-derivative (PROTECTED `sim/` edit): PARTIAL — the edit RESOLVES the B-3 value-growth blocker multi-seed, but the burst MIGRATION does not occur

**Date:** 2026-06-10
**Type:** PROTECTED `sim/` edit (additive, guarded, default-OFF byte-identical) + CPU probe extension + multi-seed validation. Owner-approved as ONE bounded attempt.

**CONTROLLER DISPOSITION (2026-06-10): the protected `sim/` edit was byte-reviewed line-by-line (clean: additive, guarded, default-off, the byte-proof COMBO hash `e728d7f19d99b5b4` confirms off==baseline) and then REVERTED — NOT shipped. Rationale: the owner approved one bounded attempt with the explicit stop-rule "if it walls, bank the negative and move on"; the migration (the edit's purpose) walled, so per "move on" the edit is not accumulated in `sim/`. This finding IS the deliverable — it documents the genuine value-growth advance, the characterized migration boundary, AND the exact diff below (so the mechanism is reconstructable if a future critic architecture — a multi-channel critic or the A-CSC tapped-delay-line — wants it). The validated diff also lives in local git history (commit `87ff8925`). The runner-side `--td-conductance` probe mode was reverted with the edit (it depends on the reverted flag).**
**Design:** `docs/plans/2026-06-10-N9-TD-cue-shift-design.md` — option **B-2** (§2.2 the strictly-minimal single-conductance leaky-derivative; §3 the protected-edit scope + byte-identity protocol; §4 validation; §5 anti-cheats; §6 the three honest outcomes).
**Fixes the blocker in:** `research/findings/2026-06-10-N9-TD-cue-shift-B3-cheap-first-derisk-NEGATIVE.md` — the zero-edit route forced DENSE critic firing → STDP eligibility net-negative → the cue value SHRANK (so migration was structurally impossible). B-2 decouples the value-derivative *delivery* from the critic's firing *density*.

---

## Terms (defined once)

- **CS / US** — conditioned stimulus (the predictive cue) / unconditioned stimulus (the primary reward).
- **SNc** — substantia nigra pars compacta, the midbrain dopamine pool; its windowed firing rate IS the dopamine signal here.
- **Critic** — the `striosome_value` medium-spiny-neuron pool that learns and represents the state value V.
- **Rescorla-Wagner (R-W)** — the one-step prediction error δ = r − V (no time axis; what the circuit already computes in spikes).
- **Temporal-difference (TD)** — δ = r + γ·V(next) − V(now); the bracketed term is the **value derivative / bootstrap** that R-W lacks and that produces the **cue-shift** (the dopamine burst migrating from the reward onto the cue — Schultz 1997, the one canonical dopamine signature the circuit does not yet show).
- **GABA_B / GIRK** — a slow (tau ≈ 150 ms) metabotropic inhibitory conductance through a G-protein inwardly-rectifying potassium channel (reversal E_K ≈ −90 mV); the engine's sign-correct way to subtract V at the SNc membrane (already shipped, owner-approved).
- **Conductance-derivative** — the bootstrap delivered as the *temporal derivative of the GABA_B value channel*, computed in conductance at the SNc membrane, decoupled from the critic's firing density.
- **Value-growth gate** — does the cue→striosome value/weight RISE across trials? (the prerequisite B-3 failed; migration is impossible without it.)

---

## One-paragraph result

**B-2 is a PARTIAL: the protected conductance-derivative edit RESOLVES the value-growth prerequisite that B-3 isolated as the true blocker — robustly, multi-seed — but the full Schultz burst MIGRATION still does not occur on this single-channel rate critic.** With the derivative read from CONDUCTANCE at the SNc membrane (a slow leaky-EMA of `g_gabab`; the band-passed difference is `+dV/dt`) instead of from a dense-firing disinhibition relay, the critic fires SPARSELY and the cue value **GROWS on all three seeds** (V on the cue: seed 42 75→103, 43 100→134, 44 77→111 Hz; the cue→striosome weight 19→22 on every seed) — the exact quantity that B-3 could only make *shrink*. The R-W `−V` signature also strengthens (the US burst shrinks as value grows). **But the burst does not migrate onto the cue: migration r = −0.144 / +0.084 / +0.219 (seeds 42/43/44), not sign-consistent and far from the < −0.7 bar; the SNc peak stays pinned at the US (bin 6) on every seed.** The operating space was mapped thoroughly (reward gain 120–250, GIRK cap 3–20, derivative gain 1–8, slow tau 350–4000, 20–50 trials) and the failure is characterized as an *inherent conflict on a single-channel rate critic*: (1) the conductance-derivative tracks the rising EDGE of `g_gabab`, so the cue burst SHRINKS as the slow EMA accumulates (it does not grow with the learned value); (2) raising the derivative gain to force a larger cue burst re-triggers the B-3 eligibility poisoning (dense cue-burst firing → net-negative eligibility → value stops growing); (3) a `−V` level strong enough to transfer the US burst to the cue also clamps the SNc tonic to silence (no live baseline for migration/dip). The two anti-cheats confirm the cue-time activity is genuinely synaptic: a cue-pathway lesion silences the cue (V→0, 3/3) and removes the cue burst while the innate US reflex survives, and an unpaired-timing control shows no migration (3/3). **Verdict: PARTIAL — the protected edit advances the substrate (value-growth resolved, the B-3 blocker cleared) but does not deliver the migration; per the STOP-RULE this is the deliverable, NOT an escalation to the tapped-delay-line (A-CSC).**

---

## The protected `sim/` edit (additive, guarded, default-OFF byte-identical)

The edit is the strictly-minimal **B-2** form (design §2.2/§3): a single slow leaky-EMA of the existing GABA_B value conductance, whose band-passed difference is the value derivative, delivered as a depolarizing current at the SNc membrane. It is an exact structural mirror of the already-shipped, owner-approved GABA_B/GIRK block.

**`sim/config.py` (+15 lines, after the GABA_B config block at `:200`):**
- `enable_td_value_derivative: bool = False` (the opt-in guard)
- `td_slow_tau_ms: float = 400.0` (the slow-EMA decay constant; > `gabab_tau_decay` ~150 so the difference is a derivative)
- `td_derivative_gain: float = 1.0` (scales the derivative current)

**`sim/bridge.py` (+52 lines, four sites, all guarded / always-cheap):**
- `__init__` `:248` — `self.cp_conductance_g_gabab_slow = None` (mirror of `cp_conductance_g_gabab`).
- alloc `:1244` — guarded by `if getattr(cfg, "enable_td_value_derivative", False) and n > 0:` → `cp_conductance_g_gabab_slow = cp.zeros(n, dtype=cp.float32)` (mirror of the GABA_B alloc).
- decay cache `:1639` (and the checkpoint-load recompute `:7299`) — `self._cached_decay_gabab_slow = exp(-dt_ms / td_slow_tau_ms)`; always cached (a cheap float), only USED inside the guard (caching it unconditionally does not change behaviour when the flag is off, mirroring how `_cached_decay_gabab` / the NMDA-recurrent decays are always cached).
- per-step `:5909–5932`, INSIDE the existing GABA_B block (so it runs only when `enable_gabab` is active) and guarded by `if getattr(cfg, "enable_td_value_derivative", False) and self.cp_conductance_g_gabab_slow is not None:`
  - leaky integrator: `g_slow = g_slow*decay_slow + g_gabab*(1 − decay_slow)` (a slower-decaying EMA of the post-update `g_gabab`),
  - derivative current: `I_td_deriv = td_derivative_gain * (g_gabab − g_gabab_slow) * (E_exc − V)`, with `E_exc = cfg.syn_reversal_potential_e = 0 mV` (depolarizing), added to `total_input_current_pA`.
  - Sign (verified): on a value RISE, `(g_gabab − g_gabab_slow) > 0` and `(E_exc − V) > 0` (V ≈ −60 mV) → positive → the SNc depolarizes (a burst at the cue); on a value FALL → negative → hyperpolarize (the omission dip); flat → ≈ 0. ADDITIVE on top of the existing `−V` GABA_B subtraction.

No new kernel, no new mask (the derivative reads `g_gabab`, the GABA_B-mask-restricted striosome→SNc channel, so it naturally targets the SNc). Diffstat: `sim/bridge.py +52`, `sim/config.py +15`, `research/runners/snc_stageb_critic_probe.py +402/−1` (the `--td-conductance` B-2 probe mode + its cue-lesion variant).

---

## Byte-identity proof (off == baseline)

**Static:** the three new `cfg.*` fields are referenced ONLY inside `if enable_td_value_derivative` (config defaults + the one guarded per-step block + the always-cheap decay cache, which is a float read used only inside the guard); the new array defaults `None`; one guarded alloc + one guarded per-step block. Nothing outside the guard is touched.

**Dynamic off == baseline:** a fixed-seed harness (`research/findings/raw/_b2_byteproof_harness.py`) runs (a) a 200-neuron Izhikevich kernel smoke for 50 steps and (b) a Stage-B critic warm-up with `enable_gabab=True` (so the GABA_B block — beside which the B-2 guard sits — actually runs) for 60 steps, hashing `cp_membrane_potential_v` + `cp_firing_states` every step. With the flag default-OFF:

| | pre-edit (HEAD `f73e3954`) | post-edit (flag default-OFF) |
|---|---|---|
| IZH first / last step hash | `69173acdbd9b9418` / `dc4e09443f7d9ed1` | `69173acdbd9b9418` / `dc4e09443f7d9ed1` |
| Stage-B first / last step hash | `f191cdede9204768` / `e90d84bd35df5e6b` | `f191cdede9204768` / `e90d84bd35df5e6b` |
| **COMBO (all 50 + 60 step hashes)** | **`e728d7f19d99b5b4`** | **`e728d7f19d99b5b4`** |

Bit-identical. The guarded block is unreached and `total_input_current_pA` is byte-identical when the flag is False.

**Regression-absent (R-A):** the relevant CPU test suites pass UNCHANGED with the edit present + OFF — verified by running them at both the pre-edit (stashed) and post-edit working tree under `SIM_BACKEND=numpy`: `test_determinism / test_regions / test_neuromodulators / test_backend` give the **identical `16 failed, 115 passed`** at both (every failure is a pre-existing CuPy-required test failing under the numpy backend, NOT introduced by the edit); `test_kernels_cpu` + `test_td_critic_no_harm` pass (36 passed).

**Expected protected-guard failures (CORRECT for an unreviewed protected edit):**
- `tests/test_td_critic_no_harm.py::test_protected_byte_untouched_across_td_critic_range` currently PASSES pre-commit (it diffs `ed880244..HEAD`, committed only). It will FAIL once this protected edit is **committed** — which is the intended behaviour: the guard means "no protected module changed since the last owner byte-approval." **Do NOT re-bump `_TD_BASE` — the controller bumps it only after the owner byte-reviews this diff.**
- `tests/test_compose_bridge_no_harm.py::test_protected_byte_untouched_across_range` was ALREADY failing PRE-edit (its stale base `e8a99a2` predates the legitimately-landed GABA_B/determinism edits; it reports `sim/bridge.py`, `sim/kernels.py`, `sim/neuromodulators.py`). My edit does not newly break it.

---

## Validation on the CPU probe (`--td-conductance`, B-2 mode)

`research/runners/snc_stageb_critic_probe.py` adds a `--td-conductance` mode (`run_td_conductance` + `run_td_conductance_lesion`). It reuses the existing Stage-B bridge, the dopamine threshold + baseline calibration, the eligibility/three-factor learning, the time-course recorder, and the lesion machinery; it differs from the B-3 `--td` mode ONLY in the derivative DELIVERY: instead of the dense-firing disinhibition relay, the value derivative is delivered by the PROTECTED conductance term on the DIRECT GABA_B `striosome_value → snc` path (so the critic fires SPARSELY), and the reward enters directly at the SNc. Run `SIM_BACKEND=numpy` (CPU, ~130-neuron bridge).

### 1. VALUE-GROWTH GATE (the prerequisite B-3 failed) — PASS 3/3

| Seed | V(cue) early → late | cue→striosome weight (final) | value_grows |
|---|---|---|---|
| 42 | 74.8 → 102.8 Hz | 21.96 (from ~19) | **True** |
| 43 | 99.7 → 133.6 Hz | 22.18 | **True** |
| 44 | 76.7 → 110.8 Hz | 22.89 | **True** |

**The cue value GROWS on every seed** — exactly what B-3 could only make *shrink* (B-3: V 70→13, 61→17, 63→22; weight 19→13 on every seed). This is the genuine, multi-seed substrate advance the protected edit delivers: decoupling the derivative from firing density lets the critic learn the value up at a sparse, STDP-causal operating point.

### 2. MIGRATION (the headline) — NEGATIVE 3/3 (0 GO + 3 PARTIAL)

| Seed | migration r (bar < −0.7) | SNc peak-bin early → late (US onset = bin 6) | US-window early → late | late_burst_at_cs |
|---|---|---|---|---|
| 42 | −0.144 | 6.0 → 6.0 | 64.9 → 58.3 Hz | ✗ |
| 43 | +0.084 | 6.0 → 6.0 | 63.4 → 58.5 Hz | ✗ |
| 44 | +0.219 | 6.0 → 6.1 | 72.8 → 61.8 Hz | ✗ |

**The burst does NOT migrate onto the cue.** The SNc peak stays at the US (bin 6) on every seed; r is not sign-consistent and nowhere near < −0.7. The US burst shrinks only slightly (the residual R-W `−V`) and never transfers. Supporting gates: `early_burst_at_us` ✓ 3/3 (the reflex), `no_gap_burst` ✓ 3/3, `omission_dip_at_reward` ✓ only on seed 44 (a tight-threshold residual), `late_burst_at_cs` ✗ 3/3 (the decisive transfer gate). Headline JSON: `research/findings/raw/_td_cue_shift_b2_3seed.json`.

Operating point (the live-tonic, value-growing config): `--snc-reward-gain 120 --gabab-conductance-max 6.0 --snc-tonic-pa 320 --td-derivative-gain 1.0 --td-slow-tau-ms 350`, 50 trials, window = 6 CS + 4 ISI + 4 post bins × 20 sub-steps.

### 3. ANTI-CHEATS (the cue-time activity is brain-based, not host)

1. **Cue-pathway lesion — decisive part PASS 3/3.** After training, zeroing the `cue → striosome` edges silences the critic on the cue (V → **0.00 Hz on all 3 seeds**), removes the cue-time SNc elevation (no_cue_burst ✓ 3/3), and the innate US reflex survives (US 57–65 Hz vs tonic ~50). The `no_dip` sub-check reads False (a tight-threshold artifact: with the cue lesioned the SNc is no longer value-clamped so its tonic JUMPS to ~50 Hz and the residual omission/baseline gap of ~12–20 Hz exceeds the 0.5 Hz bar — the cue is genuinely silenced, not a real dip). The decisive assertion — cue silenced + no cue burst + US reflex intact — holds 3/3, proving the cue-time activity is carried by the `cue→striosome→GABA_B(+derivative)` synaptic conduit, not host arithmetic. JSON: `research/findings/raw/_td_cue_shift_b2_lesion_3seed.json`.
2. **Unpaired-timing control — PASS 3/3 (no migration).** Decoupling CS and US in time gives no migration on all 3 seeds (r = +0.152 / −0.129 / −0.148, none < −0.7, peak unmoved). Honest caveat (same as B-3): because the *paired* condition already shows no migration, this control is *consistent* rather than *discriminating* here — it confirms no spurious cue-present back-channel but cannot show a paired-vs-unpaired contrast. JSON: `research/findings/raw/_td_cue_shift_b2_unpaired_3seed.json`.
3. **Provenance — PASS.** Under `--td-conductance`, the SNc drive is `tonic + reward_us(direct) + synaptic GABA_B(−V) + the synaptic conductance-derivative(+dV/dt)` ONLY — no host δ, no host γV′−V, no host value/EMA (`current_reward_signal = 0`, `reward_baseline = 0`, `enable_td_value_derivative = True`, `enable_gabab = True`, all asserted in `run_td_conductance`). Recorded in the JSON `provenance` block.

---

## Root cause (fully characterized — the inherent single-channel-critic conflict)

The operating space was swept (reward gain 120–250, GIRK cap 3–20, derivative gain 1–8, slow tau 350–4000, 20–50 trials). There is **no window where the value GROWS and the cue burst GROWS with it**, because of three coupled constraints on a single-channel rate critic:

1. **The conductance-derivative captures the rising EDGE, not the value LEVEL.** `(g_gabab − g_gabab_slow)` is large at cue onset early in learning (sharp transient) but SHRINKS across trials as the slow EMA `g_gabab_slow` accumulates a persistent baseline (the cue is presented every trial). So the cue burst *shrinks* as the value grows — the opposite of the migration signature, which needs the cue burst to GROW with the learned value. (Time-course: seed 42 CS-window 47→29 Hz while V 44→97 Hz.)
2. **Forcing a larger cue burst (higher derivative gain) re-triggers the B-3 eligibility poisoning.** At derivative gain 3–8 the cue burst is large (CS-window 74–222 Hz), but that dense SNc cue-burst makes the STDP eligibility net-negative again and the value STOPS growing (value_grows flips False) — the identical mechanism that killed B-3, just relocated from the relay to the derivative.
3. **A `−V` strong enough to transfer the US burst also kills the tonic.** Making `g_gabab` (hence `−V`) large enough that the value cancels the reward at the US (so the US burst vanishes and could "transfer" to the cue) also clamps the SNc tonic to silence (last-trial AND baseline time-courses go all-zero) — leaving no live baseline for a migration peak or an omission dip. The GIRK cap that keeps the SNc alive at tonic simultaneously caps `−V` too low to cancel the US.

This is precisely the design's anticipated honest outcome (§6.1: a small rate-coded critic's value estimate is noisy and the single-step derivative is fragile; §6.2(ii)/(iii)). The conductance-derivative **does** remove the B-3 blocker (value growth), but the single-channel rate critic cannot simultaneously (a) keep the SNc alive at tonic, (b) grow the value, and (c) translate a growing value into a growing, transferring cue burst.

---

## Honest three-outcome placement (design §6.2)

- (i) Clean migration — **not reached** (r ≈ 0, peak unmoved, no transfer).
- (ii) **Partial / graded migration — this is the closest placement, with a precise scope.** The protected edit DELIVERS the prerequisite (value growth, 3/3) and strengthens the R-W `−V` (US burst shrinks) — a real substrate advance over B-3 — but the *peak migration itself* does not occur (it is a PARTIAL on the value-growth + R-W axes, NEGATIVE on the migration axis).
- (iii) No migration on the migration axis — the deliverable is the value-growth fix + the characterized boundary above.

**Net: PARTIAL.** The protected conductance-derivative edit is the right mechanism to clear the B-3 blocker (value growth, multi-seed) — but the burst migration is blocked by an inherent single-channel-rate-critic conflict, not by the derivative delivery, so the edit does not on its own deliver the Schultz cue-shift.

---

## Verdict + recommendation

**VERDICT: PARTIAL.** The B-2 protected conductance-derivative edit (additive, guarded, default-OFF byte-identical — proven) RESOLVES the value-growth prerequisite that the B-3 zero-edit route failed (V on the cue grows 3/3; the cue weight grows 3/3), and strengthens the R-W `−V` signature (the US burst shrinks). **But the Schultz burst MIGRATION does not occur multi-seed** (r = −0.144/+0.084/+0.219, peak pinned at the US 3/3) — blocked by an inherent conflict on a single-channel rate critic (the derivative tracks the rising edge not the value level; a larger cue burst re-poisons eligibility; a `−V` strong enough to transfer the US kills the tonic). The anti-cheats confirm the cue-time activity is synaptic (cue lesion → V→0 + no cue burst + US reflex intact, 3/3; unpaired → no migration, 3/3).

**RECOMMENDATION (per the STOP-RULE): report this PARTIAL and STOP — do NOT escalate to the tapped-delay-line (A-CSC) or any larger build in this attempt.** The value-growth gate was the cheap prerequisite B-3 isolated; the conductance-derivative edit passed it, which is the substrate advance worth banking. The residual — translating a *grown* value into a *growing, transferring* cue burst — is what the migration needs, and the characterization shows a single-step conductance-derivative on a single-channel rate critic cannot supply it (the edge-vs-level problem + the tonic/`−V` floor-ceiling conflict). The literal Montague-Dayan-Sejnowski mechanism that solves this (a complete-serial-compound tapped-delay chain that back-propagates value one tap per trial, design §2.1/§6.3, A-CSC) is a larger, higher-variance build and is the documented next option IF the owner chooses to pursue the cue-shift further — but it is explicitly OUT of scope for this bounded attempt.

**Status of the protected edit:** committed LOCALLY, tagged FOR OWNER BYTE-REVIEW, NOT pushed. The off==baseline hash evidence (COMBO `e728d7f19d99b5b4`, identical) and the R-A evidence are above. `test_td_critic_no_harm`'s byte-untouched guard will fail after this commit (correct for an unreviewed protected edit); the base SHA is NOT re-bumped here.

---

## Artifacts

- Protected edit: `sim/config.py` (+15), `sim/bridge.py` (+52, sites `:248 / :1244 / :1639 / :5909–5932 / :7299`).
- Probe (extended): `research/runners/snc_stageb_critic_probe.py` (`--td-conductance`, `run_td_conductance`, `run_td_conductance_lesion`, `_run_td_conductance_mode`).
- Byte-proof harness: `research/findings/raw/_b2_byteproof_harness.py` (COMBO `e728d7f19d99b5b4` pre==post).
- Headline 3-seed: `research/findings/raw/_td_cue_shift_b2_3seed.json`.
- Cue-lesion anti-cheat 3-seed: `research/findings/raw/_td_cue_shift_b2_lesion_3seed.json`.
- Unpaired anti-cheat 3-seed: `research/findings/raw/_td_cue_shift_b2_unpaired_3seed.json`.
- Run (headline): `SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe --td-conductance --seeds 42,43,44 --n-train 50 --snc-reward-gain 120 --gabab-conductance-max 6.0 --snc-tonic-pa 320 --td-derivative-gain 1.0 --td-slow-tau-ms 350` (+ `--td-lesion-cue` / `--td-unpaired`).
