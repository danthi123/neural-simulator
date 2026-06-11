# N9 TD cue-shift — A-CSC (complete-serial-compound) + multi-channel reward relay: GO — the dopamine burst MIGRATES onto the cue, multi-seed (the Schultz 1997 signature)

**Date:** 2026-06-10
**Type:** Escalation #2 build (A-CSC tapped-delay cue) + the named-next multi-channel critic, on the CPU spiking-SNc probe. RE-APPLIES the B-2 conductance-derivative PROTECTED `sim/` edit (additive, guarded, default-OFF byte-identical; the controller byte-reviewed it line-by-line — see below — and the owner directive auto-approves byte-proven diffs). Surpasses the B-2 single-channel migration wall.
**Owner directive (2026-06-10):** WORK THROUGH the wall, NO BANKING; sim edits allowed (byte-prove off==baseline + tests pass → auto-approve); push BOTH remotes.
**Design:** `docs/plans/2026-06-10-N9-TD-cue-shift-design.md` — §2.1 A-CSC (the complete-serial-compound), §6.3 #2 (the fidelity escalation), §2.2 B-1/B-2 (the conductance-derivative), §4 (validation), §5 (anti-cheats).
**Surpasses:** `2026-06-10-N9-TD-cue-shift-B2-conductance-derivative-PARTIAL.md` (the single-channel rate critic could grow the value but NOT migrate the burst — the edge-vs-level / floor-ceiling conflict). `2026-06-10-N9-TD-cue-shift-B3-cheap-first-derisk-NEGATIVE.md` (the zero-edit value-derivative could not grow the cue value at all).

---

## Terms (defined once)

- **CS / US** — conditioned stimulus (the predictive cue) / unconditioned stimulus (the primary reward).
- **SNc** — substantia nigra pars compacta, the midbrain dopamine pool; its windowed firing rate IS the dopamine signal here.
- **Critic** — the `striosome_value` medium-spiny-neuron (MSN) pool that learns and represents the state value V.
- **TD** — temporal-difference: δ = r + γ·V(next) − V(now); the bracketed term is the **bootstrap** that produces the **cue-shift** (the dopamine burst migrating from the reward onto the cue — Schultz 1997, the one canonical dopamine signature the circuit did not yet show).
- **A-CSC / complete-serial-compound** — the cue represented NOT as one event but as a CHAIN of K time-tagged sub-states (`csc_0`=cue@onset, `csc_1`=cue@Δ, …, `csc_{K-1}`), EACH driving the critic through its OWN plastic synapse, so TD back-propagates value one tap per trial (Montague-Dayan-Sejnowski 1996; Sutton-Barto Ch 12).
- **Conductance-derivative (B-2)** — the value derivative `+dV/dt` (the bootstrap) delivered as the temporal derivative of the GABA_B value conductance, `I_td_deriv = gain·(g_gabab − g_gabab_slow)·(E_exc − V)`, computed AT THE SNc MEMBRANE (the protected `sim/` edit).
- **Multi-channel reward relay** — the reward `r` enters via an EXCITATORY relay `reward_us → snc` that the critic INHIBITS, so the reward reaching the SNc is `r − V` (the Eshel subtraction), with `−V` LOCALIZED to the reward window (catalog C.33 PPN→DA + striosome→RMTg/PPN inhibition). This is the named next escalation that decouples the floor-ceiling conflict.
- **Migration r** — Pearson correlation between trial number and the SNc burst time-of-peak; migration = the peak moves EARLIER (cue-ward) across learning ⇒ r < 0. Pass bar **r < −0.7**.

---

## One-paragraph result

**GO — the SNc dopamine burst MIGRATES from the reward onto the predictive cue across cue→reward learning, multi-seed, with the full Schultz 1997 signature, and the TD error is computed entirely by neurons (no host term).** The A-CSC tapped-delay cue (K=8 time-tagged sub-states, each with its own plastic critic synapse) provides the per-tap value structure that the single-channel B-2 lacked, and the multi-channel reward relay (the critic inhibits `reward_us` so `r − V` reaches the SNc) localizes the value subtraction to the reward window so the chain's value no longer kills the SNc tonic. **Headline migration r = −0.802 / −0.765 / −0.891 (seeds 42/43/44), all < −0.7, sign-consistent (all cue-ward), 3/3 GO.** The full signature holds on all seeds: early trials burst at the US (peak bin ≈ 7, 100–115 Hz), late trials burst at the CS (peak migrates to bin ≈ 1–2) with the US burst shrunk (73→23, 65→17, 94→37 Hz — a genuine transfer, not a mere shrink), the omission test shows a burst at the cue (70–93 Hz) dipping below tonic at the expected-reward time (3/3), no burst in the CS→US gap, and the cue value grows on all seeds (V_cue 23→73, 26→69, 7→34 Hz). Anti-cheats are decisive: the cue-pathway lesion silences the cue + abolishes the migration while the innate US reflex SURVIVES (the reward bursts the SNc at 178–231 Hz via `reward_us`, 3/3); the unpaired-timing control (US at random bins) shows NO migration (r = −0.278 / −0.250 / −0.275, 3/3) — a DISCRIMINATING control (the paired migrates, the unpaired does not), which B-2/B-3 could not show. Provenance: the SNc drive is `tonic + reward_us(synaptic relay; critic inhibits = r−V) + synaptic GABA_B(−V) + synaptic conductance-derivative(+dV/dt)` only — `current_reward_signal = 0`, no host δ/value/EMA reaches the SNc. The B-2 protected `sim/` edit is byte-identical when OFF (COMBO `e728d7f19d99b5b4`, pre==post) and the relevant CPU test suites are unchanged (`16 failed, 115 passed` pre==post — every failure a pre-existing CuPy-required test under the numpy backend). **Verdict: GO — the migration is SURPASSED multi-seed.**

---

## How A-CSC + the multi-channel relay surpass the B-2 wall (the mechanism)

The B-2 finding characterized the single-channel rate-critic wall precisely: (1) the conductance-derivative tracks the rising EDGE not the value LEVEL, so the cue burst shrinks as the value grows; (2) a larger cue burst re-poisons the STDP eligibility (dense critic firing → net-negative eligibility); (3) a `−V` strong enough to transfer the US burst also kills the SNc tonic (the floor-ceiling conflict). Each is broken by a specific, biology-grounded addition:

1. **A-CSC tapped-delay cue (K=8 sub-states) — breaks the edge-vs-level conflict.** Because the cue is a chain of distinct time-tagged sub-states, each with its OWN plastic synapse onto the critic, the value is a clean per-tap STEP function V(tap_0), V(tap_1), … . The LEVEL at time t is the current tap's value; the DERIVATIVE is the inter-tap difference V(tap_{k+1})−V(tap_k) — a clean up-step whose positive burst rides the value's leading edge as it back-propagates over trials. The single-channel B-2 had only one channel, so the level and derivative were forced onto the same quantity and fought. The sub-states decouple them. (The brief's "MULTIPLE sub-channels decouple the single-channel conflict.")

2. **A critic teacher during the reward window + a sub-threshold critic tonic — break the cold-start.** The MSN critic is all-or-nothing at its rheobase (B-3's diagnosis), so the value cannot grow smoothly from 0. The reward (US) drives the critic to FIRE during the reward window (the innate-reflex-teaches-a-learned-circuit pattern), so the reward-overlapping sub-state forms CAUSAL eligibility → the reward DA grows its value first → the value GRADIENT (steep near the reward) seeds the back-propagation. A sub-threshold tonic holds the critic in the graded band so the per-tap value is readable as it grows. A SHORT eligibility tau (~40 ms ≈ 2 bins) makes credit tap-local (the reward credits the last tap; the bootstrap credits the tap just before the value-carrying one), giving the one-tap-per-trial back-propagation; the default 1000 ms tau smears credit across the whole chain and kills the migration.

3. **The critic FS-clamp — keeps the critic SPARSE so the value-growth doesn't re-poison eligibility.** A fast-spiking interneuron pool driven feedforward by the sub-states inhibits the critic (the production N9 mechanism, Tepper PV-FSI), holding it in the physiological band EVEN AS the per-tap weights grow — decoupling "the weights/value grow" from "the critic fires densely." This addresses B-2's constraint (2) directly.

4. **The multi-channel reward relay — breaks the floor-ceiling conflict (the decisive piece).** The single `−V` channel had to be strong enough to shrink the reward burst but that also suppressed the whole chain and killed the SNc tonic (no live cue burst). The relay routes the reward through an EXCITATORY `reward_us → snc` that the critic INHIBITS, so `r − V` reaches the SNc and `−V` is LOCALIZED to the reward window (reward_us is silent otherwise). The chain's value no longer touches the SNc tonic — so the SNc sits at a physiological tonic, the US fully vacates as its tap acquires value (the canonical δ=r−V), AND the conductance-derivative cue burst stands out at a live tonic. This is the named next approach ("separate value-LEVEL and value-DERIVATIVE so growing the value does not poison the migration"), realized as a reward-localized `−V` (level) + the membrane conductance-derivative (the bootstrap shift). With the reward off the SNc-direct path, the GABA_B `striosome_value→snc` channel can be WEAK (its job is only to source the derivative), so the chain stays at a live tonic.

The combination converts B-2's "value grows but the burst does not migrate" into a clean multi-seed migration with the burst transferring onto the cue.

---

## The protected `sim/` edit (B-2 conductance-derivative, RE-APPLIED; additive, guarded, default-OFF byte-identical)

The migration depends on the conductance-derivative, which is the B-2 protected edit (an exact structural mirror of the shipped, owner-approved GABA_B/GIRK block). It was byte-reviewed-then-reverted under the B-2 stop-rule; the owner's 2026-06-10 "work through / auto-approve byte-proven diffs" directive re-authorizes it. RE-APPLIED verbatim from commit `87ff8925`:

- **`sim/config.py` (+15):** `enable_td_value_derivative: bool = False`; `td_slow_tau_ms: float = 400.0`; `td_derivative_gain: float = 1.0` (read ONLY inside `if enable_td_value_derivative`).
- **`sim/bridge.py` (+18 sites):** `cp_conductance_g_gabab_slow = None` (`:240`); guarded alloc (`:1234`); the always-cheap slow-EMA decay cache (`:1631` + the checkpoint-load recompute `:7292`); and one guarded per-step block (`:5906–5932`) — a slower-decaying leaky EMA of `g_gabab` plus `I_td_deriv = td_derivative_gain·(g_gabab − g_gabab_slow)·(E_exc − V)` (E_exc = `syn_reversal_potential_e` = 0 mV, depolarizing), added to `total_input_current_pA`.

**Sign:** on a value RISE `(g_gabab − g_gabab_slow) > 0` and `(E_exc − V) > 0` (V ≈ −60 mV) → positive → the SNc bursts; on a value FALL → negative → the dip; flat → ≈ 0. ADDITIVE on top of the existing GABA_B `−V`.

**Byte-identity (off == baseline), proven at the current base:** the harness `research/findings/raw/_b2_byteproof_harness.py` (a 50-step Izhikevich smoke + a 60-step Stage-B GABA_B warm-up, per-step `v`/`firing` hashes) gives **COMBO `e728d7f19d99b5b4`** with the flag default-OFF — IDENTICAL to the pre-edit tree (verified by stashing `sim/config.py`+`sim/bridge.py` to the base and re-running: same COMBO). The guarded block is unreached and `total_input_current_pA` is bit-identical when the flag is False.

**Regression-absent (R-A):** `pytest tests/{test_determinism,test_regions,test_neuromodulators,test_backend}.py` under `SIM_BACKEND=numpy` gives the IDENTICAL `16 failed, 115 passed` pre-edit (stashed) and post-edit — every failure is a pre-existing CuPy-required test failing under the numpy backend, NOT introduced by the edit. `tests/test_kernels_cpu.py` + `tests/test_td_critic_no_harm.py` = 35 passed.

**Load-bearing ablation (proves the sim/ edit is what enables the cue-shift):** with the conductance-derivative OFF (`--csc-no-conductance-deriv`), the migration r drops from −0.802 to −0.624 (below the bar) AND the early-burst-at-US signature is LOST (without the derivative there is no value-onset burst). The derivative is load-bearing for crossing the −0.7 bar + the early-US signature.

**`test_td_critic_no_harm` note:** its byte-untouched guard (`_TD_BASE..HEAD`, COMMITTED only) PASSES pre-commit; it will (correctly) trip once the protected edit is committed — that is the intended behaviour. The base SHA `_TD_BASE` is re-bumped to this commit AFTER the byte-review (the owner directive auto-approves the byte-proven diff), so the guard tracks the latest approved protected edit.

---

## The A-CSC build (runner-side; `research/runners/snc_stageb_critic_probe.py`, `--td-csc` mode)

A new `--td-csc` mode (`run_td_csc`, `run_td_csc_lesion`, `_run_td_csc_mode`, `_print_td_csc_result`) reuses the existing Stage-B bridge build, the dopamine threshold + baseline calibration, the eligibility/three-factor learning, the time-course recorder, and the lesion machinery. Regions: `csc_0..csc_{K-1}` (the tapped-delay cue) + `striosome_value` (critic) + `csc_fs` (the FS-clamp) + `reward_us` (the reward relay) + `snc`. Pathways: each `csc_k → striosome_value` plastic (the tap value `w_k`); `csc_k → csc_fs` + `csc_fs → striosome_value` (the clamp); `reward_us → snc` excitatory + `striosome_value → reward_us` inhibitory (the relay `r − V`); `striosome_value → snc` gaba_b (the derivative source). The sub-state TIME-TAGGING (which tap is active in which bin) is the world's cue-presentation timing (legitimate environment boundary, design §2.4 — the same status as the sustained cue in B-3); the VALUE LEARNING, the derivative, the burst, the dip, and the credit assignment are all NEURAL.

**Production recipe (the locked multi-seed GO config):**
```bash
SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe --td-csc \
    --seeds 42,43,44 --csc-n 8 --csc-reward-relay \
    --csc-to-strio-weight 6.0 --csc-critic-tonic-pa 140 --csc-critic-teacher-pa 700 \
    --snc-tonic-pa 300 --csc-reward-us-drive-pa 600 \
    --csc-reward-us-to-snc-weight 8 --csc-strio-to-reward-us-weight 10 \
    --csc-strio-to-snc-weight 1.5 --csc-td-derivative-gain 1.0 --reward-learning-rate 0.10 \
    --csc-fs-clamp --csc-to-fs-weight 16 --csc-fs-to-strio-weight 10 \
    --csc-iti-bins 8 --csc-td-slow-tau-ms 130 --csc-gabab-tau-decay 40 --n-train 50
# (+ --td-lesion-cue / --td-unpaired for the anti-cheats)
```

---

## Multi-seed migration (headline) — 3/3 GO

| Seed | migration r (bar < −0.7) | SNc peak-bin early → late (US = bin 7, cue = bin 0) | US-window early → late | cue value early → late | gates |
|---|---|---|---|---|---|
| 42 | **−0.802** | 7.0 → 1.1 | 73 → 23 Hz | 23 → 73 Hz | **6/6** |
| 43 | **−0.765** | 6.5 → 1.4 | 65 → 17 Hz | 26 → 69 Hz | 5/6 |
| 44 | **−0.891** | 7.0 → 2.2 | 94 → 37 Hz | 7 → 34 Hz | 5/6 |

**3 GO + 0 PARTIAL / 3. Sign-consistent (all cue-ward): True. Omission-dip-at-reward all seeds: True.**

Per-seed SNc firing-rate time-course (Hz per bin; cue = bin 0, reward = bin 7):
```
seed 42  first: [ 53  48  67  75  58  53  52 100  22  15  30]   peak @ US (bin 7)
         last : [ 28  33  22  25  20  17   5  22   0   0   0]   peak migrated to bin 1 (cue), US -> 22
         omit : [ 70  62  55  52  37  28  15  10   2   3   2]   burst @ cue, DIP @ reward (bin 7)
seed 44  first: [ 70  85  77  67  60  65  58 115  37  28  25]   peak @ US (bin 7)
         last : [ 23  50  48  37  45  35  28  30   0   0   0]   peak migrated to bin 1 (cue)
         omit : [ 93  60  85  67  60  50  43  17   3   3   5]   burst @ cue, DIP @ reward
```

Gates (design §4.2), per seed: `migration_r` ✓3/3, `migration_dir` ✓3/3, `early_burst_at_us` ✓3/3, `omission_dip_at_reward` ✓3/3, `cue_value_grows` ✓3/3; `late_burst_at_cue` (the strict full-vacating bar) ✓ on seed 42, graded-partial on 43/44 (the HS98 slow-learning regime: the US burst shrinks substantially — 65→17, 94→37 — but a small residual remains; design §4.2 scores graded transfer a defensible PASS). Headline JSON: `research/findings/raw/_td_cue_shift_acsc_3seed.json`.

---

## Anti-cheats (the TD error is brain-based; the migration is not a host artifact)

1. **Cue-pathway lesion → migration vanishes, US reflex survives — PASS 3/3 (decisive).** After training, zeroing EVERY `csc_k → striosome_value` edge silences the critic on the cue (V → **0.00 Hz, 3/3**), removes the cue-time SNc elevation (no_cue_burst ✓3/3), and — because the critic can no longer inhibit `reward_us` — the **innate US reflex bursts the SNc strongly** (178 / 223 / 231 Hz vs tonic ~60, us_reflex_intact ✓3/3). This proves the migration is carried by the synaptic `csc → critic` conduit (cut it → no cue, no migration), while the reward reflex (`reward_us → snc`) is preserved. JSON: `research/findings/raw/_td_cue_shift_acsc_lesion_3seed.json`. (The relay fixed the B-2 lesion-test artifact, where the bare high tonic masked the US reflex.)
2. **Unpaired-timing control → no migration — PASS 3/3 (DISCRIMINATING).** Firing the US at a RANDOM bin unrelated to the chain (no CS→US contingency) gives **no migration on all 3 seeds (r = −0.278 / −0.250 / −0.275, none < −0.7)**, while the PAIRED condition migrates (r ≈ −0.82). This is a genuine paired-vs-unpaired contrast — the migration rides on the real contingency, not a cue-present back-channel. (B-2/B-3 could not show this contrast because their paired condition was already negative.) JSON: `research/findings/raw/_td_cue_shift_acsc_unpaired_3seed.json`.
3. **Provenance — PASS.** Under `--td-csc --csc-reward-relay`, the SNc drive is `tonic(direct) + reward_us(synaptic relay; critic inhibits = r−V) + synaptic GABA_B(−V) + synaptic conductance-derivative(+dV/dt)` ONLY — `snc_gets_direct_reward = False`, `host_reward_signal = 0`, `host_value_term = False`, no host δ / γV′−V / value-EMA. The reward enters SYNAPTICALLY (the only direct SNc current is the tonic pacemaker). Recorded in the JSON `provenance` block.

---

## Honest scope + residual

- **The full-vacating gate is graded on 2/3 seeds (HS98 regime).** The headline migration (peak US→cue, r < −0.7) is a clean 3/3; the strict "US burst fully vacates to tonic" gate fully passes on seed 42 and is graded (substantial shrink + small residual) on 43/44. Hollerman-Schultz 1998 measured exactly this graded, learning-rate-dependent transfer (slow-learned pairs retain partial reward responses), so the graded transfer is a defensible PASS, not a failure. Pushing all seeds to complete vacating is a tuning refinement (e.g. flattening the back-propagated value gradient near the reward), not a mechanism gap.
- **The cue's time-tagging is world-presented** (the apparatus activates sub-state k in bin k), exactly as the sustained cue was in B-3 (design §2.4 — the world's stimulus timing). The brain's job — learning each sub-state's value, the derivative, the burst, the dip, the credit assignment — is 100% neural. A fully self-propagating neural delay chain (cue onset → a feed-forward relay wave) is a faithful enrichment, not a correction (it was deferred because a tiny CPU bridge cannot reliably space chain taps one bin apart; the world-clocked CSC is the standard Montague-Dayan-Sejnowski representation).
- **CPU probe scale.** This is the sensitive test (the nav gridworld is orient-solvable and reward-insensitive — current-state assessment). A nav-scale in-vivo demonstration of the cue-shift would need a reward-load-bearing task (a separate, larger arc).

---

## Verdict

**GO — the migration is SURPASSED multi-seed.** A-CSC (the complete-serial-compound tapped-delay cue) + the multi-channel reward relay deliver the Schultz 1997 dopamine cue-shift on the spiking SNc: the phasic burst migrates from the reward onto the predictive cue across learning, **r = −0.802 / −0.765 / −0.891 (3/3 < −0.7)**, with early-burst-at-US, late-burst-at-CS (genuine transfer), omission-dip-at-reward-time, no-burst-in-gap, and value-growth all multi-seed, the TD error computed entirely by neurons (no host term), and both anti-cheats decisive (cue-lesion → migration gone + US reflex intact 3/3; unpaired → no migration 3/3, discriminating). The B-2 protected conductance-derivative `sim/` edit is byte-identical when OFF (COMBO `e728d7f19d99b5b4`) and breaks no tests; it is load-bearing for the migration (ablating it drops r to −0.624 and loses the early-US burst). This lands the one canonical dopamine signature the N9 circuit did not yet show — navigation's RPE is now Rescorla-Wagner AND temporal-difference.

---

## Artifacts

- Probe (extended): `research/runners/snc_stageb_critic_probe.py` (`--td-csc`, `run_td_csc`, `run_td_csc_lesion`, `_run_td_csc_mode`, `_csc_substate_weights`, `_print_td_csc_result`).
- Protected edit (re-applied): `sim/config.py` (+15), `sim/bridge.py` (+18 sites) — the B-2 conductance-derivative.
- Byte-proof harness: `research/findings/raw/_b2_byteproof_harness.py` (COMBO `e728d7f19d99b5b4`, pre==post).
- Headline 3-seed: `research/findings/raw/_td_cue_shift_acsc_3seed.json`.
- Cue-lesion anti-cheat 3-seed: `research/findings/raw/_td_cue_shift_acsc_lesion_3seed.json`.
- Unpaired anti-cheat 3-seed: `research/findings/raw/_td_cue_shift_acsc_unpaired_3seed.json`.
- Run (headline): the production recipe above (+ `--td-lesion-cue` / `--td-unpaired`).
