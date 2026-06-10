# N9 TD cue-shift — B-3 cheap-first de-risk: NEGATIVE (zero-edit value-derivative cannot grow the cue value → no burst migration)

**Date:** 2026-06-10
**Type:** Cheap-first CPU de-risk (runner-side only, ZERO protected `sim/` edit, `SIM_BACKEND=numpy`, no GPU). An honest NEGATIVE is a valid deliverable (design §6, project standard).
**Design:** `docs/plans/2026-06-10-N9-TD-cue-shift-design.md` — option **B-3** (§2.2: the zero-edit value-derivative), validation protocol §4, anti-cheats §5, three-outcome framing §6.
**Probe extended:** `research/runners/snc_stageb_critic_probe.py` — added a `--td` Pavlovian cue→reward mode (`run_td`), a cue-lesion anti-cheat (`run_td_lesion`), an unpaired-timing anti-cheat (`--td-unpaired`), and a SNc firing-rate time-course recorder (`_drive_timecourse`). Reuses the existing Stage-B bridge build, the dopamine-threshold calibration, the eligibility/three-factor learning, and the lesion machinery.

---

## Terms (defined once)

- **CS / US** — conditioned stimulus (the predictive cue) / unconditioned stimulus (the primary reward).
- **SNc** — substantia nigra pars compacta, the midbrain dopamine-cell pool; its windowed firing rate IS the dopamine signal here.
- **Critic** — the `striosome_value` medium-spiny-neuron pool that learns and represents the state value V.
- **Rescorla-Wagner (R-W)** — the one-step prediction error δ = r − V (no time axis; what the circuit computes today).
- **Temporal-difference (TD)** — δ = r + γ·V(next) − V(now); the bracketed term is the **value derivative / bootstrap** that R-W lacks and that produces the **cue-shift**: across learning the dopamine burst migrates from the reward onto the predictive cue (Schultz 1997 — the one canonical dopamine signature the circuit does not yet show).
- **Eligibility trace** — a per-synapse decaying memory of recent pre/post co-firing (the engine's `cp_eligibility_trace`), formed by STDP; the three-factor reward update is `Δw = learning_rate × dopamine_signal × eligibility`.
- **Time-of-peak** — the time-bin index of the maximum SNc firing rate within a trial window; migration = the peak moves earlier (toward the cue) across learning.

---

## One-paragraph result

**B-3 is NEGATIVE: the zero-edit value-derivative does not produce a dopamine-burst cue-shift on this circuit, multi-seed.** Across 60 cue→reward trials, the SNc burst time-of-peak stays pinned at the reward (US) onset on all three seeds — **migration r = +0.000 (42/43/44)**, the peak never moves toward the cue. The root cause is fully characterized and decisive: **the critic's cue value cannot grow across learning — it shrinks** (V on the cue 70→13, 61→17, 63→22 Hz on seeds 42/43/44; the `cue→striosome` weight 19→13 on every seed). Migration requires the cue value to *rise* from zero toward the reward magnitude; here it *falls*. Isolating the two plasticity terms shows why: at the dense critic-firing operating point the disinhibition route requires, **raw STDP is net-depressing** and the **eligibility trace is net-negative**, so the (positive) reward dopamine signal — applied to a negative eligibility — *amplifies* the depression rather than building the value. No operating point grows the value (swept cue weight 9–14, learning rate 0.08–0.8, dopamine sensitivity 8–40 — all shrink or stay flat). The two anti-cheats confirm the (small, level-driven) cue-time SNc activity is genuinely synaptic and not a host artifact: a cue-pathway lesion silences the cue response while the innate US reflex survives (2/3 clean, 1/3 a tight-threshold near-miss), and an unpaired-timing control shows no migration and no artifact (3/3). **Verdict: NEGATIVE — escalate to a protected conductance-derivative edit (B-1/B-2) or the tapped-delay-line CSC (A-CSC), both of which decouple the value-derivative delivery from the critic's STDP-unfriendly dense firing.**

---

## What was built (runner-side only; zero `sim/` edit)

`research/runners/snc_stageb_critic_probe.py` (`+555/−5`, additive — the 5 deletions are the bridge-builder signature, one `if`→`elif`, and the `_lesion_pathway` orientation block rewritten with a both-orientation fallback):

1. **`--td` Pavlovian protocol (`run_td`).** Each trial = **cue ON at t0, SUSTAINED across the CS→US interval** (the A-trace, design §2.1) → **reward (US) at t0+ISI** → an inter-trial gap, all stepped in one continuous window so the cue trace and the reward co-exist in the trajectory. Learning is ON (the critic learns V from the SNc-derived dopamine). Window layout (bins): `[ CS-only | ISI (CS continues; US fires in the first half) | post ]`.
2. **The B-3 value-derivative via DISINHIBITION (`td_disinhibit=True`, the one new wiring).** To deliver a value *derivative* with the sign the cue-shift needs (a value RISE must EXCITE the SNc = a burst at the cue), the critic is routed through one extra inhibitory stage so a value rise *releases* the SNc drive:
   `striosome_value (phasic V) —(inhib)→ disinhib —(inhib)→ snc_drive —(exc)→ snc`.
   This reuses the already-shipped B′-disinhibit excitatory relay (`snc_drive`) plus a tonically-paced GABAergic `disinhib` stage (the B′-SNr recipe). The reward (US) enters at the relay. ZERO `sim/` edit — only `BrainRegion`/`RegionPathway` data.
3. **SNc firing-rate time-course recorder (`_drive_timecourse`).** Records the per-bin SNc and critic rates across the whole window so the time-of-peak is measurable; supports within-window CS/US event scheduling (the world's trial clock — legitimate environment timing, design §2.4).
4. **Dopamine-signal centering (`_calibrate_da_baseline`).** A required TD-mode calibration: after the firing-rate threshold is set, the modulator *baseline* is set to the settled tonic dopamine concentration so `da_signal = da_conc − baseline` is centered at zero at tonic (a burst → +signal = potentiation, a dip → −signal = depression). Without it the dopamine concentration settles near zero at tonic, far below the fixed 0.5 baseline → a constant negative signal → pure depression. (This fix was necessary but, as the result shows, not sufficient.)
5. **Anti-cheats:** `run_td_lesion` (cue→striosome lesion) and `--td-unpaired` (decoupled CS/US timing), plus a provenance assertion (no host reward scalar / no host value / no direct SNc current).

---

## Protocol

`SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe --td --seeds 42,43,44 --n-train 60`
(+ `--td-lesion-cue` and `--td-unpaired` for the anti-cheats.)

- Operating point (calibrated for SNc tonic ≈ 10–14 Hz with both up-headroom (cue burst) and down-headroom (omission dip)): relay tonic 300 pA, disinhibitor tone 100 pA, disinhibitor→relay weight 8, cue→striosome weight 20, reward gain 400 pA, learning rate 0.08, dopamine sensitivity 8.
- 60 acquisition trials; window = 6 CS bins + 4 ISI bins + 4 post bins, 20 sub-steps/bin (1 ms each). The US/reward onset is bin 6.
- **Headline metric (design §4.2):** Pearson r between trial number and the SNc burst time-of-peak. Migration ⇒ peak moves earlier (toward the cue) ⇒ **r < −0.7**. Supporting gates: early burst at US, late burst at CS (transfer, not mere shrink), no burst in the CS→US gap, omission dip at the expected-reward time, value learned (rises).

---

## Multi-seed result (headline)

| Seed | migration r | peak-bin early→late (US onset = bin 6) | V(cue) early→late | cue weight early→late | verdict |
|---|---|---|---|---|---|
| 42 | **+0.000** | 6.00 → 6.00 | 70.8 → 13.3 Hz | 19.1 → 12.7 | NEGATIVE (PARTIAL on support 3/5) |
| 43 | **+0.000** | 6.00 → 6.00 | 61.3 → 16.7 Hz | 19.4 → 13.3 | NEGATIVE |
| 44 | **+0.000** | 6.00 → 6.00 | 63.5 → 22.2 Hz | 19.6 → 13.5 | NEGATIVE |

**MULTI-SEED: 0 GO + 1 PARTIAL / 3.** The SNc burst peak stays at the reward (US) onset on every seed — **no migration toward the cue** (r ≈ 0, not < −0.7). The early-burst-at-US gate passes on all seeds (the US reflex is intact); the late-burst-at-CS gate fails on all (no transfer). The cue value falls on every seed. Result JSON: `research/findings/raw/_td_cue_shift_3seed.json`.

### Per-seed supporting gates

| Gate | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| migration_r (r < −0.7) | ✗ | ✗ | ✗ |
| migration_dir (peak earlier) | ✗ | ✗ | ✗ |
| early burst @ US | ✓ | ✓ | ✓ |
| late burst @ CS (transferred) | ✗ | ✗ | ✗ |
| no burst in CS→US gap | ✓ | ✓ | ✗ |
| omission dip @ reward time | ✓ | ✗ | ✗ |
| value learned (rises) | ✗ | ✗ | ✗ |

The omission dip is present at seed 42 (a residual R-W signature: the SNc dips below tonic at the expected-reward time) but inconsistent across seeds — expected, since it rides on the cue value that is decaying.

---

## Root cause (fully characterized — the decisive finding)

**Migration requires the cue value to grow across learning; on this substrate it shrinks.** The chain was isolated by ablating the two plasticity terms at the working operating point (cue weight 14, seed 42, 20 trials):

| Configuration | cue→striosome weight (start → end) |
|---|---|
| Both plasticity terms ON (normal) | 14.12 → **12.38** (drops most) |
| Reward modulation OFF (raw STDP only) | 14.12 → **13.05** (raw STDP is net-depressing) |
| STDP OFF (reward modulation only) | 0.98 → 0.98 (no eligibility forms → no learning) |

- **Raw STDP is net-depressing** at the dense critic-firing rates the disinhibition route requires (the critic must fire densely to drive the disinhibition; dense bidirectional firing produces anti-causal-dominated spike pairs → net depression).
- **The eligibility trace is net-negative**, so the *positive* reward dopamine signal (verified +0.017…+0.023 above the centered baseline, 100 % of the trial) applied to a negative eligibility **amplifies the depression** — turning the reward signal *against* value-building. Stronger reward learning made the value shrink *faster*, not grow (learning rate 0.5 / sensitivity 30: weight 10.9 → 8.4 vs the milder 19 → 13).
- **No operating point grows the value.** A sweep of the initial cue weight (9, 10, 11, 12, 14) found only two regimes: too weak to fire (value ≈ 0, flat) or saturated (value high but shrinking). There is no gradual-growth regime — the MSN-typed critic is effectively all-or-nothing at its rheobase.

This is exactly the design's anticipated honest-NEGATIVE (§6.1: a small rate-coded critic's value estimate is noisy and the single-step derivative is fragile) made concrete: the **prerequisite for migration — the cue value rising — is structurally prevented** by the STDP-eligibility sign under the dense firing the zero-edit disinhibition route demands.

---

## Anti-cheats (the circuit is brain-based; the negative is not an artifact)

1. **Cue-pathway lesion (decisive) — PASS 2/3 clean, 1/3 tight near-miss.** After training, zero the `cue → striosome` edges (1412 synapses; the lesion helper was fixed to handle both CSR orientations). Result: the critic on the cue → **0.00 Hz on all 3 seeds** (cue silenced), the cue-time SNc response collapses to tonic, the dip vanishes, **and the US reflex stays intact** (SNc bursts to the US at 75–102 Hz vs tonic 7–12 Hz). Seeds 42/43 PASS cleanly; seed 44 is UNEXPECTED only on the `no_cue_burst` numerical bar (CS-rate 9.72 vs tonic 7.14, ratio 1.36 vs the 1.30 threshold — a tight-threshold near-miss with the cue genuinely silenced, not a real cue burst). This proves the cue-time SNc elevation is carried by the `cue→striosome→disinhibition` **synaptic conduit**, not host arithmetic. JSON: `research/findings/raw/_td_cue_shift_lesion_3seed.json`.
2. **Unpaired-timing control — PASS 3/3 (no artifact).** Decoupling the CS and US in time (US at random offsets unrelated to the cue) gives no migration (r = +0.070 / −0.044 / −0.010) and no consistent dip on all 3 seeds. This confirms there is no spurious cue-present back-channel. (Honest caveat: because the *paired* condition is already negative (r ≈ 0), this control is *consistent* rather than *discriminating* here — it cannot show a paired-vs-unpaired contrast when the paired result shows no transfer to begin with.) JSON: `research/findings/raw/_td_cue_shift_unpaired_3seed.json`.
3. **Provenance assertion — PASS.** Under `--td`, the SNc receives **no direct external current**; its drive is `tonic(relay) + reward_us(at the relay) + synaptic disinhibition` only — no host δ, no host γV′−V, no host value/EMA (`current_reward_signal = 0`, `reward_baseline = 0`, asserted in `run_td`). Recorded in the JSON `provenance` block.

---

## Honest three-outcome placement (design §6.2)

- (i) Clean migration — **not reached** (r ≈ 0, no transfer).
- (ii) Partial / graded migration — **not reached** (the peak does not move toward the cue at all; the support count of 3/5 at seed 42 is carried by the US-reflex + no-gap-burst + omission-dip gates, none of which is the migration itself).
- (iii) **No migration — this is the outcome.** The deliverable is the negative + the diagnosis: the zero-edit value-derivative cannot grow the cue value because the dense critic firing the disinhibition route requires makes the STDP eligibility net-negative, so the reward dopamine signal depresses rather than builds the value. This maps a precise substrate limit (per the project standard: "the spiking TD failing to show a clean cue-shift maps a substrate limit and IS a valid deliverable").

---

## Verdict + recommendation

**VERDICT: NEGATIVE.** The B-3 zero-edit value-derivative (phasic critic + disinhibition relay) does not produce a dopamine-burst cue-shift, multi-seed (r = +0.000 on 42/43/44). The root cause is decisive and characterized: the cue value cannot grow because the disinhibition route forces dense critic firing, which makes the STDP eligibility net-negative, so reward learning unlearns the value. R-W signatures (the omission dip) partially survive; the TD migration does not — consistent with the project's standing result that this circuit computes Rescorla-Wagner, not TD.

**RECOMMENDATION: escalate to a protected conductance-derivative edit (B-1/B-2) before, or alongside, the tapped-delay-line CSC (A-CSC).** The reason B-3 failed points directly at the fix: B-3 ties the value-derivative *delivery* to the critic's *firing density*, and that density is what poisons the STDP eligibility. The escalation options **decouple** the two:

- **B-1/B-2 (the recommended next protected edit, design §3).** Read the value derivative from the **conductance** (a difference of two slow GABA_B/GIRK channels at two latencies, B-1; or a single channel read as a leaky derivative, B-2) rather than from the critic's instantaneous firing. The critic can then be driven at a *sparse, STDP-friendly* rate (where the eligibility is causal/positive and the value can grow) while the derivative is still computed at the SNc membrane. This is one additive, default-OFF, byte-identical-when-off edit (an exact mirror of the already-shipped GABA_B/GIRK block). It also removes the dependence on the MSN rheobase that gave the all-or-nothing value here.
- **A-CSC (the fidelity escalation if B-1/B-2 under-migrates, design §2.1/§6.3).** A complete-serial-compound tapped-delay chain of cue sub-states, each with its own critic synapse, lets TD back-propagate value one tap per trial — the literal Montague-Dayan-Sejnowski mechanism, more robust than a single-step derivative. Zero protected edit but a larger build and slower learning; warranted only if the conductance-derivative still under-migrates.

Either escalation should **first re-establish that the cue value can GROW** (the prerequisite this de-risk isolated as the true blocker) at a sparse critic-firing operating point, *then* test migration — the value-growth check is the cheap gate that B-3 failed and that any TD mechanism must pass.

---

## Artifacts

- Probe (extended): `research/runners/snc_stageb_critic_probe.py` (`--td`, `run_td`, `run_td_lesion`, `_drive_timecourse`, `_calibrate_da_baseline`).
- Headline 3-seed: `research/findings/raw/_td_cue_shift_3seed.json`.
- Cue-lesion anti-cheat 3-seed: `research/findings/raw/_td_cue_shift_lesion_3seed.json`.
- Unpaired anti-cheat 3-seed: `research/findings/raw/_td_cue_shift_unpaired_3seed.json`.
- Run: `SIM_BACKEND=numpy python -m research.runners.snc_stageb_critic_probe --td --seeds 42,43,44 --n-train 60` (+ `--td-lesion-cue` / `--td-unpaired`).
