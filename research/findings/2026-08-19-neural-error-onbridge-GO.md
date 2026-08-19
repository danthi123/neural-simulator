---
type: finding
status: live
date: 2026-08-19
mechanism: urbanczik-senn-dendritic-prediction
lane: brain-based-purity burndown (board #69 — the "was I wrong?" teaching error, ON the live bridge)
verdict: onbridge-GO-6seed
artifacts:
  - research/findings/raw/_neural_error_onbridge.json
  - research/findings/raw/_neural_error_onbridge_ksweep.json
supersedes_status_of: research/findings/2026-06-17-onbridge-neural-error-realization-boundary.md
---

# The neural "was I wrong?" error drives the LIVE spiking read-out — the host subtraction is retired ON the bridge, 6-seed GO

**Board #69 (BRAIN-BASED-ONLY burndown, the #39 next rung).** The read-out that learns the brain's word choices is
trained ON the live spiking `SimulationBridge` by the bridge's OWN three-factor plasticity
`weight_update = lr * cp_per_synapse_reward_override[synapse] * cp_eligibility_trace[synapse]` = `lr * err_post *
pre` (the cerebellar climbing-fiber delta rule, 6-seed GO in `2026-06-17-onsubstrate-readout-rule-bridge-GO.md`).
But the per-output error written into that override channel was `err_j = target_j - est_j` — a HOST subtraction. Under
the brain-based-only standard the host formula is a documented SHORTCUT: the *brain* was not computing the error.

This finding delivers the read-out neuron's OWN somato-dendritic mismatch (Urbanczik & Senn, Neuron 2014) THROUGH
that same per-synapse channel, on the live bridge. The value written into `cp_per_synapse_reward_override` for output
`j` is the neuron's intrinsic `(soma_rate_j - phi(v_basal_j))` mismatch, computed by the SHIPPED
`sim.dendritic_plasticity.urbanczik_senn_update` (the neuron's biophysics), decoded by a fixed transfer slope. No
host `target - est` anywhere in the on-bridge learning loop.

**Verdict: GO (the runner's own verdict, preconditions guard: GO), 6 seeds (42,43,44,100,101,102).** The neural
on-bridge error drives the live plasticity as well as the host on-bridge error (NEURAL = 97% of HOST held-out
generalization), and every anti-cheat collapses. This is the on-bridge realization the 2026-06-17 boundary finding
banked as a *deferred confirmation* — now done.

**Runner:** `research/runners/_neural_error_onbridge_derisk.py` · **Raw:**
`research/findings/raw/_neural_error_onbridge.json` · CPU/numpy, **NO `sim/` edit**, additive only.

## What is distinct from the 2026-08-19 numpy de-risk and the 2026-06-17 boundary

- The numpy de-risk (`2026-08-19-neural-error-population-GO.md`, NEURAL=0.964) proved the U-S soma-vs-dendrite error
  drives a numpy `W_O +=` delta rule. This finding delivers the SAME error through the LIVE bridge's real
  synaptic-plasticity channel — the error is computed from the bridge's OWN forward drive (`est = W @ rate`, read
  from `cp_connections.data`) and written into `cp_per_synapse_reward_override`, so the production learning loop USES
  it. The U-S error math (gain/beta/spike-window/slope-decode) is byte-for-byte the de-risk's; only WHERE the error
  goes changed (live bridge, not numpy).
- The 2026-06-17 on-bridge attempt (`2026-06-17-onbridge-neural-error-realization-boundary.md`) used a *separate*
  Rao-Ballard error population on a *second* bridge and did not converge at a tractable budget (banked as a deferred,
  non-load-bearing confirmation). This realization uses the read-out neuron's OWN two compartments (no separate error
  bridge) and converges — that deferred confirmation is now RESOLVED (see that doc's RESOLVED banner). The boundary
  was honest, not wrong: a different, stronger method (the read-out's own compartments) closed it.

## Method (reuse-by-import; NO sim/ edit)

- **Learning substrate:** the shipped bridge three-factor block. Stage-0 linchpin re-verified per seed:
  a single reward step with a known `cp_eligibility_trace` (= pre) and a known `cp_per_synapse_reward_override` (=
  per-output error) yields `ΔW = lr * outer(err, pre)` to float precision (max |ΔW − lr·outer| rel ~8e-8, all seeds).
- **Bridge:** `inp(2·D_h=128) → out(D_in=128)`, dense + plastic + reward-modulated; STDP/Hebbian/OU OFF so the only
  plasticity is the reward-modulated update; signed weights allowed (the clip does not rectify the signed decoder).
  Verbatim plumbing from the on-substrate read-out GO runner. `cfg.seed = seed` seeds the substrate (verified:
  identical `cp_neuron_firing_thresholds` hash at a repeated seed, differs across seeds — NOT `actual_seed_used`).
- **NEURAL error (per output j), from the bridge's OWN drive:** dendrite `v_basal_j = g·est_j` (est read from the
  bridge weights); soma `u_j = (1−beta)·est_j + beta·target_j` (finite teacher nudging, beta=0.5); spiking soma rate
  `s_j = <Poisson(sigma(g·u_j)·W)/W>_K` POOLED over K=16 error-neurons per output; the SHIPPED rule returns the
  mismatch `s_j − sigma(v_basal_j)`; decoded by the fixed transfer slope (beta·g/4) and written into
  `cp_per_synapse_reward_override`. No host `est − target`.
- **HOST error (reference / the current default):** the identical bridge trained by `err = target − est` written to
  the same channel — a like-for-like head-to-head (same bridge, budget=40 passes, lr=0.5, task, seeds; the ONLY
  difference is the error source).
- **Task:** the role-filler word/sequence-acquisition systematicity harness (R=4, F=16, 3 leakage-free splits);
  metric = held-out (never-trained role-filler combination) generalization; chance = 0.0625 (= 1/F, F=16).

## Result — 6 seeds, per-seed + pooled

<!--derived-->

All figures are rounded reads of the cited raw artifact `research/findings/raw/_neural_error_onbridge.json`
(pooled means, per-seed split-means, K-sweep, byte-identical panel).

| arm | held-out generalization (6-seed mean) | per-seed [42, 43, 44, 100, 101, 102] |
|---|---|---|
| HOST-onbridge (reference / current default) | 0.961 | [0.952, 1.000, 1.000, 1.000, 1.000, 0.811] |
| **NEURAL-onbridge (U-S soma-vs-dendrite, live)** | **0.929** | [0.905, 1.000, 1.000, 1.000, 1.000, 0.670] |
| LESION-nodend (silence dendritic self-prediction) | 0.032 | [0.000, 0.067, 0.000, 0.095, 0.000, 0.030] |
| LESION-noteach (silence somatic teaching) | 0.024 | [0.000, 0.000, 0.000, 0.143, 0.000, 0.000] |
| SCRAMBLE (mis-address error across outputs) | 0.042 | [0.000, 0.000, 0.000, 0.143, 0.111, 0.000] |

- **NEURAL == HOST within noise:** NEURAL-onbridge = 0.929 = **97% of HOST-onbridge**; NEURAL ≥ 0.85× HOST in **5/6**
  seeds. The read-out learns its word choices on the LIVE substrate as well from the neuron's own error as from the
  host subtraction. (numpy de-risk reference: 0.964; this on-bridge realization: 0.929.)
- **Anti-cheat #1 (on-bridge learning works with the neural error):** met — NEURAL 0.929, 97% of HOST, 5/6 parity.
- **Anti-cheat #2 (the neural error is load-bearing on-bridge) — three dissociations, each collapses learning:**
  1. **Silence the dendritic self-prediction** (pin `v_basal=0` so the dendrite stops predicting the soma):
     on-bridge learning collapses to 0.032 ≈ chance. A residual host `target−est` would be immune to this.
  2. **Silence the somatic teaching** (beta=0 → soma == dendrite → mismatch ≈ pure Poisson noise): 0.024 ≈ chance.
  3. **Scramble** (mis-address the neural error across outputs): 0.042 ≈ chance.
  All three floors sit far below 0.5× NEURAL (0.46), and the lesions are structural (re-applied every step — they
  persist at the moment of measurement, not a decaying weight). The runner's precondition guard confirms all three
  collapse, so the GO is earned (a control that did not collapse would downgrade to UNDEFINED).
- **Attribution** (`tools.lab.attributable_to`): dendritic self-prediction **0.966**, somatic teaching **0.974**,
  per-output addressing **0.954** — almost all of the on-bridge learning is owned by the neuron's own error
  machinery, not a residual host term.

## Byte-identical-when-OFF (anti-cheat #3)

- **No `sim/` edit** (git-verified, `no_sim_edit=True` in the artifact): the neural error is routed entirely
  runner-side into the already-present, default-None `cp_per_synapse_reward_override` array. The production bridge
  weight-update code is therefore **byte-identical to main by construction** — this is the load-bearing OFF
  guarantee. No feature flag was added because none is needed.
- **HOST/off path is deterministic + unperturbed:** training the HOST (off) path on two independent FRESH bridges at
  the same `cfg.seed` yields **bit-identical** learned-weight md5 (`byte_off_ok=True`, all 6 seeds) — asserted in the
  data (md5 compare), not inferred. The default on-bridge learning is what it was; the neural error is opt-in
  runner-side computation that does not touch it.

## Honest limits (the residual is real and stated)

<!--derived-->
(This section quotes rounded reads of the cited artifacts — the seed-42 K-sweep
`research/findings/raw/_neural_error_onbridge_ksweep.json` and the 6-seed per-seed block of
`research/findings/raw/_neural_error_onbridge.json`.)

- **The spiking-soma noise costs a population read.** At the LIVE on-bridge budget (40 passes, ~50× shorter than the
  numpy de-risk's 24000 steps), a SINGLE somatic read is noise-dominated: at seed 42, K=1 → NEURAL 0.133. The fix is
  the biological SNR lift the 2026-06-17 boundary finding named — **K error-neurons per output, pooled** (cortical
  words are redundantly coded): K=16 → 0.905, K=64 → 0.919 (plateau), against a K-independent HOST of 0.952. K=16 is
  the SNR knee, reported transparently, not tuned to win. This is faithfulness > speed: the operating point is a
  population of read-out error-neurons, not a cheaper host read.
- **Seed 102 is the honest miss.** On the hardest split (where even HOST drops to 0.811), NEURAL = 0.670
  (ratio 0.826, below the 0.85 bar). The GO rests on 5/6 parity, not on one lucky seed — the residual is exactly the
  finite-nudging + spike-count noise carried through the live plasticity, and it bites hardest where the task itself
  is hardest.
- **Scope (respected).** Closes the ERROR-SOURCE on the production path only. `target` remains a legitimate
  env/teacher scaffold (a somatic nudge, the innate-teacher-teaches-a-learned-circuit pattern). Does NOT touch the
  mouth/word-readout READ-REGIME (board #37) and does NOT claim deep/hidden-layer credit — the read-out is a single
  plastic layer. Tiny bridge (128→128), CPU/numpy, task-scale: the realization proves the mechanism composes on the
  live bridge, which is what the deferred confirmation asked.

## Verification (verify-go lenses run before landing)

- **Control-integrity / no host term:** the only `target − est` in the runner is gated by `mode=="host"`; the neural
  arm calls `us_neural_error` exclusively (shipped-rule mismatch). The lesion dissociation is the empirical proof —
  a residual host term would survive pinning the dendrite / zeroing the teacher; both collapse to chance.
- **Trace-the-array:** Stage-0 proves the override the runner writes IS the array the weight update reads
  (`ΔW = lr·override·elig`, rel ~8e-8). **Seeding:** `cfg.seed` controls the substrate (threshold-hash check).
  **Inert-error arm:** lesion-noteach (≈ lr-0 in expectation) sits at chance while the init is identical across all
  arms → the 0.929 is LEARNED, not a structural head-start. **Metric-can-detect:** scramble (matched magnitude,
  mis-addressed) collapses → not a mass artifact. **Instrument:** the headline is the runner's OWN GO verdict.

## Reproduce
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 SIM_BACKEND=numpy \
  python -u -m research.runners._neural_error_onbridge_derisk --seeds 42,43,44,100,101,102 --dh 64 --n-error-pop 16
```

## Biology
- Urbanczik & Senn, "Learning by the Dendritic Prediction of Somatic Spiking," Neuron 81:521-528, 2014
  (PubMed 24507189) — the local dendritic-voltage third-factor rule, shipped as `sim/dendritic_plasticity.py`.
  Binding: `research/biology/urbanczik-senn-dendritic-prediction.md`.
- On-bridge learning channel: the cerebellar climbing-fiber per-output third factor (Albus `Δw_i = −η·pf_i·cf_burst`)
  realized as `lr · cp_per_synapse_reward_override · cp_eligibility_trace`
  (`2026-06-17-onsubstrate-readout-rule-bridge-GO.md`).
- Mikulasch et al., Trends Neurosci 46:45-59, 2023 (PubMed 36577388) — prediction errors are computed locally in
  dendritic compartments, not in separate units.
