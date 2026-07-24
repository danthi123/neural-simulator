# gap#5 de-risk #1 NEGATIVE (banked) → DIAGNOSIS: "STDP writes nothing" was a BDSP `[-5,5]` weight-clamp bug, NOT the STDP rule; the Ecker symmetric band FORMS once fixed (2026-07-24)

**Follows** the research gate `2026-07-24-gap5-replay-sequence-encoding-shuffle-bar-research-gate.md` (commit 5cf4a205),
whose de-risk **#1** hypothesis was: *"remove the hard `_silence_soma_apical` theta reset (chain_overlap) so the previous
assembly decays THROUGH the next assembly's spike → STDP's fast decaying kernel writes an adjacent-dominant
distance-decaying forward BAND (the Ecker-2022 near-diagonal band), instead of BTSP's flat fan-out or STDP-with-hard-reset's
nothing."* **De-risk #1 is FALSIFIED — banked here — but the diagnosis found the real bug and the band now forms.**

## 1. De-risk #1 result (banked NEGATIVE)

Encode-only CPU run, `scratchpad/swr_band_encode.py` → `research/findings/raw/gap5_r4/swr_band_encode.json`
(n_ca3=1000, n_mem=8, chain_fwd=30). The GO pre-check (adj_fwd > skip_fwd > skip2, monotone_decay, adj_dominance > 0.6,
reverse ≈ baseline) **did NOT pass** for any STDP config:

| config | within | fwd_d1 | fwd_d2 | fwd_d5+ | rev_d1 | adj_dom | monotone_decay |
|---|---|---|---|---|---|---|---|
| STDP + overlap + overlapping-fields | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 1.0 | **FALSE** |
| STDP + overlap + disjoint | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 1.0 | **FALSE** |
| STDP + HARD-reset (control) | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 1.0 | **FALSE** |
| BTSP + HARD-reset (control) | 14.1 | 13.99 | 16.86 | **34.46** | 5.0 | 0.83 | **FALSE (fan-OUT)** |

- STDP (with or without overlap): every distance weight = a uniform **5.0** — no band.
- BTSP: a **FAN-OUT** — fwd_d1=13.99 **rising** to fwd_d5+=34.46 (skip links STRONGER than adjacent). The opposite of a
  decaying band. So removing the hard reset did NOT let STDP write the band. Hypothesis #1 falsified.

## 2. DIAGNOSIS — the collapse to 5.0 is a BDSP weight-clamp bug, not STDP's LTD

The prior read said *"STDP wrote NOTHING (flat at the init value 5.0)."* **Both halves are wrong, and finding the real
cause unblocks the arc.** Checkpoint isolation via the REAL `_prepare_sequence` at INIT / POST-WITHIN (chain_fwd=0) /
POST-CHAIN (`scratchpad/swr_stdp_diag_fast.py` → `raw/gap5_r4/swr_stdp_diag_fast.json`, n_ca3=300, seed 42):

| checkpoint | within | base | fwd_d1 | fwd_d2 | rev_d1 | bet_max |
|---|---|---|---|---|---|---|
| **INIT** (no encode) | 0.50 | 0.52 | 0.51 | 0.56 | 0.50 | 0.73 |
| **STDP post-WITHIN** (chain_fwd=0) | **208.4** | 12.8 | **192.6** | 25.4 | **157.6** | 300 |
| **STDP post-CHAIN** (chain_fwd=15) | **5.0** | **5.0** | **5.0** | 5.0 | **5.0** | **5.0** |
| **BTSP post-CHAIN** (chain_fwd=15) | 28.4 | 5.0 | 30.4 | **71.7** | 5.0 | 79.5 |

- **5.0 is NOT the init value** (init ca3→ca3 = 0.5). The within (BTSP) phase alone builds a strong, adjacent-dominant,
  ~symmetric store (within 208, fwd_d1 192 ≫ fwd_d2 25, rev_d1 157) — this is *already* a decaying band.
- **The chain phase then COLLAPSES the entire store to a uniform 5.0** (within 208→5, fwd_d1 192→5, bet_max 300→5).

**Root cause (definitively isolated + verified — read the substrate, don't theorize):** a **`chain_rule="none"` control
(ALL plasticity disabled) collapses the store to 5.0 in ONE step at the within→chain boundary** — no NaN/inf
(`scratchpad/swr_collapse_trace2.py`). So the collapse is **drive-intrinsic, not the plasticity rule.** Tracing it to the
exact op:

1. **BDSP clamps every active synapse to `[bdsp_w_min=-5, bdsp_w_max=5]`.** `enable_bdsp=True` for the whole encode
   (`_prepare_sequence:192`, `bdsp_learning_rate=0.0`, for the within-phase bistable apical plateau). `fused_bdsp_update`
   (`sim/kernels.py`) returns `cp.clip(w_new, bdsp_w_min, bdsp_w_max)` **even at lr=0** (`w_new = w + 0`). During the
   WITHIN phase the bistable plateau keeps cells self-predicting (`P≈Pbar → dev≈0 → the P0 moat → BDSP inactive`), so BTSP
   builds the store to 300. In the **TRANSIENT-plateau chain** (`self_regen=0`) the sequential drive breaks
   self-prediction → ca3→ca3 enters BDSP's `active_bd` set → **hard-clamped to ≤5.0 every step.** BTSP's strong
   plateau-gated potentiation OUTRUNS the per-step clamp (→ the 14–34 fan-out); the weaker STDP/hebb_sym rules cannot, so
   they are pinned at the 5.0 BDSP ceiling. **This — not STDP's LTD — is why "STDP wrote nothing."**
2. **A second latent clamp:** `enable_stdp` **defaults True** (`config.py:577`). With `stdp_w_max = max(10, 2.5·ca3w) = 10`
   (`_build`), the asymmetric STDP clips ca3→ca3 to ≤10 wherever it is active — the source of a separate uniform-~10
   collapse seen in the hebb_sym port before it explicitly disabled STDP.

So the arc spent effort tuning the STDP kernel/schedule (asymmetric a±, the hard-reset, the overlap) against a store that
two **weight-clamp bugs** were flattening regardless of the rule.

## 3. The RIGHT band-forming rule (Ecker 2022 / Mishra 2016 — read in depth, PMC8865846)

Ecker et al. 2022 (eLife 11:e71850) build SWR + replay on **AdExpIF POINT neurons** (our substrate class). The band comes
from the **temporally-SYMMETRIC, BROAD** rule — the only STDP measured in CA3 (Mishra et al. 2016):
`Δw = A·exp(−|Δt|/τ)`, **`τ = 62.5 ms`, `A+ = A− = +80 pA` (BOTH orders POTENTIATE)**, `w_max = 20 nS`. Temporal
PROXIMITY (not order) sets the weight → the near-diagonal DECAYING band, **symmetric → bidirectional replay** (real CA3
replays the same store forward AND reverse — Kandel Fig 5-2). Their *asymmetric* comparison rule (`τ=20 ms, A+=+400,
A−=−400 pA`) gives forward-only chains — **this is the rule class our `fused_stdp_weight_update` implements.** Adaptation
is required for a *traveling* (vs stationary) bump; place fields Gaussian σ=10% of a 3 m track, explore 32.5 cm/s, ~43 laps.

## 4. The FIX + de-risk #3: the monotone-decaying band FORMS (encode-only pre-check GO on the core criterion)

Two runner-side fixes to `_prepare_sequence` (**NO `sim/` edit; default OFF = byte-identical**): (a) **widen the BDSP clip**
to `[−hebb_max, hebb_max]` during the chain for the new rules, (b) a new **`chain_rule="hebb_sym"`** realizing Ecker/Mishra's
symmetric rule via the EXISTING `hebbian_rate_window` (BCM windowed co-activity, decay set to Ecker's τ=62.5 ms), which also
disables the default-on STDP. Encode-only pre-check (n_ca3=400, n_mem=4, within_events=6, chain_fwd=30, disjoint, seed 42):

| rule | within | base | FWD d1 | d2 | d3 | REV d1 | d2 | monotone_fwd | above_base |
|---|---|---|---|---|---|---|---|---|---|
| none (within store preserved) | 235 | 95 | **196** | 175 | 34 | 201 | 122 | **True** | **True** |
| **hebb_sym (Ecker symmetric)** | 238 | 126 | **204** | 185 | 49 | 206 | 132 | **True** | **True** |
| stdp (BDSP-fixed) | 235 | 95 | 196 | 175 | 34 | 201 | 122 | **True** | **True** |
| btsp | 23 | 5 | 32 | 39 | **73** | 5 | 5 | **False (fan-OUT)** | — |

**⇒ A monotone-decaying, above-baseline, near-diagonal band FORMS** (`adj_fwd > skip1 > skip2`, `adj_dominance ≈ 0.98`)
once the clamp bugs are fixed, where BTSP gives the fan-out. **HONEST SCOPE / two caveats:**
- The band is **SYMMETRIC** (rev ≈ fwd), i.e. **Ecker's PRIMARY biological regime → bidirectional replay** (Kandel Fig 5-2).
  It satisfies the core "monotone-decaying near-diagonal band" but NOT the task's literal `reverse ≈ baseline` — which the
  research gate explicitly relaxes ("biology replays the same store forward AND reverse; require reverse≈chance only under a
  deliberately asymmetric encode"). The **forward-DOMINANT** variant (Ecker's asymmetric regime, `reverse≈baseline`) is a
  tuning follow-on that must overcome the symmetric within-store.
- The band's ORIGIN is the **within-phase** BTSP-eligibility-decay + bistable-plateau store, NOT the chain rule: `none`
  and `stdp` give ~the same band as `hebb_sym`, and the **hard-reset anti-cheat does NOT collapse it** (hebb_sym +
  hard-reset = BAND=True, identical to overlap) — so the research gate's "overlap-is-load-bearing" anti-cheat FAILS here
  because the overlap isn't what makes the band; the within-phase store is. The genuinely-chain-written Ecker band (the
  faithful port target) would require FREEZING the between-links during the within phase so the `hebb_sym` symmetric chain
  is the SOLE between-writer (the `freeze_between_refresh` machinery already exists; this is the immediate next tuning
  step). Baseline is also moderately high for the same reason. Sharpness is encode-parameter-dependent (sharp at
  n_mem≤4/within≤6; saturates at n_mem=6/within=10).

The **moving-bump replay GO** (GPU, SWR-envelope + adaptation + Bayesian population decode + the structure-shuffle/reverse
anti-cheats) is the next step, held until the GPU frees.

## Files
- Banked negative: `research/findings/raw/gap5_r4/swr_band_encode.json`
- Diagnosis: `raw/gap5_r4/swr_stdp_diag_fast.json` (checkpoints), `scratchpad/swr_stdp_diag_fast.py`,
  `scratchpad/swr_collapse_trace2.py` (the `none`-control drive-intrinsic proof)
- Fix + port: `research/runners/_gap5_sequence_replay_derisk.py` (`chain_rule="hebb_sym"` + BDSP-clip widen + STDP-disable,
  all default-OFF), `scratchpad/swr_ecker_band_port.py` → `raw/gap5_r4/swr_ecker_band_port.json`
- Root-cause code: `sim/kernels.py::fused_bdsp_update` (`cp.clip(w_new, bdsp_w_min=-5, bdsp_w_max=5)`),
  `sim/config.py:577` (`enable_stdp: bool = True`)
