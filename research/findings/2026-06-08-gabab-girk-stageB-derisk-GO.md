# GABA_B → GIRK protected `sim/` edit — SHIPPED + Stage-B de-risk GO (byte-identity independently confirmed)

**Date:** 2026-06-08
**Type:** Protected `sim/` edit (owner-approved, controller byte-reviewed + byte-identity-verified) + its cheap-first CPU de-risk.
**Predecessors:**
- Design (owner byte-reviewed + approved): `2026-06-08-gabab-girk-conductance-design.md`
- The wall this fixes: `2026-06-08-spiking-snc-stageB-critic-derisk.md` (value learns; GABA_A membrane subtraction failed the state-specific gap 0/3)
- B′ circuit attempts (also failed): `2026-06-08-spiking-snc-stageB-Bprime-value-subtraction-circuit-research.md`

---

## One-paragraph result

The spiking-SNc actor-critic's neural value critic *learns* a cue-gated state value robustly, but on the
GABA_A-only engine no **strong, sign-correct subtraction of that value at the dopamine cell** emerged — the
**state-specific SNc gap** gate failed 0/3 across three circuit variants, root-caused as a substrate limit
(the SNc lacks KCC2 → depolarized `E_GABA ≈ −55 mV` → direct GABA_A is weak/shunting). The biologically
accurate fix is the mechanism biology actually uses to subtract expected reward onto dopamine cells:
**GABA_B (metabotropic) → GIRK potassium channel, reversal `E_K ≈ −90 mV`** — a non-chloride, genuinely
hyperpolarizing, slow conductance (Eshel 2015; Cohen 2012; Tepper & Lee 2007; the baclofen-evoked DA IPSP
reverses at −90 mV). It is now implemented as a **second inhibitory conductance** in the engine (the NMDA
pattern inverted), default-OFF, byte-identical-when-unused, routed per-`RegionPathway` `receptor="gaba_b"`.
With it, the **state-specific gap gate now PASSES 3/3** (seeds 42/43/44), the value stays cue-gated and the
omission dip is retained, and a clean A/B (same circuit, `gaba_a` vs `gaba_b`) plus a conductance lesion
localize the win to the new conductance.

## The protected edit (additive, default-off, byte-identical-when-off)

Owner-approved **Option (a): per-`RegionPathway` `receptor` field**. Four files, +133/−6:

| File | Change | Off-path |
|---|---|---|
| `sim/kernels.py` (+11) | ADD `fused_gabab_decay_and_current` (single-exponential slow K⁺ kernel). `fused_conductance_decay_and_current` **byte-unchanged**. | new fn never called unless `enable_gabab` |
| `sim/config.py` (+8) | ADD 4 fields: `enable_gabab=False`, `gabab_reversal_potential=-90.0`, `gabab_tau_decay=150.0`, `gabab_propagation_strength=0.105` | read only inside `enable_gabab` guards |
| `sim/regions.py` (+14) | ADD `RegionPathway.receptor: str = "gaba_a"` + thread `"receptor"` into the `_build_pathway` wiring-plan dict | default `"gaba_a"` ⇒ no synapse tagged ⇒ identical routing |
| `sim/bridge.py` (+100/−6) | import (+1); 3 `None` array decls; guarded `enable_gabab` alloc of `cp_conductance_g_gabab` + per-neuron `E_gabab`; `_cached_decay_gabab` at both decay-cache sites; per-synapse `cp_gabab_synapse_mask` build in `inject_explicit_wiring`; the guarded per-step GABA_B current block (B4) after the NMDA block | `cp_conductance_g_gabab is None` ⇒ B4 skipped ⇒ `total_input_current_pA` bit-identical |

**The only edits to *existing* lines in `bridge.py`** are: the `keyed` zip going from 5- to 6-tuples (threading
`all_receptors`) and the three downstream destructures going `[... for _,_,p,_,_ in keyed]` →
`[... for _,_,p,_,_,_ in keyed]`. This is **provably inert when off**: the sort key is unchanged
(`lambda t: (t[0], t[1])` — the 6th element never enters the comparison), `sorted` is stable, and the existing
destructures extract the same positional values (the new element at index 5 is ignored). The Izhikevich/HH/AdEx
dynamics kernels and call sites are **untouched**.

## Byte-identity — INDEPENDENTLY CONFIRMED (controller trust-but-verify)

Harness `research/findings/raw/_gabab_byte_identity_check.py` (pins `cfg.seed`/`heterogeneity_seed`/`ou_seed`,
steps the de-risk topology 200×, SHA-256 of the full V + g_i + g_e trajectory; only passes `receptor=` when
`--on`, so it runs identically on the pre-edit baseline). Run under `SIM_BACKEND=numpy`:

| Tree | `enable_gabab` | `cp_conductance_g_gabab` | trajectory SHA-256 | final V̄ |
|---|---|---|---|---|
| Clean baseline (`git stash` the `sim/` edit) | OFF | None | `0a4f6ecf…b8b60027` | −52.92 mV |
| **Edited tree** | OFF | None | `0a4f6ecf…b8b60027` | −52.92 mV |
| Edited tree | ON | ALLOC | `4fe4eaa1…60ca4da3` | −61.16 mV |

**The OFF hashes are bit-for-bit identical** (`0a4f6ecf…` == `0a4f6ecf…`). Note `nnz=2315` with fixed
pathways means `any_fixed=True`, so the edited tree *does* build `keyed` as 6-tuples — yet the trajectory is
identical, empirically proving the zip 5→6 change is order- and value-preserving (not just by argument). With
the feature ON the trajectory differs and the SNc hyperpolarizes ~8 mV — the GIRK K⁺ current is functional.
The CPU test suites (`test_regions`/`test_neuromodulators`/`test_webapp_server`/`test_backend`/
`test_synapse_storage`/`test_core_sim_composition`) show the same pass/fail set before and after the edit
(the failures are pre-existing CuPy-on-numpy-array import issues unrelated to this change; the edit adds zero
new failures).

## De-risk result (`SIM_BACKEND=numpy`, seeds 42/43/44; `--gabab --n-train 18 --snc-reward-gain 120 --strio-to-snc-weight 10 --snc-tonic-pa 220`)

| Seed | predicted (CS+US) | unpredicted (US) | gap (unpred/pred) | V cue-gated | omission dip | PRIMARY GATE |
|---|---|---|---|---|---|---|
| 42 | 0.00 Hz | 110.83 Hz | ∞ | ✓ (60/52 ≫ 17/13) | ✓ | **PASS** |
| 43 | 4.17 Hz | 99.17 Hz | 23.8× | ✓ | ✓ | **PASS** |
| 44 | 11.67 Hz | 90.00 Hz | 7.7× | ✓ | ✓ | **PASS** |

**PRIMARY GATE (state-specific gap + V-learned + omission-dip): 3/3 PASS** — the precise delta over the
GABA_A direct projection (0/3) and the B′ circuit attempts. All-4-gates 2/3 (seed 44 misses only
`us_burst_shrank`, the honest-scope Rescorla-Wagner consequence, not a primary gate). Result JSON:
`research/findings/raw/_gabab_derisk_3seed.json`.

## Anti-cheat — both hold

1. **GABA_A-only A/B control (controller re-ran seed 42 first-hand).** *Same circuit*, no `--gabab`:
   predicted 70.00 Hz, unpredicted 64.17 Hz, gap 0.92, **state-specific FALSE → PRIMARY GATE FAIL**
   (reproduces the wall). With `--gabab`: predicted 0.00, unpredicted 110.83, **PASS**. The only difference is
   the receptor → the win is the new GABA_B/GIRK conductance, not anything else.
2. **Conductance lesion** (zero `cp_gabab_synapse_mask`, ~880–900 synapses/seed): the state-specific gap
   **VANISHES 3/3** (predicted ≈ or > unpredicted) — the SNc bursts to every reward once the GABA_B conduit
   is cut, proving the subtraction is carried by the conductance, not host arithmetic. (The probe holds
   `current_reward_signal = 0.0`; no host `V`/reward_ema reaches the SNc.)

## Honest scope + modeling choices

- **Calibration was required** (the design anticipated it): default `snc_reward_gain=400` saturates the SNc
  at its 500 Hz spike ceiling (no gap headroom); `rg≈120` + 18 training trials is the robust operating point.
  Over-training (≥30 trials) also saturates as V/weights hit the cap → a training-duration sweet spot.
- **Single-state conductance** (one decay τ=150 ms) is a phenomenological model of a metabotropic receptor;
  the cooperative Destexhe-Sejnowski G-protein kinetics + a finite dual-exponential rise are ranked future
  refinements (design §6), not needed for the value-subtraction.
- **GABA_A/GABA_B co-occurrence:** a `receptor="gaba_b"` synapse still also drives the GABA_A `g_i` (routing
  is presynaptic-trait-based and unchanged) — i.e. it delivers a weak GABA_A (E=−55) *and* a strong GABA_B
  (E=−90) current; the de-risk confirms GABA_B dominates. A pure-GABA_B route (skip the `g_i` increment) is a
  follow-on, not required.
- **R-W not TD:** the scheme is Rescorla-Wagner (CS-gated prediction → US-burst shrink + omission dip), not
  the full TD cue-shift burst-migration onto the CS (a deeper, orthogonal later increment).
- **Probe determinism fix (probe-only, not `sim/`):** the probe now pins `cfg.seed` so each `--seed` is
  reproducible across processes (previously it time-seeded → a per-process lottery). Required for a credible
  multi-seed verdict.

## Status + next

The GABA_B/GIRK conductance is the biologically accurate fix for the depolarized-SNc-GABA subtraction wall.
The protected edit is additive, default-off, **byte-identical-when-off (independently verified)**, and the
de-risk PASSES the gate three circuit variants failed. **Next:** present to the owner (the landed diff + this
verification), then — on the owner's go — the **nav 6-seed regression gate** (flagship A+E+G v2.5 +
`--spiking-snc --enable-neural-critic` with the critic→SNc routed `receptor="gaba_b"`; acceptance = summed
reward ≥ Stage A; an honest negative is still a valid deliverable, mapping a neural-critic limit). This is the
last subtraction piece of Stage B — the neural value critic now both *learns* V and *subtracts* it onto the
dopamine cell through the real GABA_B/GIRK biology.
