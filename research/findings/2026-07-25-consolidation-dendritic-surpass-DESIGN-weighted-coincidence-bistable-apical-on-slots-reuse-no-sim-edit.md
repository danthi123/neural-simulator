# Consolidation dendritic surpass — DESIGN (design-gate): make the `comp_attr` slots a GRADED one-of-N selective assembly via the on-bridge two-compartment bistable WEIGHTED-coincidence plateau (reuse, NO sim/ edit); the new ingredient r-iii lacked = the co-activation-potentiated DISTINCT-engram feedforward for the plateau to amplify (2026-07-25)

**Design-gate output** (read-only) for the systematically-confirmed selectivity boundary (`a8597ee2`: point-neuron NMDA+WTA
attractor saturates to a single dominant winner; SFA-eviction exhausted 9 configs; the named surpass is a dendritic
graded/one-of-N assembly).

## 1. The on-bridge dendritic substrate ALREADY EXISTS + is reusable with ZERO sim/ edit
All shipped, guarded, default-off, byte-identical-when-off, CI-pinned (`tests/test_dendritic_bistability.py`); the
r-iii/gap5 arcs ran on these on the REAL bridge (not a numpy scaffold):
- **NMDA-spike COINCIDENCE subunit** — `cfg.enable_coincidence_detection` + `coincidence_detector=True` on a
  `RegionPathway` (`sim/bridge.py:7080-7194`, `sim/regions.py:353`). **WEIGHTED** variant (`coincidence_weighted_drive`,
  `config.py:439`) grades the plateau by the *learned synaptic value* — reads the potentiated `ca1→slot` weights.
- **Two-compartment dAP** — `cfg.enable_two_compartment_dap` → a separate apical voltage `cp_v_apical`, attenuated
  apical→soma read (`apical_g_couple_to_soma` ≫ back-coupling), `apical_R` thin-high-R (`config.py:265,275`).
- **Intrinsic dendritic BISTABILITY** (the gap5 keystone) — `coincidence_plateau_self_regen>0` + `coincidence_plateau_v_hold`
  + KIR `apical_kir_g>0` → a v-gated self-sustaining plateau that LATCHES after the volley + a KIR down-state = a robust
  bistable band with a STABLE SILENT REST STATE a linear leak cannot give (`config.py:253,281`).
- **Graded plateau read-out** (`enable_graded_dendritic_plateau`, `config.py:461`) + **BTSP** plateau-gated one-shot
  credit (`enable_btsp`, `config.py:346`).
- The STANDALONE numpy classes (`sim/dendritic_neuron.DendriticLayer`, `dendritic_plasticity`, `dendritic_mlp`) are NOT
  on the bridge — a Stage-0 teaching scaffold only; the slots must ignite/hold as spiking bridge neurons (read by
  `slot_ignition` on `cp_firing_states`), so the on-bridge ops are the right tool.

## 2. Mechanism — why two compartments grade into one-of-N where point-neuron NMDA+WTA saturates
Point-neuron trilemma: a recurrent NMDA attractor strong enough to HOLD self-sustains, and under a shared WTA the first
slot to cross threshold suppresses the others and wins *regardless of which ca1 engram drove it* → single-dominant-winner
(SFA only rescales one global somatic race → sweep exhausted). **Two compartments DECOUPLE completion (a one-shot
coincidence trigger on the apical) from sustaining (intrinsic per-cell bistability)** → the recurrent need not be
supercritical to hold → the WTA need not win a global race → ignition becomes **per-cell + input-specific**: slot-i's
apicals latch only if fact-i's distinct engram, through the potentiated `ca1→slot_i` synapses in WEIGHTED coincident
drive, coincides above `k_thresh`; slot-j (fact-i's engram only via weak un-potentiated `ca1→slot_j`) stays in the silent
down-state. N slots hold their own latch simultaneously → one-of-N SELECTIVE, not collapse. Biology: Major-Larkum-Schiller
NMDA plateaus, Larkum apical-basal BAC coincidence, the silent down-state a point soma cannot provide.

## 3. The r-iii NEGATIVE, designed-against
Plateau-alone did NOT fix CA3 completion (`2026-07-08-riii-onsubstrate-coincidence-wired-but-blocked-by-missing-attractor`):
"blocked one level deeper" — held-out member's weighted drive ≈ a random non-member's (`c_drive` 2.99≈2.98) because NO
specific attractor structure existed to amplify. The plateau is NECESSARY-NOT-SUFFICIENT; r-iii's surpass only worked on a
HAND-INSTALLED attractor (`2026-07-08-riii-...-SURPASS-6seed`, 0.571 vs LINEAR 0.007). **Consolidation has the ingredient
r-iii lacked:** (1) distinct ca1 engrams (Jaccard 0.00-0.11, measured), (2) co-activation replay that writes `ca1→slot_i`
SELECTIVELY off zero (confirmed directional). So the specific structure is in the FEEDFORWARD `ca1→slot` weights read by
the WEIGHTED subunit — not ambiguous recurrents. And the slots are DISJOINT regions (no shared cells / between-slot
recurrents; only the shared FS pool), so gap5's shared-cell coupling that defeated co-storage is structurally absent.

## 4. Recommendation + GO-gate + exact harness changes (NO sim/ edit)
**Option 1 (cheapest, pure config reuse):** route the `ca1→comp_attr_s` pathways through the WEIGHTED coincidence subunit +
two-compartment bistable apical; keep the slots + co-activation replay unchanged. Operating point (gap5 GO_CFG + r-iii):
`self_regen 0.15, apical_kir_g 3.0, apical_g_couple 1.0, apical_g_couple_to_soma 5.0, apical_R≈50`; calibrate
`coincidence_k_threshold` to the PER-STEP weighted `ca1→slot` drive (held vs non, set k between — the r-iii Rung-0 probe).
Escalate: Option 2 = also move the slot self-loop HOLD from point `nmda_slow` to the bistable plateau; Option 3 = write the
`ca1→slot` selectivity ONE-SHOT via `enable_btsp` if STDP separation is too weak.

**Exact changes (research/-file only):** (a) `nmda_compositional_consolidation.py build_substrate` — add default-off
`args.comp_dendritic` that sets `coincidence_detector=True` on the `ca1→comp_attr_s` pathways (`:241-244`) + the cfg flags
(`enable_coincidence_detection`, `coincidence_weighted_drive`, `enable_two_compartment_dap`, self_regen, kir_g, couple,
apical_R, k_threshold), byte-identical when unset — same discipline as the runner-side SFA injection. (b)
`_consol_coactivation_derisk.py` — replace the SFA sweep with a k_thresh calibration probe (per-step `c_drive` held vs non)
+ the dendritic run + anti-cheat arms + 6-seed; add a `cp_v_apical`/coincidence-conductance reset in `slot_ignition`'s
quiet steps (gap5 silence-reset lesson).

**GO-gate (per seed 42/43/44/100/101/102):** one-of-N SELECTIVE ≥⌈N/2⌉ (vs the confirmed 1/3 point-neuron floor) ≥5/6
seeds (mechanism 6/6, magnitude may be 5/6 — the gap5 profile); HOLD after drive-off (bistable-on ≥50% ≥300ms, off <10%);
survives hippo-lesion-after; the 7 anti-cheats — no-region · **LINEAR (coincidence OFF, SAME potentiated wires → ≤chance =
plateau is load-bearing)** · no-replay · no-co-activation · **apical-lesion collapses selectivity** · permuted-tag · control-
outperforms; plus silent-rest 0.000 + the `c_drive[slot_i|fact_i] ≫ c_drive[slot_j|fact_i]` separation (r-iii diagnostic:
direct proof the plateau has specific structure to amplify). Dev on 42/43/44, freeze k from seed-42, blind-confirm
100/101/102. Cached Phase-1 substrate for the functional recall leg.

**Risk register:** magnitude cap (dendritic completion seed-variable ~0.18-0.33; gate on mechanism); k_thresh is a narrow
PER-STEP window (calibrate per-step not max, check k±1); if `c_drive` shows NO held-vs-non separation → the r-iii
"no-specific-attractor" failure → escalate to Option 3 (BTSP) BEFORE declaring negative; soften `comp_wta_weight` if a
residual global race remains (the per-cell bistable down-state does the separation now, not the WTA).

## Provenance
Design-gate (read-only) 2026-07-25. Findings cited inline (r-iii onsubstrate ×3, gap5 CA3-completion, dendrite-derisk-A,
P0.3, the consolidation co-activation finding, D2 scoping). Builds on `a8597ee2`. Reuse-by-config; NO sim/ edit.
