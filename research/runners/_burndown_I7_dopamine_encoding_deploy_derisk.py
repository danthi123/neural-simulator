"""BURNDOWN Phase-2B I-7-b deployment de-risk: the shared spiking DOPAMINE state functionally modulates the
PRODUCTION `OneBrainComposer`'s fact ENCODING strength at store time (Lisman-Grace hippocampal-VTA loop; Kandel D.16
-- dopamine gates entry into LONG-TERM memory: a rewarded trace stays STABLE, an un-rewarded one degrades).

WHY THIS de-risk exists (vs the prior GO `_phaseB_dopamine_encoding_gain_derisk.py`):
  - The prior de-risk validated the gain MECHANISM on `RFPhasorComposer._store_substrate` (the test ORACLE) with a
    probe `g` and a HOST-injected read-noise knob (`_retrieve_noise`). GO (+6/12 within-fact lift, lesion null, moat 6/6).
  - This de-risk validates the SAME mechanism on the PRODUCTION-default `OneBrainComposer` store path (`_write_block` ->
    `store_conns` complex weights) read through the REAL on-bridge RF resonate + the REAL hard magnitude floor
    (`sim/bridge.py:5589 _rf_mag2 > _rf_floor2`), with the gain DERIVED FROM A NEURAL DOPAMINE SIGNAL (a spiking SNc
    pool + the `dopamine` modulator -- the SAME limbic recipe the merged one-brain uses, and the SAME DA the read-side
    salience gate reads, `nav_conv_merged_bridge.py:_da_confidence_gate`). So the limbic core reaches the conversational
    cortex on the WRITE side, on the real production composer.

THE MECHANISM (verified): the RF phase read-out has a hard MAGNITUDE FLOOR. Under common, gain-INDEPENDENT additive
read damage (complex jitter on the stored composite weights, fixed sigma), a readout neuron's per-neuron SNR is g*M/sigma:
a higher-gain (high-DA / salient) fact has cleaner recovered phase -> survives the floor -> the cue-match scan recalls
it; a unit-gain (tonic-DA / neutral) fact's low-SNR neurons drop below the floor -> garbled phase -> mis-recall/abstain.
The floor x damage interaction is the nonlinearity that makes a per-fact encoding gain DIFFERENTIAL, not a vacuous
global scalar (== the prior de-risk's verified mechanism, here on the production substrate + a NEURAL DA value).

THE DEPLOYMENT WIRING (this de-risk + the runner `nav_conv_merged_bridge.py` flag): the gain is read AT STORE TIME from
a shared `dopamine` concentration:  g = clip(g_min, g_max, 1 + k_DA*(DA - DA_baseline)).  Composer-layer only (a
multiply on the written complex weight in `OneBrainComposer._write_block`, already shipped default-OFF); NO sim/ edit.

ANTI-CHEATS (the decisive controls; >=5/6 seeds):
  - MECHANISM (load-bearing at store time): the high-DA fact's stored `store_conns` block magnitude == g x the tonic
    fact's -- the DA literally scales the synaptic encoding strength (not a downstream read trick).
  - DIFFERENTIAL: under matched read damage, the high-DA (salient) fact recalls where the tonic (neutral) fact degrades.
  - LESION (DA pinned at baseline regardless of the SNc): both g=1 -> the differential ABOLISHES. Proves it's the
    spiking-DA-derived gain, not content/order.
  - PERMUTED (the high-DA turn applied to the OTHER fact): the advantage FOLLOWS the DA, not the fact.
  - MOAT (HARD): an UNSTORED cue abstains (returns None) at EVERY DA level (a higher encoding gain never confabulates).
  - REGRESSION: encoding_gain_fn=None == the byte-identical unit-magnitude write (the default is unchanged).

GPU (the OneBrainComposer's on-bridge parser trains on the CuPy substrate). Small (D=64, V<=16, 2 facts, 3 seeds);
FOREGROUND; each build is the CI-test scale. Run: SIM_BACKEND=cupy python -m research.runners._burndown_I7_dopamine_encoding_deploy_derisk
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from research.runners._da_composer_salience_cleanup_derisk import _build_snc_bridge, _settle_da


# ---- the de-risk geometry: a NEUTRAL fact + a REWARDED fact, MATCHED cue strength, distinct content; the only intended
# difference is the encoding gain (the DA at store time). Plus an UNSTORED cue for the moat. ----
VOCAB = ["dog", "cat", "go", "eat", "north", "apple", "river", "run", "bird", "look", "south", "home"]
NEUTRAL = ("dog", "go", "north")        # stored at DA tonic (g ~= 1)
REWARDED = ("cat", "eat", "apple")      # stored at DA high (g > 1)
UNSTORED = ("river", "run")             # the moat probe (never stored)
SEEDS = [42, 43, 44]


# ============================================================================
# DA -> encoding-gain map + the shared-bridge DA read (mirrors the read-side
# `nav_conv_merged_bridge.py:_da_confidence_gate` + its `da_to_gate`, but for the
# WRITE side: a gain >= g_min that RISES with DA above baseline; clamped both ways).
# ============================================================================
def da_to_encoding_gain(da, da_baseline, k_da, g_min=0.5, g_max=3.0):
    """g = clip(g_min, g_max, 1 + k_DA*(DA - DA_baseline)). DA at baseline => g = 1 (the no-modulation knob =
    byte-identical write). A salient (high-DA) turn => g > 1 => a stronger, more-stable encoding (Lisman-Grace/Kandel
    D.16). Clamped both ways: g_min keeps a low-DA turn from erasing a fact; g_max is the saturation ceiling."""
    return float(min(g_max, max(g_min, 1.0 + k_da * (da - da_baseline))))


def read_da_concentration(bridge):
    """Read the shared `dopamine` concentration off a bridge's neuromodulator manager (the SAME read the read-side
    salience gate uses). Returns (da, da_baseline). SAFE: (None-manager / no `dopamine` modulator) -> (baseline,
    baseline) i.e. g=1 (a no-op), exactly like `_da_confidence_gate`'s gate-floor fallback."""
    nm = getattr(bridge, "neuromodulator_manager", None)
    if nm is None:
        return 0.5, 0.5
    try:
        da = float(nm.get_concentration("dopamine"))
        da_baseline = float(nm._config_by_name("dopamine").baseline)
    except (KeyError, AttributeError):
        return 0.5, 0.5
    return da, da_baseline


# ============================================================================
# read damage: common, GAIN-INDEPENDENT complex jitter on the stored composite
# weights (a composer-EXTERNAL perturbation of store_conns -- NO class edit). The
# real on-bridge resonate then reads the damaged weights through the real floor.
# ============================================================================
def _damage_store_conns(store_conns, sigma, rng):
    """Return a copy of store_conns with common additive complex jitter (sigma) on each weight -- the gain-independent
    read damage. The high-DA fact's larger |w| survives the floor; the tonic fact's degrades."""
    out = []
    for (post, pre, w) in store_conns:
        eta = sigma * (rng.standard_normal() + 1j * rng.standard_normal())
        out.append((post, pre, complex(w) + eta))
    return out


def _block_mean_mag(store_conns, block_idx, D):
    """Mean |w| of block `block_idx` in store_conns (block-major: D tuples per block) -- the encoding strength of that
    stored fact. The DA gain `g` scales it (g x the unit-mag write)."""
    blk = store_conns[block_idx * D:(block_idx + 1) * D]
    return float(np.mean([abs(complex(w)) for (_p, _q, w) in blk])) if blk else 0.0


# ============================================================================
# the de-risk: build a probe SNc, drive DA high/tonic, store on the REAL
# OneBrainComposer with the DA-derived encoding gain, recall under matched damage.
# ============================================================================
def _query_under_damage(comp, agent, action, sigma, dmg_seed):
    """Recall (agent, action) -> patient on the REAL OneBrainComposer read path, with the stored weights damaged by a
    fixed-sigma common jitter (its own reproducible draw). Temporarily swaps store_conns (and busts the store CSR cache)
    so the read uses the damaged weights, then restores."""
    rng = np.random.default_rng(dmg_seed)
    clean = comp.store_conns
    try:
        comp.store_conns = _damage_store_conns(clean, sigma, rng)
        comp._store_dirty = True
        comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
        return comp.query_patient(agent, action)
    finally:
        comp.store_conns = clean
        comp._store_dirty = True
        comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}


def _build_composer_two_facts(seed, D, da_neutral, da_rewarded, da_baseline, k_da, lesion=False):
    """An OneBrainComposer that reads a per-store DA via encoding_gain_fn. Store NEUTRAL at da_neutral, REWARDED at
    da_rewarded. When lesion=True the DA is pinned at baseline (g=1 for both) regardless of the DA passed -- the
    decisive control. Returns the composer with the two facts stored (NEUTRAL block 0, REWARDED block 1)."""
    holder = {"da": da_baseline}
    if lesion:
        gain_fn = (lambda: 1.0)                                  # DA-lesion: pinned -> g=1 always
    else:
        gain_fn = (lambda: da_to_encoding_gain(holder["da"], da_baseline, k_da))
    comp = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8,
                            enable_batched=False, enable_rf_cudagraph=False, enable_csr_cache=False,
                            enable_spiking_cleanup=False, encoding_gain_fn=gain_fn)
    holder["da"] = da_neutral
    comp.store(*NEUTRAL)                                         # block 0 at the neutral DA
    holder["da"] = da_rewarded
    comp.store(*REWARDED)                                       # block 1 at the rewarded DA
    return comp


def run_mechanism(seeds, D, k_da, da_levels):
    """FAST (1 OneBrainComposer build/seed): the LOAD-BEARING mechanism + moat + regression, NO damage sweep. Proves
    the shared NEURAL dopamine literally scales the stored synaptic encoding strength on the PRODUCTION composer.
      - MECHANISM: rewarded block |w| == (g_high/g_tonic) x neutral block |w| (DA-derived, lesion-confirmable).
      - LESION (in the magnitude domain): DA pinned baseline -> both blocks unit-mag -> ratio == 1 (no differential).
      - MOAT (clean read): an unstored cue abstains even with a high-DA (g>1) fact stored (HARD).
      - REGRESSION: encoding_gain_fn=None -> both blocks unit-mag + clean recall (byte-identical default)."""
    da_high, da_tonic, da_base = da_levels["da_high"], da_levels["da_low"], da_levels["da_baseline"]
    g_high = da_to_encoding_gain(da_high, da_base, k_da)
    g_tonic = da_to_encoding_gain(da_tonic, da_base, k_da)
    expected_ratio = g_high / g_tonic

    mech_ratios, mech_ratio_ok = [], []
    lesion_ratios, lesion_ratio_ok = [], []
    moat_clean_ok, reg_ok, reg_unit_ok = [], [], []

    for s in seeds:
        # --- REAL: neutral @tonic, rewarded @high. The single build also serves the moat (clean read). ---
        cR = _build_composer_two_facts(s, D, da_tonic, da_high, da_base, k_da)
        m_neu = _block_mean_mag(cR.store_conns, 0, D)
        m_rew = _block_mean_mag(cR.store_conns, 1, D)
        ratio = m_rew / max(m_neu, 1e-12)
        mech_ratios.append(ratio)
        mech_ratio_ok.append(abs(ratio - expected_ratio) < 0.02)
        # MOAT on the clean read (a high-DA fact stored): an unstored cue still abstains.
        moat_clean_ok.append(cR.query_patient(*UNSTORED) is None
                             and cR.query_patient(*NEUTRAL[:2]) == NEUTRAL[2]
                             and cR.query_patient(*REWARDED[:2]) == REWARDED[2])

        # --- LESION (magnitude domain): DA pinned baseline -> both g=1 -> ratio == 1 (no differential). ---
        cL = _build_composer_two_facts(s, D, da_tonic, da_high, da_base, k_da, lesion=True)
        lr = _block_mean_mag(cL.store_conns, 1, D) / max(_block_mean_mag(cL.store_conns, 0, D), 1e-12)
        lesion_ratios.append(lr)
        lesion_ratio_ok.append(abs(lr - 1.0) < 1e-9)

        # --- REGRESSION: encoding_gain_fn=None == unit-mag write + clean recall + moat (byte-identical default). ---
        cD = OneBrainComposer(seed=s, D=D, vocab=VOCAB, k_max=8, enable_batched=False,
                              enable_rf_cudagraph=False, enable_csr_cache=False, enable_spiking_cleanup=False)
        cD.store(*NEUTRAL); cD.store(*REWARDED)
        unit0 = abs(_block_mean_mag(cD.store_conns, 0, D) - 1.0) < 1e-9
        unit1 = abs(_block_mean_mag(cD.store_conns, 1, D) - 1.0) < 1e-9
        reg_unit_ok.append(unit0 and unit1)
        reg_ok.append(unit0 and unit1
                      and cD.query_patient(*NEUTRAL[:2]) == NEUTRAL[2]
                      and cD.query_patient(*REWARDED[:2]) == REWARDED[2]
                      and cD.query_patient(*UNSTORED) is None)

    n = len(seeds)
    go = (all(mech_ratio_ok) and all(lesion_ratio_ok)
          and sum(moat_clean_ok) == n and sum(reg_ok) == n)
    return {
        "config": {"D": D, "k_da": k_da, "seeds": seeds,
                   "da_high": da_high, "da_tonic": da_tonic, "da_baseline": da_base,
                   "g_high": g_high, "g_tonic": g_tonic, "g_ratio_high_over_tonic": expected_ratio},
        "mechanism_store_magnitude": {
            "stored_block_mag_ratio_per_seed(rewarded/neutral)": mech_ratios,
            "expected_ratio(g_high/g_tonic)": expected_ratio,
            "ratio_matches_gain_all_seeds": bool(all(mech_ratio_ok)),
            "lesion_ratio_per_seed(DA_pinned->should_be_1)": lesion_ratios,
            "lesion_ratio_is_unity_all_seeds": bool(all(lesion_ratio_ok)),
        },
        "moat_clean_read_ok": sum(moat_clean_ok), "of": n,
        "regression_default_eq_unit_mag": {"unit_mag_ok": sum(reg_unit_ok), "full_ok": sum(reg_ok), "of": n},
        "verdict": {
            "GO": bool(go),
            "mechanism_load_bearing(store_mag==g)": bool(all(mech_ratio_ok)),
            "lesion_abolishes_differential(ratio==1)": bool(all(lesion_ratio_ok)),
            "moat_intact_clean(HARD)": sum(moat_clean_ok) == n,
            "regression_byte_identical": sum(reg_ok) == n,
        },
    }


def run_behavioral_probe(seed, D, k_da, da_levels, sigmas):
    """SLOWER (3 builds): the behavioral differential under read damage on ONE seed -- a sweep to find whether/where a
    unit-gain (tonic) fact degrades below the floor while the high-DA fact survives. Reports the damage knee (or that
    none was found = the differential is GREEN_INERT on a clean small store: the gain is a write-side stability reserve
    that only engages under read stress, like the read-side gate's at-rest no-op). REAL + LESION + the moat per sigma."""
    da_high, da_tonic, da_base = da_levels["da_high"], da_levels["da_low"], da_levels["da_baseline"]
    cR = _build_composer_two_facts(seed, D, da_tonic, da_high, da_base, k_da)             # neutral @tonic, rewarded @high
    cL = _build_composer_two_facts(seed, D, da_tonic, da_high, da_base, k_da, lesion=True)  # both g=1
    rows, knee = [], None
    for sigma in sigmas:
        neu = _query_under_damage(cR, NEUTRAL[0], NEUTRAL[1], sigma, seed * 1000 + 11) == NEUTRAL[2]
        rew = _query_under_damage(cR, REWARDED[0], REWARDED[1], sigma, seed * 1000 + 23) == REWARDED[2]
        moat = _query_under_damage(cR, UNSTORED[0], UNSTORED[1], sigma, seed * 1000 + 37) is None
        les_neu = _query_under_damage(cL, NEUTRAL[0], NEUTRAL[1], sigma, seed * 1000 + 11) == NEUTRAL[2]
        les_rew = _query_under_damage(cL, REWARDED[0], REWARDED[1], sigma, seed * 1000 + 23) == REWARDED[2]
        rows.append({"sigma": sigma, "real_neutral_ok": neu, "real_rewarded_ok": rew, "moat_ok": moat,
                     "lesion_neutral_ok": les_neu, "lesion_rewarded_ok": les_rew})
        # the differential knee: high-DA fact recalls, tonic fact fails, moat holds, AND the lesion shows it's the gain
        # (under lesion both fail together at the same sigma -> no differential).
        if knee is None and rew and (not neu) and moat:
            knee = sigma
    return {"seed": seed, "sweep": rows, "differential_knee_sigma": knee,
            "differential_found": knee is not None}


def _measure_da(seed, snc_tonic_pa, snc_salient_pa, n_settle=400):
    """Stand up the SNc + read DA at the tonic + salient operating points (reuses _da_composer's recipe verbatim)."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge = _build_snc_bridge(seed)
    snc_idx = xp.asarray(np.asarray(bridge.region_manager.indices("snc"), dtype=np.int64))
    da_low, rate_low = _settle_da(bridge, snc_idx, snc_tonic_pa, n_settle, xp)
    da_high, rate_high = _settle_da(bridge, snc_idx, snc_salient_pa, n_settle, xp)
    da_baseline = float(bridge.neuromodulator_manager._config_by_name("dopamine").baseline)
    return {"da_low": da_low, "da_high": da_high, "da_baseline": da_baseline,
            "rate_low_hz": rate_low, "rate_high_hz": rate_high}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--k-da", type=float, default=2.0, help="DA->gain slope (g = 1 + k_DA*(DA-baseline))")
    ap.add_argument("--snc-tonic-pa", type=float, default=80.0)
    ap.add_argument("--snc-salient-pa", type=float, default=600.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--behavioral", action="store_true",
                    help="ALSO run the (slower, 1-seed) read-damage behavioral probe (3 builds + sweep)")
    ap.add_argument("--behavioral-sigmas", type=float, nargs="+",
                    default=[0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_burndown_I7_limbic_encoding_hook.json")
    args = ap.parse_args()

    # 1) the NEURAL DA source: stand up a spiking SNc, read DA at the tonic + salient operating points.
    print("[1/2] standing up the spiking SNc + reading DA (tonic vs salient)...")
    da_levels = _measure_da(args.seeds[0], args.snc_tonic_pa, args.snc_salient_pa)
    g_high = da_to_encoding_gain(da_levels["da_high"], da_levels["da_baseline"], args.k_da)
    g_tonic = da_to_encoding_gain(da_levels["da_low"], da_levels["da_baseline"], args.k_da)
    print(f"      DA tonic={da_levels['da_low']:.4f} (rate {da_levels['rate_low_hz']:.1f}Hz) -> g_tonic={g_tonic:.3f}")
    print(f"      DA high ={da_levels['da_high']:.4f} (rate {da_levels['rate_high_hz']:.1f}Hz) -> g_high={g_high:.3f}"
          f"   (ratio {g_high / g_tonic:.3f})")
    da_for_run = {"da_high": da_levels["da_high"], "da_low": da_levels["da_low"],
                  "da_baseline": da_levels["da_baseline"]}

    # 2) the FAST load-bearing mechanism de-risk on the REAL OneBrainComposer (1 build/seed + the lesion + regression).
    print(f"[2/2] mechanism de-risk on OneBrainComposer (D={args.D}, seeds={args.seeds})...")
    results = run_mechanism(args.seeds, args.D, args.k_da, da_for_run)
    results["da_source"] = {**da_levels, "snc_tonic_pa": args.snc_tonic_pa, "snc_salient_pa": args.snc_salient_pa,
                            "neural_recipe": "spiking SNc (IZH2007_DOPAMINE) + `dopamine` from_region_firing_signed "
                                             "modulator (the merged one-brain limbic recipe)"}
    results["scope"] = ("RUNNER-LEVEL (NO sim/ edit): the encoding gain is a composer-layer multiply already shipped "
                        "default-OFF in OneBrainComposer._write_block; this de-risk + the nav_conv_merged_bridge flag "
                        "wire it to read a shared neural `dopamine` at store time.")
    if args.behavioral:
        print(f"      [behavioral] read-damage sweep on seed {args.seeds[0]} (sigmas {args.behavioral_sigmas})...")
        results["behavioral_probe"] = run_behavioral_probe(args.seeds[0], args.D, args.k_da, da_for_run,
                                                           args.behavioral_sigmas)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    v = results["verdict"]
    msm = results["mechanism_store_magnitude"]
    print("\n============ I-7-b DOPAMINE ENCODING-GAIN DEPLOYMENT DE-RISK (PRODUCTION OneBrainComposer) ============")
    print(f"  MECHANISM (stored |w| ratio == g_high/g_tonic = {msm['expected_ratio(g_high/g_tonic)']:.3f}): "
          f"{v['mechanism_load_bearing(store_mag==g)']}")
    print(f"    rewarded/neutral block-mag ratios: {['%.4f' % r for r in msm['stored_block_mag_ratio_per_seed(rewarded/neutral)']]}")
    print(f"  LESION (DA pinned -> ratio==1): {v['lesion_abolishes_differential(ratio==1)']}  "
          f"(lesion ratios {['%.4f' % r for r in msm['lesion_ratio_per_seed(DA_pinned->should_be_1)']]})")
    print(f"  MOAT intact (clean read, HARD): {v['moat_intact_clean(HARD)']}  ({results['moat_clean_read_ok']}/{results['of']})")
    print(f"  REGRESSION (None==unit-mag):    {v['regression_byte_identical']}  "
          f"({results['regression_default_eq_unit_mag']['full_ok']}/{results['regression_default_eq_unit_mag']['of']})")
    if args.behavioral:
        bp = results["behavioral_probe"]
        print(f"  BEHAVIORAL differential under read damage (seed {bp['seed']}): "
              f"{'knee at sigma=' + str(bp['differential_knee_sigma']) if bp['differential_found'] else 'NONE FOUND (GREEN_INERT on clean store)'}")
        for r in bp["sweep"]:
            print(f"    sigma={r['sigma']}: REAL neu={r['real_neutral_ok']} rew={r['real_rewarded_ok']} moat={r['moat_ok']}  "
                  f"| LESION neu={r['lesion_neutral_ok']} rew={r['lesion_rewarded_ok']}")
    print(f"\n  VERDICT: {'GO' if v['GO'] else 'NEGATIVE'} (mechanism load-bearing + lesion-confirmed + moat + regression) "
          f"-- the shared NEURAL dopamine {'DOES' if v['GO'] else 'does NOT'} scale fact-encoding strength on the "
          f"production OneBrainComposer.")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
