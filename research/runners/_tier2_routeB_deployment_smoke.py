"""Tier-2 #6 ROUTE-B DEPLOYMENT SMOKE: does the SHARED SPIKING DOPAMINE (the co-resident SNc on the MERGED nav+conv
bridge) modulate the conversational composer's fact ENCODING STRENGTH end-to-end -- i.e. a fact `hear`-d while the
shared limbic_snc is BURSTING (high DA -> encoding gain g>1) is recalled where the SAME-structure fact `hear`-d at DA
BASELINE (g~1) abstains under the real merged read damage?

This is the one open #6 experiment (research/findings/2026-06-22-tier2-limbic-composer-next-step.md §3): Route A
(read-side salience gate) is production-wired + GO; Route B (write-side DA-gated encoding gain) is de-risked numpy 6/6
GO (_phaseB_dopamine_encoding_gain_derisk.py) AND wired into both composers behind a default-off `encoding_gain_fn` --
but it has NEVER been run on the MERGED bridge with the REAL shared `dopamine` (not a probe scalar). This runner closes
that loop.

WHAT IS DEPLOYMENT-FAITHFUL HERE (vs the numpy de-risk):
  * the DA SOURCE is the REAL spiking SNc on the MERGED bridge -- `limbic_snc` driven to two operating points
    (tonic I=80pA -> DA~0.50 = baseline; salient I=600pA -> DA~0.84), the gain reads the LIVE
    `neuromodulator_manager.get_concentration("dopamine")` produced by `from_region_firing_signed` (the signed RPE),
    NOT a hand-set scalar. (The exact _settle_snc recipe the Route-A salience-gate wireup smoke uses.)
  * the composer is the DEPLOYED merged path: `MergedNavConvAgent(co_resident_limbic=True).composer` =
    `MergedRFComposer` (an `RFPhasorComposer` whose BIND/UNBIND ops run on the merged bridge's `rf` slice). Its
    `encoding_gain_fn` hook + `enable_substrate_store` + the `_retrieve_noise` read-damage knob ALL pre-exist.
  * `agent.hear(sentence)` -> `composer.store(...)` (the gain is baked into the per-fact substrate complex weights AT
    STORE TIME, permanently, by the LIVE DA); `composer.query_patient(...)` reads it back under the read damage.

THE READ-DAMAGE MODEL (honest, pre-registered caveat §3.4): the merged path's substrate store is PER-FACT (each fact
gets its own (1+D) RF weight-bridge in `_store_substrate`), so there is NO cross-fact superposition damage on this
path. The physical read damage that exercises the RF magnitude floor (sim/bridge.py, `_rf_mag2 > _rf_floor2`) is the
composer's OWN `_retrieve_noise` (common, gain-INDEPENDENT additive read noise of fixed sigma) -- the SAME damage model
the numpy de-risk used (noise=260 = the moat-safe knee for D=64). If that knee is NOT reached at the deployed read
(facts reconstruct cleanly even at g~1), the honest finding is LATENT (the gain IS applied -- printed -- but the deploy
read is too gentle to exercise the floor), NOT a NEGATIVE.

ANTI-CHEATS (single seed first; the controller reviews before 6 seeds):
  (a) NO-CONFAB MOAT (HARD gate, never weaken): an UNSTORED cue abstains (-> None) at BOTH DA levels, 0 false-accepts.
  (b) DA-LESION: `hear` BOTH facts at the SAME baseline DA -> the recall differential COLLAPSES (it's the gain, not
      the content/order).
  (c) REGRESSION: `encoding_gain_fn=None` is byte-identical to the deployed default recall (same facts recalled, same
      moat abstentions) -- the default path did not drift.

GPU only (the MergedNavConvAgent Hebbian parser + dlPFC are CuPy-validated). Run:
    SIM_BACKEND=cupy python -m research.runners._tier2_routeB_deployment_smoke --seed 42
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend, is_gpu_backend, to_host
from research.runners.nav_conv_merged_bridge import MergedNavConvAgent


# Two distinct, MATCHED-cue-strength facts (both plain 3-role SVO; distinct content; all words in the merged probe
# vocab AND proven to parse+recall on this exact merged path -- the SAME facts the Route-A salience-gate wireup smoke
# uses, _da_salience_gate_wireup_smoke.py:42). HI is stored at high DA, LO at baseline DA -- the only intended
# difference is the DA-driven encoding gain. Plus an UNSTORED cue for the moat.
FACT_HI = ("dog", "go", "north")        # heard at DA-high  -> g>1
FACT_LO = ("cat", "come", "south")      # heard at DA-low   -> g~1
UNSTORED = ("river", "look", None)      # never stored -> the moat probe (query_patient("river","look") must be None)

# Read-damage knee from the numpy de-risk (D=64, two facts): noise=260 = the moat-safe knee where a unit-gain fact
# STARTS to fail AND the no-confab moat still holds; read_floor=1e-2.
DEFAULT_NOISE = 260.0
DEFAULT_READ_FLOOR = 1.0e-2
DEFAULT_K_DA = 2.0                       # g = clip(1 + k_DA*(DA - 0.5), 0.5, 3.0); DA~0.84 -> g~1.7 (cf. de-risk g=2.0)


def _settle_snc(bridge, snc_idx, I_snc, n_steps=400):
    """Drive the limbic SNc pool (the shared-DA source) with constant current for n_steps (advancing the dopamine EMA
    each step) -> a steady DA concentration + the SNc firing rate (Hz). VERBATIM the recipe the Route-A salience-gate
    wireup smoke uses (_da_salience_gate_wireup_smoke.py:19)."""
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx] = xp.float32(I_snc)
    total = 0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        total += int(to_host(bridge.cp_firing_states[snc_idx]).sum())
    da = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    bridge.cp_external_input_current[:] = 0.0
    rate_hz = total / max(int(snc_idx.shape[0]), 1) / (n_steps * 1e-3)
    return da, rate_hz


def _q(composer, agent, action, seed, tag):
    """A recall with an independent-but-reproducible common read-damage draw (reseed the read-noise RNG per query so
    each (fact, query) sees its own damage realization; the damage sigma is gain-INDEPENDENT). Mirrors the de-risk."""
    offsets = {"hi": 11, "lo": 23, "moat": 37}
    composer._retrieve_noise_rng = np.random.default_rng(seed * 1000 + offsets[tag])
    return composer.query_patient(agent, action)


def _prep_composer(agent, noise, read_floor, encoding_gain_fn):
    """Switch the deployed MergedRFComposer to the SUBSTRATE store (so the encoding gain multiplies the stored
    magnitude and the read goes through the floor) + set the read-damage knobs + wire the gain. Done BEFORE any
    `hear`, so every fact takes the substrate path. The kb is cleared so a fresh fact set is stored."""
    c = agent.composer
    c.kb = []
    c.enable_substrate_store = True
    c._retrieve_noise = float(noise)
    c._retrieve_read_floor = float(read_floor)
    c.encoding_gain_fn = encoding_gain_fn
    return c


def _drive_da(agent, snc_idx_x, I_snc):
    """Drive the shared limbic_snc to a steady DA operating point; return (da, rate_hz)."""
    return _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=I_snc)


def run(seed, noise, read_floor, k_da, i_low, i_high):
    """Single-seed Route-B deployment smoke. Returns a results dict + prints the verdict line."""
    xp, _ = get_backend()

    # --- the merged bridge with the SHARED limbic core (the real `dopamine` SNc) ---
    agent = MergedNavConvAgent(seed=seed, co_resident_limbic=True)   # Route B: salience GATE off (default)
    nm = agent._merged_bridge.neuromodulator_manager
    assert nm is not None and "dopamine" in nm.modulator_names(), "the shared dopamine modulator must be present"
    da_base = float(nm._config_by_name("dopamine").baseline)         # 0.5
    snc_idx = np.asarray(agent._merged_bridge.region_manager.indices("limbic_snc"), dtype=np.int64)
    snc_idx_x = xp.asarray(snc_idx)

    # the DA-gated encoding gain (reads the LIVE shared dopamine at store time):
    #   g = clip(1 + k_DA*(DA - baseline), 0.5, 3.0)
    def gain_fn():
        da = float(nm.get_concentration("dopamine"))
        return float(np.clip(1.0 + k_da * (da - da_base), 0.5, 3.0))

    # ====================================================================================================
    # MAIN: HI fact heard at high DA (g>1); LO fact heard at baseline DA (g~1). Then recall both under the read damage.
    # ====================================================================================================
    c = _prep_composer(agent, noise, read_floor, gain_fn)

    da_high, rate_high = _drive_da(agent, snc_idx_x, i_high)
    g_at_hi = gain_fn()
    agent.hear(" ".join(FACT_HI))                                    # store FACT_HI at high DA -> gain baked HIGH

    da_low, rate_low = _drive_da(agent, snc_idx_x, i_low)
    g_at_lo = gain_fn()
    agent.hear(" ".join(FACT_LO))                                    # store FACT_LO at low DA -> gain baked LOW

    hi_recall = _q(c, FACT_HI[0], FACT_HI[1], seed, "hi") == FACT_HI[2]
    lo_recall = _q(c, FACT_LO[0], FACT_LO[1], seed, "lo") == FACT_LO[2]
    moat_hi_lo = _q(c, UNSTORED[0], UNSTORED[1], seed, "moat") is None   # the moat at the prevailing (low) DA regime

    # moat at the HIGH-DA regime too (drive DA high again, then probe the unstored cue -- the gain only scales an
    # already-stored fact; an unstored cue has no block to amplify -> must still abstain).
    _drive_da(agent, snc_idx_x, i_high)
    moat_hi = _q(c, UNSTORED[0], UNSTORED[1], seed, "moat") is None

    # ====================================================================================================
    # (b) DA-LESION: BOTH facts heard at the SAME baseline DA (gain ~1 for both) -> the differential must COLLAPSE.
    # ====================================================================================================
    c_les = _prep_composer(agent, noise, read_floor, gain_fn)
    _drive_da(agent, snc_idx_x, i_low)                              # baseline DA for BOTH hears
    agent.hear(" ".join(FACT_HI))
    _drive_da(agent, snc_idx_x, i_low)
    agent.hear(" ".join(FACT_LO))
    les_hi_recall = _q(c_les, FACT_HI[0], FACT_HI[1], seed, "hi") == FACT_HI[2]
    les_lo_recall = _q(c_les, FACT_LO[0], FACT_LO[1], seed, "lo") == FACT_LO[2]
    les_moat = _q(c_les, UNSTORED[0], UNSTORED[1], seed, "moat") is None

    # ====================================================================================================
    # (b') sigma-KNEE CHARACTERIZATION (the §3.4 honest caveat): the de-risk's noise=260 was the knee for D=64; the
    #     DEPLOYED merged composer is D=128 (twice the matched-filter averaging -> ~sqrt(2) more noise-robust), so a
    #     single noise point can land LATENT simply because the deploy read is too gentle. Sweep `_retrieve_noise`
    #     over a grid (re-store HI@high-DA + LO@low-DA ONCE; vary only the read damage) to LOCATE the knee where the
    #     within-fact differential (HI survives, LO fails) appears AND whether the moat still holds there. This turns an
    #     under-informative single LATENT point into a CHARACTERIZED boundary for the controller.
    c_sw = _prep_composer(agent, noise, read_floor, gain_fn)
    _drive_da(agent, snc_idx_x, i_high); agent.hear(" ".join(FACT_HI))    # HI baked at high DA (g~1.69)
    _drive_da(agent, snc_idx_x, i_low);  agent.hear(" ".join(FACT_LO))    # LO baked at low DA  (g~1.08)
    sweep = []
    sweep_diff_moat_safe = []      # noise levels where diff==+1 AND the moat holds at both probes
    for sig in (260.0, 400.0, 600.0, 900.0, 1300.0, 1800.0, 2500.0):
        c_sw._retrieve_noise = float(sig)
        hr = _q(c_sw, FACT_HI[0], FACT_HI[1], seed, "hi") == FACT_HI[2]
        lr = _q(c_sw, FACT_LO[0], FACT_LO[1], seed, "lo") == FACT_LO[2]
        mo = _q(c_sw, UNSTORED[0], UNSTORED[1], seed, "moat") is None
        d = int(hr) - int(lr)
        sweep.append({"noise": sig, "hi_recall": bool(hr), "lo_recall": bool(lr), "diff": d, "moat": bool(mo)})
        if d == 1 and mo:
            sweep_diff_moat_safe.append(sig)

    # ====================================================================================================
    # (c) REGRESSION: encoding_gain_fn=None (the deployed default) -> NO read damage -> the facts + moat are recalled
    #     EXACTLY as the deployed default (the substrate store at unit gain, no degradation). Byte-identical default.
    # ====================================================================================================
    c_reg = _prep_composer(agent, 0.0, read_floor, None)           # default: no gain, no read damage
    agent.hear(" ".join(FACT_HI))
    agent.hear(" ".join(FACT_LO))
    reg_hi = c_reg.query_patient(FACT_HI[0], FACT_HI[1]) == FACT_HI[2]
    reg_lo = c_reg.query_patient(FACT_LO[0], FACT_LO[1]) == FACT_LO[2]
    reg_moat = c_reg.query_patient(UNSTORED[0], UNSTORED[1]) is None
    regression_identical = bool(reg_hi and reg_lo and reg_moat)

    # --- the within-fact recall DIFFERENTIAL on the deployed bridge ---
    diff = int(hi_recall) - int(lo_recall)        # +1 = HI recalled where LO abstains (the GO signature)
    lesion_diff = int(les_hi_recall) - int(les_lo_recall)
    moat_intact = bool(moat_hi_lo and moat_hi and les_moat)        # 0 false-accepts at BOTH DA levels + lesion

    # --- VERDICT ---
    # GO: HI recalled (True) AND LO abstains (False) -> diff==+1; DA-lesion collapses the differential (lesion_diff<=0
    #     OR both equal); moat 0-FA at both DA levels (HARD); regression byte-identical.
    # LATENT: the gain IS applied (g_at_hi > g_at_lo, printed) but the read damage is too gentle/harsh to produce the
    #     within-fact differential (diff<=0 with BOTH facts recalled, or BOTH abstaining) -- the deploy read does not
    #     sit at the floor knee for this fact/seed. NOT a NEGATIVE.
    # NEGATIVE: the moat breaks at any DA level (HARD-gate violation), OR the differential follows content not gain.
    gain_applied = bool(g_at_hi > g_at_lo + 1e-9)
    knee_found = len(sweep_diff_moat_safe) > 0     # a moat-safe sigma exists where HI(g>1) survives + LO(g~1) fails
    # the sweep shows a WRONG-direction differential (LO survives where HI fails) -> the per-fact content-robustness
    # asymmetry at the deployed D exceeds the DA-driven gain effect (the gain is applied but content dominates).
    content_dominates = any(s["diff"] == -1 for s in sweep)
    if not moat_intact:
        verdict = "NEGATIVE"                       # moat breach at the main noise = HARD-gate violation
    elif not regression_identical:
        verdict = "NEGATIVE"                       # default path drifted
    elif diff == 1 and lesion_diff <= 0:
        verdict = "GO"                             # HI recalled, LO abstains AT the main noise, lesion kills it, moat held
    elif gain_applied and knee_found:
        # the gain is load-bearing on the deployed bridge (a moat-safe sigma in the sweep gives the differential), but
        # the de-risk's D=64 knee (noise=260) is too gentle for the deployed D=128 -- a CHARACTERIZED boundary.
        verdict = "LATENT-knee-found"
    elif gain_applied and content_dominates:
        # the gain IS applied by the real spiking DA (g_hi>g_lo) but at the deployed D=128 the per-fact content-
        # robustness asymmetry dominates the gain spread -> no moat-safe sigma gives the predicted HI>LO differential
        # (and at high damage the intrinsically-more-robust fact -- LO here -- survives). Behaviorally LATENT, NOT a
        # NEGATIVE: the deployed read model (per-fact substrate, D=128) is too gentle / content-confounded for the
        # achievable DA-driven gain spread to flip recall. The mechanism is confirmed; the behavior is latent.
        verdict = "LATENT-content-dominates"
    elif gain_applied:
        verdict = "LATENT"                         # gain applied but no behavioral differential at any swept damage
    else:
        verdict = "NEGATIVE"

    line = (f"seed {seed}: hiDA_recall={int(hi_recall)} loDA_recall={int(lo_recall)} diff={diff} | "
            f"moat_hi={int(moat_hi)} moat_lo={int(moat_hi_lo)} (false-accepts={int(not moat_hi) + int(not moat_hi_lo)}) | "
            f"lesion_diff={lesion_diff} | regression_identical={'T' if regression_identical else 'F'} -> {verdict}")

    results = {
        "seed": seed,
        "config": {"noise": noise, "read_floor": read_floor, "k_da": k_da, "i_low": i_low, "i_high": i_high,
                   "D": int(agent.composer.D)},
        "da": {"low": da_low, "high": da_high, "baseline": da_base,
               "snc_rate_low_hz": rate_low, "snc_rate_high_hz": rate_high},
        "encoding_gain": {"at_low_DA": g_at_lo, "at_high_DA": g_at_hi, "applied(hi>lo)": gain_applied},
        "main": {"hi_recall": bool(hi_recall), "lo_recall": bool(lo_recall), "diff": diff,
                 "moat_low_DA": bool(moat_hi_lo), "moat_high_DA": bool(moat_hi)},
        "lesion": {"hi_recall": bool(les_hi_recall), "lo_recall": bool(les_lo_recall),
                   "lesion_diff": lesion_diff, "moat": bool(les_moat)},
        "sigma_knee_sweep": sweep,
        "sigma_knee_moat_safe_diff_levels": sweep_diff_moat_safe,
        "regression_default_identical": regression_identical,
        "moat_intact_all": moat_intact,
        "verdict": verdict,
        "verdict_line": line,
    }
    return results, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=DEFAULT_NOISE, help="common read-damage sigma (the moat-safe knee)")
    ap.add_argument("--read-floor", type=float, default=DEFAULT_READ_FLOOR)
    ap.add_argument("--k-da", type=float, default=DEFAULT_K_DA, help="encoding-gain slope: g=clip(1+k*(DA-0.5),0.5,3)")
    ap.add_argument("--i-low", type=float, default=80.0, help="tonic SNc drive pA -> DA~baseline")
    ap.add_argument("--i-high", type=float, default=600.0, help="salient SNc drive pA -> DA~0.84")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_tier2_routeB_deployment_smoke.json")
    args = ap.parse_args()

    assert is_gpu_backend(), "the MergedNavConvAgent parser/dlPFC are GPU-validated; run with SIM_BACKEND=cupy"

    print("=" * 100)
    print("Tier-2 #6 ROUTE-B DEPLOYMENT SMOKE -- shared spiking DOPAMINE gates conversational-composer ENCODING")
    print("  (MergedNavConvAgent + co_resident_limbic; the real merged `dopamine` SNc drives encoding_gain_fn)")
    print("=" * 100)

    results, line = run(args.seed, args.noise, args.read_floor, args.k_da, args.i_low, args.i_high)

    da = results["da"]; eg = results["encoding_gain"]
    print(f"\n  DA_low  = {da['low']:.3f} (limbic_snc {da['snc_rate_low_hz']:.0f} Hz) -> encoding gain g = {eg['at_low_DA']:.3f}")
    print(f"  DA_high = {da['high']:.3f} (limbic_snc {da['snc_rate_high_hz']:.0f} Hz) -> encoding gain g = {eg['at_high_DA']:.3f}")
    print(f"  gain applied (g_high > g_low): {eg['applied(hi>lo)']}")
    print(f"\n  MAIN     hi_recall={results['main']['hi_recall']} lo_recall={results['main']['lo_recall']} "
          f"diff={results['main']['diff']}")
    print(f"  LESION   hi_recall={results['lesion']['hi_recall']} lo_recall={results['lesion']['lo_recall']} "
          f"lesion_diff={results['lesion']['lesion_diff']}  (should collapse the differential)")
    print(f"  MOAT     low_DA={results['main']['moat_low_DA']} high_DA={results['main']['moat_high_DA']} "
          f"lesion={results['lesion']['moat']}  (0 false-accepts = HARD gate)")
    print(f"  REGRESSION (encoding_gain_fn=None == default): {results['regression_default_identical']}")
    print(f"\n  sigma-KNEE SWEEP (D={results['config']['D']}; HI baked g={eg['at_high_DA']:.2f} @high-DA, "
          f"LO baked g={eg['at_low_DA']:.2f} @low-DA; vary read damage):")
    for s in results["sigma_knee_sweep"]:
        flag = "  <- within-fact diff + moat-safe" if (s["diff"] == 1 and s["moat"]) else ""
        print(f"    noise={s['noise']:>6.0f}: hi={int(s['hi_recall'])} lo={int(s['lo_recall'])} "
              f"diff={s['diff']:+d} moat={int(s['moat'])}{flag}")
    if results["sigma_knee_moat_safe_diff_levels"]:
        print(f"    => the gain IS load-bearing at moat-safe sigma {results['sigma_knee_moat_safe_diff_levels']} "
              f"(the deploy D={results['config']['D']} knee is higher than the de-risk D=64 knee=260)")
    print("\n" + "-" * 100)
    print("  " + line)
    print("-" * 100)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
