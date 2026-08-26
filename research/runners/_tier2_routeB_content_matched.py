"""Tier-2 #6 ROUTE-B CONTENT-MATCHED test: the DECISIVE isolation of the DA-driven encoding gain from per-fact content.

THE BACKGROUND (research/findings/2026-06-22-tier2-routeB-deployment-smoke-LATENT.md): the single-fact-pair deployment
smoke (`_tier2_routeB_deployment_smoke.py`) showed the SHARED spiking dopamine MODULATES the conversational composer's
encoding gain end-to-end on the merged one brain (g 1.08->1.69 from the REAL SNc), but the within-fact recall
differential was LATENT -- diagnosed as the SPECIFIC FACT_HI/FACT_LO pair's intrinsic content-robustness asymmetry (at
the deployed D=128) dominating the achievable DA-gain spread. That single pair cannot tell "gain does nothing" apart
from "gain is swamped by THIS pair's content."

THE DECISIVE TEST (the LATENT finding's §"Next" #1): store N>>2 DISTINCT facts, RANDOMLY assign each to HI-DA or LO-DA
encoding (balanced ~half/half), so the per-fact content-robustness asymmetry AVERAGES OUT across the assignment. Then,
at a moat-safe read-damage knee (the noise where unit-gain facts START to fail but the no-confab moat still holds),
measure the MEAN recall rate of HI-DA-encoded facts vs LO-DA-encoded facts.

  The gain is REAL + behaviorally LOAD-BEARING iff mean(HI-DA recall) > mean(LO-DA recall) at a moat-safe knee,
  CONTROLLING for content (the random assignment).

WHAT IS DEPLOYMENT-FAITHFUL (identical to the smoke -- this is a sibling, reuse-by-import):
  * the DA SOURCE is the REAL spiking SNc on the MERGED bridge -- `limbic_snc` driven to two operating points (tonic
    I=80pA -> baseline; salient I=600pA -> DA~0.84) via the smoke's VERBATIM `_settle_snc`; the gain reads the LIVE
    `neuromodulator_manager.get_concentration("dopamine")` (`from_region_firing_signed`, the signed RPE), NOT a scalar.
  * the composer is the DEPLOYED merged path: `MergedNavConvAgent(co_resident_limbic=True).composer` (`MergedRFComposer`,
    `enable_substrate_store=True` so the gain multiplies the per-fact substrate magnitude and the read goes through the
    RF floor). The `encoding_gain_fn` hook + `_retrieve_noise` read-damage knob pre-exist. NO `sim/` edit.
  * each fact is stored by `agent.hear("agent action patient")` -> `composer.store(...)` (the LIVE DA's gain is baked
    into that fact's complex weights AT STORE TIME); `composer.query_patient(...)` reads it back under the read damage.

THE READ-DAMAGE MODEL (the smoke's §3.4, unchanged): the merged substrate store is PER-FACT (no cross-fact
superposition damage), so the physical read damage that exercises the RF magnitude floor is the composer's OWN
`_retrieve_noise` (common, gain-INDEPENDENT additive read noise of fixed sigma) -- the SAME damage model the numpy
de-risk used. Higher encoding gain -> higher per-fact readout magnitude (g*M) -> higher SNR under that fixed-sigma
noise -> survives to a higher noise. The knee SWEEP locates the noise where unit-gain facts start to fail while the
moat holds, and reports the HI-vs-LO mean recall THERE.

ANTI-CHEATS (all kept, single seed first; the controller reviews):
  (a) NO-CONFAB MOAT (HARD, never weaken): UNSTORED cues abstain (-> None) at the operating DA AND in the lesion;
      0 false-accepts at the chosen knee.
  (b) DA-LESION: re-store ALL facts at BASELINE DA (the HI/LO labels are then ARBITRARY) -> no recall difference
      between the (now meaningless) HI/LO groups (`lesion_effect` ~ 0). Isolates the gain from any label/content quirk.
  (c) REGRESSION: `encoding_gain_fn=None` + no read damage is byte-identical to the deployed default recall
      (all facts recalled, moat abstains) -- the default path did not drift.

GPU only (the MergedNavConvAgent Hebbian parser + dlPFC are CuPy-validated). Run:
    SIM_BACKEND=cupy python -m research.runners._tier2_routeB_content_matched --seed 42
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend, is_gpu_backend
from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
# Reuse-by-import: the smoke's VERBATIM SNc-settle, composer-prep, and reproducible-read-draw helpers.
from research.runners._tier2_routeB_deployment_smoke import _settle_snc, _prep_composer


# N=16 DISTINCT facts: each (agent, action) CUE pair is UNIQUE (so query_patient's first-match is unambiguous) and
# every word is in the merged 16-word probe vocab (rf_phasor_composer.DEFAULT_VOCAB) -> all parse + recall on this
# exact merged path. 4 agents (dog/cat + the vocab nouns apple/river acting positionally in the agent slot) x actions
# {go,run,come,stop,look} give 16 unique (agent,action) cues; patients span the spatial/noun/adj banks. Content is
# HETEROGENEOUS by design (the whole point: average out per-fact robustness asymmetry across the random HI/LO split).
FACTS = [
    # agent=dog
    ("dog", "go", "north"),
    ("dog", "run", "south"),
    ("dog", "come", "east"),
    ("dog", "stop", "west"),
    ("dog", "look", "river"),
    # agent=cat
    ("cat", "go", "apple"),
    ("cat", "run", "big"),
    ("cat", "come", "small"),
    ("cat", "stop", "hot"),
    ("cat", "look", "cold"),
    # agent=apple (a vocab noun in the agent slot -> unique cues)
    ("apple", "go", "dog"),
    ("apple", "run", "cat"),
    ("apple", "come", "north"),
    # agent=river
    ("river", "go", "south"),
    ("river", "run", "east"),
    ("river", "stop", "west"),
]
# the UNSTORED moat cue: a (agent, action) pair that is NOT any stored fact's cue (apple+stop is unused) -> must abstain
UNSTORED = ("apple", "stop")

DEFAULT_READ_FLOOR = 1.0e-2
DEFAULT_K_DA = 2.0                       # g = clip(1 + k_DA*(DA - baseline), 0.5, 3.0); DA~0.84 -> g~1.7
# the knee sweep grid (same span as the smoke; D=128 sits higher than the de-risk D=64 knee=260)
SWEEP_NOISE = (260.0, 400.0, 600.0, 900.0, 1300.0, 1800.0, 2500.0, 3400.0, 4500.0)


def _assert_unique_cues():
    cues = [(a, v) for (a, v, _p) in FACTS]
    assert len(cues) == len(set(cues)), f"FACT cues are NOT unique (first-match would be ambiguous): {cues}"
    assert UNSTORED not in cues, f"UNSTORED cue {UNSTORED} collides with a stored fact cue"


def _query_one(composer, agent, action, patient, seed, fact_i):
    """A recall with an independent-but-reproducible common read-damage draw, per (fact, query) -- the smoke's `_q`
    recipe generalized to N facts (reseed the read-noise RNG with a per-fact deterministic seed; the damage sigma is
    gain-INDEPENDENT). Returns True iff the recalled patient == the stored patient."""
    composer._retrieve_noise_rng = np.random.default_rng(seed * 100000 + fact_i)
    return composer.query_patient(agent, action) == patient


def _moat_abstains(composer, seed, tag_off=999):
    """The no-confab moat probe: an UNSTORED cue must abstain (-> None). Reproducible read draw."""
    composer._retrieve_noise_rng = np.random.default_rng(seed * 100000 + tag_off)
    return composer.query_patient(UNSTORED[0], UNSTORED[1]) is None


def _store_all(agent, snc_idx_x, i_low, i_high, labels):
    """Store every FACT once: drive the REAL shared limbic_snc to HI vs LO operating point per the fact's label BEFORE
    its `hear`, so the LIVE DA's encoding gain is baked into that fact's substrate weights. `labels[i] in {'HI','LO'}`."""
    for i, (a, v, p) in enumerate(FACTS):
        I_snc = i_high if labels[i] == "HI" else i_low
        _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=I_snc)
        agent.hear(f"{a} {v} {p}")


def _mean_recall_by_group(composer, labels, seed, noise):
    """At read-damage sigma `noise`, recall every fact and return (meanHI, meanLO, perfact_ok[list], moat_ok)."""
    composer._retrieve_noise = float(noise)
    ok = []
    for i, (a, v, p) in enumerate(FACTS):
        ok.append(bool(_query_one(composer, a, v, p, seed, i)))
    hi = [ok[i] for i in range(len(FACTS)) if labels[i] == "HI"]
    lo = [ok[i] for i in range(len(FACTS)) if labels[i] == "LO"]
    moat_ok = _moat_abstains(composer, seed)
    meanHI = float(np.mean(hi)) if hi else 0.0
    meanLO = float(np.mean(lo)) if lo else 0.0
    return meanHI, meanLO, ok, bool(moat_ok)


def run(seed, read_floor, k_da, i_low, i_high):
    """Single-seed Route-B CONTENT-MATCHED averaged test. Returns a results dict + the verdict line."""
    _assert_unique_cues()
    xp, _ = get_backend()

    # --- the merged bridge with the SHARED limbic core (the real `dopamine` SNc) ---
    agent = MergedNavConvAgent(seed=seed, co_resident_limbic=True)   # Route B: salience GATE off (default)
    nm = agent._merged_bridge.neuromodulator_manager
    assert nm is not None and "dopamine" in nm.modulator_names(), "the shared dopamine modulator must be present"
    da_base = float(nm._config_by_name("dopamine").baseline)         # 0.5
    snc_idx = np.asarray(agent._merged_bridge.region_manager.indices("limbic_snc"), dtype=np.int64)
    snc_idx_x = xp.asarray(snc_idx)

    # the DA-gated encoding gain (reads the LIVE shared dopamine at store time): g = clip(1 + k*(DA-baseline), 0.5, 3.0)
    def gain_fn():
        da = float(nm.get_concentration("dopamine"))
        return float(np.clip(1.0 + k_da * (da - da_base), 0.5, 3.0))

    # --- BALANCED random HI/LO assignment (the content control): ~half HI, ~half LO, shuffled by the seed RNG ---
    N = len(FACTS)
    rng = np.random.default_rng(seed)
    labels = np.array(["HI"] * (N // 2) + ["LO"] * (N - N // 2))
    rng.shuffle(labels)
    labels = list(labels)

    # measure the realized gains at the two operating points (printed; also confirms g_hi>g_lo is applied)
    _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=i_high); da_high = float(nm.get_concentration("dopamine")); g_hi = gain_fn()
    _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=i_low);  da_low = float(nm.get_concentration("dopamine")); g_lo = gain_fn()
    gain_applied = bool(g_hi > g_lo + 1e-9)

    # ====================================================================================================
    # MAIN: store all N facts at their assigned DA, then SWEEP the read damage to find a moat-safe knee where unit-gain
    #       facts START to fail and the moat holds; report mean(HI) vs mean(LO) THERE (content averaged out).
    # ====================================================================================================
    c = _prep_composer(agent, noise=0.0, read_floor=read_floor, encoding_gain_fn=gain_fn)
    _store_all(agent, snc_idx_x, i_low, i_high, labels)

    sweep = []
    for sig in SWEEP_NOISE:
        meanHI, meanLO, ok, moat_ok = _mean_recall_by_group(c, labels, seed, sig)
        overall = float(np.mean(ok))
        sweep.append({"noise": sig, "meanHI": meanHI, "meanLO": meanLO, "overall": overall,
                      "effect": meanHI - meanLO, "moat_ok": moat_ok,
                      "n_HI_ok": int(sum(ok[i] for i in range(N) if labels[i] == "HI")),
                      "n_LO_ok": int(sum(ok[i] for i in range(N) if labels[i] == "LO"))})

    # the MOAT-SAFE KNEE: the noise level where (i) the moat holds AND (ii) unit-gain facts have STARTED to fail
    # (overall recall has dropped below ~1.0 but not collapsed to 0) -- i.e. the regime that exercises the RF floor.
    # Pick the moat-safe level with overall recall closest to 0.5 (the most discriminating damage); among ties prefer
    # the one with the largest |effect| sample. Fall back to any moat-safe partial-failure level.
    moat_safe = [s for s in sweep if s["moat_ok"] and 0.0 < s["overall"] < 1.0]
    knee = None
    if moat_safe:
        knee = min(moat_safe, key=lambda s: abs(s["overall"] - 0.5))

    # n_flip: among facts, how many flip recall HI-vs-LO at the knee is captured by the per-group counts; report the
    # knee effect + the per-group ok-counts. (Effect size = meanHI - meanLO at the knee.)
    if knee is not None:
        meanHI_knee, meanLO_knee = knee["meanHI"], knee["meanLO"]
        effect = knee["effect"]
        knee_noise = knee["noise"]
        n_HI_ok, n_LO_ok = knee["n_HI_ok"], knee["n_LO_ok"]
        n_HI = sum(1 for l in labels if l == "HI"); n_LO = sum(1 for l in labels if l == "LO")
        moat_FA_knee = 0 if knee["moat_ok"] else 1
    else:
        meanHI_knee = meanLO_knee = effect = float("nan")
        knee_noise = None; n_HI_ok = n_LO_ok = 0; n_HI = n_LO = N // 2; moat_FA_knee = 0

    # ====================================================================================================
    # (b) DA-LESION: re-store ALL facts at BASELINE DA (gain ~1 for every fact) -> the HI/LO labels are now ARBITRARY,
    #     so mean(HI) ~ mean(LO) at the SAME knee (no gain to separate them). Effect must collapse to ~0.
    # ====================================================================================================
    c_les = _prep_composer(agent, noise=0.0, read_floor=read_floor, encoding_gain_fn=gain_fn)
    for (a, v, p) in FACTS:
        _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=i_low)   # BASELINE DA for EVERY fact
        agent.hear(f"{a} {v} {p}")
    lesion_noise = knee_noise if knee_noise is not None else 1300.0
    les_meanHI, les_meanLO, les_ok, les_moat = _mean_recall_by_group(c_les, labels, seed, lesion_noise)
    lesion_effect = les_meanHI - les_meanLO    # labels arbitrary under baseline DA -> should be ~0

    # ====================================================================================================
    # (c) REGRESSION: encoding_gain_fn=None + no read damage -> the facts + moat are recalled EXACTLY as the deployed
    #     default (substrate store at unit gain, no degradation). Byte-identical default path.
    # ====================================================================================================
    c_reg = _prep_composer(agent, noise=0.0, read_floor=read_floor, encoding_gain_fn=None)
    for (a, v, p) in FACTS:
        agent.hear(f"{a} {v} {p}")
    reg_ok = all(c_reg.query_patient(a, v) == p for (a, v, p) in FACTS)
    reg_moat = c_reg.query_patient(UNSTORED[0], UNSTORED[1]) is None
    regression_identical = bool(reg_ok and reg_moat)

    # --- VERDICT ---
    # GO       : a moat-safe knee exists with mean(HI) > mean(LO) by a meaningful margin AND the DA-lesion nulls it.
    # LATENT   : a moat-safe knee exists but mean(HI) ~ mean(LO) (no content-controlled gain effect) -> the
    #            encoding-strength gain genuinely does not help on the deployed D=128 read model (boundary stands).
    # real-but-small: mean(HI) > mean(LO) positive but tiny / few facts flip -> a real-but-content-swamped lever.
    # NEGATIVE : the moat breaks at the knee (HARD violation), the regression drifts, or the effect is NEGATIVE
    #            (mean(LO) > mean(HI), gain anti-helps) with the lesion not explaining it.
    GO_EFFECT = 0.15           # mean-recall margin to call the gain behaviorally load-bearing (>=~2-3 facts of 8)
    SMALL_EFFECT = 1.0e-9      # any positive
    LESION_NULL = 0.10         # the lesion effect must be within this of 0 to credit the gain (not a label/content quirk)

    moat_ok_knee = bool(knee is None or knee["moat_ok"])
    if not regression_identical:
        verdict = "NEGATIVE"                   # default path drifted
    elif knee is None:
        verdict = "LATENT"                     # no moat-safe partial-failure knee (read too gentle even at max sweep)
    elif not moat_ok_knee:
        verdict = "NEGATIVE"                   # moat breach at the knee = HARD-gate violation
    elif effect >= GO_EFFECT and abs(lesion_effect) <= LESION_NULL:
        verdict = "GO"                         # content-controlled gain effect, lesion-nulled, moat-safe
    elif effect >= GO_EFFECT and abs(lesion_effect) > LESION_NULL:
        verdict = "NEGATIVE"                   # the "effect" survives the DA-lesion -> it's a label/content quirk, not DA
    elif effect > SMALL_EFFECT and abs(lesion_effect) <= LESION_NULL:
        verdict = "GO-small"                   # real but content-swamped (positive, but below the load-bearing margin)
    elif effect <= -GO_EFFECT:
        verdict = "NEGATIVE"                   # the gain ANTI-helps (content-robustness inverts it) at a moat-safe knee
    else:
        verdict = "LATENT"                     # meanHI ~ meanLO -> encoding-strength gain does not help on this read

    n_flip = abs(n_HI_ok - n_LO_ok)            # rough "how many facts flip between groups" at the knee
    line = (f"seed {seed} content-matched: meanHI={meanHI_knee:.3f} meanLO={meanLO_knee:.3f} effect={effect:+.3f} "
            f"(N={N}, n_flip={n_flip}, knee_noise={knee_noise}) | moat_FA={moat_FA_knee} | "
            f"lesion_effect={lesion_effect:+.3f} -> {verdict}")

    results = {
        "seed": seed,
        "config": {"read_floor": read_floor, "k_da": k_da, "i_low": i_low, "i_high": i_high,
                   "D": int(agent.composer.D), "N_facts": N, "GO_EFFECT": GO_EFFECT, "LESION_NULL": LESION_NULL},
        "labels": labels,
        "da": {"low": da_low, "high": da_high, "baseline": da_base},
        "encoding_gain": {"at_low_DA": g_lo, "at_high_DA": g_hi, "applied(hi>lo)": gain_applied},
        "knee": {"noise": knee_noise, "meanHI": meanHI_knee, "meanLO": meanLO_knee, "effect": effect,
                 "n_HI_ok": n_HI_ok, "n_LO_ok": n_LO_ok, "n_HI": n_HI, "n_LO": n_LO,
                 "moat_FA": moat_FA_knee},
        "sigma_knee_sweep": sweep,
        "lesion": {"noise": lesion_noise, "meanHI": les_meanHI, "meanLO": les_meanLO,
                   "lesion_effect": lesion_effect, "moat_ok": bool(les_moat)},
        "regression_default_identical": regression_identical,
        "verdict": verdict,
        "verdict_line": line,
    }
    return results, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--read-floor", type=float, default=DEFAULT_READ_FLOOR)
    ap.add_argument("--k-da", type=float, default=DEFAULT_K_DA, help="encoding-gain slope: g=clip(1+k*(DA-0.5),0.5,3)")
    ap.add_argument("--i-low", type=float, default=80.0, help="tonic SNc drive pA -> DA~baseline")
    ap.add_argument("--i-high", type=float, default=600.0, help="salient SNc drive pA -> DA~0.84")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_tier2_routeB_content_matched.json")
    args = ap.parse_args()

    assert is_gpu_backend(), "the MergedNavConvAgent parser/dlPFC are GPU-validated; run with SIM_BACKEND=cupy"

    print("=" * 110)
    print("Tier-2 #6 ROUTE-B CONTENT-MATCHED -- DA encoding gain isolated from per-fact content (N facts, random HI/LO)")
    print("  (MergedNavConvAgent + co_resident_limbic; the real merged `dopamine` SNc bakes encoding_gain_fn per fact)")
    print("=" * 110)

    results, line = run(args.seed, args.read_floor, args.k_da, args.i_low, args.i_high)

    da = results["da"]; eg = results["encoding_gain"]
    print(f"\n  DA_low  = {da['low']:.3f} -> encoding gain g = {eg['at_low_DA']:.3f}")
    print(f"  DA_high = {da['high']:.3f} -> encoding gain g = {eg['at_high_DA']:.3f}   (applied hi>lo: {eg['applied(hi>lo)']})")
    print(f"  labels (HI/LO per fact, seed-shuffled, balanced): {results['labels']}")
    print(f"\n  sigma-KNEE SWEEP (D={results['config']['D']}, N={results['config']['N_facts']} facts; "
          f"HI baked g={eg['at_high_DA']:.2f}, LO baked g={eg['at_low_DA']:.2f}; vary read damage):")
    knee_noise = results["knee"]["noise"]
    for s in results["sigma_knee_sweep"]:
        flag = "  <- KNEE (moat-safe, partial-fail)" if (knee_noise is not None and s["noise"] == knee_noise) else ""
        print(f"    noise={s['noise']:>6.0f}: meanHI={s['meanHI']:.3f} meanLO={s['meanLO']:.3f} "
              f"effect={s['effect']:+.3f} overall={s['overall']:.3f} moat={int(s['moat_ok'])}{flag}")
    k = results["knee"]
    print(f"\n  KNEE @ noise={k['noise']}: meanHI={k['meanHI']:.3f} ({k['n_HI_ok']}/{k['n_HI']}) "
          f"meanLO={k['meanLO']:.3f} ({k['n_LO_ok']}/{k['n_LO']}) effect={k['effect']:+.3f}  moat_FA={k['moat_FA']}")
    les = results["lesion"]
    print(f"  LESION @ noise={les['noise']} (all facts baseline-DA): meanHI={les['meanHI']:.3f} meanLO={les['meanLO']:.3f} "
          f"effect={les['lesion_effect']:+.3f}  (should be ~0)  moat_ok={les['moat_ok']}")
    print(f"  REGRESSION (encoding_gain_fn=None == default): {results['regression_default_identical']}")
    print("\n" + "-" * 110)
    print("  " + line)
    print("-" * 110)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
