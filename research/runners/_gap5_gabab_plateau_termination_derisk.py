"""Does GABA_B (slow K+ inhibition) TERMINATE the apical plateau and thereby NARROW the learned place field?

MECHANISM (external search 2026-07-30 + engine inspection): BTSP writes a field for as long as its plateau
lasts. The plateau HOLDS because a v-gated self-regenerating conductance keeps the apical compartment above
`coincidence_plateau_v_hold` (-35 mV). GABA_B reverses at -90 mV (E_K/GIRK) and decays over 150 ms -- slow
enough to bound a hundreds-of-ms plateau, where GABA_A's ~10 ms is far too fast. Both features already exist
and are default-off; this composes them, no sim/ edit.

PRE-REGISTERED, stated before the run (the discipline that made the rest of this arc trustworthy):
  * w_gabab=0 MUST reproduce the validated baseline (width ~16/60, place-specific circ ~0.59).
  * SUCCESS = width FALLS toward the sigma=5 oracle's ~12/60 while place-specific circ (sweep - randset) is
    HELD OR IMPROVED. Narrowing that also kills place-specificity is over-inhibition, not success.
  * FAILURE MODE TO EXPECT: too much GABA_B abolishes the plateau entirely -> dW ~ 0 -> UNDEFINED, not a
    negative (O'Keefe-Nadel's own "too many inputs totally inhibit the unit" horn). Report mean|dW| so that
    case is visible rather than scored.
  * A flat result refutes THIS ROUTE to plateau termination, not the capability.
"""
import argparse, json, os, sys
os.environ.setdefault("SIM_BACKEND", "cupy")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.INFO)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim import SimulationBridge
import importlib
B = importlib.import_module("research.runners._gap5_btsp_place_field_derisk")
NPLACE, NREAD = B.NPLACE, B.NREAD


def build(seed, w_gabab, w_max=2500.0, w0=600.0):
    """place -> read (BTSP, coincidence subunit) + an inhibitory pool driving GABA_B onto the readers."""
    R = [BrainRegion(name="place", n_neurons=NPLACE, exc_fraction=1.0, internal_density=0.0),
         BrainRegion(name="read", n_neurons=NREAD, exc_fraction=1.0, internal_density=0.0)]
    P = [RegionPathway(from_region="place", to_region="read", density=1.0, weight_mean=w0,
                       weight_jitter=0.3, plastic=True, coincidence_detector=True)]
    if w_gabab > 0:
        # SST/Martinotti-like: driven BY the readers, inhibiting them back on the slow GABA_B/GIRK timecourse
        R.append(BrainRegion(name="sst", n_neurons=NREAD, exc_fraction=0.0, internal_density=0.0))
        P += [RegionPathway(from_region="read", to_region="sst", density=1.0, weight_mean=300.0,
                            weight_jitter=0.0, plastic=False),
              RegionPathway(from_region="sst", to_region="read", density=1.0, weight_mean=w_gabab,
                            weight_jitter=0.0, plastic=False)]
    cfg = CoreSimConfig(seed=seed, dt_ms=1.0, enable_brain_region_framework=True, brain_regions=R,
                        region_pathways=P, enable_hebbian_learning=False, enable_stdp=False,
                        enable_homeostasis=False, enable_structural_plasticity=False, enable_ou_process=False,
                        enable_btsp=True, btsp_w_max=w_max, btsp_w_min=0.0, btsp_learning_rate=0.002,
                        btsp_hetero_dep=0.2, btsp_elig_exponent=4.0, btsp_elig_tau_ms=1000.0,
                        enable_two_compartment_dap=True, enable_coincidence_detection=True,
                        coincidence_k_threshold=4.0,
                        enable_gabab=(w_gabab > 0))     # THE LEVER
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def stats(dW):
    pk, wd = [], []
    for row in dW:
        r = np.maximum(row, 0.0)
        if r.max() <= 0:
            continue
        ab = r > 0.5 * r.max(); runs, prev = 0, ab[-1]
        for a_ in ab:
            if a_ and not prev:
                runs += 1
            prev = a_
        pk.append(max(runs, 1)); wd.append(int(ab.sum()))
    return (float(np.mean(pk)), float(np.mean(wd))) if pk else (float("nan"), float("nan"))


def run(seed, w_gabab, randset=False, dwell=30, drive=8000.0, width=5.0):
    b = build(seed, w_gabab)
    rm = b.region_manager; pl = rm.indices("place")
    M0 = B.wmat(b); x = np.arange(NPLACE); rs = np.random.default_rng(seed * 31337)
    for c in range(NPLACE):
        bump = np.exp(-0.5 * ((x - c) / width) ** 2)
        p = bump
        if randset:
            p = np.zeros(NPLACE); p[rs.permutation(NPLACE)[:int(round(bump.sum()))]] = 1.0
        for _ in range(dwell):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[pl] = drive * p
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
    M1 = B.wmat(b); dW = np.maximum(M1 - M0, 0.0); pk, wd = stats(M1 - M0)
    return pk, wd, float(np.mean([B.circ_resultant(r) for r in dW])), float(np.abs(M1 - M0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--gabab", type=float, nargs="+", default=[0.0, 60.0, 150.0, 400.0])
    ap.add_argument("--out", default="research/findings/raw/gap5_reader/gabab_plateau_termination.json")
    a = ap.parse_args()
    print("GABA_B PLATEAU TERMINATION — does slow K+ inhibition NARROW the learned field?")
    print("  PRE-REGISTERED: w=0 reproduces baseline (width ~16, place-spec ~0.59). SUCCESS = width FALLS")
    print("  toward the oracle's ~12 with place-specific circ HELD. dW~0 = plateau abolished => UNDEFINED.")
    print("  %-9s %-11s %-12s %-12s %-12s %-11s" % ("w_gabab", "peaks", "WIDTH /60", "circ(dW)", "randset", "mean|dW|"))
    rows = []
    for w in a.gabab:
        P, W, C, D, CR = [], [], [], [], []
        for s in a.seeds:
            pk, wd, c, d = run(s, w); P.append(pk); W.append(wd); C.append(c); D.append(d)
            _, _, cr, _ = run(s, w, randset=True); CR.append(cr)
        m = lambda v: float(np.mean(v))
        rows.append(dict(w=w, peaks=m(P), width=m(W), circ=m(C), randset=m(CR), dW=m(D),
                         place_specific=m(C) - m(CR)))
        flag = "  ⛔ PLATEAU ABOLISHED => UNDEFINED" if m(D) < 1.0 else ""
        print("  %-9.0f %-11.2f %-12.1f %-12.4f %-12.4f %-11.0f%s" % (w, m(P), m(W), m(C), m(CR), m(D), flag))
    base = rows[0]
    print("  " + "-" * 74)
    for r in rows[1:]:
        dw_ = base["width"] - r["width"]; dps = r["place_specific"] - base["place_specific"]
        verdict = ("✅ NARROWS, place-specificity held" if dw_ > 1.5 and dps > -0.05
                   else ("⚠️ narrows but LOSES place-specificity" if dw_ > 1.5 else "⛔ no narrowing"))
        print("  w=%-6.0f width %+.1f | place-specific %+.4f  => %s" % (r["w"], -dw_, dps, verdict))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
