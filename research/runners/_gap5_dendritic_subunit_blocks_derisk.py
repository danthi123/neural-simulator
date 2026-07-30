"""SHARED DEPENDENCY: does assigning place inputs to SEPARATE DENDRITIC SUBUNITS by contiguous block
de-fragment the learned place fields? (gap#5's named residual, and the substrate the consolidation
successor's relational-binding/schema separation would rest on.)

WHY THIS SHAPE (catalog G.02 + this session's measurements):
  * G.02 "Active dendrites" describes the target nonlinearity as "cluster of inputs on ONE BRANCH >> scattered
    inputs on MANY branches", and lists Sim status "missing ... would require multi-compartment model".
  * That entry is PARTLY STALE: `enable_coincidence_detection` already gives each postsynaptic neuron a dendritic
    SUBUNIT **per tagged pathway** (config.py:159), and `enable_two_compartment_dap` supplies the apical
    compartment. What is absent is MULTIPLE subunits per CELL -- one per pathway, not many per neuron.
  * ⇒ So the mechanism is reachable by SPLITTING the input across pathways: give each contiguous place-index
    BLOCK its own pathway (hence its own subunit). Neighbouring place inputs then share a local plateau while
    scattered ones do not. NO sim/ edit, NO multi-compartment model.

THE DEFECT IT TARGETS: at the validated operating point the learned fields are narrow but GAPPY -- peaks/cell
3.19 against an ideal 1, width 16/60 against a sigma=5 oracle's ~12. Competition narrows; nothing enforces
CONTIGUITY. Subunit-per-block is the structural constraint that should.

METRICS (all four, because three metrics in this arc turned out blind): peaks/cell (the target), WIDTH at fixed
comparison, circ(dW), and the RANDSET no-place null -- circ alone rewards concentration regardless of place.
PRE-REGISTERED EXPECTATION, stated before the run: n_blocks=1 reproduces the baseline (peaks ~3.2, circ ~0.65);
if subunit-per-block works, peaks FALLS toward 1 while width and the randset null stay put. If peaks does not
move, the mechanism is refuted for this factoring -- NOT the capability.
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
from sim.backend import to_host
import importlib
B = importlib.import_module("research.runners._gap5_btsp_place_field_derisk")
NPLACE, NREAD = B.NPLACE, B.NREAD


def build_blocks(seed, n_blocks, w_max=2500.0, lr=0.002, w0=600.0):
    """Split the place population into n_blocks CONTIGUOUS sub-regions, each with its OWN
    coincidence_detector pathway to read => its own dendritic subunit per postsynaptic cell."""
    per = NPLACE // n_blocks
    R = [BrainRegion(name="p%d" % b, n_neurons=per, exc_fraction=1.0, internal_density=0.0)
         for b in range(n_blocks)]
    R.append(BrainRegion(name="read", n_neurons=NREAD, exc_fraction=1.0, internal_density=0.0))
    P = [RegionPathway(from_region="p%d" % b, to_region="read", density=1.0, weight_mean=w0,
                       weight_jitter=0.3, plastic=True, coincidence_detector=True)
         for b in range(n_blocks)]
    cfg = CoreSimConfig(seed=seed, dt_ms=1.0, enable_brain_region_framework=True, brain_regions=R,
                        region_pathways=P, enable_hebbian_learning=False, enable_stdp=False,
                        enable_homeostasis=False, enable_structural_plasticity=False,
                        enable_ou_process=False, enable_btsp=True, btsp_w_max=w_max, btsp_w_min=0.0,
                        btsp_learning_rate=lr, btsp_hetero_dep=0.2, btsp_elig_exponent=4.0,
                        btsp_elig_tau_ms=1000.0, enable_two_compartment_dap=True,
                        enable_coincidence_detection=True, coincidence_k_threshold=4.0)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, per


def wmat_blocks(b, n_blocks, per):
    """Reassemble the (NREAD x NPLACE) profile across the block sub-regions, preserving place order."""
    rm = b.region_manager; rd = np.asarray(rm.indices("read"))
    C = b.cp_connections; ip = to_host(C.indptr); idx = to_host(C.indices); dat = to_host(C.data)
    rs = {int(r): i for i, r in enumerate(rd)}
    M = np.zeros((NREAD, NPLACE))
    for bl in range(n_blocks):
        pl = np.asarray(rm.indices("p%d" % bl))
        for j, pre in enumerate(pl):
            pre = int(pre)
            for k in range(int(ip[pre]), int(ip[pre + 1])):
                po = int(idx[k])
                if po in rs:
                    M[rs[po], bl * per + j] += float(dat[k])
    return M


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


def run(seed, n_blocks, dwell=30, drive=8000.0, width=5.0, randset=False):
    b, per = build_blocks(seed, n_blocks)
    rm = b.region_manager
    pls = [np.asarray(rm.indices("p%d" % bl)) for bl in range(n_blocks)]
    M0 = wmat_blocks(b, n_blocks, per); x = np.arange(NPLACE)
    rs = np.random.default_rng(seed * 31337)
    for c in range(NPLACE):
        bump = np.exp(-0.5 * ((x - c) / width) ** 2)
        if randset:
            p = np.zeros(NPLACE); p[rs.permutation(NPLACE)[:int(round(bump.sum()))]] = 1.0
        else:
            p = bump
        for _ in range(dwell):
            b.cp_external_input_current[:] = 0.0
            for bl in range(n_blocks):
                b.cp_external_input_current[pls[bl]] = drive * p[bl * per:(bl + 1) * per]
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
    M1 = wmat_blocks(b, n_blocks, per)
    dW = np.maximum(M1 - M0, 0.0); pk, wd = stats(M1 - M0)
    return pk, wd, float(np.mean([B.circ_resultant(r) for r in dW])), float((M1 >= 2500.0 * 0.98).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--out", default="research/findings/raw/gap5_reader/dendritic_subunit_blocks.json")
    a = ap.parse_args()
    print("DENDRITIC SUBUNIT-PER-BLOCK: does contiguity-by-construction de-fragment the fields?")
    print("  PRE-REGISTERED: n_blocks=1 reproduces baseline (peaks ~3.2, circ ~0.65). Success = peaks FALLS")
    print("  toward 1 with width and the randset null unmoved. No movement => refuted for THIS factoring.")
    print("  %-10s %-12s %-12s %-13s %-12s %-9s" % ("n_blocks", "peaks/cell", "WIDTH /60", "circ(dW)", "randset", "sat"))
    rows = []
    for nb in a.blocks:
        if NPLACE % nb:
            print("  skip n_blocks=%d (does not divide %d)" % (nb, NPLACE)); continue
        P, W, C, S, CR = [], [], [], [], []
        for s in a.seeds:
            pk, wd, c, sat = run(s, nb)
            _, _, cr, _ = run(s, nb, randset=True)
            P.append(pk); W.append(wd); C.append(c); S.append(sat); CR.append(cr)
        m = lambda v: float(np.mean(v))
        rows.append(dict(n_blocks=nb, peaks=m(P), width=m(W), circ=m(C), randset=m(CR), sat=m(S)))
        print("  %-10d %-12.2f %-12.1f %-13.4f %-12.4f %-9.3f%s" % (
            nb, m(P), m(W), m(C), m(CR), m(S), "  ⛔ SATURATED" if m(S) > 0.2 else ""))
    if rows:
        base = rows[0]
        print("  " + "-" * 74)
        for r in rows[1:]:
            d = base["peaks"] - r["peaks"]
            print("  n_blocks=%d: peaks %+.2f vs 1-block | width %+.1f | place-specific circ %+.4f  => %s" % (
                r["n_blocks"], -d, r["width"] - base["width"], (r["circ"] - r["randset"]) - (base["circ"] - base["randset"]),
                "✅ DE-FRAGMENTS" if d > 0.8 else "⛔ no movement"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
