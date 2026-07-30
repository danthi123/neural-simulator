"""THE INTEGRATION: does BTSP-LEARNED place tuning support the spiking ORDER read?

The gap#5 arc has two halves demonstrated separately:
  * READING replay direction  -- 0.969 single-trial on GPU (pairwise relay+coincidence), but on HAND-SET tuning.
  * ACQUIRING spatial tuning  -- BTSP + between-reader soft-WTA, +0.1281 circ on 6 seeds, all controls passing,
                                 but measured in isolation and only ~19% of the sigma=5 oracle ceiling.
This joins them: learn the tuning with BTSP, then read ORDER through relay+coincidence detectors wired by the
LEARNED preferred positions. If it works, the host Bayesian decoder's shortcut is replaced end-to-end with
learned representations feeding a spiking read.

PHASE 1 (learn): place -> read, BTSP at the non-saturating optimum (lr=0.002, dwell=30, 1 lap, sat_frac 0.000),
                 with the FS soft-WTA between readers. Learned weights are read back.
PHASE 2 (read):  a fresh bridge with those learned weights INSTALLED (set_pathway_weights), plus 2-hop relay +
                 coincidence detectors between readers ADJACENT IN LEARNED PREFERENCE ORDER. Forward vs reverse
                 sweep -> order discrimination.

HONEST SCOPE, stated up front: the TUNING is learned (that was the last hand-set element), but PAIRING readers by
learned preference order is still a HOST step -- in biology that adjacency would be developmental/topographic.
So this closes the tuning shortcut, NOT the wiring one.

CONTROLS: lr=0 tuning (untrained weights installed -> order read must degrade); reverse sweep; relay-lesion
(order-blind); and a SCRAMBLED-PAIRING arm (pair readers at random instead of by learned order -> must degrade).
"""
import argparse, json, os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
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
NS, WREL, WDET = 50, 300.0, 10.0


def learn_tuning_btsp(seed, lr, w_max=2500.0, w_inh=150.0, laps=1, dwell=30):
    """PHASE 1 -- returns (learned place->read matrix, learned preferred positions, sat_frac)."""
    M0, M1, nread, nplace, apmax = B.run(seed, w_inh, True, lr, w_max, laps=laps, dwell=dwell)
    sat = float((M1 >= w_max * 0.98).mean())
    return M1, np.argmax(M1, axis=1), sat, nread, apmax


def build_reader(seed, K, lesion=False):
    """PHASE 2 -- readers (driven by installed learned weights) + 2-hop relays + coincidence detectors."""
    R = [BrainRegion(name="place", n_neurons=NPLACE, exc_fraction=1.0, internal_density=0.0)]
    for k in range(K):
        R.append(BrainRegion(name="c%d" % k, n_neurons=NS, exc_fraction=1.0, internal_density=0.0))
    P = []
    for k in range(K):
        # place -> reader k : weights are OVERWRITTEN post-build with the LEARNED tuning of reader k
        P.append(RegionPathway(from_region="place", to_region="c%d" % k, density=1.0, weight_mean=1.0,
                               weight_jitter=0.0, plastic=False))
    for k in range(K - 1):
        R.append(BrainRegion(name="d%d" % k, n_neurons=NS, exc_fraction=1.0, internal_density=0.0))
        if lesion:
            P.append(RegionPathway(from_region="c%d" % k, to_region="d%d" % k, density=1.0,
                                   weight_mean=WDET, weight_jitter=0.0, plastic=False))
        else:
            R.append(BrainRegion(name="a%d" % k, n_neurons=NS, exc_fraction=1.0, internal_density=0.0))
            R.append(BrainRegion(name="b%d" % k, n_neurons=NS, exc_fraction=1.0, internal_density=0.0))
            P += [RegionPathway(from_region="c%d" % k, to_region="a%d" % k, density=1.0, weight_mean=WREL, weight_jitter=0.0, plastic=False),
                  RegionPathway(from_region="a%d" % k, to_region="b%d" % k, density=1.0, weight_mean=WREL, weight_jitter=0.0, plastic=False),
                  RegionPathway(from_region="b%d" % k, to_region="d%d" % k, density=1.0, weight_mean=WDET, weight_jitter=0.0, plastic=False)]
        P.append(RegionPathway(from_region="c%d" % (k + 1), to_region="d%d" % k, density=1.0,
                               weight_mean=WDET, weight_jitter=0.0, plastic=False))
    cfg = CoreSimConfig(seed=seed, dt_ms=1.0, enable_brain_region_framework=True, brain_regions=R,
                        region_pathways=P, enable_hebbian_learning=False, enable_stdp=False,
                        enable_homeostasis=False, enable_structural_plasticity=False, enable_ou_process=False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def install_learned(b, M, order, K, gain):
    """Overwrite place->c_k with the LEARNED tuning of the k-th reader in `order` (learned-preference order)."""
    rm = b.region_manager
    pl = np.asarray(rm.indices("place"))
    n_inst = 0
    for k in range(K):
        j = int(order[k])
        tgt = np.asarray(rm.indices("c%d" % k))
        w = M[j] * gain                                   # the learned place-tuning row, scaled to drive spikes
        pre = np.repeat(pl, len(tgt)); post = np.tile(tgt, len(pl))
        wts = np.repeat(w, len(tgt))
        n_inst += b.set_pathway_weights("place_to_c%d" % k, pre, post, wts, add_missing=True)
    return n_inst


def order_vote(b, K, direction, lag=12, drive=3000.0, width=5.0, dwell=6):
    """Drive a place sweep; sum detector spikes. Forward should exceed reverse."""
    rm = b.region_manager; pl = rm.indices("place")
    dets = [rm.indices("d%d" % k) for k in range(K - 1)]
    x = np.arange(NPLACE); tot = 0
    seq = range(NPLACE) if direction > 0 else reversed(range(NPLACE))
    for c in seq:
        p = np.exp(-0.5 * ((x - c) / width) ** 2)
        for _ in range(dwell):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[pl] = drive * p
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            for d in dets: tot += int(to_host(b.cp_firing_states[d]).sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--out", default="research/findings/raw/gap5_reader/learned_tuning_order_read.json")
    a = ap.parse_args()
    print("INTEGRATION: BTSP-LEARNED tuning -> spiking ORDER read   (K=%d readers)" % a.K)
    print("=" * 96)
    rows = []
    for s in a.seeds:
        # PHASE 1 -- learn (and its lr=0 control)
        M_tr, pref_tr, sat_tr, nread_tr, ap_tr = learn_tuning_btsp(s, a.lr)
        M_un, pref_un, sat_un, _, _ = learn_tuning_btsp(s, 0.0)
        if sat_tr > 0.2:
            print("  seed %d ⛔ SATURATED (sat_frac %.3f) -> UNDEFINED, skipping" % (s, sat_tr)); continue
        # readers ordered by LEARNED preferred position; take K spread across the range
        ordr = np.argsort(pref_tr)[np.linspace(0, NREAD - 1, a.K).astype(int)]
        ordu = np.argsort(pref_un)[np.linspace(0, NREAD - 1, a.K).astype(int)]
        rng = np.random.default_rng(s * 13)
        ordscr = rng.permutation(NREAD)[:a.K]                      # SCRAMBLED pairing control
        out = {"seed": s, "sat_frac": sat_tr, "prefs_learned": pref_tr.tolist()}
        for tag, M, od, les in (("LEARNED", M_tr, ordr, False),
                                ("LEARNED_lesion", M_tr, ordr, True),
                                ("UNTRAINED", M_un, ordu, False),
                                ("SCRAMBLED_pairing", M_tr, ordscr, False)):
            b = build_reader(s, a.K, lesion=les)
            install_learned(b, M, od, a.K, a.gain)
            f = order_vote(b, a.K, +1)
            b2 = build_reader(s, a.K, lesion=les); install_learned(b2, M, od, a.K, a.gain)
            r = order_vote(b2, a.K, -1)
            out[tag] = dict(fwd=f, rev=r, ratio=float((f + 1e-9) / (r + 1e-9)))
            print("  seed %-5d %-19s fwd=%-7d rev=%-7d ratio=%.3f%s" % (
                s, tag, f, r, out[tag]["ratio"], "   ⛔ NO DETECTOR SPIKES" if (f + r) == 0 else ""))
        rows.append(out)
    if not rows:
        print("  no usable seeds -> UNDEFINED"); return 1
    print("-" * 96)
    for tag in ("LEARNED", "LEARNED_lesion", "UNTRAINED", "SCRAMBLED_pairing"):
        v = [r[tag]["ratio"] for r in rows if tag in r]
        print("  %-19s mean ratio = %.3f   (per-seed %s)" % (tag, float(np.mean(v)), [round(x, 2) for x in v]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(K=a.K, lr=a.lr, gain=a.gain, rows=rows), open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
