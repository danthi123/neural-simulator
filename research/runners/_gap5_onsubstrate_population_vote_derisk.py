"""ON-SUBSTRATE POPULATION order vote: K reader cells, K-1 pairwise coincidence detectors, votes summed.

Extends `_gap5_onsubstrate_order_detector_derisk.py` (one pair, fwd/rev 3.286, 6/6 seeds) to a POPULATION, which
is what converts a per-pair ratio into single-trial discriminability. Off-substrate the envelope was measured at
16 cells x 83 ms = 1.000 single-trial (8 x 42 ms = 0.889, chance 0.500); this asks whether that holds in spikes.

WIRING per adjacent pair (k, k+1), at the PINNED operating point (n=50/stage, w_relay=300, 2 hops = 11.50 ms):
    cell_k   -> [relay a_k -> relay b_k] -> DET_k     (delayed ~11.5 ms)
    cell_k+1 ----------------------------> DET_k      (direct)
w_det=10 keeps DET SUBTHRESHOLD to either input alone -- the defining coincidence property, ASSERTED below,
because a detector suprathreshold to one input reads order BACKWARDS via refractory collision.

Forward sweep => each cell_k leads cell_k+1 by ~the pair lag => every DET_k sees coincidence => high total vote.
Reverse => no coincidence => low vote. Controls: SIMULTANEOUS (no order), LESION (relays bypassed => order-blind).
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
from sim.backend import to_host  # cupy arrays refuse np.asarray(); to_host works on BOTH backends

N_STAGE, W_RELAY, W_DET = 50, 300.0, 10.0


def build(seed, K, lesion=False, n_stage=N_STAGE, w_det=W_DET):
    R, P = [], []
    for k in range(K):
        R.append(BrainRegion(name="c%d" % k, n_neurons=n_stage, exc_fraction=1.0, internal_density=0.0))
    for k in range(K - 1):
        R.append(BrainRegion(name="d%d" % k, n_neurons=n_stage, exc_fraction=1.0, internal_density=0.0))
        if lesion:
            # LESION: cell_k goes DIRECT to its detector -> the delay is gone -> must be order-blind
            P.append(RegionPathway(from_region="c%d" % k, to_region="d%d" % k, density=1.0,
                                   weight_mean=w_det, weight_jitter=0.0, plastic=False))
        else:
            R.append(BrainRegion(name="a%d" % k, n_neurons=n_stage, exc_fraction=1.0, internal_density=0.0))
            R.append(BrainRegion(name="b%d" % k, n_neurons=n_stage, exc_fraction=1.0, internal_density=0.0))
            P += [RegionPathway(from_region="c%d" % k, to_region="a%d" % k, density=1.0, weight_mean=W_RELAY, weight_jitter=0.0, plastic=False),
                  RegionPathway(from_region="a%d" % k, to_region="b%d" % k, density=1.0, weight_mean=W_RELAY, weight_jitter=0.0, plastic=False),
                  RegionPathway(from_region="b%d" % k, to_region="d%d" % k, density=1.0, weight_mean=w_det, weight_jitter=0.0, plastic=False)]
        P.append(RegionPathway(from_region="c%d" % (k + 1), to_region="d%d" % k, density=1.0,
                               weight_mean=w_det, weight_jitter=0.0, plastic=False))
    cfg = CoreSimConfig(seed=seed, dt_ms=1.0, enable_brain_region_framework=True, brain_regions=R,
                        region_pathways=P, enable_hebbian_learning=False, enable_stdp=False,
                        enable_homeostasis=False, enable_structural_plasticity=False, enable_ou_process=False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def vote(seed, K, order, lag=12, lesion=False, drive=8000.0, jitter=0.0, w_det=W_DET, t0=3):
    """order: +1 forward (c0 first), -1 reverse (cK-1 first), 0 simultaneous. Returns total detector spikes."""
    b = build(seed, K, lesion, w_det=w_det)
    rm = b.region_manager
    rng = np.random.default_rng(seed * 7919 + order + int(jitter * 1000))
    if order == 0:
        times = {k: t0 for k in range(K)}
    else:
        seq = range(K) if order > 0 else reversed(range(K))
        times = {}
        for i, k in enumerate(seq):
            j = int(round(rng.normal(0, jitter))) if jitter > 0 else 0
            times[k] = max(0, t0 + i * lag + j)
    T = max(times.values()) + 40
    dets = [rm.indices("d%d" % k) for k in range(K - 1)]
    total = 0
    for step in range(T):
        b.cp_external_input_current[:] = 0.0
        for k, tk in times.items():
            if tk <= step <= tk + 1:
                b.cp_external_input_current[rm.indices("c%d" % k)] = drive
        b._run_one_simulation_step()
        for d in dets:
            total += int(to_host(b.cp_firing_states[d]).sum())
    return total


def single_input_check(seed, K, w_det=W_DET, drive=8000.0, T=60):
    """ASSERT the coincidence property: driving ONE cell must leave the detectors ~silent."""
    b = build(seed, K, False, w_det=w_det); rm = b.region_manager
    dets = [rm.indices("d%d" % k) for k in range(K - 1)]
    n = 0
    for step in range(T):
        b.cp_external_input_current[:] = 0.0
        if 3 <= step <= 4:
            b.cp_external_input_current[rm.indices("c0")] = drive
        b._run_one_simulation_step()
        for d in dets:
            n += int(to_host(b.cp_firing_states[d]).sum())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--jitter", type=float, default=2.0)
    ap.add_argument("--out", default="research/findings/raw/gap5_reader/onsub_population_vote.json")
    a = ap.parse_args()
    print("ON-SUBSTRATE POPULATION ORDER VOTE  K=%d cells, %d detectors, n=%d/stage, w_relay=%.0f, w_det=%.0f"
          % (a.K, a.K - 1, N_STAGE, W_RELAY, W_DET))
    print("=" * 78)
    sc = float(np.mean([single_input_check(s, a.K) for s in a.seeds]))
    print("STEP 0 coincidence property: single-cell drive -> detector spikes = %.1f  %s"
          % (sc, "OK (subthreshold)" if sc < 2 else "⛔ SUPRATHRESHOLD -> order reads BACKWARDS; lower w_det"))
    if sc >= 2:
        print("  ABORT: the defining property fails, so any ratio below would be uninterpretable.")
        return 1
    rows = []
    for s in a.seeds:
        F = [vote(s + 1000 * t, a.K, +1, jitter=a.jitter) for t in range(a.trials)]
        R = [vote(s + 1000 * t, a.K, -1, jitter=a.jitter) for t in range(a.trials)]
        S = [vote(s + 1000 * t, a.K, 0) for t in range(a.trials)]
        LF = [vote(s + 1000 * t, a.K, +1, lesion=True, jitter=a.jitter) for t in range(a.trials)]
        LR = [vote(s + 1000 * t, a.K, -1, lesion=True, jitter=a.jitter) for t in range(a.trials)]
        acc = float(np.mean([f > r for f, r in zip(F, R)]))          # paired single-trial accuracy
        lacc = float(np.mean([f > r for f, r in zip(LF, LR)]))
        rows.append(dict(seed=s, fwd=float(np.mean(F)), rev=float(np.mean(R)), sim=float(np.mean(S)),
                         ratio=float((np.mean(F) + 1e-9) / (np.mean(R) + 1e-9)), acc=acc,
                         lesion_ratio=float((np.mean(LF) + 1e-9) / (np.mean(LR) + 1e-9)), lesion_acc=lacc))
        print("  seed %-5d fwd=%-7.1f rev=%-7.1f sim=%-7.1f ratio=%-6.2f acc=%-6.3f | LESION ratio=%-6.2f acc=%.3f"
              % (s, rows[-1]["fwd"], rows[-1]["rev"], rows[-1]["sim"], rows[-1]["ratio"], acc,
                 rows[-1]["lesion_ratio"], lacc))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print("-" * 78)
    print("  MEAN ratio=%.3f  single-trial acc=%.3f (chance 0.500)  |  LESION ratio=%.3f acc=%.3f (must be ~1 / ~0.5)"
          % (m("ratio"), m("acc"), m("lesion_ratio"), m("lesion_acc")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(K=a.K, n_stage=N_STAGE, w_relay=W_RELAY, w_det=W_DET, jitter=a.jitter,
                   trials=a.trials, single_input_det_spikes=sc, rows=rows), open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
