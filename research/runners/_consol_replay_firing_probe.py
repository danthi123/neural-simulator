"""Consolidation Option-3 diagnostic (2026-07-25): is the CA1 firing DURING the co-activation replay FACT-SPECIFIC
(the clean fire-under-tag pattern the oracle's 8.19 assumes) or FLOODED (flat)? The oracle over-predicts the real write
8x; this probe measures WHICH stage breaks. Instrument a replicated co-activation replay: for each fact i, drive tag_i +
its concept pools + slot_i, and ACCUMULATE the CA1 firing during the burst. Then measure, over the distinctive cores:
  replay-fire OWN/OTHER = (core_i firing during fact_i's replay) / mean_j!=i (core_i firing during fact_j's replay).

  own/other >> 1 (fact-specific)  -> the replay firing IS clean -> the flat WRITE is the STDP-timing failure -> a
      RATE-based Hebbian write should localize (the next Option-3 test).
  own/other ~= 1 (flooded)         -> the replay FLOODS CA1 -> the write can't localize because its input is flat ->
      confirms the replay-flooding wall (route to Option 2, the different substrate).

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_replay_firing_probe --seed 42
"""
import os, sys, json, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from research.runners.text_minimal_isolation import set_sleep_gates
from research.runners._consol_dg_overlap_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def _fire_under_tag(b, tag, ca1, steps=40):
    _try_tgate(b, "nmda_attractor", 0.0); set_sleep_gates(b)
    b.cp_external_input_current[:] = 0.0
    for _ in range(30):
        b._run_one_simulation_step()
    b.stimulate_tag(tag, drive_pA=1500.0, additive=False)
    acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]))
    for _ in range(steps):
        b._run_one_simulation_step()
        acc += to_host(b.cp_firing_states).astype(np.float64)
    try: b.clear_tag_drive(tag)
    except Exception: pass
    b.cp_external_input_current[:] = 0.0
    return acc[ca1]


def run(seed):
    b = build_substrate(seed, SimpleNamespace(**BASE))
    rm = b.region_manager
    ca1 = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    all_regions = {r.name for r in b.core_config.brain_regions}

    def _idx(nm):
        return np.asarray(sorted(rm.indices(nm)), dtype=np.int64) if nm in all_regions else None
    # distinctive cores from fire-under-tag (the oracle's pattern)
    ut = np.stack([_fire_under_tag(b, tag, ca1) for tag in tags])  # (N, n_ca1)
    core = ut > 0.25 * 40
    dcore = {}
    for i in range(N):
        others = set()
        for j in range(N):
            if j != i: others |= set(np.where(core[j])[0].tolist())
        dcore[i] = np.asarray([c for c in np.where(core[i])[0].tolist() if c not in others], dtype=np.int64)

    # pool + slot indices per fact
    pool_idx, slot_idx = {}, {}
    for i, (noun, adj) in enumerate(CONSOLIDATED_FACTS):
        ps = [_idx("noun_pool_%s" % noun.upper()), _idx("adjective_pool_%s" % adj.upper())]
        pool_idx[i] = [p for p in ps if p is not None]
        slot_idx[i] = _idx("comp_attr_%d" % i)

    # replicated co-activation replay, recording CA1 firing per fact
    set_sleep_gates(b)
    for g in ("ca1_to_concept_pool", "ca1_to_comp_attr", "concept_to_comp_attr", "cross_pool_concept"):
        _try_pgate(b, g, 1.0)
    _try_tgate(b, "nmda_attractor", 1.0)
    replay_fire = np.zeros((N, ca1.size))
    for i, tag in enumerate(tags):
        b.cp_external_input_current[:] = 0.0
        b.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        drv = cp.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=cp.float32)
        for a in pool_idx[i]:
            drv[cp.asarray(a)] = 1400.0
        if slot_idx[i] is not None:
            drv[cp.asarray(slot_idx[i])] = 1400.0
        acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]))
        for _ in range(30):
            b.cp_external_input_current[:] = drv   # sustain pool+slot alongside the tag drive
            b._run_one_simulation_step()
            acc += to_host(b.cp_firing_states).astype(np.float64)
        try: b.clear_tag_drive(tag)
        except Exception: pass
        b.cp_external_input_current[:] = 0.0
        replay_fire[i] = acc[ca1]

    # own/other of the REPLAY firing over the distinctive cores + (reference) the fire-under-tag
    def oo(mat):
        r = []
        for i in range(N):
            d = dcore[i]
            if d.size == 0:
                r.append(0.0); continue
            own = mat[i][d].sum()
            oth = np.mean([mat[j][d].sum() for j in range(N) if j != i])
            r.append(own / oth if oth > 1e-9 else 0.0)
        return [round(x, 3) for x in r]
    return dict(seed=seed, dcore_sizes={i: int(dcore[i].size) for i in range(N)},
                replay_fire_own_over_other=oo(replay_fire),
                fire_under_tag_own_over_other=oo(ut),
                replay_active_frac=[round(float((replay_fire[i] > 0).mean()), 3) for i in range(N)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed)
    Path(f"{args.out}/replay_firing_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    print(f"[seed {args.seed}] is the CA1 firing DURING replay fact-specific or flooded?")
    print(f"  distinctive core sizes: {r['dcore_sizes']}")
    print(f"  REPLAY firing own/other (over distinctive cores): {r['replay_fire_own_over_other']}")
    print(f"  fire-under-tag own/other (reference, the oracle's pattern): {r['fire_under_tag_own_over_other']}")
    print(f"  replay CA1 active_frac: {r['replay_active_frac']}")
    rm_ = float(np.mean([x for x in r['replay_fire_own_over_other'] if x > 0]))
    print(f"  VERDICT: {'REPLAY firing IS fact-specific (own/other %.2f) -> STDP-timing is the flat-write cause -> try a RATE-based write' % rm_ if rm_ > 1.5 else 'REPLAY firing is FLOODED (own/other %.2f ~1) -> the replay floods CA1 -> replay-flooding wall confirmed' % rm_}")
    print("REPLAY-FIRING-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
