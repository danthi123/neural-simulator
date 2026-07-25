"""Consolidation Family-C GATE (2026-07-25) — the cheap numpy ORACLE that decides whether the months-scale multi-branch
dendritic build is warranted, BEFORE building it. On the REAL measured CA1 codes (build+encode+fire-under-tag), model K
per-branch apical plateaus per slot + a plateau-gated ca1->slot write, and sweep the ONE variable = branch assignment:
  - ORACLE-clustered: fact_i's distinct CORE cells' synapses to slot_i are clustered on ONE branch (branch 0).
  - RANDOM: synapses assigned to branches at random.
Per-branch gate: branch b of slot_i plateaus (under fact_i's replay, only slot_i driven) iff its pooled ca1 inputs have
>= k_thresh coincident firers; the write potentiates ca1->slot_i ONLY for synapses whose branch plateaued.

  GO  : oracle own/other >= 2.5 (own-is-max 3/3) AND random ~= 1.0  -> per-branch gating WORKS in principle -> the
        months multi-branch + clustering build is warranted (C2, the emergent clustering, is then the real open problem).
  KILL: even ORACLE-clustered own/other < 2.5 -> per-branch gating CANNOT localize on this dense-halo code even with
        PERFECT clustering -> Family C is dead on this substrate -> the capability needs a different substrate entirely.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_multibranch_oracle --seed 42
"""
import os, sys, json, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, CONSOLIDATED_FACTS, _try_tgate)
from research.runners._consol_dg_overlap_probe import BASE
from research.runners.text_minimal_isolation import set_sleep_gates
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def measure_fire(seed):
    """Real CA1 firing per fact under its tag (the actual code the write must localize on)."""
    b = build_substrate(seed, SimpleNamespace(**BASE))
    rm = b.region_manager
    ca1 = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    fire = np.zeros((N, ca1.size), dtype=np.float64)
    for i, tag in enumerate(tags):
        _try_tgate(b, "nmda_attractor", 0.0); set_sleep_gates(b)
        b.cp_external_input_current[:] = 0.0
        for _ in range(30):
            b._run_one_simulation_step()
        b.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]))
        for _ in range(40):
            b._run_one_simulation_step()
            acc += to_host(b.cp_firing_states).astype(np.float64)
        try: b.clear_tag_drive(tag)
        except Exception: pass
        b.cp_external_input_current[:] = 0.0
        fire[i] = acc[ca1]
    return fire  # (N facts, n_ca1) firing counts 0..40


def oracle(seed, K=6, density=0.25, k_thresh=4, eta=1.0, steps=40, rng_seed=0):
    fire = measure_fire(seed)
    n_ca1 = fire.shape[1]
    rng = np.random.default_rng(seed * 100 + rng_seed)
    active = fire > 0                                  # who fires per fact (dense halo)
    core = fire > 0.25 * steps                         # the distinct strong-firing core per fact
    # ca1 -> slot_i connectivity (random subset per slot)
    conn = {i: np.where(rng.random(n_ca1) < density)[0] for i in range(N)}   # ca1 indices synapsing onto slot_i

    def run_write(mode):
        W = np.zeros((N, n_ca1))                        # W[slot_i, ca1] potentiated weight
        for i in range(N):                              # fact_i replay: only slot_i driven
            pre = conn[i]                               # ca1 cells synapsing onto slot_i
            # assign each pre-synapse to a branch
            if mode == "oracle":                        # cluster fact_i's core onto branch 0
                br = np.empty(pre.size, dtype=np.int64)
                is_core = core[i][pre]
                br[is_core] = 0
                br[~is_core] = rng.integers(1, K, size=int((~is_core).sum())) if K > 1 else 0
            else:                                       # random branch assignment
                br = rng.integers(0, K, size=pre.size)
            # which branches plateau (under fact_i's replay): coincident core... use ANY-firing under fact_i (the write
            # sees firing, not core-labels) -- the branch plateaus on the COUNT of fact_i-firing pooled inputs
            fires_i = active[i][pre]
            plateau = np.zeros(K, dtype=bool)
            for bnum in range(K):
                sel = (br == bnum)
                plateau[bnum] = int(np.logical_and(sel, fires_i).sum()) >= k_thresh
            # write: potentiate ca1->slot_i for firing synapses whose branch plateaued (eligibility ~ firing rate)
            for idx, ca1i in enumerate(pre):
                if plateau[br[idx]] and fires_i[idx]:
                    W[i, ca1i] += eta * (fire[i][ca1i] / steps)
        # read: engram_i = distinctive core of fact_i; own/other of ca1_engram_i -> slot_j
        oo = []
        for i in range(N):
            others = set()
            for j in range(N):
                if j != i: others |= set(np.where(core[j])[0].tolist())
            dist = np.asarray([c for c in np.where(core[i])[0].tolist() if c not in others], dtype=np.int64)
            if dist.size == 0:
                oo.append(0.0); continue
            own = W[i, dist].sum()
            oth = np.mean([W[j, dist].sum() for j in range(N) if j != i])
            oo.append(own / oth if oth > 1e-9 else (999.0 if own > 1e-9 else 0.0))
        return [round(min(x, 999.0), 3) for x in oo], sum(1 for x in oo if x == max(oo))
    o_oo, _ = run_write("oracle")
    r_oo, _ = run_write("random")
    return dict(seed=seed, K=K, k_thresh=k_thresh, oracle_own_over_other=o_oo, random_own_over_other=r_oo,
                oracle_mean=round(float(np.mean([x for x in o_oo if x < 999])), 3),
                random_mean=round(float(np.mean([x for x in r_oo if x < 999])), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--k-thresh", type=int, default=4)
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = oracle(args.seed, K=args.K, k_thresh=args.k_thresh)
    Path(f"{args.out}/mbranch_oracle_seed{args.seed}_K{args.K}_kt{args.k_thresh}.json").write_text(json.dumps(r, indent=2))
    print(f"[seed {args.seed} K={args.K} k_thresh={args.k_thresh}] multi-branch oracle:")
    print(f"  ORACLE-clustered own/other = {r['oracle_own_over_other']} (mean {r['oracle_mean']})")
    print(f"  RANDOM          own/other = {r['random_own_over_other']} (mean {r['random_mean']})")
    go = r['oracle_mean'] >= 2.5 and r['random_mean'] < 1.5
    print(f"  VERDICT: {'GO -- per-branch gating WORKS in principle (oracle>=2.5, random~1) -> months build warranted' if go else 'KILL/PARTIAL -- oracle own/other %.2f (need >=2.5) -> per-branch gating %s' % (r['oracle_mean'], 'cannot localize even with perfect clustering' if r['oracle_mean']<2.5 else 'works; check random baseline')}")
    print("MBRANCH-ORACLE DONE", flush=True)


if __name__ == "__main__":
    main()
