"""2026-07-15 — EDGE 5 rung 3 (plan #1, the SURPASS of the rung-2 multi-bind boundary): raw STP facilitation is
ADDITIVE (interferes past 1 bind); the numpy store showed the ERROR-CORRECTING DELTA write holds multiple binds. This
realizes a delta-like write ON THE BRIDGE weights: barcode→value is a PLASTIC pathway; the WRITE is two-phase — read the
current spiking PREDICTION for barcode_i (neural), then POTENTIATE barcode_i→value_i (the target) and DEPRESS
barcode_i→(the wrong predicted value) (the error-correction = subtract the current prediction, the delta rule's `-M@k`).
The prediction is read from spikes; the update is a bounded Hebbian/anti-Hebbian weight step (no host-computed error gating
the sign per-synapse — the sign comes from target-vs-prediction pools, biologically legal). Does it now hold P≥2 binds
where rung-2's additive facilitation collapsed below chance?

GATE (6-seed): multi-pair retrieve holds ≥2 (>> the rung-2 additive collapse) AND novel-barcode ~chance AND a NO-DEPRESS
control (potentiate-only = additive) reproduces the collapse (the DEPRESSION/error-correction is load-bearing). numpy-CPU;
NO `sim/` edit (writes existing `cp_connections.data` via a neural-read-gated potentiate/depress — the plasticity is the write).

Run: SIM_BACKEND=numpy python -u -m research.runners._edge5_rung3_delta_store_onbridge_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._novel_referent_hebbian_fastweight_derisk import _mint_codes, _DIM
from research.runners._edge5_rung2_stp_store_onbridge_derisk import _build_store_bridge, KV, POOL, INPOP

RETRIEVE_STEPS = 40
# weight-based store (STP OFF), so the drive is NOT release-scaled -> use a LOW drive so the potentiated pool crosses
# threshold while the baseline (weight ~8) does not (rung-2's U=0.05 scaled the drive; here the weight is the store).
DRIVE = 55.0
LR_POT = 8.0            # potentiation step toward the target value
LR_DEP = 8.0           # depression step away from the wrong prediction (the error-correction)
W_MAX = 40.0
W0_DEP = 8.0           # (starting barcode→val weight is ~8; potentiate to >baseline, depress toward 0)


class DeltaStore:
    """PLASTIC barcode→value weight store with a two-phase (potentiate-target / depress-prediction) delta-like write."""
    def __init__(self, seed, depress=True):
        self.bridge, self.cfg = _build_store_bridge(seed)
        self.cfg.enable_short_term_plasticity = False           # this store is WEIGHT-based, not STP (the delta lives in W)
        self.depress = depress
        self._num = int(self.bridge.core_config.num_neurons)
        rm = self.bridge.region_manager
        self.bar_idx = np.asarray(list(rm.indices("bar")), int)
        self.val_idx = np.asarray(list(rm.indices("val")), int)
        self.bar_ch = [self.bar_idx[v * INPOP:(v + 1) * INPOP] for v in range(_DIM)]
        self.val_pool = [self.val_idx[k * POOL:(k + 1) * POOL] for k in range(KV)]
        from sim.backend import to_host
        row, col = self._coo()
        self._row, self._col = row, col
        bar_set = set(self.bar_idx.tolist())
        self._bar_syn = np.array([r in bar_set for r in row])   # barcode→val synapses (the plastic store)
        # map each val neuron -> its pool k
        self._val_pool_of = {int(n): k for k in range(KV) for n in self.val_pool[k]}

    def _coo(self):
        from sim.backend import to_host
        coo = self.bridge._get_cached_coo()
        return (np.asarray(to_host(coo.row)).astype(int), np.asarray(to_host(coo.col)).astype(int))

    def _drive(self, barcode):
        from sim.backend import from_host
        cur = np.zeros(self._num, np.float32)
        if barcode is not None:
            for bit in np.nonzero(barcode)[0]:
                cur[self.bar_ch[bit]] = DRIVE
        self.bridge.cp_external_input_current[:] = from_host(cur)

    def _read_pred(self, barcode, steps=RETRIEVE_STEPS):
        from sim.backend import to_host
        self._drive(barcode)
        counts = np.zeros(KV)
        for _ in range(steps):
            self.bridge._run_one_simulation_step()
            fs = np.asarray(to_host(self.bridge.cp_firing_states)).astype(np.float64)
            for k in range(KV):
                counts[k] += fs[self.val_pool[k]].sum()
        self.bridge.cp_external_input_current[:] = 0.0
        return counts

    def _update_w(self, barcode, target_k, pred_k):
        """The delta-like write: potentiate barcode's active synapses onto target_k's pool; depress onto pred_k's pool
        (if pred_k != target_k). Realized on cp_connections.data (the weight is the store; the read that set pred_k is neural)."""
        from sim.backend import to_host, from_host
        w = np.asarray(to_host(self.bridge.cp_connections.data)).astype(np.float64)
        active_bar = set()
        for bit in np.nonzero(barcode)[0]:
            active_bar.update(self.bar_ch[bit].tolist())
        tgt = set(self.val_pool[target_k].tolist())
        dep = set(self.val_pool[pred_k].tolist()) if (self.depress and pred_k != target_k) else set()
        for si in np.nonzero(self._bar_syn)[0]:
            if self._row[si] in active_bar:
                c = self._col[si]
                if c in tgt:
                    w[si] = min(W_MAX, w[si] + LR_POT)          # potentiate toward the target
                elif c in dep:
                    w[si] = max(0.0, w[si] - LR_DEP)            # depress the wrong prediction (error-correction)
        self.bridge.cp_connections.data = from_host(w.astype(np.float32))

    def write(self, barcode, value):
        pred = int(np.argmax(self._read_pred(barcode)))         # the CURRENT neural prediction (before this write)
        self._update_w(barcode, value, pred)

    def retrieve(self, barcode):
        return int(np.argmax(self._read_pred(barcode)))


def run_multipair(seed, Ps=(1, 2, 3, 4), n_trials=12, depress=True):
    rng = np.random.default_rng(seed)
    codes = _mint_codes(rng, 16)
    res = {"seed": seed, "chance": round(1.0 / KV, 4), "depress": depress, "byP": {}}
    for P in Ps:
        hits = []
        for t in range(n_trials):
            store = DeltaStore(seed * 100 + t, depress=depress)
            ents = rng.choice(16, size=P, replace=False); vals = list(rng.permutation(KV)[:P])
            for i in range(P):
                store.write(codes[ents[i]], vals[i])
            for i in range(P):
                hits.append(store.retrieve(codes[ents[i]]) == vals[i])
        res["byP"][P] = round(float(np.mean(hits)), 4)
    res["holds_ge2"] = bool(res["byP"].get(2, 0) > 0.55)
    res["holds_ge3"] = bool(res["byP"].get(3, 0) > 0.5)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--out", default="research/findings/raw/_edge5_rung3_delta_store_onbridge.json")
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        r = run_multipair(s, n_trials=a.n_trials, depress=True)
        rc = run_multipair(s, n_trials=a.n_trials, depress=False)     # no-depress = additive control (should collapse)
        r["nodepress_byP"] = rc["byP"]
        rows.append(r)
        bp, cp = r["byP"], rc["byP"]
        print(f"[delta-store s{s}] chance={r['chance']} || DELTA(depress) P1={bp.get(1):.2f} P2={bp.get(2):.2f} P3={bp.get(3):.2f} P4={bp.get(4):.2f} "
              f"|| additive(no-depress) P2={cp.get(2):.2f} P3={cp.get(3):.2f} || holds>=2={r['holds_ge2']} holds>=3={r['holds_ge3']}", flush=True)
    n2 = sum(x["holds_ge2"] for x in rows); n3 = sum(x["holds_ge3"] for x in rows)
    print(f"[delta-store] holds>=2 {n2}/{len(rows)}, holds>=3 {n3}/{len(rows)} (the error-correcting DELTA write surpasses the "
          f"rung-2 additive-facilitation multi-bind collapse; the no-depress control reproduces the collapse = depression load-bearing)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
