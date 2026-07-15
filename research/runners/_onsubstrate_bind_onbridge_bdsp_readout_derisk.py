"""2026-07-15 — RUNG 3 (fully-spiking culmination of the systematicity ladder, plan #3): the read-out over the FIXED spiking
coincidence bind (RUNG 1) is learned ON THE BRIDGE by the COMMITTED `enable_bdsp` rule (apical credit k*(Y@delta), fixed-random
Y = NO weight transport), so the WHOLE systematicity path — bind AND read-out — runs on spikes with biological credit.

Composition: RUNG-1's fixed spiking bind emits bound rates B (n, 2D) per (cat,qt). Those bound rates are the GRADED INPUT DRIVE
into a spiking read-out pool on a real SimulationBridge; the input->pool projection W_in is learned by the committed BDSP rule
(exactly the validated `_reslm_onbridge_learn_win` machinery, reused: input-region graded drive -> frozen pool -> apical credit
moves W_in) so the pool's spike-count decodes intent. Does the ON-BRIDGE-LEARNED spiking read-out over the fixed spiking bind
STILL extrapolate to held-out (cat,qt) combinations (>> a from-scratch classifier on raw codes)?

Reuse-by-import: RUNG-2's `bound_rates` (the fixed spiking bind), learn-win's `_build_bridge`/`_snapshot_state`/`_restore_state`
(the committed on-bridge BDSP machinery). NO `sim/` edit. numpy = smoke; SIM_BACKEND=cupy for GPU.

Run: SIM_BACKEND=numpy python -u -m research.runners._onsubstrate_bind_onbridge_bdsp_readout_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._onsubstrate_bind_learned_readout_derisk import bound_rates
from research.runners._fixedbind_systematicity_derisk import _dataset, N_INTENT  # noqa (N_INTENT for chance)
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T
from research.runners._reslm_onbridge_learn_win_derisk import _build_bridge, _snapshot_state, _restore_state, _T_STEP


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class OnBridgeBindReadout:
    """A spiking INPUT region (one graded channel per bound-rate dim) --[plastic input->pool W_in, learned by the committed
    BDSP rule]--> a FIXED spiking read-out pool on ONE SimulationBridge. The read feature = the pool per-neuron spike-count."""

    def __init__(self, n_feat, n_classes, seed, n_pool=120, in_pop=2, in_hi=650.0, res_bias=55.0, k_apical=150.0,
                 bdsp_lr=0.02, bdsp_p0=0.30, bdsp_beta=1.0, w_min=0.0, w_max=160.0, soma_g=120.0,
                 fwd_wmean=32.0, fwd_wjit=6.0, fwd_density=1.0, present=2):
        self.n_feat = int(n_feat); self.n = int(n_pool); self.in_pop = int(in_pop)
        self.in_hi = float(in_hi); self.res_bias = float(res_bias); self.k_apical = float(k_apical)
        self._eta = float(bdsp_lr); self.n_classes = int(n_classes); self.present = int(present)
        self.bridge, self.cfg = _build_bridge(seed, n_pool, n_feat, in_pop, fwd_wmean, fwd_wjit, fwd_density, soma_g,
                                              bdsp_lr, bdsp_p0, bdsp_beta, w_min, w_max)
        self._num = int(self.bridge.core_config.num_neurons)
        rm = self.bridge.region_manager
        self.in_idx = np.asarray(list(rm.indices("input")), dtype=int)
        self.res_idx = np.asarray(list(rm.indices("reservoir")), dtype=int)
        self.chan_idx = [self.in_idx[v * in_pop:(v + 1) * in_pop] for v in range(n_feat)]
        self.Y = np.random.RandomState(seed + 9973).normal(0.0, 1.0, (self.n, self.n_classes))  # fixed-random; no transport
        self._Y0 = self.Y.copy()
        self._snap = _snapshot_state(self.bridge)

    def _set_apical(self, vec):
        from sim.backend import from_host
        ap = np.zeros(self._num, np.float32)
        if vec is not None:
            ap[self.res_idx] = np.asarray(vec, np.float32)
        self.bridge.cp_bdsp_apical_drive = from_host(ap)

    def _drive(self, feat):
        cur = np.zeros(self._num, np.float32)
        for v in range(self.n_feat):
            cur[self.chan_idx[v]] = self.in_hi * float(feat[v])     # GRADED per-channel drive ∝ bound rate
        cur[self.res_idx] += self.res_bias
        return cur

    def forward(self, feat, learn, apical_vec):
        from sim.backend import from_host, to_host
        b = self.bridge
        _restore_state(b, self._snap)
        self.cfg.bdsp_learning_rate = (self._eta if learn else 0.0)
        self._set_apical(apical_vec if learn else None)
        drive = from_host(self._drive(feat))
        counts = np.zeros(self.n, np.float64)
        for _ in range(_T_STEP * self.present):
            b.cp_external_input_current[:] = drive
            if learn:
                self._set_apical(apical_vec)
            b._run_one_simulation_step()
            counts += np.asarray(to_host(b.cp_firing_states)).astype(np.float64)[self.res_idx]
        b.cp_external_input_current[:] = 0.0
        return counts / (_T_STEP * self.present)

    def train(self, feats, ys, epochs, lr_out, rng, mode="learn"):
        """PASS A: clean read + Wout delta rule + the terminal delta. PASS B (skip for mode='fixed'): credited teach with the
        constant apical k*(Y@delta) so the committed BDSP kernel moves input->pool W_in. mode 'lesion' -> apical off (no credit)."""
        Wout = np.zeros((self.n_classes, self.n + 1))
        order = list(range(len(feats)))
        for _ep in range(epochs):
            rng.shuffle(order)
            for i in order:
                r = self.forward(feats[i], learn=False, apical_vec=None)
                x = np.concatenate([r, [1.0]])
                p = _softmax(Wout @ x)
                delta = -p; delta[ys[i]] += 1.0
                Wout += lr_out * np.outer(delta, x)
                if mode == "fixed":
                    continue
                credit = None if mode == "lesion" else (self.k_apical * (self.Y @ delta))
                self.forward(feats[i], learn=True, apical_vec=credit)
        return Wout

    def predict(self, feat, Wout):
        r = self.forward(feat, learn=False, apical_vec=None)
        return int(np.argmax(Wout @ np.concatenate([r, [1.0]])))

    def no_weight_transport(self):
        return bool(np.array_equal(self.Y, self._Y0))


def run_one(seed, epochs=8, lr_out=0.02):
    B, y, is_held, cells, cat_code, q_code, D = bound_rates(seed)
    tr = ~is_held
    # COMMON-MODE REMOVAL (the composer's own opponency insight + the diagnostic root cause): the coincidence AND-banks carry
    # a large common mode (across-example std tiny), so a raw B/max drive fires the pool ~uniformly on the common mode. Drive
    # with the per-channel MEAN-SUBTRACTED (standardized on train) signal, shifted to [0,1], so the DISCRIMINATIVE variation is
    # the drive. Sparse structure-preserving expansion (Marr-Albus) so different pool neurons read different channel subsets.
    _, Bz = standardize(B[tr], B)
    feats = (Bz - Bz.min(0)) / (Bz.max(0) - Bz.min(0) + 1e-6)        # per-channel min-max of the common-mode-removed signal
    rng = np.random.default_rng(seed)
    out = {"seed": seed, "chance": round(1.0 / N_INTENT, 4), "n_held": int(is_held.sum()), "D": D}
    cfg = dict(n_pool=400, in_pop=3, fwd_density=0.25, in_hi=760.0, res_bias=60.0, fwd_wmean=44.0, present=4)
    # RUNG 3: the read-out over the fixed spiking bind LEARNED ON-BRIDGE by the committed BDSP rule (fully spiking)
    ro = OnBridgeBindReadout(feats.shape[1], N_INTENT, seed, **cfg)
    Wout = ro.train([feats[i] for i in range(len(feats))], y, epochs, lr_out, rng, mode="learn")
    pred = np.array([ro.predict(feats[i], Wout) for i in range(len(feats))])
    out["onbridge_held"] = round(float(np.mean(pred[is_held] == y[is_held])), 4)
    out["onbridge_train"] = round(float(np.mean(pred[tr] == y[tr])), 4)
    out["no_weight_transport"] = ro.no_weight_transport()
    # anti-cheat: apical-lesion (no on-bridge credit -> the read-out can't learn -> chance-ish on held-out)
    ro_l = OnBridgeBindReadout(feats.shape[1], N_INTENT, seed, **cfg)
    Wl = ro_l.train([feats[i] for i in range(len(feats))], y, epochs, lr_out, np.random.default_rng(seed), mode="lesion")
    predl = np.array([ro_l.predict(feats[i], Wl) for i in range(len(feats))])
    out["lesion_held"] = round(float(np.mean(predl[is_held] == y[is_held])), 4)
    # control: from-scratch classifier on the RAW [cat;q] (no fixed bind) -> the systematicity wall
    CAT = np.array([cat_code[c] for (c, q) in cells]); Q = np.array([q_code[q] for (c, q) in cells])
    C = np.concatenate([CAT, Q], axis=1); Ctr, Cev = standardize(C[tr], C)
    lay = _train_snn(Ctr, y[tr], [C.shape[1], 48, 48, N_INTENT], T, 120, 0.05, 1.0, seed, credit_mode="eprop")
    _, mlp_h, _ = score_snn(lay, Cev, y, is_held, 1.0); out["mlp_held"] = round(mlp_h, 4)
    out["GO"] = bool(out["onbridge_held"] > 0.5 and out["onbridge_held"] > out["mlp_held"] + 0.1
                     and out["onbridge_held"] > out["lesion_held"] + 0.1 and out["no_weight_transport"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="research/findings/raw/_onsubstrate_bind_onbridge_bdsp_readout.json")
    a = ap.parse_args()
    rows = [run_one(s, epochs=a.epochs) for s in a.seeds]
    for r in rows:
        print(f"[onbridge-bdsp-read s{r['seed']}] chance={r['chance']} || ON-BRIDGE-BDSP-LEARNED read-out over the FIXED "
              f"spiking bind held={r['onbridge_held']:.3f} (train {r['onbridge_train']:.3f}) | apical-lesion={r['lesion_held']:.3f} "
              f"| MLP-on-raw={r['mlp_held']:.3f} | no_transport={r['no_weight_transport']} || {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[onbridge-bdsp-read] {ngo}/{len(rows)} GO (fully-spiking: the read-out over the fixed spiking bind, LEARNED ON-BRIDGE "
          f"by the committed BDSP rule [no weight transport], extrapolates held-out >> from-scratch; apical-lesion collapses)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
