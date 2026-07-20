"""OnBridgeWKVFaculty — the FULLY-SPIKING grounded-answer renderer (drop-in for FTFaculty), on a real bridge.

The WKV cortex forward runs ON a `SimulationBridge`: each token's value is delivered through the PHASE of an RF
(resonate-and-fire) spike (gap#1's spiking input), charging the graded `cp_ssm_state`; the SSM's own trained read-out
picks the next word. This is `WKVFaculty` (the numpy reference) realized on spikes — so the WHOLE grounded-fluent turn
(the composer's retrieval + gate on a cupy bridge, and THIS renderer on a cupy bridge) runs in ONE cupy process =
the true one-brain-one-process north-star (the EMERGE-70/71 pattern for grounded fluent conversation).

`answer(facts_ctx, question)` matches FTFaculty/WKVFaculty EXACTLY. Reuse-by-import of the on-bridge builders
(`_build_ssm_state_bridge`, `_build_rf_encoder`, `_build_synaptic_rf_encoder`) from `_emerge_wkv_onbridge_derisk`;
the charge/generate loop replicates that runner's validated generation block (De-risk 3, RF-phase parity). GPU/cupy
(the bridge steps are the substrate). NO `sim/` edit.
"""
from __future__ import annotations
import numpy as np

from sim.backend import to_host, get_backend
from research.runners._emerge_wkv_onbridge_derisk import (
    _build_ssm_state_bridge, _build_rf_encoder, _build_synaptic_rf_encoder)
from research.runners._wkv_faculty import BIG_CKPT

_FT_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"


class OnBridgeWKVFaculty:
    def __init__(self, ckpt: str = _FT_CKPT, max_new: int = 8, seed: int = 42, rf_period: int = 200,
                 rf_synaptic: bool = False, rf_vmax_pct: float = 99.5):
        xp, _ = get_backend()
        self.xp = xp
        z = np.load(ckpt, allow_pickle=True)
        self.emb = np.asarray(z["emb.weight"], np.float64)
        self.ln_w = np.asarray(z["ln.weight"], np.float64); self.ln_b = np.asarray(z["ln.bias"], np.float64)
        self.Wv = np.asarray(z["Wv.weight"], np.float64); self.Wr = np.asarray(z["Wr.weight"], np.float64)
        self.Wo_sp = np.asarray(z["Wo_sp.weight"], np.float64)
        self.head_w = np.asarray(z["head.weight"], np.float64); self.head_b = np.asarray(z["head.bias"], np.float64)
        self.decay = float(np.exp(-np.log1p(np.exp(float(np.asarray(z["w"]).ravel()[0])))))
        self.words = [str(w) for w in z["words"]]
        self.V = len(self.words); self.D = self.emb.shape[1]
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.unk = self.w2i.get("<unk>", self.V - 1)
        self.max_new = int(max_new); self.seed = int(seed); self.ckpt = ckpt
        self.device = "cupy(on-bridge)" if str(xp.__name__) == "cupy" else "numpy(on-bridge)"
        self.n_invocations = 0
        self.npar = (self.emb.size + self.head_w.size + self.head_b.size + self.Wv.size + self.Wr.size
                     + self.Wo_sp.size + self.ln_w.size + self.ln_b.size) / 1e6

        D = self.D
        self._scg = max(1e-6, 1.0 - self.decay)
        # the ssm-state bridge (the WKV leaky state lives in cp_ssm_state)
        self.b, chan_groups, _cg2, _snap = _build_ssm_state_bridge(D, seed, self.decay, pop_k=1)
        self.nnrn = int(self.b.cp_membrane_potential_v.size)
        self.read_idx = np.concatenate([np.asarray(g) for g in chan_groups]).astype(np.int64)
        self.chan_of = np.concatenate([[c] * len(chan_groups[c]) for c in range(2 * D)]).astype(np.int64)
        self.gsize = np.array([len(g) for g in chan_groups], dtype=np.float64)
        # RF encoder (spiking input): 2D independent oscillators; VMAX from the vocab's inject distribution
        _vall = np.stack([self.Wv @ self._ln(self.emb[t]) for t in range(self.V)], 0)         # [V, D]
        _INJ = np.concatenate([np.maximum(_vall, 0.0), np.maximum(-_vall, 0.0)], 1) / self._scg  # [V, 2D]
        _flat = _INJ.reshape(-1)
        self._RF_VMAX = float(np.percentile(_flat[_flat > 0], rf_vmax_pct)) if (_flat > 0).any() else 1.0
        self._PLO, self._PHI = 0.05, 0.95
        self.rf_period = int(rf_period); self.rf_synaptic = bool(rf_synaptic)
        if rf_synaptic:
            self._rfb, self._enc, self._rdt = _build_synaptic_rf_encoder(2 * D, w=30.0, seed=seed)
            cal_v = np.linspace(0.0, self._RF_VMAX, 2 * D)
            g = self._syn_run(self._PLO + (self._PHI - self._PLO) * (cal_v / max(self._RF_VMAX, 1e-9)))
            self._gmax = max(g.max(), 1e-9); cm = g > 1e-9
            xc = np.log(np.clip(g[cm] / self._gmax, 1e-12, 1.0))
            self._coef = np.linalg.lstsq(np.vstack([xc, np.ones_like(xc)]).T, cal_v[cm], rcond=None)[0]
        else:
            self._rfb = _build_rf_encoder(2 * D, seed=seed)

    def _ln(self, v):
        return (v - v.mean()) / (v.std() + 1e-5) * self.ln_w + self.ln_b

    def ids(self, words):
        return [self.w2i.get(w, self.unk) for w in words]

    def in_vocab(self, w):
        return w in self.w2i and self.w2i[w] != self.unk

    def _syn_run(self, pvec):
        z = np.zeros(2 * (2 * self.D), np.complex64); z[self._enc] = np.exp(1j * 2 * np.pi * pvec).astype(np.complex64)
        mask = np.zeros(2 * (2 * self.D), bool); mask[self._enc] = True
        try:
            self._rfb.rf_kick(z, period=self.rf_period, neuron_mask=mask)
        except TypeError:
            self._rfb.rf_kick(z, period=self.rf_period)
        for _ in range(self.rf_period + 8):
            self._rfb.cp_external_input_current[:] = 0.0; self._rfb._run_one_simulation_step()
        return np.asarray(to_host(self._rfb.cp_conductance_g_nmda), np.float64)[self._rdt]

    def _rf_encode_decode(self, inj):
        d = np.clip(np.asarray(inj, np.float64), 0.0, self._RF_VMAX)
        p = self._PLO + (self._PHI - self._PLO) * (d / max(self._RF_VMAX, 1e-9))
        if self.rf_synaptic:
            g = self._syn_run(p)
            return np.clip(self._coef[0] * np.log(np.clip(g / self._gmax, 1e-12, 1.0)) + self._coef[1], 0.0, self._RF_VMAX)
        z = np.exp(1j * 2 * np.pi * p).astype(np.complex64)
        self._rfb.rf_kick(z, period=self.rf_period)
        for _ in range(self.rf_period + 8):
            self._rfb.cp_external_input_current[:] = 0.0; self._rfb._run_one_simulation_step()
        pr = np.asarray(to_host(self._rfb.rf_read_phases()), np.float64)
        return np.clip((pr - self._PLO) / (self._PHI - self._PLO) * self._RF_VMAX, 0.0, self._RF_VMAX)

    def _wash(self):
        for nm in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise", "cp_conductance_g_nmda",
                   "cp_conductance_g_e", "cp_conductance_g_i", "cp_firing_states",
                   "cp_ssm_state", "cp_ssm_inject", "cp_ssm_shunt"):
            arr = getattr(self.b, nm, None)
            if arr is not None:
                arr[:] = 0.0
        if self.b.cp_membrane_potential_v is not None:
            self.b.cp_membrane_potential_v[:] = -65.0
        if self.b.cp_recovery_variable_u is not None:
            self.b.cp_recovery_variable_u[:] = 0.0

    def _charge(self, tid):
        v = self.Wv @ self._ln(self.emb[tid])
        inj = np.concatenate([np.maximum(v, 0.0), np.maximum(-v, 0.0)]) / self._scg
        inj = self._rf_encode_decode(inj)                            # spiking input (RF-phase / synaptic)
        cur = np.zeros(self.nnrn, np.float32); cur[self.read_idx] = inj[self.chan_of].astype(np.float32)
        self.b.cp_ssm_inject[:] = self.xp.asarray(cur)
        self.b.cp_ssm_shunt[:] = 0.0
        self.b._run_one_simulation_step()
        st = np.asarray(to_host(self.b.cp_ssm_state)).astype(np.float64)
        agg = np.zeros(2 * self.D, np.float64); np.add.at(agg, self.chan_of, st[self.read_idx])
        return agg / self.gsize

    def _next_logits(self, tid, state):
        rh = 1.0 / (1.0 + np.exp(-(self.Wr @ self._ln(self.emb[tid]))))
        return self.head_w @ (rh * (self.Wo_sp @ state)) + self.head_b

    def generate(self, prompt_words, max_new=None, stop_words=None, stop_on_repeat=True):
        max_new = self.max_new if max_new is None else int(max_new)
        ids = self.ids([w for w in prompt_words if w]) or [self.w2i.get("the", 0)]
        self._wash()
        state = None
        for t in ids:
            state = self._charge(t)
        gen = list(ids); n0 = len(ids); stop = set(stop_words or [])
        for _ in range(max_new):
            lg = self._next_logits(gen[-1], state).copy(); lg[self.unk] = -1e30
            nxt = int(np.argmax(lg)); w = self.words[nxt] if 0 <= nxt < self.V else "<unk>"
            if w in stop:
                break
            gen.append(nxt)
            g = gen[n0:]
            if stop_on_repeat and ((len(g) >= 2 and g[-1] == g[-2]) or (len(g) >= 4 and g[-1] == g[-3] and g[-2] == g[-4])):
                gen.pop(); break
            state = self._charge(nxt)
        return [self.words[i] for i in gen[n0:]]

    def answer(self, facts_ctx, question, max_new=None):
        self.n_invocations += 1
        prompt = [w for w in facts_ctx.replace(".", " ").split() if self.in_vocab(w)]
        out = self.generate(prompt + ["<ans>"], max_new=(self.max_new if max_new is None else max_new),
                            stop_words={"<eos>"})
        out = [w for w in out if w not in ("<ans>", "<eos>")]
        return " ".join(out).strip()


if __name__ == "__main__":
    import sys
    fac = OnBridgeWKVFaculty(rf_period=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    print(f"OnBridgeWKVFaculty: V={fac.V} D={fac.D} decay={fac.decay:.4f} dev={fac.device} period={fac.rf_period}")
    # verify-first: reproduce the validated De-risk 3 on-bridge output
    for ctx, exp in [("the dog eats meat .", "the dog eats meat"), ("the fox chases rabbit .", "the fox chases rabbit"),
                     ("the bee makes honey .", "the bee makes honey")]:
        got = fac.answer(ctx, "")
        ok = "OK" if got == exp else "MISMATCH"
        print(f"  [{ok}] answer('{ctx}') -> '{got}' (expect '{exp}')")
