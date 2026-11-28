# Capability: Language
# Record: brain/experiments/results/2026-11-27-spiking-twin-e1/run/parity/bias_diagnosis.md
# Expected-fail: none (a diagnostic of the S1 parity miss, not a capability claim)
"""Why does fla's spikes-on mean NLL differ from the loop's (S1, teacher_on_stat.json)? GPU diagnostic.

Same setup as test_gpu_spikes_on_statistical_teacher (0.1B teacher twin, calibrated on 2 WT103 val windows, read on 8
other windows x 512). DiagTwin re-implements _tmix_b verbatim with a pluggable WKV kernel and checks itself bit-equal to
SpikingTwin (loop and fla) first. Variants (end-to-end per-token NLL, per-site count histograms, o/head pre-quantiser
values u = x / theta against the loop fp32 reference):
  loop32, loop32 again (determinism), loop AMP, chunk fp32, fused_recurrent fp32, chunk AMP bf16-in, fused AMP bf16-in,
  loop with an fp64 recurrence, and the loop with the chunk's own error injected at every layer: y = y_loop + s * d,
  d = y_chunk - y_loop on the SAME inputs, s = +1, -1, or a random sign per element (6 seeds: the null for a signless
  perturbation of the same size). Isolated per-layer kernel stats (same inputs): rel err, signed mean, slope of d on y,
  o-site count flips up/down. --tag names the run (e.g. ieee under TRITON_F32_DEFAULT=ieee)."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from brain.spiking_twin import SpikingTwin, fla_import, wkv_fla_args, LOG_DECAY_MIN

SEEDS = (42, 43, 44, 100, 101, 102)


class DiagTwin(SpikingTwin):
    kern = "loop"          # loop | chunk | fused | loop64 | inject
    kdt = torch.float32    # dtype handed to an fla kernel
    inj = None             # inject: +1.0, -1.0 or a seed (random sign per element)
    iso = None             # dict to fill with isolated per-layer stats (or None)
    rec = None             # dict site -> {"hist": 9 ints, "u": tensor (only for KEEP sites)}
    keep_u = False

    def _q(self, name, x):
        if self.rec is not None and self.spikes and name in self.sites and self._calib is None:
            th = self.theta[name].to(x.dtype)
            u = (x / th).float()
            c = torch.round(torch.clamp(u, -self.max_count, self.max_count))
            r = self.rec.setdefault(name, {"hist": torch.zeros(2 * self.max_count + 1, dtype=torch.float64)})
            r["hist"] += torch.bincount((c + self.max_count).long().flatten(), minlength=2 * self.max_count + 1
                                        ).double().cpu()
            if self.keep_u and (name.endswith(".o") or name == "head"):
                r["u"] = u.detach().reshape(-1).cpu()
        return super()._q(name, x)

    # -- kernels on identical inputs ------------------------------------------------------------------------------
    def _k_fla(self, which, r, logw, k, v, kk, a, S, dt, chunk_size=None, lower_bound=None):
        chunk_rwkv7, fused_rwkv7 = fla_import()
        B, T, C = r.shape
        with torch.autocast(device_type="cuda", enabled=False):
            kw = wkv_fla_args(r.to(dt), logw.to(dt), k.to(dt), v.to(dt), kk.to(dt), a.to(dt), S.float())
            kw["initial_state"] = kw["initial_state"].contiguous()
            if which == "chunk":
                if chunk_size is not None:
                    kw["chunk_size"] = chunk_size
                if lower_bound is not None:
                    kw["lower_bound"] = lower_bound
                o, ht = chunk_rwkv7(**kw)
            else:
                o, ht = fused_rwkv7(**kw)
        return o.reshape(B, T, C).float(), ht.float().transpose(-1, -2)

    def _k_loop64(self, B, T, H, N, r, logw, k, v, kk, a, S):
        w = torch.exp(logw.double())
        S = S.double()
        rf, kf, vf, kkf, af = r.double(), k.double(), v.double(), kk.double(), a.double()
        outs = []
        with torch.autocast(device_type="cuda", enabled=False):
            for t in range(T):
                kt = kkf[:, t].view(B, H, N)
                sa = S @ (-kt).unsqueeze(-1)
                S = S * w[:, t].view(B, H, 1, N) + sa * (kt * af[:, t].view(B, H, N)).unsqueeze(-2) \
                    + vf[:, t].view(B, H, N, 1) * kf[:, t].view(B, H, 1, N)
                outs.append((S @ rf[:, t].view(B, H, N, 1)).view(B, H * N))
        return torch.stack(outs, 1).float(), S.float()

    def _post(self, i, y, r, k, v, g, B, T, C, dtype):
        """y -> the o-site input (group norm, r_k bonus, gate), exactly as _tmix_b does it."""
        z, H, N = self.z, self.H, self.N
        a_ = f"blocks.{i}.att."
        y = y.to(dtype)
        y = F.group_norm(y.reshape(B * T, C), num_groups=H, weight=z[a_ + "ln_x.weight"], bias=z[a_ + "ln_x.bias"],
                         eps=64e-5).view(B, T, C)
        y = y + ((r * k * z[a_ + "r_k"]).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C)
        return y * g

    def _tmix_b(self, i, x, x_prev, v_first, S):
        # verbatim SpikingTwin._tmix_b up to the WKV call
        z, H, N, q = self.z, self.H, self.N, self._q
        a_ = f"blocks.{i}.att."
        n = f"{i}."
        B, T, C = x.shape
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), 1) - x
        xr, xw, xk = x + xx * z[a_ + "x_r"], x + xx * z[a_ + "x_w"], x + xx * z[a_ + "x_k"]
        xv, xa, xg = x + xx * z[a_ + "x_v"], x + xx * z[a_ + "x_a"], x + xx * z[a_ + "x_g"]
        r = q(n + "r", xr) @ z[a_ + "receptance.weight"]
        w = q(n + "w2", torch.tanh(q(n + "w1", xw) @ z[a_ + "w1"])) @ z[a_ + "w2"]
        k = q(n + "k", xk) @ z[a_ + "key.weight"]
        v = q(n + "v", xv) @ z[a_ + "value.weight"]
        a = torch.sigmoid(z[a_ + "a0"] + q(n + "a2", q(n + "a1", xa) @ z[a_ + "a1"]) @ z[a_ + "a2"])
        g = q(n + "g2", torch.sigmoid(q(n + "g1", xg) @ z[a_ + "g1"])) @ z[a_ + "g2"]
        kk = F.normalize((k * z[a_ + "k_k"]).view(B, T, H, N), dim=-1, p=2.0).view(B, T, H * N)
        k = k * (1 + (a - 1) * z[a_ + "k_a"])
        if i == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[a_ + "v0"] + q(n + "v2", q(n + "v1", xv) @ z[a_ + "v1"]) @ z[a_ + "v2"])
        logw = -0.606531 * torch.sigmoid((z[a_ + "w0"] + w).float())
        kern = self.kern
        if kern == "chunk":
            y, S2 = self._k_fla("chunk", r, logw, k, v, kk, a, S, self.kdt)
            y = y.to(self.kdt).to(x.dtype)            # SpikingTwin: o (kdt) .to(x.dtype)
        elif kern == "fused":
            y, S2 = self._k_fla("fused", r, logw, k, v, kk, a, S, self.kdt)
            y = y.to(self.kdt).to(x.dtype)
        elif kern == "loop64":
            y, S2 = self._k_loop64(B, T, H, N, r, logw, k, v, kk, a, S)
            y = y.to(x.dtype)
        else:
            y, S2 = self._wkv_loop(B, T, H, N, r, w, k, v, kk, a, S, z, a_, x)
        if self.iso is not None:
            self._iso_layer(i, y, r, logw, k, v, kk, a, S, g, B, T, H, N, C, x.dtype)
        if kern == "inject":
            yc, _ = self._k_fla("chunk", r, logw, k, v, kk, a, S, self.kdt)
            d = yc.to(x.dtype) - y
            if isinstance(self.inj, float):
                s = self.inj
            else:
                gen = torch.Generator(device=d.device).manual_seed(int(self.inj) * 1000 + i)
                s = torch.randint(0, 2, d.shape, generator=gen, device=d.device).to(d.dtype) * 2 - 1
            y = y + s * d
        y = F.group_norm(y.reshape(B * T, C), num_groups=H, weight=z[a_ + "ln_x.weight"], bias=z[a_ + "ln_x.bias"],
                         eps=64e-5).view(B, T, C)
        y = y + ((r * k * z[a_ + "r_k"]).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C)
        return q(n + "o", y * g) @ z[a_ + "output.weight"], x[:, -1], S2, v_first

    @torch.no_grad()
    def _iso_layer(self, i, y_ref, r, logw, k, v, kk, a, S, g, B, T, H, N, C, dtype):
        """Every kernel on THIS layer's inputs (from the reference run): output error stats and o-site flips."""
        th = self.theta[f"{i}.o"].float()
        K = self.max_count
        u_ref = (self._post(i, y_ref, r, k, v, g, B, T, C, dtype).float() / th)
        c_ref = torch.round(torch.clamp(u_ref, -K, K))
        yr = y_ref.double()
        variants = {"chunk32": lambda: self._k_fla("chunk", r, logw, k, v, kk, a, S, torch.float32)[0],
                    "chunk32_c64lb": lambda: self._k_fla("chunk", r, logw, k, v, kk, a, S, torch.float32, 64,
                                                         LOG_DECAY_MIN)[0],
                    "fused32": lambda: self._k_fla("fused", r, logw, k, v, kk, a, S, torch.float32)[0],
                    "chunk_bf16in": lambda: self._k_fla("chunk", r, logw, k, v, kk, a, S, torch.bfloat16)[0],
                    "fused_bf16in": lambda: self._k_fla("fused", r, logw, k, v, kk, a, S, torch.bfloat16)[0],
                    "loop64": lambda: self._k_loop64(B, T, H, N, r, logw, k, v, kk, a, S)[0]}
        for name, fn in variants.items():
            yv = fn().to(dtype)
            d = yv.double() - yr
            pos = torch.arange(T, device=d.device)
            within = (d.pow(2).mean(dim=(0, 2)).sqrt())          # rms error per position
            u = self._post(i, yv, r, k, v, g, B, T, C, dtype).float() / th
            c = torch.round(torch.clamp(u, -K, K))
            du = (u - u_ref).double()
            sgn = torch.sign(u_ref).double()
            st = {"rel_err": float(d.norm() / yr.norm()),
                  "mean_d_over_rms_y": float(d.mean() / yr.pow(2).mean().sqrt()),
                  "slope_d_on_y": float((d * yr).sum() / yr.pow(2).sum()),            # y_v = (1 + slope) y_ref + ...
                  "radial_d_over_rms_y": float((d * torch.sign(yr)).mean() / yr.pow(2).mean().sqrt()),
                  "err_rms_first16_vs_rest": [float(within[:16].mean()), float(within[16:].mean())],
                  "err_rms_by_pos_mod16": [float(within[pos % 16 == j].mean()) for j in range(16)],
                  "err_rms_by_quarter": [float(within[j * T // 4:(j + 1) * T // 4].mean()) for j in range(4)],
                  "o_du_mean": float(du.mean()), "o_du_radial": float((du * sgn).mean()),
                  "o_du_rms": float(du.pow(2).mean().sqrt()),
                  "o_flips": int((c != c_ref).sum()), "o_n": int(c.numel()),
                  "o_flips_up": int((c > c_ref).sum()), "o_flips_down": int((c < c_ref).sum()),
                  "o_flips_abs_up": int((c.abs() > c_ref.abs()).sum()),
                  "o_flips_abs_down": int((c.abs() < c_ref.abs()).sum())}
            self.iso.setdefault(name, {})[str(i)] = st


def nll_of(tw, ev, amp):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        lg = tw.forward_batch(ev[:, :-1])[0].float()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ev[:, 1:].reshape(-1), reduction="none").double().cpu()


def dstats(nll, ref, W, T):
    d = (nll - ref)
    dw = d.view(W, T).mean(1)
    g = torch.Generator().manual_seed(0)
    blocks = d.view(W, T // 64, 64).mean(-1).reshape(-1)          # 64-token blocks
    boot = blocks[torch.randint(0, blocks.numel(), (4000, blocks.numel()), generator=g)].mean(1)
    se_w = float(dw.std() / W ** 0.5)
    pos = d.view(W, T)
    return {"mean": float(d.mean()), "abs_mean": abs(float(d.mean())), "rms": float(d.pow(2).mean().sqrt()),
            "window_means": [round(float(x), 4) for x in dw], "se_windows": se_w,
            "se_block_bootstrap": float(boot.std()), "t_windows": float(d.mean()) / max(se_w, 1e-12),
            "frac_tokens_equal": float((d == 0).double().mean()),
            "abs_d_by_quarter": [float(pos[:, j * T // 4:(j + 1) * T // 4].abs().mean()) for j in range(4)],
            "nll_mean": float(nll.mean())}


def site_cmp(rec, ref):
    """Per-site count histogram and o/head pre-quantiser comparison against the reference run."""
    out, tot_l1, tot_n, sm = {}, 0.0, 0.0, 0.0
    levels = torch.arange(-4, 5, dtype=torch.float64)
    for s in sorted(ref):
        h, h0 = rec[s]["hist"], ref[s]["hist"]
        n = float(h0.sum())
        tot_l1 += float((h - h0).abs().sum()) / 2
        tot_n += n
        e = {"mean_abs_count": float((h * levels.abs()).sum() / n), "ref_mean_abs_count": float((h0 * levels.abs()).sum() / n),
             "mean_count": float((h * levels).sum() / n), "ref_mean_count": float((h0 * levels).sum() / n)}
        sm += e["mean_abs_count"] - e["ref_mean_abs_count"]
        if "u" in rec[s] and "u" in ref[s]:
            u, u0 = rec[s]["u"].double(), ref[s]["u"].double()
            c, c0 = torch.round(torch.clamp(u, -4, 4)), torch.round(torch.clamp(u0, -4, 4))
            du = u - u0
            e.update({"hist": [int(x) for x in h], "ref_hist": [int(x) for x in h0],
                      "du_mean": float(du.mean()), "du_radial": float((du * torch.sign(u0)).mean()),
                      "du_rms": float(du.pow(2).mean().sqrt()), "flips": int((c != c0).sum()),
                      "flips_abs_up": int((c.abs() > c0.abs()).sum()), "flips_abs_down": int((c.abs() < c0.abs()).sum()),
                      "frac_exact": float((du == 0).double().mean())})
        out[s] = e
    return {"hist_moved_share": tot_l1 / tot_n, "sum_d_mean_abs_count_over_sites": sm, "sites": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="fewer variants (ieee rerun)")
    args = ap.parse_args()
    from brain.tests.test_spiking_twin_fla import _teacher, _heldout_windows
    t0 = time.time()
    c = _teacher()
    win = _heldout_windows(c.pipe, 10)
    base = SpikingTwin(c, spikes=True)
    base.calibrate([win[:2, :-1]])
    ev = win[2:]
    W, T = ev.shape[0], ev.shape[1] - 1
    res = {"tag": args.tag, "env": {k: os.environ.get(k) for k in ("TRITON_F32_DEFAULT", "FLA_TRIL_PRECISION",
                                                                     "TRITON_CACHE_DIR")},
           "tf32_torch": torch.backends.cuda.matmul.allow_tf32, "tokens": W * T}

    tw = DiagTwin(c, spikes=True)
    tw.theta = dict(base.theta)

    def run(kern, amp=False, kdt=torch.float32, inj=None, iso=False, keep_u=True):
        tw.kern, tw.kdt, tw.inj = kern, kdt, inj
        tw.iso = {} if iso else None
        tw.rec, tw.keep_u = {}, keep_u
        nll = nll_of(tw, ev, amp)
        rec, isod = tw.rec, tw.iso
        tw.rec, tw.iso = None, None
        return nll, rec, isod

    # self-check: DiagTwin == SpikingTwin, loop and fla (bit-equal per-token NLL)
    sp_loop = nll_of(base, ev, False)
    fla_tw = SpikingTwin(c, spikes=True, wkv="fla")
    fla_tw.theta = dict(base.theta)
    sp_fla = nll_of(fla_tw, ev, False)
    ref, ref_rec, iso = run("loop", iso=True)
    chk = {"diag_loop_eq_spikingtwin_loop": bool(torch.equal(ref, sp_loop))}
    nll_c, rec_c, _ = run("chunk")
    chk["diag_chunk_eq_spikingtwin_fla"] = bool(torch.equal(nll_c, sp_fla))
    res["selfcheck"] = chk
    res["iso_per_layer"] = iso
    res["ref_nll_mean"] = float(ref.mean())
    E = {}
    E["chunk32"] = (nll_c, rec_c)
    variants = [("loop32_again", dict(kern="loop")), ("fused32", dict(kern="fused")),
                ("loop64", dict(kern="loop64")), ("inject_plus", dict(kern="inject", inj=1.0)),
                ("inject_minus", dict(kern="inject", inj=-1.0))]
    if not args.quick:
        variants += [(f"inject_randsign_s{s}", dict(kern="inject", inj=s)) for s in SEEDS]
    for name, kw in variants:
        nll, rec, _ = run(**kw)
        E[name] = (nll, rec)
        print(f"[{time.time() - t0:.0f}s] {name} mean dNLL {float((nll - ref).mean()):+.4f}", flush=True)
    res["vs_loop32"] = {name: {"nll": dstats(n, ref, W, T), "sites": site_cmp(r, ref_rec)} for name, (n, r) in E.items()}
    if not args.quick:
        amp_ref, amp_rec, iso_amp = run("loop", amp=True, iso=True)
        res["iso_per_layer_amp"] = iso_amp
        A = {"loop_amp": (amp_ref, amp_rec)}
        for name, kw in [("chunk_amp_bf16in", dict(kern="chunk", amp=True, kdt=torch.bfloat16)),
                         ("fused_amp_bf16in", dict(kern="fused", amp=True, kdt=torch.bfloat16)),
                         ("chunk_amp_fp32in", dict(kern="chunk", amp=True, kdt=torch.float32)),
                         ("inject_amp_bf16_plus", dict(kern="inject", amp=True, kdt=torch.bfloat16, inj=1.0)),
                         ("inject_amp_bf16_minus", dict(kern="inject", amp=True, kdt=torch.bfloat16, inj=-1.0))]:
            nll, rec, _ = run(**kw)
            A[name] = (nll, rec)
            print(f"[{time.time() - t0:.0f}s] {name} mean dNLL vs loop_amp {float((nll - amp_ref).mean()):+.4f}",
                  flush=True)
        res["floor_loop_amp_vs_loop32"] = {"nll": dstats(amp_ref, ref, W, T), "sites": site_cmp(amp_rec, ref_rec)}
        res["vs_loop_amp"] = {name: {"nll": dstats(n, amp_ref, W, T), "sites": site_cmp(r, amp_rec)}
                              for name, (n, r) in A.items() if name != "loop_amp"}
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
