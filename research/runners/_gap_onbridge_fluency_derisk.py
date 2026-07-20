"""End-goal deliverable (owner steer 2026-07-20): the WKV cortex's PRETRAINING (fluency) learned ON the spiking
substrate by a pure exact delta rule — fully-spiking, one shared substrate.

The read-out FORWARD runs IN the bridge step loop (`cp_ssm_readout_out = cp_ssm_readout_w @ cp_ssm_state`, the committed
additive mechanism, verified byte-exact) + a host current-token term; the read-out is trained ON the substrate by the
DELTA rule (`dw = -eta * err * state`, `cp_ssm_state` as the presynaptic eligibility — no BPTT, no weight transport,
no adaptive optimizer). The WKV cortex (emb/Wv/decay) is the FIXED reservoir. Task: TinyStories next-token (full vocab).
Metric: held-out ppl (does the cortex's fluency LEARN on the substrate?). SMOKE-scale (on-bridge per-token stepping is
slow) — proof the mechanism runs on the substrate + ppl drops; off-bridge (`_gap_pretraining_shallow_fluency_derisk`)
already characterized the plateau (~40). ⚠ cfg.seed set. GPU (cupy).
"""
from __future__ import annotations
import argparse, os, random, time

os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402
from research.runners._emerge_wkv_onbridge_derisk import _build_ssm_state_bridge  # noqa: E402
from research.runners._gap_grounded_wkv_finetune import load_tiny_sentences  # noqa: E402

BIG = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_big_seed42.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BIG)
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frozen", action="store_true", help="anti-cheat: no weight update -> ppl stays at chance")
    args = ap.parse_args()

    xp, _ = get_backend()
    rng = np.random.default_rng(args.seed)
    z = np.load(args.ckpt, allow_pickle=True)
    emb = np.asarray(z["emb.weight"], np.float64); ln_w = np.asarray(z["ln.weight"], np.float64); ln_b = np.asarray(z["ln.bias"], np.float64)
    Wv = np.asarray(z["Wv.weight"], np.float64)
    decay = float(np.exp(-np.log1p(np.exp(float(np.asarray(z["w"]).ravel()[0])))))
    words = [str(w) for w in z["words"]]; V = len(words); D = int(z["d_model"]); w2i = {w: i for i, w in enumerate(words)}
    unk = w2i.get("<unk>", V - 1)

    def _ln(v): return (v - v.mean()) / (v.std() + 1e-5) * ln_w + ln_b

    b, chan_groups, _cg2, _snap = _build_ssm_state_bridge(D, args.seed, decay, pop_k=1)
    N = int(b.cp_membrane_potential_v.size)
    read_idx = np.concatenate([np.asarray(gp) for gp in chan_groups]).astype(np.int64)
    chan_of = np.concatenate([[c] * len(chan_groups[c]) for c in range(2 * D)]).astype(np.int64)
    _scg = max(1e-6, 1.0 - decay)

    # single-linear read-out over the on-bridge state + a host current-token term (the off-bridge de-risk's config)
    Wsl = (rng.standard_normal((V, N)) / N ** 0.5).astype(np.float32)   # state read-out (ON-BRIDGE via cp_ssm_readout_w)
    Wh = (rng.standard_normal((V, D)) / D ** 0.5).astype(np.float32)    # current-token term (host)
    b.cp_ssm_readout_w = xp.asarray(Wsl)

    # precompute the per-token inject (Wv@ln(emb)) once (the fixed reservoir input) + the current-token feature
    _emb_ln = np.stack([_ln(emb[t]) for t in range(V)], 0)             # [V,D]
    _vtok = _emb_ln @ Wv.T                                             # [V,D]
    _inj = np.concatenate([np.maximum(_vtok, 0.0), np.maximum(-_vtok, 0.0)], 1) / _scg   # [V,2D]

    def wash():
        for nm in ("cp_ssm_state", "cp_ssm_inject", "cp_ssm_shunt", "cp_conductance_g_nmda"):
            arr = getattr(b, nm, None)
            if arr is not None:
                arr[:] = 0.0

    def charge(tid):
        cur_ = np.zeros(N, np.float32); cur_[read_idx] = _inj[tid][chan_of].astype(np.float32)
        b.cp_ssm_inject[:] = xp.asarray(cur_); b.cp_ssm_shunt[:] = 0.0
        b._run_one_simulation_step()
        return (np.asarray(to_host(b.cp_ssm_readout_out), np.float64),   # = Wsl @ state (on-bridge)
                np.asarray(to_host(b.cp_ssm_state), np.float64))

    tiny = load_tiny_sentences(args.corpus, args.n_train + args.n_eval, w2i, min_len=5, max_len=18)
    tr = tiny[:args.n_train]; ev = tiny[args.n_train:args.n_train + args.n_eval]

    def run_sentence(ids, train):
        wash(); nll = 0.0; nt = 0
        for t in range(len(ids) - 1):
            out, state = charge(ids[t])
            logits = out + Wh @ _emb_ln[ids[t]]                        # on-bridge state read-out + host current-token
            p = np.exp(logits - logits.max()); p /= p.sum()
            nll += -np.log(max(p[ids[t + 1]], 1e-12)); nt += 1
            if train:
                err = p.copy(); err[ids[t + 1]] -= 1.0                 # softmax - onehot (exact, local for the output layer)
                Wsl[:] = Wsl - args.lr * np.outer(err, state).astype(np.float32)          # DELTA over the on-bridge state
                Wh[:] = Wh - args.lr * np.outer(err, _emb_ln[ids[t]]).astype(np.float32)  # delta on the current-token term
                b.cp_ssm_readout_w = xp.asarray(Wsl)
        return nll, nt

    def ppl():
        tot = 0.0; nt = 0
        for ids in ev:
            n, c = run_sentence(ids, False); tot += n; nt += c
        return float(np.exp(tot / max(1, nt)))

    p0 = ppl()
    print(f"[on-bridge fluency] V={V} N={N} decay={decay:.3f}; ppl BEFORE {p0:.2f} (chance ~{V}; off-bridge shallow plateau ~40)")
    t0 = time.time()
    for ep in range(args.epochs):
        order = rng.permutation(len(tr))
        for i in order:
            run_sentence(tr[i], not args.frozen)
        print(f"[epoch {ep+1}/{args.epochs}] on-bridge held-out ppl = {ppl():.2f} ({time.time()-t0:.0f}s)", flush=True)
    tag = "FROZEN" if args.frozen else "MAIN"
    print(f"\n[RESULT {tag}] the WKV cortex's FLUENCY learned ON the spiking substrate by a pure exact delta rule: "
          f"held-out ppl {p0:.2f} -> {ppl():.2f} -- does the pretraining LEARN on the substrate?")


if __name__ == "__main__":
    main()
