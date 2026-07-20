"""Rung (ii) — the on-bridge fully-spiking read-out LEARNING de-risk: a graded read-out over the on-bridge SSM state,
trained ON the substrate by a pure local plasticity rule (delta rule), no BPTT, no weight transport.

Builds on the committed additive `cp_ssm_readout_w`/`cp_ssm_readout_out` mechanism (`sim/bridge.py`): the read-out
FORWARD `out = W @ cp_ssm_state` runs IN the bridge step loop (verified byte-exact). Here the read-out is a SINGLE
linear layer (`logits = W @ state`) trained by the DELTA RULE `dw = -eta * error[post] * state[pre]` — for a single
output layer the error is LOCAL (no feedback pathway, no transport), and reading `cp_ssm_state` as the presynaptic
eligibility makes the whole update local + on-substrate (no BPTT: the state is the fixed reservoir's own dynamics).

The WKV cortex (emb/Wv/decay) is the FIXED reservoir: per token it charges the on-bridge graded state (M1 path,
`cp_ssm_inject = v/(1-decay)`); the read-out learns the grounded next-token map over that state. Reduced vocab (the
grounded words) so the classifier is tractable. Metric: grounded next-token accuracy on the answer span.

⚠ set cfg.seed (the actual_seed_used bug). Anti-cheats: shuffle the (state->readout) association must collapse; a
frozen (un-trained) read-out is at chance; memoryless state (k_leak=1) collapses. GPU (cupy) for speed.
"""
from __future__ import annotations
import argparse, json, os, sys

os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402
from research.runners._emerge_wkv_onbridge_derisk import _build_ssm_state_bridge  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, SUBJECTS, OBJECTS  # noqa: E402

BIG = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_big_seed42.npz"
CUR = "research/findings/raw/_grounded_lang_curriculum_p2.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BIG)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--shuffle-elig", action="store_true", help="anti-cheat: shuffle the state->readout association (must collapse)")
    ap.add_argument("--memoryless", action="store_true", help="anti-cheat: k_leak=1 (no memory) -> collapse")
    ap.add_argument("--frozen", action="store_true", help="anti-cheat: no weight update -> chance")
    args = ap.parse_args()

    xp, bname = get_backend()
    rng = np.random.default_rng(args.seed)
    z = np.load(args.ckpt, allow_pickle=True)
    emb = np.asarray(z["emb.weight"], np.float64); ln_w = np.asarray(z["ln.weight"], np.float64); ln_b = np.asarray(z["ln.bias"], np.float64)
    Wv = np.asarray(z["Wv.weight"], np.float64)
    decay = float(np.exp(-np.log1p(np.exp(float(np.asarray(z["w"]).ravel()[0])))))
    words = [str(w) for w in z["words"]]; w2i = {w: i for i, w in enumerate(words)}; unk = w2i.get("<unk>", len(words) - 1)
    D = int(z["d_model"])

    def _ln(v): return (v - v.mean()) / (v.std() + 1e-5) * ln_w + ln_b

    # reduced grounded vocab (curriculum + function + markers, all in-vocab)
    cur = json.load(open(CUR)); heldout = {tuple(f) for f in cur["facts"]}
    subs = [s for s in SUBJECTS if s in w2i and w2i[s] != unk]; objs = [o for o in OBJECTS if o in w2i and w2i[o] != unk]
    verbs = [(b, s) for (b, s, _p) in VERBS if b in w2i and s in w2i and w2i[s] != unk]
    red = ["the"] + subs + objs + [x for bs in verbs for x in bs] + ["<ANS>", "<EOS>"]
    red = list(dict.fromkeys(red))                                   # dedup, keep order
    r2i = {w: i for i, w in enumerate(red)}; nv = len(red)
    ANS, EOS = r2i["<ANS>"], r2i["<EOS>"]
    print(f"[reduced vocab] {nv} grounded words; reservoir decay={decay:.4f}, mem~{1/(1-decay):.1f} tok")

    # WKV emb id for a reduced word (markers use a fixed random emb since the base ckpt has none)
    _marker_emb = {ANS: 0.02 * rng.standard_normal(D), EOS: 0.02 * rng.standard_normal(D)}

    def wkv_emb(rid):
        w = red[rid]
        return _marker_emb[rid] if rid in _marker_emb else emb[w2i[w]]

    # the frozen reservoir: build the ssm-state bridge
    from sim.config import CoreSimConfig  # noqa: F401  (bridge builder reads its own cfg)
    dec = 0.0 if args.memoryless else decay
    b, chan_groups, _cg2, _snap = _build_ssm_state_bridge(D, args.seed, dec, pop_k=1)
    N = int(b.cp_membrane_potential_v.size)
    read_idx = np.concatenate([np.asarray(gp) for gp in chan_groups]).astype(np.int64)
    chan_of = np.concatenate([[c] * len(chan_groups[c]) for c in range(2 * D)]).astype(np.int64)
    gsize = np.array([len(gp) for gp in chan_groups], np.float64)
    _scg = max(1e-6, 1.0 - decay)

    # the READ-OUT weights W [nv, N] (single linear layer over the on-bridge state); small random init
    W = (0.01 * rng.standard_normal((nv, N))).astype(np.float32)
    perm = rng.permutation(N) if args.shuffle_elig else None         # anti-cheat: shuffle state->readout association
    b.cp_ssm_readout_w = xp.asarray(W)

    def charge(rid):
        v = Wv @ _ln(wkv_emb(rid))
        inj = np.concatenate([np.maximum(v, 0.0), np.maximum(-v, 0.0)]) / _scg   # [2D]
        cur_ = np.zeros(N, np.float32); cur_[read_idx] = inj[chan_of].astype(np.float32)
        b.cp_ssm_inject[:] = xp.asarray(cur_); b.cp_ssm_shunt[:] = 0.0
        b._run_one_simulation_step()
        return np.asarray(to_host(b.cp_ssm_readout_out), np.float64), np.asarray(to_host(b.cp_ssm_state), np.float64)

    def wash():
        for nm in ("cp_ssm_state", "cp_ssm_inject", "cp_ssm_shunt", "cp_conductance_g_nmda"):
            arr = getattr(b, nm, None)
            if arr is not None:
                arr[:] = 0.0

    def frame():
        a = rng.choice(subs); (vb, vs) = verbs[rng.integers(len(verbs))]; p = rng.choice(objs)
        if (a, vb, p) in heldout:
            return frame()
        f = [r2i["the"], r2i[a], r2i[vs], r2i[p]]
        return f + [ANS] + f + [EOS], [0] * 4 + [0] + [1] * 4 + [1]

    def run_frame(train):
        seq, mask = frame()
        wash(); correct = tot = 0
        for t in range(len(seq) - 1):
            out, state = charge(seq[t])
            if mask[t + 1]:
                pred = int(np.argmax(out)); correct += int(pred == seq[t + 1]); tot += 1
                if train:
                    p = np.exp(out - out.max()); p /= p.sum(); err = p.copy(); err[seq[t + 1]] -= 1.0   # softmax - onehot
                    elig = state[perm] if perm is not None else state           # presynaptic eligibility = the on-bridge state
                    W[:] = W - args.lr * np.outer(err, elig).astype(np.float32)  # DELTA RULE (local, no transport, no BPTT)
                    b.cp_ssm_readout_w = xp.asarray(W)
        return correct, tot

    # verify-first: one training frame must reduce that frame's error (else a sign bug)
    _c0 = sum(run_frame(False)[0] for _ in range(20))
    for _ in range(50):
        run_frame(not args.frozen)
    _c1 = sum(run_frame(False)[0] for _ in range(20))
    print(f"[verify-first] grounded-correct/20-frames: {_c0} -> {_c1} ({'LEARNS' if _c1 >= _c0 else 'no gain'})")

    for ep in range(args.epochs):
        for _ in range(args.n_frames):
            run_frame(not args.frozen)
        ev = np.array([run_frame(False) for _ in range(60)]); acc = ev[:, 0].sum() / max(1, ev[:, 1].sum())
        print(f"[epoch {ep+1}/{args.epochs}] grounded next-token acc = {acc:.3f}", flush=True)

    ev = np.array([run_frame(False) for _ in range(200)]); acc = float(ev[:, 0].sum() / max(1, ev[:, 1].sum()))
    tag = "SHUFFLE-ELIG" if args.shuffle_elig else ("MEMORYLESS" if args.memoryless else ("FROZEN" if args.frozen else "MAIN"))
    print(f"\n[RESULT {tag}] on-bridge graded read-out, DELTA-rule learned: grounded next-token acc = {acc:.3f} "
          f"(chance ~1/{nv}={1/nv:.3f}) -- does the on-bridge read-out LEARN the grounded map by a local rule?")


if __name__ == "__main__":
    main()
