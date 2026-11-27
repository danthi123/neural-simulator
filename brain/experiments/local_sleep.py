# Capability: Sleep consolidation
# (by a local plasticity rule instead of backprop: the 12-31 milestone de-risk)
# Record: brain/research/2026-11-27-local-sleep-rule-prereg-DRAFT.md
# Expected-fail: recall below the gradient arm at 0.1B (slow convergence of local rules, old-track KP ppl 63 vs 34); a
#   weak or absent effect through the WKV state (no credit through time here); NOT a verdict on local rules in general.
"""Sleep learning without backprop: a delta rule on the rank-8 channel-mix adapters of PlasticCortex.

Per replayed text: one forward under no_grad records, per layer, the channel-mix input k_in, the hidden pre-activation
and the hidden h. The output error e = softmax - target (replay: one-hot next token; keep: the pretrained cortex's
softmax, whose gradient is exactly the KL's) is sent back by FIXED random matrices only (no weight transport):
head feedback (V,C) to the residual, a per-layer mix (C,C) to the channel-mix output, a per-layer (C,hid) to the hidden
units, gated by the unit's own derivative (2 relu(pre)). Each factor then changes by pre-synaptic activity x
post-synaptic error (dB = a^T e, dA = x^T (e Fb)), the A updates read the error through feedback copies of B (Kolen-
Pollack: the copy gets B's own update and decay, so it aligns to B; "fa": the copy stays random). Keep term: a local
decay of B each step, plus the keep-error step on ordinary text. No credit through time (the WKV state is a constant
here): a declared limit. Scaffolds: text replay and the keep bank as in brain/sleep.py; update-norm clip = divisive
normalisation stand-in. Nothing frozen is patched: LocalCortex subclasses PlasticCortex and records in _cmix.

  python -m brain.experiments.local_sleep --model <path.pth> --seed 42 --steps 240 --out <json>
  dev sweep (seed 7, 0.1B, one config per process): bash brain/experiments/results/2026-11-27-local-sleep/dev_sweep.sh
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from brain.plastic_cortex import PlasticCortex


class LocalCortex(PlasticCortex):
    """PlasticCortex that can record each channel-mix's (k_in, pre, h) when `self.rec` is a list."""
    rec = None

    def _cmix(self, i, x, x_prev):
        s = str(i)
        if self.rec is None or not (self.enabled and s in self.A_k):
            return super()._cmix(i, x, x_prev)
        assert not torch.is_grad_enabled(), "local sleep forward must run under no_grad"
        f = f"blocks.{i}.ffn."
        k_in = x + (torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x) * self.z[f + "x_k"]
        pre = k_in @ self.z[f + "key.weight"] + (k_in @ self.A_k[s]) @ self.B_k[s]
        h = torch.relu(pre) ** 2
        self.rec.append((i, k_in, pre, h))
        return h @ self.z[f + "value.weight"] + (h @ self.A_v[s]) @ self.B_v[s], x[-1, :]


def adopt(plastic) -> LocalCortex:
    """A LocalCortex sharing the adapters (Parameters) of an existing PlasticCortex; the original is untouched."""
    lc = LocalCortex.__new__(LocalCortex)
    lc.__dict__.update(plastic.__dict__)
    return lc


PRESETS = {"random": dict(head="random", mix="random", hid="random"),         # strict: nothing transported
           "mixed": dict(head="transport", mix="identity", hid="random"),      # readout + residual carry the error
           "transport": dict(head="transport", mix="identity", hid="transport")}


class Feedback:
    """Fixed random feedback pathways (seeded, no weight transport) + the Kolen-Pollack copies of B."""

    def __init__(self, pc, mode="kp", seed=0, head="random", mix="random", hid="random"):
        assert mode in ("kp", "fa") and head in ("random", "transport") and mix in ("random", "identity") and hid in ("random", "transport")
        g = torch.Generator().manual_seed(seed)
        R = lambda *s: torch.randn(*s, generator=g)
        V = pc.z["head.weight"].shape[1]
        hdim, r = pc.B_k[str(pc.layers[0])].shape[1], pc.B_k[str(pc.layers[0])].shape[0]
        dev = pc.z["emb.weight"].device
        self.mode = mode
        # head="transport": the frozen readout's own transpose carries the output error (reciprocal connections, a
        # declared weight-transport scaffold; the readout is never trained, so a KP copy could not align to it).
        self.head = (R(V, pc.C) / V ** 0.5).to(dev) if head == "random" else pc.z["head.weight"].T.clone() * (pc.C / V) ** 0.5
        # mix="identity": the residual stream carries the error as it is (it adds, so its Jacobian is the identity).
        self.mix = {i: torch.eye(pc.C, device=dev) if mix == "identity" else (R(pc.C, pc.C) / pc.C ** 0.5).to(dev)
                    for i in pc.layers}
        self.hid = {i: (R(pc.C, hdim) / pc.C ** 0.5).to(dev) if hid == "random" else pc.z[f"blocks.{i}.ffn.value.weight"].T.clone()
                    for i in pc.layers}   # hid="transport": the cmix value weights' transpose (declared scaffold)
        self.v = {i: (R(pc.C, r) / pc.C ** 0.5).to(dev) for i in pc.layers}     # ~ B_v^T
        self.k = {i: (R(hdim, r) / hdim ** 0.5).to(dev) for i in pc.layers}       # ~ B_k^T


def local_update(pc, fb, err, *, lr, decay=0.0, clip=1.0, state=None):
    """One delta-rule step on the adapters of `pc` (a LocalCortex), from per-position logit errors `err` (T,V) that the
    caller computed from this same forward (see replay_error). No autograd anywhere."""
    with torch.no_grad():
        T = err.shape[0]
        e_res = err @ fb.head
        for i, k_in, pre, h in pc.rec:
            s = str(i)
            e_l = e_res @ fb.mix[i]                          # error at the channel-mix output
            e_h = (e_l @ fb.hid[i]) * 2 * torch.relu(pre)    # error at the hidden units, own derivative
            a_v, a_k = h @ pc.A_v[s], k_in @ pc.A_k[s]       # rank-space pre-synaptic activity
            dBv, dBk = a_v.T @ e_l / T, a_k.T @ e_h / T
            dAv, dAk = h.T @ (e_l @ fb.v[i]) / T, k_in.T @ (e_h @ fb.k[i]) / T
            for p, d in ((pc.B_v[s], dBv), (pc.B_k[s], dBk), (pc.A_v[s], dAv), (pc.A_k[s], dAk)):
                n = float(d.norm())
                d.mul_(min(1.0, clip / (n + 1e-12)) if clip else 1.0)
                p.sub_(lr * d)
            if fb.mode == "kp":     # the feedback copy gets B's own update
                fb.v[i].sub_(lr * dBv.T)
                fb.k[i].sub_(lr * dBk.T)
            if decay:
                for p in (pc.B_v[s], pc.B_k[s]):
                    p.mul_(1 - decay)
                if fb.mode == "kp":
                    fb.v[i].mul_(1 - decay)
                    fb.k[i].mul_(1 - decay)


def _forward(pc, ids, state):
    pc.rec = []
    with torch.no_grad():
        logits, _ = pc(ids, state)
    return logits


def local_sleep(plastic, base_state, episodes, generic, *, steps=200, lr=0.05, kl_weight=1.0, decay=1e-3, seed=0,
                mode="kp", clip=1.0, feedback="random", log=None):
    """Same interface idea as brain.sleep.sleep (episodes: (text, strength); generic: texts), local rule. Returns
    per-step (replay CE, keep KL). The adapters of `plastic` are changed in place."""
    pc = adopt(plastic)
    rng = random.Random(seed)
    fb = Feedback(pc, mode, seed, **PRESETS[feedback])
    ids_of = pc.cortex.pipe.encode
    weights = [max(s, 1e-3) for _, s in episodes]
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    pc.enabled = False
    with torch.no_grad():
        ref = [F.softmax(pc(g, base_state)[0], -1) for g in gen_ids]
    pc.enabled = True
    losses = []
    for step in range(steps):
        text, _ = rng.choices(episodes, weights=weights)[0]
        ids = ids_of(text)
        logits = _forward(pc, ids, base_state)
        p = F.softmax(logits, -1)
        tgt = F.one_hot(torch.tensor(ids[1:], device=p.device), p.shape[1]).to(p.dtype)
        err = torch.cat((p[:-1] - tgt, torch.zeros_like(p[:1])))          # last position has no target
        ce = float(F.cross_entropy(logits[:-1], torch.tensor(ids[1:], device=p.device)))
        local_update(pc, fb, err, lr=lr, decay=decay, clip=clip)
        j = rng.randrange(len(gen_ids))
        gl = _forward(pc, gen_ids[j], base_state)
        gp = F.softmax(gl, -1)
        local_update(pc, fb, kl_weight * (gp - ref[j]), lr=lr, clip=clip)
        kl = float(F.kl_div(F.log_softmax(gl, -1), ref[j], reduction="batchmean"))
        losses.append((ce, kl))
        if log and step % 20 == 0:
            log(f"local sleep step {step}: replay {ce:.3f} keep {kl:.4f}")
    pc.rec = None
    return losses


def main():
    from brain.brain import BIRTH
    from brain.cortex import Cortex, DEFAULT_MODEL
    from brain.experiments.facts import FACTS
    from brain.experiments.sleep_probe import GENERIC, build_arm_episodes, evaluate
    from brain.sleep import sleep
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=0.05, help="local rule step")
    ap.add_argument("--grad-lr", type=float, default=5e-4, help="gradient reference (as sleep_probe)")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=1e-3)
    ap.add_argument("--mode", default="kp", choices=["kp", "fa"])
    ap.add_argument("--feedback", default="mixed", choices=list(PRESETS), help="feedback preset of the local arm")
    ap.add_argument("--arms", default="all", help="comma list of: no_sleep,gradient,local,local_shuffled,local_random")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    eps = build_arm_episodes(a.seed)
    arms = {"no_sleep": ("none", None), "gradient": ("grad", eps["sleep_true"]), "local": ("local", eps["sleep_true"]),
            "local_shuffled": ("local", eps["sleep_shuffled"]),
            "local_random": ("local_random", eps["sleep_true"])}
    want = list(arms) if a.arms == "all" else [x.strip() for x in a.arms.split(",") if x.strip()]
    if [x for x in want if x not in arms]:
        raise SystemExit(f"unknown arms; choose from {list(arms)}")
    keys_all = [k for _, _, k in FACTS]
    res = {"model": a.model, "seed": a.seed, "steps": a.steps, "lr": a.lr, "mode": a.mode, "feedback": a.feedback, "decay": a.decay, "arms": {}}
    for arm in want:
        kind, e = arms[arm]
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        losses = []
        lg = lambda m: print(arm, m, flush=True)
        if kind == "grad":
            losses = sleep(pc, base, e, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight, seed=a.seed, log=lg)
        elif kind.startswith("local"):
            losses = local_sleep(pc, base, e, GENERIC, steps=a.steps, lr=a.lr, kl_weight=a.kl_weight, decay=a.decay,
                                 seed=a.seed, mode=a.mode, feedback="random" if kind == "local_random" else a.feedback, log=lg)
        r = evaluate(pc, base, keys_all)
        r["final_replay_loss"] = losses[-1][0] if losses else None
        r["adapter_norm"] = pc.adapter_norm()
        res["arms"][arm] = r
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    res["elapsed_s"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
