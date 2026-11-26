# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-26-routing-screen3-prereg.md (part J); Anthropic 2026, "Verbalizable Representations
#   Form a Global Workspace in Language Models" (transformer-circuits.pub/2026/workspace, the Jacobian lens)
# Expected-fail: partly. The J-lens was shown on transformers; on a recurrent RWKV-7 cortex at 1.5B it may not carry
#   concepts, and near-duplicate separation is MISSING (PREREQUISITES.md). A measurement tool, not a brain default.
"""Jacobian lens for the RWKV-7 cortex: per-layer mean Jacobians J_l = E[dh_L,t' / dh_l,t] (t <= t'), computed through
PlasticCortex's differentiable forward, and the residual stream read for keys. Host-side measurement only.

h_l = the residual after block l (l = 0..L-1); h_L = the residual after the last block, before ln_out (= h_{L-1}, so
J_{L-1} is the identity). For one prompt of T tokens, row j of J_l is sum_t d(sum_t' h_L,t'[j]) / dh_l,t divided by T:
the sum over every pair t <= t' (causality zeroes t' < t) over the source positions, which is the pair mean up to one
constant per prompt length and makes J_{L-1} exactly I. Prompts are averaged (all the same length here).

    python -m brain.experiments.jlens compute --model 1.5b --device cuda --tokens 64 --layers 12 18 23 --out J.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

import torch
import torch.nn.functional as F


def residuals(p, tokens, state=None, grad_input: bool = False):
    """PlasticCortex.forward's math, returning the residual after every block: list of L tensors (T, C), the input
    embedding (requires grad when grad_input) and the final state. Same ops as PlasticCortex.forward (adapters as set)."""
    z = p.z
    state = [s.clone() for s in (state if state is not None else p.zero_state())]
    x = z["emb.weight"][tokens]
    if grad_input:
        x = x.detach().requires_grad_(True)
    x0 = x
    v_first = torch.empty_like(x)
    hs = []
    for i in range(p.L):
        b = f"blocks.{i}."
        xx = F.layer_norm(x, (p.C,), weight=z[b + "ln1.weight"], bias=z[b + "ln1.bias"])
        xx, state[i * 3], state[i * 3 + 1], v_first = p._tmix(i, xx, state[i * 3], v_first, state[i * 3 + 1])
        x = x + xx
        xx = F.layer_norm(x, (p.C,), weight=z[b + "ln2.weight"], bias=z[b + "ln2.bias"])
        xx, state[i * 3 + 2] = p._cmix(i, xx, state[i * 3 + 2])
        x = x + xx
        hs.append(x)
    return hs, x0, state


def readout(p, h):
    """The unembedding read of a residual (as PlasticCortex.forward ends): logits (..., vocab)."""
    z = p.z
    return F.layer_norm(h, (p.C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]) @ z["head.weight"]


def jacobians_one(p, tokens, layers, chunk: int = 16, batched: bool = True) -> dict:
    """{l: (C, C)} mean Jacobian of h_L w.r.t. h_l for one prompt (rows = output dims)."""
    C = p.C
    hs, _, _ = residuals(p, tokens, grad_input=True)
    hL = hs[-1]
    T = hL.shape[0]
    ins = [hs[l] for l in layers]
    out = {l: torch.zeros(C, C, dtype=torch.float32, device=hL.device) for l in layers}
    eye = torch.eye(C, dtype=hL.dtype, device=hL.device)
    for j0 in range(0, C, chunk):
        U = eye[j0:j0 + chunk]                                      # (k, C)
        G = U[:, None, :].expand(U.shape[0], T, C).contiguous()     # e_j at every position t'
        if batched:
            gs = torch.autograd.grad(hL, ins, grad_outputs=G, retain_graph=True, is_grads_batched=True,
                                     allow_unused=True)
        else:
            per = [torch.autograd.grad(hL, ins, grad_outputs=G[k], retain_graph=True, allow_unused=True)
                   for k in range(G.shape[0])]
            gs = [torch.stack([q[n] if q[n] is not None else torch.zeros_like(ins[n]) for q in per])
                  for n in range(len(ins))]
        for l, g in zip(layers, gs):
            if g is None:
                continue
            out[l][j0:j0 + U.shape[0]] = (g.sum(1) / T).float()
    return out


def compute(p, prompts, layers, chunk=16, batched=True, log=print) -> dict:
    acc = {l: None for l in layers}
    t0 = time.time()
    for n, toks in enumerate(prompts):
        J = jacobians_one(p, toks, layers, chunk, batched)
        for l in layers:
            acc[l] = J[l] if acc[l] is None else acc[l] + J[l]
        log(f"prompt {n + 1}/{len(prompts)} ({len(toks)} tokens) {time.time() - t0:.0f}s")
    return {l: (acc[l] / len(prompts)).cpu() for l in layers}


def heldout_prompts(pipe, n_tokens: int) -> list:
    from brain.eval import life_heldout
    life_heldout.verify()
    return [pipe.encode(t)[:n_tokens] for t in life_heldout.HELDOUT]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--model", default="1.5b")
    c.add_argument("--device", default="cpu")
    c.add_argument("--tokens", type=int, default=64)
    c.add_argument("--layers", type=int, nargs="+", default=[12, 18])
    c.add_argument("--chunk", type=int, default=16)
    c.add_argument("--unbatched", action="store_true")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from brain.cortex import Cortex, resolve_model
    from brain.plastic_cortex import PlasticCortex
    cx = Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision="fp32")
    p = PlasticCortex(cx, dtype=torch.float32)
    p.enabled = False                                   # pretrained cortex: adapters off
    layers = sorted(set(a.layers) | {p.L - 1})          # the last layer is the identity check
    prompts = heldout_prompts(cx.pipe, a.tokens)
    if a.device == "cuda":
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, "TF32 must be off"
    with torch.no_grad():                               # review 11-26: the residual path is PlasticCortex.forward
        lf, _ = p(prompts[0], None)
        hs, _, _ = residuals(p, prompts[0])
        fwd_err = float((readout(p, hs[-1]) - lf).abs().max())
    assert fwd_err < 1e-4 * max(1.0, float(lf.abs().max())), f"residual path != PlasticCortex.forward ({fwd_err})"
    batched = not a.unbatched
    t0 = time.time()                                    # time one prompt each way on a few chunks, keep the faster
    if batched:
        try:
            jacobians_one(p, prompts[0][:8], layers, a.chunk, True)
            tb = time.time() - t0
            t1 = time.time()
            jacobians_one(p, prompts[0][:8], layers, a.chunk, False)
            batched = tb <= time.time() - t1
        except RuntimeError as e:
            print(f"batched VJP failed ({e}); using the looped VJP")
            batched = False
    print(f"forward check err {fwd_err:.2e}; VJP {'batched' if batched else 'looped'}")
    J = compute(p, prompts, layers, a.chunk, batched)
    eye_err = float((J[p.L - 1] - torch.eye(p.C)).abs().max())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    commit = commit or os.environ.get("BRAIN_CODE_COMMIT")
    meta = {"forward_check_err": fwd_err, "vjp": "batched" if batched else "looped", "model": a.model, "device": a.device, "tokens": a.tokens, "layers": layers, "n_prompts": len(prompts),
            "identity_max_err_last_layer": eye_err, "code_commit": commit, "C": p.C, "L": p.L}
    torch.save({"J": J, "meta": meta}, a.out)
    with open(a.out, "rb") as fh:
        meta["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    with open(a.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
