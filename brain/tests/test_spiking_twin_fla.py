# Capability: Language
# Record: brain/research/2026-11-27-spiking-twin-e1-prereg.md (section 7 item 1, kernel addendum)
# Expected-fail: none (a kernel parity check, not a capability claim)
"""fla training kernel (fla-core 0.6.0 chunk_rwkv7) for the spiking twin's WKV recurrence.

CPU (always): the variable mapping (brain.spiking_twin.wkv_fla_args) is exact. Our per-token loop
(SpikingTwin._wkv_loop) equals (a) a pure-torch transcription of fla's rwkv7 kernel recurrence
(fla/ops/rwkv7/fused_recurrent.py 112-117) fed through the mapping, and (b) fla's own CPU reference
fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence (skipped if fla is not on PYTHONPATH), outputs, final state
and gradients, with a non-zero initial state. The switch defaults to the loop, and wkv="fla" refuses CPU tensors.

GPU (skipped unless BRAIN_ALLOW_W3090=1, CUDA is present and nvidia-smi shows < 2 GB used; with BRAIN_REQUIRE_GPU=1 an
unmet condition FAILS instead, so the launch script's parity stage cannot pass on a skip): the fla path against the loop
path on (1) a tiny random cortex (head size 32, two layers, carried state) and (2) the real 0.1B teacher
(rwkv7-g1d-0.1b), spikes off and spikes on, fp32 and bf16 AMP, trainable twin with nonzero slow-store adapters:
logits, final state, adapter gradients and the weight/theta gradients, each gradient error taken per parameter (worst
parameter, and separately the kernel-only att decay / a / kk group, whose gradients come only through fla's backward).
Each fla case counts its _wkv_fla calls (a silent fall-through to the loop fails) and checks the fp32 logits are not
bit-equal to the loop. Also: T = 512 with per-block checkpointing (the training shape) and T = 50 (ragged last chunk).
The gate is evaluated once per session (nvidia-smi would count this process's own tensors after the first case); CPU
tests never import the fla package (its import probes CUDA), they load fla's naive reference by file path.

Parity bar (registered in the prereg section 7 item 1). err(x, ref) = ||x - ref|| / ||ref|| (relative L2):
  fp32, spikes off: logits <= 2e-3, state <= 2e-3, gradients <= 1e-2, mean token KL(loop || fla) <= 1e-4 nats.
  fp32, spikes on:  logits <= 1e-2, gradients <= 5e-2, KL <= 1e-3 (a 1e-6 change can flip an integer count).
  bf16 AMP: the fla path's error to the fp32 loop must be <= max(the fp32 bar, 2 x the bf16 loop's own error to the
  fp32 loop), i.e. the kernel may not add more than the precision the loop already loses at bf16.
The fp32 bars allow TF32 inside fla's Triton dots (about 1e-3 relative); the bf16 bar is relative to the loop's floor.
With BRAIN_PARITY_OUT=<dir> every GPU case writes its measured errors to <dir>/<case>.json."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

from brain.spiking_twin import FLA_VERSION, LOG_DECAY_MIN, SpikingTwin, fla_version, wkv_fla_args
from brain.tests.test_source_tag import _Cortex, _Model

BARS = {("fp32", False): dict(logits=2e-3, state=2e-3, grad=1e-2, kl=1e-4),
        ("fp32", True): dict(logits=1e-2, state=1e-2, grad=5e-2, kl=1e-3)}


# The loop computes in fp32 (it casts state and inputs with .float()); the references below run in fp64 on the same
# inputs, so "exact" means fp32 rounding: relative L2 error below 1e-6 (fp32 epsilon is 6e-8). A wrong sign, a missing
# transpose, a log-vs-linear decay or a scale of K**-0.5 gives errors of order 1e-2 to 1.
FP32_EXACT = 1e-6


def rel(x, ref):
    x, ref = x.detach().double(), ref.detach().double()
    return float((x - ref).norm() / ref.norm().clamp_min(1e-30))


# ---------------------------------------------------------------------------------------------------- CPU mapping
def _fla_root():
    """fla's package directory, found without importing it (find_spec of a top-level package runs no code), or skip."""
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("fla-core not on PYTHONPATH")
    return list(spec.submodule_search_locations)[0]


def _load_naive_dplr():
    """fla's pure-torch reference dplr_recurrence (generalized_delta_rule/dplr/naive.py, needs only torch + einops),
    loaded by file path so fla/__init__ (which initialises CUDA) never runs in a CPU test."""
    pytest.importorskip("einops", reason="einops (fla's dependency) not on PYTHONPATH")
    path = os.path.join(_fla_root(), "ops", "generalized_delta_rule", "dplr", "naive.py")
    spec = importlib.util.spec_from_file_location("_fla_dplr_naive_by_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dplr_recurrence


def _rand(B=2, T=9, H=2, N=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    R = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    C = H * N
    r, k, v, wpre, a_pre = R(B, T, C), R(B, T, C), R(B, T, C), R(B, T, C) * 2, R(B, T, C)
    kk = torch.nn.functional.normalize(R(B, T, H, N), dim=-1).view(B, T, C)
    a = torch.sigmoid(a_pre)
    S0 = R(B, H, N, N) * 0.3
    return r, wpre, k, v, kk, a, S0


def _loop(r, wpre, k, v, kk, a, S0):
    """Our recurrence exactly as the twin runs it (SpikingTwin._wkv_loop), with w0 = 0 and w = wpre."""
    B, T, C = r.shape
    H, N = S0.shape[1], S0.shape[-1]
    tw = SpikingTwin.__new__(SpikingTwin)
    z = {"x.w0": torch.zeros(C, dtype=r.dtype)}
    x = torch.zeros(B, T, C, dtype=r.dtype)
    y, S = tw._wkv_loop(B, T, H, N, r, wpre, k, v, kk, a, S0, z, "x.", x)
    return y, S


def _fla_kernel_formula(r, w, k, v, a, b, scale, initial_state, **_):
    """fla/ops/rwkv7/fused_recurrent.py 99-101 / 112-117 and generalized_delta_rule/dplr/fused_recurrent.py 94-96, in
    torch: h[K, V] <- exp(w)[:, None] * h + b[:, None] * sum_K(a[:, None] * h) + k[:, None] * v[None, :];
    o = sum_K(h * (scale * r)[:, None]). Layout [B, T, H, K]."""
    B, T, H, K = r.shape
    h = initial_state.clone()
    outs = []
    for t in range(T):
        sa = (a[:, t, :, :, None] * h).sum(-2, keepdim=True)            # (B, H, 1, V)
        h = torch.exp(w[:, t])[..., None] * h + b[:, t, :, :, None] * sa + k[:, t, :, :, None] * v[:, t, :, None, :]
        outs.append((h * (scale * r[:, t])[..., None]).sum(-2))         # (B, H, V)
    return torch.stack(outs, 1), h


def _logw(wpre):
    """gk exactly as SpikingTwin._tmix_b hands it to the kernel."""
    return -0.606531 * torch.sigmoid(wpre)


def test_mapping_matches_fla_kernel_formula_outputs_state_and_grads():
    ins = [t.requires_grad_(True) for t in _rand()]
    y, S = _loop(*ins)
    loss = (y * torch.linspace(-1, 1, y.numel(), dtype=y.dtype).view_as(y)).sum() + (S ** 2).sum()
    g_loop = torch.autograd.grad(loss, ins)
    r, wpre, k, v, kk, a, S0 = ins
    kw = wkv_fla_args(r, _logw(wpre), k, v, kk, a, S0)
    assert kw["scale"] == 1.0 and kw["output_final_state"] is True
    o, ht = _fla_kernel_formula(**kw)
    y2, S2 = o.reshape(y.shape), ht.transpose(-1, -2)
    assert rel(y2, y) < FP32_EXACT and rel(S2, S) < FP32_EXACT
    loss2 = (y2 * torch.linspace(-1, 1, y.numel(), dtype=y.dtype).view_as(y)).sum() + (S2 ** 2).sum()
    for g1, g2 in zip(g_loop, torch.autograd.grad(loss2, ins)):
        assert rel(g2, g1) < FP32_EXACT


def test_mapping_matches_fla_naive_dplr_reference():
    dplr_recurrence = _load_naive_dplr()                                  # fla's own pure-torch reference (CPU)
    ins = _rand(seed=1)
    y, S = _loop(*ins)
    r, wpre, k, v, kk, a, S0 = ins
    kw = wkv_fla_args(r, _logw(wpre), k, v, kk, a, S0)
    K = r.shape[-1] // S0.shape[1]
    hf = lambda t: t.transpose(1, 2).float()                             # [B, T, H, D] -> [B, H, T, D]; fp32
    # (dplr_recurrence casts q, k, v, beta, gk to fp32 but not alpha: naive.py 19)
    # dplr_recurrence hard-codes q * K**-0.5 (naive.py 23); chunk_rwkv7 passes scale=1, so undo it on q.
    o, ht = dplr_recurrence(hf(kw["r"]) * kw["scale"] * K ** 0.5, hf(kw["k"]), hf(kw["v"]), hf(kw["a"]),
                            hf(kw["b"]), hf(kw["w"]), initial_state=kw["initial_state"].float())
    assert rel(o.transpose(1, 2).reshape(y.shape), y) < FP32_EXACT
    assert rel(ht.transpose(-1, -2), S) < FP32_EXACT


def test_log_decay_is_in_fla_range():
    wpre = torch.linspace(-30, 30, 1001)
    lw = _logw(wpre)
    assert float(lw.min()) >= LOG_DECAY_MIN and float(lw.max()) < 0
    assert torch.allclose(lw.exp(), torch.exp(-0.606531 * torch.sigmoid(wpre)))


def test_switch_defaults_to_loop_and_fla_refuses_cpu():
    c = _Cortex()
    assert SpikingTwin(c).wkv == "loop"
    with pytest.raises(ValueError):
        SpikingTwin(c, wkv="cuda")
    _fla_root()
    had_fla = "fla" in sys.modules
    tw = SpikingTwin(c, wkv="fla", spikes=False)   # checks the installed version from metadata, imports nothing
    with pytest.raises(RuntimeError, match="CUDA only"):
        tw.forward_batch(torch.tensor([[1, 2, 3]]))
    with torch.no_grad():                    # single-token steps take the loop by design (sampling), on any device
        tw.forward_batch(torch.tensor([[1]]))
    if not had_fla:                          # nothing on the CPU path imports fla (whose import touches CUDA)
        assert "fla" not in sys.modules


def test_fla_version_is_checked_from_metadata():
    _fla_root()
    assert fla_version() == FLA_VERSION
    import importlib.metadata as md
    real = md.version
    try:
        md.version = lambda name: "0.0.1" if name == "fla-core" else real(name)
        with pytest.raises(ImportError, match="validated against"):
            fla_version()
    finally:
        md.version = real


# ---------------------------------------------------------------------------------------------------- GPU parity
_GATE = {}      # evaluated once per session: nvidia-smi counts this process's own tensors after the first case


@pytest.fixture(autouse=True)
def _free_cuda():
    yield
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _gpu_gate():
    if "why" not in _GATE:
        _GATE["why"] = _gpu_gate_why()
    why = _GATE["why"]
    if why:
        if os.environ.get("BRAIN_REQUIRE_GPU") == "1":
            pytest.fail(f"parity required but cannot run: {why}")
        pytest.skip(why)
    pytest.importorskip("fla", reason="fla-core not on PYTHONPATH")


def _gpu_gate_why():
    why = None
    if os.environ.get("BRAIN_ALLOW_W3090") != "1":
        why = "BRAIN_ALLOW_W3090 != 1"
    elif not torch.cuda.is_available():
        why = "no CUDA"
    else:
        try:
            used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=30).stdout.split()[0])
        except Exception as e:  # pragma: no cover - environment
            used, why = None, f"nvidia-smi failed: {e}"
        if used is not None and used >= 2048:
            why = f"GPU busy: {used} MiB used (>= 2 GB)"
    return why


def _twin(cortex, wkv, spikes, calib, seed=0, ckpt=False):
    torch.manual_seed(seed)
    tw = SpikingTwin(cortex, spikes=spikes, trainable=True, wkv=wkv, rank=4, ckpt=ckpt)
    with torch.no_grad():
        for d in (tw.B_k, tw.B_v):
            for p in d.values():
                p.normal_(0, 0.05, generator=torch.Generator(device=p.device).manual_seed(11))
    if calib is not None:
        tw.calibrate([calib])
    return tw


def _run(tw, toks, amp, wkv_dtype):
    tw.wkv_dtype = wkv_dtype
    for p in tw.parameters():
        p.grad = None
    half = toks.shape[1] // 2
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        l1, st = tw.forward_batch(toks[:, :half])                        # carried state across two calls
        l2, st = tw.forward_batch(toks[:, half:], st)
        logits = torch.cat([l1, l2], 1).float()
        lp = torch.log_softmax(logits, -1)
        tgt = torch.roll(toks, -1, 1)
        loss = -lp.gather(-1, tgt[..., None]).mean() + 1e-3 * sum(s.float().pow(2).mean() for s in st[1::3])
    loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in tw.named_parameters() if p.grad is not None}
    return logits.detach(), torch.cat([s.detach().float().reshape(-1) for s in st[1::3]]), grads


# parameters whose gradient reaches them only through the kernel's dgk / da / db (decay, in-context rate, kk)
KERNEL_ONLY = ("|att|w0", "|att|w1", "|att|w2", "|att|a0", "|att|a1", "|att|a2", "|att|k_k", "|att|k_a")


def _grad_errs(got, ref):
    """Per-parameter relative L2 (one concatenated norm would let the small kernel-only gradients hide under emb/head):
    worst over adapters, over all other parameters, and over the kernel-only group, with the worst parameter's name."""
    ad = [k for k in ref if k.startswith(("A_", "B_"))]
    per = {k: rel(got[k], ref[k]) for k in ref if float(ref[k].norm()) > 0}
    wt = {k: v for k, v in per.items() if k not in ad}
    kern = {k: v for k, v in wt.items() if k.endswith(KERNEL_ONLY)}
    assert kern, "no kernel-only (att decay / a / kk) gradients found"
    worst = max(wt, key=wt.get)
    return {"grad_adapters": max(per[k] for k in ad if k in per), "grad_weights": wt[worst],
            "grad_att_kernel": max(kern.values())}, {"grad_weights_worst": worst,
                                                     "grad_att_kernel_worst": max(kern, key=kern.get)}


def _count_fla(tw):
    """Wrap tw._wkv_fla with a call counter (so a silent fall-through to the loop cannot pass as parity)."""
    calls, orig = [], tw._wkv_fla

    def wrapped(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    tw._wkv_fla = wrapped
    return calls


def _compare(case, cortex, toks, spikes, ckpt=False):
    ref_tw = _twin(cortex, "loop", spikes, toks[:1], ckpt=ckpt)
    fla_tw = _twin(cortex, "fla", spikes, None, ckpt=ckpt)
    calls = _count_fla(fla_tw)
    # per _run: two forward_batch calls x L layers, plus the same again when checkpointing recomputes in backward
    per_run = 2 * fla_tw.L * (2 if ckpt else 1)
    # identical starting point: the fla twin takes the loop twin's calibrated thetas
    with torch.no_grad():
        for k in fla_tw.theta:
            fla_tw.theta[k].copy_(ref_tw.theta[k])
    ref = _run(ref_tw, toks, False, torch.float32)
    out = {"fla_calls_per_run": per_run}
    for prec, amp in (("fp32", False), ("bf16", True)):
        res = {}
        loop_p = _run(ref_tw, toks, amp, torch.float32) if amp else ref
        for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            if prec == "fp32" and dt_name == "bf16":
                continue
            n0 = len(calls)
            got = _run(fla_tw, toks, amp, dt)
            assert len(calls) - n0 == per_run, f"{case}: fla ran {len(calls) - n0} times, expected {per_run}"
            if prec == "fp32":
                assert not torch.equal(got[0], ref[0]), f"{case}: fla logits bit-equal to the loop (kernel not used?)"
            kl = float((torch.softmax(ref[0], -1) * (torch.log_softmax(ref[0], -1) - torch.log_softmax(got[0], -1)))
                       .sum(-1).mean())
            ge, worst = _grad_errs(got[2], ref[2])
            res[dt_name] = {"logits": rel(got[0], ref[0]), "state": rel(got[1], ref[1]), "kl": kl, **ge}
            out.setdefault("worst_params", {})[f"{prec}/{dt_name}"] = worst
        floor = None
        if amp:
            kl0 = float((torch.softmax(ref[0], -1)
                         * (torch.log_softmax(ref[0], -1) - torch.log_softmax(loop_p[0], -1))).sum(-1).mean())
            floor = {"logits": rel(loop_p[0], ref[0]), "state": rel(loop_p[1], ref[1]), "kl": kl0,
                     **_grad_errs(loop_p[2], ref[2])[0]}
        out[prec] = {"fla": res, "loop_floor": floor}
    if os.environ.get("BRAIN_PARITY_OUT"):
        os.makedirs(os.environ["BRAIN_PARITY_OUT"], exist_ok=True)
        with open(os.path.join(os.environ["BRAIN_PARITY_OUT"], f"{case}.json"), "w") as f:
            json.dump(out, f, indent=1)
    bar = BARS[("fp32", spikes)]
    lim = {"logits": bar["logits"], "state": bar["state"], "kl": bar["kl"], "grad_adapters": bar["grad"],
           "grad_weights": bar["grad"], "grad_att_kernel": bar["grad"]}
    fails = []
    for m, v in out["fp32"]["fla"]["fp32"].items():
        if not v <= lim[m]:
            fails.append(f"fp32 {m} {v:.3g} > {lim[m]:.3g}")
    for dt_name, r in out["bf16"]["fla"].items():
        for m, v in r.items():
            b = max(lim[m], 2 * out["bf16"]["loop_floor"][m])
            if not v <= b:
                fails.append(f"bf16-amp kernel-{dt_name} {m} {v:.3g} > {b:.3g}")
    assert not fails, f"{case}: " + "; ".join(fails) + f"\n{json.dumps(out, indent=1)}"


class _CudaCortex:
    def __init__(self, model):
        from brain.tests.test_brain_key_switch import _Pipe
        self.pipe, self.model = _Pipe(), model


@pytest.mark.parametrize("spikes", [False, True], ids=["spikes_off", "spikes_on"])
def test_gpu_parity_tiny_random(spikes):
    _gpu_gate()
    m = _Model(C=64, H=2, L=2, hid=128, V=256, r=8, seed=5)
    m.z = {k: v.cuda() for k, v in m.z.items()}
    g = torch.Generator().manual_seed(3)
    toks = torch.randint(0, 256, (3, 96), generator=g).cuda()
    _compare(f"tiny_{'on' if spikes else 'off'}", _CudaCortex(m), toks, spikes)


@pytest.mark.parametrize("T,ckpt", [(50, False), (512, True)], ids=["T50_ragged", "T512_ckpt"])
def test_gpu_parity_tiny_training_shapes(T, ckpt):
    """The training shape (T = 512 with per-block checkpointing, as train() runs it) and a T that is not a multiple of
    the 16-token chunk (two calls of 25: a ragged last chunk in each), spikes on."""
    _gpu_gate()
    m = _Model(C=64, H=2, L=2, hid=128, V=256, r=8, seed=7)
    m.z = {k: v.cuda() for k, v in m.z.items()}
    toks = torch.randint(0, 256, (2, T), generator=torch.Generator().manual_seed(8)).cuda()
    _compare(f"tiny_T{T}{'_ckpt' if ckpt else ''}", _CudaCortex(m), toks, True, ckpt=ckpt)


def test_gpu_tiny_lower_bound_option_matches_default():
    _gpu_gate()
    m = _Model(C=64, H=2, L=2, hid=128, V=256, r=8, seed=6)
    m.z = {k: v.cuda() for k, v in m.z.items()}
    toks = torch.randint(0, 256, (2, 128), generator=torch.Generator().manual_seed(4)).cuda()
    a = SpikingTwin(_CudaCortex(m), spikes=False, wkv="fla")
    b = SpikingTwin(_CudaCortex(m), spikes=False, wkv="fla", wkv_lower_bound=True)
    ref = SpikingTwin(_CudaCortex(m), spikes=False)
    ca, cb = _count_fla(a), _count_fla(b)
    with torch.no_grad():
        la, lb, lr = a.forward_batch(toks)[0], b.forward_batch(toks)[0], ref.forward_batch(toks)[0]
    assert len(ca) == len(cb) == a.L
    assert not torch.equal(la, lr) and not torch.equal(lb, lr)
    assert rel(la, lr) <= BARS[("fp32", False)]["logits"]
    assert rel(lb, lr) <= BARS[("fp32", False)]["logits"]


@pytest.mark.parametrize("spikes", [False, True], ids=["spikes_off", "spikes_on"])
def test_gpu_parity_real_teacher(spikes):
    _gpu_gate()
    from brain.experiments.spiking_twin_e1 import load_teacher
    c = load_teacher("cuda")
    ids = c.pipe.encode("The history of the river valley begins with the first farmers, who settled along the banks "
                        "and learned to read the floods. User: what is my cat called?\n\nAssistant: Your cat is "
                        "called Pixel, and she likes the window seat in the morning sun. " * 8)
    toks = torch.tensor([ids[:256], ids[64:320]], device="cuda")
    _compare(f"teacher_{'on' if spikes else 'off'}", c, toks, spikes)


def test_e1_train_flag_defaults_to_loop_and_fla_refuses_cpu(tmp_path):
    import numpy as np
    from brain.experiments import spiking_twin_e1 as E
    import inspect
    assert inspect.signature(E.train).parameters["wkv"].default == "loop"
    pools = (np.arange(4096) % 100, np.arange(4096) % 100)
    with pytest.raises(SystemExit, match="CUDA only"):
        E.train("a", 7, str(tmp_path), device="cpu", cortex=_Cortex(), pools=pools, max_steps=1, wkv="fla")
