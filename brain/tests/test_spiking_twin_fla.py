# Capability: Language
# Record: brain/research/2026-11-27-spiking-twin-e1-prereg.md (section 7 item 1, kernel addendum)
# Expected-fail: none (a kernel parity check, not a capability claim)
"""fla training kernel (fla-core 0.6.0 chunk_rwkv7) for the spiking twin's WKV recurrence.

CPU (always): the variable mapping (brain.spiking_twin.wkv_fla_args) is exact. Our per-token loop
(SpikingTwin._wkv_loop) equals (a) a pure-torch transcription of fla's rwkv7 kernel recurrence
(fla/ops/rwkv7/fused_recurrent.py 112-117) fed through the mapping, and (b) fla's own CPU reference
fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence (skipped if fla is not on PYTHONPATH), outputs, final state
and gradients, with a non-zero initial state. The switch defaults to the loop, and wkv="fla" refuses CPU tensors.

GPU (skipped unless BRAIN_ALLOW_W3090=1, CUDA is present and no other python/cuda job holds the GPU and < 6 GB is used;
with BRAIN_REQUIRE_GPU=1 an unmet condition FAILS instead, so the launch script's parity stage cannot pass on a skip).
The gate is evaluated once per session (nvidia-smi would count this process's own tensors after the first case); CPU
tests never import the fla package (its import probes CUDA), they load fla's naive reference by file path.

Parity bar (registered in the prereg section 7 item 1, as amended before data by A1, 2026-11-27).
err(x, ref) = ||x - ref|| / ||ref|| (relative L2); every fla case counts its _wkv_fla calls (a silent fall-through to
the loop fails), including the checkpoint recompute.
Spikes OFF, per logit (tiny random cortex and the 0.1B teacher, trainable twin, nonzero adapters, carried state):
  fp32: logits <= 2e-3, state <= 2e-3, gradients <= 1e-2 (per parameter: worst adapter, worst other weight/theta, and
  the kernel-only att decay / a / kk group), mean token KL(loop || fla) <= 1e-4 nats; fp32 logits not bit-equal.
  bf16 AMP with bf16 kernel inputs (the AMP training config): every metric <= max(the fp32 bar, 2 x the bf16 loop's own
  error to the fp32 loop). AMP with fp32 kernel inputs is measured and recorded, not asserted: it failed the gradient
  bar on the teacher (0.67 > 0.27) and train() refuses it.
Spikes ON: a per-logit bar cannot be met by any implementation (integer counts are discontinuous: the loop's own
  bf16-vs-fp32 error on the teacher is KL 1.59, the same size as fla's 1.49), so it is replaced by:
  (tiny cortex, T = 96, ragged T = 50, T = 512 with checkpointing) the kernel ran (calls) and the graded final state
  is not bit-equal to the loop; when the fp32 logits ARE bit-equal (no count flipped: kernel differences below the
  rounding margin are absorbed by the counts, see test_spike_counts_absorb_small_kernel_differences) the state and
  gradients must meet the spikes-off fp32 bars, since then the kernel is the only difference.
  (0.1B teacher, statistical; fla vs the loop at the same precision; floor F = the loop bf16 AMP vs the loop fp32)
  S1 per-token NLL on held-out WT103 val (8 windows x 512): rms(dNLL) <= max(1.5 F, 1e-3) and |mean dNLL| <=
     max(1.5 F, 0.01 nats); S2 spike counts per channel (mean |count| and firing share, all sites): mean |diff| over
     channels <= max(1.5 F, 1e-4); S3 50 steps of the real train() (arm a, seq 256, micro-batch 4, warmup 10, bf16 AMP,
     fla with bf16 kernel inputs vs the loop): mean |d loss| per step and |d mean loss of the last 10 steps| <=
     max(1.5 F, 1e-5), F from the loop's own bf16 vs fp32 training runs.
The fp32 bars allow TF32 inside fla's Triton dots (about 1e-3 relative); the bf16 bars are relative to the loop's floor.
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

BARS = {("fp32", False): dict(logits=2e-3, state=2e-3, grad=1e-2, kl=1e-4)}   # spikes on: statistical (docstring)
STAT_X = 1.5                       # spikes-on statistical bars: x the loop's own bf16-vs-fp32 floor (amendment A1)
AMP_KERNEL_DTYPE = "bf16"          # the only kernel input dtype train() accepts under bf16 AMP (amendment A1)


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
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout
        jobs = [l for l in apps.splitlines() if ("python" in l.lower() or "cuda" in l.lower()) and str(os.getpid()) not in l]
        if jobs:      # desktop graphics (compositor, chat apps) are not jobs; another python/cuda job is
            why = f"GPU busy: compute job(s) {jobs}"
        elif used is not None and used >= 6144:
            why = f"GPU busy: {used} MiB used (>= 6 GB)"
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
    """Per-logit parity (spikes off) / kernel-ran + no-flip parity (spikes on); see the module docstring."""
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
    out = {"fla_calls_per_run": per_run, "spikes": spikes}
    fails = []
    for prec, amp in (("fp32", False), ("bf16", True)):
        res = {}
        loop_p = _run(ref_tw, toks, amp, torch.float32) if amp else ref
        for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            if prec == "fp32" and dt_name == "bf16":
                continue
            n0 = len(calls)
            got = _run(fla_tw, toks, amp, dt)
            assert len(calls) - n0 == per_run, f"{case}: fla ran {len(calls) - n0} times, expected {per_run}"
            assert all(bool(torch.isfinite(t).all()) for t in (got[0], got[1], *got[2].values())), f"{case}: non-finite"
            if prec == "fp32":
                # the graded state is never quantised: bit-equal would mean the kernel's result was not used
                assert not torch.equal(got[1], ref[1]), f"{case}: fla final state bit-equal to the loop (kernel not used?)"
                out["fp32_logits_bit_equal"] = bool(torch.equal(got[0], ref[0]))
                if not spikes and out["fp32_logits_bit_equal"]:
                    fails.append("spikes off: fp32 logits bit-equal to the loop (kernel not used?)")
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
    bar = BARS[("fp32", False)]
    lim = {"logits": bar["logits"], "state": bar["state"], "kl": bar["kl"], "grad_adapters": bar["grad"],
           "grad_weights": bar["grad"], "grad_att_kernel": bar["grad"]}
    # spikes on: the per-logit numbers are recorded only, except when no count flipped (fp32 logits bit-equal), where
    # the kernel is the only difference and the spikes-off fp32 bars apply to the state and the gradients
    no_flip = spikes and out["fp32_logits_bit_equal"]
    out["asserted"] = ("fp32 per-logit + bf16-AMP/bf16-kernel" if not spikes else
                       "fp32 state + gradients (no count flipped)" if no_flip else "kernel ran + state differs only")
    if not spikes or no_flip:
        for m, v in out["fp32"]["fla"]["fp32"].items():
            if not v <= lim[m]:
                fails.append(f"fp32 {m} {v:.3g} > {lim[m]:.3g}")
    if not spikes:
        r = out["bf16"]["fla"][AMP_KERNEL_DTYPE]        # fp32 kernel inputs under AMP: recorded, refused by train()
        for m, v in r.items():
            b = max(lim[m], 2 * out["bf16"]["loop_floor"][m])
            if not v <= b:
                fails.append(f"bf16-amp kernel-{AMP_KERNEL_DTYPE} {m} {v:.3g} > {b:.3g}")
    out["fails"] = fails
    _dump(case, out)
    assert not fails, f"{case}: " + "; ".join(fails) + f"\n{json.dumps(out, indent=1)}"


def _dump(case, out):
    if os.environ.get("BRAIN_PARITY_OUT"):
        os.makedirs(os.environ["BRAIN_PARITY_OUT"], exist_ok=True)
        with open(os.path.join(os.environ["BRAIN_PARITY_OUT"], f"{case}.json"), "w") as f:
            json.dump(out, f, indent=1)


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


_TEACHER = {}


def _teacher():
    if "c" not in _TEACHER:
        from brain.experiments.spiking_twin_e1 import load_teacher
        _TEACHER["c"] = load_teacher("cuda")
    return _TEACHER["c"]


def test_gpu_parity_real_teacher_spikes_off():
    """Per-logit parity on the 0.1B teacher, spikes off (spikes on: the statistical tests below, amendment A1)."""
    _gpu_gate()
    c = _teacher()
    ids = c.pipe.encode("The history of the river valley begins with the first farmers, who settled along the banks "
                        "and learned to read the floods. User: what is my cat called?\n\nAssistant: Your cat is "
                        "called Pixel, and she likes the window seat in the morning sun. " * 8)
    toks = torch.tensor([ids[:256], ids[64:320]], device="cuda")
    _compare("teacher_off", c, toks, False)


def _need_file(path):
    if not os.path.exists(path):
        msg = f"missing {path}"
        if os.environ.get("BRAIN_REQUIRE_GPU") == "1":
            pytest.fail(msg)
        pytest.skip(msg)


def _heldout_windows(pipe, n, seq=512):
    """n windows of seq + 1 tokens from the start of WT103 val (held out: E1 trains on fineweb and teacher chat)."""
    import pyarrow.parquet as pq
    from brain.experiments.spiking_twin_e1 import WT103_VAL
    _need_file(WT103_VAL)
    text, ids = "", []
    for row in pq.read_table(WT103_VAL).column("text").to_pylist():
        text += row
        if len(text) > 8 * n * (seq + 1):           # about 4+ chars per token: tokenise once, with room to spare
            ids = pipe.encode(text)
            if len(ids) >= n * (seq + 1):
                break
    ids = ids if len(ids) >= n * (seq + 1) else pipe.encode(text)
    assert len(ids) >= n * (seq + 1), "WT103 val too short"
    return torch.tensor([ids[w * (seq + 1):(w + 1) * (seq + 1)] for w in range(n)], device="cuda")


def _nll_and_counts(tw, win, amp):
    """Per-token NLL (flat) and per-channel mean |count| and firing share over every spiking site (sorted)."""
    tw.stats = {}
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg = tw.forward_batch(win[:, :-1])[0].float()
        nll = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), win[:, 1:].reshape(-1), reduction="none")
        m = torch.cat([tw.stats[k][0] / tw.stats[k][1] for k in sorted(tw.stats)])
        f = torch.cat([tw.stats[k][2] / tw.stats[k][1] for k in sorted(tw.stats)])
    finally:
        tw.stats = None
    return nll.double(), m.double(), f.double()


def _stat_diff(a, b):
    d = a[0] - b[0]
    return {"nll_rms": float(d.pow(2).mean().sqrt()), "nll_mean": abs(float(d.mean())),
            "count_mean_abs": float((a[1] - b[1]).abs().mean()), "fire_mean_abs": float((a[2] - b[2]).abs().mean())}


STAT_ABS = {"nll_rms": 1e-3, "nll_mean": 1e-2, "count_mean_abs": 1e-4, "fire_mean_abs": 1e-4}


def test_gpu_spikes_on_statistical_teacher():
    """S1 + S2 (amendment A1): the 0.1B teacher's weights as a spiking twin (calibrated on 2 held-out windows), read on 8
    other held-out windows by the loop fp32 (reference), the loop bf16 AMP (floor), fla fp32 and fla bf16 AMP with bf16
    kernel inputs (the AMP training config); each fla read is compared with the loop at its own precision."""
    _gpu_gate()
    c = _teacher()
    win = _heldout_windows(c.pipe, 10)
    loop_tw = SpikingTwin(c, spikes=True)
    loop_tw.calibrate([win[:2, :-1]])
    fla_tw = SpikingTwin(c, spikes=True, wkv="fla")
    fla_tw.theta = dict(loop_tw.theta)
    calls = _count_fla(fla_tw)
    ev = win[2:]
    ref = _nll_and_counts(loop_tw, ev, False)
    loop_amp = _nll_and_counts(loop_tw, ev, True)
    fla32 = _nll_and_counts(fla_tw, ev, False)
    fla_tw.wkv_dtype = torch.bfloat16 if AMP_KERNEL_DTYPE == "bf16" else torch.float32
    fla_amp = _nll_and_counts(fla_tw, ev, True)
    assert len(calls) == 2 * fla_tw.L, f"fla ran {len(calls)} times, expected {2 * fla_tw.L}"
    floor = _stat_diff(loop_amp, ref)
    out = {"tokens": int(ev[:, 1:].numel()), "nll_loop_fp32": float(ref[0].mean()), "floor": floor,
           "fla_fp32_vs_loop_fp32": _stat_diff(fla32, ref),
           f"fla_amp_{AMP_KERNEL_DTYPE}_vs_loop_amp": _stat_diff(fla_amp, loop_amp),
           "bar": {m: max(STAT_X * floor[m], STAT_ABS[m]) for m in floor}}
    fails = [f"{name} {m} {v:.3g} > {out['bar'][m]:.3g}" for name in (k for k in out if k.startswith("fla_"))
             for m, v in out[name].items() if not v <= out["bar"][m]]
    out["fails"] = fails
    _dump("teacher_on_stat", out)
    assert not fails, "; ".join(fails) + f"\n{json.dumps(out, indent=1)}"


S3_CFG = dict(seq=256, micro_b=4, accum=1, warmup=10, tokens_per_arm=60 * 256 * 4, chat_share=0.0, calib_seqs=4)
S3_STEPS = 50


def _fineweb_pool(pipe, n_chars=600_000):
    import numpy as np
    from brain.experiments.spiking_twin_e1 import FINEWEB
    _need_file(FINEWEB)
    with open(FINEWEB, "rb") as fh:
        text = fh.read(n_chars).decode("utf-8", errors="ignore")
    return np.asarray(pipe.encode(text), dtype=np.int64)


def test_gpu_spikes_on_training_run_teacher(tmp_path, monkeypatch):
    """S3 (amendment A1): 50 optimizer steps of the real train() (arm a, seed 7, teacher-initialised spiking student,
    per-block checkpointing, teacher targets through the same kernel) with the loop fp32, the loop bf16 AMP (the floor
    between them) and fla bf16 AMP with bf16 kernel inputs (the E1 config), compared with the loop bf16 AMP."""
    _gpu_gate()
    from brain.experiments import spiking_twin_e1 as E
    c = _teacher()
    pool = _fineweb_pool(c.pipe)
    cfg = dict(E.CFG, **S3_CFG)
    calls, orig = [], SpikingTwin._wkv_fla

    def counted(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)
    monkeypatch.setattr(SpikingTwin, "_wkv_fla", counted)

    def run(name, precision, wkv):
        d = tmp_path / name
        d.mkdir()
        n0 = len(calls)
        E.train("a", 7, str(d), device="cuda", precision=precision, max_steps=S3_STEPS, cfg=cfg, cortex=c,
                pools=(pool, None), log=lambda *a, **k: None, wkv=wkv, wkv_dtype=AMP_KERNEL_DTYPE)
        blob = torch.load(d / "ckpt_a_s7.pt", map_location="cpu", weights_only=False)
        gc.collect()
        torch.cuda.empty_cache()
        return torch.tensor([row[1] for row in blob["losses"]], dtype=torch.float64), len(calls) - n0

    l32, n32 = run("loop_fp32", "fp32", "loop")
    l16, n16 = run("loop_bf16", "bf16", "loop")
    lfla, nfla = run("fla_bf16", "bf16", "fla")
    assert n32 == n16 == 0
    L = c.n_layers
    # per step: teacher targets + student forward + the student's checkpoint recompute, L layers each
    assert nfla >= S3_STEPS * 3 * L, f"fla ran {nfla} times in train(), expected >= {S3_STEPS * 3 * L}"
    assert len(l32) == len(l16) == len(lfla) == S3_STEPS
    assert bool(torch.isfinite(lfla).all())

    def diff(a, b):
        return {"loss_mean_abs": float((a - b).abs().mean()), "loss_last10": abs(float(a[-10:].mean() - b[-10:].mean()))}
    floor = diff(l16, l32)
    got = diff(lfla, l16)
    bar = {m: max(STAT_X * floor[m], 1e-5) for m in floor}
    out = {"steps": S3_STEPS, "cfg": S3_CFG, "fla_calls": nfla, "loss_loop_fp32": l32.tolist(),
           "loss_loop_bf16": l16.tolist(), "loss_fla_bf16": lfla.tolist(), "floor": floor,
           "fla_amp_vs_loop_amp": got, "bar": bar}
    fails = [f"{m} {v:.3g} > {bar[m]:.3g}" for m, v in got.items() if not v <= bar[m]]
    out["fails"] = fails
    _dump("teacher_on_train", out)
    assert not fails, "; ".join(fails) + f"\n{json.dumps({k: out[k] for k in ('floor', 'fla_amp_vs_loop_amp', 'bar')})}"


# ---------------------------------------------------------------------------------------------------- CPU (more)
class _NoisyLoop(SpikingTwin):
    """The loop with a kernel-sized relative perturbation (4e-5, the tiny spikes-off fla logits error) on its output
    and state: what fla's TF32 / reordered sums look like to the rest of the twin."""
    eps = 4e-5

    def _wkv_loop(self, *a):
        y, S = super()._wkv_loop(*a)
        g = torch.Generator().manual_seed(y.numel())
        return (y * (1 + self.eps * torch.randn(y.shape, generator=g, dtype=y.dtype)),
                S * (1 + self.eps * torch.randn(S.shape, generator=g, dtype=S.dtype)))


@pytest.mark.parametrize("T,seed,tseed,B", [(96, 5, 3, 3), (50, 7, 8, 2), (512, 7, 8, 2)],
                         ids=["tiny_on", "T50", "T512"])
def test_spike_counts_absorb_small_kernel_differences(T, seed, tseed, B):
    """Why the 11-27 GPU run saw fla logits bit-equal to the loop on the tiny cortex with spikes on although the kernel
    ran (its call count passed): the same cortex and tokens, a kernel-sized perturbation of the WKV output changes the
    spikes-off logits but not one spike count downstream, so the spikes-on logits stay bit-equal; the graded state
    differs. Not a bypass."""
    m = _Model(C=64, H=2, L=2, hid=128, V=256, r=8, seed=seed)
    toks = torch.randint(0, 256, (B, T), generator=torch.Generator().manual_seed(tseed))
    for spikes in (False, True):
        a, b = SpikingTwin(_CudaCortex(m), spikes=spikes), _NoisyLoop(_CudaCortex(m), spikes=spikes)
        a.calibrate([toks[:1]])
        b.theta = dict(a.theta)
        with torch.no_grad():
            la, sa = a.forward_batch(toks)
            lb, sb = b.forward_batch(toks)
        assert not torch.equal(sa[1], sb[1])
        assert torch.equal(la, lb) == spikes


def test_e1_train_refuses_fp32_kernel_inputs_under_amp(tmp_path):
    import numpy as np
    from brain.experiments import spiking_twin_e1 as E
    pools = (np.arange(4096) % 100, np.arange(4096) % 100)
    with pytest.raises(SystemExit, match="wkv-dtype bf16 only"):
        E.train("a", 7, str(tmp_path), device="cuda", precision="bf16", cortex=_Cortex(), pools=pools, max_steps=1,
                wkv="fla", wkv_dtype="fp32")
    import inspect
    assert inspect.signature(E.train).parameters["wkv_dtype"].default == "bf16"


def test_e1_train_flag_defaults_to_loop_and_fla_refuses_cpu(tmp_path):
    import numpy as np
    from brain.experiments import spiking_twin_e1 as E
    import inspect
    assert inspect.signature(E.train).parameters["wkv"].default == "loop"
    pools = (np.arange(4096) % 100, np.arange(4096) % 100)
    with pytest.raises(SystemExit, match="CUDA only"):
        E.train("a", 7, str(tmp_path), device="cpu", cortex=_Cortex(), pools=pools, max_steps=1, wkv="fla")
