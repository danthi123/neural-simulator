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

Parity bar (registered in the prereg section 7 item 1, as amended before data by A1 and A2, 2026-11-27).
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
  S1 per-token NLL on held-out WT103 val, A2: 64 windows x 512 tokens (32768 tokens), drawn by a fixed seed
     (S1_SEED) from the non-overlapping 513-token grid of the whole file minus the 2 calibration windows, the same
     windows for every arm: rms(dNLL) <= max(1.5 F, 1e-3); the paired mean dNLL with a window-clustered SE (sd of the
     64 per-window mean dNLLs / 8) must have a 95% t CI (df 63) that includes 0 AND |mean| <= 0.03 nats (A2 replaced
     A1's |mean| <= max(1.5 F, 0.01) on 8 windows: SE 0.03-0.06 there, a correct kernel passed 18-50%). One
     pre-registered repeat (BRAIN_S1_REPEAT=1, only after a rerun failing the mean rule alone): draw 2 by S1_REPEAT_SEED,
     disjoint from draw 1; S1-mean then needs the rule on draw 2 AND on the pooled 128 windows (t df 127);
     Kernel bias, per layer (A2, skeptic): the mean NLL is a weak bias detector (the chaotic twin turns a bias into
     noise), so the first 8 S1 windows also run fla on each layer's loop-fp32 inputs: fla fp32 |slope of
     (y_fla - y_loop) on y_loop| <= 1e-5 in every layer; o-site flips lowering |count| 0.47-0.53 of all flips when
     there are >= 1000 (fp32 IEEE and bf16 inputs);
     S2 spike counts per channel (mean |count| and firing share, all sites): mean |diff| over
     channels <= max(1.5 F, 1e-4); S3 50 steps of the real train() (arm a, seq 256, micro-batch 4, warmup 10, bf16 AMP,
     fla with bf16 kernel inputs vs the loop): mean |d loss| per step and |d mean loss of the last 10 steps| <=
     max(1.5 F, 1e-5), F from the loop's own bf16 vs fp32 training runs.
Kernel config (A2): fla's fp32 tl.dot runs IEEE fp32 (TRITON_F32_DEFAULT=ieee, set by brain.spiking_twin.fla_import;
every GPU case records the setting and the fla cases assert it). The fp32 bars were set allowing TF32 (about 1e-3
relative) and are not tightened; the bf16 bars are relative to the loop's floor.
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
# S1 sample and mean rule (amendment A2, written before the rerun; prereg section 7 item 1)
S1_WINDOWS, S1_SEQ, S1_CALIB, S1_SEED, S1_BATCH = 64, 512, 2, 1127, 8   # S1_BATCH: windows per forward (memory only)
S1_MEAN_MAX = 0.03                 # nats: <= 10% of the 0.3-0.4 nat teacher-twin gap E1 measures (SpikingMamba, memo)
S1_T95 = 1.998340542520741         # two-sided 95% t quantile, df = S1_WINDOWS - 1 = 63 (scipy.stats.t.ppf(0.975, 63))
# A2 repeat (skeptic change 3): one pre-registered repeat on a second disjoint draw, seed fixed now; run only with
# BRAIN_S1_REPEAT=1 after a rerun whose only failure was the S1 mean rule. It decides S1-mean by the rule on draw 2
# AND on the pooled 128 windows (t df 127); every other S1/S2 bar stays on draw 1.
S1_REPEAT_SEED = 1128
S1_T95_POOLED = 1.9788195347028539   # scipy.stats.t.ppf(0.975, 2 * S1_WINDOWS - 1 = 127)
# A2 per-layer kernel bias check (skeptic change 1): the mean NLL is a weak bias detector (the chaotic twin turns a
# kernel bias into noise); the detector is the isolated per-layer statistic on identical inputs
# (brain/experiments/spiking_twin_fla_bias_diag.py, DiagTwin._iso_layer), read on the first ISO_WINDOWS S1 windows.
ISO_WINDOWS = 8
ISO_SLOPE_MAX_FP32 = 1e-5          # |slope of (y_fla - y_loop) on y_loop|, every layer, fla fp32 (IEEE 1.5e-6 at most,
                                   #   loop fp64 1.4e-6; TF32 2.9e-4 at least: bias_diagnosis.md)
ISO_FLIP_MIN = 1000                # flip asymmetry is asserted only with >= this many o-site flips (all layers)
ISO_FLIP_DOWN_BAND = (0.47, 0.53)  # share of o-site flips lowering |count| (bf16 inputs 0.4987-0.5002; TF32 0.63)


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
    out.setdefault("triton_f32_default", os.environ.get("TRITON_F32_DEFAULT"))   # A2: part of the kernel config
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
    """n windows of seq + 1 tokens from the start of WT103 val (held out: E1 trains on fineweb and teacher chat). The A1
    S1 sample (8 eval windows), kept for spiking_twin_fla_bias_diag; S1 itself reads _s1_windows (A2)."""
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


def _wt103_val_ids(pipe):
    """The whole WT103 val file (rows concatenated), tokenised once."""
    import pyarrow.parquet as pq
    from brain.experiments.spiking_twin_e1 import WT103_VAL
    _need_file(WT103_VAL)
    return pipe.encode("".join(pq.read_table(WT103_VAL).column("text").to_pylist()))


def _s1_window_ids(n_grid, n=S1_WINDOWS, calib=S1_CALIB, seed=S1_SEED, exclude=()):
    """A2: calibration = grid windows 0 .. calib-1 (as in A1); evaluation = n distinct other grid windows (not in
    `exclude`) drawn by a fixed seed, sorted. The same windows for every arm and implementation."""
    import numpy as np
    pool = [i for i in range(calib, n_grid) if i not in set(exclude)]
    assert len(pool) >= n, f"WT103 val has {n_grid} windows, need {calib} + {len(set(exclude))} + {n}"
    pick = np.random.default_rng(seed).choice(np.asarray(pool), size=n, replace=False)
    return list(range(calib)), sorted(int(i) for i in pick)


def _s1_repeat_window_ids(n_grid):
    """A2 repeat: draw 2 = S1_WINDOWS grid windows by S1_REPEAT_SEED, disjoint from draw 1 and the calibration."""
    cal, ev1 = _s1_window_ids(n_grid)
    return cal, ev1, _s1_window_ids(n_grid, seed=S1_REPEAT_SEED, exclude=ev1)[1]


def _s1_windows(pipe, seq=S1_SEQ, repeat=False):
    """(calibration windows, evaluation windows, evaluation grid indices[, draw-2 windows, draw-2 indices]); windows
    are seq + 1 tokens."""
    ids = _wt103_val_ids(pipe)
    grid = torch.tensor(ids[:len(ids) // (seq + 1) * (seq + 1)], device="cuda").view(-1, seq + 1)
    if repeat:
        cal, ev, ev2 = _s1_repeat_window_ids(grid.shape[0])
        return grid[cal], grid[ev], ev, grid[ev2], ev2
    cal, ev = _s1_window_ids(grid.shape[0])
    return grid[cal], grid[ev], ev


def _nll_and_counts(tw, win, amp, batch=None):
    """Per-token NLL (flat, window-major) and per-channel mean |count| and firing share over every spiking site
    (sorted), read over all windows; `batch` windows per forward (memory only: the windows are independent)."""
    tw.stats = {}
    nll = []
    try:
        for i in range(0, win.shape[0], batch or win.shape[0]):
            w = win[i:i + (batch or win.shape[0])]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg = tw.forward_batch(w[:, :-1])[0].float()
            nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), w[:, 1:].reshape(-1),
                                                         reduction="none"))
            del lg
        m = torch.cat([tw.stats[k][0] / tw.stats[k][1] for k in sorted(tw.stats)])
        f = torch.cat([tw.stats[k][2] / tw.stats[k][1] for k in sorted(tw.stats)])
    finally:
        tw.stats = None
    return torch.cat(nll).double(), m.double(), f.double()


def _paired_mean(d, n_win):
    """A2: paired mean of the per-token dNLL and its window-clustered SE (windows are the independent items: each
    starts from a zero state). Equal-size windows, so the mean of window means is the token mean."""
    wm = d.reshape(n_win, -1).mean(1)
    return float(wm.mean()), float(wm.std(unbiased=True)) / n_win ** 0.5


def _s1_mean_verdict(mean, se, t=S1_T95, cap=S1_MEAN_MAX):
    """A2 mean rule: the 95% CI of the mean dNLL includes 0 AND |mean| <= cap (both inclusive)."""
    lo, hi = mean - t * se, mean + t * se
    ci0, small = lo <= 0 <= hi, abs(mean) <= cap
    return {"mean": mean, "se_window": se, "ci95": [lo, hi], "ci_includes_0": ci0, "abs_mean_le_cap": small,
            "cap": cap, "pass": ci0 and small}


def _stat_diff(a, b, n_win=None):
    d = a[0] - b[0]
    out = {"nll_rms": float(d.pow(2).mean().sqrt()), "nll_mean": abs(float(d.mean())),
           "count_mean_abs": float((a[1] - b[1]).abs().mean()), "fire_mean_abs": float((a[2] - b[2]).abs().mean())}
    if n_win is not None:
        out["nll_mean_rule"] = _s1_mean_verdict(*_paired_mean(d, n_win))
    return out


def _s1_repeat_verdict(d1, d2, n_win=S1_WINDOWS):
    """A2 repeat: per-token dNLL on draw 1 and draw 2 (flat, window-major). Draw 1 is recorded; S1-mean passes only if
    the A2 rule holds on draw 2 AND on the pooled 2 x n_win windows (t df 2 n_win - 1)."""
    r1 = _s1_mean_verdict(*_paired_mean(d1, n_win))
    r2 = _s1_mean_verdict(*_paired_mean(d2, n_win))
    rp = _s1_mean_verdict(*_paired_mean(torch.cat([d1, d2]), 2 * n_win), t=S1_T95_POOLED)
    return {"draw1": r1, "draw2": r2, "pooled": rp, "pass": r2["pass"] and rp["pass"]}


def _kernel_bias_verdict(iso, slope_max=None):
    """A2 per-layer bias check on isolated per-layer stats ({layer: DiagTwin._iso_layer dict}): every layer
    |slope_d_on_y| <= slope_max (if given), and, with >= ISO_FLIP_MIN o-site flips over all layers, the share lowering
    |count| inside ISO_FLIP_DOWN_BAND (fewer flips: recorded only)."""
    slopes = [iso[k]["slope_d_on_y"] for k in sorted(iso, key=int)]
    down = sum(iso[k]["o_flips_abs_down"] for k in iso)
    flips = down + sum(iso[k]["o_flips_abs_up"] for k in iso)
    share = down / flips if flips else None
    fails = []
    if slope_max is not None and not max(abs(x) for x in slopes) <= slope_max:
        fails.append(f"max |slope| {max(abs(x) for x in slopes):.3g} > {slope_max:.3g}")
    if flips >= ISO_FLIP_MIN and not ISO_FLIP_DOWN_BAND[0] <= share <= ISO_FLIP_DOWN_BAND[1]:
        fails.append(f"{share:.4f} of {flips} o-site flips lower |count|, outside {ISO_FLIP_DOWN_BAND}")
    return {"slopes": slopes, "slope_max": slope_max, "o_flips_abs_changed": flips, "share_abs_down": share,
            "flip_asym_asserted": flips >= ISO_FLIP_MIN, "fails": fails, "pass": not fails}


STAT_ABS = {"nll_rms": 1e-3, "nll_mean": 1e-2, "count_mean_abs": 1e-4, "fire_mean_abs": 1e-4}
FLOOR_BARRED = ("nll_rms", "count_mean_abs", "fire_mean_abs")   # A2: the mean has its own rule (_s1_mean_verdict)


def test_gpu_spikes_on_statistical_teacher():
    """S1 + S2 (amendments A1, A2): the 0.1B teacher's weights as a spiking twin (calibrated on WT103 val windows 0-1),
    read on 64 other held-out windows (fixed seed) by the loop fp32 (reference), the loop bf16 AMP (floor), fla fp32
    and fla bf16 AMP with bf16 kernel inputs (the AMP training config); each fla read is compared with the loop at its
    own precision, the windows paired."""
    _gpu_gate()
    c = _teacher()
    repeat = os.environ.get("BRAIN_S1_REPEAT") == "1"
    if repeat:
        cal, ev, ev_ids, ev2, ev2_ids = _s1_windows(c.pipe, repeat=True)
    else:
        cal, ev, ev_ids = _s1_windows(c.pipe)
    loop_tw = SpikingTwin(c, spikes=True)
    loop_tw.calibrate([cal[:, :-1]])
    fla_tw = SpikingTwin(c, spikes=True, wkv="fla")
    fla_tw.theta = dict(loop_tw.theta)
    calls = _count_fla(fla_tw)
    n = ev.shape[0]
    ref = _nll_and_counts(loop_tw, ev, False, S1_BATCH)
    loop_amp = _nll_and_counts(loop_tw, ev, True, S1_BATCH)
    dt32 = fla_tw.wkv_dtype
    fla32 = _nll_and_counts(fla_tw, ev, False, S1_BATCH)
    fla_tw.wkv_dtype = torch.bfloat16 if AMP_KERNEL_DTYPE == "bf16" else torch.float32
    fla_amp = _nll_and_counts(fla_tw, ev, True, S1_BATCH)
    n_fwd = -(-n // S1_BATCH)
    assert len(calls) == 2 * n_fwd * fla_tw.L, f"fla ran {len(calls)} times, expected {2 * n_fwd * fla_tw.L}"
    if repeat:                                       # draw 2, the same four reads (amp flag and kernel dtype as above)
        fla_tw.wkv_dtype = dt32
        ref2, loop_amp2 = _nll_and_counts(loop_tw, ev2, False, S1_BATCH), _nll_and_counts(loop_tw, ev2, True, S1_BATCH)
        fla32_2 = _nll_and_counts(fla_tw, ev2, False, S1_BATCH)
        fla_tw.wkv_dtype = torch.bfloat16 if AMP_KERNEL_DTYPE == "bf16" else torch.float32
        fla_amp2 = _nll_and_counts(fla_tw, ev2, True, S1_BATCH)
        assert len(calls) == 4 * n_fwd * fla_tw.L
    assert os.environ.get("TRITON_F32_DEFAULT") == "ieee", "A2: fla must run with IEEE fp32 dots"
    floor = _stat_diff(loop_amp, ref, n)
    out = {"amendment": "A2", "tokens": int(ev[:, 1:].numel()), "windows": n, "seq": S1_SEQ, "seed": S1_SEED,
           "eval_window_ids": ev_ids, "calib_window_ids": list(range(S1_CALIB)),
           "nll_loop_fp32": float(ref[0].mean()), "floor": floor,
           "fla_fp32_vs_loop_fp32": _stat_diff(fla32, ref, n),
           f"fla_amp_{AMP_KERNEL_DTYPE}_vs_loop_amp": _stat_diff(fla_amp, loop_amp, n),
           "bar": {m: max(STAT_X * floor[m], STAT_ABS[m]) for m in FLOOR_BARRED},
           "mean_rule": f"95% t CI (df {n - 1}) of the paired mean dNLL includes 0 and |mean| <= {S1_MEAN_MAX}"}
    if repeat:
        out["repeat"] = {"seed": S1_REPEAT_SEED, "eval_window_ids": ev2_ids, "floor": _stat_diff(loop_amp2, ref2, n),
                         "rule": "A2 rule on draw 2 AND on the pooled 2 x 64 windows (t df 127); other bars on draw 1"}
        for name, (a, b, a2, b2) in {"fla_fp32_vs_loop_fp32": (fla32, ref, fla32_2, ref2),
                                     f"fla_amp_{AMP_KERNEL_DTYPE}_vs_loop_amp": (fla_amp, loop_amp, fla_amp2,
                                                                                 loop_amp2)}.items():
            out["repeat"][name] = dict(_stat_diff(a2, b2, n), nll_mean_rule=_s1_repeat_verdict(a[0] - b[0],
                                                                                               a2[0] - b2[0], n))
            out[name]["nll_mean_rule_draw1"] = out[name]["nll_mean_rule"]
            out[name]["nll_mean_rule"] = out["repeat"][name]["nll_mean_rule"]   # the repeat decides S1-mean
    fails = [f"{name} {m} {out[name][m]:.3g} > {out['bar'][m]:.3g}" for name in (k for k in out if k.startswith("fla_"))
             for m in FLOOR_BARRED if not out[name][m] <= out["bar"][m]]
    fails += [f"{name} mean dNLL {r['mean']:+.4f} (SE {r['se_window']:.4f}, "
              f"CI {r['ci95'][0]:+.4f}..{r['ci95'][1]:+.4f}) fails the A2 rule" for name in (k for k in out if k.startswith("fla_"))
              for r in [out[name]["nll_mean_rule"]] if not repeat and not r["pass"]]
    fails += [f"{name} A2 repeat fails (draw 2 {r['draw2']['mean']:+.4f} SE {r['draw2']['se_window']:.4f}; pooled "
              f"{r['pooled']['mean']:+.4f} SE {r['pooled']['se_window']:.4f})" for name in (k for k in out
                                                                                          if k.startswith("fla_"))
              for r in [out[name]["nll_mean_rule"]] if repeat and not r["pass"]]
    out["fails"] = fails
    _dump("teacher_on_stat_repeat" if repeat else "teacher_on_stat", out)
    assert not fails, "; ".join(fails) + f"\n{json.dumps(out, indent=1)}"


def test_gpu_kernel_bias_per_layer_teacher():
    """A2 (skeptic change 1): the kernel-bias detector. On the first ISO_WINDOWS S1 windows, the loop fp32 run of the
    calibrated spiking teacher hands each layer's WKV inputs to fla (fp32 IEEE and bf16 inputs, the E1 config) on the
    SAME inputs (DiagTwin, brain/experiments/spiking_twin_fla_bias_diag.py). fla fp32: every layer |slope of
    (y_fla - y_loop) on y_loop| <= ISO_SLOPE_MAX_FP32; both paths: with >= ISO_FLIP_MIN o-site count flips, the share
    lowering |count| inside ISO_FLIP_DOWN_BAND. bf16 slopes are recorded only (+-4e-4, no sign pattern)."""
    _gpu_gate()
    from brain.experiments.spiking_twin_fla_bias_diag import DiagTwin
    c = _teacher()
    cal, ev, ev_ids = _s1_windows(c.pipe)
    base = SpikingTwin(c, spikes=True)
    base.calibrate([cal[:, :-1]])
    tw = DiagTwin(c, spikes=True)
    tw.theta = dict(base.theta)
    tw.kern, tw.iso, tw.rec = "loop", {}, None
    tw.iso_variants = ("chunk32", "chunk_bf16in")
    with torch.no_grad():
        tw.forward_batch(ev[:ISO_WINDOWS, :-1])
    assert os.environ.get("TRITON_F32_DEFAULT") == "ieee", "A2: fla must run with IEEE fp32 dots"
    iso = tw.iso
    assert set(iso) == set(tw.iso_variants) and all(len(iso[v]) == tw.L for v in iso)
    out = {"amendment": "A2", "windows": ISO_WINDOWS, "eval_window_ids": ev_ids[:ISO_WINDOWS],
           "fla_fp32": _kernel_bias_verdict(iso["chunk32"], ISO_SLOPE_MAX_FP32),
           "fla_bf16in": _kernel_bias_verdict(iso["chunk_bf16in"]), "iso_per_layer": iso}
    fails = [f"{k}: {f}" for k in ("fla_fp32", "fla_bf16in") for f in out[k]["fails"]]
    out["fails"] = fails
    _dump("teacher_on_kernel_bias", out)
    assert not fails, "; ".join(fails)


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
def test_s1_windows_fixed_seed_distinct_and_held_apart_from_calibration():
    """A2: 64 eval windows drawn once by the fixed seed: reproducible, distinct, sorted, never a calibration window."""
    cal, ev = _s1_window_ids(489)            # 489 = the 513-token grid of the whole WT103 val file (rwkv vocab)
    assert cal == [0, 1] and ev == _s1_window_ids(489)[1]
    assert len(ev) == S1_WINDOWS == len(set(ev)) and ev == sorted(ev) and min(ev) >= S1_CALIB and max(ev) < 489
    assert _s1_window_ids(489, seed=S1_SEED + 1)[1] != ev
    with pytest.raises(AssertionError):
        _s1_window_ids(S1_WINDOWS + 1)


def test_s1_repeat_draw_fixed_disjoint_from_draw1_and_calibration():
    """A2 repeat: draw 2 is fixed by S1_REPEAT_SEED and shares no window with draw 1 or the calibration."""
    cal, ev1, ev2 = _s1_repeat_window_ids(489)
    assert ev1 == _s1_window_ids(489)[1] and (cal, ev1, ev2) == _s1_repeat_window_ids(489)
    assert len(ev2) == S1_WINDOWS == len(set(ev2)) and ev2 == sorted(ev2)
    assert not set(ev2) & set(ev1) and min(ev2) >= S1_CALIB and max(ev2) < 489


def _wins(means, seq=S1_SEQ):
    return torch.tensor(means, dtype=torch.float64)[:, None].expand(-1, seq).reshape(-1)


def test_s1_repeat_needs_draw2_and_pooled():
    """A2 repeat: S1-mean passes only if draw 2 AND the pooled 128 windows pass; draw 1 alone never decides."""
    g = torch.Generator().manual_seed(1)
    e = torch.randn(S1_WINDOWS, generator=g, dtype=torch.float64)
    e = (e - e.mean()) / e.std(unbiased=True)                    # window means with exactly mean 0, sd 1
    z = lambda m, sd: (m + sd * e).tolist()                      # mean m, window SE sd / 8
    ok, bad = z(0.0, 0.02), z(0.045, 0.2)                        # bad: CI includes 0 but |mean| > 0.03
    assert _s1_repeat_verdict(_wins(bad), _wins(ok))["draw1"]["pass"] is False
    r = _s1_repeat_verdict(_wins(bad), _wins(ok))
    assert r["draw2"]["pass"] and r["pooled"]["pass"] and r["pass"]          # draw 1 failed, the repeat decides
    assert _s1_repeat_verdict(_wins(z(0.06, 0.02)), _wins(ok))["pooled"]["pass"] is False   # pooled mean at 0.03, CI
    assert _s1_repeat_verdict(_wins(ok), _wins(bad))["pass"] is False        # draw 2 fails
    r = _s1_repeat_verdict(_wins(z(0.075, 0.15)), _wins(z(0.025, 0.15)))    # draw 2 passes, pooled mean 0.05 > cap
    assert r["draw2"]["pass"] and not r["pooled"]["pass"] and not r["pass"]
    assert abs(r["pooled"]["mean"] - (r["draw1"]["mean"] + r["draw2"]["mean"]) / 2) < 1e-12


def _iso(slopes, down, up):
    return {str(i): {"slope_d_on_y": s, "o_flips_abs_down": d, "o_flips_abs_up": u}
            for i, (s, d, u) in enumerate(zip(slopes, down, up))}


def test_kernel_bias_verdict_on_the_diagnosis_numbers():
    """A2 per-layer detector, on numbers of the size in bias_diagnosis.md: TF32 fp32 (slopes -2.9e-4..-9e-4, 63% of
    flips lower |count|) fails both parts; IEEE fp32 (slopes <= 1.5e-6, 10 flips: asymmetry recorded only) and bf16
    inputs (symmetric flips, slope not asserted) pass; each part fails alone."""
    tf32 = _kernel_bias_verdict(_iso([-2.9e-4, -9e-4], [6776, 6776], [3942, 3941]), ISO_SLOPE_MAX_FP32)
    assert not tf32["pass"] and len(tf32["fails"]) == 2
    ieee = _kernel_bias_verdict(_iso([-6.5e-8, -1.5e-6], [2, 2], [3, 3]), ISO_SLOPE_MAX_FP32)
    assert ieee["pass"] and not ieee["flip_asym_asserted"]
    assert _kernel_bias_verdict(_iso([4e-4, -9e-5], [33530, 33530], [33539, 33539]))["pass"]
    assert not _kernel_bias_verdict(_iso([1e-6, 2e-5], [3, 3], [3, 3]), ISO_SLOPE_MAX_FP32)["pass"]     # slope alone
    assert not _kernel_bias_verdict(_iso([1e-6, 1e-6], [540, 540], [460, 460]), ISO_SLOPE_MAX_FP32)["pass"]  # 0.54
    assert _kernel_bias_verdict(_iso([1e-6, 1e-6], [529, 0], [470, 0]))["pass"] is True   # 999 flips: recorded only


def test_s1_t_quantile_and_clustered_se():
    sp = pytest.importorskip("scipy.stats")
    assert abs(S1_T95 - sp.t.ppf(0.975, S1_WINDOWS - 1)) < 1e-12
    assert abs(S1_T95_POOLED - sp.t.ppf(0.975, 2 * S1_WINDOWS - 1)) < 1e-12
    g = torch.Generator().manual_seed(0)
    wm = torch.randn(S1_WINDOWS, generator=g, dtype=torch.float64)
    d = (wm[:, None] + 0.1 * torch.randn(S1_WINDOWS, S1_SEQ, generator=g, dtype=torch.float64)).reshape(-1)
    mean, se = _paired_mean(d, S1_WINDOWS)
    w = d.reshape(S1_WINDOWS, -1).mean(1)
    assert abs(mean - float(d.mean())) < 1e-12
    assert abs(se - float(w.std(unbiased=True)) / S1_WINDOWS ** 0.5) < 1e-12
    # the token-level SE would treat 32768 correlated tokens as independent and be about sqrt(512) x too small here
    assert se > 5 * float(d.std(unbiased=True)) / d.numel() ** 0.5


@pytest.mark.parametrize("mean,se,ok", [(0.0, 0.01, True), (0.02, 0.0101, True), (0.02, 0.0099, False),
                                        (-0.03, 0.02, True), (0.0301, 0.05, False), (-0.031, 0.05, False),
                                        (0.012, 0.005, False)])
def test_s1_mean_rule_needs_ci_through_zero_and_small_mean(mean, se, ok):
    """A2: pass only if the 95% CI includes 0 AND |mean| <= 0.03; each part fails alone (0.02 at SE 0.0099: CI
    excludes 0; 0.0301 / -0.031 at SE 0.05: CI includes 0 but over the cap)."""
    r = _s1_mean_verdict(mean, se)
    assert r["pass"] is ok
    assert r["ci_includes_0"] == (abs(mean) <= S1_T95 * se) and r["abs_mean_le_cap"] == (abs(mean) <= S1_MEAN_MAX)


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
