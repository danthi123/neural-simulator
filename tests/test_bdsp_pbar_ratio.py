"""Pin the BDSP ratio baseline (`bdsp_pbar_ratio_tau_ms`, gap#4 clamp-companion lever, 2026-09-25).

WHAT IT IS. Payeur, Guerguiev, Zenke, Richards & Naud (bioRxiv 2020.03.30.015511 v1; Nat Neurosci 2021) set the
burst-probability baseline of the BDSP rule to "a moving average of the proportion of events that are bursts in
postsynaptic neuron i, with a slow (~ 1 - 10 s) time scale", and give the reason: "To ensure a finite growth of
synaptic weights". The engine's legacy baseline is an EMA of the instantaneous P at alpha 0.05/step (tau ~20 ms), and
the gap#4 C21 config presets it to the constant p0. `bdsp_pbar_ratio_tau_ms > 0` adds the source form: per neuron,
Pbar = EMA(B_post) / EMA(E), both EMAs with time constant tau (B_post is the same postsynaptic burst factor the kernel
uses: E*P under graded credit, the sampled B otherwise). With that baseline the time-integral of the kernel's
postsynaptic drive (B_post - Pbar*E) over ~tau is zero by construction, so a rectified (Jensen) bias of P cannot
accumulate into a one-sided weight drift. `cp_bdsp_pbar_ratio_mask` (runner-set, None = every neuron) restricts it.

WHAT THESE TESTS PIN.
  1. OFF IS BYTE-IDENTICAL: with the default (0.0) the weights and Pbar after a short BDSP training run hash to the
     values recorded from the engine BEFORE this edit (commit 4a516d387), and an explicit 0.0 equals the default.
  2. ON CHANGES THE BASELINE: Pbar of a masked-in neuron equals EMA(B_post)/EMA(E) (recomputed here from the same
     arrays), weights differ from OFF, and a masked-OUT neuron keeps the legacy baseline exactly (preset p0).
  3. THE REASON IT EXISTS: for a neuron driven by a zero-mean apical credit, the preset baseline accumulates a
     one-sided drive (the sigmoid is convex below P = 0.5, so mean P > p0), while the ratio baseline's time-integrated
     drive over several tau is a small fraction of it.

CPU/numpy, no GPU needed.
"""
import hashlib
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

# md5 of (cp_connections.data, cp_bdsp_Pbar) after _run_net(tau=None) on the engine BEFORE the ratio-baseline edit
# (commit 4a516d387, numpy backend). A change here means the OFF path is no longer byte-identical.
GOLDEN_OFF = {"weights_md5": "0ee1656891d3dd00a166a1a4e4e01ad9", "pbar_md5": "007d78f68dd2fd61d5a2d83c70e3cd20"}


def _md5(a):
    from sim.backend import to_host
    return hashlib.md5(np.ascontiguousarray(np.asarray(to_host(a))).tobytes()).hexdigest()


def _run_net(tau=None, mask_hidden_only=False, pbar_alpha=0.0, n_examples=6, record=False):
    """A tiny Gap4OnBridgeNet (fixed random feedback) trained on a few random examples. tau=None leaves the new
    config field untouched (the pre-edit code path)."""
    from research.runners._gap4_onbridge_spiking_selfpredict_derisk import Gap4OnBridgeNet
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, (n_examples, 5))
    y = rng.integers(0, 3, n_examples)
    net = Gap4OnBridgeNet(5, 4, 3, seed=11, feedback="fixed", n_hidden_layers=2, pool_k=1, settle_steps=10,
                          credit_steps=8, graded_credit=True, pbar_alpha=pbar_alpha, lr=2.0, ff_w_init=20.0,
                          tonic_h_pA=225.0, tonic_o_pA=250.0)
    if tau is not None:
        net.cfg.bdsp_pbar_ratio_tau_ms = float(tau)
    if mask_hidden_only:
        m = np.zeros(net.n_total, dtype=bool)
        m[net.slices[1].start:net.slices[len(net.sizes) - 2].stop] = True
        net.br.cp_bdsp_pbar_ratio_mask = net._xp.asarray(m)
    trace = []
    if record:
        step = net.br._run_one_simulation_step

        def rec():
            out = step()
            from sim.backend import to_host
            E = np.asarray(to_host(net.br.cp_bdsp_E), dtype=np.float64)
            P = np.asarray(to_host(net.br.cp_bdsp_P), dtype=np.float64)
            Pb = np.asarray(to_host(net.br.cp_bdsp_Pbar), dtype=np.float64)
            trace.append((E, P, Pb))
            return out
        net.br._run_one_simulation_step = rec
    for i in range(n_examples):
        net._train_one(X[i], int(y[i]), "bdsp")
    return net, trace


def test_ratio_baseline_default_off_is_byte_identical_to_pre_edit_engine():
    net, _ = _run_net(tau=None)
    got = {"weights_md5": _md5(net.br.cp_connections.data), "pbar_md5": _md5(net.br.cp_bdsp_Pbar)}
    assert getattr(net.br, "cp_bdsp_Ebar", None) is None, "OFF must never allocate the ratio EMAs"
    assert got == GOLDEN_OFF, "the OFF path changed: %s != pre-edit %s" % (got, GOLDEN_OFF)
    net0, _ = _run_net(tau=0.0)
    assert _md5(net0.br.cp_connections.data) == got["weights_md5"]
    assert _md5(net0.br.cp_bdsp_Pbar) == got["pbar_md5"]


def test_ratio_baseline_on_is_the_ratio_of_emas_and_respects_the_mask():
    from sim.backend import to_host
    off, _ = _run_net(tau=None)
    on, _ = _run_net(tau=50.0, mask_hidden_only=True)
    Ebar = np.asarray(to_host(on.br.cp_bdsp_Ebar), dtype=np.float64)
    Bbar = np.asarray(to_host(on.br.cp_bdsp_Bbar), dtype=np.float64)
    Pbar = np.asarray(to_host(on.br.cp_bdsp_Pbar), dtype=np.float64)
    hid = slice(on.slices[1].start, on.slices[len(on.sizes) - 2].stop)
    ok = Ebar[hid] > 1e-9
    assert ok.any()
    np.testing.assert_allclose(Pbar[hid][ok], (Bbar[hid] / Ebar[hid])[ok], rtol=1e-5)
    # masked-out (input + output) neurons keep the legacy baseline: pbar_alpha 0 => exactly the preset p0
    p0 = np.float32(on.cfg.bdsp_p0)
    assert np.all(np.asarray(to_host(on.br.cp_bdsp_Pbar))[on.slices[-1]] == p0)
    assert np.any(np.abs(Pbar[hid] - float(p0)) > 1e-4), "hidden baseline never moved off p0"
    assert _md5(on.br.cp_connections.data) != _md5(off.br.cp_connections.data), "the lever is inert"


def _symmetric_apical_drive(tau, n_blocks=160, block=25, amp=60.0):
    """Hidden neurons under steady input and a ZERO-MEAN apical current (blocks of +amp / -amp / 0, equally often, in a
    fixed random order), with lr 0. Returns the time-integrated postsynaptic drive sum_t E*(P - Pbar) over the hidden
    neurons in the second half of the run (after the ratio EMAs have had >= 8 tau to settle)."""
    from sim.backend import to_host
    net, _ = _run_net(tau=tau, mask_hidden_only=tau is not None, n_examples=0)
    net.cfg.bdsp_learning_rate = 0.0
    xp = net._xp
    drive = net._base_drive()
    drive[net.slices[0]] = 700.0
    net.br.cp_external_input_current = xp.asarray(drive)
    hid = slice(net.slices[1].start, net.slices[len(net.sizes) - 2].stop)
    order = np.random.default_rng(5).permutation(np.repeat([1.0, -1.0, 0.0], n_blocks // 3 + 1))[:n_blocks]
    tot = 0.0
    for bi, s in enumerate(order):
        ap = np.zeros(net.n_total, dtype=np.float32)
        ap[hid] = s * amp
        net.br.cp_bdsp_apical_drive = xp.asarray(ap)
        for _ in range(block):
            net.br._run_one_simulation_step()
            if bi >= n_blocks // 2:
                E = np.asarray(to_host(net.br.cp_bdsp_E), dtype=np.float64)[hid]
                P = np.asarray(to_host(net.br.cp_bdsp_P), dtype=np.float64)[hid]
                Pb = np.asarray(to_host(net.br.cp_bdsp_Pbar), dtype=np.float64)[hid]
                tot += float(np.sum(E * (P - Pb)))
    return tot


def test_ratio_baseline_cancels_the_one_sided_drive_a_zero_mean_apical_gives_the_preset_baseline():
    """The reason the lever exists. P = sigmoid(beta*scale*(v_apical - E_rest) + logit(p0)) is CONVEX below P = 0.5, so a
    zero-mean apical credit around p0 = 0.3 gives mean P > p0 (Jensen): under the preset baseline every active synapse
    onto the neuron gets a net-LTP drive, which a hard clamp then catches. The ratio baseline tracks the event-weighted
    mean of P, so the same zero-mean apical integrates to a small fraction of that drive."""
    d_pre = _symmetric_apical_drive(tau=None)
    d_rat = _symmetric_apical_drive(tau=200.0)
    assert d_pre > 0.0, "a zero-mean apical should give the preset baseline a net-LTP drive, got %r" % d_pre
    assert abs(d_rat) < 0.1 * d_pre, "ratio baseline drive %r is not << the preset's %r" % (d_rat, d_pre)
