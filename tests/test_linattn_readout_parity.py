"""EXACT-PARITY gate for `LinAttnReadout` (research/runners/_wkv_fewspike_read_derisk.py, 2026-09-03) --
the deployment read-back for a `--recurrence linattn` checkpoint (research/findings/2026-09-03-linattn-
production-mouth-wiring-DESIGN.md).

WHY THIS TEST IS THE DELIVERABLE, NOT A NICE-TO-HAVE. The design doc's own §6 failure-mode-4 names the risk
explicitly: a numpy transcription that silently diverges from the torch `LinAttnLayer.forward` reads a real
checkpoint as near-uniform garbage WITH NO ERROR -- the 2026-07-20 gap#1 ROOT-CAUSE class
(`research/findings/2026-07-20-gap1-ROOT-CAUSE-wrong-recurrence-mode-retrain-fixes-catastrophe...`) already
burned real GPU-hours on exactly this failure mode for a DIFFERENT recurrence family. `LinAttnReadout` imports
and instantiates cleanly regardless of whether its per-position algebra is correct, so "it loads" proves
nothing; only a numeric agreement against the REAL torch forward (not a re-derivation of the same formula) does.

METHOD (mirrors `tests/test_wkv_readout_multilayer.py::TestMultiLayerNumericalCorrectness`, the established
pattern for this exact class of test in this repo): build a REAL `--recurrence linattn` torch model via the
production `build_and_train_wkv` (never a reimplementation of the trainer), save it byte-for-byte the same
shape `--save-ssm` writes, load it back through `LinAttnReadout`, and walk the SAME token sequence through both
the torch model's teacher-forced `forward()` and the numpy readout's `advance`/`logits` autoregressive loop --
comparing every position's logits.

PRECISION: the primary comparisons cast the torch model to float64 (`net.double()`) before comparing, isolating
the TRANSCRIPTION ALGEBRA from float32-vs-float64 rounding noise -- this is the harder, more decisive bar than
the task's own "<1e-4" ask, and is what actually tells us whether the recurrence math is right (a bug in the
read would show up as a large, structural mismatch, not float noise). A secondary float32 check (matching
`test_wkv_readout_multilayer.py`'s own precision-crossing tolerance) additionally confirms realistic deployment
precision is fine too.

`epochs=0` (architecturally correct init, no training): this test is about the LOAD/TRANSCRIBE plumbing, not
about a trained model's fluency -- exactly `test_wkv_readout_multilayer.py::_build_tiny_2layer_net`'s own
justification for the identical choice. A `--recurrence linattn` random-init checkpoint is `--save-ssm`'s own
output shape (`np.savez(..., V=V, d_model=D, words=..., **net.state_dict())`), a legitimate "synthetic
checkpoint" per the task brief.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest


def _build_tiny_linattn_net(seed=42, V=37, D=12, n_layers=2, phi="elu", norm=True, gate=False,
                             uniform_decay=False, epochs=0):
    """A REAL `--recurrence linattn` torch WKV model via the production trainer function itself
    (`build_and_train_wkv`). `epochs=0` -> stays at random init (fast, and this test is about the
    LOAD/TRANSCRIBE plumbing -- see the module docstring)."""
    from research.runners._emerge_wkv_lm_derisk import build_and_train_wkv

    args = types.SimpleNamespace(
        d_model=D, n_layers=n_layers, recurrence="linattn", linattn_phi=phi, linattn_norm=norm,
        assoc_gate=gate, uniform_decay=uniform_decay, freeze_emb=False, lr=3e-3, weight_decay=1e-4,
        batch=8, epochs=epochs, compile=False, pred_aux_weight=0.0,
    )
    # unused at epochs=0, only needed so build_and_train_wkv's setup doesn't choke on an empty corpus
    tr_ids = [[1, 2, 3, 4, 5], [2, 3, 1, 4, 2, 5, 6], [6, 5, 4, 3, 2, 1]]
    net, _WKV_cls = build_and_train_wkv(tr_ids, V, seed, args, device="cpu")
    net.eval()
    return net, V, D


def _save_ssm_style(net, V, D, path, words=None):
    """Byte-for-byte the SAME save shape `_emerge_wkv_lm_derisk.py main()`'s `--save-ssm` branch writes:
    `np.savez(f"{path}_seed{seed}.npz", V=V, d_model=D, words=np.array(vocab.i2w, dtype=object), **state_dict)`."""
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    words = words or [f"tok{i}" for i in range(V - 1)] + ["<unk>"]
    np.savez(path, V=V, d_model=D, words=np.array(words, dtype=object), **sd)


@pytest.fixture()
def readout_mod():
    from research.runners import _wkv_fewspike_read_derisk as mod
    return mod


def _walk_torch(net, tokseq, double=False):
    import torch
    net = net.double() if double else net
    with torch.no_grad():
        dtype_seq = torch.tensor([tokseq])
        logits = net(dtype_seq)[0]
    return logits.numpy()


def _walk_numpy(ro, tokseq):
    """`advance(state, t)` does the full per-position write+read computation and caches the resulting residual
    stream; `logits(state, t)` only reads that cache (see `LinAttnReadout.advance`'s docstring -- calling
    `logits` without a preceding `advance` for the SAME position would raise, and an early bug in this class
    called the update a second time inside `logits`, silently double-applying the current token's write)."""
    state = ro.init_state()
    out = []
    for t in tokseq:
        state = ro.advance(state, t)
        out.append(ro.logits(state, t))
    return np.stack(out, 0)


TOKSEQ = [3, 7, 1, 22, 5, 9, 14, 2, 30, 6, 11, 8]


# ---------------------------------------------------------------------------------------------------------------
# Property 1 (THE load-bearing gate): numpy LinAttnReadout matches the real torch LinAttnLayer.forward.
# ---------------------------------------------------------------------------------------------------------------
class TestExactParityAgainstTorchForward:
    def test_double_precision_parity_2layer_default_config(self, tmp_path, readout_mod):
        """The primary gate: n_layers=2, phi=elu, norm=True (the P0 deployable-checkpoint recipe's own config,
        design doc §3e), compared at float64 precision -- isolates the transcription algebra from float32
        rounding, so any residual gap is a REAL algorithmic divergence, not fp noise."""
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_linattn_net(n_layers=2, phi="elu", norm=True)
        assert len(net.linattn_layers) == 2

        torch_logits = _walk_torch(net, TOKSEQ, double=True)

        ckpt_path = tmp_path / "dummy_linattn_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))

        ro = readout_mod.LinAttnReadout(str(ckpt_path), phi="elu", norm=True)
        assert len(ro.layers) == 2
        np_logits = _walk_numpy(ro, TOKSEQ)

        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-6, f"double-precision transcription mismatch (max diff {diff.max()})"
        assert (np_logits.argmax(1) == torch_logits.argmax(1)).all(), "argmax disagreement vs torch"

    def test_float32_cross_precision_matches_task_tolerance(self, tmp_path, readout_mod):
        """Realistic deployment precision (torch stays float32, the numpy readout is float64) -- the task's own
        <1e-4 ask, reported directly (not the harder double-precision bar above)."""
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_linattn_net(n_layers=2, phi="elu", norm=True)
        torch_logits = _walk_torch(net, TOKSEQ, double=False)

        ckpt_path = tmp_path / "dummy_linattn_fp32_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = readout_mod.LinAttnReadout(str(ckpt_path), phi="elu", norm=True)
        np_logits = _walk_numpy(ro, TOKSEQ)

        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-3, f"float32 cross-precision mismatch (max diff {diff.max()})"
        assert (np_logits.argmax(1) == torch_logits.argmax(1)).all(), "argmax disagreement vs torch"

    @pytest.mark.parametrize("phi", ["elu", "relu", "exp", "sparse"])
    def test_double_precision_parity_across_phi_kinds(self, tmp_path, readout_mod, phi):
        """`--linattn-phi` selects a different non-negative feature map; each must transcribe correctly, not
        just the default 'elu'."""
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_linattn_net(n_layers=1, phi=phi, norm=True)
        torch_logits = _walk_torch(net, TOKSEQ, double=True)
        ckpt_path = tmp_path / f"dummy_linattn_phi_{phi}_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = readout_mod.LinAttnReadout(str(ckpt_path), phi=phi, norm=True)
        np_logits = _walk_numpy(ro, TOKSEQ)
        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-6, f"phi={phi}: transcription mismatch (max diff {diff.max()})"

    def test_double_precision_parity_no_norm_ablation(self, tmp_path, readout_mod):
        """`--no-linattn-norm` (norm=False): the raw unnormalized sum branch (`read = num`, no `/den`) -- the
        design's own "KEY ABLATION" must transcribe correctly too, not just the normalized default."""
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_linattn_net(n_layers=2, phi="elu", norm=False)
        torch_logits = _walk_torch(net, TOKSEQ, double=True)
        ckpt_path = tmp_path / "dummy_linattn_nonorm_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = readout_mod.LinAttnReadout(str(ckpt_path), phi="elu", norm=False)
        np_logits = _walk_numpy(ro, TOKSEQ)
        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-6, f"norm=False: transcription mismatch (max diff {diff.max()})"

    def test_double_precision_parity_assoc_gate(self, tmp_path, readout_mod):
        """`--assoc-gate` (the learned-retrieval-gate `Wg`) must also transcribe correctly."""
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_linattn_net(n_layers=2, phi="elu", norm=True, gate=True)
        torch_logits = _walk_torch(net, TOKSEQ, double=True)
        ckpt_path = tmp_path / "dummy_linattn_gate_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = readout_mod.LinAttnReadout(str(ckpt_path), phi="elu", norm=True)
        assert ro.layers[0]["Wg"] is not None, "expected --assoc-gate Wg weights to be present and loaded"
        np_logits = _walk_numpy(ro, TOKSEQ)
        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-6, f"--assoc-gate: transcription mismatch (max diff {diff.max()})"

    def test_double_precision_parity_uniform_decay(self, tmp_path, readout_mod):
        """`--uniform-decay` (a scalar shared `lam` instead of per-channel) must broadcast correctly through the
        numpy transcription too."""
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_linattn_net(n_layers=1, phi="elu", norm=True, uniform_decay=True)
        torch_logits = _walk_torch(net, TOKSEQ, double=True)
        ckpt_path = tmp_path / "dummy_linattn_unidecay_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = readout_mod.LinAttnReadout(str(ckpt_path), phi="elu", norm=True)
        assert ro.layers[0]["lam"].shape == (1,)
        np_logits = _walk_numpy(ro, TOKSEQ)
        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-6, f"--uniform-decay: transcription mismatch (max diff {diff.max()})"

    def test_state_dict_key_layout_matches_documented_contract(self):
        """Pins the exact key layout `LinAttnReadout`'s docstring claims -- a future trainer refactor that
        renames a submodule would show up here as a missing/renamed key, not a silent misread."""
        net, V, D = _build_tiny_linattn_net(n_layers=2, phi="elu", norm=True)
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        expected_prefixes = {"emb.weight", "ln.weight", "ln.bias", "head.weight", "head.bias"}
        for i in range(2):
            expected_prefixes |= {
                f"linattn_layers.{i}.ln.weight", f"linattn_layers.{i}.ln.bias",
                f"linattn_layers.{i}.Wq.weight", f"linattn_layers.{i}.Wk.weight",
                f"linattn_layers.{i}.Wv.weight", f"linattn_layers.{i}.Wr.weight",
                f"linattn_layers.{i}.Wo.weight", f"linattn_layers.{i}.w",
            }
        assert expected_prefixes <= set(sd.keys()), f"missing keys: {expected_prefixes - set(sd.keys())}"
        # the base (unused-on-this-branch) keys must ALSO be present -- LinAttnReadout must ignore them, not error
        assert "Wk.weight" in sd and "Wo_sp.weight" in sd and "u" in sd


# ---------------------------------------------------------------------------------------------------------------
# Property 2: fail loud on the wrong checkpoint family, never silently guess.
# ---------------------------------------------------------------------------------------------------------------
class TestFailsLoudNotSilentOnWrongCheckpointFamily:
    def test_raises_on_a_wkv_family_checkpoint(self, tmp_path, readout_mod):
        """A `--recurrence wkv` (or ssm/dual-nonneg) checkpoint has no `linattn_layers.*` keys -- reading it as
        linattn must raise, never silently produce near-uniform garbage (the 2026-07-20 ROOT-CAUSE class)."""
        torch = pytest.importorskip("torch")
        from research.runners._emerge_wkv_lm_derisk import build_and_train_wkv
        args = types.SimpleNamespace(
            d_model=8, n_layers=1, recurrence="wkv", uniform_decay=False, freeze_emb=False,
            lr=3e-3, weight_decay=1e-4, batch=8, epochs=0, compile=False,
        )
        tr_ids = [[1, 2, 3, 4, 5]]
        net, _WKV_cls = build_and_train_wkv(tr_ids, 20, 42, args, device="cpu")
        net.eval()
        ckpt_path = tmp_path / "dummy_wkv_seed42.npz"
        _save_ssm_style(net, 20, 8, str(ckpt_path))
        with pytest.raises(RuntimeError, match="linattn_layers"):
            readout_mod.LinAttnReadout(str(ckpt_path))


# ---------------------------------------------------------------------------------------------------------------
# Property 3: resource budget (CPU-only, small memory) -- same discipline as the multi-layer wkv readout's pin.
# ---------------------------------------------------------------------------------------------------------------
class TestResourceBudget:
    def test_full_pipeline_stays_small_and_fast(self, tmp_path, readout_mod):
        import resource
        import time

        pytest.importorskip("torch")
        t0 = time.time()
        net, V, D = _build_tiny_linattn_net(n_layers=2)
        ckpt_path = tmp_path / "dummy_linattn_budget_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = readout_mod.LinAttnReadout(str(ckpt_path))
        state = ro.init_state()
        for t in TOKSEQ:
            state = ro.advance(state, t)
            ro.logits(state, t)
        elapsed = time.time() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        assert elapsed < 60.0, f"build+save+load+read took {elapsed:.1f}s -- unexpectedly slow"
        assert peak_rss_mb < 4096.0, f"peak RSS {peak_rss_mb:.0f}MB exceeds the 4GB CPU-lane budget"
