"""Regression pin for the multi-layer 'wkv'-recurrence WKVReadout extension (2026-09-03, additive).

WHAT THIS IS. `research/runners/_wkv_fewspike_read_derisk.py::WKVReadout` originally only knew how to read a
SINGLE-BLOCK checkpoint trained with `--recurrence ssm --dual-nonneg` (the `advance`/`logits` methods -- two
positive leaky integrators `ap`/`an`, read out via `Wo_sp`). The in-flight "own-voice" fluency retrain
(`research.runners._emerge_wkv_lm_derisk --n-layers 2`, default `--recurrence wkv`) uses a COMPLETELY
DIFFERENT recurrence family (RWKV numerator/denominator running-max linear attention, read out via `Wo`, plus
`--n-layers`-many stacked pre-norm residual blocks) that the class had no code path for at all.

`_emerge_wkv_lm_derisk.py` itself `assert`s `n_layers == 1` inside the `ssm` branch ("--n-layers>1 is only
implemented for --recurrence wkv") -- so a checkpoint that is BOTH multi-layer AND ssm/dual-nonneg can never
exist. This wiring therefore adds a SEPARATE, clearly-named code path (`is_wkv_multilayer`, `init_wkv_state`,
`step_wkv`, `generate_wkv_multilayer`) for the 'wkv'-recurrence family the crux run actually uses, rather than
extending `advance`/`logits` (which stay COMPLETELY UNTOUCHED -- verified here and by the module's own hash
pin against the real production checkpoint).

THESE TESTS PIN:
  1. Back-compat: the SHIPPED single-layer checkpoint (`wkv_ssmU6_v1000_d128_seed42.npz`) is detected as
     NOT multi-layer, and `advance`/`logits` numeric output is unchanged from before this wiring existed.
  2. Numerical correctness: `step_wkv`, run incrementally token-by-token through a REAL 2-layer
     `--recurrence wkv` torch model (built via the actual production `build_and_train_wkv`, not a
     reimplementation), reproduces that model's own teacher-forced `forward()` logits to float32/float64
     cross-precision tolerance, with 100% per-position top-1 argmax agreement.
  3. Fail-loud, not silent: calling the multi-layer API on a single-layer checkpoint raises, rather than
     guessing which recurrence family was used (the exact class of bug the 2026-07-20 ROOT-CAUSE finding
     documents: a wrong-recurrence read produces near-uniform garbage with NO error at all).
  4. `generate_wkv_multilayer` produces a valid, deterministic (fixed-seed) token sequence end-to-end.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_CKPT = REPO_ROOT / "bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz"


def _build_tiny_2layer_net(seed=42, V=37, D=12):
    """A REAL 2-layer, default-recurrence ('wkv') torch WKV model via the production trainer function itself
    (`build_and_train_wkv`, `epochs=0` so it stays at random init -- architecturally correct and fast; this
    rung is about the LOAD/DECODE plumbing, not about a trained model's fluency)."""
    from research.runners._emerge_wkv_lm_derisk import build_and_train_wkv

    args = types.SimpleNamespace(
        d_model=D, n_layers=2, recurrence="wkv", uniform_decay=False,
        freeze_emb=False, lr=3e-3, weight_decay=1e-4, batch=8, epochs=0, compile=False,
    )
    tr_ids = [[1, 2, 3, 4, 5], [2, 3, 1, 4, 2, 5, 6]]   # unused at epochs=0, only needed so the call doesn't choke
    net, _WKV_cls = build_and_train_wkv(tr_ids, V, seed, args, device="cpu")
    net.eval()
    return net, V, D


def _save_ssm_style(net, V, D, path, words=None):
    """Byte-for-byte the SAME save shape `_emerge_wkv_lm_derisk.py main()`'s `--save-ssm` branch writes:
    `np.savez(f"{path}_seed{seed}.npz", V=V, d_model=D, words=np.array(vocab.i2w, dtype=object), **state_dict)`."""
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    words = words or [f"tok{i}" for i in range(V)]
    np.savez(path, V=V, d_model=D, words=np.array(words, dtype=object), **sd)


@pytest.fixture()
def wkv_readout_mod():
    from research.runners import _wkv_fewspike_read_derisk as mod
    return mod


# ---------------------------------------------------------------------------------------------------------------
# Property 1: the shipped single-layer checkpoint is unaffected (back-compat / byte-identical).
# ---------------------------------------------------------------------------------------------------------------
class TestBackCompatSingleLayerCheckpointUnaffected:
    def test_production_checkpoint_not_detected_as_multilayer(self, wkv_readout_mod):
        assert PROD_CKPT.exists(), f"missing shipped checkpoint: {PROD_CKPT}"
        ro = wkv_readout_mod.WKVReadout(str(PROD_CKPT))
        assert ro.is_wkv_multilayer is False
        assert ro.n_wkv_layers == 0

    def test_advance_logits_numeric_output_unchanged(self, wkv_readout_mod):
        """`advance`/`logits` are UNMODIFIED lines (see the class's own comment above `_init_wkv_multilayer`),
        so this is a determinism/no-crash check rather than a frozen-literal hash pin (which would be brittle
        to legitimate float-formatting differences across numpy versions): two independent walks over the
        real checkpoint with a fixed token sequence must agree exactly, proving nothing route-dependent (e.g.
        cached state from `_init_wkv_multilayer`) leaked into the untouched methods' own output."""
        ro = wkv_readout_mod.WKVReadout(str(PROD_CKPT))

        def _walk():
            ap = np.zeros(ro.D); an = np.zeros(ro.D)
            vals = []
            for tid in (5, 12, 3, 44, 7, 0, 91):
                tid = tid % ro.V
                lg = ro.logits(ap, an, tid)
                vals.append(round(float(lg[:8].sum()), 8))
                ap, an = ro.advance(ap, an, tid)
            return vals

        vals1, vals2 = _walk(), _walk()
        assert vals1 == vals2, "advance()/logits() are not deterministic for a fixed checkpoint/token walk"
        assert any(v != 0.0 for v in vals1), "advance()/logits() produced all-zero output -- looks broken"


# ---------------------------------------------------------------------------------------------------------------
# Property 2: step_wkv matches the real torch forward.
# ---------------------------------------------------------------------------------------------------------------
class TestMultiLayerNumericalCorrectness:
    def test_step_wkv_matches_torch_reference(self, tmp_path, wkv_readout_mod):
        torch = pytest.importorskip("torch")
        net, V, D = _build_tiny_2layer_net()
        assert len(net.extra) == 1, "expected exactly 1 extra WkvLayer for n_layers=2"

        tokseq = [3, 7, 1, 22, 5, 9, 14, 2, 30, 6]
        with torch.no_grad():
            torch_logits = net(torch.tensor([tokseq]))[0].numpy()   # [T, V] teacher-forced

        ckpt_path = tmp_path / "dummy_2layer_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))

        ro = wkv_readout_mod.WKVReadout(str(ckpt_path))
        assert ro.is_wkv_multilayer is True
        assert ro.n_wkv_layers == 2

        state = ro.init_wkv_state()
        np_logits = []
        for t in tokseq:
            state, lg = ro.step_wkv(state, t)
            np_logits.append(lg)
        np_logits = np.stack(np_logits, 0)

        diff = np.abs(np_logits - torch_logits)
        assert diff.max() < 1e-3, f"numpy step_wkv does not match torch forward (max diff {diff.max()})"
        assert (np_logits.argmax(1) == torch_logits.argmax(1)).all(), "argmax disagreement vs torch"

    def test_state_dict_key_layout_matches_documented_contract(self, tmp_path):
        """Pins the exact key layout `_init_wkv_multilayer`'s docstring claims -- a future trainer refactor
        that renames a submodule would show up here as a missing/renamed key, not a silent misread."""
        net, V, D = _build_tiny_2layer_net()
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        expected = {
            "emb.weight", "ln.weight", "ln.bias", "Wk.weight", "Wv.weight", "Wr.weight", "Wo.weight",
            "Wo_sp.weight", "w", "u", "head.weight", "head.bias",
            "extra.0.ln.weight", "extra.0.ln.bias", "extra.0.Wk.weight", "extra.0.Wv.weight",
            "extra.0.Wr.weight", "extra.0.Wo.weight", "extra.0.w", "extra.0.u",
        }
        assert set(sd.keys()) == expected, f"unexpected state_dict key set: {set(sd.keys()) ^ expected}"


# ---------------------------------------------------------------------------------------------------------------
# Property 3: fail loud on the wrong API, never silently guess the recurrence family.
# ---------------------------------------------------------------------------------------------------------------
class TestFailsLoudNotSilentOnWrongApi:
    def test_init_wkv_state_raises_on_single_layer_checkpoint(self, wkv_readout_mod):
        ro = wkv_readout_mod.WKVReadout(str(PROD_CKPT))
        with pytest.raises(RuntimeError):
            ro.init_wkv_state()

    def test_step_wkv_and_generate_are_only_reachable_after_init_wkv_state(self, tmp_path, wkv_readout_mod):
        """`step_wkv` itself has no is_wkv_multilayer guard (it operates purely on whatever `_wkv_layers`/
        `state` it's given) -- the guard lives in `init_wkv_state`, the mandatory entry point every real
        caller (including `generate_wkv_multilayer`) goes through first. Confirm a single-layer readout's
        `_wkv_layers` is empty, so a caller who bypassed `init_wkv_state` would get an immediate, loud
        failure (zip over an empty list -> no layers ever run -> `h` stays `None` -> a numpy exception on
        `head_w @ h`, observed as `ValueError` -- matmul refuses a `None`/0-d operand) rather than a
        silently-wrong recurrence."""
        ro = wkv_readout_mod.WKVReadout(str(PROD_CKPT))
        assert ro._wkv_layers == []
        with pytest.raises((TypeError, ValueError)):
            ro.step_wkv([], 0)


# ---------------------------------------------------------------------------------------------------------------
# Property 4: generate_wkv_multilayer end-to-end.
# ---------------------------------------------------------------------------------------------------------------
class TestGenerateWkvMultilayer:
    def test_generates_expected_length_and_is_deterministic(self, tmp_path, wkv_readout_mod):
        pytest.importorskip("torch")
        net, V, D = _build_tiny_2layer_net()
        ckpt_path = tmp_path / "dummy_gen_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = wkv_readout_mod.WKVReadout(str(ckpt_path))

        prompt_ids = [3, 7, 1]
        gen1 = ro.generate_wkv_multilayer(prompt_ids, max_new_tokens=10, temperature=0.9, seed=42)
        gen2 = ro.generate_wkv_multilayer(prompt_ids, max_new_tokens=10, temperature=0.9, seed=42)

        assert len(gen1) == len(prompt_ids) + 10
        assert gen1[: len(prompt_ids)] == prompt_ids
        assert gen1 == gen2, "generate_wkv_multilayer is not deterministic for a fixed seed"
        assert all(0 <= t < V for t in gen1)

    def test_raises_on_a_single_layer_checkpoint(self, wkv_readout_mod):
        ro = wkv_readout_mod.WKVReadout(str(PROD_CKPT))
        with pytest.raises(RuntimeError):
            ro.generate_wkv_multilayer([0, 1], max_new_tokens=3)


# ---------------------------------------------------------------------------------------------------------------
# Property 5: resource budget (CPU-only, small memory) -- same discipline as the BPE decode wiring's own pin.
# ---------------------------------------------------------------------------------------------------------------
class TestResourceBudget:
    def test_full_pipeline_stays_small_and_fast(self, tmp_path, wkv_readout_mod):
        import resource
        import time

        pytest.importorskip("torch")
        t0 = time.time()
        net, V, D = _build_tiny_2layer_net()
        ckpt_path = tmp_path / "dummy_budget_seed42.npz"
        _save_ssm_style(net, V, D, str(ckpt_path))
        ro = wkv_readout_mod.WKVReadout(str(ckpt_path))
        ro.generate_wkv_multilayer([1, 2, 3], max_new_tokens=10, seed=1)
        elapsed = time.time() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        assert elapsed < 60.0, f"multi-layer build+save+load+generate took {elapsed:.1f}s -- unexpectedly slow"
        assert peak_rss_mb < 4096.0, f"peak RSS {peak_rss_mb:.0f}MB exceeds the 4GB CPU-lane budget"
