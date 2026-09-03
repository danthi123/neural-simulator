"""CI guard for the gap#1 WKV open-generation arc (2026-07-19): the mission-primary lever that removes the non-fading-store
wall. Lightweight, CPU/numpy, offline; skips the torch-training assertion if torch is unavailable. Guards the load-bearing
pieces so the arc's result is protected for the continuation:
  - the runner + the fair interpolated trigram + PPMI-code builder + contiguous-story loader import + run;
  - the WKV op TRAINS and clears CHANCE at a small scale (the mechanism works), with the perm/memoryless anti-cheat
    controls wired (they must degrade the deep-context read).
The FULL result (WKV beats the FAIR interpolated trigram at deep context, 6-seed, at scale) is the runner's own GPU sweep
(`_emerge_wkv_lm_derisk.py`), not this fast CI smoke — this only pins that the code is sound + the mechanism clears chance.
"""
import math
import numpy as np
import pytest


def test_bpe_vocab_adapter_is_a_drop_in_for_vocab_interface(tmp_path):
    """Subword-tokenizer swap (additive, --tokenizer bpe, 2026-09-02): the trainer was word-vocab-only (<unk>-riddled
    on arbitrary chat topics) -- _BPEVocabAdapter wraps a loaded BPETokenizer with Vocab's own interface
    (.ids/.i2w/.w2i/.unk/.size) so fit_bigram/fit_interp_trigram/build_and_train_wkv/--save-ssm/--generate need ZERO
    changes. Pins: the adapter's surface matches the tokenizer, and --tokenizer defaults to 'word' (the pre-swap
    default path stays byte-identical -- verified separately by SHA256-identical --save-ssm checkpoints, see the
    2026-09-02 landing commit)."""
    from sim.bpe_tokenizer import BPETokenizer
    from research.runners._emerge_wkv_lm_derisk import _BPEVocabAdapter, DEFAULT_BPE_PATH
    corpus = "the cat sat on the mat . the cat ran away . " * 30
    tok = BPETokenizer(); tok.train(corpus, vocab_size=60)
    p = tmp_path / "bpe.json"; tok.save(str(p))
    loaded = BPETokenizer.load(str(p))
    adapter = _BPEVocabAdapter(loaded)
    assert adapter.size == loaded.vocab_size == len(adapter.i2w)
    assert adapter.i2w == loaded.vocab
    assert adapter.unk == adapter.w2i["<UNK>"]
    s = ["the", "cat", "sat"]
    assert adapter.ids(s) == loaded.encode(" ".join(s))                    # exact re-use of the tokenizer's own contract
    assert DEFAULT_BPE_PATH == "bridges/wkv_ckpt/wkv_bpe8k.json"


def test_tokenizer_cli_flag_defaults_to_word(monkeypatch):
    """--tokenizer must default to 'word' -- the additive-swap guarantee (byte-identical when the flag is unused)."""
    import argparse
    from research.runners import _emerge_wkv_lm_derisk as mod
    captured = {}
    real_parse_args = argparse.ArgumentParser.parse_args
    def fake_parse_args(self, *a, **kw):
        ns = real_parse_args(self, ["--seeds", "0"])   # only what's needed to populate defaults
        captured["tokenizer"] = ns.tokenizer
        captured["bpe_path"] = ns.bpe_path
        raise SystemExit(0)   # bail before any actual training work
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    try:
        mod.main()
    except SystemExit:
        pass
    assert captured["tokenizer"] == "word"
    assert captured["bpe_path"] == mod.DEFAULT_BPE_PATH


def test_fair_trigram_and_ppmi_and_loader_import_and_run():
    from research.runners._emerge_wkv_lm_derisk import fit_interp_trigram, build_ppmi_codes, load_stories, _bucket
    V = 30
    rng = np.random.default_rng(0)
    ids = [list(rng.integers(0, V, size=rng.integers(6, 16))) for _ in range(200)]
    tri, lam = fit_interp_trigram(ids, V, ids[:40])
    assert abs(sum(lam) - 1.0) < 1e-6 and all(l >= 0 for l in lam)          # a proper interpolation
    p = tri(ids[0][0], ids[0][1], ids[0][2])
    assert 0.0 <= p <= 1.0
    codes = build_ppmi_codes(ids, V, d=16, window=3)
    assert codes.shape == (V, 16)
    assert np.allclose(np.linalg.norm(codes, axis=1), 1.0, atol=1e-4)       # unit-normalized emergent codes
    assert _bucket(15) == "10-99" and _bucket(1) == "1"                     # deep-context bucketing


def test_wkv_op_trains_and_clears_chance_with_anticheats():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from research.runners._emerge_wkv_lm_derisk import build_and_train_wkv
    # a tiny LEARNABLE sequence task: token t+1 depends on token t-3 (a d=4 dependency the WKV recurrence can carry,
    # a memoryless/bigram model cannot) -> the WKV must clear chance AND the memoryless control must be worse.
    V = 12
    rng = np.random.default_rng(1)
    seqs = []
    for _ in range(600):
        s = list(rng.integers(0, V, size=12))
        for t in range(4, len(s)):
            s[t] = (s[t - 4] + 1) % V                                       # long-range (lag-4) deterministic dependency
        seqs.append(s)
    args = SimpleNamespace(d_model=32, epochs=6, batch=64, lr=3e-3, weight_decay=1e-4, recurrence="wkv",
                           input="learned", spiking_state=False, uniform_decay=False)
    net, _ = build_and_train_wkv(seqs, V, seed=1, args=args, device="cpu")

    def deep_nll(memoryless):
        net.memoryless = memoryless
        ce = n = 0
        with torch.no_grad():
            for s in seqs[:200]:
                logp = torch.log_softmax(net(torch.tensor([s]))[0], -1).numpy()
                for t in range(4, len(s) - 1):
                    ce += -logp[t, s[t + 1]]; n += 1
        net.memoryless = False
        return ce / n
    wkv = deep_nll(False); mless = deep_nll(True); chance = math.log(V)
    assert wkv < chance - 0.3, f"WKV should clear chance on the lag-4 task (wkv {wkv:.3f} vs chance {chance:.3f})"
    assert mless > wkv + 0.1, f"memoryless control must be WORSE (mless {mless:.3f} vs wkv {wkv:.3f}) = the recurrence is load-bearing"


def test_spikegpt_faithful_spike_output_still_learns():
    """SpikeGPT-faithful architecture (GRADED state + SPIKE-CODED output y_t via straight-through) still carries the lag-4
    dependency = the output binarization is absorbed end-to-end (the 6-seed-GO turnaround: spiked I/O + graded local state)."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from research.runners._emerge_wkv_lm_derisk import build_and_train_wkv
    V = 12
    rng = np.random.default_rng(2)
    seqs = []
    for _ in range(600):
        s = list(rng.integers(0, V, size=12))
        for t in range(4, len(s)):
            s[t] = (s[t - 4] + 1) % V
        seqs.append(s)
    args = SimpleNamespace(d_model=32, epochs=8, batch=64, lr=3e-3, weight_decay=1e-4, recurrence="ssm",
                           input="learned", spiking_state=True, uniform_decay=True, quantize_state=False,
                           spike_output=True, t_step=6)
    net, _ = build_and_train_wkv(seqs, V, seed=2, args=args, device="cpu")
    ce = n = 0
    with torch.no_grad():
        for s in seqs[:200]:
            logp = torch.log_softmax(net(torch.tensor([s]))[0], -1).numpy()
            for t in range(4, len(s) - 1):
                ce += -logp[t, s[t + 1]]; n += 1
    wkv = ce / n; chance = math.log(V)
    # the output binarization costs some margin (the 6-seed-GO result: it eats a thin small-scale margin but SCALES); on this
    # tiny task it still clearly clears chance, confirming the spike-coded-output architecture carries the dependency.
    assert wkv < chance - 0.1, f"spike-output WKV should still clear chance (wkv {wkv:.3f} vs chance {chance:.3f})"
