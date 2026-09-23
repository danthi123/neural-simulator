"""D6 learn-through-use: the in-conversation fact WRITE by a local Hebbian rule (research/runners/d6_hebbian_store.py).

Pins: (1) OFF (unset) is byte-identical to the PINNED pre-D6 commit's direct composite copy (exact sha256, reference
run from a git archive of d6_offpath_parity.PRE_D6_REF -- a fixed SHA, not origin/main, which would be tautological
after the merge; a reference that contains D6 reads UNDEFINED -- the unset-vs-'0' check is kept only as a labelled
integrity smoke); (2) the Hebbian-written block recalls with the
direct path's answers and phase (content = the rule's output, not a copy); (3) the FREEZE lesion blocks ONLY
in-conversation writes -- the taught fact is gone, the build-time fact still recalls, the frozen weight is 0;
(4) the freeze flag without the conversation context does not freeze; (5) the Hebbian write is deterministic;
(6) the D6 probe's verdict fails in every failing direction (runner selftest). numpy, tiny vocab (~1 min total).
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

VOCAB = sorted({"dog", "chase", "cat", "eat", "fish", "wolf", "hunt", "deer", "fox", "berry", "bird", "worm"})


def _build(store=None, freeze=None, seed=42):
    from research.runners.one_brain_composer import OneBrainComposer
    for k, v in (("BRAIN_D6_HEBBIAN_STORE", store), ("BRAIN_D6_HEBBIAN_FREEZE", freeze)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, vocab_headroom=2)


def _block(c, i):
    return np.array([complex(w) for (_p, _q, w) in c.store_conns[i * c.D:(i + 1) * c.D]])


@pytest.fixture(scope="module")
def arms():
    from research.runners import d6_hebbian_store as d6
    out = {}
    c = _build(None); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["unset"] = c
    c = _build("0"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["off"] = c
    c = _build("1"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["hebb"] = c
    c = _build("1"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["hebb_rep"] = c
    c = _build("1", "1"); c.hear("dog chase cat")
    with d6.conversation_write(c):
        c.hear("wolf hunt deer")
    out["frozen_conv"] = c
    c = _build("1", "1"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["frozen_no_ctx"] = c
    for k in ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE"):
        os.environ.pop(k, None)
    return out


def test_unset_vs_zero_same_branch_integrity_smoke(arms):
    """INTEGRITY SMOKE, not a byte-identical test: unset and '0' take the identical code path on this branch, so this
    cannot fail by construction. The real off-path check vs the pre-change code is
    `test_off_is_byte_identical_vs_origin_main` below (2026-09-23 fix round)."""
    assert arms["unset"].store_conns == arms["off"].store_conns


def _git_ok(ref):
    import subprocess
    try:
        return subprocess.run(["git", "cat-file", "-e", ref + "^{commit}"], capture_output=True).returncode == 0
    except Exception:
        return False


def _pre_d6_ref():
    from research.runners.d6_offpath_parity import PRE_D6_REF
    return PRE_D6_REF


@pytest.mark.skipif(not _git_ok(_pre_d6_ref()), reason="pinned pre-D6 commit not in this clone (shallow/CI): UNDEFINED")
def test_off_is_byte_identical_vs_pinned_pre_d6():
    """BYTE-IDENTICAL (docs/TERMS.md): with every BRAIN_D6_* flag unset, this branch's store_conns (exact complex values,
    sha256), kb and recalls equal those produced by the PINNED pre-D6 commit's code (PRE_D6_REF, a fixed SHA -- not
    origin/main, which contains D6 after the merge and would compare the branch with itself), run from a `git archive`
    in a separate process. And the comparison CAN fail: the Hebbian write (flag on) hashes differently."""
    from research.runners import d6_offpath_parity as P
    res = P.compare(P.PRE_D6_REF, "store")
    assert res["reference_is_pre_change"] is True
    assert res["reference"]["d6_module_present"] is False                     # the reference really is the pre-D6 path
    assert res["byte_identical"] is True, res["diff_keys"]
    os.environ["BRAIN_D6_HEBBIAN_STORE"] = "1"
    try:
        from research.runners.one_brain_composer import OneBrainComposer
        c = OneBrainComposer(seed=42, D=128, vocab=P.VOCAB, k_max=8, vocab_headroom=2)
        for s in ("dog chase cat", "wolf hunt deer", "fox eat berry"):
            c.hear(s)
        assert P._sha_store(c.store_conns) != res["reference"]["store_conns_sha256"]   # discriminates
    finally:
        os.environ.pop("BRAIN_D6_HEBBIAN_STORE", None)


@pytest.mark.skipif(not _git_ok("HEAD"), reason="no git checkout")
def test_parity_refuses_a_reference_that_already_contains_d6():
    """The tautology guard: compared against a ref that already has the D6 module (HEAD here; origin/main after the
    merge), the tool must read UNDEFINED -- the pre-fix test PASSED in exactly this situation (fix round 3 proof)."""
    from research.runners import d6_offpath_parity as P
    assert P.ref_contains_d6("HEAD", ".") is True
    if _git_ok(P.PRE_D6_REF):
        assert P.ref_contains_d6(P.PRE_D6_REF, ".") is False
    res = P.compare("HEAD", "store")
    assert res["byte_identical"] is None and res["reference_is_pre_change"] is False


def test_remove_block_record_removes_only_a_zero_record():
    """NOREC_H's experimenter control removes a HOST RECORD, never synaptic content: it refuses a potentiated block,
    removes a frozen (all-zero) last block, and leaves every other recall intact."""
    from research.runners import d6_hebbian_store as d6
    try:
        c = _build("1", "1"); c.hear("dog chase cat")
        with d6.conversation_write(c):
            c.hear("wolf hunt deer")                                   # frozen -> the block's synapses are exactly 0
        with pytest.raises(ValueError):
            d6.remove_block_record(c, 0)                               # not the last block
        r = d6.remove_block_record(c, 1)
        assert r["removed"] is True and len(c.kb) == 1 and len(c.store_conns) == c.D
        assert c.query_patient("dog", "chase") == "cat" and c.query_patient("wolf", "hunt") is None
        h = _build("1"); h.hear("dog chase cat"); h.hear("wolf hunt deer")
        with pytest.raises(ValueError):
            d6.remove_block_record(h, 1)                               # potentiated: holds content -> refused
        assert len(h.kb) == 2
    finally:
        for k in ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE"):
            os.environ.pop(k, None)


def test_visible_kb_off_is_the_same_object_and_on_follows_the_engram(arms):
    from research.runners import d6_hebbian_store as d6
    f = arms["frozen_conv"]
    os.environ.pop("BRAIN_D6_ENGRAM_READTIME", None)
    assert d6.visible_kb(f) is f.kb                                     # off: the host list itself (byte-identical)
    os.environ["BRAIN_D6_ENGRAM_READTIME"] = "1"
    try:
        vis = d6.visible_kb(f)
        assert [e[0]["agent"] for e in vis] == ["dog"]                  # the frozen (never-potentiated) block is not held
        assert len(f.kb) == 2                                           # ... and NO host record was deleted
    finally:
        os.environ.pop("BRAIN_D6_ENGRAM_READTIME", None)


def test_readtime_view_reflects_a_post_hoc_ablation():
    """The read-time view is re-read when the synapses change: ablating a learned block after the write (the ABL_H
    lesion) removes it from the view with the kb record intact, and the recall abstains. A write-time check (the
    banked prune) could not see this."""
    from research.runners import d6_hebbian_store as d6
    os.environ["BRAIN_D6_ENGRAM_READTIME"] = "1"
    try:
        c = _build("1"); c.hear("dog chase cat")
        with d6.conversation_write(c):
            c.hear("wolf hunt deer")
        assert [e[0]["agent"] for e in d6.visible_kb(c)] == ["dog", "wolf"]
        rec = d6.ablate_block(c, 1)
        assert rec["mean_abs_w_before"] > 0.5 and rec["mean_abs_w_after"] == 0.0
        assert [e[0]["agent"] for e in d6.visible_kb(c)] == ["dog"] and len(c.kb) == 2
        assert c.query_patient("wolf", "hunt") is None and c.query_patient("dog", "chase") == "cat"
        assert c._d6_ops["retractions"] == 0 and c._d6_ops["ablations"] == 1
    finally:
        for k in ("BRAIN_D6_ENGRAM_READTIME", "BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE"):
            os.environ.pop(k, None)


def test_hebbian_block_recalls_and_matches_direct_phase(arms):
    h, d = arms["hebb"], arms["unset"]
    assert h.query_patient("dog", "chase") == "cat"
    assert h.query_patient("wolf", "hunt") == "deer"
    assert h.query_patient("fox", "eat") is None                  # moat: nothing stored
    for i in range(2):
        dphi = np.angle(_block(h, i) * np.conj(_block(d, i)))
        assert float(np.mean(np.abs(dphi))) < 0.05                 # phase = the rule's output ~= the composite
    assert h._d6_last_encode["frozen"] is False and h._d6_last_encode["n_saturated"] == h.D


def test_freeze_blocks_only_the_in_conversation_write(arms):
    f = arms["frozen_conv"]
    assert f._d6_last_encode["frozen"] is True
    assert float(np.max(np.abs(_block(f, 1)))) == 0.0            # the lever moved: no weight change
    assert float(np.mean(np.abs(_block(f, 0)))) > 0.5             # the build-time block was written
    assert f.query_patient("wolf", "hunt") is None                # taught-in-conversation fact is gone
    assert f.query_patient("dog", "chase") == "cat"               # read path intact


def test_engram_held_reads_the_substrate(arms):
    from research.runners.d6_hebbian_store import engram_held
    f, h = arms["frozen_conv"], arms["hebb"]
    r_frozen, r_built = engram_held(f, 1), engram_held(f, 0)
    assert r_frozen["held"] is False and r_frozen["readout"] == 0.0     # never-potentiated block: no engram
    assert r_built["held"] is True and r_built["readout"] > 100 * r_built["floor"]
    assert all(engram_held(h, i)["held"] for i in range(2))
    assert f.query_patient("dog", "chase") == "cat"                     # the read left recall intact


def test_engram_prune_retracts_an_unencoded_conversation_write():
    from research.runners import d6_hebbian_store as d6
    os.environ["BRAIN_D6_ENGRAM_PRUNE"] = "1"
    try:
        c = _build("1", "1"); c.hear("dog chase cat")
        with d6.conversation_write(c):
            c.hear("wolf hunt deer")                                   # frozen -> no engram -> retracted
        assert c._d6_last_retract["retracted"] is True
        assert len(c.kb) == 1 and len(c.store_conns) == c.D
        assert c.query_patient("wolf", "hunt") is None and c.query_patient("dog", "chase") == "cat"
        c.hear("fox eat berry")                                         # build-time write reuses the freed block
        assert c.query_patient("fox", "eat") == "berry" and c.query_patient("dog", "chase") == "cat"
    finally:
        for k in ("BRAIN_D6_ENGRAM_PRUNE", "BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE"):
            os.environ.pop(k, None)


def test_freeze_without_conversation_context_does_not_freeze(arms):
    f = arms["frozen_no_ctx"]
    assert f.query_patient("wolf", "hunt") == "deer"


def test_hebbian_write_is_deterministic(arms):
    assert arms["hebb"].store_conns == arms["hebb_rep"].store_conns


def test_d6_probe_verdict_selftest():
    from research.runners.d6_learn_through_use_lb import selftest
    assert selftest() == 0
