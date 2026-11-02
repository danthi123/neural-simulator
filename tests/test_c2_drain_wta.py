"""C2 match-unit smoke build (RULING c2-drain-wta-1101): quiet drain, role WTA, role relay, row shuffle, S0b acc logging,
pool lines and the scorer. Every flag default OFF.

(a) off-path byte-identity: a short reduced-sequencer run (5 consecutive queries on one persistent bridge, the production
    `run_sequencerK_reduced_with_drive`) with every flag unset hashes to GOLDEN_SEQ, computed by running the same fixture
    with `reset_sequencerK_state` from the pre-change code (25e360740); `make_block_drives` / `wta_drive` without the new
    kwargs return what they returned; the composer's flag-off drives keep the pre-change GOLDEN_DRIVES of
    test_c2_relay_patient_inhib.
(b) BRAIN_C2_QUIET_DRAIN=N runs N drain steps (recorded), =20 equals unset, and a long drain leaves no pending spikes.
(c) BRAIN_C2_ROLE_WTA / BRAIN_C2_ROLE_RELAY swap the agent/action drive source on the real `_ensure_sequencer` (spiking
    bridges, engagement recorded); their I->E lesion knobs keep the population and zero the weight; both set is refused.
(d) BRAIN_C2_ROW_SHUFFLE feeds block b row perm[b], a seeded derangement.
(e) read (b) records the flags, the drive source and accA/accX per block/role/word; the read guard removes the records.
(f) pool lines: 7 arms x 37, pinned, env inserted, out root moved; the scorer's frozen denominators and A0 bar on the
    committed gate-reset wave, and UNDEFINED on a missing or non-engaged shard.
"""
import copy
import hashlib
import json
import os
import shutil
import types

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _c2_patient_inhib as CI
from research.runners import _c2_role_drives as RD
from research.runners import _phaseB_onebrain_sequencerK_derisk as SK

FLAGS = ("BRAIN_C2_QUIET_DRAIN", "BRAIN_C2_ROLE_WTA", "BRAIN_C2_ROLE_WTA_I2E", "BRAIN_C2_ROLE_RELAY",
         "BRAIN_C2_ROLE_RELAY_I2E", "BRAIN_C2_ROW_SHUFFLE", "BRAIN_C2_GATE_RESET", "BRAIN_C2_COMPOSER_PATIENT_INHIB")

WORDS = ["a0", "a1", "a2", "x0", "x1", "x2", "zz"]
MAPA = {"a0": 0, "a1": 1, "a2": 2}
MAPX = {"x0": 0, "x1": 1, "x2": 2}
QUERIES = (("a0", "x0"), ("a1", "x1"), ("a0", "x1"), ("a2", "x2"), ("a0", "x2"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)


def _line(*ws):
    d = np.zeros(len(WORDS))
    for w in ws:
        d[WORDS.index(w)] = 1500.0
    return d


def seq_run_hash(drain_record=None):
    """m_rates of 5 consecutive queries on one reduced sequencer + the final v / prev_firing, hashed."""
    from research.runners._seq_vocab_shrink_derisk import build_sequencerK_reduced_bridge, run_sequencerK_reduced_with_drive
    from sim.backend import to_host
    sb, meta = build_sequencerK_reduced_bridge(seed=7, VA=3, VX=3, K=3)
    drives = [(_line("a0"), _line("x0")), (_line("a1", "a0"), _line("x1")), (_line("a2"), _line("x0", "x2"))]
    h = hashlib.sha256()
    rates = []
    for a, x in QUERIES:
        dec, r = run_sequencerK_reduced_with_drive(sb, meta, WORDS, MAPA, MAPX, a, x, drives, match_thresh=0.06)
        rates.append([r["m%d" % b] for b in range(3)])
        h.update(json.dumps([dec, rates[-1]]).encode())
        if drain_record is not None:
            drain_record.append(dict(getattr(sb, "_c2_drain_record", {}) or {}))
    for nm in ("cp_membrane_potential_v", "cp_prev_firing_states"):
        h.update(np.ascontiguousarray(np.asarray(to_host(getattr(sb, nm)), dtype=np.float64)).tobytes())
    return h.hexdigest()[:16], rates


# the pre-change code (25e360740 `reset_sequencerK_state`, monkeypatched into _seq_vocab_shrink_derisk) on this fixture,
# with BRAIN_C2_GATE_RESET unset / =1
GOLDEN_SEQ = {None: "07e8a2d18aa8ae75", "1": "4fba62af5b09b886"}


@pytest.mark.parametrize("gate_reset", [None, "1"])
def test_quiet_drain_unset_byte_identical_to_pre_change(monkeypatch, gate_reset):
    if gate_reset:
        monkeypatch.setenv("BRAIN_C2_GATE_RESET", gate_reset)
    h, rates = seq_run_hash()
    assert h == GOLDEN_SEQ[gate_reset]
    assert rates[2][1] > 0.06          # the fixture's both-lit false fire: cue (a0, x1) fires block 1 (a1+a0 lit)


def test_quiet_drain_20_equals_unset_and_long_drain_engages(monkeypatch):
    rec0 = []
    h0, r0 = seq_run_hash(rec0)
    assert all(r["drain_steps"] == 20 and r["quiet_drain_env"] is None for r in rec0)
    monkeypatch.setenv("BRAIN_C2_QUIET_DRAIN", "20")
    h20, _ = seq_run_hash()
    assert h20 == h0
    monkeypatch.setenv("BRAIN_C2_QUIET_DRAIN", "60")
    rec = []
    _h60, _ = seq_run_hash(rec)
    assert all(r["drain_steps"] == 60 and r["quiet_drain_env"] == 60 for r in rec)
    # pending spikes at the clear fall with the quiet period (measured on this fixture: 20 -> [183,51,101,77],
    # 60 -> [96,29,18,17], 400 -> all 0); the fabric's residual activity decays slowly, it is not gone at 60
    assert sum(r["prev_firing_at_clear"] for r in rec) < sum(r["prev_firing_at_clear"] for r in rec0)
    monkeypatch.setenv("BRAIN_C2_QUIET_DRAIN", "400")
    rec400 = []
    seq_run_hash(rec400)
    assert all(r["prev_firing_at_clear"] == 0 for r in rec400)     # a long enough quiet period silences it physically
    monkeypatch.setenv("BRAIN_C2_QUIET_DRAIN", "-1")
    with pytest.raises(ValueError):
        SK._quiet_drain_steps()


def test_make_block_drives_and_wta_drive_unchanged_without_new_kwargs():
    from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import (make_block_drives, wta_drive,
                                                                                build_wta_score_bridge)
    from tests.test_c2_relay_patient_inhib import _score_bridge, AG, AX, V
    bs = [(AG, AX), (AX, AG)]
    sb = _score_bridge()
    d0, l0 = make_block_drives(sb, V, bs, 1.0, "divnorm")
    accs = []
    d1, l1 = make_block_drives(sb, V, bs, 1.0, "divnorm", accs_out=accs)
    assert l0 == l1 and all(np.array_equal(a, b) for p, q in zip(d0, d1) for a, b in zip(p, q))
    assert [(int((a > 0).sum()), int((x > 0).sum())) for a, x in accs] == l0
    assert all(np.array_equal(a > 0, d > 0) for (a, x), (d, e) in zip(accs, d0))
    # wta_drive with the new kwargs off == the old call (two fresh bridges: the old function's reset is partial, so a
    # second read on the SAME bridge differs; that leak is what `full_reset` removes)
    w0 = wta_drive(build_wta_score_bridge(7, V, sigma=1.0, gain=0.11), V, AG, 1.0)
    st = {}
    w1 = wta_drive(build_wta_score_bridge(7, V, sigma=1.0, gain=0.11), V, AG, 1.0, stats=st)
    assert np.array_equal(w0[1], w1[1]) and st["reads"] == 1 and st["inh_spikes"] >= 0
    wsb = build_wta_score_bridge(7, V, sigma=1.0, gain=0.11)
    f0 = wta_drive(wsb, V, AG, 1.0, full_reset=True)[1]
    wta_drive(wsb, V, AX, 1.0, full_reset=True)
    assert np.array_equal(wta_drive(wsb, V, AG, 1.0, full_reset=True)[1], f0)     # no read-to-read leak


def _stub(monkeypatch, K=3):
    from tests.test_c2_relay_patient_inhib import stub_buf, build
    buf = stub_buf(K=K)
    calls, _ = build(monkeypatch, buf, K=K)
    return buf


def test_composer_flag_off_drives_and_records(monkeypatch):
    from tests.test_c2_relay_patient_inhib import GOLDEN_DRIVES, _drives_hash
    buf = _stub(monkeypatch, K=2)
    assert _drives_hash(buf._seq_drives) == GOLDEN_DRIVES
    assert buf._seq_role_info == {"mode": "divnorm", "row_shuffle_perm": None}
    assert len(buf._seq_drive_accs) == 2
    for (aA, aX), (dA, dX) in zip(buf._seq_drive_accs, buf._seq_drives):
        assert np.array_equal(aA > 0, dA > 0) and np.array_equal(aX > 0, dX > 0)
    assert not hasattr(buf, "_seq_role_inhib") and not hasattr(buf, "_seq_score_wta")


def test_role_relay_on_and_lesion(monkeypatch):
    from tests.test_c2_relay_patient_inhib import AG, V
    monkeypatch.setenv(CI.ROLE_RELAY_ENV, "1")
    buf = _stub(monkeypatch)
    ri = buf._seq_role_info
    assert ri["mode"] == "relay" and ri["w_i2e"] == CI.COMPOSER_KNOBS["w_i2e"] and ri["inh_spikes"] > 0
    assert ri["reads"] == 6 and ri["inh_spikes"] <= ri["inh_ceiling"]
    for (dA, dX), (aA, aX) in zip(buf._seq_drives, buf._seq_drive_accs):
        assert np.array_equal(dA > 0, aA > 0) and set(np.unique(dA)) <= {0.0, CI.PATIENT_HI_PA}
    want, _inh = CI.onbridge_inhib_drive(CI.composer_role_inhib_bridge(buf), V, AG, 1.0,
                                         settle=CI.COMPOSER_KNOBS["settle"], ff_gain=CI.COMPOSER_KNOBS["ff_gain"])
    assert np.array_equal(buf._seq_drive_accs[0][0], want)          # the drive IS the bridge's spiking
    assert not hasattr(buf, "_seq_patient_drives")                    # the patient relay is a separate flag
    on_lit = sum(int((a > 0).sum()) + int((x > 0).sum()) for a, x in buf._seq_drive_accs)
    monkeypatch.setenv(CI.ROLE_RELAY_I2E_ENV, "0")
    les = _stub(monkeypatch)
    assert les._seq_role_info["w_i2e"] == 0.0
    les_lit = sum(int((a > 0).sum()) + int((x > 0).sum()) for a, x in les._seq_drive_accs)
    assert les_lit > on_lit                                           # the inhibition is load-bearing on the fixture


def test_role_wta_runs_end_to_end(monkeypatch):
    from research.runners._seq_vocab_shrink_derisk import build_sequencerK_reduced_bridge, run_sequencerK_reduced_with_drive
    monkeypatch.setenv(RD.ROLE_WTA_ENV, "1")
    buf = _stub(monkeypatch)
    ri = buf._seq_role_info
    assert ri["mode"] == "wta" and ri["reads"] == 6 and ri["w_i2e"] == RD.WTA_KNOBS["w_inh_to_exc"]
    assert ri["inh_ceiling"] == RD.WTA_KNOBS["n_inh"] * RD.WTA_KNOBS["settle"] * 6
    sb = list(buf._seq_score_wta.values())[0]
    nw = buf.V * RD.WTA_KNOBS["n_word"]
    assert np.array_equal(np.asarray(sb.cp_izh_a)[:nw], np.asarray(buf._seq_score.cp_izh_a)[:nw])
    # the WTA drives feed the reduced sequencer (global word index -> reduced lines) without error
    words = ["w%d" % i for i in range(buf.V)]
    lit = sorted({words[j] for dA, dX in buf._seq_drives for j in np.flatnonzero((dA > 0) | (dX > 0))})
    mp = {w: i for i, w in enumerate(lit)}
    rsb, meta = build_sequencerK_reduced_bridge(seed=7, VA=len(mp), VX=len(mp), K=3)
    dec, rates = run_sequencerK_reduced_with_drive(rsb, meta, words, mp, mp, lit[0], lit[0], buf._seq_drives)
    assert dec in ["abstain"] + ["ans%d" % b for b in range(3)]
    monkeypatch.setenv(RD.ROLE_WTA_I2E_ENV, "0")
    les = _stub(monkeypatch)
    assert les._seq_role_info["w_i2e"] == 0.0


def test_both_role_flags_refused(monkeypatch):
    monkeypatch.setenv(RD.ROLE_WTA_ENV, "1")
    monkeypatch.setenv(CI.ROLE_RELAY_ENV, "1")
    with pytest.raises(ValueError):
        RD.role_drive_mode()


def test_row_shuffle_is_a_seeded_derangement(monkeypatch):
    base = _stub(monkeypatch, K=3)
    monkeypatch.setenv(RD.ROW_SHUFFLE_ENV, "1")
    sh = _stub(monkeypatch, K=3)
    perm = sh._seq_role_info["row_shuffle_perm"]
    assert sorted(perm) == [0, 1, 2] and all(perm[b] != b for b in range(3))
    assert perm == RD.row_shuffle_perm(7, 3)
    for b in range(3):
        for r in range(2):
            assert np.array_equal(sh._seq_drives[b][r], base._seq_drives[perm[b]][r])
    # the acc record stays indexed by the SOURCE row
    assert all(np.array_equal(sh._seq_drive_accs[b][0], base._seq_drive_accs[b][0]) for b in range(3))
    assert RD.row_shuffle_perm(7, 1) is None


def test_read_b_drive_record_and_guard(monkeypatch):
    from research.runners import c2_hops_step0 as S0
    from research.runners import stage2_exit_gate as SG
    monkeypatch.setenv("BRAIN_C2_QUIET_DRAIN", "60")
    buf = _stub(monkeypatch, K=2)
    buf.words = ["w%d" % i for i in range(buf.V)]
    rec = S0.read_b_drive_record(buf)
    assert rec["flags"]["BRAIN_C2_QUIET_DRAIN"] == "60" and rec["quiet_drain"] == 60
    assert rec["accs_recorded"] and rec["role_info"]["mode"] == "divnorm"
    for (aA, aX), (dA, dX) in zip(rec["accs"], buf._seq_drives):
        assert sorted(aA) == sorted(buf.words[j] for j in np.flatnonzero(dA > 0))
        assert all(v > 0 for v in aA.values())
    json.dumps(rec)
    nb = types.SimpleNamespace(b=None)
    with SG.substrate_read_guard(nb) as info:
        nb._seq_drive_accs, nb._seq_role_info = [1], {}
    assert not hasattr(nb, "_seq_drive_accs") and not hasattr(nb, "_seq_role_info")
    assert "optional_composer_attrs_restored" not in info
    nb._seq_drive_accs = ["kept"]
    with SG.substrate_read_guard(nb) as info:
        nb._seq_drive_accs = ["changed"]
    assert nb._seq_drive_accs == ["kept"] and "optional_composer_attrs_restored" not in info


# ═══════════════════════════════════════════════ pool lines + scorer ═══════════════════════════════════════════════
SHA = "0123456789abcdef0123456789abcdef01234567"


def test_pool_lines():
    from research.runners import c2_drain_wta_score as D
    lines = D.pool_lines(SHA)
    assert len(lines) == 7 * 37
    for arm in D.ARMS:
        mine = [c for c, w in lines
                if w == "RULING c2-drain-wta-1101: C2 match-unit smoke arm %s (dev seeds 1-7); mem_gb=9 max_age_h=240" % arm]
        assert len(mine) == 37
        env = " ".join("%s=%s" % kv for kv in D.ARMS[arm])
        for c in mine:
            assert c.startswith("cd ~/derisk-pool/revisions/%s && " % SHA)
            assert (" env %s SIM_BACKEND=numpy" % env) in c
            assert c.count("research/findings/raw/_c2_drain_wta/%s/s" % arm) == 2 and "_c2_step0/" not in c
            assert "--allow-gate-seed" not in c
    assert "BRAIN_C2_ROLE_WTA_I2E=0" in [c for c, w in lines if "arm L1w" in w][0]
    with pytest.raises(ValueError):
        D.pool_lines("abc")


REF = "research/findings/raw/_c2_step0_gatereset"
need_ref = pytest.mark.skipif(not os.path.isdir(REF), reason="gate-reset wave not in this checkout")


@need_ref
def test_frozen_denominators_on_the_reference_wave():
    from research.runners import c2_drain_wta_score as D
    pairs, info, why = D.classify_reference(D.load_wave(REF))
    assert why == [] and D.frozen_counts(pairs, info) == D.FROZEN


def _fake_arm(tmp, arm, mutate=None):
    """The reference shards copied as arm `arm`, with the engagement record the branch's read (b) writes."""
    from research.runners import c2_drain_wta_score as D
    from research.runners import c2_hops_step0 as S0
    env = dict(D.ARMS[arm])
    for (s, t), sh in D.load_wave(REF).items():
        sh = copy.deepcopy(sh)
        b = sh["c2_hops"]["reads"]["b"]
        b["c2dw"] = {"flags": {n: env.get(n) for n in D.ALL_FLAGS}, "quiet_drain": None, "accs_recorded": True,
                     "role_info": {"mode": "divnorm", "row_shuffle_perm": None},
                     "accs": [[{w: 1.0 for w in dA}, {w: 1.0 for w in dX}] for dA, dX in b["drives_lit_words"]]}
        for ck in D.CUE_ORDER:
            b["cues"][ck]["drain"] = {"drain_steps": 20, "quiet_drain_env": None, "prev_firing_at_clear": 0}
        if mutate:
            mutate(s, t, sh)
        p = S0.shard_path(os.path.join(tmp, arm), s, S0.TABLE_ARMS[t])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(sh, f)


@need_ref
def test_scorer_a0_pass_and_undefined(tmp_path, monkeypatch):
    from research.runners import c2_drain_wta_score as D
    from research.runners import c2_hops_step0 as S0
    monkeypatch.setattr(S0, "shard_validity", lambda shard, seed, arm: [] if shard else ["shard missing"])
    root = str(tmp_path)
    _fake_arm(root, "A0")
    res = D.score(root, REF, arms=("A0",))
    a0 = res["arms"]["A0"]
    assert a0["verdict"] == "PASS", a0
    assert a0["m_rates_identical_to_ref"] == 140
    assert a0["fires"]["one"] == 51 and a0["fires"]["both"] == 86     # the gate-reset fingerprint
    # a non-engaged record (wrong flag) and a missing shard are UNDEFINED, never PASS/FAIL
    shutil.rmtree(os.path.join(root, "A0"))
    _fake_arm(root, "A0", mutate=lambda s, t, sh: sh["c2_hops"]["reads"]["b"]["c2dw"]["flags"].update(
        BRAIN_C2_QUIET_DRAIN="60") if (s, t) == (3, "infer") else None)
    assert D.score(root, REF, arms=("A0",))["arms"]["A0"]["verdict"] == "UNDEFINED"
    os.remove(S0.shard_path(os.path.join(root, "A0"), 3, "c2s0_infer"))
    assert D.score(root, REF, arms=("A0",))["arms"]["A0"]["verdict"] == "UNDEFINED"
    assert D.score(root, REF, arms=("A1",))["arms"]["A1"]["verdict"] == "UNDEFINED"       # no shards at all


@need_ref
def test_scorer_a2r_bars_on_a_planted_ideal(tmp_path, monkeypatch):
    """A planted ideal A2r (m fires exactly on true blocks; role accs light only the decoded words) passes B1-B6, and
    the same arm with the both-lit fires left in fails B2."""
    from research.runners import c2_drain_wta_score as D
    from research.runners import c2_hops_step0 as S0
    monkeypatch.setattr(S0, "shard_validity", lambda shard, seed, arm: [] if shard else ["shard missing"])
    root = str(tmp_path)
    _fake_arm(root, "A0")

    def ideal(keep_both):
        def mut(s, t, sh):
            r = sh["c2_hops"]["reads"]
            b = r["b"]
            for ck in D.CUE_ORDER:
                c = b["cues"][ck]
                a, x = c["cue"]
                m = []
                for blk in range(len(c["m_rates"])):
                    both = a in b["drives_lit_words"][blk][0] and x in b["drives_lit_words"][blk][1]
                    m.append(0.3 if blk in c["true_matches"] or (keep_both and both) else 0.0)
                c["m_rates"] = m
                c["drain"]["drain_steps"] = 60
            b["c2dw"]["role_info"] = {"mode": "relay", "w_i2e": 2.0, "inh_spikes": 10.0, "reads": 2 * r["K"],
                                      "inh_ceiling": 600.0 * 2 * r["K"], "row_shuffle_perm": None}
            b["c2dw"]["accs"] = [[{row["agent"]: 1.0}, {row["action"]: 1.0}] for row in r["decoded_rows"]]
        return mut
    _fake_arm(root, "A2r", mutate=ideal(False))
    res = D.score(root, REF, arms=("A0", "A2r"))
    assert res["arms"]["A2r"]["verdict"] == "PASS", res["arms"]["A2r"]
    assert res["arms"]["A2r"]["role_reads_exact"] == 630
    shutil.rmtree(os.path.join(root, "A2r"))
    _fake_arm(root, "A2r", mutate=ideal(True))
    res = D.score(root, REF, arms=("A0", "A2r"))
    b2 = [x for x in res["arms"]["A2r"]["bars"] if x["bar"] == "B2_both_lit"][0]
    assert res["arms"]["A2r"]["verdict"] == "FAIL" and b2["got"] == 86
