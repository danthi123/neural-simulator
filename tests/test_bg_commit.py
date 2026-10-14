"""The shared spiking commit stage (research/runners/spiking_commit.py) and its BG live-seam application.

Plan: docs/drafts/spiking-commit/PLAN.md section 3; worklog docs/drafts/bg-commit/WORKLOG.md. Pins (numpy, CPU):
  * A1 moved onto the shared module is byte-identical: its commit battery on dev seed 1 at row 0 hashes to the value
    captured on S main cc9b4bf3e before the move;
  * the module: `commit_plan` and `region_parts` emit the same edges; `sequence_read` reads A1's semantics;
  * BG (flag BRAIN_BG_COMMIT): flag-off neural-seam trials hash to the value captured on cc9b4bf3e; the readers
    default off; flag off keeps the server's organ seed 42; with the flag the commit decides, its read is invariant,
    the commit and latch lesions commit nothing, and the old latch read cannot change a commit;
  * the scorer: without `commit` it reproduces the committed 10-02 score; with it, G4 reads the commit record
    (synthetic records on the 10-02 shards: invariant -> GO, one changed read -> NO-GO, unengaged -> UNDEFINED);
    `arm_env` sets the flag on the seam arms only.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

# captured on cc9b4bf3e (before any edit) with the same `_h` over `commit_battery(1, 0)`
A1_COMMIT_BATTERY_SHA256 = "6e0bd58ef1d460e64f6951c6f8819ab8db781b2b188138caf029fb2470f4a5bc"


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                     else str(o)).encode()).hexdigest()


# ── A1 on the shared module ────────────────────────────────────────────────────────────────────────────────────────
def test_a1_commit_battery_byte_identical():
    import research.runners.word_role_commit_organ as C
    assert _h(C.commit_battery(1, 0)) == A1_COMMIT_BATTERY_SHA256


# ── the module ─────────────────────────────────────────────────────────────────────────────────────────────────────
P = dict(NG=3, NGI=2, NS=4, W_XG=50.0, W_GG=20.0, W_GGI=60.0, W_GIG=60.0, W_GS=12.0, W_SS=60.0)


def _edges_from_plan(plan):
    out = set()
    for e in plan.values():
        for a, b, w in zip(e["pre_indices"], e["post_indices"], e["initial_weights"]):
            out.add((int(a), int(b), round(float(w), 4), e.get("transmission_gate") or None))
    return out


def test_region_parts_match_commit_plan():
    from research.runners import spiking_commit as SC
    from research.runners._vocal_action_selector_gate import _region
    from sim.enums import NeuronType
    K, src_n = 3, 5
    src_regions = ["src_%d" % k for k in range(K)]
    regions, pathways = SC.region_parts(src_regions, P, prefix="c_", gate_in="i", gate_excl="x", latch_loop="l",
                                        make_region=_region, exc_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL,
                                        inh_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON)
    # place every region at the index block `layout` gives it (sources first), then compare edge sets
    lay = SC.layout(K, K * src_n, P["NG"], P["NGI"], P["NS"])
    sizes = {r: src_n for r in src_regions}
    sizes.update({r.name: r.n_neurons for r in regions})
    start = {nm: k * src_n for k, nm in enumerate(src_regions)}
    for k in range(K):
        for pop, nm in SC.region_names("c_", k).items():
            start[nm] = int(lay[pop][k][0])
            assert sizes[nm] == len(lay[pop][k])
    reg_edges = set()
    for pw in pathways:
        assert pw.density == 1.0 and pw.weight_jitter == 0.0 and not pw.plastic
        for a in range(sizes[pw.from_region]):
            for b in range(sizes[pw.to_region]):
                reg_edges.add((start[pw.from_region] + a, start[pw.to_region] + b, round(pw.weight_mean, 4),
                               pw.transmission_gate or None))
    plan = SC.commit_plan([np.arange(k * src_n, (k + 1) * src_n) for k in range(K)], lay, P, prefix="",
                          gate_in="i", gate_excl="x", latch_loop="l")
    assert reg_edges == _edges_from_plan(plan)
    inh = [pw for pw in pathways if pw.from_region.startswith("c_gate_inh_")]
    assert inh and all(pw.receptor == "gaba_a" and pw.transmission_gate == "x" for pw in inh)
    assert [r.exc_fraction for r in regions[:3]] == [1.0, 0.0, 1.0]
    assert all(r.internal_density == 0.0 for r in regions)


def test_sequence_read_semantics():
    from research.runners import spiking_commit as SC
    # thr = 0.5 * 4 * 10 = 20. Item 0 latches 0; item 1 latches 1 (0 still on); item 2 latches nothing new.
    rd = SC.sequence_read([[30, 0, 0], [30, 25, 0], [30, 25, 0]], 4, 10, 0.5)
    assert rd["threshold"] == 20.0 and [p["role"] for p in rd["per_item"]] == [0, 1, None]
    assert rd["filled"] == {0: 0, 1: 1} and rd["co_commit"] == 0 and rd["latched_at_end"] == [0, 1]
    co = SC.sequence_read([[30, 30, 0]], 4, 10, 0.5)
    assert co["per_item"][0]["role"] is None and co["co_commit"] == 1
    re = SC.sequence_read([[30, 0, 0], [0, 0, 0], [30, 0, 0]], 4, 10, 0.5)
    assert re["relatch"] == 1 and 0 not in re["filled"]
    one = SC.read_with_variants([77, 0], 20, 30, 0.03)                     # A1 dev margin: 77 vs 0, thr 18
    assert one["winner"] == 0 and one["read_invariant"]


# ── BG: the flag, flag-off identity, the commit decides, lesions ───────────────────────────────────────────────────
# captured on cc9b4bf3e (before any edit): NeuralSeamSelector(seed=1, lesion in (None, 'value')).select_once over
# ('...', 'what does the dog chase', '?', 'hi'), hashed with `_h`
BG_NEURAL_SEAM_TRIALS_SHA256 = "4b54f5c4ac1a39a0ef8471245dcbc754a333686641ce26f1f6ca37547eb46d18"
BG_MSGS = ("...", "what does the dog chase", "?", "hi")


def _bg():
    from research.runners import bg_action_selection_production_organ as BG
    return BG


def test_bg_flag_off_neural_seam_trials_byte_identical():
    BG = _bg()
    out = []
    for les in (None, "value"):
        org = BG.NeuralSeamSelector(seed=1, lesion=les)
        out.append([org.select_once(m) for m in BG_MSGS])
    assert _h(out) == BG_NEURAL_SEAM_TRIALS_SHA256


def test_bg_commit_flag_readers(monkeypatch):
    BG = _bg()
    for k in (BG.BG_COMMIT_FLAG, BG.BG_COMMIT_OP_FLAG, BG.BG_COMMIT_LESION_FLAG, "BRAIN_CHAT_SEED"):
        monkeypatch.delenv(k, raising=False)
    assert BG.bg_commit_enabled() is False and BG.bg_commit_lesion() is None and BG.bg_commit_op() == BG.BG_COMMIT_OP
    assert BG.bg_commit_seed() == 42
    for v, want in (("1", True), ("on", True), ("0", False), ("", False)):
        monkeypatch.setenv(BG.BG_COMMIT_FLAG, v)
        assert BG.bg_commit_enabled() is want
    for v, want in (("commit", "commit"), ("latch", "latch"), ("excl", "excl"), ("value", None)):
        monkeypatch.setenv(BG.BG_COMMIT_LESION_FLAG, v)
        assert BG.bg_commit_lesion() == want
    monkeypatch.setenv(BG.BG_COMMIT_OP_FLAG, "99")
    assert BG.bg_commit_op() == BG.BG_COMMIT_OP
    monkeypatch.setenv("BRAIN_CHAT_SEED", "7")
    assert BG.bg_commit_seed() == 7
    assert 0 <= BG.BG_COMMIT_OP < len(BG.BG_COMMIT_GRID)


class _Stub:
    def select_once(self, msg):
        return {"winner": None, "committed": False, "latch_onset_ms": [None, None]}


def test_bg_flag_off_organ_seed_is_42_and_flag_on_follows_the_chat_seed(monkeypatch):
    BG = _bg()
    for k in (BG.BG_COMMIT_FLAG, BG.BG_COMMIT_LESION_FLAG, BG.BG_COMMIT_OP_FLAG, BG.NEURAL_SEAM_LESION_FLAG):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BRAIN_CHAT_SEED", "3")
    seen = []
    monkeypatch.setattr(BG, "get_neural_organ", lambda **kw: seen.append(kw) or _Stub())
    assert BG.decide_action_neural("...") is None
    assert seen == [{"seed": 42, "lesion": None}]                         # the server's organ seed, unchanged
    monkeypatch.setenv(BG.BG_COMMIT_FLAG, "1")
    monkeypatch.setenv(BG.BG_COMMIT_LESION_FLAG, "latch")
    BG.decide_action_neural("...")
    assert seen[-1] == {"seed": 3, "lesion": None, "commit_op": BG.BG_COMMIT_OP, "commit_lesion": "latch"}


@pytest.fixture(scope="module")
def bg_commit_trials():
    BG = _bg()
    out = {}
    for les in (None, "commit", "latch"):
        org = BG.NeuralSeamSelector(seed=1, commit=BG.bg_commit_row(BG.BG_COMMIT_OP), commit_lesion=les)
        out[les] = [org.select_once(m) for m in BG_MSGS]
    return out


def test_bg_commit_decides_and_reads_invariantly(bg_commit_trials):
    BG = _bg()
    for r in bg_commit_trials[None]:
        c = r["commit"]
        assert r["winner"] == c["winner"] and r["committed"] == (c["winner"] is not None)
        assert c["read_invariant"] and set(c["read_variants"]) == {"x0.5", "x2"}
        assert "old_latch_winner" in c and set(c["old_latch_variants"]) == {"x0.5", "x2"}
        assert c["slot_carry"] == [0, 0] and isinstance(c["co_commit"], bool)
        assert c["op"] == BG.BG_COMMIT_OP and c["organ_seed"] == 1 and c["lesion"] is None
    w = [r["winner"] for r in bg_commit_trials[None]]
    assert w[0] == BG.STAY_SILENT and w[1] == BG.SPEAK                    # dev seed 1: '...' held, the question not


def test_bg_commit_and_latch_lesions_commit_nothing(bg_commit_trials):
    for les in ("commit", "latch"):
        assert all(r["winner"] is None for r in bg_commit_trials[les]), les
    # the commit lesion cuts only the stage: no gate spike, while the race upstream still runs
    assert all(sum(r["commit"]["gate_counts_drive"]) == 0 for r in bg_commit_trials["commit"])
    assert any(sum(r["motor_total"]) > 0 for r in bg_commit_trials["commit"])


def test_old_latch_read_cannot_change_a_commit(monkeypatch):
    """The old fixed-rate latch read is recorded, never used: moving its threshold leaves the decision unchanged."""
    BG = _bg()
    row = BG.bg_commit_row(BG.BG_COMMIT_OP)
    base = [BG.NeuralSeamSelector(seed=1, commit=row).select_once(m) for m in BG_MSGS[:2]]
    monkeypatch.setattr(BG, "LATCH_READ_RATE", 10.0)                    # no old latch can read ON now
    again = [BG.NeuralSeamSelector(seed=1, commit=row).select_once(m) for m in BG_MSGS[:2]]
    assert [r["winner"] for r in again] == [r["winner"] for r in base]
    assert all(r["commit"]["old_latch_winner"] is None for r in again)


def test_derisk_refuses_gate_seeds():
    import research.runners._bg_commit_derisk as D
    with pytest.raises(SystemExit):
        D.main(["--sweep", "--seeds", "42"])


# ── the scorer (amendment 2026-10-14) ──────────────────────────────────────────────────────────────────────────────
GATE_DIR = os.path.join(REPO, "research", "findings", "raw", "_bg_live_seam", "gate")
GATE = (42, 43, 44, 100, 101, 102)


def test_scorer_without_commit_reproduces_the_1002_score():
    import research.runners.host_share_bg_seam as H
    sc = H.score(GATE_DIR, GATE)
    ref = json.load(open(os.path.join(GATE_DIR, "score.json")))
    assert sc["pooled"] == ref["pooled"] and sc["n"] == ref["n"] and "amendment" not in sc
    vd = H.verdict(sc)
    assert vd["verdict"] == ref["verdict"] == "NO-GO" and vd["gates"] == ref["gates"]


def _amended_copy(tmp_path, mutate=None):
    import shutil
    d = tmp_path / "gate"
    shutil.copytree(GATE_DIR, d)
    for s in GATE:
        for arm in ("seam_b", "seam_readout", "seam_lesion_value"):
            p = d / ("s%d" % s) / (arm + ".json")
            sh = json.load(open(p))
            sh["env"]["BRAIN_BG_COMMIT"] = "1"
            for label, evs in sh["events"].items():
                for e in evs:
                    if e.get("kind") == "bg_neural":
                        w = e.get("winner")
                        e["commit"] = {"winner": w, "read_invariant": True, "read_variants": {"x0.5": w, "x2": w},
                                       "co_commit": False, "old_latch_winner": w}
                        if mutate:
                            mutate(s, arm, label, e)
            json.dump(sh, open(p, "w"))
    return str(d)


def test_amended_scorer_g4_reads_the_commit(tmp_path):
    import research.runners.host_share_bg_seam as H
    sc = H.score(_amended_copy(tmp_path), GATE, commit=True)
    assert sc["amendment"] == H.AMENDMENT_ID
    assert sc["pooled"]["commit_thr_change"] == 0.0 and sc["n"]["commit_thr_change"] == 24
    assert sc["pooled"]["commit_engaged"] == 1.0 and sc["n"]["commit_engaged"] == 72
    assert sc["pooled"]["latch_thr_change"] == 0.25                     # the old latch read, against its own choice
    vd = H.verdict(sc)
    assert vd["gates"]["G4_commit_read_threshold_invariant"] is True
    assert vd["gates"]["G4_old_latch_read_threshold_invariant (reported, not gated)"] is False
    assert vd["verdict"] == "GO"            # the 10-02 data with a synthetic invariant commit read passes G1-G5


def test_amended_scorer_one_changed_read_is_no_go(tmp_path):
    import research.runners.host_share_bg_seam as H

    def flip(s, arm, label, e):
        if s == 42 and arm == "seam_b" and label == "hsb_q":
            e["commit"]["read_variants"]["x2"] = None
            e["commit"]["read_invariant"] = False
    sc = H.score(_amended_copy(tmp_path, flip), GATE, commit=True)
    assert sc["pooled"]["commit_thr_change"] == round(1 / 24, 4)
    vd = H.verdict(sc)
    assert vd["gates"]["G4_commit_read_threshold_invariant"] is False and vd["verdict"] == "NO-GO"


def test_amended_scorer_unengaged_is_undefined(tmp_path):
    import research.runners.host_share_bg_seam as H

    def drop(s, arm, label, e):
        if s == 43 and arm == "seam_lesion_value":
            e.pop("commit")
    sc = H.score(_amended_copy(tmp_path, drop), GATE, commit=True)
    assert sc["pooled"]["commit_engaged"] < 1.0
    assert H.verdict(sc)["verdict"] == "UNDEFINED"


def test_arm_env_sets_the_commit_flag_on_seam_arms_only():
    import research.runners.host_share_bg_seam as H
    base = {"SIM_BACKEND": "numpy"}
    on = {a: H.arm_env(base, a, 5, commit=True, outer={}) for a in H.ARMS}
    assert {a: e["BRAIN_BG_COMMIT"] for a, e in on.items()} == {
        a: ("1" if H.ARMS[a][0].get("BRAIN_BG_NEURAL_SEAM") == "1" else "") for a in H.ARMS}
    assert {a for a, e in on.items() if e["BRAIN_BG_COMMIT"] == "1"} == {"seam_b", "seam_readout", "seam_lesion_value"}
    off = H.arm_env(base, "seam_b", 5, commit=False, outer={})
    assert not any(k in off for k in H.COMMIT_ENV)                       # the 09-29 environment, unchanged
    stray = H.arm_env(base, "seam_b", 5, commit=False, outer={"BRAIN_BG_COMMIT": "1"})
    assert stray["BRAIN_BG_COMMIT"] == ""                                # an outer flag cannot leak into a run
