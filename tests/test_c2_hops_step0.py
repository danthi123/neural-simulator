"""C2 M1 step 0 (research/coordination/c2-spiking-hops-PACKET.md 2.10, round 9): the Stage-2 runner changes (1)-(7) and
the step-0 scorer. No brain build: stubs stand in for the composer, the chat and the LTM; the D4 recorder wraps the real
organ module's functions (with the lexicon functions stubbed underneath it, so no learned lexicon is loaded).

Behaviour preservation: every pre-existing arm keeps its turn table, spec and scramble (the GOLDEN fingerprints of
tests/test_stage2_kb_scramble.py, re-checked here), and the gate's pool lines are byte-identical to the base revision's
(26dcdea48, sha256 pinned below). Each new fixture's engagement record is exercised on stubs, and the scorer is run on
planted shards: a GO case, every STOP line, the UNDEFINED rules and a NOT_GO."""
import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import stage2_exit_gate as SG
from research.runners import c2_hops_step0 as S0
from tests import test_stage2_kb_scramble as KB

# sha256 of "\n".join("cmd\twhy") over SG.pool_job_lines("f" * 40) at the base revision 26dcdea48 (210 lines)
BASE_GATE_LINES_SHA = "53f5e2db8addb01a5a2b2de42b4f4ea85c267d331e3b57466ceb496d8860578b"
BASE_GATE_LINES_NOINFO_SHA = "060d0d7dac6193f967ef8c7aa581ec4cb029c34e66642080ef9588286f34a8b4"


# ── behaviour preservation ─────────────────────────────────────────────────────────────────────────────────────────────
def test_every_pre_existing_arm_keeps_its_golden_fingerprint():
    for key, want in KB.GOLDEN.items():
        group, arm = key.split("/", 1)
        assert arm in SG.arms_for(group) and KB._fp(group, arm) == want, key
    for arm in KB.KB_ARMS:
        assert SG.turns_for("infer", 7, arm) == SG.turns_for("infer", 7, "intact_a")
        assert SG.arm_c2("infer", arm) is None


def test_the_gate_pool_lines_are_byte_identical_to_the_base_revision():
    def h(lines):
        return hashlib.sha256("\n".join("%s\t%s" % x for x in lines).encode()).hexdigest()
    assert h(SG.pool_job_lines("f" * 40)) == BASE_GATE_LINES_SHA
    assert h(SG.pool_job_lines("f" * 40, informational=False)) == BASE_GATE_LINES_NOINFO_SHA


def test_no_pre_existing_arm_carries_a_c2_record():
    for key in KB.GOLDEN:
        group, arm = key.split("/", 1)
        assert SG.arm_c2(group, arm) is None and not SG.is_c2_step0(group, arm), key


# ── (1) per-arm turn tables ────────────────────────────────────────────────────────────────────────────────────────────
def test_step0_entries_run_their_own_tables_and_are_informational():
    want = {"c2s0_infer": "infer", "c2s0_amb1": "infer_amb1", "c2s0_amb2": "infer_amb2", "c2s0_dup": "infer_dup",
            "c2s0_xtier": "infer_xtier", "c2s0_infer_rec": "infer", "c2s0_infer_bare": "infer"}
    for arm, table in want.items():
        assert SG.turns_for("infer", 1, arm) == SG.C2_TABLES[table]
        assert SG.arm_spec("infer", arm)[2] == "informational"
        assert SG.is_c2_step0("infer", arm) and SG.arm_c2("infer", arm)["table"] == table
        assert arm not in SG.gate_arms_for("infer") and arm in SG.arms_for("infer")
        shard = {"seed": 1, "responses": {}, "sources": [SG.EXPECTED_SOURCE], "pythonhashseed_ok": True,
                 "turns": [{"label": l, "message": m, "role": r} for l, m, r in SG.C2_TABLES[table]]}
        assert not any("turn table differs" in w for w in SG.arm_validity("infer", arm, shard)), arm
    assert SG.arm_c2("infer", "c2s0_infer_rec")["step0"] == "recorder"
    assert SG.arm_c2("infer", "c2s0_infer_bare")["step0"] == "bare"
    assert SG.arm_spec("infer", "c2s0_xtier")[0] == {"BRAIN_SHARED_LTM": "0", "BRAIN_LTM_RNG_ISOLATE": "1"}
    assert "BRAIN_ONE_BRAIN" not in SG.arm_spec("infer", "c2s0_xtier")[0]


def test_the_four_new_tables_are_the_packets():
    T = SG.C2_TABLES
    lab = lambda t: [x[0] for x in t]      # noqa: E731
    assert lab(T["infer_amb1"]) == ["t_heron_hunt", "t_frog_eat", "t_heron_hunt_worm", "p_prey_eat", "c_heron_hunt",
                                    "c_dog_chase"]
    assert ("t_heron_hunt_worm", "the heron hunts the worm", "teach") in T["infer_amb1"]
    assert ("t_frog_eat_apple", "the frog eats the apple", "teach") in T["infer_amb2"]
    assert lab(T["infer_dup"])[:6] == ["t_heron_hunt", "t_frog_eat", "t_crow_chase", "t_wren_chase", "t_heron_hunt_re",
                                       "t_frog_eat_re"]
    assert T["infer_dup"][4][1] == "the heron hunts the frog" and T["infer_dup"][5][1] == "the frog eats the gnat"
    assert T["infer"] == T["infer_xtier"] == SG.GROUPS["infer"]["turns"]
    # worm is in no other table string and no built-in fact (packet 2.8, round 5)
    others = [m for t in ("infer", "infer_amb2", "infer_dup", "infer_xtier") for _l, m, _r in T[t]]
    assert not any("worm" in m for m in others)


def test_table_permuted_key_selects_the_groups_permuted_turns(monkeypatch):
    info = dict(SG.GROUPS["infer"]["informational"])
    info["_tmp_perm"] = {"env": {}, "patch": None, "table": "permuted"}
    monkeypatch.setitem(SG.GROUPS["infer"], "informational", info)
    assert SG.turns_for("infer", 1, "_tmp_perm") == SG.GROUPS["infer"]["permuted_turns"]


# ── (2) the D4 recorder ────────────────────────────────────────────────────────────────────────────────────────────────
HAND_ANIMACY = {"frog": "animate", "worm": "animate", "apple": "inanimate", "dog": "animate"}
HAND_VS = {"eat": {"agent": "animate", "patient": "inanimate"}}


@pytest.fixture
def cpo(monkeypatch):
    """The real organ module, its two lexicon functions stubbed to the hand facts the packet cites (no learned
    lexicon loaded), and every attribute the recorder wraps restored after the test."""
    from research.runners import comprehension_production_organ as CPO
    monkeypatch.setattr(CPO, "_verb_selects_of", lambda v: HAND_VS.get(v))
    monkeypatch.setattr(CPO, "_animacy_of", lambda n: HAND_ANIMACY.get(n))
    monkeypatch.setattr(CPO, "extract_transitive", CPO.extract_transitive)
    monkeypatch.setattr(CPO.ComprehensionProductionOrgan, "competent", CPO.ComprehensionProductionOrgan.competent)
    rec = SG.D4Recorder()
    rec.install()
    org = CPO.ComprehensionProductionOrgan(seed=7)
    org.ensure_built = lambda: None
    org.read_margin = lambda *a, **k: 0.5
    org.threshold, org.calib = 0.2, {}
    return CPO, rec, org


def test_d4_recorder_records_inner_returns_and_the_three_branches(cpo):
    CPO, rec, org = cpo
    for label, text, want in (("t_frog_eat_apple", "the frog eats the apple", "covered"),
                              ("t_heron_hunt", "the heron hunts the frog", "pass_through"),
                              ("t_heron_hunt_worm", "the heron hunts the worm", "pass_through"),
                              ("t_zib", "the zib vops the tav", "oov"),
                              ("q", "what does the frog eat", "out_of_scope")):
        rec.label = label
        org.judge(text, brain_vocab={"heron", "hunt", "frog", "gnat"})
        br = SG.d4_branch(rec.calls, rec.extract_calls, label)
        assert br["branch"] == want, (label, br)
    calls = rec.calls
    assert calls and all(c["or_matches_return"] for c in calls) and all(c["caller_kind"] == "judge" for c in calls)
    apple = [c for c in calls if c["label"] == "t_frog_eat_apple"][0]
    assert apple["inner"] == [{"fn": "_verb_selects_of", "arg": "eat", "ret": HAND_VS["eat"]},
                              {"fn": "_animacy_of", "arg": "frog", "ret": "animate"},
                              {"fn": "_animacy_of", "arg": "apple", "ret": "inanimate"},
                              {"fn": "_verb_selects_of", "arg": "eat", "ret": HAND_VS["eat"]}]
    assert apple["fully_covered"] is True and apple["fully_oov"] is False and apple["return"] is True
    assert apple["terms"]["oov_n0"] == SG._NOT_EVALUATED
    heron = [c for c in calls if c["label"] == "t_heron_hunt"][0]
    v_heron = CPO.extract_transitive("the heron hunts the frog")[1]                 # the verb as the organ reads it
    assert heron["inputs"][1] == v_heron and heron["brain_vocab_size"] == 4
    assert heron["in_brain_vocab"] == {"n0": True, "v": v_heron in {"heron", "hunt", "frog", "gnat"}, "n1": True}
    assert heron["return"] is False and heron["fully_covered"] is False and heron["fully_oov"] is False
    assert SG.d4_branch(rec.calls, rec.extract_calls, "never_spoken")["branch"] == "not_called"
    q = [e for e in rec.extract_calls if e["label"] == "q" and e["text"] == "what does the frog eat"]
    assert q and q[-1]["ret"] is None and q[-1]["caller"]["function"] == "judge"
    assert rec.extract_calls[-1]["caller"]["function"] != "judge"                    # the test's own call, labelled q


def test_d4_recorder_is_behaviour_preserving_and_labels_other_callers(cpo):
    CPO, rec, org = cpo
    rec.label = "x"
    assert org.competent("frog", "eat", "apple") is True
    assert org.competent("zib", "vop", "tav") is True
    assert org.competent("heron", "hunt", "frog") is False
    assert [c["caller_kind"] for c in rec.calls] == ["other"] * 3
    assert CPO.extract_transitive("the frog eats the apple") == ("frog", "eat", "apple")
    assert SG.d4_branch(rec.calls, rec.extract_calls, "x")["branch"] == "not_called"     # never read off another caller


def test_replay_flags_a_call_sequence_that_does_not_follow_the_expression():
    inner = [{"fn": "_animacy_of", "arg": "frog", "ret": "animate"}]
    with pytest.raises(LookupError):
        SG.replay_competent("frog", "eat", "apple", set(), inner)
    ok = [{"fn": "_verb_selects_of", "arg": "hunt", "ret": None}, {"fn": "_verb_selects_of", "arg": "hunt", "ret": None},
          {"fn": "_animacy_of", "arg": "heron", "ret": None}, {"fn": "_animacy_of", "arg": "frog", "ret": "animate"}]
    r = SG.replay_competent("heron", "hunt", "frog", set(), ok)
    assert r["fully_covered"] is False and r["fully_oov"] is False and r["terms"]["covered_n0"] == SG._NOT_EVALUATED
    with pytest.raises(LookupError):
        SG.replay_competent("heron", "hunt", "frog", set(), ok + [ok[-1]])            # a left-over call
    calls = [{"label": "t", "caller_kind": "judge", "or_matches_return": False, "return": True}]
    assert SG.d4_branch(calls, [], "t")["branch"] == "undefined"


# ── (6) the propose-path wrappers ──────────────────────────────────────────────────────────────────────────────────────
def test_propose_path_of_reads_the_three_paths():
    fc = {"fn": "fact_shard_candidates", "cue_roles": {"agent": "heron", "action": "hunt"}, "ret": [5]}
    assert SG.propose_path_of([fc, {"fn": "fact_shard_first_match", "escalate": False, "idx": 5}]) == {
        "fs_candidates": [5], "fs_cue_roles": fc["cue_roles"], "propose_path": "fact_shard", "selected_block": 5}
    esc = SG.propose_path_of([{"fn": "fact_shard_first_match", "escalate": True, "idx": None},
                              {"fn": "seq_block", "ret": 2}])
    assert esc["propose_path"] == "escalated" and esc["selected_block"] == 2
    assert SG.propose_path_of([{"fn": "seq_block", "ret": None}])["propose_path"] == "seq_block"
    assert SG.propose_path_of([])["propose_path"] is None


def test_propose_path_probe_wraps_and_delegates(monkeypatch):
    from research.runners import one_brain_composer as OBC
    C = OBC.OneBrainComposer
    monkeypatch.setattr(C, "_fact_shard_candidates", lambda self, cue: [3])
    monkeypatch.setattr(C, "_fact_shard_first_match", lambda self, cue: (3, {"agent": "heron"}))
    monkeypatch.setattr(C, "_seq_block", lambda self, a, x: None)
    p = SG.ProposePathProbe()
    p.install()
    p.label = "turn"
    with p.scope() as ev:
        assert C._fact_shard_first_match(object(), {"agent": "heron", "action": "hunt"}) == (3, {"agent": "heron"})
        assert C._fact_shard_candidates(object(), {"agent": "heron"}) == [3]
        assert C._seq_block(object(), "a", "x") is None
    assert [e["fn"] for e in ev] == ["fact_shard_first_match", "fact_shard_candidates", "seq_block"]
    assert all(e["label"] == "turn" for e in p.events) and len(p.events) == 3


# ── the read guard ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_substrate_read_guard_restores_bridge_arrays_composer_state_and_the_global_rng():
    b = types.SimpleNamespace(cp_membrane_potential_v=np.arange(5.0), cp_recovery_variable_u=np.ones(5),
                              cp_firing_states=np.zeros(5, dtype=bool))
    buf = types.SimpleNamespace(b=b, _seq=None, _seq_dirty=True, _fact_shard_built_K=4, last_trace="t")
    np.random.seed(11)
    random.seed(11)
    want_np, want_py = np.random.rand(), random.random()
    np.random.seed(11)
    random.seed(11)
    with SG.substrate_read_guard(buf) as info:
        b.cp_membrane_potential_v[:] = 0.0
        b.cp_recovery_variable_u[:] = 7.0
        buf._seq, buf._seq_dirty, buf._fact_shard_built_K, buf.last_trace = ("sb", {}), False, -1, None
        np.random.seed(42)
        random.seed(42)
        np.random.rand()
    assert list(b.cp_membrane_potential_v) == [0.0, 1.0, 2.0, 3.0, 4.0] and list(b.cp_recovery_variable_u) == [1.0] * 5
    assert buf._seq is None and buf._seq_dirty is True and buf._fact_shard_built_K == 4 and buf.last_trace == "t"
    assert np.random.rand() == want_np and random.random() == want_py
    assert "cp_membrane_potential_v" in info["bridge_arrays_restored"]


# ── (4) the kb_overwrite fixture: engagement (i)-(iv) ───────────────────────────────────────────────────────────────────
BUILTIN = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
           ("dog", "chase", "cat"), ("cat", "eat", "fish")]


class _StoreBuf:
    """A buffer composer whose `.kb` (the list) and `bound` (what `_read_blocks` decodes: the bound store) are separate."""
    def __init__(self, facts):
        self.kb = [({"agent": a, "action": v, "patient": p}, None) for a, v, p in facts]
        self.bound = [tuple(f) for f in facts]
        self._fact_shard_built_K = len(facts)
        self.b = None

    def _read_blocks(self):
        return [{"agent": a, "action": v, "patient": p, "polarity": "AFFIRM"} for a, v, p in self.bound]


class _Chat:
    def __init__(self, buf):
        self.inner = types.SimpleNamespace(composer=types.SimpleNamespace(buffer=buf, ltm=None))
        self._refresh_facts()

    def _refresh_facts(self):
        kb = self.inner.composer.buffer.kb
        self.stored_facts = [(f["agent"], f["action"], f["patient"]) for f, _ in kb]
        self.patients_set = {p for _a, _v, p in self.stored_facts}


AMB1 = BUILTIN + [("heron", "hunt", "frog"), ("frog", "eat", "gnat"), ("heron", "hunt", "worm")]
DUP = BUILTIN + [("heron", "hunt", "frog"), ("frog", "eat", "gnat"), ("crow", "chase", "wren"),
                 ("wren", "chase", "moth"), ("heron", "hunt", "frog"), ("frog", "eat", "gnat")]
PROBE = SG.GROUPS["infer"]["scramble_cfg"]["probe"]


def test_kb_overwrite_amb1_removal_engages_and_reads_back_every_condition():
    buf = _StoreBuf(AMB1)
    chat = _Chat(buf)
    rec = SG.apply_kb_overwrite(chat.inner.composer, chat, "amb1_worm_to_frog", PROBE, after_turn=2)
    assert rec["engaged"] is True, rec
    assert rec["planned_index"] == 7 and rec["changed_indices"] == [7] and rec["after_turn_index"] == 2
    assert all(rec["conditions"].values())
    iv = rec["condition_iv"]
    assert iv["absent_from_every_kb_field"] and iv["still_decoded_under_cue"] and iv["fact_shard_built_K"] == -1
    assert iv["absent_from_chat_stored_facts"] and iv["absent_from_patients_set"]
    assert rec["predicted_from_list"]["svo"] == ["frog", "eat", "gnat"] and rec["predicted_from_decoded"] is None
    assert buf.bound[7] == ("heron", "hunt", "worm")                                   # the bound store untouched
    assert buf.kb[7][0]["patient"] == "frog"


def test_kb_overwrite_dup_addition_engages_and_does_not_touch_the_chat_lists():
    buf = _StoreBuf(DUP)
    chat = _Chat(buf)
    before_sf = list(chat.stored_facts)
    rec = SG.apply_kb_overwrite(chat.inner.composer, chat, "dup_first_frog_to_worm", PROBE, after_turn=5)
    assert rec["engaged"] is True, rec
    assert rec["planned_index"] == 5 and rec["condition_iv"]["is_patient_of_first_entry"]
    assert rec["condition_iv"]["absent_from_decoded_rows_under_cue"]
    assert rec["predicted_from_list"] is None and rec["predicted_from_decoded"]["svo"] == ["frog", "eat", "gnat"]
    assert chat.stored_facts == before_sf and buf._fact_shard_built_K == len(DUP)      # not refreshed, not reset


def test_kb_overwrite_is_not_engaged_when_the_plan_cannot_hold():
    chat = _Chat(_StoreBuf(BUILTIN + [("heron", "hunt", "frog"), ("frog", "eat", "gnat")]))
    rec = SG.apply_kb_overwrite(chat.inner.composer, chat, "amb1_worm_to_frog", PROBE)
    assert rec["engaged"] is False and "exactly one" in rec["reason"]
    # a bound store that moves under the write fails (ii)
    buf = _StoreBuf(AMB1)
    chat = _Chat(buf)
    reads = iter([buf._read_blocks(), [dict(r, patient="frog") for r in buf._read_blocks()]])
    buf._read_blocks = lambda: next(reads)
    rec = SG.apply_kb_overwrite(chat.inner.composer, chat, "amb1_worm_to_frog", PROBE)
    assert rec["engaged"] is False and rec["conditions"]["ii_decoded_rows_unchanged"] is False
    # worm still held by another entry: (iv) fails
    chat = _Chat(_StoreBuf(AMB1 + [("owl", "see", "worm")]))
    rec = SG.apply_kb_overwrite(chat.inner.composer, chat, "amb1_worm_to_frog", PROBE)
    assert rec["engaged"] is False and rec["condition_iv"]["absent_from_every_kb_field"] is False


# ── (5) the infer_xtier LTM fixture ────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_gnw(monkeypatch):
    mod = types.ModuleType("webapp.gnw_deliberation")
    mod.all_candidate_patients = lambda shard, a, x: [p for (aa, xx, p) in shard.facts if (aa, xx) == (a, x)]
    mod._DEFAULT_SEED = 7
    monkeypatch.setitem(sys.modules, "webapp.gnw_deliberation", mod)
    return mod


class ShardedPhasorStore:          # the name is what the fixture checks (`type(ltm).__name__`)
    def __init__(self, fail=None):
        self.facts, self.fail = [], fail

    def store(self, a, x, p, polarity=None):
        if self.fail:
            raise self.fail
        self.facts.append((a, x, p))

    def query_patient(self, a, x):
        return next((p for aa, xx, p in self.facts if (aa, xx) == (a, x)), None)

    def shard_for(self, a):
        return self

    def route(self, a):
        return 0


class SharedLTMView(ShardedPhasorStore):
    def store(self, *a, **k):
        raise RuntimeError("SharedLTMWriteError: the shared LTM is read-only")


def _ltm_chat(ltm):
    return types.SimpleNamespace(inner=types.SimpleNamespace(composer=types.SimpleNamespace(ltm=ltm)))


def test_ltm_fixture_writes_only_into_a_private_sharded_store(fake_gnw, monkeypatch):
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    monkeypatch.setenv("BRAIN_LTM_RNG_ISOLATE", "1")
    monkeypatch.delenv("BRAIN_ONE_BRAIN", raising=False)
    ltm = ShardedPhasorStore()
    rec = SG.apply_ltm_fixture(_ltm_chat(ltm), "xtier_heron_hunt_fish")
    assert rec["ltm_type"] == "ShardedPhasorStore" and rec["written"] is True and rec["store_exception"] is None
    assert rec["env_as_read"] == {"BRAIN_SHARED_LTM": "0", "BRAIN_LTM_RNG_ISOLATE": "1", "BRAIN_ONE_BRAIN": None}
    assert rec["readback_after_write"]["query_patient"] == "fish"
    assert rec["readback_after_write"]["all_candidate_patients"] == ["fish"]
    view = SharedLTMView()
    rec = SG.apply_ltm_fixture(_ltm_chat(view), "xtier_heron_hunt_fish")
    assert rec["ltm_type"] == "SharedLTMView" and rec["written"] is False and view.facts == []    # never written
    rec = SG.apply_ltm_fixture(_ltm_chat(ShardedPhasorStore(fail=KeyError("not in codebook"))), "xtier_heron_hunt_fish")
    assert rec["written"] is False and "KeyError" in rec["store_exception"]
    rec = SG.apply_ltm_fixture(_ltm_chat(None), "xtier_heron_hunt_fish")
    assert rec["ltm_type"] == "NoneType" and rec["written"] is False


# ── the worker integration over a fake server (no brain): step-0 entries, the gate-seed refusal ────────────────────────
@pytest.fixture
def worker(monkeypatch, tmp_path, fake_gnw):
    from research.runners import one_brain_composer as OBC
    from research.runners import comprehension_production_organ as CPO
    from research.runners import compositional_chain_route as CCR
    for mod, names in ((OBC.OneBrainComposer, ("query_patient", "_fact_shard_candidates", "_fact_shard_first_match",
                                               "_seq_block")),
                       (CPO.ComprehensionProductionOrgan, ("competent",)),
                       (CPO, ("extract_transitive", "_verb_selects_of", "_animacy_of")),
                       (CCR, ("_distinct_patients",))):
        for n in names:
            monkeypatch.setattr(mod, n, getattr(mod, n))
    srv = KB._fake_server()

    class _ChatS(KB._FakeChat):
        def __init__(self):
            super().__init__()
            self.inner.composer = types.SimpleNamespace(buffer=_StoreBuf(BUILTIN), ltm=None)

        def _refresh_facts(self):
            self.stored_facts = [[f["agent"], f["action"], f["patient"]] for f, _ in self.inner.composer.buffer.kb]
            self.patients_set = {t[2] for t in self.stored_facts}

    def brain_chat(req):
        key = (req.session, req.brain, req.renderer)
        if req.reset or key not in srv._BRAIN_CHATS:
            srv._BRAIN_CHATS[key] = _ChatS()
        chat = srv._BRAIN_CHATS[key]
        buf = chat.inner.composer.buffer
        t = SG._canonical_teach(req.message)
        body = {"answer": "ok", "abstained": False, "recalled_svo": None, "derived": False, "source": SG.EXPECTED_SOURCE}
        if t:
            buf.kb.append(({"agent": t[0], "action": t[1], "patient": t[2]}, None))
            buf.bound.append(tuple(t))
            chat.stored_facts.append(list(t))
            body.update(answer=req.message, recalled_svo=list(t))
        return types.SimpleNamespace(body=json.dumps(body))
    srv.brain_chat = brain_chat
    pkg = types.ModuleType("webapp")
    pkg.__path__ = []
    pkg.server = srv
    pkg.gnw_deliberation = fake_gnw
    monkeypatch.setitem(sys.modules, "webapp", pkg)
    monkeypatch.setitem(sys.modules, "webapp.server", srv)
    base_env = dict(os.environ)

    def run(arm, seed=7):
        monkeypatch.setattr(os, "environ", dict(base_env))
        out = str(tmp_path / ("infer__%s_s%d.json" % (arm, seed)))
        rc = SG.run_worker(seed, "infer", arm, out)
        return rc, (json.load(open(out)) if os.path.exists(out) else None)
    return run


def test_step0_entries_run_through_the_worker_with_their_records(worker):
    rc, sh = worker("c2s0_amb1")
    assert rc == 0 and [t["label"] for t in sh["turns"]] == [t[0] for t in SG.C2_TABLES["infer_amb1"]]
    c2 = sh["c2_hops"]
    assert c2["mode"] == "full" and c2["table"] == "infer_amb1" and c2["teach_block_last_turn"] == 2
    assert ["heron", "hunt", "worm"] in c2["stored_facts_after_teach"]                 # read BEFORE the overwrite
    eng = sh["kb_scramble"]["engagement"]
    assert sh["kb_scramble"]["mode"] == "kb_overwrite" and eng["plan"] == "amb1_worm_to_frog" and eng["engaged"] is True
    assert eng["after_turn_index"] == 2
    assert "reads" in c2 and c2["ws_seed"] == 7 and set(c2["d4_branches"]) == {"t_heron_hunt", "t_frog_eat",
                                                                               "t_heron_hunt_worm"}
    assert set(c2["per_turn"]) == {t[0] for t in SG.C2_TABLES["infer_amb1"]}
    assert c2["per_turn"]["t_heron_hunt_worm"]["kb_len"] == len(BUILTIN) + 3
    rc, sh = worker("c2s0_infer_bare")
    assert rc == 0 and sh["c2_hops"]["mode"] == "bare" and "d4_recorder" not in sh["c2_hops"]
    assert "stored_facts_after_teach" not in sh["c2_hops"] and "kb_scramble" not in sh
    rc, sh = worker("c2s0_infer_rec")
    assert rc == 0 and "d4_recorder" in sh["c2_hops"] and "reads" not in sh["c2_hops"]
    assert "propose_path_events" not in sh["c2_hops"]


def test_the_xtier_entry_sets_offiso_and_records_the_fixture(worker):
    rc, sh = worker("c2s0_xtier")
    assert rc == 0 and sh["env"] == {"BRAIN_SHARED_LTM": "0", "BRAIN_LTM_RNG_ISOLATE": "1"}
    fx = sh["c2_hops"]["ltm_fixture"]
    assert fx["env_as_read"]["BRAIN_SHARED_LTM"] == "0" and fx["env_as_read"]["BRAIN_LTM_RNG_ISOLATE"] == "1"
    assert fx["ltm_type"] == "NoneType" and fx["written"] is False                      # the fake chat has no LTM
    assert "readback_before_first_probe" in fx


def test_a_step0_entry_is_refused_on_a_gate_seed(worker):
    rc, sh = worker("c2s0_infer", seed=42)
    assert rc == 2 and sh is None


def test_a_pre_existing_arm_gets_no_c2_record_through_the_worker(worker):
    for arm in ("intact_a", "kb_scramble"):
        rc, sh = worker(arm)
        assert rc == 0 and "c2_hops" not in sh


# ── the scorer on planted shards ───────────────────────────────────────────────────────────────────────────────────────
def _teach_facts(table):
    return [SG._canonical_teach(m) for _l, m, r in SG.C2_TABLES[table] if r == "teach"]


def _good_shard(seed, table, arm=None, mode="full"):
    arm = arm or S0.TABLE_ARMS[table]
    turns = SG.turns_for("infer", seed, arm)
    facts = [list(f) for f in BUILTIN] + _teach_facts(table)
    rows = [{"block": i, "agent": a, "action": v, "patient": p, "polarity": "AFFIRM"} for i, (a, v, p) in enumerate(facts)]
    resp = {}
    for l, m, r in turns:
        if r == "teach":
            f = SG._canonical_teach(m)
            resp[l] = {"answer": m, "abstained": False, "recalled_svo": f, "derived": False, "source": SG.EXPECTED_SOURCE,
                       "comprehension": None, "surprise": None, "reconsolidation": None}
        else:
            resp[l] = {"answer": "reply %s" % l, "abstained": False, "recalled_svo": None, "derived": False,
                       "source": SG.EXPECTED_SOURCE}
    if "t_frog_eat_apple" in resp:
        resp["t_frog_eat_apple"]["comprehension"] = {"in_scope": True, "margin": 0.01, "known_binding": True}
    calls, branches = [], {}
    for l, m, r in turns:
        if r == "teach":
            want = S0.EXPECTED_BRANCH.get(m, "pass_through")
            call = {"label": l, "caller_kind": "judge", "or_matches_return": True, "return": want == "covered",
                    "fully_covered": want == "covered", "fully_oov": False}
            calls.append(call)
            branches[l] = {"label": l, "branch": want, "call": call}
    under = lambda a, x: [r for r in rows if (r["agent"], r["action"]) == (a, x)]     # noqa: E731
    a_reads = {S0._key(c): {"cue": list(c), "decoded_rows_under_cue": under(*c), "host_routed": False,
                            "propose_path": "fact_shard", "buffer_propose": (under(*c) or [{}])[0].get("patient")}
               for c in S0.CUES_A}
    b_cues = {}
    for c in S0.CUES_A:
        tm = [r["block"] for r in under(*c)]
        b_cues[S0._key(c)] = {"cue": list(c), "true_matches": tm, "fired": list(tm), "missed_true_matches": [],
                              "fired_non_matching": [], "m_rates": [0.3 if i in tm else 0.0 for i in range(len(rows))]}
    blocks = [{"block": r["block"], "decoded_patient": r["patient"], "fired_pools": [r["patient"]], "n_fired": 1,
               "exactly_one": True, "fires_decoded_patient": True,
               "probe_vs_production": {k: {"equal": True, "max_abs_diff": 0.0} for k in ("agent", "action", "patient")}}
              for r in rows]
    vp = []
    for r in rows:
        if r["patient"] not in vp:
            vp.append(r["patient"])
    ltm_hh = ["fish"] if table == "infer_xtier" else []
    f_cues = {}
    for c in S0.premise_cues(table):
        lt = ltm_hh if (table == "infer_xtier" and c == ("heron", "hunt")) else []
        f_cues[S0._key(c)] = {"cue": list(c), "decoded_rows": under(*c), "decoded_patients": [r["patient"] for r in under(*c)],
                              "ltm": {"query_patient": lt[0] if lt else None, "all_candidate_patients": list(lt)}}
    c2 = {"mode": mode, "table": table, "stored_facts_after_teach": facts, "ws_seed": seed,
          "d4_recorder": {"calls": calls, "extract_calls": []}, "d4_branches": branches,
          "reads": {"decoded_rows": rows, "a": a_reads, "b": {"cues": b_cues}, "c": {
              "blocks": blocks, "vp": vp, "decoded_patients": vp, "vp_equals_decoded_patients": True,
              "pools_no_block_decodes": []}, "f": {"cues": f_cues}}}
    sh = {"seed": seed, "group": "infer", "arm": arm, "corpus_present": True, "sources": [SG.EXPECTED_SOURCE],
          "pythonhashseed_ok": True, "process": {"ltm_bundle": "wikidata_100k"},
          "builtin_facts": [list(f) for f in BUILTIN] + [["heron", "hunt", "frog"]],
          "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns], "responses": resp, "c2_hops": c2}
    if table in S0.OVERWRITE_TABLES:
        sh["kb_scramble"] = {"mode": "kb_overwrite", "engagement": {"engaged": True, "plan": S0.OVERWRITE_TABLES[table]}}
    if table == S0.XTIER_TABLE:
        rb = {"query_patient": "fish", "all_candidate_patients": ["fish"]}
        c2["ltm_fixture"] = {"ltm_type": "ShardedPhasorStore", "written": True, "store_exception": None,
                             "env_as_read": {"BRAIN_SHARED_LTM": "0", "BRAIN_LTM_RNG_ISOLATE": "1"},
                             "readback_after_write": dict(rb), "readback_before_first_probe": dict(rb)}
    return sh


def _plant(root, seeds=S0.SEEDS, edit=None):
    for seed in seeds:
        for table, arm in S0.TABLE_ARMS.items():
            sh = _good_shard(seed, table)
            if edit:
                sh = edit(seed, table, sh) or sh
            if sh is not None:
                p = S0.shard_path(root, seed, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(sh, open(p, "w"))
        if seed == S0.INERT_SEED:
            for mode in ("recorder", "bare"):
                arm = S0.INERT_ARMS[mode]
                sh = _good_shard(seed, "infer", arm=arm, mode=mode)
                if edit:
                    sh = edit(seed, "inert_" + mode, sh) or sh
                json.dump(sh, open(S0.shard_path(root, seed, arm), "w"))


def test_scorer_go_on_a_planted_clean_set(tmp_path):
    root = str(tmp_path)
    _plant(root)
    res = S0.score(root)
    assert res["verdict"] == "GO" and res["complete_seed_set"] and not res["stops"] and not res["undefined"], res["posts"]
    assert res["inert_h"]["inert"] is True
    # a clean seed-7-only set (the smoke) is not a GO: the rule needs 7/7 seeds
    res7 = S0.score(root, seeds=(7,))
    assert res7["verdict"] == "UNDEFINED" and res7["go_on_scored_seeds"] is True


def _edit(seed_t, table_t, fn):
    def e(seed, table, sh):
        if seed == seed_t and table == table_t:
            fn(sh)
        return sh
    return e


def _set(path):
    def f(sh):
        d = sh
        for k in path[:-2]:
            d = d[k]
        d[path[-2]] = path[-1]
    return f


STOP_PLANTS = {
    "S1_missed_true_match": ("infer", lambda sh: sh["c2_hops"]["reads"]["b"]["cues"]["heron,hunt"].update(
        fired=[], missed_true_matches=[5])),
    "S2_patient_pool": ("infer_amb1", lambda sh: sh["c2_hops"]["reads"]["c"]["blocks"][7].update(
        fired_pools=["frog", "worm"], fires_decoded_patient=False)),
    "S3_probe_differs": ("infer", lambda sh: sh["c2_hops"]["reads"]["c"]["blocks"][2]["probe_vs_production"].update(
        patient={"equal": False, "max_abs_diff": 3.0})),
    "S4_host_routed": ("infer", lambda sh: sh["c2_hops"]["reads"]["a"]["frog,eat"].update(host_routed=True)),
    "S5_or_mismatch": ("infer_dup", lambda sh: sh["c2_hops"]["d4_recorder"]["calls"][0].update(or_matches_return=False)),
    "S6_teach_branch": ("infer_amb1", lambda sh: sh["c2_hops"]["d4_branches"]["t_heron_hunt_worm"].update(
        branch="covered")),
    "S7_known_binding": ("infer_amb2", lambda sh: sh["responses"]["t_frog_eat_apple"]["comprehension"].update(
        known_binding=False)),
    "S8_rewrite_or_rows": ("infer_dup", lambda sh: sh["responses"]["t_heron_hunt_re"].update(
        reconsolidation={"action": "rewrite"})),
    "S9_premise": ("infer", lambda sh: sh["c2_hops"]["reads"]["f"]["cues"]["dog,chase"]["ltm"].update(
        all_candidate_patients=["mouse"])),
    "S10_overwrite": ("infer_amb1", lambda sh: sh["kb_scramble"]["engagement"].update(engaged=False, reason="iv")),
    "S11_xtier": ("infer_xtier", lambda sh: sh["c2_hops"]["ltm_fixture"].update(ltm_type="SharedLTMView",
                                                                                 written=False)),
    "S12_not_inert": ("inert_recorder", lambda sh: sh["responses"]["p_prey_eat"].update(answer="different")),
}


@pytest.mark.parametrize("line", sorted(STOP_PLANTS))
def test_scorer_fires_each_stop_line(tmp_path, line):
    table, fn = STOP_PLANTS[line]
    _plant(str(tmp_path), edit=_edit(3 if not table.startswith("inert") else 7, table, fn))
    res = S0.score(str(tmp_path))
    assert res["verdict"] == "STOP" and res["stop_lines_fired"] == [line], (line, res["stop_lines_fired"],
                                                                            res["undefined"])
    assert line in res["stop_text"]


def test_scorer_stop_details_carry_the_remedy_and_posts():
    sh = _good_shard(7, "infer_xtier")
    sh["c2_hops"]["ltm_fixture"].update(ltm_type="SharedLTMView", written=False,
                                         store_exception="SharedLTMWriteError: read-only")
    r = S0.score_table_shard(sh, 7, "infer_xtier")
    s11 = [s for s in r["stop_lines"] if s["line"] == "S11_xtier"][0]["detail"]
    assert "offiso env did not reach the server" in s11["remedy"] and "never a write through" in s11["remedy"]
    sh = _good_shard(7, "infer")
    sh["c2_hops"]["reads"]["f"]["cues"]["frog,chase"]["ltm"]["all_candidate_patients"] = ["fly"]
    r = S0.score_table_shard(sh, 7, "infer")
    assert not r["stop_lines"] and any("nothing taught" in p["what"] for p in r["posts"])   # posted, not a premise STOP
    sh = _good_shard(7, "infer")
    sh["c2_hops"]["stored_facts_after_teach"].append(["frog", "chase", "fly"])           # an absence premise broken
    r = S0.score_table_shard(sh, 7, "infer")
    assert [s["line"] for s in r["stop_lines"]] == ["S9_premise"]
    sh = _good_shard(7, "infer_dup")
    sh["c2_hops"]["reads"]["decoded_rows"] = [r for r in sh["c2_hops"]["reads"]["decoded_rows"] if r["block"] != 10]
    r = S0.score_table_shard(sh, 7, "infer_dup")
    assert "S8_rewrite_or_rows" in [s["line"] for s in r["stop_lines"]]
    assert any("B11 is predicted UNDEFINED" in p["what"] for p in r["posts"])


@pytest.mark.parametrize("what", ["corpus_false", "corpus_missing", "shard_missing", "wrong_bundle", "reads_raised"])
def test_scorer_undefined_rules(tmp_path, what):
    def fn(sh):
        if what == "corpus_false":
            sh["corpus_present"] = False
        elif what == "corpus_missing":
            del sh["corpus_present"]
        elif what == "wrong_bundle":
            sh["process"]["ltm_bundle"] = "wikidata_core_15k"
        elif what == "reads_raised":
            sh["c2_hops"]["reads"] = {"error": "RuntimeError: boom"}

    def edit(seed, table, sh):
        if seed == 2 and table == "infer_amb2":
            if what == "shard_missing":
                return None
            fn(sh)
        return sh
    root = str(tmp_path)
    _plant(root, edit=edit)
    if what == "shard_missing":
        os.remove(S0.shard_path(root, 2, "c2s0_amb2"))
    res = S0.score(root)
    assert res["verdict"] == "UNDEFINED" and not res["stops"], (what, res["stop_lines_fired"])
    assert any(u["seed"] == 2 and u["table"] == "infer_amb2" for u in res["undefined"])


def test_a_corpus_absent_shard_is_never_a_stop_input(tmp_path):
    """packet 2.10: a corpus_present-false shard is UNDEFINED for every check in it, never a GO or STOP input."""
    def edit(seed, table, sh):
        if seed == 4 and table == "infer":
            sh["corpus_present"] = False
            sh["c2_hops"]["reads"]["a"]["frog,eat"]["host_routed"] = True
        return sh
    _plant(str(tmp_path), edit=edit)
    res = S0.score(str(tmp_path))
    assert res["verdict"] == "UNDEFINED" and not res["stops"]


def test_scorer_not_go_when_a_go_clause_fails_that_no_stop_line_names(tmp_path):
    _plant(str(tmp_path), edit=_edit(5, "infer", lambda sh: sh["c2_hops"].update(ws_seed=42)))
    res = S0.score(str(tmp_path))
    assert res["verdict"] == "NOT_GO" and not res["stops"]
    assert any("ws_seed" in p["what"] for p in res["posts"])


def test_premise_map_is_the_packets():
    rows = {r["probe"]: r for r in S0.PREMISE_MAP["infer_amb1"]}
    assert rows["p_prey_eat"]["cues"][("heron", "hunt")] == (frozenset({"frog", "worm"}), frozenset())
    rows = {r["probe"]: r for r in S0.PREMISE_MAP["infer_amb2"]}
    assert rows["p_prey_eat"]["cues"][("frog", "eat")][0] == {"gnat", "apple"}
    rows = {r["probe"]: r for r in S0.PREMISE_MAP["infer_xtier"]}
    assert rows["p_prey_eat"]["cues"][("heron", "hunt")] == (frozenset({"frog"}), frozenset({"fish"}))
    rows = {r["probe"]: r for r in S0.PREMISE_MAP["infer"]}
    assert rows["c_prey_chase"]["absence"] == [("frog", "chase")] and rows["p_yesno"]["reported_only"]
    rows = {r["probe"]: r for r in S0.PREMISE_MAP["infer_dup"]}
    assert "t_heron_hunt_re" in rows["p_prey_eat"]["sources"] and "t_frog_eat_re" in rows["p_prey_eat"]["sources"]
    for table in SG.C2_TABLES:            # every map probe is a probe of its table
        labels = {l for l, _m, r in SG.C2_TABLES[table] if r != "teach"}
        assert {r["probe"] for r in S0.PREMISE_MAP[table]} <= labels, table


# ── the pool lines ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_pool_lines_seed7_first_one_table_seed_per_line_stage2_prefix():
    lines = S0.pool_lines("<sha>")
    assert len(lines) == 7 + 6 * 5
    seeds = [int(c.split("--seed ")[1].split()[0]) for c, _w in lines]
    assert seeds[:7] == [7] * 7 and seeds[7:] == sorted(seeds[7:]) and set(seeds) == set(S0.SEEDS)
    gate = SG.pool_job_lines("<sha>", seeds=(1,))[0][0]
    pre = gate.split(" && mkdir -p")[0]
    for cmd, why in lines:
        assert cmd.startswith(pre.replace("mem_ok.sh %d" % SG.MEM_GB, "mem_ok.sh %d" % S0.MEM_GB))
        assert ("bash tools/memcap.sh %d -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                "research.runners.stage2_exit_gate --shard --seed" % S0.MEMCAP_GB) in cmd
        assert "--allow-gate-seed" not in cmd and "--prereg" not in cmd
        assert " --out research/findings/raw/_c2_step0/s" in cmd and "~/derisk-pool/revisions/<sha>" in cmd
        assert why.endswith("mem_gb=%d max_age_h=240" % S0.MEM_GB) and "record:" in why
        assert not any(("--seed %d " % g) in cmd for g in SG.GATE_SEEDS)
    arms7 = [c.split("--arm ")[1].split()[0] for c, _w in lines[:7]]
    assert arms7 == list(S0.TABLE_ARMS.values()) + ["c2s0_infer_rec", "c2s0_infer_bare"]
    with pytest.raises(ValueError):
        S0.pool_lines("<sha>", seeds=(42,))


def test_the_committed_pool_lines_file_matches_the_emitter():
    p = os.path.join(os.path.dirname(__file__), "..", "research", "coordination", "c2-step0-pool-lines.txt")
    if not os.path.exists(p):
        pytest.fail("research/coordination/c2-step0-pool-lines.txt is missing")
    got = [l.rstrip("\n") for l in open(p) if l.strip()]
    assert got == ["%s\t%s" % x for x in S0.pool_lines("<sha>")]
