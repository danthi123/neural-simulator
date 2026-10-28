"""C2 host-ownership intervention (research/runners/stage2_exit_gate.py, informational arms `kb_scramble` and
`kb_scramble_spiking_control`): the scramble is a derangement, the read-back from `.kb` matches the plan, the chain
route follows the list (and, with its own fallback condition forced, the bound store), and every pre-existing arm of
the gate is unchanged. No brain build: a fake `webapp.server` drives the real `run_worker` loop, the real
`resolve_compositional_chain` and the real `KbProbe` instrumentation over a stub composer."""
import hashlib
import json
import os
import sys
import types
from collections.abc import MutableMapping

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import compositional_chain_route as CCR
from research.runners import stage2_exit_gate as SG

KB_ARMS = ("kb_scramble", "kb_scramble_spiking_control")
# C2 spiking-hops packet (research/coordination/c2-spiking-hops-PACKET.md, round 9) 2.8 / 2.10 step 0 (7): the two
# assertions below that pinned "the only new arms are the two kb arms" and "no other arm carries a scramble" are now
# EXPLICIT ALLOWLISTS, extended by each packet step that adds arms. Step 0 adds the seven read-only step-0 entries (the
# five tables of 2.8 and the two inert-check workers of (h)); `infer_amb1` and `infer_dup` apply A17's and A19's `.kb`
# overwrite fixture (g) in mode `kb_overwrite`. The GOLDEN dict and the per-arm fingerprint loop are NOT changed.
C2_STEP0_ARMS = ("c2s0_infer", "c2s0_amb1", "c2s0_amb2", "c2s0_dup", "c2s0_xtier", "c2s0_infer_rec", "c2s0_infer_bare")
EXTRA_ARMS = {"infer": list(KB_ARMS) + list(C2_STEP0_ARMS)}          # every other group's extras: none
# ask: empty until M3 step 3 adds `m3_ask` (packet 3.6, round 9); a table implemented as its own group is named here too
SCRAMBLE_MODES = {("infer", "kb_scramble"): "kb", ("infer", "kb_scramble_spiking_control"): "kb_spiking",
                  ("infer", "c2s0_amb1"): "kb_overwrite", ("infer", "c2s0_dup"): "kb_overwrite"}

# sha256[:16] of json([turns_for(group, 7, arm), arm_spec(group, arm)]) for every arm that existed BEFORE the kb_scramble
# arms were added (main @ 3065c5614). A change to any pre-existing arm's turn table, env, patch or kind breaks this.
GOLDEN = {
    "ask/da_lesion": "8a2bceafa4a74c83", "ask/intact_a": "5dc587dc857aed4c", "ask/intact_b": "5dc587dc857aed4c",
    "ask/lesion_curiosity": "c9dd8e60424d4e01", "ask/lesion_curiosity.rep0": "c9dd8e60424d4e01",
    "ask/lesion_graded": "67b65bfed162dd8c", "ask/lesion_graded.rep0": "67b65bfed162dd8c",
    "ask/permuted": "d0deb0e53217930e", "ask/reversed": "a3d5c4044328ebe4",
    "conv/intact_a": "048d7917ab38c58c", "conv/intact_b": "048d7917ab38c58c", "conv/lesion_mouth": "4ba6e4336b5d7fce",
    "conv/lesion_mouth.rep0": "4ba6e4336b5d7fce", "conv/permuted": "471ed6c62564874f",
    "emo/intact_a": "85ab22da6652c064", "emo/intact_b": "85ab22da6652c064", "emo/lesion_affect": "33d6e4bce654ce9e",
    "emo/lesion_affect.rep0": "33d6e4bce654ce9e", "emo/permuted": "e5ef7d3daf92e490",
    "infer/intact_a": "3112f313c8c69357", "infer/intact_b": "3112f313c8c69357",
    "infer/lesion_chain_route": "f9c6559a012820e2", "infer/lesion_chain_route.rep0": "f9c6559a012820e2",
    "infer/permuted": "cfdc5565b6b40a05", "infer/qp_lesion": "77022f2c64fed232", "infer/transitive_chat": "ab4ff8bb361cf358",
    "inner/intact_a": "53b68d0a552c4dea", "inner/intact_b": "53b68d0a552c4dea", "inner/lesion_selfinit": "6da8fd2dd84f43c9",
    "inner/lesion_selfinit.rep0": "6da8fd2dd84f43c9", "inner/lesion_workspace": "14171c0be54e0de1",
    "inner/lesion_workspace.rep0": "14171c0be54e0de1", "inner/permuted": "7cfc10c31e10d8fa",
}


def _fp(group, arm):
    return hashlib.sha256(json.dumps([SG.turns_for(group, 7, arm), list(map(str, SG.arm_spec(group, arm)))],
                                     sort_keys=True).encode()).hexdigest()[:16]


# ── the arms: informational, uncounted, and every pre-existing arm untouched ───────────────────────────────────────────
def test_every_pre_existing_arm_has_its_frozen_turn_table_and_spec():
    for key, want in GOLDEN.items():
        group, arm = key.split("/", 1)
        assert arm in SG.arms_for(group), key
        assert _fp(group, arm) == want, key
    for group in SG.GROUPS:                                  # nothing else changed: the only new arms are the allowlist's
        extra = [a for a in SG.arms_for(group) if "%s/%s" % (group, a) not in GOLDEN]
        assert extra == EXTRA_ARMS.get(group, []), (group, extra)


def test_kb_arms_are_informational_never_counted_and_run_the_intact_turn_table():
    assert not set(KB_ARMS) & set(SG.arms_for("infer", with_informational=False))
    for arm in KB_ARMS:
        assert SG.arm_spec("infer", arm) == ({}, None, "informational")
        assert SG.turns_for("infer", 7, arm) == SG.turns_for("infer", 7, "intact_a")
    assert SG.arm_scramble("infer", "kb_scramble") == "kb"
    assert SG.arm_scramble("infer", "kb_scramble_spiking_control") == "kb_spiking"
    got = {(group, arm): SG.arm_scramble(group, arm) for group in SG.GROUPS for arm in SG.arms_for(group)
           if SG.arm_scramble(group, arm) is not None}
    assert got == SCRAMBLE_MODES                             # no arm outside the explicit arm-to-mode map carries one


# ── the derangement ────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_derange_patients_is_a_derangement_that_preserves_the_patients(n):
    agents = ["a%d" % i for i in range(n)]
    patients = ["p%d" % i for i in range(n)]
    new = SG.derange_patients(agents, patients)
    assert len(new) == n and all(o != p for o, p in zip(patients, new))          # no fixed point
    assert sorted(new) == sorted(patients) and len(set(new)) == n                # same distinct patients, a permutation
    assert all(a != p for a, p in zip(agents, new))                              # no self-referential fact


def test_derange_patients_refuses_what_it_cannot_honestly_derange():
    with pytest.raises(ValueError):
        SG.derange_patients(["a"], ["x"])                                        # one fact
    with pytest.raises(ValueError):
        SG.derange_patients(["a", "b"], ["x", "x"])                              # patients not distinct
    with pytest.raises(ValueError):
        SG.derange_patients(["x", "y"], ["y", "x"])                              # every shift would be self-referential


# ── the engagement record over a stub composer ─────────────────────────────────────────────────────────────────────────
BUILTIN = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
           ("dog", "chase", "cat"), ("cat", "eat", "fish")]
TOLD = [["heron", "hunt", "frog"], ["frog", "eat", "gnat"], ["crow", "chase", "wren"], ["wren", "chase", "moth"]]
CFG = SG.GROUPS["infer"]["scramble_cfg"]
Q = "what does the heron's prey eat"


class _Buf:
    def __init__(self, facts):
        self.kb = [({"agent": a, "action": v, "patient": p, "polarity": "affirm"}, None) for a, v, p in facts]


class _Tiered:
    """TieredFactStore-shaped stub: `.buffer.kb` is the list the route scans first; `query_patient` reads a separate
    `bound` dict = the bound store the substrate would read (set at teach time, never touched by the scramble)."""
    def __init__(self, facts):
        self.buffer = _Buf(facts)
        self.ltm = None
        self.bound = {(a, v): p for a, v, p in facts}
        self.qp_calls = []

    def query_patient(self, agent, action, order_fn=None):
        r = self.bound.get((agent, action))
        self.qp_calls.append((agent, action, r))
        return r


def _stub():
    return _Tiered(BUILTIN + [tuple(t) for t in TOLD])


def test_apply_kb_scramble_reads_back_a_deranged_list_and_predicts_the_conclusion():
    comp = _stub()
    rec = SG.apply_kb_scramble(comp, TOLD, CFG["keep"], CFG["probe"])
    assert rec["engaged"] is True, rec
    after = [[f["agent"], f["action"], f["patient"]] for f, _ in comp.buffer.kb]        # read back independently here
    assert after == rec["kb_after"] and rec["readback_matches_plan"] is True
    before = rec["kb_before"]
    assert before == [list(t) for t in BUILTIN + [tuple(t) for t in TOLD]]
    # exactly the scoped facts changed; the hop-1 bridge and the built-ins did not
    assert rec["n_scoped"] == rec["n_changed"] == 3 and rec["n_facts"] == len(BUILTIN) + 4
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert [before[i][:2] for i in changed] == [["frog", "eat"], ["crow", "chase"], ["wren", "chase"]]
    assert after[len(BUILTIN)] == ["heron", "hunt", "frog"] and after[:len(BUILTIN)] == before[:len(BUILTIN)]
    assert rec["is_derangement"] and rec["patients_distinct_after"] and rec["multiset_preserved"]
    # the polarity key (and any other fact field) survives the rewrite
    assert all(f.get("polarity") == "affirm" for f, _ in comp.buffer.kb)
    # the conclusion-relevant fact changed, and the two predictions differ
    assert rec["conclusion_fact_changed"] is True
    assert rec["predicted_from_bound"]["svo"] == ["frog", "eat", "gnat"]
    assert rec["predicted_from_list"]["svo"] == ["frog", "eat", "wren"]
    assert rec["predicted_from_list"]["derived_from"] == [["heron", "hunt", "frog"], ["frog", "eat", "wren"]]
    # the bound store the substrate reads was not touched
    assert comp.bound[("frog", "eat")] == "gnat"


def test_apply_kb_scramble_is_not_engaged_when_the_teach_did_not_land_or_nothing_is_scannable():
    comp = _Tiered(BUILTIN + [("heron", "hunt", "frog")])                             # the hop-2 premise was never stored
    rec = SG.apply_kb_scramble(comp, TOLD, CFG["keep"], CFG["probe"])
    assert rec["engaged"] is False and rec["reason"]
    rec = SG.apply_kb_scramble(object(), TOLD, CFG["keep"], CFG["probe"])
    assert rec["engaged"] is False and "no .kb" in rec["reason"]


def test_the_oracle_matches_the_route_on_the_intact_and_deranged_lists():
    comp = _stub()
    route = CCR.resolve_compositional_chain(comp, Q)
    assert list(route) == SG.list_chain([list(t) for t in BUILTIN] + TOLD, "heron", ["hunt", "eat"], "eat")["svo"]
    SG.apply_kb_scramble(comp, TOLD, CFG["keep"], CFG["probe"])
    route = CCR.resolve_compositional_chain(comp, Q)
    triples = [[f["agent"], f["action"], f["patient"]] for f, _ in comp.buffer.kb]
    want = SG.list_chain(triples, "heron", ["hunt", "eat"], "eat")
    assert list(route) == want["svo"] == ["frog", "eat", "wren"] and route.derived_from == want["derived_from"]
    assert comp.qp_calls == []                                    # the route never touched the spiking read (the diagnosis)


def test_a_multi_valued_hop_abstains_in_the_oracle_like_the_route():
    t = [["heron", "hunt", "frog"], ["heron", "hunt", "toad"], ["frog", "eat", "gnat"]]
    assert SG.list_chain(t, "heron", ["hunt", "eat"], "eat") is None


# ── the whole worker loop over a fake brain: pre-existing arms untouched, kb arms engaged and read ─────────────────────
class _FakeChats(MutableMapping):
    def __init__(self):
        self.d = {}

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def __delitem__(self, k):
        del self.d[k]

    def __iter__(self):
        return iter(self.d)

    def __len__(self):
        return len(self.d)


class _FakeChat:
    def __init__(self):
        self.inner = types.SimpleNamespace(composer=_Tiered(BUILTIN))
        self.stored_facts = [list(t) for t in BUILTIN]


def _fake_server():
    srv = types.ModuleType("webapp.server")
    srv._BRAIN_CHATS = _FakeChats()
    srv._resolve_ltm_bundle = lambda: "wikidata_100k"
    srv.kb_writes = []

    class BrainChatRequest:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def brain_chat(req):
        key = (req.session, req.brain, req.renderer)
        if req.reset or key not in srv._BRAIN_CHATS:
            srv._BRAIN_CHATS[key] = _FakeChat()
        chat = srv._BRAIN_CHATS[key]
        comp = chat.inner.composer
        msg = req.message
        t = SG._canonical_teach(msg)
        body = {"answer": "ok", "abstained": False, "recalled_svo": None, "derived": False, "derived_svo": None,
                "derived_from": None, "source": SG.EXPECTED_SOURCE}
        if t and msg.startswith("the "):
            comp.buffer.kb.append(({"agent": t[0], "action": t[1], "patient": t[2]}, None))
            comp.bound[(t[0], t[1])] = t[2]
            chat.stored_facts.append(list(t))
            body.update(answer=msg, recalled_svo=list(t))
        elif "'s prey" in msg:
            r = CCR.resolve_compositional_chain(comp, msg)
            if r is None:
                body.update(answer="I don't know about that.", abstained=True)
            else:
                body.update(answer="I derived this from ...", derived=True, derived_svo=list(r),
                            derived_from=[list(h) for h in r.derived_from])
        else:                                                   # a direct recall reads the BOUND store (the substrate)
            ag = msg.split("the ")[-1].split()[0]
            act = msg.split()[-1]
            p = comp.bound.get((ag, act))
            body.update(answer=("the %s %ss the %s" % (ag, act, p)) if p else "I don't know about that.",
                        abstained=p is None, recalled_svo=[ag, act, p] if p else None)
        return types.SimpleNamespace(body=json.dumps(body))

    srv.brain_chat, srv.BrainChatRequest = brain_chat, BrainChatRequest
    return srv


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    srv = _fake_server()
    pkg = types.ModuleType("webapp")
    pkg.__path__ = []
    pkg.server = srv
    monkeypatch.setitem(sys.modules, "webapp", pkg)
    monkeypatch.setitem(sys.modules, "webapp.server", srv)
    from research.runners import one_brain_composer as OBC
    # KbProbe patches these two module attributes in-process; register their restore before any worker installs it
    monkeypatch.setattr(CCR, "_distinct_patients", CCR._distinct_patients)
    monkeypatch.setattr(OBC.OneBrainComposer, "query_patient", OBC.OneBrainComposer.query_patient)
    base_env = dict(os.environ)

    def run(arm, group="infer"):
        # production runs each arm in its OWN process; run_worker sets the arm's env flags on os.environ, so each call
        # here gets a private copy of the pre-test environment (a lesion arm's BRAIN_CHAIN_ROUTE=0 must not leak on)
        monkeypatch.setattr(os, "environ", dict(base_env))
        out = str(tmp_path / ("%s__%s.json" % (group, arm)))
        assert SG.run_worker(7, group, arm, out) == 0
        return json.load(open(out)), srv
    return run


def _probe_fields(shard, label="p_prey_eat"):
    r = shard["responses"][label]
    return r["derived_svo"], r["derived_from"]


def test_a_pre_existing_arm_runs_untouched_through_the_worker(fake_worker):
    for arm in ("intact_a", "qp_lesion", "lesion_chain_route"):
        shard, _srv = fake_worker(arm)
        assert "kb_scramble" not in shard                                             # no new key on any other arm
        assert set(shard) == {"runner", "prereg", "seed", "group", "arm", "kind", "criterion", "env", "patch",
                              "pythonhashseed_ok", "corpus_present", "builtin_facts", "session", "turns", "responses",
                              "internal", "timings_s", "import_s", "wall_s", "elapsed_seconds", "expected_source",
                              "process", "sources"}
        assert [t["label"] for t in shard["turns"]] == [t[0] for t in SG.turns_for("infer", 7, arm)]
    shard, _srv = fake_worker("intact_a")
    ds, df = _probe_fields(shard)
    assert ds == ["frog", "eat", "gnat"] and df == [["heron", "hunt", "frog"], ["frog", "eat", "gnat"]]


def test_kb_scramble_arm_follows_the_list_and_the_control_follows_the_bound_vectors(fake_worker):
    shard, _srv = fake_worker("kb_scramble")
    ks = shard["kb_scramble"]
    assert ks["mode"] == "kb" and ks["engagement"]["engaged"] is True and ks["engagement"]["n_changed"] == 3
    assert ks["engagement"]["composer_types"]["buffer"] == "_Buf"
    assert ks["engagement"]["chat_stored_facts_after"][-4:] == TOLD                  # the chat's own snapshot is untouched
    assert ks["engagement"]["told"] == TOLD
    ds, df = _probe_fields(shard)
    assert ds == ["frog", "eat", "wren"] and df == [["heron", "hunt", "frog"], ["frog", "eat", "wren"]]
    reads = [(d["agent"], d["action"], d["result"]) for d in ks["distinct_calls"] if d["label"] == "p_prey_eat"]
    assert reads == [("heron", "hunt", ["frog"]), ("frog", "eat", ["wren"])]            # the route's list reads, recorded
    # the direct recall of the SAME fact reads the bound store, not the list
    assert shard["responses"]["c_frog_eat"]["recalled_svo"] == ["frog", "eat", "gnat"]
    rd = SG.kb_scramble_read(shard)
    assert rd["follows"] == "scrambled list" and rd["engaged"] is True

    shard, srv = fake_worker("kb_scramble_spiking_control")
    ks = shard["kb_scramble"]
    assert ks["mode"] == "kb_spiking" and ks["engagement"]["engaged"] is True
    ds, df = _probe_fields(shard)
    assert ds == ["frog", "eat", "gnat"] and df == [["heron", "hunt", "frog"], ["frog", "eat", "gnat"]]
    assert all(d["forced_unscannable"] and d["result"] is None for d in ks["distinct_calls"] if d["label"] == "p_prey_eat")
    chat = srv._BRAIN_CHATS[("s2_infer_kb_scramble_spiking_control_7", SG.BRAIN_NAME, SG.RENDERER)]
    assert ("frog", "eat", "gnat") in chat.inner.composer.qp_calls                      # the hops were spiking-store reads
    rd = SG.kb_scramble_read(shard)
    assert rd["follows"] == "bound vectors"


def test_kb_scramble_read_withholds_the_follow_read_when_not_engaged_and_diffs_against_intact(fake_worker):
    shard, _ = fake_worker("kb_scramble")
    intact, _ = fake_worker("intact_a")
    rd = SG.kb_scramble_read(shard, intact=intact)
    assert rd["follows"] == "scrambled list" and "p_prey_eat" in rd["changed_vs_intact"]
    assert "c_frog_eat" not in rd["changed_vs_intact"]                                  # bound-store recalls did not move
    bad = json.loads(json.dumps(shard))
    bad["kb_scramble"]["engagement"]["engaged"] = False
    assert SG.kb_scramble_read(bad)["follows"] == "not engaged"
    assert SG.kb_scramble_read({"responses": {}})["follows"] == "not engaged"           # a shard from a non-kb arm


def test_aggregate_reports_the_kb_arms_as_informational_and_never_in_the_verdict(fake_worker, tmp_path):
    from tests import test_stage2_exit_gate as T                  # the ideal-brain shard builder (every group, every arm)
    td = str(tmp_path / "gate")
    T._e2e_build(td, (7,), T._e2e_ideal)
    before = SG.aggregate(td, seeds=(7,))
    for arm in KB_ARMS:                                           # replace the two kb shards with the fake-brain worker output
        shard, _ = fake_worker(arm)
        json.dump(shard, open(SG.shard_path(td, 7, "infer", arm), "w"))
    agg = SG.aggregate(td, seeds=(7,))
    info = agg["groups"]["infer"]["seeds"][7]["informational"]
    assert info["kb_scramble"]["kb_scramble"]["follows"] == "scrambled list"
    assert info["kb_scramble_spiking_control"]["kb_scramble"]["follows"] == "bound vectors"
    # never counted: the verdict, the criteria, the lesion and permuted reads are identical with or without the kb arms' content
    for k in ("verdict", "criteria", "GO"):
        assert agg[k] == before[k], k
    assert agg["groups"]["infer"]["verdict"] == before["groups"]["infer"]["verdict"] == "PASS"
    assert not set(KB_ARMS) & set(agg["groups"]["infer"]["seeds"][7].get("lesions") or {})
    rep = SG.report(agg)
    assert "C2 host-ownership intervention" in rep and "conclusion follows: scrambled list" in rep
    assert "conclusion follows: bound vectors" in rep


def test_kb_summary_saves_the_probe_fields_the_engagement_and_the_follow_read(fake_worker, tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    for arm in SG.KB_SUMMARY_ARMS:
        shard, _ = fake_worker(arm)
        json.dump(shard, open(str(d / ("infer__%s.json" % arm)), "w"))
    res = SG.kb_scramble_summary({7: str(d)})
    row = res["seeds"]["7"]
    assert set(row["turns"]) == set(SG.KB_SUMMARY_ARMS) and row["shards"]["intact_a"]["sha256"]
    assert row["turns"]["intact_a"]["p_prey_eat"]["derived_svo"] == ["frog", "eat", "gnat"]
    assert row["turns"]["kb_scramble"]["p_prey_eat"]["derived_svo"] == ["frog", "eat", "wren"]
    assert row["reads"]["kb_scramble"]["follows"] == "scrambled list"
    assert row["reads"]["kb_scramble_spiking_control"]["follows"] == "bound vectors"
    assert row["reads"]["kb_scramble"]["n_changed"] == 3 and "p_prey_eat" in row["reads"]["kb_scramble"]["changed_vs_intact"]
    # the fake composer is not the OneBrainComposer class the wrapper patches, so no spiking read is recorded here; the
    # structure is what is pinned (teach turns are never listed, intact_a has no instrumentation)
    assert isinstance(row["spiking_reads_by_turn"]["kb_scramble"], dict) and "intact_a" not in row["spiking_reads_by_turn"]
    assert not any(l.startswith("t_") for l in row["spiking_reads_by_turn"]["kb_scramble"])
    assert row["flagged_guesses"]["intact_a"] == {}
    # a missing arm shard is recorded as absent, never filled in
    os.remove(str(d / "infer__kb_scramble.json"))
    row = SG.kb_scramble_summary({7: str(d)})["seeds"]["7"]
    assert row["shards"]["kb_scramble"] is None and "kb_scramble" not in row["turns"]
