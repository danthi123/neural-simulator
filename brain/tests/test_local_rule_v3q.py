# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md
# Expected-fail: none (plumbing on a tiny random cortex and synthetic records; no model weights)
"""Local rule v3q: cues are only life questions or the cortex's own (refused test lives, pinned file, excluded facts
counted), targets sit at the answer-predicting rows only, the lesion is inert, top_only moves only the last block, the
graded read-out is right, G0Q gates the nights, and the verdict gives the exact word with each clause and each
engagement gate failing alone. Engine specs render, are DRAFT, chain and delegate."""
import copy
import json
import math
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from brain.experiments import local_rule_v3 as V
from brain.experiments import local_rule_v3q as Q
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_source_tag import _Cortex

TF = [("my cat is Pixel", "what is my cat?", "pixel"), ("my dog is Bolt", "what is my dog?", "bolt"),
      ("my car is Volvo", "what car do I drive?", "volvo")]
LIFE = {"pixel": [{"q": "what is my cat?"}], "bolt": []}
SELF = {0: ["Is it nice?"], 1: ["Tell me more", "what pet do I have?", "what pet do I have?", "I said bolt?"], 2: []}
GEN = ["User: hi\n\nAssistant: yo\n\n"]


def _pc(noise=0.0):
    torch.manual_seed(1)
    p = PlasticCortex(_Cortex(), rank=2)
    if noise:
        with torch.no_grad():
            for d in (p.B_k, p.B_v):
                for v in d.values():
                    v.normal_(0, noise)
    return p


def _st(pc):
    return [s.float() for s in pc.zero_state()]


def _fake_answer(pc, ctx, prefix, st):
    text = pc.cortex.pipe.decode(list(ctx))
    for s, _, k in TF:
        if s in text:
            return f" It is {k.capitalize()}."
    return " no idea"


@pytest.fixture
def fake_cues(monkeypatch):
    monkeypatch.setattr(Q, "teacher_answer", _fake_answer)
    monkeypatch.setattr(Q, "self_cues", lambda pc, told, st, item: list(SELF[item]))
    monkeypatch.setattr(Q, "G0Q_MIN", 2)


def _cs(pc=None):
    pc = pc or _pc()
    return Q.build_cue_set(pc, _st(pc), TF, LIFE)


def _told_ids(pc):
    from brain.experiments.sleep_probe import _told
    return {j: pc.cortex.pipe.encode(_told(s)[0]) for j, (s, _, _) in enumerate(TF)}


# ------------------------------------------------------------------ cues
def test_key_as_told_and_cue_problem():
    assert Q.key_as_told("I'm allergic to peanuts.", "peanut") == "peanuts"
    assert Q.key_as_told("My sister's name is Marguerite.", "marguerite") == "Marguerite"
    with pytest.raises(ValueError):
        Q.key_as_told("My cat", "volvo")
    assert Q.cue_problem("What is my cat called?", "pixel") is None
    assert Q.cue_problem("My cat is Pixel.", "pixel") == "not a question"
    assert Q.cue_problem("Is it Pixel?", "pixel") == "names the answer"
    assert Q.cue_problem("Hi?\nThere?", "pixel") == "empty or multi-line" and Q.cue_problem("", "x") == "empty or multi-line"


def _life(split="dev"):
    rows = [{"day": 1, "kind": "tell", "you": "My cat is called Pixel.", "item": "pixel"},
            {"day": 1, "kind": "sameday", "you": "What is my cat called?", "item": "pixel"},
            {"day": 5, "kind": "remention", "you": "What is my cat called?", "item": "pixel"},
            {"day": 6, "kind": "remention", "you": "Is my cat Pixel?", "item": "pixel"},
            {"day": 7, "kind": "small", "you": "What is the weather?", "item": None},
            {"day": 8, "kind": "remention", "you": "Which car do I own?", "item": "volvo"}]
    return {"seed": 319, "split": split, "model": "m", "lived": [
        {"line": "trunk", "rows": rows},
        {"line": "prunedA", "rows": [{"day": 9, "kind": "remention", "you": "Who is my sister?", "item": "marguerite"}]}]}


def test_extract_life_cues_reads_trunk_user_questions_and_refuses_test_lives():
    d = Q.extract_life_cues([("a.json", _life())], ["pixel", "volvo", "marguerite", "teal"])
    assert [c["q"] for c in d["items"]["pixel"]] == ["What is my cat called?"]          # deduped; key-naming dropped
    assert d["items"]["volvo"][0] == {"q": "Which car do I own?", "life_seed": 319, "day": 8, "kind": "remention"}
    assert d["items"]["marguerite"] == [] and d["items"]["teal"] == []                  # only the trunk line is read
    for split in ("test", "test2", None):
        with pytest.raises(SystemExit, match="split"):
            Q.extract_life_cues([("t.json", _life(split))], ["pixel"])


def test_pinned_life_cues_file_loads_and_a_changed_file_refuses(tmp_path):
    life = Q.load_life_cues()
    assert set(life) == {k for _, _, k in __import__("brain.experiments.facts", fromlist=["FACTS"]).FACTS}
    assert sum(bool(v) for v in life.values()) == 3 and all(Q.cue_problem(c["q"], k) is None for k, v in life.items() for c in v)
    meta = json.load(open(Q.LIFE_CUES_PATH))
    assert all(m["split"] == "dev" for m in meta["lives"])
    p = tmp_path / "c.json"
    shutil.copy(Q.LIFE_CUES_PATH, p)
    with open(p, "a") as fh:
        fh.write(" ")
    with pytest.raises(SystemExit, match="sha256"):
        Q.load_life_cues(str(p))


def test_build_cue_set_life_first_self_cues_filtered_and_excluded_counted(fake_cues):
    pc = _pc()
    cs = _cs(pc)
    u = cs["units"]
    assert [(x["item"], x["cue"], x["source"]) for x in u] == [(0, "what is my cat?", "life"), (0, "Is it nice?", "self"),
                                                               (1, "what pet do I have?", "self")]
    assert cs["n_items"] == 2 and cs["items_kept"] == [0, 1] and cs["n_excluded"] == 1
    assert cs["excluded"] == {"volvo": "no brain-sourced question (no life question, no self-generated text)"}
    assert {(r["cue"], r["why"]) for r in cs["rejected"]} == {("Tell me more", "not a question"),
                                                               ("I said bolt?", "names the answer")}
    assert cs["eval_cue_items"] == [0] and u[0]["cue_is_eval"] and not u[1]["cue_is_eval"]
    assert cs["by_source"] == {"life": 1, "self": 2} and cs["pass"] is True
    pipe = pc.cortex.pipe
    for x in u:                                             # rows predict exactly the answer tokens
        ans = pipe.encode(x["answer"])
        assert [x["qa_ids"][p + 1] for p in x["pos"]] == ans
        assert pipe.decode(x["qa_ids"][:x["pos"][0] + 1]) == Q.cue_prefix(x["cue"])
    assert cs["cue_sha256"] == _cs()["cue_sha256"]           # deterministic


def test_a_cue_the_teacher_cannot_answer_is_rejected_and_g0q_gates(fake_cues, monkeypatch):
    monkeypatch.setattr(Q, "teacher_answer", lambda pc, ctx, prefix, st: " no idea")
    cs = _cs()
    assert cs["n_items"] == 0 and cs["pass"] is False and set(cs["excluded"]) == {"pixel", "bolt", "volvo"}
    assert cs["excluded"]["pixel"] == "no candidate cue survived (see rejected)"
    assert Q.g0q_pass(2) and not Q.g0q_pass(1) and not Q.g0q_pass(None) and not Q.g0q_pass(True)


def test_cues_per_item_cap(fake_cues, monkeypatch):
    monkeypatch.setattr(Q, "self_cues", lambda pc, told, st, item: [f"q{n}?" for n in range(9)])
    cs = _cs()
    assert max(sum(x["item"] == i for x in cs["units"]) for i in range(3)) == Q.CUES_PER_ITEM


def test_gen_text_is_deterministic_and_self_cues_use_the_turn_marker():
    pc = _pc(0.1)
    ids = pc.cortex.pipe.encode("User: hi\n\nAssistant:")
    a = Q.gen_text(pc, ids, _st(pc), 8)
    assert a == Q.gen_text(pc, ids, _st(pc), 8) and "\n\n" not in a
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert Q.gen_text(pc, ids, _st(pc), 8, gen=g1) == Q.gen_text(pc, ids, _st(pc), 8, gen=g2)
    c1 = Q.self_cues(pc, ids, _st(pc), 0)
    assert len(c1) == Q.CUE_SAMPLES and c1 == Q.self_cues(pc, ids, _st(pc), 0) and pc.enabled is True


def test_real_teacher_answer_is_the_pretrained_cortex():
    pc = _pc(0.5)
    ids = pc.cortex.pipe.encode("User: hi\n\nAssistant:")
    a = Q.teacher_answer(pc, [], ids, _st(pc))
    pc.enabled = False
    assert a == Q.gen_text(pc, ids, _st(pc), Q.ANSWER_MAX_TOKENS)
    pc.enabled = True


# ------------------------------------------------------------------ the night
def test_rows_slices_every_recorded_tensor():
    pc = _pc(0.1)
    ids = pc.cortex.pipe.encode("User: x\n\nAssistant: y")
    _, act = V.activity(V.adopt3(pc), ids, _st(pc))
    r = Q._rows(act, [2, 5])
    for i in act:
        for full, cut in zip(act[i], r[i]):
            assert torch.equal(cut, full[[2, 5]])


def test_v3q_night_engages_at_answer_rows_lesion_inert_top_only_and_sham(fake_cues):
    pc = _pc()
    cs = _cs(pc)
    tid = _told_ids(pc)
    info = {}
    Q.v3q_sleep(pc, _st(pc), cs["units"], tid, GEN, steps=40, lr=0.3, info=info)
    assert pc.adapter_norm() > 0 and info["err_end"] < info["err_start"] and info["layers_on"] == [0, 1]
    assert info["target_form"] == "question" and info["teacher"] == "double" and info["n_items"] == 2
    assert info["target_rows"] == sum(len(u["pos"]) for u in cs["units"]) / 3 and info["target_ctx_tokens"] > 0
    pc = _pc()
    info = {}
    Q.v3q_sleep(pc, _st(pc), cs["units"], tid, GEN, steps=10, lr=0.3, teacher_kind="none", info=info)
    assert pc.adapter_norm() == 0.0 and info["err_start"] == 0.0 and info["err_end"] == 0.0 and info["target_ctx_tokens"] == 0
    pc = _pc()
    info = {}
    Q.v3q_sleep(pc, _st(pc), cs["units"], tid, GEN, steps=40, lr=0.3, top_only=True, info=info)
    assert info["layers_on"] == [1] and float((pc.B_k["0"].abs().sum() + pc.B_v["0"].abs().sum()).detach()) == 0.0
    pc = _pc()
    info = {}
    Q.v3q_sleep(pc, _st(pc), cs["units"], tid, GEN, steps=10, lr=0.3, teacher_kind="sham", seed=3, info=info)
    assert info["teacher"] == "sham" and info["sham_perm"] == [1, 0]
    ctx = Q.ctx_of(cs["units"], tid, "sham", [1, 0])
    assert ctx[0] == tid[1] and ctx[2] == tid[0] and Q.ctx_of(cs["units"], tid, "double")[2] == tid[1]


def test_target_is_taken_only_at_the_answer_rows(fake_cues):
    """The engagement error (and so the rule's target) is the teacher-student gap at the answer rows only, not over the
    whole cue + answer."""
    pc = _pc()
    cs = _cs(pc)
    u = cs["units"][0]
    tid = _told_ids(pc)
    info = {}
    Q.v3q_sleep(pc, _st(pc), [u], tid, GEN, steps=0, lr=0.3, info=info)
    v = V.adopt3(_pc())
    t = V.teacher(v, tid[u["item"]], u["qa_ids"], _st(pc))[1]
    _, act = V.activity(v, u["qa_ids"], _st(pc))
    rows = lambda x: x[u["pos"]]
    want = sum(float((rows(act[i][3]) - rows(t[i][1])).pow(2).sum()) / len(u["pos"]) for i in act)
    full = sum(float((act[i][3] - t[i][1]).pow(2).sum()) / act[i][3].shape[0] for i in act)
    assert abs(info["err_start"] - want) < 1e-6 * max(1.0, want) and abs(want - full) > 1e-6
    assert len(u["pos"]) < len(u["qa_ids"])


def test_distill_q_trains_by_backprop_at_answer_rows(fake_cues):
    pc = _pc(0.5)                                     # adapters start away from the teacher, so the KL can fall
    cs = _cs(pc)
    info = {}
    Q.distill_q(pc, _st(pc), cs["units"], _told_ids(pc), GEN, steps=30, lr=1e-2, info=info)
    assert pc.adapter_norm() > 0 and info["kind"] == "grad_distill" and info["loss_end"] < info["loss_start"]


def test_qa_episodes_weight_each_fact_one(fake_cues):
    eps = Q.qa_episodes(_cs()["units"])
    assert [w for _, w in eps] == [0.5, 0.5, 1.0] and eps[0][0] == "User: what is my cat?\n\nAssistant: It is Pixel.\n\n"


def test_graded_readout_logprob_and_rank():
    pc = _pc(0.3)
    g = Q.graded(pc, _st(pc), TF)
    assert len(g["logprob_items"]) == 3 and all(x <= 0 for x in g["logprob_items"])
    pipe = pc.cortex.pipe
    p, a = pipe.encode("User: what is my cat?\n\nAssistant:"), pipe.encode(" Pixel")
    with torch.no_grad():
        lp = F.log_softmax(pc(p + a, _st(pc))[0], -1)
    want = sum(float(lp[len(p) - 1 + j, a[j]]) for j in range(len(a)))
    assert abs(g["logprob_items"][0] - want) < 1e-5 and abs(g["first_logprob_items"][0] - float(lp[len(p) - 1, a[0]])) < 1e-6
    assert g["rank_items"][0] == 1 + int((lp[len(p) - 1] > lp[len(p) - 1, a[0]]).sum()) and 1 <= g["rank_items"][0] <= 128
    assert g["median_rank"] == sorted(g["rank_items"])[1]


def test_guard():
    with pytest.raises(SystemExit, match="POOL_NODE"):
        Q.guard({})
    Q.guard({"POOL_NODE": "1"})
    Q.guard({"BRAIN_ALLOW_W3090": "1"})


# ------------------------------------------------------------------ dev pick
REF = {"no_sleep": {"recall": 0}, "gradient": {"recall": 4, "generic_kl": 0.04, "heldout_kl_dev": 0.08}}
G0Q = {"pass": True, "n": 12, "n_items": 8}


def _row(lr=0.1, keep=1.0, recall=3, g=0.03, h=0.05, norm=1.0, eh=1.0, et=0.5):
    return {"lr": lr, "keep": keep, "recall": recall, "generic_kl": g, "heldout_kl_dev": h, "adapter_norm": norm,
            "err_start": eh, "err_end": et}


def test_dev_pick_gates_on_g0q_and_delegates():
    assert Q.dev_pick(G0Q, REF, [_row()])["pick"] == {"lr": 0.1, "keep": 1.0}
    for g in ({"pass": True, "n": 12, "n_items": 5}, {"pass": False, "n": 12, "n_items": 8}, None, {"n_items": 8}):
        assert Q.dev_pick(g, REF, [_row()])["verdict"] == "STOP-G0Q", g
    for bad in (dict(recall=1), dict(g=0.051), dict(h=0.101), dict(norm=0.0), dict(et=1.0)):
        assert Q.dev_pick(G0Q, REF, [_row(**bad)])["verdict"] == "NEGATIVE-DEV", bad


# ------------------------------------------------------------------ verdict
LAYERS = list(range(12))
CUES = {"pass": True, "n": 12, "n_items": 6, "items_kept": [0, 1, 2, 3, 4, 5], "eval_cue_items": [0, 3],
        "n_units": 8, "n_excluded": 6, "excluded": {}, "by_source": {"life": 4, "self": 4}, "cue_sha256": "c"}


def _items(n_on):
    return [i < n_on for i in range(12)]


def _arm(recall, g=0.03, h=0.04, false=1, norm=1.0, eng=None, lp=-5.0):
    return {"recall": recall, "false_recall": false, "generic_kl": g, "heldout_kl": h, "adapter_norm": norm,
            "recall_items": _items(int(recall)), "paraphrase": recall, "paraphrase_items": _items(int(recall)),
            "engagement": eng or {},
            "graded": {"mean_logprob": lp, "median_rank": 50.0, "logprob_items": [lp] * 12}}


def _go():
    loc = lambda t, lo, ctx: {"kind": "local", "teacher": t, "steps_run": 240, "layers_on": lo, "target_ctx_tokens": ctx,
                              "err_start": 1.0, "err_end": 0.5, "target_form": "question"}
    lossy = {"steps_run": 240, "loss_start": 2.0, "loss_end": 0.5}
    a = {"no_sleep": _arm(0, 0, 0, 0, 0.0, {"kind": "none", "steps_run": 0}, lp=-9.0),
         "gradient": _arm(4, eng=lossy),
         "v3q": _arm(4, eng=loc("double", LAYERS, 20), lp=-3.0),
         "sham_ctx": _arm(0, eng=loc("sham", LAYERS, 20)),
         "no_target": _arm(0, 0, 0, 0, 0.0, loc("none", LAYERS, 0)),
         "top_only": _arm(2, eng=loc("double", [11], 20)),
         "grad_distill": _arm(4, eng=lossy),
         "gradient_q": _arm(3, eng=lossy)}
    return {arm: {s: copy.deepcopy(r) for s in Q.SEEDS} for arm, r in a.items()}


def _set(recs, arm, seeds=Q.SEEDS, **kw):
    for s in seeds:
        for k, v in kw.items():
            if k.startswith("eng_"):
                recs[arm][s]["engagement"][k[4:]] = v
            else:
                recs[arm][s][k] = v
    return recs


def _per_seed(recs, arm, key, vals):
    for s, v in zip(Q.SEEDS, vals):
        recs[arm][s][key] = v
    return recs


def test_verdict_go_baseline_and_report():
    v = Q.verdict(_go(), 11, CUES)
    assert v["verdict"] == "GO-V3Q" and all(v["clauses"].values()) and v["bar"] == 2 and v["v3q_beats_sham"] == 6
    assert "v3q" in v["means"] and "v3" not in v["means"] and v["means"]["gradient_q"]["recall"] == 3
    r = v["report"]
    assert r["graded"]["v3q"]["mean_logprob_minus_no_sleep"] == 6.0 and r["graded"]["v3q"]["logprob_kept"] == -3.0
    assert r["graded"]["v3q"]["recall_kept"] == 4 / 6 and r["graded"]["v3q"]["recall_excluded"] == 0.0
    assert r["m3_contrast"]["gradient_q_minus_gradient"] == -1 and r["report_engagement"] == []
    assert r["eval_cue_items"] == [0, 3] and r["excluded_items"] == [6, 7, 8, 9, 10, 11]
    assert abs(r["eval_cue_gain_share"] - 0.5) < 1e-9 and r["eval_cue_carried"] is False   # items 0..3 on: 2 eval of 4


@pytest.mark.parametrize("name,edit,want", [
    ("a recall", lambda r: _set(r, "v3q", recall=1), "NEGATIVE"),
    ("b wins", lambda r: _per_seed(_per_seed(r, "v3q", "recall", [2, 2, 5, 5, 5, 5]), "sham_ctx", "recall", [3, 3, 0, 0, 0, 0]), "NEGATIVE"),
    ("b sham mean", lambda r: _set(r, "sham_ctx", recall=1.5), "NEGATIVE"),
    ("c generic", lambda r: _set(r, "v3q", generic_kl=0.05), "NEGATIVE"),
    ("c heldout", lambda r: _set(r, "v3q", heldout_kl=0.06), "NEGATIVE"),
    ("c night", lambda r: _per_seed(_set(r, "gradient", generic_kl=2.0), "v3q", "generic_kl", [1.2, .03, .03, .03, .03, .03]), "NEGATIVE"),
    ("d recall", lambda r: _set(r, "no_target", recall=1), "UNDEFINED"),
    ("d norm", lambda r: _set(r, "no_target", seeds=[42], adapter_norm=0.01), "UNDEFINED"),
    ("e false", lambda r: _set(r, "v3q", false_recall=2.5), "NEGATIVE"),
    ("shallow", lambda r: _set(r, "top_only", recall=3.5), "GO-SHALLOW"),
    ("gradient < 2", lambda r: _set(r, "gradient", recall=1), "UNDEFINED"),
])
def test_each_clause_failing_alone_gives_the_exact_verdict(name, edit, want):
    v = Q.verdict(edit(_go()), 11, CUES)
    assert v["verdict"] == want, (name, v.get("clauses"), v.get("why"))
    if want == "NEGATIVE" and name[0] in "ace":
        failed = [k for k, ok in v["clauses"].items() if not ok]
        assert len(failed) == 1 and failed[0][0] == name[0], failed


@pytest.mark.parametrize("name,edit", [
    ("no_sleep norm", lambda r: _set(r, "no_sleep", seeds=[43], adapter_norm=0.1)),
    ("gradient loss flat", lambda r: _set(r, "gradient", seeds=[44], eng_loss_end=3.0)),
    ("gradient norm 0", lambda r: _set(r, "gradient", seeds=[44], adapter_norm=0.0)),
    ("v3q err flat", lambda r: _set(r, "v3q", seeds=[100], eng_err_end=1.0)),
    ("v3q wrong teacher", lambda r: _set(r, "v3q", seeds=[100], eng_teacher="sham")),
    ("v3q norm 0", lambda r: _set(r, "v3q", seeds=[100], adapter_norm=0.0)),
    ("sham_ctx weak", lambda r: _set(r, "sham_ctx", seeds=[101], adapter_norm=0.4)),
    ("sham_ctx err flat", lambda r: _set(r, "sham_ctx", seeds=[101], eng_err_end=2.0)),
    ("sham_ctx no context", lambda r: _set(r, "sham_ctx", seeds=[101], eng_target_ctx_tokens=0)),
    ("top_only norm 0", lambda r: _set(r, "top_only", seeds=[102], adapter_norm=0.0)),
    ("top_only all layers", lambda r: _set(r, "top_only", seeds=[102], eng_layers_on=LAYERS)),
    ("top_only wrong layer", lambda r: _set(r, "top_only", seeds=[102], eng_layers_on=[5])),
    ("no_target with context", lambda r: _set(r, "no_target", seeds=[42], eng_target_ctx_tokens=20, eng_teacher="double")),
    ("no_target never ran", lambda r: _set(r, "no_target", seeds=[42], eng_steps_run=0)),
])
def test_each_engagement_gate_alone_is_undefined_not_go(name, edit):
    v = Q.verdict(edit(_go()), 11, CUES)
    assert v["verdict"] == "UNDEFINED", (name, v)


@pytest.mark.parametrize("cues", [None, {**CUES, "pass": False}, {**CUES, "n_items": 5}])
def test_a_failed_or_missing_g0q_is_undefined(cues):
    assert Q.verdict(_go(), 11, cues)["verdict"] == "UNDEFINED"


def test_report_arms_never_change_the_word_and_readings_are_question_form():
    r = _set(_set(_go(), "gradient_q", adapter_norm=0.0), "grad_distill", adapter_norm=0.0)
    v = Q.verdict(r, 11, CUES)
    assert v["verdict"] == "GO-V3Q" and len(v["report"]["report_engagement"]) == 6
    assert Q.verdict(_set(_go(), "gradient_q", recall=float("nan")), 11, CUES)["verdict"] == "GO-V3Q"
    neg = lambda: _set(_go(), "v3q", recall=1)
    assert Q.verdict(neg(), 11, CUES)["reading"].startswith("rule-limit: grad_distill passes (a) and (c) on the same question-form")
    t = Q.verdict(_set(neg(), "grad_distill", recall=1), 11, CUES)["reading"]
    assert t.startswith("target-limit") and "M3" in t and "next: question-form" not in t
    assert Q.verdict(_set(neg(), "grad_distill", eng_loss_end=9.0), 11, CUES)["reading"].startswith("undetermined")


def test_eval_cue_carried_flag():
    r = _go()
    for s in Q.SEEDS:                          # v3q recalls only the two eval-cued facts (0, 3) of the kept six
        r["v3q"][s]["recall_items"] = [i in (0, 3) for i in range(12)]
    assert Q.report(r, CUES)["eval_cue_carried"] is True and Q.report(r, CUES)["eval_cue_gain_share"] == 1.0


@pytest.mark.parametrize("arm", Q.GATED_ARMS)
@pytest.mark.parametrize("key", V.GATED)
def test_any_nonfinite_gated_value_is_undefined(arm, key):
    assert Q.verdict(_set(_go(), arm, seeds=[42], **{key: float("nan")}), 11, CUES)["verdict"] == "UNDEFINED"


def test_verdict_needs_exactly_the_six_seeds_in_every_arm():
    for arm in ("top_only", "gradient_q"):
        r = _go()
        del r[arm][102]
        with pytest.raises(ValueError):
            Q.verdict(r, 11, CUES)
    r = _go()
    r["v3q"][7] = r["v3q"][42]
    with pytest.raises(ValueError):
        Q.verdict(r, 11, CUES)


def test_combine_two_scales():
    assert Q.combine("GO-V3Q", "GO-V3Q") == "GO-V3Q" and Q.combine("GO-V3Q", "NEGATIVE") == "GO-V3Q-PARTIAL"
    assert Q.combine("GO-SHALLOW", "NEGATIVE-DEV") == "GO-SHALLOW" and Q.combine("NEGATIVE", "NEGATIVE-DEV") == "NEGATIVE"
    for w in ("STOP-G0Q", "UNDEFINED", "HOLD-DEV", "ERROR-SMOKE", None):
        assert Q.combine("GO-V3Q", w) == "UNDEFINED", w


# ------------------------------------------------------------------ CLI on synthetic units
def _unit(tag, seed, arm, res, eng=None, lr=0.1, keep=1.0, model="m.pth", commit="abc", csha="c"):
    loc = arm in Q.LOCAL_ARMS
    return {"tag": tag, "seed": seed, "arm": arm, "model_id": model, "commit": commit, "prompts_sha256": "p",
            "cue_sha256": csha, "cues": CUES, "lr": lr if loc else None, "keep": keep if loc else None,
            "last_layer": 11, "engagement": eng or {}, "result": res}


def _write_six(d, recs, **kw):
    os.makedirs(d, exist_ok=True)
    for arm, by in recs.items():
        for s, r in by.items():
            r = dict(r)
            eng = r.pop("engagement")
            Q.atomic_write(Q.unit_path(d, "six", s, arm), _unit("six", s, arm, r, eng, **kw))


def _pickfile(tmp_path, lr=0.1, keep=1.0, model="m.pth", commit="abc", csha="c"):
    p = str(tmp_path / "pick.json")
    Q.atomic_write(p, {"verdict": "PICKED", "pick": {"lr": lr, "keep": keep}, "model_id": model, "commit": commit,
                       "cue_sha256": csha})
    return p


def test_score_cli_go_missing_mixed_and_cue_set(tmp_path):
    d = str(tmp_path / "six")
    _write_six(d, _go())
    out = str(tmp_path / "v.json")
    Q.main(["score", "--dir", d, "--tag", "six", "--out", out, "--pick", _pickfile(tmp_path)])
    v = json.load(open(out))
    assert v["verdict"] == "GO-V3Q" and v["instrument"]["cue_sha256"] == "c" and "graded" in v["report"]
    with pytest.raises(SystemExit, match="pick"):
        Q.main(["score", "--dir", d, "--tag", "six", "--pick", _pickfile(tmp_path, csha="other")])
    u = json.load(open(Q.unit_path(d, "six", 43, "v3q")))
    u["cue_sha256"] = "zzz"
    Q.atomic_write(Q.unit_path(d, "six", 43, "v3q"), u)
    with pytest.raises(SystemExit, match="cue_sha256"):
        Q.main(["score", "--dir", d, "--tag", "six"])
    os.remove(Q.unit_path(d, "six", 42, "gradient_q"))
    with pytest.raises(SystemExit, match="missing units"):
        Q.main(["score", "--dir", d, "--tag", "six"])


def _write_dev(d, g0_items=8, skip=None, csha="c"):
    os.makedirs(d, exist_ok=True)
    g = {"pass": g0_items >= Q.G0Q_MIN, "n": 12, "n_items": g0_items, "n_units": g0_items, "n_excluded": 12 - g0_items,
         "by_source": {}}
    Q.atomic_write(Q.unit_path(d, "ref", 7, "g0q"), _unit("ref", 7, "g0q", g, csha=csha))
    Q.atomic_write(Q.unit_path(d, "ref", 7, "no_sleep"), _unit("ref", 7, "no_sleep", REF["no_sleep"]))
    Q.atomic_write(Q.unit_path(d, "ref", 7, "gradient"), _unit("ref", 7, "gradient", REF["gradient"]))
    for lr in Q.DEV_LRS:
        for k in Q.DEV_KEEPS:
            if (lr, k) == skip:
                continue
            tag = f"lr{lr:g}_keep{k:g}"
            r = _row(lr, k, recall=3 if (lr, k) == (0.1, 3.0) else 1)
            eng = {"err_start": r.pop("err_start"), "err_end": r.pop("err_end")}
            Q.atomic_write(Q.unit_path(d, tag, 7, "v3q"), _unit(tag, 7, "v3q", r, eng, lr=lr, keep=k))


def test_pick_cli(tmp_path):
    d, out = str(tmp_path / "dev"), str(tmp_path / "pick.json")
    _write_dev(d)
    Q.main(["pick", "--dir", d, "--out", out])
    p = json.load(open(out))
    assert p["verdict"] == "PICKED" and p["pick"] == {"lr": 0.1, "keep": 3.0} and p["cue_sha256"] == "c"
    d2 = str(tmp_path / "dev2")
    _write_dev(d2, g0_items=4)
    Q.main(["pick", "--dir", d2, "--out", out])
    assert json.load(open(out))["verdict"] == "STOP-G0Q"
    d3 = str(tmp_path / "dev3")
    _write_dev(d3, skip=(0.3, 0.3))
    with pytest.raises(SystemExit, match="9 registered"):
        Q.main(["pick", "--dir", d3])
    d4 = str(tmp_path / "dev4")
    _write_dev(d4)
    os.remove(Q.unit_path(d4, "ref", 7, "g0q"))
    with pytest.raises(SystemExit, match="G0Q"):
        Q.main(["pick", "--dir", d4])
    d5 = str(tmp_path / "dev5")
    _write_dev(d5, csha="other")
    with pytest.raises(SystemExit, match="cue_sha256"):
        Q.main(["pick", "--dir", d5])


# ------------------------------------------------------------------ run_units on a stub cortex
def _run_args(tmp_path, **kw):
    a = dict(seed=42, arm_list=["g0q"] + list(Q.ARMS), out_dir=str(tmp_path / "six"), tag="six", model="/x/m.pth",
             steps=3, lr=0.1, keep=1.0, decay=1e-3, kl_weight=1.0, grad_lr=5e-4, device="cpu", pick=None,
             prompts_sha=Q.PROMPTS_SHA256)
    a.update(kw)
    return SimpleNamespace(**a)


def _kw():
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import GENERIC, evaluate
    from brain.sleep import sleep
    return dict(PROMPTS=PROMPTS, GENERIC=GENERIC, facts=TF, life_cues=LIFE, evaluate=evaluate, sleep=sleep,
                keys_all=[k for _, _, k in TF], PlasticCortex=lambda cx: PlasticCortex(cx, rank=2))


def test_run_units_on_stub_cortex_resume_engagement_and_pick(tmp_path, monkeypatch, fake_cues):
    monkeypatch.setenv("COMMIT", "abc")
    c = _Cortex()
    base = _st(_pc())
    csha = _cs()["cue_sha256"]
    a = _run_args(tmp_path, pick=_pickfile(tmp_path, lr=0.3, keep=3.0, csha=csha))
    done = Q.run_units(a, c, base, **_kw())
    assert done == ["g0q"] + list(Q.ARMS)
    units = {arm: json.load(open(Q.unit_path(a.out_dir, "six", 42, arm))) for arm in Q.ARMS}
    g0 = json.load(open(Q.unit_path(a.out_dir, "six", 42, "g0q")))
    assert g0["result"]["n_items"] == 2 and g0["result"]["cue_sha256"] == csha and g0["cue_sha256"] == csha
    for arm, u in units.items():
        assert {"host", "commit", "device", "precision", "model_id", "prompts_sha256", "cue_sha256", "engagement",
                "last_layer", "cues"} <= set(u), arm
        assert all(k in u["result"] for k in V.GATED + ("heldout_kl_dev", "adapter_norm", "graded")), arm
        assert {"mean_logprob", "median_rank", "rank_items"} <= set(u["result"]["graded"]) and u["engagement"].get("kind")
    assert units["v3q"]["lr"] == 0.3 and units["v3q"]["keep"] == 3.0 and units["gradient"]["lr"] is None
    assert units["no_sleep"]["result"]["adapter_norm"] == 0.0 and units["no_target"]["result"]["adapter_norm"] == 0.0
    assert units["top_only"]["engagement"]["layers_on"] == [1] and units["sham_ctx"]["teacher"] == "sham"
    for arm in ("gradient", "grad_distill", "gradient_q"):
        assert all(math.isfinite(units[arm]["engagement"][k]) for k in ("loss_start", "loss_end")), arm
    for arm in Q.LOCAL_ARMS:
        assert {"err_start", "err_end", "clip_frac_replay", "clip_frac_keep", "target_rows"} <= set(units[arm]["engagement"])
    Q.check_units(list(units.values()), Q.read_pick(a.pick))
    assert Q.run_units(a, c, base, **_kw()) == []                                  # resume
    with pytest.raises(SystemExit, match="model"):
        Q.run_units(_run_args(tmp_path, model="/x/m04.pth", out_dir=str(tmp_path / "o2"), pick=a.pick), c, base, **_kw())
    with pytest.raises(SystemExit, match="cue set"):
        Q.run_units(_run_args(tmp_path, out_dir=str(tmp_path / "o4"), pick=_pickfile(tmp_path, csha="zz")), c, base, **_kw())
    os.remove(Q.unit_path(a.out_dir, "six", 42, "v3q"))                          # a resumed run on a changed cue set
    monkeypatch.setattr(Q, "self_cues", lambda pc, told, st, item: ["what else is there?"])
    with pytest.raises(SystemExit, match="stored G0Q"):
        Q.run_units(_run_args(tmp_path, arm_list=["v3q"], pick=None), c, base, **_kw())
    monkeypatch.setenv("COMMIT", "zzz")
    with pytest.raises(SystemExit, match="commit"):
        Q.run_units(_run_args(tmp_path, out_dir=str(tmp_path / "o3"), pick=a.pick), c, base, **_kw())


def test_failing_g0q_stops_before_any_night(tmp_path, monkeypatch, fake_cues):
    monkeypatch.setattr(Q, "G0Q_MIN", 3)
    monkeypatch.setenv("COMMIT", "abc")
    a = _run_args(tmp_path)
    with pytest.raises(SystemExit, match="G0Q FAILED"):
        Q.run_units(a, _Cortex(), _st(_pc()), **_kw())
    assert sorted(os.listdir(a.out_dir)) == ["six_s42_g0q.json"]
    with pytest.raises(SystemExit, match="G0Q FAILED"):                             # resume does not skip the gate
        Q.run_units(a, _Cortex(), _st(_pc()), **_kw())
    with pytest.raises(SystemExit, match="G0Q FAILED"):                             # nor does a night-only call
        Q.run_units(_run_args(tmp_path, arm_list=["v3q"], out_dir=str(tmp_path / "n")), _Cortex(), _st(_pc()), **_kw())
    assert not os.path.exists(str(tmp_path / "n"))


def test_dev_seed_never_computes_the_verdict_half(tmp_path, monkeypatch, fake_cues):
    monkeypatch.setenv("COMMIT", "abc")
    a = _run_args(tmp_path, seed=7, arm_list=["no_sleep", "v3q"], out_dir=str(tmp_path / "dev"), tag="lr0.1_keep1")
    Q.run_units(a, _Cortex(), _st(_pc()), **_kw())
    for arm in ("no_sleep", "v3q"):
        r = json.load(open(Q.unit_path(a.out_dir, "lr0.1_keep1", 7, arm)))["result"]
        assert r["heldout_kl"] is None and math.isfinite(r["heldout_kl_dev"])


# ------------------------------------------------------------------ engine specs
def test_engine_specs_render_chain_and_are_registered():
    from tools.engine import core
    a, b = core.Spec.find("local_rule_v3q_01b"), core.Spec.find("local_rule_v3q_04b")
    assert (a.status, b.status, a.memory_gb, b.memory_gb) == ("REGISTERED", "REGISTERED", 5, 10) and a.question == b.question
    assert a.model == "rwkv7-g1d-0.1b-20260129-ctx8192.pth" and b.model == "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
    assert "{model}" in a.cmd and "{model}" in b.cmd and a.smoke["params"].startswith("--arms g0q,")
    assert len(a.dev_grid) == 10 and a.dev_grid == b.dev_grid and a.dev_grid["ref"] == "--arms g0q,no_sleep,gradient --tag ref"
    cells = {(float(p.split("--lr ")[1].split()[0]), float(p.split("--keep ")[1].split()[0]))
             for k, p in a.dev_grid.items() if k != "ref"}
    assert cells == {(lr, k) for lr in Q.DEV_LRS for k in Q.DEV_KEEPS}
    c = a.render(7, a.dev_grid["lr0.1_keep1"], "/o.json", commit="c" * 12, job="lr0.1_keep1")
    assert "local_rule_v3q_01b/cccccccccccc/lr0.1_keep1" in c and "--model $BRAIN_MODEL_DIR/rwkv7-g1d-0.1b" in c
    assert "-m brain.experiments.local_rule_v3q run" in c and "tools.engine.bundle $D /o.json --glob '*.json'" in c
    for v in ("GO-V3Q", "GO-SHALLOW", "NEGATIVE", "NEGATIVE-DEV"):
        assert a.next_for(v) == "local_rule_v3q_04b", v
    for v in ("UNDEFINED", "ERROR", "HOLD-DEV", "STOP-G0Q"):
        assert a.next_for(v) is None, v
    assert b.next_for("GO-V3Q") is None
    txt = open("brain/research/2026-11-28-local-rule-v3q-prereg-DRAFT.md").read()
    assert "STATUS: REGISTERED" in txt and "registration review" in txt and "M3" in txt


def test_engine_pick_and_verdict_delegate_to_the_harness(tmp_path):
    from tools.engine.specs import local_rule_v3q as m
    d = tmp_path / "src"
    _write_dev(str(d))
    dev = {"all": {"units": {f: json.load(open(d / f)) for f in os.listdir(d)}}}
    p = m.pick(dev, {"res": str(tmp_path / "res")})
    assert p["params"] == f"--arms {','.join(Q.ARMS)} --lr 0.1 --keep 3.0 --tag six" and p["harness_pick"]["verdict"] == "PICKED"
    d2 = tmp_path / "src2"
    _write_dev(str(d2), g0_items=3)
    p2 = m.pick({"all": {"units": {f: json.load(open(d2 / f)) for f in os.listdir(d2)}}}, {"res": str(tmp_path / "res2")})
    assert "STOP-G0Q" in p2["stop"]
    s = tmp_path / "six"
    _write_six(str(s), _go(), keep=3.0)
    six = {seed: {"units": {f: json.load(open(s / f)) for f in os.listdir(s) if f"_s{seed}_" in f}} for seed in Q.SEEDS}
    v = m.verdict(six, {"harness_pick": dict(p["harness_pick"])}, {"res": str(tmp_path / "res")})
    assert v["verdict"] == "GO-V3Q" and "failed clauses []" in v["summary"] and "graded v3q" in v["summary"]
    del six[42]
    assert m.verdict(six, {"harness_pick": dict(p["harness_pick"])}, {"res": str(tmp_path / "res")})["verdict"] == "ERROR"


def test_paraphrase_transfer_is_reported_and_never_changes_the_word():
    r = _go()
    for s in Q.SEEDS:                          # v3q paraphrase only on the two non-eval kept facts 1, 2
        r["v3q"][s]["paraphrase"] = 2
        r["v3q"][s]["paraphrase_items"] = [i in (1, 2) for i in range(12)]
    v = Q.verdict(r, 11, CUES)
    assert v["verdict"] == "GO-V3Q"
    rep, g = v["report"], v["report"]["graded"]["v3q"]
    assert g["paraphrase"] == 2 and g["paraphrase_minus_no_sleep"] == 2
    assert g["paraphrase_kept"] == 2 / 6 and g["paraphrase_excluded"] == 0.0 and g["paraphrase_kept_minus_no_sleep"] == 2 / 6
    pt = rep["paraphrase_transfer"]
    assert pt["v3q_minus_no_sleep"] == 2 and pt["eval_cued_gain"] == 0.0 and pt["other_kept_gain"] == 0.5
    for s in Q.SEEDS:                          # unreadable paraphrase: report None, word unchanged
        r["v3q"][s].pop("paraphrase_items")
        r["v3q"][s]["paraphrase"] = float("nan")
    v = Q.verdict(r, 11, CUES)
    assert v["verdict"] == "GO-V3Q" and v["report"]["paraphrase_transfer"]["v3q_minus_no_sleep"] is None
    assert v["report"]["paraphrase_transfer"]["eval_cued_gain"] is None
