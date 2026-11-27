# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-27-local-sleep-01b-read.md
# Expected-fail: none (plumbing on a tiny random cortex and synthetic records; no model weights)
"""Local rule v2: the projector blinds the adapters to the keep bank, subspace "none" equals v1 exactly, the sham keeps
the rank, and the registered pick / verdict give the exact label with each clause failing alone."""
import copy
import json
import os

import pytest
import torch

from brain.experiments import local_rule_v2 as V
from brain.experiments.local_sleep import _forward, adopt, local_sleep
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_source_tag import _Cortex

TEXT = "User: my cat is Pixel\n\nAssistant: ok\n\n"
GEN = ["User: hi\n\nAssistant: yo\n\n"]
KEEP = ["hi"]   # shorter than the stub width (C = 8), so the keep span leaves free directions


def _pc():
    torch.manual_seed(1)
    return PlasticCortex(_Cortex(), rank=2)


def _st(pc):
    return [s.float() for s in pc.zero_state()]


def test_basis_orthonormal_energy_and_edge_cases():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(30, 3, generator=g) @ torch.randn(3, 10, generator=g)      # rank 3
    U = V.basis(X, 1.0)
    assert U.shape == (10, 3) and torch.allclose(U.T @ U, torch.eye(3), atol=1e-5)
    assert torch.allclose(X - (X @ U) @ U.T, torch.zeros_like(X), atol=1e-4)
    u1 = V.basis(X, 0.5)
    e = float((X @ u1).pow(2).sum() / X.pow(2).sum())
    assert 1 <= u1.shape[1] <= 3 and e >= 0.5
    assert V.basis(torch.zeros(4, 6), 0.9).shape == (6, 0)
    R = V.random_basis(10, 3, torch.Generator().manual_seed(1))
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)


def test_keep_activity_reads_pretrained_and_restores_B():
    pc = _pc()
    with torch.no_grad():
        for d in (pc.B_k, pc.B_v):
            for v in d.values():
                v.normal_(0, 0.3)
    before = [p.detach().clone() for p in pc.parameters()]
    ids = [pc.cortex.pipe.encode(g) for g in GEN]
    acts = V.keep_activity(pc, ids, _st(pc))
    assert all(torch.equal(b, p) for b, p in zip(before, pc.parameters()))
    fresh = _pc()
    acts0 = V.keep_activity(fresh, ids, _st(fresh))   # B == 0: same pretrained activity
    for i in acts:
        assert torch.allclose(acts[i][0], acts0[i][0], atol=1e-5) and torch.allclose(acts[i][1], acts0[i][1], atol=1e-5)


def test_none_subspace_equals_v1_exactly():
    a, b = _pc(), _pc()
    la = local_sleep(a, _st(a), [(TEXT, 1.0)], GEN, steps=12, lr=0.3, decay=1e-3, seed=3, feedback="transport")
    lb = V.v2_sleep(b, _st(b), [(TEXT, 1.0)], GEN, steps=12, lr=0.3, decay=1e-3, seed=3, feedback="transport",
                    subspace="none")
    assert la == lb
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_keep_projection_blinds_adapters_to_keep_bank():
    pc = _pc()
    info = {}
    st = _st(pc)
    losses = V.v2_sleep(pc, st, [(TEXT, 1.0)], KEEP, steps=40, lr=0.3, seed=3, tau=1.0, subspace="keep", info=info)
    assert info["kind"] == "keep" and all(v[0] >= 0.999 for v in info["keep_energy_inside"].values())
    # the adapter changes the replay text but leaves the keep text exactly as the pretrained cortex
    ids_k, ids_r = pc.cortex.pipe.encode(KEEP[0]), pc.cortex.pipe.encode(TEXT)
    with torch.no_grad():
        on_k, on_r = pc(ids_k, st)[0].clone(), pc(ids_r, st)[0].clone()
        pc.enabled = False
        off_k, off_r = pc(ids_k, st)[0], pc(ids_r, st)[0]
        pc.enabled = True
    assert torch.allclose(on_k, off_k, atol=1e-4)
    assert pc.adapter_norm() > 0 and not torch.allclose(on_r, off_r, atol=1e-4)
    assert losses[-1][1] < 1e-6


def test_noproj_drifts_keep_text_but_keep_projection_does_not():
    def keep_kl(kind):
        pc = _pc()
        st = _st(pc)
        V.v2_sleep(pc, st, [(TEXT, 1.0)], KEEP, steps=40, lr=0.3, seed=3, tau=1.0, subspace=kind)
        with torch.no_grad():
            ids = pc.cortex.pipe.encode(KEEP[0])
            on = torch.log_softmax(pc(ids, st)[0], -1)
            pc.enabled = False
            off = torch.log_softmax(pc(ids, st)[0], -1)
        return float(torch.nn.functional.kl_div(on, off, log_target=True, reduction="batchmean"))
    assert keep_kl("keep") < 1e-6 < keep_kl("none")


def test_random_sham_has_same_ranks_and_other_bases():
    pc = _pc()
    ids = [pc.cortex.pipe.encode(g) for g in GEN]
    Uk, ik = V.build_subspaces(pc, ids, _st(pc), 0.9, "keep", seed=3)
    Ur, ir = V.build_subspaces(pc, ids, _st(pc), 0.9, "random", seed=3)
    assert ik["m_k"] == ir["m_k"] and ik["m_v"] == ir["m_v"]
    assert any(not torch.allclose(Uk[i][0], Ur[i][0]) for i in Uk)
    assert V.build_subspaces(pc, ids, _st(pc), 0.9, "none") == ({}, {"kind": "none"})


def test_guard():
    with pytest.raises(SystemExit):
        V.guard({})
    V.guard({"POOL_NODE": "1"})
    V.guard({"BRAIN_ALLOW_W3090": "1"})


def test_provenance_fields():
    p = V.provenance("cpu", "float32")
    assert {"host", "commit", "device", "precision"} <= set(p) and p["precision"] == "float32"


# ------------------------------------------------------------------ registered rules on synthetic records
REF = {"no_sleep": {"recall": 0}, "gradient": {"recall": 4, "generic_kl": 0.032, "heldout_kl_dev": 0.020}}


def _row(lr, tau, recall, g, h, rank=8):
    return {"lr": lr, "tau": tau, "recall": recall, "generic_kl": g, "heldout_kl_dev": h, "rank_min": rank}


def test_dev_pick_rule():
    rows = [_row(0.3, 0.99, 4, 0.03, 0.02), _row(1.0, 0.99, 5, 0.05, 0.02),     # 2nd over the generic cap
            _row(0.1, 0.9, 4, 0.03, 0.01), _row(0.3, 0.9, 6, 0.03, 0.03)]      # 4th over the held-out cap
    r = V.dev_pick(REF, rows)
    assert r["verdict"] == "PICKED" and r["pick"] == {"lr": 0.1, "tau": 0.9}   # tie at 4 -> lower held-out drift
    r = V.dev_pick(REF, [_row(0.3, 0.99, 4, 0.03, 0.02), _row(0.1, 0.99, 4, 0.03, 0.02)])
    assert r["pick"] == {"lr": 0.1, "tau": 0.99}                                # then lower lr
    assert V.dev_pick(REF, [_row(0.3, 0.99, 1, 0.01, 0.01)])["verdict"] == "NEGATIVE-DEV"   # under the floor
    assert V.dev_pick(REF, [])["verdict"] == "NEGATIVE-DEV"
    # boundary: exactly at the caps, the floor and the rank floor qualifies
    assert V.dev_pick(REF, [_row(0.3, 0.99, 2, 0.04, 0.025, V.RANK_MIN)])["verdict"] == "PICKED"
    assert V.dev_pick(REF, [_row(0.3, 0.99, 2, 0.04, 0.025, V.RANK_MIN - 1)])["verdict"] == "NEGATIVE-DEV"   # rank
    assert V.dev_pick(REF, [_row(0.3, 0.99, float("nan"), 0.01, 0.01)])["verdict"] == "NEGATIVE-DEV"
    assert V.dev_pick(REF, [_row(0.3, 0.99, 3, float("nan"), 0.01)])["verdict"] == "NEGATIVE-DEV"


def test_dev_floor_is_the_verdict_bar():
    ref = copy.deepcopy(REF)
    ref["gradient"]["recall"] = 5                     # bar = max(2.5, 2) = 2.5: a recall-2 config does not qualify
    assert V.dev_pick(ref, [_row(0.3, 0.99, 2, 0.03, 0.02)])["verdict"] == "NEGATIVE-DEV"
    assert V.dev_pick(ref, [_row(0.3, 0.99, 3, 0.03, 0.02)])["floor"] == 2.5
    ref["gradient"]["heldout_kl_dev"] = float("nan")
    assert V.dev_pick(ref, [_row(0.3, 0.99, 3, 0.03, 0.02)])["verdict"] == "UNDEFINED-DEV"


def test_prompts_split_pinned():
    from brain.experiments.mix_night_screen import PROMPTS
    d, h = V.split_prompts(PROMPTS)
    assert len(d) == len(h) == 12 and not set(d) & set(h)
    with pytest.raises(SystemExit, match="sha256"):
        V.split_prompts(PROMPTS[:-1] + ["something else"])


def _arm(recall, g, h, false=1, norm=1.0):
    return {s: {"recall": recall, "generic_kl": g, "heldout_kl": h, "false_recall": false, "adapter_norm": norm}
            for s in V.SEEDS}


def _go():
    return {"no_sleep": _arm(0, 0.0, 0.0, 0, 0.0), "gradient": _arm(4, 0.03, 0.02, 1),
            "v2": _arm(3, 0.03, 0.02, 1), "v2_shuffled": _arm(0, 0.03, 0.02, 0),
            "noproj": _arm(3, 0.10, 0.06, 1), "sham": _arm(3, 0.09, 0.05, 1), "sham_tok": _arm(3, 0.08, 0.04, 1)}


def _set(recs, arm, **kw):
    for s in V.SEEDS:
        recs[arm][s].update(kw)
    return recs


def test_verdict_go_and_each_clause_alone():
    assert V.verdict(_go())["verdict"] == "GO-V2"
    # each core clause failing alone -> NEGATIVE
    assert V.verdict(_set(_go(), "v2", recall=1.9))["verdict"] == "NEGATIVE"                       # (a) bar 2
    r = _go()
    for s in V.SEEDS[:2]:
        r["v2_shuffled"][s]["recall"] = 3                                                      # 4/6 wins, mean 1.0
    out = V.verdict(r)
    assert out["clauses"]["b"] is False and out["verdict"] == "NEGATIVE"                            # (b) seed count
    assert V.verdict(_set(_go(), "v2", generic_kl=0.0376))["verdict"] == "NEGATIVE"                # (c1) cap 0.0375
    assert V.verdict(_set(_go(), "v2", heldout_kl=0.0251))["verdict"] == "NEGATIVE"                # (c2) cap 0.025
    assert V.verdict(_set(_go(), "v2", false_recall=1.6))["verdict"] == "NEGATIVE"                 # (d)


def test_lesion_gates_on_heldout_only():
    # noproj breaks only the GENERIC bar (the text the projector was fitted to): not a lesion effect
    assert V.verdict(_set(_go(), "noproj", generic_kl=0.10, heldout_kl=0.01))["verdict"] == "GO-PLAIN-LOCAL"
    assert V.verdict(_set(_go(), "noproj", generic_kl=0.03, heldout_kl=0.02))["verdict"] == "GO-PLAIN-LOCAL"
    assert V.verdict(_set(_go(), "noproj", generic_kl=0.03, heldout_kl=0.05))["verdict"] == "GO-V2"


def test_lesion_margin_ratio_and_seed_count():
    # v2 at 1.24x gradient, noproj at 1.26x: above the cap but under 1.5 x v2 -> not a lesion effect
    r = _set(_set(_go(), "v2", heldout_kl=0.0248), "noproj", heldout_kl=0.0252)
    assert V.verdict(r)["clauses"]["f"] is False and V.verdict(r)["verdict"] == "GO-PLAIN-LOCAL"
    # exactly 1.5 x v2 and above the cap on every seed: passes
    r = _set(_set(_go(), "v2", heldout_kl=0.02), "noproj", heldout_kl=0.03)
    assert V.verdict(r)["clauses"]["f"] is True
    # mean high from one seed: noproj above v2 on only 4 of 6 seeds -> fails
    r = _go()
    for s in V.SEEDS[:2]:
        r["noproj"][s]["heldout_kl"] = 0.01
    for s in V.SEEDS[2:]:
        r["noproj"][s]["heldout_kl"] = 0.2
    out = V.verdict(r)
    assert out["noproj_heldout_wins"] == 4 and out["clauses"]["f"] is False


def test_sham_rules_on_heldout_and_recall():
    # a sham that passes c1 on generic alone but sits far above v2 on held-out still fails (specificity shown)
    assert V.verdict(_set(_go(), "sham", generic_kl=0.03, heldout_kl=0.05))["verdict"] == "GO-V2"
    # each sham passing alone -> nonspecific, and the verdict names it
    for sh in ("sham", "sham_tok"):
        out = V.verdict(_set(_go(), sh, generic_kl=0.03, heldout_kl=0.02))
        assert out["verdict"] == "GO-NONSPECIFIC" and out["passing_shams"] == [sh]
    # sham drift at 1.4 x v2 (under the ratio) and recall as good: nonspecific
    assert V.verdict(_set(_go(), "sham_tok", heldout_kl=0.028))["verdict"] == "GO-NONSPECIFIC"
    # sham drift as v2's but sham does not learn: fails on recall -> GO-V2
    assert V.verdict(_set(_go(), "sham", heldout_kl=0.02, recall=1))["verdict"] == "GO-V2"
    # sham recall under the bar but v2 beats it on only 4 seeds: not shown
    r = _set(_go(), "sham", heldout_kl=0.02, recall=1.5)
    for s in V.SEEDS[:2]:
        r["sham"][s]["recall"] = 3
    assert V.verdict(r)["verdict"] == "GO-NONSPECIFIC"


def test_verdict_undefined_cases():
    assert V.verdict(_set(_go(), "gradient", recall=1.9))["verdict"] == "UNDEFINED"
    r = _go()
    r["v2"][42]["adapter_norm"] = 0.0
    assert V.verdict(r)["verdict"] == "UNDEFINED"
    r = _go()
    r["v2"][43]["adapter_norm"] = float("nan")
    assert V.verdict(r)["verdict"] == "UNDEFINED"
    assert V.verdict(_set(_go(), "v2_shuffled", recall=1.1))["verdict"] == "UNDEFINED"
    assert V.verdict(_set(_go(), "v2_shuffled", recall=1.0))["verdict"] == "GO-V2"                  # boundary


@pytest.mark.parametrize("arm", V.ARMS)
@pytest.mark.parametrize("key", V.GATED)
def test_any_nonfinite_gated_value_is_undefined(arm, key):
    for bad in (float("nan"), float("inf"), None):
        r = _go()
        r[arm][101][key] = bad
        out = V.verdict(r)
        assert out["verdict"] == "UNDEFINED" and any(arm in w and key in w for w in out["why"])


def test_verdict_needs_exactly_the_six_seeds_in_every_arm():
    r = _go()
    for arm in r:
        del r[arm][102]
    with pytest.raises(ValueError):
        V.verdict(r)
    r = _go()
    del r["sham"][44]
    with pytest.raises(ValueError):
        V.verdict(r)
    r = _go()
    del r["sham_tok"]
    with pytest.raises(ValueError):
        V.verdict(r)


def test_bar_uses_half_gradient_when_it_binds():
    r = _set(_go(), "gradient", recall=8)            # bar = max(4, 2) = 4
    assert V.verdict(_set(r, "v2", recall=3.9))["verdict"] == "NEGATIVE"
    assert V.verdict(_set(copy.deepcopy(r), "v2", recall=4))["verdict"] == "GO-V2"


PSHA = V.PROMPTS_SHA256


def _unit(tag, seed, arm, res, lr=0.3, tau=0.99, model="m.pth", commit="abc"):
    loc = arm in V.LOCAL_ARMS
    return {"tag": tag, "seed": seed, "arm": arm, "result": res, "model_id": model, "commit": commit,
            "prompts_sha256": PSHA, "lr": lr if loc else None, "tau": tau if loc and arm != "noproj" else None}


def _write_six(d, recs, **kw):
    for arm, by in recs.items():
        for s, res in by.items():
            V.atomic_write(V.unit_path(str(d), "six", s, arm), _unit("six", s, arm, res, **kw))


def _pickfile(tmp_path, lr=0.3, tau=0.99, model="m.pth", commit="abc"):
    f = tmp_path / "pick.json"
    V.atomic_write(str(f), {"verdict": "PICKED", "pick": {"lr": lr, "tau": tau}, "model_id": model, "commit": commit})
    return str(f)


def test_score_cli_resume_and_missing(tmp_path):
    d = tmp_path / "six"
    _write_six(d, _go())
    out = tmp_path / "v.json"
    V.main(["score", "--dir", str(d), "--tag", "six", "--out", str(out), "--pick", _pickfile(tmp_path)])
    got = json.loads(out.read_text())
    assert got["verdict"] == "GO-V2" and got["instrument"]["commit"] == "abc"
    os.remove(V.unit_path(str(d), "six", 42, "sham"))
    with pytest.raises(SystemExit):
        V.main(["score", "--dir", str(d), "--tag", "six"])
    assert not list(d.glob("*.tmp"))


def test_score_refuses_mixed_units(tmp_path):
    d = tmp_path / "six"
    _write_six(d, _go())
    u = json.load(open(V.unit_path(str(d), "six", 43, "noproj")))
    V.atomic_write(V.unit_path(str(d), "six", 43, "noproj"), {**u, "lr": 1.0})            # stale lr unit
    with pytest.raises(SystemExit, match="lr"):
        V.main(["score", "--dir", str(d), "--tag", "six"])
    _write_six(d, _go())
    u = json.load(open(V.unit_path(str(d), "six", 43, "gradient")))
    V.atomic_write(V.unit_path(str(d), "six", 43, "gradient"), {**u, "model_id": "other.pth"})
    with pytest.raises(SystemExit, match="model_id"):
        V.main(["score", "--dir", str(d), "--tag", "six"])
    _write_six(d, _go())
    u = json.load(open(V.unit_path(str(d), "six", 100, "sham")))
    V.atomic_write(V.unit_path(str(d), "six", 100, "sham"), {**u, "tau": 0.9})
    with pytest.raises(SystemExit, match="tau"):
        V.main(["score", "--dir", str(d), "--tag", "six"])
    _write_six(d, _go())
    with pytest.raises(SystemExit, match="pick"):                                        # pick of the other scale
        V.main(["score", "--dir", str(d), "--tag", "six", "--pick", _pickfile(tmp_path, model="m04.pth")])
    with pytest.raises(SystemExit, match="pick"):
        V.main(["score", "--dir", str(d), "--tag", "six", "--pick", _pickfile(tmp_path, tau=0.9)])


def _write_dev(d, skip=None, rows=None):
    V.atomic_write(V.unit_path(d, "ref", 7, "no_sleep"), _unit("ref", 7, "no_sleep", {"recall": 0}))
    V.atomic_write(V.unit_path(d, "ref", 7, "gradient"), _unit("ref", 7, "gradient", {**REF["gradient"]}))
    for lr in V.DEV_LRS:
        for tau in V.DEV_TAUS:
            if (lr, tau) == skip:
                continue
            res = (rows or {}).get((lr, tau), {"recall": 1, "generic_kl": 0.03, "heldout_kl_dev": 0.02})
            u = _unit(f"lr{lr}_tau{tau}", 7, "v2", res, lr=lr, tau=tau)
            u["subspace"] = {"rank_min": 8, "m_k": {"0": 8}, "m_v": {"0": 9}}
            V.atomic_write(V.unit_path(d, f"lr{lr}_tau{tau}", 7, "v2"), u)


def test_pick_cli(tmp_path):
    d = str(tmp_path / "dev")
    _write_dev(d, rows={(0.3, 0.99): {"recall": 3, "generic_kl": 0.03, "heldout_kl_dev": 0.02}})
    out = tmp_path / "pick.json"
    V.main(["pick", "--dir", d, "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["pick"] == {"lr": 0.3, "tau": 0.99} and got["model_id"] == "m.pth" and got["commit"] == "abc"


def test_pick_refuses_incomplete_grid(tmp_path):
    d = str(tmp_path / "dev")
    _write_dev(d, skip=(1.0, 0.999))          # one config crashed: no unit
    out = tmp_path / "pick.json"
    with pytest.raises(SystemExit, match="cells"):
        V.main(["pick", "--dir", d, "--out", str(out)])
    assert not out.exists()
    d2 = str(tmp_path / "dev2")
    _write_dev(d2)
    os.remove(V.unit_path(d2, "ref", 7, "gradient"))
    with pytest.raises(SystemExit, match="ref"):
        V.main(["pick", "--dir", d2])


def test_run_refuses_without_guard(monkeypatch, tmp_path):
    monkeypatch.delenv("POOL_NODE", raising=False)
    monkeypatch.delenv("BRAIN_ALLOW_W3090", raising=False)
    with pytest.raises(SystemExit):
        V.main(["run", "--model", "x.pth", "--seed", "7", "--tag", "t", "--out-dir", str(tmp_path)])


def test_heldout_drift_zero_at_birth_positive_after_noproj_night():
    pc = _pc()
    st = _st(pc)
    assert V.heldout_drift(pc, st, ["hi", "ok"]) == 0.0
    V.v2_sleep(pc, st, [(TEXT, 1.0)], KEEP, steps=20, lr=0.3, seed=3, subspace="none")
    assert V.heldout_drift(pc, st, ["my cat"]) > 0 and pc.enabled


def test_night_at_tau_below_one_keeps_A_off_the_subspace():
    pc = _pc()
    st = _st(pc)
    info = {}
    ids = [pc.cortex.pipe.encode(g) for g in GEN]
    U, _ = V.build_subspaces(pc, ids, st, 0.9, "keep", seed=3)
    V.v2_sleep(pc, st, [(TEXT, 1.0)], GEN, steps=15, lr=0.3, seed=3, tau=0.9, subspace="keep", info=info)
    assert info["tau"] == 0.9 and all(v[0] >= 0.9 and v[1] >= 0.9 for v in info["keep_energy_inside"].values())
    for i, (uk, uv) in U.items():
        assert float((uk.T @ pc.A_k[str(i)]).abs().max()) < 1e-5 and float((uv.T @ pc.A_v[str(i)]).abs().max()) < 1e-5
    assert pc.adapter_norm() > 0 and info["step_norms"]["replay"] > 0 and info["step_norms"]["keep"] >= 0
    assert info["rank_min"] == min(min(u[0].shape[1], u[1].shape[1]) for u in U.values())


def test_token_sham_rank_matched_real_activity():
    pc = _pc()
    st = _st(pc)
    ids = [pc.cortex.pipe.encode(g) for g in GEN]
    Uk, ik = V.build_subspaces(pc, ids, st, 0.9, "keep", seed=3)
    Ut, it = V.build_subspaces(pc, ids, st, 0.9, "tokens", seed=3)
    assert ik["m_k"] == it["m_k"] and it["m_k_used"] == ik["m_k"] and it["m_v_used"] == ik["m_v"]
    for i in Uk:
        assert torch.allclose(Ut[i][0].T @ Ut[i][0], torch.eye(Ut[i][0].shape[1]), atol=1e-4)
        assert not torch.allclose(Uk[i][0], Ut[i][0], atol=1e-3)
    tok = V.token_ids_like(ids, 128, 3)
    assert [len(t) for t in tok] == [len(x) for x in ids] and tok == V.token_ids_like(ids, 128, 3)
    # the random sham's subspace is not keep-aligned: it holds less keep energy than the keep subspace
    _, ir = V.build_subspaces(pc, ids, st, 0.9, "random", seed=3)
    assert ir["m_k"] == ik["m_k"]
    assert all(ir["keep_energy_inside"][i][0] < ik["keep_energy_inside"][i][0] for i in ik["keep_energy_inside"])


def test_run_units_on_stub_cortex_resume_and_pick(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import GENERIC, evaluate
    from brain.sleep import sleep
    c = _Cortex()
    base = _st(_pc())
    eps = {"sleep_true": [(TEXT, 1.0)], "sleep_shuffled": [("User: my cat is Bolt\n\nAssistant: ok\n\n", 1.0)]}
    monkeypatch.setenv("COMMIT", "abc")
    pick = _pickfile(tmp_path, lr=0.1, tau=0.9, model="m.pth", commit="abc")
    a = SimpleNamespace(seed=42, arm_list=list(V.ARMS), out_dir=str(tmp_path / "six"), tag="six", model="/x/m.pth",
                        steps=3, lr=0.3, tau=0.99, decay=1e-3, feedback="transport", kl_weight=1.0, grad_lr=5e-4,
                        device="cpu", pick=pick, prompts_sha=V.PROMPTS_SHA256)
    kw = dict(PROMPTS=PROMPTS, GENERIC=GENERIC, build_arm_episodes=lambda s: eps, evaluate=evaluate, sleep=sleep,
              keys_all=["pixel", "bolt"], PlasticCortex=lambda cx: PlasticCortex(cx, rank=2))
    done = V.run_units(a, c, base, **kw)
    assert done == list(V.ARMS)
    units = [json.load(open(V.unit_path(a.out_dir, "six", 42, arm))) for arm in V.ARMS]
    for u in units:
        assert {"host", "commit", "device", "precision", "model_id", "prompts_sha256"} <= set(u)
        r = u["result"]
        assert all(k in r for k in V.GATED + ("heldout_kl_dev", "adapter_norm"))
    by = {u["arm"]: u for u in units}
    assert by["v2"]["lr"] == 0.1 and by["v2"]["tau"] == 0.9 and by["noproj"]["tau"] is None
    assert by["sham_tok"]["subspace"]["kind"] == "tokens" and by["noproj"]["subspace"]["kind"] == "none"
    assert by["no_sleep"]["result"]["adapter_norm"] == 0.0
    V.check_units(units, V.read_pick(pick))
    assert V.run_units(a, c, base, **kw) == []                                     # resume: every unit exists
    a2 = SimpleNamespace(**{**vars(a), "model": "/x/m04.pth", "out_dir": str(tmp_path / "other")})
    with pytest.raises(SystemExit, match="model"):
        V.run_units(a2, c, base, **kw)
    monkeypatch.setenv("COMMIT", "zzz")
    a3 = SimpleNamespace(**{**vars(a), "out_dir": str(tmp_path / "other3")})
    with pytest.raises(SystemExit, match="commit"):
        V.run_units(a3, c, base, **kw)
