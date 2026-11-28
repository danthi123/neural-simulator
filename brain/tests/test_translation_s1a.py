# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-28-translation-layer-DESIGN.md
# Expected-fail: none (plumbing, exact statistics and the exact verdict rule on synthetic units, a tiny random cortex and a
#   stub life; no model weights)
"""Translation layer S1a: the host guard refuses, residual capture and the write path are exact, vectors are built as
declared, the run resumes per unit and writes atomically with provenance, and every clause of the verdict rule fails alone
with the exact verdict word (HEADROOM / BARRIER / NEITHER / UNDEFINED, -DEV / -UNANCHORED / -FALLBACK-KNOWS)."""
import copy
import json
import math
import os

import pytest
import torch

from brain import source_monitor as SM
from brain.experiments import memory_ceiling as MC
from brain.experiments import translation_s1a as T
from brain.experiments import voice_ceiling_oracle as VC
from brain.experiments import voice_tag as VT
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_source_tag import _plastic
from brain.tests.test_voice_tag import _life  # noqa: F401
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)

TEXT = "User: my cat is called Pixel.\n\n"


# ------------------------------------------------------------------ guard
def test_guard_refuses_without_a_node_flag():
    for env in ({}, {"POOL_NODE": "0"}):
        with pytest.raises(SystemExit):
            T.guard(env)
    T.guard({"POOL_NODE": "1"})
    T.guard({"BRAIN_ALLOW_W3090": "1"})


def test_run_command_is_guarded(monkeypatch):
    monkeypatch.delenv("POOL_NODE", raising=False)
    monkeypatch.delenv("BRAIN_ALLOW_W3090", raising=False)
    with pytest.raises(SystemExit):
        T.main(["run", "--life", "x", "--snaps", "x", "--trunk", "x", "--out-dir", "x", "--model", "x.pth"])


# ------------------------------------------------------------------ residual capture, write path, perplexity
def test_grab_layer_is_every_position_and_a_tag_at_the_layer_shifts_every_position_by_v():
    p = _plastic()
    ids = p.cortex.pipe.encode(TEXT)
    for layer in range(p.L):
        sink, last = [], []
        with torch.no_grad(), T.grab_layer(p.L, layer, sink):
            ref, _ = p(ids)
        with torch.no_grad(), VC.grab_residuals(p.L, last):
            p(ids)
        assert len(sink) == 1 and sink[0].shape == (len(ids), p.C)
        assert torch.equal(sink[0][-1], last[layer])               # same index as the voice ceiling's capture
        with torch.no_grad():
            assert torch.equal(p(ids)[0], ref)                     # the patch is gone and changed nothing
        v = torch.randn(p.C)
        prov = SM.Provenance("all")
        SM.attach(p, v, layer, False, prov)
        s2 = []
        with torch.no_grad(), T.grab_layer(p.L, layer, s2):
            p(ids)
        SM.detach(p)
        assert torch.allclose(s2[0], sink[0] + v, atol=1e-5)       # every position, exactly v
        assert prov.stats["tagged"] == prov.stats["tokens"] == len(ids)


def _fakeb(p, active=True):
    class _C:
        def __init__(self):
            self.pipe, self.active = p.cortex.pipe, active

        def read(self, text, state=None):
            return p(p.cortex.pipe.encode(text), state)

    class _B:
        cortex = _C()
        base = [x.clone() for x in p.zero_state()]

        def _plastic(self):
            return p
    return _B()


def test_telling_residual_is_the_token_mean_and_refuses_a_tagged_or_inactive_brain():
    p = _plastic()
    b = _fakeb(p)
    r = T.telling_residual(b, TEXT, 1)
    sink = []
    with torch.no_grad(), T.grab_layer(p.L, 1, sink):
        p(p.cortex.pipe.encode(TEXT), [x.clone() for x in b.base])
    assert torch.allclose(r, sink[0].mean(0))
    SM.attach(p, torch.ones(p.C), 0, False, SM.Provenance("all"))
    with pytest.raises(RuntimeError):
        T.telling_residual(b, TEXT, 1)
    SM.detach(p)
    with pytest.raises(RuntimeError):
        T.telling_residual(_fakeb(p, active=False), TEXT, 1)


def test_neutral_nll_zero_vector_is_the_baseline_and_a_write_changes_it_and_is_removed():
    p = _plastic()
    b = _fakeb(p)
    ps = ["Rain fell on the old town all night.", "The river ran past the mill."]
    base = T.neutral_nll(b, ps, None, 1)
    assert T.neutral_nll(b, ps, torch.zeros(p.C), 1) == pytest.approx(base, abs=1e-6)
    assert abs(T.neutral_nll(b, ps, torch.randn(p.C) * 5, 1) - base) > 1e-4
    assert p.source_tag is None and p.tag_mask_fn is None
    assert T.neutral_nll(b, ps, None, 1) == pytest.approx(base, abs=1e-7)


def test_offframe_passages_are_sha_guarded():
    ps = T.offframe_passages()
    assert len(ps) == T.PPL_PASSAGES and all(isinstance(x, str) and x for x in ps)


# ------------------------------------------------------------------ vectors
def test_write_vectors_centre_sparsify_project_and_scale():
    torch.manual_seed(0)
    R = {f"i{j}": torch.randn(32) + 10.0 for j in range(4)}
    V = T.write_vectors(R, k=5, m=128)
    for it in R:
        want = R[it] - torch.stack([R[j] for j in R if j != it]).mean(0)
        assert torch.allclose(V["DENSE"][it], want)                      # leave-self-out centring removes the shared +10
        sr = V["SPARSE-RAW"][it]
        assert int((sr != 0).sum()) == 5
        top = torch.topk(want.abs(), 5).indices
        assert torch.equal(sr[top], want[top])
        P = T.proj_matrix(32, 128)
        assert torch.allclose(V["SPARSE-PROJ"][it], P.t() @ T.topk_abs(P @ want, 5))
    assert torch.equal(T.proj_matrix(32, 128), T.proj_matrix(32, 128))  # seeded
    v = T.scaled(V["SPARSE-RAW"]["i0"], 2.0, 3.0)
    assert float(v.norm()) == pytest.approx(6.0, rel=1e-6)
    ns = sorted(float(x.norm()) for x in V["DENSE"].values())
    assert T.nu_of(V) == pytest.approx((ns[1] + ns[2]) / 2)                # median of 4
    with pytest.raises(RuntimeError):
        T.scaled(torch.zeros(3), 1.0, 1.0)


def test_echo_verbatim_and_row_measures():
    assert T.echo_verbatim("My cat is called Pixel!", "My cat is called Pixel.")
    assert not T.echo_verbatim("Your cat is called Pixel.", "My cat is called Pixel.")
    led = {"pixel": {"keys": ["Pixel"], "telling": "My cat is called Pixel."}}
    m = T.row_measures({"item": "pixel", "brain": "Sorry, my cat is called Pixel.", "ok": False}, led)
    assert m["knows"] and m["refusal"] and m["echo_verbatim"] and not m["ok"]


def test_dev_facts_exclude_ledger_keys():
    from brain.experiments.facts_bank import DEV_FACTS36
    allf = T.dev_facts({})
    assert len(allf) == len(DEV_FACTS36)
    k = allf[0]["key"]
    assert all(f["key"] != k for f in T.dev_facts({"x": {"keys": [k.upper()]}}))


# ------------------------------------------------------------------ statistics
def test_clopper_pearson_bounds_are_exact():
    scipy = pytest.importorskip("scipy.stats")
    for k, n in ((0, 5), (3, 10), (10, 10), (24, 48), (7, 9)):
        up = 1.0 if k == n else scipy.beta.ppf(0.95, k + 1, n - k)
        lo = 0.0 if k == 0 else scipy.beta.ppf(0.05, k, n - k + 1)
        assert T.cp_upper(k, n) == pytest.approx(up, abs=1e-9)
        assert T.cp_lower(k, n) == pytest.approx(lo, abs=1e-9)


def test_holm_step_down():
    adj = T.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert T.holm({"x": 0.6, "y": 0.5}) == pytest.approx({"y": 1.0, "x": 1.0})


def test_paired_counts_p_and_bounds():
    x = {(1, f"i{j}"): j < 6 for j in range(10)}
    y = {(1, f"i{j}"): j < 2 or j == 9 for j in range(10)}
    r = T.paired(x, y)
    assert (r["N"], r["b"], r["c"], r["diff"]) == (10, 4, 1, 3)
    assert r["p"] == pytest.approx(MC.sign_p(4, 1))
    assert r["upper"] == pytest.approx(5 * (2 * T.cp_upper(4, 5) - 1))
    assert r["lower"] == pytest.approx(5 * (2 * T.cp_lower(4, 5) - 1))
    assert r["boot_lower"] <= 3 <= r["boot_upper"]
    z = T.paired(x, x)
    assert (z["upper"], z["lower"], z["boot_upper"]) == (0.0, 0.0, 0.0)


def test_pick_gain_rule():
    g = T.GAINS
    ok = {"0.5": 1, "1": 4, "2": 4, "4": 9, "8": 2}
    ppl = {"0.5": 1.0, "1": 1.01, "2": 1.05, "4": 1.06, "8": 1.0}
    assert T.pick_gain(ok, ppl, g) == (1.0, True)                 # 4 inadmissible; tie 1 / 2 -> the smaller
    assert T.pick_gain(ok, {**ppl, "4": 1.05}, g) == (4.0, True)  # 1.05 is admissible (<=)
    assert T.pick_gain(ok, {k: 2.0 for k in ppl}, g) == (0.5, False)


# ------------------------------------------------------------------ synthetic units and the verdict rule
N_ITEMS = 20
BASE_OK = {"OFF": range(8), "C": range(14), "DELTA": range(10), "DELTA-NOTEXT": range(10), "SHAM-DELTA": range(8),
           "ORACLE-RES": range(8), "DELTA-NULL": range(8)}
NU = 2.0


def _rows(name, ok, s, knows=(), refuse=(), g=None, onset_off=False):
    rows = []
    for j in range(N_ITEMS):
        right = j in ok
        br = f"Your thing is k{j}." if right or j in knows else "That is a nice question."
        if j in refuse:
            br = "Sorry, " + br
        r = {"kind": "rotating", "item": f"i{j}", "you": f"Q{j}?", "brain": br, "ok": right, "recalled": None,
             "told_first": name == "C", "onset": 0.0, "label_tagged": None, "vec_norm": None}
        if name == "DELTA-NULL":
            r.update(forced_idx=None, recall_calls=1, reinstated=0, delta_norm=None, recalled_ok=True)
        if name in ("DELTA", "DELTA-NOTEXT", "SHAM-DELTA"):
            r.update(forced_idx=j + (1 if name == "SHAM-DELTA" else 0), recall_calls=1, reinstated=1, delta_norm=3.0,
                     recalled_ok=True, recalled="" if name == "DELTA-NOTEXT" else f"User: My thing {j} is k{j}.")
        if g is not None:
            r.update(label_tagged=True, vec_norm=g * NU, onset=0.0 if onset_off else 1.0)
        rows.append(r)
    return rows


def _units(seeds=T.KR_SEEDS, ok=None, knows=None, refuse=None, ppl_told=None, ppl_dev=None, dev_ok=None, mod=None):
    """Baseline = HEADROOM: DENSE right on 14 of 20 per life (OFF 8), its sham on OFF's 8, sparse arms at OFF."""
    ok, knows, refuse = ok or {}, knows or {}, refuse or {}
    out = {}
    for s in seeds:
        prov = {"seed": s, "host": "h", "commit": "c0", "device": "cpu", "model": "m", "layer": 12, "night": 14,
                "spec": f"sp{s}", "tag": ""}
        U = {}
        led = {f"i{j}": {"keys": [f"k{j}"], "telling": f"My thing {j} is k{j}."} for j in range(N_ITEMS)}
        qs = [f"Q{j}?" for j in range(N_ITEMS)]
        U["META"] = {**prov, "name": "META", "rec": {
            "n_questions": N_ITEMS, "questions": qs, "population": qs, "no_right_episode": [],
            "right_idx": {str(j): j for j in range(N_ITEMS)}, "sham_idx": {str(j): j + 1 for j in range(N_ITEMS)},
            "items": {q: f"i{j}" for j, q in enumerate(qs)}, "nu": NU, "ledger": led}}
        pt = {a: {f"{g:g}": 1.0 for g in T.GAINS} for a in T.WRITE}
        pd = copy.deepcopy(pt)
        for (a, g, ss), v in (ppl_told or {}).items():
            if ss in (None, s):
                pt[a][f"{g:g}"] = v
        for (a, g), v in (ppl_dev or {}).items():
            pd[a][f"{g:g}"] = v
        U["PPL"] = {**prov, "name": "PPL", "rec": {"told": pt, "dev": pd}}
        dv = {("DENSE", 1.0): 5, **(dev_ok or {})}
        for a in T.WRITE:
            for g in T.GAINS:
                U[f"DEV-{T.gname(a, g)}"] = {**prov, "name": "x", "rec": {"ok": dv.get((a, g), 0), "n": 36, "rows": []}}

        def put(name, default, g=None):
            o = ok.get(name, default)
            o = o(s) if callable(o) else o
            rec = {"rows": _rows(name, set(o), s, set(knows.get(name, ())), set(refuse.get(name, ())), g)}
            if g is not None:
                rec["mask"] = {"calls": 20, "tokens": 400, "tagged": 400}
            U[name] = {**prov, "name": name, "rec": rec}
        for n, d in BASE_OK.items():
            put(n, d)
        for a in T.WRITE:
            for g in T.GAINS:
                put(T.gname(a, g), range(14) if a == "DENSE" else range(8), g)
                put(T.gname("SHAM-" + a, g), range(8), g)
        U["OFF"]["rec"]["reproduction"] = {"n": N_ITEMS, "same": N_ITEMS, "frac": 1.0, "onset_same": N_ITEMS,
                                           "onset_frac": 1.0}
        U["DELTA-NOTEXT"]["rec"]["notext"] = True
        U["REPRO"] = {**prov, "name": "REPRO", "rec": {"banked": {"n": 8, "same": 8, "frac": 1.0, "aligned": True},
                                                       "banked_all": {"n": 20, "same": 20, "frac": 1.0, "aligned": True}}}
        out[s] = U
    if mod:
        mod(out)
    return out


def _v(**kw):
    return T.verdict(_units(**kw), T.KR_SEEDS)


def _head_failed(v):
    return {k for k, x in v["headroom_clauses"].items() if not x}


def _bar_failed(v):
    return {k for k, x in v["barrier_clauses"].items() if not x}


BARRIER_OK = {**{T.gname("DENSE", g): range(8) for g in T.GAINS}}


def test_baseline_headroom_exact():
    v = _v()
    assert v["verdict"] == "HEADROOM-DEV" and v["acts"] is True and not _head_failed(v)
    assert v["primary"]["b"] == 18 and v["primary"]["c"] == 0 and v["margin_items"] == 9
    assert {s: v["picks"]["DENSE"][s]["gain"] for s in T.KR_SEEDS} == {319: 1.0, 320: 1.0, 321: 1.0}
    assert v["arms"]["DENSE"]["ok"] == 42 and v["arms"]["OFF"]["ok"] == 24 and v["measure"] == "ok"
    assert v["text_path_rows"] == 0 and v["label"].startswith("DEV")


def test_baseline_barrier_exact():
    v = _v(ok=BARRIER_OK)
    assert v["verdict"] == "BARRIER-DEV" and v["acts"] is True and not _bar_failed(v)
    assert _head_failed(v) >= {H["H1"]}
    assert set(v["admissible_gains"]["DENSE"]) == set(T.GAINS)


def test_six_project_seeds_have_no_dev_label():
    v = T.verdict(_units(seeds=T.PROJECT_SEEDS), T.PROJECT_SEEDS)
    assert v["verdict"] == "HEADROOM" and v["label"].startswith("SIX") and v["margin_items"] == 18


H = {"H1": "H1 DENSE-OFF positive, exact McNemar p < 0.05 and clustered-bootstrap 5th percentile > 0",
     "H2": "H2 DENSE ppl ratio <= 1.05 at the picked gain on every life",
     "H3": "H3 DENSE-SHAM-DENSE positive, Holm p < 0.05 (S1) and clustered-bootstrap 5th percentile > 0",
     "H4": "H4 DENSE not below DELTA-NOTEXT by the margin (exact lower bound > -margin)",
     "H5": "H5 refusals not above DELTA-NOTEXT (exact upper bound < 5% of item-lives)"}


@pytest.mark.parametrize("clause,kw", [
    ("H1", dict(ok={"DENSE@1": range(9), "SHAM-DENSE@1": ()})),                     # +1 per life: p 0.25; sham at 0
    ("H2", dict(ppl_told={("DENSE", 1.0, 320): 1.06})),
    ("H3", dict(ok={"SHAM-DENSE@1": range(14)})),
    ("H4", dict(ok={"DELTA": range(20), "DELTA-NOTEXT": range(20)})),
    ("H5", dict(refuse={"DENSE@1": range(3)})),
])
def test_each_headroom_clause_fails_alone(clause, kw):
    v = _v(**kw)
    assert _head_failed(v) == {H[clause]} and v["verdict"] == "NEITHER-DEV" and v["acts"] is True


def test_h1_and_h5_boundaries():
    # H1: b = 6, c = 0 pooled on 6 distinct items -> exact two-sided p 0.03125 < 0.05 passes; b = 5 -> 0.0625 fails
    extra = {319: [8, 9], 320: [10, 11], 321: [12, 13]}
    six = {"DENSE@1": lambda s: list(range(8)) + extra[s], "SHAM-DENSE@1": ()}
    five = {"DENSE@1": lambda s: list(range(8)) + extra[s][:1 if s == 321 else 2], "SHAM-DENSE@1": ()}
    assert H["H1"] not in _head_failed(_v(ok=six))
    assert H["H1"] in _head_failed(_v(ok=five))
    # the same 6 wins carried by 2 items (each right in all three lives): exact p passes, the clustered bootstrap fails
    two = _v(ok={"DENSE@1": lambda s: list(range(8)) + [8, 9], "SHAM-DENSE@1": ()})
    assert two["primary"]["p"] < 0.05 and two["primary"]["boot_lower"] <= 0 and H["H1"] in _head_failed(two)
    # H5: margin 0.05 x 60 = 3.0 item-lives; with c = 0 the exact upper bound is the extra-refusal count itself
    def refuse(k):
        def mod(u):
            for r in u[319]["DENSE@1"]["rec"]["rows"][:k]:
                r["brain"] = "Sorry, " + r["brain"]
        return mod
    v2, v3 = _v(mod=refuse(2)), _v(mod=refuse(3))
    assert v2["refusal_vs_notext"]["upper"] == pytest.approx(2.0) and H["H5"] not in _head_failed(v2)
    assert v3["refusal_vs_notext"]["upper"] == pytest.approx(3.0) and H["H5"] in _head_failed(v3)


B = {"B1": "B1 upper(X-OFF) < margin (exact and bootstrap)", "B2": "B2 upper(X-SHAM) < margin (exact and bootstrap)",
     "B3": "B3 every write arm has an admissible gain", "B4": "B4 no positive finding: H1 false and none of S1-S5 holds"}


@pytest.mark.parametrize("clause,kw", [
    ("B1", dict(ok={**BARRIER_OK, "SPARSE-RAW@2": range(8, 16), "SHAM-SPARSE-RAW@2": range(8, 16)})),
    ("B2", dict(ok={**BARRIER_OK, "SHAM-SPARSE-RAW@2": ()})),
    ("B3", dict(ok=BARRIER_OK, ppl_told={("SPARSE-PROJ", g, 321): 1.2 for g in T.GAINS})),
])
def test_each_barrier_clause_fails_alone(clause, kw):
    v = _v(**kw)
    assert _bar_failed(v) == {B[clause]} and v["verdict"] == "NEITHER-DEV"


def _point_bootstrap(monkeypatch):
    """Bootstrap bounds collapsed to the point b - c, so only the exact bounds and B4 decide (B4 isolated)."""
    real = T.paired

    def p(x, y, boot=0):
        r = real(x, y, 0)
        r["boot_lower"] = r["boot_upper"] = float(r["diff"])
        return r
    monkeypatch.setattr(T, "paired", p)


EXTRA = {319: [8, 9], 320: [10, 11], 321: [12, 13]}


def test_barrier_needs_h1_false(monkeypatch):
    # DENSE +6 (c = 0) on 6 distinct items at every gain: B1 / B2 pass (exact upper 6 < 9), H1 holds (p 0.031), H3 fails
    # under Holm (6 x 0.031): not HEADROOM, and B4 makes it NEITHER rather than BARRIER
    _point_bootstrap(monkeypatch)
    v = _v(ok={T.gname("DENSE", g): (lambda s: list(range(8)) + EXTRA[s]) for g in T.GAINS})
    assert v["headroom_clauses"][H["H1"]] and not v["headroom_clauses"][H["H3"]]
    assert _bar_failed(v) == {B["B4"]} and v["verdict"] == "NEITHER-DEV"


def test_barrier_needs_no_holding_secondary(monkeypatch):
    # SPARSE-RAW at its picked gain (0.5) +8 (c = 0): S2 and S4 hold under Holm (p 0.0078, adjusted 0.047), every exact
    # upper bound is 8 < 9, H1 is false: B4 alone fails
    _point_bootstrap(monkeypatch)
    ex = {319: [8, 9, 10], 320: [11, 12, 13], 321: [14, 15]}
    v = _v(ok={**BARRIER_OK, "SPARSE-RAW@0.5": lambda s: list(range(8)) + ex[s]})
    assert v["holm"]["S2"]["holds"] and v["holm"]["S4"]["holds"] and not v["headroom_clauses"][H["H1"]]
    assert _bar_failed(v) == {B["B4"]} and v["verdict"] == "NEITHER-DEV"


def test_inadmissible_gain_is_not_judged_by_the_null_rule():
    # the B1 failure of SPARSE-RAW@2 vanishes when gain 2 is inadmissible (ppl > 1.05 on one life)
    v = _v(ok={**BARRIER_OK, "SPARSE-RAW@2": range(8, 16), "SHAM-SPARSE-RAW@2": range(8, 16)},
           ppl_told={("SPARSE-RAW", 2.0, 319): 1.051})
    assert v["verdict"] == "BARRIER-DEV" and 2.0 not in v["admissible_gains"]["SPARSE-RAW"]


def test_equivalence_margin_boundary_is_exact():
    # every cell of SPARSE-RAW@4 one better than OFF on k items of one life (sham equal): upper = k exactly (c = 0, p_U = 1)
    def mk(k):
        return dict(ok={**BARRIER_OK, "SPARSE-RAW@4": lambda s: list(range(8)) + (list(range(8, 8 + k)) if s == 319 else []),
                        "SHAM-SPARSE-RAW@4": lambda s: list(range(8)) + (list(range(8, 8 + k)) if s == 319 else [])})
    v8 = _v(**mk(8))
    cell = v8["barrier_cells"]["SPARSE-RAW@4"]["vs_off"]
    assert cell["upper"] == pytest.approx(8.0)
    # the bootstrap bound decides alone at this size: 8 single-life clusters can resample above 9
    assert (cell["boot_upper"] < 9) == (B["B1"] not in _bar_failed(v8))
    v9 = _v(**mk(9))
    assert v9["barrier_cells"]["SPARSE-RAW@4"]["vs_off"]["upper"] == pytest.approx(9.0)
    assert B["B1"] in _bar_failed(v9)                                       # 9 is not < 9


def test_holm_family_is_fixed_and_adjusted():
    v = _v()
    assert [k for k in v["holm"]] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    ps = {k: x["p"] for k, x in v["holm"].items()}
    assert {k: x["p_holm"] for k, x in v["holm"].items()} == pytest.approx(T.holm(ps))
    assert v["holm"]["S1"]["holds"] and not v["holm"]["S2"]["holds"]


# ------------------------------------------------------------------ gates and modifiers
def test_c_gate_fallback_knows_and_undefined():
    v = _v(ok={"C": range(10)}, knows={"C": range(14)})                     # strict 0.5, knows 0.7
    assert v["verdict"] == "HEADROOM-DEV-FALLBACK-KNOWS" and v["acts"] is False and v["measure"] == "knows"
    v = _v(ok={"C": range(10)})
    assert v["verdict"] == "UNDEFINED" and "C strict" in v["why"]
    assert _v(ok={"C": range(12)})["verdict"] == "HEADROOM-DEV"              # 36/60 = 0.6 is not under 0.6


def _repro_miss(units, self_frac=None):
    units[320]["REPRO"]["rec"]["banked_all"] = {"n": 20, "same": 17, "frac": 0.85, "aligned": True}
    if self_frac is not None:
        units[320]["REPRO2"] = {**units[320]["REPRO"], "name": "REPRO2",
                                "rec": {"self": {"n": 20, "same": round(20 * self_frac), "frac": self_frac}}}


def test_delta_anchor_fallback_route():
    v = _v(mod=lambda u: _repro_miss(u, 1.0))
    assert v["verdict"] == "HEADROOM-DEV-UNANCHORED" and v["acts"] is False and v["unanchored"] == [320]
    assert _v(mod=lambda u: _repro_miss(u))["verdict"] == "UNDEFINED"
    assert _v(mod=lambda u: _repro_miss(u, 0.9))["verdict"] == "UNDEFINED"
    v = _v(mod=lambda u: u[319]["REPRO"]["rec"].update(banked_all={"n": 20, "same": 18, "frac": 0.9}))
    assert v["verdict"] == "HEADROOM-DEV"                                     # 18/20 = 0.9 anchors
    # the gate reads every battery row, not the told rows: a told-only miss (6/8) with 20/20 overall anchors
    v = _v(mod=lambda u: u[319]["REPRO"]["rec"].update(banked={"n": 8, "same": 6, "frac": 0.75}))
    assert v["verdict"] == "HEADROOM-DEV" and v["delta_anchor"][319]["told"]["frac"] == 0.75
    # nothing aligned (frac None): falls to the self-determinism route, never a TypeError
    v = _v(mod=lambda u: u[319]["REPRO"]["rec"].update(banked_all={"n": 0, "same": 0, "frac": None}))
    assert v["verdict"] == "UNDEFINED" and "not self-deterministic" in v["why"]


def _null_differs(seed, k, onset=False):
    def mod(u):
        for r in u[seed]["DELTA-NULL"]["rec"]["rows"][:k]:
            r.update(onset=0.5) if onset else r.update(brain="A different reply.")
    return mod


def test_delta_driver_link_to_off():
    v = _v()
    assert v["delta_driver"][319] == {"n": 20, "same": 20, "frac": 1.0, "onset_same": 20, "onset_frac": 1.0}
    assert _v(mod=_null_differs(320, 1))["verdict"] == "HEADROOM-DEV"               # 19/20 = 0.95 reproduces
    for mod in (_null_differs(320, 2), _null_differs(321, 2, onset=True)):           # 18/20 replies, or onsets: UNDEFINED
        v = _v(mod=mod)
        assert v["verdict"] == "UNDEFINED" and "DELTA driver with nothing forced" in v["why"] and v["acts"] is False


def _set(seed, name, **kw):
    def f(u):
        for r in u[seed][name]["rec"]["rows"][:1]:
            r.update(kw)
    return f


@pytest.mark.parametrize("mod,why", [
    (lambda u: u[319]["OFF"]["rec"]["reproduction"].update(frac=0.85), "OFF does not reproduce"),
    (lambda u: u[319]["OFF"]["rec"].pop("reproduction"), "OFF does not reproduce"),
    (_set(320, "OFF", recalled="User: x"), "OFF: a row recalled"),
    (_set(320, "C", told_first=False), "C: telling not heard first"),
    (_set(321, "DELTA", reinstated=0), "DELTA: forced episode"),
    (_set(321, "DELTA-NOTEXT", recalled_ok=False), "DELTA-NOTEXT: forced episode"),
    (lambda u: u[321]["DELTA-NOTEXT"]["rec"].update(notext=False), "text not withheld"),
    (_set(319, "SHAM-DELTA", forced_idx=None), "SHAM-DELTA: forced episode"),
    (_set(320, "DELTA-NULL", forced_idx=3), "DELTA-NULL: a row was forced"),
    (_set(321, "DELTA-NULL", recall_calls=0), "DELTA-NULL: a row was forced"),
    (lambda u: u[319].pop("DELTA-NULL"), "units missing"),
    (_set(319, "DENSE@2", vec_norm=4.1), "written norm"),
    (lambda u: u[320]["SPARSE-PROJ@8"]["rec"].update(mask={"tokens": 400, "tagged": 300}), "footprint"),
    (lambda u: [r.update(onset=0.0) for r in u[320]["SHAM-SPARSE-RAW@0.5"]["rec"]["rows"][:11]], "no-op"),
    (lambda u: u[319].pop("SHAM-DENSE@4"), "units missing"),
    (lambda u: u[319].pop("REPRO"), "units missing"),
    (lambda u: u[320]["DENSE@1"].update(commit="c1"), "disagree on commit"),
    (lambda u: u.pop(321), "seeds"),
    (lambda u: [u[s]["META"]["rec"].update(population=[]) for s in u], "empty population"),
])
def test_undefined_cases_each_alone(mod, why):
    v = _v(mod=mod)
    assert v["verdict"] == "UNDEFINED" and why in v["why"] and v["acts"] is False


def test_noop_at_exactly_half_is_engaged():
    v = _v(mod=lambda u: [r.update(onset=0.0) for r in u[320]["SHAM-SPARSE-RAW@0.5"]["rec"]["rows"][:10]])
    assert v["verdict"] == "HEADROOM-DEV"


def test_population_excludes_rows_without_a_right_episode():
    def mod(u):
        for s in u:
            u[s]["META"]["rec"]["population"] = [f"Q{j}?" for j in range(1, N_ITEMS)]
            u[s]["META"]["rec"]["no_right_episode"] = ["Q0?"]
            for r in u[s]["DELTA"]["rec"]["rows"][:1]:
                r.update(forced_idx=None, reinstated=0)          # the excluded row is not engaged, and need not be
    v = _v(mod=mod)
    assert v["verdict"] == "HEADROOM-DEV" and v["population"] == {319: 19, 320: 19, 321: 19}


def test_text_path_is_reported_not_gated():
    v = _v(mod=lambda u: u[319]["DELTA-NOTEXT"]["rec"]["rows"][15].update(brain="Something else."))
    assert v["verdict"] == "HEADROOM-DEV" and v["text_path_rows"] == 1


def test_score_cli_reads_unit_files(tmp_path, capsys):
    for s, U in _units().items():
        for n, u in U.items():
            VC._save_json(str(tmp_path / f"u_s{s}_{n}.json"), u)
    VC._save_json(str(tmp_path / "u-smoke_s319_OFF.json"), {"tag": "smoke"})       # a smoke unit is never scored
    out = tmp_path / "v.json"
    v = T.main(["score", "--dir", str(tmp_path), "--out", str(out)])
    assert v["verdict"] == "HEADROOM-DEV" and json.load(open(out))["verdict"] == "HEADROOM-DEV"


def test_run_cli_passes_the_gain_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("POOL_NODE", "1")
    got = {}
    monkeypatch.setattr(T, "run", lambda *a, **k: got.update(units=a[8], gains=a[9], tag=a[10], limit=a[11]))
    life = tmp_path / "l.json"
    life.write_text(json.dumps({"seed": 7}))
    base = ["run", "--life", str(life), "--snaps", "s", "--trunk", "t", "--out-dir", str(tmp_path), "--model", "m.pth"]
    T.main(base + ["--units", "META,PPL", "--gains", "1", "--limit", "3", "--tag", "smoke"], cortex=object())
    assert got == {"gains": (1.0,), "tag": "smoke", "limit": 3, "units": "META,PPL"}
    T.main(base, cortex=object())
    assert got["gains"] == T.GAINS


# ------------------------------------------------------------------ the run on a stub life
def test_run_on_a_stub_life_resumes_per_unit_and_records_engagement(tmp_path, monkeypatch):
    res, snaps, trunk = _life(tmp_path)
    night = 2
    qs = VT.told_questions(res, night)
    if len({q["item"] for q in qs}) < 3:
        pytest.skip("the stub life tells fewer than 3 items")
    monkeypatch.setattr(VT, "_onset_ids", lambda pipe: ([1, 3], 2))
    monkeypatch.setattr(VC, "REQUIRE_ACTIVE", False)
    monkeypatch.setattr(T, "telling_residual",
                        lambda b, text, layer: torch.randn(8, generator=torch.Generator().manual_seed(sum(map(ord, text)))))
    monkeypatch.setattr(T, "neutral_nll", lambda b, ps, vec, layer: 2.0 + (0.0 if vec is None else 0.001 * float(vec.norm())))
    monkeypatch.setattr(VC, "capture_label", lambda b, text, state: torch.randn(2, 8, generator=torch.Generator().manual_seed(
        sum(map(ord, text)) % 997)) + 0.01 * float(sum(float(x.sum()) for x in state)))
    attached = []
    monkeypatch.setattr(VC, "set_vector", lambda b, v, layer, prov: attached.append((layer, round(float(v.norm()), 5), prov.kind)))
    monkeypatch.setattr(VC, "clear_vector", lambda b: None)
    monkeypatch.setattr(VC, "_label_tagged", lambda b, kind, text: True)
    banked = {"rows": [{"you": "a", "brain": "x", "cls": "told"}, {"you": "b", "brain": "y", "cls": "fire"}], "commit": "old"}
    runs = []

    def fake_arm(res_, night_, snap, trunk_, cortex, work, weekend, arm, dec):
        runs.append((night_, arm, work))
        return {"rows": [{"you": "a", "brain": "x2", "cls": "told"}, {"you": "b", "brain": "y", "cls": "fire"}]}
    monkeypatch.setattr(MC, "run_arm", fake_arm)
    mc = tmp_path / "mc"
    mc.mkdir()
    (mc / f"dec_s7_n{night}.json").write_text(json.dumps({"void": False, "rows": []}))
    (mc / f"u_s7_n{night}_ORACLE.json").write_text(json.dumps(banked))
    out, work = str(tmp_path / "s1a"), str(tmp_path / "w")
    gains = (1.0, 2.0)
    done = T.run(res, snaps, trunk, StubCortex(), out, work, str(mc), None, gains=gains, night=night, log=lambda *a: None)
    want = set(T.required_units(gains)) | {"ORACLE-RES", "REPRO2"}
    assert set(done) == want and set(os.listdir(out)) == {f"u_s7_{n}.json" for n in want}
    meta = done["META"]["rec"]
    pop0 = set(meta["population"])
    assert meta["n_questions"] == len(qs) and done["META"]["layer"] == 1
    for u in done.values():
        assert u["host"] and u["commit"] and u["device"] and u["spec"] == done["META"]["spec"]
    assert done["REPRO"]["rec"]["banked"] == {"n": 1, "same": 0, "frac": 0.0, "aligned": True}
    assert done["REPRO"]["rec"]["banked_all"] == {"n": 2, "same": 1, "frac": 0.5, "aligned": True}
    nul = [r for r in done["DELTA-NULL"]["rec"]["rows"] if r["you"] in pop0]
    assert nul and all(r["forced_idx"] is None and r["recall_calls"] == 1 and r["reinstated"] == 0 for r in nul)
    assert done["DELTA"]["rec"]["vs_banked_oracle"]["report_only"] is True
    link = VC.reproduce(nul, [r for r in done["OFF"]["rec"]["rows"] if r["you"] in pop0])   # the driver adds nothing else
    assert link["n"] == len(nul) and link["frac"] == 1.0 and link["onset_frac"] == 1.0
    assert done["REPRO2"]["rec"]["self"]["frac"] == 1.0 and len(runs) == 2 and runs[0][2] != runs[1][2]
    pop = set(meta["population"])
    rows = [r for r in done["DELTA"]["rec"]["rows"] if r["you"] in pop]
    assert rows and all(r["recall_calls"] == 1 and r["reinstated"] == 1 and r["recalled_ok"] for r in rows)
    nt = [r for r in done["DELTA-NOTEXT"]["rec"]["rows"] if r["you"] in pop]
    assert all(r["recalled"] == "" and r["recalled_ok"] for r in nt) and done["DELTA-NOTEXT"]["rec"]["notext"]
    for r in done["SHAM-DELTA"]["rec"]["rows"]:
        i = meta["questions"].index(r["you"])
        if str(i) in meta["sham_idx"]:
            assert r["forced_idx"] == meta["sham_idx"][str(i)] != meta["right_idx"][str(i)]
    nu = meta["nu"]
    norms = {n for _, n, k in attached}
    assert all(k == "all" for _, _, k in attached) and {round(g * nu, 5) for g in gains} <= norms
    assert all(r["ok"] in (True, False) for r in done["DEV-DENSE@1"]["rec"]["rows"]) and done["DEV-DENSE@1"]["rec"]["n"] > 30
    assert set(done["PPL"]["rec"]["told"]["DENSE"]) == {"1", "2"}
    assert not [f for f in os.listdir(out) if f.endswith(".tmp")]
    n0 = len(runs)
    again = T.run(res, snaps, trunk, StubCortex(), out, work, str(mc), None, gains=gains, night=night, log=lambda *a: None)
    assert len(runs) == n0 and {k: v["rec"] for k, v in again.items()} == {k: v["rec"] for k, v in done.items()}
    with pytest.raises(SystemExit):                                     # another spec into the same dir refuses
        T.run(res, snaps, trunk, StubCortex(), out, work, str(mc), None, gains=gains, night=night, limit=3,
              log=lambda *a: None)
    sm = T.run(res, snaps, trunk, StubCortex(), out, work, str(mc), None, units="META,OFF", gains=gains, tag="smoke",
               limit=3, night=night, log=lambda *a: None)
    assert set(sm) == {"META", "OFF"} and os.path.exists(os.path.join(out, "u-smoke_s7_OFF.json"))
    assert set(T.load_units(out)) == {7} and "OFF" in T.load_units(out, "smoke")[7]


def test_delta_battery_reinstates_through_the_brains_own_path(tmp_path, monkeypatch):
    res, snaps, trunk = _life(tmp_path)
    qs = VT.told_questions(res, 2)
    monkeypatch.setattr(VT, "_onset_ids", lambda pipe: ([1, 3], 2))
    monkeypatch.setattr(VC, "REQUIRE_ACTIVE", False)
    seen = []
    from brain.hippocampus import Hippocampus
    real = Hippocampus.reinstate

    def spy(self, cortex, state, ep):
        out = real(self, cortex, state, ep)
        seen.append((ep.text, any(not torch.equal(a, b) for a, b in zip(state, out))))
        return out
    monkeypatch.setattr(Hippocampus, "reinstate", spy)
    forced = {0: 0}
    rec = T.delta_battery(res, snaps, trunk, StubCortex(), str(tmp_path / "w"), qs[:2], "D", forced, False, 2)
    assert rec["rows"][0]["reinstated"] == 1 and rec["rows"][1]["reinstated"] == 0 and rec["rows"][1]["recall_calls"] == 1
    assert rec["rows"][0]["delta_norm"] > 0 and rec["rows"][0]["gain"] > 0
    # the reply path's reinstatement (not only the onset read) went through Hippocampus.reinstate and changed the state
    assert len(seen) == 2 and seen[0][0] == seen[1][0] and all(ch for _, ch in seen)


# ------------------------------------------------------------------ engine spec
def test_engine_spec_renders_and_is_registered():
    from tools.engine import core
    s = core.Spec.find("translation_s1a")
    assert s.status == "REGISTERED" and s.model == "rwkv7-g1l-1.5b-20261031-ctx34304.pth" and "{model}" in s.cmd
    assert s.seeds == [319, 320, 321] and s.pick_fn is not None
    sm = s.smoke["params"]
    for u in ("PPL", "DEV-DENSE@1", "DELTA", "DENSE@1", "SHAM-DENSE@1", "ORACLE-RES", "REPRO"):
        assert u in sm.split("--units ")[1].split()[0].split(",")              # every heavy path runs in the smoke
    assert "--gains 1 " in sm and "--tag smoke" in sm and "--limit 3" in sm
    pk = s.pick_fn({}, {"memory_gb": s.memory_gb})
    assert pk["params"] == "" and all(s.mem("six", x, pk, measured=4) == 13 for x in s.seeds)   # not the smoke's cap
    assert s.placement == {319: ["pool40"], 320: ["pool41"], 321: ["pool42"]}
    c = s.render(320, s.six_params, "/o.json", commit="c" * 12, job="s320")
    assert "translation_s1a/cccccccccccc/s320" in c and "--model $BRAIN_MODEL_DIR/rwkv7-g1l-1.5b" in c
    assert "-m brain.experiments.translation_s1a run" in c and "--mc-ref $KR/mc_s320" in c and "vc_full_s320.json" in c
    assert "--glob 'u_s*.json'" in c
    assert s.prereg in s.requires_local and os.path.exists(s.prereg)
    assert "STATUS: REGISTERED 2026-11-28" in open(s.prereg).read()


def test_engine_verdict_delegates_to_the_harness(tmp_path):
    from tools.engine.specs import translation_s1a as m
    six = {s: {"units": {f"u_s{s}_{n}.json": u for n, u in U.items()}} for s, U in _units().items()}
    v = m.verdict(six, {}, {"res": str(tmp_path / "res")})
    assert v["verdict"] == "HEADROOM-DEV" and v["summary"].startswith("HEADROOM-DEV")
    assert m.verdict({319: six[319]}, {}, {"res": str(tmp_path / "r2")})["verdict"] == "ERROR"
    bad = copy.deepcopy(six)
    bad[320]["units"]["u_s320_DENSE@1.json"]["commit"] = "zz"
    assert m.verdict(bad, {}, {"res": str(tmp_path / "r3")})["verdict"] == "UNDEFINED"
