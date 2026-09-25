"""Pattern completion in the replay routes (webapp/replay_completion.py; default-OFF BRAIN_AWAKE_REPLAY_COMPLETION for
the awake bouts and BRAIN_SLEEP_REPLAY_COMPLETION for the night's epoch; branch research/awake-replay-completion).

Two layers:
  * WIRING + LEDGER DYNAMICS on a fake store (no brain build): the read-back is a steep Hill curve of the expressed
    increment-to-baseline ratio, chosen so the Amendment-4 margin route is SUBCRITICAL on it (fresh read ~0.19 falling
    below 0.06 over the 48 bouts -- the shape the failing gate seed reported: 0.207 -> 0.031), and the completion
    read is a fake that reinstates the ensemble (R_c = 0.786, the measured value for a fully
    reinstated three-role fact) while the expressed ratio is above an ignition ratio. Pinned: flags OFF are
    byte-identical to the pre-branch bout AND to the pre-branch night (store hashes computed with main's modules at
    05eba333f); the completion lesion writes exactly what the margin routes write; the low-margin block collapses on
    the margin route and is held by completion; a VERY low-margin block (a night read below the DA capture point) is
    held awake but lost at night unless the night completes too; the awake-edge lesion still cuts it.
  * THE SUBSTRATE READ on a real (small, D=64) OneBrainComposer: `_role_scores` is the composer's own block read; a
    fully expressed block reinstates its own three items through the spiking competition and the substrate re-bind
    (R_c = the three-role composite's coherence with the stored increment); the bare baseline does not; a silent or
    tied competition reinstates nothing; no code is grown by the read.
  * THE ITEM COMPETITION ITSELF (Amendment 8 addendum 8a: assembly-coded, a discrimination criterion, a majority
    ignition). Each test here is written to FAIL for a named mutation (verified by mutation, recorded in the addendum):
    the assemblies' pooled counts are graded with the drive; silencing the winning assembly INSIDE the bank changes the
    pick (a host argmax over the scores cannot see that); an equal drive and a near-tie reinstate nothing; a reserved
    slot or a word with no code is never reinstated even when it wins decisively; one resolved item does not ignite the
    burst; and no word other than the stored one is ever reinstated -- over an expression sweep of the real composer,
    on a wrong word leading a near-tie, and on the recorded dev-seed-1 bare-baseline read where the matched filter
    itself decodes the wrong word 'brain' past the criterion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import textwrap

import numpy as np
import pytest

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO)

from webapp import awake_replay_capture as A               # noqa: E402
from webapp import replay_completion as C                  # noqa: E402
from webapp import da_tag_capture as T                     # noqa: E402
from webapp import da_tag_capture_chat as W                # noqa: E402
from webapp import sleep_replay_capture as S               # noqa: E402

FLAGS = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_SLEEP_REPLAY_CAPTURE",
         "BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION",
         "BRAIN_SLEEP_DOWNSCALING", "BRAIN_AWAKE_REPLAY_CAPTURE", "BRAIN_AWAKE_REPLAY_CAPTURE_LESION",
         "BRAIN_AWAKE_REPLAY_COMPLETION", "BRAIN_SLEEP_REPLAY_COMPLETION", "BRAIN_REPLAY_COMPLETION_LESION")
STEEP = (0.28, 1.7, 4.0)    # (max read, half ratio, Hill n): a low-margin block whose read falls fast with its trace
VLOW = (0.10, 1.7, 4.0)     # a very-low-margin block: its read at the kept expression is below the night's capture
R_FULL = 2.19
IGNITE_RATIO = 0.5          # the fake's ignition point (the expressed increment-to-baseline ratio)
RC_FULL = 0.786             # measured: a fully reinstated three-role fact on a D=64 composer (the substrate test)
NEUTRAL = [0.5, 0.5, 0.5, 0.5, 0.5]
BOUT_H = 5.0 / 60.0
RC = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1"}
ARC = {"BRAIN_AWAKE_REPLAY_CAPTURE": "1"}
ARCC = {"BRAIN_AWAKE_REPLAY_COMPLETION": "1"}
SLPC = {"BRAIN_SLEEP_REPLAY_COMPLETION": "1"}
LESC = {"BRAIN_REPLAY_COMPLETION_LESION": "1"}
# store hashes of the fake scenarios below, computed with main's webapp/awake_replay_capture.py and
# webapp/sleep_replay_capture.py at 05eba333f (the pre-branch modules), flags as named
PRE_BRANCH_SHA = {"rc_arc_rest_night": "949dceb043ab5936b4dd8e4c935210aba7fd375624625435af434ff571844723",
                  "rc_rest_night": "a5e63512023e45bedca6573deb6edcb7a14c33bff2a28e303d6533386974dfe7",
                  "rc_arc_rest_night_vlow": "e9a9323aad5fe8ca1b0a01be390215454989874d744da3f6b0322187acfdbe5a"}


class FakeD1:
    def __init__(self, *a, **kw):
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        return float(np.clip((float(da) - T._DA_TONIC) / (S.DA_SWR_FULL - T._DA_TONIC), 0.0, 1.0)), None


class LowMarginComposer:
    """Block-major store; its read-back is the STEEP Hill curve of the expressed ratio read off the store synapses."""
    CURVE = STEEP

    def __init__(self, D=64, seed=0):
        self.D, self.store_conns, self.n_reads, self.L = D, [], 0, None
        self._rng = np.random.default_rng(seed)

    def store(self, g):
        u = g * np.exp(2j * np.pi * self._rng.random(self.D))
        trig = 1000 + (len(self.store_conns) // self.D) * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(u[k])) for k in range(self.D)]

    def ratio(self, i):
        b = self.L.blocks[i - self.L.block_offset]
        w = np.array([complex(x[2]) for x in self.store_conns[i * self.D:(i + 1) * self.D]])
        d = b["inc"] / np.maximum(np.abs(b["inc"]), 1e-12)
        return float(abs(np.mean(np.conj(d) * (w - b["base"]))) / np.mean(np.abs(b["base"])))

    def margin(self, i):
        Rm, c, n = self.CURVE
        x = self.ratio(i)
        return Rm * x ** n / (c ** n + x ** n)

    def _block_role_scores(self, i):
        self.n_reads += 1
        m = self.margin(i)
        return {"agent": ("a", 1.0, m, None), "action": ("b", 1.0, m, None), "patient": ("c", 1.0, m, None),
                "polarity": ("pos", 1.0, 1.0, None)}


def fake_completion_read(comp, block_idx, blk, reactivate_fn=None):
    """R = the route's own read (its injected `reactivate_fn`, as the real completion read takes it); R_c = a reinstated
    three-role ensemble while the trace still selects its items."""
    r = (reactivate_fn or S.reactivation_strength)(comp, block_idx)
    rc = RC_FULL if comp.ratio(block_idx) >= IGNITE_RATIO else 0.0
    return {"R": float(r), "R_c": rc, "coherence_abs": rc, "items": {}, "spikes": {}, "n_items": 3 if rc else 0}


class Inner:
    pass


class Chat:
    def __init__(self, comp):
        self.inner = Inner()
        self.inner.composer = comp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.setattr(W, "SpikingD1Activation", FakeD1)
    yield


def _store_hash(comp):
    return hashlib.sha256(json.dumps([(p, q, complex(w).real, complex(w).imag)
                                      for (p, q, w) in comp.store_conns]).encode()).hexdigest()


class VeryLowMarginComposer(LowMarginComposer):
    """A block whose night read at the kept expression (~0.93) is ~0.07 -- below the SWR DA capture point (the miss dev
    seed 2 showed on the brain: night read 0.107, SWR DA 0.579, not captured; the fake's increment is ~2x the brain's,
    so its tag is larger and its read has to be lower for the same miss)."""
    CURVE = VLOW


def _tell(monkeypatch, env=None, fact_turn=2, composer=LowMarginComposer):
    monkeypatch.setattr(C, "completion_read", fake_completion_read)   # the wiring layer's fake substrate read
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE_CLOCK", "turn")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    comp = composer()
    comp.store(1.0)                                   # build-time knowledge: unmanaged (block_offset)
    chat = Chat(comp)
    for i, da in enumerate(NEUTRAL):
        chat._last_da_drives = {"da_level": da}
        W.observe_chat_turn(chat, seed=7)
        if i == 0:
            comp.L = chat._da_tag_capture.ledger
        if i == fact_turn:
            b = comp.L._baseline(1, comp.D)
            comp.store(R_FULL * float(np.mean(np.abs(b))))
        W.after_store_chat(chat)
    return chat, chat._da_tag_capture, comp


def _rest(chat, hours=4.0, period_h=BOUT_H):
    for _ in range(int(round(hours / period_h))):
        W.advance_world_clock_h(period_h)
        W.mark_awake(chat)
        W.tick_chat(chat)


def _night(chat):
    W.advance_world_clock_h(24.0)
    W.tick_chat(chat)


def _kept(cap):
    return cap.ledger.summary()[0]["frac_synapses_z_gt_half"] > 0.9


# ── flag OFF: byte-identical to the pre-branch bout ───────────────────────────────────────────────────────────────────
_PRE_BRANCH_BOUT = textwrap.dedent('''
    def _bout(self, ledger, comp, t):
        ledger.sync_from_store(comp, t)
        ledger.on_store(comp, t)
        ledger.advance(comp, t)
        b_idx = len(self.bouts)
        pre_e = [float(ledger.early_expression(b)) for b in ledger.blocks]
        pre_z = [float(np.mean(b["z"] > 0.5)) for b in ledger.blocks]
        R = []
        for i in range(len(ledger.blocks)):
            with self.rng_ctx(self.seed, _K_AWAKE + b_idx * 1000 + i):
                r = self.reactivate_fn(comp, ledger.block_offset + i)
            R.append(None if r is None else float(r))
        coupling = 0.0 if awake_replay_lesioned() else 1.0
        R_eff = [coupling * (0.0 if r is None else min(1.0, max(0.0, r))) for r in R]
        post_e = []
        for blk, r, e in zip(ledger.blocks, R_eff, pre_e):
            if r > 0.0:
                e_new = e + r * (1.0 - e)
                blk["e_rep"] = float(e_new)
                blk["t_erep"] = float(t)
                h_new = e_new * np.abs(blk["inc"]).astype(np.float64)
                if blk.get("h_rep") is not None:
                    h_new = np.maximum(h_new, blk["h_rep"] * math.exp(-(t - blk["t_rep"]) / TAU_TAG_H))
                blk["h_rep"] = h_new
                blk["t_rep"] = float(t)
            post_e.append(float(ledger.early_expression(blk)))
        ledger._write(comp)
        self.t_last = float(t)
        self.bouts.append({"t_h": float(t), "R": [None if r is None else round(r, 9) for r in R],
                           "R_eff": [round(r, 9) for r in R_eff],
                           "early_before": [round(v, 9) for v in pre_e], "early_after": [round(v, 9) for v in post_e],
                           "pre_frac_z_gt_half": [round(v, 9) for v in pre_z],
                           "p_at_bout": round(float(ledger.p), 12), "n_drive_entries": len(ledger.drive),
                           "lesioned": bool(coupling == 0.0), "no_reader": bool(any(r is None for r in R))})
''')


def _pre_branch_bout():
    ns = {"np": np, "math": math, "_K_AWAKE": A._K_AWAKE, "awake_replay_lesioned": A.awake_replay_lesioned,
          "TAU_TAG_H": A.TAU_TAG_H}
    exec(_PRE_BRANCH_BOUT, ns)
    return ns["_bout"]


def test_flag_off_is_identical_to_the_pre_branch_bout(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat); _night(chat)
    h_new, arc_new = _store_hash(comp), cap._arc.summary()
    assert "completion" not in arc_new and all("completion" not in x for x in arc_new["bouts"])
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.setattr(A.AwakeReplayCapture, "_bout", _pre_branch_bout())    # the method as it was before this branch
    chat2, cap2, comp2 = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat2); _night(chat2)
    assert _store_hash(comp2) == h_new
    assert cap2._arc.summary()["bouts"] == arc_new["bouts"]


def test_completion_flag_without_the_awake_route_is_inert(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARCC})
    _rest(chat); _night(chat)
    h = _store_hash(comp)
    assert not hasattr(cap, "_arc")
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.delenv("BRAIN_AWAKE_REPLAY_COMPLETION")
    chat2, cap2, comp2 = _tell(monkeypatch, env=RC)
    _rest(chat2); _night(chat2)
    assert _store_hash(comp2) == h


def test_completion_lesion_writes_exactly_what_the_margin_routes_write(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat); _night(chat)
    h_arc = _store_hash(comp)
    R_arc = [x["R"] for x in cap._arc.summary()["bouts"]]
    ep_arc = cap._src.summary()["epochs"]
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    chat2, cap2, comp2 = _tell(monkeypatch, env={**RC, **ARC, **ARCC, **SLPC, **LESC})
    _rest(chat2); _night(chat2)
    bo = cap2._arc.summary()["bouts"]
    assert all(x["completion"] is not None and x["completion_lesioned"] for x in bo)        # the reads ran ...
    assert all(x["R_eff"] == [round(min(1.0, max(0.0, x["R"][0])), 9)] for x in bo)        # ... induced with R ...
    ep = cap2._src.summary()["epochs"]
    assert len(ep) == 1 and ep[0]["completion"] is not None and ep[0]["completion_lesioned"]
    assert ep[0]["R_eff"] == ep[0]["R"] and ep[0]["da_swr"] == ep_arc[0]["da_swr"]         # ... the night used R ...
    assert [x["R"] for x in bo] == R_arc and _store_hash(comp2) == h_arc                   # ... = the margin routes


def test_routes_pass_their_injected_read_to_the_completion(monkeypatch):
    """Each route's injected `reactivate_fn` is the partial-cue read the completion runs (not a hard-wired module
    function): with both completion flags armed, a spy injected into the awake route and into the night's route is
    called once per bout and once per epoch. A route that dropped it would fall back to the module read and the spy
    would stay at zero."""
    calls = {"awake": 0, "sleep": 0}

    def spy(tag):
        def f(comp, i):
            calls[tag] += 1
            return S.reactivation_strength(comp, i)
        return f

    class ArcSpy(A.AwakeReplayCapture):
        def __init__(self, seed, reactivate_fn=None, rng_ctx=None):
            super().__init__(seed, reactivate_fn=spy("awake"), rng_ctx=rng_ctx)

    class SrcSpy(S.SleepReplayCapture):
        def __init__(self, seed, d1, reactivate_fn=None, rng_ctx=None):
            super().__init__(seed, d1, reactivate_fn=spy("sleep"), rng_ctx=rng_ctx)

    monkeypatch.setattr(A, "AwakeReplayCapture", ArcSpy)
    monkeypatch.setattr(S, "SleepReplayCapture", SrcSpy)
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, **ARCC, **SLPC})
    _rest(chat, hours=1.0)
    _night(chat)
    assert cap._arc.summary()["n_bouts"] == 12 and len(cap._src.summary()["epochs"]) == 1
    assert calls == {"awake": 12, "sleep": 1}


def _run_hash(monkeypatch, env, composer=LowMarginComposer, rest=True):
    """The long-delay telling, then 4 h awake WITH quiet rest (rest=True: the arc family's datr) or WITHOUT an idle
    tick (rest=False: datl), then the night. Returns (store hash, capture state)."""
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    chat, cap, comp = _tell(monkeypatch, env=env, composer=composer)
    if rest:
        _rest(chat)
    else:
        W.advance_world_clock_h(4.0)
        W.mark_awake(chat)
    _night(chat)
    return _store_hash(comp), cap


@pytest.mark.parametrize("key,env,composer", [
    ("rc_arc_rest_night", {**RC, **ARC}, LowMarginComposer),
    ("rc_rest_night", RC, LowMarginComposer),
    ("rc_arc_rest_night_vlow", {**RC, **ARC}, VeryLowMarginComposer)])
def test_flags_off_reproduce_the_pre_branch_store_hashes(monkeypatch, key, env, composer):
    """Both completion flags unset: the awake bouts AND the night epoch run their pre-branch code paths exactly (the
    hashes were computed with main's modules at 05eba333f, see PRE_BRANCH_SHA)."""
    h, cap = _run_hash(monkeypatch, env, composer)
    assert h == PRE_BRANCH_SHA[key]
    assert all("completion" not in e for e in cap._src.summary()["epochs"])
    if hasattr(cap, "_arc"):
        assert all("completion" not in x for x in cap._arc.summary()["bouts"])


def test_night_completion_flag_alone_leaves_the_awake_route_on_its_margin(monkeypatch):
    h_off, cap_off = _run_hash(monkeypatch, {**RC, **ARC})
    h_on, cap_on = _run_hash(monkeypatch, {**RC, **ARC, **SLPC})
    assert [x["R_eff"] for x in cap_on._arc.summary()["bouts"]] == [x["R_eff"] for x in cap_off._arc.summary()["bouts"]]
    assert all("completion" not in x for x in cap_on._arc.summary()["bouts"])
    assert cap_on._src.summary()["epochs"][0]["completion"] is not None


def test_a_very_low_margin_fact_kept_awake_is_lost_at_night_unless_the_night_completes(monkeypatch):
    """Dev seed 2's shape: the awake completion keeps the trace expressed, but the night's margin read (and so its re-tag
    and SWR DA) is below the capture point; with the night's completion the reinstated ensemble sets the tag and the DA."""
    _h, cap = _run_hash(monkeypatch, {**RC, **ARC, **ARCC}, VeryLowMarginComposer)
    bo, ep = cap._arc.summary()["bouts"], cap._src.summary()["epochs"]
    assert bo[-1]["early_after"][0] > 0.9                                   # kept through the rest ...
    assert ep[0]["R"][0] < 0.13 and ep[0]["da_swr"] < T.prp_threshold()     # ... but the night reads it weakly ...
    assert not _kept(cap)                                                   # ... and does not capture it
    _h, cap2 = _run_hash(monkeypatch, {**RC, **ARC, **ARCC, **SLPC}, VeryLowMarginComposer)
    ep2 = cap2._src.summary()["epochs"]
    assert ep2[0]["R_eff"] == [RC_FULL] and ep2[0]["da_swr"] > T.prp_threshold()
    assert _kept(cap2)


def test_night_completion_does_not_capture_without_rest(monkeypatch):
    _h, cap = _run_hash(monkeypatch, {**RC, **ARC, **ARCC, **SLPC}, rest=False)
    ep = cap._src.summary()["epochs"]
    assert cap._arc.summary()["n_bouts"] == 0                               # 4 h awake, no idle tick: no bout
    assert len(ep) == 1 and ep[0]["completion"][0]["R_c"] == 0.0 and not _kept(cap)


# ── flag ON: a low-margin block is subcritical on the margin route and held by completion ─────────────────────────────
def test_low_margin_block_collapses_on_the_margin_route(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat)
    bo = cap._arc.summary()["bouts"]
    assert 0.15 < bo[0]["R"][0] < 0.25                                   # a low fresh read (the failing seed: 0.207)
    assert bo[-1]["R"][0] < 0.5 * bo[0]["R"][0]                          # ... that the rest loop lets collapse
    assert bo[-1]["early_after"][0] < bo[0]["early_after"][0] - 0.3
    _night(chat)
    assert not _kept(cap)


def test_completion_holds_the_low_margin_block_and_the_night_captures_it(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, **ARCC})
    _rest(chat)
    bo = cap._arc.summary()["bouts"]
    assert len(bo) == 48 and all(x["R_eff"] == [RC_FULL] for x in bo)    # every bout induced with the reinstatement
    assert all(x["completion"][0]["R_c"] == RC_FULL for x in bo)
    assert bo[-1]["early_after"][0] > 0.9 and bo[-1]["R"][0] > 0.9 * bo[0]["R"][0]
    _night(chat)
    ep = cap._src.summary()["epochs"]
    assert len(ep) == 1 and ep[0]["R"][0] > 0.15                        # the night reads the fact as a fresh one
    assert _kept(cap)


def test_awake_edge_lesion_still_cuts_the_completion(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, **ARCC, "BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1"})
    _rest(chat)
    bo = cap._arc.summary()["bouts"]
    assert all(x["R_eff"] == [0.0] and x["early_after"] == x["early_before"] for x in bo)
    assert all(x["completion"][0]["R_c"] >= 0.0 for x in bo)            # the completion read still ran
    _night(chat)
    assert not _kept(cap)


def test_completion_supplies_no_prp(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, **ARCC, "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"})
    n_drive = len(cap.ledger.drive)
    _rest(chat)
    bo = cap._arc.summary()["bouts"]
    assert len(cap.ledger.drive) == n_drive and all(b["n_drive_entries"] == n_drive for b in bo)
    assert cap.ledger.summary()[0]["frac_synapses_z_gt_half"] == 0.0
    _night(chat)
    assert not _kept(cap)


def test_a_trace_below_ignition_is_not_regrown(monkeypatch):
    """Late rest on a faint trace: below the ignition ratio the fake reinstates nothing and the margin read of a faint
    trace is ~0, so the trace keeps decaying -- completion adds no floor of its own."""
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, **ARCC})
    W.advance_world_clock_h(3.0)
    W.mark_awake(chat)
    _rest(chat, hours=1.0)
    bo = cap._arc.summary()["bouts"]
    assert len(bo) == 12 and all(x["completion"][0]["R_c"] == 0.0 for x in bo)
    assert bo[-1]["early_after"][0] < 0.2


# ── the substrate read on a real (small) OneBrainComposer ───────────────────────────────────────────────────────────
VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home", "ball", "chase"]
FACTS = [("dog", "go", "north"), ("bird", "look", "south"), ("cat", "chase", "ball")]


@pytest.fixture(scope="module")
def small():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.one_brain_composer import OneBrainComposer
    try:
        c = OneBrainComposer(seed=7, D=64, vocab=VOCAB, k_max=8, enable_rf_cudagraph=False)
    except (FileNotFoundError, KeyError) as e:          # pragma: no cover
        pytest.skip("composer unavailable: %s" % e)
    for f in FACTS:
        c.store(*f, polarity="AFFIRM")
    i, D = 2, c.D
    inc = np.array([complex(w) for (_p, _q, w) in c.store_conns[i * D:(i + 1) * D]])
    pq = [(p, q) for (p, q, _w) in c.store_conns[i * D:(i + 1) * D]]
    rng = np.random.default_rng([7, 7919, i])
    base = (1.0 / math.sqrt(2.0)) * (rng.standard_normal(D) + 1j * rng.standard_normal(D))

    def express(e):
        w = base + float(e) * inc
        c.store_conns[i * D:(i + 1) * D] = [(p, q, complex(w[k])) for k, (p, q) in enumerate(pq)]
        c._store_dirty, c._store_csr, c._persistent_dirty = True, None, True
        if getattr(c, "_csr_cache", None) is not None:
            c._csr_cache = {}
    return c, i, inc, express


def test_role_scores_are_the_composers_own_block_read(small):
    c, i, inc, express = small
    express(1.0)
    own = c._block_role_scores(i)
    sc = C._role_scores(c, i)
    for role in ("agent", "action", "patient", "polarity"):
        s, vocab = sc[role]
        assert vocab[int(np.argmax(s))] == own[role][0]
        assert abs(c._margin(s) - own[role][2]) < 1e-12


def test_full_expression_reinstates_the_fact_and_the_baseline_does_not(small):
    """Both reads run inside `_private_rng` -- the SAME seeded-substream discipline every production route uses
    (module docstring: 'every D1 read runs inside `_private_rng(seed, k)`'; `read_blocks` wraps each block's
    `completion_read` in `rng_ctx(seed, k_base + i)`). The composer's main substrate carries real background
    current noise (`ou_std_current_pA` = 20, `research/runners/one_brain_composer.py`); an UNSEEDED read (as a bare
    call from a test would be) samples a fresh, uncontrolled draw of it every time and is not what any route ever
    reads. `_izh_bank`'s concept-bank competition itself is noise-free (`ou_std_current_pA` = 0, addendum 8a)."""
    c, i, inc, express = small
    n_concepts = len(c.comp.concepts)
    express(1.0)
    with W._private_rng(7, 1):
        full = C.completion_read(c, i, {"inc": inc})
    assert full["items"] == {"agent": "cat", "action": "chase", "patient": "ball"} and full["n_items"] == 3
    assert full["ignited"] is True
    z = np.asarray(c._compose_phases(["cat", "chase", "ball"], ["agent", "action", "patient"]))
    d = inc / np.abs(inc)
    assert abs(full["R_c"] - float(np.mean(np.conj(d) * z).real)) < 1e-9 and full["R_c"] > 0.7
    assert full["R_c"] > full["R"]                                        # the burst is not the decode margin
    express(0.0)
    with W._private_rng(7, 2):
        bare = C.completion_read(c, i, {"inc": inc})
    assert bare["R_c"] == 0.0 and bare["n_items"] == 0                   # nothing of the fact left to complete ...
    assert bare["ignited"] is False and all(v is None for v in bare["items"].values())   # ... and nothing reinstated
    assert len(c.comp.concepts) == n_concepts                            # the read grew no code


# ── the item competition itself (addendum 8a) ────────────────────────────────────────────────────────────────────
def _peaked(V, pairs, floor=0.0):
    s = np.full(V, float(floor))
    for j, v in pairs:
        s[j] = float(v)
    return s


def test_assembly_counts_are_graded_with_the_drive(small):
    """The bank's operating point is graded: each assembly's pooled count follows its drive (the ratio of counts tracks
    the ratio of drives within 0.1), undriven assemblies stay silent, and the counts are large (hundreds of spikes, not
    single digits). With one cell per item the same drive ratios were overturned by single-cell excitability."""
    c, i, inc, express = small
    V = len(c.words)
    s = _peaked(V, [(0, 1.0), (1, 0.8), (2, 0.6), (3, 0.4)])
    n = C.assembly_counts(c.comp, s)
    assert n.shape == (V,) and np.all(n[4:] == 0.0)
    assert n[0] > n[1] > n[2] > n[3] > 0.0 and n[0] >= 100.0
    for j, r in ((1, 0.8), (2, 0.6), (3, 0.4)):
        assert abs(n[j] / n[0] - r) < 0.1


def test_the_bank_makes_the_pick_not_the_scores(small, monkeypatch):
    """Silence the leading item's assembly INSIDE the bank (a hyperpolarizing current on its cells at every step): the
    pick moves to the runner-up. A host argmax over the scores (or any read of the score vector) cannot see the
    lesion and would still return the leading item."""
    c, i, inc, express = small
    V = len(c.words)
    s = _peaked(V, [(0, 1.0), (1, 0.6)], floor=0.05)
    assert C.spiking_pick(c.comp, s)[0] == 0
    bank = C._assembly_bank(c.comp, V)
    step = bank._run_one_simulation_step
    cells = np.arange(0, C.ASSEMBLY_CELLS)

    def lesioned_step(*a, **kw):
        bank.cp_external_input_current[cells] = -500.0
        return step(*a, **kw)

    monkeypatch.setattr(bank, "_run_one_simulation_step", lesioned_step)
    j, (top, run) = C.spiking_pick(c.comp, s)
    assert j == 1 and top > 0.0
    assert C.assembly_counts(c.comp, s)[0] == 0.0


def test_an_unresolved_competition_reinstates_nothing(small, monkeypatch):
    c, i, inc, express = small
    V = len(c.words)
    assert C.spiking_pick(c.comp, np.zeros(V))[0] is None                  # silent -> no winner
    # an equal drive to every assembly does not let the bank's most excitable assembly win (it did with one cell per
    # item: the first build reinstated SOME item on an uninformative read)
    j, (top, run) = C.spiking_pick(c.comp, np.ones(V))
    assert j is None and top > 0.0 and (top - run) / top < C.DISCRIMINATION_G
    # a near-tie (5% apart in drive) is inside the competition's resolution: nothing reinstated
    assert C.spiking_pick(c.comp, _peaked(V, [(3, 1.0), (7, 0.95)], floor=0.05))[0] is None
    # ... while a clear lead is reinstated
    assert C.spiking_pick(c.comp, _peaked(V, [(3, 1.0), (7, 0.6)], floor=0.05))[0] == 3
    express(1.0)
    monkeypatch.setattr(C, "spiking_pick", lambda inner, s: (None, (0.0, 0.0)))
    r = C.completion_read(c, i, {"inc": inc})
    assert r["R_c"] == 0.0 and r["n_items"] == 0 and r["R"] > 0.0       # the partial read alone is not a burst


def _fake_scores(c, spec):
    """{role: (scores, vocab)} for the three content roles: spec[role] = (vocab list, {word: score}), the rest 0.05."""
    out = {}
    for role, (vocab, top) in spec.items():
        s = np.full(len(vocab), 0.05)
        for w, v in top.items():
            s[vocab.index(w)] = float(v)
        out[role] = (s, list(vocab))
    return out


def test_reserved_slots_and_codeless_words_are_never_reinstated(small, monkeypatch):
    """A reserved (unrecruited) cleanup slot, or a word the composer has no code for, is never reinstated even when its
    assembly wins decisively; no code is grown. Since IGNITION_MIN_ITEMS requires EVERY content role to resolve
    (unanimity, not a majority -- see test_one_resolved_item_does_not_ignite_the_burst), a reserved/codeless pick in
    ONE role blocks the whole burst even though the other two are individually decisive: measured (the
    2-of-3-majority design let a genuinely wrong item through when the third role's own read was this decisive but
    invalid, research/findings/raw/_awake_replay_completion_dev/scan_assembly64/; a role that cannot be trusted taints
    the whole read, not just its own slot)."""
    c, i, inc, express = small
    n_concepts = len(c.comp.concepts)
    vocab = list(c.words) + ["__free0__", "zzqx"]            # a reserved slot and a codeless word
    assert "zzqx" not in c.comp.concepts
    for intruder in ("__free0__", "zzqx"):
        spec = {"agent": (vocab, {intruder: 1.0, "cat": 0.3}), "action": (vocab, {"chase": 1.0}),
                "patient": (vocab, {"ball": 1.0})}
        monkeypatch.setattr(C, "_role_scores", lambda comp, b, _sp=spec: _fake_scores(comp, _sp))
        r = C.completion_read(c, i, {"inc": inc})
        assert r["resolved"]["agent"] is None and r["resolved"]["action"] == "chase"
        assert r["ignited"] is False and r["n_items"] == 0 and r["R_c"] == 0.0
        assert all(v is None for v in r["items"].values())
        assert len(c.comp.concepts) == n_concepts


def test_one_resolved_item_does_not_ignite_the_burst(small, monkeypatch):
    """The burst needs EVERY content role to resolve (`IGNITION_MIN_ITEMS` = 3, unanimity, not a majority). One
    resolved item -- the only thing a bare baseline's crosstalk can produce on the recorded dev reads -- reinstates
    nothing; nor do two: a 2-of-3-majority design was tried first and withdrawn (addendum 8a) after it let a
    genuinely wrong item through (a pre-existing 3-fact vocabulary, dev seed 7's own composer: at e = 0.3 two roles
    resolved, one of them decisively to a WRONG word borrowed from a different stored fact -- see
    test_a_wrong_word_is_never_reinstated). All three resolving is what ignites."""
    c, i, inc, express = small
    V = list(c.words)
    one = {"agent": (V, {"cat": 1.0}), "action": (V, {"chase": 1.0, "go": 0.97}), "patient": (V, {})}
    monkeypatch.setattr(C, "_role_scores", lambda comp, b: _fake_scores(comp, one))
    r = C.completion_read(c, i, {"inc": inc})
    assert r["resolved"] == {"agent": "cat", "action": None, "patient": None}
    assert r["ignited"] is False and r["n_items"] == 0 and r["R_c"] == 0.0
    assert all(v is None for v in r["items"].values())
    two = {"agent": (V, {"cat": 1.0}), "action": (V, {"chase": 1.0}), "patient": (V, {})}
    monkeypatch.setattr(C, "_role_scores", lambda comp, b: _fake_scores(comp, two))
    r = C.completion_read(c, i, {"inc": inc})
    assert r["resolved"] == {"agent": "cat", "action": "chase", "patient": None}
    assert r["ignited"] is False and r["n_items"] == 0 and r["R_c"] == 0.0     # two of three still does not ignite
    three = {"agent": (V, {"cat": 1.0}), "action": (V, {"chase": 1.0}), "patient": (V, {"ball": 1.0})}
    monkeypatch.setattr(C, "_role_scores", lambda comp, b: _fake_scores(comp, three))
    r = C.completion_read(c, i, {"inc": inc})
    assert r["ignited"] is True and r["items"] == {"agent": "cat", "action": "chase", "patient": "ball"}
    assert r["n_items"] == 3 and r["R_c"] > 0.0


_S1_SCAN = os.path.join(_REPO, "research/findings/raw/_awake_replay_completion_dev/scan_assembly64/s1_arcc_scan.json")


def test_a_wrong_word_is_never_reinstated(small, monkeypatch):
    """Never a false memory. (a) Over a 21-point expression sweep of the real composer's stored fact every reinstated
    item is the stored word or nothing. (b) A wrong word leading the stored one by 5% is a near-tie: nothing is
    reinstated in that role. (c) The recorded dev-seed-1 reads at e <= 0.03 (a bare baseline), where the matched filter
    itself decodes the wrong action word 'brain' past the criterion: the burst does not ignite and nothing is
    reinstated. (a)'s reads run inside `_private_rng` -- the seeded-substream discipline every production route uses
    (see test_full_expression_..._does_not); an unseeded read samples the main substrate's real background noise
    (`ou_std_current_pA` = 20) fresh and uncontrolled on every call, which no route ever does."""
    c, i, inc, express = small
    true = {"agent": "cat", "action": "chase", "patient": "ball"}
    for n, e in enumerate(np.linspace(1.0, 0.0, 21)):
        express(float(e))
        with W._private_rng(7, 100 + n):
            r = C.completion_read(c, i, {"inc": inc})
        for role, w in r["items"].items():
            assert w is None or w == true[role], (e, role, w)
    V = list(c.words)
    tie = {"agent": (V, {"dog": 1.0, "cat": 0.95}), "action": (V, {"chase": 1.0}), "patient": (V, {"ball": 1.0})}
    monkeypatch.setattr(C, "_role_scores", lambda comp, b: _fake_scores(comp, tie))
    r = C.completion_read(c, i, {"inc": inc})
    assert r["resolved"]["agent"] is None and r["resolved"]["action"] == "chase"
    assert r["ignited"] is False and all(v is None for v in r["items"].values())  # agent's near-tie taints the burst
    if not os.path.exists(_S1_SCAN):                                          # pragma: no cover
        pytest.skip("dev scan artifact missing: %s" % _S1_SCAN)
    from research.runners.rf_phasor_composer import RFPhasorComposer
    d = json.load(open(_S1_SCAN))
    inner = RFPhasorComposer(seed=int(d["inner_seed"]), D=64, vocab=list(d["vocab"]))
    rows = [x for x in d["curve"] if x["e"] <= 0.03]
    assert len(rows) == 4
    for x in rows:
        sc = {role: (np.asarray(x["scores"][role]), list(d["vocab"])) for role in C.COMPLETION_ROLES}
        s = np.asarray(x["scores"]["action"])
        top2 = np.sort(s)[::-1][:2]
        assert d["vocab"][int(np.argmax(s))] == "brain" and (top2[0] - top2[1]) / top2[0] >= C.DISCRIMINATION_G
        sel = C.select_items(inner, sc)
        assert sel["ignited"] is False and all(v is None for v in sel["items"].values())


def test_polarity_is_not_reinstated():
    assert C.COMPLETION_ROLES == S.FACT_ROLES
