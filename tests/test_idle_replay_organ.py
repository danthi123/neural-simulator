"""Idle replay (Stage-2 C5 build): the idle turn voices a fact from the brain's own store, chosen by a spiking
reactivation race biased by recency + encoding strength. CPU only, small bridges (a few seconds each), no brain.

Pins: default OFF and byte-identical off (the server only reads the flag); the latch read is a pure transducer; the
encoding record follows the store; on the Stage-2 C5 store shape the newest fact wins and the two exposures differ;
the no-encode and commit lesions silence it; the tag lesion removes the recency bias; the decoded utterance (never the
host label) is voiced, in the frame the Stage-2 scorer reads; a decode with a missing role abstains.
"""
import os
import re

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import idle_replay_organ as IR  # noqa: E402
from research.runners import idle_replay_battery as B  # noqa: E402
from research.runners import spiking_commit as SC  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── flags ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_default_off(monkeypatch):
    monkeypatch.delenv("BRAIN_IDLE_REPLAY", raising=False)
    assert IR._IDLE_REPLAY_DEFAULT_ON is False
    assert IR.idle_replay_enabled() is False
    monkeypatch.setenv("BRAIN_IDLE_REPLAY", "1")
    assert IR.idle_replay_enabled() is True
    monkeypatch.setenv("BRAIN_IDLE_REPLAY", "0")
    assert IR.idle_replay_enabled() is False


def test_no_encode_lesion_is_the_selfinit_lesion(monkeypatch):
    monkeypatch.setenv("BRAIN_SELF_INITIATE_LESION", "1")
    assert IR.encode_lesioned() is True
    monkeypatch.delenv("BRAIN_SELF_INITIATE_LESION")
    assert IR.encode_lesioned() is False


def test_server_off_path_unchanged():
    """Static: with the flag off the idle block calls the self-init organ exactly as before, and the per-turn hook is
    behind the flag read."""
    src = open(os.path.join(_ROOT, "webapp", "server.py"), encoding="utf-8").read()
    assert "_ir_on = _IR.idle_replay_enabled()\n        if _ir_on:\n            _IR.note_turn(" in src
    assert ("            else:\n                siorg = _get_selfinit_organ(cache_key)\n"
            "                si = siorg.speak(lesion=_SI.selfinit_lesioned())\n") in src
    assert "_IR_reset.reset_organ(cache_key)" in src


# ── the read ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_latch_read_is_a_transducer():
    thr = 0.03 * 20 * 30                               # 18 spikes
    assert SC.latch_read([0, 80, 0], 20, 30, 0.03)["winner"] == 1
    assert SC.latch_read([0, 17, 0], 20, 30, 0.03)["winner"] is None
    co = SC.latch_read([80, 80, 0], 20, 30, 0.03)
    assert co["winner"] is None and co["co_commit"] is True and co["threshold"] == thr
    rv = SC.read_with_variants([0, 80, 1], 20, 30, 0.03)
    assert rv["read_invariant"] is True and rv["variants"] == {"x0.5": 1, "x2": 1}
    weak = SC.read_with_variants([0, 30, 0], 20, 30, 0.03)          # above 18, below the x2 level 36
    assert weak["winner"] == 1 and weak["read_invariant"] is False


def test_excitability_monotone_in_recency_and_strength():
    p = dict(IR.REPLAY_GRID[0])
    assert IR.excitability(1, 1.0, p) > IR.excitability(2, 1.0, p) > IR.excitability(4, 1.0, p) > p["BASE"]
    assert IR.excitability(2, 2.0, p) > IR.excitability(2, 1.0, p)


# ── the store ───────────────────────────────────────────────────────────────────────────────────────────────────
class _Comp:
    """A stand-in composer: a kb list and a block read that returns what is stored (or a blank role)."""

    def __init__(self, facts, blank=None):
        self.kb = [({"agent": a, "action": v, "patient": p}, None) for a, v, p in facts]
        self.blank = blank

    def _read_one_block(self, i):
        f = dict(self.kb[i][0])
        if self.blank is not None and i == self.blank:
            f["patient"] = None
        return f


class _Chat:
    def __init__(self, facts, blank=None):
        self.inner = type("I", (), {})()
        self.inner.composer = _Comp(facts, blank)


def _conversation(chain, blank=None):
    """The Stage-2 inner turn order on a stand-in chat: build facts, then one taught fact per turn, then the idle
    turn (note_turn runs at the start of every turn, as in the server)."""
    chat = _Chat(B.BUILD)
    org = IR.IdleReplayOrgan(seed=3, params=dict(IR.REPLAY_GRID[0]))
    for f in chain:
        org.note_turn(chat)                            # turn start
        chat.inner.composer.kb.append(({"agent": f[0], "action": f[1], "patient": f[2]}, None))   # taught this turn
    org.note_turn(chat)                                # the idle turn
    chat.inner.composer.blank = blank
    return org, chat


def test_note_turn_records_encoding_turns():
    org, chat = _conversation(B.CHAINS["intact"])
    assert org.turn == 4
    assert all(org.enc[f]["t"] == 0 for f in B.BUILD)
    assert [org.enc[f]["t"] for f in B.CHAINS["intact"]] == [1, 2, 3]
    ages = {tuple(c["fact"]): c["age"] for c in org.candidates(chat, salience=False)}
    assert ages[B.CHAINS["intact"][-1]] == 1 and ages[B.BUILD[0]] == 4
    chat.inner.composer.kb[0] = ({"agent": "brain", "action": "use", "patient": "neurons"}, None)   # reconsolidated
    org.note_turn(chat)
    assert org.enc[("brain", "use", "neurons")]["t"] == 4


# ── the race ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_speak_voices_newest_and_follows_the_exposure():
    texts = {}
    for name in ("intact", "permuted"):
        org, chat = _conversation(B.CHAINS[name])
        r = org.speak(chat, lesion=False, tag_lesion=False, commit_lesion=False)
        assert r["n_utt"] == 1 and r["idle_replay"]["route"] == "idle_replay"
        assert r["utterance"] == " ".join(B.CHAINS[name][-1])
        assert r["idle_replay"]["decode_matches_winner"] is True
        assert r["idle_replay"]["selection"]["read"]["read_invariant"] is True
        from research.runners import self_initiated_production_organ as SI
        texts[name] = SI.self_initiated_text(r)
    for t in texts.values():                           # the frame the Stage-2 scorer (score_inner) parses
        assert re.search(r"Something's been on my mind\s+[-—]+\s+([a-z]+) ([a-z]+) ([a-z]+)", t)
    assert texts["intact"] != texts["permuted"]


def test_lesions_silence_or_unbias():
    org, chat = _conversation(B.CHAINS["intact"])
    assert org.speak(chat, lesion=True, tag_lesion=False, commit_lesion=False)["n_utt"] == 0
    assert org.speak(chat, lesion=False, tag_lesion=False, commit_lesion=True)["n_utt"] == 0
    r = org.speak(chat, lesion=False, tag_lesion=True, commit_lesion=False)
    xs = r["idle_replay"]["selection"]["x_applied"]
    assert len(set(round(x, 9) for x in xs)) == 1      # every engram at the same excitability


def test_missing_role_abstains():
    org, chat = _conversation(B.CHAINS["intact"], blank=len(B.BUILD) + 2)    # the newest block decodes a blank
    r = org.speak(chat, lesion=False, tag_lesion=False, commit_lesion=False)
    assert r["n_utt"] == 0 and r["utterance"] is None


def test_empty_store_is_silent():
    org = IR.IdleReplayOrgan(seed=1, params=dict(IR.REPLAY_GRID[0]))
    chat = _Chat([])
    org.note_turn(chat)
    assert org.speak(chat, lesion=False, tag_lesion=False, commit_lesion=False)["n_utt"] == 0


# ── the dev battery ─────────────────────────────────────────────────────────────────────────────────────────────
def test_battery_seed_all_pass_and_rescore():
    r = B.run_seed(1, 0)
    assert r["verdict"]["all_pass"], r["verdict"]
    assert B.seed_verdict(r) == r["verdict"]
    assert B.dev_verdict({1: r}, seeds=(1,))["verdict"] == "PASS"
    assert B.dev_verdict({}, seeds=(1,))["verdict"] == "UNDEFINED"


def test_dev_lines():
    lines = B.dev_lines("deadbeef")
    assert len(lines) == len(B.DEV_SEEDS) * (len(B.REPLY_ARMS) + 1) + 2
    assert all("\t" in ln and "revisions/deadbeef" in ln for ln in lines)
    assert sum("BRAIN_IDLE_REPLAY=0" in ln for ln in lines) == len(B.DEV_SEEDS) + 1
    assert sum("BRAIN_IDLE_REPLAY=1" in ln for ln in lines) == len(lines) - len(B.DEV_SEEDS) - 1
    assert not any(" --seed %d " % s in ln for ln in lines for s in IR.GATE_SEEDS)


def test_organ_lines():
    lines = B.organ_lines("deadbeef")
    assert len(lines) == len(B.DEV_SEEDS) and all("--op 0" in ln and "\t" in ln for ln in lines)


def test_gate_seed_refused():
    with pytest.raises(SystemExit):
        B.main(["--seed", "42", "--out", "/nonexistent/x.json"])
