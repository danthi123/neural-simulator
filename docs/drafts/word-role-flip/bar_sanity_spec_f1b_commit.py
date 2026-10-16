"""bar-sanity spec: word-role-flip F1 commit row -- REGISTRATION_TEXT.md §2 F1.

THE BAR. `research/runners/lbf_rows/word_role_commit.py` (s2p packet word-role-commit step R) AT THE FLIPPED
DEFAULT (no REQUIRED_ENV override): reply-level load-bearing on 6 of 6 gate seeds -- in every DEFINED cell the
intact reply differs from that row's lesion reply (reply fields per docs/TERMS.md load-bearing) -- plus the
commit row's control requirement unchanged from step R (`word-role-commit-nofact-ctl` abstains in both arms on
6/6). A cell failing engagement (organ route `word_role_commit`, no error, `commit_lesioned` matching the arm)
is UNDEFINED and never counts. At F this trace route matches the packet's engagement exactly
(`_WORD_ROLE_COMMIT_DEFAULT_ON = True` at the branch tip).

THE DECISION IS THE REAL ONE, BELOW THE FILE LAYER. Same machinery as
docs/drafts/spiking-commit/bar_sanity_spec_a1_reply.py (whose OK record proves step R's decision reachable):
synthetic reply arms written to a temporary shard tree and scored by the real score_R -> iw2_score._cell_std /
control_cell -> s2p_score.engagement_state. One piece of glue: b2d_score.judge_dir (shard validity: lb.json,
provenance, hosts) is replaced for the duration of decide() by a reader parsing the same arm files into the
`canon` dict judge_dir returns; shard validity is the pipeline's job. Written here so F1 cites a record of its
own; the parse row's F1a spec is blocked on the parse-packet engagement route (NEEDS [wrc-f1a-engagement]).

BRAINS (per gate seed, the same generators as the step-R spec).
  perfect            intact states the self fact (paraphrases that contain 'spikes'), the commit lesion
                     abstains, the control abstains in both arms; traces engaged at route word_role_commit.
  null_lesion_inert  the commit lesion leaves the reply unchanged (the commit stage is not what answers).
  null_wrong_content intact answers without 'spikes' (a parse that commits the wrong action).
  null_confab_control  the control's intact arm answers: the moat is broken.
  null_unengaged     the lesion arm's trace says the commit lesion was not on (the cut never engaged).
  random             per seed, the cell is perfect or lesion-inert and the control ok or confabulated, 1/2 each.
  info_null_noise    perfect, but on each seed with probability 0.1 the two intact arms disagree (null noise).
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for p in (_ROOT, os.path.join(_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import b2d_score as B  # noqa: E402
import s2p_score as S  # noqa: E402

NAME = "word-role-flip F1 commit row (reply load-bearing 6/6 at the flipped default)"
DESCRIPTION = ("Synthetic reply arms -> the real s2p_score.score_R cell rules (S2, intact content, lesion-kept, "
               "control, engagement route word_role_commit); shard-validity layer (judge_dir) replaced by a "
               "reader of the same files.")
SCORER_FILES = ["tools/s2p_score.py", "tools/iw2_score.py", "tools/b2d_score.py",
                "docs/drafts/word-role-flip/bar_sanity_spec_f1b_commit.py"]

_PACKET = "word-role-commit"
_PIN = "0" * 40
_OP = 0
_TAG = "s2p-R-f1b-bar-sanity"
_SPEC = S.PACKETS[_PACKET]
_TURN = {_SPEC["row"]: "s2wc_use", _SPEC["control"]["row"]: "s2wc_nf"}
_ARM_FILES = {"intact_a": "intact_a_x.json", "intact_b": "intact_b_x.json", "lesion": "lesion_x.json",
              "lesion.rep0": "lesion_x.json.rep0"}
_ANSWERS = ["the brain uses the spikes", "I use spikes.", "spikes: the brain uses them", "I usually use spikes."]
_ABSTAIN = ["I don't know about that. My curiosity is piqued - I haven't learned about usually yet: what can you "
            "tell me about usually?", "I don't know that yet.", "I'm not sure; I haven't learned that."]


def _trace(lesion, engaged=True):
    return {"route": "word_role_commit", "lesioned": False, "commit_lesioned": bool(lesion) if engaged else False,
            "error": None, "commit_op": _OP}


def _say(ans):
    return {"answer": ans, "abstained": False, "recalled_svo": ["brain", "use", "spikes"]}


def _abstain(ans):
    return {"answer": ans, "abstained": True, "recalled_svo": None}


def _cell(kind, rng):
    """Capability cell arms {arm: body} for one seed."""
    say = _say(_ANSWERS[int(rng.integers(len(_ANSWERS)))])
    ab = _abstain(_ABSTAIN[int(rng.integers(len(_ABSTAIN)))])
    if kind == "wrong_content":
        say = dict(say, answer="the brain uses neurons", recalled_svo=["brain", "use", "neurons"])
    lesion = say if kind == "lesion_inert" else ab
    arms = {"intact_a": dict(say), "intact_b": dict(say), "lesion": dict(lesion), "lesion.rep0": dict(lesion)}
    if kind == "null_noise":
        arms["intact_b"] = _say(_ANSWERS[(_ANSWERS.index(say["answer"]) + 1) % len(_ANSWERS)])
    engaged = kind != "unengaged"
    for k, body in arms.items():
        body["word_role_trace"] = _trace(k.startswith("lesion"), engaged)
    return arms


def _control(kind, rng):
    ab = _abstain(_ABSTAIN[int(rng.integers(len(_ABSTAIN)))])
    intact = _say("I usually drink water") if kind == "confab" else ab
    if kind == "confab":
        intact["recalled_svo"] = ["brain", "drink", "water"]
    arms = {"intact_a": dict(intact), "intact_b": dict(intact), "lesion": dict(ab), "lesion.rep0": dict(ab)}
    for k, body in arms.items():
        body["word_role_trace"] = _trace(k.startswith("lesion"))
    return arms


def _brain(cell_kind, ctl_kind):
    def gen(rng):
        out = {}
        for s in S.GATE_SEEDS:
            ck = cell_kind(rng) if callable(cell_kind) else cell_kind
            tk = ctl_kind(rng) if callable(ctl_kind) else ctl_kind
            out[s] = {"cell": _cell(ck, rng), "control": _control(tk, rng)}
        return out
    return gen


def _reader(d, fac, pin, *a, **k):
    turn = _TURN[fac]
    arms = {}
    for arm, fn in _ARM_FILES.items():
        with open(os.path.join(d, fn)) as fh:
            arms[arm] = json.load(fh)
    return {"dir": d, "faculty": fac, "exists": True, "fails": [], "class": None, "kind": "neural-lesion",
            "verdict": None, "load_bearing": None, "turn": turn, "diffs": [], "host": None, "files": {},
            "organ_errors": [], "arms": arms}


def decide(data):
    with tempfile.TemporaryDirectory() as base:
        sel = os.path.join(base, "selection.json")
        with open(sel, "w") as fh:
            json.dump({"packet": _PACKET, "verdict": "SELECTED", "op": _OP}, fh)
        for s, cells in data.items():
            for row, arms in ((_SPEC["row"], cells["cell"]), (_SPEC["control"]["row"], cells["control"])):
                dd = os.path.join(base, _TAG, "s%d" % s, row)
                os.makedirs(dd)
                for arm, fn in _ARM_FILES.items():
                    with open(os.path.join(dd, fn), "w") as fh:
                        json.dump({_TURN[row]: arms[arm]}, fh)
        saved = B.judge_dir
        B.judge_dir = _reader
        try:
            res, _code = S.score_R(_PACKET, _TAG, _PIN, sel, base=base)
        finally:
            B.judge_dir = saved
        return res["verdict"] == "PASS"


perfect = _brain("perfect", "ok")
null_lesion_inert = _brain("lesion_inert", "ok")
null_wrong_content = _brain("wrong_content", "ok")
null_confab_control = _brain("perfect", "confab")
null_unengaged = _brain("unengaged", "ok")
random = _brain(lambda rng: "perfect" if rng.random() < 0.5 else "lesion_inert",
                lambda rng: "ok" if rng.random() < 0.5 else "confab")
info_null_noise = _brain(lambda rng: "null_noise" if rng.random() < 0.1 else "perfect", "ok")
