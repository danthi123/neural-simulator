"""bar-sanity spec: word-role-flip F4 (the lesions leak nothing outside the parse path) -- REGISTRATION_TEXT.md §2 F4.

THE BAR. The nine listed statement-form turns (exact strings fixed 2026-10-14, packet amendment A), which sit in
the iw1 and iw2 scripts: both runners run three arms each at F -- intact-at-default, PARSE lesion
(BRAIN_WORD_ROLE_LESION=1), COMMIT lesion (BRAIN_WORD_ROLE_COMMIT_LESION=1) -- and every listed reply
(`answer` / `abstained` / `recalled_svo`) must be IDENTICAL to intact-at-default under BOTH lesions, on 6 of 6
seeds. One changed listed reply fails the bar (the lesion leaks outside the parse path). PRECONDITION
(amendment E): the intact-at-default arm persists each listed turn's parse route as `wr_route`
(engaged / declined / not-reached, from the per-turn `word_role_trace`; readout research/runners/wr_route.py); the
`for` field is NOT accepted as this readout. A listed turn whose route reads `engaged` on any seed makes F4
UNDEFINED plus a NEEDS line, never a silent drop. (Gate stage: 6 seeds; this spec drives the SAME decision
function on synthetic records over whatever seed set the data carries.)

THE DECISION IS THE REAL ONE. The registered identity `--compare` of each runner owns the preconditions and the
turn-level diff lists; decide() reads those outputs (never reimplements them): every precondition must hold
else UNDEFINED = not passing; then, per lesion arm, NO listed turn may appear in `reply_differing_turns`, and on
the intact arm every listed turn must be present and carry `wr_route` != "engaged". Non-listed turns changing
under a lesion is EXPECTED (the lesion is real) and never fails F4 -- a perfect brain exercises exactly that.

BRAINS (per runner: intact + parse-lesion + commit-lesion records over a script holding the 9 listed turns plus
probe turns the lesions legitimately change).
  perfect             listed replies identical across all three arms, wr_route declined/not-reached, probes
                      changed under both lesions (the cut is real), every precondition green.
  perfect_unlisted    same, but the parse lesion leaves one listed turn's NON-listed sibling changed only --
                      scope check: F4 reads listed turns.
  null_leak           the parse lesion changes one listed turn's reply (the leak F4 exists to catch).
  null_engaged_route  intact's one listed turn reports wr_route="engaged" -> UNDEFINED (not passing) + NEEDS.
  null_precondition   one arm's turn status is 500 -> compare's own precondition fails -> UNDEFINED.
  null_missing_listed the script lost one listed turn -> not present -> not passing.
  random              a random responder: replies redrawn independently per arm (~0 listed identity across
                      both lesions and both runners).
  info_unlisted_noise a non-listed turn's volatile key flips (canon-level, reply equal): not a listed reply.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import iw1_default_identity as IW1  # noqa: E402
from research.runners import iw2_default_identity as IW2  # noqa: E402

NAME = "word-role-flip F4 (nine listed turns reply-identical under both lesions, wr_route never engaged)"
DESCRIPTION = ("Synthetic intact/parse-lesion/commit-lesion arm records per runner -> each runner's own "
               "compare() preconditions and diff lists, filtered to the nine registered statement-form turns "
               "plus the amendment-E wr_route precondition.")
SCORER_FILES = ["research/runners/iw1_default_identity.py", "research/runners/iw2_default_identity.py",
                "docs/drafts/word-role-flip/bar_sanity_spec_f4_offtarget.py"]

_LISTED = ("the cat eats fish", "the cat precedes the ball", "the ball precedes the worm",
           "Sally puts the marble in the basket.", "Sally leaves the room.",
           "Anne moves the marble to the box.", "the wolf hunts the deer",
           "the dog and the cat walked in", "the cat and the dog walked in")
_PROBES = ("what do you usually use?", "what do you usually drink?", "hi")   # lesions legitimately change these
_SEED = 7
_ARMS = ("intact", "parse_lesion", "commit_lesion")


def _turn(msg, listed, wr_route="declined", reply=("I am the brain.", False, ["brain", "use", "spikes"])):
    t = {"message": msg, "status": 200, "sha_canon": "canon-%s" % msg[:24],
         "answer": reply[0], "abstained": reply[1], "recalled_svo": reply[2]}
    if listed:
        t["wr_route"] = wr_route
    return t


def _rec(arm, rng, *, listed_change=None, engaged=False, status_drop=None, missing_listed=None,
         probe_change=True, volatile=False):
    turns = []
    for m in list(_LISTED) + list(_PROBES):
        if m == missing_listed:
            continue
        listed = m in _LISTED
        reply = ("I am the brain.", False, ["brain", "use", "spikes"])
        route = "declined" if m != "what do you usually drink?" else "not-reached"
        if engaged and m == listed_change:
            route = "engaged"
        if arm != "intact" and probe_change and m in _PROBES:
            reply = ("I abstain under this lesion." if arm == "parse_lesion" else "blocked by the cut.", True, None)
        if arm != "intact" and m == listed_change:
            reply = ("the lesion reworded the listed statement.", False, ["the", "list", "changed"])
        if status_drop == m:
            pass                                       # status applied after construction
        t = _turn(m, listed, route, reply)
        if status_drop == m:
            t["status"] = 500
        if volatile and not listed:
            t["wall_s"] = float(rng.uniform(1.0, 9.0))
        turns.append(t)
    return {"arm": arm, "seed": _SEED, "n_turns": len(turns), "turns": turns,
            "corpus_present": True, "ltm_bundle": "shipped-ltm-v3", "sim_backend": "numpy",
            "git_sha": "f" * 40, "script_sha": "script-f4-%d" % len(turns),
            "threads": dict(IW2.DETERMINISM_ENV)}


def _arms(rng, **kw):
    intact = _rec("intact", rng)
    out = {r: [] for r in _ARMS}
    for runner in ("iw1", "iw2"):
        out["intact"].append(intact if runner == "iw1" else dict(intact))
        out["parse_lesion"].append(_rec("parse_lesion", rng, **kw))
        out["commit_lesion"].append(_rec("commit_lesion", rng, **kw))
    return out


def decide(data):
    """The registered F4 decision, read from each runner's own compare() output."""
    ok = True
    for ri, (runner, arms) in enumerate(data.items()):     # arms: {"intact": [iw1, iw2], "parse_lesion": [...], ...}
        mod = IW1 if runner == "iw1" else IW2
        intact = arms["intact"][ri]
        listed_idx = [i for i, t in enumerate(intact["turns"]) if t.get("message") in _LISTED]
        if len(listed_idx) != len(_LISTED):           # a listed turn missing from the script: UNDEFINED
            ok = False
            continue
        for i in listed_idx:
            if intact["turns"][i].get("wr_route") == "engaged":
                ok = False                            # amendment-E precondition: UNDEFINED + NEEDS, never silent
        for arm in ("parse_lesion", "commit_lesion"):
            v = mod.compare(intact, arms[arm][ri], "same")   # the real precondition + diff list
            if not all(v["preconditions"].values()):
                ok = False
                continue
            if any(i in v["reply_differing_turns"] for i in listed_idx):
                ok = False                            # a listed reply moved under a lesion: the leak
    return bool(ok)


def perfect(rng):
    return {"iw1": _arms(rng), "iw2": _arms(rng)}


def perfect_unlisted(rng):
    # the parse lesion changes a probe (expected) and one listed turn stays identical -- scope: F4 reads listed
    d = {"iw1": _arms(rng), "iw2": _arms(rng)}
    for arms in d.values():
        for t in arms["parse_lesion"][0]["turns"]:
            if t.get("message") in _PROBES:
                t["answer"] = "a differently worded lesion abstain."
                t["sha_canon"] = "canon-lesion-probe"
                break
    return d


def null_leak(rng):
    d = {"iw1": _arms(rng, listed_change=_LISTED[int(rng.integers(len(_LISTED)))]),
         "iw2": _arms(rng, listed_change=_LISTED[int(rng.integers(len(_LISTED)))])}
    return d


def null_engaged_route(rng):
    m = _LISTED[int(rng.integers(len(_LISTED)))]
    d = {"iw1": _arms(rng), "iw2": _arms(rng)}
    for arms in d.values():
        for t in arms["intact"][0]["turns"]:
            if t.get("message") == m:
                t["wr_route"] = "engaged"
    return d


def null_precondition(rng):
    d = {"iw1": _arms(rng), "iw2": _arms(rng)}
    d["iw1"]["parse_lesion"][0]["turns"][0]["status"] = 500
    return d


def null_missing_listed(rng):
    m = _LISTED[int(rng.integers(len(_LISTED)))]
    d = {"iw1": _arms(rng, missing_listed=m), "iw2": _arms(rng, missing_listed=m)}
    return d


def random(rng):
    d = {}
    for runner in ("iw1", "iw2"):
        base = _rec("intact", rng, probe_change=False)
        arms = {"intact": [base, dict(base)], "parse_lesion": [], "commit_lesion": []}
        for arm in ("parse_lesion", "commit_lesion"):
            rec = _rec(arm, rng, probe_change=False)
            for t in rec["turns"]:
                if rng.random() < 2.0 / 3.0:
                    t["sha_canon"] = "canon-rand-%d" % int(rng.integers(1 << 30))
                    t["answer"] = "random reply %d." % int(rng.integers(1 << 20))
                    t["recalled_svo"] = ["random"]
            arms[arm] = [rec, dict(rec)]
        d[runner] = arms
    return d


def info_unlisted_noise(rng):
    d = {"iw1": _arms(rng, volatile=True), "iw2": _arms(rng, volatile=True)}
    return d
