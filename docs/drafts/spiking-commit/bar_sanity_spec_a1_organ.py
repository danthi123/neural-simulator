"""bar-sanity spec: word-role-commit step O (the organ bar), PLAN docs/drafts/spiking-commit/PLAN.md application A1.

THE BAR. Per gate seed the commit battery's `all_pass` (every case right AND every case's slot read threshold-
invariant at READ_FRAC x0.5 / x2), and step O PASS iff all_pass on 6/6 seeds (tools/s2p_score.py score_O).

THE DECISION IS THE REAL ONE. Synthetic brains emit what the network emits, per presented word: slot-latch spike
counts over the read window (plus role-ensemble counts, used only for the paired host-rule record). From there the
REAL code decides: `WordRoleParserOrgan.commit_parse_from_items` (the threshold read, the x0.5 / x2 re-reads, the
first-to-fill assignment), `word_role_commit_organ.case_record` / `battery_verdict` (case pass, all_pass), and
`s2p_score.score_O` (6/6, cell validity) on files written to a temporary directory with valid provenance sidecars.
The organ object is constructed without building its network (only its params are read).

BRAINS.
  perfect          taught single-role words latch their role (dog/brain agent, chase/use action), unheard words and
                   every lesion latch nothing; a latched slot reads N(80, 8) spikes (dev seeds 1-8: 72-86), an
                   unlatched one Poisson(0.2) with a 2% tail of 2-8 (dev: 0-1).
  perfect_noisy    as perfect with a wider latch, N(70, 12) floored at 37 (the x2 read level is 36), and a 5% tail
                   of 2-8 on unlatched slots.
  null_positional  the positional map: every word, heard or not, latches ROLES[position]; word lesion ignored (the
                   commit and latch lesions still silence it).
  null_lesion_inert  perfect intact parses, but no lesion removes a latch.
  null_graded      no bistable latch: a committed slot reads a graded U(4, 40) that tracks input, not a state.
  random           each word latches a uniformly random role or nothing.
  info_weak_latch  latched N(50, 6), unlatched as perfect: shows how the x2 read level bounds a weak latch.
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

import numpy as np  # noqa: E402

from research.runners import word_role_parser_organ as W  # noqa: E402
from research.runners import word_role_commit_organ as C  # noqa: E402
import s2p_score as S  # noqa: E402

NAME = "word-role-commit step O (organ battery, 6/6 gate seeds)"
DESCRIPTION = ("Synthetic slot-latch spike counts per word -> the real commit read, case pass, battery all_pass and "
               "s2p_score.score_O; PASS iff 6/6 seeds all_pass at COMMIT_GRID row 0.")
SCORER_FILES = ["tools/s2p_score.py", "research/runners/word_role_commit_organ.py", "research/runners/spiking_commit.py",
                "research/runners/word_role_parser_organ.py", "docs/drafts/spiking-commit/bar_sanity_spec_a1_organ.py"]

_PACKET = "word-role-commit"
_PIN = "0" * 40
_OP = 0
_LEX = {"dog": "agent", "chase": "action", "brain": "agent", "use": "action"}
_PARAMS, _COMMIT = W.commit_organ_params(_OP)
_ORGAN = W.WordRoleParserOrgan(W.BATTERY_FACTS, seed=1, params=_PARAMS, commit=_COMMIT)   # never built
_ROLE_N = {"agent": 20, "action": 20, "patient": 20}


def _latched(kind, rng):
    if kind == "graded":
        return int(rng.integers(4, 41))
    if kind == "noisy":
        return int(max(37, round(rng.normal(70, 12))))
    if kind == "weak":
        return int(max(0, round(rng.normal(50, 6))))
    return int(max(0, round(rng.normal(80, 8))))


def _unlatched(kind, rng):
    tail = 0.05 if kind == "noisy" else 0.02
    if rng.random() < tail:
        return int(rng.integers(2, 9))
    return int(rng.poisson(0.2))


def _commits(brain, word, pos, kw, rng):
    lesioned = bool(kw)
    if brain == "positional":
        return None if (kw.get("commit_lesion") or kw.get("latch_lesion")) else W.ROLES[min(pos, 2)]
    if brain == "lesion_inert":
        return _LEX.get(word)
    if brain == "random":
        k = int(rng.integers(0, 4))
        return None if k == 3 else W.ROLES[k]
    return None if lesioned else _LEX.get(word)


def _items(brain, latch_kind, content, kw, rng):
    on, items = set(), []
    for i, w in enumerate(content):
        r = _commits(brain, w, i, kw, rng)
        if r is not None:
            on.add(r)
        role_counts = {x: (_ROLE_N[x] if _LEX.get(w) == x else int(rng.integers(0, 4))) for x in W.ROLES}
        slot = {x: (_latched(latch_kind, rng) if x in on else _unlatched(latch_kind, rng)) for x in W.ROLES}
        items.append({"word": w, "pos": min(i, 2), "role_counts": role_counts,
                      "gate_counts": {x: 0 for x in W.ROLES}, "slot_counts": slot})
    return items


def _brain(brain, latch_kind):
    def gen(rng):
        seeds = {}
        for s in S.GATE_SEEDS:
            cases = {}
            for name, content, want, kw in C.CASES:
                r = _ORGAN.commit_parse_from_items(list(content), _items(brain, latch_kind, content, kw, rng))
                cases[name] = C.case_record(_ORGAN, content, want, r, bool(kw))
            seeds[s] = cases
        return seeds
    return gen


def decide(data):
    with tempfile.TemporaryDirectory() as d:
        sel = os.path.join(d, "selection.json")
        with open(sel, "w") as fh:
            json.dump({"packet": _PACKET, "verdict": "SELECTED", "op": _OP}, fh)
        prereg = S.PACKETS[_PACKET]["prereg"]
        for s, cases in data.items():
            bv = C.battery_verdict(cases)
            f = os.path.join(d, "s%d.json" % s)
            with open(f, "w") as fh:
                json.dump({"seed": s, "op_index": _OP, "prereg": prereg, "all_pass": bv["all_pass"],
                           "cases": {k: {"pass": v["pass"]} for k, v in cases.items()}}, fh)
            with open(f + ".prov.json", "w") as fh:
                json.dump({"git_sha": _PIN, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
                           "source_manifest_verified_at_exit": True, "env": {"SIM_BACKEND": "numpy"}}, fh)
        res, _code = S.score_O(_PACKET, d, sel, _PIN)
        return res["verdict"] == "PASS"


perfect = _brain("perfect", "perfect")
perfect_noisy = _brain("perfect", "noisy")
null_positional = _brain("positional", "perfect")
null_lesion_inert = _brain("lesion_inert", "perfect")
null_graded = _brain("perfect", "graded")
random = _brain("random", "perfect")
info_weak_latch = _brain("perfect", "weak")
