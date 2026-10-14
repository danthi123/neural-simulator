"""bar-sanity spec: the amended G4 of the BG live-seam retirement (amendment 2026-10-14, the spiking commit read).

THE BAR. research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md: over the six gate seeds x four target turns
(24 seam_b events), the committed action read at READ_FRAC x0.5 and x2 equals the action at READ_FRAC (no change on
any event), with the precondition that every neural-seam event (seam_b, seam_readout, seam_lesion_value: 72) carries
a commit record naming its committed action (else UNDEFINED, which this spec counts as not passing).

THE DECISION IS THE REAL ONE. A synthetic brain emits what the network emits per event: the two stage latches' spike
counts over the read window. From there the REAL code decides: `bg_commit_read` (the organ's read and its x0.5 / x2
re-reads, on a per-step latch raster carrying those counts), `host_share_bg_seam.commit_event_measures` (the per-event
measures the scorer pools) and `host_share_bg_seam.g4_commit_holds` (the bar), with the engagement and event-count
preconditions as `verdict` applies them (BARS_AMENDED).

BRAINS (a committed channel per event: SPEAK on the question, STAY-SILENT on the two empty turns, either or none on
'hi'; the other latch is unlatched).
  perfect          latched N(84, 7) (dev seeds 1-8, row 4: 67-104, mean 83.7, sd 7.2), unlatched Poisson(0.2)
                   with a 2% tail of 2-8 (dev: 0 on 780 of 781 unlatched reads, max 1).
  perfect_noisy    latched N(70, 12) floored at 37 (the x2 read level is 36), unlatched with a 5% tail of 2-8.
  null_graded      no bistable latch: the committed channel reads a graded U(4, 60) that tracks its input (the
                   2026-10-02 failure: a held rate near the read level).
  null_unengaged   perfect counts, but 10% of the 72 neural-seam events carry no commit record.
  random           both latches U(0, 100).
  info_weak_latch  latched N(45, 6): shows how the x2 read level bounds a weak latch.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners import bg_action_selection_production_organ as BG  # noqa: E402
from research.runners import host_share_bg_seam as H  # noqa: E402

NAME = "BG live seam, amended G4 (the commit read is threshold-invariant on every gate event)"
DESCRIPTION = ("Synthetic stage-latch counts per event -> the real bg_commit_read, commit_event_measures and "
               "g4_commit_holds with the engagement precondition; PASS iff no seam_b event changes at x0.5 / x2.")
SCORER_FILES = ["research/runners/host_share_bg_seam.py", "research/runners/bg_action_selection_production_organ.py",
                "research/runners/spiking_commit.py", "docs/drafts/bg-commit/bar_sanity_spec_bg_commit_g4.py"]

_ROW = BG.bg_commit_row(BG.BG_COMMIT_OP)
_T = BG.T_DRIVE + BG.T_HOLD
_SEEDS = (42, 43, 44, 100, 101, 102)
_TURNS = (("hsb_empty1", BG.STAY_SILENT), ("hsb_q", BG.SPEAK), ("hsb_empty2", BG.STAY_SILENT), ("hsb_hi", "either"))
_ARMS = ("seam_b", "seam_readout", "seam_lesion_value")


def _raster(counts):
    """A per-step latch raster whose last READ_STEPS carry `counts` (the read sees only that window)."""
    rs = int(_ROW["READ_STEPS"])
    slot = np.zeros((_T, 2), dtype=np.int32)
    for k, c in enumerate(counts):
        c = int(max(0, c))
        base, extra = divmod(c, rs)
        slot[_T - rs:, k] = base
        slot[_T - rs:_T - rs + extra, k] += 1
    return slot


def _latched(kind, rng):
    if kind == "graded":
        return int(rng.integers(4, 61))
    if kind == "noisy":
        return int(max(37, round(rng.normal(70, 12))))
    if kind == "weak":
        return int(max(0, round(rng.normal(45, 6))))
    return int(max(0, round(rng.normal(84, 7))))


def _unlatched(kind, rng):
    if rng.random() < (0.05 if kind == "noisy" else 0.02):
        return int(rng.integers(2, 9))
    return int(rng.poisson(0.2))


def _brain(kind, rng, drop=0.0):
    events = []
    for _s in _SEEDS:
        for _label, want in _TURNS:
            for arm in _ARMS:
                if kind == "random":
                    counts = [int(rng.integers(0, 101)), int(rng.integers(0, 101))]
                else:
                    ch = want if want != "either" else int(rng.integers(0, 3)) - 1        # -1: no commit
                    counts = [_unlatched(kind, rng), _unlatched(kind, rng)]
                    if ch in (0, 1):
                        counts[ch] = _latched(kind, rng)
                cm = BG.bg_commit_read(_ROW, np.zeros((_T, 2), dtype=np.int32), _raster(counts),
                                       np.zeros((_T, 2), dtype=np.int32))
                cm["old_latch_winner"] = cm["winner"]
                e = {"kind": "bg_neural", "winner": cm["winner"], "commit": cm, "arm": arm}
                if drop and rng.random() < drop:
                    e.pop("commit")
                events.append(e)
    return {"events": events}


def decide(data):
    """The amended G4 with its preconditions, as `verdict` applies them to a pooled score."""
    m_eng = [H.commit_event_measures(e, True)["engaged"] for e in data["events"]]
    thr = [H.commit_event_measures(e, True)["commit_thr_change"] for e in data["events"] if e["arm"] == "seam_b"]
    bars = H.BARS_AMENDED
    if not m_eng or sum(m_eng) / len(m_eng) < bars["commit_engaged_min"]:
        return False                                       # UNDEFINED in the registered verdict
    if len(thr) < bars["min_events_per_measure"]:
        return False
    return bool(H.g4_commit_holds({"commit_thr_change": round(sum(thr) / len(thr), 4)}, bars))


def perfect(rng):
    return _brain("perfect", rng)


def perfect_noisy(rng):
    return _brain("noisy", rng)


def null_graded(rng):
    return _brain("graded", rng)


def null_unengaged(rng):
    return _brain("perfect", rng, drop=0.10)


def random(rng):
    return _brain("random", rng)


def info_weak_latch(rng):
    return _brain("weak", rng)
