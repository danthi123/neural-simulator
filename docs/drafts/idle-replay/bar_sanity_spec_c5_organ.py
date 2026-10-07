"""bar-sanity spec: the idle-replay organ dev bar (docs/drafts/idle-replay/PLAN.md, section 4).

THE BAR. `idle_replay_battery.dev_verdict`: PASS iff on every dev seed 1-8 the organ battery's `all_pass` holds
(recent_intact / recent_permuted / differ >= 7 of 10 noise draws; no_encode and commit_lesion latch nothing on every
draw; under the tag lesion the newest fact wins <= 5 of 10 draws and >= 4 fewer than intact; every latch read is
invariant at x0.5 and x2 of the read level).

THE DECISION IS THE REAL ONE. Synthetic brains emit what the circuit emits per draw: the candidate fact list and the
latch spike counts over the read window. From there `idle_replay_battery.seed_verdict` (via `dev_verdict`) runs the
organ's own read (`spiking_commit.read_with_variants`) and the case rules; nothing is reimplemented here.

BRAINS (8 candidates per draw: 5 build facts + the 3-fact chain; latch NS 20, read window 30 steps, read level 18
spikes, x0.5 9, x2 36).
  perfect                    the newest fact latches in both exposures; nothing latches under no-encode / commit
                             lesions; under the tag lesion a uniformly random fact latches.
  perfect_noisy              as perfect, but each intact / permuted draw reactivates the second-newest fact with
                             probability 0.05 (independently), latched counts N(70, 12) floored at 37, unlatched
                             counts with a 5% tail of 2-8.
  null_fixed_fact            one built-in fact (dog chase cat) every draw in every intact/permuted/tag arm; lesions
                             honoured (nothing latches). The production self-init organ's failure mode (NEEDS #15).
  null_host_fifo             a host "most recently taught" pick, blind to every lesion (the latch it reports is the
                             FIFO's choice in every arm).
  null_host_fifo_lesion_aware  the same FIFO, silenced under no-encode and commit lesions, but a host sort ignores the
                             excitability, so the tag lesion still returns the newest fact.
  null_graded                no bistable latch: the chosen fact's count is graded U(4, 40), the others U(0, 8).
  random                     every draw (every arm) latches a uniformly random fact or nothing.
  info_fifo_mimics_lesions   a host FIFO that ALSO randomizes under the tag lesion: it passes. A behavioural bar cannot
                             tell a host that mimics every lesion from the circuit; ownership is the host-share
                             instrument's question (prereg section 5), stated here, not hidden.
  info_weak_latch            perfect choices, latched counts N(45, 6): how the x2 read level (36) bounds a weak latch.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners import idle_replay_battery as B  # noqa: E402
from research.runners import idle_replay_organ as IR  # noqa: E402

NAME = "idle-replay organ dev bar (dev seeds 1-8, REPLAY_GRID row 0)"
DESCRIPTION = ("Synthetic latch counts per noise draw -> the organ's latch read and the battery's case rules "
               "(idle_replay_battery.dev_verdict); PASS iff all_pass on dev seeds 1-8.")
SCORER_FILES = ["research/runners/idle_replay_battery.py", "research/runners/spiking_commit.py",
                "docs/drafts/idle-replay/bar_sanity_spec_c5_organ.py"]

_P = dict(IR.REPLAY_GRID[0], OP=0)
_FACTS = {c: [list(f) for f in B.BUILD + B.CHAINS[c]] for c in B.CHAINS}
_NEWEST, _SECOND, _FIXED = 7, 6, 3          # candidate positions: newest taught, second-newest, dog chase cat


def _latched(kind, rng):
    if kind == "noisy":
        return int(max(37, round(rng.normal(70, 12))))
    if kind == "weak":
        return int(max(0, round(rng.normal(45, 6))))
    if kind == "graded":
        return int(rng.integers(4, 41))
    return int(max(0, round(rng.normal(80, 8))))


def _unlatched(kind, rng):
    if kind == "graded":
        return int(rng.integers(0, 9))
    tail = 0.05 if kind == "noisy" else 0.02
    return int(rng.integers(2, 9)) if rng.random() < tail else int(rng.poisson(0.2))


def _draw(chain, pick, kind, rng):
    return {"facts": _FACTS[chain], "slot_counts": [(_latched(kind, rng) if k == pick else _unlatched(kind, rng))
                                                    for k in range(len(_FACTS[chain]))],
            "first_gate_step": [None] * len(_FACTS[chain])}


def _pick(brain, arm, rng, miss):
    rand = int(rng.integers(0, 8))
    if brain == "random":
        k = int(rng.integers(0, 9))
        return None if k == 8 else k
    if brain == "fixed":
        return None if arm in ("no_encode", "commit_lesion") else _FIXED
    if brain == "fifo":
        return _NEWEST
    if brain == "fifo_aware":
        return None if arm in ("no_encode", "commit_lesion") else _NEWEST
    if arm in ("no_encode", "commit_lesion"):
        return None
    if arm == "tag_lesion":
        return rand
    if arm == "near_tie":
        return _NEWEST if rng.random() < 0.5 else _SECOND
    return _SECOND if rng.random() < miss else _NEWEST          # intact / permuted (perfect, fifo_mimic)


def _brain(brain, kind="perfect", miss=0.0):
    def gen(rng):
        per = {}
        for s in B.DEV_SEEDS:
            d = {k: [] for k in ("intact", "permuted", "tag_lesion", "no_encode", "commit_lesion", "near_tie")}
            for n in range(B.N_DRAWS):
                for arm in ("intact", "permuted", "tag_lesion", "near_tie"):
                    chain = "permuted" if arm == "permuted" else "intact"
                    d[arm].append(_draw(chain, _pick(brain, arm, rng, miss), kind, rng))
                if n < B.N_LESION_DRAWS:
                    for arm in ("no_encode", "commit_lesion"):
                        d[arm].append(_draw("intact", _pick(brain, arm, rng, miss), kind, rng))
            per[s] = {"seed": s, "params": _P, "draws": d}
        return per
    return gen


def decide(data):
    return B.dev_verdict(data)["verdict"] == "PASS"


perfect = _brain("perfect")
perfect_noisy = _brain("perfect", "noisy", miss=0.05)
null_fixed_fact = _brain("fixed")
null_host_fifo = _brain("fifo")
null_host_fifo_lesion_aware = _brain("fifo_aware")
null_graded = _brain("perfect", "graded")
random = _brain("random")
info_fifo_mimics_lesions = _brain("perfect")        # identical draws to `perfect`: indistinguishable by behaviour
info_weak_latch = _brain("perfect", "weak")
