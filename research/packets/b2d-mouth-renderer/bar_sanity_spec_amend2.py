"""bar-sanity spec for B2d Amendment 2's row bar (A2.7): a row reads MOUTH-LOAD-BEARING only when all six gate-seed
cells are DEFINED and in each the lesion changes the mouth reply (the Qwen spans spoken on turn 2).

`decide` runs the REAL scorer (`tools/b2d_amend2.py`: `arm_state_doc`, `judge_cell`, `row_reading`) over in-memory
arm documents in the probe's own format, for the affect-coloring row on the selected candidate's turn.

  perfect        the faculty shapes what Qwen says on every seed: the mood is engaged (tone level 1..3 or -1..-3),
                 the manner clause reaches the prompt, every manner render is spoken; on the rich form a positive mood
                 sometimes adds one fact (plan). 1 to 3 Qwen sentences per turn; the null floor is clean.
  null           engaged, Qwen speaks, but the lesion does not reach the mouth (same manner, same words).
  null_hostonly  the lesion changes only host text around the Qwen spans (a tone mark), never Qwen's words.
  null_silent    Qwen never speaks on the turn (an abstention): MOUTH-SILENT.
  random         on each seed the lesion's mouth reply differs with probability 0.5, unrelated to the faculty.
  info_verifydrop  the perfect brain, but each manner render is dropped by VERIFY with p = 0.05 and re-rendered
                 without the manner (the regen path), so that fact reads as the lesion's.
  info_unengaged the perfect brain, but each seed's mood is neutral on the turn with p = 0.15 (a probe property).

  python tools/bar_sanity.py research/packets/b2d-mouth-renderer/bar_sanity_spec_amend2.py --n 400 \
      --out research/packets/b2d-mouth-renderer/bar_sanity_record_amend2.json
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners.b2d_amend2_probe import ARMS, GATE_SEEDS, q_label  # noqa: E402
from tools import b2d_amend2 as A2  # noqa: E402

NAME = "B2d Amendment 2 row bar (MOUTH-LOAD-BEARING, six of six)"
DESCRIPTION = __doc__.split("\n")[0]
SCORER_FILES = ["tools/b2d_amend2.py", "research/runners/b2d_amend2_probe.py"]
PIN = "0" * 40
ROW = "affect-coloring"
LABEL = q_label("c1")
WARM = "Keep the words {a}, {v}, {p} and the verb '{v}'. Write ONE short, warm, friendly sentence."


def _resp(facts, tone, manner_on, words, prefix="", silent=False):
    calls = []
    if not silent:
        for i, f in enumerate(facts):
            manner = WARM.format(a=f[0], v=f[1], p=f[2]) if manner_on[i] else ""
            calls.append({"method": "render_svo", "svo": list(f), "manner": manner, "generative": True,
                          "surface": words[i], "in_answer": True})
    answer = prefix + (" ".join(words) if not silent else "I don't know about that.")
    return {"answer": answer, "abstained": silent, "recalled_svo": None, "renderer": A2.QWEN_NAME,
            "_arm_renderer": {"mode": "production", "qwen_generate_calls": len(calls)},
            "affect": {"on": True, "tone_level": tone}, "affect_drives": {"on": True, "level": 1},
            "mouth_trace": {"wrapped": True, "tone_level": tone, "calls": calls}}


def _doc(seed, arm, resp):
    calls = resp["mouth_trace"]["calls"]
    return {"probe": "b2d-amend2", "arm": arm, "arm_env": ARMS[arm], "seed": seed, "pin": PIN,
            "renderer_mode": "production", "backend": "cupy",
            "revision": {"git_sha": PIN, "source_kind": "git_archive", "manifest_verified": True},
            "valence_map": {"entries": 13000}, "responses": {LABEL: resp},
            "qwen_prompts": {LABEL: [{"method": "_generate", "prompts": ["CONSTRAIN " + c["manner"]]} for c in calls]},
            "qwen_prompts_unassigned": 0}


def _facts(rng, k, seed):
    return [("topic_%d_%d" % (seed, i), "rel%d" % int(rng.integers(0, 9)), "obj_%d" % int(rng.integers(0, 99)))
            for i in range(k)]


def _cell(rng, seed, kind, drop=0.0, p_unengaged=0.0):
    k = int(rng.integers(1, 4))
    facts = _facts(rng, k, seed)
    tone = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
    if p_unengaged and rng.random() < p_unengaged:
        tone = 0
    plain = ["The %s %s %s." % f for f in facts]
    if kind == "silent":
        intact = lesion = _resp([], tone, [], [], silent=True)
    else:
        manner_on = [tone != 0 and not (drop and rng.random() < drop) for _ in facts]
        colored = [("Oh, the %s %ss the %s!" % f) if m else w for f, m, w in zip(facts, manner_on, plain)]
        extra = kind == "perfect" and tone > 0 and rng.random() < 0.5      # rich: a positive mood adds a fact
        i_facts = facts + (_facts(rng, 1, seed + 1000) if extra else [])
        i_words = colored + (["Also %s %s %s." % i_facts[-1]] if extra else [])
        i_manner = manner_on + ([True] if extra else [])
        intact = _resp(i_facts, tone, i_manner, i_words)
        if kind == "perfect":
            lesion = _resp(facts, 0, [False] * k, plain)
        elif kind == "null":
            lesion = _resp(facts, tone, manner_on, colored)
        elif kind == "hostonly":
            lesion = _resp(facts, tone, manner_on, colored, prefix="")
            intact = _resp(facts, tone, manner_on, colored, prefix="! ")
        elif kind == "random":
            lesion = (_resp(facts, 0, [False] * k, ["Other words %d." % i for i in range(k)]) if rng.random() < 0.5
                      else _resp(facts, tone, manner_on, colored))
            intact = _resp(facts, tone, manner_on, colored)
        else:
            raise ValueError(kind)
    arms = A2.cell_arms(ROW)
    return {a: _doc(seed, a, intact if a in A2.INTACT else lesion) for a in arms}


def _data(rng, kind, **kw):
    return {s: _cell(rng, s, kind, **kw) for s in GATE_SEEDS}


def decide(data):
    cells = {}
    for seed, docs in data.items():
        st = {a: A2.arm_state_doc(d, PIN, seed, a, LABEL) for a, d in docs.items()}
        cells[str(seed)] = A2.judge_cell(ROW, seed, st)
    return A2.row_reading(cells) == "MOUTH-LOAD-BEARING"


def perfect(rng):
    return _data(rng, "perfect")


def null(rng):
    return _data(rng, "null")


def null_hostonly(rng):
    return _data(rng, "hostonly")


def null_silent(rng):
    return _data(rng, "silent")


def random(rng):
    return _data(rng, "random")


def info_verifydrop(rng):
    return _data(rng, "perfect", drop=0.05)


def info_unengaged(rng):
    return _data(rng, "perfect", p_unengaged=0.15)
