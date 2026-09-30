"""STAGE 1 (INFANT) EXIT GATE -- the first developmental stage-exit attempt, run END TO END on the INTEGRATED production
brain: the live `webapp.server.brain_chat` handler at production defaults, one fresh brain per arm in its own process,
never an isolated de-risk runner. Master roadmap section 4, STAGE 1 -- "Stage-exit GO: category held-out inheritance;
word->concept recall; lived-fact recall; valence-inheritance r>=0.55" -- operationalized at the REPLY level (what the
user sees: `answer`, `abstained`, `recalled_svo`, `hypothesis_svo`, the feel read-out text), with the internal
read-outs as secondary. Preregistration (frozen bars, arms, turns, predictions, UNDEFINED rules):
research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md. Execution packet for the local model:
docs/drafts/stage1-exit/HANDOFF.md.

THE FOUR CRITERIA (one probe GROUP each; two diagnostic sub-groups):
  C1 category held-out inheritance   group `cat_disc` (primary: the category is DISCOVERED from shared contexts, never
                                      told) + `cat_told` (diagnostic: the category is told through the `isa` convention)
  C2 word->concept recall            group `word` (a NEVER-SEEN word is bound to a concept in-session and reused across
                                      roles, through anaphora, with the moat intact)
  C3 lived-fact recall               group `lived` (a fact told this session is recalled after intervening turns, and
                                      the episode itself is disclosed; the pre-exposure baseline abstains)
  C4 valence inheritance r>=0.55     group `val_heldout` (primary: the reply-visible valence differential for held-out
                                      words vs the Warriner norm, Pearson r per seed) + `val_novel` (diagnostic: a
                                      novel word acquires the valence of the company it keeps in-session)

ARMS (every group): intact_a (production, nothing installed), intact_b (an independent same-seed rebuild = the null),
one or two LESION arms (the responsible faculty cut: an env flag production already carries, or an in-process
monkeypatch in the arm's worker), a PERMUTED-EXPOSURE arm (the teacher's exposure permuted: the reply must follow the
exposure), and for C4 one INFORMATIONAL arm (a default-off candidate mechanism; never counted). Lesion arms get a
`.rep0` repeat (a second fresh process) so a reply-level effect is "reproduced against the lesion's own repeat"
(docs/TERMS.md, load-bearing).

HARNESS (identical in every arm incl. intact_a; declared in the prereg): numpy/CPU, one BLAS thread, stub renderer, no
LLM, tiny-demo brain with the LTM tier at its shipped default (expected `source` = "tiny-demo +LTM"),
`BRAIN_CHAT_SEED=<seed>`, `PYTHONHASHSEED=0`, `rich` omitted (the production default path). Group `lived` adds
`BRAIN_EPISODIC_STORE=1` (the D5 BTSP write's cupy-default value, forced on numpy exactly as the load-bearing
battery's episodic drive does). No sim/ or webapp/ edit anywhere. NO seed threading into the three host-built organs
that ignore BRAIN_CHAT_SEED at HEAD (a declared production limitation, not patched here).

  .venv/bin/python -m research.runners.stage1_exit_gate --selftest
  bash tools/memcap.sh 9 -- .venv/bin/python -m research.runners.stage1_exit_gate --shard --seed 7 --group word \
      --arm intact_a --out research/findings/raw/_stage1_exit/dev/s7/word__intact_a.json
  .venv/bin/python -m research.runners.stage1_exit_gate --aggregate research/findings/raw/_stage1_exit/gate \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_stage1_exit/gate/aggregate.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to  # noqa: E402  (the lesion-vs-null attribution, gates/attribution_required)

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 13))
EXPECTED_SOURCE = "tiny-demo +LTM"
GATE_OUT_DIR = "research/findings/raw/_stage1_exit/gate"
PREREG = "research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md"
VALENCE_R_BAR = 0.55          # the roadmap's own Stage-1 bar (DR-2, every seed)
CONTROL_R_CEILING = 0.20      # a lesion / permuted arm must read below this (or UNDEFINED) for the effect to vanish
MIN_WORDS_FOR_R = 12          # fewer parsed feel read-outs than this -> r UNDEFINED for that seed
MOOD_TOL = 0.03               # the ladder's neutral tolerance (LADDER_NEUTRAL_TOL): |differential| below = neutral

BASE_ENV = {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "BRAIN_LTM_SHIP_DEFAULT": "1", "PYTHONHASHSEED": "0"}

# The 24 held-out valence words (C4a), frozen. Every one is (a) a STRONG word under the production salience gate
# (|Warriner v - 5| >= 2, so it moves the mood at all) and (b) in the DR-2 leave-one-out learned map (so its production
# value is INFERRED from the company it keeps in the corpus, never its own norm). Chosen once from the 78 words that
# satisfy both, spanning the range on each side (every 4th positive, every 2nd negative, by norm), then frozen here.
# (word, warriner_v9): the norm is the CRITERION, read nowhere by the brain at run time.
HELDOUT_WORDS = [   # norms copied verbatim from research/runners/affect_production_organ.WARRINER (v9)
    ("nice", 7.4), ("home", 7.4), ("proud", 7.6), ("help", 7.3), ("brave", 7.2), ("sing", 7.0), ("gift", 7.6),
    ("friend", 7.9), ("warm", 7.1), ("smile", 7.7), ("kind", 7.3), ("happy", 8.5),
    ("trouble", 3.0), ("lonely", 2.3), ("bite", 3.0), ("cry", 2.4), ("scared", 2.8), ("mad", 2.9), ("bad", 2.5),
    ("hurt", 2.2), ("lost", 2.8), ("die", 1.7), ("evil", 2.2), ("war", 2.1),
]

# ── the probe GROUPS: frozen turn tables. (label, message, role). ONE session per group per arm (one fresh ChatBrain).
#    roles: pre (pre-exposure baseline) · teach (the teacher's exposure) · distract · probe (a scored reply) · control
#    (a scored known-fact / moat reply) · expose (C4 word exposure) · feel (the C4 read-out turn) ────────────────────
_CAT_CONTEXT = [  # shared by cat_disc: two categories built from shared contexts only (eat-patient + motion context)
    ("t_robin_eat", "the robin eats the worm", "teach"),
    ("t_trout_eat", "the trout eats the bug", "teach"),
    ("t_robin_fly", "the robin flies in the sky", "teach"),
    ("t_trout_swim", "the trout swims in the lake", "teach"),
    ("t_sparrow_eat", "the sparrow eats the worm", "teach"),
    ("t_salmon_eat", "the salmon eats the bug", "teach"),
    ("t_sparrow_fly", "the sparrow flies in the sky", "teach"),
    ("t_salmon_swim", "the salmon swims in the lake", "teach"),
    ("t_finch_eat", "the finch eats the worm", "teach"),
    ("t_finch_fly", "the finch flies in the sky", "teach"),
]
_CAT_PROBES = [
    ("p_canary_eat", "what does the canary eat", "probe"),          # DIRECT route: held-out inheritance (strict recall)
    ("p_canary_might", "what might the canary eat", "probe"),       # GENERATIVE route: the brain's own hypothesis
    ("c_robin_eat", "what does the robin eat", "control"),          # a taught member (identity control)
    ("c_trout_eat", "what does the trout eat", "control"),          # the other category's member
    ("c_canary_swim", "what does the canary swim", "control"),      # moat: never taught, not inheritable -> abstain
]
_WORD_PROBES = [
    ("p_bear_chase", "what does the bear chase", "probe"),          # the new word decoded as PATIENT
    ("p_it_eat", "what does it eat", "probe"),                      # anaphora through the new word: REPORTED, not counted
    ("p_otter_eat", "what does the otter eat", "probe"),            # the new word as AGENT
    ("c_otter_chase", "what does the otter chase", "control"),      # moat on the new word (intact: abstain)
    ("c_clam_eat", "what does the clam eat", "control"),            # moat (intact: abstain; permuted: otter)
    ("c_dog_chase", "what does the dog chase", "control"),          # a built-in fact: must survive every arm
]
_LIVED_DISTRACT = [   # every distractor is a QUESTION: on dev seed 7 the declarative "tell me about the sky" was
    ("d_weather", "how is the weather today", "distract"),          # ACQUIRED by the host SVO extractor as the junk
    ("d_sky", "what is the weather like tomorrow", "distract"),     # fact [tell, about, sky] (FAILURE_LOG 2026-09-29)
    ("d_cat", "what does the cat eat", "distract"),
]
_LIVED_PROBES = [
    ("p_wolf_bite", "what does the wolf bite", "probe"),            # semantic recall of the lived fact
    ("p_wolf_discuss", "did we discuss the wolf", "probe"),         # the episode itself (D5 dAP recall gate)
    ("c_wolf_chase", "what does the wolf chase", "control"),        # moat: never lived -> abstain
    ("c_dog_chase", "what does the dog chase", "control"),          # built-in fact survives
]


def _val_heldout_turns(seed):
    """C4a: a seed-specific order of the 24 held-out words; each word is exposed twice (bare, so only that word can
    carry affect) then read with the production feel query. The order is part of the design (carry-over averages
    out across seeds); the word set and the exposure count are frozen."""
    import random
    words = [w for w, _v in HELDOUT_WORDS]
    random.Random(int(seed) * 1009 + 17).shuffle(words)
    turns = [("feel_base", "how do you feel", "feel")]
    for w in words:
        turns += [("%s_e1" % w, w, "expose"), ("%s_e2" % w, w, "expose"), ("%s_feel" % w, "how do you feel", "feel")]
    return turns


def _val_novel_turns(swap_company=False):
    """C4b: two NOVEL words (outside every lexicon and store), `glorp` and `blick`, keep valenced company; then each is
    presented ALONE and must move the reply-visible mood in its inherited direction against the mood the other word
    left behind. `swap_company=True` is the PERMUTED-EXPOSURE arm: the same probe words, the same positions, only the
    company each word kept is swapped (glorp with the negative words, blick with the positive), so the expected
    moves reverse."""
    good, badc = "wonderful and happy", "terrible and sad"
    if swap_company:
        good, badc = badc, good
    P = "the glorp makes everyone %s" % good       # 5 content words: never a 3-word SVO assertion, never taught
    N = "the blick makes everyone %s" % badc
    return [
        ("feel_base", "how do you feel", "feel"),
        ("e_pos1", P, "expose"), ("e_pos2", P, "expose"), ("e_pos3", P, "expose"), ("feel_pos", "how do you feel", "feel"),
        ("e_neg1", N, "expose"), ("e_neg2", N, "expose"), ("e_neg3", N, "expose"), ("feel_neg", "how do you feel", "feel"),
        ("x_pos", "glorp", "probe"), ("feel_x_pos", "how do you feel", "feel"),         # intact: must flip toward +
        ("e_pos4", P, "expose"), ("e_pos5", P, "expose"), ("e_pos6", P, "expose"), ("feel_pos2", "how do you feel", "feel"),
        ("x_neg", "blick", "probe"), ("feel_x_neg", "how do you feel", "feel"),         # intact: must flip toward -
    ]


GROUPS = {
    "cat_disc": {
        "criterion": "C1", "primary": True, "env": {},
        "turns": _CAT_CONTEXT + [("t_canary_fly", "the canary flies in the sky", "teach")] + _CAT_PROBES,
        "permuted_turns": _CAT_CONTEXT + [("t_canary_fly", "the canary swims in the lake", "teach")] + _CAT_PROBES,
        "lesions": {"lesion_draw": {"env": {"BRAIN_SPIKING_DRAW_LESION": "1"}, "patch": None,
                                    "cuts": "the spiking generative DRAW's likelihood (uniform drive)"}},
        "informational": {},
    },
    "cat_told": {
        "criterion": "C1", "primary": False, "env": {},
        "turns": [("t_robin_isa", "the robin isa bird", "teach"), ("t_trout_isa", "the trout isa fish", "teach"),
                  ("t_sparrow_isa", "the sparrow isa bird", "teach"), ("t_salmon_isa", "the salmon isa fish", "teach"),
                  ("t_bird_eat", "the bird eats the worm", "teach"), ("t_fish_eat", "the fish eats the bug", "teach"),
                  ("t_canary_isa", "the canary isa bird", "teach"),
                  ("c_canary_isa", "what is a canary", "control"),          # premise 1 recalled (isa)
                  ("c_bird_eat", "what does the bird eat", "control"),      # premise 2 recalled (class property)
                  ] + _CAT_PROBES[:2] + [("c_canary_swim", "what does the canary swim", "control")],
        "permuted_turns": None,   # filled below: canary told `isa fish`
        "lesions": {"lesion_draw": {"env": {"BRAIN_SPIKING_DRAW_LESION": "1"}, "patch": None,
                                    "cuts": "the spiking generative DRAW's likelihood (uniform drive)"}},
        "informational": {},
    },
    "word": {
        "criterion": "C2", "primary": True, "env": {},
        "turns": [("pre_bear_chase", "what does the bear chase", "pre"),
                  ("t_otter_eat", "the otter eats the clam", "teach"),
                  ("t_bear_chase", "the bear chases the otter", "teach")] + _WORD_PROBES,
        "permuted_turns": [("pre_bear_chase", "what does the bear chase", "pre"),
                           ("t_otter_eat", "the clam eats the otter", "teach"),
                           ("t_bear_chase", "the otter chases the bear", "teach")] + _WORD_PROBES,
        "lesions": {"lesion_recruit": {"env": {}, "patch": "lesion_recruit",
                                       "cuts": "recruit-an-assembly: no reserved cleanup slot is ever claimed for a "
                                               "never-seen word (OneBrainComposer._recruit_word -> False)"}},
        "informational": {},
    },
    "lived": {
        "criterion": "C3", "primary": True, "env": {"BRAIN_EPISODIC_STORE": "1"},
        "turns": [("pre_wolf_bite", "what does the wolf bite", "pre"), ("pre_wolf_discuss", "did we discuss the wolf", "pre"),
                  ("t_wolf_bite", "the wolf bites the apple", "teach")] + _LIVED_DISTRACT + _LIVED_PROBES,
        "permuted_turns": [("pre_wolf_bite", "what does the wolf bite", "pre"), ("pre_wolf_discuss", "did we discuss the wolf", "pre"),
                           ("t_wolf_bite", "the wolf bites the pear", "teach")] + _LIVED_DISTRACT + _LIVED_PROBES,
        "lesions": {"lesion_recall": {"env": {"BRAIN_SUBSTRATE_RECALL_LESION": "1"}, "patch": None,
                                      "cuts": "the substrate-first spiking recall read (ChatBrain._substrate_recall -> None)"},
                    "lesion_episodic": {"env": {"BRAIN_EPISODIC_LESION": "1"}, "patch": None,
                                        "cuts": "the D5 hippocampal dAP episodic recall gate"}},
        "informational": {},
    },
    "val_heldout": {
        "criterion": "C4", "primary": True, "env": {},
        "turns": None,            # seed-specific order: _val_heldout_turns(seed)
        "permuted_turns": None,   # same turns; the PERMUTATION is of the learned map (an in-process patch)
        "lesions": {"lesion_affect": {"env": {"BRAIN_AFFECT_LESION": "1"}, "patch": None,
                                      "cuts": "the graded-affect ladder read (affect_out clamped to 0)"}},
        "informational": {"dr2_off": {"env": {"BRAIN_AFFECT_DR2": "0"}, "patch": None,
                                      "note": "the raw-norm LOOKUP oracle (value = the word's own norm): a ceiling, "
                                              "never a pass condition"}},
    },
    "val_novel": {
        "criterion": "C4", "primary": False, "env": {},
        "turns": _val_novel_turns(),
        "permuted_turns": _val_novel_turns(swap_company=True),   # the company is swapped, the probes are not
        "lesions": {"lesion_affect": {"env": {"BRAIN_AFFECT_LESION": "1"}, "patch": None,
                                      "cuts": "the graded-affect ladder read (affect_out clamped to 0)"}},
        "informational": {"learned_vocab": {"env": {"BRAIN_AFFECT_LEARNED_VOCAB": "1"}, "patch": None,
                                            "note": "the 2026-09-24 learned affect vocabulary (default-OFF candidate "
                                                    "mechanism); reported, never counted"}},
    },
}
GROUPS["cat_told"]["permuted_turns"] = [
    (l, ("the canary isa fish" if l == "t_canary_isa" else m), r) for (l, m, r) in GROUPS["cat_told"]["turns"]]
PRIMARY_GROUPS = tuple(g for g, d in GROUPS.items() if d["primary"])


def turns_for(group, seed, arm):
    g = GROUPS[group]
    if group == "val_heldout":
        return _val_heldout_turns(seed)
    if arm == "permuted":
        return list(g["permuted_turns"])
    return list(g["turns"])


def arms_for(group, with_repeats=True, with_informational=True):
    g = GROUPS[group]
    arms = ["intact_a", "intact_b"]
    for les in g["lesions"]:
        arms.append(les)
        if with_repeats:
            arms.append(les + ".rep0")
    arms.append("permuted")
    if with_informational:
        arms += list(g["informational"])
    return arms


def arm_spec(group, arm):
    """(env, patch, kind) for an arm. kind in intact | lesion | permuted | informational."""
    g = GROUPS[group]
    base = arm.split(".rep")[0]
    if base in ("intact_a", "intact_b"):
        return {}, None, "intact"
    if base == "permuted":
        return {}, ("permuted_map" if group == "val_heldout" else None), "permuted"
    if base in g["lesions"]:
        d = g["lesions"][base]
        return dict(d["env"]), d["patch"], "lesion"
    if base in g["informational"]:
        d = g["informational"][base]
        return dict(d["env"]), d["patch"], "informational"
    raise ValueError("unknown arm %s for group %s" % (arm, group))


# ══════════════════════════════════════════════ WORKER (inside the arm subprocess) ═════════════════════════════════
def _derangement(n, rng):
    while True:
        p = list(rng.permutation(n))
        if all(p[i] != i for i in range(n)):
            return p


def install_patch(name, seed):
    """The in-process manipulations (no sim/ or webapp/ edit). Returns a record dict the worker stores."""
    if name is None:
        return None
    if name == "lesion_recruit":
        from research.runners import one_brain_composer as OBC
        rec = {"patch": name, "calls": 0}

        def _recruit_word(self, w):        # the pool reads as exhausted: a never-seen word never gets a cleanup slot
            rec["calls"] += 1
            return False
        OBC.OneBrainComposer._recruit_word = _recruit_word
        return rec
    if name == "permuted_map":
        import numpy as np
        from research.runners import affect_production_organ as AO
        m = dict(AO._get_learned_valence())
        words = [w for w, _v in HELDOUT_WORDS if w in m]
        perm = _derangement(len(words), np.random.default_rng(int(seed) + 7))
        vals = [m[w] for w in words]
        for i, w in enumerate(words):
            m[w] = vals[perm[i]]
        AO._LEARNED_VALENCE = m
        return {"patch": name, "n_words": len(words), "perm": [int(x) for x in perm]}
    raise ValueError(name)


_DIFF_RE = re.compile(r"valence differential ([+-]?\d+\.\d+)")


def parse_feel(answer):
    """The reply-level read of a feel turn: the sign word and the differential PRINTED IN THE REPLY TEXT."""
    a = str(answer or "")
    m = _DIFF_RE.search(a)
    d = float(m.group(1)) if m else None
    sign = "positive" if "reads positive" in a else ("negative" if "reads negative" in a else
                                                     ("neutral" if "reads neutral" in a else None))
    return {"differential": d, "sign": sign}


def run_worker(seed, group, arm, out_path, labels=None):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env, patch, kind = arm_spec(group, arm)
    for k, v in dict(GROUPS[group]["env"], **env).items():
        os.environ[k] = v
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    patch_rec = install_patch(patch, seed)
    hash_ok = os.environ.get("PYTHONHASHSEED") == "0"
    # The corpus-learned production organs (the one-brain cross-edge pool, the learned animacy lexicon) read
    # data/corpus/tinystories.txt from the repo root and DEGRADE silently without it (a git worktree has none unless
    # linked; pool_provision.sh syncs it). Recorded per arm and required by the aggregate's preconditions, so a
    # degraded build can never be scored as the production brain.
    corpus_present = os.path.exists(os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    t0 = time.time()
    from webapp.server import brain_chat, BrainChatRequest
    import_s = time.time() - t0
    turns = turns_for(group, seed, arm)
    if labels:
        keep = set(labels.split(","))
        turns = [t for t in turns if t[0] in keep]
    session = "s1_%s_%s_%d" % (group, arm.replace(".", "_"), int(seed))
    responses, timings = {}, {}
    for i, (label, msg, role) in enumerate(turns):
        t1 = time.time()
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                            reset=(i == 0)))      # `rich` OMITTED = the production default path
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
        timings[label] = round(time.time() - t1, 1)
    out = {"runner": "research/runners/stage1_exit_gate.py", "prereg": PREREG,
           "seed": int(seed), "group": group, "arm": arm, "kind": kind, "criterion": GROUPS[group]["criterion"],
           "env": dict(GROUPS[group]["env"], **env), "patch": patch_rec, "pythonhashseed_ok": hash_ok,
           "corpus_present": corpus_present,
           "session": session, "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns],
           "responses": responses, "timings_s": timings, "import_s": round(import_s, 1),
           "wall_s": round(time.time() - t0, 1), "elapsed_seconds": round(time.time() - t0, 1),
           "expected_source": EXPECTED_SOURCE,
           "sources": sorted({str(r.get("source")) for r in responses.values() if isinstance(r, dict)})}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[stage1 worker] seed=%d %s/%s -> %d turns, %.0fs -> %s" % (seed, group, arm, len(responses),
                                                                     out["wall_s"], out_path), flush=True)
    return 0


def run_shard(seed, group, arm, out_path, backend="numpy", labels=None, allow_gate_seed=False, prereg=None):
    """One (seed, group, arm) arm in a FRESH subprocess with the frozen env (PYTHONHASHSEED must be set before the
    interpreter starts, so it cannot be applied in-process)."""
    env = dict(os.environ)
    env.update(BASE_ENV)
    env["SIM_BACKEND"] = backend
    if backend == "cupy":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.stage1_exit_gate", "--worker", "--seed", str(int(seed)),
           "--group", group, "--arm", arm, "--out", part]
    if labels:
        cmd += ["--labels", labels]
    if allow_gate_seed:
        # the worker's own main() applies the gate-seed guard too: without the flag every gate shard dies rc=2
        # (pre-data fix 2026-09-30, found by the pool canary; the prereg is unchanged)
        cmd += ["--allow-gate-seed"]
    if prereg:
        cmd += ["--prereg", prereg]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[stage1 shard] worker FAILED rc=%s for seed=%s %s/%s" % (p.returncode, seed, group, arm), flush=True)
        return 1
    with open(part) as f:
        d = json.load(f)
    d["backend"] = backend
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    for suffix in ("", ".prov.json"):
        try:
            os.remove(part + suffix)
        except OSError:
            pass
    return 0


# ══════════════════════════════════════════════ SCORING (pure functions over shard dicts) ══════════════════════════
def _primary(resp):
    """The reply-level primary fields (docs/TERMS.md): what the user sees on that turn."""
    if not isinstance(resp, dict) or "_error" in resp:
        return None
    return {"answer": str(resp.get("answer")), "abstained": bool(resp.get("abstained")),
            "recalled_svo": (list(resp["recalled_svo"]) if resp.get("recalled_svo") else None),
            "hypothesis_svo": (list(resp["hypothesis_svo"]) if resp.get("hypothesis_svo") else None)}


def _patient(resp):
    p = _primary(resp)
    if not p:
        return None
    if p["recalled_svo"] and len(p["recalled_svo"]) == 3:
        return str(p["recalled_svo"][2]).lower()
    if p["hypothesis_svo"] and len(p["hypothesis_svo"]) == 3:
        return str(p["hypothesis_svo"][2]).lower()
    return None


def _abstained(resp):
    p = _primary(resp)
    return None if p is None else p["abstained"]


def _episodic_in_memory(resp):
    """Secondary (internal) read for the D5 disclosure; the reply-level read is `abstained` on that turn."""
    if not isinstance(resp, dict):
        return None
    ep = resp.get("episodic") or {}
    return ep.get("in_memory") if isinstance(ep, dict) else None


def null_clean(a, b, labels):
    """The null: two independent same-seed intact builds give the SAME primary reply fields on every listed turn."""
    diffs = []
    for l in labels:
        pa, pb = _primary((a or {}).get(l)), _primary((b or {}).get(l))
        if pa is None or pb is None:
            diffs.append((l, "missing_or_error"))
        elif pa != pb:
            diffs.append((l, "differs"))
    return (not diffs), diffs


def reply_differs(a, b, label):
    pa, pb = _primary((a or {}).get(label)), _primary((b or {}).get(label))
    if pa is None or pb is None:
        return None
    return pa != pb


def n_field_diffs(a, b, labels):
    """How many primary reply FIELDS differ between two arms over the listed turns (the effect SIZE the attribution
    subtracts: the lesion's count is the treatment, the null's intact_a-vs-intact_b count on the same turns is the
    control -- the run-to-run variation that would be present with no lesion at all)."""
    n = 0
    for l in labels:
        pa, pb = _primary((a or {}).get(l)), _primary((b or {}).get(l))
        if pa is None or pb is None:
            continue
        n += sum(1 for k in pa if pa[k] != pb[k])
    return n


# per-group CORRECTNESS on the intact arm and TRACKING on the permuted arm (frozen expectations) ────────────────────
def score_cat(resp, permuted=False):
    """C1. The held-out member (canary) shares the BIRD context. Reply-level: which patient the reply names on the
    two inheritance probes; the category-consistent patient is `worm` (intact) / `bug` (permuted)."""
    want, other = ("bug", "worm") if permuted else ("worm", "bug")
    direct, gen = _patient(resp.get("p_canary_eat")), _patient(resp.get("p_canary_might"))
    return {
        "direct_patient": direct, "direct_abstained": _abstained(resp.get("p_canary_eat")),
        "generative_patient": gen, "generative_abstained": _abstained(resp.get("p_canary_might")),
        "direct_correct": direct == want, "generative_correct": gen == want,
        "inherits": (direct == want) or (gen == want),          # EITHER route naming the consistent patient
        "names_other_category": other in (direct, gen),
        "identity_robin": _patient(resp.get("c_robin_eat")) == ("worm"), "identity_trout": _patient(resp.get("c_trout_eat")) == "bug",
        "moat_canary_swim": _abstained(resp.get("c_canary_swim")) is True,
    }


def score_word(resp, permuted=False):
    """C2. The never-seen word `otter` is bound in-session and reused in both roles, with the moat intact.
    `anaphora_it_eat` (`what does it eat` -> clam through the recruited word) is REPORTED, never counted: on dev seed 7
    the pronoun resolved to the built-in `cat` (the D6 referent store is a build-time vocabulary), a discourse-WM
    property named as a residual in the prereg, not the Stage-1 binding itself."""
    if not permuted:
        ok = {"bear_chase": _patient(resp.get("p_bear_chase")) == "otter",
              "otter_eat": _patient(resp.get("p_otter_eat")) == "clam",
              "moat_otter_chase": _abstained(resp.get("c_otter_chase")) is True,
              "moat_clam_eat": _abstained(resp.get("c_clam_eat")) is True}
    else:   # exposure permuted: clam eats otter; otter chases bear
        ok = {"bear_chase_not_otter": _patient(resp.get("p_bear_chase")) != "otter",
              "otter_chase_bear": _patient(resp.get("c_otter_chase")) == "bear",
              "clam_eat_otter": _patient(resp.get("c_clam_eat")) == "otter",
              "otter_eat_abstains": _abstained(resp.get("p_otter_eat")) is True}
    ok["pre_abstained"] = _abstained(resp.get("pre_bear_chase")) is True
    ok["builtin_dog_cat"] = _patient(resp.get("c_dog_chase")) == "cat"
    ok["all"] = all(ok.values())
    ok["anaphora_it_eat"] = _patient(resp.get("p_it_eat")) == ("otter" if permuted else "clam")   # reported only
    ok["anaphora_reply_patient"] = _patient(resp.get("p_it_eat"))    # the patient the reply named (cat/fish on dev)
    return ok


def score_lived(resp, permuted=False):
    """C3. A fact told this session is recalled after three intervening turns; the episode is disclosed."""
    want = "pear" if permuted else "apple"
    ok = {"pre_abstained": _abstained(resp.get("pre_wolf_bite")) is True,
          "pre_not_in_memory": _abstained(resp.get("pre_wolf_discuss")) is True,
          "recall": _patient(resp.get("p_wolf_bite")) == want,
          "disclosure": _abstained(resp.get("p_wolf_discuss")) is False,
          "moat_wolf_chase": _abstained(resp.get("c_wolf_chase")) is True,
          "builtin_dog_cat": _patient(resp.get("c_dog_chase")) == "cat"}
    ok["in_memory_internal"] = _episodic_in_memory(resp.get("p_wolf_discuss"))
    ok["all"] = all(v for k, v in ok.items() if k != "in_memory_internal")
    return ok


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None          # zero variance (e.g. every read neutral) -> UNDEFINED, never 0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def score_val_heldout(resp):
    """C4a. Pearson r between the differential PRINTED in the feel reply after each held-out word and that word's
    Warriner norm. UNDEFINED (None) with a reason when too few reads parse or the reads have no variance."""
    xs, ys, rows = [], [], []
    for w, norm in HELDOUT_WORDS:
        f = parse_feel(((resp.get("%s_feel" % w) or {}).get("answer")))
        rows.append({"word": w, "norm": norm, "reply_differential": f["differential"], "reply_sign": f["sign"]})
        if f["differential"] is not None:
            xs.append(f["differential"]); ys.append(norm)
    n = len(xs)
    r = _pearson(xs, ys) if n >= MIN_WORDS_FOR_R else None
    r = None if r is None else round(r, 3)      # stored at the precision the report prints (claim_check reads it)
    reason = None
    if n < MIN_WORDS_FOR_R:
        reason = "only %d of %d feel reads parsed" % (n, len(HELDOUT_WORDS))
    elif r is None:
        reason = "no variance in the reply differentials (every read neutral)"
    sign_agree = [((x > MOOD_TOL and y >= 5) or (x < -MOOD_TOL and y < 5)) for x, y in zip(xs, ys)]
    return {"n_parsed": n, "r": r, "undefined_reason": reason, "rows": rows,
            "sign_agreement": (sum(sign_agree) / float(len(sign_agree))) if sign_agree else None,
            "n_neutral_reads": sum(1 for x in xs if abs(x) <= MOOD_TOL)}


def score_val_novel(resp, permuted=False):
    """C4b. After each novel word is presented ALONE, the reply-visible differential must move toward the valence of
    the company it kept, against the mood the other word left. Signs are read from the reply text."""
    pos_probe, neg_probe = ("feel_x_pos", "feel_x_neg")
    before_pos, before_neg = parse_feel((resp.get("feel_neg") or {}).get("answer")), parse_feel((resp.get("feel_pos2") or {}).get("answer"))
    at_pos, at_neg = parse_feel((resp.get(pos_probe) or {}).get("answer")), parse_feel((resp.get(neg_probe) or {}).get("answer"))
    want_pos, want_neg = ("negative", "positive") if permuted else ("positive", "negative")
    pipeline = {"company_pos": parse_feel((resp.get("feel_pos") or {}).get("answer"))["sign"] == want_pos,
                "company_neg": parse_feel((resp.get("feel_neg") or {}).get("answer"))["sign"] == want_neg}
    d_pos = (at_pos["differential"] - before_pos["differential"]) if None not in (at_pos["differential"], before_pos["differential"]) else None
    d_neg = (at_neg["differential"] - before_neg["differential"]) if None not in (at_neg["differential"], before_neg["differential"]) else None
    # in the permuted arm the SAME probe words kept the OPPOSITE company, so the expected moves reverse
    s = -1.0 if permuted else 1.0
    return {"pipeline_carries_company": all(pipeline.values()), "delta_at_pos_word": d_pos, "delta_at_neg_word": d_neg,
            "sign_at_pos_word": at_pos["sign"], "sign_at_neg_word": at_neg["sign"],
            "inherits": (d_pos is not None and d_neg is not None and s * d_pos > MOOD_TOL and s * d_neg < -MOOD_TOL)}


def _load_shards(root, seeds):
    out = {}
    for seed in seeds:
        for group in GROUPS:
            for arm in arms_for(group):
                p = shard_path(root, seed, group, arm)
                if os.path.exists(p):
                    try:
                        with open(p) as f:
                            out[(int(seed), group, arm)] = json.load(f)
                    except Exception as e:
                        out[(int(seed), group, arm)] = {"_load_error": repr(e)}
    return out


def _labels(group, seed, arm="intact_a"):
    return [t[0] for t in turns_for(group, seed, arm)]


def _scored_labels(group):
    return {"cat_disc": ["p_canary_eat", "p_canary_might"], "cat_told": ["p_canary_eat", "p_canary_might"],
            "word": ["p_bear_chase", "p_otter_eat"],
            "lived": {"lesion_recall": ["p_wolf_bite"], "lesion_episodic": ["p_wolf_discuss"]},
            "val_heldout": None, "val_novel": None}[group]


def aggregate(root, seeds=GATE_SEEDS, informational=True):
    shards = _load_shards(root, seeds)
    seeds = [int(s) for s in seeds]
    per_group = {}
    missing = []
    for group in GROUPS:
        g = GROUPS[group]
        rows, defined = {}, []
        for seed in seeds:
            need = [a for a in arms_for(group, with_informational=False)]
            have = {a: shards.get((seed, group, a)) for a in arms_for(group)}
            for a in need:
                if have[a] is None:
                    missing.append(shard_path(root, seed, group, a))
            row = {"seed": seed, "undefined_reasons": []}
            ia, ib = have.get("intact_a"), have.get("intact_b")
            if ia is None or ib is None or "_load_error" in (ia or {}) or "_load_error" in (ib or {}):
                row["undefined_reasons"].append("intact arm missing")
            else:
                labels = _labels(group, seed)
                ok, diffs = null_clean(ia["responses"], ib["responses"], labels)
                row["null_clean"], row["null_diffs"] = ok, diffs[:8]
                if not ok:
                    row["undefined_reasons"].append("null failed (intact_a != intact_b) on %d turn(s)" % len(diffs))
                srcs = set(ia.get("sources") or []) | set(ib.get("sources") or [])
                row["sources"] = sorted(srcs)
                if srcs and srcs != {EXPECTED_SOURCE}:
                    row["undefined_reasons"].append("unexpected brain source %s" % sorted(srcs))
                if not (ia.get("pythonhashseed_ok") and ib.get("pythonhashseed_ok")):
                    row["undefined_reasons"].append("PYTHONHASHSEED not 0")
                if not (ia.get("corpus_present", True) and ib.get("corpus_present", True)):
                    row["undefined_reasons"].append("corpus absent: corpus-learned production organs degraded")
                # every COUNTED arm must be present and readable: a missing lesion / repeat / permuted arm is an
                # instrument failure, so the seed is UNDEFINED for this group (never a FAIL clause)
                for les in g["lesions"]:
                    for arm_name in (les, les + ".rep0"):
                        sh = have.get(arm_name)
                        if sh is None or "responses" not in sh:
                            row["undefined_reasons"].append("%s arm missing or unreadable" % arm_name)
                if have.get("permuted") is None or "responses" not in (have.get("permuted") or {}):
                    row["undefined_reasons"].append("permuted arm missing or unreadable")
                ra = ia["responses"]
                # ── correctness (intact_a) ──
                if group in ("cat_disc", "cat_told"):
                    row["intact"] = score_cat(ra)
                    row["correct"] = row["intact"]["inherits"]
                elif group == "word":
                    row["intact"] = score_word(ra); row["correct"] = row["intact"]["all"]
                elif group == "lived":
                    row["intact"] = score_lived(ra); row["correct"] = row["intact"]["all"]
                elif group == "val_heldout":
                    row["intact"] = score_val_heldout(ra)
                    r = row["intact"]["r"]
                    row["correct"] = (r is not None and r >= VALENCE_R_BAR)
                    if r is None:
                        row["undefined_reasons"].append("r UNDEFINED: %s" % row["intact"]["undefined_reason"])
                elif group == "val_novel":
                    row["intact"] = score_val_novel(ra); row["correct"] = row["intact"]["inherits"]
                # ── lesion: the reply-level effect must be present AND reproduced ──
                row["lesions"] = {}
                for les in g["lesions"]:
                    L, R = have.get(les), have.get(les + ".rep0")
                    rec = {"present": L is not None, "repeat_present": R is not None}
                    if L is not None and "responses" in L:
                        # ATTRIBUTION (tools.lab): whose is the intact-vs-lesion difference? treatment = the effect
                        # size under the lesion; control = the same measure between the two INTACT builds (the null).
                        # 1.0 = none of it is run-to-run variation; None = no effect in either arm (UNDEFINED).
                        rb = ib["responses"]
                        if group == "val_heldout":
                            sl = score_val_heldout(L["responses"]); rec["r_lesion"] = sl["r"]
                            rec["effect"] = (sl["r"] is None) or (abs(sl["r"]) < CONTROL_R_CEILING)
                            rec["reason"] = sl["undefined_reason"]
                            r_a, r_b = (row["intact"] or {}).get("r"), score_val_heldout(rb)["r"]
                            rec["attributable"] = (attributable_to("%s/%s seed %d (|r_intact - r_lesion| vs |r_a - r_b|)" % (group, les, seed),
                                                                   abs(r_a - (sl["r"] or 0.0)), abs(r_a - (r_b if r_b is not None else r_a)))
                                                   if r_a is not None else None)
                        elif group == "val_novel":
                            sl = score_val_novel(L["responses"]); rec["lesion_inherits"] = sl["inherits"]
                            rec["effect"] = (row["correct"] and not sl["inherits"]) or (not row["correct"] and None)
                            labs = ["feel_x_pos", "feel_x_neg"]
                            rec["attributable"] = attributable_to("%s/%s seed %d (reply fields on the two probe reads)" % (group, les, seed),
                                                                  float(n_field_diffs(ra, L["responses"], labs)), float(n_field_diffs(ra, rb, labs)))
                        else:
                            labs = _scored_labels(group)
                            if isinstance(labs, dict):
                                labs = labs[les]
                            d = [reply_differs(ra, L["responses"], l) for l in labs]
                            rec["differs"] = dict(zip(labs, d)); rec["effect"] = any(x is True for x in d)
                            rec["attributable"] = attributable_to("%s/%s seed %d (primary reply fields on %s)" % (group, les, seed, ",".join(labs)),
                                                                  float(n_field_diffs(ra, L["responses"], labs)), float(n_field_diffs(ra, rb, labs)))
                        if R is not None and "responses" in R:
                            labs2 = _labels(group, seed)
                            rec["reproduced"], _ = null_clean(L["responses"], R["responses"], labs2)
                        else:
                            rec["reproduced"] = None
                        p = L.get("patch")
                        if isinstance(p, dict) and "calls" in p:
                            rec["patch_calls"] = p["calls"]; rec["applied"] = p["calls"] > 0
                        else:
                            rec["applied"] = True      # env-flag lesions are read per call by production code
                    else:
                        rec["effect"] = None; rec["reproduced"] = None
                    row["lesions"][les] = rec
                # ── permuted exposure: the reply must follow the exposure ──
                P = have.get("permuted")
                if P is not None and "responses" in P:
                    if group in ("cat_disc", "cat_told"):
                        sp = score_cat(P["responses"], permuted=True); row["permuted"] = sp
                        # tracking is only DEFINED when the intact arm inherits at all (nothing to follow otherwise)
                        row["tracks"] = sp["inherits"] if row["correct"] else None
                    elif group == "word":
                        sp = score_word(P["responses"], permuted=True); row["permuted"] = sp; row["tracks"] = sp["all"]
                    elif group == "lived":
                        sp = score_lived(P["responses"], permuted=True); row["permuted"] = sp; row["tracks"] = sp["all"]
                    elif group == "val_heldout":
                        sp = score_val_heldout(P["responses"]); row["permuted"] = {"r": sp["r"], "reason": sp["undefined_reason"]}
                        row["tracks"] = (sp["r"] is None) or (sp["r"] < CONTROL_R_CEILING)
                    elif group == "val_novel":
                        sp = score_val_novel(P["responses"], permuted=True); row["permuted"] = sp; row["tracks"] = sp["inherits"]
                else:
                    row["tracks"] = None
                # ── informational arms (never counted) ──
                if informational:
                    row["informational"] = {}
                    for inf in g["informational"]:
                        I = have.get(inf)
                        if I is not None and "responses" in I:
                            if group == "val_heldout":
                                row["informational"][inf] = {"r": score_val_heldout(I["responses"])["r"]}
                            elif group == "val_novel":
                                row["informational"][inf] = {"inherits": score_val_novel(I["responses"])["inherits"]}
            rows[seed] = row
            if not row["undefined_reasons"] and "correct" in row:
                defined.append(seed)
        n_def = len(defined)
        n_correct = sum(1 for s in defined if rows[s]["correct"])
        les_names = list(g["lesions"])
        n_les = {les: sum(1 for s in defined if (rows[s]["lesions"].get(les) or {}).get("effect") is True) for les in les_names}
        n_rep = {les: sum(1 for s in defined if (rows[s]["lesions"].get(les) or {}).get("reproduced") is True) for les in les_names}
        n_tracks = sum(1 for s in defined if rows[s].get("tracks") is True)
        n_tr_def = sum(1 for s in defined if rows[s].get("tracks") is not None)
        verdict = "INCOMPLETE" if n_def < len(seeds) else (
            "PASS" if (n_correct == n_def and all(n_les[l] == n_def for l in les_names)
                       and all(n_rep[l] == n_def for l in les_names) and n_tr_def == n_def and n_tracks == n_def)
            else "FAIL")
        why = []
        if verdict == "FAIL":
            if n_correct < n_def:
                why.append("capability absent on %d of %d seeds" % (n_def - n_correct, n_def))
            else:   # the capability is present on every seed: the lesion / exposure clauses are what failed
                for l in les_names:
                    if n_les[l] < n_def:
                        why.append("%s: reply-level effect on %d of %d seeds (not load-bearing)" % (l, n_les[l], n_def))
                    if n_rep[l] < n_def:
                        why.append("%s: reproduced on %d of %d seeds" % (l, n_rep[l], n_def))
                if n_tracks < n_tr_def:
                    why.append("permuted exposure tracked on %d of %d seeds (not exposure-driven)" % (n_tracks, n_tr_def))
                elif n_tr_def < n_def:
                    why.append("permuted tracking undefined on %d of %d seeds" % (n_def - n_tr_def, n_def))
        elif verdict == "INCOMPLETE":
            why.append("%d of %d seeds defined" % (n_def, len(seeds)))
        per_group[group] = {"criterion": g["criterion"], "primary": g["primary"], "verdict": verdict, "why": why,
                            "n_defined": n_def, "n_correct": n_correct, "n_lesion_effect": n_les,
                            "n_reproduced": n_rep, "n_permuted_tracks": n_tracks, "seeds": rows}
    crit = {}
    for c in ("C1", "C2", "C3", "C4"):
        prim = [g for g in PRIMARY_GROUPS if GROUPS[g]["criterion"] == c][0]
        crit[c] = per_group[prim]["verdict"]
    # ── the verdict is EARNED (tools.verdict): every validity precondition measured and holding, else UNDEFINED ──
    from tools.verdict import Verdict
    prim_rows = [per_group[g]["seeds"][s] for g in PRIMARY_GROUPS for s in seeds]
    v = Verdict("Stage-1 (INFANT) exit on the integrated production brain")
    v.require("no primary shard missing", sum(1 for m in missing if any("/%s__" % g in m for g in PRIMARY_GROUPS)), expect=0)
    v.require("null clean (intact_a == intact_b) on every primary (group, seed)",
              sum(1 for r in prim_rows if r.get("null_clean") is not True), expect=0,
              note="a failed null makes that seed UNDEFINED; the criterion then cannot reach 6 defined seeds")
    v.require("every response from the expected brain", sum(1 for r in prim_rows if r.get("sources") not in (None, [EXPECTED_SOURCE])), expect=0,
              note="expected `source` = %s" % EXPECTED_SOURCE)
    v.require("PYTHONHASHSEED=0 in every primary arm", sum(1 for r in prim_rows if "PYTHONHASHSEED not 0" in r.get("undefined_reasons", [])), expect=0)
    v.require("the corpus-learned production organs had their corpus (data/corpus/tinystories.txt) in every primary arm",
              sum(1 for r in prim_rows if any("corpus absent" in x for x in r.get("undefined_reasons", []))), expect=0,
              note="a worktree or pool checkout without the corpus builds a DEGRADED brain; pool_provision.sh syncs it")
    v.require("every in-process lesion applied (patch called at least once)",
              sum(1 for r in prim_rows for les in (r.get("lesions") or {}).values() if les.get("applied") is False), expect=0)
    v.require("six seeds defined for every primary criterion", sum(1 for g in PRIMARY_GROUPS if per_group[g]["n_defined"] < len(seeds)), expect=0)
    earned = v.decide(go=all(x == "PASS" for x in crit.values()), verbose=False)
    if earned["status"] == "UNDEFINED":
        stage = "UNDEFINED (INCOMPLETE): " + "; ".join(earned["undefined_reasons"])[:400]
    else:
        stage = "PASS" if earned["go"] else "FAIL"
    return {"stage": "STAGE-1 (INFANT) exit", "verdict": stage, "GO": bool(earned["go"]),
            "preconditions": earned["preconditions"], "undefined_reasons": earned["undefined_reasons"],
            "criteria": crit, "groups": per_group,
            "seeds": seeds, "missing_shards": missing, "bars": {"valence_r": VALENCE_R_BAR, "control_r_ceiling": CONTROL_R_CEILING,
                                                                 "min_words_for_r": MIN_WORDS_FOR_R, "seeds_required": len(seeds)},
            "runner": "research/runners/stage1_exit_gate.py", "prereg": PREREG, "expected_source": EXPECTED_SOURCE}


def report(agg):
    lines = ["**Stage-1 exit verdict: %s** (primary criteria: %s)" % (
        agg["verdict"], ", ".join("%s %s" % (k, v) for k, v in agg["criteria"].items())), "",
        "| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |",
        "|---|---|---|---|---|---|---|---|---|"]
    for g, d in agg["groups"].items():
        lines.append("| %s | %s | %s | %s | %d/%d | %s | %s | %d/%d | %s |" % (
            g, d["criterion"], "primary" if d["primary"] else "diagnostic", d["verdict"], d["n_correct"], d["n_defined"],
            ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["n_lesion_effect"].items()),
            ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["n_reproduced"].items()),
            d["n_permuted_tracks"], d["n_defined"], "; ".join(d["why"]) or "-"))
    vh = agg["groups"]["val_heldout"]["seeds"]
    lines += ["", "Per-seed valence r (reply-visible differential vs Warriner norm, %d held-out words):" % len(HELDOUT_WORDS)]
    for s, row in vh.items():
        it = row.get("intact") or {}
        lines.append("- seed %s: r = %s (n = %s, sign agreement %s); lesion r = %s; permuted-map r = %s; %s" % (
            s, ("%.3f" % it["r"]) if it.get("r") is not None else "UNDEFINED", it.get("n_parsed"),
            ("%.2f" % it["sign_agreement"]) if it.get("sign_agreement") is not None else "-",
            (row.get("lesions") or {}).get("lesion_affect", {}).get("r_lesion"),
            (row.get("permuted") or {}).get("r"), "; ".join(row.get("undefined_reasons") or []) or "defined"))
    if agg["missing_shards"]:
        lines += ["", "Missing shards: %d" % len(agg["missing_shards"])]
    return "\n".join(lines)


# ══════════════════════════════════════════════ POOL JOBS ═══════════════════════════════════════════════════════════
def shard_path(out_dir, seed, group, arm):
    return os.path.join(out_dir, "s%d" % int(seed), "%s__%s.json" % (group, arm.replace(":", "_")))


def pool_job_lines(sha, seeds=GATE_SEEDS, out_dir=GATE_OUT_DIR, informational=True):
    """One pool job per (seed, group, arm). A tiny-demo+LTM build is ~6.3-6.9 GB RSS -> mem_gb=7, cap 9 GB."""
    out = []
    for seed in seeds:
        for group in GROUPS:
            for arm in arms_for(group, with_informational=informational):
                p = shard_path(out_dir, seed, group, arm)
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 7 1 && mkdir -p %s && "
                       "bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                       "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                       "research.runners.stage1_exit_gate --shard --seed %d --group %s --arm %s --allow-gate-seed "
                       "--prereg %s --out %s" % (sha, os.path.dirname(p), seed, group, arm, PREREG, p))
                why = ("prereg %s (%s); Stage-1 exit gate seed %d %s %s; record: first integrated stage-exit "
                       "measurement (before_you_build + RAG 2026-09-29: only isolated de-risks exist); mem_gb=7"
                       % (PREREG, sha[:9], seed, group, arm))
                out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ SELFTEST (no brain) ════════════════════════════════════════════════
def _fake(answer, abstained=False, svo=None, hyp=None, extra=None):
    d = {"answer": answer, "abstained": abstained, "recalled_svo": svo, "source": EXPECTED_SOURCE}
    if hyp:
        d["hypothesis"] = True; d["hypothesis_svo"] = hyp
    if extra:
        d.update(extra)
    return d


def _synthetic_word(kind):
    """A synthetic `word` session: intact / lesioned (new word unbound) / permuted-consistent."""
    ab = _fake("I don't know about that.", True)
    r = {"pre_bear_chase": ab, "t_otter_eat": _fake("the otter eats the clam", svo=["otter", "eat", "clam"]),
         "t_bear_chase": _fake("the bear chases the otter", svo=["bear", "chase", "otter"]),
         "c_dog_chase": _fake("the dog chases the cat", svo=["dog", "chase", "cat"])}
    if kind == "intact":
        r.update({"p_bear_chase": _fake("the bear chases the otter", svo=["bear", "chase", "otter"]),
                  "p_it_eat": _fake("the otter eats the clam", svo=["otter", "eat", "clam"]),
                  "p_otter_eat": _fake("the otter eats the clam", svo=["otter", "eat", "clam"]),
                  "c_otter_chase": ab, "c_clam_eat": ab})
    elif kind == "lesion":
        r.update({"p_bear_chase": ab, "p_it_eat": ab, "p_otter_eat": ab, "c_otter_chase": ab, "c_clam_eat": ab})
    else:
        r.update({"p_bear_chase": ab, "p_it_eat": ab, "p_otter_eat": ab,
                  "c_otter_chase": _fake("the otter chases the bear", svo=["otter", "chase", "bear"]),
                  "c_clam_eat": _fake("the clam eats the otter", svo=["clam", "eat", "otter"])})
    return r


def _synthetic_feel(d):
    mood = "positive" if d > MOOD_TOL else ("negative" if d < -MOOD_TOL else "neutral")
    return _fake("My affect monitor reads %s right now (valence differential %+.3f, moderate arousal) -- that's a "
                 "functional read of my mood state, not a felt experience." % (mood, d))


def _synthetic_val(kind, seed):
    r = {"feel_base": _synthetic_feel(0.0)}
    import random
    rng = random.Random(seed)
    for w, norm in HELDOUT_WORDS:
        if kind == "intact":
            d = (norm - 5.0) / 4.0 * 0.5 + rng.uniform(-0.05, 0.05)
        elif kind == "lesion":
            d = 0.0
        else:
            d = rng.uniform(-0.4, 0.4)
        r["%s_e1" % w] = _fake("I don't know about that.", True); r["%s_e2" % w] = r["%s_e1" % w]
        r["%s_feel" % w] = _synthetic_feel(d)
    return r


def selftest(verbose=True):
    import tempfile
    checks = {}
    checks["every group has intact_a, intact_b, >=1 lesion (+rep0), permuted"] = all(
        arms_for(g)[:2] == ["intact_a", "intact_b"] and "permuted" in arms_for(g)
        and any(a.endswith(".rep0") for a in arms_for(g)) for g in GROUPS)
    checks["exactly one primary group per criterion"] = sorted(GROUPS[g]["criterion"] for g in PRIMARY_GROUPS) == ["C1", "C2", "C3", "C4"]
    checks["held-out words: 24, 12 per sign, all strong (|v-5|>=2 or the documented 'nice' edge)"] = (
        len(HELDOUT_WORDS) == 24 and sum(1 for _w, v in HELDOUT_WORDS if v >= 5) == 12
        and all(abs(v - 5) >= 1.4 for _w, v in HELDOUT_WORDS))
    checks["val_heldout turn order differs by seed, same word set"] = (
        [t[0] for t in _val_heldout_turns(42)] != [t[0] for t in _val_heldout_turns(43)]
        and sorted(t[0] for t in _val_heldout_turns(42)) == sorted(t[0] for t in _val_heldout_turns(43)))
    checks["permuted exposure differs from intact only on teach turns (categorical groups)"] = all(
        [(l, r) for l, _m, r in GROUPS[g]["turns"]] == [(l, r) for l, _m, r in GROUPS[g]["permuted_turns"]]
        and any(m1 != m2 for (_l, m1, _r), (_l2, m2, _r2) in zip(GROUPS[g]["turns"], GROUPS[g]["permuted_turns"]))
        and all(r == "teach" or r == "expose" for (_l, m1, r), (_l2, m2, _r2) in zip(GROUPS[g]["turns"], GROUPS[g]["permuted_turns"]) if m1 != m2)
        for g in ("cat_disc", "cat_told", "word", "lived", "val_novel"))
    checks["novel-word exposures are never 3-content-word assertions"] = all(
        len([w for w in m.split() if w not in ("the", "is", "and")]) != 3 for _l, m, r in GROUPS["val_novel"]["turns"] if r == "expose")
    checks["feel parser reads sign + differential from the production read-out text"] = (
        parse_feel(_synthetic_feel(0.123)["answer"]) == {"differential": 0.123, "sign": "positive"}
        and parse_feel("I don't know about that.") == {"differential": None, "sign": None})
    checks["word scorer: intact all-correct, lesion fails, permuted tracks"] = (
        score_word(_synthetic_word("intact"))["all"] and not score_word(_synthetic_word("lesion"))["all"]
        and score_word(_synthetic_word("permuted"), permuted=True)["all"])
    sv_i, sv_l, sv_p = score_val_heldout(_synthetic_val("intact", 1)), score_val_heldout(_synthetic_val("lesion", 1)), score_val_heldout(_synthetic_val("permuted", 1))
    checks["valence scorer: graded intact r >= bar; all-neutral lesion r UNDEFINED (not 0); permuted below ceiling"] = (
        sv_i["r"] is not None and sv_i["r"] >= VALENCE_R_BAR and sv_l["r"] is None and "variance" in (sv_l["undefined_reason"] or "")
        and sv_p["r"] is not None and abs(sv_p["r"]) < 0.5)
    checks["pearson: zero variance is UNDEFINED, not a number"] = _pearson([1.0, 1.0, 1.0], [1, 2, 3]) is None
    checks["null: identical arms clean; a differing primary field is caught"] = (
        null_clean(_synthetic_word("intact"), _synthetic_word("intact"), ["p_bear_chase"])[0]
        and not null_clean(_synthetic_word("intact"), _synthetic_word("lesion"), ["p_bear_chase"])[0])
    # end-to-end aggregate on a synthetic shard tree: PASS for word (all arms), INCOMPLETE elsewhere (missing shards)
    with tempfile.TemporaryDirectory() as td:
        seeds = (1, 2)
        for seed in seeds:
            kinds = {"intact_a": "intact", "intact_b": "intact", "lesion_recruit": "lesion", "lesion_recruit.rep0": "lesion", "permuted": "permuted"}
            for arm, kind in kinds.items():
                p = shard_path(td, seed, "word", arm); os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump({"seed": seed, "group": "word", "arm": arm, "responses": _synthetic_word(kind), "sources": [EXPECTED_SOURCE],
                           "pythonhashseed_ok": True, "patch": {"patch": "lesion_recruit", "calls": 3} if "lesion" in arm else None},
                          open(p, "w"))
        agg = aggregate(td, seeds=seeds)
        checks["aggregate: synthetic word group PASSES 2/2; the stage is UNDEFINED while other groups are missing"] = (
            agg["groups"]["word"]["verdict"] == "PASS" and agg["verdict"].startswith("UNDEFINED") and len(agg["missing_shards"]) > 0
            and agg["GO"] is False and len(agg["preconditions"]) >= 5 and all(c["ok"] is not None for c in agg["preconditions"]))
        # a lesion arm with NO effect -> the group FAILs as 'not load-bearing' even though intact is correct
        for seed in seeds:
            for arm in ("lesion_recruit", "lesion_recruit.rep0"):
                p = shard_path(td, seed, "word", arm)
                json.dump({"seed": seed, "group": "word", "arm": arm, "responses": _synthetic_word("intact"), "sources": [EXPECTED_SOURCE],
                           "pythonhashseed_ok": True, "patch": {"patch": "lesion_recruit", "calls": 3}}, open(p, "w"))
        agg2 = aggregate(td, seeds=seeds)
        checks["aggregate: correct-but-not-load-bearing reads FAIL with the reason named"] = (
            agg2["groups"]["word"]["verdict"] == "FAIL" and any("not load-bearing" in w for w in agg2["groups"]["word"]["why"]))
        checks["attribution: a genuine lesion effect with a clean null is 100% attributable; no effect -> UNDEFINED (None)"] = (
            agg["groups"]["word"]["seeds"][1]["lesions"]["lesion_recruit"]["attributable"] == 1.0
            and agg2["groups"]["word"]["seeds"][1]["lesions"]["lesion_recruit"]["attributable"] is None)
        # a MISSING permuted arm makes the seed UNDEFINED (an instrument failure), never a FAIL clause
        os.remove(shard_path(td, 2, "word", "permuted"))
        agg2b = aggregate(td, seeds=seeds)
        checks["aggregate: a missing counted arm -> that seed UNDEFINED (INCOMPLETE), not FAIL"] = (
            agg2b["groups"]["word"]["verdict"] == "INCOMPLETE" and agg2b["groups"]["word"]["n_defined"] == 1
            and any("permuted arm missing" in r for r in agg2b["groups"]["word"]["seeds"][2]["undefined_reasons"]))
        json.dump({"seed": 2, "group": "word", "arm": "permuted", "responses": _synthetic_word("permuted"), "sources": [EXPECTED_SOURCE],
                   "pythonhashseed_ok": True}, open(shard_path(td, 2, "word", "permuted"), "w"))
        # a failed null makes the seed UNDEFINED -> INCOMPLETE, never a pass or a fail
        json.dump({"seed": 1, "group": "word", "arm": "intact_b", "responses": _synthetic_word("lesion"), "sources": [EXPECTED_SOURCE],
                   "pythonhashseed_ok": True}, open(shard_path(td, 1, "word", "intact_b"), "w"))
        agg3 = aggregate(td, seeds=seeds)
        checks["aggregate: a failed null -> that seed UNDEFINED -> group INCOMPLETE"] = (
            agg3["groups"]["word"]["verdict"] == "INCOMPLETE" and agg3["groups"]["word"]["n_defined"] == 1)
        checks["report renders"] = "Stage-1 exit verdict" in report(agg3)
    lines = pool_job_lines("0123456789abcdef" * 2 + "01234567")
    checks["pool jobs: one per (seed, group, arm), pinned, memcapped, prereg-tagged, gate flag present"] = (
        len(lines) == len(GATE_SEEDS) * sum(len(arms_for(g)) for g in GROUPS)
        and all("memcap.sh 9" in c and "--allow-gate-seed" in c and PREREG in c and "revisions/0123456789" in c for c, _w in lines))
    ok = all(checks.values())
    if verbose:
        for k, v in checks.items():
            print(("PASS " if v else "FAIL ") + k)
        print("SELFTEST", "OK" if ok else "FAILED")
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true", help="internal: build one arm in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, group, arm) arm in a fresh subprocess")
    ap.add_argument("--aggregate", default=None, help="shard directory to aggregate")
    ap.add_argument("--report", default=None, metavar="AGGREGATE_JSON", help="print the fixed Markdown result block")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the gate's pool job lines (TSV)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --aggregate / --emit-pool-jobs")
    ap.add_argument("--group", default=None, choices=sorted(GROUPS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="DEV ONLY: run only these turn labels (comma-separated)")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    ap.add_argument("--prereg", default=None, help="the committed preregistration a GATE-seed run is governed by")
    ap.add_argument("--allow-gate-seed", action="store_true",
                    help="required to run a GATE seed (42/43/44/100/101/102); dev seeds need nothing")
    ap.add_argument("--no-informational", action="store_true", help="emit/aggregate without the informational arms")
    a = ap.parse_args()
    if a.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    if a.emit_pool_jobs:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        for cmd, why in pool_job_lines(a.emit_pool_jobs, seeds=seeds, informational=not a.no_informational):
            print("%s\t%s" % (cmd, why))
        return
    if a.aggregate:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        with contextlib.redirect_stdout(sys.stderr):      # the attribution/verdict prints go to stderr: stdout is
            agg = aggregate(a.aggregate, seeds=seeds, informational=not a.no_informational)   # the report only
        out = a.out or os.path.join(a.aggregate, "aggregate.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(agg, f, indent=1, default=str)
        print(report(agg)); print("\n-> %s" % out)
        return
    if a.report:
        with open(a.report) as f:
            print(report(json.load(f)))
        return
    if a.worker or a.shard:
        if a.seed is None or not a.group or not a.arm or not a.out:
            ap.error("--seed, --group, --arm and --out are required")
        if a.seed in GATE_SEEDS and not a.allow_gate_seed:
            sys.stderr.write("GATE seed %d refused: a gate seed runs only from a pool job emitted against the committed "
                             "prereg (%s); pass --allow-gate-seed --prereg <path> from such a job. Dev seeds: 1-12.\n"
                             % (a.seed, PREREG))
            sys.exit(2)
        if a.worker:
            sys.exit(run_worker(a.seed, a.group, a.arm, a.out, labels=a.labels))
        sys.exit(run_shard(a.seed, a.group, a.arm, a.out, backend=a.backend, labels=a.labels,
                           allow_gate_seed=a.allow_gate_seed, prereg=a.prereg))
    ap.error("choose one of --worker/--shard/--aggregate/--report/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    main()
