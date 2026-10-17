"""Shared builder for the Stage-1 C1 / C2 / C3 bar-sanity specs (tools/bar_sanity.py); C4 has its own
(bar_sanity_c4_common.py, RULING s1-c4-control).

The decision is the REAL aggregate: `research.runners.stage1_exit_gate.aggregate` run over synthetic six-seed shards in the
shard format the workers write (`responses`, `patch`, `sources`, `pythonhashseed_ok`, `corpus_present`), reading one group's
verdict (`cat_disc` = C1, `word` = C2, `lived` = C3). The correctness checkers (`score_cat`, `score_word`, `score_lived`),
the lesion-effect / reproduction / permuted-tracking clauses, the Amendment-A1 rule (an env-flag lesion with no reply change
and no engagement record makes the seed UNDEFINED) and the null-clean rule are the scorer's own. The only glue is that the
shards reach `aggregate` from memory (`_load_shards` swapped for the call); `selfcheck()` proves the in-memory and the
on-disk paths return the identical aggregate.

REALISM. Reply surfaces are copied from the committed attempt-1 shards (research/findings/raw/_stage1_exit/gate/s42/,
.../gate_c3rerun/s42/): the discourse leads ("Setting the held thread aside -- On <x>, then -- ", "As for it -- "), the teach
echo with the built-in frames appended, the curiosity question on an abstain, the D5 disclosure line with its completion value
("dendritic dAP completion 1.00") and the "I don't recall" line. Stage 1's reply bars are exact string/field checks on a
deterministic brain (the real null is exactly zero-noise: gate/aggregate.json reads word 6/6 defined with the null clean,
gate_c3rerun lived 6/6), so there is no read noise to model; the realistic part is the surface the checkers must survive.
EXISTENCE PROOF (not a generator): re-scored with this runner, the committed REAL attempt-1 data read C2 `word` PASS 6/6
(gate/aggregate.json) and C3 `lived` PASS 6/6 (gate_c3rerun/aggregate.json): the real brain passes both bars.

C1's brains. The registered C1 lesion is `lesion_draw` (BRAIN_SPIKING_DRAW_LESION, the generative DRAW's likelihood); the
registered correctness is "either route names `worm`". Two perfect brains exist for it:
  gen      the inheritance rides the generative route (`what might the canary eat` -> a flagged hypothesis `worm`), the one the
           prereg names; the draw lesion removes the hypothesis;
  pathway  the inheritance rides a NEW route (docs/drafts/stage1-c1-inheritance/PLAN.md: a learned agent->context pathway read
           on the strict route's recall miss, flagged as an inference), which is what attempt 2 will run; the DRAW is not on
           its path, so `lesion_draw` changes nothing.
Amendment A2 (2026-10-13) makes `lesion_inherit_pathway` the runner's own counted C1 lesion (`lesion_draw` informational)
and adds the C1 validity record: every arm records the effective flags and the direct probe's organ read records
`hop: true` (`_c1_record`). The as-registered spec re-installs `lesion_draw` as the counted arm to keep measuring the old bar.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import stage1_exit_gate as G  # noqa: E402

S = G.EXPECTED_SOURCE
SEEDS = list(G.GATE_SEEDS)
AB = "I don't know about that."
GROUP_OF = {"c1": "cat_disc", "c2": "word", "c3": "lived"}


def ask_text(x):
    return " My curiosity is piqued — I haven't learned about %s yet: what can you tell me about %s?" % (x, x)


def R(answer, svo=None, abst=False, hyp=None, **extra):
    d = {"answer": answer, "abstained": bool(abst), "recalled_svo": list(svo) if svo else None, "source": S}
    if hyp:
        d["hypothesis"] = True
        d["hypothesis_svo"] = list(hyp)
    d.update(extra)
    return d


def fx(a, v, p, prep=None):
    verb = {"eat": "eats", "chase": "chases", "bite": "bites", "fly": "flies", "swim": "swims", "isa": "isas"}[v]
    return "the %s %s %sthe %s" % (a, verb, (prep + " ") if prep else "", p)


def deco(rng, subj, core, realistic=True):
    if not realistic:
        return core
    r = rng.random()
    lead = ("" if r < 0.25 else "As for it — " if r < 0.45 else
            "Setting the held thread aside — On %s, then — " % subj if r < 0.8 else
            "Setting the held thread aside — As for it — On %s, then — " % subj)
    tail = " — worth going further here." if rng.random() < 0.15 else ""
    return lead + core + tail


def recall(rng, a, v, p, realistic=True):
    return R(deco(rng, a, fx(a, v, p), realistic), [a, v, p])


def absent(rng, subj, ask=False, realistic=True):
    t = AB + (ask_text(subj) if ask else "")
    return R(deco(rng, subj, t, realistic) if realistic else t, abst=True)


def _teach(group, perm, rng, realistic):
    """The handler's echo of each teach turn (never scored)."""
    out = {}
    for l, m, r in G.turns_for(group, SEEDS[0], "permuted" if perm else "intact_a"):
        if r in ("teach", "expose"):
            out[l] = R(m + (" the dog chases the cat the cat eats the fish" if realistic and rng.random() < 0.2 else ""))
    return out


# ══════════════════════════════════════════════ C1 (cat_disc) ═════════════════════════════════════════════════════════
def reply_cat_disc(kind, role, rng, realistic):
    """kind: gen | pathway | both | frequency | absent ; role: intact | lesion_draw | lesion_inherit_pathway | permuted |
    lesion_draw_u<k> (info: the lesioned draw picks uniformly among k patients)."""
    perm = role == "permuted"
    r = _teach("cat_disc", perm, rng, realistic)
    want = "bug" if perm else "worm"
    r["c_robin_eat"] = recall(rng, "robin", "eat", "worm", realistic)
    r["c_trout_eat"] = recall(rng, "trout", "eat", "bug", realistic)
    swim = absent(rng, "canary", False, realistic)
    if perm:
        swim = recall(rng, "canary", "swim", "lake", realistic)       # the canary WAS told it swims in the permuted arm
    r["c_canary_swim"] = swim
    gen_abst = R("As for it — " + AB + ask_text("might"), abst=True)
    strict_abst = absent(rng, "canary", False, realistic)
    r["p_canary_eat"], r["p_canary_might"] = strict_abst, gen_abst

    def hyp(p):
        return R("perhaps the canary eats the %s" % p, None, hyp=["canary", "eat", p])

    def probably(p):
        return R("the canary probably eats the %s (an inference, not something I was told)" % p, None, hyp=["canary", "eat", p])
    intact_like = role in ("intact", "permuted")
    if kind == "gen":
        if intact_like:
            r["p_canary_might"] = hyp(want)
        elif role == "lesion_draw":
            pass                                              # the uniform likelihood no longer prefers the category: abstains
        elif role.startswith("lesion_draw_u"):                # info: a uniform draw over k candidates lands on `worm` w.p. 1/k
            k = int(role[len("lesion_draw_u"):])
            r["p_canary_might"] = hyp("worm") if rng.random() < 1.0 / k else hyp("bug")
    elif kind == "pathway":
        if intact_like:
            r["p_canary_eat"] = probably(want)
        elif role == "lesion_inherit_pathway":
            pass                                              # the learned pathway is cut: back to the strict-route miss
        elif role == "lesion_draw":
            r["p_canary_eat"] = probably(want)                # the DRAW is not on this route: nothing changes
    elif kind == "both":
        if intact_like or role in ("lesion_draw",):
            r["p_canary_eat"] = probably(want)
            r["p_canary_might"] = hyp(want) if intact_like else gen_abst
        elif role == "lesion_inherit_pathway":
            r["p_canary_might"] = hyp(want)
    elif kind == "frequency":                                 # a verb->patient frequency shortcut: always the commonest patient
        r["p_canary_eat"] = probably("worm")
        r["p_canary_might"] = hyp("worm")
    return r


def reply_cat_disc_abstain_perm(kind, role, rng, realistic):
    """info: a gen brain that ABSTAINS (instead of naming `bug`) when the canary is streamed with the fish context."""
    r = reply_cat_disc(kind, "intact" if role == "permuted" else role, rng, realistic)
    if role == "permuted":
        r["p_canary_might"] = R("As for it — " + AB + ask_text("might"), abst=True)
        r["c_canary_swim"] = recall(rng, "canary", "swim", "lake", realistic)
    return r


# ══════════════════════════════════════════════ C2 (word) ═════════════════════════════════════════════════════════════
def reply_word(kind, role, rng, realistic):
    """kind: bound | no_preexposure | no_moat | one_role ; role: intact | lesion_recruit | permuted."""
    perm = role == "permuted"
    r = _teach("word", perm, rng, realistic)
    r["pre_bear_chase"] = absent(rng, "bear", True, realistic)
    if kind == "no_preexposure":                                 # the answer before the word was ever taught
        r["pre_bear_chase"] = recall(rng, "bear", "chase", "otter", realistic)
    r["p_it_eat"] = R("the cat eats the fish", ["cat", "eat", "fish"])           # the reported anaphora miss (real)
    r["c_dog_chase"] = recall(rng, "dog", "chase", "cat", realistic)
    if role == "permuted":
        r["p_bear_chase"] = absent(rng, "bear", False, realistic)
        r["p_otter_eat"] = absent(rng, "otter", False, realistic)
        r["c_otter_chase"] = recall(rng, "otter", "chase", "bear", realistic)
        r["c_clam_eat"] = recall(rng, "clam", "eat", "otter", realistic)
    elif role == "lesion_recruit":
        for l, s in (("p_bear_chase", "bear"), ("p_otter_eat", "otter"), ("c_otter_chase", "otter"), ("c_clam_eat", "clam")):
            r[l] = absent(rng, s, False, realistic)
    else:
        r["p_bear_chase"] = recall(rng, "bear", "chase", "otter", realistic)
        r["p_otter_eat"] = recall(rng, "otter", "eat", "clam", realistic)
        r["c_otter_chase"] = absent(rng, "otter", False, realistic)
        r["c_clam_eat"] = absent(rng, "clam", False, realistic)
        if kind == "no_moat":                                    # answers anything it can compose
            r["c_otter_chase"] = recall(rng, "otter", "chase", "clam", realistic)
        if kind == "one_role":                                   # the new word only works as the patient
            r["p_otter_eat"] = absent(rng, "otter", False, realistic)
    return r


# ══════════════════════════════════════════════ C3 (lived) ════════════════════════════════════════════════════════════
def reply_lived(kind, role, rng, realistic):
    """kind: lived | no_preexposure | no_disclosure | semantic_only ; role: intact | lesion_recall | lesion_episodic | permuted."""
    perm = role == "permuted"
    obj = "pear" if perm else "apple"
    r = _teach("lived", perm, rng, realistic)
    r["pre_wolf_bite"] = absent(rng, "wolf", True, realistic)
    r["pre_wolf_discuss"] = absent(rng, "discuss", True, realistic)
    if kind == "no_preexposure":                                # "recalls" a fact it was never told
        r["pre_wolf_bite"] = recall(rng, "wolf", "bite", "apple", realistic)
    r["d_weather"] = absent(rng, "weather", True, realistic)
    r["d_sky"] = absent(rng, "weather", True, realistic)
    r["d_cat"] = R("the cat eats the fish the dog chases the cat — worth going further here.", ["cat", "eat", "fish"])
    comp = "%.2f" % (float(rng.uniform(0.9, 1.0)) if realistic else 1.0)
    r["p_wolf_bite"] = recall(rng, "wolf", "bite", obj, realistic)
    r["p_wolf_discuss"] = R("Earlier you brought up wolf — my hippocampal readout completes its assembly for it "
                            "(dendritic dAP completion %s). the wolf bites the %s" % (comp, obj), None,
                            episodic={"in_memory": True})
    r["c_wolf_chase"] = R("As for it — " + AB, abst=True)
    r["c_dog_chase"] = recall(rng, "dog", "chase", "cat", realistic)
    if role == "lesion_recall":
        r["p_wolf_bite"] = absent(rng, "wolf", False, realistic)
    if role == "lesion_episodic" or kind == "no_disclosure":
        r["p_wolf_discuss"] = R("I don't recall us discussing wolf — no assembly completes for that cue (a genuine spiking "
                                "completion failure, so I won't make something up).", abst=True, episodic={"in_memory": False})
    return r


REPLIES = {"cat_disc": reply_cat_disc, "word": reply_word, "lived": reply_lived}
DEFAULT_KIND = {"cat_disc": "gen", "word": "bound", "lived": "lived"}
SCORED = {"cat_disc": ["p_canary_eat", "p_canary_might"], "word": ["p_bear_chase", "p_otter_eat"],
          "lived": {"lesion_recall": ["p_wolf_bite"], "lesion_episodic": ["p_wolf_discuss"]}}
PATCH_OF = {"lesion_recruit": "lesion_recruit", "lesion_recall": "lesion_lived_recall",
            "lesion_inherit_pathway": "lesion_inherit_pathway"}   # Amendment A2: one name for arm and patch


def reply_degenerate(kind, group, perm, rng):
    turns = G.turns_for(group, SEEDS[0], "permuted" if perm else "intact_a")
    novel = ["canary", "worm", "bug", "otter", "clam", "bear", "wolf", "apple", "pear"]
    taught = [m for _l, m, r in turns if r == "teach"]
    out = {}
    for l, m, role in turns:
        if kind == "abstain":
            out[l] = R(AB, abst=True)
        elif kind == "echo":
            out[l] = R(m)
        elif kind == "dump":
            out[l] = R(" ".join(taught) or m, [novel[0], "eat", novel[1]], episodic={"in_memory": True})
        elif kind == "random":
            c = int(rng.integers(0, 5))
            a, b = novel[int(rng.integers(0, len(novel)))], novel[int(rng.integers(0, len(novel)))]
            v = ["eat", "chase", "bite"][int(rng.integers(0, 3))]
            out[l] = (R(AB, abst=True) if c == 0 else R(AB + ask_text(a), abst=True) if c == 1 else R(fx(a, v, b), [a, v, b]) if c == 2
                      else R("perhaps %s" % fx(a, v, b), None, hyp=[a, v, b]) if c == 3 else R("Earlier you brought up %s" % a))
        else:
            raise ValueError(kind)
    return out


# ══════════════════════════════════════════════ assembling a six-seed run ══════════════════════════════════════════
def _sub(master, kind, role, salt=0):
    return np.random.default_rng([int(master), zlib.crc32(("%s|%s" % (kind, role)).encode()), int(salt)])


def lesions_of(group):
    return list(G.GROUPS[group]["lesions"])


def default_roles(group):
    v = {"intact_a": "intact", "intact_b": "intact", "permuted": "permuted"}
    for les in lesions_of(group):
        v[les] = les
    return v


def make_group(rng, group, kind=None, over=None, realistic=True, unclean=False, unreproduced=(), degenerate=None, reply=None,
               c1_hop=True, c1_flags=True, zero_calls=()):
    """One six-seed run of ONE group: {seed: {arm: shard}}.
    kind          : which perfect/null brain (see the reply_* functions); default the registered mechanism's perfect brain
    over          : {arm_base: (kind, role)} replaces that arm's kind/role (an inert lesion = (kind, "intact"))
    unclean / unreproduced / degenerate : as in the Stage-2 builder.
    c1_hop / c1_flags : C1 only (Amendment A2): the organ read records hop / the arm records the effective flags
    zero_calls    : arm bases whose in-process patch records calls == 0 (installed, never engaged)"""
    kind = kind or DEFAULT_KIND[group]
    fn = reply or REPLIES[group]
    data = {}
    for seed in SEEDS:
        master = int(rng.integers(0, 2 ** 31 - 1))
        roles = {a: (kind, rl) for a, rl in default_roles(group).items()}
        roles.update(over or {})
        data[seed] = {}
        for arm in G.arms_for(group, with_informational=False):
            base = arm.split(".rep")[0]
            k, rl = roles[base]
            sub = _sub(master, "surface", 0)      # one surface stream per seed: a probe whose content does not change reads
                                                  # the same in every arm (a lesion must not look effective through its wrapper)
            resp = reply_degenerate(degenerate, group, base == "permuted", sub) if degenerate else fn(k, rl, sub, realistic)
            if degenerate and base not in ("intact_a", "intact_b"):
                for x in resp.values():
                    x["answer"] += " (%s)" % base
            if arm.endswith(".rep0") and base in unreproduced:
                labs = SCORED[group] if group != "lived" else SCORED[group][base]
                for l in labs:
                    resp[l] = dict(resp[l], answer=resp[l]["answer"] + " (repeat)")
            if arm == "intact_b" and unclean:
                l0 = next(iter(resp))
                resp[l0] = dict(resp[l0], answer=resp[l0]["answer"] + " (b)")
            patch = None
            if base in PATCH_OF:
                patch = {"patch": PATCH_OF[base], "calls": int(_sub(master, "patch", base).integers(4, 30))}
                if base == "lesion_recall":
                    patch["calls_total"] = patch["calls"] + 26
                if base in zero_calls:
                    patch["calls"] = 0                  # the patch was installed but never called (not engaged)
            sh = {"seed": seed, "group": group, "arm": arm, "responses": resp, "sources": [S],
                  "pythonhashseed_ok": True, "corpus_present": True, "patch": patch}
            if group in G.C1_GROUPS:
                _c1_record(sh, base, c1_hop, c1_flags)
            data[seed][arm] = sh
    return data


def _c1_record(sh, base, hop, flags):
    """Amendment A2's per-arm C1 record, as the attempt-2 worker writes it at the flip revision: the effective flag
    values (`c1_flags`) and, on the direct probe, the organ read (`category_inherit.read`) with `hop: true`; a lesion
    arm's read also carries `lesion: "all"` / `lesion_held: true`. hop=False is the 09-30 organ (no hop record);
    flags=False is a shard written before A2 (no flag record)."""
    if flags:
        sh["c1_flags"] = {"BRAIN_CATEGORY_INHERIT": True, "BRAIN_CATEGORY_INHERIT_HOP": bool(hop)}
    d = sh["responses"].get(G.C1_DIRECT_PROBE)
    if not isinstance(d, dict):
        return
    hs = d.get("hypothesis_svo")
    cut = base == "lesion_inherit_pathway"
    read = {"patient": (hs[2] if hs and not d.get("abstained") else None), "reason": ("reached" if hs else "below_floor"),
            "lesion": "all" if cut else None, "lesion_held": True if cut else None}
    if hop:
        read.update(hop=True, hop_lateral_gain=0.5, query_steps=30)
    sh["responses"][G.C1_DIRECT_PROBE] = dict(d, category_inherit={"observe": None, "read": read})


# ══════════════════════════════════════════════ the decision = the real aggregate ══════════════════════════════════
def _aggregate(shards, seeds, root="<memory>"):
    orig = G._load_shards
    G._load_shards = lambda r, sd: shards
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return G.aggregate(root, seeds=seeds, informational=False)
    finally:
        G._load_shards = orig


def decide_group(data, group):
    """The real aggregate's verdict for this group's primary criterion over these six seeds is PASS."""
    shards = {(seed, group, arm): sh for seed, arms in data.items() for arm, sh in arms.items()}
    return _aggregate(shards, sorted(data))["groups"][group]["verdict"] == "PASS"


def write_tree(root, group, data):
    for seed, arms in data.items():
        for arm, sh in arms.items():
            p = G.shard_path(root, seed, group, arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump(sh, fh)


def selfcheck(n=12, seed=3):
    rng = np.random.default_rng(seed)
    bad = 0
    for group in ("cat_disc", "word", "lived"):
        for i in range(n):
            les = lesions_of(group)[0]
            kw = [dict(), dict(over={"permuted": (DEFAULT_KIND[group], "intact")}), dict(realistic=False),
                  dict(over={les: (DEFAULT_KIND[group], "intact")}), dict(unclean=True), dict(degenerate="random")][i % 6]
            data = make_group(rng, group, **kw)
            with tempfile.TemporaryDirectory() as td:
                write_tree(td, group, data)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    disk = G.aggregate(td, seeds=sorted(data), informational=False)
                mem = _aggregate({(s, group, a): sh for s, arms in data.items() for a, sh in arms.items()}, sorted(data), root=td)
                if json.dumps(disk, sort_keys=True, default=str) != json.dumps(mem, sort_keys=True, default=str):
                    bad += 1
    return bad


# ══════════════════════════════════════════════ the generators a spec exposes ═══════════════════════════════════════
def generators(group, perfect_kinds=None, null_kinds=None, info_kinds=None, lesions=None, extra=None):
    """{name: callable(rng) -> data}.
    perfect_kinds : {suffix: kind} further perfect brains (thresholded as perfect)
    null_kinds    : {name: kind} brains missing one capability, applied to the intact arms only
    info_kinds    : {name: kwargs-for-make_group}  reported, never thresholded"""
    k0 = DEFAULT_KIND[group]
    les = lesions if lesions is not None else lesions_of(group)

    def mk(**kw):
        return lambda rng: make_group(rng, group, **kw)
    out = {"perfect": mk(), "perfect_exact": mk(realistic=False),
           "null_permuted_inert": mk(over={"permuted": (k0, "intact")}),
           "null_unclean": mk(unclean=True),
           "null_all_abstain": mk(degenerate="abstain"), "null_echo": mk(degenerate="echo"),
           "null_never_abstains": mk(degenerate="dump"), "random": mk(degenerate="random")}
    for l in les:
        out["null_%s_inert" % l] = mk(over={l: (k0, "intact")})
        out["null_%s_unreproduced" % l] = mk(unreproduced=(l,))
    for suffix, kind in (perfect_kinds or {}).items():
        out["perfect_%s" % suffix] = mk(kind=kind)
    for name, kind in (null_kinds or {}).items():
        out[name] = mk(over={"intact_a": (kind, "intact"), "intact_b": (kind, "intact")})
    for name, kw in (info_kinds or {}).items():
        out[name] = mk(**kw)
    for name, fn in (extra or {}).items():
        out[name] = fn
    return out


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        b = selfcheck()
        print("selfcheck: %d mismatches between the in-memory and the on-disk aggregate" % b)
        sys.exit(1 if b else 0)
