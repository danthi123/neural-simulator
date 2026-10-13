"""bar-sanity spec: the C5 bars of the 2026-10-12 validation battery as amended by Amendment B.

THE BAR. `tools/c5_battery_score.py` reads "C5 STANDS (PROVISIONAL)" over the six gate seeds. That needs every seed
DEFINED (B-SEED provenance with session-release mode unset, the C5.1 replay-fired precondition, the commit lesion
applied with no commit-gate spike, E3: the R_B pair whole-record identical to the pin's, every taught fact encoded in
order, the tag lesion applied with equal excitability over the STAGGERED store) and clauses C5.2, C5.3, B-C5a and
B-C5b true on 6/6 seeds, plus the pooled clauses: B-C5c (the newest-taught fact voiced in at least 10 of the 12
exposure sessions; the two sessions of a seed run the same race, so these are 6 paired outcomes), B-C5d (the tag
lesion moves the voiced STAGGERED fact on at least 4 of 6 seeds) and, when RETELL_COUNTED, B-C5e (the re-told fact,
newest-tagged in an EARLY store slot, is voiced on at least 5 of 6 seeds).

THE DECISION IS THE REAL ONE. Each draw builds, per seed, the seven arm records the battery worker writes (responses,
provenance sidecar, per-turn RSS sidecar; off/on/commitlesion at the pin P, off_rb/on_rb/stagger/rotated at R_B)
and calls `c5_battery_score.score_seed` + `aggregate`; nothing is reimplemented here.

BRAINS (per draw, six seeds; KB = the tiny-demo's 5 build facts + the 4 taught facts of a session):
  perfect                 the race voices the newest-taught fact, misses to the second-newest with p 0.025 (the
                          organ's dev battery: 2 misses in 80 intact draws, idle_replay_organ.py grid row 0); the
                          pair's idle turn voices a random build fact (all at age 1); the commit lesion gives the
                          neutral line; ON-OFF RSS uniform in [-5, +50] MB.
  perfect_noisy           as perfect with miss p 0.05 and ON-OFF RSS N(0.4, 0.4) GB clipped to [-1, 1.8] GB.
  null_host_random_kb     a host random.choice over the session KB in every arm, blind to the commit lesion.
  null_host_random_kb_lesion_aware  the same pick, but silenced (neutral line) under the commit lesion.
  null_host_random_taught a lesion-aware host random pick among the session's 4 taught facts only.
  null_primacy            a lesion-aware host that voices the FIRST-taught fact (an order effect, wrong direction).
  null_fixed_fact         one fixed KB fact (dog chase cat) on every idle turn; lesion-aware.
  null_neutral_on         replay fires but nothing latches in any arm: the neutral line everywhere.
  null_replay_never_fires no select record in the ON sidecar (the g6b failure mode before the flip).
  null_active_contamination  as perfect, but the ON arm changes the bc_b reply.
  null_memory_hog         as perfect, but ON-OFF RSS +2.5 GB at the idle turn.
  null_unprovenanced_seed as perfect, but one seed's sidecars lack BRAIN_CHAT_SEED (RULING c5-seed-provenance).
  null_rb_drift           as perfect, but at R_B the ON pair's idle turn voices another fact than at the pin (E3
                          fails: the R_B sessions cannot be read as the pin's organ).
  null_release_mode_on    as perfect, but the OFF and ON arms ran with LB_BATTERY_RELEASE_SESSIONS=1 (the
                          memory bar would not measure production-like accumulation).
  null_tag_lesion_inert   as perfect, but the tag lesion changes nothing (the newest fact still wins): B-C5d fails.
  null_tag_lesion_silent  as perfect, but the tag-lesion arm never voices: silence is never "moved" (B-C5d undecided).
  null_host_newest_lifo   a host that voices the LAST store slot (newest appended: LIFO), silenced under the commit
                          lesion, blind to the tag lesion: B-C5d fails.
  random                  every idle turn of every arm: a uniformly random KB fact or the neutral line.
  (null|info)_retell_last_slot  as perfect, but the re-tell idle turn voices the last store slot: a null (must fail
                          B-C5e) when RETELL_COUNTED, else informational.
  (null|info)_host_newest_lifo_lesion_aware  the last-slot host that also mimics the tag lesion with a random pick.
                          Before the 2026-10-13 review this brain was mislabelled `info_host_fifo_lesion_aware` (it
                          voiced the newest/last slot, i.e. LIFO, not FIFO) and passed 1000 of 1000: the STAGGERED and
                          ROTATED sessions put the newest fact in the last slot and run the same race. Only the re-tell
                          arm (newest tag in an early slot) separates it; when the re-tell is informational it still
                          passes, and C5 STANDS then reads "selects by tag recency", never "the spiking race owns
                          content selection".
  info_host_tag_argmax    a HOST argmax over the organ's own tag (newest t_enc wins, slot 0 on ties, silent under the
                          commit lesion; no race). Review 2026-10-13: it passes every counted clause (and would pass a
                          counted re-tell), so C5 ownership of content selection is UNDETERMINED (race vs host
                          argmax); the named next method is a counted race-steer arm (Amendment C candidate).
  info_weak_race          as perfect with miss p 0.15 (how the pooled 10-of-12 clause bounds a weak race).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import c5_battery_score as C  # noqa: E402

NAME = "C5 bars + Amendment B (validation battery 2026-10-12, six gate seeds)"
DESCRIPTION = ("Synthetic per-seed arm records (off/on/commitlesion at P; off_rb/on_rb/stagger/rotated/taglesion/"
               "retell at R_B) -> "
               "c5_battery_score.score_seed + "
               "aggregate; PASS iff the verdict is 'C5 STANDS (PROVISIONAL)'.")
SCORER_FILES = ["tools/c5_battery_score.py", "docs/drafts/idle-replay/bar_sanity_spec_c5_battery_amendB.py"]

PIN = "f" * 40
RB_PIN = "b" * 40
SEEDS = (42, 43, 44, 100, 101, 102)
BUILD = [["brain", "use", "spikes"], ["brain", "learn", "words"], ["brain", "store", "memory"], ["dog", "chase", "cat"],
         ["cat", "eat", "fish"]]
F = [["lynx", "chase", "vole"], ["heron", "hunt", "frog"], ["yak", "eat", "grass"], ["crow", "chase", "wren"]]
SESSIONS = {"stagger": {"labels": ["c5st_t1", "c5st_t2", "c5st_t3", "c5st_t4", "c5st_idle"], "idle_label": "c5st_idle",
                        "taught": [F[0], F[1], F[2], F[3]]},
            "rotated": {"labels": ["c5rt_t1", "c5rt_t2", "c5rt_t3", "c5rt_t4", "c5rt_idle"],
                         "idle_label": "c5rt_idle", "taught": [F[2], F[3], F[0], F[1]]},
            "retell": {"labels": ["c5re_t1", "c5re_t2", "c5re_t3", "c5re_t4", "c5re_t5", "c5re_t6", "c5re_idle"],
                       "idle_label": "c5re_idle", "taught": [F[0], F[1], F[2], F[3]],
                       "retold": [["heron", "hunt", "frog"], ["lynx", "chase", "frog"]]}}
F1R = ["lynx", "chase", "frog"]       # F1 re-told with a new patient
DEMO = "Something's been on my mind — cat eat worm. What does cat eat?"
GB_KIB = 976562                       # 1e9 bytes in KiB


def _idle(cands, pick, lesion_commit=False, organ=True, lesion_tag=False):
    """An idle-turn body. `pick` = index into cands, or None (nothing latched -> the neutral idle line)."""
    if not organ:
        return {"answer": DEMO, "abstained": False, "recalled_svo": None,
                "self_initiated": {"composer": "onebrain", "path": "light-numpy-deferred-wander", "n_utt": 4}}
    fact = cands[pick]["fact"] if pick is not None else None
    gates = [0] * len(cands)
    if pick is not None and not lesion_commit:      # the closed E->G gate never spikes; a host may still voice a fact
        gates[pick] = 9
    x = [1.0] * len(cands) if lesion_tag else [0.8 + 0.1 * i for i in range(len(cands))]
    ir = {"lesions": {"no_encode": False, "tag": bool(lesion_tag), "commit": bool(lesion_commit)}, "candidates": cands,
          "selection": {"K": len(cands), "gate_counts": gates, "x_applied": x}, "winner": pick, "winner_fact": fact,
          "decoded": fact,
          "decode_matches_winner": (True if fact else None)}
    si = {"composer": "brain_store", "path": "idle-replay", "n_utt": 1 if fact else 0,
          "utterance": " ".join(fact) if fact else None, "idle_replay": ir}
    return {"answer": ("Something's been on my mind — %s." % " ".join(fact)) if fact else C.NEUTRAL_IDLE_LINE,
            "abstained": not fact, "recalled_svo": None, "self_initiated": si}


def _cands(taught=()):
    out = [{"fact": f, "kb_idx": i, "t_enc": 0, "age": 1} for i, f in enumerate(BUILD)]
    out += [{"fact": f, "kb_idx": len(BUILD) + i, "t_enc": i + 1, "age": len(taught) - i} for i, f in enumerate(taught)]
    return out


def _retell_cands(inplace=False):
    """The re-tell store. Default: as the dev-7 smoke left it (c5_s7_retell.json): both re-tells APPENDED a slot, F2
    re-told verbatim at slot 9 (its old slot 6 record replaced), F1 re-told with a new patient at slot 10 beside the
    untouched F1 at slot 5, so the newest-tagged fact is in the LAST slot (the arm is informational). `inplace=True`:
    the shape a counted re-tell needs (F1 rewritten in its own early slot), used only to test the counted logic."""
    out = [{"fact": f, "kb_idx": i, "t_enc": 0, "age": 7} for i, f in enumerate(BUILD)]
    if inplace:
        out += [{"fact": F1R, "kb_idx": 5, "t_enc": 6, "age": 1}, {"fact": F[1], "kb_idx": 6, "t_enc": 2, "age": 5},
                {"fact": F[2], "kb_idx": 7, "t_enc": 3, "age": 4}, {"fact": F[3], "kb_idx": 8, "t_enc": 4, "age": 3}]
        return out
    out += [{"fact": F[0], "kb_idx": 5, "t_enc": 1, "age": 6}, {"fact": F[2], "kb_idx": 7, "t_enc": 3, "age": 4},
            {"fact": F[3], "kb_idx": 8, "t_enc": 4, "age": 3}, {"fact": F[1], "kb_idx": 9, "t_enc": 5, "age": 2},
            {"fact": F1R, "kb_idx": 10, "t_enc": 6, "age": 1}]
    return out


def _prov(seed, arm, with_seed=True, release=False):
    env = dict(C.ARM_ENV[arm], BRAIN_IDLE_REPLAY_LEAKLOG="x_rss.jsonl")
    if with_seed:
        env["BRAIN_CHAT_SEED"] = str(seed)
    if release:
        env[C.RELEASE_ENV] = "1"
    return {"git_sha": RB_PIN if arm in C.RB_ARMS else PIN, "env": env}


def _active(seed):
    return {t: {"answer": "%s reply seed %d" % (t, seed), "abstained": False, "recalled_svo": None} for t in C.ACTIVE_TURNS}


def _rss(base, fired, delta_idle_kib=0, rng=None, spread=(0, 0)):
    recs = []
    for i, t in enumerate(C.PAIR_TURNS):
        if t == C.IDLE_TURN and fired:
            recs.append({"event": "select", "circuits_built": 1, "pid": 1})
        d = delta_idle_kib if t == C.IDLE_TURN else (int(rng.uniform(*spread)) if rng is not None else 0)
        recs.append({"event": "battery_turn", "turn": t, "rss_kb": base[i] + d, "pid": 1})
    return recs


def _brain(policy, miss=0.025, rss=("small",), fire=True, contaminate=False, unprov_seed=None, rb_drift=False,
           release=False):
    """policy(rng, cands, taught, arm) -> index into cands or None. `arm` in on|commitlesion|stagger|rotated."""
    def gen(rng):
        per = {}
        for s in SEEDS:
            base = [6_231_000, 11_685_000, 11_685_000, 16_914_000]
            pair_c = _cands()
            act = _active(s)
            off = dict(act, selfinit=_idle(pair_c, None, organ=False))
            on_pick = policy(rng, pair_c, [], "on", miss)
            on = dict(act, selfinit=_idle(pair_c, on_pick))
            if contaminate:
                on["bc_b"] = dict(on["bc_b"], answer="bc_b changed by the organ")
            cl_pick = policy(rng, pair_c, [], "commitlesion", miss)
            cle = dict(act, selfinit=_idle(pair_c, cl_pick, lesion_commit=True))
            if rss[0] == "small":
                d_idle, spread = int(rng.uniform(-5_000, 50_000)), (-5_000, 50_000)
            elif rss[0] == "noisy":
                d_idle = int(min(1.8, max(-1.0, rng.normal(0.4, 0.4))) * GB_KIB)
                spread = (-1.0 * GB_KIB, 1.8 * GB_KIB)
            else:
                d_idle, spread = int(2.5 * GB_KIB), (-5_000, 50_000)
            on_rb = dict(on)
            if rb_drift:
                other = (on_pick + 1) % len(pair_c) if on_pick is not None else 0
                on_rb["selfinit"] = _idle(pair_c, other)
            ok = s != unprov_seed
            arms = {"off": {"resp": off, "prov": _prov(s, "off", ok, release), "rss": _rss(base, False)},
                    "on": {"resp": on, "prov": _prov(s, "on", ok, release),
                           "rss": _rss(base, fire, d_idle, rng, spread)},
                    "commitlesion": {"resp": cle, "prov": _prov(s, "commitlesion", ok), "rss": None},
                    "off_rb": {"resp": dict(off), "prov": _prov(s, "off_rb", ok), "rss": None},
                    "on_rb": {"resp": on_rb, "prov": _prov(s, "on_rb", ok), "rss": None}}
            for arm, name in C.ARM_SESSION.items():
                sess = SESSIONS[name]
                c = _retell_cands() if name == "retell" else _cands(sess["taught"])
                body = {lab: {"answer": "taught", "abstained": False, "recalled_svo": None}
                        for lab in sess["labels"][:-1]}
                body[sess["idle_label"]] = _idle(c, policy(rng, c, sess["taught"], arm, miss),
                                                 lesion_tag=(arm == "taglesion"))
                arms[arm] = {"resp": body, "prov": _prov(s, arm, s != unprov_seed), "rss": None}
            per[s] = arms
        return per
    return gen


TAG_SILENT_P = 1 / 80          # dev seeds 1-8 stand-in: tag-lesion draws that latched nothing


def _tag_argmax(rng, cands, taught, arm, miss):
    """A HOST argmax over the organ's own tag, no spiking race: the newest t_enc wins, slot 0 on ties (every tag equal
    under the tag lesion, and in the pair's build-only store), silent under the commit lesion."""
    if arm == "commitlesion":
        return None
    if arm == "taglesion" or not taught:
        return 0
    return _newest(cands)


def _newest(cands):
    return max(range(len(cands)), key=lambda i: cands[i]["t_enc"])


def _second(cands):
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["t_enc"])
    return order[1]


def _organ(rng, cands, taught, arm, miss, tag_inert=False, retell_last=False, tag_silent=False):
    """The organ's design: the most excitable (newest-tagged) engram wins the race, missing to the second-newest with
    p `miss`; under the tag lesion every engram is equally excitable and the noise picks (dev seeds 1-8 stand-in:
    newest won 10 of 80 tag-lesion draws, about chance 1/9, and 1 of 80 latched nothing: TAG_SILENT_P)."""
    if arm == "commitlesion":
        return None
    if not taught:
        return int(rng.integers(0, len(cands)))          # build facts only, all at age 1: the noise decides
    if arm == "taglesion" and not tag_inert:
        if tag_silent or rng.random() < TAG_SILENT_P:
            return None                                  # nothing latched: B-C5d reads that seed UNDEFINED
        return int(rng.integers(0, len(cands)))
    if arm == "retell" and retell_last:
        return len(cands) - 1
    return _second(cands) if rng.random() < miss else _newest(cands)


def _host_random_kb(aware):
    def pol(rng, cands, taught, arm, miss):
        if aware and arm == "commitlesion":
            return None
        return int(rng.integers(0, len(cands)))
    return pol


def _host_random_taught(rng, cands, taught, arm, miss):
    if arm == "commitlesion":
        return None
    if not taught:
        return int(rng.integers(0, len(cands)))
    return len(BUILD) + int(rng.integers(0, len(taught)))


def _primacy(rng, cands, taught, arm, miss):
    if arm == "commitlesion":
        return None
    return len(BUILD) if taught else int(rng.integers(0, len(cands)))


def _fixed(rng, cands, taught, arm, miss):
    return None if arm == "commitlesion" else 3                 # dog chase cat


def _neutral(rng, cands, taught, arm, miss):
    return None


def _random(rng, cands, taught, arm, miss):
    k = int(rng.integers(0, len(cands) + 1))
    return None if k == len(cands) else k


def _lifo(tag_aware):
    """A host that voices the LAST store slot (newest-appended, i.e. LIFO), silenced under the commit lesion;
    `tag_aware` also mimics the tag lesion with a random pick."""
    def pol(rng, cands, taught, arm, miss):
        if arm == "commitlesion":
            return None
        if not taught or (tag_aware and arm == "taglesion"):
            return int(rng.integers(0, len(cands)))
        return len(cands) - 1
    return pol


def decide(data):
    per = {s: C.score_seed(s, data[s], sessions=SESSIONS, pin=PIN, rb_pin=RB_PIN) for s in SEEDS}
    return C.aggregate(per, SEEDS)["verdict"] == "C5 STANDS (PROVISIONAL)"


perfect = _brain(_organ)
perfect_noisy = _brain(lambda r, c, t, a, m: _organ(r, c, t, a, 0.05), rss=("noisy",))
null_host_random_kb = _brain(_host_random_kb(False))
null_host_random_kb_lesion_aware = _brain(_host_random_kb(True))
null_host_random_taught = _brain(_host_random_taught)
null_primacy = _brain(_primacy)
null_fixed_fact = _brain(_fixed)
null_neutral_on = _brain(_neutral)
null_replay_never_fires = _brain(_organ, fire=False)
null_active_contamination = _brain(_organ, contaminate=True)
null_memory_hog = _brain(_organ, rss=("hog",))
null_unprovenanced_seed = _brain(_organ, unprov_seed=43)
null_rb_drift = _brain(_organ, rb_drift=True)
null_release_mode_on = _brain(_organ, release=True)
null_tag_lesion_inert = _brain(lambda r, c, t, a, m: _organ(r, c, t, a, m, tag_inert=True))
null_tag_lesion_silent = _brain(lambda r, c, t, a, m: _organ(r, c, t, a, m, tag_silent=True))
null_host_newest_lifo = _brain(_lifo(False))
random = _brain(_random)
info_host_tag_argmax = _brain(_tag_argmax)
if C.RETELL_COUNTED:
    null_retell_last_slot = _brain(lambda r, c, t, a, m: _organ(r, c, t, a, m, retell_last=True))
    null_host_newest_lifo_lesion_aware = _brain(_lifo(True))
else:
    info_retell_last_slot = _brain(lambda r, c, t, a, m: _organ(r, c, t, a, m, retell_last=True))
    info_host_newest_lifo_lesion_aware = _brain(_lifo(True))
info_weak_race = _brain(lambda r, c, t, a, m: _organ(r, c, t, a, 0.15))
