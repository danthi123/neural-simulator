#!/usr/bin/env python3
"""c5_battery_score.py -- the C5 scorer (idle-replay default-ON flip) of the 2026-10-12 validation battery, with the
clauses of its Amendment B (research/findings/2026-10-12-validation-battery-prereg.md, section 5 "C5" and section
"AMENDMENT B"). Reads files only: it never builds a brain and never imports webapp.

WHY AMENDMENT B (RULING c5-battery-lesion, Claude 2026-10-13). The prereg's C5 bars (1)-(3) cannot tell the spiking
reactivation race from a host `random.choice(kb)`. Bar (2) holds by construction, because the OFF demo lexicon shares
no fact with the KB. In the g6b pair every candidate engram was build-time (age 1, equal excitability), so recency
and salience never engaged. Without more arms, C5 reads "replay fired", not "replay selects". B adds, per seed, a
COMMIT-LESION arm and two EXPOSURE sessions (STAGGERED and ROTATED teaching; ROTATED is the ruling's "permuted-exposure"
session, the teach order rotated by two). Each seed's identity now comes from its
provenance sidecar, because the arm JSON records no seed (RULING c5-seed-provenance).

TWO REVISIONS. The pair and the commit-lesion arm run at the battery pin P (f24963bfb; BRAIN_IDLE_REPLAY_COMMIT_LESION
exists there). The exposure sessions need turn labels P does not have, so they run at R_B = P plus Amendment B's
label-only turn block in research/runners/onebrain_regression_battery.py and nothing else. Check E3 ties the two: at
R_B the OFF and ON pair are re-run and must be whole-record identical to the pin's OFF and ON records, seed by seed,
before the R_B sessions are read as the pin's organ.

LAYOUT. One directory, default research/findings/raw/_regression_battery/c5val_1012/. For each seed N and arm A:
    c5_sN_A.json            the battery worker's responses (research/runners/onebrain_regression_battery.py --worker)
    c5_sN_A.json.prov.json  its provenance sidecar (git_sha, the BRAIN_* and LB_* env)
    c5_sN_A_rss.jsonl       the per-turn RSS sidecar (BRAIN_IDLE_REPLAY_LEAKLOG; read for off/on only)
  at P:
    off           BRAIN_IDLE_REPLAY=0, turns emo,bc_a,bc_b,selfinit (the registered pair)
    on            no BRAIN_IDLE_REPLAY token (the shipped default), the same turns
    commitlesion  B: as `on` plus BRAIN_IDLE_REPLAY_COMMIT_LESION=1 (the organ's own E->G commit gate closed)
  at R_B:
    off_rb        B (E3): as `off`
    on_rb         B (E3): as `on`
    stagger       B: as `on`, turns C5B_SESSIONS["stagger"]["labels"] (four facts taught one per turn, then idle)
    rotated       B: as `on`, turns C5B_SESSIONS["rotated"]["labels"] (the same facts in another order, then idle)
    taglesion     B (review 2026-10-13): the STAGGERED turns with BRAIN_IDLE_REPLAY_TAG_LESION=1 (equal excitability)
    retell        B (review 2026-10-13): turns C5B_SESSIONS["retell"]["labels"] (F1..F4, then F2 re-told verbatim and F1
                  re-told with a new patient, then idle)

THE CONFOUND THE REVIEW FOUND. The organ lists candidates in store-slot order and builds its circuit from (K, seed),
with noise from (seed, turn, K). STAGGERED and ROTATED have the same K, idle turn and slot ages, so on one seed they
run the SAME race and differ only in slot contents: B-C5b holds by construction for any brain that picks by slot, the
12 sessions of B-C5c are 6 paired outcomes, and a host that voices the LAST slot (newest appended, LIFO) passes both.
The TAG-LESION arm (B-C5d) and the RE-TELL arm (B-C5e, newest tag in an EARLY slot) are what separate them.

PER-SEED CLAUSES (the registered C5 bars, then Amendment B's; a clause reads True / False, or None when its inputs
are missing, and a seed is DEFINED only when every clause reads True or False and no validity reason was raised):
  B-SEED   validity: every arm's sidecar exists, its env.BRAIN_CHAT_SEED equals the seed (set explicitly on every
           arm), its git_sha equals --pin for the P arms and --rb-pin for the R_B arms (when given), its BRAIN_* env
           is the arm's registered env and nothing else (BRAIN_CHAT_SEED and BRAIN_IDLE_REPLAY_LEAKLOG allowed), it
           carries no LB_* key at all (in particular LB_BATTERY_RELEASE_SESSIONS must be unset: the battery
           worker's session-release mode stays OFF for every C5 arm, so the memory bar measures production-like
           accumulation), and no turn body carries `_error`. Else UNDEFINED, and no clause reads that arm (a clause
           that needs it reads None), so invalid data can never fail a clause.
  C5.1     precondition: the ON RSS sidecar holds a `select` record with circuits_built > 0. Else UNDEFINED
           ("replay-fired precondition"), never a pass.
  C5.2     the ON idle reply differs from the OFF idle reply (answer/abstained/recalled_svo), the ON idle reply is
           organ-authored (self_initiated.composer brain_store, path idle-replay), and emo/bc_a/bc_b are
           byte-identical (whole turn body) between OFF and ON.
  C5.3     the max per-turn ON-OFF rss_kb over the four turns is <= 1953125 KiB (2 GB = 2e9 bytes, the stricter
           reading; 2 GiB would be 2097152 KiB). Read ONLY between the OFF and ON arms at P (one revision, the same
           session-release mode: unset in both, enforced by B-SEED).
  B-C5a    the commit lesion changes the idle reply: the lesion must have reached the organ
           (idle_replay.lesions.commit true; else UNDEFINED); the commit-lesion idle reply differs from the ON idle
           reply AND is the neutral idle line (abstained, n_utt 0, no winner, the exact fallback text); and
           emo/bc_a/bc_b are byte-identical across OFF, ON and commit-lesion (all at P). The lesion must also show
           no commit-gate spike (sum of selection.gate_counts == 0), else UNDEFINED (docs/TERMS.md "lesion").
  E3       cross-revision check (precondition of B-C5b and of the seed's sessions in B-C5c): off_rb and on_rb are
           whole-record identical (every turn body, sorted-key JSON) to off and on. Else that seed is UNDEFINED, its
           B-C5b reads None and its two sessions are undecided in B-C5c.
  B-C5b    exposure changes which fact (both sessions at R_B, so the comparison is within one revision): validity
           first (each session's idle turn ran the organ unlesioned with K >= 1, and every taught fact is a
           candidate with t_enc >= 1, t_enc strictly increasing in teaching order; else UNDEFINED). Then each
           session voices (n_utt 1, composer brain_store, decode_matches_winner) a fact taught in that session, and
           the STAGGERED fact differs from the ROTATED fact.
POOLED CLAUSES:
  B-C5c    the organ's design prediction: the most recently taught fact is the voiced fact in all but at most
           MAX_NEWEST_MISSES (2) of the 2 x n_seeds exposure sessions (10 of 12 at six seeds). The two sessions of a
           seed run the same race (same K, turn and seed), so these are n_seeds PAIRED outcomes, not 12 draws.
  B-C5d    the tag lesion moves the voiced fact: validity per seed (lesions.tag true, the other lesions off, equal
           x_applied, the same candidate store as the intact STAGGERED arm; a winner that fails to decode makes the
           seed UNDEFINED); `moved` = the tag-lesion voiced fact differs from the intact STAGGERED voiced fact. A
           SILENT tag-lesion turn (nothing latched) is undecided for B-C5d, never counted as moved. Bar: at least
           n_seeds - MAX_TAG_UNMOVED (4 of 6) seeds MOVED and at most MAX_TAG_UNMOVED (2) unmoved; undecided when
           neither is settled. Set on dev seeds 1-8 only, with the stand-in organ, before gate data.
  B-C5e    the re-tell: the newest-tagged candidate (unique max t_enc) is a re-told fact in an EARLY store slot (kb_idx
           below the last slot; else the seed's re-tell is undecided and, when counted, UNDEFINED), and it is the
           voiced fact; at most MAX_RETELL_MISSES (1) misses. Counted only when RETELL_COUNTED (fixed from the dev-7
           smoke before gate data); otherwise reported, never in the verdict.

VERDICT. A failed clause on any seed, or a failed pooled clause: "C5 REVERT-CANDIDATE seed <n> clause <k>". Else,
any UNDEFINED seed (or an undecidable pooled clause): "C5 UNDEFINED". Else "C5 STANDS (PROVISIONAL)", with `reading`:
STANDS means "the idle reply selects by tag recency"; ownership of content selection is UNDETERMINED. The tag-lesion
and re-tell arms separate the TAG from the store SLOT only, never the spiking race from a host argmax over the same
tag (newest t_enc wins, slot 0 on ties, silent under the commit lesion: it passes every counted clause, bar-sanity
`info_host_tag_argmax`). The named next method is a counted race-steer arm (Amendment C candidate,
docs/drafts/idle-replay/WORKLOG.md). Seed 42 is
reported NON-BLIND (the g6b pair ran at seed 42 and was read before the prereg) and still counts: no seed is removed.

Usage:
  .venv/bin/python tools/c5_battery_score.py --dir research/findings/raw/_regression_battery/c5val_1012 \\
      --pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea --rb-pin <R_B> \\
      --out research/findings/raw/_regression_battery/c5val_1012/c5_score.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SEEDS_DEFAULT = (42, 43, 44, 100, 101, 102)
DIR_DEFAULT = "research/findings/raw/_regression_battery/c5val_1012"
ARMS = ("off", "on", "commitlesion", "off_rb", "on_rb", "stagger", "rotated", "taglesion", "retell")
PAIR_ARMS = ("off", "on", "commitlesion", "off_rb", "on_rb")      # run the four pair turns
SESSION_ARMS = ("stagger", "rotated")
RB_ARMS = ("off_rb", "on_rb", "stagger", "rotated", "taglesion", "retell")   # run at R_B
# the session (runner C5B_SESSIONS key) whose turns each session arm runs
ARM_SESSION = {"stagger": "stagger", "rotated": "rotated", "taglesion": "stagger", "retell": "retell"}
ACTIVE_TURNS = ("emo", "bc_a", "bc_b")
IDLE_TURN = "selfinit"
PAIR_TURNS = ACTIVE_TURNS + (IDLE_TURN,)
REPLY_FIELDS = ("answer", "abstained", "recalled_svo")
MEM_BAR_KIB = 1953125            # 2 GB = 2e9 bytes / 1024
MAX_NEWEST_MISSES = 2            # B-C5c: newest-taught fact voiced in >= 2*n_seeds - 2 sessions (10 of 12)
# B-C5d (tag lesion; review ruling 2026-10-13, bar set on DEV seeds 1-8 with the stand-in organ before gate data:
# the tag lesion moved the voiced fact in 69 of 80 dev draws (1 of them silent), 5 of 10 on the worst dev seed, and on
# 8 of 8 dev battery-noise draws). At most this many of the six seeds may keep the intact STAGGERED fact under it.
MAX_TAG_UNMOVED = 2
# B-C5e (re-tell; counted only if the dev-7 smoke showed a re-tell that re-tags a fact in an early slot, see
# RETELL_COUNTED). Dev seeds 1-8 stand-in: the in-place re-told fact won 80 of 80 draws. Misses allowed of six:
MAX_RETELL_MISSES = 1
# Fixed from the dev-7 re-tell smoke (2026-10-13 08:27, research/findings/raw/_regression_battery/c5val_1012_dev/
# c5_s7_retell.json), before any gate data: BOTH re-tells APPENDED a new store slot (F2 verbatim -> kb_idx 9; F1 with a
# new patient -> lynx chase frog at kb_idx 10, the old lynx chase vole kept at 5; no in-slot rewrite on this route), so
# the newest-tagged fact sat in the LAST slot and recency and a last-slot host agree. The re-tell is INFORMATIONAL.
RETELL_COUNTED = False
RELEASE_ENV = "LB_BATTERY_RELEASE_SESSIONS"
# research/runners/self_initiated_production_organ.py::idle_fallback_text() (pinned by tests/test_battery_amendment_b.py)
NEUTRAL_IDLE_LINE = "Nothing in particular is surfacing for me to bring up right now."
ARM_ENV = {"off": {"BRAIN_IDLE_REPLAY": "0"}, "on": {}, "commitlesion": {"BRAIN_IDLE_REPLAY_COMMIT_LESION": "1"},
           "off_rb": {"BRAIN_IDLE_REPLAY": "0"}, "on_rb": {}, "stagger": {}, "rotated": {},
           "taglesion": {"BRAIN_IDLE_REPLAY_TAG_LESION": "1"}, "retell": {}}
ALLOWED_BRAIN_KEYS = ("BRAIN_CHAT_SEED", "BRAIN_IDLE_REPLAY_LEAKLOG")
NON_BLIND = {42: ("the seed-42 OFF/ON pair repeats the g6b pair (seed 42, read 2026-10-12 before the prereg: "
                  "research/findings/2026-10-12-c5-g6b-replay-fired-pair.md); reported NON-BLIND, still counted")}


def c5b_sessions():
    """The exposure sessions as registered in the battery runner (labels, idle label, taught facts in order)."""
    from research.runners.onebrain_regression_battery import C5B_SESSIONS
    return C5B_SESSIONS


# ── reading ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _load_jsonl(path):
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def arm_paths(d, seed, arm):
    base = os.path.join(d, "c5_s%d_%s" % (int(seed), arm))
    return {"resp": base + ".json", "prov": base + ".json.prov.json", "rss": base + "_rss.jsonl"}


def load_seed(d, seed):
    """{arm: {"resp", "prov", "rss"}} for one seed, each None when its file is absent or unreadable."""
    out = {}
    for arm in ARMS:
        p = arm_paths(d, seed, arm)
        out[arm] = {"resp": _load_json(p["resp"]), "prov": _load_json(p["prov"]),
                    "rss": _load_jsonl(p["rss"]) if arm in ("off", "on") else None}
    return out


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
def reply_tuple(body):
    return json.dumps([(body or {}).get(f) for f in REPLY_FIELDS], sort_keys=True)


def body_bytes(body):
    return json.dumps(body, sort_keys=True, default=str)


def _si(body):
    si = (body or {}).get("self_initiated")
    return si if isinstance(si, dict) else {}


def _ir(body):
    ir = _si(body).get("idle_replay")
    return ir if isinstance(ir, dict) else {}


def prov_reasons(seed, arm, prov, pin=None):
    """B-SEED validity reasons for one arm's provenance sidecar (empty = valid)."""
    if not isinstance(prov, dict):
        return ["prov-missing:%s" % arm]
    out = []
    env = prov.get("env") or {}
    s = env.get("BRAIN_CHAT_SEED")
    if s is None:
        out.append("seed-missing:%s" % arm)
    elif str(s).strip() != str(int(seed)):
        out.append("seed-mismatch:%s=%s" % (arm, s))
    if pin and prov.get("git_sha") != pin:
        out.append("pin:%s=%s" % (arm, prov.get("git_sha")))
    want = ARM_ENV[arm]
    for k, v in want.items():
        if env.get(k) != v:
            out.append("arm-env:%s:%s=%s" % (arm, k, env.get(k)))
    for k in sorted(env):
        if k == RELEASE_ENV:
            out.append("release-mode:%s=%s" % (arm, env.get(k)))
        elif k.startswith("LB_") or (k.startswith("BRAIN_") and k not in want and k not in ALLOWED_BRAIN_KEYS):
            out.append("arm-env-extra:%s:%s" % (arm, k))
    return out


def battery_turn_rss(rss):
    """label -> rss_kb from the LAST run (pid) in a per-turn RSS sidecar; None when there is no battery_turn record."""
    recs = [r for r in (rss or []) if isinstance(r, dict) and r.get("event") == "battery_turn"]
    if not recs:
        return None
    pid = recs[-1].get("pid")
    out = {}
    for r in recs:
        if r.get("pid") == pid and isinstance(r.get("rss_kb"), int):
            out[r.get("turn")] = r["rss_kb"]
    return out


def replay_fired(rss):
    return any(isinstance(r, dict) and r.get("event") == "select" and int(r.get("circuits_built") or 0) > 0
               for r in (rss or []))


# ── per-seed scoring ─────────────────────────────────────────────────────────────────────────────────────────────
def session_read(body, taught):
    """Validity + the voiced fact for one exposure session's idle turn. Returns (reasons, voiced_fact_or_None,
    detail)."""
    si, ir = _si(body), _ir(body)
    why = []
    if si.get("path") != "idle-replay" or not ir:
        why.append("not-organ-turn")
        return why, None, {}
    les = ir.get("lesions") or {}
    if any(bool(les.get(k)) for k in ("no_encode", "tag", "commit")):
        why.append("lesioned")
    sel = ir.get("selection") or {}
    if int(sel.get("K") or 0) < 1:
        why.append("replay-not-fired")
    cands = ir.get("candidates") or []
    t_enc = []
    for f in taught:
        hit = [c for c in cands if isinstance(c, dict) and list(c.get("fact") or []) == list(f)]
        if not hit or int(hit[0].get("t_enc") or 0) < 1:
            why.append("not-encoded:%s" % "-".join(f))
        else:
            t_enc.append(int(hit[0]["t_enc"]))
    if len(t_enc) == len(taught) and any(b <= a for a, b in zip(t_enc, t_enc[1:])):
        why.append("t_enc-not-increasing:%s" % t_enc)
    voiced = None
    dec = ir.get("decoded")
    if (si.get("composer") == "brain_store" and int(si.get("n_utt") or 0) == 1 and ir.get("decode_matches_winner")
            is True and isinstance(dec, list)):
        voiced = [str(x) for x in dec]
    detail = {"answer": (body or {}).get("answer"), "voiced": voiced, "winner_fact": ir.get("winner_fact"),
              "t_enc_taught": t_enc, "K": sel.get("K"), "newest_taught": list(taught[-1]) if taught else None}
    return why, voiced, detail


def _voiced(body):
    si, ir = _si(body), _ir(body)
    dec = ir.get("decoded")
    if (si.get("composer") == "brain_store" and int(si.get("n_utt") or 0) == 1 and ir.get("decode_matches_winner")
            is True and isinstance(dec, list)):
        return [str(x) for x in dec]
    return None


def tag_read(body, stagger_body):
    """B-C5d validity + outcome for the TAG-LESION arm's idle turn (the STAGGERED turns with the tag lesion). Returns
    (reasons, voiced_or_None, detail). Valid only when the tag lesion reached the organ alone (equal excitability
    applied, the other lesions off) over the same candidate store as the intact STAGGERED arm."""
    si, ir = _si(body), _ir(body)
    why = []
    if si.get("path") != "idle-replay" or not ir:
        return ["not-organ-turn"], None, {}
    les = ir.get("lesions") or {}
    if not les.get("tag"):
        why.append("tag-lesion-not-applied")
    if les.get("no_encode") or les.get("commit"):
        why.append("other-lesion-on")
    sel = ir.get("selection") or {}
    x = sel.get("x_applied")
    if not isinstance(x, list) or not x or max(x) - min(x) > 1e-9:
        why.append("x-not-equal")
    if stagger_body is not None:
        key = lambda b: [(list(c.get("fact") or []), c.get("kb_idx")) for c in (_ir(b).get("candidates") or [])]  # noqa: E731
        if key(body) != key(stagger_body):
            why.append("store-differs-from-stagger")
    v = _voiced(body)
    silent = v is None and ir.get("winner") is None
    if v is None and not silent:
        why.append("decode-failed")             # a winner that did not decode: a broken arm, the seed is UNDEFINED
    # a SILENT arm (nothing latched) is no evidence the tag lesion MOVED the selection (review ruling 2026-10-13): it is
    # undecided in B-C5d, never counted as moved (see aggregate)
    return why, v, {"answer": (body or {}).get("answer"), "voiced": v, "K": sel.get("K"), "silent": silent}


def retell_read(body, sess):
    """B-C5e validity + outcome for the RE-TELL arm's idle turn. The newest-tagged candidate (unique max t_enc) must be
    a re-told fact sitting in an EARLY store slot (kb_idx below the last slot): only then can recency and a last-slot
    host disagree. Returns (reasons, hit_or_None, detail)."""
    si, ir = _si(body), _ir(body)
    if si.get("path") != "idle-replay" or not ir:
        return ["not-organ-turn"], None, {}
    why = []
    les = ir.get("lesions") or {}
    if any(bool(les.get(k)) for k in ("no_encode", "tag", "commit")):
        why.append("lesioned")
    cands = [c for c in (ir.get("candidates") or []) if isinstance(c, dict)]
    if not cands:
        return why + ["no-candidates"], None, {}
    tmax = max(int(c.get("t_enc") or 0) for c in cands)
    top = [c for c in cands if int(c.get("t_enc") or 0) == tmax]
    last_idx = max(int(c.get("kb_idx") or 0) for c in cands)
    newest = list(top[0].get("fact") or []) if len(top) == 1 else None
    if newest is None:
        why.append("newest-tag-not-unique")
    elif newest not in [list(f) for f in sess.get("retold") or []]:
        why.append("newest-not-retold:%s" % "-".join(newest))
    elif int(top[0].get("kb_idx") or 0) >= last_idx:
        why.append("newest-in-last-slot")          # the re-tell appended a slot: recency == last slot here
    v = _voiced(body)
    hit = (v == newest) if (not why and newest is not None) else None
    return why, hit, {"answer": (body or {}).get("answer"), "voiced": v, "newest_tagged": newest,
                      "newest_kb_idx": (top[0].get("kb_idx") if newest else None), "last_kb_idx": last_idx}


def score_seed(seed, arms, sessions=None, pin=None, rb_pin=None):
    """Score one seed from its loaded arms ({arm: {"resp", "prov", "rss"}}). Pure: no file or brain access.
    `pin` is the revision P the off/on/commitlesion sidecars must record; `rb_pin` (default `pin`) the revision R_B
    the off_rb/on_rb/stagger/rotated sidecars must record."""
    sessions = sessions if sessions is not None else c5b_sessions()
    r = {"seed": int(seed), "state": None, "undefined_reasons": [], "clauses": {}, "detail": {},
         "non_blind": NON_BLIND.get(int(seed)), "sessions": {}}
    und, cl, det = r["undefined_reasons"], r["clauses"], r["detail"]
    resp = {}                       # only VALID arms (B-SEED clean, every turn present): a clause never reads the rest
    for arm in ARMS:
        a = arms.get(arm) or {}
        bad = prov_reasons(seed, arm, a.get("prov"), (rb_pin or pin) if arm in RB_ARMS else pin)
        body = a.get("resp")
        need = PAIR_TURNS if arm in PAIR_ARMS else tuple(sessions[ARM_SESSION[arm]]["labels"])
        if not isinstance(body, dict):
            bad.append("arm-missing:%s" % arm)
        else:
            for lab in need:
                t = body.get(lab)
                if not isinstance(t, dict):
                    bad.append("turn-missing:%s:%s" % (arm, lab))
                elif t.get("_error"):
                    bad.append("turn-error:%s:%s" % (arm, lab))
        und += bad
        if not bad:
            resp[arm] = body
    off, on, cle = resp.get("off"), resp.get("on"), resp.get("commitlesion")
    on_rss = (arms.get("on") or {}).get("rss") if on is not None else None
    off_rss = (arms.get("off") or {}).get("rss") if off is not None else None

    # C5.1 replay-fired precondition
    if on_rss is None:
        und.append("rss-missing:on")
    elif not replay_fired(on_rss):
        und.append("replay-not-fired:on")

    # C5.2 the pair: ON idle differs from OFF, is organ-authored; active turns byte-identical
    if off is not None and on is not None:
        oi, fi = on[IDLE_TURN], off[IDLE_TURN]
        authored = _si(oi).get("composer") == "brain_store" and _si(oi).get("path") == "idle-replay"
        differs = reply_tuple(oi) != reply_tuple(fi)
        same_active = all(body_bytes(on[t]) == body_bytes(off[t]) for t in ACTIVE_TURNS)
        cl["C5.2"] = bool(authored and differs and same_active)
        det["C5.2"] = {"on_answer": oi.get("answer"), "off_answer": fi.get("answer"), "organ_authored": authored,
                       "differs": differs, "active_identical": same_active}
    else:
        cl["C5.2"] = None

    # C5.3 memory (OFF vs ON at P only)
    on_t, off_t = battery_turn_rss(on_rss), battery_turn_rss(off_rss)
    if on_t is None or off_t is None or any(t not in on_t or t not in off_t for t in PAIR_TURNS):
        cl["C5.3"] = None
        und.append("rss-turns-missing")
    else:
        deltas = {t: on_t[t] - off_t[t] for t in PAIR_TURNS}
        cl["C5.3"] = max(deltas.values()) <= MEM_BAR_KIB
        det["C5.3"] = {"delta_rss_kb": deltas, "max_delta_kb": max(deltas.values()), "bar_kb": MEM_BAR_KIB}

    # B-C5a the commit lesion (all at P)
    if cle is not None and on is not None and off is not None:
        ci = cle[IDLE_TURN]
        gates = (_ir(ci).get("selection") or {}).get("gate_counts")
        if _si(ci).get("path") != "idle-replay" or not (_ir(ci).get("lesions") or {}).get("commit"):
            und.append("commit-lesion-not-applied")
            cl["B-C5a"] = None
        elif not isinstance(gates, list) or sum(int(g or 0) for g in gates) != 0:
            # TERMS "lesion": the cut must be shown to have removed the pathway's output: no commit-gate spike
            und.append("commit-lesion-gate-fired:%s" % (gates,))
            cl["B-C5a"] = None
        else:
            neutral = (ci.get("answer") == NEUTRAL_IDLE_LINE and ci.get("abstained") is True
                       and int(_si(ci).get("n_utt") or 0) == 0 and _ir(ci).get("winner") is None)
            changed = reply_tuple(ci) != reply_tuple(on[IDLE_TURN])
            same_active = all(body_bytes(cle[t]) == body_bytes(on[t]) == body_bytes(off[t]) for t in ACTIVE_TURNS)
            cl["B-C5a"] = bool(neutral and changed and same_active)
            det["B-C5a"] = {"lesion_answer": ci.get("answer"), "neutral_line": neutral, "changed": changed,
                            "active_identical_all_arms": same_active}
    else:
        cl["B-C5a"] = None

    # E3 cross-revision check: the R_B pair reproduces the pin's pair, whole record, before R_B sessions count
    off_rb, on_rb = resp.get("off_rb"), resp.get("on_rb")
    if off is None or on is None or off_rb is None or on_rb is None:
        e3 = None
        und.append("E3:unavailable")
    else:
        same = {"off": body_bytes(off_rb) == body_bytes(off), "on": body_bytes(on_rb) == body_bytes(on)}
        e3 = all(same.values())
        und += ["E3:%s_rb-differs" % k for k, v in same.items() if not v]
    det["E3"] = e3

    # B-C5b exposure changes which fact (both sessions at R_B)
    voiced = {}
    for name in SESSION_ARMS:
        body = resp.get(name)
        if body is None:
            continue
        sess = sessions[name]
        taught = [list(f) for f in sess["taught"]]
        why, v, d = session_read(body[sess["idle_label"]], taught)
        und += ["%s:%s" % (name, w) for w in why]
        ok = not why and e3 is True
        r["sessions"][name] = dict(d, valid=not why, e3=e3, newest_hit=(v == taught[-1]) if ok else None)
        if ok:
            voiced[name] = (v, taught)
    if len(voiced) == len(SESSION_ARMS):
        (vs, ts), (vp, tp) = voiced["stagger"], voiced["rotated"]
        cl["B-C5b"] = bool(vs is not None and vp is not None and vs in ts and vp in tp and vs != vp)
        det["B-C5b"] = {"stagger_voiced": vs, "rotated_voiced": vp}
    else:
        cl["B-C5b"] = None

    # B-C5d the tag lesion (pooled in aggregate): does removing the excitability tag move the voiced fact?
    tl = resp.get("taglesion")
    if tl is not None:
        st_body = resp.get("stagger")
        why, tv, td = tag_read(tl[sessions["stagger"]["idle_label"]],
                               st_body[sessions["stagger"]["idle_label"]] if st_body is not None else None)
        und += ["taglesion:%s" % w for w in why]
        st = r["sessions"].get("stagger") or {}
        moved = (tv != st.get("voiced")) if (not why and not td.get("silent") and e3 is True and st.get("valid")) \
            else None
        r["sessions"]["taglesion"] = dict(td, valid=not why, e3=e3, moved=moved)

    # B-C5e the re-tell (pooled in aggregate; counted only when RETELL_COUNTED)
    rt = resp.get("retell")
    if rt is not None:
        why, hit, rd = retell_read(rt[sessions["retell"]["idle_label"]], sessions["retell"])
        reasons = ["retell:%s" % w for w in why]
        if RETELL_COUNTED:
            und += reasons
        r["sessions"]["retell"] = dict(rd, valid=not why, e3=e3, counted=bool(RETELL_COUNTED), reasons=reasons,
                                       retold_hit=hit if (not why and e3 is True) else None)

    r["state"] = "DEFINED" if (not und and all(v is not None for v in cl.values())) else "UNDEFINED"
    return r


def pooled_newest(per_seed, seeds):
    """B-C5c over every exposure session of every seed. Returns (ok True/False/None, detail)."""
    n_total = 2 * len(seeds)
    hits = misses = undecided = 0
    for s in seeds:
        sess = (per_seed.get(s) or {}).get("sessions") or {}
        for name in SESSION_ARMS:
            h = (sess.get(name) or {}).get("newest_hit")
            if h is True:
                hits += 1
            elif h is False:
                misses += 1
            else:
                undecided += 1
    detail = {"n_sessions": n_total, "newest_hits": hits, "misses": misses, "undecided": undecided,
              "max_misses": MAX_NEWEST_MISSES}
    if misses > MAX_NEWEST_MISSES:
        return False, detail
    if undecided:
        return None, detail
    return True, detail


def _pooled_count(per_seed, seeds, arm, key, bad_value, max_bad):
    """A pooled per-session clause: False when more than `max_bad` seeds read `bad_value`, None while any seed is
    undecided, else True."""
    vals = [((per_seed.get(s) or {}).get("sessions") or {}).get(arm, {}).get(key) for s in seeds]
    bad = sum(1 for v in vals if v is bad_value)
    und = sum(1 for v in vals if v is None)
    detail = {"n_seeds": len(seeds), "n_bad": bad, "undecided": und, "max_bad": max_bad,
              "per_seed": {str(s): v for s, v in zip(seeds, vals)}}
    if bad > max_bad:
        return False, detail
    if und:
        return None, detail
    return True, detail


def aggregate(per_seed, seeds):
    seeds = [int(s) for s in seeds]
    fails = [(s, k) for s in seeds for k, v in (per_seed[s]["clauses"] or {}).items() if v is False]
    undef = [s for s in seeds if per_seed[s]["state"] != "DEFINED"]
    pooled_ok, pooled = pooled_newest(per_seed, seeds)
    if pooled_ok is False:
        fails.append(("pooled", "B-C5c"))
    tag_ok, tag = _pooled_count(per_seed, seeds, "taglesion", "moved", False, MAX_TAG_UNMOVED)
    # B-C5d passes on MOVED seeds only: at least n - MAX_TAG_UNMOVED seeds must read moved; a silent seed is undecided,
    # never moved. If too few seeds are decided to reach that count (and none fails it), the clause is undecided.
    n_moved = sum(1 for v in tag["per_seed"].values() if v is True)
    tag["n_moved"], tag["min_moved"] = n_moved, len(seeds) - MAX_TAG_UNMOVED
    if tag_ok is not False:
        tag_ok = True if n_moved >= len(seeds) - MAX_TAG_UNMOVED else None
    if tag_ok is False:
        fails.append(("pooled", "B-C5d"))
    rt_ok, rt = _pooled_count(per_seed, seeds, "retell", "retold_hit", False, MAX_RETELL_MISSES)
    rt["counted"] = bool(RETELL_COUNTED)
    if RETELL_COUNTED and rt_ok is False:
        fails.append(("pooled", "B-C5e"))
    pooled_undecided = pooled_ok is None or tag_ok is None or bool(RETELL_COUNTED and rt_ok is None)
    if fails:
        verdict = "C5 REVERT-CANDIDATE " + "; ".join("seed %s clause %s" % (s, k) if s != "pooled"
                                                     else "pooled clause %s" % k for s, k in fails)
    elif undef or pooled_undecided:
        verdict = "C5 UNDEFINED (seeds %s)" % (",".join(str(s) for s in undef) or "a pooled clause undecided")
    else:
        verdict = "C5 STANDS (PROVISIONAL)"
    nb = {s: NON_BLIND[s] for s in seeds if s in NON_BLIND}
    reading = None
    if verdict.startswith("C5 STANDS"):
        reading = ("the idle reply selects by tag recency (the newest fact wins; the tag lesion moves it%s). "
                   "Ownership of content selection is UNDETERMINED (spiking race vs a host argmax over the same "
                   "tag): the tag-lesion and re-tell arms separate tag from slot only, never the race from a host "
                   "argmax. Next method: a counted race-steer arm (Amendment C candidate)." % (
                       "; the counted re-tell separates the tag from the store slot" if RETELL_COUNTED and rt_ok is True
                       else "; the re-tell arm is informational, so a last-slot host is not excluded either"))
    return {"verdict": verdict, "reading": reading, "failed_clauses": [list(x) for x in fails],
            "undefined_seeds": undef, "pooled_B-C5c": dict(pooled, ok=pooled_ok, paired_outcomes=len(seeds)),
            "pooled_B-C5d": dict(tag, ok=tag_ok), "pooled_B-C5e": dict(rt, ok=rt_ok), "non_blind": nb,
            "n_defined": sum(1 for s in seeds if per_seed[s]["state"] == "DEFINED"), "n_seeds": len(seeds)}


def score_dir(d, seeds=SEEDS_DEFAULT, pin=None, sessions=None, rb_pin=None):
    sessions = sessions if sessions is not None else c5b_sessions()
    per = {int(s): score_seed(s, load_seed(d, s), sessions=sessions, pin=pin, rb_pin=rb_pin) for s in seeds}
    res = aggregate(per, seeds)
    res.update(dir=d, pin=pin, rb_pin=rb_pin or pin, seeds=[int(s) for s in seeds],
               per_seed={str(s): per[s] for s in per},
               amendment="B (research/findings/2026-10-12-validation-battery-prereg.md)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=DIR_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--pin", default=None, help="P: the revision the off/on/commitlesion sidecars must record")
    ap.add_argument("--rb-pin", default=None,
                    help="R_B: the revision the off_rb/on_rb/stagger/rotated sidecars must record (default: --pin)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = score_dir(a.dir, a.seeds, pin=a.pin, rb_pin=a.rb_pin)
    print(json.dumps({k: res[k] for k in ("verdict", "reading", "failed_clauses", "undefined_seeds", "pooled_B-C5c",
                                          "pooled_B-C5d", "pooled_B-C5e", "non_blind", "n_defined", "n_seeds")},
                     indent=1))
    for s, r in res["per_seed"].items():
        if r["undefined_reasons"]:
            print("seed %s UNDEFINED: %s" % (s, ", ".join(r["undefined_reasons"])))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
