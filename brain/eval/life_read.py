# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: blind whole-transcript reading (canary recall or decoy bars may fail on DEV; then the instrument is fixed before it scores Stage 1)
"""Blind whole-transcript reading instrument (design section 7; section 5 common rule for RIGHT). Host only: it
prepares blinded material from life JSONs and scores reader outputs; the readers (sonnet, prompts in
life_read_spec.py) are run by the orchestrator, never by this code.

Stage A (emit):
  python -m brain.eval.life_read --emit DIR --lives out/life_s42.json ... [--weeks 1,3] [--lived-context all|intact]
  UNITS per (seed, week, arm), arms intact / hippocampus-off / both-off:
    week w of a life of >= 15 days = lived days 5w-4..5w (trunk) + the battery of night 5w; a shorter life (Stage 0,
    7 days): week 1 = day 1 + night 1, week 3 = the last day + the last night.
    The lived days are lived by the intact trunk only (the lesioned arms are probe copies that never live a day). By
    default (--lived-context all) the same trunk lived days precede every arm's battery, as section 7 asks ("whole
    lived days and probe batteries" for each arm): every unit then has the same shape and length class, so the arm is
    not readable from the unit's length, and a reader can judge "wrong" / "invented" / "wrong-episode" in the lesioned
    batteries against what the user really said. The lived-day defects are then the trunk's in every arm, so the key
    records each turn's section and Stage B reports battery-only defect counts: the arm comparison that is like for
    like. --lived-context intact (lived days in the intact unit only) is kept for comparison; it unblinds the arm by
    length (about 3x shorter lesioned units) and leaves the lesioned answers without the tellings they answer.
  Rendered as numbered "you:" / "brain:" lines, conversations split by spec.SEP; no recalled episode, no marks, no
  arm, seed, night or day names. Per batch: 2 duplicate pairs, 6 canaries (spec.CANARY_TYPES, one planted defect in
  a copy of a real intact unit), 3 decoys (spec.DECOY_TYPES, every reply replaced in a copy of a last-week intact unit).
  Writes DIR/units.jsonl {id, text}, DIR/reader_prompts.jsonl {id, prompt}, DIR/pairs.jsonl {id, prompt},
  DIR/answer_prompts.jsonl {id, prompt} and the sealed DIR/blind_key.json. Ids T*/P*/A* are shuffled (fixed rng).
  Pairs: per seed intact w1 vs intact w3 (whole units); intact w3 vs both-off w3 (the intact lived days as shared
  context, then each battery); each decoy vs both-off of its seed (same context form). EVERY pair is emitted in BOTH
  A/B orders as two prompts with different ids (never adjacent in the id order); Stage B: a pair has a winner only if
  both orders name the same side, else it is a tie (inconsistent); the per-reader consistency rate is the
  position-bias read-out. Ids are shuffled, with no near-copy units adjacent.
  Answer prompts: every battery row of kind spec.STRICT_KINDS carrying a strict "ok", question + reply only, plus 6
  controls (spec.ANSWER_CONTROLS) with a known verdict; the key holds the strict ok.
Stage B (score):
  python -m brain.eval.life_read --score --key DIR/blind_key.json --reads R1.jsonl [--reads2 R2.jsonl]
      --pairs P1.jsonl [--pairs2 P2.jsonl] [--answers A1.jsonl --answers2 A2.jsonl] --out PREFIX
  reads: {"id", "defects": [{"type", "quote", "turn"}], "coherence", "honesty": [...]}; pairs: {"id", "choice"};
  answers: {"id", "verdict"}. Writes PREFIX.json + PREFIX.md (validity table with each section-7 bar marked).
  RIGHT = strict ok AND reader 1 verdict yes (reader 2 reported beside it; disagreements listed for adjudication).
G9 cross-arm pairwise (brain/research/2026-11-23-stage2-test2-prereg.md section 5; adds functions only):
  python -m brain.eval.life_read --emit-cross DIR --arm-lives A_s*.json --b-lives B_s*.json [--cross-week 3]
      [--rng-seed N] [--cross-id-prefix C]
  python -m brain.eval.life_read --score-cross --key DIR/cross_key.json --pairs P1.jsonl --pairs2 P2.jsonl --out PREFIX
  Per seed the arm's intact unit vs B's intact unit (whole units, frozen spec.PAIR_PROMPT, no shared context) plus a
  decoy pair (decoy vs the same seed's both-off unit, section-7 form; must not beat both-off), a duplicate pair
  (report-only noise) and per batch the six canary pairs (copy must not win); both orders; two readers required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

from brain.eval import life_read_spec as SP

SPEC_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_read_spec.sha256")
# structural strings that must never reach a reader (arm, seed, night, ledger, marks, recalled episodes)
LEAK_RE = re.compile(r"recalled:|✅|❌|(?-i:WRONG-FETCH|INTRUSION|FALSE\b)|\bintact\b|hippocampus|both-off|consolidated|\bseed\b|"
                     r"\bledger\b|\bnight \d|\bL\d+\b|morning after|\barm\b|canary|decoy", re.I)


def spec_sha() -> str:
    return hashlib.sha256(open(SP.__file__, "rb").read()).hexdigest()


def load_spec():
    """The spec module; its sha256 must equal the recorded one (fail closed)."""
    want = open(SPEC_HASH_FILE).read().split()[0]
    got = spec_sha()
    if got != want:
        raise RuntimeError(f"read spec changed: sha256 {got[:12]} != recorded {want[:12]} ({SPEC_HASH_FILE})")
    return SP


def leaks(text: str) -> list:
    return sorted({m.group(0) for m in LEAK_RE.finditer(text)})


# ---------------------------------------------------------------- collecting units
def week_plan(res: dict, w: int):
    """(lived days, battery night) of week w, or None if the life does not reach it."""
    n = res["days"]
    if n >= 3 * SP.WEEK_DAYS:
        days = list(range(SP.WEEK_DAYS * w - SP.WEEK_DAYS + 1, SP.WEEK_DAYS * w + 1))
        return (days, days[-1]) if days[-1] <= n else None
    if w == 1:
        return [1], 1
    if w == 3:
        return [n], n
    return None


def _battery(res: dict, night: int, arm: str):
    ps = [p for p in res["probes"] if p.get("line", "trunk") == "trunk" and p["night"] == night and p["arm"] == arm
          and not p.get("tag")]
    return ps[0]["rows"] if ps else None


def _turn(r: dict, section: str, day=None) -> dict:
    return {"you": r["you"], "brain": r["brain"], "kind": r["kind"], "item": r.get("item"), "ok": r.get("ok"),
            "section": section, "day": day}


def collect(lives: list, weeks, lived_context: str = "all") -> list:
    """Real units: {seed, week, arm, night, days, turns: [turn...]} with turns as conversations (lists)."""
    out = []
    for res in sorted(lives, key=lambda r: r["seed"]):
        for w in weeks:
            plan = week_plan(res, w)
            if plan is None:
                continue
            days, night = plan
            lived = []
            for d in days:
                x = next((x for x in res["lived"] if x["line"] == "trunk" and x["day"] == d), None)
                if x:
                    lived.append([_turn(r, "lived", d) for r in x["rows"]])
            for arm in SP.ARMS:
                rows = _battery(res, night, arm)
                if rows is None:
                    continue
                convs = ([c for c in lived] if (arm == "intact" or lived_context == "all") else [])
                convs = convs + [[_turn(r, "battery") for r in rows]]
                out.append({"seed": res["seed"], "week": w, "arm": arm, "night": night, "days": days, "convs": convs})
    return out


def _one_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def render(convs: list) -> tuple[str, list]:
    """Plain numbered dialogue; returns (text, turn table [{turn, section, kind, day}])."""
    L, table, k = [], [], 0
    for ci, conv in enumerate(convs):
        if ci:
            L.append(SP.SEP)
        for t in conv:
            k += 1
            L.append(f"[{k}] you: {_one_line(t['you'])}")
            L.append(f"    brain: {_one_line(t['brain'])}")
            table.append({"turn": k, "section": t["section"], "kind": t["kind"], "day": t["day"]})
    return "\n".join(L), table


def _flat(convs):
    return [t for c in convs for t in c]


def _copy(convs):
    return [[dict(t) for t in c] for c in convs]


# ---------------------------------------------------------------- perspective helpers
_SECOND = ((r"\bI'm\b", "you're"), (r"\bI am\b", "you are"), (r"\bI was\b", "you were"), (r"\bI've\b", "you've"),
           (r"\bI'd\b", "you'd"), (r"\bI'll\b", "you'll"), (r"\bmyself\b", "yourself"), (r"\bI\b", "you"),
           (r"\bmy\b", "your"), (r"\bMy\b", "Your"), (r"\bme\b", "you"), (r"\bmine\b", "yours"))


def second_person(s: str) -> str:
    """Last sentence of a first-person telling, turned to the second person ("My cat is called Pixel." -> "Your cat ...")."""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
    t = sents[-1] if sents else s
    for a, b in _SECOND:
        t = re.sub(a, b, t)
    return t[0].upper() + t[1:] if t else t


def last_sentence(s: str) -> str:
    sents = [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
    return sents[-1] if sents else s


def _tellings(res: dict, night: int) -> dict:
    """item -> telling, for ledger items told on or before the night (with a telling text)."""
    return {iid: e["telling"] for iid, e in sorted(res["ledger"].items())
            if e.get("telling") and any(d <= night for d in e["told_days"])}


# ---------------------------------------------------------------- canaries
def _told_in_unit(flat: list) -> set:
    """Items whose telling is one of the unit's own user turns (the reader can see them)."""
    return {t["item"] for t in flat if t["kind"] == "tell" and t["item"]}


def plant(unit: dict, ctype: str, res: dict, rng: random.Random):
    """A copy of `unit` with one planted defect of ctype in its battery, or None if the unit has no site.
    Returns (convs, plant {type, turns (1-based), text, original}).
    The user facts a plant uses (the fox's name, the other episode of a wrong-episode answer) are told IN the unit,
    so a reader who knows only this transcript can see the defect; a unit without such a fact has no site."""
    convs = _copy(unit["convs"])
    flat = _flat(convs)
    bat = [i for i, t in enumerate(flat) if t["section"] == "battery"]
    seen = _told_in_unit(flat)
    tell = {iid: x for iid, x in _tellings(res, unit["night"]).items() if iid in seen}
    site = SP.CANARY_SITE[ctype]

    def put(i, text):
        orig = flat[i]["brain"]
        flat[i]["brain"] = text
        return convs, {"type": ctype, "defect": SP.CANARY_DEFECT[ctype], "turns": [i + 1], "text": text,
                       "original": orig}

    if site in ("open-story", "greet", "open-feeling"):
        cand = [i for i in bat if flat[i]["kind"] == site]
        if not cand:
            return None
        i = cand[0]
        if ctype == "pet-named":
            told_keys = {k for iid in tell for k in res["ledger"][iid]["keys"][:1]}
            # a key the unit's own story reply already uses would not be a planted change
            used = _one_line(flat[i]["brain"]).lower()
            keys = [k for k in SP.PET_NAME_PREFERENCE if k in told_keys and k not in used]
            if not keys:     # else a proper name told in the unit (capitalised in its telling), drawn at random
                keys = sorted(k for iid in tell for k in res["ledger"][iid]["keys"][:1]
                              if k.isalpha() and k not in used and k.capitalize() in tell[iid])
                keys = [rng.choice(keys)] if keys else []
            if not keys:
                return None
            return put(i, SP.PLANT[ctype].replace("{name}", keys[0].capitalize()))
        return put(i, SP.PLANT[ctype])
    if site == "answered":
        cand = [i for i in bat if flat[i]["kind"] in ("H", "rotating") and flat[i]["ok"] and flat[i]["item"] in tell]
        if ctype == "wrong-episode":     # the asked item must have another told item to swap in
            cand = [i for i in cand if any(iid != flat[i]["item"] and res["ledger"][iid]["cls"] in
                                           ("H", "AB", "E", "F", "D") for iid in tell)]
        if not cand:
            return None
        i = rng.choice(cand)
        item = flat[i]["item"]
        if ctype == "perspective":
            return put(i, last_sentence(tell[item]))
        others = [iid for iid in tell if iid != item and res["ledger"][iid]["cls"] in ("H", "AB", "E", "F", "D")]
        # a one-fact telling, so the planted answer is as short as the replies around it (a long E episode would
        # stand out by length); an E telling only if nothing shorter was told in the unit
        short = [iid for iid in others if len(tell[iid]) <= 60]
        others = short or others
        if not others:
            return None
        return put(i, second_person(tell[rng.choice(others)]))
    if site == "run4":
        starts = [j for j in range(len(bat) - 3) if bat[j + 3] == bat[j] + 3
                  and len({_one_line(flat[bat[j] + m]["brain"]).lower() for m in range(4)}) == 4]
        if not starts:
            return None
        i0 = bat[rng.choice(starts)]
        text = flat[i0]["brain"]
        orig = [flat[i0 + m]["brain"] for m in range(1, 4)]
        for m in range(1, 4):
            flat[i0 + m]["brain"] = text
        return convs, {"type": ctype, "defect": SP.CANARY_DEFECT[ctype], "turns": [i0 + 1 + m for m in range(4)],
                       "text": text, "original": orig}
    raise ValueError(ctype)


# ---------------------------------------------------------------- decoys
def _content_word(s: str) -> str:
    ws = [w for w in re.findall(r"[A-Za-z]+", s) if len(w) >= 4 and w.lower() not in SP.GENERIC_STOP]
    return max(ws, key=len).lower() if ws else "things"


def decoy(unit: dict, dtype: str, res: dict) -> list:
    """A copy of `unit` with every reply replaced (user turns kept)."""
    convs = _copy(unit["convs"])
    first = min(unit["days"]) if any(t["section"] == "lived" for t in _flat(convs)) else unit["night"] + 1
    told_keys = []      # keys of items told before the unit's first turn, then as they are told in it
    for iid, e in sorted(res["ledger"].items()):
        if e.get("telling") and e["keys"] and any(d < first for d in e["told_days"]):
            told_keys.append(e["keys"][0])
    k = 0
    for t in _flat(convs):
        it = res["ledger"].get(t["item"]) if t["item"] else None
        if dtype == "refuse":
            t["brain"] = SP.REFUSE
        elif dtype == "generic":
            w = _content_word(t["you"])
            t["brain"] = SP.GENERIC_TEMPLATES[k % len(SP.GENERIC_TEMPLATES)].replace("{W}", w.capitalize()).replace("{w}", w)
        elif dtype == "keyword-stuffer":
            if t["kind"] == "tell":
                t["brain"] = t["you"]
                if it:
                    told_keys += [x for x in it["keys"][:1] if x not in told_keys]
            elif told_keys:
                t["brain"] = ", ".join(x.capitalize() for x in told_keys[-8:]) + "."
            else:
                t["brain"] = t["you"]
        else:
            raise ValueError(dtype)
        k += 1
    return convs


# ---------------------------------------------------------------- Stage A
def _fill_reader(uid: str, text: str) -> str:
    tax = "\n".join(f"- {t}: {d}" for t, d in SP.TAXONOMY)
    return SP.READER_PROMPT.replace("{taxonomy}", tax).replace("{id}", uid).replace("{text}", text)


def _fill_pair(pid: str, a: str, b: str, context: str | None) -> str:
    ctx = SP.PAIR_CONTEXT.replace("{text}", context) if context else ""
    return SP.PAIR_PROMPT.replace("{context}", ctx).replace("{a}", a).replace("{b}", b).replace("{id}", pid)


def _fill_answer(aid: str, q: str, r: str) -> str:
    return SP.ANSWER_PROMPT.replace("{id}", aid).replace("{question}", _one_line(q)).replace("{reply}", _one_line(r))


def _sections(convs, section):
    return [c for c in ([[t for t in conv if t["section"] == section] for conv in convs]) if c]


def build(lives: list, weeks=(1, 3), lived_context: str = "all", rng_seed: int = 0) -> dict:
    """All blinded material + the key, in memory (deterministic for the same inputs)."""
    load_spec()
    rng = random.Random(f"life-read:{rng_seed}")
    by_seed = {r["seed"]: r for r in lives}
    real = collect(lives, weeks, lived_context)
    if not real:
        raise ValueError("no units: the lives have no battery on the week nights")
    wlast = max(weeks)
    entries = []        # {convs, meta}
    for u in real:
        entries.append({"convs": u["convs"], "meta": {"kind": "real", "seed": u["seed"], "week": u["week"],
                                                      "arm": u["arm"], "night": u["night"], "days": u["days"]}, "u": u})
    # duplicate pairs
    dup_src = rng.sample(range(len(real)), min(SP.N_DUP_PAIRS, len(real)))
    for j, i in enumerate(dup_src):
        e = entries[i]
        e["dup_group"] = j
        entries.append({"convs": _copy(e["convs"]), "meta": dict(e["meta"], kind="duplicate"), "u": e["u"], "dup_group": j})
    # canaries: one per type, on intact units (full lived days + battery), sources rotated
    intact = [u for u in real if u["arm"] == "intact"]
    order = list(range(len(intact)))
    rng.shuffle(order)
    canary_log = []
    for ci, ctype in enumerate(SP.CANARY_TYPES):
        done = None
        for step in range(len(order)):
            u = intact[order[(ci + step) % len(order)]]
            got = plant(u, ctype, by_seed[u["seed"]], rng)
            if got:
                done = (u, got)
                break
        if not done:
            canary_log.append({"type": ctype, "planted": False})
            continue
        u, (convs, pl) = done
        canary_log.append({"type": ctype, "planted": True})
        entries.append({"convs": convs, "meta": {"kind": "canary", "seed": u["seed"], "week": u["week"], "arm": u["arm"],
                                                 "night": u["night"], "days": u["days"], "canary": pl}, "u": u})
    # decoys on last-week intact units
    last = [u for u in intact if u["week"] == wlast] or intact
    d0 = rng.randrange(len(last))
    for di, dtype in enumerate(SP.DECOY_TYPES):
        u = last[(d0 + di) % len(last)]
        entries.append({"convs": decoy(u, dtype, by_seed[u["seed"]]),
                        "meta": {"kind": "decoy", "decoy": dtype, "seed": u["seed"], "week": u["week"], "arm": u["arm"],
                                 "night": u["night"], "days": u["days"]}, "u": u})
    # ids: shuffled, and no two near-copies (a duplicate and its source, a canary and its source) get adjacent ids
    def _src(e):
        m = e["meta"]
        return (m["seed"], m["week"], m["arm"]) if m["kind"] in ("real", "duplicate", "canary") else None

    perm = list(range(len(entries)))
    for _ in range(10000):
        rng.shuffle(perm)
        if all(_src(entries[perm[k]]) is None or _src(entries[perm[k]]) != _src(entries[perm[k + 1]])
               for k in range(len(perm) - 1)):
            break
    else:
        raise RuntimeError("could not separate near-copy units in the id order")
    units, prompts, key_units = [], [], {}
    for k, i in enumerate(perm):
        e = entries[i]
        e["id"] = f"T{k}"
    for e in sorted(entries, key=lambda e: int(e["id"][1:])):
        text, table = render(e["convs"])
        e["text"] = text
        m = dict(e["meta"])
        m["turns"] = table
        m["n_turns"] = len(table)
        m["chars"] = len(text)
        units.append({"id": e["id"], "text": text})
        prompts.append({"id": e["id"], "prompt": _fill_reader(e["id"], text)})
        key_units[e["id"]] = m
    for e in entries:
        if "dup_group" in e:
            partner = [x["id"] for x in entries if x.get("dup_group") == e["dup_group"] and x is not e]
            key_units[e["id"]]["dup_partner"] = partner[0]
    # pairs
    real_e = {(e["meta"]["seed"], e["meta"]["week"], e["meta"]["arm"]): e for e in entries if e["meta"]["kind"] == "real"}
    pair_specs = []
    w1 = min(weeks)
    for s in sorted(by_seed):
        a, b = real_e.get((s, w1, "intact")), real_e.get((s, wlast, "intact"))
        if a and b and w1 != wlast:
            pair_specs.append({"what": "w3-vs-w1", "seed": s, "x": ("intact-w%d" % wlast, b["text"]),
                               "y": ("intact-w%d" % w1, a["text"]), "context": None})
        it, bo = real_e.get((s, wlast, "intact")), real_e.get((s, wlast, "both-off"))
        if it and bo:
            ctx = render(_sections(it["convs"], "lived"))[0] if _sections(it["convs"], "lived") else None
            pair_specs.append({"what": "intact-vs-bothoff", "seed": s,
                               "x": ("intact", render(_sections(it["convs"], "battery"))[0]),
                               "y": ("both-off", render(_sections(bo["convs"], "battery"))[0]), "context": ctx})
    for e in entries:
        if e["meta"]["kind"] == "decoy":
            s = e["meta"]["seed"]
            bo = real_e.get((s, e["meta"]["week"], "both-off"))
            src = real_e.get((s, e["meta"]["week"], "intact"))
            if bo and src:
                lived = _sections(src["convs"], "lived")
                pair_specs.append({"what": "decoy-vs-bothoff", "seed": s, "unit": e["id"],
                                   "x": ("decoy:" + e["meta"]["decoy"], render(_sections(e["convs"], "battery"))[0]),
                                   "y": ("both-off", render(_sections(bo["convs"], "battery"))[0]),
                                   "context": render(lived)[0] if lived else None})
    # every real comparison is emitted in BOTH A/B orders (two prompts, two ids), because a reader's position bias
    # otherwise makes a single order measure noise; a pair has a winner only if both orders name the same side (Stage B)
    # pair rng is its own stream, so the answer prompts below do not depend on the pair count
    prng = random.Random(f"pairs-{rng_seed}")
    pair_prompts = []
    for g, ps in enumerate(pair_specs):
        for flip in (False, True):
            pair_prompts.append(dict(ps, flip=flip, group=g))
    pperm = list(range(len(pair_prompts)))
    for _ in range(10000):
        prng.shuffle(pperm)
        gs = [pair_prompts[i]["group"] for i in pperm]
        if all(gs[k] != gs[k + 1] for k in range(len(gs) - 1)):
            break
    else:
        raise RuntimeError("could not separate the two orders of a pair in the id order")
    pairs, key_pairs = [], {}
    for k, i in enumerate(pperm):
        ps = pair_prompts[i]
        pid = f"P{k}"
        A, B = (ps["y"], ps["x"]) if ps["flip"] else (ps["x"], ps["y"])
        pairs.append({"id": pid, "prompt": _fill_pair(pid, A[1], B[1], ps["context"])})
        key_pairs[pid] = {"what": ps["what"], "seed": ps["seed"], "A": A[0], "B": B[0], "unit": ps.get("unit"),
                          "context": ps["context"] is not None, "group": ps["group"]}
    for pid, m in key_pairs.items():
        m["partner"] = next(q for q, mm in key_pairs.items() if q != pid and mm["group"] == m["group"])
    pairs.sort(key=lambda p: int(p["id"][1:]))
    # answer prompts
    arows = []
    for e in entries:
        if e["meta"]["kind"] != "real":
            continue
        for t in _flat(e["convs"]):
            if t["section"] == "battery" and t["kind"] in SP.STRICT_KINDS and t["ok"] is not None:
                arows.append({"q": t["you"], "r": t["brain"], "meta": {"seed": e["meta"]["seed"], "week": e["meta"]["week"],
                              "arm": e["meta"]["arm"], "night": e["meta"]["night"], "kind": t["kind"],
                              "item": t["item"], "ok": bool(t["ok"]), "control": None}})
    base = [a for a in arows]
    for ctype, expect in SP.ANSWER_CONTROLS:
        a = base[rng.randrange(len(base))] if base else None
        if a is None:
            break
        q = a["q"]
        if ctype == "refuse":
            r = SP.REFUSE
        elif ctype == "echo":
            r = q
        elif ctype == "subject":
            r = SP.CONTROL_SUBJECT
        else:
            res = by_seed[a["meta"]["seed"]]
            cand = [x for x in base if x["meta"]["item"] in _tellings(res, x["meta"]["night"])
                    and x["meta"]["seed"] == a["meta"]["seed"]]
            if not cand:
                continue
            a = cand[rng.randrange(len(cand))]
            q = a["q"]
            r = second_person(_tellings(res, a["meta"]["night"])[a["meta"]["item"]])
        arows.append({"q": q, "r": r, "meta": dict(a["meta"], control=ctype, expect=expect, ok=None)})
    aperm = list(range(len(arows)))
    rng.shuffle(aperm)
    answers, key_answers = [], {}
    for k, i in enumerate(aperm):
        aid = f"A{k}"
        answers.append({"id": aid, "prompt": _fill_answer(aid, arows[i]["q"], arows[i]["r"])})
        key_answers[aid] = arows[i]["meta"]
    key = {"spec": SP.VERSION, "spec_sha256": spec_sha(), "weeks": list(weeks), "lived_context": lived_context,
           "rng_seed": rng_seed, "seeds": sorted(by_seed), "canaries": canary_log,
           "units": key_units, "pairs": key_pairs, "answers": key_answers}
    return {"units": units, "reader_prompts": prompts, "pairs": pairs, "answer_prompts": answers, "key": key}


def emit(lives: list, d: str, weeks=(1, 3), lived_context: str = "all", rng_seed: int = 0) -> dict:
    out = build(lives, weeks, lived_context, rng_seed)
    os.makedirs(d, exist_ok=True)
    for name in ("units", "reader_prompts", "pairs", "answer_prompts"):
        with open(os.path.join(d, name + ".jsonl"), "w") as fh:
            for row in out[name]:
                fh.write(json.dumps(row) + "\n")
    with open(os.path.join(d, "blind_key.json"), "w") as fh:
        json.dump(out["key"], fh, indent=1)
    return out


def stage_a_summary(out: dict) -> dict:
    k = out["key"]["units"]
    kinds = {}
    for m in k.values():
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    rp = [len(p["prompt"]) for p in out["reader_prompts"]]
    pp = [len(p["prompt"]) for p in out["pairs"]]
    ap = [len(p["prompt"]) for p in out["answer_prompts"]]
    leak = {r["id"]: leaks(r["prompt"]) for r in out["reader_prompts"] + out["pairs"] + out["answer_prompts"] if leaks(r["prompt"])}
    return {"units": len(k), "by_kind": kinds, "pairs": len(pp), "answers": len(ap),
            "reader_chars": {"min": min(rp), "median": sorted(rp)[len(rp) // 2], "max": max(rp)},
            "pair_chars_max": max(pp) if pp else 0, "answer_chars_max": max(ap) if ap else 0,
            "over_60k": [r["id"] for r in out["reader_prompts"] if len(r["prompt"]) > 60000], "leaks": leak,
            "canaries": out["key"]["canaries"]}


# ---------------------------------------------------------------- Stage B
def _load_jsonl(p):
    if not p:
        return None
    rows = []
    for line in open(p):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return {r["id"]: r for r in rows}


def kappa(a: list, b: list):
    """Cohen kappa of two equal-length label lists; None if empty; 1.0 if both constant and equal."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labs = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else None
    return round((po - pe) / (1 - pe), 4)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (s or "").lower())).strip()


def _lcs_len(a: str, b: str) -> int:
    """Longest common substring length (chars)."""
    if not a or not b:
        return 0
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def overlaps(quote: str, planted: str) -> bool:
    """The quote shares at least OVERLAP_MIN characters with the plant (the whole plant if it is shorter): a short
    quote ("I", "Pixel") never matches by being a substring of the plant."""
    q, p = _norm(quote), _norm(planted)
    if not q or not p:
        return False
    return _lcs_len(q, p) >= min(SP.OVERLAP_MIN, len(p))


def _int(x):
    try:
        return int(str(x).strip().strip("[]"))
    except (TypeError, ValueError):
        return None


def canary_hit(defects: list, pl: dict) -> dict:
    """strict: a defect of the planted type, on a planted turn exactly, whose quote overlaps the plant; loose: any
    type, turn within the planted turns +- TURN_TOL (or not given), quote overlapping the plant. The exact turn keeps
    a real defect on a neighbouring turn that shares words with the plant (e.g. "I'm allergic to cashier." next to a
    planted "You're allergic to cashews.") from counting as a strict hit."""
    turns = set(pl["turns"])
    lo, hi = min(turns) - SP.TURN_TOL, max(turns) + SP.TURN_TOL
    strict = loose = False
    for d in defects or []:
        t = _int(d.get("turn"))
        if t is not None and not (lo <= t <= hi):
            continue
        if not overlaps(d.get("quote", ""), pl["text"]):
            continue
        loose = True
        if d.get("type") == pl["defect"] and t in turns:
            strict = True
    return {"strict": strict, "loose": loose}


def _unit_counts(read: dict, meta: dict) -> dict:
    sec = {t["turn"]: t["section"] for t in meta["turns"]}
    by_type = {t: 0 for t in SP.TYPES}
    battery = 0
    for d in read.get("defects") or []:
        ty = d.get("type")
        if ty in by_type:
            by_type[ty] += 1
            if sec.get(_int(d.get("turn"))) == "battery":
                battery += 1
    n_bat = sum(1 for t in meta["turns"] if t["section"] == "battery")
    coh = _int(read.get("coherence"))
    coh = coh if coh is not None and 1 <= coh <= 5 else None
    total = sum(by_type.values())
    return {"total": total, "battery": battery, "by_type": by_type, "coherence": coh,
            "rate": round(total / meta["n_turns"], 4) if meta["n_turns"] else None,
            "n_turns": meta["n_turns"], "n_battery": n_bat,
            "honesty": {v: sum(1 for h in read.get("honesty") or [] if h.get("verdict") == v) for v in SP.HONESTY_VERDICTS}}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def score_reads(key: dict, reads: dict) -> dict:
    """One reader: per-unit counts, aggregates, canary recall, decoys, duplicate spread."""
    U = key["units"]
    per = {uid: _unit_counts(reads[uid], m) for uid, m in U.items() if uid in reads}
    missing = sorted(set(U) - set(reads), key=lambda x: int(x[1:]))
    agg = {}
    for uid, c in per.items():
        m = U[uid]
        if m["kind"] != "real":
            continue
        k = f"{m['arm']}|w{m['week']}"
        a = agg.setdefault(k, {"n": 0, "total": 0, "battery": 0, "by_type": {t: 0 for t in SP.TYPES}, "coh": []})
        a["n"] += 1
        a["total"] += c["total"]
        a["battery"] += c["battery"]
        a["coh"].append(c["coherence"])
        for t, v in c["by_type"].items():
            a["by_type"][t] += v
    for a in agg.values():
        a["coherence_mean"] = _mean(a.pop("coh"))
    per_seed = {}
    for uid, c in per.items():
        m = U[uid]
        if m["kind"] == "real":
            per_seed.setdefault(str(m["seed"]), {})[f"{m['arm']}|w{m['week']}"] = {
                "total": c["total"], "battery": c["battery"], "coherence": c["coherence"], "n_turns": c["n_turns"],
                "rate": c["rate"]}
    can = []
    for uid, m in U.items():
        if m["kind"] == "canary":
            h = canary_hit(reads[uid].get("defects"), m["canary"]) if uid in reads else {"strict": False, "loose": False}
            can.append({"id": uid, "type": m["canary"]["type"], "read": uid in reads, **h})
    dec = []
    for uid, m in U.items():
        if m["kind"] == "decoy":
            c = per.get(uid)
            bo = next((u for u, mm in U.items() if mm["kind"] == "real" and mm["seed"] == m["seed"]
                       and mm["week"] == m["week"] and mm["arm"] == "both-off"), None)
            dec.append({"id": uid, "decoy": m["decoy"], "seed": m["seed"], "defects": c["total"] if c else None,
                        "battery_defects": c["battery"] if c else None, "coherence": c["coherence"] if c else None,
                        "bothoff_id": bo, "bothoff_defects": per[bo]["total"] if bo in per else None,
                        "bothoff_coherence": per[bo]["coherence"] if bo in per else None,
                        "flagged": bool(c and c["total"] > 0)})
    dups = []
    for uid, m in U.items():
        if m["kind"] == "duplicate" and uid in per and m.get("dup_partner") in per:
            a, b = per[uid], per[m["dup_partner"]]
            dc = None if None in (a["coherence"], b["coherence"]) else abs(a["coherence"] - b["coherence"])
            dups.append({"ids": [m["dup_partner"], uid], "defect_spread": abs(a["total"] - b["total"]),
                         "rate_spread": abs(a["rate"] - b["rate"]),
                         "battery_spread": abs(a["battery"] - b["battery"]), "coherence_spread": dc})
    mde = max((d["defect_spread"] for d in dups), default=None)
    mde_rate = max((d["rate_spread"] for d in dups), default=None)
    return {"per_unit": per, "missing": missing, "by_arm_week": agg, "per_seed": per_seed, "canaries": can,
            "canary_strict": sum(c["strict"] for c in can), "canary_loose": sum(c["loose"] for c in can),
            "canary_n": len(can), "decoys": dec, "duplicates": dups, "mde": mde, "mde_rate": mde_rate,
            "coherence_mde": max((d["coherence_spread"] for d in dups if d["coherence_spread"] is not None), default=None)}


def _ranks(x):
    """Average ranks (ties share the mean rank)."""
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list, b: list):
    """Spearman rank correlation (Pearson on average ranks); None if fewer than 2 points or a constant side."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return round(cov / (va * vb) ** 0.5, 4) if va > 0 and vb > 0 else None


def reader_kappa(key: dict, r1: dict, r2: dict) -> dict:
    ids = sorted(set(r1) & set(r2) & set(key["units"]), key=lambda x: int(x[1:]))

    def present(r, uid, t=None):
        ds = r[uid].get("defects") or []
        return int(any(t is None or d.get("type") == t for d in ds))
    out = {"n": len(ids), "any_defect": kappa([present(r1, u) for u in ids], [present(r2, u) for u in ids])}
    out["by_type"] = {t: kappa([present(r1, u, t) for u in ids], [present(r2, u, t) for u in ids]) for t in SP.TYPES}
    co = [(r1[u].get("coherence"), r2[u].get("coherence")) for u in ids]
    co = [(a, b) for a, b in co if a is not None and b is not None]
    out["coherence_kappa"] = kappa([a for a, _ in co], [b for _, b in co])
    out["coherence_mean_abs_diff"] = _mean([abs(a - b) for a, b in co])
    # count-level agreement over real units only (canaries and decoys excluded)
    real = [u for u in ids if key["units"][u]["kind"] == "real"]
    c1 = {u: _unit_counts(r1[u], key["units"][u]) for u in real}
    c2 = {u: _unit_counts(r2[u], key["units"][u]) for u in real}
    out["spearman_n"] = len(real)
    out["spearman_total"] = spearman([c1[u]["total"] for u in real], [c2[u]["total"] for u in real])
    out["spearman_battery"] = spearman([c1[u]["battery"] for u in real], [c2[u]["battery"] for u in real])
    out["spearman_by_type"] = {t: spearman([c1[u]["by_type"][t] for u in real], [c2[u]["by_type"][t] for u in real])
                               for t in SP.TYPES}
    return out


def score_pairs(key: dict, pr: dict) -> dict:
    """Each pair is read in both A/B orders. A group has a winner (the underlying side) only if both orders name the
    same side; two ties give "tie" (consistent); anything else is "tie" and counted inconsistent (position bias)."""
    P = key["pairs"]
    groups = {}
    for pid, m in P.items():
        if pid not in pr:
            continue
        ch = str(pr[pid].get("choice", "")).strip().strip('"').upper()
        win = m["A"] if ch == "A" else m["B"] if ch == "B" else "tie"
        groups.setdefault(m["group"], []).append({"id": pid, "what": m["what"], "seed": m["seed"], "unit": m.get("unit"),
                                                  "choice": ch if ch in ("A", "B") else "TIE", "winner": win})
    rows = []
    for g, rs in sorted(groups.items()):
        if len(rs) != 2:
            continue    # a group read in one order only has no result
        w = [r["winner"] for r in rs]
        consistent = w[0] == w[1]
        same_pos = rs[0]["choice"] == rs[1]["choice"] and rs[0]["choice"] in ("A", "B")
        rows.append({"group": g, "ids": [r["id"] for r in rs], "what": rs[0]["what"], "seed": rs[0]["seed"],
                     "unit": rs[0]["unit"], "winners": w, "consistent": consistent, "same_position": same_pos,
                     "winner": w[0] if consistent else "tie"})
    by = {}
    for r in rows:
        by.setdefault(r["what"], []).append(r)
    ib = {str(r["seed"]): r["winner"] for r in by.get("intact-vs-bothoff", [])}
    ww = {str(r["seed"]): r["winner"] for r in by.get("w3-vs-w1", [])}
    n = len(rows)
    inc = sum(1 for r in rows if not r["consistent"])
    return {"rows": rows, "n_pairs": n, "inconsistent": inc, "consistency_rate": round(1 - inc / n, 3) if n else None,
            "position_bias_rate": round(sum(r["same_position"] for r in rows) / n, 3) if n else None,
            "intact_vs_bothoff": ib, "intact_preferred_seeds": sum(1 for v in ib.values() if v == "intact"),
            "intact_vs_bothoff_inconsistent": sum(1 for r in by.get("intact-vs-bothoff", []) if not r["consistent"]),
            "w3_vs_w1": ww, "w3_preferred_seeds": sum(1 for v in ww.values() if v == f"intact-w{max(key['weeks'])}"),
            "decoys": {r["unit"]: r["winner"] for r in by.get("decoy-vs-bothoff", [])}}


def score_answers(key: dict, a1: dict, a2: dict | None = None) -> dict:
    A = key["answers"]

    def yes(r, aid):
        return None if r is None or aid not in r else str(r[aid].get("verdict", "")).strip().lower() == "yes"
    rows, ctrl = [], []
    for aid, m in A.items():
        v1, v2 = yes(a1, aid), yes(a2, aid)
        if m.get("control"):
            ctrl.append({"id": aid, "control": m["control"], "expect": m["expect"], "v1": v1, "v2": v2,
                         "ok1": None if v1 is None else (v1 == (m["expect"] == "yes")),
                         "ok2": None if v2 is None else (v2 == (m["expect"] == "yes"))})
            continue
        rows.append({"id": aid, **{k: m[k] for k in ("seed", "week", "arm", "kind", "item", "ok")}, "v1": v1, "v2": v2,
                     "right": None if v1 is None else bool(m["ok"] and v1),
                     "right2": None if v2 is None else bool(m["ok"] and v2)})
    agg = {}
    for r in rows:
        for k in (f"{r['arm']}|all", f"{r['arm']}|{r['kind']}", f"{r['arm']}|w{r['week']}"):
            a = agg.setdefault(k, {"n": 0, "strict": 0, "verdict_yes": 0, "right": 0, "disagree": 0, "veto": 0, "read": 0})
            a["n"] += 1
            a["strict"] += r["ok"]
            if r["v1"] is not None:
                a["read"] += 1
                a["verdict_yes"] += r["v1"]
                a["right"] += r["right"]
                a["disagree"] += (r["ok"] != r["v1"])
                a["veto"] += (r["ok"] and not r["v1"])
    for a in agg.values():
        a["disagree_rate"] = round(a["disagree"] / a["read"], 3) if a["read"] else None
    both = [r for r in rows if r["v1"] is not None and r["v2"] is not None]
    return {"rows": rows, "controls": ctrl, "by": agg,
            "kappa": kappa([r["v1"] for r in both], [r["v2"] for r in both]) if both else None,
            "reader_disagreements": [r["id"] for r in both if r["v1"] != r["v2"]],
            "strict_reader_disagree_rate": round(sum(r["ok"] != r["v1"] for r in rows if r["v1"] is not None)
                                                 / max(1, sum(r["v1"] is not None for r in rows)), 3)}


def _need(n: int) -> int:
    """SEEDS_BAR of 6, scaled to n seeds (DEV runs have fewer than 6): ceil(4 n / 6)."""
    return -(-SP.SEEDS_BAR * n // 6)


def _bars(key, R, PW):
    """Section-7 validity bars per reader + the whole-transcript bars (DEV/Stage 0 report-only)."""
    bars = []
    for name, r in R.items():
        bars.append({"bar": f"{name}: canary recall strict >= {SP.CANARY_BAR}/6",
                     "value": f"{r['canary_strict']}/{r['canary_n']}", "pass": r["canary_strict"] >= SP.CANARY_BAR})
        bars.append({"bar": f"{name}: canary recall loose >= {SP.CANARY_BAR}/6",
                     "value": f"{r['canary_loose']}/{r['canary_n']}", "pass": r["canary_loose"] >= SP.CANARY_BAR})
        pw = PW.get(name.replace("reader", "pairs")) or PW.get("pairs1")     # reader 2 falls back to pairs 1
        ok_all = True
        vals = []
        for d in r["decoys"]:
            won = pw["decoys"].get(d["id"]) if pw else None
            at_or_below = won is not None and not won.startswith("decoy")
            ok = d["flagged"] and at_or_below
            ok_all &= ok
            vals.append(f"{d['decoy']}: flagged {d['flagged']}, pairwise {won}")
        bars.append({"bar": f"{name}: decoys flagged and rated at or below both-off", "value": "; ".join(vals),
                     "pass": ok_all and bool(r["decoys"])})
        bars.append({"bar": f"{name}: duplicate spread (MDE, defects)", "value": r["mde"], "pass": None})
    r1 = R.get("reader1")
    p1 = PW.get("pairs1")
    if p1:
        n = len(p1["intact_vs_bothoff"])
        bars.append({"bar": f"intact preferred to both-off, last week, >= {SP.SEEDS_BAR}/6 seeds (pairs1; scaled to "
                            f"{_need(n)}/{n}; a seed counts only when both A/B orders name intact = consistent)",
                     "value": f"{p1['intact_preferred_seeds']}/{n}; inconsistent (tie) {p1['intact_vs_bothoff_inconsistent']}/{n}; "
                              f"all pairs consistent {p1['n_pairs'] - p1['inconsistent']}/{p1['n_pairs']}",
                     "pass": p1["intact_preferred_seeds"] >= _need(n)})
    if r1:
        w1, w3 = min(key["weeks"]), max(key["weeks"])
        n = ok = indist = 0
        # per turn: the two weeks' units differ in length (DEV: 72 vs 62 turns), so raw counts favour the shorter week
        for s, d in r1["per_seed"].items():
            a, b = d.get(f"intact|w{w1}"), d.get(f"intact|w{w3}")
            if a and b:
                n += 1
                ok += b["rate"] <= a["rate"]
                indist += r1["mde_rate"] is not None and abs(b["rate"] - a["rate"]) <= r1["mde_rate"]
        bars.append({"bar": f"week-{w3} defects per turn not above week {w1}, >= {SP.SEEDS_BAR}/6 seeds (reader1; scaled "
                            f"to {_need(n)}/{n}; within the duplicate rate spread = not distinguishable)",
                     "value": f"{ok}/{n}; within MDE {indist}/{n}", "pass": ok >= _need(n)})
    return bars


def score(key: dict, reads: dict, reads2=None, pairs=None, pairs2=None, answers=None, answers2=None) -> dict:
    load_spec()
    if key.get("spec_sha256") != spec_sha():
        raise RuntimeError("blind key was emitted under a different read spec")
    R = {"reader1": score_reads(key, reads)}
    if reads2:
        R["reader2"] = score_reads(key, reads2)
    PW = {}
    if pairs:
        PW["pairs1"] = score_pairs(key, pairs)
    if pairs2:
        PW["pairs2"] = score_pairs(key, pairs2)
    out = {"spec": SP.VERSION, "spec_sha256": spec_sha(), "seeds": key["seeds"], "weeks": key["weeks"],
           "lived_context": key["lived_context"], "readers": R, "pairwise": PW}
    if reads2:
        out["kappa"] = reader_kappa(key, reads, reads2)
    if pairs and pairs2:
        ids = sorted(set(pairs) & set(pairs2))
        out["pair_kappa"] = kappa([str(pairs[i].get("choice")) for i in ids], [str(pairs2[i].get("choice")) for i in ids])
    if answers:
        out["answers"] = score_answers(key, answers, answers2)
    out["bars"] = _bars(key, R, PW)
    return out


def _mark(p):
    return "n/a" if p is None else ("PASS" if p else "FAIL")


def to_md(r: dict) -> str:
    L = [f"# Blind whole-transcript read ({r['spec']}, spec sha256 {r['spec_sha256'][:12]})\n",
         f"Seeds {r['seeds']}; weeks {r['weeks']}; lived context: {r['lived_context']}. Stage-0/DEV bars are report-only.\n",
         "## Instrument validity (design section 7)\n", "| bar | value | mark |", "|---|---|---|"]
    L += [f"| {b['bar']} | {b['value']} | {_mark(b['pass'])} |" for b in r["bars"]]
    if "kappa" in r:
        k = r["kappa"]
        L += ["", f"Cohen kappa (n={k['n']} units): any defect {k['any_defect']}; coherence {k['coherence_kappa']} "
              f"(mean abs diff {k['coherence_mean_abs_diff']})", "",
              "| type | kappa | Spearman (per-unit count) |", "|---|---|---|"]
        L += [f"| {t} | {v} | {k['spearman_by_type'][t]} |" for t, v in k["by_type"].items()]
        L += ["", f"Spearman of per-unit defect counts (n={k['spearman_n']} real units, canaries and decoys excluded): "
              f"total {k['spearman_total']}, battery {k['spearman_battery']} (bar to show, report-only: total >= 0.6 -> "
              f"{'PASS' if (k['spearman_total'] or 0) >= 0.6 else 'FAIL'})"]
    if "pair_kappa" in r:
        L.append(f"\nPairwise kappa: {r['pair_kappa']}")
    for name, rr in r["readers"].items():
        L += ["", f"## {name}", "", f"missing reads: {len(rr['missing'])}; MDE (defects) {rr['mde']}, coherence {rr['coherence_mde']}",
              "", "| arm, week | units | defects | battery defects | coherence | " + " | ".join(SP.TYPES) + " |",
              "|---|---|---|---|---|" + "---|" * len(SP.TYPES)]
        for k, a in sorted(rr["by_arm_week"].items()):
            L.append(f"| {k} | {a['n']} | {a['total']} | {a['battery']} | {a['coherence_mean']} | "
                     + " | ".join(str(a["by_type"][t]) for t in SP.TYPES) + " |")
        L += ["", "| canary | id | strict | loose |", "|---|---|---|---|"]
        L += [f"| {c['type']} | {c['id']} | {c['strict']} | {c['loose']} |" for c in rr["canaries"]]
        L += ["", "| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |", "|---|---|---|---|---|---|---|"]
        L += [f"| {d['decoy']} | {d['id']} | {d['defects']} | {d['coherence']} | {d['bothoff_defects']} | "
              f"{d['bothoff_coherence']} | {d['flagged']} |" for d in rr["decoys"]]
        L += ["", "| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |",
              "|---|---|---|---|---|"]
        L += [f"| {'/'.join(d['ids'])} | {d['defect_spread']} | {d['rate_spread']} | {d['battery_spread']} | "
              f"{d['coherence_spread']} |" for d in rr["duplicates"]]
    for name, p in r["pairwise"].items():
        L += ["", f"## {name}", "", f"intact vs both-off (last week) winners per seed: {p['intact_vs_bothoff']}",
              f"last week vs first week (intact): {p['w3_vs_w1']}", f"decoys vs both-off: {p['decoys']}",
              f"pairs consistent over both A/B orders: {p['n_pairs'] - p['inconsistent']}/{p['n_pairs']} "
              f"(consistency rate {p['consistency_rate']}; same-position rate {p['position_bias_rate']})",
              f"inconsistent pairs: {[(x['what'], x['seed'], x['winners']) for x in p['rows'] if not x['consistent']]}"]
    if "answers" in r:
        a = r["answers"]
        L += ["", "## RIGHT = strict ok AND reader verdict yes (section 5)", "",
              f"strict vs reader-1 disagreement rate {a['strict_reader_disagree_rate']}; reader kappa {a['kappa']}; "
              f"reader disagreements for adjudication: {len(a['reader_disagreements'])}", "",
              "| arm / cut | n | strict ok | verdict yes | RIGHT | strict-reader disagree | veto (strict ok, verdict no) |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['strict']} | {v['verdict_yes']} | {v['right']} | {v['disagree_rate']} | {v['veto']} |"
              for k, v in sorted(a["by"].items())]
        L += ["", "| control | expect | reader 1 | reader 2 |", "|---|---|---|---|"]
        L += [f"| {c['control']} | {c['expect']} | {c['ok1']} | {c['ok2']} |" for c in a["controls"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- G9 cross-arm pairwise (Stage-2 test-2 prereg section 5)
# Every existing mode above is unchanged; this mode only adds functions. The prompt is the frozen spec.PAIR_PROMPT with
# no shared context (each side is a whole unit: its own lived days of week 3 + its own battery), as the existing
# w3-vs-w1 pairs: no new reader wording is needed, so life_read_spec.py and its sha256 stay as they are.
CROSS_ARM, CROSS_B = "arm", "B"


def _cross_units(lives: list, week: int, lived_context: str) -> dict:
    return {u["seed"]: u for u in collect(lives, (week,), lived_context) if u["arm"] == "intact"}


CROSS_MAX_PROMPT = 60000
CROSS_MAX_CANARIES_PER_UNIT = 2
# Positive-discrimination gate (cross v2, stated before read 2): the section-7 "at or below both-off" rule lets a tie
# pass, so a reader answering all A, all B or all tie would pass every validity check. A valid reader must also make
# decoys STRICTLY lose to both-off in both orders on >= _need(n) seeds (the SEEDS_BAR scaling, 2 of 3 on DEV), and pick
# the same position in both orders on at most 1/3 of the non-duplicate groups (duplicates are excluded: a tie there is
# the right answer and is usually read as the same position twice).
CROSS_MAX_POSITION_BIAS = 1 / 3


def build_cross(arm_lives: list, b_lives: list, week: int = 3, lived_context: str = "all", rng_seed: int = 0,
                id_prefix: str = "C") -> dict:
    """G9 material: per seed, the arm's intact unit vs B's intact unit (same seed, week `week`, whole units), plus the
    validity pairs: per seed one decoy (the arm unit with every reply replaced, spec.DECOY_TYPES rotated over seeds) vs
    the SAME life's both-off unit of that seed and week, in the section-7 form (battery vs battery, the unit's lived
    days as shared context), which the decoy must not beat; per seed one duplicate (the same unit both sides, arm's and
    B's alternating over seeds; report-only noise); per batch one canary pair per spec.CANARY_TYPES (a real intact unit
    vs its copy with one planted defect; every type must plant, at most two per unit, arm units first), whose copy must
    not win. Every pair is emitted in both A/B orders with two ids never adjacent in the id order; ids <id_prefix>N
    (deterministic for the same inputs). Raises on duplicate seeds, a missing unit, a leak or a prompt > 60k chars."""
    load_spec()
    rng = random.Random(f"life-read-cross:{rng_seed}")
    for tag, lives in (("arm", arm_lives), ("B", b_lives)):
        ss = [r["seed"] for r in lives]
        if len(ss) != len(set(ss)):
            raise ValueError(f"duplicate seeds in the {tag} lives: {sorted(ss)}")
    by_a, by_b = {r["seed"]: r for r in arm_lives}, {r["seed"]: r for r in b_lives}
    if set(by_a) != set(by_b):
        raise ValueError(f"arm and B lives have different seeds: {sorted(by_a)} vs {sorted(by_b)}")
    ua, ub = _cross_units(arm_lives, week, lived_context), _cross_units(b_lives, week, lived_context)
    bo = {u["seed"]: u for u in collect(arm_lives, (week,), lived_context) if u["arm"] == "both-off"}
    seeds = sorted(by_a)
    missing = [s for s in seeds if s not in ua or s not in ub]
    if missing:
        raise ValueError(f"no intact week-{week} unit for seeds {missing}")
    missing = [s for s in seeds if s not in bo]
    if missing:
        raise ValueError(f"no both-off week-{week} unit in the arm lives for seeds {missing}")
    specs, user_match = [], {}
    for i, s in enumerate(seeds):
        ta, tb = render(ua[s]["convs"])[0], render(ub[s]["convs"])[0]
        user_match[str(s)] = [t["you"] for t in _flat(ua[s]["convs"])] == [t["you"] for t in _flat(ub[s]["convs"])]
        specs.append({"what": "cross", "seed": s, "x": (CROSS_ARM, ta), "y": (CROSS_B, tb), "context": None})
        dtype = SP.DECOY_TYPES[i % len(SP.DECOY_TYPES)]
        lived = _sections(ua[s]["convs"], "lived")
        specs.append({"what": "decoy", "seed": s, "decoy": dtype,
                      "x": ("decoy:" + dtype, render(_sections(decoy(ua[s], dtype, by_a[s]), "battery"))[0]),
                      "y": ("both-off", render(_sections(bo[s]["convs"], "battery"))[0]),
                      "context": render(lived)[0] if lived else None})
        dup_of = CROSS_ARM if i % 2 == 0 else CROSS_B
        t = ta if dup_of == CROSS_ARM else tb
        specs.append({"what": "dup", "seed": s, "dup_of": dup_of, "x": ("left", t), "y": ("right", t), "context": None})
    cands = [(CROSS_ARM, ua[s], by_a[s]) for s in seeds] + [(CROSS_B, ub[s], by_b[s]) for s in seeds]
    na = len(seeds)
    canary_log, load = [], {}
    for ci, ctype in enumerate(SP.CANARY_TYPES):
        order = [(ci + k) % na for k in range(na)] + [na + (ci + k) % na for k in range(na)]   # arm first, then B
        done = None
        for j in order:
            side, u, res = cands[j]
            if load.get((side, u["seed"]), 0) >= CROSS_MAX_CANARIES_PER_UNIT:
                continue
            got = plant(u, ctype, res, rng)
            if got:
                done = (side, u, got)
                break
        if not done:
            raise ValueError(f"canary {ctype} has no site on any unit (at most {CROSS_MAX_CANARIES_PER_UNIT} per unit)")
        side, u, (convs, pl) = done
        load[(side, u["seed"])] = load.get((side, u["seed"]), 0) + 1
        canary_log.append({"type": ctype, "planted": True, "seed": u["seed"], "source": side, "plant": pl})
        specs.append({"what": "canary", "seed": u["seed"], "canary": ctype, "source": side,
                      "x": ("source", render(u["convs"])[0]), "y": ("canary:" + ctype, render(convs)[0]), "context": None})
    prng = random.Random(f"cross-pairs-{rng_seed}")
    pp = [dict(ps, flip=flip, group=g) for g, ps in enumerate(specs) for flip in (False, True)]
    perm = list(range(len(pp)))
    for _ in range(10000):
        prng.shuffle(perm)
        gs = [pp[i]["group"] for i in perm]
        if all(gs[k] != gs[k + 1] for k in range(len(gs) - 1)):
            break
    else:
        raise RuntimeError("could not separate the two orders of a pair in the id order")
    pairs, key_pairs = [], {}
    for k, i in enumerate(perm):
        ps = pp[i]
        pid = f"{id_prefix}{k}"
        A, B = (ps["y"], ps["x"]) if ps["flip"] else (ps["x"], ps["y"])
        pairs.append({"id": pid, "prompt": _fill_pair(pid, A[1], B[1], ps["context"])})
        key_pairs[pid] = {"what": ps["what"], "seed": ps["seed"], "A": A[0], "B": B[0], "group": ps["group"],
                          **{f: ps[f] for f in ("decoy", "dup_of", "canary", "source") if f in ps}}
    bad = {p["id"]: leaks(p["prompt"]) for p in pairs if leaks(p["prompt"])}
    if bad:
        raise ValueError(f"leaks in emitted prompts: {bad}")
    big = [p["id"] for p in pairs if len(p["prompt"]) > CROSS_MAX_PROMPT]
    if big:
        raise ValueError(f"prompts over {CROSS_MAX_PROMPT} chars: {big}")
    for pid, m in key_pairs.items():
        m["partner"] = next(q for q, mm in key_pairs.items() if q != pid and mm["group"] == m["group"])
    key = {"mode": "cross", "cross_version": 2, "spec": SP.VERSION, "spec_sha256": spec_sha(), "week": week,
           "lived_context": lived_context, "rng_seed": rng_seed, "id_prefix": id_prefix, "seeds": seeds,
           "decoy_reference": "both-off (arm life, same seed and week)", "user_turns_match": user_match,
           "canaries": canary_log,
           "unit_chars": {str(s): {CROSS_ARM: len(render(ua[s]["convs"])[0]), CROSS_B: len(render(ub[s]["convs"])[0])}
                          for s in seeds},
           "pairs": key_pairs}
    return {"pairs": pairs, "key": key}


def emit_cross(arm_lives: list, b_lives: list, d: str, week: int = 3, lived_context: str = "all",
               rng_seed: int = 0, id_prefix: str = "C") -> dict:
    out = build_cross(arm_lives, b_lives, week, lived_context, rng_seed, id_prefix)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cross_pairs.jsonl"), "w") as fh:
        for row in out["pairs"]:
            fh.write(json.dumps(row) + "\n")
    with open(os.path.join(d, "cross_key.json"), "w") as fh:
        json.dump(out["key"], fh, indent=1)
    return out


def cross_summary(out: dict) -> dict:
    k = out["key"]
    by = {}
    for m in k["pairs"].values():
        by[m["what"]] = by.get(m["what"], 0) + 1
    pc = [len(p["prompt"]) for p in out["pairs"]]
    per_unit = {}
    for c in k["canaries"]:
        per_unit[f"{c['source']}:{c['seed']}"] = per_unit.get(f"{c['source']}:{c['seed']}", 0) + 1
    return {"prompts": len(pc), "by_what": by, "seeds": k["seeds"], "user_turns_match": k["user_turns_match"],
            "canaries_planted": [c["type"] for c in k["canaries"] if c["planted"]], "canaries_per_unit": per_unit,
            "prompt_chars": {"min": min(pc), "median": sorted(pc)[len(pc) // 2], "max": max(pc)},
            "ids": f"{out['pairs'][0]['id']}..", "decoy_reference": k["decoy_reference"]}


CROSS_UNREAD = "UNREAD"


def norm_cross_choice(ch) -> str:
    """A / B / TIE after normalising case, whitespace, quotes, trailing punctuation and a 'transcript ' prefix;
    anything else is UNREAD."""
    if not isinstance(ch, str):
        return CROSS_UNREAD
    s = re.sub(r"\s+", " ", ch).strip().strip("\"'`").strip().rstrip(".!").strip().lower()
    s = re.sub(r"^transcript ", "", s)
    return {"a": "A", "b": "B", "tie": "TIE"}.get(s, CROSS_UNREAD)


def load_cross_reads(p):
    """A reads file -> {id: row}; an id that appears more than once is kept as unread (choice None, duplicate True)."""
    if not p:
        return None
    out = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pid = r.get("id")
        if pid in out:
            out[pid] = {"id": pid, "choice": None, "duplicate": True}
        else:
            out[pid] = r
    return out


def _cross_rows(key: dict, pr: dict) -> tuple[list, dict]:
    """One row per pair group with both orders validly read: winner = the underlying side if both orders name it, else
    tie. Also the read counts (valid / unread choice / duplicate id / missing / extra)."""
    groups, cnt = {}, {"valid": 0, "bad_choice": 0, "duplicate_id": 0, "missing": 0,
                       "extra": len(set(pr) - set(key["pairs"]))}
    for pid, m in key["pairs"].items():
        if pid not in pr:
            cnt["missing"] += 1
            continue
        if pr[pid].get("duplicate"):
            cnt["duplicate_id"] += 1
            continue
        ch = norm_cross_choice(pr[pid].get("choice"))
        if ch == CROSS_UNREAD:
            cnt["bad_choice"] += 1
            continue
        cnt["valid"] += 1
        groups.setdefault(m["group"], []).append(
            {"id": pid, "m": m, "choice": ch, "winner": m["A"] if ch == "A" else m["B"] if ch == "B" else "tie"})
    rows = []
    for g, rs in sorted(groups.items()):
        if len(rs) != 2:
            continue
        w = [r["winner"] for r in rs]
        m = rs[0]["m"]
        rows.append({"group": g, "ids": [r["id"] for r in rs], "what": m["what"], "seed": m["seed"],
                     **{f: m[f] for f in ("decoy", "dup_of", "canary", "source") if f in m},
                     "choices": [r["choice"] for r in rs], "winners": w, "consistent": w[0] == w[1],
                     "same_position": rs[0]["choice"] == rs[1]["choice"] and rs[0]["choice"] in ("A", "B"),
                     "winner": w[0] if w[0] == w[1] else "tie"})
    return rows, cnt


def score_cross_reader(key: dict, pr: dict) -> dict:
    """One reader: per-seed G9 winner (arm / B / tie; None if the seed's user turns differ), order consistency, decoy
    (must not beat both-off) and canary validity, duplicates report-only, G9 bar.
    Valid = complete, no decoy beats both-off, decoys strictly lose to both-off on >= _need(n) seeds, same-position rate
    over non-duplicate groups <= CROSS_MAX_POSITION_BIAS, and canary copy not preferred >= CANARY_BAR/6. The canary
    check counts a tie as a pass, so it only catches a reader who PREFERS the defect; source-preferred is report-only
    (no unit read in cross mode, so section-7 "flagged" and canary recall are not implemented here)."""
    rows, cnt = _cross_rows(key, pr or {})
    seeds = [str(s) for s in key["seeds"]]
    um = key.get("user_turns_match", {})
    by = {}
    for r in rows:
        by.setdefault(r["what"], []).append(r)
    win = {str(r["seed"]): r["winner"] for r in by.get("cross", [])}
    undefined = [s for s in seeds if um.get(s) is False]
    per_seed = {}
    for s in seeds:
        dec = [r for r in by.get("decoy", []) if str(r["seed"]) == s]
        dup = [r for r in by.get("dup", []) if str(r["seed"]) == s]
        cr = next((r for r in by.get("cross", []) if str(r["seed"]) == s), None)
        per_seed[s] = {"winner": None if s in undefined else win.get(s), "user_turns_match": um.get(s),
                       "consistent": cr["consistent"] if cr else None, "winners_by_order": cr["winners"] if cr else None,
                       "decoy": dec[0]["decoy"] if dec else None, "decoy_winner": dec[0]["winner"] if dec else None,
                       "decoy_ok": (not dec[0]["winner"].startswith("decoy")) if dec else None,
                       "dup_of": dup[0]["dup_of"] if dup else None, "dup_winner": dup[0]["winner"] if dup else None,
                       "dup_choices": dup[0]["choices"] if dup else None}
    n_groups = len({m["group"] for m in key["pairs"].values()})
    n = len(rows)
    inc = sum(1 for r in rows if not r["consistent"])
    decs = by.get("decoy", [])
    dec_ok = sum(1 for r in decs if not r["winner"].startswith("decoy"))
    dec_strict = sum(1 for r in decs if r["winner"] == "both-off")
    nd = [r for r in rows if r["what"] != "dup"]
    pos_nd = round(sum(r["same_position"] for r in nd) / len(nd), 3) if nd else None
    dups = by.get("dup", [])
    can = [{"type": r["canary"], "seed": r["seed"], "source_side": r["source"], "winner": r["winner"],
            "source_preferred": r["winner"] == "source", "copy_not_preferred": not r["winner"].startswith("canary")}
           for r in by.get("canary", [])]
    can_planted = sum(1 for c in key["canaries"] if c["planted"])
    can_need = -(-SP.CANARY_BAR * can_planted // 6)
    can_np = sum(c["copy_not_preferred"] for c in can)
    complete = n == n_groups and cnt["valid"] == len(key["pairs"])
    need = _need(len(seeds))
    discriminates = dec_strict >= need and pos_nd is not None and pos_nd <= CROSS_MAX_POSITION_BIAS
    valid = (complete and len(decs) == len(seeds) and dec_ok == len(decs) and can_np >= can_need
             and discriminates)
    arm_or_tie = sum(1 for s in seeds if per_seed[s]["winner"] in (CROSS_ARM, "tie"))
    if not valid:
        g9 = None
    elif arm_or_tie >= need:
        g9 = True
    elif arm_or_tie + len(undefined) >= need:
        g9 = None       # the undefined seeds decide the bar
    else:
        g9 = False
    return {"rows": rows, "per_seed": per_seed, "n_groups": n_groups, "n_read": n, "complete": complete,
            "read_counts": cnt, "undefined_seeds": undefined,
            "inconsistent": inc, "consistency_rate": round(1 - inc / n, 3) if n else None,
            "position_bias_rate": round(sum(r["same_position"] for r in rows) / n, 3) if n else None,
            "cross_consistency": (f"{sum(1 for r in by.get('cross', []) if r['consistent'])}/{len(by.get('cross', []))}"),
            "decoys_ok": f"{dec_ok}/{len(decs)}",
            "decoys_strict": f"{dec_strict}/{len(decs)} (need {need})",
            "position_bias_nondup": pos_nd, "position_bias_cap": round(CROSS_MAX_POSITION_BIAS, 3),
            "discriminates": discriminates,
            "dups_tie": f"{sum(1 for r in dups if r['winner'] == 'tie')}/{len(dups)}",
            "dup_choices": {str(r["seed"]): r["choices"] for r in dups}, "canaries": can,
            "canary_copy_not_preferred": f"{can_np}/{len(can)} (need {can_need} of {can_planted} planted)",
            "canary_source_preferred": f"{sum(c['source_preferred'] for c in can)}/{len(can)}",
            "valid": valid, "arm_preferred": sum(1 for s in seeds if per_seed[s]["winner"] == CROSS_ARM),
            "b_preferred": sum(1 for s in seeds if per_seed[s]["winner"] == CROSS_B),
            "ties": sum(1 for s in seeds if per_seed[s]["winner"] == "tie"),
            "arm_or_tie": arm_or_tie, "need": need, "n_seeds": len(seeds), "g9_pass": g9}


def score_cross(key: dict, pairs: dict, pairs2: dict | None = None) -> dict:
    """G9 needs exactly two readers: UNDEFINED unless pairs2 is supplied and non-empty. If either reader is invalid
    (failed decoy, discrimination or canary check, unread or missing pairs) G9 is UNDEFINED: a half-invalid instrument
    carries no verdict. With both readers valid: PASS if both pass, FAIL if either fails, else UNDEFINED."""
    load_spec()
    if key.get("mode") != "cross":
        raise RuntimeError("not a cross key")
    if key.get("spec_sha256") != spec_sha():
        raise RuntimeError("cross key was emitted under a different read spec")
    if key.get("cross_version") != 2:
        raise RuntimeError("cross key from cross mode v1 (decoys vs B); re-emit")
    R = {"pairs1": score_cross_reader(key, pairs)}
    two = bool(pairs2)
    if two:
        R["pairs2"] = score_cross_reader(key, pairs2)
    vals = [r["g9_pass"] for r in R.values()]
    if not two or not all(r["valid"] for r in R.values()):
        verdict = "UNDEFINED"
    else:
        verdict = "FAIL" if any(v is False for v in vals) else "PASS" if all(v is True for v in vals) else "UNDEFINED"
    comb = None
    if two:
        a, b = R["pairs1"]["per_seed"], R["pairs2"]["per_seed"]
        comb = {s: (a[s]["winner"] if a[s]["winner"] == b[s]["winner"] else "tie") for s in a}
    out = {"mode": "cross", "spec": SP.VERSION, "spec_sha256": spec_sha(), "seeds": key["seeds"], "week": key["week"],
           "lived_context": key["lived_context"], "user_turns_match": key["user_turns_match"], "readers": R,
           "n_readers": len(R), "readers_combined": comb, "g9": verdict,
           "g9_bar": f"arm preferred or tie on >= {SP.SEEDS_BAR}/6 seeds (scaled ceil(4n/6)), per reader; two readers, "
                     "both valid (else UNDEFINED)"}
    if two:
        ids = sorted(set(pairs) & set(pairs2))
        out["pair_kappa"] = kappa([norm_cross_choice(pairs[i].get("choice")) for i in ids],
                                  [norm_cross_choice(pairs2[i].get("choice")) for i in ids])
    return out


def cross_to_md(r: dict) -> str:
    L = [f"# G9 cross-arm pairwise read ({r['spec']}, spec sha256 {r['spec_sha256'][:12]})\n",
         f"Seeds {r['seeds']}; week {r['week']}; lived context {r['lived_context']}; user turns identical arm vs B: "
         f"{r['user_turns_match']}.\n",
         f"**G9: {r['g9']}** ({r['g9_bar']}; every reader valid and passing; readers: {r['n_readers']}).\n",
         "| reader | valid | complete | reads valid / bad / dup id / missing | arm / B / tie | arm or tie | need | G9 | "
         "decoys not above both-off | decoys strictly below both-off | same-position non-dup (cap) | "
         "dups tie (report-only) | canary copy not preferred | canary source preferred (report-only) | "
         "order consistency (all / cross) | same-position |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in r["readers"].items():
        c = x["read_counts"]
        L.append(f"| {name} | {x['valid']} | {x['complete']} | {c['valid']} / {c['bad_choice']} / {c['duplicate_id']} / "
                 f"{c['missing']} | {x['arm_preferred']} / {x['b_preferred']} / {x['ties']} | "
                 f"{x['arm_or_tie']}/{x['n_seeds']} | {x['need']} | {_mark(x['g9_pass'])} | {x['decoys_ok']} | "
                 f"{x['decoys_strict']} | {x['position_bias_nondup']} ({x['position_bias_cap']}) | "
                 f"{x['dups_tie']} | {x['canary_copy_not_preferred']} | {x['canary_source_preferred']} | "
                 f"{x['consistency_rate']} / {x['cross_consistency']} | {x['position_bias_rate']} |")
    for name, x in r["readers"].items():
        L += ["", f"## {name} per seed", "",
              "| seed | winner | orders | decoy | decoy vs both-off winner | dup of | dup raw choices |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {s} | {p['winner']} | {p['winners_by_order']} | {p['decoy']} | {p['decoy_winner']} | {p['dup_of']} | "
              f"{p['dup_choices']} |" for s, p in x["per_seed"].items()]
        L += ["", "| canary | seed | planted on | winner |", "|---|---|---|---|"]
        L += [f"| {c['type']} | {c['seed']} | {c['source_side']} | {c['winner']} |" for c in x["canaries"]]
    if r.get("readers_combined"):
        L.append(f"\nReaders combined (report-only; a winner only when both readers agree): {r['readers_combined']}")
    if "pair_kappa" in r:
        L.append(f"\nPairwise kappa: {r['pair_kappa']}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit")
    ap.add_argument("--lives", nargs="+")
    ap.add_argument("--weeks", default="1,3")
    ap.add_argument("--lived-context", default="all", choices=("all", "intact"))
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--key")
    ap.add_argument("--reads")
    ap.add_argument("--reads2")
    ap.add_argument("--pairs")
    ap.add_argument("--pairs2")
    ap.add_argument("--answers")
    ap.add_argument("--answers2")
    ap.add_argument("--out")
    ap.add_argument("--emit-cross")
    ap.add_argument("--arm-lives", nargs="+")
    ap.add_argument("--b-lives", nargs="+")
    ap.add_argument("--cross-week", type=int, default=3)
    ap.add_argument("--score-cross", action="store_true")
    ap.add_argument("--cross-id-prefix", default="C")
    a = ap.parse_args(argv)
    if a.emit_cross:
        if not a.arm_lives or not a.b_lives:
            ap.error("--emit-cross needs --arm-lives A_s*.json and --b-lives B_s*.json")
        out = emit_cross([json.load(open(p)) for p in a.arm_lives], [json.load(open(p)) for p in a.b_lives], a.emit_cross,
                         a.cross_week, a.lived_context, a.rng_seed, a.cross_id_prefix)
        print(json.dumps(cross_summary(out), indent=1))
        return
    if a.score_cross:
        if not a.key or not a.pairs or not a.out:
            ap.error("--score-cross needs --key, --pairs, --pairs2 (two readers) and --out")
        if a.pairs2 and os.path.exists(a.pairs2) and os.path.samefile(a.pairs, a.pairs2):
            ap.error("--pairs and --pairs2 are the same file: two different readers are required")
        r = score_cross(json.load(open(a.key)), load_cross_reads(a.pairs), load_cross_reads(a.pairs2))
        with open(a.out + ".json", "w") as fh:
            json.dump(r, fh, indent=1)
        with open(a.out + ".md", "w") as fh:
            fh.write(cross_to_md(r))
        print(a.out + ".md")
        return
    if a.emit:
        lives = [json.load(open(p)) for p in a.lives]
        out = emit(lives, a.emit, tuple(int(x) for x in a.weeks.split(",")), a.lived_context, a.rng_seed)
        print(json.dumps(stage_a_summary(out), indent=1))
        return
    if a.score:
        key = json.load(open(a.key))
        r = score(key, _load_jsonl(a.reads), _load_jsonl(a.reads2), _load_jsonl(a.pairs), _load_jsonl(a.pairs2),
                  _load_jsonl(a.answers), _load_jsonl(a.answers2))
        with open(a.out + ".json", "w") as fh:
            json.dump(r, fh, indent=1)
        with open(a.out + ".md", "w") as fh:
            fh.write(to_md(r))
        print(a.out + ".md")
        return
    ap.error("give --emit DIR --lives ... or --score --key ... --reads ... --out PREFIX")


if __name__ == "__main__":
    main()
