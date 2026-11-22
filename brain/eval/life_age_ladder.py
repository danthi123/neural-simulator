# Capability: Lifelong learning
# Record: brain/research/2026-11-22-fresh-test-split-design.md
# Expected-fail: P0-v2 NOT VALID if forgetting is total (recall 0.0 at age >= 10); a measured result, not a fault
"""The age-ladder sub-battery (design 2026-11-22 sections 2 and 3) for brain/eval/life.py --age-ladder (default OFF).

Plan (age_ladder_plan, pure and deterministic per seed; named apart from life_ladder.ladder_plan, the cue ladder):
- ladder items: class F or D, told exactly once on days 2-6, not in life_ladder.live_asked_late and not a same-day
  question. Probe nights for told day t: t, t+3, t+6, t+9 and 15 (clipped: t+3k kept only if <= days; the last probe is
  min(15, days)). age_nights = night - t + 1 (nights slept, life_bars.AGE_OFFSET 1).
- controls: F/D items told once on days 11-15, not the same-day question, probed the night of telling (age 1).
- never-told questions (NEVER_LADDER of the bank; none when the bank has no NEVER_LADDER): 24 drawn with
  random.Random(f"ladder-never:{seed}") under slot quotas proportional to the ladder items' slots (largest remainder, at
  least 4 per slot present), in two halves of 12: half A on even nights, half B on odd nights.
- arms: hippocampus-off on nights 2..min(15, days); intact and both-off on nights 5, 10, 15 (when that arm is probed).
Every turn is one independent fresh conversation (life_interview.turn: state = base, reseeded, every trace restored), so
no question is stored and no turn changes another. Rows go to probe["age_ladder"], never probe["rows"].

Scoring: items with life.mark_item (strict correct_answer). Never rows carry three marks:
- slot_false (the P0-v2 chance-recall floor, section 3a): any(correct_answer(reply, k, q, told=telling)) over the keys of
  items told by that morning with the question's slot, minus STOCK (H dan, nova); telling = the ledger telling (H: its
  weekly.SESSION_1 line);
- fm_v2 (section 3b): the reply names any told key as a word, excluding keys in the question and STOCK keys;
- v1_false: the Stage-1 definition (any told key, STOCK included), for comparability.
"""
from __future__ import annotations

import random
import re

from brain.eval.weekly import SESSION_1
from brain.experiments.strict_scorer import correct_answer

LADDER_CLS = ("F", "D")
LADDER_DAYS = (2, 6)          # told on days 2..6 inclusive
CONTROL_DAYS = (11, 15)       # told on days 11..15 inclusive
STEP = 3                      # probe nights t, t+3, t+6, t+9, then LAST
N_STEPS = 4
LAST = 15
FULL_NIGHTS = (5, 10, 15)     # intact and both-off ladder nights (FM-v2)
N_NEVER = 24
NEVER_MIN = 4
STOCK = ("dan", "nova")       # the stock self-names (H): their leak is the stock-leak read-out, not the floor
H_TELLING = {"admin": SESSION_1[1], "pixel": SESSION_1[2], "marguerite": SESSION_1[4], "volvo": SESSION_1[5]}
FLOOR_NIGHTS = (11, 15)       # the floor and FM-v2 window (inclusive)


def _sameday(sched: list) -> set:
    return {t["item"] for day in sched for t in day["turns"] if t["kind"] == "sameday" and t.get("item")}


def _quotas(counts: dict, pool_sizes: dict, total: int = N_NEVER, floor: int = NEVER_MIN) -> dict:
    """Largest-remainder quotas of `total` proportional to `counts` (slots with a pool only), at least `floor` per slot
    present, capped at the pool size. Ties by slot name."""
    slots = sorted(s for s, c in counts.items() if c > 0 and pool_sizes.get(s, 0) > 0)
    if not slots:
        return {}
    tot = sum(counts[s] for s in slots)
    raw = {s: total * counts[s] / tot for s in slots}
    q = {s: int(raw[s]) for s in slots}
    for s in sorted(slots, key=lambda s: (-(raw[s] - q[s]), s))[: total - sum(q.values())]:
        q[s] += 1
    for _ in range(4 * total):      # raise every slot to the floor (and cap at the pool), taking from the largest
        low = [s for s in slots if q[s] < min(floor, pool_sizes[s])]
        over = [s for s in slots if q[s] > pool_sizes[s]]
        if not low and not over:
            break
        if over:
            s = over[0]
            q[s] -= 1
            give = [x for x in slots if q[x] < pool_sizes[x] and x != s]
            if give:
                q[min(give, key=lambda x: (q[x], x))] += 1
            continue
        s = low[0]
        donor = max((x for x in slots if q[x] > floor and x != s), key=lambda x: (q[x], x), default=None)
        if donor is None:
            break
        q[s] += 1
        q[donor] -= 1
    return q


def age_ladder_plan(seed: int, days: int, sched: list, ledger: dict, bank) -> dict:
    """The age-ladder plan of one life (see the module docstring). JSON-ready (lists, str keys)."""
    from brain.eval import life_ladder
    late = life_ladder.live_asked_late(sched, ledger)
    sd = _sameday(sched)
    last = min(LAST, days)
    items, controls = [], []
    for iid in sorted(ledger):
        e = ledger[iid]
        if e["cls"] not in LADDER_CLS or len(e["told_days"]) != 1 or iid in late or iid in sd:
            continue
        t = e["told_days"][0]
        if t > days:
            continue
        slot = bank.ALL[iid].slot
        if LADDER_DAYS[0] <= t <= LADDER_DAYS[1]:
            nights = sorted({t + STEP * k for k in range(N_STEPS) if t + STEP * k <= last} | {last})
            nights = [n for n in nights if n >= t]
            items.append({"item": iid, "cls": e["cls"], "slot": slot, "told_day": t, "nights": nights,
                          "ages": [n - t + 1 for n in nights]})
        elif CONTROL_DAYS[0] <= t <= CONTROL_DAYS[1]:
            controls.append({"item": iid, "cls": e["cls"], "slot": slot, "told_day": t, "nights": [t], "ages": [1]})
    pool = list(getattr(bank, "NEVER_LADDER", []) or [])
    never_ids = {n[0] for n in getattr(bank, "NEVER", [])}
    never_txt = {n[1] for n in getattr(bank, "NEVER", [])}
    clash = sorted(n[0] for n in pool if n[0] in never_ids or n[1] in never_txt)
    if clash:
        raise ValueError(f"NEVER_LADDER must be disjoint from the battery/interview NEVER pool: {clash}")
    by_slot: dict = {}
    for nid, _, slot in pool:
        by_slot.setdefault(slot, []).append(nid)
    counts: dict = {}
    for it in items:
        counts[it["slot"]] = counts.get(it["slot"], 0) + 1
    q = _quotas(counts, {s: len(v) for s, v in by_slot.items()}) if pool else {}
    rng = random.Random(f"ladder-never:{seed}")
    picks = []
    for s in sorted(q):
        picks += rng.sample(sorted(by_slot[s]), q[s])
    rng.shuffle(picks)
    half = len(picks) // 2
    hoff = list(range(2, last + 1))
    return {"days": days, "items": items, "controls": controls,
            "never": {"A": sorted(picks[:half]), "B": sorted(picks[half:]), "quotas": q,
                      "slot_counts": {s: counts[s] for s in sorted(counts)}},
            "arm_nights": {"hippocampus-off": hoff, "intact": [n for n in FULL_NIGHTS if n <= days],
                           "both-off": [n for n in FULL_NIGHTS if n <= days]},
            "exclude": sorted({x["item"] for x in items + controls}),
            "n_items_age10p": sum(1 for x in items if any(a >= 10 for a in x["ages"]))}


def never_half(night: int) -> str:
    return "A" if night % 2 == 0 else "B"


def due(plan: dict, night: int, bank) -> list:
    """The questions of one ladder morning, in a fixed order: ladder items, controls (by id), then that night's never half."""
    qs = []
    for kind, group in (("ladder", plan["items"]), ("control", plan["controls"])):
        for x in group:
            if night in x["nights"]:
                qs.append({"kind": kind, "item": x["item"], "text": bank.ALL[x["item"]].direct, "told_day": x["told_day"],
                           "age_nights": night - x["told_day"] + 1})
    nl = {n[0]: (n[1], n[2]) for n in getattr(bank, "NEVER_LADDER", []) or []}
    h = never_half(night)
    for nid in plan["never"][h]:
        qs.append({"kind": "never", "item": nid, "text": nl[nid][0], "slot": nl[nid][1], "half": h})
    return qs


def arms_for(plan: dict, night: int) -> tuple:
    return tuple(a for a, ns in plan["arm_nights"].items() if night in ns)


def _has(text: str, keys) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text or "", re.I) for k in keys)


def _telling(iid: str, e: dict):
    return H_TELLING.get(iid, e.get("telling"))


def lures(ledger: dict, night: int, slot: str, bank) -> list:
    """[(key, telling)] of items told by that morning with that slot, STOCK and non-bank entries (incidents) excluded."""
    out = []
    for iid, e in sorted(ledger.items()):
        if iid in STOCK or iid not in bank.ALL or not e["told_days"] or min(e["told_days"]) > night:
            continue
        if bank.ALL[iid].slot != slot:
            continue
        out += [(k, _telling(iid, e)) for k in e["keys"]]
    return out


def slot_false(reply: str, question: str, slot: str, ledger: dict, night: int, bank) -> bool:
    """Section 3a: the strict scorer credits a same-slot told key (STOCK out) to a never-told question."""
    return any(correct_answer(reply, k, question, told=t) for k, t in lures(ledger, night, slot, bank))


def any_key_false(reply: str, question: str, ledger: dict, night: int, stock: bool = False) -> bool:
    """Section 3b: the reply names any told key as a word, keys in the question excluded; STOCK keys excluded unless
    stock=True (stock=True is the Stage-1 v1 definition, life.run_battery's never 'false')."""
    keys = sorted({k for i, e in ledger.items() if e["told_days"] and min(e["told_days"]) <= night
                   and (stock or i not in STOCK) for k in e["keys"]})
    return _has(reply, [k for k in keys if not _has(question, [k])])


def run_age_ladder(b, plan: dict, night: int, ledger: dict, bank, sim, t_morning: float, seed: int, L) -> list:
    """Ask the ladder questions due on `night` of an open probe copy; every turn independent (life_interview.turn).
    L = the running life module (mark_item)."""
    from brain.eval import life_interview
    rows = []
    for q in due(plan, night, bank):
        reply, info, changed = life_interview.turn(b, q["text"], sim, t_morning, seed, night, f"age-ladder:{q['item']}")
        row = {k: v for k, v in q.items() if k != "text"}
        row.update(night=night, you=q["text"], brain=reply, recalled=info.get("recalled"), use_restored=changed)
        if q["kind"] == "never":
            row["slot_false"] = slot_false(reply, q["text"], q["slot"], ledger, night, bank)
            row["fm_v2"] = any_key_false(reply, q["text"], ledger, night)
            row["v1_false"] = any_key_false(reply, q["text"], ledger, night, stock=True)
        else:
            row["ok"] = bool(L.mark_item(bank.ALL[q["item"]], q["text"], reply))
        rows.append(row)
    return rows


def ladder_md(res: dict) -> list:
    """The age-ladder section of a life's .md (--age-ladder only)."""
    plan = res["age_ladder"]
    L_ = [f"\n## Age ladder (design 2026-11-22; {len(plan['items'])} items, {len(plan['controls'])} controls, "
          f"{len(plan['never']['A']) + len(plan['never']['B'])} never questions, {plan['n_items_age10p']} items with an "
          "age >= 10 row)\n"]
    for p in res["probes"]:
        rows = p.get("age_ladder")
        if not rows:
            continue
        L_.append(f"\n#### ladder after night {p['night']}, arm: {p['arm']}\n")
        for r in rows:
            if r["kind"] == "never":
                tag = (" SLOT-FALSE" if r["slot_false"] else "") + (" FM" if r["fm_v2"] else "")
                L_.append(f"- [never/{r['slot']}] **you:** {r['you']}\n  **brain:** {r['brain']}{tag}")
            else:
                L_.append(f"- [{r['kind']} age {r['age_nights']}] **you:** {r['you']}\n  **brain:** {r['brain']}"
                          f"{' ✅' if r['ok'] else ' ❌'}")
    return L_
