# Capability: Lifelong learning
# Record: brain/research/2026-11-21-replay-priority-prereg.md (section 3.2, primary read-out)
# Expected-fail: none (host-only post-hoc read of a results JSON; the leak itself is expected to stay high in B)
"""Stock self-reply leak: the share of replies to questions NOT about names/identity whose text contains the Assistant
part of a permanent self episode ("Of course, you're Dan, my friend.", "My name is Nova.", "I'm Nova, a young digital
brain."). Read from a life results JSON: the battery probe rows (res['probes']) and the interview rows
(res['interview_probes'], kinds know / forced / generic) of the intact and hippocampus-off arms, per probe night.

The permanent episodes are the first ones stored (Brain._grow_self; their replies are fixed by the brain's name and
friend, defaults Nova / Dan, which the ledger does not record). A question is about names/identity when its item is
one of the brain's own facts in the life's ledger (telling is None: the battery items "dan" and "nova"); no English
keyword rule. Matching is case-insensitive on the permanent reply or its identity core ("you're Dan", "My name is
Nova", "I'm Nova")."""
from __future__ import annotations

NAME, FRIEND = "Nova", "Dan"
ARMS = ("intact", "hippocampus-off")


def permanent_replies(name: str = NAME, friend: str = FRIEND) -> list:
    """The Assistant parts of the permanent self episodes and their identity cores, lower-case, quotes normalised."""
    full = [f"My name is {name}.", f"I'm {name}, a young digital brain.", f"Of course, you're {friend}, my friend."]
    core = [f"you're {friend}", f"my name is {name}", f"i'm {name}"]
    return [_norm(x) for x in full + core]


def _norm(s: str) -> str:
    return (s or "").replace("’", "'").replace("‘", "'").lower()


def is_leak(reply: str, needles: list) -> bool:
    r = _norm(reply)
    return any(n in r for n in needles)


def identity_items(res: dict) -> set:
    """Items that are about names/identity: the brain's own facts (ledger telling None)."""
    return {k for k, v in res["ledger"].items() if v.get("telling") is None}


def _rows(res: dict):
    """(source, arm, night, row) for every probe and interview row of the intact and hippocampus-off arms, trunk line
    only: the pruned-branch probes (line "prunedA") and the extra tagged trunk reads (tag "ab-recall", an AB
    recall re-read) are left out so each (arm, night) cell is one battery read plus that night's interview."""
    for p in res.get("probes", []):
        if p["arm"] in ARMS and p.get("line", "trunk") == "trunk" and not p.get("tag"):
            for r in p["rows"]:
                yield "probe", p["arm"], p["night"], r
    for p in res.get("interview_probes", []):
        if p["arm"] in ARMS and (p.get("line") or "trunk") == "trunk" and not p.get("tag"):
            for r in p["rows"]:
                yield "interview", p["arm"], p["night"], r


def leak_table(res: dict, name: str | None = None, friend: str | None = None) -> dict:
    """{(arm, night): {"n", "leaks", "share", "by_source": {source: [n, leaks]},
    "by_kind": {"source:kind": [n, leaks]}}}: replies to non-identity questions."""
    kw = res.get("brain_kw") or {}
    needles = permanent_replies(name or kw.get("name") or NAME, friend or kw.get("friend") or FRIEND)
    ident = identity_items(res)
    out = {}
    for src, arm, night, r in _rows(res):
        if r.get("item") in ident:
            continue
        c = out.setdefault((arm, night), {"n": 0, "leaks": 0, "by_source": {}, "by_kind": {}})
        s = c["by_source"].setdefault(src, [0, 0])
        k = c["by_kind"].setdefault(f"{src}:{r.get('kind')}", [0, 0])
        leak = is_leak(r.get("brain", ""), needles)
        c["n"] += 1
        c["leaks"] += leak
        for x in (s, k):
            x[0] += 1
            x[1] += leak
    for c in out.values():
        c["share"] = c["leaks"] / c["n"] if c["n"] else None
    return dict(sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0])))


def pooled(table: dict, nights, arms=ARMS) -> dict:
    """Pool (arm, night) cells over the given nights and arms (the prereg reads nights 10 and 15, both arms)."""
    n = sum(c["n"] for (a, d), c in table.items() if d in nights and a in arms)
    k = sum(c["leaks"] for (a, d), c in table.items() if d in nights and a in arms)
    return {"n": n, "leaks": k, "share": k / n if n else None}


def render(table: dict) -> str:
    L = ["| arm | night | n replies | n leaks | share | probe n/leaks | interview n/leaks |", "|---|---|---|---|---|---|---|"]
    for (arm, night), c in table.items():
        pr, iv = c["by_source"].get("probe", [0, 0]), c["by_source"].get("interview", [0, 0])
        L.append(f"| {arm} | {night} | {c['n']} | {c['leaks']} | {c['share']:.3f} | {pr[0]}/{pr[1]} | {iv[0]}/{iv[1]} |")
    return "\n".join(L)


if __name__ == "__main__":
    import json
    import sys
    res = json.load(open(sys.argv[1]))
    t = leak_table(res)
    print(render(t))
    for nights in ({10}, {15}, {10, 15}):
        print(f"pooled nights {sorted(nights)}, both arms: {pooled(t, nights)}")
