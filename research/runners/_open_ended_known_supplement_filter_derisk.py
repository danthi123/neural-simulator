"""A CONTRADICTION filter catches the known-topic WRONG-supplement residual of open-ended generation (2026-08-21).

The wired open-ended mode + post-filter fully resolves the UNKNOWN-topic 100% fabrication, but leaves a named
residual: on a KNOWN topic the free Qwen reply keeps the grounded facts AND adds confident WRONG supplements on the
SAME relations (Canada "borders ... Mexico" [store: united states]; France "bordered by Italy/Germany/Switzerland"
[store: spain]; plus unsupported numbers "35 million" / dates "1867"). The reused post_filter's `contradicts` is a
stub, so those survived.

This de-risks the fix: a CONTRADICTION filter that, per sentence, drops it when it (a) carries a specific number or
year (never in the SVO store), or (b) asserts a stored relation (borders / continent / capital) with a DIFFERENT
object than the store holds. Validated on the SAVED open-ended replies (no new Qwen render — memory-safe): it drops
the wrong supplements while keeping the grounded content.

SCOPE (honest): it catches the CONTRADICTION class (wrong-object on a known relation + numbers/dates) — the observed
high-frequency failures — using a small gazetteer + the topic's store facts; it does NOT catch ungrounded-but-
non-contradicting supplements (France "on the Mediterranean" — true, not in store), and a general solution wants a
store-backed entity check (any store-known entity that isn't the stored object) or an NLI model. The primary
unknown-topic moat is already GO+wired; this is the named next rung toward known-topic honesty. NO sim/ edit.
  python -m research.runners._open_ended_known_supplement_filter_derisk
"""
from __future__ import annotations
import json, os, re, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import logging; logging.disable(logging.INFO)
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._open_ended_state_driven_generation_derisk import _sentences  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_known_supplement_filter_derisk.json"
PRIOR = _REPO / "research" / "findings" / "raw" / "_open_ended_verify_postfilter_derisk.json"

# the topics' store facts (from the 100k Wikidata store) + the wrong supplements they must drop
FACTS = {
    "canada":  [("isa", "country"), ("capital", "ottawa"), ("continent", "north america"), ("borders", "united states")],
    "france":  [("isa", "country"), ("capital", "paris"), ("continent", "europe"), ("borders", "spain")],
    "morocco": [("isa", "country"), ("capital", "rabat"), ("continent", "africa"), ("borders", "spain")],
}
COUNTRIES = {"united states", "usa", "mexico", "canada", "spain", "italy", "germany", "switzerland", "france",
             "portugal", "algeria", "tunisia", "libya", "egypt", "morocco", "brazil", "china", "india", "russia",
             "belgium", "austria", "luxembourg"}
CONTINENTS = {"north america", "south america", "europe", "africa", "asia", "oceania", "antarctica"}
# the wrong-supplement objects that MUST be dropped (the ground truth for this de-risk)
MUST_DROP = {"canada": {"mexico", "35 million", "1867"}, "france": {"italy", "germany", "switzerland"},
             "morocco": {"algeria", "tunisia", "libya", "egypt"}}


def _obj(facts, rel):
    return {o.lower() for r, o in facts if r == rel}


def sentence_contradicts(sent, topic, facts):
    s = sent.lower()
    if re.search(r"\b\d{3,}\b|\bmillion\b|\bbillion\b|\b1[0-9]{3}\b|\b20[0-9]{2}\b", s):
        return "number/date not in store"
    bord = _obj(facts, "borders")
    if bord and re.search(r"\bborder", s):
        mentioned = {c for c in COUNTRIES if re.search(r"\b" + re.escape(c) + r"\b", s)} - {topic}
        wrong = mentioned - bord - {w for b in bord for w in b.split()}
        if wrong:
            return "wrong border object(s): %s" % sorted(wrong)
    cont = _obj(facts, "continent")
    if cont:
        for c in CONTINENTS:
            if re.search(r"\b" + re.escape(c) + r"\b", s) and c not in cont:
                return "wrong continent: %s" % c
    cap = _obj(facts, "capital")
    if cap and "capital" in s:
        m = re.search(r"capital[^.]*?\bis\b\s+([a-z]+)", s)
        if m and m.group(1) not in cap and m.group(1) not in topic:
            return "wrong capital: %s" % m.group(1)
    return None


def main():
    prior = json.load(open(PRIOR))
    by_topic = {r["topic"]: r["raw"] for r in prior["known_rows"]}
    rows = []
    for topic, facts in FACTS.items():
        raw = by_topic.get(topic)
        if not raw:
            continue
        kept, dropped = [], []
        for sent in _sentences(raw):
            why = sentence_contradicts(sent, topic, facts)
            (dropped if why else kept).append(sent)
        droptext = " ".join(dropped).lower()
        keptext = " ".join(kept).lower()
        caught = {m for m in MUST_DROP[topic] if m in droptext}
        leaked = {m for m in MUST_DROP[topic] if m in keptext}
        rows.append({"topic": topic, "n_kept": len(kept), "n_dropped": len(dropped),
                     "must_drop": sorted(MUST_DROP[topic]), "caught_in_dropped": sorted(caught),
                     "leaked_into_kept": sorted(leaked), "kept_nonempty": bool(kept)})

    total_must = sum(len(MUST_DROP[r["topic"]]) for r in rows)
    total_caught = sum(len(r["caught_in_dropped"]) for r in rows)
    total_leaked = sum(len(r["leaked_into_kept"]) for r in rows)
    all_nonempty = all(r["kept_nonempty"] for r in rows)
    catch_rate = round(total_caught / (total_must or 1), 3)

    art = {"probe": "open_ended_known_supplement_filter_derisk", "backend": "numpy",
           "source": str(PRIOR.relative_to(_REPO)), "rows": rows,
           "wrong_supplements_total": total_must, "caught": total_caught, "leaked": total_leaked,
           "catch_rate": catch_rate, "all_replies_kept_nonempty": all_nonempty}

    v = Verdict("contradiction filter catches known-topic wrong supplements on the saved open-ended replies")
    v.floor("wrong-supplement catch rate (relation-object + numbers/dates)", measured=catch_rate, floor=0.8)
    v.require("no wrong supplement leaks into the kept text", total_leaked, expect=0)
    v.require("the filter never empties a reply (still conversational)", all_nonempty, expect=True)
    v.disabled("ungrounded-but-non-contradicting supplements + general (non-gazetteer) entity check",
               why="v1 catches the CONTRADICTION class via a gazetteer + the topic's store facts; a store-backed "
                   "entity check (any store-known entity != the stored object) or an NLI model is the general rung")
    go = (catch_rate >= 0.8 and total_leaked == 0 and all_nonempty)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k] for k in ("wrong_supplements_total", "caught", "leaked", "catch_rate",
                                          "all_replies_kept_nonempty", "GO")}, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
