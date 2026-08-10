"""Standalone proof of the episodic-recall case-selection + render logic wired into
`research/runners/_conversation_turing_test_derisk.py` (INTEGRATION #2, 2026-08-10).

It mirrors the runner's `_render_facts` + Case A/B/C selection, stubbing `SA._gm_fact_to_english`
with the SAME motion-verb rule the runner reuses (so the rendered surface matches). Confirms that a
referent which WAS discussed produces a GENUINE recall of ITS facts (Case A), not the false-premise
fallback (Case B) that the seed-42 turn-7 (cat never discussed) exercises.

Run: python research/findings/raw/lanes/stageA/turing/case_a_episodic_recall_check.py
"""


def fact_to_english(svo):
    a, v, p = svo
    prep = "at " if v == "look" else ("to the " if v in ("go", "come") else "")
    vv = v + ("es" if v.endswith(("s", "sh", "ch", "x", "z")) else "s")
    an = ("an " if a[:1] in "aeiou" else "a ") + a
    return f"{an.capitalize()} {vv} {prep}{p}."


def recall(ref, episode_mem):
    recalled = [ep for ep in episode_mem if ref is not None and ep["topic"] == ref]
    discussed = []
    for ep in episode_mem:
        if ep["topic"] not in discussed:
            discussed.append(ep["topic"])

    def render(fact_list):
        uniq = []
        for f in fact_list:
            if list(f) not in uniq:
                uniq.append(list(f))
        return " ".join(fact_to_english(tuple(f)) for f in uniq)

    if recalled:
        return "CASE_A", render([f for ep in recalled for f in ep["facts"]])
    elif discussed:
        return "CASE_B", render([f for ep in episode_mem for f in ep["facts"]])
    return "CASE_C", ""


if __name__ == "__main__":
    dog_mem = [
        {"turn": 3, "topic": "dog", "facts": [["dog", "go", "east"], ["dog", "look", "river"], ["dog", "run", "north"]]},
        {"turn": 4, "topic": "dog", "facts": [["dog", "go", "east"], ["dog", "look", "river"], ["dog", "run", "north"]]},
    ]
    print("ref=cat, only dog discussed (turn-7 seed42) ->", recall("cat", dog_mem))
    cat_mem = dog_mem + [
        {"turn": 6, "topic": "cat", "facts": [["cat", "run", "south"], ["cat", "go", "west"], ["cat", "look", "apple"]]},
    ]
    print("ref=cat, a cat turn preceded (Case A)        ->", recall("cat", cat_mem))
    print("ref=cat, nothing discussed (Case C)          ->", recall("cat", []))
