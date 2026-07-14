"""D3 -> the EMERGENT talkable console (the anti-recency anaphora wire): the emergent no-Qwen console
(`_realcorpus_unified_talkable_console.UnifiedTalkableConsole`) resolves a pronoun ("it"/"they") to the COMPOSED
discourse FOCUS -- who we're talking about across turns -- tracked by the already-GO D3 Centering-Cb composed-focus
mechanism (`_d3_agent_centering_wire_derisk.D3CenteringFocusSource`), instead of the console's HOST last-subject
recency (`self.last_subject`).

WHY the Centering-Cb source (justify the pick): the console's turns are SVO facts, and the goal is "who we're talking
about ACROSS turns" = the backward-looking discourse CENTER. The D3 Centering-Cb tracker (Grosz-Joshi-Weinstein 1995;
`_d3_centering_focus_derisk.make_centering_task` + `_d3_group_composition_derisk.discrete_attractor_rnn`) is exactly a
composed-focus-over-SVO tracker, and it is the GO-validated adapter the D3 arc deployed into `MultiTurnAgent`'s
`focus_bias_source` hook (`2026-07-09-D3-live-agent-wire-GO.md`). The `PairEventRegister.who_agent()` also tracks the
running-event agent, but adds a prior-slot mechanism the console (no connectives) does not use; the Centering Cb is the
minimal, purpose-built "discourse center" and reuses the deployed adapter verbatim -> chosen.

THE WIRE (additive, DEFAULT-OFF): `D3FocusConsole(UnifiedTalkableConsole)` adds `use_d3_focus=False`. When ON, each
heard fact `hear_fact(subj, verb, obj)` (a) teaches the fact into the console's KB, (b) updates the HOST recency via the
console's OWN mechanism (`self.last_subject = subj`), and (c) `observe(subj, obj)` into a `D3CenteringFocusSource`; and
`_resolve` of a pronoun returns the composed focus `referents[Cb]` instead of `self.last_subject`. Default OFF ==
byte-identical to the stock console (`_resolve` falls straight through to `super()._resolve`, the focus register is
never even built). Reuse-by-import; numpy (`SIM_BACKEND=numpy`); NO `sim/` edit.

THE TASK + GO BAR. A FOCUS-SHIFTED discourse = a short SVO sequence where the composed center is realized in the final
utterance as its OBJECT while a NEW subject appears (Centering CONTINUE), so the true center != the last SUBJECT (the
console's host recency). GO = on focus-shifted discourses D3-focus resolves the composed center DECISIVELY more than the
host last-subject recency (gap >> 0); on NON-shifted (continued-subject) discourses D3 and the host AGREE (no
regression); the DEFAULT-OFF path is byte-identical to the stock console. Anti-cheats: (a) register-LESION (freeze the
D3 register -> no observations -> the composed focus collapses -> the D3 win vanishes / abstains, proving the register
is load-bearing); (b) the non-shifted control (D3 == host, no spurious divergence); (c) the no-confab MOAT (an
ungrounded pronoun -- empty discourse -- still ABSTAINS, never confabulates to the Cb-identity default); (d) the console
is deterministic (no RNG in resolution) -> multi-seed over distinct seeds + several distinct focus-shifted discourses.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_emergent_console_wire_derisk --seeds 42 43 44 \
          --json research/findings/raw/_d3_emergent_console_wire.json
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_unified_talkable_console import UnifiedTalkableConsole
from research.runners._d3_agent_centering_wire_derisk import D3CenteringFocusSource

# The composed-focus referents. Index 0 ('tree') is a NEVER-USED sentinel so the Centering delta's ident=0 start never
# coincidentally equals a scenario's center. The other five are all in the console's top-128 corpus vocab (row_of), so
# facts about them teach + answer, and the resolved focus routes to a real KB subject. Verb 'see' is in row_of.
REFS = ["tree", "dog", "cat", "bird", "ball", "box"]
VERB = "see"
_IDX = {r: i for i, r in enumerate(REFS)}


def _true_center(facts):
    """The Centering backward-looking center Cb over the SVO facts (ground truth; ident=0). cb = cb if realized else s."""
    cb = 0
    for (s, _v, o) in facts:
        si, oi = _IDX[s], _IDX[o]
        cb = cb if cb in (si, oi) else si
    return REFS[cb]


# FOCUS-SHIFTED discourses (type B): the center is realized as the OBJECT of the final utterance while a NEW subject
# appears (Centering CONTINUE) -> true center != last SUBJECT (the host recency) -> the composed focus must beat it.
FOCUS_SHIFTED = [
    [("cat", VERB, "box"), ("cat", VERB, "ball"), ("cat", VERB, "box"), ("dog", VERB, "cat")],   # center cat, last-subj dog
    [("bird", VERB, "box"), ("bird", VERB, "ball"), ("bird", VERB, "box"), ("cat", VERB, "bird")],  # center bird, last-subj cat
    [("dog", VERB, "ball"), ("dog", VERB, "box"), ("dog", VERB, "ball"), ("bird", VERB, "dog")],   # center dog, last-subj bird
]
# NON-SHIFTED controls (type A): the subject CONTINUES as the center (realized as subject in the final utterance) -> the
# true center == the last SUBJECT -> D3 and the host recency should AGREE (no spurious divergence).
NON_SHIFTED = [
    [("bird", VERB, "ball"), ("dog", VERB, "cat"), ("dog", VERB, "box"), ("dog", VERB, "ball")],   # center dog == last-subj
    [("cat", VERB, "box"), ("bird", VERB, "ball"), ("bird", VERB, "box"), ("bird", VERB, "cat")],   # center bird == last-subj
    [("ball", VERB, "box"), ("cat", VERB, "dog"), ("cat", VERB, "box"), ("cat", VERB, "ball")],     # center cat == last-subj
]


class D3FocusConsole(UnifiedTalkableConsole):
    """The emergent console with the D3 composed-focus pronoun resolution wired in (additive, default OFF)."""

    def __init__(self, *args, use_d3_focus=False, focus_seed=42, focus_referents=REFS, frozen_focus=False, **kw):
        super().__init__(*args, **kw)
        self.use_d3_focus = bool(use_d3_focus)
        self._frozen_focus = bool(frozen_focus)   # LESION: register present but observe() is a no-op (no composed focus)
        self._focus_reg = None
        if self.use_d3_focus:
            self._focus_reg = D3CenteringFocusSource(list(focus_referents), seed=int(focus_seed))

    # --- the composed discourse focus (who we're talking about across turns) ---
    def _focus(self):
        """The composed discourse center (Centering Cb) over the observed SVO facts, or None if none observed (moat:
        an ungrounded pronoun has no antecedent -- do NOT return the Cb-identity default)."""
        reg = self._focus_reg
        if reg is None or not reg.facts:
            return None
        foc = reg.referents[reg._cb()]
        return foc

    def hear_fact(self, subj, verb, obj):
        """One discourse turn: teach the fact (KB) + update the HOST recency the console's OWN way (last-mentioned
        SUBJECT) + fold (subj, obj) into the composed-focus register (unless the register is lesioned/frozen)."""
        ok = self.teach_relational(subj, verb, obj)
        UnifiedTalkableConsole._resolve(self, subj)   # host recency: self.last_subject = subj (the stock mechanism)
        if self.use_d3_focus and self._focus_reg is not None and not self._frozen_focus:
            self._focus_reg.observe(subj, obj)
        return ok

    def reset_discourse(self):
        self.last_subject = None
        if self._focus_reg is not None:
            self._focus_reg.reset()

    def _resolve(self, word):
        """Pronoun -> the COMPOSED discourse focus (D3 Centering-Cb) when use_d3_focus, else the stock behavior
        (byte-identical: super()._resolve, which returns self.last_subject). A non-pronoun always falls through to
        super (so the last_subject bookkeeping + the whole default path are unchanged)."""
        if self.use_d3_focus and isinstance(word, str) and word in ("it", "they", "them"):
            foc = self._focus()
            return word if foc is None else foc   # ungrounded pronoun -> unchanged (downstream moat); else composed focus.
            # NOTE: deliberately do NOT mutate self.last_subject here -- that would overwrite the host last-subject
            # recency baseline (measured in parallel by the de-risk); the D3 path never reads last_subject anyway.
        return super()._resolve(word)


def _build(seed, use_d3_focus, corpus, K, n_clusters, bridge, frozen=False):
    return D3FocusConsole(corpus, K, n_clusters, bridge, seed, "run", "sleep", VERB,
                          use_d3_focus=use_d3_focus, focus_seed=seed, frozen_focus=frozen)


def _byte_identity(seed, corpus, K, n_clusters, bridge):
    """Assert D3FocusConsole(use_d3_focus=False) is byte-identical to the stock UnifiedTalkableConsole on a battery of
    questions INCLUDING a pronoun question (proving the D3 override never fires when the flag is off)."""
    stock = UnifiedTalkableConsole(corpus, K, n_clusters, bridge, seed, "run", "sleep", VERB)
    off = _build(seed, False, corpus, K, n_clusters, bridge)
    battery = [
        f"does a {REFS[1]} run?", f"what does the {REFS[1]} {VERB}?", "what does the dog see?",
        "tell me about dog", "compare dog and cat", "what does the zzzqqx see?",
        f"what does the {REFS[1]} {VERB}?",  # sets last_subject=dog on both
        "what does it see?",                 # pronoun: must resolve identically (both -> host last_subject)
        "does it run?", "describe it",
    ]
    diffs = []
    for q in battery:
        a1 = stock.ask(q)
        a2 = off.ask(q)
        if a1 != a2:
            diffs.append((q, a1, a2))
    return {"identical": len(diffs) == 0, "n_probes": len(battery), "diffs": diffs[:5]}


def _score_set(d3con, hostcon, scenarios):
    """Score resolve-to-Cb on TWO INDEPENDENT consoles -- the D3-ON console (composed Centering-Cb focus) and a
    SEPARATE host-OFF console (the stock last-subject recency, exactly like the demo builds it). For each scenario:
    reset both, ASSERT the D3 register is empty (no cross-scenario leak), hear the SAME facts into both, ASSERT the
    register accumulated exactly len(facts) observations, then read each console's own `_resolve('it')` and score
    against the true Centering center. Reading two independent consoles removes any shared-state confound."""
    d3_ok = host_ok = lastobj_ok = tot = 0
    detail = []
    for facts in scenarios:
        d3con.reset_discourse()
        hostcon.reset_discourse()
        assert not d3con._focus_reg.facts, "leak: D3 register not cleared by reset_discourse"   # no cross-scenario leak
        for (s, v, o) in facts:
            d3con.hear_fact(s, v, o)
            hostcon.hear_fact(s, v, o)
        assert len(d3con._focus_reg.facts) == len(facts), "leak: register did not accumulate exactly this discourse"
        center = _true_center(facts)
        d3_res = d3con._resolve("it")                     # the WIRED resolution (composed Centering-Cb focus)
        host_res = hostcon._resolve("it")                 # the STOCK resolution on a SEPARATE console (last-subject)
        lastobj = facts[-1][2]
        d3_ok += int(d3_res == center)
        host_ok += int(host_res == center)
        lastobj_ok += int(lastobj == center)
        tot += 1
        detail.append({"facts": [f"{s} {v} {o}" for (s, v, o) in facts], "center": center,
                       "d3_resolves_it": d3_res, "host_resolves_it": host_res, "last_object": lastobj,
                       "d3_correct": d3_res == center, "host_correct": host_res == center})
    m = max(tot, 1)
    return {"d3": round(d3_ok / m, 3), "host_last_subject": round(host_ok / m, 3),
            "last_object": round(lastobj_ok / m, 3), "n": tot}, detail


def _moat_check(con):
    """MOAT: a fresh discourse (no facts heard) + a pronoun question must ABSTAIN (never confabulate to the Cb default)."""
    con.reset_discourse()
    out, kind = con.ask("what does it see?")
    return {"abstains": (kind == "moat" or "don't know" in out.lower()), "answer": out, "kind": kind}


def run_seed(seed, corpus, K, n_clusters, bridge, verbose=False):
    d3con = _build(seed, True, corpus, K, n_clusters, bridge)               # D3-ON console
    hostcon = _build(seed, False, corpus, K, n_clusters, bridge)            # SEPARATE host-OFF console (the baseline)
    shift, shift_detail = _score_set(d3con, hostcon, FOCUS_SHIFTED)
    nonshift, nonshift_detail = _score_set(d3con, hostcon, NON_SHIFTED)
    moat = _moat_check(d3con)

    # register-LESION: a FROZEN D3-ON console (observe is a no-op) -> no composed focus -> abstains. Proves the
    # register's observations are load-bearing (the host baseline is recomputed on hostcon, unchanged).
    les_d3 = _build(seed, True, corpus, K, n_clusters, bridge, frozen=True)
    lesion, _ = _score_set(les_d3, hostcon, FOCUS_SHIFTED)

    # END-TO-END deployment demonstration: the SAME focus-shifted discourse, asked through ask(), answers about the
    # COMPOSED center under D3 vs the last-subject under the host -- a real behavioral change on the console.
    facts = FOCUS_SHIFTED[0]
    d3con.reset_discourse()
    hostcon.reset_discourse()
    for (s, v, o) in facts:
        d3con.hear_fact(s, v, o)
        hostcon.hear_fact(s, v, o)
    d3_ans = d3con.ask(f"what does it {VERB}?")
    host_ans = hostcon.ask(f"what does it {VERB}?")
    demo = {"discourse": [f"{s} {v} {o}" for (s, v, o) in facts], "center": _true_center(facts),
            "d3_resolves_it_to": d3con._focus(), "d3_answer": d3_ans,
            "host_resolves_it_to": hostcon.last_subject, "host_answer": host_ans}

    # ALWAYS write the per-discourse detail (auditable numbers), not only under --verbose.
    return {"seed": seed, "focus_shifted": shift, "non_shifted": nonshift, "lesion_focus_shifted": lesion,
            "moat": moat, "demo": demo, "focus_shifted_detail": shift_detail, "non_shifted_detail": nonshift_detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--bridge", default="bridges/breadth_aw/seed42.simstate.h5")
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("[D3 -> EMERGENT talkable console] the emergent console resolves a pronoun to the COMPOSED discourse focus "
          "(D3 Centering-Cb over the SVO facts it hears) instead of the host last-subject recency", flush=True)

    # byte-identity (default-off == stock) -- once, on the first seed
    bid = _byte_identity(a.seeds[0], a.corpus_path, a.K, a.n_clusters, a.bridge)
    print(f"  [byte-identity seed {a.seeds[0]}] default-off console == stock UnifiedTalkableConsole over "
          f"{bid['n_probes']} probes: {'IDENTICAL' if bid['identical'] else 'DIFFERS: ' + str(bid['diffs'])}", flush=True)

    rows = []
    for s in a.seeds:
        r = run_seed(s, a.corpus_path, a.K, a.n_clusters, a.bridge, verbose=a.verbose)
        rows.append(r)
        fs, ns, le, mo = r["focus_shifted"], r["non_shifted"], r["lesion_focus_shifted"], r["moat"]
        print(f"  [seed {s}] FOCUS-SHIFTED: D3={fs['d3']} vs host-last-subject={fs['host_last_subject']} "
              f"(last-object={fs['last_object']}) | NON-SHIFTED: D3={ns['d3']} vs host={ns['host_last_subject']} | "
              f"LESION(frozen reg) D3={le['d3']} | moat abstains={mo['abstains']}", flush=True)
        d = r["demo"]
        print(f"      demo: {d['discourse']} (center={d['center']}) -> D3 'it'->{d['d3_resolves_it_to']} : \"{d['d3_answer'][0]}\" "
              f"|| host 'it'->{d['host_resolves_it_to']} : \"{d['host_answer'][0]}\"", flush=True)
        if r["seed"] == a.seeds[0]:                       # per-discourse audit detail for the first seed
            print(f"      focus-shifted per-discourse (seed {r['seed']}):", flush=True)
            for dd in r["focus_shifted_detail"]:
                print(f"        {dd['facts']} center={dd['center']} | D3 'it'->{dd['d3_resolves_it']} ({'OK' if dd['d3_correct'] else 'MISS'}) "
                      f"| host 'it'->{dd['host_resolves_it']} ({'OK' if dd['host_correct'] else 'MISS'}) | last-obj={dd['last_object']}", flush=True)

    out = {"byte_identity": bid, "seeds": rows, "REFS": REFS, "VERB": VERB}
    if a.json:
        import json
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}", flush=True)

    def _m(sel):
        return float(np.mean([sel(r) for r in rows]))
    fs_d3 = _m(lambda r: r["focus_shifted"]["d3"])
    fs_host = _m(lambda r: r["focus_shifted"]["host_last_subject"])
    fs_lo = _m(lambda r: r["focus_shifted"]["last_object"])
    ns_d3 = _m(lambda r: r["non_shifted"]["d3"])
    ns_host = _m(lambda r: r["non_shifted"]["host_last_subject"])
    ns_lo = _m(lambda r: r["non_shifted"]["last_object"])
    le_d3 = _m(lambda r: r["lesion_focus_shifted"]["d3"])
    moat_ok = all(r["moat"]["abstains"] for r in rows)

    go = (fs_d3 > fs_host + 0.4) and (fs_host < 0.2) and (abs(ns_d3 - ns_host) < 0.15) and \
         (le_d3 < fs_d3 - 0.4) and moat_ok and bid["identical"]
    print(f"\n  AGGREGATE ({len(rows)} seeds):", flush=True)
    print(f"    FOCUS-SHIFTED : D3 composed-focus={fs_d3:.3f} | host last-subject={fs_host:.3f} | (last-object={fs_lo:.3f})", flush=True)
    print(f"    SCOPE NOTE    : the Centering Cb is ALWAYS realized in the final clause, so Cb in {{last-subject, last-object}}. "
          f"These CONTINUE-as-OBJECT discourses put Cb on the OBJECT -> a pure last-object heuristic coincides here "
          f"(last-object={fs_lo:.3f}), but the console's HOST recency is last-SUBJECT (what D3 replaces), and last-object "
          f"FAILS the non-shifted class (below) -- so the deployed win is D3 over the host last-subject.", flush=True)
    print(f"    NON-SHIFTED   : D3={ns_d3:.3f} | host last-subject={ns_host:.3f}  (should AGREE) | (last-object={ns_lo:.3f} FAILS here)", flush=True)
    print(f"    LESION(frozen): D3 focus-shifted={le_d3:.3f}  (should COLLAPSE vs {fs_d3:.3f})", flush=True)
    print(f"    MOAT abstains : {moat_ok} | byte-identity default-off==stock: {bid['identical']}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- "
          f"{'the EMERGENT talkable console resolves a pronoun to the COMPOSED discourse focus (D3 Centering-Cb) '+format(fs_d3,'.2f')+' where the host last-subject recency FAILS on focus-shifted discourses '+format(fs_host,'.2f')+', AGREES with the host on non-shifted discourses (D3 '+format(ns_d3,'.2f')+' == host '+format(ns_host,'.2f')+'), COLLAPSES under register-lesion '+format(le_d3,'.2f')+' (the composed focus is load-bearing), preserves the no-confab moat (an ungrounded pronoun abstains), and is byte-identical to the stock console when default-off -> a HOST recency shortcut REPLACED by the emergent composed-focus mechanism on the deployed no-Qwen console' if go else 'read the focus-shifted D3-vs-host gap, the non-shifted agreement, the lesion collapse, the moat, and byte-identity'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
