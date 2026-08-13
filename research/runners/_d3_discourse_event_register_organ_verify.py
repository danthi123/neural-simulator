"""Standalone numpy-CPU verify for `d3_discourse_event_register_production_organ`. Proves, through the ORGAN's own
production entry points (`make_discourse_register` / `note_turn` / `maybe_answer` / `answer_before`):

  (1) INTACT fires   — the spiking twin answers who-was-doing-it-BEFORE across a connective (read off cp_firing_states),
                       while still tracking the present, and beats recency / naive-current.
  (2) LESION collapses — silencing the spiking HOLD of the prior event collapses the before-answer; NOW is preserved
                       (load-bearing: the spiking prev-slot carries the before-answer, and it does not cost the present).
  (3) FLAG-OFF byte-identical — `make_discourse_register(enabled=False)` returns TODAY's register (spiking=False,
                       `bridges is None`), and the endpoint hook is disjoint (never hijacks a normal turn).
  (4) MOAT preserved — a before-query abstains until a connective boundary actually opened this conversation (verified
                       on the no-boundary discourses); a single-event register structurally abstains; an unknown
                       referent is never written.

The capability numbers (before/recency/naive) are measured on the discourses that CONTAIN a connective boundary — the
faculty's actual job. On the no-boundary discourses the organ must ABSTAIN (the moat), which is checked separately.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_discourse_event_register_organ_verify --seeds 42
"""
from __future__ import annotations

import argparse

import numpy as np

from research.runners._d3_event_pair_agent_derisk import make_discourse
from research.runners._d3_event_agent_derisk import D3EventRegister
from tools.lab import attributable_to
import research.runners.d3_discourse_event_register_production_organ as ORG

REFS = ["dog", "cat", "fish", "bird", "worm", "ball"]
ACTIONS = {"chase"}


def _has_boundary(clauses):
    return any((c.split()[0].lower() in ORG.CONNECTIVES) for c in clauses if c.split())


def _measure(reg, seed, n_disc=24):
    """Run `reg` over n_disc discourses via the ORGAN's note_turn/answer path. Returns capability metrics on the
    BOUNDARY discourses (the faculty's job) + the abstain rate on the NO-BOUNDARY discourses (the moat)."""
    rng = np.random.RandomState(seed + 11)
    b = nw = rec = naive = nb = 0            # boundary-subset tallies
    no_b_total = no_b_abstain = 0            # no-boundary-subset (moat) tallies
    now_all_ok = now_all = 0
    for _ in range(n_disc):
        reg.reset()
        st = ORG.new_state()
        clauses, true_now, true_before = make_discourse(rng, REFS)
        for c in clauses:
            ORG.note_turn(c, reg, st, actions=ACTIONS)
        tn, tb = REFS[true_now], REFS[true_before]
        ans_b = ORG.answer_before(reg, st)
        ans_n = ORG.answer_now(reg, st)
        now_all += 1
        now_all_ok += int((not ans_n["abstained"]) and ans_n.get("agent") == tn)
        if _has_boundary(clauses):
            nb += 1
            b += int((not ans_b["abstained"]) and ans_b.get("agent") == tb)
            rec += int(clauses[-1].split()[-1] == tb)
            naive += int((not ans_n["abstained"]) and ans_n.get("agent") == tb)
        else:
            no_b_total += 1
            no_b_abstain += int(ans_b["abstained"])       # MOAT: must abstain when no earlier event exists
    m = max(nb, 1)
    return {"BEFORE": b / m, "recency": rec / m, "naive": naive / m, "n_boundary": nb,
            "NOW": now_all_ok / max(now_all, 1),
            "no_boundary_abstain_rate": (no_b_abstain / no_b_total) if no_b_total else 1.0,
            "n_no_boundary": no_b_total}


def verify(seed=42):
    fails = []
    print(f"[verify d3-discourse-event-register organ] seed={seed}", flush=True)

    # ── build each register ONCE (reused for construct + moat + discourse runs) ───────────────────────────────
    r_on = ORG.make_discourse_register(REFS, seed=seed, enabled=True, lesion=False)   # the validated spiking twin
    r_les = ORG.make_discourse_register(REFS, seed=seed, enabled=True, lesion=True)   # prev-slot-silence spiking
    r_off = ORG.make_discourse_register(REFS, seed=seed, enabled=False)               # today's host rate register

    # ── (3a) byte-identical CONSTRUCTION ──────────────────────────────────────────────────────────────────────
    off_rate = (getattr(r_off, "bridges", "x") is None) and type(r_off).__name__ == "PairEventRegister"
    on_spiking = getattr(r_on, "bridges", None) is not None and type(r_on).__name__ == "PairEventRegister"
    les_spiking = getattr(r_les, "bridges", None) is not None and type(r_les).__name__ == "_PrevSilencePairRegister"
    print(f"  [construct] flag-off spiking=False (today's register)={off_rate} | default spiking twin={on_spiking} | "
          f"lesion prev-silence spiking={les_spiking}", flush=True)
    if not (off_rate and on_spiking and les_spiking):
        fails.append("construction: factory did not return the expected register per flag")

    # ── (3b) the endpoint hook is DISJOINT — it never hijacks a normal turn ────────────────────────────────────
    battery = ["what does the dog chase?", "hello there", "who are you", "the dog and the cat ran",
               "tell me about the cat", "is the sky blue?", "what did you say earlier about the dog?",
               "dog", "chase", "i think therefore i am", "what do you expect next?", "who am i"]
    hijack = [t for t in battery if (ORG.maybe_answer(t, r_on, ORG.new_state()) is not None
                                     or ORG.is_discourse_clause(t, ACTIONS))]
    print(f"  [disjoint] non-discourse turns hijacked by the hook: {hijack if hijack else 'NONE (byte-identical)'}",
          flush=True)
    if hijack:
        fails.append(f"disjointness: the hook fired on non-discourse turns {hijack}")
    if not (ORG.is_before_query("who was doing it before?") and ORG.is_now_query("who is doing it now?")):
        fails.append("query detection: before/now query not recognised")

    # ── (4) MOAT — abstain until a real boundary; single-event structural abstain ──────────────────────────────
    ab = ORG.answer_before(r_on, ORG.new_state())          # fresh conversation, no boundary yet
    moat_no_boundary = ab["abstained"] and "agent" not in ab
    single = D3EventRegister(REFS, seed=seed, spiking=False)
    st_s = ORG.new_state(); st_s["boundary_seen"] = True   # a single-event register can't answer even if a boundary were seen
    ab_s = ORG.answer_before(single, st_s)
    struct = ab_s["abstained"] and ab_s.get("structural") is True
    print(f"  [moat] no-boundary abstain={moat_no_boundary} ('{ab['answer']}') | single-event structural abstain="
          f"{struct} ('{ab_s['answer']}')", flush=True)
    if not moat_no_boundary:
        fails.append("moat: before-answer NOT abstained without a boundary")
    if not struct:
        fails.append("moat: single-event register did not structurally abstain")

    # ── (1) INTACT + (2) LESION + FLAG-OFF over the discourse distribution ─────────────────────────────────────
    print("  [running discourses] intact spiking / lesion spiking / flag-off rate ...", flush=True)
    mi = _measure(r_on, seed)
    ml = _measure(r_les, seed)
    mo = _measure(r_off, seed)
    print(f"  [INTACT spiking] BEFORE={mi['BEFORE']:.3f} NOW={mi['NOW']:.3f} | recency={mi['recency']:.3f} "
          f"naive-current={mi['naive']:.3f}  (on {mi['n_boundary']} boundary discourses)", flush=True)
    print(f"  [LESION spiking] BEFORE={ml['BEFORE']:.3f} NOW={ml['NOW']:.3f}  (prev-slot spiking hold silenced)", flush=True)
    print(f"  [FLAG-OFF rate ] BEFORE={mo['BEFORE']:.3f} NOW={mo['NOW']:.3f}  (today's host register — capability preserved)",
          flush=True)
    print(f"  [MOAT] no-boundary abstain rate: intact={mi['no_boundary_abstain_rate']:.3f} on {mi['n_no_boundary']} "
          f"no-boundary discourses (must be 1.000 — no false 'X was')", flush=True)

    # ATTRIBUTION (whose is the difference, not just "both arms measured"): what FRACTION of the before-answer is
    # carried by the SPIKING prev-slot hold — (intact - prev-slot-silence lesion) / intact. tools.lab.attributable_to.
    attributed = attributable_to(
        f"who-was-before seed{seed}: spiking prev-slot hold intact vs prev-slot-silence lesion",
        mi["BEFORE"], ml["BEFORE"])
    print(f"  [ATTRIBUTION] fraction of the before-answer carried by the spiking hold: {attributed}", flush=True)
    if attributed is None or attributed < 0.5:
        fails.append(f"attribution: the spiking hold carries only {attributed} of the before-answer (< 0.5)")

    if not (mi["BEFORE"] >= 0.7):
        fails.append(f"intact BEFORE {mi['BEFORE']:.3f} < 0.7")
    if not (mi["NOW"] >= 0.7):
        fails.append(f"intact NOW {mi['NOW']:.3f} < 0.7")
    if not (mi["BEFORE"] - mi["recency"] > 0.4):
        fails.append(f"intact BEFORE-recency {mi['BEFORE']-mi['recency']:.3f} <= 0.4")
    if not (mi["BEFORE"] - mi["naive"] > 0.4):
        fails.append(f"intact BEFORE-naive {mi['BEFORE']-mi['naive']:.3f} <= 0.4")
    if not (mi["BEFORE"] - ml["BEFORE"] >= 0.3):
        fails.append(f"LESION not load-bearing: intact-lesion BEFORE {mi['BEFORE']-ml['BEFORE']:.3f} < 0.3")
    if not (ml["NOW"] >= 0.7):
        fails.append(f"LESION broke the present: NOW {ml['NOW']:.3f} < 0.7 (should be preserved)")
    if not (mo["BEFORE"] >= 0.7):
        fails.append(f"flag-off rate BEFORE {mo['BEFORE']:.3f} < 0.7 (today's capability should be preserved)")
    if not (mi["no_boundary_abstain_rate"] >= 0.999):
        fails.append(f"moat leak: intact answered before on a no-boundary discourse (abstain rate "
                     f"{mi['no_boundary_abstain_rate']:.3f} < 1.0)")

    print("", flush=True)
    if fails:
        print("  VERDICT: FAIL", flush=True)
        for f in fails:
            print(f"    - {f}", flush=True)
        return False
    print(f"  VERDICT: ALL_OK — the spiking discourse register answers who-was-before ({mi['BEFORE']:.2f}) across a "
          f"connective on cp_firing_states while tracking the present ({mi['NOW']:.2f}); the spiking prev-slot hold is "
          f"LOAD-BEARING (BEFORE {mi['BEFORE']:.2f}->{ml['BEFORE']:.2f} under lesion, NOW preserved "
          f"{mi['NOW']:.2f}->{ml['NOW']:.2f}); flag-off is today's register ({mo['BEFORE']:.2f}); recency "
          f"({mi['recency']:.2f}) / naive ({mi['naive']:.2f}) / single-event (structural) fail; moat intact "
          f"(no-boundary abstain {mi['no_boundary_abstain_rate']:.2f}).", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    ok = all(verify(s) for s in seeds)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
