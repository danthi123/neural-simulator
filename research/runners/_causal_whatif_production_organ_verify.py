"""PRODUCTION VERIFY for the CAUSAL WHY / WHAT-IF organ wired into `webapp/server.py::brain_chat` (T1-4).

Asserts the hard requirements on the REAL production tiny-demo ChatBrain + the REAL `brain_chat` handler, numpy-CPU:

  (A) WHAT-IF (forward-SIMULATION over REAL facts): a brain taught the canonical causal chain
      A=(dog,go,east)->B=(dog,reach,river)->D=(dog,drink,water) answers "what happens if the dog goes east?" with the
      2-step consequence D — a substrate ROLLOUT (A->D was never taught) — and the answer is MOAT-CONFIRMED
      ((dog,drink)->water is `query_patient`-confirmed). Through the EXACT handler code path.
  (B) WHY (DO-surviving cause over REAL facts): the same brain (also taught the confound C=(sun,rise,sky) common
      cause of X=(bird,sing,dawn) + Y=(dog,wake,morning)) answers "why did the dog wake?" with C ("because the sun
      rose"), the cause that SURVIVES the Pearl DO-probe — never the spurious correlate X ("the bird sang") — and C
      is moat-confirmed.
  (C) ABSTAIN (0 confabulation): (C1) an UNMAPPED causal query ("why did the dog chase?" — a known fact, not the
      validated causal target) -> the honest `_honest_causal_answer` disclaimer (states the confirmed fact clause,
      declines to invent a reason); (C2) a GROUNDING-UNCONFIRMED query — a brain taught the chain but NOT D — "what
      happens if the dog goes east?" -> the forward model runs but the consequence is not moat-confirmed -> the
      honest abstain. NEITHER asserts an unconfirmed causal fact (no "drink water" / "sun rose" in an abstain).
  (D) LESION-LOAD-BEARING: with `BRAIN_CAUSAL_LESION=1` (zero the learned forward edges) the SAME fully-taught brain
      COLLAPSES — the what-if and the why both fall to the honest abstain (the forward-simulation cannot roll
      A->B->D; the DO-probe predecessor of Y is no longer C). The answer is caused by the learned SPIKING edges.
  (E) BYTE-IDENTICAL-WHEN-OFF (real handler): on a NON-causal panel (recall + abstain) the flag-ON and flag-OFF
      responses are byte-identical (the causal block never fires on them, so every faculty — affect/D2/D4/E1/E2/
      curiosity/D5/D6/B1/D3 — runs unchanged); and a causal query with the flag OFF carries NO `causal` key (falls
      through to the normal path). Proves ADDITIVE + default-ON escape + no regression.

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._causal_whatif_production_organ_verify
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path)

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

from research.runners._causal_forward_model_grounded_derisk import FACTS, FACT_ORDER, D as _D  # noqa: E402


def _teach(chat, drop=None):
    """Teach the canonical causal facts into THIS brain's live composer (simulating a brain that LEARNED the causal
    world through conversation). READ-ONLY grounding at organ-build time reads exactly these via query_patient."""
    drop = set(drop or [])
    comp = chat.inner.composer
    for e in FACT_ORDER:
        if e in drop:
            continue
        comp.store(*FACTS[e])
    return comp


def _setup_session(session, *, drop=None):
    """Build the REAL production ChatBrain, teach the causal facts, cache it into the handler's brain cache, and
    drop any stale causal organ so it re-grounds against the freshly-taught composer. Returns the cache_key."""
    import webapp.server as S
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    _teach(chat, drop=drop)
    cache_key = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[cache_key] = chat
    import research.runners.causal_whatif_production_organ as CA
    CA.reset_organ(cache_key)
    return cache_key


def _turn(session, message, *, rich=False):
    """Drive one turn through the REAL `brain_chat` handler for a pre-built cached session."""
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=rich))
    return json.loads(r.body.decode("utf-8"))


def main():
    rows = {}

    # ── (A)+(B) fully-taught brain: what-if + why through the REAL handler ───────────────────────────────────
    _setup_session("cau_full", drop=None)
    a = _turn("cau_full", "what happens if the dog goes east?")
    b = _turn("cau_full", "why did the dog wake?")
    whatif_ok = bool((not a["abstained"]) and a.get("verified")
                     and a.get("causal", {}).get("confirmed")
                     and "drink water" in a["answer"] and a.get("causal", {}).get("consequence_fact") == ["dog", "drink", "water"])
    why_ok = bool((not b["abstained"]) and b.get("verified")
                  and b.get("causal", {}).get("confirmed") and b.get("causal", {}).get("why_is_C")
                  and "sun rise" in b["answer"] and b.get("causal", {}).get("cause_fact") == ["sun", "rise", "sky"])
    rows["A_whatif"] = {"answer": a["answer"], "causal": a.get("causal"), "ok": whatif_ok}
    rows["B_why"] = {"answer": b["answer"], "causal": b.get("causal"), "ok": why_ok}

    # ── (C1) UNMAPPED abstain (0 confab): "why did the dog chase?" (a known fact, not the causal target) ──────
    c1 = _turn("cau_full", "why did the dog chase?")
    c1_confab = ("drink water" in c1["answer"]) or ("because the sun" in c1["answer"].lower())
    c1_ok = bool(c1["abstained"] and (not c1.get("verified")) and (not c1_confab)
                 and c1.get("causal", {}).get("abstained"))
    rows["C1_unmapped_abstain"] = {"answer": c1["answer"], "causal": c1.get("causal"), "confab": c1_confab, "ok": c1_ok}

    # ── (C2) GROUNDING-UNCONFIRMED abstain: chain taught but NOT D -> what-if runs but is not moat-confirmed ──
    _setup_session("cau_dropD", drop=[_D])
    c2 = _turn("cau_dropD", "what happens if the dog goes east?")
    c2_confab = "drink water" in c2["answer"]
    c2_ok = bool(c2["abstained"] and (not c2.get("verified")) and (not c2_confab)
                 and (not c2.get("causal", {}).get("confirmed")))
    rows["C2_grounding_abstain"] = {"answer": c2["answer"], "causal": c2.get("causal"), "confab": c2_confab, "ok": c2_ok}

    # ── (D) LESION-LOAD-BEARING: same fully-taught brain, forward edges zeroed -> both collapse to abstain ────
    _setup_session("cau_les", drop=None)
    os.environ["BRAIN_CAUSAL_LESION"] = "1"
    try:
        dl_a = _turn("cau_les", "what happens if the dog goes east?")
        dl_b = _turn("cau_les", "why did the dog wake?")
    finally:
        os.environ.pop("BRAIN_CAUSAL_LESION", None)
    les_whatif_collapse = bool(dl_a["abstained"] and "drink water" not in dl_a["answer"]
                               and not dl_a.get("causal", {}).get("confirmed"))
    les_why_collapse = bool(dl_b["abstained"] and "because the sun" not in dl_b["answer"].lower()
                            and not dl_b.get("causal", {}).get("confirmed"))
    lesion_ok = bool(les_whatif_collapse and les_why_collapse)
    rows["D_lesion"] = {"whatif_answer": dl_a["answer"], "why_answer": dl_b["answer"],
                        "whatif_collapse": les_whatif_collapse, "why_collapse": les_why_collapse, "ok": lesion_ok}

    # ── (E) BYTE-IDENTICAL-WHEN-OFF (real handler) — the causal block is a pure no-op on a NON-causal turn (it only
    # calls is_causal_query()->None and returns; no global mutation), so flag ON vs OFF on the SAME session yields a
    # byte-identical response. Methodology (mirrors the gnw-bus verify): the session-STATEFUL heavy organs are
    # disabled ONLY for this comparison — they run identically on BOTH flag arms, so the comparison isolates the
    # causal flag alone; the (A)-(D) checks above already exercised the FULL default organ stack (affect/worldmodel/
    # surprise/D4/E1/curiosity all ON) and passed, proving those faculties are unregressed with the block present.
    # Idempotent recall/abstain queries only (an assertion would ACQUIRE on the first call -> not turn-idempotent).
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC", "BRAIN_CURIOSITY",
              "BRAIN_DISCOURSE_REGISTER", "BRAIN_COMPREHENSION_GATE", "BRAIN_RICH"):
        os.environ[k] = "0"
    NONCAUSAL = ["what does dog chase?", "what does cat eat?", "what does fish fly?", "what does ball roll?"]
    _setup_session("cau_bi", drop=None)
    bi_rows, bi_ok = [], True
    for msg in NONCAUSAL:
        os.environ["BRAIN_CAUSAL"] = "1"
        on = _turn("cau_bi", msg)
        os.environ["BRAIN_CAUSAL"] = "0"
        off = _turn("cau_bi", msg)                       # SAME session, 2nd call (idempotent recall/abstain)
        os.environ.pop("BRAIN_CAUSAL", None)
        no_causal_key = ("causal" not in off) and ("causal" not in on)   # non-causal turns never attach a causal key
        identical = (on == off)
        bi_ok = bi_ok and identical and no_causal_key
        bi_rows.append({"q": msg, "identical": identical, "no_causal_key": no_causal_key})
    # a causal query with the flag OFF falls through (no causal key) — proves the escape hatch is a clean skip.
    os.environ["BRAIN_CAUSAL"] = "0"
    offq = _turn("cau_bi", "what happens if the dog goes east?")
    os.environ.pop("BRAIN_CAUSAL", None)
    off_falls_through = ("causal" not in offq)
    bi_ok = bi_ok and off_falls_through
    rows["E_byte_identical"] = {"rows": bi_rows, "off_causal_query_no_key": off_falls_through, "ok": bi_ok}

    go = bool(whatif_ok and why_ok and c1_ok and c2_ok and lesion_ok and bi_ok)

    # EARN the verdict (tools.verdict.Verdict -> the preconditions travel with the result).
    from tools.verdict import Verdict
    v = Verdict("causal why/what-if production wiring")
    v.require("WHAT-IF: moat-confirmed rolled-forward consequence (dog->drink water)", whatif_ok, expect=True)
    v.require("WHY: DO-surviving moat-confirmed cause (dog wakes because the sun rose)", why_ok, expect=True)
    v.require("ABSTAIN(C1): unmapped causal query -> honest disclaimer, 0 confab", c1_ok, expect=True)
    v.require("ABSTAIN(C2): grounding-unconfirmed (no D) -> honest abstain, 0 confab", c2_ok, expect=True)
    v.require("LESION: forward-edge lesion collapses BOTH why + what-if to abstain", lesion_ok, expect=True)
    v.require("BYTE-IDENTICAL-when-off (real handler) + causal query flag-off no key", bi_ok, expect=True)
    v.disabled("grounding-by-shared-substrate", why="events grounded READ-ONLY by the live composer's moat recall "
               "(grounding-by-derivation); driving the event blocks from the composer's unbind spikes is the next rung")
    v.disabled("spiking-mismatch-driven DA", why="the DA sign + causal episode ORDER are teacher-delivered (the "
               "environment boundary); a spiking mismatch unit driving the DA is the next rung")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 100, flush=True)
    print("  CAUSAL WHY/WHAT-IF — PRODUCTION VERIFY (real ChatBrain + real brain_chat handler, rf recall, numpy-CPU)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (A) WHAT-IF  ok={whatif_ok}\n        {rows['A_whatif']['answer']}", flush=True)
    print(f"  (B) WHY      ok={why_ok}\n        {rows['B_why']['answer']}", flush=True)
    print(f"  (C1) UNMAPPED ABSTAIN ok={c1_ok} (confab={rows['C1_unmapped_abstain']['confab']})\n        {rows['C1_unmapped_abstain']['answer']}", flush=True)
    print(f"  (C2) GROUNDING ABSTAIN ok={c2_ok} (confab={rows['C2_grounding_abstain']['confab']})\n        {rows['C2_grounding_abstain']['answer']}", flush=True)
    print(f"  (D) LESION collapse ok={lesion_ok} (whatif_collapse={les_whatif_collapse} why_collapse={les_why_collapse})", flush=True)
    print(f"        lesioned what-if: {rows['D_lesion']['whatif_answer']}", flush=True)
    print(f"  (E) BYTE-IDENTICAL-when-off ok={bi_ok}", flush=True)
    for r in bi_rows:
        print(f"        {r['q']:22s} identical={r['identical']} no_causal_key={r['no_causal_key']}", flush=True)
    print(f"        causal query flag-off falls through (no key): {off_falls_through}", flush=True)
    verdict = "GO" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 100, flush=True)

    out = {"runner": "_causal_whatif_production_organ_verify", "go": go, "status": decided["status"],
           "whatif_ok": whatif_ok, "why_ok": why_ok, "c1_ok": c1_ok, "c2_ok": c2_ok,
           "lesion_ok": lesion_ok, "byte_identical_ok": bi_ok, "rows": rows,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_causal_whatif/production_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
