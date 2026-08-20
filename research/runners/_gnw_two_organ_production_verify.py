"""PRODUCTION VERIFY for the TWO-GENUINELY-DISTINCT-ORGANS GNW coincidence bus wired into
`webapp/server.py::brain_chat` via `webapp/gnw_two_organ_bus.py`. NOW DEFAULT-ON (2026-08-20,
`webapp/gnw_two_organ_bus.py::two_organ_enabled` — unset env var means ON; explicit `BRAIN_GNW_2ORGAN=0`/`off` still
disables). Because unset now means ON, this runner forces the flag EXPLICITLY on every OFF-path measurement (never
relies on unset/pop) -- an unset env var here would silently install the bus and contaminate the "HEAD" baseline
with two-organ-bus output, making the byte-identical-when-off check compare the bus against itself.

Tested at the COMBINE level (`ChatBrain.gate` / `two_organ_combine`) — exactly where the wiring lives — on the REAL
production tiny-demo ChatBrain (built by `webapp.server._build_chat_brain`), numpy-CPU. (The full `brain_chat` handler
is a heavy numpy-CPU path that builds+steps many default-on organs per turn; the combine level is where the 2-organ
change actually operates and is fast + deterministic. The handler is byte-identical when off BY CONSTRUCTION: the
server hook that installs this bus is guarded by the SAME `BRAIN_GNW_2ORGAN` flag, so with it off the module is never
even imported and the executed code path is exactly HEAD.)

  (D) BYTE-IDENTICAL-WHEN-OFF (off == HEAD). On the production chat (with the existing N-organ bus installed as HEAD
      does), `install_two_organ_gate` with the flag OFF is a NO-OP (returns False, leaves `chat.gate` unwrapped), and
      even with the wrapper installed, a RUNTIME flag flip to OFF makes the wrapper delegate to the original gate — so
      the gate output is byte-identical to the HEAD (N-organ-bus) output on every panel turn. And with the flag ON, on
      a cleanly-stored fact the 2-organ coincidence commits the SAME patient (the answer bytes are unchanged — the flip
      changes the MECHANISM, not the behaviour, on a clean fact; only the lesion levers change the answer).
  (A) INTACT COVERED CLASS. With the flag ON, the 2-organ coincidence AUTHORS the covered routable recall: organ A
      (composer `query_patient`) + organ B (the production `SurpriseProductionOrgan` corroborating the candidate
      against its OWN expectation e_B) co-ignite -> committed patient == the host's recalled patient on every STORED turn.
  (B) LESION-LOAD-BEARING (two distinct levers). (i) ORGAN-B lesion (zero the patient_expected->surprise prediction
      edges) -> CONFIRM fires as high as CONTRADICT -> organ B withholds even on a match -> the coincidence collapses ->
      abstain. (ii) WORKSPACE lesion (zero the assembly self-recurrence) -> the coincidence cannot sustain -> abstain,
      WHILE the forward-recall reflex (direct query_patient) still answers (the dissociation).
  (C) MOAT-SAFE. On the abstain panel (no stored binding) organ A misses -> the bus abstains by construction.
  (D2) ORGAN-B DISCRIMINATES IN THE WIRED PATH. The organ's confirm (agree) Hz << its calibrated threshold << its
      contradict (disagree, via the organ-B lesion) Hz — on WHICHEVER backend this runs (load-bearing on cupy).
  (N) SAFE-INERT GATE. When organ B would NOT discriminate on this backend (backend-neutral init off on cupy),
      `install_two_organ_gate` returns False (inert) — the bus NEVER runs the mis-discriminating organ.

Run on BOTH backends (fast rf recall path). The organ discriminates on cupy via backend-neutral threshold init:
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_two_organ_production_verify
  SIM_BACKEND=cupy  BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_two_organ_production_verify
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path; ~s not ~180s)

# STORED turns the host answers + ABSTAIN turns the host withholds (moat). The 2-organ bus must reproduce BOTH.
STORED = [
    ("what does dog chase?", ["dog", "chase", "cat"]),
    ("what does cat eat?", ["cat", "eat", "fish"]),
    ("what does brain use?", ["brain", "use", "spikes"]),
    ("what does brain learn?", ["brain", "learn", "words"]),
    ("what does brain store?", ["brain", "store", "memory"]),
]
ABSTAIN = [
    "what does dog eat?",       # dog has a CHASE fact, not EAT -> the substrate has no binding -> moat abstain
    "what does cat chase?",     # cat has an EAT fact, not CHASE -> moat abstain
    "what does fish fly?",      # unstored agent/action -> moat abstain
]
PANEL = [q for q, _w in STORED] + ABSTAIN


def _svo(x):
    return list(x) if x is not None else None


def _production_chat():
    """The REAL production ChatBrain with the existing N-organ bus installed (exactly as `brain_chat` does at HEAD)."""
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    chat, _src = _build_chat_brain("tiny-demo", "stub")
    gbs.install_bus_gate(chat)                       # HEAD default: the N-organ (composer-only) bus authors the combine
    return chat


def run():
    from webapp import gnw_two_organ_bus as g2

    chat = _production_chat()
    # INSTRUMENT VALIDITY: the real production ChatBrain built with the surface this runner reads (`chat.inner.composer`
    # + `chat.stored_facts`) -- everything below assumes this, and a silent build regression would otherwise show up as
    # a downstream AttributeError rather than a named precondition.
    chat_built_ok = bool(hasattr(chat, "inner") and hasattr(chat.inner, "composer")
                         and hasattr(chat, "stored_facts")
                         and len(getattr(chat, "stored_facts", []) or []) >= len(STORED))

    # ── (D) phase 1 — flag OFF: install is a NO-OP; the gate output IS the HEAD (N-organ-bus) output. ──────────────
    # EXPLICIT "0", NOT unset/pop: `two_organ_enabled()` flipped DEFAULT-ON 2026-08-20 (unset now means ON, see
    # `webapp/gnw_two_organ_bus.py::two_organ_enabled`), so an unset env var no longer tests the OFF path -- it
    # would silently install the bus here too, contaminating `head_out` with two-organ-bus output and making every
    # downstream "== HEAD" comparison compare the bus against itself. Caught by the `install-off-is-noop`
    # precondition below (measured False on the very first re-run after this fix was needed).
    os.environ["BRAIN_GNW_2ORGAN"] = "0"
    installed_off = g2.install_two_organ_gate(chat)                       # expect False (no-op)
    off_noop = (installed_off is False) and (not getattr(chat, "_two_organ_installed", False))
    head_out = [_svo(chat.gate(q)) for q in PANEL]                        # the HEAD / N-organ-bus decisions

    # ── (D) phase 2 + (A) + (C) — flag ON: the 2-organ coincidence AUTHORS the covered class. ─────────────────────
    os.environ["BRAIN_GNW_2ORGAN"] = "1"
    from sim.backend import get_backend
    backend = get_backend()[1]
    organ_discriminates_gate = g2._organ_discriminates()                 # numpy always; cupy iff backend-neutral init on
    installed_on = g2.install_two_organ_gate(chat)                       # expect True (wraps chat.gate)
    on_out, stored_rows = [], []
    for q, want in STORED:
        svo = chat.gate(q)
        info = getattr(chat, "_last_two_organ", {})
        on_out.append(_svo(svo))
        stored_rows.append({"q": q, "svo": _svo(svo), "want": want,
                            "agrees_host": (_svo(svo) == want), "authored_by": info.get("authored_by"),
                            "organ_b_confirmed": info.get("organ_b_confirmed"),
                            "organ_b_surprise_hz": info.get("organ_b_surprise_hz"),
                            "organ_b_threshold_hz": info.get("organ_b_threshold_hz"), "n_ignited": info.get("n_ignited")})
    abstain_rows = []
    for q in ABSTAIN:
        svo = chat.gate(q)
        info = getattr(chat, "_last_two_organ", {})
        on_out.append(_svo(svo))
        abstain_rows.append({"q": q, "abstained": (svo is None), "reason": info.get("abstain_reason") or info.get("reason")})
    intact_ok = all(r["agrees_host"] for r in stored_rows)
    moat_ok = all(r["abstained"] for r in abstain_rows)
    # (D) the ANSWER bytes are unchanged on the whole panel: on a clean stored fact the coincidence commits the same
    # patient, and on an abstain both withhold -> the flag-ON gate output == the HEAD output, turn for turn.
    on_matches_head = (on_out == head_out)

    # ── (D) phase 3 — RUNTIME flag flip to OFF (wrapper still installed): delegates to the original gate == HEAD. ──
    # EXPLICIT "0" -- same reasoning as phase 1 (unset now means ON, post default-flip).
    os.environ["BRAIN_GNW_2ORGAN"] = "0"
    off_runtime_out = [_svo(chat.gate(q)) for q in PANEL]
    off_runtime_matches_head = (off_runtime_out == head_out)
    os.environ["BRAIN_GNW_2ORGAN"] = "0"

    # ── (B) LESION-LOAD-BEARING (combine level, both levers), REUSING the same production chat (read-only). ────────
    composer = chat.inner.composer
    organb_rows, organb_ok = [], True
    for (a, v, p) in chat.stored_facts:
        info = g2.two_organ_combine(chat, a, v, ws_lesion=False, organb_lesion=True)
        collapses = (info.get("committed") is None) and (info.get("organ_a_recall") == p)
        organb_ok = organb_ok and collapses
        organb_rows.append({"fact": [a, v, p], "committed": info.get("committed"),
                            "organ_a_recall": info.get("organ_a_recall"),
                            "organ_b_surprise_hz": info.get("organ_b_surprise_hz"), "collapses": collapses})
    ws_rows, ws_ok = [], True
    for (a, v, p) in chat.stored_facts:
        info = g2.two_organ_combine(chat, a, v, ws_lesion=True, organb_lesion=False)
        reflex = composer.query_patient(a, v)
        ok = (info.get("committed") is None) and (reflex == p)
        ws_ok = ws_ok and ok
        ws_rows.append({"fact": [a, v, p], "committed": info.get("committed"), "reflex": reflex,
                        "collapses": (info.get("committed") is None), "reflex_survives": (reflex == p)})

    # ── (N) SAFE-INERT GATE: when organ B would NOT discriminate on this backend (backend-neutral init off on cupy),
    #        install is inert (no chat build needed — the gate is checked BEFORE the wrapper touches chat.gate, so a
    #        light stub suffices). This is the safety fallback: the bus NEVER runs the mis-discriminating organ. ─────
    class _Stub:
        def gate(self, q):
            return None
    orig = g2._organ_discriminates
    try:
        g2._organ_discriminates = lambda: False
        g2._WARNED_CUPY = False
        os.environ["BRAIN_GNW_2ORGAN"] = "1"
        stub = _Stub()
        inst = g2.install_two_organ_gate(stub)
        inert_when_nondiscriminating = (inst is False) and (not getattr(stub, "_two_organ_installed", False))
    finally:
        g2._organ_discriminates = orig
        os.environ["BRAIN_GNW_2ORGAN"] = "0"       # leave the flag in a KNOWN off state, not the new ON default

    # ── the surprise organ's REAL-wired-path discrimination on THIS backend: agree (confirm) Hz << threshold <<
    #    disagree (contradict, via the organ-B lesion which removes the prediction) Hz. Load-bearing on cupy. ────────
    agree_hz = [r["organ_b_surprise_hz"] for r in stored_rows if r["organ_b_surprise_hz"] is not None]
    thr_hz = [r["organ_b_threshold_hz"] for r in stored_rows if r.get("organ_b_threshold_hz") is not None]
    disagree_hz = [r["organ_b_surprise_hz"] for r in organb_rows if r["organ_b_surprise_hz"] is not None]
    max_agree = (max(agree_hz) if agree_hz else None)
    thr_val = (min(thr_hz) if thr_hz else None)
    min_disagree = (min(disagree_hz) if disagree_hz else None)
    discriminates_wired = bool(max_agree is not None and thr_val is not None and min_disagree is not None
                               and max_agree < thr_val < min_disagree)

    discrimination_measured = bool(max_agree is not None and thr_val is not None and min_disagree is not None)

    # ── INSTRUMENT VALIDITY (preconditions). Everything registered here must hold for the outcome measures below
    #    to mean what they claim -- a FAILED precondition earns UNDEFINED, never a NO-GO (that is the exact
    #    affect-eviction miss `tools.verdict` exists to catch: a validity failure reported as a negative result).
    #    None of these are the mechanism's own commit/abstain/discriminate behaviour -- those stay plain booleans
    #    below and drive `go=` directly, so a genuine failure there reads as a real NO-GO, not UNDEFINED. ─────────
    vd = Verdict(f"GNW two-genuinely-distinct-organs bus — production verify (real ChatBrain.gate, rf/{backend})")
    vd.disabled("full per-turn organ-stepping brain_chat handler",
              why="tested at the ChatBrain.gate/combine level where the 2-organ wiring actually lives, not the "
                  "heavy numpy-CPU per-turn handler that steps every default-on organ each turn; byte-identical-"
                  "when-off is guaranteed by construction there (the server hook is gated by the SAME flag)")
    vd.require("backend-recognized", backend in ("numpy", "cupy"), expect=True,
              note="get_backend() returned a known production backend for this invocation "
                  "(this runner is invoked once per backend -- see the module docstring)")
    vd.require("production-chat-built", chat_built_ok, expect=True,
              note="the real production ChatBrain (webapp.server._build_chat_brain) built with the composer + "
                  "stored_facts surface this runner reads")
    vd.require("install-off-is-noop", off_noop, expect=True,
              note="flag OFF -> install_two_organ_gate is a no-op (proves the flag genuinely gates installation, "
                  "a precondition for reading the flag-ON measurements below as attributable to the flag)")
    vd.require("install-on-installs", installed_on, expect=True,
              note="flag ON -> install_two_organ_gate actually wraps chat.gate on the real production chat "
                  "(proves the ON-path teeth below were measured on an installed coincidence, not a no-op)")
    vd.require("byte-identical-when-off", off_runtime_matches_head, expect=True,
              note="wrapper installed but flag runtime-flipped OFF delegates to the ORIGINAL gate -> HEAD bytes "
                  "(the runtime safety escape actually works, not just the install-time no-op)")
    vd.require("organ-discriminates-on-this-backend", organ_discriminates_gate, expect=True,
              note="the pre-flight _organ_discriminates() gate reads True on this backend -- otherwise the "
                  "ON-path teeth below would be measuring the safe-inert fallback, not the coincidence")
    vd.require("discrimination-hz-recorded", discrimination_measured, expect=True,
              note=f"agree/threshold/disagree Hz values were actually recorded (agree={max_agree} thr={thr_val} "
                  f"disagree={min_disagree}) -- prerequisite for the discrimination teeth to mean anything")
    vd.require("safe-inert-fallback-works", inert_when_nondiscriminating, expect=True,
              note="(N) the bus's OWN safety branch, verified via a stub with _organ_discriminates forced False: "
                  "install_two_organ_gate returns False and never runs a mis-discriminating organ -- an "
                  "instrument-level property of the mechanism, independent of whether THIS backend discriminates")

    # ── TEETH — the actual GO conditions, left as plain booleans (NOT registered as preconditions) so a genuine
    #    failure reads NO-GO, not UNDEFINED: (A) stored recalls commit the host patient, (B) organ-B lesion +
    #    workspace lesion both collapse to abstain, (C) the moat abstains, (D2) the organ discriminates in the
    #    real wired path, and (D) the flag-ON answer bytes reproduce HEAD on the clean-fact panel. ────────────────
    go = bool(on_matches_head and intact_ok and moat_ok and organb_ok and ws_ok and discriminates_wired)
    decided = vd.decide(go=go)

    # ── attribute the commit/abstain behaviour to the two-organ coincidence itself (wired vs each lesion lever),
    #    rather than leaving two adjacent numbers for a reader to subtract by eye. ───────────────────────────────
    frac_commit_wired = (sum(1 for r in stored_rows if r["agrees_host"]) / len(stored_rows)) if stored_rows else 0.0
    abstain_rate_wired = 1.0 - frac_commit_wired
    abstain_rate_organb_lesion = ((sum(1 for r in organb_rows if r["collapses"]) / len(organb_rows))
                                  if organb_rows else 0.0)
    abstain_rate_ws_lesion = (sum(1 for r in ws_rows if r["collapses"]) / len(ws_rows)) if ws_rows else 0.0
    attr_organb = attributable_to(
        f"[{backend}] stored-fact abstention: organ-B-lesion vs wired coincidence",
        abstain_rate_organb_lesion, abstain_rate_wired)
    attr_ws = attributable_to(
        f"[{backend}] stored-fact abstention: workspace-lesion vs wired coincidence",
        abstain_rate_ws_lesion, abstain_rate_wired)

    print("\n" + "=" * 100, flush=True)
    print(f"  GNW TWO-DISTINCT-ORGANS BUS — PRODUCTION VERIFY (real ChatBrain.gate, rf/{backend})", flush=True)
    print("=" * 100, flush=True)
    print(f"  (D) BYTE-IDENTICAL-WHEN-OFF (off == HEAD): install no-op={off_noop} | runtime-flip-off == HEAD="
          f"{off_runtime_matches_head}", flush=True)
    print(f"      flag-ON gate output == HEAD (clean-fact answer bytes unchanged) = {on_matches_head}", flush=True)
    for q, h, o in zip(PANEL, head_out, on_out):
        print(f"      {q:24s} HEAD={h} ON={o} match={h == o}", flush=True)
    print(f"  (A) INTACT covered class (2-organ coincidence commits the host patient): {intact_ok}", flush=True)
    for r in stored_rows:
        print(f"      STORED  {r['q']:24s} svo={r['svo']} agrees={r['agrees_host']} by={r['authored_by']} "
              f"b_confirm={r['organ_b_confirmed']} hz={r['organ_b_surprise_hz']} n_ign={r['n_ignited']}", flush=True)
    print(f"  (C) MOAT-SAFE (abstain panel): {moat_ok}", flush=True)
    for r in abstain_rows:
        print(f"      ABSTAIN {r['q']:24s} abstained={r['abstained']} reason={r['reason']}", flush=True)
    print(f"  (B.i) ORGAN-B LESION collapses the coincidence (2nd organ's spiking prediction load-bearing): {organb_ok}", flush=True)
    for r in organb_rows:
        print(f"      {str(r['fact']):28s} committed={r['committed']!r} a_recall={r['organ_a_recall']!r} "
              f"hz={r['organ_b_surprise_hz']} collapses={r['collapses']}", flush=True)
    print(f"  (B.ii) WORKSPACE LESION collapses; reflex survives (dissociation): {ws_ok}", flush=True)
    for r in ws_rows:
        print(f"      {str(r['fact']):28s} committed={r['committed']!r} reflex={r['reflex']!r} "
              f"collapses={r['collapses']} reflex_survives={r['reflex_survives']}", flush=True)
    print(f"  (D2) ORGAN-B DISCRIMINATES IN THE WIRED PATH (agree << thr << disagree): {discriminates_wired}  "
          f"[max_agree={max_agree} << thr={thr_val} << min_disagree={min_disagree}]", flush=True)
    print(f"  (N) SAFE-INERT when organ would NOT discriminate (backend-neutral init off): {inert_when_nondiscriminating}  "
          f"(this run backend={backend}, organ_discriminates_gate={organ_discriminates_gate})", flush=True)
    print(f"  ATTRIBUTABLE: organ-B-lesion vs wired = {attr_organb}  |  workspace-lesion vs wired = {attr_ws}", flush=True)
    print(f"\n  VERDICT: {decided['status']}\n" + "=" * 100, flush=True)

    out = {"runner": "_gnw_two_organ_production_verify", "go": decided["go"], "status": decided["status"],
           "verdict": decided, "backend": backend,
           "byte_identical_when_off": {"ok": bool(off_noop and off_runtime_matches_head),
                                       "install_noop_when_off": off_noop,
                                       "flag_on_matches_head": on_matches_head,
                                       "runtime_flip_off_matches_head": off_runtime_matches_head,
                                       "head_out": head_out, "on_out": on_out, "off_runtime_out": off_runtime_out},
           "intact_ok": intact_ok, "moat_ok": moat_ok, "organb_lesion_ok": organb_ok, "ws_lesion_ok": ws_ok,
           "organ_discriminates_gate": organ_discriminates_gate, "inert_when_nondiscriminating": inert_when_nondiscriminating,
           "discrimination_wired": {"ok": discriminates_wired, "max_agree_hz": max_agree,
                                    "threshold_hz": thr_val, "min_disagree_hz": min_disagree},
           "stored": stored_rows, "abstain": abstain_rows, "organb_lesion": organb_rows, "ws_lesion": ws_rows,
           # top-level (not just nested under "verdict") so tools/gates/verdict_preconditions.py can enforce
           # presence directly, per its documented schema (it scans top-level go/GO/verdict/status + preconditions).
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"],
           "attributable": {"organb_lesion_vs_wired": attr_organb, "ws_lesion_vs_wired": attr_ws,
                            "abstain_rate_wired": round(abstain_rate_wired, 4),
                            "abstain_rate_organb_lesion": round(abstain_rate_organb_lesion, 4),
                            "abstain_rate_ws_lesion": round(abstain_rate_ws_lesion, 4)}}
    op = f"research/findings/raw/_gnw_two_organ/production_verify_{backend}.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if decided["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(run())
