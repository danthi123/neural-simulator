"""PRODUCTION VERIFY for the PROSPECTIVE MEMORY organ wired into `webapp/server.py::brain_chat` (Gate-B, 2026-08-13).

Asserts the hard requirements on the REAL production tiny-demo ChatBrain + the REAL `brain_chat` handler, numpy-CPU,
by running a MULTI-TURN conversation through the exact handler code path (a persistent co-resident spiking latch
holds the intention BETWEEN turns):

  (A) FIRES-ON-CUE + SILENT-BEFORE (a full conversation): turn 1 FORMS the intention ("remind me to water the plants
      when I mention the garden") -> a disjoint acknowledgement (prospective.kind=formation, held). Several INTERVENING
      distractor turns HOLD the intention SILENT (no reminder in the answer; prospective.fired=False; the held assembly
      stays alive). The CUE turn ("the garden is blooming nicely") FIRES: the answer is PREPENDED with the reminder and
      prospective.fired=True — a spiking held x cue coincidence read off cp_firing_states, never a host string match.
  (B) SILENT-ON-WRONG-CUE: (B1) a turn that mentions a DIFFERENT topic (not the registered cue) does NOT fire; (B2) a
      DIRECT spiking-specificity read — driving the UNLATCHED slot-B cue assembly while the slot-A intention is held
      keeps rel_A sub-threshold (the cue-monitor is cue-SPECIFIC on spikes, not merely "we didn't drive it").
  (C) LESION-LOAD-BEARING: with `BRAIN_PMEM_LESION=1` the SAME intention is formed but the latch is zeroed at
      formation (the held assembly collapses) -> the SAME cue turn does NOT fire -> NO reminder. The fire is caused by
      the SPIKING latch (the coincidence), not the host cue-match: intact fires, lesioned is silent.
  (D) BYTE-IDENTICAL-WHEN-OFF (real handler): on a NON-prospective panel (recall + abstain) the flag-ON and flag-OFF
      responses are byte-identical (no held intention -> the block is a pure no-op -> no `prospective` key on either);
      and a "remind me..." FORMATION-phrased turn with the flag OFF carries NO `prospective` key and is NOT the
      acknowledgement (it falls through to the normal path). Proves ADDITIVE + default-ON escape + no regression.
  (E) LEARNED CUE->ACTION BINDING (the retirement rung, 2026-08-13): the cue->action content binding is LEARNED via a
      ONE-SHOT HEBBIAN potentiation at intention-formation (Gollwitzer implementation-intention), NOT installed at
      build. (E1) the binding is ABSENT before formation (live |w|~0 read off the substrate CSR), LEARNED to ~canonical
      by the formation event, and the cue FIRES. (E2) the BINDING lesion (`BRAIN_PMEM_HEBBIAN_LESION=1`, latch WITHOUT
      the Hebbian event) leaves the binding absent -> the cue does NOT fire (load-bearing: the fire is caused by the
      learned formation event, not a residual install). (E3) the `BRAIN_PMEM_HEBBIAN=0` escape reverts to the
      build-time install (fires; no hebbian markers) — byte-identical to the pre-wiring organ.

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._prospective_memory_production_verify
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

from research.runners._pmem_intention_latch_derisk import FIRE_THR  # noqa: E402  (the frozen release threshold)

_REMIND = "remind me to water the plants when I mention the garden"
_CUE_TURN = "the garden is blooming nicely today"          # contains the registered cue word "garden" -> fires
_WRONG_CUE = "the weather looks stormy this afternoon"     # a different topic -> a distractor -> stays silent
_DISTRACTORS = ["what does the dog chase?", "what does the cat eat?", "tell me about the sky"]


def _setup_session(session):
    """Build the REAL production ChatBrain (tiny-demo, stub renderer), cache it into the handler's brain cache, and
    drop any stale per-session prospective organ. Returns the cache_key."""
    import webapp.server as S
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    cache_key = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[cache_key] = chat
    S._SESSION_PMEM.pop(cache_key, None)
    return cache_key


def _turn(session, message, *, rich=False):
    """Drive one turn through the REAL `brain_chat` handler for a pre-built cached session."""
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=rich))
    return json.loads(r.body.decode("utf-8"))


def main():
    rows = {}

    # ── (A) FIRES-ON-CUE + SILENT-BEFORE — a full multi-turn conversation through the REAL handler ─────────────
    _setup_session("pm_full")
    t_form = _turn("pm_full", _REMIND)
    form_ok = bool((not t_form["abstained"])
                   and t_form.get("prospective", {}).get("kind") == "formation"
                   and t_form.get("prospective", {}).get("held") is True
                   and "remind you to water the plants" in t_form["answer"])
    # intervening distractor turns: the intention is HELD and the monitor stays SILENT (no reminder). The held/rel
    # read is taken from the ORGAN's own last_read (the prospective block runs BEFORE any disjoint short-circuit, so
    # the organ state is always current even on a turn whose RESPONSE omits the prospective debug key).
    import webapp.server as S
    _ck_full = ("pm_full", "tiny-demo", "stub")
    inter_rows, inter_silent, held_alive = [], True, True
    for d in _DISTRACTORS:
        ti = _turn("pm_full", d)
        org = S._SESSION_PMEM.get(_ck_full)
        lr = (org.last_read if org is not None else None) or {}
        fired = bool(lr.get("fired"))
        held = float(lr.get("held", 0.0))
        reminder_in_answer = ti["answer"].startswith("(Reminder")
        inter_silent = inter_silent and (not fired) and (not reminder_in_answer)
        held_alive = held_alive and (held > 0.0)
        inter_rows.append({"q": d, "fired": fired, "held": round(held, 4), "reminder_leak": reminder_in_answer})
    # the CUE turn: the held x cue coincidence FIRES -> the reminder is prepended.
    t_cue = _turn("pm_full", _CUE_TURN)
    pc = t_cue.get("prospective", {})
    fire_ok = bool(pc.get("kind") == "monitor" and pc.get("is_cue") is True and pc.get("fired") is True
                   and float(pc.get("rel", 0.0)) >= FIRE_THR
                   and t_cue["answer"].startswith("(Reminder")
                   and "water the plants" in t_cue["answer"])
    rows["A_fire_on_cue"] = {"formation": {"answer": t_form["answer"], "prospective": t_form.get("prospective"), "ok": form_ok},
                             "intervening": inter_rows, "silent_before_ok": inter_silent, "held_alive_ok": held_alive,
                             "cue": {"answer": t_cue["answer"], "prospective": pc, "ok": fire_ok}}
    a_ok = bool(form_ok and inter_silent and held_alive and fire_ok)

    # ── (B) SILENT-ON-WRONG-CUE — (B1) a different-topic turn does not fire; (B2) a DIRECT spiking specificity read
    _setup_session("pm_wrong")
    _turn("pm_wrong", _REMIND)                    # form the garden intention
    t_wrong = _turn("pm_wrong", _WRONG_CUE)       # a different topic (weather) -> distractor -> silent
    pw = t_wrong.get("prospective", {})
    b1_ok = bool(pw.get("fired") is False and (not t_wrong["answer"].startswith("(Reminder")))
    # (B2) DIRECT spiking-specificity: drive the UNLATCHED slot-B cue assembly while slot-A is held -> rel_A silent.
    import webapp.server as S
    porg = S._SESSION_PMEM.get(("pm_wrong", "tiny-demo", "stub"))
    rel_a_on_wrong_cue = float(porg.read_named_cue("B")) if porg is not None else None
    b2_ok = bool(rel_a_on_wrong_cue is not None and rel_a_on_wrong_cue < FIRE_THR)
    b_ok = bool(b1_ok and b2_ok)
    rows["B_wrong_cue"] = {"b1_topic": {"answer": t_wrong["answer"], "prospective": pw, "ok": b1_ok},
                           "b2_spiking_specificity": {"rel_A_on_slotB_cue": rel_a_on_wrong_cue,
                                                      "threshold": float(FIRE_THR), "ok": b2_ok}, "ok": b_ok}

    # ── (C) LESION-LOAD-BEARING — same intention, latch zeroed at formation -> the SAME cue does NOT fire ────────
    _setup_session("pm_lesion")
    os.environ["BRAIN_PMEM_LESION"] = "1"
    try:
        tl_form = _turn("pm_lesion", _REMIND)
        # intervening turns (the collapsed latch cannot re-sustain)
        for d in _DISTRACTORS:
            _turn("pm_lesion", d)
        tl_cue = _turn("pm_lesion", _CUE_TURN)
    finally:
        os.environ.pop("BRAIN_PMEM_LESION", None)
    pl_form = tl_form.get("prospective", {})
    pl_cue = tl_cue.get("prospective", {})
    # ATTRIBUTION (tools.lab): the reminder FIRE amplitude owed to the SPIKING latch (the held x cue coincidence),
    # not the host cue-match — intact cue-rel vs latch-lesioned cue-rel (the cue-detection is IDENTICAL in both arms;
    # only the latch is zeroed), read off the organ's own last_read for robustness.
    from tools.lab import attributable_to
    _org_full = S._SESSION_PMEM.get(("pm_full", "tiny-demo", "stub"))
    _org_les = S._SESSION_PMEM.get(("pm_lesion", "tiny-demo", "stub"))
    intact_fire_rel = float((pc or {}).get("rel", 0.0))
    lesion_fire_rel = float((pl_cue or {}).get("rel", (_org_les.last_read or {}).get("rel", 0.0) if _org_les else 0.0))
    fire_attribution = attributable_to("reminder FIRE owed to the spiking latch (intact cue-rel vs latch-lesioned cue-rel)",
                                       intact_fire_rel, lesion_fire_rel)
    lesion_collapse = bool(pl_cue.get("fired") is False and (not tl_cue["answer"].startswith("(Reminder")))
    # the lesion actually collapsed the held assembly at formation (a measured collapse, not just "did not fire").
    held_after_lesion = pl_form.get("held_after_lesion")
    lesion_measured = bool(held_after_lesion is not None and float(held_after_lesion) <= 0.02)
    lesion_ok = bool(lesion_collapse and lesion_measured and pl_form.get("lesioned") is True)
    # the CONTRAST: intact (A) fired; lesion is silent -> the fire is caused by the spiking latch.
    intact_vs_lesion_ok = bool(fire_ok and lesion_collapse)
    rows["C_lesion"] = {"formation": {"lesioned": pl_form.get("lesioned"), "held_after_lesion": held_after_lesion},
                        "cue": {"answer": tl_cue["answer"], "prospective": pl_cue},
                        "intact_fire_rel": intact_fire_rel, "lesion_fire_rel": lesion_fire_rel,
                        "fire_attribution": fire_attribution,
                        "collapse_ok": lesion_collapse, "measured_collapse_ok": lesion_measured,
                        "intact_fires_vs_lesion_silent": intact_vs_lesion_ok, "ok": lesion_ok}

    # ── (D) BYTE-IDENTICAL-WHEN-OFF (real handler). The prospective block is a pure no-op when NO intention is held
    # (parse_intention->None, no per-session organ built), so flag ON vs OFF on the SAME session yields a byte-
    # identical response with NO `prospective` key. The session-STATEFUL heavy organs are disabled ONLY for this
    # comparison (they run identically on BOTH arms); (A)-(C) above exercised the FULL default organ stack.
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC", "BRAIN_CURIOSITY",
              "BRAIN_DISCOURSE_REGISTER", "BRAIN_COMPREHENSION_GATE", "BRAIN_CAUSAL", "BRAIN_RICH"):
        os.environ[k] = "0"
    NONPROSPECTIVE = ["what does dog chase?", "what does cat eat?", "what does fish fly?", "what does the dragon do?"]
    _setup_session("pm_bi")
    bi_rows, bi_ok = [], True
    for msg in NONPROSPECTIVE:
        os.environ["BRAIN_PMEM"] = "1"
        on = _turn("pm_bi", msg)
        os.environ["BRAIN_PMEM"] = "0"
        off = _turn("pm_bi", msg)                        # SAME session, 2nd call (idempotent recall/abstain)
        os.environ.pop("BRAIN_PMEM", None)
        no_key = ("prospective" not in off) and ("prospective" not in on)
        identical = (on == off)
        bi_ok = bi_ok and identical and no_key
        bi_rows.append({"q": msg, "identical": identical, "no_prospective_key": no_key})
    # a FORMATION-phrased turn with the flag OFF carries NO prospective key and is NOT the acknowledgement (falls through).
    os.environ["BRAIN_PMEM"] = "0"
    off_form = _turn("pm_bi", _REMIND)
    os.environ.pop("BRAIN_PMEM", None)
    off_form_falls_through = bool(("prospective" not in off_form)
                                  and ("remind you to water the plants" not in off_form["answer"]))
    bi_ok = bi_ok and off_form_falls_through
    rows["D_byte_identical"] = {"rows": bi_rows, "off_formation_falls_through": off_form_falls_through, "ok": bi_ok}

    # ── (E) LEARNED CUE->ACTION BINDING (the retirement rung, 2026-08-13) — the binding is LEARNED one-shot at
    # formation (Gollwitzer implementation-intention), NOT installed at build: ABSENT before formation, PRESENT
    # (canonical) after, load-bearing (the BINDING lesion -> no fire), and the BRAIN_PMEM_HEBBIAN=0 escape reverts to
    # the build-time install (byte-identical fire, no hebbian markers). Verified through the REAL handler. ──────────
    import webapp.server as S
    # E1: absent-before / present-after + fires (default: the Hebbian binding is ON). The live binding norm is read
    # off the organ's own substrate (the SAME CSR read the de-risk anti-cheat uses).
    _setup_session("pm_hebb")
    _ck_h = ("pm_hebb", "tiny-demo", "stub")
    porg_h = S._get_pmem_organ(_ck_h)
    pm_h = porg_h._ensure_pm()
    norm_before = float(pm_h.binding_weight_norm()) if hasattr(pm_h, "binding_weight_norm") else None
    t_hform = _turn("pm_hebb", _REMIND)
    ph = t_hform.get("prospective", {})
    learned_norm = float((ph.get("binding_learned") or {}).get("learned_norm", 0.0))
    norm_after = float(pm_h.binding_weight_norm()) if hasattr(pm_h, "binding_weight_norm") else None
    for d in _DISTRACTORS:
        _turn("pm_hebb", d)
    t_hcue = _turn("pm_hebb", _CUE_TURN)
    pch = t_hcue.get("prospective", {})
    e1_ok = bool(ph.get("hebbian") is True
                 and norm_before is not None and norm_before <= 1e-3            # binding ABSENT before formation
                 and learned_norm > 0.0                                        # the Hebbian event LEARNED it
                 and norm_after is not None and norm_after > 0.5 * learned_norm  # present (~canonical) after
                 and pch.get("fired") is True and float(pch.get("rel", 0.0)) >= FIRE_THR
                 and t_hcue["answer"].startswith("(Reminder"))
    # E2: BINDING LESION (BRAIN_PMEM_HEBBIAN_LESION=1) — latch WITHOUT the Hebbian event -> binding absent -> no fire.
    _setup_session("pm_hebb_les")
    os.environ["BRAIN_PMEM_HEBBIAN_LESION"] = "1"
    try:
        _turn("pm_hebb_les", _REMIND)
        for d in _DISTRACTORS:
            _turn("pm_hebb_les", d)
        t_hles_cue = _turn("pm_hebb_les", _CUE_TURN)
    finally:
        os.environ.pop("BRAIN_PMEM_HEBBIAN_LESION", None)
    phl = t_hles_cue.get("prospective", {})
    e2_ok = bool(phl.get("fired") is False and (not t_hles_cue["answer"].startswith("(Reminder")))
    # E3: ESCAPE (BRAIN_PMEM_HEBBIAN=0) -> the build-time install path fires with NO hebbian/binding_learned markers.
    _setup_session("pm_install")
    os.environ["BRAIN_PMEM_HEBBIAN"] = "0"
    try:
        t_iform = _turn("pm_install", _REMIND)
        for d in _DISTRACTORS:
            _turn("pm_install", d)
        t_icue = _turn("pm_install", _CUE_TURN)
    finally:
        os.environ.pop("BRAIN_PMEM_HEBBIAN", None)
    pif, pic = t_iform.get("prospective", {}), t_icue.get("prospective", {})
    e3_ok = bool(("hebbian" not in pif) and ("binding_learned" not in pif)      # install path: no hebbian markers
                 and pic.get("fired") is True and float(pic.get("rel", 0.0)) >= FIRE_THR
                 and t_icue["answer"].startswith("(Reminder"))
    e_ok = bool(e1_ok and e2_ok and e3_ok)
    rows["E_learned_binding"] = {
        "e1_learned_fires": {"norm_before": norm_before, "learned_norm": learned_norm, "norm_after": norm_after,
                             "hebbian": ph.get("hebbian"), "cue_fired": pch.get("fired"), "ok": e1_ok},
        "e2_binding_lesion_silent": {"fired": phl.get("fired"), "answer": t_hles_cue["answer"], "ok": e2_ok},
        "e3_install_escape_fires": {"install_no_hebbian_keys": ("hebbian" not in pif),
                                    "cue_fired": pic.get("fired"), "ok": e3_ok}, "ok": e_ok}

    go = bool(a_ok and b_ok and lesion_ok and bi_ok and intact_vs_lesion_ok and e_ok)

    # EARN the verdict (tools.verdict.Verdict -> the preconditions travel with the result).
    from tools.verdict import Verdict
    v = Verdict("prospective memory production wiring")
    v.require("FIRES-ON-CUE + SILENT-BEFORE: formed, held silent across distractors, fired on the cue", a_ok, expect=True)
    v.require("SILENT-ON-WRONG-CUE: different topic silent + spiking cue-specificity (rel_A<thr on slot-B cue)", b_ok, expect=True)
    v.require("LESION-LOAD-BEARING: latch zeroed at formation -> held collapses -> the cue does NOT fire", lesion_ok, expect=True)
    v.require("INTACT fires vs LESION silent (the fire is caused by the spiking latch)", intact_vs_lesion_ok, expect=True)
    v.require("BYTE-IDENTICAL-when-off (real handler) + formation-phrased flag-off no key/no ack", bi_ok, expect=True)
    v.require("LEARNED cue->action binding: absent before formation, ~canonical after, fires on cue", e1_ok, expect=True)
    v.require("BINDING lesion (latch without the Hebbian event) -> binding absent -> the cue does NOT fire", e2_ok, expect=True)
    v.require("BRAIN_PMEM_HEBBIAN=0 escape reverts to the build-time install (fires; no hebbian markers)", e3_ok, expect=True)
    v.disabled("engine-native STDP realization of the local Hebbian rule",
               why="the cue->action CONTENT binding is now LEARNED one-shot at formation (a host-applied local pre x "
               "post outer product on real cp_firing_states; Gollwitzer implementation-intention) — the build-time "
               "synaptic INSTALL is RETIRED. What remains host: the intention/cue TEXT->slot mapping + cue-presence "
               "(a language/sensory boundary), the formation goal-activation drive, and the operating-point "
               "calibration. The HOLD + coincidence-gated RELEASE are spiking; the engine-native STDP form of the "
               "same local rule is the further step.")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 104, flush=True)
    print("  PROSPECTIVE MEMORY — PRODUCTION VERIFY (real ChatBrain + real brain_chat handler, rf recall, numpy-CPU)", flush=True)
    print("=" * 104, flush=True)
    print(f"  (A) FIRES-ON-CUE ok={a_ok}", flush=True)
    print(f"        formation ({form_ok}): {rows['A_fire_on_cue']['formation']['answer']}", flush=True)
    for r in inter_rows:
        print(f"        intervening {r['q']:24s} fired={r['fired']} held={r['held']} leak={r['reminder_leak']}", flush=True)
    print(f"        silent-before={inter_silent} held-alive={held_alive}", flush=True)
    print(f"        CUE fire ({fire_ok}): {rows['A_fire_on_cue']['cue']['answer']}", flush=True)
    print(f"  (B) SILENT-ON-WRONG-CUE ok={b_ok} (topic-silent={b1_ok}; spiking rel_A on slot-B cue="
          f"{rel_a_on_wrong_cue} < {FIRE_THR} -> {b2_ok})", flush=True)
    print(f"  (C) LESION ok={lesion_ok} (collapse={lesion_collapse}; held_after_lesion={held_after_lesion}; "
          f"intact-fires-vs-lesion-silent={intact_vs_lesion_ok})", flush=True)
    print(f"        lesioned cue turn: {rows['C_lesion']['cue']['answer']}", flush=True)
    print(f"  (D) BYTE-IDENTICAL-when-off ok={bi_ok}", flush=True)
    for r in bi_rows:
        print(f"        {r['q']:22s} identical={r['identical']} no_prospective_key={r['no_prospective_key']}", flush=True)
    print(f"        formation-phrased flag-off falls through (no key, no ack): {off_form_falls_through}", flush=True)
    print(f"  (E) LEARNED CUE->ACTION BINDING ok={e_ok}", flush=True)
    print(f"        E1 learned+fires ({e1_ok}): |w| before={norm_before} -> learned={learned_norm} "
          f"after={norm_after}; cue fired={pch.get('fired')}", flush=True)
    print(f"        E2 binding-lesion silent ({e2_ok}): {rows['E_learned_binding']['e2_binding_lesion_silent']['answer']}", flush=True)
    print(f"        E3 install escape fires ({e3_ok}): no-hebbian-keys={('hebbian' not in pif)} "
          f"cue fired={pic.get('fired')}", flush=True)
    verdict = "GO" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 104, flush=True)

    out = {"runner": "_prospective_memory_production_verify", "go": go, "status": decided["status"],
           "a_fire_on_cue_ok": a_ok, "b_wrong_cue_ok": b_ok, "lesion_ok": lesion_ok,
           "intact_vs_lesion_ok": intact_vs_lesion_ok, "byte_identical_ok": bi_ok,
           "learned_binding_ok": e_ok, "e1_learned_fires_ok": e1_ok,
           "e2_binding_lesion_silent_ok": e2_ok, "e3_install_escape_ok": e3_ok,
           "binding_norm_before": norm_before, "binding_learned_norm": learned_norm, "binding_norm_after": norm_after,
           "intact_fire_rel": intact_fire_rel, "lesion_fire_rel": lesion_fire_rel,
           "fire_attribution": fire_attribution, "rows": rows,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_prospective_memory/production_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
