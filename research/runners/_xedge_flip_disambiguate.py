"""TIGHT DIAGNOSTIC (2026-08-28): disambiguate the two candidate causes named in
`2026-08-28-onebrain-xedge-production-default-flip-NO-GO.md` for why the d6-WM->comprehension cross-edge's
content-change is invisible through the REAL `/api/brain-chat` handler even though it is load-bearing at the
organ level (PART 1/2/3 GO).

  (B) edge not loaded/driving in the handler path: the cross-edge weight the handler actually reads is still
      near its W0=0.05 baseline (not the converged ~16 magnitude), so nothing downstream could possibly shift.
  (A) edge drives the organ but not the rendered output: the weight IS converged and DOES shift the organ's own
      internal read (margin / wm_resolved), but the decision the FULL HANDLER renders (repair.role) doesn't move
      -- either because the wm_focus threaded from d6org.current_focus() never reaches the read, or because
      something downstream re-derives the answer.

Single seed (42), numpy backend (CPU, RAM-light), the SAME converged-edge config the flip-verify harness used for
its visible-on-real-traffic ARM (b_edge=learn, BRAIN_ONEBRAIN_XEDGE=1 + _LEARN=1, set_live_per_turn(False) -- the
build-curriculum edge at its converged magnitude, exactly `_xedge_flip_production_verify.py`'s B_on_learn config).

Prints three instrumented numbers/blocks:
  (i)   the ACTUAL loaded cross-edge weight in the handler path (pool.cross_weights, post-build).
  (ii)  the comprehension organ's own read (net_lean / content_role / wm_margin / wm_resolved / role), called
        DIRECTLY on the process-shared organ with an EXPLICIT wm_focus='w0' -- edge intact vs edge-lesioned.
  (iii) the repair_role actually returned through the REAL brain_chat handler, held vs no-held (mirrors the
        flip-verify harness's ARM-B visibility check) -- plus what `d6org.current_focus()` resolved to on the
        held session, to see whether the wm_focus VALUE that reaches the organ in the real handler matches (ii)'s
        explicit probe.

Run:
  SIM_BACKEND=numpy python -m research.runners._xedge_flip_disambiguate \
      --out research/findings/raw/_xedge_flip_disambiguate/seed42.json
"""
from __future__ import annotations

import argparse
import json
import os


AMB_ITEM = "the wolf watches the owl"
HOLD_TURN = "the fox and the wolf walked in"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")
    for k in ("BRAIN_ONEBRAIN_XEDGE", "BRAIN_ONEBRAIN_XEDGE_LEARN", "BRAIN_ONEBRAIN_XEDGE_LESION"):
        os.environ.pop(k, None)
    os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
    os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"

    out = {"seed": SEED, "backend": os.environ.get("SIM_BACKEND"), "config": "B_on_learn (converged, per_turn=False)"}

    # ── prime the process-global xedge pool EXACTLY as the flip-verify worker does, BEFORE importing the server ──
    import research.runners.onebrain_xedge_production as OX
    OX.set_live_per_turn(False)                 # b_edge=learn, converged build-curriculum (not per-turn)
    pool = OX.get_xedge_pool(SEED)
    assert pool is not None and pool.ok, "xedge pool failed to build"

    # ── (i) the ACTUAL loaded cross-edge weight in the handler path ──
    w_loaded = dict(pool.cross_weights)
    out["i_loaded_cross_weights"] = w_loaded
    w_max = max(w_loaded.values())
    out["i_verdict"] = ("CONVERGED (grown)" if w_max > 1.0 else "UNGROWN (~W0=0.05 baseline)")
    print(f"[i] loaded cross_weights = {w_loaded}  -> {out['i_verdict']}", flush=True)
    print(f"[i] pool.role = {pool.role}  learned={pool.learned} live_per_turn={pool.live_per_turn}", flush=True)

    # ── (ii) organ-level direct read: edge intact vs edge-lesioned, EXPLICIT wm_focus='w0' ──
    from research.runners.comprehension_production_organ import get_organ
    corg = get_organ(seed=SEED)
    corg.ensure_built()
    assert corg._shared is pool.pool, "comprehension organ is NOT sharing the primed xedge pool!"

    def probe(wm_focus, tag):
        r = corg.repair_target(AMB_ITEM, wm_focus=wm_focus)
        rec = {
            "tag": tag, "wm_focus": wm_focus, "role": (r or {}).get("role"),
            "content_role": (r or {}).get("content_role"), "net_lean": (r or {}).get("net_lean"),
            "a0": (r or {}).get("a0"), "a1": (r or {}).get("a1"),
            "wm_resolved": (r or {}).get("wm_resolved"), "wm_margin": (r or {}).get("wm_margin"),
            "raw": r,
        }
        print(f"[probe {tag}] wm_focus={wm_focus} -> role={rec['role']} content_role={rec['content_role']} "
              f"net_lean={rec['net_lean']} wm_resolved={rec['wm_resolved']} wm_margin={rec['wm_margin']}",
              flush=True)
        return rec

    ii_content_only = probe(None, "ii_content_only_no_focus")
    ii_held_intact = probe("w0", "ii_held_edge_intact")

    # lesion the cross-edge in place (zeroes the w{k}->sel synapses), then re-probe with the SAME explicit focus.
    pool.lesion_cross()
    ii_held_lesioned = probe("w0", "ii_held_edge_lesioned")

    out["ii_content_only_no_focus"] = ii_content_only
    out["ii_held_edge_intact"] = ii_held_intact
    out["ii_held_edge_lesioned"] = ii_held_lesioned
    out["ii_edge_shifts_organ_read"] = bool(
        ii_held_intact["net_lean"] != ii_content_only["net_lean"]
        or ii_held_intact["wm_resolved"] != ii_content_only["wm_resolved"]
        or ii_held_intact["role"] != ii_content_only["role"]
    )
    out["ii_lesion_reverts_organ_read"] = bool(
        ii_held_lesioned["wm_resolved"] is not True
        and ii_held_lesioned["role"] == ii_content_only["role"]
    )

    # ── (iii) the REAL /api/brain-chat handler: held vs no-held, seed 42 (matches the flip-verify shipped seed) ──
    # NOTE: the cross-edge is now LESIONED in-process from step (ii) above -- re-grow a FRESH primed pool for the
    # handler test so (iii) exercises the intact converged edge (a separate process would be cleaner, but the
    # in-process re-prime keeps this a single tight diagnostic; the module-global _POOL/_ORGAN are rebuilt below).
    import research.runners.onebrain_xedge_production as OX2
    import research.runners.comprehension_production_organ as CO2
    OX2._POOL = None
    CO2._ORGAN = None
    OX2.set_live_per_turn(False)
    pool2 = OX2.get_xedge_pool(SEED)
    assert pool2 is not None and pool2.ok, "re-primed xedge pool failed to build"
    out["iii_reprimed_cross_weights"] = dict(pool2.cross_weights)
    print(f"[iii] re-primed cross_weights = {pool2.cross_weights}", flush=True)

    from webapp.server import brain_chat, BrainChatRequest, _get_multiref_organ

    def turn(msg, session, reset=False):
        r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                        renderer="stub", rich=False, reset=reset))
        d = json.loads(r.body)
        rep = d.get("repair") or {}
        comp = d.get("comprehension") or {}
        return {
            "answer": d.get("answer"), "abstained": bool(d.get("abstained")),
            "comprehended": comp.get("comprehended"),
            "repair_role": rep.get("role"), "repair_content_role": rep.get("content_role"),
            "repair_net_lean": rep.get("net_lean"), "repair_wm_resolved": rep.get("wm_resolved"),
            "repair_wm_margin": rep.get("wm_margin"), "repaired": rep.get("repaired"),
            "multiref_n": (d.get("multiref") or {}).get("n_referents"),
        }

    # no-held (fresh session): the content-only baseline through the real handler.
    novisi = turn(AMB_ITEM, "diag_nv", reset=True)
    # held: HOLD_TURN sets this session's d6 focus, then read the SAME ambiguous item in-session.
    turn(HOLD_TURN, "diag_hd", reset=True)
    cache_key = ("diag_hd", "tiny-demo", "stub")
    d6org = _get_multiref_organ(cache_key)
    resolved_focus = d6org.current_focus()
    print(f"[iii] d6org.current_focus() after HOLD_TURN (session diag_hd) = {resolved_focus!r}", flush=True)
    held = turn(AMB_ITEM, "diag_hd", reset=False)

    out["iii_novisi"] = novisi
    out["iii_held"] = held
    out["iii_resolved_wm_focus_in_handler"] = resolved_focus
    out["iii_role_differs"] = bool(novisi["repair_role"] != held["repair_role"])
    out["iii_answer_differs"] = bool(novisi["answer"] != held["answer"])
    print(f"[iii] novisi repair_role={novisi['repair_role']!r}  held repair_role={held['repair_role']!r}  "
          f"role_differs={out['iii_role_differs']}  answer_differs={out['iii_answer_differs']}", flush=True)

    # ── verdict ──
    b_edge_not_loaded = (w_max <= 1.0)
    a_drives_organ_not_output = (
        (not b_edge_not_loaded) and out["ii_edge_shifts_organ_read"] and (not out["iii_role_differs"])
    )
    if b_edge_not_loaded:
        verdict = "B"
    elif a_drives_organ_not_output:
        verdict = "A"
    elif out["iii_role_differs"]:
        verdict = "NEITHER (visible through the handler in this diagnostic run -- re-check the flip-verify harness)"
    else:
        verdict = "AMBIGUOUS (edge loaded but the EXPLICIT organ probe (ii) also failed to shift -- see raw reads)"
    out["verdict"] = verdict
    print(f"\n===== VERDICT: {verdict} =====", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
