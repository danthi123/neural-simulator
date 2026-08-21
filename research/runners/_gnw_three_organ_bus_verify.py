"""PRODUCTION VERIFY for the GNW THREE-GENUINELY-DISTINCT-ORGANS consensus bus (`webapp/gnw_three_organ_bus.py`),
wired into `webapp/server.py::brain_chat` behind the NEW DEFAULT-OFF flag `BRAIN_GNW_3ORGAN`.

The 2-organ bus (`gnw_two_organ_bus.py`, DEFAULT-ON) commits a recall by the COINCIDENCE of organ A (spiking recall)
+ organ B (the spiking surprise/expectation-violation monitor). This bus adds a THIRD genuinely-distinct spiking
organ — organ C, the production COMPREHENSION monitor (`ComprehensionProductionOrgan`, the 6/6-GO D4 faculty, a
Wong-Wang `SpikingRoleCompetition` sel-pool WTA read off `cp_firing_states`). Organ C reads whether the RECALLED
PROPOSITION (agent, action, cand) is role-RESOLVABLE and CORROBORATES (votes) only when it comprehended it; on a LOW
comprehension margin (role-ambiguous) it WITHHOLDS. The consensus is Q=3 UNANIMITY (`norgan_hop`, d_sub calibrated so
2*d_sub < the ignition knee <= 3*d_sub): the workspace ignites — the brain commits — ONLY when it RECALLS ∧ is NOT
surprised ∧ COMPREHENDED. Any organ withholding leaves slot(cand) subthreshold -> ABSTAIN.

Proven here on the REAL production tiny-demo ChatBrain (numpy-CPU, `BRAIN_COMPOSER_KIND=rf`), SYNCHRONOUS/foreground:

  (A) OFF (`BRAIN_GNW_3ORGAN` unset) -> BYTE-IDENTICAL to the current bus (the DEFAULT-ON 2-organ bus) on every query
      — install is a no-op AND a runtime flag-flip-off makes the wrapper delegate to the 2-organ gate, turn for turn,
      across covered + out-of-scope (self / open-ended) classes.
  (B) ON, LOAD-BEARING. On a HIGH-comprehension / out-of-competence stored query the 3-organ decision == the 2-organ
      decision (organ C corroborates or defers -> no behaviour change). On a LOW-comprehension probe (a role-ambiguous
      stored proposition — two-animate symmetric verb / verbfit conflict — where recall ∧ ¬surprise WOULD commit) the
      3-organ bus ABSTAINS (organ C withholds), a decision the 2-organ bus CANNOT make. Per-organ votes reported.
  (C) LESION (`BRAIN_GNW_3ORGAN_ORGANC_LESION=1`) -> organ C corroborates unconditionally -> the consensus collapses
      to the 2-organ decision -> the low-comprehension abstain REVERTS to the 2-organ commit (the veto is attributed
      to organ C's spiking participation, not a host `if margin < x`).
  (D) MOAT preserved: no unstored / inconsistent query is turned into an assertion on ANY arm.

GO = OFF byte-identical AND organ C adds a genuine load-bearing veto AND the lesion severs it AND the moat holds.

Run (numpy-CPU, fast rf recall path; ~5-10 min, SYNCHRONOUS):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_three_organ_bus_verify
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
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path; ~s not ~180s)

from tools.verdict import Verdict          # noqa: E402
from tools.lab import attributable_to      # noqa: E402


def _svo(x):
    return list(x) if x is not None else None


def _svo_eq(x, y) -> bool:
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


# ── the panel (through the REAL production ChatBrain) ─────────────────────────────────────────────────────────────
# Covered recall queries. `tag` records WHY organ C votes/withholds. `want2` is the 2-organ (current-bus) decision.
#   high_comp     : a well-formed transitive (animate agent + asymmetric verb + inanimate patient) -> margin HIGH ->
#                   organ C VOTES -> 3-organ == 2-organ (commit). (taught this turn: "dog eat apple")
#   deferred      : real-but-untabled words the brain knows (out of the D4 cue lexicon) -> organ C DEFERS -> commit.
#   ambig_2animate: two animate nouns + a symmetric verb ("dog chase cat") -> roles genuinely un-resolvable from
#                   content -> margin LOW -> organ C WITHHOLDS -> 3-organ ABSTAINS while 2-organ commits.
#   verbfit_confl : an asymmetric verb whose patient is animate in the toy lexicon ("cat eat fish") -> verbfit fights
#                   animacy -> margin LOW -> organ C WITHHOLDS. (A declared cue-lexicon-ceiling case.)
STORED_UNCHANGED = [
    ("what does dog eat?",    "dog",   "eat",   ["dog", "eat", "apple"],    "high_comp"),
    ("what does brain use?",  "brain", "use",   ["brain", "use", "spikes"], "deferred"),
    ("what does brain learn?", "brain", "learn", ["brain", "learn", "words"], "deferred"),
    ("what does brain store?", "brain", "store", ["brain", "store", "memory"], "deferred"),
]
LOW_COMP_PROBES = [
    ("what does dog chase?", "dog", "chase", ["dog", "chase", "cat"],  "ambig_2animate"),
    ("what does cat eat?",   "cat", "eat",   ["cat", "eat", "fish"],   "verbfit_confl"),
]
MOAT = [
    ("what does fish fly?",  "fish", "fly"),    # unstored agent/action -> organ A misses -> abstain
    ("what does cat chase?", "cat",  "chase"),  # cat has an EAT fact, not CHASE -> the substrate has no binding
]
# out-of-scope classes exercised only in the OFF byte-identical gate panel (organ C never touches them).
OUT_OF_SCOPE = ["what are you", "what might a dog do"]


def _production_chat():
    """The REAL production ChatBrain with the existing N-organ + 2-organ buses installed (exactly as `brain_chat`
    does at HEAD default), then TEACH the one well-formed high-comprehension fact the panel needs."""
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    from webapp import gnw_two_organ_bus as g2
    chat, _src = _build_chat_brain("tiny-demo", "stub")
    gbs.install_bus_gate(chat)                         # HEAD: the N-organ (composer-only) bus
    os.environ["BRAIN_GNW_2ORGAN"] = "1"
    g2.install_two_organ_gate(chat)                    # HEAD default-on: the 2-organ coincidence authors the combine
    # teach a well-formed transitive so organ C has a HIGH-comprehension covered query that it ACTIVELY corroborates
    # (dog=animate, eat=asymmetric, apple=inanimate -> the roles decisively separate -> margin >> threshold).
    chat.gate("dog eat apple")
    return chat, g2


def run():
    from webapp import gnw_three_organ_bus as g3

    chat, g2 = _production_chat()
    from sim.backend import get_backend
    backend = get_backend()[1]

    chat_built_ok = bool(hasattr(chat, "inner") and hasattr(chat.inner, "composer")
                         and hasattr(chat, "stored_facts")
                         and ("dog", "eat", "apple") in [tuple(f) for f in getattr(chat, "stored_facts", [])])

    # ── (A) phase 1 — flag OFF: install is a NO-OP; the gate output IS the 2-organ-bus (current) output. ───────────
    os.environ.pop("BRAIN_GNW_3ORGAN", None)
    installed_off = g3.install_three_organ_gate(chat)                       # expect False (no-op)
    off_noop = (installed_off is False) and (not getattr(chat, "_three_organ_installed", False))
    # the CURRENT bus (2-organ) gate decisions over the whole panel (covered + moat + out-of-scope).
    off_panel = [q for (q, *_r) in STORED_UNCHANGED] + [q for (q, *_r) in LOW_COMP_PROBES] \
        + [q for (q, *_r) in MOAT] + OUT_OF_SCOPE
    head_out = [_svo(chat.gate(q)) for q in off_panel]                     # HEAD == the 2-organ bus decision

    # ── (A) phase 2 — install the 3-organ wrapper, runtime-flip OFF -> delegates to the 2-organ gate == HEAD. ──────
    os.environ["BRAIN_GNW_3ORGAN"] = "1"
    installed_on = g3.install_three_organ_gate(chat)                       # wraps chat.gate (expect True)
    os.environ.pop("BRAIN_GNW_3ORGAN", None)                               # runtime flip OFF
    off_runtime_out = [_svo(chat.gate(q)) for q in off_panel]
    off_runtime_matches_head = (off_runtime_out == head_out)

    # ── combine-level teeth (read-only, deterministic) — the 2-organ vs 3-organ vs 3-organ-lesion decisions. ───────
    def dec2(a, v):
        return g2.two_organ_combine(chat, a, v).get("committed")

    def dec3(a, v, organc_lesion=False):
        return g3.three_organ_combine(chat, a, v, organc_lesion=organc_lesion)

    # (B) STORED_UNCHANGED: 3-organ ON commits the SAME patient the 2-organ bus commits (organ C votes / defers).
    unchanged_rows, unchanged_ok = [], True
    for (q, a, v, want, tag) in STORED_UNCHANGED:
        c2 = dec2(a, v)
        i3 = dec3(a, v)
        c3 = i3.get("committed")
        same = _svo_eq([a, v, c2] if c2 is not None else None,
                       [a, v, c3] if c3 is not None else None) and (c2 is not None) and (c3 is not None)
        unchanged_ok = unchanged_ok and same
        unchanged_rows.append({"q": q, "tag": tag, "two_organ": c2, "three_organ": c3, "unchanged": same,
                               "organ_c_votes": i3.get("organ_c_votes"), "organ_c_margin": i3.get("organ_c_margin"),
                               "organ_c_threshold": i3.get("organ_c_threshold"),
                               "organ_c_deferred": i3.get("organ_c_deferred"),
                               "organ_c_comprehended": i3.get("organ_c_comprehended"),
                               "n_votes": i3.get("n_votes"), "n_ignited": i3.get("n_ignited")})

    # (B)+(C) LOW_COMP_PROBES: 2-organ COMMITS, 3-organ ON ABSTAINS (organ C veto), 3-organ LESION REVERTS to commit.
    probe_rows, probe_veto_ok, probe_revert_ok = [], True, True
    for (q, a, v, want, tag) in LOW_COMP_PROBES:
        c2 = dec2(a, v)
        i3 = dec3(a, v, organc_lesion=False)
        c3 = i3.get("committed")
        i3l = dec3(a, v, organc_lesion=True)
        c3l = i3l.get("committed")
        veto = (c2 is not None) and (c3 is None) and (i3.get("organ_c_votes") is False)
        revert = _svo_eq([a, v, c2] if c2 is not None else None,
                         [a, v, c3l] if c3l is not None else None) and (c3l is not None)
        probe_veto_ok = probe_veto_ok and veto
        probe_revert_ok = probe_revert_ok and revert
        probe_rows.append({"q": q, "tag": tag, "two_organ": c2, "three_organ_on": c3, "three_organ_lesion": c3l,
                           "veto": veto, "reverts_under_lesion": revert,
                           "organ_c_margin": i3.get("organ_c_margin"), "organ_c_threshold": i3.get("organ_c_threshold"),
                           "organ_c_comprehended": i3.get("organ_c_comprehended"),
                           "organ_b_confirmed": i3.get("organ_b_confirmed"),
                           "n_votes_on": i3.get("n_votes"), "n_ignited_on": i3.get("n_ignited"),
                           "abstain_reason": i3.get("abstain_reason"),
                           "n_votes_lesion": i3l.get("n_votes"), "n_ignited_lesion": i3l.get("n_ignited")})

    # (D) MOAT: no unstored/inconsistent query is committed on EITHER the 2-organ or the 3-organ arm.
    moat_rows, moat_ok = [], True
    for (q, a, v) in MOAT:
        c2 = dec2(a, v)
        i3 = dec3(a, v)
        c3 = i3.get("committed")
        ok = (c2 is None) and (c3 is None)
        moat_ok = moat_ok and ok
        moat_rows.append({"q": q, "two_organ": c2, "three_organ": c3, "both_abstain": ok,
                          "organ_a_recall": i3.get("organ_a_recall"), "reason": i3.get("abstain_reason")})

    # ── instrument precondition: the Q=3 UNANIMITY window on the SHARED production workspace bridge. The consensus
    #    claim needs (N-1)*d_sub < the ignition knee <= N*d_sub (2 votes subthreshold, 3 votes suprathreshold). ─────
    from research.runners._p1_2_workspace_deliberation_loop_derisk import _ignite_and_read
    from research.runners._gnw_norgan_bus_derisk import THR as _NORGAN_THR
    b_i, xp_i, slots_i, snap_i = g2._get_bridge(42, False)
    d_sub = float(g3._D_SUB_3)
    rate_2 = float(_ignite_and_read(b_i, xp_i, slots_i, snap_i, [2 * d_sub] + [0.0] * (len(slots_i) - 1))[0])
    rate_3 = float(_ignite_and_read(b_i, xp_i, slots_i, snap_i, [3 * d_sub] + [0.0] * (len(slots_i) - 1))[0])
    unanimity_window_ok = bool(rate_2 < _NORGAN_THR <= rate_3)

    # ── organ C read-provenance: the D4 margin is a cp_firing_states sel-pool read (the host _semantic_contrast is
    #    never called for it — the 6/6-GO D4 de-risk asserts read_from_firing_states / host_semantic_contrast_used
    #    False). Record the organ's own claim + a distinct HIGH vs LOW margin so the veto is a graded spiking read. ──
    comp_margins = ([r["organ_c_margin"] for r in unchanged_rows if r.get("organ_c_margin") is not None]
                    + [r["organ_c_margin"] for r in probe_rows if r.get("organ_c_margin") is not None])
    high_margins = [r["organ_c_margin"] for r in unchanged_rows
                    if r.get("organ_c_margin") is not None and r.get("organ_c_votes")]
    low_margins = [r["organ_c_margin"] for r in probe_rows if r.get("organ_c_margin") is not None]
    thr = next((r["organ_c_threshold"] for r in (unchanged_rows + probe_rows)
                if r.get("organ_c_threshold") is not None), None)
    margin_separates = bool(high_margins and low_margins and thr is not None
                            and min(high_margins) >= thr > max(low_margins))

    # ── Verdict: preconditions (UNDEFINED, not NO-GO, on failure) + the plain-boolean teeth that drive go=. ────────
    vd = Verdict(f"GNW three-distinct-organs consensus bus — production verify (real ChatBrain.gate, rf/{backend})")
    vd.disabled("full per-turn organ-stepping brain_chat handler",
                why="tested at the ChatBrain.gate/combine level where the 3-organ wiring lives, not the heavy "
                    "numpy-CPU per-turn handler; byte-identical-when-off is guaranteed by construction there (the "
                    "server hook is gated by the SAME BRAIN_GNW_3ORGAN flag, unset by default -> never imported)")
    vd.require("backend-recognized", backend in ("numpy", "cupy"), expect=True)
    vd.require("production-chat-built", chat_built_ok, expect=True,
               note="the real production ChatBrain built + the well-formed high-comprehension fact taught")
    vd.require("install-off-is-noop", off_noop, expect=True,
               note="flag OFF -> install_three_organ_gate is a no-op (the flag genuinely gates installation)")
    vd.require("install-on-installs", installed_on, expect=True,
               note="flag ON -> install_three_organ_gate wraps chat.gate on the real production chat")
    vd.require("q3-unanimity-window", unanimity_window_ok, expect=True,
               note=f"on the shared workspace bridge 2*d_sub={2*d_sub:.0f}->rate={rate_2:.3f} < THR={_NORGAN_THR:.3f} "
                    f"<= 3*d_sub={3*d_sub:.0f}->rate={rate_3:.3f} (the consensus-veto needs exactly this window)")
    vd.require("organ-c-margin-separates-high-from-low", margin_separates, expect=True,
               note=f"organ C's cp_firing_states sel-pool margin: min HIGH={min(high_margins) if high_margins else None} "
                    f">= thr={thr} > max LOW={max(low_margins) if low_margins else None} (a graded spiking read, not "
                    f"a binary host flag)")

    # TEETH (plain booleans -> a genuine failure reads NO-GO, not UNDEFINED).
    go = bool(off_noop and off_runtime_matches_head and unchanged_ok and probe_veto_ok and probe_revert_ok and moat_ok)
    decided = vd.decide(go=go)

    # attribution: the low-comprehension abstain is owed to organ C's spiking veto (intact) vs its lesion (silenced).
    intact_abstain_rate = (sum(1 for r in probe_rows if r["three_organ_on"] is None) / len(probe_rows)) if probe_rows else 0.0
    lesion_abstain_rate = (sum(1 for r in probe_rows if r["three_organ_lesion"] is None) / len(probe_rows)) if probe_rows else 0.0
    attr_organc = attributable_to("low-comprehension abstain: intact organ C veto vs organ-C-lesion",
                                  intact_abstain_rate, lesion_abstain_rate)

    print("\n" + "=" * 110, flush=True)
    print(f"  GNW THREE-DISTINCT-ORGANS CONSENSUS BUS — PRODUCTION VERIFY (real ChatBrain.gate, rf/{backend})", flush=True)
    print("=" * 110, flush=True)
    print(f"  (A) BYTE-IDENTICAL-WHEN-OFF: install no-op={off_noop} | runtime-flip-off == 2-organ bus="
          f"{off_runtime_matches_head}", flush=True)
    for q, h, o in zip(off_panel, head_out, off_runtime_out):
        print(f"      {q:26s} 2ORGAN={h} 3ORGAN_OFF={o} match={h == o}", flush=True)
    print(f"  (B) STORED UNCHANGED (organ C votes / defers -> 3-organ == 2-organ): {unchanged_ok}", flush=True)
    for r in unchanged_rows:
        print(f"      {r['q']:24s} [{r['tag']:9s}] 2organ={r['two_organ']!r} 3organ={r['three_organ']!r} "
              f"C_votes={r['organ_c_votes']} margin={r['organ_c_margin']} thr={r['organ_c_threshold']} "
              f"deferred={r['organ_c_deferred']} n_votes={r['n_votes']} n_ign={r['n_ignited']}", flush=True)
    print(f"  (B/C) LOW-COMPREHENSION VETO: veto_ok={probe_veto_ok} reverts_under_lesion={probe_revert_ok}", flush=True)
    for r in probe_rows:
        print(f"      {r['q']:24s} [{r['tag']:13s}] 2organ={r['two_organ']!r} -> 3organ_ON={r['three_organ_on']!r} "
              f"(margin={r['organ_c_margin']} < thr={r['organ_c_threshold']}, b_confirm={r['organ_b_confirmed']}, "
              f"n_votes={r['n_votes_on']}, reason={r['abstain_reason']}) -> 3organ_LESION={r['three_organ_lesion']!r} "
              f"(n_votes={r['n_votes_lesion']})", flush=True)
    print(f"  (D) MOAT (no unstored/inconsistent committed on either arm): {moat_ok}", flush=True)
    for r in moat_rows:
        print(f"      {r['q']:24s} 2organ={r['two_organ']!r} 3organ={r['three_organ']!r} "
              f"a_recall={r['organ_a_recall']!r} reason={r['reason']}", flush=True)
    print(f"  ATTRIBUTABLE: low-comprehension abstain owed to organ C veto (intact vs lesion) = {attr_organc}", flush=True)
    print(f"\n  VERDICT: {decided['status']}\n" + "=" * 110, flush=True)

    out = {"runner": "_gnw_three_organ_bus_verify", "go": decided["go"], "status": decided["status"],
           "verdict": decided, "backend": backend,
           "byte_identical_when_off": {"install_noop_when_off": off_noop,
                                       "runtime_flip_off_matches_two_organ": off_runtime_matches_head,
                                       "two_organ_out": head_out, "three_organ_off_out": off_runtime_out,
                                       "panel": off_panel},
           "stored_unchanged": {"ok": unchanged_ok, "rows": unchanged_rows},
           "low_comprehension_probes": {"veto_ok": probe_veto_ok, "reverts_under_lesion_ok": probe_revert_ok,
                                        "rows": probe_rows},
           "moat": {"ok": moat_ok, "rows": moat_rows},
           "q3_unanimity_window": {"ok": unanimity_window_ok, "d_sub": d_sub, "rate_2votes": rate_2,
                                   "rate_3votes": rate_3, "THR": float(_NORGAN_THR)},
           "organ_c_margin_separates": {"ok": margin_separates, "high_margins": high_margins,
                                        "low_margins": low_margins, "threshold": thr},
           "attributable": {"low_comp_abstain_intact_vs_organc_lesion": attr_organc,
                            "intact_abstain_rate": intact_abstain_rate, "lesion_abstain_rate": lesion_abstain_rate},
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = f"research/findings/raw/_gnw_three_organ/production_verify_{backend}.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if decided["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(run())
