"""PRODUCTION VERIFY for the GNW THREE-GENUINELY-DISTINCT-ORGANS consensus bus (`webapp/gnw_three_organ_bus.py`),
wired into `webapp/server.py::brain_chat` behind `BRAIN_GNW_3ORGAN` (DEFAULT-ON since 2026-08-21; `BRAIN_GNW_3ORGAN=0`
is the byte-identical escape). NOTE: this runner sets `="0"` explicitly for its OFF arms — `unset` is now ON.

The 2-organ bus (`gnw_two_organ_bus.py`, DEFAULT-ON) commits a recall by the COINCIDENCE of organ A (spiking recall)
+ organ B (the spiking surprise/expectation-violation monitor). This bus adds a THIRD genuinely-distinct organ —
organ C, the COMPREHENSION monitor of the RECALLED PROPOSITION (agent, action, cand).

2026-08-21 D4 REAL-VOCAB FIX (this runner re-verifies it). Organ C's veto AUTHORITY is now a REAL-VOCAB entity/role
COMPETENCE read over the recalled proposition, NOT the toy animacy/verbfit cue-competition margin. WHY: a RECALLED
fact's thematic roles are already RESOLVED by its stored engram (the brain knows dog is the agent / cat the patient
BECAUSE it stored that fact), so comprehension of a RECALL is "are all its entities/roles KNOWN in the brain's own
learned vocab", NOT "can bottom-up cues separate the roles" (the toy-competition question — right for a NOVEL
assertion, but which false-vetoed the LEGITIMATELY-recalled two-animate `dog chase cat` and verbfit-conflict
`cat eat fish`, the regression that HELD the flip). A content entity/role OUTSIDE the learned vocab (genuine
non-comprehension) still routes to the spiking D4 sel-pool WTA as its correct "does this UNKNOWN proposition's roles
resolve?" instrument, so the spiking read stays load-bearing exactly where it is the right tool.

Proven here on the REAL production tiny-demo ChatBrain (numpy-CPU, `BRAIN_COMPOSER_KIND=rf`), SYNCHRONOUS/foreground:

  (A) OFF (`BRAIN_GNW_3ORGAN=0`) -> BYTE-IDENTICAL to the current bus (the DEFAULT-ON 2-organ bus) on every query
      — install is a no-op AND a runtime flag-flip-off makes the wrapper delegate to the 2-organ gate.
  (B) NO REGRESSION (the FIX). Every LEGITIMATELY-recalled fact — including the two the toy-margin veto used to
      wrongly abstain (`dog chase cat`, `cat eat fish`) — now COMMITS on the 3-organ arm EXACTLY as the 2-organ bus
      does (organ C reads them real-vocab-KNOWN -> corroborates). This is the regression fix.
  (C) GENUINE NON-COMPREHENSION VETO, load-bearing + lesion-severable. A proposition carrying an entity/role OUTSIDE
      the brain's learned vocab (a `wizard` the brain never learned) reads real-vocab-UNKNOWN -> organ C WITHHOLDS
      (votes False); under `organc_lesion` it corroborates (votes True). By the proven Q=3 UNANIMITY window (2 votes
      subthreshold, 3 votes supra) a withheld organ C -> ABSTAIN, a corroborating one -> commit — so the veto is a
      real consensus-veto the 2-organ bus cannot make, attributable to organ C's participation. A real-vocab-KNOWN
      proposition votes True (the discrimination).
  (D) MOAT preserved: no unstored / inconsistent query is turned into an assertion on ANY arm.

GO = OFF byte-identical AND no legitimate recall regresses AND organ C adds a genuine load-bearing veto AND the
lesion severs it AND the moat holds.

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
# NO-REGRESSION recall queries. `want` is the taught patient; the 3-organ bus MUST commit it exactly as the 2-organ
# bus does (organ C reads the proposition real-vocab-KNOWN -> corroborates). The last two are the fix's headline:
# `dog chase cat` (two-animate + symmetric verb) and `cat eat fish` (toy lexicon marks fish animate) — both were
# WRONGLY vetoed by the old toy-margin monitor, and must now COMMIT.
NO_REGRESSION = [
    ("what does dog eat?",     "dog",   "eat",   ["dog", "eat", "apple"],     "high_comp"),
    ("what does brain use?",   "brain", "use",   ["brain", "use", "spikes"],  "untabled_known"),
    ("what does brain learn?", "brain", "learn", ["brain", "learn", "words"], "untabled_known"),
    ("what does brain store?", "brain", "store", ["brain", "store", "memory"], "untabled_known"),
    ("what does dog chase?",   "dog",   "chase", ["dog", "chase", "cat"],     "was_falsely_vetoed_2animate"),
    ("what does cat eat?",     "cat",   "eat",   ["cat", "eat", "fish"],      "was_falsely_vetoed_verbfit"),
]
# GENUINE NON-COMPREHENSION probes (organ-C vote level). The proposition carries an entity the brain never learned
# (`wizard` / `dragon` -> real-vocab-UNKNOWN) -> organ C WITHHOLDS; under organc_lesion it corroborates. A known
# proposition (drawn from the recalled facts) is the positive control (organ C votes True).
NONCOMP_VETO = [
    ("wizard", "chase", "cat",   "oov_agent"),
    ("dragon", "eat",   "apple", "oov_agent"),
]
KNOWN_CONTROL = [
    ("dog", "chase", "cat",   "known_2animate"),
    ("cat", "eat",   "fish",  "known_verbfit"),
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
    # teach a well-formed transitive so organ C has a HIGH-comprehension covered query it ACTIVELY corroborates.
    chat.gate("dog eat apple")
    return chat, g2


def run():
    from webapp import gnw_three_organ_bus as g3

    chat, g2 = _production_chat()
    from sim.backend import get_backend
    backend = get_backend()[1]

    # the brain's learned vocab (the same real vocab organ C's competence read uses).
    _a, _v, all_concepts, _e_b, _sp = g2._chat_concepts(chat)
    brain_vocab = set(all_concepts)

    chat_built_ok = bool(hasattr(chat, "inner") and hasattr(chat.inner, "composer")
                         and hasattr(chat, "stored_facts")
                         and ("dog", "eat", "apple") in [tuple(f) for f in getattr(chat, "stored_facts", [])])

    # ── (A) phase 1 — flag OFF: install is a NO-OP; the gate output IS the 2-organ-bus (current) output. ───────────
    os.environ["BRAIN_GNW_3ORGAN"] = "0"   # explicit OFF (the default is now ON after the 2026-08-27 flip; unset != OFF)
    installed_off = g3.install_three_organ_gate(chat)                       # expect False (no-op)
    off_noop = (installed_off is False) and (not getattr(chat, "_three_organ_installed", False))
    # Byte-identity can only be tested on DETERMINISTIC outputs. "what might a dog do" is a stochastic generative
    # hypothesis: the generative-DRAW WTA draws a fresh sample each call, so its output drifts with RNG position
    # between the two panel evaluations below REGARDLESS of the 3-organ code (the runtime-off wrapper provably
    # delegates to orig_gate) -> it is a category error to include it in a byte-identity check. Exclude it here;
    # its delegation is covered structurally (install_three_organ_gate returns orig_gate's result when off).
    _DETERMINISTIC_OOS = [q for q in OUT_OF_SCOPE if q != "what might a dog do"]
    off_panel = [q for (q, *_r) in NO_REGRESSION] + [q for (q, *_r) in MOAT] + _DETERMINISTIC_OOS
    head_out = [_svo(chat.gate(q)) for q in off_panel]                     # HEAD == the 2-organ bus decision

    # ── (A) phase 2 — install the 3-organ wrapper, runtime-flip OFF -> delegates to the 2-organ gate == HEAD. ──────
    os.environ["BRAIN_GNW_3ORGAN"] = "1"
    installed_on = g3.install_three_organ_gate(chat)                       # wraps chat.gate (expect True)
    os.environ["BRAIN_GNW_3ORGAN"] = "0"                                   # runtime flip OFF (explicit; unset now = ON)
    off_runtime_out = [_svo(chat.gate(q)) for q in off_panel]
    off_runtime_matches_head = (off_runtime_out == head_out)

    # ── combine-level teeth (read-only, deterministic) — the 2-organ vs 3-organ decisions. ────────────────────────
    def dec2(a, v):
        return g2.two_organ_combine(chat, a, v).get("committed")

    def dec3(a, v, organc_lesion=False):
        return g3.three_organ_combine(chat, a, v, organc_lesion=organc_lesion)

    # (B) NO REGRESSION: every legitimately-recalled fact commits on the 3-organ arm EXACTLY as the 2-organ bus does
    # (organ C reads it real-vocab-KNOWN -> corroborates). Includes the two the old toy-margin veto wrongly abstained.
    noreg_rows, noreg_ok = [], True
    for (q, a, v, want, tag) in NO_REGRESSION:
        c2 = dec2(a, v)
        i3 = dec3(a, v)
        c3 = i3.get("committed")
        same = _svo_eq([a, v, c2] if c2 is not None else None,
                       [a, v, c3] if c3 is not None else None) and (c2 is not None) and (c3 is not None)
        noreg_ok = noreg_ok and same
        noreg_rows.append({"q": q, "tag": tag, "two_organ": c2, "three_organ": c3, "commits_unchanged": same,
                           "organ_c_votes": i3.get("organ_c_votes"),
                           "organ_c_real_vocab_known": i3.get("organ_c_real_vocab_known"),
                           "organ_c_unknown_tokens": i3.get("organ_c_unknown_tokens"),
                           "n_votes": i3.get("n_votes"), "n_ignited": i3.get("n_ignited")})

    # (C) GENUINE NON-COMPREHENSION VETO (organ-C vote level, through the real _comprehension_vote):
    #   * an OOV-entity proposition -> organ C WITHHOLDS (votes False, real_vocab_known False); organc_lesion -> True.
    #   * a real-vocab-KNOWN proposition -> organ C VOTES True (the discrimination).
    def vote(a, v, p, lesion=False):
        return g3._comprehension_vote(a, v, p, brain_vocab, seed=42, lesion=lesion)

    veto_rows, veto_ok, revert_ok = [], True, True
    for (a, v, p, tag) in NONCOMP_VETO:
        c = vote(a, v, p, lesion=False)
        cl = vote(a, v, p, lesion=True)
        vetoed = (c.get("votes") is False) and (c.get("organ_c_real_vocab_known") is False)
        reverts = (cl.get("votes") is True)
        veto_ok = veto_ok and vetoed
        revert_ok = revert_ok and reverts
        veto_rows.append({"prop": [a, v, p], "tag": tag, "votes": c.get("votes"),
                          "real_vocab_known": c.get("organ_c_real_vocab_known"),
                          "unknown_tokens": c.get("organ_c_unknown_tokens"),
                          "competent": c.get("organ_c_competent"), "margin": c.get("organ_c_margin"),
                          "vetoed": vetoed, "votes_under_lesion": cl.get("votes"), "reverts_under_lesion": reverts})

    known_rows, known_ok = [], True
    for (a, v, p, tag) in KNOWN_CONTROL:
        c = vote(a, v, p, lesion=False)
        ok = (c.get("votes") is True) and (c.get("organ_c_real_vocab_known") is True)
        known_ok = known_ok and ok
        known_rows.append({"prop": [a, v, p], "tag": tag, "votes": c.get("votes"),
                           "real_vocab_known": c.get("organ_c_real_vocab_known"), "corroborates": ok})
    # the discrimination: KNOWN corroborates AND UNKNOWN vetoes (real-vocab separates them cleanly).
    real_vocab_discriminates = bool(known_ok and veto_ok)

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
    #    claim needs (N-1)*d_sub < the ignition knee <= N*d_sub (2 votes subthreshold, 3 votes suprathreshold) — so a
    #    withheld organ C (2 votes) -> ABSTAIN and a corroborating one (3 votes) -> commit. ─────────────────────────
    from research.runners._p1_2_workspace_deliberation_loop_derisk import _ignite_and_read
    from research.runners._gnw_norgan_bus_derisk import THR as _NORGAN_THR
    b_i, xp_i, slots_i, snap_i = g2._get_bridge(42, False)
    d_sub = float(g3._D_SUB_3)
    rate_2 = float(_ignite_and_read(b_i, xp_i, slots_i, snap_i, [2 * d_sub] + [0.0] * (len(slots_i) - 1))[0])
    rate_3 = float(_ignite_and_read(b_i, xp_i, slots_i, snap_i, [3 * d_sub] + [0.0] * (len(slots_i) - 1))[0])
    unanimity_window_ok = bool(rate_2 < _NORGAN_THR <= rate_3)

    # ── Verdict: preconditions (UNDEFINED, not NO-GO, on failure) + the plain-boolean teeth that drive go=. ────────
    vd = Verdict(f"GNW three-distinct-organs consensus bus — production verify, D4 real-vocab fix (rf/{backend})")
    vd.disabled("full per-turn organ-stepping brain_chat handler",
                why="tested at the ChatBrain.gate/combine level where the 3-organ wiring lives, not the heavy "
                    "numpy-CPU per-turn handler; byte-identical-when-off is guaranteed by construction there (the "
                    "server hook is gated by the SAME BRAIN_GNW_3ORGAN flag; =0 to skip -> never imported)")
    vd.disabled("a full-combine (three_organ_combine) OOV-recall veto",
                why="the recall composer (both rf and onebrain) is EXACT-MATCH: organ A returns a cand ONLY for a "
                    "stored (agent, action), so on this path a recalled proposition's entities are always in-vocab "
                    "and the moat independently abstains on an unknown (agent, action). Organ C's genuine "
                    "non-comprehension veto is therefore exercised at the _comprehension_vote level (the real vote "
                    "fn the combine calls) + mapped to abstain via the proven Q=3 window; it becomes a live "
                    "full-combine guard only when a noisier composer path returns an ungrounded recalled token")
    vd.require("backend-recognized", backend in ("numpy", "cupy"), expect=True)
    vd.require("production-chat-built", chat_built_ok, expect=True,
               note="the real production ChatBrain built + the well-formed high-comprehension fact taught")
    vd.require("install-off-is-noop", off_noop, expect=True,
               note="flag OFF -> install_three_organ_gate is a no-op (the flag genuinely gates installation)")
    vd.require("install-on-installs", installed_on, expect=True,
               note="flag ON -> install_three_organ_gate wraps chat.gate on the real production chat")
    vd.require("q3-unanimity-window", unanimity_window_ok, expect=True,
               note=f"on the shared workspace bridge 2*d_sub={2*d_sub:.0f}->rate={rate_2:.3f} < THR={_NORGAN_THR:.3f} "
                    f"<= 3*d_sub={3*d_sub:.0f}->rate={rate_3:.3f} (a withheld organ C -> abstain; corroborating -> commit)")
    vd.require("real-vocab-discriminates-known-from-unknown", real_vocab_discriminates, expect=True,
               note="organ C corroborates every real-vocab-KNOWN proposition (incl. dog-chase-cat / cat-eat-fish) "
                    "and vetoes an OOV-entity proposition (wizard/dragon) — the veto fires ONLY on genuine "
                    "non-comprehension, not on a role-ambiguous surface form")

    # TEETH (plain booleans -> a genuine failure reads NO-GO, not UNDEFINED).
    go = bool(off_noop and off_runtime_matches_head and noreg_ok and veto_ok and revert_ok and known_ok and moat_ok)
    decided = vd.decide(go=go)

    # attribution: the non-comprehension veto is owed to organ C's participation (intact veto vs its lesion).
    intact_veto_rate = (sum(1 for r in veto_rows if r["votes"] is False) / len(veto_rows)) if veto_rows else 0.0
    lesion_veto_rate = (sum(1 for r in veto_rows if r["votes_under_lesion"] is False) / len(veto_rows)) if veto_rows else 0.0
    attr_organc = attributable_to("non-comprehension veto: intact organ C vs organ-C-lesion",
                                  intact_veto_rate, lesion_veto_rate)

    print("\n" + "=" * 110, flush=True)
    print(f"  GNW THREE-DISTINCT-ORGANS CONSENSUS BUS — PRODUCTION VERIFY (D4 REAL-VOCAB FIX, rf/{backend})", flush=True)
    print("=" * 110, flush=True)
    print(f"  (A) BYTE-IDENTICAL-WHEN-OFF: install no-op={off_noop} | runtime-flip-off == 2-organ bus="
          f"{off_runtime_matches_head}", flush=True)
    for q, h, o in zip(off_panel, head_out, off_runtime_out):
        print(f"      {q:26s} 2ORGAN={h} 3ORGAN_OFF={o} match={h == o}", flush=True)
    print(f"  (B) NO REGRESSION (every legitimate recall commits, incl. the 2 the old toy veto abstained): {noreg_ok}",
          flush=True)
    for r in noreg_rows:
        print(f"      {r['q']:24s} [{r['tag']:27s}] 2organ={r['two_organ']!r} 3organ={r['three_organ']!r} "
              f"C_votes={r['organ_c_votes']} real_vocab_known={r['organ_c_real_vocab_known']} "
              f"n_votes={r['n_votes']} n_ign={r['n_ignited']}", flush=True)
    print(f"  (C) GENUINE NON-COMPREHENSION VETO: veto_ok={veto_ok} reverts_under_lesion={revert_ok} "
          f"known_control_ok={known_ok}", flush=True)
    for r in veto_rows:
        print(f"      {r['prop']!r:26s} [{r['tag']:9s}] votes={r['votes']} real_vocab_known={r['real_vocab_known']} "
              f"unknown={r['unknown_tokens']} competent={r['competent']} -> vetoed={r['vetoed']} "
              f"lesion_votes={r['votes_under_lesion']} reverts={r['reverts_under_lesion']}", flush=True)
    for r in known_rows:
        print(f"      {r['prop']!r:26s} [{r['tag']:14s}] votes={r['votes']} real_vocab_known={r['real_vocab_known']} "
              f"corroborates={r['corroborates']}", flush=True)
    print(f"  (D) MOAT (no unstored/inconsistent committed on either arm): {moat_ok}", flush=True)
    for r in moat_rows:
        print(f"      {r['q']:24s} 2organ={r['two_organ']!r} 3organ={r['three_organ']!r} "
              f"a_recall={r['organ_a_recall']!r} reason={r['reason']}", flush=True)
    print(f"  ATTRIBUTABLE: non-comprehension veto owed to organ C (intact vs lesion) = {attr_organc}", flush=True)
    print(f"\n  VERDICT: {decided['status']}\n" + "=" * 110, flush=True)

    out = {"runner": "_gnw_three_organ_bus_verify", "go": decided["go"], "status": decided["status"],
           "verdict": decided, "backend": backend, "fix": "d4-real-vocab-competence-veto",
           "byte_identical_when_off": {"install_noop_when_off": off_noop,
                                       "runtime_flip_off_matches_two_organ": off_runtime_matches_head,
                                       "two_organ_out": head_out, "three_organ_off_out": off_runtime_out,
                                       "panel": off_panel},
           "no_regression": {"ok": noreg_ok, "rows": noreg_rows},
           "non_comprehension_veto": {"veto_ok": veto_ok, "reverts_under_lesion_ok": revert_ok,
                                      "known_control_ok": known_ok, "real_vocab_discriminates": real_vocab_discriminates,
                                      "veto_rows": veto_rows, "known_rows": known_rows},
           "moat": {"ok": moat_ok, "rows": moat_rows},
           "q3_unanimity_window": {"ok": unanimity_window_ok, "d_sub": d_sub, "rate_2votes": rate_2,
                                   "rate_3votes": rate_3, "THR": float(_NORGAN_THR)},
           "attributable": {"noncomp_veto_intact_vs_organc_lesion": attr_organc,
                            "intact_veto_rate": intact_veto_rate, "lesion_veto_rate": lesion_veto_rate},
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = f"research/findings/raw/_gnw_three_organ/production_verify_realvocab_{backend}.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if decided["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(run())
