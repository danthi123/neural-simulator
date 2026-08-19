"""GO-GATE VERIFY for wiring the GNW confidence/conflict-GATED deliberation (THE KEYSTONE, T1-1 rung d) into the LIVE
production `/api/brain-chat` turn (`webapp/gnw_deliberation.py` + the `install_deliberation_gate` call in
`webapp/server.py::brain_chat`). Verified numpy-CPU through the REAL production ChatBrain + the REAL handler.

THE WIRED FUNCTION. After the GNW ignition bus commits, the WORKSPACE'S OWN spiking conflict read (n_ignited + the
nmda_norm confidence balance) decides commit-vs-abstain. When the brain has >=2 genuinely-competing stored answers
under the SAME (agent, action) — today's bus commits the arbitrary FIRST-match patient — the candidates are driven
EQUALLY into the P1.2 GNW workspace; a sustained co-ignition / low-confidence read (the keystone acc_conflict_gate)
makes the brain ABSTAIN instead of committing the shaky answer (deliberation-until-sure + halt-if-unsure).

GO GATE (>=all four):
  (A) ABSTAIN-ON-CONFLICT (through the REAL handler + the REAL gate): on a genuinely-ambiguous prompt (two facts share
      (agent, action)) the wired brain ABSTAINS, while the pre-deliberation bus COMMITS the arbitrary first-match.
  (B) BYTE-IDENTICAL on the full reactive panel (recall/abstain/learn/anaphora): the deliberation gate == the
      pre-deliberation (pristine) gate turn-by-turn (in-process, EXACT), AND the real handler responses are md5-identical
      with the flag default-ON vs BRAIN_GNW_DELIBERATE=0 (separate processes).
  (C) LESION-LOAD-BEARING: BRAIN_GNW_DELIBERATE_LESION=1 (the workspace self-recurrence ZEROED) -> the conflict cannot
      co-ignite -> the brain COMMITS the shaky answer again (the abstain is the SPIKING competition, not a host len()).
  (D) MOAT-SAFE: it only ADDS abstentions on a genuine conflict; never un-abstains an already-abstained turn, never
      invents a fact, never flips a confident single-answer recall.

Run (numpy-CPU):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_deliberation_wired_verify
  # internal separate-process panel-hash mode (invoked by the runner; not called directly):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_deliberation_wired_verify --panel-hash
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")


# ── the reactive panel (stateless classes + stateful sequences) — recall / abstain / self / learn / anaphora ──
PANEL_STATELESS = [
    ("what does dog chase?", "stored"),        # single-answer recall -> commit (byte-identical)
    ("what does cat eat?", "stored"),
    ("what does brain use?", "stored"),
    ("what does brain learn?", "stored"),
    ("what does fish fly?", "unstored"),       # abstain
    ("what does dog eat?", "inconsistent"),    # abstain (dog does not eat in-store)
    ("what does cat chase?", "inconsistent"),  # abstain
    ("what are you", "self"),                  # host router / self
]
ACQUIRE_SEQ = [("sky hold cloud", None), ("what does sky hold?", ["sky", "hold", "cloud"])]     # learn -> recall
ANAPHORA_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it eat?", ["cat", "eat", "fish"])]


def _svo_eq(x, y) -> bool:
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


def _build(install_delib: bool):
    """The REAL production ChatBrain (rf recall). Install the bus gate always; the deliberation gate only when asked
    (install_delib=False => the PRISTINE pre-deliberation gate = today's production)."""
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    from webapp import gnw_deliberation as gdel
    chat, _src = _build_chat_brain("tiny-demo", "stub")
    gbs.install_bus_gate(chat)
    if install_delib:
        gdel.install_deliberation_gate(chat)
    return chat


def _run_seq(chat, seq):
    """Drive a sequence through chat.gate (the installed wrapper) and return per-turn SVOs (byte-identical unit)."""
    out = []
    for utt, _want in seq:
        svo = chat.gate(utt)
        out.append(list(svo) if svo is not None else None)
    return out


# ── (B) byte-identical: the deliberation gate == the pristine pre-deliberation gate, turn-by-turn ──
def _panel_byte_identical():
    rows = []
    # stateless: fresh brain per arm, run the whole panel (fresh brains avoid double side effects)
    pristine = _build(install_delib=False)
    wired = _build(install_delib=True)
    for q, cls in PANEL_STATELESS:
        p = pristine.gate(q)
        w = wired.gate(q)
        rows.append({"cls": cls, "q": q, "pristine": (list(p) if p is not None else None),
                     "wired": (list(w) if w is not None else None), "identical": _svo_eq(p, w)})
    # stateful sequence (learn -> recall + anaphora, one combined replay per arm -> 2 builds not 4): compare turn-by-turn
    stateful = ACQUIRE_SEQ + ANAPHORA_SEQ
    p_run = _run_seq(_build(install_delib=False), stateful)
    w_run = _run_seq(_build(install_delib=True), stateful)
    for i, (utt, _want) in enumerate(stateful):
        cls = "acquisition" if i < len(ACQUIRE_SEQ) else "anaphora"
        rows.append({"cls": cls, "q": utt, "pristine": p_run[i], "wired": w_run[i],
                     "identical": _svo_eq(p_run[i], w_run[i])})
    n_id = sum(int(r["identical"]) for r in rows)
    return {"rows": rows, "n_identical": n_id, "n_total": len(rows), "ok": (n_id == len(rows))}


# ── (A) + (C) + (D): the CONFLICT (two facts share (agent, action)) — abstain vs commit, lesion, moat ──
def _teach_conflict(chat):
    """Teach a SECOND fact under an existing (agent, action) key so the store holds two competing patients
    (dog chase cat [built-in] + dog chase bird [taught]); both patients are in-vocab (clean bindings)."""
    chat.inner.hear("dog chase bird", polarity="AFFIRM")


def _conflict_gate_arms():
    from webapp import gnw_deliberation as gdel
    q, agent, action = "what does dog chase?", "dog", "chase"

    # PRISTINE (pre-deliberation bus): commits the arbitrary first-match
    pristine = _build(install_delib=False)
    _teach_conflict(pristine)
    n_cands = len(gdel.all_candidate_patients(pristine.inner.composer, agent, action))
    pristine_svo = pristine.gate(q)

    # WIRED intact: the substrate conflict read -> ABSTAIN
    wired = _build(install_delib=True)
    _teach_conflict(wired)
    os.environ.pop("BRAIN_GNW_DELIBERATE", None)          # default-ON
    os.environ.pop("BRAIN_GNW_DELIBERATE_LESION", None)
    wired_svo = wired.gate(q)
    wired_info = dict(getattr(wired, "_last_gnw_delib", {}) or {})

    # WIRED lesion: the workspace self-recurrence zeroed -> the conflict cannot co-ignite -> COMMIT again
    os.environ["BRAIN_GNW_DELIBERATE_LESION"] = "1"
    lesion_svo = wired.gate(q)
    lesion_info = dict(getattr(wired, "_last_gnw_delib", {}) or {})
    os.environ.pop("BRAIN_GNW_DELIBERATE_LESION", None)

    # WIRED flag-off: pure pass-through -> COMMIT (same as pristine)
    os.environ["BRAIN_GNW_DELIBERATE"] = "0"
    off_svo = wired.gate(q)
    os.environ.pop("BRAIN_GNW_DELIBERATE", None)

    # a single-answer control (cat eat fish) must NOT be over-abstained by the wired brain
    single_pristine = pristine.gate("what does cat eat?")
    single_wired = wired.gate("what does cat eat?")

    return {
        "q": q, "n_candidates": n_cands,
        "pristine_svo": (list(pristine_svo) if pristine_svo is not None else None),
        "wired_svo": (list(wired_svo) if wired_svo is not None else None),
        "lesion_svo": (list(lesion_svo) if lesion_svo is not None else None),
        "off_svo": (list(off_svo) if off_svo is not None else None),
        "wired_info": {k: wired_info.get(k) for k in ("decision", "abstained", "n_candidates", "conf", "n_ignited")},
        "lesion_info": {k: lesion_info.get(k) for k in ("decision", "abstained", "n_candidates", "conf", "n_ignited")},
        "single_pristine": (list(single_pristine) if single_pristine is not None else None),
        "single_wired": (list(single_wired) if single_wired is not None else None),
        # (A) intact abstains where pristine commits
        "abstain_on_conflict": (pristine_svo is not None and wired_svo is None),
        # (C) lesion re-commits the shaky answer (load-bearing)
        "lesion_commits": (lesion_svo is not None),
        # flag-off is a pure pass-through (== pristine commit)
        "off_commits": _svo_eq(off_svo, pristine_svo),
        # (D) moat: never over-abstain a single clean answer; never un-abstain
        "single_not_flipped": _svo_eq(single_pristine, single_wired) and single_wired is not None,
    }


# ── 6-SEED robustness of the decisive workspace conflict read (the one seed-varying claim: does the substrate's
#    ignition/conflict read still separate 2-equal-candidates from 1 across substrate seeds?) ──
def _conflict_seed_sweep(seeds=(42, 43, 44, 100, 101, 102)):
    from webapp import gnw_deliberation as gdel
    rows = []
    for s in seeds:
        d2, c2, n2 = gdel.conflict_gate(2, seed=s, lesion=False)     # 2 equal competing answers -> conflict
        d1, c1, n1 = gdel.conflict_gate(1, seed=s, lesion=False)     # 1 clean answer -> confident
        dl, cl, nl = gdel.conflict_gate(2, seed=s, lesion=True)      # 2 answers, recurrence-zeroed -> conflict undetected
        ok = bool(d2 == "ABSTAIN" and d1 == "ADVANCE" and dl != "ABSTAIN")
        rows.append({"seed": s, "two_cand": d2, "one_cand": d1, "lesion_two": dl,
                     "conf2": round(c2, 3), "n2": n2, "conf1": round(c1, 3), "n1": n1, "nl": nl, "ok": ok})
    return {"rows": rows, "n_ok": sum(int(r["ok"]) for r in rows), "n": len(seeds),
            "all_ok": all(r["ok"] for r in rows)}


# ── (A) at the RESPONSE level, through the REAL /api/brain-chat handler ──
def _handler_conflict():
    """Teach the two competing facts via the REAL handler (acquisition messages), then query — the DEFAULT (deliberation
    on) response ABSTAINS; BRAIN_GNW_DELIBERATE=0 and _LESION=1 COMMIT. Heavy Gate-B organs off (identical on all arms)."""
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_CAUSAL", "BRAIN_PMEM", "BRAIN_PRAGMATIC", "BRAIN_SELF_INITIATE", "BRAIN_DISCOURSE_REGISTER",
              "BRAIN_RICH", "BRAIN_GNW_BUS"):
        os.environ[k] = "0"
    os.environ.pop("BRAIN_GNW_BUS", None)
    from webapp.server import brain_chat, BrainChatRequest as Req

    def _turn(session, msg):
        return json.loads(brain_chat(Req(session=session, message=msg, brain="tiny-demo",
                                         renderer="stub", rich=False)).body.decode("utf-8"))

    def _arm(session, env):
        for k in ("BRAIN_GNW_DELIBERATE", "BRAIN_GNW_DELIBERATE_LESION"):
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        _turn(session, "dog chase bird")                 # teach the competitor (dog chase cat is built-in)
        resp = _turn(session, "what does dog chase?")
        for k in env:
            os.environ.pop(k, None)
        return resp

    default = _arm("delib_default", {})                                   # deliberation ON (default)
    off = _arm("delib_off", {"BRAIN_GNW_DELIBERATE": "0"})               # pass-through
    lesion = _arm("delib_lesion", {"BRAIN_GNW_DELIBERATE_LESION": "1"})  # workspace lesion
    return {
        "default_abstained": bool(default.get("abstained")),
        "off_abstained": bool(off.get("abstained")),
        "lesion_abstained": bool(lesion.get("abstained")),
        "default_answer": default.get("answer"), "off_answer": off.get("answer"),
        "lesion_answer": lesion.get("answer"),
        "default_recalled": default.get("recalled_svo"), "off_recalled": off.get("recalled_svo"),
        # (A): default abstains; (C)/off: both commit
        "ok": bool(default.get("abstained") and not off.get("abstained") and not lesion.get("abstained")),
    }


# ── (B) separate-process handler byte-identical: reactive panel md5, flag default-ON vs BRAIN_GNW_DELIBERATE=0 ──
def _panel_hash_mode():
    """Build the REAL handler, run the reactive panel, print `PANELHASH <md5>` of the concatenated responses. Heavy
    organs off (identical on both arms; the point is the deliberation's inertness on the reactive panel)."""
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_CAUSAL", "BRAIN_PMEM", "BRAIN_PRAGMATIC", "BRAIN_SELF_INITIATE", "BRAIN_DISCOURSE_REGISTER",
              "BRAIN_RICH", "BRAIN_GNW_BUS"):
        os.environ[k] = "0"
    os.environ.pop("BRAIN_GNW_BUS", None)
    from webapp.server import brain_chat, BrainChatRequest as Req
    h = hashlib.md5()
    # ONE session (fewest builds) covering every reactive class: recall / abstain / self / learn / anaphora.
    msgs = [q for q, _ in PANEL_STATELESS] + [u for u, _ in ACQUIRE_SEQ] + [u for u, _ in ANAPHORA_SEQ]
    for msg in msgs:
        resp = json.loads(brain_chat(Req(session="ph", message=msg, brain="tiny-demo",
                                         renderer="stub", rich=False)).body.decode("utf-8"))
        resp.pop("timing_ms", None)                      # drop any wall-clock noise if present
        h.update(json.dumps(resp, sort_keys=True, default=str).encode("utf-8"))
    print(f"PANELHASH {h.hexdigest()}", flush=True)
    return 0


def _spawn_panel_hash(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("SIM_BACKEND", "numpy")
    env.setdefault("BRAIN_COMPOSER_KIND", "rf")
    out = subprocess.run([sys.executable, "-u", "-m", "research.runners._gnw_deliberation_wired_verify", "--panel-hash"],
                         cwd=_REPO, env=env, capture_output=True, text=True, timeout=1800)
    for line in out.stdout.splitlines():
        if line.startswith("PANELHASH "):
            return line.split(" ", 1)[1].strip()
    raise RuntimeError(f"panel-hash subprocess produced no hash; stderr tail:\n{out.stderr[-800:]}")


def main():
    if "--panel-hash" in sys.argv:
        return _panel_hash_mode()

    from tools.verdict import Verdict
    from tools.lab import attributable_to

    byte = _panel_byte_identical()
    conflict = _conflict_gate_arms()
    seed_sweep = _conflict_seed_sweep()
    handler = _handler_conflict()

    # (B) separate-process handler hashes: default-ON vs BRAIN_GNW_DELIBERATE=0
    hash_on = _spawn_panel_hash({})
    hash_off = _spawn_panel_hash({"BRAIN_GNW_DELIBERATE": "0"})
    handler_hash_identical = (hash_on == hash_off)

    # ATTRIBUTION (tools.lab): whose is the abstain — the SPIKING workspace conflict read, or a host len()? treatment =
    # the INTACT abstain-rate on the conflict (1.0); control = the LESIONED abstain-rate (0.0 = it commits). A full
    # separation attributes the abstain to the workspace ignition (lesioning it removes the abstain entirely).
    attribution = attributable_to(
        "the conflict-gated ABSTAIN owed to the spiking workspace conflict read (not a host len(candidates)>=2)",
        1.0 if conflict["abstain_on_conflict"] else 0.0,
        0.0 if conflict["lesion_commits"] else 1.0)

    gateA = bool(conflict["abstain_on_conflict"] and handler["ok"] and conflict["n_candidates"] == 2
                 and seed_sweep["all_ok"])
    gateB = bool(byte["ok"] and handler_hash_identical)
    gateC = bool(conflict["lesion_commits"] and conflict["off_commits"])
    gateD = bool(conflict["single_not_flipped"])

    go = bool(gateA and gateB and gateC and gateD)

    v = Verdict("GNW confidence/conflict-gated deliberation WIRED into /api/brain-chat")
    v.require("(A) abstain-on-conflict: pristine COMMITS, wired ABSTAINS (real gate)", conflict["abstain_on_conflict"],
              expect=True, note=f"pristine={conflict['pristine_svo']} wired={conflict['wired_svo']} "
                                f"n_candidates={conflict['n_candidates']}")
    v.require("(A) 6-seed robustness: the workspace conflict read separates 2-cand->ABSTAIN / 1-cand->ADVANCE / "
              "lesion-2-cand->COMMIT on all 6 seeds", seed_sweep["all_ok"], expect=True,
              note=f"{seed_sweep['n_ok']}/{seed_sweep['n']} seeds")
    v.require("(A) real handler: default ABSTAINS, flag-off + lesion COMMIT", handler["ok"], expect=True,
              note=f"default_abstained={handler['default_abstained']} off={handler['off_abstained']} "
                   f"lesion={handler['lesion_abstained']}")
    v.require("(B) reactive panel byte-identical: wired gate == pristine gate (in-process)", byte["ok"], expect=True,
              note=f"{byte['n_identical']}/{byte['n_total']} turns identical")
    v.require("(B) real handler byte-identical: flag default-ON == BRAIN_GNW_DELIBERATE=0 (separate processes)",
              handler_hash_identical, expect=True, note=f"on={hash_on} off={hash_off}")
    v.require("(C) LESION load-bearing: recurrence-zeroed workspace RE-COMMITS the shaky answer", conflict["lesion_commits"],
              expect=True, note=f"lesion_svo={conflict['lesion_svo']} info={conflict['lesion_info']}")
    v.require("(C) flag-off is a pure pass-through (commits == pristine)", conflict["off_commits"], expect=True)
    v.require("(D) moat-safe: a single clean answer is NOT over-abstained/flipped", conflict["single_not_flipped"],
              expect=True, note=f"single pristine={conflict['single_pristine']} wired={conflict['single_wired']}")
    # dissociation: treatment = the INTACT abstain-rate on the conflict (1.0 = abstains); control = the LESIONED
    # abstain-rate (0.0 = it COMMITS the shaky answer). A full 1.0 separation => the abstain is the spiking workspace.
    v.control("the abstain is owed to the SPIKING workspace conflict read (intact ABSTAINS, lesion COMMITS)",
              treatment=(1.0 if conflict["abstain_on_conflict"] else 0.0),
              control=(0.0 if conflict["lesion_commits"] else 1.0), min_separation=0.5)
    v.disabled("heavy Gate-B organs (affect/worldmodel/... = 0) in the handler checks",
               why="disabled ONLY for speed; they run identically on every flag arm, so the comparison is unaffected")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_gnw_deliberation_wired_verify", "go": go, "status": decided["status"],
        "gateA_abstain_on_conflict": gateA, "gateB_byte_identical": gateB,
        "gateC_lesion_load_bearing": gateC, "gateD_moat_safe": gateD,
        "seed_sweep": seed_sweep, "attribution_to_workspace": attribution,
        "conflict": conflict, "handler_conflict": handler, "byte_identical": byte,
        "handler_hash_on": hash_on, "handler_hash_off": hash_off, "handler_hash_identical": handler_hash_identical,
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    op = "research/findings/raw/_gnw_deliberation_wired/verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 104, flush=True)
    print("  GNW CONFIDENCE/CONFLICT-GATED DELIBERATION — WIRED into /api/brain-chat (real ChatBrain + handler, numpy)", flush=True)
    print("=" * 104, flush=True)
    print(f"  (A) abstain-on-conflict: pristine={conflict['pristine_svo']} -> wired={conflict['wired_svo']} "
          f"(n_candidates={conflict['n_candidates']}, info={conflict['wired_info']})", flush=True)
    print(f"      handler: default_abstained={handler['default_abstained']} off={handler['off_abstained']} "
          f"lesion={handler['lesion_abstained']} | default_answer={handler['default_answer']!r}", flush=True)
    print(f"  (A) 6-seed sweep: {seed_sweep['n_ok']}/{seed_sweep['n']} "
          f"(2cand->ABSTAIN / 1cand->ADVANCE / lesion->COMMIT) | attribution_to_workspace={attribution}", flush=True)
    for r in seed_sweep["rows"]:
        print(f"        seed {r['seed']}: two={r['two_cand']}(n{r['n2']}) one={r['one_cand']}(n{r['n1']}) "
              f"lesion={r['lesion_two']}(n{r['nl']}) ok={r['ok']}", flush=True)
    print(f"  (B) reactive panel byte-identical (in-process): {byte['n_identical']}/{byte['n_total']}", flush=True)
    for r in byte["rows"]:
        mark = "OK " if r["identical"] else "DIVERGE"
        print(f"        [{mark}] {r['cls']:12s} {r['q']:26s} pristine={r['pristine']} wired={r['wired']}", flush=True)
    print(f"  (B) handler hash default-ON == flag-0: {handler_hash_identical}  (on={hash_on} off={hash_off})", flush=True)
    print(f"  (C) lesion re-commits: {conflict['lesion_commits']} (lesion_svo={conflict['lesion_svo']} "
          f"info={conflict['lesion_info']}) | off_commits={conflict['off_commits']}", flush=True)
    print(f"  (D) moat single-answer not flipped: {conflict['single_not_flipped']} "
          f"(pristine={conflict['single_pristine']} wired={conflict['single_wired']})", flush=True)
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} (status={decided['status']})", flush=True)
    print(f"  [saved] {op}\n" + "=" * 104, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
