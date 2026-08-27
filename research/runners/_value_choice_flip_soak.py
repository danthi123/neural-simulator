"""SOAK / no-regression gate for the VALUE-DRIVEN-CHOICE DEFAULT-ON flip (BRAIN_VALUE_CHOICE).

The parent runs THIS on the pool before flipping the value-choice faculty default-on. It runs the SAME multi-turn
panel on the SAME ChatBrain twice — flag OFF vs flag ON — through the REAL production wiring
(research.runners.value_choice_production_organ.install_value_choice on top of the GNW deliberation keystone). The bar:

  NO-REGRESSION (the 6-seed hard gate): with the flag ON, every ORDINARY turn's reply is BYTE-IDENTICAL to flag OFF.
  A confident single-fact recall, a single-candidate / untaught abstain (the moat), and a self/identity turn are all
  unchanged. The faculty ONLY changes its TRIGGERED turns — a >=2-distinct-patient (agent, action) recall that the
  deliberation keystone ABSTAINS on: OFF -> "I don't know about that." ; ON -> COMMIT the higher-VALUE patient.

  LOAD-BEARING (confirmation, not the no-regression gate): on the triggered turn, ON commits a stored patient (never
  invents one -> the moat holds), and the LESION (BRAIN_VALUE_CHOICE_LESION=1) reverts it to the OFF abstain.

Scenario (facts: dog->chase->{cat, ball} [the ambiguity], cat->eat->fish [confident], brain->use->spikes [self-ish]):
  T1 "what does cat eat"   -> confident recall (byte-identical OFF/ON)
  T2 "what does fox hunt"  -> untaught -> moat abstain, <2 candidates (byte-identical OFF/ON)
  T3 "what do you know"    -> self/none (byte-identical OFF/ON)
  T4 "what does dog chase" -> the TRIGGER (>=2 patients): OFF abstains ; ON commits cat|ball ; LESION reverts to abstain

Run (the parent, on the pool):
  SIM_BACKEND=numpy python -m research.runners._value_choice_flip_soak --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_value_choice_prodflip/soak_6seed.json
  # fast harness smoke (mocks; no bridge build):
  SIM_BACKEND=numpy python -m research.runners._value_choice_flip_soak --smoke
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The ordinary (byte-identical) panel + the trigger turn.
ORDINARY_TURNS = ["what does cat eat", "what does fox hunt", "what do you know about it"]
TRIGGER_TURN = "what does dog chase"
AMBIGUOUS = ("dog", "chase", ["cat", "ball"])   # the >=2-distinct-patient (agent, action)


def _set_flags(*, on: bool, lesion: bool = False):
    if on:
        os.environ["BRAIN_VALUE_CHOICE"] = "1"
    else:
        # 2026-08-27 fix: BRAIN_VALUE_CHOICE defaults ON (wave-1/2 flip, _VALUE_CHOICE_DEFAULT_ON=True) -- unset no
        # longer means OFF, so the OFF arm must set the byte-identical escape explicitly.
        os.environ["BRAIN_VALUE_CHOICE"] = "0"
    if lesion:
        os.environ["BRAIN_VALUE_CHOICE_LESION"] = "1"
    else:
        os.environ.pop("BRAIN_VALUE_CHOICE_LESION", None)


def _answer(chat, q):
    """One turn -> the user-facing answer string (mirrors the webapp single-fact path: gate -> render / abstain)."""
    try:
        gate_svo = chat.gate(q)
    except Exception as e:
        return f"__ERROR__ {type(e).__name__}: {e}"
    if gate_svo is None:
        return "I don't know about that."
    try:
        return chat.render(gate_svo)
    except Exception as e:
        return f"__ERROR__ {type(e).__name__}: {e}"


def _build_chat(seed, composer_kind):
    """Build a tiny-demo ChatBrain, add the AMBIGUOUS patient, and install the deliberation keystone (guarded) + the
    value-choice wrapper (mirrors the webapp install order: value-choice OUTSIDE deliberation). The value-choice
    context_fn reads a mutable holder so the soak can VARY the engagement through the REAL gate (the live VARY proof).
    Returns (chat, VC, ctx_holder)."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo, DEFAULT_SELF_ALIASES
    from research.runners import value_choice_production_organ as VC
    agent, aliases, _n = _build_tiny_demo(seed=seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind=composer_kind)
    inner = getattr(agent, "agent", agent)
    # the SECOND patient for (dog, chase) -> the ambiguity the deliberation keystone resolves by first-match/abstain.
    inner.hear("dog chase ball", polarity="AFFIRM")
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    chat._refresh_facts()
    # install the GNW deliberation keystone (the >=2-competing arbitrator) if available, then the value-choice wrapper.
    try:
        from webapp import gnw_deliberation as _delib
        _delib.install_deliberation_gate(chat)
    except Exception:
        pass
    ctx_holder = {"favour": None}   # None -> the default recency context; else a patient string to favour (e=1)

    def _ctx(a, v, cands):
        fav = ctx_holder["favour"]
        if fav is None:
            return VC.default_context_fn(chat)(a, v, cands)
        return [1.0 if p == fav else 0.0 for p in cands]

    VC.install_value_choice(chat, seed=seed, context_fn=_ctx)
    return chat, VC, ctx_holder


def run_seed(seed, a):
    t0 = time.time()
    chat, VC, ctx = _build_chat(seed, a.composer)
    row = {"seed": int(seed), "composer": a.composer}

    # ── ORDINARY panel: flag OFF vs ON must be byte-identical (the no-regression HARD gate) ──
    _set_flags(on=False)
    off_ord = [_answer(chat, q) for q in ORDINARY_TURNS]
    _set_flags(on=True)
    on_ord = [_answer(chat, q) for q in ORDINARY_TURNS]
    _set_flags(on=False)
    ordinary_identical = (off_ord == on_ord)
    row["ordinary_off"] = off_ord
    row["ordinary_on"] = on_ord
    row["ordinary_byte_identical"] = bool(ordinary_identical)

    # ── TRIGGER turn: OFF = the pipeline's arbitration (first-match / abstain) ; ON = the value-driven commit ──
    _set_flags(on=False)
    trig_off = _answer(chat, TRIGGER_TURN)
    row["trigger_off"] = trig_off
    row["trigger_off_abstains"] = bool(trig_off.strip().lower().startswith("i don't know"))

    trig_favA = trig_favB = trig_lesion = None
    committed_stored = vary_flips = lesion_reverts = None
    cand0, cand1 = AMBIGUOUS[2][0], AMBIGUOUS[2][1]
    if not a.no_organ:
        # LIVE VARY (the load-bearing proof through the REAL gate): favour cand0, then cand1 -> the committed patient
        # must FLIP (both are STORED patients -> the moat holds).
        _set_flags(on=True)
        ctx["favour"] = cand0
        trig_favA = _answer(chat, TRIGGER_TURN)
        lastA = getattr(chat, "_value_choice_last", None)
        ctx["favour"] = cand1
        trig_favB = _answer(chat, TRIGGER_TURN)
        lastB = getattr(chat, "_value_choice_last", None)
        chosenA = (lastA or {}).get("chosen")
        chosenB = (lastB or {}).get("chosen")
        committed_stored = (chosenA in AMBIGUOUS[2] and chosenB in AMBIGUOUS[2])
        vary_flips = (chosenA == cand0 and chosenB == cand1 and chosenA != chosenB)
        # LESION: with the coupling lesioned the value gradient vanishes -> revert to the INNER (flag-off) result.
        ctx["favour"] = cand0
        _set_flags(on=True, lesion=True)
        trig_lesion = _answer(chat, TRIGGER_TURN)
        lesion_reverts = (trig_lesion == trig_off)
        _set_flags(on=False)
        ctx["favour"] = None
        row["value_choice_lastA"] = lastA
        row["value_choice_lastB"] = lastB
    row["trigger_favour0"] = trig_favA
    row["trigger_favour1"] = trig_favB
    row["trigger_lesion"] = trig_lesion
    row["trigger_commits_stored_patient"] = (None if committed_stored is None else bool(committed_stored))
    row["trigger_vary_flips_commit"] = (None if vary_flips is None else bool(vary_flips))
    row["trigger_lesion_reverts_to_inner"] = (None if lesion_reverts is None else bool(lesion_reverts))
    row["elapsed_s"] = round(time.time() - t0, 1)

    # per-seed pass: the no-regression HARD gate is ordinary_byte_identical; the load-bearing confirmation (when the
    # organ ran) is that VARYing the engagement FLIPS the committed patient AND the lesion reverts to the inner result.
    lb_ok = True
    if not a.no_organ:
        lb_ok = bool(committed_stored and vary_flips and lesion_reverts)
    row["load_bearing_ok"] = lb_ok
    row["seed_pass"] = bool(ordinary_identical and lb_ok)
    print(f"[soak seed={seed}] ordinary_byte_identical={ordinary_identical} trigger OFF={trig_off!r} "
          f"favour0={trig_favA!r} favour1={trig_favB!r} LESION={trig_lesion!r} lb_ok={lb_ok} ({row['elapsed_s']}s)",
          flush=True)
    return row


def decide(rows):
    n = len(rows)
    ord_pass = sum(1 for r in rows if r["ordinary_byte_identical"])
    lb_pass = sum(1 for r in rows if r["load_bearing_ok"])
    all_pass = sum(1 for r in rows if r["seed_pass"])
    verdict = "GO" if (ord_pass == n and all_pass == n) else "NO-GO"
    return verdict, {"n_seeds": n, "ordinary_byte_identical_pass": ord_pass,
                     "load_bearing_pass": lb_pass, "seed_pass": all_pass}


def run(a):
    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[value-choice soak] seeds={seeds} composer={a.composer} no_organ={a.no_organ}", flush=True)
    rows = [run_seed(s, a) for s in seeds]
    verdict, detail = decide(rows)
    print(f"\n{'#'*90}\n  VALUE-CHOICE FLIP SOAK: {verdict}", flush=True)
    print(f"  ordinary byte-identical {detail['ordinary_byte_identical_pass']}/{detail['n_seeds']} | "
          f"load-bearing {detail['load_bearing_pass']}/{detail['n_seeds']} | "
          f"seed-pass {detail['seed_pass']}/{detail['n_seeds']}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s\n{'#'*90}", flush=True)
    # preconditions carried WITH the verdict (tools.verdict.Verdict): the no-regression HARD gate + the load-bearing
    # confirmation, earned per seed. An unguarded verdict is itself the defect (gates/verdict_preconditions).
    from tools.verdict import Verdict
    V = Verdict("value_choice_flip_soak")
    V.require("ordinary turns byte-identical OFF-vs-ON (all seeds)",
              detail["ordinary_byte_identical_pass"] == detail["n_seeds"])
    V.require("load-bearing: engagement-VARY flips the commit + LESION reverts to inner (all seeds)",
              detail["load_bearing_pass"] == detail["n_seeds"],
              note="skipped-as-True when --no-organ (ordinary panel only)")
    V.decide(go=(verdict == "GO"), verbose=False)
    out = {"probe": "value_choice_flip_soak", "verdict": verdict, "seeds": seeds,
           "config": {"composer": a.composer, "no_organ": a.no_organ,
                      "ordinary_turns": ORDINARY_TURNS, "trigger_turn": TRIGGER_TURN, "ambiguous": AMBIGUOUS},
           "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
           "preconditions": V.to_dict()["preconditions"],
           "detail": detail, "per_seed": rows, "elapsed_total_s": time.time() - t0}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"  [saved] {a.out}", flush=True)
    return out


def smoke():
    """Fast harness check (MOCK chat + MOCK organ; no bridge build): the panel logic + the flag toggling + the
    verdict aggregator are well-formed."""
    from research.runners import value_choice_production_organ as VC

    class _Router:
        self_aliases = {"brain", "you", "i", "me"}

    class _Chat:
        def __init__(self):
            self.stored_facts = [("dog", "chase", "cat"), ("dog", "chase", "ball"), ("cat", "eat", "fish")]
            self.agents_set = {"dog", "cat"}
            self.actions_set = {"chase", "eat"}
            self.router = _Router()
            self.is_multiturn = False

        def gate(self, q):
            ql = q.lower()
            if "cat" in ql and "eat" in ql:
                return ["cat", "eat", "fish"]
            return None

        def render(self, svo):
            return " ".join(svo) + "."

    class _Organ:
        untrained = False
        def ensure_built(self):
            return self
        def choose(self, cands, engagements, *, lesion=False, salience_seed=0):
            if lesion:
                return None, {"lesion": True}
            i = int(max(range(len(cands)), key=lambda k: engagements[k]))
            return cands[i], {"wta_choice": i}

    chat = _Chat()
    VC.install_value_choice(chat, organ=_Organ())
    # ordinary byte-identical
    _set_flags(on=False)
    off = [_answer(chat, q) for q in ORDINARY_TURNS]
    _set_flags(on=True)
    on = [_answer(chat, q) for q in ORDINARY_TURNS]
    _set_flags(on=False)
    ord_ok = (off == on)
    # trigger
    trig_off = _answer(chat, TRIGGER_TURN)
    _set_flags(on=True)
    trig_on = _answer(chat, TRIGGER_TURN)
    _set_flags(on=True, lesion=True)
    trig_les = _answer(chat, TRIGGER_TURN)
    _set_flags(on=False)
    trig_ok = (trig_off.lower().startswith("i don't know") and not trig_on.lower().startswith("i don't know")
               and trig_les == trig_off)
    v, d = decide([{ "ordinary_byte_identical": True, "load_bearing_ok": True, "seed_pass": True}])
    ok = ord_ok and trig_ok and v == "GO"
    print(f"[soak SMOKE] ordinary_identical={ord_ok} (off={off})")
    print(f"[soak SMOKE] trigger OFF={trig_off!r} ON={trig_on!r} LESION={trig_les!r} trig_ok={trig_ok}")
    print(f"[soak SMOKE] verdict-aggregator GO={v == 'GO'}")
    print(f"[soak SMOKE] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="value-driven-choice DEFAULT-ON flip soak (no-regression + load-bearing).")
    p.add_argument("--smoke", action="store_true", help="fast mock harness check (no bridge build)")
    p.add_argument("--seeds", default="42,43,44,100,101,102")
    p.add_argument("--composer", default="onebrain", help="tiny-demo recall composer ('onebrain' production default, "
                                                          "'rf' for the fast numpy path)")
    p.add_argument("--no-organ", action="store_true",
                   help="skip the triggered-turn value organ build (run only the ordinary byte-identical panel; fast)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)
    if a.smoke:
        raise SystemExit(0 if smoke() else 1)
    run(a)


if __name__ == "__main__":
    main()
