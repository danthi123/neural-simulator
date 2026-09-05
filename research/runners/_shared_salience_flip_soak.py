"""SOAK / no-regression + anti-hollow gate for the SHARED-SALIENCE DEFAULT-ON flip (`BRAIN_SHARED_SALIENCE`),
scoped to its value-choice consumer (scaffold-retirement backlog rank-20; Track-1 flip campaign).

WHY THIS RUNNER EXISTS (the gap neither existing GO closes). Two already-landed 6-seed-GO findings cover this
mechanism at the ORGAN / CONTEXT-FUNCTION level, both through a `_FakeChat` stand-in, never the real conversational
entry point `/api/brain-chat` actually calls:
  - `research/findings/2026-09-05-shared-spiking-salience-afferent-wired-GO.md` (rank-4) -- the shared organ core +
    all 3 consumer sites' host-arithmetic-vs-mediated engagement, `_FakeChat`, 6 seeds.
  - `research/findings/2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md` (rank-20) -- the
    REAL trained `striosome_value` critic fed the mediated context, `_FakeChat`, 6 seeds x 4 scenarios.
Both findings' own "Honest scope" sections name the SAME residual: *"A default-ON flip needs its own no-regression
soak on the live production default ... which this de-risk does not attempt."* THIS runner is that soak -- it
builds the REAL `research.runners.brain_chat_tui.ChatBrain` (the object `/api/brain-chat` actually drives, the
'onebrain' genuinely-spiking composer, the production default) with `value_choice_production_organ.install_value_
choice` wrapping `chat.gate` exactly as production installs it, then drives it through `chat.gate()` / `chat.
render()` -- not by calling `default_context_fn`/`ValueChoiceProductionOrgan.choose()` directly.

THE BAR (this agent's own mandate, mirrors `_value_choice_flip_soak.py`'s BRAIN_VALUE_CHOICE precedent):
  NO-REGRESSION (hard gate) -- ORDINARY turns (confident single-fact recall, an untaught abstain, a self/identity
    query -- none of them the >=2-candidate ambiguity value-choice touches) are BYTE-IDENTICAL whether
    `BRAIN_SHARED_SALIENCE` is off (today's shipped default) or on (this flip). `BRAIN_VALUE_CHOICE` is left at ITS
    OWN existing production default (unset -> ON, the 2026-08-26 flip) throughout -- this soak tests ONLY the
    INCREMENTAL effect of the shared-salience mediation, not value-choice's own (already-GO'd) load-bearing-ness.
  ANTI-HOLLOW (the crux) -- on the trigger turns (>=2 stored patients sharing an (agent,action)), the REAL critic's
    fed engagement context measurably DIFFERS between the shared-salience-off and -on arms on every seed
    (c_on_loadbearing) -- proving the mediation is genuinely in the path reaching the trained critic through the
    conversational gate, not a coincidental no-op -- and `BRAIN_SHARED_SALIENCE_LESION=1` collapses that
    differentiation back toward the OFF-like / salience-only floor on every seed (c_lesion_collapses). A 4-candidate
    near-tie scenario (mirrors rank-20's own S4, the one scenario that showed a genuine cross-arm reordering) is
    used as the sharpest test of this, alongside a wide-separation 2-candidate scenario (mirrors S1) where
    off/on/lesion all matching-or-improving is the expected, unsurprising signature.
  THE MOAT -- every commit across both trigger scenarios, in every arm, is either None (abstain) or one of the
    STORED candidates -- never an invented patient.

Scenario (built on TOP of `_build_tiny_demo`'s base demo facts, which already include `dog chase cat` and
`cat eat fish` -- see `research/runners/brain_chat_tui.py`'s tiny-demo corpus):
  ORDINARY: "what does cat eat" (confident recall) / "what does fox hunt" (untaught, no stored fact -> abstain) /
            "what do you know about it" (self/identity -- <2 candidates, never reaches value-choice)
  TRIGGER_2CAND "what does bird eat" -- 2 stored patients (worm, seed), WIDE recency separation [0.0, 1.0]
            (mirrors rank-20's S1_baseline -- off/on expected to MATCH).
  TRIGGER_4CAND "what does dog chase" -- 4 stored patients (cat, ball, shoe, stick), NEAR-TIE recency ladder
            [0.0, 1/3, 2/3, 1.0] (mirrors rank-20's S4_four_candidate -- the scenario that showed 2/6 genuine
            cross-arm reorderings; the sharpest available anti-hollow probe).

Run (controller, 6-seed gate; each seed subprocess-isolated -- the trained critic + the process-shared curiosity
organ singleton are NOT safely re-buildable in one process across seeds, same reason the sibling de-risks give):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_flip_soak --seeds 42 43 44 100 101 102 \\
      --out research/findings/raw/_shared_salience_flip_soak/verify_6seed.json
  # single-seed worker (what the controller subprocess-fans; also runnable standalone):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_flip_soak --seed 42 --out .../seed42.json
  # fast harness smoke (mocks; no bridge build):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_flip_soak --smoke
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# REUSE-BY-IMPORT (no duplicated flag logic): the SAME (fixed, 2026-09-05) env-flag helpers rank-4's own de-risk
# uses -- _clear_flags() now sets the explicit byte-identical escape rather than unsetting the var, because
# BRAIN_SHARED_SALIENCE defaults ON post-flip (see that module's _clear_flags docstring for the full story).
from research.runners._shared_salience_afferent_derisk import _clear_flags, _set_flags  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

ORDINARY_TURNS = ["what does cat eat", "what does fox hunt", "what do you know about it"]
TRIGGER_2CAND = "what does bird eat"
TRIGGER_4CAND = "what does dog chase"
CANDS_2 = ("worm", "seed")
CANDS_4 = ("cat", "ball", "shoe", "stick")


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
    """Build the REAL production ChatBrain ('onebrain' composer default) with the deliberation keystone + the
    value-choice wrapper installed EXACTLY as production installs them (mirrors `_value_choice_flip_soak.py`'s own
    `_build_chat`) -- but, unlike that soak, `context_fn` is left at its default (`default_context_fn(chat)`, the
    REAL shared-salience-reading engagement context), because THIS soak's question is whether the shared-salience
    mediation is load-bearing on the real critic, not whether value-choice itself is (already GO'd)."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners import value_choice_production_organ as VC
    agent, aliases, _n = _build_tiny_demo(seed=seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind=composer_kind)
    inner = getattr(agent, "agent", agent)
    # dog/chase: base demo already stores ("dog","chase","cat") at the earliest index -> append ball/shoe/stick to
    # complete the 4-candidate near-tie recency ladder [0, 1/3, 2/3, 1] (mirrors rank-20's S4 exactly).
    for p in ("ball", "shoe", "stick"):
        inner.hear(f"dog chase {p}", polarity="AFFIRM")
    # bird/eat: a FRESH (agent,action) pair, 2 candidates, wide recency separation [0, 1] (mirrors rank-20's S1).
    for p in ("worm", "seed"):
        inner.hear(f"bird eat {p}", polarity="AFFIRM")
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    chat._refresh_facts()
    try:
        from webapp import gnw_deliberation as _delib
        _delib.install_deliberation_gate(chat)
    except Exception:
        pass
    # BRAIN_VALUE_CHOICE is left UNSET (its own production default, ON since 2026-08-26) -- this soak varies ONLY
    # BRAIN_SHARED_SALIENCE via the module-level env, re-checked live by default_context_fn() on every call.
    VC.install_value_choice(chat, seed=seed)
    return chat, VC


def _diag_context(chat, VC, a, v, cands):
    """A read-only diagnostic call into the SAME `default_context_fn` machinery the live gate just used (not a
    second organ, not a re-derivation from a different formula) -- reports the fed engagement floats so
    c_on_loadbearing / c_lesion_collapses can be checked on the ACTUAL numbers reaching the critic, not inferred
    solely from the categorical commit (which rank-20's own S1-S3 evidence says can legitimately stay unchanged
    even when the mediation is genuinely live)."""
    ctx = VC.default_context_fn(chat)
    return ctx(a, v, list(cands))


def run_seed(seed, a):
    t0 = time.time()
    _clear_flags()   # OFF -- today's shipped default (the explicit byte-identical escape, not unset)
    chat, VC = _build_chat(seed, a.composer)
    build_s = time.time() - t0
    row = {"seed": int(seed), "composer": a.composer, "build_seconds": round(build_s, 1)}

    # ── ORDINARY panel: shared-salience OFF vs ON must be BYTE-IDENTICAL (the no-regression HARD gate). Neither
    #    arm touches BRAIN_VALUE_CHOICE, so this isolates the shared-salience flip's OWN blast radius. ──
    _clear_flags()
    off_ord = [_answer(chat, q) for q in ORDINARY_TURNS]
    _set_flags(on=True)
    on_ord = [_answer(chat, q) for q in ORDINARY_TURNS]
    _clear_flags()
    ordinary_identical = (off_ord == on_ord)
    row["ordinary_off"] = off_ord
    row["ordinary_on"] = on_ord
    row["ordinary_byte_identical"] = bool(ordinary_identical)

    # ── TRIGGER_2CAND (wide separation, mirrors S1 -- expect OFF/ON to MATCH) ──
    _clear_flags()
    ctx_off_2 = _diag_context(chat, VC, "bird", "eat", CANDS_2)
    ans_off_2 = _answer(chat, TRIGGER_2CAND)
    last_off_2 = getattr(chat, "_value_choice_last", None)
    _set_flags(on=True)
    ctx_on_2 = _diag_context(chat, VC, "bird", "eat", CANDS_2)
    ans_on_2 = _answer(chat, TRIGGER_2CAND)
    last_on_2 = getattr(chat, "_value_choice_last", None)
    _set_flags(on=True, lesion=True)
    ctx_les_2 = _diag_context(chat, VC, "bird", "eat", CANDS_2)
    ans_les_2 = _answer(chat, TRIGGER_2CAND)
    last_les_2 = getattr(chat, "_value_choice_last", None)
    _clear_flags()
    row["trigger_2cand"] = {
        "off_ctx": ctx_off_2, "on_ctx": ctx_on_2, "lesion_ctx": ctx_les_2,
        "off_answer": ans_off_2, "on_answer": ans_on_2, "lesion_answer": ans_les_2,
        "off_chosen": (last_off_2 or {}).get("chosen"), "on_chosen": (last_on_2 or {}).get("chosen"),
        "lesion_chosen": (last_les_2 or {}).get("chosen"),
        "off_meta": (last_off_2 or {}).get("meta"), "on_meta": (last_on_2 or {}).get("meta"),
        "lesion_meta": (last_les_2 or {}).get("meta"),
    }

    # ── TRIGGER_4CAND (near-tie ladder, mirrors S4 -- the sharpest anti-hollow probe: rank-20 saw 2/6 seeds
    #    genuinely reorder here) ──
    _clear_flags()
    ctx_off_4 = _diag_context(chat, VC, "dog", "chase", CANDS_4)
    ans_off_4 = _answer(chat, TRIGGER_4CAND)
    last_off_4 = getattr(chat, "_value_choice_last", None)
    _set_flags(on=True)
    ctx_on_4 = _diag_context(chat, VC, "dog", "chase", CANDS_4)
    ans_on_4 = _answer(chat, TRIGGER_4CAND)
    last_on_4 = getattr(chat, "_value_choice_last", None)
    _set_flags(on=True, lesion=True)
    ctx_les_4 = _diag_context(chat, VC, "dog", "chase", CANDS_4)
    ans_les_4 = _answer(chat, TRIGGER_4CAND)
    last_les_4 = getattr(chat, "_value_choice_last", None)
    _clear_flags()
    row["trigger_4cand"] = {
        "off_ctx": ctx_off_4, "on_ctx": ctx_on_4, "lesion_ctx": ctx_les_4,
        "off_answer": ans_off_4, "on_answer": ans_on_4, "lesion_answer": ans_les_4,
        "off_chosen": (last_off_4 or {}).get("chosen"), "on_chosen": (last_on_4 or {}).get("chosen"),
        "lesion_chosen": (last_les_4 or {}).get("chosen"),
        "off_meta": (last_off_4 or {}).get("meta"), "on_meta": (last_on_4 or {}).get("meta"),
        "lesion_meta": (last_les_4 or {}).get("meta"),
    }

    # ── GATES ──
    def _spread(ctx):
        return float(max(ctx) - min(ctx)) if ctx else 0.0

    on_spread_2, les_spread_2 = _spread(ctx_on_2), _spread(ctx_les_2)
    on_spread_4, les_spread_4 = _spread(ctx_on_4), _spread(ctx_les_4)
    attrib_2 = attributable_to("seed %d 2cand: ON spread attributable to the shared-salience pathway" % seed,
                               on_spread_2, les_spread_2)
    attrib_4 = attributable_to("seed %d 4cand: ON spread attributable to the shared-salience pathway" % seed,
                               on_spread_4, les_spread_4)
    ctx_differs_2 = any(abs(x - y) > 1e-6 for x, y in zip(ctx_off_2, ctx_on_2))
    ctx_differs_4 = any(abs(x - y) > 1e-6 for x, y in zip(ctx_off_4, ctx_on_4))
    all_commits = [row["trigger_2cand"][k] for k in ("off_chosen", "on_chosen", "lesion_chosen")] \
        + [row["trigger_4cand"][k] for k in ("off_chosen", "on_chosen", "lesion_chosen")]
    moat_holds = all((c is None) or (c in CANDS_2) or (c in CANDS_4) for c in all_commits)
    reorders_4 = bool(row["trigger_4cand"]["off_chosen"] is not None
                      and row["trigger_4cand"]["on_chosen"] is not None
                      and row["trigger_4cand"]["off_chosen"] != row["trigger_4cand"]["on_chosen"])

    row["diagnostics"] = {
        "on_spread_2cand": on_spread_2, "lesion_spread_2cand": les_spread_2,
        "on_spread_4cand": on_spread_4, "lesion_spread_4cand": les_spread_4,
        "spread_attributable_2cand": attrib_2, "spread_attributable_4cand": attrib_4,
        "reorders_on_4cand": reorders_4,
    }
    row["c_ordinary_preserved"] = bool(ordinary_identical)
    row["c_on_loadbearing"] = bool(ctx_differs_2 and ctx_differs_4)
    row["c_lesion_collapses"] = bool(les_spread_2 < 0.5 * max(on_spread_2, 1e-9)
                                     and les_spread_4 < 0.5 * max(on_spread_4, 1e-9))
    row["c_moat_holds"] = bool(moat_holds)
    row["seed_pass"] = bool(row["c_ordinary_preserved"] and row["c_on_loadbearing"]
                            and row["c_lesion_collapses"] and row["c_moat_holds"])
    row["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[flip-soak seed={seed}] ordinary_identical={ordinary_identical} "
          f"2cand off/on/lesion={row['trigger_2cand']['off_chosen']!r}/{row['trigger_2cand']['on_chosen']!r}/"
          f"{row['trigger_2cand']['lesion_chosen']!r} "
          f"4cand off/on/lesion={row['trigger_4cand']['off_chosen']!r}/{row['trigger_4cand']['on_chosen']!r}/"
          f"{row['trigger_4cand']['lesion_chosen']!r} reorders_4cand={reorders_4} "
          f"seed_pass={row['seed_pass']} ({row['elapsed_s']}s, build {build_s:.1f}s)", flush=True)
    return row


def decide(rows):
    n = len(rows)
    ord_pass = sum(1 for r in rows if r["c_ordinary_preserved"])
    lb_pass = sum(1 for r in rows if r["c_on_loadbearing"])
    les_pass = sum(1 for r in rows if r["c_lesion_collapses"])
    moat_pass = sum(1 for r in rows if r["c_moat_holds"])
    all_pass = sum(1 for r in rows if r["seed_pass"])
    n_reorders = sum(1 for r in rows if r["diagnostics"]["reorders_on_4cand"])
    verdict = "GO" if all_pass == n else "NO-GO"
    return verdict, {"n_seeds": n, "ordinary_preserved_pass": ord_pass, "on_loadbearing_pass": lb_pass,
                     "lesion_collapses_pass": les_pass, "moat_holds_pass": moat_pass, "seed_pass": all_pass,
                     "n_seeds_with_4cand_reorder": n_reorders}


def smoke():
    """Fast harness check (MOCK chat + MOCK organ; no bridge build): the panel logic + flag toggling + the
    diagnostic-context plumbing + the verdict aggregator are well-formed."""
    import research.runners.shared_salience_afferent as SH
    from research.runners import value_choice_production_organ as VC

    class _Router:
        self_aliases = {"brain", "you", "i", "me"}

    class _Agent:
        def held_referent(self):
            return (None, None)

    class _Chat:
        def __init__(self):
            self.stored_facts = [("dog", "chase", "cat"), ("cat", "eat", "fish"),
                                 ("dog", "chase", "ball"), ("dog", "chase", "shoe"), ("dog", "chase", "stick"),
                                 ("bird", "eat", "worm"), ("bird", "eat", "seed")]
            self.agents_set = {"dog", "cat", "bird"}
            self.actions_set = {"chase", "eat"}
            self.router = _Router()
            self.is_multiturn = True
            self.agent = _Agent()

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
                return None, {"lesion": True, "fed_spread_hz": 0.0, "decisive": False}
            i = int(max(range(len(cands)), key=lambda k: engagements[k]))
            spread = max(engagements) - min(engagements)
            return cands[i], {"wta_choice": i, "fed_spread_hz": spread, "decisive": True}

    chat = _Chat()
    VC.install_value_choice(chat, organ=_Organ())
    _clear_flags()
    off_ord = [_answer(chat, q) for q in ORDINARY_TURNS]
    _set_flags(on=True)
    on_ord = [_answer(chat, q) for q in ORDINARY_TURNS]
    _clear_flags()
    ord_ok = (off_ord == on_ord)
    ctx_off = _diag_context(chat, VC, "dog", "chase", CANDS_4)
    _set_flags(on=True)
    ctx_on = _diag_context(chat, VC, "dog", "chase", CANDS_4)
    _set_flags(on=True, lesion=True)
    ctx_les = _diag_context(chat, VC, "dog", "chase", CANDS_4)
    _clear_flags()
    differs = any(abs(x - y) > 1e-9 for x, y in zip(ctx_off, ctx_on))
    spread_on = max(ctx_on) - min(ctx_on)
    spread_les = max(ctx_les) - min(ctx_les)
    collapses = spread_les < 0.5 * max(spread_on, 1e-9)
    v, d = decide([{"c_ordinary_preserved": True, "c_on_loadbearing": True, "c_lesion_collapses": True,
                    "c_moat_holds": True, "seed_pass": True, "diagnostics": {"reorders_on_4cand": False}}])
    ok = ord_ok and differs and collapses and (v == "GO")
    print(f"[flip-soak SMOKE] ordinary_identical={ord_ok} ctx_off={ctx_off} ctx_on={ctx_on} ctx_lesion={ctx_les}")
    print(f"[flip-soak SMOKE] on_differs_from_off={differs} lesion_collapses={collapses}")
    print(f"[flip-soak SMOKE] verdict-aggregator GO={v == 'GO'}")
    print(f"[flip-soak SMOKE] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="BRAIN_SHARED_SALIENCE DEFAULT-ON flip soak, value-choice consumer "
                                             "(no-regression + anti-hollow), through the REAL ChatBrain.")
    ap.add_argument("--smoke", action="store_true", help="fast mock harness check (no bridge build)")
    ap.add_argument("--seed", type=int, default=None, help="single-seed worker mode")
    ap.add_argument("--seeds", default=None, help="controller mode: comma/space-fanned seeds, e.g. '42,43,44'")
    ap.add_argument("--composer", default="onebrain",
                    help="tiny-demo recall composer ('onebrain' production default, 'rf' for the fast numpy path)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)

    if a.smoke:
        raise SystemExit(0 if smoke() else 1)

    if a.seeds is not None:
        # CONTROLLER: subprocess-fan one worker per seed (process isolation -- the trained critic + the
        # process-shared curiosity-organ singleton are not safely re-buildable across seeds in one process).
        seeds = [int(s) for s in a.seeds.replace(",", " ").split()]
        per_seed = {}
        for s in seeds:
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, "-m", "research.runners._shared_salience_flip_soak", "--seed", str(s),
                 "--composer", a.composer],
                cwd=str(_REPO), capture_output=True, text=True, timeout=900,
                env={**os.environ, "SIM_NO_PROVENANCE": "1"},
            )
            if r.returncode != 0:
                per_seed[str(s)] = {"seed": s, "error": (r.stderr[-4000:] or r.stdout[-4000:]),
                                    "returncode": r.returncode}
                print(f"  seed {s}: ERROR rc={r.returncode}\n{(r.stderr or r.stdout)[-2000:]}", flush=True)
                continue
            line = None
            for ln in r.stdout.splitlines():
                if ln.startswith("RESULT_JSON:"):
                    line = ln[len("RESULT_JSON:"):]
            per_seed[str(s)] = json.loads(line) if line else {"seed": s, "error": "no RESULT_JSON line",
                                                               "stdout_tail": r.stdout[-2000:]}
            per_seed[str(s)]["wall_seconds"] = round(time.time() - t0, 2)
            print(f"  seed {s}: seed_pass={per_seed[str(s)].get('seed_pass')} "
                  f"({per_seed[str(s)]['wall_seconds']}s)", flush=True)
        n = len(seeds)

        def _g(k):
            return sum(1 for s in seeds if per_seed.get(str(s), {}).get(k))

        n_pass = _g("seed_pass")
        all_pass = (n_pass == n)
        n_ord = _g("c_ordinary_preserved")
        n_lb = _g("c_on_loadbearing")
        n_les = _g("c_lesion_collapses")
        n_moat = _g("c_moat_holds")
        n_reorders = sum(1 for s in seeds
                         if per_seed.get(str(s), {}).get("diagnostics", {}).get("reorders_on_4cand"))
        verdict = "GO" if all_pass else "NO-GO"
        # a verdict must carry what earned it (tools/gates/verdict_preconditions.py) -- the four per-seed gates,
        # checked at the CONTROLLER level across all seeds, become the artifact's `preconditions` block (mirrors
        # _value_choice_neural_context_6seed_derisk.py's own controller-level Verdict).
        V = Verdict("shared_salience_flip_soak")
        V.require("ordinary turns byte-identical OFF-vs-ON (all seeds)", n_ord == n)
        V.require("on-loadbearing: fed engagement context measurably differs OFF-vs-ON on both trigger "
                  "scenarios (all seeds)", n_lb == n)
        V.require("lesion-collapses: BRAIN_SHARED_SALIENCE_LESION collapses the ON-arm spread on both "
                  "scenarios (all seeds)", n_les == n)
        V.require("no-confab moat: every commit across every arm is None or a stored candidate (all seeds)",
                  n_moat == n)
        V.require("every subprocess worker returned a RESULT_JSON (none crashed/timed out)",
                  all("error" not in per_seed.get(str(s), {}) for s in seeds))
        decided = V.decide(go=(verdict == "GO"), verbose=False)
        result = {"mode": "controller", "probe": "shared_salience_flip_soak", "verdict": decided["status"],
                  "seeds": seeds, "n_seeds": n, "n_pass": n_pass, "all_seeds_pass": bool(all_pass),
                  "ordinary_preserved_pass": n_ord, "on_loadbearing_pass": n_lb, "lesion_collapses_pass": n_les,
                  "moat_holds_pass": n_moat, "n_seeds_with_4cand_reorder": n_reorders,
                  "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
                  "per_seed": per_seed}
    elif a.seed is not None:
        row = run_seed(a.seed, a)
        print("RESULT_JSON:" + json.dumps(row, default=str))
        result = row
    else:
        ap.error("pass --seed N (worker), --seeds N,N,N (controller), or --smoke")
        return

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"wrote {a.out}")
    if a.seed is None:
        print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2, default=str))


if __name__ == "__main__":
    main()
