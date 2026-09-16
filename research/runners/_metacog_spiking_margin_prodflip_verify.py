"""PRODUCTION-FLIP verification (ship-the-validated-wins campaign, Track 1) for the metacog spiking
recall-margin mechanism (scaffold-retirement backlog rank 9, `BRAIN_METACOG_SPIKING_MARGIN`).

The underlying mechanism is already 6-seed de-risked at the COMPOSER level
(`research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md`, `OneBrainComposer` built
directly, bypassing the full production organ stack -- documented there as a deliberate ~40x-faster scope
trade-off). THIS script re-verifies at the INTEGRATED level: the real `webapp.server` conversational handler
(`_build_tiny_demo` -> `ChatBrain` -> `S.brain_chat`, the SAME code the `/api/brain-chat` endpoint runs) with
the TRUE production default config for every OTHER faculty (BRAIN_AFFECT / BRAIN_WORLDMODEL / BRAIN_SURPRISE /
BRAIN_COMPREHENSION_GATE / BRAIN_CURIOSITY / BRAIN_MULTIREF / ... all left UNSET, i.e. at their real default-ON
state -- the opposite of the composer-level de-risk's own calibration script, which zeroed ~24 of them for
isolation). Answers the three production-flip questions:

  1. NO REGRESSION: with the flag ON, does the brain still converse (no crash/hang) through the real handler,
     do the OTHER faculties still populate their response fields (affect/worldmodel/surprise/comprehension/
     curiosity/multiref), and is the answer's SUBSTANTIVE content (recalled_svo, the answer text minus any
     hedge/prefix) preserved relative to the shipped default-OFF behaviour on the SAME real turns?
  2. LOAD-BEARING NOT HOLLOW, tested at the conversational surface (not just the composer's internal number):
     does VARYING recall difficulty (clean vs synthetic-noise-degraded, mirroring the composer-level de-risk's
     own noise sweep) change the LIVE resp["metacog"]["confident"] / the hedge prefix actually prepended to
     resp["answer"]? Does an INTEGRATED LESION of the recall circuit's own spiking discrimination (forcing
     `RFPhasorComposer._spiking_margin`'s lesion path DURING a live `S.brain_chat` call) collapse that SAME
     live decision back to a hedge, on a turn that was confident with the lesion off -- while the decoded
     content (`recalled_svo`) stays byte-identical (confirming the mechanism is trace-only at the INTEGRATED
     level too, not just by the composer docstring's claim)?
  3. Quantifies the DIRECTION of any ON-vs-OFF disagreement in the previously-characterized ambiguous middle
     band: does flag-ON ever read CONFIDENT on a turn the shipped default would have HEDGED (the false-
     confidence direction -- the failure mode that matters for the honesty-boundary mission), or only the
     reverse (extra hedging -- conservative, not the same kind of regression)?

METHOD (isolates the ONE variable under test, no seed/build confound -- the SAME design principle the
composer-level de-risk used for its own comparison): for each of the 6 mandated seeds, builds ONE real
chat-brain via `_build_tiny_demo(seed, ..., composer_kind="onebrain")` + `ChatBrain(...)` with
`BRAIN_METACOG_SPIKING_MARGIN=1` for the whole process (every role chip then carries BOTH the host margin
fields -- ALWAYS computed regardless of the flag -- and the additive `margin_spiking`). For each turn:
  (a) reads the REAL `resp["metacog"]` / `resp["answer"]` the flag-ON config actually produces live (the
      candidate production behaviour), and
  (b) independently recomputes the counterfactual flag-OFF verdict from the SAME `resp["activity"]` trace via
      the pre-2026-09-05 host-only preference chain, fed through the SAME production `MetacogProductionOrgan`
      singleton (`get_organ()` -- stateless per read past its one-time threshold calibration; verified by
      inspection of `nmda_norm_margin`, which `_restore_state()`s before every trial) -- a literal, turn-matched
      ON-vs-OFF comparison with NO separate-build noise.
A SEPARATE fresh-build cross-check (seed 42 only, two literal builds -- flag unset vs flag=1, no LTM) confirms
this same-trace-counterfactual method agrees with an actually-separate flag-OFF build on matched queries
(closes the loop on realism).

SCOPE (documented, not silent): the sweep does NOT attach the shipped 100k-fact LTM
(`BRAIN_LTM_SHIP_DEFAULT=off`) -- confirmed separately (this script's `--ltm-smoke`) to take >7 minutes to even
finish ONE build (vs ~180s without), an order of magnitude beyond what a 6-seed x multi-condition sweep can
afford, AND orthogonal to this specific flag by construction: `margin_spiking` populates identically on
whichever composer a query resolves through (`OneBrainComposer._block_role_scores` for the buffer tier,
`RFPhasorComposer._cleanup_all_score_stats` for the `ShardedPhasorStore`-backed LTM tier -- both gated by the
SAME `spiking_recall_margin` flag, confirmed by inspection), and every query here targets the tiny-demo's
buffer-native facts (answered by the buffer tier directly, never falling through to LTM) -- so the LTM tier's
own internal behaviour is not exercised either way. `--ltm-smoke` (optional, run separately) does ONE flag-ON
build WITH the LTM attached purely as a does-it-crash check in the heaviest real config.

Usage:
    python -m research.runners._metacog_spiking_margin_prodflip_verify --out <path.json>
    python -m research.runners._metacog_spiking_margin_prodflip_verify --ltm-smoke --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
# scope decision (see module docstring): skip the shipped LTM attach for the main sweep -- orthogonal to this
# flag, and >7x the per-build cost. `--ltm-smoke` overrides this back to the true (unset) production default.
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "off")
# the flag under test: ON for the whole process (matches the composer-level de-risk's own design) so a SINGLE
# built brain per seed carries BOTH the host fields (always computed) AND margin_spiking (additive) -- isolates
# the flag's causal effect with no separate-build confound. Deliberately NOT touching any other BRAIN_* flag:
# this run exercises the TRUE production default stack (every other faculty at ITS real default) for the
# "every other faculty stays alive" check -- the opposite of the composer de-risk's own isolation env.
os.environ["BRAIN_METACOG_SPIKING_MARGIN"] = "1"

import numpy as np

SEEDS = [42, 43, 44, 100, 101, 102]
SIGMAS = [0.3, 0.9, 1.5, 2.0, 2.5, 3.0, 4.0]   # spans clean -> ambiguous middle band -> clearly-degraded
QUERY = "what does the brain use"               # matches calibrate_margin.py's own real-handler query
OTHER_FACULTY_KEYS = ["affect", "worldmodel", "surprise", "comprehension", "curiosity", "multiref"]


def _host_mrc(activity):
    """The pre-2026-09-05 host-only preference chain -- mirrors `mean_role_confidence` but NEVER reads
    `margin_spiking` even when present. The counterfactual flag-OFF arm, read off the SAME trace."""
    roles = (activity or {}).get("roles") or []
    vals = []
    for r in roles:
        snr = r.get("margin_snr")
        mn = r.get("margin_norm")
        m = r.get("margin")
        v = snr if snr is not None else (mn if mn is not None else (m if m is not None else r.get("confidence")))
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _judge_counterfactual_off(MC, activity):
    """Recompute what the shipped (flag-OFF) metacog judge() would have read on this SAME real turn's trace."""
    mrc = _host_mrc(activity)
    ev = MC.evidence_from_role_conf(mrc)
    if ev is None:
        return None
    return MC.get_organ().judge(ev, lesion=MC.metacog_lesioned())


def _strip_hedge(MC, answer):
    hp = MC.hedge_prefix()
    return answer[len(hp):] if isinstance(answer, str) and answer.startswith(hp) else answer


def _build_seed_brain(seed):
    from research.runners.brain_chat_tui import _build_tiny_demo, ChatBrain, StubRenderer
    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat


def _ask(S, chat, brain_label, sid_holder, message):
    """BUG FOUND + FIXED IN-FLIGHT (2026-09-05, this same verification session): `sid_holder` used to reset to
    [0] at the top of EACH `run_seed_sweep` call, so seed 42's turns and seed 43's turns (etc.) reused the
    IDENTICAL session strings ('pf00001'..'pf00009') -- and webapp/server.py's module-level session caches
    (`_BRAIN_RICH: dict[(session,brain,renderer), RichAnswerComposer]`, `_SESSION_DISCOURSE`, ...) are keyed on
    exactly that tuple. `_get_rich_composer` does `rich = _BRAIN_RICH.get(cache_key); if rich is None: build ...`
    -- it NEVER checks whether the cached rich composer's OWN `chat` matches the `chat` just passed in. So every
    seed AFTER the first silently reused the FIRST seed's RichAnswerComposer (built around the first seed's
    OneBrainComposer instance), while the query dict-lookups in `S.brain_chat` ran against the (correct) NEW
    seed's `chat`/composer -- a state mismatch that manifested as every post-first-seed turn reading
    `activity=None`/an empty metacog (confirmed: an ISOLATED single-seed-43 diagnostic, sharing no prior
    session history, answered correctly with metacog confident=True; only the multi-seed IN-PROCESS sweep with
    colliding session ids failed). FIX: `sid_holder` now carries a GLOBALLY-unique prefix (passed in, not
    reset per seed) so no two turns in the whole run -- across any seed -- ever share a session key, matching
    how a real multi-user production deployment never reuses a session id either."""
    from webapp.server import BrainChatRequest
    prefix, counter = sid_holder
    counter[0] += 1
    sess = f"{prefix}{counter[0]:05d}"
    S._BRAIN_CHATS[(sess, brain_label, "stub")] = chat
    req = BrainChatRequest(session=sess, message=message, brain=brain_label, reset=False, rich=True,
                           renderer="stub")
    r = S.brain_chat(req)
    return json.loads(bytes(r.body))


def _other_faculty_snapshot(resp):
    """Which OTHER_FACULTY_KEYS are present + non-error on this turn (not requiring non-null -- some are
    legitimately null out-of-scope for a given turn shape; requiring instead that the key exists in the
    response schema and, if a dict, carries no 'error' key -- the code's own crash-degradation signature)."""
    out = {}
    for k in OTHER_FACULTY_KEYS:
        present = k in resp
        v = resp.get(k)
        errored = isinstance(v, dict) and "error" in v
        out[k] = {"present_in_schema": present, "non_null": v is not None, "errored": errored}
    return out


def run_seed_sweep(seed, results):
    import webapp.server as S
    import research.runners.rf_phasor_composer as RFP
    from research.runners._emergent_graceful_degradation_derisk import _noise
    import research.runners.metacog_production_organ as MC

    brain_label = "tiny-demo"
    t0 = time.time()
    chat = _build_seed_brain(seed)
    build_s = time.time() - t0
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    base_conns = list(comp.store_conns)
    # (prefix, counter) -- the prefix MUST be unique per seed (see _ask's docstring: a bug found + fixed in this
    # same session had this reset to a bare [0] per seed, so every seed's turns silently REUSED the identical
    # session ids 'pf00001'..'pf00009' -- and webapp/server.py's module-level _BRAIN_RICH/_SESSION_DISCOURSE
    # caches are keyed on (session,brain,renderer), so every seed after the first ran against the FIRST seed's
    # stale cached RichAnswerComposer instead of its own).
    sid = (f"pf{seed}_", [0])
    rng = np.random.default_rng(seed)

    turns = []

    def do_ask(label, conns=None, sigma=None):
        comp.store_conns = list(conns) if conns is not None else list(base_conns)
        try:
            resp = _ask(S, chat, brain_label, sid, QUERY)
        finally:
            comp.store_conns = list(base_conns)
        activity = resp.get("activity")
        real_mc = resp.get("metacog")
        cf_off = _judge_counterfactual_off(MC, activity)
        roles = (activity or {}).get("roles") or []
        n_spiking = sum(1 for r in roles if r.get("margin_spiking") is not None)
        rec = {
            "label": label, "sigma": sigma, "abstained": resp.get("abstained"),
            "recalled_svo": resp.get("recalled_svo"),
            "answer_stripped": _strip_hedge(MC, resp.get("answer")),
            "answer_had_hedge": (resp.get("answer") != _strip_hedge(MC, resp.get("answer"))),
            "n_roles": len(roles), "n_roles_with_margin_spiking": n_spiking,
            "real_ON_confident": (real_mc or {}).get("confident"),
            "real_ON_balance": (real_mc or {}).get("balance"),
            "real_ON_mean_role_conf": (real_mc or {}).get("mean_role_conf"),
            "cf_OFF_confident": (cf_off or {}).get("confident"),
            "cf_OFF_balance": (cf_off or {}).get("balance"),
            "cf_OFF_mean_role_conf": _host_mrc(activity),
            "other_faculties": _other_faculty_snapshot(resp),
            "confidence_forthcoming": resp.get("confidence_forthcoming"),
        }
        turns.append(rec)
        return resp, rec

    # 1. clean
    resp_clean, rec_clean = do_ask("clean")

    # 2. noise sweep (ambiguous-middle-band probe)
    for sigma in SIGMAS:
        noised = _noise(base_conns, sigma, np.random.default_rng(seed * 1000 + int(sigma * 10)))
        do_ask(f"sigma{sigma}", conns=noised, sigma=sigma)

    # 3. INTEGRATED lesion, on the SAME clean query: force RFPhasorComposer._spiking_margin's lesion path
    # DURING a live brain_chat call (not a standalone unit call) -- the vary+lesion test at the conversational
    # surface itself, per the task's explicit "test it explicitly" requirement.
    orig_fn = RFP.RFPhasorComposer._spiking_margin

    def _forced_lesion(self, scores, lesion=False):
        return orig_fn(self, scores, lesion=True)   # ignore caller's lesion arg -- always lesioned

    RFP.RFPhasorComposer._spiking_margin = _forced_lesion
    try:
        resp_lesioned, rec_lesioned = do_ask("clean_LESIONED")
    finally:
        RFP.RFPhasorComposer._spiking_margin = orig_fn

    lesion_check = {
        "clean_real_ON_confident": rec_clean["real_ON_confident"],
        "lesioned_real_ON_confident": rec_lesioned["real_ON_confident"],
        "collapsed_to_hedge": (rec_clean["real_ON_confident"] is True
                               and rec_lesioned["real_ON_confident"] is False),
        "recalled_svo_unchanged": (rec_clean["recalled_svo"] == rec_lesioned["recalled_svo"]),
        "clean_balance": rec_clean["real_ON_balance"], "lesioned_balance": rec_lesioned["real_ON_balance"],
    }

    # false-confidence-direction check across the sweep: ON confident while counterfactual-OFF would hedge.
    # Scoped to turns BOTH arms actually evaluated (real_ON_confident/cf_OFF_confident both real booleans, not
    # None) -- an abstained turn reads metacog=null on the REAL arm by construction (_metacog_qualify skips a
    # no-answer turn regardless of confidence), so `None != True/False` there is an abstain artifact, not a
    # genuine confident-vs-hedge disagreement; counting it would be exactly the "UNDEFINED, not a score" trap
    # (tools/lab.py's own undefined_if_empty discipline) applied to the wrong axis.
    evaluable = [t for t in turns if isinstance(t["real_ON_confident"], bool)
                and isinstance(t["cf_OFF_confident"], bool)]
    n_abstained_or_unevaluable = len(turns) - len(evaluable)
    false_confidence_turns = [t["label"] for t in evaluable
                              if t["real_ON_confident"] is True and t["cf_OFF_confident"] is False]
    extra_hedge_turns = [t["label"] for t in evaluable
                        if t["real_ON_confident"] is False and t["cf_OFF_confident"] is True]
    disagreement_turns = [t["label"] for t in evaluable
                         if t["real_ON_confident"] != t["cf_OFF_confident"]]

    results[str(seed)] = {
        "build_seconds": build_s, "source_label": brain_label, "n_turns": len(turns), "turns": turns,
        "n_evaluable_turns": len(evaluable), "n_abstained_or_unevaluable": n_abstained_or_unevaluable,
        "lesion_check": lesion_check,
        "on_vs_off_disagreement_turns": disagreement_turns,
        "false_confidence_direction_turns": false_confidence_turns,
        "extra_hedge_direction_turns": extra_hedge_turns,
        "content_check": {
            "clean_recalled_svo": rec_clean["recalled_svo"],
            "clean_answer_stripped": rec_clean["answer_stripped"],
        },
    }
    print(f"[seed {seed}] build={build_s:.1f}s turns={len(turns)} evaluable={len(evaluable)} "
          f"abstained/unevaluable={n_abstained_or_unevaluable} "
          f"disagreements={len(disagreement_turns)} false_conf_dir={len(false_confidence_turns)} "
          f"extra_hedge_dir={len(extra_hedge_turns)} lesion_collapsed={lesion_check['collapsed_to_hedge']}",
          flush=True)


def run_seed42_fresh_build_crosscheck(results):
    """A SEPARATE, literal fresh-build pair at seed 42 -- no shared process/env state -- confirming the main
    sweep's same-trace-counterfactual method agrees with an ACTUALLY-separate build on the SAME query (closes
    the loop on realism per the module docstring). Shipped-default semantics (the flag stays default-OFF: see
    the 2026-09-05 production-flip NO-GO finding): UNSET is the TRUE shipped default (OFF); the explicit
    `BRAIN_METACOG_SPIKING_MARGIN=1` is the candidate this whole script evaluates -- so 'OFF' below relies on
    unset (asserting the real default is correct, gates/flip-offarm-staleness's own sanctioned pattern for a
    flag that legitimately still defaults off) and 'ON' sets the literal explicitly."""
    import subprocess
    out = {}
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_metacog_spiking_margin_prodflip_verify.py")
    for cond, env_val in (("OFF", None), ("ON", "1")):
        env = dict(os.environ)
        env["BRAIN_LTM_SHIP_DEFAULT"] = "off"
        if env_val is None:
            env.pop("BRAIN_METACOG_SPIKING_MARGIN", None)   # unset -> the TRUE shipped default (OFF)
        else:
            env["BRAIN_METACOG_SPIKING_MARGIN"] = env_val
        code = (
            "import json,sys; sys.path.insert(0, %r)\n"
            "from research.runners.brain_chat_tui import _build_tiny_demo, ChatBrain, StubRenderer\n"
            "import webapp.server as S\n"
            "from webapp.server import BrainChatRequest\n"
            "import research.runners.metacog_production_organ as MC\n"
            "agent, aliases, _n = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind='onebrain')\n"
            "chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())\n"
            "S._BRAIN_CHATS[('xc','tiny-demo','stub')] = chat\n"
            "r = S.brain_chat(BrainChatRequest(session='xc', message=%r, brain='tiny-demo', reset=False, rich=True, renderer='stub'))\n"
            "resp = json.loads(bytes(r.body))\n"
            "print('XCRESULT_JSON:' + json.dumps({'answer': resp.get('answer'), 'recalled_svo': resp.get('recalled_svo'),"
            " 'metacog': resp.get('metacog'), 'abstained': resp.get('abstained')}))\n"
        ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), QUERY)
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                              timeout=900)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("XCRESULT_JSON:")), None)
        out[cond] = {
            "returncode": proc.returncode,
            "parsed": (json.loads(line[len("XCRESULT_JSON:"):]) if line else None),
            "stderr_tail": proc.stderr[-2000:] if proc.returncode != 0 else None,
        }
        print(f"[seed42 crosscheck {cond}] returncode={proc.returncode} "
              f"parsed={'yes' if line else 'NO -- see stderr_tail'}", flush=True)
    a, b = out.get("OFF", {}).get("parsed"), out.get("ON", {}).get("parsed")
    content_match = None
    if a and b:
        import research.runners.metacog_production_organ as MC
        content_match = (_strip_hedge(MC, a.get("answer")) == _strip_hedge(MC, b.get("answer"))
                         and a.get("recalled_svo") == b.get("recalled_svo"))
    results["_seed42_fresh_build_crosscheck"] = {
        "OFF": out["OFF"], "ON": out["ON"],
        "content_preserved_across_literal_builds": content_match,
    }


def run_ltm_smoke(results):
    """Optional: ONE flag-ON build WITH the shipped LTM attached (true, unmodified production default env) --
    a does-it-crash smoke test in the heaviest real config, not a full comparison (see module docstring)."""
    import webapp.server as S
    os.environ.pop("BRAIN_LTM_SHIP_DEFAULT", None)   # restore the TRUE production default (LTM attach ON)
    t0 = time.time()
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    build_s = time.time() - t0
    resp = _ask(S, chat, "tiny-demo", ("pfltm_", [0]), QUERY)
    results["_ltm_smoke"] = {
        "build_seconds": build_s, "source": source, "abstained": resp.get("abstained"),
        "recalled_svo": resp.get("recalled_svo"), "metacog": resp.get("metacog"),
        "other_faculties": _other_faculty_snapshot(resp),
    }
    print(f"[ltm-smoke] build={build_s:.1f}s source={source} abstained={resp.get('abstained')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ltm-smoke", action="store_true", help="also run the LTM-attached smoke build")
    ap.add_argument("--skip-crosscheck", action="store_true", help="skip the seed-42 fresh-build crosscheck")
    args = ap.parse_args()

    results = {"seeds": SEEDS, "sigmas": SIGMAS, "query": QUERY}
    for seed in SEEDS:
        run_seed_sweep(seed, results)
    if not args.skip_crosscheck:
        run_seed42_fresh_build_crosscheck(results)
    if args.ltm_smoke:
        run_ltm_smoke(results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
