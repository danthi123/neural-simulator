"""Scaffold-retirement backlog RANK-2 de-risk: thread `integrated_loop=True` through
webapp/server.py -> brain_chat_tui / developed_brain_io -> MultiTurnAgent -> BrainConversationalAgent ->
OneBrainComposer, so the spiking K-way cue-match SELECTION (already GO 4/4 at V=320,
2026-06-21-shortcut3-fold-integrated-loop-BUILD.md) can replace the host string-`==` first-match `_scan`
under the real webapp production construction, not just a bare hand-built `OneBrainComposer`.

WHAT WAS ALREADY TRUE (verified against current code, not re-derived): `OneBrainComposer(integrated_loop=True)`
and `BrainConversationalAgent(..., integrated_loop=...)` already existed (the 2026-06-21 #3 fold) and were
already GO 4/4 (K in {2,4,8,32}, V in {72,320}, multi-seed at V=72 / seed-42 at V=320). What did NOT exist,
confirmed by grep before this session touched anything: `webapp/server.py` had ZERO references to
`integrated_loop` anywhere -- `MultiTurnAgent.__init__` had no `integrated_loop` parameter at all, and
neither `brain_chat_tui._build_tiny_demo` nor `developed_brain_io.load_developed_brain` threaded one through.
The production entry point (`webapp.server._build_chat_brain`, which wraps the composer in a
`MultiTurnAgent` with the discourse-WM loop, the D3 event register, biased-competition wiring, the LTM
attach, and (via `brain_reply`) the GNW ignition bus -- the "~15 live faculties") had NO way to opt into the
sequencer at all. This runner de-risks the NEWLY-THREADED plumbing itself (all additive, default OFF):

  PART A (mechanical thread-check, numpy-CPU, GPU-free, seconds): builds a REAL `ChatBrain` via
  `webapp.server._build_chat_brain('tiny-demo', 'stub')` -- the actual production entry point, carrying the
  same LTM/discourse/biased-competition wiring a live turn does -- with `BRAIN_INTEGRATED_LOOP` unset vs "1",
  and confirms (a) unset is BYTE-IDENTICAL (`composer.integrated_loop is False`, the tiny 5-fact battery
  answers exactly as `_build_tiny_demo`'s docstring says) and (b) the env var genuinely reaches the
  `OneBrainComposer` instance THROUGH the full chain (`composer.integrated_loop is True`) with the no-confab
  moat still 0-false-accept even at this small, over-abstention-prone vocab (the documented SAFE-direction
  boundary, `_burndown_1A_c2_smallvocab_derisk.json` -- over-abstention is not a moat breach).

  PART B (production-scale GO, GPU/cupy, V=320/K=32, 6 seeds 42/43/44/100/101/102): extends the 2026-06-21
  gate's 1-seed V=320 CONFIRMATION (a bare `OneBrainComposer` built by hand) to 6 seeds, AND wraps it in the
  SAME `MultiTurnAgent` shape `load_developed_brain(use_multiturn=True)` / `webapp._build_chat_brain` always
  construct (not a bare composer) -- i.e. exercises the NEW plumbing (`MultiTurnAgent.integrated_loop`,
  added this session) end to end at the validated production vocab tier, not a repeat of the composer-API
  test that already existed. Reuses the validated K=32 fact set + V=320 padding recipe (by import, NO
  reimplementation) from `_phaseB_onebrain_sequencerK_k32_margin_derisk` / `_phaseB_onebrain_integrated_loop_
  fold_derisk` (including its memory-safe per-seed teardown -- the SAME accumulation bug applies here: an
  O(K*V) sequencer fabric per composer, two composers per seed, six seeds in one process).

GATES (GO): PART A -- off byte-identical + on reaches the composer + on moat 0-FA. PART B -- answer-identity
(`c_seq.<method>(...) == c_host.<method>(...)` for every present/absent/cross cue) AND moat 0-FA, at every one
of the 6 seeds. An honest NEGATIVE (a single mismatch or false-accept) is a valid deliverable -- report it,
never retune the op-point to mask it (op-point is OneBrainComposer's OWN validated default: match_thresh=0.06,
gain=0.11, sigma=1.0, input_gain=1.0 -- not overridden here).

NO `sim/` edit (reuse-by-import throughout).

  SIM_BACKEND=numpy python -u -m research.runners._rank2_integrated_loop_webapp_thread_derisk --skip-production \\
      --out research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partA.json
  SIM_BACKEND=cupy  python -u -m research.runners._rank2_integrated_loop_webapp_thread_derisk --skip-mechanical \\
      --out research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partB.json
"""
from __future__ import annotations

import argparse
import json
import os

from tools.verdict import Verdict

SEEDS = (42, 43, 44, 100, 101, 102)
K = 32


# ================================================================================================================
# PART A -- mechanical thread-check through the REAL webapp entry point (numpy-CPU, GPU-free).
# ================================================================================================================

def _build_tiny(env_value):
    """A fresh production `ChatBrain` via `webapp.server._build_chat_brain('tiny-demo', 'stub')` -- the ACTUAL
    entry point `/api/brain-chat` uses for the default (no explicit bundle) brain -- with `BRAIN_INTEGRATED_LOOP`
    set to `env_value` (None = unset = the default). `BRAIN_COMPOSER_KIND` is pinned to 'onebrain' (integrated_loop
    is a no-op on 'rf'/'rate'/'slotbinder', and this test is specifically about the onebrain branch).
    `BRAIN_LTM_SHIP_DEFAULT=off` skips the unrelated bulk-knowledge LTM attach (irrelevant to this fixture's facts,
    expensive to rebuild, no effect on the mechanism under test -- mirrors `_selfid_anaphora_scaffold_derisk.
    _build`'s own documented reasoning)."""
    if env_value is None:
        os.environ.pop("BRAIN_INTEGRATED_LOOP", None)
    else:
        os.environ["BRAIN_INTEGRATED_LOOP"] = env_value
    os.environ["BRAIN_COMPOSER_KIND"] = "onebrain"
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"
    from webapp.server import _build_chat_brain
    chat, source = _build_chat_brain("tiny-demo", "stub")
    return chat, chat.inner, source        # chat.inner IS the BrainConversationalAgent (ChatBrain unwraps it)


def part_a_mechanical_thread_check():
    v = Verdict("rank2-mechanical-thread-check")

    chat_off, inner_off, src_off = _build_tiny(None)
    # the production composer may be the bare OneBrainComposer OR (BRAIN_COMPOSER_MERGE default-on today)
    # Pool1BoundOneBrainComposer -- either way it must be an onebrain-family composer, not rf/rate/slotbinder.
    off_cls = type(inner_off.composer).__name__
    v.require("off_is_onebrain_family_composer", "OneBrainComposer" in off_cls, expect=True,
              note=f"class={off_cls}")
    v.require("off_integrated_loop_is_False", bool(getattr(inner_off.composer, "integrated_loop", None)),
              expect=False)

    # byte-identical regression pin: the tiny 5-fact battery answers exactly as _build_tiny_demo's own fixture
    # intends (facts: brain-use-spikes / brain-learn-words / brain-store-memory / dog-chase-cat / cat-eat-fish).
    off_answers = {
        "brain_use": inner_off.composer.query_patient("brain", "use"),
        "brain_learn": inner_off.composer.query_patient("brain", "learn"),
        "brain_store": inner_off.composer.query_patient("brain", "store"),
        "dog_chase": inner_off.composer.query_patient("dog", "chase"),
        "cat_eat": inner_off.composer.query_patient("cat", "eat"),
        "dog_eat_moat": inner_off.composer.query_patient("dog", "eat"),   # a real agent x a real action, never
                                                                          # stored as a pair -> must abstain
    }
    v.require("off_answers_correct", off_answers, expect=lambda a: (
        a["brain_use"] == "spikes" and a["brain_learn"] == "words" and a["brain_store"] == "memory"
        and a["dog_chase"] == "cat" and a["cat_eat"] == "fish" and a["dog_eat_moat"] is None))

    chat_on, inner_on, src_on = _build_tiny("1")
    on_cls = type(inner_on.composer).__name__
    v.require("on_is_onebrain_family_composer", "OneBrainComposer" in on_cls, expect=True, note=f"class={on_cls}")
    v.require("on_integrated_loop_is_True", bool(getattr(inner_on.composer, "integrated_loop", None)), expect=True)
    # MOAT (HARD, never traded): an absent/cross cue must never false-accept, even at this small,
    # over-abstention-prone vocab (the SAFE direction -- see _build_tiny_demo's docstring). NOTE: a query result
    # of None is a MEANINGFUL measured abstain, not "never measured" -- Verdict.require() treats a raw `None`
    # measured-value as the latter (its UNMEASURED sentinel), so compare the boolean, not the raw value.
    on_moat_probe = inner_on.composer.query_patient("dog", "eat")
    v.require("on_moat_holds", on_moat_probe is None, expect=True, note=f"query_patient(dog,eat)={on_moat_probe!r}")
    # a genuinely-stored cue should still answer correctly if it fires at all (report, don't require, since the
    # documented small-vocab margin boundary makes some present-cue over-abstention an honest possible outcome).
    on_present = {
        "brain_use": inner_on.composer.query_patient("brain", "use"),
        "dog_chase": inner_on.composer.query_patient("dog", "chase"),
    }

    go = (len(v.unmet) == 0 and len(v.unmeasured) == 0)
    result = v.decide(go=go)
    result["off_answers"] = off_answers
    result["on_moat_probe"] = on_moat_probe
    result["on_present_probe"] = on_present
    result["source_off"] = src_off
    result["source_on"] = src_on
    return result


# ================================================================================================================
# PART B -- production-scale (V=320, K=32) answer-identity + moat, 6 seeds, through the REAL MultiTurnAgent
# construction (GPU/cupy).
# ================================================================================================================

from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import ALL_FACTS, VOCAB, _build_queries  # noqa: E402
from research.runners._phaseB_onebrain_integrated_loop_fold_derisk import _free_gpu_memory  # noqa: E402


def _v320_vocab():
    """Pad the V=72 production K=32 vocab to 320 distinct words -- the SAME recipe (by import) the 2026-06-21
    320-scale gate used (`_phaseB_onebrain_integrated_loop_fold_derisk.py --vocab-320`); the fact words stay the
    first 72, the rest are filler so the composer must clear the winner over a REALISTIC codebook size."""
    extra = [f"w{i:03d}" for i in range(320 - len(VOCAB))]
    vocab = list(VOCAB) + extra
    assert len(set(vocab)) == 320, "320-scale vocab must be 320 distinct words"
    return vocab


def _build_multiturn_pair(seed, vocab, facts):
    """Two `MultiTurnAgent`s on the SAME facts/codes -- the SAME construction shape
    `load_developed_brain(use_multiturn=True)` / `webapp._build_chat_brain` always build (discourse-WM loop +
    biased-competition wiring genuinely present, `defer_planner=True` so neither is actually built unless a
    referent is written -- this battery never writes one, so the added cost over a bare composer is bookkeeping
    only). `integrated_loop=False` (the host oracle) vs `True` (the NEW plumbing, `MultiTurnAgent.integrated_loop`,
    threaded this session)."""
    from research.runners.multi_turn_agent import MultiTurnAgent
    actions = {x for (_a, x, _p) in facts}
    referents = [w for w in vocab if w not in actions]
    concepts = {w: None for w in vocab}
    common = dict(referent_concepts=referents, concepts=concepts, seed=seed, composer_kind="onebrain",
                  enable_neural_render=False, defer_planner=True,
                  wm_n=max(600, 2 * 40 * len(referents)))
    host = MultiTurnAgent(integrated_loop=False, **common)
    seq = MultiTurnAgent(integrated_loop=True, **common)
    for (a, x, p) in facts:
        host.agent.composer.store(a, x, p)
        seq.agent.composer.store(a, x, p)
    return host, seq


def run_seed(seed, vocab, facts):
    host, seq = _build_multiturn_pair(seed, vocab, facts)
    hc, sc = host.agent.composer, seq.agent.composer
    queries = _build_queries(facts)
    rows = []
    for (qa, qx), kind in queries:
        h_qp = hc.query_patient(qa, qx)
        s_qp = sc.query_patient(qa, qx)
        if kind.endswith("present"):
            patient = next(p for (a, x, p) in facts if a == qa and x == qx)
        else:
            patient = facts[0][2]
        h_yn = hc.ask_yes_no(qa, qx, patient)
        s_yn = sc.ask_yes_no(qa, qx, patient)
        eq = (h_qp == s_qp) and (h_yn == s_yn)
        rows.append(dict(cue=(qa, qx), kind=kind, host_patient=h_qp, seq_patient=s_qp,
                         host_yes_no=h_yn, seq_yes_no=s_yn, eq=eq))
    answer_identical = all(r["eq"] for r in rows)
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    fa = sum(1 for r in moat_rows if (r["seq_patient"] is not None) or (r["seq_yes_no"] != "unknown"))
    moat_ok = (fa == 0)
    result = dict(seed=seed, rows=rows, answer_identical=answer_identical, moat_ok=moat_ok, fa=fa)
    # MEMORY-SAFE teardown (2026-06-21 accumulation bug applies identically here: an O(K*V) sequencer fabric per
    # composer, two composers per seed, six seeds in one process -- see _free_gpu_memory's own docstring).
    del host, seq, hc, sc
    _free_gpu_memory()
    return result


def part_b_production_scale_6seed(seeds=SEEDS):
    from sim.backend import is_gpu_backend
    vocab = _v320_vocab()
    facts = ALL_FACTS[:K]
    print(f"PART B: V={len(vocab)} K={K} seeds={list(seeds)} gpu={is_gpu_backend()}", flush=True)
    results = []
    for s in seeds:
        r = run_seed(s, vocab, facts)
        results.append(r)
        ai = "==host" if r["answer_identical"] else "!=HOST"
        moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['fa']})"
        print(f"  seed {s}: {ai}  {moat}", flush=True)

    v = Verdict("rank2-production-v320-6seed")
    for r in results:
        v.require(f"seed{r['seed']}_answer_identical", r["answer_identical"], expect=True)
        v.require(f"seed{r['seed']}_moat_fa_zero", r["fa"], expect=0)
    go = all(r["answer_identical"] and r["moat_ok"] for r in results)
    result = v.decide(go=go)
    result["per_seed"] = results
    result["V"] = len(vocab)
    result["K"] = K
    result["seeds"] = list(seeds)
    return result


# ================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mechanical", action="store_true")
    ap.add_argument("--skip-production", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out = {}
    if not args.skip_mechanical:
        print("=== PART A: mechanical thread-check (webapp.server._build_chat_brain, tiny-demo) ===", flush=True)
        out["part_a_mechanical"] = part_a_mechanical_thread_check()
    if not args.skip_production:
        print(f"=== PART B: production-scale V=320/K={K}, {len(seeds)} seeds ===", flush=True)
        out["part_b_production"] = part_b_production_scale_6seed(seeds)

    go = all(v.get("go", False) for v in out.values()) if out else False
    out["overall_go"] = go
    print(f"\nOVERALL: {'GO' if go else 'NOT-GO'}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
