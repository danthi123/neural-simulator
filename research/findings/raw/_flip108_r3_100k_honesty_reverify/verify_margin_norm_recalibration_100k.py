"""R3 (board #108 cluster, 2026-09-02): re-verify confidence-forthcomingness's #94 discrimination check
(the one that gated its 15k default-ON flip, `research/findings/2026-09-01-confidence-forthcomingness-default-on-
flip-GO.md` -> `verify_margin_norm_recalibration_6seed.json`) against the BROADER `wikidata_100k` bundle
(78,857 facts, vocab 23,914) instead of the shipped `wikidata_core_15k` (15,000 facts, vocab 7,032).

WHY: `research/FAILURE_LOG.md` row 94 / `GAP_CLOSURE_MISSION.md`'s 2026-09-02 header note that
confidence-forthcomingness (#94) and source-provenance-honesty (#129) were validated ONLY against the 15k core,
never against the 100k bundle that board #108 wants to flip to -- an unverified interaction if the base bundle
scale changes under them. This is an otherwise-VERBATIM copy of
`research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration.py` with exactly two
changes: (1) `LTM_BUNDLE` points at `wikidata_100k`; (2) `build_chat()` passes `extra_kwargs={"enable_codebook_cache":
True, "enable_decode_escalation": True}` to `ShardedPhasorStore.load()`, matching `webapp/server.py`'s NEW
production default for ANY bundle scale after the board #108 R2 wiring fix (`_ltm_codebook_cache_on()` /
`_ltm_decode_escalation_on()`, both default-ON) -- this script builds its chat manually (bypassing
`_load_or_build_ltm_store`), so it must opt into the same flags explicitly to test what production will actually
run once R1's fix lands.

The target fact (`asimov_isaac employer university_of_boston`) and question are UNCHANGED and confirmed present,
verbatim, in the 100k bundle's facts.json (checked directly). `NOISE_SIGMA=0.12` is also UNCHANGED (NOT
re-calibrated for the 100k store's thinner margins) -- if the same jitter no longer produces a "real degraded
match" on this larger, denser-vocab store (e.g. it abstains instead), that is itself part of the honest result
this script exists to surface, not something to paper over by re-tuning the noise to force a pass.

Output goes to THIS directory (never overwrites the original 15k evidence file).

Run (queued via tools/gpu_queue.sh, NOT run concurrently with any other heavy brain-loading job):
  SIM_BACKEND=numpy .venv/bin/python \\
      research/findings/raw/_flip108_r3_100k_honesty_reverify/verify_margin_norm_recalibration_100k.py \\
      --seeds 42 43 44 100 101 102 \\
      --out research/findings/raw/_flip108_r3_100k_honesty_reverify/verify_margin_norm_recalibration_100k_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC",
           "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP",
           "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY",
           "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES",
           "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN",
           "BRAIN_BG_SELECT", "BRAIN_SILENT_WM", "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)                          # metacog stays default-ON (the confidence read)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)     # NEVER override -- the true floor
os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"
os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)             # default-ON
os.environ["BRAIN_CLAIM_MOAT"] = "0"                            # the documented escape for residual 1 (unrelated
                                                                  # vocab/grammar gap, see the 2026-09-01 finding)

import numpy as np                                                                   # noqa: E402

import webapp.server as S                                                            # noqa: E402
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo  # noqa: E402
from research.runners.developed_brain_io import _inner_agent                          # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore                        # noqa: E402
from research.runners.sharded_phasor_store import ShardedPhasorStore                  # noqa: E402

LTM_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k"   # board #108 R3: was wikidata_core_15k
Q = "who does asimov isaac work for?"
EXPECTED_SVO = ["asimov_isaac", "employer", "university_of_boston"]
NOISE_SIGMA = 0.12   # UNCHANGED from the 15k verify (see module docstring) -- not re-tuned for this store


def _real_kb_facts():
    with open(os.path.join(LTM_BUNDLE, "facts.json"), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in raw}


_KB_FACTS = _real_kb_facts()


def moat_ok(d):
    facts = d.get("supporting_facts") or []
    return all(tuple(f) in _KB_FACTS for f in facts)


def build_chat(seed):
    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    # board #108 R2: match webapp/server.py's new production default for the LTM store at ANY bundle scale
    # (_ltm_codebook_cache_on() / _ltm_decode_escalation_on(), both ON) -- this script builds the LTM manually so
    # it must opt in explicitly rather than inherit the server's defaults.
    ltm = ShardedPhasorStore.load(LTM_BUNDLE, extra_kwargs={"enable_codebook_cache": True,
                                                            "enable_decode_escalation": True})
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat, inner.composer


def _target_shard_index(store):
    sh = store.shard_for(EXPECTED_SVO[0])
    for i, (fact, _comp) in enumerate(sh.kb):
        if fact.get("agent") == EXPECTED_SVO[0] and fact.get("action") == EXPECTED_SVO[1]:
            return sh, i
    raise RuntimeError("target fact not found in its routed shard")


_sid = [0]


def ask(chat, session_prefix="s"):
    ck = (f"{session_prefix}{_sid[0]:04d}", "tiny-demo", "stub")
    _sid[0] += 1
    S._BRAIN_CHATS[ck] = chat
    r = S.brain_chat(S.BrainChatRequest(session=ck[0], message=Q, brain="tiny-demo",
                                        reset=False, rich=True, renderer="stub"))
    return json.loads(bytes(r.body))


def _extract(d):
    cf = d.get("confidence_forthcoming") or {}
    mc = d.get("metacog") or {}
    return {
        "recalled_svo": d.get("recalled_svo"), "abstained": d.get("abstained"),
        "n_sentences": d.get("n_sentences"), "confident": cf.get("confident"),
        "reason": cf.get("reason"), "mean_role_conf": mc.get("mean_role_conf"),
        "cf": cf, "moat_ok": moat_ok(d), "recall_correct": (d.get("recalled_svo") == EXPECTED_SVO),
    }


def run_seed(seed):
    out = {"seed": seed}
    chat, composer = build_chat(seed)
    sh, idx = _target_shard_index(composer.ltm)
    base_fact, base_comp = sh.kb[idx]
    base_comp = np.array(base_comp, copy=True)

    # --- (1) CLEAN turn (undamaged store) ------------------------------------------------------------------
    os.environ.pop("BRAIN_METACOG_LESION", None)
    d_clean = ask(chat, session_prefix=f"c{seed}")
    out["clean"] = _extract(d_clean)

    # --- (2) NOISE-DEGRADED turn (Gaussian phase jitter on the TARGET fact's composite, fixed seed) --------
    rng = np.random.default_rng(1000 + seed)
    sh.kb[idx] = (base_fact, base_comp + rng.normal(0.0, NOISE_SIGMA, size=base_comp.shape))
    d_degraded = ask(chat, session_prefix=f"d{seed}")
    out["degraded"] = _extract(d_degraded)
    sh.kb[idx] = (base_fact, base_comp)   # restore before any further query on this shard

    # --- (3) LESIONED turn (clean store, BRAIN_METACOG_LESION=1) -------------------------------------------
    os.environ["BRAIN_METACOG_LESION"] = "1"
    d_lesion = ask(chat, session_prefix=f"l{seed}")
    os.environ.pop("BRAIN_METACOG_LESION", None)
    out["lesion"] = _extract(d_lesion)

    checks = {
        "kb_relation_route_recalls_correct_fact": out["clean"]["recall_correct"],
        "degraded_turn_still_a_real_match_not_an_abstain": (out["degraded"]["recall_correct"]
                                                             and not out["degraded"]["abstained"]),
        "moat_clean_every_arm": bool(out["clean"]["moat_ok"] and out["degraded"]["moat_ok"]
                                     and out["lesion"]["moat_ok"]),
        # THE HEADLINE TARGET: clean reads confident/HIGH, degraded reads NOT-confident/LOW.
        "clean_confident_true": bool(out["clean"]["confident"] is True),
        "degraded_confident_false": bool(out["degraded"]["confident"] is not True),
        "vary_n_sentences": bool(out["clean"]["n_sentences"] > out["degraded"]["n_sentences"]),
        # LESION: the clean turn's confident read + reach grant COLLAPSE under BRAIN_METACOG_LESION=1.
        "lesion_collapses_confident": bool(out["lesion"]["confident"] is not True),
        "lesion_collapses_n_sentences": bool(out["lesion"]["n_sentences"] <= out["degraded"]["n_sentences"]),
    }
    out["checks"] = checks
    out["measurement_GO"] = bool(checks["kb_relation_route_recalls_correct_fact"]
                                 and checks["degraded_turn_still_a_real_match_not_an_abstain"]
                                 and checks["moat_clean_every_arm"])
    out["vary_lesion_GO"] = bool(checks["clean_confident_true"] and checks["degraded_confident_false"]
                                 and checks["vary_n_sentences"] and checks["lesion_collapses_confident"]
                                 and checks["lesion_collapses_n_sentences"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=os.path.join(_HERE, "verify_margin_norm_recalibration_100k.json"))
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        r = run_seed(seed)
        per_seed.append(r)
        print(f"[{time.time()-t0:.0f}s] seed {seed}: measurement_GO={r['measurement_GO']} "
              f"vary_lesion_GO={r['vary_lesion_GO']} clean_confident={r['clean']['confident']} "
              f"clean_mrc={r['clean']['mean_role_conf']} clean_n={r['clean']['n_sentences']} "
              f"degraded_confident={r['degraded']['confident']} degraded_mrc={r['degraded']['mean_role_conf']} "
              f"degraded_n={r['degraded']['n_sentences']} lesion_n={r['lesion']['n_sentences']}", flush=True)
    dt = time.time() - t0

    measurement_all_go = all(r["measurement_GO"] for r in per_seed)
    vary_lesion_all_go = all(r["vary_lesion_GO"] for r in per_seed)

    out = {
        "probe": "confidence_forthcomingness_margin_norm_recalibration_100k",
        "board": "108 cluster R3 (confidence-forthcomingness #94 re-verified against wikidata_100k)",
        "question": Q, "expected_svo": EXPECTED_SVO, "ltm_bundle": LTM_BUNDLE, "noise_sigma": NOISE_SIGMA,
        "ltm_load_flags": "enable_codebook_cache=True, enable_decode_escalation=True (matches webapp/server.py's "
                          "new post-R2 production default at any bundle scale)",
        "flags": "BRAIN_ELABORATE_FROM_LTM_SHARD=1 + BRAIN_CONFIDENCE_FORTHCOMING=1 + BRAIN_LTM_SHIP_DEFAULT=1 "
                "+ BRAIN_KB_RELATION_QUESTIONS default-ON + BRAIN_CLAIM_MOAT=0 escape (residual 1, unrelated), "
                "true floor",
        "backend": os.environ.get("SIM_BACKEND"), "seeds": args.seeds, "n_seeds": len(args.seeds),
        "elapsed_s": dt,
        "measurement_all_GO": measurement_all_go,
        "vary_lesion_all_GO": vary_lesion_all_go,
        "per_seed": per_seed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"  measurement_all_GO: {measurement_all_go}")
    print(f"  vary_lesion_all_GO (clean HIGH vs degraded LOW, n_sentences varies, lesion collapses): "
          f"{vary_lesion_all_go}")
    print("=" * 100)
    print(f"  wrote {os.path.relpath(args.out, _REPO)}  ({dt:.1f}s)")
    return 0 if (measurement_all_go and vary_lesion_all_go) else 1


if __name__ == "__main__":
    sys.exit(main())
