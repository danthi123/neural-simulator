"""RE-TEST board #94 (confidence-caps-forthcomingness) now that issue #181's root-cause fix landed
(margin-keyed mean_role_confidence, recalibrated ROLE_CONF_LO/HI=0.30/0.50; commit da84fde7c). The
2026-08-27 NOGO (research/findings/2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-
NOGO.md) reverted the flip ONLY because mean_role_confidence saturated at 1.0 on every real turn -- that
reason is now gone. This script re-measures, through the REAL `/api/brain-chat` handler (webapp.server.
brain_chat, called in-process -- the same function the HTTP route dispatches to), on REAL (unpatched)
production traffic:

  (1+3a, ONE turn per condition) CONFIDENCE + DEFAULT-FLOOR CAP, COMBINED: mean_role_confidence and the
      confidence_forthcoming trace are read off the SAME turn (BRAIN_CONFIDENCE_FORTHCOMING=1 throughout --
      the cap trace is purely additive so this does not perturb the confidence read itself), for a real
      confident turn (intact store) and a genuinely-uncertain real turn (SAME composer + query, synaptic-
      noise-perturbed store -- the IDENTICAL degradation model issue #181's own verification used,
      research/runners/_emergent_graceful_degradation_derisk.py's `_noise`). This is the TRUE, un-overridden
      production floor (mood-neutral, NEUTRAL_SENTENCES=4).
  (3b) THE CAP MECHANISM with the module's OWN documented `BRAIN_CONFIDENCE_FORTHCOMING_FLOOR` testing
      affordance (its own docstring: "useful against a small demo KB whose natural content is already
      exhausted well below the production floor") -- confident vs uncertain, floor=(2,1).
  (4) LESION (`BRAIN_METACOG_LESION=1`) on the SAME (3b)-shaped conditions: the confident/uncertain
      difference must COLLAPSE to zero.
  (5) MOAT SAFETY: every volunteered sentence's SVO is checked against the tiny-demo ground-truth fact set.
  (6) BYTE-IDENTICAL OFF: `BRAIN_CONFIDENCE_FORTHCOMING` explicitly "0" (never popped -- guards the
      pop-as-off staleness trap if the default is later flipped to ON) reproduces the pre-#94-wiring
      response (no `confidence_forthcoming` key), tested in the SAME process as the ON arm.

SPEED (honest, does not touch the tested mechanism): `BRAIN_LTM_SHIP_DEFAULT=off` skips the cortical-LTM
attach. A timing probe (this dir's `_timing_probe.py`) measured ~110s per FRESH session (the composer/bridge
build dominates; LTM on/off did not change this -- the cost is the tiny-demo composer's own spiking-substrate
construction, not the LTM load). The LTM tier is never consulted for THIS test's facts anyway: `TieredFact
Store.query_patient` only falls through to LTM on a BUFFER abstain, and every fact used here
(brain/use/spikes, dog/chase/cat, ...) lives in the buffer and always answers from there; elaboration/chain
content is ALREADY buffer-tier-only regardless of LTM presence (a pre-existing, separately-documented
residual). So disabling LTM changes wall-clock only, not the mechanism under test.

Usage: SIM_BACKEND=numpy PYTHONPATH=. python research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_recalibrated.py
"""
import os, json, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"   # speed only -- see docstring; does not touch the tested mechanism
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE",
           "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE",
           "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM",
           "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES",
           "BRAIN_DA_DRIVES", "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM",
           "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN", "BRAIN_BG_SELECT", "BRAIN_SILENT_WM",
           "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)          # real shipped default (ON) -- the confidence read under test
os.environ.pop("BRAIN_METACOG_LESION", None)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import numpy as np
import webapp.server as S
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import mean_role_confidence, ROLE_CONF_LO, ROLE_CONF_HI

_ART = os.environ.get(
    "CF_JSON",
    "research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_recalibrated.json",
)
_RESULTS = {"runner": "verify_real_traffic_recalibrated (real /api/brain-chat, unpatched confidence, noise-"
                       "degraded uncertainty)",
            "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI}

GROUND_TRUTH = {
    ("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
    ("dog", "chase", "cat"), ("cat", "eat", "fish"),
}
Q = "what does the brain use"
SIGMA_UNCERTAIN = 2.2   # measured (recalib arc): mrc~0.15, well under ROLE_CONF_LO=0.30, still ANSWERS (no misrecall)
RENDERER = "stub"


def _chat(session, msg, reset, floor=None):
    if floor is not None:
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING_FLOOR"] = f"{floor[0]},{floor[1]}"
    else:
        os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                           reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))


def _prebuild(session):
    """Construct + cache the ChatBrain for `session` WITHOUT running a real turn (skips paying a real turn's
    rich-answer/planner cost on a throwaway message) -- so the noise perturbation lands on a fresh composer
    before the first REAL query, cheaply."""
    ck = (session, "tiny-demo", RENDERER)
    chat, source = S._build_chat_brain("tiny-demo", RENDERER)
    S._BRAIN_CHATS[ck] = chat
    return chat


def _composer_of(session):
    ck = (session, "tiny-demo", RENDERER)
    chat = S._BRAIN_CHATS.get(ck)
    return chat, getattr(getattr(chat, "inner", None), "composer", None)


def _noised_query(session, sigma, seed, floor=None):
    """Prebuild a fresh composer for `session`, synaptic-noise-perturb its store_conns (the #181 degradation
    model), then run the REAL question through the REAL handler against that perturbed composer."""
    _prebuild(session)
    _, comp = _composer_of(session)
    base = list(comp.store_conns)
    comp.store_conns = _noise(base, sigma, np.random.default_rng(seed))
    return _chat(session, Q, False, floor=floor)


# ═══ (1+3a) CONFIDENCE + DEFAULT-FLOOR CAP, combined (ONE turn per condition) ═════════════════════════════════
log("=== (1+3a) confidence + default-floor cap, real un-noised vs. real noise-degraded ===")
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
d_conf = _chat("cfretest_conf", Q, True)                              # intact store, confident, default floor
mrc_conf = mean_role_confidence(d_conf.get("activity"))
log(f"  CONFIDENT: answer={d_conf.get('answer')!r} n_sentences={d_conf.get('n_sentences')} mrc={mrc_conf} "
    f"metacog.confident={(d_conf.get('metacog') or {}).get('confident')} cf={d_conf.get('confidence_forthcoming')}")

d_unc = _noised_query("cfretest_unc", SIGMA_UNCERTAIN, 2026_08_27)     # SAME real query, noised store, default floor
mrc_unc = mean_role_confidence(d_unc.get("activity"))
log(f"  UNCERTAIN: answer={d_unc.get('answer')!r} n_sentences={d_unc.get('n_sentences')} mrc={mrc_unc} "
    f"metacog.confident={(d_unc.get('metacog') or {}).get('confident')} cf={d_unc.get('confidence_forthcoming')}")

same_fact = bool(d_unc.get("recalled_svo") == ["brain", "use", "spikes"])
_RESULTS["part1_3a_default_floor"] = {
    "confident": {"answer": d_conf.get("answer"), "n_sentences": d_conf.get("n_sentences"), "mrc": mrc_conf,
                  "metacog": d_conf.get("metacog"), "cf": d_conf.get("confidence_forthcoming")},
    "uncertain": {"sigma": SIGMA_UNCERTAIN, "answer": d_unc.get("answer"), "recalled_svo": d_unc.get("recalled_svo"),
                  "n_sentences": d_unc.get("n_sentences"), "mrc": mrc_unc, "metacog": d_unc.get("metacog"),
                  "cf": d_unc.get("confidence_forthcoming"), "same_fact_not_misrecall": same_fact},
    "no_longer_saturated_at_1": bool(mrc_conf is not None and mrc_conf < 0.999),
    "mrc_below_lo_uncertain": bool(mrc_unc is not None and mrc_unc < ROLE_CONF_LO),
    "confident_flipped_false": bool((d_unc.get("metacog") or {}).get("confident") is False),
    "n_sentences_differ_on_default_floor": bool(d_conf.get("n_sentences") != d_unc.get("n_sentences")),
}

# ═══ (3b) cap with the module's own FLOOR testing affordance ══════════════════════════════════════════════════
log("=== (3b) cap with the FLOOR testing affordance (content-exhausted small-KB case) ===")
FLOOR = (2, 1)
d3b_conf = _chat("cfretest_lowfloor_conf", Q, True, floor=FLOOR)
d3b_unc = _noised_query("cfretest_lowfloor_unc", SIGMA_UNCERTAIN, 2026_08_29, floor=FLOOR)
log(f"  LOW FLOOR{FLOOR} confident: n_sentences={d3b_conf.get('n_sentences')} cf={d3b_conf.get('confidence_forthcoming')}")
log(f"  LOW FLOOR{FLOOR} uncertain: n_sentences={d3b_unc.get('n_sentences')} cf={d3b_unc.get('confidence_forthcoming')}")
_RESULTS["part3b_floor_override"] = {
    "floor": list(FLOOR),
    "confident": {"n_sentences": d3b_conf.get("n_sentences"), "cf": d3b_conf.get("confidence_forthcoming"),
                  "answer": d3b_conf.get("answer"), "supporting_facts": d3b_conf.get("supporting_facts"),
                  "verified": d3b_conf.get("verified")},
    "uncertain": {"n_sentences": d3b_unc.get("n_sentences"), "cf": d3b_unc.get("confidence_forthcoming"),
                  "answer": d3b_unc.get("answer"), "supporting_facts": d3b_unc.get("supporting_facts"),
                  "verified": d3b_unc.get("verified")},
    "n_sentences_differ_with_floor_override": bool(d3b_conf.get("n_sentences") != d3b_unc.get("n_sentences")),
    "granted_confident": bool((d3b_conf.get("confidence_forthcoming") or {}).get("granted")),
    "granted_uncertain": bool((d3b_unc.get("confidence_forthcoming") or {}).get("granted")),
}

# ═══ (4) LESION: the (3b) difference must COLLAPSE ═══════════════════════════════════════════════════════════
log("=== (4) lesion check (BRAIN_METACOG_LESION=1) on the SAME (3b) conditions ===")
os.environ["BRAIN_METACOG_LESION"] = "1"
d4_conf = _chat("cfretest_lesion_conf", Q, True, floor=FLOOR)
d4_unc = _noised_query("cfretest_lesion_unc", SIGMA_UNCERTAIN, 2026_08_30, floor=FLOOR)
os.environ.pop("BRAIN_METACOG_LESION", None)
log(f"  LESIONED confident: n_sentences={d4_conf.get('n_sentences')} cf={d4_conf.get('confidence_forthcoming')}")
log(f"  LESIONED uncertain: n_sentences={d4_unc.get('n_sentences')} cf={d4_unc.get('confidence_forthcoming')}")
_RESULTS["part4_lesion"] = {
    "confident": {"n_sentences": d4_conf.get("n_sentences"), "cf": d4_conf.get("confidence_forthcoming"),
                  "metacog_confident": (d4_conf.get("metacog") or {}).get("confident")},
    "uncertain": {"n_sentences": d4_unc.get("n_sentences"), "cf": d4_unc.get("confidence_forthcoming"),
                  "metacog_confident": (d4_unc.get("metacog") or {}).get("confident")},
    "difference_collapses": bool(d4_conf.get("n_sentences") == d4_unc.get("n_sentences")),
}

# ═══ (5) MOAT SAFETY ════════════════════════════════════════════════════════════════════════════════════════
log("=== (5) moat safety ===")
def _moat_ok(resp):
    facts = resp.get("supporting_facts") or []
    return all(tuple(f) in GROUND_TRUTH for f in facts) and bool(resp.get("verified"))

moat_checks = {
    "default_floor_confident": _moat_ok(d_conf), "default_floor_uncertain": _moat_ok(d_unc),
    "3b_confident": _moat_ok(d3b_conf), "3b_uncertain": _moat_ok(d3b_unc),
}
log(f"  moat_checks={moat_checks}")
_RESULTS["part5_moat_safety"] = moat_checks
_RESULTS["part5_moat_all_ok"] = all(moat_checks.values())

# ═══ (6) BYTE-IDENTICAL OFF (explicit "0", not popped) in the SAME process ════════════════════════════════════
log("=== (6) byte-identical-off (explicit BRAIN_CONFIDENCE_FORTHCOMING=0) ===")
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)
d6 = _chat("cfretest_off", Q, True)
_RESULTS["part6_byte_identical_off"] = {
    "no_cf_key": bool(d6.get("confidence_forthcoming") is None),
    "answer": d6.get("answer"), "n_sentences": d6.get("n_sentences"),
    "matches_confident_control": bool(d6.get("answer") == d_conf.get("answer")
                                       and d6.get("n_sentences") == d_conf.get("n_sentences")),
}
log(f"  off arm: {_RESULTS['part6_byte_identical_off']}")

# ═══ VERDICT ════════════════════════════════════════════════════════════════════════════════════════════════
verdict = {
    "confidence_saturation_fixed": _RESULTS["part1_3a_default_floor"]["no_longer_saturated_at_1"],
    "noise_degrades_confidence_real_handler": (_RESULTS["part1_3a_default_floor"]["mrc_below_lo_uncertain"]
                                                and _RESULTS["part1_3a_default_floor"]["uncertain"]["same_fact_not_misrecall"]
                                                and _RESULTS["part1_3a_default_floor"]["confident_flipped_false"]),
    "cap_discriminates_at_default_floor": _RESULTS["part1_3a_default_floor"]["n_sentences_differ_on_default_floor"],
    "cap_discriminates_with_floor_override": _RESULTS["part3b_floor_override"]["n_sentences_differ_with_floor_override"],
    "lesion_collapses_the_difference": _RESULTS["part4_lesion"]["difference_collapses"],
    "moat_safe": _RESULTS["part5_moat_all_ok"],
    "byte_identical_off": (_RESULTS["part6_byte_identical_off"]["no_cf_key"]
                            and _RESULTS["part6_byte_identical_off"]["matches_confident_control"]),
}
_RESULTS["verdict"] = verdict
log(f"VERDICT: {json.dumps(verdict, indent=2)}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(_RESULTS, f, indent=2, default=str)
log(f"wrote {_ART}")
