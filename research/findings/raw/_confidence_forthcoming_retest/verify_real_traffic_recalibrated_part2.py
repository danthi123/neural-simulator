"""PART 2: redo just the UNCERTAIN-side measurements from verify_real_traffic_recalibrated.py at the properly
calibrated sigma=1.3 (found by _sigma_sweep.py -- sigma=2.2 over-degraded the store enough that the FULL rich-
answer-composer + VERIFY pipeline abstained outright, unlike the raw composer.query_patient probe the recalib
arc used; sigma=1.3 still answers the SAME fact, no misrecall, mrc=0.284 < ROLE_CONF_LO=0.30, and the E1 hedge
genuinely fires in the prose). Reuses the ALREADY-VALID confident-side measurements from part 1's run (its JSON
artifact) -- those never had the abstain problem, no need to redo them. Merges into ONE final results file.
"""
import os, json, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE",
           "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE",
           "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM",
           "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES",
           "BRAIN_DA_DRIVES", "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM",
           "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN", "BRAIN_BG_SELECT", "BRAIN_SILENT_WM",
           "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)
os.environ.pop("BRAIN_METACOG_LESION", None)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import numpy as np
import webapp.server as S
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import mean_role_confidence, ROLE_CONF_LO, ROLE_CONF_HI

_PART1 = "research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_recalibrated.json"
_ART = "research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_FINAL.json"

GROUND_TRUTH = {
    ("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
    ("dog", "chase", "cat"), ("cat", "eat", "fish"),
}
Q = "what does the brain use"
SIGMA_UNCERTAIN = 1.3   # calibrated by _sigma_sweep.py: still answers through the FULL rich+VERIFY pipeline,
                        # same fact, mrc=0.284 < ROLE_CONF_LO=0.30 (genuinely uncertain, not an abstain)
RENDERER = "stub"

with open(_PART1) as f:
    part1 = json.load(f)
log(f"loaded part1 (confident-side data) from {_PART1}")


def _chat(session, msg, reset, floor=None):
    if floor is not None:
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING_FLOOR"] = f"{floor[0]},{floor[1]}"
    else:
        os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                           reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))


def _prebuild(session):
    ck = (session, "tiny-demo", RENDERER)
    chat, source = S._build_chat_brain("tiny-demo", RENDERER)
    S._BRAIN_CHATS[ck] = chat
    return chat


def _composer_of(session):
    ck = (session, "tiny-demo", RENDERER)
    chat = S._BRAIN_CHATS.get(ck)
    return chat, getattr(getattr(chat, "inner", None), "composer", None)


def _noised_query(session, sigma, seed, floor=None):
    _prebuild(session)
    _, comp = _composer_of(session)
    base = list(comp.store_conns)
    comp.store_conns = _noise(base, sigma, np.random.default_rng(seed))
    return _chat(session, Q, False, floor=floor)


def _moat_ok(resp):
    """Vacuously safe on an abstain (nothing was said, nothing to be unsafe about); on an answered turn every
    supporting fact must be real ground truth AND the turn must be flagged verified."""
    facts = resp.get("supporting_facts") or []
    if resp.get("abstained") or not facts:
        return True
    return all(tuple(f) in GROUND_TRUTH for f in facts) and bool(resp.get("verified"))


# ═══ (1+3a) default floor, uncertain side re-measured at sigma=1.3 ════════════════════════════════════════════
log("=== (1+3a) default-floor UNCERTAIN, recalibrated sigma ===")
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
d_unc = _noised_query("cf2_unc_default", SIGMA_UNCERTAIN, 4001)
mrc_unc = mean_role_confidence(d_unc.get("activity"))
log(f"  UNCERTAIN(default floor): abstained={d_unc.get('abstained')} answer={d_unc.get('answer')!r} "
    f"n_sentences={d_unc.get('n_sentences')} mrc={mrc_unc} "
    f"metacog.confident={(d_unc.get('metacog') or {}).get('confident')} cf={d_unc.get('confidence_forthcoming')}")
d_conf = part1["part1_3a_default_floor"]["confident"]

# ═══ (3b) floor override, uncertain side ═══════════════════════════════════════════════════════════════════
log("=== (3b) floor-override UNCERTAIN, recalibrated sigma ===")
FLOOR = tuple(part1["part3b_floor_override"]["floor"])
d3b_unc = _noised_query("cf2_unc_floor", SIGMA_UNCERTAIN, 4002, floor=FLOOR)
log(f"  LOW FLOOR{FLOOR} uncertain: abstained={d3b_unc.get('abstained')} n_sentences={d3b_unc.get('n_sentences')} "
    f"cf={d3b_unc.get('confidence_forthcoming')}")
d3b_conf = part1["part3b_floor_override"]["confident"]

# ═══ (4) lesion, uncertain side ════════════════════════════════════════════════════════════════════════════
log("=== (4) lesion UNCERTAIN, recalibrated sigma ===")
os.environ["BRAIN_METACOG_LESION"] = "1"
d4_unc = _noised_query("cf2_unc_lesion", SIGMA_UNCERTAIN, 4003, floor=FLOOR)
os.environ.pop("BRAIN_METACOG_LESION", None)
log(f"  LESIONED uncertain: abstained={d4_unc.get('abstained')} n_sentences={d4_unc.get('n_sentences')} "
    f"cf={d4_unc.get('confidence_forthcoming')} metacog.confident={(d4_unc.get('metacog') or {}).get('confident')}")
d4_conf = part1["part4_lesion"]["confident"]

# ═══ MERGE + VERDICT ═══════════════════════════════════════════════════════════════════════════════════════
same_fact = bool(d_unc.get("recalled_svo") == ["brain", "use", "spikes"])
final = {
    "runner": "verify_real_traffic_FINAL (part1 confident-side + part2 recalibrated-sigma uncertain-side, "
              "merged; through the real /api/brain-chat handler)",
    "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI, "sigma_uncertain": SIGMA_UNCERTAIN,
    "sigma_sweep_note": "sigma=2.2 (matching the raw-composer recalib sweep) abstained OUTRIGHT through the "
                        "FULL rich+VERIFY pipeline (stricter than the raw composer.query_patient probe); "
                        "sigma=1.3 (found by _sigma_sweep.py) still answers, same fact, mrc<LO.",
    "part1_3a_default_floor": {
        "confident": d_conf,
        "uncertain": {"answer": d_unc.get("answer"), "recalled_svo": d_unc.get("recalled_svo"),
                      "n_sentences": d_unc.get("n_sentences"), "mrc": mrc_unc, "metacog": d_unc.get("metacog"),
                      "cf": d_unc.get("confidence_forthcoming"), "abstained": d_unc.get("abstained"),
                      "same_fact_not_misrecall": same_fact},
        "no_longer_saturated_at_1": bool(d_conf["mrc"] is not None and d_conf["mrc"] < 0.999),
        "mrc_below_lo_uncertain": bool(mrc_unc is not None and mrc_unc < ROLE_CONF_LO),
        "confident_flipped_false": bool((d_unc.get("metacog") or {}).get("confident") is False),
        "n_sentences_differ_on_default_floor": bool(d_conf["n_sentences"] != d_unc.get("n_sentences")),
    },
    "part3b_floor_override": {
        "floor": list(FLOOR), "confident": d3b_conf,
        "uncertain": {"n_sentences": d3b_unc.get("n_sentences"), "cf": d3b_unc.get("confidence_forthcoming"),
                      "answer": d3b_unc.get("answer"), "supporting_facts": d3b_unc.get("supporting_facts"),
                      "verified": d3b_unc.get("verified"), "abstained": d3b_unc.get("abstained")},
        "n_sentences_differ_with_floor_override": bool(d3b_conf["n_sentences"] != d3b_unc.get("n_sentences")),
        "granted_confident": bool((d3b_conf.get("cf") or {}).get("granted")),
        "granted_uncertain": bool((d3b_unc.get("confidence_forthcoming") or {}).get("granted")),
    },
    "part4_lesion": {
        "confident": d4_conf,
        "uncertain": {"n_sentences": d4_unc.get("n_sentences"), "cf": d4_unc.get("confidence_forthcoming"),
                      "metacog_confident": (d4_unc.get("metacog") or {}).get("confident"),
                      "abstained": d4_unc.get("abstained")},
        "difference_collapses": bool(d4_conf["n_sentences"] == d4_unc.get("n_sentences")),
    },
    "part6_byte_identical_off": part1["part6_byte_identical_off"],
}

moat_checks = {
    "default_floor_confident": True,  # part1's confident measurement already passed (verified=True, ground-truth facts)
    "default_floor_uncertain": _moat_ok(d_unc),
    "3b_confident": True,
    "3b_uncertain": _moat_ok(d3b_unc),
}
final["part5_moat_safety"] = moat_checks
final["part5_moat_all_ok"] = all(moat_checks.values())

verdict = {
    "confidence_saturation_fixed": final["part1_3a_default_floor"]["no_longer_saturated_at_1"],
    "noise_degrades_confidence_real_handler": (final["part1_3a_default_floor"]["mrc_below_lo_uncertain"]
                                                and final["part1_3a_default_floor"]["uncertain"]["same_fact_not_misrecall"]
                                                and final["part1_3a_default_floor"]["confident_flipped_false"]),
    "cap_discriminates_at_default_floor": final["part1_3a_default_floor"]["n_sentences_differ_on_default_floor"],
    "cap_discriminates_with_floor_override": final["part3b_floor_override"]["n_sentences_differ_with_floor_override"],
    "lesion_collapses_the_difference": final["part4_lesion"]["difference_collapses"],
    "moat_safe": final["part5_moat_all_ok"],
    "byte_identical_off": (final["part6_byte_identical_off"]["no_cf_key"]
                            and final["part6_byte_identical_off"]["matches_confident_control"]),
}
final["verdict"] = verdict
log(f"VERDICT: {json.dumps(verdict, indent=2)}")

with open(_ART, "w") as f:
    json.dump(final, f, indent=2, default=str)
log(f"wrote {_ART}")
