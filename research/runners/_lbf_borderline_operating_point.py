"""LOAD-BEARING BORDERLINE — OPERATING-POINT DIAGNOSIS (research/lbf-borderline-diagnosis, 2026-09-21).

WHY. The 6-seed adequate load-bearing fraction (2026-09-21-load-bearing-fraction-6seed-adequate-0.85-robust-core-20)
found a ROBUST CORE of 20/26 faculties load-bearing in ALL 6 seeds and FOUR seed-dependent BORDERLINE faculties —
load-bearing in some but not all of seeds {42,43,44,100,101,102}:
    episodic-memory (5/6, off@s44), affect-marker-spiking-wta (4/6, off@s42,s43),
    source-provenance-honesty (4/6, off@s44,s102), prospective-memory (5/6, off@s44).
Each faculty's INTEGRATED decision field is a boolean derived from a CONTINUOUS spiking read crossing a FIXED host
CONSTANT threshold. This runner MEASURES, per seed, that continuous read + its threshold for BOTH the intact and the
lesion arm at the SAME operating-point drive the load-bearing instrument uses (the battery turn texts) — so the
MARGIN to the decision threshold is exposed, and WHY the flip happens in some seeds not others becomes visible.

THE FRAME (CLAUDE.md deepest lesson): "what else does the real system run ALONGSIDE this, that we replaced with a
CONSTANT?" Each threshold below is a static host bound standing in for a biological HOMEOSTATIC/competitive process
that would regulate the operating point toward reliability across the per-neuron heterogeneity draw (seeded from
BRAIN_CHAT_SEED / cfg.seed). This runner does not FIX anything — it CHARACTERIZES the operating point. It touches NO
production/default path (a new read-only diagnostic module; it only imports + calls production organ entry points).

FIDELITY. Every organ is built at the passed --seed via its OWN production constructor (EpisodicDapMemory(seed),
ProspectiveMemoryOrgan(seed), AffectDrivesWorkspace(seed), SourceProvenanceHonestyMonitor(seed)), which is exactly
how webapp/server.py builds it (each reads _brain_chat_seed()). So the per-seed continuous read here is the SAME
operating-point quantity the integrated arm produces — VALIDATED by comparing the predicted per-seed load-bearing
verdict against the committed adequate6 / pmem_v2 ground truth (see tools/_lbf_borderline_validate.py / the finding).

NOTE (reproducibility): the committed op_s*.json in this arc predate the additive `lesion_attributable_fraction`
field (a pure derived add: (intact - lesion)/intact from the intact/lesion values ALREADY recorded in each file).
Re-running reproduces byte-identical intact/lesion/margin reads plus that field — the reads are deterministic given
the seed (each organ reseeds its own bridge from `seed`).

Run (one seed; numpy CPU; memcap):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' tools/memcap.sh 10 -- .venv/bin/python \
     -m research.runners._lbf_borderline_operating_point --seed 42 \
     --out research/findings/raw/_lbf_borderline/op_s42.json
Self-test (no brain build): .venv/bin/python -m research.runners._lbf_borderline_operating_point --selftest
"""
from __future__ import annotations

import argparse
import json
import os

# the four operating-point thresholds (host CONSTANTS standing in for homeostatic regulation) — imported from the
# organs that own them, never re-typed (so a threshold change upstream surfaces here, not a stale copy).
from research.runners._episodic_dap_dialogue_memory import COMPLETE_MIN, CUE_OVER_CTRL
from research.runners._pmem_intention_latch_derisk import FIRE_THR
from research.runners._affect_marker_wta_derisk import DEAD_MARGIN
# attribution discipline (tools.lab): an intact/lesion pair must ASK whose the difference is, not just measure both
# arms (the gap#5 clamp owned 97% of a change nobody subtracted). Here: what FRACTION of the continuous operating-
# point read the LESIONED (brain) pathway owns = (intact - lesion)/intact. The organ is deterministic given seed,
# so the null is 0 by construction; a fraction ~1.0 means the read is the lesioned pathway's, not a confound.
from tools.lab import attributable_to

# the battery turn texts that DRIVE the operating point (reused verbatim; the load-bearing instrument uses these) ──
EPI_STORE_TOPIC = "dog"                     # epi_store "the dog chase the cat" -> note_topic('dog')
EPI_RECALL_REFERENT = "dog"                 # epi_recall "did we discuss the dog"
EMO_TEXT = "Wonderful! I am so happy and delighted, this is fantastic and amazing!"   # 'emo' turn
PMEM_FORM = "remind me to feed the dog when the bird sings"
PMEM_INTERVENING = ["what does the cat eat", "how is the weather today", "tell me about the sky"]
PMEM_CUE = "the bird sings"
PROV_KEY = ("wolf", "bites", "apple")       # 'well' "the wolf bites the apple" -> encoded PERCEIVED (directly taught)
PROV_ENCODED_AS = "perceived"
VOCAB = ["dog", "cat", "fish", "bird", "worm", "ball"]   # the tiny-demo agent vocabulary (episodic topics)


def _episodic(seed: int) -> dict:
    """apical dAP UP-state completion (held_cue) vs COMPLETE_MIN=0.20 (+ CUE_OVER_CTRL ratio over perm/nocue)."""
    import research.runners.d5_episodic_production_organ as EP
    org = EP.EpisodicRecallOrgan(seed, list(VOCAB))
    org.note_topic(EPI_STORE_TOPIC)
    ri = org.recall(EPI_RECALL_REFERENT, lesion=False)
    rl = org.recall(EPI_RECALL_REFERENT, lesion=True)
    cue_i = float(ri["apical_cue"])
    return {
        "quantity": "apical_cue (dendritic dAP UP-fraction, held cue)",
        "threshold_name": "COMPLETE_MIN", "threshold": float(COMPLETE_MIN),
        "intact": {"apical_cue": cue_i, "apical_perm": float(ri["apical_perm"]),
                   "apical_nocue": float(ri["apical_nocue"]), "in_memory": bool(ri["in_memory"])},
        "lesion": {"apical_cue": float(rl["apical_cue"]), "in_memory": bool(rl["in_memory"])},
        "margin_to_threshold": cue_i - float(COMPLETE_MIN),      # >0 needed (one of several gate conditions)
        "cue_over_ctrl": (cue_i / (max(float(ri["apical_perm"]), float(ri["apical_nocue"])) + 1e-6)),
        "cue_over_ctrl_required": float(CUE_OVER_CTRL),
        "lesion_attributable_fraction": attributable_to("episodic apical_cue", cue_i, float(rl["apical_cue"])),
        "predicted_load_bearing": bool(ri["in_memory"]) and not bool(rl["in_memory"]),
    }


def _prospective(seed: int) -> dict:
    """cue-monitor relative firing rate rel vs FIRE_THR=0.20 after formation + 3 intervening (the held x cue
    coincidence's operating point)."""
    import research.runners.prospective_memory_production_organ as PM
    f = PM.parse_intention(PMEM_FORM)

    def _run(lesion: bool):
        org = PM.ProspectiveMemoryOrgan(seed=seed)
        org.form_intention(f["action"], f["cue_clause"], f["cue_keywords"], lesion=lesion, hebbian_lesion=False)
        for t in PMEM_INTERVENING:
            org.read_turn(t)
        rd = org.read_turn(PMEM_CUE)
        return {"rel": float(rd.get("rel", 0.0)), "fired": bool(rd.get("fired")),
                "is_cue": bool(rd.get("is_cue")), "threshold": float(rd.get("threshold", FIRE_THR))}

    ri, rl = _run(False), _run(True)
    return {
        "quantity": "rel (cue-monitor relative firing rate off cp_firing_states)",
        "threshold_name": "FIRE_THR", "threshold": float(FIRE_THR),
        "intact": ri, "lesion": rl,
        "margin_to_threshold": ri["rel"] - float(FIRE_THR),
        "lesion_attributable_fraction": attributable_to("pmem rel", ri["rel"], rl["rel"]),
        "predicted_load_bearing": ri["fired"] and not rl["fired"],
    }


def _affect_marker(seed: int) -> dict:
    """WTA winner-minus-runner-up rate margin vs DEAD_MARGIN=0.05, at the felt mood the 'emo' ladder read produces.
    The lesion cuts the felt-state->assembly projection; the flip depends on whether the lesion's RESIDUAL margin
    still names the same marker (pass) or collapses to no clean winner / a different marker (load-bearing)."""
    from research.runners import affect_production_organ as AO
    from webapp.affect_drives_chat import AffectDrivesWorkspace
    from research.runners._affect_marker_wta_derisk import get_reader, marker_from_level
    appr = AO.appraise_text(EMO_TEXT)
    ws = AffectDrivesWorkspace(seed=seed)
    info = ws.observe(float(appr.get("valence", 0.0)), float(appr.get("arousal", 0.0)),
                      int(appr.get("n_hits", 0)), lesion=False)     # ladder INTACT (the WTA is the lesioned part)
    mood, felt, level = float(info["mood"]), float(info["felt_arousal"]), int(info["level"])
    high_arousal = bool(felt > 0.0)

    def _lead(lesion: bool):
        # RNG ISOLATION (2026-09-22 fix): build a FRESH reader per arm at the SAME seed, so both the intact and the
        # lesion read start from an IDENTICAL OU-noise RNG state and the ONLY inter-arm difference is the lesion flag.
        # The prior shared-reader form let the intact call advance the RNG before the lesion call, confounding the
        # margin with a different noise draw -> affect-marker's per-seed load-bearing label was noise-dependent
        # (the diagnosis caveat). With this, separability across seeds becomes assessable. (Re-run 6-seed is follow-on.)
        reader = get_reader(seed=seed)
        sel_level, _rates, meta = reader.select_valence(mood, lesion=lesion)
        word = marker_from_level(sel_level)
        if not word:
            return {"lead": "", "sel_level": sel_level, "margin": float(meta["margin"])}
        hi, _r, _m = reader.select_arousal(felt, lesion=lesion)
        emphatic = bool(hi) if hi is not None else high_arousal
        return {"lead": (word + "! ") if emphatic else (word + " — "),
                "sel_level": sel_level, "margin": float(meta["margin"])}

    li, ll = _lead(False), _lead(True)
    return {
        "quantity": "WTA winner-minus-runner-up rate margin (rates[top]-rates[second])",
        "threshold_name": "DEAD_MARGIN", "threshold": float(DEAD_MARGIN),
        "ladder": {"mood": mood, "felt_arousal": felt, "level": level},   # the seed-varying companion drive
        "intact": li, "lesion": ll,
        "margin_to_threshold": li["margin"] - float(DEAD_MARGIN),
        "lesion_margin_to_threshold": ll["margin"] - float(DEAD_MARGIN),  # >0 => lesion STILL names a marker
        "lesion_attributable_fraction": attributable_to("affect WTA margin", li["margin"], ll["margin"]),
        "predicted_load_bearing": (level != 0) and (li["lead"] != ll["lead"]),
    }


def _source_provenance(seed: int) -> dict:
    """signed opponent read d=(rate_perceived-rate_generated)/(rp+rg); the discretized SIGN (winner label) is what
    the instrument compares. The lesion runs learning-off -> silent pools -> a TRUE-CHANCE tie-break coin flip."""
    from research.runners.source_provenance_honesty import SourceProvenanceHonestyMonitor as Mon

    def _run(lesion: bool):
        m = Mon(seed=seed, lesion=lesion)
        m.encode_fact(PROV_KEY, PROV_ENCODED_AS)
        return m.judge_fact(PROV_KEY)

    ji, jl = _run(False), _run(True)
    cmp_i = (ji["known"], ji["label"], ji["agrees_with_encoded"], ji["encoded_as"])
    cmp_l = (jl["known"], jl["label"], jl["agrees_with_encoded"], jl["encoded_as"])
    return {
        "quantity": "d = (rate_perceived - rate_generated)/(rp+rg)  [SIGN discretized to label]",
        "threshold_name": "sign(0) + coin-flip tie-break when |margin|<1e-9 (lesion arm)",
        "threshold": 0.0,
        "intact": {k: ji.get(k) for k in ("known", "label", "d", "rate_perceived", "rate_generated",
                                          "agrees_with_encoded", "encoded_as")},
        "lesion": {k: jl.get(k) for k in ("known", "label", "d", "rate_perceived", "rate_generated",
                                          "agrees_with_encoded", "encoded_as")},
        "margin_to_threshold": float(ji.get("d") or 0.0),   # the INTACT signed opponent read magnitude
        "lesion_attributable_fraction": attributable_to("provenance d", float(ji.get("d") or 0.0),
                                                         float(jl.get("d") or 0.0)),
        "compared_fields_differ": cmp_i != cmp_l,
        "predicted_load_bearing": cmp_i != cmp_l,
    }


FACULTIES = {
    "episodic-memory": _episodic,
    "prospective-memory": _prospective,
    "affect-marker-spiking-wta": _affect_marker,
    "source-provenance-honesty": _source_provenance,
}


def run(seed: int) -> dict:
    rep = {"runner": "research.runners._lbf_borderline_operating_point", "kind": "operating-point-diagnosis",
           "seed": int(seed), "backend": os.environ.get("SIM_BACKEND", "numpy"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "per_faculty": {}}
    for name, fn in FACULTIES.items():
        try:
            rep["per_faculty"][name] = fn(int(seed))
        except Exception as e:
            import traceback
            rep["per_faculty"][name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    return rep


def selftest() -> bool:
    checks = {
        "COMPLETE_MIN is 0.20": abs(COMPLETE_MIN - 0.20) < 1e-9,
        "FIRE_THR is 0.20": abs(FIRE_THR - 0.20) < 1e-9,
        "DEAD_MARGIN is 0.05": abs(DEAD_MARGIN - 0.05) < 1e-9,
        "four borderline faculties mapped": set(FACULTIES) == {
            "episodic-memory", "prospective-memory", "affect-marker-spiking-wta", "source-provenance-honesty"},
        "prov key is the 'well' fact": PROV_KEY == ("wolf", "bites", "apple") and PROV_ENCODED_AS == "perceived",
        "epi topic in vocab": EPI_STORE_TOPIC in VOCAB,
    }
    ok = all(checks.values())
    print("=== BORDERLINE OPERATING-POINT DIAGNOSIS SELF-TEST ===")
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_lbf_borderline/op_s42.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return 0 if selftest() else 1
    # thread the substrate seed exactly as the load-bearing instrument does (byte-identical no-op at 42).
    os.environ["BRAIN_CHAT_SEED"] = str(args.seed)
    rep = run(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    print("\n===== BORDERLINE OPERATING POINT (seed %d) =====" % args.seed)
    for name, r in rep["per_faculty"].items():
        if "error" in r:
            print("  %-28s ERROR %s" % (name, r["error"]))
            continue
        print("  %-28s margin_to_thr=%+.4f  predicted_LB=%s  (%s vs %s)"
              % (name, r.get("margin_to_threshold", float("nan")), r.get("predicted_load_bearing"),
                 r.get("threshold_name"), r.get("threshold")))
    print("  wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
