"""gap#3 — MULTI-REFERENT DISAMBIGUATION via WTA BIASED-COMPETITION, decorrelated-battery de-risk.

THE GAP (research/findings/2026-06-17-multireferent-disambiguation-NEGATIVE.md). When the spiking working
memory holds >=2 discourse referents, which one does a BARE PRONOUN ("it") bind? Two documented converging
NEGATIVEs mapped the wall: (1) RECENCY is NEGATIVE (which held referent dominates is seed-dependent intrinsic
attractor competition, not the most-recent one; the order-control never flips); (2) a SALIENCE BOOST is
NEGATIVE (even a 4x write-drive boost never suppresses the competitor — a boost only ADDS activity to an
independent attractor). The specified fix (Desimone-Duncan 1995 biased competition; Wong-Wang 2006 attractor
WTA): WINNER-TAKE-ALL biased-competition inhibition between the referent attractors + a small CONTENT bias, so
the recurrence amplifies a small content asymmetry into a SUPPRESSIVE winner.

WHAT THIS DE-RISK ADDS over the prior GO (`_phaseB_biased_competition_derisk.py`, a narrow 4-trial GO-arm): a
DECORRELATED BATTERY that makes recency and salience provably uninformative, so the anti-cheat controls sit at
chance BY CONSTRUCTION, and any above-chance WTA accuracy can ONLY come from the content-steered competition.
Each trial independently randomizes the WRITE ORDER (=recency cue) and which referent is SALIENCE-BOOSTED
(=salience cue), both decorrelated from the content-correct answer (which the query verb's selectional
restriction picks). The battery is balanced so the majority-class baseline == chance.

THE FOUR ARMS (all WIRED + INVOKED per trial in run_seed; every control's accuracy feeds the printed verdict):
  A. WTA  (mechanism)  : BiasedCompetitionContextBuffer(competition=True); write (order + boost); read with a
                         CONTENT bias on the verb-selected referent; resolve via the sel_X/sel_FS_X Wong-Wang
                         WTA (mutual inhibition between the per-referent accumulators). -> EXPECT >> chance.
  B. LESION (control)  : SAME competition substrate + SAME write, but the content BIAS is REMOVED (bias_pA=0)
                         -> the WTA reverts to the intrinsic winner -> EXPECT ~chance (proves the bias, not the
                         competition alone, is the load-bearing signal).
  C. RECENCY (control) : the documented NEGATIVE #1 on the plain loop (SpikingLoopContextBuffer, NO competition,
                         NO bias): write the trial's ORDER at equal drive, plain read, pick the strongest-firing
                         held referent (argmax, no abstention = its best shot). -> EXPECT ~chance vs content.
  D. SALIENCE (control): the documented NEGATIVE #2 on the plain loop: write with the trial's referent BOOSTED
                         (drive x salience_boost), plain read, argmax. -> EXPECT ~chance vs content.
Plus two FREE heuristic-predictor controls (reported, non-gated): recency-pick = the last-written referent;
salience-pick = the boosted referent. Both are at chance vs content BY the decorrelated design (transparency).
Plus a no-confab MOAT sanity (reported): empty WM -> abstain; content-silent verb -> abstain.

GO GATE (PRE-REGISTERED, FROZEN; read the runner's OWN printed verdict, do NOT lift a field). Over the 6-seed
battery: WTA mean accuracy well above chance AND well above lesion; the LESION collapses to ~chance; the
RECENCY and SALIENCE spiking controls both stay at ~chance; and >=5/6 seeds pass per-seed. Exact numeric
conditions in GO_* constants below; the verdict line "==> GO/BOUNDARY/NEGATIVE" is the authority.

HOST-SCAFFOLD SHORTCUT (FLAGGED, BRAIN-BASED-ONLY): the content-bias TARGET (which referent to bias) is picked
by the host `content_bias_target` (animacy/selectional-restriction lexicons). The WIN is brain-based (the
spiking WTA competition + selective suppression + the Wong-Wang recurrence amplifying the small content
asymmetry); the content SCORING is host in this probe. The named follow-on neuralizes it into a learned
synaptic feature-compatibility map. This de-risk closes the MECHANISM gap (WTA disambiguation works where
recency+salience cannot); the neural-bias conversion is the separate BRAIN-BASED-ONLY follow-on.

reuse-by-import; NO sim/ edit. Run:
  SIM_BACKEND=numpy python -m research.runners._gap3_multireferent_wta_disambiguation_derisk --seeds 42 43 44 100 101 102
Smoke (proves it RUNS + controls live + prints a verdict; NOT a GO/negative claim):
  SIM_BACKEND=numpy python -m research.runners._gap3_multireferent_wta_disambiguation_derisk --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# CPU-first: default to the numpy backend unless the caller overrides (must be set before any sim import).
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the plain WM loop (the documented-NEGATIVE substrate) + the composer's referent-attractor
# WTA (the biased-competition buffer + content-bias helper + resolver + feature lexicons).
from research.runners.content_selection_spiking import SpikingLoopContextBuffer
from research.runners.biased_competition_buffer import (
    ANIMACY,
    VERB_SELECTS,
    BiasedCompetitionContextBuffer,
    content_bias_target,
    resolve_referent,
)

# --------------------------------------------------------------------------------------------------------
# PRE-REGISTERED / FROZEN GO thresholds (the verdict reads these; they are recorded into the output JSON).
# --------------------------------------------------------------------------------------------------------
GO_WTA_MEAN_MIN = 0.70          # WTA mean disambiguation accuracy must clear this (>> pooled chance ~0.46)
GO_COLLAPSE_MARGIN = 0.20       # WTA_mean - lesion_mean must be >= this (the bias is load-bearing)
GO_CONTROL_SLACK = 0.15         # a control "stays at chance" iff its mean <= chance_pooled + this
GO_PERSEED_WTA_MIN = 0.65       # a seed passes iff its WTA acc >= this ...
GO_PERSEED_CTRL_SLACK = 0.20    #   ... AND lesion/recency/salience accs <= chance_pooled + this
GO_MIN_SEEDS = 5                # >= this many of 6 seeds must pass


# --------------------------------------------------------------------------------------------------------
# The decorrelated trial battery. Each trial: referents (>=2, opposing content features), a query verb whose
# selectional restriction picks EXACTLY ONE (= content-correct), a WRITE ORDER (recency cue), and a BOOSTED
# referent (salience cue). Order and boost are decorrelated from the content-correct answer, and the battery
# is balanced (majority-class baseline == chance).
# --------------------------------------------------------------------------------------------------------
# 2-referent pairs: (animate, inanimate). Verb 'anim_verb' selects the animate; 'inanim_verb' the inanimate.
TWO_REF_PAIRS = [
    {"anim": "cat", "inanim": "ball", "anim_verb": "eat", "inanim_verb": "roll"},
    {"anim": "bird", "inanim": "river", "anim_verb": "chase", "inanim_verb": "float"},
]
# 3-referent templates: one content-compatible + two incompatible; chance = 1/3.
THREE_REF_TEMPLATES = [
    {"correct": "cat", "distractors": ["ball", "river"], "verb": "eat"},     # eat->animate; only cat is animate
    {"correct": "ball", "distractors": ["cat", "dog"], "verb": "roll"},      # roll->inanimate; only ball is inanimate
]


def _build_two_ref_trials(pairs):
    """8 trials/pair: verb {anim, inanim} x order {2} x boost {2}. Fully crosses recency & salience against
    the content-correct answer -> both cues decorrelated (P(recent==correct)=P(boosted==correct)=0.5)."""
    trials = []
    for pr in pairs:
        A, I = pr["anim"], pr["inanim"]
        for verb in (pr["anim_verb"], pr["inanim_verb"]):
            correct = content_bias_target([A, I], verb)
            assert correct is not None, f"verb {verb} does not disambiguate {[A, I]}"
            for order in ([A, I], [I, A]):
                for boosted in (A, I):
                    trials.append({"referents": [A, I], "verb": verb, "correct": correct,
                                   "order": list(order), "boosted": boosted, "n_ref": 2})
    return trials


def _build_three_ref_trials(templates):
    """3 trials/template: the correct referent placed at write-position 0/1/2 (recency decorrelated: recent==
    correct in exactly 1/3), boost always on a DISTRACTOR (salience maximally misleading: never on correct)."""
    trials = []
    for t in templates:
        C = t["correct"]
        D1, D2 = t["distractors"]
        verb = t["verb"]
        assert content_bias_target([C, D1, D2], verb) == C, f"template {t} content-correct mismatch"
        configs = [
            {"order": [C, D1, D2], "boosted": D1},   # correct first ; recent = D2
            {"order": [D1, C, D2], "boosted": D2},   # correct middle; recent = D2
            {"order": [D1, D2, C], "boosted": D1},   # correct last  ; recent = C  (1/3)
        ]
        for cfg in configs:
            trials.append({"referents": [C, D1, D2], "verb": verb, "correct": C,
                           "order": list(cfg["order"]), "boosted": cfg["boosted"], "n_ref": 3})
    return trials


# --------------------------------------------------------------------------------------------------------
# The four arms. Each builds its OWN fresh bridge (no cross-arm state contamination); each returns the
# resolved referent + whether it matched the content-correct answer.
# --------------------------------------------------------------------------------------------------------
def _write_with_order_and_boost(buf, trial, base_drive, boost, stim, settle):
    """Write the referents in the trial's ORDER; the trial's BOOSTED referent gets drive x boost (and
    proportionally more stim, matching the documented salience protocol)."""
    for c in trial["order"]:
        if c == trial["boosted"] and boost > 1.0:
            buf.update([c], drive_pA=base_drive * boost, stim=int(stim * boost), settle=settle)
        else:
            buf.update([c], drive_pA=base_drive, stim=stim, settle=settle)


def arm_wta(trial, seed, kn, lesion):
    """Arm A (lesion=False) / Arm B (lesion=True). The competition substrate holds the referents; the read
    re-presents them as co-active competitors and (unless lesioned) injects a SMALL content bias into the
    verb-selected referent's accumulator. Returns the moat-gated WTA winner (resolve_referent) + a forced
    argmax winner (raw disambiguation power, reported)."""
    fav = content_bias_target(trial["referents"], trial["verb"])
    buf = BiasedCompetitionContextBuffer(
        trial["referents"], n=kn["n"], pattern_size=kn["pattern_size"], seed=seed,
        enable_ou=False, competition=True, attractor_weight=kn["attractor_weight"])
    _write_with_order_and_boost(buf, trial, kn["drive_pA"], kn["salience_boost"], kn["stim"], kn["settle"])
    read = buf.read(window=kn["window"],
                    bias_concept=(None if lesion else fav),
                    bias_pA=(0.0 if lesion else kn["bias_pA"]))
    resolved = resolve_referent(read, spec_threshold=kn["spec_threshold"])          # moat + margin gated
    sel = read["sel"]
    argmax_win = max(sel.items(), key=lambda kv: kv[1])[0] if sel else None          # forced (raw power)
    return {"favored": fav, "resolved": resolved, "argmax": argmax_win,
            "correct": bool(resolved == trial["correct"]),
            "argmax_correct": bool(argmax_win == trial["correct"]),
            "sel": {c: round(float(v), 4) for c, v in sel.items()}}


def arm_recency_control(trial, seed, kn):
    """Arm C — documented NEGATIVE #1 (recency). The plain WM loop (no competition, no bias): write in the
    trial's ORDER at EQUAL drive (isolating recency), plain read, pick the strongest-firing held referent
    (argmax = the control's best shot). Scored vs the content-correct answer -> ~chance by design."""
    buf = SpikingLoopContextBuffer(trial["referents"], n=kn["n"], pattern_size=kn["pattern_size"],
                                   seed=seed, enable_ou=False)
    for c in trial["order"]:
        buf.update([c], drive_pA=kn["drive_pA"], stim=kn["stim"], settle=kn["settle"])
    rates = buf.read(window=kn["window"])
    win = max(rates.items(), key=lambda kv: kv[1])[0] if rates else None
    return {"answer": win, "correct": bool(win == trial["correct"]),
            "rates": {c: round(float(v), 4) for c, v in rates.items()}}


def arm_salience_control(trial, seed, kn):
    """Arm D — documented NEGATIVE #2 (salience). The plain WM loop: write with the trial's BOOSTED referent
    at drive x salience_boost (isolating salience), plain read, argmax. Scored vs content-correct -> ~chance
    (the boost only ADDS activity; the intrinsic attractor still wins, and the boosted referent is
    decorrelated from content anyway)."""
    buf = SpikingLoopContextBuffer(trial["referents"], n=kn["n"], pattern_size=kn["pattern_size"],
                                   seed=seed, enable_ou=False)
    # neutral order; only the boost matters here (order is the recency arm's variable).
    for c in trial["referents"]:
        if c == trial["boosted"]:
            buf.update([c], drive_pA=kn["drive_pA"] * kn["salience_boost"],
                       stim=int(kn["stim"] * kn["salience_boost"]), settle=kn["settle"])
        else:
            buf.update([c], drive_pA=kn["drive_pA"], stim=kn["stim"], settle=kn["settle"])
    rates = buf.read(window=kn["window"])
    win = max(rates.items(), key=lambda kv: kv[1])[0] if rates else None
    return {"answer": win, "correct": bool(win == trial["correct"]),
            "rates": {c: round(float(v), 4) for c, v in rates.items()}}


def moat_check(seed, kn):
    """No-confab sanity (reported): (a) empty WM + content bias -> abstain (None); (b) 2 held referents but a
    content-SILENT verb ('see' has no selectional restriction) -> abstain."""
    # (a) empty WM
    be = BiasedCompetitionContextBuffer(["cat", "ball"], n=kn["n"], pattern_size=kn["pattern_size"],
                                        seed=seed, enable_ou=False, competition=True,
                                        attractor_weight=kn["attractor_weight"])
    read_e = be.read(window=kn["window"], bias_concept="cat", bias_pA=kn["bias_pA"])
    empty_abstains = resolve_referent(read_e, spec_threshold=kn["spec_threshold"]) is None
    # (b) content-silent verb
    bs = BiasedCompetitionContextBuffer(["cat", "ball"], n=kn["n"], pattern_size=kn["pattern_size"],
                                        seed=seed, enable_ou=False, competition=True,
                                        attractor_weight=kn["attractor_weight"])
    bs.update(["cat"]); bs.update(["ball"])
    fav_silent = content_bias_target(["cat", "ball"], "see")   # 'see' not in VERB_SELECTS -> None
    silent_abstains = fav_silent is None
    return {"empty_abstains": bool(empty_abstains), "silent_abstains": bool(silent_abstains),
            "ok": bool(empty_abstains and silent_abstains)}


# --------------------------------------------------------------------------------------------------------
# Per-seed run: INVOKE all four arms + the free heuristic predictors + the moat, on every trial.
# --------------------------------------------------------------------------------------------------------
def run_seed(seed, trials, kn, verbose=False):
    per = {"wta": [], "lesion": [], "recency": [], "salience": [],
           "wta_argmax": [], "recency_heur": [], "salience_heur": []}
    invoked = {"wta": 0, "lesion": 0, "recency": 0, "salience": 0}
    trial_logs = []
    for tr in trials:
        a = arm_wta(tr, seed, kn, lesion=False);  invoked["wta"] += 1
        b = arm_wta(tr, seed, kn, lesion=True);   invoked["lesion"] += 1
        c = arm_recency_control(tr, seed, kn);    invoked["recency"] += 1
        d = arm_salience_control(tr, seed, kn);   invoked["salience"] += 1
        # free heuristic predictors (the pure recency/salience RULES): last-written / boosted referent.
        rec_pick = tr["order"][-1]
        sal_pick = tr["boosted"]
        per["wta"].append(a["correct"]);            per["wta_argmax"].append(a["argmax_correct"])
        per["lesion"].append(b["correct"])
        per["recency"].append(c["correct"]);        per["salience"].append(d["correct"])
        per["recency_heur"].append(rec_pick == tr["correct"])
        per["salience_heur"].append(sal_pick == tr["correct"])
        trial_logs.append({
            "referents": tr["referents"], "verb": tr["verb"], "correct": tr["correct"],
            "order": tr["order"], "boosted": tr["boosted"], "n_ref": tr["n_ref"],
            "wta": a["resolved"], "wta_correct": a["correct"], "wta_argmax": a["argmax"],
            "lesion": b["resolved"], "lesion_correct": b["correct"],
            "recency": c["answer"], "recency_correct": c["correct"],
            "salience": d["answer"], "salience_correct": d["correct"],
            "recency_pick": rec_pick, "salience_pick": sal_pick, "wta_sel": a["sel"]})
        if verbose:
            print(f"      {tr['verb']:6s} refs={tr['referents']} ord={tr['order']} boost={tr['boosted']} "
                  f"-> WTA={a['resolved']}({'Y' if a['correct'] else 'n'}) LES={b['resolved']} "
                  f"REC={c['answer']} SAL={d['answer']} (correct={tr['correct']})", flush=True)

    def acc(key):
        v = per[key]
        return float(np.mean(v)) if v else float("nan")

    moat = moat_check(seed, kn)
    n_tr = len(trials)
    chance = float(np.mean([1.0 / t["n_ref"] for t in trials])) if trials else float("nan")
    seed_row = {
        "seed": seed, "n_trials": n_tr, "chance": chance,
        "wta_acc": acc("wta"), "wta_argmax_acc": acc("wta_argmax"), "lesion_acc": acc("lesion"),
        "recency_acc": acc("recency"), "salience_acc": acc("salience"),
        "recency_heur_acc": acc("recency_heur"), "salience_heur_acc": acc("salience_heur"),
        "moat": moat, "invoked": invoked, "trials": trial_logs,
    }
    # per-seed pass (frozen bar)
    seed_row["pass"] = bool(
        seed_row["wta_acc"] >= GO_PERSEED_WTA_MIN
        and seed_row["lesion_acc"] <= chance + GO_PERSEED_CTRL_SLACK
        and seed_row["recency_acc"] <= chance + GO_PERSEED_CTRL_SLACK
        and seed_row["salience_acc"] <= chance + GO_PERSEED_CTRL_SLACK)
    return seed_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n", type=int, default=600, help="loop region size (neurons per cortex_ctx/dlpfc_wm)")
    ap.add_argument("--pattern-size", type=int, default=40, help="attractor pattern size per referent")
    ap.add_argument("--attractor-weight", type=float, default=50.0)
    ap.add_argument("--bias-pA", type=float, default=2500.0,
                    help="content-bias current (~1x drive; SMALL — the magnitude a uniform boost already FAILED "
                         "at — so any win is the competition amplifying a small content asymmetry)")
    ap.add_argument("--salience-boost", type=float, default=4.0,
                    help="salience-cue write-drive multiplier (matches the documented salience NEGATIVE's 4x)")
    ap.add_argument("--drive-pA", type=float, default=2500.0)
    ap.add_argument("--spec-threshold", type=float, default=1.3)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stim", type=int, default=40)
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--two-ref-pairs", type=int, default=2, help="how many of the 2-ref pairs to use (1-2)")
    ap.add_argument("--three-ref-templates", type=int, default=2, help="how many 3-ref templates to use (0-2)")
    ap.add_argument("--smoke", action="store_true",
                    help="TINY 1-seed config to prove it RUNS + controls live + prints a verdict (NOT a claim)")
    ap.add_argument("--out", default="research/findings/raw/_gap3_multireferent_wta_disambiguation.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.seeds = [42]
        a.n = 220
        a.pattern_size = 16
        a.window = 4
        a.stim = 4
        a.settle = 3
        a.salience_boost = 2.0
        a.two_ref_pairs = 1
        a.three_ref_templates = 1
        a.out = "research/findings/raw/_gap3_multireferent_wta_disambiguation_SMOKE.json"

    kn = {
        "n": a.n, "pattern_size": a.pattern_size, "attractor_weight": a.attractor_weight,
        "bias_pA": a.bias_pA, "salience_boost": a.salience_boost, "drive_pA": a.drive_pA,
        "spec_threshold": a.spec_threshold, "window": a.window, "stim": a.stim, "settle": a.settle,
    }
    trials = (_build_two_ref_trials(TWO_REF_PAIRS[:a.two_ref_pairs])
              + _build_three_ref_trials(THREE_REF_TEMPLATES[:a.three_ref_templates]))
    chance_pooled = float(np.mean([1.0 / t["n_ref"] for t in trials]))

    print("[gap#3 multi-referent WTA disambiguation de-risk]", flush=True)
    print(f"  question: does WTA biased-competition (mutual inhibition + a small CONTENT bias) bind a bare "
          f"pronoun to the\n  content-correct one of >=2 held referents, where RECENCY + SALIENCE (decorrelated "
          f"here) provably cannot?", flush=True)
    print(f"  battery: {len(trials)} trials/seed ({a.two_ref_pairs} two-ref pairs + {a.three_ref_templates} "
          f"three-ref templates); pooled chance={chance_pooled:.3f}", flush=True)
    print(f"  knobs: {kn}", flush=True)
    print(f"  seeds: {a.seeds}{'   [SMOKE — not a GO/negative claim]' if a.smoke else ''}\n", flush=True)

    rows = []
    for seed in a.seeds:
        print(f"  [seed {seed}] running {len(trials)} trials x 4 arms ...", flush=True)
        row = run_seed(seed, trials, kn, verbose=a.verbose)
        rows.append(row)
        print(f"    invoked: {row['invoked']} | chance={row['chance']:.3f}", flush=True)
        print(f"    WTA acc={row['wta_acc']:.3f} (argmax {row['wta_argmax_acc']:.3f}) | "
              f"LESION acc={row['lesion_acc']:.3f} | RECENCY acc={row['recency_acc']:.3f} | "
              f"SALIENCE acc={row['salience_acc']:.3f} | "
              f"heur(rec {row['recency_heur_acc']:.3f}/sal {row['salience_heur_acc']:.3f}) | "
              f"moat={row['moat']['ok']} | pass={row['pass']}", flush=True)

    # aggregate
    n = len(rows)

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    wta_mean, les_mean = m("wta_acc"), m("lesion_acc")
    rec_mean, sal_mean = m("recency_acc"), m("salience_acc")
    rec_h_mean, sal_h_mean = m("recency_heur_acc"), m("salience_heur_acc")
    n_pass = sum(r["pass"] for r in rows)
    moat_ok = sum(r["moat"]["ok"] for r in rows)

    # frozen GO conditions
    cond_wta = wta_mean >= GO_WTA_MEAN_MIN
    cond_collapse = (wta_mean - les_mean) >= GO_COLLAPSE_MARGIN and les_mean <= chance_pooled + GO_CONTROL_SLACK
    cond_recency = rec_mean <= chance_pooled + GO_CONTROL_SLACK
    cond_salience = sal_mean <= chance_pooled + GO_CONTROL_SLACK
    cond_seeds = n_pass >= min(GO_MIN_SEEDS, n)
    GO = bool(cond_wta and cond_collapse and cond_recency and cond_salience and cond_seeds)
    # BOUNDARY = the controls behave (lesion collapses, recency+salience at chance) but the WTA itself does not
    # clear the bar / not enough seeds pass -> the mechanism is real but under-powered on this battery.
    BOUNDARY = bool((not GO) and cond_collapse and cond_recency and cond_salience
                    and (wta_mean - les_mean) >= 0.10)

    summary = {
        "n_seeds": n, "chance_pooled": chance_pooled, "n_trials_per_seed": len(trials),
        "wta_mean": wta_mean, "wta_argmax_mean": m("wta_argmax_acc"), "lesion_mean": les_mean,
        "recency_mean": rec_mean, "salience_mean": sal_mean,
        "recency_heur_mean": rec_h_mean, "salience_heur_mean": sal_h_mean,
        "n_pass": n_pass, "moat_ok_seeds": moat_ok,
        "cond_wta_above_bar": cond_wta, "cond_lesion_collapses": cond_collapse,
        "cond_recency_at_chance": cond_recency, "cond_salience_at_chance": cond_salience,
        "cond_seeds": cond_seeds, "GO": GO, "BOUNDARY": BOUNDARY, "smoke": bool(a.smoke),
        "thresholds": {"GO_WTA_MEAN_MIN": GO_WTA_MEAN_MIN, "GO_COLLAPSE_MARGIN": GO_COLLAPSE_MARGIN,
                       "GO_CONTROL_SLACK": GO_CONTROL_SLACK, "GO_PERSEED_WTA_MIN": GO_PERSEED_WTA_MIN,
                       "GO_PERSEED_CTRL_SLACK": GO_PERSEED_CTRL_SLACK, "GO_MIN_SEEDS": GO_MIN_SEEDS},
        "knobs": kn, "seeds": a.seeds,
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "results": rows}, fh, indent=2, default=str)

    print(f"\n{'=' * 104}", flush=True)
    print(f"  MULTI-REFERENT WTA DISAMBIGUATION — {n} seeds, {len(trials)} trials/seed, pooled chance "
          f"{chance_pooled:.3f}", flush=True)
    print(f"    WTA (mechanism)        mean acc = {wta_mean:.3f}   [>= {GO_WTA_MEAN_MIN:.2f}? {cond_wta}]",
          flush=True)
    print(f"    WTA-LESION (bias off)  mean acc = {les_mean:.3f}   [collapse >= {GO_COLLAPSE_MARGIN:.2f} & "
          f"<= chance+{GO_CONTROL_SLACK:.2f}? {cond_collapse}]", flush=True)
    print(f"    RECENCY control        mean acc = {rec_mean:.3f}   [<= chance+{GO_CONTROL_SLACK:.2f}? "
          f"{cond_recency}]", flush=True)
    print(f"    SALIENCE control       mean acc = {sal_mean:.3f}   [<= chance+{GO_CONTROL_SLACK:.2f}? "
          f"{cond_salience}]", flush=True)
    print(f"    (free heuristics: recency-pick {rec_h_mean:.3f} / salience-pick {sal_h_mean:.3f}; "
          f"moat ok {moat_ok}/{n})", flush=True)
    print(f"    per-seed pass: {n_pass}/{n}  [>= {min(GO_MIN_SEEDS, n)}? {cond_seeds}]", flush=True)
    if a.smoke:
        print(f"\n  ==> SMOKE COMPLETE: the runner RUNS end-to-end, all 4 arms + moat were INVOKED "
              f"(per-seed 'invoked' dict), and a verdict was produced. This TINY config is NOT a GO/negative "
              f"claim — run the full 6-seed battery for the verdict.", flush=True)
    elif GO:
        print(f"\n  ==> GO: WTA biased competition resolves multi-referent pronouns by CONTENT where recency + "
              f"salience CANNOT.\n  The bias is load-bearing (lesion collapses to ~chance) and the decorrelated "
              f"recency/salience controls stay at chance;\n  {n_pass}/{n} seeds pass. gap#3 CLOSED (mechanism); "
              f"the neural content-bias is the flagged BRAIN-BASED-ONLY follow-on.", flush=True)
    elif BOUNDARY:
        print(f"\n  ==> BOUNDARY: the controls behave (lesion collapses, recency+salience at chance) so the "
              f"mechanism is REAL,\n  but WTA acc {wta_mean:.3f} does not clear {GO_WTA_MEAN_MIN:.2f} / not enough "
              f"seeds pass -> localizes competition-strength-vs-intrinsic-asymmetry as the tuning residual "
              f"(content-graded/normalized bias).", flush=True)
    else:
        print(f"\n  ==> NEGATIVE: WTA acc {wta_mean:.3f} does not exceed the controls decisively (lesion "
              f"{les_mean:.3f}, recency {rec_mean:.3f}, salience {sal_mean:.3f}) — either the bias is not "
              f"load-bearing or the controls are not at chance. Do NOT escalate into a config search; re-scope "
              f"the mechanism.", flush=True)
    print(f"  [saved] {a.out}\n{'=' * 104}", flush=True)


if __name__ == "__main__":
    main()
