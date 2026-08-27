"""6-SEED PRODUCTION FLIP-SOAK for the activity-silent-WM anaphora maintenance-mode swap (the BRAIN_SILENT_WM flip
gate). 2026-08-26.

This is the gate the parent reads before flipping `BRAIN_SILENT_WM` default-ON. It proves, at 6 seeds, that the
production organ (`activity_silent_wm_production_organ.ActivitySilentWMOrgan`) is a genuine, LOAD-BEARING production
consumer of the de-risked Mongillo activity-silent-WM mechanism — NOT a hollow checkbox:

  1. UNDERLYING MECHANISM (unchanged): re-runs the de-risk `run_one(seed)` and confirms the same 6/6 GO the finding
     reported (a nonspecific ping reactivates a silently-held item; the FAIR tau_f-lesion collapses it; delay silent).
  2. LOAD-BEARING PRODUCTION TURN: for each seed, a 3-turn discourse scenario — INTRODUCE a focus referent (+ a couple
     of distractor referents earlier), an intervening DISTRACTOR turn (silent delay), then a temporal-recall query —
     driven through the ORGAN's production `judge`. GO per seed requires ALL of:
       (a) INTACT recovers the correct focus  (recovered == focus, decisive margin > MARGIN_MIN),
       (b) the hold was genuinely SILENT      (silent_delay True — the persistent-attractor path did not leak),
       (c) the FAIR facilitation LESION (tau_f~5) ABSTAINS (recovered is None) — the de-risk oracle: the ping recovers
           the item WITH the facilitated buffer but NOT with tau_f collapsed,
       (d) the RENDERED REPLY genuinely CHANGES between intact and lesion (correct anaphor vs abstain) — the output of a
           chat turn demonstrably DEPENDS on the silent hold (anti-hollow).
  3. FLAG-OFF BYTE-IDENTICAL (wiring guard): with `BRAIN_SILENT_WM` unset the organ is never entered; and the organ's
     `judge` returns None on every out-of-scope turn (no recall query / nothing held) -> the caller stays byte-identical.
     Asserted here over a battery of ordinary turns.
  4. SPECIFICITY: the focus referent is ROTATED across seeds (dog/cat/bird) so a GO means the ping recovers WHICHEVER
     was the focus, not a fixed structural favorite.

Run: SIM_BACKEND=numpy python -u -m research.runners._activity_silent_wm_production_soak --seeds 42 43 44 100 101 102
NO `sim/` edit — reuse-by-import of the committed de-risk + the production organ.
"""
import os
import sys
import json
import argparse

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

import research.runners.activity_silent_wm_production_organ as O
from research.runners._activity_silent_wm_ping_derisk import run_one as _derisk_run_one
from tools.lab import attributable_to      # force the attribution question: whose is the intact-vs-lesion difference?
from tools.verdict import Verdict           # a verdict must travel with the preconditions that earned it

# focus rotated across seeds (specificity: the ping must recover WHICHEVER was the focus, not a fixed favorite)
_FOCI = ["dog", "cat", "bird"]
# a battery of ordinary turns that MUST be out of scope (judge -> None) so the wiring stays byte-identical when the
# recall query never fires. Includes a D6-style hold-query, which the silent-WM trigger is DISJOINT from.
_OOS_TURNS = [
    "what does it eat?", "tell me about the dog", "the dog and the cat are friends",
    "who are we talking about?", "what are you keeping in mind?", "how are you feeling?",
    "the weather is nice today", "what is two plus two?",
]


def _run_scenario(seed, lesion):
    """Drive the ORGAN through introduce-focus -> distractor -> temporal-recall, return the production `judge` dict."""
    focus = _FOCI[seed % len(_FOCI)]
    others = [f for f in _FOCI if f != focus]
    org = O.ActivitySilentWMOrgan(seed=seed)
    for r in others:                       # earlier discourse referents (distractor referents in other assemblies)
        org.write_referent(r)
    org.write_referent(focus)              # the focus = last-introduced referent, held silently
    org.note_distractor()                  # one intervening distractor turn -> the silent delay
    j = org.judge("what did we start with originally?", lesion=lesion)
    return focus, j


def run_seed(seed):
    focus, ji = _run_scenario(seed, lesion=False)
    _,     jl = _run_scenario(seed, lesion=True)
    # (a) intact recovers the correct focus
    intact_ok = bool(ji is not None and ji.get("recovered") == focus and ji.get("recovered_is_focus"))
    # (b) the hold was genuinely activity-silent (persistent-attractor path suppressed)
    silent_ok = bool(ji is not None and ji.get("silent_delay"))
    # (c) the FAIR facilitation lesion ABSTAINS (recovered None)
    lesion_abstains = bool(jl is not None and jl.get("recovered") is None)
    # (d) the rendered reply genuinely CHANGES (correct anaphor vs abstain)
    reply_intact = ji.get("readout") if ji else None
    reply_lesion = jl.get("readout") if jl else None
    reply_changes = bool(reply_intact and reply_lesion and reply_intact != reply_lesion)
    # (3) flag-off byte-identical: every ordinary turn is out of scope (judge -> None)
    org = O.ActivitySilentWMOrgan(seed=seed)
    org.write_referent(focus)
    oos_ok = all(org.judge(t) is None for t in _OOS_TURNS)
    # (also confirm the de-risk mechanism itself still reads GO on this seed)
    derisk = _derisk_run_one(seed, n_trials=40)
    go = bool(intact_ok and silent_ok and lesion_abstains and reply_changes and oos_ok and derisk["GO"])
    return {
        "seed": seed, "focus": focus,
        "intact_recovered": ji.get("recovered") if ji else None,
        "intact_margin_mean": ji.get("margin_mean") if ji else None,
        "intact_frac_argmax_focus": ji.get("frac_argmax_focus") if ji else None,
        "lesion_recovered": jl.get("recovered") if jl else None,
        "lesion_margin_mean": jl.get("margin_mean") if jl else None,
        "silent_delay": silent_ok,
        "intact_ok": intact_ok, "lesion_abstains": lesion_abstains,
        "reply_changes": reply_changes, "reply_intact": reply_intact, "reply_lesion": reply_lesion,
        "flag_off_byte_identical_oos": oos_ok,
        "derisk_reactivation_acc": derisk["reactivation_acc"],
        "derisk_taufmin_control_acc": derisk["taufmin_control_acc"],
        "derisk_silent_delay": derisk["silent_delay"], "derisk_GO": derisk["GO"],
        "GO": go,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default="research/findings/raw/_activity_silent_wm_production_soak.json")
    a = ap.parse_args()
    rows = [run_seed(s) for s in a.seeds]
    for r in rows:
        print(f"[silent-wm-soak s{r['seed']} focus={r['focus']}] "
              f"INTACT recovered={r['intact_recovered']} (margin={r['intact_margin_mean']:+.1f}, "
              f"argmax-focus={r['intact_frac_argmax_focus']:.2f}) silent={r['silent_delay']} | "
              f"LESION recovered={r['lesion_recovered']} (margin={r['lesion_margin_mean']:+.1f}) abstains={r['lesion_abstains']} | "
              f"reply-changes={r['reply_changes']} | flag-off-OOS={r['flag_off_byte_identical_oos']} | "
              f"derisk={r['derisk_reactivation_acc']:.2f}vs{r['derisk_taufmin_control_acc']:.2f} || "
              f"{'GO' if r['GO'] else 'NO-GO'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    verdict = "GO" if ngo == len(rows) else ("PARTIAL" if ngo else "NO-GO")

    # 6-seed means for the attribution + the verdict preconditions.
    mean_intact_margin = float(np.mean([r["intact_margin_mean"] for r in rows]))
    mean_lesion_margin = float(np.mean([r["lesion_margin_mean"] for r in rows]))
    mean_intact_reactivation = float(np.mean([r["intact_frac_argmax_focus"] for r in rows]))
    all_intact = all(r["intact_ok"] for r in rows)
    all_lesion_abstains = all(r["lesion_abstains"] for r in rows)
    all_silent = all(r["silent_delay"] for r in rows)
    all_reply_changes = all(r["reply_changes"] for r in rows)
    all_oos = all(r["flag_off_byte_identical_oos"] for r in rows)
    all_derisk = all(r["derisk_GO"] for r in rows)

    # ATTRIBUTION (the load-bearing subtraction): what fraction of the intact ping-window reactivation margin is OWNED
    # by the silent FACILITATED hold (INTACT) vs. survives the FAIR facilitation LESION (tau_f~5, excitability-matched)?
    print("attribution (INTACT facilitated hold vs FAIR tau_f-lesion control, 6-seed mean ping margin):")
    attribution = attributable_to("reactivation margin (6-seed mean)", mean_intact_margin, mean_lesion_margin)

    # VERDICT — earned, with the preconditions carried into the artifact.
    v = Verdict("activity-silent WM production wire-in", chance=round(1.0 / O._K, 4))
    v.require("INTACT recovers the correct focus on every seed", all_intact, expect=True)
    v.require("FAIR facilitation lesion (tau_f~5) ABSTAINS on every seed", all_lesion_abstains, expect=True)
    v.require("the delay is genuinely SILENT on every seed (persistent-attractor path suppressed)",
              all_silent, expect=True)
    v.require("the rendered reply CHANGES intact<->lesion on every seed (load-bearing, not hollow)",
              all_reply_changes, expect=True)
    v.require("FLAG-OFF out-of-scope turns return None -> byte-identical on every seed", all_oos, expect=True)
    v.require("the underlying de-risk still reads GO on every seed", all_derisk, expect=True)
    v.control("intact vs FAIR-lesion ping margin (6-seed mean)", mean_intact_margin, mean_lesion_margin,
              min_separation=3.0)
    v.floor("intact reactivation vs chance (1/K)", mean_intact_reactivation, floor=round(1.0 / O._K, 4))
    v.disabled("STDP / Hebbian / homeostasis / reward-modulation / structural-plasticity / NMDA / BDSP / OU",
               "the de-risk isolates the STP facilitation mechanism — anything measured is a property of the "
               "mechanism UNDER THIS ISOLATION (the de-risk's declared scope, imported unchanged)")
    decided = v.decide(go=(ngo == len(rows)))

    print(f"[silent-wm-soak] {ngo}/{len(rows)} GO -> {verdict} (verdict.status={decided['status']}) "
          f"(a nonspecific ping recovers the silently-held anaphora focus across a distractor turn; the FAIR tau_f "
          f"facilitation lesion abstains; the reply changes correct-anaphor<->abstain; silent verified; flag-off "
          f"out-of-scope byte-identical)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"verdict": verdict, "verdict_status": decided["status"], "ngo": ngo, "n": len(rows),
               "chance": round(1.0 / O._K, 4),
               "mean_intact_margin": mean_intact_margin, "mean_lesion_margin": mean_lesion_margin,
               "mean_intact_reactivation": mean_intact_reactivation, "attribution": attribution,
               "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
               "rows": rows}, open(a.out, "w"))


if __name__ == "__main__":
    main()
