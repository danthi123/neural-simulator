"""A5 CAPABILITY GATE: the false-belief chat wire, driven through the LIVE `FalseBeliefChatOrgan` /
`webapp.false_belief_chat` sentence grammar -- not the standalone `_false_belief_register_derisk` trial runner
directly (that mechanism is already 6/6-seed GO'd; this gate proves the NEW live-conversation ORCHESTRATION
built on top of it reproduces the same anti-cheats through the chat-facing API).

Reuses W3's own controls (`research/runners/_false_belief_register_derisk.py`, `DEFAULT_THRESHOLDS`): a
reality-baseline that must FAIL false-belief, an other-lesion that must collapse it, a scrambled-witnessing
control that must collapse it, and a true-belief-updates-when-witnessed control -- over >=8 change-of-location
items, exactly the derisk's own anti-cheat battery, replayed through `FalseBeliefChatOrgan.observe_event` /
`.query` (which the webapp hook and the LBF row both call) instead of the derisk's own `_run_tom_trial`.

Each item gets a FRESH organ (a fresh bridge build at the SAME `cfg.seed`) -- mirrors a real conversation, where
each new scenario starts a fresh `FalseBeliefChatOrgan` (`webapp/false_belief_chat.py._get_or_start_scenario`),
rather than the derisk's own single-bridge-many-trials-via-snapshot-restore economy. Slower, and a deliberately
faithful choice: speed is secondary to proving the ACTUAL production code path.

Usage (dev seed only -- 7 is NOT one of the project's 6 validation seeds 42/43/44/100/101/102; this is a
de-risk smoke, never a GO):
  bash tools/mem_ok.sh 2 4 && bash tools/memcap.sh 4 -- .venv/bin/python -u \\
      -m research.runners._tom_false_belief_chat_gate --seed 7 --n-items 8 \\
      --json research/findings/raw/_tom_false_belief_chat/smoke_seed7.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import _false_belief_register_derisk as _FB   # the validated thresholds/geometry, reused
from research.runners.tom_false_belief_chat_organ import FalseBeliefChatOrgan
from sim.backend import get_backend
from tools.lab import attributable_to   # the attribution discipline: measuring both arms is not asking WHOSE


def make_items(seed: int, n_items: int):
    """The SAME change-of-location item generator the derisk's `make_trials` uses (identical RNG recipe), so
    this gate's items are drawn the same way the 6/6-seed GO's own trials were."""
    rng = np.random.default_rng(seed * 131 + 5)
    items = []
    for i in range(n_items):
        a = int(rng.integers(_FB.K_LOC))
        b = int(rng.integers(_FB.K_LOC))
        while b == a:
            b = int(rng.integers(_FB.K_LOC))
        w = int(i % 2 == 0)   # balanced witnessed/unwitnessed, shuffled below
        items.append({"start": a, "end": b, "witnessed": w})
    rng.shuffle(items)
    return items


def _run_item(seed: int, start: int, end: int, witnessed_move: bool, *, lesion: bool):
    """ONE fresh `FalseBeliefChatOrgan` (a fresh bridge build): place at `start` (always witnessed), move to
    `end` (witnessed iff `witnessed_move`), then query. Mirrors a real conversation's PLACE -> [LEAVE] -> MOVE
    -> QUERY narration exactly (the organ has no notion of "trials"; this is genuinely the same call sequence
    `webapp/false_belief_chat.observe_turn` makes)."""
    organ = FalseBeliefChatOrgan(seed=seed)
    organ.observe_event(start, witnessed=True, lesion=lesion)
    organ.observe_event(end, witnessed=bool(witnessed_move), lesion=lesion)
    read = organ.query(lesion=lesion)
    return int(read["belief_loc"]), int(read["world_loc"])


def run_block(seed: int, items, witnessed_overrides=None, *, lesion: bool = False):
    pb = np.zeros(len(items), dtype=int)
    pr = np.zeros(len(items), dtype=int)
    for i, it in enumerate(items):
        w = it["witnessed"] if witnessed_overrides is None else int(witnessed_overrides[i])
        pb[i], pr[i] = _run_item(seed, it["start"], it["end"], bool(w), lesion=lesion)
    return pb, pr


def evaluate(seed: int, n_items: int, thresholds, verbose: bool = False):
    items = make_items(seed, n_items)
    gt_belief = np.array([it["start"] if it["witnessed"] == 0 else it["end"] for it in items], dtype=int)
    reality = np.array([it["end"] for it in items], dtype=int)
    false_mask = np.array([it["witnessed"] == 0 for it in items], dtype=bool)
    true_mask = ~false_mask

    t0 = time.time()
    # ---- INTACT (through the live organ, both variants: false-belief items AND true-belief items) ----
    pb, pr = run_block(seed, items, lesion=False)
    false_belief_acc = float(np.mean(pb[false_mask] == gt_belief[false_mask])) if false_mask.any() else None
    true_belief_acc = float(np.mean(pb[true_mask] == gt_belief[true_mask])) if true_mask.any() else None
    reality_baseline_false = float(np.mean(pr[false_mask] == gt_belief[false_mask])) if false_mask.any() else None
    true_belief_agree = float(np.mean(pb[true_mask] == pr[true_mask])) if true_mask.any() else None

    # ---- OTHER-LESION: BRAIN_FALSE_BELIEF_LESION semantics (witnessing gate forced open) ----
    pb_l, _pr_l = run_block(seed, items, lesion=True)
    lesion_false_belief_acc = float(np.mean(pb_l[false_mask] == gt_belief[false_mask])) if false_mask.any() else None
    lesion_collapsed = bool(lesion_false_belief_acc is not None and lesion_false_belief_acc <= thresholds["chance_loc"])

    # ---- SCRAMBLE-WITNESSING: permute which items are "witnessed", score vs the TRUE ground truth ----
    rng = np.random.default_rng(seed * 977 + 19)
    w_flags = np.array([it["witnessed"] for it in items], dtype=int)
    scr_flags = w_flags[rng.permutation(len(items))]
    pb_s, _pr_s = run_block(seed, items, witnessed_overrides=scr_flags, lesion=False)
    scramble_belief_acc = float(np.mean(pb_s == gt_belief))
    scramble_collapsed = bool(scramble_belief_acc <= thresholds["scramble_chance"])

    elapsed = time.time() - t0

    # ── ATTRIBUTION (tools.lab, the standing discipline): both arms were already measured above -- this asks
    #    WHOSE the intact-vs-lesion difference is. treatment = intact (the functioning belief store), control =
    #    lesion (the witnessing gate forced open, the null-lever condition, mirroring the gap#5
    #    attributable_to("lr @ tuned point", tuned, lr=0) pattern). A fraction near 1.0 means the lesion
    #    (not some other confound running identically in both arms) owns the effect; the SAME question is asked
    #    of the scramble control, since a real conversation could confound "scrambled timing" with "no timing
    #    signal at all" the same way a clamp can masquerade as a lever. ─────────────────────────────────────
    attrib_lesion = None
    attrib_scramble = None
    if false_belief_acc is not None and lesion_false_belief_acc is not None:
        attrib_lesion = attributable_to("other-lesion on false-belief accuracy",
                                        false_belief_acc, lesion_false_belief_acc)
    intact_vs_chance = false_belief_acc - (1.0 / _FB.K_LOC) if false_belief_acc is not None else None
    scramble_vs_chance = scramble_belief_acc - (1.0 / _FB.K_LOC)
    if intact_vs_chance is not None:
        attrib_scramble = attributable_to("scrambled-witnessing collapse (above-chance margin)",
                                          intact_vs_chance, scramble_vs_chance)

    result = {
        "seed": int(seed), "n_items": int(n_items),
        "n_false_items": int(false_mask.sum()), "n_true_items": int(true_mask.sum()),
        "chance": 1.0 / _FB.K_LOC,
        "intact": {
            "false_belief_acc": false_belief_acc, "true_belief_acc": true_belief_acc,
            "reality_baseline_false_acc": reality_baseline_false, "true_belief_agree": true_belief_agree,
        },
        "other_lesion": {"false_belief_acc": lesion_false_belief_acc, "collapsed": lesion_collapsed,
                        "attributable_fraction": attrib_lesion},
        "scramble_witnessing": {"belief_acc_vs_true": scramble_belief_acc, "collapsed": scramble_collapsed,
                                "attributable_fraction": attrib_scramble},
        "elapsed_seconds": elapsed,
    }
    if verbose:
        print(f"[tom-fb-gate] seed={seed} n_items={n_items} ({int(false_mask.sum())} false / "
              f"{int(true_mask.sum())} true) chance={1.0/_FB.K_LOC:.2f}", flush=True)
        print(f"[tom-fb-gate]   INTACT   false_belief_acc={false_belief_acc}  true_belief_acc={true_belief_acc}  "
              f"reality_baseline(false)={reality_baseline_false} (must FAIL)  true_belief_agree={true_belief_agree}",
              flush=True)
        print(f"[tom-fb-gate]   LESION   false_belief_acc={lesion_false_belief_acc}  collapsed={lesion_collapsed}",
              flush=True)
        print(f"[tom-fb-gate]   SCRAMBLE belief_acc_vs_true={scramble_belief_acc}  collapsed={scramble_collapsed}  "
              f"elapsed={elapsed:.1f}s", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="A5 false-belief CHAT WIRE capability-gate de-risk (live organ).")
    ap.add_argument("--seed", type=int, default=7, help="dev seed only (never 42/43/44/100/101/102 here)")
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default="research/findings/raw/_tom_false_belief_chat/smoke_seed7.json")
    args = ap.parse_args()

    if args.seed in (42, 43, 44, 100, 101, 102):
        print(f"[tom-fb-gate] REFUSING seed {args.seed}: this is a dev-seed-only smoke; the 6-seed capability "
              f"gate is deferred to B2b at the frozen SHA F per the midnight plan.", flush=True)
        return 2
    if args.backend != "auto":
        get_backend(args.backend)

    print("[tom-fb-gate] A5 false-belief CHAT WIRE de-risk (dev seed only; NOT a GO). Reuses the W3 6/6-seed "
          "GO's own anti-cheats through the live FalseBeliefChatOrgan / webapp.false_belief_chat sentence "
          "grammar, one fresh organ per item.", flush=True)
    result = evaluate(args.seed, args.n_items, _FB.DEFAULT_THRESHOLDS, verbose=True)

    from tools.verdict import Verdict   # noqa: E402
    v = Verdict("A5 false-belief chat wire (live organ, dev-seed smoke)", chance=result["chance"])
    v.require(">=8 items", int(result["n_items"]) >= 8, expect=True)
    v.require("dev seed only (not one of the 6 validation seeds)", args.seed not in (42, 43, 44, 100, 101, 102),
              expect=True)
    it = result["intact"]
    if it["false_belief_acc"] is not None:
        v.floor("false-belief acc vs chance (live organ)", it["false_belief_acc"], result["chance"])
    if it["reality_baseline_false_acc"] is not None:
        v.require("reality-baseline FAILS false-belief", it["reality_baseline_false_acc"],
                  expect=lambda x: x <= _FB.DEFAULT_THRESHOLDS["reality_baseline_max"],
                  note="a world-read that predicted reality would pass without any belief representation")
    if it["true_belief_agree"] is not None:
        v.require("true-belief control: belief updates when witnessed", it["true_belief_agree"],
                  expect=lambda x: x >= _FB.DEFAULT_THRESHOLDS["true_belief_acc"])
    if it["false_belief_acc"] is not None and result["other_lesion"]["false_belief_acc"] is not None:
        v.control("other-lesion collapses the read", treatment=it["false_belief_acc"],
                  control=result["other_lesion"]["false_belief_acc"])
    v.require("other-lesion collapsed to <= chance_loc bar", result["other_lesion"]["collapsed"], expect=True)
    v.require("scrambled witnessing collapsed", result["scramble_witnessing"]["collapsed"], expect=True)
    verdict_block = v.decide(go=bool(
        result["other_lesion"]["collapsed"] and result["scramble_witnessing"]["collapsed"]
        and it["false_belief_acc"] is not None and it["false_belief_acc"] >= _FB.DEFAULT_THRESHOLDS["false_belief_acc"]
    ))
    result["verdict_block"] = verdict_block
    result["honest_scope"] = ("A dev-seed (7) smoke of the LIVE chat-facing organ/parser, reusing the already "
                              "6/6-seed-GO'd W3 register's own anti-cheats. Never a GO; the 6-seed capability "
                              "gate over 42/43/44/100/101/102 runs at the frozen SHA F in B2b per the plan. "
                              "Witnessing/presence is host-parsed; the action read is a host argmax (see the "
                              "PRE-REGISTRATION).")

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[tom-fb-gate] === {verdict_block['status']} === wrote {args.json}", flush=True)
    return 0 if verdict_block["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
