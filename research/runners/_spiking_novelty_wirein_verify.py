"""FOCUSED wire-in verify for the spiking-habituation NOVELTY retirement (scaffold-retirement, 2026-09-09).

This proves the TWO things a guarded, additive, default-OFF production wire-in must show, WITHOUT running the full
integrated brain-chat no-regression battery (RAM-blocked on the dev box; the integrated `SIM_BACKEND=cupy` soak is
DEFERRED to an AWS-CPU batch -- see the finding + the command printed at the bottom of this module's docstring):

  (A) BYTE-IDENTICAL WHEN OFF (light, NO brain). With `BRAIN_SPIKING_NOVELTY` unset/0, `engagement_of()` must produce
      the EXACT pre-wiring value on a handful of messages. Checked two ways: (A1) `spiking_novelty_enabled()` is False
      by default; (A2) `engagement_of(tokens, seen)` == the pre-wiring reference formula (`_W_NOVELTY*novelty_host +
      (1-_W_NOVELTY)*richness`) recomputed inline, max|diff| == 0 over the message set; (A3) the explicit
      `novelty_override=None` call equals the no-kwarg call (the default path is the host path).

  (B) LOAD-BEARING WHEN ON (focused, small spiking organ; per the 6 project-standard seeds). Build the spiking
      habituation organ and show: novel words read FRESH (L1); repeating them HABITUATES the read (L2, the de-risk's
      immediate-ratio<=0.70 shape); brand-new words recruit a FRESH channel and read novel again (L3, recruit-on-demand
      + no cross-talk); engagement TRACKS the spiking novelty (L4, engagement_of high-vs-low novelty_override differs
      with tokens/seen held fixed); and the STP LESION REVERTS the habituation (L5, lesioned repeat-ratio>=0.80 AND
      attributable_to(intact_drop, lesion_drop)>=0.5 -- the load-bearing proof that synaptic depression, not some
      other artifact, produces the novelty signal).

Per-seed GO(B) = L1 and L2 and L3 and L4 and L5. Board GO = (A all pass) AND (>=5/6 seeds GO(B)).

Run (numpy-CPU, focused, ~1-2 min total; small organ, well under the RAM limits that motivate the queue):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._spiking_novelty_wirein_verify \\
      --seeds 42 43 44 100 101 102 \\
      --out research/findings/raw/_spiking_novelty_wirein/verify_6seed.json

The DEFERRED integrated no-regression verify (for the parent to run on an AWS-CPU batch, NOT run here):
  SIM_BACKEND=cupy BRAIN_SPIKING_NOVELTY=1 .venv/bin/python -u -m research.runners.onebrain_regression_battery \\
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_spiking_novelty_wirein/integrated_noregr_6seed.json
  (must show: content fields byte-identical vs BRAIN_SPIKING_NOVELTY=0, and da_drives/engagement present + sane.)
"""
from __future__ import annotations

import os
import sys
import json
import argparse

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict       # noqa: E402

# a small bank keeps the focused check fast; the production default is 64 (eviction rare per turn)
N_CHANNELS = 12
WORDS_A = ["alpha", "bravo", "charlie", "delta"]
WORDS_B = ["epsilon", "zeta", "eta", "theta"]
N_REPEAT = 4


# ============================== (A) byte-identical when OFF (no brain) ==============================
def check_byte_identical_off() -> dict:
    """(A) With the flag OFF (default), `engagement_of()` == the pre-wiring reference formula, exactly."""
    # ensure the default (unset) is treated as OFF for this check
    os.environ.pop("BRAIN_SPIKING_NOVELTY", None)
    from webapp import da_mode_drives_chat as C

    a1_default_off = (C._NOV.spiking_novelty_enabled() is False)

    # a message set with mixed novelty vs a growing `seen` set (mirrors a real conversation)
    msgs = [
        "the quantum harmonic oscillator resonates strongly",
        "the quantum oscillator resonates again here now",
        "photosynthesis converts sunlight into chemical energy",
        "tell me more about photosynthesis and sunlight",
        "",
        "a b the is are",   # all function/short words -> no content tokens -> 0.0
    ]
    seen = set()
    max_abs_diff = 0.0
    a3_override_none_matches = True
    rows = []
    for m in msgs:
        toks = C._content_tokens(m)
        # pre-wiring REFERENCE formula, recomputed inline (this is exactly what engagement_of did before the wire-in)
        if not toks:
            ref = 0.0
        else:
            nov_host = sum(1 for t in toks if t not in seen) / float(len(toks))
            rich = min(len(toks) / float(C._RICHNESS_FULL), 1.0)
            ref = float(np.clip(C._W_NOVELTY * nov_host + (1.0 - C._W_NOVELTY) * rich, 0.0, 1.0))
        got = C.engagement_of(toks, seen)                       # default (no kwarg) -> host path
        got_none = C.engagement_of(toks, seen, novelty_override=None)   # explicit None -> host path
        max_abs_diff = max(max_abs_diff, abs(got - ref))
        if got_none != got:
            a3_override_none_matches = False
        rows.append({"msg": m, "n_content": len(toks), "reference": ref, "engagement_of": got,
                     "engagement_of_override_none": got_none})
        for t in toks:
            seen.add(t)   # advance `seen` exactly as observe() does (after computing novelty)

    a2_byte_identical = (max_abs_diff == 0.0)
    return {"A1_default_is_off": bool(a1_default_off),
            "A2_byte_identical_off": bool(a2_byte_identical),
            "A2_max_abs_diff": float(max_abs_diff),
            "A3_override_none_matches_default": bool(a3_override_none_matches),
            "A_all_pass": bool(a1_default_off and a2_byte_identical and a3_override_none_matches),
            "rows": rows}


# ============================== (B) load-bearing when ON (small spiking organ) ==============================
def _organ_novelty_sequence(seed: int, lesion: bool) -> dict:
    """Build a fresh organ and run the load-bearing protocol; return the key novelty reads."""
    from research.runners.spiking_novelty_habituation_organ import SpikingNoveltyHabituationOrgan
    organ = SpikingNoveltyHabituationOrgan(seed=seed, n_channels=N_CHANNELS, lesion=lesion)
    nov_first = organ.novelty_of(list(WORDS_A))["novelty"]                 # fresh words -> should read novel
    reps = [organ.novelty_of(list(WORDS_A))["novelty"] for _ in range(N_REPEAT)]  # repeat -> habituate
    nov_rep = float(np.mean(reps[1:3]))                                    # presentations 2-3 (de-risk's window)
    nov_newfresh = organ.novelty_of(list(WORDS_B))["novelty"]             # brand-new words -> recruit fresh channel
    return {"nov_first": float(nov_first), "nov_rep": float(nov_rep), "nov_newfresh": float(nov_newfresh),
            "reps": [float(x) for x in reps]}


def run_seed_B(seed: int) -> dict:
    from webapp import da_mode_drives_chat as C
    intact = _organ_novelty_sequence(seed, lesion=False)
    lesion = _organ_novelty_sequence(seed, lesion=True)

    ratio_intact = intact["nov_rep"] / max(intact["nov_first"], 1e-9)
    ratio_lesion = lesion["nov_rep"] / max(lesion["nov_first"], 1e-9)

    # L4: engagement TRACKS the spiking novelty (tokens+seen held FIXED; only novelty_override varies).
    seen_fixed = set()
    e_high = C.engagement_of(list(WORDS_A), seen_fixed, novelty_override=intact["nov_first"])
    e_low = C.engagement_of(list(WORDS_A), seen_fixed, novelty_override=intact["nov_rep"])

    l1 = bool(intact["nov_first"] >= 0.70)                      # novel words read fresh
    l2 = bool(ratio_intact <= 0.70)                            # repeating habituates the read
    l3 = bool(intact["nov_newfresh"] >= 0.70)                  # brand-new words recruit a fresh channel (no cross-talk)
    l4 = bool((e_high - e_low) >= 0.05)                        # engagement tracks the spiking novelty
    # L5: the STP lesion REVERTS the habituation.
    drop_intact = 1.0 - ratio_intact
    drop_lesion = 1.0 - ratio_lesion
    attrib = attributable_to("stp_habituation_novelty", drop_intact, drop_lesion, warn_below=0.5)
    l5 = bool(ratio_lesion >= 0.80 and attrib >= 0.5)

    go = bool(l1 and l2 and l3 and l4 and l5)
    return {"seed": seed,
            "nov_first": round(intact["nov_first"], 4), "nov_rep": round(intact["nov_rep"], 4),
            "nov_newfresh": round(intact["nov_newfresh"], 4), "ratio_intact": round(ratio_intact, 4),
            "lesion_nov_first": round(lesion["nov_first"], 4), "lesion_nov_rep": round(lesion["nov_rep"], 4),
            "ratio_lesion": round(ratio_lesion, 4), "attribution_to_stp": round(float(attrib), 4),
            "engagement_high": round(float(e_high), 4), "engagement_low": round(float(e_low), 4),
            "L1_novel_fresh": l1, "L2_habituates": l2, "L3_recruit_fresh": l3,
            "L4_engagement_tracks": l4, "L5_lesion_reverts": l5, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="research/findings/raw/_spiking_novelty_wirein/verify.json")
    a = ap.parse_args()

    A = check_byte_identical_off()
    print(f"[novelty-wirein A] default_off={A['A1_default_is_off']} byte_identical_off={A['A2_byte_identical_off']} "
          f"(max|diff|={A['A2_max_abs_diff']:.2e}) override_none_matches={A['A3_override_none_matches_default']} "
          f"|| A_all_pass={A['A_all_pass']}", flush=True)

    rowsB = [run_seed_B(s) for s in a.seeds]
    for r in rowsB:
        print(f"[novelty-wirein B s{r['seed']}] first={r['nov_first']:.3f} rep={r['nov_rep']:.3f} "
              f"newfresh={r['nov_newfresh']:.3f} ratio={r['ratio_intact']:.3f} | lesion_ratio={r['ratio_lesion']:.3f} "
              f"attrib={r['attribution_to_stp']:.3f} | e_hi={r['engagement_high']:.3f} e_lo={r['engagement_low']:.3f} "
              f"|| {'GO' if r['GO'] else 'no'}", flush=True)

    ngo = sum(r["GO"] for r in rowsB)
    n = len(rowsB)
    v = Verdict("spiking-novelty wire-in (A byte-identical-off AND >=5/6 seeds load-bearing-on)")
    v.require("A: byte-identical when off (all A checks pass)", A["A_all_pass"], expect=True)
    v.require("all 6 project-standard seeds present for (B)", n == 6, expect=True, note="seeds run: %d" % n)
    v.floor("seed-GO count (B)", measured=float(ngo), floor=4.5, note="board bar >=5/6 -> floor=4.5 excludes 4/6")
    decided = v.decide(go=(A["A_all_pass"] and n == 6 and ngo >= 5))
    verdict = decided["status"]
    print(f"[novelty-wirein] A_all_pass={A['A_all_pass']}; B {ngo}/{n} seed-GO (bar>=5/6) || verdict={verdict}",
          flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {"A_byte_identical_off": A, "B_load_bearing_rows": rowsB, "n_go_B": ngo, "n_seeds": n,
               "verdict": verdict}
    payload.update(decided)
    json.dump(payload, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
