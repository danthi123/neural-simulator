"""FOCUSED wire-in verify for the spiking CA3 pattern-completion anaphor-DETECTION retirement (scaffold-retirement,
2026-09-09). Proves the TWO things a guarded, additive, default-OFF production wire-in must show, WITHOUT running the
full integrated brain-chat no-regression battery (RAM-blocked on the dev box; the integrated `SIM_BACKEND=cupy` soak is
DEFERRED to an AWS-CPU batch -- see the finding + the command printed at the bottom of this docstring):

  (A) BYTE-IDENTICAL WHEN OFF (light, NO brain). With `BRAIN_SPIKING_ANAPHOR` unset/0, BOTH wired call-site detection
      helpers must return EXACTLY the pre-wiring host `set` result on a representative token set. Checked:
      (A1) `spiking_anaphor_enabled()` is False by default; (A2) `ChatBrain._is_anaphor_token(stub, tl, anaphors)` ==
      `tl in anaphors` for every token (max mismatch == 0); (A3) `MultiTurnAgent._anaphor_is(stub, w)` ==
      `w.lower() in _ANAPHORS` for every token (max mismatch == 0). With the flag off both helpers short-circuit before
      touching `self` or the substrate, so the only change to `_resolve_anaphora`/`_resolve` is a call that returns the
      identical boolean -> the two call sites are byte-identical to pre-wiring.

  (B) LOAD-BEARING WHEN ON (focused, small spiking organ; per the 6 project-standard seeds). Build the detection organ
      and show, on the SUBSTRATE: (L1) every known anaphor is DETECTED on a clean cue; (L2) SPECIFICITY -- content words
      are not falsely detected (they short-circuit to a cue the de-risk's G3 proved cannot ignite); (L3) the de-risked
      SURPASS -- a CORRUPTED cue (only 20% of a stored assembly's neurons, the de-risk's G2) still COMPLETES to the
      correct anaphor (a token an exact `x in {...}` cannot recognise at all); (L4) the LESION REVERTS it -- with the
      attractor weights removed, clean detection collapses AND corrupted-cue completion collapses, and
      attributable_to(intact_detect, lesion_detect) >= 0.5 (the load-bearing proof that the recurrent CA3 attractor,
      not the host encoding, does the recognition).

Per-seed GO(B) = L1 and L2 and L3 and L4. Board GO = (A all pass) AND (>=5/6 seeds GO(B)).

Run (numpy-CPU, focused, a few minutes; small circuit, well under the RAM limits that motivate the queue):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._spiking_anaphor_wirein_verify \\
      --seeds 42 43 44 100 101 102 \\
      --out research/findings/raw/_spiking_anaphor_wirein/verify_6seed.json

The DEFERRED integrated no-regression verify (for the parent to run on an AWS-CPU batch, NOT run here):
  SIM_BACKEND=cupy BRAIN_SPIKING_ANAPHOR=1 .venv/bin/python -u -m research.runners.onebrain_regression_battery \\
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_spiking_anaphor_wirein/integrated_noregr_6seed.json
  (must show: the anaphora-resolved turn content byte-identical vs BRAIN_SPIKING_ANAPHOR=0 on clean text, and the
   spiking detection present + sane; the FLIP to default-ON waits on that verdict.)
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

# representative token set for (A): the 5 anaphors + near-neighbours + plain content/function words.
_ANAPHOR_TOKENS = ["it", "that", "they", "them", "this"]
_NON_ANAPHOR_TOKENS = ["cat", "dog", "fish", "hello", "the", "what", "then", "than", "his", "running", "water", ""]
_CONTENT_WORDS = ["cat", "dog", "fish", "hello", "banana", "house", "running", "water"]   # for (B) L2 specificity
_KEEP_FRAC = 0.20         # the de-risk's decisive corrupted-cue keep-fraction (80% of the assembly corrupted)
_N_NOISY = 3              # corrupted-cue draws per anaphor (averaged) for L3


# ============================== (A) byte-identical when OFF (no brain) ==============================
def check_byte_identical_off() -> dict:
    """(A) With the flag OFF (default), BOTH wired detection helpers == the pre-wiring host `set` result, exactly."""
    os.environ.pop("BRAIN_SPIKING_ANAPHOR", None)                 # ensure default (unset) == OFF for this check
    import research.runners.spiking_anaphor_detection_organ as _ANAPH
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.multi_turn_agent import MultiTurnAgent, _ANAPHORS

    a1_default_off = (_ANAPH.spiking_anaphor_enabled() is False)

    stub = object()                                              # OFF path never touches `self`
    tui_anaphors = {"it", "that", "they", "them", "this"}        # ChatBrain._resolve_anaphora's own literal set
    tui_mismatch = 0
    agent_mismatch = 0
    rows = []
    tokens = list(dict.fromkeys(_ANAPHOR_TOKENS + _NON_ANAPHOR_TOKENS + ["IT", "That?", "Them.", "  it "]))
    for t in tokens:
        tl = t.lower().strip(".,!?")                             # the same normalization _resolve_anaphora applies
        tui_ref = tl in tui_anaphors
        tui_got = ChatBrain._is_anaphor_token(stub, tl, tui_anaphors)
        agent_ref = isinstance(t, str) and (t.lower() in _ANAPHORS)
        agent_got = isinstance(t, str) and MultiTurnAgent._anaphor_is(stub, t)
        if bool(tui_got) != bool(tui_ref):
            tui_mismatch += 1
        if bool(agent_got) != bool(agent_ref):
            agent_mismatch += 1
        rows.append({"token": t, "tui_ref": bool(tui_ref), "tui_got": bool(tui_got),
                     "agent_ref": bool(agent_ref), "agent_got": bool(agent_got)})

    a2_tui = (tui_mismatch == 0)
    a3_agent = (agent_mismatch == 0)
    return {"A1_default_is_off": bool(a1_default_off),
            "A2_tui_byte_identical_off": bool(a2_tui), "A2_tui_mismatch": int(tui_mismatch),
            "A3_agent_byte_identical_off": bool(a3_agent), "A3_agent_mismatch": int(agent_mismatch),
            "A_all_pass": bool(a1_default_off and a2_tui and a3_agent), "n_tokens": len(tokens), "rows": rows}


# ============================== (B) load-bearing when ON (small spiking organ) ==============================
def _clean_detect_acc(organ, anaphors) -> float:
    return float(np.mean([1.0 if organ.is_anaphor(a) else 0.0 for a in anaphors]))


def _fp_rate(organ, words) -> float:
    return float(np.mean([1.0 if organ.is_anaphor(w) else 0.0 for w in words]))


def _corrupted_recovery(organ, anaphors, keep_frac, n_noisy, base_seed) -> float:
    accs = []
    for a in anaphors:
        for r in range(n_noisy):
            rng = np.random.default_rng(base_seed * 131 + hash(a) % 997 + r)
            accs.append(1.0 if organ.probe_corrupted_cue(a, keep_frac=keep_frac, rng=rng)["recovered"] else 0.0)
    return float(np.mean(accs)) if accs else 0.0


def run_seed_B(seed: int) -> dict:
    from research.runners.spiking_anaphor_detection_organ import SpikingAnaphorDetectorOrgan
    intact = SpikingAnaphorDetectorOrgan(seed=seed, lesion=False)
    lesion = SpikingAnaphorDetectorOrgan(seed=seed, lesion=True)

    clean_intact = _clean_detect_acc(intact, _ANAPHOR_TOKENS)              # L1
    fp = _fp_rate(intact, _CONTENT_WORDS)                                  # L2
    corrupt_intact = _corrupted_recovery(intact, _ANAPHOR_TOKENS, _KEEP_FRAC, _N_NOISY, seed)   # L3
    clean_lesion = _clean_detect_acc(lesion, _ANAPHOR_TOKENS)              # L4 (clean collapse)
    corrupt_lesion = _corrupted_recovery(lesion, _ANAPHOR_TOKENS, _KEEP_FRAC, 1, seed)          # L4 (corrupt collapse)

    l1 = bool(clean_intact >= 0.90)                                        # every anaphor detected on a clean cue
    l2 = bool(fp <= 0.15)                                                  # content words not falsely detected (G3)
    l3 = bool(corrupt_intact >= 0.85)                                      # corrupted cue still completes (G2 surpass)
    # L4: lesion reverts -- clean AND corrupt detection collapse, and the intact/lesion gap is attributable.
    intact_combined = float(np.mean([clean_intact, corrupt_intact]))
    lesion_combined = float(np.mean([clean_lesion, corrupt_lesion]))
    attrib = attributable_to("anaphor CA3 pattern-completion (attractor weight)", intact_combined, lesion_combined,
                             warn_below=0.5)
    l4 = bool(lesion_combined <= 0.30 and attrib is not None and attrib >= 0.5)

    go = bool(l1 and l2 and l3 and l4)
    return {"seed": seed,
            "clean_detect_intact": round(clean_intact, 4), "false_positive_rate": round(fp, 4),
            "corrupted_recovery_intact": round(corrupt_intact, 4),
            "clean_detect_lesion": round(clean_lesion, 4), "corrupted_recovery_lesion": round(corrupt_lesion, 4),
            "intact_combined": round(intact_combined, 4), "lesion_combined": round(lesion_combined, 4),
            "attribution_to_attractor": None if attrib is None else round(float(attrib), 4),
            "L1_clean_detect": l1, "L2_specificity": l2, "L3_corrupted_surpass": l3, "L4_lesion_reverts": l4,
            "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="research/findings/raw/_spiking_anaphor_wirein/verify.json")
    a = ap.parse_args()

    A = check_byte_identical_off()
    print(f"[anaphor-wirein A] default_off={A['A1_default_is_off']} tui_off={A['A2_tui_byte_identical_off']} "
          f"(mismatch={A['A2_tui_mismatch']}) agent_off={A['A3_agent_byte_identical_off']} "
          f"(mismatch={A['A3_agent_mismatch']}) || A_all_pass={A['A_all_pass']}", flush=True)

    rowsB = [run_seed_B(s) for s in a.seeds]
    for r in rowsB:
        print(f"[anaphor-wirein B s{r['seed']}] clean={r['clean_detect_intact']:.3f} fp={r['false_positive_rate']:.3f} "
              f"corrupt={r['corrupted_recovery_intact']:.3f} | lesion_clean={r['clean_detect_lesion']:.3f} "
              f"lesion_corrupt={r['corrupted_recovery_lesion']:.3f} attrib={r['attribution_to_attractor']} "
              f"|| {'GO' if r['GO'] else 'no'}", flush=True)

    ngo = sum(r["GO"] for r in rowsB)
    n = len(rowsB)
    v = Verdict("spiking-anaphor-detection wire-in (A byte-identical-off AND >=5/6 seeds load-bearing-on)")
    v.require("A: byte-identical when off (all A checks pass)", A["A_all_pass"], expect=True)
    v.require("all 6 project-standard seeds present for (B)", n == 6, expect=True, note="seeds run: %d" % n)
    v.floor("seed-GO count (B)", measured=float(ngo), floor=4.5, note="board bar >=5/6 -> floor=4.5 excludes 4/6")
    decided = v.decide(go=(A["A_all_pass"] and n == 6 and ngo >= 5))
    verdict = decided["status"]
    print(f"[anaphor-wirein] A_all_pass={A['A_all_pass']}; B {ngo}/{n} seed-GO (bar>=5/6) || verdict={verdict}",
          flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {"A_byte_identical_off": A, "B_load_bearing_rows": rowsB, "n_go_B": ngo, "n_seeds": n,
               "verdict": verdict}
    payload.update(decided)
    json.dump(payload, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
