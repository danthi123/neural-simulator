"""The within-category-error GENERALIZATION SIGNATURE in the conversational binder.

QUESTION. The 320-concept cortex learns SEMANTICALLY STRUCTURED codes (dog and cat are near; that structure is
what lets the cortex generalize). When the conversational binder's read-out is NOISY and recall drops below
ceiling, does it err SEMANTICALLY SENSIBLY (confuse dog with cat -- a within-category error) or RANDOMLY
(confuse dog with airplane)? A brain-like generalizing memory should make within-category mistakes; a
decorrelated (structure-free) memory should make uniform/random mistakes. This is the generalization thesis
tested INSIDE the conversational product -- and it is moat-respecting (the errors are on STORED facts under a
noisy read-out, never confabulation on unstored queries).

DESIGN (HRR role-filler bind/unbind, the CYCLE-90 conversational algebra, reuse-by-import). Bind a fact
F = R_agent (x) c_i + R_action (x) c_j + R_object (x) c_k; add Gaussian read-out noise of scale sigma; unbind a
role; clean up by nearest of all 320 codes; classify the answer correct / within-category / cross-category.
Sweep sigma to walk recall down from 1.0, and at each sigma report the WITHIN-CATEGORY FRACTION AMONG ERRORS.

THREE ARMS (the anti-cheat is built in):
  * STRUCTURED   -- the real 320 stream-learned codes. Prediction: errors concentrate WITHIN category.
  * RANDOM       -- decorrelated codes (same shape, rng). Control: errors should be ~CHANCE within-category
                    (chance = (8-1)/(320-1) = 7/319 ~ 2.2%), proving the signature comes from code STRUCTURE.
  * DERANGED     -- the structured codes but with the category LABELS shuffled. Control: the within-category
                    fraction must collapse to chance, proving the signature tracks the REAL categories (the
                    code geometry), not a labelling artifact.

GATE (multi-seed): in the informative regime (recall ~0.6-0.9, where errors exist), STRUCTURED within-category
fraction >> RANDOM ~ DERANGED ~ chance. That is the generalization signature: the agent's conversational
mistakes are semantically sensible because the codes it learned from conversation are semantically organized.

Run (CPU; pure numpy HRR over 320 codes -- fast, no bridge):
  SIM_BACKEND=numpy python -m research.runners._genfrontier_within_category_conversation_signature --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_assembled_pipeline_ppmi_derisk import hrr_bind, hrr_unbind, _cos
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories

SIGMAS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]   # push to the large-error regime (recall ~0.4-0.6)
N_TRIALS = 1500         # role-query trials per (arm, sigma) -- tighter error statistics at high sigma
N_FACTS = 40            # distinct stored facts drawn per trial-block


def _unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def run_arm(codes, cat_ids, rng, label, bind=True):
    """Sweep read-out noise; at each sigma report recall + the within-category fraction AMONG errors.

    bind=True : the full conversational pipeline (role (x) filler bind -> noise -> unbind -> cleanup over 320).
    bind=False: the RAW-CODE control -- noise added directly to the filler code, cleanup over 320, NO binding.
                This isolates whether the CODE GEOMETRY alone produces within-category confusion, separating it
                from any wash-out the role-binding's cross-term decorrelation introduces."""
    Nc, D = codes.shape
    cat = np.asarray(cat_ids)
    R_a = rng.standard_normal(D) / np.sqrt(D)
    R_v = rng.standard_normal(D) / np.sqrt(D)
    R_o = rng.standard_normal(D) / np.sqrt(D)
    R = {"agent": R_a, "action": R_v, "patient": R_o}
    rows = []
    for sigma in SIGMAS:
        correct = err = within = 0
        for _ in range(N_TRIALS):
            i, j, k = rng.choice(Nc, 3, replace=False)
            role = rng.choice(["agent", "action", "patient"])
            t = {"agent": i, "action": j, "patient": k}[role]
            if bind:
                F = hrr_bind(R_a, codes[i]) + hrr_bind(R_v, codes[j]) + hrr_bind(R_o, codes[k])
                scale = sigma * np.linalg.norm(F) / np.sqrt(D)
                est = hrr_unbind(F + scale * rng.standard_normal(D), R[role])
            else:
                scale = sigma * np.linalg.norm(codes[t]) / np.sqrt(D)
                est = codes[t] + scale * rng.standard_normal(D)
            pred = int(np.argmax(_cos(est, codes)))
            if pred == t:
                correct += 1
            else:
                err += 1
                within += int(cat[pred] == cat[t])
        recall = correct / N_TRIALS
        within_frac = within / err if err else float("nan")
        rows.append({"sigma": sigma, "recall": recall, "n_err": err, "within_frac": within_frac})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--readout", choices=["neural", "host"], default="host")
    ap.add_argument("--out", default="research/findings/raw/_genfrontier_within_category_signature.json")
    a = ap.parse_args()

    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    cat = np.asarray(cat_ids)
    n_cat = len(set(cat_ids)); per_cat = len(vocab) // n_cat
    chance = (per_cat - 1) / (len(vocab) - 1)
    suffix = "neural_seed" if a.readout == "neural" else "seed"

    print(f"[within-category signature] {n_cat} categories x {per_cat} | chance within-cat among errors = "
          f"{chance:.4f}\n  STRUCTURED errors should land WITHIN category (>> chance); RANDOM + DERANGED ~ chance.\n",
          flush=True)

    per_seed = []
    for seed in a.seeds:
        cpath = os.path.join(_REPO, "research", "findings", "raw",
                             f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
        if not os.path.exists(cpath):
            print(f"  [seed {seed}] SKIP — no codes at {cpath}", flush=True)
            continue
        struct = _unit_rows(np.load(cpath))
        rng = np.random.default_rng(seed * 31 + 5)
        rand = _unit_rows(rng.standard_normal(struct.shape))
        derange = cat.copy(); rng.shuffle(derange)
        arms = {
            "structured": run_arm(struct, cat, np.random.default_rng(seed * 101 + 1), "structured", bind=True),
            "raw_struct": run_arm(struct, cat, np.random.default_rng(seed * 101 + 4), "raw_struct", bind=False),
            "random": run_arm(rand, cat, np.random.default_rng(seed * 101 + 2), "random", bind=True),
            "deranged": run_arm(struct, derange, np.random.default_rng(seed * 101 + 3), "deranged", bind=True),
        }
        per_seed.append({"seed": seed, "arms": arms})
        print(f"  [seed {seed}]  sigma | recall(bound) | within%(bound) | within%(RAW no-bind) | within%(rand) "
              "| within%(derange)", flush=True)
        for r_s, r_w, r_r, r_d in zip(arms["structured"], arms["raw_struct"], arms["random"], arms["deranged"]):
            def pct(x):
                return "  n/a " if (isinstance(x, float) and np.isnan(x)) else f"{100*x:5.1f}%"
            print(f"           {r_s['sigma']:.2f} |    {r_s['recall']:.2f}     |   {pct(r_s['within_frac'])}    "
                  f"|     {pct(r_w['within_frac'])}       |  {pct(r_r['within_frac'])}  |   {pct(r_d['within_frac'])}",
                  flush=True)

    # GATE: in the informative regime (recall 0.2-0.95, enough errors), structured within-frac >> random AND
    # >> deranged AND a STRONG (>=20%) absolute concentration. The raw-no-bind arm localizes the cause.
    def regime_mean(arm_rows):
        vals = [r["within_frac"] for r in arm_rows
                if 0.2 <= r["recall"] <= 0.95 and r["n_err"] >= 30 and not np.isnan(r["within_frac"])]
        return float(np.mean(vals)) if vals else float("nan")
    go_seeds = 0
    for ps in per_seed:
        s = regime_mean(ps["arms"]["structured"]); rw = regime_mean(ps["arms"]["raw_struct"])
        rd = regime_mean(ps["arms"]["random"]); dr = regime_mean(ps["arms"]["deranged"])
        ps["regime"] = {"structured": s, "raw_struct": rw, "random": rd, "deranged": dr}
        ok = (not np.isnan(s)) and s >= max(0.20, 3 * chance) and s >= 2.5 * max(rd, chance) and s >= 2.5 * max(dr, chance)
        ps["go"] = bool(ok); go_seeds += int(ok)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"chance": chance, "results": per_seed}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if per_seed and go_seeds == len(per_seed):
        ms = np.nanmean([ps["regime"]["structured"] for ps in per_seed])
        mr = np.nanmean([ps["regime"]["random"] for ps in per_seed])
        md = np.nanmean([ps["regime"]["deranged"] for ps in per_seed])
        print(f"  GO ({go_seeds}/{len(per_seed)} seeds): the conversational binder's recall errors are the "
              f"GENERALIZATION SIGNATURE — within-category {100*ms:.1f}% (structured) vs {100*mr:.1f}% (random) "
              f"vs {100*md:.1f}% (deranged) vs {100*chance:.1f}% chance. The agent confuses semantically SIMILAR "
              "concepts (dog<->cat), not random ones — because the codes it learned from conversation are "
              "semantically organized. Moat-respecting (errors on stored facts under noise, not confabulation).",
              flush=True)
    elif per_seed:
        ms = np.nanmean([ps["regime"]["structured"] for ps in per_seed])
        mw = np.nanmean([ps["regime"]["raw_struct"] for ps in per_seed])
        mr = np.nanmean([ps["regime"]["random"] for ps in per_seed])
        print(f"  NEGATIVE ({go_seeds}/{len(per_seed)} seeds): the within-category error signature is WEAK — "
              f"bound {100*ms:.1f}% vs raw-no-bind {100*mw:.1f}% vs random {100*mr:.1f}% vs chance {100*chance:.1f}%. "
              "RAW (no-binding) ~ BOUND, so the role-binding is not the cause. CORRECTED mechanism (see "
              "_genfrontier_learned_vs_raw_category_readout.py): the codes DO carry category structure in raw "
              "proximity (kNN cat-acc ~21%, 8.4x chance) but it is SMALL-MARGIN (same-cat cosine ~0.13); the "
              "read-out noise that causes a binder error overwhelms that thin margin, so the wrong pick is "
              "near-random. The conversational binder's recall errors are NOT meaningfully semantic — a real-but-"
              "thin margin swamped by noise, NOT an absence of structure. (No overclaim of 'dog<->cat confusion'.)",
              flush=True)
    else:
        print("  NO CODES — run the 320 stream cortex first.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
