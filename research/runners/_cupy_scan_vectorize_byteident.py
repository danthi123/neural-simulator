"""Byte-identity proof for the #108 cupy-latency fix: the vectorized winner-code gather in
`RFPhasorComposer._escalate_role_match` (research/runners/rf_phasor_composer.py, ~line 986, marked
`VECTORIZED_WINCODE_GATHER`) must produce answers + selected fact indices IDENTICAL to the original
per-candidate Python-loop `np.stack` it replaced.

THE HOTSPOT (diagnosed research/findings/2026-09-02-escalation-gating-tighten-latency-correctness-safe-
not-the-lever.md): `win_codes = np.stack([self.concepts[words[i]] for i in cand])` -- a Python loop doing
one dict lookup + one backend-array touch per near-tie candidate, then np.stack. Cheap on numpy, but on
cupy each `self.concepts[w]` in the loop pays a per-element sync -- the driver of the ~1303ms #108 R1
cupy median (target <1000ms). THE FIX: build the (V,D) codebook + a word->row map ONCE per vocab state
(`_ensure_codebook_cache`, already used elsewhere in this file for the SAME reason), then gather the m
candidate rows with ONE vectorized fancy-index `self._cb_frac[row_idx]` instead of the per-candidate loop.

THIS SCRIPT proves the fix is byte-identical to the pre-fix behavior by running the REAL recall path
against the shipped 100k wikidata bundle (numpy backend -- fast, no GPU needed) with
`enable_decode_escalation=True` (the flag that makes `_escalate_role_match` fire at all) for seeds
42/43/44, TWICE per seed on the SAME already-loaded agent:
  PASS "original": `RFPhasorComposer._escalate_role_match` monkeypatched to a byte-for-byte reproduction
    of the pre-fix method (verified against `git diff` at the time this script was written -- see the
    docstring-adjacent comment on `_escalate_role_match_ORIGINAL` below).
  PASS "current": the REAL, unmodified method as it ships in the file (no reproduction risk on this side).
Both passes run the IDENTICAL probe set in the IDENTICAL order against the SAME loaded composer state (a
recall query is read-only -- no RNG advance, no cache/self.kb mutation -- so re-running the same cues
under two different `_escalate_role_match` implementations is a valid apples-to-apples comparison).

Two independent checks per seed:
  (1) decoded answers: `inner.composer.query_patient(agent, action)` for every probe + moat cue.
  (2) selected fact indices: `RFPhasorComposer._scan_first_match` is wrapped (a passthrough capture, not a
      behavior change -- installed once, active across both passes) to log its return value per call; the
      two passes' index sequences must match position-for-position.
Plus, for seed 44 specifically, the exact documented near-tie cue
(`query_patient("berkeley_county_virginia", "located_in_the_administrative_territoria")`, expected
`culture_of_west_virginia`, research/findings/2026-09-01-seed44-recall-hole-ROOT-CAUSED-phase-
quantization-decode-escalation-fix.md) is added explicitly, so the escalation branch is provably EXERCISED
(not just present-but-inert on the random sample).

Run:  .venv/bin/python -m research.runners._cupy_scan_vectorize_byteident \
        --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k \
        --seeds 42,43,44 --out research/findings/raw/_cupy_scan_vectorize/byteident.txt
(SIM_BACKEND unset/numpy -- this is a CORRECTNESS proof, not the cupy latency measurement, which is
queued separately on tools/gpu_queue.sh per the finding this fix closes.)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

# HARD-FORCE numpy: this is the correctness proof, and the GPU may be busy with another brain-loading
# proc (owner directive -- never run cupy from this script). Must be set BEFORE any sim.bridge import
# below triggers backend auto-detection (auto = cupy-if-available), so a bare env-var *check* is not
# enough -- an unset SIM_BACKEND would silently auto-select cupy if it's importable, exactly the bug this
# script hit once during development (caught via nvidia-smi showing an unexpected cupy alloc).
os.environ["SIM_BACKEND"] = "numpy"

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.developed_brain_io import load_developed_brain, _inner_agent
from research.runners._knowledge_scale_100k_production_verify import (
    _load_bundle_facts_vocab, _first_match, _make_production_brain,
)

# Keep the REAL (current, post-fix) method around before any monkeypatching touches the class.
_ESCALATE_CURRENT = RFPhasorComposer._escalate_role_match
_SCAN_FIRST_MATCH_REAL = RFPhasorComposer._scan_first_match


def _escalate_role_match_ORIGINAL(self, rec, comps, role, val, words, role_mask, prior_mask):
    """Byte-for-byte reproduction of the PRE-FIX `_escalate_role_match` (the per-candidate Python-loop
    `np.stack` gather this task replaces). Reproduced from `git diff` taken at fix time -- the only line
    that differs from the current file is the `win_codes = np.stack(...)` assignment; everything else
    (the moat guard, the near-tie gate, the finer-readout confirm loop) is copied verbatim."""
    if not isinstance(val, str) or val not in self.concepts:
        return role_mask
    cand = np.where(prior_mask & ~role_mask)[0]
    if len(cand) == 0:
        return role_mask
    rec_arr = np.asarray(rec)
    val_code = self.concepts[val]
    s_val = np.cos(2.0 * np.pi * (rec_arr[cand] - val_code[None, :])).mean(axis=1)
    win_codes = np.stack([self.concepts[words[i]] for i in cand])   # THE ORIGINAL per-candidate loop+stack
    s_win = np.cos(2.0 * np.pi * (rec_arr[cand] - win_codes)).mean(axis=1)
    near = cand[(s_win - s_val) <= self.decode_escalate_margin]
    for i in near:
        fine = self._unbind_phases(np.asarray(comps[i]), role, period=self.decode_escalate_period)
        if self._cleanup(fine) == val:
            role_mask[i] = True
    return role_mask


_SCAN_IDX_LOG = []


def _scan_first_match_CAPTURING(self, **cue_roles):
    """Passthrough wrapper around the REAL `_scan_first_match` -- calls it unchanged and logs its return
    value (the selected fact index, or None) so the two passes' index sequences can be diffed. Installed
    once for BOTH passes (it does not touch `_escalate_role_match`, so it is behavior-neutral)."""
    idx = _SCAN_FIRST_MATCH_REAL(self, **cue_roles)
    _SCAN_IDX_LOG.append((dict(cue_roles), idx))
    return idx


def _run_pass(inner, probes, moat_cues, extra_cues, label):
    """Run every probe + moat cue + extra cue through the live agent, return (answers list, scan-idx log
    copy). `probes`/`moat_cues`/`extra_cues` are (agent, action[, gt]) tuples; order is fixed across both
    passes by construction (same lists, iterated once each)."""
    global _SCAN_IDX_LOG
    _SCAN_IDX_LOG = []
    answers = []
    for (a, v, *_gt) in probes:
        answers.append(("probe", a, v, inner.composer.query_patient(a, v)))
    for (a, v) in moat_cues:
        answers.append(("moat", a, v, inner.composer.query_patient(a, v)))
    for (a, v, *_gt) in extra_cues:
        answers.append(("extra", a, v, inner.composer.query_patient(a, v)))
    return answers, list(_SCAN_IDX_LOG)


def _run_seed(seed, bundle, n_probes, n_moat, tmp_root, log):
    log(f"\n=== seed {seed} ===")
    mani, raw_facts = _load_bundle_facts_vocab(bundle)
    D = int(mani["D"])
    fm = _first_match(raw_facts)
    keys = list(fm.keys())

    tmp = tempfile.mkdtemp(prefix="cupy_scan_vec_byteident_", dir=tmp_root)
    brain_dir, _conv_facts = _make_production_brain(seed, D, tmp)
    agent, _load_manifest = load_developed_brain(
        brain_dir, ltm_bundle=bundle, use_multiturn=False, seed=seed,
        enable_codebook_cache=True, enable_decode_escalation=True, decode_escalate_margin=None,
    )
    inner = _inner_agent(agent)

    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(len(keys), size=min(n_probes, len(keys)), replace=False)
    probes = [(keys[i][0], keys[i][1], fm[keys[i]]) for i in idx]

    rng2 = np.random.default_rng(seed + 999)
    unknown_agents = [f"zzz_unknown_entity_{j}_xq" for j in range(n_moat // 2)]
    known_agent_sample = keys[int(rng2.integers(0, len(keys)))][0]
    real_actions = sorted({f.get("action") for f in raw_facts if isinstance(f.get("action"), str)})
    moat_cues = [(ua, real_actions[int(rng2.integers(0, len(real_actions)))]) for ua in unknown_agents]
    moat_cues += [(known_agent_sample, "zzz_unknown_relation_never_taught")
                  for _ in range(n_moat - len(moat_cues))]

    # Explicitly exercise the seed-44 documented near-tie escalation (root-caused 2026-09-01) so the
    # comparison is proven to run through the escalation branch at least once, not just be present-but-inert.
    extra_cues = []
    if seed == 44:
        extra_cues.append(("berkeley_county_virginia", "located_in_the_administrative_territoria",
                            "culture_of_west_virginia"))

    log(f"    bundle: n_facts={mani['n_facts']} vocab={len(mani['vocab'])} n_shards={mani['n_shards']}")
    log(f"    probes={len(probes)} moat={len(moat_cues)} extra={len(extra_cues)}")

    # --- PASS 1: ORIGINAL (pre-fix) _escalate_role_match ---
    RFPhasorComposer._escalate_role_match = _escalate_role_match_ORIGINAL
    RFPhasorComposer._scan_first_match = _scan_first_match_CAPTURING
    t0 = time.time()
    answers_orig, scanlog_orig = _run_pass(inner, probes, moat_cues, extra_cues, "original")
    t_orig = time.time() - t0

    # --- PASS 2: CURRENT (post-fix, vectorized) _escalate_role_match -- the REAL shipped method ---
    RFPhasorComposer._escalate_role_match = _ESCALATE_CURRENT
    t0 = time.time()
    answers_new, scanlog_new = _run_pass(inner, probes, moat_cues, extra_cues, "current")
    t_new = time.time() - t0

    # restore _scan_first_match to the real implementation (defensive; next seed rebuilds anyway)
    RFPhasorComposer._scan_first_match = _SCAN_FIRST_MATCH_REAL

    n_answer_mismatches = sum(1 for o, n in zip(answers_orig, answers_new) if o != n)
    n_scanidx_mismatches = sum(1 for o, n in zip(scanlog_orig, scanlog_new) if o != n)
    n_escalate_calls = 0  # informational: count near-tie gate fires by diffing role_mask sums is out of
    # scope here; the scan-index log length match + content match already proves identical selection.

    identical = (len(answers_orig) == len(answers_new) == len(probes) + len(moat_cues) + len(extra_cues)
                 and n_answer_mismatches == 0
                 and len(scanlog_orig) == len(scanlog_new)
                 and n_scanidx_mismatches == 0)

    log(f"    PASS original: {len(answers_orig)} answers in {t_orig:.2f}s, {len(scanlog_orig)} scan calls")
    log(f"    PASS current : {len(answers_new)} answers in {t_new:.2f}s, {len(scanlog_new)} scan calls")
    log(f"    answer mismatches: {n_answer_mismatches}/{len(answers_orig)}")
    log(f"    scan-index mismatches: {n_scanidx_mismatches}/{len(scanlog_orig)}")
    if seed == 44:
        extra_ans = [a for a in answers_new if a[0] == "extra"]
        log(f"    seed-44 explicit near-tie cue result: {extra_ans}")
    if n_answer_mismatches:
        for o, n in zip(answers_orig, answers_new):
            if o != n:
                log(f"      MISMATCH answer: original={o} current={n}")
    if n_scanidx_mismatches:
        for o, n in zip(scanlog_orig, scanlog_new):
            if o != n:
                log(f"      MISMATCH scan-idx: original={o} current={n}")
    log(f"    VERDICT seed {seed}: {'IDENTICAL' if identical else 'DIVERGED'}")
    return {
        "seed": seed,
        "n_probes": len(probes), "n_moat": len(moat_cues), "n_extra": len(extra_cues),
        "n_answers_compared": len(answers_orig),
        "n_answer_mismatches": n_answer_mismatches,
        "n_scanidx_compared": len(scanlog_orig),
        "n_scanidx_mismatches": n_scanidx_mismatches,
        "seed44_near_tie_cue_answer": (
            [a[3] for a in answers_new if a[0] == "extra"][0] if seed == 44 and extra_cues else None),
        "t_pass_original_s": round(t_orig, 3), "t_pass_current_s": round(t_new, 3),
        "identical": identical,
    }


def main():
    ap = argparse.ArgumentParser(description=(
        "Byte-identity proof: vectorized _escalate_role_match winner-code gather vs the pre-fix "
        "per-candidate Python-loop stack (board #108 cupy latency fix)."))
    ap.add_argument("--bundle", default="/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-probes", type=int, default=15)
    ap.add_argument("--n-moat", type=int, default=10)
    ap.add_argument("--out", default="research/findings/raw/_cupy_scan_vectorize/byteident.txt")
    ap.add_argument("--json-out", default=None,
                   help="structured per-seed results (default: --out with .txt replaced by .json)")
    a = ap.parse_args()
    json_out = a.json_out or (a.out[:-4] + ".json" if a.out.endswith(".txt") else a.out + ".json")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    from sim.backend import get_backend
    _xp, _backend_name = get_backend()
    log("Byte-identity proof: VECTORIZED_WINCODE_GATHER vs pre-fix per-candidate loop+stack")
    log(f"bundle={a.bundle}")
    log(f"SIM_BACKEND env={os.environ.get('SIM_BACKEND')} | actual resolved backend={_backend_name}")
    assert _backend_name != "cupy", (
        "This is the numpy correctness proof -- refusing to run on cupy (the GPU may be busy with another "
        "brain-loading proc). The cupy LATENCY re-verify is queued separately on tools/gpu_queue.sh, "
        "guarded on the VECTORIZED_WINCODE_GATHER marker.")

    tmp_root = tempfile.mkdtemp(prefix="cupy_scan_vec_byteident_root_")
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed = {}
    t_start = time.time()
    for seed in seeds:
        per_seed[seed] = _run_seed(seed, a.bundle, a.n_probes, a.n_moat, tmp_root, log)

    all_identical = all(r["identical"] for r in per_seed.values())
    elapsed = round(time.time() - t_start, 1)
    log(f"\n=== SUMMARY ({elapsed}s) ===")
    for seed, r in per_seed.items():
        log(f"  seed {seed}: {'IDENTICAL' if r['identical'] else 'DIVERGED'}")
    log(f"\nOVERALL: {'ALL SEEDS IDENTICAL (byte-identity PROVEN)' if all_identical else 'DIVERGENCE DETECTED -- DO NOT SHIP'}")

    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")

    import json
    from tools.verdict import Verdict

    n_ans_mismatch = sum(r["n_answer_mismatches"] for r in per_seed.values())
    n_scanidx_mismatch = sum(r["n_scanidx_mismatches"] for r in per_seed.values())
    n_ans_compared = sum(r["n_answers_compared"] for r in per_seed.values())
    n_scanidx_compared = sum(r["n_scanidx_compared"] for r in per_seed.values())

    v = Verdict("cupy-scan-vectorize byte-identity (vectorized vs pre-fix _escalate_role_match gather)")
    v.require("ran on numpy (not cupy -- GPU may be busy with another brain-loading proc)",
              _backend_name, expect="numpy")
    v.require("decoded answers identical (vectorized vs original) across all probed cues",
              n_ans_mismatch, expect=0)
    v.require("selected fact indices identical (_scan_first_match) across all probed cues",
              n_scanidx_mismatch, expect=0)
    if 44 in per_seed:
        v.require("seed-44 near-tie escalation branch actually exercised (not present-but-inert)",
                  per_seed[44].get("seed44_near_tie_cue_answer"), expect="culture_of_west_virginia")
    verdict_dict = v.decide(go=all_identical)

    summary = {
        "bundle": a.bundle, "seeds": seeds, "backend": _backend_name,
        "n_seeds_total_answer_mismatches": n_ans_mismatch,
        "n_seeds_total_scanidx_mismatches": n_scanidx_mismatch,
        "n_seeds_total_answers_compared": n_ans_compared,
        "n_seeds_total_scanidx_compared": n_scanidx_compared,
        "per_seed": per_seed,
        "all_identical": all_identical,
        "elapsed_s": elapsed,
        **verdict_dict,   # status, go, undefined_reasons, preconditions, disabled_processes, chance, label
    }
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {json_out}")
    return 0 if all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
