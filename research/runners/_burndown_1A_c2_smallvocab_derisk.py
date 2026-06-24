"""BURNDOWN 1A / C-2 de-risk: the small-vocab `integrated_loop` over-abstention.

The 2026-06-23 spiking-default-migration PARTIAL reverted the `integrated_loop` agent-default because at the SMALL
test vocab the spiking K-way sequencer at the production op-point (match_thresh=0.06) OVER-ABSTAINS -> the agent tests
(`test_one_brain_composer_agent.py`) get `what_does -> None` where they should recall. The moat stayed INTACT (over-
abstention is the SAFE direction, 0 false-accepts). This probe ISOLATES the over-abstention mechanism + tests the
cheapest fold+scale fix (a config-aware match_thresh toward the measured no-match floor) -- per the C-2 deployment
scoping's R0 (the threshold/drive margin re-cal, the cheapest untried retreat).

It mirrors the agent-test config EXACTLY (the same VOCAB + the same 4 stored facts of test_onebrain_agent_matrix_and_moat)
and, for each match_thresh in a sweep, reports:
  - the present-cue answer-identity (does the sequencer pick the SAME block the host first-match does?),
  - the moat (do absent/cross cues still abstain? -- the HARD gate, 0 false-accepts),
  - the per-cue winner-block match rate vs the worst no-match rate (the margin that the fixed threshold sits in).

GPU-only (the OneBrainComposer's on-bridge parser trains on the CuPy substrate). Reuse-by-import, NO sim/ edit.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")

# the EXACT agent-test config (tests/test_one_brain_composer_agent.py)
VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"),
         ("river", "stop", "west")]  # the 4 the test stores (the 4th via a passive frame -> agent=river)
# present (agent, action) cues -> the correct block index; and absent/cross cues -> must abstain (the moat).
PRESENT = [("dog", "go", 0), ("cat", "come", 1), ("bird", "look", 2), ("river", "stop", 3)]
ABSENT = [("apple", "stop"), ("dog", "stop"), ("cat", "go"), ("river", "go")]   # unstored (agent, action) pairs


def _host_block(c, agent, action):
    """The host first-match block index for (agent, action) -- the oracle the sequencer must match."""
    for i, got in enumerate(c._read_blocks()):
        if got.get("agent") == agent and got.get("action") == action:
            return i
    return None


def _rates_to_array(rates, K):
    """Both run_sequencerK_with_drive + run_sequencerK_reduced_with_drive return `rates` as a DICT
    {'m0':r0, ..., 'm{K-1}':r_{K-1}, 'winner':w}; this de-risk needs the per-block float array (length K) so it can
    re-threshold the cached firing rates. Extract m0..m{K-1} in order (the original `np.asarray(rates)` failed because
    a dict is not array-able -- the de-risk runner's reduced-path bug, fixed minimally here)."""
    return np.asarray([float(rates[f"m{b}"]) for b in range(K)], dtype=float)


def run(seeds, threshs, D=128):
    from research.runners.one_brain_composer import OneBrainComposer
    out = {"config": {"vocab_size": len(VOCAB), "n_facts": len(FACTS), "D": D,
                      "threshs": threshs, "seeds": seeds}, "per_seed": []}
    for seed in seeds:
        # one ON composer (integrated_loop) we re-query at different thresholds (the fabric is threshold-agnostic;
        # only the decode rule's `rate > match_thresh` cut changes -> we can read the rates once + re-threshold).
        c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, integrated_loop=True)
        for (a, v, p) in FACTS:
            c.store(a, v, p)
        # the host oracle blocks (the answer-identity target) -- read once.
        host_present = {(a, v): _host_block(c, a, v) for (a, v, _i) in PRESENT}
        host_absent = {(a, v): _host_block(c, a, v) for (a, v) in ABSENT}   # expect all None (the moat)

        # raw per-cue rates from the sequencer (threshold-independent): run the reduced fabric once per cue and read
        # the block match rates, so a single GPU pass covers the whole threshold sweep.
        from research.runners.one_brain_composer import _seq_imports
        fns = _seq_imports()
        K = len(c.kb)
        c._ensure_sequencer(K)
        sb, meta = c._seq
        per_cue = {}
        for (a, v, _i) in PRESENT:
            if c.enable_seq_vocab_shrink:
                dec, rates = fns["run_sequencerK_reduced_with_drive"](
                    sb, meta, c.words, c._seq_mapA, c._seq_mapX, a, v, c._seq_drives,
                    match_thresh=0.0)  # 0.0 -> read the raw decision/rates; we re-threshold below
            else:
                dec, rates = fns["run_sequencerK_with_drive"](
                    sb, meta, c._word_index[a], c._word_index[v], c._seq_drives, match_thresh=0.0)
            per_cue[(a, v)] = _rates_to_array(rates, K)
        # absent cues: the moat. An absent (agent, action) where the WORD is in-vocab still drives the fabric; an
        # out-of-reduced-vocab cue is caught before the sequencer (returns None). We read rates where the cue words
        # are in the reduced maps (the genuine moat stress -- a real word that matches no stored block).
        per_absent = {}
        for (a, v) in ABSENT:
            in_red = (a in (c._seq_mapA or {})) and (v in (c._seq_mapX or {}))
            if not in_red:
                per_absent[(a, v)] = None   # caught before the fabric (abstains structurally)
                continue
            dec, rates = fns["run_sequencerK_reduced_with_drive"](
                sb, meta, c.words, c._seq_mapA, c._seq_mapX, a, v, c._seq_drives, match_thresh=0.0)
            per_absent[(a, v)] = _rates_to_array(rates, K)

        # now evaluate each threshold by re-applying the first-match-over-threshold rule to the cached rates.
        seed_rows = {"seed": seed, "host_present": {f"{a},{v}": host_present[(a, v)] for (a, v, _i) in PRESENT},
                     "host_absent_all_none": all(host_absent[k] is None for k in host_absent),
                     "by_thresh": []}
        # diagnostic: per-present-cue winner rate + worst off-block rate (the margin the threshold sits in).
        margins = {}
        for (a, v, i) in PRESENT:
            r = per_cue[(a, v)]
            tgt = host_present[(a, v)]
            win = float(r[tgt]) if (tgt is not None and tgt < r.size) else 0.0
            off = float(np.max([r[j] for j in range(r.size) if j != tgt])) if r.size > 1 else 0.0
            margins[f"{a},{v}"] = {"target_block": tgt, "win_rate": round(win, 4), "worst_off_rate": round(off, 4)}
        seed_rows["margins"] = margins

        for th in threshs:
            # the first-match-over-threshold decode (== run_sequencerK_*_with_drive at match_thresh=th -> decision_to_block)
            def decode(rates):
                fired = [j for j in range(rates.size) if rates[j] > th]
                return min(fired) if fired else None
            present_ok = 0
            for (a, v, i) in PRESENT:
                blk = decode(per_cue[(a, v)])
                if blk == host_present[(a, v)]:
                    present_ok += 1
            fa = 0
            for (a, v) in ABSENT:
                r = per_absent[(a, v)]
                if r is None:
                    continue   # structurally abstained
                blk = decode(r)
                if blk is not None:
                    fa += 1   # a moat breach: an unstored cue routed to a block
            seed_rows["by_thresh"].append({
                "match_thresh": th, "present_ok": present_ok, "present_tot": len(PRESENT),
                "false_accept": fa, "answer_identical": present_ok == len(PRESENT),
                "moat_0fa": fa == 0,
            })
        out["per_seed"].append(seed_rows)
        del c
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--threshs", type=float, nargs="+",
                    default=[0.06, 0.04, 0.02, 0.01, 0.005, 0.002])
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--out", default="research/findings/raw/_burndown_1A_c2_smallvocab_derisk.json")
    a = ap.parse_args()
    res = run(a.seeds, a.threshs, D=a.D)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[c2-smallvocab] config V={len(VOCAB)} K={len(FACTS)} D={a.D}\n", flush=True)
    for sr in res["per_seed"]:
        print(f"seed {sr['seed']}  host_absent_all_none={sr['host_absent_all_none']}", flush=True)
        print(f"  margins: " + " | ".join(
            f"{k}: win {m['win_rate']:.3f} off {m['worst_off_rate']:.3f}" for k, m in sr["margins"].items()),
            flush=True)
        for row in sr["by_thresh"]:
            tag = "OK" if (row["answer_identical"] and row["moat_0fa"]) else (
                "MOAT_BREACH" if not row["moat_0fa"] else "OVER_ABSTAIN")
            print(f"  thresh {row['match_thresh']:.4f}: present {row['present_ok']}/{row['present_tot']} "
                  f"| FA {row['false_accept']}  ==> {tag}", flush=True)
    # the verdict: the smallest thresh (>= the no-match-floor safety) that is answer-identical with moat 0-FA on ALL seeds
    best = None
    for th in a.threshs:
        all_ok = all(any(r["match_thresh"] == th and r["answer_identical"] and r["moat_0fa"]
                         for r in sr["by_thresh"]) for sr in res["per_seed"])
        if all_ok:
            best = th  # keep the LAST (smallest in the default descending sweep) that works on all seeds
    print(f"\n[c2-smallvocab] VERDICT: smallest all-seed answer-identical + moat-0FA thresh = {best}", flush=True)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
