"""PURITY #4 / R-A1 de-risk: does giving the OneBrainComposer PRODUCTION-MARGIN codes (the 320 stream-learned
cortex codes) make the spiking K-way cue-match sequencer CLEAR present_ok at SMALL vocab -- where fresh-random
codes give win_rate 0.000?

CONTEXT (the C-2 small-vocab scoping `research/findings/raw/_spiking_sequencer_smallvocab_scoping.md`):
  The spiking sequencer is ALREADY default-ON + GO at the 320 flagship (== host, moat 0-FA). The "small-vocab
  revert" of `integrated_loop` is an UPSTREAM CODE-FIDELITY artifact, NOT a sequencer/threshold failure: at the
  library test config (V~15, K=4, FRESH RANDOM codes, D=128) the per-block matched-filter cleanup of the AGENT
  role produces ZERO firing on >=half the blocks (`win_rate 0.000`), so present_ok stays 0/4-2/4 at EVERY
  match_thresh, the SAFE over-abstention direction (moat 0-FA throughout). The decisive contrast: at 320 the demo
  feeds STREAM-LEARNED cortex codes (wide cleanup margin -> target match-pool rates 0.116-0.196) and the sequencer
  GOes. The lever is CODE MARGIN, and stream-learning RAISES it.

THE R-A1 LEVER (this de-risk -- the ONE real lever, run FIRST):
  Reproduce the small-vocab config but feed the spiking sequencer PRODUCTION-MARGIN codes -- the SAME 320
  stream-learned cortex codes the GO 320 demo uses, SUBSELECTED for a small fact set (the consolidated_320 demo's
  8 child-corpus SVO facts, all 23 words in the 40x8 taxonomy), grounded to D phases by the SAME fixed-complex
  `angle(M @ code)` projection the demo + the step-3 perception arc use. Compare against the fresh-random baseline
  (win_rate 0.000) to PROVE the codes are the lever (NOT the threshold).

THE BARS (the SURPASS move-1 margin read + the agent-level == host + the HARD moat):
  Per seed, per code-source (fresh-random vs production-margin), at the PRODUCTION match_thresh=0.06:
    - margin: the per-cue winner-block match-pool rate (`win_rate`) -- MUST go non-zero on ALL cues for the
      production-margin arm (the lever), and is ~0.000 for the fresh-random arm (the documented baseline);
    - present_ok: does the sequencer pick the SAME block the host first-match does, on all present cues? (== host);
    - moat_0fa: do all absent/cross cues abstain? (the HARD gate, 0 false-accepts -- NEVER relaxed by threshold);
    - agent_eq_host: a FULL agent-level matrix (query_patient / query_agent / ask_yes_no) on the integrated_loop
      composer == the host-_scan oracle (integrated_loop OFF) on the SAME store + the SAME codes.

GPU-only (the OneBrainComposer's on-bridge parser/RF resonate train on the CuPy substrate). Reuse-by-import, NO
sim/ edit. The margin read reuses the EXACT functions the committed `_burndown_1A_c2_smallvocab_derisk.py` uses
(`_seq_imports` -> block_cleanup_scores / make_block_drives / run_sequencerK_reduced_with_drive).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

D = 128  # the production OneBrainComposer phasor dimension (the agent default)

# the consolidated_320 demo's child-corpus SVO facts -- ALL 23 words are in the 40x8 stream taxonomy, so the 320
# stream-learned cortex codes can be subselected for them. agents/actions/patients all distinct so a (agent,action)
# cue uniquely selects one block (the production regime).
FACTS = [
    ("dog", "eat", "apple"),
    ("cat", "play", "ball"),
    ("bird", "sleep", "tree"),
    ("girl", "run", "park"),
    ("boy", "look", "book"),
    ("lion", "eat", "cake"),
    ("rabbit", "jump", "garden"),
    ("mouse", "walk", "house"),
]
PRESENT = [(a, v, i) for i, (a, v, _p) in enumerate(FACTS)]   # present (agent, action) cues -> the correct block idx
# absent/cross (agent, action) cues -- real in-vocab words, never stored together -> MUST abstain (the moat).
ABSENT = [("dog", "sleep"), ("cat", "eat"), ("bird", "run"), ("girl", "look"), ("lion", "jump")]


def _projection(d_out, n_in, seed):
    """The fixed random complex projection n_in -> d_out -- VERBATIM from consolidated_320_conversation_demo."""
    rng = np.random.RandomState(seed * 7919 + 13)
    return (rng.standard_normal((d_out, n_in)) + 1j * rng.standard_normal((d_out, n_in))).astype(np.complex128)


def grounded_phases(code_vec, proj):
    """Real cortex code -> composer phases[D] in [0,1) -- VERBATIM from the demo (the step-3 grounding map)."""
    z = proj @ code_vec.astype(np.complex128)
    return (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)


def _vocab_and_codes(seed, readout="neural"):
    """The small fact vocab + the SUBSELECTED 320 stream-learned codes grounded to D phases for those words."""
    from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
    from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories
    full_vocab, _cat, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    idx = {w: i for i, w in enumerate(full_vocab)}
    words = sorted({w for f in FACTS for w in f})
    suffix = "neural_seed" if readout == "neural" else "seed"
    cpath = os.path.join(_REPO, "research", "findings", "raw", f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
    if not os.path.exists(cpath):
        return words, None, cpath
    codes = np.load(cpath)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    proj = _projection(D, codes.shape[1], seed)
    grounded = {w: grounded_phases(codes[idx[w]], proj) for w in words}     # production-margin code per fact word
    return words, grounded, cpath


def _host_block(c, agent, action):
    """The host first-match block index for (agent, action) -- the oracle the sequencer must match."""
    for i, got in enumerate(c._read_blocks()):
        if got.get("agent") == agent and got.get("action") == action:
            return i
    return None


def _rates_to_array(rates, K):
    return np.asarray([float(rates[f"m{b}"]) for b in range(K)], dtype=float)


def _margins_and_present(c, thresh):
    """The SURPASS move-1 margin read (== _burndown_1A_c2_smallvocab_derisk): for the integrated_loop composer `c`,
    read the per-block sequencer match rates at match_thresh=0.0 once, report each present cue's winner-block
    win_rate + worst off rate, and the present_ok / moat at `thresh`."""
    from research.runners.one_brain_composer import _seq_imports
    fns = _seq_imports()
    K = len(c.kb)
    c._ensure_sequencer(K)
    sb, meta = c._seq
    host_present = {(a, v): _host_block(c, a, v) for (a, v, _i) in PRESENT}
    host_absent = {(a, v): _host_block(c, a, v) for (a, v) in ABSENT}

    per_cue = {}
    for (a, v, _i) in PRESENT:
        if c.enable_seq_vocab_shrink:
            _dec, rates = fns["run_sequencerK_reduced_with_drive"](
                sb, meta, c.words, c._seq_mapA, c._seq_mapX, a, v, c._seq_drives, match_thresh=0.0)
        else:
            _dec, rates = fns["run_sequencerK_with_drive"](
                sb, meta, c._word_index[a], c._word_index[v], c._seq_drives, match_thresh=0.0)
        per_cue[(a, v)] = _rates_to_array(rates, K)

    per_absent = {}
    for (a, v) in ABSENT:
        in_red = (a in (c._seq_mapA or {})) and (v in (c._seq_mapX or {}))
        if not in_red:
            per_absent[(a, v)] = None
            continue
        _dec, rates = fns["run_sequencerK_reduced_with_drive"](
            sb, meta, c.words, c._seq_mapA, c._seq_mapX, a, v, c._seq_drives, match_thresh=0.0)
        per_absent[(a, v)] = _rates_to_array(rates, K)

    margins = {}
    for (a, v, _i) in PRESENT:
        r = per_cue[(a, v)]
        tgt = host_present[(a, v)]
        win = float(r[tgt]) if (tgt is not None and tgt < r.size) else 0.0
        off = float(np.max([r[j] for j in range(r.size) if j != tgt])) if r.size > 1 else 0.0
        margins[f"{a},{v}"] = {"target_block": tgt, "win_rate": round(win, 4), "worst_off_rate": round(off, 4)}

    def decode(rates):
        fired = [j for j in range(rates.size) if rates[j] > thresh]
        return min(fired) if fired else None

    present_ok = sum(1 for (a, v, _i) in PRESENT if decode(per_cue[(a, v)]) == host_present[(a, v)])
    fa = 0
    for (a, v) in ABSENT:
        r = per_absent[(a, v)]
        if r is None:
            continue
        if decode(r) is not None:
            fa += 1
    nonzero_all = all(m["win_rate"] > 0.0 for m in margins.values())
    return {
        "margins": margins, "present_ok": present_ok, "present_tot": len(PRESENT),
        "false_accept": fa, "moat_0fa": fa == 0, "win_rate_nonzero_all": nonzero_all,
        "host_present": {f"{a},{v}": host_present[(a, v)] for (a, v, _i) in PRESENT},
        "host_absent_all_none": all(host_absent[k] is None for k in host_absent),
    }


def _agent_eq_host(seed, grounded, words, thresh):
    """The FULL agent-level matrix: build an integrated_loop (spiking) composer + a host-_scan (oracle) composer on
    the SAME store + codes, and check the spiking answers == the host answers on present cues + the moat abstains on
    absent cues. answer-identical == host AND 0 false-accepts is the GO."""
    from research.runners.one_brain_composer import OneBrainComposer
    spk = OneBrainComposer(seed=seed, D=D, vocab=words, grounded_codes=grounded,
                           integrated_loop=True, sequencer_match_thresh=thresh)
    host = OneBrainComposer(seed=seed, D=D, vocab=words, grounded_codes=grounded, integrated_loop=False)
    for (a, v, p) in FACTS:
        spk.store(a, v, p); host.store(a, v, p)
    present_eq, present_correct, mism = 0, 0, []
    for (a, v, p) in FACTS:
        s_p, h_p = spk.query_patient(a, v), host.query_patient(a, v)
        s_a, h_a = spk.query_agent(v, p), host.query_agent(v, p)
        s_y, h_y = spk.ask_yes_no(a, v, p), host.ask_yes_no(a, v, p)
        if (s_p, s_a, s_y) == (h_p, h_a, h_y):
            present_eq += 1
        else:
            mism.append({"cue": f"{a},{v}", "spk": [s_p, s_a, s_y], "host": [h_p, h_a, h_y]})
        if (s_p, s_a) == (p, a):
            present_correct += 1
    # moat: an absent (agent, action) -> what_does abstains (None); an unstored full SVO -> ask_yes_no != yes.
    fa = 0
    for (a, v) in ABSENT:
        if spk.query_patient(a, v) is not None:
            fa += 1
    for (a, v) in ABSENT:
        if spk.ask_yes_no(a, v, "apple") == "yes":   # a real patient never paired with this (agent,action)
            fa += 1
    del spk, host
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return {"present_eq_host": present_eq, "present_tot": len(FACTS), "present_correct": present_correct,
            "agent_false_accept": fa, "agent_eq_host": present_eq == len(FACTS), "agent_moat_0fa": fa == 0,
            "mismatches": mism}


def run(seeds, thresh, readout):
    from research.runners.one_brain_composer import OneBrainComposer
    out = {"config": {"n_facts": len(FACTS), "D": D, "match_thresh": thresh, "readout": readout, "seeds": seeds},
           "per_seed": []}
    for seed in seeds:
        words, grounded, cpath = _vocab_and_codes(seed, readout=readout)
        row = {"seed": seed, "n_words": len(words), "codes_path": cpath, "codes_found": grounded is not None}
        if grounded is None:
            row["error"] = f"no stream codes at {cpath}; run the 320 stream cortex first"
            out["per_seed"].append(row); continue

        # --- arm A: FRESH-RANDOM codes (the documented baseline -- win_rate ~0.000) ---
        c_fr = OneBrainComposer(seed=seed, D=D, vocab=words, integrated_loop=True,
                                sequencer_match_thresh=thresh)
        for (a, v, p) in FACTS:
            c_fr.store(a, v, p)
        row["fresh_random"] = _margins_and_present(c_fr, thresh)
        del c_fr
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        # --- arm B: PRODUCTION-MARGIN codes (the 320 stream-learned cortex, subselected) ---
        c_pm = OneBrainComposer(seed=seed, D=D, vocab=words, grounded_codes=grounded, integrated_loop=True,
                                sequencer_match_thresh=thresh)
        for (a, v, p) in FACTS:
            c_pm.store(a, v, p)
        row["production_margin"] = _margins_and_present(c_pm, thresh)
        del c_pm
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        # --- the agent-level == host (the GO bar): production-margin spiking composer == host oracle ---
        row["agent_level"] = _agent_eq_host(seed, grounded, words, thresh)
        out["per_seed"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--thresh", type=float, default=0.06)   # the PRODUCTION op-point (NEVER lowered to close the gap)
    ap.add_argument("--readout", choices=["neural", "host"], default="neural")
    ap.add_argument("--out", default="research/findings/raw/_spiking_sequencer_RA1_close.json")
    a = ap.parse_args()
    res = run(a.seeds, a.thresh, a.readout)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[RA1] config K={len(FACTS)} D={D} match_thresh={a.thresh} (the PRODUCTION op-point)\n", flush=True)
    go_seeds = 0
    for sr in res["per_seed"]:
        if not sr.get("codes_found", False):
            print(f"seed {sr['seed']}: SKIP -- {sr.get('error')}", flush=True)
            continue
        fr, pm, ag = sr["fresh_random"], sr["production_margin"], sr["agent_level"]
        print(f"seed {sr['seed']}  (V={sr['n_words']})", flush=True)
        print(f"  FRESH-RANDOM     : win_rates [" + ", ".join(
            f"{m['win_rate']:.3f}" for m in fr["margins"].values()) + f"]  present_ok {fr['present_ok']}/{fr['present_tot']}"
            f"  moat_0fa {fr['moat_0fa']}", flush=True)
        print(f"  PRODUCTION-MARGIN: win_rates [" + ", ".join(
            f"{m['win_rate']:.3f}" for m in pm["margins"].values()) + f"]  present_ok {pm['present_ok']}/{pm['present_tot']}"
            f"  moat_0fa {pm['moat_0fa']}  win_nonzero_all {pm['win_rate_nonzero_all']}", flush=True)
        print(f"  AGENT-LEVEL      : eq_host {ag['present_eq_host']}/{ag['present_tot']} "
              f"correct {ag['present_correct']}/{ag['present_tot']}  agent_moat_0fa {ag['agent_moat_0fa']}", flush=True)
        for m in ag["mismatches"]:
            print(f"      !! mismatch {m}", flush=True)
        # the GO for this seed: production-margin clears present_ok 4/4 + win nonzero + agent == host + both moats 0-FA;
        # AND the fresh-random arm is the (documented) revert (present_ok < 4/4) -- proving the codes are the lever.
        seed_go = (pm["present_ok"] == pm["present_tot"] and pm["win_rate_nonzero_all"] and pm["moat_0fa"]
                   and ag["agent_eq_host"] and ag["agent_moat_0fa"] and ag["present_correct"] == ag["present_tot"]
                   and fr["present_ok"] < fr["present_tot"])
        go_seeds += int(seed_go)
        print(f"  ==> {'GO' if seed_go else 'NOT-GO'} (codes-are-the-lever: "
              f"fresh {fr['present_ok']}/{fr['present_tot']} -> prod {pm['present_ok']}/{pm['present_tot']})", flush=True)
    valid = [sr for sr in res["per_seed"] if sr.get("codes_found", False)]
    print(f"\n[RA1] VERDICT: {go_seeds}/{len(valid)} seeds GO (production-margin codes clear present_ok at the "
          f"production thresh {a.thresh}; fresh-random reverts; agent == host; moat 0-FA both arms).", flush=True)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
