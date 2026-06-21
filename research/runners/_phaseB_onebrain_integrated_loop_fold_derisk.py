"""Shortcut #3 FOLD de-risk: the production `OneBrainComposer(integrated_loop=True)` (the spiking K-way sequencer
routing the (agent, action) cue-match) is ANSWER-IDENTICAL to the host-`_scan` oracle (`integrated_loop=False`) on the
full who/what matrix INCLUDING every `is None`/`"unknown"` abstention, the no-confab MOAT holds 0-false-accept, at
production scale (K up to 32, D=128, V=72 / V=320), multi-seed -- with the anti-cheats (sequencer-LESION fails safe,
permuted-rule inverts, the NO-DIVNORM raw control fails).

This is the COMPOSER-API-level re-assertion of the Stage S2 K=32 capability surpass
(`2026-06-21-shortcut3-K32-capability-surpass.md`: eq_n 3/3 + fa_total 0 at match_thresh 0.06): the surpass validated
the sequencer FABRIC against `host_scan_block`; THIS runner validates the FOLD -- the production composer's
`query_patient` / `ask_yes_no` / `update_on_mismatch` answers are identical with integrated_loop ON vs OFF, end to end.

GATES (GO), at K in {2,4,8,16,32}, D=128, 3+ seeds:
  * ANSWER-IDENTITY: c_seq.<method>(...) == c_host.<method>(...) for every query in the battery (present who/what +
    yes-no + the `is None`/`"unknown"` abstentions + reconsolidation abstain) -- the host is the oracle.
  * MOAT (HARD, never traded): fa_total == 0 -- no absent/cross cue selects a block on the integrated_loop path.
  * ANTI-CHEATS (on the composer's own built sequencer fabric): LESION fails safe (sever the result->op drive ->
    abstain), permuted-rule INVERTS, the NO-DIVNORM raw control fails (the divnorm is load-bearing).

NO `sim/` edit (reuse-by-import: OneBrainComposer + the S0/S2/S5 sequencer fabric). An honest NEGATIVE (a single
answer mismatch or a single false-accept at match_thresh 0.06) is a valid deliverable -- report it, never raise the
threshold to mask it.

  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_integrated_loop_fold_derisk --seeds 42,43,44 --dim 128 --ks 2,4,8
  SIM_BACKEND=cupy  python -u -m research.runners._phaseB_onebrain_integrated_loop_fold_derisk --seeds 42,43,44 --dim 128 --ks 2,4,8,16,32   # 320-scale: + --vocab-320
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import numpy as np

from sim.backend import is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer, _seq_imports


def _free_gpu_memory():
    """Release the per-iteration bridge memory so the (seed, K) loop does NOT accumulate.

    ROOT CAUSE this guards (2026-06-21): at V=320/K=32 each `OneBrainComposer(integrated_loop=True)` lazily builds an
    O(K*V) spiking sequencer fabric (~41761 regions / 836830 neurons / 21M synapses). Holding a composer is cheap
    (~0.78GB VRAM / ~1.5GB host) but the loop built one per iteration and NEVER freed: the CuPy mempool retains freed
    blocks and the 21M-synapse build's transient host structures linger, so memory grew linearly across the
    ks x seeds grid until host RAM was exhausted and the OS silently killed the process mid-build (the dead-log
    signature: progressively larger bridges, then gone, no traceback). Calling this after each iteration drops the
    only references (the composers are local to run_seed_K) + gc.collect()s the host structures + returns the device
    mempool blocks, so the steady-state peak is ONE K=32 composer (fits 24GB trivially). NO `sim/` edit -- runner-side
    teardown only. Inert on the numpy-CPU path (no cupy mempool)."""
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
# the K=32 production fact set (32 distinct facts, unique (agent, action) cues, 8 actions x 4 agents) + the V=72 vocab.
from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import (
    ALL_FACTS, VOCAB, _build_queries,
)


def _build_pair(seed, D, K, vocab, grounded_codes=None, match_thresh=0.06, gain=0.11, sigma=1.0, input_gain=1.0):
    """Two composers on the SAME facts/codes: c_host (integrated_loop=False = the oracle) and c_seq
    (integrated_loop=True = the spiking K-way routing). Both on the same backend (numpy build -> cupy under cupy)."""
    facts = ALL_FACTS[:K]
    kw = dict(seed=seed, D=D, vocab=vocab, k_max=max(32, K), enable_batched=False, enable_rf_cudagraph=False,
              grounded_codes=grounded_codes)
    c_host = OneBrainComposer(integrated_loop=False, **kw)
    c_seq = OneBrainComposer(integrated_loop=True, sequencer_match_thresh=match_thresh, sequencer_gain=gain,
                             sequencer_sigma=sigma, sequencer_input_gain=input_gain, **kw)
    for (a, x, p) in facts:
        c_host.store(a, x, p); c_seq.store(a, x, p)
    return c_host, c_seq, facts


def run_seed_K(seed, D, K, vocab, grounded_codes=None, match_thresh=0.06, gain=0.11, sigma=1.0, input_gain=1.0):
    """Answer-identity + moat + anti-cheats for one seed at store size K, through the production composer API."""
    c_host, c_seq, facts = _build_pair(seed, D, K, vocab, grounded_codes, match_thresh, gain, sigma, input_gain)
    queries = _build_queries(facts)

    rows = []
    for (qa, qx), kind in queries:
        # query_patient: the routed (agent, action) -> patient (or None = abstain)
        h_qp = c_host.query_patient(qa, qx)
        s_qp = c_seq.query_patient(qa, qx)
        # ask_yes_no: a present cue asks the TRUE SVO (-> yes) for present rows; the moat rows ask the cross/absent SVO
        if kind.endswith("present"):
            patient = next(p for (a, x, p) in facts if a == qa and x == qx)
        else:
            patient = facts[0][2]                              # an arbitrary patient on an unstored cue -> unknown
        h_yn = c_host.ask_yes_no(qa, qx, patient)
        s_yn = c_seq.ask_yes_no(qa, qx, patient)
        eq = (h_qp == s_qp) and (h_yn == s_yn)
        rows.append(dict(cue=(qa, qx), kind=kind, host_patient=h_qp, seq_patient=s_qp,
                         host_yes_no=h_yn, seq_yes_no=s_yn, eq=eq))
    answer_identical = all(r["eq"] for r in rows)

    # the MOAT (HARD): every absent/cross cue must abstain on the SEQ path (query_patient None AND ask_yes_no unknown).
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    fa = sum(1 for r in moat_rows if (r["seq_patient"] is not None) or (r["seq_yes_no"] != "unknown"))
    moat_ok = (fa == 0)

    # reconsolidation abstain (via _find_cued_block, the routed spiking decision): a never-stored cue abstains == host.
    a0, x0 = facts[0][0], facts[0][1]
    absent_agent = next(w for w in vocab if w not in {a for (a, x, p) in facts})
    rm_h = c_host.update_on_mismatch(absent_agent, x0, facts[0][2])
    rm_s = c_seq.update_on_mismatch(absent_agent, x0, facts[0][2])
    reconsolidation_ok = (rm_h["action"] == rm_s["action"] == "abstain")

    # --- anti-cheats on the composer's OWN built sequencer fabric (the same sb/meta/drives _seq_block uses) ---
    fns = _seq_imports()
    c_seq._ensure_sequencer(K)                                 # ensure the fabric + drives are built
    sb, meta = c_seq._seq
    drives = c_seq._seq_drives
    word_idx = c_seq._word_index
    # LESION: sever the result->op drive on every present cue -> must abstain (fail safe).
    les = [fns["run_sequencerK_with_drive"](sb, meta, word_idx[a], word_idx[x], drives, lesion=True,
                                            match_thresh=match_thresh)[0] for (a, x, p) in facts]
    lesion_fails_safe = all(d == "abstain" for d in les)
    # PERMUTED-RULE: a present cue for block i must route to ans{(i+1)%K} (the decision follows the rule).
    perm_ok = True
    for i, (a, x, p) in enumerate(facts):
        dec_p = fns["run_sequencerK_with_drive"](sb, meta, word_idx[a], word_idx[x], drives, permute=True,
                                                 match_thresh=match_thresh)[0]
        if dec_p != f"ans{(i + 1) % K}":
            perm_ok = False
    permuted_inverts = perm_ok
    # NO-DIVNORM raw control: drive the decoded lines from the RAW (un-normalized) scores + the same placed threshold.
    # The raw drive lights winner AND runner-up -> the whole-row leak -> some present cue mis-routes OR a moat cue
    # spuriously matches (the divnorm is load-bearing). Built once here (not cached on the composer).
    bscores = [fns["block_cleanup_scores"](c_seq, b) for b in range(K)]
    raw_sb = fns["build_divnorm_score_bridge"](seed=seed, V=c_seq.V, enable_divnorm=False)
    raw_drives, _ = fns["make_block_drives"](raw_sb, c_seq.V, bscores, input_gain=input_gain, retreat="divnorm",
                                             peak_mult=1.0)
    raw_present = [fns["decision_to_block"](
        fns["run_sequencerK_with_drive"](sb, meta, word_idx[a], word_idx[x], raw_drives,
                                         match_thresh=match_thresh)[0], K) for (a, x, p) in facts]
    raw_moat = [fns["decision_to_block"](
        fns["run_sequencerK_with_drive"](sb, meta, word_idx[r["cue"][0]] if r["cue"][0] in word_idx else 0,
                                         word_idx[r["cue"][1]] if r["cue"][1] in word_idx else 0, raw_drives,
                                         match_thresh=match_thresh)[0], K)
                for r in moat_rows if (r["cue"][0] in word_idx and r["cue"][1] in word_idx)]
    raw_present_correct = all(raw_present[i] == i for i in range(K))
    raw_moat_clean = all(b is None for b in raw_moat)
    raw_fails = not (raw_present_correct and raw_moat_clean)    # raw control behaved (failed) -> divnorm load-bearing

    result = dict(seed=seed, D=D, K=K, rows=rows, answer_identical=answer_identical, moat_ok=moat_ok, fa=fa,
                  reconsolidation_ok=reconsolidation_ok, lesion_fails_safe=lesion_fails_safe,
                  lesion_decisions=les, permuted_inverts=permuted_inverts, raw_fails=raw_fails)
    # MEMORY-SAFE: drop this iteration's bridges (the heavy K=32/V=320 sequencer fabric the integrated_loop composer
    # built) + return the GPU mempool blocks BEFORE the next (seed, K) iteration allocates. The result dict above holds
    # only primitives (strings / None / bools / the lesion-decision strings) -- no live bridge reference -- so freeing
    # here cannot corrupt it. See _free_gpu_memory's note for the root cause this guards.
    del c_host, c_seq, raw_sb, sb, meta, drives, raw_drives, raw_present, raw_moat, bscores, les
    _free_gpu_memory()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ks", default="2,4,8", help="store sizes K (the production K=32 gate adds 16,32 on GPU)")
    ap.add_argument("--match-thresh", type=float, default=0.06, help="the validated production threshold")
    ap.add_argument("--gain", type=float, default=0.11)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--input-gain", type=float, default=1.0)
    ap.add_argument("--vocab-320", action="store_true",
                    help="320-scale: build the composer on a 320-word vocab (the production tier; GPU-only)")
    ap.add_argument("--out", default="research/findings/raw/_phaseB_onebrain_integrated_loop_fold_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    vocab = VOCAB
    grounded = None
    if args.vocab_320:
        # 320-scale: pad the V=72 production vocab to 320 distinct words (the fact words stay the first 72). The
        # divnorm op-point (gain=0.11) holds at larger V (the winner clears rheobase by a wider margin). The codes are
        # the composer's own (random per-seed); grounded codes are a drop-in (the rf grounded path) tested separately.
        extra = [f"w{i:03d}" for i in range(320 - len(VOCAB))]
        vocab = list(VOCAB) + extra
        assert len(set(vocab)) == 320, "320-scale vocab must be 320 distinct words"

    print(f"integrated_loop FOLD de-risk: match_thresh={args.match_thresh} gain={args.gain} sigma={args.sigma} "
          f"input_gain={args.input_gain} V={len(vocab)} gpu={is_gpu_backend()}", flush=True)

    all_results = {}
    for K in ks:
        results = []
        for s in seeds:
            r = run_seed_K(s, args.dim, K, vocab, grounded, args.match_thresh, args.gain, args.sigma, args.input_gain)
            results.append(r)
            ai = "==host" if r["answer_identical"] else "!=HOST"
            moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['fa']})"
            rec = "recon-OK" if r["reconsolidation_ok"] else "RECON-FAIL"
            les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_decisions']})"
            perm = "perm-inverts" if r["permuted_inverts"] else "perm-FAIL"
            raw = "raw-fails" if r["raw_fails"] else "RAW-ALSO-PASSES"
            print(f"K={K} seed {s} D{args.dim}: {ai}  {moat}  {rec}  {les}  {perm}  {raw}", flush=True)
        all_results[str(K)] = results

    summary = {}
    overall_go = True
    for K in ks:
        rs = all_results[str(K)]
        n = len(rs)
        ai_n = sum(r["answer_identical"] for r in rs)
        moat_n = sum(r["moat_ok"] for r in rs)
        fa_total = sum(r["fa"] for r in rs)
        rec_n = sum(r["reconsolidation_ok"] for r in rs)
        les_n = sum(r["lesion_fails_safe"] for r in rs)
        perm_n = sum(r["permuted_inverts"] for r in rs)
        raw_n = sum(r["raw_fails"] for r in rs)
        go = (ai_n == n and moat_n == n and fa_total == 0 and rec_n == n and les_n == n and perm_n == n and raw_n == n)
        overall_go = overall_go and go
        summary[str(K)] = dict(n=n, answer_identical_n=ai_n, moat_n=moat_n, fa_total=fa_total, reconsolidation_n=rec_n,
                               lesion_n=les_n, permuted_n=perm_n, raw_fails_n=raw_n, verdict="GO" if go else "NEGATIVE")
        print(f"\nK={K} SUMMARY: ==host {ai_n}/{n}  moat {moat_n}/{n} (FA_total {fa_total})  recon {rec_n}/{n}  "
              f"lesion {les_n}/{n}  permuted {perm_n}/{n}  raw-fails {raw_n}/{n}  -> {summary[str(K)]['verdict']}",
              flush=True)

    verdict = "GO" if overall_go else "NEGATIVE"
    print(f"\nOVERALL: {verdict}  (K in {ks}, {len(seeds)} seeds, match_thresh={args.match_thresh}, V={len(vocab)})",
          flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, verdict=verdict, match_thresh=args.match_thresh, gain=args.gain,
                                    sigma=args.sigma, input_gain=args.input_gain, V=len(vocab), gpu=is_gpu_backend()),
                       results=all_results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
