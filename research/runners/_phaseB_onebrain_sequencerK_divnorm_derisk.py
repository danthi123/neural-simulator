"""Phase B / burndown #3 Stage S1: drive the K-WAY sequencer's decoded word-lines from the ON-BRIDGE
divisive-normalization (S5's Carandini-Heeger primitive), retiring the residual host `scores_to_drive` DATA read.

Stage S0 (`_phaseB_onebrain_sequencerK_derisk.py`, commit 44f4a166, GO at K in {2,4,8,16}, 6 seeds CPU/D=64)
generalized the sequencer CONTROL fabric from K=2 to a parameter K. BUT S0 still drives each block's decoded
word-lines from the HOST `scores_to_drive(block_cleanup_scores)` read -- a per-query `thr = frac*scores.max()`
peak-normalization computed in Python (`_phaseB_onebrain_sequencer_derisk.py:scores_to_drive`,
the line `thr = frac * s.max()`). That is the LAST host DATA read inside the K-way scan loop -- the residual S5
host read.

The S5 fix is PROVEN (CYCLE 294, `2026-06-20-S5-divnorm-derisk.md`, runner `_phaseC_S5_divnorm_derisk.py`,
commit 960467b0): the EXISTING `input_divisive_norm` Carandini-Heeger primitive (`sim/regions.py:240` +
`sim/config.py:440` + the guarded per-step divide `sim/bridge.py:6048`: `r_i = x_i/(sigma + gain*mean_j x_j)`,
default-off byte-identical) -- flagged on a score-pool region and driven with the cleanup scores as external input
current -- reproduces the host per-query peak-normalization ON-BRIDGE across a per-query-peak sweep, moat-0-FA, with
NO `sim/` edit. A PLACED firing threshold (the Izhikevich rheobase) then reads which words FIRE: only the winner
crosses, the runner-up stays silent (the on-bridge divide made the threshold scale-invariant). The S5 operating
point is `input_gain=1.0`, `sigma=1.0`, `gain=0.05`, rheobase as the placed threshold (the primitive is
scale-invariant but has no post-divide output gain, so `gain` is small).

STAGE S1 = wire S5's on-bridge divnorm INTO the K-way sequencer. For each stored block, the block's cleanup scores
(the op result) are driven through a divnorm-flagged score pool (S5's `build_divnorm_score_bridge` +
`onbridge_divnorm_drive`); which words FIRE is the per-block decoded-line drive. That drive feeds S0's K-way
sequencer (the SAME match cascade + first-match priority WTA + production rule). The host `scores_to_drive`
peak-read is GONE from the drive path -- that is the point of S1.

GO BAR (this stage, CPU/numpy -- the exact-algebra parity oracle):
  * the divnorm-driven K-way sequencer is == host `_scan` for who/what (the right block answers; absent/cross cues
    abstain) AND the no-confab MOAT holds (0 false-accepts), at K in {2,4,8,16}, multi-seed (match S0's K coverage);
  * the host `scores_to_drive` peak-read is GONE from the drive path (asserted by construction -- the runner imports
    `block_cleanup_scores` for the op result but NEVER calls `scores_to_drive` on the drive path; only the no-divnorm
    NEGATIVE control compares to it for the load-bearing check);
  * ANTI-CHEATS: a NO-DIVNORM control (divnorm-OFF score pool + the SAME placed rheobase threshold on the RAW
    un-normalized drive) reproduces the un-normalized failure (the divnorm is load-bearing); OFF==byte-identical
    (S5's `check_off_byte_identical` guard); sequencer-LESION fails safe (cut the result->op conditioning ->
    abstain); permuted-rule INVERTS (the decision follows the cyclic-shift rule, not a fixed scan order); per-block
    priority correct (a degenerate two-block-match cue answers the LOWER block).

The MOAT is the HARD gate -- it is NEVER traded. NO `sim/` edit (reuse-by-import: OneBrainComposer +
SimulationBridge + the S0 K-way sequencer + the EXISTING input_divisive_norm sim/ primitive flipped on a
runner-built score bridge). HONEST NEGATIVE is a valid deliverable: if the on-bridge divnorm can't hold the K-way
== host match across K (e.g. the placed threshold squeezes once K decoded-line groups compete), report it crisply
-- that is an early signal of the S2 K=32 boundary; do NOT loosen the moat or config-search beyond the S5/S0
op-points.

  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_divnorm_derisk \
      --seeds 42,43,44 --dim 64 --ks 2,4,8,16
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
# Reuse the op-result reader VERBATIM (the FHRR cleanup is unchanged; S1 only changes the DRIVE source).
from research.runners._phaseB_onebrain_sequencer_derisk import block_cleanup_scores
# Reuse the PROVEN K-way CONTROL fabric VERBATIM (builder / wiring / reset / production rule + anti-cheat helpers).
# S1 touches NONE of the sequencer control logic -- it only swaps how the decoded word-lines are driven.
from research.runners._phaseB_onebrain_sequencerK_derisk import (
    build_sequencerK_bridge, reset_sequencerK_state,
    host_scan_block, decision_to_block, patient_of, _build_queries, run_priority_check,
    ALL_FACTS, VOCAB,
)
# Reuse S5's on-bridge divnorm score bridge + the per-query-peak divisive-norm drive VERBATIM (the proven mechanism
# that retires the host scores.max() read). S1 wires it into the K-way loop.
from research.runners._phaseC_S5_divnorm_derisk import (
    build_divnorm_score_bridge, onbridge_divnorm_drive, check_off_byte_identical, region_idx,
)
from sim.backend import to_host, from_host, is_gpu_backend


# ----------------------------------------------------------------------------------------------------------------
# The drive sources (per block: cleanup (agent_scores, action_scores) -> (dA[V], dX[V])). The ONLY axis S1 changes
# vs S0 is which of these produces the decoded-line drive:
#   divnorm  -- S5 OPTION 4: the on-bridge input_divisive_norm score pool + placed rheobase threshold (NO host
#               scores.max()). THIS is the S1 production drive.
#   raw      -- the NEGATIVE control: the SAME placed threshold + input_gain, but the score pool has divnorm OFF, so
#               the RAW un-normalized drive hits rheobase and the winner AND runner-up both fire (the un-normalized
#               whole-row-lights wall) -> the moat breaks / != host. Proves the divnorm is load-bearing.
# (S0's host `scores_to_drive` is DELIBERATELY ABSENT from the drive path -- retiring it is the point of S1.)
# ----------------------------------------------------------------------------------------------------------------
def make_block_drives(score_sb, V, bscores, input_gain):
    """Drive every block's (agent, action) cleanup scores through the divnorm score pool (S5's
    `onbridge_divnorm_drive`); return per-block (dA, dX) decoded-line drives (hi_pA on the firing words, else 0).
    The per-query peak-normalization is the ON-BRIDGE divide; the placed threshold is the rheobase -- NO host
    scores.max(). Used for BOTH the divnorm production path (score_sb has divnorm ON) and the raw NEGATIVE control
    (score_sb has divnorm OFF -> the same threshold on un-normalized drive)."""
    drives, lit = [], []
    for (ag, ax) in bscores:
        dA, accA = onbridge_divnorm_drive(score_sb, V, ag, input_gain)
        dX, accX = onbridge_divnorm_drive(score_sb, V, ax, input_gain)
        drives.append((dA, dX))
        lit.append((int((accA > 0).sum()), int((accX > 0).sum())))
    return drives, lit


def run_sequencerK_with_drive(sb, meta, cue_agent_idx, cue_action_idx, block_drives, settle=60, lesion=False,
                              match_thresh=0.15, permute=False):
    """One who/what scan on the SUBSTRATE, K-way, with the decoded word-line drives supplied DIRECTLY (the option-4
    on-bridge divnorm drive computed upstream by `make_block_drives`). This is S0's `run_sequencerK` with the SAME
    K-way match cascade, first-match priority WTA, and production rule -- the ONLY change is the drive SOURCE: the
    decoded lines are driven from `block_drives` (the on-bridge divnorm firing) instead of the host
    `scores_to_drive(block_cleanup_scores, frac)` peak read. There is NO `scores_to_drive`/`s.max()` anywhere here.

    `block_drives` = [(dA[V], dX[V]), ...] (<=K) -- per block, the decoded agent/action word-lines to drive hi_pA.
    `lesion`=True severs the result->op conditioning (the decoded word-lines get ZERO drive) -> the match can never
    fire -> the sequencer fails SAFE (abstain). `permute`=True cyclically shifts the match->answer rule
    (m{b} -> ans{(b+1)%K}) -- the anti-cheat (the decision must follow the RULE). Returns (decision, rates),
    decision in {"ans0".."ans{K-1}", "abstain"} -- IDENTICAL channel semantics to S0."""
    V, K = meta["V"], meta["K"]
    idx = lambda nm: region_idx(sb, nm)                  # query-invariant cache (behavior-preserving; see region_idx)
    reset_sequencerK_state(sb)                            # clear prior-query gate/EMA/membrane leak (S0 discipline)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    # present the CUE (the question) as a spiking word-line pattern (the cue opens the per-word match gates)
    cur[idx(f"cueA_{cue_agent_idx}")] = 1500.0
    cur[idx(f"cueX_{cue_action_idx}")] = 1500.0
    # drive each block's DECODED word-lines from the ON-BRIDGE divnorm firing (the result->sequencer coupling).
    if not lesion:
        for bi, (dA, dX) in enumerate(block_drives[:K]):
            for w in range(V):
                if dA[w] > 0:
                    cur[idx(f"d{bi}A_{w}")] = dA[w]
                if dX[w] > 0:
                    cur[idx(f"d{bi}X_{w}")] = dX[w]
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur = from_host(cur)                                 # match bridge backend (numpy build -> cupy under cupy)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur            # hold the cue + decoded drive across the settle
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    sb.cp_external_input_current[:] = 0.0
    # the spiking match result per block (clean cascade: ~0.22 match / 0.00 no-match) -- S0's read, verbatim
    m_rates = [acc[idx(f"m{b}")].mean() / settle for b in range(K)]
    fired = [r > match_thresh for r in m_rates]
    rates = {f"m{b}": round(m_rates[b], 3) for b in range(K)}
    rates.update({f"f{b}": fired[b] for b in range(K)})
    # the K-way FIRST-MATCH priority production rule over the spiking match (S0 logic, verbatim)
    winner = next((b for b in range(K) if fired[b]), None)
    if winner is None:
        decision = "abstain"
    else:
        decision = f"ans{(winner + 1) % K}" if permute else f"ans{winner}"
    rates["winner"] = winner
    return decision, rates


def run_seed_K(seed, D, K, input_gain, sigma, gain):
    """Run one seed at store size K: build the composer + K facts, read each block's cleanup scores, build the K-way
    sequencer AND the divnorm score bridge, and check the DIVNORM-driven decision == host_scan + the moat +
    lesion-fails-safe + permuted-inverts. Also runs the NO-DIVNORM (raw) control on the present/moat cues so the
    divnorm is shown load-bearing (it must fail the moat OR != host)."""
    facts = ALL_FACTS[:K]
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=max(8, K), enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in facts:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    blocks = list(range(len(facts)))
    bscores = [block_cleanup_scores(c, b) for b in blocks]      # the op RESULTS (cleanup scores per block)

    sb, meta = build_sequencerK_bridge(seed=seed, V=V, K=K)
    div_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=True, sigma=sigma, gain=gain)   # ON
    raw_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=False)                          # OFF (control)

    # Precompute the per-block decoded-line drives ONCE per battery (the cleanup is fixed across cues).
    div_drives, div_lit = make_block_drives(div_sb, V, bscores, input_gain)   # S1 production drive (on-bridge divnorm)
    raw_drives, raw_lit = make_block_drives(raw_sb, V, bscores, input_gain)   # NEGATIVE control (un-normalized)

    queries = _build_queries(facts)
    rows = []
    for (qa, qx), kind in queries:
        ca, cx = word_idx[qa], word_idx[qx]
        host_blk = host_scan_block(c, qa, qx)            # the host _scan CONTROL decision (which block / abstain)
        dec, rates = run_sequencerK_with_drive(sb, meta, ca, cx, div_drives)    # DIVNORM-driven decision
        sub_blk = decision_to_block(dec, K)
        dec_raw, rates_raw = run_sequencerK_with_drive(sb, meta, ca, cx, raw_drives)   # raw control decision
        raw_blk = decision_to_block(dec_raw, K)
        host = patient_of(c, host_blk)
        sub = patient_of(c, sub_blk)
        rows.append(dict(cue=(qa, qx), kind=kind, host=host, sub=sub, decision=dec,
                         host_block=host_blk, sub_block=sub_blk, rates=rates,
                         raw_sub=patient_of(c, raw_blk), raw_decision=dec_raw, raw_block=raw_blk, raw_rates=rates_raw,
                         match_host_eq=(sub_blk == host_blk)))

    # --- the MOAT (HARD): every NON-present cue must abstain (decision==abstain, no block selected) -- FA == 0.
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    false_accepts = sum(1 for r in moat_rows if r["sub_block"] is not None)
    moat_ok = (false_accepts == 0) and all(r["decision"] == "abstain" for r in moat_rows)

    # --- sequencer-LESION (sever the result->op conditioning) on every present cue -> must FAIL SAFE (abstain).
    les = []
    for (a, x, p) in facts:
        dec_l, _ = run_sequencerK_with_drive(sb, meta, word_idx[a], word_idx[x], div_drives, lesion=True)
        les.append(dec_l)
    lesion_fails_safe = all(d == "abstain" for d in les)

    # --- PERMUTED-RULE: cyclic shift (m{b} -> ans{(b+1)%K}). A present cue for block b must route to ans{(b+1)%K}.
    perm_decs = []
    perm_ok = True
    for i, (a, x, p) in enumerate(facts):
        dec_p, _ = run_sequencerK_with_drive(sb, meta, word_idx[a], word_idx[x], div_drives, permute=True)
        perm_decs.append(dec_p)
        if dec_p != f"ans{(i + 1) % K}":
            perm_ok = False
    permuted_inverts = perm_ok

    # --- NO-DIVNORM (raw) NEGATIVE control: across the SAME battery the raw path must FAIL (some cue breaks the moat
    #     OR the present cues do not all == host). raw_fails == True means the control behaved (divnorm load-bearing).
    raw_moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    raw_fa = sum(1 for r in raw_moat_rows if r["raw_block"] is not None)
    raw_moat_ok = (raw_fa == 0) and all(r["raw_decision"] == "abstain" for r in raw_moat_rows)
    raw_eq_host = all(r["raw_block"] == r["host_block"] for r in rows)
    raw_s5_ok = raw_moat_ok and raw_eq_host
    raw_fails = not raw_s5_ok

    eq_all = all(r["match_host_eq"] for r in rows)
    return dict(seed=seed, D=D, K=K, rows=rows, eq_all=eq_all, moat_ok=moat_ok, false_accepts=false_accepts,
                lesion_fails_safe=lesion_fails_safe, lesion_decisions=les,
                permuted_inverts=permuted_inverts, permuted_decisions=perm_decs,
                raw_fails=raw_fails, raw_fa=raw_fa, raw_eq_host=raw_eq_host,
                div_lit=div_lit, raw_lit=raw_lit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ks", default="2,4,8,16", help="store sizes K to test (match S0 coverage)")
    ap.add_argument("--input-gain", type=float, default=1.0,
                    help="S5 op-point: fixed per-bridge input gain (NOT per-query); drive=input_gain*score, then divide")
    ap.add_argument("--sigma", type=float, default=1.0, help="S5 op-point: divisive semi-saturation constant")
    ap.add_argument("--gain", type=float, default=0.05,
                    help="S5 op-point: divisive strength on the mean term (small -- no post-divide output gain)")
    ap.add_argument("--out", default="research/findings/raw/_phaseB_onebrain_sequencerK_divnorm_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    # OFF==byte-identical guard (S5's primitive contract: divnorm-OFF is a guarded no-op; on differs / is load-bearing)
    off_guard = check_off_byte_identical()
    off_ok = (off_guard["off_mask_none"] and off_guard["on_mask_not_none"]
              and off_guard["off_eq_off"] and off_guard["on_differs_from_off"])
    print(f"OFF==byte-identical guard: {off_guard} -> {'PASS' if off_ok else 'FAIL'}", flush=True)

    all_results = {}
    for K in ks:
        results = []
        for s in seeds:
            r = run_seed_K(s, args.dim, K, args.input_gain, args.sigma, args.gain)
            results.append(r)
            eq = "==host" if r["eq_all"] else "!=host"
            moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['false_accepts']})"
            les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_decisions']})"
            perm = "perm-inverts" if r["permuted_inverts"] else f"perm-FAIL({r['permuted_decisions']})"
            raw = "raw-fails(divnorm-load-bearing)" if r["raw_fails"] else f"RAW-ALSO-PASSES(fa={r['raw_fa']},eq={r['raw_eq_host']})"
            det = "  ".join(f"{rr['kind']}:sub={rr['sub']}|host={rr['host']}" for rr in r["rows"])
            print(f"K={K} seed {s} D{args.dim}: {eq}  {moat}  {les}  {perm}  {raw}   [{det}]", flush=True)
        all_results[str(K)] = results

    # the per-block priority anti-cheat (degenerate two-block-match -> lower block wins, == host first-match).
    # Reused verbatim from S0 (it uses the host scores_to_drive inside `run_sequencerK`, NOT the S1 drive path -- the
    # priority STRUCTURE is what's under test there, identical in both stages; the S1 divnorm parity is the per-K
    # battery above). Kept for completeness of the anti-cheat panel.
    prio_results = [run_priority_check(s, args.dim) for s in seeds]
    prio_n = sum(p["priority_ok"] for p in prio_results)
    for p in prio_results:
        ok = "priority-OK" if p["priority_ok"] else "priority-FAIL"
        print(f"PRIORITY seed {p['seed']}: {ok}  decision={p['decision']} sub={p['sub']} host={p['host']}", flush=True)

    summary = {}
    overall_go = off_ok
    for K in ks:
        rs = all_results[str(K)]
        n = len(rs)
        eq_n = sum(r["eq_all"] for r in rs)
        moat_n = sum(r["moat_ok"] for r in rs)
        les_n = sum(r["lesion_fails_safe"] for r in rs)
        perm_n = sum(r["permuted_inverts"] for r in rs)
        raw_n = sum(r["raw_fails"] for r in rs)
        fa_total = sum(r["false_accepts"] for r in rs)
        go = (eq_n == n and moat_n == n and les_n == n and perm_n == n and raw_n == n and fa_total == 0)
        overall_go = overall_go and go
        summary[str(K)] = dict(n=n, eq_n=eq_n, moat_n=moat_n, lesion_n=les_n, permuted_n=perm_n,
                               raw_fails_n=raw_n, fa_total=fa_total, verdict="GO" if go else "NEGATIVE")
        print(f"\nK={K} SUMMARY: ==host {eq_n}/{n}  moat {moat_n}/{n} (FA_total {fa_total})  "
              f"lesion-fails-safe {les_n}/{n}  permuted-inverts {perm_n}/{n}  "
              f"raw-fails(divnorm-load-bearing) {raw_n}/{n}  -> {summary[str(K)]['verdict']}", flush=True)
    n_prio = len(prio_results)
    overall_go = overall_go and (prio_n == n_prio)
    print(f"PRIORITY SUMMARY: {prio_n}/{n_prio}  -> {'GO' if prio_n == n_prio else 'NEGATIVE'}", flush=True)
    print(f"OFF-guard: {'PASS' if off_ok else 'FAIL'}", flush=True)

    verdict = "GO" if overall_go else "NEGATIVE"
    print(f"\nOVERALL: {verdict}  (K in {ks}, {len(seeds)} seeds, S5 op-point input_gain={args.input_gain} "
          f"sigma={args.sigma} gain={args.gain}; host scores_to_drive read RETIRED from the drive path)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, priority_n=prio_n, priority_total=n_prio,
                                    off_guard=off_guard, off_ok=off_ok, verdict=verdict,
                                    gpu=is_gpu_backend(), input_gain=args.input_gain, sigma=args.sigma, gain=args.gain),
                       results=all_results, priority_results=prio_results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
