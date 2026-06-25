"""PURITY #4 / R-B de-risk: lift the spiking-sequencer DRIVE MARGIN at SMALL vocab via a V-AWARE divnorm DRIVE
op-point -- the relocated residual from R-A1 (`research/findings/raw/_spiking_sequencer_RA1_close.json`).

R-A1's HONEST NEGATIVE relocated the residual precisely (its 5 probes, seed-42 GPU):
  * the agent-role CLEANUP is CORRECT on both code arms (probe 1: every block's cleanup winner is right);
  * the win_rate 0.000 comes from the sequencer's DIVNORM-DRIVE op-point (`make_block_drives` -> `onbridge_divnorm_
    drive`, gain=0.11/sigma=1.0) interacting with the cleanup-codebook SIZE V (probe 2: both-roles-lit 0/8 at V=23,
    3/8 fresh-random; probe 4: the SAME stream codes clear 6/8 at V=320 but 0/8 at V=23);
  * the drive is SCALE-INVARIANT in input_gain (probe 3) -> input_gain is NOT a lever; the divisor SHAPE (gain/sigma)
    + the codebook SIZE V set which words clear rheobase. The moat is the hard gate -- match_thresh is FORBIDDEN.

THE MECHANISM (the relocated residual, exactly): the on-bridge divisive norm divides the pre-threshold drive by
`sigma + gain * mean_j(drive_j)`, where the MEAN is over ALL V*n_word flagged neurons (sim/bridge.py:6209-6215). For
a role read only the winner (+ maybe a runner-up) carries nonzero drive, so the mean ~= input_gain*active_sum/(V*
n_word) is INVERSELY proportional to V. At FIXED gain the divisor is large at small V (few zero-pools dilute the mean)
-> the normalized winner falls BELOW the placed rheobase -> the coincidence-AND match pool fires 0.000 (the present-cue
MISS = the SAFE over-abstention). At large V the many zero-pools shrink the mean -> the divisor -> sigma -> the winner
clears. THE FIX (this de-risk): a V-AWARE gain `gain(V) = sequencer_gain * (V / V_ref)` (V_ref=320 -> V=320 is
BYTE-UNCHANGED at gain=0.11). The V-factor cancels the 1/V in the mean, so `gain(V)*mean ~= sequencer_gain*active_sum/
(V_ref*n_word)` is V-INVARIANT -> the SAME normalized op-point at every V -> the winner clears rheobase at small V too.
A diagnostic probe (winner peak=1e6, runner-up 0.45*peak): FIXED gain=0.11 lights the winner at V=23 (1) but NOT at
V=15 (0); V-aware gain lights the winner at ALL V (15/23/72/160/320). It is a DRIVE op-point change (the divisor SHAPE
as a function of V) -- NOT match_thresh (forbidden), NOT a looser gate, NOT the cleanup, NOT the code source.

THE BARS (per seed, per V in {15 (the agent-test vocab), 23 (the R-A1 child-corpus fact set), 320 (production)}):
  * present_ok == host: the V-aware-gain sequencer picks the SAME block the host first-match does on ALL present cues;
  * win_rate-nonzero-via-MARGIN: the per-cue winner-block match-pool rate goes NON-ZERO on all present cues (probe-2
    both-roles-lit non-zero on all blocks), and it rose BECAUSE the divnorm DRIVE margin rose (the winner cleared
    rheobase), NOT because a threshold moved (match_thresh is FIXED at the production 0.06 throughout);
  * moat 0-FA (HARD): every absent/cross cue abstains -- the moat is NEVER closed via match_thresh or a looser gate;
  * the FIXED-gain arm is the documented revert (present_ok < tot at small V) -> proving the V-aware op-point is the
    lever; AND the V-aware op-point STILL holds at V=320 (no large-V regression: V=320 is gain-BYTE-UNCHANGED);
  * agent_eq_host: a full agent-level matrix (query_patient / query_agent / ask_yes_no) on the integrated_loop
    composer with the V-aware gain == the host-_scan oracle, with the moat abstaining on absent cues.

GPU-only. Reuse-by-import, NO sim/ edit: the V-aware gain is supplied at the runner layer by constructing the composer
with `sequencer_gain = base * V` (base = the V_ref-normalized coefficient), since the composer reads `self.sequencer_
gain` into `make_block_drives`. If GO, the V-awareness is then wired into `_ensure_sequencer` (a tiny composer-layer
change, NOT sim/). HONEST NEGATIVE is a valid deliverable: if the V-aware op-point (and the R-B2/R-B3 fallbacks) do
NOT clear small-V present_ok without touching the threshold/gate, #4 is a characterized small-V DRIVE-margin boundary.
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
V_REF = 320  # the V-aware gain reference: gain(V) = base * V, base = production_gain / V_REF -> gain(320)=production

# ----------------------------------------------------------------------------------------------------------------
# The TWO small-vocab fact sets (the two scales the agent-test / R-A1 actually use), plus the 320 production set:
#   * V=15 set  = the EXACT tests/test_one_brain_composer_agent.py vocab + 4 facts (the agent default the flip must keep);
#   * V=23 set  = the R-A1 child-corpus 8 SVO facts (all 23 words in the 40x8 stream taxonomy -> production-margin codes);
#   * V=320     = the consolidated_320 production demo set (the 8 facts on the FULL 320-word stream taxonomy).
# present/absent cues per set. Distinct agents+actions so a (agent, action) cue uniquely selects one block.
# ----------------------------------------------------------------------------------------------------------------
AGENT_VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
               "north", "east", "south", "west", "home"]   # V=15, the test_one_brain_composer_agent vocab
AGENT_FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"),
               ("river", "stop", "west")]
AGENT_ABSENT = [("apple", "stop"), ("dog", "stop"), ("cat", "go"), ("river", "go")]

CHILD_FACTS = [("dog", "eat", "apple"), ("cat", "play", "ball"), ("bird", "sleep", "tree"),
               ("girl", "run", "park"), ("boy", "look", "book"), ("lion", "eat", "cake"),
               ("rabbit", "jump", "garden"), ("mouse", "walk", "house")]   # the R-A1 set, V=23 child-corpus
CHILD_ABSENT = [("dog", "sleep"), ("cat", "eat"), ("bird", "run"), ("girl", "look"), ("lion", "jump")]


def _projection(d_out, n_in, seed):
    """The fixed random complex projection n_in -> d_out -- VERBATIM from consolidated_320_conversation_demo."""
    rng = np.random.RandomState(seed * 7919 + 13)
    return (rng.standard_normal((d_out, n_in)) + 1j * rng.standard_normal((d_out, n_in))).astype(np.complex128)


def grounded_phases(code_vec, proj):
    """Real cortex code -> composer phases[D] in [0,1) -- VERBATIM from the demo (the step-3 grounding map)."""
    z = proj @ code_vec.astype(np.complex128)
    return (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)


def _stream_codes(seed, words, readout="neural"):
    """The SUBSELECTED 320 stream-learned codes grounded to D phases for `words` (the production-margin code per word).
    Returns (grounded_dict, codes_path, found). Words not in the 320 taxonomy are skipped (kept None)."""
    from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
    from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories
    full_vocab, _cat, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    idx = {w: i for i, w in enumerate(full_vocab)}
    suffix = "neural_seed" if readout == "neural" else "seed"
    cpath = os.path.join(_REPO, "research", "findings", "raw", f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
    if not os.path.exists(cpath):
        return None, cpath, full_vocab
    codes = np.load(cpath)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    proj = _projection(D, codes.shape[1], seed)
    if words is None:  # None = ground ALL 320 taxonomy words (the V=320 full-vocab path)
        words = full_vocab
    grounded = {w: grounded_phases(codes[idx[w]], proj) for w in words if w in idx}
    return grounded, cpath, full_vocab


# ----------------------------------------------------------------------------------------------------------------
# The 320 production set: 8 facts whose 23 words live in the 320 taxonomy, but the composer is built over the FULL 320
# vocab (so the divnorm score bridge is V=320 -- the production regime). present/absent over those 8 facts.
# ----------------------------------------------------------------------------------------------------------------
def _prod_set(full_vocab):
    facts = CHILD_FACTS
    absent = CHILD_ABSENT
    # build the composer over the FULL 320 vocab so V=320 (the divnorm pool is 320 word-pools)
    return list(full_vocab), facts, absent


def _host_block(c, agent, action):
    for i, got in enumerate(c._read_blocks()):
        if got.get("agent") == agent and got.get("action") == action:
            return i
    return None


def _rates_to_array(rates, K):
    return np.asarray([float(rates[f"m{b}"]) for b in range(K)], dtype=float)


def _margins_and_present(c, present, absent, thresh):
    """The SURPASS move-1 margin read (== R-A1 / _burndown_1A_c2): for the integrated_loop composer `c`, read the
    per-block sequencer match rates at match_thresh=0.0 once, report each present cue's winner-block win_rate + worst
    off rate, the present_ok / moat at `thresh`, and the per-block BOTH-ROLES-LIT count (probe-2: the divnorm drive
    margin -- does each present block's agent line AND action line light)."""
    from research.runners.one_brain_composer import _seq_imports
    fns = _seq_imports()
    K = len(c.kb)
    c._ensure_sequencer(K)
    sb, meta = c._seq
    host_present = {(a, v): _host_block(c, a, v) for (a, v, _i) in present}
    host_absent = {(a, v): _host_block(c, a, v) for (a, v) in absent}

    # probe-2: both-roles-lit per block -- the divnorm drive margin. Reuse the composer's own _seq_drives (the lit
    # count is encoded in the drive: a decoded word-line is driven hi_pA iff its pool fired). dA/dX>0 == lit.
    both_lit = 0
    drives = c._seq_drives
    for bi in range(K):
        dA, dX = drives[bi]
        a_lit = int((np.asarray(dA) > 0).sum())
        x_lit = int((np.asarray(dX) > 0).sum())
        if a_lit > 0 and x_lit > 0:
            both_lit += 1

    per_cue = {}
    for (a, v, _i) in present:
        if c.enable_seq_vocab_shrink:
            _dec, rates = fns["run_sequencerK_reduced_with_drive"](
                sb, meta, c.words, c._seq_mapA, c._seq_mapX, a, v, c._seq_drives, match_thresh=0.0)
        else:
            _dec, rates = fns["run_sequencerK_with_drive"](
                sb, meta, c._word_index[a], c._word_index[v], c._seq_drives, match_thresh=0.0)
        per_cue[(a, v)] = _rates_to_array(rates, K)

    per_absent = {}
    for (a, v) in absent:
        in_red = (a in (c._seq_mapA or {})) and (v in (c._seq_mapX or {}))
        if not in_red:
            per_absent[(a, v)] = None
            continue
        _dec, rates = fns["run_sequencerK_reduced_with_drive"](
            sb, meta, c.words, c._seq_mapA, c._seq_mapX, a, v, c._seq_drives, match_thresh=0.0)
        per_absent[(a, v)] = _rates_to_array(rates, K)

    margins = {}
    for (a, v, _i) in present:
        r = per_cue[(a, v)]
        tgt = host_present[(a, v)]
        win = float(r[tgt]) if (tgt is not None and tgt < r.size) else 0.0
        off = float(np.max([r[j] for j in range(r.size) if j != tgt])) if r.size > 1 else 0.0
        margins[f"{a},{v}"] = {"target_block": tgt, "win_rate": round(win, 4), "worst_off_rate": round(off, 4)}

    def decode(rates):
        fired = [j for j in range(rates.size) if rates[j] > thresh]
        return min(fired) if fired else None

    present_ok = sum(1 for (a, v, _i) in present if decode(per_cue[(a, v)]) == host_present[(a, v)])
    fa = 0
    for (a, v) in absent:
        r = per_absent[(a, v)]
        if r is None:
            continue
        if decode(r) is not None:
            fa += 1
    nonzero_all = all(m["win_rate"] > 0.0 for m in margins.values())
    return {
        "margins": margins, "present_ok": present_ok, "present_tot": len(present),
        "both_roles_lit": both_lit, "K": K,
        "false_accept": fa, "moat_0fa": fa == 0, "win_rate_nonzero_all": nonzero_all,
        "host_present": {f"{a},{v}": host_present[(a, v)] for (a, v, _i) in present},
        "host_absent_all_none": all(host_absent[k] is None for k in host_absent),
    }


def _agent_eq_host(seed, vocab, facts, absent, grounded, thresh, gain):
    """The FULL agent-level matrix: an integrated_loop (spiking, V-aware gain) composer + a host-_scan oracle on the
    SAME store + codes; the spiking answers == the host answers on present cues + the moat abstains on absent cues."""
    from research.runners.one_brain_composer import OneBrainComposer
    spk = OneBrainComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded,
                           integrated_loop=True, sequencer_match_thresh=thresh, sequencer_gain=gain)
    host = OneBrainComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded, integrated_loop=False)
    for (a, v, p) in facts:
        spk.store(a, v, p); host.store(a, v, p)
    present_eq, present_correct, mism = 0, 0, []
    for (a, v, p) in facts:
        s_p, h_p = spk.query_patient(a, v), host.query_patient(a, v)
        s_a, h_a = spk.query_agent(v, p), host.query_agent(v, p)
        s_y, h_y = spk.ask_yes_no(a, v, p), host.ask_yes_no(a, v, p)
        if (s_p, s_a, s_y) == (h_p, h_a, h_y):
            present_eq += 1
        else:
            mism.append({"cue": f"{a},{v}", "spk": [s_p, s_a, s_y], "host": [h_p, h_a, h_y]})
        if (s_p, s_a) == (p, a):
            present_correct += 1
    fa = 0
    for (a, v) in absent:
        if spk.query_patient(a, v) is not None:
            fa += 1
    a_pat = facts[0][2]
    for (a, v) in absent:
        if spk.ask_yes_no(a, v, a_pat) == "yes":
            fa += 1
    del spk, host
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return {"present_eq_host": present_eq, "present_tot": len(facts), "present_correct": present_correct,
            "agent_false_accept": fa, "agent_eq_host": present_eq == len(facts), "agent_moat_0fa": fa == 0,
            "mismatches": mism}


def _make_composer(seed, vocab, facts, grounded, gain, thresh):
    """Build an integrated_loop composer with a FIXED sequencer_gain (the V-aware value the caller computed) + store."""
    from research.runners.one_brain_composer import OneBrainComposer
    c = OneBrainComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded, integrated_loop=True,
                         sequencer_match_thresh=thresh, sequencer_gain=gain)
    for (a, v, p) in facts:
        c.store(a, v, p)
    return c


def _free():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def run_scale(seed, scale_name, vocab, facts, absent, grounded, thresh, base, prod_gain):
    """One (seed, scale) cell: compare the FIXED-gain arm (the documented revert) against the V-AWARE-gain arm
    (gain = base*V), both at the production match_thresh. Returns the margins + present_ok + moat for each arm."""
    V = len({w for w in vocab})  # the composer's V is len(set(vocab)); the divnorm pool is V word-pools
    # the composer normalizes its vocab; build one to read the actual V it uses.
    c_probe = _make_composer(seed, vocab, facts, grounded, gain=prod_gain, thresh=thresh)
    Vactual = c_probe.V
    fixed = _margins_and_present(c_probe, [(a, v, i) for i, (a, v, _p) in enumerate(facts)], absent, thresh)
    del c_probe; _free()

    vaware_gain = float(base * Vactual)   # gain(V) = base*V  (base = prod_gain / V_REF -> gain(320)=prod_gain)
    c_va = _make_composer(seed, vocab, facts, grounded, gain=vaware_gain, thresh=thresh)
    vaware = _margins_and_present(c_va, [(a, v, i) for i, (a, v, _p) in enumerate(facts)], absent, thresh)
    del c_va; _free()

    # the agent-level == host on the V-aware-gain arm (the GO bar).
    agent = _agent_eq_host(seed, vocab, facts, absent, grounded, thresh, vaware_gain)
    return {"scale": scale_name, "V": Vactual, "fixed_gain": prod_gain, "vaware_gain": vaware_gain,
            "fixed": fixed, "vaware": vaware, "agent_level": agent}


def run(seeds, thresh, base, prod_gain, readout):
    out = {"config": {"D": D, "match_thresh": thresh, "v_ref": V_REF, "base_coeff": base,
                      "prod_gain": prod_gain, "readout": readout, "seeds": seeds}, "per_seed": []}
    for seed in seeds:
        row = {"seed": seed, "scales": []}
        # V=15: the agent-test vocab + 4 facts (fresh-random codes -- the default agent path; grounded=None)
        row["scales"].append(run_scale(seed, "V15_agent", AGENT_VOCAB, AGENT_FACTS, AGENT_ABSENT,
                                        None, thresh, base, prod_gain))
        _free()
        # V=23: the R-A1 child-corpus 8 facts, production-margin (stream) codes
        gr23, cpath, _fv = _stream_codes(seed, sorted({w for f in CHILD_FACTS for w in f}), readout)
        if gr23 is None:
            row["scales"].append({"scale": "V23_child", "error": f"no stream codes at {cpath}"})
        else:
            words23 = sorted({w for f in CHILD_FACTS for w in f})
            row["scales"].append(run_scale(seed, "V23_child", words23, CHILD_FACTS, CHILD_ABSENT,
                                            gr23, thresh, base, prod_gain))
        _free()
        # V=320: the production demo regime -- the SAME child facts but the composer built over the FULL 320 vocab
        gr_full, cpath2, full_vocab = _stream_codes(seed, None, readout)  # all 320 words grounded
        if gr_full is None:
            row["scales"].append({"scale": "V320_prod", "error": f"no stream codes at {cpath2}"})
        else:
            pv, pf, pa = _prod_set(full_vocab)
            row["scales"].append(run_scale(seed, "V320_prod", pv, pf, pa, gr_full, thresh, base, prod_gain))
        _free()
        out["per_seed"].append(row)
    return out


def _scale_go(sc):
    """A scale's GO: the V-aware arm clears present_ok == tot + win_rate nonzero + moat 0-FA + agent == host + moat;
    AND it rose via the MARGIN (both_roles_lit went up vs the fixed arm OR is already full) -- NOT a threshold move."""
    if "error" in sc:
        return False, "no-codes"
    va, fx, ag = sc["vaware"], sc["fixed"], sc["agent_level"]
    margin_rose = (va["both_roles_lit"] >= fx["both_roles_lit"]) and (va["both_roles_lit"] == va["K"])
    go = (va["present_ok"] == va["present_tot"] and va["win_rate_nonzero_all"] and va["moat_0fa"]
          and ag["agent_eq_host"] and ag["agent_moat_0fa"] and ag["present_correct"] == ag["present_tot"]
          and margin_rose)
    return go, ("GO" if go else "NOT-GO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--thresh", type=float, default=0.06)        # the PRODUCTION op-point (NEVER lowered)
    ap.add_argument("--prod-gain", type=float, default=0.11)     # the production divnorm gain (== gain(V=320))
    ap.add_argument("--base", type=float, default=None,
                    help="V-aware base coeff: gain(V)=base*V. Default = prod_gain / V_REF (so gain(320)=prod_gain).")
    ap.add_argument("--readout", choices=["neural", "host"], default="neural")
    ap.add_argument("--out", default="research/findings/raw/_spiking_sequencer_RB_close.json")
    a = ap.parse_args()
    base = a.base if a.base is not None else (a.prod_gain / float(V_REF))
    res = run(a.seeds, a.thresh, base, a.prod_gain, a.readout)
    res["config"]["base_coeff"] = base
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[RB] D={D} match_thresh={a.thresh} (PRODUCTION, FIXED) prod_gain={a.prod_gain} "
          f"V-aware gain(V)=base*V base={base:.3e} (gain(320)={base*V_REF:.4f})\n", flush=True)
    scale_go = {}
    for sr in res["per_seed"]:
        print(f"seed {sr['seed']}", flush=True)
        for sc in sr["scales"]:
            if "error" in sc:
                print(f"  {sc['scale']:12s}: SKIP -- {sc['error']}", flush=True)
                scale_go.setdefault(sc["scale"], []).append(False)
                continue
            go, tag = _scale_go(sc)
            scale_go.setdefault(sc["scale"], []).append(go)
            fx, va, ag = sc["fixed"], sc["vaware"], sc["agent_level"]
            print(f"  {sc['scale']:12s} (V={sc['V']}, gain {sc['fixed_gain']:.3f}->{sc['vaware_gain']:.4f}):", flush=True)
            print(f"     FIXED  : both_lit {fx['both_roles_lit']}/{fx['K']}  present_ok {fx['present_ok']}/{fx['present_tot']}"
                  f"  win_nonzero {fx['win_rate_nonzero_all']}  moat_0fa {fx['moat_0fa']}", flush=True)
            print(f"     V-AWARE: both_lit {va['both_roles_lit']}/{va['K']}  present_ok {va['present_ok']}/{va['present_tot']}"
                  f"  win_nonzero {va['win_rate_nonzero_all']}  moat_0fa {va['moat_0fa']}", flush=True)
            print(f"     AGENT  : eq_host {ag['present_eq_host']}/{ag['present_tot']} correct {ag['present_correct']}"
                  f"/{ag['present_tot']}  moat_0fa {ag['agent_moat_0fa']}  ==> {tag}", flush=True)
            for m in ag["mismatches"][:3]:
                print(f"        !! mismatch {m}", flush=True)
    print("\n[RB] PER-SCALE GO (all seeds):", flush=True)
    all_go = True
    for scale in ("V15_agent", "V23_child", "V320_prod"):
        gos = scale_go.get(scale, [])
        n_go = sum(gos)
        ok = (len(gos) > 0 and n_go == len(gos))
        all_go = all_go and ok
        print(f"  {scale:12s}: {n_go}/{len(gos)} seeds GO", flush=True)
    print(f"\n[RB] VERDICT: {'GO' if all_go else 'NOT-GO'} -- V-aware divnorm DRIVE op-point clears small-V present_ok "
          f"== host (win_rate nonzero via MARGIN, moat 0-FA) at V=15 AND V=23 AND holds at V=320, all seeds.", flush=True)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
