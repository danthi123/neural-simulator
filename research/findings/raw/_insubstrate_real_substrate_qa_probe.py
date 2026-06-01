"""Does the validated spiking relational memory + wh-QA survive on the REAL G.20 sparse substrate's
concept codes (captured from the deployed bridge), not just synthetic gen_sparse codes?

qa64 showed bind/QA at V=160/320 on SYNTHETIC gen_sparse codes. This is the validity step: capture the
REAL concept code for each word (drive lang_input(word) through the TRAINED sparse bridge -> accumulate
shared_concept_pool activity), then run the validated spiking SVO fact-memory + wh-QA + abstention control
on those REAL codes. Compares REAL vs SYNTHETIC head-to-head on the same words. If REAL holds the abstention
control + QA bar -> genuine composition works on the real deployed 160-concept substrate (the largest
genuine-composition conversational artifact). If REAL degrades -> honest boundary: real-substrate structure
(noise/overlap) is harder than idealized sparse, characterize it.

Cheap-first: ONE bridge (32 concepts) first; scale to all 5 (160) only if it RESOLVES.
Reuse-by-import (bind/unbind/RM machinery + sparse builder); no protected-module modification; no autograd.
load_checkpoint validates architecture -> the 2026-05-14 monkey-patch mismatch is CAUGHT, not silent.

Run (GPU), per bridge:
  python -m research.findings.raw._insubstrate_real_substrate_qa_probe \
      --bridge research/findings/raw/g11_bg/g20_sparse_bridges/bridgeA_nouns_sparse.simstate.h5 \
      --vocab  research/findings/raw/g11_bg/g20_bridgeA_nouns_vocab.txt --seed 42
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.runners.concept_pool_sparse_distributed as SP
from sim.backend import get_backend, to_host

# trained 160-bridge params (g20_sparse_5bridge_chain.ps1): MUST match the checkpoint exactly
N_LANG = 8192
N_POOL = 2000
PATTERN_SIZE = 100
SPARSITY = 0.02
DRIVE_PA = 200.0
RESET, STIM = 50, 120


def _center(v):
    v = np.asarray(v, dtype=np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_vocab(path):
    with open(path) as f:
        return [w.strip() for w in f if w.strip()]


def capture_real_codes(bridge, words, seed, xp):
    """Real concept code per word = shared_concept_pool activity when driving lang_input(word)."""
    from sim.text_embeddings import orthogonal_drive_pattern
    rm = bridge.region_manager
    pool_idx = xp.asarray(list(rm.indices("shared_concept_pool")), dtype=xp.int64)
    lang_idx = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)
    codes = {}
    for i, w in enumerate(words):
        drive = orthogonal_drive_pattern(cue_idx=i, n_cues=len(words), n_neurons=N_LANG,
                                         drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[lang_idx] = xp.asarray(drive, dtype=xp.float32)
        acc = xp.zeros(int(pool_idx.shape[0]), dtype=xp.float64)
        for _ in range(STIM):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[pool_idx].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        codes[w] = _center(to_host(acc) / STIM)
    return codes


def synthetic_codes(words, seed):
    """The idealized gen_sparse codes (what qa64 used) for the SAME words -- the comparison baseline."""
    pats = SP.generate_sparse_patterns(len(words), N_POOL, PATTERN_SIZE, seed)
    codes = {}
    for w, pat in zip(words, pats):
        v = np.zeros(N_POOL); v[pat] = 1.0
        codes[w] = _center(v)
    return codes


def run_qa(codes, words, seed, n_trials, n_facts, xp):
    """Validated spiking SVO fact-memory + wh-QA + abstention control on the given concept codes."""
    D = N_POOL
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    rng = np.random.default_rng(seed)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(seed, D, xp)

    def q(bounds, given, query_role):
        for b in bounds:
            if all(RM.unbind_spiking(bridge, idx, b, r, roles, codes, words, D, xp) == w
                   for r, w in given.items()):
                return RM.unbind_spiking(bridge, idx, b, query_role, roles, codes, words, D, xp)
        return None

    qa_ok = ctrl_ok = tot = 0
    for _ in range(n_trials):
        pk = rng.choice(len(words), 3 * n_facts, replace=False)
        facts = [{"agent": words[pk[3*f]], "action": words[pk[3*f+1]], "patient": words[pk[3*f+2]]}
                 for f in range(n_facts)]
        bounds = [RM.bind_fact_spiking(bridge, idx, fc, codes, roles, D, xp) for fc in facts]
        f = facts[rng.integers(n_facts)]
        who = q(bounds, {"action": f["action"], "patient": f["patient"]}, "agent")
        wob = q(bounds, {"agent": f["agent"], "action": f["action"]}, "patient")
        wac = q(bounds, {"agent": f["agent"], "patient": f["patient"]}, "action")
        qa_ok += int(who == f["agent"] and wob == f["patient"] and wac == f["action"])
        used = set(w for fc in facts for w in fc.values())
        spare = [w for w in words if w not in used]
        ctrl_ok += int(q(bounds, {"action": spare[0], "patient": spare[1]}, "agent") is None)
        tot += 1
    return qa_ok / tot, ctrl_ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", type=str, required=True)
    ap.add_argument("--vocab", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--n-facts", type=int, default=2)
    a = ap.parse_args()
    if not os.path.exists(a.bridge):
        print(f"CANNOT-CONCLUDE: bridge {a.bridge} not found", flush=True); return
    xp, backend = get_backend()
    words = load_vocab(a.vocab)
    print(f"=== real-substrate QA ({len(words)} concepts, backend={backend}, seed={a.seed}) ===", flush=True)

    bridge = SP.build_sparse_pool_bridge(seed=a.seed, n_lang_input=N_LANG, n_shared_pool=N_POOL,
                                         n_lang_output=N_LANG, verbose=False)
    bridge.load_checkpoint(a.bridge)   # validates architecture -> mismatch caught not silent
    real = capture_real_codes(bridge, words, a.seed, xp)
    synth = synthetic_codes(words, a.seed)

    # how close are the REAL captured codes to the idealized sparse patterns? (drive-echo sanity)
    cos_real_synth = float(np.mean([float(np.dot(real[w], synth[w])) for w in words]))
    btw_real = np.mean([float(np.dot(real[words[i]], real[words[j]]))
                        for i in range(len(words)) for j in range(i+1, len(words))])
    print(f"  mean cos(real, synthetic) per-word = {cos_real_synth:.3f}  |  real between-concept cos = "
          f"{btw_real:.3f}", flush=True)

    qa_r, ctrl_r = run_qa(real, words, a.seed, a.n_trials, a.n_facts, xp)
    qa_s, ctrl_s = run_qa(synth, words, a.seed, a.n_trials, a.n_facts, xp)
    print(f"  REAL-code   QA={qa_r:.3f}  abstention-control={ctrl_r:.3f}", flush=True)
    print(f"  SYNTH-code  QA={qa_s:.3f}  abstention-control={ctrl_s:.3f}  (chance {1.0/len(words):.4f})",
          flush=True)
    print(f"\nRESULT: real QA={qa_r:.3f}/ctrl={ctrl_r:.3f}  vs  synth QA={qa_s:.3f}/ctrl={ctrl_s:.3f}",
          flush=True)
    if qa_r >= 0.80 and ctrl_r >= 0.80:
        print("VERDICT: RESOLVES -- genuine spiking relational composition + abstention works on the REAL "
              "deployed substrate concept codes, not just synthetic sparse. Scale to 160.", flush=True)
    elif qa_r >= 0.50:
        print(f"VERDICT: PARTIAL -- real-code QA {qa_r:.2f} (synth {qa_s:.2f}); real-substrate structure "
              "degrades it. Characterize the gap honestly.", flush=True)
    else:
        print(f"VERDICT: real-code QA {qa_r:.2f} << synth {qa_s:.2f} -- real codes too overlapping/noisy "
              "for the bind at this operating point. Honest boundary.", flush=True)


if __name__ == "__main__":
    main()
