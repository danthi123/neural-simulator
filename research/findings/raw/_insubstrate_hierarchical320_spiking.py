"""Full 320-concept biological composition IN SPIKING, via the hierarchical bridge-role bind (no retrain).

Cheap-first (numpy) already showed: binding each concept with its bridge's role vector makes the 5 shared-
pattern bridges' 320 codes DISTINCT (max-cos 1.000 -> 0.323) AND composable (algebra QA 1.000). This is the
SPIKING confirmation: capture the REAL 320 codes (5 bridges x 64, temporal integration), apply the hierarchical
bridge-role bind, then run the validated SPIKING relational memory + wh-QA + abstention over ALL 320 concepts
in one cleanup space -- the full brain-analogue conversational substrate.

Captures are cached to npz (one-time, slow) so re-runs are fast. Reuse-by-import; no protected-module change;
no autograd. Run (GPU): python -m research.findings.raw._insubstrate_hierarchical320_spiking
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_real_substrate_qa_probe as Q
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.runners.concept_pool_sparse_distributed as SP
from sim.backend import get_backend, to_host

ROOT = "research/findings/raw/g11_bg"
BRIDGES = [("bridgeA_nouns", "bridgeA_nouns_sparse64"), ("bridgeB_verbs", "bridgeB_verbs_sparse64"),
           ("bridgeC_adj", "bridgeC_adj_sparse64"), ("bridgeD_spatial", "bridgeD_spatial_sparse64"),
           ("bridgeE_functional", "bridgeE_functional_sparse64")]
VOCABS = ["g20_bridgeA_nouns_vocab64", "g20_bridgeB_verbs_vocab64", "g20_bridgeC_adj_vocab64",
          "g20_bridgeD_spatial_vocab64", "g20_bridgeE_functional_vocab64"]
CACHE = "research/findings/raw/_hier320_codes.npz"
N_TRIALS = 20


def _mc(v):
    v = np.asarray(v, dtype=np.float64); v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-12)


def main():
    Q.STIM = 300; Q.SPARSITY = 0.007
    xp, backend = get_backend()
    D = Q.N_POOL
    rng = np.random.default_rng(42)
    bridge_roles = [rng.choice([-1.0, 1.0], size=D) / np.sqrt(D) for _ in range(len(BRIDGES))]

    if os.path.exists(CACHE):
        d = np.load(CACHE); hier = {w: d[w] for w in d.files if w != "_words"}; words = list(d["_words"])
        print(f"=== hierarchical-320 spiking (loaded {len(words)} cached codes, backend={backend}) ===",
              flush=True)
    else:
        print(f"=== hierarchical-320 spiking: capturing 320 real codes (5 bridges, backend={backend}) ===",
              flush=True)
        hier = {}; words = []
        for bi, (name, bf) in enumerate(BRIDGES):
            bp = f"{ROOT}/g20_sparse_bridges_320/{bf}.simstate.h5"
            vp = f"{ROOT}/{VOCABS[bi]}.txt"
            if not os.path.exists(bp):
                print(f"CANNOT-CONCLUDE: {bp} missing", flush=True); return
            vwords = Q.load_vocab(vp)
            bridge = SP.build_sparse_pool_bridge(seed=42, n_lang_input=Q.N_LANG, n_shared_pool=Q.N_POOL,
                                                 n_lang_output=Q.N_LANG, verbose=False)
            bridge.load_checkpoint(bp)
            codes = Q.capture_real_codes(bridge, vwords, 42, xp)   # real code per word (temporal integration)
            for w in vwords:
                key = w if w not in hier else f"{name[:1]}_{w}"
                hier[key] = _mc(bridge_roles[bi] * codes[w])       # HIERARCHICAL: bridge_role (Hadamard) code
                words.append(key)
            print(f"  captured + bound {len(vwords)} from {name} (total {len(words)})", flush=True)
            del bridge
            try:
                import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        np.savez_compressed(CACHE, _words=np.array(words), **hier)
        print(f"  cached -> {CACHE}", flush=True)

    V = len(words)
    import itertools
    btw = [float(np.dot(hier[a], hier[b])) for a, b in itertools.islice(itertools.combinations(words, 2), 40000)]
    print(f"  320-wide between-concept cos: mean {np.mean(btw):.3f}  max {np.max(btw):.3f}", flush=True)
    if np.max(btw) > 0.9:
        print("  WARNING: near-duplicate codes remain -> hierarchical bind insufficient in real codes", flush=True)

    # spiking relational composition over all 320 hierarchical codes
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bb, bidx = P.build(42, D, xp)

    def q(bounds, given, qr):
        for b in bounds:
            if all(RM.unbind_spiking(bb, bidx, b, r, roles, hier, words, D, xp) == w for r, w in given.items()):
                return RM.unbind_spiking(bb, bidx, b, qr, roles, hier, words, D, xp)
        return None

    qa_ok = ctrl_ok = tot = 0
    for _ in range(N_TRIALS):
        pk = rng.choice(V, 6, replace=False)
        facts = [{"agent": words[pk[3*f]], "action": words[pk[3*f+1]], "patient": words[pk[3*f+2]]}
                 for f in range(2)]
        bounds = [RM.bind_fact_spiking(bb, bidx, fc, hier, roles, D, xp) for fc in facts]
        f = facts[rng.integers(2)]
        who = q(bounds, {"action": f["action"], "patient": f["patient"]}, "agent")
        qa_ok += int(who == f["agent"])
        used = set(w for fc in facts for w in fc.values()); spare = [w for w in words if w not in used]
        ctrl_ok += int(q(bounds, {"action": spare[0], "patient": spare[1]}, "agent") is None)
        tot += 1
    qa, ctrl = qa_ok / tot, ctrl_ok / tot
    print(f"\nRESULT: SPIKING 320-concept relational QA (who) = {qa:.3f}   abstention = {ctrl:.3f}  "
          f"(chance {1.0/V:.4f})", flush=True)
    if qa >= 0.80 and ctrl >= 0.80:
        print("VERDICT: RESOLVES -- biological spiking composition works over ALL 320 concepts in one cleanup "
              "space via the hierarchical bridge-role bind (no retrain). The brain-analogue conversational "
              "substrate scales to 320 concepts.", flush=True)
    else:
        print(f"VERDICT: QA {qa:.2f} / ctrl {ctrl:.2f} at 320 spiking -- characterize the gap vs the algebra "
              "1.000 (spiking SNR at 320-way cleanup).", flush=True)


if __name__ == "__main__":
    main()
