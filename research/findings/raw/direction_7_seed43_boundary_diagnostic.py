"""Isolate D7 seed 43's L=5 OI drop: crash-retrain artifact (E_functional-
specific) vs genuine capacity degradation (spread across bridges).

Per-(seed, bridge) within-bridge mean-centred different-concept cosine.
Lower = more orthogonal = cleaner. If seed 43's E_functional is an outlier
vs seeds 42/44 E_functional -> retrain confound. If seed 43 is uniformly
worse across ALL bridges -> genuine seed variance at capacity boundary.
CPU-only on cached production activity.
"""
import os, sys
os.environ["SIM_BACKEND"] = "numpy"
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from research.findings.raw.direction_7_vocab_spec import (
    DIRECTION_7_BRIDGE_A_WORDS, DIRECTION_7_BRIDGE_B_WORDS,
    DIRECTION_7_BRIDGE_C_WORDS, DIRECTION_7_BRIDGE_D_WORDS,
    DIRECTION_7_BRIDGE_E_WORDS,
)
WORDS = {"A_nouns":DIRECTION_7_BRIDGE_A_WORDS, "B_verbs":DIRECTION_7_BRIDGE_B_WORDS,
         "C_adj":DIRECTION_7_BRIDGE_C_WORDS, "D_spatial":DIRECTION_7_BRIDGE_D_WORDS,
         "E_functional":DIRECTION_7_BRIDGE_E_WORDS}
CACHE = "research/findings/raw/direction_7_cache"
SEEDS = [42,43,44]

def cos(a,b):
    na,nb = np.linalg.norm(a),np.linalg.norm(b)
    return 0.0 if na<1e-12 or nb<1e-12 else float(np.dot(a,b)/(na*nb))

def mc_diff_cos(acts):
    words=list(acts.keys())
    means={w:acts[w].mean(0) for w in words}
    common=np.stack(list(means.values())).mean(0)
    means={w:means[w]-common for w in words}
    cs=[cos(means[words[i]],means[words[j]]) for i in range(len(words)) for j in range(i+1,len(words))]
    arr=np.abs(np.asarray(cs)); return arr.mean()

def same_cos(acts):
    cs=[]
    for w,obs in acts.items():
        for i in range(obs.shape[0]):
            for j in range(i+1,obs.shape[0]):
                cs.append(cos(obs[i],obs[j]))
    return float(np.mean(cs))

print("per-(seed,bridge) abs mean-centred different-concept cosine (lower=cleaner):")
print(f"{'bridge':<14}", *[f'seed{s:>6}' for s in SEEDS], "  spread(43 vs 42/44 avg)")
rows={}
for b in WORDS:
    vals={}
    for s in SEEDS:
        d=np.load(f"{CACHE}/activity_full_{b}_seed{s}.npz")
        acts={w:d[w] for w in WORDS[b]}
        vals[s]=mc_diff_cos(acts)
    rows[b]=vals
    other=(vals[42]+vals[44])/2
    delta=vals[43]-other
    flag = "  <-- seed43 WORSE" if delta>0.01 else ("  (seed43 cleaner)" if delta<-0.01 else "")
    print(f"{b:<14}", *[f'{vals[s]:.4f}    ' for s in SEEDS], f" d43={delta:+.4f}{flag}")

print()
print("per-seed mean across 5 bridges (overall geometry quality):")
for s in SEEDS:
    m=np.mean([rows[b][s] for b in WORDS])
    print(f"  seed {s}: {m:.4f}")
