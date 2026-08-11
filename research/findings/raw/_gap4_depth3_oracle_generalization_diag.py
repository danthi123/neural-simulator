"""LEAN diagnostic: is the depth-3 RATE oracle collapse fixable by moderate hyperparameter tuning?
Train a 3-hidden DendriticMLP on the SAME compositional-inheritance task across a few (lr, epochs, width)."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-a082c27adbf577c33")
import numpy as np
from research.runners._semantic_inheritance_deep_credit_derisk import (
    make_task_semantic_inheritance, _train_oracle, _acc_on)
from sim.dendritic_mlp import DendriticMLP

seed = 42
tk = dict(n_super=12, n_members=8, held_per_super=3, n_prop=2, member_id_dim=3, n_obs=16, noise=0.02, feature_seed=0)
(Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(seed, **tk)
k = meta["k_classes"]; n_in = Xtr.shape[1]; inh = idx["inh_idx"]
yv = yte[inh]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
print("n_in=%d k=%d n_inherit_heldout=%d chance=%.3f" % (n_in, k, len(inh), chance), flush=True)

def train_eval(sizes, epochs, lr):
    net = DendriticMLP(sizes, seed=seed)
    _train_oracle(net, Xtr, ytr, epochs, lr, 128, seed)
    return float(net.accuracy(Xtr, ytr)), _acc_on(net, Xte, yte, inh)

print("\n--- depth ladder at DEFAULT (epochs=250, lr=0.3, hidden=96) [run_seed's oracle setting] ---", flush=True)
for nh in (1, 2, 3):
    tr, te = train_eval([n_in] + [96]*nh + [k], 250, 0.3)
    print("  %d-hidden: train=%.3f inherit=%.3f" % (nh, tr, te), flush=True)

print("\n--- 3-hidden: can moderate tuning reach the ceiling (>=0.80)? ---", flush=True)
for lr, epochs in [(0.05, 800), (0.1, 800), (0.05, 250), (0.1, 250), (0.3, 800)]:
    tr, te = train_eval([n_in, 96, 96, 96, k], epochs, lr)
    print("  lr=%.2f ep=%-4d : train=%.3f inherit=%.3f%s"
          % (lr, epochs, tr, te, "  <== CEILING" if te >= 0.80 else ""), flush=True)

print("\n--- 3-hidden wider (width=192) at lr=0.1 ep=800 ---", flush=True)
tr, te = train_eval([n_in, 192, 192, 192, k], 800, 0.1)
print("  width=192: train=%.3f inherit=%.3f%s" % (tr, te, "  <== CEILING" if te >= 0.80 else ""), flush=True)
print("DONE", flush=True)
