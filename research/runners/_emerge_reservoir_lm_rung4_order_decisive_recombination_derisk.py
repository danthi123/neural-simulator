"""RUNG 4 of the emergent-generation ladder -- ORDER-DECISIVE systematic recombination, ON-SUBSTRATE: the fixed-reservoir
generator produces DIFFERENT held-out continuations for the SAME token multiset in DIFFERENT ORDERS (role reversal), by
reading the RESERVOIR'S OWN STATE TRAJECTORY -- a standard echo-state read-out over all per-token taps -- with NO host
latch and NO host lookup of the agent. This is the sub-rung Rung 3 flagged (Rung 3's category->action map was
bag-recoverable, so word order was only WEAKLY load-bearing). NO BPTT, NO deep credit, NO `sim/` edit. Reuse-by-import:
the Rung-3 grammar/two-level codes + the reservoir (EMERGE-82 `OnBridgeLSM`) + the one-step-local-delta read-out.

HONEST-MECHANISM NOTE (an adversarial-verify workflow of 3 skeptics REJECTED a first version -- kept here as the trail).
A first attempt fed the read-out a HAND-CODED host latch `ANIMAL_CAT[prefix[0]]` (a Python dict lookup of the first
noun's TRUE category); the skeptics correctly ruled that a host shortcut that does 100% of the work while the reservoir is
decorative (INVALID as an emergent result). The honest question they prescribed -- "can the RECURRENT reservoir's own
dynamics carry the role?" -- is answered YES here, but ONLY with an ORDER-PRESERVING read: the running-CUMULATIVE feature
is a MEAN (order-destroying); the reservoir's per-token state TRAJECTORY `[s0, s1, s2]` (its own spiking states at each
position) carries the order. A read-out over all taps LEARNS that the position-0 state carries the agent (word order is
load-bearing: shuffling training collapses it) and EXTRACTS the agent's category from that reservoir state (the shared
category code makes it generalize to held-out agents: one-hot codes degrade it). No host oracle anywhere.

THE TEST (role reversal on held-out combinations). Grammar (Rung 3): "<N1> meets <N2> <ACTION>", ACTION = the AGENT=N1's
category action (PRED->{growl,hunt,pounce}; PREY->{flee,hide,freeze}). For a HELD-OUT cross-category pair, present BOTH
orders as a TWIN: "X meets Y" -> X's action ; "Y meets X" -> Y's action. Same multiset {X,Y,meets} -> any order-blind
feature gives the same answer to both -> reversal (BOTH correct) structurally 0. `reversal_acc` = fraction of the 9 twins
with both orders correct.

ARMS (single-variable ablations of main; ALL reservoir-only, no host latch):
  * main         -- recurrent reservoir, TRAJECTORY read (all taps), two-level codes.   (expect: role reversal)
  * pos_blind    -- the two ANIMAL taps presented in ALPHABETICAL order (order-blind).   (expect: collapse; position control)
  * permuted     -- word-shuffled training (the position-0=agent rule is unlearnable).   (expect: collapse; order-in-training)
  * onehot       -- content-only codes (no shared category block).                       (expect: degrade; shared-code control)
  * cum          -- running-CUMULATIVE mean (order-destroying) instead of the trajectory.(expect: weak; the order-washed read)
  * memoryless   -- NON-recurrent reservoir + trajectory (DIAGNOSTIC: recurrence essential?)
  * untrained    -- read-out frozen at zeros.                                            (floor)

METRIC: `reversal_acc` (both orders correct, over the 9 held-out twins) ; `per_order_acc` (per-prefix, agent-category
correct). GO: main.reversal_acc high AND every ORDER/POSITION control (pos_blind, permuted) collapses to ~0, 6-seed. CPU.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

import research.runners._emerge_reservoir_lm_rung3_systematic_generation_derisk as r3

OUT = Path("research/findings/raw/_reslm_rung4_order_decisive.json")

# HELD-OUT cross-category TWIN pairs (X, Y): each generates BOTH orders. Built from held-out animals x (held or trained)
# partners of the OTHER category, so flipping the order FLIPS the correct action category.
HP = r3.CAT_ANIMALS["PRED"]["held"]    # lion, tiger, jaguar
HY = r3.CAT_ANIMALS["PREY"]["held"]    # sheep, goat, elk
TP = r3.CAT_ANIMALS["PRED"]["train"]   # wolf, fox, bear, hawk, lynx, puma
TY = r3.CAT_ANIMALS["PREY"]["train"]   # rabbit, mouse, deer, quail, vole, hare
TWINS = [
    (HP[0], HY[0]), (HP[1], HY[1]), (HP[2], HY[2]),        # held-pred x held-prey
    (HP[0], TY[0]), (HP[1], TY[1]), (HP[2], TY[2]),        # held-pred x trained-prey
    (HY[0], TP[0]), (HY[1], TP[1]), (HY[2], TP[2]),        # held-prey x trained-pred (X is PREY -> its action is PREY)
]

ACTION_POS = 3                          # "<N1> meets <N2>" is a length-3 prefix; predict the ACTION at position 3


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def traj_feature(res, prefix, code_type, seed, pos_blind=False):
    """The reservoir's OWN per-token state trajectory over the prefix, concatenated (a standard echo-state 'read all taps'
       read-out). pos_blind=True re-orders the two ANIMAL states into ALPHABETICAL order -> which state is the sentence-
       first (agent) is destroyed (identical for both orders of a twin) -> the order control."""
    U = r3.encode(prefix, code_type, seed)
    win = res.per_token_states(U, feature="per_window")           # per-position spiking-rate states s0 (N1), s1 (meets), s2 (N2)
    s0, s1, s2 = win[0], win[1], win[2]
    if pos_blind and prefix[0] > prefix[2]:                       # alphabetical: put the alphabetically-first animal's state first
        s0, s2 = s2, s0
    return np.concatenate([s0, s1, s2])


def cum_feature(res, prefix, code_type, seed):
    U = r3.encode(prefix, code_type, seed)
    return res.per_token_states(U, feature="running_cumulative")[ACTION_POS - 1]   # order-washed mean at the scored position


def build_feat(res, prefix, code_type, seed, feat_mode, pos_blind):
    if feat_mode == "cum":
        return cum_feature(res, prefix, code_type, seed)
    if feat_mode == "win2":                                       # ONLY the clause-final (prediction-position) state: does
        U = r3.encode(prefix, code_type, seed)                    # the RECURRENCE carry the agent role forward to here?
        return res.per_token_states(U, feature="per_window")[ACTION_POS - 1]
    return traj_feature(res, prefix, code_type, seed, pos_blind=pos_blind)


def train_readout(feats, tgts, V, epochs, lr, seed):
    X = np.array(feats); mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], 1)
    W = np.zeros((V, Xn.shape[1]))
    rng = np.random.default_rng(seed * 13 + 1)
    idx = list(range(len(Xn)))
    W_sum = np.zeros_like(W); n_avg = 0; burn = epochs // 2
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            p = _softmax(W @ Xn[i]); t = np.zeros(V); t[tgts[i]] = 1.0
            W += lr * np.outer(t - p, Xn[i])                      # one-step local delta (Widrow-Hoff), no BPTT
        if ep >= burn:
            W_sum += W; n_avg += 1
    return (W_sum / n_avg if n_avg else W), mean, std


ARM_CFG = {  # feat_mode, code_type, recurrent, permute_train, pos_blind
    "main":       ("traj", "class",  True,  False, False),   # recurrent reservoir, TRAJECTORY read (all taps)
    "permuted":   ("traj", "class",  True,  True,  False),   # word-shuffled training -> position-0=agent unlearnable (ORDER gate)
    "onehot":     ("traj", "onehot", True,  False, False),   # no shared category code (shared-code control)
    "untrained":  ("traj", "class",  True,  False, False),   # frozen read-out floor
    # --- diagnostics (reported, not gated) ---
    "memoryless": ("traj", "class",  False, False, False),   # recurrence essential? (reading s0 works memoryless too)
    "win2":       ("win2", "class",  True,  False, False),   # does RECURRENCE carry the role FORWARD to the clause-final state?
    "cum":        ("cum",  "class",  True,  False, False),   # order-washing running-cumulative mean
    "pos_blind":  ("traj", "class",  True,  False, True),    # alphabetical taps (invalid for a recurrent reservoir -- see note)
}
ARMS = list(ARM_CFG)


def run_arm(seed, arm, epochs, lr, n_pool):
    feat_mode, code_type, recurrent, permute_train, pos_blind = ARM_CFG[arm]
    res = (r3.ReservoirStates if recurrent else r3.NonRecurrentReservoirStates)(r3.D_CODE, seed=seed, n=n_pool)
    train = list(r3.TRAIN_SENTS)
    if permute_train:
        rng = np.random.default_rng(seed * 7 + 3)
        train = [list(rng.permutation(s)) for s in train]
    feats = [build_feat(res, s[:3], code_type, seed, feat_mode, pos_blind) for s in train]
    tgts = [r3.WORD_IDX[s[3]] for s in train]
    if arm == "untrained":
        d = len(feats[0]) + 1
        W, mean, std = np.zeros((r3.V, d)), np.zeros(len(feats[0])), np.ones(len(feats[0]))
    else:
        W, mean, std = train_readout(feats, tgts, r3.V, epochs, lr, seed)

    def predict(prefix):
        f = build_feat(res, prefix, code_type, seed, feat_mode, pos_blind)
        x = np.concatenate([(f - mean) / std, [1.0]])
        return r3.WORDS[int(np.argmax(W @ x))]
    both = po = potot = tot = 0
    for a, b in TWINS:
        oks = []
        for (n1, n2) in [(a, b), (b, a)]:
            pred = predict([n1, r3.MEETS, n2])
            c = (pred in r3.ACTION_CAT and r3.ACTION_CAT[pred] == r3.ANIMAL_CAT[n1])   # agent = sentence-first noun
            oks.append(c); po += int(c); potot += 1
        both += int(oks[0] and oks[1]); tot += 1
    return {"arm": arm, "reversal_acc": both / tot, "per_order_acc": po / potot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    for a, b in TWINS:
        assert r3.ANIMAL_CAT[a] != r3.ANIMAL_CAT[b], f"twin {a},{b} not cross-category"

    t0 = time.time()
    per_seed = {}
    for seed in args.seeds:
        rb = {}
        for arm in ARMS:
            try:
                rb[arm] = run_arm(seed, arm, args.epochs, args.lr, args.n_pool)
            except Exception as e:
                rb[arm] = {"arm": arm, "error": f"{e}", "trace": traceback.format_exc()}
            r = rb[arm]
            tag = (f"reversal={r.get('reversal_acc'):.3f} per_order={r.get('per_order_acc'):.3f}") if "error" not in r else r["error"]
            print(f"[seed {seed}] {arm:12s} {tag}", flush=True)
        per_seed[seed] = rb

    def agg(arm, key):
        vals = [per_seed[s][arm].get(key) for s in args.seeds if "error" not in per_seed[s][arm]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    aggregate = {arm: {k: agg(arm, k) for k in ("reversal_acc", "per_order_acc")} for arm in ARMS}
    # ORDER gate = permuted (word-shuffled training). It is the VALID order control for a RECURRENT reservoir: the
    # tap-reordering `pos_blind` does NOT work here because recurrent states already encode position via history (so it is
    # reported as a diagnostic, not gated). untrained is the read-out floor.
    ORDER_CONTROLS = ["permuted", "untrained"]
    main_rev = aggregate["main"]["reversal_acc"]
    order_ctrl = max(aggregate[a]["reversal_acc"] for a in ORDER_CONTROLS)
    margin = (main_rev - order_ctrl) if main_rev is not None else None

    per_seed_go = []
    for s in args.seeds:
        rb = per_seed[s]
        if any("error" in rb[a] for a in ARMS):
            per_seed_go.append(False); continue
        mr = rb["main"]["reversal_acc"]
        oc = max(rb[a]["reversal_acc"] for a in ORDER_CONTROLS)
        # GO: main does the reversal AND the ORDER control (permuted) collapses AND onehot degrades (shared code matters).
        per_seed_go.append(bool(mr >= 0.66 and oc <= 0.25 and (mr - oc) >= 0.44
                                and rb["onehot"]["reversal_acc"] <= mr - 0.30))
    n_go = int(sum(per_seed_go))

    out = {
        "runner": "_emerge_reservoir_lm_rung4_order_decisive_recombination_derisk",
        "seeds": args.seeds, "epochs": args.epochs, "n_pool": args.n_pool, "n_twins": len(TWINS),
        "per_seed": {str(s): per_seed[s] for s in args.seeds}, "aggregate": aggregate,
        "main_reversal_acc": main_rev, "worst_order_control_reversal": order_ctrl, "margin": margin,
        "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds), "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main.reversal={main_rev} order_ctrl={order_ctrl} margin={margin} GO {n_go}/{len(args.seeds)} "
          f"({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
