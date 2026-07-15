"""2026-07-15 — the CORRECT-TARGET wire-in rung (from the target-correction finding): learn the FLUID console's
DISCOURSE-MARKER -> intent routing over SEMANTIC marker codes, and generalise to a HELD-OUT SYNONYM marker -- the
open-vocabulary capability a fixed keyword set CANNOT do.

THE HAND ROUTER (`FluidChat.turn`): keyword-set checks -- `{share,common}`->SHARE, `{compare,different,difference}`
->COMPARE, `{classify,trace,ancestry,ultimately}`->TAXONOMY, else -> the ALREADY-NEURAL wh->type parse (Phase-7). The
keyword sets are hand-coded + CLOSED: an unlisted synonym ("versus","unlike","akin","lineage") mis-routes. Replace the
keyword checks with a LEARNED deep-credit classifier over the marker's SEMANTIC code, so a novel synonym routes by
semantic proximity to its intent group.

FAITHFUL geometry (mirrors the PPMI stream-cortex codes, already GO): each intent group shares a SEMANTIC block (compare
/different/versus cluster) + a per-word identity block. Hold out one synonym per group from training -> it must route to
the correct intent via its shared semantic block (NOT memorised identity). Anti-cheats: permuted labels -> chance; a
semantically-OOD marker (no group block) -> the NEURAL-FALLTHROUGH class (the moat: not forced into a marker intent).

GATES (6-seed): train-parity >> chance; HELD-OUT-SYNONYM routes correctly (the load-bearing open-vocab test) >> a
1-NN memorisation floor; OOD marker -> fallthrough (moat); permuted -> chance.

Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._discourse_marker_dispatch_derisk
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T
from numpy.linalg import norm

# the FLUID console's real discourse-marker intent groups (line 455/464/486) + a fallthrough (wh-parse) class.
# each group: the ATTESTED keyword-set markers (train) + a HELD-OUT synonym (never trained -> the open-vocab test).
GROUPS = {
    "SHARE":    {"train": ["share", "common", "both"],              "heldout": "alike"},
    "COMPARE":  {"train": ["compare", "different", "difference"],   "heldout": "versus"},
    "TAXONOMY": {"train": ["classify", "trace", "ancestry"],        "heldout": "lineage"},
    "FALL":     {"train": ["what", "who", "does", "eat"],           "heldout": "chase"},   # wh/content -> neural parse
}
INTENTS = list(GROUPS.keys())
D_SEM = 24        # shared per-group SEMANTIC block (the PPMI cluster) -> intent is semantic-readable
D_ID = 24         # per-word identity block
SPARS = 0.25


def _sparse(rng, D):
    v = np.zeros(D); k = max(1, int(SPARS * D)); v[rng.choice(D, k, replace=False)] = 1.0; return v


def build_task(seed, ood_per_group=2):
    rng = np.random.default_rng(seed * 613 + 5)
    sem = {g: _sparse(rng, D_SEM) for g in INTENTS}
    ood_sem = _sparse(rng, D_SEM)                                  # a semantic block NO group teaches (moat->FALL)
    X, y, held, is_ood, names = [], [], [], [], []
    for gi, g in enumerate(INTENTS):
        for w in GROUPS[g]["train"]:
            X.append(np.concatenate([sem[g], _sparse(rng, D_ID)])); y.append(gi)
            held.append(False); is_ood.append(False); names.append(w)
        # the held-out SYNONYM: shares the group SEMANTIC block, NEW identity, NOT in train -> open-vocab generalisation
        w = GROUPS[g]["heldout"]
        X.append(np.concatenate([sem[g], _sparse(rng, D_ID)])); y.append(gi)
        held.append(True); is_ood.append(False); names.append(w)
    # OOD markers: a novel semantic block -> must route to FALL (the neural-parse fallthrough = the moat, not a marker intent)
    fall_i = INTENTS.index("FALL")
    for j in range(ood_per_group):
        X.append(np.concatenate([ood_sem, _sparse(rng, D_ID)])); y.append(fall_i)
        held.append(False); is_ood.append(True); names.append(f"ood{j}")
    return (np.array(X), np.array(y), np.array(held, bool), np.array(is_ood, bool), names, len(INTENTS))


def run_one(seed, hidden=48, epochs=120, lr=0.05, in_gain=1.0):
    X, y, held, is_ood, names, F = build_task(seed)
    tr = ~held                                                     # train on attested markers + OOD-as-fallthrough
    Xtr, ytr = X[tr], y[tr]
    Xtr_n, Xev_n = standardize(Xtr, X)
    n_in = X.shape[1]
    out = {"seed": seed, "n_in": n_in, "F": F, "chance": round(1.0 / F, 4), "intents": INTENTS,
           "n_train": int(tr.sum()), "n_heldout_syn": int(held.sum()), "n_ood": int(is_ood.sum())}
    # THE LEARNED MARKER ROUTER: deep 2-hidden e-prop spiking classifier
    lb = _train_snn(Xtr_n, ytr, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["parity_train"], _, _ = score_snn(lb, Xev_n[tr], y[tr], np.zeros(int(tr.sum()), bool), in_gain)
    # THE LOAD-BEARING open-vocab test: held-out SYNONYM markers route to the correct intent via the shared semantic block
    out["heldout_synonym_acc"], _, _ = score_snn(lb, Xev_n[held], y[held], np.zeros(int(held.sum()), bool), in_gain)
    # the moat: OOD markers -> FALL (neural fallthrough), 0 forced-into-a-marker-intent
    out["ood_to_fallthrough_acc"], _, _ = score_snn(lb, Xev_n[is_ood], y[is_ood], np.zeros(int(is_ood.sum()), bool), in_gain)
    # anti-cheat memorisation floor: 1-NN on the RAW code (held-out synonym has NO same-identity train neighbour ->
    # only shared SEMANTIC block can route it; 1-NN keys on identity too -> weaker)
    hi = np.where(held)[0]

    def nn(xq):
        d = [norm(xq - Xtr[i]) for i in range(len(Xtr))]; return ytr[int(np.argmin(d))]
    out["memfloor_heldout"] = round(float(np.mean([nn(X[i]) == y[i] for i in hi])), 4) if len(hi) else 0.0
    # anti-cheat permuted: shuffle labels -> parity collapses
    rp = np.random.default_rng(seed + 17); yp = ytr[rp.permutation(len(ytr))]
    lp = _train_snn(Xtr_n, yp, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["permuted_parity"], _, _ = score_snn(lp, Xev_n[tr], y[tr], np.zeros(int(tr.sum()), bool), in_gain)
    out["GO"] = bool(out["parity_train"] >= 0.9 and out["heldout_synonym_acc"] >= 0.75
                     and out["heldout_synonym_acc"] > out["memfloor_heldout"] + 0.15
                     and out["ood_to_fallthrough_acc"] >= 0.9 and out["permuted_parity"] <= out["chance"] + 0.2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="research/findings/raw/_discourse_marker_dispatch.json")
    a = ap.parse_args()
    rows = [run_one(s, hidden=a.hidden, epochs=a.epochs) for s in a.seeds]
    for r in rows:
        print(f"[marker s{r['seed']}] chance={r['chance']} parity={r['parity_train']:.3f} "
              f"HELDOUT-SYN={r['heldout_synonym_acc']:.3f} (memfloor {r['memfloor_heldout']:.3f}) "
              f"ood->fall={r['ood_to_fallthrough_acc']:.3f} permuted={r['permuted_parity']:.3f} "
              f"{'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(r["GO"] for r in rows)
    print(f"[marker] {ngo}/{len(rows)} seeds GO (parity>=.9 & heldout-synonym>=.75 & >memfloor+.15 & ood->fall>=.9 & permuted collapses)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
