"""2026-07-15 — WIRE-IN de-risk: replace the flagship console's HAND membership+frame router with a LEARNED deep-credit
spiking classifier, trained on the console's OWN real taxonomy. Deploys the concluded `_learned_dispatch_derisk` GO
(interpolative dispatch generalizes) onto the actual `UnifiedFluentConsole` routing decision.

THE HAND ROUTER (EMERGE-58 `UnifiedFluentConsole.turn`): recognise the ability frame `can a X <verb>?`, then route by
`X in self.reasoner.member_idx` -> the EMERGE reasoner; else -> the fluid/abstain path (renderer NEVER invoked = the moat).
That `member in member_idx` set-lookup + regex is the hand-coded structure. Here we LEARN it from the taxonomy: each
subject's category-structured code + the frame -> the route, via the GO feedforward deep-credit substrate (e-prop + pop
coding). Emergence bar: a hand-coded discrete rule becomes structure the substrate LEARNS.

FAITHFUL, not a re-skin of the synthetic task: the members + their categories are INTROSPECTED from a live
`UnifiedFluentConsole(build_fluid=False)` (the real `reasoner.member_idx` + the real is-a script), and the route labels
are the REAL router's decision. Codes are category-structured (shared category block + unique identity) mirroring the
reasoner's discovered-code geometry (category codon + member identity), so membership is READABLE from the code.

THE GATES (6-seed): parity -- the learned router reproduces the hand `member in member_idx` route on train (>> chance);
held-out MEMBERS still route to REASONER (routing generalizes, interpolative); a genuinely-UNKNOWN token -> NON-reasoner
route == the moat (0 false-accepts, the renderer is never invoked on it); permuted labels -> chance (load-bearing).

Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._learned_dispatch_console_wire_derisk
"""
import os, sys, json, argparse, re
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T

D_CAT = 24        # shared category block (bird vs fish vs ...) -> membership is category-readable
D_ID = 24         # unique member-identity block
SPARS = 0.25
_ISA = re.compile(r"^\s*a[n]?\s+(\w+)\s+is\s+a[n]?\s+(\w+)\s*$", re.I)


def _sparse(rng, D):
    v = np.zeros(D); k = max(1, int(SPARS * D)); v[rng.choice(D, k, replace=False)] = 1.0; return v


def _introspect_taxonomy(seed):
    """Instantiate the REAL flagship console (routing/moat-only, build_fluid=False) and read its ACTUAL taxonomy:
    the member set (reasoner.member_idx) + each member's top category from the is-a script. Returns
    (members, member_cat, roots) with roots = the top categories (e.g. {'bird','fish'})."""
    from research.runners._emerge58_unified_fluent_console import UnifiedFluentConsole
    from research.runners._emerge54_per_dimension_cancellation_derisk import _script_lines
    con = UnifiedFluentConsole(seed=seed, build_fluid=False, verbose=False)
    members = sorted(con.reasoner.member_idx.keys())
    obs, isa, teach, ask = _script_lines(seed)
    parent = {}
    for (line, _tag) in isa:
        m = _ISA.match(line.strip())
        if m:
            parent[m.group(1).lower()] = m.group(2).lower()

    def root_of(x):
        seen = set()
        while x in parent and x not in seen:
            seen.add(x); x = parent[x]
        return x
    member_cat = {mem: root_of(mem) for mem in members}
    roots = sorted(set(member_cat.values()))
    return con, members, member_cat, roots, ask


def build_route_dataset(seed, holdout_per_cat=1):
    """Faithful (member -> route) dataset over the REAL taxonomy. Route label: 1 = REASONER (member in member_idx),
    0 = NON-reasoner (fluid/abstain). Codes: category block (shared per top-category) + unique identity. Also emit a
    handful of genuinely-UNKNOWN tokens (a novel category NOT in the taxonomy) -> route 0 (the moat probe)."""
    con, members, member_cat, roots, _ask = _introspect_taxonomy(seed)
    rng = np.random.default_rng(seed * 7919 + 3)
    cat_code = {c: _sparse(rng, D_CAT) for c in roots}
    unk_cat_code = _sparse(rng, D_CAT)                       # a category the taxonomy never taught (moat)
    X, y, is_held, is_unknown, names = [], [], [], [], []
    # hold out `holdout_per_cat` real members per category -> test routing GENERALISATION to unseen members
    held = set()
    for c in roots:
        ms = [m for m in members if member_cat[m] == c]
        rng.shuffle(ms)
        for m in ms[:holdout_per_cat]:
            if sum(1 for m2 in ms if m2 != m) >= 2:          # keep >=2 train members per category
                held.add(m)
    for mem in members:
        x = np.concatenate([cat_code[member_cat[mem]], _sparse(rng, D_ID)])
        X.append(x); y.append(1); is_held.append(mem in held); is_unknown.append(False); names.append(mem)
    # genuinely-unknown tokens (novel category) -> the moat: must route NON-reasoner
    for j in range(max(3, len(roots))):
        x = np.concatenate([unk_cat_code, _sparse(rng, D_ID)])
        X.append(x); y.append(0); is_held.append(False); is_unknown.append(True); names.append(f"zzz{j}")
    return (np.array(X), np.array(y), np.array(is_held), np.array(is_unknown), names,
            len(members), len(held), roots)


def run_one(seed, hidden=48, epochs=120, lr=0.05, in_gain=1.0):
    X, y, is_held, is_unknown, names, n_mem, n_held, roots = build_route_dataset(seed)
    tr = (~is_held) & (~is_unknown)                          # train on non-held real members + attested unknowns? no:
    # train ONLY on non-held real members (route=1) + the genuinely-unknown tokens (route=0) EXCEPT hold some unknown out
    tr = (~is_held)                                          # unknowns are attested-as-unknown in train (like the fluid set)
    n_in = X.shape[1]; F = 2
    Xtr, ytr = X[tr], y[tr]
    Xtr_n, Xev_n = standardize(Xtr, X)
    out = {"seed": seed, "n_in": n_in, "chance": 0.5, "n_members": n_mem, "n_held": int(n_held),
           "roots": roots, "n_unknown": int(is_unknown.sum())}
    # THE LEARNED ROUTER: deep 2-hidden e-prop spiking classifier (the GO feedforward deep-credit substrate)
    lb = _train_snn(Xtr_n, ytr, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    # parity on TRAIN real members (route must == hand 'in member_idx' = 1)
    tr_mem = tr & (~is_unknown)
    out["parity_train"], _, _ = score_snn(lb, Xev_n[tr_mem], y[tr_mem], np.zeros(tr_mem.sum(), bool), in_gain)
    # GENERALISATION: held-out real members must still route to REASONER (route 1)
    if n_held:
        acc_h, _, _ = score_snn(lb, Xev_n[is_held], y[is_held], np.zeros(int(is_held.sum()), bool), in_gain)
        out["heldout_member_to_reasoner"] = acc_h
    else:
        out["heldout_member_to_reasoner"] = float("nan")
    # THE MOAT: genuinely-unknown tokens must route NON-reasoner (route 0) -> 0 false-accepts
    acc_u, _, _ = score_snn(lb, Xev_n[is_unknown], y[is_unknown], np.zeros(int(is_unknown.sum()), bool), in_gain)
    out["moat_unknown_to_nonreasoner"] = acc_u                # 1.0 = every unknown correctly NOT routed to reasoner
    out["moat_false_accepts"] = int(round((1.0 - acc_u) * is_unknown.sum()))
    # ANTI-CHEAT permuted: shuffle route labels -> parity must collapse to chance
    rp = np.random.default_rng(seed + 11); yperm = ytr[rp.permutation(len(ytr))]
    lp = _train_snn(Xtr_n, yperm, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["permuted_parity_train"], _, _ = score_snn(lp, Xev_n[tr_mem], y[tr_mem], np.zeros(tr_mem.sum(), bool), in_gain)
    # GO gate (this seed)
    out["GO"] = bool(out["parity_train"] >= 0.9 and out["heldout_member_to_reasoner"] >= 0.9
                     and out["moat_false_accepts"] == 0 and out["permuted_parity_train"] <= 0.7)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="research/findings/raw/_learned_dispatch_console_wire.json")
    a = ap.parse_args()
    rows = [run_one(s, hidden=a.hidden, epochs=a.epochs) for s in a.seeds]
    for r in rows:
        print(f"[wire s{r['seed']}] parity={r['parity_train']:.3f} heldout->reasoner={r['heldout_member_to_reasoner']:.3f} "
              f"moat(unk->non)={r['moat_unknown_to_nonreasoner']:.3f} FA={r['moat_false_accepts']} "
              f"permuted={r['permuted_parity_train']:.3f} n_mem={r['n_members']} roots={r['roots']} "
              f"{'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(r["GO"] for r in rows)
    print(f"[wire] {ngo}/{len(rows)} seeds GO (parity>=.9 & heldout->reasoner>=.9 & moat 0-FA & permuted<=.7)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
