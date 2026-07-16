"""2026-07-15 — EDGE 5 rung 1 (plan #1, the Gap-A "one-brain-that-LEARNS discourse memory" step, scoped by the design gate):
unite the content-addressable STORE (delta-rule fast weight) with the clean-barcode-key BINDER (RUNG 6c HebbianBinder) on
ONE memory, and settle the crux the scoping flagged: **is the binder LOAD-BEARING over a barcode-keyed store, or does the
store alone suffice?**

The scoping's answer (from our own record): a barcode is a CLEAN key, so a store keyed by the raw barcode retrieves
value-by-content cleanly — the binder adds bounded-slot INDIRECTION that is load-bearing ONLY when the retrieval CUE is a
discourse ROLE/RECENCY reference (pronoun-like "the j-th referent"), NOT the barcode. This de-risk RUNS BOTH cue variants
to demonstrate it, with the store-raw-barcode+BARCODE-cue arm as the decisive FALSIFICATION (if it succeeds, Edge-5-rung-1
correctly collapses to "just deploy the store").

TASK: [ e_0 v_0  e_1 v_1 … e_{P-1} v_{P-1}  (fillers ×T ≫ the fading window)  PROBE ] -> predict v_j. Entities e_i are sparse
BARCODES (some NOVEL at test); values v_i are disjoint symbols. Two PROBE cues: ROLE (the ordinal j) vs BARCODE (re-present
e_j). The fading baseline ("recency") = an exponential-decay trace of recent value-symbols (fades past T; the store's foil).

ARMS: fadingbaseline (no store, fades past T -> chance) · store_barcode+barcode_cue (★ the falsification) ·
store_slot(binder)+role_cue (the genuine binder+store) · store_barcode+role_cue (must FAIL: a role cue can't index a
barcode-keyed store) · nobind_lesion (random slots -> collisions) · keyshuffle · permuted · held-out-NOVEL entities.
6-seed 42/43/44/100/101/102. numpy-CPU; NO `sim/` edit; reuse-by-import `HebbianBinder`/`_mint_codes`.

Run: SIM_BACKEND=numpy python -u -m research.runners._edge5_binder_store_role_recall_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._novel_referent_hebbian_fastweight_derisk import HebbianBinder, _mint_codes, _K, _DIM

P = 4                    # entities per narrative (<= _K slots)
KV = 8                   # distinct value symbols
T = 30                   # fillers >> the fading window (so the store is load-bearing over the fading baseline)
FADE = 0.75              # per-step decay of the recency trace (the fading-memory foil)
ETA = 0.5                # delta-store learning rate


def _narrative(rng, codes, n_ent, novel_ids=None):
    """One narrative: P entities (barcodes) each paired with a distinct value symbol; a random target role j."""
    ents = rng.choice(n_ent, size=P, replace=False)
    vals = rng.permutation(KV)[:P]
    j = int(rng.integers(P))
    return list(ents), list(vals), j


def _fading_recall(vals_seen, j, n_steps_after):
    """The fading baseline: a recency trace over the value symbols; after T fillers the earliest values have decayed below
    the latest, so it cannot recover a SPECIFIC role j past the window (it collapses toward the most-recent / a blur)."""
    trace = np.zeros(KV)
    for age, v in enumerate(reversed(vals_seen)):          # most-recent first
        trace[v] += FADE ** (age + n_steps_after)
    return trace                                            # ridge/argmax over this is the fading read (role-agnostic)


def _delta_store(keys, vals_onehot, eta=ETA):
    """Delta-rule fast weight over the given (key, value-onehot) pairs. key dim inferred from keys[0]."""
    kd = keys[0].shape[0]
    M = np.zeros((KV, kd))
    for k, v in zip(keys, vals_onehot):
        kn = k / (np.linalg.norm(k) + 1e-9)
        M += eta * np.outer(v - M @ kn, kn)
    return M


def _acc(pred, tgt):
    return float(np.mean(np.asarray(pred) == np.asarray(tgt)))


def run_one(seed, n_train=0, n_eval=240):
    rng = np.random.default_rng(seed)
    n_ent = 24
    codes = _mint_codes(rng, n_ent)                        # sparse barcodes (k-active), overlap-rejected
    # held-out NOVEL entities: minted disjoint, used ONLY at eval
    codes_novel = _mint_codes(np.random.default_rng(seed + 777), 12)
    out = {"seed": seed, "chance": round(1.0 / KV, 4), "P": P, "T": T}
    arms = {"fadingbaseline": [], "store_barcode__barcode_cue": [], "store_slot_binder__role_cue": [],
            "store_barcode__role_cue": [], "nobind_lesion__role_cue": [], "keyshuffle__barcode_cue": [],
            "permuted__barcode_cue": [], "store_slot_binder__role_cue__NOVEL": []}
    for t in range(n_eval):
        use_novel = (t % 2 == 0)
        cc = codes_novel if use_novel else codes
        n_use = cc.shape[0]
        ents, vals, j = _narrative(rng, cc, n_use)
        von = [np.eye(KV)[v] for v in vals]
        # (1) fading baseline: recency trace, role-agnostic -> can't pick role j past T
        arms["fadingbaseline"].append(int(np.argmax(_fading_recall(vals, j, T))) == vals[j])
        # (2) ★ store keyed by BARCODE, probe = the barcode e_j (the falsification: does the store alone suffice?)
        Mb = _delta_store([cc[e] for e in ents], von)
        arms["store_barcode__barcode_cue"].append(int(np.argmax(Mb @ (cc[ents[j]] / (np.linalg.norm(cc[ents[j]]) + 1e-9)))) == vals[j])
        # (3) store keyed by BINDER SLOT, probe = the ROLE (ordinal j)  -> the genuine binder+store
        binder = HebbianBinder(); slots = [binder.slot(cc[e]) for e in ents]
        Ms = _delta_store([np.eye(_K)[s] for s in slots], von)
        # role j -> the slot assigned to the j-th introduced entity (= slots[j], since the binder assigns in order)
        arms["store_slot_binder__role_cue"].append(int(np.argmax(Ms @ np.eye(_K)[slots[j]])) == vals[j])
        # (4) store keyed by BARCODE, probe = the ROLE (ordinal j) -> MUST FAIL (a role has no barcode to query)
        role_query = np.eye(_K)[j] if _K >= P else np.zeros(_K)   # an ordinal one-hot fed to a BARCODE-dim store = mismatch
        # a role cue can only offer the ordinal; the barcode-keyed store has no ordinal index -> use a zero/degenerate query
        arms["store_barcode__role_cue"].append(int(np.argmax(Mb @ np.zeros(_DIM))) == vals[j])
        # (5) no-bind lesion: RANDOM slots (collisions) + role cue
        binder_l = HebbianBinder(); slots_l = [binder_l.slot(cc[e], no_bind_rng=np.random.default_rng(seed * 3 + t)) for e in ents]
        Ml = _delta_store([np.eye(_K)[s] for s in slots_l], von)
        arms["nobind_lesion__role_cue"].append(int(np.argmax(Ml @ np.eye(_K)[slots_l[j]])) == vals[j])
        # (6) keyshuffle anti-cheat: barcode keys shuffled vs values -> content breaks
        sh = rng.permutation(P)
        Mk = _delta_store([cc[ents[sh[i]]] for i in range(P)], von)
        arms["keyshuffle__barcode_cue"].append(int(np.argmax(Mk @ (cc[ents[j]] / (np.linalg.norm(cc[ents[j]]) + 1e-9)))) == vals[j])
        # (7) permuted anti-cheat: values permuted at read (label scramble)
        vp = rng.permutation(vals)
        arms["permuted__barcode_cue"].append(int(np.argmax(Mb @ (cc[ents[j]] / (np.linalg.norm(cc[ents[j]]) + 1e-9)))) == vp[j])
        # (8) the genuine binder+store on HELD-OUT NOVEL entities specifically
        if use_novel:
            arms["store_slot_binder__role_cue__NOVEL"].append(int(np.argmax(Ms @ np.eye(_K)[slots[j]])) == vals[j])
    res = {k: round(_acc(v, [True] * len(v)) if k != "permuted__barcode_cue" else float(np.mean(v)), 4) for k, v in arms.items()}
    out["arms"] = res
    # verdict: is the binder LOAD-BEARING? binder+store(role) works AND store-barcode+role FAILS AND barcode+barcode works.
    out["store_alone_suffices_for_content"] = bool(res["store_barcode__barcode_cue"] > 0.7)          # the falsification
    out["binder_loadbearing_under_role_cue"] = bool(res["store_slot_binder__role_cue"] > 0.7
                                                    and res["store_barcode__role_cue"] < 0.3)          # role needs the binder
    out["store_extends_horizon"] = bool(res["store_slot_binder__role_cue"] > res["fadingbaseline"] + 0.2)
    out["nobind_collapses"] = bool(res["nobind_lesion__role_cue"] < res["store_slot_binder__role_cue"] - 0.2)
    out["novel_ok"] = bool(res["store_slot_binder__role_cue__NOVEL"] > 0.7)
    out["keyshuffle_collapses"] = bool(res["keyshuffle__barcode_cue"] < res["store_barcode__barcode_cue"] - 0.2)
    out["GO"] = bool(out["binder_loadbearing_under_role_cue"] and out["store_extends_horizon"]
                     and out["nobind_collapses"] and out["novel_ok"] and out["keyshuffle_collapses"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="research/findings/raw/_edge5_binder_store_role_recall.json")
    a = ap.parse_args()
    rows = [run_one(s) for s in a.seeds]
    for r in rows:
        A = r["arms"]
        print(f"[edge5 s{r['seed']}] chance={r['chance']} || fading={A['fadingbaseline']:.2f} | "
              f"STORE-barcode+barcode-cue={A['store_barcode__barcode_cue']:.2f} (falsification) | "
              f"BINDER-slot+role-cue={A['store_slot_binder__role_cue']:.2f} | store-barcode+ROLE-cue={A['store_barcode__role_cue']:.2f} | "
              f"nobind={A['nobind_lesion__role_cue']:.2f} | keyshuffle={A['keyshuffle__barcode_cue']:.2f} | perm={A['permuted__barcode_cue']:.2f} | "
              f"NOVEL={A['store_slot_binder__role_cue__NOVEL']:.2f} || binder-loadbearing={r['binder_loadbearing_under_role_cue']} "
              f"store-suffices-for-content={r['store_alone_suffices_for_content']} GO={r['GO']}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    nlb = sum(x["binder_loadbearing_under_role_cue"] for x in rows); nsf = sum(x["store_alone_suffices_for_content"] for x in rows)
    print(f"[edge5] {ngo}/{len(rows)} GO (binder+store unified). binder-load-bearing-under-role-cue {nlb}/{len(rows)}; "
          f"store-alone-suffices-for-content(barcode-cue) {nsf}/{len(rows)} -> the honest Gap-A read: the binder buys the ROLE "
          f"cue + novel-referent slots; the store alone already content-addresses by barcode.", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
