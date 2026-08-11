"""EMERGENCE-ENGINE SELECTIVE-WRITE CONTENT-ADDRESSABLE STORE — surpass the on-bridge HTM Temporal-Memory's
chain-integrity horizon (the residual the 2026-08-11 horizon smoke named) with a SELECTIVE-WRITE, capacity-bounded
content-addressable store riding the HTM-TM's OWN (clean) ALLOCATION KEYS, so a BROKEN/ambiguous priming chain is
RECOVERABLE by content-addressed completion.

WHY THIS IS THE FRONTIER (our-own-record first):
  * `2026-08-11-emergence-engine-htm-horizon-...SMOKE` MEASURED the on-bridge HTM-TM horizon (EMERGE-14): it genuinely
    learns high-order structure (clean HOLD at dist 9, htm 1.000, recurrence load-bearing, dAP-lesion 0.000) and is
    NON-FADING but FINITE — it carries a distal cue to dist ~17 at LOW interference, but under interference (n_seq>=3)
    ONE context's priming CHAIN BREAKS by dist 17 (htm 0.667 / 0.750; swap tracks htm, so the failure is a MERGED chain
    genuinely following the wrong cue, not a readout artifact). The residual is CHAIN-INTEGRITY under interference.
  * The NAMED surpass (verbatim, the finding + both prior banked threads converged on it): a SELECTIVE-WRITE
    content-addressable store over the HTM-TM's own (clean) ALLOCATION keys. The delta/STP store extended a horizon only
    given CLEAN keys (the reservoir's keys were diffuse -> NEGATIVE); the HTM-TM's allocation cells ARE clean/allocated,
    which is exactly the key-quality the reservoir lacked. This unifies the two banked threads on the substrate the
    roadmap wants to scale.

MECHANISM (this de-risk):
  KEYS = the HTM-TM's ALLOCATION SDRs at the SHARED-MIDDLE positions (t in 1..L) of the priming chain — the primed,
    context-specific winner-cell subsets the bridge's coincidence recurrence produces (NOT the raw cue input at t=0: that
    would be a trivial cue->branch lookup table on the INPUT, bypassing the HTM; keys are the HTM's OWN high-order
    allocation cells). Early in the middle these are clean/context-specific; late they collide (the chain break).
  SELECTIVE WRITE (novelty/mismatch-gated) = on each CONFIDENT step (the chain is intact -> `active` is a primed subset,
    not the whole-column burst), WRITE (allocation-SDR key -> branch). The gate: if the key already maps to a DIFFERENT
    branch, it is a COLLISION (an ambiguous/merged key) -> POISON it (remove + never re-add). So a merged late key is
    EXCLUDED by construction; only clean, discriminative allocation keys survive. (Biologically: mismatch/novelty-gated
    encoding — do not commit a memory for an ambiguous pattern.)
  READ / COMPLETION = at test, capture the confident middle allocation SDRs; content-address the store, preferring the
    HIGHEST-overlap match, ties broken toward the LATER (freshest, branch-proximal) position; if that best match is
    UNAMBIGUOUS -> emit its branch (a broken 16-step relay recovered from an earlier clean allocation key); if it is
    AMBIGUOUS (only possible when the store was polluted) -> VETO -> fall back to the bare HTM prediction.

ANTI-CHEATS (each EXECUTES via tools.lab; the earned teeth):
  (a) LOAD-BEARING: lesion the store's READ -> collapse back to the bare-HTM horizon (store adds the recovery).
  (b) SELECTIVE-WRITE GATES: an ALWAYS-WRITE control (no mismatch gate) stores the merged ambiguous keys -> the read is
      trapped at an ambiguous late key -> VETO -> bare -> does NOT recover. If always-write ALSO recovers, the value is
      "more memory", NOT selectivity — reported honestly.
  (c) ATTRIBUTABLE to the TRUE drive: PERMUTE the keys' branch labels (a derangement) -> recovery -> chance (the
      completion is driven by the real key->branch structure, not by having any store).
  (d) SWAP-FOLLOWS-CONTEXT (no confabulation): inject a DIFFERENT cue -> the store must complete to the INJECTED cue's
      branch (the allocation SDR reflects the traversed cue), not the memorised one.
  Plus the n-gram floor (pinned at chance by the shared middle) + oracle (task solvable) + multi-seed.

GO = under interference (n_seq>=3) at the FAR distance (dist >= 17) where the bare HTM chain BREAKS (bare < 0.90), the
store RESTORES the horizon: store >= 0.90 AND >= bare + 0.15, store-lesion collapses to ~bare (store load-bearing),
always-write does NOT match store (selectivity load-bearing), permute -> <= chance, swap-follows >= 0.90, oracle > 0.99,
multi-seed. HONEST NEGATIVE (first-class) = the store does not restore the horizon, or selectivity is inert — reported
with numbers + the next mechanism.

Reuse-by-import (EMERGE-14 on-bridge machinery); NO sim/ edit. SIM_BACKEND=numpy (the horizon finding established these
sub-1k-neuron coincidence loops are LAUNCH-BOUND: cupy is SLOWER; CPU/numpy is correct + faster).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, OnBridgeLearner, coincidence_predict)
from research.runners._emerge9b_htm_faithful_derisk import (
    make_overlap_sequences, markov_branch_acc, full_oracle)

try:
    from tools.lab import lever, attributable_to, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond: print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_emerge_selective_write_store.json")


def traverse_capture(lr, seq):
    """Replicate OnBridgeLearner.predict_branch's ON-SUBSTRATE forward pass, ALSO capturing per-position the `active`
    allocation SDR (cells), whether the step was CONFIDENT (primed -> a context-specific subset, not the whole-column
    burst), and the predicted columns. steps[t]['pred_cols'] == predict_branch(seq, ...)[t] (bare-HTM prediction)."""
    predictive = set()
    steps = []
    for c in seq:
        col = lr._col(c)
        primed = [i for i in col if i in predictive] if not lr.lesion else []
        active = set(primed[:lr.k_win]) if primed else set(col)
        confident = len(primed) > 0                       # chain intact -> `active` is a primed context-specific subset
        predictive = coincidence_predict(lr.b, lr.cells_idx, active, lr.N, lr.nE)
        steps.append({"active": frozenset(active), "confident": bool(confident),
                      "pred_cols": set(i // lr.nE for i in predictive)})
    return steps


def _overlap(q, key):
    if not q or not key:
        return 0.0
    return len(q & key) / max(len(q), len(key))           # exact same set -> 1.0; subset vs superset penalised


class ContentStore:
    """Content-addressable heteroassociative store keyed on the HTM-TM's allocation SDRs. Selective (mismatch-gated) or
    always-write. Read = highest-overlap match (ties -> later/fresher position), unambiguous -> emit, ambiguous -> veto."""

    def __init__(self, selective=True):
        self.selective = selective
        self.store = {}                                   # frozenset(cells) -> {branch_col: count}
        self.poisoned = set()

    def write(self, active, branch):
        key = frozenset(active)
        if self.selective:
            if key in self.poisoned:
                return
            d = self.store.get(key)
            if d is None:
                self.store[key] = {branch: 1}
            elif set(d.keys()) == {branch}:
                d[branch] += 1
            else:                                         # COLLISION: an ambiguous/merged key -> poison (mismatch gate)
                del self.store[key]
                self.poisoned.add(key)
        else:                                             # ALWAYS-WRITE: keep the ambiguous key (store pollutes)
            d = self.store.setdefault(key, {})
            d[branch] = d.get(branch, 0) + 1

    def read(self, query_steps, tau):
        """query_steps: list of (t, active_frozenset) confident middle positions, t ascending. Returns a branch column
        or None (abstain/veto -> caller falls back to the bare HTM)."""
        cands = [(_overlap(q, key), t, key) for (t, q) in query_steps for key in self.store]
        cands = [c for c in cands if c[0] >= tau]
        if not cands:
            return None
        ov, t, key = max(cands, key=lambda c: (c[0], c[1]))   # max overlap, tie -> later (fresher, branch-proximal)
        d = self.store[key]
        return next(iter(d)) if len(d) == 1 else None         # UNAMBIGUOUS -> emit; AMBIGUOUS -> veto

    def permuted(self, branches, seed):
        """Attribution control: remap every stored key's branch by a DERANGEMENT of the branch labels (no fixed point)
        -> content-addressed recall now returns a WRONG branch -> accuracy collapses to <= chance if the recovery was
        genuinely driven by the TRUE key->branch structure."""
        rng = np.random.default_rng(seed)
        perm = list(branches)
        for _ in range(64):
            rng.shuffle(perm)
            if all(perm[i] != branches[i] for i in range(len(branches))):
                break
        m = {branches[i]: perm[i] for i in range(len(branches))}
        out = ContentStore(selective=self.selective)
        out.poisoned = set(self.poisoned)
        for key, d in self.store.items():
            out.store[key] = {m.get(b, b): c for b, c in d.items()}
        return out


def build_htm(seed, arm, n_seq, L, n_cells, k_win, act_th, epochs):
    """Trained on-bridge HTM-TM (arm: 'htm'|'lesion'|'untrained'), as in the horizon runner's _run_point."""
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    b, cells_idx, row, col = build_pool_bridge(vocab, n_cells, seed, act_th=act_th, coincidence=(arm != "lesion"))
    lr = OnBridgeLearner(b, row, col, cells_idx, vocab, n_cells, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in seqs:
                lr.train_sequence(s)
    return lr, seqs, vocab


def harvest_store(lr, seqs, selective):
    """SELECTIVE (or always) write of (allocation-SDR key -> branch) over the CONFIDENT shared-middle positions of each
    training sequence's clean forward pass. Middle = positions 1..L (exclude cue t=0 and branch t=L+1)."""
    store = ContentStore(selective=selective)
    for s in seqs:
        branch_col = s[-1]                                # the branch symbol == its column index
        steps = traverse_capture(lr, s)
        for t in range(1, len(s) - 1):                    # 1..L (shared middle allocation cells)
            if steps[t]["confident"]:
                store.write(steps[t]["active"], branch_col)
    return store


def store_predict_branch(lr, store, seq, L, tau, lesion_read=False):
    """Bare-HTM prediction at the branch, RECOVERED by content-addressed completion when the store returns an
    unambiguous hit. lesion_read=True disables the store (-> collapses to the bare HTM: the load-bearing control)."""
    steps = traverse_capture(lr, seq)
    bare = steps[L]["pred_cols"]
    if lesion_read or store is None:
        return bare
    query = [(t, steps[t]["active"]) for t in range(1, L + 1) if steps[t]["confident"]]
    hit = store.read(query, tau)
    return bare if hit is None else {hit}


def _acc(pred_fn, seqs, L):
    return sum(int(pred_fn(s) == {s[-1]}) for s in seqs) / len(seqs)


def swap_follows_store(lr, store, seqs, L, tau):
    """CONTEXT-NECESSITY: inject a DIFFERENT cue (word 0) -> the store completion must FOLLOW the injected cue's branch
    (the allocation SDR reflects the traversed cue), not the memorised one. High -> distal-cue-driven, no confabulation."""
    n = len(seqs); ok = 0; tot = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            swapped = list(seqs[i]); swapped[0] = seqs[j][0]
            pred = store_predict_branch(lr, store, swapped, L, tau)
            ok += int(pred == {seqs[j][-1]})              # must predict cue j's branch (followed the injected context)
            tot += 1
    return ok / max(1, tot)


def _run_point(seed, n_seq, L, n_cells, k_win, act_th, epochs, tau):
    """One (seed, distance) point: bare / store(selective) / store-lesion / always-write / permute arms + swap + floors."""
    lr, seqs, vocab = build_htm(seed, "htm", n_seq, L, n_cells, k_win, act_th, epochs)

    # sanity: my capture reproduces the bare-HTM branch prediction exactly (else the comparison is invalid)
    for s in seqs:
        assert traverse_capture(lr, s)[L]["pred_cols"] == lr.predict_branch(s, L)[L], "capture != predict_branch"

    sel = harvest_store(lr, seqs, selective=True)
    alw = harvest_store(lr, seqs, selective=False)
    branches = [s[-1] for s in seqs]
    perm = sel.permuted(branches, seed)

    bare = _acc(lambda s: store_predict_branch(lr, None, s, L, tau), seqs, L)
    store = _acc(lambda s: store_predict_branch(lr, sel, s, L, tau), seqs, L)
    lesion = _acc(lambda s: store_predict_branch(lr, sel, s, L, tau, lesion_read=True), seqs, L)
    always = _acc(lambda s: store_predict_branch(lr, alw, s, L, tau), seqs, L)
    permute = _acc(lambda s: store_predict_branch(lr, perm, s, L, tau), seqs, L)
    swap = swap_follows_store(lr, sel, seqs, L, tau)

    markov = markov_branch_acc(seqs, L, n_seq)
    oracle = full_oracle(seqs, L)
    return {"seed": seed, "L": L, "distance": L + 1, "n_cells": n_cells, "chance": 1.0 / n_seq,
            "bare": bare, "store": store, "store_lesion": lesion, "always_write": always, "permute": permute,
            "swap_follows": swap, "markov": markov, "oracle": oracle,
            "n_sel_keys": len(sel.store), "n_sel_poisoned": len(sel.poisoned), "n_always_keys": len(alw.store),
            "n_always_ambiguous": sum(1 for d in alw.store.values() if len(d) > 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--distances", type=int, nargs="+", default=[16],
                    help="shared-middle lengths L (dependency distance = L+1); the interference break is at dist>=17 (L>=16)")
    ap.add_argument("--n-seq", type=int, default=3, help="# interfering contexts (chance = 1/n_seq); break needs >=3")
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--slack", type=int, default=8, help="fair capacity n_cells = k_win*n_seq + slack")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--tau-read", type=float, default=0.5, help="min allocation-SDR overlap for a content-addressed hit")
    ap.add_argument("--l-far", type=int, default=16, help="GO evaluated at the largest swept L >= this")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 1:
        print("NOT-RUNNABLE: need >=1 seed"); return 2
    smoke = len(a.seeds) < 6

    backend = os.environ.get("SIM_BACKEND", "numpy")
    n_cells = int(a.k_win * a.n_seq + a.slack)            # FAIR capacity (matches the horizon finding's fair point)
    chance = 1.0 / a.n_seq
    grid = sorted(set(a.distances))
    print(f"backend={backend} | n_seq={a.n_seq} chance={chance:.3f} | n_cells(fair)={n_cells} | epochs={a.epochs} "
          f"| tau_read={a.tau_read} | distances(L)={grid} | seeds={a.seeds}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in grid:
            per = [_run_point(s, a.n_seq, L, n_cells, a.k_win, a.act_th, a.epochs, a.tau_read) for s in a.seeds]
            def m(k):
                return float(np.mean([p[k] for p in per]))
            agg = {"L": L, "distance": L + 1, "n_cells": n_cells, "chance": chance,
                   "bare": m("bare"), "store": m("store"), "store_lesion": m("store_lesion"),
                   "always_write": m("always_write"), "permute": m("permute"), "swap_follows": m("swap_follows"),
                   "markov": m("markov"), "oracle": m("oracle"),
                   "n_sel_keys": m("n_sel_keys"), "n_sel_poisoned": m("n_sel_poisoned"),
                   "n_always_keys": m("n_always_keys"), "n_always_ambiguous": m("n_always_ambiguous"),
                   "per_seed": per}
            points.append(agg)
            print(f"  [L={L:>3} dist={L+1:>3}] bare {agg['bare']:.3f} | STORE {agg['store']:.3f} | store-lesion "
                  f"{agg['store_lesion']:.3f} | always {agg['always_write']:.3f} | permute {agg['permute']:.3f} "
                  f"|| swap {agg['swap_follows']:.3f} | markov {agg['markov']:.3f} chance {chance:.3f} oracle "
                  f"{agg['oracle']:.3f} | sel_keys {agg['n_sel_keys']:.0f}(pois {agg['n_sel_poisoned']:.0f}) "
                  f"alw_ambig {agg['n_always_ambiguous']:.0f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    verdict = None
    if err is None and points:
        far = max((p for p in points if p["L"] >= a.l_far), key=lambda p: p["L"], default=points[-1])
        print("\n-- anti-cheats at the FAR distance (L=%d, dist=%d) --" % (far["L"], far["distance"]), flush=True)
        void_if(far["oracle"] <= 0.99, "task not context-solvable (oracle %.3f)" % far["oracle"])
        lever("store vs store-lesion (load-bearing)", round(far["store_lesion"], 3), round(far["store"], 3),
              required=False)
        lever("selective vs always-write (gate)", round(far["always_write"], 3), round(far["store"], 3),
              required=False)
        attributable_to("store recovery over bare-HTM", far["store"], far["bare"])
        attributable_to("recovery via TRUE keys (vs permuted)", far["store"], far["permute"])

        broke = far["bare"] < 0.90                        # the interference regime the store must rescue
        restores = far["store"] >= 0.90 and far["store"] >= far["bare"] + 0.15
        load_bearing = far["store"] >= far["store_lesion"] + 0.15
        selective = far["store"] >= far["always_write"] + 0.15
        attributed = far["permute"] <= chance + 0.10
        context_driven = far["swap_follows"] >= 0.90
        solvable = far["oracle"] > 0.99

        go = bool(smoke is False and broke and restores and load_bearing and selective and attributed
                  and context_driven and solvable)
        smoke_go = bool(broke and restores and load_bearing and selective and attributed and context_driven and solvable)

        if not solvable:
            verdict = f"INCONCLUSIVE — task not context-solvable (oracle {far['oracle']:.3f})."
        elif not broke:
            verdict = (f"INCONCLUSIVE — bare HTM did NOT break at dist {far['distance']} (bare {far['bare']:.3f} >= 0.90); "
                       f"no chain-integrity residual to rescue at this resource point. Push distance/interference or "
                       f"reduce capacity/epochs to the break regime the horizon finding measured.")
        elif restores and load_bearing and selective and attributed and context_driven:
            tag = "GO" if go else ("SMOKE-GO (1-seed indicator; run the 6-seed sweep)" if smoke else "GO")
            verdict = (f"{tag} — the SELECTIVE-WRITE content-addressable store over the HTM-TM's own allocation keys "
                       f"RESTORES the interference-broken horizon: at dist {far['distance']} (n_seq={a.n_seq}) the bare "
                       f"HTM chain breaks to {far['bare']:.3f} (>= one context's priming chain merges) but the store "
                       f"recovers to {far['store']:.3f} (>= bare+0.15). It is LOAD-BEARING (lesion the store -> "
                       f"{far['store_lesion']:.3f}, back to bare), SELECTIVITY GATES (always-write "
                       f"{far['always_write']:.3f} does NOT recover — the ambiguous merged keys trap the read), "
                       f"ATTRIBUTABLE (permute keys -> {far['permute']:.3f} <= chance {chance:.3f}), and "
                       f"CONTEXT-DRIVEN (swap-follows {far['swap_follows']:.3f}, no confabulation). oracle "
                       f"{far['oracle']:.3f}. Reuse-by-import of EMERGE-14; NO sim/ edit.")
        else:
            miss = []
            if not restores: miss.append(f"store {far['store']:.3f} did not restore (need >=0.90 and >= bare+0.15, bare {far['bare']:.3f})")
            if not load_bearing: miss.append(f"not load-bearing (store {far['store']:.3f} vs lesion {far['store_lesion']:.3f})")
            if not selective: miss.append(f"selectivity inert (store {far['store']:.3f} vs always-write {far['always_write']:.3f}) — the recovery is 'more memory', not selective write")
            if not attributed: miss.append(f"not attributable (permute {far['permute']:.3f} > chance+0.10)")
            if not context_driven: miss.append(f"not context-driven (swap {far['swap_follows']:.3f} < 0.90)")
            verdict = ("HONEST NEGATIVE / BOUNDARY — the store did not cleanly restore the interference-broken horizon: "
                       + "; ".join(miss) + ". This maps the residual and names the next mechanism (the content-addressed "
                       "completion of a merged allocation chain is the frontier the horizon finding handed over).")
    else:
        verdict = f"ERROR — {err}" if err else "ERROR — no points computed"

    # --- earned verdict: VALIDITY preconditions travel with the verdict (tools/gates/verdict_preconditions) ---
    # PRECONDITIONS are validity checks that must hold for ANY verdict (GO or a legitimate boundary) to be meaningful:
    # the task is context-solvable, the interference-break regime the store must rescue actually EXISTS at this point,
    # and the attribution instrument (permute-keys) discriminates. The RESULTS (store restores / selectivity gates /
    # load-bearing) are the DECISION passed to decide(), NOT preconditions — a valid point below threshold is a
    # legitimate NO-GO/boundary, not UNDEFINED.
    preconditions = []
    try:
        from tools.verdict import Verdict
        V = Verdict("emerge_selective_write_store", chance=chance)
        _far = far if (err is None and points) else None
        if _far is not None:
            V.require("oracle>0.99_task_solvable", round(_far["oracle"], 4), expect=lambda x: x > 0.99,
                      note="else the task is not context-solvable -> INCONCLUSIVE, not a negative")
            V.require("bare_broke_under_interference", round(_far["bare"], 4), expect=lambda x: x < 0.90,
                      note="the store's job is to rescue a BROKEN horizon; if bare already holds there is nothing to rescue")
            V.require("permute_attribution_instrument_collapses", round(_far["permute"], 4),
                      expect=lambda x: x <= chance + 0.10,
                      note="permuted keys must fall to <= chance, else the content-addressed recall is not key-driven")
        else:
            V.require("reached_l_far", 1 if (points and any(p["L"] >= a.l_far for p in points)) else 0,
                      expect=lambda x: x >= 1, note="no swept L reached l_far (or run errored)")
        _go = bool(err is None and points and far is not None and (far["oracle"] > 0.99) and (far["bare"] < 0.90)
                   and (far["store"] >= 0.90) and (far["store"] >= far["bare"] + 0.15)
                   and (far["store"] >= far["store_lesion"] + 0.15) and (far["store"] >= far["always_write"] + 0.15)
                   and (far["permute"] <= chance + 0.10) and (far["swap_follows"] >= 0.90))
        dec = V.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_selective_write_store", "verdict": verdict, "backend": backend, "sim_backend": backend,
               "cost_acknowledged": True, "smoke": smoke, "preconditions": preconditions,
               "mechanism": "SELECTIVE-WRITE (mismatch/novelty-gated) content-addressable store keyed on the on-bridge "
                            "HTM Temporal-Memory's allocation SDRs (the priming-chain winner-cell subsets in the shared "
                            "middle); content-addressed completion recovers a broken/merged priming chain by an earlier "
                            "clean allocation key; the merged (ambiguous) keys are POISONED at write, so the read is not "
                            "trapped by them",
               "task": "EMERGE-14 overlap corpus [cue, <L middle>, branch]; branch depends on the cue L+1 tokens back "
                       "(n-gram pinned at chance 1/n_seq); anti-cheats: store-lesion (load-bearing) + always-write "
                       "(selectivity gate) + permute-keys (attribution) + swap-follows-context (no confab) + oracle",
               "seeds": a.seeds, "config": {"distances": grid, "n_seq": a.n_seq, "k_win": a.k_win, "act_th": a.act_th,
               "slack": a.slack, "n_cells": n_cells, "epochs": a.epochs, "tau_read": a.tau_read, "l_far": a.l_far,
               "chance": chance},
               "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the EMERGE-14 on-bridge learner; NO sim/ edit. Memorise-and-recall "
                              "horizon (apples-to-apples with the deltastore reservoir KV horizon). Keys are the HTM's "
                              "OWN allocation SDRs in the SHARED MIDDLE (t>=1), NOT the raw cue input (t=0) — keying on "
                              "t=0 would be a trivial cue->branch input lookup bypassing the HTM. 1-seed is a SMOKE "
                              "indicator; the decisive run is the 6-seed sweep."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge_selective_write_store] VERDICT: {verdict}", flush=True)
    print(f"[emerge_selective_write_store] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
