"""KNOWLEDGE-half of breadth, CANCELLATION: a member's OWN property overrides its category's inherited one.

The rung-4 reasoner inherits UNIFORMLY -- every member of a discovered category gets the category's
taught property. Real semantic cognition has EXCEPTIONS: a penguin IS a bird (inherits "fly") but its
OWN property ("walks") OVERRIDES the inherited one -> "can a penguin fly? No, the penguin walks."
(EMERGE-54's per-member cancellation, now over REAL-corpus-discovered categories.)

The biological mechanism (EMERGE-54 / graded apical drive): the exception member carries a member-specific
property binding that is STRONGER than the weak, generalized (inherited) category drive, so its own
property WINS the argmax. Concretely, on the rung-4 associative memory M:
  * teach the class to SOME members  -> bind them to P[cat]        (the category property)
  * the exception member is HELD-OUT (would INHERIT P[cat] by generalization)
  * teach the exception              -> bind it to P_exc, weight w_exc > 1 (its own, stronger property)
  * predict(member) = argmax over {class properties} U {exception properties}
      - held-out class member   -> P[cat]   ("yes" it has the class property; inheritance intact)
      - exception member         -> P_exc   ("no" it has the class property; its own property overrides)

Cheap-first, single mechanism (w_exc), 6-seed(-blind), anti-cheated. Rate-level (reuse rung-4). NO sim/ edit.

Gates (all 6 seeds):
  * INHERIT: held-out class members still -> yes (cancellation does NOT break inheritance).
  * CANCEL:  the exception member -> NO for the class property (its own property wins).
  * OWN:     the exception member -> YES for its OWN property (predict picks P_exc).
  * CONTROL (load-bearing): WITHOUT the exception teaching, the SAME member -> yes (inherits) ->
    proves the exception binding is what flips it (not a code artifact).
  * PERMUTED anti-cheat: bind a RANDOM (non-member) word as the "exception" -> the real member still
    inherits (yes) -> the cancellation is SPECIFIC to the taught exception, not a generic M perturbation.
  * MOAT: unknown word -> idk (by construction).
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._realcorpus_inheritance_rung4_conversation_derisk import (
    RealCorpusConsole, _splits, _coherence,
)
from research.runners.corpus_stream import load_token_stream_multi


class CancellingConsole(RealCorpusConsole):
    """Adds member-specific EXCEPTION properties that override inherited category properties."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.exc_ids = []                 # exception-property ids (each a distinct random tag)
        self.P_exc = {}                   # exc_id -> property vector

    def teach_exception(self, member, exc_id, w_exc):
        """Bind `member`'s code to its OWN property tag with weight w_exc (the stronger, member-specific drive)."""
        if member not in self.row_of:
            return
        if exc_id not in self.P_exc:
            self.P_exc[exc_id] = self.rng.randn(self.D)
            self.exc_ids.append(exc_id)
        self.M = self.M + w_exc * np.outer(self.U[self.row_of[member]], self.P_exc[exc_id])

    def teach_exception_adaptive(self, member, exc_id, margin=2.0):
        """Graded apical drive REGULATED to override: set the exception weight just large enough that the
        member's OWN property beats its top inherited class property by `margin` -- large for a strongly-
        inherited member, small for a weakly-inherited one, so cross-talk is minimized (not a fixed gain)."""
        if member not in self.row_of:
            return 0.0
        if exc_id not in self.P_exc:
            self.P_exc[exc_id] = self.rng.randn(self.D)
            self.exc_ids.append(exc_id)
        h = self.U[self.row_of[member]] @ self.M                       # inherited drive (post class-teach)
        smax = max(self.P[c] @ h for c in self.cat_ids)               # top inherited class score
        pe = self.P_exc[exc_id]
        s_exc_base = pe @ h
        c_star = max(self.cat_ids, key=lambda c: self.P[c] @ h)
        denom = float(pe @ pe - pe @ self.P[c_star])                  # gain per unit weight (|P_exc|^2 dominates)
        w = max(0.0, (smax - s_exc_base + margin) / denom) if denom > 1e-6 else margin
        self.M = self.M + w * np.outer(self.U[self.row_of[member]], pe)
        return float(w)

    def _predict_all(self, word):
        """argmax over BOTH the class property tags and the exception property tags."""
        phat = self.U[self.row_of[word]] @ self.M
        scores = {("cat", c): float(self.P[c] @ phat) for c in self.cat_ids}
        for e in self.exc_ids:
            scores[("exc", e)] = float(self.P_exc[e] @ phat)
        return max(scores, key=scores.get)

    def ask_class(self, category, word):
        """'does <word> have <category>'s property?' with cancellation: an exception member -> no."""
        if word not in self.row_of:
            return "idk"
        pred = self._predict_all(word)
        return "yes" if pred == ("cat", category) else "no"

    def ask_own(self, exc_id, word):
        """'does <word> have its OWN (exception) property?' -> yes iff predict picks that exc tag."""
        if word not in self.row_of:
            return "idk"
        return "yes" if self._predict_all(word) == ("exc", exc_id) else "no"


# a small known-animal list (subset of the breadth A->W vocab) to PREFER a semantically coherent
# animal cluster over a character-name cluster (lily/tim), so the demo reads "the penguin walks".
_ANIMALS = {"dog", "cat", "bird", "fish", "frog", "bear", "mouse", "duck", "cow", "pig",
            "sheep", "hen", "owl", "wolf", "fox", "lion", "goat", "bee", "ant", "bug"}


def _pick_pos(con, coh):
    """PREFER a discovered cluster containing >=2 known animals (coherent + speakable); else most coherent."""
    animal_cats = [(c, sum(1 for w in con.members[c] if w in _ANIMALS)) for c in con.cat_ids]
    animal_cats = [(c, n) for c, n in animal_cats if n >= 2]
    if animal_cats:
        return max(animal_cats, key=lambda cn: cn[1])[0]     # the cluster with the MOST animals
    return max(con.cat_ids, key=lambda c: coh[c])            # fallback: most coherent


def _apply_exc(con, member, w_exc, margin, adaptive):
    return con.teach_exception_adaptive(member, "own", margin) if adaptive else con.teach_exception(member, "own", w_exc)


def run_seed(seed, stories, K, w_exc, emergent=True, n_clusters=10, margin=2.0, adaptive=False):
    con = CancellingConsole(seed, stories, K, emergent=emergent, n_clusters=n_clusters)
    if len(con.cat_ids) < 2:
        return None
    coh = {c: _coherence(con, c) for c in con.cat_ids}
    pos = _pick_pos(con, coh)
    taught_by_cat, held_by_cat = _splits(con.members, con.cat_ids, con.rng)
    held = held_by_cat[pos]
    con.teach(taught_by_cat)                                  # class properties (uniform inheritance)

    # the EXCEPTION member = a HELD-OUT member of pos that INHERITS before the exception (so the flip is meaningful).
    exc_member = next((w for w in held if con.ask_class(pos, w) == "yes"), None)
    if exc_member is None:
        return None                                          # no inheriting held-out member to make an exception of
    other_held = [w for w in held if w != exc_member]
    before = {w: con.ask_class(pos, w) for w in other_held}  # other members' answers BEFORE the exception

    # teach the exception: bind exc_member to its OWN property (fixed gain, or adaptive graded drive).
    w_used = _apply_exc(con, exc_member, w_exc, margin, adaptive)
    cancel = con.ask_class(pos, exc_member)                   # expect "no" (own property overrides the inherited)
    own = con.ask_own("own", exc_member)                      # expect "yes" (its own property)
    after = {w: con.ask_class(pos, w) for w in other_held}    # other members AFTER the exception
    n_collateral = sum(1 for w in other_held if before[w] != after[w])  # exception must NOT flip others

    # PERMUTED anti-cheat: on a FRESH console, teach the exception to a RANDOM non-pos word -> exc_member
    # should STILL inherit (yes). Proves the cancellation is specific to the taught member, not generic.
    con_p = CancellingConsole(seed, stories, K, emergent=emergent, n_clusters=n_clusters)
    con_p.teach(taught_by_cat)
    non_pos = [w for c in con_p.cat_ids if c != pos for w in con_p.members[c] if w in con_p.row_of]
    rand_word = con_p.rng.choice(non_pos) if non_pos else exc_member
    _apply_exc(con_p, rand_word, w_exc, margin, adaptive)
    permuted_still_inherits = con_p.ask_class(pos, exc_member)  # expect "yes"

    moat = con.ask_class(pos, "zzzqqx") == "idk"
    return {
        "seed": seed, "pos": pos, "coh": round(coh[pos], 3), "exc_member": exc_member, "w_used": round(w_used, 2),
        "cancel": cancel, "own": own, "n_other_held": len(other_held), "n_collateral": int(n_collateral),
        "permuted_still_inherits": permuted_still_inherits, "moat_ok": bool(moat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--w-exc", type=float, default=3.0)
    ap.add_argument("--adaptive", action="store_true", help="regulated graded drive (weight scaled to override by --margin)")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[cancellation] corpus={a.corpus_path} K={a.K} "
          f"{'ADAPTIVE(margin '+str(a.margin)+')' if a.adaptive else 'w_exc='+str(a.w_exc)} EMERGENT-clusters", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.w_exc, emergent=True, n_clusters=a.n_clusters, margin=a.margin, adaptive=a.adaptive)
        if r is None:
            print(f"  [seed {s}] not evaluable (need >=2 held-out in the coherent cat)", flush=True); continue
        recs.append(r)
        print(f"  [seed {s}] pos={r['pos']}(coh {r['coh']}) exc='{r['exc_member']}' w={r['w_used']} | "
              f"CANCEL={r['cancel']} OWN={r['own']} | collateral={r['n_collateral']}/{r['n_other_held']} | "
              f"permuted_inherits={r['permuted_still_inherits']} | moat={int(r['moat_ok'])}", flush=True)

    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    # gates (the exc_member inherited BEFORE by construction -> CANCEL=no is a genuine yes->no flip)
    cancel_ok = all(r["cancel"] == "no" for r in recs)                     # exception overrides the inherited property
    own_ok = all(r["own"] == "yes" for r in recs)                          # its own property wins
    no_collateral = all(r["n_collateral"] == 0 for r in recs)             # exception does NOT flip other members
    permuted_ok = all(r["permuted_still_inherits"] == "yes" for r in recs)
    moat_ok = all(r["moat_ok"] for r in recs)
    go = cancel_ok and own_ok and no_collateral and permuted_ok and moat_ok
    print(f"\n  AGGREGATE ({len(recs)} seeds): CANCEL(inherited->no) all={cancel_ok} | OWN=yes all={own_ok} | "
          f"no-collateral all={no_collateral} | permuted-inherits all={permuted_ok} | moat all={moat_ok}", flush=True)
    _verdict_msg = ('OVERRIDES its category inheritance (cancellation) while other members still inherit, specific to '
                    'the taught exception (permuted control inherits), moat intact'
                    if go else 'does NOT cleanly cancel')
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- a member's OWN property {_verdict_msg}.", flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "w_exc": a.w_exc, "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
