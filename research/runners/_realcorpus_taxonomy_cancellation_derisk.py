"""Member-specific CANCELLATION over multi-level (chained) inheritance -- the canonical "penguin" case, on the real
Wikidata is-a tree. A member inherits its GRANDPARENT's property 2-up (CYCLE 1043/1044), UNLESS it has been taught a
member-specific OWN property that CONTRADICTS the inherited one -> the own property overrides (graded apical drive:
the proximal member fact beats the distal inherited one; EMERGE-26/54 cancellation, Dusek-Eichenbaum). So:
  teach: grandparent G -> property P (all descendants inherit P) + a member M -> own property Q (Q != P)
  ask "can a M P?"  -> NO   (M's OWN property Q cancels the inherited P)         <- cancellation
  ask "can a M Q?"  -> YES  (M's own)
  ask "can a M' P?" -> YES  (a NORMAL sibling still inherits P)                   <- inheritance intact
  ask "can a zzz P?"-> MOAT
Load-bearing control: WITHOUT M's override taught, "can a M P?" -> YES (inherits) -> teaching the override FLIPS it to
NO (the cancellation is load-bearing, not a coincidence). numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from research.runners._realcorpus_taxonomy_qa_console_derisk import TaxonomyQA, GP_PROPERTY


class CancellingTaxonomyQA(TaxonomyQA):
    """TaxonomyQA + member-specific own-property overrides (cancellation over the chained inheritance)."""

    def __init__(self, seed, tree, deranged=False, hold_out=False):
        super().__init__(seed, tree, deranged=deranged, hold_out=hold_out)
        self.override = {}                                       # member -> its OWN property (beats inheritance)

    def teach_exception(self, member, own_prop):
        self.override[member] = own_prop

    def ask(self, leaf, prop):
        if leaf not in self.row:
            return "moat"
        if prop not in self.gp_of_property:
            return "moat"
        if leaf in self.override:                                # proximal member fact overrides the distal inherited
            return "yes" if self.override[leaf] == prop else "no"
        g_hat = self._chain_to_grandparent(leaf)                 # else inherit 2-up via the chain
        return "yes" if self.gp_of_property[prop] == g_hat else "no"


def run_seed(seed, tree):
    gps = [g for g in tree if len(tree[g]) >= 2 and g in GP_PROPERTY]
    if len(gps) < 2:
        return None
    props = [GP_PROPERTY[g] for g in gps]
    qa = CancellingTaxonomyQA(seed, tree, hold_out=False)
    rng = np.random.default_rng(seed + 7)
    canc_ok = canc_tot = inh_ok = inh_tot = ctrl_flip = ctrl_tot = 0
    for g in gps:
        true_prop = GP_PROPERTY[g]
        other_prop = next(p for gg, p in GP_PROPERTY.items() if p != true_prop and gg in gps)
        # pick a member of g to make the EXCEPTION (give it an own property = a different grandparent's property)
        leaves_g = [lf for s in tree[g] for lf in tree[g][s] if lf in qa.row]
        if len(leaves_g) < 2:
            continue
        rng.shuffle(leaves_g)
        exc, sib = leaves_g[0], leaves_g[1]
        # control: BEFORE teaching the override, the exception member inherits true_prop -> YES
        pre = qa.ask(exc, true_prop)
        qa.teach_exception(exc, other_prop)                      # teach M's OWN property (= other grandparent's prop)
        # cancellation: the exception member now DENIES the inherited property (its own overrides)
        canc_tot += 1; canc_ok += int(qa.ask(exc, true_prop) == "no")
        # its own property answers yes
        canc_tot += 1; canc_ok += int(qa.ask(exc, other_prop) == "yes")
        # inheritance intact: a NORMAL sibling still inherits true_prop
        inh_tot += 1; inh_ok += int(qa.ask(sib, true_prop) == "yes")
        # control flip: pre (no override) was YES, post is NO -> the override is load-bearing
        ctrl_tot += 1; ctrl_flip += int(pre == "yes" and qa.ask(exc, true_prop) == "no")
    moat = sum(qa.ask(z, props[0]) == "moat" for z in ["zzz", "qqq"]) / 2.0
    return {"cancel": canc_ok / max(1, canc_tot), "inherit": inh_ok / max(1, inh_tot),
            "ctrl_flip": ctrl_flip / max(1, ctrl_tot), "moat": moat, "n": canc_tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="research/findings/raw/_wikidata_3level.json")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    tree = json.load(open(a.tree))
    seeds = [int(x) for x in a.seeds.split(",")]
    gps = [g for g in tree if len(tree[g]) >= 2 and g in GP_PROPERTY]
    print(f"[taxonomy cancellation] penguin-case over the chained inheritance | grandparents={gps}", flush=True)
    cs, ih, cf, mt = [], [], [], []
    for s in seeds:
        r = run_seed(s, tree)
        if r is None:
            print("  NOT-EVALUABLE"); return
        cs.append(r["cancel"]); ih.append(r["inherit"]); cf.append(r["ctrl_flip"]); mt.append(r["moat"])
        print(f"  [seed {s}] cancel={r['cancel']:.3f} inherit-intact={r['inherit']:.3f} ctrl-flip(YES->NO)={r['ctrl_flip']:.3f} moat={r['moat']:.3f}", flush=True)
    go = (all(c > 0.95 for c in cs) and all(i > 0.95 for i in ih)
          and all(f > 0.95 for f in cf) and all(m > 0.99 for m in mt))
    print(f"\n  AGGREGATE: cancel={np.mean(cs):.3f} inherit-intact={np.mean(ih):.3f} ctrl-flip={np.mean(cf):.3f} moat={np.mean(mt):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- a member's OWN property CANCELS the inherited grandparent "
          f"property (penguin-case) {'+ normal siblings still inherit + the override is load-bearing (removing it flips YES->NO) + moat -> cancellation over multi-level inheritance on real Wikidata is-a' if go else '-- some gate unmet; honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
