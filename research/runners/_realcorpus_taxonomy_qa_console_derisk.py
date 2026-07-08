"""Multi-level taxonomic property-QA in a CONVERSATIONAL form (the CYCLE-1043 chained inheritance, wired to Q&A):
teach a PROPERTY at the GRANDPARENT level ("animals breathe"), then ASK "can a <held-out member> breathe?" -> the
member's leaf routes leaf->super->grandparent via the validated CHAINED read (CYCLE 1043) -> the inherited grandparent
property answers YES; a mismatched property ("can a <member> <a-different-grandparent's-property>?") answers NO; an
unknown token ("can a zzz breathe?") hits the no-confab MOAT ("I don't know what a zzz is"). vs a grandparent-
derangement control (the property bound to a WRONG grandparent's supers -> the chain routes to the wrong grandparent
-> the YES-answers collapse). This is the multi-level taxonomy usable in dialogue, on real Wikidata is-a. numpy.
NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from research.runners._emergent_vocab_breadth_scale_derisk import learn_stream_codes
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows

# one grandparent-level property per grandparent (the property is INHERITED 2-up by every descendant leaf).
GP_PROPERTY = {"animal": "breathe", "plant": "grow", "vehicle": "move", "food": "nourish", "tool": "help"}


def _stream(tree, rng):
    sents = []
    for g, smap in tree.items():
        for s, leaves in smap.items():
            sents.append(["the", s, "is", "a", g])
            for lf in leaves:
                sents.append(["the", lf, "is", "a", s])
    stream, order = [], list(range(len(sents)))
    for _ in range(40):
        rng.shuffle(order)
        stream += [list(sents[i]) for i in order]
    return stream


class TaxonomyQA:
    """Answers 'can a <X> <property>?' via chained multi-level inheritance + the no-confab moat."""

    def __init__(self, seed, tree, deranged=False):
        self.rng = np.random.default_rng(seed)
        self.gps = [g for g in tree if len(tree[g]) >= 2]
        self.supers = sorted({s for g in self.gps for s in tree[g]})
        self.leaves = sorted({lf for g in self.gps for s in tree[g] for lf in tree[g][s]})
        stream = _stream(tree, self.rng)
        self.allw = self.leaves + self.supers
        codes, _ = learn_stream_codes(seed, stream, self.allw, self.gps + self.supers, window=4)
        self.U = _unit_rows(codes)
        self.row = {w: i for i, w in enumerate(self.allw)}
        self.tree = tree
        # hold out one whole super per grandparent -> its leaves are the never-taught QA queries
        self.held = {}
        for g in self.gps:
            sl = list(tree[g]); self.rng.shuffle(sl); self.held[g] = sl[-1]
        # L2 grandparent prototypes from the TAUGHT supers; DERANGEMENT binds a grandparent's proto to a WRONG
        # grandparent's supers -> the chain routes wrong -> collapse.
        gp_supers = {g: [s for s in tree[g] if s != self.held[g]] for g in self.gps}
        if deranged:
            perm = list(self.gps)
            while any(perm[i] == self.gps[i] for i in range(len(perm))):
                self.rng.shuffle(perm)
            gp_supers = {self.gps[i]: gp_supers[perm[i]] for i in range(len(self.gps))}
        self.gp_protos = {g: self._proto([s for s in gp_supers[g] if s in self.row]) for g in self.gps}
        self.gp_of_property = {GP_PROPERTY[g]: g for g in self.gps if g in GP_PROPERTY}

    def _proto(self, keys):
        return self.U[[self.row[k] for k in keys]].mean(0) if keys else np.zeros(self.U.shape[1])

    def _super_proto(self, s, exclude):
        ls = [lf for g in self.gps for ss in self.tree[g] if ss == s
              for lf in self.tree[g][ss] if lf != exclude and lf in self.row]
        return self._proto(ls)

    def _chain_to_grandparent(self, leaf):
        """leaf -> nearest super (L1) -> nearest grandparent (L2). Returns the inferred grandparent."""
        s_scores = {s: self.U[self.row[leaf]] @ self._super_proto(s, leaf) for s in self.supers if s in self.row}
        s_hat = max(s_scores, key=s_scores.get)
        g_scores = {gg: self.U[self.row[s_hat]] @ self.gp_protos[gg] for gg in self.gps}
        return max(g_scores, key=g_scores.get)

    def ask(self, leaf, prop):
        """'can a <leaf> <prop>?' -> yes / no / moat-abstain."""
        if leaf not in self.row:
            return "moat"                                        # no-confab: never-heard token -> abstain
        if prop not in self.gp_of_property:
            return "moat"                                        # unknown property
        g_hat = self._chain_to_grandparent(leaf)
        return "yes" if self.gp_of_property[prop] == g_hat else "no"


def run_seed(seed, tree, deranged=False):
    qa = TaxonomyQA(seed, tree, deranged=deranged)
    gps = qa.gps
    props = [GP_PROPERTY[g] for g in gps if g in GP_PROPERTY]
    yes_ok = yes_tot = no_ok = no_tot = 0
    for g in gps:
        if g not in GP_PROPERTY:
            continue
        true_prop = GP_PROPERTY[g]
        for lf in tree[g][qa.held[g]]:                           # held-out members of g (never taught)
            if lf not in qa.row:
                continue
            # positive: can this member do its grandparent's property? -> should be YES
            yes_tot += 1; yes_ok += int(qa.ask(lf, true_prop) == "yes")
            # negative: can it do a DIFFERENT grandparent's property? -> should be NO
            for op in props:
                if op == true_prop:
                    continue
                no_tot += 1; no_ok += int(qa.ask(lf, op) == "no")
    # moat: unknown tokens abstain
    moat_ok = sum(qa.ask(z, props[0]) == "moat" for z in ["zzz", "qqq", "xyzzy"]) / 3.0
    return {"yes": yes_ok / max(1, yes_tot), "no": no_ok / max(1, no_tot), "moat": moat_ok,
            "n_yes": yes_tot, "n_no": no_tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="research/findings/raw/_wikidata_3level.json")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    tree = json.load(open(a.tree))
    seeds = [int(x) for x in a.seeds.split(",")]
    gps = [g for g in tree if len(tree[g]) >= 2]
    print(f"[taxonomy QA] multi-level property inheritance in Q&A form | grandparents={gps} props={[GP_PROPERTY.get(g) for g in gps]}", flush=True)
    yes, no, moat, dyes = [], [], [], []
    for s in seeds:
        r = run_seed(s, tree, deranged=False); d = run_seed(s, tree, deranged=True)
        yes.append(r["yes"]); no.append(r["no"]); moat.append(r["moat"]); dyes.append(d["yes"])
        print(f"  [seed {s}] YES(inherit)={r['yes']:.3f} NO(mismatch)={r['no']:.3f} moat={r['moat']:.3f} | deranged-YES={d['yes']:.3f}  ({r['n_yes']} pos, {r['n_no']} neg)", flush=True)
    go = (all(y > 0.85 for y in yes) and all(n > 0.85 for n in no) and all(m > 0.99 for m in moat)
          and all(yes[i] - dyes[i] > 0.30 for i in range(len(yes))))
    print(f"\n  AGGREGATE: YES={np.mean(yes):.3f} NO={np.mean(no):.3f} moat={np.mean(moat):.3f} deranged-YES={np.mean(dyes):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- 'can a <held-out member> <property>?' answered by CHAINED "
          f"multi-level inheritance (member->super->grandparent->inherited property) {'+ mismatched property NO + no-confab moat + grandparent-derangement collapses the YES-answers -> the multi-level taxonomy is CONVERSATIONALLY usable on real Wikidata is-a' if go else '-- some gate unmet; honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
