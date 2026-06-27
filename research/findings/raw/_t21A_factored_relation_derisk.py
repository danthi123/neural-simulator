"""Tier 2.1-A -- FACTORED-RELATION analogy DE-RISK (cheap-first, numpy).

Per research/findings/2026-06-27-analogy-representation-research-gate.md option (a): when the relation lives on
an EXPLICIT FACTORED axis, A:B::C:D is the transform OVER THAT axis (the ADDITIVE-codes 1.000 case), NOT analogy
over raw learned concept codes (the regime-B 0.000 wall).

THE REPRESENTATION (faithful + anti-cheatable + non-trivial):
  Every item (entity OR value) is a point in a shared FACTORED attribute space. Each relation family has a
  RELATION axis and a CONTENT axis:
    * GENDER:  item = bind(GENDER_axis, gender_val) (+) bind(SEM_axis, sem_val)
               king = (GENDER=male, SEM=royal_high), queen=(GENDER=female, SEM=royal_high),
               prince=(GENDER=male, SEM=royal_low), princess=(GENDER=female, SEM=royal_low) ...
               The relation 'gender-flip' = the OFFSET along GENDER_axis. king:queen :: prince:? -> princess.
    * TAXON:   item = bind(CAT_axis, category) (+) bind(INST_axis, instance)
               dog=(CAT=mammal, INST=i_dog), mammal_token=(CAT=mammal, INST=GENERIC) ...
               The relation 'is_a' = the OFFSET that strips INST + sets CAT-readout -> the category token.
    * CAPITAL: item = bind(REGION_axis, region) (+) bind(ROLE_axis, city|country)
               paris=(REGION=fr, ROLE=city), france=(REGION=fr, ROLE=country) ...
               The relation 'capital_of' = the OFFSET along ROLE_axis (city->country), region shared.

  The factored axes are realized through the composer's OWN role-bind. We test TWO compositions to settle the
  §1.3 question (does an EXPLICIT factored slot dodge the BUNDLED crosstalk wall?):
    - "add"    : direct additive sum of the per-axis VALUE phases  (the ADDITIVE 1.000 structure)
    - "bind"   : bundle of bind(axis_role, value) single pairs     (the composer's role-bind form)

  Analogy A:B::C:?  EXTRACTS the transform from the EXAMPLE pair (A,B) WITHOUT naming the relation:
      T = phasor(B) (x) conj(phasor(A));  rec = T (x) phasor(C);  D = cleanup(rec) over ALL item codes,
      operands {A,B,C} excluded. Genuine: T is learned from one pair and must transfer to a HELD-OUT (C,D).

Anti-cheats (ALL mandatory): (i) held-out >> floor; (ii) permuted-relation collapses + TRUE ranks 1/k;
(iii) lesion (skip transform) collapses; (iv) scrambled-source -> chance; (v) no-confab moat; (vi) 6 seeds.
"""
import os
import sys

import numpy as np

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# Each relation family is a list of analogy QUADRUPLES sharing ONE relation, plus the factored attributes of every
# item. attribute layout: {axis_name: value}. The analogy is the offset on the family's RELATION axis.
# ------------------------------------------------------------------------------------------------------------------
def gender_family():
    # SEM groups: items sharing a SEM value form an analogy pair across GENDER.
    pairs = [("king", "queen", "royal_hi"), ("prince", "princess", "royal_lo"), ("man", "woman", "person"),
             ("actor", "actress", "perform"), ("uncle", "aunt", "kin"), ("boy", "girl", "young"),
             ("lord", "lady", "noble"), ("waiter", "waitress", "serve")]
    attrs = {}
    for m, f, sem in pairs:
        attrs[m] = {"GENDER": "male", "SEM": sem}
        attrs[f] = {"GENDER": "female", "SEM": sem}
    return attrs, "GENDER", [(m, f) for m, f, _ in pairs]


def tense_family():
    # PAST-TENSE: a clean BIJECTIVE morphological relation (the classic analogy family). present:past on a TENSE
    # axis, lemma shared. walk:walked :: play:played -> the TENSE offset.
    rows = [("walk", "walked"), ("play", "played"), ("jump", "jumped"), ("talk", "talked"),
            ("open", "opened"), ("close", "closed"), ("start", "started"), ("call", "called")]
    attrs = {}
    for pres, past in rows:
        attrs[pres] = {"TENSE": "present", "LEMMA": pres}
        attrs[past] = {"TENSE": "past", "LEMMA": pres}
    return attrs, "TENSE", [(p, q) for p, q in rows]


def capital_family():
    rows = [("paris", "france", "fr"), ("rome", "italy", "it"), ("berlin", "germany", "de"),
            ("madrid", "spain", "es"), ("lisbon", "portugal", "pt"), ("vienna", "austria", "at"),
            ("athens", "greece", "gr"), ("oslo", "norway", "no")]
    attrs = {}
    for city, country, reg in rows:
        attrs[city] = {"REGION": reg, "ROLE": "city"}
        attrs[country] = {"REGION": reg, "ROLE": "country"}
    return attrs, "ROLE", [(c, k) for c, k, _ in rows]


FAMILIES = {"GENDER": gender_family, "TENSE": tense_family, "CAPITAL": capital_family}


def build_codes(seed, D, attrs, mode="add", scramble=False):
    """Build a factored phasor code for every item. mode='add': sum the per-axis VALUE phases; mode='bind': bundle
    bind(axis_role, value) single pairs through the composer. scramble=True: each item gets a UNIQUE random code
    (no shared factored structure) -- the scrambled-source control."""
    rng = np.random.default_rng(seed + 7777)
    # discover axes + axis values
    axes = sorted({ax for a in attrs.values() for ax in a})
    axis_role = {ax: rng.uniform(0.0, 1.0, D) for ax in axes}
    val_phase = {}
    for ax in axes:
        for a in attrs.values():
            v = a.get(ax)
            if v is not None and (ax, v) not in val_phase:
                val_phase[(ax, v)] = rng.uniform(0.0, 1.0, D)
    comp = RFPhasorComposer(seed=seed, D=D, vocab=["x"])
    codes = {}
    for item, a in attrs.items():
        if scramble:
            codes[item] = rng.uniform(0.0, 1.0, D)
            continue
        comps = []
        for ax in axes:
            if ax in a:
                vp = val_phase[(ax, a[ax])]
                if mode == "add":
                    comps.append(vp)
                else:  # bind
                    comps.append(comp._bind(axis_role[ax], vp))
        if mode == "add":
            codes[item] = np.sum(comps, axis=0) % 1.0
        else:
            codes[item] = comp._bundle(comps) if len(comps) > 1 else comps[0]
    return codes, comp


def _spiking_subtract(comp, x_phases, y_phases):
    """T = x (x) conj(y) via the REAL RF conj synapse (a phase SUBTRACT through resonate-and-fire). y acts as the
    'role' to unbind. Runs on the spiking substrate (the composer's _resonate)."""
    D = comp.D
    zx = comp._to_phasor(x_phases)
    zy_conj = np.conj(comp._to_phasor(y_phases))
    conns = [(D + k, k, zy_conj[k]) for k in range(D)]
    kick = np.zeros(2 * D, dtype=np.complex128); kick[:D] = zx
    return comp._resonate(2 * D, conns, kick)[D:]


def analogy(comp, codes, a, b, c, candidates, lesion=False, fake_b=None, spiking=False):
    """a:b::c:?  T = phasor(b) o conj(phasor(a));  rec = T o phasor(c). lesion: T:=phasor(b). fake_b: substitute
    b's code (permuted-relation control). cleanup over `candidates`. spiking=True: run the transform-extract
    (B (x) conj A) AND the apply (T (x) C) through the REAL RF spiking bind/unbind (comp._bind / a conj synapse)."""
    bcode = fake_b if fake_b is not None else codes[b]
    if spiking:
        if lesion:
            t_phases = np.asarray(bcode) % 1.0
        else:
            t_phases = _spiking_subtract(comp, bcode, codes[a])     # T = B (x) conj(A) through RF conj synapse
        rec = comp._bind(t_phases, codes[c])                        # apply: T (x) C through the RF resonate-and-fire bind
    else:
        za, zc = comp._to_phasor(codes[a]), comp._to_phasor(codes[c])
        zb = comp._to_phasor(bcode)
        zt = zb if lesion else (zb * np.conj(za))
        zrec = zt * zc
        rec = (np.angle(zrec) / (2.0 * np.pi)) % 1.0
    sims = [float(np.mean(np.cos(2.0 * np.pi * (rec - codes[w])))) for w in candidates]
    j = int(np.argmax(sims))
    return candidates[j], float(sims[j]), rec


def nearest(codes, q, candidates):
    sims = [float(np.mean(np.cos(2.0 * np.pi * (q - codes[w])))) for w in candidates]
    j = int(np.argmax(sims))
    return candidates[j], float(sims[j])


def family_tests(attrs, pairs):
    """For a relation family with pairs [(src, tgt), ...] sharing ONE relation: each test uses ONE pair (a,b) to
    build the transform and a DIFFERENT held-out pair (c,d) to score. d is the answer; candidates = all TARGET
    items of the family (so the transform must pick d among same-relation distractors). (a,b)!=(c,d)."""
    targets = sorted({t for _s, t in pairs})
    tests = []
    for (c, d) in pairs:
        # source pair (a,b): a different family pair (a != c) -- prefer one whose target differs from d.
        ab = next(((s, t) for (s, t) in pairs if s != c and t != d), None)
        if ab is None:
            ab = next((s, t) for (s, t) in pairs if (s, t) != (c, d))
        a, b = ab
        cands = [d] + [t for t in targets if t != d]
        tests.append((a, b, c, d, cands))
    return tests


def eval_family(codes, comp, attrs, pairs, spiking=False):
    tests = family_tests(attrs, pairs)
    correct = floor = 0
    sims = []
    for (a, b, c, d, cands) in tests:
        pred, sim, _ = analogy(comp, codes, a, b, c, cands, spiking=spiking)
        correct += (pred == d); sims.append(sim)
        fl, _ = nearest(codes, codes[c], cands)         # floor: nearest target to C (no transform)
        floor += (fl == d)
    return correct, floor, len(tests), sims


def run_seed(seed, D=256, mode="add", spiking=False, verbose=False):
    out = {"seed": seed, "D": D, "mode": mode, "per_family": {}}
    tot_c = tot_f = tot_n = 0
    all_sims = []
    perm_pool = []     # (codes, comp, attrs, pairs) per family for the controls
    for fam, fn in FAMILIES.items():
        attrs, rel_axis, pairs = fn()
        codes, comp = build_codes(seed, D, attrs, mode=mode)
        c, f, n, sims = eval_family(codes, comp, attrs, pairs, spiking=spiking)
        out["per_family"][fam] = {"acc": c / n, "floor": f / n, "n": n}
        tot_c += c; tot_f += f; tot_n += n; all_sims += sims
        perm_pool.append((fam, fn, codes, comp, attrs, pairs))
        if verbose:
            print(f"   {fam:8s}: acc={c/n:.3f} ({c}/{n})  floor={f/n:.3f}")
    acc = tot_c / tot_n; floor_acc = tot_f / tot_n

    # PERMUTED-relation: substitute a random fake_b -> the transform mis-extracts. k permutations; TRUE rank 1/k.
    rng = np.random.default_rng(seed + 999); k_perm = 5
    perm_accs = []
    for _ in range(k_perm):
        pc = pn = 0
        for (_fam, _fn, codes, comp, attrs, pairs) in perm_pool:
            for (a, b, c, d, cands) in family_tests(attrs, pairs):
                pred, _, _ = analogy(comp, codes, a, b, c, cands, fake_b=rng.uniform(0.0, 1.0, D),
                                     spiking=spiking)
                pc += (pred == d); pn += 1
        perm_accs.append(pc / pn)
    perm_acc = float(np.mean(perm_accs)); true_rank = 1 + sum(1 for pa in perm_accs if pa > acc)

    # LESION: skip transform.
    lc = ln = 0
    for (_fam, _fn, codes, comp, attrs, pairs) in perm_pool:
        for (a, b, c, d, cands) in family_tests(attrs, pairs):
            pred, _, _ = analogy(comp, codes, a, b, c, cands, lesion=True, spiking=spiking)
            lc += (pred == d); ln += 1
    lesion_acc = lc / ln

    # SCRAMBLED-source: per-family scrambled codes (no shared factored structure) -> chance.
    sc = sn = 0
    for fam, fn in FAMILIES.items():
        attrs, _ra, pairs = fn()
        codes_s, comp_s = build_codes(seed, D, attrs, mode=mode, scramble=True)
        for (a, b, c, d, cands) in family_tests(attrs, pairs):
            pred, _, _ = analogy(comp_s, codes_s, a, b, c, cands, spiking=spiking)
            sc += (pred == d); sn += 1
    scram_acc = sc / sn

    # MOAT: confidence separation (correct-analogy sims vs random-query sims). Abstention is the composer's
    # store-scan (tested at the agent level); here we record the sim separation that a confidence gate uses.
    rng_q = np.random.default_rng(seed + 31337)
    fam0_codes = perm_pool[0][2]; fam0_words = sorted(fam0_codes)
    rand_sims = [nearest(fam0_codes, rng_q.uniform(0.0, 1.0, D), fam0_words)[1] for _ in range(30)]
    rand_top = float(np.mean(rand_sims))

    out.update({"acc": acc, "floor_acc": floor_acc, "perm_acc": perm_acc, "true_rank": true_rank,
                "k_perm": k_perm, "lesion_acc": lesion_acc, "scram_acc": scram_acc,
                "mean_sim": float(np.mean(all_sims)), "min_sim": float(np.min(all_sims)),
                "rand_sim": rand_top})
    return out


def main(seeds=(42, 43, 44, 45, 46, 47), D=256, mode="add", spiking=False):
    print("=" * 100)
    tag = "  [analogy op THROUGH RF SPIKING bind/unbind]" if spiking else ""
    print(f"TIER 2.1-A FACTORED-RELATION analogy DE-RISK  mode={mode}  D={D}{tag}")
    print("=" * 100)
    results = []
    for s in seeds:
        r = run_seed(s, D=D, mode=mode, spiking=spiking, verbose=(s == seeds[0]))
        results.append(r)
        print(f"seed {s}: acc={r['acc']:.3f} floor={r['floor_acc']:.3f} perm={r['perm_acc']:.3f}"
              f"(rank {r['true_rank']}/{1+r['k_perm']}) lesion={r['lesion_acc']:.3f} scram={r['scram_acc']:.3f}"
              f" sim={r['mean_sim']:.3f} rand={r['rand_sim']:.3f}")
    print("\n" + "=" * 100)
    macc = np.mean([r["acc"] for r in results]); mfloor = np.mean([r["floor_acc"] for r in results])
    mperm = np.mean([r["perm_acc"] for r in results]); mlesion = np.mean([r["lesion_acc"] for r in results])
    mscram = np.mean([r["scram_acc"] for r in results])
    all_rank1 = all(r["true_rank"] == 1 for r in results)
    print(f"MEAN/{len(seeds)} seeds: acc={macc:.3f} floor={mfloor:.3f} perm={mperm:.3f} "
          f"lesion={mlesion:.3f} scram={mscram:.3f} | TRUE rank-1 all seeds={all_rank1}")
    go = (macc > mfloor + 0.2 and mperm < mfloor + 0.1 and all_rank1 and mlesion < mfloor + 0.1
          and mscram < mfloor + 0.1)
    print(f"GATE: held-out>>floor? {macc:.3f}>{mfloor+0.2:.3f}  perm collapse? {mperm:.3f}  "
          f"TRUE rank-1? {all_rank1}  lesion collapse? {mlesion:.3f}  scram chance? {mscram:.3f}  "
          f"=> {'GO' if go else 'NO-GO'}")
    return results, go


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["add", "bind"], default="add")
    ap.add_argument("--spiking", action="store_true", help="run the analogy transform+apply through the real RF bind")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--D", type=int, default=256)
    args = ap.parse_args()
    main(seeds=tuple(args.seeds), D=args.D, mode=args.mode, spiking=args.spiking)
