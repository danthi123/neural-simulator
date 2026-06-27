"""Regime-B B1 — a CORPUS-MINED ORDINAL RELATION AXIS — the cheap-first DE-RISK.

THE REGIME-A -> REGIME-B UNLOCK. Tier 2.3 (transitive inference) GO-ed with the ordinal axis GIVEN by hand:
its `ADJ_PAIRS` premises were hand-coded (`A>B, B>C, ...`). That is REGIME-A (curated structure). This runner
asks the qualitatively STRONGER regime-B question: can the SAME ordinal-map learner build a relation axis from
premises MINED FROM THE CORPUS over the brain's OWN learned vocab -- structure ACQUIRED, not given?
Spec: research/findings/2026-06-27-regime-b-learned-knowledge-reasoning-research-gate.md (option a + the §4 bar).

THE MECHANISM (the converter = wiring two validated halves; reuse-by-import, NO sim/ edit):
  half 1 -- MINE ordered premises from the corpus. A relation with an ordinal SIGNATURE = SIZE. Encyclopedic
    text almost never states pairwise "a lion is bigger than a mouse" (empirically 0 such hits over animals in
    Simple-Wiki), so the biologically-correct mining is DISTRIBUTIONAL / Hearst-style relation extraction over
    SCALAR ADJECTIVES (the spec's "corpus-attested comparatives, scalar adjectives"): an item that co-occurs with
    huge/giant/enormous ranks HIGH on size; one co-occurring with tiny/small/little ranks LOW. Each item gets a
    corpus-derived scalar score = (#HIGH-context - #LOW-context)/freq, computed ONLY over the brain's learned
    vocab. Sort -> a CORPUS-MINED ordering -> the ADJACENT mined pairs are the PREMISES (each provably from N
    attested co-occurrences; provenance logged). This is host-side curriculum prep (legitimate per BRAIN-BASED-
    ONLY: preparing the syllabus), exactly like _corpus_svo_extract.py's attested-fact mining + provenance.
  half 2 -- LEARN THE AXIS via the Tier 2.3 Betasort biased-ordinal objective (`learn_positions`, REUSED
    verbatim) over the MINED premises -- the SAME objective Tier 2.3 used on hand-coded premises. Infer HELD-OUT
    UNSTATED comparisons by comparing learned map positions through the SAME Wang-2002 spiking accumulator.

WHY THIS IS STRONGER THAN TIER 2.3, AND HOW WE PROVE IT (the burned-capability bar; reasoning-over-learned-
knowledge is exactly where over-claims are tempting):
  - the HELD-OUT inferences are graded against an EXTERNAL ground-truth size order (NOT the mined order -- that
    would be circular). A held-out pair is one whose order is NOT an adjacent MINED premise (no train/test leak).
  - THE SYMBOLIC-DISTANCE EFFECT (mandatory, same as 2.3): accuracy/margin rise monotonically with ground-truth
    ordinal distance -- the artifact-proof positive signature a lookup / co-occurrence-overlap cannot fake.
  - ** THE PERMUTED-MINING CONTROL (the NEW, decisive regime-B control) **: mine premises for a SCRAMBLED
    relation (the size-marking adjectives RELABELLED onto random words -> a "size" signal that is noise) -> the
    learned axis must NOT predict the ground-truth held-out. This proves the MINING is load-bearing -- that the
    corpus-attested size premises, not the mining apparatus, carry the order. (Tier 2.3 had no analogue: its
    premises were given, so it could not ask "is the corpus the source?")
  - PROVENANCE / no-leakage: every premise is corpus-attested (>= min co-occurrence); the held-out pairs are
    asserted NEVER to be adjacent mined premises.
  - permuted-ORDER collapse + the mined order in the TOP ~2% of orderings; lesion (scramble the map) collapses;
    spreading-activation
    baseline FAILS its signature; moat 0-FA; 6-seed.

GATE (>=6 seeds) -- GO requires ALL of:
  (i)   held-out unstated-comparison accuracy >> chance (0.5) AND >> the memorization floor (stored-premise lookup
        = chance on unstated pairs);
  (ii)  THE SYMBOLIC-DISTANCE EFFECT: margin (host) AND accuracy (spiking) rise monotonically with ground-truth
        distance (rho>0 every seed). A FLAT curve => NO-GO regardless of raw accuracy.
  (iii) PERMUTED-MINING collapses: the scrambled-relation axis is at/near chance on the ground-truth held-out
        (the corpus-attested size premises, not the apparatus, carry the order). permuted-mining ~ true => NO-GO.
  (iv)  permuted-ORDER collapses + the mined order is in the TOP ~2% of orderings for predicting the GT held-out
        (the mined order is lossy, so "uniquely-best/rank-1" is the wrong bar; "extreme top of the distribution"
        is -- it beats >=98% of random orderings; perms_beating_true reported transparently);
  (v)   LESION the map (scramble positions) -> held-out drops to chance + the curve flattens;
  (vi)  the SPREADING-ACTIVATION baseline (symmetric co-occurrence over the mined premises) is at chance on the
        order 2AFC (associative models fail TI -- literature-guaranteed);
  (vii) PROVENANCE: every premise corpus-attested; held-out pairs never an adjacent mined premise (no leak);
  (viii) no-confab MOAT 0-FA: an item never placed on the axis -> abstain (None).

Run (CPU/numpy fast path for the mine + map + curves; spiking-accumulator parity on >=1 seed):
  SIM_BACKEND=numpy python -m research.runners._regimeb_corpus_mined_axis_derisk --seeds 42 43 44 45 46 47
  SIM_BACKEND=cupy  python -m research.runners._regimeb_corpus_mined_axis_derisk --seeds 42 --spiking-accumulator
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# REUSE-BY-IMPORT the Tier 2.3 ordinal-map machinery VERBATIM (half 2's comparator + the curve statistics).
# (`learn_positions` itself can only place the module-level A..G items, so the axis learner is reused via the
# byte-identical `_learn_positions_items` below -- same Betasort objective, our animal item universe.)
from research.runners._transitive_ordinal_map_derisk import (  # noqa: E402
    compare_host, build_comparator_bridge, compare_spiking, _curve, _curve_rho, _spearman,
)

# ------------------------------------------------------------------------------------------------------------
# The relation: SIZE. The EXTERNAL ground-truth ordinal order (ascending size, smallest -> largest) -- a
# widely-agreed, biology-real ordering of common animals (the reference the held-out inferences are graded
# against; NOT the mined order). All 16 are in the brain's learned vocab (brainALL_w7000); Park 2020 used 16.
# rank 0 = smallest. "x > y" in this runner means "x is LARGER than y" (higher size rank).
GT_ORDER = ["ant", "mouse", "rabbit", "cat", "fox", "dog", "pig", "sheep", "wolf",
            "lion", "tiger", "bear", "horse", "cow", "elephant", "whale"]

# scalar adjectives that MARK the relation in text (the Hearst-style relation-extraction patterns for SIZE).
HIGH_ADJ = ["huge", "giant", "enormous", "massive", "large", "big", "great", "largest", "biggest", "largely"]
LOW_ADJ = ["tiny", "small", "little", "smallest", "tiniest", "smaller"]


# ------------------------------------------------------------------------------------------------------------
# Stage 0 -- MINE the ordinal axis from the corpus (half 1). DISTRIBUTIONAL scalar-adjective co-occurrence.
# ------------------------------------------------------------------------------------------------------------
def mine_size_scores(corpus_path, items, vocab, high_adj, low_adj, window=4, max_chars=80_000_000):
    """Compute each item's corpus-derived SIZE score = (#HIGH-context - #LOW-context)/freq, over a +-`window`
    token co-occurrence window, restricted to the brain's learned `vocab`. Returns (scores, provenance) where
    scores[item] is the scalar and provenance[item] = {'freq','hi','lo','examples':[sentence,...]}. Anti-cheat:
    every count is a corpus-attested co-occurrence; an example sentence is logged per item (mirrors
    _corpus_svo_extract.py's provenance discipline -- a premise is provably from the corpus, not invented)."""
    aset = set(items)
    hiset, loset = set(a for a in high_adj if a in vocab), set(a for a in low_adj if a in vocab)
    with open(corpus_path, encoding="utf-8") as fh:
        text = fh.read(max_chars).lower()
    # token stream with char offsets so we can recover an example sentence for provenance
    toks = re.findall(r"[a-z]+", text)
    hi = collections.Counter(); lo = collections.Counter(); freq = collections.Counter()
    examples = collections.defaultdict(list)
    for i, t in enumerate(toks):
        if t not in aset:
            continue
        freq[t] += 1
        a, b = max(0, i - window), min(len(toks), i + window + 1)
        ctx = toks[a:i] + toks[i + 1:b]
        hit_hi = any(c in hiset for c in ctx)
        hit_lo = any(c in loset for c in ctx)
        if hit_hi:
            hi[t] += 1
        if hit_lo:
            lo[t] += 1
        if (hit_hi or hit_lo) and len(examples[t]) < 3:
            examples[t].append(" ".join(toks[a:b]))
    scores, prov = {}, {}
    for it in items:
        f = freq[it] or 1
        scores[it] = (hi[it] - lo[it]) / f
        prov[it] = {"freq": int(freq[it]), "hi": int(hi[it]), "lo": int(lo[it]),
                    "score": scores[it], "examples": examples[it]}
    return scores, prov


def mined_order(scores, items):
    """Sort items by corpus-derived score (ascending = smallest first) -> the CORPUS-MINED ordering. Ties broken
    deterministically by item name (stable, mining-content-independent)."""
    return sorted(items, key=lambda it: (scores[it], it))


def adjacent_premises(order):
    """The ADJACENT pairs of an ordering -> the MINED PREMISES fed to the axis learner. Emitted as (Hi, Lo) =
    (larger, smaller) so the Betasort update nudges the larger item UP (matching the Tier 2.3 (Hi, Lo) convention,
    where the FIRST element is the 'greater'). order is ascending size, so the LATER item is larger."""
    return [(order[i + 1], order[i]) for i in range(len(order) - 1)]   # (larger, smaller)


# ------------------------------------------------------------------------------------------------------------
# Per-seed evaluation
# ------------------------------------------------------------------------------------------------------------
def _heldout_results(pos, heldout_pairs, gt_rank, mapped):
    """Score held-out unstated comparisons against the EXTERNAL ground-truth order. Returns per-pair
    (gt_distance, correct, margin). 'x > y' = x is LARGER (higher gt rank)."""
    res = []
    for (x, y) in heldout_pairs:
        d = abs(gt_rank[x] - gt_rank[y])
        truth = x if gt_rank[x] > gt_rank[y] else y           # higher gt rank = larger = the correct "greater"
        w, m = compare_host(pos, x, y, mapped)
        res.append((d, int(w == truth), m))
    return res


def run_seed(seed, corpus_path, npz_path, use_spiking=False, n_epochs=400, lr=0.08, asym=0.5,
             window=4, min_freq=8, max_chars=80_000_000, spiking_max_pairs=0, _cache={}):
    """One seed. The MINING is corpus-deterministic (seed-independent); the SEED varies the learned embedding's
    random init (half 2), the permuted-order sampling, the lesion scramble, and the spiking read-out noise --
    i.e. exactly what Tier 2.3 varied. (Mining once and caching keeps 6 seeds fast; the cache key includes the
    corpus/vocab/mining knobs.)"""
    t0 = time.time()
    items = list(GT_ORDER)
    gt_rank = {it: i for i, it in enumerate(items)}            # 0 = smallest

    # ---- Stage 0: mine the axis from the corpus (cached -- deterministic across seeds) ----
    ckey = (corpus_path, npz_path, window, min_freq, max_chars)
    if ckey not in _cache:
        d = np.load(npz_path, allow_pickle=True)
        vocab = set(str(w).lower() for w in d["vocab"])
        scores, prov = mine_size_scores(corpus_path, items, vocab, HIGH_ADJ, LOW_ADJ,
                                        window=window, max_chars=max_chars)
        # PROVENANCE / attestation: keep ONLY items with enough corpus evidence (>= min_freq occurrences AND at
        # least one HIGH-or-LOW context -- a premise must be corpus-ATTESTED). Items below threshold are dropped
        # from the axis (and the moat must then abstain on them).
        attested = [it for it in items if prov[it]["freq"] >= min_freq and (prov[it]["hi"] + prov[it]["lo"]) >= 1]
        m_order = mined_order(scores, attested)
        premises = adjacent_premises(m_order)                 # MINED (Hi=larger, Lo=smaller)
        _cache[ckey] = (vocab, scores, prov, attested, m_order, premises)
    vocab, scores, prov, attested, m_order, premises = _cache[ckey]
    mapped_attested = set(attested)

    # ---- the HELD-OUT set: pairs that are NOT adjacent in the MINED ordering (never a premise). Graded vs GT. ----
    premise_set = {(hi, lo) for hi, lo in premises} | {(lo, hi) for hi, lo in premises}
    heldout = [(attested[i], attested[j]) for i in range(len(attested)) for j in range(i + 1, len(attested))
               if (attested[i], attested[j]) not in premise_set and (attested[j], attested[i]) not in premise_set]
    # PROVENANCE assert: no held-out pair is a mined premise (no train/test leak)
    leak = [p for p in heldout if p in premise_set or (p[1], p[0]) in premise_set]
    assert not leak, f"LEAK: held-out pairs that are mined premises: {leak}"

    # ---- Stage 1+2: learn the axis from the MINED premises; infer held-out ----
    # NOTE on objective reuse: the imported Tier-2.3 `learn_positions` SEEDS its position dict from the module-
    # level ITEMS list (A..G), so it can only place THOSE items. Our items are the brain's learned animals, so we
    # use `_learn_positions_items` -- the BYTE-IDENTICAL Betasort asymmetric update rule, generalized only in
    # WHICH items it initializes (see its docstring). Same objective; different item universe.
    pos = _learn_positions_items(premises, attested, seed, n_epochs=n_epochs, lr=lr, asym=asym)
    rates, step_hz = positions_to_rates_items(pos, attested)

    map_res = _heldout_results(pos, heldout, gt_rank, mapped_attested)
    held_acc = float(np.mean([ok for _, ok, _ in map_res])) if map_res else 0.0
    map_acc_curve, map_mar_curve = _curve(map_res)
    rho_acc = _curve_rho(map_acc_curve)
    rho_mar = _curve_rho(map_mar_curve)

    # ---- (i) memorization floor: a stored-PREMISE lookup. Held-out pairs are NOT premises -> chance. ----
    rng_l = np.random.default_rng(seed * 17 + 3)
    lookup_hits = []
    for (x, y) in heldout:
        truth = x if gt_rank[x] > gt_rank[y] else y
        if (x, y) in premise_set:
            guess = x if (x, y) in {(hi, lo) for hi, lo in premises} else y
        else:
            guess = x if rng_l.random() < 0.5 else y          # unstored -> guess (chance)
        lookup_hits.append(int(guess == truth))
    mem_floor = float(np.mean(lookup_hits)) if lookup_hits else 0.5

    # ---- (iii) ** PERMUTED-MINING control ** (the decisive regime-B control). Relabel the size-marking
    # adjectives onto RANDOM words -> mine a SCRAMBLED 'size' signal that is noise -> the resulting axis must NOT
    # predict the GT held-out. We realize the scramble cheaply + faithfully by PERMUTING the mined scores across
    # items (= the adjectives marked random items), re-deriving the order, learning, and scoring the SAME held-out
    # pairs vs GT. (Permuting scores is equivalent to a random relabelling of which items the size-adjectives
    # attach to -- the mining apparatus is identical; only the corpus-attested SIGNAL is destroyed.) ----
    # variant 1 (cheap, seed-varied): PERMUTE the mined scores across items (= the size-adjectives attached to
    # random items). Identical apparatus; only the corpus-attested SIGNAL destroyed.
    rng_pm = np.random.default_rng(seed * 733 + 11)
    perm_scores = dict(zip(attested, rng_pm.permutation([scores[it] for it in attested])))
    pm_order = mined_order(perm_scores, attested)
    pm_premises = adjacent_premises(pm_order)
    pm_pos = _learn_positions_items(pm_premises, attested, seed, n_epochs=n_epochs, lr=lr, asym=asym)
    pm_res = _heldout_results(pm_pos, heldout, gt_rank, mapped_attested)
    pm_acc = float(np.mean([ok for _, ok, _ in pm_res])) if pm_res else 0.5
    pm_rho_mar = _spearman([d for d, _, _ in pm_res], [m for _, _, m in pm_res])
    # variant 2 (the spec's exact example -- the strongest, end-to-end form): RE-MINE from the corpus with the
    # size-marking adjectives RELABELLED onto RANDOM in-vocab words ("random word pairs labelled 'bigger'"). This
    # re-runs the ACTUAL corpus mining (mine_size_scores) with bogus markers -> a "size" signal that is noise ->
    # the resulting axis must NOT predict the GT held-out. Cached per (corpus,vocab,knobs,seed-of-relabel). ----
    rkey = ("relabel", corpus_path, npz_path, window, min_freq, max_chars)
    if rkey not in _cache:
        rng_rl = np.random.default_rng(7919)             # fixed: the bogus-marker set is corpus-deterministic
        in_vocab = sorted(w for w in vocab if w.isalpha() and len(w) >= 3 and w not in set(items)
                          and w not in set(HIGH_ADJ) and w not in set(LOW_ADJ))
        pick = rng_rl.choice(len(in_vocab), size=min(len(HIGH_ADJ) + len(LOW_ADJ), len(in_vocab)), replace=False)
        bogus = [in_vocab[i] for i in pick]
        bogus_hi, bogus_lo = bogus[:len(HIGH_ADJ)], bogus[len(HIGH_ADJ):]
        rl_scores, _ = mine_size_scores(corpus_path, items, vocab, bogus_hi, bogus_lo,
                                        window=window, max_chars=max_chars)
        _cache[rkey] = (rl_scores, bogus_hi, bogus_lo)
    rl_scores, bogus_hi, bogus_lo = _cache[rkey]
    rl_order = mined_order(rl_scores, attested)
    rl_pos = _learn_positions_items(adjacent_premises(rl_order), attested, seed, n_epochs=n_epochs, lr=lr, asym=asym)
    rl_res = _heldout_results(rl_pos, heldout, gt_rank, mapped_attested)
    pm_relabel_acc = float(np.mean([ok for _, ok, _ in rl_res])) if rl_res else 0.5

    # ---- (iv) permuted-ORDER collapse + mined-order TOP-2% (the v16_compose_permuted_check discipline) ----
    rng_p = np.random.default_rng(seed * 211 + 9)
    n_perm_sample = 200
    n_at = len(attested)
    all_idx = list(range(n_at))
    # sample random orderings of the attested items; learn each; score held-out vs GT; TRUE rank-1 iff none beats it
    beat = 0
    # the TRUE order is the MINED order; we test whether random orderings beat the MINED order's held-out acc
    for _ in range(n_perm_sample):
        perm = list(all_idx); rng_p.shuffle(perm)
        if perm == all_idx:
            continue
        p_items = [attested[k] for k in perm]
        p_premises = adjacent_premises(p_items)                # ascending in this random order
        ppos = _learn_positions_items(p_premises, attested, seed, n_epochs=120, lr=lr, asym=asym)
        hits = [ok for _, ok, _ in _heldout_results(ppos, heldout, gt_rank, mapped_attested)]
        if hits and np.mean(hits) >= held_acc - 1e-9:
            beat += 1
    # PERCENTILE-based rank discipline (the regime-B-correct form). In Tier 2.3 the trained order WAS the
    # ground-truth, so it was provably the global optimum -> strict rank-1 (0 beat it). Here the trained order is
    # the *mined* (lossy: corpus != perfect GT) order, so the right bar is "the mined order sits at the EXTREME
    # TOP of the ordering distribution" -- it must beat >=98% of random orderings on the GT held-out (a random
    # ordering closer to GT than the lossy mined one can occasionally tie). `perms_beating_true` is reported
    # transparently; the mean-over-seeds is the aggregate (cf. the Tier 2.3 lesion-noise reasoning).
    true_top2pct = (beat <= max(1, int(0.02 * n_perm_sample)))    # mined order in the top ~2%
    # a single random "order" accuracy point (for the collapse number in the summary)
    rng_p2 = np.random.default_rng(seed * 211 + 99)
    rperm = list(all_idx); rng_p2.shuffle(rperm)
    rp_items = [attested[k] for k in rperm]
    rp_pos = _learn_positions_items(adjacent_premises(rp_items), attested, seed, n_epochs=n_epochs, lr=lr, asym=asym)
    perm_order_acc = float(np.mean([ok for _, ok, _ in _heldout_results(rp_pos, heldout, gt_rank, mapped_attested)]))

    # ---- (v) lesion: scramble the learned positions -> held-out collapses + curve flattens ----
    rng_les = np.random.default_rng(seed * 53 + 7)
    scrambled_vals = rng_les.permutation([pos[it] for it in attested])
    les_pos = {it: float(scrambled_vals[i]) for i, it in enumerate(attested)}
    les_res = _heldout_results(les_pos, heldout, gt_rank, mapped_attested)
    les_acc = float(np.mean([ok for _, ok, _ in les_res])) if les_res else 0.5
    les_rho_mar = _spearman([d for d, _, _ in les_res], [m for _, _, m in les_res])

    # ---- (vi) spreading-activation negative control: symmetric co-occurrence over the MINED premises ----
    spr_sym_acc = _spreading_symmetric_acc(premises, heldout, gt_rank, attested, seed)

    # ---- (viii) moat: an item NEVER placed on the axis -> abstain (None), zero false-accepts ----
    moat_unmapped = (compare_host(pos, "Zzz", attested[0], mapped_attested)[0] is None)
    moat_both = (compare_host(pos, "Zzz", "Qqq", mapped_attested)[0] is None)

    out = {
        "seed": seed, "elapsed_s": round(time.time() - t0, 1),
        "n_attested": len(attested), "n_premises": len(premises), "n_heldout": len(heldout),
        "mined_order": m_order, "premises": premises,
        "held_out_acc": held_acc, "chance": 0.5, "mem_floor": mem_floor,
        "rho_acc": rho_acc, "rho_margin": rho_mar,
        "map_acc_curve": map_acc_curve, "map_margin_curve": map_mar_curve,
        "permuted_mining_acc": pm_acc, "permuted_mining_rho_margin": pm_rho_mar,
        "permuted_mining_relabel_acc": pm_relabel_acc,
        "perm_order_acc": perm_order_acc, "true_top2pct": true_top2pct, "perms_beating_true": beat,
        "n_perm_sample": n_perm_sample,
        "lesion_acc": les_acc, "lesion_rho_margin": les_rho_mar,
        "spread_sym_acc": spr_sym_acc,
        "moat_unmapped_abstains": bool(moat_unmapped and moat_both),
        "no_leak": True,
    }
    if use_spiking:
        bridge, idx = build_comparator_bridge(seed)
        rng_pn = np.random.default_rng(seed * 911 + 3)
        pos_read_noise_steps = 1.5
        n_trials = 8
        spiking_res = []
        # the spiking confirmation runs the comparison ON REAL SPIKES (the Wang-2002 accumulator) -- "confirm on
        # >=1 seed" per the spec. With 16 items there are 105 held-out pairs; sampling a balanced subset
        # (`spiking_max_pairs`, STRATIFIED across distances so the psychometric curve still spans 1..15) keeps the
        # spiking run tractable while preserving the distance-effect read. 0 = all 105 pairs.
        spk_pairs = heldout
        if spiking_max_pairs and len(heldout) > spiking_max_pairs:
            by_d = collections.defaultdict(list)
            for p in heldout:
                by_d[abs(gt_rank[p[0]] - gt_rank[p[1]])].append(p)
            rng_s = np.random.default_rng(seed * 1009 + 7)
            per_d = max(1, spiking_max_pairs // max(1, len(by_d)))
            spk_pairs = []
            for dd in sorted(by_d):
                grp = by_d[dd]
                take = grp if len(grp) <= per_d else [grp[i] for i in rng_s.choice(len(grp), per_d, replace=False)]
                spk_pairs.extend(take)
        # psychometric curve over the (sampled) held-out distances (the canonical distance-effect plot)
        for (x, y) in spk_pairs:
            d = abs(gt_rank[x] - gt_rank[y])
            truth = x if gt_rank[x] > gt_rank[y] else y
            for _ in range(n_trials):
                sig = pos_read_noise_steps * step_hz
                rx = rates[x] + float(rng_pn.normal(0.0, sig))
                ry = rates[y] + float(rng_pn.normal(0.0, sig))
                sp_w, sp_m = compare_spiking(bridge, idx, rx, ry)
                sp_winner = x if sp_w == "X" else (y if sp_w == "Y" else None)
                spiking_res.append((d, int(sp_winner == truth), sp_m))
        sp_acc_curve, sp_mar_curve = _curve(spiking_res)
        out["spiking_held_acc"] = float(np.mean([ok for _, ok, _ in spiking_res]))
        out["spiking_rho_acc"] = _curve_rho(sp_acc_curve)
        out["spiking_rho_margin"] = _curve_rho(sp_mar_curve)
        out["spiking_acc_curve"] = sp_acc_curve
        out["spiking_margin_curve"] = sp_mar_curve
    return out, prov


def _learn_positions_items(adj_pairs, items, seed, n_epochs=400, lr=0.08, asym=0.5):
    """The Tier 2.3 Betasort biased-ordinal objective, generalized to an ARBITRARY item set (the imported
    `learn_positions` seeds positions from the module-level ITEMS list; ours are animals). Byte-identical update
    rule -- this is the SAME OBJECTIVE, only the item universe differs. Each (Hi, Lo) nudges Hi UP, Lo DOWN by
    asym x. Near-degenerate random start -> the structure is LEARNED."""
    rng = np.random.default_rng(seed)
    pos = {it: float(rng.normal(0.0, 0.01)) for it in items}
    for _ in range(n_epochs):
        for k in rng.permutation(len(adj_pairs)):
            hi, lo = adj_pairs[int(k)]
            err = 1.0 - (pos[hi] - pos[lo])
            pos[hi] += lr * err
            pos[lo] -= lr * err * asym
    return pos


def positions_to_rates_items(pos, items, lo_hz=12.0, span_hz=8.0):
    """positions_to_rates generalized to our item set (the imported one iterates module-level ITEMS). Same linear
    rescale onto [lo_hz, lo_hz+span_hz]; returns (rates, step_hz). span over (len(items)-1) rank steps."""
    vals = np.array([pos[it] for it in items], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    rng_span = (hi - lo) if (hi - lo) > 1e-9 else 1.0
    rates = {it: lo_hz + span_hz * (pos[it] - lo) / rng_span for it in items}
    step_hz = span_hz / max(len(items) - 1, 1)
    return rates, step_hz


def _spreading_symmetric_acc(premises, heldout, gt_rank, items, seed):
    """The retracted family: UNDIRECTED co-occurrence (each mined premise an undirected edge), diffuse, rank by
    reach -> order-blind -> a 2AFC 'is x larger?' is ~a coin flip. Must FAIL (be at chance) on the held-out."""
    idx = {it: i for i, it in enumerate(items)}
    n = len(items)
    A = np.zeros((n, n))
    for hi, lo in premises:
        A[idx[hi], idx[lo]] = 1.0
        A[idx[lo], idx[hi]] = 1.0
    A = A / (A.sum(1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed * 131 + 5)
    hits = []
    for (x, y) in heldout:
        truth = x if gt_rank[x] > gt_rank[y] else y
        px = np.zeros(n); px[idx[x]] = 1.0
        py = np.zeros(n); py[idx[y]] = 1.0
        for _ in range(3):
            px = A @ px; py = A @ py
        sx = float(px.sum() + 1e-9 * rng.random())
        sy = float(py.sum() + 1e-9 * rng.random())
        guess = x if sx >= sy else y
        hits.append(int(guess == truth))
    return float(np.mean(hits)) if hits else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--corpus", default="data/corpus/simplewiki.txt")
    ap.add_argument("--npz", default="bridges/firstchat/brainALL_w7000.npz_seed42.npz")
    ap.add_argument("--spiking-accumulator", action="store_true",
                    help="also run the comparison through the REAL spiking two-pool accumulator (>=1 seed; GPU)")
    ap.add_argument("--n-epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--asym", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-freq", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=80_000_000)
    ap.add_argument("--spiking-max-pairs", type=int, default=48,
                    help="stratified-by-distance cap on held-out pairs run through the spiking accumulator "
                         "(0=all 105). Keeps the spiking confirmation tractable; the curve still spans all "
                         "distances.")
    ap.add_argument("--out", default="research/findings/raw/_regimeb_corpus_mined_axis.json")
    a = ap.parse_args()

    print(f"[regime-B corpus-mined ORDINAL AXIS de-risk] relation=SIZE | corpus={a.corpus} | "
          f"brain={os.path.basename(a.npz)}\n  half1: MINE premises from scalar-adjective co-occurrence over the "
          f"brain's learned vocab; half2: REUSE the Tier 2.3 ordinal-map objective.\n  HEADLINE controls = the "
          f"symbolic-distance effect (margin RISES with GT distance) + PERMUTED-MINING (scrambled relation "
          f"collapses).\n", flush=True)

    rows, prov0 = [], None
    for s in a.seeds:
        r, prov = run_seed(s, a.corpus, a.npz, use_spiking=a.spiking_accumulator,
                           n_epochs=a.n_epochs, lr=a.lr, asym=a.asym,
                           window=a.window, min_freq=a.min_freq, max_chars=a.max_chars,
                           spiking_max_pairs=a.spiking_max_pairs)
        rows.append(r)
        prov0 = prov
        mc = r["map_margin_curve"]
        print(f"  [seed {s}] held-out {r['held_out_acc']:.2f} (mem-floor {r['mem_floor']:.2f}, chance 0.50) | "
              f"rho(margin) {r['rho_margin']:+.2f} | "
              f"map-margin/dist {[(d, round(mc[d], 1)) for d in sorted(mc)][:6]}...", flush=True)
        print(f"           ** PERMUTED-MINING perm-score {r['permuted_mining_acc']:.2f} / relabel-adj "
              f"{r['permuted_mining_relabel_acc']:.2f} ** | permuted-order "
              f"{r['perm_order_acc']:.2f} (top2% {r['true_top2pct']}, beat {r['perms_beating_true']}/"
              f"{r['n_perm_sample']}) | "
              f"lesion {r['lesion_acc']:.2f} | spread-sym {r['spread_sym_acc']:.2f} | "
              f"moat {'ok' if r['moat_unmapped_abstains'] else 'X'} | n_attested {r['n_attested']} "
              f"n_premises {r['n_premises']} n_heldout {r['n_heldout']}", flush=True)
        if a.spiking_accumulator:
            sa = r["spiking_acc_curve"]
            print(f"           SPIKING held-out {r['spiking_held_acc']:.2f} | rho(acc) {r['spiking_rho_acc']:+.2f} "
                  f"rho(margin) {r['spiking_rho_margin']:+.2f} | spk-acc/dist "
                  f"{[(d, round(sa[d], 2)) for d in sorted(sa)][:8]}", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    held = m("held_out_acc"); rho_m = m("rho_margin"); rho_a = m("rho_acc")
    pmA = m("permuted_mining_acc"); pmR = m("permuted_mining_relabel_acc")
    permO = m("perm_order_acc"); lesA = m("lesion_acc"); memF = m("mem_floor")
    sprS = m("spread_sym_acc")
    all_top2pct = all(r["true_top2pct"] for r in rows)
    all_moat = all(r["moat_unmapped_abstains"] for r in rows)
    all_noleak = all(r["no_leak"] for r in rows)
    every_rho_mar_pos = all(r["rho_margin"] > 0.0 for r in rows)

    # GATE -- host path
    distance_effect = (rho_m > 0.3 and every_rho_mar_pos)
    spiking_ran = all("spiking_held_acc" in r for r in rows)
    spk_distance_effect = True
    if spiking_ran:
        spk_held = float(np.mean([r["spiking_held_acc"] for r in rows]))
        spk_rho_acc = float(np.mean([r["spiking_rho_acc"] for r in rows]))
        spk_rho_mar = float(np.mean([r["spiking_rho_margin"] for r in rows]))
        spk_distance_effect = (spk_rho_acc > 0.0 and spk_rho_mar > 0.0 and spk_held >= 0.7)
    # Held-out bar (0.7 / +0.15 over mem-floor) is calibrated for the LOSSIER regime-B task: the axis is MINED
    # from noisy distributional evidence (vs Tier 2.3's hand-GIVEN 0.8/+0.25), so raw accuracy is expected to be
    # lower. CRUCIALLY the *signature* controls -- the symbolic-distance effect and the permuted-mining collapse
    # -- are NOT loosened; they carry the believability (a lossy-but-real metric map still shows the monotone
    # curve and still beats a scrambled-relation mining). A high-accuracy claim here would be the LESS honest one.
    held_ok = (held >= 0.7 and held >= memF + 0.15)
    permuted_mining_ok = (pmA <= 0.62 and pmR <= 0.62)     # ** the decisive regime-B control (BOTH variants) **
    permuted_order_ok = (permO <= 0.65 and all_top2pct)
    lesion_ok = (lesA <= 0.65)
    spreading_ok = (sprS <= 0.65)
    go = (distance_effect and spk_distance_effect and held_ok and permuted_mining_ok and permuted_order_ok
          and lesion_ok and spreading_ok and all_moat and all_noleak)

    os.makedirs(os.path.join(_REPO, os.path.dirname(a.out)), exist_ok=True)
    full = os.path.join(_REPO, a.out)
    summary = {"relation": "size", "corpus": a.corpus, "brain": os.path.basename(a.npz),
               "n_items_gt": len(GT_ORDER), "n_seeds": len(a.seeds),
               "mined_order": rows[0]["mined_order"], "gt_order": GT_ORDER,
               "premises": rows[0]["premises"], "provenance": prov0,
               "held_out_acc": held, "mem_floor": memF, "rho_margin": rho_m, "rho_acc": rho_a,
               "permuted_mining_acc": pmA, "permuted_mining_relabel_acc": pmR,
               "perm_order_acc": permO, "true_top2pct_all": all_top2pct,
               "lesion_acc": lesA, "spread_sym_acc": sprS, "moat_all": all_moat, "no_leak_all": all_noleak,
               "distance_effect_margin": distance_effect, "permuted_mining_ok": permuted_mining_ok,
               "spiking_ran": spiking_ran, "go": go}
    if spiking_ran:
        summary.update({"spiking_held_acc": spk_held, "spiking_rho_acc": spk_rho_acc,
                        "spiking_rho_margin": spk_rho_mar, "spiking_distance_effect": spk_distance_effect})
    summary["per_seed"] = rows
    with open(full, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  RELATION=size | MINED ORDER (corpus, ascending): {' < '.join(rows[0]['mined_order'])}", flush=True)
    print(f"  GROUND-TRUTH  (ascending size):                  {' < '.join(GT_ORDER)}", flush=True)
    print(f"  MEAN ({len(a.seeds)} seeds): held-out {held:.3f} (chance 0.50, mem-floor {memF:.3f}) | "
          f"rho(margin) {rho_m:+.3f} [every-seed>0 {every_rho_mar_pos}]", flush=True)
    print(f"    ** PERMUTED-MINING perm-score {pmA:.3f} / relabel-adj {pmR:.3f} (both must be <=0.62 = scrambled "
          f"relation collapses) ** | "
          f"permuted-order {permO:.3f} (mined-order top-2% all-seeds {all_top2pct}) | lesion {lesA:.3f} | "
          f"spread-sym {sprS:.3f} | moat {all_moat} | no-leak {all_noleak}", flush=True)
    if spiking_ran:
        print(f"    SPIKING accumulator: held-out {spk_held:.3f} | rho(acc) {spk_rho_acc:+.3f} | "
              f"rho(margin) {spk_rho_mar:+.3f} -> distance-effect {spk_distance_effect}", flush=True)
    if go:
        print(f"\n  GO: regime-B reasoning over the brain's OWN learned knowledge is UNLOCKED for ordinal "
              f"relations. A SIZE axis MINED from corpus scalar-adjective co-occurrence (NOT hand-coded) -> the "
              f"Tier 2.3 ordinal-map learner -> held-out UNSTATED comparisons ({held:.2f}) >> chance + mem-floor "
              f"with a monotone symbolic-distance curve (rho {rho_m:+.2f}). ** PERMUTED-MINING collapses "
              f"({pmA:.2f}) ** -> the corpus-attested premises, NOT the apparatus, carry the order: structure "
              f"ACQUIRED, not given. Permuted-order collapses (mined-order top-2%), lesion collapses, spreading "
              f"fails, moat 0-FA, no leak.", flush=True)
    else:
        why = []
        if not distance_effect:
            why.append(f"NO symbolic-distance (margin) effect (rho {rho_m:+.2f}, every-seed>0 {every_rho_mar_pos})")
        if spiking_ran and not spk_distance_effect:
            why.append(f"spiking distance-effect absent (held {spk_held:.2f}, rho-acc {spk_rho_acc:+.2f})")
        if not held_ok:
            why.append(f"held-out {held:.2f} not >> chance+mem-floor ({memF:.2f})")
        if not permuted_mining_ok:
            why.append(f"** PERMUTED-MINING did not collapse (perm-score {pmA:.2f}, relabel {pmR:.2f}) -- the "
                       f"mining is NOT load-bearing **")
        if not permuted_order_ok:
            why.append(f"permuted-order did not collapse / mined-order not top-2% ({permO:.2f}, top2% {all_top2pct})")
        if not lesion_ok:
            why.append(f"lesion did not collapse ({lesA:.2f})")
        if not spreading_ok:
            why.append(f"spreading control did not fail ({sprS:.2f})")
        if not all_moat:
            why.append("moat breach")
        if not all_noleak:
            why.append("train/test LEAK")
        print(f"\n  NO-GO: {'; '.join(why)}. Per the spec this is the honest NEGATIVE -- write it up, do not "
              f"over-claim.", flush=True)
    print(f"  [saved] {full}\n{'=' * 110}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
