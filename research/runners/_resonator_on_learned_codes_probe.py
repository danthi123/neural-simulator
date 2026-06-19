"""De-risk: does the F=3 resonator's nested / two-attribute decode HOLD on the production's LEARNED 320 codes?

THE ONE QUESTION (cheap CPU/numpy de-risk before the conversational-scaling consolidation build):
  The scoping (research/findings/2026-06-19-conversational-scaling-next-lever-scoping.md) recommends
  consolidating the resonator NESTED / two-attribute decode (validated GO 6/6 in
  research/runners/nested_composition_agent.py) into the production OneBrainComposer. BUT the resonator is
  documented to need CLEAN phasor codes and to degrade on correlated codes. The production conversation runs
  on the LEARNED 320 codes (the stream/PPMI cortex, _phaseB_stream_codes_320_neural_seed{42,43,44}.npy),
  grounded into composer phases by the SAME fixed-projection map the production uses
  (consolidated_320_conversation_demo.grounded_phases: phase = angle(proj @ code) / 2pi). Those learned codes
  carry SEMANTIC CORRELATION (measured off-diag cosine mean ~0.03 but MAX ~0.83-0.89 -- a real "dog/cat are
  related" tail). Does the resonator two-attribute decode survive that, or does correlation drag it to chance?

METHOD (all numpy, no sim/ edit, no GPU):
  * CLEAN control (the ceiling the resonator GO used): random uniform phases per token (the agent's own
    fallback code()). This is the F=3-resonator best case.
  * PRODUCTION-grounded: load the genuine learned 320 stream codes, apply the production grounding projection
    angle(proj @ code) -> composer phases, hand those (in radians) to NestedCompositionAgent.external_codes
    (which expects phase arrays: exp(1j*phase)). This is EXACTLY the phasor the production composer would bind.
  * For each code source x several agent seeds: build resonator agents whose adj/noun/verb codebooks are the
    learned-grounded (or clean) phasors, store + decode:
      - FLAT patient ("dog chase cat")
      - ONE-attribute patient ("dog eat (red ball)")           [2-factor resonator]
      - TWO-attribute patient ("cat want ((big red) ball))")   [3-factor resonator -- the at-risk capability]
    over a sweep of attributed facts, count exact recovery (both attributes + the noun, with crosstalk
    subtraction -- the production query_patient path). Report recovery vs CLEAN vs chance.

  python -m research.runners._resonator_on_learned_codes_probe
"""
from __future__ import annotations
import os
import sys
import json
import itertools
import numpy as np

# the resonator agent under test (reuse-by-import; the production nested-decode capability)
from research.runners.nested_composition_agent import NestedCompositionAgent

# the production grounding map + vocab/taxonomy (verbatim source of how learned codes become composer phases)
from research.runners.consolidated_320_conversation_demo import _projection, grounded_phases, D as PROD_D
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8

RAW = "research/findings/raw"
# the genuine learned-from-conversation production codes (on-bridge "neural" read), 3 seeds
STREAM_CODES = {
    42: f"{RAW}/_phaseB_stream_codes_320_neural_seed42.npy",
    43: f"{RAW}/_phaseB_stream_codes_320_neural_seed43.npy",
    44: f"{RAW}/_phaseB_stream_codes_320_neural_seed44.npy",
}

# resonator vocab drawn from the REAL 320 taxonomy (every token exists in the learned codes)
NOUNS = TAXONOMY_40x8["animals_pets"][:4] + TAXONOMY_40x8["toys"][:4] + TAXONOMY_40x8["places"][:4]  # 12 nouns
VERBS = TAXONOMY_40x8["motion_actions"][:4] + TAXONOMY_40x8["manipulate"][:4]                          # 8 verbs
ADJS = TAXONOMY_40x8["colors"][:3] + TAXONOMY_40x8["sizes"][:3] + TAXONOMY_40x8["texture_temp"][:2]    # 8 adjs


def _vocab_order():
    """The 320 vocab order matching the stream-code rows (consolidated_320 uses this exact mapping)."""
    from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories
    vocab, _, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    return list(vocab)


def learned_grounded_phases(seed, d_out):
    """Load the learned 320 codes for `seed`, ground them with the production projection map (the SAME
    angle(proj @ code) grounding consolidated_320 uses, at output dim `d_out`), return a token ->
    phase-in-radians dict (NestedCompositionAgent.external_codes wants radians: exp(1j*phase))."""
    codes = np.load(STREAM_CODES[seed])                       # (320, 300) real PPMI cortex codes
    vocab = _vocab_order()
    proj = _projection(d_out, codes.shape[1], seed)           # fixed complex projection (d_out x 300), production-identical map
    ext = {}
    for i, w in enumerate(vocab):
        frac = grounded_phases(codes[i], proj)                # production phase in [0,1)  (== angle(proj@code)/2pi)
        ext[w] = frac * 2.0 * np.pi                            # -> radians for the agent's exp(1j*phase)
    return ext, codes, proj


def _grounded_offdiag_cos(ext, tokens):
    """Mean/max pairwise phasor cosine among the grounded phasors of `tokens` (the correlation the resonator sees)."""
    zs = {t: np.exp(1j * ext[t]) for t in tokens}
    D = len(next(iter(zs.values())))
    sims = []
    for a, b in itertools.combinations(tokens, 2):
        sims.append(float(np.abs(np.vdot(zs[a], zs[b])) / D))
    return float(np.mean(sims)), float(np.max(sims))


def eval_recovery(external_codes, agent_seed, D, n_attr_facts=8):
    """Build a resonator agent on `external_codes` (None => clean random phasors = the ceiling control) and
    measure exact decode recovery for FLAT / ONE-attribute / TWO-attribute patients via the production
    query_patient crosstalk-subtraction path. Returns dict of per-kind (n_correct, n_total)."""
    agent = NestedCompositionAgent(NOUNS, VERBS, ADJS, D=D, seed=agent_seed,
                                   external_codes=external_codes)
    rng = np.random.default_rng(agent_seed + 100)

    # --- FLAT facts (agent action noun) ---
    flat_facts = []
    for k in range(n_attr_facts):
        ag = NOUNS[k % len(NOUNS)]
        ac = VERBS[k % len(VERBS)]
        pn = NOUNS[(k + 3) % len(NOUNS)]
        if ag == pn:
            pn = NOUNS[(k + 4) % len(NOUNS)]
        flat_facts.append((ag, ac, pn))

    # --- ONE-attribute facts (agent action (adj, noun)) ---
    one_facts = []
    for k in range(n_attr_facts):
        ag = NOUNS[(k + 1) % len(NOUNS)]
        ac = VERBS[(k + 2) % len(VERBS)]
        adj = ADJS[k % len(ADJS)]
        nn = NOUNS[(k + 5) % len(NOUNS)]
        if ag == nn:
            nn = NOUNS[(k + 6) % len(NOUNS)]
        one_facts.append((ag, ac, (adj, nn)))

    # --- TWO-attribute facts (agent action ((adj, adj), noun)) -- THE AT-RISK capability ---
    two_facts = []
    adj_pairs = [p for p in itertools.combinations(range(len(ADJS)), 2)]
    rng.shuffle(adj_pairs)
    for k in range(n_attr_facts):
        ag = NOUNS[(k + 2) % len(NOUNS)]
        ac = VERBS[(k + 1) % len(VERBS)]
        ai, aj = adj_pairs[k % len(adj_pairs)]
        nn = NOUNS[(k + 7) % len(NOUNS)]
        if ag == nn:
            nn = NOUNS[(k + 8) % len(NOUNS)]
        two_facts.append((ag, ac, ((ADJS[ai], ADJS[aj]), nn)))

    out = {}

    def _run(facts, label, expect_fn):
        # each fact gets its own fresh agent KB so a unique (agent, action) cue selects it cleanly
        nc = 0
        for f in facts:
            a2 = NestedCompositionAgent(NOUNS, VERBS, ADJS, D=D, seed=agent_seed,
                                        external_codes=external_codes)
            a2.learn(*f)
            got = a2.query_patient(f[0], f[1])
            exp = expect_fn(f)
            nc += int(got == exp)
        out[label] = (nc, len(facts))

    _run(flat_facts, "flat", lambda f: f[2])
    _run(one_facts, "one_attr", lambda f: f"{f[2][0]} {f[2][1]}")
    # two-attr expected render: adjectives in vocabulary order + noun (the agent renders sorted-by-vocab)
    def two_expect(f):
        (a1, a2_), nn = f[2]
        ordered = sorted([a1, a2_], key=ADJS.index)
        return " ".join(ordered + [nn])
    _run(two_facts, "two_attr", two_expect)
    return out


def main():
    # The resonator NESTED capability fundamentally needs D>=2048 (the GO test's dimension; clause/two-attr
    # collapse at the production composer's default D=128 -- see the doc's "clause-in-clause needs D>=2048" and
    # the probe's own D=128 clean-MISS check). So we ground the LEARNED codes to D=2048 phases via the SAME
    # production projection map (just a 2048x300 fixed complex projection instead of 128x300) and test there:
    # this isolates the ONE question (does learned-code CORRELATION degrade the decode?) from the orthogonal
    # "D too small" effect. Consolidating the resonator path would require the composer run the nested ops at
    # this higher D (recorded in the findings as a cost).
    D = 2048
    n_facts = 8
    agent_seeds = [42, 43, 44]
    print(f"=== resonator nested-decode on LEARNED vs CLEAN codes (D={D}, n_facts/kind={n_facts}) ===",
          flush=True)
    print(f"    resonator vocab: {len(NOUNS)} nouns, {len(VERBS)} verbs, {len(ADJS)} adjs", flush=True)

    results = {"D": D, "n_facts": n_facts, "agent_seeds": agent_seeds,
               "vocab": {"nouns": NOUNS, "verbs": VERBS, "adjs": ADJS},
               "clean": {}, "learned": {}, "correlation": {}}

    # ---- CLEAN control (the ceiling the resonator GO used: random phasors) ----
    print("\n-- CLEAN random-phasor codes (the resonator's documented best case = ceiling) --", flush=True)
    clean_acc = {"flat": [], "one_attr": [], "two_attr": []}
    for s in agent_seeds:
        r = eval_recovery(None, s, D, n_facts)
        for k in clean_acc:
            clean_acc[k].append(r[k])
        print(f"   seed {s}: flat {r['flat'][0]}/{r['flat'][1]}  "
              f"one_attr {r['one_attr'][0]}/{r['one_attr'][1]}  "
              f"two_attr {r['two_attr'][0]}/{r['two_attr'][1]}", flush=True)
    results["clean"] = {k: [list(t) for t in v] for k, v in clean_acc.items()}

    # ---- PRODUCTION learned-grounded codes ----
    print("\n-- LEARNED stream-cortex codes, production-grounded (angle(proj@code)) --", flush=True)
    learned_acc = {"flat": [], "one_attr": [], "two_attr": []}
    used_tokens = list(dict.fromkeys(NOUNS + VERBS + ADJS))
    for s in agent_seeds:
        if not os.path.exists(STREAM_CODES[s]):
            print(f"   seed {s}: MISSING {STREAM_CODES[s]} -- skipped", flush=True)
            continue
        ext, codes, proj = learned_grounded_phases(s, D)
        cmean, cmax = _grounded_offdiag_cos(ext, used_tokens)
        results["correlation"][str(s)] = {"grounded_offdiag_cos_mean": cmean, "grounded_offdiag_cos_max": cmax}
        r = eval_recovery(ext, s, D, n_facts)
        for k in learned_acc:
            learned_acc[k].append(r[k])
        print(f"   seed {s}: flat {r['flat'][0]}/{r['flat'][1]}  "
              f"one_attr {r['one_attr'][0]}/{r['one_attr'][1]}  "
              f"two_attr {r['two_attr'][0]}/{r['two_attr'][1]}   "
              f"[grounded vocab off-diag cos mean {cmean:.3f} max {cmax:.3f}]", flush=True)
    results["learned"] = {k: [list(t) for t in v] for k, v in learned_acc.items()}

    # ---- summary + verdict ----
    def _tot(acc, k):
        c = sum(t[0] for t in acc[k]); n = sum(t[1] for t in acc[k]); return c, n
    print("\n=== SUMMARY (correct / total over agent seeds) ===", flush=True)
    chance_two = 1.0 / (len(NOUNS) * len(list(itertools.combinations(range(len(ADJS)), 2))))  # rough: noun x adj-pair
    for k in ["flat", "one_attr", "two_attr"]:
        cc, cn = _tot(clean_acc, k)
        lc, ln = _tot(learned_acc, k) if learned_acc[k] else (0, 0)
        cpct = 100.0 * cc / cn if cn else 0.0
        lpct = 100.0 * lc / ln if ln else float("nan")
        print(f"   {k:9s}:  CLEAN {cc}/{cn} ({cpct:.1f}%)   LEARNED {lc}/{ln} ({lpct:.1f}%)", flush=True)
    print(f"   (rough chance for two_attr exact (noun x adj-pair): {100.0*chance_two:.2f}%)", flush=True)
    results["chance_two_attr_pct"] = 100.0 * chance_two

    # verdict heuristic: HOLDS if learned two_attr >= 80% of clean two_attr AND well above chance
    cc2, cn2 = _tot(clean_acc, "two_attr")
    lc2, ln2 = _tot(learned_acc, "two_attr") if learned_acc["two_attr"] else (0, 0)
    clean2 = cc2 / cn2 if cn2 else 0.0
    learned2 = lc2 / ln2 if ln2 else 0.0
    verdict = "INCONCLUSIVE"
    if ln2:
        if learned2 >= 0.8 * clean2 and learned2 >= 5 * chance_two:
            verdict = "HOLDS"
        elif learned2 <= 2 * chance_two:
            verdict = "DEGRADES_TO_CHANCE"
        else:
            verdict = "PARTIAL_DEGRADE"
    results["verdict_two_attr"] = verdict
    print(f"\n=== VERDICT (two-attribute decode): {verdict} "
          f"(clean {100*clean2:.1f}% vs learned {100*learned2:.1f}%) ===", flush=True)

    os.makedirs(RAW, exist_ok=True)
    outp = f"{RAW}/_resonator_on_learned_codes.json"
    with open(outp, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {outp}", flush=True)
    return results


if __name__ == "__main__":
    main()
