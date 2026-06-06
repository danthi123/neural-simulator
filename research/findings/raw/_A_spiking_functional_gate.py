"""A deep-grounding functional gate: do the ON-BRIDGE SPIKING-decorrelated grounded codes COMPOSE? Reuses the
2026-06-04 multimodal benchmark (nouns->V1 Gabor, verbs+adjs->word encoder; numpy VSA reference agent) but replaces
the numpy ZCA decorrelation with the spiking competitive+anti-Hebbian layer (`_A_spiking_decorrelation`). Compares
RAW (no decorrelation) vs ZCA (the 2026-06-04 stand-in) vs SPIKING (the on-bridge realization). Quantifies the
functional cost of the worst-pair boundary: the spiking layer drops MEAN coherence but leaves a residual worst-pair,
so composition should land between RAW (66.7%) and ZCA (100%). NO sim/ edits. Honest framing: numpy VSA reference.
"""
import json
import numpy as np

from research.runners.unified_agent_multimodal_grounded import build_multimodal_features, D
from research.runners.unified_agent_visual_grounded import _decorrelate
from research.runners.unified_agent_benchmark import build_vocab, ALL_FACTS, CATEGORIES, WHO_QUERIES, ABSTAIN_QUERIES
from research.runners.nested_composition_agent import NestedCompositionAgent
from research.findings.raw._A_spiking_decorrelation import build, it_code, coherence


def spiking_codes(feats, seed, n_it=4000, epochs=20, ff_density=0.06, ff_weight=25.0):
    """Run the on-bridge spiking competitive+anti-Hebbian decorrelation on the 320 grounded features -> IT codes.
    Fair capacity: n_it=4000 (12.5 neurons/concept) + SPARSE feed-forward (density 0.06 -> ~2.5M synapses, feasible;
    dense would be ~hours) + stronger weight for the sparse fan-in (homeostasis bootstraps the rest)."""
    b = build(seed, feats.shape[1], n_it=n_it, ff_density=ff_density, ff_weight=ff_weight)
    rm = b.region_manager
    inp = np.asarray(rm.indices("inp")); it = np.asarray(rm.indices("it"))
    for g in ("ff", "lat"):
        try:
            b.set_plasticity_gate(g, 1.0)
        except KeyError:
            pass
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(feats)):
            it_code(b, inp, it, feats[i])
    for g in ("ff", "lat"):
        try:
            b.set_plasticity_gate(g, 0.0)
        except KeyError:
            pass
    return np.stack([it_code(b, inp, it, feats[i]) for i in range(len(feats))])


def bench_codes(seed, codes, tokens, nouns, verbs, adjs):
    """Project codes -> phases -> NestedCompositionAgent -> the full capability benchmark."""
    rng = np.random.default_rng(seed)
    dim = codes.shape[1]
    proj = rng.standard_normal((D, dim)) + 1j * rng.standard_normal((D, dim))
    Z = proj @ codes.T
    ext = {t: np.angle(Z[:, i]) for i, t in enumerate(tokens)}
    agent = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=ext)
    for ag, ac, pa in ALL_FACTS:
        agent.learn(ag, ac, pa)
    res = {}
    for name, facts in CATEGORIES:
        ok = sum(int(agent.query_patient(ag, ac) == (pa if isinstance(pa, str) else agent._render_filler(pa)))
                 for ag, ac, pa in facts)
        res[name] = [ok, len(facts)]
    res["who"] = [sum(int(agent.query_agent(ac, pn) == w) for ac, pn, w in WHO_QUERIES), len(WHO_QUERIES)]
    res["abstain"] = [sum(int(agent.query_patient(ag, ac) is None) for ag, ac in ABSTAIN_QUERIES), len(ABSTAIN_QUERIES)]
    tot_ok = sum(v[0] for v in res.values()); tot_n = sum(v[1] for v in res.values())
    return res, tot_ok, tot_n


def main():
    nouns, verbs, adjs = build_vocab()
    feats, dim, tokens = build_multimodal_features(nouns, verbs, adjs)
    print(f"features: {feats.shape}, tokens: {len(tokens)} ({len(nouns)}n/{len(verbs)}v/{len(adjs)}a)", flush=True)
    out = {}
    n_it = int(__import__("os").environ.get("NIT", "6000"))
    for seed in (42,):
        sp = spiking_codes(feats, seed, n_it=n_it)
        cm, cx = coherence(sp); active = (sp > 0).sum(1)
        print(f"  seed={seed} SPIKING codes: coh mean={cm:.3f}/max={cx:.3f}  mean_active={active.mean():.1f}/"
              f"{sp.shape[1]}  n_silent={int((active == 0).sum())}/{len(sp)}  (RAW coh "
              f"{coherence(feats)[0]:.3f}, ZCA {coherence(_decorrelate(feats))[0]:.3f})", flush=True)
        for label, codes in (("RAW", feats), ("ZCA", _decorrelate(feats)), ("SPIKING", sp)):
            res, ok, n = bench_codes(seed, codes, tokens, nouns, verbs, adjs)
            out[f"{label}_{seed}"] = {"overall": [ok, n], "cats": res}
            print(f"  seed={seed} {label:8s}: overall {ok}/{n} = {100*ok/n:.1f}%  | "
                  + "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in res.items()), flush=True)
    with open("research/findings/raw/_A_spiking_functional_gate.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
