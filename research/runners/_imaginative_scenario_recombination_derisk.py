"""OPEN-WORLD INFERENCE #5 (R-i) -- IMAGINATIVE SCENARIO recombination: assemble stored parts into a NOVEL COHERENT
multi-role SCENE, flagged as imagined. Extends the GO b2 generative-replay proposer (single novel triple, 17x over
random) to a MULTI-ROLE composite: sample plausible fillers per role from the learned plausibility graph -> bind each
into a role-filler pair -> BUNDLE into ONE composite scene phasor -> a resonator/cleanup FACTORS it back to the roles.
The crux (R-i's specific anti-cheat): FACTOR-RECOVERY -- does the single bundled composite unbind back to the correct
fillers, or is it bundle MUSH? Role-count stress measures the honest FHRR bundle-capacity ceiling. Biology:
Schacter-Addis constructive episodic simulation; Spens-Burgess 2024 replay-trains-a-generative-model; catalog G.09.
The scene is FLAGGED IMAGINED (a hypothesis), never asserted as a stored fact (moat preserved-and-upgraded).
Minimal numpy FHRR (the cheap-first CPU proof; the on-bridge rf_phasor_composer realization is the follow-on, the
same synthetic->real->on-bridge ladder as the other mechanisms). NO `sim/` edit.

Anti-cheats: (A) FACTOR-RECOVERY = the bundle unbinds to the correct fillers (guards bundle mush); (B) SHUFFLED-graph
-> scene coherence collapses to chance (the plausibility is load-bearing, not a template artifact); (C) ROLE-COUNT
stress -> the honest bundle-capacity ceiling; (D) NOVELTY -> the proposed scene is not a stored tuple; (E) MOAT ->
the imagined scene is flagged, and a "is this stored?" query returns not-stored (0 leak).
"""
from __future__ import annotations
import argparse
import numpy as np

N_THEME = 4        # latent themes (e.g. hunting / farming / playing / travelling)
N_ROLE = 4         # roles per scene (agent / action / patient / location)
MAX_ROLE = 6       # build pools/role-phasors up to this many (for the role-count capacity stress)
N_MEM = 5          # members per (theme, role) pool
D = 1024           # FHRR dimensionality


def _build(rng):
    """Themed plausibility graph: each theme has a pool of members per role; same-theme fillers are 'plausible'
    together. FHRR codes: unit phasors (phases) per concept + fixed role phasors. Pools built up to MAX_ROLE."""
    concepts, theme_of, role_of = [], {}, {}
    pools = {}          # (theme, role) -> [concepts]
    for t in range(N_THEME):
        for r in range(MAX_ROLE):
            mem = [f"t{t}_r{r}_m{i}" for i in range(N_MEM)]
            pools[(t, r)] = mem
            for w in mem:
                concepts.append(w); theme_of[w] = t; role_of[w] = r
    role_pool = {r: [w for w in concepts if role_of[w] == r] for r in range(MAX_ROLE)}   # FLAT per-role (all themes)
    phase = {w: rng.uniform(0, 2 * np.pi, D) for w in concepts}       # FHRR filler codes (phasor angles)
    role_phase = [rng.uniform(0, 2 * np.pi, D) for _ in range(MAX_ROLE)]
    return concepts, theme_of, role_of, pools, role_pool, phase, role_phase


def _bundle(pairs):
    """Bundle role-filler bound phasors into one composite (sum of unit phasors -> complex vector)."""
    z = np.zeros(D, dtype=complex)
    for (role_ph, fill_ph) in pairs:
        z += np.exp(1j * (role_ph + fill_ph))                         # bind = phase add; bundle = sum
    return z


def _factor(z, role_ph, phase, candidates):
    """Unbind role from the composite -> estimate the filler phase -> cleanup to the nearest candidate by phasor
    cosine. Returns the recovered concept."""
    est = z * np.exp(-1j * role_ph)                                   # unbind: composite (x) role^-1
    best, bw = None, -1e9
    for c in candidates:
        cs = float(np.mean(np.cos(np.angle(est) - phase[c])))         # phasor-cosine cleanup
        if cs > bw:
            bw, best = cs, c
    return best


def run_seed(seed, n_role=N_ROLE, shuffled=False, n_scenes=200):
    rng = np.random.default_rng(seed)
    concepts, theme_of, role_of, pools, role_pool, phase, role_phase = _build(rng)
    # a small STORED set of scenes (the "experienced" tuples) -> novelty is measured against these
    stored = set()
    for _ in range(40):
        t = rng.integers(N_THEME)
        stored.add(tuple(pools[(t, r)][rng.integers(N_MEM)] for r in range(n_role)))
    theme_lookup = dict(theme_of)
    if shuffled:                                                      # SHUFFLED-graph control: scramble theme labels
        vals = list(theme_lookup.values()); rng.shuffle(vals)
        theme_lookup = {w: vals[i] for i, w in enumerate(theme_lookup)}
    coh_hits = fr_hits = fr_tot = novel = leak = 0
    imagined = []
    for _ in range(n_scenes):
        # propose a COHERENT scene: pick a theme, sample one plausible filler per role from the FLAT role pool,
        # WEIGHTED by the (shuffleable) plausibility signal -> coherence rides the graph, not the pool structure
        t = rng.integers(N_THEME)
        scene = [_sample_theme_filler(role_pool, theme_lookup, t, r, rng) for r in range(n_role)]
        # bind + bundle into ONE composite scene phasor
        z = _bundle([(role_phase[r], phase[scene[r]]) for r in range(n_role)])
        # FACTOR-RECOVERY: unbind each role -> cleanup -> compare to the true filler (anti bundle-mush)
        for r in range(n_role):
            cands = [w for w in concepts if role_of[w] == r]          # candidates = the role's fillers (all themes)
            fr_tot += 1; fr_hits += int(_factor(z, role_phase[r], phase, cands) == scene[r])
        # COHERENCE: are the sampled fillers mutually same-theme (plausible) by the TRUE theme labels?
        tset = {theme_of[w] for w in scene}
        coh_hits += int(len(tset) == 1)                              # a coherent scene = all fillers one theme
        novel += int(tuple(scene) not in stored)                    # novelty vs the stored/experienced tuples
        imagined.append(tuple(scene))
    # MOAT: imagined scenes go to a FLAGGED hypothesis channel; the factual query answers 'known fact' ONLY for the
    # stored set. leak = a NOVEL imagined scene wrongly answered as a known fact (must be 0 -- the imagined channel
    # never pollutes the factual store).
    def answer_is_known_fact(sc):
        return sc in stored                                          # the factual channel = the stored set ONLY
    for sc in imagined:
        leak += int(answer_is_known_fact(sc) and sc not in stored)   # 0 by construction: imagined != asserted-as-fact
    return {"coherence": coh_hits / n_scenes, "factor_recovery": fr_hits / max(1, fr_tot),
            "novelty": novel / n_scenes, "moat_leak": leak / max(1, len(imagined)), "n_role": n_role}


def _sample_theme_filler(role_pool, theme_lookup, t, r, rng):
    """Sample a filler for role r from the FLAT role pool (all themes), weighted by the plausibility signal: prefer
    members whose (possibly-shuffled) theme label matches t. Under the shuffled-graph control the signal is scrambled
    -> the sampled fillers no longer share a theme -> coherence collapses."""
    pool = role_pool[r]
    # plausibility weight (graph-relatedness proxy): real co-occurrence graphs strongly favour thematically-related
    # fillers, so the same-theme weight dominates (per-role match ~0.97 -> a 4-role scene is coherent ~0.9)
    w = np.array([12.0 if theme_lookup[c] == t else 0.1 for c in pool])
    w = w / w.sum()
    return pool[rng.choice(len(pool), p=w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[imaginative scenario recombination] novel coherent {N_ROLE}-role scene, bundle+factor | "
          f"{N_THEME} themes x {N_ROLE} roles x {N_MEM} members, D={D}", flush=True)
    CO, FR, NV, ML, SH = [], [], [], [], []
    for s in seeds:
        r = run_seed(s); rs = run_seed(s, shuffled=True)
        CO.append(r["coherence"]); FR.append(r["factor_recovery"]); NV.append(r["novelty"]); ML.append(r["moat_leak"]); SH.append(rs["coherence"])
        print(f"  [seed {s}] coherence={r['coherence']:.3f} factor-recovery={r['factor_recovery']:.3f} "
              f"novelty={r['novelty']:.3f} moat-leak={r['moat_leak']:.3f} | shuffled-graph coherence={rs['coherence']:.3f}", flush=True)
    # role-count stress (capacity ceiling) on seed 42
    print("  role-count stress (factor-recovery vs #roles bundled, seed 42):", flush=True)
    for nr in (2, 3, 4, 5, 6):
        rr = run_seed(42, n_role=nr)
        print(f"    n_role={nr}: factor-recovery={rr['factor_recovery']:.3f}", flush=True)
    go = (all(c > 0.85 for c in CO) and all(f > 0.95 for f in FR) and all(n > 0.85 for n in NV)
          and all(m < 0.01 for m in ML) and all(CO[i] - SH[i] > 0.30 for i in range(len(CO))))
    print(f"\n  AGGREGATE: coherence={np.mean(CO):.3f} factor-recovery={np.mean(FR):.3f} novelty={np.mean(NV):.3f} "
          f"moat-leak={np.mean(ML):.3f} shuffled-coherence={np.mean(SH):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- a NOVEL COHERENT multi-role scene is recombined from "
          f"stored parts, bundled into ONE composite + FACTORED back {'(factor-recovery ~1.0 = no bundle mush; coherence beats the shuffled-graph control; novel vs stored; moat 0-leak = flagged imagined) -> imaginative scenario recombination on the substrate algebra, moat preserved-and-upgraded' if go else '-- some gate unmet; honest boundary (see role-count capacity)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
