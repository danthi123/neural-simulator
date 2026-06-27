"""Tier 2.5 -- CHEAP-FIRST DE-RISK (HARD GATE): tense/aspect as a bound fact-tag.

WHAT THIS DE-RISKS (the HARD GATE the build is gated on)
  A bare SVO triple has no TIME. Tense is the cheapest temporal-representation step (front-4 #3). This binds a
  TENSE role-tag (PAST/PRESENT/FUTURE, cleaned ONLY against a 3-word codebook) onto each fact and DRIVES the
  surface verb form (PAST->"went" / PRESENT->"goes" / FUTURE->"will go").

THE MECHANISM IT COMPOSES (so no research gate -- the PROVEN polarity/negation + common-ground tag pattern)
  RFPhasorComposer binds a POLARITY role (AFFIRM/NEGATE) onto a fact, cleaned only against pol_words, read back at
  query time (ask_yes_no). CommonGroundComposer (commit 43f6bda4) is the SAME pattern at a SHARED/PRIVATE role.
  Tense is that pattern again at a TENSE role. PLUS the no-confab moat.

THE HARD GATE -- ALL must hold or STOP + write the honest NEGATIVE:
  (1) the tense tag is READ BACK faithfully (PAST/PRESENT/FUTURE) >> chance (1/3);
  (2) it DRIVES the rendered surface verb form correctly (went / goes / will go);
  (3) a PERMUTED-tag control COLLAPSES the rendered tense (the form follows the REAL tag, not chance);
  (4) LESION the tag -> the rendered tense defaults/fails (degrades to present-only);
  (5) the no-confab MOAT holds (an unknown fact -> no fabricated tense, 0 false-accepts).

Reuse-by-import: SUBCLASS ArgStructureComposer (NO sim/ edit, NO existing-composer edit -- the proven tag
mechanism + moat + the argstructure frame renderer). The tag bind/unbind run THROUGH the composer's RF spiking
_bind/_unbind; numpy phase arithmetic is the == CPU/test-oracle path.

CPU-safe (numpy fast path). Run:
  SIM_BACKEND=numpy python -m research.runners._tier2_5_tense_aspect_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.tense_aspect_composer import TenseAspectComposer, inflect  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# The de-risk corpus: distinct argument-structure facts, each tagged PAST / PRESENT / FUTURE (balanced over the
# three classes so chance for tag-fidelity = 1/3 and a permuted control has a clean ceiling).
# ---------------------------------------------------------------------------------------------------------------
def _corpus():
    """9 distinct (agent, action, GOAL) facts, balanced 3 PAST / 3 PRESENT / 3 FUTURE. Verbs are GOAL-frame verbs
    (go/come/run/walk) so the frame renders 'the <agent> <V> to the <GOAL>'. Content words drawn from a vocab that
    includes them all (the _build extends the composer vocab)."""
    return [
        ({"agent": "boy", "action": "go", "GOAL": "park"}, "PAST"),
        ({"agent": "cat", "action": "run", "GOAL": "home"}, "PRESENT"),
        ({"agent": "dog", "action": "come", "GOAL": "home"}, "FUTURE"),
        ({"agent": "girl", "action": "walk", "GOAL": "school"}, "PAST"),
        ({"agent": "dog", "action": "run", "GOAL": "park"}, "PRESENT"),
        ({"agent": "boy", "action": "come", "GOAL": "school"}, "FUTURE"),
        ({"agent": "cat", "action": "go", "GOAL": "river"}, "PAST"),
        ({"agent": "girl", "action": "run", "GOAL": "river"}, "PRESENT"),
        ({"agent": "dog", "action": "walk", "GOAL": "home"}, "FUTURE"),
    ]


def _vocab_for(facts, base):
    need = set(base)
    for fct, _ in facts:
        need |= {fct["agent"], fct["action"], fct.get("GOAL")}
    return sorted(w for w in need if w is not None)


def _build(seed, D=64):
    """A TenseAspectComposer with the de-risk corpus stored, each fact carrying its PAST/PRESENT/FUTURE tag."""
    facts = _corpus()
    base = TenseAspectComposer(seed=seed, D=D).words
    vocab = _vocab_for(facts, base)
    c = TenseAspectComposer(seed=seed, D=D, vocab=vocab)
    for fct, tn in facts:
        c.store_tensed(fct, tense=tn)
    return c, facts


def _tense_of_render(rendered, verb):
    """Classify the surface tense of a rendered sentence by the inflected verb form it contains. Returns one of
    PAST/PRESENT/FUTURE or None (unclassifiable). Matches the form against the three inflections of THIS verb so we
    measure that the bound tag drove the RIGHT form (not just that some tense word appeared)."""
    if rendered is None:
        return None
    forms = {t: inflect(verb, t) for t in ("PAST", "PRESENT", "FUTURE")}
    toks = rendered.split()
    if forms["FUTURE"].split()[0] == "will" and "will" in toks:   # 'will <verb>'
        return "FUTURE"
    if forms["PAST"] in toks:
        return "PAST"
    if forms["PRESENT"] in toks:
        return "PRESENT"
    return None


def run_seed(seed, D=64, verbose=False):
    c, facts = _build(seed, D=D)
    rng = np.random.default_rng(seed + 991)
    n = len(facts)

    # (1) tag read-fidelity: does the bound tense tag read back correctly? (chance = 1/3)
    tag_correct = sum(1 for (fct, tn) in facts if c.read_tense(fct) == tn)
    tag_fidelity = tag_correct / n

    # (2) render-fidelity: does the bound tag DRIVE the correct surface verb form?
    render_correct = 0
    sample = []
    for (fct, tn) in facts:
        rendered = c.render_tensed(fct)
        surf = _tense_of_render(rendered, fct["action"])
        render_correct += int(surf == tn)
        sample.append((fct, tn, rendered, surf))
    render_fidelity = render_correct / n

    # (3) permuted-tag control: store the SAME facts but with the tag assignment SHUFFLED (decorrelated from each
    #     fact's truth). The rendered tense should then track the PERMUTED tag, NOT the true tense -> agreement with
    #     the TRUE tense collapses toward chance (1/3). Average over several derangement-ish permutations.
    perm_render_correct = []
    for _ in range(8):
        p = rng.permutation(n)
        cp = TenseAspectComposer(seed=seed, D=D, vocab=c.words)
        for i, (fct, _tn) in enumerate(facts):
            cp.store_tensed(fct, tense=facts[p[i]][1])   # this fact gets the PERMUTED tag
        ok = 0
        for (fct, tn) in facts:
            surf = _tense_of_render(cp.render_tensed(fct), fct["action"])
            ok += int(surf == tn)            # agreement with the TRUE tense (should collapse)
        perm_render_correct.append(ok / n)
    permuted_render = float(np.mean(perm_render_correct))

    # (4) lesion: sever the tense read -> the render defaults to PRESENT regardless of the stored tag. Agreement
    #     with the true tense collapses to the fraction that were actually PRESENT (1/3 by construction).
    lesion_correct = 0
    for (fct, tn) in facts:
        surf = _tense_of_render(c.render_tensed(fct, lesion_tense=True), fct["action"])
        lesion_correct += int(surf == tn)
    lesion_render = lesion_correct / n

    # (5) the no-confab MOAT:
    # (a) a held-out fact whose cue roles were NEVER stored -> read_tense returns None (no fabricated tense).
    held_out = [
        {"agent": "boy", "action": "stop", "GOAL": "river"},   # (boy,stop) never stored
        {"agent": "dog", "action": "look", "GOAL": "park"},    # (dog,look) never stored
        {"agent": "boy", "action": "go", "GOAL": "school"},    # (boy,go) stored but as GOAL=park, not school
    ]
    moat_fab = sum(1 for fct in held_out if c.read_tense(fct) is not None)
    # (b) a render over an unknown subject -> None (no invented tensed sentence).
    moat_render_fab = sum(1 for fct in [{"agent": "horse", "action": "go", "GOAL": "park"}]
                          if c.render_tensed(fct) is not None)

    if verbose:
        print(f"  [seed {seed}] tag_fidelity={tag_fidelity:.3f} render_fidelity={render_fidelity:.3f} "
              f"permuted_render={permuted_render:.3f} lesion_render={lesion_render:.3f} | "
              f"moat_fab={moat_fab} moat_render_fab={moat_render_fab}")
        if seed == 42:
            for (fct, tn, rendered, surf) in sample[:3]:
                print(f"      {fct['agent']} {fct['action']} {fct.get('GOAL')} [{tn}] -> {rendered!r}  (surf={surf})")

    return {
        "seed": seed,
        "tag_fidelity": tag_fidelity,
        "render_fidelity": render_fidelity,
        "permuted_render": permuted_render,
        "lesion_render": lesion_render,
        "moat_fab": moat_fab,
        "moat_render_fab": moat_render_fab,
    }


def main():
    ap = argparse.ArgumentParser(description="Tier 2.5 tense/aspect HARD-GATE de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    # D=128 is the validated GO point: at D=64 a 4-bound-role composite (agent+action+GOAL+TENSE) occasionally
    # mis-cleans an UNRELATED cue role (e.g. seed 101's GOAL home->go) -> read_tense's whole-fact cue match abstains
    # (never a WRONG tense; the tag itself reads FUTURE faithfully). Raising D is the standard VSA bundle-SNR lever
    # (production composers run D=2048); D=128 clears all 6 seeds (tag+render fidelity 1.000). See the finding.
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()

    print("=" * 98)
    print("Tier 2.5 -- tense/aspect as a bound fact-tag (PAST/PRESENT/FUTURE -> surface verb form) -- HARD-GATE")
    print(f"  seeds={args.seeds} D={args.D}")
    print("  GATE: tag read >> 1/3 AND render drives the form AND permuted collapses AND lesion collapses AND moat 0-FA")
    print("=" * 98)

    rows = [run_seed(s, D=args.D, verbose=True) for s in args.seeds]

    tag_fid = float(np.mean([r["tag_fidelity"] for r in rows]))
    rnd_fid = float(np.mean([r["render_fidelity"] for r in rows]))
    permuted = float(np.mean([r["permuted_render"] for r in rows]))
    lesion = float(np.mean([r["lesion_render"] for r in rows]))
    moat_fab = sum(r["moat_fab"] for r in rows)
    moat_rfab = sum(r["moat_render_fab"] for r in rows)

    print("-" * 98)
    print(f"AGGREGATE ({len(rows)} seeds): tag_fidelity={tag_fid:.3f} render_fidelity={rnd_fid:.3f} "
          f"permuted_render={permuted:.3f} lesion_render={lesion:.3f}")
    print(f"           moat_fab={moat_fab}  moat_render_fab={moat_rfab}  (chance for tag/render = 1/3 = 0.333)")

    # GATE conditions (all must hold)
    g_tag = tag_fid >= 0.99                          # the bound tense reads back faithfully (>> 1/3 chance)
    g_render = rnd_fid >= 0.99                        # the bound tag drives the correct surface verb form
    g_permuted = permuted <= 0.45                     # permuted collapses toward chance (1/3) -- form follows real tag
    g_lesion = lesion <= 0.45                         # lesion collapses to present-only (~1/3 by construction)
    g_moat = (moat_fab == 0 and moat_rfab == 0)       # no fabricated tense / no invented tensed sentence
    verdict = g_tag and g_render and g_permuted and g_lesion and g_moat

    print("-" * 98)
    print(f"  GATE tag-read-fidelity (>=0.99, chance 0.333):   {g_tag}")
    print(f"  GATE render-drives-form (>=0.99):                {g_render}")
    print(f"  GATE permuted-collapses (<=0.45):                {g_permuted}")
    print(f"  GATE lesion-collapses (<=0.45):                  {g_lesion}")
    print(f"  GATE moat-0-FA (fab==0 & render_fab==0):         {g_moat}")
    print("=" * 98)
    print(f"  VERDICT: {'GO' if verdict else 'NO-GO'}")
    print("=" * 98)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
