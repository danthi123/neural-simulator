"""Tier 2.4 -- CHEAP-FIRST DE-RISK (HARD GATE): minimal common-ground (shared-vs-private fact tagging -> audience
design), the cheapest theory-of-mind slice.

WHAT THIS DE-RISKS (the HARD GATE the build is gated on)
  A competent speaker tracks what is MUTUALLY KNOWN (in common ground) vs what only THEY know (private), and tailors
  what they SAY to it (Clark & Brennan 1991 grounding; Stephens-Silbert-Hasson 2010 speaker-listener coupling; the
  discourse-pragmatics research front-2 target #3). AUDIENCE DESIGN = VOLUNTEER private facts (new to the listener);
  SUPPRESS-or-merely-ACKNOWLEDGE shared facts (the listener already knows them). This is the cheapest, most tractable
  slice of ToM -- it needs NO recursive belief reasoning, only a per-fact SHARED/PRIVATE tag + a read-at-response.

THE MECHANISM IT COMPOSES (so no research gate is needed -- it is the PROVEN polarity/negation tag)
  RFPhasorComposer already binds a POLARITY role onto a stored fact (AFFIRM/NEGATE, cleaned ONLY against
  `pol_words` -- rf_phasor_composer.py:159-162, 528-544, 802-822) and reads it back at query time
  (ask_yes_no -> 'yes'/'no'). Common-ground is the SAME pattern at a different role: a SHARED/PRIVATE tag bound onto
  each fact (cleaned only against a 2-word tag codebook), read at response time to decide audience design. PLUS the
  no-confab moat (an un-stored fact / cue -> abstain; never fabricate a fact OR a tag).

THE HARD GATE -- ALL must hold or STOP + write the honest NEGATIVE:
  (1) audience design TRACKS the tag (private -> tell, shared -> suppress/ack) and BEATS a no-tag baseline (which
      cannot distinguish, so it either tells everything or suppresses everything);
  (2) a PERMUTED-tag control COLLAPSES it (the behaviour follows the real tag, not chance);
  (3) LESION the tag -> audience design FAILS (degrades to the no-tag baseline);
  (4) the no-confab MOAT holds (0 false-accepts; never fabricate a fact or a tag).

Reuse-by-import: SUBCLASS RFPhasorComposer (NO sim/ edit, NO composer edit -- the proven tag mechanism + moat).
The tag bind/unbind run THROUGH the composer's RF spiking _bind/_unbind (use_spiking_bind path); numpy phase
arithmetic is the == CPU/test-oracle path.

CPU-safe (numpy fast path). Run:
  SIM_BACKEND=numpy python -m research.runners._tier2_4_common_ground_derisk --seeds 42 43 44 100 101 102
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

from research.runners.common_ground_composer import CommonGroundComposer  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# The de-risk corpus: a small set of distinct SVO facts, each tagged SHARED (listener already knows) or PRIVATE
# (only the speaker knows). A held-out fact (never stored) probes the moat.
# ---------------------------------------------------------------------------------------------------------------
def _corpus(rng):
    """8 distinct facts; 4 SHARED, 4 PRIVATE (a balanced ground-truth listener model). Returns
    [(agent, action, patient, tag)] with tag in {'SHARED','PRIVATE'}. Concepts are drawn from the composer's
    default vocab so codes exist for all of them."""
    facts = [
        ("dog", "go", "north", "SHARED"),
        ("cat", "run", "south", "PRIVATE"),
        ("dog", "look", "river", "PRIVATE"),
        ("cat", "stop", "east", "SHARED"),
        ("dog", "come", "west", "SHARED"),
        ("cat", "go", "apple", "PRIVATE"),
        ("dog", "run", "hot", "PRIVATE"),
        ("cat", "look", "cold", "SHARED"),
    ]
    return facts


def _build(seed, use_spiking_bind=False, D=64):
    """A CommonGroundComposer with the de-risk corpus stored, each fact carrying its SHARED/PRIVATE tag."""
    rng = np.random.default_rng(seed)
    facts = _corpus(rng)
    cg = CommonGroundComposer(seed=seed, D=D, use_spiking_bind=use_spiking_bind)
    for (a, act, pt, tag) in facts:
        cg.store_cg(a, act, pt, common_ground=tag)
    return cg, facts


# ---------------------------------------------------------------------------------------------------------------
# Audience-design scoring. For each stored fact, the agent decides whether to TELL it (an informative contribution)
# given the listener model. CORRECT audience design = TELL private (listener doesn't know) + SUPPRESS shared
# (listener already knows). We score the fraction of facts handled correctly.
#   - REAL (tag-read):  reads the bound SHARED/PRIVATE tag -> tell iff PRIVATE.
#   - NO-TAG baseline:  cannot read a tag -> a fixed policy (tell-all OR suppress-all); the BEST it can do on a
#                       balanced 50/50 corpus is 0.5 (it gets one class right, the other wrong).
#   - PERMUTED:         the tag->fact assignment is shuffled -> the read tag is decorrelated from the truth.
#   - LESION:           the tag read is forced None (the role is severed) -> falls back to the no-tag policy.
# ---------------------------------------------------------------------------------------------------------------
def _audience_design_score(cg, facts, mode="real", perm=None, lesion=False):
    """Fraction of facts whose TELL/SUPPRESS decision matches correct audience design (tell private, suppress
    shared). `mode`: 'real' (read the bound tag), 'notag_tellall', 'notag_suppressall'. `perm`: a permutation of
    the stored-fact indices used to scramble which tag each fact reads (the permuted-tag control). `lesion`: force
    the tag read to None (severed role)."""
    n_correct = 0
    for i, (a, act, pt, true_tag) in enumerate(facts):
        # what tag does the agent READ for fact i (the decision input)?
        if mode == "real":
            if lesion:
                read_tag = None
            elif perm is not None:
                # read the tag of the PERMUTED fact (its cue roles) -- decorrelates tag from this fact's truth
                pa, pact, ppt, _ = facts[perm[i]]
                read_tag = cg.read_common_ground(pa, pact, ppt)
            else:
                read_tag = cg.read_common_ground(a, act, pt)
            tell = (read_tag == "PRIVATE")        # audience design: volunteer private, suppress shared
            if read_tag is None:                  # severed/lesioned -> no signal -> fall back to tell-all policy
                tell = True
        elif mode == "notag_tellall":
            tell = True
        elif mode == "notag_suppressall":
            tell = False
        else:
            raise ValueError(mode)
        correct = (tell == (true_tag == "PRIVATE"))   # correct iff (told a private) or (suppressed a shared)
        n_correct += int(correct)
    return n_correct / len(facts)


# ---------------------------------------------------------------------------------------------------------------
# The HARD-GATE de-risk.
# ---------------------------------------------------------------------------------------------------------------
def run_seed(seed, use_spiking_bind=False, D=64, verbose=False):
    cg, facts = _build(seed, use_spiking_bind=use_spiking_bind, D=D)
    rng = np.random.default_rng(seed + 991)

    real = _audience_design_score(cg, facts, mode="real")
    notag_tell = _audience_design_score(cg, facts, mode="notag_tellall")
    notag_supp = _audience_design_score(cg, facts, mode="notag_suppressall")
    notag_best = max(notag_tell, notag_supp)          # the best a tag-blind policy can do

    # permuted-tag control: average over several derangement-ish permutations
    perms = []
    for _ in range(8):
        p = rng.permutation(len(facts))
        perms.append(_audience_design_score(cg, facts, mode="real", perm=p))
    permuted = float(np.mean(perms))

    lesion = _audience_design_score(cg, facts, mode="real", lesion=True)

    # --- the no-confab MOAT: a never-stored fact must NOT read a tag (no fabricated tag) and a never-stored
    #     query must abstain. Also: the tag read for a stored fact must recover its TRUE tag (tag fidelity). ---
    # (a) tag fidelity on stored facts (does the bound tag read back correctly?)
    tag_correct = sum(1 for (a, act, pt, tg) in facts if cg.read_common_ground(a, act, pt) == tg)
    tag_fidelity = tag_correct / len(facts)
    # (b) TAG-fabrication moat: a held-out fact whose FULL SVO was never stored -> read_common_ground returns None
    #     (no fabricated tag). ('dog go cold' is a good test: the (dog,go) cue IS stored as 'dog go north', so the
    #     tag-read must NOT return a tag for the WRONG patient 'cold' -- the full-SVO match is what gates it.)
    held_out_tag = [("dog", "stop", "small"), ("cat", "come", "big"), ("dog", "go", "cold")]
    moat_fab = sum(1 for (a, act, pt) in held_out_tag if cg.read_common_ground(a, act, pt) is not None)
    # (c) QUERY-abstention moat: an (agent,action) cue absent from EVERY stored fact -> query_patient abstains.
    #     ('dog stop' + 'cat come' are the two (agent,action) pairs the corpus never uses -> must both abstain.)
    held_out_query = [("dog", "stop"), ("cat", "come")]
    moat_query_fab = sum(1 for (a, act) in held_out_query if cg.query_patient(a, act) is not None)

    if verbose:
        print(f"  [seed {seed}] audience-design real={real:.3f} notag_best={notag_best:.3f} "
              f"permuted={permuted:.3f} lesion={lesion:.3f} | tag_fidelity={tag_fidelity:.3f} "
              f"moat_fab={moat_fab} moat_query_fab={moat_query_fab}")

    return {
        "seed": seed,
        "real": real,
        "notag_best": notag_best,
        "permuted": permuted,
        "lesion": lesion,
        "tag_fidelity": tag_fidelity,
        "moat_fab": moat_fab,
        "moat_query_fab": moat_query_fab,
    }


def main():
    ap = argparse.ArgumentParser(description="Tier 2.4 common-ground HARD-GATE de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--spiking", action="store_true", help="run the tag bind/unbind through the real RF spiking bind")
    args = ap.parse_args()

    print("=" * 96)
    print("Tier 2.4 -- minimal common-ground (shared/private fact tag -> audience design) -- HARD-GATE de-risk")
    print(f"  seeds={args.seeds} D={args.D} spiking_bind={args.spiking}")
    print("  GATE: real >> no-tag-baseline (0.5 ceiling) AND permuted collapses AND lesion collapses AND moat 0-FA")
    print("=" * 96)

    rows = [run_seed(s, use_spiking_bind=args.spiking, D=args.D, verbose=True) for s in args.seeds]

    real = float(np.mean([r["real"] for r in rows]))
    notag = float(np.mean([r["notag_best"] for r in rows]))
    permuted = float(np.mean([r["permuted"] for r in rows]))
    lesion = float(np.mean([r["lesion"] for r in rows]))
    tag_fid = float(np.mean([r["tag_fidelity"] for r in rows]))
    moat_fab = sum(r["moat_fab"] for r in rows)
    moat_qfab = sum(r["moat_query_fab"] for r in rows)

    print("-" * 96)
    print(f"AGGREGATE ({len(rows)} seeds): real={real:.3f} notag_best={notag:.3f} (gap +{real - notag:.3f}) "
          f"permuted={permuted:.3f} lesion={lesion:.3f}")
    print(f"           tag_fidelity={tag_fid:.3f}  moat_fab={moat_fab}  moat_query_fab={moat_qfab}")

    # GATE conditions (all must hold)
    g_beats = real >= notag + 0.25 and real >= 0.85         # tracks the tag, well above the 0.5 no-tag ceiling
    g_permuted = permuted <= notag + 0.10                   # permuted collapses toward the tag-blind ceiling
    g_lesion = lesion <= notag + 0.10                       # lesion degrades to the no-tag baseline
    g_moat = (moat_fab == 0 and moat_qfab == 0)             # no fabricated tag / no fabricated answer
    g_fidelity = tag_fid >= 0.99                            # the bound tag reads back faithfully
    verdict = g_beats and g_permuted and g_lesion and g_moat and g_fidelity

    print("-" * 96)
    print(f"  GATE beats-no-tag (real>=notag+0.25 & >=0.85): {g_beats}")
    print(f"  GATE permuted-collapses (<=notag+0.10):        {g_permuted}")
    print(f"  GATE lesion-collapses (<=notag+0.10):          {g_lesion}")
    print(f"  GATE moat-0-FA (fab==0 & query_fab==0):        {g_moat}")
    print(f"  GATE tag-fidelity (>=0.99):                    {g_fidelity}")
    print("=" * 96)
    print(f"  VERDICT: {'GO' if verdict else 'NO-GO'}")
    print("=" * 96)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
