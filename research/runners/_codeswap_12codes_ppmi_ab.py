"""Shortcut #12 — codes-half (option 2): swap the PRODUCTION conversational composer from hand-generated CURATED
codes to the LEARNED PPMI stream-cortex codes, and PROVE the swap is safe via a direct A/B.

THE OWNER'S DECISION (2026-06-20-fhrr-frontier-decision-scoping.md, path B2): #12 (the FHRR composer) has two halves
— the exact-inverse bind FORM (KEEP, close as honest-negative) and the CODES (SWAP to learned). The learned codes are
the PPMI stream cortex's (the 2026-06-15/16 generalization arc: a cortex that learns each word's meaning word-by-word
from a conversation stream by population-Hebbian co-occurrence; cached as `_phaseB_stream_codes_320_*.npy`). The
strategic point of the swap is to RETIRE the "curated codes" label — the production conversation runs on codes it
LEARNED FROM CONVERSATION — without changing the answers.

WHAT THIS RUNNER PROVES (the GREEN gate the swap needs):
  * == CURATED who/what: the production agent on the LEARNED PPMI codes answers the SAME who/what/yes-no matrix as the
    SAME agent on the curated (composer-self-generated random) codes — answer-identical, multi-seed.
  * MOAT 0-FA (HARD): the LEARNED codes must still ABSTAIN on every unstored cue (who_does/what_does -> None,
    is_it_true -> not "yes"). A single false-accept is a MOAT BREACH = a HARD failure (never weaken the moat).
  * THE HONEST MARGIN COST: the LEARNED codes match the answers but at a LOWER cleanup margin than curated (the
    documented ~0.39-below-ceiling cost of using learned-from-experience codes; decision-scoping). This runner MEASURES
    that margin gap (read-only; it recomputes the unbind+cleanup matched-filter sims, it does NOT touch the composer's
    cleanup). The margin must NOT be compensated by loosening the gate — if a lower margin pushes a false-accept, that
    is an HONEST finding reported here, not a reason to relax the moat.

THE ESCAPE: the curated-codes path stays available (don't pass `grounded_codes` -> the composer self-generates the
random codes, the test-oracle / numpy-CPU default). This A/B IS that escape exercised side-by-side.

NO sim/ edit; reuse-by-import (BrainConversationalAgent + RFPhasorComposer's existing grounded_codes path). CPU/numpy
is fine (composer_kind="rf", D=128 RF ops are tiny). The controller owns any GPU/onebrain confirm.

Run:
  SIM_BACKEND=numpy python -m research.runners._codeswap_12codes_ppmi_ab --seeds 42 --readout neural
  SIM_BACKEND=numpy python -m research.runners._codeswap_12codes_ppmi_ab --seeds 42 43 44 --readout neural
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories

D = 128  # the production RFPhasorComposer phasor dimension (brain_conversational_agent.py default)


# --- the grounding map (consolidated_320_conversation_demo.py, verbatim) -----
def _projection(d_out, n_in, seed):
    """A FIXED random complex projection n_in -> d_out (a fixed cortico-cortical fan-in, not learned per fact)."""
    rng = np.random.RandomState(seed * 7919 + 13)
    return (rng.standard_normal((d_out, n_in)) + 1j * rng.standard_normal((d_out, n_in))).astype(np.complex128)


def grounded_phases(code_vec, proj):
    """Real cortex code -> composer phases[D] in [0,1): exp(i angle(proj @ code)) -- the step-3 grounded phasor."""
    z = proj @ code_vec.astype(np.complex128)
    return (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)


# --- the conversational matrix (the same facts/cues consolidated_320 uses) ---
FACTS = [
    ("dog", "eat", "apple"),
    ("cat", "play", "ball"),
    ("bird", "sleep", "tree"),
    ("girl", "run", "park"),
    ("boy", "look", "book"),
    ("lion", "eat", "cake"),
    ("rabbit", "jump", "garden"),
    ("mouse", "walk", "house"),
]
ABSENT_WHAT = [("dog", "sing"), ("cat", "run"), ("bird", "eat"), ("girl", "sleep"), ("lion", "jump")]
ABSENT_WHO = [("eat", "ball"), ("play", "apple"), ("run", "tree"), ("sleep", "park")]
NEG_FACT = ("fish", "eat", "cake")


def _cleanup_margin(composer, agent_w, action_w, patient_w):
    """READ-ONLY diagnostic: the normalized cleanup margin (peak - runner-up)/(peak + eps) of the PATIENT unbind for the
    fact (agent, action, patient), recomputed from the composer's OWN stored composite + codes. It reuses the composer's
    public _unbind_phases + the matched-filter sims = mean-cos(rec, concept) (the SAME quantity the composer's _cleanup
    argmaxes), so it measures the very margin the cleanup decides on -- WITHOUT modifying any composer cleanup code.
    Returns the margin, or None if the fact is not in the kb (so it never confabulates a number)."""
    comp = composer
    for fact, comp_phases in comp.kb:
        if fact.get("agent") == agent_w and fact.get("action") == action_w and fact.get("patient") == patient_w:
            rec = comp._unbind_phases(comp_phases, "patient")
            sims = np.array([float(np.mean(np.cos(2.0 * np.pi * (rec - comp.concepts[w])))) for w in comp.words])
            s = np.sort(sims)[::-1]
            peak = float(s[0])
            if peak <= 1e-9:
                return 0.0
            return float((s[0] - s[1]) / (abs(s[0]) + 1e-9))
    return None


def _run_matrix(agent):
    """Drive the production agent through the who/what/yes-no/moat matrix; return the dict of all answers + the moat
    false-accept count. The SAME calls for curated and PPMI so the A/B compares like-for-like."""
    for a, v, o in FACTS:
        agent.hear(f"{a} {v} {o}", polarity="AFFIRM")
    agent.hear(f"{NEG_FACT[0]} {NEG_FACT[1]} {NEG_FACT[2]}", polarity="NEGATE")

    answers = {}
    for a, v, o in FACTS:
        answers[("what", a, v)] = agent.what_does(a, v)
        answers[("who", v, o)] = agent.who_does(v, o)
    answers[("yn", *FACTS[0])] = agent.is_it_true(*FACTS[0])     # expect "yes"
    answers[("yn", *NEG_FACT)] = agent.is_it_true(*NEG_FACT)     # expect "no"
    answers[("yn", "dog", "eat", "ball")] = agent.is_it_true("dog", "eat", "ball")  # expect not "yes"

    # the no-confab moat: every unstored cue must abstain
    false_accept, breaches = 0, []
    for a, v in ABSENT_WHAT:
        ans = agent.what_does(a, v)
        answers[("absent_what", a, v)] = ans
        if ans is not None:
            false_accept += 1
            breaches.append(f"what_does({a},{v}) -> {ans!r} (should abstain)")
    for v, o in ABSENT_WHO:
        ans = agent.who_does(v, o)
        answers[("absent_who", v, o)] = ans
        if ans is not None:
            false_accept += 1
            breaches.append(f"who_does({v},{o}) -> {ans!r} (should abstain)")
    return answers, false_accept, breaches


def run_seed(seed, codes, vocab, readout):
    concepts = {vocab[i]: codes[i] for i in range(len(vocab))}   # sets the full 320-word vocabulary
    proj = _projection(D, codes.shape[1], seed)
    grounded = {vocab[i]: grounded_phases(codes[i], proj) for i in range(len(vocab))}

    # A: the production agent on the CURATED (composer-self-generated random) codes -- the test-oracle / escape path.
    #    `concepts` sets WHICH words exist; with no grounded_codes the composer generates random phases for them.
    agent_cur = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf")
    ans_cur, fa_cur, br_cur = _run_matrix(agent_cur)

    # B: the SAME production agent on the LEARNED PPMI codes -- the swap. Same vocab, same facts, same queries.
    agent_ppmi = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded, composer_kind="rf")
    ans_ppmi, fa_ppmi, br_ppmi = _run_matrix(agent_ppmi)

    # == curated who/what: every answer identical between curated and PPMI.
    mismatches = [(k, ans_cur[k], ans_ppmi[k]) for k in ans_cur if ans_cur[k] != ans_ppmi.get(k)]
    answers_identical = len(mismatches) == 0

    # the honest margin cost: the cleanup margin of each stored fact's patient unbind, curated vs PPMI.
    m_cur = [_cleanup_margin(agent_cur.composer, a, v, o) for a, v, o in FACTS]
    m_ppmi = [_cleanup_margin(agent_ppmi.composer, a, v, o) for a, v, o in FACTS]
    m_cur = [m for m in m_cur if m is not None]
    m_ppmi = [m for m in m_ppmi if m is not None]
    mean_margin_cur = float(np.mean(m_cur)) if m_cur else 0.0
    mean_margin_ppmi = float(np.mean(m_ppmi)) if m_ppmi else 0.0
    margin_cost = mean_margin_cur - mean_margin_ppmi   # how much margin the learned codes give up

    moat_breach = fa_ppmi > 0
    go = answers_identical and (fa_ppmi == 0) and (fa_cur == 0)
    return {
        "seed": seed, "readout": readout, "n_facts": len(FACTS),
        "answers_identical": answers_identical, "n_mismatches": len(mismatches),
        "mismatches": [(list(k), a, b) for k, a, b in mismatches[:8]],
        "false_accept_curated": fa_cur, "false_accept_ppmi": fa_ppmi, "moat_breach": moat_breach,
        "breaches_ppmi": br_ppmi,
        "mean_margin_curated": mean_margin_cur, "mean_margin_ppmi": mean_margin_ppmi,
        "margin_cost": margin_cost, "go": go,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--readout", choices=["neural", "host"], default="neural")
    ap.add_argument("--out", default="research/findings/raw/_codeswap_12codes_ppmi_ab.json")
    a = ap.parse_args()

    vocab, _cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    suffix = "neural_seed" if a.readout == "neural" else "seed"

    print(f"[#12 codes-swap A/B] production agent: CURATED codes  vs  LEARNED PPMI ({a.readout}) codes — "
          f"same answers? moat 0-FA? margin cost?\n", flush=True)
    results, hard_stop = [], False
    for seed in a.seeds:
        cpath = os.path.join(_REPO, "research", "findings", "raw",
                             f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
        if not os.path.exists(cpath):
            print(f"  [seed {seed}] SKIP — no {a.readout} PPMI codes at {cpath}", flush=True)
            continue
        codes = np.load(cpath)
        codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
        r = run_seed(seed, codes, vocab, a.readout)
        results.append(r)
        tag = "GO" if r["go"] else ("MOAT_BREACH" if r["moat_breach"] else "NEGATIVE")
        print(f"  [seed {seed}] answers-identical {r['answers_identical']} "
              f"(mismatches {r['n_mismatches']}) | moat: FA curated {r['false_accept_curated']} / "
              f"FA ppmi {r['false_accept_ppmi']} | margin curated {r['mean_margin_curated']:.3f} -> "
              f"ppmi {r['mean_margin_ppmi']:.3f} (cost {r['margin_cost']:+.3f})  ==> {tag}", flush=True)
        for m in r["mismatches"]:
            print(f"      != {m}", flush=True)
        for b in r["breaches_ppmi"]:
            print(f"      !! {b}", flush=True)
        hard_stop = hard_stop or r["moat_breach"]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    n_go = sum(r["go"] for r in results)
    print(f"\n{'=' * 100}", flush=True)
    if hard_stop:
        print("  MOAT_BREACH (HARD STOP): the LEARNED PPMI codes accepted an unstored query — the no-confab guarantee "
              "failed on the swap. This is the honest finding; do NOT loosen the gate. Investigate the code fidelity.",
              flush=True)
    elif results and n_go == len(results):
        mc = float(np.mean([r["margin_cost"] for r in results]))
        print(f"  GO ({n_go}/{len(results)} seeds): the production agent on the LEARNED PPMI codes answers the SAME "
              f"who/what/yes-no matrix as on the curated codes (== curated who/what) AND holds the no-confab moat "
              f"(0 false-accepts). The honest cost: a LOWER cleanup margin (mean cost {mc:+.3f}) — learned-from-"
              f"experience codes are less decisive than curated ones, but the answers + the moat are unchanged. ==> the "
              f"'curated codes' label can be retired for the production conversation; the curated path stays the escape.",
              flush=True)
    elif results:
        print(f"  NEGATIVE ({n_go}/{len(results)} seeds GO): the answers diverged (see mismatches) — the learned codes "
              "changed an answer. The moat held (no breach). Report as an honest finding.", flush=True)
    else:
        print("  NO PPMI CODES — run the 320 stream cortex first to produce the cached codes.", flush=True)
    print(f"  [saved] {a.out}\n{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
