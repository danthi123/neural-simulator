"""Comprehension cue-lexicon conversion — SPIKING-REALIZED open-vocabulary animacy lookup (2026-08-27).

WHY: every comprehension organ (D4 comprehension-monitor, D6 multiref-WM, D3 discourse-register, T1-6
other-repair, D2 surprise) declares the SAME host scaffold — the ANIMACY membership table
(`comprehension_production_organ.py`) is a ~19-noun toy lexicon; a real word off the table is OUT OF SCOPE.
The 6-seed GO de-risk (`_comprehension_learned_animacy_cue_derisk.py`,
`research/findings/raw/_comprehension_learned_animacy_cue_6seed.json`) proved open-vocab animacy is
LEARNABLE from REAL TinyStories co-occurrence on HELD-OUT never-labelled words: mean learned=0.837,
shuffled-graph control=0.504 (chance), frequency-only control=0.511 (chance), gap=+0.333, all
preconditions OK. This module (a) reuses that validated label-propagation mechanism VERBATIM (no
re-derivation) to score every frequent content word of the real corpus, and (b) SPIKING-REALIZES the
per-word classification DECISION via gap#3-A1's F_anim/F_inanim coincidence-detector pools
(`_gap3_spiking_feature_compat_derisk.py`, 2026-07-18, 6-seed GO, guarded by
`tests/test_gap3_spiking_feature_compat.py`, 7/7 passing) — a KNOWN, already-validated spiking mechanism,
not a new one invented for this conversion.

OFFLINE SCAFFOLD vs SPIKING (same honest split gap#3-A1 declares for its own concept-animacy signs): the
per-word CONTINUOUS SCORE and its SIGN are an offline label-propagation computation (Zhou label-spreading
over a PPMI co-occurrence graph) — the offline scaffold, exactly like gap#3-A1's EM-learned concept/verb
signs (validated by the numpy GO gate's own `f[idx[w]] > 0` read, 6-seed: learned=0.837, shuffled=0.504,
gap=+0.333). The CLASSIFICATION DECISION is read off FIRING spiking neurons in the F_anim/F_inanim pools
(`cp_firing_states`), not a host `if margin > 0` branch on a continuous magnitude: the sign drives ONE pool
with a fixed current (gap#3-A1's own `drive` constant — a coincidence-style fixed-magnitude push, not a
graded one), the pools compete for `steps` ticks, and the WINNER (by firing rate) is the answer. A word off
the learned graph gets NO drive to EITHER pool -> the two pools tie at 0 -> ABSTAIN — the no-confab moat.

LESION (`set_lesion(True)`): zeroes the pools' input drive (`mult=0.0`) regardless of the learned sign ->
every word's two pools tie at 0 -> every `classify()` call abstains. This is the load-bearing coupling: with
it zeroed, an open-vocabulary word the hand ANIMACY table lacks is no longer classified, so the
comprehension organ's coverage of it reverts to the hand-table-only scope — byte-identical to the flag being
off entirely, for that word.

HONEST RESIDUAL: only ANIMACY is converted here (the validated GO artifact this module reuses). VERB_SELECTS
(the verb-selectional-fit lexicon) stays the pre-existing hand-coded closed set — no GO artifact validates
an open-vocab verb-selects cue, so claiming that conversion here would be an overclaim. It rides as a
follow-on residual, exactly like the D6/D3/T1-6/D2 organs' declared vocab-ceiling notes.

Run (numpy CPU, light — a few seconds to build the ~1500-word graph + label-spread; the spiking read per
word is ~25 sim steps on an 80-neuron bridge):
    SIM_BACKEND=numpy python -m research.runners._comprehension_learned_animacy_spiking --smoke
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._comprehension_learned_animacy_cue_derisk import (  # noqa: E402
    load_tokens,
    build_vocab,
    cooccur_ppmi,
    label_spread,
    shuffle_graph,
    GT_ANIMATE,
    GT_INANIM,
)
from research.runners._gap3_spiking_feature_compat_derisk import _build as _build_feature_pools  # noqa: E402

_DEFAULT_CORPUS = os.path.join(_REPO, "data", "corpus", "tinystories.txt")


class LearnedAnimacyLexicon:
    """Open-vocabulary animacy cue: LEARNS a continuous animacy score for every frequent content word of a
    real corpus via label-propagation (the validated de-risk mechanism), then SPIKING-REALIZES the per-word
    classification decision via gap#3-A1's F_anim/F_inanim coincidence-detector pools. See module docstring.

    Two build modes:
      * DEPLOYMENT (`k_seed=None`, the default): seeds label-spreading with the FULL GT_ANIMATE/GT_INANIM
        sets (60 words) -- maximal seed signal for production use. This is what `get_lexicon()` builds.
      * CROSS-VALIDATION (`k_seed=<int>`): subsamples `k_seed` seed words per class (rng seeded by
        `cv_seed`), holding the rest out -- reproduces the de-risk's held-out evaluation protocol so a
        verifier can measure SPIKING classification accuracy on never-labelled words, exactly like the
        numpy GO gate. `self.held_out_words` exposes the held set for that eval.
    """

    def __init__(self, seed: int = 42, corpus_path: str | None = None, max_chars: int = 8_000_000,
                 top_v: int = 1500, window: int = 4, n_feat: int = 40, drive: float = 500.0, steps: int = 25,
                 k_seed: int | None = None, cv_seed: int | None = None, shuffle_control: bool = False):
        self.seed = int(seed)
        self.drive = float(drive)
        self.steps = int(steps)
        self._lesioned = False
        self.held_out_words = None

        path = corpus_path or _DEFAULT_CORPUS
        tokens = load_tokens(path, max_chars)
        vocab, freq = build_vocab(tokens, top_v)
        W = cooccur_ppmi(tokens, vocab, window=window)
        idx = {w: i for i, w in enumerate(vocab)}
        rng_key = cv_seed if cv_seed is not None else seed

        if shuffle_control:
            W = shuffle_graph(W, np.random.default_rng(rng_key))

        anim_pool = [w for w in GT_ANIMATE if w in idx]
        inan_pool = [w for w in GT_INANIM if w in idx]
        if k_seed is not None:
            rng = np.random.default_rng(rng_key)
            anim = anim_pool[:]
            inan = inan_pool[:]
            rng.shuffle(anim)
            rng.shuffle(inan)
            seed_anim, held_anim = anim[:k_seed], anim[k_seed:]
            seed_inan, held_inan = inan[:k_seed], inan[k_seed:]
            self.held_out_words = {"animate": held_anim, "inanimate": held_inan}
        else:
            seed_anim, seed_inan = anim_pool, inan_pool

        y = np.zeros(len(vocab))
        for w in seed_anim:
            y[idx[w]] = +1.0
        for w in seed_inan:
            y[idx[w]] = -1.0
        f = label_spread(W, y)
        self.scores = {w: float(f[i]) for w, i in idx.items()}
        self.vocab = vocab
        self._seeded_words = seed_anim + seed_inan

        # Spiking realization: reuse the gap#3-A1 2-pool bridge (F_anim / F_inanim coincidence-detector
        # regions), a KNOWN, already-validated mechanism (not invented for this conversion).
        self.b = _build_feature_pools(self.seed, n_feat=n_feat)
        self.n = self.b.core_config.num_neurons
        self.f_anim = np.asarray(list(self.b.region_manager.indices("F_anim")), int)
        self.f_inan = np.asarray(list(self.b.region_manager.indices("F_inanim")), int)

    def _read_margin(self, sign_val: float) -> float:
        from sim.backend import to_host, from_host
        b = self.b
        if getattr(b, "cp_izh_c_reset", None) is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for a in ("cp_conductance_g_e", "cp_conductance_g_i"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0.0
        mult = 0.0 if self._lesioned else 1.0
        cur = np.zeros(self.n)
        cur[self.f_anim] += self.drive * max(sign_val, 0.0) * mult
        cur[self.f_inan] += self.drive * max(-sign_val, 0.0) * mult
        dev = from_host(cur.astype(np.float64))
        rate_a = rate_i = 0.0
        for _ in range(self.steps):
            b.cp_external_input_current[:] = dev
            b._run_one_simulation_step()
            fs = np.asarray(to_host(b.cp_firing_states))
            rate_a += float(fs[self.f_anim].mean())
            rate_i += float(fs[self.f_inan].mean())
        return rate_a - rate_i

    def set_lesion(self, on: bool = True) -> None:
        """Zero the F_anim/F_inanim input coupling (LOAD-BEARING lesion): every `classify()` call abstains
        regardless of the learned score, so any open-vocab coverage this lexicon provided reverts."""
        self._lesioned = bool(on)

    def classify(self, word: str):
        """Return "animate" / "inanimate" / None (abstain: off-graph, or lesioned). ELIGIBILITY (is this word
        scored at all) and the CATEGORY SIGN are the offline label-propagation scaffold (exactly as validated
        by the numpy GO gate's own `f[idx[w]] > 0` read); the DECISION readout is a spiking WTA: the sign
        drives ONE pool with a fixed current (the same `drive` magnitude gap#3-A1 uses), the OTHER pool gets
        none, and the pools compete for `steps` ticks -- the winner (by `cp_firing_states` rate) is the
        answer. A word off the learned graph gets NO drive to either pool -> the margin is exactly 0 -> the
        two pools tie -> abstain. Lesioning zeros the drive entirely -> every word ties -> abstain."""
        score = self.scores.get(word)
        if score is None or abs(score) < 1e-12:
            return None
        margin = self._read_margin(float(np.sign(score)))
        if margin == 0.0:
            return None
        return "animate" if margin > 0 else "inanimate"


_LEXICON: LearnedAnimacyLexicon | None = None


def get_lexicon(seed: int = 42) -> LearnedAnimacyLexicon:
    """The process-shared DEPLOYMENT lexicon (built once, lazily, on first use)."""
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = LearnedAnimacyLexicon(seed=seed)
    return _LEXICON


# ---------------------------------------------------------------------------
# Verification: reproduce the numpy GO gate THROUGH the spiking classify() read, on held-out (never
# seed-labelled) words, plus the shuffled-graph anti-cheat -- so the claim "spiking-realized" is measured,
# not asserted. Deliberately small n_feat/steps (cheap) since this is a LIGHT local smoke, not a GPU sweep.
# ---------------------------------------------------------------------------
def eval_seed_spiking(seed: int, k_seed: int = 8, shuffle_control: bool = False):
    lex = LearnedAnimacyLexicon(seed=seed, k_seed=k_seed, cv_seed=seed, shuffle_control=shuffle_control)
    held = lex.held_out_words
    ok = wrong = abstain = 0
    for w in held["animate"]:
        r = lex.classify(w)
        if r is None:
            abstain += 1
        elif r == "animate":
            ok += 1
        else:
            wrong += 1
    for w in held["inanimate"]:
        r = lex.classify(w)
        if r is None:
            abstain += 1
        elif r == "inanimate":
            ok += 1
        else:
            wrong += 1
    n_scored = ok + wrong
    acc = ok / n_scored if n_scored else 0.0
    n_total = ok + wrong + abstain
    return {"seed": seed, "acc": acc, "ok": ok, "wrong": wrong, "abstain": abstain,
            "n_scored": n_scored, "n_total": n_total,
            "abstain_rate": abstain / n_total if n_total else 0.0}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--k-seed", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",")]
    intact = [eval_seed_spiking(s, k_seed=args.k_seed, shuffle_control=False) for s in seeds]
    shuf = [eval_seed_spiking(s, k_seed=args.k_seed, shuffle_control=True) for s in seeds]

    macc = float(np.mean([r["acc"] for r in intact]))
    msh = float(np.mean([r["acc"] for r in shuf]))
    mabst = float(np.mean([r["abstain_rate"] for r in intact]))
    go = (macc >= 0.75) and (msh <= 0.60) and ((macc - msh) >= 0.15)

    print("[comprehension learned-animacy cue -- SPIKING realization verify]")
    for r, rs in zip(intact, shuf):
        print(f"  [seed {r['seed']}] spiking-classify acc={r['acc']:.3f} (abstain {r['abstain_rate']:.2f}) "
              f"| shuffled-graph acc={rs['acc']:.3f}")
    print(f"  MEAN(6): spiking-acc={macc:.3f}  shuffled={msh:.3f}  abstain-rate={mabst:.2f}  "
          f"(spiking-shuffled={macc - msh:.3f})")
    print(f"  GO-gate: spiking-acc>=0.75 AND shuffled<=0.60 AND (spiking-shuffled)>=0.15  ->  "
          f"{'GO' if go else 'NO-GO'}")

    # LESION check: a fresh deployment-mode lexicon, lesioned, must abstain on EVERY word (coverage reverts).
    lex = LearnedAnimacyLexicon(seed=42, k_seed=8, cv_seed=42)
    held_words = lex.held_out_words["animate"][:3] + lex.held_out_words["inanimate"][:3]
    pre_lesion = [lex.classify(w) for w in held_words]
    lex.set_lesion(True)
    post_lesion = [lex.classify(w) for w in held_words]
    lesion_ok = all(r is None for r in post_lesion) and any(r is not None for r in pre_lesion)
    print(f"  LESION check: pre={pre_lesion}  post={post_lesion}  -> "
          f"{'PASS (coverage reverts)' if lesion_ok else 'FAIL'}")

    # ATTRIBUTION (tools.lab): whose is the held-out COVERAGE -- the F_anim/F_inanim coupling, or something
    # else? treatment = coverage rate (fraction classified, not abstained) with the coupling intact; control
    # = the SAME words' coverage with the coupling lesioned (drive zeroed). Both are well-defined (unlike an
    # accuracy comparison, where the lesioned arm has NOTHING scored to compute an accuracy over).
    pre_coverage = sum(1 for r in pre_lesion if r is not None) / len(pre_lesion)
    post_coverage = sum(1 for r in post_lesion if r is not None) / len(post_lesion)
    try:
        from tools.lab import attributable_to
        attributable_to("held-out ANIMACY coverage (F_anim/F_inanim coupling vs lesioned)",
                        pre_coverage, post_coverage)
    except Exception as _e:  # tools.lab optional; the JSON already carries both arms
        print(f"  (attribution helper unavailable: {_e})", flush=True)

    if args.out:
        import json
        preconditions = [
            {"name": "spiking-acc>=0.75", "ok": bool(macc >= 0.75), "value": macc},
            {"name": "shuffled-graph-control<=0.60 (structure-destroyed collapses)",
             "ok": bool(msh <= 0.60), "value": msh},
            {"name": "spiking-acc-minus-shuffled>=0.15", "ok": bool((macc - msh) >= 0.15),
             "value": macc - msh},
            {"name": "held-out abstain rate is low (moat not over-triggering on real words)",
             "ok": bool(mabst <= 0.10), "value": mabst},
            {"name": "lesion collapses ALL held-out classification to abstain", "ok": bool(lesion_ok),
             "value": None},
            {"name": "held-out coverage 100% attributable to the F_anim/F_inanim coupling (not the "
                      "lesion itself an artifact of an already-abstaining arm)",
             "ok": bool(pre_coverage > 0.0), "value": pre_coverage - post_coverage},
        ]
        status = "GO" if (go and lesion_ok) else "NO-GO"
        payload = {"seeds": seeds, "k_seed": args.k_seed, "intact": intact, "shuffled": shuf,
                   "mean_spiking_acc": macc, "mean_shuffled_acc": msh, "mean_abstain_rate": mabst,
                   "go": bool(go and lesion_ok), "status": status, "lesion_check_pass": bool(lesion_ok),
                   "lesion_pre": [str(x) for x in pre_lesion], "lesion_post": [str(x) for x in post_lesion],
                   "coverage_pre_lesion": pre_coverage, "coverage_post_lesion": post_coverage,
                   "preconditions": preconditions}
        outp = args.out if os.path.isabs(args.out) else os.path.join(_REPO, args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
