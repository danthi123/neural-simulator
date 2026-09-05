"""SCAFFOLD-RETIREMENT BACKLOG RANK-10 — the curiosity novelty signal was a binary host constant; retire it with
a genuine spiking familiarity/mismatch read (2026-09-05).

CONTEXT (verified against CURRENT code, not the backlog's abridged one-liner). The live production call
(`webapp/server.py::_curiosity_followup`, reached on every no-confab-moat ABSTAIN) invokes:

    j = _get_curiosity_organ().judge(novelty=_CU.NOVEL_SIGNAL, lesion=_CU.curiosity_lesioned())

`_CU.NOVEL_SIGNAL = 0.95` is a HOST CONSTANT — every abstain, on every topic, feeds the SAME "maximally novel"
scalar into the DR-1 `from_novelty` -> ASK-pool spiking drive. `curiosity_production_organ.py`'s own docstring
already names this as a declared residual: "NOVELTY = the ABSTAIN (a binary epistemic gap) ... a graded
familiarity-gate novelty (Bogacz-Brown) is the next rung." THE DE-RISKED START (the backlog's own phrase) this
builds on: DR-1 (2026-07-23-DR1-curiosity-inversion-6seed-GO.md / -ONBRIDGE-spiking.md) already proved the
`from_novelty` -> ASK-pool pathway is genuinely spiking + load-bearing (corr(gap,want)=+0.99, 6-seed GO) and
already reused the SAME Bogacz-Brown anti-Hebbian familiarity gate (`RealAntiHebbianFamiliarity`, catalog D.04)
as its novelty SOURCE — but only inside an isolated multi-concept ask-and-learn demo, never as the topic-specific
INPUT the live chat's `_curiosity_followup` actually uses. This runner closes that specific gap: a NEW
`TopicNoveltyGate` (added to `curiosity_production_organ.py`, reuse-by-import of `AntiHebbianFamiliarity` +
the genuine spike-phasor bind `phase_sum_neuron` the v320 gate's spiking realization already uses — NEITHER
reinvented) renders an arbitrary chat TOPIC word's cue and reads its novelty continuously, imprinted with the
brain's own known vocabulary (`_brain_vocab(chat)`, the comprehension organ's SAME source).

WHAT THIS RUNNER VALIDATES (all reuse-by-import; no sim/ edit — see `sim/` diff note at the bottom):
  (1) GRADED, not binary: known < noisy-partial(lo) < noisy-partial(hi) < wholly-unrelated, monotonically, with a
      real margin — replacing the single constant with a value that actually depends on the topic.
  (2) LOAD-BEARING on the REAL downstream decision: feeding each group's novelty into the SAME production
      `CuriosityProductionOrgan.judge()` (the genuinely-spiking on-bridge ASK-pool read, reuse-by-import,
      untouched) produces a correspondingly graded `want_hz`, and a DIFFERENT `curious` verdict across groups
      (known -> not curious; wholly-unrelated -> curious) — unlike the OLD constant, which reads `curious=True`
      on literally every abstain (want_novel_hz=126.6Hz > threshold=65.9Hz, calibrated at build, seed-invariant
      by construction). This is the actual retirement: the crave decision now depends on the topic.
  (3) LESION load-bearing: a PERMANENTLY-unimprinted twin (the production `lesion=True` semantics — never
      imprinted, matching `get_topic_gate(lesion=True)`) reads the SAME ceiling novelty (~1.0) for every group —
      the graded spread COLLAPSES, and the downstream want_hz/curious collapses to the old constant's
      undifferentiated always-curious behavior. The gradation rides the LEARNED weights, not an artifact.
  (4) PERMUTED control: imprinting a DISJOINT decoy vocabulary (never the actual "known" words), then querying
      the real "known" words against it, must make them read HIGH novelty (like "novel") — the low reading in
      the real arm is caused by the SPECIFIC imprint<->query correspondence, not incidental word-string shape.

HONEST SCOPE (declared, not hidden — see `curiosity_production_organ.py`'s own docstring on `TopicNoveltyGate`):
the word->phase code carries no lexical-semantic structure (a declared host boundary, like the v320 gate's
percept->phase projection); the validated gradation axis is CUE FIDELITY (clean vs. a noisy/partial draw of the
SAME word), not between-different-word semantic relatedness. The anti-Hebbian basis is capacity-bounded at 2*D
orthogonal directions (D=256 -> 512): this validates the MECHANISM at battery scale, not production-vocabulary
scale (the v320 gate's OWN "does this hold at V=320" question is a named next rung here, not re-litigated).

DE-RISK ONLY: additive, default-OFF (`BRAIN_CURIOSITY_GRADED_NOVELTY` unset), byte-identical when off (pinned by
`tests/test_curiosity_graded_novelty.py`, not this runner). Not flipped to production default.

Run: SIM_BACKEND=numpy python -m research.runners._curiosity_graded_novelty_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import lever  # noqa: E402

from research.runners.curiosity_production_organ import (  # noqa: E402
    TopicNoveltyGate, CuriosityProductionOrgan, NOVEL_SIGNAL,
)

N_KNOWN = 12
N_NOVEL = 12
N_DECOY = 12            # the permuted-control's disjoint imprint vocabulary
NOISE_LO = 0.03          # a mild perceptual jitter of a known word's own cue
NOISE_HI = 0.20          # a stronger jitter (still the SAME word's own code, not a different word)
STEP_EPS = 1e-4          # the four graded groups must be STRICTLY monotonic step-to-step (a real ordering, not
                         # a fixed per-step size -- the two noise levels are chosen points on a continuum, so
                         # only their RELATIVE order is claimed step-to-step; the magnitude claim is the full-
                         # range margins below)
NOVELTY_RANGE_MARGIN = 0.3    # required known->novel TOTAL novelty range (of a possible ~[0,1] span)
WANT_RANGE_MARGIN_HZ = 20.0   # required known->novel TOTAL want_hz range (the load-bearing downstream claim)
WANT_REPS = 12                # independent judge() repeats averaged per novelty value, denoising the ASK-pool's
                              # own OU-process trial noise (the SAME "repeated glances" pattern as the project's
                              # `novelty_settled(n=15)`) -- a single read is noisy at the finest novelty step
                              # (known vs a MILDLY noisy cue, a real but small 0.035 novelty gap)


def _mean(xs):
    return float(sum(xs) / len(xs))


def _judge_avg(organ: CuriosityProductionOrgan, novelty: float, reps: int = WANT_REPS) -> dict:
    """The organ's `judge()` averaged over `reps` independent reads (each already averages `N_READ_REPS` internal
    glances; this is a second, outer denoising pass over the ASK-pool's own OU-process trial variability at a
    FIXED novelty input). `curious` is recomputed from the AVERAGED want (more representative than any single
    noisy read) against the organ's own calibrated threshold."""
    reads = [organ.judge(novelty=novelty) for _ in range(reps)]
    want = _mean([r["want_hz"] for r in reads])
    threshold = float(reads[0]["threshold"])
    return {"want_hz": want, "threshold": threshold, "curious": bool(want >= threshold)}


def evaluate(seed: int, verbose: bool = False) -> dict:
    known = [f"knownword{i}" for i in range(N_KNOWN)]
    novel = [f"unrelatedword{i}" for i in range(N_NOVEL)]
    decoy = [f"decoyword{i}" for i in range(N_DECOY)]

    # ── REAL arm: the graded gate, imprinted with the ACTUAL known vocabulary ──────────────────────────────────
    real = TopicNoveltyGate(seed=seed)
    real.imprint_vocab(known)
    nov_known = [real.novelty(w) for w in known]
    nov_noisy_lo = [real.novelty(w, noise=NOISE_LO) for w in known]
    nov_noisy_hi = [real.novelty(w, noise=NOISE_HI) for w in known]
    nov_novel = [real.novelty(w) for w in novel]
    m_known, m_lo, m_hi, m_novel = _mean(nov_known), _mean(nov_noisy_lo), _mean(nov_noisy_hi), _mean(nov_novel)

    # ── LESION arm: the production `lesion=True` semantics — a twin NEVER imprinted ────────────────────────────
    lesioned = TopicNoveltyGate(seed=seed)   # never imprint_vocab()'d -> permanently the ceiling for everything
    les_known = _mean([lesioned.novelty(w) for w in known])
    les_lo = _mean([lesioned.novelty(w, noise=NOISE_LO) for w in known])
    les_hi = _mean([lesioned.novelty(w, noise=NOISE_HI) for w in known])
    les_novel = _mean([lesioned.novelty(w) for w in novel])
    les_spread = max(les_known, les_lo, les_hi, les_novel) - min(les_known, les_lo, les_hi, les_novel)
    real_spread = m_novel - m_known

    # ── PERMUTED control: imprint a DISJOINT decoy vocabulary, then query the REAL "known" words against it ────
    permuted = TopicNoveltyGate(seed=seed)
    permuted.imprint_vocab(decoy)
    perm_known = _mean([permuted.novelty(w) for w in known])   # "known" words, never actually imprinted here
    perm_decoy = _mean([permuted.novelty(w) for w in decoy])   # the words ACTUALLY imprinted in this arm

    # ── DOWNSTREAM load-bearing check: feed each REAL-arm group mean into a FRESH, this-seed's OWN production
    # ASK-pool organ (a genuine independent substrate build per seed -- NOT the module-level `get_organ()`
    # singleton, which is process-shared and ignores `seed` after its first build; production always calls it at
    # a fixed seed=42, so a real 6-seed check of the DOWNSTREAM spiking coupling needs 6 independent builds).
    # Each read is averaged over WANT_REPS independent judge() calls (see `_judge_avg`) to denoise the ASK-pool's
    # own OU-process trial variability, which otherwise swamps the SMALLEST novelty step (known vs. a mildly-
    # noisy cue, a real but small 0.035 gap).
    organ = CuriosityProductionOrgan(seed=seed)
    organ.ensure_built()
    j_known = _judge_avg(organ, m_known)
    j_lo = _judge_avg(organ, m_lo)
    j_hi = _judge_avg(organ, m_hi)
    j_novel = _judge_avg(organ, m_novel)
    j_old_constant = _judge_avg(organ, NOVEL_SIGNAL)   # the CURRENT production behavior, for contrast
    j_les = _judge_avg(organ, les_known)               # what a lesioned read would drive the ASK pool to

    want_known, want_lo, want_hi, want_novel = (j_known["want_hz"], j_lo["want_hz"], j_hi["want_hz"],
                                                 j_novel["want_hz"])

    if verbose:
        print(f"  [seed {seed}] REAL graded novelty: known={m_known:.4f} noisy_lo={m_lo:.4f} "
              f"noisy_hi={m_hi:.4f} novel={m_novel:.4f}")
        print(f"             REAL want_hz:        known={want_known:.2f} noisy_lo={want_lo:.2f} "
              f"noisy_hi={want_hi:.2f} novel={want_novel:.2f}  (threshold={j_known['threshold']:.2f})")
        print(f"             curious:             known={j_known['curious']} noisy_lo={j_lo['curious']} "
              f"noisy_hi={j_hi['curious']} novel={j_novel['curious']}  | OLD constant curious="
              f"{j_old_constant['curious']} (want={j_old_constant['want_hz']:.2f}, always True by calibration)")
        print(f"             LESIONED (never imprinted): known={les_known:.4f} noisy_lo={les_lo:.4f} "
              f"noisy_hi={les_hi:.4f} novel={les_novel:.4f}  spread={les_spread:.4f} (real spread={real_spread:.4f}) "
              f"-> collapsed want={j_les['want_hz']:.2f} curious={j_les['curious']}")
        print(f"             PERMUTED (decoy-imprinted): 'known' words now read {perm_known:.4f} "
              f"(vs {m_known:.4f} when actually imprinted) | actually-imprinted decoy reads {perm_decoy:.4f}")

    # ── GATES ───────────────────────────────────────────────────────────────────────────────────────────────
    # STEP-WISE: a real (strictly monotonic) ordering across the four points on the fidelity continuum. The two
    # noise levels are arbitrary points on a continuum, so only their RELATIVE order is asserted step-to-step;
    # the MAGNITUDE claim is the full-range margins below.
    gate_graded_order = (m_known + STEP_EPS < m_lo and m_lo + STEP_EPS < m_hi and m_hi + STEP_EPS < m_novel)
    gate_want_tracks = (want_known + STEP_EPS < want_lo and want_lo + STEP_EPS < want_hi
                        and want_hi + STEP_EPS < want_novel)
    # FULL-RANGE: known->novel must differ by a SUBSTANTIAL margin (not a noise-floor tremor).
    gate_novelty_range = (m_novel - m_known) >= NOVELTY_RANGE_MARGIN
    gate_want_range = (want_novel - want_known) >= WANT_RANGE_MARGIN_HZ
    # the retirement's OWN point: the graded read must produce a DIFFERENT curious verdict than the old constant
    # (which is curious=True on literally every abstain) for at least one group -- the decision now depends on topic.
    gate_discriminates_old_constant = (j_old_constant["curious"] is True) and (j_known["curious"] is False) \
        and (j_novel["curious"] is True)
    gate_lesion_collapses = (les_spread < STEP_EPS * 10) and (real_spread >= NOVELTY_RANGE_MARGIN)
    # a lesioned read should drive the ASK pool like the OLD undifferentiated constant (a ceiling-novelty read),
    # i.e. it reverts to "always curious," not to "never curious."
    gate_lesion_reverts_to_old_behavior = j_les["curious"] is True
    gate_permuted_collapses = (perm_known > m_novel - NOVELTY_RANGE_MARGIN) and \
        (perm_decoy < perm_known - NOVELTY_RANGE_MARGIN)

    lever("graded novelty spread (lesioned vs real, known-to-novel)", round(les_spread, 4), round(real_spread, 4))

    GO = bool(gate_graded_order and gate_want_tracks and gate_novelty_range and gate_want_range
              and gate_discriminates_old_constant and gate_lesion_collapses
              and gate_lesion_reverts_to_old_behavior and gate_permuted_collapses)

    return {
        "seed": seed,
        "real": {"novelty": {"known": m_known, "noisy_lo": m_lo, "noisy_hi": m_hi, "novel": m_novel},
                 "want_hz": {"known": want_known, "noisy_lo": want_lo, "noisy_hi": want_hi, "novel": want_novel},
                 "curious": {"known": j_known["curious"], "noisy_lo": j_lo["curious"],
                             "noisy_hi": j_hi["curious"], "novel": j_novel["curious"]},
                 "threshold_hz": j_known["threshold"]},
        "old_constant": {"novelty": NOVEL_SIGNAL, "want_hz": j_old_constant["want_hz"],
                         "curious": j_old_constant["curious"]},
        "lesioned": {"novelty": {"known": les_known, "noisy_lo": les_lo, "noisy_hi": les_hi, "novel": les_novel},
                     "spread": les_spread, "want_hz": j_les["want_hz"], "curious": j_les["curious"]},
        "permuted": {"known_against_decoy_gate": perm_known, "decoy_against_decoy_gate": perm_decoy},
        "real_spread": real_spread,
        "gates": {"graded_order": gate_graded_order, "want_tracks_novelty": gate_want_tracks,
                  "novelty_range": gate_novelty_range, "want_range": gate_want_range,
                  "discriminates_old_constant": gate_discriminates_old_constant,
                  "lesion_collapses": gate_lesion_collapses,
                  "lesion_reverts_to_old_behavior": gate_lesion_reverts_to_old_behavior,
                  "permuted_collapses": gate_permuted_collapses},
        "GO": GO,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--probe", action="store_true", help="verbose single-seed (seeds[0]) diagnostic dump")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = "research/findings/raw/_curiosity_graded_novelty_derisk.json"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from sim.backend import get_backend
    _, backend = get_backend()
    print(f"[RANK-10 CURIOSITY GRADED NOVELTY] backend={backend}\n"
          f"  GATES: (1) graded order known<noisy_lo<noisy_hi<novel  (2) want_hz tracks it on the REAL production "
          f"ASK-pool organ  (3) discriminates the OLD always-curious constant  (4) lesion (never-imprinted twin) "
          f"collapses the spread + reverts to the old always-curious behavior  (5) permuted-decoy control "
          f"collapses (a 'known' word not actually imprinted reads like 'novel')\n", flush=True)

    if a.probe:
        evaluate(a.seeds[0], verbose=True)
        return

    results = [evaluate(s, verbose=True) for s in a.seeds]
    n_go = sum(r["GO"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "backend": backend, "seeds": a.seeds}, fh, indent=2, default=str)
    print(f"{'='*104}", flush=True)
    print(f"  RANK-10 CURIOSITY GRADED NOVELTY: {n_go}/{len(results)} seeds GO "
          f"({'ALL GO' if n_go == len(results) else 'partial/negative'})", flush=True)
    print(f"  [saved] {a.out}\n{'='*104}", flush=True)


if __name__ == "__main__":
    main()
