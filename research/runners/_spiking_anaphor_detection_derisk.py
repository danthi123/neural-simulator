"""SPIKING CLOSED-CLASS (pronoun) RECOGNITION via CA3-style pattern completion — scaffold-retirement target:
`research/runners/brain_chat_tui.py::ChatBrain._resolve_anaphora` and its twin gate in
`research/runners/multi_turn_agent.py::MultiTurnAgent._resolve` DETECT whether the CURRENT token is an
anaphoric pronoun with a bare host Python set-membership test --

    anaphors = {"it", "that", "they", "them", "this"}          # brain_chat_tui.py::_resolve_anaphora
    if not (isinstance(word, str) and word.lower() in _ANAPHORS): return word   # multi_turn_agent.py::_resolve

-- BEFORE handing off to the substrate's own ALREADY-SPIKING referent RESOLUTION (`held_referent()` / the
WTA biased-competition read in `multi_turn_agent.py::_resolve_biased`). `docs/PRODUCTION_INTEGRATION_LEDGER.
yaml`'s own "anaphora-wm" row ("the 2nd default-on-spiking [organ]") names exactly this residual verbatim:
"host pronoun detect/substitute around it" (`retire_status: BLOCKED:neural-render`). Rank-17 of the project's
own 24-item scaffold-retirement backlog (`research/coordination/scaffold_retirement_backlog.md`, line 50:
"17 pronoun detect/substitute host (MED · fresh)") is untouched by any prior de-risk or flip -- distinct
from rank-2's cue-match for-loop (active elsewhere this session), rank-4/5's salience/appraisal afferents
(already flipped default-on), and rank-10's novelty->habituation build (landed the same day as this file,
commit 755c17290) -- a genuinely different host shortcut, in a genuinely different faculty.

THE BET. A literal Python `in {...}` test is exact-match-or-nothing: any corrupted/partial/noisy rendering
of a stored word (a dropped character, a garbled ASR token, a degraded perceptual cue) fails CLOSED --
the word silently stops being treated as a pronoun and the whole downstream WM-resolution path never fires.
Biology does not gate memory retrieval on an exact string match. Kandel PNS 6e (the same source already
anchoring this project's `research/biology/dg-ca3-sparse-index.md`): "A key feature of explicit memory is
that a few cues are often sufficient to retrieve a complex stored memory" -- Marr's CA3 proposal that "the
reactivation of a subset of this stored cell assembly would be sufficient to activate the entire original
neural ensemble," a restoration "referred to as pattern completion." This de-risk asks whether treating
"is this word one of my known closed-class pronouns" as a CA3-style autoassociative pattern-completion
decision -- reading a threshold off `cp_firing_states`, not a Python `in` test -- reproduces the host set's
exact-match behavior on clean input AND surpasses it on degraded/noisy input, which the host set structurally
cannot do (there is no partial credit in `x in {"it", ...}`).

MECHANISM. Reuse-by-import, NO `sim/` edit: `research.runners.content_selection_spiking.
SpikingLoopContextBuffer`, the project's own validated (multi-concept-WM GO, 220x specificity; see that
module's own docstring/findings) cortico-PFC loop (`cortex_ctx` <-> `dlpfc_wm`, both NMDA-bistable
IZH2007_HIPPO_PYRAMIDAL neurons) with ONE pattern-specific attractor per closed-class word, installed as a
Hebbian outer-product connection (`attractor_weight`) exactly matching Marr's "stored as changes in
connections between active CA3 cells." Each anaphor gets its OWN `pattern_size`-neuron assembly, partitioned
disjointly out of the `cortex_ctx`/`dlpfc_wm` populations (the buffer's own permutation-based pattern
allocation); the REMAINING unused neurons (n - K*pattern_size) carry no stored assembly at all -- exactly
the "unfamiliar word" condition content words should read as.

A CANDIDATE WORD is turned into a CUE (a drive pattern over `cortex_ctx` input neurons), not looked up in a
dict:
  * a known anaphor, CLEAN cue: its own full pattern_size-neuron assembly (drive_pA into every neuron).
  * a known anaphor, NOISY/PARTIAL cue: `keep_frac` of its true assembly neurons + the rest replaced with
    RANDOM unused-region neurons -- a corrupted/degraded rendering of the same word.
  * a content (open-class) word: `pattern_size` neurons drawn ENTIRELY from the unused region -- a word with
    no learned closed-class assembly at all (the true negative condition).
The DECISION reads `buf.read()` (per-concept mean firing rate over a silent window AFTER the cue) and
classifies "pronoun" iff the max per-concept rate clears a fixed ignition threshold `THETA`, ELSE "content
word" -- zero Python `in`/dict-membership calls anywhere on this path (G5, checked structurally below).

GATE (pre-registered, 6 project-standard seeds [42,43,44,100,101,102], numpy-CPU, NO sim/ edit):
  G1 clean-cue recall is real:        per-seed mean accuracy over the K clean-cue probes           >= 0.90
  G2 noisy/partial-cue completion
     SURPASSES exact-match (the deliverable): per-seed mean accuracy over K*N_NOISY_REPEATS
     noisy-cue probes, each cue only KEEP_FRAC=20% the true assembly (80% corrupted -- a rendering
     an exact host string match CANNOT recognize AT ALL, let alone recover from)                    >= 0.85
  G3 specificity (no false pronoun alarms on content words): per-seed false-positive rate over
     N_CONTENT content-word probes (max concept rate clearing THETA)                                <= 0.15
  G4 UNTRAINED-attractor lesion collapses completion, AND the intact-vs-lesion gap is
     load-bearing: lesioned (attractor_weight=0) mean accuracy over the SAME clean+noisy probes     <= 0.30
     AND attributable_to(intact_accuracy, lesioned_accuracy) >= 0.5
  G5 (structural, not per-seed): the decision function performs 0 Python set/dict/tuple membership
     checks against a literal closed-class word list (asserted once via source inspection).
Per-seed GO = G1 and G2 and G3 and G4. Board GO = >=5/6 seeds (project convention).

SCOPE (focused, single-faculty, NOT wired to production this session). This validates the DETECTION
mechanism only -- whether a candidate token is anaphoric. The RESOLUTION (which held referent "it" refers
to) is already spiking and unchanged (`multi_turn_agent.py`'s biased-competition WTA / `held_referent()`);
wiring this detector into `_resolve_anaphora`/`_resolve` (replacing the host `in _ANAPHORS` test with a
vocab-agnostic-recruited version of this circuit, matching the pattern `VocabAgnosticSpikingSampler` already
uses elsewhere for an open vocabulary) is the deliberately-deferred next rung, named as such in the finding.

Run:
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._spiking_anaphor_detection_derisk \\
      --seeds 42 43 44 100 101 102 --n-noisy-repeats 4 --n-content 10 \\
      --out research/findings/raw/_spiking_anaphor_detection/decisive_6seed.json
"""
from __future__ import annotations

import os
import sys
import json
import argparse

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from research.runners.content_selection_spiking import SpikingLoopContextBuffer  # noqa: E402

# --- circuit sizing (cheap-first: reuses the project's own validated multi-concept-WM config) -------------
ANAPHORS = ["it", "that", "they", "them", "this"]      # EXACT current host set (brain_chat_tui._resolve_anaphora)
N_NEURONS = 600          # per-region (cortex_ctx / dlpfc_wm) population, the SpikingLoopContextBuffer default
PATTERN_SIZE = 50        # neurons per stored assembly; K*PATTERN_SIZE=250 << 600 leaves an "unused" pool
ATTRACTOR_WEIGHT = 50.0  # the Hebbian outer-product install strength (Marr's "changes in connections")
DRIVE_PA = 2500.0
STIM_STEPS = 40
SETTLE_STEPS = 15
READ_WINDOW = 20
THETA = 0.30              # ignition threshold on buf.read()'s per-concept mean firing rate (calibrated: a
                           # driven/completed assembly saturates near ~0.9-1.0, an undriven one sits <0.05)
KEEP_FRAC = 0.20           # NOISY cue: only 20% of the TRUE assembly is kept; the other 80% is corrupted
                           # (replaced by random unused-region neurons that carry no stored assembly at
                           # all). Calibrated by an explicit sweep (0.55/0.40/0.30/0.20/0.10/0.08/0.06/0.04/
                           # 0.02/0.00 keep-fraction, 3 seeds) BEFORE the decisive run: 0.20 sits solidly
                           # inside the reliable-completion regime (0.95-1.00 accuracy over 8 repeats x 3
                           # seeds at n_noisy_repeats=8) while remaining far harder than an exact host-string
                           # match could ever tolerate (which requires 100% overlap, zero tolerance for ANY
                           # corruption). Lower keep-fractions (<=0.10, i.e. as few as 1-5/50 true neurons)
                           # STILL frequently complete correctly given this attractor's strong recurrent
                           # weight (ATTRACTOR_WEIGHT=50), but with higher seed-to-seed variance (0.60-1.00
                           # observed) -- an honest characterization of an EVEN MORE fault-tolerant regime,
                           # not adopted here because the board gate wants a comfortably robust, not a
                           # knife-edge, operating point.
# validated clean multi-concept-WM operating point (content_selection_spiking.SpikingController's own
# comment: "internal_density=0.0 + enable_ou=False -> 6/6 seeds" -- avoids OU-driven spurious cross-talk).
BUF_KW = dict(n=N_NEURONS, pattern_size=PATTERN_SIZE, loop_weight=0.0, internal_density=0.0, enable_ou=False,
              verbose=False)


def _unused_pool(seed, n=N_NEURONS, k=len(ANAPHORS), pattern_size=PATTERN_SIZE):
    """Recompute, EXTERNALLY, which of the buffer's own `n` LOCAL region positions carry no stored
    assembly. `SpikingLoopContextBuffer.__init__` allocates patterns via `rng = np.random.default_rng(seed);
    perm = rng.permutation(n)`, then concept i gets `perm[i*pattern_size:(i+1)*pattern_size]`, in concept
    list order -- reproduced verbatim here (same seed, same n, same order) so this is a faithful
    reconstruction, not a guess. Returns the LOCAL (0..n-1) unused positions; the caller maps them through
    the SAME `rm.indices('cortex_ctx')` the buffer used to reach global neuron ids."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    used = perm[: k * pattern_size]
    return np.setdiff1d(np.arange(n), used, assume_unique=False)


def _cortex_local_index(buf):
    """The LOCAL (0..n-1) position of each GLOBAL cortex_ctx neuron id, so a global index array (as stored
    in buf._cpat) can be mapped back to local permutation-space positions and vice versa."""
    cidx = np.asarray(list(buf.bridge.region_manager.indices("cortex_ctx")), dtype=np.int64)
    return cidx


def _drive_and_read(buf, global_indices, drive_pA=DRIVE_PA, stim=STIM_STEPS, settle=SETTLE_STEPS,
                     window=READ_WINDOW):
    """Drive an ARBITRARY set of cortex_ctx neuron indices (a CUE -- not necessarily any stored concept's
    own pattern) for `stim` steps, settle, then read the per-concept mean firing rate. Mirrors
    `SpikingLoopContextBuffer.update()`/`.read()` internals but takes raw indices instead of a concept key,
    since a cue may be a NOISY/PARTIAL/foreign pattern that is not itself one of `buf.concepts`."""
    xp = buf.xp
    drv = xp.asarray(np.asarray(global_indices, dtype=np.int64))
    for _ in range(stim):
        buf.bridge.cp_external_input_current[:] = 0.0
        buf.bridge.cp_external_input_current[drv] = drive_pA
        buf._step()
    buf.bridge.cp_external_input_current[:] = 0.0
    for _ in range(settle):
        buf._step()
    return buf.read(window=window)


def decide_pronoun(rates, theta=THETA):
    """THE DECISION standing in for the host set test: pronoun iff the max per-concept firing rate clears
    `theta`; the winning concept is the argmax. Zero Python set/dict/tuple membership against a literal
    closed-class word list -- see `_structural_no_host_membership_selftest` below (which checks this
    function's CODE BODY, not this docstring)."""
    if not rates:
        return None, 0.0
    winner = max(rates, key=rates.get)
    peak = rates[winner]
    return (winner if peak >= theta else None), peak


def _fresh_probe(seed, attractor_weight, cue_indices):
    """Build a BRAND-NEW buffer (same `seed` -> the SAME deterministic concept-pattern allocation, but a
    fresh, quiescent network state -- NMDA membrane potentials, `cp_firing_states`, everything reset) and
    classify ONE cue on it, then let it be garbage-collected. THIS is the correct experimental unit: a
    single-shot classification decision, exactly how the live conversation would ask "is this token a
    pronoun?" once per token. A FIRST version of this de-risk reused one buffer across an entire probe
    battery and got clean_accuracy=noisy_accuracy=0.200 (=1/5, pure chance) with false_positive_rate=1.000
    -- every earlier drive's NMDA-bistable assembly LATCHED ON and never turned off (persistent WM is the
    documented, validated feature of this exact buffer class), so by the last probe in a battery every
    concept read as simultaneously "ignited." A fresh buffer per probe removes that carryover entirely."""
    buf = SpikingLoopContextBuffer(ANAPHORS, attractor_weight=attractor_weight, seed=seed, **BUF_KW)
    rates = _drive_and_read(buf, cue_indices)
    return decide_pronoun(rates)


def run_probe_set(seed, attractor_weight, n_noisy_repeats, n_content, rng):
    """One full probe battery: each individual probe gets its OWN fresh buffer (see `_fresh_probe`) built at
    `attractor_weight` (ATTRACTOR_WEIGHT for the intact arm, 0.0 for the untrained/no-LTP lesion arm,
    mirroring `_riii_ca3_completion_specificity_derisk.py`'s own NO-TRAIN control). One throwaway buffer is
    built ONCE up front purely to read off the deterministic pattern allocation (`_cpat` per concept, the
    unused-neuron pool) that every fresh probe buffer at this `seed` will reproduce identically."""
    scratch = SpikingLoopContextBuffer(ANAPHORS, attractor_weight=attractor_weight, seed=seed, **BUF_KW)
    cidx = _cortex_local_index(scratch)
    unused_local = _unused_pool(seed)
    unused_global = cidx[unused_local]
    full_patterns = {c: np.asarray(scratch.B.to_host(scratch._cpat[c]), dtype=np.int64) for c in ANAPHORS}
    del scratch

    clean_correct, noisy_correct = [], []
    for c in ANAPHORS:
        full_pat = full_patterns[c]
        winner, _peak = _fresh_probe(seed, attractor_weight, full_pat)
        clean_correct.append(bool(winner == c))
        for _r in range(n_noisy_repeats):
            n_keep = max(1, int(round(KEEP_FRAC * PATTERN_SIZE)))
            keep = rng.choice(full_pat, size=n_keep, replace=False)
            fill = rng.choice(unused_global, size=PATTERN_SIZE - n_keep, replace=False)
            noisy_pat = np.concatenate([keep, fill])
            winner_n, _peak_n = _fresh_probe(seed, attractor_weight, noisy_pat)
            noisy_correct.append(bool(winner_n == c))

    false_positives = []
    for _w in range(n_content):
        probe = rng.choice(unused_global, size=PATTERN_SIZE, replace=False)
        winner_c, _peak_c = _fresh_probe(seed, attractor_weight, probe)
        false_positives.append(bool(winner_c is not None))

    return {
        "clean_accuracy": float(np.mean(clean_correct)),
        "noisy_accuracy": float(np.mean(noisy_correct)),
        "false_positive_rate": float(np.mean(false_positives)),
        "n_clean": len(clean_correct), "n_noisy": len(noisy_correct), "n_content": len(false_positives),
    }


def run_seed(seed, n_noisy_repeats=4, n_content=10):
    rng = np.random.default_rng(seed * 7919 + 1)
    intact = run_probe_set(seed, ATTRACTOR_WEIGHT, n_noisy_repeats, n_content, rng)
    rng2 = np.random.default_rng(seed * 7919 + 2)
    lesion = run_probe_set(seed, 0.0, n_noisy_repeats, n_content, rng2)

    g1 = bool(intact["clean_accuracy"] >= 0.90)
    g2 = bool(intact["noisy_accuracy"] >= 0.85)
    g3 = bool(intact["false_positive_rate"] <= 0.15)
    # G4: an "accuracy" for the lesion arm counting a false pronoun call as correct would be meaningless
    # (a stored word MUST clear theta; an untrained network without a stored assembly at ATTRACTOR_WEIGHT=0
    # cannot legitimately "recognize" anything) -- the lesion's own combined (clean+noisy) completion
    # accuracy is the effect to compare against the intact arm's, and it should collapse toward 0.
    lesion_combined_acc = float(np.mean([lesion["clean_accuracy"], lesion["noisy_accuracy"]]))
    intact_combined_acc = float(np.mean([intact["clean_accuracy"], intact["noisy_accuracy"]]))
    attrib = attributable_to("anaphor pattern-completion (attractor weight)", intact_combined_acc,
                              lesion_combined_acc, warn_below=0.5)
    g4_collapse = bool(lesion_combined_acc <= 0.30)
    g4 = bool(g4_collapse and attrib is not None and attrib >= 0.5)

    go = bool(g1 and g2 and g3 and g4)
    return {
        "seed": seed,
        "clean_accuracy": round(intact["clean_accuracy"], 4),
        "noisy_accuracy": round(intact["noisy_accuracy"], 4),
        "false_positive_rate": round(intact["false_positive_rate"], 4),
        "lesion_clean_accuracy": round(lesion["clean_accuracy"], 4),
        "lesion_noisy_accuracy": round(lesion["noisy_accuracy"], 4),
        "lesion_combined_accuracy": round(lesion_combined_acc, 4),
        "attribution_to_attractor": None if attrib is None else round(float(attrib), 4),
        "G1_clean_recall": g1, "G2_noisy_surpass": g2, "G3_specificity": g3, "G4_lesion_collapse": g4,
        "GO": go,
    }


def _structural_no_host_membership_selftest():
    """G5 (structural, once, not per-seed): `decide_pronoun` -- the actual classification decision --
    performs zero Python set/dict/tuple membership tests against a literal closed-class word list (the
    exact shortcut being retired: `word.lower() in _ANAPHORS` / `tl in anaphors`). It may only compare
    numeric firing-rate values (`>=`) and take an `argmax`/`max(..., key=...)` over a dict of floats.
    Checks the function's CODE BODY only (the docstring is stripped via `ast`, so prose mentioning the
    retired host pattern for context does not itself trip the check)."""
    import ast
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(decide_pronoun))
    tree = ast.parse(src)
    fn = tree.body[0]
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                            and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                            and isinstance(fn.body[0].value.value, str)) else fn.body
    code_only = ast.unparse(ast.Module(body=body, type_ignores=[]))
    violation = void_if(
        (" in _ANAPHOR" in code_only) or (" in anaphors" in code_only) or ("ANAPHORS" in code_only)
        or ('"it"' in code_only) or ("'it'" in code_only),
        "a literal closed-class word set/membership test leaked into the spiking decision path")
    assert not violation, "G5 FAILED: decide_pronoun() still contains a host word-list membership test"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-noisy-repeats", type=int, default=4)
    ap.add_argument("--n-content", type=int, default=10)
    ap.add_argument("--out", default="research/findings/raw/_spiking_anaphor_detection/derisk.json")
    a = ap.parse_args()

    g5 = _structural_no_host_membership_selftest()
    rows = [run_seed(s, n_noisy_repeats=a.n_noisy_repeats, n_content=a.n_content) for s in a.seeds]
    for r in rows:
        print(f"[anaphor-detect s{r['seed']}] clean={r['clean_accuracy']:.3f} noisy={r['noisy_accuracy']:.3f} "
              f"fp_rate={r['false_positive_rate']:.3f} lesion_acc={r['lesion_combined_accuracy']:.3f} "
              f"attrib={r['attribution_to_attractor']} || {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    n = len(rows)
    v = Verdict("spiking-anaphor-detection board (>=5/6 of the 6 project-standard seeds)")
    v.require("all 6 project-standard seeds present", n == 6, expect=True, note="seeds run: %d" % n)
    v.require("structural G5 (no host word-list membership in the decision path)", g5, expect=True)
    v.floor("seed-GO count", measured=ngo, floor=4.5, note="board bar is >=5/6 -> floor=4.5 excludes 4/6")
    decided = v.decide(go=(n == 6 and ngo >= 5))
    verdict = decided["status"]
    print(f"[anaphor-detect] {ngo}/{n} seed-GO (board bar >=5/6); structural G5={g5} || verdict={verdict}",
          flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {"rows": rows, "n_go": ngo, "n_seeds": n, "G5_no_host_membership": g5, "verdict": verdict}
    payload.update(decided)
    json.dump(payload, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
