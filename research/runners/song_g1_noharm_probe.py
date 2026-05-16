"""Increment G1 Task 8 -- the LOAD-BEARING no-harm safety gate.

Proves the new pure SongHVC controller (sim/song_hvc.py), when present
but SILENT (constructed, never reset/step/rollout-driven into any
bridge), does NOT regress the multi-seed-validated catalog G.20
comprehension path. Validated by RUNNING it on the real 5-bridge
320-sparse production substrate -- not by a contrived unit test
(project pattern: tests/test_song_g1_ignite_smoke.py is import-smoke
only; THIS probe is the real validation).

WHY THE "KNOWN" PROBES ARE ENCODED CROSS-BRIDGE PAIRS
-----------------------------------------------------
The 320-tier checkpoints restore 64 *single-word* base tags per bridge.
Empirically (verified during Task 8 design via a discovery probe) NONE
of the 320 base-tag words clear the abstention gate (650) on a
checkpoint-only `_query_top`: the top associate sits at the noise floor
(e.g. apple->spoon ~617 < 650). That is *correct* abstention -- those
concepts have no encoded association. The 650 threshold
(2026-05-16-G20-320-abstention-benchmark: encoded mean ~796, control
max ~584, AUC 0.990) was calibrated on RUNTIME-ENCODED cross-bridge
`A is B` tags, the exact regime the validated comprehension path uses
(g20_abstention_benchmark / g20_xbridge_benchmark both encode via
SharedPoolMember.encode_partial, then `_query_top`).

So the "known" probes here are a FROZEN, deterministic set of
cross-bridge pairs that we encode through the UNMODIFIED validated
`encode_partial` path (DRY -- the exact mechanism the comprehension
path uses), making them genuine encoded engram tags that SHOULD clear
650. The pair list was selected from
`sample_xbridge_pairs(vocabs, n_pairs=30, seed=42, exclude_idx=12)`
-- the same validated deterministic sampler the abstention / xbridge
benchmarks use -- keeping only pairs whose expected associate is the
robust top-1 AND clears 650 in THREE independent control passes (min
rate >= 776, well above the gate so intrinsic bridge variance can't
push them under it). The chosen 12 are then FROZEN here as a
documented constant so the probe is byte-reproducible.

650 IS USED LITERALLY AND CORRECTLY HERE
----------------------------------------
`_query_top` decodes via the UNMODIFIED `stim_recall_sparse_rates`
CONTINUOUS-DRIVE path -- the exact regime 650 was calibrated on. This
is NOT the no-drive `self_comprehend` residual regime (Task 9/10),
which needs its own pre-registered floor. Here the literal 650 is the
right invariant.

THE NO-HARM TEST (rigorous, control-banded form)
------------------------------------------------
The G.20 bridge has OU noise + stochastic Izhikevich dynamics, AND
`_query_top` itself drives `stimulate_tag` into the shared pool and
advances the bridge, leaving residual state. So two *independent*
`_query_top` passes of the same query are NOT bit-identical: a Task 8
design control measured ~12-16% pass-to-pass top-rate variance with NO
SongHVC anywhere in the process. A naive fixed tolerance would
therefore flag intrinsic bridge stochasticity as a "regression". The
honest no-harm test must compare the silent-SongHVC effect against the
bridge's OWN no-SongHVC pass-to-pass variance band:

  PASS A1 : _query_top(word_a)  -- NO SongHVC in process
  PASS A2 : _query_top(word_a)  -- NO SongHVC in process (control band)
            |A1-A2| = the bridge's intrinsic pass-to-pass variance.
  PASS B  : construct the REAL sim.song_hvc.SongHVC(8,64,seed=42),
            hold it SILENT (never reset/step/rollout -- pure +
            bridge-independent BY CONSTRUCTION), re-run _query_top.

SELF-REFERENTIAL "VALIDATED-KNOWN" GATE (the load-bearing honesty fix)
---------------------------------------------------------------------
A second Task 8 design finding: a cross-bridge association's rate
depends on cross-bridge ENCODING INTERFERENCE -- how many *other*
cross-bridge tags are co-encoded into the shared pool. The same pair
(e.g. ride->false) can be a rock-solid 989 top-1 under one encoding
set and collapse to a 456 noise-floor different-top under another, in
the NO-SongHVC control passes themselves. This is an intrinsic
property of the validated G.20 sparse substrate, fully SongHVC-
INDEPENDENT (it appears identically in A1/A2). Pre-freezing an
"expected to clear 650" list from a *different* run therefore mixes
that interference confound into the no-harm verdict.

The fix: define "known" RELATIVE TO THIS RUN'S OWN no-SongHVC control.
A pair is a VALIDATED-KNOWN probe iff the validated comprehension path
itself answers it WITHOUT any SongHVC -- i.e. its expected associate
is top-1 AND clears 650 in BOTH control passes A1 AND A2. Only those
pairs can possibly be "regressed" by adding SongHVC; a pair the
validated path itself abstains on in this run's interference condition
is not a no-harm subject (excluded + recorded transparently, never
silently). The frozen list is just the deterministic CANDIDATE pool
(robust under selection); the A1 INTERSECT A2 control decides the
actual known set per run -- no number is tuned, the controller is
untouched, and the keep/drop rule is a pre-stated principled
invariant, not cherry-picking.

Per VALIDATED-KNOWN pair, no-harm holds iff:
  (i)  PASS B: expected associate is top-1 AND its rate clears 650
       (the silent SongHVC must not turn a path-answered query into
       an abstention / wrong answer);
  (ii) the with-vs-without shift |B - A2| is within the bridge's own
       intrinsic control band |A1 - A2| (+ documented slack/floor).

WHICH CRITERION IS LOAD-BEARING (honest -- read this)
-----------------------------------------------------
The BINDING no-harm guarantee is criterion (i): every VALIDATED-KNOWN
subject must, WITH the silent SongHVC present, STILL return its
expected associate as top-1 AND STILL clear the absolute 650 gate (650
used in its correct continuous-drive calibration regime -- see above).
That is the assertion that catches the v12/v13/v15-class catastrophic
selectivity loss and ANY regression that crosses 650 or flips top-1 on
any validated-known subject.

Criterion (ii) (the A1/A2/B run-relative band) is a COARSE SECONDARY
sanity bound, NOT a "the silent SongHVC adds no variance" guarantee.
Worked out, the per-word allowance is
  allowed = |a1 - a2| + 0.06*denom + 60.0
i.e. the FIXED 0.06*rate + 60 floor (on rates ~600-1100 that is
~96-126 pA) dominates over the measured intrinsic |a1 - a2| (single-
to mid-double-digit pA on most subjects). So (ii) is effectively a
~12-20%-of-rate bound, of which the intrinsic component is the
minority. It would NOT detect a sub-~13% uniform shift that keeps
every subject above 650 -- and that blindness is ACCEPTABLE: a
never-reset/step/rollout pure-numpy SongHVC is structurally bridge-
INDEPENDENT by construction (it shares no array, no RNG stream, no
pathway with the bridge), so this probe empirically CORROBORATES a
structural guarantee; it is not the sole line of defense. (ii) stays
in the verdict as a cheap extra tripwire and a recorded sanity number,
but the load-bearing axis -- the one a regression must violate to be
caught -- is (i)'s absolute-650 + top-1.

A pure, never-driven controller cannot perturb the bridge; this
formalizes that as: of every query the validated path answers in
THIS run, the silent SongHVC regresses none (criterion (i) is the
binding test of that claim).

ABSTENTION MOAT: `_query_top("zzznonsense")` (in NO vocab) with the
silent SongHVC present -> `gate(ranked, 650)` MUST return None (the
agent abstains; no confabulation).

PASS iff: >= 8 candidate pairs survive the A1 INTERSECT A2
validated-known gate (so the test has real subjects), EVERY surviving
validated-known pair satisfies (i)+(ii) WITH the silent SongHVC, AND
the abstention moat holds.

Heavy imports / IO are lazy (inside main): loading 5 sparse 320
bridges is slow + GPU-bound (several minutes expected).

ASCII-only output (Windows cp1252 safe).
"""
from __future__ import annotations

# --- FROZEN deterministic CANDIDATE cross-bridge pairs (documented) --
# WIDENED 2026-05-16 (anti-flaky integrity fix, PRE-DATA, no bar moved):
# the prior 12-pair pool yielded EXACTLY 8/12 validated-known on the
# committed PASS run (4 candidates excluded because the validated path
# itself stochastically abstained on them that run). 8 is the
# pre-registered >= 8 minimum, so the prior pool met it by a razor-thin
# 0-pair margin: one unlucky A1/A2 control sample on one more candidate
# -> n=7 -> a spurious FAIL UNRELATED to any regression (which would
# stall Task 9 or tempt a re-run-until-pass anti-cheat violation). The
# fix is to widen the deterministic CANDIDATE pool so the >= 8 minimum
# is cleared with comfortable margin by genuinely-robust subjects, NOT
# by luck. The >= 8 PASS minimum itself is UNCHANGED (it is the
# pre-registered bar); only the pool it draws from is widened. No
# criterion logic, the literal 650, or the band formula is touched.
#
# Construction (SAME validated deterministic idiom as before, just a
# larger n_pairs prefix): take
#   sample_xbridge_pairs(
#       [ALL_BRIDGES_64[n] for n in BRIDGE_NAMES],
#       n_pairs=60, seed=42, exclude_idx=12)
# -- the exact validated sampler the abstention / xbridge benchmarks
# use; seed 42 unchanged; exclude_idx=12 is the documented 320-tier
# sparse-pattern gap -- then dedup by queried word `word_a` (keep first
# occurrence; the probe queries word_a, so a repeated word_a would just
# re-query the same word with a different expected associate) and keep
# the first 26 unique-word_a pairs. That deterministic prefix is a
# strict SUPERSET of the prior pool: ALL 12 prior pairs (independently
# rate-characterized as robust top-1 + >650 over 3 control passes
# during the original Task 8 design selection -- their measured
# `# min rate` provenance is preserved verbatim below) reappear at the
# SAME (word_a -> word_b) the sampler emits, plus 14 additional
# deterministic candidates. The 14 new candidates are sampler-derived
# but were NOT independently rate-pre-characterized; they do not need
# to be -- the per-run self-referential A1 INTERSECT A2 control gate
# empirically decides VALIDATED-KNOWN membership for EVERY candidate
# (12 prior + 14 new alike). With a robust-rich superset, > 12 of the
# 26 are expected to clear that gate, clearing >= 8 with real margin
# rather than by chance.
#
# This is still the CANDIDATE POOL, not the final known set. Because a
# cross-bridge association's rate depends on cross-bridge encoding
# interference (a Task 8 design finding -- see module docstring; the
# project lesson "probes must match deployed config"), the ACTUAL
# known set is decided per-run by the self-referential A1 INTERSECT
# A2 no-SongHVC control gate inside main(): a candidate counts as
# VALIDATED-KNOWN only if the validated comprehension path itself
# answers it (expected associate top-1 AND >650) in BOTH control
# passes of THIS run. That removes the interference confound from the
# no-harm verdict without tuning any number or touching the
# controller. The frozen list just fixes the deterministic candidate
# pool so the probe is reproducible. (word_a queried, word_b =
# expected encoded associate; "min rate N" = measured min top-rate
# over the 3 probe-condition control passes during the original
# selection [prior 12]; "deterministic widen" = sampler-derived
# widening candidate, A1 n A2 control gate decides it per-run).
_KNOWN_PAIRS = [
    ("only", "down"),       # min rate  761
    ("fix", "false"),       # deterministic widen
    ("one", "touch"),       # deterministic widen
    ("apple", "when"),      # deterministic widen
    ("old", "take"),        # min rate  799
    ("ride", "false"),      # min rate  989
    ("root", "take"),       # deterministic widen
    ("find", "apple"),      # min rate  897
    ("loud", "need"),       # min rate  760
    ("stand", "always"),    # deterministic widen
    ("walk", "ok"),         # min rate  877
    ("maybe", "last"),      # deterministic widen
    ("then", "each"),       # min rate  766
    ("wet", "maybe"),       # min rate  961
    ("smell", "sweet"),     # deterministic widen
    ("long", "bee"),        # deterministic widen
    ("bag", "warm"),        # min rate  880
    ("narrow", "feel"),     # deterministic widen
    ("that", "eye"),        # deterministic widen
    ("hit", "every"),       # deterministic widen
    ("bad", "beyond"),      # deterministic widen
    ("another", "wolf"),    # min rate 1011
    ("whenever", "short"),  # deterministic widen
    ("nose", "hit"),        # min rate  782
    ("leg", "cook"),        # deterministic widen
    ("if", "rich"),         # min rate  758
]

# Word guaranteed absent from all 5 vocabs (abstention-moat probe).
_NONSENSE_WORD = "zzznonsense"

# Slack added to the bridge's OWN measured intrinsic control band
# |A1-A2| when bounding the silent-SongHVC shift |B-A2|. The silent
# SongHVC effect must be <= (intrinsic band + this slack). Slack
# absorbs the asymmetry of comparing two finite stochastic samples
# (A1,A2) vs (A2,B); it is NOT a free tolerance on a real effect --
# the Task 8 design control showed |A1-A2| ~= |B-A2| with no SongHVC
# in existence at all.
_BAND_REL_SLACK = 0.06   # 6 pp on top of the per-word intrinsic band
_BAND_ABS_FLOOR = 60.0   # pA absolute floor (tiny-rate guard)

# SongHVC construction params (the real class; pure, bridge-independent
# by construction -- never reset/step/rollout here).
_SONG_N_STATES = 8
_SONG_N_CONCEPTS = 64
_SONG_SEED = 42

_OUT_JSON = "research/findings/raw/g11_bg/song_g1_noharm.json"


def main() -> int:
    # --- lazy/heavy imports (loading 5 sparse bridges is slow + GPU) -
    import json
    import time
    from pathlib import Path

    from research.runners.song_g1_ignite import load_members
    from research.runners.g20_xbridge_benchmark import _query_top
    from research.runners.abstention_gate import DEFAULT_THRESHOLD, gate
    from sim.song_hvc import SongHVC

    gate_thr = float(DEFAULT_THRESHOLD)  # literal 650 -- correct here
    t0 = time.time()

    print("=" * 64, flush=True)
    print("SONG G1 TASK 8 -- NO-HARM SAFETY GATE", flush=True)
    print("(silent SongHVC must NOT regress validated G.20 "
          "comprehension)", flush=True)
    print("=" * 64, flush=True)

    # --- load the 5 validated G.20 320-sparse bridges (DRY) ----------
    print("Loading 5 sparse 320-tier bridges via "
          "song_g1_ignite.load_members(seed=42) ...", flush=True)
    members = load_members(seed=42)
    print(f"  loaded: {[m.name for m in members]} "
          f"({int(time.time()-t0)}s)", flush=True)
    for m in members:
        print(f"    {m.name}: {len(m.vocab)} vocab, "
              f"{len(m.encoded_tags)} base tags", flush=True)

    def _member_for(word: str):
        for m in members:
            if word in m.vocab_set:
                return m
        raise KeyError(f"word {word!r} not in any bridge vocab")

    # --- encode the FROZEN known cross-bridge pairs via the UNMODIFIED
    #     validated encode_partial path (exact comprehension-path
    #     mechanism; DRY -- not reimplemented). These become genuine
    #     encoded engram tags that SHOULD clear 650. -----------------
    print(f"\nEncoding {len(_KNOWN_PAIRS)} frozen known cross-bridge "
          f"pairs via validated encode_partial ...", flush=True)
    for wa, wb in _KNOWN_PAIRS:
        ma = _member_for(wa)
        mb = _member_for(wb)
        tag = f"{wa}_{wb}"
        ma.encode_partial(wa, tag)
        mb.encode_partial(wb, tag)
        for m in (ma, mb):
            if tag not in m.encoded_tags:
                m.encoded_tags.append(tag)
        print(f"  encoded {tag}: {ma.name}({wa}) + {mb.name}({wb})",
              flush=True)

    def _run_pass(label: str) -> dict:
        """One full _query_top sweep over the frozen known words."""
        out = {}
        for wa, wb in _KNOWN_PAIRS:
            ranked = _query_top(members, wa)
            if ranked:
                out[wa] = (ranked[0][0], float(ranked[0][1]))
            else:
                out[wa] = (None, 0.0)
            print(f"  {wa:<10} -> {out[wa][0]!s:<10} "
                  f"rate={out[wa][1]:>8.1f}  (expected {wb})",
                  flush=True)
        return out

    # --- PASS A1 / A2: NO SongHVC -> bridge intrinsic control band ---
    print("\n[PASS A1] _query_top WITHOUT any SongHVC (control) ...",
          flush=True)
    a1 = _run_pass("A1")
    print("\n[PASS A2] _query_top WITHOUT any SongHVC again "
          "(intrinsic pass-to-pass control band) ...", flush=True)
    a2 = _run_pass("A2")

    # --- PASS B: construct the REAL SongHVC, hold it SILENT ----------
    print(f"\n[PASS B] construct REAL sim.song_hvc.SongHVC("
          f"n_states={_SONG_N_STATES}, n_concepts={_SONG_N_CONCEPTS}, "
          f"seed={_SONG_SEED}) and hold it SILENT", flush=True)
    print("         (constructed only -- NEVER reset/step/rollout; "
          "pure + bridge-independent by construction)", flush=True)
    silent_song = SongHVC(n_states=_SONG_N_STATES,
                          n_concepts=_SONG_N_CONCEPTS,
                          seed=_SONG_SEED)
    # Hard-check the recorded inertness claim instead of only
    # documenting it: a freshly-constructed SongHVC is unstarted
    # (_state == -1); reset()/step()/rollout() are the ONLY ways to
    # advance it, and this probe calls none of them. Asserting here
    # makes the JSON's internal_state_unstarted claim load-bearing.
    assert silent_song._state == -1, \
        "SongHVC must be unstarted/silent"
    _song_inert = {
        "type": type(silent_song).__name__,
        "module": type(silent_song).__module__,
        "n_states": silent_song.n_states,
        "n_concepts": silent_song.n_concepts,
        "internal_state_unstarted": (silent_song._state == -1),
        "W_shape": list(silent_song.W.shape),
        "driven": False,  # never reset/step/rollout in this probe
    }
    print(f"  SongHVC present: {_song_inert}", flush=True)

    print("\n[PASS B] re-run the SAME _query_top queries WITH the "
          "silent SongHVC present ...", flush=True)
    b = _run_pass("B")

    # --- evaluate: self-referential A1 INTERSECT A2 validated-known
    #     gate, then the control-banded no-harm assertions ----------
    per_word = []
    n_validated_known = 0   # candidates the path answers in A1 AND A2
    n_known_ok = 0          # validated-known that also pass (i)+(ii)
    max_excess = 0.0        # max (silent-shift - allowed), <=0 == ok
    for wa, wb in _KNOWN_PAIRS:
        a1_assoc, a1_rate = a1[wa]
        a2_assoc, a2_rate = a2[wa]
        b_assoc, b_rate = b[wa]

        # VALIDATED-KNOWN gate: the validated comprehension path itself
        # answers this query WITHOUT any SongHVC -- expected associate
        # top-1 AND clears 650 in BOTH control passes of THIS run.
        # Only such pairs are no-harm subjects (a pair the path itself
        # abstains on in this run's interference condition cannot be
        # "regressed" by adding an inert controller).
        a1_ok = (a1_assoc == wb) and (a1_rate > gate_thr)
        a2_ok = (a2_assoc == wb) and (a2_rate > gate_thr)
        validated_known = bool(a1_ok and a2_ok)

        # (i) WITH-SongHVC: expected associate top-1 AND clears 650
        assoc_top1 = (b_assoc == wb)
        cleared_650 = b_rate > gate_thr

        # (ii) silent-SongHVC shift must be within the bridge's OWN
        #      intrinsic pass-to-pass control band (+ documented slack)
        intrinsic_band = abs(a1_rate - a2_rate)
        silent_shift = abs(b_rate - a2_rate)
        denom = max(abs(a2_rate), abs(b_rate), abs(a1_rate), 1.0)
        allowed = (intrinsic_band
                   + _BAND_REL_SLACK * denom
                   + _BAND_ABS_FLOOR)
        excess = silent_shift - allowed
        within_band = silent_shift <= allowed

        if validated_known:
            n_validated_known += 1
            # band-excess only counts among the no-harm subjects
            max_excess = max(max_excess, excess)
            word_ok = bool(assoc_top1 and cleared_650 and within_band)
            if word_ok:
                n_known_ok += 1
            status = "KNOWN_OK" if word_ok else "KNOWN_REGRESSED"
        else:
            # Not a no-harm subject in this run -- excluded from the
            # verdict, recorded transparently (never silently dropped).
            word_ok = None
            status = "EXCLUDED_PATH_ABSTAINS_THIS_RUN"

        per_word.append({
            "word": wa,
            "expected_assoc": wb,
            "rate_a1_without": round(a1_rate, 2),
            "top_a1_without": a1_assoc,
            "rate_a2_without": round(a2_rate, 2),
            "top_a2_without": a2_assoc,
            "rate_with": round(b_rate, 2),
            "top_assoc": b_assoc,
            "validated_known": validated_known,
            "assoc_is_top1": bool(assoc_top1),
            "cleared_650": bool(cleared_650),
            "intrinsic_band_abs": round(intrinsic_band, 2),
            "silent_shift_abs": round(silent_shift, 2),
            "allowed_band_abs": round(allowed, 2),
            "shift_within_band": bool(within_band),
            "band_excess_abs": round(excess, 2),
            "status": status,
            "word_ok": word_ok,
        })

    # --- abstention moat (with the silent SongHVC still present) -----
    print(f"\n[MOAT] _query_top({_NONSENSE_WORD!r}) with silent "
          f"SongHVC present -> gate must return None ...", flush=True)
    nonsense_ranked = _query_top(members, _NONSENSE_WORD)
    gated = gate(nonsense_ranked, gate_thr)
    abstain_ok = gated is None
    if nonsense_ranked:
        nr_top = nonsense_ranked[0]
        print(f"  ranked top: {nr_top[0]!s} rate={float(nr_top[1]):.1f}"
              f"  gate(.,650) -> "
              f"{'None (ABSTAIN)' if abstain_ok else nr_top}",
              flush=True)
    else:
        print(f"  ranked EMPTY  gate(.,650) -> "
              f"{'None (ABSTAIN)' if abstain_ok else gated}",
              flush=True)

    # --- verdict ----------------------------------------------------
    n_total = len(_KNOWN_PAIRS)
    # need a real test: >= 8 candidates must survive the A1 n A2 gate
    enough_subjects = n_validated_known >= 8
    # every validated-known subject must pass (i)+(ii)
    all_subjects_ok = (n_known_ok == n_validated_known
                       and n_validated_known > 0)
    band_systematic_ok = max_excess <= 0.0
    passed = bool(enough_subjects
                  and all_subjects_ok
                  and abstain_ok
                  and band_systematic_ok)

    # report max relative with/without delta over the VALIDATED-KNOWN
    # subjects (record only -- the control band is the criterion)
    rel_deltas = [
        abs(r["rate_with"] - r["rate_a2_without"])
        / max(abs(r["rate_a2_without"]), abs(r["rate_with"]), 1.0)
        for r in per_word if r["validated_known"]
    ]
    max_rel_delta = max(rel_deltas) if rel_deltas else 0.0

    result = {
        "task": "song_g1 Task 8 no-harm safety gate",
        "substrate": "G.20 320-sparse 5-bridge (seed 42)",
        "method": ("self-referential A1 INTERSECT A2 validated-known "
                   "gate, then control-banded: silent-SongHVC shift "
                   "|B-A2| must be within the bridge's own no-SongHVC "
                   "intrinsic pass-to-pass band |A1-A2| + slack"),
        "candidate_words": [wa for wa, _ in _KNOWN_PAIRS],
        "candidate_pairs": [{"word": wa, "expected_assoc": wb}
                            for wa, wb in _KNOWN_PAIRS],
        "n_candidates": n_total,
        "n_validated_known": n_validated_known,
        "validated_known_words": [
            r["word"] for r in per_word if r["validated_known"]],
        "excluded_words": [
            r["word"] for r in per_word if not r["validated_known"]],
        "n_known_ok": n_known_ok,
        "enough_subjects": bool(enough_subjects),
        "all_validated_known_ok": bool(all_subjects_ok),
        "abstain_ok": bool(abstain_ok),
        "with_vs_without_max_rel_delta": round(max_rel_delta, 4),
        "max_band_excess_abs": round(max_excess, 2),
        "band_rel_slack": _BAND_REL_SLACK,
        "band_abs_floor": _BAND_ABS_FLOOR,
        "gate_threshold": gate_thr,
        "songhvc_inert": _song_inert,
        "nonsense_word": _NONSENSE_WORD,
        "nonsense_top": (
            [nonsense_ranked[0][0],
             round(float(nonsense_ranked[0][1]), 2)]
            if nonsense_ranked else None),
        "per_word": per_word,
        "elapsed_seconds": round(time.time() - t0, 1),
        "PASS": passed,
    }

    Path(_OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # --- ASCII verdict block ----------------------------------------
    print("\n" + "=" * 64, flush=True)
    print("SONG G1 TASK 8 NO-HARM VERDICT", flush=True)
    print("=" * 64, flush=True)
    print("  substrate           : G.20 320-sparse 5-bridge (seed 42)",
          flush=True)
    print(f"  candidate pairs     : {n_total} "
          f"(frozen deterministic pool)", flush=True)
    print(f"  validated-known     : {n_validated_known} "
          f"(path answers in A1 AND A2 -- no-harm subjects)",
          flush=True)
    print(f"  gate threshold      : {gate_thr:.0f} "
          f"(continuous-drive regime; literal-correct here)",
          flush=True)
    print("  no-harm test        : silent-SongHVC shift within "
          "bridge's OWN no-SongHVC variance band", flush=True)
    print("  -" * 32, flush=True)
    for r in per_word:
        if not r["validated_known"]:
            flag = "EXC"
        elif r["word_ok"]:
            flag = "OK "
        else:
            flag = "BAD"
        print(f"  [{flag}] {r['word']:<9} "
              f"a1={r['rate_a1_without']:>7.1f} "
              f"a2={r['rate_a2_without']:>7.1f} "
              f"B={r['rate_with']:>7.1f} | "
              f"intr={r['intrinsic_band_abs']:>6.1f} "
              f"shift={r['silent_shift_abs']:>6.1f} "
              f"allow={r['allowed_band_abs']:>6.1f} "
              f"vk={'Y' if r['validated_known'] else 'N'} "
              f"top1={'Y' if r['assoc_is_top1'] else 'N'} "
              f"650={'Y' if r['cleared_650'] else 'N'}", flush=True)
    print("  (EXC = validated path itself abstains this run -- not a "
          "no-harm subject; recorded, not silently dropped)",
          flush=True)
    print("  -" * 32, flush=True)
    print(f"  validated-known subjects     : "
          f"{n_validated_known} (need >= 8)", flush=True)
    print(f"  no-harm OK among subjects    : "
          f"{n_known_ok}/{n_validated_known}", flush=True)
    print(f"  max band excess (<=0 == ok)  : "
          f"{max_excess:+.1f} pA", flush=True)
    print(f"  (record) max with/without rel: "
          f"{max_rel_delta*100:.1f}% "
          f"-- intrinsic, NOT the criterion", flush=True)
    print(f"  abstention moat (nonsense)   : "
          f"{'ABSTAIN (OK)' if abstain_ok else 'CONFABULATED (BAD)'}",
          flush=True)
    print("  -" * 32, flush=True)
    print(f"  VERDICT : {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        why = []
        if not enough_subjects:
            why.append(f"only {n_validated_known} validated-known "
                       f"(< 8)")
        if n_validated_known > 0 and not all_subjects_ok:
            why.append(f"{n_validated_known - n_known_ok} subject(s) "
                       f"regressed WITH SongHVC")
        if not band_systematic_ok:
            why.append(f"band excess {max_excess:+.1f} pA > 0")
        if not abstain_ok:
            why.append("abstention moat broken")
        print(f"  WHY FAIL: {'; '.join(why)}", flush=True)
    print(f"  -> {_OUT_JSON}", flush=True)
    print("=" * 64, flush=True)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
