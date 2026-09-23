"""GNW SWAP CONTINUOUS CROSS-TURN IGNITION — the named next rung after board #77/#85 (2026-08-19 findings:
'A truly continuous cross-turn ignition (no restore) is the named next rung'). This runner tests whether making the
production held-topic workspace's substrate genuinely CONTINUOUS across turns (no per-turn restore-to-snapshot) is
SAFE (preserves the shipped swap-vs-hold correctness, no regression, INCLUDING on hold/lesion/slot-reuse turn shapes
and against a REAL restore-mode comparison arm). It ALSO tested whether that continuity carries a measurable
synaptic "recency trace" with a reply-relevant causal effect -- IT DOES NOT; see RETRACTION below.

⛔ RETRACTION (2026-09-23 fix round; adversarial review verdict=fix-required, all issues addressed here). The
ORIGINAL version of this finding called a seed-42 near-threshold dissociation (RECENT fails to re-ignite, FRESH
succeeds) an "existence proof" that the STD carryover causes a recency-driven swap failure. That claim is FALSE,
refuted by the runner's OWN neural lesion arm: forcibly resetting ONLY A's carryover (`std.deps[A].x[:] = 1.0`)
immediately before re-proposing it does NOT rescue the swap at seed 42 (`nt_recent_lesioned["swapped"] ==
nt_recent["swapped"] == False`) -- and this null holds on ALL 6 seeds
(`carryover_causal_at_near_threshold` = 0/6, including the one seed with a raw dissociation). If wiping the
carryover to a virgin state does not change the outcome, the carryover was never what blocked the swap; the
dissociation is a pattern-identity (A vs C) or per-seed-heterogeneity confound, not a recency effect. Separately,
production advances ZERO simulated time between HTTP turns while `STD_TAU_D=250ms` -- a real inter-turn gap of
seconds is 8-40*tau_D (full recovery), so citing Mongillo, Barak & Tsodyks 2008 (a real-time-elapsing ~1s working-
memory trace) as the biological substrate for this turn-to-turn carryover is NOT supported; that framing is
DROPPED, not merely softened. **What survives, verified 6/6 seeds below, is narrower and purely about SAFETY: the
substrate's own STD state CAN persist across the HTTP turn boundary (a raw computational fact, not a validated
biological recency trace) without changing any TESTED swap-vs-hold verdict** -- see "GO GATE" below. The recency/
mechanism claim is BANKED AS NO-GO; the next method (NOT attempted here) is either (a) find the actual seed-42/102
near-threshold confound (likely per-pattern Izhikevich-heterogeneity margin, not carryover) directly, or (b) model
real elapsed wall-clock time as inter-turn free-run recovery steps before re-attempting any recency-trace claim.

WHY THIS IS THE GENUINE NEXT RUNG, NOT A RE-DERIVATION (verify-first, `before_you_build.sh` + `git log --grep`
+ findings scan, all run before writing a line of this file). Board #77 (2026-08-19-gnw-swap-into-chat-GO.md) and
board #85 (2026-08-19-swap-drives-chat-load-bearing-GO.md) are BOTH already merged to origin/main, DEFAULT-ON
(`_SWAP_DRIVES_DEFAULT_ON=True` in webapp/server.py), and lesion-verified load-bearing on the live `/api/brain-chat`
reply (a topic-change SWAP prepends a transition lead the mismatch-detector lesion makes vanish) -- re-confirmed
independently by 2026-09-05-rank11-topic-swap-scaffold-backlog-item-already-integrated.md (byte-identical re-run
against TODAY's code) and already wired into `research/runners/load_bearing_fraction.py`'s FACULTY_LESIONS battery
(row `swap-drives-response`). Building that capability again here would DUPLICATE shipped work. What is NOT built is
honest-limit #1 of `webapp/gnw_thought_swap.py`'s own docstring: "the cross-turn CONTINUITY of the held thought is
carried by a host label... RE-ESTABLISHED on the substrate each turn via `run_intention_swap(isolate=True)` (restore
clean snapshot -> re-ignite the held topic...). A truly continuous cross-turn ignition (no restore) is the named next
rung." This runner builds and de-risks exactly that rung.

THE MECHANISM (reuse-by-import, NO `sim/` edit; every primitive below is imported from the ALREADY-6/6-seed-GO
`_gnw_neural_swap_intention_derisk` module -- `build`, `run_intention_swap`, `MultiLoopSTD`; `run_intention_swap`
already exposes `isolate=False` as "a CONTINUOUS run (0 restore calls)" and `run_two_swap` already uses it for a
two-swap A->B->A reversibility headline -- this runner is the FIRST use of that existing continuous-mode plumbing
as a genuinely MULTI-TURN, production-shaped conversation, not a single two-swap probe).

  Per-seed protocol: establish topic A (first-thought, one necessary COLD-START `isolate=True` -- there is no prior
  turn to be continuous WITH), then swap continuously (`isolate=False`, zero restores) A -> B. This evicts A via the
  SAME recurrence-weakening STD the shipped #77/#85 mechanism already uses -- A's recurrent loop is left with a
  depleted resource variable `x_A < 1` (Tsodyks-Markram short-term depression). Two turns now branch from this
  IDENTICAL point (same seed -> matching substrate state to floating-point tolerance, confirmed below; a stricter
  full-state hash is also reported but is a documented BLAS-thread-count artifact under this file's own
  OMP_NUM_THREADS=2 recipe -- see `branch_state_hash_match`'s own comment):
    RECENT — the very next turn re-proposes A (the topic the user JUST left).
    FRESH  — the very next turn proposes C, a topic NEVER held this session (x_C == 1, virgin loop).

GO GATE (robust, 6/6, at the SHIPPED production operating point `SALIENT_PA`) — what THIS finding actually proves,
SAFETY ONLY (2026-09-23 fix round; see RETRACTION above -- no mechanism/recency claim is gated here anymore):
  1. THE CARRYOVER IS REAL (a raw fact, not yet a validated biological trace -- see RETRACTION): x_A at the branch
     point is always < 1 (measured 0.73-0.78 across seeds) -- continuous mode genuinely carries STD state across
     the turn boundary; this is not a bookkeeping label.
  2. RESTORE MODE IS STRUCTURALLY BLIND TO IT: the exact reset call `isolate=True` performs (`std.reset()`) always
     wipes this to EXACTLY 1.0 for every pattern -- a code-level, not merely statistical, guarantee that today's
     shipped restore-every-turn default cannot carry this state, regardless of operating point.
  3. SAFE / NO REGRESSION AGAINST A REAL RESTORE-MODE ARM: at the shipped production drive strength, continuous
     mode's swap-vs-hold VERDICT on every turn (establish, evict, re-admit-recent, admit-fresh) is compared against
     an ACTUAL `isolate=True` run of the same proposal (`restore_recent`/`restore_fresh` -- not a hard-coded `True`
     and not two continuous arms compared to each other) and matches it, 6/6.
  4. SAFE ON UNTESTED TURN SHAPES TOO: a separate check runs the ACTUAL production glue
     (`webapp.gnw_thought_swap.ThoughtSwapWorkspace`) through same-topic holds, the board-#85 mismatch-lesion path,
     and LRU slot reuse past `N_PATTERNS=3` -- turns the original version of this runner never exercised -- and
     confirms continuous-ON reaches the identical swap-vs-hold verdict as continuous-OFF on every one, 6/6
     (`full_conversation_no_regression`).
  5. DETERMINISM (build-twice Izhikevich-parameter hash) holds 6/6, and the two-arm fork's population-level
     decision scalars match to floating-point tolerance (1e-9) at the branch point, 6/6 (`branch_identical`, no
     longer described as byte-identical -- docs/TERMS.md requires a hash or exact compare for that word). A
     stricter full-STD-state-array hash (`branch_state_hash_match`) IS a genuine byte-identical check and is
     reported per-seed, but is NOT gated: under this file's own OMP_NUM_THREADS=2 recipe it is confirmed to fail
     purely from threaded-BLAS floating-point non-associativity (verified bit-identical under OMP_NUM_THREADS=1) --
     gating on it would report a false NO-GO caused by thread count, not by the mechanism.

HONEST RESIDUAL, NAMED AND QUANTIFIED, NOT CLAIMED CLOSED. The recency/mechanism claim above the SAFETY gate is
BANKED AS NO-GO (see RETRACTION): `carryover_causal_at_near_threshold` = 0/6 -- lesioning ONLY the carryover never
changes the near-threshold swap decision on any seed, including the one (42) with a raw dissociation. The
dissociation itself (`near_threshold_diagnostic.dissociation`, 1/6) is left in the artifact as a DIAGNOSTIC-ONLY
observation (never gates `seed_go`/`pooled_go`) precisely because it does not survive its own lesion test -- a
future session should not re-read it as evidence without re-reading this retraction. The lesion tool itself
(`std.deps[0].x[:]=1.0`, forcibly wiping ONLY A's carryover while leaving every other continuous-mode state variable
untouched) remains available for whichever NEXT method actually investigates the seed-42/102 confound (see
RETRACTION's (a)/(b) options) -- neither is attempted in this fix round.

CONTRACT if wired to production (this runner is the DE-RISK; production wiring in `webapp/gnw_thought_swap.py`
behind a NEW, default-off, additive flag `BRAIN_GNW_SWAP_CONTINUOUS` -- unset/0 -> BYTE-IDENTICAL to the shipped
#77/#85 isolate=True-every-turn behavior, asserted in the data by `verify_byte_identical`-style hash compare, not
inferred from reading the code; see that module for the flag and its own honest-residual update). The flag ships as
verified-safe PLUMBING ONLY -- it does not carry, and must not be described as carrying, any capability claim.

Biology: the shipped #77/#85 eviction mechanism's own citation for Tsodyks-Markram short-term synaptic depression
(Mongillo, Barak & Tsodyks 2008, Science 319:1543) is UNCHANGED and still applies to that mechanism. It is NO LONGER
cited here as support for a cross-turn "recency trace" (see RETRACTION: production advances zero simulated time
between turns, so the ~1s real-time trace that citation describes does not apply to what this runner measures).
Corpus check (`before_you_build.sh "GNW continuous cross-turn ignition recency swap"` + rag_search) was run before
writing this file; see the accompanying finding for the near-threshold calibration sweep transcript
(pa in {5000,3000,2000,1500,1200,1000,800}) and this fix round's retraction record.

Usage (CPU cheap-first; export OMP/OPENBLAS/MKL_NUM_THREADS=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_swap_continuous_recency_derisk --smoke --seed 42 \\
      --json research/findings/raw/_gnw_swap_continuous_recency_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_swap_continuous_recency_derisk --six-seed \\
      --json research/findings/raw/_gnw_swap_continuous_recency_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim.backend import to_host
from tools.verdict import Verdict
from tools.lab import lever, void_if

from research.runners._gnw_neural_swap_intention_derisk import (
    build, run_intention_swap, MultiLoopSTD, SALIENT_PA, N_PATTERNS, W_REC,
)

# ── operating point (reused UNCHANGED from the shipped #77/#85 mechanism's own calibrated point) ──────────────────
A, B, C = 0, 1, 2                       # three disjoint held-topic slots; N_PATTERNS must be >= 3 (it is: 3)
assert N_PATTERNS >= 3, "this probe needs >=3 disjoint pattern slots (A, B, and a never-touched C)"

MIN_X_DEFICIT = 0.05      # x_A_at_branch must be at least this far below 1.0 (the carryover LEVER must have moved)
NEAR_THRESHOLD_PA = 1500.0  # hand-swept diagnostic drive (see module docstring); NEVER gates seed_go.


def _izh_hash(bridge):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest() if parts else ""


def _fresh_branch(seed, w_rec, heterogeneity):
    """Build a substrate and drive it, CONTINUOUSLY (one necessary cold-start restore for the very first thought,
    then zero restores), through: establish A -> swap A->B. Returns (S, std, first, ab) at the branch point: B is
    held, A was JUST evicted (its recurrence carries a partial STD debt)."""
    S = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std = MultiLoopSTD(S["bridge"], S["xp"], S["ws_used"], S["patterns_host"])
    first = run_intention_swap(S, std, incumbent=A, proposed=A, proposal_pa=SALIENT_PA, isolate=True)
    ab = run_intention_swap(S, std, incumbent=A, proposed=B, proposal_pa=SALIENT_PA, isolate=False)
    return S, std, first, ab


_SAFETY_TURNS = [  # a,b,c,d,e: 5 distinct topics against N_PATTERNS=3 -> forces LRU slot reuse; two same-topic
    # holds; a lesion turn (board-#85 path) immediately followed by its un-lesioned re-proposal.
    ("a", False), ("a", False), ("b", False), ("b", False), ("a", False),
    ("c", False), ("d", False), ("e", True), ("e", False),
]
_SAFETY_VERDICT_FIELDS = ("acted", "swapped", "reason", "held_topic", "evicted_topic")


def _full_conversation_verdicts(seed, continuous):
    """Run `webapp.gnw_thought_swap.ThoughtSwapWorkspace` through `_SAFETY_TURNS` (2026-09-23 review fix: the
    original gate exercised ONLY swap turns -- establish/A->B/re-propose-A/propose-C -- and never same-topic holds,
    the board-#85 lesion path, or LRU slot reuse past N_PATTERNS. This runs the actual production glue module, not
    just the low-level primitive, so it also catches a regression the low-level arms above cannot see.) Returns the
    per-turn swap-vs-hold VERDICT fields only (not internal numeric state, which is EXPECTED to differ once
    continuity is on) so a caller can compare continuous-ON against continuous-OFF turn-by-turn."""
    import os as _os
    prior = _os.environ.get("BRAIN_GNW_SWAP_CONTINUOUS")
    try:
        if continuous:
            _os.environ["BRAIN_GNW_SWAP_CONTINUOUS"] = "1"
        else:
            _os.environ.pop("BRAIN_GNW_SWAP_CONTINUOUS", None)
        from webapp.gnw_thought_swap import ThoughtSwapWorkspace
        ws = ThoughtSwapWorkspace(seed=seed)
        return [{k: ws.observe(topic, lesion=lesion).get(k) for k in _SAFETY_VERDICT_FIELDS}
                for topic, lesion in _SAFETY_TURNS]
    finally:
        if prior is None:
            _os.environ.pop("BRAIN_GNW_SWAP_CONTINUOUS", None)
        else:
            _os.environ["BRAIN_GNW_SWAP_CONTINUOUS"] = prior


def _full_conversation_no_regression(seed):
    off = _full_conversation_verdicts(seed, continuous=False)
    on = _full_conversation_verdicts(seed, continuous=True)
    return bool(off == on), off, on


def evaluate_seed(seed, *, w_rec=None, heterogeneity=True, near_threshold_pa=NEAR_THRESHOLD_PA, verbose=True):
    if w_rec is None:
        w_rec = W_REC

    # ── GATING ARMS (production operating point, SALIENT_PA -- what "safe to wire" is measured on) ────────────────
    S1, std1, first1, ab1 = _fresh_branch(seed, w_rec, heterogeneity)
    xA_at_branch = std1.x_mean(A)
    recent = run_intention_swap(S1, std1, incumbent=B, proposed=A, proposal_pa=SALIENT_PA, isolate=False)

    S2, std2, first2, ab2 = _fresh_branch(seed, w_rec, heterogeneity)
    xC_at_branch = std2.x_mean(C)
    fresh = run_intention_swap(S2, std2, incumbent=B, proposed=C, proposal_pa=SALIENT_PA, isolate=False)

    # ── REAL RESTORE-MODE ARMS (2026-09-23 review fix: the gate previously compared two CONTINUOUS arms against a
    # hard-coded `True` and called that "no regression" -- it never ran the thing production actually does by
    # default, isolate=True. `run_intention_swap(..., isolate=True)` restores the substrate to a virgin snapshot AND
    # re-establishes `incumbent` from scratch before deciding on `proposed` (see its own docstring, step (1)) -- this
    # is EXACTLY what the shipped default does every turn (residual #1: "RE-ESTABLISHED on the substrate each turn").
    # Because isolate=True wipes all prior history unconditionally, a single freshly-built substrate reproduces
    # restore mode's own verdict for "propose X while B is held" with no need to replay the whole conversation.
    S6 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std6 = MultiLoopSTD(S6["bridge"], S6["xp"], S6["ws_used"], S6["patterns_host"])
    restore_recent = run_intention_swap(S6, std6, incumbent=B, proposed=A, proposal_pa=SALIENT_PA, isolate=True)

    S7 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std7 = MultiLoopSTD(S7["bridge"], S7["xp"], S7["ws_used"], S7["patterns_host"])
    restore_fresh = run_intention_swap(S7, std7, incumbent=B, proposed=C, proposal_pa=SALIENT_PA, isolate=True)

    # the two-arm fork is a CONTROLLED comparison only if S1/S2 reach the SAME state up to the branch (same seed,
    # same sequence of steps up to turn 3 -> must match before diverging on turn 3's proposal). TERMINOLOGY FIX
    # (2026-09-23 review, docs/TERMS.md): the prior version compared 4 scalar read-outs and called that "byte-
    # identical branch state" -- TERMS.md requires "byte-identical" to be a hash or exact array compare, not 4
    # floats, so that claim is DROPPED for `branch_identical` below (kept, renamed in wording only, as a
    # floating-point-TOLERANCE scalar match -- never described as byte-identical anywhere in this file anymore).
    branch_identical = bool(abs(ab1["new_rate_post"] - ab2["new_rate_post"]) < 1e-9
                             and abs(ab1["old_residual_post"] - ab2["old_residual_post"]) < 1e-9
                             and ab1["winner_post"] == ab2["winner_post"] and ab1["n_ignited_post"] == ab2["n_ignited_post"])

    def _std_state_hash(std):
        parts = [np.asarray(d.x, dtype=np.float64) for d in std.deps]
        return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest()

    # `branch_state_hash_match`: a GENUINE byte-identical check (full per-neuron STD resource-variable array hash,
    # docs/TERMS.md-compliant) -- but measured here to be a NON-GATING diagnostic, not part of seed_go/pooled_go.
    # Verified directly (see this fix round's notes): under OMP_NUM_THREADS=1 this hash MATCHES exactly on repeated
    # builds at the same seed (bit-for-bit); under this module's own documented OMP_NUM_THREADS=2 recipe it does NOT
    # match, because threaded BLAS reduction order is not deterministic across runs at the ~1e-15 level -- a
    # floating-point non-associativity artifact of thread count, not a substantive divergence between S1 and S2
    # (the population-level `branch_identical` scalars above, which is what the swap DECISION actually depends on,
    # match to 1e-9 regardless of thread count). Gating on the hash under the recipe this file tells users to run
    # would report a false NO-GO caused by BLAS threading, not by the mechanism -- so it is reported, not gated.
    branch_state_hash_match = bool(_std_state_hash(std1) == _std_state_hash(std2))

    # restore-mode-blind: the EXACT operation isolate=True performs before every shipped-mode turn (std.reset(),
    # called on std1 -- which has JUST come out of a real continuous A->B swap and carries the 0.7x debt measured
    # above) unconditionally wipes it back to 1.0 for EVERY pattern. This is a code-level guarantee, verified here
    # rather than merely asserted by reading the source.
    std1.reset()
    restore_blind = bool(abs(std1.x_mean(A) - 1.0) < 1e-12 and abs(std1.x_mean(C) - 1.0) < 1e-12)

    h1 = _izh_hash(S1["bridge"]); h2 = _izh_hash(S2["bridge"])
    seed_deterministic = bool(h1 == h2 and h1 != "")

    swaps_correct = bool(ab1["swapped"] and ab2["swapped"] and recent["swapped"] and fresh["swapped"]
                         and restore_recent["swapped"] and restore_fresh["swapped"])
    # FIX (2026-09-23 review): this used to compare the two CONTINUOUS arms (recent vs fresh) against each other and
    # a hard-coded `True` -- it never measured continuous mode against what restore mode (isolate=True, the shipped
    # default) actually decides. Now it compares each continuous arm against ITS OWN real restore-mode counterpart.
    no_regression_at_production_pa = bool(
        recent["swapped"] == restore_recent["swapped"]
        and fresh["swapped"] == restore_fresh["swapped"]
        and abs(recent["new_rate_post"] - restore_recent["new_rate_post"]) < 1e-6
        and abs(fresh["new_rate_post"] - restore_fresh["new_rate_post"]) < 1e-6
    )

    carryover_insufficient = void_if(xA_at_branch >= 1.0 - MIN_X_DEFICIT,
                                      "x_A_at_branch did not drop below 1.0-%.3f -- no STD carryover to measure; "
                                      "the whole comparison is void" % MIN_X_DEFICIT)
    carryover_ok = bool(not carryover_insufficient)

    # ── DIAGNOSTIC ARMS (near-threshold PA -- reported, NEVER gates seed_go; see module docstring) ─────────────────
    S3, std3, first3, ab3 = _fresh_branch(seed, w_rec, heterogeneity)
    nt_recent = run_intention_swap(S3, std3, incumbent=B, proposed=A, proposal_pa=near_threshold_pa, isolate=False)
    S4, std4, first4, ab4 = _fresh_branch(seed, w_rec, heterogeneity)
    nt_fresh = run_intention_swap(S4, std4, incumbent=B, proposed=C, proposal_pa=near_threshold_pa, isolate=False)
    S5, std5, first5, ab5 = _fresh_branch(seed, w_rec, heterogeneity)
    std5.deps[A].x[:] = 1.0    # the neural lesion: wipe ONLY A's carryover, keep everything else continuous
    nt_recent_lesioned = run_intention_swap(S5, std5, incumbent=B, proposed=A, proposal_pa=near_threshold_pa, isolate=False)
    near_threshold_dissociation = bool(nt_recent["swapped"] is False and nt_fresh["swapped"] is True)
    # FALSIFICATION CHECK (2026-09-23 review): a dissociation (recent fails, fresh succeeds) is evidence the STD
    # carryover CAUSED the failure only if wiping ONLY the carryover (nt_recent_lesioned, std5.deps[A].x[:]=1.0)
    # RESCUES the swap. If the lesioned arm decides the SAME as the unlesioned arm, the carryover was never the thing
    # blocking it -- something else (pattern identity, per-seed heterogeneity) is, and the dissociation is a
    # confound, not a recency effect. This is REQUIRED evidence for a causal claim, not merely reported alongside it.
    carryover_causal_at_near_threshold = bool(near_threshold_dissociation
                                              and nt_recent_lesioned["swapped"] != nt_recent["swapped"])

    # ── FULL-CONVERSATION SAFETY (2026-09-23 review fix): the low-level arms above only ever exercise swap turns.
    # Run the ACTUAL production glue (`ThoughtSwapWorkspace`) through hold turns, the board-#85 lesion path, and LRU
    # slot reuse past N_PATTERNS=3, comparing continuous-ON to continuous-OFF turn-by-turn on the swap-vs-hold
    # VERDICT (not internal numeric state, which legitimately differs once continuity is on).
    full_conv_ok, full_conv_off, full_conv_on = _full_conversation_no_regression(seed)

    if verbose:
        print(f"[swap-continuous-recency] seed={seed} xA_at_branch={xA_at_branch:.4f} xC_at_branch={xC_at_branch:.4f} "
              f"branch_identical={branch_identical} branch_state_hash_match={branch_state_hash_match} "
              f"restore_blind={restore_blind}", flush=True)
        print(f"  @SALIENT_PA={SALIENT_PA:.0f}  recent.swapped={recent['swapped']} (restore={restore_recent['swapped']})  "
              f"fresh.swapped={fresh['swapped']} (restore={restore_fresh['swapped']})  "
              f"no_regression={no_regression_at_production_pa}", flush=True)
        print(f"  @near_threshold_pa={near_threshold_pa:.0f} (diagnostic, NOT gating)  recent.swapped={nt_recent['swapped']}  "
              f"fresh.swapped={nt_fresh['swapped']}  lesioned-recent.swapped={nt_recent_lesioned['swapped']}  "
              f"dissociation={near_threshold_dissociation}  carryover_causal={carryover_causal_at_near_threshold}", flush=True)
        print(f"  full_conversation_no_regression={full_conv_ok} (hold+lesion+LRU-slot-reuse, "
              f"{len(_SAFETY_TURNS)} turns)", flush=True)

    lever("std_x_at_branch (A vs C, both start at 1.0)", 1.0, round(xA_at_branch, 4), continuous=xA_at_branch)

    # NOTE: `branch_state_hash_match` is deliberately NOT in seed_go -- see its own comment above (BLAS-thread-count
    # floating-point artifact under this module's own documented OMP_NUM_THREADS=2 recipe, confirmed bit-identical
    # under OMP_NUM_THREADS=1). Gating on it here would fail this exact recipe for a reason unrelated to the mechanism.
    seed_go = bool(swaps_correct and branch_identical and seed_deterministic
                   and carryover_ok and restore_blind and no_regression_at_production_pa and full_conv_ok)

    return {
        "seed": int(seed),
        "xA_at_branch": float(xA_at_branch), "xC_at_branch": float(xC_at_branch),
        "branch_identical": branch_identical, "branch_state_hash_match": branch_state_hash_match,
        "restore_blind": restore_blind,
        "recent": {k: recent[k] for k in ("swapped", "new_rate_post", "old_residual_post", "b_ignite_step")},
        "fresh": {k: fresh[k] for k in ("swapped", "new_rate_post", "old_residual_post", "b_ignite_step")},
        "restore_recent_swapped": restore_recent["swapped"], "restore_fresh_swapped": restore_fresh["swapped"],
        "seed_deterministic": seed_deterministic,
        "near_threshold_diagnostic": {
            "pa": near_threshold_pa,
            "recent_swapped": nt_recent["swapped"], "fresh_swapped": nt_fresh["swapped"],
            "recent_lesioned_swapped": nt_recent_lesioned["swapped"],
            "dissociation": near_threshold_dissociation,
            "carryover_causal": carryover_causal_at_near_threshold,
        },
        "full_conversation_safety": {
            "ok": full_conv_ok, "turns": _SAFETY_TURNS,
            "off": full_conv_off if not full_conv_ok else None,   # only stash the diff payload on a FAILURE
            "on": full_conv_on if not full_conv_ok else None,
        },
        "go_gate": {
            "swaps_correct": swaps_correct,
            "branch_identical": branch_identical,
            "branch_state_hash_match": branch_state_hash_match,
            "carryover_ok": carryover_ok,
            "restore_blind": restore_blind,
            "no_regression_at_production_pa": no_regression_at_production_pa,
            "seed_deterministic": seed_deterministic,
            "full_conversation_no_regression": full_conv_ok,
        },
        "seed_go": seed_go,
        "operating_point": {"salient_pa": SALIENT_PA, "min_x_deficit": MIN_X_DEFICIT,
                             "near_threshold_pa": near_threshold_pa},
    }


def run_smoke(seed, args):
    r = evaluate_seed(seed, heterogeneity=not args.no_heterogeneity, near_threshold_pa=args.near_threshold_pa, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_swap_continuous_recency_derisk", "mode": "smoke", "seed": seed, "result": r},
                  f, indent=2, default=str)
    print(f"\n[swap-continuous-recency smoke] wrote {args.json}  seed_go={r['seed_go']}  "
          f"near_threshold_dissociation={r['near_threshold_diagnostic']['dissociation']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[swap-continuous-recency six-seed] seeds={seeds}", flush=True)
    per_seed = [evaluate_seed(s, heterogeneity=not args.no_heterogeneity, near_threshold_pa=args.near_threshold_pa,
                              verbose=True) for s in seeds]
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_swap = sum(1 for r in per_seed if r["go_gate"]["swaps_correct"])
    n_branch = sum(1 for r in per_seed if r["go_gate"]["branch_identical"])
    n_branch_hash = sum(1 for r in per_seed if r["go_gate"]["branch_state_hash_match"])
    n_carry = sum(1 for r in per_seed if r["go_gate"]["carryover_ok"])
    n_blind = sum(1 for r in per_seed if r["go_gate"]["restore_blind"])
    n_noreg = sum(1 for r in per_seed if r["go_gate"]["no_regression_at_production_pa"])
    n_det = sum(1 for r in per_seed if r["go_gate"]["seed_deterministic"])
    n_fullconv = sum(1 for r in per_seed if r["go_gate"]["full_conversation_no_regression"])
    n_dissoc = sum(1 for r in per_seed if r["near_threshold_diagnostic"]["dissociation"])
    n_causal = sum(1 for r in per_seed if r["near_threshold_diagnostic"]["carryover_causal"])
    # n_branch_hash is reported, NOT gated (see per-seed comment: a BLAS-thread-count floating-point artifact under
    # this module's own OMP_NUM_THREADS=2 recipe, confirmed bit-identical under OMP_NUM_THREADS=1).
    pooled_go = bool(n_go == 6 and n_swap == 6 and n_branch == 6 and n_carry == 6
                     and n_blind == 6 and n_noreg == 6 and n_det == 6 and n_fullconv == 6)
    verdict = "GO" if pooled_go else ("PARTIAL" if n_go >= 1 else "NO-GO")

    # RETRACTION (2026-09-23 fix round, review verdict=fix-required): the ORIGINAL 6-seed run of this runner called
    # the near-threshold dissociation at seed 42 an "existence proof" of a recency-driven swap failure. It is not.
    # `carryover_causal_at_near_threshold` (rescuing the swap by lesioning ONLY the carryover) is 0/6, INCLUDING at
    # seed 42 where the raw dissociation exists: `nt_recent_lesioned["swapped"] == nt_recent["swapped"]` there too --
    # wiping the carryover back to 1.0 does NOT rescue the swap, so the carryover was never what blocked it. The
    # dissociation is a pattern-identity or per-seed-heterogeneity confound. THIS FINDING NO LONGER CLAIMS A
    # RECENCY-TRACE MECHANISM OF ANY REPRODUCIBILITY, existence-proof included -- see module docstring "RETRACTION".
    mechanism_go = bool(n_causal >= 1)

    v = Verdict("GNW swap continuous cross-turn ignition: 6-seed aggregate (SAFETY only -- recency-trace claim retracted)")
    v.require("all six swap decisions (A first-thought, A->B, RECENT/FRESH continuous, RECENT/FRESH restore) correct on 6/6",
              bool(n_swap == 6), expect=True)
    v.require("the two-arm fork's scalar read-outs match (floating-point tolerance, NOT claimed byte-identical) at "
              "the branch point on 6/6", bool(n_branch == 6), expect=True)
    v.require("the STD carryover lever actually moved (x_A < 1 at branch) on 6/6", bool(n_carry == 6), expect=True)
    v.require("restore mode's own reset provably wipes the carryover to exactly 1.0 on 6/6", bool(n_blind == 6), expect=True)
    v.require("continuous mode's verdict matches a REAL isolate=True restore-mode arm at production drive on 6/6",
              bool(n_noreg == 6), expect=True)
    v.require("continuous mode matches restore mode on hold/lesion/LRU-slot-reuse turns too (full-conversation glue), 6/6",
              bool(n_fullconv == 6), expect=True)
    v.require("determinism (build-twice hash) on 6/6", bool(n_det == 6), expect=True)
    v.disabled("homeostasis", why="frozen base weights, inherited from the reused #77/#85 substrate build")
    v.disabled("recency-trace mechanism", why="RETRACTED: the lesion arm refutes causality on 6/6 seeds "
               "(carryover_causal_at_near_threshold=0/6, including at the one seed with a raw dissociation) -- "
               "see module docstring 'RETRACTION'. Not gated into pooled_go; reported for the record only.")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_swap_continuous_recency_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "mechanism_go": mechanism_go, "seeds": seeds,
               "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "swaps_correct": n_swap, "branch_identical": n_branch,
                          "branch_state_hash_match": n_branch_hash,
                          "carryover_ok": n_carry, "restore_blind": n_blind,
                          "no_regression_at_production_pa": n_noreg, "seed_deterministic": n_det,
                          "full_conversation_no_regression": n_fullconv,
                          "n_seeds": len(seeds)},
               "near_threshold_diagnostic_summary": {
                   "n_dissociation": n_dissoc, "n_carryover_causal": n_causal, "n_seeds": len(seeds),
                   "note": "RETRACTED as a mechanism claim (2026-09-23 fix round): n_carryover_causal=0/6 means the "
                           "lesion arm REFUTES that the STD carryover causes the dissociation seen at seed 42 -- "
                           "diagnostic ONLY, NEVER gates pooled_go, and no longer described as an existence proof."},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[swap-continuous-recency six-seed] verdict={verdict} seed_go {n_go}/6 swap {n_swap}/6 branch {n_branch}/6 "
          f"branch_hash {n_branch_hash}/6 carry {n_carry}/6 blind {n_blind}/6 no_regression {n_noreg}/6 "
          f"full_conv {n_fullconv}/6 det {n_det}/6  "
          f"[near-threshold dissociation {n_dissoc}/6, carryover_causal {n_causal}/6 -- RETRACTED as a mechanism, "
          f"NOT gating]", flush=True)
    print(f"[swap-continuous-recency six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW swap continuous cross-turn ignition: is removing the per-turn "
                                             "restore SAFE, and does it carry a genuine synaptic recency trace?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--six-seed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-threshold-pa", type=float, default=NEAR_THRESHOLD_PA)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_swap_continuous_recency.json")
    args = ap.parse_args()
    if args.six_seed:
        return run_six_seed(args)
    return run_smoke(args.seed, args)


if __name__ == "__main__":
    raise SystemExit(main())
