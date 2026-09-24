"""GNW CONTINUOUS-SWAP BRANCH-POINT VERIFICATION INSTRUMENT — the named next attempt after the lane was PARKED
(2026-09-23, docs/.vikunja_sync "GNW continuous-thought-swap lane PARKED after 3 review rounds"): "next attempt (when
resumed) verifies against a genuine branch-point snapshot of every cp_* array." This runner builds and pre-registers
exactly that instrument. It does NOT re-open the retracted recency-causality claim (banked NO-GO,
`carryover_causal_at_near_threshold`=0/6 in the parked `_gnw_swap_continuous_recency_derisk.py`, never merged) — it
verifies the FORK MECHANISM the parked lane's own safety claim depended on, at the SHIPPED-adjacent safety operating
point (SALIENT_PA), and reports whatever the divergent arms show honestly.

WHY THE PARKED LANE'S OWN FORK CONTROL WAS NOT GENUINE (read directly off the parked branch,
`origin/research/gnw-thought-swap-drive:research/runners/_gnw_swap_continuous_recency_derisk.py`, never merged to
main): its `_fresh_branch` BUILT TWO INDEPENDENT SUBSTRATES from the same seed and HOPED they matched, verified only
by `_std_state_hash` — a hash of `std.deps[*].x` alone, ONE array family out of the ~30 live `cp_*` dynamical arrays
this substrate actually carries (measured below, `cp_completeness`: e.g. `cp_conductance_g_*`, `cp_izh_v`/`cp_izh_u`
membrane state, `cp_connections.data` — the post-STD-depression synaptic weights STD.apply() writes into — `cp_ou_current`,
none of which the parked hash ever touched). ITS OWN docstring records two retracted fix rounds inside that one
narrow hash (1st: hashed AFTER the two arms had already diverged, which cannot match by construction and was
misdiagnosed as a BLAS-thread artifact; 2nd: fixed the timing but never widened the scope). A 3rd review round
refuted the 2nd fix round too (reproduced-here scope problem: matching 4 population-level SCALARS, or even one
per-loop STD array, at a point where two INDEPENDENTLY BUILT substrates merely "happen to still agree" is not a
fork — it is an unverified coincidence of construction, and nothing in that runner ever forced two substrates into
the SAME state and proved it).

THE FIX (reuse-by-import, NO re-derivation, NO `sim/` edit): `_full_snapshot`/`_full_restore`
(`research.runners._p1_2_workspace_deliberation_loop_derisk`) is the EXISTING, already-shipped, already-GO-status
generic snapshot utility — it walks `vars(bridge)` and captures EVERY attribute whose name starts `cp_` and has
BOTH `.copy()` and `.shape` (i.e. every live array-like dynamical state, sparse `cp_connections` included via its
own `.copy()`). It already underpins the production `isolate=True` restore path in the SHIPPED #77/#85 mechanism.
The parked lane never used it for its OWN fork control — this runner does, and adds the two pieces that were still
missing for a GENUINE fork:
  1. `cp_completeness`: a runtime audit (not an assumption) that NO live (non-None) `cp_*` attribute on the actual
     built substrate is silently excluded by `_full_snapshot`'s filter. Measured once per seed below: 32/32 live
     arrays captured on this substrate at the time of writing (126 further `cp_*` names exist on the bridge class
     but read `None` here — HH channels, STDP/BCM/homeostasis/BTSP/structural-plasticity traces — because this
     substrate's own `build()` leaves those subsystems OFF (`enable_stdp=False` etc, unchanged, reused-by-import);
     `cp_completeness` FAILS if a future edit to `build()` turns one of those subsystems on without this filter
     also picking up its new live array.
  2. THE RNG STREAM: OU noise (`cfg.enable_ou_process`) draws `cp.random.randn(n)` every step from the INTERPRETER-
     GLOBAL RNG (`sim/bridge.py` — "the legacy global cp.random.randn(n) per-step draw"), not from a per-object
     generator captured by any `cp_*` array. Two forks restored to identical `cp_*`+STD state will still DIVERGE the
     instant either one takes a noisy step, unless the global RNG state is ALSO snapshotted at the branch point and
     explicitly re-loaded before each arm's continuation. This runner captures `np.random.get_state()` alongside the
     `cp_*`/STD snapshot and treats it as part of "the branch point" — a genuine branch point is the FULL state that
     determines every future step, not merely the state that happens to have a `cp_` prefix.

THE INSTRUMENT (one substrate BUILT ONCE, the true fork — not two hopeful independent builds):
  Build S1 once; drive it, continuously (one necessary cold-start `isolate=True` for the very first thought), through
  establish-A -> evict/admit-B (`_fresh_branch`, imported unchanged in spirit from the parked runner but now called
  on ONE substrate only). At the branch point (B held, A just evicted, BEFORE either arm's next, diverging step):
    `branch_snapshot` = {cp: `_full_snapshot(S1.bridge)`, std: per-loop `(x, boost)` copies, rng: `np.random.get_state()`
    copied}. `branch_hash` = sha256 over every `cp_*` array (sparse `cp_connections` hashed via `.data`/`.indices`/
    `.indptr`) + every STD array, sorted by key (deterministic ordering).
  FORK: build S2 and S3 fresh (same seed, same architecture — matching shapes are all `_full_restore` needs) and
  FORCE-RESTORE `branch_snapshot` onto each — this is the actual fork operation the parked lane never performed; S1
  is not touched by this step and remains, itself, the third leg of the fork.

GATES (pre-registered here, BEFORE this file's first run — `docs/BUILD_LANE_CHECKLIST.md`). Review round 2
(2026-09-24, journal wf_16244c9d-ca5 agentId a6dd960fd33710903) found that three of these pass BY CONSTRUCTION
on this substrate/backend and were nonetheless being counted as evidence — `docs/BUILD_LANE_CHECKLIST.md`:
"if a check passes by construction ... it is an INTEGRITY SMOKE — label it so, never count it as evidence."
They are relabelled below. **G3 is the ONLY gate that can actually tell a good fork from a bad one on this
substrate**, and it is now paired with a pre-registered NEGATIVE CONTROL (G3-neg) so a pass is evidence rather
than an artifact of a control that was never exercised in its failing direction.

  G1 cp_completeness [INTEGRITY SMOKE] — measures whether any non-None `cp_*` attribute on the built bridge is
                             absent from `_full_snapshot`'s output. On THIS substrate this was already measured
                             32/32 (0 missing) BEFORE the prereg (`scratchpad/probe_cp_types.py`, committed with
                             this file) and `build()` is unchanged by this instrument, so today's re-measurement
                             cannot show a different number — it is a regression check against a future edit to
                             `build()` (e.g. turning on STDP/BTSP/HH channels), not a check that can fail today.
                             Kept as an executable measurement (not an assumption) precisely so that future edit
                             would be caught, per this same checklist's "it must be able to fail" for FUTURE runs
                             of this file, even though it is not live evidence for THIS pre-registration.
  G2 fork_restore_exact [INTEGRITY SMOKE] — re-snapshots S2 (resp. S3) immediately after `_full_restore` writes
                             `branch_snapshot` onto it, using the SAME key set and the SAME filter that produced
                             `branch_snapshot` in the first place. `_full_restore` assigns `getattr(bridge, k)[:]
                             = arr` (or a CSR in-place restore) for each key; reading the same keys back with the
                             same filter reproduces exactly what was just written, and a genuine shape/key
                             mismatch would raise an exception in `_full_restore` itself, not surface as a False
                             here. There is no code path under which `_full_restore` writes something silently
                             different from what it was given and G2 still returns True — so a hash match is not
                             evidence the restore is scientifically correct, only that the assignment executed
                             without raising. Kept as an INTEGRITY SMOKE (crash-vs-silent-corruption check), not
                             evidence that the fork is genuine — that claim rests on G3.
  G3 control_fork_determinism [THE DISCRIMINATING GATE] — FAILS if, from the two forks (S1 and S2, each merely
                             G2-consistent, not yet shown scientifically identical), loading the SAME saved RNG
                             state before each and then running the IDENTICAL control action (propose A again,
                             i.e. "recent") on both does NOT produce byte-identical post-run state (a fresh hash
                             of each substrate after the run) and identical verdict fields. This is the one check
                             in this file with a demonstrated failing direction (see G3-neg immediately below): a
                             pass is the genuine claim "S1 and S2 are, operationally, the same branch."
  G3-neg negative control for G3 [REQUIRED, proves G3 is not vacuous] — the SAME control action, from the SAME
                             restored branch point, on a FOURTH fork (S4), but WITHOUT reloading the saved RNG
                             state first (OU noise, `cfg.enable_ou_process`, draws `cp.random.randn(n)` every step
                             from the INTERPRETER-GLOBAL stream — see "THE RNG STREAM" above). If G3 were vacuous
                             (state_match always True regardless of RNG), this negative control would ALSO show a
                             match. `g3_negative_control_mismatches` must be True — i.e. the un-RNG-matched run
                             DIVERGES from S1's post-run hash — or G3's pass this seed is not evidence of anything
                             and `seed_go` is False regardless of G3's own result.
  G4 no_aliasing [INTEGRITY SMOKE on `SIM_BACKEND=numpy`] — checks the STORED `branch_snapshot` dict's own hash is
                             unchanged after S1 and S2 are run forward past the branch point. This is the direct
                             regression test for the parked lane's 1st-fix-round bug class (a hash computed from
                             arrays that turned out to reference the live, now-diverged substrate). On the numpy
                             backend this cannot fail: `numpy.ndarray.copy()` and `scipy.sparse.csr_matrix.copy()`
                             are always genuine deep copies (verified standalone in `scratchpad/probe_cp_types.py`,
                             committed with this file — mutate-after-copy leaves the copy untouched), so
                             `_full_snapshot`'s `.copy()` cannot alias on this backend. Kept as an executable
                             regression check (the exact failure mode that sent the parked lane to a 3rd
                             retraction), but labelled an INTEGRITY SMOKE, not evidence, because it is not
                             expected to be able to fail under the backend this file asserts (see below).
  G5 carryover_ok          — FAILS if `x_A` at the branch point is not measurably below 1.0 (the STD lever the
                             branch point is supposed to exercise never actually moved). Uses `tools.lab.lever`
                             with `required=False`: a lever that did not move is a real, reportable NO-GO for
                             this seed, not a process crash that drops the whole aggregate (see "the lever fix"
                             below).
  A seed's `seed_go` = G1 AND G2(S2) AND G2(S3) AND G3 AND G3-neg AND G4 AND G5, all True. `pooled_go` = seed_go
  on 6/6 seeds. Of these, only G3, G3-neg and G5 have a realistic failing outcome on this substrate/backend
  today; G1, G2 and G4 are executable regression checks against future edits, not live evidence, and are
  reported and gated but must not be cited as "the fork was verified" on their own.

  THE LEVER FIX (review round 2, item 4): the previous draft called `tools.lab.lever(..., required=True)` for
  G5, which raises `LeverError` the instant `x_A == 1.0` exactly — inside `--six-seed` mode's list comprehension,
  one such seed would abort the WHOLE aggregate with no artifact written for any seed, turning a legitimate
  per-seed NO-GO into a silent process crash. `lever(..., required=False)` is used instead; the boolean it
  returns is recorded as `lever_moved` and `carryover_ok` is computed independently from the same threshold, so
  a seed where the lever did not move is recorded as `seed_go=False` with a full artifact, not a crash.

  BACKEND ASSERTION (review round 2, minor item): the RNG capture (`np.random.get_state()`/`np.random.set_state`)
  is only actually the stream OU noise draws from when `cp is np`, i.e. `SIM_BACKEND=numpy`. Under CuPy, OU noise
  draws from `cp.random`, a SEPARATE stream `np.random.set_state` does not touch, so G3/G3-neg would read as
  false failures/passes for the wrong reason (mis-configuration, not the instrument). `main()` asserts the
  backend below and exits loudly rather than letting that happen silently.

WHAT THIS RUNNER DOES NOT CLAIM. This is a VERIFICATION-INSTRUMENT finding, not a mechanism finding: it does not
re-open `carryover_causal_at_near_threshold` (still banked NO-GO on the parked, never-merged branch) and does not
assert the continuous-mode flag is production-load-bearing. `recent`/`fresh` swap verdicts at the shipped safety
operating point (SALIENT_PA) are reported alongside the gates for continuity with the parked lane's own safety
claim, but `seed_go`/`pooled_go` are gated ONLY on G1-G5 plus the G3-neg negative control (the instrument), per
this file's actual mandate.

POOL AGGREGATION (review round 2, minor item): per the compute plan, seed 42 runs LOCAL `--smoke` and
seeds 43/44/100/101/102 run as five separate `--smoke` pool jobs (one seed each, so each writes its own
per-seed JSON via `run_smoke`/`evaluate_seed` — there is no `--six-seed` invocation on the pool, and no single
process ever holds all 6 seeds at once). `pooled_go` is therefore NOT produced by any single run of this file
for a pool-staged battery: it must be assembled AFTER all 6 per-seed JSON files exist, by loading each file's
`result` dict and applying the same `n_go == 6` / `Verdict.require` logic `run_six_seed` applies in-process.
This runner does not do that assembly itself (it has no way to know when the pool jobs finish); the orchestrator
collecting the 6 artifacts is responsible for it, exactly as it is for any other pool-staged 6-seed battery.

Corpus check (`before_you_build.sh "GNW continuous branch point snapshot fork"` + `rag_search.py` finding/plan
corpora) was run before writing this file, alongside a direct empirical probe, now committed at
`scratchpad/probe_cp_types.py` (review round 2, item 3 — it was run before this file was written but not
originally committed, which left its preconditions unverifiable once the worktree was discarded), building this
exact substrate and enumerating every `cp_*` attribute's type — confirming EVERY name absent from `_full_snapshot`'s
output on THIS substrate reads `None` (a disabled subsystem), not a silently-dropped live array; and a standalone
scipy-sparse check that `csr.copy()` genuinely deep-copies `.data` (mutating the original does not touch the copy)
before relying on that behavior for G4. Re-run 2026-09-24 at commit-time: `included=32 missing=126` (all 126
`None`), `csr copy independent: same_data_object=False dup_unaffected_by_mutation=True` — unchanged from the
original pre-prereg measurement.

Biology: unchanged from the shipped mechanism this instrument verifies the fork of — Tsodyks-Markram short-term
depression (Mongillo, Barak & Tsodyks 2008, Science 319:1543) for the STD eviction effector; no new biological claim
is made by the verification instrument itself (it is scaffolding for measuring the substrate faithfully, per
CLAUDE.md's "the instrument is part of the emulation").

Usage (CPU cheap-first; export OMP/OPENBLAS/MKL_NUM_THREADS=2; run under tools/memcap.sh per
docs/BUILD_LANE_CHECKLIST.md):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_continuous_branchpoint_verify --smoke --seed 42 \\
      --json research/findings/raw/_gnw_continuous_branchpoint_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_continuous_branchpoint_verify --six-seed \\
      --json research/findings/raw/_gnw_continuous_branchpoint_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim.backend import to_host
from tools.verdict import Verdict
from tools.lab import lever

from research.runners._gnw_neural_swap_intention_derisk import (
    build, run_intention_swap, MultiLoopSTD, SALIENT_PA, N_PATTERNS, W_REC,
)
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore

A, B, C = 0, 1, 2
assert N_PATTERNS >= 3, "this probe needs >=3 disjoint pattern slots (A, B, and a never-touched C)"

MIN_X_DEFICIT = 0.05   # x_A_at_branch must be at least this far below 1.0 (the carryover lever must have moved)


# ── the genuine full-state snapshot: every cp_* array (`_full_snapshot`, reused unchanged) + the STD host state
#    (NOT itself cp_-prefixed -- it is host bookkeeping that DRIVES cp_connections.data via std.apply()) + the
#    interpreter-global RNG state (OU noise draws from it every step; see module docstring "THE RNG STREAM") ────────
def _array_bytes(v):
    """Byte-serialize one snapshot value for hashing. `cp_connections` is a sparse CSR matrix (scipy or cupyx) --
    hash its structural arrays directly rather than via a dense `.toarray()` (which would blow up memory on a real
    substrate and is unnecessary: STD depression only ever rewrites `.data`, never the sparsity pattern, but hashing
    `.indices`/`.indptr` too costs nothing and catches a structural change if one ever occurred)."""
    if hasattr(v, "indices") and hasattr(v, "indptr") and hasattr(v, "data"):
        d = np.asarray(to_host(v.data), dtype=np.float64)
        i = np.asarray(to_host(v.indices), dtype=np.int64)
        p = np.asarray(to_host(v.indptr), dtype=np.int64)
        return d.tobytes() + i.tobytes() + p.tobytes()
    arr = np.asarray(to_host(v), dtype=np.float64)
    return arr.tobytes()


def _copy_rng_state(state):
    kind, arr, *rest = state
    return (kind, np.asarray(arr).copy(), *rest)


def _std_snapshot(std):
    return [{"x": d.x.copy(), "boost": float(d.boost)} for d in std.deps]


def _std_restore(std, snap):
    for d, s in zip(std.deps, snap):
        d.x[:] = s["x"]
        d.boost = float(s["boost"])


def _full_bio_snapshot(bridge, std):
    """THE branch point: every cp_* array + every STD loop's (x, boost) + the global RNG state. This is what
    'genuine branch-point snapshot of every cp_* array' (the parked lane's named next attempt) means in full --
    the cp_* half by itself is necessary but not sufficient (see module docstring)."""
    return {"cp": _full_snapshot(bridge), "std": _std_snapshot(std), "rng": _copy_rng_state(np.random.get_state())}


def _bio_hash(cp_snap, std_snap):
    h = hashlib.sha256()
    for k in sorted(cp_snap.keys()):
        h.update(k.encode())
        h.update(_array_bytes(cp_snap[k]))
    for i, s in enumerate(std_snap):
        h.update(("std_%d_x" % i).encode())
        h.update(np.asarray(s["x"], dtype=np.float64).tobytes())
        h.update(("std_%d_boost" % i).encode())
        h.update(np.asarray([s["boost"]], dtype=np.float64).tobytes())
    return h.hexdigest()


def _restore_bio_snapshot(bridge, std, snap):
    _full_restore(bridge, snap["cp"])
    _std_restore(std, snap["std"])


def _cp_completeness(bridge, cp_snap):
    """G1: every non-None cp_* attribute on the LIVE bridge must be present in the snapshot dict. Returns
    (ok, missing_names) -- 'missing' would mean _full_snapshot's generic .copy()/.shape filter dropped a live array
    this substrate actually carries; NOT expected to ever fire on the substrate `build()` produces today (verified
    empirically before writing this file), but measured every run, not assumed."""
    missing = [k for k, v in vars(bridge).items()
               if k.startswith("cp_") and v is not None and k not in cp_snap]
    return (len(missing) == 0), missing


def _fresh_branch(seed, w_rec, heterogeneity):
    """Build ONE substrate and drive it, continuously (one necessary cold-start isolate=True for the very first
    thought, then zero restores), through establish-A -> evict/admit-B. Returns (S, std, first, ab) with S/std AT
    the branch point: B is held, A was JUST evicted."""
    S = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std = MultiLoopSTD(S["bridge"], S["xp"], S["ws_used"], S["patterns_host"])
    first = run_intention_swap(S, std, incumbent=A, proposed=A, proposal_pa=SALIENT_PA, isolate=True)
    ab = run_intention_swap(S, std, incumbent=A, proposed=B, proposal_pa=SALIENT_PA, isolate=False)
    return S, std, first, ab


def _verdict_fields(r):
    return {k: r[k] for k in ("swapped", "new_rate_post", "old_residual_post", "winner_post", "n_ignited_post")}


def evaluate_seed(seed, *, w_rec=None, heterogeneity=True, verbose=True):
    if w_rec is None:
        w_rec = W_REC

    # ── ONE substrate, driven to the branch point -- the fork SOURCE (not one of several independent builds) ──────
    S1, std1, first1, ab1 = _fresh_branch(seed, w_rec, heterogeneity)
    xA_at_branch = std1.x_mean(A)
    branch_snap = _full_bio_snapshot(S1["bridge"], std1)
    branch_hash = _bio_hash(branch_snap["cp"], branch_snap["std"])

    # ── G1: cp_completeness -- measured on the LIVE substrate at the branch point ───────────────────────────────
    cp_ok, cp_missing = _cp_completeness(S1["bridge"], branch_snap["cp"])

    # ── FORK: build S2/S3 fresh (architecture-matching only; their own pre-restore state is discarded) and FORCE
    #    the captured branch point onto each -- the actual fork operation the parked lane never performed ─────────
    S2 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std2 = MultiLoopSTD(S2["bridge"], S2["xp"], S2["ws_used"], S2["patterns_host"])
    _restore_bio_snapshot(S2["bridge"], std2, branch_snap)
    resnap2 = _full_bio_snapshot(S2["bridge"], std2)
    hash2_after_restore = _bio_hash(resnap2["cp"], resnap2["std"])
    fork_restore_exact_s2 = bool(hash2_after_restore == branch_hash)

    S3 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std3 = MultiLoopSTD(S3["bridge"], S3["xp"], S3["ws_used"], S3["patterns_host"])
    _restore_bio_snapshot(S3["bridge"], std3, branch_snap)
    resnap3 = _full_bio_snapshot(S3["bridge"], std3)
    hash3_after_restore = _bio_hash(resnap3["cp"], resnap3["std"])
    fork_restore_exact_s3 = bool(hash3_after_restore == branch_hash)

    # ── G3: control_fork_determinism -- the SAME control action (recent = propose A) on S1 and on the S2 fork,
    #    with the SAME saved RNG state loaded before EACH run (OU noise draws from the interpreter-global stream;
    #    without this the two arms diverge on noise alone, not on anything the fork actually got wrong) ────────────
    np.random.set_state(_copy_rng_state(branch_snap["rng"]))
    recent1 = run_intention_swap(S1, std1, incumbent=B, proposed=A, proposal_pa=SALIENT_PA, isolate=False)
    post_hash_1 = _bio_hash(_full_snapshot(S1["bridge"]), _std_snapshot(std1))

    np.random.set_state(_copy_rng_state(branch_snap["rng"]))
    recent2 = run_intention_swap(S2, std2, incumbent=B, proposed=A, proposal_pa=SALIENT_PA, isolate=False)
    post_hash_2 = _bio_hash(_full_snapshot(S2["bridge"]), _std_snapshot(std2))

    control_state_match = bool(post_hash_1 == post_hash_2)
    control_verdict_match = bool(_verdict_fields(recent1) == _verdict_fields(recent2))
    control_fork_determinism = bool(control_state_match and control_verdict_match)

    # ── G3-neg: the pre-registered NEGATIVE CONTROL for G3 (review round 2, item 2). The SAME control action,
    #    from the SAME restored branch point, on a FOURTH fork (S4) -- but WITHOUT reloading the saved RNG state
    #    first. OU noise (`cfg.enable_ou_process`) draws `cp.random.randn(n)` every step from the interpreter-
    #    global stream, which by this point has already been advanced past `branch_snap["rng"]` by recent1's and
    #    recent2's own steps above. If G3 were vacuous (state_match always True regardless of RNG), this would
    #    ALSO match; it must NOT, or G3's pass on this seed is not evidence that the RNG capture is load-bearing.
    S4 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std4 = MultiLoopSTD(S4["bridge"], S4["xp"], S4["ws_used"], S4["patterns_host"])
    _restore_bio_snapshot(S4["bridge"], std4, branch_snap)
    recent4 = run_intention_swap(S4, std4, incumbent=B, proposed=A, proposal_pa=SALIENT_PA, isolate=False)
    post_hash_4 = _bio_hash(_full_snapshot(S4["bridge"]), _std_snapshot(std4))
    g3_negative_control_mismatches = bool(post_hash_4 != post_hash_1)

    # ── G4: no_aliasing -- the STORED branch_snap dict must be UNCHANGED now that S1 and S2 have each been driven
    #    past the branch point. This is the direct regression test for the parked lane's 1st-fix-round bug class
    #    (a hash taken from arrays that turned out to reference the live, now-diverged substrate). ─────────────────
    branch_hash_recheck = _bio_hash(branch_snap["cp"], branch_snap["std"])
    no_aliasing = bool(branch_hash_recheck == branch_hash)

    # ── the actual divergent readout: FRESH (propose C, never touched) on the S3 fork, RNG-matched to the branch
    #    point same as the other two arms (reported for continuity with the parked lane's safety claim; NOT what
    #    seed_go gates on -- see module docstring "WHAT THIS RUNNER DOES NOT CLAIM"). ──────────────────────────────
    np.random.set_state(_copy_rng_state(branch_snap["rng"]))
    fresh3 = run_intention_swap(S3, std3, incumbent=B, proposed=C, proposal_pa=SALIENT_PA, isolate=False)

    # G5 (the lever fix, review round 2, item 4): required=False -- a lever that did not move is a real,
    # reportable NO-GO for this seed, not a process crash that aborts the whole --six-seed aggregate.
    lever_moved = lever("std_x_at_branch (A vs C, both start at 1.0)", 1.0, round(xA_at_branch, 4),
                         required=False, continuous=xA_at_branch)
    carryover_ok = bool(xA_at_branch < 1.0 - MIN_X_DEFICIT)

    if verbose:
        print(f"[branchpoint-verify] seed={seed} xA_at_branch={xA_at_branch:.4f} cp_completeness={cp_ok} "
              f"(missing={cp_missing if not cp_ok else 0}) fork_restore_exact S2={fork_restore_exact_s2} "
              f"S3={fork_restore_exact_s3}", flush=True)
        print(f"  control_fork_determinism={control_fork_determinism} (state_match={control_state_match} "
              f"verdict_match={control_verdict_match})  g3_negative_control_mismatches="
              f"{g3_negative_control_mismatches}  no_aliasing={no_aliasing}", flush=True)
        print(f"  recent(S1).swapped={recent1['swapped']}  recent(S2-fork).swapped={recent2['swapped']}  "
              f"recent(S4-fork,no-RNG-reload).swapped={recent4['swapped']}  "
              f"fresh(S3-fork).swapped={fresh3['swapped']}", flush=True)

    seed_go = bool(cp_ok and fork_restore_exact_s2 and fork_restore_exact_s3
                   and control_fork_determinism and g3_negative_control_mismatches
                   and no_aliasing and carryover_ok)

    return {
        "seed": int(seed),
        "xA_at_branch": float(xA_at_branch),
        "branch_hash": branch_hash,
        "lever_moved": lever_moved,
        "go_gate": {
            "cp_completeness": cp_ok,
            "fork_restore_exact_s2": fork_restore_exact_s2,
            "fork_restore_exact_s3": fork_restore_exact_s3,
            "control_fork_determinism": control_fork_determinism,
            "g3_negative_control_mismatches": g3_negative_control_mismatches,
            "no_aliasing": no_aliasing,
            "carryover_ok": carryover_ok,
        },
        "cp_missing": cp_missing,
        "control": {"state_match": control_state_match, "verdict_match": control_verdict_match},
        "negative_control": {"mismatches": g3_negative_control_mismatches,
                              "note": "S4, restored to the same branch point as S2, run WITHOUT reloading the "
                                      "saved RNG state -- must MISMATCH S1's post-run hash or G3 is not evidence"},
        "recent_s1": _verdict_fields(recent1), "recent_s2_fork": _verdict_fields(recent2),
        "recent_s4_fork_no_rng_reload": _verdict_fields(recent4),
        "fresh_s3_fork": _verdict_fields(fresh3),
        "seed_go": seed_go,
        "operating_point": {"salient_pa": SALIENT_PA, "min_x_deficit": MIN_X_DEFICIT},
    }


def run_smoke(seed, args):
    r = evaluate_seed(seed, heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_continuous_branchpoint_verify", "mode": "smoke", "seed": seed, "result": r},
                  f, indent=2, default=str)
    print(f"\n[branchpoint-verify smoke] wrote {args.json}  seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[branchpoint-verify six-seed] seeds={seeds}", flush=True)
    per_seed = [evaluate_seed(s, heterogeneity=not args.no_heterogeneity, verbose=True) for s in seeds]
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_cp = sum(1 for r in per_seed if r["go_gate"]["cp_completeness"])
    n_restore_s2 = sum(1 for r in per_seed if r["go_gate"]["fork_restore_exact_s2"])
    n_restore_s3 = sum(1 for r in per_seed if r["go_gate"]["fork_restore_exact_s3"])
    n_control = sum(1 for r in per_seed if r["go_gate"]["control_fork_determinism"])
    n_negctrl = sum(1 for r in per_seed if r["go_gate"]["g3_negative_control_mismatches"])
    n_noalias = sum(1 for r in per_seed if r["go_gate"]["no_aliasing"])
    n_carry = sum(1 for r in per_seed if r["go_gate"]["carryover_ok"])
    pooled_go = bool(n_go == len(seeds))
    verdict = "GO" if pooled_go else ("PARTIAL" if n_go >= 1 else "NO-GO")

    v = Verdict("GNW continuous-swap branch-point verification instrument: 6-seed aggregate "
                "(instrument-only -- no mechanism/recency claim is made or re-opened here)")
    v.require("[integrity smoke] no live cp_* array is dropped by the snapshot filter, 6/6",
              bool(n_cp == len(seeds)), expect=True)
    v.require("[integrity smoke] the branch-point snapshot restores exactly (hash match) onto an "
              "independently-built fork S2, 6/6", bool(n_restore_s2 == len(seeds)), expect=True)
    v.require("[integrity smoke] ...and onto a second independent fork S3, 6/6",
              bool(n_restore_s3 == len(seeds)), expect=True)
    v.require("[the discriminating gate] the identical control action on the two verified forks (RNG-matched) "
              "reproduces byte-identical post-run state and verdict, 6/6", bool(n_control == len(seeds)),
              expect=True)
    v.require("[negative control] the SAME control action WITHOUT reloading the saved RNG state MISMATCHES the "
              "RNG-matched run, 6/6 -- proves the control above is not vacuous", bool(n_negctrl == len(seeds)),
              expect=True)
    v.require("[integrity smoke on SIM_BACKEND=numpy] the stored branch snapshot is unchanged after both forks "
              "are driven past the branch point (no aliasing), 6/6", bool(n_noalias == len(seeds)), expect=True)
    v.require("the STD carryover lever actually moved (x_A < 1 at branch) on 6/6", bool(n_carry == len(seeds)), expect=True)
    v.disabled("recency-trace mechanism claim", why="OUT OF SCOPE for this runner -- still banked NO-GO on the "
               "parked, never-merged branch (carryover_causal_at_near_threshold=0/6); this instrument verifies the "
               "FORK the parked lane's safety claim depended on, nothing more")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_continuous_branchpoint_verify", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "cp_completeness": n_cp, "fork_restore_exact_s2": n_restore_s2,
                          "fork_restore_exact_s3": n_restore_s3, "control_fork_determinism": n_control,
                          "g3_negative_control_mismatches": n_negctrl,
                          "no_aliasing": n_noalias, "carryover_ok": n_carry, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[branchpoint-verify six-seed] verdict={verdict} seed_go {n_go}/{len(seeds)} cp {n_cp}/{len(seeds)} "
          f"restore_s2 {n_restore_s2}/{len(seeds)} restore_s3 {n_restore_s3}/{len(seeds)} "
          f"control {n_control}/{len(seeds)} negctrl {n_negctrl}/{len(seeds)} "
          f"no_alias {n_noalias}/{len(seeds)} carry {n_carry}/{len(seeds)}",
          flush=True)
    print(f"[branchpoint-verify six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW continuous-swap branch-point verification instrument: does a "
                                             "genuine full cp_*+STD+RNG snapshot/fork/compare at the true branch "
                                             "point actually hold, byte-identically, on 6 seeds?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--six-seed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_continuous_branchpoint.json")
    args = ap.parse_args()

    # Backend assertion (review round 2, minor item): G3/G3-neg's RNG capture is `np.random.get_state()` /
    # `np.random.set_state()`. That is only actually the stream OU noise reads from when `cp is np`
    # (`SIM_BACKEND=numpy`). Under CuPy, OU noise draws from `cp.random`, a separate stream this file never
    # touches, so G3 would spuriously FAIL (looks like a broken fork) and G3-neg would spuriously PASS (looks
    # like a good negative control) for a reason that has nothing to do with the instrument. Fail loudly as
    # MIS-CONFIGURED rather than let either read as a real result.
    # Check the RESOLVED backend, not the env string (re-review 2026-09-24): sim/backend.py resolves an UNSET
    # SIM_BACKEND to "auto", which picks CuPy when it is importable, so reading unset as numpy let a GPU run through.
    from sim.backend import get_backend
    _xp, backend = get_backend()
    if _xp is not np:
        raise SystemExit(
            f"[branchpoint-verify] MIS-CONFIGURED: resolved backend {backend!r} "
            f"(SIM_BACKEND={os.environ.get('SIM_BACKEND', '<unset>')!r}) -- this instrument's RNG capture "
            f"(np.random.get_state/set_state) only controls the stream OU noise actually reads from under "
            f"SIM_BACKEND=numpy. Re-run with SIM_BACKEND=numpy.")

    if args.six_seed:
        return run_six_seed(args)
    return run_smoke(args.seed, args)


if __name__ == "__main__":
    raise SystemExit(main())
