"""A10 (midnight plan S15c, 2026-09-24; fix round after the adversarial review of 58c400ff6) -- a PREDICTION-ERROR
(surprise) SALIENCE input to the SNc reward/context afferent of `da_mode_drives_chat`, read off an EXISTING spiking
population: the surprise organ's confirm/violate mismatch rate.

WHAT THIS IS, AND WHAT IT IS NOT (the honesty boundary; the flag names are the plan's, the claim is narrower).
The flags are called `BRAIN_REWARD_VALUE_AFFERENT` / `BRAIN_REWARD_VALUE_LESION` because the midnight plan names
them so. The signal is NOT a reward VALUE: it is the UNSIGNED magnitude of a spiking prediction-error read (how
strongly the mismatch pool fires when an assertion violates what the brain expected). It carries no sign, no
better-or-worse-than-expected, no value of an outcome. The accurate name is a prediction-error SALIENCE afferent.

WHAT THE PRE-EXISTING AFFERENT ALREADY IS (corrected; the first version of this docstring called it "the host
`engagement_of()` scalar", which understated it). With this flag off, the per-turn SNc input is:
  spiking short-term-depression NOVELTY organ (spiking)  +  content-word RICHNESS (host token count)
  -> `engagement_of()` weighted mix (host arithmetic)
  -> the shared spiking ASK-pool SALIENCE afferent (spiking transduction, default-ON)
  -> the persistent engagement EMA (host) -> `ema * _MAX_AFFERENT_PA` (host linear map) -> the spiking SNc.
It cannot tell a CONFIRMING assertion from a CONTRADICTING one: "the dog chase the cat" and "the dog chase the fish"
are three fresh content words each, so both turns read the same engagement.

WHAT THIS MODULE CHANGES (when `BRAIN_REWARD_VALUE_AFFERENT=1`). On a turn that carries an expectation-bearing
assertion (see the SVO gate below) AND for which the brain recalls a stored patient, the surprise organ's spiking
mismatch rate, normalized against the organ's OWN calibrated threshold, REPLACES the `engagement_of()` mix as the
per-turn signal. It then goes through the SAME downstream stages as before: the shared spiking salience afferent,
the persistent EMA, the existing EMA->pA map and the spiking SNc (`DaModeDrivesWorkspace.observe(...,
turn_signal_override=...)`). The spiking novelty organ still runs on that turn (its habituation state stays
continuous) and is recorded, but its value is not the source. On every other turn (no assertion, nothing recalled,
the surprise organ disabled, or an error) nothing is overridden and the pre-existing path runs unchanged, including
the content-free-turn HOLD.

THE FIRST VERSION'S AFFECT-VALENCE FALLBACK IS REMOVED. It fired on nearly every non-assertion turn once
affect-drives had run (default-ON, earlier in the same handler), bypassed the spiking novelty organ, the shared
salience afferent, the EMA and the content-free HOLD, and was never tested (an empty message drove 1120 pA /
'arousal' in the reviewer's control). It is not replaced.

THE PATH FROM TEXT TO SNc CURRENT (declared, including the parts that are NOT brain-based):
  1. `extract_assertion(message)` -- a REGEX tokenizer + a fixed function-word list + a fixed WH-word list + a '?'
     test + an exact-3-content-token rule. This IS a keyword/regex classifier ON the path: it decides whether the
     surprise read can drive the SNc on this turn at all. Declared in docs/SCAFFOLD-LEDGER.md; it needs an explicit
     owner waiver before this path can pass the S15(c) "no keyword or regex classifier" trace.
  2. `chat.inner.what_does(agent, action)` -- the recalled expected patient. Under the production default composer
     (`BRAIN_COMPOSER_KIND` unset -> onebrain, the spiking DG-CA3 OneBrainComposer) it is a spiking recall; under
     `BRAIN_COMPOSER_KIND=rf` it is the HOST closed-form RFPhasorComposer. The composer class is recorded per read.
  3. `SurpriseProductionOrgan.read_surprise` ADDRESSES the circuit in HOST code before any spiking happens (this
     is inherited from the organ, and it is a hand-designed concept code on the path, not neutral bookkeeping):
       a. the recalled patient string and the asserted patient string are mapped to circuit blocks through the
          host dict `_block_for` (first-seen round-robin: stored patients into the cue-addressable range, other
          asserted patients into the spare range);
       b. a host STRING-IDENTITY test decides confirm vs contradict: `if str(p_asserted).lower() ==
          str(p_stored).lower(): t = s` (the asserted patient shares the stored block), else a different block,
          plus a host collision step that forces t != s when the two strings differ;
       c. the prediction cue is driven at block s, taken from the RECALLED WORD, not from the agent/action.
     Only then does the spiking circuit run: the cue drives the learned prediction (patient_expected, GABA_A
     inhibition onto surprise block s), the asserted drive excites surprise block t, and the surprise pool's firing
     rate is read off `cp_firing_states`. The circuit turns s == t into a low rate and s != t into a high one; the
     decision WHICH case this turn is was made by the string compare in (b).
  4. `normalized = clip(hz / (2 * threshold), 0, 1)` -- a fixed HOST rescale (declared residual). After that the
     existing pipeline takes over (spiking salience afferent -> host EMA -> host EMA->pA map -> spiking SNc).
No step maps message content to a reward category, but step 1 is a regex/keyword gate, step 3 (a-c) is host
addressing that decides confirm vs contradict by string identity, and step 4 is a host map. Steps 1 and 3 are
registered in docs/SCAFFOLD-LEDGER.md; the S15(c) trace cannot pass while either is on the path.

THE READ RESTORES THE ORGAN STATE IT TOUCHES (fix round 2). The organ is process-shared and production reads it
again later in the same turn (and reconsolidation, default-ON, gates on that read). The A10 read snapshots every
piece of organ state it can mutate and restores it before returning -- see the block above `_SCALAR_TYPES`. Without
that, the v2 seed-7 arms measured the production CONFIRM read at 0.3472222222222222 Hz with the flag ON vs
0.4050925925925926 Hz OFF. What has been MEASURED: at the module level (seed 7, numpy, the production organ on the
merged pool, v3/footprint_module.json) the organ's read-state hash is unchanged across every A10 read and the
production reads equal the flag-OFF reference. Not yet measured: the handler level (criterion (D) in the arms), other
seeds, the cupy backend. Outside the isolation, declared: the intact organ's first-use build (block above
`_SCALAR_TYPES`) and the recall call `chat.inner.what_does` (at the call in `spiking_reward_value`).

LESION (`BRAIN_REWARD_VALUE_LESION=1`). The read uses the organ's OWN lesioned twin (`sorg.judge(..., lesion=True)`):
a STANDALONE `build_expectation_circuit` bridge, trained the same way, with the patient_expected->surprise
prediction edges zeroed. It is NOT a matched cut of the intact read: the intact read runs on the organ's slice of
the shared merged cortical pool, after the per-block prediction-gain homeostat; the twin is a separate bridge and
never runs the homeostat. So an intact-vs-lesion comparison differs in substrate and homeostat as well as in the
zeroed edges (declared). It also does not cut the afferent this path is named for: it SWAPS the source organ's
read for a disinhibited twin's read. What an intact-vs-lesion comparison measures is whether the surprise
PREDICTION reaches the DA mode through this path. (Before fix round 2 there was also a read-count asymmetry: the
intact arm's A10 read shifted the production surprise read, the lesion arm's twin read did not. Both A10 reads now
restore the organ state they touch, measured at the module level only, and the twin's first-use build leaves the
host's global generators as it found them.) Under the lesion a CONFIRM and a CONTRADICT read are both HIGH but NOT equal (seed 7:
0.861 vs 0.969 normalized in both runs). The v2 block control measured why: each lesioned read equals its own
surprise block's cue-free rate, so the residual is the block-8 vs block-0 rate difference, not a prediction effect
(research/findings/2026-09-24-reward-value-spiking-afferent-seed7-derisk-PARTIAL.md). At read time
the lesioned read records the absolute sum of the twin's patient_expected<->surprise edge weights (and the intact
organ's, for reference) under `lesion_cut`, and `tools.lab.void_if` flags a cut that no longer holds.

ERRORS NEVER DRIVE THE SNc. Any failure (import, organ build, read) returns a record with `drives=False` and no
`normalized`: the caller then runs the pre-existing afferent unchanged. (The first version returned
`normalized=0.0`, which drove the SNc at 0 pA -- a real 'rest' mode -- on a broken read.) Since the follow-up round
the same holds when the read's snapshot cannot be taken, when the restore after the read raises or is not exact
(`footprint.restored_exact` is not True), and when a host generator changed across the lesion twin's build: with no
exact restore there is no read.

CONTRACT (additive, reversible, byte-identical-off). `da_mode_drives_chat.observe_turn` checks the
`BRAIN_REWARD_VALUE_AFFERENT` env var BEFORE importing this module; unset/off -> this module is never imported or
called and `DaModeDrivesWorkspace.observe` is called with exactly the pre-existing arguments. OFF identity is
asserted from data (pre-patch vs post-patch hashes), not from reading this code: see
research/runners/_reward_value_afferent_offidentity.py and research/runners/_reward_value_afferent_derisk.py.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests/dev).
See research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md and its AMENDMENT-1.
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

import numpy as np

_TRUTHY = ("1", "true", "on", "yes")


def reward_value_enabled() -> bool:
    """`BRAIN_REWARD_VALUE_AFFERENT` truthy (1/true/on/yes). Default (unset) -> False. `da_mode_drives_chat` parses
    the SAME env var with the SAME truthy set before importing this module (pinned by
    tests/test_reward_value_afferent.py)."""
    return os.environ.get("BRAIN_REWARD_VALUE_AFFERENT", "0").strip().lower() in _TRUTHY


def reward_value_lesioned() -> bool:
    """`BRAIN_REWARD_VALUE_LESION` truthy -> the read uses the surprise organ's prediction-edges-zeroed twin (see the
    module docstring: a genuine synaptic cut, but not a substrate-matched one)."""
    return os.environ.get("BRAIN_REWARD_VALUE_LESION", "0").strip().lower() in _TRUTHY


def _composer_class(chat) -> Optional[str]:
    """The recall composer's class name (e.g. OneBrainComposer vs RFPhasorComposer), so a record says which recall
    produced the expected patient. `chat.inner` is the BrainConversationalAgent, which holds `.composer` itself (the
    first version looked for `chat.inner.agent.composer`, which does not exist, and so recorded null on every read of
    the v2 seed-7 run). None if neither chain is present."""
    try:
        inner = getattr(chat, "inner", None)
        comp = getattr(inner, "composer", None)
        if comp is None:
            comp = getattr(getattr(inner, "agent", None), "composer", None)
        return None if comp is None else type(comp).__name__
    except Exception:
        return None


def _host_csr(bridge):
    """The bridge's connection matrix as a host scipy CSR (cupyx sparse -> .get())."""
    M = bridge.cp_connections
    if type(M).__module__.startswith("cupyx"):
        M = M.get()
    return M.tocsr()


def _host_idx(a) -> np.ndarray:
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.int64)


def pe_surprise_edge_abs_sum(bridge, idx_map) -> dict:
    """Absolute sum of the connection weights between the patient_expected and surprise index sets of `bridge`, in
    both CSR orientations (the circuit has ONE pathway between them, patient_expected->surprise; which orientation
    the CSR stores it in is decided empirically by `_install_block_diagonal`, so both are read). A lesion of that
    pathway holds iff both sums are exactly 0.0 at read time."""
    M = _host_csr(bridge)
    pe = _host_idx(idx_map["patient_expected"])
    su = _host_idx(idx_map["surprise"])
    a = float(abs(M[su][:, pe]).sum())
    b = float(abs(M[pe][:, su]).sum())
    return {"rows_surprise_cols_pe": a, "rows_pe_cols_surprise": b, "total": a + b}


def _blocks(sorg, p_stored: str, p_asserted: str):
    """The circuit blocks the organ used for this read (mirrors `SurpriseProductionOrgan.read_surprise`'s own
    assignment; read-only lookups, after the read)."""
    try:
        s = sorg._block.get(str(p_stored).lower())
        if str(p_asserted).lower() == str(p_stored).lower():
            return s, s
        t = sorg._block.get(str(p_asserted).lower())
        if t == s:
            t = sorg._block.get("__" + str(p_asserted).lower())
        return s, t
    except Exception:
        return None, None


def _lesion_cut(sorg) -> dict:
    """Read-time check that the lesion still holds (docs/TERMS.md 'lesion'): the twin's patient_expected<->surprise
    weights must be exactly 0.0 now, not only when they were zeroed at build. The intact organ's sum is recorded as
    the reference the cut is measured against (a different bridge: see the module docstring)."""
    out = {"checked": True}
    try:
        les = sorg._ensure_les()
        out["lesion_twin"] = pe_surprise_edge_abs_sum(les["bridge"], les["idx_map"])
        out["holds"] = bool(out["lesion_twin"]["total"] == 0.0)
    except Exception as e:
        out.update({"holds": False, "error": f"{type(e).__name__}: {e}"})
    try:
        out["intact_reference"] = pe_surprise_edge_abs_sum(sorg.bridge, sorg.idx_map)
    except Exception as e:
        out["intact_reference_error"] = f"{type(e).__name__}: {e}"
    try:
        from tools.lab import void_if
        void_if(not out.get("holds"), "A10 lesion: the surprise twin's patient_expected->surprise edges are not all "
                                      "zero at read time -- this read is not a lesion read")
    except Exception:
        pass
    return out


# ── FOOTPRINT-FREE READ (fix round 2, after the review of 7d5c2743d) ──────────────────────────────────────────────
# The surprise organ is process-shared and default-ON: production reads it later in the SAME turn
# (webapp/server.py's surprise block, which reconsolidation also gates on). A read on the shared merged pool is
# READ-HISTORY dependent: the pool bridge carries no `_rest_extra` snapshot, so `_hard_reset` leaves the surprise
# slice's adaptive thresholds / activity EMA / refractory state where the last read put them (seed 7, v2 arms: the
# production CONFIRM read was 0.4050925925925926 Hz as the organ's first read and 0.3472222222222222 Hz as its
# second). An A10 read that ran first therefore SHIFTED the production read. The A10 read now leaves no footprint:
# every piece of state the read can mutate -- all array attributes of the bridge it drives (dense per-neuron and
# per-synapse state, the sparse weight data), its scalar and container attributes, the runtime clock, the organ's
# host block bookkeeping (`_block`, `_cue_next`, `_novel_next`) and the numpy / Python global RNG states -- is
# snapshotted right before the read and restored right after, so the production read sees exactly the state it
# would have seen with the flag off. If the restore raises or is not exact, the read does not drive the SNc.
#
# BUILDS AND THE GLOBAL GENERATORS (follow-up round, after the re-review of 4b6a9cf66). Every bridge build calls
# sim/bridge.py `_initialize_rng`, which reseeds cupy's, numpy's AND Python's global generators. Builds run BEFORE
# the snapshot, so the snapshot cannot undo them. (The fix-round-2 text named only cupy here; that was wrong.)
#   * The LESION TWIN's first-use build (`_ensure_les`, only in the lesion arm) runs inside
#     `_global_rngs_untouched`: numpy's and Python's global states are saved and restored, and on the cupy backend
#     the build is handed a private cupy generator object while the host's object is set aside untouched and put
#     back (cupy's RandomState has no get_state/set_state, and `cupy.random.seed` reseeds the current object IN
#     PLACE, so only swapping the object keeps the host's stream intact). The twin build reseeds from its own seed,
#     so the twin it builds is the same either way. The record's `footprint.twin_build` says whether the host
#     generators compare equal after the build; if not, the read does not drive the SNc.
#   * The INTACT organ's first-use build (`ensure_built`, both A10 arms) is NOT isolated, and is declared: when no
#     earlier caller built the organ (the battery worker runs no startup warm-up; the webapp startup does warm it),
#     A10's call builds it, which reseeds all three generators at that point in the turn instead of at the
#     production surprise block. It is common to the intact and lesion arms and absent from the OFF arm. Isolating it
#     would not restore the flag-OFF sequence either: production's own first build resets the generators at the
#     surprise block, and A10 cannot reproduce that reset at that point.
# The read itself: numpy's and Python's global states are restored with the rest of the snapshot. Cupy's global
# generator is not restored around the READ (no state accessor; the read draws no random numbers when the pool's
# noise is off, which is the configuration measured in v3/footprint_module.json).
_SCALAR_TYPES = (bool, int, float, complex, str, bytes, type(None), np.generic)


def _backend_xp():
    try:
        from sim.backend import get_backend
        xp, name = get_backend()
        return xp, name
    except Exception:
        return np, "numpy"


def _np_states_equal(a, b) -> bool:
    try:
        return (a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:]))
    except Exception:
        return False


@contextlib.contextmanager
def _global_rngs_untouched(xp=None, name=None):
    """Run a block (a bridge BUILD) without changing any process-global generator the production path reads later:
    numpy's and Python's global states are saved and restored; on the cupy backend the block runs on a private cupy
    generator object and the host's object is put back untouched (see the block above `_SCALAR_TYPES`). Yields a
    record that is filled in on exit: `numpy_unchanged`, `python_unchanged`, `cupy` ("swapped" / "not the backend"
    / "swap failed: ...") and `host_rngs_unchanged` (all of them held)."""
    import random as _random
    if xp is None:
        xp, name = _backend_xp()
    rec = {}
    np_state = np.random.get_state()
    py_state = _random.getstate()
    cp_saved = None
    if name == "cupy" and xp is not np:
        try:
            cp_saved = xp.random.get_random_state()
            xp.random.set_random_state(xp.random.RandomState(0))   # private; the build reseeds it from its own seed
            rec["cupy"] = "swapped"
        except Exception as e:
            cp_saved = None
            rec["cupy"] = f"swap failed: {type(e).__name__}: {e}"
    else:
        rec["cupy"] = "not the backend"
    try:
        yield rec
    finally:
        cp_ok = True
        if cp_saved is not None:
            xp.random.set_random_state(cp_saved)
            cp_ok = xp.random.get_random_state() is cp_saved
        elif rec["cupy"].startswith("swap failed"):
            cp_ok = False
        _random.setstate(py_state)
        np.random.set_state(np_state)
        rec["numpy_unchanged"] = _np_states_equal(np.random.get_state(), np_state)
        rec["python_unchanged"] = bool(_random.getstate() == py_state)
        rec["cupy_host_object_restored"] = bool(cp_ok)
        rec["host_rngs_unchanged"] = bool(rec["numpy_unchanged"] and rec["python_unchanged"] and cp_ok)


def _array_module(x):
    try:
        import cupy
        return cupy.get_array_module(x)
    except Exception:
        return np


def _is_sparse(x) -> bool:
    m = type(x).__module__
    return ((m.startswith("scipy.sparse") or m.startswith("cupyx.scipy.sparse"))
            and hasattr(x, "data") and hasattr(x, "indices") and hasattr(x, "indptr"))


def _is_dense(x) -> bool:
    if isinstance(x, np.ndarray):
        return True
    m = type(x).__module__
    return m.startswith("cupy") and not m.startswith("cupyx") and hasattr(x, "shape") and hasattr(x, "dtype")


def _arr_equal(a, b) -> bool:
    if getattr(a, "shape", None) != getattr(b, "shape", None) or getattr(a, "dtype", None) != getattr(b, "dtype", None):
        return False
    xp = _array_module(a)
    try:
        return bool(xp.array_equal(a, b, equal_nan=True))
    except TypeError:
        if getattr(a.dtype, "kind", "") in "fc":
            return bool(xp.all((a == b) | (xp.isnan(a) & xp.isnan(b))))
        return bool(xp.array_equal(a, b))
    except Exception:
        return bool(xp.array_equal(a, b))


def _scalar_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    try:
        if isinstance(a, (float, np.floating)) and np.isnan(a) and np.isnan(b):
            return True
    except Exception:
        pass
    try:
        return bool(a == b)
    except Exception:
        return a is b


def _snapshot_obj(obj) -> dict:
    """Snapshot every attribute of `obj` a read can mutate: dense arrays (copied), sparse matrices (data, indices,
    indptr copied), scalars (by value) and list/dict/set containers (shallow copy). Other objects are recorded by
    identity only (a read that REPLACES one is undone by re-binding the original)."""
    snap = {}
    for name, val in list(vars(obj).items()):
        if _is_dense(val):
            snap[name] = ("dense", val, val.copy())
        elif _is_sparse(val):
            snap[name] = ("sparse", val, val.data.copy(), val.indices.copy(), val.indptr.copy())
        elif isinstance(val, _SCALAR_TYPES):
            snap[name] = ("scalar", val)
        elif isinstance(val, (list, dict, set)):
            snap[name] = ("container", val, val.copy())
        else:
            snap[name] = ("object", val)
    return snap


def _restore_obj(obj, snap: dict) -> dict:
    """Undo every change a read made to `obj` since `_snapshot_obj`. Returns {changed, added, removed, exact}:
    the attribute names whose value or binding changed during the read (evidence of what the read touched), names
    the read added (deleted again) or removed (re-bound), and whether the post-restore state equals the snapshot."""
    changed, added, removed = [], [], []
    cur = vars(obj)
    for name in [n for n in cur if n not in snap]:
        added.append(name)
        delattr(obj, name)
    for name, rec in snap.items():
        kind, orig = rec[0], rec[1]
        present = name in vars(obj)
        now = getattr(obj, name, None) if present else None
        if not present:
            removed.append(name)
        if kind == "dense":
            if (now is not orig) or not _arr_equal(orig, rec[2]):
                changed.append(name)
            orig[...] = rec[2]
            if now is not orig:
                setattr(obj, name, orig)
        elif kind == "sparse":
            same = (now is orig and orig.data.shape == rec[2].shape and _arr_equal(orig.data, rec[2])
                    and _arr_equal(orig.indices, rec[3]) and _arr_equal(orig.indptr, rec[4]))
            if not same:
                changed.append(name)
                if orig.data.shape == rec[2].shape and _arr_equal(orig.indices, rec[3]) and _arr_equal(orig.indptr, rec[4]):
                    orig.data[...] = rec[2]
                else:
                    orig.data, orig.indices, orig.indptr = rec[2].copy(), rec[3].copy(), rec[4].copy()
                if now is not orig:
                    setattr(obj, name, orig)
        elif kind == "scalar":
            if not present or not _scalar_equal(now, orig):
                changed.append(name)
                setattr(obj, name, orig)
        elif kind == "container":
            if (now is not orig) or len(orig) != len(rec[2]) or (
                    isinstance(orig, dict) and any(orig.get(k, _MISSING) is not v for k, v in rec[2].items())) or (
                    isinstance(orig, list) and any(a is not b for a, b in zip(orig, rec[2]))) or (
                    isinstance(orig, set) and orig != rec[2]):
                changed.append(name)
            if isinstance(orig, dict):
                orig.clear()
                orig.update(rec[2])
            elif isinstance(orig, list):
                orig[:] = rec[2]
            else:
                orig.clear()
                orig.update(rec[2])
            if now is not orig:
                setattr(obj, name, orig)
        else:  # object: only a re-binding is undone
            if now is not orig:
                changed.append(name)
                setattr(obj, name, orig)
    exact = True
    for name, rec in snap.items():
        kind, orig = rec[0], rec[1]
        now = getattr(obj, name, _MISSING)
        if kind == "dense":
            exact &= (now is orig) and _arr_equal(now, rec[2])
        elif kind == "sparse":
            exact &= (now is orig) and _arr_equal(now.data, rec[2]) and _arr_equal(now.indices, rec[3]) \
                and _arr_equal(now.indptr, rec[4])
        elif kind == "scalar":
            exact &= _scalar_equal(now, orig)
        elif kind == "container":
            exact &= (now is orig) and len(now) == len(rec[2])
        else:
            exact &= now is orig
    exact &= all(n in snap for n in vars(obj))
    return {"changed": sorted(changed), "added": sorted(added), "removed": sorted(removed), "exact": bool(exact)}


_MISSING = object()
_ORGAN_BOOKKEEPING = ("_block", "_cue_next", "_novel_next")


def _snapshot_read_state(sorg, bridge) -> dict:
    import random as _random
    snap = {"bridge": _snapshot_obj(bridge), "np_rng": np.random.get_state(), "py_rng": _random.getstate()}
    rs = getattr(bridge, "runtime_state", None)
    if rs is not None and hasattr(rs, "__dict__"):
        snap["runtime_state"] = {k: v for k, v in vars(rs).items() if isinstance(v, _SCALAR_TYPES)}
    snap["organ"] = {}
    for k in _ORGAN_BOOKKEEPING:
        if hasattr(sorg, k):
            v = getattr(sorg, k)
            snap["organ"][k] = dict(v) if isinstance(v, dict) else v
    return snap


def _restore_read_state(sorg, bridge, snap: dict) -> dict:
    import random as _random
    out = _restore_obj(bridge, snap["bridge"])
    rs_changed = []
    rs = getattr(bridge, "runtime_state", None)
    for k, v in (snap.get("runtime_state") or {}).items():
        if not _scalar_equal(getattr(rs, k, _MISSING), v):
            rs_changed.append(k)
            setattr(rs, k, v)
    organ_changed = []
    for k, v in snap["organ"].items():
        cur = getattr(sorg, k, _MISSING)
        if isinstance(v, dict):
            if not isinstance(cur, dict) or cur != v:
                organ_changed.append(k)
            if isinstance(cur, dict):
                cur.clear()
                cur.update(v)
            else:
                setattr(sorg, k, dict(v))
        else:
            if not _scalar_equal(cur, v):
                organ_changed.append(k)
            setattr(sorg, k, v)
    np.random.set_state(snap["np_rng"])
    _random.setstate(snap["py_rng"])
    out["runtime_state_changed"] = sorted(rs_changed)
    out["organ_bookkeeping_changed"] = sorted(organ_changed)
    return out


def spiking_reward_value(chat, message: str, seed: int) -> Optional[dict]:
    """The entry point `da_mode_drives_chat.observe_turn` calls when the flag is on.

    Returns None when this turn carries no expectation-bearing assertion the brain holds an expectation for (or the
    surprise organ is disabled project-wide) -- the caller then runs the pre-existing afferent unchanged. Returns a
    record with `drives=True` and `normalized` in [0, 1] when the spiking surprise read applies, and a record with
    `drives=False` (and an `error`) when anything failed. Never raises.

    The read leaves no footprint on the ORGAN (see the block above `_SCALAR_TYPES`): the state it mutates is
    restored before this returns. If the snapshot cannot be taken, or the restore raises or is not exact, the read
    does not drive (`drives=False`); the record's `footprint` block says what the read touched and whether the
    restore was exact. The lesion twin's first-use build runs with the host's global generators set aside
    (`footprint.twin_build`). What is NOT covered by the isolation is declared in the block above `_SCALAR_TYPES`
    (the intact first-use build) and at the recall call below (`chat.inner.what_does`)."""
    lesion = reward_value_lesioned()
    try:
        import research.runners.surprise_production_organ as _SO
    except Exception as e:
        return {"on": True, "source": "surprise", "drives": False, "error": f"import: {type(e).__name__}: {e}"}
    try:
        if not _SO.surprise_enabled():
            return None  # the parent faculty is off project-wide: never a back door around it (declared)
        asrt = _SO.extract_assertion(message)
        if asrt is None:
            return None
        a_s, v_s, p_asserted = asrt
        try:
            p_stored = chat.inner.what_does(a_s, v_s)
        except Exception:
            p_stored = None
        if not p_stored:
            return None
        sorg = _SO.get_organ(seed=seed)                  # the SAME process-shared organ production reads
        # builds first (kept), then snapshot -> read -> restore (the read leaves no footprint on the organ)
        if hasattr(sorg, "ensure_built"):
            sorg.ensure_built()                          # intact first-use build: declared, not isolated (see above)
        twin_build = None
        if lesion:
            # the twin's first-use build reseeds all three global generators (sim/bridge.py `_initialize_rng`);
            # run it with the host's generators set aside, so the lesion arm shifts no generator production reads
            with _global_rngs_untouched() as twin_build:
                les = sorg._ensure_les()
            if twin_build.get("host_rngs_unchanged") is not True:
                return {"on": True, "source": "surprise", "drives": False, "twin_build": twin_build,
                        "error": "twin build: a host generator changed across the lesion twin's build"}
            read_bridge = les["bridge"]
        else:
            read_bridge = sorg.bridge
        try:
            snap = _snapshot_read_state(sorg, read_bridge)
        except Exception as e:   # no snapshot -> no read
            return {"on": True, "source": "surprise", "drives": False,
                    "error": f"snapshot: {type(e).__name__}: {e}"}
        footprint = {"isolated": True, "bridge": "lesion_twin" if lesion else "intact"}
        if twin_build is not None:
            footprint["twin_build"] = twin_build
        try:
            sj = sorg.judge(a_s, v_s, str(p_stored), str(p_asserted), lesion=bool(lesion))
            s_blk, t_blk = _blocks(sorg, str(p_stored), str(p_asserted))   # before the bookkeeping is restored
        finally:
            try:
                fp = _restore_read_state(sorg, read_bridge, snap)
                footprint.update({"changed_during_read": fp["changed"], "added_during_read": fp["added"],
                                  "removed_during_read": fp["removed"],
                                  "runtime_state_changed": fp["runtime_state_changed"],
                                  "organ_bookkeeping_changed": fp["organ_bookkeeping_changed"],
                                  "restored_exact": fp["exact"]})
            except Exception as e:
                footprint.update({"restored_exact": False, "restore_error": f"{type(e).__name__}: {e}"})
        if footprint.get("restored_exact") is not True:
            # the organ may now carry this read's footprint, so the production read later in the turn may differ
            # from the flag-OFF read: an unrestored read must not also drive the SNc (no exact restore -> no read)
            return {"on": True, "source": "surprise", "drives": False, "footprint": footprint,
                    "error": "restore: the organ's pre-read state was not restored exactly"
                             + (f" ({footprint['restore_error']})" if footprint.get("restore_error") else "")}
        hz = float(sj["surprise_hz"])
        threshold = float(sj["threshold"])
        if not np.isfinite(hz) or not np.isfinite(threshold) or threshold <= 0.0:
            return {"on": True, "source": "surprise", "drives": False, "footprint": footprint,
                    "error": f"degenerate read: hz={hz} threshold={threshold}"}
        normalized = float(np.clip(hz / (2.0 * threshold), 0.0, 1.0))
        info = {
            "on": True, "source": "surprise", "drives": True, "lesioned": bool(lesion),
            "agent": a_s, "action": v_s, "stored_patient": str(p_stored), "asserted_patient": str(p_asserted),
            "surprise_hz": hz, "threshold": threshold, "surprised": bool(sj["surprised"]),
            "normalized": normalized, "stored_block": s_blk, "asserted_block": t_blk,
            "composer": _composer_class(chat),
            "signal": "unsigned prediction-error magnitude (salience), not a signed reward value",
            "residual": ("hz->normalized is a fixed host rescale; extract_assertion is a regex/keyword gate; the "
                         "confirm-vs-contradict block choice is a host string-identity step (read_surprise)"),
            "footprint": footprint,
        }
        if lesion:
            info["lesion_cut"] = _lesion_cut(sorg)
        return info
    except Exception as e:
        return {"on": True, "source": "surprise", "drives": False, "error": f"{type(e).__name__}: {e}"}
