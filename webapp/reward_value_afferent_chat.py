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
  3. `sorg.judge(...)` -- the surprise organ drives its predictive-coding mismatch circuit and reads the surprise
     pool's firing rate off `cp_firing_states` (spiking). The block each patient word maps to is host bookkeeping
     (`_block_for`, first-seen round-robin), not a content classifier.
  4. `normalized = clip(hz / (2 * threshold), 0, 1)` -- a fixed HOST rescale (declared residual). After that the
     existing pipeline takes over (spiking salience afferent -> host EMA -> host EMA->pA map -> spiking SNc).
No step maps message content to a reward category, but step 1 is a regex/keyword gate and step 4 is a host map.

LESION (`BRAIN_REWARD_VALUE_LESION=1`). The read uses the organ's OWN lesioned twin (`sorg.judge(..., lesion=True)`):
a STANDALONE `build_expectation_circuit` bridge, trained the same way, with the patient_expected->surprise
prediction edges zeroed. It is NOT a matched cut of the intact read: the intact read runs on the organ's slice of
the shared merged cortical pool, after the per-block prediction-gain homeostat; the twin is a separate bridge and
never runs the homeostat. So an intact-vs-lesion comparison differs in substrate and homeostat as well as in the
zeroed edges (declared). Under the lesion a CONFIRM and a CONTRADICT read are both HIGH but NOT equal (seed 7:
0.861 vs 0.969 normalized in both runs). The v2 block control measured why: each lesioned read equals its own
surprise block's cue-free rate, so the residual is the block-8 vs block-0 rate difference, not a prediction effect
(research/findings/2026-09-24-reward-value-spiking-afferent-seed7-derisk-PARTIAL.md). At read time
the lesioned read records the absolute sum of the twin's patient_expected<->surprise edge weights (and the intact
organ's, for reference) under `lesion_cut`, and `tools.lab.void_if` flags a cut that no longer holds.

ERRORS NEVER DRIVE THE SNc. Any failure (import, organ build, read) returns a record with `drives=False` and no
`normalized`: the caller then runs the pre-existing afferent unchanged. (The first version returned
`normalized=0.0`, which drove the SNc at 0 pA -- a real 'rest' mode -- on a broken read.)

CONTRACT (additive, reversible, byte-identical-off). `da_mode_drives_chat.observe_turn` checks the
`BRAIN_REWARD_VALUE_AFFERENT` env var BEFORE importing this module; unset/off -> this module is never imported or
called and `DaModeDrivesWorkspace.observe` is called with exactly the pre-existing arguments. OFF identity is
asserted from data (pre-patch vs post-patch hashes), not from reading this code: see
research/runners/_reward_value_afferent_offidentity.py and research/runners/_reward_value_afferent_derisk.py.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests/dev).
See research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md and its AMENDMENT-1.
"""
from __future__ import annotations

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


def spiking_reward_value(chat, message: str, seed: int) -> Optional[dict]:
    """The entry point `da_mode_drives_chat.observe_turn` calls when the flag is on.

    Returns None when this turn carries no expectation-bearing assertion the brain holds an expectation for (or the
    surprise organ is disabled project-wide) -- the caller then runs the pre-existing afferent unchanged. Returns a
    record with `drives=True` and `normalized` in [0, 1] when the spiking surprise read applies, and a record with
    `drives=False` (and an `error`) when anything failed. Never raises."""
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
        sj = sorg.judge(a_s, v_s, str(p_stored), str(p_asserted), lesion=bool(lesion))
        hz = float(sj["surprise_hz"])
        threshold = float(sj["threshold"])
        if not np.isfinite(hz) or not np.isfinite(threshold) or threshold <= 0.0:
            return {"on": True, "source": "surprise", "drives": False,
                    "error": f"degenerate read: hz={hz} threshold={threshold}"}
        normalized = float(np.clip(hz / (2.0 * threshold), 0.0, 1.0))
        s_blk, t_blk = _blocks(sorg, str(p_stored), str(p_asserted))
        info = {
            "on": True, "source": "surprise", "drives": True, "lesioned": bool(lesion),
            "agent": a_s, "action": v_s, "stored_patient": str(p_stored), "asserted_patient": str(p_asserted),
            "surprise_hz": hz, "threshold": threshold, "surprised": bool(sj["surprised"]),
            "normalized": normalized, "stored_block": s_blk, "asserted_block": t_blk,
            "composer": _composer_class(chat),
            "signal": "unsigned prediction-error magnitude (salience), not a signed reward value",
            "residual": "hz->normalized is a fixed host rescale; extract_assertion is a regex/keyword gate",
        }
        if lesion:
            info["lesion_cut"] = _lesion_cut(sorg)
        return info
    except Exception as e:
        return {"on": True, "source": "surprise", "drives": False, "error": f"{type(e).__name__}: {e}"}
