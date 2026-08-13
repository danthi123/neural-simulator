"""RECONSOLIDATION — prediction-error-gated IN-PLACE fact UPDATE (belief revision) for the PRODUCTION turn (F-lane).

THE GAP. The production conversational memory is APPEND-ONLY: `store()` pushes each fact onto the composer `kb`
and `query_*` answers the FIRST match. Tell the brain "the dog went north" then correct it ("actually, south") and
today TWO contradictory facts coexist with the STALE one answered first. There is no belief revision.

THE MECHANISM. Reconsolidation (Nader-Schafe-LeDoux 2000; Osan-Tort-Amaral 2011 mismatch-gated attractor update;
Sevenster-Beckers-Kindt 2013 prediction-error NECESSITY): a reactivated memory becomes LABILE and is UPDATED IN
PLACE — but ONLY when retrieval carries a PREDICTION ERROR. That PE opens the reconsolidation WINDOW. A fully
predicted re-statement re-stabilizes unchanged; a never-stored cue is not created. Without the PE gate,
"reconsolidation" degenerates to `dict[key]=value` last-write-wins, which is neither biological nor a capability.

THE COMPOSITION (the design this organ realises — reuse-by-import, NO reimplementation, NO `sim/` edit):
  * D2 SURPRISE opens the window. The window-open decision IS the genuinely-SPIKING expectation-violation read
    (`SurpriseProductionOrgan`, `research/runners/surprise_production_organ.py`, the adversarially-verified 6/6-GO
    D2 faculty): a predictive-coding mismatch unit whose `cp_firing_states[surprise]` rate FIRES when the asserted
    patient violates the stored expectation and CANCELS (~0 Hz) on a confirmed re-statement. The gate is a threshold
    on that spiking rate — NOT a host `stored==asserted` string / cosine compare. This is the ACC-conflict / PE
    signal the mission names.
  * B3 NON-CONTRADICTION / no-confab moat. Reactivation REQUIRES an existing trace: a never-stored cue ABSTAINS
    (a reactivated trace is updated, a missing one is never fabricated) — the exact update-an-existing-belief vs
    invent-from-nothing distinction the non-contradiction gate protects.
  * THE IN-PLACE UPDATE reuses the composer's OWN, separately de-risked store mechanism:
      - `RFPhasorComposer.update_on_mismatch` (Option A, `2026-06-17-reconsolidation-update-derisk-GO.md`, 6/6 GO)
        when the composer exposes it — the reactivate + rewrite-in-place path;
      - else the PRODUCTION-DEFAULT `OneBrainComposer` substrate-slot rewrite via its OWN `_write_block` +
        `_compose_phases` (the corrected composite is written into the SAME persistent store slot the stale fact
        occupied — the fact composite lives in the device synapses, so recall returns the corrected patient with NO
        contradictory duplicate). No bind/store is reimplemented here.

BRAIN-BASED SCOPE (honest). The DECISION to open the reconsolidation window is a spiking surprise read
(`cp_firing_states[surprise]`). The store that is rewritten is the composer fact KB — the documented
"composer-as-idealization" layer that ALL recall already uses (the bind / unbind / composite / store run on the
resonate substrate; the KB selection is the same idealization). The reconsolidation WINDOW is spiking; the store
idealization is exactly the residual recall already carries — it is not a new shortcut.

MOAT-SAFE + ADDITIVE. Reconsolidation NEVER manufactures a fact (abstains on a missing trace), NEVER writes on a
re-statement (window closed), NEVER flips an abstain or enters a certainty band; it only rewrites a fact the brain
ALREADY HOLDS. Default-ON; `BRAIN_RECONSOLIDATION` in {0,false,no,off} -> the byte-identical append-only oracle
(the update path is never invoked, the store stays append-only).

LESION-LOAD-BEARING. `BRAIN_RECONSOLIDATION_LESION=1` blocks the in-place update (the window fires but the labile
trace is not rewritten) -> the correction falls back to append-only -> recall returns the STALE fact -> belief
revision COLLAPSES. The corrected answer is exactly the part the in-place update produced, so the capability is
caused by the update, not by a fixed input-driven artifact.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os


def reconsolidation_enabled() -> bool:
    """Default-ON. `BRAIN_RECONSOLIDATION` in {0,false,no,off,''} -> the byte-identical append-only oracle."""
    v = os.environ.get("BRAIN_RECONSOLIDATION")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def reconsolidation_lesioned() -> bool:
    """`BRAIN_RECONSOLIDATION_LESION` in {1,true,yes,on} -> block the in-place update (load-bearing lesion)."""
    v = os.environ.get("BRAIN_RECONSOLIDATION_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class ReconsolidationProductionOrgan:
    """A co-resident belief-revision organ. It owns NO new spiking circuit of its own: the reconsolidation WINDOW is
    the shared D2 `SurpriseProductionOrgan` (a `cp_firing_states[surprise]` read), and the IN-PLACE UPDATE is the
    composer's OWN de-risked store rewrite. This class is the thin composition layer + the moat + the lesion switch."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._surprise = None                          # the D2 spiking window gate (lazily built / shared)

    # ── the spiking window ────────────────────────────────────────────────────────────────────────────────────
    def ensure_built(self):
        """Warm the D2 surprise organ = the spiking reconsolidation-window gate (built once, process-shared)."""
        from research.runners.surprise_production_organ import get_organ as _surprise_organ
        if self._surprise is None:
            self._surprise = _surprise_organ(self.seed)
        self._surprise.ensure_built()

    def window_open(self, agent: str, action: str, p_stored: str, p_asserted: str, lesion_gate: bool = False):
        """The SPIKING reconsolidation window. Fires (returns True) when asserting `p_asserted` VIOLATES the stored
        expectation `p_stored` — the D2 `cp_firing_states[surprise]` read crosses its calibrated threshold; cancels
        (~0 Hz -> False) on a confirmed re-statement. `lesion_gate` uses the prediction-removed surprise twin.
        Returns (opened: bool, surprise_judgement: dict)."""
        self.ensure_built()
        sj = self._surprise.judge(agent, action, str(p_stored), str(p_asserted), lesion=lesion_gate)
        return bool(sj["surprised"]), sj

    # ── the in-place update (reuse the composer's OWN store mechanism) ─────────────────────────────────────────
    @staticmethod
    def _reactivate_idx(composer, agent, action):
        """Reactivation: the kb index of the FIRST stored fact whose cue roles (agent+action) match; None = no
        trace (abstain -> the moat). The composer `kb` is a list of `(fact_dict, handle)` on BOTH the rf and onebrain
        composers; the fact-dict cue is the same selection recall uses to pick WHICH stored composite to read."""
        for i, (fact, _handle) in enumerate(getattr(composer, "kb", []) or []):
            try:
                if fact.get("agent") == agent and fact.get("action") == action:
                    return i
            except AttributeError:
                continue
        return None

    @staticmethod
    def _inplace_rewrite(composer, idx, agent, action, new_patient):
        """Rewrite the reactivated fact's patient IN PLACE, reusing the composer's OWN de-risked store path.
        rf + onebrain (production default): `update_on_mismatch` (the SPIKING gate already opened the window, so
        pe_labile=0.0 defers the write decision entirely to the window -> it recomposes the fact with the new patient
        and OVERWRITES the SAME store block via `_write_block`+`_compose_phases`; NO contradictory duplicate). A bare
        composer without that method falls back to the direct compose+write below."""
        # RECRUIT a RUNTIME-NOVEL corrected patient into a reserved cleanup slot BEFORE the rewrite — the composer's
        # OWN recruit path (the exact pattern `_store_fact` uses for hear()/store()), so a revision to a word never
        # seen THIS session (e.g. "actually, south" when only "north" was taught) has a codebook entry. Without it,
        # `update_on_mismatch` -> `_patient_prediction_error` / `_compose_phases` KeyErrors on the un-coded word and
        # the rewrite silently falls back to append-only. Idempotent no-op for an already-coded word / no headroom;
        # guarded for composers without the recruit hook. NOT a mechanism change — it is the same new-word recruitment
        # the initial store already performs, applied to the corrected filler the in-place rewrite composes.
        if isinstance(new_patient, str) and hasattr(composer, "_recruit_word"):
            try:
                if composer._recruit_word(new_patient) and hasattr(composer, "_csr_cache"):
                    composer._csr_cache.clear()   # the cleanup operator changed -> rebuild on the next batched read
            except Exception:
                pass
        # rf + onebrain: the Option-A de-risked reactivate+rewrite (6/6 GO). The spiking window already gated this
        # call, so pe_labile=0.0 -> the host cosine does NOT re-gate the decision (the D2 spike read is the gate).
        if hasattr(composer, "update_on_mismatch"):
            return composer.update_on_mismatch(agent, action, new_patient, pe_labile=0.0)
        # (bare composer without update_on_mismatch): rewrite the SAME store slot via the composer's own compose+write.
        fact2 = dict(composer.kb[idx][0])
        fact2["patient"] = new_patient
        roles = [r for r in composer.bind_roles if r in fact2]
        composer._write_block(idx, composer._compose_phases([fact2[r] for r in roles], roles))
        composer._persistent_dirty = True
        composer.kb[idx] = (fact2, None)
        return {"action": "rewrite", "wrote": True, "pe": None}

    def reconsolidate(self, composer, agent, action, p_stored, p_asserted,
                      sj=None, lesion: bool = False, lesion_gate: bool = False) -> dict:
        """Route a contradicting / corrective assertion `(agent, action, p_asserted)` against the brain's recalled
        expectation `p_stored`. The caller (the D2 surprise block in brain_chat) has already confirmed a stored
        trace via `what_does` and computed the spiking surprise `sj` — pass it in to reuse the ONE spiking read.

        Returns a decision dict {action, wrote, surprised, ...}:
          * window CLOSED (not surprised)      -> 'restabilize'      (no write; the PE-gate boundary condition)
          * no reactivatable trace             -> 'abstain'          (the no-confab moat; nothing fabricated)
          * LESION (window fires, update off)  -> 'lesioned_nowrite' (append-only fallback -> stale persists)
          * window OPEN + trace present        -> 'rewrite'          (belief revision: corrected patient, no dup)"""
        # the spiking window decision (reuse the caller's sj when given, else read D2 now = the standalone path).
        if sj is None:
            opened, sj = self.window_open(agent, action, p_stored, p_asserted, lesion_gate=lesion_gate)
        else:
            opened = bool(sj.get("surprised"))

        idx = self._reactivate_idx(composer, agent, action)
        if idx is None:
            return {"action": "abstain", "wrote": False, "surprised": opened, "surprise": sj,
                    "stored": p_stored, "asserted": p_asserted}
        if not opened:
            return {"action": "restabilize", "wrote": False, "surprised": False, "surprise": sj,
                    "stored": p_stored, "asserted": p_asserted}
        if lesion:
            return {"action": "lesioned_nowrite", "wrote": False, "surprised": True, "surprise": sj,
                    "stored": p_stored, "asserted": p_asserted}
        res = self._inplace_rewrite(composer, idx, agent, action, str(p_asserted))
        res.update({"surprised": True, "surprise": sj, "stored": p_stored, "asserted": p_asserted})
        return res


_ORGAN: "ReconsolidationProductionOrgan | None" = None


def get_organ(seed: int = 42) -> ReconsolidationProductionOrgan:
    """The process-shared reconsolidation organ (its D2 window gate is built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = ReconsolidationProductionOrgan(seed=seed)
    return _ORGAN


def reconsolidation_notice(agent: str, action: str, old_patient: str, new_patient: str) -> str:
    """The honest functional NOTICE surfaced when a stored belief is revised in place. A FUNCTIONAL read of what the
    brain DID (rewrote a reactivated trace) — never a phenomenal claim."""
    return (f"Updated — I'd stored that {agent} {action} {old_patient}; I've revised it in place to "
            f"{agent} {action} {new_patient}. ")
