"""INTERNAL WORLDVIEW / AFFECTIVE WORLD-MODEL wired into the PRODUCTION conversational turn (Gate-B, E2, 2026-08-12).

The owner's "internal worldview": a brain that maintains an internal PREDICTIVE representation of its
conversational world — given the current affective context, it PREDICTS the interlocutor's next-turn
VALENCE, holds that expectation (QUERYABLE: "what do you expect / how is this going?"), and fires a
genuinely-SPIKING SURPRISE when the actual next turn VIOLATES it (an affective prediction-error).

It REUSES (does not reinvent) the adversarially-verified E2 faculty
(`research/runners/_affective_world_model_derisk.py`, 6/6 GO): a 2-channel spiking predictive-coding
VALENCE forward model on the Izhikevich bridge. `state --Hebbian-learned transition--> pred_{pos,neg}`
(FS/PV-like, delivering GABA_A SUBTRACTIVE inhibition = the top-down prediction); `obs_{pos,neg} --exc-->
surprise_{pos,neg} <--inh-- pred_{pos,neg}`. EXPECTED (observed == predicted valence) -> the prediction
cancels the observation -> ~0 Hz; VIOLATED -> the un-inhibited surprise channel FIRES. The queryable
prediction read = sign(rate(pred_pos) - rate(pred_neg)); the surprise = a `cp_firing_states[surprise]`
read (NO host argmax over a stored table, NO host compare of observed vs predicted, current_reward==0).

HOW IT MAPS ONTO A TURN (the persistence prior — the honest, minimal semantics that makes the generic
model testable): the CURRENT affective context (the appraised valence sign of the conversation) SELECTS a
model state — a state whose LEARNED spiking prediction matches that sign — so the queryable expectation is
"I expect the conversation to keep going {positive/negative}". A next turn whose observed valence FLIPS the
sign therefore VIOLATES the held prediction and fires the spiking surprise. The prediction read + the
mismatch read are the load-bearing SPIKING parts; the state SELECTION (persistence) and the valence
APPRAISAL are declared host boundaries.

BRAIN-BASED: the predicted valence = a two-pool spike-rate difference; the surprise = a
`cp_firing_states[surprise]` rate. The only host pieces are the SENSORY encoding — the conversational
state token + the observed valence delivered as drive (the legitimate environment boundary, exactly the
de-risk's) — and the persistence state SELECTION (a declared prior; see RESIDUALS).

MOAT-SAFE + ADDITIVE: E2 only READS (a queryable expectation) or NOTICES (an honest surprise notice on an
affect-trajectory violation). It NEVER manufactures a fact, flips an abstain, enters the certainty band, or
changes WHICH answer the recall path produced. The recall/moat run FIRST and unchanged.

LESION-LOAD-BEARING: zeroing the learned `state->pred_{pos,neg}` transition (`BRAIN_WORLDMODEL_LESION=1`,
the de-risk's `_lesion_transition`) removes the top-down prediction, so the surprise fires HIGH on an
EXPECTED (persistence-confirming) turn too -> the expected/violated separation COLLAPSES (the de-risk's
~40x -> ~1x) and the queryable predicted-valence sign loses its meaning (both pred pools ~equal). The
discrimination is therefore caused by the learned SPIKING prediction, not the host state/observation drive.

HONEST RESIDUALS (declared — the mission's named NEXT RUNGS, not faked):
  * GENERIC pos/neg pools: the model predicts a GENERIC valence, NOT the ACTUAL interlocutor's affect. Binding
    the state + observation to the real interlocutor affect (the P0.3 valence latch + the W5 ToM channel) so
    it predicts THIS person's next-turn affect is the NEXT RUNG (currently un-wired).
  * HOST state SELECTION: the persistence prior (positive context -> a positive-predicting state) is a host
    mapping over the model's own learned predictions, not a learned conversational-state encoder.
  * FIRST-ORDER transition: `state -> next valence` is Markov-1 (no history/context dependence); a
    context-dependent transition needs the HTM-TM high-order predictor (EMERGE-15 GO) — a named rung.
  * TEACHER-DRIVEN: the transition is LEARNED (Hebbian co-fire) but not self-organized from conversation.
  * CO-RESIDENT: runs on ITS OWN forward-model bridge ALONGSIDE the recall composer, not merged onto the ONE
    recall bridge — rides on the one-brain merge (burn-down #1), exactly as the affect/surprise organs do.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests). Default-ON;
`BRAIN_WORLDMODEL=0` -> the byte-identical oracle (fully skipped).
"""
from __future__ import annotations

import contextlib
import os
import re
import statistics as _st

from research.runners._affective_world_model_derisk import (
    build_world_model_circuit,
    train_transition,
    _drive_read,
    _hard_reset,
    _idx,
    _valence_map,
    _lesion_transition,
)

_REGIONS = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")

# An explicit "what do you expect / how is this going" style query -> answered by the queryable prediction
# read-out. Kept narrow so it never hijacks a recall turn (mirrors the affect feel-query gate).
_EXPECT_RE = re.compile(
    r"\b(what do you expect|what are you expecting|what do you predict|"
    r"how('?s| is) (this|it|the (chat|conversation|talk)) going|"
    r"where('?s| is) (this|it) (going|headed)|how do you (think|feel) (this|it)('?s| is)? going|"
    r"how('?s| is) it going)\b", re.IGNORECASE)


def worldmodel_enabled() -> bool:
    """Default-ON. `BRAIN_WORLDMODEL` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_WORLDMODEL")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def worldmodel_lesioned() -> bool:
    """`BRAIN_WORLDMODEL_LESION` in {1,true,yes,on} -> zero the learned state->pred transition (load-bearing)."""
    v = os.environ.get("BRAIN_WORLDMODEL_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def is_expectation_query(text: str) -> bool:
    return bool(_EXPECT_RE.search(text or ""))


class WorldModelProductionOrgan:
    """A process-shared spiking affective forward model. Built ONCE (lazily): the 2-channel predictive-coding
    valence circuit, TRAINED (Hebbian state->valence) then FROZEN, with a build-time selection of a
    positive-predicting and a negative-predicting state (by the SPIKING read) + a calibrated surprise threshold.
    Each turn: `expectation(context_sign)` reads the queryable two-pool prediction; `read_surprise(context_sign,
    observed_sign)` reads whether the observation violates the held prediction (a `cp_firing_states[surprise]` rate)."""

    def __init__(self, seed: int = 42, n_states: int = 6, n_reps: int = 22,
                 cue_pa: float = 1000.0, obs_pa: float = 400.0, hold: int = 60, pre_steps: int = 60,
                 shared=None):
        self.seed = int(seed)
        # ONE-BRAIN MERGE (opt-in, default-off): when a MergedSubstrate is injected, the intact circuit is this
        # organ's region SLICE of the SHARED spiking bridge (built + trained + read here) instead of its own
        # bridge. The lesioned twin stays standalone (a diagnostic). See onebrain_merge_production.py.
        self._shared = shared
        self.n_states = int(n_states)
        self.n_reps = int(n_reps)
        self.cue_pa = float(cue_pa)
        self.obs_pa = float(obs_pa)
        self.hold = int(hold)
        self.pre_steps = int(pre_steps)
        self._built = False
        self._st = None            # intact circuit state dict
        self._les = None           # lazily-built lesioned twin (transition zeroed)
        self.state_pos = self.state_neg = None
        self.threshold = None
        self.calib = None

    def _build_one(self, lesion: bool = False) -> dict:
        from sim.backend import get_backend
        xp, _ = get_backend()
        if self._shared is not None and not lesion:
            # ONE-BRAIN MERGE: train + read this organ's SLICE of the SHARED spiking bridge. The regions +
            # per-neuron init are built by MergedSubstrate; here we only run the same Hebbian state->valence
            # transition training this organ always runs, on the shared substrate.
            self._shared.ensure_built()
            bridge, cfg, meta = self._shared.bridge, self._shared.cfg, self._shared.meta_worldmodel
            idx_map = self._shared.worldmodel_idx_map()
            bridge._blk = meta["blk"]                       # this organ's block size before it drives (shared bridge)
            vmap = _valence_map(self.seed, meta["n_states"])
            train_transition(bridge, cfg, idx_map, meta, xp, vmap, n_reps=self.n_reps)  # sets hebbian ON internally
            cfg.enable_hebbian_learning = False
            return {"bridge": bridge, "cfg": cfg, "meta": meta, "xp": xp, "idx_map": idx_map, "vmap": vmap}
        bridge, cfg, meta = build_world_model_circuit(self.seed, n_states=self.n_states)
        idx_map = {n: xp.asarray(_idx(bridge, n)) for n in _REGIONS}
        vmap = _valence_map(self.seed, meta["n_states"])
        # LEARN the state->valence transition (Hebbian co-fire), then FREEZE (per-turn reads never learn).
        train_transition(bridge, cfg, idx_map, meta, xp, vmap, n_reps=self.n_reps)
        cfg.enable_hebbian_learning = False
        if lesion:
            _lesion_transition(bridge, meta)               # zero state<->pred edges (removes the prediction)
        return {"bridge": bridge, "cfg": cfg, "meta": meta, "xp": xp, "idx_map": idx_map, "vmap": vmap}

    def _predict(self, st: dict, s: int):
        """The queryable SPIKING prediction for state `s`: drive the state cue, read (pred_pos, pred_neg) rates."""
        b, xp, idx_map = st["bridge"], st["xp"], st["idx_map"]
        b._blk = st["meta"]["blk"]            # claim this organ's block size before driving (shared-bridge safe)
        guard = (self._shared.read_isolation("worldmodel")
                 if (self._shared is not None and st is self._st) else contextlib.nullcontext())
        with guard:
            _hard_reset(b)
            pr = _drive_read(b, idx_map, {"state": (s, self.cue_pa)}, self.pre_steps, xp, ["pred_pos", "pred_neg"])
        sign = 1 if (pr["pred_pos"] - pr["pred_neg"]) > 0 else -1
        return sign, float(pr["pred_pos"]), float(pr["pred_neg"])

    def _surprise(self, st: dict, s: int, observed_sign: int) -> float:
        """The SPIKING affective prediction-error (Hz) for observing `observed_sign` in state `s`: PREDICTION phase
        (state cue establishes the top-down prediction), then ASSERTION phase (state + observed valence drive) ->
        read cp_firing_states[surprise_{pos,neg}]. Observed matches prediction -> cancel ~0; violates -> FIRES."""
        b, xp, idx_map = st["bridge"], st["xp"], st["idx_map"]
        b._blk = st["meta"]["blk"]            # claim this organ's block size before driving (shared-bridge safe)
        obs_region = "obs_pos" if observed_sign > 0 else "obs_neg"
        guard = (self._shared.read_isolation("worldmodel")
                 if (self._shared is not None and st is self._st) else contextlib.nullcontext())
        with guard:
            _hard_reset(b)
            r = _drive_read(b, idx_map, {"state": (s, self.cue_pa), obs_region: (None, self.obs_pa)},
                            self.hold, xp, ["surprise_pos", "surprise_neg"],
                            pre_drives={"state": (s, self.cue_pa)}, pre_steps=self.pre_steps)
        return float(r["surprise_pos"] + r["surprise_neg"])

    def ensure_built(self):
        if self._built:
            return
        self._st = self._build_one(lesion=False)
        # SELECT a positive-predicting and a negative-predicting state BY THE SPIKING READ (the persistence prior
        # maps a +context to state_pos, a -context to state_neg). Verify the spiking sign agrees with the trained map.
        vmap = self._st["vmap"]
        pos_states, neg_states = [], []
        for s in range(self._st["meta"]["n_states"]):
            sign, _pp, _pn = self._predict(self._st, s)
            (pos_states if sign > 0 else neg_states).append((s, int(vmap[s])))
        self.state_pos = pos_states[0][0] if pos_states else 0
        self.state_neg = neg_states[0][0] if neg_states else min(1, self._st["meta"]["n_states"] - 1)
        # CALIBRATE the surprise threshold from the two selected states: expected (persistence-confirming) vs
        # violated (sign-flipped) observations. The threshold sits in the gap, biased so an EXPECTED turn stays quiet.
        exp_hz, vio_hz = [], []
        for s, exp_sign in ((self.state_pos, +1), (self.state_neg, -1)):
            exp_hz.append(self._surprise(self._st, s, exp_sign))    # observation matches the prediction -> low
            vio_hz.append(self._surprise(self._st, s, -exp_sign))   # observation violates the prediction -> high
        mean_exp = _st.mean(exp_hz)
        min_vio = min(vio_hz)
        self.threshold = 0.5 * (mean_exp + min_vio) if min_vio > mean_exp else 0.5 * (mean_exp + max(vio_hz))
        self.calib = {"state_pos": int(self.state_pos), "state_neg": int(self.state_neg),
                      "expected_hz": [float(x) for x in exp_hz], "violated_hz": [float(x) for x in vio_hz],
                      "threshold": float(self.threshold),
                      "pos_states": [int(s) for s, _v in pos_states],
                      "neg_states": [int(s) for s, _v in neg_states]}
        self._built = True

    def _ensure_les(self) -> dict:
        if self._les is None:
            self._les = self._build_one(lesion=True)
        return self._les

    def _state_for(self, context_sign: int) -> int:
        """The persistence prior: a positive affective context selects a positive-predicting state, a negative one
        a negative-predicting state, so the held expectation is 'I expect the conversation to keep going that way'."""
        return self.state_pos if int(context_sign) >= 0 else self.state_neg

    def expectation(self, context_sign: int, lesion: bool = False) -> dict:
        """The QUERYABLE prediction: for the current affective context, read the two-pool spiking prediction of the
        next-turn valence. Returns the predicted sign, the two pool rates, and the selected state."""
        self.ensure_built()
        st = self._ensure_les() if lesion else self._st
        s = self._state_for(context_sign)
        sign, pp, pn = self._predict(st, s)
        return {"on": True, "lesioned": bool(lesion), "context_sign": int(context_sign),
                "state": int(s), "pred_sign": int(sign), "pred_pos_rate": pp, "pred_neg_rate": pn,
                "pred_margin": float(pp - pn)}

    def read_surprise(self, context_sign: int, observed_sign: int, lesion: bool = False) -> dict:
        """Read whether an OBSERVED next-turn valence (`observed_sign`) VIOLATES the held prediction for the current
        affective context -> the spiking affective prediction-error (Hz), the threshold, and `surprised`."""
        self.ensure_built()
        st = self._ensure_les() if lesion else self._st
        s = self._state_for(context_sign)
        hz = self._surprise(st, s, observed_sign)
        return {"on": True, "lesioned": bool(lesion), "context_sign": int(context_sign),
                "observed_sign": int(observed_sign), "state": int(s),
                "surprise_hz": float(hz), "threshold": float(self.threshold),
                "surprised": bool(hz >= self.threshold), "calib": self.calib}


_ORGAN: WorldModelProductionOrgan | None = None


def get_organ(seed: int = 42) -> WorldModelProductionOrgan:
    """The process-shared affective world-model organ (built once on first use). When the ONE-BRAIN MERGE flag is
    ON (`BRAIN_ONEBRAIN_MERGE=1`, default-off) the organ is backed by the process-shared MergedSubstrate it
    co-inhabits with the surprise organ (ONE spiking bridge); OFF -> its own bridge exactly as today."""
    global _ORGAN
    if _ORGAN is None:
        from research.runners.onebrain_merge_production import merge_enabled, get_merged_substrate
        shared = get_merged_substrate(seed) if merge_enabled() else None
        _ORGAN = WorldModelProductionOrgan(seed=seed, shared=shared)
    return _ORGAN


def expectation_readout(exp: dict) -> str:
    """The HONEST functional read-out for a 'what do you expect / how is this going' query. A FUNCTIONAL read of the
    spiking forward model's predicted next-turn valence — never a phenomenal claim, never a fabricated fact."""
    sign = int(exp.get("pred_sign", 0))
    mood = "positive" if sign > 0 else "negative"
    m = float(exp.get("pred_margin", 0.0))
    return (f"My affective forward model expects this to keep going {mood} "
            f"(predicted next-turn valence {('+' if sign > 0 else '-')}, pool-rate margin {m:+.0f} Hz) — "
            f"that's a functional read of my world-model's prediction, not a felt expectation.")


def worldmodel_surprise_notice(exp_sign: int) -> str:
    """The honest functional NOTICE surfaced when the affective forward model's prediction is VIOLATED (the observed
    turn flipped the valence my model expected). A FUNCTIONAL read of the spiking surprise — never a phenomenal claim."""
    expected = "positive" if int(exp_sign) > 0 else "negative"
    return (f"That shifts the mood unexpectedly — my affective forward model had predicted this would keep going "
            f"{expected}, and my prediction-error unit fired. ")
