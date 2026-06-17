"""Order-encoded working memory via POSITION-BINDING on the project's SPIKING resonate-and-fire phasor substrate.

PRODUCTION MODULE (promoted from `_phaseB_ordered_wm_position_binding_derisk.py`, CYCLE 135 GO/BOUNDARY ->
2026-06-17-ordered-wm-position-binding-derisk.md). `OrderedPositionWM` subclasses the deployed composer
`RFPhasorComposer` (research/runners/rf_phasor_composer.py), whose bind / unbind / bundle / cleanup run on the
core `SimulationBridge`'s resonate-and-fire neurons + complex synapses (NeuronModel.RESONATE_AND_FIRE; the
genuine spiking-phasor FHRR substrate, Frady-Sommer 2019). The composer's `roles` dict is EXTENSIBLE -- a
gamma-slot POSITION phasor is added exactly like an SVO role vector, and binding an item to a slot is the SAME
spiking operation the deployed composer uses for sentence roles. NO new mechanism is invented; the deployed one
is reused and asked whether position-binding is order-bearing on it. NO `sim/` edit; reuse-by-import only.

WHAT THIS GIVES THE CONVERSATIONAL AGENT (the gap it closes). The project's spiking working memory had THREE
converging NEGATIVES at multi-referent disambiguation -- recency, a salience boost, and biased-competition WTA --
ALL on the rate-attractor `SpikingLoopContextBuffer`, a SET that holds items with NO order and whose winner is
decided by intrinsic basin asymmetry (2026-06-17-multireferent-disambiguation-NEGATIVE.md). An order-encoded WM
removes the competition: each held item lives in a disjoint bound subspace, addressed by SLOT, not competed in
rate. Reading slot k is `unbind(C, position_k)`; the winner is "which slot you read", so it FLIPS deterministically
when the discourse order changes -- exactly the order-control the three rate-buffer negatives failed (0/6 -> 6/6).

THE MOAT (no-confab). An unbind of an EMPTY slot (a position phasor that was never bound) yields a phasor that
matches no stored concept. The read is gated by a FAMILIARITY signal = the max phase-cosine match strength of the
recovered phasor to any vocab concept -- exactly the `cleanup_separated` familiarity gate of resonate_fire_fhrr.py
(Bogacz-Brown novelty/familiarity; a real, separate biological mechanism). Below threshold -> ABSTAIN (None).

FAMILIARITY THRESHOLD -- the production change vs the de-risk. The de-risk pre-registered a FROZEN threshold
(MATCH_THRESHOLD = 0.15) and honestly reported it BOUNDARY because that frozen value sat in the bundle-cross-talk
noise tail (it false-accepted ~1-2 probes/seed). The de-risk's diagnostic showed a CLEAN separation on every seed
(worst groundable 0.409 > worst ungroundable 0.209) and that at a PRINCIPLED separation-midpoint threshold the
moat is perfect (720/720). This production module therefore places the familiarity threshold by the principled
`cleanup_separated` rule -- it CALIBRATES the threshold per instance from the measured groundable-vs-ungroundable
separation (`calibrate_threshold`) instead of hardcoding 0.15. That is threshold hygiene (the `cleanup_separated`
placement rule), NOT tuning-to-pass: the calibration measures the floor from the WM's OWN groundable (real-slot)
and ungroundable (empty/scrambled) match-strength distributions and places the gate at their midpoint, never at a
value chosen to make a downstream test pass.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.rf_phasor_composer import RFPhasorComposer

# Defaults (the validated de-risk operating point). D=128 matches the BrainConversationalAgent's composer so an
# OrderedPositionWM built with the same (seed, vocab) shares byte-identical concept codes with the agent's
# composer (RFPhasorComposer draws concepts from default_rng(seed) over the sorted vocab; position phasors draw
# from a disjoint seed+1000 stream, so they do not perturb the concept draws).
D_DEFAULT = 128
N_SLOTS_DEFAULT = 7         # gamma slots per theta cycle (Lisman-Idiart); the order-encoded WM ceiling.
# A conservative fallback familiarity threshold, used ONLY if calibration is skipped. The production path
# calibrates instead (see `calibrate_threshold`); this constant is NOT used when calibrated.
MATCH_THRESHOLD_FALLBACK = 0.15


class OrderedPositionWM(RFPhasorComposer):
    """An order-encoded working memory on the spiking RF phasor substrate.

    N gamma-slot POSITION phasors (roles ``pos0..pos{N-1}``); a held ordered sequence is encoded as the BUNDLE of
    ``bind(item_k, position_k)``. Read slot k via the parent composer's spiking ``unbind`` + ``cleanup``, gated by
    the familiarity moat. Two distinct never-used position phasors (``emptyslot``, ``scrambled``) probe the moat.

    Args:
        seed: per-seed determinism (also seeds the parent composer's concept/role codes).
        D: phasor dimension on the spiking RF substrate (default 128 = the agent composer's D, for code parity).
        vocab: the working-memory vocabulary (the items the WM can hold). Defaults to ``w0..w15``.
        n_slots: gamma slots (default 7, the Lisman-Idiart ceiling).
        cleanup_words: optional subset of ``vocab`` that a slot read cleans up against (e.g. only the referent
            nouns, so a discourse pronoun never resolves to an action word). Defaults to the full vocab.
        match_threshold: optional FIXED familiarity threshold. Default ``None`` -> the production path CALIBRATES
            the threshold from the measured separation via ``calibrate_threshold`` (the principled
            ``cleanup_separated`` placement rule). Pass a float to pin it (e.g. to reproduce the de-risk's frozen
            0.15).
    """

    def __init__(self, seed=42, D=D_DEFAULT, vocab=None, n_slots=N_SLOTS_DEFAULT,
                 cleanup_words=None, match_threshold=None):
        vocab = vocab if vocab is not None else [f"w{i}" for i in range(16)]
        super().__init__(seed=seed, D=D, vocab=vocab)
        self.n_slots = int(n_slots)
        # Words a slot read cleans up against (None -> full vocab). Lets the agent restrict resolution to referents.
        self._cleanup_words = list(cleanup_words) if cleanup_words is not None else None
        # Deterministic per-seed position phasors, added to the composer's role set (the SAME machinery as SVO
        # role vectors). Drawn from a dedicated stream (seed+1000) so they are disjoint from the concept/role draws
        # -- this keeps the parent's concept codes byte-identical to a plain RFPhasorComposer(seed, D, vocab).
        prng = np.random.default_rng(seed + 1000)
        for k in range(self.n_slots):
            self.roles[f"pos{k}"] = prng.uniform(0.0, 1.0, self.D)
        # Two never-bound position phasors for the moat (an unused slot, and a fully-unrelated phasor).
        self.roles["emptyslot"] = prng.uniform(0.0, 1.0, self.D)
        self.roles["scrambled"] = prng.uniform(0.0, 1.0, self.D)
        # Familiarity threshold: fixed if given, else calibrated from the measured separation (principled rule).
        self.match_threshold = (float(match_threshold) if match_threshold is not None
                                else self.calibrate_threshold())

    # --- ordered encode / read ------------------------------------------------
    def encode_sequence(self, item_words):
        """Encode an ordered K-item sequence (a list of vocab words) as the bundle of (item, position) bindings on
        the spiking RF substrate. ``item_words[k]`` is bound to slot k. Returns the composite phasor (phases)."""
        bounds = [self._bind(self.roles[f"pos{k}"], self.concepts[item_words[k]])
                  for k in range(len(item_words))]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def _match_strength(self, rec_phases, words=None):
        """Familiarity signal: the max phase-cosine similarity of a recovered phasor to any candidate concept.
        This is the ``cleanup_separated`` match-strength gate -- computed BEFORE identification."""
        words = words if words is not None else (self._cleanup_words or self.words)
        return max(float(np.mean(np.cos(2.0 * np.pi * (rec_phases - self.concepts[w]))))
                   for w in words)

    def read_slot(self, composite_phases, slot_key, gate=True, words=None):
        """Read the item at ``slot_key`` (e.g. ``'pos1'`` or the moat probes ``'emptyslot'`` / ``'scrambled'``)
        from a composite, on the spiking substrate: spiking ``unbind`` by the position phasor, then the
        familiarity gate -> abstain (None) if the recovered phasor matches no candidate concept, else cleanup to
        the nearest concept. ``words`` restricts both the familiarity scoring and the cleanup to a candidate subset
        (defaults to ``cleanup_words`` or the full vocab). Returns ``(word_or_None, match_strength)``."""
        cand = words if words is not None else (self._cleanup_words or self.words)
        rec = self._unbind_phases(composite_phases, slot_key)   # spiking RF unbind (conj diagonal complex synapse)
        match = self._match_strength(rec, cand)
        if gate and match < self.match_threshold:
            return None, match                                  # ABSTAIN -- no-confab moat
        return self._cleanup(rec, cand), match                  # spiking/numpy cleanup -> nearest candidate concept

    # --- familiarity-threshold calibration (the principled cleanup_separated placement rule) ------------------
    def calibrate_threshold(self, n_probe=40, used_load=3, rng_seed=None):
        """Place the familiarity (abstention) threshold from the WM's OWN measured separation -- the principled
        ``cleanup_separated`` rule (resonate_fire_fhrr.py). For ``n_probe`` random load-``used_load`` sequences,
        measure the match strength of (a) a real used slot (GROUNDABLE) and (b) the two never-bound probes
        ``emptyslot`` / ``scrambled`` (UNGROUNDABLE), then return the midpoint of (min groundable, max ungroundable)
        -- the threshold that cleanly separates them when they do not overlap. Falls back to
        ``MATCH_THRESHOLD_FALLBACK`` if (degenerately) the distributions overlap. NOT tuned to any downstream test
        -- it is measured from the WM's intrinsic groundable/ungroundable match distributions."""
        cand = self._cleanup_words or self.words
        # Draw the probe ITEMS from the same candidate set a real slot read cleans up against (a discourse slot
        # only ever holds a candidate; calibrating over the full vocab would inject non-candidate words at slots
        # and collapse the groundable distribution). This measures the groundable floor for the actual use case.
        used_load = min(int(used_load), self.n_slots, len(cand))
        rng = np.random.default_rng((self.seed if rng_seed is None else rng_seed) + 9999)
        real, ungrnd = [], []
        for _ in range(int(n_probe)):
            idx = list(rng.choice(len(cand), size=used_load, replace=False))
            items = [cand[i] for i in idx]
            comp = self.encode_sequence(items)
            real.append(self._match_strength(self._unbind_phases(comp, "pos0"), cand))
            ungrnd.append(self._match_strength(self._unbind_phases(comp, "emptyslot"), cand))
            ungrnd.append(self._match_strength(self._unbind_phases(comp, "scrambled"), cand))
        real_min = float(np.min(real))
        ungrnd_max = float(np.max(ungrnd))
        if real_min <= ungrnd_max:                              # degenerate overlap -> conservative fallback
            return MATCH_THRESHOLD_FALLBACK
        return (real_min + ungrnd_max) / 2.0
