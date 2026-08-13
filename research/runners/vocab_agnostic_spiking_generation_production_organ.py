"""VOCAB-AGNOSTIC SPIKING generative DRAW wired into the PRODUCTION open-ended-generation channel (#3E, 2026-08-13).

WHAT THIS CONVERTS (the remaining host internal of the #3E channel):
  The production GENERATE channel (#3E, `ChatBrain._generate_hypothesis` in `brain_chat_tui.py`) VOLUNTEERS a
  novel grounded HYPOTHESIS on an open-ended prompt ("what might a dog chase") over the brain's OWN learned
  fact-association graph. Its render SURFACE is ALREADY brain-native spiking (A1a spiking Broca). But the DRAW
  itself -- picking WHICH verb / object filler from the brain's PPMI-like likelihood -- was the b2 HOST oracle
  (`GenerativeReplayProposer._sample_weighted` with `use_spiking_sampler=False` -> `np.random.choice(p=w)`). The
  b2 SPIKING soft-WTA (`SpikingWTASampler`) could NOT be used there because its `__init__` hard-locks role pools
  to `_category_pools(TAXONOMY_8x8)` and `_encodable_agents()` then indexes `self.row[<taxonomy word>]` ->
  KeyError on ANY runtime-grown conversational lexicon (`brain_chat_tui.py` L274-278 documents exactly this).

  This organ routes that DRAW through the VOCAB-AGNOSTIC spiking soft-WTA (`VocabAgnosticSpikingSampler`, the
  6-seed-GO de-risk `research/runners/_spiking_openended_generation_derisk.py`,
  `research/findings/2026-08-12-vocab-agnostic-spiking-openended-generation-6seed.md`). The winner is read from
  `cp_firing_states` of a real Izhikevich `SimulationBridge` WTA bank driven by the brain's likelihood + OU
  membrane noise -- 0 host `rng.choice` on the draw path; ou_std->0 collapses to a deterministic argmax. The
  role pools are INDUCED from the brain's OWN stored-fact concepts (nouns = agents+patients, verbs = actions) --
  the runtime lexicon -- so there is NO taxonomy and NO KeyError. Everything DOWNSTREAM of the draw is unchanged:
  the b2 likelihood (`_weight_partner`), the plausibility gate (`_plausible`), the non-contradiction gate
  (`_contradicts`), and the #3E moat verify (`what_does` / `is_it_true`). Only the categorical DRAW moves from
  host to spikes.

HOW IT WIRES (minimal, moat-safe, NO b2/sim edit): the organ PRE-INJECTS a cached `VocabAgnosticSpikingSampler`
  onto the already-built `GenerativeReplayProposer` and flips `use_spiking_sampler=True`. The proposer's own
  `_ensure_spiking_sampler` then returns OUR (taxonomy-free) sampler instead of constructing the taxonomy one,
  and `_sample_weighted` routes every draw through `sampler.draw_from_weights(weights, candidates)` (inherited
  UNCHANGED from the GO `SpikingWTASampler`; the winner is argmax-over-FIRING read from `cp_firing_states`). The
  bank is sized to cover the runtime candidate pools, so the draw never falls back to the taxonomy sampler.

BRAIN-BASED: the DRAW is a `cp_firing_states` read on firing neurons; the OU membrane noise IS the stochasticity
  (no host RNG on the draw path). The seed-agent CHOICE per event (which memory the SWR replay reactivates) uses
  the proposer's host rng -- a documented-legitimate host process, exactly as in the de-risk / followon2.

DEFAULT-ON. `BRAIN_SPIKING_DRAW` in {0,false,no,off} -> no-op (the proposer stays on the host oracle draw ->
  byte-identical to the pre-organ production path). `BRAIN_SPIKING_DRAW_LESION=1` -> build the sampler with the
  likelihood ablated (uniform drive) -> the draw ignores the brain's association graph -> plausibility collapses
  (load-bearing lesion; the de-risk's LESION gate).

HONEST RESIDUALS (declared; ride the de-risk's mapped residuals, NOT hidden): the DRAW is spiking; the role
  induction (here trivial -- roles are KNOWN from the brain's stored SVO facts, no morpho-tagger needed), the
  SVO template, the likelihood matrix, and the RF-composer moat remain host scaffolds (NOT "fully spiking"). The
  sampler runs on ITS OWN Izhikevich bridge alongside the recall composer (rides the one-brain merge, burn-down
  #1, exactly as the affect/surprise organs do).

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os

# reuse-by-import: the 6-seed-GO vocab-agnostic spiking soft-WTA sampler (taxonomy-free draw over arbitrary vocab)
from research.runners._spiking_openended_generation_derisk import VocabAgnosticSpikingSampler

# THE OPERATING POINT = the 6-seed-GO de-risk point (base/gain map the input-normalized likelihood into the
# Izhikevich firing band; read_window integrates firing; ou_std IS the soft-WTA stochasticity).
_BASE_PA = 110.0
_GAIN_PA = 160.0
_READ_WINDOW = 120
_OU_STD = 200.0
_TEMPERATURE = 1.0
_MIN_BANK = 96          # floor on the WTA bank size (matches the de-risk `n_cand = max(96, ...)`)


def spiking_draw_enabled() -> bool:
    """Default-ON. `BRAIN_SPIKING_DRAW` in {0,false,no,off,''} -> the byte-identical host oracle draw (no-op)."""
    v = os.environ.get("BRAIN_SPIKING_DRAW")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def spiking_draw_lesioned() -> bool:
    """`BRAIN_SPIKING_DRAW_LESION` in {1,true,yes,on} -> ablate the likelihood (uniform drive; load-bearing)."""
    v = os.environ.get("BRAIN_SPIKING_DRAW_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _roles_from(proposer):
    """The role pools INDUCED from the brain's OWN stored-fact concepts (the runtime lexicon): nouns = the words
    that ever fill a subject or object slot; verbs = the words that ever fill an action slot. No taxonomy.

    INTERSECTED with the plausibility graph's vocabulary (`proposer.row`): the b2 proposer builds its role pools
    from BOTH affirmed and negated facts, but the graph `P/row` is over the affirmed co-occurrence, so a concept
    that only ever appeared in a negated fact can be in the role pool yet absent from `row`. Such a word has NO
    graph edges (weight 0 -> never a likelihood winner) so dropping it is semantically inert -- AND it is what
    keeps `_encodable_agents`/`_weights` (which index `row[word]`) from a KeyError. Row-membership, not a
    taxonomy, is the only gate."""
    known = set(proposer.row)
    nouns = sorted((set(proposer.agents) | set(proposer.patients)) & known)
    verbs = sorted(set(proposer.actions) & known)
    return nouns, verbs


class VocabAgnosticSpikingDrawOrgan:
    """A process-shared spiking generative-DRAW organ. Holds ONE cached `VocabAgnosticSpikingSampler` (a real
    Izhikevich `SimulationBridge` WTA bank with OU noise), rebuilt only when the brain's runtime lexicon grows
    past the bank size or the lesion flag changes. `install(proposer)` pre-injects it onto an already-built b2
    `GenerativeReplayProposer` and routes the draw through spikes -- leaving every downstream gate + the moat
    unchanged."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._sampler = None
        self._sig = None                 # (n_nouns, n_verbs, id(P), lesion) -> rebuild when the lexicon/lesion changes

    def build_sampler(self, proposer, lesion: bool = False):
        """Build the vocab-agnostic spiking sampler over the proposer's OWN P/row/tau + INDUCED role pools."""
        nouns, verbs = _roles_from(proposer)
        n_cand = max(_MIN_BANK, len(nouns), len(verbs))
        return VocabAgnosticSpikingSampler(
            proposer.P, proposer.row, proposer.tau, nouns, verbs,
            seed=self.seed, n_cand_max=n_cand,
            base_pA=_BASE_PA, gain_pA=_GAIN_PA, read_window=_READ_WINDOW,
            ou_std_current_pA=_OU_STD, temperature=_TEMPERATURE,
            ablate_likelihood=bool(lesion))

    def install(self, proposer, lesion=None) -> dict:
        """Route `proposer`'s generative DRAW through the spiking soft-WTA (default-ON). Idempotent: reuses the
        cached sampler while the runtime lexicon (and lesion flag) are unchanged, rebuilds when they grow/flip.
        Returns a FUNCTIONAL read of what was installed (or `{"on": False}` when disabled). When disabled, the
        proposer is LEFT UNTOUCHED -> its host oracle draw is byte-identical to the pre-organ production path."""
        if proposer is None:
            return {"on": False, "reason": "no proposer (brain knows too few facts to generate)"}
        if not spiking_draw_enabled():
            return {"on": False, "reason": "BRAIN_SPIKING_DRAW=0 -> host oracle draw (byte-identical)"}
        les = spiking_draw_lesioned() if lesion is None else bool(lesion)
        nouns, verbs = _roles_from(proposer)
        sig = (len(nouns), len(verbs), id(proposer.P), les)
        if self._sampler is None or self._sig != sig:
            self._sampler = self.build_sampler(proposer, lesion=les)
            self._sig = sig
        # PRE-INJECT: the proposer's `_ensure_spiking_sampler` returns THIS sampler (taxonomy-free) instead of
        # constructing the taxonomy `SpikingWTASampler` (which KeyErrors on runtime vocab). `_sample_weighted`
        # then draws via `sampler.draw_from_weights` (inherited unchanged: winner = argmax-over-FIRING read from
        # cp_firing_states). The bank covers every candidate pool, so the size-guard never rebuilds a taxonomy one.
        proposer._spiking_sampler = self._sampler
        proposer.use_spiking_sampler = True
        return {
            "on": True, "lesioned": les,
            "n_nouns": len(nouns), "n_verbs": len(verbs), "n_cand_max": int(self._sampler.n_cand_max),
            "n_spiking_draws": int(self._sampler.n_spiking_draws),
            "n_host_rng_draws": int(self._sampler.n_host_rng_draws),
            "base_pA": _BASE_PA, "gain_pA": _GAIN_PA, "read_window": _READ_WINDOW, "ou_std": _OU_STD,
        }

    def provenance(self) -> dict:
        """Read-only draw provenance since build: how many spiking draws vs host-rng draws the sampler has done.
        The whole point is `host_rng == 0` while `spiking > 0` -- the draw is on firing neurons, no host RNG."""
        s = self._sampler
        if s is None:
            return {"built": False, "n_spiking_draws": 0, "n_host_rng_draws": 0}
        return {"built": True, "n_spiking_draws": int(s.n_spiking_draws),
                "n_host_rng_draws": int(s.n_host_rng_draws),
                "n_silent_fallbacks": int(getattr(s, "n_silent_fallbacks", 0))}


_ORGAN: VocabAgnosticSpikingDrawOrgan | None = None


def get_organ(seed: int = 42) -> VocabAgnosticSpikingDrawOrgan:
    """The process-shared spiking generative-DRAW organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = VocabAgnosticSpikingDrawOrgan(seed=seed)
    return _ORGAN


def install_spiking_draw(proposer, seed: int = 42, lesion=None) -> dict:
    """Convenience wire-in used at the #3E draw site: route the proposer's DRAW through the spiking soft-WTA
    (default-ON). Returns the install read; a disabled/absent proposer leaves the host oracle draw in place."""
    return get_organ(seed).install(proposer, lesion=lesion)
