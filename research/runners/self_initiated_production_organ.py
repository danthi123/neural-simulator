"""SELF-INITIATED UTTERANCE — the first INTERNALLY-GENERATED turn class, wired for the PRODUCTION turn (2026-08-18).

This is the production-integration glue that gives the brain a genuinely-SELF-INITIATED remark: on an IDLE/EMPTY turn
(no user content — an empty message, or a bare "say something / what's on your mind" lead-in) the brain SELECTS a stored
concept ITSELF (a noise-seeded, curiosity-biased CA3 wander — NOT a host `random.choice` over concepts) and SPEAKS it
through the production `OneBrainComposer` mouth. It REUSES (does not reinvent) the two already-GO organs of the
loop-closing de-risk (`research/runners/_self_initiated_utterance_derisk.py`, 6-seed GO
`2026-08-13-self-initiated-utterance-GO.md`):

  * SELECTION (which concept, how often — the substrate decides, 0 host content-draw) = the multibasin self-initiation
    wander (`_self_initiation_multibasin_derisk._run_condition` / `_selection`, 6-seed GO
    `2026-08-13-self-initiation-multibasin-GO.md`): DISJOINT pattern-separated CA3 basins + a curiosity recurrent-gain
    biasing WHICH surfaces (66% attributable). Under weak non-specific Poisson (NO cue, 0 external CONTENT drive) each
    noise-seeded volley ignites WHICHEVER balanced basin its coincidental overlap favours; the curiosity gain biases
    which; the bistable KIR down-state returns the net to silence between events.
  * THE MOUTH (the selected concept -> words) = the production `OneBrainComposer` (`one_brain_composer.py`),
    reuse-by-import via the de-risk's `_build_mouth`. `render_fact(concept)` reconstructs "concept verb patient" by an
    ON-BRIDGE resonate-and-fire unbind + cleanup (the spiking decode), and ABSTAINS (None) on an unknown subject (the
    no-confab moat, verified).

THE LOOP: idle turn (no content) -> spiking CA3 wander SELECTS a curiosity-biased basin -> the basin's bound concept
-> OneBrainComposer.render_fact -> a spoken SVO utterance ABOUT that concept, wrapped as a self-initiated remark /
question. Internally-generated -> selected -> SPOKEN.

SCOPE — the MINIMAL buildable-now integration (declared honestly, per the design fork):
  * This is the idle/empty-turn SHORT-CIRCUIT. The TIMING is still HTTP/user-triggered ("say something"); only the
    CONTENT is internally selected by the substrate wander. It is "internally-selected content on an idle-turn
    trigger", NOT "fully autonomous proactive speech". A TRULY proactive background/idle-tick that speaks with NO HTTP
    request at all is a larger endpoint/infra build (the named deferred next rung — see the finding).

WHAT IS SPIKING vs WHAT IS HOST (declared honestly — the honesty boundary is a deliverable, not a caveat):
  * SPIKING (load-bearing): (i) the SELECTION of WHICH concept is spoken + HOW OFTEN — the CA3 dendritic-plateau
    attractor competition under non-specific noise (0 host content-draw / no `random.choice` over concepts); (ii) the
    steering VALUE (the curiosity ASK-pool want read off cp_firing_states); (iii) the VERBALISATION — the SVO
    proposition decoded by the OneBrainComposer's on-bridge RF resonate unbind + cleanup (`render_fact` reads the
    complex synapses, not host labels).
  * HOST (declared, rides existing burn-downs): (i) the per-concept NOVELTY levels are the ENVIRONMENT; (ii) the
    basin<->lexical-concept BINDING (which stored word each disjoint CA3 basin denotes) and each concept's stored FACT
    are the learned store / environment; (iii) the curiosity want->recurrent-gain PROJECTION (the one-brain-merge
    rung); (iv) the QUESTION/REMARK-template wrapper + any natural-language FLUENCY (the Broca/Qwen articulation
    scaffold) — the MEASURED content is the bare spiking SVO proposition.

LATENCY / BACKEND (the SAME structural residual d5-episodic declares): the CA3 wander store WRITE + the wander are
~seconds on cupy (the production substrate) but minutes on numpy@2000, so on numpy the heavy wander is DEFERRED
(`selfinit_store_ok()` == False -> `speak()` returns a DEFERRED result -> the handler emits the honest neutral idle
line). `BRAIN_SELF_INITIATE_STORE=1` FORCES it (tests + the cupy deployment); `BRAIN_SELF_INITIATE_REST` sets the
wander length (the numpy verify uses a reduced rest so the (A)+(C) substrate test runs in bounded time; production on
cupy uses the full 4000-step operating point of the GO). NO `sim/` edit; additive; default-ON with the
`BRAIN_SELF_INITIATE` env escape (byte-identical: the idle block is fully skipped) and the `BRAIN_SELF_INITIATE_LESION`
load-bearing flag (the CA3-store NO-ENCODE control -> the wander surfaces nothing -> the utterance stream collapses).

FUNCTIONAL CORRELATE, NOT phenomenal: measures + reports a self-initiated-UTTERANCE correlate. No claim of experience.
"""
from __future__ import annotations

import os
import re

import numpy as np

# reuse-by-import the loop-closing de-risk's machinery (multibasin selection + curiosity want + OneBrainComposer mouth)
from research.runners._self_initiated_utterance_derisk import (
    _lexicon, _build_mouth, _utterance_stream,
)
from research.runners._self_initiation_multibasin_derisk import _run_condition, _selection, NOV_BY_NMEM
from research.runners._self_initiated_spontaneous_thought_derisk import _curiosity_wants
from research.runners._gap5_spontaneous_reactivation_derisk import GO_CFG
from research.runners.one_brain_composer import OneBrainComposer


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Enable / lesion / store-gate flags — the exact contract the other Gate-B organs use.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def selfinit_enabled() -> bool:
    """Default-ON. `BRAIN_SELF_INITIATE` in {0,false,no,off} -> the idle block is fully skipped (byte-identical)."""
    v = os.environ.get("BRAIN_SELF_INITIATE")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def selfinit_lesioned() -> bool:
    """`BRAIN_SELF_INITIATE_LESION` in {1,true,yes,on} -> the CA3-store NO-ENCODE control (+ curiosity gain flattened):
    the wander surfaces no coherent basin -> the utterance stream collapses -> the honest neutral idle fallback."""
    v = os.environ.get("BRAIN_SELF_INITIATE_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def selfinit_store_ok() -> bool:
    """Whether the heavy CA3 wander (store WRITE + noise wander) may run this turn. ~seconds on cupy (the production
    substrate) but minutes on numpy@2000 — so on numpy the wander is DEFERRED (a declared latency residual, the SAME
    one d5-episodic declares for its BTSP write). `BRAIN_SELF_INITIATE_STORE` (1/on force-ON, 0/off force-OFF)
    overrides the backend gate for tests/deployments."""
    v = os.environ.get("BRAIN_SELF_INITIATE_STORE")
    if v is not None:
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    try:
        from sim.backend import get_backend
        return get_backend()[1] == "cupy"
    except Exception:
        return False


def _wander_rest_steps() -> int:
    """The wander length. Full 4000-step operating point of the GO on cupy (production); a reduced rest on numpy so the
    (A)+(C) substrate test runs in bounded time. `BRAIN_SELF_INITIATE_REST` overrides."""
    v = os.environ.get("BRAIN_SELF_INITIATE_REST")
    if v is not None:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    try:
        from sim.backend import get_backend
        return 4000 if get_backend()[1] == "cupy" else 500
    except Exception:
        return 500


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Idle/empty-turn detection (host — the environment/parse side). The SELECTION that follows is spiking.
# A DISJOINT class: an empty message, OR a bare "say something / what's on your mind" lead-in. Crafted NOT to overlap
# any reactive turn (a factual "what does X ..." recall, an assertion, a self/identity query, an anaphora turn).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_SELFINIT_LEADIN_RE = re.compile(
    r"^\s*(?:"
    r"say something(?: to me)?|"
    r"(?:so\s+)?(?:tell|say) me something(?:\s+(?:new|else))?|"
    r"what(?:'s| is| ?s)\s+on\s+your\s+mind|"
    r"(?:got |is there |do you have |have you got |anything|is anything)\s*(?:anything\s+)?on\s+your\s+mind\??|"
    r"what\s+are\s+you\s+thinking(?:\s+about)?|"
    r"what\s+do\s+you\s+want\s+to\s+talk\s+about|"
    r"what\s+would\s+you\s+like\s+to\s+talk\s+about|"
    r"got\s+anything\s+to\s+say|"
    r"anything\s+you\s+want\s+to\s+say"
    r")\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def is_selfinit_trigger(text) -> bool:
    """The idle/empty-turn class: an EMPTY message, or a bare 'say something / what's on your mind' lead-in (and
    nothing else — the whole message must be the lead-in, so a reactive turn that merely contains 'mind' is not
    caught). This owns a DISJOINT turn class; every reactive (non-idle) turn returns False -> byte-identical."""
    s = (text or "").strip()
    if s == "":
        return True
    return bool(_SELFINIT_LEADIN_RE.match(s))


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The self-initiation organ. Builds its OWN self-contained selection substrate + mouth (like d6's MultiSlotHold),
# reuse-by-import from the de-risk. `speak()` runs ONE curiosity-biased wander and routes the DOMINANT surfaced basin
# through the mouth. The heavy wander is gated behind selfinit_store_ok(); the mouth (decode) is always cheap.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class SelfInitiationOrgan:
    def __init__(self, seed: int = 42, n_mem: int = 4, D: int = 256, gain_scale: float = 1.0,
                 min_frac: float = 0.30):
        self.seed = int(seed)
        self.n_mem = int(n_mem)
        self.D = int(D)
        self.gain_scale = float(gain_scale)
        self.min_frac = float(min_frac)
        self._mouth_built = False
        self.comp = None
        self.utt_by_agent = None
        self.decode_ok = None
        self.moat_abstains = None
        self.agents = self.verbs = self.patients = self.vocab = None
        self.gains_on = None
        self.novelties = None
        self.comp_lesion = None          # the NO-ENCODE mouth (RF store with no facts) — the store-lesion control

    # ---- the MOUTH: build once, store n_mem facts, decode each on-bridge (mouth fidelity). Cheap (RF resonate). ----
    def _ensure_mouth(self):
        if self._mouth_built:
            return
        agents, verbs, patients, vocab = _lexicon(self.n_mem)
        comp, utt_by_agent, decode_ok, moat = _build_mouth(self.seed, agents, verbs, patients, vocab, self.D)
        self.comp, self.utt_by_agent, self.decode_ok, self.moat_abstains = comp, utt_by_agent, decode_ok, moat
        self.agents, self.verbs, self.patients, self.vocab = agents, verbs, patients, vocab
        # NOVELTY (the ENVIRONMENT) -> curiosity ASK-pool want -> graded recurrent gain (identical to the GO).
        nov_rng = np.random.default_rng(self.seed * 7919 + 1)
        novelties = [float(v) for v in nov_rng.permutation(np.asarray(NOV_BY_NMEM[self.n_mem], dtype=float))]
        wants, _ = _curiosity_wants(self.seed, novelties)
        wmax = max(wants) if wants else 1.0
        self.gains_on = [1.0 + self.gain_scale * (w / wmax if wmax > 1e-9 else 0.0) for w in wants]
        self.novelties = novelties
        self._mouth_built = True

    def _ensure_lesion_mouth(self):
        """The NO-ENCODE store-lesion control for the VERBALISATION substrate: the SAME OneBrainComposer built with NO
        facts stored -> render_fact abstains (None) for every concept (the RF store genuinely cannot decode). Lesioning
        the store, not a host flag: with nothing written into the complex RF synapses the mouth surfaces nothing."""
        if self.comp_lesion is None:
            self.comp_lesion = OneBrainComposer(seed=self.seed, D=self.D, vocab=list(self.vocab),
                                                k_max=max(8, self.n_mem), enable_rf_cudagraph=False)
            # DELIBERATELY store nothing (the NO-ENCODE lesion).

    def mouth_ok(self) -> bool:
        self._ensure_mouth()
        return bool(all(self.decode_ok) and self.moat_abstains)

    def speak(self, lesion: bool = False) -> dict:
        """Produce ONE self-initiated utterance. Two substrate paths, gated by selfinit_store_ok():
          * FULL (cupy / BRAIN_SELF_INITIATE_STORE=1): the real curiosity-biased noise CA3 wander SELECTS the basin
            (stochastic attractor competition), routed through the mouth (`_wander_speak`). This is the GO substrate.
          * LIGHT (numpy default / verify): the heavy CA3 wander is DEFERRED; the mouth decodes the stored concepts
            and the brain speaks the CURIOSITY-TOP decodable one (`_light_speak`) — the CONTENT is the mouth's spiking
            RF decode, ranked by the curiosity want; the stochastic multibasin WHICH is deferred to cupy.
        In BOTH paths the STORE NO-ENCODE lesion (an emptied RF store, not a host flag) collapses the stream -> n_utt=0
        -> the caller emits the honest neutral idle line."""
        self._ensure_mouth()
        return self._wander_speak(lesion) if selfinit_store_ok() else self._light_speak(lesion)

    def _base(self, lesion):
        return {"on": True, "lesioned": bool(lesion), "composer": "onebrain",
                "mouth_fidelity": bool(all(self.decode_ok)), "moat_abstains": bool(self.moat_abstains),
                "n_utt": 0, "about_rate": 0.0, "n_concepts_spoken": 0,
                "concept": None, "utterance": None, "question": None, "examples": []}

    def _dominant(self, out, agent):
        utt = self.utt_by_agent.get(agent)
        out["concept"] = agent
        out["utterance"] = utt
        out["question"] = (f"what does {agent} {utt.split()[1]}?" if utt and len(utt.split()) >= 2 else None)
        return out

    def _light_speak(self, lesion: bool) -> dict:
        """LIGHT path (numpy default): the heavy CA3 wander is DEFERRED. The mouth (RF unbind) decodes each stored
        concept; the brain self-initiates about the CURIOSITY-TOP decodable concept. n_utt = # decodable concepts (the
        surfaceable set). LESION = the NO-ENCODE mouth (emptied RF store) -> every render abstains -> n_utt=0."""
        out = dict(self._base(lesion), path="light-numpy-deferred-wander")
        if lesion:
            self._ensure_lesion_mouth()
            decodable = [i for i in range(self.n_mem) if self.comp_lesion.render_fact(self.agents[i]) is not None]
        else:
            decodable = [i for i in range(self.n_mem) if self.decode_ok[i]]
        out["n_utt"] = len(decodable)
        out["n_concepts_spoken"] = len(decodable)
        out["about_rate"] = 1.0 if decodable else 0.0
        if decodable:
            # curiosity-TOP: the most-novel decodable concept (the curiosity want ranks WHICH surfaces; identity to the
            # multibasin bias direction). Ties broken by index for determinism.
            nov = np.asarray(self.novelties, dtype=float)
            ci = max(decodable, key=lambda i: (nov[i], -i))
            out["examples"] = [{"concept": self.agents[i], "utterance": self.utt_by_agent.get(self.agents[i]),
                                "novelty": float(nov[i])} for i in decodable[:4]]
            self._dominant(out, self.agents[ci])
        return out

    def _wander_speak(self, lesion: bool) -> dict:
        """FULL path (cupy / forced): the real curiosity-biased noise CA3 wander SELECTS the basin (stochastic
        attractor competition under non-specific noise, 0 host content-draw), route the DOMINANT surfaced basin through
        the mouth. LESION = the CA3-store NO-ENCODE control (no assembly forms -> no coherent surfacing -> collapse)."""
        out = dict(self._base(lesion), path="ca3-wander-cupy")
        cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(self.n_mem)
        rest_steps = _wander_rest_steps()
        gains = [1.0] * self.n_mem if lesion else self.gains_on
        do_encode = not lesion
        ident = list(range(self.n_mem))
        F, prep, diag = _run_condition(self.seed, cfg, rest_steps, noise_on=True, gains=gains, do_encode=do_encode)
        st = _utterance_stream(F, prep["assemblies_local"], self.agents, self.utt_by_agent, self.decode_ok,
                               self.min_frac, ident)
        sel = _selection(F, prep["assemblies_local"], self.seed, self.min_frac)
        out.update(n_utt=int(st["n_utt"]), about_rate=float(st["about_rate"]),
                   n_concepts_spoken=int(st["n_concepts_spoken"]), share=st["share"],
                   examples=st["examples"][:4], max_pair_overlap=int(prep["max_pair_overlap"]),
                   weights_frozen=bool(diag.get("weights_frozen", False)),
                   pooled_member=float(sel["pooled_member"]), pooled_random=float(sel["pooled_random"]))
        counts = np.asarray(st["counts"], dtype=float)
        if counts.sum() > 0:
            self._dominant(out, self.agents[int(np.argmax(counts))])
        return out


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Per-conversation registry (keyed by the same cache_key the server uses for _SESSION_MOOD etc.; cleared on reset).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_ORGANS: dict = {}


def get_organ(cache_key, seed: int = 42) -> SelfInitiationOrgan:
    org = _ORGANS.get(cache_key)
    if org is None:
        org = SelfInitiationOrgan(seed=seed)
        _ORGANS[cache_key] = org
    return org


def reset_organ(cache_key) -> None:
    """Drop a conversation's self-initiation organ (called on the server's reset_conversation)."""
    _ORGANS.pop(cache_key, None)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The honest self-initiated surface text (an internally-selected remark + question about the surfaced concept) and
# the neutral idle fallback (when the wander surfaces nothing — deferred or lesioned). NEVER a phenomenal claim.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def self_initiated_text(result: dict) -> str:
    """Compose the self-initiated remark/question from a speak() result. The utterance is the bare spiking SVO
    proposition the wander selected + decoded; the surrounding frame is the declared host articulation scaffold."""
    utt = result.get("utterance")
    q = result.get("question")
    if not utt:
        return idle_fallback_text()
    lead = f"Something's been on my mind — {utt}."
    if q:
        return f"{lead} {q[0].upper() + q[1:]}"
    return lead


def idle_fallback_text() -> str:
    """The honest neutral idle read-out when nothing surfaces (the wander is deferred on numpy, or lesioned away).
    A functional read-out, never a fabricated topic — the moat's honesty extended to the idle turn."""
    return "Nothing in particular is surfacing for me to bring up right now."
