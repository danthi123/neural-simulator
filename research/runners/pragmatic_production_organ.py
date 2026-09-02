"""SCALAR-IMPLICATURE PRAGMATIC BELIEF wired into the PRODUCTION conversational turn (D-pragmatics, Task-#12, 2026-08-13).

WHAT THIS ORGAN IS. When the user makes a SCALAR-QUANTITY claim/probe ("I ate some of the cookies", "some of them",
"not all") the pragmatically-competent listener enriches the weak scalar term "some" with its IMPLICATURE: "some but
not all" (SBNA) is favored over the literal "all"-compatible reading -- yet "all" stays LOGICALLY possible. The brain
now FORMS a graded listener-belief distribution over the interpretations {none, SBNA, all} for that utterance and
SURFACES an honest functional pragmatic reading. This is the W4 depth-2 RSA (Frank & Goodman 2012) graded-implicature
belief -- the de-risk-CLOSED mechanism (`2026-08-13-w4-detector-operating-point-homeostat-GO.md`, 6/6) -- wired as the
production BELIEF SOURCE, replacing the winner-take-all ONE-HOT collapse the leg2_v2 de-risk pipeline used.

THE ONE-HOT vs GRADED belief (read from the substrate, NOT assumed -- verified in
`_pragmatic_graded_belief_source_derisk`):
  * ONE-HOT (the leg2_v2 WTA baseline, `onehot_belief_sources`): belief("some") = [none 0, SBNA 1, all 0]. The final
    hard-WTA `_compete` collapses the loser to EXACTLY 0 -> it FALSELY claims "all" is IMPOSSIBLE after "some".
  * GRADED (the W4 faithful RSA posterior, `graded_belief_sources`): belief("some") = [0, ~0.73, ~0.27], read one
    competition-step before the hard collapse. It carries the real "some -> not all" content (SBNA 0.73 preferred)
    while "all" stays ~0.27-possible -- matching the analytic Frank-Goodman L1("some") = [0, 0.75, 0.25]. The graded
    residual (0.27) is the calibrated pragmatic hedge the one-hot destroys.
  * MOAT / LESION (`BRAIN_PRAGMATIC_LESION=1` == the normalization-lesion, RSA_FS_EXC_W=0): belief("some") collapses
    to FLAT [0, 0.5, 0.5], implicature margin ~0 -- so the graded implicature content is ATTRIBUTABLE to the
    substrate's FS divisive normalization (the W4 mechanism), NOT host-injected. LOAD-BEARING: the read still runs,
    the belief flattens, the implicature vanishes.

HONEST SCOPE (this is a SCOPED wiring -- the production speaking pipeline had NO pragmatic-implicature slot before this;
`webapp/server.py` / `brain_chat` / the composer form no belief over interpretations). This organ adds the MINIMAL
genuine end-to-end path: a SINGLE scalar-implicature turn class (a scalar-quantity claim/probe -> the graded belief ->
an honest functional pragmatic reading prepended to the answer). It does NOT:
  * parse arbitrary pragmatic inference (only the {none, some, all} scalar family, in a partitive/quantity context);
  * change the recall/moat/abstain (it only PREPENDS a reading; never manufactures a fact, never flips an abstain,
    never enters the certainty band);
  * claim phenomenal access to another mind (a self-report would be a FUNCTIONAL read-out of the belief).
The gap -- a general pragmatic comprehension front-end (embedded/downward-entailing environments, non-lexical scalars,
Q-under-discussion) that would let this belief drive arbitrary pragmatic responses -- is mapped in the finding.

BRAIN-BASED. The graded belief is a spiking read of the real Izhikevich RSA substrate (`build_rsa_bridge` +
`_rsa_recursion`, plasticity OFF, a FIXED operating point -- exactly as the W4 GO specifies). It is computed ONCE at
organ-build and FROZEN (there is no learning to re-run; a live per-turn re-read is identical), then cached -- the same
build-once-freeze pattern the surprise/affect organs use. The only HOST boundary is the SENSORY encoding: mapping a
surface scalar token ("some"/"all"/"none" in a quantity context) to its RSA utterance (the legitimate environment
boundary, exactly like `surprise_production_organ.extract_assertion` mapping tokens to concept blocks).

Default-ON; `BRAIN_PRAGMATIC=0` -> fully skipped (byte-identical oracle). `BRAIN_PRAGMATIC_LESION=1` -> the
normalization-lesion belief (flat; load-bearing). NO `sim/` edit; reuse-by-import; process backend (cupy in
production, numpy in tests).

EXTERNAL GROUNDING: Frank & Goodman (2012) Science 336(6084):998 (the RSA depth-2 listener posterior). Grice (1975)
(scalar implicature: "some" +> "not all"). Carandini & Heeger (2012) Nat Rev Neurosci 13:51 (the FS divisive
normalization the graded content is attributable to). Builds on
`2026-08-13-w4-detector-operating-point-homeostat-GO.md` (arc CLOSED 6/6) and
`_pragmatic_graded_belief_source_derisk.py` (the faithful graded-vs-onehot belief source + the normalization-lesion moat).
"""
from __future__ import annotations

import os
import re

import numpy as np

# reuse-by-import: the faithful graded RSA posterior + the leg2_v2 WTA one-hot baseline + the RSA state/utterance names.
from research.runners._pragmatic_graded_belief_source_derisk import (
    graded_belief_sources,
    onehot_belief_sources,
    ANALYTIC_L1,
)
from research.runners._recursive_tom_rsa_derisk import STATES, UTTS, TRUTH, _rsa_recursion

# surface scalar-quantifier token -> the RSA utterance {none, some, all}. "some" is the implicature-bearing term
# (its graded belief is non-degenerate); "all"/"none" are handled but degenerate (one-hot regardless of belief source).
_SCALAR_TO_UTT = {
    "all": "all", "every": "all", "everything": "all", "everyone": "all", "everybody": "all",
    "each": "all", "entire": "all", "whole": "all", "both": "all",
    "some": "some", "several": "some", "few": "some", "part": "some", "partly": "some", "partially": "some",
    "none": "none", "nothing": "none", "nobody": "none", "neither": "none",
}
# the human-readable interpretation phrases per RSA state (the ENRICHED reading is the argmax state).
_STATE_PHRASE = {"none": "none", "SBNA": "some but not all", "all": "all"}
_LITERAL_PHRASE = {"none": "none", "some": "at least some (possibly all)", "all": "all"}
_WORD_RE = re.compile(r"[a-zA-Z']+")
IMPLICATURE_EPS = 0.05   # belief[SBNA]-belief[all] must exceed this for the implicature to be REPRESENTED


def pragmatic_enabled() -> bool:
    """Default-ON. `BRAIN_PRAGMATIC` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_PRAGMATIC")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def pragmatic_lesioned() -> bool:
    """`BRAIN_PRAGMATIC_LESION` in {1,true,yes,on} -> the normalization-lesion belief (flat; load-bearing lesion)."""
    v = os.environ.get("BRAIN_PRAGMATIC_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def extract_scalar_utterance(text: str):
    """Return the RSA utterance {none, some, all} when `text` makes a SCALAR-QUANTITY claim/probe, else None.

    Conservative (moat-safe): a scalar quantifier token must appear IN A QUANTITY CONTEXT -- a partitive
    "<scalar> of" (some of the / all of them / none of it), an explicit "not all"/"not every", or a direct
    scalar probe pairing a weak term with "all"/"every" in a question. This keeps casual "some" fillers
    ("tell me some facts") from firing a pragmatic reading. Prefers the implicature-bearing "some" when present.
    """
    raw = (text or "")
    toks = [w.lower() for w in _WORD_RE.findall(raw)]
    if not toks:
        return None
    low = raw.lower()
    # explicit implicature phrasings always qualify.
    if "not all" in low or "not every" in low or "not everything" in low:
        return "some"
    # find scalar tokens in a partitive/quantity context (<scalar> ... of, within 3 tokens).
    found = []  # (index_in_priority, utterance)
    for i, t in enumerate(toks):
        u = _SCALAR_TO_UTT.get(t)
        if u is None:
            continue
        window = toks[i + 1:i + 4]
        partitive = ("of" in window)
        found.append((u, partitive, i))
    if not found:
        return None
    partitives = [f for f in found if f[1]]
    # a direct scalar probe: a "some"-family term AND an "all"-family term co-occur in a QUESTION.
    utts_present = {f[0] for f in found}
    is_question = ("?" in raw) or (toks[0] in ("do", "does", "did", "is", "are", "was", "were", "have", "has"))
    probe = is_question and ("some" in utts_present) and ("all" in utts_present)
    if not partitives and not probe:
        return None
    # prefer the implicature-bearing "some"; else the first partitive; else the probe's weak term.
    for u, part, _ in partitives:
        if u == "some":
            return "some"
    if probe:
        return "some"
    return partitives[0][0]


class PragmaticProductionOrgan:
    """Process-shared spiking scalar-implicature organ. Built ONCE (lazily): the W4 graded RSA posterior belief
    (spiking Izhikevich substrate, plasticity off, fixed operating point) + the leg2_v2 WTA one-hot baseline, cached
    as {utterance: distribution over states}. The normalization-lesion belief (flat) is built on first lesioned read.
    A read INTERPRETS a scalar utterance -> the graded (or lesioned) listener-belief + the enriched pragmatic reading."""

    def __init__(self, seed: int = 42, shared=None):
        self.seed = int(seed)
        # ONE-BRAIN MERGE pool #2 (opt-in, default per BRAIN_ONEBRAIN_MERGE2): when a MergedSubstrate2 is injected,
        # the load-bearing GRADED belief is read from this organ's item/item_fs slice of the SHARED spiking bridge
        # it co-inhabits with the metacog organ (one cp_membrane_potential_v) instead of its own bridge. The one-hot
        # comparison arm (a separate speaker circuit) + the normalization-lesion belief stay standalone (diagnostics).
        # See research/runners/onebrain_merge_production2.py.
        self._shared = shared
        self._built = False
        self.graded = None      # {utt: np.array over STATES} -- the W4 faithful graded posterior (default belief)
        self.onehot = None      # {utt: np.array} -- the leg2_v2 WTA one-hot baseline (the A/B comparison arm)
        self.lesion = None      # {utt: np.array} -- the normalization-lesion (flat) belief, lazily built

    def ensure_built(self):
        if self._built:
            return
        if self._shared is not None:
            self.graded = self._graded_from_shared()                    # spiking substrate, frozen, on pool #2
        else:
            self.graded = graded_belief_sources(self.seed, normalize=True)   # spiking substrate, frozen
        self.onehot = onehot_belief_sources(self.seed)                   # the WTA one-hot baseline
        self._built = True

    def _graded_from_shared(self):
        """Compute the graded RSA L1 posterior on the SHARED pool-#2 bridge's item slice -- byte-identical to
        graded_belief_sources(seed, normalize=True) minus the standalone build (the recursion + state-normalize logic
        is reproduced verbatim). This routes the load-bearing spiking pragmatic read through the merged pool."""
        self._shared.ensure_built()
        b, xp = self._shared.bridge, self._shared.xp
        item_dev, snap = self._shared.pragmatic_item_dev(), self._shared.snap
        _L0, S1, _L1 = _rsa_recursion(b, xp, item_dev, snap, TRUTH, 25)
        out = {}
        for j, u in enumerate(UTTS):
            v = np.asarray(S1[j], dtype=np.float64).copy()
            if v.sum() <= 1e-9:
                v = np.array([TRUTH[u][s] for s in STATES], dtype=np.float64)
            out[u] = v / v.sum()
        return out

    def _ensure_lesion(self):
        if self.lesion is None:
            self.lesion = graded_belief_sources(self.seed, normalize=False)  # RSA_FS_EXC_W=0 -> flat
        return self.lesion

    def interpret(self, utterance: str, lesion: bool = False) -> dict:
        """Read the listener-belief distribution over interpretation states {none, SBNA, all} for a scalar
        `utterance` in {none, some, all}, plus the enriched pragmatic reading. `lesion` uses the flat
        normalization-lesion belief (load-bearing). Returns a FUNCTIONAL read -- never a phenomenal claim."""
        self.ensure_built()
        src = self._ensure_lesion() if lesion else self.graded
        bel = np.asarray(src[utterance], dtype=np.float64)
        oh = np.asarray(self.onehot[utterance], dtype=np.float64)
        i_sbna, i_all = STATES.index("SBNA"), STATES.index("all")
        margin = float(bel[i_sbna] - bel[i_all])
        enriched_state = STATES[int(np.argmax(bel))]
        # a near-tie (lesion collapse) has no represented implicature -> the reading is UNDIFFERENTIATED.
        represented = bool(margin > IMPLICATURE_EPS)
        calib_l1 = float(np.sum(np.abs(bel - ANALYTIC_L1[utterance])))       # distance to analytic Frank-Goodman RSA
        calib_l1_onehot = float(np.sum(np.abs(oh - ANALYTIC_L1[utterance])))
        return {
            "on": True, "lesioned": bool(lesion),
            "utterance": utterance, "belief_source": "graded_lesioned" if lesion else "graded",
            "states": list(STATES),
            "belief": [round(float(x), 4) for x in bel],
            "onehot_belief": [round(float(x), 4) for x in oh],
            "enriched_interpretation": _STATE_PHRASE.get(enriched_state, enriched_state) if represented
                                        else "undetermined (implicature collapsed)",
            "literal_interpretation": _LITERAL_PHRASE.get(utterance, utterance),
            "implicature_margin": round(margin, 4),
            "residual_all_prob": round(float(bel[i_all]), 4),     # the graded "all still possible" hedge (one-hot=0)
            "onehot_residual_all_prob": round(float(oh[i_all]), 4),
            "implicature_represented": represented,
            "calib_l1_to_analytic": round(calib_l1, 4),
            "calib_l1_to_analytic_onehot": round(calib_l1_onehot, 4),
        }

    def judge_text(self, text: str, lesion: bool = False):
        """Detect a scalar-implicature turn class in `text` and return its interpretation, or None (out of scope)."""
        u = extract_scalar_utterance(text)
        if u is None:
            return None
        return self.interpret(u, lesion=lesion)


_ORGAN: PragmaticProductionOrgan | None = None


def get_organ(seed: int = 42) -> PragmaticProductionOrgan:
    """The process-shared pragmatic organ (built once on first use). When the ONE-BRAIN MERGE pool-#2 flag is ON
    (`BRAIN_ONEBRAIN_MERGE2`, default per _MERGE2_DEFAULT_ON) the organ's graded belief is read from the
    process-shared MergedSubstrate2 it co-inhabits with the metacog organ (ONE spiking bridge); OFF -> its own
    bridge as today."""
    global _ORGAN
    if _ORGAN is None:
        # ONE-BRAIN SINGLE-POOL merge (opt-in, `BRAIN_ONEBRAIN_SINGLE_POOL`, default-OFF) WINS when on: all 4 core
        # organs co-inhabit ONE merge_organs pool. OFF -> the current pool-#2 pairwise path, byte-identical.
        from research.runners.onebrain_single_pool_production import single_pool_enabled, get_single_pool
        if single_pool_enabled():
            shared = get_single_pool(seed)
        else:
            from research.runners.onebrain_merge_production2 import merge2_enabled, get_merged_substrate2
            shared = get_merged_substrate2(seed) if merge2_enabled() else None
        _ORGAN = PragmaticProductionOrgan(seed=seed, shared=shared)
    return _ORGAN


def pragmatic_notice(info: dict) -> str:
    """The honest functional pragmatic reading surfaced when the graded belief carries the implicature. A FUNCTIONAL
    read of the spiking listener-belief -- never a phenomenal claim. Empty when the implicature is not represented
    (e.g. the lesion flattened the belief) so a collapsed read adds nothing misleading."""
    if not info or not info.get("implicature_represented"):
        return ""
    if info["utterance"] != "some":
        return ""   # only the scalar "some" bears a non-trivial implicature; "all"/"none" are degenerate
    sbna = info["belief"][STATES.index("SBNA")]
    allp = info["residual_all_prob"]
    return (f"(Pragmatically I read \"some\" as \"some but not all\" -- my listener model puts that at "
            f"{sbna:.2f}, though strictly \"all\" stays {allp:.2f}-possible.) ")
