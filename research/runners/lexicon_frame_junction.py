"""FRAME-JUNCTION referent lexicon: heard context reaches the noun/non-noun category pools only through spiking
coincidence units, one per (word before, word after) pair (language lane E, 2026-09-24; default OFF).

PRE-REGISTRATION: research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md (committed
before this file). Selected by `BRAIN_LEARNED_REFERENT_JUNCTION=1` inside
`lexicon_spiking_frame_category.get_lexicon()`; it only matters when `BRAIN_LEARNED_REFERENT_LEXICON=1` routes the
lexicon into the D6 referent parse. Unset -> this module is never imported.

WHY. The single-offset lexicon (`lexicon_spiking_frame_category.SpikingFrameCategoryLexicon`) learned its noun
evidence almost entirely from the LEFT neighbour ("the word before is *the* / *a*"; frame_proxy_s7.json: held-out
nouns +85.7 left vs -3.6 right). Any word that follows a determiner is pushed toward "noun" ('the MOST beautiful',
'a WONDERFUL day'), and closed-class words the curriculum never covered sit on the decision boundary. Infants
categorize by the FRAME, the joint (before, after) context (Mintz 2003, Cognition 90:91; Chemla et al. 2009, Dev Sci
12:396: the discontinuous two-sided frame is what makes it work). One side added to the other is a different
signal. This variant replaces the additive single-offset edge by a conjunction.

WHAT IT IS (every step between the heard corpus and the decision is neurons/synapses, as in v2):
  * ENVIRONMENT (host, unchanged): the same FrameEnvironment presentation: K sampled real occurrences of the word,
    each driving its <=4 frame afferents FR(offset, context word) for T_ON steps.
  * JUNCTION POOL FJ (new): C x C excitatory neurons. J(a, b) has exactly two input synapses, from FR(-1, a) and
    FR(+1, b), at a fixed weight W_J at which a lone afferent does not fire J and the two together do: a two-input
    threshold AND on a point neuron, the stand-in for a thin-dendrite conjunction subunit (Polsky, Mel & Schiller
    2004, Nat Neurosci 7:621). The -2/+2 afferents project nowhere in this variant.
  * LEARNED EDGE: FJ -> CN (referent pool) and FJ -> CX (non-referent pool), all-to-all, uniform-with-jitter start,
    the SAME synapse-local Oja rule and teacher curriculum as v2, with the junction RATES as the pre factor. There is
    NO FR -> CN/CX edge: every route from heard context to a category pool needs both neighbours at once.
  * COMPETITION + READ-OUT: unchanged (reciprocal FSI lateral inhibition; the host reads the winning pool with the
    v2 MIN_RATE / DEAD_MARGIN abstain rule -- "spiking with a host read-out").
Nothing names the closed class to the circuit; no word list is read.

LESIONS (`set_lesion`, each verified at measurement by the inherited weight-hash check in `decide()`):
  "learned_edge"  FJ -> CN/CX restored to the start weights (route runner R4).
  "competition"   the reciprocal inhibitory weights zeroed.
  "afferent_zero" FJ -> CN/CX zeroed (integrity smoke).
  "coincidence"   every FR -> FJ weight set to OR_LESION_FACTOR x W_J, so one afferent delivers what two did: the
                  AND becomes an OR. The learned FJ -> CN/CX weights are untouched (the G3 lesion).
  "elemental"     (AMENDMENT 3, elemental variant only) the elemental FR(+-1) -> CN/CX edge zeroed.
  "conjunctive"   (AMENDMENT 3, elemental variant only) every FR -> FJ weight zeroed: no junction can fire.

CONSTANTS were set at DEV seed 7 (not an evaluation seed), by the rules of AMENDMENT 1 of the pre-registration:
  * T_ON_J = 50: the junction's first spike to a coincident pair comes 19-43 steps after onset across the committed
    calibration grid (and_calibration_s7.json; AMENDMENT 2 corrects a doc error here -- an earlier draft said 12-26,
    which matches no row of that grid), so no AND completes inside v2's 10-step occurrence (still ~5x shorter than a
    spoken word).
  * W_J, I_TONIC_J: the max-margin point of the `calibrate_and` grid (lone afferent held 150 steps -> 0 junction
    spikes; pair -> >= 1 spike within T_ON_J; on every sampled junction). and_calibration_s7.json.
  * DRIVE_MATCH_S: `measure_drive_ratio` (drive_ratio_s7.json); start weight, Oja rate and normalisation follow.

AMENDMENT 2 (2026-09-24, research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md):
three mechanism changes, each a new companion process, none a threshold hack on the DECISION:
  * SHORT-TERM DEPRESSION (Tsodyks-Markram; Abbott, Varela, Sen & Nelson 1997, Science 275:220) on the FR->FJ
    synapses ONLY (`stp_disabled` left False on the `fr_fj` explicit-wiring group, set True on `built`, so the
    learned FJ->CN/CX edge and the inhibitory pathways stay STP-free): a single fast-firing afferent (the 'day'
    column that drove 100 of 102 AND violations in the dev-s7-not-ready finding) DEPRESSES with repeated firing and
    can no longer deliver full-strength drive alone, while a fresh coincident PAIR still can. Re-calibrated via
    `and_population` (every junction, not a 64-sample) at dev seed 7; see W_J/I_TONIC_J below.
  * A SENTENCE-BOUNDARY PAUSE TOKEN in the heard token stream (`load_tokens_with_pause`; environment-only, the same
    "host is legitimate for the syllabus" boundary EMERGE-62b already used for the identical defect --
    research/findings/2026-07-03-emerge62b-position-cue-GO.md -- the shared tokenizer's `[a-z']+` regex strips ALL
    punctuation, so a sentence-final word's right-frame afferent silently spliced in the NEXT sentence's first word).
    DECLARED SIDE EFFECT (adversarial review, AMENDMENT 2): `FrameEnvironment.ctx` (the C=100 context words with a
    frame afferent) is built from RAW token counts with no exclusion list, and PAUSE_TOKEN is the single MOST
    FREQUENT token in the corpus (176,822 occurrences -- a sentence boundary is more common than any single word;
    measured on the full 19,971,040-byte tinystories.txt, seed-independent since `ctx` never depends on seed). It
    therefore wins a context-word slot on the same frequency-ranking basis every other context word does, displacing
    exactly ONE word from the top-100: 'make' (a common verb, not a curriculum or battery-critical word). This is
    the INTENDED mechanism, not incidental: PAUSE_TOKEN must occupy a real context-word slot to be usable as a frame
    neighbour at all. Not measured: whether losing 'make' as a neighbour-context measurably changes any OTHER
    word's frame evidence (plausible, not expected to be large -- one slot in 100).
  * A DRIVE-MATCHED OR CONTROL (`OR_MATCH_FACTOR`, lesion kind "coincidence_matched") alongside the existing
    "coincidence" (2x) lesion, and per-word CN-CX margins on every parse arm: the fixed 2x OR factor raises BOTH
    "this junction fires on either neighbour alone" AND the total population drive, so a G3 mismatch increase could
    be either. The matched control holds mean FJ drive equal to the intact arm's, isolating the conjunction.
  * R4 (the learned_edge lesion's uniform-jitter start weights deciding words at random): a RUNNER-SIDE Turrigiano
    synaptic-scaling settle (Turrigiano & Nelson 2004, Nat Rev Neurosci 5:97) of the FJ->CN/CX edge, engaged ONCE
    when the lesion first applies, before any decide() reads it -- see `_r4_homeostatic_settle`'s docstring for why
    this is runner-side (the engine's `enable_synaptic_scaling` clip bound, hebbian_max_weight=5.0 with Hebbian
    learning off on this circuit, is a BOUND TRAP at this circuit's ~1000-unit weight scale; `tools.lab.bound_check`
    exists for exactly this failure).

AMENDMENT 3 (2026-09-25; flag BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL, default OFF, read only when the junction
variant is routed; research/biology/elemental-partial-match-beside-conjunction.md): an ELEMENTAL partial-match edge
beside the junction edge. FR(-1,a) and FR(+1,b) -- the two afferent blocks every junction reads -- also project
straight to CN0/CX0 (2C x 2 N_CAT synapses, no STP), learned by the same Oja rule jointly with the junction edge, at
v2's OWN frame->category constants, unscaled (W_INIT_E / ETA_E / OJA_BETA_E): the junction edge keeps its
DRIVE_MATCH_S boost because junctions are sparse, the elemental afferents are not, so the elemental vote is the
unamplified, weaker component and no new constant is chosen. A lone input on a real conjunction branch is integrated
passively (small, not zero; Kandel ch. 13); a conjunctive code must still represent every active input (Marr 1969
section 4). Lesions added: "elemental" (the elemental edge zeroed; the AND pathway kept) and "conjunctive" (every
FR->FJ weight zeroed so no junction can fire; the elemental edge kept) -- AMENDMENT 3's G3', both of which REMOVE
drive; "learned_edge" now resets BOTH learned edges and the homeostatic settle scales both per postsynaptic neuron;
"afferent_zero" zeroes both. `drive_of(word)` records the afferent drive into each pool per edge (an instrument).
Flag unset -> the AMENDMENT 2 lexicon exactly (no group added to the wiring plan, no extra learning step).

HONEST RESIDUALS (declared in the pre-registration): the junction WIRING is host-designed (exhaustive, fixed, one
unit per (-1,+1) pair over the C most-heard words), not grown by development; the AND is a somatic threshold, not a
dendritic plateau (the engine's plateau lasts ~80 ms, and its coincidence count needs same-step spikes); the
junction threshold is set by a constant hyperpolarizing current (I_TONIC_J), the stand-in for tonic inhibition;
only the immediate frame; teacher-driven curriculum; host read-out; noun-hood not referent-hood.

Calibration (dev seed 7, AMENDMENT 1, pre-STP): SIM_BACKEND=numpy python -m research.runners.lexicon_frame_junction \
    --calibrate --seed 7 --json research/findings/raw/_lexicon_closed_class/and_calibration_s7.json
AMENDMENT 2 re-calibration (with STP; scored via `and_population`, every junction, not a sample):
    SIM_BACKEND=numpy python -m research.runners.lexicon_frame_junction --and-population --seed 7 \
        --weights=2900,2950,3000,3050,3100 --biases=-750,-755,...,-830 \
        --json research/findings/raw/_lexicon_closed_class/and_population_stp_grid_s7_width.json
    SIM_BACKEND=numpy python -m research.runners.lexicon_frame_junction --drive-ratio --seed 7 \
        --corpus /path/to/tinystories.txt \
        --json research/findings/raw/_lexicon_closed_class/drive_ratio_s7_amendment2.json
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import lexicon_spiking_frame_category as L  # noqa: E402

# ── constants (dev seed 7 calibration; see the module docstring and AMENDMENT 1 of the pre-registration) ─────────
# AMENDMENT 2: RE-calibrated with short-term depression now active on FR->FJ (STD lowers steady-state synaptic
# efficacy substantially, so the AND needs a much larger nominal weight than Amendment 1's STP-free W_J=300).
# Selection rule unchanged from Amendment 1 (widest feasible `and_population`-zero-violation bias range at a given
# weight; ties -> the smaller weight; middle of that range): and_population_stp_grid_s7*.json, dev seed 7, untrained
# _Env() circuit. W_J=2950 and 3000 tie at feasible width 15 (2950: -770..-755; 3000: -785..-770); 2950 wins (smaller).
W_J = 2950.0               # FR(-1,a) -> J(a,b) and FR(+1,b) -> J(a,b) weight (was 300.0 pre-STP)
I_TONIC_J = -762.5         # constant hyperpolarizing current on every junction (pA) (was -650.0 pre-STP)
T_ON_J = 50                # steps each heard occurrence drives its afferents in THIS variant (v2 uses L.T_ON = 10)
OR_LESION_FACTOR = 2.0     # "coincidence" lesion: one afferent delivers what two did (fixed a priori, not calibrated)
OR_MATCH_FACTOR = 1.02     # AMENDMENT 2 G3: drive-matched OR factor, measured by `measure_or_match_factor` at dev
                           # seed 7 (or_match_factor_s7.json; UNTRAINED circuit -- FJ firing does not depend on the
                           # trained FJ->CN/CX edge). HONEST FINDING, not the clean control it set out to be: mean
                           # FJ population rate is FLAT (0.0053-0.0065) for factor 1.02-1.1 then rises steeply from
                           # 1.2 (0.015) through 1.4 (0.256) to 1.6+ (>1.4) -- "matches the intact arm's rate" and
                           # "genuinely breaks the AND" are NOT jointly achievable with one scalar weight factor on
                           # this circuit: 1.02/1.05 tie for the closest match (gap 0.000167) and sit WELL INSIDE the
                           # zero-violation `and_population` feasible band (W_J 2950-3000 at this bias), i.e. this
                           # factor barely perturbs the AND at all. Frozen at 1.02 (ties -> the smaller value, same
                           # rule as the W_J/I_TONIC_J selection). Read `coincidence_matched` as a check that a small
                           # irrelevant weight change does not spuriously move the parse, NOT as a drive-matched OR.
STP_ENABLED = True         # AMENDMENT 2: short-term depression on FR->FJ synapses only (see module docstring)
# DRIVE MATCHING (AMENDMENT 1, RE-MEASURED under AMENDMENT 2). Junctions fire far more sparsely than v2's frame
# afferents, so v2's weight scale leaves the category pools silent after training. DRIVE_MATCH_S is MEASURED at dev
# seed 7 (`measure_drive_ratio`: v2 FR spikes/step over FJ spikes/step, mean over the untrained curriculum
# presentations; drive_ratio_s7_amendment2.json). With W_J = S*W, x_J = x/S, the rescaling below maps the junction
# Oja update exactly onto v2's (dW_J = S*dW): start weight x S, rate x S^2, normalisation / S^2.
# AMENDMENT 2: re-measured after STP + the higher W_J/I_TONIC_J + the pause-token environment (27.53 -> 165.1): STP
# suppresses the leaky firing the old ratio partly reflected (junctions fire far more RARELY now that a lone
# afferent can no longer drive one on its own), so junctions are sparser than before, not just differently scaled.
DRIVE_MATCH_S = 165.10471204188482
W_INIT_J = L.W_INIT * DRIVE_MATCH_S                   # FJ -> category start weight (uniform, jittered by L.W_JITTER)
ETA_J = L.ETA * DRIVE_MATCH_S ** 2                    # Oja rate
OJA_BETA_J = L.OJA_BETA / DRIVE_MATCH_S ** 2          # Oja normalisation strength

# AMENDMENT 3 (2026-09-25): the ELEMENTAL partial-match edge FR(-1,a), FR(+1,b) -> CN/CX, beside the junction edge
# (flag BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL, default OFF; read only when the junction variant is routed). Its
# constants are v2's OWN frame->category constants, UNSCALED -- deliberately no drive-matching boost: the junction
# edge is boosted by DRIVE_MATCH_S because junctions fire sparsely, the elemental afferents are not sparse, so the
# elemental vote is the unamplified (weaker) component and its weight relative to the conjunction follows from
# constants already frozen. No constant is chosen for this edge. research/biology/elemental-partial-match-beside-
# conjunction.md.
W_INIT_E = L.W_INIT                                   # elemental start weight (uniform, jittered by L.W_JITTER)
ETA_E = L.ETA                                         # elemental Oja rate
OJA_BETA_E = L.OJA_BETA                               # elemental Oja normalisation
ELEMENTAL_ENV = "BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL"
ELEMENTAL_JITTER_SEED = 0xE1E                         # the elemental start-weight jitter stream: rng([seed, this])

# "elemental" and "conjunctive" (AMENDMENT 3's G3' lesions) exist only on a lexicon built WITH the elemental edge.
LESION_KINDS = L.LESION_KINDS + ("coincidence", "coincidence_matched", "elemental", "conjunctive")
ELEMENTAL_ONLY_LESIONS = ("elemental", "conjunctive")
_OFF_L, _OFF_R = L.OFFSETS.index(-1), L.OFFSETS.index(1)

# AMENDMENT 2 mechanism C: a sentence-boundary PAUSE token fed into the heard stream as an ordinary context word
# (environment-only; see the module docstring and research/findings/2026-07-03-emerge62b-position-cue-GO.md, which
# added the identical sentence-aware front end for the identical `[a-z']+`-strips-all-punctuation defect). The
# sentinel cannot collide with a real corpus token (`[a-z']+` never matches it) and is never itself presented as a
# candidate referent (no battery turn's text can produce this string via D6._WORD_RE).
PAUSE_TOKEN = "\x00pause\x00"
_TOK_OR_BOUND_RE = re.compile(r"[a-z']+|[.?!]")


def load_tokens_with_pause(path: str, max_chars: int):
    """As `_comprehension_learned_animacy_cue_derisk.load_tokens`, but a sentence-ending mark [.?!] becomes ONE
    PAUSE_TOKEN in the flat token stream instead of being silently dropped (consecutive marks collapse to one
    PAUSE_TOKEN, e.g. "?!" or a stray double space before the next sentence). v2 never calls this function -- only
    `lexicon_spiking_frame_category.get_lexicon()`'s junction branch does -- so v2's own tokenisation, and its
    default-OFF byte-identity, are untouched."""
    txt = open(path, encoding="utf-8", errors="ignore").read(max_chars).lower()
    out = []
    for m in _TOK_OR_BOUND_RE.finditer(txt):
        t = m.group(0)
        if t in (".", "?", "!"):
            if out and out[-1] != PAUSE_TOKEN:
                out.append(PAUSE_TOKEN)
        elif t != "endoftext":
            out.append(t)
    return out


def junction_enabled() -> bool:
    """`BRAIN_LEARNED_REFERENT_JUNCTION` in {1,true,yes,on}. DEFAULT OFF."""
    v = os.environ.get("BRAIN_LEARNED_REFERENT_JUNCTION")
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def elemental_enabled() -> bool:
    """`BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL` in {1,true,yes,on} (AMENDMENT 3). DEFAULT OFF; it only has an
    effect when `BRAIN_LEARNED_REFERENT_JUNCTION` also routes the junction variant."""
    v = os.environ.get(ELEMENTAL_ENV)
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def elemental_afferents(fr, C: int):
    """The FR afferents the elemental edge reads: the -1 block then the +1 block (the two blocks each junction
    reads), 2*C neurons."""
    return np.concatenate([fr[_OFF_L * C:(_OFF_L + 1) * C], fr[_OFF_R * C:(_OFF_R + 1) * C]])


def build_junction_circuit(seed: int, n_frame: int, C: int, w_j: float = W_J, w_init: float = W_INIT_J,
                           elemental: bool = False, w_init_e: float = W_INIT_E):
    """FR (n_frame afferents) + FJ (C*C junctions) -> {CN0, CX0} with the v2 reciprocal FSI inhibition.

    The region framework builds every pathway except FR -> FJ (a RegionPathway cannot express "exactly these two
    presynaptic neurons"); the built connectivity is then re-installed verbatim together with the 2*C*C FR -> FJ
    synapses through `inject_explicit_wiring` (presynaptic polarity traits preserved).

    AMENDMENT 3 `elemental=True` adds one more explicit group: FR(-1,*) and FR(+1,*) -> CN0 and CX0, all-to-all
    (2*C x 2*N_CAT synapses), start weight w_init_e x (1 + N(0, L.W_JITTER)) floored at 0.01 (the RegionPathway
    jitter form v2's own frame->category edge uses), drawn from rng([seed, ELEMENTAL_JITTER_SEED]); no STP on these
    synapses. With `elemental=False` the plan is exactly AMENDMENT 2's (no group added)."""
    from sim import CoreSimConfig, GPUConfig, RuntimeState, SimulationBridge, VisualizationConfig
    from sim.backend import to_host
    from sim.enums import NeuronType
    from sim.regions import RegionPathway
    rs, fs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL, NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = False
    for flag in ("enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
                 "enable_synaptic_scaling"):
        setattr(cfg, flag, False)
    # AMENDMENT 2 mechanism A: short-term DEPRESSION on the FR->FJ synapses only (Tsodyks-Markram; per-type E->E
    # defaults U=0.5, tau_d=200ms, tau_f=20ms, sim/config.py). `stp_disabled=True` is set on the "built" wiring-plan
    # group below (FJ->CN/CX + the inhibitory pathways), so those stay STP-free -- the `fr_fj` group is left at its
    # default (stp_disabled omitted -> False -> gets STP). `cfg.enable_short_term_plasticity` must be set BEFORE
    # `SimulationBridge(...)` is constructed (`inject_explicit_wiring` reads it at call time to decide whether to
    # allocate/reset the per-synapse STP state arrays).
    cfg.enable_short_term_plasticity = bool(STP_ENABLED)
    regions = [L._region("FR", n_frame, exc_fraction=1.0, neuron_type=rs),
               L._region("FJ", C * C, exc_fraction=1.0, neuron_type=rs)]
    pathways = []
    for p in ("CN", "CX"):
        regions.append(L._region(f"{p}0", L.N_CAT, exc_fraction=1.0, neuron_type=rs))
    for p in ("IN", "IX"):
        regions.append(L._region(f"{p}0", L.N_FSI, exc_fraction=0.0, neuron_type=fs))
    for p in ("CN", "CX"):
        pathways.append(RegionPathway(from_region="FJ", to_region=f"{p}0", density=1.0, weight_mean=w_init,
                                      weight_jitter=L.W_JITTER, plastic=False))
    pathways.append(RegionPathway(from_region="CN0", to_region="IN0", density=1.0, weight_mean=L.TO_FSI_W,
                                  weight_jitter=0.05, plastic=False))
    pathways.append(RegionPathway(from_region="CX0", to_region="IX0", density=1.0, weight_mean=L.TO_FSI_W,
                                  weight_jitter=0.05, plastic=False))
    pathways.append(RegionPathway(from_region="IN0", to_region="CX0", density=1.0, weight_mean=L.CROSS_W,
                                  weight_jitter=0.05, plastic=False, receptor="gaba_a"))
    pathways.append(RegionPathway(from_region="IX0", to_region="CN0", density=1.0, weight_mean=L.CROSS_W,
                                  weight_jitter=0.05, plastic=False, receptor="gaba_a"))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b._initialize_simulation_data()
    rm = b.region_manager
    fr = np.asarray(list(rm.indices("FR")), dtype=np.int64)
    fj = np.asarray(list(rm.indices("FJ")), dtype=np.int64)
    coo = b.cp_connections.tocoo()
    pre0 = np.asarray(to_host(coo.row), dtype=np.int64)
    post0 = np.asarray(to_host(coo.col), dtype=np.int64)
    w0 = np.asarray(to_host(coo.data), dtype=np.float64)
    a_idx, b_idx = np.divmod(np.arange(C * C), C)                  # J(a, b) = fj[a*C + b]
    pre_l = fr[_OFF_L * C + a_idx]
    pre_r = fr[_OFF_R * C + b_idx]
    pre_j = np.concatenate([pre_l, pre_r])
    post_j = np.concatenate([fj, fj])
    plan = {
        # stp_disabled=True: FJ->CN/CX + the inhibitory pathways stay STP-free (AMENDMENT 2 mechanism A scopes the
        # depression to the FR->FJ coincidence-detector input only, not the learned edge or the competition).
        "built": {"pre_indices": pre0.tolist(), "post_indices": post0.tolist(), "initial_weights": w0.tolist(),
                  "plastic": False, "conn_type": "region-framework pathways, re-installed verbatim",
                  "stp_disabled": True},
        # stp_disabled omitted -> False -> these get real Tsodyks-Markram depression (E->E per-type params).
        "fr_fj": {"pre_indices": pre_j.tolist(), "post_indices": post_j.tolist(),
                  "initial_weights": [float(w_j)] * len(pre_j), "plastic": False,
                  "conn_type": "frame junction: FR(-1,a) and FR(+1,b) -> J(a,b)"},
    }
    if elemental:
        # AMENDMENT 3: the elemental (partial-match) edge. stp_disabled=True: like v2's frame->category synapses,
        # these carry no short-term depression (STP stays scoped to the FR->FJ coincidence-detector input).
        fr_e = elemental_afferents(fr, C)
        pools = np.concatenate([np.asarray(list(rm.indices("CN0")), dtype=np.int64),
                                np.asarray(list(rm.indices("CX0")), dtype=np.int64)])
        pre_e = np.repeat(fr_e, len(pools))
        post_e = np.tile(pools, len(fr_e))
        rng_e = np.random.default_rng([int(seed), ELEMENTAL_JITTER_SEED])
        w_e = np.maximum(0.01, float(w_init_e) * (1.0 + rng_e.normal(0.0, L.W_JITTER, size=len(pre_e))))
        plan["elemental"] = {"pre_indices": pre_e.tolist(), "post_indices": post_e.tolist(),
                             "initial_weights": w_e.tolist(), "plastic": False, "stp_disabled": True,
                             "conn_type": "elemental partial-match edge: FR(-1,a), FR(+1,b) -> CN0/CX0 (AMENDMENT 3)"}
    b.inject_explicit_wiring(plan)
    return b


class FrameJunctionLexicon(L.SpikingFrameCategoryLexicon):
    """See the module docstring. Same public surface as v2: train / decide / classify / is_referent / set_lesion."""

    variant = "junction"

    def __init__(self, seed: int, env, n_replicas: int = 1, *, eta=ETA_J, oja_beta=OJA_BETA_J,
                 teacher_i=L.TEACHER_I, i_frame=L.I_FRAME, k_occ=L.K_OCC, t_on=T_ON_J, epochs=L.EPOCHS,
                 w_j=W_J, w_init=W_INIT_J, i_tonic=I_TONIC_J,
                 elemental=None, w_init_e=W_INIT_E, eta_e=ETA_E, oja_beta_e=OJA_BETA_E):
        from sim.backend import to_host
        if int(n_replicas) != 1:
            raise ValueError("FrameJunctionLexicon is the deployment variant: n_replicas must be 1")
        self.seed, self.env, self.R = int(seed), env, 1
        self.eta, self.beta, self.teacher_i, self.i_frame = float(eta), float(oja_beta), float(teacher_i), float(i_frame)
        self.k_occ, self.t_on, self.epochs = int(k_occ), int(t_on), int(epochs)
        self.C, self.w_j, self.i_tonic = int(env.C), float(w_j), float(i_tonic)
        # AMENDMENT 3: the elemental edge. None -> follow the env flag (default OFF). The instance attribute
        # `variant` lets get_lexicon() rebuild when the requested variant differs.
        self.elemental = elemental_enabled() if elemental is None else bool(elemental)
        if self.elemental:
            self.variant = "junction_elemental"
            self.eta_e, self.beta_e = float(eta_e), float(oja_beta_e)
        self.b = build_junction_circuit(self.seed, env.n_frame, self.C, w_j=self.w_j, w_init=w_init,
                                        elemental=self.elemental, w_init_e=w_init_e)
        rm = self.b.region_manager
        self.fr = np.asarray(list(rm.indices("FR")), dtype=np.int64)
        self.fj = np.asarray(list(rm.indices("FJ")), dtype=np.int64)
        self.cn = [np.asarray(list(rm.indices("CN0")), dtype=np.int64)]
        self.cx = [np.asarray(list(rm.indices("CX0")), dtype=np.int64)]
        self.inn = [np.asarray(list(rm.indices("IN0")), dtype=np.int64)]
        self.ixx = [np.asarray(list(rm.indices("IX0")), dtype=np.int64)]
        self.n = int(self.b.core_config.num_neurons)
        self.post_groups = [self.cn[0], self.cx[0]]
        self.S = L._synapse_slots(self.b, self.fj, self.post_groups)             # (C*C, 2*N_CAT) learned edge
        self.S_inh = np.concatenate([L._synapse_slots(self.b, self.inn[0], [self.cx[0]]).ravel(),
                                     L._synapse_slots(self.b, self.ixx[0], [self.cn[0]]).ravel()])
        self.S_j = self._junction_input_slots()                                   # all 2*C*C FR -> FJ synapses
        data = np.asarray(to_host(self.b.cp_connections.data))
        self.W_init = data[self.S].astype(np.float64).copy()
        self.W = self.W_init.copy()
        self.inh_init = data[self.S_inh].astype(np.float64).copy()
        self.wj_init = data[self.S_j].astype(np.float64).copy()
        # AMENDMENT 2 R4 fix: the learned_edge lesion installs THIS (not W_init directly) so a homeostatic settle
        # can update it once, the first time the lesion engages (see `_r4_homeostatic_settle` / `set_lesion` below).
        self.W_lesion_settled = self.W_init.copy()
        self._le_settled = False
        if self.elemental:
            self.fr_e = elemental_afferents(self.fr, self.C)                      # 2*C afferents (-1 then +1)
            self.S_E = L._synapse_slots(self.b, self.fr_e, self.post_groups)      # (2*C, 2*N_CAT) elemental edge
            self.WE_init = data[self.S_E].astype(np.float64).copy()
            self.WE = self.WE_init.copy()
            self.WE_lesion_settled = self.WE_init.copy()
        self.lesion = None
        self._cache = {}
        self._drive = {}                 # (word, lesion) -> afferent drive into the pools (instrument; AMENDMENT 3)
        self._last_present = None
        self._install()

    # AMENDMENT 2 mechanism A: the per-presentation washout ALSO resets short-term-plasticity state (u->stp_U,
    # x->1), extending the ALREADY-DECLARED residual ("per-presentation state reset... = a washout convenience
    # between words", v2's docstring) to the new STP arrays. Needed for a clean, reproducible AND-integrity read
    # (`and_smoke`/`and_population`/`junction_response` share this washout via `present()`'s existing call and their
    # own direct calls below); real training/parsing still lets STP evolve CONTINUOUSLY within one presentation's
    # K_OCC occurrences, unaffected by this (the reset happens once, before the occurrence loop starts).
    def _reset_state(self):
        super()._reset_state()
        b = self.b
        stp_u, stp_x = getattr(b, "cp_stp_u", None), getattr(b, "cp_stp_x", None)
        if stp_u is not None and stp_x is not None:
            stp_u[:] = float(getattr(b.core_config, "stp_U", 0.15))
            stp_x[:] = 1.0

    def _junction_input_slots(self):
        from sim.backend import to_host
        indptr = np.asarray(to_host(self.b.cp_connections.indptr))
        indices = np.asarray(to_host(self.b.cp_connections.indices))
        is_fj = np.zeros(self.n, dtype=bool)
        is_fj[self.fj] = True
        slots = []
        for p in self.fr:
            ks = np.arange(indptr[p], indptr[p + 1])
            slots.append(ks[is_fj[indices[ks]]])
        S_j = np.concatenate(slots)
        if len(S_j) != 2 * self.C * self.C:
            raise RuntimeError(f"expected {2 * self.C * self.C} FR->FJ synapses, found {len(S_j)}")
        return S_j

    # ── weights on the bridge ────────────────────────────────────────────────────────────────────────────────
    def _install(self):
        from sim.backend import to_host, from_host
        data = np.asarray(to_host(self.b.cp_connections.data)).copy()
        W = self.W
        if self.lesion == "learned_edge":
            W = self.W_lesion_settled          # AMENDMENT 2: settled-then-frozen, not the raw pre-learning start
        elif self.lesion == "afferent_zero":
            W = np.zeros_like(self.W)
        data[self.S] = W
        data[self.S_inh] = 0.0 if self.lesion == "competition" else self.inh_init
        or_factor = {"coincidence": OR_LESION_FACTOR,
                    "coincidence_matched": (OR_MATCH_FACTOR if OR_MATCH_FACTOR is not None else OR_LESION_FACTOR),
                    "conjunctive": 0.0,       # AMENDMENT 3 G3': every FR->FJ weight zeroed -> no junction can fire
                    }.get(self.lesion, 1.0)
        data[self.S_j] = self.wj_init * or_factor
        if self.elemental:
            # AMENDMENT 3: the elemental edge. learned_edge -> its settled start weights (settled together with the
            # junction edge, per postsynaptic neuron); afferent_zero and the G3' `elemental` lesion -> zero.
            WE = self.WE
            if self.lesion == "learned_edge":
                WE = self.WE_lesion_settled
            elif self.lesion in ("afferent_zero", "elemental"):
                WE = np.zeros_like(self.WE)
            data[self.S_E] = WE
        self.b.cp_connections.data = from_host(data.astype(np.float32))
        self._w_hash = self.weight_hash()

    def set_lesion(self, kind):
        if kind not in (None,) + LESION_KINDS:
            raise ValueError(kind)
        if kind in ELEMENTAL_ONLY_LESIONS and not self.elemental:
            raise ValueError(f"lesion {kind!r} needs the elemental edge (AMENDMENT 3); this lexicon has none")
        # AMENDMENT 2 R4 fix: the FIRST transition into "learned_edge" runs a one-time homeostatic settle (below)
        # BEFORE the lesioned weights are installed for reading; later re-entries reuse the already-settled result
        # (deterministic, and matches "settle once after the lesioning event" biology, not per-read).
        entering_le = kind == "learned_edge" and self.lesion != "learned_edge" and not self._le_settled
        if entering_le:
            self._r4_homeostatic_settle()
            self._le_settled = True
        if kind != self.lesion:
            self.lesion = kind
            self._install()

    # ── R4 fix: homeostatic synaptic SCALING settle of the learned edge, once, on lesion entry ───────────────────
    def _r4_homeostatic_settle(self, epochs: int = 1, target_rate: float = 0.02, rate_gain: float = 0.05,
                               clip=(0.5, 2.0), settle_k_occ: int = 8):
        """Turrigiano-style homeostatic synaptic SCALING (Turrigiano 2008; Turrigiano & Nelson 2004, Nat Rev
        Neurosci 5:97) of the FJ->CN/CX learned edge's per-neuron TOTAL input, run ONCE when the `learned_edge`
        lesion first engages, BEFORE any `decide()` reads it: `scale = 1 + rate*(target-actual_rate)` per
        postsynaptic neuron -- the SAME formula `sim/config.py`'s engine-level `enable_synaptic_scaling` implements
        (`sim/bridge.py`'s fused synaptic-scaling block), executed RUNNER-SIDE on these synapses only, for the same
        reason the Oja rule above is runner-side and not the engine's generic path: the engine's own synaptic-
        scaling clip bound is `cfg.hebbian_max_weight if cfg.enable_hebbian_learning else 5.0`, and this circuit's
        weight scale (W_INIT_J ~ S x W_INIT, S = DRIVE_MATCH_S) is orders of magnitude above that -- flipping the
        engine flag would clip EVERY synapse in the bridge to <= 5.0 on the very first step (`tools.lab.bound_check`
        exists to catch exactly this BOUND TRAP). Biologically: this models the compensatory re-equilibration of
        population activity a real circuit runs after losing its learned synaptic specificity (the lesion), not a
        per-decision threshold -- the companion process the uniform-jittered start weights alone do not supply.
        `settle_k_occ` (< L.K_OCC=32) trades presentation fidelity for wall-clock cost on this dev-only settle pass
        (declared, not hidden): fewer sampled occurrences per word still gives each of the 40 CN0/CX0 neurons many
        rate readings across `epochs` x len(words) presentations, which is what the scale update needs -- it is not
        the fine-grained per-frame detail the Oja rule's OWN presentations (unaffected by this) require.

        GUARD (adversarial review, AMENDMENT 2): this settle only touches `data[self.S]` (the FJ->CN/CX edge) and
        installs it directly, bypassing `_install()` -- so it never re-writes `data[self.S_inh]` / `data[self.S_j]`.
        If some OTHER lesion ("coincidence", "competition", ...) were installed when `learned_edge` first engages,
        the settle would run its curriculum presentations against THOSE lesioned inhibition/junction weights instead
        of the intact circuit, and `self.W_lesion_settled` would silently reflect that. No current call site does
        this (`set_lesion` is always reached from an otherwise-intact lexicon before `learned_edge` is first
        requested), so this is a LATENT bug, not a manifested one -- asserted here so a future caller cannot
        introduce it silently."""
        assert self.lesion is None, (
            f"_r4_homeostatic_settle must run before any other manipulation is installed (self.lesion={self.lesion!r} "
            "here): it settles against whatever inhibition/junction weights are currently on the bridge, so a prior "
            "manipulation would get baked into the settled result instead of the default circuit's own weights.")
        from sim.backend import to_host, from_host
        words, _ = L.seed_curriculum(self.env)
        Wl = self.W_init.copy()
        # AMENDMENT 3: with the elemental edge, synaptic scaling is CELL-WIDE -- one scale per postsynaptic neuron,
        # applied to both learned edges it receives (both start from their pre-learning weights).
        WEl = self.WE_init.copy() if self.elemental else None
        rng = np.random.default_rng(self.seed + 5)
        data = np.asarray(to_host(self.b.cp_connections.data)).copy()
        saved_k_occ = self.k_occ
        self.k_occ = int(settle_k_occ)
        try:
            for _ in range(int(epochs)):
                for w in rng.permutation(np.asarray(words, dtype=object)):
                    data[self.S] = Wl
                    if WEl is not None:
                        data[self.S_E] = WEl
                    self.b.cp_connections.data = from_host(data.astype(np.float32))
                    counts, steps = self.present(str(w), teacher=None)
                    if counts is None:
                        continue
                    y = np.concatenate([counts[g] for g in self.post_groups]) / steps
                    scale = np.clip(1.0 + rate_gain * (target_rate - y), clip[0], clip[1])
                    Wl = Wl * scale[None, :]
                    if WEl is not None:
                        WEl = WEl * scale[None, :]
        finally:
            self.k_occ = saved_k_occ
        self.W_lesion_settled = Wl
        if WEl is not None:
            self.WE_lesion_settled = WEl

    # ── one presentation: v2's, plus the junctions' constant tonic current ───────────────────────────────────
    def present(self, word: str, teacher=None):
        """As v2 `present`, with I_TONIC_J added on every junction for the whole presentation."""
        from sim.backend import to_host, from_host
        occ = self.env.occurrences(word, self.k_occ, self.seed)
        if occ is None:
            return None, 0
        self._reset_state()
        base = np.zeros(self.n, dtype=np.float64)
        base[self.fj] = self.i_tonic
        if teacher is not None:
            if teacher[0] > 0:
                base[self.cn[0]] = self.teacher_i
            elif teacher[0] < 0:
                base[self.cx[0]] = self.teacher_i
        counts = np.zeros(self.n, dtype=np.float64)
        steps = 0
        b = self.b
        for feats in occ:
            cur = base.copy()
            if feats:
                cur[self.fr[np.asarray(feats, dtype=np.int64)]] += self.i_frame
            dev = from_host(cur.astype(np.float32))
            for _ in range(self.t_on):
                b.cp_external_input_current[:] = dev
                b._run_one_simulation_step()
                counts += np.asarray(to_host(b.cp_firing_states), dtype=np.float64)
                steps += 1
        b.cp_external_input_current[:] = 0.0
        self._last_present = (counts, steps)     # read by decide() for the afferent-drive instrument only
        return counts, steps

    # ── learning: the v2 Oja rule with the JUNCTION rates as the pre factor ──────────────────────────────────
    def hebbian_update(self, counts, steps):
        x = counts[self.fj] / steps
        y = np.concatenate([counts[g] for g in self.post_groups]) / steps
        self.W += self.eta * (np.outer(x, y) - self.beta * (y * y)[None, :] * self.W)
        if self.elemental:
            # AMENDMENT 3: the SAME synapse-local Oja rule on the elemental edge, v2's own rate/normalisation, the
            # same post factor (the pools' rates this presentation), pre = the -1/+1 afferent rates.
            xe = counts[self.fr_e] / steps
            self.WE += self.eta_e * (np.outer(xe, y) - self.beta_e * (y * y)[None, :] * self.WE)

    def reset_learning(self):
        if self.elemental:
            self.WE = self.WE_init.copy()
        super().reset_learning()

    # ── the afferent-drive instrument (AMENDMENT 3's G3' drive condition; never read by the decision) ─────────
    def decide(self, word: str):
        key = (word, self.lesion)
        fresh = key not in self._cache
        self._last_present = None
        out = super().decide(word)
        if fresh:
            lp = self._last_present
            self._drive[key] = None if (lp is None or lp[0] is None) else self._afferent_drive(*lp)
        return out

    def _afferent_drive(self, counts, steps):
        """Afferent drive into each category pool over one presentation, from the INSTALLED weights: sum over
        presynaptic units of (spikes / steps) x weight, averaged over the pool's N_CAT neurons, per edge. FR and FJ
        spike trains are feed-forward (they do not depend on the pools), so zeroing an edge can only lower this."""
        from sim.backend import to_host
        data = np.asarray(to_host(self.b.cp_connections.data), dtype=np.float64)
        n = L.N_CAT
        dj = (counts[self.fj] / steps) @ data[self.S]
        out = {"junction_cn": float(dj[:n].mean()), "junction_cx": float(dj[n:].mean())}
        if self.elemental:
            de = (counts[self.fr_e] / steps) @ data[self.S_E]
            out.update({"elemental_cn": float(de[:n].mean()), "elemental_cx": float(de[n:].mean())})
        out["total"] = float(sum(out.values()))
        return out

    def drive_of(self, word: str):
        """The recorded afferent drive for `word` under the CURRENT lesion and weights (None if not decided since the
        last weight change, or unheard). Tied to `_cache`, which training and reset_learning clear."""
        key = (word, self.lesion)
        return self._drive.get(key) if key in self._cache else None

    def graded_drive(self, word: str):
        """Diagnostic only: the CN-minus-CX drive of the word's COMPLETE (-1,+1) frames through the installed
        weights, assuming every complete frame fires its junction (the AND) and nothing else fires."""
        occ = self.env.occurrences(word, self.k_occ, self.seed)
        if occ is None:
            return None
        C, x = self.C, np.zeros(self.C * self.C)
        for feats in occ:
            left = [f - _OFF_L * C for f in feats if _OFF_L * C <= f < (_OFF_L + 1) * C]
            right = [f - _OFF_R * C for f in feats if _OFF_R * C <= f < (_OFF_R + 1) * C]
            for a in left:
                for bb in right:
                    x[a * C + bb] += 1
        data_w = {None: self.W, "learned_edge": self.W_lesion_settled,
                 "afferent_zero": 0 * self.W}.get(self.lesion, self.W)
        if self.lesion == "conjunctive":
            data_w = 0 * self.W                     # no junction can fire under the AMENDMENT 3 conjunctive lesion
        d = (x @ data_w).reshape(1, 2, L.N_CAT).mean(axis=2)
        if self.elemental:
            # AMENDMENT 3: plus the elemental edge's drive from every -1/+1 afferent the occurrences activate
            xe = np.zeros(2 * C)
            for feats in occ:
                for f in feats:
                    if _OFF_L * C <= f < (_OFF_L + 1) * C:
                        xe[f - _OFF_L * C] += 1
                    elif _OFF_R * C <= f < (_OFF_R + 1) * C:
                        xe[C + f - _OFF_R * C] += 1
            we = {"learned_edge": self.WE_lesion_settled, "afferent_zero": 0 * self.WE,
                  "elemental": 0 * self.WE}.get(self.lesion, self.WE)
            d = d + (xe @ we).reshape(1, 2, L.N_CAT).mean(axis=2)
        return d[:, 0] - d[:, 1]

    # ── the AND, measured (integrity smoke + dev calibration) ────────────────────────────────────────────────
    def junction_response(self, pairs, which: str, steps=None):
        """Spikes of junction J(a,b) over `steps` (default one occurrence window, T_ON_J) from a washed-out state,
        with the tonic current on, when driving only FR(-1,a) ('left'), only FR(+1,b) ('right') or both ('both').
        Uses the installed (possibly lesioned) weights; plasticity is off. Returns (counts, first-spike steps)."""
        from sim.backend import to_host, from_host
        C, steps = self.C, int(steps or self.t_on)
        counts, first = [], []
        for a, bb in pairs:
            self._reset_state()
            cur = np.zeros(self.n, dtype=np.float64)
            cur[self.fj] = self.i_tonic
            if which in ("left", "both"):
                cur[self.fr[_OFF_L * C + a]] += self.i_frame
            if which in ("right", "both"):
                cur[self.fr[_OFF_R * C + bb]] += self.i_frame
            dev = from_host(cur.astype(np.float32))
            j, n_sp, t0 = self.fj[a * C + bb], 0, -1
            for t in range(steps):
                self.b.cp_external_input_current[:] = dev
                self.b._run_one_simulation_step()
                s = int(np.asarray(to_host(self.b.cp_firing_states))[j])
                if s and t0 < 0:
                    t0 = t
                n_sp += s
            self.b.cp_external_input_current[:] = 0.0
            counts.append(n_sp)
            first.append(t0)
        return np.asarray(counts), np.asarray(first)


SUSTAIN_STEPS = 150   # a lone afferent must stay silent for 3 occurrence windows, not just one


def and_smoke(lex, n_sample: int = 64, seed: int = 0):
    """The pre-registered AND integrity smoke on `lex`'s INSTALLED weights (target (i), AMENDMENT 1 wording): over a
    random sample of junctions, a lone left or right afferent held for SUSTAIN_STEPS fires no junction, and the pair
    fires every sampled junction within one occurrence window (T_ON_J)."""
    rng = np.random.default_rng(seed)
    C = lex.C
    pairs = [(int(a), int(b)) for a, b in zip(rng.integers(0, C, n_sample), rng.integers(0, C, n_sample))]
    left, _ = lex.junction_response(pairs, "left", SUSTAIN_STEPS)
    right, _ = lex.junction_response(pairs, "right", SUSTAIN_STEPS)
    both, first = lex.junction_response(pairs, "both", lex.t_on)
    return {"n": n_sample, "lone_left_fired": int((left > 0).sum()), "lone_right_fired": int((right > 0).sum()),
            "pair_fired_in_window": int((both > 0).sum()), "pair_spikes_mean": float(both.mean()),
            "pair_first_spike_max": int(first.max()) if (first >= 0).all() else -1,
            "and_holds": bool((left == 0).all() and (right == 0).all() and (both > 0).all())}


def and_population(lex, sustain: int = None):
    """The AND checked on EVERY junction at once. Junctions have no lateral or feedback input (only their two FR
    afferents; FJ -> CN/CX is feed-forward), so driving ALL left (-1) afferents tests every junction's lone-left
    response, ALL right (+1) afferents every lone-right response, and both sets together every pair, in parallel.
    Uses the installed (possibly lesioned) weights. Returns per-junction violation counts."""
    from sim.backend import to_host, from_host
    C, sustain = lex.C, int(sustain or SUSTAIN_STEPS)
    left = lex.fr[_OFF_L * C:(_OFF_L + 1) * C]
    right = lex.fr[_OFF_R * C:(_OFF_R + 1) * C]

    def run(drive, steps):
        lex._reset_state()
        cur = np.zeros(lex.n, dtype=np.float64)
        cur[lex.fj] = lex.i_tonic
        cur[drive] += lex.i_frame
        dev = from_host(cur.astype(np.float32))
        counts = np.zeros(lex.n)
        for _ in range(steps):
            lex.b.cp_external_input_current[:] = dev
            lex.b._run_one_simulation_step()
            counts += np.asarray(to_host(lex.b.cp_firing_states))
        lex.b.cp_external_input_current[:] = 0.0
        return counts[lex.fj], counts / steps

    lone_l, rl = run(left, sustain)
    lone_r, rr = run(right, sustain)
    pair, _ = run(np.concatenate([left, right]), lex.t_on)
    L2, R2 = (lone_l > 0).reshape(C, C), (lone_r > 0).reshape(C, C)     # [a, b]
    rate_l, rate_r = rl[left], rr[right]
    return {"n_junctions": int(len(lex.fj)), "lone_left_fired": int((lone_l > 0).sum()),
            "lone_right_fired": int((lone_r > 0).sum()), "pair_silent": int((pair == 0).sum()),
            "pair_spikes_mean": float(pair.mean()),
            "and_violations": int(((lone_l > 0) | (lone_r > 0) | (pair == 0)).sum()),
            "and_holds_all": bool(((lone_l == 0) & (lone_r == 0) & (pair > 0)).all()),
            # afferent-level structure: a whole row (left word a) / column (right word b) firing alone
            "lone_left_full_rows": [int(a) for a in np.nonzero(L2.sum(axis=1) == C)[0]],
            "lone_right_full_cols": [int(b) for b in np.nonzero(R2.sum(axis=0) == C)[0]],
            "left_afferent_rate": {"median": float(np.median(rate_l)), "max": float(rate_l.max()),
                                   "argmax": int(rate_l.argmax())},
            "right_afferent_rate": {"median": float(np.median(rate_r)), "max": float(rate_r.max()),
                                    "argmax": int(rate_r.argmax())}}


def calibrate_and(seed: int, env, weights, biases, n_sample: int = 64):
    """Dev-seed grid over (W_J, I_TONIC_J) on the UNTRAINED junction circuit: the AND smoke at each point.
    Selection rule (AMENDMENT 1): the weight whose feasible bias range (AND holds) is widest, and the middle bias
    of that range (a max-margin choice; ties -> the smaller weight)."""
    rows = []
    for w in weights:
        lex = FrameJunctionLexicon(seed, env, w_j=float(w))
        for bias in biases:
            lex.i_tonic = float(bias)
            rows.append({"w_j": float(w), "i_tonic": float(bias), **and_smoke(lex, n_sample)})
    feas = collections_defaultdict_list()
    for r in rows:
        if r["and_holds"]:
            feas[r["w_j"]].append(r["i_tonic"])
    best = None
    for w in sorted(feas):
        bs = sorted(feas[w])
        width = bs[-1] - bs[0]
        if best is None or width > best[1]:
            best = (w, width, bs[len(bs) // 2] if len(bs) % 2 else 0.5 * (bs[len(bs) // 2 - 1] + bs[len(bs) // 2]))
    choice = None if best is None else {"w_j": best[0], "i_tonic": best[2], "feasible_bias_width": best[1]}
    return {"rows": rows, "choice": choice}


def measure_drive_ratio(seed: int, env):
    """AMENDMENT 1 drive matching: mean afferent spikes per step into the category pools, v2 (FR) over this variant
    (FJ), over the UNTRAINED presentations of the full seed curriculum (no teacher)."""
    v2 = L.SpikingFrameCategoryLexicon(seed, env)
    jx = FrameJunctionLexicon(seed, env)
    words, _ = L.seed_curriculum(env)
    a, b = [], []
    for w in words:
        c, st = v2.present(w)
        a.append(float(c[v2.fr].sum() / st))
        c, st = jx.present(w)
        b.append(float(c[jx.fj].sum() / st))
    return {"seed": seed, "n_words": len(words), "v2_fr_spikes_per_step_mean": float(np.mean(a)),
            "junction_fj_spikes_per_step_mean": float(np.mean(b)), "ratio_of_means": float(np.mean(a) / np.mean(b)),
            "per_word_ratio_median": float(np.median(np.asarray(a) / np.maximum(np.asarray(b), 1e-12))),
            "w_j": jx.w_j, "i_tonic": jx.i_tonic, "t_on_j": jx.t_on}


def measure_or_match_factor(lex, words=None, factors=(1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0),
                            k_occ: int = 8):
    """AMENDMENT 2 G3 fix: calibrate a DRIVE-MATCHED OR factor on `lex`'s TRAINED weights (the "coincidence" lesion's
    fixed 2x raises both "an afferent alone now fires the junction" AND total population drive together, so a parse
    mismatch increase under it does not separate "the conjunction mattered" from "there is simply more drive now").
    Sweeps FR->FJ weight factors, installing each on `self.S_j` only (leaving the learned FJ->CN/CX edge and
    inhibition at their trained/intact values), and measures the mean FJ population rate driven by real curriculum
    occurrences (the SAME operating regime the parse arms run in, not the synthetic all-or-nothing AND-smoke drive).
    Picks the factor whose mean rate is CLOSEST to the intact (AND) arm's own mean rate. Report-only control,
    alongside G3, not a replacement for it. Restores the lexicon's lesion state before returning.
    `k_occ` (< L.K_OCC=32, like `_r4_homeostatic_settle`'s `settle_k_occ`) trades presentation fidelity for
    wall-clock cost on this dev-only calibration sweep -- declared, not hidden."""
    from sim.backend import to_host, from_host
    if words is None:
        words, _ = L.seed_curriculum(lex.env)
    saved_k_occ = lex.k_occ
    lex.k_occ = int(k_occ)

    def mean_fj_rate():
        tot, n = 0.0, 0
        for w in words:
            counts, steps = lex.present(w)
            if counts is None:
                continue
            tot += float(counts[lex.fj].sum() / steps)
            n += 1
        return tot / max(n, 1)

    try:
        saved = lex.lesion
        lex.set_lesion(None)
        and_rate = mean_fj_rate()
        data = np.asarray(to_host(lex.b.cp_connections.data)).copy()
        tried = []
        best = None
        for f in factors:
            data[lex.S_j] = lex.wj_init * f
            lex.b.cp_connections.data = from_host(data.astype(np.float32))
            r = mean_fj_rate()
            gap = abs(r - and_rate)
            tried.append({"factor": float(f), "mean_fj_rate": r, "gap": gap})
            if best is None or gap < best["gap"]:
                best = tried[-1]
        lex.set_lesion(saved)   # restores the properly-installed state (and self._w_hash) via _install()
    finally:
        lex.k_occ = saved_k_occ
    return {"and_rate": and_rate, "best_factor": best["factor"], "best_rate": best["mean_fj_rate"],
            "gap": best["gap"], "tried": tried, "k_occ": int(k_occ)}


def collections_defaultdict_list():
    import collections
    return collections.defaultdict(list)


if __name__ == "__main__":
    import argparse
    import json
    import time
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default="300,400,500")
    ap.add_argument("--biases", default="-500,-600,-700,-800,-900")
    ap.add_argument("--n-sample", type=int, default=64)
    ap.add_argument("--drive-ratio", action="store_true")
    ap.add_argument("--and-population", action="store_true",
                    help="the AND on every junction at each (W_J, I_TONIC_J) grid point (--weights x --biases)")
    ap.add_argument("--corpus", default=L._DEFAULT_CORPUS)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    class _Env:                       # the AND needs only the afferent geometry, not the corpus
        C = L.CTX_C
        n_frame = L.CTX_C * len(L.OFFSETS)
    t0 = time.time()
    if a.calibrate:
        out = calibrate_and(a.seed, _Env(), [float(x) for x in a.weights.split(",")],
                            [float(x) for x in a.biases.split(",")], n_sample=a.n_sample)
        out.update({"seed": a.seed, "t_on_j": T_ON_J, "sustain_steps": SUSTAIN_STEPS,
                    "backend": os.environ.get("SIM_BACKEND"), "elapsed_s": round(time.time() - t0, 1)})
        for r in out["rows"]:
            print(r, flush=True)
        print("choice:", out["choice"])
        if a.json:
            dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(out, open(dst, "w"), indent=1)
            print("wrote", dst)
    if a.and_population:
        rows = []
        for w in [float(x) for x in a.weights.split(",")]:
            lexp = FrameJunctionLexicon(a.seed, _Env(), w_j=w)
            for bias in [float(x) for x in a.biases.split(",")]:
                lexp.i_tonic = bias
                rows.append({"w_j": w, "i_tonic": bias, **and_population(lexp)})
                print(rows[-1], flush=True)
        out = {"seed": a.seed, "rows": rows, "t_on_j": T_ON_J, "sustain_steps": SUSTAIN_STEPS,
               "frozen": {"w_j": W_J, "i_tonic": I_TONIC_J}, "backend": os.environ.get("SIM_BACKEND"),
               "elapsed_s": round(time.time() - t0, 1)}
        if a.json:
            dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(out, open(dst, "w"), indent=1)
            print("wrote", dst)
    if a.drive_ratio:
        from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab
        tokens = load_tokens(a.corpus, 8_000_000)
        vocab, _ = build_vocab(tokens, 2000)
        env = L.FrameEnvironment(tokens, vocab + [w for w in L.HAND_NOUN_SEEDS + L.NONNOUN_SEEDS if w not in vocab])
        out = measure_drive_ratio(a.seed, env)
        out.update({"backend": os.environ.get("SIM_BACKEND"), "corpus_path": a.corpus,
                    "elapsed_s": round(time.time() - t0, 1)})
        print(out)
        if a.json:
            dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(out, open(dst, "w"), indent=1)
            print("wrote", dst)
    print(f"{time.time() - t0:.1f}s")
