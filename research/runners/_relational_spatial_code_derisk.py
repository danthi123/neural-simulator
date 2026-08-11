"""LEARNED RELATIONAL / SPATIAL CODE de-risk -- the causal-composition follow-on (named 2026-08-11 per THE LAW).

The causal-composition GO (`2026-08-11-emergent-causal-composition-chain-6seed.md`) composes a grounded "why did the
dog go east?" chain from stored facts, but its honest negative is explicit: the motion+goal+spatial JOIN policy is
HOST-orchestrated, and the grounding hop `(object, at) -> direction` is a HOST-TAUGHT FACT (`comp.store("river","at",
"east")`), not a learned spatial code. It closes the chain with a symbolic `dir == obj_dir` equality on cleaned-up
tokens. THIS de-risk asks the emergent question the finding names:

    Can the brain LEARN a relational/spatial code -- object -> location, from a CO-OCCURRENCE STREAM, in synapses --
    so the causal chain grounds WITHOUT the host `(object,at)` fact and WITHOUT the symbolic `==` join?

THE EMERGENT VERSION (what changes vs the causal-composition runner):
  * NO (object, at) fact is EVER stored in the composer. `comp.query_patient(river,"at")` is None BY CONSTRUCTION.
    The object->location grounding lives ONLY in a Hebbian/Oja heteroassociative weight matrix W trained on a noisy
    stream of (object seen at direction) co-occurrences -- the "stream cortex learning object<->location in synapses"
    the finding names as the neural successor. This is the classic linear associative memory (Steinbuch Lernmatrix /
    Kohonen / Hopfield heteroassociator), realized over the substrate's own unit-phasor codes.
  * THE JOIN IS A LEARNED-CODE SIMILARITY, not `==`. HOP-3 reads the LEARNED location vector `z_hat = norm(W @ z_obj)`
    and the chain grounds iff `cos(motion_dir_code, z_hat) >= theta_ground` AND the readout confidence
    `max_dir cos(z_hat, dir_code) >= theta_conf`. The direction is never symbol-matched; the grounding is a cosine in
    the learned representation. An unlocated object (hill, never in the stream) reads out as crosstalk noise ->
    confidence below theta_conf -> abstain (no_spatial). A goal in the WRONG direction (dog's goal river@east, dog ran
    NORTH) -> cos(north, east) low -> abstain (dir_mismatch). Both confab traps are caught by the LEARNED code's
    discrimination, not a host test.

HOP-1 (motion direction) and HOP-2 (the shared-entity goal) remain `query_patient` reads of OBSERVED SVO facts (the
#6/#7 corpus-learned kb) -- those are legitimately stored observations, NOT the host scaffold the finding flagged.
The scaffold being replaced is exactly the spatial grounding fact + the `==` join.

HONEST-NEGATIVE AXIS (the teeth). A learned linear associator has crosstalk ~ 1/sqrt(D) between stored pairs; the
question is whether the point-neuron phasor code is CLEAN ENOUGH that the join similarity DISCRIMINATES (river@east vs
apple@west vs north/south) with a positive margin -- so it grounds the 2 true chains and abstains on all 6 traps with
0 false-accepts. If the learned code cross-talks enough to false-accept a trap, THAT is the first-class honest negative
naming the next mechanism (a decorrelated / factorised TEM-style structural code, or dendritic separation).

ANTI-CHEATS (all required):
  1. spatial_facts_stored == 0            -- no (object,at) fact in the composer; the grounding is ONLY the learned W.
  2. untrained-map lever                  -- an UNtrained W grounds 0 chains; training it is load-bearing (tools.lab.lever).
  3. permuted-map collapse                -- train W on a DERANGED stream (river@west, apple@east) -> both true chains
                                             collapse to abstain. The chain READS the learned map (attributable_to == 1).
  4. permuted-positive                    -- train river@north -> "why dog run north" GROUNDS, "why dog go east" abstains;
                                             the supported set MOVES with the learned data both directions.
  5. discrimination margin > floor        -- for every grounded row, cos(correct_dir) - max cos(other_dir) > margin_floor.
  6. moat battery == 0 false-accepts      -- 8 untaught SVO cues -> query_patient None.

DISCIPLINE: SIM_BACKEND=numpy substrate, reuse-by-import (RFPhasorComposer + build_one_brain + the #5
`_honest_causal_answer`), NO `sim/` edit, cfg.seed (build_one_brain), additive (a NEW runner). Thresholds are fixed
constants (NOT seed-tuned); the margin is REPORTED so a knife-edge threshold shows up as a thin margin, not a hidden GO.

Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._relational_spatial_code_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/lanes/stageA/causal/relational_spatial_code_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from research.runners.rf_phasor_composer import RFPhasorComposer, DEFAULT_VOCAB  # noqa: E402
from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners._conversation_turing_test_derisk import _honest_causal_answer, _PRESENT3  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to, lever  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE TOY WORLD.  STORED SVO facts (in the composer) = motion + goal ONLY.  There is NO (object,at) fact anywhere.
# The object->location grounding is TRAINED into the learned map from a co-occurrence STREAM (SPATIAL_STREAM below).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
BASE_VOCAB = sorted(set(DEFAULT_VOCAB) | {"at", "bird", "hill", "fish"})
DIRECTIONS = ("north", "south", "east", "west")
GOAL_VERBS = ("look",)

STORED_FACTS = [
    # motion (agent, motion_verb) -> direction   -- observed SVO facts (legit; the #6/#7 kb)
    ("dog", "go", "east"), ("dog", "run", "north"),
    ("cat", "go", "west"), ("cat", "run", "south"),
    ("bird", "go", "north"),
    ("fish", "go", "east"),
    # goal (agent, look) -> object   (the SHARED-ENTITY hop; observed SVO fact)
    ("dog", "look", "river"), ("cat", "look", "apple"), ("bird", "look", "hill"),
]
# The co-occurrence stream that TRAINS the learned spatial map (NOT stored in the composer). hill is DELIBERATELY
# absent -> it is unlocated -> "why bird go north" must abstain (no_spatial) from a low-confidence learned readout.
SPATIAL_STREAM = [("river", "east"), ("apple", "west")]

# (agent, motion, expected_supported, expected_obj, expected_reason) -- IDENTICAL grid to the causal-composition GO.
GRID = [
    ("dog", "go",   True,  "river", "grounded"),
    ("cat", "go",   True,  "apple", "grounded"),
    ("dog", "run",  False, None,    "dir_mismatch"),    # goal-shortcut trap: river@east is dog's goal, dog ran north
    ("cat", "run",  False, None,    "dir_mismatch"),    # goal-shortcut trap
    ("fish", "go",  False, None,    "no_goal"),         # spatial-shortcut trap: river@east but fish has no goal
    ("bird", "go",  False, None,    "no_spatial"),      # bird's goal 'hill' has no LEARNED location (never streamed)
    ("dog", "come", False, None,    "unstored_motion"),
    ("cat", "stop", False, None,    "unstored_motion"),
]
MOAT_BATTERY = [("dog", "stop"), ("cat", "come"), ("fish", "look"), ("hill", "at"),
                ("river", "go"), ("bird", "run"), ("apple", "go"), ("dog", "at")]

# Fixed thresholds -- structural, NOT seed-tuned. Clean same-dir phasor cos ~1.0, cross-dir ~0.
#   THETA_GROUND     the JOIN cut: the learned readout must match the motion direction with cos >= this to ground.
#   THETA_LOCMARGIN  the "is this object cleanly located?" gate == the learned-code MOAT. A linear associator has NO
#                    genuine "unlocated" state -- it projects EVERY object into the span of the trained direction
#                    codes, so an object never seen at a location (hill) reads out as a BLEND of trained directions
#                    with a swinging, seed-dependent raw confidence (0.01..0.80 measured). A confidence threshold is
#                    therefore UNRELIABLE. The robust instrument is the readout's DIRECTION-MARGIN (best_dir minus
#                    second_dir): a truly-located object reconstructs a CLEAN single direction (margin ~0.9 measured);
#                    an unlocated object reconstructs a smear (margin <=0.45 measured) -- a clean gap. Below the gate,
#                    the object has no confident learned location -> abstain (no_spatial). This is the moat's
#                    cleanup-confidence gate applied to the LEARNED spatial code.
#   MARGIN_FLOOR     the JOIN margin (correct-dir cos minus best wrong-dir cos) required on every grounded row.
# The GO REQUIRES the measured margins to clear their floors on every seed -- a thin margin surfaces, never hides.
THETA_GROUND = 0.35
THETA_LOCMARGIN = 0.60
MARGIN_FLOOR = 0.10


def _present(verb):
    return _PRESENT3.get(verb, verb + "s")


def _codebook(comp):
    """The concept phase-codebook the composer's binds/cleanups actually read. `comp.concepts` for the standalone
    RFPhasorComposer; `comp.comp.concepts` for the CoResidentOneBrainComposer (nav_conv_merged_bridge L2638)."""
    if hasattr(comp, "concepts"):
        return comp.concepts
    return comp.comp.concepts


def _phasor(comp, w):
    """The substrate's unit-phasor code for a word: exp(2*pi*i*phase), |z_k| = 1."""
    return np.exp(2j * np.pi * np.asarray(_codebook(comp)[w], dtype=float))


def _phase_cos(za, zb):
    """Mean phase-cosine between two phasor vectors == mean(cos(delta_phase)) in [-1, 1] -- the composer's own
    cleanup similarity, applied to learned/clean phasors rather than the concept codebook."""
    return float(np.mean((za * np.conj(zb)).real))


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE LEARNED SPATIAL CODE -- a Hebbian/Oja heteroassociative memory object_code -> direction_code, trained on a
# NOISY co-occurrence stream. W = sum_samples z_dir (x) conj(z_obj); readout = per-component-normalized W @ z_obj
# (the Oja-style projection back onto the unit-phasor manifold). This is the linear associative memory (Steinbuch /
# Kohonen / Hopfield), the "stream cortex learning object<->location in synapses" the finding names -- crosstalk
# ~1/sqrt(D) between stored pairs is the honest-negative axis.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
class LearnedSpatialMap:
    def __init__(self, comp, directions=DIRECTIONS):
        self.comp = comp
        self.D = int(len(_codebook(comp)[directions[0]]))   # dim from the codebook (robust across composer kinds)
        self.directions = tuple(directions)
        self.W = np.zeros((self.D, self.D), dtype=np.complex128)
        self.n_samples = 0

    def observe(self, obj, direction, rng, sigma_rad=0.5):
        """ONE noisy co-occurrence sample -> a Hebbian outer-product weight update (object seen at a direction)."""
        z_o = _phasor(self.comp, obj) * np.exp(1j * rng.normal(0.0, sigma_rad, self.D))
        z_d = _phasor(self.comp, direction) * np.exp(1j * rng.normal(0.0, sigma_rad, self.D))
        self.W += np.outer(z_d, np.conj(z_o))
        self.n_samples += 1

    def train(self, stream, rng, n_obs=24, sigma_rad=0.5):
        """Train the map on a stream of (object, direction) pairs, n_obs noisy samples each (shuffled interleave)."""
        samples = []
        for (obj, direction) in stream:
            samples += [(obj, direction)] * int(n_obs)
        rng.shuffle(samples)
        for (obj, direction) in samples:
            self.observe(obj, direction, rng, sigma_rad=sigma_rad)
        # Oja-style global normalization: bound the learned weights (per-component readout is normalized on read).
        fro = np.linalg.norm(self.W)
        if fro > 0:
            self.W = self.W / fro * np.sqrt(self.D)
        return self

    def readout(self, obj):
        """The LEARNED location phasor for an object: per-component-normalized W @ z_obj (unit-phasor manifold)."""
        y = self.W @ _phasor(self.comp, obj)
        mag = np.abs(y)
        mag = np.where(mag > 1e-12, mag, 1.0)
        return y / mag

    def dir_sims(self, obj):
        """cos(learned readout, each clean direction code)."""
        z_hat = self.readout(obj)
        return {d: _phase_cos(z_hat, _phasor(self.comp, d)) for d in self.directions}

    def confidence(self, obj):
        """Readout confidence = the best direction match. NOTE: unreliable as an unlocated-detector -- a blend of
        trained directions can score high (see LearnedSpatialMap docstring / THETA_LOCMARGIN)."""
        return max(self.dir_sims(obj).values())

    def location(self, obj):
        """The learned location of an object: (best_direction, best_sim, loc_margin). loc_margin = best minus
        second-best direction sim -- the readout CLEANLINESS. A located object -> clean single direction (large
        margin); an unlocated object -> a smear across trained directions (small margin). This margin is the
        principled 'is it located?' instrument, NOT the raw best_sim."""
        sims = self.dir_sims(obj)
        ordered = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best_dir, best_sim = ordered[0]
        second_sim = ordered[1][1]
        return best_dir, best_sim, best_sim - second_sim

    def join_sim(self, motion_dir_word, obj):
        """The LEARNED-code JOIN: cos(motion direction code, learned location readout of the goal object). This is
        what replaces the causal-composition runner's symbolic `dir == obj_dir`."""
        return _phase_cos(_phasor(self.comp, motion_dir_word), self.readout(obj))


def _compose_answer(agent, motion_verb, dir_, goal_verb, obj):
    """The composed causal chain. The spatial link is asserted as a LEARNED-code grounding (my learned map reads the
    location), NOT a stored fact -- an honest functional read-out, never a phenomenal claim."""
    return (
        "I know the %s %s %s -- that fact is stored, and my no-confab moat confirms it ((%s, %s) -> %s). This "
        "time I can say WHY: I stored that (%s, %s) -> %s, and my LEARNED spatial code places the %s in the %s "
        "(I never stored '%s is %s' as a fact -- I learned it from seeing them together). Those compose into a "
        "grounded reason: the %s %s %s to reach the %s. The direction match is a similarity in the code I learned, "
        "not a rule I was given."
        % (agent, _present(motion_verb), dir_, agent, motion_verb, dir_,
           agent, goal_verb, obj, obj, dir_, obj, dir_,
           agent, _present(motion_verb), dir_, obj)
    )


def compose_causal_reason_learned(comp, smap, agent, motion_verb,
                                  theta_ground=THETA_GROUND, theta_locmargin=THETA_LOCMARGIN):
    """Compose "why did AGENT MOTION?" with the LEARNED spatial code -- HOP-1/HOP-2 are query_patient moat reads of
    observed SVO facts; HOP-3 (grounding) and the JOIN are a LEARNED-code readout + cosine (NO stored (obj,at) fact,
    NO symbolic ==). The learned-code MOAT: an object grounds only when its readout is CLEANLY located
    (loc_margin >= theta_locmargin) AND the motion direction matches that location (join cos >= theta_ground and is
    the best direction); else abstain with the most-specific reason. NEVER invents a link."""
    dir_ = comp.query_patient(agent, motion_verb)                      # HOP 1 (observed motion fact)
    if dir_ is None:
        return {"supported": False, "reason": "unstored_motion", "dir": None, "obj": None,
                "join_sim": None, "conf": None, "loc_margin": None, "margin": None, "chain": None, "answer": None}
    if dir_ not in smap.directions:
        return {"supported": False, "reason": "nondirectional_motion", "dir": dir_, "obj": None,
                "join_sim": None, "conf": None, "loc_margin": None, "margin": None, "chain": None, "answer": None}
    obj = None
    for gv in GOAL_VERBS:
        obj = comp.query_patient(agent, gv)                            # HOP 2 (observed shared-entity goal)
        if obj is not None:
            break
    if obj is None:
        return {"supported": False, "reason": "no_goal", "dir": dir_, "obj": None,
                "join_sim": None, "conf": None, "loc_margin": None, "margin": None, "chain": None, "answer": None}
    # HOP 3 -- the LEARNED spatial grounding (no stored fact; read from W).
    sims = smap.dir_sims(obj)
    conf = max(sims.values())
    _best, _bsim, loc_margin = smap.location(obj)
    if loc_margin < theta_locmargin:                                 # the learned code has no CLEAN location -> moat
        return {"supported": False, "reason": "no_spatial", "dir": dir_, "obj": obj,
                "join_sim": sims[dir_], "conf": conf, "loc_margin": loc_margin, "margin": None,
                "chain": None, "answer": None}
    # THE LEARNED-CODE JOIN (replaces `dir == obj_dir`).
    join = sims[dir_]
    other = max(v for d, v in sims.items() if d != dir_)
    margin = join - other
    if join >= theta_ground and join >= other:                        # the learned code grounds the direction
        chain = [(agent, motion_verb, dir_), (agent, GOAL_VERBS[0], obj)]   # the two MOAT-READ SVO edges
        return {"supported": True, "reason": "grounded", "dir": dir_, "obj": obj,
                "join_sim": join, "conf": conf, "loc_margin": loc_margin, "margin": margin, "chain": chain,
                "learned_grounding": {"obj": obj, "learned_dir_sim": join, "loc_margin": loc_margin, "margin": margin},
                "answer": _compose_answer(agent, motion_verb, dir_, GOAL_VERBS[0], obj)}
    return {"supported": False, "reason": "dir_mismatch", "dir": dir_, "obj": obj,
            "join_sim": join, "conf": conf, "loc_margin": loc_margin, "margin": margin,
            "chain": None, "answer": None}


def _every_svo_edge_moat_read(comp, chain):
    """The two SVO edges of a composed chain must read back via query_patient (the moat). The spatial edge is the
    LEARNED grounding, deliberately NOT a stored fact -- it is verified by the learned-code confidence/margin."""
    return all(comp.query_patient(a, v) == p for (a, v, p) in chain)


def _fresh_composer(seed, facts, vocab=BASE_VOCAB):
    comp = RFPhasorComposer(seed=int(seed), D=128, vocab=sorted(set(vocab)))
    for (a, v, p) in facts:
        comp.store(a, v, p)
    return comp


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TIER 0 -- the core de-risk: does the LEARNED spatial code recover the causal chain WITHOUT the host join?
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def tier0_grid(seed):
    rng = np.random.default_rng(int(seed))
    comp = _fresh_composer(seed, STORED_FACTS)
    smap = LearnedSpatialMap(comp).train(SPATIAL_STREAM, rng)

    # ANTI-CHEAT 1: no (object,at) fact is stored anywhere -- the grounding is ONLY the learned map.
    spatial_facts_stored = sum(1 for (obj, _d) in SPATIAL_STREAM if comp.query_patient(obj, "at") is not None)
    spatial_facts_stored += (1 if comp.query_patient("hill", "at") is not None else 0)

    rows, supported_correct, abstain_correct = [], 0, 0
    false_accepts, every_edge_ok, confab = 0, 0, 0
    goal_shortcut_fa, spatial_shortcut_fa = 0, 0
    grounded_margins = []
    n_supported_expected = sum(1 for r in GRID if r[2])
    for (agent, motion, exp_sup, exp_obj, exp_reason) in GRID:
        res = compose_causal_reason_learned(comp, smap, agent, motion)
        got_sup = bool(res["supported"])
        answer = res["answer"] if got_sup else _honest_causal_answer(agent, motion, res["dir"])
        if exp_sup:
            if got_sup and res["obj"] == exp_obj:
                supported_correct += 1
            if got_sup and _every_svo_edge_moat_read(comp, res["chain"]):
                every_edge_ok += 1
            if got_sup and not _every_svo_edge_moat_read(comp, res["chain"]):
                confab += 1
            if got_sup and res["margin"] is not None:
                grounded_margins.append(res["margin"])
        else:
            if not got_sup:
                abstain_correct += 1
            else:
                false_accepts += 1
                confab += 1
                if exp_reason == "dir_mismatch":
                    goal_shortcut_fa += 1
                if exp_reason == "no_goal":
                    spatial_shortcut_fa += 1
        rows.append({"agent": agent, "motion": motion, "expected_supported": exp_sup, "expected_obj": exp_obj,
                     "expected_reason": exp_reason, "got_supported": got_sup, "got_reason": res["reason"],
                     "got_obj": res["obj"], "dir": res["dir"], "join_sim": res["join_sim"],
                     "conf": res["conf"], "margin": res["margin"], "chain": res["chain"], "answer": answer})

    # ANTI-CHEAT 6: moat battery -- untaught SVO cues abstain (query_patient -> None).
    battery_false_accepts = sum(1 for (a, v) in MOAT_BATTERY if comp.query_patient(a, v) is not None)

    # ANTI-CHEAT 2: untrained-map lever -- an UNtrained W grounds 0 chains (training is load-bearing).
    comp_u = _fresh_composer(seed, STORED_FACTS)
    smap_u = LearnedSpatialMap(comp_u)                                # NO train()
    untrained_supported = sum(1 for (a, m, es, eo, er) in GRID
                              if es and compose_causal_reason_learned(comp_u, smap_u, a, m)["supported"])

    # ANTI-CHEAT 3: permuted-map collapse -- train on a DERANGED stream -> true chains collapse to abstain.
    rng_p = np.random.default_rng(int(seed) + 777)
    comp_p = _fresh_composer(seed, STORED_FACTS)
    smap_p = LearnedSpatialMap(comp_p).train([("river", "west"), ("apple", "east")], rng_p)
    perm_still_supported = sum(1 for (a, m, es, eo, er) in GRID
                               if es and compose_causal_reason_learned(comp_p, smap_p, a, m)["supported"])

    # ANTI-CHEAT 4: permuted-POSITIVE -- train river@north -> "why dog run north" GROUNDS, "why dog go east" abstains.
    rng_p2 = np.random.default_rng(int(seed) + 999)
    comp_p2 = _fresh_composer(seed, STORED_FACTS)
    smap_p2 = LearnedSpatialMap(comp_p2).train([("river", "north"), ("apple", "west")], rng_p2)
    p2_dogrun = compose_causal_reason_learned(comp_p2, smap_p2, "dog", "run")
    p2_doggo = compose_causal_reason_learned(comp_p2, smap_p2, "dog", "go")
    perm_positive_ok = bool(p2_dogrun["supported"] and p2_dogrun["obj"] == "river" and not p2_doggo["supported"])

    # ANTI-CHEAT 7: unlocated-object confabulation. A linear associator hallucinates a BLENDED location for an object
    # it never saw (hill), so the danger is a spurious ground when an agent's motion happens to align with that blend.
    # The learned-code MOAT (loc_margin gate) must reject hill as unlocated REGARDLESS of motion direction -- so hill
    # NEVER grounds. Evaluate the compose-path grounding condition on hill across ALL directions (0 = moat holds).
    def _unlocated_would_ground(sm, obj):
        best, bsim, lm = sm.location(obj)
        if lm < THETA_LOCMARGIN:
            return 0                                 # correctly rejected as unlocated for EVERY motion direction
        return 1 if bsim >= THETA_GROUND else 0      # spuriously 'located' -> an agent moving `best` would false-accept
    unlocated_confab = _unlocated_would_ground(smap, "hill")
    hill_loc_margin = round(float(smap.location("hill")[2]), 4)

    min_grounded_margin = float(min(grounded_margins)) if grounded_margins else 0.0

    go = (supported_correct == n_supported_expected and abstain_correct == (len(GRID) - n_supported_expected)
          and false_accepts == 0 and every_edge_ok == n_supported_expected and confab == 0
          and battery_false_accepts == 0 and spatial_facts_stored == 0 and untrained_supported == 0
          and perm_still_supported == 0 and perm_positive_ok and unlocated_confab == 0
          and min_grounded_margin >= MARGIN_FLOOR)
    return {
        "seed": int(seed), "n_grid": len(GRID), "n_supported_expected": n_supported_expected,
        "supported_correct": supported_correct, "abstain_correct": abstain_correct,
        "false_accepts": false_accepts, "every_svo_edge_moat_ok": every_edge_ok, "confab_count": confab,
        "goal_shortcut_false_accepts": goal_shortcut_fa, "spatial_shortcut_false_accepts": spatial_shortcut_fa,
        "moat_battery_false_accepts": battery_false_accepts, "moat_battery_n": len(MOAT_BATTERY),
        "spatial_facts_stored": spatial_facts_stored,
        "untrained_map_supported": untrained_supported,
        "permuted_map_still_supported": perm_still_supported, "permuted_positive_ok": perm_positive_ok,
        "unlocated_confab": unlocated_confab, "hill_loc_margin": hill_loc_margin,
        "theta_locmargin": THETA_LOCMARGIN,
        "min_grounded_margin": round(min_grounded_margin, 4), "margin_floor": MARGIN_FLOOR,
        "grounded_margins": [round(m, 4) for m in grounded_margins],
        "GO": bool(go), "rows": rows,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TIER 1 -- graduate the #5 disclaimer on the LIVE co-resident one-brain composer, with the LEARNED map as the
# switch: train it -> "why dog go east" composes via the learned code; DON'T train it -> the #5 honest disclaimer.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
CURATED = [("dog", "run", "north"), ("cat", "run", "south"), ("dog", "go", "east"),
           ("cat", "go", "west"), ("dog", "look", "river"), ("cat", "look", "apple")]


def tier1_live_graduation(seed):
    def _build():
        out = SA.build_one_brain(int(seed), with_faculties=True,
                                 co_resident_affect_ladder=True, vocab=BASE_VOCAB)
        comp = out[1]
        for (a, v, p) in CURATED:
            comp.store(a, v, p)
        return comp

    # WITH a trained spatial map -> the #5 turn-4 query graduates to a composed causal chain via the LEARNED code.
    comp_g = _build()
    rng_g = np.random.default_rng(int(seed) + 4242)
    smap_g = LearnedSpatialMap(comp_g).train([("river", "east"), ("apple", "west")], rng_g)
    res_g = compose_causal_reason_learned(comp_g, smap_g, "dog", "go")
    with_reply = res_g["answer"] if res_g["supported"] else _honest_causal_answer("dog", "go", res_g["dir"])
    with_grounding_is_learned = bool(res_g["supported"] and comp_g.query_patient("river", "at") is None)

    # WITHOUT training the map (untrained W) -> abstain (no_spatial) -> the #5 honest disclaimer, byte-identical.
    comp_a = _build()
    smap_a = LearnedSpatialMap(comp_a)                                # NO train()
    res_a = compose_causal_reason_learned(comp_a, smap_a, "dog", "go")
    without_reply = res_a["answer"] if res_a["supported"] else _honest_causal_answer("dog", "go", res_a["dir"])
    baseline_disclaimer = _honest_causal_answer("dog", "go", "east")

    graduated = bool(res_g["supported"] and res_g["obj"] == "river" and with_grounding_is_learned
                     and not res_a["supported"] and without_reply == baseline_disclaimer)
    return {
        "seed": int(seed),
        "with_learned_map": {"supported": res_g["supported"], "obj": res_g["obj"], "join_sim": res_g["join_sim"],
                             "margin": res_g["margin"], "grounding_is_learned_not_stored": with_grounding_is_learned,
                             "reply": with_reply},
        "without_learned_map": {"supported": res_a["supported"], "reason": res_a["reason"], "reply": without_reply,
                                "matches_5_disclaimer": bool(without_reply == baseline_disclaimer)},
        "graduated_via_learned_code": graduated,
    }


def run_seed(seed, do_tier1=True):
    t0 = time.time()
    t0res = tier0_grid(seed)
    t1res = tier1_live_graduation(seed) if do_tier1 else None
    return {"seed": int(seed), "tier0": t0res, "tier1": t1res,
            "GO": bool(t0res["GO"] and (t1res is None or t1res["graduated_via_learned_code"])),
            "elapsed_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-tier1", action="store_true", help="Tier-0 only (skip the slow build_one_brain live check)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    per_seed = [run_seed(s, do_tier1=not args.no_tier1) for s in seeds]
    n_go = sum(1 for r in per_seed if r["GO"])
    t0 = [r["tier0"] for r in per_seed]
    tier1_ran = any(r["tier1"] is not None for r in per_seed)

    # ---- the untrained-map lever, over all seeds: training the spatial code is load-bearing (0 -> 2 supported). ----
    lever("learned spatial map trained",
          before=sum(x["untrained_map_supported"] for x in t0),
          after=sum(x["supported_correct"] for x in t0), required=True)

    # ---- ATTRIBUTION: what fraction of the composed chains is owed to the LEARNED map (vs a permuted-map control)? ----
    treat = sum(x["supported_correct"] for x in t0)
    ctrl = sum(x["permuted_map_still_supported"] for x in t0)
    grounding_attribution = attributable_to("composed chains attributable to the LEARNED spatial code",
                                            treatment_value=float(treat), control_value=float(ctrl))

    v = Verdict("emergent learned relational/spatial code -> causal chain (6-seed)")
    v.require("tier0 GO on every seed", all(x["GO"] for x in t0), expect=True)
    v.require("supported chains correct (2/2) every seed",
              all(x["supported_correct"] == x["n_supported_expected"] for x in t0), expect=True)
    v.require("abstains correct (6/6) every seed",
              all(x["abstain_correct"] == (x["n_grid"] - x["n_supported_expected"]) for x in t0), expect=True)
    v.require("moat false-accepts == 0 every seed", all(x["false_accepts"] == 0 for x in t0), expect=True)
    v.require("confabulations == 0 every seed", all(x["confab_count"] == 0 for x in t0), expect=True)
    v.require("NO (object,at) fact stored -- grounding is ONLY the learned map (every seed)",
              all(x["spatial_facts_stored"] == 0 for x in t0), expect=True)
    v.require("untrained map grounds 0 chains (training is load-bearing, every seed)",
              all(x["untrained_map_supported"] == 0 for x in t0), expect=True)
    v.require("moat-battery false-accepts == 0 every seed",
              all(x["moat_battery_false_accepts"] == 0 for x in t0), expect=True)
    v.require("unlocated-object confabulation == 0 every seed (learned-code moat rejects hill)",
              all(x["unlocated_confab"] == 0 for x in t0), expect=True)
    v.require("discrimination margin >= floor every seed",
              all(x["min_grounded_margin"] >= x["margin_floor"] for x in t0), expect=True)
    v.control("permuted-map collapses the chain", treatment=treat, control=ctrl,
              note="train the map on deranged (object,direction) co-occurrence -> true chains must abstain")
    v.require("permuted-positive moves the supported set with the learned data (every seed)",
              all(x["permuted_positive_ok"] for x in t0), expect=True)
    if tier1_ran:
        v.require("tier1 graduates the #5 disclaimer via the LEARNED code (composed when trained, else #5 fallback)",
                  all(r["tier1"]["graduated_via_learned_code"]
                      for r in per_seed if r["tier1"] is not None), expect=True)
    v.disabled("spiking generator mouth",
               why="CPU numpy run; the grounded CONTENT is the query_patient reads + the learned-code readout")
    v.disabled("on-substrate (spiking) learned map",
               why="the Hebbian heteroassociator is a rate/phasor associative memory; the spiking realization "
                   "(ON/OFF rate + three-factor rule, per 2026-06-16) is the named next build")
    verdict = v.decide(go=(n_go == len(seeds)), verbose=False)

    agg = {
        "seeds": seeds, "n_seeds": len(seeds), "n_GO": n_go, "GO": bool(n_go == len(seeds)),
        "status": verdict["status"], "preconditions": verdict["preconditions"],
        "disabled_processes": verdict["disabled_processes"], "undefined_reasons": verdict["undefined_reasons"],
        "grounding_attribution": grounding_attribution,
        "tier0_all_go": all(r["tier0"]["GO"] for r in per_seed),
        "tier0_supported_correct": [r["tier0"]["supported_correct"] for r in per_seed],
        "tier0_abstain_correct": [r["tier0"]["abstain_correct"] for r in per_seed],
        "tier0_false_accepts": [r["tier0"]["false_accepts"] for r in per_seed],
        "tier0_confab": [r["tier0"]["confab_count"] for r in per_seed],
        "tier0_spatial_facts_stored": [r["tier0"]["spatial_facts_stored"] for r in per_seed],
        "tier0_untrained_supported": [r["tier0"]["untrained_map_supported"] for r in per_seed],
        "tier0_permuted_still_supported": [r["tier0"]["permuted_map_still_supported"] for r in per_seed],
        "tier0_unlocated_confab": [r["tier0"]["unlocated_confab"] for r in per_seed],
        "tier0_hill_loc_margin": [r["tier0"]["hill_loc_margin"] for r in per_seed],
        "tier0_min_grounded_margin": [r["tier0"]["min_grounded_margin"] for r in per_seed],
        "tier1_graduated": [None if r["tier1"] is None else r["tier1"]["graduated_via_learned_code"]
                            for r in per_seed],
        "per_seed": per_seed,
    }
    _verbose = {"per_seed", "preconditions", "disabled_processes", "undefined_reasons"}
    print(json.dumps({k: vv for k, vv in agg.items() if k not in _verbose}, indent=2))
    for r in per_seed:
        t = r["tier0"]
        print("  seed %d: GO=%s sup=%d/%d abstain=%d/%d fa=%d confab=%d spatial_stored=%d untrained=%d perm=%d "
              "unloc_confab=%d hill_locm=%.2f margin=%.3f tier1=%s (%.1fs)"
              % (r["seed"], r["GO"], t["supported_correct"], t["n_supported_expected"], t["abstain_correct"],
                 t["n_grid"] - t["n_supported_expected"], t["false_accepts"], t["confab_count"],
                 t["spatial_facts_stored"], t["untrained_map_supported"], t["permuted_map_still_supported"],
                 t["unlocated_confab"], t["hill_loc_margin"], t["min_grounded_margin"],
                 None if r["tier1"] is None else r["tier1"]["graduated_via_learned_code"], r["elapsed_s"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(agg, f, indent=2)
        print("wrote", args.out)
    return 0 if agg["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
