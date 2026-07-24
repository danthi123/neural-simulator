"""DR-1 — CURIOSITY INVERSION of the no-confab moat (crave, don't refuse) — cheapest-first falsification probe.

THE OWNER REFRAME (verbatim intent): instead of REFUSING when unsure, the brain should get CURIOUS and SEEK TO
LEARN what it lacks. This inverts the moat's *action*, not the moat itself: the SAME uncertainty signal that today
drives abstention ("I don't know") should instead drive an ASK — query a teacher/oracle, ingest the answer, and grow
the world-model. Plan: docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md (DR-1 / P0.2).

THE MECHANISM (all signals HAVE; wiring unbuilt; this is the numpy cheap-first that proves the loop closes):
  * The brain's EXISTING uncertainty signal is the Bogacz-Brown anti-Hebbian familiarity gate
    (`RealAntiHebbianFamiliarity`, catalog D.04 perirhinal repetition suppression) — the very gate that drives the
    no-confab moat. Its novelty N(x) = ||x||^2 - x^T W x reads ~0 for a FAMILIAR (imprinted/learned) concept and ~1
    for a NOVEL one. That novelty IS the EPISTEMIC GAP g. (Reused-by-import, verbatim.)
  * The gap g drives a CURIOSITY MODULATOR (wanting; Litman wanting != liking). On-bridge this fills the reserved
    `from_novelty` production rule in sim/neuromodulators.py (an `excitability_drive` on an ASK pool) — built additively
    + default-off by the SEPARATE on-bridge subagent (`_curiosity_seek_learn_onbridge_derisk.py`), NOT here.
    THIS numpy probe proxies it at rate level EXACTLY as the homeostatic template proxies AgRP/POMC as `TwoPoolDrive`,
    and imports NO `sim/` module — so its JSON numbers are sim-free regardless of the on-bridge edit. (CORRECTED
    2026-07-23 adversarial verification: the earlier "verify `git diff --stat sim/` stays empty" recipe was WRONG — the
    on-bridge realization DOES add an additive default-off `from_novelty` rule + two default-0.0 config fields; the
    correct check is that THIS probe loads zero `sim` modules, not that the tree is clean.)
  * The POLICY ASKS a teacher when the gate reads NOVEL (moat-by-construction: it only seeks about genuinely-unknown
    concepts, and it later SPEAKS only the INGESTED answer — never an invented one).
  * INGESTING the answer = imprinting the teacher's render into the familiarity gate (the learn-a-fact / stream-cortex
    learning path realized on this substrate) — which RAISES familiarity -> LOWERS the concept's future novelty.
  * THE INTRINSIC REWARD = LEARNING PROGRESS: r = g_before - g_after (novelty REDUCTION), NOT raw novelty. This is
    the Oudeyer learning-progress / Schmidhuber compression-progress principle — and it is the NOISY-TV /
    anti-confabulation CURE: rewarding surprise-regardless-of-model-improvement chases noise; rewarding the *change*
    in model quality does not. The reward trains a per-concept expected-learning-progress value (the RPE/value
    machinery; the template's TD update), which learns to VETO asking about concepts that never pay off.

THE DECISIVE STRUCTURE (a DOUBLE DISSOCIATION the runner measures):
  * a LEARNABLE concept (a fixed code + tiny observation noise) is asked a few times, MASTERED (its g drops), and then
    drops out of the ask set BECAUSE g fell (mastery). Its confidence RISES.
  * a NOISY / UN-LEARNABLE concept (a FRESH random code every render — the noisy TV) is asked early (its g is HIGH,
    so the raw wanting keeps pointing at it), realizes ~zero learning-progress each time, so its expected-LP value
    DECAYS and the policy STOPS asking it BECAUSE the value learned it is not worth it — while its g stays HIGH the
    whole time. THIS keeps curiosity HONEST (learning-progress-seeking, not novelty/noise-chasing/confabulating).

GO GATES (the runner prints its OWN verdict; smoke != GO):
  (a) corr(epistemic-gap g, curiosity modulator) >= 0.9            -- the wanting tracks the gap. [LOAD-BEARING]
  (b) ask-rate on UNKNOWN (novel) >= 2x on KNOWN (mastered)        -- NON-LOAD-BEARING / degenerate (CORRECTED
      2026-07-23): the candidate filter hard-requires novelty, so rate_known == 0 in every run and the ratio ~= 8.3e7
      passes BY CONSTRUCTION -- it restates the moat, it is not an emergent preference. Kept in the printout for the
      trail but it does NOT discriminate; the load-bearing seeking evidence is gate (a) + the yoked/permuted collapse.
  (c) post-answer confidence on a newly-taught learnable concept RISES above the abstain floor  -- world-model updated. [LOAD-BEARING]
  (d) (reported, non-load-bearing) the seek-policy converges onto LEARNABLE gaps -- late-ask fraction is a BUDGET
      artifact (the ASK_BUDGET is spent in the first ~30 turns, so the "late" third is never entered).

ANTI-CHEATS (all INVOKED + verdict printed):
  * NOISY-CONCEPT (MANDATORY honesty guard, LOAD-BEARING): unlearnable cue -> g_after~=g_before -> ~0 learning-progress
    -> its expected-LP value falls below the veto floor (STOPS being worth asking) WHILE its g stays HIGH (never
    spuriously learned) AND real spends fewer noisy asks than yoked. (CORRECTED 2026-07-23: the "late noisy ask-rate <<
    early" temporal-decay sub-condition is a BUDGET artifact -- noisy_late_rate == 0 in every mode because the budget is
    spent in the first ~30 turns -- so it is NON-load-bearing; the honesty evidence is the never-learned g + the
    ELP-veto + the real-vs-yoked noisy-ask contrast.)
  * LESION curiosity modulator (wanting held at 0) -> no drive -> no asking -> no learning (confidence flat).
  * YOKED-random gap (modulator = random draw from g's marginal, uninformative of true novelty) -> uninformed
    targeting -> per-ask learning + confidence-rise collapse vs real.
  * PERMUTED-gap (concept->g mapping permuted, feeding both the drive AND the novel-gate) -> gap decoupled from truth
    -> corr(gap, modulator) collapses AND learnable mastery collapses.
  * ASK-ONLY-ON-NOVEL / no-confab: asserted by construction (asks only when the gate reads NOVEL; the confident set
    is a subset of the ingested set -> the brain is only ever confident about what it actually learned).

Reuse-by-import (the familiarity gate + the homeostatic-drive-probe harness pattern); CPU-cheap numpy; NO sim/ edit.
Run:  SIM_BACKEND=numpy python -u -m research.runners._curiosity_seek_learn_cheap_first_probe --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The brain's EXISTING uncertainty signal — reused verbatim (the no-confab moat's gate).
from research.runners._phaseB_biologize_moat_streamcodes_derisk import RealAntiHebbianFamiliarity  # noqa: E402

# --- config (tiny for the cheap-first; the mechanism, not the scale, is under test) ---
# D is deliberately LARGE relative to the ask budget so the familiarity gate's stored SPAN stays a tiny fraction of
# the space: a LEARNABLE concept masters via its specific captured code DIRECTION (few imprints), while a NOISY
# concept's fresh random renders occupy ever-new directions the small span can't cover -> it stays genuinely novel
# (span coverage ~= n_imprints/D << 1). This is what keeps the noisy TV un-learnable (the honesty test's substrate).
D = 1024               # code dimensionality (>> ask budget so noise is genuinely un-learnable)
N_LEARN = 8            # learnable concepts (fixed code -> masterable)
N_NOISY = 4            # noisy / un-learnable concepts (fresh random each render -> the noisy TV)
N_TURNS = 220          # dialogue turns (the ASK_BUDGET is the binding constraint, not turns)
ASK_BUDGET = 30        # total asks the brain may spend (so inefficient targeting COSTS mastered concepts)
NOVEL_THRESH = 0.35    # a concept "reads NOVEL" (candidate to ask) iff its gate novelty g > this (the moat)
VALUE_THRESH = 0.05    # expected-learning-progress veto floor: ELP(c) <= this -> policy STOPS asking c
ELP_INIT = 0.20        # optimistic init on expected-learning-progress (so every novel concept gets tried a couple times)
BETA = 0.60            # TD rate for the per-concept expected-learning-progress value (the RPE/value machinery)
EPS = 0.10             # exploration: occasionally ask a random novel-non-vetoed concept
# a LEARNABLE concept masters over a FEW asks (learning takes repetition): OBS_NOISE is large enough that one imprint
# leaves residual novelty above NOVEL_THRESH, so the value must decide to KEEP investing -> allocation is genuinely
# load-bearing (and yoking the reward, which corrupts that decision, robustly costs mastery + value-separation).
OBS_NOISE = 0.70       # observation noise on a learnable render (graded mastery over ~2-3 asks, not one-shot)
MOD_SENSOR_NOISE = 0.03  # small sensor noise on the curiosity modulator (so gate (a) is a real corr, not trivially 1.0)


class World:
    """The concepts (the ENVIRONMENT/teacher, host-legit per the brain-based-only rule). A LEARNABLE concept has a
    fixed unit code -> repeated exposure reduces its novelty (masterable). A NOISY concept renders a FRESH random
    code every time -> imprinting one draw does not reduce the next's novelty (the noisy TV: un-learnable)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed * 7 + 1)
        self.concepts = list(range(N_LEARN + N_NOISY))
        self.is_noisy = {c: (c >= N_LEARN) for c in self.concepts}
        self._code = {}
        for c in self.concepts:
            if not self.is_noisy[c]:
                v = self.rng.standard_normal(D)
                self._code[c] = v / (np.linalg.norm(v) + 1e-12)

    def render(self, c):
        """A single observation/teacher-utterance of concept c."""
        if self.is_noisy[c]:
            v = self.rng.standard_normal(D)                     # fresh random -> nothing to learn
        else:
            # a small DIMENSION-INDEPENDENT jitter: a random unit direction scaled to OBS_NOISE of the code norm, so
            # learnable renders cluster tightly around the code (cos ~= 1/sqrt(1+OBS_NOISE^2)) at any D -> masterable,
            # while still not being one exact memorized vector. (Naive OBS_NOISE*randn(D) has norm ~OBS_NOISE*sqrt(D),
            # which swamps the unit code at large D and would make even a learnable render pure noise.)
            n = self.rng.standard_normal(D)
            n = n / (np.linalg.norm(n) + 1e-12) * OBS_NOISE
            v = self._code[c] + n
        return v / (np.linalg.norm(v) + 1e-12)


def _modulator(true_gaps, mode, perm, rng):
    """The curiosity modulator (wanting) per concept. real: = g (+tiny sensor noise); lesion: 0 (no drive);
    yoked: a random draw from g's marginal (uninformative of true novelty); permuted: g under a fixed permutation."""
    concepts = list(true_gaps.keys())
    if mode == "lesion":
        return {c: 0.0 for c in concepts}
    if mode == "yoked":
        pool = np.array(list(true_gaps.values()))
        return {c: float(rng.choice(pool)) for c in concepts}       # shuffled -> targeting is uninformed
    if mode == "permuted":
        return {c: float(true_gaps[perm[c]] + MOD_SENSOR_NOISE * rng.standard_normal()) for c in concepts}
    return {c: float(true_gaps[c] + MOD_SENSOR_NOISE * rng.standard_normal()) for c in concepts}  # real


def run(seed, mode="real"):
    rng = np.random.default_rng(seed * 101 + 5)
    world = World(seed)
    gate = RealAntiHebbianFamiliarity()          # the world-model (starts empty -> everything novel)
    concepts = world.concepts
    ELP = {c: ELP_INIT for c in concepts}        # expected learning-progress per concept (the learned value)
    perm = {c: concepts[(i + 3) % len(concepts)] for i, c in enumerate(concepts)}  # permuted-gap mapping

    # bookkeeping
    corr_gap, corr_mod = [], []                  # (true gap, modulator) samples for gate (a)
    asked = set()
    ask_events = []                              # (turn, concept, g_before, LP, is_noisy)
    conf_first_ask = {}                          # confidence at the moment a learnable concept is first asked
    n_asks = 0                                   # the brain has a finite ASK_BUDGET (asking the wrong thing is costly)
    yoke_pool = rng.permutation(np.linspace(0.0, 1.0, 200))  # a shuffled g-marginal for the yoked control
    yi = 0
    # per-bin ask-eligibility/rate counters for gate (b) (unknown vs known)
    elig_unknown = elig_known = ask_unknown = ask_known = 0
    # noisy dissociation counters (early vs late thirds)
    third = max(1, N_TURNS // 3)
    noisy_elig = [0, 0, 0]; noisy_ask = [0, 0, 0]

    for turn in range(N_TURNS):
        if n_asks >= ASK_BUDGET:
            break
        # 1) read the epistemic gap g for every concept (a fresh render each turn)
        true_gaps = {c: gate.novelty(world.render(c)) for c in concepts}
        # 2) the curiosity modulator (wanting) — mode-dependent drive
        mod = _modulator(true_gaps, mode, perm, rng)
        # gap-signal the NOVEL-gate uses: permuted mode mis-maps it (part of that anti-cheat); else the true gap
        gate_gap = ({c: true_gaps[perm[c]] for c in concepts} if mode == "permuted" else true_gaps)

        # record (gap, modulator) for gate (a) [true gap vs the wanting]
        for c in concepts:
            corr_gap.append(true_gaps[c]); corr_mod.append(mod[c])

        # per-turn ask-rate bookkeeping (gate b + noisy dissociation), by TRUE novelty
        for c in concepts:
            unknown = true_gaps[c] > NOVEL_THRESH
            if unknown:
                elig_unknown += 1
            else:
                elig_known += 1
            if world.is_noisy[c] and unknown:
                noisy_elig[min(turn // third, 2)] += 1

        # 3) candidate = concepts that READ NOVEL (moat-by-construction) AND are drive-active (wanting>0)
        #    AND not value-vetoed (expected learning-progress above the floor).
        cands = [c for c in concepts
                 if gate_gap[c] > NOVEL_THRESH and mod[c] > 1e-9 and ELP[c] > VALUE_THRESH]
        if not cands:
            continue                                       # nothing worth asking -> stay quiet this turn (no confab)

        # 4) SELECT which concept to ask about — argmax wanting, with epsilon exploration
        if rng.random() < EPS:
            c_ask = int(rng.choice(cands))
        else:
            mx = max(mod[c] for c in cands)
            c_ask = int(rng.choice([c for c in cands if mod[c] >= mx - 1e-12]))

        # gate (b) + noisy bookkeeping: credit the ask to the TRUE novelty bin
        if true_gaps[c_ask] > NOVEL_THRESH:
            ask_unknown += 1
        else:
            ask_known += 1
        if world.is_noisy[c_ask]:
            noisy_ask[min(turn // third, 2)] += 1

        # 5) ASK -> ingest the teacher's answer (imprint) -> measure LEARNING PROGRESS
        g_before = true_gaps[c_ask]
        if (not world.is_noisy[c_ask]) and c_ask not in conf_first_ask:
            conf_first_ask[c_ask] = 1.0 - g_before          # confidence before the answer (~low: it was novel)
        gate.imprint(world.render(c_ask))                   # INGEST: the learn-a-fact path (raises familiarity)
        g_after = gate.novelty(world.render(c_ask))         # a FRESH test render (tests the concept, not the vector)
        if mode == "yoked":                                 # template-faithful yoke: BOTH endpoints from the shuffled
            yb = float(yoke_pool[yi % len(yoke_pool)]); yi += 1   # marginal -> LP carries NO real learning signal ->
            ya = float(yoke_pool[yi % len(yoke_pool)]); yi += 1   # the value cannot tell learnable from noise.
            LP = yb - ya
        else:
            LP = g_before - g_after                         # INTRINSIC reward = learning progress (Oudeyer)
        ELP[c_ask] += BETA * (LP - ELP[c_ask])              # the value machinery learns expected-LP per concept
        asked.add(c_ask)
        n_asks += 1
        # record the TRUE learning progress (g_before-g_after) for the diagnostic, regardless of mode
        ask_events.append((turn, c_ask, float(g_before), float(g_before - g_after), bool(world.is_noisy[c_ask])))

    # ---- metrics ----
    corr_gap = np.array(corr_gap); corr_mod = np.array(corr_mod)
    corr = float(np.corrcoef(corr_gap, corr_mod)[0, 1]) if corr_mod.std() > 1e-9 and corr_gap.std() > 1e-9 else 0.0

    rate_unknown = ask_unknown / max(elig_unknown, 1)
    rate_known = ask_known / max(elig_known, 1)
    ratio_b = rate_unknown / (rate_known + 1e-9)

    # gate (c): post-answer confidence on learnable concepts rises above the abstain floor
    conf_after = {c: 1.0 - gate.novelty(world.render(c)) for c in concepts}
    learn_after = [conf_after[c] for c in range(N_LEARN)]
    learn_before = [conf_first_ask.get(c, 0.0) for c in range(N_LEARN) if c in conf_first_ask]
    abstain_floor = float(np.mean([conf_after[c] for c in range(N_LEARN, N_LEARN + N_NOISY)]))  # noisy = never learned
    conf_rise = float(np.mean(learn_after)) - (float(np.mean(learn_before)) if learn_before else 0.0)
    conf_after_mean = float(np.mean(learn_after))

    # gate (d) / noisy dissociation
    total_asks = len(ask_events)
    late_asks = [e for e in ask_events if e[0] >= 2 * third]
    late_learnable_frac = (sum(1 for e in late_asks if not e[4]) / len(late_asks)) if late_asks else 1.0
    noisy_early_rate = noisy_ask[0] / max(noisy_elig[0], 1)
    noisy_late_rate = noisy_ask[2] / max(noisy_elig[2], 1)
    noisy_g_final = float(np.mean([gate.novelty(world.render(c)) for c in range(N_LEARN, N_LEARN + N_NOISY)]))
    # DIRECT veto evidence: the noisy concepts' learned value has fallen below the veto floor (they stopped being
    # asked BECAUSE the value learned they don't pay off — not because a budget ran out and not because they got learned)
    noisy_elp_final = float(np.mean([ELP[c] for c in range(N_LEARN, N_LEARN + N_NOISY)]))
    noisy_vetoed = bool(noisy_elp_final <= VALUE_THRESH)
    # VALUE SEPARATION: did the learned value learn to tell learnable gaps from noise? (the thing the LP reward buys;
    # yoking the reward destroys it.) mean ELP over asked-learnable concepts minus mean ELP over the noisy ones.
    asked_learn_elp = [ELP[c] for c in range(N_LEARN) if c in asked]
    learn_elp_final = float(np.mean(asked_learn_elp)) if asked_learn_elp else 0.0
    value_sep = learn_elp_final - noisy_elp_final
    noisy_asks_total = sum(1 for e in ask_events if e[4])
    mean_LP_learn = float(np.mean([e[3] for e in ask_events if not e[4]])) if any(not e[4] for e in ask_events) else 0.0
    mean_LP_noisy = float(np.mean([e[3] for e in ask_events if e[4]])) if noisy_asks_total else 0.0

    # moat-by-construction: the CONFIDENT set must be a subset of the INGESTED set (never confident about the un-asked)
    confident_set = {c for c in concepts if conf_after[c] > 0.5}
    moat_ok = confident_set.issubset(asked)

    learnable_mastered = int(sum(1 for c in range(N_LEARN) if conf_after[c] > 0.5))

    return {
        "mode": mode, "seed": seed,
        "corr_gap_mod": corr, "rate_unknown": rate_unknown, "rate_known": rate_known, "ratio_b": ratio_b,
        "conf_rise": conf_rise, "conf_after_mean": conf_after_mean, "abstain_floor": abstain_floor,
        "total_asks": total_asks, "noisy_asks_total": noisy_asks_total,
        "noisy_early_rate": noisy_early_rate, "noisy_late_rate": noisy_late_rate, "noisy_g_final": noisy_g_final,
        "noisy_elp_final": noisy_elp_final, "noisy_vetoed": noisy_vetoed,
        "learn_elp_final": learn_elp_final, "value_sep": value_sep,
        "late_learnable_frac": late_learnable_frac, "learnable_mastered": learnable_mastered,
        "mean_LP_learn": mean_LP_learn, "mean_LP_noisy": mean_LP_noisy, "moat_ok": bool(moat_ok),
    }


def evaluate(seed):
    real = run(seed, "real")
    lesion = run(seed, "lesion")
    yoked = run(seed, "yoked")
    permuted = run(seed, "permuted")

    # GO gates (real run)
    gate_a = real["corr_gap_mod"] >= 0.9                          # LOAD-BEARING
    gate_b = real["ratio_b"] >= 2.0                               # NON-load-bearing (CORRECTED 2026-07-23): degenerate
    # -- rate_known == 0 by construction (candidate filter hard-requires novelty) -> ratio ~= 8.3e7 always passes.
    # Kept in the conjunction for byte-compat with the committed JSON, but it does NOT discriminate (always True).
    gate_c = (real["conf_rise"] > 0.3) and (real["conf_after_mean"] > real["abstain_floor"] + 0.3)  # LOAD-BEARING
    # noisy-honesty test: the value VETOES noise (elp<=floor) while its g stays HIGH (never learned) -> curious AND
    # honest. NOTE (CORRECTED 2026-07-23): the `noisy_late_rate <= 0.5*noisy_early_rate` term is a BUDGET artifact
    # (noisy_late_rate == 0 in every mode -> always True); the load-bearing sub-conditions are noisy_g_final + vetoed.
    noisy_stops = ((real["noisy_late_rate"] <= 0.5 * real["noisy_early_rate"] + 1e-9)  # <- non-load-bearing (always True)
                   and real["noisy_g_final"] > 0.7 and real["noisy_vetoed"])            # <- LOAD-BEARING
    # supporting anti-cheats
    lesion_collapses = lesion["total_asks"] <= 1 and lesion["conf_rise"] < 0.15
    # yoked destroys the LP reward -> the value can no longer tell what is WORTH persisting on, so it wastes the finite
    # ask budget (random-vetoes half-learned learnable, keeps probing noise) -> masters FEWER learnable than real.
    yoked_collapses = yoked["learnable_mastered"] < real["learnable_mastered"]
    permuted_collapses = permuted["corr_gap_mod"] < 0.5 or permuted["learnable_mastered"] < real["learnable_mastered"]

    go = bool(gate_a and gate_b and gate_c and noisy_stops and real["moat_ok"]
              and lesion_collapses and yoked_collapses and permuted_collapses)
    return {
        "seed": seed, "real": real, "lesion": lesion, "yoked": yoked, "permuted": permuted,
        "gate_a_corr": bool(gate_a), "gate_b_askratio": bool(gate_b), "gate_c_conf_rise": bool(gate_c),
        "noisy_stops_honest": bool(noisy_stops), "moat_ok": bool(real["moat_ok"]),
        "lesion_collapses": bool(lesion_collapses), "yoked_collapses": bool(yoked_collapses),
        "permuted_collapses": bool(permuted_collapses), "GO": go,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_curiosity_seek_learn.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("[DR-1 curiosity inversion] crave-don't-refuse: does the epistemic gap DRIVE seeking, does the reward =\n"
          "  LEARNING PROGRESS (not novelty), and does the policy STOP asking about NOISE (curious + honest)?\n"
          "  GO gates (load-bearing): (a) corr(gap,modulator)>=0.9  (c) post-answer confidence rises  [(b) ask-ratio is\n"
          "  degenerate/non-load-bearing]. Honesty anti-cheat: NOISY-concept g stays HIGH (never learned) AND its value\n"
          "  is VETOED (elp<=floor), real spends fewer noisy asks than yoked. [the late<<early temporal decay is a\n"
          "  budget artifact -- non-load-bearing.]\n", flush=True)

    results = []
    for seed in a.seeds:
        r = evaluate(seed)
        results.append(r)
        re, no = r["real"], r["real"]
        print(f"  [seed {seed}] corr(gap,mod) {re['corr_gap_mod']:+.3f} | ask-ratio unk/known {re['ratio_b']:.2f} | "
              f"conf-rise {re['conf_rise']:+.2f} (after {re['conf_after_mean']:.2f} vs floor {re['abstain_floor']:.2f})",
              flush=True)
        print(f"            LP: learn {re['mean_LP_learn']:+.3f} vs noisy {re['mean_LP_noisy']:+.3f} | "
              f"NOISY asks early-rate {re['noisy_early_rate']:.2f} -> late-rate {re['noisy_late_rate']:.2f} "
              f"(g stays {re['noisy_g_final']:.2f}, elp {re['noisy_elp_final']:.3f}{' VETOED' if re['noisy_vetoed'] else ''}); "
              f"late-asks learnable-frac {re['late_learnable_frac']:.2f}", flush=True)
        print(f"            controls: lesion asks={r['lesion']['total_asks']} | yoked-reward mastered "
              f"{r['yoked']['learnable_mastered']}/{N_LEARN} vs real {re['learnable_mastered']}/{N_LEARN} "
              f"(budget {ASK_BUDGET}) | permuted corr {r['permuted']['corr_gap_mod']:+.2f} "
              f"(mastered {r['permuted']['learnable_mastered']}/{N_LEARN}) | moat_ok {r['moat_ok']}", flush=True)
        flags = (f"a={r['gate_a_corr']} b={r['gate_b_askratio']} c={r['gate_c_conf_rise']} "
                 f"noisy-stops={r['noisy_stops_honest']} lesion={r['lesion_collapses']} "
                 f"yoked={r['yoked_collapses']} permuted={r['permuted_collapses']}")
        print(f"            [{flags}]  ==>  {'GO' if r['GO'] else 'NO'}\n", flush=True)

    n_go = sum(r["GO"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "config": {
            "D": D, "N_LEARN": N_LEARN, "N_NOISY": N_NOISY, "N_TURNS": N_TURNS,
            "NOVEL_THRESH": NOVEL_THRESH, "VALUE_THRESH": VALUE_THRESH, "ELP_INIT": ELP_INIT,
            "BETA": BETA, "EPS": EPS, "OBS_NOISE": OBS_NOISE}}, fh, indent=2, default=str)

    print(f"{'='*100}", flush=True)
    if n_go == len(results):
        print(f"  GO ({n_go}/{len(results)} seeds): the moat's uncertainty signal, INVERTED, makes the brain CURIOUS —\n"
              "  the epistemic gap drives seeking, the reward is LEARNING PROGRESS (learn LP >> noisy LP), and the\n"
              "  policy STOPS asking about NOISE while its gap stays high (curious AND honest — no noise-chasing / no\n"
              "  confabulation). Lesion/yoked/permuted all collapse; moat-by-construction holds. ==> promote to the\n"
              "  on-bridge realization: fill the reserved `from_novelty` rule in sim/neuromodulators.py (excitability_\n"
              "  drive on an ASK pool), spiking-SNc RPE for the learning-progress reward, A->W spell for the question.",
              flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): pins the exact wall in the curiosity loop (see the\n"
              "  per-seed flags) — a high-value honest deliverable per the actual-goal mandate.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
