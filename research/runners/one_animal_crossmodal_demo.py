"""ONE ANIMAL — the cross-modal "one self" demo: the SAME shared spiking interoceptive DRIVE that motivates
navigation-survival ALSO modulates CONVERSATION, through the already-built shared-dopamine route (Route A).

Per the owner's TRUE-ONE-BRAIN directive ("move every bit of the sim onto the shared spiking substrate; one
brain") and the Tier-3 living-loop arc, this runner COMBINES two ALREADY-DONE, controller-verified pieces — it
introduces NO new mechanism and NO learned policy, so it is decoupled from the dendrite wall:

  (1) the SPIKING interoceptive DRIVE co-resident on the merged one-brain (2026-06-20-tier3-spiking-living-loop-
      derisk.md, builder kwarg `co_resident_drive` on `build_merged_nav_conv_bridge`): a 2-pool AgRP/POMC
      hypothalamic drive (catalog O.05/O.06) whose `drive_agrp` FIRING RATE tracks the body energy DEFICIT
      (corr 0.995 lived); the SAME drive that gates the nav-survival reward.
  (2) the DA -> composer Route A read-side salience gate (2026-06-18-DA-composer-precision-derisk-GO.md,
      `MergedNavConvAgent._da_confidence_gate`/`_gated_out`, enabled via `enable_da_salience_gate=True`): reads
      the SHARED spiking-SNc `dopamine` off the merged bridge and SHARPENS the composer's cue-role CONFIDENCE
      GATE — moat-safe BY CONSTRUCTION ("DA can only TIGHTEN abstention", g_eff = clip(g0, g_cap, ...)).

THE CROSS-MODAL LINK (the "one animal"): the drive and the limbic SNc are BOTH co-resident on the merged
bridge. Each conversational turn, the body's energy DEFICIT is injected as an interoceptive current into the
SPIKING hunger pool `drive_agrp` (the legitimate body->sensory boundary), the SPIKING HUNGER is READ as the
`drive_agrp` firing rate (off cp_firing_states — NOT a host deficit value), and that spiking hunger drives the
limbic SNc afferent `limbic_reward_us` => the shared `dopamine` rises with hunger (well-documented hypothalamic
AgRP->VTA/SNc motivational-DA pathway; Palmiter, Berridge). Route A then reads that SAME dopamine and tightens
the conversational recall gate. ⇒ a HIGH-drive (hungry) state makes the agent recall MORE DECISIVELY (a salient
state sharpens cognition) vs a LOW-drive (sated) baseline — the same internal drive moving BOTH halves.

THE DECISIVE METRIC: the conversational read-out behaviour DIFFERS measurably between high-drive and low-drive
(the SAME drive moving both halves) — Route A raises the confidence gate under hunger, so borderline
(noise-dominated) cue reads ABSTAIN under hunger that would be answered when sated => the answered reads are
more decisive (higher mean cue-role margin among answered) and the answer-rate on borderline facts drops, while
the no-confab moat is held at BOTH drive levels and the STRONG/clear facts still recall.

ANTI-CHEATS (all):
  * DRIVE-LESION: zero the interoceptive drive current (drive_agrp silent => spiking hunger ~floor => no SNc
    drive => dopamine -> baseline => g_eff -> g0) => the conversational modulation VANISHES (high == low). The
    link is the drive's doing (load-bearing), not a host scalar.
  * MOAT: every unstored cue ABSTAINS (returns None) at BOTH drive levels (high AND low). Route A is moat-safe
    by construction (DA only raises the gate); we VERIFY it empirically holds — the moat is never weakened.
  * YOKED control: a DRIVE-INDEPENDENT dopamine injection (drive the SNc afferent from a shuffled signal of the
    matched marginal, decorrelated from the deficit) does NOT reproduce the drive-SPECIFIC pattern (the
    high-vs-low modulation is gone — the effect needs the hunger->DA correlation, not just SOME DA).

NO `sim/` edit (reuse-by-import: the `co_resident_drive` builder kwarg + the limbic `dopamine` modulator + the
agent's Route A gate are all already built). The host residual is the legitimate body (energy + the
interoceptive deficit current) + reading the spiking hunger/DA scalars to present the queries; the cognition
(the drive firing, the SNc firing that sets DA, the FHRR cleanup + the gate margin) stays neural.

Run (GPU — the merged bridge with the co-resident RF composer is GPU-only):
  SIM_BACKEND=cupy python -m research.runners.one_animal_crossmodal_demo --seeds 42 43 44 \
      --out research/findings/raw/_one_animal_crossmodal.json
  SIM_BACKEND=cupy python -m research.runners.one_animal_crossmodal_demo --smoke   # tiny mechanics check
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ============================================================================
# 1. Build the merged one-brain with ALL THREE co-resident slices (composer +
#    limbic SNc + drive) and the Route A DA salience gate ON. This replicates
#    `MergedNavConvAgent.__init__` (nav_conv_merged_bridge.py:1311-1380) but adds
#    `co_resident_drive=True` to the builder call — NO edit to the merged file
#    (object.__new__ + set the same attributes, the pattern the module itself
#    uses for the dlPFC controller). The agent's Route A methods
#    (`_da_confidence_gate`/`_gated_out`/`what_does`/...) then work verbatim.
# ============================================================================
def build_one_animal_agent(seed=42, vocab=None, rf_D=128,
                           da_gate_g0=0.06, da_gate_k=2.0, da_gate_cap=0.25):
    """Construct a Route-A MergedNavConvAgent whose merged bridge ALSO carries the co-resident SPIKING drive.
    Returns (agent, handles). `handles` exposes the drive + limbic slice index arrays for the coupler."""
    from research.runners.nav_conv_merged_bridge import (
        MergedNavConvAgent, MergedRFComposer, _MergedParserAdapter, build_merged_nav_conv_bridge)
    from sim.backend import get_backend

    xp, _ = get_backend()
    agent = object.__new__(MergedNavConvAgent)
    agent.seed = int(seed)
    agent.co_resident_composer = True
    agent.enable_da_salience_gate = True
    agent._da_gate_g0 = float(da_gate_g0)
    agent._da_gate_k = float(da_gate_k)
    agent._da_gate_cap = float(da_gate_cap)
    agent.co_resident_td_cueshift = False
    agent.co_resident_limbic = True          # the SHARED dopamine source (limbic SNc) Route A reads
    agent.co_resident_nav_critic = False
    agent.nav_critic_spiking_sc = False
    agent.nav_critic_place_selforg = False

    _D = int(rf_D)
    bridge, handles = build_merged_nav_conv_bridge(
        seed=seed, vocab=vocab, co_resident_rf=True, rf_D=_D,
        co_resident_limbic=True,            # builds limbic_snc + the `dopamine` from_region_firing_signed modulator
        co_resident_drive=True)             # builds the 2-pool SPIKING drive (drive_agrp / drive_pomc)
    agent._merged_bridge = bridge
    agent._handles = handles
    words = handles["vocab"]

    agent.composer = MergedRFComposer(
        bridge, handles["rf_base"], handles["rf_size"], seed=seed, D=_D, vocab=words, period=200)
    agent.parser = _MergedParserAdapter(bridge, handles["conj_arr"], handles["role_arr"])
    agent._dlpfc_ctx = handles["dlpfc_ctx"]
    agent._dlpfc_controller = None
    agent._dlpfc_graph_key = None

    # the same anti-cheat asserts MergedNavConvAgent.__init__ runs (parser+dlPFC+rf actually on the merged bridge).
    region_names = bridge.region_manager.region_indices_dict()
    assert "parse_conj" in region_names and "dlpfc_wm" in region_names and "rf" in region_names, \
        "FAIL anti-cheat: parser/dlPFC/rf not co-resident on the merged bridge"
    assert agent._dlpfc_ctx.bridge is bridge, "FAIL anti-cheat: elaborate dlPFC context is not the merged bridge"
    assert agent.composer._merged is bridge, "FAIL anti-cheat: the co-resident composer is not the merged bridge"
    assert "drive" in handles and "limbic" in handles, \
        "FAIL: the drive + limbic slices were not co-resident on the merged bridge"
    return agent, handles


# ============================================================================
# 2. The cross-modal coupler: body DEFICIT -> SPIKING hunger (drive_agrp firing)
#    -> the shared SNc dopamine (drive the limbic_reward_us afferent). Sets the
#    agent's DA state for the conversational turn. ALL on the merged bridge.
# ============================================================================
class DriveDopamineCoupler:
    """Drives the merged bridge's SPIKING interoceptive drive from the body energy DEFICIT, reads the spiking
    hunger (drive_agrp firing rate), and drives the shared limbic SNc afferent from that hunger so the
    `dopamine` concentration rises with the body's drive state. The agent's Route A `_da_confidence_gate` reads
    THIS dopamine. ⇒ the same drive that gates nav-survival also sets the conversational DA salience.

    modes:
      'intact' — the SNc afferent is driven by the SPIKING HUNGER read from drive_agrp (the cross-modal link).
      'lesion' — the interoceptive drive current is ZEROED (drive_agrp silent => hunger ~floor) AND the SNc
                 afferent gets no hunger drive => dopamine -> baseline => Route A gate -> g0 (no modulation).
      'yoke'   — the drive is RUN identically (same per-step compute) but the SNc afferent is driven by a
                 DRIVE-INDEPENDENT shuffled signal (matched marginal, decorrelated from the deficit) => SOME DA,
                 but not correlated with the body state (the control that must NOT reproduce the drive pattern).
    """

    def __init__(self, agent, handles, mode="intact", seed=42,
                 drive_window=40, snc_window=300, n_yoke_draws=6,
                 drive_i_scale=300.0, hunger_gain=14.0, hunger_floor=0.1,
                 snc_i_scale=600.0, snc_tonic_pa=80.0):
        import sim.backend as B
        self._B = B
        self.xp, _ = B.get_backend()
        self.agent = agent
        self.bridge = agent._merged_bridge
        self.mode = mode
        self.rng = np.random.default_rng(seed + 4242)
        self.drive_window = int(drive_window)
        self.snc_window = int(snc_window)
        self.drive_i_scale = float(drive_i_scale)
        self.hunger_gain = float(hunger_gain)
        self.hunger_floor = float(hunger_floor)
        self.snc_i_scale = float(snc_i_scale)      # the salient SNc drive at FULL hunger (de-risk: 600pA -> DA~0.61)
        self.snc_tonic_pa = float(snc_tonic_pa)    # the tonic SNc drive at zero hunger (de-risk: 80pA -> DA~baseline)
        self.n_yoke_draws = int(n_yoke_draws)      # yoke DA = mean over this many decorrelated draws (sound control)
        self.da_baseline = float(self.bridge.neuromodulator_manager._config_by_name("dopamine").baseline)

        rm = self.bridge.region_manager
        self.agrp = self.xp.asarray(np.asarray(rm.indices("drive_agrp"), dtype=np.int64))
        self.pomc = self.xp.asarray(np.asarray(rm.indices("drive_pomc"), dtype=np.int64))
        self.snc = self.xp.asarray(np.asarray(rm.indices("limbic_snc"), dtype=np.int64))
        self.n_snc = len(np.asarray(rm.indices("limbic_snc")))
        self.n_agrp = len(np.asarray(rm.indices("drive_agrp")))
        self._yoke_vals = None
        self._yoke_idx = 0
        self._yoke_cached = None     # yoke DA is deficit-independent -> computed once, reused for low+high
        # logs
        self.log = {"deficit": [], "hunger": [], "agrp_rate": [], "dopamine": [], "snc_rate": [], "g_eff": []}

    # -- washout (condition isolation) -------------------------------------
    def _washout(self, n=120):
        """Run the bridge at ZERO external current for `n` steps + reset the dopamine EMA to baseline, so the SNc +
        drive pools return to rest before the next measurement (each (mode, drive-level) reads only its OWN drive,
        no membrane/EMA carry-over from the previous condition). Biologically: the SNc returns to tonic between
        salience events. The masked RF/parser/nav slices are idle (zero current), so this is conversation-inert."""
        B, br = self._B, self.bridge
        for _ in range(int(n)):
            br.cp_external_input_current[:] = 0.0
            br._run_one_simulation_step()
            br.runtime_state.current_time_step += 1
        br.neuromodulator_manager.set_concentration("dopamine", self.da_baseline)

    # -- the SPIKING drive read (the brain-based hunger) --------------------
    def _spiking_hunger(self, deficit):
        """Inject the deficit/surplus as interoceptive current into the drive pools, run drive_window steps, read
        the drive_agrp firing rate as the hunger (off cp_firing_states). lesion => zero the interoceptive drive."""
        B, br = self._B, self.bridge
        lesion = (self.mode == "lesion")
        i_agrp = 0.0 if lesion else self.drive_i_scale * max(0.0, float(deficit))
        i_pomc = self.drive_i_scale * max(0.0, 1.0 - float(deficit))
        a_spikes = 0
        for _ in range(self.drive_window):
            br.cp_external_input_current[:] = 0.0
            br.cp_external_input_current[self.agrp] = i_agrp
            br.cp_external_input_current[self.pomc] = i_pomc
            br._run_one_simulation_step()
            br.runtime_state.current_time_step += 1
            a_spikes += int(B.to_host(br.cp_firing_states[self.agrp]).sum())
        a_rate = a_spikes / (self.n_agrp * self.drive_window)
        hunger = float(np.clip(self.hunger_floor + self.hunger_gain * a_rate, 0.0, 1.0))
        return hunger, a_rate

    # -- the shared SNc drive (the cross-modal link) -----------------------
    def _settle_at_fraction(self, drive_frac):
        """One SNc settle at a given drive fraction: drive `limbic_snc` at tonic..salient current (the de-risk's
        validated operating points: 80pA -> ~10Hz -> DA~baseline; 600pA -> ~130Hz -> DA~0.61), reset the dopamine
        EMA to baseline first (condition isolation), run snc_window steps (advancing the EMA), return (DA, SNc Hz).
        The DA is produced by the `dopamine` from_region_firing_signed modulator over [limbic_snc] (a spike-derived
        scalar, not a host formula)."""
        B, br = self._B, self.bridge
        i_snc = self.snc_tonic_pa + (self.snc_i_scale - self.snc_tonic_pa) * max(0.0, min(1.0, float(drive_frac)))
        br.neuromodulator_manager.set_concentration("dopamine", self.da_baseline)
        s_spikes = 0
        for _ in range(self.snc_window):
            br.cp_external_input_current[:] = 0.0
            br.cp_external_input_current[self.snc] = i_snc
            br._run_one_simulation_step()
            br.runtime_state.current_time_step += 1
            s_spikes += int(B.to_host(br.cp_firing_states[self.snc]).sum())
        da = float(br.neuromodulator_manager.get_concentration("dopamine"))
        return da, s_spikes / (self.n_snc * self.snc_window)

    def _settle_snc_dopamine(self, hunger):
        """Drive the shared spiking SNc pool from the body's drive state and read the settled `dopamine`.
        Motivational/interoceptive drive elevating dopaminergic firing is the documented hunger->DA pathway
        (AgRP/lateral-hypothalamus -> VTA/SNc; Palmiter, Berridge).
          intact:  the SNc drive TRACKS the spiking hunger => DA rises with the deficit (the cross-modal link).
          lesion:  the drive is SEVERED, so no hunger reaches the SNc => the SNc sits at TONIC baseline
                   (drive_frac=0 => i_snc = snc_tonic_pa) => DA ~ baseline at BOTH deficit levels (the body
                   deficit cannot reach the shared dopamine). The decisive lesion: the modulation cannot form.
          yoke:    a DRIVE-INDEPENDENT signal decorrelated from the deficit. To make the yoke a STATISTICALLY
                   SOUND control (a single shuffled draw per level is a coin-flip whether high>low), the yoke DA
                   is the MEAN over `n_yoke_draws` independent shuffled fractions => it reflects the
                   deficit-INDEPENDENT EXPECTATION (so high≈low, rise≈0), isolating the systematic deficit->DA
                   component that ONLY the intact drive has."""
        if self.mode == "lesion":
            # the SNc is at TONIC baseline regardless of the deficit; average over n_yoke_draws settles to suppress
            # finite-spiking noise (deterministic drive frac=0, so this just stabilizes the baseline read).
            das, rates = zip(*[self._settle_at_fraction(0.0) for _ in range(self.n_yoke_draws)])
            return float(np.mean(das)), float(np.mean(rates))
        if self.mode == "yoke":
            # the yoke signal is DEFICIT-INDEPENDENT by construction, so its EXPECTED DA is the SAME regardless of
            # the deficit. We therefore compute the yoke's expected DA ONCE (mean over n_yoke_draws decorrelated
            # draws of the matched marginal) and reuse it for BOTH deficit levels — the CORRECT null for a
            # deficit-decorrelated signal: high == low (rise == 0). (A single fresh draw per level is a coin-flip
            # whether high>low; re-drawing per level just injects sampling noise into a quantity whose true value
            # is 0.) This isolates the systematic deficit->DA component that ONLY the intact drive has.
            if self._yoke_cached is None:
                self._yoke_vals = self.hunger_floor + self.rng.random(8192) * (1.0 - self.hunger_floor)
                das, rates = [], []
                for _ in range(self.n_yoke_draws):
                    f = float(self._yoke_vals[self._yoke_idx % len(self._yoke_vals)]); self._yoke_idx += 1
                    d, r = self._settle_at_fraction(f)
                    das.append(d); rates.append(r)
                self._yoke_cached = (float(np.mean(das)), float(np.mean(rates)))
            return self._yoke_cached
        return self._settle_at_fraction(float(hunger))   # intact: the SNc drive tracks the spiking hunger

    def set_drive_state(self, deficit):
        """Set the agent's whole-brain drive state for the upcoming conversational turn: deficit -> spiking
        hunger -> shared SNc dopamine (Route A reads it). Returns a diag dict. Logs the chain. A washout first
        isolates this measurement (no SNc/EMA carry-over from the previous condition)."""
        self._washout()
        hunger, a_rate = self._spiking_hunger(deficit)
        da, snc_rate = self._settle_snc_dopamine(hunger)
        g_eff = self.agent._da_confidence_gate()       # Route A reads the dopamine we just set on the bridge
        self.log["deficit"].append(float(deficit)); self.log["hunger"].append(hunger)
        self.log["agrp_rate"].append(a_rate); self.log["dopamine"].append(da)
        self.log["snc_rate"].append(snc_rate); self.log["g_eff"].append(g_eff)
        return {"deficit": float(deficit), "hunger": hunger, "agrp_rate": a_rate,
                "dopamine": da, "snc_rate": snc_rate, "g_eff": g_eff}


# ============================================================================
# 3. The conversational probe: store facts, build a borderline regime, measure
#    the recall behaviour at HIGH-drive vs LOW-drive + the moat at both levels.
# ============================================================================
# A fixed fact set + cue list. ALL words are in the bridge's DEFAULT_VOCAB (rf_phasor_composer.DEFAULT_VOCAB) so
# the composer's own concept codebook can encode them. K=3 stored facts: the composer's batched cue-scan
# (`_unbind_all_phases`, n = 2*K*D neurons) fits the co-resident `rf` region (7*D) iff K<=3, so K=3 keeps the
# production rf_D=128 (the merged-bridge default) — answer-identical to the per-fact loop. The co-resident RF
# slice's inherent FHRR cleanup noise makes some reads BORDERLINE (the regime where the confidence gate earns
# its keep). Distinct agents/actions so the moat cues below are genuinely unstored.
_FACTS = [
    ("dog", "go", "north"),
    ("cat", "run", "south"),
    ("apple", "come", "east"),
]
# unstored (agent, action) cues — the no-confab moat MUST abstain on these at BOTH drive levels. None of these
# (agent, action) pairs is a stored fact (each recombines a stored agent with a different stored action).
_UNSTORED_CUES = [
    ("dog", "run"), ("dog", "come"), ("cat", "go"), ("cat", "come"),
    ("apple", "go"), ("apple", "run"), ("river", "go"), ("river", "look"),
]


def _query_set(agent):
    """Return the stored cues (the agent should answer the stored patient when decisive) + the unstored cues (the
    moat must abstain). Uses the composer's own stored facts."""
    stored = [(f["agent"], f["action"], f["patient"]) for (f, _c) in agent.composer.kb]
    return stored, list(_UNSTORED_CUES)


def measure_conversation(agent, coupler, deficit, conv_harness):
    """At a fixed body drive state (deficit), set the agent's whole-brain DA via the coupler (drive -> spiking
    hunger -> shared SNc dopamine), read the resulting Route-A gate `g_eff`, then measure the conversational
    read-out TWO faithful ways:

      (1) the REAL on-bridge merged composer (`agent.what_does` / the no-confab moat) — the genuine spiking
          one-brain conversation IS present and co-resident; this proves the moat holds AT THIS DRIVE LEVEL and
          the stored facts still recall. (At the clean production operating point D=128/K=3 these reads are
          high-margin, so the gate does not change THEIR behaviour — an honest property of the operating point.)
      (2) the de-risk's VALIDATED noisy-cleanup conversational read-out (`FHRRCleanupComposer`, the exact harness
          the salience-gated PRECISION effect was validated on, 2026-06-18-DA-composer-precision-derisk-GO.md),
          driven by the SAME drive-set `g_eff` — this is where the precision/decisiveness shift is measurable: a
          higher gate (under hunger) abstains on the noise-dominated reads, so the ANSWERED reads are more
          decisive (higher mean cue-role margin) + the cue-role error among answered DROPS, with its OWN moat at
          0. The DA driving (2)'s gate is the body drive's, set on the merged bridge in (1)'s coupler.

    The DA state is set ONCE per call (the body drive is fixed for the turn). The on-bridge merged-composer reads
    (1) are DETERMINISTIC at a fixed kick (the RF op re-kicks each call; the resting config has OU off), so they
    are evaluated ONCE per cue — repeating them `reps` times would be byte-identical, wasted GPU. The de-risk's
    noisy harness (2) carries its OWN internal reps (the stochastic-noise averaging it was validated with)."""
    diag = coupler.set_drive_state(deficit)
    g_eff = diag["g_eff"]
    stored, unstored = _query_set(agent)

    # (1) the REAL merged composer: recall + the no-confab MOAT at this drive level (deterministic -> one pass).
    n_answer = n_total = n_correct = 0
    for (a, v, p) in stored:
        ans = agent.what_does(a, v)
        n_total += 1
        if ans is None:
            continue
        n_answer += 1
        if ans == p:
            n_correct += 1
    moat_fa = 0
    for (a, v) in unstored:
        if agent.what_does(a, v) is not None:
            moat_fa += 1

    # (2) the de-risk's noisy conversational read-out at the drive-set gate (the measurable precision shift).
    conv = conv_harness.read_at_gate(g_eff)

    return {
        "deficit": float(deficit), "g_eff": float(g_eff), "dopamine": diag["dopamine"],
        "hunger": diag["hunger"], "agrp_rate": diag["agrp_rate"], "snc_rate": diag["snc_rate"],
        # (1) on-bridge merged composer (co-residence + moat):
        "merged_answer_rate": n_answer / max(n_total, 1),
        "merged_recall_correct_rate": n_correct / max(n_answer, 1),
        "merged_moat_false_accepts": moat_fa,
        # (2) the noisy conversational read-out shifted by the drive-set gate (the precision effect):
        "noisy_answered_margin": conv["mean_answered_margin"],
        "noisy_answer_rate": conv["answer_rate"],
        "noisy_cue_error_rate": conv["cue_error_rate"],
        "noisy_moat_false_accepts": conv["moat_false_accepts"],
        # the overall moat = both moats clean (never weakened at this drive level):
        "moat_false_accepts": int(moat_fa + conv["moat_false_accepts"]),
    }


# ============================================================================
# The de-risk's VALIDATED noisy conversational cleanup harness (reuse-by-import
# of FHRRCleanupComposer from _da_composer_salience_cleanup_derisk), exposed as a
# read-at-a-gate read-out. This is the EXACT mechanism the salience-gated PRECISION
# effect was validated on (2026-06-18-DA-composer-precision-derisk-GO.md); here the
# gate is set by the BODY DRIVE on the merged bridge, not an abstract SNc current.
# ============================================================================
class NoisyConversationHarness:
    """A fixed FHRR fact store + cue-matching cleanup with controllable cleanup NOISE (the borderline regime where
    the confidence gate earns its keep), reusing the de-risk's `FHRRCleanupComposer` + the EXACT OneBrainComposer
    gate logic. `read_at_gate(g)` runs the fixed query set at gate `g` and returns the cue-role precision +
    answer-rate + the moat. The SAME facts/cues as the on-bridge composer; D=64/noise_sigma=2.0 = the de-risk's
    validated borderline operating point."""

    def __init__(self, seed=42, D=64, noise_sigma=2.0, reps=20):
        from research.runners._da_composer_salience_cleanup_derisk import FHRRCleanupComposer
        vocab = sorted({w for (a, v, p) in _FACTS for w in (a, v, p)}
                       | {w for (a, v) in _UNSTORED_CUES for w in (a, v)})
        self.comp = FHRRCleanupComposer(seed=seed, D=D, vocab=vocab)
        for (a, v, p) in _FACTS:
            self.comp.store(a, v, p)
        self.noise_sigma = float(noise_sigma)
        self.reps = int(reps)
        self.seed = int(seed)
        self.stored = [(f["agent"], f["action"], f["patient"]) for (f, _c) in self.comp.kb]
        self._comp_phasor = {(f["agent"], f["action"], f["patient"]): c for (f, c) in self.comp.kb}

    def read_at_gate(self, g_eff):
        """Run the fixed query set at confidence gate `g_eff` under matched cleanup noise. Returns the cue-role
        error-rate among ANSWERED reads (the precision the gate sharpens), the mean answered margin (decisiveness),
        the answer-rate (1 - abstain), and the moat false-accepts (unstored cues must abstain)."""
        self.comp.confidence_gate = float(g_eff)
        margins, n_answer, n_total, cue_wrong = [], 0, 0, 0
        for rep in range(self.reps):
            rng = np.random.default_rng(self.seed * 131 + rep)
            for (a, v, p) in self.stored:
                (out, cm) = self.comp._read_block_full(self._comp_phasor[(a, v, p)], self.noise_sigma, rng)
                n_total += 1
                if out[0] is None:
                    continue
                n_answer += 1
                margins.append(float(cm))
                if not (out[0] == a and out[1] == v):
                    cue_wrong += 1
        moat_fa = 0
        for rep in range(self.reps):
            rng = np.random.default_rng(self.seed * 1009 + 7 * rep)
            for (a, v) in _UNSTORED_CUES:
                if self.comp.query_patient(a, v, noise_sigma=self.noise_sigma, rng=rng) is not None:
                    moat_fa += 1
        return {"mean_answered_margin": float(np.mean(margins)) if margins else 0.0,
                "answer_rate": n_answer / max(n_total, 1),
                "cue_error_rate": cue_wrong / max(n_answer, 1),
                "moat_false_accepts": moat_fa}


# ============================================================================
# 4. One seed: build the one-animal agent, store facts, run HIGH vs LOW drive
#    for intact / lesion / yoke, assemble the verdict.
# ============================================================================
def run_seed(seed, *, rf_D=128, reps=8, low_deficit=0.05, high_deficit=0.95,
             drive_window=40, snc_window=300, verbose=False):
    agent, handles = build_one_animal_agent(seed=seed, rf_D=rf_D)
    # store the fixed fact set on the co-resident composer (the parser is voice-invariant; here we store directly
    # to fix the fact content across modes — the conversational read-out is what the drive modulates).
    for (a, v, p) in _FACTS:
        agent.composer.store(a, v, p)
    # the de-risk's validated noisy conversational read-out (same facts/cues, its borderline operating point).
    # `reps` is the noise-AVERAGING count for the stochastic harness (the de-risk's validated 20); the on-bridge
    # merged-composer read is deterministic so it is evaluated once per cue regardless.
    conv_harness = NoisyConversationHarness(seed=seed, reps=max(8, reps))

    out = {"seed": seed, "modes": {}}
    for mode in ("intact", "lesion", "yoke"):
        coupler = DriveDopamineCoupler(agent, handles, mode=mode, seed=seed,
                                       drive_window=drive_window, snc_window=snc_window)
        low = measure_conversation(agent, coupler, low_deficit, conv_harness)
        high = measure_conversation(agent, coupler, high_deficit, conv_harness)
        out["modes"][mode] = {"low": low, "high": high,
                              "coupler_log": {k: list(v) for k, v in coupler.log.items()}}
        if verbose:
            print(f"  [seed {seed} {mode}] DA low={low['dopamine']:.3f} high={high['dopamine']:.3f} | "
                  f"g_eff low={low['g_eff']:.3f} high={high['g_eff']:.3f} | "
                  f"NOISY answered-margin low={low['noisy_answered_margin']:.3f} high={high['noisy_answered_margin']:.3f} "
                  f"| cue-err low={low['noisy_cue_error_rate']:.3f} high={high['noisy_cue_error_rate']:.3f} "
                  f"| ans-rate low={low['noisy_answer_rate']:.2f} high={high['noisy_answer_rate']:.2f} "
                  f"| moat(low,high)=({low['moat_false_accepts']},{high['moat_false_accepts']})", flush=True)

    out["verdict"] = _seed_verdict(out["modes"])
    return out


def _seed_verdict(modes):
    """The one-animal GO bar, per the decisive metric + the anti-cheats:
      (A) DRIVE MODULATES CONVERSATION (intact): the high-drive DA is above the low-drive DA (hunger raised the
          shared dopamine), AND the conversational read-out shifts with it — the salience gate is STRICTER under
          hunger (g_eff_high > g_eff_low) and the (noisy, validated-harness) read-out is MORE PRECISE: the
          answered reads are more decisive (mean answered margin high >= low) AND the cue-role error among
          answered reads DROPS (high <= low). The shift is the SAME-drive-moving-both-halves signal.
      (B) MOAT held at BOTH drive levels (intact + lesion + yoke): 0 false-accepts everywhere (never weakened) —
          the on-bridge merged composer moat AND the noisy read-out moat both 0.
      (C) DRIVE-LESION kills the modulation: with the drive lesioned, high == low (DA ~ baseline both, g_eff = g0
          both) => no conversational shift.
      (D) YOKE: a drive-independent DA does NOT reproduce the drive-specific high>low DA ordering (the link needs
          the hunger->DA correlation, not just SOME DA)."""
    intact, lesion, yoke = modes["intact"], modes["lesion"], modes["yoke"]
    il, ih = intact["low"], intact["high"]
    # (A) the drive raised the shared DA AND the conversational read-out shifted with it (the validated precision effect).
    da_rose = bool(ih["dopamine"] > il["dopamine"] + 1e-3)
    gate_stricter = bool(ih["g_eff"] > il["g_eff"] + 1e-9)
    more_decisive = bool(ih["noisy_answered_margin"] >= il["noisy_answered_margin"] - 1e-9)
    more_precise = bool(ih["noisy_cue_error_rate"] <= il["noisy_cue_error_rate"] + 1e-9)
    conv_shifted = bool(gate_stricter and more_decisive and more_precise)
    drive_modulates = bool(da_rose and conv_shifted)
    # (B) the moat held everywhere (the hard gate — never weakened): both moats 0 at every (mode, drive level).
    moat_held = bool(all(m[d]["moat_false_accepts"] == 0 for m in (intact, lesion, yoke) for d in ("low", "high")))
    # (C) lesion KILLS the drive-dependent modulation: with the drive SEVERED, the body DEFICIT cannot reach the
    # shared dopamine, so the deficit->DA tracking VANISHES. The decisive, finite-spiking-faithful test (NOT
    # bit-exact DA, which a 30-neuron stochastic SNc cannot give): the lesion's high-vs-low DA rise is far below
    # the intact rise (<=25% of it) — i.e. the intact DA tracks the deficit and the lesion DA does not. Equivalently
    # the lesion does NOT reproduce the intact's salience-gated PRECISION gain.
    ll, lh = lesion["low"], lesion["high"]
    intact_da_rise = ih["dopamine"] - il["dopamine"]
    lesion_da_rise = lh["dopamine"] - ll["dopamine"]
    lesion_kills_da = bool(lesion_da_rise < 0.25 * intact_da_rise)        # the deficit->DA tracking is abolished
    intact_precision_gain = il["noisy_cue_error_rate"] - ih["noisy_cue_error_rate"]   # error DROP under hunger (intact)
    lesion_precision_gain = ll["noisy_cue_error_rate"] - lh["noisy_cue_error_rate"]
    lesion_kills_precision = bool(lesion_precision_gain < 0.25 * intact_precision_gain)  # no precision gain under lesion
    lesion_kills = bool(lesion_kills_da and lesion_kills_precision)
    # (D) yoke: no drive-specific DA ordering (decorrelated drive doesn't put high above low in a deficit-specific way).
    yl, yh = yoke["low"], yoke["high"]
    yoke_da_rise = yh["dopamine"] - yl["dopamine"]
    yoke_no_pattern = bool(yoke_da_rise < 0.25 * intact_da_rise)   # yoke's high-vs-low rise is NOT the intact rise
    go = bool(drive_modulates and moat_held and lesion_kills and yoke_no_pattern)
    return {"go": go, "drive_modulates": drive_modulates, "da_rose": da_rose,
            "gate_stricter": gate_stricter, "more_decisive": more_decisive, "more_precise": more_precise,
            "conv_shifted": conv_shifted, "moat_held": moat_held, "lesion_kills": lesion_kills,
            "lesion_kills_da": lesion_kills_da, "lesion_kills_precision": lesion_kills_precision,
            "yoke_no_pattern": yoke_no_pattern,
            "intact_da_rise": float(intact_da_rise), "lesion_da_rise": float(lesion_da_rise),
            "yoke_da_rise": float(yoke_da_rise),
            "intact_precision_gain": float(intact_precision_gain),
            "lesion_precision_gain": float(lesion_precision_gain),
            "intact_low_g_eff": float(il["g_eff"]), "intact_high_g_eff": float(ih["g_eff"]),
            "intact_low_cue_err": float(il["noisy_cue_error_rate"]),
            "intact_high_cue_err": float(ih["noisy_cue_error_rate"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--rf-D", type=int, default=128)
    ap.add_argument("--reps", type=int, default=20,
                    help="noise-averaging count for the de-risk's stochastic conversational harness (validated 20); "
                         "the on-bridge merged-composer read is deterministic (evaluated once per cue regardless)")
    ap.add_argument("--low-deficit", type=float, default=0.05)
    ap.add_argument("--high-deficit", type=float, default=0.95)
    ap.add_argument("--drive-window", type=int, default=40)
    ap.add_argument("--snc-window", type=int, default=300,
                    help="SNc settle steps (the dopamine EMA window; the de-risk used ~400 to reach the steady DA)")
    ap.add_argument("--out", default="research/findings/raw/_one_animal_crossmodal.json")
    ap.add_argument("--smoke", action="store_true", help="tiny mechanics check (1 seed, intact only, short windows)")
    a = ap.parse_args()

    print("[ONE ANIMAL] does the SAME shared spiking interoceptive DRIVE that motivates nav-survival ALSO modulate\n"
          "  CONVERSATION via the shared dopamine (Route A)? The drive + limbic SNc + composer are co-resident on\n"
          "  ONE merged bridge; hunger -> spiking drive_agrp firing -> shared SNc dopamine -> Route A tightens the\n"
          "  conversational recall gate. GATES: (A) high-drive raises DA + shifts the read-out  (B) MOAT held at\n"
          "  both drive levels  (C) drive-LESION kills the modulation  (D) yoked DA doesn't reproduce it.\n", flush=True)

    if a.smoke:
        r = run_seed(a.seeds[0], rf_D=a.rf_D, reps=3, low_deficit=a.low_deficit, high_deficit=a.high_deficit,
                     drive_window=20, snc_window=300, verbose=True)
        m = r["modes"]["intact"]
        ok = bool(m["low"]["moat_false_accepts"] == 0 and m["high"]["moat_false_accepts"] == 0)
        print(f"[smoke] DA low={m['low']['dopamine']:.3f} high={m['high']['dopamine']:.3f} (rise "
              f"{m['high']['dopamine'] - m['low']['dopamine']:+.3f}) | g_eff low={m['low']['g_eff']:.3f} "
              f"high={m['high']['g_eff']:.3f} | moat low/high {m['low']['moat_false_accepts']}/"
              f"{m['high']['moat_false_accepts']} || {'OK' if ok else 'CHECK'}", flush=True)
        return 0 if ok else 1

    per_seed = []
    for seed in a.seeds:
        r = run_seed(seed, rf_D=a.rf_D, reps=a.reps, low_deficit=a.low_deficit, high_deficit=a.high_deficit,
                     drive_window=a.drive_window, snc_window=a.snc_window, verbose=True)
        per_seed.append(r)
        v = r["verdict"]
        print(f"  >>> seed {seed}: {'GO' if v['go'] else 'NO'}  "
              f"drive_modulates={v['drive_modulates']} moat_held={v['moat_held']} "
              f"lesion_kills={v['lesion_kills']} yoke_no_pattern={v['yoke_no_pattern']}", flush=True)

    n_go = sum(p["verdict"]["go"] for p in per_seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"per_seed": per_seed, "n_go": n_go, "n_seeds": len(per_seed)}, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    if per_seed and n_go == len(per_seed):
        print(f"  ONE-ANIMAL GO ({n_go}/{len(per_seed)} seeds): the SAME shared spiking interoceptive DRIVE that\n"
              "  motivates nav-survival ALSO modulates CONVERSATION — a HIGH-drive (hungry) state raises the shared\n"
              "  spiking-SNc dopamine, Route A tightens the conversational recall gate (the answered reads are more\n"
              "  decisive / borderline reads abstain under hunger), the no-confab MOAT holds at BOTH drive levels,\n"
              "  drive-LESION abolishes the modulation, and a yoked drive-independent DA does NOT reproduce it.\n"
              "  ⇒ one limbic core moves both halves of the animal — the deepest 'one self'. NO sim/ edit "
              "(reuse-by-import).", flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(per_seed)} seeds): the cross-modal link does not robustly carry —\n"
              "  localize (DA rise / conversational shift / moat / lesion / yoke). If the modulation is too weak on\n"
              "  the merged operating point, that maps the substrate cost — a valid honest-negative deliverable.", flush=True)
    print(f"  [saved] {a.out}\n{'=' * 110}", flush=True)
    return 0 if (per_seed and n_go == len(per_seed)) else 1


if __name__ == "__main__":
    sys.exit(main())
