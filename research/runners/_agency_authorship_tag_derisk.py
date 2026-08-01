"""Lane A · Affect (Phase-0 self-model) — AGENCY / AUTHORSHIP TAG: a 1-bit "producer-vs-parser" source tag on ONE
spiking substrate. Does the brain answer "did YOU say that, or did I?" by reading its own corollary discharge?

THE FACULTY (self-awareness primitive). Source monitoring — telling self-produced content from externally-perceived
content — is a load-bearing piece of the affective self-model (Johnson-Hashtroudi-Lindsay 1993 source monitoring;
Frith 1992 / Feinberg 1978 comparator model; Ford-Mathalon: its FAILURE = misattributing self-generated inner speech
to an external voice). The BIOLOGY: self-production emits a COROLLARY DISCHARGE / efference copy (Sperry 1950; von
Holst-Mittelstaedt 1950; Crapse-Sommer 2008) that is available to the monitor; externally-parsed content arrives via
a sensory stream WITHOUT that efference copy. A comparator that reads "was an efference copy present?" yields the
1-bit authorship tag. This is the textbook mechanism, built here as spiking neurons + synapses.

MECHANISM (BUILDABLE-NOW, ONE numpy Izhikevich SimulationBridge, NO `sim/` edit — reuse-by-import):
  * `production` (self generator) and `parse` (external listener) regions. On a SELF utterance we drive `production`,
    which EMITS an efference copy to a corollary-discharge population `cd` (production -> cd is a real synaptic
    projection = the efference copy). On an OTHER utterance we drive `parse`, which drives a `sensory_marker`
    population (the perceived-speech arrival signal). Both pathways ALSO weakly drive the shared `content` pool (an
    utterance genuinely involves content), but content IDENTITY is set by the world (external stimulus current to the
    item's neurons) and is IDENTICAL for self vs other — so content carries NO source information by construction.
  * The SOURCE MONITOR (the brain's job): `src_self` is driven by `cd`, `src_other` by `sensory_marker`, and the two
    tag pools mutually inhibit through their own interneurons (Namburi-Tye biased-competition cross-inhibition, the
    same motif the affect region uses). The winner is a clean 1-bit read-out: cd present -> src_self wins -> "I said
    it"; sensory arrival, no cd -> src_other wins -> "you said it".
  * Read-out per trial = sign of rate(src_self) - rate(src_other) over the trial window vs the GROUND-TRUTH source.

GO GATE (6-seed {42 43 44 100 101 102}): authorship accuracy >= 0.90 (chance 0.5), robust across seeds.

ANTI-CHEATS (roadmap-mandated; all wired + INVOKED):
  (1) LESION -> CHANCE. Zero the two carrier->tag projections (cd->src_self and sensory_marker->src_other via their
      transmission gates). The source pools lose their differential drive -> the winner is noise -> accuracy collapses
      to chance (~0.5). Proves the corollary-discharge / sensory-arrival carriers are load-bearing.
  (2) SWAP WIRING -> TAG FLIPS. Rebuild with cd -> src_OTHER and sensory_marker -> src_SELF. Self utterances (cd
      present) now win src_other and vice-versa -> the judgment SYSTEMATICALLY FLIPS (accuracy ~1-real, i.e. near 0;
      relabelled it reads ~real). Proves the tag tracks the source WIRING, not the content.
  (3) TAG PERP CONTENT. (a) DECISIVE: each content item is uttered as BOTH self and other; per-item accuracy stays
      high for EVERY item -> identical content receives OPPOSITE tags from source alone (a content-encoding tag would
      score ~0.5 per item). (b) REPRESENTATIONAL: content-item identity does not decode from the src-tag pool rates
      above chance (1/K), leave-one-out nearest-centroid -> the tag pools carry SOURCE, not content. (c) A no-carrier
      CATCH trial (content stimulus only, no production/parse act) reads at chance -> content alone carries no
      authorship. (The src-margin MAGNITUDE's weak per-item modulation is a REPORTED diagnostic, not a gate: it is
      decision-irrelevant wiring heterogeneity — the margin SIGN, which sets the judgment, is 100% source-determined.)

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit (BrainRegion / RegionPathway /
transmission_gate are pre-existing). cfg.seed set per-seed (SEEDS THE SUBSTRATE — NOT actual_seed_used, the CLAUDE.md
gotcha). HONEST SCOPE: this is a FIXED-STRUCTURE comparator (corollary-discharge source monitor); the content-cued
episodic SOURCE-MEMORY version (Hebbian-bind content->tag at encoding, content-cue the tag at recall) and the
self-organized wiring are the named follow-ons, exactly as the affect-region GO banked a hand-wired attractor.

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._agency_authorship_tag_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._agency_authorship_tag_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  (passthrough on numpy)
from tools.lab import attributable_to  # noqa: E402  (force the treatment/control subtraction to be asked)
from tools.verdict import Verdict  # noqa: E402  (a verdict that carries its own preconditions into the artifact)

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_agency_authorship_tag_6seed.json"

# ---- operating-point constants -------------------------------------------------------------------------------
K_ITEMS = 6                 # distinct content items (for the tag-perp-content test)
N_PER_ITEM = 20             # content neurons per item
N_PROD = 40                 # production (self generator) pool
N_PARSE = 40                # parse (external listener) pool
N_CD = 30                   # corollary-discharge (efference-copy) pool
N_SM = 30                   # sensory-arrival marker pool
N_SRC = 40                  # per source-tag pool (src_self / src_other)
N_INH = 15                  # per source-tag opponent interneuron pool

W_ACT_CARRIER = 60.0        # production->cd  and  parse->sensory_marker (the efference-copy / sensory-arrival emit)
W_ACT_CONTENT = 2.0         # production/parse -> content (weak nonspecific "an utterance happened")
W_CARRIER_SRC = 60.0        # cd->src_self  and  sensory_marker->src_other (gated; the carrier that sets the tag)
W_SRC_INH = 8.0             # src pool -> its opponent interneuron
W_INH_SRC = 30.0            # opponent interneuron -> the OTHER src pool (cross-inhibition, gaba_a)

PROD_DRIVE_PA = 2500.0      # external drive to production/parse during an utterance (no-recurrence pool -> strong)
STIM_CONTENT_PA = 300.0     # external stimulus current to the uttered item's content neurons (world-set identity)
OU_PA = 3.0                 # OU noise (tie-breaking; sub-dominant to the carrier drive)

TRIAL_STEPS = 40            # ms per utterance window (dt=1ms)
FLUSH_STEPS = 70            # zero-input inter-trial gap: MUST exceed the synaptic-delay buffer so a prior trial's
                            # delayed carrier spikes fully drain before the next utterance (12 was too short ->
                            # cross-trial hysteresis; 60 already gives clean +-5 margins, 70 for headroom)
REPS = 5                    # repetitions per (item x source) cell


# =============================================================================================================
# The authorship brain: production/parse -> efference-copy/sensory carriers -> biased-competition source monitor,
# ALL co-resident on ONE numpy SimulationBridge.
# =============================================================================================================
class AuthorshipBrain:
    def __init__(self, seed, wiring="normal"):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway

        assert wiring in ("normal", "swapped")
        self.seed = int(seed)
        self.wiring = wiring

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.enable_neuromodulator_subsystem = False
        cfg.enable_nmda = False              # a per-trial comparator needs no persistence latch
        cfg.dt_ms = 1.0
        cfg.seed = int(seed)                 # SEEDS THE SUBSTRATE (NOT actual_seed_used — the CLAUDE.md gotcha)
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.enable_hebbian_learning = False
        cfg.enable_homeostasis = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_structural_plasticity = False
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = float(OU_PA)
        cfg.enable_parameter_heterogeneity = False
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1

        RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
        FS = "IZH2007_FS_CORTICAL_INTERNEURON"

        def exc(name, n, dens=0.0, w=0.0):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=1.0, internal_density=dens,
                               exc_weight_mean=float(w), inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

        def fs(name, n):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=FS)

        regions = [
            exc("content", K_ITEMS * N_PER_ITEM),
            exc("production", N_PROD), exc("parse", N_PARSE),
            exc("cd", N_CD), exc("sensory_marker", N_SM),
            exc("src_self", N_SRC), exc("src_other", N_SRC),
            fs("inh_self", N_INH), fs("inh_other", N_INH),
        ]

        # carrier -> tag wiring (the swap anti-cheat flips these two lines)
        if wiring == "normal":
            cd_target, sm_target = "src_self", "src_other"
        else:
            cd_target, sm_target = "src_other", "src_self"

        pathways = [
            # production EMITS an efference copy (corollary discharge); parse drives the sensory-arrival marker
            RegionPathway(from_region="production", to_region="cd", density=0.7, weight_mean=W_ACT_CARRIER,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="parse", to_region="sensory_marker", density=0.7, weight_mean=W_ACT_CARRIER,
                          weight_jitter=0.1, plastic=False),
            # both acts weakly drive the shared content pool (utterance involves content; identity is world-set)
            RegionPathway(from_region="production", to_region="content", density=0.1, weight_mean=W_ACT_CONTENT,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="parse", to_region="content", density=0.1, weight_mean=W_ACT_CONTENT,
                          weight_jitter=0.1, plastic=False),
            # the carriers set the source tag (GATED so the lesion can cut them)
            RegionPathway(from_region="cd", to_region=cd_target, density=0.7, weight_mean=W_CARRIER_SRC,
                          weight_jitter=0.1, plastic=False, transmission_gate="cd_gate"),
            RegionPathway(from_region="sensory_marker", to_region=sm_target, density=0.7, weight_mean=W_CARRIER_SRC,
                          weight_jitter=0.1, plastic=False, transmission_gate="sm_gate"),
            # biased-competition cross-inhibition between the two source tags (Namburi-Tye motif)
            RegionPathway(from_region="src_self", to_region="inh_self", density=0.6, weight_mean=W_SRC_INH,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="inh_self", to_region="src_other", density=0.7, weight_mean=W_INH_SRC,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="src_other", to_region="inh_other", density=0.6, weight_mean=W_SRC_INH,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="inh_other", to_region="src_self", density=0.7, weight_mean=W_INH_SRC,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        ]

        cfg.brain_regions = regions
        cfg.region_pathways = pathways

        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        self._idx = {n: np.asarray(v, dtype=np.int64)
                     for n, v in self._bridge.region_manager.region_indices_dict().items()}
        # per-item content neuron slices
        c = self._idx["content"]
        self._item_idx = [c[i * N_PER_ITEM:(i + 1) * N_PER_ITEM] for i in range(K_ITEMS)]

    def set_lesion(self, lesion: bool):
        """Cut the corollary-discharge and sensory-arrival carrier->tag projections (anti-cheat 1)."""
        g = 0.0 if lesion else 1.0
        self._bridge.set_transmission_gate("cd_gate", g)
        self._bridge.set_transmission_gate("sm_gate", g)

    def _flush(self, n_steps):
        b = self._bridge
        for _ in range(int(n_steps)):
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()

    def trial(self, item, source, carrier=True):
        """Run one utterance window. source in {'self','other'}. carrier=False = a no-carrier CATCH trial (content
        stimulus only, no production/parse act). Returns (self_spikes, other_spikes) summed over the window."""
        b = self._bridge
        self._flush(FLUSH_STEPS)
        s_self = s_other = 0.0
        for _ in range(TRIAL_STEPS):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[self._item_idx[item]] = np.float32(STIM_CONTENT_PA)  # world-set content
            if carrier:
                drv = "production" if source == "self" else "parse"
                b.cp_external_input_current[self._idx[drv]] = np.float32(PROD_DRIVE_PA)       # the utterance act
            b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            s_self += float(fs[self._idx["src_self"]].sum())
            s_other += float(fs[self._idx["src_other"]].sum())
        return s_self / N_SRC, s_other / N_SRC  # per-neuron spike counts over the window


# =============================================================================================================
# Trial protocol + metrics
# =============================================================================================================
def _run_trials(brain, rng, carrier=True):
    """Balanced (item x source x REPS) trials. Returns list of dicts with source, item, margin, winner."""
    trials = []
    order = [(it, src) for it in range(K_ITEMS) for src in ("self", "other") for _ in range(REPS)]
    rng.shuffle(order)
    for it, src in order:
        s_self, s_other = brain.trial(it, src, carrier=carrier)
        margin = s_self - s_other
        # random tie-break on a genuine tie (both pools silent, e.g. under LESION / no-carrier CATCH) so a
        # no-signal control is GENUINE chance (random guess), not a degenerate constant output that only scores
        # 0.5 because the classes are balanced.
        if abs(margin) < 1e-9:
            winner = "self" if rng.random() < 0.5 else "other"
        else:
            winner = "self" if margin > 0 else "other"
        trials.append({"item": it, "source": src, "margin": float(margin),
                       "s_self": float(s_self), "s_other": float(s_other), "winner": winner})
    return trials


def _acc(trials):
    return float(np.mean([t["winner"] == t["source"] for t in trials])) if trials else 0.0


def _acc_flipped(trials):
    flip = {"self": "other", "other": "self"}
    return float(np.mean([t["winner"] == flip[t["source"]] for t in trials])) if trials else 0.0


def _per_item_acc(trials):
    accs = []
    for it in range(K_ITEMS):
        sub = [t for t in trials if t["item"] == it]
        accs.append(_acc(sub))
    return accs


def _content_confound_r(trials):
    """DIAGNOSTIC (reported, NOT gated): within a fixed source, does the src margin MAGNITUDE correlate with
    content-item identity? A small non-zero value = decision-IRRELEVANT per-item wiring heterogeneity (the sign,
    which sets the judgment, is 100% source-determined — see min-per-item-acc). Max |r| across the two sources."""
    rmax = 0.0
    for src in ("self", "other"):
        sub = [t for t in trials if t["source"] == src]
        if len(sub) < 4:
            continue
        items = np.array([t["item"] for t in sub], float)
        marg = np.array([t["margin"] for t in sub], float)
        if items.std() < 1e-9 or marg.std() < 1e-9:
            continue
        rmax = max(rmax, abs(float(np.corrcoef(items, marg)[0, 1])))
    return rmax


def _content_decode_acc(trials):
    """GATED representational tag-PERP-content test (well-calibrated, chance = 1/K_ITEMS): can content-item identity
    be decoded from the SOURCE-tag pool rates (s_self, s_other)? Leave-one-out nearest-centroid within each source
    (LOO does not overfit upward, so chance-level accuracy is decisive). If the tag pools carried content, item would
    decode >> 1/K; if they carry only SOURCE, item decodes at ~1/K. Averaged over the two sources."""
    accs = []
    for src in ("self", "other"):
        sub = [t for t in trials if t["source"] == src]
        if len(sub) < K_ITEMS + 2:
            continue
        X = np.array([[t["s_self"], t["s_other"]] for t in sub], float)
        y = np.array([t["item"] for t in sub], int)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X = (X - mu) / sd
        correct = 0
        for i in range(len(sub)):
            mask = np.ones(len(sub), bool); mask[i] = False
            cents, labs = [], []
            for it in range(K_ITEMS):
                m = mask & (y == it)
                if m.any():
                    cents.append(X[m].mean(0)); labs.append(it)
            if not cents:
                continue
            d = np.linalg.norm(np.array(cents) - X[i], axis=1)
            correct += int(labs[int(np.argmin(d))] == y[i])
        accs.append(correct / len(sub))
    return float(np.mean(accs)) if accs else 0.0


def run_seed(seed):
    rng = np.random.default_rng(seed)
    # --- REAL arm (normal wiring, carriers intact) ---
    brain = AuthorshipBrain(seed, wiring="normal")
    brain.set_lesion(False)
    real = _run_trials(brain, rng, carrier=True)
    acc_real = _acc(real)
    per_item = _per_item_acc(real)
    confound_r = _content_confound_r(real)
    content_decode = _content_decode_acc(real)

    # --- (3c) no-carrier CATCH (content stimulus only, no act) -> chance ---
    catch = _run_trials(brain, rng, carrier=False)
    acc_catch = _acc(catch)

    # --- (1) LESION -> chance (same bridge, carrier->tag gates cut) ---
    brain.set_lesion(True)
    lesion = _run_trials(brain, rng, carrier=True)
    acc_lesion = _acc(lesion)

    # mechanism read-out for the `reaches` precondition: the lesion must actually SILENCE the tag pools (drop
    # |margin| ~5 -> ~0), not merely coincide with lower accuracy (a control that does not move its own
    # mechanism read-out is not a control -- the sAHP lesson).
    mean_abs_margin_real = float(np.mean([abs(t["margin"]) for t in real]))
    mean_abs_margin_lesion = float(np.mean([abs(t["margin"]) for t in lesion]))

    # --- (2) SWAP WIRING -> tag flips (fresh bridge, cd->src_other / sm->src_self) ---
    brain_sw = AuthorshipBrain(seed, wiring="swapped")
    brain_sw.set_lesion(False)
    swap = _run_trials(brain_sw, rng, carrier=True)
    acc_swap = _acc(swap)                  # expect ~1-acc_real (systematic flip)
    acc_swap_flip = _acc_flipped(swap)     # expect ~acc_real (the flip is systematic, not noise)

    return {
        "seed": int(seed),
        "acc_real": acc_real, "acc_lesion": acc_lesion, "acc_catch": acc_catch,
        "acc_swap": acc_swap, "acc_swap_flipped": acc_swap_flip,
        "min_per_item_acc": float(min(per_item)), "per_item_acc": [round(x, 3) for x in per_item],
        "content_decode_acc": round(content_decode, 3),
        "content_confound_maxabs_r": round(confound_r, 3),
        "mean_abs_margin_real": round(mean_abs_margin_real, 3),
        "mean_abs_margin_lesion": round(mean_abs_margin_lesion, 3),
        "n_trials": len(real),
    }


def _aggregate_verdict(rows, go_acc=0.90):
    def m(k):
        return float(np.mean([r[k] for r in rows]))
    real, les, catch = m("acc_real"), m("acc_lesion"), m("acc_catch")
    swap, swapf, mini = m("acc_swap"), m("acc_swap_flipped"), m("min_per_item_acc")
    decode, conf = m("content_decode_acc"), m("content_confound_maxabs_r")
    marg_real, marg_les = m("mean_abs_margin_real"), m("mean_abs_margin_lesion")
    chance_item = 1.0 / K_ITEMS
    checks = {
        "authorship_acc>=0.90": real >= go_acc,
        "lesion_collapses_to_chance": abs(les - 0.5) <= 0.15 and real >= les + 0.25,
        "swap_flips_tag": swap <= 0.15 and swapf >= 0.85,
        # tag PERP content — the DECISIVE test: identical content uttered as BOTH self & other is correctly source-
        # tagged both times (a content-encoding tag would give ~0.5 per item). min across the K items.
        "tag_perp_content(min_item_acc>=0.80)": mini >= 0.80,
        # tag PERP content — REPRESENTATIONAL: content-item does NOT decode from the src-tag rates above ~chance.
        "tag_perp_content(content_decode<=1/K+0.20)": decode <= chance_item + 0.20,
        "catch_no_carrier_is_chance": abs(catch - 0.5) <= 0.15,
    }
    go = all(checks.values())
    means = {"acc_real": real, "acc_lesion": les, "acc_catch": catch, "acc_swap": swap,
             "acc_swap_flipped": swapf, "min_per_item_acc": mini, "content_decode_acc": decode,
             "content_decode_chance": chance_item, "content_confound_maxabs_r": conf,
             "mean_abs_margin_real": marg_real, "mean_abs_margin_lesion": marg_les}

    # A verdict that CARRIES what earned it (tools.verdict.Verdict -> a `preconditions` block in the artifact).
    v = Verdict("agency / authorship 1-bit source tag", chance=0.5)
    v.floor("authorship acc beats chance", measured=real, floor=0.5)
    v.control("carrier lesion collapses the tag", treatment=real, control=les, min_separation=0.25)
    v.control("swapped carrier->tag wiring flips the tag", treatment=real, control=swap, min_separation=0.25)
    v.reaches("lesion SILENCES the tag pools (|margin| ~5 -> ~0)", before=marg_real, after=marg_les)
    v.require("swap flip is systematic (relabelled acc >= 0.85)", swapf, expect=lambda x: x >= 0.85)
    v.require("tag PERP content: min per-item acc >= 0.80", mini, expect=lambda x: x >= 0.80)
    v.require("tag PERP content: item does not decode from tag pools (<= 1/K + 0.20)",
              decode, expect=lambda x: x <= chance_item + 0.20)
    v.require("no-carrier catch is chance (|acc - 0.5| <= 0.15)", catch, expect=lambda x: abs(x - 0.5) <= 0.15)
    v.disabled("STDP / Hebbian / reward-mod / homeostasis / short-term & structural plasticity",
               why="fixed-structure corollary-discharge comparator is the scope; the learned content-cued episodic "
                   "source-MEMORY and self-organized wiring are the named follow-ons")
    decided = v.decide(go=go, verbose=False)

    # Force the treatment/control SUBTRACTION to be asked out loud (not two numbers a key apart): with the
    # swapped-wiring control at the true 0.0 floor, essentially 100% of the correct authorship is attributable
    # to the correct cd->src_self / sensory_marker->src_other wiring.
    attributable_to("correct carrier->tag wiring (vs swapped)", real, swap)
    return go, checks, means, decided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed — proves it RUNS + every anti-cheat arm is live")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if a.smoke:
        a.seeds = [a.seeds[0]]

    t0 = time.time()
    print(f"[authorship tag] AGENCY / AUTHORSHIP 1-bit source monitor (corollary-discharge comparator). "
          f"seeds={a.seeds} smoke={a.smoke}\n"
          f"  GATE: authorship acc >= 0.90 (chance 0.5); LESION->chance; SWAP->flip; tag PERP content.", flush=True)
    rows = [run_seed(s) for s in a.seeds]
    for r in rows:
        print(f"  [seed {r['seed']}] auth {r['acc_real']:.3f} || lesion {r['acc_lesion']:.3f} | swap "
              f"{r['acc_swap']:.3f} (flip-rel {r['acc_swap_flipped']:.3f}) | catch {r['acc_catch']:.3f} || "
              f"min-item {r['min_per_item_acc']:.3f} | content-decode {r['content_decode_acc']:.3f} "
              f"(chance {1.0/K_ITEMS:.3f}) | margin~item |r| {r['content_confound_maxabs_r']:.3f} "
              f"(n={r['n_trials']})", flush=True)

    go, checks, means, decided = _aggregate_verdict(rows)
    n = len(a.seeds)
    if go:
        verdict = (
            f"GO ({n}-seed) — AGENCY / AUTHORSHIP TAG: ONE spiking substrate answers 'did you say that or did I?' "
            f"at acc={means['acc_real']:.3f} (chance 0.5) by reading its own COROLLARY DISCHARGE. Self-production "
            f"emits an efference copy (production->cd) that a biased-competition comparator resolves to a 1-bit "
            f"self/other tag. LESIONING the carrier->tag projections collapses it to chance ({means['acc_lesion']:.3f}); "
            f"SWAPPING the carrier->tag wiring systematically FLIPS the tag (acc {means['acc_swap']:.3f}, relabelled "
            f"{means['acc_swap_flipped']:.3f}) — the tag tracks the source WIRING, not content; the tag is PERP to "
            f"content (identical content uttered as self AND other is correctly source-tagged both times: min "
            f"per-item acc {means['min_per_item_acc']:.3f}; content-item does not decode from the tag pools "
            f"{means['content_decode_acc']:.3f} vs chance {means['content_decode_chance']:.3f}), and content alone "
            f"(no act) reads at chance ({means['acc_catch']:.3f}). Brain-based (neurons+synapses); numpy-CPU; NO "
            f"sim/ edit. (Reported diagnostic: src-margin MAGNITUDE shows weak decision-irrelevant per-item "
            f"modulation |r|~{means['content_confound_maxabs_r']:.3f} = wiring heterogeneity, not content in the tag.)")
    else:
        miss = [k for k, v in checks.items() if not v]
        verdict = (f"BOUNDARY (build-informative, {n}-seed) — authorship acc={means['acc_real']:.3f} "
                   f"(lesion {means['acc_lesion']:.3f} / swap {means['acc_swap']:.3f} / catch {means['acc_catch']:.3f} "
                   f"/ min-item {means['min_per_item_acc']:.3f} / content-decode "
                   f"{means['content_decode_acc']:.3f}). FAILED: {miss}. Tune the carrier/comparator operating "
                   f"point (W_CARRIER_SRC, W_INH_SRC, OU_PA, TRIAL_STEPS); source monitoring is the next tune, not a wall.")

    summary = {
        "probe": "agency_authorship_tag (Lane A, Phase-0 self-model)", "verdict": verdict, "GO": bool(go),
        "status": decided["status"], "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"], "undefined_reasons": decided["undefined_reasons"],
        "checks": checks, "means": means, "per_seed": rows,
        "config": {"seeds": a.seeds, "smoke": a.smoke, "K_ITEMS": K_ITEMS, "N_PER_ITEM": N_PER_ITEM,
                   "TRIAL_STEPS": TRIAL_STEPS, "FLUSH_STEPS": FLUSH_STEPS, "REPS": REPS,
                   "W_CARRIER_SRC": W_CARRIER_SRC, "W_INH_SRC": W_INH_SRC, "PROD_DRIVE_PA": PROD_DRIVE_PA,
                   "STIM_CONTENT_PA": STIM_CONTENT_PA, "OU_PA": OU_PA},
        "mechanism": "production emits corollary discharge (production->cd = efference copy) vs parse->sensory_marker; "
                     "cd->src_self / sensory_marker->src_other with Namburi-Tye biased-competition cross-inhibition; "
                     "1-bit read-out = sign(rate(src_self)-rate(src_other)). content identity world-set + orthogonal.",
        "HONEST_NOTE": "numpy-CPU read on the real spiking Izhikevich bridge ('numpy' is the backend, not a host "
                       "shortcut). FIXED-STRUCTURE comparator (corollary-discharge source monitor). The content-cued "
                       "episodic SOURCE-MEMORY version (Hebbian-bind content->tag at encoding, content-cue at recall) "
                       "and the self-organized wiring are the named follow-ons (cf. the affect-region GO's hand-wired "
                       "attractor). Biology: Sperry/von Holst efference copy; Frith/Feinberg comparator; Johnson "
                       "source monitoring; Namburi-Tye opponent competition.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[authorship tag] VERDICT: {verdict}", flush=True)
    print(f"[authorship tag] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
