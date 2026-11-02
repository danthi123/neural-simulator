"""DISCRETE CHAT ACTION SELECTION — SPEAK vs STAY-SILENT via the two-channel spiking basal-ganglia selector (2026-08-26).

This is the production consumer for the Gate-A v2 vocal action-selection GO (4 development seeds):
  * research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md
  * runner: research/runners/_vocal_action_selector_gate.py

A discrete chat action decision — should the brain SPEAK this turn, or STAY SILENT (hold, emit nothing salient)? — is
routed through a genuine two-channel basal-ganglia RACE instead of a host `if`. Each candidate action is a competing
striatal D1 channel: channel 0 = SPEAK, channel 1 = STAY-SILENT. The composer hands the selector a per-candidate
SALIENCE (SPEAK salience = the turn has answerable content; STAY-SILENT salience = the turn is content-empty). Two
drives combine at the striatum, mirroring the biology: (1) SHARED practice arousal -> the cortical proposal populations
-> a strong proposal->D1 barrage that brings BOTH channels' MSNs toward threshold (the enabling drive); (2) the
per-candidate salience -> a modulatory EXCITABILITY BIAS on that candidate's D1 MSN pool (which channel the shared
barrage pushes over first). Ongoing OU noise breaks any residual symmetry. The FIRST channel whose motor pool crosses
the GPi->thalamus disinhibition commit threshold IS the selected action (a real race, NOT an argmax over rates).
Crucially the salience bias ALONE cannot fire the MSNs — without the arousal-driven cortical barrage the D1 pool stays
sub-threshold — so SHARED AROUSAL is intrinsically load-bearing (the finding's control, reproduced: no arousal -> no
commit at any salience). Reuse-by-import of the de-risk selector (`build_selector_bridge` + the v2 topology + the
direct-path transmission gate); NO `sim/` edit; additive.

BRAIN-BASED-ONLY boundary (CLAUDE.md standing standard): host code is legitimate ONLY for the ENVIRONMENT/body — here,
the per-candidate SALIENCE is a cortical/neuromodulatory afferent the composer supplies (the same declared boundary the
SVO parser / the vision percept occupy), and surfacing a STAY-SILENT commit as a brief hold line is the body's
articulation layer. Everything between the salience input and the selected action — the striatal D1 channels, the
D1->GPi direct-path disinhibition, the GPe/STN indirect path, the GPi->thalamus gate, the thalamo-cortical commit
burst, the cross-channel commit inhibition — is neurons/synapses on a real `SimulationBridge` (NO numpy argmax
anywhere). Which action wins is the substrate's race, not a host max.

Flag: `BRAIN_BG_SELECT` (2026-08-26 FLIPPED DEFAULT-ON, wave 3, 6/6 flip-soak GO; `BRAIN_BG_SELECT=0` is the
byte-identical escape to the pre-flip turn — the wiring block reads the flag first and imports nothing). When ON, the selector is
CONSULTED only on a content-empty turn (STAY-SILENT is a genuine contender there); it SHORT-CIRCUITS the turn with a
hold ONLY when the BG race COMMITS to STAY-SILENT. A SPEAK commit, a non-commit, or an ordinary content turn all fall
through to the normal path.

Lesion oracles (the finding's OWN Gate-A anti-cheats, reused verbatim):
  * NO SHARED AROUSAL (`BRAIN_BG_SELECT_LESION=arousal`): remove the shared practice-arousal state -> no proposal->D1
    barrage AND the arousal-gated salience bias is withdrawn (the UP-state gating below) -> the D1 MSNs cannot commit
    at any salience -> `decide_action()` returns None -> the turn falls through to the default/host path. This is the
    de-risk's own arousal-off control (which floored 0/400). STAY-SILENT VANISHES.
  * NO DIRECT-PATH DISINHIBITION (`BRAIN_BG_SELECT_LESION=direct_path`): cut the D1->GPi direct path (the transmission
    gate to 0) -> GPi is never inhibited -> the thalamus is never disinhibited -> NO commit -> None -> fall through.
    The STAY-SILENT decision VANISHES.
Either lesion collapses the commit, so the selected-action change is attributable to the BG cascade (shared arousal +
the gated direct path), NOT to a host argmax — exactly the finding's `arousal_is_load_bearing` /
`direct_path_is_load_bearing` controls, now the production load-bearing proof.
"""
from __future__ import annotations

import os
import re

import numpy as np

# reuse-by-import: the Gate-A v2 selector (4-seed GO). We import the bridge builder + the v2 topology + the primitives
# and add ONE production ingredient — a per-candidate salience bias on the cortical proposal drive. NO sim/ edit; the
# de-risk runner is untouched.
from research.runners._vocal_action_selector_gate import (
    build_selector_bridge,
    selector_config,
    DIRECT_PATH_GATE,
    CHANNELS,
    _indices,
    _step,
    _set_equal_tonic_current,
    tonic_drive_enabled,
    tonic_drive_pA,
)
from sim.backend import get_backend, to_host
# reuse-by-import the shared spiking novelty/salience afferent (scaffold-retirement backlog rank-4, 2026-09-05,
# research/runners/shared_salience_afferent.py) -- default-OFF (BRAIN_SHARED_SALIENCE); see salience()'s docstring.
import research.runners.shared_salience_afferent as _SHARED


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The two candidate chat actions = the two striatal channels of the de-risk selector.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
SPEAK = 0
STAY_SILENT = 1
ACTION_NAME = {0: "SPEAK", 1: "STAY_SILENT"}

# The brief, honest surface for a STAY-SILENT commit (the body's articulation of "the brain chose to hold"). It is a
# hold, not a fabricated answer: the moat's abstain honesty is preserved.
HOLD_TEXT = "(Holding — nothing salient to add to that.)"

# Salience gain: how strongly a candidate's salience (in [0, 1]) biases its channel's striatal D1 MSN pool (pA added
# to that channel's `str_d1` population — a modulatory excitability bias, NOT the bulk drive). Calibrated (seed-42
# smoke, confirmed 6-seed by the flip-soak) so that
#   (a) WITH shared arousal (the proposal->D1 barrage present), the higher-salience channel wins the single race
#       reliably (salience DRIVES the selection: 8/8 both directions at seed 42); and
#   (b) WITHOUT shared arousal the bias is WITHDRAWN (it is arousal-gated — see `_run_biased_trial`) and there is no
#       proposal->D1 barrage, so NOTHING commits at any salience — arousal stays strictly load-bearing (the exact
#       de-risk arousal-off control, floored on every seed). The direct-path lesion floors independently (it cuts the
#       D1->GPi gate). This is the implicit operating point the animal runs (bulk cortical drive + a modulatory,
#       arousal-gated striatal bias); we set it explicitly.
SALIENCE_GAIN_PA = 600.0

# A content token = an alphanumeric run of length >= 2. The count is the environment/body's crude salience read of the
# incoming message (a cortical afferent to the selector, NOT part of the neural selection).
_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")


def bg_select_enabled() -> bool:
    """DEFAULT-ON (2026-08-26 flip, wave 3, 6/6 flip-soak GO). `BRAIN_BG_SELECT` unset -> the SPEAK-vs-STAY-SILENT
    BG selector is live; an explicit off (0/false/no/off/'') -> byte-identical to pre-flip. Mirrors the server.py
    `_BG_SELECT_DEFAULT_ON` anchor and its `_bg_select_flag_on()` reader."""
    v = os.environ.get("BRAIN_BG_SELECT")
    return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))


def bg_select_lesion() -> str | None:
    """Read the live lesion flag. `BRAIN_BG_SELECT_LESION` in {arousal} -> remove shared arousal; in
    {direct_path,direct,directpath} -> cut the D1->GPi direct path; a bare truthy value defaults to 'arousal'. Any
    other/unset value -> None (intact). Either lesion collapses every commit -> the STAY-SILENT decision vanishes."""
    v = os.environ.get("BRAIN_BG_SELECT_LESION")
    if v is None:
        return None
    s = v.strip().lower()
    if s in ("arousal", "no_arousal", "noarousal"):
        return "arousal"
    if s in ("direct_path", "direct", "directpath", "no_direct", "nodirect"):
        return "direct_path"
    if s in ("1", "true", "yes", "on"):
        return "arousal"
    return None


def salience(msg: str) -> tuple[float, float]:
    """The composer's per-candidate salience read of `msg` -> (speak_salience, silent_salience), each in [0, 1].

    SPEAK salience rises with answerable content (saturates at 2 content tokens); STAY-SILENT salience is high only when
    the message carries no content token (a bare '...', a lone symbol). This is the environment/body salience layer — a
    cortical afferent to the neural selector, deliberately narrow so STAY-SILENT is a genuine contender ONLY on a
    content-empty turn (a normal question always favors SPEAK, so the selector is never even consulted on it).

    SHARED SPIKING AFFERENT (rank-4, default-OFF `BRAIN_SHARED_SALIENCE`, research/runners/shared_salience_afferent.py).
    The ENTRY GATE (whether the turn even carries a content token — the environment's crude "is anything here" read)
    stays the SAME host boolean (`n == 0`); what changes is the SALIENCE MAGNITUDE the composer hands the selector on a
    content-empty turn: instead of the hardcoded (0.0, 1.0) pair, `speak` is the shared curiosity-organ ASK-pool's
    spiking transduction of the SAME raw content-count scalar (`min(1, n/2)`), and `silent = 1 - speak`. So the exact
    bias the striatal D1 race receives at the ONE reachable STAY-SILENT-candidate point is now a genuine spiking read
    (mediated by the shared ASK-pool population), not two bare host formulas — and it collapses toward the SAME
    baseline as the DA-mode/value-choice consumers under the SAME `BRAIN_SHARED_SALIENCE_LESION` lesion. OFF (unset)
    -> byte-identical to the bare host formula below."""
    n = len(_CONTENT_TOKEN_RE.findall(msg or ""))
    if _SHARED.shared_salience_enabled():
        raw = min(1.0, n / 2.0)
        speak = float(max(0.0, _SHARED.read_salience(raw)["normalized"]))
        silent = max(0.0, 1.0 - speak) if n == 0 else 0.0   # entry-gate boolean unchanged (host content-count == 0)
        return float(speak), float(silent)
    speak = min(1.0, n / 2.0)
    silent = max(0.0, 1.0 - float(n))
    return float(speak), float(silent)


def _run_biased_trial(bridge, config, sal_speak, sal_silent, *, arousal=True, gain=SALIENCE_GAIN_PA):
    """ONE basal-ganglia race with a per-candidate salience bias. Mirrors the de-risk `_run_trial` commit-detection
    (first channel to cross `commit_threshold_spikes` with the other suppressed below `clean_loser_ratio` = the winner)
    and its between-trial reset+washout, ADDING the salience bias as a striatal-D1 EXCITABILITY modulation. The bulk
    drive still flows shared-arousal -> proposal -> D1, so `arousal=False` (the NO-AROUSAL lesion) leaves the MSNs
    sub-threshold and NOTHING commits at any salience."""
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, config)
    motor_idx = {ch: _indices(bridge, f"motor_{ch}") for ch in CHANNELS}
    if arousal:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(config.practice_pA)
        # per-candidate salience -> a modulatory excitability bias on that channel's striatal D1 MSN pool. The bias is
        # GATED BY the shared arousal state (striatal UP-state / neuromodulatory gating: cortical salience only reaches
        # MSN threshold when the enabling arousal context is present). Consequently the NO-AROUSAL lesion removes BOTH
        # the proposal->D1 barrage AND the salience gain -> the MSNs cannot commit at any salience (the de-risk's own
        # arousal-off control, which floored 0/400). Intact, the bias only tips WHICH arousal-driven channel wins.
        sal = (float(sal_speak), float(sal_silent))
        d1_idx = {ch: xp.asarray(_indices(bridge, f"str_d1_{ch}")) for ch in CHANNELS}
        for ch in CHANNELS:
            idx = d1_idx[ch]
            bridge.cp_external_input_current[idx] = (
                bridge.cp_external_input_current[idx] + xp.float32(sal[ch] * gain)
            )

    counts = np.zeros(2, dtype=np.int64)
    first_crossing = None
    simultaneous = False
    decision_step = None
    for step in range(int(config.action_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        previous = counts.copy()
        for ch in CHANNELS:
            counts[ch] += int(firing[motor_idx[ch]].sum())
        crossed = [
            ch for ch in CHANNELS
            if previous[ch] < config.commit_threshold_spikes <= counts[ch]
        ]
        if first_crossing is None and len(crossed) == 1:
            first_crossing = int(crossed[0])
            decision_step = int(step)
            break
        elif first_crossing is None and len(crossed) > 1:
            simultaneous = True
            decision_step = int(step)
            break

    winner = None
    loser_ratio = None
    if first_crossing is not None and not simultaneous:
        loser = 1 - first_crossing
        loser_ratio = float(counts[loser] / max(1, counts[first_crossing]))
        if loser_ratio <= config.clean_loser_ratio:
            winner = int(first_crossing)

    # between-trial reset + washout (verbatim de-risk behavior so a cached bridge returns to the resting operating point).
    _set_equal_tonic_current(bridge, config)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_equal_tonic_current(bridge, config)
    _step(bridge, config.washout_steps)
    return {
        "winner": winner,
        "first_crossing": first_crossing,
        "committed": winner is not None,
        "simultaneous": bool(simultaneous),
        "motor_spikes": counts.tolist(),
        "loser_ratio": loser_ratio,
        "decision_step": decision_step,
    }


class BGActionSelector:
    """A warm two-channel BG selector for chat action selection. Built ONCE (lazily) on a real `SimulationBridge` from
    the de-risk v2 topology (the GO topology). `lesion` in {None, 'arousal', 'direct_path'} installs the finding's own
    anti-cheat: 'direct_path' cuts the D1->GPi transmission gate at build; 'arousal' removes the shared practice-arousal
    drive per trial. `select_once(speak_sal, silent_sal)` runs ONE salience-biased race -> the selected action."""

    def __init__(self, seed: int = 42, version: str = "v2", lesion: str | None = None):
        self.seed = int(seed)
        self.config = selector_config(version)
        self.lesion = lesion
        self._bridge = None
        self._built = False

    def ensure_built(self):
        if self._built:
            return
        b = build_selector_bridge(self.seed, self.config)
        # direct-path lesion = the finding's NO-DIRECT-PATH control (gate 0 -> GPi never inhibited -> no disinhibition).
        gate = 0.0 if self.lesion == "direct_path" else 1.0
        b.set_transmission_gate(DIRECT_PATH_GATE, gate)
        _set_equal_tonic_current(b, self.config)
        _step(b, self.config.warmup_steps)
        self._bridge = b
        self._built = True

    def select_once(self, speak_salience: float, silent_salience: float) -> dict:
        """Run one salience-biased BG race. Returns the `_run_biased_trial` dict (winner in {0,1} or None)."""
        self.ensure_built()
        arousal = self.lesion != "arousal"
        return _run_biased_trial(
            self._bridge, self.config, speak_salience, silent_salience, arousal=arousal
        )


# process-shared selector cache, keyed by (seed, lesion) so the intact + two lesion variants coexist (the load-bearing
# verify + the soak build all three).
_ORGANS: dict = {}


def get_organ(seed: int = 42, lesion: str | None = None) -> BGActionSelector:
    key = (int(seed), lesion)
    org = _ORGANS.get(key)
    if org is None:
        org = BGActionSelector(seed=seed, lesion=lesion)
        _ORGANS[key] = org
    return org


def reset_organs():
    """Drop every cached selector (tests/soak that rebuild across configs)."""
    _ORGANS.clear()


def decide_action(msg: str, seed: int = 42, lesion: str | None = None) -> dict | None:
    """Consult the BG selector for the SPEAK-vs-STAY-SILENT decision on this turn, or None to speak/fall through.

    Returns None when: STAY-SILENT is not a genuine contender (a normal content turn -> speak by default, the selector
    is not even consulted), OR the BG race did NOT commit (either lesion floors it, or no clean winner), OR the BG race
    committed to SPEAK. In all those cases the live handler proceeds with the normal SPEAK path, so an ordinary turn and
    a lesioned/no-commit turn are BYTE-IDENTICAL to the flag-off behavior (the load-bearing lesion-vanish).

    Returns a dict only when the BG race COMMITS to STAY-SILENT:
      {'action': 'STAY_SILENT', 'winner': 1, 'speak_salience': .., 'silent_salience': .., 'decision_step': ..,
       'loser_ratio': .., 'lesion': None}
    The winner is the SPIKING race outcome (not a host max): under either lesion it never reaches here (no commit),
    which is exactly why a STAY-SILENT hold is a read of the substrate, not a host branch."""
    speak_sal, silent_sal = salience(msg)
    if not (silent_sal > speak_sal):
        return None  # STAY-SILENT is not on the table this turn -> speak by default (fall through, byte-identical)
    les = bg_select_lesion() if lesion is None else lesion
    org = get_organ(seed=seed, lesion=les)
    r = org.select_once(speak_sal, silent_sal)
    if not r["committed"] or r["winner"] != STAY_SILENT:
        return None  # no commit (lesion floor / no clean winner) OR the BG chose SPEAK -> fall through
    return {
        "action": ACTION_NAME[STAY_SILENT],
        "winner": int(r["winner"]),
        "speak_salience": speak_sal,
        "silent_salience": silent_sal,
        "decision_step": r["decision_step"],
        "loser_ratio": r["loser_ratio"],
        "lesion": les,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BEGIN neural-seam (lane A, branch research/selection-seam-need-reversal, 2026-09-29). DEFAULT-OFF.
# Owner of this region: the selection-seam lane (value INPUT path + neural COMMITMENT). Corticostriatal PLASTICITY on
# the value-afferent pathways below belongs to the long-delay-credit lane: it may make `VALUE_AFFERENT_PLASTICITY_GATE`
# plastic in its own functions; this region keeps them frozen. Nothing above this line is changed by this region.
#
# WHAT IT RETIRES (plan docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md step 3). With the flag OFF the
# production seam has three host decisions: (1) the ADMISSION pre-filter in `decide_action` (`if not (silent_sal >
# speak_sal): return None`: the race is consulted only when the host's own scores already favor STAY-SILENT); (2) the
# host SCORING `salience(msg)` (a content-token count mapped to two scalars) injected as `salience * 600 pA` into the
# D1 pools; (3) the host COMMIT read in `_run_biased_trial` (the first motor pool to reach 12 spikes, then `break`,
# then a 0.25 loser ratio) -- the trial stops before the commit circuit acts, so the host timing rule is the commitment.
# With `BRAIN_BG_NEURAL_SEAM=1`:
#   (1) no pre-filter: the race runs on EVERY turn;
#   (2) no host score: the message is rendered as a SENSORY stimulus (each word -> a fixed group of `heard` neurons,
#       the body/senses boundary) and value reaches the D1 pools only through synapses: heard -> str_d1_SPEAK, and a
#       tonically driven `hold` population (the default when nothing is heard) -> str_d1_STAY_SILENT, with heard ->
#       hold_inh -| hold (words suppress the default). The host computes no count, no comparison, no threshold;
#   (3) neural commitment: per channel an action LATCH (recurrent NMDA, bistable) driven by that channel's motor pool,
#       with mutual FS inhibition between latches. The race runs a FIXED window (no break at a crossing); arousal and
#       the stimulus are withdrawn on a fixed schedule; the body reads which latch is still active in the final read
#       window. The latch that ignites first suppresses the other, and it holds after the drive is gone.
# RESIDUAL HOST STEPS (declared): the word rendering (which character runs are words; the same regex as
# `_CONTENT_TOKEN_RE`), the trial schedule, and the body's read of the latch (a rate threshold on two populations;
# its share is measured by re-reading the same trials with other rules, `readouts` in the trial record).
# LEARNED-CONTENT FRACTION: 0. The heard->SPEAK and hold->SILENT routes are wired, not learned.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
import zlib as _zlib

NEURAL_SEAM_FLAG = "BRAIN_BG_NEURAL_SEAM"
NEURAL_SEAM_LESION_FLAG = "BRAIN_BG_NEURAL_SEAM_LESION"
VALUE_AFFERENT_GATE = "bg_value_afferent"                  # transmission gate on heard->D1 and hold->D1 (lesion)
VALUE_AFFERENT_PLASTICITY_GATE = "bg_value_afferent_plasticity"   # frozen here; the credit lane's handle
LATCH_INPUT_GATE = "bg_action_latch_input"                 # transmission gate on motor->latch (lesion)

N_HEARD = 60
HEARD_GROUP = 12              # neurons per rendered word (a fixed crc32-addressed group)
HEARD_PA = 500.0              # sensory drive onto a rendered word's group during the stimulus window
N_HOLD = 24
HOLD_TONIC_PA = 350.0         # the default-mode drive onto `hold`, identical on every turn
N_HOLD_INH = 16
VALUE_DENSITY = 0.5
VALUE_WEIGHT = 20.0           # heard/hold -> str_d1, slow NMDA (voltage-gated: biases MSNs the barrage depolarizes)
HEARD_TO_HOLD_INH_W = 30.0
HOLD_INH_TO_HOLD_W = 20.0
N_LATCH = 30
LATCH_DENSITY = 0.5
LATCH_INTERNAL_W = 64.0       # DEV (seeds 1-7): 48 latches unreliably; 64 self-sustains after the drive is withdrawn
MOTOR_TO_LATCH_W = 40.0        # DEV (seeds 1, 5, 7): 80 drove latch neurons to +68..+145 mV during refractory; 40 <= +47
LATCH_TO_FS_W = 30.0
LATCH_CROSS_W = 10.0          # DEV (seeds 1, 2, 7): 20-40 overshot the losing latch to -121..-141 mV
N_LATCH_FS = 16
T_DRIVE = 300                 # arousal + stimulus window (ms)
T_HOLD = 200                  # drive withdrawn; the latch must hold
READ_WINDOW = 100             # the body reads the last READ_WINDOW ms
LATCH_READ_RATE = 0.02        # spikes/neuron/ms that count a latch as active (~20 Hz); readout variants re-read it
# RULING bg-gate-1101 instrument (report-only; nothing reads it to decide): the inhibitory synaptic reversal and the
# STEP1-SPEC STOP-4 bound (a step counts for a region when that region's lowest end-of-step v is below E_inh - 5 mV,
# the definition of research/runners/bg_tonic_loop_probe.py). Kept OUT of the trial record (golden-hashed by
# tests/test_bg_commit.py); written to `NeuralSeamSelector.last_diag` / `decide_action_neural.last_diag` instead.
DIAG_E_INH_MV = -75.0
DIAG_SUB_REVERSAL_MV = DIAG_E_INH_MV - 5.0
# RULING bg-gate-amend-1101 STOP-4: E_inh - 5 mV = -80 mV coincides with the MSN preset rest (vr = -80 mV), so the raw
# count above also counts preset MSN rest. Beside it (never replacing it) a per-region count against
# min(E_inh, vr_region) - 5 mV, vr_region = the region's Izhikevich rest/reset potential as configured on the built
# bridge (cp_izh_vr; the region's minimum if its neurons differ; E_inh alone when the bridge has no Izhikevich vr).
# STOP 4 is REPORT-ONLY in the bg-gate-1101 gate: neither count gates anything.
DIAG_SUB_REVERSAL_MARGIN_MV = 5.0
DIAG_RATE_REGIONS = ("gpi", "thal", "motor", "gpe")      # per channel; plus selector_stn

_WORD_RE = _CONTENT_TOKEN_RE  # the sensory word segmentation (declared residual host transduction)


def neural_seam_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_BG_NEURAL_SEAM` in {1,true,yes,on} -> the neural seam replaces decide_action's host
    admission, scoring and commit read. Unset / anything else -> byte-identical to today's production path."""
    return (os.environ.get(NEURAL_SEAM_FLAG) or "").strip().lower() in ("1", "true", "yes", "on")


def neural_seam_lesion() -> str | None:
    """`BRAIN_BG_NEURAL_SEAM_LESION` in {value, latch, arousal, direct_path}. value: close heard/hold -> D1;
    latch: close motor -> latch; arousal / direct_path: the selector's own anti-cheats."""
    v = (os.environ.get(NEURAL_SEAM_LESION_FLAG) or "").strip().lower()
    return v if v in ("value", "latch", "arousal", "direct_path") else None


def render_utterance(msg: str) -> list:
    """The body's auditory rendering: each word of `msg` -> the start index of a fixed HEARD_GROUP-neuron group in
    `heard` (crc32 of the lower-cased word, so the address is stable across processes; repeated words re-address the
    same group). No count, weight or comparison is computed here."""
    starts = []
    for w in _WORD_RE.findall(msg or ""):
        starts.append(int(_zlib.crc32(w.lower().encode("utf-8")) % (N_HEARD - HEARD_GROUP + 1)))
    return starts


def neural_seam_parts(commit: dict | None = None):
    """(extra_regions, extra_pathways, core_config_updates) added to the Gate-A v2 selector by the neural seam.
    `commit` (a BG_COMMIT_GRID row, flag BRAIN_BG_COMMIT): the spiking commit stage is appended AFTER every region and
    pathway below (region `bg-commit`); None -> byte-identical to the build without it."""
    from sim.enums import NeuronType
    from sim.regions import RegionPathway
    from research.runners._vocal_action_selector_gate import _region
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    regions = [
        _region("heard", N_HEARD, exc_fraction=1.0, neuron_type=rs),
        _region("hold", N_HOLD, exc_fraction=1.0, neuron_type=rs),
        _region("hold_inh", N_HOLD_INH, exc_fraction=0.0, neuron_type=fs),
    ]
    P = []
    for src, ch in (("heard", SPEAK), ("hold", STAY_SILENT)):
        P.append(RegionPathway(from_region=src, to_region=f"str_d1_{ch}", density=VALUE_DENSITY,
                               weight_mean=VALUE_WEIGHT, weight_jitter=0.05, plastic=False,
                               plasticity_gate=VALUE_AFFERENT_PLASTICITY_GATE,
                               transmission_gate=VALUE_AFFERENT_GATE, exc_receptor="nmda_slow"))
    P.append(RegionPathway(from_region="heard", to_region="hold_inh", density=1.0, weight_mean=HEARD_TO_HOLD_INH_W,
                           weight_jitter=0.0, plastic=False))
    P.append(RegionPathway(from_region="hold_inh", to_region="hold", density=1.0, weight_mean=HOLD_INH_TO_HOLD_W,
                           weight_jitter=0.0, plastic=False, receptor="gaba_a"))
    for ch in CHANNELS:
        other = 1 - ch
        regions += [_region(f"latch_{ch}", N_LATCH, exc_fraction=1.0, neuron_type=rs, internal_density=LATCH_DENSITY,
                            internal_weight=LATCH_INTERNAL_W, enable_nmda=True),
                    _region(f"latch_fs_{ch}", N_LATCH_FS, exc_fraction=0.0, neuron_type=fs)]
        P += [RegionPathway(from_region=f"motor_{ch}", to_region=f"latch_{ch}", density=1.0,
                            weight_mean=MOTOR_TO_LATCH_W, weight_jitter=0.05, plastic=False,
                            transmission_gate=LATCH_INPUT_GATE),
              RegionPathway(from_region=f"latch_{ch}", to_region=f"latch_fs_{ch}", density=1.0,
                            weight_mean=LATCH_TO_FS_W, weight_jitter=0.0, plastic=False),
              RegionPathway(from_region=f"latch_fs_{ch}", to_region=f"latch_{other}", density=1.0,
                            weight_mean=LATCH_CROSS_W, weight_jitter=0.0, plastic=False, receptor="gaba_a")]
    for tgt in ("heard", "hold", "latch_0", "latch_1"):
        P.append(RegionPathway(from_region="selector_reset", to_region=tgt, density=0.70, weight_mean=16.0,
                               weight_jitter=0.0, plastic=False, receptor="gaba_a"))
    if commit is not None:
        r2, p2 = bg_commit_parts(commit)
        regions += r2
        P += p2
    return regions, P, {"enable_nmda_recurrent": True}


def _neural_baseline(bridge, config):
    """Candidate-agnostic currents present between trials: the selector's equal tonic currents only."""
    _set_equal_tonic_current(bridge, config)


class _TrialDiag:
    """RULING bg-gate-1101 report-only instrument: per-region sub-reversal step counts and v_min over every v-checked
    step (trial, reset, washout: the membrane precondition's scope), and per-region spike counts over the drive and
    hold windows. It only READS the host copies of cp_membrane_potential_v / cp_firing_states the trial already makes,
    so the trial's state, spikes and record are unchanged. With BRAIN_BG_IZH_SUBSTEPS > 1 only the end-of-step v is
    visible, so `steps_below` counts bridge steps (dt), not sub-steps (the step-0 probe's definition)."""

    def __init__(self, bridge):
        self.names = [r.name for r in bridge.region_manager.regions()]
        self.idx = {n: _indices(bridge, n) for n in self.names}
        self.below = {n: 0 for n in self.names}
        vr_all = getattr(bridge, "cp_izh_vr", None)
        vr_host = np.asarray(to_host(vr_all)) if vr_all is not None else None
        self.vr = {}
        for n in self.names:
            ii = self.idx[n]
            self.vr[n] = (float(vr_host[ii].min()) if vr_host is not None and ii.size and vr_host.size > int(ii.max())
                          else None)
        self.rest_bound = {n: (min(DIAG_E_INH_MV, self.vr[n]) if self.vr[n] is not None else DIAG_E_INH_MV)
                           - DIAG_SUB_REVERSAL_MARGIN_MV for n in self.names}
        self.below_rest = {n: 0 for n in self.names}
        self.vmin = {n: float("inf") for n in self.names}
        self.steps_checked = 0
        rn = [f"{k}_{c}" for k in DIAG_RATE_REGIONS for c in CHANNELS] + ["selector_stn"]
        self.rate_names = [n for n in rn if n in self.idx and self.idx[n].size]
        self.spikes = {w: {n: 0 for n in self.rate_names} for w in ("drive", "hold")}

    def see_v(self, v):
        v = np.asarray(v)
        self.steps_checked += 1
        for n in self.names:
            ii = self.idx[n]
            if ii.size == 0:
                continue
            lo = float(v[ii].min())
            if lo < DIAG_SUB_REVERSAL_MV:
                self.below[n] += 1
            if lo < self.rest_bound[n]:
                self.below_rest[n] += 1
            if lo < self.vmin[n]:
                self.vmin[n] = lo

    def see_f(self, f, window):
        acc = self.spikes[window]
        for n in self.rate_names:
            acc[n] += int(f[self.idx[n]].sum())

    def record(self, dt_ms):
        rates = {}
        for w, steps in (("drive", T_DRIVE), ("hold", T_HOLD)):
            rates[w] = {n: round(float(self.spikes[w][n]) / (float(self.idx[n].size) * steps * dt_ms) * 1000.0, 2)
                        for n in self.rate_names}
        rates["window_steps"] = {"drive": int(T_DRIVE), "hold": int(T_HOLD)}
        return {"sub_reversal": {"threshold_mv": DIAG_SUB_REVERSAL_MV, "e_inh_mv": DIAG_E_INH_MV,
                                 "steps_checked": int(self.steps_checked), "steps_below": dict(self.below),
                                 "v_min": {n: (round(x, 3) if np.isfinite(x) else None) for n, x in self.vmin.items()},
                                 "regions_below": sorted(n for n, k in self.below.items() if k),
                                 # RULING bg-gate-amend-1101 STOP-4 (report-only; STOP 4 never gates this gate)
                                 "vr_mv": {n: (round(x, 3) if x is not None else None) for n, x in self.vr.items()},
                                 "rest_bound_mv": {n: round(x, 3) for n, x in self.rest_bound.items()},
                                 "steps_below_rest_bound": dict(self.below_rest),
                                 "regions_below_rest_bound": sorted(n for n, k in self.below_rest.items() if k),
                                 "stop4": "report-only"},
                "bg_rates_hz": rates}


def _run_neural_seam_trial(bridge, config, heard_starts, *, arousal=True, v_check=True, commit=None, diag=None):
    """ONE neural-seam race on a fixed schedule. The host adds only candidate-agnostic drive (equal tonic, shared
    arousal, the default-mode `hold` drive) and the rendered stimulus. Returns the latch read (the committed action)
    plus the same trial re-read under alternative host readout rules (`readouts`), the membrane range over EVERY
    step, and the latch rates. `diag` (a `_TrialDiag`, report-only) also sees every firing / membrane read."""
    xp, _ = get_backend()
    names = [f"motor_{c}" for c in CHANNELS] + [f"latch_{c}" for c in CHANNELS]
    idx = {n: _indices(bridge, n) for n in names}
    heard_idx = _indices(bridge, "heard")
    hold_idx = xp.asarray(_indices(bridge, "hold"))
    stim = np.zeros(len(heard_idx), dtype=np.float32)
    for s in heard_starts:
        stim[s:s + HEARD_GROUP] = HEARD_PA
    stim_dev = xp.asarray(stim)
    heard_dev = xp.asarray(heard_idx)

    def drive_on():
        _neural_baseline(bridge, config)
        if arousal:
            bridge.cp_external_input_current[xp.asarray(_indices(bridge, "practice_arousal"))] = \
                xp.float32(config.practice_pA)
        bridge.cp_external_input_current[hold_idx] = xp.float32(HOLD_TONIC_PA)
        bridge.cp_external_input_current[heard_dev] = stim_dev

    T = T_DRIVE + T_HOLD
    motor_t = np.zeros((T, 2), dtype=np.int32)
    latch_t = np.zeros((T, 2), dtype=np.int32)
    if commit is not None:                            # bg-commit: the stage's gate and latch spikes, every step
        cidx = bg_commit_indices(bridge)
        gate_t = np.zeros((T, 2), dtype=np.int32)
        slot_t = np.zeros((T, 2), dtype=np.int32)
    # amendment 2 report-only diagnostics: gpe_{ch} and selector_stn spike counts per step, reported as
    # baseline Hz over the drive window in `out` -- accumulation and reporting are flag-ON only.
    tonic_on = tonic_drive_enabled()
    gpe_idx = {c: _indices(bridge, f"gpe_{c}") for c in CHANNELS}
    stn_idx = _indices(bridge, "selector_stn")
    gpe_t = np.zeros((T, 2), dtype=np.int32)
    stn_t = np.zeros(T, dtype=np.int32)
    v_lo, v_hi = np.inf, -np.inf
    drive_on()
    for t in range(T):
        if t == T_DRIVE:
            _neural_baseline(bridge, config)          # fixed schedule: arousal, hold drive and stimulus withdrawn
        _step(bridge)
        f = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        if diag is not None:
            diag.see_f(f, "drive" if t < T_DRIVE else "hold")
        for c in CHANNELS:
            motor_t[t, c] = int(f[idx[f"motor_{c}"]].sum())
            latch_t[t, c] = int(f[idx[f"latch_{c}"]].sum())
            if commit is not None:
                gate_t[t, c] = int(f[cidx["gate"][c]].sum())
                slot_t[t, c] = int(f[cidx["slot"][c]].sum())
            if tonic_on:
                gpe_t[t, c] = int(f[gpe_idx[c]].sum())
        if tonic_on:
            stn_t[t] = int(f[stn_idx].sum())
        if v_check:
            v = to_host(bridge.cp_membrane_potential_v)
            v_lo, v_hi = min(v_lo, float(np.min(v))), max(v_hi, float(np.max(v)))
            if diag is not None:
                diag.see_v(v)
    # between-trial reset + washout (the selector's own, candidate-agnostic), membrane checked there too
    _neural_baseline(bridge, config)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "selector_reset"))] = xp.float32(config.reset_pA)
    for _ in range(int(config.reset_steps)):
        _step(bridge)
        if v_check:
            v = to_host(bridge.cp_membrane_potential_v)
            v_lo, v_hi = min(v_lo, float(np.min(v))), max(v_hi, float(np.max(v)))
            if diag is not None:
                diag.see_v(v)
    _neural_baseline(bridge, config)
    for _ in range(int(config.washout_steps)):
        _step(bridge)
        if v_check:
            v = to_host(bridge.cp_membrane_potential_v)
            v_lo, v_hi = min(v_lo, float(np.min(v))), max(v_hi, float(np.max(v)))
            if diag is not None:
                diag.see_v(v)

    def latch_read(rate_thr):
        rates = latch_t[T - READ_WINDOW:].sum(axis=0) / float(READ_WINDOW * N_LATCH)
        act = [c for c in CHANNELS if rates[c] >= rate_thr]
        return (act[0] if len(act) == 1 else None), [round(float(r), 4) for r in rates]

    def first_crossing(thr, loser_ratio=config.clean_loser_ratio):
        cum = np.cumsum(motor_t[:T_DRIVE], axis=0)
        for t in range(T_DRIVE):
            crossed = [c for c in CHANNELS if cum[t, c] >= thr and (t == 0 or cum[t - 1, c] < thr)]
            if len(crossed) > 1:
                return None
            if len(crossed) == 1:
                w = crossed[0]
                return w if cum[t, 1 - w] / max(1, cum[t, w]) <= loser_ratio else None
        return None

    def majority():
        tot = motor_t[:T_DRIVE].sum(axis=0)
        return None if tot[0] == tot[1] else int(np.argmax(tot))

    winner, latch_rates = latch_read(LATCH_READ_RATE)
    readouts = {
        "latch_rate_x0.5": latch_read(LATCH_READ_RATE * 0.5)[0],
        "latch_rate_x2": latch_read(LATCH_READ_RATE * 2.0)[0],
        "first_to_12_motor (production rule)": first_crossing(config.commit_threshold_spikes),
        "first_to_6_motor": first_crossing(6),
        "first_to_24_motor": first_crossing(24),
        "motor_majority_drive_window": majority(),
    }
    out = {"winner": winner, "committed": winner is not None, "latch_rates": latch_rates, "readouts": readouts,
           "motor_total": motor_t.sum(axis=0).tolist(), "v_min": v_lo, "v_max": v_hi,
           "latch_onset_ms": [int(np.argmax(latch_t[:, c] > 0)) if latch_t[:, c].any() else None for c in CHANNELS]}
    if tonic_on:               # report-only (flag-ON): baseline firing over the drive window, diagnostics 3(a)/(b)
        dt_ms = float(bridge.core_config.dt_ms)
        out["tonic_rates_hz"] = {
            **{f"gpe_{c}": round(float(gpe_t[:T_DRIVE, c].sum()) / (float(config.n_gpe) * T_DRIVE * dt_ms) * 1000.0, 2)
               for c in CHANNELS},
            "selector_stn": round(float(stn_t[:T_DRIVE].sum()) / (float(config.n_stn) * T_DRIVE * dt_ms) * 1000.0, 2),
            "window": "drive", "drive_steps": int(T_DRIVE)}
    if commit is not None:                            # the commit read decides; the old latch read is reported only
        cm = bg_commit_read(commit, gate_t, slot_t, motor_t)
        cm.update({"old_latch_winner": winner, "old_latch_rates": latch_rates,
                   "old_latch_variants": {"x0.5": readouts["latch_rate_x0.5"], "x2": readouts["latch_rate_x2"]}})
        out.update({"winner": cm["winner"], "committed": cm["winner"] is not None, "commit": cm})
    return out


class NeuralSeamSelector:
    """The Gate-A v2 selector + the neural seam, built once per (seed, lesion) on a real SimulationBridge."""

    def __init__(self, seed: int = 42, lesion: str | None = None, commit: dict | None = None,
                 commit_lesion: str | None = None):
        self.seed = int(seed)
        self.config = selector_config("v2")
        self.lesion = lesion
        self.commit = dict(commit) if commit is not None else None      # a BG_COMMIT_GRID row (flag BRAIN_BG_COMMIT)
        self.commit_lesion = commit_lesion if commit is not None else None
        self._bridge = None
        self.last_diag = None           # RULING bg-gate-1101 report-only instrument of the latest trial

    def ensure_built(self):
        if self._bridge is not None:
            return
        regions, pathways, updates = neural_seam_parts(self.commit)
        b = build_selector_bridge(self.seed, self.config, extra_regions=regions, extra_pathways=pathways,
                                  core_config_updates=updates)
        b.set_transmission_gate(DIRECT_PATH_GATE, 0.0 if self.lesion == "direct_path" else 1.0)
        b.set_transmission_gate(VALUE_AFFERENT_GATE, 0.0 if self.lesion == "value" else 1.0)
        b.set_transmission_gate(LATCH_INPUT_GATE, 0.0 if self.lesion == "latch" else 1.0)
        if self.commit is not None:
            for les, gname in BG_COMMIT_LESION_GATES.items():
                b.set_transmission_gate(gname, 0.0 if self.commit_lesion == les else 1.0)
        _neural_baseline(b, self.config)
        _step(b, self.config.warmup_steps)
        if self.commit is not None:
            bg_commit_settle(b, self.config)      # the warmup can ignite a latch: start from the post-reset state
        self._bridge = b

    def select_once(self, msg: str) -> dict:
        self.ensure_built()
        diag = _TrialDiag(self._bridge)
        r = _run_neural_seam_trial(self._bridge, self.config, render_utterance(msg),
                                   arousal=self.lesion != "arousal", commit=self.commit, diag=diag)
        self.last_diag = diag.record(float(self._bridge.core_config.dt_ms))
        r["heard_groups"] = len(render_utterance(msg))
        if self.commit is not None:
            r["commit"].update({"op": self.commit.get("op"), "lesion": self.commit_lesion, "organ_seed": self.seed})
        return r


_NEURAL_ORGANS: dict = {}


def get_neural_organ(seed: int = 42, lesion: str | None = None, commit_op: int | None = None,
                     commit_lesion: str | None = None) -> NeuralSeamSelector:
    """`commit_op` None -> the neural-seam organ exactly as before (cache key (seed, lesion)); an int -> the same
    build plus the spiking commit stage of BG_COMMIT_GRID[commit_op] (its own cache entry)."""
    key = (int(seed), lesion) if commit_op is None else (int(seed), lesion, "commit", int(commit_op), commit_lesion)
    org = _NEURAL_ORGANS.get(key)
    if org is None:
        if commit_op is None:
            org = NeuralSeamSelector(seed=seed, lesion=lesion)
        else:
            org = NeuralSeamSelector(seed=seed, lesion=lesion, commit=bg_commit_row(commit_op),
                                     commit_lesion=commit_lesion)
        _NEURAL_ORGANS[key] = org
    return org


def decide_action_neural(msg: str, seed: int | None = None, lesion: str | None = None) -> dict | None:
    """The neural seam's replacement for `decide_action`. Same return contract: None -> the turn falls through to the
    normal path; a dict only when the neural commit is STAY-SILENT. Differences: the race runs on EVERY turn (no host
    admission), the host injects no candidate score, and the committed action is the latched one. A non-commit (no
    latch or both) falls through, as in production. The full trial record is attached as `trial` for tracing.
    `seed` None -> 42 (the organ seed the server has always used). With BRAIN_BG_COMMIT on (region `bg-commit`): the
    spiking commit stage decides, and a None seed follows BRAIN_CHAT_SEED (declared in the 2026-10-14 amendment)."""
    les = neural_seam_lesion() if lesion is None else lesion
    if bg_commit_enabled():
        seed = bg_commit_seed() if seed is None else seed
        org = get_neural_organ(seed=seed, lesion=les, commit_op=bg_commit_op(), commit_lesion=bg_commit_lesion())
    else:
        seed = 42 if seed is None else seed
        org = get_neural_organ(seed=seed, lesion=les)
    r = org.select_once(msg)
    decide_action_neural.last = r
    decide_action_neural.last_diag = getattr(org, "last_diag", None)   # RULING bg-gate-1101 report-only
    if not r["committed"] or r["winner"] != STAY_SILENT:
        return None
    step = (r["commit"] if "commit" in r else r)["latch_onset_ms"][STAY_SILENT]
    return {"action": ACTION_NAME[STAY_SILENT], "winner": int(r["winner"]), "speak_salience": None,
            "silent_salience": None, "decision_step": step, "loser_ratio": None,
            "lesion": les, "neural_seam": True, "trial": r}


decide_action_neural.last = None
decide_action_neural.last_diag = None
# END neural-seam


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BEGIN bg-commit (branch claude-build/bg-commit, 2026-10-14). DEFAULT-OFF; read only under BRAIN_BG_NEURAL_SEAM=1.
# The spiking commit stage (docs/drafts/spiking-commit/PLAN.md section 3; the one copy: research/runners/
# spiking_commit.py) on the BG channels, on the neural-seam organ's OWN bridge, replacing the fixed-rate latch read
# that failed G4 on 2026-10-02 (research/findings/2026-10-02-bg-live-seam-retirement-RESULT.md: the SPEAK latch of
# "what does the dog chase" held a graded 0.039 spikes/neuron/ms, so the x2 read dropped it). Worklog:
# docs/drafts/bg-commit/WORKLOG.md. Amendment: research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md.
#   sources X_k = motor_k (the selector's output pools, which already compete through the D1/GPi/thalamus race and the
#   commit_fs cross-inhibition); commit gate G_k (motor_k -> G_k, gate `bg_commit_in`, the COMMIT lesion), exclusive
#   across channels (GI_k -| G_j, gate `bg_commit_excl`, the EXCLUSION lesion); latch S_k (G_k -> S_k, S_k -> S_k,
#   gate `bg_commit_latch`, the LATCH lesion). The between-trial selector_reset also reaches G and S
#   (candidate-agnostic, like every other reset target).
#   READ: over the last READ_STEPS of the zero-stimulus hold (T_HOLD), S_k is ON iff its spike count reaches READ_FRAC
#   x NS x READ_STEPS; exactly one ON -> that action; none -> no commit (falls through, as before); both -> co-commit
#   (no commit, counted). Re-read at x0.5 and x2 on the same spikes. The old latch (latch_k) is still built and its
#   read is recorded from the same trial (`old_latch_winner`, reported, never used).
# RESIDUAL HOST STEPS (declared): the word rendering, the trial schedule (drive / hold / read window), the latch read
# (a threshold on the stage's latches whose invariance is measured, not assumed), and which env flag is on.
# LEARNED-CONTENT FRACTION: 0 (wired, as the neural seam is).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
from research.runners import spiking_commit as _SC  # noqa: E402

BG_COMMIT_FLAG = "BRAIN_BG_COMMIT"
BG_COMMIT_OP_FLAG = "BRAIN_BG_COMMIT_OP"
BG_COMMIT_LESION_FLAG = "BRAIN_BG_COMMIT_LESION"
BG_COMMIT_IN_GATE = "bg_commit_in"            # motor_k -> G_k (lesion `commit`)
BG_COMMIT_LATCH_GATE = "bg_commit_latch"      # S_k -> S_k (lesion `latch`)
BG_COMMIT_EXCL_GATE = "bg_commit_excl"        # GI_k -| G_j (lesion `excl`)
BG_COMMIT_LESION_GATES = {"commit": BG_COMMIT_IN_GATE, "latch": BG_COMMIT_LATCH_GATE, "excl": BG_COMMIT_EXCL_GATE}
BG_COMMIT_PREFIX = "bgc_"
BG_COMMIT_CARRY_STEPS = 5                     # latch spikes in a trial's first steps = a latch left on by the last one
# Rows. Row 0 is A1's COMMIT_GRID row 0 (word_role_parser_organ.py) unchanged, with W_XG = A1's W_RG and the
# selector's own reset weight onto G and S. A1's latches coexist (several roles are held at once); a BG trial is ONE
# decision whose race can switch channel within the 300 ms drive, so rows 1-4 add a single-slot exclusion: W_SGI (a
# latch keeps its GI pool on: the first commitment keeps the decision) or W_GIS (a bursting gate's GI silences the
# other latch: a later commitment replaces it), each with the A1 gate and a less sensitive one. DEV table (Claude
# build agent 2026-10-14, numpy, dev seeds 1-8, `_bg_commit_derisk.py`; raw research/findings/raw/_bg_commit/
# dev_organ; per row 320 intact trials, 288 lesion trials; gate seeds never run):
#   row  read invariant  latched min / unlatched max  empty held  content held  co-commit  agree old read  seeds all_pass
#   0    320/320         65 / 1                       77/112      0/104         96         0.738           3/8
#   1    320/320         65 / 1                       94/112      8/104         7          0.778           2/8
#   2    320/320         67 / 1                       95/112      8/104         6          0.781           2/8
#   3    320/320         65 / 1                       98/112      7/104         0          0.956           1/8
#   4    320/320         67 / 1                       98/112      7/104         0          0.963           1/8
#   (the old latch read on the same trials: empty 99/112, content 6/104; commit and latch lesions: 0 commits, every
#   row.) The pre-data rule (all_pass on all eight) selected NONE because C2/C3/C6 test the race: the old read fails
#   them on 5 of 8 seeds. Selected by the declared post-data rule (max agreement with the old read among rows whose
#   stage criteria hold on all eight): row 4. See docs/drafts/bg-commit/WORKLOG.md.
_BCR = dict(NG=12, NGI=8, NS=20, W_XG=50.0, W_GG=20.0, W_GGI=60.0, W_GIG=60.0, W_GS=12.0, W_SS=60.0, READ_STEPS=30,
            READ_FRAC=0.03, W_RESET=16.0)
BG_COMMIT_GRID = [
    dict(_BCR),                                  # 0: A1 row 0 (latches coexist)
    dict(_BCR, W_SGI=60.0),                      # 1: first commitment keeps the decision
    dict(_BCR, W_SGI=60.0, W_XG=30.0),           # 2: ... with a less sensitive gate
    dict(_BCR, W_GIS=60.0),                      # 3: a later commitment replaces an earlier one
    dict(_BCR, W_GIS=60.0, W_XG=30.0),           # 4: ... with a less sensitive gate
]
BG_COMMIT_OP = 4                      # the selected row (dev sweep + declared post-data rule; worklog)


def bg_commit_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_BG_COMMIT` in {1,true,yes,on} -> under BRAIN_BG_NEURAL_SEAM=1 the commit stage decides."""
    return (os.environ.get(BG_COMMIT_FLAG) or "").strip().lower() in ("1", "true", "yes", "on")


def bg_commit_op() -> int:
    """`BRAIN_BG_COMMIT_OP` (a BG_COMMIT_GRID index); unset or invalid -> BG_COMMIT_OP."""
    try:
        i = int((os.environ.get(BG_COMMIT_OP_FLAG) or "").strip())
    except ValueError:
        return BG_COMMIT_OP
    return i if 0 <= i < len(BG_COMMIT_GRID) else BG_COMMIT_OP


def bg_commit_lesion() -> str | None:
    """`BRAIN_BG_COMMIT_LESION` in {commit, latch, excl} (each closes one transmission gate); else None."""
    v = (os.environ.get(BG_COMMIT_LESION_FLAG) or "").strip().lower()
    return v if v in BG_COMMIT_LESION_GATES else None


def bg_commit_seed() -> int:
    """The organ seed under the flag: BRAIN_CHAT_SEED when it parses, else 42. (Flag off, the server always built the
    organ at 42, so the 10-02 gate's six chat seeds shared one organ build; declared in the amendment.)"""
    try:
        return int((os.environ.get("BRAIN_CHAT_SEED") or "").strip())
    except ValueError:
        return 42


def bg_commit_row(i: int) -> dict:
    i = int(i)
    if not (0 <= i < len(BG_COMMIT_GRID)):
        raise ValueError("BG commit operating point %d outside 0..%d" % (i, len(BG_COMMIT_GRID) - 1))
    return dict(BG_COMMIT_GRID[i], op=i)


def bg_commit_parts(c: dict):
    """(regions, pathways) of the stage on the motor pools, appended to the neural-seam build (same bridge)."""
    from sim.enums import NeuronType
    from sim.regions import RegionPathway
    from research.runners._vocal_action_selector_gate import _region
    regions, P = _SC.region_parts([f"motor_{ch}" for ch in CHANNELS], c, prefix=BG_COMMIT_PREFIX,
                                  gate_in=BG_COMMIT_IN_GATE, gate_excl=BG_COMMIT_EXCL_GATE,
                                  latch_loop=BG_COMMIT_LATCH_GATE, make_region=_region,
                                  exc_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL,
                                  inh_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON)
    for ch in CHANNELS:
        nm = _SC.region_names(BG_COMMIT_PREFIX, ch)
        for pop in ("gate", "slot"):
            P.append(RegionPathway(from_region="selector_reset", to_region=nm[pop], density=0.70,
                                   weight_mean=float(c["W_RESET"]), weight_jitter=0.0, plastic=False,
                                   receptor="gaba_a"))
    return regions, P


def bg_commit_settle(bridge, config):
    """After the build's warmup, the selector's own between-trial reset and washout (candidate-agnostic), so the first
    trial starts from the same state as every later one (dev: the warmup's thalamic transient ignited both latches)."""
    xp, _ = get_backend()
    _neural_baseline(bridge, config)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "selector_reset"))] = xp.float32(config.reset_pA)
    _step(bridge, int(config.reset_steps))
    _neural_baseline(bridge, config)
    _step(bridge, int(config.washout_steps))


def bg_commit_indices(bridge) -> dict:
    out = {"gate": [], "slot": []}
    for ch in CHANNELS:
        nm = _SC.region_names(BG_COMMIT_PREFIX, ch)
        out["gate"].append(_indices(bridge, nm["gate"]))
        out["slot"].append(_indices(bridge, nm["slot"]))
    return out


def bg_commit_read(c: dict, gate_t, slot_t, motor_t) -> dict:
    """The commit read of one trial from its recorded per-step gate / latch spikes (pure). The decision is
    `winner`; everything else is the record the scorer and the worklog read."""
    T = int(slot_t.shape[0])
    rs = int(c["READ_STEPS"])
    slot_counts = [int(x) for x in slot_t[T - rs:].sum(axis=0)]
    rd = _SC.read_with_variants(slot_counts, int(c["NS"]), rs, float(c["READ_FRAC"]))
    onset = [int(np.argmax(slot_t[:, ch] > 0)) if slot_t[:, ch].any() else None for ch in CHANNELS]
    return {"winner": rd["winner"], "on": rd["on"], "co_commit": bool(rd["co_commit"]), "threshold": rd["threshold"],
            "read_variants": rd["variants"], "read_invariant": bool(rd["read_invariant"]), "slot_counts": slot_counts,
            "gate_counts_drive": [int(x) for x in gate_t[:T_DRIVE].sum(axis=0)],
            "gate_counts_hold": [int(x) for x in gate_t[T_DRIVE:].sum(axis=0)],
            "motor_counts_hold": [int(x) for x in motor_t[T_DRIVE:].sum(axis=0)],
            "slot_carry": [int(x) for x in slot_t[:BG_COMMIT_CARRY_STEPS].sum(axis=0)],
            "latch_onset_ms": onset}
# END bg-commit
