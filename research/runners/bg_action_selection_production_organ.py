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
)
from sim.backend import get_backend, to_host
# reuse-by-import the shared spiking novelty/salience afferent (scaffold-retirement backlog rank-4, 2026-09-05,
# research/runners/shared_salience_afferent.py) -- BRAIN_SHARED_SALIENCE, default-ON since 2026-09-05 (Track-1
# flip); see that module's own docstring for the current default and salience()'s docstring below for this site.
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

    SHARED SPIKING AFFERENT (rank-4, `BRAIN_SHARED_SALIENCE`, research/runners/shared_salience_afferent.py;
    default-ON since 2026-09-05's Track-1 flip -- see that module for the current default and the escape hatch).
    The ENTRY GATE (whether the turn even carries a content token — the environment's crude "is anything here" read)
    stays the SAME host boolean (`n == 0`); what changes is the SALIENCE MAGNITUDE the composer hands the selector on a
    content-empty turn: instead of the hardcoded (0.0, 1.0) pair, `speak` is the shared curiosity-organ ASK-pool's
    spiking transduction of the SAME raw content-count scalar (`min(1, n/2)`), and `silent = 1 - speak`. So the exact
    bias the striatal D1 race receives at the ONE reachable STAY-SILENT-candidate point is now a genuine spiking read
    (mediated by the shared ASK-pool population), not two bare host formulas — and it collapses toward the SAME
    baseline as the DA-mode/value-choice consumers under the SAME `BRAIN_SHARED_SALIENCE_LESION` lesion. OFF
    (`BRAIN_SHARED_SALIENCE` explicitly `{0,false,no,off,''}`, the byte-identical escape post-flip) -> byte-identical
    to the bare host formula below."""
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
