"""ACTIVITY-SILENT WORKING MEMORY wired as a maintenance-mode swap on the PRODUCTION anaphora referent store
(Mongillo/Barak/Tsodyks 2008 synaptic theory of WM; Stokes activity-silent WM). 2026-08-26.

THE FACULTY. The live anaphora store (`MultiTurnAgent.wm`, a `SpikingLoopContextBuffer`) holds the discourse focus in
a PERSISTENT-ACTIVITY attractor — it must keep FIRING to remember. Biology's alternative (Mongillo 2008) holds the item
in short-term synaptic FACILITATION (`cp_stp_u`) with the assembly SILENT, and reactivates it with a NONSPECIFIC ping
(a uniform pulse carrying no item identity). This organ is that alternative as a production maintenance mode: a
discourse referent is held across an intervening distractor turn in the facilitated recurrent synapses, delay genuinely
SILENT, and read back on the next referential turn by the ping.

REUSE-BY-IMPORT (NO `sim/` edit, nothing re-implemented). The whole spiking mechanism is the adversarially-verified
de-risk `research/runners/_activity_silent_wm_ping_derisk.py` -> the 6/6 GO
`2026-08-10-parallel-push-results-activity-silent-WM-GO-...md`: K isolated excitatory assemblies with within-assembly
recurrent E->E (STP ON, `stp_tau_f=1500`, `stp_tau_d=200`, `w_rec=60` sub-self-sustaining so the delay is SILENT),
`ActivitySilentWM.load(k)` -> `.delay()` -> `.ping()`. This organ only BINDS discourse-referent strings to assemblies
and surfaces the reactivation as an honest read-out. The de-risk's constants (W_REC/STP_U/TAU_F/TAU_D/DELAY_STEPS) are
imported and NOT re-tuned.

WEIGHT-REGIME PIN (the named RISK — else the "silent" claim leaks). The persistent-attractor path is suppressed by the
de-risk's `w_rec=60` (sub-self-sustaining: the assembly cannot hold itself firing without drive, so the delay falls
silent — verified `delay_firing < 0.01`). The lesion is the FAIR, excitability-MATCHED control: STP stays ON (the u*x
multiplier and thus net excitability are identical) but `stp_tau_f` collapses to ~5 ms so facilitation cannot BRIDGE the
delay. We do NOT use `enable_short_term_plasticity=False` as the lesion — that removes the u*x multiplier, jumps
effective recurrence to full weight, and turns the net into a Wang-2002 PERSISTENT-FIRING attractor (delay no longer
silent). The organ reports `silent_delay` on every recall so a test asserts the hold really was activity-silent.

WHAT IT DOES IN A TURN (additive, moat-safe, honest — mirrors is_hold_query / is_expectation_query):
  * MAINTAIN (write-only side effect): a turn that names a discourse referent -> bind it to a stable assembly and set
    it as the silently-held FOCUS (facilitation regime); an intervening turn with no new referent -> note a distractor
    (the silent delay grows). Neither changes the reply -> byte-identical.
  * READ-OUT (disjoint short-circuit): an explicit "what were we ORIGINALLY / FIRST talking about / go back to before"
    temporal-recall query -> reactivate the silently-held focus via the NONSPECIFIC ping and answer with an honest
    functional read-out. Reactivation is DECISION-GATED on the ping-window margin (a no-confab gate): the focus is
    surfaced only when the facilitated assembly reactivates DECISIVELY (margin > MARGIN_MIN); otherwise the organ
    ABSTAINS ("I don't recall ...") rather than manufacture a referent. This trigger class is DISJOINT from the D6
    hold-query ("who/what are we talking about / keeping in mind" — the CURRENTLY-held set) — D6 reports the live
    multi-referent buffer, this reports a SINGLE focus recovered from silence across a distractor.

LESION-LOAD-BEARING (`BRAIN_SILENT_WM_LESION=1`): recall builds the buffer with `tau_f`~5 (facilitation lesion). The
facilitation then decays away during the delay -> at ping time the focus assembly is no longer specially reverberant ->
the margin collapses below MARGIN_MIN -> the read-out ABSTAINS. The host referent PARSE and the assembly BINDING are
byte-identical with/without the lesion, so the discrimination (correct anaphor vs abstain) is caused by the SILENT
FACILITATED HOLD, not the host bookkeeping. This is the de-risk's own oracle: the ping recovers the item with the
facilitated buffer but NOT with tau_f collapsed.

HONEST RESIDUALS (declared):
  * Capacity is the de-risk's K=4 assemblies (chance 1/4). A 5th+ distinct referent collides on the last assembly (a
    binder-cap collision, not a WM-time limit) — the same vocab/capacity ceiling class D6 declares.
  * The referent EXTRACTION + the recall-query TRIGGER are a host parse (reuse D6's `extract_referents` lexicon + a
    small temporal-recall regex) — the declared vocab-ceiling residual; a learned referent/query detector is the next
    rung.
  * STP-based WM is genuinely TIME-LIMITED (`tau_f`~1.5 s): it bridges a distractor turn or two, not an arbitrary span
    (the de-risk's declared honest wall). A durable hold is the persistent store / LTM, not this mode.
  * CO-RESIDENT: the buffer runs on ITS OWN `ActivitySilentWM` bridge alongside the recall composer (rides the
    one-brain merge), exactly as the affect/comprehension/D6 organs do.

Additive, DEFAULT-OFF (`BRAIN_SILENT_WM` unset/0 -> the block imports nothing + returns nothing -> byte-identical; the
parent flips default-on after the pool soak). Uses the process backend (cupy in production, numpy in tests) via
reuse-by-import.
"""
from __future__ import annotations

import os
import re

import numpy as np

# --- the de-risked activity-silent-WM spiking core + its PINNED constants (imported, NOT re-tuned) ---
from research.runners._activity_silent_wm_ping_derisk import (
    ActivitySilentWM,
    K as _K,                     # assemblies (chance = 1/K)
    W_REC as _W_REC,             # sub-self-sustaining recurrent weight -> SILENT delay (the persistent-path pin)
    STP_U as _STP_U,
    TAU_F as _TAU_F_INTACT,      # 1500 ms Mongillo augmentation time constant (the silent hold lives here)
    TAU_F_LESION as _TAU_F_LESION,  # ~5 ms: the FAIR facilitation lesion (excitability matched, hold can't bridge)
    DELAY_STEPS as _DELAY_STEPS,
    PING_DRIVE as _PING_DRIVE,
    PING_STEPS as _PING_STEPS,
)
# --- reuse the D6 host referent lexicon/extractor (the declared vocab-ceiling residual; nothing new to maintain) ---
from research.runners.d6_multiref_wm_production_organ import extract_referents as _extract_refs

# The ping-window firing MARGIN (loaded-assembly minus mean of the others) above which a reactivation is DECISIVE
# enough to surface (a no-confab gate). The de-risk's intact margin is +12.6..+19.7 (40-trial); the FAIR-lesion margin
# regresses to ~0 (the focus is not specially facilitated). Averaging over RECALL_TRIALS micro-seed builds regresses a
# lesion seed's structural-favorite coincidence toward 0 while the intact margin stays high; 7.0 sits cleanly in the
# empirically-verified gap (intact per-seed ensemble >=8.7, lesion per-seed ensemble <=5.7 single-trial -> ~0 ensembled).
MARGIN_MIN = 7.0
# Read the buffer as an ENSEMBLE of RECALL_TRIALS load->silent-delay->ping builds (different heterogeneity micro-seeds).
# The decision statistic is the ENSEMBLE-MEAN margin (a single spiking read is noisy; the mean is what the de-risk
# reports). This makes the lesion robustly abstain: its structural-favorite coincidence varies across micro-seeds so
# the mean margin regresses toward ~0, while the facilitated intact mean stays high.
RECALL_TRIALS = 7

# A DISJOINT temporal-recall query: "what did we start with", "the original topic", "go back to the beginning", "the
# topic from before". It uses ONLY temporal-distance phrasings and DELIBERATELY AVOIDS every lexeme in the D6
# hold-query regex ("talking about", "discussing", "referring to", "keeping in mind", "holding", "remember",
# "tracking", "referents") — so the two query classes are strictly disjoint and D6 (which runs first) never pre-empts a
# silent-WM recall. D6 reports the CURRENTLY-held multi-referent set; this reports a SINGLE focus recovered from silence
# across a distractor. (The trigger lexicon is the declared host-parse residual, the same class D6/comprehension carry.)
_SILENT_RECALL_RE = re.compile(
    r"\b("
    r"what did (we|i|you) (start|begin) (with|out|on)"                 # what did we start with
    r"|what (was|were) (the|we|our) (first|original) (topic|subject|thing)"  # what was the first/original topic
    r"|(the )?original (topic|subject|thing)"                          # the original topic
    r"|first (topic|subject|thing) (we|i|you)"                         # first topic we (had)
    r"|(go|going|get) back to (the )?(start|beginning|first|original)"  # go back to the beginning
    r"|before (the distraction|we changed|that came up|the tangent)"   # before we changed
    r"|(the )?(earlier|previous|original) (topic|subject)"             # the earlier topic
    r"|(the )?(topic|subject|thing) from before"                       # the topic from before
    r")\b",
    re.IGNORECASE,
)


def silent_wm_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_SILENT_WM` in {1,true,yes,on} -> the organ runs; unset/0/false/no/off -> byte-identical."""
    v = os.environ.get("BRAIN_SILENT_WM")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def silent_wm_lesioned() -> bool:
    """`BRAIN_SILENT_WM_LESION` in {1,true,yes,on} -> recall builds the buffer with tau_f~5 (facilitation lesion)."""
    v = os.environ.get("BRAIN_SILENT_WM_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def is_silent_recall_query(text: str) -> bool:
    """An explicit temporal 'what were we ORIGINALLY / FIRST talking about / go back to before' recall query. DISJOINT
    from the D6 hold-query (which reports the CURRENTLY-held set with no temporal-distance marker)."""
    return bool(_SILENT_RECALL_RE.search(text or ""))


def silent_recall_readout(referent) -> str:
    """An honest functional read-out of what the NONSPECIFIC ping reactivated from the silently-held facilitation (never
    a phenomenal claim). Abstains rather than confabulate when the ping did not decisively reactivate a focus."""
    if not referent:
        return "I don't recall what we were discussing before that."
    return f"Going back — earlier we were talking about the {referent}."


class ActivitySilentWMOrgan:
    """A per-session activity-silent discourse buffer: binds referent strings to the de-risk's K assemblies, holds the
    FOCUS in short-term facilitation across an intervening distractor, and reactivates it with a nonspecific ping. The
    spiking load->silent-delay->ping is the de-risk's `ActivitySilentWM`; this class only does the host bookkeeping
    (which referent -> which assembly, how long the silent delay ran) and the honest read-out."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._assembly_of_ref: dict[str, int] = {}   # referent string -> stable assembly index 0.._K-1
        self._ref_of_assembly: dict[int, str] = {}
        self._next = 0
        self._focus: str | None = None               # the last-introduced referent (silently-held anaphora focus)
        self._distractors_since = 0                  # intervening turns since the focus was written (silent-delay len)

    # --- host bookkeeping -------------------------------------------------
    def _bind(self, ref: str) -> int:
        if ref in self._assembly_of_ref:
            return self._assembly_of_ref[ref]
        if self._next >= _K:
            a = _K - 1                               # capacity ceiling (declared residual): collide on the last assembly
        else:
            a = self._next
            self._next += 1
        self._assembly_of_ref[ref] = a
        self._ref_of_assembly[a] = ref
        return a

    def write_referent(self, ref: str) -> int:
        """MAINTAIN: a turn introduced discourse referent `ref` -> bind it to a stable assembly, set it as the
        silently-held FOCUS, reset the intervening-delay counter. Pure side effect (does not change any reply)."""
        a = self._bind(ref)
        self._focus = ref
        self._distractors_since = 0
        return a

    def note_distractor(self) -> None:
        """An intervening turn with NO new referent -> the silent delay grows (facilitation decays with tau_f)."""
        if self._focus is not None:
            self._distractors_since += 1

    # --- the spiking reactivation (reuse-by-import) -----------------------
    def _recall_trial(self, assembly: int, n_delays: int, lesion: bool, trial: int):
        """One load->silent-delay(s)->ping on a fresh `ActivitySilentWM` (tau_f=intact/lesion). Returns
        (recovered_assembly, delay_firing, margin). The recurrent-weight/STP regime is the de-risk's, imported."""
        tau_f = _TAU_F_LESION if lesion else _TAU_F_INTACT
        wm = ActivitySilentWM(self.seed * 100 + trial, stp_on=True, w_rec=_W_REC, stp_u=_STP_U, tau_f=tau_f,
                              delay_steps=_DELAY_STEPS, ping_drive=_PING_DRIVE, ping_steps=_PING_STEPS)
        wm.load(assembly)
        df = 0.0
        for _ in range(max(1, n_delays)):
            df = wm.delay()                          # SILENT delay (verified ~0); repeat per intervening distractor turn
        counts = wm.ping()                           # NONSPECIFIC uniform ping -> per-assembly reactivation
        rec_a = int(np.argmax(counts))
        others = float(np.mean([counts[k] for k in range(_K) if k != assembly]))
        margin = float(counts[assembly] - others)
        return rec_a, float(df), margin

    def recall_focus(self, lesion: bool = False, n_trials: int = RECALL_TRIALS) -> dict | None:
        """Reactivate the silently-held FOCUS. Builds an ENSEMBLE of `n_trials` fresh buffers, each LOADing the focus
        assembly, running the (distractor) silent delay, and pinging. The focus is RECOVERED only when a MAJORITY of
        the trials place the ping-window argmax on the focus AND with a DECISIVE margin (> MARGIN_MIN) — a no-confab
        gate; otherwise the organ abstains (recovered=None). Ensembling regresses the FAIR-lesion's per-seed
        structural-favorite coincidence toward chance, so the lesion robustly abstains (the de-risk oracle)."""
        if self._focus is None:
            return None
        a_focus = self._assembly_of_ref[self._focus]
        n_delays = max(1, self._distractors_since)   # at least one silent interval (the intervening turn)
        margins, silent_ok, argmax_focus = [], [], []
        for t in range(n_trials):
            rec_a, df, margin = self._recall_trial(a_focus, n_delays, lesion, t)
            margins.append(margin)
            silent_ok.append(df < 0.01)
            argmax_focus.append(rec_a == a_focus)
        margin_mean = float(np.mean(margins))
        frac_argmax_focus = float(np.mean(argmax_focus))
        # DECISIVE iff the focus assembly is the ensemble's MAJORITY winner AND its ensemble-mean ping-window margin
        # clears MARGIN_MIN (the facilitation-driven boost). The FAIR lesion clears neither robustly -> abstain.
        decisive = bool(frac_argmax_focus > 0.5 and margin_mean > MARGIN_MIN)
        recovered = self._focus if decisive else None
        return {
            "on": True, "lesioned": bool(lesion), "focus": self._focus,
            "recovered": recovered,                  # the referent the ping reactivated (None = abstain, no confab)
            "recovered_is_focus": decisive,
            "margin_mean": margin_mean,
            "frac_argmax_focus": frac_argmax_focus,
            "reactivation_acc": frac_argmax_focus,   # fraction of ensemble trials whose argmax landed on the focus
            "silent_delay": bool(np.mean(silent_ok) > 0.5),       # the hold was genuinely activity-silent
            "n_delays": int(n_delays), "n_held": len(self._assembly_of_ref),
        }

    # --- production entry -------------------------------------------------
    def judge(self, text: str, lesion: bool = False) -> dict | None:
        """Production entry for the referential RECALL turn. Returns None when OUT OF SCOPE (not a temporal-recall
        query, or nothing held) -> the caller leaves the turn byte-identical. Otherwise reactivates the silently-held
        focus via the ping and returns an honest functional read-out."""
        if not is_silent_recall_query(text):
            return None
        if self._focus is None:
            return None
        rec = self.recall_focus(lesion=lesion)
        if rec is None:
            return None
        out = dict(rec, in_scope=True, is_silent_recall=True, composer="onebrain")
        out["readout"] = silent_recall_readout(rec["recovered"])
        return out


_ORGAN: ActivitySilentWMOrgan | None = None


def get_organ(seed: int = 42) -> ActivitySilentWMOrgan:
    """The process-shared activity-silent-WM organ (built once on first use). The live wiring keeps a PER-SESSION organ
    instead (a process singleton would leak one conversation's focus into another's recall), mirroring D6."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = ActivitySilentWMOrgan(seed=seed)
    return _ORGAN
