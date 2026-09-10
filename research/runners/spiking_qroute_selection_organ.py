"""SPIKING 4-WAY LATERAL-INHIBITION WTA question-ROUTE SELECTION as a per-session organ — the PRODUCTION WIRE-IN of
the 6/6-seed-GO mechanism de-risk (`research/runners/_rank14_question_route_selection_derisk.py`, finding
`2026-09-09-rank14-question-route-selection-wta-derisk-GO.md`, biology binding
`research/biology/question-route-selection-wta.md`), behind a DEFAULT-OFF flag.

WHAT THIS RETIRES. `research/runners/brain_chat_tui.py`'s `ChatBrain._extract_route` decides, in host Python, WHICH of
four comprehension routes handles an incoming question by testing three route feature-extractors in a FIXED PRIORITY
ORDER and falling through only when none returned a candidate:

    _relf = self._relation_fronted_route(question)      # 'what country is chelsea fc from?'
    if _relf is not None: return _relf                  # RELFRONT
    _kbrel = self._kb_relation_question_route(question)  # 'where was X born?' (29-relation curated table)
    if _kbrel is not None: return _kbrel                # KBREL
    if len(content) <= 1:
        _defo = self._definitional_copula_route(question)  # 'what is X?'
        if _defo is not None: return _defo             # DEFCOP
    ... falls through to the already-neural generic SVO parse (GENERIC, the CHOOSE branch)

Per CLAUDE.md's brain-based-only standard the individual regex/shape TESTS are legitimate matched-filter reads of the
surface string (the same honesty class as any other host-side sensory feature extraction here) — what was NOT neural is
the COMBINATION: which construction wins WHEN MORE THAN ONE COULD APPLY, decided unconditionally by a fixed textual
`if`/`elif` order rather than by the STRENGTH of the recognized cue. The de-risk moved that DECISION onto the substrate
as four excitatory assemblies (RELFRONT, KBREL, DEFCOP, GENERIC), each with its own fast-spiking cross-inhibition
sub-pool, competing under mutual/reciprocal lateral inhibition (Grossberg on-center/off-surround; the Pinker-Ullman
default/elsewhere rule for the always-driven GENERIC baseline), and proved (6/6 seeds) it reproduces host priority on
the full battery (>=95%), resolves the deliberately ambiguous RELFRONT/KBREL overlap to RELFRONT via genuinely stronger
drive (not a host tie-break), collapses every item to GENERIC under a full exception lesion, and each pathway resolves
independently.

THE MECHANISM (reuse-by-import, NO `sim/` edit; VERBATIM the de-risk's circuit + drive mapping). This organ imports the
de-risk's OWN `evidence_to_currents` drive mapping + operating-point constants (`ROUTES`, `_IDX`, `OFF_PA`,
`GENERIC_BASELINE_PA`, `EXCEPTION_ON_PA`, `RELFRONT_PRIORITY_TILT`, `WARMUP/WASHOUT/RUN_STEPS`, `DEAD_MARGIN`) and the
already-GO'd N-pool cross-inhibition primitive (`_affect_marker_wta_derisk._build_bridge`/`_pool_rates`, N=4 here) so
the wired mechanism is byte-for-byte the de-risked one. A FRESH private SimulationBridge is built per DECISION (one
`select` call = one "which route handles this question?" decision), driven, read, and discarded — the same
fresh-substrate-per-decision discipline `SpikingAnaphorDetectorOrgan` uses. This is deliberate and load-bearing: the
de-risk's `evaluate_seed` reuses ONE bridge across its whole battery (a research convenience) and `_pool_rates` washes
out residual drive at the start of each read, but the 40-step washout does NOT fully reset the Izhikevich adaptation
(recovery variable) between reads, and the deliberately AMBIGUOUS RELFRONT/KBREL case — resolved by a small ~10% drive
tilt — has a margin tight enough that reuse across an arbitrary live question sequence can shrink it below `DEAD_MARGIN`
(measured 2026-09-09: the ambiguous case collapses from margin 0.14 fresh to 0.009 after four prior single-route reads
on the SAME bridge, independent of RNG isolation). A fresh, quiescent bridge per decision makes every route decision
independent and carryover-free — the correct live unit for "one decision per question" — at trivial cost (144 neurons,
one build + 160 steps per chat turn; speed is secondary to faithfulness per the mission). Determinism is preserved:
every build reseeds from `cfg.seed`, so the network is identical each decision.

WHAT IS ON THE SUBSTRATE (and load-bearing), AND WHAT IS A DECLARED HOST SHORTCUT. The DECISION — which route's evidence
wins when more than one construction could apply, and whether GENERIC is genuinely unopposed when none does — is on the
substrate (the argmax-with-dead-margin read off the settled `cp_firing_states` rates), and the lesion proves it (zeroing
all three exception pathways' evidence-driven gain routes EVERY question to GENERIC). The route feature EXTRACTION stays
host code — this organ never runs a regex: `select(relf_on, kbrel_on, defcop_on)` receives THREE BOOLEANS the caller
computed by running the pre-existing host extractors (`_relation_fronted_route`/`_kb_relation_question_route`/
`_definitional_copula_route`), exactly the residual the de-risk + biology binding already declare ("teaching the
substrate to recognize these constructions from corpus statistics rather than a curated table" — named there as out of
scope). So on every question the ON path recognises exactly the routes the host extractors recognise, but the DISPATCH
among them now lives on the neurons (retiring the host `if`/`elif`), with the substrate's drive-strength resolution of
the ambiguous overlap case the de-risked surpass.

CONTRACT (additive, reversible, DEFAULT-OFF). `BRAIN_SPIKING_QROUTE` truthy (1/true/on/yes) ARMS the spiking route
DISPATCH in `_extract_route`; UNSET or in {0,false,no,off,''} (the DEFAULT) leaves the pre-existing host `if`/`elif`
priority cascade UNCHANGED and this organ NEVER BUILT -> `_extract_route` is byte-identical to pre-wiring. The call site
also falls back to the exact host priority cascade on ANY organ error (never raises out). The flip to default-ON is a
SEPARATE step, gated on an integrated `/api/brain-chat` no-regression soak (this wire-in lands default-OFF; see the
finding).

LESION (the load-bearing proof). `BRAIN_SPIKING_QROUTE_LESION=1` forces every exception pathway's evidence-driven gain
to OFF (the de-risk's OWN G3 full-exception lesion, `lesion_relf=lesion_kbrel=lesion_defcop=True`): with only GENERIC's
baseline "elsewhere" drive above the OFF floor, EVERY question routes to GENERIC regardless of which regexes matched, so
the whole exception-route dispatch reverts to the substrate's default case.

RNG ISOLATION (the #77 footgun). Building the bridge reseeds the backend RNG (via `cfg.seed`); enabling this organ must
not perturb downstream RNG-dependent organs. Every substrate build + read runs inside `_isolated`, which snapshots the
host numpy + backend RNG, runs on this organ's OWN private continuous timeline, and restores the host RNG byte-untouched
(the same isolation `DaModeDrivesWorkspace`/`SpikingAnaphorDetectorOrgan` use). When the flag is OFF the organ is never
built, so byte-identity is unconditional.

FUNCTIONAL CORRELATE, NOT phenomenal. This reads + reports a spiking competitive-selection CORRELATE; it makes no claim
of subjective experience.
"""
from __future__ import annotations

import os
import threading

import numpy as np


def spiking_qroute_enabled() -> bool:
    """The master flag, DEFAULT-OFF. `BRAIN_SPIKING_QROUTE` truthy (1/true/on/yes) arms the spiking 4-way WTA route
    DISPATCH in `_extract_route`; UNSET (the default) or in {0,false,no,off,''} leaves the pre-existing host `if`/`elif`
    priority cascade unchanged and this organ never built -> `_extract_route` byte-identical to pre-wiring. Mirrors
    `spiking_anaphor_detection_organ.spiking_anaphor_enabled()`'s default-off semantics (a NEW retirement landing
    default-OFF; the flip to default-ON rides a separate integrated no-regression soak)."""
    return os.environ.get("BRAIN_SPIKING_QROUTE", "0").strip().lower() in ("1", "true", "on", "yes")


def spiking_qroute_lesioned() -> bool:
    """`BRAIN_SPIKING_QROUTE_LESION` truthy -> force every exception pathway's evidence-driven gain to OFF (the de-risk's
    G3 full-exception lesion): with only GENERIC's baseline drive above floor, EVERY question routes to GENERIC and the
    exception-route dispatch reverts to the default case. The load-bearing proof."""
    return os.environ.get("BRAIN_SPIKING_QROUTE_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


class SpikingQRouteSelectorOrgan:
    """A per-SESSION spiking 4-way lateral-inhibition WTA question-route selector (the route decision is
    per-conversation, exactly like the ChatBrain that owns it — so this is per-agent/per-ChatBrain, NOT a process
    singleton). `select(relf_on, kbrel_on, defcop_on)` maps the three host-extractor booleans to the de-risk's OWN drive
    currents, builds a FRESH private SimulationBridge (four excitatory route assemblies + four dedicated FSI
    cross-inhibition sub-pools, `_build_bridge` reused verbatim with n_pools=4), drives the marker pools, reads the
    settled per-pool rates, discards the bridge, and returns the winning ROUTE (or None on a dead-margin tie -> the
    caller then uses host priority order). Fresh-per-decision (not one reused bridge) because the Izhikevich adaptation
    does not fully wash out between reads and the tight ambiguous-case margin is sensitive to that carryover across an
    arbitrary live question sequence — see the module docstring's measured 0.14->0.009 collapse."""

    def __init__(self, seed: int = 42, lesion: bool = False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self._rng_state = None                # this organ's PRIVATE RNG timeline (host process-global RNG untouched)
        self._lock = threading.Lock()

    # ── RNG isolation (the #77 footgun; copied from SpikingAnaphorDetectorOrgan._isolated) ──────────────────────────
    def _isolated(self, fn):
        """Run `fn()` (a substrate build or spiking read) on this organ's PRIVATE RNG timeline, leaving the host
        process-global RNG (numpy + the sim backend) byte-untouched. Snapshot host RNG, swap in the organ's own
        continuous timeline, run, capture the advanced private timeline, restore host."""
        xp = None
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:
            xp = None
        host_np = np.random.get_state()
        host_xp = None
        if xp is not None and xp is not np:
            try:
                host_xp = xp.random.get_random_state().get_state()
            except Exception:
                host_xp = None
        if self._rng_state is None:
            np.random.seed(self.seed)
            if xp is not None and xp is not np:
                try:
                    xp.random.seed(self.seed)
                except Exception:
                    pass
        else:
            try:
                np.random.set_state(self._rng_state["np"])
            except Exception:
                pass
            if xp is not None and xp is not np and self._rng_state.get("xp") is not None:
                try:
                    xp.random.get_random_state().set_state(self._rng_state["xp"])
                except Exception:
                    pass
        try:
            return fn()
        finally:
            st = {"np": np.random.get_state(), "xp": None}
            if xp is not None and xp is not np:
                try:
                    st["xp"] = xp.random.get_random_state().get_state()
                except Exception:
                    st["xp"] = None
            self._rng_state = st
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _drive_for(self, relf_on: bool, kbrel_on: bool, defcop_on: bool):
        """The de-risk's OWN evidence->current mapping for the three booleans. GENERIC's evidence is always 1.0 (the
        elsewhere case is always structurally available -- not something a flag disables). `evidence_to_currents`
        applies the exact OFF/BASELINE/ON regime + RELFRONT's priority tilt (so the ambiguous RELFRONT/KBREL overlap
        resolves to RELFRONT via genuinely stronger drive, matching host priority); the lesion forces every exception
        pathway's gain OFF."""
        from research.runners._rank14_question_route_selection_derisk import evidence_to_currents
        ev = {"GENERIC": 1.0,
              "DEFCOP": 1.0 if defcop_on else 0.0,
              "RELFRONT": 1.0 if relf_on else 0.0,
              "KBREL": 1.0 if kbrel_on else 0.0}
        return evidence_to_currents(ev, lesion_relf=self.lesion, lesion_kbrel=self.lesion, lesion_defcop=self.lesion)

    def _fresh_rates(self, drive):
        """Build a FRESH quiescent cross-inhibition WTA bridge (n_pools=4), drive it with `drive`, read the settled
        per-pool rates, and discard the bridge -- all RNG-isolated. Fresh-per-decision so no residual adaptation from a
        prior read can perturb this decision (see the module docstring). Deferred imports (not module top) avoid the
        de-risk module's circular top-level `from ...brain_chat_tui import ...` at organ-import time and its
        SIM_BACKEND setdefault before the caller has chosen a backend."""
        from research.runners._affect_marker_wta_derisk import _build_bridge, _pool_rates
        from research.runners._rank14_question_route_selection_derisk import N_ROUTES, WARMUP_STEPS, WASHOUT_STEPS, RUN_STEPS

        def _run():
            bridge, marker_idx, _fsi_idx = _build_bridge(self.seed, N_ROUTES, "qroute")
            rates = _pool_rates(bridge, marker_idx, drive, warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
            del bridge
            return rates

        return self._isolated(_run)

    def select(self, relf_on: bool, kbrel_on: bool, defcop_on: bool):
        """Which of the four comprehension routes wins for a question whose host extractors returned candidates as
        indicated by the three booleans. GENERIC is always structurally available (its baseline "elsewhere" drive), so
        it wins whenever no exception pathway is genuinely driven. Returns the winning ROUTE name, or None on a
        dead-margin tie (the caller then uses host priority order). Builds a FRESH bridge per call, RNG-isolated; the
        wire-in wraps this in try/except and falls back to the host cascade on any error, so a substrate failure never
        changes the turn's contract."""
        from research.runners._rank14_question_route_selection_derisk import ROUTES, DEAD_MARGIN
        with self._lock:
            rates = self._fresh_rates(self._drive_for(relf_on, kbrel_on, defcop_on))
            order = np.argsort(rates)[::-1]
            top, second = int(order[0]), int(order[1])
            margin = float(rates[top] - rates[second])
            return ROUTES[top] if margin > DEAD_MARGIN else None

    def select_record(self, relf_on: bool, kbrel_on: bool, defcop_on: bool) -> dict:
        """Full record (rates + margin + winner) for verification/introspection; `select` returns just the winner."""
        from research.runners._rank14_question_route_selection_derisk import ROUTES, DEAD_MARGIN
        with self._lock:
            drive = self._drive_for(relf_on, kbrel_on, defcop_on)
            rates = self._fresh_rates(drive)
            order = np.argsort(rates)[::-1]
            top, second = int(order[0]), int(order[1])
            margin = float(rates[top] - rates[second])
            winner = ROUTES[top] if margin > DEAD_MARGIN else None
            return {"winner": winner, "margin": margin, "rates": rates.tolist(), "drive": drive.tolist(),
                    "routes": list(ROUTES), "lesioned": self.lesion,
                    "evidence": {"RELFRONT": bool(relf_on), "KBREL": bool(kbrel_on), "DEFCOP": bool(defcop_on)}}
