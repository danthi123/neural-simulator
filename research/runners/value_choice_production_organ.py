"""Value-driven choice production organ — the RANK-1 value-critic GO wired into the live chat gate.

FACULTY (the owner's "make the brain COMMIT [by value] instead of abstaining/guessing"): when >=2 stored facts share
the same (agent, action) with DIFFERENT patients, today's live GNW chain resolves the ambiguity by an ARBITRARY
FIRST-MATCH (verified live on the onebrain composer) or by a halt-if-unsure abstain. This organ replaces that with a
VALUE-DRIVEN CHOICE: each candidate patient is scored by the brain's OWN LEARNED spiking value (the striosome_value
critic, grown by DA-gated STDP), and a spiking value-WTA (Wang-2002 biased competition) COMMITS the higher-value
patient. Spiking value + spiking decision — no host argmax in the decision path. On decline (lesion / non-decisive
value) the gate wrapper returns the INNER pipeline result verbatim, so the turn reverts to exactly what the chain
would have done (the first-match, or the abstain).

GO SOURCE: research/findings/2026-07-23-value-critic-closure-RANK1-GO.md (6/6 seeds, all anti-cheats) via the runner
research/runners/_navcloseout_R5b_learned_value_choice.py. That GO proved: the learned spiking V drives the choice
(headline), the value LESION collapses it (G_LESION), and the UNTRAINED critic has no advantage (G_UNTRAINED — the
substrate's LEARNING is load-bearing, not a wired prior).

REUSE-BY-IMPORT (NO sim/ edit — the organ is a thin adapter over two validated organs):
  - the LEARNED spiking VALUE: research.runners._merged_navcritic_valuetrain (build_merged, run_value_train,
    _critic_rate_via_afferent, _vs_place_prefs, _idx, GOAL, _far_of, _mean_afferent_weight, GIRK_CAP).
  - the spiking DECISION:      research.runners._navcloseout_R5_value_driven_choice (SpikingValueChoice, _drives,
    make_salience_bias).

THE WIRE-IN ADAPTER (the honest residual, identical in kind to the GO's own construction): the GO reads V at nav
PLACE cues along the goal->far diagonal (near = high learned V). Here each candidate patient carries an ENGAGEMENT/
reward CONTEXT scalar e in [0,1] (the "prior reward/engagement/DA context" the sketch names — supplied by the
ChatBrain: fact recency + the discourse-WM referent). e is mapped to a cue position pos = near + (1-e)*(far-near),
so a MORE-engaged candidate sits nearer the goal and the LEARNED critic reads a HIGHER V for it. The critic's
learned V (a real cp_firing_states read) + the spiking WTA then do the work — and the G_LESION / G_UNTRAINED
anti-cheats prove the LEARNED SPIKING VALUE is load-bearing (host engagement ordering ALONE, without the trained
critic, does NOT produce the commit: lesion -> abstain; untrained -> the engagement advantage vanishes).

FLAGS (contract mirrors the other Gate-B production organs; 2026-08-26 FLIPPED DEFAULT-ON after the 6-seed pool soak
passed — ordinary byte-identical 6/6, load-bearing 6/6; BRAIN_VALUE_CHOICE=0 is the byte-identical escape to the
pre-flip abstain/first-match oracle):
  BRAIN_VALUE_CHOICE            unset -> ACTIVE (the production default). {0,false,no,off,''} (explicit) -> OFF
                                (byte-identical escape). {1,true,yes,on} -> ACTIVE (explicit, redundant now).
  BRAIN_VALUE_CHOICE_LESION     in {1,true,yes,on} -> G_LESION: pin each candidate's learned V to the MEAN -> the
                                value gradient vanishes -> the organ DECLINES -> the turn reverts to abstain.
  BRAIN_VALUE_CHOICE_UNTRAINED  in {1,true,yes,on} -> G_UNTRAINED: score with the UNTRAINED critic (no value-train)
                                -> the trained engagement-advantage vanishes (proves the LEARNING is load-bearing).
"""
from __future__ import annotations

import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# reuse-by-import: the LEARNED spiking value critic (the merged one-brain bridge + DA-gated STDP value-train)
from research.runners import _merged_navcritic_valuetrain as VT  # noqa: E402
# reuse-by-import: the spiking value-WTA DECISION organ (a neural pool's firing) + the drive builder + salience
from research.runners._navcloseout_R5_value_driven_choice import (  # noqa: E402
    SpikingValueChoice, _drives, make_salience_bias,
)
# reuse-by-import the shared spiking novelty/salience afferent (scaffold-retirement backlog rank-4, 2026-09-05,
# research/runners/shared_salience_afferent.py) -- BRAIN_SHARED_SALIENCE, default-ON since 2026-09-05 (Track-1
# flip, rank-20 verification); see default_context_fn() below and that module's own docstring.
import research.runners.shared_salience_afferent as _SHARED  # noqa: E402

# whose-the-difference attribution (the R5 non-circularity question, asked per COMMIT): what fraction of the winning
# pool's DRIVE is the LEARNED VALUE vs the value-INDEPENDENT salience baseline? A choice that is a relabeled salience
# attributes ~0. Guarded so the production import path never breaks on a missing tools.lab.
try:
    from tools.lab import attributable_to  # noqa: E402
except Exception:  # pragma: no cover - production fallback
    def attributable_to(label, treatment_value, control_value, warn_below=0.5):
        t = float(treatment_value)
        return None if abs(t) < 1e-12 else (t - float(control_value)) / t

# ── drive mapping (VERBATIM the R5b closure constants; a FIXED pA-per-Hz gain applied identically to every read so
#    it cannot smuggle the answer — an untrained/flat critic produces flat/anti drives regardless of the gain) ──
VALUE_HZ_GAIN_PA = 5.0      # pA per Hz of learned critic firing (R5b VALUE_HZ_GAIN_PA)
SPEAK_BASE_PA = 70.0        # R5b option-pool base drive
SALIENCE_GAIN_PA = 40.0     # R5b salience -> drift gain (the value-INDEPENDENT 'default pull'; the lesion baseline)
DEFAULT_VALUE_TRAIN_TRIALS = 40   # the R5b GO value-train length
# The decisiveness read-out (a documented scalar read of the brain's learned value, like an argmax over spike
# counts): commit only when the LEARNED V carries a real gradient across the candidates. Under the mean-pin lesion
# the fed value is FLAT (spread == 0) -> decline -> the turn reverts to abstain (the surface change VANISHES).
DEFAULT_V_MARGIN_HZ = 2.0


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _falsy_explicit(v: str | None) -> bool:
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


# 2026-08-26 FLIPPED DEFAULT-ON (wave 1/2 flip, 6-seed pool soak GO: ordinary byte-identical 6/6, load-bearing 6/6).
# The production-integration anchor.
_VALUE_CHOICE_DEFAULT_ON = True


def value_choice_enabled() -> bool:
    """DEFAULT-ON anchor (post wave-1/2 flip; 6-seed pool soak GO). Unset -> `_VALUE_CHOICE_DEFAULT_ON` (True).
    `BRAIN_VALUE_CHOICE` in {0,false,no,off,''} (explicitly set) is the byte-identical escape back to today's
    pre-flip oracle; any other explicit value stays ON."""
    v = os.environ.get("BRAIN_VALUE_CHOICE")
    if _VALUE_CHOICE_DEFAULT_ON:
        return not _falsy_explicit(v)
    return _truthy(v)


def value_choice_lesioned() -> bool:
    """`BRAIN_VALUE_CHOICE_LESION` -> G_LESION: pin each candidate's learned V to the MEAN (remove the gradient)."""
    return _truthy(os.environ.get("BRAIN_VALUE_CHOICE_LESION"))


def value_choice_untrained() -> bool:
    """`BRAIN_VALUE_CHOICE_UNTRAINED` -> G_UNTRAINED: score with the UNTRAINED critic (no value-train)."""
    return _truthy(os.environ.get("BRAIN_VALUE_CHOICE_UNTRAINED"))


class ValueChoiceProductionOrgan:
    """Builds (once) the LEARNED spiking striosome_value critic + the spiking value-WTA, then scores candidate
    patients by their learned V and commits the higher-V one. `untrained=True` skips the value-train (the
    G_UNTRAINED control -> no learned gradient)."""

    def __init__(self, seed: int = 42, value_train_trials: int = DEFAULT_VALUE_TRAIN_TRIALS,
                 untrained: bool = False, v_margin_hz: float = DEFAULT_V_MARGIN_HZ,
                 acc_steps: int = 120, verbose: bool = False):
        self.seed = int(seed)
        self.value_train_trials = int(value_train_trials)
        self.untrained = bool(untrained)
        self.v_margin_hz = float(v_margin_hz)
        self.acc_steps = int(acc_steps)
        self.verbose = bool(verbose)
        self.attribute = False   # verify/soak set True to bank the per-commit value-vs-salience attribution (it PRINTS)
        self._built = False
        self._lock = threading.Lock()
        self._bridge = None
        self._cfg = None
        self._idx = None
        self._prefs = None
        self._xp = None
        self._wta = None
        self.value_train = None
        self.near = None
        self.far = None

    def ensure_built(self):
        """Build the merged one-brain bridge, (unless untrained) value-train the striosome_value critic, and build
        the tiny spiking value-WTA. Idempotent + thread-safe (the webapp prewarm + first turn can race)."""
        if self._built:
            return self
        with self._lock:
            if self._built:
                return self
            from sim.backend import get_backend
            xp, _backend = get_backend()
            self._xp = xp
            b, _h = VT.build_merged(self.seed, convergent_upstate=True)
            cfg = b.core_config
            cfg.gabab_conductance_max = float(VT.GIRK_CAP)
            cfg.enable_ou_process = True
            cfg.ou_std_current_pA = 100.0
            cfg.homeostasis_threshold_adapt_rate = 0.0
            # FREEZE the value arm for every read (mirrors R5b.run_seed): the merged default reward_learning_rate +
            # value_input open would grow the weight during a read.
            _vt_lr = float(cfg.reward_learning_rate) if (cfg.reward_learning_rate and cfg.reward_learning_rate > 0) \
                else 0.01
            cfg.reward_learning_rate = 0.0
            cfg.current_reward_signal = 0.0
            b.set_plasticity_gate("value_input", 0.0)

            idx = {nm: VT._idx(b, nm, xp) for nm in
                   (VT.SNC, VT.CRITIC, VT.REWARD_US, VT.CRITIC_AFFERENT, VT.UPSTATE_AFFERENT)}
            prefs = VT._vs_place_prefs(int(len(VT._host(idx[VT.CRITIC_AFFERENT]))))
            self.near = tuple(float(x) for x in VT.GOAL)
            self.far = tuple(float(x) for x in VT._far_of(VT.GOAL))

            if not self.untrained:
                # VALUE-TRAIN: grow vs_place_context->striosome_value via pair-then-reward DA-gated STDP (learn V).
                cfg.reward_learning_rate = _vt_lr
                self.value_train = VT.run_value_train(
                    b, idx, prefs, xp, cfg, near=self.near, far=self.far,
                    trials=self.value_train_trials, verbose=self.verbose)
                cfg.reward_learning_rate = 0.0
                cfg.current_reward_signal = 0.0
                b.set_plasticity_gate("value_input", 0.0)

            self._bridge = b
            self._cfg = cfg
            self._idx = idx
            self._prefs = prefs
            # the spiking value-WTA decision organ (n_options set at choose-time via a per-arity cache)
            self._wta_cache: dict[int, SpikingValueChoice] = {}
            self._built = True
        return self

    def _wta_for(self, n_options: int) -> SpikingValueChoice:
        w = self._wta_cache.get(n_options)
        if w is None:
            w = SpikingValueChoice(seed=12345, n_options=int(n_options), n_steps=self.acc_steps)
            self._wta_cache[n_options] = w
        return w

    def read_values(self, engagements) -> list[float]:
        """Map each candidate's engagement e in [0,1] to a cue on the goal->far diagonal (higher e -> nearer goal ->
        higher learned V) and read the critic firing rate (Hz) = V. A frozen, deployment-faithful read (no direct
        critic drive, no teacher, weights frozen)."""
        self.ensure_built()
        near = np.asarray(self.near, dtype=float)
        far = np.asarray(self.far, dtype=float)
        vals = []
        for e in engagements:
            t = 1.0 - float(max(0.0, min(1.0, e)))       # e=1 -> t=0 -> at the goal (highest learned V)
            gx, gy = (near + t * (far - near)).tolist()
            v = float(VT._critic_rate_via_afferent(self._bridge, self._idx, self._prefs, gx, gy, self._xp, self._cfg))
            vals.append(v)
        return vals

    def choose(self, candidates, engagements, *, lesion: bool = False, salience_seed: int = 0):
        """Score the candidate patients by their LEARNED spiking V (mapped from engagement) and COMMIT the winner via
        the spiking value-WTA. Returns (chosen_patient or None, meta). Returns None (decline -> the caller keeps the
        abstain) when the learned value carries no decisive gradient (the mean-pin lesion collapses to this)."""
        cands = list(candidates)
        n = len(cands)
        if n < 2:
            return None, {"reason": "fewer than 2 candidates", "n": n}
        V = self.read_values(engagements)                  # the learned spiking V per candidate (Hz)
        Vv = np.asarray(V, dtype=float)
        # the value AS FED to the WTA drift: intact = per-candidate V; lesion = the MEAN (gradient removed, op-point
        # held) — the R5b G_LESION ablation. The decisiveness read is the spread of the FED value.
        fed = np.full(n, float(Vv.mean())) if lesion else Vv
        spread = float(fed.max() - fed.min())
        salience = make_salience_bias(n, seed=int(salience_seed))
        drives = _drives(Vv, salience, speak_base_pA=SPEAK_BASE_PA, value_gain_pA=VALUE_HZ_GAIN_PA,
                         salience_gain_pA=SALIENCE_GAIN_PA, lesion_value=bool(lesion))
        choice, counts, margin = self._wta_for(n).decide(drives)
        decisive = spread >= self.v_margin_hz
        # WHOSE the difference: of the WINNING pool's total drive, what fraction is the LEARNED VALUE (treatment) vs
        # the value-INDEPENDENT base+salience baseline (control)? attributable_to = (drive - baseline) / drive. A
        # commit that is a relabeled salience attributes ~0 -> the choice is NOT the learned value's (non-circular).
        # Guarded behind `self.attribute` (default OFF) because attributable_to PRINTS -- the hot per-turn production
        # path stays quiet; the verify/soak set organ.attribute=True to bank the non-circularity read.
        w = int(choice)
        drive_win = float(drives[w])
        baseline_win = float(SPEAK_BASE_PA) + float(SALIENCE_GAIN_PA) * float(salience[w])
        value_attrib = (attributable_to("learned value @ commit", drive_win, baseline_win)
                        if getattr(self, "attribute", False) else None)
        meta = {
            "V_hz": [round(float(x), 3) for x in V], "fed_spread_hz": round(spread, 3),
            "wta_counts": [round(float(c), 2) for c in counts], "wta_margin": round(float(margin), 3),
            "wta_choice": int(choice), "decisive": bool(decisive), "lesion": bool(lesion),
            "untrained": bool(self.untrained), "v_margin_hz": self.v_margin_hz,
            "value_attribution_frac": (None if value_attrib is None else round(float(value_attrib), 4)),
        }
        if not decisive:
            return None, meta
        return cands[int(choice)], meta


# ── module-level singleton cache (keyed by (seed, untrained); mirrors the other organs' get_*_organ pattern) ──
_ORGANS: dict[tuple[int, bool], ValueChoiceProductionOrgan] = {}
_ORGANS_LOCK = threading.Lock()


def get_value_choice_organ(seed: int = 42, untrained: bool = False,
                           value_train_trials: int = DEFAULT_VALUE_TRAIN_TRIALS) -> ValueChoiceProductionOrgan:
    key = (int(seed), bool(untrained))
    org = _ORGANS.get(key)
    if org is None:
        with _ORGANS_LOCK:
            org = _ORGANS.get(key)
            if org is None:
                org = ValueChoiceProductionOrgan(seed=seed, untrained=untrained,
                                                 value_train_trials=value_train_trials)
                _ORGANS[key] = org
    return org


# ── the (agent, action) extraction + candidate enumeration (mirrors ChatBrain._substrate_recall's extraction) ──
_STOP = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
         "to", "it", "that", "this", "they", "them", "of", "about"}


def extract_agent_action(question: str, chat):
    """Resolve (agent, action) from a free-text question the SAME way ChatBrain._substrate_recall does — prefer a
    KNOWN agent/action, else structural position. Returns (a, v) or None. A self/identity query is left to the host
    router (returns None), exactly as the substrate recall does, so the value-choice never hijacks a self question."""
    toks = [t.lower().strip(".,!?") for t in str(question).split()]
    content = [t for t in toks if t and t not in _STOP]
    agents_set = getattr(chat, "agents_set", set())
    actions_set = getattr(chat, "actions_set", set())
    a = next((t for t in content if t in agents_set), None) or (content[0] if content else None)
    v = next((t for t in content if t in actions_set), None) or (content[1] if len(content) > 1 else None)
    if not (a and v) or a == v:
        return None
    self_aliases = getattr(getattr(chat, "router", None), "self_aliases", set()) or set()
    if a in self_aliases or v in self_aliases:
        return None
    return a, v


def candidate_patients(chat, a: str, v: str) -> list[str]:
    """The DISTINCT patients stored for (a, v) — the ambiguity the abstain declines on. Order preserved (recency)."""
    seen, out = set(), []
    for (fa, fv, fp) in getattr(chat, "stored_facts", []):
        if fa == a and fv == v and isinstance(fp, str) and fp not in seen:
            seen.add(fp)
            out.append(fp)
    return out


def default_context_fn(chat):
    """Build the per-candidate ENGAGEMENT context from the ChatBrain: fact recency (later-stored -> more engaged) +
    a boost if the candidate is the current discourse-WM referent. A real, deterministic 'prior reward/engagement'
    signal (the DA/limbic context the value critic converts to a learned V).

    SHARED SPIKING AFFERENT (rank-4, `BRAIN_SHARED_SALIENCE`, research/runners/shared_salience_afferent.py;
    default-ON since 2026-09-05's Track-1 flip -- `_shared_salience_flip_soak.py` (rank-20) verifies this specific
    critic through the real ChatBrain path; see that runner's own finding for the verdict).
    The RECENCY/referent bookkeeping above stays host (a legitimate environment/episodic-memory-provenance boundary --
    WHICH fact was stored when, and which is the live discourse referent, is not a cognitive computation); what
    changes is that the per-candidate scalar this function hands the critic is no longer that host ratio directly, but
    the shared curiosity-organ ASK-pool's spiking transduction of it (the SAME afferent da_mode_drives_chat and
    bg_action_selection_production_organ read). OFF (`BRAIN_SHARED_SALIENCE` explicitly `{0,false,no,off,''}`, the
    byte-identical escape post-flip) -> byte-identical to the bare recency ratio below."""
    def ctx(a, v, cands):
        order = {}
        for i, (fa, fv, fp) in enumerate(getattr(chat, "stored_facts", [])):
            if fa == a and fv == v:
                order[fp] = i
        idxs = [order.get(p, 0) for p in cands]
        lo, hi = min(idxs), max(idxs)
        eng = [((ix - lo) / (hi - lo) if hi > lo else 0.5) for ix in idxs]
        ref = None
        try:
            if getattr(chat, "is_multiturn", False):
                ref = chat.agent.held_referent()[0]
        except Exception:
            ref = None
        eng = [min(1.0, e + (0.5 if p == ref else 0.0)) for e, p in zip(eng, cands)]
        if _SHARED.shared_salience_enabled():
            eng = [float(max(0.0, _SHARED.read_salience(e)["normalized"])) for e in eng]
        return eng
    return ctx


def _stable_seed(a: str, v: str) -> int:
    """A stable per-(agent,action) salience seed so the value-INDEPENDENT baseline pull is fixed for a given query
    (the value gradient — not a fresh salience draw — is what differs across turns)."""
    return (abs(hash((str(a), str(v)))) % (2 ** 31 - 1))


def install_value_choice(chat, *, organ=None, context_fn=None, seed: int = 42):
    """Wrap ChatBrain.gate so that when the resolved (agent, action) has >=2 distinct stored patients — the ambiguity
    the deliberation keystone today resolves by ARBITRARY FIRST-MATCH or by a halt-if-unsure ABSTAIN — the value-
    driven choice COMMITS the higher-VALUE patient instead (the owner's "commit by VALUE"). On decline (lesion / non-
    decisive value), the wrapper returns the INNER gate result VERBATIM, so the turn reverts to EXACTLY what the
    pipeline would have done (the arbitrary first-match, or the abstain) — the load-bearing lesion oracle. ADDITIVE +
    guarded: `value_choice_enabled()` is re-checked per call, so with `BRAIN_VALUE_CHOICE` unset the wrapper returns
    the inner result verbatim on EVERY turn (byte-identical to today). A <2-candidate turn (a confident single recall,
    a single-patient/untaught abstain, a self query) is ALWAYS returned verbatim -> the faculty ONLY touches the
    >=2-competing case, and NEVER invents a fact (it commits only STORED candidates -> the no-confab moat holds).
    Idempotent. NO sim/ edit — a ChatBrain method wrap only."""
    if getattr(chat, "_value_choice_installed", False):
        return chat
    _orig_gate = chat.gate
    _ctx = context_fn or default_context_fn(chat)
    _org_holder = {"organ": organ}

    def gate_with_value_choice(question):
        result = _orig_gate(question)
        if not value_choice_enabled():
            return result                       # byte-identical escape (re-checked per call)
        av = extract_agent_action(question, chat)
        if av is None:
            return result                       # not a well-formed (agent, action) query -> inner result verbatim
        a, v = av
        cands = candidate_patients(chat, a, v)
        if len(cands) < 2:
            return result                       # NOT the ambiguous-multi-patient case -> inner result verbatim
        org = _org_holder["organ"]
        if org is None:
            org = get_value_choice_organ(seed=seed, untrained=value_choice_untrained())
            _org_holder["organ"] = org
        try:
            org.ensure_built()
            engagements = _ctx(a, v, cands)
            chosen, meta = org.choose(cands, engagements, lesion=value_choice_lesioned(),
                                      salience_seed=_stable_seed(a, v))
        except Exception:
            return result                       # any failure -> the inner result (never break a turn)
        chat._value_choice_last = {"agent": a, "action": v, "candidates": cands,
                                   "chosen": chosen, "meta": meta, "inner_result": result}
        if chosen is None:
            return result                       # value not decisive (the lesion collapses to this) -> inner result
        return [a, v, chosen]

    chat.gate = gate_with_value_choice
    chat._value_choice_installed = True
    chat._value_choice_orig_gate = _orig_gate
    return chat
