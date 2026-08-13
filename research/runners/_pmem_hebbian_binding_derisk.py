"""PROSPECTIVE MEMORY scaffold retirement -- LEARN the cue->action binding via ONE-SHOT HEBBIAN potentiation at
intention-formation (Gollwitzer implementation-intentions), replacing the build-time synaptic INSTALL.

THE RESIDUAL (the #1 declared `scaffold_retired: NO` for prospective memory). The full faculty is de-risk-GO and
production-wired (organ #13): a spiking PFC intention LATCH + BA10 cue-MONITOR, `fire_on_cue` closed 6/6 by a
supralinear NMDA/dendritic-plateau coincidence amplifier (2026-08-13-prospective-sfa-nmda-amplifier-GO.md). Its ONE
flagged host scaffold: the cue->action CONTENT binding -- WHICH cue Y releases WHICH action X -- is INSTALLED
synaptically AT BUILD (the fixed `cue_X.cortex->rel_X` @cue_to_rel_weight + `act_X.cortex->rel_X` @hold_to_rel_weight
outer-product edges in `_pmem_intention_latch_derisk.ProspectiveMemory.__init__`), exactly like every
SpikingLoopContextBuffer attractor. The parent named the retirement rung explicitly: LEARN the binding via one-shot
Hebbian potentiation at intention-formation.

THE MECHANISM (this runner; additive, NO `sim/` edit; reuse-by-import of the de-risked GO `SFANmdaProspectiveMemory`).
Gollwitzer's implementation-intention -- stating "when Y, do X" -- forms a rapid associative link between the
situational cue Y and the goal action X. Biology writes a conjunction by COINCIDENCE, locally, with no algebra
(`research/biology/coincidence-binding.md`; Kandel 6e: a spine Ca2+ signal is "a biochemical detector of the near
simultaneity of the input (EPSP) and output (backpropagating action potential)"), and it does so ONE-SHOT over a
behavioral timescale (`research/biology/btsp-place-field-formation.md`; Bittner et al. 2017: a SINGLE plateau creates
a place field). The established repo pattern is a Hebbian pre x post OUTER PRODUCT, coincidence/post-gated -- NOT
presynaptic TM facilitation (2026-07-13-RUNG6d-spiking-STP-binder-needs-HEBBIAN-not-presynaptic-6seed-GO.md).

So: build the substrate normally (the canonical binding is installed at build so the homeostat bias + the plateau
threshold CALIBRATE against it -- a DEVELOPMENTAL operating-point tuning of the release circuit, an innate readiness
to hold a cue->action association at the standard synaptic strength), then ZERO the cue->action binding so NONE
exists before the formation turn. AT FORMATION ("remind me to X when Y"), a single Hebbian event co-activates the
cue assembly (Y), the action assembly (X) and -- as the deliberate PFC goal activation that forms an implementation
intention -- an instructional drive to the release pool `rel_X`; the coincident spiking (pre = cue u action cortex,
post = rel_X) POTENTIATES `cue_Y.cortex->rel_X` and `act_X.cortex->rel_X` in ONE shot (a saturating Hebbian outer
product, w_ij = ceiling * sat(pre_i) * sat(post_j)). The cue-monitor + latch then operate on the LEARNED binding.

BRAIN-BASED: the potentiation is a LOCAL pre x post rule driven by REAL spikes (`cp_firing_states`), read + applied
via `set_pathway_weights` -- the SAME class of host-applied local Hebbian outer-product the repo's other spiking
binders use (RUNG6c/6d, gap#2). HOST-BOUNDARY, FLAGGED + narrowed: the binding is now CONTINGENT on a spike-driven
formation event (absent until formation, load-bearing on it) -- the build-time INSTALL is RETIRED. What REMAINS host
is (1) the text->slot / cue-presence sensory boundary (unchanged; declared, like curiosity's novelty derivation),
(2) the formation instructional drive to `rel_X` (the goal-activation ENCODING input, the same host-provides-input
boundary as `_write`'s drive), and (3) the developmental calibration of the pool operating point against the
canonical binding strength. The engine-native STDP realization of the same rule is the further step.

THE ANTI-CHEATS (load-bearing; the mission's central risk = a binding that isn't really learned).
  * NO BINDING BEFORE FORMATION -- assert the summed |cue_monitor weight| is ~0 on a fresh build before any
    formation turn (the install is gone); it becomes > 0 only after the Hebbian event.
  * HEBBIAN-LESION -> NO FIRE -- latch the intention WITHOUT the Hebbian event (`form_intention_no_hebbian`): the
    binding stays 0, so the correct cue does NOT fire (rel_A <= SILENT_MAX). This proves the fire is caused by the
    one-shot formation event, not a residual install. (Distinct from the parent's latch-lesion, which zeroes the
    HELD attractor; here the LATCH is intact and the BINDING is what is absent.)
  * SILENCE STAYS 6/6 -- a learned binding must still be cue-SPECIFIC: every silence clause (wrong-cue,
    no-intention, no-fire-before, lesion) must hold, or the Hebbian event smeared the association (VOID).

THE GATE. The FROZEN gate (thresholds + per-seed clause logic) is IMPORTED from
`_pmem_intention_latch_derisk`; the substrate class is monkey-patched to this Hebbian-binding subclass, and
`base.run_seed` calls `pm.encode_intention(...)` which -- post-calibration -- routes through the one-shot Hebbian
formation. So every clause is scored by the SAME code, now with the binding LEARNED not installed. 6 seeds
42/43/44/100/101/102. THE QUESTION: does the full prospective faculty still fire 6/6 (fire_on_cue + all silence)
with the binding LEARNED, AND is the binding absent-before / Hebbian-lesion-load-bearing on every seed?

  SIM_BACKEND=numpy python -m research.runners._pmem_hebbian_binding_derisk --smoke     # 1 seed, N=3, fast
  SIM_BACKEND=numpy python -m research.runners._pmem_hebbian_binding_derisk --derisk    # 6 seeds, N=5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the de-risked GO substrate + the [H] homeostat + the FROZEN gate (thresholds + clause logic).
import research.runners._pmem_intention_latch_derisk as base            # noqa: E402
import research.runners._pmem_perpool_homeostat_derisk as homeo         # noqa: E402
import research.runners._pmem_sfa_nmda_amplifier_derisk as sfa          # noqa: E402
from research.runners._pmem_sfa_nmda_amplifier_derisk import SFANmdaProspectiveMemory  # noqa: E402
from research.runners._pmem_intention_latch_derisk import (             # noqa: E402  (FROZEN gate constants)
    FIRE_THR, SILENT_MAX, HOLD_FLOOR, LESION_HELD_MAX, SEP_RATIO, GO_MIN_SEEDS_FRAC,
)
from tools.lab import attributable_to, void_if   # noqa: E402
from tools.verdict import Verdict                # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_hebbian_binding.json")

SILENCE_CLAUSES = homeo.SILENCE_CLAUSES
ALL_CLAUSES = ("fire_on_cue",) + SILENCE_CLAUSES + ("separation",)
BINDING_ABSENT_EPS = 1e-3   # summed |cue_monitor weight| must be <= this on a fresh build (no install)


class HebbianBindingProspectiveMemory(SFANmdaProspectiveMemory):
    """SFANmda GO substrate, but the cue->action CONTENT binding is LEARNED via a one-shot Hebbian event at
    intention-formation instead of installed at build.

    Build order: super().__init__ installs the canonical `cue_monitor` binding and calibrates the homeostat bias +
    the plateau theta against it (a developmental operating-point tuning). Then the binding is ZEROED -> none exists
    before formation. `encode_intention` (called by the frozen gate AFTER calibration) routes to
    `form_intention_hebbian`, a single formation event whose coincident spiking potentiates the binding back into
    existence -- absent-before, load-bearing on the event.
    """

    def __init__(self, actions, distractors,
                 form_window=30, form_drive=2500.0, form_rel_drive=1800.0, form_sat=0.06, **kw):
        # set BEFORE super().__init__: super's calibration calls self.encode_intention (my override), which must
        # see _calibrated=False so it plain-latches (no Hebbian) during calibration.
        self._calibrated = False
        self._binding_installed = True     # canonical binding is present through super()'s calibration
        self._ceil_act = float(kw.get("hold_to_rel_weight", 3.2))   # act_X->rel_X potentiated ceiling (LTP saturation)
        self._ceil_cue = float(kw.get("cue_to_rel_weight", 4.2))    # cue_X->rel_X potentiated ceiling
        self._form = dict(window=int(form_window), drive=float(form_drive),
                          rel_drive=float(form_rel_drive), sat=float(form_sat))
        self._last_form = None

        super().__init__(actions, distractors, **kw)   # installs canonical binding + calibrates bias + theta

        # reconstruct the canonical binding edge STRUCTURE (same repeat/tile order as the base install) so the
        # Hebbian event can re-weight EXACTLY those synapses (they already exist in the CSR; no add_missing needed).
        self._act_pairs = {}
        self._cue_pairs = {}
        for a in self.actions:
            actc = self.B.to_host(self._cpat[a]).astype(np.int64)
            cuec = self.B.to_host(self._cpat[f"cue_{a}"]).astype(np.int64)
            rel = np.asarray(self._rel_idx[a], dtype=np.int64)
            self._act_pairs[a] = (np.repeat(actc, rel.size), np.tile(rel, actc.size))
            self._cue_pairs[a] = (np.repeat(cuec, rel.size), np.tile(rel, cuec.size))
        # per-action canonical |weight| sum (the fully-installed reference); total over both actions.
        per_action = self._psize * self._n_rel * (self._ceil_act + self._ceil_cue)
        self._canonical_norm_per_action = float(per_action)
        self._canonical_norm = float(per_action * len(self.actions))

        self._zero_binding()               # RETIRE the install: no cue->action binding before formation
        self._binding_installed = False
        self._calibrated = True

    # ---- binding read / zero ----
    def _all_binding_pairs(self):
        pres, posts = [], []
        for a in self.actions:
            ap, apo = self._act_pairs[a]
            cp_, cpo = self._cue_pairs[a]
            pres.append(ap); posts.append(apo)
            pres.append(cp_); posts.append(cpo)
        return np.concatenate(pres), np.concatenate(posts)

    def binding_weight_norm(self):
        """Summed |weight| over ALL cue_monitor (cue->rel + act->rel) edges, read from the live CSR. ~0 before any
        formation event (the install is gone); > 0 only after the Hebbian potentiation."""
        pre, post = self._all_binding_pairs()
        conn = self.bridge.cp_connections
        indptr = np.asarray(self.B.to_host(conn.indptr))
        indices = np.asarray(self.B.to_host(conn.indices))
        data = np.asarray(self.B.to_host(conn.data))
        pair_idx = {}
        for r in {int(x) for x in np.unique(pre)}:
            s, e = int(indptr[r]), int(indptr[r + 1])
            for off in range(s, e):
                pair_idx[(r, int(indices[off]))] = off
        tot = 0.0
        for i in range(pre.size):
            off = pair_idx.get((int(pre[i]), int(post[i])))
            if off is not None:
                tot += abs(float(data[off]))
        return float(tot)

    def _zero_binding(self):
        for a in self.actions:
            ap, apo = self._act_pairs[a]
            cp_, cpo = self._cue_pairs[a]
            self.bridge.set_pathway_weights("cue_monitor_zero", ap, apo,
                                            np.zeros(ap.size, np.float32), add_missing=False)
            self.bridge.set_pathway_weights("cue_monitor_zero", cp_, cpo,
                                            np.zeros(cp_.size, np.float32), add_missing=False)
        self._reset_dynamics()

    # ---- the ONE-SHOT HEBBIAN FORMATION event ----
    def _formation_step(self, a, act_c, cue_c):
        """One formation step: drive the action assembly (X) AND the cue assembly (Y) (the co-active content of
        'when Y, do X'), the per-pool tonic bias on all rel pools (operating point), plus an instructional drive on
        the target release pool rel_X (the deliberate PFC goal activation). Coincident pre (cortex) x post (rel_X)."""
        cur = self.bridge.cp_external_input_current
        cur[:] = 0.0
        d = np.float32(self._form["drive"])
        cur[act_c] = d
        cur[cue_c] = d
        for act in self.actions:
            b = self._bias_pool[act] if self.homeostat_on else self._rel_bias_pA
            cur[self._rel_idx_dev[act]] = np.float32(b)
        b_a = self._bias_pool[a] if self.homeostat_on else self._rel_bias_pA
        cur[self._rel_idx_dev[a]] = np.float32(b_a + self._form["rel_drive"])   # instructional goal activation
        self.bridge._run_one_simulation_step()

    def form_intention_hebbian(self, action):
        """LEARN the cue->action binding in ONE shot: a formation window of coincident cue+action+rel_X spiking
        potentiates cue_Y->rel_X and act_X->rel_X (a saturating Hebbian outer product), THEN latch the intention."""
        a = action
        self._reset_dynamics()
        act_c = self._cpat[a]
        cue_c = self._cpat[f"cue_{a}"]
        rel_dev = self._rel_idx_dev[a]
        ps, nr = self._psize, self._n_rel
        pre_act = np.zeros(ps, np.float32)
        pre_cue = np.zeros(ps, np.float32)
        post = np.zeros(nr, np.float32)
        w = max(int(self._form["window"]), 1)
        for _ in range(w):
            self._formation_step(a, act_c, cue_c)
            fs = self.bridge.cp_firing_states
            pre_act += self.B.to_host(fs[act_c]).astype(np.float32)
            pre_cue += self.B.to_host(fs[cue_c]).astype(np.float32)
            post += self.B.to_host(fs[rel_dev]).astype(np.float32)
        s = float(self._form["sat"])
        sat_pre_act = np.minimum(1.0, (pre_act / w) / s)
        sat_pre_cue = np.minimum(1.0, (pre_cue / w) / s)
        sat_post = np.minimum(1.0, (post / w) / s)
        # one-shot Hebbian outer product to the potentiated ceiling (LTP saturation); 0 for any silent neuron.
        w_act = (self._ceil_act * np.outer(sat_pre_act, sat_post)).astype(np.float32).ravel()
        w_cue = (self._ceil_cue * np.outer(sat_pre_cue, sat_post)).astype(np.float32).ravel()
        ap, apo = self._act_pairs[a]
        cp_, cpo = self._cue_pairs[a]
        self.bridge.set_pathway_weights("cue_monitor_hebb", ap, apo, w_act, add_missing=False)
        self.bridge.set_pathway_weights("cue_monitor_hebb", cp_, cpo, w_cue, add_missing=False)
        self._last_form = {
            "pre_act_rate": round(float((pre_act / w).mean()), 4),
            "pre_cue_rate": round(float((pre_cue / w).mean()), 4),
            "post_rel_rate": round(float((post / w).mean()), 4),
            "learned_norm": round(float(np.abs(w_act).sum() + np.abs(w_cue).sum()), 2),
            "canonical_norm_per_action": round(self._canonical_norm_per_action, 2),
        }
        self._binding_installed = True
        self._reset_dynamics()
        super().encode_intention(a)        # latch the held intention (self-sustaining cortex<->dlpfc attractor)

    def form_intention_no_hebbian(self, action):
        """LOAD-BEARING LESION: latch the intention WITHOUT the Hebbian event -> the binding stays absent, so the
        correct cue cannot fire (the fire is caused by the one-shot formation event, not a residual install)."""
        self._reset_dynamics()
        super().encode_intention(action)   # latch only; binding remains 0

    def encode_intention(self, action):
        """Frozen-gate entry point. During super()'s calibration (_calibrated False) -> plain latch. In the gate
        (calibrated, binding not yet formed) -> the one-shot Hebbian formation, then latch."""
        if getattr(self, "_calibrated", False) and not self._binding_installed:
            self.form_intention_hebbian(action)
        else:
            super().encode_intention(action)


# --------------------------------------------------------------------------------------------------------
def _binding_checks(seed, N, n_distractors, **kw):
    """Per-seed anti-cheats that the frozen gate does NOT cover: (a) binding ABSENT before formation, present after
    the Hebbian event; (b) HEBBIAN-LESION (latch without the event) -> binding stays 0 -> the cue does NOT fire."""
    dists = [f"d{i}" for i in range(n_distractors)]
    inter = [dists[i % len(dists)] for i in range(N)]
    actions = ["A", "B"]

    # (a) absent-before / present-after
    pm = HebbianBindingProspectiveMemory(actions, dists, seed=seed, **kw)
    norm_before = pm.binding_weight_norm()
    canonical_per_action = pm._canonical_norm_per_action
    pm.encode_intention("A")                       # one-shot Hebbian formation of A + latch
    norm_after = pm.binding_weight_norm()
    learned = dict(pm._last_form or {})

    # (b) Hebbian-lesion: latch A but SKIP the Hebbian event -> the cue must not fire
    pm2 = HebbianBindingProspectiveMemory(actions, dists, seed=seed, **kw)
    pm2.form_intention_no_hebbian("A")
    norm_lesion = pm2.binding_weight_norm()
    for d in inter:
        pm2.intervening_turn(d)
    rel_lesion = float(pm2.present_cue("A")["rel"]["A"])

    return {
        "seed": seed,
        "norm_before": round(norm_before, 3),
        "norm_after_A": round(norm_after, 2),
        "canonical_norm_per_action": round(canonical_per_action, 2),
        "learned": learned,
        "hebbian_lesion_norm": round(norm_lesion, 3),
        "hebbian_lesion_rel_A": round(rel_lesion, 4),
        "binding_absent_before": bool(norm_before <= BINDING_ABSENT_EPS),
        "binding_present_after": bool(norm_after >= 0.5 * canonical_per_action),
        "hebbian_lesion_silent": bool(rel_lesion <= SILENT_MAX),
    }


def _agg_gate(per):
    n_pass = sum(int(p["passed"]) for p in per)
    agg = {c: sum(int(p["clauses"][c]) for p in per) for c in per[0]["clauses"]}
    fire_per_seed = {p["seed"]: round(min(p["fireA"]["rel_A_on_cueA"], p["fireB"]["rel_B_on_cueB"]), 4) for p in per}
    silent_per_seed = {p["seed"]: round(p["max_silent"], 4) for p in per}
    mean_fire = float(np.mean([min(p["fireA"]["rel_A_on_cueA"], p["fireB"]["rel_B_on_cueB"]) for p in per]))
    return dict(n_pass=n_pass, agg=agg, fire_per_seed=fire_per_seed,
                silent_per_seed=silent_per_seed, mean_fire=mean_fire)


def _derisk(seeds, N, n_distractors, smoke=False, **kw):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"PMEM HEBBIAN BINDING [{tag}] -- one-shot Hebbian potentiation of cue->action AT formation (retire the "
          f"build-time install); {len(seeds)} seed(s), N={N}, {n_distractors} distractors; "
          f"form_window={kw.get('form_window')} form_drive={kw.get('form_drive')} "
          f"form_rel_drive={kw.get('form_rel_drive')} form_sat={kw.get('form_sat')}", flush=True)
    t0 = time.time()
    err = None
    gate = checks = None
    try:
        # both the substrate and the frozen gate see the Hebbian-binding class.
        base.ProspectiveMemory = HebbianBindingProspectiveMemory
        homeo._BIAS_CACHE.clear(); sfa._THETA_CACHE.clear()

        gate = []
        checks = []
        print("\n--- FROZEN GATE (binding LEARNED via one-shot Hebbian at formation) + binding anti-cheats ---",
              flush=True)
        for s in seeds:
            g = base.run_seed(s, N, n_distractors, **kw)
            gate.append(g)
            c = _binding_checks(s, N, n_distractors, **kw)
            checks.append(c)
            fails = " ".join(k for k, v in g["clauses"].items() if not v) or "ALL-PASS"
            print(f"  [seed {s}] gate_pass={g['passed']} | fireA={g['fireA']['rel_A_on_cueA']:.3f} "
                  f"fireB={g['fireB']['rel_B_on_cueB']:.3f} max_silent={g['max_silent']:.3f} | {fails}", flush=True)
            print(f"           binding: before={c['norm_before']:.3f} (<= {BINDING_ABSENT_EPS}) "
                  f"after_A={c['norm_after_A']:.1f}/{c['canonical_norm_per_action']:.1f} | "
                  f"HEBB-LESION rel_A={c['hebbian_lesion_rel_A']:.4f} (<= {SILENT_MAX}) "
                  f"silent={c['hebbian_lesion_silent']} | learned={c['learned']}", flush=True)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        traceback.print_exc()

    if err is not None:
        summary = {"probe": "pmem_hebbian_binding", "verdict": f"ERROR -- {err}", "go": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}
        _write(summary)
        return 1

    A = _agg_gate(gate)
    min_seeds = int(np.ceil(GO_MIN_SEEDS_FRAC * len(seeds)))

    # anti-cheat 1: silence must stay 6/6 (a smeared binding = spurious fires).
    silence_regressed = [c for c in SILENCE_CLAUSES if A["agg"].get(c, 0) < len(seeds)]
    # anti-cheat 2: the binding must be ABSENT before formation on every seed.
    absent_before = [c["seed"] for c in checks if not c["binding_absent_before"]]
    present_after = [c["seed"] for c in checks if not c["binding_present_after"]]
    # anti-cheat 3 (load-bearing): the Hebbian-lesion must NOT fire on every seed.
    hebb_lesion_fires = [c["seed"] for c in checks if not c["hebbian_lesion_silent"]]

    cheat_silence = void_if(bool(silence_regressed),
                            f"the learned binding REGRESSED a silence clause {silence_regressed} -> the Hebbian "
                            f"event smeared the cue->action association (a learned binding must stay cue-specific)")
    cheat_absent = void_if(bool(absent_before),
                           f"the cue->action binding was NOT absent before formation on seeds {absent_before} "
                           f"(the build-time install was not actually retired)")
    cheat_lesion = void_if(bool(hebb_lesion_fires),
                           f"the Hebbian-LESION still fired on seeds {hebb_lesion_fires} -> the fire is NOT caused "
                           f"by the one-shot formation event (a residual install remains)")

    fire_on = A["agg"].get("fire_on_cue", 0)
    go = (bool(A["n_pass"] >= min_seeds) and not silence_regressed and not absent_before
          and not present_after and not hebb_lesion_fires and not smoke)

    # ATTRIBUTION: of the correct-cue release with the binding LEARNED (frozen-gate fireA), what fraction is owned
    # by the one-shot formation event vs. what leaks through with the binding lesioned (Hebbian-lesion rel_A)?
    mean_fire_intact = float(np.mean([g["fireA"]["rel_A_on_cueA"] for g in gate]))
    mean_rel_lesion = float(np.mean([c["hebbian_lesion_rel_A"] for c in checks]))
    hebb_share = attributable_to("release owned by the one-shot Hebbian formation (rel_A: formed vs Hebbian-lesion)",
                                 mean_fire_intact, mean_rel_lesion)

    vd = Verdict("pmem_hebbian_binding")
    for c in SILENCE_CLAUSES:
        vd.require(f"silence held with the binding LEARNED: {c} (per-seed count)", A["agg"].get(c, 0),
                   expect=lambda x, n=len(seeds): x == n)
    vd.require("binding ABSENT before formation (per-seed count)",
               sum(int(c["binding_absent_before"]) for c in checks), expect=lambda x, n=len(seeds): x == n)
    vd.require("Hebbian-lesion silent: no binding -> no fire (per-seed count)",
               sum(int(c["hebbian_lesion_silent"]) for c in checks), expect=lambda x, n=len(seeds): x == n)
    vd.reaches("binding norm: before formation -> after the one-shot Hebbian event",
               float(np.mean([c["norm_before"] for c in checks])),
               float(np.mean([c["norm_after_A"] for c in checks])))
    vd.control("correct-cue release: binding formed vs Hebbian-lesioned", mean_fire_intact, mean_rel_lesion,
               min_separation=0.05)
    vd.disabled("STDP / TM-presynaptic facilitation / OU-noise",
                "the ONLY added mechanism is a one-shot Hebbian outer-product potentiation of the cue->action "
                "binding at formation (coincidence-binding.md; RUNG6d Hebbian-not-presynaptic); the SFA + NMDA "
                "plateau amplifier + the per-pool homeostat are inherited unchanged from the GO substrate")
    decided = vd.decide(go)

    silence_counts = ", ".join(f"{c}:{A['agg'][c]}" for c in SILENCE_CLAUSES)
    if smoke:
        verdict = (f"SMOKE OK -- one-shot Hebbian binding RUNS end-to-end; every condition live/measured. "
                   f"gate fire_on_cue={fire_on}/{len(seeds)}; binding absent-before "
                   f"{sum(int(c['binding_absent_before']) for c in checks)}/{len(seeds)}, Hebbian-lesion silent "
                   f"{sum(int(c['hebbian_lesion_silent']) for c in checks)}/{len(seeds)}; "
                   f"silence-regressed={silence_regressed or 'none'}. Not a GO claim; run --derisk for the verdict.")
    elif cheat_silence or cheat_absent or cheat_lesion:
        reasons = [r for r in (
            (f"silence regressed {silence_regressed}" if silence_regressed else None),
            (f"binding not absent before formation {absent_before}" if absent_before else None),
            (f"Hebbian-lesion fired {hebb_lesion_fires}" if hebb_lesion_fires else None)) if r]
        verdict = (f"VOID -- {'; '.join(reasons)}. The binding was not genuinely LEARNED-not-installed or lost "
                   f"cue-specificity. gate fire_on_cue={fire_on}/{len(seeds)}.")
    elif go:
        verdict = (
            f"GO -- the cue->action content binding is RETIRED from the build-time install: it is LEARNED via a "
            f"ONE-SHOT HEBBIAN potentiation at intention-formation (Gollwitzer implementation-intention), and the "
            f"full prospective faculty still fires {fire_on}/{len(seeds)} with every silence clause 6/6 "
            f"({silence_counts}). {A['n_pass']}/{len(seeds)} seeds pass EVERY frozen-gate clause (need {min_seeds}). "
            f"LOAD-BEARING: the binding is ABSENT before the formation turn on every seed "
            f"(|w|<= {BINDING_ABSENT_EPS}) and the Hebbian-LESION (latch WITHOUT the event) does NOT fire on any "
            f"seed (rel_A <= {SILENT_MAX}) -> the fire is caused by the one-shot formation event, not a residual "
            f"install. The one-shot Hebbian outer product (coincident cue+action+rel_X spiking -> saturating "
            f"potentiation to the LTP ceiling) reconstructs the cue-specific binding from REAL spikes "
            f"(cp_firing_states). Remaining host boundary (declared, narrowed): the text->slot / cue-presence "
            f"sensory scaffold, the formation goal-activation encoding drive, and the developmental calibration of "
            f"the pool operating point. NO sim/ edit; reuse-by-import of the SFANmda GO substrate + frozen gate.")
    else:
        fails = {c: A["agg"][c] for c in A["agg"] if A["agg"][c] < len(seeds)}
        verdict = (f"BOUNDARY -- with the binding LEARNED via one-shot Hebbian, {A['n_pass']}/{len(seeds)} seeds "
                   f"pass all clauses (need {min_seeds}); fire_on_cue={fire_on}/{len(seeds)}. Residual clauses: "
                   f"{sorted(fails)}; absent-before-fails={absent_before}; present-after-fails={present_after}; "
                   f"hebbian-lesion-fires={hebb_lesion_fires}. Honest residual -- name the next single-variable "
                   f"lever (form_drive / form_sat / form_window / form_rel_drive), do NOT force GO.")

    summary = {
        "probe": "pmem_hebbian_binding", "verdict": verdict, "go": bool(go),
        "task": ("prospective-memory scaffold retirement: LEARN the cue->action content binding via a ONE-SHOT "
                 "HEBBIAN outer-product potentiation at intention-formation (coincident cue+action+rel spiking "
                 "saturating to the LTP ceiling), replacing the build-time synaptic install. The homeostat bias + "
                 "plateau theta are calibrated against the canonical binding (developmental operating-point tuning) "
                 "then the binding is zeroed; formation relearns it from real spikes. Frozen gate (thresholds + "
                 "clause logic) imported; substrate monkey-patched. NO sim/ edit; reuse-by-import."),
        "gate": {"FIRE_THR": FIRE_THR, "SILENT_MAX": SILENT_MAX, "HOLD_FLOOR": HOLD_FLOOR,
                 "LESION_HELD_MAX": LESION_HELD_MAX, "SEP_RATIO": SEP_RATIO,
                 "GO_MIN_SEEDS_FRAC": GO_MIN_SEEDS_FRAC},
        "mechanism": {"form_window": kw.get("form_window"), "form_drive": kw.get("form_drive"),
                      "form_rel_drive": kw.get("form_rel_drive"), "form_sat": kw.get("form_sat"),
                      "ceil_act": kw.get("hold_to_rel_weight"), "ceil_cue": kw.get("cue_to_rel_weight")},
        "N_intervening": N, "n_distractors": n_distractors, "seeds": list(seeds), "min_seeds_to_go": min_seeds,
        "n_pass": A["n_pass"], "per_clause_pass_counts": A["agg"], "fire_on_cue": fire_on,
        "fire_per_seed": A["fire_per_seed"], "max_silent_per_seed": A["silent_per_seed"], "mean_fire": A["mean_fire"],
        "binding_checks": checks,
        "binding_absent_before_fails": absent_before,
        "binding_present_after_fails": present_after,
        "hebbian_lesion_fires": hebb_lesion_fires,
        "silence_regressed": silence_regressed,
        "release_owned_by_hebbian_formation": hebb_share,
        "preconditions": (decided or {}).get("preconditions"),
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed_gate": gate,
        "BIOLOGY": ("Binding by coincidence (research/biology/coincidence-binding.md; Kandel 6e: a spine Ca2+ signal "
                    "is 'a biochemical detector of the near simultaneity of the input (EPSP) and output "
                    "(backpropagating action potential)') written ONE-SHOT over a behavioral timescale "
                    "(research/biology/btsp-place-field-formation.md; Bittner et al. 2017: a single plateau creates "
                    "a place field). The established repo realization is a Hebbian pre x post outer product, "
                    "coincidence/post-gated, NOT presynaptic TM facilitation "
                    "(2026-07-13-RUNG6d-spiking-STP-binder-needs-HEBBIAN-not-presynaptic-6seed-GO.md). Realized as a "
                    "saturating one-shot Hebbian outer product w_ij = ceiling * sat(pre_i) * sat(post_j) on real "
                    "cp_firing_states; a host-applied LOCAL rule (the same class the repo's spiking binders use).")
    }
    _write(summary)
    print("\n" + "=" * 118, flush=True)
    print(f"[pmem-hebbian] VERDICT: {verdict}", flush=True)
    print(f"[pmem-hebbian] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (summary["go"] or smoke) else 1


def _write(summary):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    # substrate knobs (parity with the GO substrate; defaults MATCH it so the operating point is unchanged)
    ap.add_argument("--hold-to-rel-weight", type=float, default=3.2)
    ap.add_argument("--cue-to-rel-weight", type=float, default=4.2)
    ap.add_argument("--rel-recurrent-weight", type=float, default=0.10)
    ap.add_argument("--rel-bias-pA", type=float, default=-1050.0)
    ap.add_argument("--n-rel", type=int, default=60)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--pattern-size", type=int, default=40)
    # homeostat knobs (identical for ALL seeds; defaults match [H])
    ap.add_argument("--homeostat-r-set", type=float, default=0.045)
    ap.add_argument("--homeostat-eta", type=float, default=4000.0)
    ap.add_argument("--homeostat-iters", type=int, default=15)
    ap.add_argument("--homeostat-window", type=int, default=6)
    ap.add_argument("--homeostat-bias-min", type=float, default=-4000.0)
    ap.add_argument("--homeostat-bias-max", type=float, default=0.0)
    # SFA + NMDA-plateau knobs (the GO config; plateau owns the fire_on_cue closure)
    ap.add_argument("--sfa-g", type=float, default=2500.0)
    ap.add_argument("--sfa-tau", type=float, default=100.0)
    ap.add_argument("--plateau-g", type=float, default=6000.0)
    ap.add_argument("--plateau-margin", type=float, default=1.05)
    ap.add_argument("--plateau-cap", type=float, default=1600.0)
    # one-shot Hebbian FORMATION knobs (label-free; identical for ALL seeds)
    ap.add_argument("--form-window", type=int, default=30, help="formation coincidence window (steps)")
    ap.add_argument("--form-drive", type=float, default=2500.0, help="cue+action assembly formation drive (pA)")
    ap.add_argument("--form-rel-drive", type=float, default=1800.0,
                    help="instructional goal-activation drive on rel_X during formation (pA)")
    ap.add_argument("--form-sat", type=float, default=0.06,
                    help="Hebbian per-factor saturation firing-fraction (sat(x)=min(1, (rate)/form_sat)); 0.06 fully "
                         "saturates every participating neuron -> the learned binding reconstructs EXACTLY canonical")
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else a.seeds
    kw = dict(hold_to_rel_weight=a.hold_to_rel_weight, cue_to_rel_weight=a.cue_to_rel_weight,
              rel_recurrent_weight=a.rel_recurrent_weight, rel_bias_pA=a.rel_bias_pA,
              n_rel=a.n_rel, n=a.n, pattern_size=a.pattern_size,
              homeostat_on=True, sfa_on=True, plateau_on=True,
              homeostat_r_set=a.homeostat_r_set, homeostat_eta=a.homeostat_eta,
              homeostat_iters=a.homeostat_iters, homeostat_window=a.homeostat_window,
              homeostat_bias_min=a.homeostat_bias_min, homeostat_bias_max=a.homeostat_bias_max,
              sfa_g=a.sfa_g, sfa_tau=a.sfa_tau, plateau_g=a.plateau_g,
              plateau_margin=a.plateau_margin, plateau_cap=a.plateau_cap,
              form_window=a.form_window, form_drive=a.form_drive,
              form_rel_drive=a.form_rel_drive, form_sat=a.form_sat)
    if a.smoke:
        return _derisk([seeds[0]], N=3, n_distractors=min(3, a.n_distractors), smoke=True, **kw)
    return _derisk(seeds, N=a.N, n_distractors=a.n_distractors, smoke=False, **kw)


if __name__ == "__main__":
    raise SystemExit(main())
