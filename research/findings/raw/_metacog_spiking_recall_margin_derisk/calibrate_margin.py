"""Calibration artifact for the spiking recall-margin de-risk (scaffold-retirement backlog rank 9).

Measures, on the REAL tiny-demo `OneBrainComposer` (seed=42, `BRAIN_METACOG_SPIKING_MARGIN=1`, through the
real `/api/brain-chat`-building path `webapp.server._build_chat_brain`), the per-ROLE rectified matched-filter
score arrays a real query produces (clean + 4 synaptic-noise-degraded conditions), then:

  1. Establishes that the winner-pick's `_cleanup_drive_pA=60` (the pre-existing, validated constant
     `_spiking_cleanup`/`OneBrainComposer._spiking_select` already use) drives ZERO spikes over the cleanup
     window on this neuron population -- unusable as a MARGIN source (a single-outcome argmax-over-firing pick
     never needs the loser to fire, so this was never noticed before).
  2. Sweeps candidate (drive_pA, window) operating points for `_spiking_margin` and reports each one's
     correlation with the host `_margin` ((peak-runner_up)/peak) formula, to choose `_margin_drive_pA`.
  3. Fits a linear regression `margin_spiking ~= a*margin + b` at the chosen operating point (300pA, the
     existing 120-step window) over the resulting 25 role-level (host, spiking) pairs, and inverts it onto the
     ROLE_CONF_LO/HI band to derive SPIKING_MARGIN_LO/HI (the SAME anchor-remap methodology `margin_snr`
     already uses via SNR_LO/HI in `metacog_production_organ.py`).

Run standalone (not via `-m research.runners.X`): this is calibration/exploration support for the de-risk, not
a runner producing a decision artifact itself -- the actual 6-seed GO/PARTIAL evidence is
`_metacog_spiking_recall_margin_derisk.py`. See research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md.
"""
import os
import sys
import json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC",
           "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP",
           "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY",
           "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES",
           "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN",
           "BRAIN_BG_SELECT", "BRAIN_SILENT_WM", "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)
os.environ["BRAIN_METACOG_SPIKING_MARGIN"] = "1"

import numpy as np
import research.runners.rf_phasor_composer as RFP
import webapp.server as S
from research.runners._emergent_graceful_degradation_derisk import _noise

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate_margin_output.json")


def capture_real_role_scores():
    """Capture the raw per-role rectified score arrays `_spiking_margin` receives on a real tiny-demo turn,
    clean + 4 synaptic-noise levels, deduplicated (the composer's internal consumers call the cleanup for the
    SAME role more than once per turn)."""
    captured = []
    orig = RFP.RFPhasorComposer._spiking_margin

    def _wrap(self, scores, lesion=False):
        if not lesion:
            captured.append(np.maximum(np.asarray(scores, dtype=float), 0.0).tolist())
        return orig(self, scores, lesion=lesion)

    RFP.RFPhasorComposer._spiking_margin = _wrap
    try:
        chat, _source = S._build_chat_brain("tiny-demo", "stub")
        comp = getattr(getattr(chat, "inner", None), "composer", None)
        base_conns = list(comp.buffer.store_conns)
        Q = "what does the brain use"
        sid = [0]

        def ask(noised=None):
            comp.buffer.store_conns = noised if noised is not None else list(base_conns)
            sid[0] += 1
            ck = (f"cal{sid[0]:03d}", "tiny-demo", "stub")
            S._BRAIN_CHATS[ck] = chat
            try:
                r = S.brain_chat(S.BrainChatRequest(session=f"cal{sid[0]:03d}", message=Q, brain="tiny-demo",
                                                    reset=False, rich=True, renderer="stub"))
                return json.loads(bytes(r.body))
            finally:
                comp.buffer.store_conns = list(base_conns)

        conditions = {}
        captured.clear(); ask(None)
        conditions["clean"] = list(captured)
        for sigma in [0.6, 1.1, 1.5, 2.0]:
            captured.clear()
            ask(_noise(base_conns, sigma, np.random.default_rng(42)))
            conditions[f"sigma{sigma}"] = list(captured)
    finally:
        RFP.RFPhasorComposer._spiking_margin = orig

    # dedup identical arrays within a condition (multiple internal consumers re-cleanup the same role)
    deduped = {}
    for cond, arrs in conditions.items():
        seen, uniq = set(), []
        for a in arrs:
            key = tuple(round(x, 1) for x in a[:3])
            if key not in seen:
                seen.add(key); uniq.append(a)
        deduped[cond] = uniq
    return deduped


def host_margin(scores):
    s = np.sort(np.maximum(np.asarray(scores, dtype=float), 0.0))[::-1]
    return float((s[0] - s[1]) / (s[0] + 1e-9)) if s.size >= 2 and s[0] > 0 else 0.0


def spiking_margin_at(comp, scores, drive_pA, window):
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    V = scores.size
    if V < 2:
        return 0.0, 0.0
    peak = float(scores.max())
    if peak <= 1e-9:
        return 0.0, 0.0
    drive = (scores / peak) * drive_pA
    bank = comp._izh_bank(V)
    import sim.backend as _b
    xp, _ = _b.get_backend()
    bank.cp_membrane_potential_v[:] = bank._cleanup_v0
    bank.cp_recovery_variable_u[:] = bank._cleanup_u0
    bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
    firing = np.zeros(V)
    for _ in range(window):
        bank._run_one_simulation_step()
        firing += np.asarray(bank.cp_firing_states).astype(float)
    bank.cp_external_input_current[:] = 0.0
    s = np.sort(firing)[::-1]
    if s[0] <= 0.0:
        return 0.0, 0.0
    return float((s[0] - s[1]) / (s[0] + 1e-9)), float(s[0])


def main():
    out = {}

    # 1. establish the winner-pick's drive constant produces zero firing (the root observation)
    comp0 = RFP.RFPhasorComposer(seed=42, D=64, enable_spiking_cleanup=True)
    V = len(comp0.words)
    scores = np.zeros(V); scores[0] = 1.0; scores[1] = 0.1
    zero_drive_margin, winner_spikes_at_60pA = spiking_margin_at(comp0, scores, comp0._cleanup_drive_pA, comp0._cleanup_window)
    out["winner_pick_drive_pA"] = comp0._cleanup_drive_pA
    out["winner_spikes_at_winner_pick_drive"] = winner_spikes_at_60pA
    print(f"at the winner-pick's drive ({comp0._cleanup_drive_pA}pA): winner spike count over "
          f"{comp0._cleanup_window} steps = {winner_spikes_at_60pA} (0 -> unusable as a margin)")

    # 2. capture real role scores, clean + 4 noise levels
    print("capturing real tiny-demo role scores (clean + 4 noise levels)...")
    conditions = capture_real_role_scores()
    out["captured_conditions"] = {k: len(v) for k, v in conditions.items()}

    # 3. sweep (drive, window) operating points
    comp = RFP.RFPhasorComposer(seed=42, D=64)
    sweep_results = []
    for drive_pA, window in [(300, 120), (300, 300), (500, 300), (300, 600), (500, 600), (800, 600)]:
        pairs = []
        for cond, arrs in conditions.items():
            for arr in arrs:
                hm = host_margin(arr)
                sm, _ = spiking_margin_at(comp, arr, drive_pA, window)
                pairs.append((cond, hm, sm))
        hosts = np.array([p[1] for p in pairs]); spikes = np.array([p[2] for p in pairs])
        pear = float(np.corrcoef(hosts, spikes)[0, 1]) if hosts.std() > 0 and spikes.std() > 0 else None

        def rankdata(x):
            order = np.argsort(x); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(x))
            return ranks
        rh, rs = rankdata(hosts), rankdata(spikes)
        spear = float(np.corrcoef(rh, rs)[0, 1]) if hosts.std() > 0 and spikes.std() > 0 else None
        sweep_results.append({"drive_pA": drive_pA, "window": window, "pearson_r": pear, "spearman_rho": spear,
                              "n": len(pairs), "pairs": [(c, float(h), float(s)) for c, h, s in pairs]})
        print(f"drive={drive_pA}pA window={window}: pearson_r={pear:.4f} spearman_rho={spear:.4f} n={len(pairs)}")
    out["operating_point_sweep"] = sweep_results

    # 4. fit the FINAL chosen operating point (300pA, 120 steps -- reuses the existing _cleanup_window) and
    # derive SPIKING_MARGIN_LO/HI via the SAME anchor-remap methodology SNR_LO/HI uses for margin_snr.
    chosen = next(r for r in sweep_results if r["drive_pA"] == 300 and r["window"] == 120)
    hosts = np.array([p[1] for p in chosen["pairs"]]); spikes = np.array([p[2] for p in chosen["pairs"]])
    a, b = np.polyfit(hosts, spikes, 1)
    ROLE_CONF_LO, ROLE_CONF_HI = 0.30, 0.50
    SPIKING_LO = float(a * ROLE_CONF_LO + b)
    SPIKING_HI = float(a * ROLE_CONF_HI + b)
    out["chosen_operating_point"] = {"drive_pA": 300, "window": 120}
    out["linear_fit"] = {"a": float(a), "b": float(b), "pearson_r": chosen["pearson_r"],
                         "spearman_rho": chosen["spearman_rho"], "n": chosen["n"]}
    out["SPIKING_MARGIN_LO"] = SPIKING_LO
    out["SPIKING_MARGIN_HI"] = SPIKING_HI
    print(f"\nCHOSEN: drive=300pA window=120 (reuses existing _cleanup_window, no new constant)")
    print(f"linear fit: spiking = {a:.6f}*host + {b:.6f}  (r={chosen['pearson_r']:.4f})")
    print(f"SPIKING_MARGIN_LO={SPIKING_LO!r}  SPIKING_MARGIN_HI={SPIKING_HI!r}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
