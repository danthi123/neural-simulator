"""Bank-level POWER SIMULATION for amendment 3 of `research/open-ended-production-turn-lb`
(docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md, "Amendment 3" amendment log).

WHY THIS EXISTS (round-4 review, 2026-09-23). The amendment-3 log asserts a power simulation was run BEFORE
choosing M=4 sessions/arm, K=8 asks/session and the 0.10 Delta floor, and reports numbers (six bank seeds
295/302/309/701/708/715, Delta 0.09-0.46, mean 0.25, 6/6 positive) that were traced to NEITHER a script NOR an
output artifact (`git log -S` on those numbers finds them only inside the prose of the prereg). This script and
its committed output are the review's requested fix: it re-derives the SAME question -- does the amendment-3
design (independent per-session noise streams, M sessions/arm, K asks/session) separate an "intact" (real,
peaked host-likelihood weights) arm from a "lesion" (uniform weights) arm across independent bank builds -- using
the SAME production sampler class, the SAME real seed-42 host weight vector, and the SAME 6 seed values named in
the prereg.

HONESTY BOUNDARY: this is a FRESH re-derivation, not a byte-identical replay of whatever produced the original
numbers. The original script was never committed and cannot be recovered (confirmed above by `git log -S`), so
this script's own numbers -- not the prose ones -- are the ones now traceable to a cited artifact. See the
amendment log correction in the PREREG for the exact wording.

NOT a governed run and NOT the production turn (same declaration as the original power simulation): no webapp
brain build, no corpus, no TEACH/ASK chat turns. It drives the PRODUCTION sampler class
(`research.runners._followon2_spiking_wta_sampler_derisk.SpikingWTASampler.draw_from_weights`, the exact method
`GenerativeReplayProposer._sample_weighted` calls in production) directly on a fixed weight vector, at the
production operating point (base_pA=110, gain_pA=160, read_window=120, ou_std_current_pA=200, n_cand_max=64 --
all defaults, none overridden here).

DESIGN (mirrors the amendment-3 session protocol exactly, minus the brain/corpus):
  - The seed-42 HOST weight vector is the real one: `likelihood_weight` from the committed
    `research/findings/raw/_load_bearing/_oe_production_turn/a2/default/default_s42_intact.json` (read AFTER the
    real seed-42 asks in that governed run -- not synthesized here).
  - "intact" arm: that weight vector, unchanged, drives BOTH the draw and the score. "lesion" arm: `np.ones_like`
    (the SAME ablation `BRAIN_SPIKING_DRAW_LESION=1` performs on `_weights()`) drives the draw, but -- exactly as
    `score_seed_a3`'s `w_ref = intact[0]["likelihood_weight"]` does in the real probe -- every session of BOTH
    arms is SCORED against the SAME intact weight vector. Scoring the lesion arm's replies against its own
    (uniform) weights would make its session value identically 1.0 by construction (every candidate has equal
    weight, so w(reply)/max(w) = 1 for any reply) and hide the entire effect; scoring against the shared w_ref is
    what makes Delta measure "did the likelihood-blind arm's replies still land on high-intact-weight words".
  - For each of the 6 BANK SEEDS (295, 302, 309, 701, 708, 715 -- named in the prereg, used here as the
    sampler's `seed=` build parameter, i.e. the Izhikevich WTA bank's own heterogeneity seed): build ONE
    `SpikingWTASampler`. Because `cfg.seed` deterministically seeds the bank's heterogeneity (pinned by
    `tests/test_determinism.py::TestSubstrateActuallySeeded`), re-using one build across a bank seed's sessions is
    equivalent to the amendment-3 protocol's per-session fresh-process rebuild at a FIXED seed (same seed -> same
    heterogeneity every time) -- just without paying for M+M redundant identical builds.
  - M=4 "intact" sessions and M=4 "lesion" sessions per bank seed. Each session installs its OWN noise stream
    (`noise_seed = 1000*bank_seed + j` intact, `+500+j` lesion -- the SAME formula as `a3_noise_seed` in the
    probe) by swapping the backend's global RNG to `RandomState(noise_seed)` for the session's asks only, then
    restoring it (via `sim.backend.get_random_state`/`set_random_state`, NOT the probe's `_install_noise_stream`
    wrapper -- that wrapper mutates a shared class attribute and is only safe called ONCE per process, as
    amendment 3 does with a fresh subprocess per session; this script calls it many times in one process, so it
    swaps state directly to avoid a wrapper-chaining bug that would silently start running one session's draws on
    a PRIOR session's stream).
  - K=8 asks per session. Each ask: a draw-until-admissible loop of up to 8 attempts (re-drawing while the reply
    has zero host weight -- an "inadmissible" reply mirrors the probe's own `admissible_set` notion; the lesion
    arm's uniform weights make every reply admissible on attempt 1 by construction). Each attempt is one
    `draw_from_weights` call at the production default `max_retries=3`.
  - Session value v = mean over the K asks of w(reply) / max(w) (an ABSTAIN cannot occur here -- `draw_from_weights`
    always returns a candidate -- so this differs from the probe's ABSTAIN=0 convention only in that ABSTAIN never
    arises in this synthetic harness).
  - Delta(bank_seed) = mean(v over intact sessions) - mean(v over lesion sessions).

Run:  SIM_BACKEND=numpy python -u -m research.runners._lbf_oe_a3_power_simulation \
          --out research/findings/raw/_load_bearing/_oe_production_turn/a3_power_simulation/power_sim.json
Selftest (no brain build): python -m research.runners._lbf_oe_a3_power_simulation --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_BANK_SEEDS = (295, 302, 309, 701, 708, 715)
DEFAULT_M = 4                  # sessions per arm
DEFAULT_K = 8                  # asks per session
DEFAULT_MAX_RETRIES = 3        # production default of SpikingWTASampler.draw_from_weights
DEFAULT_ADMISSIBLE_ATTEMPTS = 8
DEFAULT_N_CAND_MAX = 64        # production default (max(64, len(candidates)) in _ensure_spiking_sampler)
DEFAULT_WEIGHTS_JSON = os.path.join(
    "research", "findings", "raw", "_load_bearing", "_oe_production_turn",
    "a2", "default", "default_s42_intact.json")


def load_seed42_weights(path=DEFAULT_WEIGHTS_JSON):
    """The REAL seed-42 host weight vector: `likelihood_weight` from the committed governed-run artifact (read
    AFTER the real seed-42 asks; see `_worker`'s `weights = {p: float(x) for p, x in zip(prop.patients, w)}`)."""
    with open(path) as fh:
        d = json.load(fh)
    lw = d["likelihood_weight"]
    candidates = list(lw.keys())
    weights = np.array([float(lw[c]) for c in candidates], dtype=np.float64)
    return candidates, weights


def _dummy_row_and_P():
    """`SpikingWTASampler.__init__` needs a co-occurrence matrix P + word->index row + threshold tau ONLY to compute
    `self.encodable_agents` (an attribute unused by `draw_from_weights`, the sole method this script calls). A
    trivial all-zero P over the taxonomy's agent+action vocabulary lets `__init__` run without touching the real
    corpus; `_encodable_agents` then falls back to `self.agents` (its own `out or self.agents` clause) -- inert."""
    from research.runners._genfrontier_b2_generative_replay_derisk import _category_pools
    from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8
    agents, actions, _patients = _category_pools(TAXONOMY_8x8)
    vocab = sorted(set(agents) | set(actions))
    row = {w: i for i, w in enumerate(vocab)}
    P = np.zeros((len(vocab), len(vocab)), dtype=np.float64)
    tau = 1.0
    return P, row, tau


def make_sampler(bank_seed, n_cand_max=DEFAULT_N_CAND_MAX):
    from research.runners._followon2_spiking_wta_sampler_derisk import SpikingWTASampler
    P, row, tau = _dummy_row_and_P()
    return SpikingWTASampler(P, row, tau, seed=int(bank_seed), n_cand_max=int(n_cand_max))


def draw_until_admissible(sampler, drive_weights, candidates, max_attempts=DEFAULT_ADMISSIBLE_ATTEMPTS,
                           max_retries=DEFAULT_MAX_RETRIES):
    """Re-ask while the reply is INADMISSIBLE (zero DRIVE weight) -- mirrors the probe's `admissible_set` notion.
    The lesion arm's uniform drive weights make attempt 1 always admissible. Gives up and returns the last draw
    after `max_attempts` (never silently retries forever)."""
    reply = None
    for _ in range(max(1, int(max_attempts))):
        reply = sampler.draw_from_weights(drive_weights, candidates, max_retries=max_retries)
        idx = candidates.index(reply)
        if drive_weights[idx] > 0:
            return reply
    return reply


def run_session(sampler, drive_weights, score_weights, candidates, noise_seed, k=DEFAULT_K,
                 max_retries=DEFAULT_MAX_RETRIES, admissible_attempts=DEFAULT_ADMISSIBLE_ATTEMPTS):
    """ONE session: K asks, all drawn on a dedicated `RandomState(noise_seed)` stream swapped in for the
    session's duration then restored (direct state save/restore -- see the module docstring for why this differs
    from the probe's `_install_noise_stream` wrapper when called more than once per process). `drive_weights`
    feeds the sampler (arm-specific: intact w, or uniform for the lesion arm); `score_weights` is the FIXED
    reference vector (always the intact host w, exactly as `score_seed_a3`'s `w_ref` is) used to compute the
    session value -- see the module docstring for why scoring must NOT use the lesion's own uniform weights."""
    from sim.backend import get_random_state, set_random_state
    peak = float(np.max(score_weights)) if score_weights.size else 0.0
    saved = get_random_state()
    set_random_state(np.random.RandomState(int(noise_seed)).get_state())
    replies = []
    try:
        for _ in range(int(k)):
            r = draw_until_admissible(sampler, drive_weights, candidates, max_attempts=admissible_attempts,
                                       max_retries=max_retries)
            replies.append(r)
    finally:
        set_random_state(saved)
    vals = [float(score_weights[candidates.index(r)]) / peak if peak > 0 else 0.0 for r in replies]
    return {"replies": replies, "session_value": float(np.mean(vals)) if vals else 0.0}


def run_bank_seed(bank_seed, candidates, weights, m=DEFAULT_M, k=DEFAULT_K, max_retries=DEFAULT_MAX_RETRIES,
                   admissible_attempts=DEFAULT_ADMISSIBLE_ATTEMPTS, n_cand_max=DEFAULT_N_CAND_MAX):
    sampler = make_sampler(bank_seed, n_cand_max=n_cand_max)
    uniform = np.ones_like(weights)
    intact_sessions = [run_session(sampler, weights, weights, candidates, noise_seed=1000 * bank_seed + j, k=k,
                                    max_retries=max_retries, admissible_attempts=admissible_attempts)
                       for j in range(int(m))]
    lesion_sessions = [run_session(sampler, uniform, weights, candidates, noise_seed=1000 * bank_seed + 500 + j,
                                    k=k, max_retries=max_retries, admissible_attempts=admissible_attempts)
                       for j in range(int(m))]
    mean_intact = float(np.mean([s["session_value"] for s in intact_sessions]))
    mean_lesion = float(np.mean([s["session_value"] for s in lesion_sessions]))
    return {"bank_seed": int(bank_seed), "intact_sessions": intact_sessions, "lesion_sessions": lesion_sessions,
            "mean_intact": mean_intact, "mean_lesion": mean_lesion, "delta": mean_intact - mean_lesion}


def run_power_simulation(bank_seeds=DEFAULT_BANK_SEEDS, m=DEFAULT_M, k=DEFAULT_K,
                          max_retries=DEFAULT_MAX_RETRIES, admissible_attempts=DEFAULT_ADMISSIBLE_ATTEMPTS,
                          n_cand_max=DEFAULT_N_CAND_MAX, weights_json=DEFAULT_WEIGHTS_JSON):
    from tools.lab import attributable_to
    candidates, weights = load_seed42_weights(weights_json)
    per_seed = [run_bank_seed(s, candidates, weights, m=m, k=k, max_retries=max_retries,
                               admissible_attempts=admissible_attempts, n_cand_max=n_cand_max)
                for s in bank_seeds]
    deltas = [r["delta"] for r in per_seed]
    # ATTRIBUTION (not just measuring both arms): of the mean session value, how much is present ONLY in the
    # intact (real-likelihood) arm vs ALSO present in the lesion (uniform-weights) control? A gap#5-shaped check
    # -- the lesion holds the DRAW fixed (same sampler, same noise-stream protocol) but not the SCORING scale
    # (both are scored against the same w_ref), so a nonzero lesion mean is expected (a uniformly random reply
    # still lands on an admissible, positive-weight word some of the time) and is not itself a defect; what
    # matters is whether the manipulation (removing the likelihood from the DRIVE) actually moves the score.
    mean_intact_all = float(np.mean([r["mean_intact"] for r in per_seed]))
    mean_lesion_all = float(np.mean([r["mean_lesion"] for r in per_seed]))
    attribution = attributable_to("intact-likelihood-drive vs lesion (uniform-drive) @ seed-42 host vector, "
                                   "mean over %d bank seeds" % len(bank_seeds), mean_intact_all, mean_lesion_all)
    return {
        "purpose": "amendment-3 power simulation re-derivation (round-4 review fix, 2026-09-23) -- see module "
                   "docstring; NOT a governed run",
        "config": {"bank_seeds": list(bank_seeds), "m_sessions_per_arm": m, "k_asks_per_session": k,
                   "max_retries": max_retries, "admissible_attempts": admissible_attempts,
                   "n_cand_max": n_cand_max, "weights_json": weights_json, "candidates": candidates,
                   "weights": weights.tolist(),
                   "sampler_operating_point": {"base_pA": 110.0, "gain_pA": 160.0, "read_window": 120,
                                                "ou_std_current_pA": 200.0, "temperature": 1.0}},
        "attribution": {"mean_intact_all_seeds": mean_intact_all, "mean_lesion_all_seeds": mean_lesion_all,
                        "attributable_to_likelihood_drive": attribution},
        "per_bank_seed": per_seed,
        "aggregate": {"delta_min": min(deltas), "delta_max": max(deltas), "delta_mean": float(np.mean(deltas)),
                      "n_positive": sum(1 for d in deltas if d > 0), "n_seeds": len(deltas)},
    }


def selftest():
    """Pure checks, no brain build -- each must be able to FAIL."""
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    class FakeSampler:
        """Deterministic-enough fake: draws the argmax-weighted candidate whose index equals a value pulled off
        the (swapped) global RNG, mod len(candidates) -- exercises noise-stream isolation without a real bank."""

        def draw_from_weights(self, weights, candidates, max_retries=3):
            idx = int(np.random.randint(0, len(candidates)))
            return candidates[idx]

    candidates = ["a", "b", "c", "d"]
    weights = np.array([0.0, 1.0, 2.0, 0.0])
    sampler = FakeSampler()

    # draw_until_admissible only ever returns a positive-weight candidate when one exists within max_attempts
    np.random.seed(0)
    admissible_hits = 0
    for _ in range(200):
        r = draw_until_admissible(sampler, weights, candidates, max_attempts=8, max_retries=1)
        if weights[candidates.index(r)] > 0:
            admissible_hits += 1
    chk("draw_until_admissible: lands on a positive-weight candidate almost always (>=190/200)",
        admissible_hits >= 190)

    uniform = np.ones_like(weights)
    r_u = draw_until_admissible(sampler, uniform, candidates, max_attempts=8, max_retries=1)
    chk("draw_until_admissible: uniform (lesion) weights -- every candidate admissible", uniform[candidates.index(r_u)] > 0)

    # run_session: same noise_seed -> identical reply sequence (the rebuild check); a different one -> not
    s_a = run_session(sampler, weights, weights, candidates, noise_seed=1000, k=6)
    s_b = run_session(sampler, weights, weights, candidates, noise_seed=1000, k=6)
    s_c = run_session(sampler, weights, weights, candidates, noise_seed=1500, k=6)
    chk("run_session: same noise_seed -> same reply sequence", s_a["replies"] == s_b["replies"])
    chk("run_session: different noise_seed -> a different reply sequence", s_a["replies"] != s_c["replies"])

    # run_session must leave the global RNG it was called under untouched (the amendment-3 isolation property)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    run_session(sampler, weights, weights, candidates, noise_seed=999, k=4)
    after = np.random.get_state()[1]
    chk("run_session: leaves the caller's global RNG state untouched", (before == after).all())

    # A degenerate (A/A) world: identical weights in both "arms" must NOT be mistaken for a real effect by
    # construction here (the harness itself does not compute deltas; this checks the ingredient it feeds
    # run_bank_seed-style aggregation is sane: two sessions on the same weights but different streams still vary).
    same_a = run_session(sampler, weights, weights, candidates, noise_seed=2000, k=8)
    same_b = run_session(sampler, weights, weights, candidates, noise_seed=2500, k=8)
    chk("run_session: two independent sessions on the SAME weights need not be identical (real null exists)",
        same_a["replies"] != same_b["replies"] or same_a["session_value"] == same_b["session_value"])

    # THE SCORING FIX ITSELF: scoring must use score_weights (the shared reference), never the lesion's own
    # (uniform) drive weights -- a lesion arm scored against ITS OWN uniform weights is 1.0 by construction on
    # every ask (any candidate has equal weight to the peak), which would hide the whole effect.
    lesion_scored_own = run_session(sampler, uniform, uniform, candidates, noise_seed=3000, k=10)
    chk("run_session: scoring a uniform-drive arm against ITS OWN weights is degenerate (always 1.0) -- the bug "
        "this design avoids by scoring against score_weights instead",
        lesion_scored_own["session_value"] == 1.0)
    lesion_scored_ref = run_session(sampler, uniform, weights, candidates, noise_seed=3000, k=10)
    chk("run_session: scoring the SAME uniform-drive draws against the shared reference is NOT degenerate",
        lesion_scored_ref["session_value"] < 1.0)

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bank-seeds", default=",".join(str(s) for s in DEFAULT_BANK_SEEDS))
    ap.add_argument("--m-sessions", type=int, default=DEFAULT_M)
    ap.add_argument("--k-asks", type=int, default=DEFAULT_K)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--admissible-attempts", type=int, default=DEFAULT_ADMISSIBLE_ATTEMPTS)
    ap.add_argument("--n-cand-max", type=int, default=DEFAULT_N_CAND_MAX)
    ap.add_argument("--weights-json", default=DEFAULT_WEIGHTS_JSON)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    os.environ.setdefault("SIM_BACKEND", "numpy")
    bank_seeds = tuple(int(x) for x in a.bank_seeds.split(","))
    result = run_power_simulation(bank_seeds=bank_seeds, m=a.m_sessions, k=a.k_asks, max_retries=a.max_retries,
                                   admissible_attempts=a.admissible_attempts, n_cand_max=a.n_cand_max,
                                   weights_json=a.weights_json)
    agg = result["aggregate"]
    print("[power-sim] delta_min=%.3f delta_max=%.3f delta_mean=%.3f n_positive=%d/%d"
          % (agg["delta_min"], agg["delta_max"], agg["delta_mean"], agg["n_positive"], agg["n_seeds"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("[power-sim] wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
