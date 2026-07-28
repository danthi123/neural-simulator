"""INDEPENDENT adversarial control for the gradedtie DISTINGUISHING-test construct validity.

Reuse-by-import ONLY (NO sim/ edit; numpy/CPU). Rebuilds the reservoir + analytic-Dale read-out
from the same primitives the target runner uses, then INDEPENDENTLY:

  (1) Reproduce the AGENT-favouring synthetic ties + THEME-favouring real ties on >=2 seeds.
  (2) GENUINENESS of the AGENT ties: for each, verify with FRESH `predict_spikes` calls that it is
      (a) an EXACT slot0 spike-count tie [n,0,n] (top-2 output spike counts equal), AND
      (b) the graded drive genuinely favours AGENT (drive_AGENT strictly the argmax, with a real margin).
      Also verify the graded drive == the ridge discriminant IN_SCALE*f@W_ridge (a real neural quantity).
  (3) ANSWER-INDEPENDENCE: on the GENUINE AGENT ties, does gradedtie give AGENT (reads the drive)?
      Compare vs the refuted THEME-prior (calibrated) + gainnorm (should give THEME => a disguised prior).
      On THEME ties, all drive-followers give THEME.

  (4) ADVERSARIAL EXTRA: an INDEPENDENT tie construction that does NOT use the builder's directional
      blend (a random-direction perturbation search around real slot0 features) -> collect ties of BOTH
      drive directions and confirm gradedtie tracks the drive both ways (not an artifact of the blend geometry).
"""
from __future__ import annotations
import os, json, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np

import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
import research.runners._rungB1c_objrel_per_role_readout_derisk as PR
import research.runners._rungB1c_objrel_dann_readout_derisk as D
import research.runners._rungB1c_objrel_reservoir_robustness_sweep_derisk as SW
from research.runners._emerge78_reservoir_form_to_role_derisk import (
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX,
)

N_TRAIN = SW.N_TRAIN
N_TEST = SW.N_TEST
N_ROLES3 = SW.N_ROLES3
AGENT = _ROLE_IDX["AGENT"]; THEME = _ROLE_IDX["THEME"]; PRED = _ROLE_IDX["PREDICATE"]


def _graded_drive_indep(ro, f):
    """INDEPENDENT reimplementation of the analog pre-threshold net output drive, from ro's OWN weights.
    Mirrors the analytic Dale readout: drive = IN_SCALE*f @ W_e  +  (IN_SCALE*f @ W_fi) @ W_io.
    Written from scratch (not calling SW._graded_output_drive) so we can cross-check."""
    f_s = np.asarray(f, dtype=np.float64) * float(D.IN_SCALE)
    de = f_s @ ro.W_e.astype(np.float64)
    dih = f_s @ ro.W_fi.astype(np.float64)
    di = dih @ ro.W_io.astype(np.float64)
    return de + di


def _ridge_discriminant(Wr, f):
    """IN_SCALE * f @ W_ridge (the pure linear ridge score) -- what the graded drive should EQUAL for the
    analytic Dale read-out (E=positive rows, I=identity-interneuron negative rows)."""
    f_s = np.asarray(f, dtype=np.float64) * float(D.IN_SCALE)
    return f_s @ Wr.astype(np.float64)[:, :N_ROLES3]


def _spike_counts(ro, f):
    _p, out, _i = ro.predict_spikes(np.asarray(f, dtype=np.float32))
    return out.astype(np.float64)


def _mech_pred(ro, ro_gn, bias0, f, mech, tie_margin=0):
    """Independent reimplementation of each mechanism's slot0 prediction (NOT calling SW._fix_predict_slot0)."""
    o = _spike_counts(ro, f)
    if mech == "gainnorm":
        return int(np.argmax(_spike_counts(ro_gn, f)))
    top2 = np.sort(o)[::-1]
    tied = (top2[0] - top2[1]) <= tie_margin
    if mech == "raw" or not tied:
        return int(np.argmax(o))
    if mech == "gradedtie":
        return int(np.argmax(_graded_drive_indep(ro, f)))
    if mech == "theme_prior":  # the refuted calibrated read: subtract the task-blind per-pool bias on a tie
        return int(np.argmax(o - bias0)) if bias0 is not None else int(np.argmax(o))
    return int(np.argmax(o))


def build(seed, corpus):
    C.WS_BIAS_SCALE_C2 = 0.0
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    trng = np.random.default_rng(seed * 977 + 13)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)
    ub, ens, inh, res, res_idx = PR._build(seed, corpus, enc)
    slot_train = SW._cache_slot_features(res, enc, train, C.RES_T_STEP)
    feat_dim = next(iter(slot_train.values()))[0].shape[1]
    ros = D._analytic_dale_readout(slot_train, feat_dim, seed)
    # the ridge matrix for slot0 (to cross-check graded drive == ridge discriminant)
    X0, y0 = slot_train[0]
    Wr0 = D._ridge_readout(X0, y0, 0.1)
    ro0 = ros[0]
    ref = SW._class_balanced_ref_feat(*slot_train[0])
    ro0_gn, _g = SW._gain_normalize_readout(ro0, ref, target_count=2.0)
    bias0 = SW._pool_bias_vector(ro0, ref)
    return dict(enc=enc, res=res, ros=ros, ro0=ro0, ro0_gn=ro0_gn, bias0=bias0,
                Wr0=Wr0, canon=canon, objr=objr, slot_train=slot_train)


def slot0_features(res, enc, sentences):
    out = []
    for toks, roles in sentences:
        positions = sorted(roles)
        pos0 = positions[0]
        tgt = _ROLE_IDX[roles[pos0]]
        if tgt >= N_ROLES3:
            continue
        f = PR._feature(res, enc, toks)
        out.append((f, tgt))
    return out


def synth_agent_ties(B, seed, verbose=True):
    """INDEPENDENTLY reconstruct AGENT-favouring synthetic ties via convex blends (same construction idea as
    the builder, but reimplemented + fully audited). Returns list of (feat, drive_g, counts)."""
    ro0 = B["ro0"]; res = B["res"]; enc = B["enc"]
    def feat_pool(sentences, want):
        return [f for (f, _t) in slot0_features(res, enc, sentences)
                if int(np.argmax(_graded_drive_indep(ro0, f))) == want]
    agent_strong = feat_pool(B["canon"], AGENT)
    theme_strong = feat_pool(B["objr"], THEME)
    if verbose:
        print(f"    [construct] agent-strong pool n={len(agent_strong)}  theme-strong pool n={len(theme_strong)}")
    ties = []
    for fa in agent_strong:
        for ft in theme_strong:
            for lam in np.linspace(0.02, 0.98, 49):
                fb = ((1.0 - lam) * ft + lam * fa).astype(np.float32)
                o = _spike_counts(ro0, fb)
                top2 = np.sort(o)[::-1]
                if (top2[0] - top2[1]) > 0:
                    continue
                g = _graded_drive_indep(ro0, fb)
                if int(np.argmax(g)) != AGENT:
                    continue
                ties.append((fb, g, o))
                break
        if len(ties) >= 24:
            break
    return ties


def random_dir_ties(B, seed, want, n_max=30, verbose=True):
    """ADVERSARIAL: a construction that does NOT use the AGENT-vs-THEME directional blend. Take real slot0
    features and add small RANDOM-DIRECTION perturbations (scaled to feature magnitude), search for exact
    count-ties, and keep those whose drive-argmax is `want`. If gradedtie merely followed the blend geometry
    (not the actual drive), it would FAIL here."""
    ro0 = B["ro0"]; res = B["res"]; enc = B["enc"]
    rng = np.random.default_rng(seed * 71 + 3)
    base = [f for (f, _t) in slot0_features(res, enc, B["canon"]) + slot0_features(res, enc, B["objr"])]
    fmag = np.mean([np.linalg.norm(f) for f in base]) + 1e-12
    ties = []
    for f0 in base:
        for _try in range(400):
            d = rng.standard_normal(len(f0)).astype(np.float32)
            d[-1] = 0.0  # leave the +1 bias element fixed
            d = d / (np.linalg.norm(d) + 1e-12) * (fmag * rng.uniform(0.01, 0.20))
            fb = (np.asarray(f0, np.float32) + d).astype(np.float32)
            if np.any(fb[:-1] < 0):  # keep the feature physical (spike-rate >= 0); bias element stays 1
                continue
            o = _spike_counts(ro0, fb)
            top2 = np.sort(o)[::-1]
            if (top2[0] - top2[1]) > 0:
                continue
            g = _graded_drive_indep(ro0, fb)
            if int(np.argmax(g)) != want:
                continue
            ties.append((fb, g, o))
            break
        if len(ties) >= n_max:
            break
    return ties


def audit_ties(B, ties, favour, label):
    """GENUINENESS audit: for each tie verify (a) FRESH predict_spikes gives an EXACT count tie [n,0,n]
    (top-2 equal), (b) drive argmax == favour with a REAL margin, (c) graded_drive == ridge discriminant."""
    ro0 = B["ro0"]; Wr0 = B["Wr0"]
    n = len(ties)
    exact_tie = 0; drive_ok = 0; ridge_match = 0
    margins = []; count_shape_ok = 0
    max_ridge_err = 0.0
    for (f, g_stored, o_stored) in ties:
        # (a) FRESH spike-count check (do not trust the stored o)
        o = _spike_counts(ro0, f)
        top2 = np.sort(o)[::-1]
        is_tie = (top2[0] - top2[1]) == 0
        exact_tie += int(is_tie)
        # count shape: PRED (idx1) should be 0 and AGENT==THEME (the [n,0,n] shape); verify the two tied are AGENT,THEME
        tied_roles = set(np.flatnonzero(o == o.max()).tolist())
        count_shape_ok += int(o[PRED] < o.max() and {AGENT, THEME}.issubset(tied_roles))
        # (b) FRESH drive check
        g = _graded_drive_indep(ro0, f)
        drv_arg = int(np.argmax(g))
        drive_ok += int(drv_arg == favour)
        # drive margin between favoured and the runner-up among {AGENT,THEME}
        other = THEME if favour == AGENT else AGENT
        margins.append(float(g[favour] - g[other]))
        # (c) graded drive must equal ridge discriminant IN_SCALE*f@Wridge
        gr = _ridge_discriminant(Wr0, f)
        err = float(np.max(np.abs(g - gr)))
        max_ridge_err = max(max_ridge_err, err)
        ridge_match += int(err < 1e-6)
    return dict(
        label=label, n=n, favour=int(favour),
        exact_count_tie=exact_tie, count_shape_n0n=count_shape_ok,
        drive_favours_target=drive_ok, ridge_matches_drive=ridge_match,
        max_ridge_err=max_ridge_err,
        drive_margin_min=(round(min(margins), 6) if margins else None),
        drive_margin_mean=(round(float(np.mean(margins)), 6) if margins else None),
    )


def eval_mechs(B, ties, favour):
    ro0 = B["ro0"]; ro0_gn = B["ro0_gn"]; bias0 = B["bias0"]
    res = {}
    for mech in ("raw", "gradedtie", "theme_prior", "gainnorm"):
        hits = sum(int(_mech_pred(ro0, ro0_gn, bias0, f, mech) == favour) for (f, _g, _o) in ties)
        res[mech] = round(hits / max(len(ties), 1), 3)
    return res


def run(seeds):
    corpus = C.setup_corpus(seed=42)
    rows = []
    for s in seeds:
        print(f"\n=== SEED {s} ===", flush=True)
        B = build(s, corpus)
        at = synth_agent_ties(B, s)
        tt = [(f, _graded_drive_indep(B["ro0"], f), _spike_counts(B["ro0"], f))
              for (f, tgt) in slot0_features(B["res"], B["enc"], B["objr"])
              if (lambda o: (np.sort(o)[::-1][0] - np.sort(o)[::-1][1]) == 0)(_spike_counts(B["ro0"], f))
              and int(np.argmax(_graded_drive_indep(B["ro0"], f))) == THEME]
        aud_a = audit_ties(B, at, AGENT, "synth_agent")
        aud_t = audit_ties(B, tt, THEME, "real_theme")
        mech_a = eval_mechs(B, at, AGENT)
        mech_t = eval_mechs(B, tt, THEME)
        # adversarial random-direction ties (both directions)
        ra = random_dir_ties(B, s, AGENT)
        rt = random_dir_ties(B, s, THEME)
        aud_ra = audit_ties(B, ra, AGENT, "rand_agent")
        aud_rt = audit_ties(B, rt, THEME, "rand_theme")
        mech_ra = eval_mechs(B, ra, AGENT)
        mech_rt = eval_mechs(B, rt, THEME)
        print(f"  synth AGENT ties n={aud_a['n']}: exact-tie {aud_a['exact_count_tie']}/{aud_a['n']} "
              f"shape[n,0,n] {aud_a['count_shape_n0n']}/{aud_a['n']} drive->AGENT {aud_a['drive_favours_target']}/{aud_a['n']} "
              f"ridge==drive {aud_a['ridge_matches_drive']}/{aud_a['n']} (maxerr {aud_a['max_ridge_err']:.2e}) "
              f"drive-margin min {aud_a['drive_margin_min']} mean {aud_a['drive_margin_mean']}")
        print(f"    mechs on AGENT ties (frac->AGENT): {mech_a}")
        print(f"  real THEME ties n={aud_t['n']}: exact-tie {aud_t['exact_count_tie']}/{aud_t['n']} "
              f"drive->THEME {aud_t['drive_favours_target']}/{aud_t['n']} ridge==drive {aud_t['ridge_matches_drive']}/{aud_t['n']}")
        print(f"    mechs on THEME ties (frac->THEME): {mech_t}")
        print(f"  [ADV] rand-dir AGENT ties n={aud_ra['n']}: exact-tie {aud_ra['exact_count_tie']}/{aud_ra['n']} "
              f"drive->AGENT {aud_ra['drive_favours_target']}/{aud_ra['n']} -> mechs {mech_ra}")
        print(f"  [ADV] rand-dir THEME ties n={aud_rt['n']}: exact-tie {aud_rt['exact_count_tie']}/{aud_rt['n']} "
              f"drive->THEME {aud_rt['drive_favours_target']}/{aud_rt['n']} -> mechs {mech_rt}")
        rows.append(dict(seed=s, audit_agent=aud_a, audit_theme=aud_t, mech_agent=mech_a, mech_theme=mech_t,
                         audit_rand_agent=aud_ra, audit_rand_theme=aud_rt,
                         mech_rand_agent=mech_ra, mech_rand_theme=mech_rt))
    outp = os.environ.get("CTRL_OUT", "control_distinguish_result.json")
    with open(outp, "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\nwrote {outp}")
    return rows


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [103, 104]
    run(seeds)
