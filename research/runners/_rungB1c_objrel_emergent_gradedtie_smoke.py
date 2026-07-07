"""RUNG B-1c OBJREL SMOKE: thread the answer-independent `gradedtie` tie-break into the EMERGENT LEARNED read-out.

CONTEXT (established; NOT re-derived here).
  * The ANALYTIC Dale read + `gradedtie` (in _rungB1c_objrel_reservoir_robustness_sweep_derisk, `--read gradedtie`)
    genuinely closes objrel-slot0 on all 10 seeds ANSWER-INDEPENDENTLY: on a slot0 spike-count TIE (the saturation
    [4,0,4] failure -> argmax defaults to AGENT) it breaks the tie by the argmax of the ACTUAL graded output DRIVE
    (the sub-threshold membrane the spike count quantizes away, computed from the read-out's OWN weights -- gives
    whichever role the DRIVE favours, AGENT or THEME, NOT a THEME prior).
  * The EMERGENT LEARNED read-out (_rungB1c_objrel_dopamine_plasticity_derisk: DopaminePlasticReadout = N_ROLES3
    per-role Dale-legal spiking BinaryRoleDetectors, each learned from a RANDOM init by a graded reward-modulated
    delta rule -- NOT the ridge) also produces per-role output-LIF spike COUNTS and argmaxes them, so the SAME
    saturation-tie can afflict it on 103/104. It has its OWN residual: a per-seed init-basin fragility on 45/101
    (a LEARNING-basin issue, distinct from the read-TIE).

THE BUILD (this file). Apply the gradedtie tie-break to the EMERGENT LEARNED read-out's OUTPUT read: on a slot0
spike-count tie, break by the argmax of the ANSWER-INDEPENDENT graded output drive computed from the LEARNED
detectors' OWN weights. The learned read-out stays EMERGENT (delta-rule from random init -- NOT the analytic ridge;
the gradedtie tie-break uses the LEARNED weights' graded drive, not the ridge). Clean override, reuse-by-import;
NO sim/ edit.

  The emergent per-role graded drive (answer-independent, from the LEARNED weights). Each BinaryRoleDetector `det`,
  after training + `_rebuild_dale`, deploys as: fpos = f @ det._wpos (E path >= 0); fneg = f @ det._wneg (drives the
  inhibitory interneuron); output per-step drive = (fpos + det._tonic) - inh_spike; the output spike COUNT quantizes
  this. The ANALOG (pre-threshold, quantization-free) net output drive is therefore
      g_r = fpos - fneg + det._tonic       (interneuron replaced by its CONTINUOUS activation fneg)
  -- the exact analog membrane the count read quantizes away, derived PURELY from the learned detector's own weights
  (no labels, no 'THEME'). The per-slot graded-drive vector g = [g_0, g_1, g_2] over the N_ROLES3 role detectors; the
  tie-break = argmax(g). This mirrors _graded_output_drive in the sweep runner, adapted to the emergent per-role arch
  (per-detector single-output weights + a learned _tonic, vs the DANN's shared 3-output W_e/W_fi/W_io).

SMOKE (then STOP). On {103,104,45,101,42,100}: compare the emergent learned read-out objrel-slot0 WITH vs WITHOUT
gradedtie (RAW = the causal control). Reports per-seed WITH/WITHOUT + canon; the honest end-to-end emergent close
count (N of the smoke seeds with emergent+gradedtie objrel-slot0 >= 0.9); whether 45/101 remain a separate LEARNING
residual (the tie-break only breaks read-TIES; a learning-basin miss with NO tie is unmoved -> reported honestly).

ANTI-CHEATS carried from the dopamine runner (emergent + reward-load-bearing + held-out + Dale-legal + canon):
  * EMERGENT: the read-out is DopaminePlasticReadout trained by the graded reward-modulated delta from a random Dale
    init (NOT ridge-warm-started). Also reports PRE-learning (epochs=0) objrel WITH gradedtie -> should stay ~chance
    (the tie-break does NOT manufacture the signal; the plasticity does the real work).
  * REWARD LOAD-BEARING: a no-reward (DA==0) emergent read + gradedtie -> should NOT recover (the tie-break cannot
    rescue an un-learned read-out; the graded drive of un-learned weights carries no role signal).
  * HELD-OUT: test facts from a DISTINCT rng (DP.run_seed's canon/objr split); Dale-legal asserted per detector.
  * RAW is the causal control for the tie-fix (WITHOUT gradedtie must still fail on the read-TIE seeds).

Run:
  SIM_BACKEND=numpy python -u -m research.runners._rungB1c_objrel_emergent_gradedtie_smoke \
      --seeds 103 104 45 101 42 100 \
      --json research/findings/raw/_rungB1c_objrel_emergent_gradedtie_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C  # noqa: E402
import research.runners._rungB1c_objrel_per_role_readout_derisk as PR  # noqa: E402
import research.runners._rungB1c_objrel_dann_readout_derisk as D  # noqa: E402
import research.runners._rungB1c_objrel_dopamine_plasticity_derisk as DP  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX,
)

N_ROLES3 = DP.N_ROLES3
N_TRAIN = DP.N_TRAIN
N_TEST = DP.N_TEST


# ── the EMERGENT per-role answer-independent graded output drive (from the LEARNED detectors' OWN weights) ──────────
def _emergent_graded_drive(readout, f):
    """The ANALOG (pre-threshold) net OUTPUT drive per role for an EMERGENT DopaminePlasticReadout `readout`, computed
    from each role detector's OWN LEARNED weights (ANSWER-INDEPENDENT -- no test labels, no 'THEME'). For each role r,
    the trained+rebuilt BinaryRoleDetector deploys per-step output drive (fpos + tonic) - inh_spike; the ANALOG
    quantization-free membrane is fpos - fneg + tonic (interneuron replaced by its continuous activation fneg), where
    fpos = f @ det._wpos, fneg = f @ det._wneg. This is the exact analog quantity the spike COUNT quantizes away --
    the same construction as the sweep runner's _graded_output_drive, adapted to the emergent per-detector arch.
    Returns a (N_ROLES3,) net analog output membrane."""
    g = np.zeros(N_ROLES3, dtype=np.float64)
    fa = np.asarray(f, dtype=np.float64)
    for r in range(N_ROLES3):
        det = readout.det[r]
        fpos = float(fa @ det._wpos.astype(np.float64))     # E-path analog drive (>= 0)
        fneg = float(fa @ det._wneg.astype(np.float64))     # I-path analog drive (carried by the interneuron)
        g[r] = fpos - fneg + float(det._tonic)              # net analog output membrane (fpos - fneg + tonic)
    return g


def _score_emergent(ros, res, enc, sentences, gradedtie, tie_margin=0):
    """Deploy the EMERGENT learned read (per-slot argmax over the role detectors' OUTPUT SPIKE COUNTS). If `gradedtie`,
    on an EXACT slot0 spike-count TIE (top-2 counts equal within `tie_margin`) break it by argmax of the ANSWER-
    INDEPENDENT emergent graded drive (`_emergent_graded_drive`); off a tie -> RAW spike-count argmax; slots 1/2 always
    RAW (canonical never perturbed). `gradedtie=False` = the RAW causal control (still fails on the read-TIE seeds).
    Also counts how many slot0 examples were TIED (diagnostic: read-TIE vs learning-basin residual). Returns
    (overall_acc, slot0_acc, per_slot_hits, per_slot_tot, n_slot0, n_slot0_tie)."""
    ok = tot = s0ok = s0t = 0
    n_s0_tie = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    for toks, roles in sentences:
        f = PR._feature(res, enc, toks)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:
                continue
            if k not in ros:
                continue
            _pred, out, _inh = ros[k].predict_spikes(f)          # RAW per-role output spike count (genuinely spiking)
            o = out.astype(np.float64)
            if k == 0:                                            # slot0: the ambiguous THEME/AGENT slot
                top2 = np.sort(o)[::-1]
                tied = (top2[0] - top2[1]) <= tie_margin
                if tied:
                    n_s0_tie += 1
                if gradedtie and tied:
                    g = _emergent_graded_drive(ros[0], f)         # break by the ACTUAL graded drive (answer-independent)
                    pred = int(np.argmax(g))
                else:
                    pred = int(np.argmax(o))
            else:
                pred = int(np.argmax(o))                          # slots 1/2 always RAW -> canonical untouched
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return (ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot, s0t, n_s0_tie)


def _build_emergent(seed, corpus):
    """Build the byte-identical c2 reservoir (FROZEN) + cache the spiking feature + the held-out canon/objrel test
    (DP.run_seed's exact recipe), and train the EMERGENT dopamine read-outs (main / pre-learning / no-reward). Returns
    (res, enc, canon, objr, ros_main, ros_pre, ros_nr, dale_legal, slot0_counts)."""
    C.WS_BIAS_SCALE_C2 = 0.0
    C.WS_REPLAY = PR.WS_REPLAY
    C.READ_T_STEP_C2 = PR.READ_T_STEP
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    trng = np.random.default_rng(seed * 977 + 13)              # DISTINCT rng => test facts held out (no leakage)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    ub, ens, inh, res, res_idx = PR._build(seed, corpus, enc)
    slot_train = D._cache_slot_features(res, enc, train)
    feat_dim = next(iter(slot_train.values()))[0].shape[1]
    slot0_counts = np.bincount(slot_train[0][1], minlength=N_ROLES3).tolist() if 0 in slot_train else []

    # MAIN emergent read-out: the graded reward-modulated delta from a random Dale init (salience on, reward on) -- NOT
    # the ridge (the emergent constraint). Same call DP.run_seed's main path uses.
    ros_main = DP._train_dopamine(slot_train, feat_dim, seed, epochs=DP.EPOCHS, salience=True, reward_on=True)
    # PRE-LEARNING (epochs=0 random Dale init) -- the tie-break must NOT manufacture the signal from an unlearned read.
    ros_pre = DP._train_dopamine(slot_train, feat_dim, seed, epochs=0)
    # NO-REWARD (DA==0) -- the tie-break must NOT rescue an un-learned read-out (reward load-bearing under the tie-break).
    ros_nr = DP._train_dopamine(slot_train, feat_dim, seed, epochs=DP.EPOCHS, salience=True, reward_on=False)

    dale = [ro.dale_legal() for ro in ros_main.values()]
    dale_legal = all(dd["legal"] for dd in dale)
    return res, enc, canon, objr, ros_main, ros_pre, ros_nr, dale_legal, slot0_counts


def run_seed(seed, corpus):
    t0 = time.time()
    (res, enc, canon, objr, ros_main, ros_pre, ros_nr,
     dale_legal, slot0_counts) = _build_emergent(seed, corpus)

    # MAIN emergent read -- WITHOUT gradedtie (RAW causal control) vs WITH gradedtie
    raw_o_acc, raw_o_s0, _rop, _rot, n_s0, n_s0_tie = _score_emergent(ros_main, res, enc, objr, gradedtie=False)
    gt_o_acc, gt_o_s0, _gop, _got, _n2, _t2 = _score_emergent(ros_main, res, enc, objr, gradedtie=True)
    # canonical (sanity: must stay high under both; slots 1/2 are always RAW so gradedtie only touches slot0 ties)
    raw_c_acc, raw_c_s0, _rcp, _rct, _nc, _tc = _score_emergent(ros_main, res, enc, canon, gradedtie=False)
    gt_c_acc, gt_c_s0, _gcp, _gct, _nc2, _tc2 = _score_emergent(ros_main, res, enc, canon, gradedtie=True)

    # PRE-LEARNING (epochs=0) WITH gradedtie -> should stay ~chance (the plasticity does the work, not the tie-break)
    pre_o_acc, pre_o_s0, _pop, _pot, _pn, _pt = _score_emergent(ros_pre, res, enc, objr, gradedtie=True)
    # NO-REWARD WITH gradedtie -> should NOT recover (reward load-bearing under the tie-break)
    nr_o_acc, nr_o_s0, _nop, _not, _nn, _nt = _score_emergent(ros_nr, res, enc, objr, gradedtie=True)

    elapsed = round(time.time() - t0, 1)
    d = {
        "seed": int(seed), "slot0_class_counts": slot0_counts,
        "n_slot0_objrel": int(n_s0), "n_slot0_objrel_ties": int(n_s0_tie),
        "emergent_raw": {                       # WITHOUT gradedtie (causal control)
            "objrel_slot0_THEME": round(raw_o_s0, 3), "objrel_acc": round(raw_o_acc, 3),
            "canonical_acc": round(raw_c_acc, 3), "canonical_slot0": round(raw_c_s0, 3),
        },
        "emergent_gradedtie": {                 # WITH gradedtie
            "objrel_slot0_THEME": round(gt_o_s0, 3), "objrel_acc": round(gt_o_acc, 3),
            "canonical_acc": round(gt_c_acc, 3), "canonical_slot0": round(gt_c_s0, 3),
        },
        "pre_learning_gradedtie": {"objrel_slot0_THEME": round(pre_o_s0, 3), "objrel_acc": round(pre_o_acc, 3)},
        "no_reward_gradedtie": {"objrel_slot0_THEME": round(nr_o_s0, 3), "objrel_acc": round(nr_o_acc, 3)},
        "dale_legal": bool(dale_legal),
        "elapsed_s": elapsed,
        # per-seed flags
        "raw_closes_slot0": bool(raw_o_s0 >= 0.90),          # was the read-TIE the only residual? (RAW alone closes)
        "gradedtie_closes_slot0": bool(gt_o_s0 >= 0.90),     # the honest end-to-end emergent close (per seed)
        "gradedtie_helps": bool(gt_o_s0 - raw_o_s0 >= 0.05), # the tie-break lifted objrel-slot0
        "canonical_not_regressed": bool(gt_c_acc >= 0.90),
        "learning_does_work": bool(gt_o_s0 - pre_o_s0 >= 0.15),   # gradedtie does NOT manufacture from an unlearned read
        "reward_load_bearing": bool(gt_o_s0 - nr_o_s0 >= 0.15),   # gradedtie does NOT rescue a no-reward read
    }
    return d


def _print_seed(s, d):
    raw = d["emergent_raw"]; gt = d["emergent_gradedtie"]
    pre = d["pre_learning_gradedtie"]; nr = d["no_reward_gradedtie"]
    print(f"[seed {s}] slot0-cls {d['slot0_class_counts']} | objrel slot0 TIES {d['n_slot0_objrel_ties']}/{d['n_slot0_objrel']} | "
          f"EMERGENT-RAW objrel-slot0 {raw['objrel_slot0_THEME']:.2f} (canon {raw['canonical_acc']:.2f}) -> "
          f"EMERGENT+GRADEDTIE objrel-slot0 {gt['objrel_slot0_THEME']:.2f} (canon {gt['canonical_acc']:.2f}) | "
          f"PRE-LEARN+GT {pre['objrel_slot0_THEME']:.2f} | NO-REWARD+GT {nr['objrel_slot0_THEME']:.2f} | "
          f"[raw-closes {d['raw_closes_slot0']} gt-closes {d['gradedtie_closes_slot0']} gt-helps {d['gradedtie_helps']} "
          f"canon-ok {d['canonical_not_regressed']} learn-work {d['learning_does_work']} reward-LB {d['reward_load_bearing']} "
          f"dale-legal {d['dale_legal']}] ({d['elapsed_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[103, 104, 45, 101, 42, 100])
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_rungB1c_objrel_emergent_gradedtie_smoke.json")
    args = ap.parse_args()

    t0 = time.time()
    corpus = C.setup_corpus(seed=42)
    print(f"[emergent-gradedtie SMOKE] corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])} | thread the "
          f"ANSWER-INDEPENDENT gradedtie tie-break into the EMERGENT LEARNED read-out (DopaminePlasticReadout, delta-rule "
          f"from a random Dale init -- NOT the ridge). On a slot0 spike-count TIE, break by argmax of the graded output "
          f"drive from the LEARNED detectors' OWN weights (fpos - fneg + tonic). RAW = the causal control. seeds "
          f"{args.seeds}. NO sim/ edit; CPU/numpy. SMOKE (controller fans out + verifies).", flush=True)

    rows = []
    for s in args.seeds:
        d = run_seed(s, corpus)
        rows.append(d)
        _print_seed(s, d)

    n_gt_close = sum(r["gradedtie_closes_slot0"] for r in rows)
    n_raw_close = sum(r["raw_closes_slot0"] for r in rows)
    n_gt_helps = sum(r["gradedtie_helps"] for r in rows)
    canon_all_ok = all(r["canonical_not_regressed"] for r in rows)
    dale_all = all(r["dale_legal"] for r in rows)
    learn_all = all(r["learning_does_work"] for r in rows)
    reward_all = all(r["reward_load_bearing"] for r in rows)

    def _m(k1, k2):
        return round(float(np.mean([r[k1][k2] for r in rows])), 3)

    agg = {
        "mode": "emergent_gradedtie_smoke",
        "seeds": [int(r["seed"]) for r in rows],
        "n_seeds": len(rows),
        "n_gradedtie_closes_slot0": int(n_gt_close),         # honest end-to-end emergent close count on the SMOKE seeds
        "n_raw_closes_slot0": int(n_raw_close),
        "n_gradedtie_helps": int(n_gt_helps),
        "mean_objrel_slot0_emergent_raw": _m("emergent_raw", "objrel_slot0_THEME"),
        "mean_objrel_slot0_emergent_gradedtie": _m("emergent_gradedtie", "objrel_slot0_THEME"),
        "mean_canonical_emergent_gradedtie": _m("emergent_gradedtie", "canonical_acc"),
        "mean_objrel_slot0_pre_learning_gradedtie": _m("pre_learning_gradedtie", "objrel_slot0_THEME"),
        "mean_objrel_slot0_no_reward_gradedtie": _m("no_reward_gradedtie", "objrel_slot0_THEME"),
        "per_seed_gradedtie_closes": {int(r["seed"]): bool(r["gradedtie_closes_slot0"]) for r in rows},
        "per_seed_raw_closes": {int(r["seed"]): bool(r["raw_closes_slot0"]) for r in rows},
        "per_seed_slot0_ties": {int(r["seed"]): [int(r["n_slot0_objrel_ties"]), int(r["n_slot0_objrel"])] for r in rows},
        "canonical_not_regressed_all": bool(canon_all_ok),
        "dale_legal_all": bool(dale_all),
        "learning_does_work_all": bool(learn_all),
        "reward_load_bearing_all": bool(reward_all),
        "epochs": DP.EPOCHS, "lr": DP.LR, "read_t": DP.READ_T,
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\n[emergent-gradedtie SMOKE] SUMMARY (smoke seeds {agg['seeds']}): EMERGENT+GRADEDTIE closes objrel-slot0 "
          f">=0.90 on {n_gt_close}/{len(rows)} (RAW alone closes {n_raw_close}/{len(rows)}); gradedtie helps on "
          f"{n_gt_helps}/{len(rows)}. mean objrel-slot0 RAW {agg['mean_objrel_slot0_emergent_raw']:.2f} -> "
          f"GRADEDTIE {agg['mean_objrel_slot0_emergent_gradedtie']:.2f} (canon {agg['mean_canonical_emergent_gradedtie']:.2f}). "
          f"PRE-LEARN+GT {agg['mean_objrel_slot0_pre_learning_gradedtie']:.2f} | NO-REWARD+GT "
          f"{agg['mean_objrel_slot0_no_reward_gradedtie']:.2f}. canon-ok {canon_all_ok} dale-legal {dale_all} "
          f"learn-work {learn_all} reward-LB {reward_all}.", flush=True)
    print(f"[emergent-gradedtie SMOKE] per-seed close (gradedtie): {agg['per_seed_gradedtie_closes']}", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[emergent-gradedtie SMOKE] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
