#!/usr/bin/env python3
"""Step 0, the deadlock read (PACKET.md docs/drafts/word-role-frame line 62;
RULINGS wrc-frame-go-1020, parallel-builds-1021).

READ-ONLY: no production module is edited. The probe instantiates exactly the
production organ `WordRoleParserOrgan(sorted(facts), seed, *commit_organ_params(0))`
on the iw1 turn-5 fact set (five built-ins + wolf-hunt-deer), and mirrors the
`_commit_run` presentation loop (word_role_parser_organ.py:495-547) locally so it
can (i) also record the role-/gate-inhibitory pools and (ii) inject the packet's
bias titration into the drive window -- the production loop has no bias knob.
Reads go through the organ's own `_commit_read` transducer, unchanged.

Records per seed (packet section 3):
  (a) production read of cat, eat: role/gate/slot counts + inh pools;
  (b) the same read with the exclusion lesion + the dual-role singles pair;
  (c) pos=None commit of every single-role word (13 words);
  (d) bias titration on cat, {0,50,100,200,400,800} pA at pos 0 and pos=None
      into (i) the agent role ensemble, (ii) the agent gate, (iii) negative
      current into the patient ensemble: B_min (smallest bias that latches
      agent alone) and B_false (bias at which a patient-only word latches
      agent, or eat/zorp latches anything wrong);
  (e) eat still commits action with the agent bias held on;
  (f) learned mean weights cat/fish -> agent/action/patient.

GO_RULE -- WRITTEN BEFORE THE PROBE RUNS (this module is committed first):
on >= 14 of 15 dev seeds, a role-layer B_min exists at pos=None with
B_false >= 2 x B_min (B_false absent = infinite margin = pass), and every
single-role word commits its own role at pos=None. If only the gate layer has
a window, the entry moves to the gate layer. If single-role words fail at
pos=None on >= 2 seeds, the pos-free path is not viable as written: STOP and
post the counts. The probe prints the verdict; Claude rules before step 1.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    _HERE = os.path.dirname(_HERE)
    if os.path.isdir(os.path.join(_HERE, "research")):
        break
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from research.runners import word_role_parser_organ as W  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

FACTS = [
    ("brain", "use", "spikes"),
    ("brain", "learn", "words"),
    ("brain", "store", "memory"),
    ("dog", "chase", "cat"),
    ("cat", "eat", "fish"),
    ("wolf", "hunt", "deer"),
]
AGENTS = ("dog", "wolf", "brain")
ACTIONS = ("chase", "eat", "hunt", "use")
PATIENTS = ("deer", "spikes", "memory", "words", "fish")
SINGLE_ROLE = AGENTS + ACTIONS + PATIENTS
PATIENT_ONLY = PATIENTS
UNHEARD = "zorp"
BIAS_GRID = (0, 50, 100, 200, 400, 800)
TARGETS = (("role_agent", 1.0), ("role_patient", -1.0), ("gate_agent", 1.0))
LAYER = {"role_agent": "role", "role_patient": "role", "gate_agent": "gate"}

GO_N = 15
GO_MIN_SEEDS = 14
GO_RULE = (
    ">= 14 of 15 dev seeds: a role-layer B_min exists at pos=None with "
    "B_false >= 2 x B_min (absent B_false = infinite margin = pass) AND every "
    "single-role word commits its own role at pos=None. Gate-layer-only window "
    "-> entry moves to the gate layer. Single-role failures >= 2 seeds -> "
    "STOP, pos-free path not viable as written; post the counts."
)

OWN = {}
for f in FACTS:
    for k, r in enumerate(W.ROLES):
        OWN.setdefault(f[k], r)


def own_role(w):
    return OWN.get(w)


def run_items(organ, seq, *, excl_lesion=False, bias_pa=0.0, bias_target=None):
    """Mirror of WordRoleParserOrgan._commit_run (word_role_parser_organ.py:495-547)
    with inhibitory-pool counters and an optional constant bias on one target
    during the WORD drive window only. Same gates, resets, steps, windows."""
    def _go():
        b, xp, p, c = organ._bridge, organ._xp, organ.p, organ.c
        # production: 0.0 if lesion else 1.0 (the word->role gate is CLOSED
        # only by the word-transmission lesion, which this probe never sets).
        b.set_transmission_gate(W.GATE_WORD_TRANSMISSION, 1.0)
        b.set_transmission_gate(W.GATE_COMMIT_IN, 1.0)
        b.set_transmission_gate(W.GATE_SLOT_LOOP, 1.0)
        b.set_transmission_gate(
            W.GATE_COMMIT_EXCL, 0.0 if excl_lesion else 1.0)
        rdev = {r: xp.asarray(organ.role_idx[r]) for r in W.ROLES}
        gdev = {r: xp.asarray(organ.gate_idx[r]) for r in W.ROLES}
        sdev = {r: xp.asarray(organ.slot_idx[r]) for r in W.ROLES}
        idev = {r: xp.asarray(organ.inh_idx[r]) for r in W.ROLES}
        gidev = {r: xp.asarray(organ.gate_inh_idx[r]) for r in W.ROLES}
        hold = int(c["HOLD_STEPS"])
        read = int(c["READ_STEPS"])
        out = []
        try:
            for i, (word, pos) in enumerate(seq):
                if i == 0:
                    W._reset_quiescent(b, xp, organ._snap)
                else:
                    organ._reset_keep_slots()
                cur = xp.zeros(organ._n, dtype=xp.float32)
                if word is not None:
                    dev = organ._word_dev(word)
                    if dev is not None:
                        cur[dev] = xp.float32(p["DRIVE_PA"])
                if pos is not None:
                    cur[xp.asarray(organ.pos_idx[min(max(int(pos), 0), 2)])] = \
                        xp.float32(p["DRIVE_PA"])
                if bias_pa and bias_target:
                    tname, sign = bias_target
                    tidx = (organ.role_idx["agent"] if tname == "role_agent"
                            else organ.role_idx["patient"]
                            if tname == "role_patient"
                            else organ.gate_idx["agent"])
                    cur[xp.asarray(tidx)] = cur[xp.asarray(tidx)] + \
                        xp.float32(sign * float(bias_pa))
                rc = {r: 0 for r in W.ROLES}
                gc = {r: 0 for r in W.ROLES}
                sc = {r: 0 for r in W.ROLES}
                ih = {r: 0 for r in W.ROLES}
                gih = {r: 0 for r in W.ROLES}
                b.cp_external_input_current[:] = cur
                for _ in range(int(c.get("WORD_STEPS", p["TEST_STEPS"]))):
                    b._run_one_simulation_step()
                    f = b.cp_firing_states
                    for r in W.ROLES:
                        rc[r] += int(W.to_host(
                            f[rdev[r]].astype(xp.float64).sum()))
                        gc[r] += int(W.to_host(
                            f[gdev[r]].astype(xp.float64).sum()))
                        ih[r] += int(W.to_host(
                            f[idev[r]].astype(xp.float64).sum()))
                        gih[r] += int(W.to_host(
                            f[gidev[r]].astype(xp.float64).sum()))
                b.cp_external_input_current[:] = 0.0
                for t in range(hold):
                    b._run_one_simulation_step()
                    f = b.cp_firing_states
                    for r in W.ROLES:
                        gc[r] += int(W.to_host(
                            f[gdev[r]].astype(xp.float64).sum()))
                        ih[r] += int(W.to_host(
                            f[idev[r]].astype(xp.float64).sum()))
                        gih[r] += int(W.to_host(
                            f[gidev[r]].astype(xp.float64).sum()))
                        if t >= hold - read:
                            sc[r] += int(W.to_host(
                                f[sdev[r]].astype(xp.float64).sum()))
                out.append({"word": word, "pos": pos, "role_counts": rc,
                            "gate_counts": gc, "slot_counts": sc,
                            "role_inh_counts": ih, "gate_inh_counts": gih})
        finally:
            b.cp_external_input_current[:] = 0.0
            for gname in (W.GATE_WORD_TRANSMISSION, W.GATE_COMMIT_IN,
                          W.GATE_SLOT_LOOP, W.GATE_COMMIT_EXCL):
                b.set_transmission_gate(gname, 1.0)
        return out
    return W._isolated(_go)


def latched_new(rd, i=0):
    return rd["per_item"][i]["new"]


def latched_role(rd, i=0):
    return rd["per_item"][i]["role"]


def weight_pairs(organ, words):
    """(f): per-word per-role mean learned weight (host read of the synapse
    matrix, same method as WordRoleParserOrgan._weights)."""
    out = {}
    row, col, dat = organ.bridge_coo()
    for w in words:
        idx = organ.word_idx.get(w)
        if idx is None:
            out[w] = None
            continue
        per = {}
        for r in W.ROLES:
            m = np.isin(row, idx) & np.isin(col, organ.role_idx[r])
            per[r] = round(float(dat[m].mean()), 4) if m.any() else None
        out[w] = per
    return out


def seed_probe(seed):
    params, commit = W.commit_organ_params(0)
    organ = W.WordRoleParserOrgan(sorted(FACTS), seed, params, commit).train()
    rec: dict = {"seed": int(seed)}

    # (a) production read of cat, eat (pair) + dual-role singles for (b) pair.
    prod = run_items(organ, [("cat", 0), ("eat", 1)])
    rd_prod = organ._commit_read(prod)
    singles = [("cat", 0), ("cat", 2), ("fish", 0), ("fish", 2)]
    singles_intact = {("%s@%d" % w): run_items(organ, [w])
                      for w in singles}
    rd_intact = {k: organ._commit_read(v)
                 for k, v in singles_intact.items()}
    rec["a_production"] = {"items": prod, "read": rd_prod}

    # (b) the same read with the exclusion lesion + lesioned singles pair.
    prod_l = run_items(organ, [("cat", 0), ("eat", 1)], excl_lesion=True)
    rd_prod_l = organ._commit_read(prod_l)
    singles_l = {k: run_items(organ, [w], excl_lesion=True)
                 for w in singles
                 for k in [("%s@%d" % w)]}
    rd_les = {k: organ._commit_read(v) for k, v in singles_l.items()}
    want = {"cat@0": "agent", "cat@2": "patient",
            "fish@0": "agent", "fish@2": "patient"}
    right_i = sum(1 for k in want
                  if rd_intact[k]["per_item"][0]["role"] == want[k])
    right_l = sum(1 for k in want
                  if rd_les[k]["per_item"][0]["role"] == want[k])
    rec["b_excl_lesion"] = {
        "items": prod_l, "read": rd_prod_l,
        "singles_intact": {k: rd_intact[k] for k in want},
        "singles_lesioned": {k: rd_les[k] for k in want},
        "right_intact": int(right_i), "right_lesioned": int(right_l),
        "attributable": attributable_to(
            "exclusion-lesion right reads (lesioned vs intact)",
            float(right_l), float(right_i)),
    }

    # (c) pos=None commit of every single-role word.
    rec_c = {}
    for w in SINGLE_ROLE:
        items = run_items(organ, [(w, None)])
        rd = organ._commit_read(items)
        rec_c[w] = {"own_role": own_role(w),
                    "got": latched_role(rd), "new": latched_new(rd),
                    "slot_counts": items[0]["slot_counts"],
                    "ok": bool(latched_role(rd) == own_role(w))}
    rec["c_pos_none_singles"] = rec_c

    # (d) bias titration on cat at pos 0 and pos=None.
    titration = {}
    for tname, sign in TARGETS:
        for B in BIAS_GRID:
            for pos in (0, None):
                key = "%s/B%d/pos%s" % (
                    tname, B, "None" if pos is None else pos)
                items = run_items(organ, [("cat", pos)], bias_pa=B,
                                  bias_target=(tname, sign))
                rd = organ._commit_read(items)
                titration[key] = {"role": latched_role(rd),
                                  "new": latched_new(rd),
                                  "slot_counts": items[0]["slot_counts"]}
    rec["d_titration_cat"] = titration

    # B_false probes per target: patient-only words, eat, unheard spare --
    # all at pos=None with the same bias on the same target.
    false_probes = {}
    for tname, sign in TARGETS:
        for B in BIAS_GRID:
            for w in list(PATIENT_ONLY) + ["eat", UNHEARD]:
                organ._spare_map = {}     # production resets per parse
                items = run_items(organ, [(w, None)], bias_pa=B,
                                  bias_target=(tname, sign))
                rd = organ._commit_read(items)
                false_probes["%s/B%d/%s" % (tname, B, w)] = {
                    "role": latched_role(rd), "new": latched_new(rd)}

    def b_min(layer):
        tgts = [t for t in (x[0] for x in TARGETS) if LAYER[t] == layer]
        best = None
        per_target = {}
        for t in tgts:
            got = None
            for B in BIAS_GRID:
                cell = titration.get(
                    "%s/B%d/posNone" % (t, B), {})
                if cell.get("new") == ["agent"] and \
                        cell.get("role") == "agent":
                    got = B
                    break
            per_target[t] = got
            if got is not None:
                best = got if best is None else min(best, got)
        return best, per_target

    def b_false(layer):
        tgts = [t for t in (x[0] for x in TARGETS) if LAYER[t] == layer]
        for B in BIAS_GRID:
            hit = []
            for t in tgts:
                for w in PATIENT_ONLY:
                    cell = false_probes.get(
                        "%s/B%d/%s" % (t, B, w), {})
                    if cell.get("role") == "agent" or \
                            "agent" in (cell.get("new") or []):
                        hit.append({"word": w, "target": t,
                                    "got": cell.get("role")})
                for w in ("eat", UNHEARD):
                    cell = false_probes.get(
                        "%s/B%d/%s" % (t, B, w), {})
                    wrong = (w == "eat" and cell.get("role") in
                             ("agent", "patient")) or \
                            (w != "eat" and cell.get("role") is not None)
                    if wrong:
                        hit.append({"word": w, "target": t,
                                    "got": cell.get("role")})
            if hit:
                return B, hit
        return None, []

    bm_r, bm_r_t = b_min("role")
    bm_g, bm_g_t = b_min("gate")
    bf_r, bf_r_h = b_false("role")
    bf_g, bf_g_h = b_false("gate")
    rec["d_derived"] = {
        "B_min_role_posNone": bm_r, "B_min_role_per_target": bm_r_t,
        "B_min_gate_posNone": bm_g, "B_min_gate_per_target": bm_g_t,
        "B_false_role": bf_r, "B_false_role_hits": bf_r_h,
        "B_false_gate": bf_g, "B_false_gate_hits": bf_g_h,
        "false_probes": false_probes,
    }

    # (e) eat with the agent (role) bias held on, every grid step.
    eat_bias = {}
    for B in BIAS_GRID:
        organ._spare_map = {}
        items = run_items(organ, [("eat", None)], bias_pa=B,
                          bias_target=("role_agent", 1.0))
        rd = organ._commit_read(items)
        eat_bias["B%d" % B] = {"role": latched_role(rd),
                               "ok_action": bool(
                                   latched_role(rd) == "action")}
    rec["e_eat_agent_bias"] = eat_bias

    # (f) learned weights.
    rec["f_weights"] = weight_pairs(organ, ("cat", "fish"))

    # GO verdict for this seed.
    singles_ok = all(v["ok"] for v in rec_c.values())
    role_window = (bm_r is not None and
                   (bf_r is None or bf_r >= 2 * bm_r))
    gate_window = (bm_g is not None and
                   (bf_g is None or bf_g >= 2 * bm_g))
    rec["go"] = {"singles_ok": bool(singles_ok),
                 "role_window": bool(role_window),
                 "gate_window": bool(gate_window),
                 "seed_role_go": bool(role_window and singles_ok),
                 "seed_gate_go": bool(gate_window and singles_ok)}
    return rec


def evaluate(records: list) -> dict:
    n = len(records)
    n_role = sum(1 for r in records if r["go"]["seed_role_go"])
    n_gate = sum(1 for r in records if r["go"]["seed_gate_go"])
    n_single = sum(1 for r in records if r["go"]["singles_ok"])
    if n_single < n - 2:
        verdict = ("STOP: pos-free path not viable as written "
                   "(single-role words fail at pos=None on %d seeds)"
                   % (n - n_single))
    elif n_role >= GO_MIN_SEEDS:
        verdict = "GO: role-layer entry (%d/%d seeds)" % (n_role, n)
    elif n_gate >= GO_MIN_SEEDS:
        verdict = ("GO: only the gate layer has a window -- entry moves to "
                   "the gate layer (%d/%d seeds)" % (n_gate, n))
    else:
        verdict = ("STOP: no window at either layer on >= %d seeds "
                   "(role %d, gate %d, singles %d of %d)"
                   % (GO_MIN_SEEDS, n_role, n_gate, n_single, n))
    return {"go_rule": GO_RULE, "n_seeds": n,
            "n_role_go": n_role, "n_gate_go": n_gate,
            "n_singles_ok": n_single, "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser(description="wrc-frame step 0 deadlock read")
    ap.add_argument("--seeds", default="1-15")
    ap.add_argument("--out",
                    default="research/findings/raw/_wrc_frame/step0")
    args = ap.parse_args(argv)
    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    out = args.out
    os.makedirs(out, exist_ok=True)
    recs: list = []
    for seed in seeds:
        rec = seed_probe(seed)
        path = os.path.join(out, "seed%d.json" % seed)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
        recs.append(rec)
        g = rec["go"]
        d = rec["d_derived"]
        print("seed %d: singles_ok=%s role(B_min=%s B_false=%s)="
              "go=%s gate(B_min=%s)=%s -> %s"
              % (seed, g["singles_ok"], d["B_min_role_posNone"],
                 d["B_false_role"], g["seed_role_go"],
                 d["B_min_gate_posNone"], g["seed_gate_go"],
                 "GO" if g["seed_role_go"] else "no"))
    summary = evaluate(recs)
    summary["seeds"] = seeds
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("VERDICT:", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
