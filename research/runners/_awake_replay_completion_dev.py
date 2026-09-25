"""DEV-SEED instrument for the awake-replay completion arc (branch research/awake-replay-completion). NOT a gate runner.

WHY. The awake-rest replay route (webapp/awake_replay_capture.py) scored NO-GO 5/6
(research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md): on one gate seed the reactivation read R of the
long-delay fact starts low and collapses across the 48 rest bouts, because each bout's re-induction is proportional to
the current read (e <- e + R (1 - e)), so a weak trace is subcritical. This instrument works ONLY on development seeds
(never 42/43/44/100/101/102): it builds the tiny-demo brain in-process through the real `webapp.server.brain_chat`
handler, tells the fact exactly as the battery's `datr` group does, and then

  1. maps the read R as a function of the block's early-phase expression e (the store block is rewritten as
     base + e * inc for each e on a grid, read, and restored -- read-only for the ledger), together with the per-role
     decoded words and margins, and the completion read of webapp/replay_completion.py when a completion flag is armed;
  2. runs the battery's own `awake_rest_4h` world step (48 idle ticks of 5 min), the night and the recall question,
     and records every awake bout, the sleep epoch and the recall outcome.

It refuses a gate seed. Output: one JSON per seed. Use under tools/memcap.sh (one tiny-demo build, ~1 GB).

  bash tools/mem_ok.sh 8 4 && bash tools/memcap.sh 8 -- .venv/bin/python -u -m \
      research.runners._awake_replay_completion_dev --seed 7 --arm arc \
      --out research/findings/raw/_awake_replay_completion_dev/scan/s7_arc.json

ARMS (env on top of ON + RC, as the arc family; "both" = BRAIN_AWAKE_REPLAY_COMPLETION + BRAIN_SLEEP_REPLAY_COMPLETION):
arc (the Amendment-4 route), arcc (both), arcc_awake (awake completion only), arcc_sleep (night completion only),
arcc_nocomp (both flags + BRAIN_REPLAY_COMPLETION_LESION: every read runs, both routes use R), arcc_lesion (both,
awake edge cut), noarc_c (no awake route, night completion), arcc_dalesion (both + BRAIN_DA_ENCODING_LESION),
arcc_sleeplesion (both + BRAIN_SLEEP_REPLAY_CAPTURE_LESION), arcc_norest (both, the datl group: 4 h awake without an
idle tick), arcc_late / arc_late (the datz group: 3 h without rest, then 1 h of rest), off (ledger off).
Artifacts under research/findings/raw/_awake_replay_completion_dev/arms/ were run at commit 938ee5ad5, where `arcc`
meant the AWAKE completion only (== `arcc_awake` here) and `arcc_nocomp` used BRAIN_AWAKE_REPLAY_COMPLETION_LESION
(the lesion's former name; the night used R then as now). arms_both/ holds the runs of this version.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
ON = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
RC = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1"}
ARC = {"BRAIN_AWAKE_REPLAY_CAPTURE": "1"}
CA = {"BRAIN_AWAKE_REPLAY_COMPLETION": "1"}
CS = {"BRAIN_SLEEP_REPLAY_COMPLETION": "1"}
LES = {"BRAIN_REPLAY_COMPLETION_LESION": "1"}
ARMS = {
    "arc": ({**ON, **RC, **ARC}, "datr"),
    "arcc": ({**ON, **RC, **ARC, **CA, **CS}, "datr"),
    "arcc_awake": ({**ON, **RC, **ARC, **CA}, "datr"),
    "arcc_sleep": ({**ON, **RC, **ARC, **CS}, "datr"),
    "arcc_nocomp": ({**ON, **RC, **ARC, **CA, **CS, **LES}, "datr"),
    "arcc_lesion": ({**ON, **RC, **ARC, **CA, **CS, "BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1"}, "datr"),
    "noarc": ({**ON, **RC}, "datr"),
    "noarc_c": ({**ON, **RC, **CS}, "datr"),
    "arcc_dalesion": ({**ON, **RC, **ARC, **CA, **CS, "BRAIN_DA_ENCODING_LESION": "1"}, "datr"),
    "arcc_sleeplesion": ({**ON, **RC, **ARC, **CA, **CS, "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"}, "datr"),
    "arcc_norest": ({**ON, **RC, **ARC, **CA, **CS}, "datl"),
    "arcc_late": ({**ON, **RC, **ARC, **CA, **CS}, "datz"),
    "arc_late": ({**ON, **RC, **ARC}, "datz"),
    "off": ({"BRAIN_DA_TAG_CAPTURE": "0", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}, "datr"),
}
E_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.0)
_K_DEV = 900000                         # private-RNG stream offset for the curve reads (disjoint from every route's)
_K_DEV_SCORES = 950000                  # ... and for the recorded score vectors (`--record-scores`)


def _set_block_expression(L, comp, e):
    """Rewrite the ledger's first managed block as base + e * inc (read-only use; restored by L.advance)."""
    blk = L.blocks[0]
    D = comp.D
    j = L.block_offset
    w = blk["base"] + float(e) * blk["inc"]
    comp.store_conns[j * D:(j + 1) * D] = [(p, q, complex(w[k])) for k, (p, q) in enumerate(blk["pq"])]
    comp._store_dirty = True
    comp._store_csr = None
    comp._persistent_dirty = True
    if getattr(comp, "_csr_cache", None) is not None:
        comp._csr_cache = {}


def curve(seed, chat, grid=E_GRID, record_scores=False):
    from webapp.da_tag_capture_chat import _private_rng, store_composer
    from webapp.sleep_replay_capture import reactivation_strength
    cap = chat._da_tag_capture
    L = cap.ledger
    comp = store_composer(chat)
    j = L.block_offset
    comp_mod = None
    if any(os.environ.get(f, "0").strip().lower() in ("1", "true", "on", "yes")
           for f in ("BRAIN_AWAKE_REPLAY_COMPLETION", "BRAIN_SLEEP_REPLAY_COMPLETION")):
        from webapp import replay_completion as comp_mod
    rows = []
    for n, e in enumerate(grid):
        _set_block_expression(L, comp, e)
        with _private_rng(seed, _K_DEV + 2 * n):
            r = reactivation_strength(comp, j)
        with _private_rng(seed, _K_DEV + 2 * n):
            roles = comp._block_role_scores(j)
        row = {"e": e, "R": r,
               "roles": {k: {"word": v[0], "margin": (None if v[2] is None else round(float(v[2]), 9))}
                         for k, v in roles.items()}}
        if comp_mod is not None:
            with _private_rng(seed, _K_DEV + 2 * n + 1):
                row["completion"] = comp_mod.completion_read(comp, j, L.blocks[0])
        if record_scores:
            # the per-role matched-filter drive the completion's item competition receives (the rectified cleanup
            # membrane, `replay_completion._role_scores`), so the competition can be studied offline on the recorded
            # vectors; its own private-RNG stream (the read is a deterministic substrate op)
            from webapp import replay_completion as _rc
            with _private_rng(seed, _K_DEV_SCORES + n):
                sc = _rc._role_scores(comp, j)
            row["scores"] = (None if sc is None else
                             {k: [round(float(x), 9) for x in v[0]] for k, v in sc.items() if k in _rc.COMPLETION_ROLES})
        rows.append(row)
    L.advance(comp, L.t)                   # restore the store exactly as the ledger holds it
    return rows


SCAN_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.065)   # seeds 1-12: (1.0, 0.9, 0.7, 0.5, 0.3)


def run(seed, arm, out, scan=False, grid=None, record_scores=False):
    if int(seed) in GATE_SEEDS:
        raise SystemExit("⛔ %d is a GATE seed; this dev instrument refuses it (develop on dev seeds only)" % seed)
    env, group = ARMS[arm]
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    for k, v in env.items():
        os.environ[k] = v
    from research.runners import onebrain_regression_battery as OB
    from research.runners._da_tag_capture_chat_probe import corpus_missing, outcome
    if corpus_missing() and os.environ.get("LB_ALLOW_NO_CORPUS") != "1":
        raise SystemExit("⛔ data/corpus/ missing %s -- link the corpus first" % corpus_missing())
    from webapp.server import brain_chat, BrainChatRequest
    from webapp import server as S
    t0 = time.time()
    labels = [t[0] for t in OB._EXTRA_TURNS if t[2] == group]
    tell = [lb for lb in labels if lb.startswith(group + "_t")]
    rest = [lb for lb in labels if lb not in tell]
    resp = {}

    def _turn(label):
        if label in OB._WORLD_STEPS:
            return OB._run_world_step(OB._WORLD_STEPS[label])
        _, msg, session, reset, percept, rich = OB._TURN_BY_LABEL[label]
        r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                        rich=bool(rich), reset=reset))
        return json.loads(r.body)

    for lb in tell:
        resp[lb] = _turn(lb)
    t_tell = time.time()
    chat = [c for k, c in S._BRAIN_CHATS.items() if k[0] == group][0]
    out_d = {"seed": int(seed), "arm": arm, "group": group, "env": env, "labels": labels,
             "tell_seconds": round(t_tell - t0, 1)}
    if getattr(chat, "_da_tag_capture", None) is not None and chat._da_tag_capture.ledger.blocks:
        g = {"full": E_GRID, "scan": SCAN_GRID}[grid] if grid else (SCAN_GRID if scan else E_GRID)
        out_d["curve"] = curve(int(seed), chat, g, record_scores=record_scores)
        if record_scores:
            from webapp.da_tag_capture_chat import store_composer
            sc_comp = store_composer(chat)
            out_d["vocab"] = list(sc_comp.words)
            out_d["inner_seed"] = int(sc_comp.comp.seed)
            out_d["true_items"] = {"agent": "cat", "action": "chase", "patient": "ball"}   # the datr telling
    out_d["curve_seconds"] = round(time.time() - t_tell, 1)
    if scan:                                  # the scan stops at the curve: no rest, no night, no recall
        out_d.update({"scan": True, "elapsed_seconds": round(time.time() - t0, 1),
                      "tell_da": [((resp[lb].get("da_drives") or {}).get("da_level")) for lb in tell]})
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(out_d, open(out, "w"), indent=1, default=str)
        c0 = (out_d.get("curve") or [{}])[0]
        print("[dev scan seed %d] R(e=1)=%s roles=%s  %.0fs" % (seed, c0.get("R"), c0.get("roles"),
                                                                out_d["elapsed_seconds"]), flush=True)
        return out_d
    t_rest = time.time()
    for lb in rest:
        resp[lb] = _turn(lb)
    out_d["rest_night_recall_seconds"] = round(time.time() - t_rest, 1)
    rec = resp[group + "_recall"]
    tc = rec.get("da_tag_capture") or {}
    out_d.update({"outcome": outcome(rec), "recalled_svo": rec.get("recalled_svo"), "abstained": rec.get("abstained"),
                  "awake_replay": tc.get("awake_replay_capture"), "sleep_replay": tc.get("sleep_replay_capture"),
                  "blocks_at_recall": tc.get("blocks"), "world_steps": {k: v for k, v in resp.items()
                                                                       if isinstance(v, dict) and "world_step" in v},
                  "tell_da": [((resp[lb].get("da_drives") or {}).get("da_level")) for lb in tell],
                  "elapsed_seconds": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(out_d, open(out, "w"), indent=1, default=str)
    aw = out_d["awake_replay"] or {}
    b = aw.get("bouts") or []
    print("[dev seed %d arm %s] outcome=%s  R first=%s last=%s  n_bouts=%d  %.0fs" % (
        seed, arm, out_d["outcome"], (b[0]["R"] if b else None), (b[-1]["R"] if b else None), len(b),
        out_d["elapsed_seconds"]), flush=True)
    return out_d


def _z_at_recall(d):
    bl = (d or {}).get("blocks_at_recall") or []
    return float(bl[0]["frac_synapses_z_gt_half"]) if bl else None


def _last_expression(d):
    b = ((d or {}).get("awake_replay") or {}).get("bouts") or []
    return float(b[-1]["early_after"][0]) if b else None


def attribute(out_dir):
    """Whose difference is it? Per dev seed, over the arms present in `out_dir` (s<seed>_<arm>.json): the share of the
    block's expression after the last rest bout that the COMPLETION edge owns (arcc vs arcc_nocomp -- same reads,
    induction with R_c vs R) and that the AWAKE edge owns (arcc vs arcc_lesion). `tools.lab.attributable_to` prints
    UNDEFINED rather than a number when both arms are ~0. Writes attribution.json next to the arms."""
    import glob
    import re
    from tools.lab import attributable_to
    runs = {}
    for p in glob.glob(os.path.join(out_dir, "s*_*.json")):
        m = re.match(r"s(\d+)_([a-z_]+)\.json$", os.path.basename(p))
        if m and not p.endswith(".prov.json"):
            runs.setdefault(int(m.group(1)), {})[m.group(2)] = json.load(open(p))
    out = {}
    for seed, arms in sorted(runs.items()):
        rec = {"outcomes": {a: d.get("outcome") for a, d in arms.items()},
               "e_last": {a: _last_expression(d) for a, d in arms.items()},
               "z_at_recall": {a: _z_at_recall(d) for a, d in arms.items()}}
        t = rec["e_last"].get("arcc")
        for ctrl in ("arcc_nocomp", "arcc_lesion", "arcc_sleep"):
            c = rec["e_last"].get(ctrl)
            if t is not None and c is not None:
                rec["attributable_to_e_last_vs_" + ctrl] = attributable_to(
                    "seed %d e_last arcc vs %s" % (seed, ctrl), t, c)
        if "arc" in arms and "arcc_nocomp" in arms:
            # THE READS ARE INERT UNDER THE LESION: the completion-lesioned arm (every completion read runs, both
            # routes use R) must reproduce the pure margin-route arm (no completion flag) field for field.
            a, b = arms["arc"], arms["arcc_nocomp"]
            kb = ("t_h", "R", "R_eff", "early_before", "early_after", "pre_frac_z_gt_half", "p_at_bout",
                  "n_drive_entries")
            ke = ("t_h", "R", "R_eff", "sum_R_eff", "da_swr", "da_seen_by_d1", "a_eff_mean", "pre_frac_z_gt_half")
            ab, bb = a["awake_replay"]["bouts"], b["awake_replay"]["bouts"]
            ae, be = a["sleep_replay"]["epochs"], b["sleep_replay"]["epochs"]
            rec["lesion_reproduces_margin_route"] = {
                "bouts_identical": len(ab) == len(bb) and all(all(x[k] == y[k] for k in kb) for x, y in zip(ab, bb)),
                "epochs_identical": len(ae) == len(be) and all(all(x[k] == y[k] for k in ke) for x, y in zip(ae, be)),
                "blocks_at_recall_identical": a["blocks_at_recall"] == b["blocks_at_recall"],
                "outcome_identical": (a["outcome"], a["recalled_svo"]) == (b["outcome"], b["recalled_svo"])}
        t = rec["z_at_recall"].get("arcc")
        for ctrl in ("arcc_nocomp", "arcc_awake", "arcc_sleep", "arcc_lesion"):
            c = rec["z_at_recall"].get(ctrl)
            if t is not None and c is not None:
                rec["attributable_to_z_vs_" + ctrl] = attributable_to(
                    "seed %d captured fraction arcc vs %s" % (seed, ctrl), t, c)
        out[seed] = rec
    json.dump(out, open(os.path.join(out_dir, "attribution.json"), "w"), indent=1, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", default=None, help="attribute the per-seed differences over a dev output dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", default="arc", choices=sorted(ARMS))
    ap.add_argument("--out")
    ap.add_argument("--scan", action="store_true", help="tell + the R(e) curve on a short grid only (no rest/night)")
    ap.add_argument("--grid", choices=("scan", "full"), default=None,
                    help="the curve's e grid (default: the short SCAN_GRID with --scan, the 17-point E_GRID otherwise)")
    ap.add_argument("--record-scores", action="store_true",
                    help="record each curve point's per-role matched-filter score vectors + the vocab and composer seed")
    a = ap.parse_args()
    if a.attribute:
        attribute(a.attribute)
        return
    if a.seed is None or a.out is None:
        ap.error("--seed and --out are required (or --attribute <dir>)")
    run(a.seed, a.arm, a.out, scan=a.scan, grid=a.grid, record_scores=a.record_scores)


if __name__ == "__main__":
    main()
