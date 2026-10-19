"""Live-turn host share of the BG selection seam and the ignition-bus read, before and after the neural seam.

Plan step 3 (docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md). Preregistration:
research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md. Handoff: docs/drafts/selection-seam/HANDOFF.md.

Scoring reuses lane C's instrument (`research.runners.host_share`: `reply_diff`, `sham_diff`, `classify_seam`, `_get`);
this module adds only the BG turns, the arms and the in-process wrappers. If lane C's module is not on this branch
yet, set HOST_SHARE_INSTRUMENT_DIR to the directory holding its host_share.py (DEV only).

TURNS (one session, one fresh ChatBrain per arm, the live `webapp.server.brain_chat`; stub renderer, numpy):
  hsb_empty1 '...'   hsb_q 'what does the dog chase'   hsb_empty2 '?'   hsb_hi 'hi'      (all target turns)

ARMS
  prod_a                flag off, no wrapper                    (production)
  prod_b                flag off, observe wrappers              (sham: prod_a == prod_b proves observing is inert)
  prod_intervene        flag off; on target turns the host decision is flipped: a turn the host would not send to
                        the race is sent with (speak 0, silent 1); a turn it would send is sent with (speak 1, silent 0)
  prod_substitute       flag off; on target turns decide_action is replaced by decide_action_neural (the neural seam)
  seam_b                BRAIN_BG_NEURAL_SEAM=1 + BRAIN_GNW_UNIQUE_IGNITION_READ=1, observe wrappers
  seam_readout          as seam_b; on target turns the committed action is re-read with the production first-to-12
                        motor rule on the SAME trial (neural state fixed, only the host read changes)
  seam_lesion_value     as seam_b + BRAIN_BG_NEURAL_SEAM_LESION=value (neural intervention: heard/hold -> D1 closed)

AMENDMENT 2026-10-14 (research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md): `--commit` adds
BRAIN_BG_COMMIT=1 to the three seam arms only (the spiking commit stage decides; the organ seed follows
BRAIN_CHAT_SEED); the prod arms are unchanged. `--score ... --commit` reads the commit record of every neural-seam
event (`commit`: winner, read_invariant, read_variants, co_commit, old_latch_winner) and applies G4 to the commit read.

Run one arm (~6 GB RSS; one ChatBrain build):
  bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 .venv/bin/python -m research.runners.host_share_bg_seam \
      --worker --seed 7 --arm prod_a --out research/findings/raw/_bg_live_seam/dev/s7/prod_a.json
Score:
  .venv/bin/python -m research.runners.host_share_bg_seam --score research/findings/raw/_bg_live_seam/dev --seeds 7 \
      --out research/findings/raw/_bg_live_seam/dev/score.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to  # noqa: E402

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
SESSION = "hs_bg_0"
TURNS = [("hsb_empty1", "...", "empty"), ("hsb_q", "what does the dog chase", "content"),
         ("hsb_empty2", "?", "empty"), ("hsb_hi", "hi", "one_word")]
ARMS = {
    "prod_a": ({}, "none"), "prod_b": ({}, "observe"),
    "prod_intervene": ({}, "intervene"), "prod_substitute": ({}, "substitute"),
    "seam_b": ({"BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_GNW_UNIQUE_IGNITION_READ": "1"}, "observe"),
    "seam_readout": ({"BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_GNW_UNIQUE_IGNITION_READ": "1"}, "readout"),
    "seam_lesion_value": ({"BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_GNW_UNIQUE_IGNITION_READ": "1",
                           "BRAIN_BG_NEURAL_SEAM_LESION": "value"}, "observe"),
}
BG_FIELDS_EXTRA = ("bg_select.action",)
AMENDMENT_ID = "2026-10-14-bg-live-seam-commit"
FROZEN_COMMIT_OP = 4                 # the amendment's frozen BG_COMMIT_GRID row; any other op makes the run UNDEFINED
SEAM_ARMS = ("seam_b", "seam_readout", "seam_lesion_value")
COMMIT_ENV = ("BRAIN_BG_COMMIT", "BRAIN_BG_COMMIT_OP", "BRAIN_BG_COMMIT_LESION")
_EV = {"label": None, "events": {}}


def lane_c():
    """Lane C's host-share instrument module (the shared scorer)."""
    try:
        return importlib.import_module("research.runners.host_share")
    except ImportError:
        d = os.environ.get("HOST_SHARE_INSTRUMENT_DIR")
        if not d:
            raise
        import research.runners as RR
        RR.__path__.append(d)
        return importlib.import_module("research.runners.host_share")


def _rec(**kw):
    _EV["events"].setdefault(_EV["label"], []).append(kw)


def install(mode):
    """Observe / intervene / substitute / readout wrappers on the BG seam, plus an observe-only record of the ignition
    bus read (every norgan_hop call re-read under the alternative host rules; the returned value is untouched)."""
    from research.runners import bg_action_selection_production_organ as BG
    import research.runners._gnw_norgan_bus_derisk as NB
    from research.runners._gnw_coincidence_integrator_derisk import _assign_slots
    orig_decide, orig_neural = BG.decide_action, BG.decide_action_neural

    def decide_action(msg, seed=42, lesion=None):
        sp, si = BG.salience(msg)
        consulted = si > sp
        if mode == "intervene" and _EV["label"]:
            les = BG.bg_select_lesion() if lesion is None else lesion
            flip = (1.0, 0.0) if consulted else (0.0, 1.0)
            r = BG.get_organ(seed=seed, lesion=les).select_once(*flip)
            _rec(kind="bg", mode=mode, applied=True, consulted=consulted, flip=list(flip), winner=r["winner"])
            if not r["committed"] or r["winner"] != BG.STAY_SILENT:
                return None
            return {"action": BG.ACTION_NAME[BG.STAY_SILENT], "winner": int(r["winner"]), "speak_salience": flip[0],
                    "silent_salience": flip[1], "decision_step": r["decision_step"], "loser_ratio": r["loser_ratio"],
                    "lesion": les}
        if mode == "substitute" and _EV["label"]:
            out = orig_neural(msg, seed=seed)
            _rec(kind="bg", mode=mode, applied=True, neural_winner=(BG.decide_action_neural.last or {}).get("winner"))
            return out
        out = orig_decide(msg, seed=seed, lesion=lesion)
        _rec(kind="bg", mode="observe", consulted=consulted, speak_salience=sp, silent_salience=si,
             host_action=(out or {}).get("action"))
        return out

    def decide_action_neural(msg, seed=None, lesion=None):      # seed None -> the organ's own rule (42 flag-off)
        out = orig_neural(msg, seed=seed, lesion=lesion)
        t = BG.decide_action_neural.last or {}
        rd = dict(t.get("readouts") or {})
        extra = {"commit": t["commit"]} if t.get("commit") is not None else {}   # BRAIN_BG_COMMIT (amendment)
        _rec(kind="bg_neural", winner=t.get("winner"), latch_rates=t.get("latch_rates"), readouts=rd,
             v_range=[t.get("v_min"), t.get("v_max")], **extra,
             **({"tonic_rates_hz": t["tonic_rates_hz"]} if t.get("tonic_rates_hz") else {}))   # RULING bg-tonic-persist-1019
        if mode == "readout" and _EV["label"]:
            alt = rd.get("first_to_12_motor (production rule)")
            _rec(kind="bg_readout", applied=True, latch_winner=t.get("winner"), alt_winner=alt)
            if alt != BG.STAY_SILENT:
                return None
            return {"action": BG.ACTION_NAME[BG.STAY_SILENT], "winner": 1, "speak_salience": None,
                    "silent_salience": None, "decision_step": None, "loser_ratio": None, "lesion": None}
        return out

    BG.decide_action, BG.decide_action_neural = decide_action, decide_action_neural
    decide_action_neural.last = None
    BG.decide_action_neural.last = None

    orig_hop = NB.norgan_hop

    def norgan_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub, **kw):
        out = orig_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub, **kw)
        present = [c for c in candidates if c is not None]
        _slot_of, order = _assign_slots(present + ([decoy] if decoy is not None else []))
        _rec(kind="bus", committed=out[0], n_ignited=out[3], rates=[round(float(r), 5) for r in out[1]],
             variants=NB.readout_variants(out[1], order, len(slots_dev)))
        return out

    for modname in ("webapp.gnw_three_organ_bus", "webapp.gnw_bus_shadow"):
        m = importlib.import_module(modname)
        if getattr(m, "norgan_hop", None) is orig_hop:
            m.norgan_hop = norgan_hop


def arm_env(base_env, arm, seed, backend="numpy", commit=False, outer=None):
    """The environment of one arm (pure). `commit` (amendment 2026-10-14): BRAIN_BG_COMMIT=1 on the seam arms only;
    the commit variables are then explicitly empty everywhere else, as they are whenever the outer environment
    (`outer`, default os.environ) carries one. Without either, the 09-29 environment, unchanged."""
    env_extra, _mode = ARMS[arm]
    env = dict(base_env)
    if backend != "numpy":                               # DEV-only replication on the GPU backend
        env["SIM_BACKEND"] = backend
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(env_extra)
    env["BRAIN_CHAT_SEED"] = str(int(seed))
    for k in ("BRAIN_BG_NEURAL_SEAM", "BRAIN_GNW_UNIQUE_IGNITION_READ", "BRAIN_BG_NEURAL_SEAM_LESION"):
        env.setdefault(k, "")                         # explicit empty = OFF in every arm that does not set it
    outer = os.environ if outer is None else outer
    if commit or any(outer.get(k) for k in COMMIT_ENV):
        for k in COMMIT_ENV:
            env[k] = ""
        if commit and env_extra.get("BRAIN_BG_NEURAL_SEAM") == "1":
            env["BRAIN_BG_COMMIT"] = "1"
    return env


def run_worker(seed, arm, out_path, backend="numpy", commit=False):
    _env_extra, mode = ARMS[arm]
    C = lane_c()
    env = arm_env(C.BASE_ENV, arm, seed, backend=backend, commit=commit)
    for k, v in env.items():
        os.environ[k] = v
    import research.runners.onebrain_regression_battery as B
    for i, (label, msg, _cls) in enumerate(TURNS):
        B._TURN_BY_LABEL[label] = (label, msg, SESSION, i == 0, None, True)
    import webapp.server as S
    if mode != "none":
        install(mode)
    orig_bc = S.brain_chat
    by_msg = {msg: label for label, msg, _ in TURNS}

    def brain_chat(req):
        _EV["label"] = by_msg.get(req.message)
        try:
            return orig_bc(req)
        finally:
            _EV["label"] = None

    S.brain_chat = brain_chat
    t0 = time.time()
    resp_path = out_path + ".responses.json"
    B._collect_worker(json.dumps(env), [t[0] for t in TURNS], resp_path)
    responses = json.load(open(resp_path))
    os.remove(resp_path)
    res = {"instrument": "host_share_bg_seam", "seed": int(seed), "arm": arm, "mode": mode, "env": env,
           **({"amendment": AMENDMENT_ID} if commit else {}),
           "pythonhashseed_ok": os.environ.get("PYTHONHASHSEED") == "0", "responses": responses,
           "events": _EV["events"], "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print("[bg seam worker] seed=%s arm=%s -> %s (%.0fs)" % (seed, arm, out_path, res["wall_seconds"]), flush=True)
    return 0


def run_all(seed, out_dir, backend="numpy", arms=None, commit=False):
    """Every arm of one seed, each in its OWN fresh subprocess (PYTHONHASHSEED=0), sequentially; an existing arm file is
    kept (resumable). One brain-loading process at a time, so it is safe as ONE gpu_queue job."""
    import subprocess
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    ok = True
    for arm in (arms or list(ARMS)):
        out = os.path.join(out_dir, "s%d" % int(seed), arm + ".json")
        if os.path.exists(out):
            continue
        cmd = [sys.executable, "-u", "-m", "research.runners.host_share_bg_seam", "--worker", "--seed", str(seed),
               "--arm", arm, "--out", out, "--backend", backend]
        if int(seed) in GATE_SEEDS:
            cmd.append("--allow-gate-seeds")
        if commit:
            cmd.append("--commit")
        p = subprocess.run(cmd, env=env)
        ok = ok and p.returncode == 0 and os.path.exists(out)
    return 0 if ok else 1


def _held(resp):
    return bool(((resp or {}).get("bg_select") or {}).get("action") == "STAY_SILENT")


def commit_event_measures(e, env_on, chat_seed=None):
    """The amended score's per-event measures (pure; 2026-10-14 amendment), from one `bg_neural` event: `engaged`
    (the shard ran with BRAIN_BG_COMMIT=1 and the event carries a commit record; the winner clause is presence-only,
    because the organ copies the commit winner into the event's winner), `op_ok` (the record's op is the frozen
    FROZEN_COMMIT_OP), `seed_ok` (the record's organ_seed is the shard's chat seed), `commit_thr_change` (no commit
    record, or the commit read changes at READ_FRAC x0.5 or x2), `old_read_agree` and `co_commit` (reported)."""
    cm = e.get("commit")
    var = (cm or {}).get("read_variants") or {}
    return {"engaged": int(bool(env_on) and cm is not None and cm.get("winner") == e.get("winner")),
            "op_ok": int(cm is not None and cm.get("op") == FROZEN_COMMIT_OP),
            "seed_ok": int(cm is not None and chat_seed is not None and cm.get("organ_seed") == int(chat_seed)),
            "commit_thr_change": int(cm is None or not cm.get("read_invariant") or not var
                                     or any(v != cm.get("winner") for v in var.values())),
            "old_read_agree": int(cm is not None and cm.get("winner") == cm.get("old_latch_winner")),
            "co_commit": int(bool((cm or {}).get("co_commit")))}


def g4_commit_holds(m, bars=None):
    """The amended G4 on a pooled score: the commit read changed on no seam_b event."""
    bars = bars if bars is not None else BARS_AMENDED
    return m.get("commit_thr_change") is not None and m["commit_thr_change"] <= bars["commit_thr_change_max"]


def score(shard_dir, seeds, commit=False):
    """Per seed and target turn, lane C's F / S / D and outcome labels for the production seam; the capability,
    neural share and readout share for the neural seam; the bus-read variant disagreement. UNDEFINED reasons kept.
    `commit` (amendment 2026-10-14): also read every neural-seam event's commit record: `commit_thr_change` (the
    commit read changes at READ_FRAC x0.5 or x2, or the event carries no commit record), `commit_engaged` (the event
    carries a commit record whose winner is the event's committed action, in a shard run with BRAIN_BG_COMMIT=1),
    `old_read_agree` and `co_commit` (reported); the old latch's `latch_thr_change` is then measured against the old
    read's own choice; `commit_event_count_ok` (exactly one neural-seam event per seam-arm turn: a turn whose stage
    raised leaves none, because the server swallows the error and speaks), `commit_op_ok` and `commit_seed_ok`.
    Without `commit` the score is the 09-29 one, unchanged; shards run with the commit stage are then REFUSED
    (ValueError), never scored as a 09-29 run."""
    C = lane_c()
    fields = tuple(C.DECISION_FIELDS) + BG_FIELDS_EXTRA
    out = {"seeds": [int(s) for s in seeds], "per_seed": {}, "fields": list(fields)}
    pooled = {"F": [], "S": [], "D": [], "cap": [], "ns": [], "readout": [], "bus_disagree": [],
              "latch_thr_change": [], "first12_change": [], "membrane_ok": [], "content_held": [], "empty_held": [],
              "sham_ok": [], "bus_unique_agree": [], "seam_intact_held_for_ns": [], "seam_lesion_held": []}
    if commit:
        pooled.update({"commit_thr_change": [], "commit_engaged": [], "old_read_agree": [], "co_commit": [],
                       "commit_event_count_ok": [], "commit_op_ok": [], "commit_seed_ok": []})
    prod_events = []          # per-turn {F,S,D,f_outcome,seed,live} records, for SCHEMA.seam_stats/classify_seam
    for s in seeds:
        sh = {}
        for arm in ARMS:
            p = os.path.join(shard_dir, "s%d" % int(s), arm + ".json")
            sh[arm] = json.load(open(p)) if os.path.exists(p) else None
        if not commit and any(x is not None and (x.get("amendment") == AMENDMENT_ID
                                                 or (x.get("env") or {}).get("BRAIN_BG_COMMIT") == "1")
                              for x in sh.values()):
            raise ValueError("seed %s: shards ran with the commit stage (amendment %s); score them with --commit "
                             "(commit=True), never as a 09-29 run" % (s, AMENDMENT_ID))
        rows = []
        for label, _msg, cls in TURNS:
            ev = {"turn": label, "class": cls, "undefined": []}
            R = {a: ((sh[a] or {}).get("responses") or {}).get(label) for a in ARMS}
            if any(R[a] is None or "_error" in (R[a] or {}) for a in ("prod_a", "prod_b")):
                ev["undefined"].append("production intact arm missing/errored")
            elif C.sham_diff(R["prod_a"], R["prod_b"]) or _held(R["prod_a"]) != _held(R["prod_b"]):
                ev["undefined"].append("sham null failed (prod_a != prod_b)")
            else:
                obs = [e for e in ((sh["prod_b"] or {}).get("events") or {}).get(label, []) if e.get("kind") == "bg"]
                ev["host_consulted"] = bool(obs and obs[0].get("consulted"))
                ev["prod_held"] = _held(R["prod_a"])
                host_alt_held = not ev["prod_held"]
                if R["prod_intervene"] is not None:
                    ev["F"] = int(_held(R["prod_intervene"]) == host_alt_held)
                    changed = bool(C.reply_diff(R["prod_a"], R["prod_intervene"], fields=fields))
                    ev["intervene_outcome"] = ("follows_host" if ev["F"] else
                                               ("returned_to_intact" if not changed else "other_change"))
                else:
                    ev["undefined"].append("prod_intervene missing")
                if R["prod_substitute"] is not None:
                    ev["S"] = int(bool(C.reply_diff(R["prod_a"], R["prod_substitute"], fields=fields)))
                    ev["D"] = int(_held(R["prod_substitute"]) == ev["prod_held"])
                else:
                    ev["undefined"].append("prod_substitute missing")
            if R["seam_b"] is not None and "_error" not in R["seam_b"]:
                want_hold = cls == "empty"
                ev["seam_held"] = _held(R["seam_b"])
                ev["capability"] = int(ev["seam_held"] == want_hold) if cls != "one_word" else None
                if R["seam_lesion_value"] is not None:
                    ev["neural_share"] = int(_held(R["seam_lesion_value"]) != ev["seam_held"]
                                             or bool(C.reply_diff(R["seam_b"], R["seam_lesion_value"], fields=fields)))
                    ev["seam_lesion_held"] = _held(R["seam_lesion_value"])
                if R["seam_readout"] is not None:
                    rd = [e for e in ((sh["seam_readout"] or {}).get("events") or {}).get(label, [])
                          if e.get("kind") == "bg_readout"]
                    ev["readout_alt_differs"] = bool(rd and rd[0].get("alt_winner") != rd[0].get("latch_winner"))
                    ev["readout_share"] = int(ev["readout_alt_differs"]
                                              and _held(R["seam_readout"]) != ev["seam_held"])
            else:
                ev["undefined"].append("seam_b missing/errored")
            for arm in ("prod_b", "seam_b"):
                for e in ((sh[arm] or {}).get("events") or {}).get(label, []):
                    if e.get("kind") == "bus":
                        v = e["variants"]
                        base = v["argmax_thr (production)"]
                        pooled["bus_disagree"].append(int(any(x != base for x in v.values())))
                        pooled["bus_unique_agree"].append(int(v["unique"] == base))
            for arm in ("seam_b", "seam_readout", "seam_lesion_value"):
                if commit:
                    n_ev = sum(1 for e in ((sh[arm] or {}).get("events") or {}).get(label, [])
                               if e.get("kind") == "bg_neural")
                    pooled["commit_event_count_ok"].append(int(n_ev == 1))
                for e in ((sh[arm] or {}).get("events") or {}).get(label, []):
                    if e.get("kind") == "bg_neural":
                        lo, hi = e.get("v_range") or [None, None]
                        pooled["membrane_ok"].append(int(lo is not None and lo >= -150.0 and hi <= 100.0))
                        cm = e.get("commit") if commit else None
                        if commit:
                            env = (sh[arm] or {}).get("env") or {}
                            cmx = commit_event_measures(e, env.get("BRAIN_BG_COMMIT") == "1",
                                                        chat_seed=env.get("BRAIN_CHAT_SEED"))
                            pooled["commit_engaged"].append(cmx["engaged"])
                            pooled["commit_op_ok"].append(cmx["op_ok"])
                            pooled["commit_seed_ok"].append(cmx["seed_ok"])
                        if arm == "seam_b":
                            rd = e.get("readouts") or {}
                            old = cm.get("old_latch_winner") if cm is not None else e.get("winner")
                            pooled["latch_thr_change"].append(int(rd.get("latch_rate_x0.5") != old
                                                                  or rd.get("latch_rate_x2") != old))
                            if commit:
                                for k in ("commit_thr_change", "old_read_agree", "co_commit"):
                                    pooled[k].append(cmx[k])
                            pooled["first12_change"].append(
                                int(rd.get("first_to_12_motor (production rule)") != e.get("winner")))
            ev["sham_ok"] = int(not any("sham" in u for u in ev["undefined"]))
            pooled["sham_ok"].append(ev["sham_ok"])
            if ev.get("seam_held") is not None and cls in ("empty", "content"):
                pooled["%s_held" % cls].append(int(ev["seam_held"]))
            if ev.get("seam_lesion_held") is not None:                    # paired with seam_held for attribution
                pooled["seam_intact_held_for_ns"].append(int(ev["seam_held"]))
                pooled["seam_lesion_held"].append(int(ev["seam_lesion_held"]))
            rows.append(ev)
            for k, key in (("F", "F"), ("S", "S"), ("D", "D"), ("cap", "capability"), ("ns", "neural_share"),
                           ("readout", "readout_share")):
                if ev.get(key) is not None:
                    pooled[k].append(ev[key])
            # SCHEMA.classify_seam takes a `stats` dict built by SCHEMA.seam_stats over per-event {F,S,D,
            # f_outcome,seed,live} records (research/runners/host_share_schema.py's own docstring: "the step-3
            # selection-seam builder imports it for its before/after instead of keeping its own statistic") --
            # NOT three pooled scalars. f_outcome maps this file's local "returned_to_intact" vocabulary onto
            # SCHEMA's "returned_to_neural" (both mean: unperturbed by the host flip, i.e. the reply took the
            # neural/baseline outcome rather than the host's alternative or some third, unexplained path).
            f_outcome = None
            if ev.get("F") == 0:
                f_outcome = ("returned_to_neural" if ev.get("intervene_outcome") == "returned_to_intact"
                             else "masked")
            prod_events.append({"F": ev.get("F"), "S": ev.get("S"), "D": ev.get("D"), "f_outcome": f_outcome,
                                "seed": int(s), "live": ev.get("F") is not None})
        out["per_seed"][int(s)] = rows
    m = {k: (round(sum(v) / len(v), 4) if v else None) for k, v in pooled.items()}
    out["pooled"] = m
    out["n"] = {k: len(v) for k, v in pooled.items()}
    # `C.classify_seam` does not exist on the host_share module itself (only on host_share_schema, imported
    # there as SCHEMA, and it takes a stats dict from seam_stats(events) -- not three pooled scalars). Bug
    # found 2026-09-29 landing the selection-seam merge (the HANDOFF's own sanity check caught it: printed an
    # AttributeError, not TRANSDUCED). Fixed per the schema module's own docstring ("the step-3 selection-seam
    # builder imports it for its before/after instead of keeping its own statistic"): build per-event records
    # and go through SCHEMA.seam_stats -> SCHEMA.classify_seam, exactly like host_share.py's own aggregate().
    prod_stats = C.SCHEMA.seam_stats(prod_events)
    out["production_seam_class"] = C.SCHEMA.classify_seam(prod_stats)
    out["production_seam_stats"] = prod_stats
    # ATTRIBUTION (gates/attribution_required): G3 (ns) already reports the FRACTION of events whose outcome
    # changes when the value pathway is lesioned -- but a fraction alone still leaves "whose is the seam's held
    # rate" unasked. Ask it directly: treatment = the intact seam's held rate on the same paired events; control
    # = that rate with the value pathway lesioned. Informational only (out["diagnostics"]); does not feed BARS,
    # verdict() or any G1-G5 gate above.
    intact_held_rate, lesion_held_rate = m["seam_intact_held_for_ns"], m["seam_lesion_held"]
    seam_held_attribution = None
    if intact_held_rate is not None and lesion_held_rate is not None:
        seam_held_attribution = attributable_to(
            "seam held-rate (intact vs value-lesioned, paired events)", intact_held_rate, lesion_held_rate)
    out["diagnostics"] = {"seam_held_attribution": seam_held_attribution}
    if commit:
        out["amendment"] = AMENDMENT_ID
    return out


# ═══ the preregistered verdict (bars frozen in research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md)
BARS = {"F_before_min": 0.8, "empty_held_min": 0.75, "content_held_max": 0.0, "ns_after_min": 0.3,
        "latch_thr_change_max": 0.10, "bus_unique_agree_min": 0.95, "membrane_ok_min": 1.0, "sham_ok_min": 1.0,
        "min_events_per_measure": 6}
# AMENDMENT 2026-10-14 (research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md): G4 on the commit read, no
# change allowed; the commit stage engaged on every neural-seam event (a precondition: UNDEFINED, never NO-GO).
BARS_AMENDED = dict(BARS, commit_thr_change_max=0.0, commit_engaged_min=1.0, frozen_commit_op=FROZEN_COMMIT_OP)


def verdict(sc, bars=None):
    """G1-G5 over the pooled six-seed score. Preconditions (UNDEFINED, never NO-GO, when they fail): the sham null
    holds on every target event, no membrane-bound violation in any neural-seam trial, and every gated measure has
    at least `min_events_per_measure` events. A score made with `commit` (the 2026-10-14 amendment) is judged with
    BARS_AMENDED: G4 reads the commit read (`commit_thr_change` must be 0) and the commit stage must be engaged on
    every neural-seam event (a precondition). G1, G2a, G2b, G3, G5 and the other preconditions are unchanged."""
    from tools.verdict import Verdict
    amended = sc.get("amendment") == AMENDMENT_ID
    bars = bars if bars is not None else (BARS_AMENDED if amended else BARS)
    m, n = sc["pooled"], sc["n"]
    v = Verdict("BG live seam retirement (plan step 3)")
    v.require("sham null holds on every target event", m["sham_ok"], expect=lambda x: x is not None
              and x >= bars["sham_ok_min"])
    v.require("membrane in [-150, +100] mV in every neural-seam trial", m["membrane_ok"],
              expect=lambda x: x is not None and x >= bars["membrane_ok_min"])
    for k in ("F", "empty_held", "content_held", "ns", "latch_thr_change", "bus_unique_agree"):
        v.require("enough events for %s" % k, n.get(k, 0), expect=lambda x: x >= bars["min_events_per_measure"])
    if amended:
        v.require("commit stage engaged on every neural-seam event", m.get("commit_engaged"),
                  expect=lambda x: x is not None and x >= bars["commit_engaged_min"])
        n_turns = len(SEAM_ARMS) * len(TURNS) * len(sc.get("seeds") or [])
        v.require("exactly one neural-seam event per seam-arm turn (%d)" % n_turns,
                  (m.get("commit_event_count_ok"), n.get("commit_event_count_ok", 0)),
                  expect=lambda x: x[0] == 1.0 and x[1] == n_turns)
        v.require("commit.op == %d on every neural-seam event" % FROZEN_COMMIT_OP, m.get("commit_op_ok"),
                  expect=lambda x: x == 1.0)
        v.require("commit.organ_seed == the shard's chat seed on every neural-seam event", m.get("commit_seed_ok"),
                  expect=lambda x: x == 1.0)
        v.require("enough events for commit_thr_change", n.get("commit_thr_change", 0),
                  expect=lambda x: x >= bars["min_events_per_measure"])
    g = {
        "G1_production_host_owns_the_reply": m["F"] is not None and m["F"] >= bars["F_before_min"],
        "G2a_neural_seam_holds_on_empty_turns": m["empty_held"] is not None
        and m["empty_held"] >= bars["empty_held_min"],
        "G2b_neural_seam_never_holds_a_content_turn": m["content_held"] is not None
        and m["content_held"] <= bars["content_held_max"],
        "G3_value_lesion_moves_the_reply": m["ns"] is not None and m["ns"] >= bars["ns_after_min"],
        "G4_latch_read_threshold_invariant": m["latch_thr_change"] is not None
        and m["latch_thr_change"] <= bars["latch_thr_change_max"],
        **({"G4_commit_read_threshold_invariant": g4_commit_holds(m, bars)} if amended else {}),
        "G5_bus_argmax_does_no_work": m["bus_unique_agree"] is not None
        and m["bus_unique_agree"] >= bars["bus_unique_agree_min"],
    }
    if amended:                                     # the amendment's G4 replaces the 09-29 G4 (reported, not gated)
        g["G4_old_latch_read_threshold_invariant (reported, not gated)"] = g.pop("G4_latch_read_threshold_invariant")
    gated = {k: x for k, x in g.items() if "not gated" not in k}
    d = v.decide(go=all(gated.values()), verbose=False)
    return {"verdict": d["status"], "gates": g, "status": d["status"], "preconditions": d["preconditions"],
            "production_seam_class_lane_c": sc.get("production_seam_class")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arm", choices=sorted(ARMS), default="prod_a")
    ap.add_argument("--score", type=str, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--allow-gate-seeds", action="store_true")
    ap.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    ap.add_argument("--run-all", type=str, default=None, help="out dir: run every arm of --seed sequentially")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", action="store_true", help="amendment 2026-10-14: the commit stage on the seam arms")
    a = ap.parse_args(argv)
    if a.worker:
        if a.seed in GATE_SEEDS and not a.allow_gate_seeds:
            raise SystemExit("gate seed without --allow-gate-seeds (preregistration first)")
        return run_worker(a.seed, a.arm, a.out, backend=a.backend, commit=a.commit)
    if a.run_all:
        if a.seed in GATE_SEEDS and not a.allow_gate_seeds:
            raise SystemExit("gate seed without --allow-gate-seeds (preregistration first)")
        return run_all(a.seed, a.run_all, backend=a.backend, commit=a.commit)
    if a.score:
        res = score(a.score, a.seeds, commit=a.commit)
        res["label"] = ("GATE" if sorted(a.seeds) == sorted(GATE_SEEDS) else "DEV, not a verdict")
        vd = verdict(res)
        if res["label"] == "GATE":
            res.update(vd)                         # top-level verdict + preconditions (gates/verdict_preconditions)
        else:
            res["dev_readout"] = {k: vd[k] for k in ("gates", "status")}
        print(json.dumps({"pooled": res["pooled"], "n": res["n"], "production_seam_class": res["production_seam_class"],
                          "label": res["label"], "verdict": res.get("verdict", res.get("dev_readout"))}))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
