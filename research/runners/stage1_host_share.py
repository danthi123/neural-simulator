"""STAGE-1 HOST SHARE -- who owns the decision at the Stage-1 exit gate's C2/C3 seams, measured by INTERVENTION.

Packet T22 (`docs/drafts/stage1-host-share/HANDOFF.md`; prereg
`research/findings/2026-10-02-stage1-host-share-PREREGISTRATION.md`). It answers the Stage-1 exit HANDOFF's
NEEDS-CLAUDE #10: "run the host-share instrument's intervention arms on the C2/C3 probe turns; report F/S/D beside the
criterion verdicts" (`docs/drafts/stage1-exit/HANDOFF.md` section 10; Stage-1 prereg section 5).

Nothing here is re-derived. The seams, wrappers, per-event measures (live / F / f_outcome / S / D), buckets, classes
and panel are the host-share baseline's (`research/runners/host_share.py` + `host_share_schema.py`, prereg
`2026-09-29-host-share-baselines-PREREGISTRATION.md`), imported. What this file adds:

  - two probe GROUPS that replay the Stage-1 exit gate's frozen `word` (C2) and `lived` (C3) sessions turn for turn
    (`research/runners/stage1_exit_gate.py` GROUPS at `c238ff325`), each label prefixed `s1w_` / `s1l_`;
  - a worker that resolves each turn by its POSITION in the session, not its message (the Stage-1 sessions repeat a
    message: the pre-exposure probe and the later probe are the same text, so the baseline's (session, message)
    lookup would label both turns as the later one);
  - `corpus_present` per arm (V7: the corpus-learned production organs degrade silently without
    `data/corpus/tinystories.txt`; the Stage-1 gate found this, the baseline instrument does not record it);
  - a per-criterion read-out (C2, C3): per seam class + panel, a per-probe-turn table, and the frozen credit line
    "PARTIAL (host decision: <seams>)" (runbook RULING `iw2-stage1-credit` wording) or the alternatives in prereg 6;
  - optionally, a descriptive agreement table with the Stage-1 gate's own intact_a replies (`--stage1-dir`).

Arms per group: intact_a, intact_b (sham), <seam>:intervene and <seam>:substitute for k-select, bg-select, organ-c,
metacog, and ignition:intervene = 11. Two groups x 11 arms x 6 gate seeds = 132 pool jobs (one fresh brain each).

  python -m research.runners.stage1_host_share --selftest
  python -m research.runners.stage1_host_share --emit-jobs <PIN> --seeds 7 --out-dir research/findings/raw/_stage1_host_share/dev
  python -m research.runners.stage1_host_share --devcheck research/findings/raw/_stage1_host_share/dev --seeds 7
  python -m research.runners.stage1_host_share --aggregate research/findings/raw/_stage1_host_share/gate \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_stage1_host_share/gate/aggregate.json
  python -m research.runners.stage1_host_share --report research/findings/raw/_stage1_host_share/gate/aggregate.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import host_share as H  # noqa: E402
from research.runners import host_share_schema as SCHEMA  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

GATE_SEEDS = H.GATE_SEEDS
DEV_SEEDS = H.DEV_SEEDS
PREREG = "research/findings/2026-10-02-stage1-host-share-PREREGISTRATION.md"
STAGE1_PREREG = "research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md"
GATE_OUT_DIR = "research/findings/raw/_stage1_host_share/gate"
DEV_OUT_DIR = "research/findings/raw/_stage1_host_share/dev"
MODULE = "research.runners.stage1_host_share"
SEAMS5 = ("k-select", "bg-select", "organ-c", "metacog", "ignition")

# ── the Stage-1 sessions, turn for turn (stage1_exit_gate.py `word` / `lived` intact turns at c238ff325). Tuple:
#    (label, message, target_cue or None, is_target). Targets = every recall probe and control whose cue is an
#    (agent, action) question; the pre-exposure probes, teach turns, distractors, the anaphora probe (no named agent)
#    and the disclosure probe (a routing seam this instrument has no wrapper for; prereg section 7) are not targets.
S1_GROUPS = {
    "s1_word": {
        "criterion": "C2", "stage1_group": "word", "env": {}, "seams": SEAMS5,
        "sessions": [[
            ("s1w_pre_bear_chase", "what does the bear chase", None, False),
            ("s1w_t_otter_eat", "the otter eats the clam", None, False),
            ("s1w_t_bear_chase", "the bear chases the otter", None, False),
            ("s1w_p_bear_chase", "what does the bear chase", ("bear", "chase"), True),
            ("s1w_p_it_eat", "what does it eat", None, False),
            ("s1w_p_otter_eat", "what does the otter eat", ("otter", "eat"), True),
            ("s1w_c_otter_chase", "what does the otter chase", ("otter", "chase"), True),
            ("s1w_c_clam_eat", "what does the clam eat", ("clam", "eat"), True),
            ("s1w_c_dog_chase", "what does the dog chase", ("dog", "chase"), True),
        ]],
    },
    "s1_lived": {
        "criterion": "C3", "stage1_group": "lived", "env": {"BRAIN_EPISODIC_STORE": "1"}, "seams": SEAMS5,
        "sessions": [[
            ("s1l_pre_wolf_bite", "what does the wolf bite", None, False),
            ("s1l_pre_wolf_discuss", "did we discuss the wolf", None, False),
            ("s1l_t_wolf_bite", "the wolf bites the apple", None, False),
            ("s1l_d_weather", "how is the weather today", None, False),
            ("s1l_d_sky", "what is the weather like tomorrow", None, False),
            ("s1l_d_cat", "what does the cat eat", None, False),
            ("s1l_p_wolf_bite", "what does the wolf bite", ("wolf", "bite"), True),
            ("s1l_p_wolf_discuss", "did we discuss the wolf", None, False),
            ("s1l_c_wolf_chase", "what does the wolf chase", ("wolf", "chase"), True),
            ("s1l_c_dog_chase", "what does the dog chase", ("dog", "chase"), True),
        ]],
    },
}
# the probe each target turn is, for the per-criterion table (counted = in the Stage-1 criterion's pass condition)
PROBE_ROLE = {
    "s1w_p_bear_chase": "counted (new word as patient)", "s1w_p_otter_eat": "counted (new word as agent)",
    "s1w_c_otter_chase": "counted (moat: abstain)", "s1w_c_clam_eat": "counted (moat: abstain)",
    "s1w_c_dog_chase": "counted (built-in fact)",
    "s1l_p_wolf_bite": "counted (lived fact recall)", "s1l_c_wolf_chase": "counted (moat: abstain)",
    "s1l_c_dog_chase": "counted (built-in fact)",
}
HOST_CLASSES = ("HOST-OWNED", "HOST-OWNED (replaceable)")


@contextlib.contextmanager
def stage1_groups():
    """Swap the instrument's GROUPS for the Stage-1 groups (every H function reads H.GROUPS), then restore."""
    saved = dict(H.GROUPS)
    H.GROUPS.clear()
    H.GROUPS.update(S1_GROUPS)
    try:
        yield
    finally:
        H.GROUPS.clear()
        H.GROUPS.update(saved)


def arms_for(group):
    with stage1_groups():
        return H.arms_for(group)


def stage1_label(label):
    """s1w_p_bear_chase -> p_bear_chase (the Stage-1 gate's own label)."""
    return label.split("_", 1)[1]


class LabelCursor:
    """Resolve each brain_chat call to its turn label by POSITION in the session, checking the message matches."""

    def __init__(self, specs, labels):
        self.order = {}
        for lab in labels:
            sp = specs[lab]
            self.order.setdefault(sp["session"], []).append((lab, sp["message"]))
        self.pos = {s: 0 for s in self.order}

    def next(self, session, message):
        seq = self.order.get(session) or []
        i = self.pos.get(session, 0)
        if i >= len(seq) or seq[i][1] != message:
            return None
        self.pos[session] = i + 1
        return seq[i][0]


# ══════════════════════════════════════════════ WORKER (inside the arm subprocess) ═════════════════════════════════
def run_worker(seed, group, arm, out_path, session_index=None, only_labels=None):
    """H.run_worker with the position cursor and the corpus record. Same env, same seed threading, same wrappers."""
    with stage1_groups():
        seam, mode = H._parse_arm(arm)
        env = dict(H.BASE_ENV)
        env.update(S1_GROUPS[group]["env"])
        env["BRAIN_CHAT_SEED"] = str(int(seed))
        for k, v in env.items():                          # BEFORE any webapp import: organs read env at import time
            os.environ[k] = v
        hash_ok = (os.environ.get("PYTHONHASHSEED") == "0")
        corpus_present = os.path.exists(os.path.join(_REPO, "data", "corpus", "tinystories.txt"))

        import research.runners.onebrain_regression_battery as B
        specs = H.turn_specs(group)
        for label, sp in specs.items():                   # rich=True = the omitted-`rich` production default
            B._TURN_BY_LABEL[label] = (label, sp["message"], sp["session"], sp["reset"], None, True)

        import webapp.server as S
        seeds_used = H.install_seed_threading(seed)
        H._ST.update({"seam": seam, "mode": mode})
        if mode != "none":
            H.install_wrappers(seed)
        _orig_bc = S.brain_chat
        snapshots, timings, unresolved = {}, {}, []
        labels = H.group_labels(group, session_index)
        if only_labels:
            labels = [lab for lab in labels if lab in set(only_labels)]
        cursor = LabelCursor(specs, labels)

        def brain_chat(req):
            label = cursor.next(req.session, req.message)
            if label is None:
                unresolved.append([req.session, req.message])
            sp = specs.get(label, {})
            H._ST.update({"label": label, "cue": sp.get("cue"), "target": bool(sp.get("target")), "msg": req.message})
            t0 = time.time()
            try:
                return _orig_bc(req)
            finally:
                timings[label] = round(time.time() - t0, 2)
                snapshots[label] = H._chat_snapshot(req)
                H._ST.update({"label": None, "cue": None, "target": False, "msg": None})

        S.brain_chat = brain_chat
        resp_path = out_path + ".responses.json"
        t0 = time.time()
        B._collect_worker(json.dumps(env), labels, resp_path)
        responses = json.load(open(resp_path))
        os.remove(resp_path)
        result = {"instrument": "stage1_host_share", "prereg": PREREG, "seed": int(seed), "group": group,
                  "criterion": S1_GROUPS[group]["criterion"], "arm": arm, "seam": seam, "mode": mode,
                  "backend": env["SIM_BACKEND"], "session_index": session_index, "env": env,
                  "pythonhashseed_ok": hash_ok, "corpus_present": corpus_present, "seeds_used": seeds_used,
                  "unresolved_turns": unresolved, "turns": specs, "responses": responses,
                  "events": {k: v for k, v in H._ST["events"].items()}, "snapshots": snapshots,
                  "turn_seconds": timings, "wall_seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, default=str)
        print("[stage1_host_share worker] seed=%s group=%s arm=%s -> %s (%.0fs)" % (
            seed, group, arm, out_path, result["wall_seconds"]), flush=True)
        return 0


def run_shard(seed, group, arm, out_path, only_labels=None, allow_gate_seed=False):
    """One (seed, group, arm) shard: each session in its own fresh subprocess with PYTHONHASHSEED=0 (as H.run_shard)."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    parts = []
    for si in range(len(S1_GROUPS[group]["sessions"])):
        part = "%s.part%d" % (out_path, si)
        if not os.path.exists(part):
            cmd = [sys.executable, "-u", "-m", MODULE, "--worker", "--seed", str(seed), "--group", group,
                   "--arm", arm, "--out", part, "--session-index", str(si)]
            if allow_gate_seed:
                cmd += ["--allow-gate-seed", "--prereg", PREREG]
            if only_labels:
                cmd += ["--labels", ",".join(only_labels)]
            p = subprocess.run(cmd, env=env)
            if p.returncode != 0 or not os.path.exists(part):
                return False
        parts.append(json.load(open(part)))
    merged = dict(parts[0])
    merged["session_index"] = None
    for k in ("responses", "events", "snapshots", "turn_seconds"):
        merged[k] = {}
    merged["wall_seconds"], merged["unresolved_turns"] = 0.0, []
    for pt in parts:
        for k in ("responses", "events", "snapshots", "turn_seconds"):
            merged[k].update(pt.get(k) or {})
        merged["wall_seconds"] += float(pt.get("wall_seconds") or 0.0)
        merged["unresolved_turns"] += list(pt.get("unresolved_turns") or [])
        merged["pythonhashseed_ok"] = bool(merged["pythonhashseed_ok"] and pt.get("pythonhashseed_ok"))
        merged["corpus_present"] = bool(merged.get("corpus_present") and pt.get("corpus_present"))
    merged["n_session_processes"] = len(parts)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=1, default=str)
    for si in range(len(parts)):
        for suffix in ("", ".prov.json"):
            fp = "%s.part%d%s" % (out_path, si, suffix)
            if os.path.exists(fp):
                os.remove(fp)
    return True


def job_lines(sha, seeds=GATE_SEEDS, out_dir=GATE_OUT_DIR):
    """One pool line per (seed, group, arm). ~6.3-7.7 GB RSS per build (Stage-1 gate) -> mem_ok 7, memcap 9."""
    if len(sha) != 40:
        raise ValueError("pin must be a full 40-char sha")
    out = []
    for seed in seeds:
        gate = int(seed) in GATE_SEEDS
        for group in S1_GROUPS:
            for arm in arms_for(group):
                p = H.shard_path(out_dir, seed, group, arm)
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 7 1 && mkdir -p %s && "
                       "bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                       "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m %s "
                       "--shard --seed %d --group %s --arm %s%s --out %s"
                       % (sha, os.path.dirname(p), MODULE, seed, group, arm,
                          (" --allow-gate-seed --prereg %s" % PREREG) if gate else "", p))
                why = ("prereg %s (%s); stage-1 host share (T22, stage1-exit NEEDS-CLAUDE #10) seed %d %s %s; "
                       "mem_gb=7" % (PREREG, sha[:9], seed, group, arm))
                out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ AGGREGATION (no brain build) ═══════════════════════════════════════
def credit_line(classes):
    """Frozen (prereg section 6): the Stage-1 credit reading for one criterion from its seam classes."""
    host = [s for s in SEAMS5 if classes.get(s) in HOST_CLASSES]
    mixed = [s for s in SEAMS5 if classes.get(s) == "MIXED"]
    defined = [s for s in SEAMS5 if classes.get(s) not in (None, "UNDEFINED")]
    if host:
        return "PARTIAL (host decision: %s)" % ", ".join(host)
    if mixed:
        return "PARTIAL (host share mixed: %s)" % ", ".join(mixed)
    if defined:
        return "NO HOST-OWNED SEAM among the defined seams (%s)" % ", ".join(defined)
    return "UNDEFINED (no seam defined)"


def _stage1_agreement(stage1_dir, seeds, shard_dir):
    """Descriptive only: does this harness's intact_a give the Stage-1 gate's intact_a reply on each target turn?"""
    rows = []
    for seed in seeds:
        for group, g in S1_GROUPS.items():
            mine = H.shard_path(shard_dir, seed, group, "intact_a")
            theirs = H.shard_path(stage1_dir, seed, g["stage1_group"], "intact_a")
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                rows.append({"seed": seed, "group": group, "agree": None, "why": "shard missing"})
                continue
            a, b = json.load(open(mine)).get("responses") or {}, json.load(open(theirs)).get("responses") or {}
            for lab, sp in H.turn_specs(group).items():
                if not sp["target"]:
                    continue
                ra, rb = a.get(lab) or {}, b.get(stage1_label(lab)) or {}
                ka = (bool(ra.get("abstained")), H.committed_patient(ra) if not ra.get("abstained") else None)
                kb = (bool(rb.get("abstained")), H.committed_patient(rb) if not rb.get("abstained") else None)
                rows.append({"seed": seed, "group": group, "turn": lab, "here": list(ka), "stage1": list(kb),
                             "agree": ka == kb})
    n = [r for r in rows if r.get("agree") is not None]
    return {"rows": rows, "n": len(n), "n_agree": sum(1 for r in n if r["agree"])}


def aggregate(shard_dir, seeds, min_live_seeds=SCHEMA.MIN_LIVE_SEEDS_GATE, stage1_dir=None):
    seeds = [int(s) for s in seeds]
    with stage1_groups():
        rep = H.aggregate(shard_dir, seeds, min_live_seeds=min_live_seeds)
        corpus, unresolved, errored = {}, [], []
        for seed in seeds:
            for group in S1_GROUPS:
                for arm in H.arms_for(group):
                    p = H.shard_path(shard_dir, seed, group, arm)
                    if not os.path.exists(p):
                        continue
                    try:
                        d = json.load(open(p))
                    except Exception:
                        continue
                    key = "s%d/%s/%s" % (seed, group, arm)
                    corpus[key] = bool(d.get("corpus_present"))
                    if d.get("unresolved_turns"):
                        unresolved.append(key)
                    for lab, r in (d.get("responses") or {}).items():
                        if "_error" in (r or {}):
                            errored.append("%s/%s" % (key, lab))
        v = rep["validity"]
        v["V7_corpus_all_present"] = bool(corpus) and all(corpus.values())
        v["V7_corpus_missing"] = sorted(k for k, ok in corpus.items() if not ok)
        v["unresolved_turn_shards"] = unresolved
        v["errored_turns"] = errored
        ok = rep["verdict"] == "BASELINE-VALID" and v["V7_corpus_all_present"] and not unresolved
        rep["verdict"] = "MEASUREMENT-VALID" if ok else "INVALID-OR-INCOMPLETE"
        rep["instrument"] = "stage1_host_share"
        rep["prereg"] = PREREG
        crit = {}
        for group, g in S1_GROUPS.items():
            evs = [e for e in rep["events"] if e.get("group") == group]
            seams = {}
            for seam in g["seams"]:
                sevs = [e for e in evs if e["seam"] == seam]
                st = SCHEMA.seam_stats(sevs)
                seams[seam] = {"class": SCHEMA.classify_seam(st, min_live_seeds=min_live_seeds), "stats": st,
                               "reported_as": rep["seams"][seam]["reported_as"]}
            turns = {}
            for lab, sp in H.turn_specs(group).items():
                if not sp["target"]:
                    continue
                turns[lab] = {"probe": PROBE_ROLE.get(lab), "seams": {}}
                for seam in g["seams"]:
                    te = [e for e in evs if e["seam"] == seam and e["turn"] == lab]
                    turns[lab]["seams"][seam] = {
                        "F": [e["F"] for e in te], "S": [e["S"] for e in te], "D": [e["D"] for e in te],
                        "buckets": [e["bucket"] for e in te], "n_live": sum(1 for e in te if e["live"])}
            classes = {s: d["class"] for s, d in seams.items()}
            pm = SCHEMA.panel(evs, [s for s in H.MEASURED_SEAMS if s in g["seams"]])
            # ATTRIBUTION (gates/attribution_required): the criterion's host share vs the sham (no-intervention)
            # reply-noise floor, as the instrument does for its own panel.
            # per-criterion sham noise: share of this group's seeds whose intact_a != intact_b on any turn
            sham = []
            for seed in seeds:
                pa, pb = (H.shard_path(shard_dir, seed, group, a) for a in ("intact_a", "intact_b"))
                if os.path.exists(pa) and os.path.exists(pb):
                    ra, rb = json.load(open(pa)).get("responses") or {}, json.load(open(pb)).get("responses") or {}
                    sham.append(any(H.sham_diff(ra.get(lab), rb.get(lab)) for lab in H.turn_specs(group)))
            sham_rate = (sum(sham) / len(sham)) if sham else None
            attr = None
            if pm.get("host_decision_share") is not None and sham_rate is not None:
                attr = attributable_to("%s host_decision_share vs sham reply-noise floor" % g["criterion"],
                                       pm["host_decision_share"], sham_rate)
            crit[g["criterion"]] = {
                "group": group, "stage1_group": g["stage1_group"], "seams": seams, "turns": turns,
                "panel_measured": pm, "sham_noise_rate": sham_rate, "sham_noise_attribution": attr,
                "panel_call_graph_checks": SCHEMA.panel(evs, [s for s in g["seams"] if s not in H.MEASURED_SEAMS]),
                "credit": credit_line(classes) if ok else "UNDEFINED (measurement %s)" % rep["verdict"]}
        rep["criteria"] = crit
        if stage1_dir:
            rep["stage1_agreement"] = _stage1_agreement(stage1_dir, seeds, shard_dir)
    return rep


def devcheck(shard_dir, seeds):
    """Dev-seed mechanical check (no verdict): every shard present, no errored or unresolved turn, every reply from
    the expected brain with the corpus, hash + seed threading, sham clean. Liveness per seam is REPORTED only."""
    rep = aggregate(shard_dir, seeds, min_live_seeds=1)
    v = rep["validity"]
    checks = {
        "no shard missing": not v["missing_shards"],
        "no errored turn": not v["errored_turns"],
        "every turn resolved to its label": not v["unresolved_turn_shards"],
        "V6 every reply from tiny-demo +LTM": v["V6_pass"],
        "V7 corpus present in every arm": v["V7_corpus_all_present"],
        "PYTHONHASHSEED=0 in every arm": v["pythonhashseed_all_ok"],
        "seed threading in every arm": v["seed_threading_all_ok"],
        "V1 sham clean": v["V1_pass"],
        "V3 every live intervention applied": v["V3_interventions_applied"],
    }
    live = {c: {s: len(d["stats"]["live_seeds"]) for s, d in cr["seams"].items()} for c, cr in rep["criteria"].items()}
    return all(checks.values()), checks, live, rep


def report_markdown(rep):
    f = lambda x: "UNDEFINED" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))  # noqa: E731
    L = ["**Stage-1 host share (preregistered validity): %s** (seeds %s)" % (rep["verdict"], rep["seeds"]), ""]
    for c, cr in rep.get("criteria", {}).items():
        L += ["### %s (`%s`, Stage-1 group `%s`): %s" % (c, cr["group"], cr["stage1_group"], cr["credit"]), "",
              "| seam | reported as | class | HS | NS | MS | SC | DA | n events with F | live seeds |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s, d in cr["seams"].items():
            st = d["stats"]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                s, d["reported_as"], d["class"], f(st["HS"]), f(st["NS"]), f(st["MS"]), f(st["SC"]), f(st["DA"]),
                st["n_F"], len(st["live_seeds"])))
        pm, pc = cr["panel_measured"], cr["panel_call_graph_checks"]
        L += ["", "Panel, measured seams: host_decision_share %s, neural_decision_fraction %s, masked_fraction %s "
              "(n %s). Call-graph checks: host_decision_share %s (n %s)." % (
                  f(pm["host_decision_share"]), f(pm["neural_decision_fraction"]), f(pm["masked_fraction"]),
                  pm["n_events_F_defined"], f(pc["host_decision_share"]), pc["n_events_F_defined"]), "",
              "| target turn | probe | " + " | ".join("%s F per seed" % s for s in SEAMS5) + " |",
              "|---|---|" + "---|" * len(SEAMS5)]
        for lab, t in cr["turns"].items():
            L.append("| %s | %s | %s |" % (lab, t["probe"], " | ".join(
                ",".join("U" if x is None else str(x) for x in t["seams"][s]["F"]) or "-" for s in SEAMS5)))
        L.append("")
    v = rep["validity"]
    L += ["Validity: V1 sham-clean seeds %s (pass %s); V3 applied %s; V4 %s; V5 drift events %s; V6 pass %s; V7 corpus "
          "%s; PYTHONHASHSEED %s; seed threading %s; unresolved-turn shards %d; missing shards %d." % (
              v["V1_sham_clean_seeds"], v["V1_pass"], v["V3_interventions_applied"],
              v["V4_both_outcomes_registrable"], v["V5_state_drift_events"], v["V6_pass"],
              v["V7_corpus_all_present"], v["pythonhashseed_all_ok"], v["seed_threading_all_ok"],
              len(v["unresolved_turn_shards"]), len(v["missing_shards"]))]
    c = rep["content"]
    L += ["", "Content (%s turns): learned_content_fraction %s, python_selected_fraction %s, unattributed %s." % (
        c["n_turns"], f(c["learned_content_fraction"]), f(c["python_selected_fraction"]),
        f(c["unattributed_fraction"]))]
    if rep.get("stage1_agreement"):
        a = rep["stage1_agreement"]
        L += ["", "Harness agreement with the Stage-1 gate's intact_a (descriptive): %d of %d target turns." % (
            a["n_agree"], a["n"])]
    return "\n".join(L)


# ══════════════════════════════════════════════ SELF-TEST (no brain build) ═════════════════════════════════════════
def selftest(verbose=True):
    ok_h, _ = H.selftest(verbose=False)
    labels = [t[0] for g in S1_GROUPS.values() for s in g["sessions"] for t in s]
    with stage1_groups():
        specs = {lab: sp for g in S1_GROUPS for lab, sp in H.turn_specs(g).items()}
    cur = LabelCursor(specs, [t[0] for t in S1_GROUPS["s1_word"]["sessions"][0]])
    sess = specs["s1w_pre_bear_chase"]["session"]
    seq = [cur.next(sess, t[1]) for t in S1_GROUPS["s1_word"]["sessions"][0]]
    jl = job_lines("0" * 40)
    checks = {
        "host_share selftest": ok_h,
        "labels unique": len(labels) == len(set(labels)),
        "repeated message resolves by position": seq[0] == "s1w_pre_bear_chase" and seq[3] == "s1w_p_bear_chase",
        "a message out of order does not resolve": LabelCursor(specs, ["s1w_t_otter_eat"]).next(sess, "x") is None,
        "22 arms per seed": sum(len(arms_for(g)) for g in S1_GROUPS) == 22,
        "132 gate lines, each with the guard and the prereg": len(jl) == 132
            and all("--allow-gate-seed --prereg %s" % PREREG in c and ("prereg %s" % PREREG) in w for c, w in jl),
        "dev lines carry no gate flag": not any("--allow-gate-seed" in c for c, _ in job_lines("0" * 40, seeds=(7,))),
        "H.GROUPS restored after the swap": "recall" in H.GROUPS and "s1_word" not in H.GROUPS,
        "credit: host-owned seam -> PARTIAL": credit_line({"k-select": "HOST-OWNED"}).startswith("PARTIAL (host "
                                                                                                "decision: k-select"),
        "credit: all undefined -> UNDEFINED": credit_line({s: "UNDEFINED" for s in SEAMS5}).startswith("UNDEFINED"),
        "credit: neural only -> no host-owned": credit_line({"k-select": "NEURAL-OWNED"}).startswith("NO HOST"),
    }
    ok = all(checks.values())
    if verbose:
        for k, val in checks.items():
            print(("PASS " if val else "FAIL ") + k)
        print("SELFTEST", "OK" if ok else "FAILED")
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--devcheck", default=None, metavar="SHARD_DIR")
    ap.add_argument("--report", default=None, metavar="AGGREGATE_JSON")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-jobs", default=None, metavar="SHA", help="print pool job lines (TSV: command, reason)")
    ap.add_argument("--stage1-dir", default=None, help="Stage-1 gate shard dir for the descriptive agreement table")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--group", default=None, choices=sorted(S1_GROUPS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--session-index", type=int, default=None)
    ap.add_argument("--labels", default=None, help="DEV ONLY")
    ap.add_argument("--allow-gate-seed", action="store_true")
    ap.add_argument("--prereg", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in (a.seeds or "").split(",") if s.strip()]

    if a.selftest:
        return 0 if selftest()[0] else 1
    if a.report:
        print(report_markdown(json.load(open(a.report))))
        return 0
    if a.emit_jobs:
        od = a.out_dir or (GATE_OUT_DIR if not seeds or all(s in GATE_SEEDS for s in seeds) else DEV_OUT_DIR)
        for cmd, why in job_lines(a.emit_jobs, seeds=tuple(seeds) or GATE_SEEDS, out_dir=od):
            print("%s\t%s" % (cmd, why))
        return 0
    if a.devcheck:
        ok, checks, live, _ = devcheck(a.devcheck, seeds)
        for k, val in checks.items():
            print(("PASS " if val else "FAIL ") + k)
        print("LIVE SEEDS per criterion x seam (reported, not a pass condition): %s" % json.dumps(live))
        print("DEVCHECK", "OK" if ok else "FAILED")
        return 0 if ok else 1
    if a.aggregate:
        rep = aggregate(a.aggregate, seeds, stage1_dir=a.stage1_dir)
        out = a.out or os.path.join(a.aggregate, "aggregate.json")
        with open(out, "w") as f:
            json.dump(rep, f, indent=1, default=str)
        print(report_markdown(rep))
        return 0 if rep["verdict"] == "MEASUREMENT-VALID" else 3
    if a.seed is None:
        ap.error("--seed is required")
    if a.seed in GATE_SEEDS and not a.allow_gate_seed:
        ap.error("seed %d is a GATE seed: pass --allow-gate-seed (only after the preregistration is committed)" % a.seed)
    if a.seed in GATE_SEEDS and a.prereg != PREREG:
        ap.error("a GATE seed needs --prereg %s" % PREREG)
    only = [x for x in (a.labels or "").split(",") if x.strip()] or None
    if only and a.seed in GATE_SEEDS:
        ap.error("--labels is a dev aid; the gate runs every preregistered turn")
    if a.worker:
        return run_worker(a.seed, a.group, a.arm, a.out, session_index=a.session_index, only_labels=only)
    if a.shard:
        return 0 if run_shard(a.seed, a.group, a.arm, a.out, only_labels=only,
                              allow_gate_seed=a.allow_gate_seed) else 1
    ap.error("choose one of --worker/--shard/--aggregate/--devcheck/--report/--selftest/--emit-jobs")


if __name__ == "__main__":
    sys.exit(main())
