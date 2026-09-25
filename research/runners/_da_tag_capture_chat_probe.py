"""DA TAG-AND-CAPTURE IN CHAT: does lesioning the DA gate change the brain's NEXT-DAY answer? (default-off wiring)

QUESTION. The v3 synaptic tag-and-capture ledger (webapp/da_tag_capture.py, GO at runner level, finding
2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-6seed-GO-runner-level.md) is now wired into the live
`/api/brain-chat` store path and the continuous engine's idle/sleep tick behind BRAIN_DA_TAG_CAPTURE
(webapp/da_tag_capture_chat.py). Through the REAL handler (webapp.server.brain_chat, tiny-demo brain, numpy, stub
renderer, no LLM): a fact told inside surprising news -> a night passes through the brain's own idle/sleep tick -> the
fact is asked for. Does the reply depend on the DA->encoding edge (BRAIN_DA_ENCODING_LESION), selectively for the
salient telling, only when the companion is armed, and without touching immediate recall?

ARMS (each a fresh brain build in its own subprocess, `load_bearing_fraction._spawn_arm`, seed via BRAIN_CHAT_SEED;
turn groups in onebrain_regression_battery: datc / datn / datci / datni):
  sal_night_intact_a / _b  salient telling -> night -> recall, companion ON (b = the null-control rebuild)
  sal_night_lesion         the same + BRAIN_DA_ENCODING_LESION=1 (the lesion the battery already uses for this row)
  neu_night_intact / _lesion  plain telling (words introduced first) -> night -> recall, companion ON
  sal_imm_intact / _lesion    salient telling -> recall at once (no night), companion ON
  neu_imm_intact              plain telling -> recall at once, companion ON
  sal_night_off_intact / _lesion  salient -> night -> recall, companion OFF (today's production default)
ON = {BRAIN_DA_TAG_CAPTURE=1, BRAIN_DA_TAG_CAPTURE_CLOCK=turn}; OFF = {BRAIN_DA_TAG_CAPTURE=0, ...CLOCK=turn}.

The gates are pre-registered in research/findings/2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md (its own
commit, before any gate-seed run). `grade_seed` below implements them verbatim.

Run one seed (numpy CPU, 10 brain builds, `--workers N` concurrent subprocesses; ALWAYS memcap, gate with mem_ok):
  bash tools/mem_ok.sh 12 && BRAIN_CHAT_SEED unset is fine (the runner sets it per seed):
  tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --seed 42 \
      --out research/findings/raw/_da_tag_capture_chat
SLEEP-REPLAY CAPTURE FAMILY (`--family rc`, branch research/sleep-replay-capture): RC_ARMS / grade_seed_rc /
aggregate_rc, gates pre-registered in research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md. Does an
ordinary fact survive the night once BRAIN_SLEEP_REPLAY_CAPTURE (webapp/sleep_replay_capture.py) is armed, does cutting
the replay edge remove that, and do the DA lesions still gate capture? Output defaults to RC_OUT.
  tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --family rc --seed 42
  ... --family rc --aggregate research/findings/raw/_sleep_replay_capture
Selftest (no brain):   ... --selftest
Aggregate:             ... --aggregate research/findings/raw/_da_tag_capture_chat
R2 FAMILY (`--family r2`, branch research/sleep-replay-capture-r2, Amendment 1 of the sleep-replay-capture prereg):
a fact told 4 h before sleep onset (awake world step), and three nights with sleep downscaling (BRAIN_SLEEP_DOWNSCALING).
Byte-identical OFF (r2: a COUNTERFACTUAL built from the current tree -- HEAD vs HEAD minus the feature's own commits,
plus a HEAD-vs-HEAD null control; three tiny-demo builds, exact sha256; replaces the stale fixed-pin check):
  tools/memcap.sh 8 -- .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --offcheck \
      --out research/findings/raw/_sleep_replay_capture_r2/offcheck_counterfactual.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))

SEEDS = [42, 43, 44, 100, 101, 102]
FACT = ["cat", "chase", "ball"]
PINNED_SHA = "36a175534"            # HISTORICAL ONLY (recorded in base-family seed files; r2 retired the pinned offcheck
                                    #  for the counterfactual one, see `offcheck`). Was: merge-base of this branch and origin/main (2026-09-24 Amendment 2, review
                                    #  v2:2a37f2493): was f35196e66, a SHA that PREDATES this branch's own two
                                    #  origin/main merges, so webapp/server.py (+55/-1 between the two SHAs) could
                                    #  make branch-OFF differ from the pinned tree for reasons that are not this
                                    #  branch's change; the merge-base is the correct byte-identical-off reference
ON = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
OFF = {"BRAIN_DA_TAG_CAPTURE": "0", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
LES = {"BRAIN_DA_ENCODING_LESION": "1"}
# (arm name, recall label, env)
ARMS = [
    ("sal_night_intact_a", "datc_recall", dict(ON)),
    ("sal_night_intact_b", "datc_recall", dict(ON)),
    ("sal_night_lesion", "datc_recall", {**ON, **LES}),
    ("neu_night_intact", "datn_recall", dict(ON)),
    ("neu_night_lesion", "datn_recall", {**ON, **LES}),
    ("sal_imm_intact", "datci_recall", dict(ON)),
    ("sal_imm_lesion", "datci_recall", {**ON, **LES}),
    ("neu_imm_intact", "datni_recall", dict(ON)),
    ("sal_night_off_intact", "datc_recall", dict(OFF)),
    ("sal_night_off_lesion", "datc_recall", {**OFF, **LES}),
    # neu_night_off_intact (branch research/da-tag-capture-ltm-on, 2026-09-24, Amendment 3): the plain telling's
    # OWN companion-OFF control -- today's production default (no DA-tag-capture at all) tells the SAME neutral
    # fact, sleeps, and is asked. Existed for the salient group (sal_night_off_*) since the original wiring but
    # never for the neutral one, so nothing in this runner could show whether flipping BRAIN_DA_TAG_CAPTURE ON
    # makes an ORDINARY (non-salient) fact WORSE off than it is today -- see g["ordinary_fact_flip_forgetting"].
    ("neu_night_off_intact", "datn_recall", dict(OFF)),
]
# ── SLEEP-REPLAY CAPTURE FAMILY (`--family rc`; branch research/sleep-replay-capture, webapp/sleep_replay_capture.py;
# gates pre-registered in research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md). A SEPARATE arm list
# and grader: the base family's ARMS / grade_seed / aggregate are untouched, so no committed seed*.json is re-graded.
RC = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1"}
RC_LES = {"BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"}
CAP_LES = {"BRAIN_DA_CAPTURE_LESION": "1"}
RC_ARMS = [
    ("neu_night_norc", "datn_recall", dict(ON)),                          # the flag-off baseline (ledger on, no route)
    ("neu_night_rc_a", "datn_recall", {**ON, **RC}),
    ("neu_night_rc_b", "datn_recall", {**ON, **RC}),                      # null-control rebuild of the new path
    ("neu_night_rc_replaylesion", "datn_recall", {**ON, **RC, **RC_LES}),
    ("neu_night_rc_dalesion", "datn_recall", {**ON, **RC, **LES}),        # REPORTED: is the rescue DA-gated too?
    ("neu_imm_rc", "datni_recall", {**ON, **RC}),                         # precondition: stored + immediately recalled
    ("sal_night_rc", "datc_recall", {**ON, **RC}),
    ("sal_night_rc_dalesion", "datc_recall", {**ON, **RC, **LES}),
    ("sal_night_rc_caplesion", "datc_recall", {**ON, **RC, **CAP_LES}),
    ("sal_night_rc_replaylesion", "datc_recall", {**ON, **RC, **RC_LES}),
]
RC_OUT = "research/findings/raw/_sleep_replay_capture"
# ── r2 FAMILY (`--family r2`; branch research/sleep-replay-capture-r2; Amendment 1 of the sleep-replay-capture prereg).
# Item 1: the neutral telling, then 4 h AWAKE without conversation, then the night (group 'datl'). Item 2: three nights
# with sleep downscaling (BRAIN_SLEEP_DOWNSCALING) for a weak telling (d3w), a salient one (d3c) and the weak one
# re-mentioned on the next two days (d3r).
SHY = {"BRAIN_SLEEP_DOWNSCALING": "1"}
R2_ARMS = [
    ("ld_ledger_off", "datl_recall", dict(OFF)),                          # today's production default (REPORTED)
    ("ld_norc", "datl_recall", dict(ON)),
    ("ld_rc", "datl_recall", {**ON, **RC}),
    ("ld_rc_replaylesion", "datl_recall", {**ON, **RC, **RC_LES}),
    ("neu_imm_rc", "datni_recall", {**ON, **RC}),                         # precondition (item 1)
    ("d3w_rc", "d3w_recall", {**ON, **RC}),                               # precondition (item 2): kept w/o downscaling
    ("d3w_shy_a", "d3w_recall", {**ON, **RC, **SHY}),
    ("d3w_shy_b", "d3w_recall", {**ON, **RC, **SHY}),                     # null-control rebuild
    ("d3c_shy", "d3c_recall", {**ON, **RC, **SHY}),
    ("d3r_shy", "d3r_recall", {**ON, **RC, **SHY}),
    # Amendment 3 (REPORTED only, never in a gate or an UNDEFINED rule): the ten-night horizon, a read-only daily probe
    ("d10w_rc", "d10w_recall10", {**ON, **RC}),
    ("d10w_shy", "d10w_recall10", {**ON, **RC, **SHY}),
]
HORIZON_ARMS = ("d10w_rc", "d10w_shy")
HORIZON_NIGHTS = 10
R2_OUT = "research/findings/raw/_sleep_replay_capture_r2"
# ── AWAKE-REST FAMILY (`--family arc`; branch research/awake-replay-capture, webapp/awake_replay_capture.py; gates
# pre-registered as Amendment 4 of research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md). The
# Amendment-1 long-delay telling, now with quiet rest in the 4 waking hours (battery groups datr / datcr / datq / datz;
# datl = the same 4 h with NO idle tick). Every arm of one group gets the same idle ticks; only the flags differ.
ARC = {"BRAIN_AWAKE_REPLAY_CAPTURE": "1"}
ARC_LES = {"BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1"}
ARC_ARMS = [
    ("lr_arc_a", "datr_recall", {**ON, **RC, **ARC}),                        # ARC1: the rescue
    ("lr_arc_b", "datr_recall", {**ON, **RC, **ARC}),                        # G0: null-control rebuild
    ("lr_noarc", "datr_recall", {**ON, **RC}),                               # ARC1: the SAME rest, awake route off
    ("lr_arc_lesion", "datr_recall", {**ON, **RC, **ARC, **ARC_LES}),        # ARC2: the awake edge cut
    ("ln_arc", "datl_recall", {**ON, **RC, **ARC}),                          # ARC3: no idle period at all (datl)
    ("lr_arc_sleeplesion", "datr_recall", {**ON, **RC, **ARC, **RC_LES}),    # ARC4: rest, but no night replay/PRP
    ("lr_arc_dalesion", "datr_recall", {**ON, **RC, **ARC, **LES}),          # ARC5: DA stays the gate
    ("lsr_arc_sleeplesion", "datcr_recall", {**ON, **RC, **ARC, **RC_LES}),  # ARC6: salient kept on waking capture
    ("neu_imm_arc", "datni_recall", {**ON, **RC, **ARC}),                    # P1: stored + recalled at once
    ("lq_arc", "datq_recall", {**ON, **RC, **ARC}),                          # REPORTED: one rest tick per hour
    ("lz_arc", "datz_recall", {**ON, **RC, **ARC}),                          # REPORTED: rest only in the last hour
    ("lr_ledger_off", "datr_recall", dict(OFF)),                             # REPORTED: today's default, same rest
]
ARC_REPORTED_ARMS = ("lq_arc", "lz_arc", "lr_ledger_off")
ARC_BOUTS = {"datr_recall": 48, "datcr_recall": 48, "datq_recall": 4, "datz_recall": 12, "datl_recall": 0,
             "datni_recall": 0}                        # == the battery's rest ticks (one bout per tick, 5-min limit)
ARC_OUT = "research/findings/raw/_awake_replay_capture"
AWAKE_H = 4.0                        # == onebrain_regression_battery._run_world_step("awake_4h")
LESION_HELD_MAX_RATIO = 0.25        # G6: lesion PRP p_max must stay below 25 % of the intact arm's (the D1 pool's
                                    #  tonic-rate noise floor gives a ~0.1 per turn at DA=0.5; see the prereg)


def outcome(resp):
    """correct / abstain / confab / undefined for one recall-turn response."""
    if not isinstance(resp, dict) or resp.get("_error"):
        return "undefined"
    svo = resp.get("recalled_svo")
    if svo is not None and list(svo) == FACT:
        return "correct"
    if svo is None and resp.get("abstained"):
        return "abstain"
    if svo is not None:
        return "confab"
    return "undefined"


def _tc(resp):
    return (resp or {}).get("da_tag_capture") or {}


def _fact_block(resp):
    """The managed store block holding the fact (the ledger's first managed block on these groups), or None."""
    blocks = _tc(resp).get("blocks") or []
    return blocks[0] if blocks else None


# ── arms ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, out_dir, ltm="off", workers=1, family="base", only=None):
    sys.path.insert(0, _REPO)
    arm_list = {"rc": RC_ARMS, "r2": R2_ARMS, "arc": ARC_ARMS}.get(family, ARMS)
    grader = {"rc": grade_seed_rc, "r2": grade_seed_r2, "arc": grade_seed_arc}.get(family, grade_seed)
    if only:                                   # a de-risk subset (never a gate row): run only these arms, do not grade
        arm_list = [a for a in arm_list if a[0] in set(only)]
        grader = lambda _res: {"partial": True, "arms_run": [a[0] for a in arm_list],   # noqa: E731
                               "outcomes": {k: v["recall_outcome"] for k, v in _res["arms"].items()}}
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))       # every arm's worker inherits it (the battery's seed thread)
    if ltm == "off":                                     # the LTM tier is a separate routed store the ledger never
        os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"       #  touches; its load does not fit the 15 GB pool nodes (declared)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.load_bearing_fraction import _spawn_arm, turn_group
    sdir = os.path.join(out_dir, "seed%d" % int(seed))
    os.makedirs(sdir, exist_ok=True)
    arms = {}

    def _one(spec):
        name, label, env = spec
        grp = turn_group(label)
        r = _spawn_arm(dict(env), grp, os.path.join(sdir, "%s.json" % name))   # a fresh subprocess per arm
        print("[seed %d] arm %s -> %s" % (seed, name, outcome((r or {}).get(label))), flush=True)
        return name, {"label": label, "env": env, "turns": grp, "responses": r}

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for name, rec in ex.map(_one, arm_list):
            arms[name] = rec
    res = {"seed": int(seed), "fact": FACT, "arms": {}, "pinned_sha": PINNED_SHA, "ltm": ltm,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "argv": list(sys.argv), "workers": int(workers)}
    if family in ("rc", "r2", "arc"):
        res["family"] = family
    for name, a in arms.items():
        r = a["responses"] or {}
        rec = r.get(a["label"])
        res["arms"][name] = {
            "label": a["label"], "env": a["env"], "turns": a["turns"], "artifact": os.path.join(sdir, name + ".json"),
            "recall_outcome": outcome(rec),
            "recalled_svo": (rec or {}).get("recalled_svo"), "abstained": (rec or {}).get("abstained"),
            "tag_capture_at_recall": {k: _tc(rec).get(k) for k in ("world_t_h", "n_turns", "p", "p_max",
                                                                    "n_managed_blocks", "external_rescales",
                                                                    "external_rewrites", "gamma", "d1_a_go")},
            "fact_block_at_recall": _fact_block(rec),
            "turn_da": [((r.get(t) or {}).get("da_drives") or {}).get("da_level") for t in a["turns"]],
            "turn_a_eff": [(((r.get(t) or {}).get("da_tag_capture") or {}).get("observe") or {}).get("observed", {})
                           .get("a_eff") for t in a["turns"]],
            "errors": [str(v.get("_error"))[:300] for v in r.values() if isinstance(v, dict) and v.get("_error")],
        }
        if family in ("rc", "r2", "arc"):
            res["arms"][name]["sleep_replay_at_recall"] = _tc(rec).get("sleep_replay_capture")
            res["arms"][name]["blocks_at_recall"] = _tc(rec).get("blocks")
        if family == "arc":
            res["arms"][name]["awake_replay_at_recall"] = _tc(rec).get("awake_replay_capture")
            res["arms"][name]["awake_until_h"] = _tc(rec).get("awake_until_h")
            res["arms"][name]["world_steps"] = {t: r.get(t) for t in a["turns"]
                                                if isinstance(r.get(t), dict) and "world_step" in r.get(t)}
        if family == "r2":
            res["arms"][name]["awake_until_h"] = _tc(rec).get("awake_until_h")
            if name in HORIZON_ARMS:
                daily = [r.get("d10w_recall%d" % n) for n in range(1, HORIZON_NIGHTS + 1)]
                res["arms"][name]["daily_outcomes"] = [outcome(d) for d in daily]
                res["arms"][name]["daily_inc_mag"] = [((_tc(d).get("blocks") or [{}])[0].get("inc_mag")) for d in daily]
                res["arms"][name]["daily_base_mag"] = [((_tc(d).get("blocks") or [{}])[0].get("base_mag")) for d in daily]
    res["gates"] = grader(res)
    json.dump(res, open(os.path.join(out_dir, "seed%d.json" % int(seed)), "w"), indent=2, default=str)
    print(json.dumps(res["gates"], indent=2, default=str), flush=True)
    return res


# ── gates (pre-registered; see the PREREGISTRATION finding) ─────────────────────────────────────────────────────
def grade_seed(res):
    A = res["arms"]
    o = {k: v["recall_outcome"] for k, v in A.items()}
    g = {}
    # G0 null control: the intact rebuild reproduces the recall decision AND the ledger's own state exactly
    a, b = A["sal_night_intact_a"], A["sal_night_intact_b"]
    g["G0_null_clean"] = bool(o["sal_night_intact_a"] == o["sal_night_intact_b"]
                              and a["recalled_svo"] == b["recalled_svo"] and a["abstained"] == b["abstained"]
                              and a["tag_capture_at_recall"] == b["tag_capture_at_recall"]
                              and a["fact_block_at_recall"] == b["fact_block_at_recall"])
    # P1 precondition: the fact is stored and immediately recallable in BOTH tellings (else the next-day contrast is
    # about encoding, not persistence) -> UNDEFINED, never a pass or a fail
    g["P1_immediate_precondition"] = bool(o["sal_imm_intact"] == "correct" and o["neu_imm_intact"] == "correct")
    # G1 the load-bearing claim: salient told, next day -> intact answers, DA-gate lesion does not
    g["G1_lesion_changes_next_day_reply"] = bool(o["sal_night_intact_a"] == "correct"
                                                 and o["sal_night_lesion"] == "abstain")
    # G2 the lesion spares immediate recall (it acts on persistence, Bethus 2010)
    g["G2_lesion_spares_immediate"] = bool(o["sal_imm_lesion"] == "correct")
    # G3 selectivity: the plainly-told fact is NOT kept overnight (capture follows the salient DA drive)
    g["G3_neutral_not_kept"] = bool(o["neu_night_intact"] == "abstain")
    # G4 companion-off contrast: without the companion the lesion does NOT change the next-day reply
    g["G4_off_lesion_no_change"] = bool(o["sal_night_off_intact"] == o["sal_night_off_lesion"]
                                        and A["sal_night_off_intact"]["recalled_svo"]
                                        == A["sal_night_off_lesion"]["recalled_svo"])
    # G5 no confabulation anywhere
    g["G5_no_confab"] = bool(all(v != "confab" for v in o.values()))
    # G6 the lesion held at measurement: lesion-arm PRP p_max < 25 % of intact (UNDEFINED if intact p_max is 0/None)
    pi = A["sal_night_intact_a"]["tag_capture_at_recall"].get("p_max")
    pl = A["sal_night_lesion"]["tag_capture_at_recall"].get("p_max")
    g["G6_lesion_held"] = (None if not pi else bool(pl is not None and pl < LESION_HELD_MAX_RATIO * pi))
    # ATTRIBUTION (reported; tools.lab): the lesion's next-day reply change vs (i) the null rebuild and (ii) the same
    # lesion with the companion OFF -- what fraction of the change is the lesion, and what fraction needs the companion.
    from tools.lab import attributable_to

    def _ndiff(x, y):
        return int(x["recalled_svo"] != y["recalled_svo"]) + int(x["abstained"] != y["abstained"])
    treat = _ndiff(A["sal_night_intact_a"], A["sal_night_lesion"])
    g["attribution"] = {
        "lesion_vs_null": attributable_to("da-tag-capture lesion vs null rebuild", treat,
                                          _ndiff(A["sal_night_intact_a"], A["sal_night_intact_b"])),
        "needs_companion": attributable_to("da-tag-capture lesion effect ON vs companion OFF", treat,
                                           _ndiff(A["sal_night_off_intact"], A["sal_night_off_lesion"])),
        "treatment_diffs": treat}
    # ISOLATION: gamma / d1_a_go must be identical across every companion-ON arm at this seed, independent of
    # BRAIN_DA_ENCODING_LESION (2026-09-23 fix, review v2:dd14adaf7 -- `ChatTagCapture`'s D1 reader used to share
    # the process-level cache with the production spiking write gain, so a lesion arm's ledger got a DIFFERENT
    # gamma/d1_a_go than an intact arm's purely from build order, not from the lesioned edge). Only checked when
    # gamma data is actually present: a record with no "env" / no "gamma" (a hand-built selftest arm, or an older
    # artifact from before this field existed) is exempt, never penalized for a field it never had.
    on_gammas, on_d1s = [], []
    for name, rec in A.items():
        if (rec.get("env") or {}).get("BRAIN_DA_TAG_CAPTURE") != "1":
            continue
        tc = rec.get("tag_capture_at_recall") or {}
        if tc.get("gamma") is not None:
            on_gammas.append(tc["gamma"])
        if tc.get("d1_a_go") is not None:
            on_d1s.append(tc["d1_a_go"])
    g["G_isolation_gamma_consistent"] = bool((not on_gammas or all(abs(v - on_gammas[0]) < 1e-6 for v in on_gammas))
                                             and (not on_d1s or all(abs(v - on_d1s[0]) < 1e-6 for v in on_d1s)))
    # ORDINARY_FACT_FLIP_FORGETTING (REPORTED, NOT gating -- branch research/da-tag-capture-ltm-on, 2026-09-24
    # Amendment 3, board #227 item (c)): does flipping BRAIN_DA_TAG_CAPTURE ON cost a plainly-told fact its
    # overnight survival, relative to TODAY'S production default (the flag off)? True iff the companion-OFF
    # control recalls the neutral fact correctly (today's baseline: nothing decays it) AND the companion-ON
    # arm does not (G3 already requires the ON arm to abstain here BY DESIGN -- that is the mechanism's own
    # selectivity, not a bug -- so a True reading here is EXPECTED under a working mechanism, not a failure of
    # it; it exists to make the flip's true cost to ordinary conversational recall visible in the record rather
    # than assumed). Deliberately excluded from `core`/`seed_verdict`: it must never retroactively change the
    # already-scored G0-G6 verdict of a seed*.json committed before this arm existed. `neu_night_off_intact` is
    # ABSENT on every seed*.json committed under the prior finding (2026-09-24-da-tag-capture-chat-wire-6seed-
    # GO-runner-level-ltm-off.md) -- reported None (undefined), never crashes aggregate()'s re-grade of those.
    off_neu = A.get("neu_night_off_intact")
    g["ordinary_fact_flip_forgetting"] = (None if off_neu is None else
                                          bool(off_neu["recall_outcome"] == "correct"
                                               and o.get("neu_night_intact") != "correct"))
    errs = sum(len(v["errors"]) for v in A.values())
    undefined = (not g["G0_null_clean"]) or (not g["P1_immediate_precondition"]) or g["G6_lesion_held"] is None \
        or errs > 0 or any(v == "undefined" for v in o.values()) or not g["G_isolation_gamma_consistent"]
    core = all(g[k] for k in ("G1_lesion_changes_next_day_reply", "G2_lesion_spares_immediate",
                              "G3_neutral_not_kept", "G4_off_lesion_no_change", "G5_no_confab", "G6_lesion_held"))
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


_DA_TONIC_REF = 0.5                 # == webapp.da_tag_capture._DA_TONIC (what both DA lesions pin the D1 read to)


def _epochs(arm):
    return ((arm or {}).get("sleep_replay_at_recall") or {}).get("epochs") or []


def grade_seed_rc(res):
    """The sleep-replay-capture family's pre-registered gates (research/findings/2026-09-24-sleep-replay-capture-
    PREREGISTRATION.md), verbatim. Pure function of res["arms"]."""
    A = res["arms"]
    o = {k: v["recall_outcome"] for k, v in A.items()}
    g = {}
    a, b = A["neu_night_rc_a"], A["neu_night_rc_b"]
    # G0 null control on the NEW path: the rebuild reproduces the decision, the ledger state and the sleep record
    g["G0_null_clean"] = bool(o["neu_night_rc_a"] == o["neu_night_rc_b"]
                              and a["recalled_svo"] == b["recalled_svo"] and a["abstained"] == b["abstained"]
                              and a["tag_capture_at_recall"] == b["tag_capture_at_recall"]
                              and a.get("blocks_at_recall") == b.get("blocks_at_recall")
                              and a.get("sleep_replay_at_recall") == b.get("sleep_replay_at_recall"))
    # P1 precondition: the plainly told fact is stored and recalled at once with the route armed
    g["P1_immediate_precondition"] = bool(o["neu_imm_rc"] == "correct")
    # I1 instrument: on every night arm with the route armed the replay branch EXECUTED (>= 1 SWR epoch, every block
    # read back by the store's own cleanup); on the immediate arm it did NOT (no sleep-depth idle)
    night = [k for k in A if "_night_rc" in k]
    g["I1_replay_branch_executed"] = bool(
        all(len(_epochs(A[k])) >= 1 and all(not e.get("no_reader") and e.get("R") is not None for e in _epochs(A[k]))
            for k in night)
        and len(_epochs(A["neu_imm_rc"])) == 0)
    # I2 every lesion held at measurement (read off the sleep record itself, not assumed from the env)
    held = True
    for k in night:
        env = A[k].get("env") or {}
        for e in _epochs(A[k]):
            if env.get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1":
                held &= bool(e.get("replay_lesioned") and all(r == 0.0 for r in e.get("R_eff") or [])
                             and abs(e.get("da_swr", -1) - _DA_TONIC_REF) < 1e-9)
            else:
                held &= not e.get("replay_lesioned")
            if env.get("BRAIN_DA_CAPTURE_LESION") == "1":
                held &= bool(e.get("capture_lesioned") and e.get("a_eff_mean") == 0.0)
            if env.get("BRAIN_DA_ENCODING_LESION") == "1":
                held &= bool(abs(e.get("da_seen_by_d1", -1) - _DA_TONIC_REF) < 1e-9)
    g["I2_lesions_held"] = bool(held)
    # the calibration is identical across every companion-ON arm (the Amendment-1 confound, re-checked here)
    gam = [v["tag_capture_at_recall"].get("gamma") for v in A.values()
           if (v.get("env") or {}).get("BRAIN_DA_TAG_CAPTURE") == "1" and v["tag_capture_at_recall"].get("gamma")]
    g["G_isolation_gamma_consistent"] = bool(not gam or all(abs(x - gam[0]) < 1e-6 for x in gam))
    # RC1 the route rescues the ordinary fact: kept with the flag on, gone with it off (ledger on in both)
    g["RC1_ordinary_fact_rescued"] = bool(o["neu_night_rc_a"] == "correct" and o["neu_night_norc"] == "abstain")
    # RC2 cutting the replay edge removes the rescue
    g["RC2_replay_lesion_removes_rescue"] = bool(o["neu_night_rc_replaylesion"] == "abstain")
    # RC3 salient-vs-neutral separation kept: with the replay edge cut the salient fact is still kept (its waking DA
    # capture alone) while the ordinary one is not (RC2) -- the G3 contrast, reproduced with the route armed
    g["RC3_salient_kept_without_replay_edge"] = bool(o["sal_night_rc_replaylesion"] == "correct")
    # RC4 the capture lesion still blocks salient capture (it gates the sleep route too)
    g["RC4_capture_lesion_blocks_salient"] = bool(o["sal_night_rc_caplesion"] == "abstain")
    # RC5 the DA-encoding lesion still changes the salient next-day reply with the route armed (G1 under the route)
    g["RC5_da_lesion_still_changes_next_day_reply"] = bool(o["sal_night_rc"] == "correct"
                                                            and o["sal_night_rc_dalesion"] == "abstain")
    g["RC6_no_confab"] = bool(all(v != "confab" for v in o.values()))
    # REPORTED (never gating)
    ep_a = _epochs(a)
    g["reported"] = {
        "neu_night_rc_dalesion_outcome": o.get("neu_night_rc_dalesion"),
        "R_epoch0": {k: (_epochs(A[k])[0].get("R") if _epochs(A[k]) else None) for k in night},
        "da_swr_epoch0": {k: (_epochs(A[k])[0].get("da_swr") if _epochs(A[k]) else None) for k in night},
        "a_eff_mean_epoch0": {k: (_epochs(A[k])[0].get("a_eff_mean") if _epochs(A[k]) else None) for k in night},
        "pre_sleep_frac_z_gt_half": {k: (_epochs(A[k])[0].get("pre_frac_z_gt_half") if _epochs(A[k]) else None)
                                     for k in ("neu_night_rc_a", "sal_night_rc")},
        "n_managed_blocks_neu_rc": (a["tag_capture_at_recall"] or {}).get("n_managed_blocks"),
        "epoch_t_h_neu_rc": (ep_a[0].get("t_h") if ep_a else None)}
    errs = sum(len(v["errors"]) for v in A.values())
    undefined = (not g["G0_null_clean"]) or (not g["P1_immediate_precondition"]) or (not g["I1_replay_branch_executed"]) \
        or (not g["I2_lesions_held"]) or (not g["G_isolation_gamma_consistent"]) or errs > 0 \
        or any(v == "undefined" for v in o.values())
    core = all(g[k] for k in ("RC1_ordinary_fact_rescued", "RC2_replay_lesion_removes_rescue",
                              "RC3_salient_kept_without_replay_edge", "RC4_capture_lesion_blocks_salient",
                              "RC5_da_lesion_still_changes_next_day_reply", "RC6_no_confab"))
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


def aggregate_rc(d):
    """6-seed combine for the rc family: GO iff all 6 pre-registered seeds are GO (re-graded with the current code)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("family") == "rc":
            rows.append(r)
    for r in rows:
        r["gates"] = grade_seed_rc(r)
    verdicts = {r["seed"]: r["gates"]["seed_verdict"] for r in rows}
    n_go = sum(1 for v in verdicts.values() if v == "GO")
    rescue = [int(r["gates"]["outcomes"]["neu_night_rc_a"] == "correct")
              - int(r["gates"]["outcomes"]["neu_night_norc"] == "correct") for r in rows]
    da_lb = [int(r["gates"]["outcomes"]["sal_night_rc"] == "correct")
             - int(r["gates"]["outcomes"]["sal_night_rc_dalesion"] == "correct") for r in rows]
    complete = sorted(verdicts) == sorted(SEEDS)
    verdict = "INCOMPLETE" if not complete else ("GO" if n_go == 6 else "NO-GO")
    out = {"family": "rc", "seeds": sorted(verdicts), "seed_verdicts": verdicts, "n_go": n_go, "verdict": verdict,
           "ordinary_fact_rescue_rate_on": (sum(1 for r in rows if r["gates"]["outcomes"]["neu_night_rc_a"] == "correct")
                                            / float(len(rows)) if rows else None),
           "ordinary_fact_rescue_rate_off": (sum(1 for r in rows if r["gates"]["outcomes"]["neu_night_norc"] == "correct")
                                             / float(len(rows)) if rows else None),
           "signflip_p_rescue_on_vs_off": (seed_signflip_p(rescue) if rows else None),
           "signflip_p_da_lesion_under_route": (seed_signflip_p(da_lb) if rows else None),
           "diffs_rescue_on_minus_off": rescue, "diffs_da_intact_minus_lesion_under_route": da_lb}
    json.dump(out, open(os.path.join(d, "aggregate.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def grade_seed_r2(res):
    """The r2 family's pre-registered read (Amendment 1 of research/findings/2026-09-24-sleep-replay-capture-
    PREREGISTRATION.md), verbatim. Item 1 (long delay) is REPORTED with its own verdict; item 2 (downscaling) is GATED.
    Pure function of res["arms"]."""
    A = res["arms"]
    o = {k: v["recall_outcome"] for k, v in A.items()}
    # the Amendment-3 horizon arms are REPORTED only: excluded from every error count, gate and UNDEFINED rule
    errs = sum(len(v["errors"]) for k, v in A.items() if k not in HORIZON_ARMS)
    g = {}
    gam = [v["tag_capture_at_recall"].get("gamma") for k, v in A.items() if k not in HORIZON_ARMS
           and (v.get("env") or {}).get("BRAIN_DA_TAG_CAPTURE") == "1" and v["tag_capture_at_recall"].get("gamma")]
    g["G_isolation_gamma_consistent"] = bool(not gam or all(abs(x - gam[0]) < 1e-6 for x in gam))
    # ── item 1: a fact told ~4 h before sleep onset (REPORTED) ──────────────────────────────────────────────────────
    g["P1_immediate_precondition"] = bool(o["neu_imm_rc"] == "correct")
    ld_ok = True
    for k in ("ld_rc", "ld_rc_replaylesion"):
        ep, aw = _epochs(A[k]), A[k].get("awake_until_h")
        ld_ok &= bool(len(ep) == 1 and aw is not None and aw >= AWAKE_H and ep[0].get("t_h", -1) > aw
                      and not ep[0].get("no_reader"))
    e_les = _epochs(A["ld_rc_replaylesion"])
    ld_ok &= bool(e_les and e_les[0].get("replay_lesioned") and all(r == 0.0 for r in e_les[0].get("R_eff") or [])
                  and abs(e_les[0].get("da_swr", -1) - _DA_TONIC_REF) < 1e-9)
    ld_ok &= bool(A["ld_norc"].get("awake_until_h") is not None)
    g["I_LD_sleep_after_waking_and_lesion_held"] = bool(ld_ok)
    e_ld = _epochs(A["ld_rc"])
    g["LD_reported"] = {
        "outcomes": {k: o[k] for k in ("ld_ledger_off", "ld_norc", "ld_rc", "ld_rc_replaylesion")},
        "R_at_sleep_onset": (e_ld[0].get("R") if e_ld else None),
        "da_swr": (e_ld[0].get("da_swr") if e_ld else None),
        "sleep_onset_h_after_telling": ((e_ld[0].get("t_h") - (A["ld_rc"].get("blocks_at_recall") or [{}])[0]
                                         .get("t_w", 0.0)) if e_ld and A["ld_rc"].get("blocks_at_recall") else None),
        "frac_z_gt_half_at_recall": [b.get("frac_synapses_z_gt_half") for b in (A["ld_rc"].get("blocks_at_recall")
                                                                                or [])]}
    ld_undef = (not g["P1_immediate_precondition"]) or (not ld_ok) or errs > 0 or not g["G_isolation_gamma_consistent"] \
        or any(o[k] == "undefined" for k in ("ld_ledger_off", "ld_norc", "ld_rc", "ld_rc_replaylesion"))
    if ld_undef:
        g["LD_verdict"] = "UNDEFINED"
    elif o["ld_rc"] == "correct" and o["ld_norc"] == "abstain" and o["ld_rc_replaylesion"] == "abstain":
        g["LD_verdict"] = "RESCUED"
    elif o["ld_rc"] == "abstain" and o["ld_norc"] == "abstain":
        g["LD_verdict"] = "NOT-RESCUED"
    else:
        g["LD_verdict"] = "OTHER"
    # ── item 2: three nights with sleep downscaling (GATED) ────────────────────────────────────────────────────────
    a, b = A["d3w_shy_a"], A["d3w_shy_b"]
    g["G0_null_clean"] = bool(o["d3w_shy_a"] == o["d3w_shy_b"] and a["recalled_svo"] == b["recalled_svo"]
                              and a["abstained"] == b["abstained"]
                              and a["tag_capture_at_recall"] == b["tag_capture_at_recall"]
                              and a.get("blocks_at_recall") == b.get("blocks_at_recall")
                              and a.get("sleep_replay_at_recall") == b.get("sleep_replay_at_recall"))
    g["P2_weak_fact_kept_without_downscaling"] = bool(o["d3w_rc"] == "correct")
    inst = True
    for k in ("d3w_rc", "d3w_shy_a", "d3w_shy_b", "d3c_shy", "d3r_shy"):
        ep = _epochs(A[k])
        shy_on = (A[k].get("env") or {}).get("BRAIN_SLEEP_DOWNSCALING") == "1"
        # d3w/d3c: one idle stretch of three nights -> 3 epochs; d3r: a re-mention each day -> 3 one-night stretches
        inst &= bool(len(ep) == 3 and all(not e.get("no_reader") for e in ep)
                     and all(("shy_scale" in e) == shy_on for e in ep))
    g["I_SHY_three_nights_and_scaling_as_armed"] = bool(inst)
    g["SHY1_weak_unmentioned_fact_fades"] = bool(o["d3w_shy_a"] == "abstain")
    g["SHY2_salient_fact_survives"] = bool(o["d3c_shy"] == "correct")
    g["SHY3_remention_fact_survives"] = bool(o["d3r_shy"] == "correct")
    g["SHY4_no_confab"] = bool(all(v != "confab" for v in o.values()))
    g["SHY_reported"] = {k: {"R_per_night": [e.get("R") for e in _epochs(A[k])],
                             "shy_scale_per_night": [e.get("shy_scale") for e in _epochs(A[k])],
                             "inc_mag_at_recall": [bl.get("inc_mag") for bl in (A[k].get("blocks_at_recall") or [])],
                             "base_mag_at_recall": [bl.get("base_mag") for bl in (A[k].get("blocks_at_recall") or [])],
                             "frac_z_gt_half_at_recall": [bl.get("frac_synapses_z_gt_half")
                                                          for bl in (A[k].get("blocks_at_recall") or [])]}
                         for k in ("d3w_rc", "d3w_shy_a", "d3c_shy", "d3r_shy")}
    undefined = (not g["G0_null_clean"]) or (not g["P2_weak_fact_kept_without_downscaling"]) or (not inst) \
        or errs > 0 or not g["G_isolation_gamma_consistent"] \
        or any(o[k] == "undefined" for k in ("d3w_rc", "d3w_shy_a", "d3w_shy_b", "d3c_shy", "d3r_shy"))
    core = all(g[k] for k in ("SHY1_weak_unmentioned_fact_fades", "SHY2_salient_fact_survives",
                              "SHY3_remention_fact_survives", "SHY4_no_confab"))
    hz = {}
    for k in HORIZON_ARMS:
        if k not in A:
            continue
        daily = A[k].get("daily_outcomes") or []
        first = next((i + 1 for i, v in enumerate(daily) if v != "correct"), None)
        hz[k] = {"daily_outcomes": daily, "first_night_not_correct": first,
                 "daily_inc_mag": A[k].get("daily_inc_mag"), "daily_base_mag": A[k].get("daily_base_mag"),
                 "R_per_night": [e.get("R") for e in _epochs(A[k])], "errors": A[k].get("errors")}
    g["SHY_horizon_reported"] = hz
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


def aggregate_r2(d):
    """6-seed combine for the r2 family: item 2 GO iff all 6 seeds GO; item 1 reported as a count per LD_verdict."""
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("family") == "r2":
            rows.append(r)
    for r in rows:
        r["gates"] = grade_seed_r2(r)
    verdicts = {r["seed"]: r["gates"]["seed_verdict"] for r in rows}
    n_go = sum(1 for v in verdicts.values() if v == "GO")
    complete = sorted(verdicts) == sorted(SEEDS)
    fade = [int(r["gates"]["outcomes"]["d3w_rc"] == "correct") - int(r["gates"]["outcomes"]["d3w_shy_a"] == "correct")
            for r in rows]
    ld = {}
    for r in rows:
        ld[r["gates"]["LD_verdict"]] = ld.get(r["gates"]["LD_verdict"], 0) + 1
    out = {"family": "r2", "seeds": sorted(verdicts), "seed_verdicts": verdicts, "n_go": n_go,
           "verdict": "INCOMPLETE" if not complete else ("GO" if n_go == 6 else "NO-GO"),
           "signflip_p_downscaling_fade": (seed_signflip_p(fade) if rows else None), "diffs_fade": fade,
           "long_delay_verdict_counts": ld,
           "horizon_first_night_not_correct": {k: {r["seed"]: (r["gates"].get("SHY_horizon_reported") or {})
                                                   .get(k, {}).get("first_night_not_correct") for r in rows}
                                               for k in HORIZON_ARMS}}
    json.dump(out, open(os.path.join(d, "aggregate.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def _bouts(arm):
    return ((arm or {}).get("awake_replay_at_recall") or {}).get("bouts") or []


def grade_seed_arc(res):
    """The awake-rest family's pre-registered gates (Amendment 4 of research/findings/2026-09-24-sleep-replay-capture-
    PREREGISTRATION.md), verbatim. Pure function of res["arms"]. The ARC_REPORTED_ARMS never enter a gate, an error
    count, the gamma check or an UNDEFINED rule -- except ARC7 (no confab), which reads every arm (declared)."""
    A = res["arms"]
    o = {k: v["recall_outcome"] for k, v in A.items()}
    gated = [k for k in A if k not in ARC_REPORTED_ARMS]
    errs = sum(len(A[k]["errors"]) for k in gated)
    g = {}
    a, b = A["lr_arc_a"], A["lr_arc_b"]
    # G0 null control on the new path: the rebuild reproduces the decision, the ledger, and both replay records
    g["G0_null_clean"] = bool(o["lr_arc_a"] == o["lr_arc_b"] and a["recalled_svo"] == b["recalled_svo"]
                              and a["abstained"] == b["abstained"]
                              and a["tag_capture_at_recall"] == b["tag_capture_at_recall"]
                              and a.get("blocks_at_recall") == b.get("blocks_at_recall")
                              and a.get("sleep_replay_at_recall") == b.get("sleep_replay_at_recall")
                              and a.get("awake_replay_at_recall") == b.get("awake_replay_at_recall"))
    g["P1_immediate_precondition"] = bool(o["neu_imm_arc"] == "correct")
    gam = [A[k]["tag_capture_at_recall"].get("gamma") for k in gated
           if (A[k].get("env") or {}).get("BRAIN_DA_TAG_CAPTURE") == "1" and A[k]["tag_capture_at_recall"].get("gamma")]
    g["G_isolation_gamma_consistent"] = bool(not gam or all(abs(x - gam[0]) < 1e-6 for x in gam))

    def _inst(k):
        """The awake branch ran exactly as the protocol scheduled it, before the one sleep epoch, every block read."""
        arm = A[k]
        env = arm.get("env") or {}
        bo, ep, aw = _bouts(arm), _epochs(arm), arm.get("awake_until_h")
        want = ARC_BOUTS.get(arm.get("label"), None)
        if env.get("BRAIN_AWAKE_REPLAY_CAPTURE") != "1":
            return arm.get("awake_replay_at_recall") is None          # flag off: no awake record at all
        if want is None or len(bo) != want:
            return False
        if any(x.get("no_reader") or x.get("R") is None or None in (x.get("R") or [None]) for x in bo):
            return False
        if arm.get("label") == "datni_recall":
            return len(ep) == 0                                       # no night: no epoch, no bout
        if len(ep) != 1 or aw is None or aw < AWAKE_H:
            return False
        return all(x["t_h"] <= aw + 1e-9 and x["t_h"] < ep[0]["t_h"] for x in bo)

    g["I1_awake_branch_as_scheduled"] = bool(all(_inst(k) for k in gated))
    g["I1_reported_arms_as_scheduled"] = {k: bool(_inst(k)) for k in ARC_REPORTED_ARMS if k in A}
    # I2 every lesion held at measurement, read off the records themselves
    held = True
    for k in gated:
        env = A[k].get("env") or {}
        for x in _bouts(A[k]):
            if env.get("BRAIN_AWAKE_REPLAY_CAPTURE_LESION") == "1":
                held &= bool(x.get("lesioned") and all(r == 0.0 for r in x.get("R_eff") or [])
                             and x.get("early_after") == x.get("early_before"))
            else:
                held &= not x.get("lesioned")
        if env.get("BRAIN_AWAKE_REPLAY_CAPTURE_LESION") == "1":
            held &= all("e_rep" not in bl for bl in (A[k].get("blocks_at_recall") or []))
        for e in _epochs(A[k]):
            if env.get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1":
                held &= bool(e.get("replay_lesioned") and all(r == 0.0 for r in e.get("R_eff") or [])
                             and abs(e.get("da_swr", -1) - _DA_TONIC_REF) < 1e-9)
            else:
                held &= not e.get("replay_lesioned")
            if env.get("BRAIN_DA_ENCODING_LESION") == "1":
                held &= bool(abs(e.get("da_seen_by_d1", -1) - _DA_TONIC_REF) < 1e-9)
    g["I2_lesions_held"] = bool(held)
    # I3 the awake bouts supply NO PRP: nothing is added to the D1 drive and the PRP pool only decays across the rest
    noprp = True
    for k in gated:
        bo = _bouts(A[k])
        if len(bo) >= 2:
            noprp &= len({x.get("n_drive_entries") for x in bo}) == 1
            noprp &= all(y.get("p_at_bout", 0.0) <= x.get("p_at_bout", 0.0) + 1e-15 for x, y in zip(bo, bo[1:]))
    g["I3_awake_bouts_add_no_prp"] = bool(noprp)
    # the pre-registered behavioural gates
    g["ARC1_rest_rescues_long_delay_fact"] = bool(o["lr_arc_a"] == "correct" and o["lr_noarc"] == "abstain")
    g["ARC2_awake_edge_lesion_removes_rescue"] = bool(o["lr_arc_lesion"] == "abstain")
    g["ARC3_no_rescue_without_rest"] = bool(o["ln_arc"] == "abstain")
    g["ARC4_rest_alone_does_not_make_it_permanent"] = bool(o["lr_arc_sleeplesion"] == "abstain")
    g["ARC5_da_lesion_blocks_the_rescue"] = bool(o["lr_arc_dalesion"] == "abstain")
    g["ARC6_salient_kept_on_waking_capture"] = bool(o["lsr_arc_sleeplesion"] == "correct")
    g["ARC7_no_confab"] = bool(all(v != "confab" for v in o.values()))

    def _rep(k):
        arm = A.get(k)
        if arm is None:
            return None
        bo, ep = _bouts(arm), _epochs(arm)
        return {"outcome": o.get(k), "n_bouts": len(bo),
                "R_first_bout": (bo[0]["R"] if bo else None), "R_last_bout": (bo[-1]["R"] if bo else None),
                "early_after_last_bout": (bo[-1]["early_after"] if bo else None),
                "R_at_sleep_onset": (ep[0].get("R") if ep else None), "da_swr": (ep[0].get("da_swr") if ep else None),
                "frac_z_gt_half_at_sleep_onset": (ep[0].get("pre_frac_z_gt_half") if ep else None),
                "frac_z_gt_half_at_recall": [bl.get("frac_synapses_z_gt_half")
                                             for bl in (arm.get("blocks_at_recall") or [])],
                "errors": arm.get("errors")}
    g["reported"] = {k: _rep(k) for k in A}
    undefined = (not g["G0_null_clean"]) or (not g["P1_immediate_precondition"]) \
        or (not g["I1_awake_branch_as_scheduled"]) or (not g["I2_lesions_held"]) \
        or (not g["I3_awake_bouts_add_no_prp"]) or (not g["G_isolation_gamma_consistent"]) or errs > 0 \
        or any(o[k] == "undefined" for k in gated)
    core = all(g[k] for k in ("ARC1_rest_rescues_long_delay_fact", "ARC2_awake_edge_lesion_removes_rescue",
                              "ARC3_no_rescue_without_rest", "ARC4_rest_alone_does_not_make_it_permanent",
                              "ARC5_da_lesion_blocks_the_rescue", "ARC6_salient_kept_on_waking_capture",
                              "ARC7_no_confab"))
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


def aggregate_arc(d):
    """6-seed combine for the arc family: GO iff all 6 pre-registered seeds are GO (re-graded with the current code)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("family") == "arc":
            rows.append(r)
    for r in rows:
        r["gates"] = grade_seed_arc(r)
    verdicts = {r["seed"]: r["gates"]["seed_verdict"] for r in rows}
    n_go = sum(1 for v in verdicts.values() if v == "GO")
    oc = lambda r, k: int(r["gates"]["outcomes"].get(k) == "correct")   # noqa: E731
    rescue = [oc(r, "lr_arc_a") - oc(r, "lr_noarc") for r in rows]
    edge = [oc(r, "lr_arc_a") - oc(r, "lr_arc_lesion") for r in rows]
    complete = sorted(verdicts) == sorted(SEEDS)
    out = {"family": "arc", "seeds": sorted(verdicts), "seed_verdicts": verdicts, "n_go": n_go,
           "verdict": "INCOMPLETE" if not complete else ("GO" if n_go == 6 else "NO-GO"),
           "signflip_p_rest_rescue_on_vs_off": (seed_signflip_p(rescue) if rows else None),
           "signflip_p_rest_rescue_intact_vs_awake_lesion": (seed_signflip_p(edge) if rows else None),
           "diffs_on_minus_off": rescue, "diffs_intact_minus_lesion": edge,
           "reported_correct_counts": {k: sum(oc(r, k) for r in rows) for k in ARC_REPORTED_ARMS}}
    json.dump(out, open(os.path.join(d, "aggregate.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def seed_signflip_p(diffs):
    """One-sided exact sign-flip p over seeds for mean(diff) > 0 (the seed is the unit of replication)."""
    obs = sum(diffs)
    n = len(diffs)
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n) if sum(s * d for s, d in zip(signs, diffs)) >= obs)
    return hits / float(2 ** n)


def aggregate(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        try:
            rows.append(json.load(open(p)))
        except Exception:
            continue
    # RE-GRADE every row with the CURRENT grade_seed rather than trusting the stored gates.seed_verdict (2026-09-24,
    # review v2:2a37f2493 of commit 5d3810f2d): a seed*.json written before a grading-logic fix lands (e.g. the
    # confounded research/findings/raw/_da_tag_capture_chat/seed42.json, run at c4c62d066 before the D1-reader-
    # isolation fix daa4b382d, whose stored gates read seed_verdict=GO) must not silently count toward n_go / the
    # 6-seed verdict on its stale grade. grade_seed is a pure function of res["arms"], so re-calling it here is
    # idempotent for a row already graded under the current code and corrective for one that is not.
    for r in rows:
        r["gates"] = grade_seed(r)
    verdicts = {r["seed"]: r["gates"]["seed_verdict"] for r in rows}
    n_go = sum(1 for v in verdicts.values() if v == "GO")
    diffs = [int(r["gates"]["outcomes"]["sal_night_intact_a"] == "correct")
             - int(r["gates"]["outcomes"]["sal_night_lesion"] == "correct") for r in rows]
    sal_vs_neu = [int(r["gates"]["outcomes"]["sal_night_intact_a"] == "correct")
                  - int(r["gates"]["outcomes"]["neu_night_intact"] == "correct") for r in rows]
    complete = sorted(verdicts) == sorted(SEEDS)
    if not complete:
        verdict = "INCOMPLETE"
    elif n_go == 6:
        verdict = "GO"
    elif n_go >= 4 and not any(v == "NO-GO" for v in verdicts.values()):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    out = {"seeds": sorted(verdicts), "seed_verdicts": verdicts, "n_go": n_go, "verdict": verdict,
           "signflip_p_intact_vs_lesion": (seed_signflip_p(diffs) if rows else None),
           "signflip_p_salient_vs_neutral": (seed_signflip_p(sal_vs_neu) if rows else None),
           "diffs_intact_minus_lesion": diffs, "diffs_salient_minus_neutral": sal_vs_neu}
    json.dump(out, open(os.path.join(d, "aggregate.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


# ── byte-identical OFF vs the pinned pre-change SHA ─────────────────────────────────────────────────────────────
_OFFCHECK_TURNS = [  # the salient next-day group, verbatim (texts duplicated here so a pinned tree can run it)
    "Guess what, something unbelievable happened at the circus today!",
    "You will never believe this crazy story, it is absolutely amazing!",
    "the cat chases the ball",
    "Everyone in the audience was screaming and laughing in total shock!",
    "Honestly it was the most astonishing spectacle anybody has ever witnessed!",
    "__NIGHT__",
    "what does the cat chase",
]


def offcheck_worker(repo, out):
    """Run the salient next-day group IN-PROCESS against the tree at `repo` with BRAIN_DA_TAG_CAPTURE unset; hash every
    reply and the composer's store synapses. The night is run by calling continuous_engine.tick_idle_sessions at
    now+24h directly (the pinned tree has no world-step label)."""
    import time as _time
    os.chdir(repo)
    sys.path.insert(0, repo)
    for k in ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_DA_ENCODING_LESION"):
        os.environ.pop(k, None)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    from webapp import server as S
    from webapp import continuous_engine as CE
    replies = []
    for i, msg in enumerate(_OFFCHECK_TURNS):
        if msg == "__NIGHT__":
            n = CE.tick_idle_sessions(S._SESSION_MOOD, S._get_affect_organ, now=_time.time() + 24 * 3600.0,
                                      selfinit_getter=S._get_selfinit_organ,
                                      episodic_getter=S._get_episodic_organ_existing,
                                      chat_getter=S._get_chat_existing)
            replies.append({"night_ticked": int(n or 0)})
            continue
        r = S.brain_chat(S.BrainChatRequest(session="offchk", message=msg, brain="tiny-demo", renderer="stub",
                                            rich=False, reset=(i == 0)))
        replies.append(json.loads(r.body))
    chat = S._BRAIN_CHATS.get(("offchk", "tiny-demo", "stub"))
    comp = chat.inner.composer
    store = [(int(p), int(q), complex(w).real, complex(w).imag) for (p, q, w) in comp.store_conns]
    rj = json.dumps(replies, sort_keys=True, default=str)
    sj = json.dumps(store)
    json.dump({"repo": repo, "replies_sha256": hashlib.sha256(rj.encode()).hexdigest(),
               "store_sha256": hashlib.sha256(sj.encode()).hexdigest(), "n_store_conns": len(store),
               "replies": replies}, open(out, "w"), indent=1, default=str)


def _ledger_scenario_hash(repo):
    """The SynapticTagCaptureLedger edit (block_offset / last_w / sync_from_store) must leave the v3 runner's use
    byte-identical: run one deterministic synthetic scenario (a_override drives, no brain) on the tree at `repo`."""
    code = r'''
import sys, json, hashlib, numpy as np
sys.path.insert(0, sys.argv[1])
from webapp.da_tag_capture import SynapticTagCaptureLedger
class C:
    D = 16
    def __init__(self): self.store_conns = []
rng = np.random.default_rng(3)
comp = C(); L = SynapticTagCaptureLedger(5, gamma=30.0)
for i, t in enumerate([0.0, 0.01, 0.02, 0.5]):
    L.observe_turn(t, 30/3600., 0.9, a_override=[0.6, 0.0, 0.8, 0.3][i])
    for k in range(C.D):
        comp.store_conns.append((k + 1, 100 + k, complex(*rng.standard_normal(2))))
    L.on_store(comp, t)
L.advance(comp, 1.0); L.advance(comp, 24.0)
print(hashlib.sha256(json.dumps([(p, q, w.real, w.imag) for (p, q, w) in comp.store_conns]).encode()).hexdigest())
'''
    r = subprocess.run([sys.executable, "-c", code, repo], capture_output=True, text=True, cwd=repo)
    return r.stdout.strip() or ("ERR " + r.stderr[-500:])


def _offcheck_first_diff(pinned_replies, branch_replies):
    """The index + content of the first reply that differs between the two trees' turn-by-turn reply lists (a
    per-index compare over the LONGER length; an extra turn on one side reads `None` on the other). None if the
    lists are equal. Pulled out of `offcheck()` so this diffing logic (not the git/subprocess machinery around
    it) can be selftested directly."""
    n = max(len(pinned_replies), len(branch_replies))
    for i in range(n):
        p = pinned_replies[i] if i < len(pinned_replies) else None
        b = branch_replies[i] if i < len(branch_replies) else None
        if p != b:
            return {"turn_index": i, "pinned": p, "branch": b}
    return None


# ── byte-identical OFF against a COUNTERFACTUAL built from the CURRENT tree (r2 item 3; replaces the PINNED_SHA check) ─
# WHY. A fixed historical pin conflates "this feature's own diff" with "everything else merged since" (Amendments 2 and
# 4 of the chat-wire prereg: it went stale twice, the second time at 19 hunks of unrelated server.py changes). The
# counterfactual here is derived on every run: the committed HEAD, and the committed HEAD with the feature's OWN commits
# reverse-applied. Nothing else differs, so it cannot go stale as main moves.
#   * feature commits = every non-merge commit reachable from HEAD that touches a FEATURE_MODULES file, or adds/removes
#     a line matching FEATURE_HOOK_RE in a FEATURE_HOOK_FILES file (the call sites the feature added to shared code);
#   * each is reverse-applied (newest first; a 3-way reverse, falling back per commit to a zero-context reverse that
#     matches only the feature's own lines) into a temporary worktree under /home/dant123/Projects/sim/.claude/
#     worktrees/, restricted to `production_scope()` -- webapp/, sim/, and the research/runners modules webapp imports.
#     Everything else (the instrument, the battery harness, standalone runners, findings/tests/docs) stays at HEAD in
#     both trees and so cannot confound the comparison; no feature reference may remain in the counterfactual webapp/;
#   * a failed reverse-apply makes the check UNDEFINED (never a pass);
#   * a same-tree NULL CONTROL (HEAD run twice) must be identical, else UNDEFINED (the reply is not deterministic
#     enough for a byte check);
#   * both temporary worktrees are removed afterwards.
FEATURE_MODULES = ["webapp/da_tag_capture.py", "webapp/da_tag_capture_chat.py", "webapp/sleep_replay_capture.py",
                   "webapp/awake_replay_capture.py"]   # (awake-replay-capture: the new module joins the feature)
FEATURE_HOOK_FILES = ["webapp/server.py", "webapp/continuous_engine.py"]
FEATURE_HOOK_RE = r"da_tag_capture|sleep_replay_capture|awake_replay_capture"
REVERT_HELD_EQUAL = ["research/findings", "tests", "docs", "research/biology", "research/queue", "research/coordination",
                     "research/runners/_da_tag_capture_chat_probe.py", "research/runners/onebrain_regression_battery.py",
                     "research/runners/load_bearing_fraction.py"]


def production_scope(repo=None):
    """The paths whose reversal can change a /api/brain-chat reply: webapp/, sim/, and every research/runners module
    some webapp/*.py imports (derived from the tree at `repo` on every call). Everything else a feature commit touched
    (standalone runners, findings, tests, docs) is held equal in both trees."""
    import re as _re
    repo = repo or _REPO
    mods = set()
    pat = _re.compile(r"research\.runners(?:\.(\w+)|\s+import\s+([\w ,()]+))")
    for root, _dirs, fnames in os.walk(os.path.join(repo, "webapp")):
        for fn in fnames:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(root, fn), errors="ignore") as fh:
                for m in pat.finditer(fh.read()):
                    if m.group(1):
                        mods.add(m.group(1))
                    elif m.group(2):
                        mods.update(x.strip() for x in m.group(2).replace("(", " ").replace(")", " ").split(",")
                                    if x.strip().split(" ")[0].isidentifier())
    paths = ["webapp", "sim"]
    for m in sorted(mods):
        name = m.split(" ")[0]
        if os.path.exists(os.path.join(repo, "research", "runners", name + ".py")):
            paths.append("research/runners/%s.py" % name)
    return [p for p in paths if p not in REVERT_HELD_EQUAL]
_WT_PARENT = "/home/dant123/Projects/sim/.claude/worktrees"
_OFF_ENV = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_DA_ENCODING_LESION", "BRAIN_DA_CAPTURE_LESION",
            "BRAIN_SLEEP_REPLAY_CAPTURE", "BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_SLEEP_DOWNSCALING")


def _git(repo, *args, inp=None):
    return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, input=inp)


def feature_commits(repo=None):
    """The feature's own commits, newest first, derived from the tree at `repo` (default: this checkout)."""
    repo = repo or _REPO
    mods = set(_git(repo, "log", "--no-merges", "--format=%H", "HEAD", "--", *FEATURE_MODULES).stdout.decode().split())
    hooks = set(_git(repo, "log", "--no-merges", "--format=%H", "-G", FEATURE_HOOK_RE, "HEAD", "--",
                     *FEATURE_HOOK_FILES).stdout.decode().split())
    sel = mods | hooks
    order = _git(repo, "log", "--no-merges", "--format=%H", "HEAD").stdout.decode().split()
    return [c for c in order if c in sel]


def _corpus_src():
    """data/corpus for the brain build: this checkout's, else the main checkout's (worktrees do not carry it)."""
    own = os.path.join(_REPO, "data", "corpus")
    if os.path.exists(own):
        return os.path.realpath(own)
    common = _git(_REPO, "rev-parse", "--git-common-dir").stdout.decode().strip()
    return os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(os.path.join(_REPO, common))), "data", "corpus"))


def _restore_files(wt, snap):
    """Put the files of one commit's patch back to their pre-apply content (worktree + index)."""
    for f, data in snap.items():
        path = os.path.join(wt, f)
        if data is None:
            if os.path.exists(path):
                os.remove(path)
            _git(wt, "rm", "--cached", "-q", "--ignore-unmatch", "--", f)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(data)
            _git(wt, "add", "--", f)


def build_counterfactual(wt, commits):
    """Reverse-apply `commits` (newest first) into worktree `wt`, restricted to `production_scope(wt)` (everything else
    stays at HEAD). Per commit: a 3-way reverse-apply first; if it conflicts (later, unrelated edits right next to a
    feature hunk), that commit's partial application is rolled back and it is retried with a ZERO-CONTEXT reverse-apply,
    which matches only the feature's own lines. Afterwards no feature reference may remain in webapp/
    (`residual_feature_refs`), else the counterfactual is incomplete. Returns a dict; `ok` False on any failure."""
    import re as _re
    scope = production_scope(wt)
    spec = scope              # an INCLUDE pathspec: only production-reachable paths are reverted
    steps, ok = [], True
    for c in commits:
        rng = ("%s^" % c, c)
        patch = _git(wt, "diff", "--binary", *rng, "--", *spec).stdout
        if not patch.strip():
            steps.append({"commit": c, "status": "nothing-in-scope"})
            continue
        files = _git(wt, "diff", "--name-only", *rng, "--", *spec).stdout.decode().split()
        snap = {}
        for f in files:
            path = os.path.join(wt, f)
            snap[f] = open(path, "rb").read() if os.path.exists(path) else None
        r = _git(wt, "apply", "-R", "--3way", inp=patch)
        conflicted = _git(wt, "diff", "--name-only", "--diff-filter=U").stdout.decode().split()
        if r.returncode == 0 and not conflicted:
            steps.append({"commit": c, "status": "reverted-3way"})
            continue
        _restore_files(wt, snap)
        patch0 = _git(wt, "diff", "--binary", "-U0", *rng, "--", *spec).stdout
        r0 = _git(wt, "apply", "-R", "--unidiff-zero", "--index", inp=patch0)
        if r0.returncode == 0:
            steps.append({"commit": c, "status": "reverted-zero-context", "3way_conflicted": conflicted})
            continue
        _restore_files(wt, snap)
        ok = False
        steps.append({"commit": c, "status": "FAILED", "conflicted": conflicted,
                      "stderr_3way": r.stderr.decode(errors="replace")[-400:],
                      "stderr_zero_context": r0.stderr.decode(errors="replace")[-400:]})
        break
    residual = []           # CODE references only (an import of a feature module); prose mentions do not count
    pat = _re.compile(r"^\s*(?:from\s+[\w.]*\b(?:da_tag_capture\w*|sleep_replay_capture)\b"
                      r"|(?:from\s+[\w.]+\s+import\s+[^#]*|import\s+[^#]*)\b(?:da_tag_capture\w*|sleep_replay_capture)\b)")
    for root, _dirs, fnames in os.walk(os.path.join(wt, "webapp")):
        for fn in fnames:
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                with open(path, errors="ignore") as fh:
                    for n, line in enumerate(fh, 1):
                        if pat.search(line):
                            residual.append("%s:%d" % (os.path.relpath(path, wt), n))
    if residual:
        ok = False
    changed = _git(wt, "diff", "--name-status", "HEAD").stdout.decode().splitlines()
    untracked = _git(wt, "ls-files", "--others", "--exclude-standard").stdout.decode().split()
    return {"ok": ok, "steps": steps, "changed_vs_head": changed, "untracked": untracked,
            "residual_feature_refs": residual[:50], "n_scope_paths": len(scope)}


def _run_worker(tree, tag, td):
    o = os.path.join(td, tag + ".json")
    env = dict(os.environ)
    for k in _OFF_ENV:
        env.pop(k, None)
    r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--offcheck-worker", tree, o],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(o):
        raise RuntimeError("offcheck worker %s failed: %s" % (tag, r.stderr[-2000:]))
    return json.load(open(o))


def offcheck(out, ltm="off"):
    """Flag unset on HEAD vs HEAD with the feature reverted (+ a HEAD-vs-HEAD null control). See the block comment."""
    if ltm == "off":
        os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"     # both trees; the workers inherit it (declared, as run_seed)
    head = _git(_REPO, "rev-parse", "HEAD").stdout.decode().strip()
    dirty = _git(_REPO, "status", "--porcelain", "--", "webapp", "sim", "research/runners").stdout.decode().split("\n")
    commits = feature_commits()
    tag = "offcheck-cf-%d" % os.getpid()
    wt_head, wt_cf = os.path.join(_WT_PARENT, tag + "-head"), os.path.join(_WT_PARENT, tag + "-cf")
    corpus = _corpus_src()
    res = {"method": "counterfactual: HEAD vs HEAD minus the feature's own commits (built from the current tree)",
           "head": head, "uncommitted_code_paths_ignored": [d for d in dirty if d.strip()], "ltm": ltm,
           "feature_commits": commits, "reverted_scope": "production_scope(): webapp/, sim/, webapp-imported runners"}
    try:
        for wt in (wt_head, wt_cf):
            r = _git(_REPO, "worktree", "add", "--detach", wt, head)
            if r.returncode != 0:
                raise RuntimeError("worktree add failed: %s" % r.stderr.decode(errors="replace")[-500:])
            os.makedirs(os.path.join(wt, "data"), exist_ok=True)
            os.symlink(corpus, os.path.join(wt, "data", "corpus"))
        cf = build_counterfactual(wt_cf, commits)
        res["counterfactual"] = cf
        if not cf["ok"]:
            res["byte_identical_off"] = None
            res["verdict"] = "UNDEFINED: the feature's own diff did not reverse-apply cleanly on the current tree"
        else:
            with tempfile.TemporaryDirectory() as td:
                a = _run_worker(wt_head, "head_a", td)
                b = _run_worker(wt_head, "head_b", td)
                c = _run_worker(wt_cf, "counterfactual", td)
            res["null_replies_identical"] = a["replies_sha256"] == b["replies_sha256"]
            res["null_store_identical"] = a["store_sha256"] == b["store_sha256"]
            res["replies_identical"] = a["replies_sha256"] == c["replies_sha256"]
            res["store_identical"] = a["store_sha256"] == c["store_sha256"]
            res["head_run"] = {"replies_sha256": a["replies_sha256"], "store_sha256": a["store_sha256"],
                               "n_store_conns": a["n_store_conns"]}
            res["counterfactual_run"] = {"replies_sha256": c["replies_sha256"], "store_sha256": c["store_sha256"],
                                         "n_store_conns": c["n_store_conns"]}
            if not (res["null_replies_identical"] and res["null_store_identical"]):
                res["byte_identical_off"] = None
                res["verdict"] = "UNDEFINED: the same tree run twice differs (null control failed)"
                fd = _offcheck_first_diff(a.get("replies") or [], b.get("replies") or [])
                if fd is not None:
                    res["null_first_diff"] = fd
            else:
                res["byte_identical_off"] = bool(res["replies_identical"] and res["store_identical"])
                res["verdict"] = "IDENTICAL" if res["byte_identical_off"] else "DIFFERENT"
                if not res["replies_identical"]:
                    fd = _offcheck_first_diff(c.get("replies") or [], a.get("replies") or [])
                    if fd is not None:
                        res["first_diff"] = fd          # "pinned" = the counterfactual, "branch" = HEAD
    finally:
        for wt in (wt_head, wt_cf):
            if os.path.exists(wt):
                _git(_REPO, "worktree", "remove", "--force", wt)
        _git(_REPO, "worktree", "prune")
        res["temp_worktrees_removed"] = not (os.path.exists(wt_head) or os.path.exists(wt_cf))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("counterfactual",)}, indent=2))
    return res


# ── selftest (no brain) ─────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    sys.path.insert(0, _REPO)
    import numpy as np
    from webapp import da_tag_capture_chat as W
    from webapp.da_tag_capture import SynapticTagCaptureLedger
    checks = {}

    class C:
        D = 8

        def __init__(self, n):
            self.store_conns = [(k % 8 + 1, 50 + k % 8, complex(1.0, 0.0)) for k in range(8 * n)]
    # block_offset: pre-existing blocks untouched, a new block managed
    comp = C(2)
    before = list(comp.store_conns)
    L = SynapticTagCaptureLedger(1, gamma=30.0, block_offset=2)
    comp.store_conns += [(k + 1, 60 + k, complex(0.0, 1.0)) for k in range(8)]
    L.on_store(comp, 0.0)
    L.advance(comp, 24.0)
    checks["block_offset leaves build-time blocks untouched"] = comp.store_conns[:16] == before
    checks["new block is managed"] = len(L.blocks) == 1
    # uncaptured block decays to its baseline at 24 h (no drive)
    w = np.array([c[2] for c in comp.store_conns[16:24]])
    checks["uncaptured block ~ baseline at 24 h"] = bool(np.allclose(w, L.blocks[0]["base"], atol=1e-6))
    # sync_from_store: a pure rescale is absorbed, a rewrite re-tags
    comp.store_conns[16:24] = [(p, q, 2.0 * x) for (p, q, x) in comp.store_conns[16:24]]
    s = L.sync_from_store(comp, 24.0)
    checks["external rescale detected"] = s == {"rescaled": 1, "rewritten": 0}
    comp.store_conns[16:24] = [(p, q, complex(3.0, float(i))) for i, (p, q, _x) in enumerate(comp.store_conns[16:24])]
    s = L.sync_from_store(comp, 25.0)
    checks["external rewrite re-tagged"] = s == {"rescaled": 0, "rewritten": 1} and L.blocks[0]["t_w"] == 25.0
    # flag OFF -> every hook is inert
    os.environ.pop("BRAIN_DA_TAG_CAPTURE", None)

    class Chat:
        pass
    ch = Chat()
    checks["off: observe_chat_turn is None"] = W.observe_chat_turn(ch, 42) is None
    checks["off: after_store_chat is None"] = W.after_store_chat(ch) is None
    checks["off: tick_chat is None"] = W.tick_chat(ch) is None
    checks["off: no ledger attached"] = not hasattr(ch, "_da_tag_capture")
    # clock modes
    os.environ["BRAIN_DA_TAG_CAPTURE_CLOCK"] = "turn"
    checks["clock turn mode parses"] = W.clock_mode() == "turn"
    os.environ.pop("BRAIN_DA_TAG_CAPTURE_CLOCK", None)
    checks["clock default is wall"] = W.clock_mode() == "wall"
    # private rng restores the global stream
    st = np.random.get_state()[1].copy()
    with W._private_rng(42, 3):
        np.random.random(10)
    checks["private rng restores global numpy state"] = bool(np.array_equal(st, np.random.get_state()[1]))
    # grading logic can fail: a synthetic seed where the lesion does not change the reply must NOT be GO
    base = {"recalled_svo": FACT, "abstained": False, "tag_capture_at_recall": {"p_max": 0.05},
            "fact_block_at_recall": None, "errors": []}
    arms = {n: dict(base, recall_outcome="correct") for n, _l, _e in ARMS}
    arms["neu_night_intact"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True)
    arms["sal_night_lesion"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True,
                                    tag_capture_at_recall={"p_max": 0.001})
    checks["grade: designed-GO pattern -> GO"] = grade_seed({"arms": arms})["seed_verdict"] == "GO"
    arms2 = dict(arms)
    arms2["sal_night_lesion"] = dict(base, recall_outcome="correct", tag_capture_at_recall={"p_max": 0.001})
    checks["grade: lesion no change -> NO-GO"] = grade_seed({"arms": arms2})["seed_verdict"] == "NO-GO"
    arms3 = dict(arms)
    arms3["neu_night_intact"] = dict(base, recall_outcome="correct")
    checks["grade: neutral kept -> NO-GO"] = grade_seed({"arms": arms3})["seed_verdict"] == "NO-GO"
    arms4 = dict(arms)
    arms4["sal_imm_intact"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True)
    checks["grade: no immediate recall -> UNDEFINED"] = grade_seed({"arms": arms4})["seed_verdict"] == "UNDEFINED"
    arms5 = dict(arms)
    arms5["sal_night_off_lesion"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True)
    checks["grade: off arm changes under lesion -> NO-GO"] = grade_seed({"arms": arms5})["seed_verdict"] == "NO-GO"
    arms6 = dict(arms)
    arms6["sal_night_intact_a"] = dict(base, recall_outcome="correct", tag_capture_at_recall={"p_max": 0.0})
    checks["grade: intact p_max 0 -> UNDEFINED"] = grade_seed({"arms": arms6})["seed_verdict"] == "UNDEFINED"
    checks["signflip 6/6 -> 1/64"] = abs(seed_signflip_p([1] * 6) - 1 / 64.0) < 1e-12
    # G_isolation_gamma_consistent: a record with no env/gamma at all (the synthetic `arms` pattern above) must not
    # be penalized for a field it never had -- the designed-GO pattern must still read GO.
    checks["grade: no gamma data anywhere -> gate exempt, still GO"] = \
        grade_seed({"arms": arms})["G_isolation_gamma_consistent"] is True
    # consistent gamma/d1_a_go across every companion-ON arm, companion-OFF arms excluded (they carry no D1 read) ->
    # the gate passes and the seed reads its ordinary verdict (GO, from the designed-GO base pattern).
    arms_iso_ok = {}
    for name, label, env in ARMS:
        rec = dict(arms[name])
        rec["env"] = dict(env)
        if env.get("BRAIN_DA_TAG_CAPTURE") == "1":
            rec["tag_capture_at_recall"] = dict(rec["tag_capture_at_recall"], gamma=40.0, d1_a_go=0.15)
        arms_iso_ok[name] = rec
    g_iso_ok = grade_seed({"arms": arms_iso_ok})
    checks["grade: isolation gamma consistent across ON arms -> gate passes, GO"] = \
        g_iso_ok["G_isolation_gamma_consistent"] is True and g_iso_ok["seed_verdict"] == "GO"
    # THE FAILING DIRECTION (the gate must be able to fail): one companion-ON arm reads a different gamma than the
    # rest -- the exact confound this fix closes (a lesion arm calibrating differently from an intact arm) -- and
    # the seed must go UNDEFINED even though every G1-G6 outcome is otherwise the designed-GO pattern.
    arms_iso_bad = {k: dict(v) for k, v in arms_iso_ok.items()}
    arms_iso_bad["sal_night_lesion"] = dict(arms_iso_bad["sal_night_lesion"],
                                            tag_capture_at_recall=dict(
                                                arms_iso_bad["sal_night_lesion"]["tag_capture_at_recall"],
                                                gamma=32.8, d1_a_go=0.187))
    g_bad = grade_seed({"arms": arms_iso_bad})
    checks["grade: isolation gamma MISMATCH -> gate fails"] = g_bad["G_isolation_gamma_consistent"] is False
    checks["grade: isolation gamma mismatch -> seed UNDEFINED"] = g_bad["seed_verdict"] == "UNDEFINED"
    # ordinary_fact_flip_forgetting (branch research/da-tag-capture-ltm-on, Amendment 3): must be able to read
    # True (the flip costs a fact that today's default keeps), False (no such cost), and None (an artifact from
    # before this arm existed) -- and must NEVER perturb seed_verdict (excluded from `core` by construction).
    g_designed = grade_seed({"arms": arms_iso_ok})
    checks["ordinary forgetting: designed-GO pattern (off=correct, on=abstain) -> True"] = \
        g_designed["ordinary_fact_flip_forgetting"] is True and g_designed["seed_verdict"] == "GO"
    arms_no_regress = dict(arms_iso_ok)
    arms_no_regress["neu_night_intact"] = dict(base, recall_outcome="correct")  # ON arm ALSO keeps it -> no cost
    g_no_regress = grade_seed({"arms": arms_no_regress})
    checks["ordinary forgetting: off=correct, on=correct -> False (no regression)"] = \
        g_no_regress["ordinary_fact_flip_forgetting"] is False
    arms_old = {k: v for k, v in arms_iso_ok.items() if k != "neu_night_off_intact"}  # pre-branch artifact shape
    g_old = grade_seed({"arms": arms_old})
    checks["ordinary forgetting: arm absent (old artifact) -> None, not a crash"] = \
        g_old["ordinary_fact_flip_forgetting"] is None
    checks["ordinary forgetting: excluded from core -- absent arm still reads its ordinary seed_verdict"] = \
        g_old["seed_verdict"] == g_designed["seed_verdict"]
    # aggregate() must RE-GRADE every seed.json with the CURRENT grade_seed, never trust a stored gates.seed_verdict
    # (2026-09-24, review v2:2a37f2493: the confounded research/findings/raw/_da_tag_capture_chat/seed42.json was
    # written under pre-fix code and its stored gates read seed_verdict=GO, but re-grading its own arms under the
    # isolation-consistency gate above reads UNDEFINED). Build one seed*.json on disk whose STORED gates say GO
    # (as if written by old code) but whose "arms" data is the isolation-mismatch pattern above, and confirm
    # aggregate() reports it as UNDEFINED, not GO.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        stale = {"seed": 900, "arms": arms_iso_bad,
                 "gates": {"seed_verdict": "GO", "outcomes": g_iso_ok["outcomes"]}}   # stale, pre-fix-shaped record
        json.dump(stale, open(os.path.join(_td, "seed900.json"), "w"))
        agg = aggregate(_td)
        checks["aggregate: re-grades a stale stored-GO row to the current UNDEFINED verdict"] = \
            agg["seed_verdicts"][900] == "UNDEFINED" and agg["seed_verdicts"][900] != stale["gates"]["seed_verdict"]
    # ── rc family (sleep-replay capture): the grader must be able to read GO, NO-GO and UNDEFINED ────────────────────
    def _rc_arm(name, env, outcome):
        ep = {"R": [0.7], "R_eff": [0.0 if env.get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1" else 0.7],
              "da_swr": (0.5 if env.get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1" else 1.02),
              "da_seen_by_d1": (0.5 if (env.get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1"
                                        or env.get("BRAIN_DA_ENCODING_LESION") == "1") else 1.02),
              "a_eff_mean": (0.0 if env.get("BRAIN_DA_CAPTURE_LESION") == "1" else 0.5),
              "replay_lesioned": env.get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1",
              "capture_lesioned": env.get("BRAIN_DA_CAPTURE_LESION") == "1", "no_reader": False,
              "pre_frac_z_gt_half": [0.0], "t_h": 0.1}
        rc_on = env.get("BRAIN_SLEEP_REPLAY_CAPTURE") == "1"
        night = "_night_" in name
        sr = ({"n_epochs": 1 if night else 0, "epochs": [ep] if night else []} if rc_on else None)
        return {"recall_outcome": outcome, "recalled_svo": FACT if outcome == "correct" else None,
                "abstained": outcome == "abstain", "env": dict(env), "errors": [],
                "tag_capture_at_recall": {"gamma": 32.77, "n_managed_blocks": 1}, "fact_block_at_recall": None,
                "blocks_at_recall": [], "sleep_replay_at_recall": sr}
    designed = {"neu_night_norc": "abstain", "neu_night_rc_a": "correct", "neu_night_rc_b": "correct",
                "neu_night_rc_replaylesion": "abstain", "neu_night_rc_dalesion": "abstain", "neu_imm_rc": "correct",
                "sal_night_rc": "correct", "sal_night_rc_dalesion": "abstain", "sal_night_rc_caplesion": "abstain",
                "sal_night_rc_replaylesion": "correct"}
    rc_ok = {n: _rc_arm(n, env, designed[n]) for n, _l, env in RC_ARMS}
    checks["rc grade: designed-GO pattern -> GO"] = grade_seed_rc({"arms": rc_ok})["seed_verdict"] == "GO"
    for arm_name, bad, want in (("neu_night_rc_a", "abstain", "UNDEFINED"),         # a != b rebuild -> G0 fails
                                ("neu_night_norc", "correct", "NO-GO"),             # flag-off already keeps it
                                ("neu_night_rc_replaylesion", "correct", "NO-GO"),  # lesion does not remove rescue
                                ("sal_night_rc_replaylesion", "abstain", "NO-GO"),  # separation lost
                                ("sal_night_rc_caplesion", "correct", "NO-GO"),     # capture lesion bypassed
                                ("sal_night_rc_dalesion", "correct", "NO-GO"),      # DA no longer load-bearing
                                ("neu_imm_rc", "abstain", "UNDEFINED")):            # precondition
        arms_x = dict(rc_ok)
        arms_x[arm_name] = _rc_arm(arm_name, rc_ok[arm_name]["env"], bad)
        checks["rc grade: %s=%s -> %s" % (arm_name, bad, want)] = grade_seed_rc({"arms": arms_x})["seed_verdict"] == want
    arms_x = dict(rc_ok)
    arms_x["sal_night_rc"] = dict(rc_ok["sal_night_rc"], sleep_replay_at_recall={"n_epochs": 0, "epochs": []})
    checks["rc grade: replay branch never executed -> UNDEFINED"] = \
        grade_seed_rc({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    arms_x = dict(rc_ok)
    bad_ep = dict(_epochs(rc_ok["neu_night_rc_replaylesion"])[0], R_eff=[0.7], replay_lesioned=False)
    arms_x["neu_night_rc_replaylesion"] = dict(rc_ok["neu_night_rc_replaylesion"],
                                               sleep_replay_at_recall={"n_epochs": 1, "epochs": [bad_ep]})
    checks["rc grade: replay lesion not held -> UNDEFINED"] = \
        grade_seed_rc({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    with _tf.TemporaryDirectory() as _td:
        for s in SEEDS:
            json.dump({"seed": s, "family": "rc", "arms": rc_ok}, open(os.path.join(_td, "seed%d.json" % s), "w"))
        agg = aggregate_rc(_td)
        checks["rc aggregate: 6 designed-GO seeds -> GO, p=1/64"] = \
            agg["verdict"] == "GO" and abs(agg["signflip_p_rescue_on_vs_off"] - 1 / 64.0) < 1e-12
    # ── r2 family: item 2 gated (GO / NO-GO / UNDEFINED), item 1 reported (RESCUED / NOT-RESCUED / UNDEFINED) ─────
    def _r2_arm(name, env, outcome):
        rec = _rc_arm(name, env, outcome)
        rc_on = env.get("BRAIN_SLEEP_REPLAY_CAPTURE") == "1"
        ep0 = (_epochs(_rc_arm("x_night_x", env, outcome)) or [{}])[0]      # the synthetic epoch shape, lesion-aware
        rec["sleep_replay_at_recall"] = ({"n_epochs": 0, "epochs": []} if rc_on else None)
        shy_on = env.get("BRAIN_SLEEP_DOWNSCALING") == "1"
        if name.startswith("ld_"):
            rec["awake_until_h"] = 4.01 if env.get("BRAIN_DA_TAG_CAPTURE") == "1" else None
            if rc_on:
                rec["sleep_replay_at_recall"] = {"n_epochs": 1, "epochs": [dict(ep0, t_h=4.1)]}
        if name.startswith("d3") and rc_on:
            eps = []
            for _n in range(3):
                ep = dict(ep0)
                if shy_on:
                    ep["shy_scale"] = [0.9]
                eps.append(ep)
            rec["sleep_replay_at_recall"] = {"n_epochs": 3, "epochs": eps}
        return rec
    r2_designed = {"ld_ledger_off": "correct", "ld_norc": "abstain", "ld_rc": "abstain",
                   "ld_rc_replaylesion": "abstain", "neu_imm_rc": "correct", "d3w_rc": "correct",
                   "d3w_shy_a": "abstain", "d3w_shy_b": "abstain", "d3c_shy": "correct", "d3r_shy": "correct",
                   "d10w_rc": "correct", "d10w_shy": "abstain"}
    r2_ok = {n: _r2_arm(n, env, r2_designed[n]) for n, _l, env in R2_ARMS}
    r2_ok["d10w_shy"]["daily_outcomes"] = ["correct"] * 6 + ["abstain"] * 4
    g_r2 = grade_seed_r2({"arms": r2_ok})
    checks["r2 grade: designed pattern -> item2 GO, item1 NOT-RESCUED"] = \
        g_r2["seed_verdict"] == "GO" and g_r2["LD_verdict"] == "NOT-RESCUED"
    checks["r2 horizon: first night not correct is read off the daily probe"] = \
        g_r2["SHY_horizon_reported"]["d10w_shy"]["first_night_not_correct"] == 7
    arms_x = dict(r2_ok)
    arms_x["d10w_shy"] = dict(r2_ok["d10w_shy"], errors=["boom"], recall_outcome="undefined")
    checks["r2 horizon: a horizon-arm error never touches the item-2 verdict (REPORTED only)"] = \
        grade_seed_r2({"arms": arms_x})["seed_verdict"] == "GO"
    arms_x = {k: v for k, v in r2_ok.items() if k not in HORIZON_ARMS}
    checks["r2 horizon: arms absent (pre-Amendment-3 file) -> still graded"] = \
        grade_seed_r2({"arms": arms_x})["seed_verdict"] == "GO"
    for arm_name, bad, key, want in (("d3w_shy_a", "correct", "seed_verdict", "UNDEFINED"),   # a != b rebuild
                                     ("d3c_shy", "abstain", "seed_verdict", "NO-GO"),         # salient lost
                                     ("d3r_shy", "abstain", "seed_verdict", "NO-GO"),         # re-mention lost
                                     ("d3w_rc", "abstain", "seed_verdict", "UNDEFINED"),      # nothing to fade
                                     ("ld_rc", "correct", "LD_verdict", "RESCUED"),           # route on keeps it
                                     ("neu_imm_rc", "abstain", "LD_verdict", "UNDEFINED")):
        arms_x = dict(r2_ok)
        arms_x[arm_name] = _r2_arm(arm_name, r2_ok[arm_name]["env"], bad)
        checks["r2 grade: %s=%s -> %s %s" % (arm_name, bad, key, want)] = grade_seed_r2({"arms": arms_x})[key] == want
    arms_x = dict(r2_ok)
    arms_x["d3w_shy_a"] = _r2_arm("d3w_shy_a", r2_ok["d3w_shy_a"]["env"], "correct")
    arms_x["d3w_shy_b"] = _r2_arm("d3w_shy_b", r2_ok["d3w_shy_b"]["env"], "correct")
    checks["r2 grade: weak fact NOT faded (both rebuilds) -> NO-GO"] = grade_seed_r2({"arms": arms_x})["seed_verdict"] == "NO-GO"
    arms_x = dict(r2_ok)
    arms_x["ld_rc"] = _r2_arm("ld_rc", r2_ok["ld_rc"]["env"], "correct")
    arms_x["ld_rc_replaylesion"] = _r2_arm("ld_rc_replaylesion", r2_ok["ld_rc_replaylesion"]["env"], "correct")
    checks["r2 grade: long-delay kept even with the replay edge cut -> OTHER (not attributable to the route)"] = \
        grade_seed_r2({"arms": arms_x})["LD_verdict"] == "OTHER"
    arms_x = dict(r2_ok)
    rec_x = dict(r2_ok["d3w_rc"])
    rec_x["sleep_replay_at_recall"] = {"n_epochs": 1, "epochs": _epochs(r2_ok["d3w_rc"])[:1]}
    arms_x["d3w_rc"] = rec_x
    checks["r2 grade: only one night ran -> UNDEFINED"] = grade_seed_r2({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    arms_x = dict(r2_ok)
    arms_x["ld_rc"] = dict(r2_ok["ld_rc"], sleep_replay_at_recall={"n_epochs": 1,
                                                                   "epochs": [dict(_epochs(r2_ok["ld_rc"])[0], t_h=0.1)]})
    checks["r2 grade: sleep before the waking interval ended -> LD UNDEFINED"] = \
        grade_seed_r2({"arms": arms_x})["LD_verdict"] == "UNDEFINED"
    # ── arc family (awake-rest replay): the grader must be able to read GO, NO-GO and UNDEFINED ─────────────────────
    def _arc_arm(name, label, env, outcome):
        rec = _rc_arm("x_night_x" if label != "datni_recall" else "x_imm", env, outcome)
        rec["label"] = label
        arc_on, les = env.get("BRAIN_AWAKE_REPLAY_CAPTURE") == "1", env.get("BRAIN_AWAKE_REPLAY_CAPTURE_LESION") == "1"
        n = ARC_BOUTS[label]
        bouts = [{"t_h": 0.05 + (k + 1) * (4.0 / max(n, 1)), "R": [0.4], "R_eff": [0.0 if les else 0.4],
                  "early_before": [0.9], "early_after": [0.9 if les else 0.94], "lesioned": les, "no_reader": False,
                  "p_at_bout": 0.01 * 0.9 ** k, "n_drive_entries": 6} for k in range(n)]
        rec["awake_replay_at_recall"] = ({"n_bouts": n, "bouts": bouts} if (arc_on and n) else None)
        if label == "datni_recall":
            rec["awake_until_h"], rec["sleep_replay_at_recall"] = None, {"n_epochs": 0, "epochs": []}
        else:
            rec["awake_until_h"] = 4.05 if env.get("BRAIN_DA_TAG_CAPTURE") == "1" else None
            if rec["sleep_replay_at_recall"]:
                rec["sleep_replay_at_recall"] = {"n_epochs": 1, "epochs": [dict(_epochs(rec)[0], t_h=4.13)]}
        return rec
    arc_designed = {"lr_arc_a": "correct", "lr_arc_b": "correct", "lr_noarc": "abstain", "lr_arc_lesion": "abstain",
                    "ln_arc": "abstain", "lr_arc_sleeplesion": "abstain", "lr_arc_dalesion": "abstain",
                    "lsr_arc_sleeplesion": "correct", "neu_imm_arc": "correct", "lq_arc": "abstain",
                    "lz_arc": "correct", "lr_ledger_off": "correct"}
    arc_ok = {n: _arc_arm(n, lab, env, arc_designed[n]) for n, lab, env in ARC_ARMS}
    checks["arc grade: designed-GO pattern -> GO"] = grade_seed_arc({"arms": arc_ok})["seed_verdict"] == "GO"
    for arm_name, bad, want in (("lr_arc_a", "abstain", "UNDEFINED"),          # a != b rebuild -> G0 fails
                                ("lr_noarc", "correct", "NO-GO"),              # rest alone (flag off) keeps it
                                ("lr_arc_lesion", "correct", "NO-GO"),         # lesion does not remove the rescue
                                ("ln_arc", "correct", "NO-GO"),                # rescued without any rest
                                ("lr_arc_sleeplesion", "correct", "NO-GO"),    # awake replay alone makes it permanent
                                ("lr_arc_dalesion", "correct", "NO-GO"),       # DA no longer gates it
                                ("lsr_arc_sleeplesion", "abstain", "NO-GO"),   # separation lost
                                ("neu_imm_arc", "abstain", "UNDEFINED"),       # precondition
                                ("lq_arc", "correct", "GO"),                   # a REPORTED arm never moves the verdict
                                ("lr_ledger_off", "abstain", "GO"),
                                ("lz_arc", "confab", "NO-GO")):                # ... except a confab (ARC7 reads all)
        arms_x = dict(arc_ok)
        lab = [l for n, l, _e in ARC_ARMS if n == arm_name][0]
        arms_x[arm_name] = _arc_arm(arm_name, lab, arc_ok[arm_name]["env"], bad)
        checks["arc grade: %s=%s -> %s" % (arm_name, bad, want)] = grade_seed_arc({"arms": arms_x})["seed_verdict"] == want
    arms_x = dict(arc_ok)
    arms_x["lr_arc_lesion"] = dict(arc_ok["lr_arc_lesion"], awake_replay_at_recall=arc_ok["lr_arc_a"]["awake_replay_at_recall"])
    checks["arc grade: awake lesion not held on the record -> UNDEFINED"] = \
        grade_seed_arc({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    arms_x = dict(arc_ok)
    arms_x["lr_arc_a"] = dict(arc_ok["lr_arc_a"], awake_replay_at_recall={"n_bouts": 3, "bouts": _bouts(arc_ok["lr_arc_a"])[:3]})
    arms_x["lr_arc_b"] = dict(arms_x["lr_arc_a"])
    checks["arc grade: fewer bouts than the rest ticks scheduled -> UNDEFINED"] = \
        grade_seed_arc({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    arms_x = dict(arc_ok)
    bo = [dict(x) for x in _bouts(arc_ok["lr_arc_a"])]
    bo[5]["n_drive_entries"] = 7
    arms_x["lr_arc_a"] = dict(arc_ok["lr_arc_a"], awake_replay_at_recall={"n_bouts": 48, "bouts": bo})
    arms_x["lr_arc_b"] = dict(arms_x["lr_arc_a"])
    checks["arc grade: a bout added D1 drive (PRP) -> UNDEFINED"] = \
        grade_seed_arc({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    arms_x = dict(arc_ok)
    arms_x["lr_noarc"] = dict(arc_ok["lr_noarc"], awake_replay_at_recall=arc_ok["lr_arc_a"]["awake_replay_at_recall"])
    checks["arc grade: an awake record on a flag-OFF arm -> UNDEFINED"] = \
        grade_seed_arc({"arms": arms_x})["seed_verdict"] == "UNDEFINED"
    with _tf.TemporaryDirectory() as _td:
        for s in SEEDS:
            json.dump({"seed": s, "family": "arc", "arms": arc_ok}, open(os.path.join(_td, "seed%d.json" % s), "w"))
        agg = aggregate_arc(_td)
        checks["arc aggregate: 6 designed-GO seeds -> GO, p=1/64"] = \
            agg["verdict"] == "GO" and abs(agg["signflip_p_rest_rescue_on_vs_off"] - 1 / 64.0) < 1e-12
    # counterfactual offcheck: the feature's own commits are derived from the tree (never a fixed pin)
    sc = production_scope()
    checks["offcheck counterfactual: scope = production-reachable paths (webapp-imported runner in, instrument out)"] = \
        ("webapp" in sc and "research/runners/_da_write_gain_spiking_derisk.py" in sc
         and "research/runners/_da_tag_capture_chat_probe.py" not in sc
         and "research/runners/onebrain_regression_battery.py" not in sc)
    fc = feature_commits()
    checks["offcheck counterfactual: feature commits derived from HEAD (>= the 5 known)"] = \
        len(fc) >= 5 and all(any(c.startswith(k) for c in fc)
                             for k in ("48bb87bef", "492231df3", "a201293f5", "daa4b382d", "fd664ef3f"))
    # _offcheck_first_diff (Amendment 4 candidate fix, 2026-09-24): a mismatch must be DIAGNOSABLE (which turn,
    # what content) without a full two-tree re-run. Both directions: identical lists -> None (never falsely
    # flags a diff); a real difference -> the correct index + both sides' content, including the "one tree ran
    # an extra turn" case (index past the shorter list's end reads None on that side, not an IndexError).
    same = [{"a": 1}, {"a": 2}, {"a": 3}]
    checks["offcheck first_diff: identical lists -> None"] = _offcheck_first_diff(same, list(same)) is None
    diff_at_1 = [{"a": 1}, {"a": 2}, {"a": 3}]
    diff_at_1b = [{"a": 1}, {"a": 99}, {"a": 3}]
    fd = _offcheck_first_diff(diff_at_1, diff_at_1b)
    checks["offcheck first_diff: finds the first differing turn, both sides' content"] = \
        fd == {"turn_index": 1, "pinned": {"a": 2}, "branch": {"a": 99}}
    fd_extra = _offcheck_first_diff([{"a": 1}], [{"a": 1}, {"a": 2}])
    checks["offcheck first_diff: an extra turn on one side reads None on the other, not a crash"] = \
        fd_extra == {"turn_index": 1, "pinned": None, "branch": {"a": 2}}
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    ok = all(checks.values())
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ltm", choices=["off", "on"], default="off")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--family", choices=["base", "rc", "r2", "arc"], default="base",
                    help="base = the G0-G6 family (unchanged); rc = the sleep-replay-capture family (RC_ARMS); "
                         "r2 = long delay + sleep downscaling (R2_ARMS); arc = awake-rest replay (ARC_ARMS)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None,
                    help="comma list of arm names: run only these, ungraded (a de-risk subset, never a gate row)")
    ap.add_argument("--aggregate")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--offcheck", action="store_true",
                    help="byte-identical OFF: HEAD vs HEAD minus the feature's own commits (counterfactual, r2)")
    ap.add_argument("--offcheck-worker", nargs=2)
    a = ap.parse_args()
    if a.offcheck_worker:
        return offcheck_worker(*a.offcheck_worker)
    if a.selftest:
        return 0 if selftest() else 1
    if a.out is None:                                   # base keeps its pre-branch default exactly
        a.out = {"rc": RC_OUT, "r2": R2_OUT, "arc": ARC_OUT}.get(a.family, "research/findings/raw/_da_tag_capture_chat")
    if a.offcheck:
        return 0 if offcheck(a.out, ltm=a.ltm)["byte_identical_off"] else 1
    if a.aggregate:
        {"rc": aggregate_rc, "r2": aggregate_r2, "arc": aggregate_arc}.get(a.family, aggregate)(a.aggregate)
        return 0
    if a.seed is None:
        ap.error("--seed required")
    run_seed(a.seed, a.out, ltm=a.ltm, workers=a.workers, family=a.family,
             only=([x.strip() for x in a.only.split(",") if x.strip()] if a.only else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
