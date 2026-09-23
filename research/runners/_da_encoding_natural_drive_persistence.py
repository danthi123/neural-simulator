"""DA-GATED ENCODING under a NATURAL conversational drive, measured where the biology says it acts: next-day recall.

QUESTION. Does the brain's OWN dopamine, driven by a naturally surprising statement (no induced arousal, no swept
read-damage knob), decide whether a fact told in conversation is still recalled 24 h later, and does lesioning the
DA->encoding edge change that reply?

WHY 24 H AND NOT A CLEAN IMMEDIATE READ. The 2026-09-20 natural probe (branch research/gap-da-gated-encoding-v2) showed
the DA write gain is invisible to an immediate clean read. Bethus, Tse & Morris 2010 found the same in rats: D1/D5
blockade left encoding and immediate recall intact and changed PERSISTENCE over ~24 h. The substrate had replaced the
two processes persistence depends on with constants (infinite E-LTP lifetime; zero synaptic baseline). The companion
`webapp/da_tag_capture.py` (default OFF, `BRAIN_DA_TAG_CAPTURE`) restores them with pre-registered biology constants.

THE NATURAL DRIVE (host = the world only). Two conversations built from the SAME four facts:
  * NEUTRAL: each fact's words are introduced first in short plain turns ("the zebra is here"), then the fact is told
    plainly ("the zebra swallowed the violin"). It is expected, so the spiking habituation organ reads it as familiar.
  * SALIENT: three content-free turns ("ok"), then the fact told as surprising news with fresh, rich content
    ("Guess what, at the circus today the zebra swallowed the violin completely whole!").
Every turn is read by the production DA path: `DaModeDrivesWorkspace.observe` (spiking habituation novelty -> shared
spiking salience afferent -> spiking SNc -> self-produced DA). The DA level at a fact turn feeds the production write
gain (`da_encoding_drives_chat.install_encoding_gain`, spiking gain population) and the capture machinery.
Host shortcuts, declared: the runner is the parse boundary (it calls `comp.store(agent, action, patient)` for the fact
turns); the world clock is 30 s per turn and 24 h of delay; the capture/decay maintenance is host arithmetic on the
store synapses (see the companion module).

ARMS (each a fresh composer build at the same seed):
  intact              companion on, all edges intact
  lesion_da_encoding  BRAIN_DA_ENCODING_LESION=1 -> write gain pinned to 1 AND the PRP read pinned to tonic
                      (the lesion the load-bearing battery uses for da-gated-encoding)
  lesion_capture      BRAIN_DA_CAPTURE_LESION=1 -> only the DA->capture sub-edge severed (write gain intact)
  companion_off       no ledger = the production default today (reproduces the 2026-09-20 honest negative)
  lesion_novelty      DA trace recomputed with BRAIN_SPIKING_NOVELTY_LESION=1 (exploratory, not gated)

Reply = `query_patient(agent, action)` for each fact: the patient word, or None (the brain abstains: "I don't know").

Run one seed (numpy CPU, ~15-25 min, always under memcap):
  bash tools/memcap.sh 6 -- .venv/bin/python -u -m research.runners._da_encoding_natural_drive_persistence \
      --seed 42 --out research/findings/raw/_da_encoding_natural_drive/seed42.json
Selftest (instrument checks only, ~1 min): ... --selftest
Aggregate: ... --aggregate research/findings/raw/_da_encoding_natural_drive
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import itertools
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]
TURN_H = 30.0 / 3600.0          # 30 s per conversational turn (world clock)
IMMEDIATE_H = 1.0 / 60.0        # the immediate recall: 1 min after the last turn
DELAY_H = 24.0                  # next-day recall (Bethus 2010 / Wang 2010 persistence test)
FILLERS = ["ok", "oh", "hm"]    # content-free turns (<3 letters -> no content tokens -> the DA EMA HOLDS)

FACTS = [
    dict(lem=("zebra", "swallow", "violin"),
         primes=["the zebra is here", "the violin is here", "the zebra saw the violin"],
         plain="the zebra swallowed the violin",
         surprise="Guess what, at the circus today the zebra swallowed the violin completely whole!"),
    dict(lem=("otter", "steal", "lantern"),
         primes=["the otter is here", "the lantern is here", "the otter saw the lantern"],
         plain="the otter stole the lantern",
         surprise="Unbelievable news from the harbor: the otter stole the lantern right off a fishing boat!"),
    dict(lem=("goat", "eat", "passport"),
         primes=["the goat is here", "the passport is here", "the goat saw the passport"],
         plain="the goat ate the passport",
         surprise="You won't believe it, at the airport the goat ate the passport of a traveling diplomat!"),
    dict(lem=("parrot", "hide", "key"),
         primes=["the parrot is here", "the key is here", "the parrot saw the key"],
         plain="the parrot hid the key",
         surprise="Crazy story: during the thunderstorm the parrot hid the key inside the grandfather clock!"),
]
VOCAB = sorted({w for f in FACTS for w in f["lem"]} | {"dog", "cat", "see"})

# pre-registered band points (beta = baseline/increment ratio; tau_e = E-LTP decay, hours). PRIMARY first.
PRIMARY = (1.0, 1.5)
BAND = [(0.67, 1.5), (2.0, 1.5), (1.0, 1.0), (1.0, 3.0)]
ARMS_PRIMARY = ["intact", "lesion_da_encoding", "lesion_capture", "companion_off", "lesion_novelty"]
ARMS_BAND = ["intact", "lesion_da_encoding"]
ARM_ENV = {
    "intact": {},
    "lesion_da_encoding": {"BRAIN_DA_ENCODING_LESION": "1"},
    "lesion_capture": {"BRAIN_DA_CAPTURE_LESION": "1"},
    "companion_off": {},
    "lesion_novelty": {},
}
# the production-default DA-encoding env, pinned EXPLICITLY (off-arm discipline: never rely on unset == default)
BASE_ENV = {"BRAIN_DA_ENCODING": "1", "BRAIN_DA_ENCODING_SUBSTRATE": "1", "BRAIN_DA_ENCODING_SPIKING_GAIN": "1",
            "BRAIN_DA_ENCODING_LESION": "0", "BRAIN_DA_CAPTURE_LESION": "0", "BRAIN_SPIKING_NOVELTY_LESION": "0"}


@contextlib.contextmanager
def _env(overrides):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def conversation(condition):
    """[(text, fact_index_or_None)] -- equal turn counts in both conditions (4 facts x 4 turns)."""
    turns = []
    for i, f in enumerate(FACTS):
        if condition == "neutral":
            turns += [(p, None) for p in f["primes"]] + [(f["plain"], i)]
        elif condition == "salient":
            turns += [(x, None) for x in FILLERS] + [(f["surprise"], i)]
        else:
            raise ValueError(condition)
    return turns


def da_trace(seed, condition, novelty_lesion=False):
    """The brain's own DA read for every turn of the conversation (production DA path, fresh workspace)."""
    with _env({"BRAIN_SPIKING_NOVELTY_LESION": "1" if novelty_lesion else "0"}):
        from webapp.da_mode_drives_chat import DaModeDrivesWorkspace
        ws = DaModeDrivesWorkspace(seed=seed)
        out = []
        for text, fi in conversation(condition):
            info = ws.observe(text)
            nov = (info.get("spiking_novelty") or {}).get("novelty")
            out.append({"text": text, "fact": fi, "da": float(info["da_level"]), "mode": info["mode"],
                        "novelty": (None if nov is None else float(nov)),
                        "ema_engagement": float(info["ema_engagement"])})
    return out


class _ShimChat:
    """The two attributes the production DA-encoding coupling reads off a ChatBrain: inner.composer + _last_da_drives."""

    class _Inner:
        def __init__(self, comp):
            self.composer = comp

    def __init__(self, comp):
        self.inner = self._Inner(comp)
        self._last_da_drives = {"da_level": 0.5}


# composer block dimension. 64 = the 2026-09-20 natural probe's value and the v1 pre-registered run; 128 = the
# production default (BrainConversationalAgent D=128), pre-registered as v2 after v1 read UNDEFINED on G2.
COMPOSER_D = 64


def _build_composer(seed):
    from research.runners.one_brain_composer import OneBrainComposer
    return OneBrainComposer(seed=seed, D=COMPOSER_D, vocab=list(VOCAB), k_max=16)


def _recall(comp):
    out = []
    for f in FACTS:
        a, act, p = f["lem"]
        ans = comp.query_patient(a, act)
        out.append({"cue": [a, act], "want": p, "got": ans,
                    "outcome": ("correct" if ans == p else ("abstain" if ans is None else "confab"))})
    return out


def _counts(rec):
    return {k: sum(1 for r in rec if r["outcome"] == k) for k in ("correct", "abstain", "confab")}


def run_arm(seed, condition, arm, trace, beta, tau_e):
    """One fresh brain: tell the conversation, idle-tick homeostasis, recall at +1 min and at +24 h."""
    from webapp import da_encoding_drives_chat as DAE
    from webapp.da_tag_capture import TagCaptureLedger
    env = dict(BASE_ENV)
    env.update(ARM_ENV[arm])
    with _env(env):
        comp = _build_composer(seed)
        chat = _ShimChat(comp)
        DAE.install_encoding_gain(chat)
        ledger = None if arm == "companion_off" else TagCaptureLedger(seed, beta=beta, tau_early_h=tau_e)
        conv = conversation(condition)
        gains = []
        for ti, ((text, fi), tr) in enumerate(zip(conv, trace)):
            t = ti * TURN_H
            chat._last_da_drives = {"da_level": tr["da"]}
            if ledger is not None:
                ledger.observe_da(t, tr["da"])
            if fi is not None:
                gains.append(float(DAE.encoding_gain_for(chat, advance=False)))
                comp.store(*FACTS[fi]["lem"])
                if ledger is not None:
                    ledger.on_store(comp, t)
        t_end = len(conv) * TURN_H
        scales = DAE.apply_substrate_homeostasis(chat)          # the production idle-tick pass (once per batch)
        if ledger is not None and scales:
            ledger.apply_homeostasis_scales(scales)
        if ledger is not None:
            ledger.advance(comp, t_end + IMMEDIATE_H)
        imm = _recall(comp)
        if ledger is not None:
            ledger.advance(comp, t_end + DELAY_H)
        late = _recall(comp)
        return {"arm": arm, "condition": condition, "beta": beta, "tau_e": tau_e, "write_gains": gains,
                "homeostasis_scales": (None if scales is None else [float(s) for s in scales]),
                "prp_events_h": (None if ledger is None else list(ledger.prp_events)),
                "blocks": (None if ledger is None else ledger.summary(t_end + DELAY_H)),
                "immediate": imm, "immediate_counts": _counts(imm),
                "delayed_24h": late, "delayed_counts": _counts(late)}


def _exact_perm_p(a_correct, b_correct, n):
    """One-sided exact permutation p for 'arm A recalls more facts than arm B' over 2n binary fact outcomes."""
    obs = a_correct - b_correct
    pool = [1] * (a_correct + b_correct) + [0] * (2 * n - a_correct - b_correct)
    hits = tot = 0
    for idx in itertools.combinations(range(2 * n), n):
        s = set(idx)
        ca = sum(pool[i] for i in s)
        cb = sum(pool[i] for i in range(2 * n) if i not in s)
        tot += 1
        hits += (ca - cb) >= obs
    return hits / tot


def run_seed(seed, out):
    t0 = time.time()
    res = {"seed": seed, "composer_D": COMPOSER_D, "turn_h": TURN_H, "delay_h": DELAY_H, "vocab": VOCAB, "primary": list(PRIMARY),
           "band": [list(b) for b in BAND], "traces": {}, "arms": []}
    for cond in ("neutral", "salient"):
        res["traces"][cond] = da_trace(seed, cond)
        res["traces"][cond + "_novelty_lesion"] = da_trace(seed, cond, novelty_lesion=True)
        print("[seed %d] trace %s done %.0fs" % (seed, cond, time.time() - t0), flush=True)
    rep = da_trace(seed, "salient")
    res["da_trace_repeat_identical"] = [r["da"] for r in rep] == [r["da"] for r in res["traces"]["salient"]]
    for cond in ("neutral", "salient"):
        for arm in ARMS_PRIMARY:
            tr = res["traces"][cond + ("_novelty_lesion" if arm == "lesion_novelty" else "")]
            res["arms"].append(run_arm(seed, cond, arm, tr, *PRIMARY))
            print("[seed %d] primary %s/%s done %.0fs" % (seed, cond, arm, time.time() - t0), flush=True)
        for (beta, tau_e) in BAND:
            for arm in ARMS_BAND:
                res["arms"].append(run_arm(seed, cond, arm, res["traces"][cond], beta, tau_e))
        print("[seed %d] band %s done %.0fs" % (seed, cond, time.time() - t0), flush=True)
    res["gates"] = grade_seed(res)
    res["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(json.dumps(res["gates"], indent=2, default=str), flush=True)
    return res


def _arm(res, cond, arm, bp):
    for a in res["arms"]:
        if a["condition"] == cond and a["arm"] == arm and (a["beta"], a["tau_e"]) == tuple(bp):
            return a
    return None


def grade_seed(res):
    """The pre-registered per-seed gates (see the PREREG finding). UNDEFINED is never a pass."""
    from webapp.da_tag_capture import prp_threshold
    th = prp_threshold()
    n = len(FACTS)
    g = {"threshold": th}
    sal_fact_da = [r["da"] for r in res["traces"]["salient"] if r["fact"] is not None]
    neu_all_da = [r["da"] for r in res["traces"]["neutral"]]
    g["G1_salient_fact_da_min"] = min(sal_fact_da)
    g["G1_neutral_all_da_max"] = max(neu_all_da)
    g["G1_natural_drive"] = bool(min(sal_fact_da) >= th + 0.10 and max(neu_all_da) < th)
    g["G1_neutral_margin"] = th - max(neu_all_da)
    g["G1b_da_trace_deterministic"] = bool(res.get("da_trace_repeat_identical"))
    P = tuple(res["primary"])

    def dc(cond, arm, bp=P):
        a = _arm(res, cond, arm, bp)
        return None if a is None else a["delayed_counts"]["correct"]

    imm_ok = all(_arm(res, c, a, P)["immediate_counts"]["correct"] == n
                 for c in ("neutral", "salient") for a in ARMS_PRIMARY)
    g["G2_immediate_recall_intact_all_arms"] = bool(imm_ok)
    si, sl = dc("salient", "intact"), dc("salient", "lesion_da_encoding")
    g["salient_24h_intact"], g["salient_24h_lesion_da_encoding"] = si, sl
    g["G3_load_bearing"] = bool(si >= n - 1 and sl <= 1 and (si - sl) >= n - 1)
    g["G3_exact_perm_p"] = _exact_perm_p(si, sl, n)
    ni = dc("neutral", "intact")
    g["neutral_24h_intact"] = ni
    g["G4_specific_to_surprise"] = bool(ni <= 1 and (si - ni) >= n - 1)
    sc = dc("salient", "lesion_capture")
    g["salient_24h_lesion_capture"] = sc
    g["G5_capture_subedge"] = bool(sc <= 1)
    so, no = dc("salient", "companion_off"), dc("neutral", "companion_off")
    g["salient_24h_companion_off"], g["neutral_24h_companion_off"] = so, no
    g["G6_production_default_null"] = bool(so == n and no == n)
    g["null_diffs_intact_minus_lesion"] = {
        "neutral": ni - dc("neutral", "lesion_da_encoding"),
        "companion_off_salient_vs_neutral": so - no}
    # attribution: how much of the salient intact-vs-lesion reply change is absent from the same comparison in the
    # neutral (no-surprise) conversation? (None == UNDEFINED when the salient effect itself is ~0.)
    from tools.lab import attributable_to
    g["attributable_to_surprise_context"] = attributable_to(
        "DA-gate 24h recall effect: salient vs neutral context", si - sl, g["null_diffs_intact_minus_lesion"]["neutral"])
    confab = sum(a["delayed_counts"]["confab"] + a["immediate_counts"]["confab"] for a in res["arms"])
    g["confab_total_all_arms"] = confab
    band = {}
    for bp in BAND:
        imm = all(_arm(res, c, a, bp)["immediate_counts"]["correct"] == n
                  for c in ("neutral", "salient") for a in ARMS_BAND)
        bi, bl, bn = dc("salient", "intact", bp), dc("salient", "lesion_da_encoding", bp), dc("neutral", "intact", bp)
        band["beta=%s,tau_e=%s" % bp] = {"immediate_ok": imm, "salient_intact": bi, "salient_lesion": bl,
                                         "neutral_intact": bn,
                                         "holds": bool((not imm) or (bi >= n - 1 and bl <= 1 and bn <= 1))}
    g["G8_band"] = band
    g["G8_robust"] = all(v["holds"] for v in band.values())
    g["exploratory_lesion_novelty"] = {c: dc(c, "lesion_novelty") for c in ("neutral", "salient")}
    # the lesion must REACH its mechanism variables: the salient write gain and the PRP events
    si_arm, sl_arm = _arm(res, "salient", "intact", P), _arm(res, "salient", "lesion_da_encoding", P)
    g["salient_write_gain_intact_mean"] = float(np.mean(si_arm["write_gains"]))
    g["salient_write_gain_lesion_mean"] = float(np.mean(sl_arm["write_gains"]))
    g["salient_prp_events_intact"] = len(si_arm["prp_events_h"] or [])
    g["salient_prp_events_lesion"] = len(sl_arm["prp_events_h"] or [])
    from tools.verdict import Verdict
    v = Verdict("da-gated encoding under a natural drive, 24 h recall (seed %s)" % res["seed"])
    v.require("G1 natural DA contrast (salient facts >= th+0.10, every neutral turn < th)", g["G1_natural_drive"])
    v.require("G1b DA trace deterministic at this seed", g["G1b_da_trace_deterministic"])
    v.require("G2 immediate recall intact in every arm (DA gates persistence, not encoding)",
              g["G2_immediate_recall_intact_all_arms"])
    v.reaches("lesion reaches the write gain", before=g["salient_write_gain_intact_mean"],
              after=g["salient_write_gain_lesion_mean"])
    v.reaches("lesion reaches the PRP read", before=g["salient_prp_events_intact"],
              after=g["salient_prp_events_lesion"])
    v.disabled("recall-turn DA -> ledger", why="the recall questions' DA is not fed to the capture ledger "
               "(no tag is live at 24 h; at +1 min it would be a testing-effect confound)")
    go = all(g[k] for k in ("G3_load_bearing", "G4_specific_to_surprise", "G5_capture_subedge",
                            "G6_production_default_null", "G8_robust"))
    d = v.decide(go, verbose=False)
    g["verdict"] = d["status"]
    res["status"] = d["status"]
    res["preconditions"] = d["preconditions"]
    res["undefined_reasons"] = d["undefined_reasons"]
    res["disabled_processes"] = d["disabled_processes"]
    return g


def aggregate(d):
    rows = []
    for p in sorted(x for x in glob.glob(os.path.join(d, "seed*.json")) if not x.endswith(".prov.json")):
        r = json.load(open(p))
        rows.append({"seed": r["seed"], **{k: r["gates"][k] for k in (
            "verdict", "G1_salient_fact_da_min", "G1_neutral_all_da_max", "salient_24h_intact",
            "salient_24h_lesion_da_encoding", "neutral_24h_intact", "salient_24h_lesion_capture",
            "salient_24h_companion_off", "neutral_24h_companion_off", "G3_exact_perm_p", "confab_total_all_arms",
            "G8_robust")}})
    seeds = sorted(r["seed"] for r in rows)
    out = {"seeds": seeds, "rows": rows, "all_six_seeds_present": seeds == sorted(SEEDS),
           "n_go": sum(r["verdict"] == "GO" for r in rows)}
    # pooled null distribution: seed-wise random label permutations of the intact/lesion 24 h fact outcomes
    rng = np.random.default_rng(0)
    n = len(FACTS)
    obs = sum(r["salient_24h_intact"] - r["salient_24h_lesion_da_encoding"] for r in rows)
    null = []
    for _ in range(10000):
        s = 0
        for r in rows:
            pool = np.array([1] * (r["salient_24h_intact"] + r["salient_24h_lesion_da_encoding"])
                            + [0] * (2 * n - r["salient_24h_intact"] - r["salient_24h_lesion_da_encoding"]))
            rng.shuffle(pool)
            s += int(pool[:n].sum() - pool[n:].sum())
        null.append(s)
    null = np.array(null)
    out["pooled_effect_obs"] = int(obs)
    out["pooled_null_p"] = float((null >= obs).mean()) if rows else None
    out["pooled_null_q95"] = float(np.quantile(null, 0.95)) if rows else None
    from tools.verdict import Verdict
    v = Verdict("da-gated encoding natural drive, 6-seed aggregate")
    v.require("all six seeds present", out["all_six_seeds_present"])
    v.require("no seed UNDEFINED", all(r["verdict"] != "UNDEFINED" for r in rows) if rows else None)
    v.floor("pooled intact-minus-lesion effect vs the permutation-null 95th percentile",
            measured=out["pooled_effect_obs"], floor=out["pooled_null_q95"])
    d = v.decide(out["n_go"] == len(SEEDS), verbose=False)
    out["status"] = d["status"]
    out["preconditions"] = d["preconditions"]
    out["undefined_reasons"] = d["undefined_reasons"]
    return out


def selftest(seed=7):
    """Instrument checks that do not need the DA workspace (seed 7, not a gate seed):
      S1 exact pass-through: a ledger with beta=0 at t=t_write leaves the composer's own write BIT-EXACT.
      S2 the flag gate: maybe_ledger() is None with BRAIN_DA_TAG_CAPTURE unset/0 (production byte-identical).
      S3 the instrument can report FORGOTTEN: an uncaptured block at +24 h no longer recalls its fact.
      S4 the instrument can report REMEMBERED: a captured block at +24 h still recalls its fact.
      S5 lesions pin the PRP read to tonic."""
    from webapp import da_tag_capture as TC
    out = {}
    comp_a = _build_composer(seed)
    comp_b = _build_composer(seed)
    for f in FACTS:
        comp_a.store(*f["lem"])
    L0 = TC.TagCaptureLedger(seed, beta=0.0)
    for f in FACTS:
        comp_b.store(*f["lem"])
        L0.on_store(comp_b, 0.0)
    out["S1_bit_exact_passthrough"] = [complex(w) for (_p, _q, w) in comp_a.store_conns] == \
        [complex(w) for (_p, _q, w) in comp_b.store_conns] and \
        [(p, q) for (p, q, _w) in comp_a.store_conns] == [(p, q) for (p, q, _w) in comp_b.store_conns]
    with _env({"BRAIN_DA_TAG_CAPTURE": "0"}):
        out["S2_flag_off_no_ledger"] = TC.maybe_ledger(seed) is None
    comp_c = _build_composer(seed)
    L1 = TC.TagCaptureLedger(seed, beta=1.0, threshold=0.62)
    for i, f in enumerate(FACTS):
        if i < 2:
            L1.observe_da(0.0, 0.9)        # facts 0,1: a PRP event at their write -> captured
        comp_c.store(*f["lem"])
        L1.on_store(comp_c, 0.0 if i < 2 else 5.0)   # facts 2,3 written 5 h later, outside every PRP window
    L1.advance(comp_c, 5.0 + IMMEDIATE_H)
    out["S0_all_recalled_fresh"] = [r["outcome"] for r in _recall(comp_c)]
    L1.advance(comp_c, 5.0 + DELAY_H)
    rec = _recall(comp_c)
    out["S3_S4_outcomes_24h"] = [r["outcome"] for r in rec]
    out["S3_forgotten_detectable"] = all(r["outcome"] != "correct" for r in rec[2:])
    out["S4_remembered_detectable"] = all(r["outcome"] == "correct" for r in rec[:2])
    with _env({"BRAIN_DA_ENCODING_LESION": "1"}):
        a = TC.prp_da(0.9)
    with _env({"BRAIN_DA_ENCODING_LESION": "0", "BRAIN_DA_CAPTURE_LESION": "1"}):
        b = TC.prp_da(0.9)
    with _env({"BRAIN_DA_ENCODING_LESION": "0", "BRAIN_DA_CAPTURE_LESION": "0"}):
        c = TC.prp_da(0.9)
    out["S5_lesions_pin_tonic"] = bool(a == 0.5 and b == 0.5 and c == 0.9)
    out["pass"] = all(out[k] for k in ("S1_bit_exact_passthrough", "S2_flag_off_no_ledger",
                                       "S3_forgotten_detectable", "S4_remembered_detectable",
                                       "S5_lesions_pin_tonic")) and all(o == "correct"
                                                                         for o in out["S0_all_recalled_fresh"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=64, help="composer block dimension (v1=64; v2=128 production default)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot-da", action="store_true", help="print the DA traces only (stimulus pilot)")
    ap.add_argument("--aggregate", default=None)
    a = ap.parse_args()
    global COMPOSER_D
    COMPOSER_D = int(a.D)
    if a.selftest:
        r = selftest()
        print(json.dumps(r, indent=2, default=str))
        sys.exit(0 if r["pass"] else 1)
    if a.aggregate:
        r = aggregate(a.aggregate)
        p = os.path.join(a.aggregate, "aggregate.json")
        json.dump(r, open(p, "w"), indent=2, default=str)
        print(json.dumps(r, indent=2, default=str))
        return
    if a.pilot_da:
        pilot = {"seed": a.seed, "note": "stimulus/instrument PILOT on a non-gate seed; not a gate artifact",
                 "traces": {}, "smoke_salient_primary": []}
        for cond in ("neutral", "salient"):
            pilot["traces"][cond] = da_trace(a.seed, cond)
            for r in pilot["traces"][cond]:
                print(cond, "fact" if r["fact"] is not None else "    ", "%.3f" % r["da"], r["mode"], r["novelty"],
                      repr(r["text"]), flush=True)
        for arm in ("intact", "lesion_da_encoding", "lesion_capture", "companion_off"):
            x = run_arm(a.seed, "salient", arm, pilot["traces"]["salient"], *PRIMARY)
            pilot["smoke_salient_primary"].append({k: x[k] for k in ("arm", "write_gains", "prp_events_h",
                                                                     "immediate_counts", "delayed_counts")})
            print(arm, x["immediate_counts"], x["delayed_counts"], flush=True)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(pilot, open(a.out, "w"), indent=2, default=str)
        return
    out = a.out or os.path.join("research/findings/raw/_da_encoding_natural_drive", "seed%d.json" % a.seed)
    run_seed(a.seed, out)


if __name__ == "__main__":
    main()
