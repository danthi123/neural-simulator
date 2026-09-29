"""HOST-DECISION SHARE — the plan-step-2 baseline instrument (docs/plans/2026-09-25-prove-who-owns-the-computation-
PLAN.md sections 1-2). At the content and commitment seams of one live `/api/brain-chat` turn, who owns the decision
that reaches the reply: the spiking brain or host Python? The event buckets, per-seam classes and the panel are
defined ONCE in `research/runners/host_share_schema.py` (importable by the step-3 builder); this runner produces the
events.

SEAMS (verified in docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md; SEAM_SPEC below is the frozen table):
  k-select   CONTENT     which stored block answers (agent, action): host first-match over the spiking decodes
                         (`_fact_shard_first_match` on the default path, `_seq_block` on an escalation).
  v-choice   CONTENT     which of >=2 competing stored patients is committed: the host engagement scalar (recency +
                         WM referent) -> shared spiking salience transduction -> learned critic -> spiking value-WTA.
  bg-select  COMMITMENT  speak vs stay silent: host salience pair + the pre-filter `if not (silent > speak): return None`
                         (`bg_action_selection_production_organ.decide_action`), which skips the spiking BG race on every
                         content turn.
  organ-c    COMMITMENT  organ C's consensus vote for a known-word proposition: a host set-membership test.
  metacog    COMMITMENT  assert vs hedge: host arithmetic over matched-filter margins -> evidence -> the spiking monitor.
  ignition   COMMITMENT  the consensus read-out: `norgan_hop`'s host `argmax(rates)` + `rates[winner] >= THR`.

LEVELS (plan section 2), never merged: DECODING (screen), SUBSTITUTION (host rule -> neural-derived rule, does a
seam-downstream reply field change?), INTERVENTION (the primary causal measure: neural state at the seam held fixed,
the host decision set to its preregistered alternative on the target turn; does the reply follow the host?). Seams
whose only route from the host decision to the reply is host code or a pure function of the host value
(`downstream_neural_path=False`) are CALL-GRAPH CHECKS: reported, never pooled into the panel.

ARMS (one fresh brain per arm in its own process; the load-bearing-fraction discipline; numpy/CPU, stub renderer, no
LLM, LTM tier pinned ON): intact_a (production, no seam wrapper), intact_b (SHAM: every seam wrapper in observe mode),
<seam>:intervene and <seam>:substitute (the seam active on target turns only; every other seam observes).
Validity: V1 sham+determinism, V2 liveness, V3 intervention applied, V4 both outcomes registrable, V5 neural state at
the seam held fixed (each arm's pre-decision seam record equals intact_b's), V6 same brain (response `source`) in every
arm and seed, PYTHONHASHSEED=0 everywhere.

HARNESS-LEVEL DEPARTURES (declared in the prereg, applied identically in EVERY arm incl. intact_a):
  - BRAIN_CHAT_SEED is threaded into the value-choice critic, the BG selector and the shared salience organ, which
    production builds at a hardcoded seed 42 (so a six-seed run otherwise repeats one organ six times);
  - PYTHONHASHSEED=0 (value choice's `_stable_seed` uses Python `hash()`);
  - compete group only: BRAIN_RECONSOLIDATION=0 (else no second patient can be taught through chat).
NO sim/ or webapp/ edit: every wrapper is an in-process monkeypatch in the arm's worker subprocess.

Preregistration: research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md. Execution packet for the local
model: docs/drafts/host-share-baselines/HANDOFF.md. Dev seeds 1-12 only until the prereg is committed.

  .venv/bin/python -m research.runners.host_share --selftest
  bash tools/memcap.sh 9 -- .venv/bin/python -m research.runners.host_share --shard --seed 7 --group recall \
      --arm intact_b --out research/findings/raw/_host_share/dev/s7/recall__intact_b.json
  .venv/bin/python -m research.runners.host_share --aggregate research/findings/raw/_host_share/gate \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_host_share/gate/aggregate.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import host_share_schema as SCHEMA  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 13))
EXPECTED_SOURCE = "tiny-demo +LTM"
GATE_OUT_DIR = "research/findings/raw/_host_share/gate"
PREREG = "research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md"
HEDGE_MARK = "decision-margin reads this as low-confidence"

# reply-level DECISION fields (categorical; the load-bearing-fraction convention). The sham also compares `answer`.
DECISION_FIELDS = ("abstained", "recalled_svo", "supporting_facts", "n_sentences", "verified", "metacog.confident",
                   "hedge", "hold")
SHAM_FIELDS = ("answer",) + DECISION_FIELDS

# The frozen seam table. s_fields: the seam-DOWNSTREAM reply fields S is read on. upstream_fields: a difference there in
# the substitute arm is a side effect upstream of the seam -> S UNDEFINED. modes: the arms the seam gets.
SEAM_SPEC = {
    "k-select": dict(category="content", downstream_neural_path=True, modes=("intervene", "substitute"),
                     s_fields=DECISION_FIELDS, upstream_fields=(),
                     neural_alternative="spiking K-way sequencer (integrated_loop=True) for the target cue's calls"),
    "v-choice": dict(category="content", downstream_neural_path=True, modes=("intervene", "substitute"),
                     s_fields=DECISION_FIELDS, upstream_fields=(),
                     neural_alternative="raw engagement = each candidate block's spiking decode margin (margin_spiking)"
                                        ", min-max normalised; 0 if the block does not decode to that candidate"),
    "bg-select": dict(category="commitment", downstream_neural_path=True, modes=("intervene", "substitute"),
                      s_fields=DECISION_FIELDS, upstream_fields=(),
                      neural_alternative="the spiking BG race run on every target turn at equal salience (0.5, 0.5), "
                                         "no pre-filter"),
    "organ-c": dict(category="commitment", downstream_neural_path=False, modes=("intervene", "substitute"),
                    s_fields=DECISION_FIELDS, upstream_fields=(),
                    neural_alternative="the spiking D4 sel-pool margin read (the function's own unknown-word path)"),
    "metacog": dict(category="commitment", downstream_neural_path=False, modes=("intervene", "substitute"),
                    s_fields=("metacog.confident", "hedge", "n_sentences", "supporting_facts"),
                    upstream_fields=("abstained", "recalled_svo", "verified"),
                    neural_alternative="host band arithmetic on the recall circuit's SPIKING margin (margin_spiking) "
                                       "instead of the matched-filter margins"),
    "ignition": dict(category="commitment", downstream_neural_path=False, modes=("intervene",),
                     s_fields=DECISION_FIELDS, upstream_fields=(),
                     neural_alternative=None),
}
SEAMS = tuple(SEAM_SPEC)
MEASURED_SEAMS = tuple(s for s, d in SEAM_SPEC.items() if d["downstream_neural_path"])

# ── the probe turns: (label, message, target_cue or None, is_target). ONE session per group (one ChatBrain build per
#    arm; a numpy build + first turn ~25 min, a later turn ~5 min on dev seed 7). rich=True (the production default;
#    the battery worker coerces None to False). ─────────────────────────────────────────────────────────────────────
GROUPS = {
    "recall": {
        "env": {},
        "seams": ("k-select", "bg-select", "organ-c", "metacog", "ignition"),
        "sessions": [
            [("hsr_dog_q", "what does the dog chase", ("dog", "chase"), True),
             ("hsr_cat_q", "what does the cat eat", ("cat", "eat"), True),
             ("hsr_wolf_t", "the wolf bites the apple", None, False),
             ("hsr_wolf_q", "what does the wolf bite", ("wolf", "bite"), True)],
        ],
    },
    "compete": {
        "env": {"BRAIN_RECONSOLIDATION": "0"},
        "seams": ("k-select", "v-choice", "ignition"),
        "sessions": [
            [("hsc_dog_t", "the dog chase the ball", None, False),
             ("hsc_dog_q", "what does the dog chase", ("dog", "chase"), True),
             ("hsc_cat_t", "the cat eat the worm", None, False),
             ("hsc_cat_q", "what does the cat eat", ("cat", "eat"), True)],
        ],
    },
}

BASE_ENV = {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "BRAIN_LTM_SHIP_DEFAULT": "1"}


def group_labels(group, session_index=None):
    sessions = GROUPS[group]["sessions"]
    if session_index is not None:
        sessions = [sessions[int(session_index)]]
    return [t[0] for s in sessions for t in s]


def turn_specs(group):
    out = {}
    for si, sess in enumerate(GROUPS[group]["sessions"]):
        for ti, (label, msg, cue, target) in enumerate(sess):
            out[label] = {"message": msg, "session": "hs_%s_%d" % (group, si), "reset": ti == 0,
                          "cue": (list(cue) if cue else None), "target": bool(target)}
    return out


def arms_for(group):
    arms = ["intact_a", "intact_b"]
    for s in GROUPS[group]["seams"]:
        for m in SEAM_SPEC[s]["modes"]:
            arms.append("%s:%s" % (s, m))
    return arms


def _parse_arm(arm):
    if arm in ("intact_a", "intact_b"):
        return None, ("none" if arm == "intact_a" else "observe")
    seam, mode = arm.split(":")
    assert seam in SEAMS and mode in SEAM_SPEC[seam]["modes"], arm
    return seam, mode


# ══════════════════════════════════════════════ WORKER (inside the arm subprocess) ═════════════════════════════════
_ST = {"seam": None, "mode": "none", "label": None, "cue": None, "target": False, "combine_cue": None, "msg": None,
       "events": defaultdict(list)}


def _mode_for(seam):
    if _ST["seam"] == seam and _ST["target"]:
        return _ST["mode"]
    return "observe"


def _on_cue(agent, action):
    cue = _ST["cue"]
    return bool(_ST["target"] and cue and [agent, action] == list(cue))


def _rec(seam, **kw):
    kw["seam"] = seam
    _ST["events"][_ST["label"]].append(kw)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


def _r6(x):
    try:
        return None if x is None else round(float(x), 6)
    except Exception:
        return None


def install_seed_threading(seed):
    """EVERY arm (incl. intact_a): thread BRAIN_CHAT_SEED into the three organs production builds at seed 42."""
    from research.runners import value_choice_production_organ as VC
    from research.runners import bg_action_selection_production_organ as BG
    from research.runners import shared_salience_afferent as SH
    s = int(seed)
    _ivc, _da, _rs = VC.install_value_choice, BG.decide_action, SH.read_salience

    def install_value_choice(chat, *, organ=None, context_fn=None, seed=None):
        return _ivc(chat, organ=organ, context_fn=context_fn, seed=(s if seed is None else seed))

    def decide_action(msg, seed=None, lesion=None):
        return _da(msg, seed=(s if seed is None else seed), lesion=lesion)

    def read_salience(raw, *, seed=None):
        return _rs(raw, seed=(s if seed is None else seed))

    VC.install_value_choice, BG.decide_action, SH.read_salience = install_value_choice, decide_action, read_salience
    return {"value_choice_seed": s, "bg_select_seed": s, "shared_salience_seed": s}


def install_wrappers(seed):
    """The seam wrappers (observe unless the arm's seam is active on a target turn) + observation-only recorders."""
    from research.runners import one_brain_composer as OBC
    from research.runners import value_choice_production_organ as VC
    from research.runners import metacog_production_organ as MC
    from research.runners import bg_action_selection_production_organ as BG
    from research.runners import rich_answer_composer as RAC
    from research.runners import _gnw_norgan_bus_derisk as NB
    from webapp import gnw_three_organ_bus as TO

    # ── k-select, escalation path: `_seq_block` (host first-match while integrated_loop is off) ─────────────────
    _orig_seq = OBC.OneBrainComposer._seq_block

    def _seq_block(self, agent, action):
        if not _on_cue(agent, action) or getattr(self, "integrated_loop", False):
            return _orig_seq(self, agent, action)
        mode = _mode_for("k-select")
        if mode == "substitute":
            self.integrated_loop = True
            try:
                idx = _orig_seq(self, agent, action)
            finally:
                self.integrated_loop = False
            _rec("k-select", mode=mode, path="seq_block", applied=True, neural_idx=idx)
            return idx
        blocks = self._read_blocks()                     # the SAME single read the host loop makes
        matches = [i for i, g in enumerate(blocks) if g.get("agent") == agent and g.get("action") == action]
        host_idx = matches[0] if matches else None
        out, applied = host_idx, False
        if mode == "intervene" and matches:
            out, applied = (matches[-1] if len(matches) >= 2 else None), True
        _rec("k-select", mode=mode, path="seq_block", applied=applied, host_idx=host_idx, returned_idx=out,
             matches=matches, match_patients=[_jsonable(blocks[i].get("patient")) for i in matches])
        return out

    OBC.OneBrainComposer._seq_block = _seq_block

    # ── k-select, DEFAULT path: `_fact_shard_first_match` (the DG-CA3 fact shard is default-on) ──────────────────
    _orig_fsfm = OBC.OneBrainComposer._fact_shard_first_match

    def _fact_shard_first_match(self, cue_roles):
        cr = dict(cue_roles or {})
        if set(cr) != {"agent", "action"} or not _on_cue(cr["agent"], cr["action"]):
            return _orig_fsfm(self, cue_roles)
        mode = _mode_for("k-select")
        a, v = cr["agent"], cr["action"]
        kb = getattr(self, "kb", []) or []
        kb_matches = [i for i, (f, _h) in enumerate(kb) if f.get("agent") == a and f.get("action") == v]
        kb_patients = [_jsonable(kb[i][0].get("patient")) for i in kb_matches]
        if mode == "substitute":                         # escalate: the `_seq_block` wrapper runs the sequencer
            _rec("k-select", mode=mode, path="fact_shard", applied=False, escalated=True, matches=kb_matches,
                 match_patients=kb_patients)
            return OBC._FS_ESCALATE, None
        if mode != "intervene":
            idx, got = _orig_fsfm(self, cue_roles)     # observe: EXACTLY the production reads, nothing extra
            esc = idx is OBC._FS_ESCALATE
            _rec("k-select", mode=mode, path="fact_shard", applied=False, host_idx=(None if esc else idx),
                 escalated=esc, matches=kb_matches, match_patients=kb_patients)
            return idx, got
        shard = self._fact_shard_candidates(cue_roles)
        if shard is None:
            _rec("k-select", mode=mode, path="fact_shard", applied=False, escalated=True, matches=kb_matches,
                 match_patients=kb_patients)
            return OBC._FS_ESCALATE, None
        decoded = []
        for i in shard:                                  # the intervention reads the whole shard to find every match
            got = self._read_one_block(i)
            if all(got.get(r) == w for r, w in cue_roles.items()):
                decoded.append((i, got))
        host = decoded[0] if decoded else (None, None)
        if len(decoded) >= 2:
            out, applied = decoded[-1], True             # order flip: the LAST decoded match
        elif len(decoded) == 1:
            out, applied = (None, None), True            # match flip: the host declares no block matches
        else:
            out, applied = (None, None), False
        _rec("k-select", mode=mode, path="fact_shard", applied=applied, host_idx=host[0], returned_idx=out[0],
             decoded_matches=[i for i, _g in decoded], matches=kb_matches, match_patients=kb_patients)
        return out

    OBC.OneBrainComposer._fact_shard_first_match = _fact_shard_first_match

    # ── observation only: substrate recalls vs host shadow-list reads (the learned-content fraction) ────────────
    _orig_qp = OBC.OneBrainComposer.query_patient

    def query_patient(self, agent, action, order_fn=None):
        ans = _orig_qp(self, agent, action, order_fn=order_fn)
        if _ST["label"] is not None:
            _rec("_recall", agent=agent, action=action, patient=_jsonable(ans))
        return ans

    OBC.OneBrainComposer.query_patient = query_patient

    def _make_hostlist(orig, name):
        def host_list_read(self, concept):
            out = orig(self, concept)
            if _ST["label"] is not None:
                _rec("_hostlist", fn=name, concept=_jsonable(concept), facts=[_jsonable(list(f)) for f in (out or [])])
            return out
        return host_list_read
    for _name in ("_facts_about", "_facts_mentioning"):
        setattr(RAC.RichAnswerComposer, _name, _make_hostlist(getattr(RAC.RichAnswerComposer, _name), _name))

    # ── v-choice: the RAW host engagement scalar, manipulated upstream of the spiking salience transduction ──────
    _orig_dcf = VC.default_context_fn

    def _neural_engagement(chat, a, v, cands):
        """The neural-derived ranking: per candidate, the recall circuit's own SPIKING decode margin of its stored block
        (`margin_spiking`, patient role); 0 if that block does not decode to the candidate; min-max normalised
        (rank-preserving; ties -> 0.5)."""
        comp = getattr(getattr(chat, "inner", None), "composer", None)
        buf = getattr(comp, "buffer", comp)
        kb = getattr(buf, "kb", []) or []
        scores, decodable = [], []
        for p in cands:
            idxs = [i for i, (f, _h) in enumerate(kb)
                    if f.get("agent") == a and f.get("action") == v and f.get("patient") == p]
            best, ok = 0.0, False
            inner = getattr(buf, "comp", None)
            for i in idxs:
                saved = getattr(inner, "spiking_recall_margin", False)
                try:
                    if inner is not None:
                        inner.spiking_recall_margin = True
                    bs = buf._block_role_scores(i)
                finally:
                    if inner is not None:
                        inner.spiking_recall_margin = saved
                word, _conf, margin, margin_spk = bs.get("patient", (None, None, None, None))
                m = margin_spk if margin_spk is not None else margin
                if word == p and m is not None:
                    ok, best = True, max(best, float(m))
            scores.append(best)
            decodable.append(ok)
        lo, hi = min(scores), max(scores)
        return {"scores": scores, "decodable": decodable,
                "raw": [((x - lo) / (hi - lo)) if hi > lo else 0.5 for x in scores]}

    def default_context_fn(chat):
        base = _orig_dcf(chat)

        def ctx(a, v, cands):
            SH = VC._SHARED
            mode = _mode_for("v-choice") if _on_cue(a, v) else "observe"
            neural = _neural_engagement(chat, a, v, cands) if mode == "substitute" else None
            raws, k = [], [0]
            shared_on = bool(SH.shared_salience_enabled())
            orig_rs = SH.read_salience

            def read_salience(raw, **kw):
                r = float(raw)
                raws.append(r)
                if mode == "intervene":
                    r = 1.0 - r
                elif mode == "substitute" and k[0] < len(neural["raw"]):
                    r = float(neural["raw"][k[0]])
                k[0] += 1
                return orig_rs(r, **kw)
            if shared_on:
                SH.read_salience = read_salience
            try:
                used = [float(x) for x in base(a, v, cands)]
            finally:
                if shared_on:
                    SH.read_salience = orig_rs
            if not shared_on:                            # escape path: the raw scalar IS what the critic reads
                raws = list(used)
                if mode == "intervene":
                    used = [1.0 - x for x in used]
                elif mode == "substitute":
                    used = list(neural["raw"])
            _rec("v-choice", mode=mode, applied=(mode in ("intervene", "substitute")), cands=list(cands),
                 host_raw=[_r6(x) for x in raws], shared_salience=shared_on, used_engagement=[_r6(x) for x in used],
                 neural=neural)
            return used
        return ctx

    VC.default_context_fn = default_context_fn

    # ── bg-select: the host salience pair + the pre-filter that skips the spiking race ────────────────────────────
    _orig_sal = BG.salience
    _orig_decide = BG.decide_action                      # (already the seed-threading wrapper)

    def salience(msg):
        sp, si = _orig_sal(msg)
        mode = _mode_for("bg-select")
        out = (si, sp) if mode == "intervene" else (sp, si)   # intervene: swap the host's scoring
        if _ST["label"] is not None:
            _rec("bg-select", kind="salience", mode=mode, applied=(mode == "intervene"), host_speak=_r6(sp),
                 host_silent=_r6(si), used=[_r6(out[0]), _r6(out[1])], prefilter_host_runs_race=bool(si > sp))
        return out

    def decide_action(msg, seed=None, lesion=None):
        if _mode_for("bg-select") != "substitute":
            return _orig_decide(msg, seed=seed, lesion=lesion)
        BG.salience(msg)                                 # the one salience read production also makes (recorded)
        les = BG.bg_select_lesion() if lesion is None else lesion
        s = int(seed) if seed is not None else int(os.environ.get("BRAIN_CHAT_SEED", "42"))
        r = BG.get_organ(seed=s, lesion=les).select_once(0.5, 0.5)   # race decides: no host scoring, no pre-filter
        _rec("bg-select", kind="race", mode="substitute", applied=True, committed=bool(r.get("committed")),
             winner=_jsonable(r.get("winner")))
        if not r["committed"] or r["winner"] != BG.STAY_SILENT:
            return None
        return {"action": BG.ACTION_NAME[BG.STAY_SILENT], "winner": int(r["winner"]), "speak_salience": 0.5,
                "silent_salience": 0.5, "decision_step": r.get("decision_step"), "loser_ratio": r.get("loser_ratio"),
                "lesion": les}

    BG.salience, BG.decide_action = salience, decide_action

    # ── organ-c: organ C's vote (host set-membership for known propositions) ──────────────────────────────────────
    _orig_cv, _orig_rvc = TO._comprehension_vote, TO._real_vocab_competence

    def _comprehension_vote(agent, action, cand, brain_vocab, *, seed, lesion, ltm_exempt=False):
        mode = _mode_for("organ-c") if _on_cue(agent, action) else "observe"
        pre_known, _unk = _orig_rvc(agent, action, cand, brain_vocab)        # pure function: pre-decision state
        if mode == "substitute":
            TO._real_vocab_competence = lambda *a, **k: (False, ["__host_share_forced_d4__"])
            try:
                info = _orig_cv(agent, action, cand, brain_vocab, seed=seed, lesion=lesion, ltm_exempt=ltm_exempt)
            finally:
                TO._real_vocab_competence = _orig_rvc
            _rec("organ-c", mode=mode, applied=True, cand=_jsonable(cand), pre_known=bool(pre_known),
                 neural_votes=bool(info.get("votes")), d4_margin=_r6(info.get("organ_c_margin")),
                 d4_competent=info.get("organ_c_competent"))
            return info
        info = _orig_cv(agent, action, cand, brain_vocab, seed=seed, lesion=lesion, ltm_exempt=ltm_exempt)
        host_votes = bool(info.get("votes"))
        applied = False
        if mode == "intervene":
            info = dict(info)
            info["votes"] = not host_votes
            info["organ_c_comprehended"] = not host_votes
            applied = True
        if _on_cue(agent, action):
            _rec("organ-c", mode=mode, applied=applied, cand=_jsonable(cand), pre_known=bool(pre_known),
                 host_votes=host_votes, returned_votes=bool(info.get("votes")))
        return info

    TO._comprehension_vote = _comprehension_vote

    # ── ignition: the consensus read-out (host argmax over slot rates + threshold) ────────────────────────────────
    _orig_combine, _orig_hop = TO.three_organ_combine, TO.norgan_hop

    def three_organ_combine(chat, agent, action, **kw):
        _ST["combine_cue"] = (agent, action)
        try:
            return _orig_combine(chat, agent, action, **kw)
        finally:
            _ST["combine_cue"] = None

    def norgan_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub, **kw):
        committed, rates, winner, n_ign = _orig_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub, **kw)
        cc = _ST["combine_cue"]
        if not (cc and _on_cue(cc[0], cc[1])):
            return committed, rates, winner, n_ign
        mode = _mode_for("ignition")
        out = committed
        if mode == "intervene":
            out = None if committed is not None else (candidates[0] if candidates and candidates[0] else None)
        _rec("ignition", mode=mode, applied=(mode == "intervene" and out != committed),
             candidates=[_jsonable(c) for c in candidates], decoy=_jsonable(decoy),
             rates=[_r6(x) for x in (rates or [])], winner=_jsonable(winner), thr=_r6(NB.THR),
             host_committed=_jsonable(committed), returned=_jsonable(out))
        return out, rates, winner, n_ign

    TO.three_organ_combine, TO.norgan_hop = three_organ_combine, norgan_hop

    # ── metacog: host arithmetic -> evidence (the spiking margin for the substitution) ────────────────────────────
    _orig_brs = OBC.OneBrainComposer._block_role_scores

    def _block_role_scores(self, block_idx):
        if _mode_for("metacog") == "substitute" and hasattr(self, "comp"):
            saved = getattr(self.comp, "spiking_recall_margin", False)
            self.comp.spiking_recall_margin = True
            try:
                return _orig_brs(self, block_idx)
            finally:
                self.comp.spiking_recall_margin = saved
        return _orig_brs(self, block_idx)

    OBC.OneBrainComposer._block_role_scores = _block_role_scores
    _orig_mrc, _orig_ev = MC.mean_role_confidence, MC.evidence_from_role_conf

    def mean_role_confidence(activity):
        v = _orig_mrc(activity)
        if _ST["label"] is not None:
            roles = (activity or {}).get("roles") or []
            _rec("metacog", kind="mrc", mode=_mode_for("metacog"), mrc=_r6(v),
                 spiking_present=any(r.get("margin_spiking") is not None for r in roles),
                 role_margins=[_r6(r.get("margin")) for r in roles],
                 role_margins_spiking=[_r6(r.get("margin_spiking")) for r in roles])
        return v

    def evidence_from_role_conf(mrc):
        ev = _orig_ev(mrc)
        mode = _mode_for("metacog")
        out, applied = ev, False
        if mode == "intervene" and ev is not None:
            out, applied = float(1.0 - ev), True
        if _ST["label"] is not None:
            _rec("metacog", kind="evidence", mode=mode, applied=applied, host_evidence=_r6(ev),
                 returned_evidence=_r6(out))
        return out

    MC.mean_role_confidence, MC.evidence_from_role_conf = mean_role_confidence, evidence_from_role_conf


_LAST_VC = {}


def _chat_snapshot(req):
    """Post-turn read of the ChatBrain's per-turn trace attributes (observation only; mutates nothing)."""
    try:
        import webapp.server as S
        chat = S._BRAIN_CHATS.get((req.session, req.brain, (req.renderer or "stub").lower()))
    except Exception:
        chat = None
    if chat is None:
        return {}
    out = {}
    t = getattr(chat, "_last_three_organ", None)
    if isinstance(t, dict):
        out["three_organ"] = {k: _jsonable(t.get(k)) for k in (
            "authored_by", "committed", "organ_a_recall", "organ_b_confirmed", "organ_c_votes",
            "organ_c_real_vocab_known", "abstain_reason", "n_votes", "recall_source")}
    d = getattr(chat, "_last_gnw_delib", None)
    if isinstance(d, dict):
        out["deliberation"] = {k: _jsonable(d.get(k)) for k in ("acted", "reason", "decision", "n_candidates")}
    vc = getattr(chat, "_value_choice_last", None)
    stale = isinstance(vc, dict) and _LAST_VC.get(req.session) is vc
    if isinstance(vc, dict):
        _LAST_VC[req.session] = vc
    if isinstance(vc, dict) and not stale:
        meta = vc.get("meta") or {}
        out["value_choice"] = {"candidates": _jsonable(vc.get("candidates")), "chosen": _jsonable(vc.get("chosen")),
                               "inner_result": _jsonable(vc.get("inner_result")),
                               "meta": {k: _jsonable(v) for k, v in meta.items()
                                        if isinstance(v, (int, float, str, bool, list, type(None)))}}
    return out


def run_worker(seed, group, arm, out_path, backend="numpy", session_index=None, only_labels=None):
    seam, mode = _parse_arm(arm)
    env = dict(BASE_ENV)
    if backend != "numpy":                                # dev-only replication on the GPU backend
        env["SIM_BACKEND"] = backend
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(GROUPS[group]["env"])
    env["BRAIN_CHAT_SEED"] = str(int(seed))
    for k, v in env.items():                              # BEFORE any webapp import: organs read env at import time
        os.environ[k] = v
    hash_ok = (os.environ.get("PYTHONHASHSEED") == "0")

    import research.runners.onebrain_regression_battery as B
    specs = turn_specs(group)
    for label, sp in specs.items():                       # label-only turns, in-process (the battery file is untouched)
        B._TURN_BY_LABEL[label] = (label, sp["message"], sp["session"], sp["reset"], None, True)
    by_msg = {(sp["session"], sp["message"]): label for label, sp in specs.items()}

    import webapp.server as S
    seeds_used = install_seed_threading(seed)             # EVERY arm, incl. intact_a
    _ST.update({"seam": seam, "mode": mode})
    if mode != "none":
        install_wrappers(seed)
    _orig_bc = S.brain_chat
    snapshots, timings = {}, {}

    def brain_chat(req):
        label = by_msg.get((req.session, req.message))
        sp = specs.get(label, {})
        _ST.update({"label": label, "cue": sp.get("cue"), "target": bool(sp.get("target")), "msg": req.message})
        t0 = time.time()
        try:
            return _orig_bc(req)
        finally:
            timings[label] = round(time.time() - t0, 2)
            snapshots[label] = _chat_snapshot(req)
            _ST.update({"label": None, "cue": None, "target": False, "msg": None})

    S.brain_chat = brain_chat
    labels = group_labels(group, session_index)
    if only_labels:                                       # DEV aid only (refused on gate seeds)
        labels = [lab for lab in labels if lab in set(only_labels)]
    resp_path = out_path + ".responses.json"
    t0 = time.time()
    B._collect_worker(json.dumps(env), labels, resp_path)
    responses = json.load(open(resp_path))
    os.remove(resp_path)
    result = {"instrument": "host_share", "seed": int(seed), "group": group, "arm": arm, "seam": seam, "mode": mode,
              "backend": env["SIM_BACKEND"], "session_index": session_index, "env": env,
              "pythonhashseed_ok": hash_ok, "seeds_used": seeds_used, "turns": specs, "responses": responses,
              "events": {k: v for k, v in _ST["events"].items()}, "snapshots": snapshots, "turn_seconds": timings,
              "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print("[host_share worker] seed=%s group=%s arm=%s -> %s (%.0fs)" % (seed, group, arm, out_path,
                                                                        result["wall_seconds"]), flush=True)
    return 0


def run_shard(seed, group, arm, out_path, backend="numpy", only_labels=None):
    """One (seed, group, arm) shard; each session in its OWN fresh subprocess (PYTHONHASHSEED=0 set before interpreter
    start): one ChatBrain per process, no carry-over through process-shared organ singletons. Parts are merged."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    parts = []
    for si in range(len(GROUPS[group]["sessions"])):
        part = "%s.part%d" % (out_path, si)
        if not os.path.exists(part):
            cmd = [sys.executable, "-u", "-m", "research.runners.host_share", "--worker", "--seed", str(seed),
                   "--group", group, "--arm", arm, "--out", part, "--backend", backend, "--session-index", str(si)]
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
    merged["wall_seconds"] = 0.0
    for pt in parts:
        for k in ("responses", "events", "snapshots", "turn_seconds"):
            merged[k].update(pt.get(k) or {})
        merged["wall_seconds"] += float(pt.get("wall_seconds") or 0.0)
        merged["pythonhashseed_ok"] = bool(merged["pythonhashseed_ok"] and pt.get("pythonhashseed_ok"))
    merged["n_session_processes"] = len(parts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=1, default=str)
    for si in range(len(parts)):
        for suffix in ("", ".prov.json"):               # the part and the sidecar its worker process stamped
            fp = "%s.part%d%s" % (out_path, si, suffix)
            if os.path.exists(fp):
                os.remove(fp)
    return True


def shard_path(out_dir, seed, group, arm):
    return os.path.join(out_dir, "s%d" % int(seed), "%s__%s.json" % (group, arm.replace(":", "_")))


def pool_job_lines(sha, seeds=GATE_SEEDS, out_dir=GATE_OUT_DIR):
    """One pool job per (seed, group, arm): (command, checked-reason). ~6.3 GB RSS per build -> mem_gb=7, cap 9 GB."""
    out = []
    for seed in seeds:
        for group in GROUPS:
            for arm in arms_for(group):
                p = shard_path(out_dir, seed, group, arm)
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 7 1 && mkdir -p %s && "
                       "bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                       "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                       "research.runners.host_share --shard --seed %d --group %s --arm %s --allow-gate-seed "
                       "--out %s" % (sha, os.path.dirname(p), seed, group, arm, p))
                why = ("prereg %s (%s); host-share baseline gate (plan step 2) seed %d %s %s; record: first "
                       "host-share measurement, before_you_build + RAG 2026-09-29 found none; mem_gb=7"
                       % (PREREG, sha[:9], seed, group, arm))
                out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ AGGREGATION (no brain build) ═══════════════════════════════════════
_STOP = set("the a an of to and or but is are was were be been it its this that i you we they he she my your me "
            "so as at in on for with by from do does did can could might may will would not no yes than then there "
            "here what who which how why".split())


def _get(d, path):
    if path == "hedge":
        return HEDGE_MARK in str((d or {}).get("answer") or "")
    if path == "hold":
        return bool(((d or {}).get("bg_select") or {}).get("on"))
    cur = d
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def reply_diff(a, b, fields=DECISION_FIELDS):
    return [f for f in fields if _get(a or {}, f) != _get(b or {}, f)]


def sham_diff(a, b):
    return reply_diff(a, b, SHAM_FIELDS)


def committed_patient(resp):
    svo = (resp or {}).get("recalled_svo")
    return svo[2] if isinstance(svo, list) and len(svo) == 3 else None


def seam_outcome(seam, resp):
    """The reply's outcome AT THIS SEAM (what F and the neural preference are compared on)."""
    if resp is None:
        return None
    if seam == "metacog":
        c = _get(resp, "metacog.confident")
        return None if c is None else bool(c)
    if seam == "bg-select":
        return "__hold__" if _get(resp, "hold") else "__speak__"
    return "__abstain__" if resp.get("abstained") else committed_patient(resp)


def _seam_events(shard, label, seam, kind=None):
    evs = [e for e in ((shard or {}).get("events") or {}).get(label, []) if e.get("seam") == seam]
    return [e for e in evs if kind is None or e.get("kind") == kind]


def _pre_state(seam, evs):
    """The pre-decision state recorded at the seam (V5 compares it with intact_b's)."""
    if not evs:
        return None
    if seam == "k-select":
        e = next((x for x in evs if x.get("matches") is not None), None)
        return None if e is None else tuple(e.get("matches") or [])
    if seam == "v-choice":
        return (tuple(evs[0].get("cands") or []), tuple(evs[0].get("host_raw") or []))
    if seam == "bg-select":
        e = next((x for x in evs if x.get("kind") == "salience"), None)
        return None if e is None else (e.get("host_speak"), e.get("host_silent"))
    if seam == "organ-c":
        return (evs[0].get("cand"), evs[0].get("pre_known"))
    if seam == "metacog":
        e = next((x for x in evs if x.get("kind") == "mrc"), None)
        return None if e is None else tuple(e.get("role_margins") or [])
    if seam == "ignition":
        return (tuple(evs[0].get("candidates") or []), tuple(evs[0].get("rates") or []))
    return None


def _host_alt_and_live(seam, obs, ra):
    """(live, host_alt, reason_if_no_alt, extra) from intact_b's observe events + the intact reply."""
    extra = {}
    if seam == "k-select":
        o = [e for e in obs if e.get("mode") == "observe" and e.get("matches")]
        if not o:
            return False, None, "not live: no target-cue block selection with >=1 match", extra
        e0 = o[0]
        extra.update(sub_kind=("order" if len(e0["matches"]) >= 2 else "match"), host_idx=e0.get("host_idx"),
                     match_patients=e0.get("match_patients"))
        return True, (e0["match_patients"][-1] if extra["sub_kind"] == "order" else "__abstain__"), None, extra
    if seam == "v-choice":
        o = [e for e in obs if e.get("mode") == "observe" and len(e.get("cands") or []) >= 2]
        if not o:
            return False, None, "not live: value choice not consulted with >=2 candidates", extra
        rev = [1.0 - x for x in (o[0].get("host_raw") or [])]
        if not rev or sum(1 for x in rev if x == max(rev)) != 1:
            return True, None, "no host alternative: reversed engagement has no unique top", extra
        return True, o[0]["cands"][rev.index(max(rev))], None, extra
    if seam == "bg-select":
        o = [e for e in obs if e.get("kind") == "salience" and e.get("mode") == "observe"]
        if not o:
            return False, None, "not live: no salience read on the target turn", extra
        extra["prefilter_host_runs_race"] = o[0].get("prefilter_host_runs_race")
        return True, ("__hold__" if seam_outcome(seam, ra) == "__speak__" else "__speak__"), None, extra
    if seam == "organ-c":
        o = [e for e in obs if e.get("mode") == "observe"]
        if not o:
            return False, None, "not live: organ C not consulted on the target cue", extra
        extra.update(host_votes=o[0].get("host_votes"), real_vocab_known=o[0].get("pre_known"),
                     sub_kind=("known" if o[0].get("pre_known") else "unknown"))
        return True, ("__abstain__" if o[0].get("host_votes") else o[0].get("cand")), None, extra
    if seam == "metacog":
        o = [e for e in obs if e.get("kind") == "evidence" and e.get("host_evidence") is not None]
        if not o:
            return False, None, "not live: no metacog evidence read on the target turn", extra
        extra["host_evidence"] = o[0]["host_evidence"]
        conf = _get(ra, "metacog.confident")
        if conf is None:
            return True, None, "no host alternative: intact reply carries no metacog verdict", extra
        if abs(float(o[0]["host_evidence"]) - 0.5) < 1e-9:
            return True, None, "no host alternative: mirrored evidence equals the host evidence (0.5)", extra
        return True, (not bool(conf)), None, extra
    if seam == "ignition":
        o = [e for e in obs if e.get("mode") == "observe"]
        if not o:
            return False, None, "not live: no consensus read-out on the target cue", extra
        e0 = o[0]
        extra.update(host_committed=e0.get("host_committed"), rates=e0.get("rates"), thr=e0.get("thr"))
        if e0.get("host_committed") is not None:
            return True, "__abstain__", None, extra
        c0 = (e0.get("candidates") or [None])[0]
        if c0 is None:
            return True, None, "no host alternative: no organ-A candidate to commit", extra
        return True, c0, None, extra
    return False, None, "unknown seam", extra


def _neural_pref(seam, sub_evs, rs, sub_snapshot):
    """The NEURAL-DERIVED decision's outcome at this seam (from the substitute arm), or None if it made none."""
    if seam == "k-select":
        n = [e for e in sub_evs if e.get("applied") and "neural_idx" in e]
        m = [e for e in sub_evs if e.get("matches") is not None]
        if not n:
            return None
        ni = n[0]["neural_idx"]
        if ni is None:
            return "__abstain__"
        if m and ni in (m[0].get("matches") or []):
            return m[0]["match_patients"][m[0]["matches"].index(ni)]
        return None
    if seam == "v-choice":
        return _get(sub_snapshot or {}, "value_choice.chosen")        # None = the organ declined (no decision)
    if seam == "bg-select":
        r = [e for e in sub_evs if e.get("kind") == "race"]
        if not r or not r[0].get("committed"):
            return None
        return "__hold__" if r[0].get("winner") == 1 else "__speak__"
    if seam == "organ-c":
        n = [e for e in sub_evs if e.get("applied")]
        return None if not n else (n[0].get("cand") if n[0].get("neural_votes") else "__abstain__")
    if seam == "metacog":
        return seam_outcome("metacog", rs)
    return None


def _host_decision(seam, extra, ib_snapshot, intact_out):
    """The host's own decision at the seam, in the same currency as `_neural_pref` (for DECODING)."""
    if seam == "k-select":
        mp = extra.get("match_patients") or []
        return mp[0] if (extra.get("host_idx") is not None and mp) else "__abstain__"
    if seam == "v-choice":
        return _get(ib_snapshot or {}, "value_choice.chosen")
    if seam == "bg-select":
        return None if extra.get("prefilter_host_runs_race") else "__speak__"
    if seam == "organ-c":
        return "__vote__" if extra.get("host_votes") else "__abstain__"
    if seam == "metacog":
        return intact_out
    return None


def seam_event(seam, label, ia, ib, intervene, substitute):
    """One (seam, target turn) event for one seed: live / F / f_outcome / S / D, each possibly None = UNDEFINED with
    its reason recorded. Buckets and classes come from host_share_schema."""
    spec = SEAM_SPEC[seam]
    ev = {"seam": seam, "turn": label, "live": False, "F": None, "f_outcome": None, "S": None, "D": None,
          "undefined": [], "downstream_neural_path": spec["downstream_neural_path"]}
    ra = ((ia or {}).get("responses") or {}).get(label)
    rb = ((ib or {}).get("responses") or {}).get(label)
    if ra is None or rb is None or "_error" in ra or "_error" in rb:
        ev["undefined"].append("intact arm missing/errored")
        return ev
    if sham_diff(ra, rb):
        ev["undefined"].append("V1 sham/determinism failed: intact_a != intact_b on %s" % sham_diff(ra, rb))
        return ev
    obs = _seam_events(ib, label, seam)
    live, host_alt, why, extra = _host_alt_and_live(seam, obs, ra)
    ev.update(extra)
    ev["live"], ev["host_alt"] = live, host_alt
    if not live:
        ev["undefined"].append(why)
        return ev
    if why:
        ev["undefined"].append(why)
    intact_out = seam_outcome(seam, ra)
    ev["intact_outcome"] = intact_out
    pre_b = _pre_state(seam, obs)
    ib_snap = ((ib.get("snapshots") or {}).get(label) or {})

    # ── SUBSTITUTION + DECODING (and the neural preference used to label F=0 events) ──────────────────────────────
    neural_pref = None
    rs = ((substitute or {}).get("responses") or {}).get(label) if substitute else None
    if spec["neural_alternative"] is None:
        ev["undefined"].append("S: no neural-derived alternative exists for this seam (by design)")
    elif rs is None or "_error" in (rs or {}):
        ev["undefined"].append("S: substitute arm missing/errored")
    else:
        sub_evs = _seam_events(substitute, label, seam)
        if seam == "metacog":
            applied = any(e.get("spiking_present") for e in sub_evs if e.get("kind") == "mrc")
        elif seam == "k-select":
            applied = any(e.get("applied") and "neural_idx" in e for e in sub_evs)
        else:
            applied = any(e.get("applied") for e in sub_evs)
        pre_s = _pre_state(seam, sub_evs)
        if not applied:
            ev["undefined"].append("S: substitution not applied on the target turn")
        elif seam != "metacog" and pre_b is not None and pre_s is not None and pre_s != pre_b:
            ev["undefined"].append("V5 seam state drifted in the substitute arm")
        elif reply_diff(ra, rs, spec["upstream_fields"]):
            ev["undefined"].append("S: substitution side effect upstream of the seam: %s"
                                   % reply_diff(ra, rs, spec["upstream_fields"]))
        else:
            ev["S"] = 1 if reply_diff(ra, rs, spec["s_fields"]) else 0
            ev["substitute_diff"] = reply_diff(ra, rs, spec["s_fields"])
            sub_snap = ((substitute.get("snapshots") or {}).get(label) or {})
            neural_pref = _neural_pref(seam, sub_evs, rs, sub_snap)
            ev["neural_pref"] = neural_pref
            if seam == "organ-c":
                n = [e for e in sub_evs if e.get("applied")]
                ev["D"] = int(bool(n[0].get("neural_votes")) == bool(extra.get("host_votes"))) if n else None
                ev["d4_margin"] = n[0].get("d4_margin") if n else None
            else:
                hd = _host_decision(seam, extra, ib_snap, intact_out)
                ev["D"] = None if (neural_pref is None or hd is None) else int(neural_pref == hd)
            if seam == "v-choice":
                nv = [e for e in sub_evs if e.get("neural")]
                if nv:
                    ev["candidates_decodable"] = nv[0]["neural"].get("decodable")
    if seam == "ignition":
        ev["D"] = 1
        ev["D_note"] = "trivial by construction: the host read-out IS the argmax of the neural rates"

    # ── INTERVENTION ─────────────────────────────────────────────────────────────────────────────────────────────
    rv = ((intervene or {}).get("responses") or {}).get(label) if intervene else None
    if rv is None or "_error" in (rv or {}):
        ev["undefined"].append("F: intervene arm missing/errored")
        return ev
    iv_evs = _seam_events(intervene, label, seam)
    if not any(e.get("applied") for e in iv_evs):
        ev["undefined"].append("V3 intervention not applied on the target turn")
        return ev
    pre_i = _pre_state(seam, iv_evs)
    if pre_b is not None and pre_i is not None and pre_i != pre_b:
        ev["undefined"].append("V5 seam state drifted in the intervene arm")
        return ev
    if host_alt is None:
        return ev
    if host_alt == intact_out:
        ev["undefined"].append("F: the host alternative coincides with the intact outcome (not observable)")
        return ev
    out_v = seam_outcome(seam, rv)
    ev["intervene_outcome_value"] = out_v
    ev["intervene_diff"] = reply_diff(ra, rv)
    if out_v == host_alt:
        ev["F"], ev["f_outcome"] = 1, "follows_host"
    else:
        ev["F"] = 0
        if _get((intervene.get("snapshots") or {}).get(label) or {}, "three_organ.recall_source") == "ltm":
            ev["f_outcome"] = "ltm_fallback"
        elif neural_pref is not None and out_v == neural_pref and neural_pref != host_alt:
            ev["f_outcome"] = "returned_to_neural"
        else:
            ev["f_outcome"] = "masked"
    if seam == "k-select":                               # descriptive: did the CONSENSUS accept the host's pick?
        ev["consensus_intact"] = _get(ib_snap, "three_organ.committed")
        ev["consensus_intervene"] = _get((intervene.get("snapshots") or {}).get(label) or {}, "three_organ.committed")
    return ev


def content_fractions(resp, events, vc_snapshot=None):
    """Per asserted fact, WHO SELECTED IT: 'substrate' (a spiking recall `query_patient` returned that exact triple this
    turn), 'host_list' (it came from the host's shadow fact list: `_facts_about` / `_facts_mentioning`, or value
    choice's host candidate list), else 'unattributed'. learned_content_fraction = substrate share;
    python_selected_fraction = host_list share. fact_word_share = the lexical share of the reply's content words that
    are words of the asserted facts (vs template words): a wording measure, reported separately."""
    facts = [tuple(f) for f in ((resp or {}).get("supporting_facts") or []) if isinstance(f, list) and len(f) == 3]
    if not facts and isinstance((resp or {}).get("recalled_svo"), list):
        facts = [tuple(resp["recalled_svo"])]
    if not facts or (resp or {}).get("abstained"):
        return None
    recalls = {(e.get("agent"), e.get("action"), e.get("patient")) for e in (events or []) if e.get("seam") == "_recall"}
    hostl = {tuple(f) for e in (events or []) if e.get("seam") == "_hostlist" for f in (e.get("facts") or [])}
    chosen = (vc_snapshot or {}).get("chosen")
    src = []
    for f in facts:
        if f in recalls:
            src.append("substrate")
        elif f in hostl or (chosen is not None and f[2] == chosen):
            src.append("host_list")
        else:
            src.append("unattributed")
    n = float(len(facts))
    fact_words = {str(w).lower() for f in facts for w in f for w in str(w).split()}
    words = [t for t in re.findall(r"[a-z]+", str(resp.get("answer") or "").lower()) if t not in _STOP]

    def _is_fact_word(t):
        return any(t == w or (len(w) >= 3 and t.startswith(w)) for w in fact_words)
    return {"n_facts": len(facts), "sources": src,
            "learned_content_fraction": src.count("substrate") / n,
            "python_selected_fraction": src.count("host_list") / n,
            "unattributed_fraction": src.count("unattributed") / n,
            "fact_word_share": (sum(1 for t in words if _is_fact_word(t)) / float(len(words))) if words else None}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (float(sum(xs)) / len(xs)) if xs else None


def aggregate(shard_dir, seeds, min_live_seeds=SCHEMA.MIN_LIVE_SEEDS_GATE):
    seeds = [int(s) for s in seeds]
    all_events, content_rows = [], []
    validity = {"sham_clean": {}, "hash_ok": {}, "missing_shards": [], "sources": defaultdict(set),
                "seeds_used_ok": {}}
    for seed in seeds:
        for group in GROUPS:
            sh = {}
            for arm in arms_for(group):
                p = shard_path(shard_dir, seed, group, arm)
                if os.path.exists(p):
                    try:
                        sh[arm] = json.load(open(p))
                    except Exception:
                        validity["missing_shards"].append(p + " (unreadable)")
                else:
                    validity["missing_shards"].append(p)
            for arm, d in sh.items():
                key = "s%d/%s/%s" % (seed, group, arm)
                validity["hash_ok"][key] = bool(d.get("pythonhashseed_ok"))
                su = d.get("seeds_used") or {}
                validity["seeds_used_ok"][key] = bool(su) and all(int(v) == seed for v in su.values())
                for r in (d.get("responses") or {}).values():
                    validity["sources"][str((r or {}).get("source"))].add(key)
            ia, ib = sh.get("intact_a"), sh.get("intact_b")
            specs = turn_specs(group)
            if ia and ib:
                validity["sham_clean"]["s%d/%s" % (seed, group)] = all(
                    not sham_diff(ia["responses"].get(lab), ib["responses"].get(lab)) for lab in specs)
                for lab, sp in specs.items():
                    if sp["target"]:
                        cf = content_fractions(ib["responses"].get(lab), (ib.get("events") or {}).get(lab),
                                               ((ib.get("snapshots") or {}).get(lab) or {}).get("value_choice"))
                        if cf:
                            cf.update({"seed": seed, "group": group, "turn": lab})
                            content_rows.append(cf)
            for seam in GROUPS[group]["seams"]:
                for lab, sp in specs.items():
                    if sp["target"]:
                        e = seam_event(seam, lab, ia, ib, sh.get("%s:intervene" % seam), sh.get("%s:substitute" % seam))
                        e.update({"seed": seed, "group": group})
                        e["bucket"] = SCHEMA.event_bucket(e["F"], e["S"], e["f_outcome"])
                        all_events.append(e)
    seams = {}
    for seam in SEAMS:
        evs = [e for e in all_events if e["seam"] == seam]
        st = SCHEMA.seam_stats(evs)
        sub = {sk: SCHEMA.seam_stats([e for e in evs if e.get("sub_kind") == sk])
               for sk in sorted({e.get("sub_kind") for e in evs if e.get("sub_kind")})}
        outc = defaultdict(int)
        for e in evs:
            if e.get("f_outcome"):
                outc[e["f_outcome"]] += 1
        seams[seam] = {"category": SEAM_SPEC[seam]["category"],
                       "downstream_neural_path": SEAM_SPEC[seam]["downstream_neural_path"],
                       "reported_as": "MEASURED" if SEAM_SPEC[seam]["downstream_neural_path"] else "CALL-GRAPH CHECK",
                       "class": SCHEMA.classify_seam(st, min_live_seeds=min_live_seeds), "stats": st,
                       "sub_kinds": sub, "f_outcomes": dict(outc),
                       "undefined_reasons": sorted({u for e in evs for u in e["undefined"]})}
    sham_ok_seeds = sorted(s for s in seeds
                           if any(k.startswith("s%d/" % s) for k in validity["sham_clean"])
                           and all(v for k, v in validity["sham_clean"].items() if k.startswith("s%d/" % s)))
    sources = sorted(validity["sources"])
    v = {
        "V1_sham_clean_seeds": sham_ok_seeds, "V1_pass": len(sham_ok_seeds) >= max(1, len(seeds) - 1),
        "V2_live_seeds_per_seam": {s: len(seams[s]["stats"]["live_seeds"]) for s in SEAMS},
        "V3_interventions_applied": not any("V3 intervention not applied on the target turn" in e["undefined"]
                                            for e in all_events if e["live"]),
        "V4_both_outcomes_registrable": any(e["F"] == 1 for e in all_events) and bool(sham_ok_seeds),
        "V5_state_drift_events": sum(1 for e in all_events if any(u.startswith("V5") for u in e["undefined"])),
        "V6_sources": sources, "V6_pass": sources == [EXPECTED_SOURCE],
        "pythonhashseed_all_ok": bool(validity["hash_ok"]) and all(validity["hash_ok"].values()),
        "seed_threading_all_ok": bool(validity["seeds_used_ok"]) and all(validity["seeds_used_ok"].values()),
        "missing_shards": validity["missing_shards"],
    }
    ok = (v["V1_pass"] and v["V3_interventions_applied"] and v["V4_both_outcomes_registrable"] and v["V6_pass"]
          and v["pythonhashseed_all_ok"] and v["seed_threading_all_ok"] and not v["missing_shards"])
    panel_measured = SCHEMA.panel(all_events, MEASURED_SEAMS)
    # ATTRIBUTION (gates/attribution_required): the panel's headline host_decision_share is a treatment-shaped
    # quantity (it moves only on an applied intervention). Its natural control is the sham comparison already
    # computed above (validity["sham_clean"]): intact_a vs intact_b, NO intervention applied at all, so any
    # reply difference there is pure reply-instability noise, not host ownership. Ask whose the effect is,
    # rather than banking host_decision_share on faith the way gap#5 banked `inherit` on faith.
    n_sham_checks = len(validity["sham_clean"])
    sham_noise_rate = (1.0 - sum(1 for ok_ in validity["sham_clean"].values() if ok_) / n_sham_checks
                        if n_sham_checks else None)
    host_share_val = panel_measured.get("host_decision_share")
    sham_noise_attribution = None
    if host_share_val is not None and sham_noise_rate is not None:
        sham_noise_attribution = attributable_to(
            "panel host_decision_share vs sham (no-intervention) reply-noise floor", host_share_val, sham_noise_rate)
    v["sham_noise_rate"] = sham_noise_rate
    v["sham_noise_attribution"] = sham_noise_attribution
    return {
        "instrument": "host_share", "schema": "research/runners/host_share_schema.py", "seeds": seeds,
        "verdict": "BASELINE-VALID" if ok else "INVALID-OR-INCOMPLETE",
        "panel_measured": panel_measured,
        "panel_call_graph_checks": SCHEMA.panel(all_events, [s for s in SEAMS if s not in MEASURED_SEAMS]),
        "content": {"learned_content_fraction": _mean([r["learned_content_fraction"] for r in content_rows]),
                    "python_selected_fraction": _mean([r["python_selected_fraction"] for r in content_rows]),
                    "unattributed_fraction": _mean([r["unattributed_fraction"] for r in content_rows]),
                    "fact_word_share": _mean([r["fact_word_share"] for r in content_rows]),
                    "n_turns": len(content_rows)},
        "seams": seams, "validity": v, "content_rows": content_rows, "events": all_events,
    }


# ══════════════════════════════════════════════ SELF-TEST (no brain build) ═════════════════════════════════════════
def _fake_resp(patient, abstained=False, confident=True, answer=None):
    return {"answer": answer or ("i don't know" if abstained else "the dog chases the %s" % patient),
            "abstained": abstained, "recalled_svo": (None if abstained else ["dog", "chase", patient]),
            "supporting_facts": ([] if abstained else [["dog", "chase", patient]]), "n_sentences": 0 if abstained else 1,
            "verified": not abstained, "metacog": {"confident": confident}, "source": EXPECTED_SOURCE}


def selftest(verbose=True):
    ok_schema, _sc = SCHEMA.selftest()
    lab = "hsc_dog_q"
    intact = _fake_resp("ball")
    obs_v = {"seam": "v-choice", "mode": "observe", "applied": False, "cands": ["cat", "ball"], "host_raw": [0.0, 1.0],
             "used_engagement": [0.0, 1.0]}
    snap_ball = {lab: {"value_choice": {"chosen": "ball"}}}
    ia = {"responses": {lab: intact}, "events": {}, "snapshots": snap_ball}
    ib = {"responses": {lab: intact}, "events": {lab: [obs_v, {"seam": "_recall", "agent": "dog", "action": "chase",
                                                                "patient": "cat"}]}, "snapshots": snap_ball}

    def iv(resp, applied=True, raw=(0.0, 1.0)):
        return {"responses": {lab: resp}, "snapshots": {},
                "events": {lab: [dict(obs_v, mode="intervene", applied=applied, host_raw=list(raw))]}}

    def sb(resp, chosen):
        return {"responses": {lab: resp}, "events": {lab: [dict(obs_v, mode="substitute", applied=True)]},
                "snapshots": {lab: {"value_choice": {"chosen": chosen}}}}
    e_host = seam_event("v-choice", lab, ia, ib, iv(_fake_resp("cat")), sb(_fake_resp("cat"), "cat"))
    e_repl = seam_event("v-choice", lab, ia, ib, iv(_fake_resp("cat")), sb(intact, "ball"))
    e_neur = seam_event("v-choice", lab, ia, ib, iv(intact), sb(intact, "ball"))
    e_unap = seam_event("v-choice", lab, ia, ib, iv(intact, applied=False), sb(intact, "ball"))
    e_drift = seam_event("v-choice", lab, ia, ib, iv(_fake_resp("cat"), raw=(0.2, 1.0)), sb(intact, "ball"))
    e_sham = seam_event("v-choice", lab, ia, dict(ib, responses={lab: _fake_resp("cat")}), iv(_fake_resp("cat")),
                        sb(intact, "ball"))
    e_nolive = seam_event("k-select", lab, ia, ib, iv(_fake_resp("cat")), sb(intact, "ball"))
    e_decline = seam_event("v-choice", lab, ia, ib, iv(_fake_resp("cat")), sb(intact, None))
    cf = content_fractions(_fake_resp("ball"), ib["events"][lab], {"chosen": "ball"})
    cf2 = content_fractions(_fake_resp("cat", answer="my decision-margin reads this as low-confidence, so take it "
                                                     "as uncertain: the dog chases the cat"), ib["events"][lab])
    bucket = lambda e: SCHEMA.event_bucket(e["F"], e["S"], e["f_outcome"])  # noqa: E731
    checks = {
        "schema selftest": ok_schema,
        "follows host + neural differs -> host_decided": (e_host["F"], e_host["S"], bucket(e_host)) == (1, 1,
                                                                                                        "host_decided"),
        "F=1,S=0 -> host_causal_replaceable, NEVER neural": bucket(e_repl) == "host_causal_replaceable",
        "returned to the neural preference -> neural_decided": (e_neur["F"], bucket(e_neur)) == (0, "neural_decided"),
        "intervention not applied -> F UNDEFINED (V3)": e_unap["F"] is None
            and any(u.startswith("V3") for u in e_unap["undefined"]),
        "seam state drift -> F UNDEFINED (V5)": e_drift["F"] is None
            and any(u.startswith("V5") for u in e_drift["undefined"]),
        "sham failure -> event UNDEFINED (V1)": (e_sham["live"], e_sham["F"], e_sham["S"]) == (False, None, None),
        "seam not live -> UNDEFINED": (e_nolive["live"], e_nolive["F"]) == (False, None),
        "organ declined in the substitute arm -> D UNDEFINED, not 0": e_decline["D"] is None,
        "value-choice-committed fact is python-selected, not learned": cf is not None
            and cf["python_selected_fraction"] == 1.0 and cf["learned_content_fraction"] == 0.0,
        "substrate-recalled fact counts as learned content": cf2 is not None and cf2["learned_content_fraction"] == 1.0,
        "hedge template words lower the fact-word share": cf2 is not None and cf2["fact_word_share"] < 0.5,
        "abstain -> no content row": content_fractions(_fake_resp(None, abstained=True), []) is None,
        "hedge / hold pseudo-fields read": _get({"answer": "x " + HEDGE_MARK}, "hedge") is True
            and _get({"bg_select": {"on": True}}, "hold") is True,
        "call-graph seams are never in the measured panel": not (set(MEASURED_SEAMS) & {"organ-c", "metacog",
                                                                                       "ignition"}),
    }
    ok = all(checks.values())
    if verbose:
        for k, val in checks.items():
            print(("PASS " if val else "FAIL ") + k)
        print("SELFTEST", "OK" if ok else "FAILED")
    return ok, checks


def report_markdown(rep):
    """A fixed-format Markdown result block from an aggregate (no judgment needed to write the finding body)."""
    f = lambda x: "UNDEFINED" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))  # noqa: E731
    L = ["**Verdict (preregistered validity): %s** (seeds %s)" % (rep["verdict"], rep["seeds"]), "",
         "| seam | reported as | class | HS | NS | MS | SC | DA | n events with F | live seeds |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in rep["seams"].items():
        st = d["stats"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s, d["reported_as"], d["class"], f(st["HS"]), f(st["NS"]), f(st["MS"]), f(st["SC"]), f(st["DA"]),
            st["n_F"], len(st["live_seeds"])))
    for name, key in (("Panel, measured seams", "panel_measured"), ("Panel, call-graph checks", "panel_call_graph_checks")):
        pn = rep[key]
        L += ["", "%s (%s): host_decision_share %s, neural_decision_fraction %s, masked_fraction %s, "
              "replaceable_share_of_host %s, n events %s." % (name, ", ".join(pn["measured_seams"]),
                                                              f(pn["host_decision_share"]),
                                                              f(pn["neural_decision_fraction"]),
                                                              f(pn["masked_fraction"]),
                                                              f(pn["replaceable_share_of_host"]),
                                                              pn["n_events_F_defined"])]
    c = rep["content"]
    L += ["", "Content (%s turns): learned_content_fraction %s, python_selected_fraction %s, unattributed %s, "
          "fact_word_share %s." % (c["n_turns"], f(c["learned_content_fraction"]), f(c["python_selected_fraction"]),
                                   f(c["unattributed_fraction"]), f(c["fact_word_share"]))]
    v = rep["validity"]
    L += ["", "Validity: V1 sham-clean seeds %s (pass %s); V3 applied %s; V4 %s; V5 drift events %s; V6 sources %s "
          "(pass %s); PYTHONHASHSEED %s; seed threading %s; missing shards %d." % (
              v["V1_sham_clean_seeds"], v["V1_pass"], v["V3_interventions_applied"],
              v["V4_both_outcomes_registrable"], v["V5_state_drift_events"], v["V6_sources"], v["V6_pass"],
              v["pythonhashseed_all_ok"], v["seed_threading_all_ok"], len(v["missing_shards"]))]
    for s, d in rep["seams"].items():
        if d["undefined_reasons"]:
            L.append("- %s UNDEFINED reasons: %s" % (s, "; ".join(d["undefined_reasons"])))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=None, metavar="AGGREGATE_JSON", help="print the fixed Markdown result block")
    ap.add_argument("--worker", action="store_true", help="internal: build one arm in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, group, arm) arm in fresh subprocesses")
    ap.add_argument("--seed-all", action="store_true", help="run every (group, arm) shard of --seed sequentially")
    ap.add_argument("--aggregate", default=None, help="shard directory to aggregate")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the gate's pool job lines (TSV)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --aggregate")
    ap.add_argument("--group", default=None, choices=sorted(GROUPS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--session-index", type=int, default=None, help="internal: the one session a --worker runs")
    ap.add_argument("--labels", default=None, help="DEV ONLY: run only these turn labels (comma-separated)")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"),
                    help="numpy = the preregistered gate backend; cupy = dev-only")
    ap.add_argument("--allow-gate-seed", action="store_true",
                    help="required to run a GATE seed (42/43/44/100/101/102): only after the prereg is committed")
    a = ap.parse_args()

    if a.selftest:
        ok, _ = selftest()
        return 0 if ok else 1
    if a.report:
        print(report_markdown(json.load(open(a.report))))
        return 0
    if a.emit_pool_jobs:
        for cmd, why in pool_job_lines(a.emit_pool_jobs):
            print("%s\t%s" % (cmd, why))
        return 0
    if a.aggregate:
        seeds = [int(s) for s in (a.seeds or "").split(",") if s.strip()]
        rep = aggregate(a.aggregate, seeds)
        out = a.out or os.path.join(a.aggregate, "aggregate.json")
        with open(out, "w") as f:
            json.dump(rep, f, indent=1, default=str)
        slim = {k: rep[k] for k in ("verdict", "panel_measured", "panel_call_graph_checks", "content", "validity")}
        slim["seams"] = {s: {"class": d["class"], "reported_as": d["reported_as"], "HS": d["stats"]["HS"],
                             "NS": d["stats"]["NS"], "MS": d["stats"]["MS"], "SC": d["stats"]["SC"],
                             "DA": d["stats"]["DA"], "n_F": d["stats"]["n_F"], "live_seeds": d["stats"]["live_seeds"]}
                         for s, d in rep["seams"].items()}
        print(json.dumps(slim, indent=1, default=str))
        return 0
    if a.seed is None:
        ap.error("--seed is required")
    if a.seed in GATE_SEEDS and not a.allow_gate_seed:
        ap.error("seed %d is a GATE seed: pass --allow-gate-seed (only after the preregistration is committed)" % a.seed)
    if a.seed in GATE_SEEDS and a.backend != "numpy":
        ap.error("the gate run is preregistered on numpy; cupy is dev-only")
    only = [x for x in (a.labels or "").split(",") if x.strip()] or None
    if only and a.seed in GATE_SEEDS:
        ap.error("--labels is a dev aid; the gate runs every preregistered turn")
    if a.worker:
        return run_worker(a.seed, a.group, a.arm, a.out, backend=a.backend, session_index=a.session_index,
                          only_labels=only)
    if a.shard:
        return 0 if run_shard(a.seed, a.group, a.arm, a.out, backend=a.backend, only_labels=only) else 1
    if a.seed_all:
        fails = []
        for group in (GROUPS if a.group is None else [a.group]):
            for arm in arms_for(group):
                p = shard_path(a.out_dir, a.seed, group, arm)
                if a.skip_existing and os.path.exists(p):
                    continue
                if not run_shard(a.seed, group, arm, p, backend=a.backend, only_labels=only):
                    fails.append(p)
        print("[host_share] seed %d done; failed shards: %s" % (a.seed, fails), flush=True)
        return 0 if not fails else 1
    ap.error("choose one of --worker/--shard/--seed-all/--aggregate/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    sys.exit(main())
