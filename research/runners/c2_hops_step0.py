"""C2 M1 STEP 0: the engagement smoke of the spiking-hops packet, read-only (research/coordination/c2-spiking-hops-PACKET.md,
round 9, section 2.10 step 0; tables 2.8; premise map 2.9).

WHAT. The step-0 reads (a)-(h) of packet 2.10 as the informational step-0 mode of the Stage-2 arm worker
(research/runners/stage2_exit_gate.py `run_worker`, entries `infer/c2s0_*`), one shard per (table, seed), and a scorer
that applies the packet's GO rule and STOP lines VERBATIM to the shards. No production edit, no default changed, no M1
code (none exists yet): every read runs on the production default path after the teach block, inside
`stage2_exit_gate.substrate_read_guard` (global RNG preserved, composer bridge v/u/firing and lazy sequencer state put
back), and the inert-read check (h) tests that on seed 7.

THE READS (made once per full step-0 worker, after the teach block, before any `.kb` fixture and the first probe):
  (a) `composer.query_patient_source` for (heron, hunt), (heron, eat), (frog, eat), (frog, chase): result, tier, the
      selected block, `propose_path`, `fs_candidates` with the cue roles, the decoded rows under the cue from one full
      `_read_blocks` scan, and `host_routed` (decoded rows under the cue, buffer PROPOSE None).
  (b) the sequencer the production path builds (`_ensure_sequencer` at the composer defaults, the REDUCED fabric), every
      block's `m{b}` rate for the same cues through `run_sequencerK_reduced_with_drive` with the composer's own drives,
      the sequencer decision against the host first-match, and the build's memory and time.
  (c) for EVERY stored block, the production `block_cleanup_scores` then the PATIENT row of the cleanup membrane read
      right after it (clipped at 0), relayed through `onbridge_divnorm_drive` on the composer's score bridge at its
      `input_gain`: the patient pools that fire per block, V'_P (their union in block order) beside the decoded stored
      patients; and the full probe `block_cleanup_scores_full` as a cross-check under the P10 treatment (clip at 0,
      every reserve slot -- `_silent_reserve(j)` True -- zeroed, compared on the non-reserve slots).
  (f) every premise-map cue of the table (2.9) on the bound store (decoded rows) and the LTM (`ltm.query_patient`,
      `all_candidate_patients` on the routed shard), plus the record-only `.kb` audit counts per tier.
  The worker itself records (d) `chat._last_gnw_delib` per probe turn, (e) the D4 recorder and the teach replies'
  `comprehension` / `surprise` / `reconsolidation`, the `infer_xtier` fixture record and `ws_seed`, and (g) the `.kb`
  overwrite fixtures (A17's plan on `infer_amb1`, A19's on `infer_dup`) with their engagement record (i)-(iv).
  (h) seed 7 only: the `infer` table in three fresh workers (`c2s0_infer_bare`, `c2s0_infer_rec`, `c2s0_infer`).

M3 STEP 0 (packet 3.8), the part that fits this runner: (d), the nine `infer_garbled` nonce forms checked against the
hand tables, the learned lexicons, the corpus vocabulary and `_brain_vocab(chat)` on the `infer` table's full worker
right after its four intact teaches (`read_m3_d`, scored report-only by `score_m3_step0`). Not built here: (a) needs the
M3 nonce item generator (W, G1-G5, out-of-scope; fixed seed) as a step-0 table; (b) builds `WordRoleParserOrgan` with
no server import on seeds 7, 1-6 and 8-16; (c) is a document read of the cs-race f1h/f1i results; (e) needs
`frame_conforming` and the frame packet's exposure frames.

  .venv/bin/python -m research.runners.c2_hops_step0 --score research/findings/raw/_c2_step0 [--seeds 7] [--out X]
  .venv/bin/python -m research.runners.c2_hops_step0 --emit-pool-lines '<sha>' [--mem-gb 9]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage2_exit_gate as SG  # noqa: E402

PACKET = "research/coordination/c2-spiking-hops-PACKET.md"
OUT_ROOT = "research/findings/raw/_c2_step0"
GROUP = "infer"
SEEDS = (7, 1, 2, 3, 4, 5, 6)                    # design seed 7 first, then the dev seeds 1-6 (packet 2.10 step 0)
INERT_SEED = 7
TABLE_ARMS = {"infer": "c2s0_infer", "infer_amb1": "c2s0_amb1", "infer_amb2": "c2s0_amb2", "infer_dup": "c2s0_dup",
              "infer_xtier": "c2s0_xtier"}
INERT_ARMS = {"full": "c2s0_infer", "recorder": "c2s0_infer_rec", "bare": "c2s0_infer_bare"}
NEW_TABLES = ("infer_amb1", "infer_amb2", "infer_dup", "infer_xtier")
CUES_A = (("heron", "hunt"), ("heron", "eat"), ("frog", "eat"), ("frog", "chase"))
EXPECTED_LTM_BUNDLE = "wikidata_100k"
# packet 2.8 "Which competent() branch each new teach takes" (keyed by the teach sentence, which is what the table rows name)
EXPECTED_BRANCH = {"the heron hunts the frog": "pass_through", "the frog eats the gnat": "pass_through",
                   "the heron hunts the worm": "pass_through", "the frog eats the apple": "covered"}
AMBIGUOUS_CUES = {"infer_amb1": (("heron", "hunt"),), "infer_amb2": (("frog", "eat"),)}
RETOLD_CUES = {"infer_dup": (("heron", "hunt"), ("frog", "eat"))}
OVERWRITE_TABLES = {"infer_amb1": "amb1_worm_to_frog", "infer_dup": "dup_first_frog_to_worm"}
XTIER_TABLE, XTIER_FIXTURE = "infer_xtier", "xtier_heron_hunt_fish"
ABSENCE_CUE = ("frog", "chase")
# M3 step 0 (d) (packet 3.8, 3.6): the three garbled teaches of `infer_garbled`, as the packet writes them, and their
# nine nonce forms. The forms are checked on the `infer` table's full worker right after its four intact teaches (the
# packet's "right after the four intact teaches"); a form found anywhere is redrawn before the table is pinned. Report
# only: never a C2 GO or STOP input.
M3_GARBLED_TEACHES = ("the zib the vop tavs", "pels the kig the dov", "the bim ruvs fep the")
M3_GARBLED_FORMS = ("zib", "vop", "tavs", "pels", "kig", "dov", "bim", "ruvs", "fep")
M3_D_TABLE = "infer"
M3_D_SOURCES = ("hand_animacy", "hand_verb_selects", "animacy_of", "verb_selects_of", "learned_animacy_vocab",
                "learned_verbselects_vocab", "corpus", "brain_vocab")
# pool-line resources, measured on the seed-7 local smoke under memcap 10 (2026-10-28, numpy, one thread): worker peak
# RSS 6.49-6.92 GB over the seven entries (c2s0_dup the largest, 6925 MB; wall 725-1222 s), so mem_gb 9 (peak x 1.3)
# under the Stage-2 lane's memcap 10
MEM_GB = 9
MEMCAP_GB = 10
MAX_AGE_H = 240


# ══════════════════════════════════════ THE PREMISE MAP (packet 2.9), as data ══════════════════════════════════════
def _row(probe, cues, sources, absence=(), reported_only=False):
    """One map row: `cues` = {(agent, action): (buffer patients, LTM patients)}, exact sets; `sources` = teach labels,
    `builtin:<a>,<v>,<p>` or `fixture:<name>`; `absence` = cues whose premise is that nothing is stored (read from the
    teach log and builtin_facts ONLY, round 5)."""
    return {"probe": probe, "cues": {k: (frozenset(b), frozenset(l)) for k, (b, l) in cues.items()},
            "sources": list(sources), "absence": [tuple(a) for a in absence], "reported_only": bool(reported_only)}


def _infer_rows(ltm_hh=(), extra_hh=(), extra_fe=(), hh_fixture=()):
    hh, fe = ("heron", "hunt"), ("frog", "eat")
    return [
        _row("p_prey_eat", {hh: ({"frog"}, ltm_hh), fe: ({"gnat"}, ())},
             ["t_heron_hunt", "t_frog_eat"] + list(extra_hh) + list(extra_fe) + list(hh_fixture)),
        _row("c_heron_hunt", {hh: ({"frog"}, ltm_hh)}, ["t_heron_hunt"] + list(extra_hh) + list(hh_fixture)),
        _row("c_frog_eat", {fe: ({"gnat"}, ())}, ["t_frog_eat"] + list(extra_fe)),
        _row("c_prey_chase", {hh: ({"frog"}, ltm_hh)}, ["t_heron_hunt"] + list(extra_hh) + list(hh_fixture),
             absence=[ABSENCE_CUE]),
        _row("p_yesno", {("crow", "chase"): ({"wren"}, ()), ("wren", "chase"): ({"moth"}, ())},
             ["t_crow_chase", "t_wren_chase"], reported_only=True),
        _row("c_dog_chase", {("dog", "chase"): ({"cat"}, ())}, ["builtin:dog,chase,cat"]),
    ]


def _amb_rows(hh_set, fe_set, sources):
    return [_row("p_prey_eat", {("heron", "hunt"): (hh_set, ()), ("frog", "eat"): (fe_set, ())}, sources),
            _row("c_heron_hunt", {("heron", "hunt"): (hh_set, ()), ("frog", "eat"): (fe_set, ())}, sources),
            _row("c_dog_chase", {("dog", "chase"): ({"cat"}, ())}, ["builtin:dog,chase,cat"])]


PREMISE_MAP = {
    "infer": _infer_rows(),
    "infer_amb1": _amb_rows({"frog", "worm"}, {"gnat"}, ["t_heron_hunt", "t_heron_hunt_worm", "t_frog_eat"]),
    "infer_amb2": _amb_rows({"frog"}, {"gnat", "apple"}, ["t_heron_hunt", "t_frog_eat", "t_frog_eat_apple"]),
    "infer_dup": _infer_rows(extra_hh=["t_heron_hunt_re"], extra_fe=["t_frog_eat_re"]),
    "infer_xtier": _infer_rows(ltm_hh={"fish"}, hh_fixture=["fixture:%s" % XTIER_FIXTURE]),
}


def premise_cues(table):
    """Every cue (f) reads on `table`: the map's cues, the absence cue where a row names it, and the four (a) cues."""
    out = []
    for r in PREMISE_MAP[table]:
        for c in list(r["cues"]) + list(r["absence"]):
            if c not in out:
                out.append(c)
    for c in CUES_A:
        if c not in out:
            out.append(c)
    return out


# ═════════════════════════════════════════ THE READS (inside the arm worker) ════════════════════════════════════════
def _rss_mb():
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    out[line.split(":")[0]] = round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    out["ru_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    return out


def _err(e):
    return {"error": "%s: %s" % (type(e).__name__, e), "traceback": traceback.format_exc()[-1500:]}


def _key(cue):
    return "%s,%s" % tuple(cue)


def _read_layout(buf):
    if getattr(buf, "enable_sparse_index", False) and getattr(buf, "confidence_gate", 0.0) == 0.0:
        return "indexed_per_block"
    return "batched" if getattr(buf, "enable_batched", False) else "per_block_loop"


def run_reads(chat, seed, table, pprobe):
    """Reads (a)-(c) and (f) on the live session, after the teach block (one call per full step-0 worker). Every read
    records its own exception instead of raising, so one failing read never loses the others or the turn table."""
    out = {"table": table, "seed": int(seed), "packet": PACKET}
    t0 = time.time()
    try:
        comp = chat.inner.composer
        buf = getattr(comp, "buffer", comp)
        ltm = getattr(comp, "ltm", None)
        out["composer_types"] = {"composer": type(comp).__name__, "buffer": type(buf).__name__,
                                 "ltm": type(ltm).__name__}
        out["K"] = len(buf.kb)
        out["kb_rows"] = SG._kb_triples(buf)                       # record-only: the list, beside the bound store
        out["read_layout"] = _read_layout(buf)
        out["fact_shard_active"] = bool(buf._fact_shard_active())
        out["integrated_loop"] = SG._jsonable(getattr(buf, "integrated_loop", None))
        with SG.substrate_read_guard(buf) as guard:
            out["guard"] = guard
            rows = SG.decoded_rows(buf)
            out["decoded_rows"] = rows
            out["a"] = read_a(comp, buf, rows, pprobe)
            out["b"] = read_b(buf, rows)
            out["c"] = read_c(buf, rows)
            out["f"] = read_f(buf, ltm, rows, table)
            if table == M3_D_TABLE:
                try:                                            # M3's read never voids C2's reads
                    out["m3_d"] = read_m3_d(chat)
                except Exception as e:
                    out["m3_d"] = _err(e)
    except Exception as e:
        out.update(_err(e))
    out["wall_s"] = round(time.time() - t0, 1)
    out["rss_after"] = _rss_mb()
    return out


def _corpus_counts(words, path=None):
    """Whole-word counts of `words` in the corpus file, tokenized as the learned lexicons tokenize it
    (`_comprehension_learned_animacy_cue_derisk.load_tokens`, over the WHOLE file, not the lexicons' 8M-char prefix).
    None when the corpus file is absent (the read is then UNDEFINED, packet 3.6)."""
    path = path or os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(path):
        return None
    from research.runners._comprehension_learned_animacy_cue_derisk import _WORD_RE as word_re
    want = set(words)
    cnt = {w: 0 for w in words}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for t in word_re.findall(line.lower()):
                if t in want:
                    cnt[t] += 1
    return cnt


def read_m3_d(chat, forms=M3_GARBLED_FORMS):
    """M3 step 0 (d) (packet 3.8, 3.6): each `infer_garbled` nonce form against ANIMACY, VERB_SELECTS, the learned
    lexicons (the production `_animacy_of` / `_verb_selects_of` at its defaults, and the learned lexicons' scored
    vocabularies), the corpus vocabulary and `_brain_vocab(chat)` read right after the four intact teaches. A form found
    in any of them is listed in `redraw`."""
    from research.runners import comprehension_production_organ as CPO
    from research.runners import _comprehension_learned_animacy_spiking as LA
    from research.runners import _comprehension_learned_verbselects_cue_derisk as LV
    import webapp.server as srv
    out = {"forms": list(forms), "learned_animacy_enabled": bool(CPO.learned_animacy_cue_enabled()),
           "learned_verbselects_enabled": bool(CPO.learned_verb_selects_enabled()),
           "animacy_lexicon_prebuilt": LA._LEXICON is not None, "verbselects_lexicon_prebuilt": LV._LEXICON is not None}
    bv = srv._brain_vocab(chat)
    out["brain_vocab_size"] = len(bv)
    try:
        corpus = _corpus_counts(forms)
    except Exception as e:
        corpus = None
        out["corpus_error"] = "%s: %s" % (type(e).__name__, e)
    out["corpus_read"] = corpus is not None
    per = {}
    for w in forms:
        r = {"hand_animacy": w in CPO.ANIMACY, "hand_verb_selects": w in CPO.VERB_SELECTS}
        try:
            r["animacy_of"] = CPO._animacy_of(w)
            r["verb_selects_of"] = SG._jsonable(CPO._verb_selects_of(w))
        except Exception as e:
            r["lexicon_error"] = "%s: %s" % (type(e).__name__, e)
        la, lv = LA._LEXICON, LV._LEXICON
        r["learned_animacy_vocab"] = (w in la.scores) if la is not None else None
        r["learned_verbselects_vocab"] = (w in lv.scores) if lv is not None else None
        r["corpus_count"] = None if corpus is None else int(corpus.get(w, 0))
        r["in_brain_vocab"] = w in bv
        hits = {"hand_animacy": r["hand_animacy"], "hand_verb_selects": r["hand_verb_selects"],
                "animacy_of": r.get("animacy_of") is not None, "verb_selects_of": r.get("verb_selects_of") is not None,
                "learned_animacy_vocab": bool(r["learned_animacy_vocab"]),
                "learned_verbselects_vocab": bool(r["learned_verbselects_vocab"]),
                "corpus": bool(r["corpus_count"]), "brain_vocab": r["in_brain_vocab"]}
        r["found_in"] = [k for k in M3_D_SOURCES if hits[k]]
        per[w] = r
    out["per_form"] = per
    out["redraw"] = [w for w in forms if per[w]["found_in"]]
    return out


def read_a(comp, buf, rows, pprobe):
    out = {}
    for a, x in CUES_A:
        r = {"cue": [a, x]}
        try:
            with (pprobe.scope() if pprobe is not None else contextlib.nullcontext([])) as ev:
                if hasattr(comp, "query_patient_source"):
                    res, tier = comp.query_patient_source(a, x)
                else:
                    res = comp.query_patient(a, x)
                    tier = "buffer" if res is not None else None
            r["result"], r["tier"] = SG._jsonable(res), tier
            r["buffer_propose"] = SG._jsonable(res) if tier == "buffer" else None
            r.update(SG.propose_path_of(ev))
            r["events"] = SG._jsonable(ev)
            dec = SG.rows_under(rows, a, x)
            r["decoded_rows_under_cue"] = dec
            # packet 2.10 step 0 (a), round 5: decoded rows under the cue while the buffer's PROPOSE is None
            r["host_routed"] = bool(dec) and r["buffer_propose"] is None
        except Exception as e:
            r.update(_err(e))
        out[_key((a, x))] = r
    return out


def read_b(buf, rows):
    from research.runners.one_brain_composer import _seq_imports
    fns = _seq_imports()
    K = len(buf.kb)
    thr = float(buf.sequencer_match_thresh)
    out = {"K": K, "match_thresh": thr, "reduced": bool(buf.enable_seq_vocab_shrink),
           "input_gain": float(buf.sequencer_input_gain)}
    try:
        out["rss_before"] = _rss_mb()
        t = time.time()
        buf._ensure_sequencer(K)
        out["build_s"] = round(time.time() - t, 2)
        out["rss_after_build"] = _rss_mb()
        sb, meta = buf._seq
        out["fabric"] = {"n_neurons": int(sb.core_config.num_neurons),
                         "meta": {k: SG._jsonable(v) for k, v in meta.items() if isinstance(v, (int, float, str))}}
        if out["reduced"]:
            out["fabric"]["agentsA"] = list(buf._seq_mapA)
            out["fabric"]["actionsX"] = list(buf._seq_mapX)
        out["drives_lit"] = [[int((dA > 0).sum()), int((dX > 0).sum())] for dA, dX in buf._seq_drives]
        out["drives_lit_words"] = [[[buf.words[j] for j in range(len(dA)) if dA[j] > 0],
                                    [buf.words[j] for j in range(len(dX)) if dX[j] > 0]] for dA, dX in buf._seq_drives]
    except Exception as e:
        out.update(_err(e))
        return out
    cues = {}
    for a, x in CUES_A:
        c = {"cue": [a, x]}
        true_m = [r["block"] for r in SG.rows_under(rows, a, x)]
        c["true_matches"] = true_m
        c["host_first_match"] = true_m[0] if true_m else None
        try:
            if out["reduced"]:
                in_vocab = a in buf._seq_mapA and x in buf._seq_mapX
            else:
                in_vocab = a in buf._word_index and x in buf._word_index
            c["cue_in_reduced_vocab"] = bool(in_vocab)
            fired = []
            if in_vocab:
                spur = {}
                t = time.time()
                if out["reduced"]:
                    dec, rates = fns["run_sequencerK_reduced_with_drive"](
                        sb, meta, buf.words, buf._seq_mapA, buf._seq_mapX, a, x, buf._seq_drives, match_thresh=thr,
                        spurious_counter=spur)
                else:
                    dec, rates = fns["run_sequencerK_with_drive"](sb, meta, buf._word_index[a], buf._word_index[x],
                                                                  buf._seq_drives, match_thresh=thr)
                c["run_s"] = round(time.time() - t, 2)
                m = {b: float(rates["m%d" % b]) for b in range(K)}
                fired = [b for b in range(K) if m[b] > thr]
                c.update(decision=dec, winner=SG._jsonable(rates.get("winner")), m_rates=[m[b] for b in range(K)],
                         spurious_dropped=spur, sequencer_block=fns["decision_to_block"](dec, K))
            else:
                c.update(decision=None, winner=None, m_rates=None, sequencer_block=None)
            c["fired"] = fired
            c["missed_true_matches"] = [b for b in true_m if b not in fired]
            c["fired_non_matching"] = [b for b in fired if b not in true_m]
            c["decision_equals_host_first_match"] = c.get("sequencer_block") == c["host_first_match"]
        except Exception as e:
            c.update(_err(e))
        cues[_key((a, x))] = c
    out["cues"] = cues
    return out


# ═══════════════════════════════════ STEP 0b: local feedback inhibition among the patient word pools ═══════════════════════
# c2-s0-result-1029: the step-0 read (c) lit 3-6 patient pools per block because divisive normalization divides but nothing
# makes the pools compete. The missing companion process is local feedback inhibition (E->I->E, soft winner-take-all): ONE
# shared fast-spiking population on the score bridge, excited by every word pool and projecting inhibitory synapses back onto
# them. Runner-side only (a separate score bridge built here with the same word-pool layout, flags and seed as the composer's
# `build_divnorm_score_bridge`, plus the `pinh` region; the inhibitory sign comes from the region being exc_fraction=0, the
# precedent of `inh{b}` in _seq_vocab_shrink_derisk.py). No sim/ edit, no composer edit, default OFF: the baseline read_c path
# is untouched unless `C2S0_PATIENT_INHIB` is set. NOT a cross-channel divnorm pool (the 2026-09-03 NO-GO): E->I->E feedback
# restricted to the V word pools of this one bridge. No host argmax / threshold / top-k: only spiking synapses compete.
PATIENT_INHIB_ENV = "C2S0_PATIENT_INHIB"
PATIENT_INHIB_MODES = ("off", "on", "lesion", "all", "ff", "ff_lesion", "ffx", "ffx_lesion")
# step 0d: `ffx` = arms ffx_off + ffx (+ I->E sweep arms ffx@g) + ffx_lesion on a score bridge built with
# cfg.input_divisive_norm_external_only=True (the divisive stage divides only the external drive; sim/ flag, default OFF);
# `ffx_lesion` = the lesion arm alone. The WORD pools are read directly.
# step 0c (feedforward inhibition, c2 ruling after the 0b NO-GO): `ff` = arms stable_off + ff + ff_lesion (all on the
# stable-integration bridge), `ff_lesion` = the lesion arm alone. See the finding 2026-10-29-c2-step0b-patient-inhibition.md.
INHIB_DEFAULTS = {"ff_gain": 3e-4, "ff_w_i2e": 40.0, "ffx_w_i2e": 2.0, "w_relay": 300.0, "relay_w_i2e": 5.0, "n_inh": 30, "w_e2i": 60.0, "w_i2e": 320.0, "izh_type": "IZH2007_FS_CORTICAL_INTERNEURON", "settle": 20}
STEP0B_STOP_LINES = (
    "S0B_lesion_equals_on: STOP if the lesion arm (same population, I->E zeroed) reads identically to the `on` arm "
    "(per block, same fired_pools): the inhibition is not load-bearing.",
    "S0C_lesion_equals_on / S0C_on_not_one_correct_pool: the same two lines for the feedforward arms (`ff` vs `ff_lesion`).",
    "S0B_on_not_one_correct_pool: STOP if any block in `on` fires more than one pool, fires a pool that is not the decoded "
    "patient, or fires zero pools (over-suppression).",
)


def patient_inhib_env():
    v = os.environ.get(PATIENT_INHIB_ENV, "off") or "off"
    if v not in PATIENT_INHIB_MODES:
        raise ValueError("%s=%r not in %s" % (PATIENT_INHIB_ENV, v, PATIENT_INHIB_MODES))
    return v


def patient_inhib_knobs():
    k = dict(INHIB_DEFAULTS)
    for name, cast in (("n_inh", int), ("w_e2i", float), ("w_i2e", float), ("settle", int)):
        e = os.environ.get("C2S0_INHIB_" + name.upper())
        if e not in (None, ""):
            k[name] = cast(e)
    e = os.environ.get("C2S0_INHIB_FF_GAIN")
    if e:
        k["ff_gain"] = float(e)
    e = os.environ.get("C2S0_INHIB_FF_W_I2E")
    if e:
        k["ff_w_i2e"] = float(e)
    e = os.environ.get("C2S0_INHIB_FFX_W_I2E")
    if e:
        k["ffx_w_i2e"] = float(e)
    sw = os.environ.get("C2S0_INHIB_FFX_SWEEP")
    k["sweep_ffx_w_i2e"] = [float(x) for x in sw.split(",")] if sw else []
    sw = os.environ.get("C2S0_INHIB_FF_SWEEP")
    k["sweep_ff_gain"] = [float(x) for x in sw.split(",")] if sw else []
    sw = os.environ.get("C2S0_INHIB_SWEEP")                  # step 0b's ONE allowed gain sweep: extra `on@<w_i2e>` arms
    k["sweep_w_i2e"] = [float(x) for x in sw.split(",")] if sw else []
    return k


def build_inhib_score_bridge(seed, V, sigma, gain, n_word=20, n_inh=30, w_e2i=60.0, w_i2e=320.0,
                             izh_type="IZH2007_FS_CORTICAL_INTERNEURON", ref_sb=None, stable=False, w_relay=None, external_only=False):
    """`build_divnorm_score_bridge(seed, V, n_word, True, sigma, gain)` plus ONE shared FS inhibitory region `pinh`:
    w{w} -> pinh (excitatory, w_e2i) and pinh -> w{w} (inhibitory by the presynaptic region's exc_fraction=0, w_i2e).
    w_i2e=0 builds the same population and the same synapses with zero weight (the lesion arm)."""
    from sim import CoreSimConfig, SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_input_divisive_norm = True
    cfg.input_divisive_sigma = float(sigma)
    cfg.input_divisive_gain = float(gain)
    if external_only:   # sim/ flag (default OFF): divisive stage divides only the external drive, synaptic current undivided
        cfg.input_divisive_norm_external_only = True
    if stable:      # the SIM_STABLE_INTEGRATION set (SimulationBridge._apply_stable_integration), set explicitly
        cfg.enable_bounded_conductance_integration = True
        cfg.izh_membrane_substeps = 4
        cfg.izh_substep_refractory_rule = "model_reset"
    regions = [BrainRegion(name="w%d" % w, n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                           input_divisive_norm=True) for w in range(V)]
    regions.append(BrainRegion(name="pinh", n_neurons=int(n_inh), exc_fraction=0.0, internal_density=0.0,
                               izh_neuron_type=izh_type))
    if w_relay:     # step 0c variant: an UNFLAGGED relay layer r{w} (the divnorm block divides every synaptic current onto the
        # flagged word pools by the pool-mean divisor, so inhibition onto them is ~1/D; the relay is read instead)
        regions += [BrainRegion(name="r%d" % w, n_neurons=n_word, exc_fraction=1.0, internal_density=0.0) for w in range(V)]
    cfg.brain_regions = regions
    P = []
    if w_relay:
        for w in range(V):
            P.append(RegionPathway(from_region="w%d" % w, to_region="r%d" % w, density=1.0, weight_mean=float(w_relay),
                                   weight_jitter=0.0, plastic=False))
            P.append(RegionPathway(from_region="pinh", to_region="r%d" % w, density=1.0, weight_mean=float(w_i2e),
                                   weight_jitter=0.0, plastic=False))
    for w in range(V):
        P.append(RegionPathway(from_region="w%d" % w, to_region="pinh", density=1.0, weight_mean=float(w_e2i),
                               weight_jitter=0.0, plastic=False))
        P.append(RegionPathway(from_region="pinh", to_region="w%d" % w, density=1.0, weight_mean=float(w_i2e),
                               weight_jitter=0.0, plastic=False))
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                          gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    # The global heterogeneity overwrites a region's preset with RS-centred jitter (finding 2026-10-20 RS-centred
    # heterogeneity): re-apply the FS preset (sim/enums.py IZH2007_FS_CORTICAL_INTERNEURON, +-10% jitter, seeded) to the
    # `pinh` slice ONLY, and copy the word pools' per-neuron parameters from the composer's own score bridge (`ref_sb`), so the pools are the composer's.
    import numpy as np
    from sim.backend import to_host, from_host
    from research.runners._phaseC_S5_divnorm_derisk import region_idx
    fs = dict(cp_izh_C=20.0, cp_izh_k=1.0, cp_izh_vr=-55.0, cp_izh_vt=-40.0, cp_izh_vpeak=25.0, cp_izh_a=0.2,
              cp_izh_b=-2.0, cp_izh_c_reset=-45.0, cp_izh_d_increment=25.0)
    rng = np.random.RandomState(int(seed) + 7919)
    ii = region_idx(sb, "pinh")
    for attr, val in fs.items():
        arr = np.array(to_host(getattr(sb, attr)), dtype=np.float32)
        arr[ii] = (val * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=len(ii)))).astype(np.float32) if val > 0 else \
            (val * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=len(ii)))).astype(np.float32)
        if ref_sb is not None:                     # word pools: the composer's own per-neuron parameters, exactly
            nw = int(V) * int(n_word)
            arr[:nw] = np.array(to_host(getattr(ref_sb, attr)), dtype=np.float32)[:nw]
        setattr(sb, attr, from_host(arr))
    return sb


def onbridge_inhib_drive(score_sb, V, scores, input_gain, settle=20, ff_gain=0.0, trace=None, read="w"):
    """`onbridge_divnorm_drive` on the inhibition bridge (same drive, same settle, same read), plus the inhibitory
    population's spike count (ENGAGEMENT). Returns (acc_fired[V], inh_spikes). `ff_gain` > 0 (step 0c): the SAME external
    drive that goes into the word pools also goes into `pinh`, as ff_gain x the summed patient-slice drive (a fixed
    per-bridge gain; no normalization, no argmax). `trace` (dict) is filled with the first-spike step of every fired pool
    and of pinh, and the max |v| and max inhibitory conductance seen (physical validity)."""
    import numpy as np
    from sim.backend import to_host, from_host
    from research.runners._phaseC_S5_divnorm_derisk import _reset_score_bridge, region_idx
    _reset_score_bridge(score_sb)
    _pi = region_idx(score_sb, "pinh")             # the FS cells start at REST (vr), not at the spike-reset potential:
    score_sb.cp_membrane_potential_v[_pi] = score_sb.cp_izh_vr[_pi]      # c_reset > vt would fire them with no input
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    cur = np.zeros(score_sb.core_config.num_neurons, dtype=np.float64)
    for w in range(V):
        drv = float(input_gain * s[w])
        if drv > 0:
            cur[region_idx(score_sb, "w%d" % w)] = drv
    acc = np.zeros(V)
    inh_idx = region_idx(score_sb, "pinh")
    inh_spikes = 0.0
    if ff_gain:
        cur[inh_idx] = float(ff_gain) * float(sum(input_gain * s[w] for w in range(V)))
    first = np.full(V, -1, dtype=int)
    first_inh, max_v, max_gi = -1, 0.0, 0.0
    cur = from_host(cur)
    for t in range(settle):
        score_sb.cp_external_input_current[:] = cur
        score_sb._run_one_simulation_step()
        fir = np.asarray(to_host(score_sb.cp_firing_states)).astype(float)
        for w in range(V):
            f = fir[region_idx(score_sb, "%s%d" % (read, w))].mean()
            acc[w] += f
            if f > 0 and first[w] < 0:
                first[w] = t
        ni = float(fir[inh_idx].sum())
        inh_spikes += ni
        if ni > 0 and first_inh < 0:
            first_inh = t
        if trace is not None:
            max_v = max(max_v, float(np.abs(np.asarray(to_host(score_sb.cp_membrane_potential_v))).max()))
            max_gi = max(max_gi, float(np.asarray(to_host(score_sb.cp_conductance_g_i)).max()))
    score_sb.cp_external_input_current[:] = 0.0
    if trace is not None:
        trace.update(first_spike_step={int(w): int(first[w]) for w in range(V) if first[w] >= 0},
                     first_spike_step_pinh=first_inh, max_abs_v=max_v, max_g_i=max_gi,
                     ff_current_pA=float(ff_gain) * float(sum(input_gain * s[w] for w in range(V))) if ff_gain else 0.0)
    return acc, inh_spikes


def _inhib_arm_specs(mode, knobs):
    """{arm: dict(w_i2e, ff_gain, stable)}. 0b arms (on/lesion/on@g) are the feedback-only build on the default-integration
    bridge; 0c arms (stable_off/ff/ff_lesion/ff@g) are ALL on the stable-integration bridge so they differ only in the FF
    inhibition (stable_off = the population built, no FF drive, I->E zero)."""
    spec = {}
    if mode in ("on", "all"):
        spec["on"] = dict(w_i2e=knobs["w_i2e"], ff_gain=0.0, stable=False)
    if mode in ("lesion", "all"):
        spec["lesion"] = dict(w_i2e=0.0, ff_gain=0.0, stable=False)
    if mode in ("on", "all"):
        for g in knobs.get("sweep_w_i2e") or []:
            spec["on@%g" % g] = dict(w_i2e=g, ff_gain=0.0, stable=False)
    if mode == "ff":
        spec["stable_off"] = dict(w_i2e=0.0, ff_gain=0.0, stable=True)
        spec["ff"] = dict(w_i2e=knobs["ff_w_i2e"], ff_gain=knobs["ff_gain"], stable=True)
        for g in knobs.get("sweep_ff_gain") or []:
            spec["ff@%g" % g] = dict(w_i2e=knobs["ff_w_i2e"], ff_gain=g, stable=True)
    if mode in ("ff", "ff_lesion"):
        spec["ff_lesion"] = dict(w_i2e=0.0, ff_gain=knobs["ff_gain"], stable=True)
    if mode == "ffx":
        spec["ffx_off"] = dict(w_i2e=0.0, ff_gain=0.0, stable=True, ext=True)
        spec["ffx"] = dict(w_i2e=knobs["ffx_w_i2e"], ff_gain=knobs["ff_gain"], stable=True, ext=True)
        for g in knobs.get("sweep_ffx_w_i2e") or []:
            spec["ffx@%g" % g] = dict(w_i2e=g, ff_gain=knobs["ff_gain"], stable=True, ext=True)
    if mode in ("ffx", "ffx_lesion"):
        spec["ffx_lesion"] = dict(w_i2e=0.0, ff_gain=knobs["ff_gain"], stable=True, ext=True)
    if mode == "ff":      # the relay variant (inhibition delivered AFTER the divnorm stage, onto unflagged relay pools)
        spec["ffr"] = dict(w_i2e=knobs["relay_w_i2e"], ff_gain=knobs["ff_gain"], stable=True, relay=knobs["w_relay"])
        spec["ffr_lesion"] = dict(w_i2e=0.0, ff_gain=knobs["ff_gain"], stable=True, relay=knobs["w_relay"])
    return spec


def _inhib_bridges(buf, mode, knobs):
    """{arm: (bridge, ff_gain)} for the requested mode."""
    out = {}
    for arm, sp in _inhib_arm_specs(mode, knobs).items():
        sb = build_inhib_score_bridge(buf.seed, int(buf.V), buf.sequencer_sigma, buf.sequencer_gain,
                                      n_inh=knobs["n_inh"], w_e2i=knobs["w_e2i"], w_i2e=sp["w_i2e"],
                                      izh_type=knobs["izh_type"], ref_sb=buf._seq_score, stable=sp["stable"], w_relay=sp.get("relay"), external_only=sp.get("ext", False))
        out[arm] = (sb, sp["ff_gain"], "r" if sp.get("relay") else "w")
    return out


def score_step0b(blocks):
    import numpy as np
    """Per arm: blocks with exactly one pool firing that equals the decoded patient, and the two registered STOP lines."""
    res = {"arms": {}, "stop_lines": list(STEP0B_STOP_LINES), "stops": []}
    arms = sorted({a for b in blocks for a in (b.get("inhib") or {})})
    for arm in arms:
        rows = [b["inhib"][arm] for b in blocks if arm in (b.get("inhib") or {}) and "fired_pools" in b["inhib"][arm]]
        res["arms"][arm] = {"n_blocks": len(rows),
                            "one_correct": sum(1 for r in rows if r["fires_decoded_patient"]),
                            "multi": sum(1 for r in rows if r["n_fired"] > 1),
                            "zero": sum(1 for r in rows if r["n_fired"] == 0),
                            "wrong": sum(1 for r in rows if r["n_fired"] == 1 and not r["fires_decoded_patient"]),
                            "inh_spikes_total": float(sum(r["inh_spikes"] for r in rows))}
    for on, les in (("on", "lesion"), ("ff", "ff_lesion"), ("ffr", "ff" + "r_lesion"), ("ffx", "ffx_lesion")):
        tag = {"on": "S0B", "ff": "S0C", "ffr": "S0C_relay", "ffx": "S0D"}[on]
        if on in res["arms"]:
            a = res["arms"][on]
            if a["multi"] or a["wrong"] or a["zero"]:
                res["stops"].append(tag + "_on_not_one_correct_pool")
        if on in res["arms"] and les in res["arms"]:
            same = all((b["inhib"][on].get("fired_pools") == b["inhib"][les].get("fired_pools"))
                       for b in blocks if on in (b.get("inhib") or {}) and les in (b.get("inhib") or {}))
            res["lesion_equals_on" if on == "on" else "%s_equals_%s" % (les, on)] = bool(same)
            from tools.lab import attributable_to      # whose is the drop in fired pools: the I->E synapses, or the shared drive?
            mean_n = lambda arm: float(np.mean([b["inhib"][arm]["n_fired"] for b in blocks if arm in (b.get("inhib") or {})]))
            res["fired_pools_drop_attributable_to_" + on] = attributable_to("patient inhib %s (mean pools fired)" % on,
                                                                           mean_n(les), mean_n(on))
            if same:
                res["stops"].append(tag + "_lesion_equals_on")
    for arm in [a for a in res["arms"] if a.startswith("ffx")]:     # 0d: physiological conductance + FS-first + drive margin
        rr = [b["inhib"][arm] for b in blocks if arm in (b.get("inhib") or {})]
        res["arms"][arm].update(max_abs_v=max(r["max_abs_v"] for r in rr), max_g_i=max(r["max_g_i"] for r in rr),
                                g_i_unphysiological_over_50nS=bool(max(r["max_g_i"] for r in rr) > 50.0),
                                pinh_first_in_all_blocks=all(
                                    r["first_spike_step_pinh"] >= 0 and all(r["first_spike_step_pinh"] <= v
                                                                           for v in r["first_spike_step_pools"].values()) for r in rr))
    ff = [b["inhib"]["ff"] for b in blocks if "ff" in (b.get("inhib") or {})]
    if ff:      # physical validity + FS-first
        res["ff_max_abs_v"] = max(r["max_abs_v"] for r in ff)
        res["ff_max_g_i"] = max(r["max_g_i"] for r in ff)
        res["ff_pinh_first_in_all_blocks"] = all(
            r["first_spike_step_pinh"] >= 0 and all(r["first_spike_step_pinh"] <= v for v in r["first_spike_step_pools"].values())
            for r in ff)
    return res


def read_c(buf, rows):
    import numpy as np
    from sim.backend import to_host
    from research.runners.one_brain_composer import _seq_imports
    from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
    from research.runners._phaseC_task1_cleanup_probe import block_cleanup_scores_full
    fns = _seq_imports()
    out = {}
    try:
        K, V = len(buf.kb), int(buf.V)
        iA, iX, iP = (buf.main_roles.index(r) for r in ("agent", "action", "patient"))
        reserve = np.array([bool(buf._silent_reserve(j)) for j in range(V)])
        nonres = ~reserve
        out.update(K=K, V=V, patient_row_index=iP, n_reserve_slots=int(reserve.sum()),
                   input_gain=float(buf.sequencer_input_gain), score_bridge_built=buf._seq_score is not None)
        inhib_mode = patient_inhib_env()
        inhib_knobs = patient_inhib_knobs()
        inhib_sbs = _inhib_bridges(buf, inhib_mode, inhib_knobs) if inhib_mode != "off" else {}
        if inhib_sbs:
            out["patient_inhib"] = dict(mode=inhib_mode, knobs=inhib_knobs)
        blocks = []
        for b in range(K):
            blk = {"block": b, "decoded_patient": rows[b].get("patient") if b < len(rows) else None}
            try:
                ag, ax = fns["block_cleanup_scores"](buf, b)          # the production read (agent + action slices)
                mem = np.asarray(to_host(buf.b.cp_membrane_potential_v)).astype(float)
                pslice = np.maximum(mem[buf.c_base + iP * V: buf.c_base + (iP + 1) * V], 0.0)   # read right after it
                _drive, acc = onbridge_divnorm_drive(buf._seq_score, V, pslice, buf.sequencer_input_gain)
                fired = [buf.words[j] for j in range(V) if acc[j] > 0]
                order = [int(j) for j in np.argsort(-pslice)[:3]]
                blk.update(fired_pools=fired, n_fired=len(fired), exactly_one=len(fired) == 1,
                           fires_decoded_patient=fired == [blk["decoded_patient"]],
                           pool_spikes={buf.words[j]: float(acc[j]) for j in range(V) if acc[j] > 0},
                           patient_slice_top=[[buf.words[j], float(pslice[j])] for j in order],
                           patient_slice_nz={buf.words[j]: float(pslice[j]) for j in range(V) if pslice[j] > 0})
                if inhib_sbs:                                       # step 0b only; absent when the env flag is off
                    blk["inhib"] = {}
                    for arm, (isb, ffg, rd) in inhib_sbs.items():
                        tr = {}
                        iacc, ispk = onbridge_inhib_drive(isb, V, pslice, buf.sequencer_input_gain,
                                                          settle=inhib_knobs["settle"], ff_gain=ffg, trace=tr, read=rd)
                        ifired = [buf.words[j] for j in range(V) if iacc[j] > 0]
                        blk["inhib"][arm] = dict(fired_pools=ifired, n_fired=len(ifired),
                                                 fires_decoded_patient=ifired == [blk["decoded_patient"]],
                                                 pool_spikes={buf.words[j]: float(iacc[j]) for j in range(V) if iacc[j] > 0},
                                                 inh_spikes=ispk, ff_gain=ffg,
                                                 first_spike_step_pools={buf.words[j]: tr["first_spike_step"][j] for j in range(V)
                                                                         if j in tr["first_spike_step"]},
                                                 first_spike_step_pinh=tr["first_spike_step_pinh"],
                                                 max_abs_v=tr["max_abs_v"], max_g_i=tr["max_g_i"],
                                                 ff_current_pA=tr["ff_current_pA"])
                probe = block_cleanup_scores_full(buf, b)           # the cross-check (wires every slot, reserves too)
                cmp = {}
                for role, prod, ri in (("agent", ag, iA), ("action", ax, iX), ("patient", pslice, iP)):
                    pr = np.maximum(np.asarray(probe[ri], dtype=float), 0.0)
                    pr[reserve] = 0.0                                  # the P10 treatment
                    prod = np.asarray(prod, dtype=float)
                    d = np.abs(prod[nonres] - pr[nonres])
                    cmp[role] = {"equal": bool(np.array_equal(prod[nonres], pr[nonres])),
                                 "max_abs_diff": float(d.max()) if d.size else 0.0,
                                 "n_slots_differ": int((d > 0).sum())}
                blk["probe_vs_production"] = cmp
            except Exception as e:
                blk.update(_err(e))
            blocks.append(blk)
        out["blocks"] = blocks
        if inhib_sbs:
            out["step0b"] = score_step0b(blocks)
        vp = []
        for blk in blocks:
            for w in blk.get("fired_pools") or []:
                if w not in vp:
                    vp.append(w)
        dec_p = []
        for r in rows:
            if r.get("patient") not in dec_p:
                dec_p.append(r.get("patient"))
        out.update(vp=vp, decoded_patients=dec_p, vp_equals_decoded_patients=set(vp) == set(dec_p),
                   pools_no_block_decodes=[w for w in vp if w not in dec_p],
                   decoded_patients_without_pool=[p for p in dec_p if p not in vp])
    except Exception as e:
        out.update(_err(e))
    return out


def _distinct(kb, agent, action):
    out = []
    for fact, _h in kb:
        if fact.get("agent") == agent and fact.get("action") == action:
            p = fact.get("patient")
            if isinstance(p, str) and p not in out:
                out.append(p)
    return out


def read_f(buf, ltm, rows, table):
    out = {"cues": {}, "ltm_type": type(ltm).__name__}
    for a, x in premise_cues(table):
        d = {"cue": [a, x]}
        try:
            dec = SG.rows_under(rows, a, x)
            d["decoded_rows"] = dec
            d["decoded_patients"] = [r["patient"] for r in dec]
            d["kb_patients"] = _distinct(buf.kb, a, x)                  # record-only (the list beside the store)
            d["ltm"] = SG.ltm_readback(ltm, a, x) if ltm is not None else None
        except Exception as e:
            d.update(_err(e))
        audit = {"buffer": len(d.get("kb_patients") or [])}
        try:                                    # record-only: an audit failure never voids the premise read
            if ltm is not None:
                audit["ltm_routed"] = len(_distinct(ltm.shard_for(a).kb, a, x))
                allp = []
                for sh in getattr(ltm, "shards", None) or []:
                    for p in _distinct(sh.kb, a, x):
                        if p not in allp:
                            allp.append(p)
                audit["ltm_all"] = len(allp)
                audit["ltm_all_patients"] = allp
        except Exception as e:
            audit["error"] = "%s: %s" % (type(e).__name__, e)
        d["audit"] = audit
        out["cues"][_key((a, x))] = d
    return out


# ═══════════════════════════════════════════ THE SCORER (GO rule + STOP lines) ══════════════════════════════════════
def shard_path(root, seed, arm):
    return SG.shard_path(root, seed, GROUP, arm)


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": repr(e)}


def shard_validity(shard, seed, arm):
    """Why a step-0 shard cannot be read (empty = readable). Packet 2.10 step 0: a shard that records corpus_present
    false is UNDEFINED for every check in it and is re-run on a corpus lane, never a GO or STOP input; a shard with NO
    `corpus_present` key is UNDEFINED too (2.9 rule (i): never `arm_validity`'s default); section 5: `sources` must be
    {tiny-demo +LTM} and `process.ltm_bundle` wikidata_100k, else the lane is not used."""
    if shard is None:
        return ["shard missing"]
    if "_load_error" in shard or "responses" not in shard:
        return ["shard unreadable"]
    why = []
    if int(shard.get("seed", -1)) in SG.GATE_SEEDS:
        why.append("a gate seed: voids the step")
    if int(shard.get("seed", -1)) != int(seed) or shard.get("arm") != arm:
        why.append("shard is seed %s arm %s, expected seed %s arm %s" % (shard.get("seed"), shard.get("arm"), seed, arm))
    if "corpus_present" not in shard:
        why.append("no corpus_present key")
    elif shard.get("corpus_present") is not True:
        why.append("corpus_present false: re-run on a corpus lane")
    if set(shard.get("sources") or []) != {SG.EXPECTED_SOURCE}:
        why.append("sources %s, expected {%s}" % (shard.get("sources"), SG.EXPECTED_SOURCE))
    if (shard.get("process") or {}).get("ltm_bundle") != EXPECTED_LTM_BUNDLE:
        why.append("ltm_bundle %r, expected %s" % ((shard.get("process") or {}).get("ltm_bundle"), EXPECTED_LTM_BUNDLE))
    if not shard.get("pythonhashseed_ok"):
        why.append("PYTHONHASHSEED not 0")
    want = [(l, m) for l, m, _r in SG.turns_for(GROUP, seed, arm)]
    got = [(t.get("label"), t.get("message")) for t in shard.get("turns") or []]
    if got != want:
        why.append("turn table differs from the entry's table")
    errored = sorted(l for l, r in (shard.get("responses") or {}).items() if not isinstance(r, dict) or "_error" in r)
    if errored:
        why.append("errored turn(s) %s" % ",".join(errored))
    c2 = shard.get("c2_hops")
    if not isinstance(c2, dict):
        why.append("no c2_hops record")
    return why


def _fact_in(fact, triples):
    return any(list(t) == list(fact) for t in triples or [])


def _canon(message):
    return SG._canonical_teach(message)


def _stop(lines, key, detail):
    lines.append({"line": key, "detail": detail})


STOP_TEXT = {
    "S1_missed_true_match": "a missed true match: post the counts; the circuit is banked and the named next carrier is "
                            "the cs-race engram race read for co-latching (two latches ON = conflict)",
    "S2_patient_pool": "a patient pool missing, wrong or not unique: the patient identity the unit would count is not "
                       "on the substrate at this fidelity, and so neither is V'_P",
    "S3_probe_differs": "the probe's clipped agent or action slice differs from block_cleanup_scores on a non-reserve "
                        "slot, or its P10-treated patient slice from the membrane read of (c): post the difference "
                        "(the V'_P read stays the membrane read)",
    "S4_host_routed": "a probe cue recorded host_routed: post the candidate ids and the decoded rows; Claude rules on "
                      "whether A21 carries the affected bars",
    "S5_or_mismatch": "an or_matches_return false: post the call; that teach's branch is UNDEFINED",
    "S6_teach_branch": "a new-table teach refused or taking another branch than 2.8's: post the recorder's inputs, terms "
                       "and return; Claude rules on the word",
    "S7_known_binding": "t_frog_eat_apple with known_binding false: the hop-2 table then rides on the margin; post it",
    "S8_rewrite_or_rows": "a rewrite on an ambiguity or re-tell teach, or a cue holding one row where two are expected: "
                          "D2 took the fact; Claude rules on the named fallback (a composer.store fixture)",
    "S9_premise": "a premise-map row failing on the bound store or the LTM (a missing premise or an extra patient "
                  "under a hop cue; a (frog, chase) fact in the teach log or builtin_facts): the probes that row "
                  "feeds are predicted UNDEFINED; Claude rules on the table words",
    "S10_overwrite": "an overwrite fixture not engaging: its half of B13 is predicted UNDEFINED",
    "S11_xtier": "the infer_xtier fixture not landing",
    "S12_not_inert": "reads that are not inert: post the differing turns; (d) is then re-read in a worker that makes "
                     "none of the other reads",
}


def score_table_shard(shard, seed, table):
    """GO clauses and STOP lines of ONE full step-0 shard (seed, table). Clauses are True / False, or None when the
    clause does not apply to this table."""
    c2 = shard.get("c2_hops") or {}
    reads = c2.get("reads") or {}
    resp = shard.get("responses") or {}
    turns = shard.get("turns") or []
    go, stops, posts, faults = {}, [], [], []
    if c2.get("mode") != "full":
        faults.append("c2_hops mode %r, expected full" % c2.get("mode"))
    if reads.get("error"):
        faults.append("reads raised: %s" % reads["error"])
    rows = reads.get("decoded_rows") or []
    rows_read = not reads.get("error") and isinstance(reads.get("decoded_rows"), list)
    if not rows_read:
        faults.append("decoded_rows not read: the row checks of (e) are UNDEFINED on this shard")
    dec_triples = [[r.get("agent"), r.get("action"), r.get("patient")] for r in rows]

    # (b) every true match drives its m{b} over match_thresh and no non-matching block does
    b = reads.get("b") or {}
    if b.get("error") or not b.get("cues"):
        faults.append("read (b) missing or raised: %s" % b.get("error"))
        go["match_pools"] = None
    else:
        ok = True
        for k, c in b["cues"].items():
            if c.get("error"):
                faults.append("read (b) %s raised: %s" % (k, c["error"]))
                ok = None if ok is not False else ok
                continue
            if c.get("missed_true_matches"):
                ok = False
                _stop(stops, "S1_missed_true_match", {"cue": k, "true_matches": c["true_matches"],
                                                      "fired": c["fired"], "m_rates": c.get("m_rates"),
                                                      "cue_in_reduced_vocab": c.get("cue_in_reduced_vocab")})
            if c.get("fired_non_matching"):
                ok = False
                posts.append({"what": "a non-matching block fired its m{b} (GO clause fails; no STOP line names it)",
                              "cue": k, "fired_non_matching": c["fired_non_matching"], "m_rates": c.get("m_rates")})
        go["match_pools"] = ok

    # (c) one patient pool per block, its decoded patient's; V'_P == decoded patients; re-told same / ambiguous different
    cc = reads.get("c") or {}
    if cc.get("error") or not cc.get("blocks"):
        faults.append("read (c) missing or raised: %s" % cc.get("error"))
        go["patient_pools"] = go["vp_equals_decoded"] = go["probe_equals_production"] = None
    else:
        bad, probe_bad = [], []
        for blk in cc["blocks"]:
            if blk.get("error"):
                faults.append("read (c) block %s raised: %s" % (blk.get("block"), blk["error"]))
                continue
            if not blk.get("fires_decoded_patient"):
                bad.append({"block": blk["block"], "decoded_patient": blk.get("decoded_patient"),
                            "fired_pools": blk.get("fired_pools"), "patient_slice_top": blk.get("patient_slice_top")})
            for role, cmp in (blk.get("probe_vs_production") or {}).items():
                if not cmp.get("equal"):
                    probe_bad.append({"block": blk["block"], "role": role, "max_abs_diff": cmp.get("max_abs_diff"),
                                      "n_slots_differ": cmp.get("n_slots_differ")})
        stray = cc.get("pools_no_block_decodes") or []
        go["patient_pools"] = not bad and not stray
        if bad or stray:
            _stop(stops, "S2_patient_pool", {"blocks": bad, "pools_no_block_decodes": stray})
        go["vp_equals_decoded"] = bool(cc.get("vp_equals_decoded_patients"))
        go["probe_equals_production"] = not probe_bad
        if probe_bad:
            _stop(stops, "S3_probe_differs", probe_bad)
        by_block = {blk["block"]: blk.get("fired_pools") for blk in cc["blocks"] if "block" in blk}
        pool_rel = True
        for cue in RETOLD_CUES.get(table, ()):
            ids = [r["block"] for r in SG.rows_under(rows, *cue)]
            pools = [tuple(by_block.get(i) or ()) for i in ids]
            if not (len(pools) >= 2 and len(set(pools)) == 1 and pools[0]):
                pool_rel = False
                posts.append({"what": "a re-told cue's blocks do not fire the same pool", "cue": _key(cue),
                              "blocks": ids, "pools": pools})
        for cue in AMBIGUOUS_CUES.get(table, ()):
            ids = [r["block"] for r in SG.rows_under(rows, *cue)]
            pools = [tuple(by_block.get(i) or ()) for i in ids]
            if not (len(pools) >= 2 and len(set(pools)) == len(pools) and all(pools)):
                pool_rel = False
                posts.append({"what": "an ambiguous cue's blocks do not fire different pools", "cue": _key(cue),
                              "blocks": ids, "pools": pools})
        go["retold_same_ambiguous_different"] = pool_rel if (table in RETOLD_CUES or table in AMBIGUOUS_CUES) else None

    # (e) every teach of 2.8's table took its expected branch (D4 recorder), and its fact is in the decoded rows
    branches = c2.get("d4_branches") or {}
    br_ok = True
    for t in turns:
        if t.get("role") != "teach" or t.get("message") not in EXPECTED_BRANCH:
            continue
        lab, want = t["label"], EXPECTED_BRANCH[t["message"]]
        got = (branches.get(lab) or {}).get("branch")
        r = resp.get(lab) or {}
        refused = bool(r.get("not_understood")) or bool((r.get("comprehension") or {}).get("abstained"))
        fact = _canon(t["message"])
        in_rows = _fact_in(fact, dec_triples)
        if got != want or refused or not in_rows:
            br_ok = False
            detail = {"label": lab, "message": t["message"], "expected": want, "branch": got, "refused": refused,
                      "fact_in_decoded_rows": in_rows, "call": (branches.get(lab) or {}).get("call")}
            if table in NEW_TABLES and (got != want or refused):
                _stop(stops, "S6_teach_branch", detail)
            else:
                posts.append(dict(detail, what="a teach of 2.8's table off its expected branch or not decoded"))
    go["teach_branches"] = br_ok

    # (e) the recorder's or_matches_return on every call
    calls = (c2.get("d4_recorder") or {}).get("calls")
    if calls is None:
        faults.append("no D4 recorder record")
        go["or_matches_return"] = None
    else:
        badc = [c for c in calls if not c.get("or_matches_return")]
        go["or_matches_return"] = not badc
        if badc:
            _stop(stops, "S5_or_mismatch", badc)
        if not calls:
            posts.append({"what": "the D4 recorder saw no competent() call on this shard"})

    # (e) t_frog_eat_apple known_binding (amb2)
    if table == "infer_amb2":
        comp = (resp.get("t_frog_eat_apple") or {}).get("comprehension") or {}
        kb_ok = comp.get("known_binding") is True
        go["apple_known_binding"] = kb_ok
        if not kb_ok:
            _stop(stops, "S7_known_binding", {"comprehension": comp})
    else:
        go["apple_known_binding"] = None

    # (e) no rewrite on an ambiguity / re-tell teach; ambiguous cues hold 2 rows, 2 patients; re-told 2 rows, 1 patient
    if table in ("infer_amb1", "infer_amb2", "infer_dup"):
        rw = [t["label"] for t in turns if t.get("role") == "teach"
              and ((resp.get(t["label"]) or {}).get("reconsolidation") or {}).get("action") == "rewrite"]
        rows_ok = True if rows_read else None
        for cue in (AMBIGUOUS_CUES.get(table, ()) if rows_read else ()):
            under = SG.rows_under(rows, *cue)
            if not (len(under) == 2 and len({r["patient"] for r in under}) == 2):
                rows_ok = False
                _stop(stops, "S8_rewrite_or_rows", {"cue": _key(cue), "rows": under, "expected": "2 rows, 2 patients"})
        for cue in (RETOLD_CUES.get(table, ()) if rows_read else ()):
            under = SG.rows_under(rows, *cue)
            if not (len(under) == 2 and len({r["patient"] for r in under}) == 1):
                rows_ok = False
                _stop(stops, "S8_rewrite_or_rows", {"cue": _key(cue), "rows": under, "expected": "2 rows, 1 patient"})
                if len(under) < 2:
                    posts.append({"what": "a re-tell did not append: B11 is predicted UNDEFINED and the step-1 "
                                          "circuit test carries the duplicate case", "cue": _key(cue)})
        if rw:
            _stop(stops, "S8_rewrite_or_rows", {"rewrite_on": rw,
                                                "reconsolidation": {l: resp[l].get("reconsolidation") for l in rw}})
        go["no_rewrite_rows_as_expected"] = False if rw else rows_ok
    else:
        go["no_rewrite_rows_as_expected"] = None

    # (f) every row of 2.9's premise map holds exactly on the bound store and the LTM (and on its teach-log sources)
    f = (reads.get("f") or {}).get("cues") or {}
    after = c2.get("stored_facts_after_teach") or []
    builtin = shard.get("builtin_facts") or []
    acquired = SG.acquired_teaches(shard)
    msg_of = {t["label"]: t["message"] for t in turns}
    fixture = c2.get("ltm_fixture") or {}
    prem_ok, prem_fault = True, False
    for row in PREMISE_MAP[table]:
        fails = []
        for src in row["sources"]:
            if src.startswith("builtin:"):
                fact = src.split(":", 1)[1].split(",")
                if not _fact_in(fact, builtin):
                    fails.append("built-in %s not in builtin_facts" % fact)
            elif src.startswith("fixture:"):
                # the fixture landing (its type, write and both read-backs) is S11's line, not a premise failure; the
                # LTM half of this row is still read below, from (f)
                continue
            else:
                fact = _canon(msg_of.get(src))
                if not acquired.get(src):
                    fails.append("%s not acquired (teach_acquired)" % src)
                if not _fact_in(fact, after):
                    fails.append("%s's fact %s not in stored_facts_after_teach" % (src, fact))
        for cue, (want_buf, want_ltm) in row["cues"].items():
            d = f.get(_key(cue))
            if d is None or d.get("error"):
                prem_fault = True
                faults.append("read (f) %s missing or raised: %s" % (_key(cue), (d or {}).get("error")))
                continue
            got_buf = d.get("decoded_patients") or []
            if set(got_buf) != set(want_buf):
                fails.append("bound store %s holds %s, expected exactly %s" % (_key(cue), sorted(map(str, got_buf)),
                                                                                sorted(want_buf)))
            lt = d.get("ltm") or {}
            if "all_candidate_patients_error" in lt:
                prem_fault = True
                faults.append("LTM read %s raised: %s" % (_key(cue), lt["all_candidate_patients_error"]))
                continue
            got_ltm = lt.get("all_candidate_patients") or []
            if set(got_ltm) != set(want_ltm):
                fails.append("LTM %s holds %s, expected exactly %s" % (_key(cue), sorted(map(str, got_ltm)),
                                                                       sorted(want_ltm)))
        for cue in row["absence"]:
            held = [t for t in list(after) + list(builtin) if list(t[:2]) == list(cue)]
            if held:
                fails.append("absence premise %s: %s in the teach log or builtin_facts" % (_key(cue), held))
            d = f.get(_key(cue)) or {}
            extra = {"decoded": d.get("decoded_patients") or [],
                     "ltm": (d.get("ltm") or {}).get("all_candidate_patients") or []}
            if extra["decoded"] or extra["ltm"]:
                posts.append({"what": "a decoded or LTM %s row that nothing taught (not a premise failure; B8 then "
                                      "predicts a moat FAIL wherever it reaches an answer)" % _key(cue),
                              "rows": extra, "probe": row["probe"]})
        if fails:
            prem_ok = False
            _stop(stops, "S9_premise", {"probe": row["probe"], "reported_only": row["reported_only"],
                                        "fails": fails})
    go["premise_map"] = None if prem_fault and prem_ok else prem_ok

    # (g) the overwrite fixture engages under (i)-(iv)
    if table in OVERWRITE_TABLES:
        eng = (shard.get("kb_scramble") or {}).get("engagement") or {}
        ok = bool(eng.get("engaged")) and eng.get("plan") == OVERWRITE_TABLES[table]
        go["overwrite_engaged"] = ok
        if not ok:
            _stop(stops, "S10_overwrite", {"plan": eng.get("plan"), "reason": eng.get("reason"),
                                           "conditions": eng.get("conditions"), "condition_iv": eng.get("condition_iv")})
    else:
        go["overwrite_engaged"] = None

    # (e)/(5) the infer_xtier fixture landed: ShardedPhasorStore, no store_exception, frog in the buffer only, fish in
    # the LTM only, on both read-backs
    if table == XTIER_TABLE:
        fx = fixture
        typ, exc = fx.get("ltm_type"), fx.get("store_exception")
        rbs = [fx.get("readback_after_write") or {}, fx.get("readback_before_first_probe") or {}]
        hh = f.get(_key(("heron", "hunt"))) or {}
        buf_p = set(hh.get("decoded_patients") or [])
        ltm_ok = all(rb.get("query_patient") == "fish" and set(rb.get("all_candidate_patients") or []) == {"fish"}
                     for rb in rbs)
        landed = typ == "ShardedPhasorStore" and not exc and bool(fx.get("written")) and ltm_ok and buf_p == {"frog"}
        go["xtier_fixture_landed"] = landed
        if not landed:
            if typ != "ShardedPhasorStore" or "SharedLTMWriteError" in str(exc or ""):
                remedy = ("the offiso env did not reach the server: post ltm_type and the two env values as read, fix "
                          "the entry's env (BRAIN_SHARED_LTM=0, BRAIN_LTM_RNG_ISOLATE=1, BRAIN_ONE_BRAIN unset) and "
                          "re-run that step-0 read; never a write through ltm.shared_store or on the shared default; "
                          "A15 and A16 stay UNDEFINED until a re-run records ShardedPhasorStore")
            else:
                remedy = "the type is right and the write or a read-back still fails: post it; Claude rules on the words"
            _stop(stops, "S11_xtier", {"ltm_type": typ, "env_as_read": fx.get("env_as_read"),
                                       "store_exception": exc, "written": fx.get("written"), "readbacks": rbs,
                                       "buffer_patients_heron_hunt": sorted(buf_p), "remedy": remedy})
    else:
        go["xtier_fixture_landed"] = None

    # ws_seed equals the arm seed
    go["ws_seed"] = c2.get("ws_seed") == int(seed)
    if not go["ws_seed"]:
        posts.append({"what": "ws_seed is not the arm seed (GO clause fails; no STOP line names it)",
                      "ws_seed": c2.get("ws_seed"), "seed": seed})

    # (a) no probe cue is host_routed
    a = reads.get("a") or {}
    if not a:
        faults.append("read (a) missing")
        go["no_host_routed"] = None
    else:
        hr = [dict(cue=k, fs_candidates=v.get("fs_candidates"), propose_path=v.get("propose_path"),
                   decoded_rows=v.get("decoded_rows_under_cue")) for k, v in a.items() if v.get("host_routed")]
        for k, v in a.items():
            if v.get("error"):
                faults.append("read (a) %s raised: %s" % (k, v["error"]))
        go["no_host_routed"] = not hr
        if hr:
            _stop(stops, "S4_host_routed", hr)
    return {"go": go, "stop_lines": stops, "posts": posts, "instrument_faults": faults}


PRIMARY_KEYS = ("answer", "abstained", "recalled_svo", "derived")
TEACH_KEYS = ("comprehension", "surprise", "reconsolidation")


def inert_check(bare, rec, full):
    """(h): the three workers give identical primary fields (`stage2_exit_gate._primary`) on every probe turn, and the
    recorder-only worker's teach turns match the bare worker's comprehension / surprise / reconsolidation fields."""
    out = {"probe_diffs": [], "teach_diffs": [], "reported_full_teach_diffs": []}
    turns = bare.get("turns") or []
    rb, rr, rf = (s.get("responses") or {} for s in (bare, rec, full))
    for t in turns:
        lab = t["label"]
        if t.get("role") == "teach":
            for k in TEACH_KEYS:
                if (rb.get(lab) or {}).get(k) != (rr.get(lab) or {}).get(k):
                    out["teach_diffs"].append({"label": lab, "field": k})
                if (rb.get(lab) or {}).get(k) != (rf.get(lab) or {}).get(k):
                    out["reported_full_teach_diffs"].append({"label": lab, "field": k})
            continue
        pb, pr, pf = (SG._primary(x.get(lab)) for x in (rb, rr, rf))
        if pb is None or pr is None or pf is None or not (pb == pr == pf):
            out["probe_diffs"].append({"label": lab, "bare": pb, "recorder": pr, "full": pf})
    out["inert"] = not out["probe_diffs"] and not out["teach_diffs"]
    return out


def score(root, seeds=SEEDS):
    """The step-0 verdict over `seeds` (packet 2.10: GO on 7/7 seeds, 7 and 1-6). Precedence: a STOP line on a readable
    shard -> STOP; else a missing, unreadable or UNDEFINED shard (corpus absent, wrong bundle, an instrument fault) ->
    UNDEFINED (re-run; never a GO or STOP input); else every GO clause on every shard and (h) -> GO; else NOT_GO (a GO
    clause failed that no STOP line names: posted for Claude). Every outcome posts `[c2-hops-0] RESULT` and STOPs."""
    seeds = [int(s) for s in seeds]
    per = {}
    undefined, stops, posts = [], [], []
    go_all = True
    for seed in seeds:
        per[seed] = {}
        for table, arm in TABLE_ARMS.items():
            path = shard_path(root, seed, arm)
            sh = _load(path)
            why = shard_validity(sh, seed, arm)
            if why:
                per[seed][table] = {"path": path, "undefined": why}
                undefined.append({"seed": seed, "table": table, "why": why})
                go_all = False
                continue
            r = score_table_shard(sh, seed, table)
            r["path"] = path
            per[seed][table] = r
            if r["instrument_faults"]:
                undefined.append({"seed": seed, "table": table, "why": r["instrument_faults"]})
                go_all = False
            for s in r["stop_lines"]:
                stops.append(dict(s, seed=seed, table=table))
            for p in r["posts"]:
                posts.append(dict(p, seed=seed, table=table))
            if any(v is not True for v in r["go"].values() if v is not None) or any(v is None for k, v in r["go"].items()
                                                                                     if k in _ALWAYS_CLAUSES):
                go_all = False
    inert = None
    if INERT_SEED in seeds:
        shs = {k: _load(shard_path(root, INERT_SEED, a)) for k, a in INERT_ARMS.items()}
        why = {k: shard_validity(shs[k], INERT_SEED, INERT_ARMS[k]) for k in shs}
        if any(why.values()):
            inert = {"undefined": why}
            undefined.append({"seed": INERT_SEED, "table": "inert (h)", "why": why})
            go_all = False
        else:
            inert = inert_check(shs["bare"], shs["recorder"], shs["full"])
            if not inert["inert"]:
                stops.append({"line": "S12_not_inert", "detail": inert, "seed": INERT_SEED, "table": "infer"})
                go_all = False
    else:
        go_all = False
    complete = sorted(seeds) == sorted(SEEDS)
    if stops:
        verdict = "STOP"
    elif undefined:
        verdict = "UNDEFINED"
    elif go_all and complete:
        verdict = "GO"
    elif go_all:
        verdict = "UNDEFINED"            # every scored seed passes, but the GO rule needs all of 7 and 1-6
    else:
        verdict = "NOT_GO"
    return {"packet": PACKET, "step": "M1 step 0 (2.10)", "root": root, "seeds": seeds, "complete_seed_set": complete,
            "verdict": verdict, "go_on_scored_seeds": bool(go_all and not stops),
            "stop_lines_fired": sorted({s["line"] for s in stops}), "stop_text": {k: STOP_TEXT[k] for k in
                                                                                 sorted({s["line"] for s in stops})},
            "stops": stops, "undefined": undefined, "posts": posts, "inert_h": inert, "per_seed": per,
            "result_line": "[c2-hops-0] RESULT %s (seeds %s%s)" % (verdict, ",".join(map(str, seeds)),
                                                                   "" if complete else "; incomplete seed set"),
            "next": "post [c2-hops-0] RESULT, STOP for Claude"}


def score_m3_d(shard, seed):
    """M3 step 0 (d) on one `infer` full shard (packet 3.8, 3.6): CLEAN (no nonce form found anywhere), REDRAW (the
    forms the fixed-seed generator must redraw before `infer_garbled` is pinned) or UNDEFINED (the shard is not valid,
    the corpus was not read, or a lexicon read raised). Report only: never a C2 GO or STOP input."""
    why = shard_validity(shard, seed, TABLE_ARMS[M3_D_TABLE])
    if why:
        return {"verdict": "UNDEFINED", "why": why}
    d = ((shard.get("c2_hops") or {}).get("reads") or {}).get("m3_d")
    if not d:
        return {"verdict": "UNDEFINED", "why": ["no m3_d read on the shard"]}
    bad = []
    if not d.get("corpus_read"):
        bad.append("the corpus was not read: %s" % d.get("corpus_error", "file absent"))
    for w in M3_GARBLED_FORMS:
        r = (d.get("per_form") or {}).get(w)
        if r is None:
            bad.append("form %s not read" % w)
        elif r.get("lexicon_error"):
            bad.append("form %s: %s" % (w, r["lexicon_error"]))
    if bad:
        return {"verdict": "UNDEFINED", "why": bad}
    redraw = {w: d["per_form"][w]["found_in"] for w in M3_GARBLED_FORMS if d["per_form"][w]["found_in"]}
    return {"verdict": "REDRAW" if redraw else "CLEAN", "redraw": redraw, "brain_vocab_size": d.get("brain_vocab_size")}


def score_m3_step0(root, seeds=SEEDS):
    """The part of M3 step 0 this runner carries, (d) only, over `seeds` (dev seeds 1-6 per 3.6, seed 7 reported
    beside them). (a), (b), (c) and (e) are not built here (packet 3.8; see the module docstring)."""
    per = {int(s): score_m3_d(_load(shard_path(root, int(s), TABLE_ARMS[M3_D_TABLE])), int(s)) for s in seeds}
    vs = {r["verdict"] for r in per.values()}
    verdict = "REDRAW" if "REDRAW" in vs else ("UNDEFINED" if "UNDEFINED" in vs or not per else "CLEAN")
    if verdict == "CLEAN" and not set(range(1, 7)) <= set(per):
        verdict = "UNDEFINED"            # 3.6 checks every dev seed 1-6
    return {"step": "M3 step 0 (d) (3.8)", "verdict": verdict, "per_seed": per,
            "not_built_here": ["(a) W/G/out-of-scope D4 decisions", "(b) word-role parser commit read",
                               "(c) carrier status read", "(e) frame_conforming exclusion"]}


# the GO clauses every table carries (a None there is an instrument fault, never a pass)
_ALWAYS_CLAUSES = ("match_pools", "patient_pools", "vp_equals_decoded", "probe_equals_production", "teach_branches",
                   "or_matches_return", "premise_map", "ws_seed", "no_host_routed")


# ═════════════════════════════════════════════════ POOL LINES ═══════════════════════════════════════════════════════
def pool_arms(seed):
    """The entries one seed runs: the five tables, plus the two inert-check workers on seed 7."""
    arms = list(TABLE_ARMS.values())
    if int(seed) == INERT_SEED:
        arms += [INERT_ARMS["recorder"], INERT_ARMS["bare"]]
    return arms


def pool_lines(sha="<sha>", seeds=SEEDS, mem_gb=MEM_GB, memcap_gb=MEMCAP_GB, max_age_h=MAX_AGE_H, out_root=OUT_ROOT):
    """One (command, reason) per (table, seed), seed 7 first then 1-6, the Stage-2 pool-line prefix byte for byte
    (stage2_exit_gate.pool_job_lines) with this step's module args and outputs; dev seeds only (no --allow-gate-seed)."""
    out = []
    max_age_h = min(int(max_age_h), 240)
    for seed in seeds:
        if int(seed) in SG.GATE_SEEDS:
            raise ValueError("step 0 never runs a gate seed: %d" % seed)
        for arm in pool_arms(seed):
            p = shard_path(out_root, seed, arm)
            table = (SG.arm_c2(GROUP, arm) or {}).get("table")
            mode = (SG.arm_c2(GROUP, arm) or {}).get("step0")
            cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && "
                   "bash tools/memcap.sh %d -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                   "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                   "research.runners.stage2_exit_gate --shard --seed %d --group %s --arm %s --out %s"
                   % (sha, mem_gb, os.path.dirname(p), memcap_gb, seed, GROUP, arm, p))
            why = ("C2 M1 step 0 (%s 2.10, round 9, on main via 26dcdea48): dev seed %d table %s arm %s (%s); "
                   "read-only, no production edit, never counted; scored by research/runners/c2_hops_step0.py --score; "
                   "record: no step-0 read of the four new C2 tables or of the match pools, patient pools and premise "
                   "map exists (research/findings/2026-10-26-c2-kb-scramble-host-ownership.md read the infer table's "
                   "conclusion only); mem_gb=%d max_age_h=%d"
                   % (PACKET, seed, table, arm, "inert-read check (h), %s worker" % mode if mode != "full"
                      else ("reads (a)-(g) and M3 step-0 (d)" if table == M3_D_TABLE else "reads (a)-(g)"),
                      mem_gb, max_age_h))
            out.append((cmd, why))
    return out


# ═══════════════════════════════════════════════════ CLI ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", default=None, metavar="ROOT", help="score the step-0 shards under ROOT")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default 7,1,2,3,4,5,6)")
    ap.add_argument("--out", default=None, help="write the score JSON here")
    ap.add_argument("--emit-pool-lines", default=None, metavar="SHA", help="print the step-0 pool lines (TSV)")
    ap.add_argument("--mem-gb", type=int, default=MEM_GB)
    ap.add_argument("--patient-inhib", default="off", choices=PATIENT_INHIB_MODES,
                    help="step 0b: patient-pool feedback inhibition for the seed-7 worker of --smoke-0b (off = the current "
                         "step 0; on / lesion; all = both arms from the same cleanup rows)")
    ap.add_argument("--smoke-0b", default=None, metavar="OUT", help="run one step-0 worker at dev seed 7, write the shard")
    ap.add_argument("--inhib-n", type=int, default=None)
    ap.add_argument("--inhib-w-e2i", type=float, default=None)
    ap.add_argument("--inhib-w-i2e", type=float, default=None)
    ap.add_argument("--inhib-ffx-w-i2e", type=float, default=None)
    ap.add_argument("--inhib-ffx-sweep", default=None, help="comma list of I->E gains for step 0d (the ONE allowed sweep, <=4)")
    ap.add_argument("--inhib-ff-gain", type=float, default=None)
    ap.add_argument("--inhib-ff-w-i2e", type=float, default=None)
    ap.add_argument("--inhib-ff-sweep", default=None, help="comma list of extra FF gains (the ONE allowed 0c sweep, <=4)")
    ap.add_argument("--inhib-sweep", default=None, help="comma list of extra I->E gains (the ONE allowed sweep, <=4)")
    a = ap.parse_args()
    if a.smoke_0b:
        import subprocess
        env = dict(os.environ, SIM_BACKEND=os.environ.get("SIM_BACKEND", "numpy"), **{PATIENT_INHIB_ENV: a.patient_inhib})
        for k, v in (("N_INH", a.inhib_n), ("W_E2I", a.inhib_w_e2i), ("W_I2E", a.inhib_w_i2e), ("SWEEP", a.inhib_sweep),
                     ("FFX_W_I2E", a.inhib_ffx_w_i2e), ("FFX_SWEEP", a.inhib_ffx_sweep),
                     ("FF_GAIN", a.inhib_ff_gain), ("FF_W_I2E", a.inhib_ff_w_i2e), ("FF_SWEEP", a.inhib_ff_sweep)):
            if v is not None:
                env["C2S0_INHIB_" + k] = str(v)
        sys.exit(subprocess.call([sys.executable, "-u", "-m", "research.runners.stage2_exit_gate", "--shard", "--seed", "7",
                                  "--group", GROUP, "--arm", TABLE_ARMS[GROUP], "--out", a.smoke_0b], env=env, cwd=_REPO))
    seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else SEEDS
    if any(s in SG.GATE_SEEDS for s in seeds):
        ap.error("step 0 is dev seeds only")
    if a.emit_pool_lines:
        for cmd, why in pool_lines(a.emit_pool_lines, seeds=seeds, mem_gb=a.mem_gb):
            print("%s\t%s" % (cmd, why))
        return
    if a.score:
        res = score(a.score, seeds=seeds)
        res["m3_step0_d"] = score_m3_step0(a.score, seeds=seeds)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1, default=str)
        print(json.dumps({k: res[k] for k in ("result_line", "verdict", "complete_seed_set", "go_on_scored_seeds",
                                              "stop_lines_fired")}, indent=1))
        for s in res["stops"]:
            print("STOP %s seed %s %s: %s" % (s["line"], s["seed"], s["table"], json.dumps(s["detail"], default=str)[:600]))
        for u in res["undefined"]:
            print("UNDEFINED seed %s %s: %s" % (u["seed"], u["table"], u["why"]))
        for p in res["posts"]:
            print("POST seed %s %s: %s" % (p["seed"], p["table"], json.dumps(p, default=str)[:400]))
        m3 = res["m3_step0_d"]
        print("M3 step 0 (d), report only: %s %s" % (m3["verdict"], json.dumps(m3["per_seed"], default=str)[:600]))
        return
    ap.error("choose --score or --emit-pool-lines")


if __name__ == "__main__":
    main()
