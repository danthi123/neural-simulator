# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md
# Expected-fail: M3 (11-03) measured retrieval practice LOSING to restudy under gradient descent (-0.5/12 at 0.1B,
#   -2.67 at 0.4B), so a question-form target may store less than v3's copy-mode one; layerwise fitting may again leave
#   the output-relevant part unfitted; few items may have a brain-sourced cue (only 3 of the 12 are asked in the dev
#   lives; the 0.1B cortex may not generate an answerable question), so G0Q may stop the run. A negative is a verdict on
#   this question-form layerwise-target method at this adapter rank and scale, not on local learning.
"""Local rule v3q: the v3 rule (local_rule_v3, imported, not edited) with QUESTION-FORM targets (candidate 3 of
brain/research/2026-11-28-local-rule-research-round.md, section 2.3).

Same rule, same update, same keep, same clip and KP copies as v3. Only the clamped phase and the target positions
change:
1. Clamped phase (teacher), adapters off: the cortex reads the fact's telling (the stored episode text, `_told`), then
   a CUE in the chat frame ("User: <cue>\\n\\nAssistant:") followed by the cortex's OWN in-context answer (greedy, adapters
   off, telling in context). It records per block the residual after the block and the channel-mix hidden at the
   positions that PREDICT the answer tokens (the last prompt position through the second-to-last answer token).
2. Free phase (student), adapters on: the cortex reads the same cue + answer ids with NO telling in context.
3. Update: local_rule_v3.v3_update on those positions only (rows sliced; the channel mix is per position, so slicing
   is the exact masked rule). Keep step: v3's, unchanged (whole keep text, the pretrained cortex's own activity).

Cues must be the brain's own; no host-written question template:
- LIFE cues: questions the user actually asked in a DEV life (brain.eval.life `lived` rows, trunk line, kinds sameday
  and remention), extracted once by `cues-from-life` into a pinned data file (sha256 below). Test / test2 lives are
  refused.
- SELF cues: the cortex itself, adapters off, reads the telling and the next turn marker ("User:", the chat format's
  own role marker that every telling already carries) and writes the next user turn: greedy plus CUE_SAMPLES - 1
  seeded samples (CUE_SEED, fixed across seeds so every seed trains on one cue set).
A cue is kept only if it is a question (ends with "?"), does not name the answer key, and the teacher with the telling
in context answers it with the key. At most CUES_PER_ITEM cues per fact, life cues first. A fact with no kept cue is
EXCLUDED and counted (reason recorded); it is still scored in recall /12. Gate G0Q: >= G0Q_MIN facts kept.

Arms: v3q, gradient (brain.sleep on the tellings: v3's reference), no_sleep, sham_ctx (the teacher reads ANOTHER kept
fact's telling before the cue; same cue + answer ids), no_target (lesion: no telling, must stay inert), top_only (depth
lesion), grad_distill (report only: backprop KL to the same teacher's logits at the answer positions), gradient_q
(report only: brain.sleep on the cue + answer exchanges, retrieval practice under gradient: the M3 contrast).
The verdict is local_rule_v3.verdict on the v3 arms (v3 := v3q), words renamed (GO-V3 -> GO-V3Q), plus report-only
read-outs: the graded read-out (answer log-probability and rank at the question, hippocampus off; research note
section 4), recall on kept vs excluded facts, the eval-cue share of the gain, and the M3 contrast.

    POOL_NODE=1 python -m brain.experiments.local_rule_v3q run --model M.pth --seed 7 --arms g0q,no_sleep,gradient \\
        --out-dir D --tag ref
    POOL_NODE=1 python -m brain.experiments.local_rule_v3q run --model M.pth --seed 7 --arms v3q --lr 0.1 --keep 1 \\
        --out-dir D --tag lr0.1_keep1
    python -m brain.experiments.local_rule_v3q pick --dir D --out D/pick.json   # G0Q unit + 2 refs + 9 cells required
    python -m brain.experiments.local_rule_v3q score --dir S --tag six --pick D/pick.json
    python -m brain.experiments.local_rule_v3q cues-from-life --life L1.json L2.json L3.json --out F   # once, pinned
Prereg: brain/research/2026-11-28-local-rule-v3q-prereg-DRAFT.md (DRAFT).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import statistics
import time

import torch
import torch.nn.functional as F

from brain.experiments import local_rule_v3 as V
from brain.experiments.local_rule_v2 import (PROMPTS_SHA256, _finite, _load_dir, _mean, _model_id, _one, atomic_write,
                                             heldout_drift, prompts_sha, provenance, read_pick, split_prompts, unit_path)

SEEDS = V.SEEDS
DEV_SEED = V.DEV_SEED
ARMS = ("no_sleep", "gradient", "v3q", "sham_ctx", "no_target", "top_only", "grad_distill", "gradient_q")
LOCAL_ARMS = ("v3q", "sham_ctx", "no_target", "top_only")
TEACHER_OF = {"v3q": "double", "sham_ctx": "sham", "no_target": "none", "top_only": "double"}
REPORT_ARMS = ("grad_distill", "gradient_q")
GATED_ARMS = ("no_sleep", "gradient", "v3q", "sham_ctx", "no_target", "top_only")
TO_V3 = {"v3q": "v3"}                       # arm renames into local_rule_v3.verdict
DEV_LRS, DEV_KEEPS = V.DEV_LRS, V.DEV_KEEPS
G0Q_MIN = 6             # G0Q: >= 6 of 12 facts with a kept brain-sourced cue (the bar needs no_sleep + 2 trainable facts)
CUE_SAMPLES = 8         # self cues: greedy + 7 seeded samples
CUE_SEED = 0            # fixed across seeds: one cue set per model
CUE_TEMP = 1.0
CUE_MAX_TOKENS = 24
ANSWER_MAX_TOKENS = 16
CUES_PER_ITEM = 3
TURN = "User:"          # the chat format's own role marker (every telling carries it); not a question template
LIFE_KINDS = ("sameday", "remention")
LIFE_SPLITS = ("dev",)  # test / test2 lives refused
LIFE_CUES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_rule_v3q_life_cues.json")
LIFE_CUES_SHA256 = "5d24cf1a5bf26d8d784ac65e5b2291e40f502460101e6eeb8c490b0db2209ad4"   # pinned after cues-from-life
EVAL_CUE_SHARE_MAX = 0.5   # report: a GO whose gain is carried > 50% by facts cued with the eval question is flagged


def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("local_rule_v3q run: refuses to run unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1")


# ------------------------------------------------------------------ cues
def key_as_told(sentence, key):
    """The whole word of `sentence` that contains `key` (case as told), e.g. ("I'm allergic to peanuts.", "peanut") ->
    "peanuts"."""
    m = re.search(r"[\w']*" + re.escape(key) + r"[\w']*", sentence, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"key {key!r} not in {sentence!r}")
    return m.group(0)


def cue_problem(text, key):
    """None if `text` can be a cue for `key`: a one-line question (ends with "?") that does not name the answer."""
    t = (text or "").strip()
    if len(t) < 3 or "\n" in t:
        return "empty or multi-line"
    if not t.endswith("?"):
        return "not a question"
    if key.lower() in t.lower():
        return "names the answer"
    return None


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def extract_life_cues(lives, keys):
    """lives: [(name, life dict)]. Returns {"lives": [...], "kinds", "items": {key: [{q, life_seed, day, kind}]}}: the
    questions the user asked about each key in the lived trunk transcript, first-seen order, deduplicated. Refuses any
    life whose split is not dev (TEST and TEST2 are never read)."""
    items, meta = {k: [] for k in keys}, []
    for name, d in lives:
        if d.get("split") not in LIFE_SPLITS:
            raise SystemExit(f"cues-from-life: {name} is split {d.get('split')!r}; only {LIFE_SPLITS} lives may be read")
        meta.append({"name": name, "seed": d.get("seed"), "split": d.get("split"), "model": d.get("model")})
        for line in d.get("lived") or []:
            if line.get("line") != "trunk":
                continue
            for r in line.get("rows") or []:
                k = r.get("item")
                if k in items and r.get("kind") in LIFE_KINDS and cue_problem(r.get("you"), k) is None:
                    q = r["you"].strip()
                    if q not in [c["q"] for c in items[k]]:
                        items[k].append({"q": q, "life_seed": d.get("seed"), "day": r.get("day"), "kind": r["kind"]})
    return {"lives": meta, "kinds": list(LIFE_KINDS), "items": items}


def load_life_cues(path=LIFE_CUES_PATH, sha=LIFE_CUES_SHA256):
    got = file_sha(path)
    if sha is not None and got != sha:
        raise SystemExit(f"life cues {path} sha256 {got} != pinned {sha}: the file changed; re-register before running")
    with open(path) as fh:
        d = json.load(fh)
    return {k: list(v) for k, v in d["items"].items()}


def gen_text(pc, ids, state, max_tokens, gen=None, temperature=CUE_TEMP):
    """Text the cortex writes after `ids` (as it is now: enabled or not), up to `max_tokens` or the turn end ("\\n\\n",
    cut there) or token 0. gen None = greedy; else seeded sampling (on CPU, so it is device-independent)."""
    pipe = pc.cortex.pipe
    out = []
    with torch.no_grad():
        logits, st = pc(list(ids), state)
        lg = logits[-1]
        for _ in range(max_tokens):
            if gen is None:
                tok = int(torch.argmax(lg))
            else:
                p = torch.softmax(lg.detach().float().cpu() / temperature, -1)
                tok = int(torch.multinomial(p, 1, generator=gen))
            if tok == 0:
                break
            out.append(tok)
            text = pipe.decode(out)
            if "\n\n" in text:
                return text.split("\n\n")[0]
            lg, st = pc([tok], st)
            lg = lg[-1]
    return pipe.decode(out)


def _off(pc):
    """Context: the pretrained cortex (adapters off), restored after."""
    class _Ctx:
        def __enter__(self):
            self.was, pc.enabled = pc.enabled, False

        def __exit__(self, *exc):
            pc.enabled = self.was
    return _Ctx()


def teacher_answer(pc, ctx_ids, prefix_ids, state):
    """The pretrained cortex's greedy answer after ctx + the cue prefix."""
    with _off(pc):
        return gen_text(pc, list(ctx_ids) + list(prefix_ids), state, ANSWER_MAX_TOKENS)


def self_cues(pc, told_ids, state, item):
    """The cortex's own next user turns after the telling (adapters off): greedy, then seeded samples."""
    pipe = pc.cortex.pipe
    prefix = list(told_ids) + pipe.encode(TURN)
    g = torch.Generator().manual_seed(CUE_SEED * 1000 + item)
    with _off(pc):
        out = [gen_text(pc, prefix, state, CUE_MAX_TOKENS)]
        out += [gen_text(pc, prefix, state, CUE_MAX_TOKENS, gen=g) for _ in range(CUE_SAMPLES - 1)]
    return [o.strip() for o in out]


def cue_prefix(cue):
    return f"User: {cue}\n\nAssistant:"


def build_cue_set(pc, state, facts, life_cues, self_gen=True):
    """The kept (fact, cue, answer) units. facts: [(sentence, eval question, key)] (sleep_probe's FACTS layout);
    life_cues: {key: [{q, ...}]}. Life cues are tried first. Each unit: item index, key, cue, source, the teacher's
    in-context answer, qa_ids (cue prefix + answer ids) and pos (rows that predict the answer tokens)."""
    from brain.experiments.sleep_probe import _told
    pipe = pc.cortex.pipe
    units, rejected, excluded = [], [], {}
    for j, (sentence, q, key) in enumerate(facts):
        told_ids = pipe.encode(_told(sentence)[0])
        cands = [(c["q"], "life") for c in life_cues.get(key, [])]
        if self_gen:
            cands += [(c, "self") for c in self_cues(pc, told_ids, state, j)]
        kept, seen = [], set()
        for cue, src in cands:
            if len(kept) >= CUES_PER_ITEM:
                break
            if cue in seen:
                continue
            seen.add(cue)
            why = cue_problem(cue, key)
            if why:
                rejected.append({"item": j, "key": key, "cue": cue, "source": src, "why": why})
                continue
            prefix_ids = pipe.encode(cue_prefix(cue))
            ans = teacher_answer(pc, told_ids, prefix_ids, state)
            if key.lower() not in ans.lower():
                rejected.append({"item": j, "key": key, "cue": cue, "source": src, "answer": ans,
                                 "why": "the teacher with the telling in context does not answer with the key"})
                continue
            ans_ids = pipe.encode(ans)
            if not ans_ids:
                rejected.append({"item": j, "key": key, "cue": cue, "source": src, "why": "empty answer"})
                continue
            single = teacher_answer(pc, [], prefix_ids, state)
            qa = list(prefix_ids) + list(ans_ids)
            kept.append({"item": j, "key": key, "cue": cue, "source": src, "answer": ans, "single_answer": single,
                         "single_ok": key.lower() in single.lower(),
                         "cue_is_eval": cue.strip().lower() == q.strip().lower(),
                         "qa_ids": [int(t) for t in qa], "pos": list(range(len(prefix_ids) - 1, len(qa) - 1))})
        if kept:
            units.extend(kept)
        else:
            excluded[key] = ("no brain-sourced question (no life question, no self-generated text)" if not cands
                             else "no candidate cue survived (see rejected)")
    return summarize_cues(units, rejected, excluded, len(facts))


def cue_sha(units):
    return hashlib.sha256(json.dumps([(u["item"], u["cue"], u["qa_ids"], u["pos"]) for u in units]).encode()).hexdigest()


def summarize_cues(units, rejected, excluded, n):
    items = sorted({u["item"] for u in units})
    return {"n": n, "n_items": len(items), "items_kept": items, "n_units": len(units), "n_excluded": len(excluded),
            "excluded": excluded, "by_source": {s: sum(u["source"] == s for u in units) for s in ("life", "self")},
            "items_by_source": {s: len({u["item"] for u in units if u["source"] == s}) for s in ("life", "self")},
            "eval_cue_items": sorted({u["item"] for u in units if u["cue_is_eval"]}),
            "single_ok": sum(u["single_ok"] for u in units), "min_items": G0Q_MIN, "pass": g0q_pass(len(items)),
            "cue_sha256": cue_sha(units), "units": units, "rejected": rejected}


def g0q_pass(n_items):
    return isinstance(n_items, int) and not isinstance(n_items, bool) and n_items >= G0Q_MIN


# ------------------------------------------------------------------ the question-form night
def _rows(act, pos):
    """Slice every recorded tensor of an activity dict {layer: tuple of (T, .)} to rows `pos`."""
    out = {}
    for i, t in act.items():
        idx = torch.tensor(pos, dtype=torch.long, device=t[0].device)
        out[i] = tuple(x.index_select(0, idx) for x in t)
    return out


def ctx_of(units, told_ids, teacher_kind, perm=None):
    """Per unit, what the clamped phase reads before the cue: the fact's own telling ("double"), another kept fact's
    telling by a derangement over kept facts ("sham"), nothing ("none")."""
    items = sorted({u["item"] for u in units})
    out = []
    for u in units:
        if teacher_kind == "double":
            out.append(list(told_ids[u["item"]]))
        elif teacher_kind == "sham":
            out.append(list(told_ids[items[perm[items.index(u["item"])]]]))
        elif teacher_kind == "none":
            out.append([])
        else:
            raise ValueError(teacher_kind)
    return out


def q_error(pc, base_state, units, tgts, layers):
    """Fixed-order, forward-only engagement probe: every unit once, adapters as they are; mean over units of the summed
    per-layer value error at the answer positions. Changes nothing."""
    tot, per = [], {str(i): [] for i in layers}
    for u, tgt in zip(units, tgts):
        _, act = V.activity(pc, u["qa_ids"], base_state)
        act = _rows(act, u["pos"])
        e = {i: float((act[i][3] - tgt[i][1]).pow(2).sum()) / act[i][3].shape[0] for i in layers}
        tot.append(sum(e.values()))
        for i, v in e.items():
            per[str(i)].append(v)
    return _mean(tot), {k: _mean(v) for k, v in per.items()}


def _pick_unit(rng, by_item):
    """One fact uniformly, then one of its cues uniformly (two draws every step)."""
    j = rng.choices(range(len(by_item)), weights=[1.0] * len(by_item))[0]
    return by_item[j][rng.randrange(len(by_item[j]))]


def v3q_sleep(plastic, base_state, units, told_ids, generic, *, steps=240, lr=0.1, keep=1.0, decay=1e-3, seed=0,
              clip=1.0, teacher_kind="double", top_only=False, log=None, info=None):
    """The v3q night. units: build_cue_set(...)["units"]; told_ids: {item: ids of the fact's telling}. v3's rule, v3's
    keep; targets at the answer-predicting rows of the cue + answer, from the teacher reading `teacher_kind`'s context."""
    pc = V.adopt3(plastic)
    rng = random.Random(seed)
    kp = V.KPCopies(pc, seed)
    ids_of = pc.cortex.pipe.encode
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    items = sorted({u["item"] for u in units})
    perm = V.derangement(len(items), seed) if teacher_kind == "sham" else None
    layers = [pc.layers[-1]] if top_only else list(pc.layers)
    ctxs = ctx_of(units, told_ids, teacher_kind, perm)
    tgts = [_rows(V.teacher(pc, c, u["qa_ids"], base_state)[1], u["pos"]) for c, u in zip(ctxs, units)]
    keep_t, keep_ref = [], []
    for g in gen_ids:
        lg, t = V.teacher(pc, [], g, base_state)
        keep_t.append(t)
        keep_ref.append(F.log_softmax(lg, -1))
    by_item = [[k for k, u in enumerate(units) if u["item"] == it] for it in items]
    pc.enabled = True
    err0, err0_layer = q_error(pc, base_state, units, tgts, layers) if info is not None else (None, {})
    st_r, st_k = {}, {}
    losses, per_layer = [], []
    for step in range(steps):
        k = _pick_unit(rng, by_item)
        _, act = V.activity(pc, units[k]["qa_ids"], base_state)
        er = V.v3_update(pc, kp, _rows(act, units[k]["pos"]), tgts[k], lr=lr, layers=layers, decay=decay, clip=clip,
                         stats=st_r)
        g = rng.randrange(len(gen_ids))
        lg, gact = V.activity(pc, gen_ids[g], base_state)
        ek = V.v3_update(pc, kp, gact, keep_t[g], lr=lr, layers=layers, clip=clip, weight=keep, stats=st_k)
        kl = float(F.kl_div(F.log_softmax(lg, -1), keep_ref[g], log_target=True, reduction="batchmean"))
        losses.append((sum(v for v, _ in er.values()), sum(v for v, _ in ek.values()), kl))
        per_layer.append({str(i): v for i, (v, _) in er.items()})
        if log and step % 20 == 0:
            log(f"v3q step {step}: answer err {losses[-1][0]:.4f} keep err {losses[-1][1]:.4f} keep kl {kl:.4f}")
    if info is not None:
        err1, err1_layer = q_error(pc, base_state, units, tgts, layers)
        frac = lambda st: (st.get("clipped", 0) / st["n"]) if st.get("n") else None
        head, tail = losses[:V.LOSS_WIN], losses[-V.LOSS_WIN:]
        info.update(kind="local", teacher=teacher_kind, target_form="question", steps_run=len(losses), layers_on=layers,
                    targets_cached=len(tgts), n_units=len(units), n_items=len(items),
                    target_rows=_mean(len(u["pos"]) for u in units) if units else 0.0,
                    target_ctx_tokens=_mean(len(c) for c in ctxs) if ctxs else 0.0,
                    err_start=err0, err_end=err1, err_layer_start=err0_layer, err_layer_end=err1_layer,
                    clip_frac_replay=frac(st_r), clip_frac_keep=frac(st_k), clip=clip, keep=keep,
                    err_head=_mean(x[0] for x in head) if head else None,
                    err_tail=_mean(x[0] for x in tail) if tail else None,
                    keep_err_head=_mean(x[1] for x in head) if head else None,
                    keep_err_tail=_mean(x[1] for x in tail) if tail else None,
                    sham_perm=perm)
    return losses


def distill_q(plastic, base_state, units, told_ids, generic, *, steps=240, lr=5e-4, kl_weight=1.0, seed=0, log=None,
              info=None):
    """grad_distill (report only): Adam on KL(teacher || student) at the answer-predicting rows (teacher = the telling
    in context, adapters off) plus kl_weight x the keep KL, as brain.sleep. Same unit draws as v3q."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    pc = V.adopt3(plastic)
    ids_of = pc.cortex.pipe.encode
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    items = sorted({u["item"] for u in units})
    ctxs = ctx_of(units, told_ids, "double")
    idx = lambda u, t: torch.tensor(u["pos"], dtype=torch.long, device=t.device)
    tlog = []
    for c, u in zip(ctxs, units):
        lg = V.teacher(pc, c, u["qa_ids"], base_state)[0]
        tlog.append(F.log_softmax(lg.index_select(0, idx(u, lg)), -1))
    pc.enabled = False
    with torch.no_grad():
        ref = [F.log_softmax(pc(g, base_state)[0], -1) for g in gen_ids]
    pc.enabled = True
    by_item = [[k for k, u in enumerate(units) if u["item"] == it] for it in items]

    def student(u):
        lg = pc(u["qa_ids"], base_state)[0]
        return F.log_softmax(lg.index_select(0, idx(u, lg)), -1)

    def fixed_kl():
        with torch.no_grad():
            return _mean(float(F.kl_div(student(u), t, log_target=True, reduction="batchmean"))
                         for u, t in zip(units, tlog))
    k0 = fixed_kl() if info is not None else None
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        k = _pick_unit(rng, by_item)
        d = F.kl_div(student(units[k]), tlog[k], log_target=True, reduction="batchmean")
        g = rng.randrange(len(gen_ids))
        gl, _ = pc(gen_ids[g], base_state)
        kp = F.kl_div(F.log_softmax(gl, -1), ref[g], log_target=True, reduction="batchmean")
        opt.zero_grad()
        (d + kl_weight * kp).backward()
        opt.step()
        losses.append((float(d.detach()), float(kp.detach())))
        if log and step % 20 == 0:
            log(f"distill_q step {step}: kl to teacher {float(d):.4f} keep {float(kp):.4f}")
    if info is not None:
        info.update(V._loss_engagement("grad_distill", losses), loss_start=k0, loss_end=fixed_kl(),
                    target_form="question")
    return losses


def qa_episodes(units):
    """gradient_q's episodes: the cue + answer exchanges, each fact weighted 1 (its cues share it)."""
    n = {}
    for u in units:
        n[u["item"]] = n.get(u["item"], 0) + 1
    return [(cue_prefix(u["cue"]) + u["answer"] + "\n\n", 1.0 / n[u["item"]]) for u in units]


# ------------------------------------------------------------------ graded read-out (hippocampus off)
def graded(pc, state, facts):
    """Per fact, at the eval question's frame "User: <q>\\n\\nAssistant:", the log-probability of " <key as told>" (sum
    over its tokens, and its first token) and the rank of its first token (1 = top). The plastic cortex as it is."""
    pipe = pc.cortex.pipe
    lp_all, lp_first, ranks = [], [], []
    with torch.no_grad():
        for sentence, q, key in facts:
            p = pipe.encode(f"User: {q}\n\nAssistant:")
            a = pipe.encode(" " + key_as_told(sentence, key))
            lp = F.log_softmax(pc(list(p) + list(a), state)[0].float(), -1)
            row = len(p) - 1
            vals = [float(lp[row + j, a[j]]) for j in range(len(a))]
            lp_all.append(sum(vals))
            lp_first.append(vals[0])
            ranks.append(1 + int((lp[row] > lp[row, a[0]]).sum()))
    return {"logprob_items": lp_all, "first_logprob_items": lp_first, "rank_items": ranks,
            "mean_logprob": _mean(lp_all), "mean_first_logprob": _mean(lp_first),
            "median_rank": float(statistics.median(ranks)) if ranks else None,
            "n_rank_le10": sum(r <= 10 for r in ranks)}


# ------------------------------------------------------------------ registered rules (pure; tested on synthetic rows)
def dev_pick(g0q, ref, rows):
    """local_rule_v3.dev_pick with G0 := G0Q (pass = enough kept facts). STOP-G0 is named STOP-G0Q."""
    g0 = None if not isinstance(g0q, dict) else {"pass": g0q.get("pass") is True and g0q_pass(g0q.get("n_items")),
                                                 "n_ok": g0q.get("n_items"), "n": g0q.get("n")}
    out = V.dev_pick(g0, ref, rows)
    if out["verdict"] == "STOP-G0":
        out = {**out, "verdict": "STOP-G0Q", "why": f"G0Q not passed: {None if g0 is None else g0['n_ok']} facts with "
               f"a kept brain-sourced cue (needs >= {G0Q_MIN})"}
    return out


def _to_v3(recs):
    return {TO_V3.get(a, a): r for a, r in recs.items() if a in ARMS and a != "gradient_q"}


def _m(recs, arm, k, seeds):
    vals = [recs[arm][s].get(k) for s in seeds]
    return _mean(vals) if all(_finite(x) for x in vals) else None


def _items_mean(recs, arm, key, items, seeds):
    """Mean over seeds of the per-seed mean of `key` (a per-fact list) over `items`; None when unreadable."""
    out = []
    for s in seeds:
        xs = recs[arm][s].get(key)
        if not isinstance(xs, list) or not items or max(items) >= len(xs):
            return None
        out.append(_mean(float(xs[i]) for i in items))
    return _mean(out)


def report(recs, cues, seeds=SEEDS):
    """Report-only read-outs (never change the verdict word): graded (answer log-prob / rank at the question), recall and
    paraphrase /12 on kept vs excluded facts, eval-cue share of v3q's gain, paraphrase transfer (v3q minus no_sleep), the M3 contrast (gradient_q vs gradient), report-arm
    engagement."""
    kept = list((cues or {}).get("items_kept") or [])
    n = (cues or {}).get("n") or 12
    excl = [i for i in range(n) if i not in kept]
    evc = list((cues or {}).get("eval_cue_items") or [])
    other = [i for i in kept if i not in evc]
    g = {}
    for arm in ARMS:
        gr = {s: (recs[arm][s].get("graded") or {}) for s in seeds}
        lp = [gr[s].get("mean_logprob") for s in seeds]
        rk = [gr[s].get("median_rank") for s in seeds]
        g[arm] = {"mean_logprob": _mean(lp) if all(_finite(x) for x in lp) else None,
                  "median_rank": _mean(rk) if all(_finite(x) for x in rk) else None,
                  "logprob_kept": _items_mean({arm: {s: gr[s] for s in seeds}}, arm, "logprob_items", kept, seeds),
                  "logprob_excluded": _items_mean({arm: {s: gr[s] for s in seeds}}, arm, "logprob_items", excl, seeds),
                  "recall_kept": _items_mean(recs, arm, "recall_items", kept, seeds),
                  "recall_excluded": _items_mean(recs, arm, "recall_items", excl, seeds),
                  "paraphrase": _m(recs, arm, "paraphrase", seeds),
                  "paraphrase_kept": _items_mean(recs, arm, "paraphrase_items", kept, seeds),
                  "paraphrase_excluded": _items_mean(recs, arm, "paraphrase_items", excl, seeds)}
    ns = g["no_sleep"]
    for arm in ARMS:
        for k in ("mean_logprob", "logprob_kept", "paraphrase", "paraphrase_kept"):
            g[arm][f"{k}_minus_no_sleep"] = (g[arm][k] - ns[k]) if _finite(g[arm][k]) and _finite(ns[k]) else None
    gain = lambda items: (None if not items else
                          (lambda a, b: a - b if _finite(a) and _finite(b) else None)(
                              _items_mean(recs, "v3q", "recall_items", items, seeds),
                              _items_mean(recs, "no_sleep", "recall_items", items, seeds)))
    pgain = lambda items: (None if not items else
                           (lambda a, b: a - b if _finite(a) and _finite(b) else None)(
                               _items_mean(recs, "v3q", "paraphrase_items", items, seeds),
                               _items_mean(recs, "no_sleep", "paraphrase_items", items, seeds)))
    ge, go = gain(evc), gain(other)
    tot = (ge or 0.0) * len(evc) + (go or 0.0) * len(other)
    share = ((ge or 0.0) * len(evc) / tot) if tot > 0 else None
    eng_rep = []
    for s in seeds:
        e = recs["gradient_q"][s].get("engagement") or {}
        nn = recs["gradient_q"][s].get("adapter_norm")
        if not (_finite(nn) and nn > 0 and V._falls(e, "loss_start", "loss_end")):
            eng_rep.append(f"gradient_q s{s} not engaged: norm {nn}, loss {e}")
    gq, gr_ = _m(recs, "gradient_q", "recall", seeds), _m(recs, "gradient", "recall", seeds)
    return {"graded": g, "kept_items": kept, "excluded_items": excl, "eval_cue_items": evc,
            "eval_cue_gain_share": share, "eval_cue_carried": share is not None and share > EVAL_CUE_SHARE_MAX,
            "paraphrase_transfer": {"v3q_minus_no_sleep": g["v3q"]["paraphrase_minus_no_sleep"],
                                    "v3q_minus_no_sleep_kept": g["v3q"]["paraphrase_kept_minus_no_sleep"],
                                    "eval_cued_gain": pgain(evc), "other_kept_gain": pgain(other)},
            "m3_contrast": {"gradient_q_recall": gq, "gradient_recall": gr_,
                            "gradient_q_minus_gradient": (gq - gr_) if _finite(gq) and _finite(gr_) else None},
            "report_engagement": eng_rep}


def verdict(recs, last_layer=None, cues=None):
    """Per model. recs: {arm: {seed: {...}}} for every arm of ARMS on exactly SEEDS (else ValueError). The verdict word
    is local_rule_v3.verdict on the v3 arms with v3 := v3q (all its clauses, engagement gates and order), renamed:
    GO-V3 -> GO-V3Q; a core NEGATIVE's reading from grad_distill is re-worded for the question-form target. `cues` (the
    G0Q record) must pass, else UNDEFINED. Report-only read-outs under "report"."""
    for arm in ARMS:
        if arm not in recs or set(recs[arm]) != set(SEEDS):
            raise ValueError(f"verdict needs every arm on exactly the seeds {SEEDS}: {arm} has {sorted(recs.get(arm, {}))}")
    if not (isinstance(cues, dict) and cues.get("pass") is True and g0q_pass(cues.get("n_items"))):
        return {"n_seeds": len(SEEDS), "verdict": "UNDEFINED",
                "why": [f"G0Q not passed or missing: {None if not isinstance(cues, dict) else cues.get('n_items')} kept"]}
    v = V.verdict(_to_v3(recs), last_layer)
    word = {"GO-V3": "GO-V3Q"}.get(v["verdict"], v["verdict"])
    out = {**v, "verdict": word}
    if "means" in out:
        out["means"] = {("v3q" if a == "v3" else a): x for a, x in out["means"].items()}
        out["means"]["gradient_q"] = {k: _m(recs, "gradient_q", k, SEEDS) for k in V.GATED + ("adapter_norm",)}
    if "v3_beats_sham" in out:
        out["v3q_beats_sham"] = out.pop("v3_beats_sham")
        out["v3q_minus_top_only"] = out.pop("v3_minus_top_only")
    r = out.get("reading") or ""
    if r.startswith("rule-limit"):
        out["reading"] = ("rule-limit: grad_distill passes (a) and (c) on the same question-form target; the delta-rule "
                          "family stops at this rank")
    elif r.startswith("target-limit"):
        out["reading"] = ("target-limit: grad_distill fails (a) or (c) on the question-form target too; neither target "
                          "form stores at this scale (consistent with M3: retrieval practice lost to restudy)")
    out["report"] = report(recs, cues)
    out["cues"] = {k: cues.get(k) for k in ("n", "n_items", "n_units", "n_excluded", "excluded", "by_source",
                                            "items_by_source", "eval_cue_items", "cue_sha256")}
    return out


def combine(v01, v04):
    """local_rule_v3.combine on the v3q words (GO-V3Q -> GO-V3Q / GO-V3Q-PARTIAL; STOP-G0Q is undefined)."""
    m = lambda w: "GO-V3" if w == "GO-V3Q" else w
    return {"GO-V3": "GO-V3Q", "GO-V3-PARTIAL": "GO-V3Q-PARTIAL"}.get(V.combine(m(v01), m(v04)), V.combine(m(v01), m(v04)))


# ------------------------------------------------------------------ CLI
def cmd_pick(a):
    """Dev pick. Refuses unless the dir holds exactly the G0Q unit, the 2 ref units and one v3q unit per (lr, keep) cell,
    all seed 7, one model, one commit, one prompts list, one cue set."""
    units = [u for u in _load_dir(a.dir, "*.json") if isinstance(u, dict) and "arm" in u and u.get("seed") == DEV_SEED]
    ref_u = [u for u in units if u["tag"] == a.ref_tag and u["arm"] in ("no_sleep", "gradient")]
    g0_u = [u for u in units if u["tag"] == a.ref_tag and u["arm"] == "g0q"]
    q_u = [u for u in units if u["tag"] != a.ref_tag and u["arm"] == "v3q"]
    bad = []
    if sorted(u["arm"] for u in ref_u) != ["gradient", "no_sleep"]:
        bad.append(f"ref units {sorted(u['arm'] for u in ref_u)} != [gradient, no_sleep]")
    if len(g0_u) != 1:
        bad.append(f"{len(g0_u)} G0Q units != 1")
    cells = sorted((u.get("lr"), u.get("keep")) for u in q_u)
    if cells != sorted(itertools.product(DEV_LRS, DEV_KEEPS)):
        bad.append(f"v3q cells {cells} != the 9 registered (lr, keep)")
    if bad:
        raise SystemExit("pick refused: " + "; ".join(bad))
    allu = g0_u + ref_u + q_u
    model, commit = _one(allu, "model_id", "pick"), _one(allu, "commit", "pick")
    _one(ref_u + q_u, "prompts_sha256", "pick")
    csha = _one(allu, "cue_sha256", "pick")
    if not commit:
        raise SystemExit("pick refused: units carry no commit")
    ref = {u["arm"]: u["result"] for u in ref_u}
    rows = [{"lr": u["lr"], "keep": u["keep"], **u["result"],
             **{k: (u.get("engagement") or {}).get(k)
                for k in ("err_start", "err_end", "clip_frac_replay", "clip_frac_keep")}} for u in q_u]
    g0r = g0_u[0]["result"]
    res = {**dev_pick(g0r, ref, rows), "model_id": model, "commit": commit, "cue_sha256": csha,
           "prompts_sha256": ref_u[0].get("prompts_sha256"),
           "g0q": {k: g0r.get(k) for k in ("n", "n_items", "n_units", "n_excluded", "by_source", "pass")},
           "graded_rows": [{"lr": u["lr"], "keep": u["keep"],
                            "mean_logprob": (u["result"].get("graded") or {}).get("mean_logprob")} for u in q_u]}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def check_units(units, pick=None):
    """One instrument: one model, commit, prompts list and cue set; one (lr, keep) across the local arms; the pick's."""
    model, commit = _one(units, "model_id", "score"), _one(units, "commit", "score")
    _one(units, "prompts_sha256", "score")
    csha = _one(units, "cue_sha256", "score")
    loc = [u for u in units if u["arm"] in LOCAL_ARMS]
    lr, keep = _one(loc, "lr", "score (local arms)"), _one(loc, "keep", "score (local arms)")
    if pick is not None:
        want = (pick["pick"]["lr"], pick["pick"]["keep"], pick["model_id"], pick["commit"], pick.get("cue_sha256"))
        if (lr, keep, model, commit, csha) != want:
            raise SystemExit(f"units (lr, keep, model, commit, cue_sha) {(lr, keep, model, commit, csha)} != pick {want}")
    return {"lr": lr, "keep": keep, "model_id": model, "commit": commit, "cue_sha256": csha}


def cmd_score(a):
    units = [u for u in _load_dir(a.dir, f"{a.tag}_s*_*.json") if u.get("seed") in SEEDS and u.get("tag") == a.tag
             and u.get("arm") in ARMS]
    recs = {}
    for u in units:
        rec = dict(u["result"])
        rec["engagement"] = u.get("engagement") or {}
        recs.setdefault(u["arm"], {})[u["seed"]] = rec
    missing = [(arm, s) for arm in ARMS for s in SEEDS if s not in recs.get(arm, {})]
    if missing:
        raise SystemExit(f"missing units: {missing}")
    inst = check_units(units, read_pick(a.pick) if a.pick else None)
    last = _one(units, "last_layer", "score")
    cues = units[0].get("cues")
    res = {**verdict(recs, last, cues), "instrument": inst}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def cmd_combine(a):
    vs = []
    for p in (a.v01, a.v04):
        with open(p) as fh:
            vs.append(json.load(fh).get("verdict"))
    res = {"v01": vs[0], "v04": vs[1], "verdict": combine(*vs)}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def cmd_cues_from_life(a):
    from brain.experiments.facts import FACTS
    lives = []
    for p in a.life:
        with open(p) as fh:
            lives.append((os.path.basename(p), json.load(fh)))
    d = extract_life_cues(lives, [k for _, _, k in FACTS])
    for (name, _), m, p in zip(lives, d["lives"], a.life):
        m["sha256"] = file_sha(p)
    atomic_write(a.out, d)
    print(json.dumps({k: len(v) for k, v in d["items"].items()}), file_sha(a.out), flush=True)


def _cues_public(cs):
    """The cue record as stored in every unit (ids kept: the unit set is reproducible from it)."""
    return {k: cs[k] for k in cs}


def run_units(a, c, base, *, PROMPTS, GENERIC, facts, life_cues, evaluate, sleep, keys_all, PlasticCortex,
              self_gen=True):
    """The per-(seed, arm) loop of `run` on a loaded cortex `c` (tests pass a stub cortex). The cue set is built once
    (adapters off, CUE_SEED) before any night; a failing G0Q writes the g0q unit (if asked) and stops before any night.
    Each arm is one atomic unit; existing units are skipped (resume); a resumed run whose cue set differs from the
    stored G0Q unit's refuses."""
    from brain.experiments.sleep_probe import _told
    dev_p, ver_p = split_prompts(PROMPTS, a.prompts_sha)
    psha = prompts_sha(PROMPTS)
    if a.pick:
        p = read_pick(a.pick)
        if p.get("model_id") != _model_id(a.model):
            raise SystemExit(f"pick model {p.get('model_id')} != --model {_model_id(a.model)}")
        a.lr, a.keep = p["pick"]["lr"], p["pick"]["keep"]
    prov = provenance(a.device, "float32")
    if a.pick and p.get("commit") != prov["commit"]:
        raise SystemExit(f"pick commit {p.get('commit')} != this code's commit {prov['commit']}")
    order = (["g0q"] if "g0q" in a.arm_list else []) + [x for x in a.arm_list if x != "g0q"]
    todo = [arm for arm in order if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]
    if not todo:
        return []
    t0 = time.time()
    torch.manual_seed(a.seed)
    cs = build_cue_set(PlasticCortex(c), base, facts, life_cues, self_gen=self_gen)
    cue_s = time.time() - t0
    if a.pick and p.get("cue_sha256") not in (None, cs["cue_sha256"]):
        raise SystemExit(f"pick cue set {p.get('cue_sha256')} != this run's {cs['cue_sha256']}")
    g0f = unit_path(a.out_dir, a.tag, a.seed, "g0q")
    if os.path.exists(g0f):
        with open(g0f) as fh:
            old = json.load(fh)["result"]
        if old.get("cue_sha256") != cs["cue_sha256"]:
            raise SystemExit(f"cue set {cs['cue_sha256']} != the stored G0Q unit's {old.get('cue_sha256')}: refusing")
    base_unit = {"tag": a.tag, "seed": a.seed, "model": a.model, "model_id": _model_id(a.model), "steps": a.steps,
                 "prompts_sha256": psha, "pick": a.pick, "cue_sha256": cs["cue_sha256"],
                 "life_cues_sha256": getattr(a, "life_cues_sha", None), **prov}
    if "g0q" in todo:
        unit = {**base_unit, "arm": "g0q", "elapsed_s": cue_s, "engagement": {"kind": "gate", "n": cs["n"]},
                "result": _cues_public(cs)}
        atomic_write(g0f, unit)
        print("g0q", {k: cs[k] for k in ("n", "n_items", "n_units", "n_excluded", "by_source", "pass")}, flush=True)
    if not cs["pass"]:
        raise SystemExit(f"G0Q FAILED: {cs['n_items']} of {cs['n']} facts have a kept brain-sourced cue (needs >= "
                         f"{G0Q_MIN}); excluded {cs['excluded']}: no nights run")
    units = cs["units"]
    pipe = c.pipe
    told_ids = {j: pipe.encode(_told(s)[0]) for j, (s, _, _) in enumerate(facts)}
    tellings = [_told(s) for s, _, _ in facts]
    cues_rec = {k: cs[k] for k in ("n", "n_items", "items_kept", "n_units", "n_excluded", "excluded", "by_source",
                                   "items_by_source", "eval_cue_items", "pass", "cue_sha256")}
    for arm in [x for x in todo if x != "g0q"]:
        t1 = time.time()
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        last = pc.layers[-1]
        eng = {}
        lg = lambda msg, arm=arm: print(arm, msg, flush=True)
        if arm == "no_sleep":
            eng = {"kind": "none", "steps_run": 0}
        elif arm == "gradient":
            ce0 = V.replay_ce(pc, base, tellings)
            losses = sleep(pc, base, tellings, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight,
                           seed=a.seed, log=lg)
            eng = {**V._loss_engagement("gradient", losses), "loss_start": ce0, "loss_end": V.replay_ce(pc, base, tellings)}
        elif arm == "gradient_q":
            qeps = qa_episodes(units)
            ce0 = V.replay_ce(pc, base, qeps)
            losses = sleep(pc, base, qeps, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight, seed=a.seed,
                           log=lg)
            eng = {**V._loss_engagement("gradient_q", losses), "loss_start": ce0, "loss_end": V.replay_ce(pc, base, qeps),
                   "target_form": "question"}
        elif arm == "grad_distill":
            distill_q(pc, base, units, told_ids, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight,
                      seed=a.seed, log=lg, info=eng)
        else:
            v3q_sleep(pc, base, units, told_ids, GENERIC, steps=a.steps, lr=a.lr, keep=a.keep, decay=a.decay,
                      seed=a.seed, teacher_kind=TEACHER_OF[arm], top_only=(arm == "top_only"), log=lg, info=eng)
        r = evaluate(pc, base, keys_all)
        r["graded"] = graded(pc, base, facts)
        r["heldout_kl"] = None if a.seed == DEV_SEED else heldout_drift(pc, base, ver_p)   # dev/verdict firewall
        r["heldout_kl_dev"] = heldout_drift(pc, base, dev_p)
        r["adapter_norm"] = pc.adapter_norm()
        r["factor_norms"] = pc.factor_norms()
        eng["adapter_norm"] = r["adapter_norm"]
        unit = {**base_unit, "arm": arm, "last_layer": last, "cues": cues_rec,
                "lr": a.lr if arm in LOCAL_ARMS else None, "keep": a.keep if arm in LOCAL_ARMS else None,
                "decay": a.decay if arm in LOCAL_ARMS else None,
                "grad_lr": a.grad_lr if arm in ("gradient", "grad_distill", "gradient_q") else None,
                "teacher": TEACHER_OF.get(arm), "elapsed_s": time.time() - t1, "cue_build_s": cue_s,
                "engagement": eng, "result": r}
        atomic_write(unit_path(a.out_dir, a.tag, a.seed, arm), unit)
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items") and k != "graded"},
              "graded", {k: r["graded"][k] for k in ("mean_logprob", "median_rank")}, flush=True)
    return todo


def cmd_run(a):
    guard()
    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    if [x for x in want if x not in ARMS + ("g0q",)]:
        raise SystemExit(f"unknown arms; choose from {ARMS + ('g0q',)}")
    a.arm_list = want
    if not [arm for arm in want if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]:
        print("all units done", flush=True)
        return
    from brain.brain import BIRTH
    from brain.cortex import Cortex
    from brain.experiments.facts import FACTS
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import GENERIC, evaluate
    from brain.plastic_cortex import PlasticCortex as PC
    from brain.sleep import sleep
    life = load_life_cues(a.life_cues)
    a.life_cues_sha = file_sha(a.life_cues)
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    todo = run_units(a, c, base, PROMPTS=PROMPTS, GENERIC=GENERIC, facts=list(FACTS), life_cues=life,
                     evaluate=evaluate, sleep=sleep, keys_all=[k for _, _, k in FACTS], PlasticCortex=PC)
    print(f"done {len(todo)} units in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--lr", type=float, default=0.1)
    r.add_argument("--keep", type=float, default=1.0)
    r.add_argument("--decay", type=float, default=1e-3)
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--grad-lr", type=float, default=5e-4)
    r.add_argument("--steps", type=int, default=240)
    r.add_argument("--device", default="cpu")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--pick", default=None, help="dev pick.json: lr and keep from it; model, commit, cue set must match")
    r.add_argument("--prompts-sha", default=PROMPTS_SHA256)
    r.add_argument("--life-cues", default=LIFE_CUES_PATH)
    p = sub.add_parser("pick")
    p.add_argument("--dir", required=True)
    p.add_argument("--ref-tag", default="ref")
    p.add_argument("--out", default=None)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--pick", default=None)
    c = sub.add_parser("combine")
    c.add_argument("--v01", required=True)
    c.add_argument("--v04", required=True)
    c.add_argument("--out", default=None)
    e = sub.add_parser("cues-from-life")
    e.add_argument("--life", nargs="+", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"run": cmd_run, "pick": cmd_pick, "score": cmd_score, "combine": cmd_combine,
     "cues-from-life": cmd_cues_from_life}[a.cmd](a)


if __name__ == "__main__":
    main()
