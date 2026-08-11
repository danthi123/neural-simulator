"""INTEGRATION #7 -- wire PLASTICITY-LEARNED grounded facts into the LIVE multi-turn chat, so the brain answers about
facts it acquired BY ITS OWN SYNAPTIC PLASTICITY (an e-prop weight change on a spiking Izhikevich readout), with a
LEARNED familiarity gate as the no-confab moat for those facts. This is the EMERGENCE-BAR burn-down of INTEGRATION #6:
#6 injected content via a host `comp.store` (VSA write); #7's fact ACQUISITION is a weight change (the win #6 lacked).

THE WIRE-IN (all runner-side, reuse-by-import). Every grounded-content decision in the live chat routes through ONE
method -- `comp.query_patient(agent, action)` (used by `_classify`, `_gm_retrieve_neighbourhood`, and the post-hoc
moat `_gm_posthoc_verify`). The plasticity-learned fact lives in the e-prop readout weights of an `OnBridgeEpropNet`,
NOT in `comp.kb`, so a raw `query_patient` abstains -- it needs the adapter. `AcquiredReadComposer` (reused from the
moat-closure de-risk) wraps `comp` and overrides `query_patient`: (1) structural kb moat FIRST (byte-identical for the
curated facts + when disabled); (2) on a kb abstain, consult the LEARNED e-prop read GATED by the LEARNED conjunctive
familiarity gate (novel cue -> abstain; familiar -> argmax patient). Because ALL chat content routes through
`query_patient`, ONE wrap of `comp` propagates through classification + retrieval + post-hoc verification (the same
leverage #6 got from `comp.kb`). The one adapter delta is a transparent `__getattr__` passthrough (the chat path also
touches `.store`/`.kb`/`.concepts`/`.words`), added here as a ~3-line `ChatShim(AcquiredReadComposer)` subclass.

SCALE (HONEST, per the mapping's capacity finding). Jointly-taught K=3 is the reliable 6/6 regime; sequential/continual
breadth is NOT solved (frac_recalled ~ 1/N -- an open arc). So teach K=3 facts TOGETHER
(TAUGHT={dax:grass, dog:bone, cat:fish}), ONE contrastive pass. The deliverable is a SMALL-K learned-facts live demo
STANDING BESIDE #6's host-stored breadth, NOT replacing it. Scale-up to #6 breadth is explicitly gated on the
continual-learning arc.

GO GATE (6 seeds 42/43/44/100/101/102; cfg.seed-controlled, SIM_BACKEND=numpy):
  1. RECALL from synapses: after plasticity-teaching the live chat answers each taught cue with a grounded,
     moat-verified reply; the taught-fact grounded-recall count RISES vs a matched (pre-teaching) baseline.
  2. MOAT holds at chat scale: untaught cues (dax+chases, wug+eats + a sampled battery) AND the OOD HUMAN_TURNS ->
     0 confabulations; `_gm_posthoc_verify` drops 100% of unsupported props; `_detect_ungrounded == 0`.
  3. BYTE-IDENTITY off: shim `enabled=False` -> decision-transcript bit-identical to the #6 default build.

ANTI-CHEATS (all required):
  FROZEN-READOUT (eprop_lr=0 during the IDENTICAL teaching): the taught cue's CORRECT patient is NOT recalled
    (readout un-moved -> argmax rides the zero-init attractor 'apple', never 'grass') -> taught-recall stays 0 ->
    the CONTENT rode the weight change, not a host path. THE KEY ANTI-CHEAT.
  kb-unchanged: `len(comp.kb)` identical before/after teaching (acquisition is a weight change, not a host append),
    and the answer comes from the net's forward record, not `comp.store`.
  untrained-net baseline (same shim, fresh net + un-imprinted gate): every taught cue abstains -> chat silent on them.
  LESION-gate: (a) conf-only gate-OFF (the old mechanism) -> untaught cues start answering (false-accepts return ->
    the learned familiarity gate is load-bearing); (b) `fam.lesion()` -> the taught-vs-untaught novelty margin
    collapses to ~0 (the abstain rides the LEARNED projector W).
  mispaired-teacher: a consistent WRONG referent->patient pairing learns a deterministic map wrong on the true
    targets -> main-net held-out >> mispaired held-out (the acquired ANSWER is the teacher's specific pairing).
  single-class control: teaching ONLY dax->grass collapses to a constant grass-bias (dog/cat also read grass) ->
    contrastive teaching is LOAD-BEARING for the multi-fact map.

HONEST SCOPE (per THE LAW + docs/TERMS.md). GENUINELY brain-based: fact ACQUISITION is synaptic (e-prop weight change
in a spiking Izhikevich substrate); moat DISCRIMINATION is on the spiking readout + a learned anti-Hebbian projector.
DECLARED BURN-DOWNS (named, not deferred): (1) TWO bridges -- `OnBridgeEpropNet` builds its OWN `SimulationBridge`
co-resident with the conversational one, NOT merged (brain-based but not yet ONE-brain; the merge is the named next
step); (2) the familiarity gate is a numpy anti-Hebbian projector (host-idealized; spiking `familiarity_gate_v320`
exists to swap in); (3) the conjunctive cue codebook + patient argmax read-out (composer-idealization +
neural-motor-readout targets). LEGITIMATELY host: the teacher/curriculum (AI-teacher environment), the generator mouth
(deliberately OFF here -- the grounded CONTENT is the learned read, not the mouth). If it does NOT compose (likely
failure: the NOV_GATE=0.5 margin shrinking at chat-vocab scale), that is a first-class NEGATIVE naming "re-fit the
source-monitor at chat-vocab scale / drop in the spiking v320 gate" as the next mechanism.

DISCIPLINE: SIM_BACKEND=numpy, reuse-by-import, NO `sim/` edit (only the runner-side additive `ChatShim.__getattr__`
passthrough + a superset-verified render tweak for the pre-inflected plasticity actions), cfg.seed (via
build_one_brain + the a1 net's CoreSimConfig.seed), additive/default-off.

Run (cheap-first single-seed SMOKE, mouth-free):
  PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._teacher_loop_facts_into_live_chat_derisk --seeds 42 --smoke
Full 6-seed sweep:
  PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._teacher_loop_facts_into_live_chat_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/lanes/stageA/plasticity_facts_live_chat_6seed.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402

from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners import _conversation_turing_test_derisk as TT  # noqa: E402
from research.runners import _corpus_facts_into_live_chat_derisk as CF  # noqa: E402
from research.runners.rf_phasor_composer import DEFAULT_VOCAB  # noqa: E402
# reuse-by-import: the plasticity substrate + the learned moat (do NOT reinvent).
from research.runners._teacher_loop_contrastive_familiarity_moat_derisk import (  # noqa: E402
    AcquiredReadComposer, ConjunctiveFamiliarityGate, ReferentEnv, _contrastive_batch, TAUGHT,
    _train_eprop, _predict_settled, PATIENT_WORDS, ACTIONS, UNTAUGHT_REFERENT, HEADLINE_REFERENT,
    _mk_net, _single_class_batch, _heldout_acc, _majority, _readout_norm,
)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# VOCAB + rendering. V = DEFAULT_VOCAB U {dax, wug, eats, chases, grass, bone, fish, seed} so the plasticity
# referents/actions/patients are all in comp.words (needed so the learned answer is a GROUNDED word the surface
# scan recognises, and so the classifier can name the referents).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
PLASTICITY_WORDS = ["dax", "wug", "eats", "chases", "grass", "bone", "fish", "seed"]
V = sorted(set(DEFAULT_VOCAB) | set(PLASTICITY_WORDS))

# ---- render tweak (superset-verified below): the plasticity actions 'eats'/'chases' are ALREADY 3rd-person
# present; SA._gm_fact_to_english would DOUBLE-inflect ('eatses'), which the surface scan then flags as ungrounded.
# For those verbs we render verbatim; for every DEFAULT_VOCAB motion verb we DELEGATE to the original untouched, so
# the render is a PURE SUPERSET of #6's (asserted at import) -> the byte-identity path is unaffected.
_ORIG_FACT_TO_ENGLISH = SA._gm_fact_to_english


def _fact_to_english(svo):
    a, v, p = svo
    if v in ACTIONS:                       # 'eats'/'chases': already 3rd-person present -> verbatim
        return f"{a.capitalize()} {v} {p}."
    return _ORIG_FACT_TO_ENGLISH(svo)


# superset guard: for every #6 vocabulary fact the render is BYTE-identical to the original (delegated). This is
# what makes the additive render safe for the byte-identity gate (curated facts never use an ACTIONS verb).
for _a, _v, _p in SA.CURATED_FACTS:
    assert _fact_to_english((_a, _v, _p)) == _ORIG_FACT_TO_ENGLISH((_a, _v, _p)), "render is not a #6 superset"
SA._gm_fact_to_english = _fact_to_english   # runner-side, additive; only changes 'eats'/'chases' (absent from #6)

# teacher presentation hyperparameters (the moat-closure runner's 6/6 regime).
D_P, NOISE, HIDDEN, SETTLE, EPOCHS, BATCH, EPROP_LR, W_CLIP, N_DRAWS, D_FAM = (
    32, 0.12, 40, 25, 80, 20, 0.5, 4000.0, 48, 256)
N_IN = D_P + len(ACTIONS)
K = len(PATIENT_WORDS)
CHANCE = 1.0 / K
TAUGHT_FACTS = [(r, "eats", TAUGHT[r]) for r in TAUGHT]      # the 3 plasticity facts, as (agent, action, patient)


class ChatShim(AcquiredReadComposer):
    """The moat-closure adapter + the ONE #7 delta: a transparent `__getattr__` passthrough so the shim is a drop-in
    `comp` for the whole chat path (which also touches `.store`/`.kb`/`.concepts`). `__getattr__` fires only for
    attributes AcquiredReadComposer does not define (comp/env/net/fam/words/query_patient are all normal lookups)."""

    def __getattr__(self, k):
        if k == "comp":                    # bootstrap guard: avoid recursion before self.comp is set
            raise AttributeError(k)
        return getattr(self.comp, k)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The TEACHER PRESENTATION (the AI-teacher environment = a LEGITIMATE host boundary): move the brain's OWN e-prop
# readout weights over dax+dog+cat AND imprint the source-monitor on the taught cues. Returns (net, fam, readout_moved).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _make_env(seed):
    env = ReferentEnv(seed, d_p=D_P, noise=NOISE)
    for r in list(TAUGHT) + [UNTAUGHT_REFERENT]:            # instantiate the world's referents (fixed order)
        env.proto(r)
    return env


def _teach(seed, env, mispaired=False, single_class=False, freeze=False):
    """One teacher presentation. e-prop moves the readout over the contrastive (or single-class / mispaired) batch;
    a fresh source-monitor imprints the taught cues. freeze=True sets eprop_lr=0 (the FROZEN-READOUT control)."""
    net = _mk_net(N_IN, K, seed, HIDDEN, SETTLE, EPROP_LR, W_CLIP)
    if freeze:
        net.eprop_lr = 0.0
    fam = ConjunctiveFamiliarityGate(seed, d_p=D_P, D=D_FAM)
    for r in TAUGHT:
        fam.imprint(env, r, "eats")
    ro0 = _readout_norm(net)
    if single_class:
        Xtr, ytr = _single_class_batch(env, seed, N_DRAWS)
    else:
        Xtr, ytr = _contrastive_batch(env, seed, N_DRAWS, mispaired=mispaired)
    _train_eprop(net, Xtr, ytr, EPOCHS, BATCH, seed)
    return net, fam, float(abs(_readout_norm(net) - ro0))


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CHAT turn list + recall extraction.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _turns():
    """The shipped 14 human turns (incl. the OOD probes that MUST abstain) + teacher probes for each taught referent
    (RECALL) + an untaught-referent probe (must abstain live)."""
    probes = [("Tell me about the %s." % r, "plasticity-taught subject probe (%s)" % r) for r in TAUGHT]
    probes.append(("You mentioned a %s earlier -- what does the %s do?" % (UNTAUGHT_REFERENT, UNTAUGHT_REFERENT),
                   "untaught-referent probe -> must abstain"))
    return list(TT.HUMAN_TURNS) + probes


def _taught_recall(transcript):
    """Taught-fact grounded recall: the count of taught (ref,'eats',patient) that appear as a grounded,
    moat-verified proposition in the chat (the exact GO-gate-1 quantity -- 'the chat answers each taught cue')."""
    got = set()
    for r in transcript:
        if r.get("category") == "grounded" and not r.get("confabulated"):
            for f in (r.get("neighbourhood") or []):
                a, v, p = f
                if v == "eats" and a in TAUGHT and p == TAUGHT[a]:
                    got.add(a)
    return len(got), sorted(got)


def _probe_answer(transcript, referent):
    """The brain's reply on the 'Tell me about the <referent>.' turn (for the transcript record)."""
    for r in transcript:
        if r.get("cue_agent") == referent and "subject probe" in (r.get("tag") or ""):
            return r.get("brain_reply") or ""
    return ""


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The MOAT battery at chat scale: the untaught (referent, action) cues the shim's learned gate must ABSTAIN on.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _moat_battery(shim):
    """The reachable untaught cues (agent in env.protos x action in ACTIONS, minus the 3 taught). All must abstain
    (query_patient -> None). Returns (false_accepts, examples)."""
    taught_cues = {(r, "eats") for r in TAUGHT}
    fa, ex = 0, []
    for a in list(TAUGHT) + [UNTAUGHT_REFERENT]:
        for v in ACTIONS:
            if (a, v) in taught_cues:
                continue
            ans = shim.query_patient(a, v)
            if ans is not None:
                fa += 1
                ex.append((a, v, ans))
    return fa, ex


def _lesion_margin(fam, env):
    """FG3(b): the taught-vs-untaught novelty margin, intact vs after lesioning the LEARNED projector W. The abstain
    rides the learned weights iff lesioning COLLAPSES the margin. Re-imprints to restore the gate afterwards."""
    def _mean_nov(cues):
        return float(np.mean([fam.novelty(env, r, a) for (r, a) in cues for _ in range(20)]))
    taught_cues = [(r, "eats") for r in TAUGHT]
    untaught_cues = [(HEADLINE_REFERENT, "chases"), (UNTAUGHT_REFERENT, "eats")]
    intact = _mean_nov(untaught_cues) - _mean_nov(taught_cues)
    fam.lesion()
    lesioned = _mean_nov(untaught_cues) - _mean_nov(taught_cues)
    for r in TAUGHT:                                        # restore
        fam.imprint(env, r, "eats")
    return float(intact), float(lesioned)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BYTE-IDENTITY (gate 3): the shim with enabled=False is bit-identical to the #6 default build over HUMAN_TURNS.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def byte_identity(seed):
    xp, _ = get_backend()
    turns = list(TT.HUMAN_TURNS)
    # #6 DEFAULT reference: raw composer, curated facts only, DEFAULT_VOCAB.
    b0, c0, i0, s0 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True)
    th0 = SA._threshold_hash(b0, b0.core_config.num_neurons)
    cc0 = CF._concept_hash(c0)
    nn0 = int(b0.core_config.num_neurons)
    _v0, f0 = SA._store_facts(c0)
    t0 = CF.run_chat(b0, xp, i0, s0, c0, f0, turns)
    # shim-DISABLED at DEFAULT_VOCAB: the additive shim wrapping an identically-built composer -> must reproduce t0.
    b1, c1, i1, s1 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True)
    th1 = SA._threshold_hash(b1, b1.core_config.num_neurons)
    cc1 = CF._concept_hash(c1)
    nn1 = int(b1.core_config.num_neurons)
    _v1, f1 = SA._store_facts(c1)
    env = _make_env(int(seed))
    shim_off = ChatShim(c1, env, net=None, fam=None, enabled=False, use_gate=True)
    t1 = CF.run_chat(b1, xp, i1, s1, shim_off, f1, turns)
    dec_ident = bool(json.dumps(CF._decision_view(t0), sort_keys=True, default=str)
                     == json.dumps(CF._decision_view(t1), sort_keys=True, default=str))
    return {
        "threshold_hash_identical": bool(th0 == th1), "concept_codes_identical": bool(cc0 == cc1),
        "num_neurons_identical": bool(nn0 == nn1), "num_neurons": nn0,
        "decision_transcript_identical": dec_ident,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def run_seed(seed, do_byte_identity=False):
    t_start = time.time()
    xp, _ = get_backend()

    # ---- ONE conversational brain (vocab=V) + the curated host-stored baseline facts (byte-identity/complement) ----
    bridge, comp, idx, snap = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True,
                                                 vocab=V)
    kb_len_before = len(comp.kb)
    _vc, curated_facts = SA._store_facts(comp)             # the 6 curated facts -> comp.kb (host store, the #6 path)
    kb_len_after_store = len(comp.kb)

    env = _make_env(int(seed))

    # ---- the teacher presentation: the brain's OWN e-prop plasticity acquires dax->grass, dog->bone, cat->fish ----
    net, fam, readout_moved = _teach(int(seed), env)
    kb_len_after_teach = len(comp.kb)                      # MUST equal kb_len_after_store (no store-write in teaching)

    facts_all = list(curated_facts) + list(TAUGHT_FACTS)   # the cue bookkeeping (agents/actions/patients for the chat)
    turns = _turns()

    # ---- TREATMENT chat: the trained net + imprinted gate, wired through the shim ----
    shim_treat = ChatShim(comp, env, net=net, fam=fam, enabled=True, use_gate=True)
    tr_treat = CF.run_chat(bridge, xp, idx, snap, shim_treat, facts_all, turns)
    sum_treat = CF._chat_summary(tr_treat)
    recall_treat, recalled = _taught_recall(tr_treat)

    # ---- MATCHED (pre-teaching) BASELINE: same substrate, fresh net + UN-imprinted gate -> taught cues abstain ----
    net_u, fam_u, _ = _teach(int(seed), _make_env(int(seed)), freeze=True)   # net irrelevant; gate below is EMPTY
    fam_empty = ConjunctiveFamiliarityGate(int(seed), d_p=D_P, D=D_FAM)      # NOT imprinted -> every cue novel
    shim_base = ChatShim(comp, env, net=net_u, fam=fam_empty, enabled=True, use_gate=True)
    tr_base = CF.run_chat(bridge, xp, idx, snap, shim_base, facts_all, turns)
    sum_base = CF._chat_summary(tr_base)
    recall_base, _ = _taught_recall(tr_base)

    # ---- FROZEN-READOUT anti-cheat (THE KEY ONE): identical teaching, eprop_lr=0 -> readout un-moved -> the CORRECT
    # patient is not recalled (argmax rides the zero-init attractor 'apple'), so the taught fact is NOT in the chat ----
    net_fz, fam_fz, readout_moved_frozen = _teach(int(seed), _make_env(int(seed)), freeze=True)
    shim_frozen = ChatShim(comp, env, net=net_fz, fam=fam_fz, enabled=True, use_gate=True)
    tr_frozen = CF.run_chat(bridge, xp, idx, snap, shim_frozen, facts_all, turns)
    recall_frozen, _ = _taught_recall(tr_frozen)
    frozen_answers = {r: _probe_answer(tr_frozen, r) for r in TAUGHT}
    frozen_heldout = _heldout_acc(net_fz, env, HEADLINE_REFERENT, PATIENT_WORDS.index(TAUGHT[HEADLINE_REFERENT]))

    # ---- MOAT at chat scale: the reachable untaught cues (dax+chases, wug+eats, ...) abstain ----
    moat_fa, moat_ex = _moat_battery(shim_treat)

    # ---- LESION-gate: (a) conf-only gate-OFF leaks (learned gate load-bearing); (b) lesion collapses the margin ----
    off_shim = ChatShim(comp, env, net=net, fam=fam, enabled=True, use_gate=False, use_conf=True)
    gate_off_fa, _ = _moat_battery(off_shim)
    intact_margin, lesion_margin = _lesion_margin(fam, env)

    # ---- CT1 discrimination (the taught headline genuinely learned + not a constant bias) ----
    heldout = {r: _heldout_acc(net, env, r, PATIENT_WORDS.index(p)) for r, p in TAUGHT.items()}
    maj_words = {r: PATIENT_WORDS[_majority(net, env, r, "eats")[0]] for r in TAUGHT}
    distinct_classes = len(set(maj_words.values()))
    n_facts_correct = sum(1 for r in TAUGHT if maj_words[r] == TAUGHT[r])
    not_constant = bool(distinct_classes >= 2 and n_facts_correct >= 2 and maj_words[HEADLINE_REFERENT] == "grass")
    ct1 = bool(heldout[HEADLINE_REFERENT] > 0.6 and not_constant)

    # ---- single-class control: teaching ONLY dax->grass collapses to a constant grass-bias (contrastive is needed) ----
    sc_net, _sc_fam, _ = _teach(int(seed), _make_env(int(seed)), single_class=True)
    sc_maj = {r: PATIENT_WORDS[_majority(sc_net, env, r, "eats")[0]] for r in TAUGHT}
    sc_n_correct = sum(1 for r in TAUGHT if sc_maj[r] == TAUGHT[r])
    sc_constant_grass = bool(sc_maj["dog"] == "grass" and sc_maj["cat"] == "grass")
    single_class_ok = bool(sc_n_correct <= 1 and n_facts_correct >= 2)

    # ---- mispaired-teacher: a consistent WRONG pairing learns a map wrong on the true targets ----
    mis_net, _mis_fam, _ = _teach(int(seed), _make_env(int(seed)), mispaired=True)
    mis_heldout = {r: _heldout_acc(mis_net, env, r, PATIENT_WORDS.index(p)) for r, p in TAUGHT.items()}
    main_mean = float(np.mean(list(heldout.values())))
    mispaired_mean = float(np.mean(list(mis_heldout.values())))
    mispaired_ok = bool(main_mean > mispaired_mean + 0.15)

    # ---- post-hoc no-confab MOAT teeth (drop 100% of unsupported props over the taught + curated facts) ----
    teeth = CF.posthoc_teeth(shim_treat, facts_all, seed=seed)

    if do_byte_identity:
        with contextlib.redirect_stdout(io.StringIO()):
            bi = byte_identity(int(seed))
    else:
        bi = None

    # ---- ATTRIBUTION (tools.lab): the taught-recall RISE is attributed to the WEIGHT CHANGE by subtracting the
    # frozen-readout control (identical teaching, eprop_lr=0) -- same gate, same substrate, only the readout differs ----
    recall_attrib = attributable_to(
        "taught-fact chat recall from the e-prop weight change (trained vs frozen-readout, identical teaching)",
        float(recall_treat), float(recall_frozen))
    attributable_to("teacher pairing (main vs mispaired mean held-out)", main_mean, mispaired_mean)

    # ---- per-seed GO ----
    recall_ok = bool(recall_treat == len(TAUGHT) and sum_treat["grounded"] > sum_base["grounded"])
    moat_ok = bool(moat_fa == 0)
    ood_ok = bool(sum_treat["ood_abstained"] == sum_treat["ood_turns"]
                  and sum_base["ood_abstained"] == sum_base["ood_turns"]
                  and sum_treat["ungrounded_word_total"] == 0 and sum_base["ungrounded_word_total"] == 0
                  and sum_treat["confabulated"] == 0 and sum_base["confabulated"] == 0)
    posthoc_ok = bool(abs(teeth["unsupported_drop_rate"] - 1.0) < 1e-9 and teeth["unsupported_props"] > 0
                      and abs(teeth["supported_keep_rate"] - 1.0) < 1e-9)
    frozen_ok = bool(readout_moved_frozen <= 1e-3 and recall_frozen == 0 and frozen_heldout <= CHANCE + 0.10)
    kb_unchanged = bool(kb_len_after_teach == kb_len_after_store)
    untrained_ok = bool(recall_base == 0)
    lesion_ok = bool(gate_off_fa > 0 and lesion_margin < intact_margin - 0.30)

    seed_go = bool(recall_ok and moat_ok and ood_ok and posthoc_ok and frozen_ok and kb_unchanged
                   and untrained_ok and lesion_ok and mispaired_ok and single_class_ok and ct1
                   and (bi is None or bi["decision_transcript_identical"]))

    return {
        "seed": int(seed), "elapsed_s": round(time.time() - t_start, 1),
        "vocab_size": len(V), "kb_len_before": kb_len_before, "kb_len_after_store": kb_len_after_store,
        "kb_len_after_teach": kb_len_after_teach, "readout_moved": readout_moved,
        "recall_treatment": recall_treat, "recalled_referents": recalled, "recall_baseline": recall_base,
        "grounded_treatment": sum_treat["grounded"], "grounded_baseline": sum_base["grounded"],
        "recall_attributable_to_weight_change": recall_attrib,
        "chat_treatment_summary": sum_treat, "chat_baseline_summary": sum_base,
        "taught_probe_answers": {r: _probe_answer(tr_treat, r) for r in TAUGHT},
        "moat_false_accepts": moat_fa, "moat_examples": moat_ex,
        "frozen_readout_moved": readout_moved_frozen, "recall_frozen": recall_frozen,
        "frozen_probe_answers": frozen_answers, "frozen_heldout": frozen_heldout,
        "gate_off_false_accepts": gate_off_fa, "intact_margin": intact_margin, "lesion_margin": lesion_margin,
        "heldout": heldout, "majority_words": maj_words, "not_constant": not_constant,
        "distinct_classes": distinct_classes, "n_facts_correct": n_facts_correct,
        "single_class_majority": sc_maj, "sc_n_correct": sc_n_correct, "sc_constant_grass": sc_constant_grass,
        "main_mean_heldout": main_mean, "mispaired_mean_heldout": mispaired_mean, "mispaired_heldout": mis_heldout,
        "posthoc_teeth": teeth, "byte_identity": bi,
        "gate": {
            "recall_rises": recall_ok, "moat_0_false_accepts": moat_ok, "ood_abstains_no_confab": ood_ok,
            "posthoc_drop_100pct": posthoc_ok, "frozen_readout_no_recall": frozen_ok, "kb_unchanged": kb_unchanged,
            "untrained_baseline_silent": untrained_ok, "lesion_gate_load_bearing": lesion_ok,
            "mispaired_teacher": mispaired_ok, "single_class_control": single_class_ok, "ct1_discrimination": ct1,
        },
        "seed_go": seed_go,
        "transcript_treatment": tr_treat,
    }


def build_verdict(recs, go):
    def _allmin(fn):
        vals = [fn(r) for r in recs]
        return (min(vals) if vals else None)
    r0 = recs[0]
    bi = next((r["byte_identity"] for r in recs if r.get("byte_identity")), None)
    v = Verdict("INTEGRATION #7 plasticity-learned-facts-into-live-chat (K=%d joint, %d seeds)"
                % (len(TAUGHT), len(recs)), chance=CHANCE)
    v.require("all seeds GO", int(sum(1 for r in recs if r["seed_go"])), expect=len(recs))
    v.require("taught-recall == K all seeds (each taught cue answered)",
              int(_allmin(lambda r: r["recall_treatment"]) or 0), expect=len(TAUGHT))
    v.control("taught-recall: trained vs frozen-readout (identical teaching)",
              r0["recall_treatment"], r0["recall_frozen"], min_separation=0.0)
    v.control("grounded replies: treatment vs pre-teaching baseline",
              r0["grounded_treatment"], r0["grounded_baseline"], min_separation=0.0)
    v.require("moat 0 false-accepts (untaught cues, all seeds)",
              int(sum(r["moat_false_accepts"] for r in recs)), expect=0)
    v.require("chat confab == 0 (treatment + baseline, all turns)",
              int(sum(r["chat_treatment_summary"]["confabulated"] + r["chat_baseline_summary"]["confabulated"]
                      for r in recs)), expect=0)
    v.require("posthoc teeth drop 100% of unsupported props",
              float(_allmin(lambda r: r["posthoc_teeth"]["unsupported_drop_rate"]) or 0.0), expect=1.0)
    v.require("FROZEN-READOUT: 0 taught-recall (content rode the weight change)",
              int(max(r["recall_frozen"] for r in recs)), expect=0)
    v.require("kb unchanged by teaching (no store-write, all seeds)",
              bool(all(r["kb_len_after_teach"] == r["kb_len_after_store"] for r in recs)), expect=True)
    v.require("LESION-gate: conf-only gate-OFF leaks (>0 false-accepts)",
              int(min(r["gate_off_false_accepts"] for r in recs)), expect=lambda m: m > 0)
    v.reaches("LESION collapses the novelty margin (first seed)",
              before=r0["intact_margin"], after=r0["lesion_margin"])
    v.control("mispaired-teacher (main vs mispaired mean held-out)",
              r0["main_mean_heldout"], r0["mispaired_mean_heldout"], min_separation=0.15)
    v.control("single-class control (contrastive facts vs single-class facts)",
              r0["n_facts_correct"], r0["sc_n_correct"], min_separation=0.0)
    v.floor("main mean held-out vs chance (all seeds)", float(_allmin(lambda r: r["main_mean_heldout"])), CHANCE)
    if bi is not None:
        v.require("byte-identity: shim-off decision transcript identical to #6 default",
                  bool(bi["decision_transcript_identical"]), expect=True)
        v.require("byte-identity: substrate threshold hash identical", bool(bi["threshold_hash_identical"]),
                  expect=True)
    v.disabled("spiking-generator MOUTH (GPU/torch)",
               "CPU eval; the grounded CONTENT is the LEARNED read (what the mouth would render), not the mouth")
    v.disabled("ONE merged bridge",
               "OnBridgeEpropNet builds its OWN co-resident SimulationBridge; the merge is the named next step")
    v.disabled("spiking familiarity gate (v320)",
               "the source-monitor is a numpy anti-Hebbian projector; the spiking v320 gate is the swap-in target")
    return v.decide(go=bool(go), verbose=False)


def main():
    ap = argparse.ArgumentParser(description="INTEGRATION #7: plasticity-learned facts into the live chat.")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--byte-identity", choices=["auto", "on", "off"], default="auto",
                    help="auto=first seed only (default); on=every seed; off=never")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap-first single-seed smoke assertions (recall/abstain/lesion/byte-identity), then exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.replace(",", " ").split()]

    if a.smoke:
        return _smoke(seeds[0])

    def _want_bi(i):
        return {"on": True, "off": False, "auto": (i == 0)}[a.byte_identity]

    print("[INTEGRATION #7] plasticity-learned facts -> live chat | vocab=%d | TAUGHT=%s" % (len(V), TAUGHT),
          flush=True)
    recs = []
    for i, s in enumerate(seeds):
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_seed(s, do_byte_identity=_want_bi(i))
        recs.append(r)
        print("  [seed %d] recall base->treat %d->%d/%d (frozen %d) | grounded base->treat %d->%d | moat_fa=%d | "
              "held-out dax=%.2f maj=%s | frozen-moved=%.4f | mispaired %.2f<main %.2f | lesion %.2f->%.2f | "
              "gate-off FA=%d | GO=%s (%.1fs)"
              % (s, r["recall_baseline"], r["recall_treatment"], len(TAUGHT), r["recall_frozen"],
                 r["grounded_baseline"], r["grounded_treatment"], r["moat_false_accepts"],
                 r["heldout"][HEADLINE_REFERENT], r["majority_words"], r["frozen_readout_moved"],
                 r["mispaired_mean_heldout"], r["main_mean_heldout"], r["intact_margin"], r["lesion_margin"],
                 r["gate_off_false_accepts"], r["seed_go"], r["elapsed_s"]), flush=True)
        if r["byte_identity"] is not None:
            print("    byte-identity(shim-off vs #6 default): %s" % r["byte_identity"], flush=True)

    n_go = sum(1 for r in recs if r["seed_go"])
    go = bool(n_go == len(recs) and len(recs) > 0)
    decided = build_verdict(recs, go)

    print("\n  AGGREGATE (%d seeds): taught-recall treat=%s frozen=%s | grounded delta=%s | moat_fa=%s | "
          "held-out(dax)=%s"
          % (len(recs), [r["recall_treatment"] for r in recs], [r["recall_frozen"] for r in recs],
             [r["grounded_treatment"] - r["grounded_baseline"] for r in recs],
             [r["moat_false_accepts"] for r in recs],
             [round(r["heldout"][HEADLINE_REFERENT], 2) for r in recs]), flush=True)
    print("  VERDICT: %s -- %d/%d seeds. The brain answers about facts it learned BY ITS OWN PLASTICITY "
          "(taught-recall rises via the weight change; frozen-readout recalls 0), the learned familiarity gate holds "
          "the no-confab moat at chat scale (0 false-accepts), and the shim is byte-identical off." %
          ("GO" if go else "PARTIAL/NEGATIVE", n_go, len(recs)), flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        payload = {"verdict": "GO" if go else "PARTIAL", "verdict_earned": decided["status"],
                   "n_go": n_go, "n_seeds": len(recs), "seeds": seeds, "K_taught": len(TAUGHT), "taught": TAUGHT,
                   "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
                   "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
                   "byte_identity": recs[0]["byte_identity"], "per_seed": recs}
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print("  [saved] %s" % a.out, flush=True)
    return 0 if go else 1


def _smoke(seed):
    """Cheap-first single-seed smoke (mouth-free): (i) each taught cue answers its own patient in the chat, (ii)
    dax+chases/wug+eats + OOD turns abstain, (iii) fam.lesion() collapses the margin (confab-abstain rides the
    learned projector), (iv) enabled=False reproduces the #6 default transcript. Only then is the 6-seed worth it."""
    print("[SMOKE seed %d] building + teaching ..." % seed, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_seed(int(seed), do_byte_identity=True)
    ok = True

    def _check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)

    _check("(i) each taught cue recalled in chat", r["recall_treatment"] == len(TAUGHT),
           "recall=%d/%d recalled=%s answers=%s" % (r["recall_treatment"], len(TAUGHT),
                                                    r["recalled_referents"], r["taught_probe_answers"]))
    _check("(i) grounded rises vs baseline", r["grounded_treatment"] > r["grounded_baseline"],
           "%d -> %d" % (r["grounded_baseline"], r["grounded_treatment"]))
    _check("(ii) untaught cues + OOD abstain (0 confab, 0 false-accept)",
           r["moat_false_accepts"] == 0 and r["chat_treatment_summary"]["confabulated"] == 0
           and r["chat_treatment_summary"]["ood_abstained"] == r["chat_treatment_summary"]["ood_turns"],
           "moat_fa=%d confab=%d ood=%d/%d" % (r["moat_false_accepts"], r["chat_treatment_summary"]["confabulated"],
                                               r["chat_treatment_summary"]["ood_abstained"],
                                               r["chat_treatment_summary"]["ood_turns"]))
    _check("(ii) FROZEN-READOUT recalls 0 taught facts", r["recall_frozen"] == 0,
           "frozen answers=%s moved=%.4f" % (r["frozen_probe_answers"], r["frozen_readout_moved"]))
    _check("(iii) lesion collapses the novelty margin + conf-only gate-OFF leaks",
           r["lesion_margin"] < r["intact_margin"] - 0.30 and r["gate_off_false_accepts"] > 0,
           "margin %.2f -> %.2f | gate-off FA=%d" % (r["intact_margin"], r["lesion_margin"],
                                                     r["gate_off_false_accepts"]))
    _check("(iv) enabled=False == #6 default transcript",
           r["byte_identity"]["decision_transcript_identical"], str(r["byte_identity"]))
    _check("kb unchanged by teaching (no store-write)", r["kb_len_after_teach"] == r["kb_len_after_store"],
           "%d == %d" % (r["kb_len_after_teach"], r["kb_len_after_store"]))
    print("\n[SMOKE] %s (seed GO=%s)" % ("ALL PASS" if ok else "FAILURES ABOVE", r["seed_go"]), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
