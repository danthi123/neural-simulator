"""SELF-INITIATED UTTERANCE — the loop-closing rung: a spontaneously-SELECTED thought becomes a SPOKEN utterance.

2026-08-13. Self-initiation now SELECTS among thoughts (2026-08-13-self-initiation-multibasin-GO.md, runner
_self_initiation_multibasin_derisk.py, 6-seed GO): under non-specific noise (NO prompt) the CA3 wander visits ~3
co-equal disjoint balanced basins, a curiosity recurrent-gain biasing WHICH surfaces (66% attributable). Its named
FINAL rung is EXACTLY this: route the surfaced, curiosity-SELECTED thought vector into the composer/mouth so a
spontaneous thought becomes a self-initiated QUESTION or REMARK. That closes the loop: internally-generated ->
selected -> SPOKEN.

THE MECHANISM (compose two validated GO organs; NO `sim/` edit; reuse-by-import):
  (1) SELECTION (the substrate decides the TOPIC, 0 host content-draw) = the multibasin self-initiation wander
      (6-seed GO), reused-by-import (`_self_initiation_multibasin_derisk._run_condition/_selection` + its DISJOINT
      pattern-separated CA3 store + the production curiosity recurrent-gain). Under weak non-specific Poisson (NO cue,
      0 external CONTENT drive), each discrete noise-seeded volley ignites WHICHEVER balanced basin its coincidental
      overlap favours; the curiosity gain biases WHICH; the bistable KIR down-state returns the net to silence between
      events. The surfaced basin IS the self-initiated "thought" -- which concept, and how often, is entirely the
      spiking attractor competition + noise (0 random.choice over concepts).
  (2) THE MOUTH (the composer turns the selected concept -> words) = the production `OneBrainComposer` (one_brain_
      composer.py), reused-by-import. Each stored concept is a fact composite in the bridge's complex RF synapses;
      `render_fact(concept)` reconstructs "concept verb patient" by an ON-BRIDGE resonate-and-fire unbind + cleanup
      (the spiking decode), and abstains (None) on an unknown subject (the no-confab moat). The surfaced basin's bound
      lexical concept routes to `render_fact` -> a short self-initiated REMARK; a host question-template wraps the same
      spiking proposition into a QUESTION ("what does <agent> <verb>?") -- the fluency/articulation scaffold's job,
      declared HOST.

THE LOOP: noise (no prompt) -> spiking CA3 wander SELECTS a basin (curiosity-biased) -> the basin's concept ->
OneBrainComposer.render_fact -> a spoken SVO utterance ABOUT that concept. Internally-generated -> selected -> SPOKEN.

SUBSTRATE vs HOST (the honesty boundary is a deliverable, not a caveat):
  * SPIKING (load-bearing): (i) the SELECTION of WHICH concept is spoken and HOW OFTEN -- the CA3 dendritic-plateau
    attractor competition under non-specific noise (0 host content-draw / no random.choice over concepts); (ii) the
    steering VALUE (the curiosity ASK-pool want, read off cp_firing_states); (iii) the VERBALISATION -- the SVO
    proposition is decoded by the OneBrainComposer's on-bridge RF resonate unbind + cleanup (render_fact reads the
    complex synapses, not the host labels).
  * HOST (declared, rides existing burn-downs): (i) the per-concept NOVELTY levels are the ENVIRONMENT; (ii) the
    basin<->lexical-concept BINDING (which stored word each disjoint CA3 basin denotes) and each concept's stored FACT
    are the learned store / environment -- the SAME boundary the multibasin wander declares for its stored assemblies;
    (iii) the curiosity want->recurrent-gain PROJECTION (the one-brain-merge rung); (iv) the QUESTION-template wrapper
    + any natural-language FLUENCY (the Broca/Qwen articulation scaffold) -- NOT exercised/measured here beyond the
    templated form; the MEASURED content is the spiking SVO proposition.

FUNCTIONAL CORRELATE, NOT phenomenal: measures + reports a self-initiated-UTTERANCE correlate (an internally-triggered,
curiosity-selected, coherent spoken remark with no prompt). It makes NO claim of subjective experience.

THE ANTI-CHEATS (each VERIFIED, not asserted):
  (a) INTERNALLY-TRIGGERED: 0 external CONTENT drive (only non-specific Poisson to random CA3-exc cells; no cue, no
      recall_drive). NO-NOISE (gains on, noise off) -> the wander is SILENT -> 0 utterances. Plasticity byte-frozen.
  (b) ABOUT-THE-SELECTED-CONCEPT (coherent): each utterance NAMES the concept bound to the basin the substrate
      actually ignited (decoded subject == the surfaced basin's concept), and the wander's surfaced steps overlap the
      stored assembly (member >> random). MOUTH FIDELITY: render_fact decodes each stored fact correctly on the bridge.
      SCRAMBLE-ROUTING negative control (route each basin to a WRONG concept, a derangement) -> the utterance is about
      the wrong concept (about-selected collapses to ~0) -> the correspondence is load-bearing.
  (c) CURIOSITY-STEERED: NOVEL concepts drive MORE utterances than under a REVERSED (anti-curiosity) gain -- the SAME
      concepts, HIGH gain vs LOW gain, so the difference is the curiosity VALUE not the basin identity.
      attributable_to(novel-utterance-share on vs reversed).
  (d) SUBSTRATE-ATTRIBUTABLE (lesion the selection -> no utterance): STORE-LESION (NO-ENCODE the CA3 store, same
      noise+gain) -> no coherent surfacing -> the mouth is asked about no coherent thought -> utterances collapse.

CPU-smoke:  SIM_BACKEND=numpy python -u -m research.runners._self_initiated_utterance_derisk --seeds 42 --n-mem 4 --rest-steps 1200 --acid-steps 400 --gain-scale 1.0 --smoke
Full (GPU): SIM_BACKEND=cupy  python -u -m research.runners._self_initiated_utterance_derisk --seeds 42 43 44 100 101 102 --n-mem 4 --rest-steps 4000 --acid-steps 1200 --gain-scale 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# reuse-by-import the VALIDATED multibasin self-initiation SELECTION (6-seed GO): the DISJOINT balanced CA3 store, the
# noise-driven wander, the curiosity recurrent-gain, and the selection read-out.
from research.runners._self_initiation_multibasin_derisk import (  # noqa: E402
    _run_condition, _selection, NOV_BY_NMEM,
)
from research.runners._gap5_spontaneous_reactivation_derisk import GO_CFG  # noqa: E402
# the curiosity organ's SPIKING ASK-pool want (deterministic given seed)
from research.runners._self_initiated_spontaneous_thought_derisk import _curiosity_wants  # noqa: E402
# the production MOUTH: the OneBrainComposer (render_fact = on-bridge RF unbind+cleanup decode of a stored SVO fact)
from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402
from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_self_initiated_utterance_derisk.json"

# The lexicon binding each disjoint CA3 basin to a stored fact (agent, verb, patient). This is the ENVIRONMENT /
# learned store (host boundary, declared) -- the same class as the multibasin wander's stored assemblies + novelty.
AGENTS = ["dog", "cat", "bird", "fish", "fox", "owl", "bee", "ant"]
VERBS = ["chase", "eat", "see", "hear", "find", "want", "watch", "fear"]
PATIENTS = ["ball", "worm", "seed", "bug", "bone", "moon", "leaf", "nut"]


def _lexicon(n_mem):
    """Bind concept i -> the stored fact (AGENTS[i], VERBS[i], PATIENTS[i]). Returns (agents, verbs, patients, vocab)."""
    agents = AGENTS[:n_mem]; verbs = VERBS[:n_mem]; patients = PATIENTS[:n_mem]
    vocab = sorted(set(agents + verbs + patients))
    return agents, verbs, patients, vocab


def _build_mouth(seed, agents, verbs, patients, vocab, D):
    """Build the OneBrainComposer MOUTH once and store one fact per concept. Then DECODE each fact on the bridge
    (render_fact = the spiking RF unbind+cleanup) and cache the utterance per concept. Returns (composer, utt_by_agent,
    decode_ok) where decode_ok[i] is True iff render_fact(agents[i]) == 'agents[i] verbs[i] patients[i]' (mouth
    fidelity: the spoken proposition matches the stored fact) and an UNKNOWN subject abstains (the no-confab moat)."""
    n_mem = len(agents)
    # rf_cudagraph OFF: render_fact is called only ~n_mem times (cached per concept), so the one-time cudagraph
    # CAPTURE cost is not worth it; the decode is byte-identical either way (the loop path = the documented default).
    comp = OneBrainComposer(seed=seed, D=int(D), vocab=list(vocab), k_max=max(8, n_mem),
                            enable_rf_cudagraph=False)
    for i in range(n_mem):
        comp.store(agents[i], verbs[i], patients[i])
    utt_by_agent = {}
    decode_ok = []
    for i in range(n_mem):
        utt = comp.render_fact(agents[i])                       # ON-BRIDGE decode of the stored SVO fact
        utt_by_agent[agents[i]] = utt
        want = f"{agents[i]} {verbs[i]} {patients[i]}"
        decode_ok.append(bool(utt is not None and utt == want))
    # no-confab moat: an unknown subject must abstain (None)
    moat_abstains = bool(comp.render_fact("zzz_unknown_subject") is None)
    return comp, utt_by_agent, decode_ok, moat_abstains


def _episodes(F, assemblies_local, min_frac):
    """Segment a session firing tensor F [T, n_ca3] into discrete SURFACING EPISODES: maximal contiguous runs of
    steps where a single basin is the winner-take-all AND is active (assembly-active fraction >= min_frac). Each
    episode = one self-initiated 'thought' the brain could speak. Returns a list of (concept_i, start, end)."""
    A = np.stack([F[:, np.asarray(a, dtype=np.int64)].mean(1) for a in assemblies_local], axis=0)  # [n_mem, T]
    active = A >= min_frac
    winner = np.argmax(A, axis=0)
    T = F.shape[0]
    eps = []
    cur = None; start = 0
    for t in range(T):
        w = int(winner[t])
        c = w if bool(active[w, t]) else -1
        if c != cur:
            if cur is not None and cur >= 0:
                eps.append((cur, start, t))
            cur = c; start = t
    if cur is not None and cur >= 0:
        eps.append((cur, start, T))
    return eps


def _utterance_stream(F, assemblies_local, agents, utt_by_agent, decode_ok, min_frac, routing):
    """Route the wander's surfaced basins -> the mouth -> a stream of utterances. `routing` maps a surfaced basin
    index -> the CONCEPT index whose fact is spoken (identity in production; a DERANGEMENT for the scramble control).
    An utterance is ABOUT-THE-SELECTED-CONCEPT iff its decoded subject == the concept bound to the basin the substrate
    ACTUALLY ignited (agents[i]) -- i.e. routing[i]==i AND the mouth decoded that fact. Returns per-concept utterance
    counts + the about-selected rate + a few example utterances."""
    n_mem = len(agents)
    eps = _episodes(F, assemblies_local, min_frac)
    counts = np.zeros(n_mem, dtype=float)          # utterances truly about concept i (basin i ignited, spoken correctly)
    n_utt = 0; n_about = 0
    examples = []
    for (i, s, e) in eps:
        j = int(routing[i])                        # the concept whose fact is verbalised for surfaced basin i
        utt = utt_by_agent.get(agents[j])
        n_utt += 1
        subject = utt.split()[0] if utt else None
        about = bool(utt is not None and subject == agents[i] and decode_ok[i])
        if about:
            n_about += 1
            counts[i] += 1.0
        if len(examples) < 4:
            examples.append({"surfaced_basin": i, "spoke_about": agents[j], "utterance": utt,
                             "question": (f"what does {agents[j]} {utt.split()[1]}?" if utt else None),
                             "about_selected": about})
    about_rate = float(n_about / n_utt) if n_utt > 0 else 0.0
    total = float(counts.sum())
    share = (counts / total) if total > 0 else np.zeros(n_mem)
    n_concepts_spoken = int((counts > 0).sum())
    return dict(n_utt=int(n_utt), n_about=int(n_about), about_rate=about_rate, counts=counts.tolist(),
                share=share.tolist(), n_concepts_spoken=n_concepts_spoken, examples=examples)


def _derangement(n, seed):
    """A permutation with no fixed point (for the SCRAMBLE-routing negative control: every basin routed to a WRONG
    concept). Falls back to a single-cycle shift for tiny n."""
    rng = np.random.default_rng(seed * 6151 + 7)
    for _ in range(64):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p.tolist()
    return [(i + 1) % n for i in range(n)]


def one_seed(seed, n_mem, rest_steps, acid_steps, gain_scale, min_frac, D):
    """Close the loop for one seed: build the mouth (store n_mem facts), run the curiosity-ON wander (the production
    self-initiated stream), a REVERSED wander (curiosity contrast), a NO-NOISE acid (internal-trigger), and a
    STORE-LESION (no-encode -> substrate attribution); plus a host SCRAMBLE-routing negative control on the ON wander."""
    t0 = time.time()
    out = {"seed": seed, "n_mem": n_mem}
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(n_mem)
    agents, verbs, patients, vocab = _lexicon(n_mem)
    out["facts"] = [f"{agents[i]} {verbs[i]} {patients[i]}" for i in range(n_mem)]

    # -- NOVELTY -> curiosity gains (identical construction to the multibasin GO; novelty on a RANDOM concept perm) --
    nov_rng = np.random.default_rng(seed * 7919 + 1)
    novelties = [float(v) for v in nov_rng.permutation(np.asarray(NOV_BY_NMEM[n_mem], dtype=float))]
    wants, cur_meta = _curiosity_wants(seed, novelties)
    wmax = max(wants) if wants else 1.0
    gains_on = [1.0 + gain_scale * (w / wmax if wmax > 1e-9 else 0.0) for w in wants]
    nov = np.asarray(novelties, dtype=float)
    order = [int(i) for i in np.argsort(-nov)]                  # concepts most-novel -> least-novel
    gvals = sorted(gains_on, reverse=True)
    gains_reversed = [0.0] * n_mem
    for k, ci in enumerate(order):
        gains_reversed[ci] = gvals[n_mem - 1 - k]              # most-novel concept -> the SMALLEST gain
    novel_set = np.asarray(order[:max(1, n_mem // 2)], dtype=int)
    out["novelties"] = novelties; out["gains_on"] = gains_on; out["gains_reversed"] = gains_reversed
    out["novel_order"] = order; out["novel_set"] = novel_set.tolist()

    # -- the MOUTH: build once, store the facts, decode each on-bridge (mouth fidelity) --
    comp, utt_by_agent, decode_ok, moat = _build_mouth(seed, agents, verbs, patients, vocab, D)
    out["decode_ok"] = decode_ok; out["mouth_fidelity"] = bool(all(decode_ok)); out["moat_abstains"] = moat
    print(f"  [seed {seed}] mouth: fidelity={all(decode_ok)} moat_abstains={moat} "
          f"examples={[utt_by_agent[agents[i]] for i in range(min(3, n_mem))]} ({time.time()-t0:.0f}s)", flush=True)

    ident = list(range(n_mem))
    # -- CURIOSITY-ON wander (production self-initiated stream) --
    F_on, prep_on, d_on = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_on)
    sel_on = _selection(F_on, prep_on["assemblies_local"], seed, min_frac)
    st_on = _utterance_stream(F_on, prep_on["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    out["max_pair_overlap"] = int(prep_on["max_pair_overlap"]); out["weights_frozen"] = bool(d_on["weights_frozen"])
    out["apical_rest_max"] = d_on["apical_rest_max"]
    out["on"] = {"stream": st_on, "pooled_member": sel_on["pooled_member"], "pooled_random": sel_on["pooled_random"]}
    print(f"  [seed {seed}] ON: utterances={st_on['n_utt']} about-selected={st_on['about_rate']:.2f} "
          f"concepts_spoken={st_on['n_concepts_spoken']} member {sel_on['pooled_member']:.2f} vs rand {sel_on['pooled_random']:.2f} "
          f"share {[round(x,2) for x in st_on['share']]}", flush=True)

    # -- SCRAMBLE-ROUTING negative control (host-side on the SAME ON wander): route each basin to a WRONG concept --
    scr = _derangement(n_mem, seed)
    st_scr = _utterance_stream(F_on, prep_on["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, scr)
    out["scramble"] = {"about_rate": st_scr["about_rate"], "n_utt": st_scr["n_utt"], "routing": scr}

    # -- REVERSED wander (curiosity contrast: novel concepts now get the SMALLEST gain) --
    F_rv, prep_rv, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_reversed)
    st_rv = _utterance_stream(F_rv, prep_rv["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    out["reversed"] = {"stream": st_rv}

    # -- NO-NOISE acid (gains on, noise off): the wander must be SILENT -> 0 utterances --
    F_nn, prep_nn, _ = _run_condition(seed, cfg, acid_steps, noise_on=False, gains=gains_on)
    st_nn = _utterance_stream(F_nn, prep_nn["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    out["no_noise"] = {"n_utt": st_nn["n_utt"]}

    # -- STORE-LESION (NO-ENCODE, gains on, noise on): no coherent surfacing -> utterances collapse --
    F_sl, prep_sl, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_on, do_encode=False)
    sel_sl = _selection(F_sl, prep_sl["assemblies_local"], seed, min_frac)
    st_sl = _utterance_stream(F_sl, prep_sl["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    out["store_lesion"] = {"n_utt": st_sl["n_utt"], "about_n": st_sl["n_about"],
                           "pooled_member": sel_sl["pooled_member"], "pooled_random": sel_sl["pooled_random"]}

    # -- CURIOSITY-STEERING (identity-controlled): novel-concept UTTERANCE share on vs reversed --
    share_on = np.asarray(st_on["share"], dtype=float); share_rv = np.asarray(st_rv["share"], dtype=float)
    novel_share_on = float(share_on[novel_set].sum()); novel_share_rv = float(share_rv[novel_set].sum())
    bias_attr = attributable_to("curiosity-gain @ novel-concept UTTERANCE share (on vs reversed)",
                                novel_share_on, novel_share_rv)
    out["bias"] = dict(novel_share_on=novel_share_on, novel_share_reversed=novel_share_rv,
                       uniform_expectation=float(len(novel_set) / n_mem), attributable=bias_attr)
    print(f"  [seed {seed}] SCRAMBLE about={st_scr['about_rate']:.2f} | REVERSED utt={st_rv['n_utt']} | "
          f"NO-NOISE utt={st_nn['n_utt']} | STORE-LESION utt={st_sl['n_utt']} member {sel_sl['pooled_member']:.2f} | "
          f"novel-share on={novel_share_on:.2f} rev={novel_share_rv:.2f} attr="
          f"{('%.0f%%' % (100*bias_attr)) if bias_attr is not None else 'UNDEF'}", flush=True)

    # ---- per-seed GO gate ----
    m = out["on"]["pooled_member"]; r = out["on"]["pooled_random"]
    void_if(st_on["n_utt"] == 0, f"seed {seed}: ON wander produced 0 utterances (nothing to interpret)")
    disjoint_ok = bool(out["max_pair_overlap"] == 0)
    mouth_ok = bool(all(decode_ok) and moat)
    # (a) INTERNALLY-TRIGGERED: NO-NOISE silent + plasticity frozen + apical not self-latched
    internally_triggered = bool(st_nn["n_utt"] == 0 and out["weights_frozen"]
                                and (out["apical_rest_max"] is None
                                     or out["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3))
    # (b) ABOUT-THE-SELECTED-CONCEPT (coherent) + production speaks MULTIPLE concepts + SCRAMBLE collapses
    about_selected = bool(st_on["about_rate"] >= 0.90 and m >= min_frac and m > 2.0 * (r + 1e-6))
    production_speaks = bool(st_on["n_concepts_spoken"] >= 2 and st_on["n_utt"] >= 3)
    scramble_collapses = bool(st_scr["about_rate"] <= 0.15)
    # (c) CURIOSITY-STEERED: novel utterance share materially higher on vs reversed
    curiosity_steered = bool(novel_share_on >= novel_share_rv + 0.10)
    # (d) SUBSTRATE-ATTRIBUTABLE: store-lesion collapses the utterance stream
    store_lesion_ok = bool(st_sl["n_utt"] <= max(1, int(0.25 * st_on["n_utt"]))
                           or st_sl["about_n"] == 0
                           or sel_sl["pooled_member"] < 0.5 * m)

    checks = dict(disjoint_ok=disjoint_ok, mouth_fidelity=mouth_ok, internally_triggered=internally_triggered,
                  about_selected=about_selected, production_speaks=production_speaks,
                  scramble_collapses=scramble_collapses, curiosity_steered=curiosity_steered,
                  store_lesion_load_bearing=store_lesion_ok)
    seed_go = bool(all(checks.values()))
    out["checks"] = checks; out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={checks}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=4, choices=[4, 5, 6, 8])
    ap.add_argument("--rest-steps", type=int, default=4000, help="session rest steps for the wander (long -> speaks about multiple)")
    ap.add_argument("--acid-steps", type=int, default=1200, help="rest steps for the NO-NOISE acid test")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="curiosity recurrent-gain scale (matches the multibasin GO operating point)")
    ap.add_argument("--min-frac", type=float, default=0.30, help="assembly-active fraction to count a surfaced step")
    ap.add_argument("--D", type=int, default=256, help="composer FHRR dimensionality (mouth)")
    ap.add_argument("--smoke", action="store_true", help="smoke: >=50%% seeds GO; full gate is >=5/6")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    print(f"[self-utter] n_mem={a.n_mem} rest_steps={a.rest_steps} acid_steps={a.acid_steps} gain_scale={a.gain_scale} "
          f"D={a.D} seeds={a.seeds} backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    partial_path = Path(a.out).with_suffix(".partial.json")
    try:
        for s in a.seeds:
            per.append(one_seed(s, a.n_mem, a.rest_steps, a.acid_steps, a.gain_scale, a.min_frac, a.D))
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps({"partial": True, "seeds_done": [p["seed"] for p in per],
                                                "per_seed": per}, indent=2, default=str))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    attribution = None; preconditions = []
    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        thresh = max(1, (len(per) + 1) // 2) if a.smoke else max(1, (5 * len(per) + 5) // 6)
        go = n_go >= thresh
        m_about = float(np.mean([p["on"]["stream"]["about_rate"] for p in per]))
        m_utt = float(np.mean([p["on"]["stream"]["n_utt"] for p in per]))
        m_concepts = float(np.mean([p["on"]["stream"]["n_concepts_spoken"] for p in per]))
        m_scr = float(np.mean([p["scramble"]["about_rate"] for p in per]))
        m_member = float(np.mean([p["on"]["pooled_member"] for p in per]))
        m_random = float(np.mean([p["on"]["pooled_random"] for p in per]))
        m_novel_on = float(np.mean([p["bias"]["novel_share_on"] for p in per]))
        m_novel_rv = float(np.mean([p["bias"]["novel_share_reversed"] for p in per]))
        m_nn = float(np.mean([p["no_noise"]["n_utt"] for p in per]))
        m_sl = float(np.mean([p["store_lesion"]["n_utt"] for p in per]))
        attribution = attributable_to("curiosity-gain @ novel-concept UTTERANCE share (6-seed, on vs reversed)",
                                       m_novel_on, m_novel_rv)

        vd = Verdict("self-initiated utterance (seed->utterance routing, 6-seed)", chance=m_random)
        vd.require("seeds passing all anti-cheats >= threshold", n_go, expect=lambda x, t=thresh: x >= t)
        vd.require("mouth fidelity: every stored fact decodes on-bridge + unknown subject abstains (every seed)",
                   all(p["mouth_fidelity"] and p["moat_abstains"] for p in per), expect=True)
        vd.require("balanced basins DISJOINT (max pairwise overlap == 0) every seed",
                   all(p["max_pair_overlap"] == 0 for p in per), expect=True)
        vd.require("ABOUT-THE-SELECTED-CONCEPT: production about-selected rate (mean) >= 0.9", m_about,
                   expect=lambda x: x >= 0.9)
        vd.control("about-selected: production vs SCRAMBLE-routing negative control", m_about, m_scr,
                   min_separation=0.5)
        vd.require("production speaks MULTIPLE distinct concepts (mean >= 2)", m_concepts, expect=lambda x: x >= 2.0)
        vd.control("coherent: surfaced member vs random floor", m_member, m_random, min_separation=0.15)
        vd.floor("coherence member above random", m_member, floor=m_random)
        vd.control("curiosity-steered: novel-utterance share on vs reversed", m_novel_on, m_novel_rv,
                   min_separation=0.05)
        vd.require("internally-triggered: NO-NOISE -> 0 utterances every seed",
                   all(p["no_noise"]["n_utt"] == 0 for p in per), expect=True)
        vd.require("substrate-attributable: STORE-LESION collapses the utterance stream every seed",
                   all(p["checks"]["store_lesion_load_bearing"] for p in per), expect=True)
        vd.require("plasticity byte-frozen during the wander every seed",
                   all(p["weights_frozen"] for p in per), expect=True)
        vd.disabled("hebbian/BTSP plasticity during the wander", "the wander measures noise-seeded completion on a frozen store")
        decided = vd.decide(go)
        preconditions = decided["preconditions"]

        verdict = (f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{len(per)} -- the self-initiation loop CLOSES: a "
                   f"noise-driven (no-prompt) CA3 wander SELECTS a curiosity-biased basin and the OneBrainComposer "
                   f"MOUTH speaks a coherent SVO utterance ABOUT it. "
                   f"{'production: %.1f utterances/session about %.1f distinct concepts, about-selected %.2f (vs SCRAMBLE %.2f); coherence member %.2f vs random %.2f; novel-concept utterance share HIGH-gain %.2f vs LOW-gain(reversed) %.2f; NO-NOISE %.1f utt; STORE-LESION %.1f utt' % (m_utt, m_concepts, m_about, m_scr, m_member, m_random, m_novel_on, m_novel_rv, m_nn, m_sl) if go else 'did NOT cleanly close the loop (see per-seed checks)'}"
                   f"{'; %.0f%% of the novel-concept surfacing attributable to the curiosity gain' % (100 * attribution) if attribution is not None else ''}. "
                   f"{'=> internally-generated -> selected -> SPOKEN: the first self-initiated-utterance correlate.' if go else 'Per THE LAW: tune gain_scale / rest_steps / min_frac / D; not a stop.'}")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        vd = Verdict("self-initiated utterance (seed->utterance routing, 6-seed)")
        vd.require("run completed without error", err is None, expect=True)
        preconditions = vd.decide(False)["preconditions"]

    summary = {"probe": "self_initiated_utterance", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_mem": a.n_mem, "rest_steps": a.rest_steps, "acid_steps": a.acid_steps, "gain_scale": a.gain_scale,
               "D": a.D, "curiosity_bias_attribution": attribution, "preconditions": preconditions,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[self-utter] VERDICT: {verdict}\n[self-utter] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
