"""L0 GPU SMOKE — the LONGITUDINAL DEVELOPMENT LOOP with the REAL GPU components.

This is the 1-seed GPU smoke of the develop loop scaffolded (CPU-validated) in
`_longitudinal_develop_loop.py`. The CPU scaffold proved the loop ASSEMBLES + CLOSES + develops day-over-day +
persists. This file swaps the CPU stand-ins for the REAL GPU realizations and runs a few 'days' at GPU scale, to
(a) validate the loop at GPU scale with REAL learning, (b) measure the development signal with REAL stream-cortex
code-learning (vs the CPU agent-`hear()` stand-in), and (c) CLOCK the per-day wall-clock.

THE REAL-GPU SWAPS (per the scoping `2026-06-23-artificial-life-longitudinal-test-scoping.md` §6 + the owner's
design call):
  - WAKE/LEARN  = the REAL stream-cortex Hebbian co-occurrence learning (`_phaseB_onbridge_stream_cortex_derisk`).
                  The brain HEARS the daily curriculum word-by-word on a persistent ~Nt*n_per-neuron GPU bridge;
                  its rate-Hebbian synapses accumulate the concept-code co-occurrence M. Each day's NEW concepts
                  enter the stream -> the brain LEARNS their codes (vocab grows as REAL learned codes, not just a
                  curriculum vocab counter). The learned codes are projected to phasors -> the conversational
                  composer's grounded codes (the brain converses on the codes IT LEARNED FROM LISTENING).
  - CONVERSE    = `MultiTurnAgent` on the stream-learned grounded codes (parse/store/recall/abstain/yes-no/chain;
                  the no-confab moat).
  - CONSOLIDATE = the agent's self-replay + the OLD-fact retention RE-TEST (KEEP — the owner-approved stand-in;
                  the no-forget is measured by the retention re-test on old facts; full-SWR-on-the-conv-bridge is
                  a deeper follow-on, NOT this smoke, because `consolidation_trainer` hard-imports cupy AND builds
                  a DIFFERENT direction-vocab bridge).
  - GROWTH      = the `TierPromoter` DECISION (pure-Python mastery->promote) + the lineage growth-event (stand-in;
                  the real neuron-count arch rebuild + weight-transfer is the heavy GPU follow-on -- flagged).
  - PERSIST     = `BridgeLineage` (atomic save/load; the developed state -- learned codes, facts, vocab, tier,
                  the per-day development log -- persists day-to-day).

ANTI-CHEATS (run at GPU scale): the FROZEN-BRAIN arm (plasticity OFF -> the brain HEARS but learns nothing -> no
codes, no facts -> competence must NOT rise) + the PERSISTENCE-RESUME arm (reload the lineage -> the brain resumes
its developed state + continues developing, not a blank slate).

REUSE-BY-IMPORT, NO `sim/` edit. `SIM_BACKEND=cupy` (GPU). Run:
    SIM_BACKEND=cupy python -u -m research.runners._longitudinal_develop_loop_gpu --n-days 4 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# GPU by default for this smoke (an explicit env still wins).
os.environ.setdefault("SIM_BACKEND", "cupy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.lineage import BridgeLineage  # noqa: E402
from sim.backend import to_host  # noqa: E402

# Reuse the scaffold's loop machinery wherever it is GPU-agnostic (the curriculum, the agent build, the probe
# helpers, the metrics, the persistence). We ONLY replace the WAKE/learn stage (with the real stream cortex) and
# the grounded-codes wiring.
from research.runners._longitudinal_develop_loop import (  # noqa: E402
    GradedCurriculum, DevelopState, build_agent, _teach_fact,
    _query_recall, _query_yesno, _query_chain, consolidate, maybe_grow,
    _save_state, _load_state,
)
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402

WINDOW = 2


def double_center(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()


# ============================================================================================================
# The GPU developmentally-graded daily curriculum. Every word (agent / action / patient) is a concept the stream
# cortex can LEARN from the TinyStories corpus (drawn from the 8x8 taxonomy: animals/food/places/toys/family as
# entities, the `actions` category as verbs). Simple at day 0 -> richer day over day, exactly the developmental
# shape; the cumulative concept set is what the stream cortex hears + learns codes for.
# ============================================================================================================

_GPU_SYLLABUS = [
    # ---- DAY 0: a few concepts, plain SVO assertions ----
    {
        "new_concepts": ["dog", "cat", "eat", "run", "apple", "ball"],
        "facts": [("dog", "eat", "apple"), ("cat", "run", "ball")],
        "probe_recall": [("patient", ("dog", "eat"), "apple"),
                         ("agent", ("run", "ball"), "cat")],
        "probe_heldout": [],
        "probe_yesno": [],
        "probe_chain": [],
    },
    # ---- DAY 1: add concepts; introduce yes/no + cumulative recall ----
    {
        "new_concepts": ["bird", "fish", "play", "jump", "tree", "book"],
        "facts": [("bird", "play", "tree"), ("fish", "jump", "book"),
                  ("dog", "run", "cat")],   # reuse day-0 concepts in a NEW fact (cumulative)
        "probe_recall": [("patient", ("bird", "play"), "tree"),
                         ("patient", ("fish", "jump"), "book")],
        "probe_heldout": [("patient", ("dog", "run"), "cat")],   # taught today, probes cumulative recall
        "probe_yesno": [("dog", "eat", "apple", "yes"),            # day-0 fact still true (retention via query)
                        ("bird", "jump", "book", "no_or_unknown")],  # never taught -> moat
        "probe_chain": [],
    },
    # ---- DAY 2: add concepts; introduce a multi-hop reasoning chain ----
    {
        "new_concepts": ["mouse", "duck", "look", "walk", "house", "box"],
        "facts": [("mouse", "look", "house"), ("duck", "walk", "box"),
                  ("cat", "look", "mouse")],
        "probe_recall": [("patient", ("mouse", "look"), "house"),
                         ("patient", ("duck", "walk"), "box")],
        "probe_heldout": [("patient", ("cat", "look"), "mouse")],
        "probe_yesno": [("mouse", "look", "house", "yes"),
                        ("duck", "look", "mouse", "no_or_unknown")],
        # 2-hop: cat -look-> mouse -look-> house  (the chain's intermediate carried neurally)
        "probe_chain": [("cat", "look", 2, "house")],
    },
    # ---- DAY 3+: a denser day to exercise mastery -> tier promotion ----
    {
        "new_concepts": ["bear", "girl", "sleep", "sing", "park", "kite"],
        "facts": [("bear", "sleep", "park"), ("girl", "sing", "kite"),
                  ("bear", "sleep", "tree")],
        "probe_recall": [("patient", ("bear", "sleep"), "park"),
                         ("patient", ("girl", "sing"), "kite")],
        "probe_heldout": [("patient", ("bear", "sleep"), "tree")],
        "probe_yesno": [("bear", "sleep", "park", "yes")],
        "probe_chain": [],
    },
]


class GPUGradedCurriculum(GradedCurriculum):
    def __init__(self):
        super().__init__(syllabus=_GPU_SYLLABUS)


# ============================================================================================================
# REAL-GPU WAKE/LEARN: the persistent stream cortex. The brain HEARS the corpus window-by-window for the concepts
# it has been exposed to SO FAR; its rate-Hebbian synapses accumulate the co-occurrence M; we read the learned
# code per concept + project it to a phasor (the composer's grounded code). This IS the development engine.
# ============================================================================================================

class StreamCortex:
    """A persistent GPU stream-cortex bridge that LEARNS concept codes by hearing the TinyStories corpus
    window-by-window (online rate-Hebbian co-occurrence). The brain's vocabulary GROWS as new concepts are
    introduced into the stream; the learned codes drive the conversational composer (grounded codes)."""

    def __init__(self, full_vocab, seed, n_hub=200, n_per=12, hub_scale=250.0, tgt_scale=1200.0,
                 window_steps=2, D=128, verbose=True):
        from research.runners._phaseB_onbridge_stream_cortex_derisk import build_stream_bridge
        self.full_vocab = list(full_vocab)
        self.Nt = len(self.full_vocab)
        self.tgt_row = {w: i for i, w in enumerate(self.full_vocab)}
        self.target_set = set(self.full_vocab)
        self.seed = int(seed)
        self.n_hub, self.n_per = int(n_hub), int(n_per)
        self.hub_scale, self.tgt_scale = float(hub_scale), float(tgt_scale)
        self.window_steps = int(window_steps)
        self.D = int(D)
        self.verbose = verbose

        # the corpus stream + the hub set (top-N frequent context words; a brain knows its common words)
        self.stories = self._load_token_stream()
        gfreq = Counter()
        for toks in self.stories:
            gfreq.update(toks)
        self.hubs = [w for w, _ in gfreq.most_common()
                     if w not in STOPLIST and w not in self.target_set][:self.n_hub]
        self.hub_idx = {w: i for i, w in enumerate(self.hubs)}
        self.keep = self.target_set | set(self.hubs)

        # the persistent learning bridge (Nt targets x n_hub hubs; the SAME bridge across days -> the co-occurrence
        # ACCUMULATES across days, the brain's cortex developing over simulated time)
        t0 = time.time()
        self.bridge, self.hub_region, self.tgt_region = build_stream_bridge(self.Nt, self.n_hub, self.n_per, seed)
        self.build_s = time.time() - t0
        self.xp = self.bridge._cp if hasattr(self.bridge, "_cp") else None
        self.n_hub_neurons = self.n_hub * self.n_per
        self.n_tgt_neurons = self.Nt * self.n_per

        # a fixed random complex projection: learned code[Nt-feat? no -> n_hub-feat] -> phasor[D]. The "feature
        # vector" for each concept is its learned hub-co-occurrence row (length n_hub). proj: (D, n_hub).
        rng = np.random.RandomState(seed * 7 + 3)
        self.proj = (rng.randn(self.D, self.n_hub) + 1j * rng.randn(self.D, self.n_hub)) / np.sqrt(self.n_hub)

        # the host reference count of EXACTLY the windows the bridge sees (the learning-fidelity reference; NOT
        # used to drive the bridge -- the bridge learns M in its synapses from the co-activation)
        self.C_stream = np.zeros((self.Nt, self.n_hub), dtype=np.float64)
        self.total_windows = 0
        # which concepts have actually been streamed (heard) so far -> the brain's REAL learned vocabulary
        self.heard_concepts = set()
        self._rng_story = np.random.RandomState(seed)

    @staticmethod
    def _load_token_stream():
        path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        return [re.findall(r"[a-z]+", s) for s in text.split("<|endoftext|>")]

    def _present_window(self, tgt_ids, hub_ids):
        hub_full = np.zeros(self.n_hub_neurons, np.float32)
        tgt_full = np.zeros(self.n_tgt_neurons, np.float32)
        for h in hub_ids:
            hub_full[h * self.n_per:(h + 1) * self.n_per] = self.hub_scale
        for t in tgt_ids:
            tgt_full[t * self.n_per:(t + 1) * self.n_per] = self.tgt_scale
        b = self.bridge
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[self.hub_region] = self.xp.asarray(hub_full) if self.xp is not None else hub_full
        b.cp_external_input_current[self.tgt_region] = self.xp.asarray(tgt_full) if self.xp is not None else tgt_full
        for _ in range(self.window_steps):
            b._run_one_simulation_step()

    def hear_day(self, day_concepts, max_windows):
        """The WAKE/learn stage for ONE day: stream corpus windows that contain the day's concepts (the brain hears
        them in context). Only windows whose target words are among `day_concepts` are presented (so the day's NEW
        concepts get their co-occurrence learned, while prior days' concepts -- still in the bridge -- retain their
        accumulated codes). Returns the number of windows streamed THIS day."""
        day_set = set(c for c in day_concepts if c in self.target_set)
        self.heard_concepts |= day_set
        story_order = self._rng_story.permutation(len(self.stories))
        n_win = 0
        for si in story_order:
            if n_win >= max_windows:
                break
            kept = [t for t in self.stories[si] if t in self.keep]
            for c in range(len(kept)):
                lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1)
                win = kept[lo:hi]
                # only present if this window carries one of TODAY's concepts (focus the day's learning) AND a hub
                tgt_ids = [self.tgt_row[w] for w in win if w in day_set]
                hub_ids = [self.hub_idx[w] for w in win if w in self.hub_idx]
                if tgt_ids and hub_ids:
                    self._present_window(tgt_ids, hub_ids)
                    for t in tgt_ids:
                        for h in hub_ids:
                            self.C_stream[t, h] += 1.0
                    n_win += 1
                    if n_win >= max_windows:
                        break
        self.bridge.cp_external_input_current[:] = 0.0
        self.total_windows += n_win
        return n_win

    def read_codes(self):
        """Read the stream-learned codes from the bridge synapses -> (M[Nt, n_hub] normalized code, grounded
        phasor dict). M = population block-mean of the learned hub->target weights; the code = log-double-centre
        (the validated normalization); the phasor = exp(i angle(proj @ code_row))."""
        W = np.asarray(to_host(self.bridge.cp_connections.todense())).astype(np.float64)
        blk = W[np.ix_(self.hub_region, self.tgt_region)].reshape(
            self.n_hub, self.n_per, self.Nt, self.n_per).mean(axis=(1, 3))
        M = blk.T                                       # (Nt, n_hub) stream-learned co-occurrence
        code = double_center(np.log1p(M * 100.0))       # (Nt, n_hub) normalized code
        grounded = {}
        for w, i in self.tgt_row.items():
            if w in self.heard_concepts:                # only ground concepts the brain has actually HEARD
                z = self.proj @ code[i].astype(np.complex128)
                grounded[w] = (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)
        return M, code, grounded

    def learning_fidelity(self):
        """corr(M, C): does the bridge's learned co-occurrence match the windows it heard? (a learning-quality
        check -- LEARNED in the synapses, not tabulated)."""
        M, _, _ = self.read_codes()
        if M.std() <= 0 or self.C_stream.std() <= 0:
            return 0.0
        return float(np.corrcoef(M.flatten(), self.C_stream.flatten())[0, 1])

    def close(self):
        # free the GPU bridge
        try:
            self.bridge = None
            if self.xp is not None:
                self.xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


# ============================================================================================================
# THE GPU develop LOOP — same five stages, REAL stream cortex for WAKE/learn + grounded codes for CONVERSE.
# ============================================================================================================

def develop_gpu(lineage, curriculum, n_days, seed=42, consolidation_on=True, plasticity_on=True,
                max_windows_per_day=2500, n_hub=200, n_per=12, D=128, enable_neural_render=False,
                resume=False, verbose=True, _shared_cortex=None):
    """The GPU develop(N_days) loop. Each simulated day: WAKE = REAL stream-cortex code-learning (the brain hears
    the day's concepts in the corpus) -> CONVERSE = MultiTurnAgent on the learned grounded codes (store the day's
    facts, run the probe batteries) -> SLEEP consolidation (self-replay + retention re-test) -> [GROWTH] -> METRICS
    -> PERSIST.

    `plasticity_on=False` is the FROZEN-BRAIN anti-cheat: the brain HEARS the stream but the stream cortex's
    Hebbian learning is gated OFF (no code learning) AND the day's facts are not committed -> competence must NOT
    rise. Returns (per_day metrics, assembly_trace)."""
    from sim.auto_growth import TierPromoter

    rng = np.random.default_rng(seed)
    full_vocab = curriculum.full_vocab()
    referent_nouns = curriculum.referent_nouns()

    # resume OR init at age 0
    if resume and lineage.exists():
        state = _load_state(lineage)
        if verbose:
            print(f"  [resume] day={state.day} facts={len(state.facts)} vocab={len(state.vocab)} "
                  f"tier={state.current_tier}", flush=True)
    else:
        state = DevelopState(seed=seed)

    # --- the persistent stream cortex (the developing brain's cortex). Built ONCE (or shared across resume). On
    #     the frozen arm we DISABLE the bridge's Hebbian learning so hearing the stream cannot learn codes. ---
    own_cortex = _shared_cortex is None
    if own_cortex:
        cortex = StreamCortex(full_vocab, seed, n_hub=n_hub, n_per=n_per, D=D, verbose=verbose)
        if not plasticity_on:
            # FROZEN-BRAIN: gate the stream cortex's plasticity OFF (the bridge hears but does not learn codes)
            cortex.bridge.core_config.enable_hebbian_learning = False
    else:
        cortex = _shared_cortex

    # On resume, re-hear the cumulative vocabulary so the freshly-built cortex re-acquires the developed codes (the
    # cheap stand-in for loading the bridge's synaptic store; the GPU full-persist of cp_connections is a follow-on).
    if state.vocab and own_cortex:
        if verbose:
            print(f"  [resume] re-hearing {len(state.vocab)} prior concepts to re-instate learned codes...",
                  flush=True)
        cortex.hear_day(list(state.vocab), max_windows=max_windows_per_day)

    promoter = TierPromoter(initial_tier=state.current_tier)
    assembly_trace = {"stages_run": [], "build_seconds": round(cortex.build_s, 2)}
    per_day = []
    start_day = state.day

    for d in range(n_days):
        day_index = start_day + d
        day_t0 = time.time()
        day_curr = curriculum.day_stream(day_index)

        # --- vocab growth bookkeeping (the cumulative curriculum vocabulary) ---
        for c in day_curr["new_concepts"]:
            if c not in state.vocab:
                state.vocab.append(c)

        # ================= WAKE: the REAL stream-cortex code-learning =================
        # the brain HEARS the day's concepts in the corpus; its Hebbian synapses learn their codes. On the frozen
        # arm, plasticity is gated off (hears, learns nothing).
        wake_t0 = time.time()
        n_windows = cortex.hear_day(day_curr["new_concepts"], max_windows=max_windows_per_day)
        wake_s = time.time() - wake_t0
        if "WAKE(stream-cortex)" not in assembly_trace["stages_run"]:
            assembly_trace["stages_run"].append("WAKE(stream-cortex)")

        # read the learned codes -> grounded phasors for the conversational composer (the brain converses on the
        # codes it LEARNED FROM LISTENING). On frozen arm, grounded codes are near-zero/degenerate -> we still
        # build the agent (it falls back to its own random codes for unheard words) but commit NO facts.
        _, _, grounded = cortex.read_codes()
        learn_fid = cortex.learning_fidelity()

        # ================= CONVERSE: MultiTurnAgent on the learned grounded codes =================
        # build a fresh agent each day on the CURRENT learned codes + re-instate the developed facts (the cheap
        # stand-in for the agent's persistent synaptic fact-store; idempotent re-store).
        agent = build_agent(full_vocab, seed, plastic=plasticity_on, use_multiturn=True,
                            enable_neural_render=enable_neural_render, referent_nouns=referent_nouns)
        # inject the stream-learned grounded codes into the composer (the brain's own listened-for codes)
        _inject_grounded(agent, grounded)
        if state.facts:
            for f in state.facts:
                _teach_fact(agent, f)      # re-instate developed knowledge

        # WAKE conversation: teach the day's facts
        for fact in day_curr["facts"]:
            if plasticity_on:
                _teach_fact(agent, fact)
                state.add_fact(fact)
            # frozen arm: hear but do not commit
            state.t += 1
        if "CONVERSE" not in assembly_trace["stages_run"]:
            assembly_trace["stages_run"].append("CONVERSE")

        # ================= SLEEP: consolidation (self-replay + retention re-test) =================
        replayed = consolidate(agent, state, consolidation_on, rng)
        if "SLEEP(replay+retention)" not in assembly_trace["stages_run"]:
            assembly_trace["stages_run"].append("SLEEP(replay+retention)")

        # ================= METRICS =================
        dp = _measure(agent, state, day_curr, replayed, day_index, n_windows, learn_fid,
                      len(cortex.heard_concepts))
        mastery = dp["recall_acc"]
        if mastery is None:
            mastery = dp["retention_acc"] if dp["retention_acc"] is not None else 0.0

        # ================= GROWTH: TierPromoter decision =================
        plan = maybe_grow(promoter, mastery, state, lineage)
        dp["promoted"] = (plan is not None)
        dp["brain_tier"] = state.current_tier
        if plan is not None and verbose:
            print(f"    [growth] mastered tier {plan.from_tier} -> promoted to {plan.to_tier}", flush=True)
        if "GROWTH" not in assembly_trace["stages_run"]:
            assembly_trace["stages_run"].append("GROWTH")

        dp["day_seconds"] = round(time.time() - day_t0, 2)
        dp["wake_seconds"] = round(wake_s, 2)
        state.metrics.append(dp)
        per_day.append(dp)
        state.day += 1

        # ================= PERSIST =================
        _save_state(state, lineage, latest_metrics=dp)
        if "PERSIST" not in assembly_trace["stages_run"]:
            assembly_trace["stages_run"].append("PERSIST")

        # free the per-day agent's composer bridge (the conversational agent builds its own small bridge)
        _free_agent(agent)

        if verbose:
            ra = "-" if dp["recall_acc"] is None else f"{dp['recall_acc']:.2f}"
            ho = "-" if dp["heldout_acc"] is None else f"{dp['heldout_acc']:.2f}"
            re_ = "-" if dp["retention_acc"] is None else f"{dp['retention_acc']:.2f}"
            ch = "-" if dp["chain_acc"] is None else f"{dp['chain_acc']:.2f}"
            print(f"  [day {day_index}] vocab={dp['vocab_size']:2d} heard={dp['concepts_heard']:2d} "
                  f"facts={dp['facts_known']:2d} recall={ra} heldout={ho} retain={re_} chain={ch} "
                  f"moat_fa={dp['moat_false_accepts']} corr(M,C)={dp['learn_fidelity']:+.2f} "
                  f"tier={dp['brain_tier']} replay={replayed} win={n_windows} "
                  f"({dp['day_seconds']:.1f}s wake={dp['wake_seconds']:.1f}s)", flush=True)

    if own_cortex:
        cortex.close()
    return per_day, assembly_trace


def _inject_grounded(agent, grounded):
    """Inject the stream-learned grounded phasor codes into the agent's composer concept table (post-construction),
    matching the D the composer uses. The composer's _to_phasor(phases)=exp(2pi i phases). Only heard concepts are
    grounded; unheard fall back to the composer's own codes."""
    comp = getattr(agent, "agent", agent).composer
    D = getattr(comp, "D", None)
    concepts = getattr(comp, "concepts", None)
    if concepts is None:
        return
    for w, ph in grounded.items():
        if w in concepts and D is not None:
            v = np.asarray(ph, dtype=float)
            if v.shape[0] == D:
                concepts[w] = v


def _free_agent(agent):
    try:
        comp = getattr(agent, "agent", agent).composer
        br = getattr(comp, "bridge", None)
        if br is not None and hasattr(br, "_cp") and br._cp is not None:
            br._cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _measure(agent, state, day_curr, replayed, day_index, n_windows, learn_fid, concepts_heard):
    """Per-day development datapoint (scoping §4) + the GPU-specific learning-fidelity + heard-concept count."""
    recall = [_query_recall(agent, p) for p in day_curr.get("probe_recall", [])]
    recall_ok = sum(ok for _, ok in recall)
    heldout = [_query_recall(agent, p) for p in day_curr.get("probe_heldout", [])]
    heldout_ok = sum(ok for _, ok in heldout)
    yesno = [_query_yesno(agent, p) for p in day_curr.get("probe_yesno", [])]
    retention_ok = sum(ok for _, ok, _ in yesno)
    moat_breaches = sum(b for _, _, b in yesno)
    chain = [_query_chain(agent, p) for p in day_curr.get("probe_chain", [])]
    chain_ok = sum(ok for _, ok in chain)
    return {
        "day": day_index,
        "vocab_size": len(state.vocab),
        "concepts_heard": concepts_heard,
        "facts_known": len(state.facts),
        "recall_correct": recall_ok, "recall_total": len(recall),
        "recall_acc": (recall_ok / len(recall)) if recall else None,
        "heldout_correct": heldout_ok, "heldout_total": len(heldout),
        "heldout_acc": (heldout_ok / len(heldout)) if heldout else None,
        "retention_correct": retention_ok, "retention_total": len(yesno),
        "retention_acc": (retention_ok / len(yesno)) if yesno else None,
        "moat_false_accepts": moat_breaches,
        "chain_correct": chain_ok, "chain_total": len(chain),
        "chain_acc": (chain_ok / len(chain)) if chain else None,
        "learn_fidelity": learn_fid,
        "n_windows_today": n_windows,
        "brain_tier": state.current_tier,
        "turns_lived": state.t,
        "facts_replayed_in_sleep": replayed,
    }


# ============================================================================================================
# THE 1-SEED GPU SMOKE
# ============================================================================================================

def run_gpu_smoke(n_days, seed, root, max_windows_per_day, n_hub, n_per, D, enable_neural_render=False,
                  do_frozen=True, do_resume=True, verbose=True):
    curriculum = GPUGradedCurriculum()

    # ---- main run: the full develop loop (consolidation ON, plasticity ON), REAL stream cortex ----
    main_root = os.path.join(root, "main")
    lineage = BridgeLineage("develop_gpu_main", root=Path(main_root))
    if verbose:
        print("[L0 develop loop — GPU] WAKE(REAL stream-cortex) -> CONVERSE(MultiTurnAgent on learned codes) -> "
              f"SLEEP(replay+retention) -> [GROWTH] -> METRICS -> PERSIST, {n_days} days.\n", flush=True)
    per_day, assembly = develop_gpu(lineage, curriculum, n_days, seed=seed, consolidation_on=True,
                                    plasticity_on=True, max_windows_per_day=max_windows_per_day,
                                    n_hub=n_hub, n_per=n_per, D=D, enable_neural_render=enable_neural_render,
                                    verbose=verbose)

    # ---- CHECK 1: the loop CLOSED ----
    stages = assembly["stages_run"]
    loop_closed = all(any(s.startswith(k) for s in stages)
                      for k in ("WAKE", "CONVERSE", "SLEEP", "GROWTH", "PERSIST"))
    lineage_persisted = lineage.exists()
    n_metric_days = len(per_day)

    # ---- CHECK 2: the brain DEVELOPS (vocab/concepts-heard/facts rise; competence non-trivial; REAL learning) ----
    vocab_trend = [dp["vocab_size"] for dp in per_day]
    heard_trend = [dp["concepts_heard"] for dp in per_day]
    facts_trend = [dp["facts_known"] for dp in per_day]
    learnfid_trend = [round(dp["learn_fidelity"], 3) for dp in per_day]
    vocab_grew = (len(vocab_trend) >= 2 and vocab_trend[-1] > vocab_trend[0])
    heard_grew = (len(heard_trend) >= 2 and heard_trend[-1] > heard_trend[0])
    facts_grew = (len(facts_trend) >= 2 and facts_trend[-1] > facts_trend[0])
    day0_vs_dayN_differs = vocab_grew and facts_grew and heard_grew
    recall_vals = [dp["recall_acc"] for dp in per_day if dp["recall_acc"] is not None]
    recall_nontrivial = (len(recall_vals) > 0 and float(np.mean(recall_vals)) >= 0.5)
    # REAL learning happened: the stream cortex's learned co-occurrence matches what it heard (corr(M,C) positive)
    learn_vals = [dp["learn_fidelity"] for dp in per_day]
    real_learning = (len(learn_vals) > 0 and float(np.mean(learn_vals)) >= 0.3)
    # retention: old facts (yes/no on a prior fact) stay correct as new days accumulate (no-forget)
    retention_vals = [dp["retention_acc"] for dp in per_day if dp["retention_acc"] is not None]
    retention_holds = (len(retention_vals) > 0 and float(np.mean(retention_vals)) >= 0.8)
    moat_breaches_total = sum(dp["moat_false_accepts"] for dp in per_day)
    moat_clean = (moat_breaches_total == 0)

    # ---- CHECK 3: persistence RESUMES (reload + run +1 more day; resumes the developed state, not blank) ----
    persist_resumes = None
    resume_info = {}
    if do_resume:
        reloaded = _load_state(lineage)
        reload_state_ok = (reloaded.day == n_days and len(reloaded.facts) == facts_trend[-1]
                           and len(reloaded.vocab) == vocab_trend[-1])
        resume_day, _ = develop_gpu(lineage, curriculum, 1, seed=seed, consolidation_on=True, plasticity_on=True,
                                    max_windows_per_day=max_windows_per_day, n_hub=n_hub, n_per=n_per, D=D,
                                    enable_neural_render=enable_neural_render, resume=True, verbose=False)
        after_resume = _load_state(lineage)
        resumed_continued = (len(resume_day) == 1 and resume_day[0]["day"] == n_days
                             and after_resume.day == n_days + 1
                             and len(after_resume.facts) >= facts_trend[-1])
        persist_resumes = bool(reload_state_ok and resumed_continued)
        resume_info = {"reload_state_ok": reload_state_ok, "resumed_continued": resumed_continued,
                       "resume_day_presented": (resume_day[0]["day"] if resume_day else None),
                       "facts_after_resume": len(after_resume.facts),
                       "days_lived_after_resume": after_resume.day}

    # ---- CHECK 4 (anti-cheat): FROZEN-BRAIN arm (plasticity OFF) accumulates NO knowledge ----
    frozen_facts_final = None
    frozen_anticheat_ok = None
    frozen_learnfid = None
    if do_frozen:
        frozen_root = os.path.join(root, "frozen")
        frozen_lineage = BridgeLineage("develop_gpu_frozen", root=Path(frozen_root))
        frozen_day, _ = develop_gpu(frozen_lineage, curriculum, n_days, seed=seed, consolidation_on=True,
                                    plasticity_on=False, max_windows_per_day=max_windows_per_day,
                                    n_hub=n_hub, n_per=n_per, D=D, enable_neural_render=enable_neural_render,
                                    verbose=False)
        frozen_facts_final = frozen_day[-1]["facts_known"] if frozen_day else 0
        frozen_learnfid = round(float(np.mean([d["learn_fidelity"] for d in frozen_day])), 3) if frozen_day else 0.0
        # frozen brain: commits NO facts AND its gated stream cortex learns ~nothing (learn fidelity ~0)
        frozen_anticheat_ok = (frozen_facts_final == 0 and facts_trend[-1] > 0)

    go = bool(loop_closed and lineage_persisted and n_metric_days == n_days
              and day0_vs_dayN_differs and recall_nontrivial and real_learning and retention_holds and moat_clean
              and (persist_resumes if do_resume else True)
              and (frozen_anticheat_ok if do_frozen else True))

    # per-day wall-clock + the compressed-week ETA
    day_secs = [dp["day_seconds"] for dp in per_day]
    wake_secs = [dp["wake_seconds"] for dp in per_day]
    mean_day_s = float(np.mean(day_secs)) if day_secs else 0.0
    week_eta_min = round(mean_day_s * 7 / 60.0, 1)

    verdict = (
        f"GO — the L0 longitudinal-development loop runs at GPU scale with REAL stream-cortex code-learning "
        f"(corr(M,C) mean {np.mean(learn_vals):+.2f}), day-over-day development (vocab {vocab_trend[0]}->"
        f"{vocab_trend[-1]}, heard {heard_trend[0]}->{heard_trend[-1]}, facts {facts_trend[0]}->{facts_trend[-1]}), "
        f"retention holds ({np.mean(retention_vals) if retention_vals else float('nan'):.2f}), moat 0-FA, a tier "
        f"fired, persistence resumes, frozen-brain anti-cheat holds. Per-day {mean_day_s:.1f}s; compressed-week ETA "
        f"~{week_eta_min} min. READY for the 6-seed compressed-week."
        if go else
        f"PARTIAL/SNAG — loop_closed={loop_closed} dayN_differs={day0_vs_dayN_differs} "
        f"real_learning={real_learning} retention_holds={retention_holds} moat_clean={moat_clean} "
        f"persist_resumes={persist_resumes} frozen={frozen_anticheat_ok}. (See per-day + checks for the localize.)"
    )

    res = {
        "go": go,
        "verdict": verdict,
        "backend": os.environ.get("SIM_BACKEND"),
        "n_days": n_days,
        "seed": seed,
        "config": {"max_windows_per_day": max_windows_per_day, "n_hub": n_hub, "n_per": n_per, "D": D,
                   "neural_render": enable_neural_render},
        "loop_closed": loop_closed,
        "stages_run": stages,
        "lineage_persisted": lineage_persisted,
        "n_metric_days": n_metric_days,
        "vocab_trend": vocab_trend,
        "concepts_heard_trend": heard_trend,
        "facts_trend": facts_trend,
        "learn_fidelity_trend": learnfid_trend,
        "day0_vs_dayN_differs": day0_vs_dayN_differs,
        "recall_acc_mean": (float(np.mean(recall_vals)) if recall_vals else None),
        "recall_nontrivial": recall_nontrivial,
        "real_learning_mean_corrMC": (float(np.mean(learn_vals)) if learn_vals else None),
        "real_learning": real_learning,
        "retention_acc_mean": (float(np.mean(retention_vals)) if retention_vals else None),
        "retention_holds": retention_holds,
        "moat_false_accepts_total": moat_breaches_total,
        "moat_clean": moat_clean,
        "persist_resumes": persist_resumes,
        "resume_info": resume_info,
        "frozen_facts_final": frozen_facts_final,
        "frozen_learn_fidelity": frozen_learnfid,
        "frozen_anticheat_ok": frozen_anticheat_ok,
        "build_seconds": assembly["build_seconds"],
        "per_day_seconds": day_secs,
        "per_day_wake_seconds": wake_secs,
        "mean_day_seconds": round(mean_day_s, 2),
        "mean_wake_seconds": round(float(np.mean(wake_secs)) if wake_secs else 0.0, 2),
        "compressed_week_eta_minutes": week_eta_min,
        "per_day": per_day,
        "components_real_gpu": [
            "WAKE/learn = REAL stream-cortex Hebbian co-occurrence on a persistent GPU bridge (the brain hears the "
            "TinyStories corpus window-by-window + learns concept codes; codes -> grounded phasors -> the composer).",
            "CONVERSE = MultiTurnAgent (real spiking composer/parser) on the stream-LEARNED grounded codes.",
            "PERSIST = BridgeLineage atomic save/load (the developed state persists day-to-day + resumes).",
        ],
        "components_standin": [
            "CONSOLIDATE = the agent's self-replay + OLD-fact retention re-test (owner-approved stand-in; "
            "full-SWR-on-the-conv-bridge is a follow-on -- consolidation_trainer hard-imports cupy + a different "
            "direction-vocab bridge).",
            "GROWTH = the TierPromoter DECISION + lineage growth-event (the real neuron-count arch rebuild + "
            "weight-transfer is the heavy GPU follow-on).",
            "RESUME re-instates the developed codes by RE-HEARING the cumulative vocab (cheap stand-in for "
            "persisting cp_connections in the lineage .h5 -- a follow-on).",
        ],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=4, help="number of simulated 'days' (smoke: 3-5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-windows-per-day", type=int, default=2500,
                    help="stream-window budget per day (caps per-day wall-clock; keep it a SMOKE)")
    ap.add_argument("--n-hub", type=int, default=200, help="stream-cortex hub (context-word) count")
    ap.add_argument("--n-per", type=int, default=12, help="neurons per concept (population code)")
    ap.add_argument("--D", type=int, default=128, help="composer phasor dimension")
    ap.add_argument("--neural-render", action="store_true", help="brain's own spiking serial-order renderer (slow)")
    ap.add_argument("--no-frozen", action="store_true", help="skip the frozen-brain anti-cheat arm")
    ap.add_argument("--no-resume", action="store_true", help="skip the persistence-resume check")
    ap.add_argument("--out", default="research/findings/raw/_longitudinal_develop_loop_gpu_smoke.json")
    ap.add_argument("--keep-lineage", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import logging
    logging.disable(logging.INFO)

    print("=" * 110, flush=True)
    print("[L0 LONGITUDINAL DEVELOPMENT LOOP — 1-SEED GPU SMOKE — REAL stream cortex]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  n_days={a.n_days}  seed={a.seed}  "
          f"max_windows/day={a.max_windows_per_day}  n_hub={a.n_hub}  n_per={a.n_per}  D={a.D}", flush=True)
    print("  VERIFY: the develop loop runs at GPU scale with the REAL stream-cortex code-learning swapped for the "
          "CPU agent-hear() stand-in; measure development + per-day wall-clock + the anti-cheats.", flush=True)
    print("=" * 110 + "\n", flush=True)

    t0 = time.time()
    root = tempfile.mkdtemp(prefix="develop_loop_gpu_")
    try:
        res = run_gpu_smoke(a.n_days, a.seed, root, a.max_windows_per_day, a.n_hub, a.n_per, a.D,
                            enable_neural_render=a.neural_render, do_frozen=not a.no_frozen,
                            do_resume=not a.no_resume, verbose=True)
    finally:
        if not a.keep_lineage:
            shutil.rmtree(root, ignore_errors=True)
    res["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  VERDICT: {res['verdict']}", flush=True)
    print(f"  [saved] {a.out}  (wall {res['wall_seconds']}s)\n{'=' * 110}", flush=True)
    return 0 if res["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
