"""The spiking commit stage as a reusable piece: K competing sources -> exclusive commit gates -> bistable latches,
read by persistence (docs/drafts/spiking-commit/PLAN.md, section 3).

PLAN section 3 "Reuse" names this file as the shared home of the stage. Its first user is the idle-replay organ
(`research/runners/idle_replay_organ.py`, Stage-2 C5 build, 2026-10-06). A1 (`word_role_parser_organ.py`) still
carries its own inline copy of the same wiring; application A2 moves A1 onto this module with a byte-identity test
(PLAN: "one copy, two users, before a third"). Until then the two copies are declared, not hidden.

The stage, for K alternatives with source populations X_k (which already compete upstream):
  * commit gate G_k: NG excitatory neurons, X_k -> G_k (W_XG, transmission gate `gate_in`), G_k -> G_k (W_GG), G_k ->
    GI_k (NGI inhibitory, W_GGI), GI_k -| G_j for j != k (W_GIG, transmission gate `gate_excl`): at most one gate
    bursts at a time (Lo & Wang 2006; Stine et al. 2023).
  * latch S_k: NS excitatory neurons, G_k -> S_k (W_GS), S_k -> S_k (W_SS, transmission gate `latch_loop`): a bistable
    assembly that holds itself after its input stops (Goldman-Rakic 1995; Compte et al. 2000).
  * read (`latch_read`): after a zero-input hold, a latch is ON iff its spike count over the last READ_STEPS reaches
    READ_FRAC x scale x NS x READ_STEPS. Exactly one ON -> that alternative; none -> abstain; two or more -> co-commit
    (abstain, counted). No count is compared with another alternative's count. `read_with_variants` re-reads the same
    counts at scale 0.5 and 2 (the threshold-invariance bar, the bg-live-seam G4 lesson).

Pure plan building and pure reads: no bridge is touched here, so this module imports only numpy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

GATE_FIXED = "commit_fixed"          # plasticity gate name for the frozen commit-stage projections


def dense(pre_idx, post_idx, weight: float, *, conn_type: str, transmission_gate: Optional[str] = None,
          plasticity_gate: str = GATE_FIXED, plastic: bool = False, receptor: Optional[str] = None) -> dict:
    """All-to-all projection pre -> post at one weight, in the `inject_explicit_wiring` plan-entry format."""
    pre = np.repeat(np.asarray(pre_idx, dtype=np.int64), len(post_idx))
    post = np.tile(np.asarray(post_idx, dtype=np.int64), len(pre_idx))
    d = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
         "initial_weights": np.full(pre.shape[0], float(weight), dtype=np.float32), "plastic": bool(plastic),
         "plasticity_gate": plasticity_gate, "conn_type": conn_type, "count": int(pre.size)}
    if transmission_gate:
        d["transmission_gate"] = transmission_gate
    if receptor:
        d["receptor"] = receptor
    return d


def layout(K: int, offset: int, NG: int, NGI: int, NS: int) -> Dict[str, object]:
    """Neuron index ranges for K gates, gate-inhibitory pools and latches appended at `offset`. Returns
    dict(gate=[idx]*K, gate_inh=[idx]*K, slot=[idx]*K, all_slot, all_gate_inh, end)."""
    o_g = int(offset)
    o_gi = o_g + K * NG
    o_s = o_gi + K * NGI
    end = o_s + K * NS
    gate = [np.arange(o_g + k * NG, o_g + (k + 1) * NG, dtype=np.int64) for k in range(K)]
    gate_inh = [np.arange(o_gi + k * NGI, o_gi + (k + 1) * NGI, dtype=np.int64) for k in range(K)]
    slot = [np.arange(o_s + k * NS, o_s + (k + 1) * NS, dtype=np.int64) for k in range(K)]
    return {"gate": gate, "gate_inh": gate_inh, "slot": slot,
            "all_gate": np.arange(o_g, o_gi, dtype=np.int64),
            "all_gate_inh": np.arange(o_gi, o_s, dtype=np.int64),
            "all_slot": np.arange(o_s, end, dtype=np.int64), "end": int(end)}


def commit_plan(sources: Sequence[np.ndarray], lay: Dict[str, object], p: dict, *, prefix: str,
                gate_in: str, gate_excl: str, latch_loop: str) -> dict:
    """Plan entries wiring sources[k] -> G_k -> S_k with the exclusion and the latch loop. `p` holds W_XG, W_GG,
    W_GGI, W_GIG, W_GS, W_SS. The three transmission-gate names are the stage's lesions."""
    K = len(sources)
    plan = {}
    for k in range(K):
        g, gi, s = lay["gate"][k], lay["gate_inh"][k], lay["slot"][k]
        plan["%sxg_%d" % (prefix, k)] = dense(sources[k], g, p["W_XG"], conn_type="E_TO_E", transmission_gate=gate_in)
        plan["%sgg_%d" % (prefix, k)] = dense(g, g, p["W_GG"], conn_type="E_TO_E")
        plan["%sggi_%d" % (prefix, k)] = dense(g, gi, p["W_GGI"], conn_type="E_TO_I")
        if K > 1:
            other_g = np.concatenate([lay["gate"][j] for j in range(K) if j != k])
            plan["%sgig_%d" % (prefix, k)] = dense(gi, other_g, p["W_GIG"], conn_type="I_TO_E",
                                                   transmission_gate=gate_excl)
        plan["%sgs_%d" % (prefix, k)] = dense(g, s, p["W_GS"], conn_type="E_TO_E")
        plan["%sss_%d" % (prefix, k)] = dense(s, s, p["W_SS"], conn_type="E_TO_E", transmission_gate=latch_loop)
    return plan


def latch_read(slot_counts: Sequence[int], NS: int, read_steps: int, read_frac: float, scale: float = 1.0) -> dict:
    """The transducer read on recorded latch counts. Returns dict(threshold, on=[k...], winner=k|None,
    co_commit=bool). A pure function: the same counts always read the same."""
    thr = float(read_frac) * float(scale) * int(NS) * int(read_steps)
    on = [k for k, c in enumerate(slot_counts) if float(c) >= thr]
    return {"threshold": thr, "on": on, "winner": (on[0] if len(on) == 1 else None), "co_commit": len(on) > 1}


def read_with_variants(slot_counts: Sequence[int], NS: int, read_steps: int, read_frac: float) -> dict:
    """`latch_read` at the registered level plus the x0.5 and x2 re-reads of the same counts. `read_invariant` is
    True iff all three name the same winner (or all three abstain)."""
    base = latch_read(slot_counts, NS, read_steps, read_frac)
    variants = {tag: latch_read(slot_counts, NS, read_steps, read_frac, sc)["winner"]
                for tag, sc in (("x0.5", 0.5), ("x2", 2.0))}
    return dict(base, variants=variants, read_invariant=all(v == base["winner"] for v in variants.values()))
