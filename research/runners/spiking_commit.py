"""The spiking commit stage as a reusable piece: K competing sources -> exclusive commit gates -> bistable latches,
read by persistence (docs/drafts/spiking-commit/PLAN.md, section 3).

PLAN section 3 "Reuse" names this file as the shared home of the stage. Users: the idle-replay organ
(`research/runners/idle_replay_organ.py`, Stage-2 C5 build, 2026-10-06); A1 word-role-parse
(`word_role_parser_organ.py`, built with `commit=`; moved onto this module 2026-10-14 with a byte-identity test,
`tests/test_bg_commit.py::test_a1_commit_battery_byte_identical`); the BG live seam (`bg_action_selection_production_
organ.py`, flag BRAIN_BG_COMMIT, a region-built bridge: `region_parts`). Declared remaining copy: the emotion-category
organ (`emotion_category_organ.py`) still wires the same stage inline.

One topology (`projections`), two emitters: `commit_plan` (explicit-wiring entries for `inject_explicit_wiring`
builds) and `region_parts` (BrainRegion / RegionPathway lists for region-built bridges; density 1.0 and no jitter, so
the same all-to-all edges, autapses included). Two reads: `latch_read` / `read_with_variants` (one presentation) and
`sequence_read` (several presentations; a latch already ON before an item cannot be claimed by it).

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

Pure plan building and pure reads: no bridge is touched here (`region_parts` imports `sim.regions` lazily).
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


def projections(K: int, p: dict, *, gate_in: str, gate_excl: str, latch_loop: str) -> List[dict]:
    """The stage's topology in emission order, per alternative k: xg (source -> G, `gate_in`), gg (G -> G), ggi
    (G -> GI), gig (GI -| every other G, `gate_excl`; only when K > 1), gs (G -> S), ss (S -> S, `latch_loop`).
    Each entry: name, k, pre/post as lists of (population, index) with population in {source, gate, gate_inh, slot},
    the weight key, the transmission gate and the connection kind (I_TO_E is inhibitory)."""
    out = []
    for k in range(K):
        out.append({"name": "xg", "k": k, "pre": [("source", k)], "post": [("gate", k)], "w": "W_XG",
                    "gate": gate_in, "kind": "E_TO_E"})
        out.append({"name": "gg", "k": k, "pre": [("gate", k)], "post": [("gate", k)], "w": "W_GG", "gate": None,
                    "kind": "E_TO_E"})
        out.append({"name": "ggi", "k": k, "pre": [("gate", k)], "post": [("gate_inh", k)], "w": "W_GGI",
                    "gate": None, "kind": "E_TO_I"})
        if K > 1:
            out.append({"name": "gig", "k": k, "pre": [("gate_inh", k)],
                        "post": [("gate", j) for j in range(K) if j != k], "w": "W_GIG", "gate": gate_excl,
                        "kind": "I_TO_E"})
        out.append({"name": "gs", "k": k, "pre": [("gate", k)], "post": [("slot", k)], "w": "W_GS", "gate": None,
                    "kind": "E_TO_E"})
        out.append({"name": "ss", "k": k, "pre": [("slot", k)], "post": [("slot", k)], "w": "W_SS",
                    "gate": latch_loop, "kind": "E_TO_E"})
    return out


def commit_plan(sources: Sequence[np.ndarray], lay: Dict[str, object], p: dict, *, prefix: str,
                gate_in: str, gate_excl: str, latch_loop: str, plasticity_gate: str = GATE_FIXED) -> dict:
    """Plan entries wiring sources[k] -> G_k -> S_k with the exclusion and the latch loop. `p` holds W_XG, W_GG,
    W_GGI, W_GIG, W_GS, W_SS. The three transmission-gate names are the stage's lesions. `plasticity_gate` names the
    (frozen) plasticity gate of every entry; the default is this module's GATE_FIXED."""
    pops = {"source": list(sources), "gate": lay["gate"], "gate_inh": lay["gate_inh"], "slot": lay["slot"]}
    plan = {}
    for e in projections(len(sources), p, gate_in=gate_in, gate_excl=gate_excl, latch_loop=latch_loop):
        pre = np.concatenate([pops[r][j] for r, j in e["pre"]])
        post = np.concatenate([pops[r][j] for r, j in e["post"]])
        plan["%s%s_%d" % (prefix, e["name"], e["k"])] = dense(pre, post, p[e["w"]], conn_type=e["kind"],
                                                              transmission_gate=e["gate"],
                                                              plasticity_gate=plasticity_gate)
    return plan


def region_names(prefix: str, k: int) -> Dict[str, str]:
    """Region names of alternative k's gate, gate-inhibitory pool and latch in a region-built bridge."""
    return {"gate": "%sgate_%d" % (prefix, k), "gate_inh": "%sgate_inh_%d" % (prefix, k),
            "slot": "%sslot_%d" % (prefix, k)}


def region_parts(source_regions: Sequence[str], p: dict, *, prefix: str, gate_in: str, gate_excl: str,
                 latch_loop: str, make_region, exc_type, inh_type):
    """The stage for a region-built bridge: (BrainRegion list, RegionPathway list). Same topology as `commit_plan`
    (`projections`), every pathway at density 1.0 with no jitter (all-to-all; a region pathway onto its own region
    keeps autapses, as `dense` does). The new regions have no internal connectivity (the recurrences are pathways),
    so appended after a build's own regions and pathways they leave every earlier pathway's random draw unchanged.
    `make_region(name, n, exc_fraction=, neuron_type=)` is the caller's BrainRegion factory; `p` also holds NG, NGI,
    NS. Inhibitory pathways use GABA_A."""
    from sim.regions import RegionPathway
    K = len(source_regions)
    regions = []
    for k in range(K):
        nm = region_names(prefix, k)
        regions += [make_region(nm["gate"], int(p["NG"]), exc_fraction=1.0, neuron_type=exc_type),
                    make_region(nm["gate_inh"], int(p["NGI"]), exc_fraction=0.0, neuron_type=inh_type),
                    make_region(nm["slot"], int(p["NS"]), exc_fraction=1.0, neuron_type=exc_type)]

    def region(pop, j):
        return source_regions[j] if pop == "source" else region_names(prefix, j)[pop]

    pathways = []
    for e in projections(K, p, gate_in=gate_in, gate_excl=gate_excl, latch_loop=latch_loop):
        for pr, pj in e["pre"]:
            for qr, qj in e["post"]:
                kw = {"receptor": "gaba_a"} if e["kind"] == "I_TO_E" else {}
                pathways.append(RegionPathway(from_region=region(pr, pj), to_region=region(qr, qj), density=1.0,
                                              weight_mean=float(p[e["w"]]), weight_jitter=0.0, plastic=False,
                                              transmission_gate=e["gate"], **kw))
    return regions, pathways


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


def sequence_read(counts: Sequence[Sequence[float]], NS: int, read_steps: int, read_frac: float,
                  scale: float = 1.0) -> dict:
    """The read over a SEQUENCE of presentations whose latches carry over (A1's incremental assignment). counts[i][k]
    is latch k's count over the last READ_STEPS of item i's hold. Item i's outcome = the one latch that turned ON
    during item i (ON now and not ON after item i-1); none -> None; several -> None and a co-commit. An alternative
    claimed by two items (a latch that died and re-latched) is unfilled and counted as a re-latch. Returns
    dict(threshold, per_item=[{new: [k...], role: k|None}], filled={k: item}, co_commit, relatch, latched_at_end)."""
    thr = float(read_frac) * float(scale) * int(NS) * int(read_steps)
    on_before = set()
    per, co, claim = [], 0, {}
    for i, row in enumerate(counts):
        on_now = {k for k, c in enumerate(row) if float(c) >= thr}
        new = on_now - on_before
        if len(new) == 1:
            role = next(iter(new))
            claim.setdefault(role, []).append(i)
        else:
            role = None
            co += int(len(new) > 1)
        per.append({"new": sorted(new), "role": role})
        on_before = on_now
    filled = {r: idx[0] for r, idx in claim.items() if len(idx) == 1}
    relatch = sum(1 for idx in claim.values() if len(idx) > 1)
    return {"threshold": thr, "per_item": per, "filled": filled, "co_commit": co, "relatch": relatch,
            "latched_at_end": sorted(on_before)}
