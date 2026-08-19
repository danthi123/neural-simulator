"""Replay pattern-separation, decisive spiking demonstration (board #43).

A DG-style sparse-EXPANSIVE separator on the consolidation stream keeps two
SIMILAR memories from blurring in the cortical store; lesion the separator and
the blur returns.

WHY A CONTROLLED LIF NETWORK (and not the production bridge):
    The sibling `_replay_dg_pattern_separation_gate.py` builds this exact circuit
    on the production `SimulationBridge` (Izhikevich + rate-window Hebbian). It
    reproduces the separation *dissociation* -- DG engram Jaccard drops from ~1.00
    (dense, competition OFF) to ~0.39-0.68 (competition ON, 6 seeds) -- but two
    SUBSTRATE properties block a clean consolidation read there: (a) Izhikevich RS
    granule cells POST-INHIBITORY REBOUND-burst under the strong phasic basket
    inhibition needed for a sparse code (measured: competition ON delivers g_i ~18x
    g_e yet raises DG spikes 851->2679), and (b) the spiking k-WTA has a razor-thin,
    seed-variable operating window (basket->granule 3->5 flips DG from ~28 active
    cells to ~0). Consolidation discriminability there stays at chance (mean
    selectivity ~0 across 6 seeds; artifact bridge_substrate_6seed.json). This is
    the 2026-05-31 separation-vs-reliability boundary manifesting at the substrate
    level. To decide the SCIENCE (does DG separation prevent replay-consolidation
    blur?) we isolate it in a controlled leaky-integrate-and-fire network where the
    granule model does not rebound and the k-WTA is stable. Everything that computes
    is still neurons + synapses: the perforant projection, the basket feedforward
    inhibition (the separator), the Hebbian consolidation write, and the opponent
    output read-out. Porting the mechanism back onto the bridge (solving the
    Izhikevich rebound + k-WTA-stability) is the named next step.

Biology: the dentate gyrus recodes overlapping entorhinal inputs into a sparse,
expansive code (few granule winners) via a random perforant projection plus
strong feedforward PV-basket inhibition. Similar EC inputs -> orthogonal DG
engrams (Leutgeb 2007; Bakker 2008; Marr 1971; O'Reilly & McClelland 1994).
Systems consolidation transfers the hippocampal engram to cortex through offline
replay (Kandel; Buzsaki). If replay re-emits the SEPARATED engram, the offline
Hebbian cortical write binds each memory's answer to distinct cells -> the
memories stay discriminable without the hippocampus.

Circuit (all leaky-integrate-and-fire; current-based exponential synapses):
    input (EC)  --fixed random EXPANSIVE-->  dg (granule)
    input,dg    --feedforward-->  dg_basket (PV)  --gate: separator-->  dg   (k-WTA)
    dg  --PLASTIC rate-Hebbian (replay/sleep only)-->  answer (cortex)
    answer  <--opponent inhibition-->  answer_inh
Consolidation is OFFLINE replay: each event reinstates a memory's input (-> its
sparse dg engram) with its answer assembly (the hippocampal index), and the
coincidence potentiates dg_engram -> answer. Retrieval drives ONLY the input
(index off, plasticity off) and reads which answer assembly wins.

The ONLY variable across the dissociation is the basket->granule (separator)
gain; every drive, pattern, seed, and schedule is identical.

Acknowledged scaffolds (tracked): host-defined input (sensory) patterns and
answer assemblies; host reinstatement of each memory's input AND answer during
replay (the hippocampal index / SWR trigger); a rate-window Hebbian write (the
same coactivity stand-in the consolidation gates use); an argmax over answer
spike counts for MEASUREMENT only.

Run:
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_lif \
        --seeds 42 43 44 100 101 102 --out research/findings/raw/replay_dg_sep/lif.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SEEDS = (42, 43, 44, 100, 101, 102)

CONDITIONS = (
    "similar_separator_on",
    "similar_separator_off",     # separator lesion -> the NULL / blur baseline
    "dissimilar_separator_on",
    "dissimilar_separator_off",
    "single_separator_on",       # single-memory recall guard (no interference)
)


@dataclass(frozen=True)
class Config:
    # populations
    n_input: int = 80
    n_dg: int = 400            # expansive (5x the input assembly)
    n_dg_basket: int = 30      # PV feedforward interneurons (the separator)
    n_answer: int = 60
    n_answer_inh: int = 16
    answer_assembly: int = 16
    # input memory patterns
    input_assembly: int = 40
    similar_overlap: int = 30      # Jaccard 30/50 = 0.60 (the memories that blur)
    dissimilar_overlap: int = 3    # Jaccard ~0.04
    # perforant projection input->dg
    dg_fan_in: int = 12
    w_input_dg: float = 1.9
    # separator: feedforward PV-basket k-WTA
    w_input_basket: float = 0.52   # middle of the stable [0.45,0.60] window (cliff to dg-collapse at ~0.65)
    w_dg_basket: float = 0.10
    w_basket_dg: float = 3.50      # basket->granule (scaled by the separator gate)
    # answer opponent competition
    w_answer_inh: float = 0.6
    w_inh_answer: float = 0.9
    # plastic consolidation write dg->answer
    w_dg_answer_init: float = 0.0
    w_dg_answer_max: float = 3.0
    hebb_lr: float = 0.004
    # index reinstatement drive to the answer during replay
    answer_teacher_current: float = 26.0
    # LIF dynamics (normalized: rest 0, threshold 1)
    tau_m: float = 20.0
    tau_syn: float = 5.0
    dt: float = 1.0
    v_threshold: float = 1.0
    t_refractory: int = 2
    input_drive: float = 2.6       # supra-threshold drive to an active input cell
    # schedule
    replay_events_per_memory: int = 16
    event_steps: int = 45
    probe_steps: int = 60


def smoke_config() -> Config:
    return Config(n_dg=240, replay_events_per_memory=8, event_steps=35, probe_steps=40)


# --------------------------------------------------------------------------- #
# A minimal current-based LIF network. All computation is spiking/synaptic.
# --------------------------------------------------------------------------- #
class LIFNet:
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed * 1_000_003 + 7)
        sizes = {
            "input": cfg.n_input, "dg": cfg.n_dg, "dg_basket": cfg.n_dg_basket,
            "answer": cfg.n_answer, "answer_inh": cfg.n_answer_inh,
        }
        self.names = list(sizes.keys())
        self.slice = {}
        off = 0
        for name in self.names:
            self.slice[name] = slice(off, off + sizes[name])
            off += sizes[name]
        self.N = off
        # per-neuron state
        self.v = np.zeros(self.N)
        self.ref = np.zeros(self.N, dtype=np.int32)
        self.g_exc = np.zeros(self.N)     # excitatory synaptic current trace
        self.g_inh = np.zeros(self.N)     # inhibitory synaptic current trace
        # connectivity: list of (pre_slice, post_slice, W[post,pre], sign, gate_name, plastic)
        self.conns = []
        self.gates = {"separator": 1.0}

    def idx(self, name):
        s = self.slice[name]
        return np.arange(s.start, s.stop)

    def _random_fanin(self, n_post, n_pre, fan_in, w):
        W = np.zeros((n_post, n_pre))
        for i in range(n_post):
            aff = self.rng.choice(n_pre, size=min(fan_in, n_pre), replace=False)
            W[i, aff] = w
        return W

    def add_dense(self, pre, post, w, sign, gate=None, plastic=False, self_zero=False):
        npre = self.slice[pre].stop - self.slice[pre].start
        npost = self.slice[post].stop - self.slice[post].start
        W = np.full((npost, npre), float(w))
        if self_zero and npre == npost:
            np.fill_diagonal(W, 0.0)
        self.conns.append([pre, post, W, sign, gate, plastic])
        return len(self.conns) - 1

    def add_random(self, pre, post, fan_in, w, sign, gate=None, plastic=False):
        npre = self.slice[pre].stop - self.slice[pre].start
        npost = self.slice[post].stop - self.slice[post].start
        W = self._random_fanin(npost, npre, fan_in, w)
        self.conns.append([pre, post, W, sign, gate, plastic])
        return len(self.conns) - 1

    def reset_dynamics(self):
        self.v[:] = 0.0
        self.ref[:] = 0
        self.g_exc[:] = 0.0
        self.g_inh[:] = 0.0

    def step(self, ext_current):
        cfg = self.cfg
        # decay synaptic traces
        decay = np.exp(-cfg.dt / cfg.tau_syn)
        self.g_exc *= decay
        self.g_inh *= decay
        I = ext_current + self.g_exc - self.g_inh
        # LIF membrane update (neurons in refractory are clamped)
        active = self.ref <= 0
        self.v[active] += cfg.dt / cfg.tau_m * (-self.v[active] + I[active])
        self.v[~active] = 0.0
        self.ref[~active] -= 1
        spikes = self.v >= cfg.v_threshold
        self.v[spikes] = 0.0
        self.ref[spikes] = cfg.t_refractory
        # deliver spikes along connections into next-step traces
        sp = spikes.astype(np.float64)
        for pre, post, W, sign, gate, _plastic in self.conns:
            pre_sp = sp[self.slice[pre]]
            if not pre_sp.any():
                continue
            gain = self.gates.get(gate, 1.0) if gate else 1.0
            if gain == 0.0:
                continue
            contrib = (W @ pre_sp) * gain
            if sign > 0:
                self.g_exc[self.slice[post]] += contrib
            else:
                self.g_inh[self.slice[post]] += contrib
        return spikes

    def run(self, ext_by_name: dict, steps: int, record=("dg", "answer")):
        counts = {name: np.zeros(self.slice[name].stop - self.slice[name].start) for name in record}
        ext = np.zeros(self.N)
        for _ in range(steps):
            ext[:] = 0.0
            for name, (idx, cur) in ext_by_name.items():
                ext[idx] = cur
            spikes = self.step(ext)
            for name in record:
                counts[name] += spikes[self.slice[name]].astype(np.float64)
        return counts


def build_net(cfg: Config, seed: int) -> LIFNet:
    net = LIFNet(cfg, seed)
    # perforant path: expansive random projection
    net.add_random("input", "dg", cfg.dg_fan_in, cfg.w_input_dg, sign=+1)
    # separator: feedforward + feedback drive onto PV basket, basket inhibits granule (gated)
    net.add_dense("input", "dg_basket", cfg.w_input_basket, sign=+1)
    net.add_dense("dg", "dg_basket", cfg.w_dg_basket, sign=+1)
    net.add_dense("dg_basket", "dg", cfg.w_basket_dg, sign=-1, gate="separator")
    # consolidation write (plastic)
    net.w_answer_conn = net.add_dense("dg", "answer", cfg.w_dg_answer_init, sign=+1, plastic=True)
    # opponent output competition
    net.add_dense("answer", "answer_inh", cfg.w_answer_inh, sign=+1)
    net.add_dense("answer_inh", "answer", cfg.w_inh_answer, sign=-1)
    return net


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _input_patterns(seed: int, cfg: Config, kind: str) -> dict:
    rng = np.random.default_rng(seed * 97 + (7 if kind == "similar" else 19 if kind == "dissimilar" else 3))
    pool = np.arange(cfg.n_input)
    overlap = cfg.similar_overlap if kind == "similar" else cfg.dissimilar_overlap
    size = cfg.input_assembly
    draw = rng.choice(pool, 2 * size - overlap, replace=False)
    shared = draw[:overlap]
    m0 = np.sort(np.concatenate([shared, draw[overlap:size]]))
    m1 = np.sort(np.concatenate([shared, draw[size:]]))
    return {"m0": m0, "m1": m1}


def _answer_assemblies(seed: int, cfg: Config) -> dict:
    rng = np.random.default_rng(seed * 61 + 29)
    draw = rng.choice(cfg.n_answer, 2 * cfg.answer_assembly, replace=False)
    a0, a1 = np.split(np.sort(draw), 2)
    return {"m0": np.sort(a0), "m1": np.sort(a1)}


def _dg_engram(net: LIFNet, cfg: Config, input_pat: np.ndarray, separator: bool) -> np.ndarray:
    net.gates["separator"] = 1.0 if separator else 0.0
    net.reset_dynamics()
    dg0 = net.slice["dg"].start
    in_idx = net.idx("input")[input_pat]
    counts = net.run({"input": (in_idx, cfg.input_drive)}, cfg.event_steps, record=("dg",))
    active = np.where(counts["dg"] > 0)[0]
    return active  # DG-local indices


def _replay_consolidate(net: LIFNet, cfg: Config, memories: dict, answers: dict, separator: bool) -> dict:
    net.gates["separator"] = 1.0 if separator else 0.0
    conn = net.conns[net.w_answer_conn]
    W = conn[2]                      # [answer, dg]
    dg_local = np.arange(cfg.n_dg)
    names = list(memories.keys())
    order = []
    for _ in range(cfg.replay_events_per_memory):
        order.extend(names)
    dg_spk = {n: 0 for n in names}
    ans_spk = {n: 0 for n in names}
    for name in order:
        net.reset_dynamics()
        in_idx = net.idx("input")[memories[name]]
        ans_idx_local = answers[name]
        ans_idx = net.idx("answer")[ans_idx_local]
        counts = net.run(
            {"input": (in_idx, cfg.input_drive),
             "answer": (ans_idx, cfg.answer_teacher_current)},
            cfg.event_steps, record=("dg", "answer"))
        # Hebbian coactivity write over the replay event: the weight change
        # reflects how often the pre (dg granule) and post (answer) co-fired.
        # Using spike COUNTS (not step-normalised rates) keeps a sparse engram's
        # single-spike-per-cell code from diluting the write into nothing.
        W += cfg.hebb_lr * np.outer(counts["answer"], counts["dg"])
        np.clip(W, 0.0, cfg.w_dg_answer_max, out=W)
        dg_spk[name] += int(counts["dg"].sum())
        ans_spk[name] += int(counts["answer"].sum())
    return {"replay_events": len(order), "dg_spikes": dg_spk, "answer_spikes": ans_spk,
            "mean_dg_answer_weight": float(W.mean())}


def _probe(net: LIFNet, cfg: Config, memories: dict, answers: dict, target: str, separator: bool) -> dict:
    net.gates["separator"] = 1.0 if separator else 0.0
    net.reset_dynamics()
    in_idx = net.idx("input")[memories[target]]
    counts = net.run({"input": (in_idx, cfg.input_drive)}, cfg.probe_steps, record=("answer",))
    ans = counts["answer"]
    other = [n for n in memories if n != target][0]
    correct = float(ans[answers[target]].mean() / cfg.probe_steps)
    wrong = float(ans[answers[other]].mean() / cfg.probe_steps)
    occ = set(answers[target].tolist()) | set(answers[other].tolist())
    bg = np.asarray([i for i in range(cfg.n_answer) if i not in occ], dtype=np.int64)
    background = float(ans[bg].mean() / cfg.probe_steps) if bg.size else 0.0
    total = float(ans.sum())
    false_spk = float(ans[answers[other]].sum()) + (float(ans[bg].sum()) if bg.size else 0.0)
    return {
        "target": target, "correct_rate": correct, "wrong_rate": wrong,
        "background_rate": background, "margin": correct - max(wrong, background),
        "selectivity": (correct - wrong) / (correct + wrong + 1e-9),
        "false_recall_fraction": false_spk / total if total > 0 else 0.0,
        "target_assembly_wins": bool(correct > wrong + 1e-9),
        "total_answer_spikes": int(total),
    }


def run_condition(seed: int, condition: str, cfg: Config) -> dict:
    kind = ("similar" if condition.startswith("similar")
            else "dissimilar" if condition.startswith("dissimilar") else "single")
    separator = condition.endswith("separator_on")
    net = build_net(cfg, seed)
    inputs = _input_patterns(seed, cfg, kind)
    answers = _answer_assemblies(seed, cfg)
    memories = {"m0": inputs["m0"], "m1": inputs["m1"]}
    replay_mems = {"m0": inputs["m0"]} if kind == "single" else memories
    replay_ans = {"m0": answers["m0"]} if kind == "single" else answers

    e0 = _dg_engram(net, cfg, inputs["m0"], separator)
    e1 = _dg_engram(net, cfg, inputs["m1"], separator)
    dg_sep = {
        "input_jaccard": _jaccard(inputs["m0"], inputs["m1"]),
        "dg_jaccard": _jaccard(e0, e1),
        "dg_active_frac_m0": e0.size / cfg.n_dg,
        "dg_active_frac_m1": e1.size / cfg.n_dg,
    }
    replay = _replay_consolidate(net, cfg, replay_mems, replay_ans, separator)
    probes = {n: _probe(net, cfg, memories, answers, n, separator) for n in memories}

    # SCRAMBLE-TEACH dependency control (causally diagnostic, cf. the
    # systems-consolidation gate): on a FRESH network, run the identical replay
    # but with the memory->answer pairing SWAPPED (m0 taught with m1's answer and
    # vice versa), then probe with the TRUE pairing. If discriminability rides the
    # learned engram->answer mapping (not the separator or the readout geometry),
    # each memory now recalls the OTHER answer -> selectivity must invert/collapse.
    # Run only where it is diagnostic (the similar pair, separator ON).
    scramble = None
    if kind == "similar" and separator:
        net2 = build_net(cfg, seed)
        swapped = {"m0": answers["m1"], "m1": answers["m0"]}
        _replay_consolidate(net2, cfg, memories, swapped, separator)
        sp = {n: _probe(net2, cfg, memories, answers, n, separator) for n in memories}
        scramble = {
            "mean_selectivity": float(np.mean([p["selectivity"] for p in sp.values()])),
            "mean_correct": float(np.mean([p["correct_rate"] for p in sp.values()])),
            "mean_wrong": float(np.mean([p["wrong_rate"] for p in sp.values()])),
            "both_win_true_pairing": all(p["target_assembly_wins"] for p in sp.values()),
        }

    return {
        "seed": int(seed), "condition": condition, "kind": kind, "separator": bool(separator),
        "dg_separation": dg_sep, "replay": replay, "probes": probes, "scramble_teach": scramble,
        "mean_selectivity": float(np.mean([p["selectivity"] for p in probes.values()])),
        "mean_correct": float(np.mean([p["correct_rate"] for p in probes.values()])),
        "mean_wrong": float(np.mean([p["wrong_rate"] for p in probes.values()])),
        "mean_false_recall": float(np.mean([p["false_recall_fraction"] for p in probes.values()])),
        "both_win": all(p["target_assembly_wins"] for p in probes.values()),
    }


def run_seed(seed: int, cfg: Config) -> dict:
    from tools.lab import attributable_to
    rows = {c: run_condition(seed, c, cfg) for c in CONDITIONS}
    on, off = rows["similar_separator_on"], rows["similar_separator_off"]
    # Attribute the similar-memory discriminability to the SEPARATOR, not the
    # scaffold: the treatment (separator ON) vs its lesion control (OFF). A pair
    # measured is not a pair attributed (tools.lab).
    separator_attribution = attributable_to(
        "DG separator on similar-memory discriminability",
        on["mean_selectivity"], off["mean_selectivity"])
    checks = {
        # anti-cheat 1: similar memories discriminable after consolidation; NULL blurs
        "similar_on_discriminable": on["both_win"] and on["mean_selectivity"] >= 0.30 and on["mean_correct"] >= 0.02,
        "null_shows_blur": off["mean_selectivity"] < on["mean_selectivity"] - 0.20,
        # anti-cheat 2: separator load-bearing (lesion -> blur returns)
        "separator_dissociation": (on["mean_selectivity"] - off["mean_selectivity"]) >= 0.20,
        "dg_actually_separates": (on["dg_separation"]["dg_jaccard"] <= off["dg_separation"]["dg_jaccard"] - 0.15
                                  and on["dg_separation"]["dg_jaccard"] < on["dg_separation"]["input_jaccard"] - 0.15),
        # anti-cheat 3: no catastrophic cost on dissimilar / single memory
        "dissimilar_preserved": rows["dissimilar_separator_on"]["both_win"]
        and rows["dissimilar_separator_on"]["mean_selectivity"] >= 0.30,
        "single_memory_recall": rows["single_separator_on"]["probes"]["m0"]["target_assembly_wins"]
        and rows["single_separator_on"]["probes"]["m0"]["correct_rate"] >= 0.02,
        # anti-cheat 4: discriminability rides the LEARNED engram->answer mapping
        # (scramble-teach the swapped pairing -> the true-pairing read collapses)
        "scramble_teach_collapses": (on["scramble_teach"] is not None
                                     and on["scramble_teach"]["mean_selectivity"] < on["mean_selectivity"] - 0.20),
    }
    return {
        "seed": int(seed), "conditions": rows,
        "summary": {
            "similar_on_selectivity": on["mean_selectivity"],
            "similar_off_selectivity": off["mean_selectivity"],
            "selectivity_dissociation": on["mean_selectivity"] - off["mean_selectivity"],
            "similar_on_false_recall": on["mean_false_recall"],
            "similar_off_false_recall": off["mean_false_recall"],
            "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
            "dg_jaccard_off": off["dg_separation"]["dg_jaccard"],
            "input_jaccard": on["dg_separation"]["input_jaccard"],
            "dissimilar_on_selectivity": rows["dissimilar_separator_on"]["mean_selectivity"],
            "single_correct": rows["single_separator_on"]["probes"]["m0"]["correct_rate"],
            "scramble_teach_selectivity": (on["scramble_teach"]["mean_selectivity"]
                                           if on["scramble_teach"] else None),
            "separator_attribution": separator_attribution,
        },
        "checks": checks, "seed_go": all(checks.values()),
    }


def run(seeds: Iterable[int], cfg: Config) -> dict:
    started = time.time()
    rows = [run_seed(int(s), cfg) for s in seeds]
    n = len(rows)

    def pooled(fn):
        return float(np.mean([fn(r) for r in rows]))

    keys = list(rows[0]["summary"].keys())
    pooled_summary = {k: pooled(lambda r, k=k: r["summary"][k]) for k in keys}
    check_names = list(rows[0]["checks"].keys())
    pooled_checks = {name: int(sum(r["checks"][name] for r in rows)) for name in check_names}
    all_go = all(r["seed_go"] for r in rows)
    return {
        "gate": "replay_dg_pattern_separation_lif",
        "seeds": [int(s) for s in seeds], "n_seeds": n,
        "aggregate_status": "GO" if all_go else "NO-GO",
        "seeds_go": [r["seed"] for r in rows if r["seed_go"]],
        "pooled_summary": pooled_summary,
        "pooled_check_counts": pooled_checks,
        "per_seed": rows,
        "scaffolds": [
            "host-defined input (sensory) patterns and answer assemblies",
            "host reinstatement of each memory's input AND answer during replay (hippocampal index / SWR trigger)",
            "rate-window Hebbian coactivity write (the stand-in the consolidation gates use)",
            "argmax over answer-assembly spike counts for measurement only",
            "fixed random perforant projection and fixed basket anatomy (not developed)",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = smoke_config() if args.smoke else Config()
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        s = r["summary"]
        print(f"  seed {r['seed']}: GO={r['seed_go']} sel_on={s['similar_on_selectivity']:.3f} "
              f"sel_off={s['similar_off_selectivity']:.3f} dissoc={s['selectivity_dissociation']:+.3f} "
              f"dgJ_on={s['dg_jaccard_on']:.3f} dgJ_off={s['dg_jaccard_off']:.3f} inJ={s['input_jaccard']:.3f} "
              f"fr_on={s['similar_on_false_recall']:.2f} fr_off={s['similar_off_false_recall']:.2f}", flush=True)
        print(f"          checks={r['checks']}", flush=True)
    print(f"  AGGREGATE: {payload['aggregate_status']} seeds_go={payload['seeds_go']} "
          f"check_counts={payload['pooled_check_counts']}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled_summary'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
