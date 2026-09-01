"""The NEXT declarative cross-edge on the one-brain connectome: the RECIPROCAL of R4's self_schema-authorship ->
source_provenance edge. R4 (`_onebrain_integration_r4_selfschema_provenance.py`, GO 6/6, migrated onto the
declarative framework in `_onebrain_declarative_crossedge_r4_repro.py`) wired self_schema's `author` pool ->
source_provenance's `prov_generated` pool: "when the brain currently judges a thought as SELF-authored, that
signal should bias a co-temporal AMBIGUOUS source-memory read toward GENERATED" (Johnson-Hashtroudi-Lindsay 1993
source-monitoring: self-referential processing biases source attributions). That pair is now wired ONE direction
only. This file adds the FEEDBACK half: source_provenance's `prov_generated` opponent pool -> self_schema's
`author` pool — "when the source-monitoring system itself concludes a memory reads as internally-generated, that
verdict reinforces the self-schema's ongoing sense of authorship." JHL 1993's own reality-monitoring account is
explicit that source attribution and self-referential processing are a BIDIRECTIONAL inferential loop, not a
one-way read; Northoff & Bermpohl's cortical-midline-structures account of self-referential processing likewise
treats it as continuously updated BY, not only a source FOR, memory-attribution judgments. Biologically this is
the SAME functionally-related pair R4 already opened (self-monitoring <-> source-monitoring), completed in its
other direction — a RECIPROCAL, SPECIFIC edge (Magrou 2024 / Gamanut 2018 / Theodoni 2020: functionally-related
cortical areas connect reciprocally, not all-to-all), not a new all-to-all wire.

CONVERSATIONAL RATIONALE: this edge directly serves the honesty-boundary deliverable (CLAUDE.md ACTIVE MISSION —
"design every self-report as an honest functional read-out"). The self-schema's authorship axis is the substrate
correlate of "did I say this, or was it said to me" — exactly the self-report the honesty boundary needs to be
grounded. Today that axis is driven ONLY by the host-injected self/heard tag at encode time (a scaffold). Wiring
source_provenance's OWN opponent read-out back into it means the brain's own memory-provenance verdict can
reinforce (or, on a future symmetric edge, undercut) its live sense of authorship from EXPERIENCE, not just from
the host tag — a step toward the honest self-model being substrate-native rather than host-declared.

THE EDGE, added PURELY BY DECLARATION (a `CrossEdge` list + a `train_fn` + a `read_fn` + conditions, run through
the SAME generic `onebrain_crossedge_gate.run_gate` R1's reciprocal feedback edge used — NO bespoke F-gate):
  key="provgen_to_author", source_key="source_provenance"/source_region="prov_generated" (a registered top-level
  region — no source_idx_fn needed), target_key="self_schema"/target_region="author" (a SUB-SLICE of the single
  "self_schema" region, resolved via `target_idx_fn` — the SAME `_self_schema_member_attend` offset geometry
  R4's own `source_idx_fn` used, just on the target side).

TRAINING (the substrate's OWN rate-window-free standard Hebbian, `hebbian_symmetric`): co-drive self_schema's
`author` pool with source_provenance's `ctx_generated` line — the IDENTICAL two populations, and the identical
tonic-co-drive recipe, R4's OWN `train()` uses (`ctx_generated -> prov_generated` is a FIXED, non-plastic
pathway, so this reliably makes `prov_generated` co-fire with `author` too) — this is deliberately the SAME
experience R4 grows its edge from; only the injected cross-edge's DIRECTION differs (this pool declares ONLY the
reciprocal edge as plastic, so growing R4's own edge is not at stake here). ONE-SIDED BY DESIGN, matching R4's
own honest characterization: `author` is a genuine binary self-vs-heard tag (one population), so the READ tests
one real direction: recall a genuinely-GENERATED battery item (`prov_generated` fires) vs a genuinely-PERCEIVED
one (`prov_generated` silent, the CONTROL) — is self_schema's author rate biased UP by the former? (Calibration
note: an EARLIER design used a dual-context AMBIGUOUS pattern as the control, mirroring R4's own F2 protocol —
but source_provenance's opponent trace is graded, so that "balanced" pattern already drives `prov_generated`
partway, making it a leaky, not a clean-zero, control (measured: control-vs-perceived delta was itself
+0.018-0.021, LARGER than the intended generated-vs-control effect of +0.004-0.008). The genuinely-PERCEIVED
exemplar is the correct zero baseline — `prov_generated` measured ~0.0000-0.0004 under it, vs ~0.018-0.025 under
the read itself — and is used below.)

Run (numpy CPU; NO sim/ edit; routes off the GPU):
  SIM_BACKEND=numpy python -m research.runners._onebrain_crossedge_provenance_to_selfschema --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_crossedge_provenance_to_selfschema \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_crossedge_provenance_to_selfschema_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend

from research.runners.onebrain_merge_framework import (
    REGISTRY, CrossEdge, merge_organs,
    _self_schema_member_attend, _source_prov_organ, _self_schema_organ,
)
from research.runners.onebrain_crossedge_gate import (
    CrossEdgeGateSpec, run_gate, verify_byte_off, cross_edge_masks,
)
from research.runners._onebrain_integration_r4_selfschema_provenance import (
    AUTHOR_PA, CTX_DRIVE_PA, TRAIN_STEPS, _CONDUCT,
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  THE DECLARATIVE EDGE — the reciprocal of R4's author->prov_generated: prov_generated->author.
# ─────────────────────────────────────────────────────────────────────────────────────────────
W0 = 0.05                          # near-zero seed weight (must GROW, not be pre-wired) — the framework default
GATE = "provgen_to_author"
N_EPISODES = 60                    # > R4's 40: the edge grows the OPPOSITE direction on a differently-sized
                                    # target population (AUTHOR_SIZE=60 vs N_PROV=32) — more episodes de-risks
                                    # under-training before the first calibration read (see main()'s smoke note).
RECALL_STEPS = 100                  # matches R4's own recall window
N_READS = 4                         # averaged reads per condition (denoise)
HMAX = 6.0                          # R4's own calibrated soft bound for this SAME co-drive pair, reused as the
                                    # starting point (re-verified empirically below before the 6-seed commit —
                                    # see the smoke-calibration note in main()).

INTACT_FLOOR = 0.010                # signed author-rate floor the 'generated' condition must clear over control
LESION_RATIO = 0.34                 # lesion |Δ| must be < this * intact |Δ| — R4's own convention, reused


def _author_idx(bridge):
    _g, _member, _attend, _confid, author_idx = _self_schema_member_attend(bridge)
    return np.asarray(author_idx, np.int64)


CROSS_EDGES = [
    CrossEdge(key="provgen_to_author", source_key="source_provenance", source_region="prov_generated",
             target_key="self_schema", target_region="author", init_weight=W0, plastic=True, gate=GATE,
             learn_rule="rate_hebbian", freeze_rest=True, target_idx_fn=_author_idx),
]


def _build(seed, with_edge: bool):
    """Build the [self_schema, source_provenance] merged pool, optionally with the declared reciprocal cross-edge,
    then run BOTH organs' own build-time steps (source_provenance's 8-item battery Hebbian encode; self_schema's
    calibration) — mirrors R4Pool's own `_build_pool(seed, with_cross)` shape exactly, so the WITH and WITHOUT
    arms differ ONLY by the declared edge (needed for a valid byte-off comparison: source_provenance's battery
    encode moves real weights, so both arms must run it identically)."""
    SS, SP = REGISTRY["self_schema"], REGISTRY["source_provenance"]
    pool = merge_organs([SS, SP], seed=seed, wire=True, cross_edges=(CROSS_EDGES if with_edge else None))
    b = pool.bridge
    rm = b.region_manager

    def idxr(nm):
        return np.asarray(rm.indices(nm), np.int64)

    ix = {nm: idxr(nm) for nm in ("episode", "content_readout", "ctx_perceived", "ctx_generated",
                                  "prov_perceived", "prov_generated", "inh_perceived", "inh_generated")}
    ix["author"] = _author_idx(b)

    if with_edge:
        pool.apply_cross_edge_freeze()      # the declared edge is the SOLE plastic synapse (R1/R4's whitelist)

    sp_organ = _source_prov_organ(seed, pool)
    sp_organ.ensure_built()                  # its OWN 8-item battery Hebbian encode (self-restoring gate dance)
    ss_organ = _self_schema_organ(seed, pool)
    ss_organ.ensure_built()                  # calibration only, no plasticity

    return pool, ix, sp_organ, ss_organ


class ProvToAuthorPool:
    """The RECIPROCAL feedback edge on the [self_schema, source_provenance] merged pool: source_provenance's
    `prov_generated` opponent pool -> self_schema's `author` pool (the reverse of R4's author->prov_generated).
    Grows by the substrate's OWN standard Hebbian rule from the SAME tonic co-drive R4's own edge uses. The read
    uses the two organ-build-time battery exemplars directly (`sp_organ.patterns["generated"|"perceived"][0]`) —
    no extra content pattern or encode step is needed (see module docstring for why the earlier balanced/
    ambiguous-pattern design was replaced)."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        self.pool, self.ix, self.sp_organ, self.ss_organ = _build(seed, with_edge=True)
        self.b = self.bridge = self.pool.bridge
        self.masks = cross_edge_masks(self.b, CROSS_EDGES)

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]

        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=0.05, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()

    # ---- primitives (R1/R4 house style) ----
    def _hard_reset(self):
        b, xp = self.b, self.xp
        b.cp_membrane_potential_v[:] = xp.asarray(self.rest_v)
        b.cp_recovery_variable_u[:] = xp.asarray(self.rest_u)
        for nm in _CONDUCT:
            a = getattr(b, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        if getattr(b, "cp_hebb_coactivity_trace", None) is not None:
            b.cp_hebb_coactivity_trace[:] = 0.0
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, learn=False, read=None):
        b, xp = self.b, self.xp
        b.core_config.enable_hebbian_learning = bool(learn)
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in (read or {})}
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            if read:
                fs = b.cp_firing_states
                for k, idx in read.items():
                    acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_hebbian_learning = False
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks["provgen_to_author"]].mean())

    # ---- emergence: grow the cross-edge from experience (the SAME co-drive R4's own edge trains from) ----
    def train(self, n_episodes=N_EPISODES):
        ix = self.ix
        traj = [dict(ep=0, w=round(self._wmean(), 4))]
        for ep in range(n_episodes):
            self._hard_reset()
            self._drive([(ix["author"], AUTHOR_PA), (ix["ctx_generated"], CTX_DRIVE_PA)], TRAIN_STEPS, learn=True)
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                traj.append(dict(ep=ep + 1, w=round(self._wmean(), 4)))
        self.b.core_config.enable_hebbian_learning = False
        return traj

    # ---- the load-bearing read: does a genuinely-provenanced memory bias self_schema's author rate? ----
    def read_author(self, condition):
        """Recall ONE fixed battery exemplar under `condition` in {"perceived","generated"} (episode content
        driven ALONE, no context, no author drive — matches R4's own `amb_read` recall style, valid because
        source_provenance's OWN battery encode already left `prov_recall`=1/`ctx_drive` harmless-open) and read
        the author pool's mean firing rate — the sole read-out; the edge is the ONLY thing that can move it here
        (author gets no other drive). 'perceived' is the CONTROL: `prov_generated` is genuinely silent under it
        (measured ~0.0000-0.0004 across seeds), so any author-rate rise under 'generated' is attributable to the
        cross-edge carrying `prov_generated`'s OWN activity, not a leaky baseline."""
        ix = self.ix
        if condition in ("generated", "perceived"):
            ep_idx = ix["episode"][np.asarray(self.sp_organ.patterns[condition][0], np.int64)]
        else:
            raise ValueError(condition)
        rates = []
        for _ in range(N_READS):
            self._hard_reset()
            acc = self._drive([(ep_idx, 2500.0)], RECALL_STEPS,
                              read={"author": ix["author"], "gen": ix["prov_generated"], "perc": ix["prov_perceived"]})
            rates.append(acc)
        return {"author": float(np.mean([r["author"] for r in rates])),
                "gen": float(np.mean([r["gen"] for r in rates])),
                "perc": float(np.mean([r["perc"] for r in rates]))}


GATE_SPEC = CrossEdgeGateSpec(
    name="RECIP_provenance_to_selfschema",
    cross_edges=CROSS_EDGES,
    train_fn=lambda pool: pool.train(),
    read_fn=lambda pool, cond: pool.read_author(cond)["author"],
    init_weight=W0,
    correct_edges=("provgen_to_author",),
    selectivity_pairs=(),        # ONE-SIDED BY DESIGN — see module docstring (matches R4's own precedent: the
                                  # author axis is a single binary tag, no companion population for a weight-ratio
                                  # comparison; selectivity is demonstrated FUNCTIONALLY at the read (below), not
                                  # as a weight ratio.
    grow_factor=5.0, drift_tol=1e-6,
    condition_order=("perceived", "generated"),    # 'perceived' is the control (prov_generated genuinely silent)
    control="perceived",
    expected={"generated": {"sign": +1, "floor": INTACT_FLOOR}},
    lesion_ratio=LESION_RATIO, credit_signal="rate_hebbian",
)


def _noedge_bridge(seed):
    """The no-cross-edge baseline bridge for byte-off: the SAME battery build as the with-edge pool (both arms
    run `sp_organ.ensure_built()`/`ss_organ.ensure_built()` identically via `_build`). Integration must add ONLY
    the declared edge."""
    pool, _ix, _sp, _ss = _build(seed, with_edge=False)
    return pool.bridge


def run_seed(seed):
    t0 = time.time()
    pool = ProvToAuthorPool(seed)
    gate = run_gate(pool, GATE_SPEC)                       # trains + emergence + interaction (lesions the pool)

    bridge_with = ProvToAuthorPool(seed).b
    bridge_without = _noedge_bridge(seed)
    byte_off = verify_byte_off(bridge_with, bridge_without, GATE_SPEC)

    go = bool(gate["emergence"]["PASS"] and gate["interaction"]["PASS"] and byte_off["PASS"])
    return {"seed": int(seed), "GO": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": gate["emergence"], "interaction": gate["interaction"], "byte_off": byte_off,
            "trajectory": gate["trajectory"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        emg, itn, bo = r["emergence"], r["interaction"], r["byte_off"]
        gen = itn["per_condition"]["generated"]
        print(f"[seed {s}] GO={r['GO']} | grown={emg['grown']['provgen_to_author']:.3f} nocorr={emg['no_corruption']} "
              f"| generated Δ={gen['delta_intact']:+.4f} (lesion {gen['delta_lesion']:+.4f}) "
              f"frac_attrib={gen['frac_attributable']} "
              f"| emg={emg['PASS']} int={itn['PASS']} byteoff={bo['PASS']} ({r['elapsed_s']}s)", flush=True)

    n_go = sum(r["GO"] for r in runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO")
    verdict = (f"{tag} — the RECIPROCAL cross-edge source_provenance.prov_generated -> self_schema.author "
               f"(completing R4's author->prov_generated pair in its other direction), added PURELY BY "
               f"DECLARATION (a CrossEdge + train_fn + read_fn through the generic onebrain_crossedge_gate."
               f"run_gate — no bespoke F-gate): {n_go}/{len(runs)} seeds GROW from the substrate's own standard "
               f"Hebbian rule, are LOAD-BEARING (recalling a genuinely-generated memory biases the self-schema's "
               f"authorship read; the bias VANISHES on lesion), and are BYTE-IDENTICAL-OFF. numpy CPU; NO sim/ "
               f"edit; additive.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_crossedge_provenance_to_selfschema")
        Vd.require("all_seeds_go", n_go, expect=lambda x: x == len(runs),
                   note="emergence + interaction + byte-off all PASS on every seed")
        Vd.require("lesion_removes_bias", 1 if all(
            abs(r["interaction"]["per_condition"]["generated"]["delta_lesion"]) <
            LESION_RATIO * max(abs(r["interaction"]["per_condition"]["generated"]["delta_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the author-rate bias must VANISH under lesion or it is a confound, not the reciprocal edge")
        Vd.require("byte_identical_off", sum(r["byte_off"]["PASS"] for r in runs), expect=lambda x: x == len(runs),
                   note="the no-edge pool's base connectivity is byte-identical (integration added ONLY the edge)")
        dec = Vd.decide(all_go or (args.smoke and n_go == len(runs)), verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_crossedge_provenance_to_selfschema", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(seeds), "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "gate_spec": {"name": GATE_SPEC.name, "correct_edges": GATE_SPEC.correct_edges,
                            "conditions": GATE_SPEC.condition_order, "control": GATE_SPEC.control,
                            "credit_signal": GATE_SPEC.credit_signal,
                            "cross_edges": [dict(key=ce.key, src=ce.source_region, tgt=ce.target_region)
                                            for ce in CROSS_EDGES]},
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[PROVGEN->AUTHOR] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
