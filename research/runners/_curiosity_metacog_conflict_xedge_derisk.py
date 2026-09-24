"""Curiosity driven BY metacognition's own spiking read-out, through actual synapses, on ONE merged pool.

Lane B (curiosity) x E1 (metacognition). The named next rung in `curiosity_production_organ.py`'s docstring:
"curiosity on a low-confidence RECALL". Pre-registration: `docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-
PREREG.md` (read it first; the gates below are copied from it and were frozen before the 6-seed run).

WHY THIS RUNNER EXISTS (and what it replaces). The previous attempt, `_curiosity_metacog_lowconfidence_coupling_
derisk.py`, fed curiosity a HOST linear map of metacog's INPUT evidence scalar, so curiosity never received anything
metacog computed; its G4 was one seed-drawn permutation (a random draw, not a null); its G5 used `np.allclose`.
It read NOT-GO 4/6 and is banked as a method verdict in
`research/findings/2026-09-23-curiosity-metacog-lowconfidence-coupling-6seed-NOGO-banked-G4-instrument-defect.md`.

THE CIRCUIT (every step is neurons + synapses; host code only drives metacog's INPUT, as production does):
  metacog workspace WTA (2 x 80 accumulator assemblies + shared FS inhibition; the production E1 organ, unmodified,
      built with its production per-region parameter heterogeneity)
    -> metacog's OWN second-order margin comparator (the organ's legacy `confidence_read='margin'` topology from
       `_second_order_metacog_monitor_derisk.build_metacog_bridge`, instantiated on the pool's `meta_schema` region):
       assembly k excites meta_k; assembly k excites an inhibitory relay `meta_margin_fs_k`, which inhibits
       meta_(not k). meta_k therefore fires for "evidence for k NOT outweighed by the rival". Per-postsynaptic-neuron
       weight heterogeneity (seeded) gives a graded population code instead of the synchronous all-identical
       response a uniform dense projection produces (measured: uniform weights quantized meta rates to ~0.9 Hz steps).
       Measured (seed 42): as evidence rises, the rival comparator is progressively silenced while the favored one
       stays flat -> TOTAL comparator firing FALLS with confidence. Co-activation of the two comparators is a
       class-symmetric CONFLICT signal (Botvinick et al. 2001: conflict = co-activation of competing channels).
    -> ONE declared cross-organ edge `x_metacog_meta_to_curiosity_ask`: meta_schema (both comparators) -> curiosity's
       `ask` crave pool, excitatory, fixed weight (a `CrossEdge` row on `merge_organs`, the framework's declarative
       cross-edge form). No host scalar reaches `ask`: the curiosity neuromodulator is NOT installed and
       `current_novelty_signal` is never set, so every ASK spike is caused by metacog spikes through this edge.
  Result: an answered-but-uncertain recall (low metacog margin) -> comparators co-active -> ASK fires; a confident
  recall -> the rival comparator is suppressed -> ASK falls silent.

WHAT IS HOST-DESIGNED (declared, not hidden): the comparator weights and the edge weight are fixed, chosen on the
seed-42 calibration smoke (the 5 other seeds are held out). The edge does not grow by a learning rule. A learned
(Hebbian-grown) edge and the production 11-organ pool wiring (which gain-0-freezes `ask`/`workspace`, so a cross-edge
needs a freeze-seam change) are the next rungs.

FUNCTIONAL CORRELATE ONLY — no phenomenal claim. Additive research runner: no `sim/` edit, no production flag, no
default flip; nothing in the live chat path imports this file.

Run:
  SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_conflict_xedge_derisk --smoke
  SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_conflict_xedge_derisk \
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_curiosity_metacog_conflict_xedge_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import to_host  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, OrganDescriptor, merge_organs  # noqa: E402
from research.runners.onebrain_crossedge_gate import (  # noqa: E402
    cross_edge_masks, lesion_cross_edges, verify_byte_off,
)
from research.runners.metacog_production_organ import (  # noqa: E402
    MetacogProductionOrgan, nmda_norm_margin, READ_REPS,
)
from research.runners._gnw_rung1_ignition_curve_derisk import DRIVE_STEPS, FREE_STEPS  # noqa: E402

# ── FROZEN operating point (chosen on the seed-42 calibration smoke, 2026-09-23; see the PREREG doc) ─────────────
CMP_EXC = (1.2, 2.4)       # assembly_k -> meta_k excitation, per-post-neuron uniform draw (lo, hi)
CMP_REL = 2.5              # assembly_k -> relay_k excitation (uniform)
CMP_INH = (6.0, 14.0)      # relay_k -> meta_(not k) inhibition, per-post-neuron uniform draw (lo, hi)
RELAY_N_PER_CLASS = 30     # = the organ's own META_MARGIN_FS_SIZE
XEDGE_W = 4.0              # the ONE metacog -> curiosity cross-edge weight (meta_schema -> ask)
CMP_GATE = "metacog_margin_cmp"
XEDGE_KEY = "x_metacog_meta_to_curiosity_ask"

EVIDENCE_GRID = tuple(float(x) for x in np.round(np.linspace(0.0, 1.0, 11), 4))
N_PERM = 10000             # permutation null distribution size (G4)
STEPS_PER_REP = DRIVE_STEPS + FREE_STEPS

# ── pre-registered thresholds (PREREG §3) ────────────────────────────────────────────────────────────────────
G1_RHO_MAX = -0.8          # Spearman rho(evidence, level-mean ASK Hz) must be <= this
G1_MIN_RANGE_HZ = 1.0      # ...and the intact ASK dynamic range must be >= this (else UNDEFINED, not a pass)
G3_ATTRIB_MIN = 0.8        # edge lesion removes >= 80% of the ASK dynamic range
G3_LESION_MAX_FRAC = 0.2   # ...and the lesioned ASK peak is <= 20% of the intact peak
G4_P_MAX = 0.01            # permutation p-value (one-sided, rho <= observed) over per-rep observations
G7_RHO_MAX = -0.8          # class-swap arm (evidence to the OTHER assembly) must stay monotone
G8_RHO_MIN = -0.5          # comparator-relay lesion must ABOLISH the negative coupling (rho > this)

META_REGIONS = ("workspace", "workspace_fs")                 # the production metacog read's neurons
CMP_REGIONS = ("meta_schema", "meta_margin_fs")               # metacog's second-order comparator


def _metacog_het():
    """The production-faithful metacog descriptor: per-region parameter heterogeneity ON its regions (exactly the
    `_onebrain_twopool_merge_organread_verify._recon_descriptors` reconciliation), WITHOUT the pool-wide hebbian/
    gain-0 freeze seam (this 2-organ pool has no plastic organ, so nothing needs freezing; the freeze would forbid
    the cross-edge). Without this, curiosity's `param_het=True` turns global heterogeneity OFF and metacog's
    neurons become identical and synchronous (measured) — i.e. NOT the production metacog."""
    mc = dict(REGISTRY["metacog"].config)
    for k in ("enable_parameter_heterogeneity", "per_region_parameter_heterogeneity", "hebbian_max_weight"):
        mc.pop(k, None)
    return replace(REGISTRY["metacog"], config=mc, param_het=True)


def _het_projection(pre, post, lo, hi, rng) -> dict:
    """Dense pre->post projection whose weight is drawn once PER POSTSYNAPTIC neuron (uniform lo..hi), so each
    comparator neuron has its own gain -> a graded population code. Sign follows the presynaptic neuron type."""
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)
    scale = rng.uniform(lo, hi, size=post.size).astype(np.float32)
    P = np.repeat(pre, post.size)
    Q = np.tile(post, pre.size)
    return {"pre_indices": P, "post_indices": Q, "initial_weights": np.tile(scale, pre.size).astype(np.float32),
            "plastic": False, "plasticity_gate": CMP_GATE, "conn_type": "E_TO_E", "count": int(P.size)}


def _uniform_projection(pre, post, w) -> dict:
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)
    P = np.repeat(pre, post.size)
    Q = np.tile(post, pre.size)
    return {"pre_indices": P, "post_indices": Q, "initial_weights": np.full(P.size, float(w), np.float32),
            "plastic": False, "plasticity_gate": CMP_GATE, "conn_type": "E_TO_E", "count": int(P.size)}


def _comparator_spec(seed):
    return ([BrainRegion(name="meta_margin_fs", n_neurons=2 * RELAY_N_PER_CLASS, exc_fraction=0.0,
                         internal_density=0.0, enable_nmda=False)], [], {})


def _comparator_wiring(bridge, rm):
    ws = np.asarray(rm.indices("workspace"), np.int64)
    meta = np.asarray(rm.indices("meta_schema"), np.int64)
    rel = np.asarray(rm.indices("meta_margin_fs"), np.int64)
    half_ws, half_meta = ws.size // 2, meta.size // 2
    asm = {0: ws[:half_ws], 1: ws[half_ws:]}
    mm = {0: meta[:half_meta], 1: meta[half_meta:]}
    rl = {0: rel[:RELAY_N_PER_CLASS], 1: rel[RELAY_N_PER_CLASS:]}
    rng = np.random.default_rng([int(bridge.core_config.seed), zlib.crc32(b"metacog_margin_comparator")])
    u = {}
    for k in (0, 1):
        u[f"cmp_asm{k}_to_meta{k}"] = _het_projection(asm[k], mm[k], CMP_EXC[0], CMP_EXC[1], rng)
        u[f"cmp_asm{k}_to_relay{k}"] = _uniform_projection(asm[k], rl[k], CMP_REL)
        u[f"cmp_relay{k}_to_meta{1 - k}"] = _het_projection(rl[k], mm[1 - k], CMP_INH[0], CMP_INH[1], rng)
    return u


# The comparator is METACOG's second-order monitor (its own legacy margin topology): its relay region is listed as a
# metacog-side region for the lesion bookkeeping below.
METACOG_MARGIN = OrganDescriptor(
    key="metacog_margin", regions=("meta_margin_fs",), spec_fn=_comparator_spec,
    explicit_wiring_fn=_comparator_wiring, post_inject_fn=lambda b: b.set_plasticity_gate(CMP_GATE, 0.0),
    scaffold_residuals=("hand-set comparator + cross-edge weights (seed-42 calibration); not Hebbian-grown",))

XEDGE = CrossEdge(key=XEDGE_KEY, source_key="metacog", source_region="meta_schema",
                  target_key="curiosity", target_region="ask", init_weight=XEDGE_W, plastic=False,
                  learn_rule="none", freeze_rest=False)


def build_pool(seed: int, coupled: bool = True):
    """coupled=True: [metacog(het), curiosity, metacog_margin] + the ONE declared cross-edge.
    coupled=False: the SAME pool with the cross-edge absent (the byte-off / metacog-unchanged baseline)."""
    desc = [_metacog_het(), REGISTRY["curiosity"], METACOG_MARGIN]
    pool = merge_organs(desc, seed=int(seed), wire=True, cross_edges=([XEDGE] if coupled else None))
    pool.ensure_built()
    return pool


def build_bare_pool(seed: int):
    """[metacog(het), curiosity] ONLY — no comparator, no edge: the strongest metacog-unchanged baseline."""
    pool = merge_organs([_metacog_het(), REGISTRY["curiosity"]], seed=int(seed), wire=True)
    pool.ensure_built()
    return pool


class Recorder:
    """Wraps the pool's `_run_one_simulation_step` (the production metacog read calls it) and records, per step, the
    ASK spike count, plus a running sha256 over the metacog-side and comparator-side spike rasters."""

    def __init__(self, pool):
        b = pool.bridge
        rm = b.region_manager
        self.b = b
        self.ask = np.asarray(rm.indices("ask"), np.int64)
        names = rm.region_indices_dict()
        self.meta_idx = np.concatenate([np.asarray(rm.indices(n), np.int64) for n in META_REGIONS])
        self.cmp_idx = (np.concatenate([np.asarray(rm.indices(n), np.int64) for n in CMP_REGIONS])
                        if all(n in names for n in CMP_REGIONS) else None)
        self.on = False
        self._orig = b._run_one_simulation_step
        b._run_one_simulation_step = self._step
        self.reset()

    def reset(self):
        self.ask_counts = []
        self.h_meta = hashlib.sha256()
        self.h_cmp = hashlib.sha256()

    def _step(self):
        self._orig()
        if not self.on:
            return
        fs = np.asarray(to_host(self.b.cp_firing_states)).astype(bool)
        self.ask_counts.append(int(fs[self.ask].sum()))
        self.h_meta.update(np.packbits(fs[self.meta_idx]).tobytes())
        if self.cmp_idx is not None:
            self.h_cmp.update(np.packbits(fs[self.cmp_idx]).tobytes())


def _swapped_idx(idx):
    d = dict(idx)
    d["member_dev"] = {0: idx["member_dev"][1], 1: idx["member_dev"][0]}
    return d


def coupled_sweep(pool, org, rec, swap=False) -> dict:
    """Run metacog's OWN production read (`judge` -> `nmda_norm_margin`) at each evidence level and record what the
    ASK pool does during those very simulation steps. `swap=True` drives the evidence into the OTHER assembly
    (the class-symmetry anti-cheat)."""
    org.ensure_built()
    n_ask = rec.ask.size
    levels = []
    for ev in EVIDENCE_GRID:
        rec.reset()
        rec.on = True
        if swap:
            bal = float(nmda_norm_margin(org.bridge, org.xp, _swapped_idx(org.idx), org.snap, ev))
            conf = bool(bal >= org.threshold)
        else:
            j = org.judge(ev)
            bal, conf = float(j["balance"]), bool(j["confident"])
        rec.on = False
        c = np.asarray(rec.ask_counts, np.float64)
        assert c.size == READ_REPS * STEPS_PER_REP, (c.size, READ_REPS, STEPS_PER_REP)
        per_rep = c.reshape(READ_REPS, STEPS_PER_REP).sum(1) / n_ask / (STEPS_PER_REP * 1e-3)
        levels.append({"evidence": ev, "balance": bal, "confident": conf,
                       "ask_hz_per_rep": [float(x) for x in per_rep], "ask_hz": float(per_rep.mean()),
                       "meta_raster_sha256": rec.h_meta.hexdigest(),
                       "cmp_raster_sha256": (rec.h_cmp.hexdigest() if rec.cmp_idx is not None else None)})
    return {"levels": levels, "threshold": float(org.threshold)}


def _rank(x):
    from scipy.stats import rankdata
    return rankdata(np.asarray(x, np.float64))


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None                                   # UNDEFINED (a flat arm), never a score of 0
    return float(np.corrcoef(rx, ry)[0, 1])


def level_rho(sweep):
    return spearman([l["evidence"] for l in sweep["levels"]], [l["ask_hz"] for l in sweep["levels"]])


def perm_null(sweep, seed):
    """G4: Spearman rho over the PER-REP observations (11 levels x READ_REPS jittered reads) vs a null distribution
    of N_PERM random permutations of the evidence labels. One-sided p = P(null rho <= observed)."""
    ev = np.repeat([l["evidence"] for l in sweep["levels"]], READ_REPS)
    y = np.concatenate([l["ask_hz_per_rep"] for l in sweep["levels"]])
    obs = spearman(ev, y)
    if obs is None:
        return {"rho_obs": None, "p": None, "percentile": None, "n_perm": N_PERM, "n_obs": int(y.size)}
    rx, ry = _rank(ev), _rank(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rng = np.random.default_rng([int(seed), zlib.crc32(b"g4_perm_null")])
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = float(np.mean(rx[rng.permutation(rx.size)] * ry))
    n_le = int(np.sum(null <= obs + 1e-12))
    return {"rho_obs": float(obs), "p": float((1 + n_le) / (N_PERM + 1)),
            "percentile": float(100.0 * n_le / N_PERM), "n_perm": N_PERM, "n_obs": int(y.size),
            "null_q01": float(np.quantile(null, 0.01)), "null_q50": float(np.quantile(null, 0.5))}


def _range(sweep):
    v = [l["ask_hz"] for l in sweep["levels"]]
    return float(max(v) - min(v)), float(max(v))


def _digest(sweep) -> str:
    h = hashlib.sha256()
    for l in sweep["levels"]:
        h.update(np.asarray(l["ask_hz_per_rep"] + [l["balance"]], np.float64).tobytes())
        h.update(l["meta_raster_sha256"].encode())
        h.update((l["cmp_raster_sha256"] or "").encode())
    return h.hexdigest()


def _meta_exact(a, b) -> dict:
    """EXACT (==, and a sha256 of the float64 bytes) compare of metacog's own reads across two sweeps."""
    ba = np.asarray([l["balance"] for l in a["levels"]], np.float64)
    bb = np.asarray([l["balance"] for l in b["levels"]], np.float64)
    ca = [l["confident"] for l in a["levels"]]
    cb = [l["confident"] for l in b["levels"]]
    ra = [l["meta_raster_sha256"] for l in a["levels"]]
    rb = [l["meta_raster_sha256"] for l in b["levels"]]
    return {"balance_equal": bool(np.array_equal(ba, bb)),
            "balance_sha256_a": hashlib.sha256(ba.tobytes()).hexdigest(),
            "balance_sha256_b": hashlib.sha256(bb.tobytes()).hexdigest(),
            "threshold_equal": bool(a["threshold"] == b["threshold"]),
            "confident_equal": bool(ca == cb),
            "metacog_raster_equal": bool(ra == rb),
            "max_abs_balance_delta": float(np.max(np.abs(ba - bb)))}


def _curiosity_production_threshold(seed):
    """The production curiosity organ's OWN calibrated curious/incurious threshold (standalone build) — used only for
    the SECONDARY readout (does the synaptic drive alone cross the production decision boundary?)."""
    try:
        from research.runners.curiosity_production_organ import CuriosityProductionOrgan
        o = CuriosityProductionOrgan(seed=int(seed))
        o.ensure_built()
        return dict(o.calib)
    except Exception as e:  # secondary only; never blocks the verdict
        return {"error": f"{type(e).__name__}: {e}"}


def run_seed(seed: int, determinism: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    pool = build_pool(seed, coupled=True)
    org = MetacogProductionOrgan(seed=seed, shared=pool)
    rec = Recorder(pool)
    intact = coupled_sweep(pool, org, rec)
    swap = coupled_sweep(pool, org, rec, swap=True)

    b, xp = pool.bridge, pool.xp
    masks = cross_edge_masks(b, [XEDGE])
    n_edge = int(masks[XEDGE_KEY].sum())
    before = lesion_cross_edges(b, masks, xp)                  # G3: sever ONLY the metacog -> curiosity edge
    lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(before, dtype=b.cp_connections.data.dtype)

    # G8: lesion metacog's comparator RELAY (relay -> meta inhibition) with the cross-edge intact.
    rm = b.region_manager
    coo = b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    relay_mask = np.isin(row, np.asarray(rm.indices("meta_margin_fs"))) & np.isin(col, np.asarray(rm.indices("meta_schema")))
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    data2 = data.copy()
    data2[relay_mask] = 0.0
    b.cp_connections.data = xp.asarray(data2, dtype=b.cp_connections.data.dtype)
    relay_lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(data, dtype=b.cp_connections.data.dtype)
    restored = coupled_sweep(pool, org, rec)                   # integrity: restoration is exact

    nocoup = build_pool(seed, coupled=False)
    org_nc = MetacogProductionOrgan(seed=seed, shared=nocoup)
    rec_nc = Recorder(nocoup)
    nocoupled = coupled_sweep(nocoup, org_nc, rec_nc)
    byte_off = verify_byte_off(pool.bridge, nocoup.bridge, type("S", (), {"cross_edges": [XEDGE]})())

    bare = build_bare_pool(seed)
    org_b = MetacogProductionOrgan(seed=seed, shared=bare)
    rec_b = Recorder(bare)
    bare_sweep = coupled_sweep(bare, org_b, rec_b)

    # ── statistics ──
    rho = level_rho(intact)
    rng_i, peak_i = _range(intact)
    rng_l, peak_l = _range(lesion)
    rho_swap = level_rho(swap)
    rho_relay = level_rho(relay_lesion)
    g4 = perm_null(intact, seed)
    from tools.lab import attributable_to
    attrib = attributable_to(f"seed{seed} ASK dynamic range = the metacog->curiosity edge", rng_i, rng_l)

    m_lesion = _meta_exact(intact, lesion)                     # metacog's evidence still varies, unchanged, under G3
    m_nocoup = _meta_exact(intact, nocoupled)
    m_bare = _meta_exact(intact, bare_sweep)
    cmp_lesion_equal = ([l["cmp_raster_sha256"] for l in intact["levels"]]
                        == [l["cmp_raster_sha256"] for l in lesion["levels"]])
    digest = _digest(intact)
    restored_ok = _digest(restored) == digest

    det = {"checked": False}
    if determinism:
        cmd = [sys.executable, "-m", "research.runners._curiosity_metacog_conflict_xedge_derisk",
               "--digest-only", "--seeds", str(seed)]
        out = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, env=dict(os.environ))
        line = [x for x in out.stdout.splitlines() if x.startswith("DIGEST ")]
        other = line[-1].split()[-1] if line else None
        det = {"checked": True, "digest_main": digest, "digest_fresh_process": other,
               "equal": bool(other == digest), "rc": out.returncode}

    undefined = rho is None or rng_i < G1_MIN_RANGE_HZ
    checks = {
        "G1_monotone_rho<=-0.8": (not undefined) and rho <= G1_RHO_MAX,
        "G3_edge_lesion_collapses": (attrib is not None and attrib >= G3_ATTRIB_MIN
                                     and peak_l <= G3_LESION_MAX_FRAC * peak_i),
        "G3b_metacog_still_varies_under_lesion": (m_lesion["balance_equal"] and m_lesion["metacog_raster_equal"]
                                                  and cmp_lesion_equal
                                                  and (max(l["balance"] for l in lesion["levels"])
                                                       - min(l["balance"] for l in lesion["levels"])) > 0.0),
        "G4_perm_null_p<=0.01": g4["p"] is not None and g4["p"] <= G4_P_MAX,
        "G5_metacog_unchanged_EXACT": (m_nocoup["balance_equal"] and m_nocoup["threshold_equal"]
                                       and m_nocoup["confident_equal"] and m_nocoup["metacog_raster_equal"]
                                       and m_bare["balance_equal"] and m_bare["threshold_equal"]
                                       and m_bare["confident_equal"] and m_bare["metacog_raster_equal"]),
        "G6_determinism_fresh_process_hash": bool(det.get("equal")) if determinism else None,
        "G7_class_swap_monotone": rho_swap is not None and rho_swap <= G7_RHO_MAX,
        "G8_relay_lesion_abolishes_coupling": rho_relay is None or rho_relay > G8_RHO_MIN,
    }
    required = [k for k, v in checks.items() if v is not None]
    go = all(checks[k] for k in required)
    integrity = {"byte_off": byte_off, "restore_exact": restored_ok, "n_edge_synapses": n_edge,
                 "no_host_novelty": float(getattr(pool.bridge.core_config, "current_novelty_signal", 0.0) or 0.0) == 0.0,
                 "neuromod_installed": bool(getattr(pool.bridge.core_config, "enable_neuromodulator_subsystem", False))}

    # SECONDARY (pre-registered, NOT in GO): does the synaptic drive ALONE cross the production curious threshold?
    cal = _curiosity_production_threshold(seed)
    thr = cal.get("threshold_hz")
    secondary = {"production_curiosity_calib": cal}
    if thr is not None:
        uncertain = [l for l in intact["levels"] if not l["confident"]]
        confident = [l for l in intact["levels"] if l["confident"]]
        secondary.update({
            "crosses_at_some_uncertain_level": any(l["ask_hz"] >= thr for l in uncertain),
            "silent_below_threshold_at_all_confident_levels": all(l["ask_hz"] < thr for l in confident),
            "max_uncertain_ask_hz": max((l["ask_hz"] for l in uncertain), default=None),
            "max_uncertain_over_threshold": (max((l["ask_hz"] for l in uncertain), default=0.0) / thr) if thr else None})

    res = {"seed": seed, "go": bool(go), "checks": checks, "calibration_seed": seed == 42,
           "rho": rho, "rho_swap": rho_swap, "rho_relay_lesion": rho_relay, "g4": g4,
           "ask_range_hz": {"intact": rng_i, "lesion": rng_l}, "ask_peak_hz": {"intact": peak_i, "lesion": peak_l},
           "attributable_frac": attrib, "metacog_exact": {"vs_edge_lesion": m_lesion, "vs_no_edge_pool": m_nocoup,
                                                          "vs_bare_2organ_pool": m_bare},
           "comparator_raster_equal_under_edge_lesion": cmp_lesion_equal,
           "determinism": det, "integrity": integrity, "secondary": secondary,
           "arms": {"intact": intact, "edge_lesion": lesion, "class_swap": swap, "relay_lesion": relay_lesion,
                    "no_edge_pool": nocoupled, "bare_pool": bare_sweep},
           "elapsed_s": round(time.time() - t0, 1)}
    if verbose:
        print(f"[seed {seed}] rho={rho} swap={rho_swap} relay_lesion={rho_relay} "
              f"ask intact={[round(l['ask_hz'], 2) for l in intact['levels']]} "
              f"lesion_peak={peak_l:.2f} attrib={attrib} G4 p={g4['p']} pct={g4['percentile']} "
              f"metacog_exact(nocoup)={m_nocoup['balance_equal']}/{m_nocoup['metacog_raster_equal']} "
              f"bare={m_bare['balance_equal']}/{m_bare['metacog_raster_equal']} det={det.get('equal')} "
              f"GO={go} ({res['elapsed_s']}s)", flush=True)
        print(f"[seed {seed}] checks: {checks}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="first seed only")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--digest-only", action="store_true", help="internal: print the intact-arm digest and exit")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" /
                                         "_curiosity_metacog_conflict_xedge.json"))
    a = ap.parse_args()
    if a.digest_only:
        s = a.seeds[0]
        pool = build_pool(s, coupled=True)
        org = MetacogProductionOrgan(seed=s, shared=pool)
        rec = Recorder(pool)
        print("DIGEST", _digest(coupled_sweep(pool, org, rec)), flush=True)
        return 0
    seeds = [a.seeds[0]] if a.smoke else a.seeds
    t0 = time.time()
    print(f"[metacog->curiosity xedge] seeds={seeds} backend={os.environ.get('SIM_BACKEND')}", flush=True)
    rows = [run_seed(s, determinism=not a.no_determinism) for s in seeds]
    n_go = sum(1 for r in rows if r["go"])
    held_out = [r for r in rows if not r["calibration_seed"]]
    # PRECONDITIONS (tools.verdict): what must hold for ANY GO/NO-GO here to mean something. A failure makes the
    # verdict UNDEFINED, never a negative.
    from tools.verdict import Verdict
    v = Verdict("metacog margin comparator -> ONE cross-edge -> curiosity ASK (per-seed pre-registered gates)")
    for r in rows:
        s, integ = r["seed"], r["integrity"]
        bal = [l["balance"] for l in r["arms"]["intact"]["levels"]]
        v.require(f"seed{s} metacog balance varies across the evidence grid", (max(bal) - min(bal)) > 0.0)
        v.require(f"seed{s} no curiosity neuromodulator / host novelty on the pool",
                  bool(integ["no_host_novelty"] and not integ["neuromod_installed"]))
        v.require(f"seed{s} declared cross-edge wired (>0 synapses)", integ["n_edge_synapses"] > 0)
        v.require(f"seed{s} byte-off: base connectivity identical minus the edge", bool(integ["byte_off"]["PASS"]))
        v.require(f"seed{s} lesion restore exact (intact digest re-reads)", bool(integ["restore_exact"]))
        v.require(f"seed{s} every intact read ran READ_REPS x STEPS_PER_REP steps",
                  all(len(l["ask_hz_per_rep"]) == READ_REPS for l in r["arms"]["intact"]["levels"]))
    v.disabled("STDP / Hebbian / homeostasis / OU / conductance noise",
               "the production metacog pool config; the cross-edge and comparator are fixed-weight by design")
    decided = v.decide(bool(n_go == len(rows)))
    summary = {
        "mechanism": "metacog margin-comparator co-activation -> ONE cross-edge -> curiosity ASK pool (spiking)",
        "prereg": "docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md",
        "verdict": decided["status"], "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"], "disabled_processes": decided["disabled_processes"],
        "GO": bool(decided["go"]), "n_go": n_go, "n_seeds": len(rows),
        "held_out_n_go": sum(1 for r in held_out if r["go"]), "held_out_n": len(held_out),
        "operating_point": {"CMP_EXC": CMP_EXC, "CMP_REL": CMP_REL, "CMP_INH": CMP_INH, "XEDGE_W": XEDGE_W,
                            "RELAY_N_PER_CLASS": RELAY_N_PER_CLASS, "evidence_grid": EVIDENCE_GRID,
                            "N_PERM": N_PERM, "READ_REPS": READ_REPS},
        "per_seed": rows,
        "config": {"seeds": seeds, "smoke": a.smoke, "backend": os.environ.get("SIM_BACKEND")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print("=" * 100, flush=True)
    print(f"[metacog->curiosity xedge] VERDICT: {summary['verdict']} ({n_go}/{len(rows)} seeds; "
          f"held-out {summary['held_out_n_go']}/{summary['held_out_n']}) -> {a.out}", flush=True)
    return 0 if summary["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
