"""Vikunja #203 flip-prep: PRODUCTION drift probe for BRAIN_ENFORCE_PLASTIC_MASK / enforce_plastic_mask_in_hebbian.

Builds the SAME production brain the live webapp chat uses (research.runners.brain_chat_tui's tiny-demo
ChatBrain, routed through the SHARED full-faculty pipeline webapp.brain_reply.reply_over_chat -- the exact path
`run_repl` in brain_chat_tui.py uses, see that module's 2026-08-27 "THE SHARED FULL-FACULTY PIPELINE" note), runs
a handful of chat turns, and reports, per NAMED region-pathway, the max|dw| on synapses whose
`cp_synapse_plastic_mask` entry is False (declared non-plastic: `RegionPathway(plastic=False)` /
`BrainRegion(plastic_internal=False)`), plus a plastic-pathway control (should move regardless of the flag) and
the turn-by-turn reply text (to see whether the flag changes conversation OUTPUT, not just weights).

WHY THIS ORGAN: `research/runners/comprehension_production_organ.py` (comprehension, default-ON in live chat --
`comprehension_enabled()`) is the organ the board's read-driven-drift finding (13.8->56.1 over 30 reads) named.
Its `SpikingRoleCompetition` (`research/runners/_phaseB_multicue_competition_spiking_derisk.py`) builds
`sel_{role}` / `sel_FS_{role}` regions as `plastic_internal=False` and wires `sel_r->sel_FS_r` / `sel_FS_r->sel_s`
as `RegionPathway(..., plastic=False)` WITH NO named `plasticity_gate` -- exactly the exposed shape
`enforce_plastic_mask_in_hebbian` targets (a structural plastic=False with no zeroed named gate). The organ fires
on every user turn that parses as a 3-content-token transitive assertion (`corg.judge(msg, ...)`,
webapp/server.py ~L6009-6050), so a few "subject verb object" turns exercise the exact bug surface.

USAGE (run ONCE per arm, in a FRESH process -- `_get_comprehension_organ`/`get_organ` is a process-global
singleton, and the enforcement gate is read at Hebbian-step time via `os.environ`/`cfg`, so the two arms must be
separate processes for a clean "before this process ever ran a step" comparison). `research/findings/raw/` is not
a Python package (no `__init__.py`), so this runs as a plain script with the repo root on PYTHONPATH, not `-m`:

    cd <repo_root>
    SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python \\
        research/findings/raw/_plastic_mask_flip_prep/prod_drift_probe.py --seed=42 --arm=off \\
        --out=research/findings/raw/_plastic_mask_flip_prep/smoke_off_s42.json
    SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= PYTHONPATH=. BRAIN_ENFORCE_PLASTIC_MASK=1 .venv/bin/python \\
        research/findings/raw/_plastic_mask_flip_prep/prod_drift_probe.py --seed=42 --arm=on \\
        --out=research/findings/raw/_plastic_mask_flip_prep/smoke_on_s42.json

Backend: numpy (CPU) throughout -- the tiny-demo composer_kind='rf' (numpy fast-path recall) needs no GPU, and
the comprehension organ's SpikingRoleCompetition bridge is a small (~tens of neurons) net that fires fine on
numpy (unlike the STDP/Hebbian-variant SHA scenarios in tests/_plastic_mask_*_scenario.py, which need cupy at
that MUCH smaller 10-neuron scale -- this organ is bigger and the drive is a strong constant assertion read, not
a coincidence-timed spike pair, so numpy firing is not the concern here; verified empirically below via
`fired_any`/`judged_any` in the output).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np


def _to_host(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


# Turns chosen to be 3-content-token transitive assertions matching the tiny-demo's OWN facts (so comprehension
# resolves confidently rather than abstaining -- see comprehension_production_organ.py's OOV/ambiguous-margin
# abstain path) and to repeat (like the board's "30 reads" drift measurement) so any per-turn drift accumulates
# into a measurable signal over a SMOKE-sized handful of turns.
_TURNS = [
    "dog chase cat",
    "cat eat fish",
    "dog chase cat",
    "cat eat fish",
    "dog chase cat",
    "what does the dog chase",
]


def _pathway_breakdown(bridge, coo, mask_host):
    """Group synapse indices by (from_region, to_region) using region_manager.indices(), for readable per-pathway
    reporting. Returns {"from->to": {"n": int, "plastic": bool}} plus the raw index arrays needed by the caller
    to compute max|dw| per group."""
    rm = getattr(bridge, "region_manager", None)
    out = {}
    if rm is None:
        return out
    try:
        region_names = list(rm.regions.keys()) if hasattr(rm, "regions") else []
    except Exception:
        region_names = []
    idx_by_region = {}
    for rn in region_names:
        try:
            idx_by_region[rn] = set(int(i) for i in rm.indices(rn))
        except Exception:
            continue
    pathways = []
    try:
        pathways = rm.pathways()
    except Exception:
        pathways = []
    row = coo.row
    col = coo.col
    for pw in pathways:
        fr = getattr(pw, "from_region", None)
        to = getattr(pw, "to_region", None)
        if fr not in idx_by_region or to not in idx_by_region:
            continue
        fr_set = idx_by_region[fr]
        to_set = idx_by_region[to]
        sel = np.array([
            (int(r) in fr_set) and (int(c) in to_set)
            for r, c in zip(row, col)
        ], dtype=bool)
        if not sel.any():
            continue
        key = f"{fr}->{to}"
        out[key] = {
            "indices": np.where(sel)[0],
            "plastic": bool(getattr(pw, "plastic", True)),
            "named_gate": getattr(pw, "plasticity_gate", None),
        }
    return out


def run(seed: int, arm: str):
    from research.runners.brain_chat_tui import _build_tiny_demo, ChatBrain, StubRenderer
    from webapp import brain_reply as _shared
    import research.runners.comprehension_production_organ as _CO

    assert arm in ("off", "on")
    enforced = os.environ.get("BRAIN_ENFORCE_PLASTIC_MASK", "").strip().lower() not in ("", "0", "false", "no", "off")

    agent, aliases, n_facts = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                               composer_kind="rf", integrated_loop=False)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())

    assert _CO.comprehension_enabled(), "comprehension must be default-ON for this probe to test anything"
    corg = _CO.get_organ(seed=seed)
    corg.ensure_built()
    bridge = corg.comp.bridge
    coo = bridge.cp_connections.tocoo(copy=False)
    mask = getattr(bridge, "cp_synapse_plastic_mask", None)
    mask_present = mask is not None
    mask_host = _to_host(mask) if mask_present else None

    groups = _pathway_breakdown(bridge, coo, mask_host)

    w0 = _to_host(bridge.cp_connections.data).copy()

    replies = []
    judged_any = False
    for msg in _TURNS:
        payload = _shared.reply_over_chat(chat, msg, source="prod-drift-probe", brain="prod-drift-probe",
                                          renderer="raw", rich=False, session="probe")
        replies.append({"msg": msg, "answer": payload.get("answer", ""),
                        "abstained": bool(payload.get("abstained"))})
        cinfo = payload.get("comprehension")
        if cinfo is not None:
            judged_any = True

    w1 = _to_host(bridge.cp_connections.data)
    dw = np.abs(w1 - w0)

    per_pathway = {}
    for key, g in groups.items():
        idx = g["indices"]
        per_pathway[key] = {
            "n_synapses": int(idx.size),
            "declared_plastic": g["plastic"],
            "named_gate": g["named_gate"],
            "max_abs_dw": float(dw[idx].max()) if idx.size > 0 else 0.0,
        }

    frozen_max_dw = float(dw[~mask_host].max()) if (mask_present and (~mask_host).any()) else 0.0
    plastic_max_dw = float(dw[mask_host].max()) if (mask_present and mask_host.any()) else float(dw.max())

    return {
        "arm": arm,
        "enforced_flag_read": enforced,
        "seed": seed,
        "n_facts": n_facts,
        "mask_present": mask_present,
        "n_frozen_synapses": int((~mask_host).sum()) if mask_present else 0,
        "n_plastic_synapses": int(mask_host.sum()) if mask_present else int(coo.row.size),
        "frozen_max_abs_dw": frozen_max_dw,
        "plastic_max_abs_dw": plastic_max_dw,
        "per_pathway": per_pathway,
        "judged_any": judged_any,
        "replies": replies,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", choices=["off", "on"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.seed, a.arm)
    js = json.dumps(res, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(js)
    print(js)
