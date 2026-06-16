"""Step-1 de-risk (throwaway): does the MERGED bridge's nav cascade SELECT MOVES out of the box?

The KNOWN INTEGRATION RISK (per the controller scoping): `navigate_to_see_then_answer.build_navsee_bridge` builds
its nav cascade with `build_bg_brain_regions(n_cortex=100, enable_spiking_wta_readout=True)` (so `sel_X` exists and
`_cascade_select_move` reads it), but `build_merged_nav_conv_bridge` calls `build_bg_brain_regions(n_cortex=...)`
with DEFAULT kwargs (no `sel_X`). This probe answers, BEFORE the full loop:
  (a) which selection regions exist on the merged bridge (sel_X? motor_X?),
  (b) can a neural move be selected on it (via the navsee `_cascade_select_move`, building the navsee-style handles
      against the merged bridge), AND
  (c) can a percept be rendered into the bare cortex_it + grounded (the cheap-first pattern) IN the same bridge.

Run BOTH builder modes (default + the proposed enable_spiking_wta_readout pass-through) so the report states whether
the default cascade already selects, or the WTA kwarg is needed. CPU/numpy.

  SIM_BACKEND=numpy python -m research.runners._step1_merged_cascade_move_probe
"""
from __future__ import annotations

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
from research.runners.g11_bg_runner import ACTION_NAMES
from research.runners.funcint_perception_to_memory_probe import OBJECT_WORDS, N_OBJECTS
from research.runners._step3_grounded_codes_production_composer_derisk import (
    read_cortex_it_rate, _projection, grounded_phases,
)
# the navsee selection idiom (reuse-by-import); we build its handles against the MERGED bridge.
from research.runners.navigate_to_see_then_answer import _cascade_select_move


def _build_navsee_style_handles(bridge):
    """Mirror navigate_to_see_then_answer.build_navsee_bridge's readout + tonic handle construction (lines 280-305)
    against the MERGED bridge: prefer sel_X (spiking-WTA), else motor_X; collect the GPe/GPi/STN/SNc/thal tonics."""
    xp, _ = get_backend()
    rm = bridge.region_manager
    region_names = set(rm.region_indices_dict())
    h = {"with_body": True}
    if all(f"sel_{a}" in region_names for a in ACTION_NAMES):
        h["readout_region"] = "sel"
        h["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"sel_{a}")), dtype=np.int64))
                            for a in ACTION_NAMES}
    elif all(f"motor_{a}" in region_names for a in ACTION_NAMES):
        h["readout_region"] = "motor"
        h["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"motor_{a}")), dtype=np.int64))
                            for a in ACTION_NAMES}
    else:
        h["readout_region"] = None
        return h, region_names
    h["cortex_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64))
                       for a in ACTION_NAMES}

    def _ridx(name):
        return xp.asarray(np.asarray(list(rm.indices(name)), dtype=np.int64)) if name in region_names else None
    h["cascade_tonic"] = []
    for a in ACTION_NAMES:
        for name, pa in ((f"gpe_{a}", 150.0), (f"gpe_arky_{a}", 120.0), (f"gpi_{a}", 110.0), (f"thal_{a}", 300.0)):
            ii = _ridx(name)
            if ii is not None:
                h["cascade_tonic"].append((ii, float(pa)))
    for name, pa in (("stn", 150.0), ("snc", 150.0)):
        ii = _ridx(name)
        if ii is not None:
            h["cascade_tonic"].append((ii, float(pa)))
    return h, region_names


def probe(seed, enable_wta):
    vocab = list(OBJECT_WORDS) + ["chase", "near"]
    kwargs = dict(seed=seed, vocab=vocab, n_cortex=100, co_resident_rf=True, rf_D=64, co_resident_perception=True)
    if enable_wta:
        kwargs["enable_spiking_wta_readout"] = True
    bridge, handles = build_merged_nav_conv_bridge(**kwargs)
    h, region_names = _build_navsee_style_handles(bridge)
    has_sel = all(f"sel_{a}" in region_names for a in ACTION_NAMES)
    has_motor = all(f"motor_{a}" in region_names for a in ACTION_NAMES)
    print(f"\n[step1] ===== seed {seed}  enable_spiking_wta_readout={enable_wta} =====")
    print(f"[step1] sel_X present: {has_sel} | motor_X present: {has_motor} | readout_region={h.get('readout_region')}")

    if h.get("readout_region") is None:
        print("[step1] NO selection region -> cannot select a neural move with this builder mode.")
        return {"enable_wta": enable_wta, "has_sel": has_sel, "has_motor": has_motor, "selected": None}

    # (b) try to select neural moves toward each cardinal; report which steer dirs produce a clear winner.
    selections = {}
    for steer in ACTION_NAMES:
        chosen, counts = _cascade_select_move(bridge, h, steer)
        selections[steer] = {"chosen": chosen, "counts": {a: int(counts[a]) for a in ACTION_NAMES}}
        print(f"[step1]   steer {steer}: chosen={chosen}  counts={selections[steer]['counts']}")
    n_clear = sum(1 for s in selections.values() if s["chosen"] is not None)

    # (c) render a percept into bare cortex_it + ground it (the cheap-first pattern) in the SAME bridge.
    rm = bridge.region_manager
    it_indices = np.asarray(rm.indices("cortex_it"))
    proj = _projection(64, int(it_indices.size), seed)
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process, cc.ou_std_current_pA = True, 20.0
    try:
        rate0 = read_cortex_it_rate(bridge, it_indices, 0)
        ph0 = grounded_phases(rate0, proj)
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    grounded_ok = bool(np.isfinite(ph0).all() and rate0.sum() > 0)
    print(f"[step1]   cortex_it grounded: rate-sum={rate0.sum():.1f} phases-finite={np.isfinite(ph0).all()} "
          f"(grounded_ok={grounded_ok})")
    print(f"[step1]   => {n_clear}/4 steer dirs gave a clear neural winner; grounding {'OK' if grounded_ok else 'FAIL'}")
    return {"enable_wta": enable_wta, "has_sel": has_sel, "has_motor": has_motor,
            "n_clear_selections": n_clear, "selections": selections, "grounded_ok": grounded_ok}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wta", action="store_true", help="also probe with the enable_spiking_wta_readout pass-through")
    args = ap.parse_args()
    _, backend = get_backend()
    print(f"[step1] backend={backend} — does the MERGED cascade select moves out of the box (default) vs +WTA?")
    r_default = probe(42, enable_wta=False)
    print("\n[step1] ============ SUMMARY ============")
    print(f"[step1] DEFAULT builder: sel_X={r_default['has_sel']} motor_X={r_default['has_motor']} "
          f"clear-selections={r_default.get('n_clear_selections')}/4 grounded_ok={r_default.get('grounded_ok')}")
    if args.wta:
        r_wta = probe(42, enable_wta=True)
        print(f"[step1] +WTA    builder: sel_X={r_wta['has_sel']} motor_X={r_wta['has_motor']} "
              f"clear-selections={r_wta.get('n_clear_selections')}/4 grounded_ok={r_wta.get('grounded_ok')}")


if __name__ == "__main__":
    main()
