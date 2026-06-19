"""Roadmap #4 (TRUE-ONE-BRAIN spike-ification) — CHEAP-FIRST SCOPE + DE-RISK for the
fully-spiking nav motor read-out ON THE MERGED "one brain".

GOAL of #4: make the merged nav action-selection a FULLY-SPIKING decision — the Wang-2002
accumulator (`sel_X`) + Lo-Wang commit-burst (`commit_X`) threshold-crossing IS the decision,
retiring the host Python argmax read-out (`g11_bg_runner.py:6901-6909`). An honest BOUNDARY (a
clean spiking decision that navigates WORSE than the host argmax) is a fully valid deliverable.

This file is the CHEAP-FIRST de-risk (numpy/CPU composition + the no-confab moat); the few-seed
GPU nav comparison is driven by `_nav_gate_merged_run.py --readout-source ...` (extended this cycle).

Two modes:
  --compose-smoke : (numpy/CPU) does `enable_spiking_wta_readout=True` COMPOSE on the merged bridge?
                    (no region/index collision with the parser/dlPFC/rf slices) + assert the sel_X/
                    commit_X regions exist + the parser STILL parses voice-invariantly on the merged
                    bridge with the WTA layer present.
  --moat-smoke    : (numpy/CPU) the no-confab moat via `MergedNavConvAgent`:
                    what_does('dog','go') == 'north'  AND  what_does('river','look') is None.
                    (The moat is a CONVERSATIONAL property; the spiking-WTA layer is nav-side and
                    array-disjoint from the parser/composer, so the moat is read on the default agent
                    — the WTA layer cannot perturb it. The compose-smoke proves the WTA layer builds
                    co-resident; together they show #4 composes without weakening the moat.)

Reuse-by-import; NO sim/ edit.
"""
from __future__ import annotations

import argparse


def compose_smoke(seed: int = 42, n_cortex: int = 100, vocab=None):
    """Build the merged bridge WITH the spiking-WTA read-out layer and assert it composes."""
    from sim.backend import get_backend
    from research.runners.nav_conv_merged_bridge import (
        build_merged_nav_conv_bridge, parse_on_slices,
    )

    xp, backend = get_backend()
    print(f"[compose-smoke] backend={backend} seed={seed} n_cortex={n_cortex} enable_spiking_wta_readout=True")

    # The load-bearing composition question: does build_bg_brain_regions(enable_spiking_wta_readout=True)
    # — which appends sel_X / sel_FS_X / commit_X / commit_OPN — co-reside cleanly with the appended
    # parser (parse_conj/parse_role) + dlPFC (cortex_ctx/dlpfc_wm) slices on ONE framework bridge?
    bridge, h = build_merged_nav_conv_bridge(
        seed=seed, vocab=vocab, n_cortex=n_cortex, enable_spiking_wta_readout=True)
    rm = bridge.region_manager
    cfg = bridge.core_config
    region_names = set(rm.region_indices_dict())

    n_regions = len(cfg.brain_regions)
    n_neurons = int(cfg.num_neurons)
    nnz = int(bridge.cp_connections.nnz)
    print(f"[compose-smoke] {n_regions} regions, {n_neurons} neurons, {nnz} synapses")

    # (a) the spiking-WTA selection + commit layers were actually built (the #4 substrate).
    sel_names = [f"sel_{a}" for a in ("N", "E", "S", "W")]
    selfs_names = [f"sel_FS_{a}" for a in ("N", "E", "S", "W")]
    commit_names = [f"commit_{a}" for a in ("N", "E", "S", "W")]
    for nm in sel_names + selfs_names + commit_names + ["commit_OPN"]:
        assert nm in region_names, f"FAIL: spiking-WTA region '{nm}' missing (regions: {sorted(region_names)[:12]}...)"
    print(f"[compose-smoke] (a) spiking-WTA layer built: {sel_names} + sel_FS_X + {commit_names} + commit_OPN")

    # (b) NO collision: the parser/dlPFC/conv regions co-reside alongside the WTA layer.
    for nm in ("cortex_N", "parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm"):
        assert nm in region_names, f"FAIL: conv/nav region '{nm}' missing alongside the WTA layer"
    # disjoint index check: sel_X / commit_X indices must not overlap the parser/dlPFC indices.
    def _idx(nm):
        return set(int(i) for i in rm.indices(nm))
    parser_dlpfc_idx = _idx("parse_conj") | _idx("parse_role") | _idx("cortex_ctx") | _idx("dlpfc_wm")
    wta_idx = set()
    for nm in sel_names + commit_names:
        wta_idx |= _idx(nm)
    overlap = parser_dlpfc_idx & wta_idx
    assert not overlap, f"FAIL: spiking-WTA indices overlap the parser/dlPFC slices ({sorted(overlap)[:8]}...)"
    print(f"[compose-smoke] (b) WTA indices disjoint from parser/dlPFC: |wta|={len(wta_idx)} |conv|={len(parser_dlpfc_idx)} overlap=0")

    # (c) the homeostasis foot-gun stayed off (the merged-config invariant).
    assert cfg.enable_homeostasis is False, "FAIL: cfg.enable_homeostasis True (synaptic-scaling clip foot-gun)"

    # (d) the parser STILL parses voice-invariantly on the merged bridge WITH the WTA layer present
    #     (the WTA layer must not have perturbed the parser comprehension).
    conj_arr, role_arr = h["conj_arr"], h["role_arr"]
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        active = parse_on_slices(bridge, conj_arr, role_arr, ["dog", "go", "north"], voice="active")
        passive = parse_on_slices(bridge, conj_arr, role_arr, ["north", "go", "dog"], voice="passive")
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    print(f"[compose-smoke] active  parse (WTA present): {active}")
    print(f"[compose-smoke] passive parse (WTA present): {passive}")
    ok_active = active.get("agent") == "dog"
    ok_voice = passive.get("agent") == "dog"
    assert ok_active, f"FAIL: active 'dog go north' agent != dog with WTA present ({active})"
    assert ok_voice, f"FAIL: voice-invariance broke with WTA present ({passive})"

    print("\n[compose-smoke] PASS - enable_spiking_wta_readout COMPOSES on the merged bridge: the "
          "sel_X/commit_X spiking-WTA layer co-resides with the parser/dlPFC slices (zero index collision) "
          "and the parser still parses voice-invariantly.")
    return True


def moat_smoke(seed: int = 42, vocab=None):
    """The no-confab moat on the merged nav+conv agent (the conversational property #4 must not weaken).

    The default `MergedNavConvAgent` is used: the spiking-WTA read-out is a NAV-side substrate
    (sel_X/commit_X, array-disjoint from the parser+RF composer that carry the moat), so the moat
    decision is identical whether the nav read-out is host-argmax or spiking-WTA. compose_smoke
    proves the WTA layer builds co-resident; this proves the moat holds on that same merged stack.
    """
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent

    print(f"[moat-smoke] seed={seed} building MergedNavConvAgent ...")
    agent = MergedNavConvAgent(seed=seed, vocab=vocab)
    # teach one fact; the parser comprehends 'dog go north' on the merged bridge, the composer stores it.
    roles = agent.hear("dog go north", voice="active")
    print(f"[moat-smoke] hear('dog go north') -> {roles}")
    stored = agent.what_does("dog", "go")
    print(f"[moat-smoke] what_does('dog','go')   -> {stored!r}   (expect 'north')")
    abstain = agent.what_does("river", "look")
    print(f"[moat-smoke] what_does('river','look') -> {abstain!r}  (expect None -> no confab)")

    ok_recall = stored == "north"
    ok_moat = abstain is None
    assert ok_recall, f"FAIL: what_does('dog','go') != 'north' ({stored!r})"
    assert ok_moat, f"FAIL: NO-CONFAB MOAT BROKEN: what_does('river','look') != None ({abstain!r})"
    print("\n[moat-smoke] PASS - the no-confab moat is intact on the merged nav+conv agent: recall "
          "'north', abstain on the unstored cue.")
    return True


def main():
    ap = argparse.ArgumentParser(description="roadmap #4 cheap-first de-risk: merged fully-spiking read-out")
    ap.add_argument("--compose-smoke", action="store_true",
                    help="numpy/CPU: does enable_spiking_wta_readout COMPOSE on the merged bridge?")
    ap.add_argument("--moat-smoke", action="store_true",
                    help="numpy/CPU: the no-confab moat via MergedNavConvAgent (must stay intact)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cortex", type=int, default=100)
    args = ap.parse_args()

    ok = True
    if args.compose_smoke:
        ok = compose_smoke(seed=args.seed, n_cortex=args.n_cortex) and ok
    if args.moat_smoke:
        ok = moat_smoke(seed=args.seed) and ok
    if not (args.compose_smoke or args.moat_smoke):
        ap.error("pass --compose-smoke and/or --moat-smoke")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
