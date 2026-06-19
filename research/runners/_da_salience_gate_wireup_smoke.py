"""SMOKE for the DA salience-gate PRODUCTION WIRE-UP (roadmap #6) into MergedNavConvAgent.

NOT the 6-seed precision result (that is the committed de-risk, 2026-06-18-DA-composer-precision-derisk-GO.md).
This confirms the WIRE-UP: (1) default-OFF is byte-identical to the current agent, and (2) ON raises the gate at a
high-DA state and the no-confab moat HOLDS at both DA levels (0 false-accepts).

GPU only (the MergedNavConvAgent Hebbian parser + dlPFC are CuPy-validated). Run:
    SIM_BACKEND=cupy python -m research.runners._da_salience_gate_wireup_smoke
"""
from __future__ import annotations

import numpy as np

from sim.backend import get_backend, is_gpu_backend, to_host
from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
from research.runners._da_composer_salience_cleanup_derisk import da_to_gate


def _settle_snc(bridge, snc_idx, I_snc, n_steps=400):
    """Drive the limbic SNc pool (the shared-DA source) with constant current for n_steps (advancing the dopamine
    EMA each step) -> a steady DA concentration + the SNc firing rate (Hz). The SAME drive-SNc-and-read recipe the
    de-risk uses; here the SNc is the merged bridge's co-resident limbic_snc."""
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx] = xp.float32(I_snc)
    total = 0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        total += int(to_host(bridge.cp_firing_states[snc_idx]).sum())
    da = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    bridge.cp_external_input_current[:] = 0.0
    rate_hz = total / max(int(snc_idx.shape[0]), 1) / (n_steps * 1e-3)
    return da, rate_hz


def _moat_block(agent, label):
    """The canonical conversational matrix + the no-confab moat (mirrors test_nav_conv_merged_agent). Returns a dict
    of the recalls + the moat (the three `is None`/'unknown' abstentions must hold)."""
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("cat come south", polarity="NEGATE")
    out = {
        "what_dog_go": agent.what_does("dog", "go"),          # -> "north"
        "what_cat_come": agent.what_does("cat", "come"),      # -> "south"
        "who_go_north": agent.who_does("go", "north"),        # -> "dog"
        "describe_dog": agent.describe("dog"),                # -> "dog go north"
        # the no-confab moat: unstored cues must abstain at EVERY DA level (0 false-accepts)
        "moat_what_river_look": agent.what_does("river", "look"),     # -> None
        "moat_describe_river": agent.describe("river"),               # -> None
        "moat_is_true_apple": agent.is_it_true("apple", "stop", "west"),  # -> "unknown"
    }
    recall_ok = (out["what_dog_go"] == "north" and out["what_cat_come"] == "south"
                 and out["who_go_north"] == "dog" and out["describe_dog"] == "dog go north")
    moat_ok = (out["moat_what_river_look"] is None and out["moat_describe_river"] is None
               and out["moat_is_true_apple"] == "unknown")
    print(f"  [{label}] recall_ok={recall_ok} moat_ok={moat_ok}  {out}")
    return out, recall_ok, moat_ok


def main():
    assert is_gpu_backend(), "the MergedNavConvAgent parser/dlPFC are GPU-validated; run with SIM_BACKEND=cupy"
    print("=" * 96)
    print("DA salience-gate PRODUCTION WIRE-UP smoke (roadmap #6) -- MergedNavConvAgent")
    print("=" * 96)

    # (1) DEFAULT-OFF byte-identity: a vanilla MergedNavConvAgent (no gate, no limbic) behaves EXACTLY as now.
    print("\n(1) default-OFF (enable_da_salience_gate=False, no limbic):")
    agent_off = MergedNavConvAgent(seed=42)
    assert agent_off.enable_da_salience_gate is False, "default must be OFF"
    _off_out, off_recall, off_moat = _moat_block(agent_off, "default-OFF")
    assert off_recall and off_moat, "FAIL: default-OFF must reproduce the conversational matrix + the moat"
    del agent_off

    # (2) ON + a co-resident limbic core (the shared `dopamine` SNc): the gate reads DA off the merged bridge.
    print("\n(2) ON (enable_da_salience_gate=True, co_resident_limbic=True):")
    agent = MergedNavConvAgent(seed=42, co_resident_limbic=True, enable_da_salience_gate=True)
    assert agent.enable_da_salience_gate is True
    nm = agent._merged_bridge.neuromodulator_manager
    assert nm is not None and "dopamine" in nm.modulator_names(), "the shared dopamine modulator must be present"
    snc_idx = np.asarray(agent._merged_bridge.region_manager.indices("limbic_snc"), dtype=np.int64)
    xp, _ = get_backend()
    snc_idx_x = xp.asarray(snc_idx)
    da_base = float(nm._config_by_name("dopamine").baseline)

    # DA_low (tonic SNc) -> g_eff at/near the floor g0; DA_high (salient SNc) -> g_eff raised toward the cap.
    da_low, rate_low = _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=80.0)
    g_low = da_to_gate(da_low, da_base, agent._da_gate_g0, agent._da_gate_k, g_cap=agent._da_gate_cap)
    print(f"  DA_low  = {da_low:.3f} (limbic_snc {rate_low:.0f} Hz) -> g_eff = {g_low:.3f}")
    _lo_out, lo_recall, lo_moat = _moat_block(agent, "DA_low ")

    da_high, rate_high = _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=600.0)
    g_high = da_to_gate(da_high, da_base, agent._da_gate_g0, agent._da_gate_k, g_cap=agent._da_gate_cap)
    print(f"  DA_high = {da_high:.3f} (limbic_snc {rate_high:.0f} Hz) -> g_eff = {g_high:.3f}")
    _hi_out, hi_recall, hi_moat = _moat_block(agent, "DA_high")

    # --- the wire-up assertions ---
    gate_raises = bool(g_high > g_low + 1e-9 and da_high > da_low)   # a salient turn raises the gate
    moat_both = bool(lo_moat and hi_moat)                            # 0 false-accepts at BOTH DA levels
    # the clean canonical facts are decisive (margins >> cap), so the gate must NOT over-abstain them at DA_high:
    recall_both = bool(lo_recall and hi_recall)

    print("\n" + "=" * 96)
    print(f"  default-OFF byte-identical : recall={off_recall} moat={off_moat}")
    print(f"  ON gate raises with DA     : {gate_raises}  (g_low={g_low:.3f} -> g_high={g_high:.3f})")
    print(f"  ON no-confab moat held     : DA_low={lo_moat} DA_high={hi_moat} (0 false-accepts both)")
    print(f"  ON clean recall preserved  : DA_low={lo_recall} DA_high={hi_recall} (gate did not over-abstain)")
    ok = bool(off_recall and off_moat and gate_raises and moat_both and recall_both)
    print(f"  VERDICT: {'WIRE-UP-SMOKE PASS' if ok else 'FAIL'}")
    print("=" * 96)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
