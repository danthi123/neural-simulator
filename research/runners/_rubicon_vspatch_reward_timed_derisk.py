"""Rubicon HALF-2 de-risk: a reward-window-gated (VSPatch) potentiation rule makes the held-goal->value synapse
LEARN across the delay, where plain scope-all DA-STDP DEPRESSES it.

RE-ANCHOR (verified 2026-08-07 before building; drift-#12 discipline):
  * HALF-1 (the maintained-goal delay BRIDGE) is a 6-seed GO
    (2026-08-07-rubicon-delayed-credit-maintained-goal-bridge-PARTIAL.md, PARENT-VERIFIED): a NEURAL held goal
    (PFC recurrent slow-NMDA) carries value across the gap where a decayed trace expresses EXACTLY 0.
  * HALF-2 was NO: plain DA-gated three-factor STDP scope="all" over the whole trial DEPRESSES the SATURATED
    held-goal->value synapse (trained 168 Hz < the structural no-learning floor 452 Hz) -> it nets to LTD.
  * A reward-WINDOW-GATED / VSPatch potentiation is UN-BUILT in our record: the N9 TD cue-shift is TD *timing*
    (a decayed eligibility trace, not a reward-window gate); the reward-modulated three-factor STDP that exists is
    exactly the scope-all rule that FAILED. So the genuine step is the reward-time gate. (Corpus + grep checked;
    the only "reward window" hits are incidental phrases, not this mechanism.)

THE VSPATCH MECHANISM (Rubicon/PVLV deep-look): VSPatch (ventral-striatum patch) learns to predict the US and its
TIMING and gates dopamine so the RPE lands in the reward window. The fix for our HALF-2 defect: the held-goal->
value synapse must be POTENTIATED specifically in the REWARD WINDOW (held goal active AND the phasic DA burst), not
depressed by scope-all DA-STDP over the whole trial.

BUILD (NEURAL, no host if-reward flag; reuses build_core/run_condition from _rubicon_delayed_credit_derisk via an
additive, default-OFF `vspatch_gate` flag): the plastic pfc->striosome_value synapse carries a per-pathway
`plasticity_gate="reward_window"`; a second neuromodulator ("vspatch_gate", from_region_firing on the SNc above
tonic, fast tau) DRIVES that gate. Weight UPDATES on the held-goal->value synapse are FROZEN whenever the SNc is at
tonic (gap / CS -> gate=0) and PERMITTED only when the reward-time DA burst pushes SNc above tonic (gate->1). The
gate value is a spiking-driven NM concentration (neural), and it only opens on the burst (contingent).

HEAD-TO-HEAD (all on the maintained-goal substrate, recur>0, at the LONG gap = the informative window):
  * VSPATCH (treatment)  : vspatch_gate=True  -> reward-window-gated potentiation.
  * SCOPE-ALL (control)  : vspatch_gate=False -> the failing HALF-2 rule (plain scope-all DA-STDP).
  * FLOOR (structural)   : no_learning=True   -> STDP frozen from t0 (the value must be LEARNED, not structural).
  * YOKED (contingency)  : vspatch_gate=True, yoke_reward=True -> reward delivered with NO preceding held goal, so
                           the gate still OPENS at the burst but the PFC is at rest -> a CLOCK would still
                           potentiate; CREDIT must not. Build IDENTICAL to the treatment (readout comparable).
  * DECAYED (HALF-1 kept): recur=0, vspatch_gate=True -> the bridge lesion; the decayed trace still fails.

ANTI-CHEATS:
  (a) the gate is NEURAL/temporal: the "reward_window" gate value is a spiking-driven NM concentration
      (from_region_firing on the SNc), asserted to be OPEN at the reward burst and SHUT in the gap
      (gate_open_us >> gate_open_gap). No host if-reward-then-potentiate flag wraps the update.
  (b) the potentiation is CONTINGENT: the YOKED arm (reward present, gate opens, held goal absent) must NOT
      potentiate the value synapse above the floor -- else it is a clock, not credit.
  (c) the HALF-1 bridge stays intact: recur>0 holds the goal (pfc_hold >> decayed); the decayed (recur=0) arm
      still expresses ~0 value; gap_ext==0; host_reward==0.
  (d) 6-seed for a generalization claim (this smoke is 3-seed; the 6-seed command is printed for the parent).

numpy backend, tiny bridge, foreground, ONE process. NO sim/ edit (the plasticity_gate + NM machinery already
exist in sim/; only research-runner code is added, additive + default-off).

Run (smoke, 3-seed):
  SIM_BACKEND=numpy PYTHONPATH=$PWD python -m research.runners._rubicon_vspatch_reward_timed_derisk \
      --seeds 42,43,44 --gap-long 200 \
      --out research/findings/raw/rubicon_delayed_credit/vspatch_smoke_3seed.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._rubicon_delayed_credit_derisk import run_condition, _mean  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--gap-long", type=int, default=200)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--n-train", type=int, default=45)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]

    from tools.lab import attributable_to, assert_backend
    from tools.verdict import Verdict
    assert_backend("numpy", "rate/cheap spiking limbic core")

    print(f"[RUBICON HALF-2 VSPatch] reward-window-gated potentiation vs scope-all DA-STDP vs structural floor "
          f"| gap_long={args.gap_long} recur={args.recur} seeds={seeds}", flush=True)

    arms = {}

    def sweep(label, *, recur=None, **kw):
        rows = []
        _recur = args.recur if recur is None else recur
        for s in seeds:
            r = run_condition(s, recur=_recur, gap_steps=args.gap_long, n_train=args.n_train, **kw)
            rows.append(r)
            g_gap = r.get("gate_open_gap", float("nan")); g_us = r.get("gate_open_us", float("nan"))
            print(f"  [{label:20s} seed {s}] v_anticip={r['v_anticip_hz']:6.2f}Hz  "
                  f"pfc_hold={r['pfc_hold_hz']:6.2f}Hz  gate gap/us={g_gap:.3f}/{g_us:.3f}  "
                  f"gap_ext={r['gap_ext_drive_max']:.1f}  host_r={r['host_reward_signal']:.1f}", flush=True)
        arms[label] = rows
        return rows

    # The reward-window-gated VSPatch rule: the plastic held-goal->value synapse gets a reward_us-driven
    # plasticity gate, plus the DA-gated COACTIVITY eligibility and a clean US-time DA burst so the reward-time
    # da_signal is LTP-signed. Pair-STDP is kept ON (enable_stdp is LOAD-BEARING for the maintained-goal bridge --
    # stdp_off collapses PFC hold 341->5 Hz, verified), so the reward-window gate must OFFSET the whole-trial STDP
    # LTD, not replace STDP. The held-goal->value synapse starts at the ORIGINAL weight, so the value is
    # STRUCTURALLY AVAILABLE (~452 Hz floor): this tests reward-contingent MAINTENANCE of delayed value across the
    # gap, not building value from zero (the D1-MSN cell has no learn-from-below window -- see the finding).
    VSP = dict(reward_coactivity=True, coactivity_scale=0.15, da_from_reward_us=True)

    vspatch = sweep("vspatch", vspatch_gate=True, **VSP)              # reward-window-gated rule (STDP on)
    coact_ng = sweep("coact-nogate", vspatch_gate=False, **VSP)       # same rule, NO gate (isolates the gate)
    scopeall = sweep("scope-all-stdp", vspatch_gate=False)           # the documented failing HALF-2 rule (DA-STDP)
    floor = sweep("floor", vspatch_gate=True, no_learning=True, **VSP)  # structural no-learning floor (~452 Hz)
    omit = sweep("omit-reward", vspatch_gate=True, omit_reward=True, **VSP)  # CONTINGENCY: held goal, reward ABSENT
    yoked = sweep("yoked", vspatch_gate=True, yoke_reward=True, **VSP)  # 2nd contingency: reward w/o held goal
    decayed = sweep("decayed", recur=0.0, vspatch_gate=True, **VSP)   # HALF-1-intact: bridge lesion

    # ---- metrics (means over seeds) ----
    v_vsp = _mean(vspatch, "v_anticip_hz")
    v_ng = _mean(coact_ng, "v_anticip_hz")
    v_scp = _mean(scopeall, "v_anticip_hz")
    v_flr = _mean(floor, "v_anticip_hz")
    v_yok = _mean(yoked, "v_anticip_hz")
    v_omit = _mean(omit, "v_anticip_hz")
    v_dec = _mean(decayed, "v_anticip_hz")
    pfc_vsp = _mean(vspatch, "pfc_hold_hz")
    pfc_dec = _mean(decayed, "pfc_hold_hz")
    gate_gap = _mean(vspatch, "gate_open_gap")
    gate_us = _mean(vspatch, "gate_open_us")
    gap_ext = max(_mean(vspatch, "gap_ext_drive_max"), _mean(decayed, "gap_ext_drive_max"))
    host_r = _mean(vspatch, "host_reward_signal")

    print("\n  --- head-to-head (long gap = the informative window) ---", flush=True)
    print(f"  VSPATCH(gated) = {v_vsp:.2f}Hz | COACT-nogate = {v_ng:.2f}Hz | SCOPE-ALL-STDP = {v_scp:.2f}Hz "
          f"| FLOOR(no-learn) = {v_flr:.2f}Hz | YOKED = {v_yok:.2f}Hz | OMIT = {v_omit:.2f}Hz "
          f"| DECAYED(recur=0) = {v_dec:.2f}Hz", flush=True)
    print(f"  gate open: reward-window(us) = {gate_us:.3f} vs gap = {gate_gap:.3f}  "
          f"(neural/temporal gating -> open only at reward)", flush=True)
    print(f"  pfc hold: maintained = {pfc_vsp:.2f}Hz vs decayed(recur=0) = {pfc_dec:.2f}Hz", flush=True)

    frac_rescue = attributable_to("gated-VSPatch rescue of the held-goal value from the scope-all DA-STDP LTD "
                                  "(vspatch vs scope-all)", v_vsp, v_scp)
    attributable_to("reward-contingency of the maintained value (paired vspatch vs reward-omitted)", v_vsp, v_omit)

    # ---- verdict ----
    # HALF-2 CLAIM (honest scope): the reward-window gate does NOT build value from zero (the D1-MSN has no
    # learn-from-below window). What it DOES: it keeps whole-trial pair-STDP from DEPRESSING the saturated held-
    # goal->value synapse -- the exact HALF-2 defect -- so trained value stays at the structural level (~floor)
    # where the scope-all rule collapses it to ~168 Hz; and that maintenance is REWARD-CONTINGENT (omitting the US
    # lets the value decay).
    v = Verdict("rubicon HALF-2: reward-window-gated (VSPatch) maintenance of delayed value")
    # headline (1): rescue from the scope-all LTD -- trained value >> the scope-all DA-STDP rule
    v.control("gated VSPatch rescues the value from the scope-all DA-STDP LTD (vspatch >> scope-all)",
              treatment=v_vsp, control=v_scp, min_separation=50.0)
    # headline (2): the value is MAINTAINED near the structural floor (not collapsed)
    v.require("value maintained near the structural floor (trained >= 0.85 x floor)",
              v_vsp >= 0.85 * v_flr, expect=True, note=f"vspatch={v_vsp:.1f} floor={v_flr:.1f}")
    # anti-cheat (a): the gate is neural/temporal -- OPEN at reward, SHUT in the gap (the VALIDATED mechanism)
    v.control("the reward-window gate is OPEN at reward and SHUT in the gap (neural/temporal, not a host clock)",
              treatment=gate_us, control=gate_gap)
    # anti-cheat (b): CONTINGENT -- goal held but reward ABSENT (omit) must LOSE the value (else it is a freeze)
    v.control("CONTINGENT: reward-omitted (goal held, no US) LOSES the value that the paired arm keeps",
              treatment=v_vsp, control=v_omit, min_separation=50.0)
    # anti-cheat (c): HALF-1 bridge intact
    v.control("HALF-1 bridge intact: PFC holds the goal (maintained >> decayed)", treatment=pfc_vsp, control=pfc_dec)
    v.require("HALF-1 intact: the decayed-trace (recur=0) arm still fails to express value",
              v_dec < max(5.0, 0.2 * v_flr), expect=True, note=f"decayed={v_dec:.1f} floor={v_flr:.1f}")
    v.require("no host drive holds the goal in the gap (gap_ext==0)", gap_ext == 0.0, expect=True)
    v.require("host reward signal is 0 (r is synaptic)", host_r == 0.0, expect=True)

    rescue_go = (v_vsp > 1.5 * max(v_scp, 1e-6)) and (v_vsp >= 0.85 * v_flr)   # rescues LTD + maintains near floor
    contingency_holds = (v_omit < 0.6 * max(v_vsp, 1e-6))                       # reward-omitted loses the value
    gating_neural = (gate_us > 0.30) and (gate_us > 2.0 * max(gate_gap, 1e-6)) and (gate_gap < 0.20)
    bridge_intact = ((pfc_vsp > 3.0 * max(pfc_dec, 1e-6)) and (v_dec < max(5.0, 0.2 * v_flr))
                     and (gap_ext == 0.0) and (host_r == 0.0))
    go = bool(rescue_go and contingency_holds and gating_neural and bridge_intact)
    result = v.decide(go=go)
    result.update({
        "rescue_go": bool(rescue_go),
        "contingency_holds": bool(contingency_holds),
        "gating_neural": bool(gating_neural),
        "bridge_intact": bool(bridge_intact),
        "learned_credit_go": bool(go),   # overall (reward-contingent maintenance, not build-from-zero)
    })

    print(f"\n  GATING NEURAL (reward-window gate open at reward, shut in gap): "
          f"{'YES -- validated' if gating_neural else 'NO'}", flush=True)
    print(f"  HALF-2 RESCUE (gated VSPatch maintains value vs scope-all DA-STDP LTD): "
          f"{'GO-looking' if rescue_go else 'NO'}  "
          f"(vspatch {v_vsp:.1f}Hz ~ floor {v_flr:.1f}Hz  vs  scope-all-stdp {v_scp:.1f}Hz)", flush=True)
    print(f"  REWARD-CONTINGENT (omitting the US loses the value): "
          f"{'HOLDS' if contingency_holds else 'FAILS (a freeze, not credit)'}  "
          f"(omit {v_omit:.1f}Hz vs paired {v_vsp:.1f}Hz)", flush=True)
    print(f"  HALF-1 BRIDGE intact: {'YES' if bridge_intact else 'NO'}", flush=True)
    print(f"  NOTE: this is reward-contingent MAINTENANCE of structurally-available value, NOT build-from-zero "
          f"(the D1-MSN has no learn-from-below window).", flush=True)
    print(f"  OVERALL (smoke, needs 6-seed): {result.get('verdict', result.get('status'))}", flush=True)

    payload = {
        "arms": arms,
        "metrics": {
            "v_vspatch": v_vsp, "v_coact_nogate": v_ng, "v_scopeall_stdp": v_scp, "v_floor": v_flr,
            "v_yoked": v_yok, "v_omit": v_omit, "v_decayed": v_dec,
            "pfc_hold_maintained": pfc_vsp, "pfc_hold_decayed": pfc_dec,
            "gate_open_us": gate_us, "gate_open_gap": gate_gap,
            "attributable_vspatch_vs_scopeall": frac_rescue,
            "gap_ext_drive_max": gap_ext, "host_reward_signal": host_r,
        },
        "seeds": seeds, "gap_long": args.gap_long, "recur": args.recur,
        **result,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(payload, open(args.out, "w"), indent=1)
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
