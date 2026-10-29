# cs-race-f1h step 0 - P0 engine-support answer (Hermes, 2026-10-26)

Packet: `research/coordination/cs-race-f1h-PACKET.md` (ruling cs-race-f1h-1024; un-parked by
RULING batch-1026 C1: "cs-race f1h step 0 ... runs on the pool now"). Status at wake 13:38:
0 f1h claims rows today, pool empty, no f1h artifacts - nothing had ever been queued. P0 (the
packet's read-only first sub-step) is DONE here; P1 + the frozen grid still to run.

## P0: can cs_race_f1's bridge host the BDSP apical on mention-slot cells only?

VERDICT: HOSTABLE - not STOP-ENGINE. No `sim/` edit needed: every flag below is a CoreSimConfig
field defaulting False (today's circuit byte-identical, comments at bridge.py:9664, 10590-10593).

### Which flags allocate it
- `enable_bdsp` (sim/config.py:613) -> guard `sim/bridge.py:10594` (`and cp_connections.nnz > 0`;
  the probe injects explicit wiring, research/runners/cs_race_f1.py:241). Allocates the per-neuron
  `cp_v_apical` at `apical_E_rest`: bridge.py:10611-10612 (declared :481), plus cp_bdsp_* arrays
  :10597-10602.
- `bdsp_apical_bistable` (config.py:791) -> the held-plateau terms bridge.py:10666-10684:
  v-gated self-regen (params read :10667-10676) + KIR down-state (:10677-10683).
- `bdsp_apical_couples_soma` (config.py:782) + `bdsp_apical_soma_g` (:783) -> apical->soma
  coupling bridge.py:9665-9671: adds `g * scale * max(v_apical - E_rest, 0)` to
  total_input_current_pA (exactly 0 for a cell at rest).
- NOT needed: `enable_two_compartment_dap` (config.py:549) = the separate rung-4 dAP routing
  (bridge.py:9602-9606); stays False for f1h.

### Which array carries an apical drive
`cp_bdsp_apical_drive` - declared bridge.py:504; the runner supplies a raw per-neuron top-down
current, integrated into cp_v_apical (bridge.py:10607-10612; comment :10608), raw restored each
step (:10686). The packet's pulse = write the amplitude into `cp_bdsp_apical_drive[slot_idx]`
during the 50/150 ms window, 0 elsewhere - same per-index idiom the probe already uses for
`cp_external_input_current` (cs_race_f1.py:249).

### Restriction to mention-slot cells only
- Drive: EXACT - per-neuron array, written only on `self.ment` indices (cs_race_f1.py:157).
- Dynamics: gated by GLOBAL cfg flags; there is NO per-index mask flag. Undriven cells stay
  inert by construction: init v_apical = E_rest (bridge.py:10612) -> coupling term 0 (:9670);
  self-regen silent below v_hold (:10670-10671); comment :10592 "at rest the apical is silent =>
  P==Pbar". One non-zero rest term: KIR (:10677-10683) pulls v_apical toward E_K, so an
  undriven cell can settle below rest - packet items C2 (unpulsed <1 Hz) and C6 (unpulsed
  v_apical below v_hold) MEASURE this rather than assume it.

### Frozen params (from the gap4 6/6 GO record, never tuned here)
bdsp_apical_soma_g (config.py:783), coincidence_plateau_self_regen / _v_hold / _v_hold_k
(bridge.py:10667-10673), apical_kir_g / _E_K / _vhalf / _kk (:10677-10681),
bdsp_v_apical_scale (:9668), W_MM = 0 (probe wiring).

### Next (rest of step 0, still Hermes)
P1: the clause-a oddity from cs-race-f1h-INPUTS section 3 (read-only or one seed-41 run).
Then the frozen grid `--soma-g {0.002,0.004,0.008}` x `W_IM {20,40,80}`, seeds 41-48, via the
new runner flags (`--trace intrinsic`, `--soma-g`, `--w-im`, `--apical-pulse`, default off) -
the runner build lands in a research/* worktree (Claude merges), STOP-ID identity check, then
`[cs-race-f1h-0] RESULT` and STOP.