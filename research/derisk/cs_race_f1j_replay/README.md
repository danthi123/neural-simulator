# cs-race f1j open-loop replay: UNRECORDED PREDICTIONS, NOT RESULTS

**What this folder is.** The scripts and text outputs of the open-loop replay that informed the cs-race f1j packet
(`research/coordination/cs-race-f1j-PACKET.md`, section R, "Replay pre-check"). They are kept so the packet's numbers can
be re-derived and checked. They are **predictions, not results**:

- No file here is a run record. None has a `.prov.json` sidecar, none was produced by a pool line or under a pinned
  revision, and nothing here counts toward any bar, STOP rule, falsifier or GO. The packet discloses this ("Unrecorded
  (provenance)"); this folder does not change it.
- The replay is **open loop**. It re-integrates the BDSP apical voltage of the landed f1i step-0 records with the SOM ->
  apical conductance held at the recorded per-ms values (`g_ap`), so it cannot show a SOM recruitment that the f1i runs did
  not already have. `closed_loop.py` and `closed_loop_extra.py` bound that gap with margin tests; they do not close it.
  The closed-loop numbers are the f1j step-0 runs themselves (`research/runners/cs_race_f1j_score.py`).
- The engine is not the replay. The f1j E build matches the replay's single-slot reference to about 2e-5 mV
  (`tests/test_apical_nmda.py::test_replay_match_single_slot`), which checks the integration, not the closed-loop outcome.

**Provenance.** Written 2026-10-28 in a Claude session scratchpad on the workstation (under /tmp, RAM, wiped on reboot)
while drafting the f1j packet, then copied here byte for byte by the f1j build (branch `research/cs-race-f1j-build`). The
outputs were NOT regenerated for the copy. As a check the build re-ran `v0_validate.py`, `summary_sa125.py` and
`a_nmda.py lin bounded` from this folder against the landed f1i records; all three reproduced their saved outputs byte
for byte (apart from the wall-time figure in `out_v0_validate.txt`).

**Data read.** Only the landed f1i step-0 records (`research/findings/raw/_cs_race/f1i/step0/`, landed c8c9a2595):
seeds 41 and 45-51, calibration seeds. No gate seed (42-44, 100-102) is read or run.

## How to re-run (CPU, numpy, seconds per script)

```bash
cd research/derisk/cs_race_f1j_replay
ln -s ../../.. raw          # load.py reads raw/research/findings/raw/_cs_race/f1i/step0/; do NOT commit this link
/path/to/.venv/bin/python v0_validate.py > /tmp/v0.txt && diff /tmp/v0.txt out_v0_validate.txt
/path/to/.venv/bin/python a_nmda.py lin bounded      # prints the table; writes out_a_nmda_lin_bounded.json here
rm raw
```

`raw` may point at any checkout that holds the f1i step-0 records. Run the scripts from this folder (they import their
neighbours `core`, `load`, `ode` and `cf`, so not with `python -I`). `a1.py` writes `c1rows.json` into the current folder;
`a_nmda.py` writes `out_a_nmda_<kin>_<integ>.json`. Nothing else is written.

## Files

| script | saved output | what it computes |
|---|---|---|
| `load.py`, `ode.py`, `cf.py`, `core.py` | (modules) | record loader; the BDSP apical ODE reference; the f1i C6/C7 classifiers; the vectorized replay |
| `v0_validate.py` | `out_v0_validate.txt` | replay vs recorded v_apical (max 0.0035 mV), C6/C7 classification vs the records, baseline counts |
| `a_nmda.py <kin> <integ>` | `out_a_nmda_{sat,lin}_bounded.{txt,json}`, `out_a_nmda_sat_explicit.{txt,json}` | A-NMDA grid (tauN 50/75/100/150 x gN0 4/8/16), saturating vs linear gating, bounded vs explicit Euler |
| `a_nmda_map.py` | `out_a_nmda_map.txt` | tauN x gN0 map (early /45, C6 /72, corner (a) /24, grid a&b /216) |
| `a_nmda_detail.py` | `out_a_nmda_detail.txt` | detail at the window candidates: records still failing, saturating vs linear gating, the excess time course |
| `a_reservoir.py`, `a_reservoir_scan.py` | `out_a_reservoir.txt`, `out_a_reservoir_scan.txt` | the alternative companion: the engine's fused_coincidence_plateau form (80 ms slow conductance, Mg-blocked self-sustain) in place of the BDSP regen, and its bistability scan |
| `closed_loop.py`, `closed_loop_extra.py` | `out_closed_loop.txt`, `out_closed_loop_extra.txt` | what the records certify about SOM recruitment; margin tests on the SOM tail and the eraser volley |
| `summary_sa125.py` | `out_summary_sa125.txt` | per-cell (W_SA 0.125) early / late rescue for the A-NMDA grid at syn_tau_g_i 10 and 12 |
| `a1.py` ... `a13.py` | (none saved) | the memo's earlier analyses of the f1i records; their numbers are quoted in the packet and memo only |
