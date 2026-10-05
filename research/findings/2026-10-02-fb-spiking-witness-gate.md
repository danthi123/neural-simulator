---
type: finding
status: live
claim_check: measured
date: 2026-10-02
lane: false-belief (s2p packet fb-spiking-witness)
mechanism: BRAIN_FALSE_BELIEF_SPIKING_WITNESS=1 at WITNESS_OP=4 (readout mode: presence population read over 20 ms
  at a frozen threshold gates the W3 register write) with BRAIN_FALSE_BELIEF_CHAT=1, pin 40c43d520, lesion flag
  BRAIN_FALSE_BELIEF_WITNESS_LESION=1
---

# fb-spiking-witness gate: O PASS + R PASS at op 4 (read-out mode; PARTIAL Stage-1 credit)

> **Claude verify-go (2026-10-04): the PASS stands, narrowed.** A third, independent skeptic re-ran `select`, `O`
> and `R` from the raw data and got byte-identical verdicts. It traced the data path: presence spikes over 20 steps,
> divided by 80 neurons, compared with 0.03, decide the W3 write, and `_Scenario.present` is dead on this path. It
> confirmed six distinct substrates (per-seed belief and presence rates differ). Three caveats follow, and they
> replace the headline and the credit sentence below.
> (1) **What it shows:** the witness state is a spiking presence/absence population, and a host threshold read of
> it decides the write. "Substrate-decided" overstates it; the prereg's own sentence is "substrate-decided/
> host-actuated (readout)".
> (2) **Credit: PARTIAL, never FULL** in read-out mode (owner RULING `t18-credit`), whatever P4 does. The finding's
> "FULL when P4 makes this the default" sentence is superseded. The next method is the fully synaptic write, the relay
> mode: ops 0-3 failed all 8 dev seeds and must be revisited (no-defer).
> (3) **The lesion** forces the write ("ignore the presence read"); it does not silence the population (presence
> reads 0.0 in both arms). The population's necessity rests on the O battery's `silence_true_belief` case (6/6). The
> read is bimodal (0.0 vs about 0.09, the move-time `spiking_write.presence_rate` in research/findings/raw/_load_bearing/_shards/s2p-R-fb-40c43d5/s*/fb-spiking-*/*.json), so the frozen 0.03 threshold carries little weight; any value in (0, 0.085) <!--derived-->
> gives the same answer. Other host residuals: the regex event classifier and the belief-location argmax.
> (4) **The write itself is host-actuated** in read-out mode (Claude, 2026-10-04): after the read, `_write_readout`
> resets the belief store to a quiescent snapshot (`_restore_slice`, a host clear) and injects a 5000 pA helper
> current into the target slot. A synaptic write ignites the target slot but does not yet persist through the hold
> (docs/drafts/fb-spiking-witness-relay2/DIAGNOSIS.md), so these two host actuations are part of what passes.

**PROVISIONAL: owner confirm + merge pending (NEEDS-CLAUDE). Not yet the faculty's default path — the Stage-1
credit becomes FULL only when P4 makes this the faculty's default and steps B/F pass there (prereg section 6).**

All numbers below are read directly from
`research/findings/raw/_s2p/fb-spiking-witness/40c43d5/verdicts/O.json` and
`research/findings/raw/_s2p/fb-spiking-witness/40c43d5/verdicts/R.json`; no other source.

## What ran

- Pin `40c43d520cbee67210b85ad767e88bdd176f88c1`, selection op **4** (readout; GRID 0-3 = relay failed all 8 dev
  seeds, GRID 4 is the first grid-order index passing, per prereg section 3).
- Step O (organ battery, gate seeds 42/43/44/100/101/102): verdict **PASS**, `n_defined` 6, `n_all_pass` 6, every
  seed state `DEFINED` / `all_pass` true; precondition "organ cells DEFINED on 6/6 seeds" ok.
- Step R (false-belief reply load-bearing): verdict **PASS**, `n_defined` 6, `n_reply_load_bearing` 6,
  `at_default` false, `undefined_reasons` [] (empty is the earned-precondition block; the no-skip guarantee is that
  PASS structurally requires 6/6 DEFINED in BOTH the capability and the control group).

## R per-seed (all six identical in kind)

Intact reply: "Sally will look in the basket for the marble." Lesion reply: "Sally will look in the box for the
marble." — `changed_fields` [answer], `reply_verdict` regressed, `s0` [regressed, true], turn `s2fb_fb`, state
DEFINED, on seeds 42, 43, 44, 100, 101, 102. The effect is a coherent content flip (answer-only), not a
degenerate reply.

Control row `fb-spiking-true-belief-ctl` (the scorer's permuted-control precondition): `n_defined` 6, `n_ok` 6,
per-seed verdict `control-ok`, intact == lesion, `changed_fields` [] — the lesion alone carries the change.

## What a PASS lets the finding say (prereg section 6, readout mode — fixed sentences)

> "Whether the tracked agent saw the move is the state of a spiking presence/absence flip-flop. A frozen threshold
> read of that population gates the W3 write (an actuation residual of the read-out class). The host presence dict
> no longer decides."

> "the host-decision row for the false-belief chat moves "witnessing" from host-owned to substrate-decided/
> host-actuated (readout). The Stage-1 credit becomes FULL only when P4 makes this the faculty's default path and
> steps B/F pass there."

## verify-go (two MiMo Pro refuters; lenses: reproducibility-power + instrument-trust; gate-cheat + control-integrity-and-selection)

Both refuters independently re-ran select, O and R from the raw artifacts and got byte-identical verdicts; both
returned SURVIVES on every lens (selection recomputed as first-in-grid-order; lesion one-variable verified from
12/12 shard argv records; control 6/6 intact==lesion; guards wired and binding in `iw2_score.py::_cell_std`).
Precision notes carried from the refuters: (a) the scorer tolerates but does not enforce the lesion flag in
sidecars — here every lesion sidecar's recorded argv shows it; (b) degenerate-intact and intact-content are two
separate guards, both held; (c) the six seeds' surface answers are verbatim identical — per-seed variation lives
in the spiking rates.

**Headline (superseded by the Claude caveat at the top):** O PASS + R PASS at op 4 — false-belief witnessing at this operating point is substrate-decided,
load-bearing on 6/6 seeds, with a clean permuted control.
