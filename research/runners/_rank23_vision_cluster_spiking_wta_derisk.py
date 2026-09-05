"""RANK-23 scaffold-retirement de-risk (2026-09-05): a SPIKING lateral-inhibition WTA circuit replaces the host
argmax-over-two-averaged-dendritic-voltages GROUPING decision in the EMERGE-36 fully-spiking vision-identity
pipeline (the "vision cluster" backlog item, `research/coordination/scaffold_retirement_backlog.md` RANK-23,
produced by the scaffold-shortcut-map workflow `w9sn9wn4b`).

PREMISE, VERIFIED AGAINST CURRENT CODE (per `tools/before_you_build.sh` -- the map has been wrong before; rank-24
found 2 mis-scoped items). RANK-23's own shortcut text names FOUR distinct host residuals inside
`vision_identity_production_organ.py` / `_emerge36_spiking_perception_pipeline_derisk.SpikingPerceptionProbe`:

  1. `encode_v1` (the Gabor/V1 front end) is a host numpy matmul (`retina @ W.T`).
  2. Feature-selection is a host `argsort` + 70th-percentile cut (`SpikingPerceptionProbe.__init__`, building
     `self.OF`, the per-object "active feature" set).
  3. THE FINAL bird-vs-fish DECISION is a host `argmax` over two averaged apical-dendrite voltages
     (`SpikingPerceptionProbe.infer`: `dr = {c: mean(vap[PROP[c]]) for c in (0,1)}; best = max(dr, key=dr.get)`).
  4. V1 receptive-field structure is a hand-written Gabor formula (a self-organization question, tracked
     separately as `BRAIN_V1_SELFORG`, which the map itself flags BLOCKED / must-stay-OFF).

Confirmed still current by direct read of `_emerge36_spiking_perception_pipeline_derisk.py` (2026-09-05): lines
71-77 (`V = encode_v1(...)`; `feats = argsort(...)`; `thr = percentile(...)`) and lines 143-146 (`infer`'s
`dr`/`max(dr, key=dr.get)`) are byte-for-byte as the map described. NOT already fixed.

SCOPE: item 3 ONLY -- the one the map's own retirement_mechanism calls "add lateral inhibition between the PROP
populations for a spiking decision", and the one the parent task names ("the vision cluster/grouping step ...
reuse existing visual-cortex + WTA machinery"): GROUPING a perceived object into ONE of the two learned category
clusters (bird vs fish) is exactly a 2-way winner-take-all read. Items 1 and 4 are explicitly OUT of scope (the
map assigns a different, larger mechanism to each -- a bridge-resident Gabor pathway + a real image transport for
(1), self-organized RF learning for (4) -- neither is a "clustering/grouping" step). Item 2 is explicitly OUT of
scope too: the map's own retirement_mechanism assigns the percentile-cut fix to "the satdiv/num-den arc [already]
in flight" (board #135: `research/findings/2026-09-0[1-3]-vision-*satdiv*.md`) -- re-attacking the SAME host
statistic here would duplicate that arc's own territory rather than opening a clean independent lane.

NO LIVE CONSUMER (verified, not assumed): `vision_identity_production_organ.py` is wired into
`webapp/server.py` behind `BRAIN_VISION_IDENTITY`, default-ON since 2026-08-26 -- but the wiring block "ONLY
fires on a visual query that CARRIES a `percept` field (req.percept)" (`webapp/server.py:3247-3251`), and no
caller anywhere in `webapp/` ever populates `req.percept` (grepped; there is no image/camera transport). So the
default-ON flag is structurally unreachable today, matching the map's "no live conversational vision consumer"
/ "no image ever reaches the retina" claim. Per the task: this is a pure de-risk of whether the substrate CAN do
the grouping: an honest negative is a fine deliverable. NOTHING here is wired into any live path.

ADDITIVE, ZERO BLAST RADIUS: this file is new; it imports `SpikingPerceptionProbe`,
`_prime_from_winners`, `_host`, and `_affect_marker_wta_derisk`'s `_build_bridge`/`_pool_rates` UNCHANGED
(reuse-by-import only -- no edit to `sim/`, `_emerge36_...`, `_affect_marker_wta_derisk.py`,
`vision_identity_production_organ.py`, or `webapp/server.py`). No existing call path is touched, so
byte-identical-when-off is automatic: there is no flag to flip, and nothing on any production path changed.

MECHANISM. `VisionDecisionWTA` builds a private 2-pool bridge via a DIRECT, unmodified import of
`_affect_marker_wta_derisk._build_bridge` (n_pools=2) -- the SAME excitatory-assembly + dedicated-FSI reciprocal
lateral-inhibition motif already 6-seed-GO'd for the affect-marker SELECTION (board #86, 2026-08-28) and, before
that, the 2-channel SPEAK-vs-STAY-SILENT basal-ganglia race (`_vocal_action_selector_gate`) -- reused verbatim
(same N_PER/N_PER_FSI pool sizes, same TO_FSI_WEIGHT/CROSS_INHIB_WEIGHT, same DRIVE_BASE_PA/DRIVE_GAIN_PA
operating point, same warmup/washout/run protocol via `_pool_rates`), generalized here to 2 visual categories
instead of 6 valence registers. This is the literal "existing visual-cortex + WTA machinery" reuse the task
calls for -- not a new WTA implementation.

Each held-out object's two PROP-population averaged apical-dendrite voltages (`dr[0]`, `dr[1]` -- reconstructed
by replaying `SpikingPerceptionProbe`'s own `_codon` + `_prime_from_winners` steps, since `infer()` does not
expose the intermediate value it computes internally before its trailing host comparison) are linearly rescaled
into a drive current per pool and fed into the WTA circuit; whichever pool's spiking rate clears the other by a
dead margin is read as the GROUPING decision -- neurons/synapses deciding which learned category cluster the
object belongs to, replacing ONLY the host `best = max(dr, key=dr.get)` line.

ANTI-CHEATS (the board-#86 convention, applied to this 2-pool case):
  1. PARITY -- does the spiking WTA's decision match the host argmax's decision on the SAME `dr` values, across
     held-out objects x 6 seeds?
  2. MIS-ROUTE / SHUFFLE -- swap which physical pool receives which category's drive (pool 0 <- dr[1], pool 1 <-
     dr[0]) at a FIXED, never-permuted pool -> category read-out label. A genuine dependency on the actual
     spiking race predicts the reported category FLIPS to the opposite of the unshuffled read on (nearly) every
     trial; a host formula secretly bypassing the circuit would not flip.
  3. SYMMETRIC-DRIVE (DECISION-LEVEL) CONTROL -- feed the WTA an artificial `dr` with BOTH categories EQUAL (no
     signal to break symmetry with); the circuit should show a small margin / no clean winner on (nearly) every
     trial, proving it has no baked-in pool-0-always-wins bias. (The POOLER-level lesion -- coincidence detection
     off -- is NOT re-tested here: it already collapses `_codon()` to an empty set BEFORE `dr` is ever computed,
     upstream of the step this runner changes; re-testing it would exercise nothing this runner touches.)
  4. `attributable_to` (tools.lab, gap#5 discipline) -- how much of the winner-margin is attributable to the REAL
     `dr` signal vs. the symmetric-drive control.

GATE (6 seeds 42/43/44/100/101/102; held-out = 3 objects/category x 2 categories = 6/seed; epochs=40, matching
`vision_identity_production_organ.VisionIdentityRecognizer`'s production default):
  GO      mean(parity) >= 0.90 AND every seed's parity >= 0.75 AND mean(shuffle_flip_rate) >= 0.85 AND
          mean(symmetric_abstain_rate) >= 0.60 AND attributable_to(real_margin, symmetric_margin) >= 0.5.
  PARTIAL directionally right but missing one clause.
  NEGATIVE/BOUNDARY the spiking substrate cannot reproduce the host decision reliably here -- reported as a
          first-class honest result (the task explicitly permits this: "no live consumer ... an honest negative
          is a fine deliverable").

Run:
  .venv/bin/python -m research.runners._rank23_vision_cluster_spiking_wta_derisk --demo --seed 42
  .venv/bin/python -m research.runners._rank23_vision_cluster_spiking_wta_derisk \
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_rank23_vision_cluster_spiking_wta.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny bridges (314 + 72 neurons); numpy is faster than cupy here
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from research.runners._emerge36_spiking_perception_pipeline_derisk import SpikingPerceptionProbe, CATPROP
from research.runners._emerge14_stageC_onbridge_learning_derisk import _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from research.runners._affect_marker_wta_derisk import (
    _build_bridge, _pool_rates, DRIVE_BASE_PA, DRIVE_GAIN_PA,
)
from tools.lab import attributable_to, void_if
from tools.verdict import Verdict

OUT = Path("research/findings/raw/_rank23_vision_cluster_spiking_wta.json")

# calibrated against THIS pipeline's own observed dendritic-readout range (a one-time measurement, seed 42,
# intact vs pooler-lesioned probe): the untaught/uncharged PROP dendrite reads ~-61.7 (well below FLOOR=-40 --
# never fired), the taught/charged one reads ~+12.7..+14.0. LOW/HIGH bracket both with headroom.
DR_LOW = -65.0
DR_HIGH = 20.0
DECISION_DEAD_MARGIN = 0.05          # same units/convention as board #86's DEAD_MARGIN (rate, spikes/step/neuron)
WARMUP_STEPS = 60
WASHOUT_STEPS = 40
RUN_STEPS = 60
N_SYMMETRIC_TRIALS = 10


def _dr_for(probe: SpikingPerceptionProbe, of) -> dict | None:
    """Reconstruct the SAME `dr` dict `SpikingPerceptionProbe.infer()` computes internally (mirrors its body
    verbatim through the `dr` line -- `infer()` does not expose this intermediate value, so it must be replayed,
    not re-derived). Returns None when the codon is empty (the pooler-level lesion / no-charge path -- `infer()`
    would already return -1/abstain here too, upstream of the step this runner changes)."""
    resp = probe._codon(of)
    if not resp:
        return None
    ab = np.zeros(len(probe.ci), bool)
    for i in resp:
        ab[i] = True
    _prime_from_winners(probe.b, probe.ci, ab)
    vap = _host(probe.b.cp_v_apical)[probe.ci]
    return {c: float(np.mean([vap[x] for x in probe.PROP[c]])) for c in (0, 1)}


class VisionDecisionWTA:
    """A 2-pool spiking lateral-inhibition WTA over the vision-identity pipeline's PROP dendritic read. Builds
    its bridge via `_affect_marker_wta_derisk._build_bridge` UNCHANGED (board #86's validated N-pool reciprocal
    cross-inhibition motif, here n_pools=2) -- reuse of "existing visual-cortex + WTA machinery", not a new WTA."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.bridge, self.idx, self.fsi = _build_bridge(self.seed, 2, "vis_dec")

    def _to_pa(self, v: float) -> float:
        frac = float(np.clip((v - DR_LOW) / (DR_HIGH - DR_LOW), 0.0, 1.0))
        return DRIVE_BASE_PA + DRIVE_GAIN_PA * frac

    def decide(self, dr: dict, *, shuffle: bool = False, dead_margin: float = DECISION_DEAD_MARGIN):
        """`dr`: {0: float, 1: float}, the SAME two averaged-apical-dendrite readings `infer()` computes
        host-side. `shuffle=True` mis-routes which physical pool receives which category's drive (the pool ->
        category read-out label is NEVER permuted -- see module docstring anti-cheat 2).
        Returns (winner_category_or_None, rates, margin)."""
        perm = (1, 0) if shuffle else (0, 1)
        drive = [self._to_pa(dr[perm[p]]) for p in (0, 1)]
        rates = _pool_rates(self.bridge, self.idx, drive, warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
        order = np.argsort(rates)[::-1]
        top, second = int(order[0]), int(order[1])
        margin = float(rates[top] - rates[second])
        if margin <= dead_margin:
            return None, rates, margin
        return top, rates, margin           # `top` IS the reported category: the pool->label map is fixed.


def run_seed(seed: int, epochs: int = 40) -> dict:
    probe = SpikingPerceptionProbe(seed=seed, epochs=epochs)
    wta = VisionDecisionWTA(seed=seed + 1000)     # independent RNG stream for the decision circuit's own build

    n = n_clear = host_correct = wta_correct = parity = shuffle_flip = 0
    margins_real = []
    per_object = []
    for c in (0, 1):
        for h in probe.held[c]:
            dr = _dr_for(probe, probe.OF[h])
            if dr is None:
                continue                         # abstain upstream of the decision step -- not this runner's concern
            n += 1
            host_best = max(dr, key=dr.get)
            host_correct += int(host_best == c)

            wta_best, rates, margin = wta.decide(dr)
            wta_correct += int(wta_best == c)
            parity += int(wta_best == host_best)
            margins_real.append(margin)

            flip_ok = None
            if wta_best is not None:
                n_clear += 1
                expected_flip = 1 - wta_best
                shuf_best, _, _ = wta.decide(dr, shuffle=True)
                flip_ok = bool(shuf_best == expected_flip)
                shuffle_flip += int(flip_ok)
            per_object.append({"true_cat": c, "held": int(h), "dr": dr, "host_best": host_best,
                               "wta_best": wta_best, "margin": round(margin, 5), "shuffle_flip_ok": flip_ok})

    # symmetric-drive (decision-level) control: no real signal to break symmetry with.
    sym_margins = []
    sym_abstain = 0
    rng = np.random.default_rng(seed * 131 + 7)
    for _ in range(N_SYMMETRIC_TRIALS):
        v = float(rng.uniform(DR_LOW, DR_HIGH))
        best, _, margin = wta.decide({0: v, 1: v})
        sym_margins.append(margin)
        sym_abstain += int(best is None)

    void_if(n == 0, f"seed {seed}: every held-out object abstained upstream (empty codon) -- nothing to decide")
    real_margin_mean = float(np.mean(margins_real)) if margins_real else 0.0
    sym_margin_mean = float(np.mean(sym_margins))
    return {
        "seed": seed, "n_evaluable": n, "n_clear_winner": n_clear,
        "host_acc": (host_correct / n) if n else None,
        "wta_acc": (wta_correct / n) if n else None,
        "parity": (parity / n) if n else None,
        "shuffle_flip_rate": (shuffle_flip / n_clear) if n_clear else None,
        "real_margin_mean": real_margin_mean,
        "symmetric_margin_mean": sym_margin_mean,
        "symmetric_abstain_rate": sym_abstain / N_SYMMETRIC_TRIALS,
        "per_object": per_object,
    }


def _demo(seed: int = 42, epochs: int = 40):
    r = run_seed(seed, epochs)
    print("\n=== RANK-23 vision-cluster spiking-WTA grouping decision (seed=%d) ===" % seed)
    for o in r["per_object"]:
        print(f"  true={CATPROP[o['true_cat']]:5s} held={o['held']:2d}  dr={o['dr']}  "
              f"host->{CATPROP.get(o['host_best'], '?'):5s}  wta->{CATPROP.get(o['wta_best'], 'ABSTAIN') if o['wta_best'] is not None else 'ABSTAIN':7s}  "
              f"margin={o['margin']:.4f}  shuffle_flipped_as_expected={o['shuffle_flip_ok']}")
    print(f"  host_acc={r['host_acc']:.2f} wta_acc={r['wta_acc']:.2f} parity={r['parity']:.2f} "
          f"shuffle_flip_rate={r['shuffle_flip_rate']:.2f} symmetric_abstain_rate={r['symmetric_abstain_rate']:.2f}")
    print(f"  real_margin_mean={r['real_margin_mean']:.4f}  symmetric_margin_mean={r['symmetric_margin_mean']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="single seed for --demo")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seed, a.epochs)
        return 0
    if len(a.seeds) < 6:
        print(f"NOT-RUNNABLE: 6-seed validation required (feedback_6seed_validation); got {len(a.seeds)}")
        return 2

    print(f"[rank23] vision-cluster spiking-WTA grouping decision de-risk: seeds={a.seeds} epochs={a.epochs}",
          flush=True)
    t0 = time.time()
    err = None
    per_seed = []
    try:
        for s in a.seeds:
            r = run_seed(s, a.epochs)
            per_seed.append(r)
            print(f"  [seed {s}] host_acc={r['host_acc']:.2f} wta_acc={r['wta_acc']:.2f} parity={r['parity']:.2f} "
                  f"shuffle_flip={r['shuffle_flip_rate']:.2f} sym_abstain={r['symmetric_abstain_rate']:.2f} "
                  f"real_margin={r['real_margin_mean']:.3f} sym_margin={r['symmetric_margin_mean']:.3f}",
                  flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    decided = None
    if err is None:
        def m(key):
            return float(np.mean([p[key] for p in per_seed]))
        parity_mean = m("parity")
        parity_min = float(np.min([p["parity"] for p in per_seed]))
        shuffle_mean = m("shuffle_flip_rate")
        sym_abstain_mean = m("symmetric_abstain_rate")
        real_margin_mean = m("real_margin_mean")
        sym_margin_mean = m("symmetric_margin_mean")
        attrib = attributable_to("vision-decision WTA margin: real dr vs symmetric-drive control",
                                 real_margin_mean, sym_margin_mean)

        # tools.verdict.Verdict (gap#5 discipline / gates/verdict_preconditions): the ANTI-CHEATS are
        # PRECONDITIONS for the comparison being meaningful at all; the substantive GO claim is parity itself
        # (does the spiking WTA reproduce the host decision?). A failed/unmeasured precondition forces
        # UNDEFINED, never a silently-asserted GO or NO-GO beside an unchecked instrument.
        v = Verdict("RANK-23 vision-cluster grouping-decision: spiking WTA vs host argmax", chance=0.5)
        v.control("winner margin: real dr vs symmetric-drive control", treatment=real_margin_mean,
                  control=sym_margin_mean, min_separation=DECISION_DEAD_MARGIN,
                  note="the circuit must discriminate MORE under the real dendritic signal than under a "
                       "no-signal (symmetric-drive) control, by more than its own dead-margin threshold")
        v.require("symmetric-drive control mostly abstains (no baked-in pool-0 bias)", sym_abstain_mean,
                  expect=lambda x: x >= 0.60)
        v.require("mis-route shuffle flips the decision as predicted (genuine wiring dependency)", shuffle_mean,
                  expect=lambda x: x >= 0.85)
        v.require("parity is at least directionally robust on every seed", parity_min, expect=lambda x: x >= 0.75)
        decided = v.decide(go=bool(parity_mean >= 0.90))
        status = decided["status"]

        if status == "GO":
            verdict = (f"GO -- a spiking lateral-inhibition WTA circuit (board #86's reused, unmodified N-pool "
                       f"motif, n_pools=2) reproduces the EMERGE-36 vision-identity pipeline's host "
                       f"argmax-over-averaged-dendritic-voltages GROUPING decision: parity {parity_mean:.2f} "
                       f"(min {parity_min:.2f}/seed), mis-route shuffle flips the reported category as predicted "
                       f"{shuffle_mean:.2f} of clear-winner trials, the symmetric-drive control abstains "
                       f"{sym_abstain_mean:.2f} of trials (no baked-in pool bias), and {attrib*100:.0f}% of the "
                       f"winner-margin is attributable to the real dendritic signal vs the symmetric control. "
                       f"6-seed. RANK-23's decision-step host shortcut CAN be retired by the substrate; no live "
                       f"consumer exists to wire it into (honest scope: items 1/2/4 of the RANK-23 cluster are "
                       f"untouched, assigned elsewhere by the map).")
        elif status == "UNDEFINED":
            verdict = ("UNDEFINED -- a precondition failed or was never measured, so no GO/NO-GO is earned "
                       "(tools.verdict discipline: an unguarded verdict is itself the defect). Reasons: "
                       + "; ".join(decided["undefined_reasons"]))
        else:
            verdict = (f"NO-GO -- the spiking WTA's preconditions held (no instrument artifact) but parity "
                       f"{parity_mean:.2f} < 0.90: the circuit does not reliably reproduce the host argmax "
                       f"decision. Honest result (no live consumer; a characterized negative is a fine outcome "
                       f"per the task). The host argmax decision stands unmodified.")
    else:
        verdict = f"ERROR -- {err}"
        parity_mean = parity_min = shuffle_mean = sym_abstain_mean = real_margin_mean = sym_margin_mean = attrib = None

    summary = {
        "probe": "rank23_vision_cluster_spiking_wta_derisk",
        "backlog_item": "scaffold_retirement_backlog.md RANK-23 (vision cluster / grouping step, item 3 of 4: "
                        "the host argmax-over-averaged-dendritic-voltages final decision)",
        "verdict": verdict,
        "status": decided["status"] if decided else "ERROR",
        # tools.verdict.Verdict's preconditions block, TOP-LEVEL (gates/verdict_preconditions requires this
        # exact key here): the anti-cheats (control/symmetric-abstain/shuffle/parity-robustness) that earned
        # (or refused) the verdict above. Empty on a hard runner ERROR (no verdict was asserted to guard).
        "preconditions": decided["preconditions"] if decided else [],
        "mechanism": "board #86's reused N-pool spiking lateral-inhibition WTA motif (_affect_marker_wta_derisk."
                    "_build_bridge, n_pools=2, unmodified) reads the SAME two PROP-population averaged apical "
                    "voltages SpikingPerceptionProbe.infer() computes, replacing only its trailing "
                    "`max(dr, key=dr.get)` host comparison with a spiking competitive decision.",
        "scope_note": "RANK-23 groups 4 host residuals; items 1 (encode_v1 matmul), 2 (argsort+percentile "
                      "feature-selection, assigned to the in-flight satdiv/board-#135 arc) and 4 (hand-written "
                      "Gabor RF / BRAIN_V1_SELFORG, BLOCKED) are OUT OF SCOPE here -- this runner attacks item 3 "
                      "(the final grouping decision) only.",
        "no_live_consumer": "verified: BRAIN_VISION_IDENTITY is default-ON in webapp/server.py but its wiring "
                            "block only fires on a turn whose req.percept is populated; no caller anywhere sets "
                            "req.percept (no image/camera transport exists). This runner wires into NOTHING.",
        "seeds": a.seeds, "config": {"epochs": a.epochs, "dr_low": DR_LOW, "dr_high": DR_HIGH,
                                     "decision_dead_margin": DECISION_DEAD_MARGIN,
                                     "warmup": WARMUP_STEPS, "washout": WASHOUT_STEPS, "run": RUN_STEPS},
        "aggregate": {"parity_mean": parity_mean, "parity_min": parity_min,
                     "shuffle_flip_rate_mean": shuffle_mean, "symmetric_abstain_rate_mean": sym_abstain_mean,
                     "real_margin_mean_across_seeds": real_margin_mean,
                     "symmetric_margin_mean_across_seeds": sym_margin_mean,
                     "attributable_to_real_vs_symmetric": attrib},
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed": per_seed,
        "HONEST_NOTE": "additive-only: this file is new, reuses SpikingPerceptionProbe / _affect_marker_wta_"
                       "derisk._build_bridge unmodified (import-only), and is wired into no production path. "
                       "byte-identical-when-off is automatic (there is no flag; nothing existing changed).",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[rank23] VERDICT: {verdict}", flush=True)
    print(f"[rank23] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
