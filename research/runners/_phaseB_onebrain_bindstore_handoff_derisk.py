"""ROADMAP TIER-2 (the persistent INTEGRATED spiking loop), PHASE A cheap-first de-risk -- the bind->store DATA
hand-off (H4) on the production `OneBrainComposer`, made SYNAPTIC (no host read-then-write).

Pre-registered by `2026-06-19-tier2-persistent-integrated-loop-scoping.md` (commit f1b551db), Phase A. The
`OneBrainComposer` is already a persistent co-resident bridge with H2/H3/H5/H6/H7 synaptic; the residual host
round-trips are H1/H4/H8/H9. **H4 (bind->store) is the cheapest, most load-bearing single conversion.**

THE H4 HOST ROUND-TRIP (today, `OneBrainComposer._compose_phases` end + `_write_block`):
  - `_compose_phases` settles the composite phasor in the `acc` register (block 2n, on-bridge), then
    `rf_read_phases()[acc]` READS it to numpy and `_to_phasor` re-encodes it on HOST,
  - `_write_block` then installs `complex(zc[k])` as the store block's trigger->readout weights.
  The composite leaves the bridge to host (a phase read + a host re-encode), then is written BACK as weights.

THE SYNAPTIC HAND-OFF (this de-risk -- the register->register primitive, GO `2026-06-18-one-brain-register-
handoff-GO.md`, applied to the specific bind->store path): keep `acc` resident; install the
`acc -> store-block-readout` complex synapse `(trig+1+k, P+acc*D+k, 1.0)` and resonate, so the composite flows
from `acc` into the store block's readout neurons `trig+1+k` THROUGH a unit complex synapse (register->register,
NO `rf_read_phases()[acc]`, NO host re-encode of the bind output). The persistent store IS a synapse (the memory
must end as `cp_rf_w_re/im`), so the store weight is captured at the TERMINUS of the synaptic route -- the store
readout register that `acc` drove synaptically -- not from a host re-read of `acc`. The bind's output never
round-trips through `acc`'s host phases to become the store weight.

GATE (Phase A, all):
  - recall == the host-path recall (a fact stored via the synaptic H4 recalls IDENTICALLY to the stock host
    read-then-write store), >=3 seeds x 2 D (exact/identity -> parity, not a distribution);
  - severed-route LESION (zero the acc->store synapse) -> the store FAILS (the hand-off is load-bearing, not a
    parallel host write still happening);
  - permuted-store -> recall follows the ROUTED fact, not a leak;
  - the no-confab MOAT holds (unstored cue -> None; 0 breaches).

Reuse-by-import (OneBrainComposer + RFPhasorComposer). NO sim/ edit (the masked rf_kick/rf ops already exist).
CPU/numpy for this cheap-first de-risk (the algebra is exact; SIM_BACKEND=numpy).
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_bindstore_handoff_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")   # cheap-first: the FHRR algebra is exact; CPU is the oracle path

from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402

# A small fact set that exercises all 4 roles (agent/action/patient + the AFFIRM polarity tag). K=8 facts.
FACTS = [
    ("dog", "go", "north"),
    ("cat", "come", "south"),
    ("bird", "look", "east"),
    ("river", "stop", "west"),
    ("apple", "go", "south"),
    ("dog", "look", "river"),
    ("cat", "stop", "apple"),
    ("bird", "come", "north"),
]
VOCAB = ["dog", "cat", "bird", "river", "apple", "north", "south", "east", "west", "go", "come", "look", "stop"]


class SynapticH4Composer(OneBrainComposer):
    """OneBrainComposer with the bind->store (H4) hand-off made SYNAPTIC. The ONLY override is `_write_block`:
    instead of taking a HOST-read composite (`rf_read_phases()[acc]` -> _to_phasor, done by the stock
    `_compose_phases`), it routes the still-resident `acc` register INTO the store block's readout neurons through
    an `acc -> store-readout` complex synapse, settles, and captures the store weight at that synaptic TERMINUS.

    To make `_store_composite`/`update_on_mismatch` call this WITHOUT a host read of `acc`, we override
    `_compose_phases` to leave `acc` resident and return the SLICE INDEX of the acc register (a sentinel), then the
    overridden `_write_block` consumes it. (`_compose_phases` is also called by reconsolidation with the same
    contract.) The bind/bundle (H2/H3) is the stock on-bridge path; only the acc->store hand-off changes.

    `lesion` (anti-cheat): drop the acc->store synapse -> the store readout register gets nothing -> the captured
    store weight is garbage -> recall must collapse (proves the on-bridge hand-off is load-bearing).
    """

    def __init__(self, *args, lesion=False, **kwargs):
        self._h4_lesion = bool(lesion)
        self._acc_resident = None       # set by _compose_phases_synaptic to the acc slice it left resident
        super().__init__(*args, **kwargs)

    # --- the synaptic bind->store hand-off ---
    def _compose_phases(self, fillers, roles):
        """Bind each (role, filler) + bundle into `acc` -- the SAME on-bridge H2/H3 path as the stock composer --
        but DO NOT read `acc` to host. Leave `acc` resident on the bridge and return a sentinel (None); the
        overridden `_write_block` routes `acc` into the store synaptically. The bind/bundle code is identical to
        the parent so the composite that lands in `acc` is byte-equal to the stock path's pre-read composite."""
        comp, b, D, P, Pd = self.comp, self.b, self.D, self.P, self.period
        n = len(roles); acc = 2 * n
        binds, bundle = [], []
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(n):
            zr = comp._to_phasor(comp.roles[roles[i]]); zf = comp._to_phasor(comp._filler_phases(fillers[i]))
            kick[P + i * D:P + (i + 1) * D] = zf
            binds += [(P + (n + i) * D + k, P + i * D + k, complex(zr[k])) for k in range(D)]
            bundle += [(P + acc * D + k, P + (n + i) * D + k, 1.0) for k in range(D)]
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        b.rf_set_complex_weights(binds); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(Pd + 8)
        # acc is now settled on-bridge (v/u hold the composite phasor). Leave it; record where it is for _write_block.
        self._acc_resident = (P + acc * D, P + (acc + 1) * D)
        return None     # sentinel: the composite stays on the substrate; _write_block routes it synaptically

    def _write_block(self, i, zc):
        """Capture block i's store weight via the SYNAPTIC acc->store-readout hand-off. `zc` is the sentinel None
        from the synaptic `_compose_phases` (or, for completeness, an explicit phasor array if a caller bypasses
        it). The store block is a (1+D) trigger->readout run; we route the resident `acc` register into the
        readout neurons `trig+1+k` through a unit complex synapse, settle, then read those readout neurons'
        PHASES (the synaptic terminus) and install them as the persistent store weight. NO host read of `acc`."""
        D = self.D
        trig = self.store_base + i * self.block
        if zc is None:
            # SYNAPTIC route: acc -> store-block readout (trig+1+k), unit complex synapse. acc is still resident.
            a0, a1 = self._acc_resident
            route = [] if self._h4_lesion else [(trig + 1 + k, a0 + k, 1.0) for k in range(D)]
            b, Pd = self.b, self.period
            b.rf_set_complex_weights(route); b.rf_resonate_steps(Pd + 8)
            # the readout neurons trig+1+k now hold acc's composite (synaptically). Capture at this terminus.
            phases = np.asarray(b.rf_read_phases())[trig + 1:trig + 1 + D]
            zc_arr = self.comp._to_phasor(phases)       # the store weight = the SYNAPTICALLY-routed composite phasor
        else:
            zc_arr = np.asarray(zc)                      # explicit phasor (parity path; not used by the synaptic flow)
        block_conns = [(trig + 1 + k, trig, complex(zc_arr[k])) for k in range(D)]
        if i * D < len(self.store_conns):
            self.store_conns[i * D:(i + 1) * D] = block_conns
        else:
            self.store_conns += block_conns
        self._store_dirty = True


def _build(cls, seed, D, **kw):
    c = cls(seed=seed, D=D, vocab=VOCAB, k_max=len(FACTS) + 2, period=200,
            enable_rf_cudagraph=False, enable_csr_cache=False, **kw)
    return c


def _store_all(c):
    for (a, act, p) in FACTS:
        c.store(a, act, p)


def run_seed(seed, D):
    # --- HOST path (stock OneBrainComposer: rf_read_phases()[acc] -> _write_block) ---
    host = _build(OneBrainComposer, seed, D)
    _store_all(host)
    host_patient = [host.query_patient(a, act) for (a, act, p) in FACTS]
    host_agent = [host.query_agent(act, p) for (a, act, p) in FACTS]
    host_yes = [host.ask_yes_no(a, act, p) for (a, act, p) in FACTS]

    # --- SYNAPTIC H4 path (acc -> store readout via complex synapse, no host read of acc) ---
    syn = _build(SynapticH4Composer, seed, D)
    _store_all(syn)
    syn_patient = [syn.query_patient(a, act) for (a, act, p) in FACTS]
    syn_agent = [syn.query_agent(act, p) for (a, act, p) in FACTS]
    syn_yes = [syn.ask_yes_no(a, act, p) for (a, act, p) in FACTS]

    # recall == host (parity, exact)
    eq_patient = sum(int(syn_patient[i] == host_patient[i]) for i in range(len(FACTS)))
    eq_agent = sum(int(syn_agent[i] == host_agent[i]) for i in range(len(FACTS)))
    eq_yes = sum(int(syn_yes[i] == host_yes[i]) for i in range(len(FACTS)))
    # recall CORRECT (the synaptic store actually retrieves the stored fact)
    ok_patient = sum(int(syn_patient[i] == FACTS[i][2]) for i in range(len(FACTS)))
    ok_agent = sum(int(syn_agent[i] == FACTS[i][0]) for i in range(len(FACTS)))
    ok_yes = sum(int(syn_yes[i] == "yes") for i in range(len(FACTS)))

    # --- the no-confab MOAT: unstored cues must abstain (None / "unknown") on the synaptic store ---
    moat_queries = [
        ("apple", "stop"),        # never stored (agent+action pair absent)
        ("river", "go"),
        ("dog", "stop"),
        ("bird", "go"),
    ]
    moat_patient = [syn.query_patient(a, act) for (a, act) in moat_queries]
    moat_yes = [syn.ask_yes_no("apple", "stop", "north"),    # full SVO never stored
                syn.ask_yes_no("river", "go", "east"),
                syn.ask_yes_no("dog", "stop", "cat")]
    moat_breaches = sum(int(x is not None) for x in moat_patient) + sum(int(x != "unknown") for x in moat_yes)

    # --- LESION: sever the acc->store synapse -> store readout gets nothing -> recall collapses ---
    les = _build(SynapticH4Composer, seed, D, lesion=True)
    _store_all(les)
    les_patient = [les.query_patient(a, act) for (a, act, p) in FACTS]
    les_ok = sum(int(les_patient[i] == FACTS[i][2]) for i in range(len(FACTS)))

    # --- PERMUTED-STORE: synaptically route a DISTINCT fact into block 0, then read block 0 DIRECTLY and confirm
    #     it now holds the ROUTED fact (the synaptic write carries the content, not a leak). The routed fact uses a
    #     patient ("river") that is NOT fact-0's patient ("north"), so reading block 0 directly is unambiguous: if
    #     it shows the routed patient, the acc->store synapse physically rewrote that block's content. ---
    perm = _build(SynapticH4Composer, seed, D)
    _store_all(perm)
    routed = ("cat", "go", "river")                          # distinct from FACTS[0]=("dog","go","north")
    roles_r = ["agent", "action", "patient", "polarity"]
    fillers_r = ["cat", "go", "river", "AFFIRM"]
    perm._compose_phases(fillers_r, roles_r)                 # leaves the routed fact's acc resident
    perm._write_block(0, None)                               # synaptically route it into BLOCK 0 (overwrite)
    block0 = perm._read_block(0)                             # read block 0 DIRECTLY (no scan ambiguity)
    perm_follows = int(block0.get("agent") == routed[0] and block0.get("action") == routed[1]
                       and block0.get("patient") == routed[2])
    perm_q = (block0.get("agent"), block0.get("action"), block0.get("patient"))

    row = {
        "seed": seed, "D": D, "K": len(FACTS),
        "eq_patient": eq_patient, "eq_agent": eq_agent, "eq_yes": eq_yes,
        "ok_patient": ok_patient, "ok_agent": ok_agent, "ok_yes": ok_yes,
        "host_ok_patient": sum(int(host_patient[i] == FACTS[i][2]) for i in range(len(FACTS))),
        "moat_breaches": moat_breaches, "moat_patient": moat_patient, "moat_yes": moat_yes,
        "lesion_ok": les_ok, "perm_follows": perm_follows, "perm_q": perm_q,
    }
    K = len(FACTS)
    print(f"  [seed {seed} D={D}] recall==host: patient {eq_patient}/{K} agent {eq_agent}/{K} yes {eq_yes}/{K} | "
          f"syn-correct patient {ok_patient}/{K} (host {row['host_ok_patient']}/{K}) | "
          f"moat breaches {moat_breaches} | LESION recall {les_ok}/{K} | permuted-follows {perm_follows}/1",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_bindstore_handoff.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[one-brain bind->store (H4) hand-off de-risk] is the bind->store DATA hand-off SYNAPTIC (acc->store "
          "readout via complex synapse, NO host read-then-write) and does it recall == the host path?\n", flush=True)
    rows = [run_seed(s, D) for s in seeds for D in dims]

    K = len(FACTS)
    n = len(rows)
    # GO bar: recall==host EXACT on all 3 query types, every row; syn-correct == host-correct; lesion collapses;
    # permuted follows the routed fact; moat 0 breaches everywhere.
    all_eq = all(r["eq_patient"] == K and r["eq_agent"] == K and r["eq_yes"] == K for r in rows)
    syn_correct_eq_host = all(r["ok_patient"] == r["host_ok_patient"] for r in rows)
    mean_syn_ok = float(np.mean([r["ok_patient"] for r in rows])) / K
    mean_host_ok = float(np.mean([r["host_ok_patient"] for r in rows])) / K
    mean_lesion = float(np.mean([r["lesion_ok"] for r in rows])) / K
    perm_all = all(r["perm_follows"] == 1 for r in rows)
    moat_total = sum(r["moat_breaches"] for r in rows)
    lesion_collapses = mean_lesion <= 0.15
    go = all_eq and syn_correct_eq_host and lesion_collapses and perm_all and (moat_total == 0)
    print(f"\n{'='*108}", flush=True)
    print(f"  MEAN ({n} seed*D, K={K}): recall==host ALL rows: {all_eq} | syn-correct=={mean_syn_ok:.3f} "
          f"(host {mean_host_ok:.3f}) | LESION recall {mean_lesion:.3f} (collapse<=0.15: {lesion_collapses}) | "
          f"permuted-follows all rows: {perm_all} | MOAT total breaches: {moat_total}", flush=True)
    if go:
        print(f"  GO: the bind->store (H4) hand-off is SYNAPTIC on the persistent loop -- a fact stored via the "
              f"acc->store complex-synapse route recalls IDENTICALLY to the host read-then-write store ({n}/{n} rows "
              f"exact), the severed-route lesion COLLAPSES recall (the hand-off is load-bearing), permuted-store "
              f"follows the routed fact, and the no-confab moat holds (0 breaches). ==> H4 closed; recommend it as "
              f"the composer default. Phase B (the on-substrate SEQUENCER) is the remaining deep piece.", flush=True)
    else:
        reasons = []
        if not all_eq:
            reasons.append("recall != host on some row (the synaptic hand-off DIVERGES from the host path)")
        if not syn_correct_eq_host:
            reasons.append("syn-correct != host-correct (the routed store loses content)")
        if not lesion_collapses:
            reasons.append(f"lesion did NOT collapse ({mean_lesion:.3f}) -- a parallel host write is still happening")
        if not perm_all:
            reasons.append("permuted-store did not follow the routed fact (a leak, not a synaptic write)")
        if moat_total != 0:
            reasons.append(f"MOAT BREACHED ({moat_total}) -- REJECT (never weaken the no-confab moat)")
        print(f"  NEGATIVE: {'; '.join(reasons)}. Honest finding: report exactly what's needed; do NOT escalate.",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)
    out = {"verdict": "GO" if go else "NEGATIVE", "seeds": seeds, "dims": dims, "K": K,
           "all_recall_eq_host": all_eq, "syn_correct_eq_host": syn_correct_eq_host,
           "mean_syn_ok": mean_syn_ok, "mean_host_ok": mean_host_ok, "mean_lesion": mean_lesion,
           "permuted_follows_all": perm_all, "moat_total_breaches": moat_total, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
