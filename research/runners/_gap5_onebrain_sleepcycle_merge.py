"""gap#5 one-brain MERGE — the PRODUCTION-agent validation. The OneBrainComposer bridge is Izhikevich/dt1.0 (the RF
composer runs as masked complex-synapse ops on a slice, NOT a global RF model), so the validated wake/sleep phase-switch
applies directly. Does the PRODUCTION conversational agent's store + recall + no-confab MOAT survive a full
WAKE->SLEEP->WAKE phase-switch cycle (Izh/dt1.0 -> AdEx/dt0.1 sleep -> Izh/dt1.0)? The conversational memory lives in the
RF complex synapses (cp_rf_w_re/im) + the Izhikevich parser weights (cp_connections) — the switch touches NEITHER (only
v/adex_w/u/cfg). GO iff every recall + the moat-abstention is IDENTICAL before and after the sleep cycle."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from sim.backend import to_host, get_backend, is_gpu_backend
from research.runners._gap5_wake_sleep_phase_switch import switch_to_adex_sleep, _recompute_cached_decays
from research.runners._gap5_wake_sleep_roundtrip import switch_to_izhikevich_wake, reset_transient_synaptic_state

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
QUERIES = [("dog", "go"), ("cat", "come"), ("bird", "look")]
MOAT = [("apple", "swim"), ("river", "stop")]   # never stored -> must abstain (None)


def sleep_cycle(b, n_sleep=60):
    """WAKE(Izh)->SLEEP(AdEx/dt0.1, quiescent replay window)->WAKE(Izh). Touches NO synaptic weights (RF or Izhikevich)."""
    b.core_config.enable_stdp = False                       # freeze plasticity for the sleep phase (roundtrip fix)
    switch_to_adex_sleep(b, dt=0.1); reset_transient_synaptic_state(b)
    for _ in range(n_sleep):                                # the SWR/sleep window (composer quiescent; a real merge runs the CA3 replay here)
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        b.cp_external_input_current[:] = 0.0; b._run_one_simulation_step()
    switch_to_izhikevich_wake(b, dt=1.0); reset_transient_synaptic_state(b)


def run(seed):
    from research.runners.one_brain_composer import OneBrainComposer
    c = OneBrainComposer(seed=seed, D=64, vocab=VOCAB)
    for a, v, p in FACTS:
        c.store(a, v, p)
    ans1 = [c.query_patient(a, v) for a, v in QUERIES]
    moat1 = [c.query_patient(a, v) for a, v in MOAT]
    sleep_cycle(c.b)                                        # <-- the wake/sleep/wake phase-switch cycle
    ans2 = [c.query_patient(a, v) for a, v in QUERIES]
    moat2 = [c.query_patient(a, v) for a, v in MOAT]
    recall_ok = (ans1 == ans2) and all(a == p for (a, (_, _, p)) in zip(ans2, FACTS))
    moat_ok = all(m is None for m in moat1) and all(m is None for m in moat2)
    print(f"  [seed {seed}] pre-sleep recall={ans1} moat={moat1} | post-sleep recall={ans2} moat={moat2} "
          f"-> recall_preserved={recall_ok} moat_intact={moat_ok}", flush=True)
    return recall_ok and moat_ok


if __name__ == "__main__":   # guarded so `sleep_cycle` is importable without running the 6-seed suite (the production
                             # continuous-engine sleep-replay consumer reuses `sleep_cycle` verbatim by import, #64)
    if not is_gpu_backend():
        print("SKIP: needs GPU (OneBrainComposer on-bridge parser)", flush=True); sys.exit(0)
    print("gap#5 ONE-BRAIN PRODUCTION sleep-cycle — does the OneBrainComposer's store/recall/moat survive a WAKE->SLEEP->WAKE "
          "phase-switch cycle? GO iff recall + moat IDENTICAL before/after, all seeds.", flush=True)
    seeds = [42, 43, 44, 100, 101, 102]
    oks = []
    for s in seeds:
        try:
            oks.append(run(s))
        except Exception as e:
            print(f"  [seed {s}] ERROR: {type(e).__name__}: {e}", flush=True); oks.append(False)
    print(f"\n=== PRODUCTION SLEEP-CYCLE: survived {sum(oks)}/{len(seeds)} -> {'GO' if all(oks) and len(oks)==len(seeds) else 'NO-GO'} ===", flush=True)
    print("GAP5-ONEBRAIN-SLEEPCYCLE DONE", flush=True)
