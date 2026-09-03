"""CLASS CH — the production brain must BUILD + RUN a chat turn inside a SINGLE CONSUMER RTX 3090's VRAM.

WHY (owner directive, 2026-09-03; memory `project_consumer_hardware_reference_principle.md`): "the production
sim-brain's performance + runability are FRAMED AROUND THE MOST ACCESSIBLE CONSUMER HARDWARE: a single consumer
RTX 3090 (24 GB VRAM, 936 GB/s). The production brain must BUILD + run a live chat turn within a single
consumer 3090's limits, and must NOT balloon to require exotic/datacenter GPUs (A100/H100/multi-card-required)."
This is mission-aligned, not a hardware nitpick: an artificial-life brain that runs on a ~$1k consumer GPU is a
more meaningful and more reproducible claim than one needing a cluster. Like the plasticity-bound-trap and the
cfg.seed trap in CLAUDE.md, this was a REMEMBERED constraint until now — nothing blocked a commit that quietly
grew the production default past what a 3090 can hold. This gate makes it a CHECK.

WHAT IT ESTIMATES, and what it deliberately does NOT. This gate estimates ONLY the GPU-resident SPIKING
SUBSTRATE arrays — the per-neuron and per-synapse `cp_*` state CuPy allocates in `sim/bridge.py` for a
`SimulationBridge`. It does NOT estimate: the host-RAM-resident VSA knowledge store (`ShardedPhasorStore` /
`TieredFactStore`, numpy, currently ~100k facts per `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s
`tiered-knowledge-ltm` row — a system-RAM budget, not VRAM); the Qwen scaffold LLM's own VRAM (a separate,
well-known consumer-sized model, being retired per the ACTIVE MISSION, not part of "the brain"); or per-step
transient activation/scratch buffers CuPy's memory pool churns through a forward pass (folded into the fixed
overhead + safety margin below, not itemized). A gate that tried to model everything would be unmaintainable and
untrustworthy; this one is scoped to the part the ACTIVE MISSION's "ONE spiking substrate" language makes
load-bearing, and says so.

THE FORMULA:  bytes = (n_neurons * PER_NEURON_BYTES + n_synapses * PER_SYNAPSE_BYTES) * SAFETY_MULTIPLIER
                       + FIXED_OVERHEAD_BYTES

PER_NEURON_BYTES = 200, itemized from the actual `self.cp_<x> = cp.zeros/empty/full(n, ...)` allocations in
`sim/bridge.py` (grepped, not guessed) with every optional feature counted as ON (worst case): 36 float32
per-neuron arrays (external_input_current, conductance_g_{e,i,nmda,nmda_rise,nmda_recurrent{,_rise},gabab,
gabab_slow,coincidence{,_rise},graded_plateau{,_rise}}, syn_reversal_potential_i_per_neuron, input_mean_ema,
ssm_{state,inject,shunt}, neuron_activity_ema, hebb_coactivity_trace, reward_coactivity_trace,
dendritic_source_activity, izh_{C,k,vr,vt,vpeak,a,b,c_reset,d_increment}, membrane_potential_v,
recovery_variable_u, last_spike_time, inhibitory_stdp_trace, ou_current) + 3 int32 (neuron_type_ids,
refractory_timers, viz_activity_timers) + 3 bool (firing_states, prev_firing_states, rf_fired) + a 3-float
position (cp_neuron_positions_3d) + ~2 floats of traits ~= 36*4 + 3*4 + 3*1 + 12 + 8 = 179 bytes/neuron,
rounded up to 200 for arrays this pass did not enumerate.

PER_SYNAPSE_BYTES = 64, itemized from the CSR `cp_connections` (data=f32, indices=i32; indptr amortizes to
~0/synapse at scale) plus the per-synapse `cp.zeros/ones/full(nnz or capacity, ...)` arrays: data(4) +
indices(4) + plasticity_rate_gain(4) + transmission_gain(4) + eligibility_trace(4) + stp_x(4) + stp_u(4) +
synapse_survival(4) + synapse_pulse_progress(4) + traits(4) + synapse_pulse_timers(4) + synapse_action_tag(4)
+ synapse_alive(1) + gabab_synapse_mask(1) + nmda_recurrent_synapse_mask(1) + coincidence_synapse_mask(1) +
graded_synapse_mask(1) + stp_disabled_mask(1) = 54 bytes/synapse, rounded up to 64.
`cp_connections` is a SPARSE `csr_matrix` (`sim/bridge.py`, e.g. line ~2754/~11072) — synapse count scales with
`connections_per_neuron` (sparse), never O(n_neurons^2) dense, which is why this formula is linear in n_synapses
rather than quadratic in n_neurons.

SAFETY_MULTIPLIER = 8x on the itemized (neuron+synapse) cost. Documented, not hidden: (a) a live chat turn
builds MULTIPLE separate co-resident `SimulationBridge` instances, not one — most `on_by_default: YES` rows in
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` are explicitly "co-resident, not yet merged" with the largest pool
this gate's N is drawn from; (b) this itemization is not exhaustive (dendritic-clustering / structural-plasticity
growth scratch, viz buffers); (c) CuPy's default memory pool does not return freed blocks to the driver, so
observed peak can exceed the steady-state sum. FIXED_OVERHEAD_BYTES = 1 GiB for the CUDA context + CuPy's
default pool reservation + host-pinned transfer buffers — a conservative round number, not a measurement; the
owner should verify both constants against `nvidia-smi` on a real 3090 run when one is convenient (CPU-only
per this task's constraints, so that verification was NOT done here).

N (neuron count): this gate does NOT try to sum every organ's bridge across the whole live production pipeline
(that needs execution, not statics, and would need a GPU this task was told not to use). Instead it takes the
MAX of (a) every `N=<number>` / `N≈<number>` this pass can find in `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`
(self-updating — a future larger documented pool is picked up automatically without editing this file) and
(b) `N_PRODUCTION_REFERENCE` below, a citation-dated floor. `connections_per_neuron` is read LIVE via regex from
`sim/config.py`'s `CoreSimConfig` default, so a real future change to that default is NOT silently missed.

WHAT IT CANNOT CATCH: a footprint blow-up from something this pass's array itemization missed entirely (a new
`cp_*` array class added later, or a config change that switches to dense connectivity) will silently move only
the "confident" input this gate does not track (n_neurons/n_synapses/connections_per_neuron are; a wholly NEW
array class is not). And a genuinely correct config that is simply mis-projected because BOTH inputs are stale
sits inside the same blind spot every static estimator has: it can be wrong, but it cannot be wrong in the
direction of a false BLOCK on a config it was never shown. Per this task's explicit instruction, missing or
unresolvable inputs make this gate PASS with a note — it never false-blocks.
"""
from __future__ import annotations

import contextlib
import io
import os
import re

NAME = "consumer-hardware-reference"
CLASS_ID = "CH"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PY = os.path.join(_ROOT, "sim", "config.py")
_LEDGER = os.path.join(_ROOT, "docs", "PRODUCTION_INTEGRATION_LEDGER.yaml")

GIB = 1024 ** 3
HARD_CEILING_BYTES = 24 * GIB          # a single consumer RTX 3090
SOFT_WARN_BYTES = 22 * GIB             # leaves desktop/OS headroom on a real 24 GiB card

PER_NEURON_BYTES = 200                 # see docstring itemization
PER_SYNAPSE_BYTES = 64                 # see docstring itemization
SAFETY_MULTIPLIER = 8                  # see docstring rationale (co-resident bridges + non-exhaustive itemization)
FIXED_OVERHEAD_BYTES = 1 * GIB         # CUDA context + CuPy pool baseline (conservative, not measured)

# The largest documented default-on merged-pool substrate as of 2026-09-02 (11-organ wave-3 pool,
# research/findings/2026-09-02-pmem-wave3-pool-faculty-alive-and-answer-GO.md: "the wave-3 (11-organ, N=7002)
# merged-pool"). MAINTENANCE: bump this (or trust the ledger scan below, which supersedes it automatically)
# when a larger merged pool becomes the documented production default.
N_PRODUCTION_REFERENCE = 7002

_N_PATTERN = re.compile(r"N\s*[=≈~]\s*([\d,]{2,7})\b")
_CPN_PATTERN = re.compile(r"connections_per_neuron\s*:\s*int\s*=\s*(\d+)")


def estimate_vram_bytes(n_neurons, n_synapses):
    """Pure formula -- see the module docstring. No file I/O, so selftest can drive it directly."""
    raw = n_neurons * PER_NEURON_BYTES + n_synapses * PER_SYNAPSE_BYTES
    return int(raw * SAFETY_MULTIPLIER + FIXED_OVERHEAD_BYTES)


def classify(bytes_est):
    """PASS / WARN / BLOCK against the two thresholds. Strictly-greater-than at each boundary (an estimate
    AT a threshold has not yet crossed it)."""
    if bytes_est > HARD_CEILING_BYTES:
        return "BLOCK"
    if bytes_est > SOFT_WARN_BYTES:
        return "WARN"
    return "PASS"


def _max_documented_n(text):
    """The largest sane N=/N~=/N-approx value mentioned anywhere in the ledger text. Bounded to reject obvious
    noise (a version number, a byte count, a percentage) rather than trusting every regex hit."""
    vals = []
    for m in _N_PATTERN.finditer(text):
        try:
            v = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 50 <= v <= 2_000_000:
            vals.append(v)
    return max(vals) if vals else None


def _read_connections_per_neuron_default():
    try:
        text = open(_CONFIG_PY, errors="ignore").read()
    except OSError:
        return None
    m = _CPN_PATTERN.search(text)
    return int(m.group(1)) if m else None


def _production_estimate():
    """Returns (bytes_est, n_neurons, n_synapses, connections_per_neuron), or None if a required input could
    not be confidently resolved (never false-blocks on a missing input -- see docstring)."""
    cpn = _read_connections_per_neuron_default()
    if cpn is None:
        return None
    n_neurons = N_PRODUCTION_REFERENCE
    try:
        ledger_text = open(_LEDGER, errors="ignore").read()
    except OSError:
        ledger_text = ""
    ledger_n = _max_documented_n(ledger_text)
    if ledger_n is not None:
        n_neurons = max(n_neurons, ledger_n)
    n_synapses = n_neurons * cpn
    return estimate_vram_bytes(n_neurons, n_synapses), n_neurons, n_synapses, cpn


def _decide(estimate):
    """Pure decision logic, independent of file I/O, so selftest exercises it directly with synthetic inputs.
    Returns (problems: list[str], status_line: str, warn_line: str | None)."""
    if estimate is None:
        return (
            [],
            "consumer-hardware-reference: could not confidently resolve sim/config.py's "
            "connections_per_neuron default -- SKIPPING the VRAM estimate this run (never false-blocks on a "
            "missing input).",
            None,
        )
    bytes_est, n_neurons, n_synapses, cpn = estimate
    status = (
        "consumer-hardware-reference: estimated production spiking-substrate VRAM ~%.3f GiB "
        "(N=%d neurons, %d synapses @ connections_per_neuron=%d, x%d safety margin, +%.1f GiB fixed overhead) "
        "vs the 24 GiB single-consumer-RTX-3090 reference -- %.2f GiB headroom (%.1f%% of budget used)."
        % (
            bytes_est / GIB, n_neurons, n_synapses, cpn, SAFETY_MULTIPLIER, FIXED_OVERHEAD_BYTES / GIB,
            (HARD_CEILING_BYTES - bytes_est) / GIB, 100.0 * bytes_est / HARD_CEILING_BYTES,
        )
    )
    cls = classify(bytes_est)
    if cls == "BLOCK":
        problems = [
            "production spiking-substrate VRAM estimate ~%.2f GiB EXCEEDS the 24 GiB single-consumer-RTX-3090 "
            "reference (owner directive 2026-09-03: the production brain must build+run within a single "
            "consumer 3090's limits and must NOT balloon to exotic/datacenter GPUs). N=%d neurons, %d synapses "
            "@ connections_per_neuron=%d. Reduce n_neurons / connections_per_neuron on the production default "
            "path, or shard/stream the substrate, before landing this change."
            % (bytes_est / GIB, n_neurons, n_synapses, cpn)
        ]
        return problems, status, None
    if cls == "WARN":
        return (
            [],
            status,
            "⚠️  consumer-hardware-reference: estimate sits in the 22-24 GiB SOFT band -- leaves "
            "little desktop/OS headroom on a real 24 GiB 3090. Not blocking.",
        )
    return [], status, None


def check(paths):
    del paths  # a repo-state estimate, not a per-file diff -- see lane_starvation / device_and_cost for the
               # same pattern of re-deriving live state rather than trusting the staged-file list.
    problems, status, warn = _decide(_production_estimate())
    print(status)
    if warn:
        print(warn)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: a confidently over-24GiB estimate MUST block; real-scale/boundary/missing-input
    cases must NOT."""
    bad = []

    over = estimate_vram_bytes(5_000_000, 500_000_000)   # ~247 GiB -- far past 24 GiB
    if over <= HARD_CEILING_BYTES:
        bad.append("estimate_vram_bytes() did not treat a 5M-neuron/500M-synapse config as over budget")
    probs, _, _ = _decide((over, 5_000_000, 500_000_000, 100))
    if not probs:
        bad.append("did NOT block a confidently over-24GiB estimate")

    # SAFETY_MULTIPLIER must be LOAD-BEARING, not decorative: pick a config that is comfortably under budget
    # at 1x but over the hard ceiling at the real (8x) multiplier, and confirm estimate_vram_bytes() actually
    # applies the multiplier rather than silently reading as if it were 1x (the earlier draft of this selftest
    # missed exactly this -- its "far past 24GiB" case stayed over budget even with the multiplier zeroed out,
    # so a neutered multiplier passed silently until this straddle case was added).
    straddle_n, straddle_s = 1_000_000, 50_000_000
    straddle_raw = straddle_n * PER_NEURON_BYTES + straddle_s * PER_SYNAPSE_BYTES
    straddle_at_1x = straddle_raw + FIXED_OVERHEAD_BYTES
    straddle_at_real = estimate_vram_bytes(straddle_n, straddle_s)
    if straddle_at_1x > HARD_CEILING_BYTES:
        bad.append("SANITY: the multiplier-straddle case is over budget even at 1x -- picked poorly, fix the "
                    "constants in this test")
    elif straddle_at_real <= HARD_CEILING_BYTES:
        bad.append("SAFETY_MULTIPLIER has NO EFFECT on the verdict -- a case designed to block ONLY because of "
                    "the margin passed anyway (the multiplier may have been neutered)")

    # PER_NEURON_BYTES must be LOAD-BEARING too: an all-neuron, zero-synapse config isolates it from
    # PER_SYNAPSE_BYTES entirely (the "over" case above has 500M synapses, which alone clears the ceiling
    # regardless of PER_NEURON_BYTES -- this case would NOT catch PER_NEURON_BYTES being zeroed).
    if estimate_vram_bytes(20_000_000, 0) <= HARD_CEILING_BYTES:
        bad.append("PER_NEURON_BYTES has NO EFFECT on the verdict -- a 20M-neuron, ZERO-synapse config designed "
                    "to block on per-neuron cost alone passed anyway")

    # check() ITSELF must not swallow a BLOCK verdict on the way out -- the above only tests _decide(); a real
    # regression could live in check()'s own plumbing (e.g. computing problems and then returning [] anyway).
    _orig_production_estimate = globals()["_production_estimate"]
    try:
        globals()["_production_estimate"] = lambda: (over, 5_000_000, 500_000_000, 100)
        with contextlib.redirect_stdout(io.StringIO()):
            check_problems = check(None)
    finally:
        globals()["_production_estimate"] = _orig_production_estimate
    if not check_problems:
        bad.append("check() did NOT return a blocking problem for a confidently over-24GiB production estimate "
                    "(tests the real entry point, not just the internal _decide() helper)")

    normal = estimate_vram_bytes(N_PRODUCTION_REFERENCE, N_PRODUCTION_REFERENCE * 100)
    probs, _, _ = _decide((normal, N_PRODUCTION_REFERENCE, N_PRODUCTION_REFERENCE * 100, 100))
    if probs:
        bad.append("FALSE POSITIVE: blocked the real production-scale estimate (~%.3f GiB)" % (normal / GIB))
    if normal >= SOFT_WARN_BYTES:
        bad.append("SANITY: the real production-scale estimate is not comfortably under the soft threshold "
                    "-- the per-unit constants may need revisiting")

    # the soft-warn band (22-24 GiB) must warn but never block.
    probs, _, warn = _decide((SOFT_WARN_BYTES + 1, 0, 0, 100))
    if probs:
        bad.append("FALSE POSITIVE: blocked an estimate inside the 22-24 GiB soft-warn band")
    if not warn:
        bad.append("did NOT raise the soft-warn informational line in the 22-24 GiB band")

    # exactly AT each threshold has not yet crossed it.
    probs, _, warn = _decide((SOFT_WARN_BYTES, 0, 0, 100))
    if probs or warn:
        bad.append("FALSE POSITIVE: warned/blocked AT the 22 GiB threshold (should be strictly over)")
    probs, _, _ = _decide((HARD_CEILING_BYTES, 0, 0, 100))
    if probs:
        bad.append("FALSE POSITIVE: blocked AT the 24 GiB ceiling (should be strictly over)")

    # a missing input must PASS, never false-block, and must explain itself.
    probs, status, _ = _decide(None)
    if probs:
        bad.append("FALSE POSITIVE: blocked when the footprint could not be confidently computed")
    if "could not confidently" not in status:
        bad.append("the missing-input path did not explain itself in its status line")

    # classify() boundary sanity, independent of _decide's wiring.
    if classify(HARD_CEILING_BYTES + 1) != "BLOCK":
        bad.append("classify() did not BLOCK just over the hard ceiling")
    if classify(SOFT_WARN_BYTES + 1) != "WARN":
        bad.append("classify() did not WARN just over the soft threshold")
    if classify(SOFT_WARN_BYTES) != "PASS":
        bad.append("classify() treated the soft threshold itself as WARN (should be strictly over)")

    # _max_documented_n: picks the largest SANE hit, rejects out-of-range noise.
    if _max_documented_n("the wave-3 (11-organ, N=7002) merged-pool; earlier N=450 and N=1584") != 7002:
        bad.append("_max_documented_n did not pick the largest in-range N= mention")
    # 5,000,001 is 7 digits (within the regex's own length window) but past the 2,000,000 sanity ceiling --
    # this exercises the NUMERIC range guard specifically, not just the regex's digit-count limit (an earlier
    # draft used an 8-digit value here, which the regex's own \b-bounded {2,7} already excludes on length
    # alone, so it passed even with the numeric range check deleted entirely).
    if _max_documented_n("batch N=5000001 units") is not None:
        bad.append("FALSE POSITIVE: _max_documented_n accepted an out-of-sane-range value")
    if _max_documented_n("no mentions here") is not None:
        bad.append("FALSE POSITIVE: _max_documented_n found a value in text with none")

    return bad


if __name__ == "__main__":
    print("selftest:", selftest())
    check(None)
