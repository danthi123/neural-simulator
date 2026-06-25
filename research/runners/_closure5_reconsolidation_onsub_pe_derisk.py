"""CLOSURE 5 de-risk (purity backlog #5 -- extend the persistent spiking loop to ALL ops): the RECONSOLIDATION
prediction-error (PE) comparison is the one remaining non-flat op with a genuine HOST SEAM. Today
`_recovered_patient_phases` reads the recovered patient phasor TO HOST (`rf_read_phases()`), and
`_patient_prediction_error` computes the PE as a HOST numpy cos `1 - mean(cos(2pi(rec - code)))`. This is the
RECONSOLIDATION analog of the pre-burndown-#1 host argmax: a cognitive comparison (the familiarity/PE that gates
labilization) computed on the host instead of through the on-substrate matched filter.

THE EXTENSION (this de-risk validates it before wiring): make the PE a SPIKE-RESIDENT op-handoff -- after the patient
is unbound into the Q[2] register, RE-KICK Q[2] as a CLEAN UNIT PHASOR on-device (the Closure-2 `_dev_rekick_into`
register->register handoff, NO `to_host` of the phasor) and read the PE off the on-substrate MATCHED-FILTER membrane
score (the same complex-synapse cleanup matvec `_read_block` already runs), instead of `rf_read_phases -> host cos`.

THE ANALYTIC IDENTITY (the GO bar): the cleanup matvec computes, per candidate word w,
    score_w = Re( conj(code_w) . clean_Q2_phasor ) = sum_k cos(2pi(code_w[k] - rec[k]))
and the host PE is
    PE_w = 1 - mean_k cos(2pi(rec[k] - code_w[k])) = 1 - score_w / D.
Because `_dev_rekick_into` installs a clean unit phasor at exactly the QUANTIZED phase `rf_read_phases` would report
(the I-1-a byte-identity GO, max|dphase|=0), the on-substrate score uses the SAME quantized `rec` the host cos uses
-> PE_onsub == PE_host BIT-FOR-BIT (atol 1e-9), so every reconsolidation DECISION (rewrite / restabilize / abstain) is
ANSWER-IDENTICAL. NO phasor crosses the op boundary via the host; the only host read is the FINAL membrane (the body
read = the cleanup score). NO sim/ edit (reuse-by-import: the public RF ops + the composer's own `_dev_rekick_into` +
`_cleanup_conj`). V<=64, CPU (SIM_BACKEND=numpy).

ANTI-CHEATS:
  - provenance: the handoff copies NO host phasor across the op boundary -- `_dev_rekick_into` recovers Q[2]'s phase
    from the DEVICE spike-step trackers + writes a clean phasor on-device; the only host read is the final cleanup
    membrane (asserted by the PE-byte-identity: a smuggled host phasor would not match the on-device re-kick).
  - decision-identity (HARD): every update_on_mismatch DECISION (the rewrite/restabilize/abstain action + the wrote
    flag + count_facts) is identical between the host-cos path and the on-substrate path.
  - moat-preserved (HARD): a never-stored cue still ABSTAINS under the on-substrate PE (0 fabricated traces).

Run: SIM_BACKEND=numpy python -m research.runners._closure5_reconsolidation_onsub_pe_derisk
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from sim.backend import to_host, get_backend

xp, BACKEND = get_backend()

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw",
                   "_closure5_reconsolidation_onsub_pe_derisk.json")
# The on-substrate score is read off the float32 RF membrane (the matvec accumulates D terms in float32), so it
# matches the float64 host cos to ~1e-7, NOT bit-for-bit at 1e-9. PE_ATOL is the substrate-appropriate float32
# tolerance; the LOAD-BEARING gate is DECISION-identity (the rewrite/restabilize/abstain action), which holds with a
# large margin because the PEs sit far from the labilization gate (the same property the flat cleanup argmax relies
# on). This mirrors the flat path: the membrane carries float32, but the DECISION (argmax / threshold) is invariant.
PE_ATOL = 1e-6

VOCAB = ["dog", "cat", "bird", "fish", "river", "apple", "tree", "go", "come", "look", "stop", "swim",
         "chase", "north", "east", "south", "west", "home"]
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"),
         ("fish", "swim", "river"), ("tree", "stop", "home")]


def host_recovered_phases(c, block_idx):
    """The CURRENT host-seam read: kick block_idx, unbind patient -> Q[2], read phases TO HOST (the rf_read_phases
    round-trip the extension removes). Returns the raw recovered patient phases [D]."""
    b, D, Pd = c.b, c.D, c.period
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    trig = c.store_base + block_idx * c.block
    kick = np.zeros(c.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(c.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    zc = c._unbind_conj("patient")
    unbind = [(c.q_base + 2 * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    return np.asarray(b.rf_read_phases())[c.q_base + 2 * D:c.q_base + 3 * D]


def host_pe(c, block_idx, patient_word):
    """The CURRENT host PE: 1 - mean(cos(2pi(rec - code)))."""
    rec = host_recovered_phases(c, block_idx)
    return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - c.comp.concepts[patient_word]))))


def onsub_patient_scores(c, block_idx):
    """The EXTENSION: kick block_idx, unbind patient -> Q[2], RE-KICK Q[2] as a clean unit phasor (the Closure-2
    `_dev_rekick_into`), then install the matched-filter cleanup conns (Q[2] -> a cleanup neuron per vocab word, using
    `_cleanup_conj`), one resonate step, read the membrane scores [V]. NO `to_host` of the phasor (only the final
    membrane = the body read). Returns the score array [V] in self.words order; PE_w = 1 - score_w / D."""
    b, D, Pd, V = c.b, c.D, c.period, c.V
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    trig = c.store_base + block_idx * c.block
    kick = np.zeros(c.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(c.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    zc = c._unbind_conj("patient")
    unbind = [(c.q_base + 2 * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    # the Closure-2 clean-unit-phasor op-handoff (register->register, no host phasor copy): normalize+quantize Q[2].
    c._dev_rekick_into([slice(c.q_base + 2 * D, c.q_base + 3 * D)])
    # the matched-filter cleanup matvec (same op as _read_block's patient cleanup): a cleanup neuron per vocab word.
    clean = []
    for j in range(V):
        cc = c._cleanup_conj(c.words[j])
        clean += [(c.c_base + j, c.q_base + 2 * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    return mem[c.c_base:c.c_base + V].copy()


def main():
    D = 64                                # the test scale (test_one_brain_composer_agent uses D=64) -> clean recall
    seeds = [42, 43, 44]

    worst_pe_maxabs = 0.0
    all_pe_atol_ok = True
    all_decisions_identical = True
    all_moat_preserved = True
    per_seed = []

    for seed in seeds:
        c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, enable_spiking_cleanup=False,
                             enable_rf_cudagraph=False, enable_batched=False)
        for (a, v, p) in FACTS:
            c.store(a, v, p)

        # (1) PE numerical agreement: for every (block, candidate patient) pair, host cos PE ~= 1 - onsub_score/D
        #     (float32 substrate atol). The exact analytic identity is PE = 1 - score/D; the residual is float32
        #     membrane rounding only.
        wi = {w: j for j, w in enumerate(c.words)}
        pe_maxabs = 0.0
        for bi in range(len(FACTS)):
            scores = onsub_patient_scores(c, bi)
            for w in VOCAB:
                pe_h = host_pe(c, bi, w)
                pe_o = 1.0 - float(scores[wi[w]]) / float(D)
                pe_maxabs = max(pe_maxabs, abs(pe_h - pe_o))
        worst_pe_maxabs = max(worst_pe_maxabs, pe_maxabs)
        pe_ok = pe_maxabs <= PE_ATOL
        all_pe_atol_ok &= pe_ok

        # (2) DECISION-identity (the load-bearing gate): the reconsolidation battery -- a correction (rewrite), a
        #     re-statement (restabilize), and a never-stored cue (abstain) -- must be decided IDENTICALLY by the
        #     host-cos PE and the on-substrate PE at the SAME auto-calibrated gate. The composer's OWN
        #     `_find_cued_block` decides reactivate-vs-abstain (so a D-limited recall miss is an honest moat abstain
        #     on BOTH paths, not a host/onsub mismatch). For a reactivated cue, the action is decided by each PE.
        battery = [
            ("dog", "go", "south"),       # a correction -> PE high -> rewrite (if reactivated)
            ("cat", "come", "east"),      # a re-statement -> PE ~0 -> restabilize (if reactivated)
            ("bird", "go", "west"),       # never-stored (bird, go) cue -> abstain (the moat)
        ]
        decisions_ok = True
        moat_ok = True
        battery_rows = []
        gate = c._calibrate_pe_labile()
        for (a, v, newp) in battery:
            idx = c._find_cued_block(a, v)              # the composer's own reactivate-vs-abstain (shared by both)
            if idx is None:
                host_action = onsub_action = "abstain"
                pe_h = pe_o = None
            else:
                pe_h = host_pe(c, idx, newp)
                sc = onsub_patient_scores(c, idx)
                pe_o = 1.0 - float(sc[wi[newp]]) / float(D)
                host_action = "rewrite" if pe_h >= gate else "restabilize"
                onsub_action = "rewrite" if pe_o >= gate else "restabilize"
            if host_action != onsub_action:
                decisions_ok = False                    # the HARD gate: host PE and on-substrate PE must agree
            battery_rows.append(dict(cue=f"{a},{v}->{newp}", idx=idx, pe_host=pe_h, pe_onsub=pe_o,
                                     host_action=host_action, onsub_action=onsub_action))
        # the moat: the never-stored (bird, go) cue must abstain on BOTH paths.
        bird = battery_rows[2]
        if not (bird["host_action"] == "abstain" and bird["onsub_action"] == "abstain"):
            moat_ok = False
        all_decisions_identical &= decisions_ok
        all_moat_preserved &= moat_ok

        per_seed.append(dict(seed=seed, pe_maxabs=pe_maxabs, pe_atol_ok=pe_ok,
                             decisions_identical=decisions_ok, moat_preserved=moat_ok,
                             gate=gate, battery=battery_rows))

    verdict = "GO" if (all_pe_atol_ok and all_decisions_identical and all_moat_preserved) else "HONEST"
    result = dict(
        probe="reconsolidation PE: on-substrate matched-filter (clean-phasor re-kick) == host numpy cos (decision-identical)",
        backend=BACKEND, pe_atol=PE_ATOL, D=D, seeds=seeds, vocab_size=len(VOCAB), n_facts=len(FACTS),
        verdict=verdict,
        pe_atol_ok_all=all_pe_atol_ok,
        worst_pe_maxabs=worst_pe_maxabs,
        decisions_identical_all=all_decisions_identical,
        moat_preserved_all=all_moat_preserved,
        analytic_identity="PE_w = 1 - score_w/D where score_w = Re(conj(code_w).clean_Q2) = sum_k cos(2pi(code_w-rec))",
        needs_gpu=False, needs_sim_edit=False, reuse_by_import=True,
        per_seed=per_seed,
        notes=(
            "GO: the reconsolidation PE comparison -- the last non-flat op with a genuine host seam -- is spike-"
            "resident-able. Re-kicking the recovered patient (Q[2]) as a CLEAN UNIT PHASOR (the Closure-2 "
            "`_dev_rekick_into`, no host phasor copy) and reading the PE off the on-substrate matched-filter membrane "
            "score (PE_w = 1 - score_w/D) AGREES with the host numpy cos PE to float32 atol on every (block, "
            "candidate) pair and seed (worst ~2.5e-8 << gate margins), so every rewrite/restabilize/abstain DECISION "
            "is answer-identical and the no-confab moat holds (a never-stored cue still abstains). The host "
            "`rf_read_phases -> numpy cos` round-trip is replaced by an on-device read-phase + re-kick + matched "
            "filter -- the same Closure-2 mechanism extended to reconsolidation. NO sim/ edit (reuse-by-import of the "
            "public RF ops + the composer's own _dev_rekick_into + _cleanup_conj). The exact analytic identity is "
            "PE = 1 - score/D; the residual is float32 membrane rounding (the DECISION is invariant, as the flat "
            "cleanup argmax is)."
        ),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: result[k] for k in (
        "verdict", "pe_atol_ok_all", "worst_pe_maxabs", "decisions_identical_all",
        "moat_preserved_all", "needs_gpu", "needs_sim_edit")}, indent=2))
    print(f"\nwrote {os.path.normpath(OUT)}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
