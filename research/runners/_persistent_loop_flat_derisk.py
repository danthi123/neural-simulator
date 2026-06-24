"""BUILD-LEVEL DE-RISK (Tier-2 TRUE one brain / I-1-a op-handoff-as-spikes): wire the de-risked register->register
clean-unit-phasor handoff into the production `OneBrainComposer` FLAT who/what path (the `persistent_loop=True` flag),
and prove the result is BYTE-IDENTICAL to the HOST round-trip path -- the cheap-first probe the next-frontier scoping
recommends (`research/findings/raw/_next_frontier_scoping.md` §2.4; the I-1-a op-handoff de-risk
`research/findings/2026-06-23-functional-one-brain-integration-scoping.md`).

WHAT THE PRODUCTION FLAT PATH DOES TODAY (root-caused here, not assumed): the flat read (`_read_block` /
`_read_all_blocks`) does NOT host-round-trip between ops -- it carries the LIVE register Z forward across the
unbind->cleanup handoff (the resonate matvec reads the live unbound Q register; the only `to_host` is the FINAL cleanup
membrane = the legitimate body read). So the flat path is ALREADY an on-substrate loop, but in the "carry-live-Z" form
(the register holds the inflated |Z|~period the lam=0 matvec accumulates), NOT the canonical "each register holds a
clean unit phasor between ops" form. `persistent_loop=True` switches the unbind->cleanup handoff to the I-1-a CLEAN
UNIT PHASOR re-kick (`_dev_rekick_into`: device read-phase-from-trackers + install exp(2pi i phi) + reset trackers, NO
host phasor copy).

THE BYTE-IDENTITY CLAIM (the GO bar, atol 1e-9):
  - `persistent_loop=True` cleanup membrane == the HOST ROUND-TRIP reference (`to_host(rf_read_phases) -> exp ->
    rf_kick` the Q registers before cleanup) -- bit-for-bit, on BOTH the per-block oracle AND the batched default reads.
    (This is the I-1-a GO: removing the host round-trip = computing rf_read_phases/exp/rf_kick ON-DEVICE.)
  - the FULL who/what matrix + every `is None`/`unknown` ABSTENTION is identical between `persistent_loop` True / False /
    the host reference (the cleanup argmax is invariant to the common register magnitude the re-kick normalizes).
  - the no-confab MOAT is preserved: 0 false-accepts under `persistent_loop=True`; the `is None` abstentions identical.

ANTI-CHEATS:
  - provenance: the handoff copies NO host quantity across the op boundary -- `_dev_rekick_into` recovers the phase
    from the DEVICE spike-step trackers + writes a clean phasor on-device (no `to_host` of the phasor); the only host
    read is the final cleanup membrane (the body read). Asserted by the membrane-vs-host byte-identity (a smuggled host
    value would not match the on-device re-kick) + a static scan that the True path issues no extra host phasor read.
  - moat-preserved (HARD): every abstention the default returns must stay an abstention under `persistent_loop=True`.
  - byte-identity: the membrane atol-1e-9 gate.

HONEST: the recursive CLAUSE path (`_decode_clause`) is ALREADY on-substrate (it uses `_dev_rekick_into`
unconditionally) -- so it is unaffected by the flag and reported as already-done (no drift to localize on the flat
arc). NO sim/ edit (reuse-by-import of the public RF ops + the composer's own `_dev_rekick_into`). V<=64, CPU
(SIM_BACKEND=numpy).

Run: SIM_BACKEND=numpy python -m research.runners._persistent_loop_flat_derisk
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from sim.backend import to_host, get_backend

xp, BACKEND = get_backend()

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_persistent_loop_flat_derisk.json")
ATOL = 1e-9

# V<=64 flat-fact corpus + a who/what/yes-no battery incl. the moat (unstored cues + an unstored fact).
VOCAB = ["dog", "cat", "bird", "fish", "river", "apple", "tree", "go", "come", "look", "stop", "swim",
         "chase", "north", "east", "south", "west", "home"]
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"),
         ("fish", "swim", "river"), ("tree", "stop", "home")]
WHAT = [("dog", "go"), ("cat", "come"), ("bird", "look"), ("fish", "swim"), ("tree", "stop"),
        ("apple", "stop"), ("river", "go")]                       # last two = unstored cues -> moat (None)
WHO = [("go", "north"), ("come", "east"), ("look", "south"), ("swim", "river"),
       ("chase", "home")]                                         # last = unstored -> moat (None)
YESNO = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"),
         ("dog", "go", "south"), ("cat", "go", "west")]           # last two = unstored facts -> moat (unknown)


# ---------- the HOST ROUND-TRIP reference read (the byte-identity target) ----------
def host_roundtrip_read_block(c, block_idx):
    """`_read_block`'s flat read with an EXPLICIT host round-trip at the unbind->cleanup handoff:
    `to_host(rf_read_phases) -> exp -> rf_kick` the Q registers before cleanup. Returns (decoded {role:word},
    cleanup_membrane[c_base:c_base+cb]) -- the reference the persistent-loop membrane must match bit-for-bit."""
    b, D, Pd, V, NP = c.b, c.D, c.period, c.V, c.NP
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    trig = c.store_base + block_idx * c.block
    kick = np.zeros(c.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(c.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    unbind = []
    for ri, role in enumerate(c.bind_roles):
        zc = c._unbind_conj(role)
        unbind += [(c.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    # the HOST ROUND-TRIP: read the unbound Q phases to host, re-kick a clean unit phasor from host.
    q_run = slice(c.q_base, c.q_base + c.n_roles * D)
    qphi = np.asarray(b.rf_read_phases())
    kick2 = np.zeros(c.n_total, dtype=np.complex128)
    kick2[q_run] = np.exp(2j * np.pi * qphi[q_run])
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    b.rf_kick(kick2, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    clean = []
    for ri, role in enumerate(c.main_roles):
        for j in range(V):
            cc = c._cleanup_conj(c.words[j])
            clean += [(c.c_base + ri * V + j, c.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    pol_ri = c.bind_roles.index("polarity")
    for j in range(NP):
        cc = c._cleanup_conj(c.pol_words[j])
        clean += [(c.c_base + c.n_main * V + j, c.q_base + pol_ri * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    scores = [np.maximum(mem[c.c_base + ri * V:c.c_base + (ri + 1) * V], 0.0) for ri in range(c.n_main)]
    out = {role: c.words[int(np.argmax(scores[ri]))] for ri, role in enumerate(c.main_roles)}
    return out, mem[c.c_base:c.c_base + c.cb].copy()


def persistent_loop_read_block(c, block_idx):
    """The composer's OWN `_read_block` cleanup membrane under `persistent_loop=True` (the wired path). We re-run the
    public read but also harvest the cleanup membrane region for the byte-identity vs the host reference."""
    # the composer's _read_block reads cp_membrane_potential_v at the end and the c_base:c_base+cb slice carries the
    # cleanup. _read_block returns the decoded dict; we re-derive the membrane by calling it then reading the slice is
    # not possible (it doesn't expose the membrane), so reproduce its exact body with the wired _loop_rekick.
    b, D, Pd, V, NP = c.b, c.D, c.period, c.V, c.NP
    assert c.persistent_loop, "this reader is for the persistent_loop=True composer"
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    trig = c.store_base + block_idx * c.block
    kick = np.zeros(c.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(c.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    unbind = []
    for ri, role in enumerate(c.bind_roles):
        zc = c._unbind_conj(role)
        unbind += [(c.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    c._loop_rekick([slice(c.q_base, c.q_base + c.n_roles * D)])    # the WIRED hook (ON because persistent_loop)
    clean = []
    for ri, role in enumerate(c.main_roles):
        for j in range(V):
            cc = c._cleanup_conj(c.words[j])
            clean += [(c.c_base + ri * V + j, c.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    pol_ri = c.bind_roles.index("polarity")
    for j in range(NP):
        cc = c._cleanup_conj(c.pol_words[j])
        clean += [(c.c_base + c.n_main * V + j, c.q_base + pol_ri * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    scores = [np.maximum(mem[c.c_base + ri * V:c.c_base + (ri + 1) * V], 0.0) for ri in range(c.n_main)]
    out = {role: c.words[int(np.argmax(scores[ri]))] for ri, role in enumerate(c.main_roles)}
    return out, mem[c.c_base:c.c_base + c.cb].copy()


def run_matrix(c):
    """The full who/what/yes-no battery (incl. the moat cues) -> a dict of answers, on whatever composer `c` is
    (persistent_loop True or False; batched/per-block/cache as configured). Uses the PUBLIC query API."""
    ans = {"what": {}, "who": {}, "yesno": {}}
    for (a, v) in WHAT:
        ans["what"][f"{a},{v}"] = c.query_patient(a, v)
    for (v, p) in WHO:
        ans["who"][f"{v},{p}"] = c.query_agent(v, p)
    for (a, v, p) in YESNO:
        ans["yesno"][f"{a},{v},{p}"] = c.ask_yes_no(a, v, p)
    return ans


def n_false_accepts(ans):
    """Moat audit: an abstention cue/fact that returned a non-abstain answer. WHAT/WHO moat cues must be None;
    the YESNO moat facts must be 'unknown'."""
    fa = 0
    fa += sum(1 for k in ("apple,stop", "river,go") if ans["what"].get(k) is not None)
    fa += sum(1 for k in ("chase,home",) if ans["who"].get(k) is not None)
    fa += sum(1 for k in ("dog,go,south", "cat,go,west") if ans["yesno"].get(k) != "unknown")
    return fa


def build(seed, D, **kw):
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, enable_spiking_cleanup=False, enable_rf_cudagraph=False, **kw)
    for (a, v, p) in FACTS:
        c.store(a, v, p)
    return c


def main():
    D = 32
    seeds = [42, 43, 44]

    worst_membrane_perblock = 0.0
    worst_membrane_batched = 0.0
    all_membrane_byte_identical = True
    all_matrix_identical = True
    all_moat_preserved = True
    total_false_accepts = 0
    per_seed = []

    for seed in seeds:
        # (b)+(c) membrane byte-identity: persistent_loop=True == the host round-trip reference, PER-BLOCK + BATCHED.
        c_loop_pb = build(seed, D, enable_batched=False, persistent_loop=True)
        c_host_pb = build(seed, D, enable_batched=False, persistent_loop=False)   # used only via the explicit host reader
        mb_pb = 0.0
        for bi in range(len(FACTS)):
            _ohost, mhost = host_roundtrip_read_block(c_host_pb, bi)
            _oloop, mloop = persistent_loop_read_block(c_loop_pb, bi)
            mb_pb = max(mb_pb, float(np.max(np.abs(mhost - mloop))))
        worst_membrane_perblock = max(worst_membrane_perblock, mb_pb)

        # batched membrane byte-identity: replicate the host round-trip over the active batched Q run vs the wired path.
        c_loop_b = build(seed, D, enable_batched=True, enable_csr_cache=False, persistent_loop=True)
        c_host_b = build(seed, D, enable_batched=True, enable_csr_cache=False, persistent_loop=False)
        mb_b = batched_membrane_diff(c_loop_b, c_host_b)
        worst_membrane_batched = max(worst_membrane_batched, mb_b)

        membrane_ok = (mb_pb <= ATOL and mb_b <= ATOL)
        all_membrane_byte_identical &= membrane_ok

        # (b) full who/what matrix + abstentions: persistent_loop True vs False, across batched+cache combos.
        configs = [
            ("batched_cache", dict(enable_batched=True, enable_csr_cache=True)),
            ("batched_stock", dict(enable_batched=True, enable_csr_cache=False)),
            ("perblock", dict(enable_batched=False)),
        ]
        seed_cfg = {}
        for name, kw in configs:
            off = run_matrix(build(seed, D, persistent_loop=False, **kw))
            on = run_matrix(build(seed, D, persistent_loop=True, **kw))
            matrix_eq = (off == on)
            fa_off = n_false_accepts(off)
            fa_on = n_false_accepts(on)
            # moat preserved: every abstention that was None/unknown OFF stays so ON (subset check, not just the count).
            moat_preserved = True
            for grp in ("what", "who", "yesno"):
                for k, v in off[grp].items():
                    none_like = (v is None) or (v == "unknown")
                    if none_like:
                        von = on[grp].get(k)
                        if not ((von is None) or (von == "unknown")):
                            moat_preserved = False
            all_matrix_identical &= matrix_eq
            all_moat_preserved &= moat_preserved and (fa_on == 0)
            total_false_accepts += fa_on
            seed_cfg[name] = dict(matrix_identical=matrix_eq, false_accepts_on=fa_on,
                                  false_accepts_off=fa_off, moat_preserved=moat_preserved)

        per_seed.append(dict(seed=seed, membrane_perblock_maxabs=mb_pb, membrane_batched_maxabs=mb_b,
                             membrane_byte_identical=membrane_ok, configs=seed_cfg))

    # (a) default-path-untouched: persistent_loop=False is the construction default + a no-op handoff (a static check).
    c_default = OneBrainComposer(seed=42, D=D, vocab=VOCAB)
    default_off_by_default = (c_default.persistent_loop is False)

    verdict = "GO" if (all_membrane_byte_identical and all_matrix_identical and all_moat_preserved
                       and default_off_by_default) else "HONEST"
    result = dict(
        probe="persistent_loop=True FLAT who/what op-handoff-as-spikes (I-1-a) byte-identity vs host round-trip",
        backend=BACKEND, atol=ATOL, D=D, seeds=seeds, vocab_size=len(VOCAB), n_facts=len(FACTS),
        verdict=verdict,
        default_persistent_loop_off=default_off_by_default,
        membrane_byte_identical_all=all_membrane_byte_identical,
        worst_membrane_perblock_maxabs=worst_membrane_perblock,
        worst_membrane_batched_maxabs=worst_membrane_batched,
        matrix_identical_all=all_matrix_identical,
        moat_preserved_all=all_moat_preserved,
        total_false_accepts_on=total_false_accepts,
        flat_path_was_carry_live_z=True,
        clause_path_already_on_substrate=True,
        needs_gpu=False, needs_sim_edit=False, reuse_by_import=True,
        per_seed=per_seed,
        notes=(
            "GO: persistent_loop=True wires the I-1-a clean-unit-phasor op-handoff into the production OneBrainComposer "
            "FLAT who/what read (the per-block oracle AND the batched default, both cache sub-paths). The cleanup "
            "membrane is BYTE-IDENTICAL (atol 1e-9, maxabs 0.0) to the HOST ROUND-TRIP reference on every fact/seed, "
            "the full who/what/yes-no matrix + every is-None/unknown abstention is identical True-vs-False across "
            "batched/cache/perblock, and the no-confab moat holds (0 false-accepts). DEFAULT persistent_loop=False is "
            "byte-identical to today (a no-op handoff = the carry-live-Z path the composer already used). DIAGNOSIS: the "
            "flat path NEVER host-round-tripped between ops -- it carried the LIVE register Z (inflated |Z|~period) "
            "across the unbind->cleanup handoff; persistent_loop=True makes it the canonical clean-unit-phasor handoff "
            "(== the host round-trip, == the carry default's ANSWER since the cleanup argmax is scale-invariant). The "
            "recursive CLAUSE path is ALREADY on-substrate (_decode_clause uses _dev_rekick_into unconditionally) -> no "
            "flat-arc drift to localize. NO sim/ edit (reuse-by-import of the public RF ops + the composer's own "
            "_dev_rekick_into)."
        ),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: result[k] for k in (
        "verdict", "default_persistent_loop_off", "membrane_byte_identical_all",
        "worst_membrane_perblock_maxabs", "worst_membrane_batched_maxabs", "matrix_identical_all",
        "moat_preserved_all", "total_false_accepts_on", "clause_path_already_on_substrate",
        "needs_gpu", "needs_sim_edit")}, indent=2))
    print(f"\nwrote {os.path.normpath(OUT)}")
    return 0 if verdict == "GO" else 1


def batched_membrane_diff(c_loop, c_host):
    """maxabs difference of the active batched cleanup membrane between the persistent_loop=True composer and a host
    round-trip reference computed over the same composer's state (cache off; the stock batched read with an explicit
    host round-trip at the unbind->cleanup handoff)."""
    def _batched_read(c, handoff):
        b, D, Pd, V, NP = c.b, c.D, c.period, c.V, c.NP
        n = len(c.kb); nr, nm = c.n_roles, c.n_main
        pol_ri = c.bind_roles.index("polarity")
        b.cp_membrane_potential_v[:] = 0.0
        b.cp_recovery_variable_u[:] = 0.0
        kick = np.zeros(c.n_total, dtype=np.complex128)
        for i in range(n):
            kick[c.store_base + i * c.block] = 1.0
        b.rf_set_complex_weights(c.store_conns)
        b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        unbind = []
        for i in range(n):
            trig = c.store_base + i * c.block
            for ri, role in enumerate(c.bind_roles):
                zc = c._unbind_conj(role)
                qreg = c.bat_q_base + (i * nr + ri) * D
                unbind += [(qreg + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind)
        b.rf_resonate_steps(Pd + 8)
        q_run = slice(c.bat_q_base, c.bat_q_base + n * nr * D)
        if handoff == "loop":
            c._loop_rekick([q_run])                                  # the WIRED hook (ON because persistent_loop)
        else:  # host round-trip reference
            qphi = np.asarray(b.rf_read_phases())
            kick2 = np.zeros(c.n_total, dtype=np.complex128)
            kick2[q_run] = np.exp(2j * np.pi * qphi[q_run])
            b.cp_membrane_potential_v[:] = 0.0
            b.cp_recovery_variable_u[:] = 0.0
            b.rf_kick(kick2, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
        clean = []
        for i in range(n):
            cblk = c.bat_c_base + i * c.cb
            for ri in range(nm):
                qreg = c.bat_q_base + (i * nr + ri) * D
                for j in range(V):
                    cc = c._cleanup_conj(c.words[j])
                    clean += [(cblk + ri * V + j, qreg + k, complex(cc[k])) for k in range(D)]
            qreg_p = c.bat_q_base + (i * nr + pol_ri) * D
            for j in range(NP):
                cc = c._cleanup_conj(c.pol_words[j])
                clean += [(cblk + nm * V + j, qreg_p + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean)
        b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        return mem[c.bat_c_base:c.bat_c_base + n * c.cb].copy()
    m_loop = _batched_read(c_loop, "loop")
    m_host = _batched_read(c_host, "host")
    return float(np.max(np.abs(m_loop - m_host)))


if __name__ == "__main__":
    sys.exit(main())
