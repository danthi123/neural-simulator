"""Cheap-first DE-RISK PoC (2026-06-19): prove the cleanup/unbind weight CSRs are QUERY-INVARIANT and can be
built ONCE + reused, and that doing so is ANSWER-IDENTICAL while collapsing the dominant per-op cost.

Mechanism: in _read_all_blocks, the `unbind` and `clean` connection sets depend only on (n_facts, vocab, block
layout) -- NOT on the stored fact content (which lives in store_conns). So for a fixed store size they are the
SAME every query. Cache the built CSRs (cp_rf_w_re/im for each phase) keyed by n_facts; install by direct
assignment instead of rebuilding from a fresh tuple list.

This PoC monkeypatches a cached _read_all_blocks onto a OneBrainComposer and checks (a) answer-identity vs the
stock read across the full who/what matrix and (b) the speedup.

Run:  SIM_BACKEND=cupy python -u -m research.runners._latency_cache_poc
"""
from __future__ import annotations
import time, types
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp

from research.runners.one_brain_composer import OneBrainComposer, ROLES3
from sim.backend import to_host, is_gpu_backend


def sync():
    cp.cuda.Stream.null.synchronize()


def med(fn, n=7, warmup=3):
    for _ in range(warmup):
        fn(); sync()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def _csr_from_conns(n, connections):
    m = len(connections)
    rows = np.fromiter((int(p) for (p, q, w) in connections), dtype=np.int32, count=m)
    cols = np.fromiter((int(q) for (p, q, w) in connections), dtype=np.int32, count=m)
    wre = np.fromiter((float(complex(w).real) for (p, q, w) in connections), dtype=np.float64, count=m)
    wim = np.fromiter((float(complex(w).imag) for (p, q, w) in connections), dtype=np.float64, count=m)
    r = cp.asarray(rows); c = cp.asarray(cols)
    Wre = csp.csr_matrix((cp.asarray(wre), (r, c)), shape=(n, n))
    Wim = csp.csr_matrix((cp.asarray(wim), (r, c)), shape=(n, n))
    return Wre, Wim


def make_cached_read(comp):
    """Build a cached variant of _read_all_blocks: the unbind + clean CSRs are built once per n_facts and reused."""
    cache = {}

    def cached_read_all_blocks(self):
        b, D, Pd, V, NP = self.b, self.D, self.period, self.V, self.NP
        n = len(self.kb)
        if n == 0:
            return []
        roles = ROLES3 + ["polarity"]
        key = n
        if key not in cache:
            # build unbind + clean conn lists ONCE, convert to CSR, stash
            unbind = []
            for i in range(n):
                trig = self.store_base + i * self.block
                for ri, role in enumerate(roles):
                    zc = np.conj(self.comp._to_phasor(self.comp.roles[role]))
                    qreg = self.bat_q_base + (i * 4 + ri) * D
                    unbind += [(qreg + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
            clean = []
            for i in range(n):
                cblk = self.bat_c_base + i * self.cb
                for ri in range(3):
                    qreg = self.bat_q_base + (i * 4 + ri) * D
                    for j in range(V):
                        cc = np.conj(self.comp._to_phasor(self.comp.concepts[self.words[j]]))
                        clean += [(cblk + ri * V + j, qreg + k, complex(cc[k])) for k in range(D)]
                qreg_p = self.bat_q_base + (i * 4 + 3) * D
                for j in range(NP):
                    cc = np.conj(self.comp._to_phasor(self.comp.concepts[self.pol_words[j]]))
                    clean += [(cblk + 3 * V + j, qreg_p + k, complex(cc[k])) for k in range(D)]
            cache[key] = (_csr_from_conns(self.n_total, unbind), _csr_from_conns(self.n_total, clean))
        (Ure, Uim), (Cre, Cim) = cache[key]

        # store_conns CSR must still be (re)built when facts change; build it fresh here (it's small: n*D)
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(n):
            kick[self.store_base + i * self.block] = 1.0
        b.rf_set_complex_weights(self.store_conns)
        b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        # unbind: install cached CSR
        b.cp_rf_w_re, b.cp_rf_w_im = Ure, Uim
        b.rf_resonate_steps(Pd + 8)
        # clean: install cached CSR
        b.cp_rf_w_re, b.cp_rf_w_im = Cre, Cim
        b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        out = []
        for i in range(n):
            cblk = self.bat_c_base + i * self.cb
            sa = np.maximum(mem[cblk + 0 * V:cblk + 1 * V], 0.0)
            sv = np.maximum(mem[cblk + 1 * V:cblk + 2 * V], 0.0)
            row = [self.words[int(np.argmax(sa))], self.words[int(np.argmax(sv))],
                   self.words[int(np.argmax(np.maximum(mem[cblk + 2 * V:cblk + 3 * V], 0.0)))]]
            ps = np.maximum(mem[cblk + 3 * V:cblk + 3 * V + NP], 0.0)
            row.append(self.pol_words[int(np.argmax(ps))])
            out.append(tuple(row))
        return out
    return cached_read_all_blocks


def main():
    assert is_gpu_backend()
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    SEED, D, K = 42, 128, 16
    agents = ["dog", "cat", "apple", "river"]; actions = ["go", "run", "come", "stop"]; pats = ["north", "south", "east", "west"]

    # stock composer
    comp = OneBrainComposer(seed=SEED, D=D, k_max=32, enable_rf_cudagraph=True, enable_batched=True)
    for i in range(K):
        comp.store(agents[i % 4], actions[(i // 4) % 4], pats[i % 4])
    sync()

    stock = comp._read_all_blocks()
    # install cached read
    cached_fn = make_cached_read(comp)
    comp._read_all_blocks_cached = types.MethodType(cached_fn, comp)
    cached = comp._read_all_blocks_cached()

    # ---- answer-identity ----
    ok = (stock == cached)
    print(f"\n  ANSWER-IDENTITY (stock vs cached batched read, all {K} blocks): {'IDENTICAL' if ok else 'MISMATCH'}")
    if not ok:
        for i, (s, c) in enumerate(zip(stock, cached)):
            if s != c:
                print(f"    block {i}: stock={s} cached={c}")

    # full query answers across the matrix
    mism = 0
    for i in range(K):
        a, v, p = agents[i % 4], actions[(i // 4) % 4], pats[i % 4]
        # stock query
        q_stock = comp.query_patient(a, v)
        comp._orig = comp._read_all_blocks
        comp._read_all_blocks = comp._read_all_blocks_cached
        q_cached = comp.query_patient(a, v)
        comp._read_all_blocks = comp._orig
        if q_stock != q_cached:
            mism += 1; print(f"    query({a},{v}): stock={q_stock} cached={q_cached}")
    print(f"  query_patient answer-identity across {K} cues: {'ALL IDENTICAL' if mism == 0 else f'{mism} MISMATCH'}")

    # ---- speedup ----
    t_stock = med(lambda: comp._read_all_blocks(), n=7, warmup=3)
    t_cached = med(lambda: comp._read_all_blocks_cached(), n=7, warmup=3)
    print(f"\n  _read_all_blocks: stock {t_stock:6.1f} ms | cached {t_cached:6.1f} ms | speedup {t_stock/t_cached:.1f}x")


if __name__ == "__main__":
    main()
