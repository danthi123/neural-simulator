"""MULTI-SLOT variable-binding WORKING MEMORY — hold >=2 role-filler bindings SIMULTANEOUSLY across novel fillers.

THE QUESTION (scales the single-slot WM GO toward real language). Conversation needs holding MULTIPLE variables at
once: a SUBJECT *and* an OBJECT (or a small stack), each agreeing with its own verb across an arbitrary intervening
span. The single-slot WM GO
  `2026-08-11-variable-binding-working-memory-gated-slot-surpasses-HTM-heldout-1.000-vs-0.000-6seed-GO.md`
latches ONE variable (held-out 1.000) with a LOAD-BEARING spiking hold (lesion-the-hold collapses it). This de-risk asks:
does the SAME composition hold k>=2 bindings WITHOUT CROSS-TALK — each variable recovered correctly, generalising to
novel fillers — and at what k does it BREAK (the capacity ceiling)? Compare 1-slot (known GO) -> 2 -> 3 -> ...

OUR OWN RECORD FIRST (read, cited, complemented — NOT re-derived):
  * The ~2 cap is a SUPERPOSITION-SNR artifact of storing all binds in ONE shared register (EDGE-5 4-rung refutation;
    `2026-05-12-cumulative-binding-fixed-capacity.md`; `2026-06-05-B-innetwork-superposition-NEGATIVE.md`). A real brain
    ALLOCATES a distinct near-orthogonal SLOT per bind, so capacity converts from SNR-limited (~2) to slot-count-limited.
  * The gap#2 KEYSTONE arc (`2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO.md`) already showed, on
    `build_persistent_slot`: (step 2a) multi-slot COEXISTENCE is GO — P=3 pools coexist via NMDA persistence, no-recur
    collapses 3/3->0/3; (retrieval) a competitive-slot LTM store recovers P=2=1.00 / P=3=1.00 / P=4=0.79 vs the shared
    store's ~2 cap. BUT that GO is an LTM STORE (plastic slot->filler, read by RE-DRIVING; its no-recur was 1.00, i.e. the
    NMDA hold was NOT load-bearing). THIS de-risk is the complementary, un-done piece: multi-slot as WORKING MEMORY where
    the D3 NMDA HOLD is LOAD-BEARING (read the HELD bump, zero input), on the HELD-OUT AGREEMENT STREAM, with per-slot
    recovery + a cross-talk control + the memory teeth biting PER SLOT.

THE MECHANISM (reuse-by-import of three banked GOs; NO sim/ edit):
  * HOLD  = R banks of the D3 slow-NMDA persistent-activity slot on ONE bridge with ONE SHARED FS pool
            (`build_persistent_slot(seed, K=R*n_slot)`). Register r occupies pools [r*n_slot, (r+1)*n_slot); all registers
            SHARE the single FS inhibitory pool, so they compete for the same inhibitory resource -> genuine capacity
            pressure (this is exactly the keystone step-2a coexistence substrate, the faithful multi-item bump-attractor
            WM: separate assemblies, shared inhibition; NOT R fully-separate brains).
  * BIND  = the RUNG6c content-agnostic Hebbian fast-weight binder (`HebbianBinder`): each fixed entity binds to a stable
            LOCAL slot index in [0, n_slot); writing entity e into register r drives global pool r*n_slot + slot(e). Read
            register r = argmax over its n_slot pools -> local slot -> deref -> entity -> its agreeing verb.
  * GATE  = a role-by-position MARKER (subject->reg0, object->reg1, ...). This is the SCAFFOLD (gate timing/role is a host
            marker), the SAME named residual as the single-slot GO: the LEARNED spiking role-gate is gap#4 territory
            (`739a8867` established even a host position-ORACLE fails to induce role at 6 seeds -> the residual is CREDIT
            ASSIGNMENT). The MEMORY composition is what this de-risk tests; the multi-gate is NOT re-opened here.

THE TASK — a two-(k-)dependency agreement clause: [e_0]+[L fillers]+[e_1]+[L fillers]+...+[e_{R-1}]+[L fillers]+[v_0..v_{R-1}]
  R distinct entities (roles), each with its own agreeing verb; the R verbs must each agree with the CORRECT bound
  variable. Held-out TEST = disjoint NOVEL filler tuples (the fillers never touch a slot, so a latched binding is carried
  invariantly -> the exact property multi-variable language needs). The subject (reg0) is held the LONGEST (across every
  later load + filler span), so reg0 is the durability stress.

ANTI-CHEATS (all EXECUTE; teeth; each single-variable):
  (1) PER-SLOT held-out recovery — report each register's own-entity verb recovery AND all-R-correct (not just one).
  (2) LESION-THE-HOLD (recur=0) -> every bump dies over the span -> recovery collapses PER SLOT. THE WM tooth (this is the
      load-bearing difference from the keystone LTM store, whose no-recur was 1.00).
  (3) SUPERPOSED-SINGLE-SLOT collide baseline -> cram all k bindings into ONE register/bank (no clear, superpose): the
      1-of-K attractor holds ONE winner -> all-correct = 0 by construction, per-item ~1/k. THE LOAD-BEARING control: it
      reproduces the ~2-cap collision and proves the multi-register SEPARATION is what carries >=2 bindings.
  (4) CROSS-TALK read — querying register A must not return register B's filler. With disjoint banks a literal filler-swap
      is impossible by construction (A reads only A's pools), so cross-talk here manifests as SUPPRESSION (a register loses
      its bump under shared inhibition). Report BOTH: the filler-swap rate (must be ~0) AND the collapse/interference rate
      (a register silent / wrong) as k grows -- the honest capacity signal.
  (5) REFERENT-SHUFFLE -> derange each role's entity->verb readout -> ~chance. No topic->answer leakage, per slot.
  (6) HOLD-NOT-RE-READ -> external input ASSERTED zero across the whole hold+read span (the slot SUSTAINS; the read is of
      the HELD bump, not a re-drive). A per-arm assertion.
  (7) NOVEL fillers (held-out disjoint tuples) + emit backend/device.

GO (smoke = 1-seed indicator; decisive = 6-seed): at k=2 the MARKER-gated multi-slot WM recovers BOTH bindings on
held-out novel fillers (all-correct >= 0.85 AND each per-slot >= 0.85 AND >= chance+0.20), the hold is load-bearing
(lesion collapses >= 0.30 below), the SUPERPOSED-single-slot baseline collides (all-correct <= chance+0.10), cross-talk
filler-swap ~0, referent-shuffle ~chance, hold-alive > 0 with input asserted zero. THEN report k=1..K_max and the
CAPACITY CEILING (largest k with all-correct >= 0.80). HONEST NEGATIVE first-class: if it collapses at 2 (cross-talk /
suppression), report the interference numbers + name the fix (more disjoint slot allocation via the hetero-LTD allocation
lane, or independent attractor pools / per-register FS).

Reuse-by-import; NO sim/ edit. SIM_BACKEND=numpy (sub-1k-neuron D3 loops are launch-bound: CPU faster).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._multi_slot_binding_derisk --seeds 42 --ks 1 2 3 4
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._multi_slot_binding_derisk --seeds 42 43 44 100 101 102 \
    --ks 1 2 3 4 5 --n-test 60 --out research/findings/raw/_multi_slot_binding/multi_slot_6seed.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

# --- the agreement-stream vocab layout (like-for-like with the single-slot GO) ---
from research.runners._emerge_stream_language_derisk import vocab_layout
# --- the VERIFIED RUNG6c content-agnostic Hebbian fast-weight binder + barcode mint ---
from research.runners._novel_referent_hebbian_fastweight_derisk import HebbianBinder, _mint_codes, _DIM, _K as _BINDER_K
# --- the VERIFIED D3 spiking slow-NMDA persistent-activity HOLD slot ---
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset

try:
    from tools.lab import lever, attributable_to, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond: print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_multi_slot_binding/multi_slot.json")


# ====================================================================================================================
# The MULTI-SLOT spiking HOLD: R banks of D3 slow-NMDA pools on ONE bridge sharing ONE FS inhibitory pool.
# WRITE(reg, pool) = LOAD (clean protocol writes each register ONCE, no clear); HOLD = zero external drive (self-sustain);
# READ(reg) = argmax pool rate WITHIN that register, under zero input. All assert input discipline.
# ====================================================================================================================
class MultiSlotHold:
    def __init__(self, seed, R, n_slot, recur=25.0, load_steps=30, hold_steps=18, read_steps=18, clear_steps=200,
                 input_gain=400.0, clear_gain=1500.0, nmda=True, shared=None):
        from sim.backend import to_host, from_host
        self._to_host, self._from_host = to_host, from_host
        self.R, self.n_slot = R, n_slot
        self.K = R * n_slot                                       # total attractor pools (R banks of n_slot)
        # ONE-BRAIN MERGE (opt-in, byte-identical when shared is None): when a MergedPool is injected, the K slot
        # pools + shared FS are this organ's region SLICE of the SHARED spiking bridge (already built + wired by the
        # pool's per-region-seamed wiring inject) instead of its own bridge. The reads (write/hold/read) then run on
        # the shared bridge's slice; a co-resident organ is protected by the pool's read_isolation. See
        # research/runners/onebrain_merge_framework.py.
        self._shared = shared
        if shared is not None:
            shared.ensure_built()
            self.sb = shared.bridge
        else:
            self.sb = build_persistent_slot(seed, self.K, recur=recur, nmda=nmda)
        self.idx = _pool_idx(self.sb, self.K)                     # global pool index -> neuron indices
        self.fs_idx = np.asarray(list(self.sb.region_manager.indices("fs")), dtype=int)
        self.n = self.sb.core_config.num_neurons
        self.load_steps, self.hold_steps, self.read_steps = load_steps, hold_steps, read_steps
        self.clear_steps, self.input_gain, self.clear_gain = clear_steps, input_gain, clear_gain
        self._loaded = [False] * self.K                          # per-pool: is a bump latched there?
        self._zero_input_span = True                             # anti-cheat #6: input identically zero across hold+read?

    def _gp(self, reg, local):
        return reg * self.n_slot + local                        # global pool for (register, local slot)

    def reset(self):
        _reset(self.sb); self._loaded = [False] * self.K; self._zero_input_span = True

    def _run(self, cur_vec, steps, assert_zero=False):
        rates = np.zeros(self.K)
        dev = self._from_host(cur_vec)
        if assert_zero and np.asarray(cur_vec).any():
            self._zero_input_span = False
        for _ in range(steps):
            self.sb.cp_external_input_current[:] = dev
            self.sb._run_one_simulation_step()
            fir = np.asarray(self._to_host(self.sb.cp_firing_states)).astype(float)
            for k in range(self.K):
                rates[k] += fir[self.idx[k]].mean()
        return rates / max(steps, 1)

    def write(self, reg, local, superpose=False):
        """LOAD (register, local slot). A clean multi-slot protocol writes each register ONCE => first write, no clear.
        superpose=True forces a within-bank re-write WITHOUT clearing (the collide baseline: two bumps in one bank)."""
        gp = self._gp(reg, local)
        if (not superpose) and self._loaded[gp] and self.clear_steps > 0:   # overwrite the SAME pool: clear > tau_NMDA
            cc = np.zeros(self.n); cc[self.fs_idx] = self.clear_gain
            self._run(cc, self.clear_steps)
        cur = np.zeros(self.n); cur[self.idx[gp]] = self.input_gain
        self._run(cur, self.load_steps)
        self._loaded[gp] = True

    def hold(self, steps=None):
        self._run(np.zeros(self.n), steps if steps is not None else self.hold_steps, assert_zero=True)

    def read(self, reg):
        """READ register `reg`: argmax pool rate WITHIN [reg*n_slot, (reg+1)*n_slot), under zero input (the HELD bump)."""
        rates = self._run(np.zeros(self.n), self.read_steps, assert_zero=True)
        band = rates[reg * self.n_slot:(reg + 1) * self.n_slot]
        alive = float(band.max())
        local = int(np.argmax(band)) if alive > 1e-6 else -1
        return local, alive

    def probe_occupancy(self, steps=None):
        """A genuine zero-input read of EVERY register's current band-max activity in ONE pass (the SAME
        read-out instrument class as `read()`, generalised across all R registers) -- added 2026-09-01 to let a
        CALLER route a new write to the register the SUBSTRATE ITSELF currently shows as free/least-active,
        instead of a host-assigned position. External input is ASSERTED zero across the probe span (a HOLD
        read, not a re-drive); a register that has never been written sits at genuine baseline (this substrate
        has no background OU noise -- `ou_std_current_pA=0` in `build_persistent_slot` -- so an untouched
        register reads EXACTLY 0.0 and a written+held one reads its live sustained rate: a real occupancy
        signal, not a formula). Purely additive -- `read`/`write`/`hold` are unchanged."""
        steps = self.hold_steps if steps is None else steps
        rates = self._run(np.zeros(self.n), steps, assert_zero=True)
        return np.array([rates[r * self.n_slot:(r + 1) * self.n_slot].max() for r in range(self.R)])


# ====================================================================================================================
# Stream: a k-role agreement clause with disjoint NOVEL held-out filler tuples
# ====================================================================================================================
def build_codebook(n_ent, n_fill, rng):
    """Sparse developmental-random barcodes for the WRITEABLE tokens (entities + fillers)."""
    subj, fill, verb, V = vocab_layout(n_ent, n_fill)
    codes = _mint_codes(rng, n_ent + n_fill)
    code_of = {}
    for i, tok in enumerate(list(subj) + list(fill)):
        code_of[tok] = codes[i]
    return code_of, (subj, fill, verb, V)


def persistent_entity_binder(entity_tokens, code_of, n_slot):
    """One RUNG6c HebbianBinder: bind each fixed entity to a stable LOCAL slot in [0, n_slot). Shared across all
    registers (each register uses the SAME local-slot codebook). Returns (slot_of_ent, ent_of_slot)."""
    binder = HebbianBinder()
    slot_of_ent, ent_of_slot = {}, {}
    for tok in entity_tokens:
        s = binder.slot(code_of[tok]); slot_of_ent[int(tok)] = s; ent_of_slot[s] = int(tok)
    return slot_of_ent, ent_of_slot


def make_multi_stream(n_ent, n_fill, L, R, n_sent, rng, exclude=None):
    """A k-role agreement clause: for each role r, an entity token then L i.i.d. fillers; then R verb targets.
    Entities per clause are DISTINCT (so cross-talk is detectable). Returns (clauses, novel_count). Each clause =
    dict(ents=[e_0..], verbs=[v_0..], token_roles=[(token, role_or_None)...], ftuple)."""
    subj, fill, verb, V = vocab_layout(n_ent, n_fill)
    exclude = exclude or set()
    clauses, novel = [], 0
    for _ in range(n_sent):
        ents = [int(x) for x in rng.choice(n_ent, size=R, replace=False)]
        spans = [tuple(int(fill[rng.integers(n_fill)]) for _ in range(L)) for _ in range(R)]
        ftuple = tuple(x for span in spans for x in span)
        token_roles = []
        for r in range(R):
            token_roles.append((int(subj[ents[r]]), r))                    # the role/entity token -> LOAD into reg r
            for f in spans[r]:
                token_roles.append((int(f), None))                         # a filler -> HOLD
        clauses.append({"ents": ents, "verbs": [int(verb[e]) for e in ents], "token_roles": token_roles,
                        "ftuple": ftuple})
        novel += int(ftuple not in exclude)
    return clauses, novel


# ====================================================================================================================
# Evaluate the multi-slot WM over a clause set (marker gate: role-by-position)
# ====================================================================================================================
def eval_multi_wm(slot: MultiSlotHold, clauses, R, slot_of_ent, ent_of_slot, verb_of, n_slot):
    """Returns a dict of per-slot + joint metrics. gate = role-by-position marker (a role token -> LOAD its register;
    a filler -> HOLD all registers)."""
    per_slot_ok = np.zeros(R); all_ok = 0; alive_reg = np.zeros(R)
    filler_swap = 0; collapse = 0; zero_ok = True; n = len(clauses)
    for c in clauses:
        ents = c["ents"]; true_verbs = c["verbs"]
        slot.reset()
        for tok, role in c["token_roles"]:
            if role is not None:
                slot.write(role, slot_of_ent[tok])                          # LOAD entity into its register
            else:
                slot.hold()                                                 # a filler -> HOLD (all registers self-sustain)
        reads = []
        for r in range(R):
            local, alive = slot.read(r); reads.append(local); alive_reg[r] += alive
            if alive <= 1e-3:
                collapse += 1                                               # register lost its bump (suppression)
        zero_ok = zero_ok and slot._zero_input_span
        pred_ents = [ent_of_slot.get(reads[r], -1) for r in range(R)]
        clause_all = True
        for r in range(R):
            pred_verb = verb_of.get(pred_ents[r], -1)
            ok = int(pred_verb == true_verbs[r]); per_slot_ok[r] += ok
            clause_all = clause_all and bool(ok)
            # filler-swap cross-talk: register r returned a DIFFERENT role's entity (contamination)
            if pred_ents[r] in ents and pred_ents[r] != ents[r]:
                filler_swap += 1
        all_ok += int(clause_all)
    return {"per_slot": (per_slot_ok / max(1, n)).tolist(), "per_slot_mean": float(per_slot_ok.mean() / max(1, n)),
            "all_correct": all_ok / max(1, n), "alive_reg": (alive_reg / max(1, n)).tolist(),
            "filler_swap_rate": filler_swap / max(1, n * R), "collapse_rate": collapse / max(1, n * R),
            "zero_input_ok": bool(zero_ok)}


def eval_superposed_single(slot1: MultiSlotHold, clauses, R, slot_of_ent, ent_of_slot, verb_of):
    """COLLIDE baseline: cram all R bindings into ONE register/bank (superpose, no clear). The 1-of-K attractor holds ONE
    winner -> all-correct = 0 by construction, per-item ~1/R. Proves the multi-register SEPARATION is load-bearing."""
    per_item_ok = 0.0; all_ok = 0; n = len(clauses)
    for c in clauses:
        ents = c["ents"]; true_verbs = c["verbs"]
        slot1.reset()
        # write every role's entity into register 0 (superpose), interleaving the holds as in the real clause
        for tok, role in c["token_roles"]:
            if role is not None:
                slot1.write(0, slot_of_ent[tok], superpose=True)            # ALL into bank 0 -> they collide
            else:
                slot1.hold()
        local, alive = slot1.read(0)
        pred_ent = ent_of_slot.get(local, -1)
        got = 0
        for r in range(R):
            pred_verb = verb_of.get(pred_ent, -1)
            got += int(pred_verb == true_verbs[r])
        per_item_ok += got / R
        all_ok += int(got == R)
    return {"per_item": per_item_ok / max(1, n), "all_correct": all_ok / max(1, n)}


# ====================================================================================================================
# One (seed, k) point
# ====================================================================================================================
def run_point(seed, k, n_ent, n_slot, n_fill, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps):
    rng = np.random.default_rng(seed)
    chance = 1.0 / n_ent
    subj, fill, verb, V = vocab_layout(n_ent, n_fill)
    verb_of = {int(subj[i]): int(verb[i]) for i in range(n_ent)}

    code_of, _ = build_codebook(n_ent, n_fill, np.random.default_rng(seed + 7))
    slot_of_ent, ent_of_slot = persistent_entity_binder([int(s) for s in subj], code_of, n_slot)

    # train (for held-out exclusion only; the WM has NO filler-tuned params -> novel-filler generalisation is structural)
    train_clauses, _ = make_multi_stream(n_ent, n_fill, L, k, n_train, rng)
    train_ftuples = set(c["ftuple"] for c in train_clauses)
    test_clauses, novel = make_multi_stream(n_ent, n_fill, L, k, n_test, rng, exclude=train_ftuples)
    # keep only genuinely-novel held-out clauses; gen_defined if enough novel paths exist
    test_novel = [c for c in test_clauses if c["ftuple"] not in train_ftuples]
    path_space = float(n_fill) ** (L * k)
    gen_defined = len(test_novel) >= max(20, 3 * n_ent) and path_space >= 64
    eval_clauses = test_novel if gen_defined else test_clauses

    def _slot(rc=recur):
        return MultiSlotHold(seed, k, n_slot, recur=rc, hold_steps=hold_steps, load_steps=load_steps,
                             clear_steps=clear_steps)

    # (A) MARKER-gated multi-slot WM on held-out novel fillers  == the headline
    m = eval_multi_wm(_slot(), eval_clauses, k, slot_of_ent, ent_of_slot, verb_of, n_slot)

    # (2) LESION-THE-HOLD: recur=0 -> bumps die -> collapse (the WM tooth; load-bearing hold)
    les = eval_multi_wm(_slot(rc=0.0), eval_clauses, k, slot_of_ent, ent_of_slot, verb_of, n_slot)

    # (3) SUPERPOSED-SINGLE-SLOT collide baseline (k>=2 only)
    sup = None
    if k >= 2:
        sup_slot = MultiSlotHold(seed, 1, n_slot, recur=recur, hold_steps=hold_steps, load_steps=load_steps,
                                 clear_steps=clear_steps)
        sup = eval_superposed_single(sup_slot, eval_clauses, k, slot_of_ent, ent_of_slot, verb_of)

    # (5) REFERENT-SHUFFLE: derange each role's entity->verb readout -> ~chance (no topic->answer leakage)
    order = list(range(n_ent)); dr = np.random.default_rng(seed + 13)
    for _ in range(64):
        dr.shuffle(order)
        if all(order[i] != i for i in range(n_ent)):
            break
    verb_of_shuf = {int(subj[i]): int(verb[order[i]]) for i in range(n_ent)}
    ref = eval_multi_wm(_slot(), eval_clauses, k, slot_of_ent, ent_of_slot, verb_of_shuf, n_slot)

    return {"seed": seed, "k": k, "chance": chance, "gen_defined": bool(gen_defined), "path_space": path_space,
            "n_eval": len(eval_clauses), "n_novel": len(test_novel),
            "acc_all": m["all_correct"], "per_slot": m["per_slot"], "per_slot_mean": m["per_slot_mean"],
            "alive_reg": m["alive_reg"], "filler_swap_rate": m["filler_swap_rate"],
            "collapse_rate": m["collapse_rate"], "zero_input_ok": m["zero_input_ok"],
            "lesion_all": les["all_correct"], "lesion_per_slot_mean": les["per_slot_mean"],
            "superposed_all": None if sup is None else sup["all_correct"],
            "superposed_per_item": None if sup is None else sup["per_item"],
            "refshuf_all": ref["all_correct"], "refshuf_per_slot_mean": ref["per_slot_mean"]}


def agg(per):
    keys = ["acc_all", "per_slot_mean", "filler_swap_rate", "collapse_rate", "lesion_all", "lesion_per_slot_mean",
            "refshuf_all", "refshuf_per_slot_mean"]
    a = {kk: float(np.mean([p[kk] for p in per])) for kk in keys}
    sup = [p["superposed_all"] for p in per if p["superposed_all"] is not None]
    supi = [p["superposed_per_item"] for p in per if p["superposed_per_item"] is not None]
    a["superposed_all"] = float(np.mean(sup)) if sup else None
    a["superposed_per_item"] = float(np.mean(supi)) if supi else None
    a["per_slot"] = np.mean([p["per_slot"] for p in per], axis=0).round(3).tolist()
    a["alive_reg"] = np.mean([p["alive_reg"] for p in per], axis=0).round(4).tolist()
    a.update({"k": per[0]["k"], "chance": per[0]["chance"], "path_space": per[0]["path_space"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3], help="# concurrent bindings (roles) to sweep")
    ap.add_argument("--n-ent", type=int, default=6, help="entity vocab (= # agreeing verbs; chance=1/n_ent); >= max(ks)")
    ap.add_argument("--n-slot", type=int, default=6, help="pools per register bank (>= n_ent)")
    ap.add_argument("--n-fill", type=int, default=10)
    ap.add_argument("--distance", type=int, default=2, help="filler-span L per role")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--go-k", type=int, default=2, help="k at which to evaluate the GO gate")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    chance = 1.0 / a.n_ent
    ks = sorted(set(a.ks))
    if max(ks) > a.n_ent:
        print(f"NOT-RUNNABLE: max(ks)={max(ks)} > n_ent={a.n_ent} (need distinct entities per role)"); return 2
    if a.n_slot < a.n_ent:
        print(f"NOT-RUNNABLE: n_slot={a.n_slot} < n_ent={a.n_ent} (each entity needs a distinct local slot)"); return 2
    if a.n_ent > _BINDER_K:
        print(f"NOT-RUNNABLE: n_ent={a.n_ent} > RUNG6c HebbianBinder _K={_BINDER_K} (entities would COLLIDE onto the "
              f"last slot -> a binder-capacity confound, NOT a WM limit). Raise the binder's _K to probe higher k."); return 2
    print(f"backend={backend} device={device} | n_ent={a.n_ent} chance={chance:.3f} | n_slot={a.n_slot} n_fill={a.n_fill} "
          f"L={a.distance} | recur={a.recur} hold={a.hold_steps} load={a.load_steps} | n_test={a.n_test} | ks={ks} "
          f"seeds={a.seeds}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for k in ks:
            per = [run_point(s, k, a.n_ent, a.n_slot, a.n_fill, a.distance, a.n_train, a.n_test, a.recur,
                             a.hold_steps, a.load_steps, a.clear_steps) for s in a.seeds]
            p = agg(per); points.append(p)
            sup = "n/a" if p["superposed_all"] is None else f"{p['superposed_all']:.3f}(item {p['superposed_per_item']:.3f})"
            print(f"  [k={k} paths={p['path_space']:.0f} gen={p['gen_defined']} n_eval={per[0]['n_eval']}] "
                  f"ALL-correct {p['acc_all']:.3f} | per-slot {p['per_slot']} (mean {p['per_slot_mean']:.3f}) | "
                  f"alive {p['alive_reg']} zero-in {p['zero_input_ok']} || LESION-hold all {p['lesion_all']:.3f} "
                  f"(slot {p['lesion_per_slot_mean']:.3f}) | SUPERPOSED-1slot all {sup} | filler-swap "
                  f"{p['filler_swap_rate']:.3f} | collapse {p['collapse_rate']:.3f} | REF-SHUF all {p['refshuf_all']:.3f} "
                  f"|| chance {chance:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    # capacity ceiling: largest k with all-correct >= 0.80 (contiguous from k=1)
    ceiling = 0
    for p in sorted(points, key=lambda x: x["k"]):
        if p["acc_all"] >= 0.80 and p["k"] == ceiling + 1:
            ceiling = p["k"]
        else:
            break

    go_p = next((p for p in points if p["k"] == a.go_k), None)
    verdict = None
    if err is None and go_p is not None:
        print(f"\n-- GO gate + anti-cheats at k={a.go_k} (held-out novel fillers) --", flush=True)
        void_if(not go_p["gen_defined"], "too few held-out novel clauses (raise --n-test) or path space too small "
                "(raise --n-fill/--distance) -> generalisation UNDEFINED")
        lever("multi-slot ALL-correct vs LESION-the-hold (recurrence load-bearing)", round(go_p["lesion_all"], 3),
              round(go_p["acc_all"], 3), required=False)
        if go_p["superposed_all"] is not None:
            lever("multi-slot ALL-correct vs SUPERPOSED-single-slot (separation load-bearing)",
                  round(go_p["superposed_all"], 3), round(go_p["acc_all"], 3), required=False)
        attributable_to("multi-slot per-slot mean over the referent-shuffle control", go_p["per_slot_mean"],
                        go_p["refshuf_per_slot_mean"])

        gen = go_p["gen_defined"]
        headline = go_p["acc_all"] >= 0.85 and min(go_p["per_slot"]) >= 0.85 and go_p["per_slot_mean"] >= chance + 0.20
        recurrence = go_p["acc_all"] >= go_p["lesion_all"] + 0.30
        separation = (go_p["superposed_all"] is None) or (go_p["superposed_all"] <= chance + 0.10)
        no_swap = go_p["filler_swap_rate"] <= 0.05
        no_leak = go_p["refshuf_per_slot_mean"] <= chance + 0.10
        held_alive = min(go_p["alive_reg"]) > 1e-3 and go_p["zero_input_ok"]
        core = bool(gen and headline and recurrence and separation and no_swap and no_leak and held_alive)
        go = bool(core and not smoke)

        if not gen:
            verdict = (f"INCONCLUSIVE — k={a.go_k} path space {go_p['path_space']:.0f} too small to hold out novel "
                       f"fillers; raise n_fill/L.")
        elif core:
            tag = "GO" if go else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — the multi-slot spiking WM (R banks of D3 slow-NMDA pools on ONE bridge, ONE shared FS) "
                       f"HOLDS k={a.go_k} role-filler bindings SIMULTANEOUSLY and carries each across NOVEL fillers: "
                       f"held-out ALL-correct {go_p['acc_all']:.3f} (per-slot {go_p['per_slot']}, >> chance {chance:.3f}). "
                       f"The HOLD is load-bearing (lesion-the-hold all {go_p['lesion_all']:.3f}, hold-alive "
                       f"{go_p['alive_reg']} with external input ASSERTED zero across the span), the SEPARATION is "
                       f"load-bearing (superposed-single-slot collides: all {go_p['superposed_all']}), NO filler-swap "
                       f"cross-talk ({go_p['filler_swap_rate']:.3f}), no topic->answer leakage (referent-shuffle per-slot "
                       f"{go_p['refshuf_per_slot_mean']:.3f}). CAPACITY CEILING (all-correct>=0.80) = k={ceiling}. This "
                       f"scales the single-slot WM GO to multiple concurrent variables. NOTE: the role-by-position gate is "
                       f"a host marker (the LEARNED spiking role-gate is gap#4, per 739a8867); reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not headline: miss.append(f"all-correct {go_p['acc_all']:.3f}/min per-slot {min(go_p['per_slot']):.3f} not >=0.85")
            if not recurrence: miss.append(f"hold not load-bearing (lesion {go_p['lesion_all']:.3f})")
            if not separation: miss.append(f"separation not load-bearing (superposed {go_p['superposed_all']})")
            if not no_swap: miss.append(f"filler-swap cross-talk {go_p['filler_swap_rate']:.3f} > 0.05")
            if not no_leak: miss.append(f"leakage (referent-shuffle {go_p['refshuf_per_slot_mean']:.3f})")
            if not held_alive: miss.append(f"hold not alive/zero-input (alive {go_p['alive_reg']}, zero {go_p['zero_input_ok']})")
            verdict = (f"PARTIAL/HONEST-NEGATIVE — multi-slot did not clear the GO bar at k={a.go_k}: " + "; ".join(miss)
                       + f". CAPACITY CEILING (all-correct>=0.80) = k={ceiling}. If the collapse is cross-talk/suppression "
                       f"(collapse-rate {go_p['collapse_rate']:.3f}), the named fix is more disjoint slot allocation "
                       f"(hetero-LTD allocation lane) or independent attractor pools / per-register FS. Read per-arm numbers.")
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = f"ERROR — no go point at k={a.go_k}"

    # --- earned verdict preconditions (validity travels with the verdict) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("multi_slot_binding", chance=chance)
        if go_p is not None:
            Vd.require("generalisation_defined_novel_heldout", 1 if go_p["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out filler tuples disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("superposed_baseline_collides", round(go_p["superposed_all"], 4) if go_p["superposed_all"] is not None else 0.0,
                       expect=lambda x: x <= chance + 0.10,
                       note="the superposed-single-slot control MUST collide (all-correct<=chance) or the separation is not "
                            "the thing being tested (VALIDITY of the load-bearing control)")
            Vd.require("referent_shuffle_at_chance", round(go_p["refshuf_per_slot_mean"], 4), expect=lambda x: x <= chance + 0.10,
                       note="the referent-shuffle per-slot readout must sit near chance (no topic->answer leakage)")
            Vd.require("hold_zero_input_asserted", 1 if go_p["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="every register sustained with external input ASSERTED zero across the hold+read span")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before the go point")
        _go = bool(go_p is not None and go_p["gen_defined"] and go_p["acc_all"] >= 0.85
                   and min(go_p["per_slot"]) >= 0.85 and go_p["acc_all"] >= go_p["lesion_all"] + 0.30
                   and (go_p["superposed_all"] is None or go_p["superposed_all"] <= chance + 0.10)
                   and go_p["filler_swap_rate"] <= 0.05)
        dec = Vd.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "multi_slot_binding", "verdict": verdict, "capacity_ceiling_all80": ceiling,
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "R banks of the D3 slow-NMDA (tau=100ms) bistable persistent-activity slot on ONE bridge "
                            "sharing ONE FS inhibitory pool (build_persistent_slot K=R*n_slot); register r = pools "
                            "[r*n_slot,(r+1)*n_slot); each fixed entity binds to a stable local slot via the RUNG6c "
                            "content-agnostic Hebbian binder; a role-by-position MARKER gate LOADs each role's entity into "
                            "its register; the k verbs are each predicted from the HELD bump of the correct register",
               "task": "a k-role agreement clause [e_r]+[L fillers] per role then k agreeing verbs; held-out TEST = "
                       "disjoint NOVEL filler tuples; anti-cheats: per-slot+all-correct recovery, lesion-the-hold, "
                       "superposed-single-slot collide baseline, cross-talk (filler-swap + collapse), referent-shuffle, "
                       "hold-not-re-read (zero-input assert); k swept to locate the capacity ceiling",
               "seeds": a.seeds, "config": {"n_ent": a.n_ent, "n_slot": a.n_slot, "n_fill": a.n_fill, "distance": a.distance,
               "ks": ks, "recur": a.recur, "hold_steps": a.hold_steps, "load_steps": a.load_steps,
               "clear_steps": a.clear_steps, "n_train": a.n_train, "n_test": a.n_test, "chance": chance, "go_k": a.go_k},
               "go_point": go_p, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of three banked GOs (D3 spiking NMDA hold + RUNG6c Hebbian bind + the "
                              "agreement-stream vocab); NO sim/ edit. Complements the gap#2 keystone LTM-store arc "
                              "(2026-07-17, competitive-slot RETRIEVAL P=3=1.00/P=4=0.79, hold NOT load-bearing there) by "
                              "testing the WORKING-MEMORY hold (load-bearing NMDA, read the HELD bump) on the held-out "
                              "agreement stream. Registers occupy DISJOINT pool banks with a SHARED FS, so cross-register "
                              "filler-swap is impossible by construction (each register reads only its own pools, per the "
                              "keystone 'separate slots never share a value pool'); the honest capacity signal is "
                              "SUPPRESSION (a register loses its bump under shared inhibition) + the superposed-single-slot "
                              "collide baseline (the ~2-cap regime). The role-by-position gate is a host marker (the learned "
                              "spiking role-gate is gap#4). The BIND is host numpy; the verb readout is a host deref. 1-seed "
                              "is a SMOKE indicator; the 6-seed sweep is decisive."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[multi_slot_binding] VERDICT: {verdict}", flush=True)
    print(f"[multi_slot_binding] capacity ceiling (all-correct>=0.80) = k={ceiling}", flush=True)
    print(f"[multi_slot_binding] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
