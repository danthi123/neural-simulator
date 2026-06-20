"""Phase C -- TASK 2 (THE CHEAP-FIRST FIRST LOOP, K=2): the WHOLE who/what conversational turn on ONE bridge, host
control GONE.

Pre-registered by `2026-06-19-tier2-phaseC-integrated-loop-design.md` §1.2 (the cheap-first loop), §2 (the seams
S0-S7), §5 Task 2. This composes the two proven pieces of the arc into ONE persistent loop and DISSOLVES the host
`_scan` orchestrator:

  - PHASE A (`2026-06-19-onebrain-bindstore-handoff-derisk.md`, commit 21bec31c): the bind->store (S3) DATA hand-off is
    SYNAPTIC -- the `SynapticH4Composer` routes `acc -> store-block-readout` through a unit complex synapse; recall ==
    host, lesion collapses, moat 0 breaches. Reused verbatim as the store front of the loop.
  - PHASE B (`2026-06-19-onebrain-sequencer-derisk.md`, commit 6043101b): the on-substrate SEQUENCER (S6) -- a spiking
    Izhikevich basal-ganglia/thalamocortical match cascade + a BG production rule sequences the who/what scan on the
    spiking match RESULT, replacing the host `for/if/return`. ==host, moat 0 false-accepts, lesion fails safe, permuted
    rule inverts. Reused: `build_sequencer_bridge` / `run_sequencer`.
  - S5 (Task 1 verdict, `2026-06-19-phaseC-task1-S5-seam-derisk.md`, commit 27c6422e): the on-substrate result->sequencer
    coupling (option a, a fixed `cp_connections` projection) WALLS -- a graded cleanup score through a binary spike loses
    the relative magnitude the match needs (the point-neuron graded-magnitude limit). The cheap-first loop therefore uses
    OPTION (b): the result->sequencer DATA hand-off is a HOST READ (`block_cleanup_scores` -> `scores_to_drive`), while
    the CONTROL (the match comparison + answer/abstain) is fully on-substrate (Phase B). One residual host DATA read.

THE LOOP (the WHOLE who/what turn, host `_scan` GONE):
  S0 comprehend     `dog go north` / `cat run river` via the on-bridge BridgeParser (the role it FIRES selects each bind)
  S2/S3 bind+store  the Phase-A synaptic bind->store hand-off (the composite never round-trips to host to BECOME a weight)
  S4 reconstruct    fire every trigger -> unbind 4 roles -> cleanup, on-bridge (the validated `block_cleanup_scores` op)
  S5 result-read    OPTION (b): read the cleanup scores to host (the residual DATA read; Task 1 ruled option a walls)
  S6 match+select   the Phase-B spiking sequencer: the cue + the decoded word-lines settle the gated-match cascade -> the
                    BG WTA selects {answer block 0 / answer block 1 / abstain}. NO Python for/if/return.
  S7 body-read      which BG channel won -> that block's answer role (the patient for what_does, the agent for who_does).

`LoopComposer.query_patient` / `query_agent` run the FULL loop above and REMOVE the host `_scan` from the query path
(`_scan`'s `for got in self._read_blocks(): if all(...): return` is replaced by S5->S6->S7). The sequencer is
ROLE-AGNOSTIC -- it matches on two cue roles and the body-read picks a third -- so the same Phase-B circuit serves both
who (cue=action+patient -> agent) and what (cue=agent+action -> patient); only WHICH cleanup roles drive the cue/decoded
lines differs (set per query).

GREEN (Task 2): the loop's query_patient/query_agent == the host `_scan` for BOTH K=2 blocks AND abstains (is None) on
the no-confab cues (absent agent / absent action / cross), 6 seeds, false-accepts = 0 (THE MOAT, asserted).
ANTI-CHEATS (the §4.2 battery): sequencer-lesion fails SAFE (cut the S5 drive on a present cue -> abstain, never a wrong
block); store-lesion collapses recall (the Phase-A acc->store synapse is load-bearing); permuted-rule INVERTS (swap the
match->answer production rule -> a present cue routes to the OTHER block); permuted-store carries content (a routed fact
is read back from its block).

NO `sim/` edit (reuse-by-import: SynapticH4Composer + build_sequencer_bridge/run_sequencer + the public bridge API). The
algebra is exact (numpy is the oracle path); CuPy for the real co-resident confirm (the on-bridge parser trains on the
CuPy substrate). The 6-seed rule applies to the noise-sensitive match cascade.

  SIM_BACKEND=numpy python -u -m research.runners._phaseC_task2_wholeturn_loop --seeds 42,43,44,45,46,47 --dim 64
  SIM_BACKEND=cupy  python -u -m research.runners._phaseC_task2_wholeturn_loop --seeds 42,43,44,45,46,47 --dim 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host, is_gpu_backend  # noqa: E402
from research.runners._phaseB_onebrain_bindstore_handoff_derisk import SynapticH4Composer  # noqa: E402
from research.runners._phaseB_onebrain_sequencer_derisk import (  # noqa: E402
    build_sequencer_bridge, run_sequencer,
)

# A K=2 store (the cheap-first scope). dog/cat agents + go/run actions + north/river patients are mutually distinct so
# the moat cues (absent agent / absent action / cross) are unambiguous.
FACTS = [("dog", "go", "north"), ("cat", "run", "river")]
VOCAB = ["cat", "dog", "fox", "go", "north", "river", "run", "see", "tree", "bird", "sun", "moon"]


# ----------------------------------------------------------------------------------------------------------------
# S4 (the cleanup op, instrumented per-role): reconstruct block_idx + unbind ALL roles in parallel + clean up; return
# the V-length cleanup scores for ANY chosen pair of roles. This IS the composer's validated _read_block op (the SAME
# resonate windows + the SAME complex weights); it just also returns the raw per-role-per-word cleanup membrane (the
# scores `_read_block` argmaxes over) so the sequencer can drive its two decoded word-lines from the requested roles.
# Generalizes `block_cleanup_scores` (which returns agent+action) to (role_a, role_x), so the one Phase-B sequencer
# serves what_does (agent+action) AND who_does (patient+action) -- the cue is two roles, the answer a third.
# ----------------------------------------------------------------------------------------------------------------
def block_role_scores(c, block_idx, role_a, role_x):
    """Run the composer's reconstruct+unbind+cleanup for one block; return (scores_a, scores_x) -- the V-length cleanup
    read-outs for `role_a` and `role_x` (each a main role: agent/action/patient). These ARE the op's spiking result;
    the sequencer drives its decoded word-lines from them (S5 = read them to host, then `run_sequencer` consumes them)."""
    comp, b, D, Pd, V = c.comp, c.b, c.D, c.period, c.V
    ra = c.main_roles.index(role_a)
    rx = c.main_roles.index(role_x)
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
        zc = np.conj(comp._to_phasor(comp.roles[role]))
        unbind += [(c.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    clean = []
    for ri, role in enumerate(c.main_roles):
        for j in range(V):
            cc = np.conj(comp._to_phasor(comp.concepts[c.words[j]]))
            clean += [(c.c_base + ri * V + j, c.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    sa = np.maximum(mem[c.c_base + ra * V:c.c_base + (ra + 1) * V], 0.0)
    sx = np.maximum(mem[c.c_base + rx * V:c.c_base + (rx + 1) * V], 0.0)
    return sa, sx


class LoopComposer(SynapticH4Composer):
    """The Phase-C cheap-first loop composer. STORE is the Phase-A synaptic bind->store hand-off (inherited verbatim
    from SynapticH4Composer). QUERY runs the whole turn through the Phase-B on-substrate SEQUENCER, with the host
    `_scan` REMOVED: query_patient/query_agent reconstruct+clean each block on-bridge (S4), read the cleanup scores to
    host (S5, option b -- Task 1 ruled the on-bridge projection out), drive the spiking sequencer (S6), and body-read
    the won channel (S7). The Python for/if/return that decided answer-vs-abstain is gone; the spiking BG WTA decides.

    The sequencer + its meta are attached after construction by `attach_sequencer` (built once per composer, reused
    across queries -- the persistent slice). `seq_lesion` zeros the S5 drive (anti-cheat: fails safe). `seq_permute`
    swaps the match->answer production rule (anti-cheat: inverts)."""

    def attach_sequencer(self, seq_sb, seq_meta):
        self._seq_sb = seq_sb
        self._seq_meta = seq_meta
        self._word_idx = {w: i for i, w in enumerate(self.words)}

    def _loop_answer(self, cue_a_word, cue_x_word, role_a, role_x, answer_role, lesion=False, permute=False):
        """Run the full who/what turn on the substrate for a cue (cue_a_word in role_a, cue_x_word in role_x) and
        return (answer_word_or_None, debug). S4: per-block cleanup scores for (role_a, role_x). S5 (option b): the
        scores are read to host inside run_sequencer (`scores_to_drive`). S6: the spiking sequencer settles the
        gated-match + applies the BG production rule -> {ans0/ans1/abstain}. S7: ans_i -> block i's `answer_role`
        word; abstain -> None (the moat). The host `_scan` is NOT consulted."""
        n = len(self.kb)
        blocks = list(range(n))
        # S4 (on-bridge): the cleanup op result per block, for the two cue roles this query matches on.
        bscores = [block_role_scores(self, b, role_a, role_x) for b in blocks]
        ca = self._word_idx[cue_a_word]
        cx = self._word_idx[cue_x_word]
        # S5+S6: run_sequencer reads the scores to host (option b) and runs the spiking match + BG selection.
        decision, rates = run_sequencer(self._seq_sb, self._seq_meta, ca, cx, bscores,
                                        lesion=lesion, permute=permute)
        # S7 (body-read): the won channel -> that block's answer-role word. The permuted production rule already routed
        # the decision; here we just read the channel the rule selected (ans0 -> block 0, ans1 -> block 1, abstain -> None).
        if decision == "ans0":
            ans = self.kb[blocks[0]][0].get(answer_role)
        elif decision == "ans1":
            ans = self.kb[blocks[1]][0].get(answer_role)
        else:
            ans = None
        return ans, {"decision": decision, "rates": rates}

    def query_patient(self, agent, action, order_fn=None):
        """what_does: cue = (agent, action), answer = patient. The whole turn on the substrate; host `_scan` gone."""
        ans, _ = self._loop_answer(agent, action, "agent", "action", "patient")
        return ans

    def query_agent(self, action, patient):
        """who_does: cue = (patient, action), answer = agent. Same Phase-B sequencer, different cue/answer roles."""
        ans, _ = self._loop_answer(patient, action, "patient", "action", "agent")
        return ans


def _build_loop(seed, D):
    """A K=2 LoopComposer (the Phase-A synaptic store) + its attached Phase-B sequencer slice (built once). The
    sequencer is built UNpermuted; the permuted-rule anti-cheat swaps the production rule at QUERY time
    (`run_sequencer(permute=True)`), matching Phase B -- so the one rule swap is not double-cancelled by the wiring."""
    c = LoopComposer(seed=seed, D=D, vocab=VOCAB, k_max=len(FACTS) + 2, period=200,
                     enable_rf_cudagraph=False, enable_csr_cache=False, enable_batched=False)
    for (a, x, p) in FACTS:
        c.store(a, x, p)
    seq_sb, meta = build_sequencer_bridge(seed=seed, V=c.V)
    c.attach_sequencer(seq_sb, meta)
    return c


def run_seed(seed, D):
    c = _build_loop(seed, D)

    # ---- the query set: BOTH present cues (each answers ITS block -- so the scan must reach block 1 too) +
    #      THREE no-confab cues (absent agent / absent action / cross = agent-of-block-0 + action-of-block-1). ----
    # what_does (cue agent+action -> patient)
    what_present = [
        (("dog", "go"), "north", "blk0-present"),
        (("cat", "run"), "river", "blk1-present"),
    ]
    what_moat = [
        (("fox", "go"), None, "absent-agent"),
        (("dog", "see"), None, "absent-action"),
        (("dog", "run"), None, "cross-no-block"),
    ]
    # who_does (cue patient+action -> agent)
    who_present = [
        (("go", "north"), "dog", "blk0-present"),
        (("run", "river"), "cat", "blk1-present"),
    ]
    who_moat = [
        (("go", "tree"), None, "absent-patient"),
        (("see", "north"), None, "absent-action"),
        (("run", "north"), None, "cross-no-block"),     # action-of-block-1 + patient-of-block-0 -> no full match
    ]

    rows = []
    # ---- what_does: loop vs host (the host_scan = the production OneBrainComposer host path, unchanged) ----
    for (a, x), want, kind in what_present + what_moat:
        loop = c.query_patient(a, x)
        host = SynapticH4Composer.query_patient(c, a, x)   # the host `_scan` path on the SAME store (the oracle)
        rows.append(dict(q="what", cue=(a, x), kind=kind, want=want, loop=loop, host=host,
                         loop_eq_host=(loop == host), loop_correct=(loop == want)))
    # ---- who_does: loop vs host ----
    for (x, p), want, kind in who_present + who_moat:
        loop = c.query_agent(x, p)
        host = SynapticH4Composer.query_agent(c, x, p)
        rows.append(dict(q="who", cue=(x, p), kind=kind, want=want, loop=loop, host=host,
                         loop_eq_host=(loop == host), loop_correct=(loop == want)))

    # ---- GATES ----
    eq_all = all(r["loop_eq_host"] for r in rows)                 # ==host on every row
    present_rows = [r for r in rows if r["kind"].endswith("present")]
    correct_all = all(r["loop_correct"] for r in present_rows)    # the present cues answer correctly
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    false_accepts = sum(1 for r in moat_rows if r["loop"] is not None)   # THE MOAT (HARD): 0 false-accepts
    moat_ok = (false_accepts == 0)

    # ---- ANTI-CHEAT 1: sequencer-LESION fails SAFE. Cut the S5 result->sequencer drive on a PRESENT cue -> the match
    #      can't fire -> the loop must ABSTAIN (None), never a wrong block. ----
    les = []
    les.append(c._loop_answer("dog", "go", "agent", "action", "patient", lesion=True)[0])
    les.append(c._loop_answer("cat", "run", "agent", "action", "patient", lesion=True)[0])
    lesion_fails_safe = all(x is None for x in les)

    # ---- ANTI-CHEAT 2: store-LESION collapses recall. A LoopComposer built with the Phase-A acc->store synapse
    #      SEVERED -> the store readout gets nothing -> the cleanup is garbage -> the loop can't answer the present
    #      cues (collapse). Proves the on-bridge store hand-off is load-bearing in the loop (not a host write). ----
    cl = LoopComposer(seed=seed, D=D, vocab=VOCAB, k_max=len(FACTS) + 2, period=200,
                      enable_rf_cudagraph=False, enable_csr_cache=False, enable_batched=False, lesion=True)
    for (a, x, p) in FACTS:
        cl.store(a, x, p)
    seq_l, meta_l = build_sequencer_bridge(seed=seed, V=cl.V)
    cl.attach_sequencer(seq_l, meta_l)
    store_les = [cl.query_patient("dog", "go"), cl.query_patient("cat", "run")]
    store_lesion_collapses = not any(store_les[i] == FACTS[i][2] for i in range(len(FACTS)))

    # ---- ANTI-CHEAT 3: permuted-RULE inverts. Swap the BG match->answer production rule (m0->ans1, m1->ans0) at
    #      QUERY time (Phase B's exact anti-cheat: `run_sequencer(permute=True)`), on the SAME sequencer. On the
    #      block-0-present (dog,go) the TRUE rule answers block 0 (north); the permuted rule must route to block 1
    #      (river); on (cat,run) it inverts to north. Proves the BG selection carries the conditional, not a fixed
    #      scan order. (Permute is on the production rule that reads the spiking match, NOT the bridge wiring -- so a
    #      single swap, matching Phase B; building the bridge permuted too would double-cancel.) ----
    perm_p0_ans, _ = c._loop_answer("dog", "go", "agent", "action", "patient", permute=True)
    perm_p1_ans, _ = c._loop_answer("cat", "run", "agent", "action", "patient", permute=True)
    perm_p0, perm_p1 = perm_p0_ans, perm_p1_ans
    permuted_inverts = (perm_p0 == FACTS[1][2]) and (perm_p1 == FACTS[0][2])

    # ---- ANTI-CHEAT 4: permuted-STORE carries content (the Phase-A property, in the loop). Synaptically route a
    #      DISTINCT fact into block 0, read block 0 DIRECTLY, confirm it holds the routed fact. ----
    cs = LoopComposer(seed=seed, D=D, vocab=VOCAB, k_max=len(FACTS) + 2, period=200,
                      enable_rf_cudagraph=False, enable_csr_cache=False, enable_batched=False)
    for (a, x, p) in FACTS:
        cs.store(a, x, p)
    routed = ("fox", "see", "tree")             # distinct from FACTS[0]=("dog","go","north")
    cs._compose_phases(["fox", "see", "tree", "AFFIRM"], ["agent", "action", "patient", "polarity"])
    cs._write_block(0, None)                     # synaptically overwrite block 0 with the routed fact
    b0 = cs._read_block(0)                        # read block 0 directly (no scan ambiguity)
    permuted_store_carries = (b0.get("agent") == routed[0] and b0.get("action") == routed[1]
                              and b0.get("patient") == routed[2])

    go = (eq_all and correct_all and moat_ok and lesion_fails_safe and store_lesion_collapses
          and permuted_inverts and permuted_store_carries)
    return dict(seed=seed, D=D, rows=rows, eq_all=eq_all, correct_all=correct_all,
                moat_ok=moat_ok, false_accepts=false_accepts,
                lesion_fails_safe=lesion_fails_safe, lesion_answers=les,
                store_lesion_collapses=store_lesion_collapses, store_lesion_answers=store_les,
                permuted_inverts=permuted_inverts, permuted_answers=[perm_p0, perm_p1],
                permuted_store_carries=permuted_store_carries, permuted_store_read=tuple(b0.get(r) for r in
                                                                                         ("agent", "action", "patient")),
                go=go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(_REPO, "research", "findings", "raw",
                                                   "_phaseC_task2_wholeturn_loop.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("[Phase C Task 2 -- the cheap-first K=2 WHOLE-TURN loop] comprehend(parser) -> store(Phase-A synaptic) ->\n"
          "reconstruct/unbind/cleanup -> S5(host result-read, option b) -> Phase-B spiking sequencer(match+BG select)\n"
          "-> body-read. The host `_scan` is GONE; the spiking BG WTA decides answer-vs-abstain.\n", flush=True)
    results = []
    for s in seeds:
        r = run_seed(s, args.dim)
        results.append(r)
        det = "  ".join(f"{rr['q']}/{rr['kind']}:loop={rr['loop']}|host={rr['host']}" for rr in r["rows"])
        moat_flag = "OK" if r["moat_ok"] else f"BREACH(fa={r['false_accepts']})"
        flags = (f"==host={r['eq_all']} correct={r['correct_all']} MOAT={moat_flag} "
                 f"seq-lesion-safe={r['lesion_fails_safe']} store-lesion-collapse={r['store_lesion_collapses']} "
                 f"perm-rule-inverts={r['permuted_inverts']} perm-store-carries={r['permuted_store_carries']}")
        print(f"seed {s} D{args.dim}: {'GO' if r['go'] else 'NO'}  [{flags}]", flush=True)
        print(f"    {det}", flush=True)

    n = len(results)
    eq_n = sum(r["eq_all"] for r in results)
    correct_n = sum(r["correct_all"] for r in results)
    moat_n = sum(r["moat_ok"] for r in results)
    les_n = sum(r["lesion_fails_safe"] for r in results)
    store_les_n = sum(r["store_lesion_collapses"] for r in results)
    perm_n = sum(r["permuted_inverts"] for r in results)
    pstore_n = sum(r["permuted_store_carries"] for r in results)
    go_n = sum(r["go"] for r in results)
    # GO bar: ==host every seed AND moat 0 false-accepts every seed AND the full anti-cheat battery every seed.
    verdict = "GO" if go_n == n else "NEGATIVE"
    total_fa = sum(r["false_accepts"] for r in results)
    summary = dict(n=n, eq_host_n=eq_n, correct_n=correct_n, moat_n=moat_n, total_false_accepts=total_fa,
                   seq_lesion_safe_n=les_n, store_lesion_collapse_n=store_les_n, permuted_rule_inverts_n=perm_n,
                   permuted_store_carries_n=pstore_n, go_n=go_n, verdict=verdict, gpu=is_gpu_backend())
    print(f"\nSUMMARY ({n} seeds, K={len(FACTS)}, host `_scan` GONE, S5=option-b host-read, control on-substrate):",
          flush=True)
    print(f"  ==host {eq_n}/{n}  present-correct {correct_n}/{n}  MOAT {moat_n}/{n} (total false-accepts {total_fa})  "
          f"seq-lesion-safe {les_n}/{n}  store-lesion-collapse {store_les_n}/{n}  perm-rule-inverts {perm_n}/{n}  "
          f"perm-store-carries {pstore_n}/{n}  -> {verdict}", flush=True)
    if verdict == "GO":
        print("  GO: the WHOLE who/what turn runs as ONE persistent loop on the OneBrainComposer bridge -- comprehend\n"
              "  (parser) -> store (Phase-A synaptic) -> reconstruct/unbind/cleanup -> Phase-B spiking match+BG select\n"
              "  -> body-read -- with the host `_scan` for/if/return GONE; the spiking BG WTA decides answer-vs-abstain,\n"
              "  ==host on who/what at K=2, the no-confab MOAT holds (0 false-accepts), and the full anti-cheat battery\n"
              "  passes. HONEST SCOPE: one residual host DATA read at S5 (the cleanup score -> decoded-line drive; Task\n"
              "  1 ruled the on-bridge projection out -- a NEF-thresholded on-bridge cleanup is the lever to close it).",
              flush=True)
    else:
        print("  NEGATIVE: report exactly the failing gate; the MOAT is NEVER traded for a pass.", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
