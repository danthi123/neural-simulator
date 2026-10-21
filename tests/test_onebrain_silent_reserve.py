"""Silent reserve: an UNRECRUITED reserve cleanup slot (`__free<k>__`) gets NO cleanup wiring (CPU, no GPU needed).

THE REGRESSION (FAILURE_LOG 2026-10-21): `vocab_headroom` reserve slots each carried a random phasor code and were
wired into the cleanup read like real words, so at the agent's shipped headroom (128) an unrecruited placeholder
could out-score the true word (seed 42, clause read, action role: `__free10__` 5270 vs `look` 4080). Once `_select`
began returning None for a placeholder winner, that wrong-word leak became a crash in `_decode_clause`
(`" ".join(words[o] ...)` with a None) -- caught only by the GPU-only `test_onebrain_agent_clause_fact`.

THE FIX (biological framing: a reserve slot is an uncommitted assembly whose synapses are silent until it is
recruited): every cleanup-wiring site skips a slot while `self.words[j]` is still a `__free` placeholder
(`OneBrainComposer._silent_reserve`), so with no input those cleanup cells stay at rest and score exactly 0; recruiting
the slot renames it to the real word and the next read wires it. `_decode_clause` abstains (None) when any role
decodes to None instead of raising. At `vocab_headroom=0` there are no reserve slots, so every list is unchanged.

The wiring checks compare each site's connection list against an INDEPENDENT oracle (the pre-change loop, written out
here, with the silent slots defined by the placeholder NAME, not by the helper under test).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.backend import to_host  # noqa: E402

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]
HEADROOM = 6


def _build(headroom, seed=7, D=64, fact_shard=False, **kw):
    """A small composer. `fact_shard=False` (the default here) builds the BATCHED layout so every wiring site, the
    batched ones included, can run; the shipped default is fact-shard ON (no batched region), read from the env var."""
    from research.runners.one_brain_composer import OneBrainComposer
    kw.setdefault("enable_spiking_cleanup", False)
    prev = os.environ.get("BRAIN_FACT_SHARD_RETRIEVAL")
    os.environ["BRAIN_FACT_SHARD_RETRIEVAL"] = "1" if fact_shard else "0"
    try:
        return OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=4, vocab_headroom=headroom,
                                enable_rf_cudagraph=False, **kw)
    finally:
        if prev is None:
            os.environ.pop("BRAIN_FACT_SHARD_RETRIEVAL", None)
        else:
            os.environ["BRAIN_FACT_SHARD_RETRIEVAL"] = prev


def _is_placeholder(w):
    return isinstance(w, str) and w.startswith("__free")


def _record_weights(c):
    """Wrap the bridge's rf_set_complex_weights so every installed connection list is captured (in call order)."""
    calls = []
    orig = c.b.rf_set_complex_weights

    def rec(connections):
        calls.append(list(connections))
        return orig(connections)
    c.b.rf_set_complex_weights = rec
    return calls


def _oracle_main_clean(c, base, qreg_of, roles, skip_placeholders):
    """The pre-change cleanup loop for the main vocab: `for ri in roles: for j in range(V): ...` (the original wiring),
    minus -- when `skip_placeholders` -- the slots whose word is still a `__free` placeholder (the post-change rule)."""
    out = []
    for ri in roles:
        for j in range(c.V):
            if skip_placeholders and _is_placeholder(c.words[j]):
                continue
            cc = c._cleanup_conj(c.words[j])
            out += [(base + ri * c.V + j, qreg_of(ri) + k, complex(cc[k])) for k in range(c.D)]
    return out


def _oracle_pol_clean(c, base, qreg):
    out = []
    for j in range(c.NP):
        cc = c._cleanup_conj(c.pol_words[j])
        out += [(base + c.n_main * c.V + j, qreg + k, complex(cc[k])) for k in range(c.D)]
    return out


def _silent_cells(c, base):
    return {base + ri * c.V + j for ri in range(c.n_main) for j in range(c.V) if _is_placeholder(c.words[j])}


def _posts(conns):
    return {p for (p, _q, _w) in conns}


def _csr_rows_with_weights(csr_pair):
    re_, im_ = csr_pair
    rows = set()
    for m in (re_, im_):
        indptr = np.asarray(to_host(m.indptr))
        rows |= {int(r) for r in np.flatnonzero(np.diff(indptr) > 0)}
    return rows


def _two_facts(c):
    c.store("dog", "go", "north")
    c.store("cat", "come", "east")


# ---------------------------------------------------------------------------------------------------------------
# (a) headroom 0: nothing is silent, every wiring list is exactly the pre-change list
# ---------------------------------------------------------------------------------------------------------------
def test_headroom0_has_no_silent_slot_and_wiring_is_unchanged():
    c = _build(0)
    assert c.V == len(VOCAB) + 0 and not any(_is_placeholder(w) for w in c.words)
    assert not any(c._silent_reserve(j) for j in range(c.V)), "no reserve slot exists at vocab_headroom=0"
    _two_facts(c)
    D, V = c.D, c.V

    # per-block read: the cleanup list is the pre-change loop, entry for entry
    calls = _record_weights(c)
    c._read_block(0)
    want = (_oracle_main_clean(c, c.c_base, lambda ri: c.q_base + ri * D, range(c.n_main), False)
            + _oracle_pol_clean(c, c.c_base, c.q_base + c.bind_roles.index("polarity") * D))
    assert calls[-1] == want

    # batched, cache OFF (stock path): same
    c.enable_csr_cache = False
    calls = _record_weights(c)
    c._read_all_blocks()
    want = []
    pol_ri = c.bind_roles.index("polarity")
    for i in range(2):
        cblk = c.bat_c_base + i * c.cb
        want += _oracle_main_clean(c, cblk, lambda ri, i=i: c.bat_q_base + (i * c.n_roles + ri) * D,
                                   range(c.n_main), False)
        want += _oracle_pol_clean(c, cblk, c.bat_q_base + (i * c.n_roles + pol_ri) * D)
    assert calls[-1] == want

    # batched, cache ON: the cached CSR is byte-equal to the CSR built from the pre-change list
    from research.runners.one_brain_composer import _build_complex_csr
    (_U, (Cre, Cim)) = c._build_batched_unbind_clean(2)
    Wre, Wim = _build_complex_csr(c.n_total, want)
    for got, ref in ((Cre, Wre), (Cim, Wim)):
        assert np.array_equal(np.asarray(to_host(got.indptr)), np.asarray(to_host(ref.indptr)))
        assert np.array_equal(np.asarray(to_host(got.indices)), np.asarray(to_host(ref.indices)))
        assert np.array_equal(np.asarray(to_host(got.data)), np.asarray(to_host(ref.data)))

    # sequencer drive-seed codebook
    seq = c._seq_cleanup_conns()
    assert len(seq) == c.n_main * V * D


# ---------------------------------------------------------------------------------------------------------------
# (b) headroom > 0: no cleanup weight targets a free slot's cell, at every wiring site; recruit wires the slot
# ---------------------------------------------------------------------------------------------------------------
def test_silent_reserve_helper_tracks_the_placeholder_name():
    c = _build(HEADROOM)
    n_real = len(VOCAB)
    assert [c._silent_reserve(j) for j in range(c.V)] == [False] * n_real + [True] * HEADROOM
    c._recruit_word("zebra")
    s = c._word_index["zebra"]
    assert s == n_real and c.words[s] == "zebra" and not c._silent_reserve(s)
    assert [j for j in range(c.V) if c._silent_reserve(j)] == list(range(n_real + 1, n_real + HEADROOM))


def test_per_block_and_trace_and_patient_sites_skip_free_slots():
    c = _build(HEADROOM)
    _two_facts(c)
    D = c.D
    silent = _silent_cells(c, c.c_base)
    assert len(silent) == c.n_main * HEADROOM
    qreg = lambda ri: c.q_base + ri * D                                    # noqa: E731
    pol_ri = c.bind_roles.index("polarity")
    want_block = (_oracle_main_clean(c, c.c_base, qreg, range(c.n_main), True)
                  + _oracle_pol_clean(c, c.c_base, c.q_base + pol_ri * D))

    calls = _record_weights(c)
    c._read_block(0)
    assert calls[-1] == want_block
    assert not (_posts(calls[-1]) & silent)
    mem = np.asarray(to_host(c.b.cp_membrane_potential_v)).astype(float)
    assert all(mem[p] == 0.0 for p in silent), "a silent reserve cell must sit at rest (score exactly 0)"

    calls = _record_weights(c)
    c._block_role_scores(0)                                                # the trace read
    assert calls[-1] == want_block

    calls = _record_weights(c)
    c._patient_cleanup_scores(0)                                           # reconsolidation prediction-error read
    # the single-role read wires only the patient Q register into role-0's cleanup cells
    want_patient = []
    for j in range(c.V):
        if _is_placeholder(c.words[j]):
            continue
        cc = c._cleanup_conj(c.words[j])
        want_patient += [(c.c_base + j, c.q_base + 2 * D + k, complex(cc[k])) for k in range(D)]
    assert calls[-1] == want_patient
    assert not (_posts(calls[-1]) & silent)


def test_batched_sites_skip_free_slots_and_membrane_stays_at_rest():
    c = _build(HEADROOM)
    _two_facts(c)
    D, n = c.D, 2
    pol_ri = c.bind_roles.index("polarity")
    want = []
    silent = set()
    for i in range(n):
        cblk = c.bat_c_base + i * c.cb
        silent |= _silent_cells(c, cblk)
        want += _oracle_main_clean(c, cblk, lambda ri, i=i: c.bat_q_base + (i * c.n_roles + ri) * D,
                                   range(c.n_main), True)
        want += _oracle_pol_clean(c, cblk, c.bat_q_base + (i * c.n_roles + pol_ri) * D)

    # stock (cache off)
    c.enable_csr_cache = False
    calls = _record_weights(c)
    c._read_all_blocks()
    assert calls[-1] == want and not (_posts(calls[-1]) & silent)
    mem = np.asarray(to_host(c.b.cp_membrane_potential_v)).astype(float)
    assert all(mem[p] == 0.0 for p in silent), "a silent reserve cell must sit at rest (score exactly 0)"

    # cached CSR
    c.enable_csr_cache = True
    c._csr_cache.clear()
    c._read_all_blocks()
    assert n in c._csr_cache
    rows = _csr_rows_with_weights(c._csr_cache[n][1])
    assert not (rows & silent)
    assert rows == _posts(want), "every non-silent cleanup cell keeps its full wiring"
    mem = np.asarray(to_host(c.b.cp_membrane_potential_v)).astype(float)
    assert all(mem[p] == 0.0 for p in silent)


def test_clause_and_sequencer_sites_skip_free_slots():
    from research.runners.core_sim_composition import Clause
    c = _build(HEADROOM)
    c.store("dog", "go", Clause(agent="cat", action="look", patient="south"))
    D = c.D
    silent = _silent_cells(c, c.c_base)
    calls = _record_weights(c)
    c._decode_clause(0)
    want = _oracle_main_clean(c, c.c_base, lambda ri: c.q_base + ri * D, range(3), True)
    assert calls[-1] == want and not (_posts(calls[-1]) & silent)

    seq = c._seq_cleanup_conns()
    want_seq = []
    for ri in range(len(c.main_roles)):
        for j in range(c.V):
            if _is_placeholder(c.words[j]):
                continue
            cc = np.conj(c.comp._to_phasor(c.comp.concepts[c.words[j]]))
            want_seq += [(c.c_base + ri * c.V + j, c.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    assert seq == want_seq and not (_posts(seq) & silent)


def test_recruit_wires_the_slot_on_the_next_read_and_invalidates_the_cache():
    c = _build(HEADROOM)
    _two_facts(c)
    n_real = len(VOCAB)
    n = len(c.kb)
    c._read_all_blocks()                                                   # builds the cached CSR for n blocks
    assert n in c._csr_cache
    s = c._free_slots[0]
    assert s == n_real
    # the slot is silent before recruit
    before = _csr_rows_with_weights(c._csr_cache[n][1])
    cells = {c.bat_c_base + i * c.cb + ri * c.V + s for i in range(n) for ri in range(c.n_main)}
    assert not (before & cells)

    assert c._recruit_word("zebra") is True                                # direct recruit, no store in between
    assert not c._csr_cache, "recruit must drop the cached cleanup operator (the slot's wiring changed)"
    c._read_all_blocks()
    after = _csr_rows_with_weights(c._csr_cache[n][1])
    assert cells <= after, "the recruited slot is wired on the next batched read"
    still_free = {c.bat_c_base + i * c.cb + ri * c.V + j for i in range(n) for ri in range(c.n_main)
                  for j in range(c.V) if _is_placeholder(c.words[j])}
    assert not (after & still_free)

    # per-block read wires the recruited slot too, and only it
    calls = _record_weights(c)
    c._read_block(0)
    posts = _posts(calls[-1])
    assert {c.c_base + ri * c.V + s for ri in range(c.n_main)} <= posts
    assert not (posts & _silent_cells(c, c.c_base))

    # the whole store path: storing a fact with a never-seen word recruits it and the fact reads back
    c.store("bird", "look", "yak")
    assert c.words[c._word_index["yak"]] == "yak"
    assert c.query_patient("bird", "look") == "yak"


def test_restore_recruit_rewires_the_slot():
    c = _build(HEADROOM)
    _two_facts(c)
    n = len(c.kb)
    c._read_all_blocks()
    slot = len(VOCAB) + 3
    assert c.restore_recruit("heron", slot) is True
    assert not c._csr_cache
    c._read_all_blocks()
    rows = _csr_rows_with_weights(c._csr_cache[n][1])
    assert {c.bat_c_base + i * c.cb + ri * c.V + slot for i in range(n) for ri in range(c.n_main)} <= rows


def test_spiking_select_never_lands_on_a_silent_slot():
    """The spiking WTA gets only the rectified membrane scores; a silent slot scores 0, so it gets zero drive and cannot
    fire, and the all-zero degenerate fallbacks read word 0 (a real word), never a placeholder."""
    c = _build(HEADROOM, enable_spiking_cleanup=True)
    scores = np.zeros(c.V)
    scores[3] = 123.0
    assert c._spiking_select(scores, c.words) == c.words[3]
    assert c._select(scores, c.words) == c.words[3]
    assert c._spiking_select(np.zeros(c.V), c.words) == c.words[0]
    assert c._select(np.zeros(c.V), c.words) == c.words[0]
    _two_facts(c)
    row = c._read_block(0)
    assert row["agent"] == "dog" and row["action"] == "go" and row["patient"] == "north"


def test_a_free_slots_code_never_enters_the_wiring():
    """`separate_reserve_codes` re-codes free placeholders: harmless to the cleanup cache because a silent slot's code
    is not in any wiring list (only `_recruit_word` / `restore_recruit` consume it, and both drop the cache)."""
    c = _build(HEADROOM)
    _two_facts(c)
    calls = _record_weights(c)
    c._read_block(0)
    ref = calls[-1]
    ph = c.words[-1]
    assert _is_placeholder(ph)
    c.comp.concepts[ph] = np.random.default_rng(99).uniform(0.0, 1.0, c.D)
    calls = _record_weights(c)
    c._read_block(0)
    assert calls[-1] == ref


def test_shipped_default_fact_shard_path_never_wires_a_free_slot():
    """The shipped composer is fact-shard ON (no batched region): its recall runs `_read_block` on the shard. Every
    cleanup list installed by a full query round (who / what / yes-no / render) leaves the free slots' cells unwired."""
    c = _build(HEADROOM, fact_shard=True)
    assert c.enable_fact_shard and not c.enable_batched
    _two_facts(c)
    silent = _silent_cells(c, c.c_base)
    calls = _record_weights(c)
    assert c.query_patient("dog", "go") == "north"
    assert c.query_agent("come", "east") == "cat"
    assert c.ask_yes_no("dog", "go", "north") == "yes"
    assert c.render_fact("cat") == "cat come east"
    clean_lists = [cl for cl in calls if any(c.c_base <= p < c.c_base + c.cb for p in _posts(cl))]
    assert clean_lists, "the query round installed no cleanup list"
    assert not any(_posts(cl) & silent for cl in clean_lists)


# ---------------------------------------------------------------------------------------------------------------
# the defect itself, on CPU: the shipped agent shape (D=128, headroom 128, spiking cleanup) on the clause fact
# ---------------------------------------------------------------------------------------------------------------
def _agent_shape_clause_answer(seed):
    from research.runners.core_sim_composition import Clause
    from research.runners.one_brain_composer import OneBrainComposer
    c = OneBrainComposer(seed=seed, D=128, vocab=sorted(VOCAB), k_max=32, vocab_headroom=128,
                         enable_spiking_cleanup=True, enable_rf_cudagraph=False)
    c.store("dog", "go", Clause(agent="cat", action="look", patient="south"))
    return c.query_patient("dog", "go")


def test_agent_shape_clause_fact_seed42_decodes_the_true_clause():
    assert _agent_shape_clause_answer(42) == "cat look south"


@pytest.mark.parametrize("seed", [43, 44, 100, 101, 102])
def test_agent_shape_clause_fact_never_a_placeholder_never_a_crash(seed):
    """Seeds 44 and 100 return a wrong REAL word even at headroom 0 (an older, separate weakness of the two-hop clause
    decode, FAILURE_LOG 2026-10-21); this pins only what the silent reserve owns: no crash, no `__free` in the answer."""
    ans = _agent_shape_clause_answer(seed)
    assert ans is None or (isinstance(ans, str) and "__free" not in ans)


# ---------------------------------------------------------------------------------------------------------------
# (c) _decode_clause abstains (None) instead of raising when a role decodes to None; the callers pass it through
# ---------------------------------------------------------------------------------------------------------------
def _clause_composer(**kw):
    from research.runners.core_sim_composition import Clause
    c = _build(0, **kw)
    c.store("dog", "go", Clause(agent="cat", action="look", patient="south"))
    return c


def _blank_second_clause_role(c):
    """Make the 2nd role decoded inside `_decode_clause` come back None (what `_select` returns for a placeholder
    winner); every other `_select` call (the cue-matching reads) is untouched."""
    orig = c._select
    seen = []

    def sel(scores, words):
        # the call sits in a list comprehension (its own frame before Python 3.12): look two frames up as well
        # (plain calls: a generator expression here would add a frame of its own)
        f1 = sys._getframe(1).f_code.co_name
        f2 = sys._getframe(2).f_code.co_name
        if "_decode_clause" in (f1, f2):
            seen.append(1)
            if len(seen) == 2:
                return None
        return orig(scores, words)
    c._select = sel
    return seen


def test_decode_clause_returns_none_when_a_role_is_none_and_callers_pass_it_through():
    c = _clause_composer()
    assert isinstance(c._decode_clause(0), str)                            # control: an intact decode still joins
    assert isinstance(c.query_patient("dog", "go"), str)
    seen = _blank_second_clause_role(c)
    assert c._decode_clause(0) is None
    assert len(seen) == 3                                                  # all three roles were still read
    del seen[:]
    assert c.query_patient("dog", "go") is None                            # _finish_query_patient passes None through
    del seen[:]
    assert c.render_fact("dog") is None                                    # the other _decode_clause caller abstains too

    from research.runners.brain_conversational_agent import BrainConversationalAgent
    agent = SimpleNamespace(_neural_render=None, composer=c)
    del seen[:]
    assert BrainConversationalAgent.what_does(agent, "dog", "go") is None  # the agent's what_does abstains


def test_decode_clause_none_with_trace_on_records_the_abstention_chip():
    c = _clause_composer(trace=True)
    _blank_second_clause_role(c)
    assert c.query_patient("dog", "go") is None
    chips = {r["role"]: r["word"] for r in c.last_trace["roles"]}
    assert chips.get("patient") is None
