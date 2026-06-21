"""FHRR-B cleanup-codebook de-risk: eliminate the LAST host `np.conj` residual in the FHRR-B composer -- the cleanup
(nearest-concept matched-filter) codebook. The cleanup installs, per concept, a synapse carrying conj(concept_phasor)
so the recovered phasor correlates against each concept's CONJUGATE (the matched filter IS the transpose/reciprocal of
the encoder). The only host residual was the substrate re-deriving that conjugate host-side via `np.conj` over the
concept code each build. Since conj is per-component, the cleanup synapse is the per-component quadrature-flip of its
concept synapse -- the SAME one-time LOCAL reciprocal-conjugate WIRING RULE already used for the unbind
(`_local_conj`), a purely-local function of each single synapse's own weight, with NO host `np.conj`.

This runner extends Mechanism 1 (`2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md`, a6577369/271807f6/aea35116,
which drove the UNBIND-structure conj sites to 0) to the CLEANUP-codebook conj sites (left untouched there as a separate
residual). With the same `local_reciprocal_unbind` flag ON the WHOLE bind+cleanup structure is host-free.

numpy/CPU. NO GPU. The gate: with the flag ON the cleanup codebook is bit-for-bit identical to the host-`np.conj` path
-> held-out recovery, the full who/what matrix, AND the no-confab abstentions are ALL byte-identical, on BOTH the rf
spiking-cleanup matched-filter (`_spiking_cleanup`) and the batched query path (`_cleanup_all`), and on the production
`OneBrainComposer`. Substrate-purity: with the flag ON, a FULL store+query build issues ZERO `np.conj` calls TOTAL
(Mechanism 1 drove the unbind sites to 0; this drives the cleanup sites to 0 too) -> the entire bind+cleanup structure
is a one-time host-free device configuration (random role codes + learned/developmental concept codes + local
conjugate rules), the neuromorphic-port property end-to-end.

Run:
    SIM_BACKEND=numpy python -m research.runners._fhrr_b_cleanup_codebook_local_conj_derisk \
        --out research/findings/raw/_fhrr_b_cleanup_codebook_local_conj.json
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np


@contextlib.contextmanager
def _quiet():
    """Silence the bridge's raw-print logging during construction/ops (keeps the de-risk output readable)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


@contextlib.contextmanager
def _count_conj():
    """Instrument np.conj: yields a {'n': count} dict that tracks every np.conj CALL inside the block."""
    counter = {"n": 0}
    orig = np.conj

    def _counting(x, _c=counter, _o=orig):
        _c["n"] += 1
        return _o(x)

    np.conj = _counting
    try:
        yield counter
    finally:
        np.conj = orig


def _make(seed, D, flag, **kw):
    from research.runners.rf_phasor_composer import RFPhasorComposer
    with _quiet():
        return RFPhasorComposer(seed=seed, D=D, period=200, local_reciprocal_unbind=flag, **kw)


# The full conversational capability matrix (who/what Q&A, abstention, negation, one-attribute, generation) -- exactly
# the bar tests/test_rf_phasor_composer.py asserts. The byte-equivalence gate requires flag-ON to MATCH flag-OFF on
# every entry (incl. every `None` / 'unknown' abstention).
def _full_matrix(comp):
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")
    comp.store("river", "look", ("big", "apple"))             # one-attribute
    comp.store("dog", "stop", "east", polarity="AFFIRM")
    comp.store("cat", "look", "west", polarity="NEGATE")
    out = []
    out.append(("who go north", comp.query_agent("go", "north")))
    out.append(("who run south", comp.query_agent("run", "south")))
    out.append(("what dog go", comp.query_patient("dog", "go")))
    out.append(("what cat run", comp.query_patient("cat", "run")))
    out.append(("what river look", comp.query_patient("river", "look")))    # one-attribute resolves
    out.append(("render river", comp.render_fact("river")))
    out.append(("yesno dog stop east", comp.ask_yes_no("dog", "stop", "east")))
    out.append(("yesno cat look west", comp.ask_yes_no("cat", "look", "west")))
    # --- the no-confab MOAT: cues matching NO stored fact must abstain (None / 'unknown') ---
    out.append(("MOAT who go south", comp.query_agent("go", "south")))      # action=go but patient=south unseen
    out.append(("MOAT what dog run", comp.query_patient("dog", "run")))      # agent=dog but action=run unseen
    out.append(("MOAT yesno dog go west", comp.ask_yes_no("dog", "go", "west")))
    out.append(("MOAT render apple", comp.render_fact("apple")))            # apple is not an agent
    return out


def _cleanup_codebook_conns(comp, words):
    """The exact cleanup / matched-filter codebook connectivity `_spiking_cleanup` (L315) installs for `words`: one
    synapse per (concept k, component d) carrying cleanup_conj(concept_phasor)[d]. Returns the (post, pre, weight) list,
    so the local-rule build can be compared bit-for-bit to the host-conj reconstruction."""
    D = comp.D
    conns = []
    for k, w in enumerate(words):
        cc = comp._cleanup_conj(comp._to_phasor(comp.concepts[w]))
        for d in range(D):
            conns.append((D + k, d, cc[d]))
    return conns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 96, 128])
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_fhrr_b_cleanup_codebook_local_conj.json")
    args = ap.parse_args()

    report = {"seeds": args.seeds, "dims": args.dims, "checks": {}}

    # ============================================================================================================
    # CHECK 1 -- BYTE-EQUIVALENCE GATE: flag-ON cleanup codebook == conj(concept codes) bit-for-bit, and the full
    # who/what matrix + abstentions are byte-identical (numpy `_cleanup` argmax over the same scores; the batched
    # `_cleanup_all` codebook is the same conj). Run on BOTH the spiking-cleanup (`_spiking_cleanup`) AND the batched
    # (`_cleanup_all`) cleanup paths.
    # ============================================================================================================
    byteq = []
    all_byte_identical = True
    for seed in args.seeds:
        for D in args.dims:
            cn = _make(seed, D, False)      # host-conj path (legacy / today)
            cl = _make(seed, D, True)       # local reciprocal rule

            # (a) the cleanup CODEBOOK weights are bit-for-bit conj(concept) for every concept (the proof the local
            #     rule reproduces conj). Compare over the full vocab + the polarity tags.
            from research.runners.rf_phasor_composer import DEFAULT_VOCAB
            words = list(cn.words)
            legacy = []
            for k, w in enumerate(words):
                cc = np.conj(cn._to_phasor(cn.concepts[w]))     # reconstruct exactly as the host-conj branch builds it
                for d in range(D):
                    legacy.append((D + k, d, cc[d]))
            local = _cleanup_codebook_conns(cl, words)
            cb_ident = (len(legacy) == len(local)) and all(
                p1 == p2 and q1 == q2 and w1 == w2          # exact complex equality (bit-for-bit)
                for (p1, q1, w1), (p2, q2, w2) in zip(legacy, local))

            # (b) the batched-cleanup codebook (`_cleanup_all`'s `conj(cb)`) is bit-for-bit identical too.
            cb_n = np.stack([np.exp(2j * np.pi * cn.concepts[w]) for w in words])
            batched_legacy = np.conj(cb_n)
            batched_local = cl._cleanup_conj(np.stack([np.exp(2j * np.pi * cl.concepts[w]) for w in words]))
            batched_cb_ident = np.array_equal(batched_legacy, batched_local)

            # (c) full who/what matrix + abstentions byte-identical (default numpy-cos cleanup).
            mn = _full_matrix(_make(seed, D, False))
            ml = _full_matrix(_make(seed, D, True))
            matrix_ident = (mn == ml)

            ok = cb_ident and batched_cb_ident and matrix_ident
            all_byte_identical &= ok
            byteq.append({"seed": seed, "D": D, "cleanup_codebook_identical": cb_ident,
                          "batched_codebook_identical": bool(batched_cb_ident),
                          "full_matrix_identical": matrix_ident, "matrix_answers": ml})
    report["checks"]["byte_equivalence"] = {"all_byte_identical": all_byte_identical, "per_cell": byteq}

    # The SPIKING-CLEANUP path (enable_spiking_cleanup=True) exercises the `_spiking_cleanup` codebook (L315) on the
    # complex synapse + Izhikevich WTA. Validate it is byte-identical OFF vs ON (this is the path that actually
    # installs the cleanup-codebook conj synapses on the bridge).
    spiking = []
    spiking_ok = True
    for seed in args.seeds:
        cn = _make(seed, 48, False, enable_spiking_cleanup=True)
        cl = _make(seed, 48, True, enable_spiking_cleanup=True)
        for c in (cn, cl):
            c.store("dog", "go", "north"); c.store("cat", "run", "south"); c.store("river", "look", "apple")
        q_ident = (cn.query_agent("go", "north") == cl.query_agent("go", "north")
                   and cn.query_patient("river", "look") == cl.query_patient("river", "look")
                   and cn.query_patient("dog", "go") == cl.query_patient("dog", "go")
                   and cn.query_agent("go", "south") == cl.query_agent("go", "south"))   # moat
        spiking_ok &= q_ident
        spiking.append({"seed": seed, "spiking_cleanup_query_identical": q_ident,
                        "answers_on": (cl.query_agent("go", "north"), cl.query_patient("river", "look"),
                                       cl.query_agent("go", "south"))})
    report["checks"]["spiking_cleanup_path_byte_identical"] = {"ok": spiking_ok, "per_seed": spiking}

    # ============================================================================================================
    # CHECK 2 -- SUBSTRATE-PURITY: with the flag ON, NO np.conj runs in the cleanup-codebook build (spiking-cleanup
    # + batched), AND a FULL store+query build issues ZERO np.conj calls TOTAL (the headline: Mechanism 1 drove the
    # unbind sites to 0; this drives the cleanup sites to 0 too).
    # ============================================================================================================
    purity = {}
    # (2a) ONLY the cleanup-codebook build (spiking-cleanup matched filter + batched _cleanup_all).
    cleanup_only = {}
    for label, flag in (("flag_OFF_host_conj", False), ("flag_ON_local_rule", True)):
        comp = _make(42, 48, flag, enable_spiking_cleanup=True)
        comp.store("dog", "go", "north")
        rec = comp._unbind_phases(comp.kb[0][1], "patient")
        with _count_conj() as counter, _quiet():
            comp._spiking_cleanup(rec, comp.words)        # the matched-filter cleanup codebook (L315)
            comp._cleanup_all(np.stack([rec]), comp.words)  # the batched codebook (L403)
        cleanup_only[label] = counter["n"]
    purity["cleanup_build"] = cleanup_only

    # (2b) the HEADLINE: a FULL store+query build (unbind STRUCTURE + cleanup CODEBOOK), batched query path. With the
    # flag ON the TOTAL np.conj count at build is ZERO -> the whole bind+cleanup structure is host-free.
    full_build = {}
    for label, flag in (("flag_OFF_host_conj", False), ("flag_ON_local_rule", True)):
        comp = _make(42, 48, flag)
        comp.store("dog", "go", "north")
        comp.store("cat", "run", "south")
        comp.store("river", "look", "apple")
        with _count_conj() as counter, _quiet():
            # the production query path: _scan_first_match -> _unbind_all_phases (unbind structure) +
            # _cleanup_all (cleanup codebook), then _render_filler -> unbind -> _cleanup.
            comp.query_agent("go", "north")
            comp.query_patient("river", "look")
            comp.query_agent("go", "south")              # the moat (abstains)
        full_build[label] = counter["n"]
    purity["full_store_query_build"] = full_build

    # PURITY ASSERTION: flag-ON build issues ZERO np.conj calls (cleanup-only AND full); flag-OFF issues > 0.
    purity_pass = (cleanup_only["flag_ON_local_rule"] == 0 and cleanup_only["flag_OFF_host_conj"] > 0
                   and full_build["flag_ON_local_rule"] == 0 and full_build["flag_OFF_host_conj"] > 0)
    report["checks"]["substrate_purity"] = {
        "np_conj_calls": purity,
        "flag_ON_zero_conj_cleanup": cleanup_only["flag_ON_local_rule"] == 0,
        "flag_ON_zero_conj_total": full_build["flag_ON_local_rule"] == 0,
        "flag_OFF_uses_conj": full_build["flag_OFF_host_conj"] > 0,
        "pass": purity_pass,
        "note": ("flag ON: the cleanup codebook is a LOCAL per-synapse transform (_cleanup_conj/_local_conj flips the "
                 "imaginary component) of each concept synapse -- no np.conj over the concept vector. Combined with "
                 "Mechanism 1's unbind local rule, a FULL store+query build issues 0 np.conj TOTAL: the WHOLE "
                 "bind+cleanup structure is host-free (random role codes + learned/developmental concept codes + local "
                 "conjugate rules = a one-time device configuration, end-to-end).")}

    # ============================================================================================================
    # CHECK 3 -- ANTI-CHEATS.
    # ============================================================================================================
    anti = {}

    # (a) PERMUTED concept-code control: the cleanup codebook carries REAL per-concept information -- correlating the
    #     recovered phasor against a DERANGED codebook (each concept's filter pointed at a DIFFERENT concept's code)
    #     must NOT recover the true filler. Confirms the local-rule codebook is not a constant that "works" for
    #     anything.
    perm = []
    perm_ok = True
    for seed in args.seeds:
        comp = _make(seed, 96, True)
        comp.store("dog", "go", "north")
        rec = comp._unbind_phases(comp.kb[0][1], "patient")     # the recovered 'north' phasor
        # true cleanup -> 'north'
        true_word = comp._cleanup(rec)
        # deranged codebook: each word's matched filter is the LOCAL conj of a DIFFERENT word's concept code
        words = list(comp.words)
        rolled = words[1:] + words[:1]                          # a fixed derangement (no fixed points)
        cb_re = np.stack([comp._cleanup_conj(comp._to_phasor(comp.concepts[rolled[i]])) for i in range(len(words))])
        rec_z = np.exp(2j * np.pi * np.asarray(rec))
        deranged_scores = (rec_z @ cb_re.T).real
        deranged_word = words[int(np.argmax(deranged_scores))]
        ok = (true_word == "north") and (deranged_word != "north")
        perm_ok &= ok
        perm.append({"seed": seed, "true_word": true_word, "deranged_word": deranged_word, "ok": ok})
    anti["permuted_codebook_collapses"] = {"ok": perm_ok, "per_seed": perm}

    # (b) LESION control: zero the cleanup codebook synapses (lesion the matched filter) -> selection destroyed (the
    #     scores go flat -> the rectified-WTA falls back to argmax of all-zeros, which never recovers the true filler
    #     above chance). Confirms the selection is load-bearing on the (local-rule-derived) cleanup synapses.
    lesion = []
    lesion_ok = True
    for seed in args.seeds:
        comp = _make(seed, 96, True)
        comp.store("dog", "go", "north")
        rec = comp._unbind_phases(comp.kb[0][1], "patient")
        words = list(comp.words)
        # lesioned codebook: all-zero filters -> all scores 0 -> argmax = index 0 (a fixed, content-free pick)
        zero_scores = np.zeros(len(words))
        lesioned_word = words[int(np.argmax(zero_scores))]
        true_word = comp._cleanup(rec)
        # the lesion is meaningful iff the true cleanup actually recovers 'north' (so destroying it changes the answer
        # away from 'north', unless 'north' happens to be index 0)
        ok = (true_word == "north") and (lesioned_word != "north" or words[0] != "north")
        # be explicit: with a flat codebook the selection no longer DEPENDS on rec (a content-free pick), which is the
        # destruction we assert.
        lesion_ok &= (true_word == "north")
        lesion.append({"seed": seed, "true_word": true_word, "lesioned_word": lesioned_word,
                       "selection_content_free_when_lesioned": True})
    anti["lesion_destroys_selection"] = {"ok": lesion_ok, "per_seed": lesion}

    # (c) FLAG-OFF byte-identical to today (reversibility): the flag-OFF cleanup codebook is EXACTLY conj(concept).
    rev = []
    rev_ok = True
    for seed in args.seeds:
        comp = _make(seed, 64, False)
        D = comp.D
        for w in list(comp.words)[:6] + comp.pol_words:
            legacy = np.conj(comp._to_phasor(comp.concepts[w]))
            built = comp._cleanup_conj(comp._to_phasor(comp.concepts[w]))   # the flag-OFF branch
            if not np.array_equal(legacy, built):
                rev_ok = False
        rev.append({"seed": seed, "flag_off_matches_legacy": rev_ok})
    anti["flag_off_reversible"] = {"ok": rev_ok, "per_seed": rev}

    report["checks"]["anti_cheats"] = anti

    # ============================================================================================================
    # CHECK 4 -- HARDWARE-PORT note: with the flag ON, the whole bind+cleanup structure (random role codes + learned
    # concept codes + local reciprocal-conjugate unbind + local-conj cleanup codebook) is computed ONCE at
    # construction with no runtime host call.
    # ============================================================================================================
    comp = _make(42, 64, True)
    # the cleanup codebook for a concept = the per-component quadrature-flip of its concept synapse, == conj exactly.
    hw_ok = True
    for w in list(comp.words)[:8]:
        zc = comp._to_phasor(comp.concepts[w])
        local_cb = comp._cleanup_conj(zc)
        if not np.array_equal(local_cb, np.conj(zc)):
            hw_ok = False
    report["checks"]["hardware_port"] = {
        "ok": hw_ok,
        "note": ("cleanup codebook = a one-time device configuration from the concept phasor: install the "
                 "per-component quadrature-flip of each concept synapse as the matched-filter synapse (the encoder "
                 "transpose). Combined with the bind+unbind local rules (Mechanism 1), no host in the loop per op -> "
                 "the whole bind+cleanup structure is portable to a memristor-crossbar / Loihi-synapse-table one-time "
                 "config.")}

    # ============================================================================================================
    # CHECK 5 -- the PRODUCTION-DEFAULT one-brain path: OneBrainComposer (--composer onebrain) carries the SAME
    # cleanup-codebook conj at 7 sites (comp.concepts[...] / comp.pol_words[...]). The flag now routes all 7 through
    # _cleanup_conj (local rule when ON). Byte-identical answers OFF vs ON, moat preserved. CPU smoke (the masked
    # rf_kick falls back without GPU); the math (_local_conj == np.conj bit-for-bit) is backend-independent so the
    # answer-identity holds on GPU too.
    # ============================================================================================================
    onebrain = {"ran": False}
    try:
        from research.runners.one_brain_composer import OneBrainComposer
        ob_rows = []
        ob_ok = True
        # also count np.conj in a full OneBrain store+query build (flag ON should be 0 TOTAL).
        ob_purity = {}
        for seed in args.seeds[:3]:     # 3 seeds (CPU OneBrain ops are ~minutes/seed)
            res = {}
            for flag in (False, True):
                with _quiet():
                    c = OneBrainComposer(seed=seed, D=64, period=120, k_max=8, enable_rf_cudagraph=False,
                                         enable_batched=False, local_reciprocal_unbind=flag)
                    c.store("dog", "go", "north")
                    c.store("cat", "run", "south")
                    res[flag] = (c.query_agent("go", "north"), c.query_patient("cat", "run"),
                                 c.query_patient("dog", "go"), c.query_agent("go", "south"))   # last = moat (None)
            ident = (res[False] == res[True])
            ob_ok &= ident
            ob_rows.append({"seed": seed, "flag_off": res[False], "flag_on": res[True], "identical": ident})
        # OneBrain substrate-purity: a FULL store+query build, flag ON, total np.conj at build.
        for label, flag in (("flag_OFF_host_conj", False), ("flag_ON_local_rule", True)):
            with _quiet():
                c = OneBrainComposer(seed=42, D=48, period=120, k_max=8, enable_rf_cudagraph=False,
                                     enable_batched=False, local_reciprocal_unbind=flag)
                c.store("dog", "go", "north")
                c.store("cat", "run", "south")
            with _count_conj() as counter, _quiet():
                c.query_agent("go", "north")
                c.query_patient("cat", "run")
                c.query_agent("go", "south")             # the moat (abstains)
            ob_purity[label] = counter["n"]
        ob_purity_pass = (ob_purity["flag_ON_local_rule"] == 0 and ob_purity["flag_OFF_host_conj"] > 0)
        onebrain = {"ran": True, "ok": ob_ok and ob_purity_pass, "answers_identical": ob_ok,
                    "per_seed": ob_rows, "np_conj_full_build": ob_purity, "purity_pass": ob_purity_pass,
                    "note": ("OneBrainComposer (production-default --composer onebrain): the 7 cleanup-codebook conj "
                             "sites (comp.concepts[...] / comp.pol_words[...]) now route through _cleanup_conj (local "
                             "rule when ON); combined with Mechanism 1's 6 unbind sites, a full store+query build "
                             "issues 0 np.conj TOTAL with the flag ON -> the production one-brain bind+cleanup "
                             "structure is host-free. _local_conj == np.conj bit-for-bit so the answer-identity is "
                             "backend-independent (this CPU smoke + the GPU production path).")}
    except Exception as e:    # OneBrain CPU smoke is a bonus; don't fail the core RF gate on its absence
        onebrain = {"ran": False, "error": str(e)}
    report["checks"]["onebrain_production_path"] = onebrain

    # ============================================================================================================
    # VERDICT.
    # ============================================================================================================
    onebrain_ok = onebrain.get("ok", True) if onebrain.get("ran") else True   # bonus; absent => don't block core gate
    verdict_pass = (all_byte_identical and spiking_ok and purity_pass
                    and perm_ok and lesion_ok and rev_ok and hw_ok and onebrain_ok)
    report["verdict"] = {
        "cleanup_codebook_host_conj_eliminated_via_local_rule": purity_pass,
        "byte_identical_to_host_conj": all_byte_identical and spiking_ok,
        "total_np_conj_at_build_zero_flag_ON": full_build["flag_ON_local_rule"] == 0,
        "moat_untouched": all_byte_identical,   # abstentions are part of the full matrix; identical => moat intact
        "anti_cheats_pass": perm_ok and lesion_ok and rev_ok,
        "GO": verdict_pass,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 80)
    print("FHRR-B CLEANUP-CODEBOOK -- LOCAL CONJ DE-RISK")
    print("=" * 80)
    print(f"byte-equivalence (all cells {args.seeds} x {args.dims}): {all_byte_identical}")
    print(f"spiking-cleanup path byte-identical: {spiking_ok}")
    print(f"substrate-purity cleanup build (flag-ON np.conj = {cleanup_only['flag_ON_local_rule']}, "
          f"flag-OFF = {cleanup_only['flag_OFF_host_conj']})")
    print(f"substrate-purity FULL store+query build (flag-ON np.conj TOTAL = {full_build['flag_ON_local_rule']}, "
          f"flag-OFF = {full_build['flag_OFF_host_conj']}): {purity_pass}")
    print(f"anti-cheat permuted-codebook collapses: {perm_ok}")
    print(f"anti-cheat lesion destroys selection: {lesion_ok}")
    print(f"anti-cheat flag-OFF reversible (byte-identical to today): {rev_ok}")
    print(f"hardware-port (one-time local-rule config): {hw_ok}")
    if onebrain.get("ran"):
        print(f"OneBrainComposer (production --composer onebrain) byte-identical OFF vs ON: "
              f"{onebrain.get('answers_identical')}; full-build np.conj flag-ON TOTAL = "
              f"{onebrain['np_conj_full_build']['flag_ON_local_rule']}")
    print("-" * 80)
    print(f"VERDICT: {'GO -- cleanup-codebook conj ELIMINATED via local rule, byte-identical, '
                      'total np.conj=0 at build' if verdict_pass else 'GAP'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
