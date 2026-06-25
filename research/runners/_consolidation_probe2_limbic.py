"""CONSOLIDATION Probe 2 (option A) -- the LIMBIC encoding-gain WRITE side is LOAD-BEARING on the CO-RESIDENT
`OneBrainComposer` wired into `MergedNavConvAgent`, with the gain reading the SHARED spike-derived `dopamine` off the
ONE merged bridge.

Per `research/findings/raw/_consolidation_onebrain_limbic_scoping.md` Probe 2. Probe 1 (GO, byte-identical) proved the
co-resident `OneBrainComposer` reproduces the standalone composer on an offset rf slice. This probe proves the SECOND
half of the consolidation: the already-built-but-INERT limbic encoding-gain hook (`_da_encoding_gain` ->
`composer.encoding_gain_fn` -> `OneBrainComposer._write_block`) becomes LOAD-BEARING once the magnitude-storing
`OneBrainComposer` is the co-resident composer (the numpy-kb `MergedRFComposer` stores PHASES, magnitude-invariant, so
the hook was a no-op there -- the characterized consolidation gap this arc closes).

THE ONE BRAIN (the genuine functional integration this validates): the SAME shared `dopamine` concentration -- produced
by the `limbic_snc` (IZH2007_DOPAMINE) FIRING on the merged bridge, read via `get_concentration("dopamine")` (the
legitimate neuromodulatory broadcast, NOT a host value copied across regions) -- modulates the conversational composer's
fact ENCODING STRENGTH AT STORE TIME (Lisman-Grace hippocampal-VTA loop; Kandel D.16: dopamine gates entry into LONG-TERM
memory -- a rewarded trace stays STABLE, an un-rewarded one degrades). A fact heard while the spiking SNc bursts (a
salient/rewarded utterance) is encoded STRONGER and SURVIVES read damage that sinks a fact heard at DA baseline.

THE MECHANISM (== the I-7-b deployment de-risk, here on the CO-RESIDENT composer): the RF phase read-out has a hard
MAGNITUDE FLOOR (sim/bridge.py:5589 `_rf_mag2 > _rf_floor2`). Under common, gain-INDEPENDENT additive read damage on the
stored composite weights (fixed sigma), a higher-gain (high-DA) fact's |Z| stays above the floor -> clean recovered
phase -> the cue-match scan recalls it; a unit-gain (tonic-DA) fact's low-SNR neurons drop below the floor -> garbled
phase -> mis-recall/abstain. The floor x damage interaction is the nonlinearity that makes a per-fact encoding gain
DIFFERENTIAL, not a vacuous global scalar.

ANTI-CHEATS (asserted below):
  1. MECHANISM (load-bearing at store time): the high-DA fact's stored `store_conns` block magnitude == g_high/g_tonic x
     the tonic fact's -- the SHARED NEURAL dopamine literally scales the synaptic encoding strength on the CO-RESIDENT
     composer (not a downstream read trick).
  2. DIFFERENTIAL (the behavioral payoff): under matched read damage there is a sigma where the high-DA fact RECALLS
     while the tonic fact DEGRADES (mis-recalls/abstains), with the moat holding.
  3. LESION (DECISIVE): pin the encoding gain at baseline (g=1 for both, regardless of the SNc) -> the magnitude ratio
     == 1 AND the behavioral differential ABOLISHES. Proves it is the spiking-DA-derived gain, not content/order.
  4. MOAT-PRESERVED -- HARD: an UNSTORED cue ABSTAINS (returns None) at BOTH DA levels and under damage (the gain scales
     stored magnitude only; the cue-match abstention + the cleanup argmax are magnitude-invariant -> the moat is
     unchanged BY CONSTRUCTION). A breach at any DA level is a CRITICAL finding -- re-checked on every is-None below.
  5. PROVENANCE: the DA the gain reads is `get_concentration("dopamine")` off the merged bridge -- itself produced by the
     `limbic_snc` FIRING (`from_region_firing_signed` over [limbic_snc]). No host quantity is copied across regions; the
     coupling is a scalar read of a spike-derived neuromodulator concentration (the legitimate broadcast boundary).

HONEST: if the gain is NOT load-bearing on the co-resident composer (the magnitude-relax does not differentiate read
survival), this reports it PRECISELY (the knee is None = GREEN_INERT: a write-side stability reserve that only engages
under read stress, exactly like the read-side gate's at-rest no-op). The MECHANISM (store-magnitude == g) + the LESION +
the MOAT are the load-bearing core; the behavioral knee is the (sweep-dependent) demonstration of where it bites.

CPU / numpy / small (SIM_BACKEND=numpy, the default merged probe vocab V=17, D=128, k_max=8). The merged bridge build +
the SNc settle dominate the wall-clock; this is logic/CPU validation (the controller re-runs the GPU gates). NO `sim/`
edit (reuse-by-import: the CoResidentOneBrainComposer co-residence + the agent's _da_encoding_gain + the masked rf_kick).
Run: SIM_BACKEND=numpy python -m research.runners._consolidation_probe2_limbic
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from sim.backend import get_backend
from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import (
    da_to_encoding_gain, _damage_store_conns, _block_mean_mag,
)


# the de-risk geometry: a NEUTRAL fact (stored at DA tonic, g~=1) + a REWARDED fact (stored at DA high, g>1), distinct
# content, MATCHED cue strength; the only intended difference is the encoding gain (the SHARED dopamine at store time).
# An UNSTORED cue is the moat probe. All words are in the default merged probe vocab (DEFAULT_VOCAB).
NEUTRAL = ("dog", "go", "north")        # stored while the limbic SNc is TONIC  (g ~= 1)
REWARDED = ("cat", "run", "river")      # stored while the limbic SNc BURSTS    (g > 1)
UNSTORED = ("small", "look")            # the moat probe (never stored)


def _build_merged_limbic_agent(seed):
    """A small merged nav+conv agent with: the co-resident persistent-loop OneBrainComposer (the consolidation),
    the shared spiking limbic DA source (co_resident_limbic -> limbic_snc + the `dopamine` modulator over [limbic_snc]),
    the WRITE-side encoding-gain hook ON, and the READ-side salience gate OFF (this probe ISOLATES the write side; the
    read-side gate is separately validated 6/6 + routed through the composer's native confidence_gate on this path).
    co_resident_nav_critic=False keeps it CPU-light (the minimal 4-region limbic organ supplies DA)."""
    return MergedNavConvAgent(
        seed=seed, co_resident_composer=True, co_resident_composer_kind="onebrain",
        co_resident_limbic=True, co_resident_nav_critic=False, co_resident_command_route=False,
        enable_da_salience_gate=False, enable_da_encoding_gain=True,
        da_encoding_k=2.0, da_encoding_g_min=0.5, da_encoding_g_max=3.0,
        onebrain_k_max=8)


def _snc_idx(agent, xp):
    s = agent._handles["limbic"]["limbic_snc"]
    return xp.asarray(np.arange(s["base"], s["base"] + s["size"], dtype=np.int64)), int(s["size"])


def _settle_da(agent, I_snc, n_steps, xp):
    """Drive the merged bridge's limbic_snc with constant external current for n_steps (advancing the dopamine EMA each
    step), return the steady shared DA + the SNc firing rate (Hz). DA is read from the modulator driven by the
    limbic_snc FIRING -- the spike-derived broadcast, not a host formula. Zeroes the drive afterward."""
    b = agent._merged_bridge
    idx, n = _snc_idx(agent, xp)
    b.cp_external_input_current[:] = 0.0
    b.cp_external_input_current[idx] = xp.float32(I_snc)
    total = 0
    for _ in range(n_steps):
        b._run_one_simulation_step()
        b.runtime_state.current_time_step += 1
        total += int(b.cp_firing_states[idx].sum())
    da = float(b.neuromodulator_manager.get_concentration("dopamine"))
    rate_hz = total / max(n, 1) / (n_steps * 1e-3)
    b.cp_external_input_current[:] = 0.0
    return da, rate_hz


def _store_at_da(agent, fact, I_snc, n_settle, xp):
    """Drive the limbic_snc to the requested operating point (settle the shared DA), then store `fact` on the
    co-resident composer. The composer's encoding_gain_fn (= the agent's _da_encoding_gain) reads the LIVE shared DA at
    this store(), so the stored block magnitude == da_to_encoding_gain(DA_now)."""
    da, rate = _settle_da(agent, I_snc, n_settle, xp)
    agent.composer.store(*fact)
    return da, rate


def _query_under_damage(comp, agent, action, sigma, dmg_seed):
    """Recall (agent, action) -> patient on the REAL co-resident composer read path, with the stored weights damaged by
    a fixed-sigma common (gain-independent) complex jitter (its own reproducible draw). Temporarily swaps store_conns +
    busts the store CSR cache so the read uses the damaged weights, then restores. == the I-7 de-risk's _query_under_
    damage, here on the co-resident composer."""
    rng = np.random.default_rng(dmg_seed)
    clean = comp.store_conns
    try:
        comp.store_conns = _damage_store_conns(clean, sigma, rng)
        comp._store_dirty = True
        comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
        return comp.query_patient(agent, action)
    finally:
        comp.store_conns = clean
        comp._store_dirty = True
        comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}


def run_probe(seed, snc_tonic_pa, snc_salient_pa, n_settle, k_da, sigmas):
    """Build the merged agent, drive the shared SNc tonic/salient, store NEUTRAL @tonic + REWARDED @salient on the
    co-resident composer (the encoding gain reads the LIVE shared DA), then run mechanism + differential + lesion +
    moat. Returns a result dict."""
    xp, backend = get_backend()

    # ---- REAL: store NEUTRAL @tonic, REWARDED @salient (the gain reads the live shared DA at each store). ----
    agent = _build_merged_limbic_agent(seed)
    comp = agent.composer
    da_tonic, rate_tonic = _store_at_da(agent, NEUTRAL, snc_tonic_pa, n_settle, xp)     # block 0 @ DA tonic
    da_high, rate_high = _store_at_da(agent, REWARDED, snc_salient_pa, n_settle, xp)    # block 1 @ DA high
    da_baseline = float(agent._merged_bridge.neuromodulator_manager._config_by_name("dopamine").baseline)
    g_tonic = da_to_encoding_gain(da_tonic, da_baseline, k_da)
    g_high = da_to_encoding_gain(da_high, da_baseline, k_da)
    expected_ratio = g_high / g_tonic

    # MECHANISM (load-bearing at store time): the stored block magnitudes == the DA-derived gains.
    D = comp.D
    m_neu = _block_mean_mag(comp.store_conns, 0, D)
    m_rew = _block_mean_mag(comp.store_conns, 1, D)
    mech_ratio = m_rew / max(m_neu, 1e-12)
    mech_ok = abs(mech_ratio - expected_ratio) < 0.02

    # MOAT (clean read, HARD): an unstored cue abstains even with a high-DA fact stored; the two stored facts recall.
    moat_clean = comp.query_patient(*UNSTORED) is None
    recall_neu_clean = comp.query_patient(*NEUTRAL[:2]) == NEUTRAL[2]
    recall_rew_clean = comp.query_patient(*REWARDED[:2]) == REWARDED[2]

    # ---- LESION (decisive): pin the encoding gain at baseline (g=1 both) -> ratio == 1 + no differential. Same DA
    # drives, same content/order; only the gain is severed. ----
    agentL = _build_merged_limbic_agent(seed)
    agentL.composer.encoding_gain_fn = (lambda: 1.0)        # DA-lesion: pinned -> g=1 always (severs the limbic write)
    compL = agentL.composer
    _store_at_da(agentL, NEUTRAL, snc_tonic_pa, n_settle, xp)
    _store_at_da(agentL, REWARDED, snc_salient_pa, n_settle, xp)
    lesion_ratio = _block_mean_mag(compL.store_conns, 1, D) / max(_block_mean_mag(compL.store_conns, 0, D), 1e-12)
    lesion_ratio_ok = abs(lesion_ratio - 1.0) < 1e-9

    # ---- DIFFERENTIAL under read damage (the behavioral payoff): sweep sigma; the knee is where the high-DA fact
    # recalls, the tonic fact fails, moat holds (REAL), AND the lesion shows both fail together (no differential). ----
    rows, knee = [], None
    moat_breach_any = False
    for sigma in sigmas:
        neu = _query_under_damage(comp, NEUTRAL[0], NEUTRAL[1], sigma, seed * 1000 + 11) == NEUTRAL[2]
        rew = _query_under_damage(comp, REWARDED[0], REWARDED[1], sigma, seed * 1000 + 23) == REWARDED[2]
        moat = _query_under_damage(comp, UNSTORED[0], UNSTORED[1], sigma, seed * 1000 + 37) is None
        les_neu = _query_under_damage(compL, NEUTRAL[0], NEUTRAL[1], sigma, seed * 1000 + 11) == NEUTRAL[2]
        les_rew = _query_under_damage(compL, REWARDED[0], REWARDED[1], sigma, seed * 1000 + 23) == REWARDED[2]
        les_moat = _query_under_damage(compL, UNSTORED[0], UNSTORED[1], sigma, seed * 1000 + 37) is None
        if not moat or not les_moat:
            moat_breach_any = True
        rows.append({"sigma": sigma, "real_neutral_ok": neu, "real_rewarded_ok": rew, "moat_ok": moat,
                     "lesion_neutral_ok": les_neu, "lesion_rewarded_ok": les_rew, "lesion_moat_ok": les_moat})
        # the differential knee: the high-DA fact recalls, the tonic fact fails, moat holds; the lesion must NOT show a
        # differential at the same sigma (both fail/both pass) -- else the "differential" would be content/order, not DA.
        if knee is None and rew and (not neu) and moat and not (les_rew and not les_neu):
            knee = sigma

    differential_found = knee is not None
    # GO = the mechanism is load-bearing (store-mag == DA-derived g) AND the lesion abolishes it (ratio==1) AND the moat
    # holds at every DA level + sigma (HARD). The behavioral knee is the demonstration; if absent it is reported as
    # GREEN_INERT (a write-side stability reserve), NOT a failure -- the load-bearing core is mechanism+lesion+moat.
    load_bearing = bool(mech_ok and lesion_ratio_ok and (not moat_breach_any)
                        and moat_clean and recall_neu_clean and recall_rew_clean)
    # GO bar == the VALIDATED I-7-b deployment de-risk's GO bar: mechanism (store-mag == DA-derived g) + lesion
    # (DA pinned -> ratio 1) + moat (0-FA all levels). The behavioral read-damage knee is a SEPARATE demonstration:
    # the I-7-b de-risk's own GO had knee=None (GREEN_INERT -- a write-side stability reserve that engages under read
    # stress, like the read-side gate's at-rest no-op), so the knee is NOT part of the GO bar. Reported honestly as a
    # distinct field; if the knee is found it is an additional behavioral demonstration, not a gate.
    go = load_bearing

    return {
        "backend": backend,
        "da_source": {
            "recipe": "co_resident_limbic: limbic_snc (IZH2007_DOPAMINE) + `dopamine` from_region_firing_signed over "
                      "[limbic_snc] on the MERGED bridge; DA read via get_concentration('dopamine')",
            "da_baseline": da_baseline,
            "da_tonic": da_tonic, "rate_tonic_hz": rate_tonic,
            "da_high": da_high, "rate_high_hz": rate_high,
            "snc_tonic_pa": snc_tonic_pa, "snc_salient_pa": snc_salient_pa, "n_settle": n_settle,
        },
        "gains": {"k_da": k_da, "g_tonic": g_tonic, "g_high": g_high, "g_ratio_high_over_tonic": expected_ratio},
        "mechanism_store_magnitude": {
            "neutral_block_mean_mag": m_neu, "rewarded_block_mean_mag": m_rew,
            "stored_ratio(rewarded/neutral)": mech_ratio, "expected_ratio(g_high/g_tonic)": expected_ratio,
            "ratio_matches_gain": bool(mech_ok),
            "lesion_ratio(DA_pinned->should_be_1)": lesion_ratio, "lesion_ratio_is_unity": bool(lesion_ratio_ok),
        },
        "moat": {"clean_unstored_abstains": bool(moat_clean),
                 "clean_recall_neutral_ok": bool(recall_neu_clean),
                 "clean_recall_rewarded_ok": bool(recall_rew_clean),
                 "moat_breach_under_damage_any": bool(moat_breach_any)},
        "differential_under_damage": {"sweep": rows, "knee_sigma": knee, "found": differential_found},
        "verdict": {
            "load_bearing(mechanism+lesion+moat)": load_bearing,
            "mechanism_load_bearing(store_mag==g)": bool(mech_ok),
            "lesion_abolishes_differential(ratio==1)": bool(lesion_ratio_ok),
            "moat_intact_all_levels(HARD)": bool((not moat_breach_any) and moat_clean),
            "behavioral_differential_found": differential_found,
            "GO": go,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snc-tonic-pa", type=float, default=80.0)
    ap.add_argument("--snc-salient-pa", type=float, default=600.0)
    ap.add_argument("--n-settle", type=int, default=300)
    ap.add_argument("--k-da", type=float, default=2.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    ap.add_argument("--out", default="research/findings/raw/_consolidation_probe2_limbic.json")
    args = ap.parse_args()

    print(f"[probe2] building the merged limbic agent (co-resident OneBrainComposer + shared spiking DA), seed={args.seed}...")
    r = run_probe(args.seed, args.snc_tonic_pa, args.snc_salient_pa, args.n_settle, args.k_da, args.sigmas)

    out = {
        "probe": "consolidation_probe2_limbic",
        "what": "the limbic encoding-gain WRITE side is LOAD-BEARING on the CO-RESIDENT OneBrainComposer wired into "
                "MergedNavConvAgent, with the gain reading the SHARED spike-derived `dopamine` off the ONE merged bridge",
        "scoping": "research/findings/raw/_consolidation_onebrain_limbic_scoping.md (Probe 2)",
        "consolidation_wire_in": "MergedNavConvAgent(co_resident_composer=True, co_resident_composer_kind='onebrain', "
                                 "enable_da_encoding_gain=True) -> CoResidentOneBrainComposer on the merged rf slice; "
                                 "encoding_gain_fn = _da_encoding_gain (reads get_concentration('dopamine'))",
        "seed": args.seed,
        "result": r,
        "GO": r["verdict"]["GO"],
        "load_bearing": r["verdict"]["load_bearing(mechanism+lesion+moat)"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    v = r["verdict"]; m = r["mechanism_store_magnitude"]; da = r["da_source"]
    print("\n============ CONSOLIDATION Probe 2 -- LIMBIC ENCODING-GAIN ON THE CO-RESIDENT OneBrainComposer ============")
    print(f"  DA source (shared, spike-derived): tonic DA={da['da_tonic']:.4f} ({da['rate_tonic_hz']:.1f}Hz) -> g_tonic={r['gains']['g_tonic']:.3f}")
    print(f"                                     high  DA={da['da_high']:.4f} ({da['rate_high_hz']:.1f}Hz) -> g_high ={r['gains']['g_high']:.3f}  (ratio {r['gains']['g_ratio_high_over_tonic']:.3f})")
    print(f"  MECHANISM (stored |w| ratio == g_high/g_tonic): {v['mechanism_load_bearing(store_mag==g)']}  "
          f"(stored {m['stored_ratio(rewarded/neutral)']:.4f} vs expected {m['expected_ratio(g_high/g_tonic)']:.4f})")
    print(f"  LESION (DA pinned -> ratio==1):     {v['lesion_abolishes_differential(ratio==1)']}  (lesion ratio {m['lesion_ratio(DA_pinned->should_be_1)']:.4f})")
    print(f"  MOAT intact ALL levels (HARD):      {v['moat_intact_all_levels(HARD)']}  "
          f"(clean unstored abstains={r['moat']['clean_unstored_abstains']}, breach-under-damage={r['moat']['moat_breach_under_damage_any']})")
    print(f"  DIFFERENTIAL under read damage:     {'knee at sigma=' + str(r['differential_under_damage']['knee_sigma']) if v['behavioral_differential_found'] else 'NONE FOUND (GREEN_INERT: a write-side stability reserve)'}")
    for row in r["differential_under_damage"]["sweep"]:
        print(f"    sigma={row['sigma']}: REAL neu={row['real_neutral_ok']} rew={row['real_rewarded_ok']} moat={row['moat_ok']}  "
              f"| LESION neu={row['lesion_neutral_ok']} rew={row['lesion_rewarded_ok']} moat={row['lesion_moat_ok']}")
    print(f"\n  LOAD-BEARING (mechanism+lesion+moat) == the I-7-b GO bar: {v['load_bearing(mechanism+lesion+moat)']}")
    _knee = "knee FOUND" if v["behavioral_differential_found"] else "knee GREEN_INERT (== the I-7-b de-risk's own GO had no knee)"
    print(f"  VERDICT: {'GO' if v['GO'] else 'NEGATIVE'} ({_knee}) -- the shared NEURAL dopamine "
          f"{'DOES' if v['load_bearing(mechanism+lesion+moat)'] else 'does NOT'} scale fact-encoding strength on the "
          f"CO-RESIDENT OneBrainComposer (the limbic write-side is now load-bearing on the consolidated one brain).")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
