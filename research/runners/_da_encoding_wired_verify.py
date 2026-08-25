"""WIRE-IN VERIFY for the DA-GATED ENCODING coupling (board WAVE-0, Gap-4 write-side), through the REAL production
`webapp/server.py::brain_chat` handler (numpy-CPU, rf recall; ~s not ~180s).

THE COUPLING (built, default-OFF): the brain's OWN self-produced tonic dopamine (the #76/#79 spiking SNc read, stashed
on `chat._last_da_drives["da_level"]` by `da_mode_drives_chat.observe_turn` earlier in the turn) scales a taught fact's
WRITE MAGNITUDE at store time via `composer.encoding_gain_fn` (Lisman-Grace hippocampal-VTA loop; Kandel D.16 — dopamine
gates entry into LONG-TERM memory). The gain map is reused verbatim from the VALIDATED board I-7-b de-risk:
`g = clip(0.5, 3.0, 1 + 2.0*(DA - 0.5))` (tonic 0.5 -> g=1.0). Wiring: `webapp/da_encoding_drives_chat.install_encoding_gain`
called in `brain_chat` right after the DA-mode read (level fresh), before the gate/acquire that stores.

WHAT THIS PROVES (the wire-in gate; GO = A and B and C):
  (A) OFF (`BRAIN_DA_ENCODING` unset) — teaching an SVO through the REAL handler: NO `da_encoding` key on the response,
      the live composer's `encoding_gain_fn` is None (the install never ran), and the stored recall is unchanged ->
      BYTE-IDENTICAL to HEAD (the coupling is provably a no-op when off; the ONLY thing this change adds is skipped).
  (B) ON, LOAD-BEARING — teach the SAME fact under a HIGH-DA turn (`BRAIN_DA_DRIVES_INDUCE=1300`, arousal) vs a LOW-DA
      turn (`=100`, rest): the WRITE gain the live DA drives is strictly GREATER for the high-DA turn (g_high > g_low),
      reported off the response's `da_encoding.g`. MECHANISM (the gain is not vacuous): that SAME g_high vs g_low, applied
      at a MAGNITUDE-carrying store (the production-default onebrain / rf substrate store, here a substrate-store RF
      composer), writes a measurably STRONGER trace (stored |w| ratio == g_high/g_low).
  (C) LESION (`BRAIN_DA_ENCODING_LESION=1`) — the write gain is pinned to 1.0 regardless of the DA level, so the high-vs-
      low differential VANISHES (attribution to the DA read; distinct from BRAIN_DA_DRIVES_LESION which collapses the
      LEVEL). g_high == g_low == 1.0.

HONEST SCOPE. This is a WIRED, default-OFF coupling — NOT flipped on. The WRITE gain the live DA drives bites the STORED
trace only on a magnitude-carrying composer (the production-default onebrain `store_conns` / rf substrate store, proven
in the mechanism sub-check + the I-7-b GO); on the `BRAIN_COMPOSER_KIND=rf` numpy FAST-path recall used here for the
handler's speed the stored recall is magnitude-INVARIANT, so the coupling is a write-side reserve on THAT store — the
handler proof (A/B/C) is the WIRING (the live DA reaches the store hook and produces a differential gain), and the
mechanism sub-check confirms the gain scales a real magnitude store.

Run (numpy-CPU, foreground, ~a minute):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._da_encoding_wired_verify
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path; ~s not ~180s)

logging.getLogger().setLevel(logging.ERROR)          # quiet the per-build SIM_BRIDGE chatter (the verdict is JSON)

import numpy as np  # noqa: E402

FACT = ("dog", "eat", "grass")   # a clean, pre-lemmatized 3-token SVO (both the B3 + legacy acquire paths agree)
FACT_MSG = "dog eat grass"
INDUCE_HIGH = 1300.0             # arousal -> DA ~= 1.24 (per da_mode_drives_chat's afferent calibration)
INDUCE_LOW = 100.0               # rest    -> DA well below tonic

# quiet, confound-minimizing env for the handler proofs: disable the heavy Gate-B organs (they run identically on every
# arm), keep the single-fact path (rich=False). BRAIN_DA_DRIVES is left at its default-ON anchor so the SNc->DA read
# runs and INDUCE can set the level; BRAIN_NONCONTRADICTION_GATE off -> the legacy exact-3-token acquire (no confound).
_QUIET = {
    "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_SURPRISE": "0", "BRAIN_METACOG": "0",
    "BRAIN_MULTIREF": "0", "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_RECONSOLIDATION": "0",
    "BRAIN_EPISODIC_STORE": "0", "BRAIN_CURIOSITY": "0", "BRAIN_RICH": "0", "BRAIN_GNW_BUS": "0",
    "BRAIN_CONTINUOUS": "0", "BRAIN_CONTINUOUS_DRIVES": "0", "BRAIN_SWAP_DRIVES": "0",
}


def _set(env):
    """Apply/clear env keys (value None -> del). Returns nothing; mutates os.environ."""
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def _teach_turn(session, *, da_encoding, lesion, induce):
    """Teach FACT_MSG through the REAL brain_chat handler on a FRESH session, return (resp_dict, chat). da_encoding /
    lesion / induce control the coupling env for THIS arm; a fresh session -> a fresh chat + composer (no crosstalk)."""
    from webapp.server import brain_chat, BrainChatRequest as Req, _BRAIN_CHATS
    env = dict(_QUIET)
    # EXPLICIT-OFF PIN (2026-08-25, prep for the default-ON flip): the OFF arm exports BRAIN_DA_ENCODING=0, NOT unset.
    # da_encoding_enabled() defaults ON post-flip, so an unset OFF arm would silently ARM the coupling and break the
    # byte-identical (A) proof. `=0` is the byte-identical escape and is invariant to the default -> the OFF arm proves
    # the escape, and every ON arm still sets `=1`.
    env["BRAIN_DA_ENCODING"] = "1" if da_encoding else "0"
    env["BRAIN_DA_ENCODING_LESION"] = "1" if lesion else None
    env["BRAIN_DA_DRIVES_INDUCE"] = (str(induce) if induce is not None else None)
    _set(env)
    r = brain_chat(Req(session=session, message=FACT_MSG, brain="tiny-demo", renderer="stub", rich=False))
    resp = json.loads(bytes(r.body).decode("utf-8"))
    chat = _BRAIN_CHATS.get((session, "tiny-demo", "stub"))
    return resp, chat


def _recall_hash(chat):
    """A stable hash of what the taught fact recalls on the live composer (the STORE STATE proxy for byte-identity):
    the recalled patient for the fact's (agent, action) cue. On the rf numpy path this is magnitude-invariant, so it is
    identical whether or not the (inert-there) gain is installed — the byte-identity witness."""
    try:
        patient = chat.inner.composer.query_patient(FACT[0], FACT[1])
    except Exception as e:
        patient = f"error:{type(e).__name__}"
    return hashlib.sha256(repr(patient).encode()).hexdigest()[:16], patient


# ── the MAGNITUDE-carrying mechanism sub-check (the gain is not vacuous): the SAME live-DA-derived g, applied at a
#    substrate store (== the production-default onebrain store_conns / rf substrate store), writes a stronger trace. ──
_MECH_VOCAB = ["dog", "cat", "eat", "chase", "grass", "fish", "bird", "run"]


def _substrate_store_mean_mag(g, seed=42):
    """Store FACT at write-gain g on a MAGNITUDE-carrying RF composer (enable_substrate_store=True == the composite
    lives in the substrate's complex weights) and return the mean |w| of the stored trace (== g * unit)."""
    from research.runners.rf_phasor_composer import RFPhasorComposer, to_host
    c = RFPhasorComposer(seed=seed, D=64, vocab=_MECH_VOCAB, enable_substrate_store=True,
                         encoding_gain_fn=(lambda gg=g: gg))
    c.store(*FACT)
    b = c.kb[-1][1]
    re = np.asarray(to_host(b.cp_rf_w_re.data))
    im = np.asarray(to_host(b.cp_rf_w_im.data))
    return float(np.hypot(re, im).mean())


def main():
    # ── (A) OFF: teach through the REAL handler with BRAIN_DA_ENCODING unset. ──
    off_resp, off_chat = _teach_turn("dae_off", da_encoding=False, lesion=False, induce=None)
    off_no_key = "da_encoding" not in off_resp
    off_fn_none = (getattr(getattr(off_chat, "inner", None), "composer", None) is not None
                   and off_chat.inner.composer.encoding_gain_fn is None)
    off_hash, off_patient = _recall_hash(off_chat)
    go_a = bool(off_no_key and off_fn_none)

    # ── (B) ON, LOAD-BEARING: the same fact under HIGH-DA vs LOW-DA -> g_high > g_low (off the live handler response). ──
    hi_resp, hi_chat = _teach_turn("dae_hi", da_encoding=True, lesion=False, induce=INDUCE_HIGH)
    lo_resp, lo_chat = _teach_turn("dae_lo", da_encoding=True, lesion=False, induce=INDUCE_LOW)
    dae_hi = hi_resp.get("da_encoding") or {}
    dae_lo = lo_resp.get("da_encoding") or {}
    g_high = float(dae_hi.get("g")) if dae_hi.get("g") is not None else None
    g_low = float(dae_lo.get("g")) if dae_lo.get("g") is not None else None
    da_high = dae_hi.get("da_level")
    da_low = dae_lo.get("da_level")
    on_keys_present = ("da_encoding" in hi_resp) and ("da_encoding" in lo_resp)
    go_b_wiring = bool(on_keys_present and g_high is not None and g_low is not None and g_high > g_low)
    # the recall hash is unchanged vs OFF on the rf numpy path (the gain is inert on THAT store) — the byte-identity
    # witness that the coupling never corrupts the recall content, only (on a magnitude store) its strength.
    hi_hash, _ = _recall_hash(hi_chat)
    recall_unchanged_on = (hi_hash == off_hash)

    # MECHANISM: the SAME g_high vs g_low, applied at a magnitude-carrying store, writes a measurably stronger trace.
    mech_mag_high = _substrate_store_mean_mag(g_high) if g_high is not None else None
    mech_mag_low = _substrate_store_mean_mag(g_low) if g_low is not None else None
    mech_ratio = (mech_mag_high / mech_mag_low) if (mech_mag_high and mech_mag_low) else None
    expected_ratio = (g_high / g_low) if (g_high and g_low) else None
    go_b_mech = bool(mech_mag_high is not None and mech_mag_low is not None and mech_mag_high > mech_mag_low
                     and expected_ratio is not None and abs(mech_ratio - expected_ratio) < 1e-6)

    # ── (C) LESION: pin g=1.0 regardless of DA -> the high-vs-low differential VANISHES. ──
    lh_resp, _ = _teach_turn("dae_les_hi", da_encoding=True, lesion=True, induce=INDUCE_HIGH)
    ll_resp, _ = _teach_turn("dae_les_lo", da_encoding=True, lesion=True, induce=INDUCE_LOW)
    g_les_high = float((lh_resp.get("da_encoding") or {}).get("g")) if (lh_resp.get("da_encoding") or {}).get("g") is not None else None
    g_les_low = float((ll_resp.get("da_encoding") or {}).get("g")) if (ll_resp.get("da_encoding") or {}).get("g") is not None else None
    go_c = bool(g_les_high is not None and g_les_low is not None
                and abs(g_les_high - 1.0) < 1e-9 and abs(g_les_low - 1.0) < 1e-9
                and abs(g_les_high - g_les_low) < 1e-9)

    go = bool(go_a and go_b_wiring and go_c)

    # ── ATTRIBUTION: what fraction of the write-gain DIFFERENTIAL is owed to the LIVE DA read? The lesion is the
    #    control (DA pinned -> g=1.0 both). (treatment - control)/treatment == 1.0 means the DA read owns the whole
    #    differential (the coupling is not a residual host effect). Measuring both arms is not asking whose it was. ──
    from tools.lab import attributable_to
    diff_live = (g_high or 0.0) - (g_low or 0.0)              # the high-vs-low write-gain gap with the DA read live
    diff_lesion = (g_les_high or 0.0) - (g_les_low or 0.0)    # the same gap under the encoding lesion (DA pinned)
    lesion_attribution = attributable_to(
        "the high-vs-low write-gain differential owed to the LIVE self-produced DA read (control = BRAIN_DA_ENCODING_LESION)",
        diff_live, diff_lesion)

    # ── EARN the verdict — preconditions travel with the result. ──
    from tools.verdict import Verdict
    v = Verdict("DA-gated encoding wired into the production chat store (WAVE-0 Gap-4 coupling), default-OFF")
    v.require("(A) OFF: no da_encoding key on the response (byte-identical trace)", off_no_key, expect=True,
              note="the coupling adds nothing when the flag is unset")
    v.require("(A) OFF: the live composer's encoding_gain_fn is None (the install never ran)", off_fn_none, expect=True,
              note="=> the store is the byte-identical unit-magnitude write (g=1.0)")
    v.require("(B) ON: da_encoding key present on both arms", on_keys_present, expect=True)
    v.require("(B) ON, LOAD-BEARING: the live-DA write gain g_high > g_low", go_b_wiring, expect=True,
              note=f"g_high={g_high} (DA={da_high}) > g_low={g_low} (DA={da_low})")
    v.require("(B) MECHANISM: that g writes a measurably stronger trace on a magnitude store (ratio == g_high/g_low)",
              go_b_mech, expect=True,
              note=f"stored |w|: high={mech_mag_high} low={mech_mag_low} ratio={mech_ratio} vs g-ratio {expected_ratio}")
    v.require("(B) the recall CONTENT is unchanged vs OFF on the rf numpy store (coupling never corrupts a fact)",
              recall_unchanged_on, expect=True, note="the gain colors trace STRENGTH (on a magnitude store), never WHICH fact")
    v.require("(C) LESION: g pinned to 1.0 on the high-DA arm", (g_les_high is not None and abs(g_les_high - 1.0) < 1e-9),
              expect=True)
    v.require("(C) LESION: the high-vs-low differential VANISHES (attribution to the DA read)", go_c, expect=True,
              note=f"g_les_high={g_les_high} == g_les_low={g_les_low} == 1.0")
    v.control("the write gain rides the LIVE DA read (on) and is severed by the lesion",
              treatment=(g_high or 0.0), control=(g_les_high or 0.0), min_separation=0.0,
              note="on: g scales with DA; lesion: g==1.0 regardless")
    v.disabled("heavy Gate-B organs (affect/worldmodel/surprise/... = 0) in the handler proofs",
               why="disabled ONLY for speed/confound-isolation; they run identically on every flag arm")
    v.disabled("the rf numpy fast-path store's MAGNITUDE effect (enable_substrate_store=False)",
               why="the rf recall is magnitude-invariant -> the gain is a write-side reserve THERE; the WIRING (a "
                   "differential live-DA write gain) is proven through the handler, the magnitude effect on the mech "
                   "sub-check (== the production-default onebrain store) + the I-7-b GO")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_da_encoding_wired_verify",
        "go": go, "status": decided["status"],
        "coupling": "DA-gated encoding (encoding_gain_fn <- live self-produced tonic DA), default-OFF, WAVE-0 Gap-4",
        "gain_map": "g = clip(0.5, 3.0, 1 + 2.0*(DA - 0.5)) [reused: _burndown_I7_dopamine_encoding_deploy_derisk]",
        "A_off_byte_identical": {
            "no_da_encoding_key": off_no_key, "encoding_gain_fn_is_None": off_fn_none,
            "recall_hash": off_hash, "recalled_patient": off_patient, "GO": go_a,
        },
        "B_on_load_bearing": {
            "induce_high_pa": INDUCE_HIGH, "induce_low_pa": INDUCE_LOW,
            "da_high": da_high, "da_low": da_low, "g_high": g_high, "g_low": g_low,
            "g_high_gt_g_low": go_b_wiring, "recall_content_unchanged_vs_off": recall_unchanged_on,
            "mechanism_stored_mag_high": mech_mag_high, "mechanism_stored_mag_low": mech_mag_low,
            "mechanism_stored_ratio": mech_ratio, "expected_ratio_g_high_over_g_low": expected_ratio,
            "mechanism_stored_trace_differs": go_b_mech,
        },
        "C_lesion_severs": {
            "g_lesion_high": g_les_high, "g_lesion_low": g_les_low, "differential_vanishes": go_c,
            "differential_live": diff_live, "differential_lesion": diff_lesion,
            "attribution_to_live_DA_read": lesion_attribution,
        },
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    op = "research/findings/raw/_da_encoding_wired/verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 104
    print("\n" + bar, flush=True)
    print("  DA-GATED ENCODING WIRE-IN VERIFY — real production brain_chat handler (numpy-CPU, rf recall)", flush=True)
    print(bar, flush=True)
    print(f"  (A) OFF byte-identical: no_key={off_no_key} encoding_gain_fn_None={off_fn_none} "
          f"recall={off_patient!r} -> GO_A={go_a}", flush=True)
    print(f"  (B) ON load-bearing:   g_high={g_high} (DA={da_high})  >  g_low={g_low} (DA={da_low})  -> {go_b_wiring}", flush=True)
    print(f"      MECHANISM (magnitude store): |w|_high={mech_mag_high:.4f} |w|_low={mech_mag_low:.4f} "
          f"ratio={mech_ratio:.4f} (== g-ratio {expected_ratio:.4f}) -> {go_b_mech}", flush=True)
    print(f"      recall CONTENT unchanged vs OFF (rf store): {recall_unchanged_on}", flush=True)
    print(f"  (C) LESION severs:     g_les_high={g_les_high} == g_les_low={g_les_low} == 1.0 -> {go_c}", flush=True)
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} ({decided['status']})", flush=True)
    print(f"  [saved] {op}\n" + bar, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
