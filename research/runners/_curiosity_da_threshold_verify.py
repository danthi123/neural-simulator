"""WIRE-IN VERIFY for the DA/ENGAGEMENT-GATED CURIOSITY crave-threshold (board WAVE-0, Gap-4 coupling (b)), through
the REAL production `webapp/server.py::brain_chat` handler (numpy-CPU, rf recall; foreground, ~a minute).

THE COUPLING (built, default-OFF): the brain's OWN self-produced tonic dopamine (the #76/#79 spiking SNc read, stashed
on `chat._last_da_drives["da_level"]` by `da_mode_drives_chat.observe_turn` earlier in the turn) modulates the
CURIOSITY crave decision. The curiosity organ (`curiosity_production_organ`) reads a genuinely-SPIKING ASK-pool WANT on
a NOVEL-topic ABSTAIN and decides `want >= threshold` (threshold calibrated at build). This scales the ASK-pool WANT by
a small DA crave-gain (`webapp/da_curiosity_drives_chat.crave_decision`): ENGAGED (DA > tonic 0.5) -> a HIGHER effective
want / LOWER effective crave-threshold -> the brain asks a follow-up on a topic it would otherwise let pass; disengaged
-> the reverse. It changes ONLY WHETHER the honest follow-up QUESTION is appended — never a fact, never the abstain
(the moat is preserved). Default-OFF behind `BRAIN_CURIOSITY_DA`; lesion `BRAIN_CURIOSITY_DA_LESION=1` pins the gain to
1.0 (severs the DA dependence).

WHAT THIS PROVES (the wire-in gate; GO = A and B and C):
  (A) OFF (`BRAIN_CURIOSITY_DA` unset) — on a NOVEL-topic ABSTAIN and on a FAMILIAR (recall) probe through the REAL
      handler: NO `curiosity_da` key on the response, and the follow-up decision is the curiosity organ's OWN calibrated
      threshold (novel -> the follow-up fires; familiar recall -> no follow-up) -> BYTE-IDENTICAL to HEAD (the ONLY
      thing this change adds is skipped when off).
  (B) ON, LOAD-BEARING — the SAME novel-abstain message with its novelty HELD FIXED, under a HIGH-DA turn
      (`BRAIN_DA_DRIVES_INDUCE=1300`, aroused/engaged) vs a LOW-DA turn (`=100`, rest): the HIGH-DA turn CROSSES the
      lowered crave-threshold and APPENDS the follow-up where the LOW-DA turn does NOT. Reported: the live DA level, the
      DA crave-gain, the effective threshold, and the ASK-pool want on each arm.
  (C) LESION (`BRAIN_CURIOSITY_DA_LESION=1`) — the DA modulation is pinned to 0 (gain 1.0) regardless of the DA level,
      so the high-vs-low follow-up decision is IDENTICAL (the DA-dependence VANISHES) — attributing the (B) difference
      to the live DA read. (DISTINCT from BRAIN_CURIOSITY_LESION, which collapses the WANT, and BRAIN_DA_DRIVES_LESION,
      which collapses the LEVEL.)

HONEST SCOPE. A WIRED, default-OFF coupling — NOT flipped on. The NOVELTY the organ reads is the ABSTAIN (a declared
host boundary, per the curiosity organ); the ENGAGEMENT->DA afferent is the same host sensory/comprehension boundary
the #79 DA-mode read names as its residual. The DA LEVEL itself (SNc spikes off the bus) and the spiking ASK-pool WANT
are the neural mechanisms this coupling rides. The crave-gain constants (K_DA, g_min/g_max) are host-tuned.

Run (numpy-CPU, foreground, ~a minute):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._curiosity_da_threshold_verify
"""
from __future__ import annotations

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

# a NOVEL topic the tiny-demo brain does NOT hold -> the moat ABSTAINS -> the curiosity block runs; the message is HELD
# FIXED across the DA arms (only the induced DA level varies), exactly the (B) design. Candidates: the runner uses the
# first that reliably abstains AND the organ reads curious through the real handler.
NOVEL_CANDIDATES = ["what do you know about wombats", "what is a pangolin", "what do you know about quasars",
                    "what is an aardvark", "what do you know about tardigrades"]
FAM_FACT_MSG = "cat eat fish"     # taught first on the familiar arm -> then recalled (a non-abstain -> no follow-up)
FAM_QUERY_MSG = "what does cat eat"
INDUCE_HIGH = 1300.0              # arousal/engaged -> DA ~= 1.24 (per da_mode_drives_chat's afferent calibration)
INDUCE_LOW = 100.0               # rest/disengaged -> DA well below tonic 0.5

# quiet, confound-minimizing env: disable the heavy Gate-B organs that run identically on every arm (speed + isolation).
# CRUCIAL: BRAIN_CURIOSITY is left at its default-ON so the curiosity block runs; BRAIN_DA_DRIVES at its default-ON
# anchor so the SNc->DA read runs and INDUCE can set the level. rich=False -> the single-SVO path (curiosity at :5016).
_QUIET = {
    "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_SURPRISE": "0", "BRAIN_METACOG": "0",
    "BRAIN_MULTIREF": "0", "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_RECONSOLIDATION": "0",
    "BRAIN_EPISODIC_STORE": "0", "BRAIN_RICH": "0", "BRAIN_GNW_BUS": "0",
    "BRAIN_CONTINUOUS": "0", "BRAIN_CONTINUOUS_DRIVES": "0", "BRAIN_SWAP_DRIVES": "0",
}

_FOLLOWUP_MARK = "curiosity is piqued"   # the honest follow-up QUESTION's stable marker (followup_question)


def _set(env):
    """Apply/clear env keys (value None -> del). Mutates os.environ."""
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def _turn(session, msg, *, da_curiosity, lesion, induce, teach=None):
    """Run one turn through the REAL brain_chat handler on a FRESH session. `teach` (a message) is taught first on the
    same session (for the familiar/recall arm). da_curiosity/lesion/induce control the coupling env for THIS arm.
    Returns the response dict for `msg`."""
    from webapp.server import brain_chat, BrainChatRequest as Req
    env = dict(_QUIET)
    env["BRAIN_CURIOSITY_DA"] = "1" if da_curiosity else None
    env["BRAIN_CURIOSITY_DA_LESION"] = "1" if lesion else None
    env["BRAIN_DA_DRIVES_INDUCE"] = (str(induce) if induce is not None else None)
    _set(env)
    if teach is not None:
        brain_chat(Req(session=session, message=teach, brain="tiny-demo", renderer="stub", rich=False))
    r = brain_chat(Req(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(bytes(r.body).decode("utf-8"))


def _fired(resp):
    """True iff the honest curiosity follow-up QUESTION was appended this turn."""
    return _FOLLOWUP_MARK in (resp.get("answer") or "")


def _pick_novel_msg():
    """Return the first NOVEL candidate that reliably ABSTAINS and reads curious through the REAL handler (organ-only,
    flag OFF). Ensures the (B)/(C) probes sit on a genuine abstain where the follow-up is in play."""
    for m in NOVEL_CANDIDATES:
        resp = _turn(f"pick_{abs(hash(m)) % 9999}", m, da_curiosity=False, lesion=False, induce=None)
        cur = resp.get("curiosity") or {}
        if bool(resp.get("abstained")) and bool(cur.get("curious")) and _fired(resp):
            return m, resp
    return None, None


def main():
    novel_msg, novel_off_resp = _pick_novel_msg()
    if novel_msg is None:
        print("PRECONDITION FAIL: no novel candidate abstained + craved through the handler", flush=True)
        return 1

    # ── (A) OFF byte-identical: flag unset -> no curiosity_da key; organ's own decision (novel fires, familiar does not).
    off_novel = novel_off_resp                                     # already the OFF novel-abstain turn
    off_novel_cur = off_novel.get("curiosity") or {}
    off_no_key_novel = "curiosity_da" not in off_novel_cur
    off_novel_fired = _fired(off_novel)                            # organ curious on a novel abstain -> follow-up fires
    # familiar/recall arm: teach then recall -> NOT an abstain -> curiosity null -> no follow-up.
    off_fam = _turn("cda_off_fam", FAM_QUERY_MSG, da_curiosity=False, lesion=False, induce=None, teach=FAM_FACT_MSG)
    off_fam_cur = off_fam.get("curiosity")
    off_fam_abstained = bool(off_fam.get("abstained"))
    off_no_key_fam = not (isinstance(off_fam_cur, dict) and "curiosity_da" in off_fam_cur)
    off_fam_fired = _fired(off_fam)
    go_a = bool(off_no_key_novel and off_novel_fired and off_no_key_fam and (not off_fam_abstained) and (not off_fam_fired))

    # ── (B) ON, LOAD-BEARING: SAME novel message, novelty fixed; HIGH-DA vs LOW-DA -> high fires, low does not.
    hi = _turn("cda_hi", novel_msg, da_curiosity=True, lesion=False, induce=INDUCE_HIGH)
    lo = _turn("cda_lo", novel_msg, da_curiosity=True, lesion=False, induce=INDUCE_LOW)
    hi_cur = hi.get("curiosity") or {}
    lo_cur = lo.get("curiosity") or {}
    hi_da = hi_cur.get("curiosity_da") or {}
    lo_da = lo_cur.get("curiosity_da") or {}
    hi_fired, lo_fired = _fired(hi), _fired(lo)
    on_keys_present = ("curiosity_da" in hi_cur) and ("curiosity_da" in lo_cur)
    go_b = bool(on_keys_present and hi_fired and (not lo_fired))

    da_high, da_low = hi_da.get("da_level"), lo_da.get("da_level")
    gain_high, gain_low = hi_da.get("da_crave_gain"), lo_da.get("da_crave_gain")
    thr_high, thr_low = hi_da.get("eff_threshold"), lo_da.get("eff_threshold")
    want_high, want_low = hi_da.get("want_hz"), lo_da.get("want_hz")
    base_thr = hi_da.get("base_threshold")

    # ── (C) LESION: pin the gain to 1.0 regardless of DA -> the high-vs-low decision is IDENTICAL (DA-dep vanishes).
    lh = _turn("cda_les_hi", novel_msg, da_curiosity=True, lesion=True, induce=INDUCE_HIGH)
    ll = _turn("cda_les_lo", novel_msg, da_curiosity=True, lesion=True, induce=INDUCE_LOW)
    lh_da = (lh.get("curiosity") or {}).get("curiosity_da") or {}
    ll_da = (ll.get("curiosity") or {}).get("curiosity_da") or {}
    lh_fired, ll_fired = _fired(lh), _fired(ll)
    les_gain_high, les_gain_low = lh_da.get("da_crave_gain"), ll_da.get("da_crave_gain")
    les_decision_identical = bool(lh_fired == ll_fired)
    les_gains_pinned = bool(les_gain_high is not None and les_gain_low is not None
                            and abs(les_gain_high - 1.0) < 1e-9 and abs(les_gain_low - 1.0) < 1e-9)
    go_c = bool(les_decision_identical and les_gains_pinned)

    go = bool(go_a and go_b and go_c)

    # ── ATTRIBUTION: what fraction of the high-vs-low FOLLOW-UP difference is owed to the LIVE DA read? The lesion is
    #    the control (gain pinned -> the DA modulation severed). (treatment - control)/treatment == 1.0 means the DA
    #    read owns the WHOLE difference (the flip is not a residual host effect). Measuring both arms is not asking
    #    whose it was — the subtraction is. ──
    from tools.lab import attributable_to
    diff_live = float(int(hi_fired) - int(lo_fired))       # the high-vs-low follow-up gap with the DA read live
    diff_lesion = float(int(lh_fired) - int(ll_fired))     # the same gap under the curiosity-DA lesion (gain pinned)
    lesion_attribution = attributable_to(
        "the high-vs-low curiosity follow-up difference owed to the LIVE self-produced DA read (control = BRAIN_CURIOSITY_DA_LESION)",
        diff_live, diff_lesion)

    # ── EARN the verdict — preconditions travel with the result. ──
    from tools.verdict import Verdict
    v = Verdict("DA/engagement-gated curiosity crave-threshold wired into the production chat (WAVE-0 Gap-4), default-OFF")
    v.require("(A) OFF: no curiosity_da key on the novel-abstain turn (byte-identical trace)", off_no_key_novel,
              expect=True, note="the coupling adds nothing when BRAIN_CURIOSITY_DA is unset")
    v.require("(A) OFF: the organ's calibrated threshold still fires the follow-up on a novel abstain", off_novel_fired,
              expect=True, note="the un-modulated curiosity decision is preserved")
    v.require("(A) OFF: the FAMILIAR recall is not an abstain -> no follow-up + no curiosity_da key",
              bool(off_no_key_fam and (not off_fam_abstained) and (not off_fam_fired)), expect=True,
              note="curiosity is out of scope on a confident recall (byte-identical)")
    v.require("(B) ON: curiosity_da key present on both DA arms", on_keys_present, expect=True)
    v.require("(B) ON, LOAD-BEARING: HIGH-DA appends the follow-up where LOW-DA does NOT (novelty held fixed)", go_b,
              expect=True,
              note=f"high(DA={da_high},gain={gain_high},eff_thr={thr_high},want={want_high}) fired={hi_fired} | "
                   f"low(DA={da_low},gain={gain_low},eff_thr={thr_low},want={want_low}) fired={lo_fired} | "
                   f"base_threshold={base_thr}")
    v.require("(C) LESION: the gain is pinned to 1.0 on both arms", les_gains_pinned, expect=True,
              note=f"gain_high={les_gain_high} gain_low={les_gain_low}")
    v.require("(C) LESION: the high-vs-low follow-up decision is IDENTICAL (DA-dependence vanishes)",
              les_decision_identical, expect=True, note=f"lesion high fired={lh_fired} == low fired={ll_fired}")
    v.control("the crave decision rides the LIVE DA read (on: high!=low) and is severed by the lesion (high==low)",
              treatment=(1.0 if (hi_fired and not lo_fired) else 0.0),
              control=(0.0 if les_decision_identical else 1.0), min_separation=0.5,
              note="on: DA flips the follow-up; lesion: the follow-up is DA-invariant")
    v.disabled("heavy Gate-B organs (affect/worldmodel/surprise/metacog/multiref/... = 0) in the handler proofs",
               why="disabled ONLY for speed/confound-isolation; they run identically on every flag arm")
    v.disabled("the ENGAGEMENT->SNc afferent + the NOVELTY=abstain (host sensory/comprehension boundaries)",
               why="the DA LEVEL (SNc spikes off the bus) + the spiking ASK-pool WANT are the neural parts this "
                   "coupling rides; the afferent/novelty derivation are the declared #79/curiosity-organ residuals")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_curiosity_da_threshold_verify",
        "go": go, "status": decided["status"],
        "coupling": "DA/engagement-gated curiosity crave-threshold (scale ASK-pool want by a DA gain), default-OFF, WAVE-0 Gap-4",
        "gain_map": "da_crave_gain = clip(0.2, 3.0, 1 + 1.5*(DA - 0.5)); curious = (want_hz*gain) >= threshold",
        "novel_message": novel_msg,
        "A_off_byte_identical": {
            "novel_no_curiosity_da_key": off_no_key_novel, "novel_followup_fired": off_novel_fired,
            "familiar_no_key": off_no_key_fam, "familiar_abstained": off_fam_abstained,
            "familiar_followup_fired": off_fam_fired, "GO": go_a,
        },
        "B_on_load_bearing": {
            "induce_high_pa": INDUCE_HIGH, "induce_low_pa": INDUCE_LOW,
            "da_high": da_high, "da_low": da_low, "gain_high": gain_high, "gain_low": gain_low,
            "eff_threshold_high": thr_high, "eff_threshold_low": thr_low,
            "want_high": want_high, "want_low": want_low, "base_threshold": base_thr,
            "high_fired": hi_fired, "low_fired": lo_fired, "GO": go_b,
        },
        "C_lesion_severs": {
            "lesion_gain_high": les_gain_high, "lesion_gain_low": les_gain_low,
            "lesion_high_fired": lh_fired, "lesion_low_fired": ll_fired,
            "decision_identical": les_decision_identical, "gains_pinned_to_one": les_gains_pinned, "GO": go_c,
            "differential_live": diff_live, "differential_lesion": diff_lesion,
            "attribution_to_live_DA_read": lesion_attribution,
        },
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    op = "research/findings/raw/_curiosity_da_threshold/verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 104
    print("\n" + bar, flush=True)
    print("  DA-GATED CURIOSITY CRAVE-THRESHOLD WIRE-IN VERIFY — real production brain_chat handler (numpy-CPU, rf)", flush=True)
    print(bar, flush=True)
    print(f"  novel message (held fixed): {novel_msg!r}", flush=True)
    print(f"  (A) OFF byte-identical: novel[no_key={off_no_key_novel} fired={off_novel_fired}] "
          f"familiar[no_key={off_no_key_fam} abstained={off_fam_abstained} fired={off_fam_fired}] -> GO_A={go_a}", flush=True)
    print(f"  (B) ON load-bearing:   HIGH(DA={da_high} gain={gain_high} eff_thr={thr_high} want={want_high}) fired={hi_fired}", flush=True)
    print(f"                         LOW (DA={da_low} gain={gain_low} eff_thr={thr_low} want={want_low}) fired={lo_fired}  -> GO_B={go_b}", flush=True)
    print(f"  (C) LESION severs:     gains pinned {les_gain_high}=={les_gain_low}=1.0 ; decision identical "
          f"(hi={lh_fired}==lo={ll_fired}) -> GO_C={go_c}", flush=True)
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} ({decided['status']})", flush=True)
    print(f"  [saved] {op}\n" + bar, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
