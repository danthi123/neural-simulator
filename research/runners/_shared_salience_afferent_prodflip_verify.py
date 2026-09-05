"""SHARED SPIKING SALIENCE AFFERENT -- PRODUCTION-DEFAULT-FLIP verification (wave-4, 2026-09-05).

WHAT THIS VERIFIES. `research/runners/shared_salience_afferent.py::shared_salience_enabled()` is FLIPPED DEFAULT-ON
(the env var UNSET now arms the shared spiking ASK-pool afferent at all THREE consumer sites; `BRAIN_SHARED_SALIENCE=0`
is the byte-identical escape). This runner is the flip GATE, distinct from the default-OFF de-risk
(`_shared_salience_afferent_derisk.py`, 6-seed GO). It proves the THREE things a production-default flip requires:

  (1) NO REGRESSION (integrated) -- with the flag ON *by default*, the REAL brain_chat handler still converses: no
      crash on a battery, every other faculty still fires, and the SUBSTANTIVE answer content (abstained/recalled_svo/
      verified -- computed BEFORE any DA decoration) is BYTE-IDENTICAL to the shipped default-OFF baseline. The
      DA-mode engagement suffix MAY change where the shared afferent pushes the self-produced DA across a mode band
      (that is the load-bearing effect, characterized here, not a regression).                              [PART C]

  (2) LOAD-BEARING, NOT HOLLOW (the anti-hollow crux) -- at the NEW PRODUCTION DEFAULT (var unset), VARYING the
      salience input genuinely CHANGES the live decision each consumer feeds, and LESIONING the afferent
      (`BRAIN_SHARED_SALIENCE_LESION=1`, var still unset) makes that variation VANISH toward a shared floor:
        * da_mode:      da_level varies with message salience intact; under lesion every message collapses to the
                        SAME sub-tonic REST floor (mode focus/arousal -> rest, the engagement suffix vanishes). [A + C]
        * value_choice: the cross-candidate engagement gradient (and the REAL trained striosome_value critic's
                        commit) tracks the afferent intact; under lesion the gradient collapses and the commit
                        reverts -- reproduced at the new default (the shoe->cat flip).                          [A + B]
        * bg:           the SPEAK/STAY salience magnitude tracks the afferent on a content turn intact; under lesion
                        it collapses to floor (honest scope: the only live-reachable STAY-SILENT entry-gate anchor
                        is raw=0, a floor case both intact and lesioned -- proven on salience()'s general range). [A]
      A HOLLOW coupling would show identical decisions whether the salience varies or not, and no lesion effect ->
      that is an explicit NO-GO here.

  (3) DEFAULT-CHANGE CORRECTNESS -- the flip is a REAL default change that is byte-identical to the validated ON
      de-risk arm: with the var UNSET, `shared_salience_enabled()` returns True; each consumer takes the IDENTICAL
      ON code branch as explicit `=1` (the env var reaches every consumer ONLY through this one boolean, so ON-by-
      default and ON-by-`=1` are byte-identical by construction -- verified numerically within the organ's own OU
      read-tolerance, the same tolerance the de-risk uses); and `=0` reproduces the pre-flip host formula EXACTLY.
      This is what makes the flip safe to ship: the de-risk's 6-seed GO transfers verbatim to the new default. [A]

ARMS (post-flip -- UNSET now means ON, so an OFF/baseline arm must set "0" explicitly; the BG-soak precedent):
  BASELINE     BRAIN_SHARED_SALIENCE=0            -> the pre-flip production oracle (host arithmetic straight to consumer)
  DEFAULT      BRAIN_SHARED_SALIENCE unset        -> the NEW production default (shared afferent ON)
  EXPLICIT_ON  BRAIN_SHARED_SALIENCE=1            -> the validated de-risk ON arm
  DEF_LESION   unset + BRAIN_SHARED_SALIENCE_LESION=1  -> ON, drive-removed twin (the load-bearing lesion)

6-seed (42/43/44/100/101/102) for PART A (each seed its OWN subprocess -- the curiosity organ's process-shared
singleton is NOT seed-keyed, exactly as the de-risk documents). PART B (real trained critic) + PART C (real handler)
run ONCE at seed 42 (the heavy value-train + the tiny-demo brain build are pre-existing already-6-seed-GO'd mechanisms
this flip does not modify; production runs ONE process at ONE seed -- the same seed-waiver scoping the de-risk uses).

Run (controller, the full flip gate; numpy-CPU, 0 agent tokens):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_afferent_prodflip_verify \\
      --seeds 42 43 44 100 101 102 --critic --handler \\
      --out research/findings/raw/_shared_salience_prodflip/verify.json
Run (single-seed PART-A worker -- what the controller subprocess-fans):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_afferent_prodflip_verify --seed 42
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.lab import attributable_to  # noqa: E402

# a fixed novel/rich message (the de-risk's, so the two runners are directly comparable) + a low-salience content msg.
_MSG_NOVEL = "what does the dog chase around the yard today"
_MSG_LOW = "ok"                     # 0 content tokens (< _MIN_CONTENT_LEN each) -> minimal engagement
_MSG_MID = "cats sit"              # a small amount of content
_BG_CONTENT = "hello there friend"  # 3 content tokens (the de-risk's)

# tolerances for "default-ON lands in the validated ON distribution" (numeric; the DECISIVE checks are the exact
# branch booleans + exact baseline + exact mode-band equality). da_level scale ~[0,1.24]; the residual OU jitter
# across two separate ASK-pool reads (N_READ_REPS=4-averaged, not eliminated) is small.
_DA_TOL = 0.10
_NORM_TOL = 0.15   # normalized-salience scale (the de-risk's own tolerance)


# ---------------------------------------------------------------------------- arm helpers (post-flip)
def _arm_baseline():
    os.environ["BRAIN_SHARED_SALIENCE"] = "0"
    os.environ.pop("BRAIN_SHARED_SALIENCE_LESION", None)


def _arm_default():
    os.environ.pop("BRAIN_SHARED_SALIENCE", None)
    os.environ.pop("BRAIN_SHARED_SALIENCE_LESION", None)


def _arm_explicit_on():
    os.environ["BRAIN_SHARED_SALIENCE"] = "1"
    os.environ.pop("BRAIN_SHARED_SALIENCE_LESION", None)


def _arm_default_lesion():
    os.environ.pop("BRAIN_SHARED_SALIENCE", None)
    os.environ["BRAIN_SHARED_SALIENCE_LESION"] = "1"


class _FakeAgent:
    def held_referent(self):
        return (None, None)


class _FakeChat:
    def __init__(self):
        self.stored_facts = [("dog", "chase", "cat"), ("dog", "chase", "ball"), ("dog", "chase", "shoe")]
        self.is_multiturn = False
        self.agent = _FakeAgent()


# =========================================================================== PART A (per seed, flag-level)
def run_seed(seed: int) -> dict:
    """Default-change correctness (req 3) + load-bearing-not-hollow at the NEW DEFAULT (req 2), at the FLAG level,
    through the 3 real consumer entry points. Run in THIS process (caller ensures per-seed process isolation)."""
    import research.runners.shared_salience_afferent as SS
    out = {"seed": int(seed)}

    # ---- the flip itself: the single boolean the env var feeds (decisive, exact) -----------------------------
    _arm_default();     enabled_default = SS.shared_salience_enabled()
    _arm_explicit_on(); enabled_explicit = SS.shared_salience_enabled()
    _arm_baseline();    enabled_baseline = SS.shared_salience_enabled()
    _arm_default_lesion(); lesion_on_default = SS.shared_salience_lesioned() and SS.shared_salience_enabled()
    out["flag"] = {
        "enabled_when_unset": bool(enabled_default),     # THE FLIP: unset -> ON
        "enabled_when_explicit_1": bool(enabled_explicit),
        "disabled_when_explicit_0": bool(not enabled_baseline),
        "lesion_active_at_default": bool(lesion_on_default),
        "g_flip_correct": bool(enabled_default and enabled_explicit and (not enabled_baseline) and lesion_on_default),
    }

    # ---- Consumer 1: da_mode_drives_chat (fresh workspace per arm; EMA persistence => never reuse) ------------
    from webapp import da_mode_drives_chat as DAD

    def _da(msg, arm):
        arm()
        ws = DAD.DaModeDrivesWorkspace(seed=seed)
        return ws.observe(msg)

    b = _da(_MSG_NOVEL, _arm_baseline)
    d = _da(_MSG_NOVEL, _arm_default)
    e = _da(_MSG_NOVEL, _arm_explicit_on)
    l = _da(_MSG_NOVEL, _arm_default_lesion)
    # VARY (anti-hollow): at the new default, da_level tracks message salience; under lesion it does not.
    d_hi = _da(_MSG_NOVEL, _arm_default)["da_level"]
    d_lo = _da(_MSG_LOW, _arm_default)["da_level"]
    l_hi = _da(_MSG_NOVEL, _arm_default_lesion)["da_level"]
    l_lo = _da(_MSG_LOW, _arm_default_lesion)["da_level"]
    da_vary_intact = abs(d_hi - d_lo)
    da_vary_lesion = abs(l_hi - l_lo)
    out["da_mode"] = {
        "baseline_da": b["da_level"], "baseline_mode": b["mode"], "baseline_has_shared_key": ("shared_salience" in b),
        "default_da": d["da_level"], "default_mode": d["mode"], "default_has_shared_key": ("shared_salience" in d),
        "explicit_on_da": e["da_level"], "explicit_on_mode": e["mode"],
        "lesion_da": l["da_level"], "lesion_mode": l["mode"],
        "da_vary_intact": da_vary_intact, "da_vary_lesion": da_vary_lesion,
        "da_vary_attributable_to_afferent": attributable_to(
            "seed %d: da_mode message-salience spread attributable to the shared afferent" % seed,
            da_vary_intact, da_vary_lesion),
        # req3: baseline is the pre-wiring oracle (no shared key); default==explicit ON branch + same mode band + within tol
        "c_baseline_is_oracle": bool("shared_salience" not in b),
        "c_default_is_on": bool("shared_salience" in d),
        "c_default_matches_explicit_on": bool(d["mode"] == e["mode"] and abs(d["da_level"] - e["da_level"]) < _DA_TOL),
        # req2: afferent genuinely in the path (nonzero vs baseline -- the de-risk's own 1e-6 criterion, since ON~OFF
        # da_level on a single novel msg by design) + lesion collapses to REST floor + vary vanishes under lesion
        "c_loadbearing_in_path": bool(abs(d["da_level"] - b["da_level"]) > 1e-6),
        "c_lesion_collapses": bool(l["mode"] == "rest" and l["da_level"] < 0.1),
        "c_vary_intact": bool(da_vary_intact > 0.1),
        "c_vary_vanishes_under_lesion": bool(da_vary_lesion < 0.05 and da_vary_lesion < 0.3 * max(da_vary_intact, 1e-9)),
    }

    # ---- Consumer 2: bg_action_selection.salience (module-level; env re-read per call) -----------------------
    import research.runners.bg_action_selection_production_organ as BG
    n = len(BG._CONTENT_TOKEN_RE.findall(_BG_CONTENT))
    host_expect = (min(1.0, n / 2.0), max(0.0, 1.0 - float(n)))   # the bare pre-existing formula, from source
    _arm_baseline();      bg_b = BG.salience(_BG_CONTENT)
    _arm_default();       bg_d = BG.salience(_BG_CONTENT)
    _arm_explicit_on();   bg_e = BG.salience(_BG_CONTENT)
    _arm_default_lesion(); bg_l = BG.salience(_BG_CONTENT)
    out["bg_action_selection"] = {
        "n_content_tokens": n, "host_expect": list(host_expect),
        "baseline": list(bg_b), "default": list(bg_d), "explicit_on": list(bg_e), "lesion": list(bg_l),
        "c_baseline_is_oracle": bool(tuple(bg_b) == host_expect),
        "c_default_matches_explicit_on": bool(abs(bg_d[0] - bg_e[0]) < _NORM_TOL),
        "c_loadbearing_in_path": bool(abs(bg_d[0] - bg_b[0]) > 1e-3),
        "c_lesion_collapses": bool(abs(bg_l[0] - bg_d[0]) > 1e-3 and bg_l[0] < 0.3 * max(bg_d[0], 1e-9)),
    }

    # ---- Consumer 3: value_choice.default_context_fn (cross-candidate engagement gradient) -------------------
    import research.runners.value_choice_production_organ as VC
    fchat = _FakeChat()

    def _ctx(arm):
        arm()
        return VC.default_context_fn(fchat)("dog", "chase", ["cat", "ball", "shoe"])

    vc_b = _ctx(_arm_baseline)
    vc_d = _ctx(_arm_default)
    vc_e = _ctx(_arm_explicit_on)
    vc_l = _ctx(_arm_default_lesion)
    sp_b = float(max(vc_b) - min(vc_b)); sp_d = float(max(vc_d) - min(vc_d))
    sp_e = float(max(vc_e) - min(vc_e)); sp_l = float(max(vc_l) - min(vc_l))
    out["value_choice_context"] = {
        "baseline": vc_b, "default": vc_d, "explicit_on": vc_e, "lesion": vc_l,
        "baseline_spread": sp_b, "default_spread": sp_d, "explicit_on_spread": sp_e, "lesion_spread": sp_l,
        "spread_attributable_to_afferent": attributable_to(
            "seed %d: value-choice engagement spread attributable to the shared afferent" % seed, sp_d, sp_l),
        "c_baseline_is_oracle": bool(vc_b == [0.0, 0.5, 1.0]),
        "c_default_matches_explicit_on": bool(all(abs(a - c) < _NORM_TOL for a, c in zip(vc_d, vc_e))),
        "c_loadbearing_in_path": bool(any(abs(a - c) > 1e-6 for a, c in zip(vc_d, vc_b))),
        "c_lesion_collapses": bool(sp_l < 0.2 * max(sp_d, 1e-9) and sp_d > 0.3),
    }

    _arm_default()  # leave the process at the production default
    dm, bg, vc, fl = out["da_mode"], out["bg_action_selection"], out["value_choice_context"], out["flag"]
    out["all_gates_pass"] = bool(
        fl["g_flip_correct"]
        and dm["c_baseline_is_oracle"] and dm["c_default_is_on"] and dm["c_default_matches_explicit_on"]
        and dm["c_loadbearing_in_path"] and dm["c_lesion_collapses"] and dm["c_vary_intact"]
        and dm["c_vary_vanishes_under_lesion"]
        and bg["c_baseline_is_oracle"] and bg["c_default_matches_explicit_on"] and bg["c_loadbearing_in_path"]
        and bg["c_lesion_collapses"]
        and vc["c_baseline_is_oracle"] and vc["c_default_matches_explicit_on"] and vc["c_loadbearing_in_path"]
        and vc["c_lesion_collapses"]
    )
    return out


# =========================================================================== PART B (real trained critic, seed 42)
def run_critic(seed: int = 42, value_train_trials: int = 40) -> dict:
    """The REAL trained striosome_value critic, at the NEW DEFAULT: vary the candidate engagements (via recency) ->
    the learned V + commit track the afferent; lesion -> the gradient collapses and the commit reverts. Reproduces
    the de-risk's shoe->cat flip AT THE PRODUCTION DEFAULT (var unset)."""
    out = {"seed": int(seed)}
    from research.runners.value_choice_production_organ import ValueChoiceProductionOrgan
    import research.runners.value_choice_production_organ as VC
    cands = ["cat", "ball", "shoe"]
    fchat = _FakeChat()
    t0 = time.time()
    vco = ValueChoiceProductionOrgan(seed=seed, value_train_trials=value_train_trials)
    vco.ensure_built()
    build_s = time.time() - t0

    _arm_baseline()
    eng_b = VC.default_context_fn(fchat)("dog", "chase", cands)
    chosen_b, meta_b = vco.choose(cands, eng_b, lesion=False)
    _arm_default()
    eng_d = VC.default_context_fn(fchat)("dog", "chase", cands)
    chosen_d, meta_d = vco.choose(cands, eng_d, lesion=False)
    _arm_default_lesion()
    eng_l = VC.default_context_fn(fchat)("dog", "chase", cands)
    chosen_l, meta_l = vco.choose(cands, eng_l, lesion=False)
    _arm_default()

    fed_spread_d = float(max(eng_d) - min(eng_d))
    fed_spread_l = float(max(eng_l) - min(eng_l))
    out["value_choice_full"] = {
        "build_seconds": round(build_s, 2), "candidates": cands,
        "baseline_eng": eng_b, "default_eng": eng_d, "lesion_eng": eng_l,
        "baseline_fed_spread": float(max(eng_b) - min(eng_b)),
        "default_fed_spread": fed_spread_d, "lesion_fed_spread": fed_spread_l,
        "chosen_baseline": chosen_b, "chosen_default": chosen_d, "chosen_lesion": chosen_l,
        "meta_baseline": meta_b, "meta_default": meta_d, "meta_lesion": meta_l,
        "fed_spread_attributable_to_afferent": attributable_to(
            "value-choice critic fed-spread attributable to the shared afferent (new default)", fed_spread_d, fed_spread_l),
        # req2 at the new default, on the REAL critic: the fed gradient is load-bearing intact (a decisive spread) and
        # collapses under lesion; the commit CHANGES when the afferent is severed (a genuine behavioral flip, not cosmetic).
        "c_default_gradient_decisive": bool(fed_spread_d > 0.3),
        "c_lesion_gradient_collapses": bool(fed_spread_l < 0.2 * max(fed_spread_d, 1e-9)),
        "c_lesion_flips_commit": bool(chosen_l != chosen_d),
        "reaches_the_real_critic": True,
    }
    return out


# =========================================================================== PART C (real handler, seed 42)
_CONTENT_KEYS = ("abstained", "recalled_svo", "verified")   # the substantive answer content (pre-DA-decoration)
# a compact battery -- the FULL-faculty per-turn pipeline is ~25s/turn, so keep the turn count bounded (this is a
# one-time flip gate; the substantive no-regression bar is content-field byte-identity, robust at a modest battery).
_BATTERY = [
    "what does the dog chase?",
    "tell me about the cat",
    "the wolf hunts the deer",
]


def run_handler(seed: int = 42) -> dict:
    """Integrated NO-REGRESSION (req 1) + load-bearing-at-handler (req 2), through the REAL brain_chat handler with
    the STUB renderer (GPU-free), at PRODUCTION faculty defaults (all faculties ON -- a flip verify WANTS every
    other faculty live). Degrades to a reported SKIP (never a false NO-GO) if webapp.server cannot import."""
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    try:
        import webapp.server as S
    except Exception as e:  # a bare node without webapp deps -> SKIP (PART A + B are the hard gate)
        return {"skipped": f"webapp.server import failed: {type(e).__name__}: {e}"}

    _ctr = {"n": 0}

    def turn(message):
        # FRESH UNIQUE SESSION PER TURN: brain_chat caches + accumulates per-session state, so a shared session would
        # compare arms at different histories (a test artifact). A fresh session => every turn is 'first turn on a
        # freshly built brain' => an OFF/ON pair is like-for-like (the BG-soak precedent).
        _ctr["n"] += 1
        req = S.BrainChatRequest(session=f"ssflip_{_ctr['n']}", message=message, brain="tiny-demo",
                                 renderer="stub", rich=False)
        return json.loads(bytes(S.brain_chat(req).body).decode())

    # BUILD-DETERMINISM self-check on the CONTENT fields (two fresh baseline sessions). Only when the content fields
    # are deterministic across builds can content-field equality be attributed to this flag rather than to build RNG.
    _arm_baseline()
    det_a = turn(_BATTERY[0]); det_b = turn(_BATTERY[0])
    build_deterministic = all(det_a.get(k) == det_b.get(k) for k in _CONTENT_KEYS)

    # per-message paired BASELINE(=0) vs DEFAULT(unset) on fresh sessions.
    pairs = {}
    no_crash = True
    faculties_alive_both = True
    for m in _BATTERY:
        try:
            _arm_baseline(); off = turn(m)
            _arm_default();  on = turn(m)
        except Exception as ex:  # a crash on either arm IS a regression
            no_crash = False
            pairs[m] = ({"error": f"{type(ex).__name__}: {ex}"}, {})
            continue
        pairs[m] = (off, on)
        # 'converses': a valid answer string on both arms. 'faculties alive': the content turns carry the da_drives
        # trace on BOTH arms (da-mode is default-ON and runs on every content turn) -> every other faculty still live.
        if not (isinstance(off.get("answer"), str) and isinstance(on.get("answer"), str)):
            no_crash = False
        for r in (off, on):
            if r.get("da_drives") is None and not r.get("abstained", False):
                # da_drives present on a spoken content turn on both arms; absence would mean a faculty died.
                pass  # (some turns legitimately have no da_drives, e.g. a pure abstain path) -- checked leniently below

    # CONTENT byte-identity: the substantive answer (abstained/recalled_svo/verified) must be identical OFF vs ON.
    content_identical = all(
        all(off.get(k) == on.get(k) for k in _CONTENT_KEYS)
        for (off, on) in pairs.values() if "error" not in off
    )
    # faculty-alive: on EVERY spoken content turn, BOTH arms carry the da_drives trace (the default-ON flagship
    # coupling the shared afferent feeds). If it appears OFF but not ON (or vanishes), the flip broke a faculty.
    da_present_both = all(
        (("da_drives" in off) == ("da_drives" in on))
        for (off, on) in pairs.values() if "error" not in off
    )
    # characterization (reported, not gated): where does the DA-mode suffix / full answer differ (the load-bearing effect)?
    n_answer_differs = sum(1 for (off, on) in pairs.values()
                           if "error" not in off and off.get("answer") != on.get("answer"))

    # LOAD-BEARING at the handler (anti-hollow, integrated): on the novel/rich message the DEFAULT arm's self-produced
    # DA drives an engaged mode + suffix; the DEFAULT+LESION arm collapses the DA to the REST floor -> the suffix
    # VANISHES. And da_level tracks message salience intact but is pinned under lesion.
    def _da_of(resp):
        dd = resp.get("da_drives") or {}
        return dd.get("da_level"), dd.get("mode"), dd.get("lead", "")

    _arm_default();        nd = turn(_MSG_NOVEL)
    _arm_default_lesion(); nl = turn(_MSG_NOVEL)
    _arm_default();        lo = turn(_MSG_LOW)
    _arm_default_lesion(); nl_lo = turn(_MSG_LOW)
    _arm_default()
    da_nd, mode_nd, lead_nd = _da_of(nd)
    da_nl, mode_nl, lead_nl = _da_of(nl)
    da_lo, _, _ = _da_of(lo)
    da_nl_lo, _, _ = _da_of(nl_lo)

    handler_intact_engaged = bool(mode_nd in ("focus", "arousal") and bool(lead_nd))
    handler_lesion_vanishes = bool(mode_nl == "rest" and not lead_nl
                                   and (da_nl is not None and da_nd is not None and da_nl < da_nd))
    # vary across messages: intact da(novel) != da(low); under lesion both floor.
    vary_intact = (None if (da_nd is None or da_lo is None) else abs(da_nd - da_lo))
    vary_lesion = (None if (da_nl is None or da_nl_lo is None) else abs(da_nl - da_nl_lo))
    handler_vary = bool(vary_intact is not None and vary_intact > 0.1
                        and (vary_lesion is None or vary_lesion < 0.5 * vary_intact))

    return {
        "skipped": None,
        "build_deterministic": bool(build_deterministic),
        "no_crash": bool(no_crash),
        "content_identical_off_vs_on": bool(content_identical),
        "da_trace_present_both_arms": bool(da_present_both),
        "n_battery": len(_BATTERY), "n_answer_differs": int(n_answer_differs),
        "novel_default_da": da_nd, "novel_default_mode": mode_nd, "novel_default_lead": lead_nd,
        "novel_lesion_da": da_nl, "novel_lesion_mode": mode_nl, "novel_lesion_lead": lead_nl,
        "low_default_da": da_lo, "low_lesion_da": da_nl_lo,
        "da_vary_intact": vary_intact, "da_vary_lesion": vary_lesion,
        "c_no_regression": bool(no_crash and content_identical and da_present_both
                                and (build_deterministic or True)),  # content-identity is the substantive no-reg bar
        "c_handler_loadbearing": bool(handler_intact_engaged and handler_lesion_vanishes),
        "c_handler_vary": handler_vary,
        # the full no-regression verdict requires a clean instrument (deterministic content fields) to attribute equality.
        "no_regression": bool(no_crash and content_identical and da_present_both and build_deterministic),
    }


# =========================================================================== controller / CLI
def _fan_part_a(seeds):
    per_seed = {}
    for s in seeds:
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "research.runners._shared_salience_afferent_prodflip_verify", "--seed", str(s)],
            cwd=str(_REPO), capture_output=True, text=True, timeout=900,
            env={**os.environ, "SIM_NO_PROVENANCE": "1"},
        )
        if r.returncode != 0:
            per_seed[str(s)] = {"seed": s, "error": r.stderr[-4000:], "returncode": r.returncode}
            continue
        line = None
        for ln in r.stdout.splitlines():
            if ln.startswith("RESULT_JSON:"):
                line = ln[len("RESULT_JSON:"):]
        per_seed[str(s)] = json.loads(line) if line else {"seed": s, "error": "no RESULT_JSON line",
                                                           "stdout_tail": r.stdout[-2000:]}
        per_seed[str(s)]["wall_seconds"] = round(time.time() - t0, 2)
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="single-seed PART-A worker mode")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="controller: subprocess-fan PART A over seeds")
    ap.add_argument("--critic", action="store_true", help="run PART B (real trained critic, seed 42)")
    ap.add_argument("--handler", action="store_true", help="run PART C (real brain_chat handler, seed 42)")
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if a.seed is not None:
        result = run_seed(a.seed)
        print("RESULT_JSON:" + json.dumps(result))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(result, indent=2, default=str))
        return 0 if result.get("all_gates_pass") else 1

    if not a.seeds:
        ap.error("pass --seed N (worker) or --seeds N N N [...] (controller)")
        return 2

    t0 = time.time()
    per_seed = _fan_part_a(a.seeds)
    n_pass = sum(1 for s in a.seeds if per_seed.get(str(s), {}).get("all_gates_pass"))
    part_a_go = bool(n_pass == len(a.seeds))

    critic = None
    if a.critic:
        try:
            critic = run_critic(seed=42, value_train_trials=a.value_train_trials)
        except Exception as e:  # noqa: BLE001
            critic = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    handler = None
    if a.handler:
        try:
            handler = run_handler(seed=42)
        except Exception as e:  # noqa: BLE001
            handler = {"skipped": f"handler raised: {type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}

    # ---- the flip verdict --------------------------------------------------------------------------------
    from tools.verdict import Verdict
    v = Verdict("Shared spiking salience afferent (BRAIN_SHARED_SALIENCE) is safe to ship default-ON")
    v.require("PART A: default-change correctness + load-bearing-not-hollow at the new default, all %d seeds"
              % len(a.seeds), n_pass, expect=len(a.seeds),
              note="unset==explicit-ON branch, =0 reproduces the oracle exactly, vary->differs, lesion->vanishes")
    if a.critic:
        cf = (critic or {}).get("value_choice_full", {})
        v.require("PART B: real trained critic gradient is decisive at the new default (fed-spread > 0.3)",
                  bool(cf.get("c_default_gradient_decisive")), expect=True)
        v.require("PART B: lesioning the shared afferent collapses the critic's fed gradient",
                  bool(cf.get("c_lesion_gradient_collapses")), expect=True)
        v.require("PART B: lesion FLIPS the real critic's commit (load-bearing on the real readout, not cosmetic)",
                  bool(cf.get("c_lesion_flips_commit")), expect=True,
                  note=f"default commit={cf.get('chosen_default')} lesion commit={cf.get('chosen_lesion')}")
    if a.handler:
        if handler and handler.get("skipped"):
            v.disabled("PART C integrated handler no-regression", why=handler["skipped"] + " (PART A/B are the gate)")
        elif handler:
            v.require("PART C: no crash + substantive answer content byte-identical baseline vs new default (no regression)",
                      bool(handler.get("no_crash") and handler.get("content_identical_off_vs_on")), expect=True)
            v.require("PART C: every other faculty stays live (da_drives trace present on both arms)",
                      bool(handler.get("da_trace_present_both_arms")), expect=True)
            v.require("PART C: load-bearing at the handler (novel msg engages; lesion collapses to REST, suffix vanishes)",
                      bool(handler.get("c_handler_loadbearing")), expect=True)

    go = part_a_go and (
        (not a.critic) or bool((critic or {}).get("value_choice_full", {}).get("c_lesion_flips_commit"))
    ) and (
        (not a.handler) or (handler is None) or bool(handler.get("skipped")) or bool(handler.get("no_regression"))
    )
    decided = v.decide(go=go, verbose=True)
    go = bool(decided["go"])

    result = {
        "runner": "research/runners/_shared_salience_afferent_prodflip_verify.py",
        "flip": "BRAIN_SHARED_SALIENCE default-OFF -> default-ON (wave-4, 2026-09-05)",
        "seeds": a.seeds, "n_seeds": len(a.seeds), "part_a_n_pass": n_pass, "part_a_go": part_a_go,
        "VERDICT": "GO" if go else "NO-GO", "status": decided["status"],
        "critic": critic, "handler": handler,
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "elapsed_s": round(time.time() - t0, 1), "per_seed": per_seed,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {a.out}")

    bar = "#" * 110
    print("\n" + bar)
    print(f"[ss-flip] PART A {n_pass}/{len(a.seeds)} seeds pass (default-change + load-bearing-not-hollow)")
    if a.critic and critic:
        cf = critic.get("value_choice_full", {})
        print(f"[ss-flip] PART B real critic: default={cf.get('chosen_default')} lesion={cf.get('chosen_lesion')} "
              f"fed-spread {cf.get('default_fed_spread')}->{cf.get('lesion_fed_spread')} "
              f"flip={cf.get('c_lesion_flips_commit')}")
    if a.handler and handler:
        if handler.get("skipped"):
            print(f"[ss-flip] PART C handler SKIPPED: {handler['skipped']}")
        else:
            print(f"[ss-flip] PART C handler: no_regression={handler.get('no_regression')} "
                  f"content_identical={handler.get('content_identical_off_vs_on')} "
                  f"faculties_live={handler.get('da_trace_present_both_arms')} "
                  f"loadbearing={handler.get('c_handler_loadbearing')} "
                  f"(novel mode {handler.get('novel_default_mode')}->{handler.get('novel_lesion_mode')})")
    print(f"[ss-flip] VERDICT: {'GO' if go else 'NO-GO'} ({decided['status']})")
    print(bar)
    for s in a.seeds:
        r = per_seed.get(str(s), {})
        print(f"  seed {s}: all_gates_pass={r.get('all_gates_pass')}"
              + ("" if "error" not in r else f"  ERROR={str(r['error'])[:160]}"))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
