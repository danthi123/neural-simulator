"""SOAK / no-regression gate for the W5 AFFECTIVE THEORY OF MIND (empathy) DEFAULT-ON flip.

A multi-turn conversation is run through the REAL production `webapp.server.brain_chat` handler (stub renderer, so no
Qwen warm) TWICE on a FRESH session each time -- flag OFF vs flag ON -- across 6 seeds. The bar (modelled on
`_d5_graded_flip_soak`):

  NO-REGRESSION: with the flag ON, the ONLY thing that changes vs OFF is the EMPATHIC LEAD on the TRIGGERED turns
  (a turn about ANOTHER agent's affectively-charged situation). Every ORDINARY turn is BYTE-IDENTICAL: an abstain
  still abstains with the SAME text (moat), a recall returns the SAME fact, no `affective_tom` key is attached, no
  crash. On a TRIGGERED turn only the empathic LEAD is prepended -- the content fields (abstained/recalled_svo/
  verified) and the answer with the lead stripped are byte-identical OFF vs ON.

  LOAD-BEARING (the faculty must CHANGE an output, and the change must RIDE the neural OTHER-region read):
    * VARY: a bad-other turn ("Maria is devastated") leads with a COMFORT expression, a good-other turn ("Tom is
      delighted") leads with a SHARE-JOY expression -> the leads DIFFER (tone_sign flips with the OTHER situation).
    * LESION (`BRAIN_AFFECTIVE_TOM_LESION=1`): on the bad-other (INCONGRUENT: the other feels bad, the system's own
      affect is neutral) turn the OTHER region's `affect_out` is cut -> the neural tone collapses to neutral -> the
      empathic lead VANISHES and the answer reverts BYTE-IDENTICALLY to the flag-OFF surface, while the content
      fields stay byte-identical. (The finding's egocentric|incongruent=0.000 vs other|incongruent=1.000, in
      production form: the lead rides the OTHER-region read, not a host `if valence<0`.)

The other default-ON drive faculties (#84 affect / #85 swap / #79 DA / #86 continuous / Gate-B affect / PMEM ...) are
DISABLED for the soak so the ONLY OFF->ON difference is the W5 empathic lead (a clean isolation, the same discipline
`_d5_graded_flip_soak` uses for its own flag). The conversation uses only NON-mutating turns (pre-existing recalls +
abstains + other-agent triggers) so a fresh session per mode is faithful.

  Run (1-seed confirm): SIM_BACKEND=numpy python -m research.runners._affective_tom_flip_soak --seeds 42
  Run (6-seed gate):    SIM_BACKEND=numpy python -m research.runners._affective_tom_flip_soak --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
# force the GPU-free stub renderer (no Qwen warm) for the whole soak.
os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "research" / "findings" / "raw" / "_affective_tom_prodflip" / "soak_seed42.json"

# ── the conversation: NON-mutating ordinary turns (pre-existing recall + abstain) interleaved with other-agent
#    triggers (bad / good). The empathic lead may appear ONLY on the trigger turns; every ordinary turn is
#    byte-identical OFF vs ON. Kinds: "ordinary" | "trigger_bad" | "trigger_good".
CONV = [
    ("what does the dog chase?", "ordinary"),          # pre-existing recall (no store mutation)
    ("what is the capital of france?", "ordinary"),    # abstain (moat)
    ("Maria is devastated", "trigger_bad"),            # another agent, bad situation -> comfort lead
    ("Tom is delighted", "trigger_good"),              # another agent, good situation -> share-joy lead
    ("Sam's team lost", "trigger_bad"),                # possessive-name trigger, bad situation -> comfort lead
    ("what does the dog chase?", "ordinary"),          # repeat recall (deterministic)
]

# other default-ON drive faculties silenced so the ONLY OFF->ON delta is the W5 empathic lead.
_SILENCE = ["BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES", "BRAIN_DA_ENCODING",
            "BRAIN_CONTINUOUS", "BRAIN_CONTINUOUS_DRIVES", "BRAIN_CONTINUOUS_IDEATE", "BRAIN_PMEM",
            "BRAIN_GNW_SWAP", "BRAIN_GNW_BUS"]

_KEEP = ("answer", "abstained", "recalled_svo", "verified")


def _slim(resp: dict) -> dict:
    """The stable conversational surface + the ToM trace presence/lead (volatile debug keys dropped)."""
    d = {k: resp.get(k) for k in _KEEP}
    d["has_tom_key"] = ("affective_tom" in resp)
    d["tom_lead"] = (resp.get("affective_tom") or {}).get("lead", "") if "affective_tom" in resp else ""
    d["tom_tone"] = (resp.get("affective_tom") or {}).get("tone_sign") if "affective_tom" in resp else None
    return d


def _fresh_chat(S, session):
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    ck = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[ck] = chat
    return ck


def _run_conversation(S, session, *, flag_on, lesion, seed):
    """Run CONV through the REAL handler on a FRESH session. `flag_on` toggles BRAIN_AFFECTIVE_TOM; `lesion` toggles
    the OTHER-region affect_out lesion. Points the process organ at `seed` for the ON run so the neural read is
    seed-controlled. Returns the per-turn slim surfaces."""
    from webapp.server import brain_chat, BrainChatRequest as Req
    import research.runners.affective_tom_production_organ as _ATM
    os.environ["BRAIN_AFFECTIVE_TOM"] = "1" if flag_on else "0"
    os.environ["BRAIN_AFFECTIVE_TOM_LESION"] = "1" if lesion else "0"
    if flag_on:
        _ATM._ORGAN = _ATM.AffectiveToMOrgan(seed=int(seed))   # seed-controlled OTHER-region read
    _fresh_chat(S, session)
    rows = []
    for msg, kind in CONV:
        r = brain_chat(Req(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=False))
        rows.append({"msg": msg, "kind": kind, "slim": _slim(json.loads(r.body.decode("utf-8")))})
    return rows


def run_one(seed, backend):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[tom-soak] seed={seed} backend={backend} — conversation OFF vs ON: only the TRIGGERED turns' empathic "
          f"lead may change; every ordinary turn byte-identical; lesion collapses the lead.", flush=True)
    import webapp.server as S
    for f in _SILENCE:
        os.environ[f] = "0"
    result = {"seed": seed, "backend": backend}
    try:
        off = _run_conversation(S, f"tom-soak-off-{seed}", flag_on=False, lesion=False, seed=seed)
        on = _run_conversation(S, f"tom-soak-on-{seed}", flag_on=True, lesion=False, seed=seed)
        les = _run_conversation(S, f"tom-soak-les-{seed}", flag_on=True, lesion=True, seed=seed)

        # ── NO-REGRESSION: ordinary turns byte-identical (FULL slim record) OFF vs ON ──
        ordinary_identical = True
        triggered_content_identical = True
        lead_present_on = True
        lesion_collapsed = True
        bad_leads, good_leads = [], []
        per_turn = []
        for o, n, l in zip(off, on, les):
            kind = o["kind"]
            rec = {"msg": o["msg"], "kind": kind, "off": o["slim"], "on": n["slim"], "lesion": l["slim"]}
            if kind == "ordinary":
                same = (o["slim"] == n["slim"])           # full byte-identity incl. NO affective_tom key
                rec["ordinary_identical"] = bool(same)
                ordinary_identical = ordinary_identical and same
            else:
                # content fields byte-identical OFF vs ON; the ON answer = lead + OFF answer (lead stripped -> equal)
                content_same = all(o["slim"][k] == n["slim"][k] for k in ("abstained", "recalled_svo", "verified"))
                lead = n["slim"]["tom_lead"]
                answer_is_lead_plus_off = bool(lead) and (n["slim"]["answer"] == lead + o["slim"]["answer"])
                rec["content_same"] = bool(content_same)
                rec["answer_is_lead_plus_off"] = bool(answer_is_lead_plus_off)
                rec["on_lead"] = lead
                triggered_content_identical = triggered_content_identical and content_same and answer_is_lead_plus_off
                lead_present_on = lead_present_on and bool(lead) and n["slim"]["has_tom_key"]
                # LESION: the lead vanishes -> the answer reverts byte-identically to the OFF surface; content same
                les_collapse = (l["slim"]["tom_lead"] == "" and l["slim"]["answer"] == o["slim"]["answer"]
                                and all(l["slim"][k] == o["slim"][k] for k in ("abstained", "recalled_svo", "verified")))
                rec["lesion_lead"] = l["slim"]["tom_lead"]
                rec["lesion_collapsed"] = bool(les_collapse)
                lesion_collapsed = lesion_collapsed and les_collapse
                (bad_leads if kind == "trigger_bad" else good_leads).append(lead)
            per_turn.append(rec)

        # VARY (load-bearing): the bad-other lead and the good-other lead DIFFER (comfort vs share-joy).
        vary_ok = bool(bad_leads and good_leads and all(b != g for b in set(bad_leads) for g in set(good_leads)))
        # and the sign is correct: bad -> tone -1, good -> tone +1 (from the on-run tone_sign)
        on_tones = {r["kind"]: r["on"]["tom_tone"] for r in per_turn if r["kind"] != "ordinary"}
        sign_ok = bool(on_tones.get("trigger_bad") == -1 and on_tones.get("trigger_good") == 1)

        GO = bool(ordinary_identical and triggered_content_identical and lead_present_on
                  and lesion_collapsed and vary_ok and sign_ok)
        result.update(dict(
            GO=GO, ordinary_identical=ordinary_identical, triggered_content_identical=triggered_content_identical,
            lead_present_on=lead_present_on, lesion_collapsed=lesion_collapsed, vary_ok=vary_ok, sign_ok=sign_ok,
            bad_leads=bad_leads, good_leads=good_leads, on_tones=on_tones, per_turn=per_turn))
        print(f"[tom-soak] ordinary_identical={ordinary_identical} | triggered_content_identical="
              f"{triggered_content_identical} | lead_present_on={lead_present_on}", flush=True)
        print(f"[tom-soak] VARY bad_leads={sorted(set(bad_leads))} good_leads={sorted(set(good_leads))} "
              f"vary_ok={vary_ok} sign_ok={sign_ok}", flush=True)
        print(f"[tom-soak] LESION collapsed={lesion_collapsed} (bad-other lead -> '' == OFF surface)", flush=True)
        print(f"[tom-soak] seed={seed} => {'GO' if GO else 'NO-GO'}", flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["GO"] = False; traceback.print_exc()
    finally:
        os.environ["BRAIN_AFFECTIVE_TOM"] = "0"
        os.environ["BRAIN_AFFECTIVE_TOM_LESION"] = "0"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from sim.backend import get_backend
    _, backend = get_backend()
    results = {}; go = []
    for seed in a.seeds:
        r = run_one(seed, backend)
        results[seed] = r; go.append(bool(r.get("GO")))
    out_path = Path(a.out)
    if len(a.seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(a.seeds)}seed.json"
        print("\n" + "#" * 118)
        print(f"[tom-soak] {len(a.seeds)}-SEED SOAK: {int(sum(go))}/{len(a.seeds)} GO seeds={a.seeds}")
        print("#" * 118)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": a.seeds, "n_go": int(sum(go)), "go": go, "backend": backend,
                                    "results": {str(s): results[s] for s in a.seeds}}, indent=2, default=str))
    print(f"[tom-soak] wrote {out_path}")
    return 0 if (go and all(go)) else 1


if __name__ == "__main__":
    sys.exit(main())
