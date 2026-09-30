"""DEV-SEED PROBE (not a registered arm): does the affect lesion change the reply the production mouth writes when the
mood is engaged AND the reply is rendered?

Context: B2d Amendment 1 (research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md) predicts
that the mouth arm's registered turns cannot show it -- `emo` is mood-engaged but its reply is an abstention template
the mouth never renders, and `well` is rendered but carries no mood. This probe runs the candidate remap the
packet's NEEDS-CLAUDE #3 names, at a DEV seed only: the `emo` message, then a RECALL question ("what does the dog
chase", the `sw_open` message) in the SAME session (no reset), so the session mood set by the first turn is still
present when the second turn's recalled triple is rendered. (Dev seed 7 showed that `well`, an assertion, is echoed
without any renderer call, so it cannot carry the mood to a mouth.) At defaults a bounded-transitive recall is spoken by
the SPIKING recall mouth (BRAIN_SPIKING_MOUTH_RECALL, default ON since 2026-08-26), which pre-empts the Qwen renderer
and carries its own spiking mood-tone read (`_apply_mouth_mood_tone`); it runs with the stub renderer too, so this
probe is run in BOTH renderer modes (LB_ARM_RENDERER unset = stub, and production).

One arm per process (a fresh brain, like every battery arm), through the battery's own worker
(`onebrain_regression_battery._collect_worker`, so LB_ARM_RENDERER=production renders exactly as the mouth arm does);
the two probe turns are added to the worker's turn table in THIS process only (the harness file is not changed):

  SIM_BACKEND=cupy LB_ARM_RENDERER=production BRAIN_CHAT_SEED=7 .venv/bin/python -u -m \
      research.runners._b2d_mood_render_probe --arm intact_a --out <dir>/intact_a.json
  ... --arm intact_b (the null rebuild) ... --arm lesion (BRAIN_AFFECT_LESION=1, the affect-coloring lesion)

Output: the worker's per-turn responses plus {"probe": ..., "backend", "device", "arm"}. Dev seeds only: the script
refuses a gate seed (42/43/44/100/101/102).

  ... -m research.runners._b2d_mood_render_probe --compare <dir> --out <dir>/compare.json
reads the three arm files and makes the attribution call (tools.lab.attributable_to): TREATMENT = reply fields
(answer, abstained, recalled_svo) of `mr_q` that differ intact_a vs lesion; CONTROL (the null) = the same count
intact_a vs intact_b. It also reports the mood level and manner template each arm carried into `mr_q` and how many
Qwen generation calls rendered it. A dev-seed reading only: a remap is registered by an amendment, never here.
"""
import argparse
import json
import os
import sys

GATE_SEEDS = {"42", "43", "44", "100", "101", "102"}
ARMS = {"intact_a": {}, "intact_b": {}, "lesion": {"BRAIN_AFFECT_LESION": "1"}}
TURNS = ("mr_emo", "mr_q")


REPLY = ("answer", "abstained", "recalled_svo")


def compare(d):
    """The dev reading over <d>/{intact_a,intact_b,lesion}.json (see the module docstring)."""
    from tools.lab import attributable_to
    arms = {k: json.load(open(os.path.join(d, "%s.json" % k)))["responses"]["mr_q"] for k in ARMS}
    diff = lambda x, y: sum(1 for f in REPLY if arms[x].get(f) != arms[y].get(f))  # noqa: E731
    treatment, control = diff("intact_a", "lesion"), diff("intact_a", "intact_b")
    out = {"treatment_reply_fields_changed": treatment, "control_reply_fields_changed": control,
           "attributable_fraction": attributable_to("affect lesion on the mood-engaged recall reply (mr_q)",
                                                    treatment, control),
           "per_arm": {k: {"answer": v.get("answer"), "abstained": v.get("abstained"),
                           "tone_level": (v.get("affect") or {}).get("tone_level"),
                           "manner_template": (v.get("affect") or {}).get("manner_template"),
                           "renderer": v.get("renderer"), "arm_renderer": v.get("_arm_renderer")}
                       for k, v in arms.items()}}
    out["reading"] = ("UNDEFINED:null-noise" if control else
                      ("mouth-silent" if all((arms[k].get("_arm_renderer") or {}).get("qwen_generate_calls") == 0
                                             for k in ARMS) else
                       ("reply changed under the lesion" if treatment else "reply unchanged under the lesion")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--compare", default=None, help="a directory holding the three arm files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.compare:
        res = compare(a.compare)
        res.update({"backend": "n/a (reads arm files)", "device": "n/a"})
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(json.dumps({k: res[k] for k in ("reading", "treatment_reply_fields_changed",
                                              "control_reply_fields_changed", "attributable_fraction")}))
        return 0
    if not a.arm:
        ap.error("--arm or --compare is required")
    seed = os.environ.get("BRAIN_CHAT_SEED", "42")
    if seed in GATE_SEEDS:
        print("REFUSED: BRAIN_CHAT_SEED=%s is a gate seed; this probe is dev-seed only" % seed, file=sys.stderr)
        return 2
    import research.runners.onebrain_regression_battery as B
    emo_msg = B._TURN_BY_LABEL["emo"][1]
    q_msg = B._TURN_BY_LABEL["sw_open"][1]
    B._TURN_BY_LABEL["mr_emo"] = ("mr_emo", emo_msg, "mood_render", True, None, False)
    B._TURN_BY_LABEL["mr_q"] = ("mr_q", q_msg, "mood_render", False, None, False)
    tmp = a.out + ".worker.json"
    rc = B._collect_worker(json.dumps(ARMS[a.arm]), list(TURNS), tmp)
    if rc != 0:
        return rc
    responses = json.load(open(tmp))
    os.remove(tmp)
    out = {"probe": "b2d mouth remap candidate (dev seed): emo then a recall question in one session", "arm": a.arm,
           "arm_env": ARMS[a.arm], "seed": seed, "renderer_mode": B.arm_renderer_mode(),
           "backend": os.environ.get("SIM_BACKEND"), "device": "cuda" if os.environ.get("SIM_BACKEND") == "cupy"
           else "cpu", "turns": {t: B._TURN_BY_LABEL[t][1:] for t in TURNS}, "responses": responses}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("[mood-render probe] arm=%s seed=%s -> %s" % (a.arm, seed, a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
