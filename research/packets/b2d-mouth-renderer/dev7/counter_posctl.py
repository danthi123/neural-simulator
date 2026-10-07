"""Review positive control (dev seed 7, CPU): does the battery's Qwen counter see REAL generation calls made by the
warm renderer inside the real brain_chat handler? Spiking recall mouth OFF so a recall is rendered by Qwen."""
import json, os, sys, time
os.environ.update(SIM_NO_PROVENANCE="1", SIM_BACKEND="numpy", CUDA_VISIBLE_DEVICES="", BRAIN_CHAT_SEED="7",
                  BRAIN_SPIKING_MOUTH_RECALL=os.environ.get("POSCTL_SPIKING_MOUTH", "0"))
import research.runners.onebrain_regression_battery as B
from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
c = B._QwenGenerateCounter().install(SpikingQwenFaculty)
from webapp.server import brain_chat, BrainChatRequest
out = {}
for label in ("sw_open", "well", "emo"):
    _, msg, session, reset, percept, rich = B._TURN_BY_LABEL[label]
    t0 = time.time(); before = c.calls
    r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="qwen",
                                    rich=bool(rich), reset=reset))
    d = json.loads(r.body)
    out[label] = {"qwen_generate_calls": c.calls - before, "renderer": d.get("renderer"), "answer": d.get("answer"),
                  "abstained": d.get("abstained"), "secs": round(time.time() - t0, 1)}
    print(label, json.dumps(out[label]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
