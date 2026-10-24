"""Compact the raw drive output (drive_memo.py, one rich "What is <slug>?" turn per selected entity, Qwen-0.5B on CPU,
BRAIN_MOAT_SLUG_NORM unset) into corpus.json: per turn the gathered set (the facts RichAnswerComposer.render_paragraph was
handed) and every Qwen sentence the mouth produced for it (mouth_trace calls: render_svo and render_svo_regen per fact).
No verifier is run here and no label exists yet.   usage: extract_corpus.py <out.json> <raw1.json> [<raw2.json> ...]   (one raw file per generation batch)"""
import collections, hashlib, json, os, socket, subprocess, sys

import research.runners  # noqa: F401  (the provenance door: stamps corpus.json with run id, host, backend)

out_path, raw_paths = sys.argv[1], sys.argv[2:]
raw_hashes = []
d = {"responses": {}, "trace": [], "renderer": None, "ltm_bundle": None}
src_of = {}
for rp in raw_paths:
    rb = open(rp, "rb").read()
    part = json.loads(rb)
    raw_hashes.append({"file": os.path.basename(rp), "sha256": hashlib.sha256(rb).hexdigest(), "bytes": len(rb),
                       "renderer": part.get("renderer"),
                       "generation_device": {"qwen-cpu": "cpu (Qwen-0.5B fp32)", "qwen-gpu": "cuda (Qwen-0.5B fp16)"}.get(part.get("renderer"), part.get("renderer")),
                       "generation_host": part.get("host", "not recorded in this raw file (written before drive_memo recorded it)")})
    d["responses"].update(part["responses"]); d["trace"] += part["trace"]
    for lab_ in part["responses"]:
        src_of[lab_] = os.path.basename(rp)
    d["renderer"] = part.get("renderer"); d["ltm_bundle"] = part.get("ltm_bundle")
ents = json.load(open("research/findings/raw/_moat_slug_norm_heldout/entities.json"))["entities"]
paras = collections.defaultdict(list)
for t in d["trace"]:
    if t.get("fn") == "render_paragraph":
        paras[t["turn"].split("|")[0]].append([list(f) for f in t["facts"]])
turns = []
for i, ent in enumerate(ents, 1):
    lab = "h%02d" % i
    r = d["responses"].get(lab)
    if not isinstance(r, dict):
        turns.append({"label": lab, "entity": ent, "error": "no response"}); continue
    calls = [{"method": c["method"], "svo": list(c["svo"]), "surface": c["surface"]}
             for c in ((r.get("mouth_trace") or {}).get("calls") or [])]
    seen = []
    for c in calls:
        if c["svo"] not in seen: seen.append(c["svo"])
    gated = paras[lab][0] if paras.get(lab) else None
    turns.append({"label": lab, "entity": ent, "raw_file": src_of.get(lab), "question": "What is %s?" % ent, "gated": gated if gated is not None else seen,
                  "gated_source": "render_paragraph.facts" if gated is not None else "mouth_trace svo (no render_paragraph)",
                  "gated_equals_calls_svo": (gated == seen) if gated is not None else True,
                  "n_paragraph_calls": len(paras.get(lab, [])), "calls": calls,
                  "answer": r.get("answer"), "abstained": r.get("abstained")})
try:
    _gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception as e:
    _gpus = "unavailable: %r" % (e,)
out = {"_doc": "held-out real-Qwen corpus; no verifier verdict and no label in this file. Turns h01-h16 were generated with the "
               "qwen-cpu renderer (raw files hb1, hb2), h17-h29 with qwen-gpu (hb5): see raw_files[].generation_device. "
               "extraction_host / extraction_host_gpus are where THIS extraction ran (the generation host is per raw file where it was recorded).",
       "backend": "numpy", "device": "cpu (extraction only; generation devices per raw file)",
       "extraction_host": socket.gethostname(), "extraction_host_gpus": _gpus,
       "renderer": d.get("renderer"), "ltm_bundle": d.get("ltm_bundle"),
       "script": "research/findings/raw/_moat_slug_norm_heldout/extract_corpus.py", "argv": sys.argv,
       "raw_files": raw_hashes,
       "n_turns": len(turns), "turns": turns}
json.dump(out, open(out_path, "w"), indent=1)
n_with = sum(1 for t in turns if t.get("calls"))
print("turns", len(turns), "with calls", n_with, "raw sentences", sum(len(t.get("calls", [])) for t in turns),
      "unique (turn,svo,surface)", len({(t["label"], tuple(c["svo"]), c["surface"]) for t in turns for c in t.get("calls", [])}))
