#!/usr/bin/env python3
"""Blind rubric grading for the model bake-off (docs: tools/model_bakeoff/README.md).

For every result.json in a batch whose task is graded by rubric, two judges from different vendors (models.json
"judges") each score the run against the task's rubric and reference answer, without being told which model did the
work. Every request carries provider.data_collection=deny. The run's score is the mean of the judges' normalized
totals; a spread above 0.25 marks it `needs_adjudication` (Claude settles those by reading the run). Re-running
skips judgments already on disk.

  python3 judge.py --batch screen1
"""
import argparse
import glob
import json
import os
import re
import time

from run import BK, RUNS, SUITE, TOOLS, load_json, or_request, slug

PROMPT = """You are grading, blind, the work of an autonomous AI agent on a task from a research software project
(a spiking neural simulator; the project's rules are strict: brain-based mechanisms only, six seeds, gates
authoritative, honest negatives). You do not know which model did the work, and it does not matter.

Grade strictly against the RUBRIC, using the REFERENCE ANSWER (established from the project's later record) as
ground truth. Award each criterion's points only as the rubric allows. Do not reward length, confidence or style;
penalise confident claims the reference contradicts, and invented facts. If the agent did not produce the
deliverable, the criteria that depend on it score 0.

Reply with ONLY a JSON object:
{"criteria": [{"id": "<criterion>", "awarded": <number>, "max": <number>, "reason": "<one sentence>"}],
 "total": <number>, "max_total": <number>, "notes": "<at most two sentences>"}

=== TASK (what the agent was asked) ===
%(prompt)s

=== RUBRIC ===
%(rubric)s

=== REFERENCE ANSWER ===
%(reference)s

=== AGENT'S ANSWER FILE (BAKEOFF_ANSWER.md) ===
%(answer)s

=== AGENT'S FINAL MESSAGE ===
%(final)s

=== AGENT'S CHANGES TO THE REPOSITORY (diff, possibly truncated) ===
%(diff)s
"""


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in judge reply")
    d = json.loads(m.group(0))
    total, mx = float(d["total"]), float(d["max_total"])
    if mx <= 0 or total < 0 or total > mx * 1.001:
        raise ValueError("judge totals out of range: %r/%r" % (total, mx))
    return d


def judge_once(judge_model, material):
    body = {"model": judge_model, "temperature": 0, "max_tokens": 3000,
            "messages": [{"role": "user", "content": PROMPT % material}],
            "provider": {"data_collection": "deny"}}
    last = None
    for attempt in range(3):
        try:
            r = or_request("/chat/completions", body, timeout=300)
            text = r["choices"][0]["message"]["content"]
            d = parse_json(text)
            d["usage"] = r.get("usage")
            return d
        except Exception as e:      # retry transient errors and unparseable replies
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("judge %s failed: %r" % (judge_model, last))


def read(path, limit):
    try:
        t = open(path, errors="replace").read()
    except OSError:
        return "(none)"
    return t if len(t) <= limit else t[:limit] + "\n...[truncated %d chars]" % (len(t) - limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    a = ap.parse_args()
    judges = load_json(os.path.join(TOOLS, "models.json"))["judges"]
    tasks = {t["id"]: t for t in load_json(os.path.join(TOOLS, SUITE))["tasks"]}
    for res_path in sorted(glob.glob(os.path.join(RUNS, a.batch, "*", "*", "result.json"))):
        res = load_json(res_path)
        task = tasks.get(res["task"])
        if not task or task["grading"]["type"] != "rubric":
            continue
        run_dir = os.path.dirname(res_path)
        material = {"prompt": task["prompt"], "rubric": task["grading"].get("rubric", ""),
                    "reference": task["grading"].get("reference_answer", ""),
                    "answer": read(os.path.join(run_dir, "answer.md"), 30000),
                    "final": (res.get("final_text") or "(none)")[-6000:],
                    "diff": read(os.path.join(run_dir, "diff.patch"), 40000)}
        scores = []
        for jm in judges:
            jp = os.path.join(run_dir, "judge-%s.json" % slug(jm))
            if not os.path.exists(jp):
                try:
                    d = judge_once(jm, material)
                except Exception as e:
                    print("JUDGE ERROR %s %s %s: %s" % (res["model"], res["task"], jm, str(e)[:200]), flush=True)
                    continue
                with open(jp, "w") as f:
                    json.dump(d, f, indent=1)
            d = load_json(jp)
            scores.append(float(d["total"]) / float(d["max_total"]))
        if not scores:
            continue
        g = res["grade"]
        g["judge_scores"] = [round(s, 3) for s in scores]
        g["score"] = round(sum(scores) / len(scores), 3)
        g["needs_adjudication"] = len(scores) < len(judges) or (max(scores) - min(scores) > 0.25)
        g.pop("pending", None)
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)
        print("%-34s %-44s score %.2f %s" % (res["model"][:34], res["task"][:44], g["score"],
              "ADJUDICATE" if g["needs_adjudication"] else ""), flush=True)


if __name__ == "__main__":
    main()
