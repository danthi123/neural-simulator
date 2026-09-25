#!/usr/bin/env python3
"""render_and_diff.py -- render two captured REAL request bodies (from capture_session.py) through a chat
template via a live llama-server's /apply-template endpoint, and diff the rendered text to find the exact first
divergence position.

research/local-llm-prompt-cache branch, 2026-09-25 round 2 (coordinator directive): "capture two consecutive real
request bodies... render each through the template, and diff the rendered token/text sequences to find the exact
first divergence position and what differs."

/apply-template only understands the OpenAI chat-completions shape ({"messages": [...]}, each entry role
user/assistant/system/tool) -- confirmed empirically (tools/local_llm/results/template_divergence/
endpoint_probe.json): a top-level Anthropic-style "system" field is silently IGNORED by this endpoint. Captured
request bodies are the real Anthropic Messages API JSON (top-level "system" + "messages" with Anthropic content
blocks), so this script converts Anthropic -> OpenAI shape itself before calling /apply-template. The conversion
mirrors llama-server's own `server_chat_convert_anthropic_to_oai` closely enough for RENDERING purposes: it does
not need to be byte-perfect, only consistent between the two captures being compared.

Usage: render_and_diff.py <capture_file_1.json> <capture_file_2.json> [--template PATH] [--out PATH]
Starts its own llama-server (needs the GPU queue), no generation is performed (only /apply-template calls).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_probe as cp

PORT = 8093


def _text_of_blocks(blocks):
    """Anthropic content can be a plain string or a list of blocks; join every "text" block's text."""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    parts = []
    for b in blocks:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def anthropic_to_openai_messages(body):
    """body: a decoded Anthropic /v1/messages request. Returns (messages, tools) in OpenAI chat-completions
    shape, suitable for POSTing to /apply-template. Records, per system source, what was found -- so the caller
    can report exactly where "system" content came from (top-level `system` field vs a `role: "system"` entry
    already embedded in `messages`, which is the crux of hypothesis 1)."""
    messages = []
    provenance = {"top_level_system_blocks": 0, "embedded_system_messages_in_array": 0,
                  "embedded_system_positions": []}

    system = body.get("system")
    if system is not None:
        text = _text_of_blocks(system)
        if isinstance(system, list):
            provenance["top_level_system_blocks"] = len(system)
        elif text:
            provenance["top_level_system_blocks"] = 1
        if text.strip():
            messages.append({"role": "system", "content": text})

    for i, m in enumerate(body.get("messages", [])):
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            provenance["embedded_system_messages_in_array"] += 1
            provenance["embedded_system_positions"].append(i)
            messages.append({"role": "system", "content": _text_of_blocks(content)})
            continue
        if role == "user":
            # A user turn's content can mix plain text blocks with tool_result blocks (Anthropic's way of
            # returning a tool's output). Split those into a preceding "tool" message per llama.cpp/OpenAI
            # convention; keep any plain text as the user message itself.
            if isinstance(content, list):
                text_parts = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        tool_text = _text_of_blocks(b.get("content"))
                        messages.append({"role": "tool", "content": tool_text,
                                          "tool_call_id": b.get("tool_use_id", "")})
                    elif b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                if text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                messages.append({"role": "user", "content": _text_of_blocks(content)})
            continue
        if role == "assistant":
            text_parts, tool_calls, reasoning_parts = [], [], []
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        tool_calls.append({"type": "function", "id": b.get("id", ""),
                                            "function": {"name": b.get("name", ""), "arguments": b.get("input", {})}})
                    elif b.get("type") == "thinking":
                        # Verified byte-identical between what the model generated (the SSE response capture)
                        # and what Claude Code resends as history (hypothesis 2 check) -- forwarded into
                        # reasoning_content here so the template's own preserve_thinking/<think> logic applies,
                        # matching how llama-server's real Anthropic bridge must feed it back in.
                        reasoning_parts.append(b.get("thinking", ""))
            else:
                text_parts = [_text_of_blocks(content)]
            entry = {"role": "assistant", "content": "\n".join(text_parts)}
            if reasoning_parts:
                entry["reasoning_content"] = "\n".join(reasoning_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        messages.append({"role": role, "content": _text_of_blocks(content)})

    tools = []
    for t in body.get("tools") or []:
        tools.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description"),
                                                         "parameters": t.get("input_schema", {})}})
    return messages, tools, provenance


def apply_template(messages, tools):
    payload = {"messages": messages}
    if tools:
        payload["tools"] = tools
    import urllib.error
    import urllib.request
    req = urllib.request.Request("http://127.0.0.1:%d/apply-template" % PORT,
                                  data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["prompt"]
    except urllib.error.HTTPError as e:
        raise RuntimeError("/apply-template HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:2000])) from None


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture1")
    ap.add_argument("capture2")
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cp.PORT = PORT
    profile = dict(cp.load_default_profile())
    if a.template:
        profile["chat_template_file"] = os.path.relpath(a.template, cp.ROOT)
    cmd = cp.build_cmd(profile, 1, [])
    log_path = os.path.join(cp.HERE, "results", "template_divergence", "render_and_diff.server.log")
    proc = None
    report = {}
    try:
        proc, load_s = cp.start_server(cmd, log_path)
        b1 = json.load(open(a.capture1))
        b2 = json.load(open(a.capture2))
        m1, t1, prov1 = anthropic_to_openai_messages(b1)
        m2, t2, prov2 = anthropic_to_openai_messages(b2)
        p1 = apply_template(m1, t1)
        p2 = apply_template(m2, t2)
        idx = first_divergence(p1, p2)
        report = {
            # Not a sim/research run (no SIM_BACKEND/cupy/numpy applies) -- this is llama-server's OWN
            # /apply-template endpoint over HTTP, on whatever GPU llama-server itself loaded onto (see
            # cache_probe.md's vram_peak_mib columns for the actual runs' device evidence). Recorded explicitly
            # so tools/gates/device_and_cost.py's "no backend recorded" check has a truthful answer rather than
            # silently matching "not recorded" (which that gate treats as "an unintended CPU run").
            "backend": "llama.cpp/apply-template (HTTP; not a SIM_BACKEND numpy/cupy run)",
            "capture1": a.capture1, "capture2": a.capture2, "template": a.template or profile["chat_template_file"],
            "provenance1": prov1, "provenance2": prov2,
            "rendered_len1": len(p1), "rendered_len2": len(p2),
            "first_divergence_char_index": idx,
            "context_before_1": p1[max(0, idx - 200):idx], "context_at_1": p1[idx:idx + 300],
            "context_before_2": p2[max(0, idx - 200):idx], "context_at_2": p2[idx:idx + 300],
            "divergence_frac_of_p1": round(idx / len(p1), 4) if len(p1) else None,
            "divergence_frac_of_p2": round(idx / len(p2), 4) if len(p2) else None,
        }
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)
            open(a.out + ".p1.txt", "w").write(p1)
            open(a.out + ".p2.txt", "w").write(p2)
        print(json.dumps({k: v for k, v in report.items() if not k.startswith("context")}, indent=1))
        print("--- context before divergence (capture1) ---")
        print(report["context_before_1"][-200:])
        print("--- AT divergence, capture1 ---")
        print(report["context_at_1"])
        print("--- AT divergence, capture2 ---")
        print(report["context_at_2"])
    finally:
        cp.stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
