#!/usr/bin/env python3
"""Offline (no GPU, no llama-server) check for the patched chat templates in this directory.

Renders each patched *.jinja template with plain `jinja2` (stdlib-only aside from the `jinja2`
package itself) against a message sequence shaped like what Claude Code actually sends through
llama-server's Anthropic /v1/messages -> chat-template bridge: one or more LEADING system blocks
(Claude Code's system prompt is often split into several blocks) plus a SECOND, mid-conversation
system message (a "system reminder" injected between turns) and tool_use/tool_result turns.

For each profile this asserts:
  1. the ORIGINAL template raises on that exact message sequence (reproduces the bake-off failure
     -- if this stops raising, the upstream .gguf template changed and this test's premise is stale);
  2. the PATCHED template renders it with NO exception;
  3. the tool-call syntax the patched template emits for the non-system turns is BYTE-IDENTICAL to
     what the ORIGINAL template emits for the same turns with all system messages stripped out (the
     one message sequence the original template can already render without raising). This isolates
     "did the system-handling patch touch tool-call formatting" (it must not) from "did it change
     where/how system text is placed" (it deliberately does).

`jinja2` here is a stand-in for llama.cpp's C++ "minja" template engine: both are ordinary Jinja
dialects, and every construct these two templates use (namespace(), the `tojson`/`items`/`safe`/
`trim`/`sort(attribute=)`/`string`/`default` filters, `is string|mapping|iterable|none|undefined`,
`raise_exception(...)`, slicing, loop.previtem/nextitem/first/last/index0) is one the upstream
templates already relied on before this patch, so minja is known to support it; this patch adds no
new construct beyond what was already exercised by the *.orig.jinja files.

Usage: .venv/bin/python3 tools/local_llm/templates/test_templates_offline.py
"""
import os
import re
import sys

try:
    import jinja2
except ImportError:
    print("SKIP: jinja2 not installed in this interpreter", file=sys.stderr)
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))


class TemplateRaised(Exception):
    """Distinguishes an in-template raise_exception() call from a jinja2/Python bug."""


def make_env():
    env = jinja2.Environment(
        trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True,
        undefined=jinja2.ChainableUndefined,
    )
    env.filters["tojson"] = lambda v, indent=None: __import__("json").dumps(v, indent=indent)
    env.filters["items"] = lambda d: list(d.items())

    def _raise(msg):
        raise TemplateRaised(msg)

    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: __import__("datetime").datetime.now().strftime(fmt)
    return env


def render(template_path, messages, **kwargs):
    env = make_env()
    tpl = env.from_string(open(template_path, encoding="utf-8").read())
    ctx = dict(messages=messages, add_generation_prompt=True, bos_token="<s>", eos_token="</s>")
    ctx.update(kwargs)
    return tpl.render(**ctx)


# -- A Claude-Code-shaped conversation: leading system blocks, tool_use/tool_result turns, and a
#    mid-conversation "system reminder" (the pattern that made llama-server's template conversion
#    fail in the bake-off). --------------------------------------------------------------------
def cc_messages():
    # Each user turn's tool round-trip is closed by a plain (no tool_calls) assistant reply before
    # the next user turn, matching how Claude Code / Mistral-style tool-calling conversations are
    # actually shaped -- Devstral's OWN template independently enforces strict user/assistant
    # alternation across the turns it counts (a tool_calls-bearing assistant message and a tool
    # message are exempt from that count, a plain assistant reply is not), so a synthetic sequence
    # that skipped the closing reply would trip that pre-existing, correct constraint and not the
    # system-message bug this test targets.
    return [
        {"role": "system", "content": "You are Claude Code, Anthropic's official CLI for Claude."},
        {"role": "system", "content": "<system-reminder>Tool results are wrapped in tags; be concise.</system-reminder>"},
        {"role": "user", "content": "Find the file that implements the delete guard."},
        {"role": "assistant", "content": "I'll search the repo.",
         "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "grep -rl delete_guard .claude/hooks"}}}]},
        {"role": "tool", "content": ".claude/hooks/guard_protected_delete.py"},
        {"role": "assistant", "content": "Found it: .claude/hooks/guard_protected_delete.py"},
        {"role": "system", "content": "<system-reminder>Plan mode is now active; do not edit files yet.</system-reminder>"},
        {"role": "user", "content": "Now add a test case for it."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "edit_file",
                                       "arguments": {"path": "tests/test_guard.py", "old_str": "pass", "new_str": "assert ok"}}}]},
        {"role": "tool", "content": "edit applied"},
        {"role": "assistant", "content": "Added the test case."},
        {"role": "user", "content": "Run the suite."},
    ]


def strip_system(messages):
    return [m for m in messages if m["role"] != "system"]


TOOLCALL_RE = {
    "qwen38-27b-iq4nl-mtp": re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    "devstral-small2-24b-iq4xs": re.compile(r"\[TOOL_CALLS\].*?(?=</s>)", re.DOTALL),
}


def check_profile(name):
    orig_path = os.path.join(HERE, name + ".orig.jinja")
    patched_path = os.path.join(HERE, name + ".jinja")
    messages_full = cc_messages()
    messages_stripped = strip_system(messages_full)

    # 1. reproduce the bake-off failure on the ORIGINAL template + the full (multi-system) sequence.
    try:
        render(orig_path, messages_full)
    except TemplateRaised as e:
        reproduced = str(e)
    else:
        raise AssertionError("%s: ORIGINAL template did NOT raise on a mid-conversation system "
                              "message -- the premise of this fix may be stale, re-check the "
                              "upstream .gguf template" % name)

    # 2. the PATCHED template must render the same full sequence with no exception.
    patched_out = render(patched_path, messages_full)

    # 3. the ORIGINAL template must render fine once system messages are stripped (this is the
    #    reference rendering we diff tool-call syntax against).
    orig_out = render(orig_path, messages_stripped)

    # 4. tool-call syntax must be byte-identical between the two.
    rx = TOOLCALL_RE[name]
    patched_calls = rx.findall(patched_out)
    orig_calls = rx.findall(orig_out)
    assert len(patched_calls) == 2, "%s: expected 2 tool calls in patched output, got %d" % (name, len(patched_calls))
    assert len(orig_calls) == 2, "%s: expected 2 tool calls in stripped-original output, got %d" % (name, len(orig_calls))
    assert patched_calls == orig_calls, (
        "%s: tool-call syntax DIVERGED between the patched and original templates:\n--- patched ---\n%s\n"
        "--- original ---\n%s" % (name, patched_calls, orig_calls))

    # 5. sanity: the merged system text from BOTH system messages actually made it into the output
    #    (i.e. the fix didn't just swallow the exception and drop the content).
    for needle in ("Claude Code, Anthropic's official CLI", "Plan mode is now active"):
        assert needle in patched_out, "%s: merged system text %r missing from patched output" % (name, needle)

    return {"reproduced_original_error": reproduced[:200], "tool_calls_matched": len(patched_calls)}


def check_regressions(name):
    """The no-system-message case (the model's own default persona, untouched by this fix in either
    template) must render IDENTICALLY through the patched template.

    The single-leading-system-message case is a DELIBERATE behavior change for Devstral only, per the
    task design: Qwen still has a native system slot, so a lone leading system message renders exactly
    as before; Devstral has "no system role" for this fix to reuse, so per design ALL system content
    -- leading or not -- is folded into the first user turn rather than kept as the model's single-use
    [SYSTEM_PROMPT] slot (which is also the only way multiple LEADING system blocks, which Claude Code
    also sends, could be merged for Devstral at all). Qwen's single-leading case is checked for byte
    equality; Devstral's is checked for the intended fold instead.
    """
    orig_path = os.path.join(HERE, name + ".orig.jinja")
    patched_path = os.path.join(HERE, name + ".jinja")

    no_system = [m for m in cc_messages() if m["role"] != "system"]
    out_orig = render(orig_path, no_system)
    out_patched = render(patched_path, no_system)
    assert out_orig == out_patched, "%s: no-system-message output changed by the patch" % name

    one_leading = [cc_messages()[0]] + no_system
    if name == "qwen38-27b-iq4nl-mtp":
        out_orig = render(orig_path, one_leading)
        out_patched = render(patched_path, one_leading)
        assert out_orig == out_patched, "%s: single-leading-system-message output changed by the patch" % name
    else:
        out_patched = render(patched_path, one_leading)
        assert "[SYSTEM_PROMPT]" not in out_patched, (
            "%s: single leading system message should be FOLDED into the first user turn, not kept as "
            "[SYSTEM_PROMPT] (that mechanism can't hold a later mid-conversation reminder too)" % name)
        assert "[INST]You are Claude Code, Anthropic's official CLI for Claude.\n\nFind the file" in out_patched, (
            "%s: leading system text was not folded into the first [INST] turn as designed" % name)


def main():
    results = {}
    for name in ("qwen38-27b-iq4nl-mtp", "devstral-small2-24b-iq4xs"):
        results[name] = check_profile(name)
        print("PASS %-28s original raised: %s" % (name, results[name]["reproduced_original_error"]))
        print("     %-28s tool calls byte-identical: %d/2" % ("", results[name]["tool_calls_matched"]))
        check_regressions(name)
        print("     %-28s no-system-message output UNCHANGED; system folding matches design" % "")
    print("\nALL OFFLINE TEMPLATE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
