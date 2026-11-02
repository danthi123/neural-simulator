"""secret_scan (class SEC, BLOCK): a credential committed into the repo.

WHY (2026-11-02 lean-method). The repo is pushed to two remotes (tools/push_both.sh), one of them public on GitHub,
and Hermes and Pi publish commits through sim-sync. A pushed key cannot be un-published; this is the most
irreversible mistake a commit can make, and nothing checked for it.

WHAT IT ENFORCES: no line ADDED by the staged diff (and no line of a given path) matches a known credential shape
(Anthropic/OpenAI/OpenRouter/GitHub/AWS/Slack/HF keys, private-key blocks). A test fixture can opt out on that line
with the marker `secrets-ok` (a fake value, deliberately).

WHAT IT CANNOT CATCH: a credential in an unknown format, or one written into a binary file.
"""
import os
import re
import subprocess

NAME = "secret-scan"
CLASS_ID = "SEC"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATTERNS = [
    ("anthropic key", r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("openrouter key", r"sk-or-v1-[A-Za-z0-9]{20,}"),
    ("openai key", r"sk-(?:proj-)?[A-Za-z0-9]{32,}"),
    ("github token", r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{30,}"),
    ("aws access key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("slack token", r"\bxox[abpr]-[A-Za-z0-9-]{20,}"),
    ("huggingface token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("private key", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
]
_RE = [(n, re.compile(p)) for n, p in _PATTERNS]
_MAX_BYTES = 2_000_000


def scan_line(line):
    if "secrets-ok" in line:
        return None
    for name, rx in _RE:
        if rx.search(line):
            return name
    return None


def _staged_added_lines():
    """[(path, line)] for every '+' line in the staged diff (text files only)."""
    try:
        out = subprocess.run(["git", "-C", _ROOT, "diff", "--cached", "-U0", "--no-color", "--diff-filter=ACMR"],
                             capture_output=True, text=True, timeout=20, errors="ignore").stdout
    except Exception:
        return []
    res, cur = [], None
    for line in out.splitlines():
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and cur:
            res.append((cur, line[1:]))
    return res


def check(paths):
    if paths is None:
        return []
    problems, seen = [], set()
    for rel, line in _staged_added_lines():
        hit = scan_line(line)
        if hit and (rel, hit) not in seen:
            seen.add((rel, hit))
            problems.append("%s: staged line looks like a %s. Remove it (rotate the key if it was real); a fake "
                            "fixture value can carry `secrets-ok` on the same line." % (rel, hit))
    for rel in paths:
        full = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
        try:
            if os.path.getsize(full) > _MAX_BYTES:
                continue
            text = open(full, errors="ignore").read()
        except OSError:
            continue
        for line in text.splitlines():
            hit = scan_line(line)
            if hit and (rel, hit) not in seen:
                seen.add((rel, hit))
                problems.append("%s: a line looks like a %s." % (rel, hit))
    return problems


def selftest():
    bad = []
    fake = "sk-ant-" + "api03-" + "A" * 40
    if scan_line("ANTHROPIC_API_KEY=%s" % fake) != "anthropic key":
        bad.append("did NOT catch an Anthropic key")
    if not scan_line("aws_access_key_id = AKIA" + "ABCDEFGHIJKLMNOP"):
        bad.append("did NOT catch an AWS access key")
    if not scan_line("-----BEGIN OPENSSH " + "PRIVATE KEY-----"):
        bad.append("did NOT catch a private key block")
    if scan_line("key = %s  # secrets-ok fixture" % fake):
        bad.append("ignored the secrets-ok opt-out")
    if scan_line("the sk-ant prefix and AKIA prefix are documented here"):
        bad.append("FALSE POSITIVE on prose naming key prefixes")
    if check(None):
        bad.append("standalone mode must not scan")
    return bad


if __name__ == "__main__":
    print("selftest:", selftest())
