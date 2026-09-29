"""tools/agent_container/egress.sh -- iptables rule ORDER (2026-09-29 re-review, MEDIUM-1). `-I CHAIN 1`
inserts at the TOP of the chain, so the rule EXECUTED LAST ends up FIRST -- `--dry-run`'s own printed order is
EXECUTION order, not the resulting chain order. This simulates the `-I CHAIN 1` insert semantics over the
dry-run output (each new rule PREPENDED to its chain's list, matching real iptables) and asserts the FINAL
(post-insert) chain order is correct: every RFC1918-443-DROP rule above the generic 443-ACCEPT, and the
catch-all DROP at the very bottom of DOCKER-USER.

Hermetic: `--dry-run` needs no root and touches no real iptables rule; SIM_AGENT_EGRESS_IPTABLES is not even
consulted in dry-run mode (the script never execs `$IPTABLES` at all when DRY_RUN=1).
"""
import os
import re
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EGRESS = os.path.join(_ROOT, "tools", "agent_container", "egress.sh")

_INSERT_RE = re.compile(r'^DRY-RUN: iptables -I (\S+) 1 (.*)$')


def _simulated_chains(dry_run_output):
    """Returns {chain: [rule_str, ...]} in FINAL (post -I-1-insert) top-to-bottom order."""
    chains = {}
    for line in dry_run_output.splitlines():
        m = _INSERT_RE.match(line)
        if not m:
            continue
        chain, rest = m.group(1), m.group(2)
        chains.setdefault(chain, []).insert(0, rest)   # -I CHAIN 1 == prepend
    return chains


def _run_dry_run(extra_env=None):
    env = dict(os.environ)
    env.pop("SIM_AGENT_EGRESS_IPTABLES", None)   # dry-run never execs it -- confirm that stays true
    if extra_env:
        env.update(extra_env)
    res = subprocess.run(["bash", _EGRESS, "--dry-run"], env=env, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def test_dry_run_needs_no_root_and_execs_nothing_real():
    out = _run_dry_run()
    assert "DRY RUN" in out
    assert out.count("DRY-RUN: iptables") > 5


def test_final_chain_order_puts_every_rfc1918_443_drop_above_the_generic_443_accept():
    out = _run_dry_run()
    chains = _simulated_chains(out)
    docker_user = chains["DOCKER-USER"]

    generic_443_accept_idx = None
    private_drop_idxs = []
    for i, rule in enumerate(docker_user):
        if "--dport 443" in rule and "-j ACCEPT" in rule and "-d " not in rule.split("--dport 443")[0].split("-s")[-1]:
            # the generic rule has NO -d (destination) restriction at all
            if " -d " not in rule:
                generic_443_accept_idx = i
        if "--dport 443" in rule and "-j DROP" in rule and " -d " in rule:
            private_drop_idxs.append(i)

    assert generic_443_accept_idx is not None, "generic 443 ACCEPT rule not found: %s" % docker_user
    assert len(private_drop_idxs) == 5, "expected 5 RFC1918+loopback+link-local 443 DROP rules: %s" % docker_user
    for idx in private_drop_idxs:
        assert idx < generic_443_accept_idx, (
            "a private-range 443 DROP rule (index %d) must come BEFORE (above) the generic 443 ACCEPT "
            "(index %d) in the final chain -- got order: %s" % (idx, generic_443_accept_idx, docker_user))


def test_catchall_drop_is_the_last_rule_in_docker_user():
    out = _run_dry_run()
    chains = _simulated_chains(out)
    docker_user = chains["DOCKER-USER"]
    last = docker_user[-1]
    assert "-j DROP" in last and "--dport" not in last and "-d " not in last, (
        "the catch-all DROP (no port/destination qualifier) must be the LAST rule in the chain: %s" % docker_user)


def test_input_chain_catchall_drop_is_last():
    out = _run_dry_run()
    chains = _simulated_chains(out)
    input_chain = chains.get("INPUT", [])
    assert input_chain, "INPUT chain must have at least the catch-all DROP"
    assert "-j DROP" in input_chain[-1]
