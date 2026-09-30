"""Network-namespace simulation of srv-host-0 for tests/test_agent_container_firewall.py. NOT a test module.

Run as `unshare --user --map-root-user --net python3 tests/_agent_fw_netns.py <sim-agent.nft>` (no real root:
the user namespace grants CAP_NET_ADMIN over its OWN network namespaces only, so nothing on the machine running
the test is touched). It builds a miniature of the server and prints one JSON object of probe results.

Topology (every name is a network namespace; the outer one plays the SERVER):
  server  br-simagent 172.30.30.1/24 (+ veth to `agent` 172.30.30.2)   -- the sim-agent bridge
          l0 192.168.0.10/24 (veth to `lan`), default via 192.168.0.1  -- the LAN interface
          p0 172.20.0.1/24 (veth to `proxy` 172.20.0.5)                -- another stack's bridge (gitea, socket proxy)
          listeners 22 111 445 2049                                    -- sshd, rpcbind, smbd, nfsd
          a stand-in for fw.nft (input policy drop, `iifname "br-*" accept`) and for Docker's nat table
          (DNAT this host's :2222 -> gitea:22 and :18443 -> caddy:443, masquerade the container subnet)
  lan     192.168.0.1 (router + resolver), .68 (workstation), .50 (a LAN device); "internet" addresses
          1.1.1.1, 8.8.8.8 and 203.0.113.7 (stand-in for this site's own WAN address); TCP listeners on
          22 53 80 443 445 8080 and a UDP echo on 53 at .1, 1.1.1.1 and 8.8.8.8
  proxy   172.20.0.5 with listeners 22 443 2375 3000
  agent   172.30.30.2 (the container) with a listener on 7000

Probes run from `agent` (and one from `lan` INTO the agent), twice: with only the fw.nft/Docker stand-ins loaded
(the control arm -- proves each "must be blocked" probe really is reachable without sim-agent.nft), then with
sim-agent.nft loaded, then again with 203.0.113.7 added to its self_wan set.
"""
import json
import os
import subprocess
import sys
import time

NFT_FILE = sys.argv[1]

FW_STANDIN = """
table inet fw {
    chain input {
        type filter hook input priority 0; policy drop;
        ct state established,related accept
        ct state invalid drop
        iif "lo" accept
        ip protocol icmp accept
        iifname { "docker0", "lan-shim" } accept
        iifname "br-*" accept
        ip saddr 192.168.0.0/24 accept
    }
}
table ip dockernat {
    chain prerouting {
        type nat hook prerouting priority dstnat; policy accept;
        fib daddr type local tcp dport 2222 dnat to 172.20.0.5:22
        fib daddr type local tcp dport 18443 dnat to 172.20.0.5:443
    }
    chain postrouting {
        type nat hook postrouting priority srcnat; policy accept;
        ip saddr 172.30.30.0/24 oifname != "br-simagent" masquerade
    }
}
"""

LISTENER = r"""
import socket, sys, threading, time
tcp_ports = [int(p) for p in sys.argv[1].split(",") if p]
udp_addrs = [a for a in sys.argv[2].split(",") if a]
def serve(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port)); s.listen(64)
    while True:
        c, _ = s.accept(); c.close()
def echo(addr):
    # one socket PER address, so a reply leaves from the address that was asked (not the interface's primary)
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.bind((addr, 53))
    while True:
        d, a = u.recvfrom(512); u.sendto(d, a)
for p in tcp_ports:
    threading.Thread(target=serve, args=(p,), daemon=True).start()
for a in udp_addrs:
    threading.Thread(target=echo, args=(a,), daemon=True).start()
print("ready", flush=True)
time.sleep(600)
"""

PROBE = r"""
import json, socket, sys, threading
targets = json.loads(sys.argv[1])
out = {}
def tcp(key, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.8)
    try:
        s.connect((host, port)); out[key] = "open"
    except socket.timeout:
        out[key] = "timeout"
    except ConnectionRefusedError:
        out[key] = "refused"
    except OSError as e:
        out[key] = "error:%s" % e.errno
    finally:
        s.close()
def udp(key, host, port):
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.settimeout(0.8)
    try:
        u.sendto(b"q", (host, port)); u.recvfrom(64); out[key] = "open"
    except socket.timeout:
        out[key] = "timeout"
    except OSError as e:
        out[key] = "error:%s" % e.errno
    finally:
        u.close()
ts = []
for key, (proto, host, port) in targets.items():
    ts.append(threading.Thread(target=tcp if proto == "tcp" else udp, args=(key, host, port)))
for t in ts: t.start()
for t in ts: t.join()
print(json.dumps(out))
"""

AGENT_PROBES = {
    # must stay reachable
    "dns_router_tcp": ("tcp", "192.168.0.1", 53),
    "dns_router_udp": ("udp", "192.168.0.1", 53),
    "dns_fallback_tcp": ("tcp", "1.1.1.1", 53),
    "dns_fallback_udp": ("udp", "1.1.1.1", 53),
    "workstation_ssh": ("tcp", "192.168.0.68", 22),
    "workstation_model": ("tcp", "192.168.0.68", 8080),
    "gitea_ssh_via_host_2222": ("tcp", "192.168.0.10", 2222),
    "internet_https": ("tcp", "8.8.8.8", 443),
    # must be blocked
    "host_ssh_via_gateway": ("tcp", "172.30.30.1", 22),
    "host_smb_via_gateway": ("tcp", "172.30.30.1", 445),
    "host_ssh_lan_addr": ("tcp", "192.168.0.10", 22),
    "host_smb_lan_addr": ("tcp", "192.168.0.10", 445),
    "host_nfs_lan_addr": ("tcp", "192.168.0.10", 2049),
    "host_rpcbind_lan_addr": ("tcp", "192.168.0.10", 111),
    "caddy_published_18443": ("tcp", "192.168.0.10", 18443),
    "socket_proxy_direct": ("tcp", "172.20.0.5", 2375),
    "gitea_http_direct": ("tcp", "172.20.0.5", 3000),
    "gitea_ssh_direct_not_2222": ("tcp", "172.20.0.5", 22),
    "workstation_smb": ("tcp", "192.168.0.68", 445),
    "lan_device_http": ("tcp", "192.168.0.50", 80),
    "router_admin_https": ("tcp", "192.168.0.1", 443),
    "other_resolver_tcp": ("tcp", "8.8.8.8", 53),
    "other_resolver_udp": ("udp", "8.8.8.8", 53),
    "internet_plain_http": ("tcp", "8.8.8.8", 80),
    "self_wan_https": ("tcp", "203.0.113.7", 443),
}
INBOUND_PROBES = {"lan_to_agent": ("tcp", "172.30.30.2", 7000)}


def sh(*cmd, input_text=None):
    subprocess.run(list(cmd), check=True, input=input_text, text=True, capture_output=True)


def spawn_ns():
    # wait until the child really sits in a NEW netns -- entering it any earlier would configure OURS instead
    p = subprocess.Popen(["unshare", "--net", "sleep", "600"])
    own = os.readlink("/proc/self/ns/net")
    for _ in range(200):
        try:
            if os.readlink("/proc/%d/ns/net" % p.pid) != own:
                return p
        except OSError:
            pass
        time.sleep(0.01)
    raise RuntimeError("child network namespace never appeared")


def in_ns(pid, *cmd):
    return ["nsenter", "-t", str(pid), "-n", *cmd]


def listen(pid, tcp_ports, udp_echo_addrs=()):
    cmd = [sys.executable, "-c", LISTENER, ",".join(str(p) for p in tcp_ports), ",".join(udp_echo_addrs)]
    if pid is not None:
        cmd = in_ns(pid, *cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    return p


def probe(pid, targets):
    res = subprocess.run(in_ns(pid, sys.executable, "-c", PROBE, json.dumps(targets)),
                         capture_output=True, text=True, check=True, timeout=30)
    return json.loads(res.stdout)


def main():
    procs = []
    try:
        agent, lan, proxy = spawn_ns(), spawn_ns(), spawn_ns()
        procs += [agent, lan, proxy]
        A, L, P = agent.pid, lan.pid, proxy.pid
        sh("ip", "link", "set", "lo", "up")
        for pid in (A, L, P):
            sh(*in_ns(pid, "ip", "link", "set", "lo", "up"))
        # agent <-> br-simagent
        sh("ip", "link", "add", "br-simagent", "type", "bridge")
        sh("ip", "addr", "add", "172.30.30.1/24", "dev", "br-simagent")
        sh("ip", "link", "add", "a0", "type", "veth", "peer", "name", "a1")
        sh("ip", "link", "set", "a1", "netns", str(A))
        sh("ip", "link", "set", "a0", "master", "br-simagent")
        sh("ip", "link", "set", "a0", "up")
        sh("ip", "link", "set", "br-simagent", "up")
        sh(*in_ns(A, "ip", "addr", "add", "172.30.30.2/24", "dev", "a1"))
        sh(*in_ns(A, "ip", "link", "set", "a1", "up"))
        sh(*in_ns(A, "ip", "route", "add", "default", "via", "172.30.30.1"))
        # server <-> lan
        sh("ip", "link", "add", "l0", "type", "veth", "peer", "name", "l1")
        sh("ip", "link", "set", "l1", "netns", str(L))
        sh("ip", "addr", "add", "192.168.0.10/24", "dev", "l0")
        sh("ip", "link", "set", "l0", "up")
        for a in ("192.168.0.1/24", "192.168.0.68/24", "192.168.0.50/24"):
            sh(*in_ns(L, "ip", "addr", "add", a, "dev", "l1"))
        sh(*in_ns(L, "ip", "link", "set", "l1", "up"))
        for a in ("1.1.1.1/32", "8.8.8.8/32", "203.0.113.7/32"):
            sh(*in_ns(L, "ip", "addr", "add", a, "dev", "lo"))
        sh(*in_ns(L, "ip", "route", "add", "172.30.30.0/24", "via", "192.168.0.10"))
        sh("ip", "route", "add", "default", "via", "192.168.0.1")
        # server <-> proxy (another stack's bridge)
        sh("ip", "link", "add", "p0", "type", "veth", "peer", "name", "p1")
        sh("ip", "link", "set", "p1", "netns", str(P))
        sh("ip", "addr", "add", "172.20.0.1/24", "dev", "p0")
        sh("ip", "link", "set", "p0", "up")
        sh(*in_ns(P, "ip", "addr", "add", "172.20.0.5/24", "dev", "p1"))
        sh(*in_ns(P, "ip", "link", "set", "p1", "up"))
        sh(*in_ns(P, "ip", "route", "add", "default", "via", "172.20.0.1"))
        with open("/proc/sys/net/ipv4/ip_forward", "w") as f:
            f.write("1")
        # listeners
        procs.append(listen(None, [22, 111, 445, 2049]))
        procs.append(listen(L, [22, 53, 80, 443, 445, 8080], udp_echo_addrs=("192.168.0.1", "1.1.1.1", "8.8.8.8")))
        procs.append(listen(P, [22, 443, 2375, 3000]))
        procs.append(listen(A, [7000]))
        # control arm: only the fw.nft + Docker stand-ins
        sh("nft", "-f", "-", input_text=FW_STANDIN)
        control = probe(A, AGENT_PROBES)
        control.update(probe(L, INBOUND_PROBES))
        # sim-agent.nft loaded (twice: the file must be idempotent)
        sh("nft", "-f", NFT_FILE)
        sh("nft", "-f", NFT_FILE)
        loaded = probe(A, AGENT_PROBES)
        loaded.update(probe(L, INBOUND_PROBES))
        # self_wan filled (what sim-agent-fw-selfwan.sh does at runtime)
        sh("nft", "add", "element", "inet", "sim_agent", "self_wan", "{ 203.0.113.7 }")
        selfwan = probe(A, AGENT_PROBES)
        counters = subprocess.run(["nft", "list", "table", "inet", "sim_agent"], capture_output=True,
                                  text=True).stdout
        print(json.dumps({"control": control, "loaded": loaded, "selfwan": selfwan, "ruleset": counters}))
    finally:
        for p in procs:
            p.kill()


if __name__ == "__main__":
    main()
