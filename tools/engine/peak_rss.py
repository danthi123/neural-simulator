"""GNU `time -v`-compatible peak-RSS launcher for nodes without /usr/bin/time (the workstation, w3090 kind). stdlib only.

    python -m tools.engine.peak_rss [-v] -o FILE -- CMD...

Runs CMD as a subprocess (stdout/stderr pass through) and returns its exit code. While it runs it polls /proc for the
process tree (VmHWM of every descendant) and, after the wait, takes resource.getrusage(RUSAGE_CHILDREN).ru_maxrss (the max
over all waited descendants); the larger is reported. Writes to FILE the lines the engine parses (core.timev):
    Maximum resident set size (kbytes): N
    Elapsed (wall clock) time (h:mm:ss or m:ss): H:MM:SS.ss
and, when a GPU is visible, `Peak GPU memory (MiB): N (source)`. The GPU peak is the job's own torch.cuda.max_memory_allocated
if it wrote bytes (or MiB) to $ENGINE_GPU_PEAK_FILE (= FILE.gpu), else the peak of nvidia-smi memory.used sampled while it
ran, minus the level before it started (the node is only used while the GPU is idle).
"""
import argparse
import os
import resource
import subprocess
import sys
import time

POLL_S = 0.5


def children_map():
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    return kids


def tree_pids(root):
    kids, out, todo = children_map(), [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def vm_hwm_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def gpu_used_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return max(int(x) for x in r.stdout.split()) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def fmt_wall(s):
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}" if h else f"{int(m)}:{sec:05.2f}"


def run(cmd, out):
    gpu_file = out + ".gpu"
    try:
        os.remove(gpu_file)
    except OSError:
        pass
    env = {**os.environ, "ENGINE_GPU_PEAK_FILE": gpu_file}
    base = gpu_used_mib()
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env)
    peak_kb, gpu_peak = 0, base
    while p.poll() is None:
        peak_kb = max(peak_kb, max((vm_hwm_kb(x) for x in tree_pids(p.pid)), default=0))
        g = gpu_used_mib() if base is not None else None
        if g is not None:
            gpu_peak = max(gpu_peak, g)
        try:
            p.wait(POLL_S)
        except subprocess.TimeoutExpired:
            pass
    wall = time.time() - t0
    peak_kb = max(peak_kb, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)   # kB on Linux
    lines = [f"\tMaximum resident set size (kbytes): {int(peak_kb)}",
             f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {fmt_wall(wall)}"]
    torch_mib = None
    try:
        v = float(open(gpu_file).read().split()[0])
        torch_mib = v / 1048576 if v > 1e6 else v        # bytes (torch.cuda.max_memory_allocated) or MiB
    except (OSError, ValueError, IndexError):
        pass
    if torch_mib is not None:
        lines.append(f"\tPeak GPU memory (MiB): {torch_mib:.0f} (torch max_memory_allocated)")
    elif base is not None:
        lines.append(f"\tPeak GPU memory (MiB): {max(gpu_peak - base, 0)} (nvidia-smi sampled, above the idle baseline)")
    lines.append(f"\tExit status: {p.returncode}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", action="store_true", help="accepted for `time -v` compatibility")
    ap.add_argument("-o", required=True, help="report file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    rc = run(cmd, a.o)
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
