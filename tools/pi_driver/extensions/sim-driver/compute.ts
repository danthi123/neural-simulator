// Compute: experiments get the machine, and the model waits without spending tokens (design section 2).
// Wraps what exists (tools/pool_queue.sh, tools/gpu_remote.sh incl. `yield`); re-implements no queue.
// wait_for registers a waiter and the model ends its turn; a poller here checks the waiters every wait_poll_s and
// injects a message (which starts a new turn) when one finishes or times out. Waiters persist in state_dir, so a
// restarted driver keeps waiting.
import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type DriverConfig, readJson, run, text, writeJson } from "./common.ts";
import { checkBash } from "./guard_policy.ts";

interface Waiter {
  id: string;
  label: string;
  kind: "file" | "gpu" | "shell";
  target: string;
  since: string;
  deadline: string;
}

const waitersFile = (cfg: DriverConfig) => path.join(cfg.state_dir, "waiters.json");
export const readWaiters = (cfg: DriverConfig) => readJson<Waiter[]>(waitersFile(cfg), []);

export default function registerCompute(pi: ExtensionAPI, cfg: DriverConfig) {
  const sh = (cmd: string, timeoutMs = 300_000) => run("bash", ["-c", cmd], { cwd: cfg.workspace, timeoutMs });
  let timer: NodeJS.Timeout | undefined;

  async function done(w: Waiter): Promise<boolean> {
    if (w.kind === "file") {
      const p = path.isAbsolute(w.target) ? w.target : path.join(cfg.results_root, w.target);
      try {
        return fs.statSync(p).size > 0;
      } catch {
        return false;
      }
    }
    if (w.kind === "gpu") {
      const r = await sh("bash tools/gpu_remote.sh events 400", 60_000);
      return r.code === 0 && r.out.split("\n").some((l) => l.includes("DONE") && l.includes(w.target));
    }
    return (await sh(w.target, 60_000)).code === 0;
  }

  async function poll() {
    const closed = new Set<string>();
    for (const w of readWaiters(cfg)) {
      let finished = false;
      try {
        finished = await done(w);
      } catch {
        finished = false;
      }
      if (!finished && Date.now() <= Date.parse(w.deadline)) continue;
      closed.add(w.id);
      pi.sendMessage(
        {
          customType: "sim-driver.wait",
          content: finished
            ? `Wait finished: "${w.label}" (${w.kind}: ${w.target}). Harvest the result and continue.`
            : `Wait TIMED OUT: "${w.label}" (${w.kind}: ${w.target}) passed its deadline ${w.deadline}. Find out why (gpu_remote.sh status/log, the pool dispatch log) before re-running anything.`,
          display: true,
          details: w,
        },
        { triggerTurn: true, deliverAs: "followUp" },
      );
    }
    // re-read before writing, so a wait_for registered during this poll is kept
    if (closed.size) writeJson(waitersFile(cfg), readWaiters(cfg).filter((w) => !closed.has(w.id)));
  }

  pi.on("session_start", async () => {
    if (timer) return;
    timer = setInterval(() => poll().catch(() => undefined), cfg.wait_poll_s * 1000);
  });
  pi.on("session_shutdown", async () => {
    if (timer) clearInterval(timer);
    timer = undefined;
  });

  pi.registerTool({
    name: "run_pool",
    label: "Queue CPU work on the mini-PC pool",
    description:
      "Append ONE line to the pool queue (bash tools/pool_queue.sh add). Use the line format the runbook " +
      "(research/coordination/LOCAL_LLM_RUNBOOK.md) and existing dispatch lines use: POOL_CHECKED_REASON=..., mem_gb=..., " +
      "a pinned revision, numpy backend, memcap for heavy jobs. pool_queue.sh validates the line and prints why it refuses. " +
      "Then call wait_for with kind 'file' on the result path (results reach " + cfg.results_root + " through the 15-minute sync).",
    parameters: Type.Object({ line: Type.String({ description: "the complete queue line" }) }),
    async execute(_id, p) {
      const v = checkBash(p.line);
      if (v.block) throw new Error(`queue line refused by the guard: ${v.reason}`);
      const r = await run("bash", ["tools/pool_queue.sh", "add", p.line], { cwd: cfg.workspace, timeoutMs: 120_000 });
      if (r.code !== 0) throw new Error(`pool_queue.sh refused (exit ${r.code}):\n${r.out}`);
      return text(r.out || "queued");
    },
  });

  pi.registerTool({
    name: "run_gpu",
    label: "Run a job on the workstation GPU",
    description:
      "Queue a GPU job at this checkout's COMMITTED HEAD (bash tools/gpu_remote.sh add; commit and push first). Long jobs " +
      "(over ~10 min) set yield_model=true: the workstation's model -- the one you are running on -- unloads so the job gets " +
      "the whole card, and reloads by itself when the queue drains. Put a unique tag in the command (e.g. in its --out " +
      "path), then call wait_for with kind 'gpu' and that tag, and END YOUR TURN: you will be messaged when it is done.",
    parameters: Type.Object({
      command: Type.String({ description: "the job, run from the repo root at HEAD" }),
      yield_model: Type.Boolean({ description: "unload the local model for this job (long jobs)" }),
      max_hours: Type.Optional(Type.Number({ description: "yield window, 1-12 (default 10)" })),
    }),
    async execute(_id, p) {
      const v = checkBash(p.command);
      if (v.block) throw new Error(`GPU command refused by the guard: ${v.reason}`);
      const add = await run("bash", ["tools/gpu_remote.sh", "add", p.command], { cwd: cfg.workspace, timeoutMs: 600_000 });
      if (add.code !== 0) throw new Error(`gpu_remote.sh add failed (exit ${add.code}):\n${add.out}`);
      let out = add.out;
      if (p.yield_model) {
        const hours = String(Math.min(12, Math.max(1, Math.round(p.max_hours ?? 10))));
        const y = await run("bash", ["tools/gpu_remote.sh", "yield", "--max-hours", hours], { cwd: cfg.workspace, timeoutMs: 120_000 });
        out += `\n[yield] ${y.out}`;
      }
      return text(out);
    },
  });

  pi.registerTool({
    name: "wait_for",
    label: "Wait for a job without spending tokens",
    description:
      "Register a wait, then END YOUR TURN. You get a message when it finishes or passes its deadline. kind 'file': a " +
      "results path (relative to " + cfg.results_root + ", or absolute) exists and is non-empty. kind 'gpu': a DONE line " +
      "containing target appears in the workstation GPU queue log. kind 'shell': target is a command that exits 0 when done.",
    parameters: Type.Object({
      label: Type.String(),
      kind: Type.Union([Type.Literal("file"), Type.Literal("gpu"), Type.Literal("shell")]),
      target: Type.String(),
      timeout_hours: Type.Number({ description: "give up and report after this long" }),
    }),
    async execute(_id, p) {
      if (p.kind === "shell") {
        const v = checkBash(p.target);
        if (v.block) throw new Error(`wait command refused by the guard: ${v.reason}`);
      }
      const w: Waiter = {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: p.label,
        kind: p.kind,
        target: p.target,
        since: new Date().toISOString(),
        deadline: new Date(Date.now() + p.timeout_hours * 3_600_000).toISOString(),
      };
      writeJson(waitersFile(cfg), [...readWaiters(cfg), w]);
      return text(`Waiting (${w.id}, deadline ${w.deadline}). End your turn now; you will be messaged when it finishes.`);
    },
  });
}
