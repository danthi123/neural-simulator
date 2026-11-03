// Compute: experiments get the machine, and the model waits without spending tokens (design section 2).
// Wraps what exists (tools/pool_queue.sh, tools/gpu_remote.sh incl. `yield`); re-implements no queue.
// wait_for registers a waiter and the model ends its turn; a poller here checks the waiters every wait_poll_s and
// injects a message (which starts a new turn) when one finishes or times out. Waiters persist in state_dir, so a
// restarted driver keeps waiting. A waiter ends four ways: its target is met, its deadline passes (timeout_hours,
// capped at 8), the model drops it with cancel_wait, or the poller finds a gpu target that no longer exists (stale).
import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type DriverConfig, readJson, run, text, writeJson } from "./common.ts";
import { checkBash } from "./guard_policy.ts";
import { GPU_TAG, capTimeoutHours, isStale, matchWaiters, nextStrikes, parseHolds } from "./waiter_policy.ts";

interface Waiter {
  id: string;
  label: string;
  kind: "file" | "gpu" | "shell";
  target: string;
  since: string;
  deadline: string;
  result_file?: string; // gpu only: a results path (as for kind 'file'); present = the job produced its output
}

// Closed-waiter log, merged into agent_status.json by index.ts (the owner's view): why each wait ended.
interface WaitEvent {
  at: string;
  id: string;
  label: string;
  event: "cancelled" | "stale" | "finished" | "timeout";
  reason: string;
}

const waitersFile = (cfg: DriverConfig) => path.join(cfg.state_dir, "waiters.json");
export const readWaiters = (cfg: DriverConfig) => readJson<Waiter[]>(waitersFile(cfg), []);
const eventsFile = (cfg: DriverConfig) => path.join(cfg.state_dir, "wait_events.json");
export const readWaitEvents = (cfg: DriverConfig) => readJson<WaitEvent[]>(eventsFile(cfg), []);
const logWaitEvent = (cfg: DriverConfig, w: Waiter, event: WaitEvent["event"], reason: string) =>
  writeJson(eventsFile(cfg), [...readWaitEvents(cfg), { at: new Date().toISOString(), id: w.id, label: w.label, event, reason }].slice(-30));

export default function registerCompute(pi: ExtensionAPI, cfg: DriverConfig) {
  const sh = (cmd: string, timeoutMs = 300_000) => run("bash", ["-c", cmd], { cwd: cfg.workspace, timeoutMs });
  let timer: NodeJS.Timeout | undefined;
  const strikes = new Map<string, number>(); // gpu waiter id -> consecutive polls with its job gone

  const fileReady = (target: string) => {
    const p = path.isAbsolute(target) ? target : path.join(cfg.results_root, target);
    try {
      return fs.statSync(p).size > 0;
    } catch {
      return false;
    }
  };

  async function done(w: Waiter): Promise<boolean> {
    if (w.kind === "file") return fileReady(w.target);
    if (w.kind === "gpu") {
      const r = await sh("bash tools/gpu_remote.sh events 400", 60_000);
      return r.code === 0 && r.out.split("\n").some((l) => l.includes("DONE") && l.includes(w.target));
    }
    return (await sh(w.target, 60_000)).code === 0;
  }

  // gpu only: has the target job vanished (killed, withdrawn, never queued) with no result file? Two polls in a row.
  async function staleReason(w: Waiter): Promise<string | undefined> {
    if (w.kind !== "gpu" || !GPU_TAG.test(w.target)) return undefined; // old or odd targets just wait to their deadline
    const r = await sh(`bash tools/gpu_remote.sh holds ${w.target}`, 60_000);
    const holds = parseHolds(r.code, r.out);
    const n = nextStrikes(strikes.get(w.id) ?? 0, holds, w.result_file ? fileReady(w.result_file) : false);
    strikes.set(w.id, n);
    return isStale(n) ? `no job containing "${w.target}" is queued or running in the workstation GPU queue, and ${w.result_file ? `${w.result_file} is absent` : "no result file was named"}` : undefined;
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
      if (!finished && w.result_file && fileReady(w.result_file)) finished = true;
      let stale: string | undefined;
      if (!finished) {
        try {
          stale = await staleReason(w);
        } catch {
          stale = undefined;
        }
      }
      const timedOut = !finished && !stale && Date.now() > Date.parse(w.deadline);
      if (!finished && !stale && !timedOut) continue;
      closed.add(w.id);
      strikes.delete(w.id);
      logWaitEvent(cfg, w, finished ? "finished" : stale ? "stale" : "timeout", stale ?? (timedOut ? `deadline ${w.deadline}` : "target met"));
      pi.sendMessage(
        {
          customType: "sim-driver.wait",
          content: finished
            ? `Wait finished: "${w.label}" (${w.kind}: ${w.target}). Harvest the result and continue.`
            : stale
              ? `Wait CLOSED as stale: "${w.label}" (${w.kind}: ${w.target}). ${stale}. The job was probably killed or withdrawn. Check gpu_remote.sh status/log, then re-queue it with run_gpu or move on; it will not be waited on any longer.`
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
      "Queue a GPU job at this checkout's COMMITTED HEAD (bash tools/gpu_remote.sh add, which carries the commit to the workstation itself; commit first, no push needed). Long jobs " +
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
      "containing target appears in the workstation GPU queue log; if no job containing target is queued or running any " +
      "more (and result_file, if given, is absent) the wait is closed as stale. kind 'shell': target is a command that exits 0 " +
      "when done. timeout_hours defaults to 8 and is capped at 8; to re-wait longer, wait again after the timeout. " +
      "Drop a wait you no longer need with cancel_wait.",
    parameters: Type.Object({
      label: Type.String(),
      kind: Type.Union([Type.Literal("file"), Type.Literal("gpu"), Type.Literal("shell")]),
      target: Type.String(),
      timeout_hours: Type.Optional(Type.Number({ description: "give up and report after this long (default and maximum 8)" })),
      result_file: Type.Optional(Type.String({ description: "kind 'gpu' only: the job's results path (as for kind 'file'); finishing it also ends the wait" })),
    }),
    async execute(_id, p) {
      if (p.kind === "gpu" && !GPU_TAG.test(p.target))
        throw new Error("kind 'gpu' target must be a plain tag (letters, digits and . _ / = : + , @ % -), e.g. a unique --out name");
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
        deadline: new Date(Date.now() + capTimeoutHours(p.timeout_hours) * 3_600_000).toISOString(),
        ...(p.kind === "gpu" && p.result_file ? { result_file: p.result_file } : {}),
      };
      writeJson(waitersFile(cfg), [...readWaiters(cfg), w]);
      return text(`Waiting (${w.id}, deadline ${w.deadline}). End your turn now; you will be messaged when it finishes.`);
    },
  });

  pi.registerTool({
    name: "cancel_wait",
    label: "Drop a wait you no longer need",
    description:
      "Remove open waits so the driver stops waiting on them (use it when you killed the job, or the result will never come). " +
      "match is a wait id, or a substring of a wait's label or target (case-insensitive); every wait it matches is dropped. " +
      "reason is logged to the driver's status file. Does not stop any job. Use it, then continue with other work.",
    parameters: Type.Object({
      match: Type.String({ description: "wait id, or label/target substring" }),
      reason: Type.String({ description: "why the wait is dropped" }),
    }),
    async execute(_id, p) {
      if (!p.reason.trim()) throw new Error("cancel_wait needs a reason");
      const hit = matchWaiters(readWaiters(cfg), p.match);
      if (!hit.length) throw new Error(`no open wait matches "${p.match}". Open waits: ${readWaiters(cfg).map((w) => `${w.id} "${w.label}"`).join("; ") || "none"}`);
      const ids = new Set(hit.map((w) => w.id));
      // re-read before writing, so a wait_for registered meanwhile is kept
      writeJson(waitersFile(cfg), readWaiters(cfg).filter((w) => !ids.has(w.id)));
      for (const w of hit) {
        strikes.delete(w.id);
        logWaitEvent(cfg, w, "cancelled", p.reason);
      }
      return text(`Cancelled ${hit.length} wait(s): ${hit.map((w) => `${w.id} "${w.label}"`).join("; ")}.`);
    },
  });
}
