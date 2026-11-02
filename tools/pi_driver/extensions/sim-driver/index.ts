// sim-driver: the Pi extension for the project's driver (docs/plans/2026-10-30-pi-driver-harness-DESIGN.md,
// tools/pi_driver/README.md). Loaded by tools/pi_driver/supervisor.py with `--extension`.
import * as path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import registerCheckpoint, { openCheckpoints } from "./checkpoint.ts";
import { loadConfig, readLedger, writeJson } from "./common.ts";
import registerCompute, { readWaiters } from "./compute.ts";
import registerGuard from "./guard.ts";
import registerRouter, { routerState } from "./router.ts";

export default function (pi: ExtensionAPI) {
  const cfg = loadConfig();
  registerGuard(pi, cfg);
  registerRouter(pi, cfg);
  registerCompute(pi, cfg);
  registerCheckpoint(pi, cfg);

  // agent_status.json: what the agent is doing, for supervisor.py's status.json (the owner's view) and its
  // decision whether to feed the next lane item (never while a wait is open)
  let last: { at: string; event: string; detail?: string; model?: string } = { at: new Date().toISOString(), event: "start" };
  const write = () =>
    writeJson(path.join(cfg.state_dir, "agent_status.json"), {
      ...routerState(),
      last,
      ledger: readLedger(cfg),
      waiters: readWaiters(cfg),
      open_checkpoints: openCheckpoints(cfg),
    });
  const mark = (event: string, detail?: string) => {
    last = { ...last, at: new Date().toISOString(), event, detail };
    write();
  };
  pi.on("session_start", async () => mark("session_start"));
  pi.on("tool_execution_end", async (e) => mark("tool", (e as { toolName?: string }).toolName));
  pi.on("message_end", async (e) => {
    const m = e.message as { role?: string; provider?: string; model?: string; stopReason?: string; errorMessage?: string };
    if (m.role !== "assistant") return;
    last.model = `${m.provider}/${m.model}`;
    mark(m.stopReason === "error" ? "error" : "reply", m.errorMessage);
  });
  pi.on("agent_settled", async () => mark("settled"));
}
