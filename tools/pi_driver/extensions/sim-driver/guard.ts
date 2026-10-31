// Guard: applies guard_policy.ts to every bash, write and edit call. A handler error blocks the call (Pi fails
// closed on tool_call errors), so an unexpected input shape is refused rather than waved through.
import * as fs from "node:fs";
import * as path from "node:path";
import { type ExtensionAPI, isToolCallEventType } from "@earendil-works/pi-coding-agent";
import type { DriverConfig } from "./common.ts";
import { checkBash, checkPath } from "./guard_policy.ts";

export default function registerGuard(pi: ExtensionAPI, cfg: DriverConfig) {
  pi.on("tool_call", async (event) => {
    if (isToolCallEventType("bash", event)) {
      const v = checkBash(event.input.command);
      return v.block ? { block: true, reason: v.reason } : undefined;
    }
    if (isToolCallEventType("write", event) || isToolCallEventType("edit", event)) {
      const p = event.input.path;
      const abs = path.resolve(cfg.workspace, p);
      const rel = path.relative(cfg.workspace, abs);
      if (rel.startsWith("..")) return { block: true, reason: `blocked: ${p} is outside the workspace` };
      const v = checkPath(rel, fs.existsSync(abs));
      return v.block ? { block: true, reason: v.reason } : undefined;
    }
    return undefined;
  });
}
