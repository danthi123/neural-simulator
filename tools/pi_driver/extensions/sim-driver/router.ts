// Router (design section 1). The virtual model "sim/auto" sends every request to W's local model (free); when W is
// down (yielded to a GPU job, gaming, rebooting) it PARKS -- the supervisor retries later -- unless configured to let
// MiMo stand in. A second opinion is a tool, not a routing mode: second_opinion reads the files ITSELF and hands
// their text to MiMo in one call. (Smoke test 2026-10-31: routed as a "review" step class, MiMo ran `wc -l` on the
// file it was asked to review and then critiqued code that does not exist; the local model passed it on unchecked.)
// Judgment (verify-go, rulings, merges) goes to Claude through the checkpoint tool.
import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type DriverConfig, bumpLedger, clip, readLedger, text } from "./common.ts";

let wHealthy = { ok: false, at: 0 };
const MAX_REVIEW_CHARS = 120_000;   // ~30k tokens of quoted files per second opinion

// llama-swap's own /health (not a model load): is the workstation's server up at all? Cached for 60 s.
async function wUp(cfg: DriverConfig): Promise<boolean> {
  if (Date.now() - wHealthy.at < 60_000) return wHealthy.ok;
  let ok = false;
  try {
    const r = await fetch(cfg.w_base_url.replace(/\/v1\/?$/, "") + "/health", { signal: AbortSignal.timeout(5000) });
    ok = r.ok;
  } catch {
    ok = false;
  }
  wHealthy = { ok, at: Date.now() };
  return ok;
}

export function routerState() {
  return { w_up: wHealthy.ok, w_checked: wHealthy.at ? new Date(wHealthy.at).toISOString() : null };
}

export default function registerRouter(pi: ExtensionAPI, cfg: DriverConfig) {
  // what MiMo cost when it stood in for W (models.json prices); second_opinion books its own cost
  pi.on("message_end", async (event) => {
    const m = event.message as { role?: string; provider?: string; usage?: { cost?: { total?: number } } };
    if (m.role === "assistant" && m.provider === "or" && m.usage?.cost?.total) bumpLedger(cfg, { mimo_usd: m.usage.cost.total });
  });

  pi.registerVirtualModel({
    provider: "sim",
    id: "auto",
    name: "sim driver router",
    thinkingLevels: ["medium", "high"],
    contextWindow: 131072,
    async route(request, ctx) {
      const level = request.thinkingLevel === "high" ? "high" : "medium";
      if (request.reason === "retry" && request.failed?.model.provider === "w") wHealthy = { ok: false, at: 0 };
      if (await wUp(cfg)) {
        const m = ctx.modelRegistry.find("w", cfg.w_model);
        if (!m) throw new Error(`model w/${cfg.w_model} is not in models.json`);
        return { model: m, thinkingLevel: level };
      }
      if (cfg.mimo_execute_fallback && readLedger(cfg).mimo_usd < cfg.mimo_daily_usd) {
        const m = ctx.modelRegistry.find("or", cfg.mimo_model);
        if (m) return { model: m, thinkingLevel: level };
      }
      throw new Error("PARKED: W is down (yielded to a GPU job, gaming, or rebooting); waiting for W");
    },
  });

  pi.registerTool({
    name: "second_opinion",
    label: "Independent second opinion (MiMo)",
    description:
      "Ask a different model family (MiMo, metered: $" + cfg.mimo_daily_usd + "/day) for an independent review of a plan, " +
      "diff or result BEFORE you ask Claude. Name the files: this tool reads them itself and gives the reviewer their full " +
      "text, so the review is grounded in what the files actually say. Put a diff into a file first (git diff > x.diff).",
    parameters: Type.Object({
      question: Type.String({ description: "what to review and what would count as a defect" }),
      files: Type.Array(Type.String(), { description: "workspace paths to quote in full (code, findings, raw results, diffs)" }),
    }),
    async execute(_id, p, signal, _onUpdate, ctx) {
      const spent = readLedger(cfg).mimo_usd;
      if (spent >= cfg.mimo_daily_usd) throw new Error(`MiMo's daily budget is spent ($${spent.toFixed(2)} of $${cfg.mimo_daily_usd}); do the review yourself or ask again tomorrow.`);
      const model = ctx.modelRegistry.find("or", cfg.mimo_model);
      if (!model) throw new Error(`model or/${cfg.mimo_model} is not in models.json`);
      let quoted = "";
      for (const f of p.files) {
        const abs = path.resolve(cfg.workspace, f);
        if (path.relative(cfg.workspace, abs).startsWith("..") && path.relative(cfg.results_root, abs).startsWith(".."))
          throw new Error(`${f} is outside the workspace and the results root`);
        const body = fs.readFileSync(abs, "utf-8");
        const numbered = body.split("\n").map((l, i) => `${i + 1}\t${l}`).join("\n");
        quoted += `\n===== FILE ${f} (${body.split("\n").length} lines) =====\n${numbered}\n`;
        if (quoted.length > MAX_REVIEW_CHARS) throw new Error(`the files exceed ${MAX_REVIEW_CHARS} characters: name fewer or smaller files`);
      }
      const reply = await ctx.modelRegistry.complete(model, {
        systemPrompt:
          "You are an independent reviewer for a neuroscience simulation project. Ground every claim in the quoted files, " +
          "citing FILE:line. If the quoted text does not settle a point, say so instead of guessing. Report real defects " +
          "first, most serious first; say plainly when you find none.",
        messages: [{ role: "user", content: `${p.question}\n${quoted}`, timestamp: Date.now() }],
      }, { signal });
      const cost = reply.usage?.cost?.total ?? 0;
      bumpLedger(cfg, { mimo_usd: cost });
      const answer = reply.content.filter((c) => c.type === "text").map((c) => (c as { text: string }).text).join("\n");
      if (reply.stopReason === "error") throw new Error(`MiMo failed: ${reply.errorMessage}`);
      return { ...text(`Second opinion (MiMo, $${cost.toFixed(4)}). Check each cited line before acting on it:\n\n${clip(answer, 12000)}`), usage: reply.usage };
    },
  });
}
