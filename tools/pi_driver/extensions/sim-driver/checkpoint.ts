// Checkpoint: judgment goes to Claude as a packet (design section 3). The agent writes the question, the evidence
// and the bar; this tool runs headless Claude Code (`claude -p`, logged into the owner's plan, no API key) in the
// workspace and writes Claude's verdict next to the packet. Capped per day (claude_daily_calls); at the cap it
// refuses, and the supervisor reports the open packet as a NEEDS item instead.
// A merge checkpoint yields a verdict on an exact sha; landing on main stays with the existing owner/Claude flow.
import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type DriverConfig, bumpLedger, clip, readLedger, run, text, today, writeJson } from "./common.ts";

const SKILL: Record<string, string> = {
  verify: "Run the verify-go skill (.claude/skills/verify-go/SKILL.md) on this claim: try to REFUTE it; default to REFUTED when uncertain.",
  ruling: "Make the ruling asked for, as the project's judgment step (CLAUDE.md, the MASTER ROADMAP, the runbook rulings).",
  merge_review:
    "Review the branch for merging to main: read its full diff against main, run the relevant tests, check every claim it makes " +
    "against the code (verify-go for any PASS/GO/default flip). Do NOT merge, push or commit anything yourself.",
  design: "Review this plan or design before the driver builds it: is it the right method (CLAUDE.md research-the-record rules), what will fail?",
};

export const openCheckpoints = (cfg: DriverConfig) => {
  const dir = path.join(cfg.workspace, "research/coordination/checkpoints");
  try {
    return fs.readdirSync(dir).filter((f) => f.endsWith(".md") && !f.endsWith(".VERDICT.md") && !fs.existsSync(path.join(dir, f.replace(/\.md$/, ".VERDICT.md"))));
  } catch {
    return [];
  }
};

export default function registerCheckpoint(pi: ExtensionAPI, cfg: DriverConfig) {
  pi.registerTool({
    name: "checkpoint",
    label: "Ask Claude for a judgment",
    description:
      "Send a judgment call to Claude and get its verdict. REQUIRED before: claiming any PASS/GO/surpass, flipping a default, " +
      "asking for a merge (kind merge_review, with the branch and its exact sha), adopting a new method after a wall, or " +
      "any ruling the runbook reserves for Claude. Claude is scarce (" + cfg.claude_daily_calls + " calls/day): bundle " +
      "questions, give exact evidence paths and the bar, and do your own checks first (tests, a MiMo review step).",
    parameters: Type.Object({
      kind: Type.Union([Type.Literal("verify"), Type.Literal("ruling"), Type.Literal("merge_review"), Type.Literal("design")]),
      title: Type.String({ description: "short, filename-safe" }),
      question: Type.String({ description: "what exactly Claude must decide" }),
      evidence: Type.Array(Type.String(), { description: "repo paths (findings, raw results, diffs, packets) Claude must read" }),
      bar: Type.String({ description: "the pass bar or decision criteria, quoted from the prereg/packet where one exists" }),
      branch: Type.Optional(Type.String()),
      sha: Type.Optional(Type.String()),
    }),
    async execute(_id, p, signal) {
      const led = readLedger(cfg);
      if (led.claude_calls >= cfg.claude_daily_calls)
        throw new Error(`Claude's daily checkpoint cap (${cfg.claude_daily_calls}) is used up. Park this item: continue other lane work and retry tomorrow.`);
      const slug = `${today()}-${p.kind}-${p.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 48)}`;
      const rel = `research/coordination/checkpoints/${slug}.md`;
      const packet = path.join(cfg.workspace, rel);
      fs.mkdirSync(path.dirname(packet), { recursive: true });
      fs.writeFileSync(packet, [
        `# Checkpoint: ${p.title}`, "", `kind: ${p.kind}`, p.branch ? `branch: ${p.branch}` : "", p.sha ? `sha: ${p.sha}` : "",
        "", "## Question", "", p.question, "", "## Bar", "", p.bar, "", "## Evidence", "", ...p.evidence.map((e) => `- ${e}`), "",
      ].filter((l, i, a) => l !== "" || a[i - 1] !== "").join("\n"));
      const prompt = [
        `You are the judgment checkpoint for the Pi sim driver (tools/pi_driver/README.md). Packet: ${rel}. Read it and every evidence path.`,
        SKILL[p.kind],
        "End with ONE final line, exactly: `VERDICT: APPROVE`, `VERDICT: REJECT`, or `VERDICT: NEEDS-OWNER`, after the reasons and any " +
          "conditions the driver must meet. Your whole answer is written to the verdict file and returned to the driver.",
      ].join("\n\n");
      bumpLedger(cfg, { claude_calls: 1 });
      const t0 = Date.now();
      const r = await run(cfg.claude_bin, ["-p", prompt, "--output-format", "json", "--max-turns", "60",
        "--allowedTools", "Read", "Grep", "Glob", "Bash"], { cwd: cfg.workspace, timeoutMs: cfg.claude_timeout_min * 60_000 });
      let verdict = r.out;
      try {
        verdict = String((JSON.parse(r.out.split("\n[stderr]")[0]) as { result?: string }).result ?? r.out);
      } catch {
        // not JSON: keep the raw output, it says what went wrong
      }
      if (signal?.aborted) throw new Error("checkpoint aborted");
      fs.writeFileSync(packet.replace(/\.md$/, ".VERDICT.md"), `# Verdict: ${p.title}\n\n${verdict}\n`);
      writeJson(path.join(cfg.state_dir, "last_checkpoint.json"), { packet: rel, code: r.code, seconds: Math.round((Date.now() - t0) / 1000) });
      if (r.code !== 0 && !/VERDICT:/.test(verdict)) throw new Error(`claude -p failed (exit ${r.code}); the call still counts against the cap:\n${clip(verdict, 4000)}`);
      return text(`Claude's verdict (also in ${rel.replace(/\.md$/, ".VERDICT.md")}; commit both files):\n\n${clip(verdict, 10000)}`);
    },
  });
}
