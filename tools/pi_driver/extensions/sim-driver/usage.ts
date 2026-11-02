// Usage-aware Claude budget (owner 2026-11-02: "if we built it out in a claude-usage-aware way, I'm happy to increase
// how much Claude we allow Pi to use"). Headless Claude Code reports the plan's live utilization in a
// `rate_limit_event` (stream-json): the five_hour and seven_day windows, 0..1, with reset times. Before each
// checkpoint or claude_build, Pi reads it with a tiny haiku probe (cached a few minutes) and calls Claude only while
// both windows leave the owner headroom. The weekly ceiling rises with the week, so Pi cannot spend the week on day
// one. The owner's own sessions draw on the same windows, so Pi backs off when the owner is busy. The daily counts in
// config stay as a backstop against a runaway loop.
import * as path from "node:path";
import { type DriverConfig, readJson, run, writeJson } from "./common.ts";

export interface Usage {
  at: number; // ms, when read
  five_hour: number; // 0..1
  seven_day: number; // 0..1
  five_reset: number; // unix seconds
  seven_reset: number; // unix seconds
  status: string; // "allowed", "allowed_warning" or "rejected"
}

export interface Limits {
  max_five_hour: number; // stop at this 5-hour utilization
  max_seven_day: number; // weekly ceiling at the END of the week
  week_pace_start: number; // weekly ceiling at the START of the week; rises linearly to max_seven_day
}

export function limits(cfg: DriverConfig): Limits {
  return {
    max_five_hour: cfg.claude_max_five_hour ?? 0.75,
    max_seven_day: cfg.claude_max_seven_day ?? 0.85,
    week_pace_start: cfg.claude_week_pace_start ?? 0.3,
  };
}

// The last rate_limit_event with both windows in a stream-json transcript, or null.
export function parseRateLimit(out: string, now = Date.now()): Usage | null {
  let u: Usage | null = null;
  for (const line of out.split("\n")) {
    if (!line.includes('"rate_limit_event"')) continue;
    try {
      const info = JSON.parse(line).rate_limit_info;
      const w = info?.unifiedWindows;
      if (!w?.five_hour || !w?.seven_day) continue;
      const c: Usage = {
        at: now, five_hour: Number(w.five_hour.utilization), seven_day: Number(w.seven_day.utilization),
        five_reset: Number(w.five_hour.resetsAt), seven_reset: Number(w.seven_day.resetsAt), status: String(info.status),
      };
      if ([c.five_hour, c.seven_day, c.seven_reset].every(Number.isFinite)) u = c;
    } catch {
      // a truncated or foreign line: skip it
    }
  }
  return u;
}

// Weekly ceiling now: week_pace_start at the start of the 7-day window, max_seven_day at its reset.
export function weekCeiling(u: Usage, lim: Limits, now = Date.now()): number {
  const left = Math.min(Math.max((u.seven_reset * 1000 - now) / (7 * 86400_000), 0), 1);
  return lim.week_pace_start + (lim.max_seven_day - lim.week_pace_start) * (1 - left);
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const at = (s: number) => new Date(s * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC";

export function usageDecision(u: Usage | null, lim: Limits, now = Date.now()): { ok: boolean; reason: string } {
  if (!u) return { ok: false, reason: "the owner's Claude plan usage could not be read (probe failed), so Pi does not call Claude blind" };
  if (u.status === "rejected") return { ok: false, reason: `the Claude plan is at its limit; the 5-hour window resets ${at(u.five_reset)}` };
  if (u.five_hour >= lim.max_five_hour)
    return { ok: false, reason: `the 5-hour window is at ${pct(u.five_hour)} (Pi stops at ${pct(lim.max_five_hour)} to leave the owner room); it resets ${at(u.five_reset)}` };
  const ceil = weekCeiling(u, lim, now);
  if (u.seven_day >= ceil)
    return { ok: false, reason: `the weekly window is at ${pct(u.seven_day)}, above Pi's pace ceiling for this point in the week (${pct(ceil)}); it resets ${at(u.seven_reset)}` };
  return { ok: true, reason: `plan usage: 5-hour ${pct(u.five_hour)}, week ${pct(u.seven_day)} (Pi's ceilings now ${pct(lim.max_five_hour)} and ${pct(ceil)})` };
}

const usageFile = (cfg: DriverConfig) => path.join(cfg.state_dir, "claude_usage.json");

// Current plan usage: the cached reading if fresh, else a one-word haiku probe run outside any repo.
export async function currentUsage(cfg: DriverConfig, maxAgeMs = 10 * 60_000): Promise<Usage | null> {
  const cached = readJson<Usage | null>(usageFile(cfg), null);
  if (cached && Date.now() - cached.at < maxAgeMs) return cached;
  const r = await run(cfg.claude_bin, ["-p", "Reply with the single word OK", "--output-format", "stream-json", "--verbose",
    "--model", "haiku", "--max-turns", "1"], { cwd: cfg.state_dir, timeoutMs: 120_000, maxOut: 4 << 20 });
  const u = parseRateLimit(r.out);
  if (u) writeJson(usageFile(cfg), u);
  return u;
}

// After a real Claude call, force the next gate to re-read usage.
export function markUsageStale(cfg: DriverConfig): void {
  const cached = readJson<Usage | null>(usageFile(cfg), null);
  if (cached) writeJson(usageFile(cfg), { ...cached, at: 0 });
}

// The gate both Claude tools call first. Throws with a plain reason when Pi must not call Claude now.
export async function claudeGate(cfg: DriverConfig): Promise<string> {
  const d = usageDecision(await currentUsage(cfg), limits(cfg));
  if (!d.ok) throw new Error(`Not calling Claude now: ${d.reason}. Park this item, continue other lane work, and retry later.`);
  return d.reason;
}
