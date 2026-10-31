// Shared config + small JSON state files for the sim-driver extension (tools/pi_driver/README.md).
import * as fs from "node:fs";
import * as path from "node:path";
import { execFile } from "node:child_process";

export interface DriverConfig {
  workspace: string;
  merge_repo: string;
  state_dir: string;
  status_dir: string;
  lane_file: string;
  w_base_url: string;
  w_model: string;
  mimo_model: string;
  mimo_daily_usd: number;
  mimo_execute_fallback: boolean;
  claude_daily_calls: number;
  claude_bin: string;
  claude_timeout_min: number;
  wait_poll_s: number;
  results_root: string;
}

export function loadConfig(): DriverConfig {
  const p = process.env.PI_DRIVER_CONFIG;
  if (!p) throw new Error("PI_DRIVER_CONFIG is not set (path to the driver's config.json)");
  return JSON.parse(fs.readFileSync(p, "utf-8")) as DriverConfig;
}

export function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export function readJson<T>(file: string, fallback: T): T {
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
  } catch {
    return fallback;
  }
}

// write-then-rename, so a reader (the supervisor, the status page) never sees half a file
export function writeJson(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp${process.pid}`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 1));
  fs.renameSync(tmp, file);
}

// Daily counters (MiMo dollars, Claude calls), reset when the date changes.
export interface Ledger {
  date: string;
  mimo_usd: number;
  claude_calls: number;
}

export function ledgerFile(cfg: DriverConfig): string {
  return path.join(cfg.state_dir, "ledger.json");
}

export function readLedger(cfg: DriverConfig): Ledger {
  const l = readJson<Ledger>(ledgerFile(cfg), { date: today(), mimo_usd: 0, claude_calls: 0 });
  return l.date === today() ? l : { date: today(), mimo_usd: 0, claude_calls: 0 };
}

export function bumpLedger(cfg: DriverConfig, change: Partial<Pick<Ledger, "mimo_usd" | "claude_calls">>): Ledger {
  const l = readLedger(cfg);
  l.mimo_usd += change.mimo_usd ?? 0;
  l.claude_calls += change.claude_calls ?? 0;
  writeJson(ledgerFile(cfg), l);
  return l;
}

export interface RunResult {
  code: number;
  out: string;
}

// Run a command without a shell; output is capped so a chatty script cannot flood the model's context.
export function run(cmd: string, args: string[], opts: { cwd: string; timeoutMs: number; env?: NodeJS.ProcessEnv }): Promise<RunResult> {
  return new Promise((resolve) => {
    execFile(cmd, args, { cwd: opts.cwd, timeout: opts.timeoutMs, maxBuffer: 16 << 20, env: opts.env ?? process.env },
      (err, stdout, stderr) => {
        const code = err ? (typeof (err as { code?: unknown }).code === "number" ? (err as { code: number }).code : 1) : 0;
        resolve({ code, out: clip(`${stdout}${stderr ? `\n[stderr]\n${stderr}` : ""}`) });
      });
  });
}

export function clip(s: string, max = 12000): string {
  return s.length <= max ? s : `${s.slice(0, max / 2)}\n... [${s.length - max} chars cut] ...\n${s.slice(-max / 2)}`;
}

export function text(t: string) {
  return { content: [{ type: "text" as const, text: t }], details: undefined };
}
