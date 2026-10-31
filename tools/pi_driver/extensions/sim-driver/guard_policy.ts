// Pure policy for the guard (design section 4). No Pi imports, so `node --test guard_policy.test.ts` checks it alone.
// The container is the real boundary; this enforces the repo's own rules on top of it, for bash as well as
// write/edit (Pi's example protected-paths extension checks write/edit only, and bash can write any file).

export interface Verdict {
  block: boolean;
  reason?: string;
}

const OK: Verdict = { block: false };
const no = (reason: string): Verdict => ({ block: true, reason });

// [pattern, why] -- checked against the whole bash command, so a pipe or `&&` chain cannot hide one
const BASH_RULES: [RegExp, string][] = [
  [/--no-verify\b|\bcommit\b[^|;&]*\s-n\b/, "never skip the pre-commit gates (CLAUDE.md non-negotiable 7)"],
  [/\bcore\.hooksPath\b|\.git\/hooks\b|tools\/githooks\b/, "the git hooks run the gates; they are not the driver's to change"],
  [/\bgit\s+push\b/, "push only through `bash tools/push_both.sh <branch>` (it verifies both remotes)"],
  [/\bgit\s+reset\s+[^|;&]*--hard\b/, "never `git reset --hard` (it destroys uncommitted results)"],
  [/\bgit\s+clean\b/, "never `git clean` (untracked results are evidence)"],
  [/\bgit\s+(checkout|restore)\s+[^|;&]*(\s--\s+)?\.\s*($|[|;&])/, "never discard the whole working tree"],
  [/\bgit\s+(merge|rebase|cherry-pick)\b[^|;&]*\bmain\b|\bgit\s+(checkout|switch)\s+main\b/, "main is merged only through a Claude merge checkpoint"],
  [/\bgit\s+branch\s+[^|;&]*-[dD]\b|\bgit\s+push\b[^|;&]*--delete\b/, "do not delete branches"],
  [/(^|[\s;&|(])(sudo|su|ssh|scp|sftp|systemctl|crontab|docker|podman|kill(all)?|pkill|reboot|shutdown)\s/, "host and process control goes through the driver's tools, not raw commands"],
  [/\btools\/(aws_gpu|aws_pool_node|game|gpu_queue|pool_provision)\.sh\b|\btools\/local_llm\/llm\.sh\b/, "compute control goes through the run_gpu / run_pool tools"],
  [/\bpool_queue\.sh\s+withdraw\b/, "withdrawing queued pool work is a Claude/owner decision"],
  [/\brm\s+(-[a-zA-Z]*\s+)*[^|;&]*research\/findings\/raw\b/, "results under research/findings/raw are evidence: never delete them"],
  [/\brm\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+(\/|~|\$HOME)(\s|$)/, "recursive delete of / or home"],
  [/\bsecrets?\b[^|;&]*\b(cat|less|head|tail|cp|base64)\b|\b(cat|less|head|tail|cp|base64)\b[^|;&]*(\/run\/secrets|openrouter\.key|\.credentials)/, "never read or copy secrets"],
];

// write/edit targets (paths relative to the workspace or absolute)
const PATH_RULES: [RegExp, string][] = [
  [/(^|\/)\.git(\/|$)/, ".git is not edited by hand"],
  [/(^|\/)tools\/githooks\//, "the git hooks run the gates"],
  [/(^|\/)tools\/gates\//, "the gates are authoritative: a gate change is a Claude change (CLAUDE.md non-negotiable 7)"],
  [/(^|\/)docs\/FAILURE_GATE_MATRIX\.md$/, "the gate matrix is a Claude change"],
  [/(^|\/)tools\/pi_driver\//, "the driver does not edit its own harness"],
  [/(^|\/)\.claude\//, "Claude's skills and settings are not the driver's to change"],
  [/(^|\/)(CLAUDE|AGENTS)\.md$/, "project instructions are the owner's and Claude's"],
  [/^\/(etc|usr|var|boot|root|run)\//, "outside the workspace"],
];

export function checkBash(command: string): Verdict {
  for (const [re, why] of BASH_RULES) if (re.test(command)) return no(`blocked: ${why}`);
  return OK;
}

// existing files under research/findings/raw are evidence: a NEW results file is fine, an overwrite is not
export function checkPath(p: string, exists: boolean): Verdict {
  for (const [re, why] of PATH_RULES) if (re.test(p)) return no(`blocked: ${why}`);
  if (exists && /(^|\/)research\/findings\/raw\//.test(p)) return no("blocked: an existing results file is evidence; write a new file instead");
  return OK;
}
