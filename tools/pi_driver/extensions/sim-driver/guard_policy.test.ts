// node --test tools/pi_driver/extensions/sim-driver/guard_policy.test.ts   (Node >= 23 strips the types itself)
import { test } from "node:test";
import assert from "node:assert/strict";
import { checkBash, checkPath } from "./guard_policy.ts";

const blocked = (c: string) => assert.equal(checkBash(c).block, true, `should block: ${c}`);
const allowed = (c: string) => assert.equal(checkBash(c).block, false, `should allow: ${c}`);

test("repo rules on bash", () => {
  blocked("git commit --no-verify -m x");
  blocked("git commit -n -m x");
  blocked("git push gitea research/x");
  blocked("cd x && git push --force");
  blocked("git reset --hard HEAD~1");
  blocked("git clean -fdx");
  blocked("git checkout -- .");
  blocked("git merge main");
  blocked("git switch main");
  blocked("git config core.hooksPath /dev/null");
  blocked("sudo apt install x");
  blocked("ssh workstation ls");
  blocked("bash tools/aws_gpu.sh start");
  blocked("bash tools/local_llm/llm.sh off");
  blocked("bash tools/pool_queue.sh withdraw 'x' --expect 1 --reason y");
  blocked("rm -f research/findings/raw/_x/a.json");
  blocked("cat /run/secrets/openrouter.key");
  blocked("echo hi; kill 1234");
});

test("ordinary work passes", () => {
  allowed("bash tools/push_both.sh research/pi-x");
  allowed("git commit -m 'fix: x'");
  allowed("git checkout -b research/pi-x");
  allowed("git switch research/pi-x");
  allowed(".venv/bin/python -m pytest -q tests/test_x.py");
  allowed("bash tools/pool_queue.sh add 'cmd'");
  allowed("bash tools/gpu_remote.sh status");
  allowed("git log --oneline -5 main");
  allowed("git diff main...HEAD");
  allowed("grep -rn skill_name research/");
});

test("paths", () => {
  assert.equal(checkPath("tools/gates/x.py", false).block, true);
  assert.equal(checkPath(".git/config", true).block, true);
  assert.equal(checkPath("CLAUDE.md", true).block, true);
  assert.equal(checkPath("tools/pi_driver/router.ts", true).block, true);
  assert.equal(checkPath("research/findings/raw/_x/new.json", false).block, false);
  assert.equal(checkPath("research/findings/raw/_x/old.json", true).block, true);
  assert.equal(checkPath("sim/bridge.py", true).block, false);
  assert.equal(checkPath("research/runners/x.py", false).block, false);
});
