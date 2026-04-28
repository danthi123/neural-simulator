// Neural Simulator — Research Dashboard frontend
// Phase 1 vanilla JS. No build step. ES modules in the browser.
//
// All dynamic content (filenames, markdown body, JSON values) is rendered
// via textContent or escapeHTML — never via raw template-literal innerHTML.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHTML(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v != null) {
      node.setAttribute(k, v);
    }
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else node.appendChild(c);
  }
  return node;
}

// ─────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────
function setupTabs() {
  $$("nav button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = btn.dataset.tab;
      $$("nav button").forEach((b) => b.classList.toggle("active", b === btn));
      $$("section.tab").forEach((s) =>
        s.classList.toggle("active", s.id === `tab-${t}`),
      );
      if (t === "findings" && !window._findingsLoaded) loadFindings();
      if (t === "info" && !window._infoLoaded) loadInfo();
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────
// Runs tab
// ─────────────────────────────────────────────────────────────────────────
async function loadRuns() {
  const list = $("#runs-list");
  list.replaceChildren(document.createTextNode("Loading…"));
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    $("#runs-count").textContent = `${data.count} runs`;
    if (!data.runs.length) {
      const p = el("p", { class: "muted", style: "padding:16px" },
        "No runs yet — launch one from the Launch tab.");
      list.replaceChildren(p);
      return;
    }
    data.runs.sort((a, b) => {
      if (a.sum_finalQ == null) return 1;
      if (b.sum_finalQ == null) return -1;
      return a.sum_finalQ - b.sum_finalQ;
    });
    list.replaceChildren();
    for (const r of data.runs) {
      const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
      const item = el("div", { class: "list-item", dataset: { name: r.name } }, [
        el("div", { class: "name" }, r.name),
        el("div", { class: "meta" }, [
          metric("sum", sumStr),
          metric("seed", r.seed ?? "—"),
          metric("phases", r.n_phases),
        ]),
      ]);
      item.addEventListener("click", () => loadRunDetail(r.name, item));
      list.appendChild(item);
    }
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function metric(label, value) {
  return el("span", { class: "metric" }, [
    el("span", { class: "label" }, label),
    el("span", { class: "value" }, String(value)),
  ]);
}

async function loadRunDetail(name, listItem) {
  const detail = $("#run-detail");
  detail.replaceChildren(el("p", { class: "muted" }, `Loading ${name}…`));
  $$("#runs-list .list-item").forEach((el) =>
    el.classList.toggle("active", el === listItem),
  );
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    detail.replaceChildren(
      el("h2", {}, name),
      el("div", {}, [
        metric("seed", data.seed ?? "—"),
        metric("n_steps", data.n_steps ?? "—"),
        metric("grid_size", data.grid_size ?? 8),
      ]),
      el("h3", {}, "Phase stats"),
      renderPhaseStats(data.phase_stats || []),
      el("h3", {}, "Raw JSON"),
      el("pre", {}, JSON.stringify(summarizeRunData(data), null, 2)),
    );
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed to load: ${e.message}`));
  }
}

function summarizeRunData(data) {
  const heavy = new Set([
    "motor_counts", "distance_log", "trajectory", "spike_counts",
    "place_cell_log", "goal_cell_log", "raw_phase1_motor_counts",
  ]);
  const out = {};
  for (const [k, v] of Object.entries(data)) {
    if (heavy.has(k)) {
      out[k] = Array.isArray(v) ? `[…${v.length} entries…]` : "[…large…]";
    } else {
      out[k] = v;
    }
  }
  return out;
}

function renderPhaseStats(stats) {
  if (!stats.length) return el("p", { class: "muted" }, "No phase stats.");
  const head = el("tr", {}, ["Phase", "Goal", "Steps", "finalQ", "mean dist"]
    .map((h) => el("th", {}, h)));
  const rows = stats.map((ps, i) => el("tr", {}, [
    el("td", {}, String(i + 1)),
    el("td", {}, ps.goal ? `(${ps.goal[0]},${ps.goal[1]})` : "—"),
    el("td", {}, String(ps.n_steps ?? "—")),
    el("td", {}, el("strong", {}, ps.finalQ != null ? ps.finalQ.toFixed(2) : "—")),
    el("td", {}, ps.mean_distance != null ? ps.mean_distance.toFixed(2) : "—"),
  ]));
  return el("table", { class: "markdown" }, [
    el("thead", {}, head),
    el("tbody", {}, rows),
  ]);
}

// ─────────────────────────────────────────────────────────────────────────
// Findings tab
// ─────────────────────────────────────────────────────────────────────────
async function loadFindings() {
  window._findingsLoaded = true;
  const list = $("#findings-list");
  list.replaceChildren(document.createTextNode("Loading…"));
  try {
    const res = await fetch("/api/findings");
    const data = await res.json();
    $("#findings-count").textContent = `${data.count} findings`;
    list.replaceChildren();
    for (const f of data.findings) {
      const item = el("div", { class: "list-item" }, [
        el("div", { class: "name" }, f.name),
      ]);
      item.addEventListener("click", () => loadFindingDetail(f.name, item));
      list.appendChild(item);
    }
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

async function loadFindingDetail(name, listItem) {
  const detail = $("#finding-detail");
  detail.replaceChildren(el("p", { class: "muted" }, `Loading ${name}…`));
  $$("#findings-list .list-item").forEach((it) =>
    it.classList.toggle("active", it === listItem),
  );
  try {
    const res = await fetch(`/api/findings/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const text = await res.text();
    // renderMarkdown returns sanitized HTML (escapes input first, then injects
    // markdown-derived tags). Safe to set as innerHTML on a fresh div.
    const wrapper = el("div", { class: "markdown" });
    wrapper.innerHTML = renderMarkdown(text);
    detail.replaceChildren(wrapper);
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed to load: ${e.message}`));
  }
}

// Minimal markdown-ish renderer. Always escapes input first, so the only
// HTML in the output is what *this* function emits — no user-controlled
// tags. For full fidelity (footnotes, autolinks, etc.) we'd swap in
// `marked` + DOMPurify later; for Phase 1 this is enough.
function renderMarkdown(src) {
  src = escapeHTML(src);
  src = src.replace(/```(\w*)\n([\s\S]*?)```/g, (_, _lang, body) =>
    `<pre>${body}</pre>`);
  src = src.replace(/^###### (.*)$/gm, "<h6>$1</h6>");
  src = src.replace(/^##### (.*)$/gm, "<h5>$1</h5>");
  src = src.replace(/^#### (.*)$/gm, "<h4>$1</h4>");
  src = src.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  src = src.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  src = src.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  src = src.replace(/((?:^\|.*\|\n)+)/gm, (block) => {
    const rows = block.trim().split("\n");
    const isAlign = (r) => /^\|[\s:|-]+\|$/.test(r.trim());
    const cells = rows
      .filter((r) => !isAlign(r))
      .map((r) => r.trim().slice(1, -1).split("|").map((c) => c.trim()));
    if (!cells.length) return block;
    const [header, ...body] = cells;
    let html = "<table><thead><tr>" +
      header.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
    for (const row of body) {
      html += "<tr>" + row.map((c) => `<td>${c}</td>`).join("") + "</tr>";
    }
    html += "</tbody></table>";
    return html;
  });
  src = src.replace(/((?:^[-*] .*\n?)+)/gm, (block) => {
    const items = block.trim().split("\n").map((l) => l.replace(/^[-*] /, ""));
    return "<ul>" + items.map((i) => `<li>${i}</li>`).join("") + "</ul>";
  });
  src = src.replace(/^&gt; (.*)$/gm, "<blockquote>$1</blockquote>");
  src = src.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  src = src.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  src = src.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  // Links: [text](url) — but the captured text/url are HTML-escaped already,
  // so ampersands etc. are safe. We restrict the URL to http(s) and relative
  // paths to defang javascript: URLs.
  src = src.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => {
    const safe = /^(https?:|\/|\.\.\/|\.\/|#)/i.test(url) ? url : "#";
    return `<a href="${safe}" target="_blank" rel="noreferrer noopener">${text}</a>`;
  });
  src = src
    .split(/\n{2,}/)
    .map((p) => /^<(h\d|ul|ol|table|pre|blockquote)/.test(p.trim())
      ? p : `<p>${p}</p>`)
    .join("\n");
  return src;
}

// ─────────────────────────────────────────────────────────────────────────
// Launcher tab
// ─────────────────────────────────────────────────────────────────────────
function setupLauncher() {
  const form = $("#launch-form");
  const out = $("#launcher-output");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    out.replaceChildren();
    appendStatus(out, "Submitting…");

    const formData = new FormData(form);
    const extraStr = String(formData.get("extra_args") || "").trim();
    const body = {
      preset: String(formData.get("preset")),
      seed: parseInt(formData.get("seed"), 10),
      extra_args: extraStr ? extraStr.split(/\s+/) : [],
    };

    try {
      const res = await fetch("/api/runs/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const launch = await res.json();
      appendStatus(out, `Launched run_id=${launch.run_id}`);
      appendStatus(out, `cmd: ${launch.cmd.join(" ")}`);
      appendStatus(out, `out: ${launch.out_path}`);
      appendStatus(out, `streaming WebSocket at ${launch.ws_url}…`);
      tailWebSocket(launch.ws_url, out);
    } catch (e) {
      appendError(out, `Launch failed: ${e.message}`);
    }
  });
}

function tailWebSocket(path, out) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}${path}`);
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "stdout") {
      const div = el("div", { class: "stdout-line" }, msg.line);
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    } else if (msg.type === "done") {
      appendStatus(out, `Run finished. returncode=${msg.returncode}`);
      appendStatus(out, `Output saved to: ${msg.out_path}`);
      loadRuns();
    }
  };
  ws.onerror = () => appendError(out, "WebSocket error");
}

function appendStatus(out, text) {
  out.appendChild(el("div", { class: "system" }, `>>> ${text}`));
  out.scrollTop = out.scrollHeight;
}
function appendError(out, text) {
  out.appendChild(el("div", { class: "error" }, `!!! ${text}`));
  out.scrollTop = out.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────
// Info tab
// ─────────────────────────────────────────────────────────────────────────
async function loadInfo() {
  window._infoLoaded = true;
  try {
    const res = await fetch("/api/info");
    const data = await res.json();
    $("#info-output").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("#info-output").textContent = `Error: ${e.message}`;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────
setupTabs();
setupLauncher();
loadRuns();

$("#refresh-runs").addEventListener("click", loadRuns);
$("#refresh-findings").addEventListener("click", loadFindings);
