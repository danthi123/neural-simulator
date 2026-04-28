// Minimal canvas-based charts. No external deps.
// Designed for the research dashboard's specific needs:
// - Line chart with phase backgrounds + axes + cursor
// - Bar chart for motor-count histograms
// - Mini chart for live distance updates
//
// Public API:
//   makeLineChart(canvas, opts) → updateData(...)
//   makeBarChart(canvas, opts) → updateData(...)
//
// Conventions: colors use the dashboard CSS vars where possible (read via
// getComputedStyle), but functions accept overrides.

const PALETTE = {
  fg: "#e3e6ea",
  fgDim: "#9aa3ad",
  border: "#2a2f3d",
  bg: "#0a0c10",
  bg2: "#161922",
  accent: "#6ee7b7",
  warn: "#fbbf24",
  bad: "#f87171",
  blue: "#93c5fd",
  purple: "#c4b5fd",
};

/* ─── helpers ─────────────────────────────────────────────────────── */

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  const cssH = canvas.clientHeight || 200;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH };
}

function fmt(value, digits = 2) {
  if (value == null || !isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  return value.toFixed(digits);
}

function niceMax(v) {
  // Round up to a nice number for axis ticks.
  if (v <= 1) return 1;
  if (v <= 5) return Math.ceil(v);
  if (v <= 10) return Math.ceil(v);
  return Math.ceil(v / 5) * 5;
}

/* ─── line chart ──────────────────────────────────────────────────── */

/**
 * Create a line chart with optional phase backgrounds.
 *
 * opts:
 *   title:           string label rendered top-left
 *   yLabel:          y axis label
 *   yMin/yMax:       fixed y axis range (optional; auto from data)
 *   phaseRanges:     [{start, end, label, color}] — phase shaded backgrounds
 *   markers:         [{x, label, color}] — vertical event markers
 *
 * Returns:
 *   { updateData(series), redraw() }
 *
 *   series is an array of { values: number[], color: string, label: string }
 *   The x-axis is index-based (0..N-1).
 */
export function makeLineChart(canvas, opts = {}) {
  const ctx = canvas.getContext("2d");
  const state = {
    series: [],
    phaseRanges: opts.phaseRanges || [],
    markers: opts.markers || [],
    yMin: opts.yMin,
    yMax: opts.yMax,
    title: opts.title,
    yLabel: opts.yLabel,
    cursor: null,
  };

  function redraw() {
    const { ctx: c, w, h } = setupCanvas(canvas);
    c.clearRect(0, 0, w, h);
    c.fillStyle = PALETTE.bg;
    c.fillRect(0, 0, w, h);

    const ML = 44, MR = 12, MT = 24, MB = 26;
    const plotW = w - ML - MR;
    const plotH = h - MT - MB;
    if (plotW < 10 || plotH < 10) return;

    const allVals = state.series.flatMap((s) => s.values).filter((v) => v != null && isFinite(v));
    if (allVals.length === 0) {
      c.fillStyle = PALETTE.fgDim;
      c.font = "12px sans-serif";
      c.textAlign = "center";
      c.fillText("(no data)", w / 2, h / 2);
      return;
    }
    const maxLen = Math.max(...state.series.map((s) => s.values.length));
    const yMin = state.yMin != null ? state.yMin : Math.min(0, Math.min(...allVals));
    const yMax = state.yMax != null ? state.yMax : niceMax(Math.max(...allVals));
    const xToPx = (x) => ML + (x / Math.max(1, maxLen - 1)) * plotW;
    const yToPx = (y) => MT + plotH - ((y - yMin) / Math.max(1e-6, yMax - yMin)) * plotH;

    // Phase background shading
    for (const p of state.phaseRanges) {
      c.fillStyle = p.color || PALETTE.bg2;
      c.fillRect(xToPx(p.start), MT, xToPx(p.end) - xToPx(p.start), plotH);
    }

    // Axes
    c.strokeStyle = PALETTE.border;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(ML, MT);
    c.lineTo(ML, MT + plotH);
    c.lineTo(ML + plotW, MT + plotH);
    c.stroke();

    // Y-axis ticks
    const yTicks = 4;
    c.fillStyle = PALETTE.fgDim;
    c.font = "10px ui-monospace, Consolas, monospace";
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (let i = 0; i <= yTicks; i++) {
      const y = yMin + (yMax - yMin) * (i / yTicks);
      const py = yToPx(y);
      c.fillText(fmt(y, 1), ML - 6, py);
      c.strokeStyle = i === 0 ? PALETTE.border : "rgba(42,47,61,0.4)";
      c.beginPath();
      c.moveTo(ML, py);
      c.lineTo(ML + plotW, py);
      c.stroke();
    }
    // X-axis ticks (3 ticks at 0, mid, max)
    c.textAlign = "center";
    c.textBaseline = "top";
    for (const i of [0, 0.5, 1]) {
      const x = (maxLen - 1) * i;
      const px = xToPx(x);
      c.fillText(String(Math.round(x)), px, MT + plotH + 4);
    }

    // Phase labels (at top)
    c.textAlign = "center";
    c.textBaseline = "top";
    c.font = "10px sans-serif";
    for (const p of state.phaseRanges) {
      if (!p.label) continue;
      c.fillStyle = p.labelColor || PALETTE.fgDim;
      const cx = (xToPx(p.start) + xToPx(p.end)) / 2;
      c.fillText(p.label, cx, MT + 2);
    }

    // Vertical event markers
    for (const m of state.markers) {
      const px = xToPx(m.x);
      c.strokeStyle = m.color || PALETTE.warn;
      c.lineWidth = 1.5;
      c.setLineDash([4, 3]);
      c.beginPath();
      c.moveTo(px, MT);
      c.lineTo(px, MT + plotH);
      c.stroke();
      c.setLineDash([]);
      if (m.label) {
        c.fillStyle = m.color || PALETTE.warn;
        c.font = "9px sans-serif";
        c.textAlign = "left";
        c.fillText(m.label, px + 4, MT + 12);
      }
    }

    // Data lines
    for (const s of state.series) {
      c.strokeStyle = s.color || PALETTE.accent;
      c.lineWidth = 1.5;
      c.beginPath();
      let started = false;
      for (let i = 0; i < s.values.length; i++) {
        const v = s.values[i];
        if (v == null || !isFinite(v)) {
          started = false;
          continue;
        }
        const px = xToPx(i);
        const py = yToPx(v);
        if (!started) {
          c.moveTo(px, py);
          started = true;
        } else {
          c.lineTo(px, py);
        }
      }
      c.stroke();

      // Per-series point markers (e.g. goal-change events on the live
      // recent_dist line). Drawn on top of the line so they're visible.
      if (Array.isArray(s.pointIndices) && s.pointIndices.length > 0) {
        c.fillStyle = s.pointColor || PALETTE.warn;
        c.strokeStyle = PALETTE.bg;
        c.lineWidth = 1.5;
        for (const i of s.pointIndices) {
          const v = s.values[i];
          if (v == null || !isFinite(v)) continue;
          c.beginPath();
          c.arc(xToPx(i), yToPx(v), 4, 0, Math.PI * 2);
          c.fill();
          c.stroke();
        }
      }
    }

    // Title (top-left). If multi-series with legend, render at far-left only;
    // legend goes top-right in its own zone (no overlap).
    if (state.title) {
      c.fillStyle = PALETTE.fg;
      c.font = "11px sans-serif";
      c.textAlign = "left";
      c.textBaseline = "top";
      // Truncate long titles to leave space for the legend.
      const hasLegend = state.series.some((s) => s.label);
      const maxTitleW = hasLegend ? plotW * 0.5 : plotW;
      let titleText = state.title;
      let mw = c.measureText(titleText).width;
      if (mw > maxTitleW) {
        while (titleText.length > 4 && c.measureText(titleText + "…").width > maxTitleW) {
          titleText = titleText.slice(0, -1);
        }
        titleText += "…";
      }
      c.fillText(titleText, ML, 4);
    }
    // Y label (rotated)
    if (state.yLabel) {
      c.save();
      c.translate(10, MT + plotH / 2);
      c.rotate(-Math.PI / 2);
      c.textAlign = "center";
      c.fillStyle = PALETTE.fgDim;
      c.font = "10px sans-serif";
      c.fillText(state.yLabel, 0, 0);
      c.restore();
    }

    // Series legend (top-right). Truncate long labels to keep within max half-plot width.
    if (state.series.length >= 1 && state.series.some((s) => s.label)) {
      c.textAlign = "right";
      c.textBaseline = "top";
      c.font = "10px sans-serif";
      const maxLabelW = plotW * 0.45;
      let lx = w - MR - 4;
      let ly = 4;
      for (const s of state.series) {
        if (!s.label) continue;
        let label = "● " + s.label;
        if (c.measureText(label).width > maxLabelW) {
          while (label.length > 4 && c.measureText(label + "…").width > maxLabelW) {
            label = label.slice(0, -1);
          }
          label += "…";
        }
        c.fillStyle = s.color || PALETTE.accent;
        c.fillText(label, lx, ly);
        ly += 12;
      }
    }

    // Cursor (vertical line at hover position) — set externally
    if (state.cursor != null) {
      const px = xToPx(state.cursor);
      c.strokeStyle = PALETTE.fgDim;
      c.lineWidth = 1;
      c.setLineDash([2, 2]);
      c.beginPath();
      c.moveTo(px, MT);
      c.lineTo(px, MT + plotH);
      c.stroke();
      c.setLineDash([]);
    }
  }

  function updateData(series) {
    state.series = series;
    redraw();
  }

  function setCursor(x) {
    state.cursor = x;
    redraw();
  }

  function setPhaseRanges(ranges) {
    state.phaseRanges = ranges;
    redraw();
  }

  function setMarkers(markers) {
    state.markers = markers;
    redraw();
  }

  // Resize on window changes
  window.addEventListener("resize", redraw);

  redraw();
  return { updateData, setCursor, setPhaseRanges, setMarkers, redraw };
}

/* ─── bar chart (motor counts) ────────────────────────────────────── */

/**
 * opts:
 *   title:    string
 *   labels:   array of bar labels
 *   colors:   array (one per bar) — defaults to PALETTE.accent
 *
 * updateData(values) — array of numbers, parallel to labels
 */
export function makeBarChart(canvas, opts = {}) {
  const state = {
    values: [],
    labels: opts.labels || [],
    colors: opts.colors || [],
    title: opts.title,
    yMax: opts.yMax,
  };

  function redraw() {
    const { ctx: c, w, h } = setupCanvas(canvas);
    c.clearRect(0, 0, w, h);
    c.fillStyle = PALETTE.bg;
    c.fillRect(0, 0, w, h);

    const ML = 36, MR = 12, MT = 24, MB = 26;
    const plotW = w - ML - MR;
    const plotH = h - MT - MB;
    if (plotW < 10 || plotH < 10 || state.values.length === 0) return;

    const yMax = state.yMax != null ? state.yMax : niceMax(Math.max(...state.values, 1));
    const n = state.values.length;
    const gap = 8;
    const barW = (plotW - gap * (n - 1)) / n;

    // Y-axis ticks
    c.strokeStyle = PALETTE.border;
    c.fillStyle = PALETTE.fgDim;
    c.font = "10px ui-monospace, Consolas, monospace";
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (const ratio of [0, 0.5, 1]) {
      const y = yMax * ratio;
      const py = MT + plotH - (y / yMax) * plotH;
      c.fillText(fmt(y, 0), ML - 6, py);
      c.strokeStyle = ratio === 0 ? PALETTE.border : "rgba(42,47,61,0.4)";
      c.beginPath();
      c.moveTo(ML, py);
      c.lineTo(ML + plotW, py);
      c.stroke();
    }

    // Bars
    for (let i = 0; i < n; i++) {
      const v = state.values[i];
      const px = ML + i * (barW + gap);
      const ph = (v / yMax) * plotH;
      c.fillStyle = state.colors[i] || PALETTE.accent;
      c.fillRect(px, MT + plotH - ph, barW, ph);
      // Label below
      c.fillStyle = PALETTE.fgDim;
      c.font = "10px ui-monospace, Consolas, monospace";
      c.textAlign = "center";
      c.textBaseline = "top";
      c.fillText(state.labels[i] ?? String(i), px + barW / 2, MT + plotH + 4);
      // Value above
      if (v > 0) {
        c.fillStyle = PALETTE.fg;
        c.textBaseline = "bottom";
        c.fillText(fmt(v, 0), px + barW / 2, MT + plotH - ph - 2);
      }
    }

    if (state.title) {
      c.fillStyle = PALETTE.fg;
      c.font = "11px sans-serif";
      c.textAlign = "left";
      c.textBaseline = "top";
      c.fillText(state.title, ML, 4);
    }
  }

  function updateData(values) {
    state.values = values;
    redraw();
  }

  window.addEventListener("resize", redraw);
  redraw();
  return { updateData, redraw };
}

export const PALETTE_EXPORT = PALETTE;
