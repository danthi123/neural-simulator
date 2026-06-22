// Render the hand-authored brain-architecture SVGs to PNG.
//
// The .svg files in this folder are the source of truth (hand-authored, plain
// SVG); these PNGs are convenience rasters of them. GitHub renders the SVG
// directly. Any SVG->PNG renderer works — this uses @resvg/resvg-js because it
// ships its own binaries (no native cairo DLL needed, which is flaky on
// Windows).
//
// Usage:
//   cd docs/diagrams
//   npm install @resvg/resvg-js     # one-time
//   node render.js                  # re-render all three at their natural width
//   node render.js brain_master 2400   # one file at a custom width
//
const fs = require('fs');
const path = require('path');
const { Resvg } = require('@resvg/resvg-js');

// natural render widths (match each SVG's viewBox aspect for crisp text)
const DIAGRAMS = {
  brain_master: 1680,
  brain_navigation: 1520,
  brain_conversational: 1560,
};

function render(name, width) {
  const svgPath = path.join(__dirname, `${name}.svg`);
  const pngPath = path.join(__dirname, `${name}.png`);
  const svg = fs.readFileSync(svgPath, 'utf8');
  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: width },
    font: { loadSystemFonts: true, defaultFontFamily: 'Segoe UI' },
  });
  fs.writeFileSync(pngPath, resvg.render().asPng());
  console.log(`rendered ${name}.png  (width ${width})`);
}

const [, , one, w] = process.argv;
if (one) {
  render(one, w ? parseInt(w, 10) : DIAGRAMS[one] || 1600);
} else {
  for (const [name, width] of Object.entries(DIAGRAMS)) render(name, width);
}
