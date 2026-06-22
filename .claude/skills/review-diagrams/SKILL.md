---
name: review-diagrams
description: Use when editing, updating, or adding the hand-authored brain-architecture SVG diagrams in docs/diagrams/ (brain_master, brain_navigation, brain_conversational). Enforces the tile-by-tile up-close review + fix loop that keeps them visually clean — catching text overflow, label overlap/rotation, arrowhead direction + centering, and line crossings — plus the rendering gotchas. Invoke after ANY edit to a docs/diagrams/*.svg, before committing.
---

# Review Diagrams

The three brain diagrams in `docs/diagrams/` are **hand-authored plain SVG** (the `.svg` is the source of truth; the `.png` is a convenience raster produced by `render.js`). They are dense (every region + pathway), so visual quality regresses easily and silently. This skill is the review-and-fix loop that keeps them clean.

**Announce at start:** "Using review-diagrams to check the diagram up close before committing."

## The non-negotiable rule: review TILE-BY-TILE, never a single capture

A single downscaled screenshot of a ~1500px-wide dense diagram HIDES the real issues — overflowing text, an off-center arrowhead, a line clipping a box. Every problem the owner caught (across 5 rounds) was invisible at full-diagram zoom. So you MUST:

1. Render the SVG large (4500px wide).
2. Crop it into a grid of overlapping tiles (~1500px each).
3. **Read EACH tile** and inspect it like a person would: is the text inside its box? do arrows point cleanly and sit centered? do lines avoid boxes/text/each other?

## Workflow

### 1. Render + tile-crop (the CWD drifts — `cd` first)

```bash
cd /e/Documents/Projects/sim/docs/diagrams
node -e "const {Resvg}=require('@resvg/resvg-js');const fs=require('fs');const svg=fs.readFileSync('brain_master.svg','utf8');const r=new Resvg(svg,{fitTo:{mode:'width',value:4500},font:{loadSystemFonts:true,defaultFontFamily:'Segoe UI'}});fs.writeFileSync('_rev.png',r.render().asPng());"
python -c "
from PIL import Image
im=Image.open('_rev.png'); W,H=im.size
cols,rows=3,3; ox,oy=int(W*.04),int(H*.04)
for r in range(rows):
  for c in range(cols):
    im.crop((max(0,c*W//cols-ox),max(0,r*H//rows-oy),min(W,(c+1)*W//cols+ox),min(H,(r+1)*H//rows+oy))).save('_t%d.png'%(r*cols+c))
print('tiled 3x3')
"
```
`Read` each `_t*.png`. For a suspected spot, crop tighter: pixel = viewBox-coord × (4500 / viewBox_width). (viewBox widths: master 1680, nav 1520, conv 1560.)

### 2. Catalog every issue per tile

- **Text overflow** — text wider than its `<rect>` box. ("text spans wider than the box.")
- **Labels** — rotated/diagonal text; a label cramped in a narrow box-gap; a label sitting on a line or box.
- **Arrow direction** — must approach its target box edge perpendicular (horizontal into a left/right edge, vertical into a top/bottom edge), endpoint ON the edge.
- **Arrow centering** — the line must bisect the triangle's flat base; the triangle is symmetric about the line.
- **Line crossings** — a line over a box, over text, or over another line. Lines travelling the same direction should run **parallel in lanes**, not cross.

### 3. Map the boxes (to find the clear corridors)

```bash
grep -nE '<rect x="[0-9]+" y="[0-9]+" width="[0-9]+" height="[0-9]+"' docs/diagrams/brain_master.svg
```
Each box spans `x..x+width`, `y..y+height`. The clear corridors are the gutters between boxes/panels — route lines and place labels there.

### 4. Fix rules

- **Text overflow:** shorten the text or reduce its `font-size` to fit the box width. Don't widen the box (it cascades).
- **Labels:** never rotated — keep horizontal. Move out of a cramped box-gap into a gutter. Never on top of a line or box.
- **Arrow direction:** set the curve's LAST control point on the perpendicular through the endpoint (same Y as the endpoint for a left/right edge → horizontal tangent; same X for a top/bottom edge → vertical tangent), and snap the endpoint onto the box edge. Where several arrows hit one edge, space their endpoints so they don't overlap.
- **Arrow centering:** markers must use `orient="auto"`, NOT `orient="auto-start-reverse"` — resvg mis-rotates/offsets the latter. (These markers are only ever `marker-end`, so `auto` is equivalent + correct.) The triangle is symmetric with `refY` = the base's vertical centre.
- **Line routing (the hard one):** route each line through a clear gutter, not across a box or text. Lines that travel alongside one another should share a **lane** — parallel, evenly offset — rather than cross. Start a line at a box EDGE (never its interior) and approach the target edge perpendicular. If a line currently runs through a box, re-anchor its start/end to the box edges and reroute the middle through the nearest gutter.

### 5. Re-render + RE-REVIEW the fixed tiles up close

Render again, re-crop the spots you changed, `Read` them, confirm the fix actually landed. Iterate — don't assume.

### 6. Commit

```bash
cd /e/Documents/Projects/sim/docs/diagrams && node render.js <name>   # regenerate the committed PNG at natural width
git -C /e/Documents/Projects/sim add docs/diagrams/<name>.svg docs/diagrams/<name>.png   # both files together
git -C /e/Documents/Projects/sim commit -m "..." && git -C /e/Documents/Projects/sim push origin main && git -C /e/Documents/Projects/sim push gitea main
rm -f _rev.png _t*.png   # clean review temps (never commit the _*.png review crops)
```

## Gotchas (learned the hard way)

- **resvg `orient="auto-start-reverse"` bug** — renders arrowheads rotated wrong + off-centre from the line. Use `orient="auto"`. (Diagnosed with an isolated test SVG with red axis guides comparing marker variants.)
- **CWD drifts** off `docs/diagrams/` between tool calls → `cd` explicitly before each `node`/`python`, or the render fails with "file not found".
- **`render.js`** renders each SVG at its natural width (the committed PNG). Render at 4500px ONLY for review crops, to a temp `_rev.png`.
- **GitHub image caching** — after a push, the README PNG can look stale in the browser; it's caching, not a missing commit. Hard-refresh; verify the committed PNG bytes if unsure.
- **Don't trust one capture** — the whole reason this skill exists.

## Why this skill exists

The brain diagrams went through 5 rounds of "still looks weird" because fixes were made from single downscaled captures that hid the real problems (text overflow, off-centre arrowheads from the resvg marker bug, lines crossing boxes and each other). The tile-by-tile loop plus the fix rules above are what actually find and fix them.
