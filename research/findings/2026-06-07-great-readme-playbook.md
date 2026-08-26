# Great README / Project-Documentation Playbook

**Date:** 2026-06-07
**Purpose:** Actionable playbook for rewriting the README of this project — a
GPU-accelerated spiking neural-network simulator with real-time 3D
visualization, biologically realistic neurons (Izhikevich / Hodgkin-Huxley /
AdEx), brain regions, plasticity, and neuromodulators, used for research
toward biology-faithful learning agents. Audience for the README is a MIX:
(a) software developers, (b) computational neuroscientists, (c) biologists,
(d) curious general readers.

**Method:** Web research on widely-praised READMEs/landing docs, both
general developer-tool exemplars and PEER scientific / neuroscience-simulation
projects. Sources cited inline. This is a documentation playbook, not a code
change — nothing in `sim/` is touched.

---

## Part 1 — Exemplars and what specifically makes each good

### A. Developer-tool exemplars (the "hook + show-don't-tell" masters)

**1. FastAPI** — <https://github.com/fastapi/fastapi>, <https://fastapi.tiangolo.com/>
The gold standard for the *first screen*. Its structure:
- Logo + a benefit-stacked tagline: *"FastAPI framework, high performance, easy
  to learn, fast to code, ready for production."*
- A single crisp one-liner: *"a modern, fast (high-performance), web framework
  for building APIs with Python based on standard Python type hints."*
- Badges inline (tests, coverage, PyPI version, Python versions) — credibility
  before prose.
- A **bulleted key-features list where every bullet pairs a benefit with a
  parenthetical justification** ("Fast: very high performance, on par with
  NodeJS and Go"; "Fewer bugs: reduce ~40% of human-induced errors"). This is
  the single most copyable pattern for this project.
- Then it *pivots immediately to a runnable example*: Create it → Run it →
  Check it → Interactive docs, with actual terminal output and a browser
  screenshot. "Working code within 60 seconds of scrolling." The effectiveness
  is **removing friction between curiosity and competence.**

**2. Typer** — <https://typer.tiangolo.com/>
Best demonstration of **progressive disclosure of an example**. It escalates:
(1) a plain Python function with *no library import at all*, (2) add
`typer.run()` + type hints, (3) a full multi-command app — and **shows the
real terminal output (help text, errors, success) at every level.** Lesson:
let the reader stop at their comfort level. Also models "credibility by
association" ("the FastAPI of CLIs") and benefit-tagged feature bullets.

**3. Rich** — <https://github.com/Textualize/rich>
The **show-don't-tell** champion. Terminal output is hard to describe in
words, so Rich embeds high-quality screenshots and GIFs of *actual rendered
output* (tables, progress bars, syntax highlighting), and **text always
follows the visual, never precedes it.** Quick-start uses the simplest
possible API (`rich.print()`) before any advanced `Console` setup.
Directly relevant: this project's killer feature is a real-time 3D
visualization — Rich proves a GIF of the live output belongs above the fold.

**4. esbuild** (cited by awesome-readme) — strong use of **graphics for
architecture/data-flow diagrams** and an explicit **list of project
principles**. Lesson: a "Why this exists / design principles" block earns
trust with technical readers and is short.

**5. gofiber/fiber & httpie/httpie** (awesome-readme picks) — clean logo,
useful badge/link row, quick-start, **benchmark charts**, documented code
examples, and a contributor list. httpie adds a demo screenshot + table of
contents + build badges. Lesson: benchmarks-as-charts and a ToC are
expected at this maturity tier.

**6. Day8/re-frame & choojs/choo** (awesome-readme picks) — re-frame is "a
giant, well-written essay about the tech, the philosophy behind it, and how
it fits into the greater ecosystem"; choo has "a beautiful little menu above
the fold with useful links + an FAQ." Lesson: for a *research* project, an
essayistic "what this is and why it's interesting" section is a feature, not
bloat — but it goes *below* the quick-start, and an above-the-fold link menu
helps the mixed audience self-route.

### B. Peer scientific / neuroscience-simulation exemplars (the audience-signaling + credibility masters)

**7. Nengo** — <https://www.nengo.ai/> ("Build brains")
Best at **serving a mixed audience in one page via deliberate layering**:
- Tagline *"Build brains"* (approachable to everyone) + one-liner *"a Python
  package for building, testing, and deploying neural networks."*
- **Five capability icons** (spiking models / scriptable+GUI / customizable /
  dynamic processing / hardware deployment) that let each reader find their
  hook: developers see "Python package, fully scriptable"; neuroscientists see
  "spiking or non-spiking models"; engineers see "exploit the latest hardware."
- An animated GUI screenshot + hardware imagery (chip, brain) to bridge
  simulation and reality.
- Scaffolded entry points (getting-started, tutorials, forum, full docs) so
  newcomers find their level without intimidating experts.
Lesson: **explicit per-audience signaling** is the move for this exact project.

**8. Brian2** — <https://briansimulator.org/>
Best at **scientific credibility + stating the differentiator**:
- One-liner: *"a free, open source simulator for spiking neural networks."*
- States its differentiator plainly ("just write down the equations in standard
  mathematical notation and run it") — **lead with what makes you different.**
- Credibility hooks aimed at the field: references the *Neuronal Dynamics*
  textbook, "hundreds of modelling studies," and the historical Hodgkin-Huxley
  (1952) lineage.
- A dedicated **"How to cite us"** section.
- Pitch is *user-centric* ("saves scientist time"), not just "fast."
Lesson: cite the field's touchstones; give a one-line "what makes us different";
include a How-to-cite.

**9. snnTorch** — <https://snntorch.readthedocs.io/>
Best at **pedagogical layering + biomimetic framing**:
- One-liner: *"A Python package for performing gradient-based learning with
  spiking neural networks."*
- Elevator framing that hooks the general/biology reader: *"The brain is the
  perfect place to look for inspiration to develop more efficient neural
  networks."*
- Above the fold: a **Colab badge** (zero-install try-it), an **animated GIF of
  spikes propagating**, and a ~28-line runnable snippet.
- A deliberate 7-tutorial learning arc (encoding → … → neuromorphic datasets).
Lesson: a "try it with zero install" path (Colab / one-command demo) + a spike
GIF + a short complete snippet is the proven SNN-landing recipe.

**10. BindsNET** — <https://github.com/BindsNET/bindsnet> (the closest peer:
GPU SNNs on PyTorch)
What it does well: one-liner *"simulating spiking neural networks (SNNs) on
CPUs or GPUs"*; multiple install paths (pip-from-GitHub, source, editable,
Docker); a one-command runnable example (`cd examples/mnist; python
eth_mnist.py --plot`); a **BibTeX citation**; and **reproducible benchmarks vs
BRIAN2 / PyNEST / ANNarchy across 250–10,000 neurons.**
What it *misses* (so this project should NOT miss): GPU memory requirements,
scalability limits beyond 10K neurons, performance-tuning guidance, API-
stability statement, and troubleshooting. Lesson: for a GPU simulator, **state
the hardware envelope and scale limits explicitly** — it's the #1 gap reviewers
flagged.

**11. Allen Institute SDK** — <https://github.com/AllenInstitute/AllenSDK>
Best at **honest status/scope without undercutting the project**:
- One-liner ties software to data: *"code for reading and processing Allen
  Institute for Brain Science data."*
- A prominent **maintenance-mode disclaimer** ("Bug fixes, security, docs,
  tests welcome; no new feature development planned") — transparency that
  *manages expectations* rather than scaring users off.
- **Three distinct support channels** (support guide, GitHub issues, community
  forum), each mapped to a question type.
- Institutional branding for credibility.
Lesson: a calm, factual **Project status / level-of-support** block is a sign
of maturity, not weakness — crucial for a research-stage project.

**12. Norse** — <https://github.com/norse/norse> ("Deep learning with spiking
neural networks in PyTorch") — praised alongside snnTorch for a "user-centric"
docs suite and a clean benchmark harness vs BindsNET/GeNN. Reinforces the SNN-
ecosystem norm: a comparison table positioning yourself vs sibling tools is
expected and welcomed.

### C. Principles guides consulted
- **Make a README** (<https://www.makeareadme.com/>) — section order + per-
  section rules (below).
- **Art of README** (<https://github.com/hackergrrl/art-of-readme>) — "cognitive
  funneling," README-as-front-door, anti-patterns.
- **Ten Simple Rules for Documenting Scientific Software** (PLOS Comp Biol,
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006561>)
  — quick-start is non-negotiable; assume the README is the *only* doc many
  users read.
- **awesome-readme** (<https://github.com/matiassingers/awesome-readme>) — the
  curated exemplar set above.

---

## Part 2 — Cross-cutting principles of a great scientific-project README

### The first screen (above the fold) — the only screen many people read
Order, top to bottom:
1. **Project name + logo** (or a clean title).
2. **One-line description** — what it is, in plain words, in <20 words.
3. **Benefit-stacked tagline** (optional second line) — FastAPI-style.
4. **Badge row** — credibility at a glance (see badge conventions below).
5. **An above-the-fold visual** — for this project, a GIF of the live 3D
   visualization. Show-don't-tell (Rich, snnTorch).
6. **A short "what makes this different" line** (Brian2) and/or a small
   **link menu / ToC** so the mixed audience self-routes (choo).
7. **The fastest possible "try it"** — one install command + one run command,
   or a zero-install path. (Ten Simple Rules: users quit if the quick-start is
   slow.)

### Section ORDER that works (synthesized)
"Cognitive funneling" — broad → specific (Art of README), with the scientific
additions (How-to-cite, Status, Science) slotted in:

> Name/one-liner → Hook visual + badges → **Highlights/features (benefit-tagged)**
> → **Quick-start (install + minimal runnable example with expected output)** →
> Why-it's-interesting (per audience) → The science/biology → Architecture →
> Performance/hardware envelope → Research status + How-to-cite → Documentation
> links → Contributing → License.

Keep installation *close to the top* but *after* the example hook if the
example is what sells it (Art of README puts usage before install; both are
high). Put **License high only if it's restrictive**; otherwise near the bottom.

### One-line description — how to write it
Formula: *"<Name> is a <category> for <who/what> using <key tech>."*
- snnTorch: "a Python package for gradient-based learning with spiking neural
  networks." Brian2: "a free, open source simulator for spiking neural
  networks." BindsNET: "simulating SNNs on CPUs or GPUs using PyTorch."
- For this project: *"<Name> is a GPU-accelerated simulator for biologically
  realistic spiking neural networks, with real-time 3D visualization."*

### One-paragraph elevator pitch — how to write it
3–4 sentences: (1) what it is, (2) the differentiator (the thing only you do —
real-time 3D viz of large biophysical networks on a single GPU), (3) who it's
for, (4) one concrete capability/result. Lead with the differentiator (Brian2).
Make it *user-centric* (saves the scientist time / lets you *see* the network),
not just "it's fast."

### Effective quick-start
- One command to install; one command to see something happen. Show **expected
  output** (Make a README: "show the expected output if you can").
- Offer a **zero-install path** if at all possible (snnTorch's Colab badge; for
  a GPU app, a "no GPU? run the headless NumPy/CPU smoke" path — this project
  has `SIM_BACKEND=numpy`, which is a perfect low-friction on-ramp to advertise).
- Keep the inline example the *smallest meaningful one*; link out to richer
  examples (Make a README; Art of README).

### Show, don't tell
- Embed a **GIF/screenshot of the 3D visualization** near the top (Rich,
  snnTorch). For a viz-centric tool this is the single highest-leverage asset.
- Put **benchmark/scaling as a chart or small table** (fiber, BindsNET, Norse).
- Diagram the **architecture/data-flow** (esbuild).
- Always put the **image first, caption/explanation second** (Rich).

### Signaling the different audiences in ONE doc
- Use Nengo's **capability-icon / per-audience bullet** technique. Explicitly:
  "For developers: …", "For neuroscientists: …", "For the curious: …" — or a
  small table. This lets each reader find their reason to care in ~5 seconds.
- An above-the-fold **link menu** ("Quick-start • The science • Architecture •
  Cite • Gallery") routes each audience to its depth.
- Gate jargon: the general reader gets analogies in the intro; the scientist
  gets precise terms + citations deeper down (see Part 4).

### Stating status/scope/limitations honestly (without undercutting)
- A factual **Project status / level of support** block (Allen SDK) — e.g.
  "active research project; APIs may change; CUDA required for full scale."
  Framing as *information* manages expectations and reads as maturity.
- A short **Scope & limitations** list: what it is NOT (not a production ML
  framework; not peer-reviewed for clinical use), the **hardware envelope**
  (VRAM per N neurons — the BindsNET gap), and known boundaries. Honest
  negatives are credibility, not weakness — and for *this* project, the owner's
  stated goal makes "honest negatives under strict biology" the actual
  deliverable, so a candid Status section is on-brand.

### README vs link-out — how much goes where
- README = front door: enough that someone can *evaluate and start* without
  reading source (Art of README) and without clicking away (Ten Simple Rules).
- Link out for: full API reference, deep tutorials, the science roadmap,
  per-experiment findings, design docs. "If examples laden the README, move
  them to a section/dir but keep them discoverable" (Ten Simple Rules).
- Rule of thumb: if a section is >2 screens, it probably belongs in `docs/`
  with a one-paragraph teaser + link in the README.

### Badge conventions (shields.io)
- **Worth having:** build/CI status, test coverage, docs status, package
  version (if published), license, Python version, and DOI/"cite this" if there
  is a paper/Zenodo. (Make a README; Ten Simple Rules.)
- Keep to one tidy row. Badges "instill confidence in project quality"
  (Ten Simple Rules) but a wall of 15 badges is noise.
- A live-demo or Colab badge (snnTorch) is high-value when a try-it path exists.

### Common ANTI-patterns to avoid
- **No quick-start / quick-start that takes >a few minutes** — top cause of
  abandonment (Ten Simple Rules).
- **Wall of text with no visuals** for a tool whose output is visual (the
  opposite of Rich). Telling instead of showing.
- **Burying the differentiator** — say in line 1–3 what only you do (Brian2).
- **No runnable example / theory without demonstration** (Art of README:
  "missing runnable examples wastes reader time").
- **Excessive length** — "excessive length kills utility" (Art of README); move
  depth to `docs/`.
- **Vague name with no explanation; broken links; empty sections** (awesome-
  readme hygiene).
- **Over-claiming / unbacked benchmarks** — for a research project especially,
  state seeds/conditions or link the finding; don't assert "fastest."
- **No hardware/scale envelope** for a GPU simulator (the BindsNET gap).
- **No How-to-cite / no Status** for scientific software.
- **Jargon with no on-ramp** for the general/biology reader (Part 4).

---

## Part 3 — Recommended section outline for THIS project

Ordered headings; one line each on what goes in it. Tailored to a biology-
grounded GPU brain simulator for a dev/neuroscience/biology/general audience.

1. **Title + logo + one-liner.** "<Name> — a GPU-accelerated simulator for
   biologically realistic spiking neural networks, with real-time 3D
   visualization." (The hook: plain words, the differentiator = *real-time 3D
   viz of large biophysical networks*.)

2. **Badge row.** CI, coverage, docs, license, Python version, (CUDA/CuPy),
   DOI/cite-this if available. One tidy line.

3. **The hook visual (above the fold).** An animated GIF of the live 3D
   network firing — the single most persuasive asset (Rich / snnTorch).
   Caption it in one sentence.

4. **Above-the-fold link menu.** `Quick-start • The science • Architecture •
   Performance • Cite • Gallery • Docs` — lets the mixed audience self-route
   (choo).

5. **Highlights (benefit-tagged bullets).** FastAPI-style: each bullet =
   capability + parenthetical payoff. E.g. "Biophysically realistic
   (Izhikevich, Hodgkin-Huxley, AdEx neuron models) — not rate-coded
   abstractions"; "Massive scale (10K–100K+ neurons on one NVIDIA GPU via
   CUDA/CuPy)"; "See it think (real-time OpenGL 3D visualization)"; "Brain
   regions, STDP/Hebbian/reward-modulated plasticity, and neuromodulators";
   "Runs without a GPU too (NumPy/CPU backend for laptops & CI)."

6. **Who it's for (per-audience signaling).** A short "For developers / For
   computational neuroscientists / For biologists / For the curious" block or
   table (Nengo). One line each, with the reason each cares.

7. **Quick-start.** Install (one block, list prerequisites incl. CUDA, and the
   **CPU fallback** `SIM_BACKEND=numpy` as the zero-friction path) → one
   command to launch the GUI demo → one command for a headless example → show
   **expected output** (a screenshot or the printed result). Keep the inline
   example minimal; link to more.

8. **What it is / why it's interesting** (the essay, kept tight — re-frame
   style). 2–4 short paragraphs: the goal (research toward biology-faithful
   learning agents — language & spatial navigation), why spiking + biophysics
   matters, what's novel. Lead with the differentiator; make it user-centric.

9. **The science & biology.** Plain-language intro to the modeled biology
   (neuron models, plasticity rules, neuromodulators, brain regions) with
   gentle term introductions + analogies for general readers, and citations /
   textbook touchstones for scientists (Brian2). Link out to the science
   roadmap and findings.

10. **Architecture.** A data-flow / module diagram (esbuild) + a short table of
    the main packages (engine, viz, experiment, runners) and the thread model.
    Two screens max; link to design docs for depth.

11. **Performance & hardware envelope.** Scaling numbers as a small chart/table
    (BindsNET/Norse), and **explicitly**: VRAM per N neurons, the
    GPU-vs-CPU-backend tradeoff, and known scale limits. (Fixes the #1 peer gap.)

12. **Research status, scope & limitations.** Honest "active research project;
    APIs may change; not peer-reviewed for clinical use; what it is NOT"
    (Allen SDK). Frame as information. This is on-brand given the project's
    "honest negatives are the deliverable" ethos.

13. **How to cite.** BibTeX / DOI / Zenodo (Brian2, BindsNET). Even a "if you
    use this in research, please cite …" placeholder.

14. **Documentation & further reading.** Links to full docs, tutorials,
    findings, roadmap, CLAUDE.md-style developer guide.

15. **Contributing.** Whether contributions are welcome + how; link to
    CONTRIBUTING and tests (`pytest tests/`).

16. **License.** Name + link (near bottom unless restrictive).

17. **Acknowledgments / credits** (optional). Contributors, inspirations,
    funding.

---

## Part 4 — Writing-style rules for the plain↔technical mix

1. **Define-on-first-use, in a half-sentence.** Introduce a term with a tiny
   gloss inline, e.g. "spiking neurons (cells that communicate in discrete
   electrical pulses, like real brain cells)," then use the term freely
   afterward. snnTorch's "brain is the perfect place to look for inspiration"
   framing is the model for easing a general reader in.

2. **Lead each technical section with one plain sentence, then go deep.** The
   first sentence is for the biologist/general reader; the rest is for the
   computational neuroscientist. (Nengo's layering.) This lets one doc serve
   four audiences without a separate "for beginners" version.

3. **Use one good analogy per hard concept — then drop it.** Examples that work:
   STDP ≈ "neurons that fire together, wire together" (Hebb); a spike ≈ "a
   neuron's all-or-nothing text message"; neuromodulators ≈ "chemical broadcast
   that changes the whole network's mood/learning rate"; membrane potential ≈
   "the neuron's charge building toward a trigger point." Don't stack analogies;
   one, then the precise term.

4. **Prefer concrete capability over abstract description.** Not "supports
   advanced plasticity mechanisms" but "neurons strengthen or weaken their
   connections based on spike timing (STDP), reward (dopamine-like signals), or
   activity (Hebbian)." Concreteness serves every audience.

5. **Benefit-tag technical bullets** (FastAPI/Typer). Every feature bullet:
   `<technical thing> — <why a human cares>`. This is the cleanest way to keep
   one bullet readable by both a developer and a curious reader.

6. **Link, don't lecture, for the deep stuff.** When a term needs a paragraph
   (e.g. Hodgkin-Huxley kinetics, surrogate-gradient BPTT), give one sentence +
   a link to the science doc/paper. Keep the README's reading level even.

7. **Cite for scientists, analogize for everyone else — in the same passage.**
   "Models reproduce Bi & Poo (1998) STDP timing curves" (for the scientist)
   can sit right after "connections strengthen when the sending neuron fires
   just before the receiving one" (for everyone). Both readers are served.

8. **Keep sentences short and active; avoid stacked nouns.** "The simulator runs
   100,000 neurons on one GPU" beats "GPU-accelerated large-scale neuronal
   population simulation is supported." Plain syntax carries technical content
   without dumbing it down.

---

## Sources
- awesome-readme (curated exemplars): <https://github.com/matiassingers/awesome-readme>
- Make a README (section order + per-section rules): <https://www.makeareadme.com/>
- Art of README (cognitive funneling, anti-patterns): <https://github.com/hackergrrl/art-of-readme>
- Ten Simple Rules for Documenting Scientific Software (PLOS Comp Biol): <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006561>
- FastAPI: <https://github.com/fastapi/fastapi>, <https://fastapi.tiangolo.com/>
- Typer: <https://typer.tiangolo.com/>
- Rich: <https://github.com/Textualize/rich>
- Nengo: <https://www.nengo.ai/>
- Brian2: <https://briansimulator.org/>
- snnTorch: <https://snntorch.readthedocs.io/>
- BindsNET (closest peer — GPU SNNs on PyTorch): <https://github.com/BindsNET/bindsnet>
- Norse: <https://github.com/norse/norse>
- Allen Institute SDK (honest status/scope model): <https://github.com/AllenInstitute/AllenSDK>
