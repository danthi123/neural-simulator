export const meta = {
  name: 'refresh-user-facing-docs',
  description: 'Update stale user-facing docs (README/CONTRIBUTING/USER_GUIDE/CHANGELOG) against the true current project state, honesty-checked',
  phases: [
    { title: 'Ground truth', detail: 'snapshot the true current project state' },
    { title: 'Audit + edit', detail: 'one agent per user-facing doc, grounded edits in place' },
    { title: 'Honesty verify', detail: 'flag any overclaim / phenomenal-experience assertion' },
  ],
}

const WT = args && args.worktree ? args.worktree : '/home/dant123/Projects/sim-worktrees/docs-refresh'

const GROUND_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['project_one_liner', 'current_capabilities', 'status_numbers', 'honesty_framing', 'recent_milestones', 'other_user_facing_docs'],
  properties: {
    project_one_liner: { type: 'string' },
    current_capabilities: { type: 'array', items: { type: 'string' } },
    status_numbers: { type: 'object', additionalProperties: true },
    honesty_framing: { type: 'string' },
    recent_milestones: { type: 'array', items: { type: 'string' } },
    other_user_facing_docs: { type: 'array', items: { type: 'string' } },
  },
}

phase('Ground truth')
const ground = await agent(
  `You are establishing the TRUE current state of the neural-simulator project so its user-facing docs can be corrected. ` +
  `Work ONLY in the worktree ${WT} (it is checked out on main, HEAD current). Do NOT edit anything in this phase — read only.\n\n` +
  `Read these to build the snapshot (they are the ground truth):\n` +
  `  - ${WT}/ROADMAP.md  (the plain-language current-state surface — what's done / in progress / left; the shorthand glossary)\n` +
  `  - ${WT}/docs/PRODUCTION_INTEGRATION_LEDGER.yaml  (esp. the top summary fields: total faculties, on_by_default counts, default_on_spiking_faculties, the 'note' field, and the per-faculty 'faculty' one-liners) — this is the authority on what is ACTUALLY wired + on-by-default in production\n` +
  `  - the 15 most recent files in ${WT}/research/findings/ (ls -t | head -15) — read their titles + frontmatter status to know the latest landed results\n` +
  `  - the current CLAUDE.md ACTIVE MISSION block + the memory of the north-star (genuine-conversation single spiking-substrate brain; honesty-boundary-as-deliverable)\n` +
  `  - run the count commands: wc -l ${WT}/sim/*.py ; find ${WT}/tests -maxdepth 1 -name 'test_*.py' | wc -l ; ls ${WT}/research/findings/*.md | wc -l ; grep -l 'def main' ${WT}/research/runners/*.py | grep -v aggregate | wc -l\n\n` +
  `Produce a compact, ACCURATE snapshot for doc-writers: the project one-liner (the pivoted north-star, NOT an old fact-recall framing), the current headline capabilities that are ON-BY-DEFAULT in production (from the ledger, not de-risks), the real status numbers, the honesty-boundary framing (self-reports are functional read-outs; never assert phenomenal experience), the recent milestones a user-facing doc should mention, and any OTHER user-facing docs you spotted (paths) beyond README/CONTRIBUTING/USER_GUIDE/CHANGELOG. ` +
  `Be precise and grounded — every capability you list must be traceable to the ledger or a finding. Do NOT overclaim.`,
  { label: 'ground-truth', phase: 'Ground truth', schema: GROUND_SCHEMA, model: 'sonnet', effort: 'medium' }
)

const g = JSON.stringify(ground, null, 1)

const DOC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['file', 'changed', 'stale_items_fixed', 'summary'],
  properties: {
    file: { type: 'string' },
    changed: { type: 'boolean' },
    stale_items_fixed: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['what_was_stale', 'what_changed', 'evidence'],
        properties: {
          what_was_stale: { type: 'string' },
          what_changed: { type: 'string' },
          evidence: { type: 'string' },
        },
      },
    },
    lines_touched: { type: 'number' },
    summary: { type: 'string' },
  },
}

const DOCS = [
  { file: 'README.md', guide: 'The PRIMARY user-facing doc and the one that goes stale worst. Fix: the project description/tagline (it must reflect the pivoted north-star — a genuinely-conversing single spiking-substrate brain with memory/emotion/curiosity/self-awareness, honesty-as-deliverable — NOT an old fact-recall/RAG or nav-demo framing if present), the current capabilities/status section (align to what is ON-BY-DEFAULT in production per the ground truth), any status numbers/counts, feature lists, and any quick-start / usage that no longer matches the code. Keep it a real README (concise, skimmable).' },
  { file: 'CONTRIBUTING.md', guide: 'Fix stale counts (test files, runners, findings), stale module/line references, stale public-API import examples, and any workflow description that no longer matches the repo (gates, push_both, the branch model = main + research/<slug> topic branches).' },
  { file: 'USER_GUIDE.md', guide: 'Fix stale feature descriptions / UI-console references / commands that no longer match. It is GUI/console-focused; keep that scope. Correct anything describing capabilities the brain no longer has or now has.' },
  { file: 'CHANGELOG.md', guide: 'This file is 127KB and append-only history — do NOT rewrite existing entries. ONLY read the TOP (most recent ~80 lines) and ADD any missing recent milestone entries in the SAME style/format as the existing top entries, drawn from the ground truth recent_milestones + the latest findings. If the top already covers recent milestones, make no change.' },
]

phase('Audit + edit')
const reports = await parallel(DOCS.map((d) => () =>
  agent(
    `You are correcting ONE user-facing doc so it reflects the TRUE current state of the neural-simulator project. ` +
    `Edit ONLY the file ${WT}/${d.file} — in place, using your Edit/Write tools in the worktree ${WT}. Touch NO other file.\n\n` +
    `THE GROUND TRUTH (authoritative — every claim you write must be traceable to it; do NOT invent capabilities):\n${g}\n\n` +
    `SCOPE FOR THIS FILE: ${d.guide}\n\n` +
    `HARD RULES (this project's honesty discipline — violating them is worse than leaving staleness):\n` +
    `  1. NEVER assert the brain has phenomenal experience / real feelings. Emotion/self-awareness are FUNCTIONAL read-outs ('its familiarity monitor reads this as novel'), never phenomenal claims. If the doc currently overclaims sentience/feelings, FIX it.\n` +
    `  2. NEVER claim a capability that is not in the ground truth (ledger/findings). Prefer UNDERSTATING to overclaiming. A user-facing doc that overclaims is a defect.\n` +
    `  3. Distinguish what is ON-BY-DEFAULT in production from what is a research de-risk. Describe the shipped, default-on behavior as the product; mention research scope as research.\n` +
    `  4. Keep prose lines <= 800 characters (a doc gate; tables/code exempt). Preserve the doc's existing voice + structure — make SURGICAL corrections, not a wholesale rewrite, unless a whole section is clearly stale.\n` +
    `  5. Do not touch code, findings, or other docs. Only ${d.file}.\n\n` +
    `First READ the current ${WT}/${d.file} fully, identify what is stale/wrong/missing vs the ground truth, then make the edits in place. ` +
    `Return: whether you changed it, each stale item you fixed with its evidence (which ground-truth fact justifies it), lines touched, and a one-line summary. If nothing is genuinely stale, changed=false and say so honestly — do not churn.`,
    { label: `edit:${d.file}`, phase: 'Audit + edit', schema: DOC_SCHEMA, model: 'sonnet', effort: 'medium' }
  )
))

phase('Honesty verify')
const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'flags'],
  properties: {
    verdict: { type: 'string', enum: ['clean', 'issues'] },
    flags: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'quote', 'problem'],
        properties: {
          file: { type: 'string' },
          quote: { type: 'string' },
          problem: { type: 'string' },
        },
      },
    },
  },
}
const verify = await agent(
  `Adversarially review the doc edits just made in the worktree ${WT}. Run: cd ${WT} && git --no-pager diff -- README.md CONTRIBUTING.md USER_GUIDE.md CHANGELOG.md . ` +
  `Read the FULL diff. Your job is to catch HONESTY violations introduced (or left uncorrected) in the user-facing docs, against this project's discipline:\n` +
  `  - any assertion that the brain FEELS / is conscious / has phenomenal experience (must be functional read-outs only)\n` +
  `  - any capability claim NOT supported by the ground truth below (overclaim)\n` +
  `  - describing a research de-risk as if it were a shipped, on-by-default product feature\n` +
  `  - any prose line > 800 chars (doc gate W2)\n\n` +
  `THE GROUND TRUTH (the only claims allowed):\n${g}\n\n` +
  `Return verdict 'clean' if the edited docs are honest + grounded + gate-safe, else 'issues' with a flag per problem (file, the exact quoted text, and the problem). Be strict: on user-facing docs, an overclaim is a real defect.`,
  { label: 'honesty-verify', phase: 'Honesty verify', schema: VERIFY_SCHEMA, model: 'opus', effort: 'high' }
)

return {
  ground_truth: ground,
  doc_reports: reports.filter(Boolean),
  honesty: verify,
  worktree: WT,
}