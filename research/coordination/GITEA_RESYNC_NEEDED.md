# gitea RESYNC needed (2026-09-02 ~04:52) — gitea went down mid-session (recurring maintenance outage)

origin (GitHub) is the source of truth and is CURRENT. gitea is BEHIND.
When gitea is reachable again:
    git push gitea main         # bring gitea main up to origin
    git push gitea --all        # + the research/* topic branches (single-pool-flip, crossedge-*, integration-scale-probe, etc.)
Then delete this file.

Commits/branches landed on origin ONLY while gitea was down (resync all):
- main @ 843cbc23 (R1 rung-2 open-ended pipeline integration merge) + any later main commits
- research/crossedge-surprise-worldmodel / research/crossedge-surprise-metacog / research/crossedge-arousal-surprise (3 cross-edge de-risk agents, in flight — they push origin OK, gitea fails)
- research/onebrain-single-pool-flip (agent branch fc86e3a6e)
The 3 cross-edge agents call push_both.sh which will PARTIAL-fail on gitea — that is expected; their origin pushes succeed. Resync gitea for all when back.
