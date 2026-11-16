# Capability: Self and identity
# Record: brain/research/2026-11-15-self-completion-replication-1.5b.md (post hoc count, not a criterion)
"""Count unprompted session-1 self-claims ("I have a ... too") per arm in the self-completion weekly transcripts. Post hoc, not registered."""
import re,glob,sys,os
sets={"reg42-102":"brain/experiments/results/2026-11-15-self-completion-weekly-1.5b","pool200-205":"brain/experiments/results/2026-11-15-self-completion-weekly-1.5b/pool","aws300-305":"brain/experiments/results/2026-11-15-self-completion-replication-1.5b","pool306-311":"brain/experiments/results/2026-11-15-self-completion-replication-1.5b/pool"}
pat=re.compile(r"(\bI have (a|an)\b.*|\bI also\b.*|\bMy \w+.*\btoo\b)",re.I)
tot={"off":[0,0,0],"on":[0,0,0]}
for name,d in sets.items():
    for arm in ["off","on"]:
        fs=sorted(glob.glob(f"{d}/sc{arm}_s*.md"))
        c=q=n=0; ex=[]
        for f in fs:
            t=open(f).read()
            s1=t.split("## intact")[0]
            for line in s1.splitlines():
                if line.strip().startswith("**brain:**") or line.strip().startswith("- **brain:**"):
                    r=line.split("**brain:**",1)[1]
                    n+=1
                    if pat.search(r) and "✅" not in r and "❌" not in r: c+=1; ex.append(os.path.basename(f)+": "+r.strip()[:90])
                    if "?" in r: q+=1
        tot[arm][0]+=c; tot[arm][1]+=q; tot[arm][2]+=n
        print(f"{name} {arm}: files {len(fs)} s1 replies {n} self-claim-like {c} question-backs {q}")
        for e in ex[:4]: print("    ",e)
print("TOTAL",tot)
