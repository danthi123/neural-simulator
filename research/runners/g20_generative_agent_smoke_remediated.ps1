# Stage-1 generative agent — scripted GPU smoke (integration gate).
# MUST prove: grounded retrieval, coref ('what about it'), yes/no,
# and ABSTENTION on an unknown word (no confabulation = the moat).
$BD = "research/findings/raw/g11_bg/g20_sparse_bridges_320_remediated"
$V  = "research/findings/raw/g11_bg"
$names = @("bridgeA_nouns","bridgeB_verbs","bridgeC_adj","bridgeD_spatial","bridgeE_functional")
$BR = $names | ForEach-Object { "$BD/${_}_sparse64.simstate.h5" }
$VF = $names | ForEach-Object { "$V/g20_${_}_vocab64.txt" }
python -m research.runners.g20_generative_agent `
    --sparse --pattern-size 100 --n-shared-pool 2000 --sparsity 0.007 --seed 42 `
    --bridges $BR --vocab-files $VF --names $names `
    --scripted "remember apple is big,what is apple,what about it,is apple big,what is zzznonsense,quit" `
    --log "$V/g20_generative_agent_smoke_remediated.jsonl" 2>&1 |
    Tee-Object -FilePath "$V/g20_generative_agent_smoke_remediated.log"
