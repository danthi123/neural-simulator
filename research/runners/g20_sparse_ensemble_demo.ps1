# End-to-end demo of the 160-concept SPARSE-DISTRIBUTED G.20 ensemble.
#
# Loads the 5 sparse bridges (32 concepts each = 160 total) trained by
# g20_sparse_5bridge_chain.ps1, and exercises the full conversational
# stack through g20_multibridge.py --sparse:
#   - self-bridge concept recall (sparse-pattern readout)
#   - CROSS-BRIDGE pair memory (apple in nouns, big in adjectives)
#   - cross-bridge associative query + exact tag match
#   - N-word sentence spanning 3 bridges (dog=nouns, run=verbs, fast=adj)
#   - tag-name role queries (who/what-did) -- the v16-validated 100%
#     multi-seed mechanism, architecture-independent
#
# Waits for the 5th bridge if the training chain is still running.
$ErrorActionPreference = "Continue"
$BD = "research/findings/raw/g11_bg/g20_sparse_bridges"
$OUT = "research/findings/raw/g11_bg/g20_sparse_ensemble_demo.log"

# Wait (up to 60 min) for all 5 sparse bridges to exist
$maxWait = 3600; $elapsed = 0
while ($elapsed -lt $maxWait) {
    $n = (Get-ChildItem "$BD/*_sparse.simstate.h5" -ErrorAction SilentlyContinue).Count
    if ($n -ge 5) { Write-Host "[demo] all 5 sparse bridges present"; break }
    Write-Host "[demo] $(Get-Date) waiting: $n/5 sparse bridges"
    Start-Sleep -Seconds 30
    $elapsed += 30
}

$bridges = @(
    "$BD/bridgeA_nouns_sparse.simstate.h5",
    "$BD/bridgeB_verbs_sparse.simstate.h5",
    "$BD/bridgeC_adj_sparse.simstate.h5",
    "$BD/bridgeD_spatial_sparse.simstate.h5",
    "$BD/bridgeE_functional_sparse.simstate.h5"
)
$vocabs = @(
    "research/findings/raw/g11_bg/g20_bridgeA_nouns_vocab.txt",
    "research/findings/raw/g11_bg/g20_bridgeB_verbs_vocab.txt",
    "research/findings/raw/g11_bg/g20_bridgeC_adj_vocab.txt",
    "research/findings/raw/g11_bg/g20_bridgeD_spatial_vocab.txt",
    "research/findings/raw/g11_bg/g20_bridgeE_functional_vocab.txt"
)
$names = @("bridgeA_nouns","bridgeB_verbs","bridgeC_adj","bridgeD_spatial","bridgeE_functional")

# Script exercises: concepts list; pre-assoc query (noise baseline);
# cross-bridge remember (apple[A] is big[C]); cross-bridge query;
# exact tag; 3-bridge sentence (dog[A] run[B] fast[C]); role queries.
$script = @(
    "concepts",
    "what is apple",
    "remember apple is big",
    "what is apple",
    "is apple big?",
    "remember dog run fast",
    "who run fast?",
    "what did dog run?",
    "tags",
    "quit"
) -join ","

Write-Host "[demo] $(Get-Date) launching sparse ensemble demo"
python -m research.runners.g20_multibridge `
    --sparse --pattern-size 100 --n-shared-pool 2000 `
    --n-lang-input 8192 --sparsity 0.02 --seed 42 `
    --bridges $bridges `
    --vocab-files $vocabs `
    --names $names `
    --scripted $script 2>&1 | Tee-Object -FilePath $OUT

Write-Host "[demo] $(Get-Date) DONE: sparse ensemble demo complete"
