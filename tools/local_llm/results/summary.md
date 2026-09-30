| profile | ctx | load s | peak VRAM (MiB, desktop incl.) | headroom | short gen tok/s | long prompt tokens | long prompt tok/s | long gen tok/s | recall @long | T1 locate | T2 debug | T3 extend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-27b-iq4nl-mtp-128k-q4 | 131072 | 6.0 | 22831 | 1745 | 73.8 | 120000 | 694.1 | 47.0 | True | PASS (44s) | PASS (64s) | PASS (42s) |
| ollama-qwen38-27b-iq4nl | 131072 | 17.9 | 20890 | 3686 | 35.9 | 60000 | 781.5 | 1.0 | True | fail (175s) | fail (189s) | fail (184s) |
| unsloth-studio-qwen38-27b-iq4nl | 131072 | 16.1 | 23071 | 1505 | 13.2 | 60000 | 437.5 | 0.7 | True | PASS (213s) | PASS (1500s) | PASS (368s) |
