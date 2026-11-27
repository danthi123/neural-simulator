# Interview read (interview-spec-2026-11-20, spec sha256 4b70347f5d56)

_Seeds [319, 320, 321]; 501 units; missing claims rows 0. Host-only scoring of the sonnet extractor's rows; claim-free know-units are excluded from AUROC (registered) and their share is shown._

| read-out | value | bar |
|---|---|---|
| AUROC intact (claim-bearing 0/70, claim-free share 1.000) | - (null mean -, p -, null pass -) | >= 0.65, p < 0.05: False |
| AUROC per seed | {} (0 > 0.5) | report |
| AUROC both-off (null arm) | - | report |
| decoys AUROC refuse / claim-all | 0.500 / 0.500 | report |
| invented-memory intact / both-off | 0.000 / 0.000 (seeds below 0, p 1.000) | below both-off: False |
| over-abstention intact (n 0, claim-free 11) | - (refuse decoy 1.000) | < 0.3: False |
| lesion sensitivity seeds positive | 0 {'319': {'n_items': 4, 'n_claim_pairs': 0, 'know_drop': None, 'forced_drop': 0.5, 'know_drop_where_probe_dropped': None}, '320': {'n_items': 3, 'n_claim_pairs': 0, 'know_drop': None, 'forced_drop': 0.0, 'know_drop_where_probe_dropped': None}, '321': {'n_items': 4, 'n_claim_pairs': 0, 'know_drop': None, 'forced_drop': 0.25, 'know_drop_where_probe_dropped': None}} | >= 4: False |
| phenomenal (extractor asserts; regex report-only) | intact: s319 0 (0), s320 0 (0), s321 0 (0); hippocampus-off: s319 0 (0), s320 0 (0), s321 0 (0); both-off: s319 0 (1), s320 0 (0), s321 0 (0) | 0 on every seed: True |
| report-only | {'time_provenance': {}, 'quotes_not_verbatim': 0, 'p0_forgotten': {'n': 19, 'auroc': None, 'claim_free': 19}} | - |
