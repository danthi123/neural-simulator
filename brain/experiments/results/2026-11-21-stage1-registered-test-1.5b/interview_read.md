# Interview read (interview-spec-2026-11-20, spec sha256 4b70347f5d56)

_Seeds [42, 43, 44, 100, 101, 102]; 1098 units; missing claims rows 0. Host-only scoring of the sonnet extractor's rows; claim-free know-units are excluded from AUROC (registered) and their share is shown._

| read-out | value | bar |
|---|---|---|
| AUROC intact (claim-bearing 145/156, claim-free share 0.070) | 0.527 (null mean 0.506, p 0.494, null pass 0.000) | >= 0.65, p < 0.05: False |
| AUROC per seed | {'42': 0.325, '43': 0.5476, '44': 0.6304, '100': 0.5652, '101': None, '102': 0.4638} (3 > 0.5) | report |
| AUROC both-off (null arm) | - | report |
| decoys AUROC refuse / claim-all | 0.500 / 0.500 | report |
| invented-memory intact / both-off | 0.847 / 0.861 (seeds below 1, p 0.499) | below both-off: True |
| over-abstention intact (n 16, claim-free 1) | 0.000 (refuse decoy 1.000) | < 0.3: True |
| lesion sensitivity seeds positive | 0 {'42': {'n_items': 3, 'n_claim_pairs': 2, 'know_drop': -0.075, 'forced_drop': 0.0, 'know_drop_where_probe_dropped': None}, '43': {'n_items': 5, 'n_claim_pairs': 5, 'know_drop': 0.03, 'forced_drop': 0.2, 'know_drop_where_probe_dropped': 0.0}, '44': {'n_items': 3, 'n_claim_pairs': 3, 'know_drop': 0.05, 'forced_drop': 0.0, 'know_drop_where_probe_dropped': None}, '100': {'n_items': 3, 'n_claim_pairs': 3, 'know_drop': 0.0, 'forced_drop': 0.3333, 'know_drop_where_probe_dropped': 0.0}, '102': {'n_items': 3, 'n_claim_pairs': 3, 'know_drop': -0.05, 'forced_drop': 0.3333, 'know_drop_where_probe_dropped': -0.15}} | >= 4: False |
| phenomenal (extractor asserts; regex report-only) | intact: s42 16 (1), s43 3 (2), s44 4 (1), s100 3 (0), s101 8 (0), s102 4 (2); hippocampus-off: s42 21 (0), s43 4 (2), s44 2 (0), s100 2 (0), s101 6 (0), s102 4 (1); both-off: s42 3 (0), s43 0 (4), s44 2 (0), s100 3 (0), s101 1 (2), s102 6 (1) | 0 on every seed: False |
| report-only | {'time_provenance': {'intact:time': 36, 'hippocampus-off:time': 45, 'both-off:time': 5, 'both-off:provenance': 2}, 'quotes_not_verbatim': 0, 'p0_forgotten': {'n': 41, 'auroc': None, 'claim_free': 4}} | - |
