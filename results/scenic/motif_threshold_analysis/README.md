# GLI3 cisTarget motif threshold sensitivity

This analysis reruns the full GLI3 cisTarget motif enrichment and leading-edge target pruning on the six coexpression modules and two hg38 ranking databases from the expanded pySCENIC analysis. Rank threshold is 5,000; AUC fraction 0.05. It reuses the same ranked GRNBoost2 edges. The comparison universe contains 1,794 genes, including HES4, and 48 matches to the updated candidate table, as in the saved model comparison.

- NES >= 3.0, GLI3-associated motifs (the original/default setting): no motif passes; no GLI3 regulon is available.
- NES >= 2.0, GLI3-associated direct or inferred motifs: three motif-module rows, 41 unique leading-edge targets across the full module input, 16 with a positive-importance GLI3 edge within the comparison universe. One of the top ten is a candidate, GPC6. Motifs at this setting are inferred rather than directly annotated.
- NES >= 1.9, directly GLI3-annotated motifs only: one motif-module row (`transfac_pro__M07292`, NES ~1.9186), 234 unique leading-edge targets across the full module input, 60 with positive-importance GLI3 edges in the comparison universe. One of the top ten is a candidate, GPC6.

The 1.9 threshold was chosen after observing the maximum directly annotated motif NES (1.9186); 2.0 was also explored after observing the scores. These are post-hoc exploratory sensitivity checks, not conventional SCENIC evidence, prespecified tests or FDR-controlled results. The 2.0 and 1.9 rows use different annotation filters and cannot isolate the effect of threshold alone. Under either setting, neither HES4 nor HES5 appears in the extracted leading-edge target set. No claim of direct binding follows from motif enrichment. A hit denotes candidate-table membership, not an independently verified GLI3 target.

The saved filtered target sets, ranked score tables and `threshold_comparison.tsv` are preserved. The original sensitivity script is historical and retained under `src/legacy/`; it uses old machine-specific paths and has not been validated as a portable rerun. The displayed SCENIC tree is pre-motif, not a tree of these filtered results.
