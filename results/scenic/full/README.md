# Saved complete SCENIC run

The completed analysis used 9,327 sampled cells and 6,798 genes for GRNBoost2. cisTarget constructed 4,788 modules and retained 377 regulons at NES >= 3.0; none is a GLI3 regulon. AUCell was then evaluated for the 377 retained regulons on the 9,327 cells with 33,538 ranking genes. See `pruning_summary.json`, `aucell_summary.json`, `regulon_targets.tsv.gz` and `aucell_matrix.csv.gz`.

The separately displayed GLI3 Top-10 and tree use the GRNBoost2 output before motif filtering. The post-hoc lower-threshold analysis is separate in `../motif_threshold_analysis/` and is not the default run. Low-detection cells can have zero-expression ties within AUCell's ranking window; see the historical logs for details.

The current stage scripts are under `src/scenic/`; see the root README and `docs/reproducibility.md` for paths and dependencies. Saved batch checkpoints are included, but large expression matrices and motif-ranking databases are omitted from this upload. Resuming checkpoints is not a from-scratch rerun. Do not run multiple writers against the same results directory.
