# Validation record (2026-11-21)

Checks used the current relative-path repository layout. R and Python notebooks are separate.

- **R tutorial:** all code cells in `notebooks/R/01_pando_workflow.ipynb` ran sequentially using R 4.2.3 via `src/validate_r_notebook.py`; see `r_notebook_validation.log`. The tutorial inspects existing preprocessing checkpoints. It reconstructs the complete network from all 3,846 saved per-target fit checkpoints, runs BH correction and writes results to `results/generated/pando_export/`. Its 49,764-edge TF–gene CSV has the same SHA-256 as `results/pando/TF_gene_network.csv`.
- **R selected-target fit:** the HES5 Gaussian GLM was refit using the included coarse-grained RNA/ATAC and motif inputs. Its 877 coefficients and model diagnostics matched the saved HES5 fit within 1e-8. This is one target; the full upstream raw-data processing and all model fits have not been rerun from scratch.
- **Python tutorials:** four notebooks passed schema checks and their default code cells completed sequentially in two local Python environments. A fresh CPU training run selected the same architecture and epoch; all 1,794 GLI3 score values matched the archived TSV exactly in this run. HES5, PAX6 and HES4 ranked first, second and fifth. See the two `notebook_validation*.log` files.
- **SCENIC:** module construction ran anew and all 4,788 module-index rows matched. GRNBoost2 aggregation, cisTarget and AUCell resumed the supplied checkpoints; the 626,359 adjacency rows, 23,892 regulon-target rows and all 9,327 × 377 AUCell values matched. No GLI3 regulon passed the default motif threshold. This did not recompute every batch from scratch.
- **Cross-pipeline consistency:** the R network matches the Pando input actually read by Python (49,764 rows), all 62,354 significant region rows match the R re-export, and the saved comparison matches all 40 recalculated Top-10 rows.
- **Figures:** the one-panel dot matrix and three R circular trees were regenerated. The tree renderer uses saved selected edge tables and emitted nonfatal ggplot/ggraph warnings.

A Jupyter kernel could not be started in this sandbox because required ZeroMQ sockets were denied. These are direct code-cell tests, not interactive Jupyter frontend tests; the local R environment also lacks IRkernel. Installation in a fresh environment is not yet verified. Large inputs have not been deposited publicly; a GitHub checkout supports the default result-display tutorials, but not full retraining without those inputs.

## Upload-sized directory

After large files were moved to the local backup, all four Python tutorials and the R tutorial completed their default code cells (`python_notebook_upload_validation.log` and `r_notebook_upload_validation.log`). The Python training tutorial displayed saved scores because the expression matrix was absent. The R tutorial displayed the saved motif summary and reconstructed the network from retained small fit checkpoints. These display-only fallbacks did not retrain the Python model or recalculate the motif matrix. Full retraining/refitting requires restoring the separately listed large files. Interactive Jupyter kernels remain untested.

## Historical planarian source

The historical planarian source and small inputs are now included separately. The planarian notebook default code cells completed. In a temporary working directory outside the upload, phase 2 ran for one epoch using an existing phase 1 embedding; phase 3 ran for one epoch and produced a deliberately partial prediction of 1,000 pairs. These are smoke tests, not final model results or an end-to-end three-stage reproduction. Phase 1 was not retrained: the upload excludes the 66 MiB planarian RNA PCA input, and a compatible DGL environment and full end-to-end run remain to be verified. Historical human three-stage outputs are not included.

## Planarian-first tutorial layout

The four Python tutorials now present planarian data/results before the human extension. Saved planarian phase 1 and phase 2 embeddings and the phase 3 classifier are included; a Hes1-associated top-10 example was generated from the saved classifier using the current seven-class decoding logic. This is a candidate ranking, not a validated regulatory network. No comparable planarian Pando/SCENIC baseline is supplied. All default Python notebook code cells completed in the local environment; planarian full retraining and interactive Jupyter remain unverified.
