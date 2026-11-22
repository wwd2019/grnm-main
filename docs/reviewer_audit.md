# Reviewer and editor audit

## Checked in the current upload directory

- The human and planarian tutorial notebook files pass notebook-format validation. Default R and Python code cells execute sequentially in the existing local environments. The R notebook rebuilds the Pando network from included fit checkpoints; the Python notebooks show saved scores when the expression matrix is absent and recompute the comparison. See `r_notebook_upload_validation.log` and `python_notebook_upload_validation.log`. The default cell-source tests also passed in a separate distribution copy without the large files or pre-generated outputs (`fresh_distribution_R.log`, `fresh_distribution_python.log`).
- The R-exported TF–gene network matched the saved 49,764-edge table; significant regional edges matched across 62,354 rows. After adding missing-input checks, the Python model was retrained again using the full local expression input; all 1,794 saved GLI3 scores were identical in that environment. The 40 Top-10 records and four hit counts were checked.
- Default SCENIC pruning produced no GLI3 regulon. Its displayed GLI3 comparison is GRNBoost2 before motif pruning, and this boundary is shown in the figure and tutorial. The pando+GRN model uses GRNBoost2 as a feature.
- HES4 was added after prior inspection, HES4 and PAX6 were already first-stage Pando edges, and the candidate benchmark was selected for this study. These data cannot establish unbiased comparative superiority or prove direct GLI3 binding. Pando `padj_min` is a summary of regional adjusted p-values, not a joint TF–gene test.
- Current primary figures are in `results/figures/`; the Top-10 dot matrix labels genes and calls them priority candidates. Its 3/10, 2/10 and 2/10 figures are descriptive counts, not independently verified target probabilities. Historical two-panel comparison files were moved out of the upload directory.
- Files over 50 MiB are not retained in the upload directory; the preserved local backup has the same relative paths. The local fragments archive copy is incomplete and is excluded from the suggested restore command.

## Outstanding before an external reproducibility claim

1. Provide an authorized public deposit for the project's exact large prepared matrices, or a portable, tested complete preparation workflow. Original-study accession links do not supply all project-generated inputs. Without them the Python tutorial displays saved scores; full retraining and selected-target R refitting cannot run from this upload alone.
2. Verify installation in a fresh, independent R/Python environment and execute both notebooks with actual Jupyter kernels. The present sandbox denies the local sockets required to launch a Jupyter kernel; the checks here executed cell source directly. The dependency files and setup commands are not complete cross-platform lockfiles.
3. Select a license for the original project code and verify the licenses and attribution requirements for bundled third-party materials; no root project license is currently supplied.
4. Independently reproduce raw-data Pando preprocessing and a full SCENIC run from scratch if those claims are needed. The present tests use prepared inputs or resume saved checkpoints. The historical planarian three-stage code has been restored, but its full training has not been rerun in this package. Check publication permissions and sample-level correspondence for all derived input files.
5. For any claim of generalization or improvement, define an independent evaluation with prespecified genes/TFs and metrics; investigate the opposite GLI3–HES4 Pando coefficient signs across the two Pando datasets. Supply appropriate biological validation before interpreting predicted regulatory direction or direct binding.

These outstanding items cannot be resolved by relabeling saved figures or by re-running the same targeted comparison.
