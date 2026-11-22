# Figures and tables

`results/figures/gli3_reviewer_dot_matrix.pdf` is the one-panel Top-10 figure with all gene names. `results/figures/` contains the three separately rendered circular trees; `results/model/phase4_trees/` contains their selected edge tables and archival figures. `results/model/comparison_top10_four_methods.tsv` contains the corresponding rankings; the notebook recomputes the comparison and writes a new copy to `results/generated/`.

The SCENIC tree and its Top-10 list use GRNBoost2 before motif pruning: no GLI3 regulon passed the complete SCENIC pipeline. Tree edges are selected for display, with HES4/HES5 priority inclusion where available. The pando+GRN scores are unsigned predictions and Pando signs are model coefficients, not experimental evidence of activation or repression.
