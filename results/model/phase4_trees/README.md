# Separate circular trees

The `*_edges.csv` files are selected two-layer edge tables. Re-render the separate pando+GRN, Pando baseline and SCENIC pre-motif trees with `python src/visualization/build_trees.py` from the repository root, with `RSCRIPT` set if `Rscript` is not on the PATH. Outputs go to `results/generated/phase4_trees/`; current figures are in `results/figures/`.

The saved trees contain 80, 50 and 81 first-layer targets respectively; HES4/HES5 were priority-included where available, and downstream edges are selected for display. The SCENIC plot shows GRNBoost2 before motif filtering: the completed default SCENIC run returned no GLI3 regulon. Model/GRNBoost2 scores are unsigned; Pando coefficient signs do not establish regulatory direction. Red labels identify the supplied watchlist plus HES5. These selected views are not independent experimental evidence.
