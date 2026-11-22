# Preserved project source

This directory preserves historical project-owned source files with their original directory layout. `grnm-0.3/` contains the three planarian phase scripts, the human three-stage script, `phase2_rna2z.py`, `phase3_edge_predict.py`, `grnm2.py`, `grn_common.py`, and the `dimensionality_reduction/`, `clustering/` and `plotting/` dependencies. `2026_11_16_GLI3/` contains the historical GLI3 and SCENIC scripts, including a subset of upstream SCENIC source with its license and package metadata. Install the official upstream package for a complete SCENIC installation. Installed third-party R library files are not project source and are not copied here. These historical scripts retain their original assumptions; they are not the current human pando+GRN implementation.

The historical planarian entry points are `grnm-0.3/scripts/phase1_grn_z.py`, `phase2_rna2z.py` and `phase3_edge_predict.py`. They use SMED/SMESG gene identifiers and must not be confused with the human GLI3 workflow. Small planarian inputs are in `dataset/legacy/planarian/`; the large `data/rna_pca_new.csv` input is excluded from the upload and exists in the local large-data backup. See `dataset/legacy/planarian/README.md` and `notebooks/planarian/01_planarian_legacy.ipynb`.

The human historical `new_human_results.py` previously expected `example_data/glm_network_coefs_human.csv`. That original input is unavailable and the user has excluded the old human result from the release. The old human pipeline cannot be claimed to run from scratch with this repository; no replacement input was invented. The supported current human analysis is in `src/model/`, `src/pando/`, and `notebooks/python/`.

The original scripts use their historical working-directory layout and some contain local machine paths. A compatible historical environment (including DGL and python-louvain for phase 1) is required. Neither the historical human nor the historical planarian full pipeline has been rerun end to end in this reviewer package.

The historical planarian `grnm-0.3/notebooks/example.ipynb` is preserved with its code cells intact and old execution outputs cleared.

