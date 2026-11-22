# Reproducibility

The four tutorials run from the repository root or `notebooks/`. Default execution reconstructs saved model scores and comparisons; it does not automatically retrain or rerun SCENIC. Generated files are written to `results/generated/`, while archived results remain in `results/model/`, `results/pando/`, and `results/scenic/`.

## Python dependency status

The main `requirements.txt` was checked against an existing Python 3.10 environment: all ten declared packages satisfied their version ranges and imported successfully, `pip check` reported no broken requirements, and the default Python tutorial cells ran. The existing SCENIC environment satisfied and imported the SCENIC-specific packages, but lacks JupyterLab required by the shared base file. The historic planarian environment has not been validated against all pinned versions. A clean-environment installation could not be tested because this runtime cannot access PyPI; Jupyter kernel execution is also blocked by the sandbox. These observations do not prove a new installation will succeed.

## Model

`src/model/train.py` trains from the included Pando network, expression, motif annotations and GRNBoost2 adjacency; use `--out PATH` for a new, nonexistent output directory. Training was verified locally with Python 3.11.18, NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.9.1, PyTorch 2.6.0 and two CPU threads. In this environment all 1,794 rerun GLI3 scores exactly matched the saved TSV values. Other environments may yield small floating-point differences; score reconstruction uses a 1e-5 tolerance. These version records are not a complete lockfile.

## SCENIC

`src/scenic/grn.py`, `prepare_modules.py`, `prune_batches.py`, and `run_aucell.py` implement the saved GRNBoost2, module construction, cisTarget and AUCell stages. They write into `results/scenic/`; run them only on a working copy. Existing batch checkpoints are automatically reused, and the number of GRNBoost2 workers determines batch filenames: use the original default of eight to resume the supplied checkpoints. A fresh complete run is computationally expensive. Python 3.10.22, pySCENIC 0.12.1, Arboreto 0.1.6, ctxcore 0.2.0, NumPy 1.26.4 and pandas 2.3.3 were used. Pickle artifacts must only be loaded from a trusted source. Check `validation.md` for the precise scope of verification; resuming saved checkpoints is not a fresh calculation of each batch.

## Pando

The included `src/legacy/pando_original/` scripts record the original analysis; their absolute local paths require adaptation. The large original ATAC, fragments and selected intermediate matrices are excluded from the upload and locally preserved in the sibling `grnm-main-large-data-local/` directory. The original public data have links in the README; project-generated large intermediates have not been publicly deposited. The separate local full archive is `../GLI3_publication_20261121/`. The raw Pando pipeline has not been independently rerun end to end from this repository. Data access, permissions and an accessible deposit must be resolved before publication.

## Figures

The dot-matrix script reads the saved comparison table. `src/visualization/build_trees.py` renders the three saved selected-edge CSVs with the R renderers. The selected two-layer graphs include exploratory HES4/HES5 priority inclusion; scores are unsigned except for the Pando coefficient sign. Red labels mark a watchlist plus HES5, and do not establish binding or regulatory direction. Historical scripts under `src/legacy/` retain old paths and are not portable entry points.
