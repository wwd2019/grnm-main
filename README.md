# GRNM: analysis source code

Source code and notebook workflows for the planarian GRNM analysis and the separate human brain organoid Pando-based prediction extension.

This is a **code-only release**. Expression and accessibility data, sample metadata, intermediate objects, trained weights and analysis results are not included. Notebook code and Markdown cells are preserved, with saved outputs and execution counts cleared. Running the workflows requires the corresponding input files under the original `dataset/` and `results/` paths.

## Repository layout

- `notebooks/python/`: four tutorials covering processing, model training, results and baseline comparisons; each presents the planarian analysis followed by the human extension.
- `notebooks/R/`: Pando processing and result reconstruction.
- `notebooks/planarian/`: historical planarian workflow.
- `src/pando/`: R scripts for saved-fit export and selected-target refitting.
- `src/model/`: human neural edge-ranking model.
- `src/scenic/`: GRNBoost2, motif refinement and AUCell scripts.
- `src/visualization/`: figure generation.
- `src/legacy/`: historical scripts, documentation and selected upstream SCENIC source with its original license and package metadata.

## Environment

Use Python 3.10 or 3.11 for the main workflows:

```sh
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
python -m jupyterlab
```

Use separate environments for `requirements-scenic.txt` and `requirements-legacy-planarian.txt`. R notebooks require R and an IRkernel installation; the R scripts specify their analysis package dependencies. These dependency lists are not lockfiles.

Before executing notebooks, restore the data and model files required by their paths. Historical scripts retain their original path assumptions. This code-only checkout does not reproduce results without those inputs.

## Workflow scope

Pando infers the initial regulatory network. The planarian AE/GCN learns gene representations; the added RNA-only prediction stage uses a residual network and MLP. The human edge-ranking extension uses a separate architecture and Pando-derived labels. GRNBoost2 performs SCENIC's initial co-expression network inference and provides candidate importance scores before motif refinement.

The original local project files are unchanged by this upload.
