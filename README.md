# GRNM: planarian analysis and human pando+GRN extension

This repository presents the planarian GRNM analysis first and the human brain organoid pando+GRN extension second in each Python tutorial. The human R Pando tutorial is separate. Paths in the runnable tutorials are relative to the repository root. Start Jupyter anywhere inside the repository and choose the corresponding R or Python kernel.

```text
├── notebooks/
│   ├── R/                  Pando processing, fitted models and result reconstruction
│   ├── python/             each step: planarian first, then human extension
│   └── planarian/          separate historical planarian three-stage tutorial
├── src/
│   ├── pando/              portable R result export and selected-target refit
│   ├── model/              pando+GRN training
│   ├── scenic/             SCENIC stage scripts
│   ├── visualization/      figure scripts and R renderers
│   └── legacy/             original historical scripts and tutorials
├── dataset/                locally available source files and intermediate data
├── results/
│   ├── planarian/          saved planarian model outputs and example predictions
│   ├── figures/            human comparison figures
│   ├── tables/             compact display and comparison tables
│   ├── pando/              Pando model results
│   ├── model/              pando+GRN model and rankings
│   ├── scenic/             SCENIC results and checkpoints
│   └── generated/          recreated outputs (not tracked by Git)
└── docs/                   data provenance and validation records
```

## Quick start: view the results

The tutorial notebooks include saved outputs, so their tables and figures can be viewed immediately after opening them. Running the cells again is optional for inspection.

From the repository root on Linux or macOS, create a Python 3.10/3.11 environment, install the notebook dependencies, and start JupyterLab:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

In JupyterLab, open the four notebooks in `notebooks/python/` in numerical order; each presents planarian first and human second and select **Run All Cells**. The uploaded files support result inspection and the comparison; when the large expression matrix is absent, Step 2 shows saved scores and says that reconstruction is skipped. The commands above describe setup, but installation in a fresh environment and interactive Jupyter have not yet been verified. Windows users should use the equivalent environment activation command.

The R notebook is separate. With R installed and the Python environment above activated, the following commands install its packages and register the kernel with that Jupyter environment:

```bash
Rscript -e 'install.packages(c("IRkernel", "data.table", "jsonlite", "BiocManager"), repos="https://cloud.r-project.org")'
Rscript -e 'BiocManager::install("GenomicRanges", ask=FALSE, update=FALSE)'
Rscript -e 'IRkernel::installspec(user=TRUE)'
```

`Matrix` is also required and is normally supplied with R; install a compatible version if your R installation does not contain it. Restart JupyterLab, open `notebooks/R/01_pando_workflow.ipynb`, choose the R kernel, and select **Run All Cells**. It re-exports the Pando network from included fit checkpoints; when the large motif matrix is absent, it shows the saved summary instead. Re-exporting writes files to `results/generated/`, including files above 50 MiB. These installation commands, a fresh R setup and Jupyter kernel launch have not been verified here.

For optional retraining or the complete original-data workflow, follow **Downloading and placing large files** below first. The exact project-generated large inputs do not yet have a public download, so a reviewer currently cannot perform full retraining from a GitHub download alone.

## Tutorials

1. [R Pando tutorial](notebooks/R/01_pando_workflow.ipynb): inspect the 00–09 preprocessing and model checkpoints, then regenerate the Pando network from the saved fitted models. The notebook describes which original steps are being inspected rather than recomputed.
2. [Step 1: processing](notebooks/python/01_data_processing.ipynb), [Step 2: training](notebooks/python/02_model_training.ipynb), [Step 3: model results](notebooks/python/03_model_results.ipynb), [Step 4: baseline comparison](notebooks/python/04_baseline_comparison.ipynb). Set `RUN_MODEL_TRAINING=1` for an optional new training run.
3. [Planarian historical three-stage workflow](notebooks/planarian/01_planarian_legacy.ipynb): separate SMED/SMESG code and inputs. Its historical full run requires a large planarian RNA PCA file that is not included; the notebook defaults to inspecting the small inputs. The original scripts and their dependencies are preserved under `src/legacy/original_project/grnm-0.3/`.

Install `requirements.txt` for the main tutorials and model training. SCENIC uses a separate environment (`requirements-scenic.txt`); the historical planarian training code uses `requirements-legacy-planarian.txt` in its own environment. These files are not fully validated lockfiles; the locally recorded training versions are in `docs/reproducibility.md`. The R tutorial requires R with IRkernel, data.table, GenomicRanges, Matrix and jsonlite; refitting additionally requires Signac. In R, install/register IRkernel according to the Jupyter installation in use. Validate the actual code cells without a notebook kernel from the repository root:

```sh
python src/validate_r_notebook.py --rscript Rscript
python src/validate_notebooks.py
```

For saved-fit Pando export use `Rscript src/pando/export_saved_fits.R`; for a selected target use `GRN_FIT_TARGETS=HES5 Rscript src/pando/refit_selected_targets.R`. Both read relative to the repository root and write into `results/generated/`. The full historical Pando pipeline is preserved under `src/legacy/`; it has machine-specific paths and is not a verified portable end-to-end entry point.

## Original data sources and availability

The input data originate from Fleck, J. S., Jansen, S. M. J., Wollny, D. et al., *Inferring and perturbing cell fate regulomes in human brain organoids*, **Nature** 621, 365–372 (2023), [doi:10.1038/s41586-022-05279-8](https://doi.org/10.1038/s41586-022-05279-8); see the article's [data availability statement](https://www.nature.com/articles/s41586-022-05279-8#data-availability).

| Data | Public source | Relevance |
|---|---|---|
| Developmental time-course scRNA-seq | [E-MTAB-12001](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12001) | Raw experiment |
| Developmental time-course scATAC-seq | [E-MTAB-11998](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11998) | Raw experiment |
| Processed objects and RNA matrices | [Zenodo v5, 15371701](https://doi.org/10.5281/zenodo.15371701) | Provides `ATAC_all_merged_srt.rds`, `RNA_all_merged_srt.rds`, the integrated object and RNA matrix archive |
| Processed fragments | [Zenodo v4, 13254037](https://doi.org/10.5281/zenodo.13254037) | Provides `timecourse_fragments.tar.gz` |
| Other processed data | [Zenodo concept DOI, 5242913](https://doi.org/10.5281/zenodo.5242913) | Data deposit cited by the paper |
| GLI3 knockout scRNA-seq / multiome | [E-MTAB-11997](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11997) / [E-MTAB-12002](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12002) | Separate perturbation experiments in the paper; not automatically the inputs to this tutorial |
| QC reference | [Developing human cortex](https://cells.ucsc.edu/cortex-dev/) | Separate reference dataset for classifier QC |

This upload directory contains selected small inputs and checkpoints under `dataset/`. Files larger than 50 MiB have been moved out of this directory; their relative paths are listed in `docs/large_files_local_backup.tsv`. Original data are available from the public links above. Locally, the removed files are preserved in the sibling `grnm-main-large-data-local/` directory. Public source objects and the processed matrices, checkpoints, candidate tables and model outputs generated by this project are distinct artifacts. The local integrated Seurat object was previously checked against the Zenodo v5 source; the extracted RNA matrix matches source metadata cell order and per-cell counts. See [data provenance](docs/01_data_provenance.md) for the exact checks and remaining mapping details.

Large raw and derived files have been removed from this upload directory and are also excluded by `.gitignore`. While the original authors' data are publicly available at the links above, this project's exact prepared tutorial bundle is not yet publicly deposited; a GitHub checkout alone can run the default result-display tutorials after installing their dependencies, but cannot perform every retraining or recomputation until that bundle or a tested preparation workflow is available. The notebooks display saved results when a large input is absent and explicitly state which computation was skipped. Full retraining and selected-target refitting require restoring the listed large inputs; raw source links alone do not supply every project-generated intermediate. The older human three-stage input and results are not part of this release. Planarian checkpoints and an example of current selected predictions are in `results/planarian/`; human comparison tables and figures are in `results/tables/` and `results/figures/`. For example, `comparison_top10_four_methods.tsv` records each gene and rank; `pando_GLI3_edges.tsv`, `pando_grn_GLI3_scores.tsv`, and `scenic_grnboost2_GLI3_before_motif.tsv` contain method-specific GLI3 result tables.

## Downloading and placing large files

Run the following commands **from the repository root**. The source-study files and this project's processed intermediates are different: downloading the former does not automatically create the latter.

### 1. Original data from the authors (optional for the default display tutorial)

The following downloads are approximately 24.3 GB and 11.3 GB. They put the author's ATAC and RNA Seurat objects in the paths used in the provenance records. `curl -C -` resumes a partial download.

```bash
mkdir -p dataset/pando/raw
curl -fL -C - 'https://zenodo.org/records/15371701/files/ATAC_all_merged_srt.rds?download=1' -o dataset/pando/raw/ATAC_all_merged_srt.rds
curl -fL -C - 'https://zenodo.org/records/15371701/files/RNA_all_merged_srt.rds?download=1' -o dataset/pando/raw/RNA_all_merged_srt.rds
```

The time-course fragments archive (about 44.1 GB) is provided separately in [Zenodo v4](https://zenodo.org/records/13254037). Download it as follows; keep it in a download directory until its internal paths are inspected.

```bash
mkdir -p dataset/downloads
curl -fL -C - 'https://zenodo.org/records/13254037/files/timecourse_fragments.tar.gz?download=1' -o dataset/downloads/timecourse_fragments.tar.gz
tar -tzf dataset/downloads/timecourse_fragments.tar.gz | head
```

The historic workflow expects extracted fragment and index files at `dataset/pando/intermediates/fragments/<batch>/fragments.tsv.gz` and `fragments.tsv.gz.tbi`. Inspect the archive structure and map its batch directories to those paths before attempting the historic preprocessing; downloading or extracting the archive alone does not run that preprocessing. Use a suitable extraction tool and sufficient disk space. The local backup contains an archive copy of approximately 22 GB, substantially smaller than the official 44.1 GB archive; treat it as incomplete and do not use it. The restore commands below exclude that copy. Download the complete official archive instead.

### 2. Project-generated inputs required for recomputation

| Task | Large input required at this exact relative path |
|---|---|
| pando+GRN score reconstruction and retraining | `dataset/scenic/expression.mtx` |
| Selected-target Pando GLM refit | `dataset/pando/intermediates/coarse/paper_model_inputs.rds` and `dataset/pando/intermediates/motifs/paper_motifs.rds` |
| SCENIC motif pruning | The two `*.rankings.feather` databases and motif annotation table under `dataset/resources/` |
| SCENIC AUCell | `dataset/scenic/rna_all_genes.mtx` |

These are prepared or derived inputs. **No public download has yet been provided for this project's exact copies**, and the historical raw-data pipeline has not been validated as a portable way to regenerate all of them. The original-study links above do not by themselves complete this step. Before release for full reproduction, the authors must deposit the required prepared files, or supply and verify a complete preparation workflow, and add its URL and instructions here.

For the existing local installation only, the removed files are already in a sibling backup with the same relative paths. To restore the `dataset/` files without overwriting existing files, review the dry run and then run:

```bash
rsync -avn --ignore-existing --exclude='/pando/intermediates/fragments/timecourse_fragments.tar.gz' ../grnm-main-large-data-local/dataset/ dataset/
rsync -av --ignore-existing --exclude='/pando/intermediates/fragments/timecourse_fragments.tar.gz' ../grnm-main-large-data-local/dataset/ dataset/
```

The sibling backup is a local path, not a public download. Its file inventory is `docs/large_files_local_backup.tsv`; Once the exact prepared inputs are available, verify their expected paths before starting an optional recomputation:

```bash
# Python model reconstruction / retraining
test -f dataset/scenic/expression.mtx && echo 'Model expression matrix available'
# Optional Pando selected-target refit
test -f dataset/pando/intermediates/coarse/paper_model_inputs.rds && test -f dataset/pando/intermediates/motifs/paper_motifs.rds && echo 'Pando refit inputs available'
```

## Interpretation

The complete default SCENIC pipeline produced 377 regulons in 9,327 cells but no GLI3 regulon at NES ≥ 3. The GLI3 Top-10 SCENIC comparison is consequently the GRNBoost2 **pre-motif** ranking. The pando+GRN model uses GRNBoost2 features; its benchmark is post hoc and includes HES4 by design. Scores and plotted edges are predictions, not proof of direct binding or causality. See [reproducibility](docs/reproducibility.md), [data provenance](docs/01_data_provenance.md) [validation](docs/validation.md), and the [reviewer audit](docs/reviewer_audit.md).
