# Planarian model outputs

The planarian analysis uses SMED/SMESG gene identifiers and is separate from the human brain organoid analysis.

- `phase1_output/gene_z.npz`: saved phase 1 gene embeddings.
- `phase2_output/`: saved phase 2 RNA-to-embedding outputs and trained model.
- `phase3_output/edge_predictor.pt`: saved seven-class classifier weights. `Hes1_top10.tsv` and `figures/Hes1_top10.png` present a selected-regulator demonstration generated from those weights and the saved phase 2 embeddings by `python src/legacy/planarian_selected_predictions.py --output results/generated/planarian_Hes1_top10.tsv`.

The example ranks all available targets for `SMED30006440`, annotated Hes1, by absolute signed model score. Scores are class × softmax probability and are not calibrated probabilities. When the highest model class is zero, the strongest nonzero class is used as a fallback, recorded in `zero_class_fallback`. These are predicted candidates, not experimentally established regulatory interactions. The original seven-class classifier weights predate the class-index correction; the displayed selected predictions were generated using the current corrected decoding logic. No planarian Pando/SCENIC baseline is available for this analysis.

The original input data and source paths, optional planarian training workflow, and data availability limitations are explained in `notebooks/planarian/01_planarian_legacy.ipynb` and `dataset/legacy/planarian/README.md`. The complete original human three-stage results are not included.
