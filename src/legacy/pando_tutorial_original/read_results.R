# Read-only tutorial: inspect the completed result; never refit or overwrite it.
options(error=function(){traceback(15);quit(status=1)})
suppressPackageStartupMessages(library(data.table))
root <- '/home/wwd/py_code/mhr/final/new_grn'
grn <- readRDS(file.path(root, 'results', 'GRN_results.rds'))
required <- c('region_coefficients', 'TF_gene_edges', 'model_QC', 'gene_modules')
stopifnot(all(required %in% names(grn)))
edges <- as.data.frame(grn$TF_gene_edges)
co <- as.data.frame(grn$region_coefficients)
qc <- as.data.frame(grn$model_QC)
stopifnot(!anyDuplicated(edges[c('tf', 'target')]), all(edges$tf != edges$target),
          all(qc$inference_valid), all(qc$converged), all(qc$residual_df > 0),
          all(qc$nobs - qc$rank == qc$residual_df))
sig <- co[!is.na(co$padj) & co$padj < 0.05, ]
stopifnot(nrow(sig) > 0, all(sig$inference_valid))
cat('RDS members:', paste(names(grn), collapse=', '), '\n')
cat('Coefficient rows:', nrow(co), '\nSignificant region edges:', nrow(sig),
    '\nTF-gene edges:', nrow(edges), '\nTFs:', length(unique(edges$tf)),
    '\nTargets:', length(unique(edges$target)), '\nModules:', length(grn$gene_modules), '\n')
example <- edges[edges$tf == 'FOXG1', ]
example <- example[order(-abs(example$estimate)), ]
cat('Example FOXG1 targets (within-TF coefficient ordering):\n')
print(head(example[c('tf', 'target', 'estimate', 'n_regions', 'direction', 'mixed_region_signs')]))
cat('TUTORIAL_INSPECT_COMPLETE\n')
