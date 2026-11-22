suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
source <- normalizePath(args[1]); out <- normalizePath(args[2]); raw <- normalizePath(args[3])
cells <- read.csv(file.path(source,'cells.csv'),stringsAsFactors=FALSE)$cell
x <- readRDS(raw); idx <- match(cells,colnames(x)); stopifnot(!anyNA(idx),!anyDuplicated(rownames(x)),!anyDuplicated(cells))
x <- x[,idx,drop=FALSE]
writeMM(x,file.path(out,'rna_all_genes.mtx'))
writeLines(rownames(x),file.path(out,'rna_all_genes.txt'))
writeLines(cells,file.path(out,'rna_cells.txt'))
cat('Full RNA count matrix:',nrow(x),'genes x',ncol(x),'cells; nonzero:',length(x@x),'\n')
