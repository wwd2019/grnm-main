suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
root <- args[[1]]
out <- args[[2]]
audit <- file.path(root,"new_grn","audit")
cells <- readRDS(file.path(audit,"cell_names.rds"))
x <- readRDS(args[[3]])
cat("Matrix dimensions:",dim(x)," RNA audit cell count:",length(cells[[2]])," exact ID overlap:",length(intersect(colnames(x),cells[[2]])),"\n")
print(head(colnames(x),3))
print(head(cells[[2]],3))
stopifnot(inherits(x,"sparseMatrix"),setequal(colnames(x),cells[[2]]), !anyDuplicated(colnames(x)), !anyDuplicated(rownames(x)))
rna <- readRDS(file.path(audit,"RNA_all_merged_srt.rds.metadata.rds"))[colnames(x),,drop=FALSE]
atac <- readRDS(file.path(audit,"ATAC_all_merged_srt.rds.metadata.rds"))
groups <- function(m) ifelse(is.na(m$line) | !nzchar(as.character(m$line)) | is.na(m$age), NA, paste(m$age,m$line,sep="__"))
rna_group <- groups(rna)
atac_group <- groups(atac)
set.seed(42)
idx <- sort(unlist(lapply(split(which(rna_group %in% na.omit(unique(atac_group))),rna_group[rna_group %in% na.omit(unique(atac_group))]),function(i) if(length(i)>300) sample(i,300) else i)))
stopifnot(length(idx)>0)
universe <- read.delim(file.path(root,"2026_11_16_GLI3","inputs","target_scores.tsv"))$target
tfs <- unique(read.csv(file.path(root,"2026_11_16_GLI3","inputs","TF_gene_network_us.csv"))$tf)
genes <- intersect(unique(c("GLI3",universe,tfs)),rownames(x))
stopifnot("GLI3" %in% genes)
total <- Matrix::colSums(x[,idx,drop=FALSE])
stopifnot(all(total>0))
expr <- log1p(x[genes,idx,drop=FALSE] %*% Diagonal(x=10000/total))
writeMM(expr,file.path(out,"expression.mtx"))
writeLines(genes,file.path(out,"genes.txt"))
write.csv(data.frame(cell=colnames(x)[idx],group=rna_group[idx]),file.path(out,"cells.csv"),row.names=FALSE)
cat("Audited matrix:",nrow(x),"genes",ncol(x),"cells; subset:",nrow(expr),"genes",ncol(expr),"cells\n")


