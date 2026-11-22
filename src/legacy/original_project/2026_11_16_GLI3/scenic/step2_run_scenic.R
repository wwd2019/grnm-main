args <- commandArgs(trailingOnly=TRUE)
if(length(args)<5) stop("Usage: Rscript step2_run_scenic.R SCENIC_DIR RUN_DIR MOTIF_ANNOT CORES RANKING_DB [RANKING_DB ...]")
.libPaths(c(file.path(args[[1]], "Rlib"), .libPaths()))
packages <- c("Matrix","GENIE3","RcisTarget","AUCell")
missing <- packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) stop("Missing R packages: ",paste(missing,collapse=", "))
source_dir <- normalizePath(args[[1]],mustWork=TRUE)
dir.create(args[[2]],recursive=TRUE,showWarnings=FALSE)
run <- normalizePath(args[[2]],mustWork=TRUE)
annotation <- normalizePath(args[[3]],mustWork=TRUE)
cores <- as.integer(args[[4]])
dbs <- normalizePath(args[5:length(args)],mustWork=TRUE)
stopifnot(is.finite(cores),cores>=1,length(unique(dirname(dbs)))==1)
setwd(run)
if(file.exists("gli3_scenic.tsv")) stop("Use a new run directory to preserve existing results")
set.seed(42)
writeLines(c(capture.output(sessionInfo()),"Ranking resources:",dbs,annotation),"environment.txt")
rna <- Matrix::readMM(file.path(source_dir,"expression.mtx"))
genes <- readLines(file.path(source_dir,"genes.txt"))
cells <- read.csv(file.path(source_dir,"cells.csv"))
stopifnot(nrow(rna)==length(genes),ncol(rna)==nrow(cells),!anyDuplicated(genes))
dimnames(rna) <- list(genes,cells$cell)
if(requireNamespace("SCENIC",quietly=TRUE)){
  library(SCENIC)
}else{
  src <- list.dirs(file.path(source_dir,"vendor"),recursive=FALSE)
  stopifnot(length(src)==1L)
  pkgload::load_all(src[[1]],export_all=FALSE)
}
opts <- initializeScenic(org="hgnc",dbDir=dirname(dbs[[1]]),dbs=setNames(basename(dbs),basename(dbs)),
                         datasetTitle="Human organoid developmental time course",nCores=cores, dbIndexCol="motifs", dbAnnotFiles=annotation)
opts@settings$db_annotFiles <- annotation
opts@settings$seed <- 42
expr <- as.matrix(rna)
expr <- expr[apply(expr,1,var)>0,,drop=FALSE]
stopifnot("GLI3" %in% rownames(expr))
# Values are already log1p(CP10K). Do not log-transform them again.
runCorrelation(expr,opts)
runGenie3(expr,opts,nTrees=100,nParts=30)
opts <- runSCENIC_1_coexNetwork2modules(opts)
opts <- runSCENIC_2_createRegulons(opts,dbIndexCol="motifs")
targets <- loadInt(opts,"regulonTargetsInfo")
stopifnot(all(c("TF","gene","CoexWeight","highConfAnnot") %in% names(targets)))
gli3 <- targets[targets$TF=="GLI3" & as.character(targets$highConfAnnot)=="TRUE" &
                targets$gene!="GLI3" & is.finite(as.numeric(targets$CoexWeight)),]
write.table(gli3,"gli3_scenic.tsv",sep="\t",row.names=FALSE,quote=FALSE)
saveRDS(opts,"scenicOptions.rds")
cat("SCENIC high-confidence GLI3 targets:",nrow(gli3),"\n")




