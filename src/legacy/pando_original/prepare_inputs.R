options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(Seurat);library(Signac);library(Matrix);library(GenomicRanges)})
set.seed(42)
root <- Sys.getenv('PANDO_DATADIR','/home/wwd/py_code/mhr/final')
work <- file.path(root,'new_grn'); out <- file.path(work,'inputs')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
logmsg <- function(...) {message(format(Sys.time(),'%F %T'),' ',...);flush.console()}
meta <- readRDS(file.path(work,'audit','ATAC_all_merged_srt.rds.metadata.rds'))
expected <- readRDS(file.path(work,'audit','cell_names.rds'))[[1]]
# This checkpoint was recovered from ATAC_all_merged_srt.rds in the prior input audit.
# It is used only when its complete row/column identities match the supplied inputs.
peakfile <- file.path(root,'pando_zenodo15371701_from_atac_obj','peakMatrix_recovered.rds')
p <- readRDS(peakfile)
stopifnot(inherits(p,'dgCMatrix'),identical(colnames(p),expected),
          identical(rownames(p),readRDS(file.path(root,'atacrowname.rds'))))
meta <- meta[colnames(p),,drop=FALSE]
ann <- readRDS(file.path(root,'final_geneGr.rds'))
names(ann) <- NULL
pr <- StringToGRanges(rownames(p),sep=c(':','-'))
genome(pr) <- 'hg38'
activity <- list(); qc <- list()
batches <- unique(as.character(meta$batch))
requested <- Sys.getenv('GRN_ACTIVITY_BATCHES','')
if(nzchar(requested)) batches <- intersect(batches,strsplit(requested,',',fixed=TRUE)[[1]])
if(nzchar(requested)) future::plan(future::multicore,workers=2)
for(batch in batches) {
  checkpoint <- file.path(out,paste0(batch,'.GeneActivity.rds'))
  chosen <- rownames(meta)[as.character(meta$batch)==batch]
  fragment_file <- file.path(work,'fragments',batch,'fragments.tsv.gz')
  stopifnot(file.exists(fragment_file),file.exists(paste0(fragment_file,'.tbi')),
            all(startsWith(chosen,paste0(batch,'_'))))
  if(file.exists(checkpoint)) {logmsg('Resume ',batch);activity[[batch]]<-readRDS(checkpoint);next}
  raw_barcodes <- substring(chosen,nchar(batch)+2L)
  stopifnot(!anyDuplicated(raw_barcodes))
  cell_map <- setNames(raw_barcodes,chosen)
  logmsg('GeneActivity ',batch,' cells=',length(chosen))
  fragment <- CreateFragmentObject(path=fragment_file,cells=cell_map,validate.fragments=TRUE)
  assay <- CreateChromatinAssay(counts=p[,chosen,drop=FALSE],ranges=pr,
      fragments=fragment,annotation=ann,min.cells=0,min.features=0,genome='hg38')
  object <- CreateSeuratObject(counts=assay,assay='peaks',meta.data=meta[chosen,,drop=FALSE])
  # Original paper: Signac::GeneActivity defaults, then LogNormalize(scale.factor=10000).
  a <- GeneActivity(object,assay='peaks',verbose=TRUE)
  stopifnot(identical(colnames(a),chosen),all(Matrix::colSums(a)>0))
  saveRDS(a,checkpoint,compress=FALSE);activity[[batch]]<-a
  qc[[batch]]<-data.frame(batch=batch,cells=length(chosen),genes=nrow(a),
                          fragment_file=fragment_file,min_gene_activity=min(Matrix::colSums(a)))
  rm(assay,object,a,fragment);gc()
}
if(nzchar(requested)) {logmsg('BATCH WORKER COMPLETE');quit(status=0)}
common <- Reduce(intersect,lapply(activity,rownames))
a <- do.call(cbind,lapply(activity,function(x)x[common,,drop=FALSE]))
a <- a[,expected,drop=FALSE]
saveRDS(a,file.path(out,'ATAC_gene_activity.rds'),compress=FALSE)
qc <- lapply(names(activity),function(batch) data.frame(batch=batch,cells=ncol(activity[[batch]]),
 genes=nrow(activity[[batch]]),fragment_file=file.path(work,'fragments',batch,'fragments.tsv.gz'),
 min_gene_activity=min(Matrix::colSums(activity[[batch]]))))
write.csv(do.call(rbind,qc),file.path(out,'GeneActivity_QC.csv'),row.names=FALSE)
writeLines(c(paste0('peak_source=',peakfile),'peak_identity=full_row_and_column_match',
 'activity=Signac::GeneActivity_defaults_from_original_fragments',
 'RNA_source=rnaMatrix.rds_measured_RNA','expression_transfer=none'),file.path(out,'provenance.txt'))
logmsg('PREPARE_INPUTS COMPLETE ',nrow(a),' genes x ',ncol(a),' cells')
