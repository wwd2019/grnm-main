options(error=function(){traceback(15);quit(status=1)})
suppressPackageStartupMessages(library(Matrix))
root<-'/home/wwd/py_code/mhr/final';work<-file.path(root,'new_grn')
spec<-list(RNA=list(file=file.path(root,'rnaMatrix.rds'),meta='RNA_all_merged_srt.rds.metadata.rds',field='nCount_RNA'),
 ATAC=list(file=file.path(root,'pando_zenodo15371701_from_atac_obj','peakMatrix_recovered.rds'),meta='ATAC_all_merged_srt.rds.metadata.rds',field='nCount_peaks'))
for(label in names(spec)) {
 s<-spec[[label]];m<-readRDS(s$file);meta<-readRDS(file.path(work,'audit',s$meta))
 stopifnot(all(colnames(m)%in%rownames(meta)))
 observed<-Matrix::colSums(m);expected<-meta[colnames(m),s$field]
 delta<-max(abs(observed-expected))
 cat(label,'cells',ncol(m),'maximum_nCount_difference',delta,'\n')
 stopifnot(is.finite(delta),delta==0)
 rm(m,meta);gc()
}
cat('INPUT_TOTALS_VERIFIED\n')
