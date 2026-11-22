options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(Seurat);library(Matrix)})
set.seed(42)
root <- Sys.getenv('PANDO_DATADIR','/home/wwd/py_code/mhr/final')
work <- file.path(root,'new_grn');out<-file.path(work,'cca')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
logmsg<-function(...){message(format(Sys.time(),'%F %T'),' ',...);flush.console()}
r <- readRDS(file.path(root,'rnaMatrix.rds'))
a <- readRDS(file.path(work,'inputs','ATAC_gene_activity.rds'))
cells <- readRDS(file.path(work,'audit','cell_names.rds'))
stopifnot(identical(colnames(r),cells[[2]]),identical(colnames(a),cells[[1]]),
          !anyDuplicated(rownames(r)),!anyDuplicated(rownames(a)))
rm <- readRDS(file.path(work,'audit','RNA_all_merged_srt.rds.metadata.rds'))[colnames(r),,drop=FALSE]
am <- readRDS(file.path(work,'audit','ATAC_all_merged_srt.rds.metadata.rds'))[colnames(a),,drop=FALSE]
valid <- function(m) !is.na(m$line)&nzchar(as.character(m$line))&!is.na(m$age)
excluded <- rbind(data.frame(cell=rownames(rm)[!valid(rm)],modality='RNA',reason='missing_line_or_age'),
                  data.frame(cell=rownames(am)[!valid(am)],modality='ATAC',reason='missing_line_or_age'))
write.csv(excluded,file.path(out,'excluded_cells.csv'),row.names=FALSE)
rm<-rm[valid(rm),,drop=FALSE];am<-am[valid(am),,drop=FALSE]
rm$match_group<-paste(rm$age,rm$line,sep='__');am$match_group<-paste(am$age,am$line,sep='__')
groups<-sort(intersect(unique(rm$match_group),unique(am$match_group)))
stopifnot(length(groups)>0)
summary<-list()
for(g in groups) {
  path<-file.path(out,paste0(g,'.tsv'))
  if(file.exists(path)) {logmsg('Resume CCA ',g);next}
  rc<-rownames(rm)[rm$match_group==g];ac<-rownames(am)[am$match_group==g]
  stopifnot(min(length(rc),length(ac))>10)
  logmsg('CCA ',g,' RNA=',length(rc),' ATAC=',length(ac))
  ro<-CreateSeuratObject(counts=r[,rc,drop=FALSE],min.cells=0,min.features=0)
  ao<-CreateSeuratObject(counts=a[,ac,drop=FALSE],min.cells=0,min.features=0)
  ro<-NormalizeData(ro,scale.factor=10000,verbose=FALSE)
  ao<-NormalizeData(ao,scale.factor=10000,verbose=FALSE)
  ro<-FindVariableFeatures(ro,nfeatures=2000,verbose=FALSE)
  ao<-FindVariableFeatures(ao,nfeatures=2000,verbose=FALSE)
  features<-SelectIntegrationFeatures(object.list=list(ro,ao),nfeatures=2000,verbose=FALSE)
  ro<-ScaleData(ro,features=features,verbose=FALSE)
  ao<-ScaleData(ao,features=features,verbose=FALSE)
  # RunCCA's 30-component default is reduced only for a group with <31 cells.
  ndim<-min(30L,min(length(rc),length(ac))-1L)
  cc<-RunCCA(ro,ao,features=features,num.cc=ndim,verbose=TRUE)
  emb<-Embeddings(cc,'cca')
  stopifnot(all(c(rc,ac)%in%rownames(emb)))
  emb<-emb[c(rc,ac),,drop=FALSE]
  frame<-data.frame(cell=rownames(emb),modality=c(rep('RNA',length(rc)),rep('ATAC',length(ac))),emb,check.names=FALSE)
  temporary<-paste0(path,'.partial')
  write.table(frame,temporary,sep='\t',row.names=FALSE,quote=FALSE);file.rename(temporary,path)
  writeLines(features,file.path(out,paste0(g,'.features.txt')))
  summary[[g]]<-data.frame(group=g,RNA=length(rc),ATAC=length(ac),CCA_dimensions=ndim,features=length(features))
  saveRDS(summary,file.path(out,'group_summary.rds'))
  rm(ro,ao,cc,emb);gc()
}
logmsg('CCA COMPLETE ',length(groups),' age-by-line groups')
