options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(glmnet);library(matrixStats);library(data.table)})
set.seed(42)
root<-'/home/wwd/py_code/mhr/final';work<-file.path(root,'new_grn');out<-file.path(work,'qc')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
label_field<-Sys.getenv('GRN_QC_LABEL_FIELD','WGCNAcluster')
reference_file<-Sys.getenv('NOWAKOWSKI_REFERENCE_RDS','')
approved_public_label<-identical(label_field,'WGCNAcluster')
if(approved_public_label) {
 reference_file<-file.path(work,'resources','nowakowski','exprMatrix.tsv.gz')
 d<-fread(reference_file)
 symbols<-sub('^.*\\|','',d[[1]])
 mat<-as.matrix(d[,-1]);rownames(mat)<-symbols;rm(d);gc()
 mat<-mat[!duplicated(rownames(mat)),,drop=FALSE]
 meta<-read.delim(file.path(work,'resources','nowakowski','meta.tsv'),check.names=FALSE)
 rownames(meta)<-meta$Cell
 writeLines(c('User explicitly accepted public WGCNAcluster labels in this chat.',
  'Same rank-expression multinomial LASSO; alpha=1.',
  'Deviation: WGCNAcluster replaces the unavailable author MainCellType annotation.'),
  file.path(out,'WGCNAcluster_authorization.txt'))
} else {
 stopifnot(identical(label_field,'MainCellType'),nzchar(reference_file),file.exists(reference_file))
 reference<-readRDS(reference_file)
 stopifnot(is.list(reference),all(c('expr','meta')%in%names(reference)))
 mat<-as.matrix(reference$expr);meta<-as.data.frame(reference$meta)
}
stopifnot(label_field%in%colnames(meta))
stopifnot(!anyDuplicated(rownames(mat)),!anyDuplicated(rownames(meta)),all(colnames(mat)%in%rownames(meta)))
measured<-readRDS(file.path(root,'rnaMatrix.rds'))
genes<-intersect(rownames(mat),rownames(measured));rm(measured);gc()
stopifnot(length(genes)>10000)
mat<-mat[genes,,drop=FALSE]
y<-factor(meta[colnames(mat),label_field])
glycolysis_classes<-levels(y)[tolower(levels(y))%in%c('glyc','glycolysis')]
stopifnot(!anyNA(y),length(glycolysis_classes)>0)
print(table(y))
# The paper uses within-cell gene expression ranks and multinomial lasso (alpha=1).
x<-t(matrixStats::colRanks(mat,ties.method='average',preserveShape=TRUE))
colnames(x)<-genes;rownames(x)<-colnames(mat);rm(mat);gc()
message('Train rank-expression multinomial lasso: ',nrow(x),' cells, ',ncol(x),' genes')
foldid<-sample(rep(seq_len(10),length.out=nrow(x)))
write.csv(data.frame(cell=rownames(x),label=as.character(y),fold=foldid),file.path(out,'reference_CV_folds.csv'),row.names=FALSE)
cores<-as.integer(Sys.getenv('GRN_QC_CORES','10'))
doParallel::registerDoParallel(cores=cores)
fit<-cv.glmnet(x,y,family='multinomial',alpha=1,foldid=foldid,nfolds=10,parallel=TRUE,trace.it=1)
doParallel::stopImplicitCluster()
saveRDS(list(model=fit,genes=genes,classes=levels(y),label_field=label_field,
             approved_public_label=approved_public_label,foldid=foldid,
             glycolysis_classes=glycolysis_classes,reference_file=reference_file,
             rank_definition='within_cell_average_tie_ranks_over_shared_genes',
             lambda='lambda.1se',seed=42),file.path(out,'nowakowski_rank_lasso.rds'))
cat('QC_CLASSIFIER_COMPLETE lambda.min=',fit$lambda.min,' lambda.1se=',fit$lambda.1se,'\n')
