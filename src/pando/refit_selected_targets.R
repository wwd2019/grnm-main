options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(Matrix);library(GenomicRanges);library(Signac)})
set.seed(42)
root<-normalizePath(Sys.getenv('GRNM_ROOT','.'),mustWork=TRUE);work<-file.path(root,'dataset','pando','intermediates')
out<-file.path(root,'results','generated','pando_refit')
required<-file.path(work,c('coarse/paper_model_inputs.rds','motifs/paper_motifs.rds','resources/EnsDb.Hsapiens.v86.genes.hg38.rds'))
missing<-required[!file.exists(required)]
if(length(missing)) stop('Cannot refit: missing inputs: ',paste(missing,collapse=', '),'. See README: Downloading and placing large files.')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
z<-readRDS(file.path(work,'coarse','paper_model_inputs.rds'))
stopifnot(identical(colnames(z$rna),colnames(z$peak_probability)))
m<-readRDS(file.path(work,'motifs','paper_motifs.rds'))
genes<-readRDS(file.path(work,'resources','EnsDb.Hsapiens.v86.genes.hg38.rds'))
targets<-intersect(z$targets,genes$gene_name)
ann<-genes[genes$gene_name%in%targets]
domains<-Signac::Extend(ann,upstream=100000,downstream=0)
h<-findOverlaps(m$regions,domains,ignore.strand=TRUE)
gene_regions<-split(queryHits(h),ann$gene_name[subjectHits(h)])
gene_regions<-lapply(gene_regions,unique)
tfs<-intersect(unique(m$motif2tf$tf),rownames(z$rna))
mt<-sparseMatrix(i=match(m$motif2tf$motif,colnames(m$motifs)),j=match(m$motif2tf$tf,tfs),x=1,
 dims=c(ncol(m$motifs),length(tfs)))
tf_regions<-m$motifs%*%mt
sm<-summary(tf_regions)
region_tfs<-split(tfs[sm$j],sm$i)
region_names<-Signac::GRangesToString(m$regions)
peak_names<-rownames(z$peak_probability)
rna<-as.matrix(z$rna)
fit_one<-function(i) {
 g<-targets[i]
 path<-file.path(out,sprintf('%05d_%s.rds',i,gsub('[^A-Za-z0-9_.-]','_',g)))
 if(file.exists(path) && readRDS(path)$status!='error') return(invisible(NULL))
 warnings<-character()
 calculate<-function() withCallingHandlers({
  ridx<-gene_regions[[g]]
  ridx<-ridx[vapply(as.character(ridx),function(k)length(region_tfs[[k]])>0,logical(1))]
  if(!length(ridx)) return(list(status='no_motif_regions',target=g))
  y<-as.numeric(rna[g,])
  if(!is.finite(sd(y))||sd(y)==0) return(list(status='constant_target',target=g))
  px<-as.matrix(Matrix::t(z$peak_probability[m$peak_index[ridx],,drop=FALSE]))
  cor_peak<-as.numeric(cor(px,y))
  keep<-is.finite(cor_peak)&abs(cor_peak)>0
  ridx<-ridx[keep];px<-px[,keep,drop=FALSE]
  if(!length(ridx)) return(list(status='no_correlating_regions',target=g))
  relevant<-setdiff(unique(unlist(region_tfs[as.character(ridx)])),g)
  if(!length(relevant)) return(list(status='no_candidate_TFs',target=g))
  tx<-t(rna[relevant,,drop=FALSE])
  cor_tf<-as.numeric(cor(tx,y));names(cor_tf)<-relevant
  relevant<-relevant[is.finite(cor_tf)&abs(cor_tf)>0.1]
  if(!length(relevant)) return(list(status='no_correlating_TFs',target=g))
  tx<-t(rna[relevant,,drop=FALSE])
  interactions<-do.call(rbind,lapply(seq_along(ridx),function(j) {
   ti<-match(intersect(region_tfs[[as.character(ridx[j])]],relevant),relevant)
   if(length(ti)) data.frame(p=j,t=ti) else NULL
  }))
  if(is.null(interactions)||!nrow(interactions)) return(list(status='no_interactions',target=g))
  colnames(px)<-paste0('peak',seq_len(ncol(px)))
  colnames(tx)<-paste0('tf',seq_len(ncol(tx)))
  data<-data.frame(response=y,px,tx,check.names=FALSE)
  # Safe column aliases change only names. The design is the authors' unchanged
  # TF-expression x region-accessibility interaction model with an intercept.
  formula<-as.formula(paste('response ~',paste(paste0('peak',interactions$p,':tf',interactions$t),collapse=' + ')))
  fit<-stats::glm(formula,data=data,family=stats::gaussian(link='identity'))
  s<-summary(fit)
  co<-as.data.frame(s$coefficients)
  co$term<-rownames(co);co<-co[co$term!='(Intercept)',,drop=FALSE]
  if(!nrow(co)) return(list(status='all_interactions_aliased',target=g))
  terms<-strsplit(co$term,':',fixed=TRUE)
  pi<-vapply(terms,function(x)as.integer(sub('peak','',x[startsWith(x,'peak')])),integer(1))
  ti<-vapply(terms,function(x)as.integer(sub('tf','',x[startsWith(x,'tf')])),integer(1))
  coefs<-data.frame(target=g,tf=relevant[ti],region=region_names[ridx[pi]],
   peak=peak_names[m$peak_index[ridx[pi]]],estimate=co[[1]],std_err=co[[2]],
   statistic=co[[3]],pval=co[[4]],tf_target_cor=unname(cor_tf[relevant[ti]]))
  qc<-data.frame(target=g,nobs=nobs(fit),nvariables=nrow(interactions),rank=fit$rank,
   residual_df=fit$df.residual,aliased=sum(is.na(coef(fit))),
   rsq=1-fit$deviance/fit$null.deviance,converged=fit$converged,
   inference_valid=fit$converged&&fit$df.residual>0&&all(is.finite(coefs$pval)))
  list(status='fit',target=g,coefs=coefs,qc=qc)
 },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
 result<-tryCatch(calculate(),error=function(e)list(status='error',target=g,error=conditionMessage(e)))
 result$warnings<-unique(warnings)
 temporary<-paste0(path,'.partial');saveRDS(result,temporary);file.rename(temporary,path)
 cat('TARGET',i,g,result$status,'\n');flush.console()
 invisible(NULL)
}
requested<-strsplit(Sys.getenv('GRN_FIT_TARGETS','HES5'),',',fixed=TRUE)[[1]]
indices<-match(requested,targets)
if(anyNA(indices)) stop('Unknown targets: ',paste(requested[is.na(indices)],collapse=', '))
for(i in indices) {
  fit_one(i)
  path<-file.path(out,sprintf('%05d_%s.rds',i,gsub('[^A-Za-z0-9_.-]','_',targets[i])))
  actual<-readRDS(path)
  if(actual$status=='error') stop('Model failed: ',actual$error)
  cat('Completed',targets[i],actual$status,'\n')
}
