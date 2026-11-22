options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(Seurat);library(Signac);library(Matrix);library(GenomicRanges);library(igraph);library(glmnet)})
set.seed(42)
root<-Sys.getenv('PANDO_DATADIR','/home/wwd/py_code/mhr/final');work<-file.path(root,'new_grn')
out<-file.path(work,'coarse');dir.create(out,recursive=TRUE,showWarnings=FALSE)
logmsg<-function(...){message(format(Sys.time(),'%F %T'),' ',...);flush.console()}
pairs<-read.delim(file.path(work,'matches','all_pairs.tsv'),stringsAsFactors=FALSE)
am<-readRDS(file.path(work,'audit','ATAC_all_merged_srt.rds.metadata.rds'))
rm<-readRDS(file.path(work,'audit','RNA_all_merged_srt.rds.metadata.rds'))
stopifnot(all(pairs$RNA%in%rownames(rm)),all(pairs$ATAC%in%rownames(am)),
 all(rm[pairs$RNA,'age']==am[pairs$ATAC,'age']),all(rm[pairs$RNA,'line']==am[pairs$ATAC,'line']))
pairs$metacell<-character(nrow(pairs))
for(g in unique(pairs$group)) {
 idx<-which(pairs$group==g)
 # The authors' uniform-capacity MCMF pools the larger modality around each smaller-modality cell.
 small<-if(length(unique(pairs$RNA[idx]))>=length(unique(pairs$ATAC[idx]))) pairs$ATAC[idx] else pairs$RNA[idx]
 pairs$metacell[idx]<-paste(g,small,sep='::')
}
metas<-unique(pairs$metacell)
weights<-function(column,all_cells) {
 d<-unique(pairs[,c(column,'metacell')]);j<-match(d$metacell,metas)
 sparseMatrix(i=match(d[[column]],all_cells),j=j,x=1/tabulate(j,nbins=length(metas))[j],
              dims=c(length(all_cells),length(metas)),dimnames=list(all_cells,metas))
}
write.table(pairs,file.path(out,'metacell_membership.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
checkpoint<-file.path(out,'matched_metacells.rds')
if(file.exists(checkpoint)) {
 cached<-readRDS(checkpoint);expr<-cached$expression;meta<-cached$metadata
 stopifnot(identical(colnames(expr),metas));rm(cached);gc()
} else {
 r<-readRDS(file.path(root,'rnaMatrix.rds'))
 r_cells<-colnames(r)
 r<-r%*%Diagonal(x=10000/Matrix::colSums(r));colnames(r)<-r_cells;r@x<-log1p(r@x)
 # Preserve measured expression: average log-normalized RNA over matched RNA cells.
 expr<-as(r%*%weights('RNA',colnames(r)),'dgCMatrix');rm(r);gc()
 meta<-am[pairs$ATAC[match(metas,pairs$metacell)],c('age','line'),drop=FALSE];rownames(meta)<-metas
 saveRDS(list(expression=expr,metadata=meta),checkpoint,compress=FALSE)
}
if(Sys.getenv('GRN_PREPARE_ONLY','0')=='1') {
 logmsg('MATCHED_METACELLS_COMPLETE ',ncol(expr),' metacells from ',nrow(pairs),' pairs')
 quit(status=0)
}
qc<-readRDS(file.path(work,'qc','nowakowski_rank_lasso.rds'))
stopifnot(identical(qc$label_field,'MainCellType') ||
 (identical(qc$label_field,'WGCNAcluster') && isTRUE(qc$approved_public_label)))
stopifnot(all(qc$genes%in%rownames(expr)))
pred<-character(ncol(expr))
for(start in seq.int(1,ncol(expr),by=512)) {
 idx<-start:min(start+511L,ncol(expr))
 mat<-as.matrix(expr[qc$genes,idx,drop=FALSE])
 ranks<-t(matrixStats::colRanks(mat,ties.method='average',preserveShape=TRUE))
 pred[idx]<-as.character(predict(qc$model,newx=ranks,s='lambda.1se',type='class'))
}
meta$nowakowski_prediction<-pred
meta$qc_label_field<-qc$label_field
meta$glycolysis_excluded<-pred%in%qc$glycolysis_classes
write.csv(meta,file.path(out,'metacell_QC.csv'))
keep<-!meta$glycolysis_excluded
logmsg('Exclude glycolysis metacells ',sum(!keep),' / ',length(keep))
expr<-expr[,keep,drop=FALSE];meta<-meta[keep,,drop=FALSE];metas<-metas[keep]
targets<-intersect(readRDS(file.path(work,'resources','grn_features.rds')),rownames(expr))
features<-intersect(readRDS(file.path(work,'resources','full_timecourse_highly_variable_genes.rds')),rownames(expr))
stopifnot(length(features)>500,length(targets)>1000)
obj<-CreateSeuratObject(counts=expr,meta.data=meta,min.cells=0,min.features=0)
obj[['RNA']]<-SetAssayData(obj[['RNA']],layer='data',new.data=expr)
VariableFeatures(obj)<-features
obj<-ScaleData(obj,features=features,verbose=FALSE)
obj<-RunPCA(obj,features=features,npcs=30,verbose=FALSE)
# Source the authors' CSS implementation unchanged; its unused optional dependencies
# are not required for corr_ztransform/Spearman mode.
for(f in list.files(file.path(work,'reference','simspec-master','R'),full.names=TRUE)) source(f)
css<-cluster_sim_spectrum.default(expr[features,,drop=FALSE],labels=meta$line,
 dr=Embeddings(obj,'pca')[,1:20,drop=FALSE],use_fast_rank=FALSE,verbose=TRUE)
css<-as.matrix(css)
if(!is.null(rownames(css))) css<-css[colnames(expr),,drop=FALSE]
stopifnot(nrow(css)==ncol(expr),all(is.finite(css)))
pcs<-prcomp(css,center=TRUE,scale.=FALSE,rank.=20)$x[,1:20,drop=FALSE]
rownames(pcs)<-colnames(expr)
saveRDS(list(css=css,css_pcs=pcs,metadata=meta),file.path(out,'metacell_embedding.rds'))
# This follows the authors' pseudocells.R: seed Bernoulli(0.3), k=10,
# igraph ego territories, random assignment of multiply covered cells, singleton fallback.
knn<-FindNeighbors(pcs,compute.SNN=FALSE,k.param=10,verbose=FALSE)$nn
seeds<-runif(ncol(knn))<=0.3
g<-igraph::graph_from_adjacency_matrix(knn)
territories<-lapply(igraph::ego(g,order=1,nodes=colnames(knn)[seeds]),igraph::as_ids)
names(territories)<-colnames(knn)[seeds]
inverse<-split(rep(names(territories),lengths(territories)),unlist(territories))
assignment<-vapply(colnames(knn),function(cell) if(is.null(inverse[[cell]])) cell else sample(inverse[[cell]],1),character(1))
pseudo<-unique(assignment)
j<-match(assignment,pseudo)
W<-sparseMatrix(i=seq_along(j),j=j,x=1/tabulate(j,nbins=length(pseudo))[j],
 dims=c(length(j),length(pseudo)),dimnames=list(names(assignment),pseudo))
rx<-as(expr%*%W,'dgCMatrix');rm(expr,obj,knn,g,css,pcs);gc()
write.table(data.frame(metacell=names(assignment),pseudocell=unname(assignment)),
 file.path(out,'pseudocell_membership.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
p<-readRDS(file.path(root,'pando_zenodo15371701_from_atac_obj','peakMatrix_recovered.rds'))
allmetas<-unique(pairs$metacell)
# Build ATAC weights for the full matched set before applying metacell QC.
metas<-allmetas
px<-as(p%*%weights('ATAC',colnames(p)),'dgCMatrix');rm(p);gc()
px<-Matrix::drop0(px[,rownames(W),drop=FALSE])
px_bin<-px;px_bin@x[]<-1
prob<-as(px_bin%*%W,'dgCMatrix');rm(px_bin)
counts<-as(px%*%W,'dgCMatrix');rm(px);gc()
po<-CreateSeuratObject(counts=CreateAssayObject(counts=counts),assay='peaks',min.cells=0,min.features=0)
VariableFeatures(po)<-rownames(counts)[Matrix::rowSums(counts)>0]
po<-RunTFIDF(po,assay='peaks',verbose=FALSE)
po<-RunSVD(po,assay='peaks',n=30,verbose=FALSE)
po<-FindNeighbors(po,reduction='lsi',dims=1:20,verbose=FALSE)
po<-FindClusters(po,resolution=100,algorithm=1,random.seed=42,verbose=FALSE)
clusters<-as.character(Idents(po));names(clusters)<-colnames(po)
levels<-unique(clusters);j<-match(clusters,levels)
C<-sparseMatrix(i=seq_along(j),j=j,x=1/tabulate(j,nbins=length(levels))[j],
 dims=c(length(j),length(levels)),dimnames=list(names(clusters),levels))
cluster_prob<-as(prob%*%C,'dgCMatrix')
# The authors' fit_multiome_glm.R retains peaks with max cluster accessibility >0.1.
# Apply the same filter before expanding cluster vectors back to pseudocells.
good<-sparseMatrixStats::rowMaxs(cluster_prob)>0.1
genes<-readRDS(file.path(work,'resources','EnsDb.Hsapiens.v86.genes.hg38.rds'))
domains<-Signac::Extend(genes[genes$gene_name%in%targets],upstream=100000,downstream=0)
ranges<-StringToGRanges(rownames(cluster_prob),sep=c(':','-'))
near<-unique(queryHits(findOverlaps(ranges,domains,ignore.strand=TRUE)))
good<-good & seq_len(nrow(cluster_prob))%in%near
cluster_prob<-cluster_prob[good,,drop=FALSE]
smoothed<-cluster_prob[,clusters,drop=FALSE];colnames(smoothed)<-colnames(prob)
write.csv(data.frame(pseudocell=names(clusters),accessibility_cluster=clusters),file.path(out,'accessibility_clusters.csv'),row.names=FALSE)
saveRDS(list(rna=rx[targets,,drop=FALSE],peak_probability=smoothed,
             cluster_probability=cluster_prob,targets=targets,clusters=clusters),
 file.path(out,'paper_model_inputs.rds'),compress=FALSE)
logmsg('COARSE_GRAIN COMPLETE pseudocells=',ncol(rx),' accessibility_clusters=',length(levels),' features=',length(targets))
