suppressPackageStartupMessages(library(GenomicRanges))
root<-'/home/wwd/py_code/mhr/final/new_grn'
m<-readRDS(file.path(root,'motifs','paper_motifs.rds'))
genes<-readRDS(file.path(root,'resources','EnsDb.Hsapiens.v86.genes.hg38.rds'))
missing<-setdiff(unique(m$motif2tf$tf),genes$gene_name)
cat('TFs absent from v86 annotation:',length(missing),'\n')
print(missing)
write.csv(data.frame(tf=missing),file.path(root,'audit','TFs_without_v86_annotation.csv'),row.names=FALSE)
