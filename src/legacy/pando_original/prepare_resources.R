options(error=function(){traceback(15);quit(status=1)})
suppressPackageStartupMessages({library(Pando);library(GenomicRanges);library(EnsDb.Hsapiens.v86)})
root <- Sys.getenv('PANDO_DATADIR','/home/wwd/py_code/mhr/final')
out <- file.path(root,'new_grn','resources')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
e <- new.env()
for(n in c('SCREEN.ccRE.UCSC.hg38','motifs','motif2tf')) data(list=n,package='Pando',envir=e)
stopifnot(all(c('SCREEN.ccRE.UCSC.hg38','motifs','motif2tf')%in%ls(e)))
for(n in ls(e)) {x<-get(n,e); cat(n,'class',class(x),'length',length(x),'dim',dim(x),'\n'); saveRDS(x,file.path(out,paste0(n,'.rds')))}
genes <- ensembldb::genes(EnsDb.Hsapiens.v86)
GenomeInfoDb::seqlevelsStyle(genes) <- 'UCSC'
genes <- GenomeInfoDb::keepStandardChromosomes(genes,pruning.mode='coarse')
GenomeInfoDb::genome(genes) <- 'hg38'
saveRDS(genes,file.path(out,'EnsDb.Hsapiens.v86.genes.hg38.rds'))
cat('Gene annotation',length(genes),'\n')
conserved_file <- file.path(out,'phastConsElements30way.txt.gz')
stopifnot(file.exists(conserved_file))
d <- read.delim(gzfile(conserved_file),header=FALSE,stringsAsFactors=FALSE)
stopifnot(ncol(d)>=4,all(d[[4]]>d[[3]]))
# UCSC table starts with bin; BED starts are zero-based, GRanges starts one-based.
cons <- GRanges(d[[2]],IRanges(d[[3]]+1L,d[[4]]))
GenomeInfoDb::genome(cons) <- 'hg38'
ann <- readRDS(file.path(root,'final_geneGr.rds'))
exons <- ann[ann$type=='exon']; names(exons) <- NULL
cons <- GenomicRanges::setdiff(cons,exons,ignore.strand=TRUE)
ccre <- get('SCREEN.ccRE.UCSC.hg38',e)
stopifnot(inherits(ccre,'GRanges'))
candidates <- GenomicRanges::reduce(c(cons,ccre),ignore.strand=TRUE)
saveRDS(candidates,file.path(out,'paper_candidate_prior_regions.hg38.rds'))
cat('Conserved nonexonic regions',length(cons),'cCRE',length(ccre),'union',length(candidates),'\n')
writeLines(c('conservation=UCSC_hg38_phastConsElements30way',
 'conservation_exons=excluded_using_final_geneGr.rds',
 'cCRE=Pando_bundled_SCREEN.ccRE.UCSC.hg38;historical_release_not_yet_verified',
 'motifs=Pando_bundled_motifs_and_motif2tf;historical_release_not_yet_verified',
 'gene_annotation=EnsDb.Hsapiens.v86'),file.path(out,'provenance.txt'))
