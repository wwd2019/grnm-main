options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(Signac);library(Matrix);library(GenomicRanges);library(BSgenome.Hsapiens.UCSC.hg38)})
root<-Sys.getenv('PANDO_DATADIR','/home/wwd/py_code/mhr/final');work<-file.path(root,'new_grn')
out<-file.path(work,'motifs');dir.create(out,recursive=TRUE,showWarnings=FALSE)
z<-readRDS(file.path(work,'coarse','paper_model_inputs.rds'))
peaks<-StringToGRanges(rownames(z$peak_probability),sep=c(':','-'));genome(peaks)<-'hg38'
prior<-readRDS(file.path(work,'resources','paper_candidate_prior_regions.hg38.rds'))
hits<-findOverlaps(peaks,prior,ignore.strand=TRUE)
regions<-GenomicRanges::pintersect(peaks[queryHits(hits)],prior[subjectHits(hits)],ignore.strand=TRUE)
peak_index<-queryHits(hits)
ids<-GRangesToString(regions)
stopifnot(!anyDuplicated(ids),length(regions)>0)
map<-readRDS(file.path(work,'resources','motif2tf.rds'))
map<-unique(as.data.frame(map[,c('motif','tf')]))
map<-map[map$tf%in%rownames(z$rna),,drop=FALSE]
pfm<-readRDS(file.path(work,'resources','motifs.rds'))
pfm<-pfm[intersect(names(pfm),unique(map$motif))]
map<-map[map$motif%in%names(pfm),,drop=FALSE]
message('Scan ',length(regions),' candidate regions; ',length(pfm),' motifs; ',length(unique(map$tf)),' TFs')
background<-readRDS(file.path(work,'resources','paper_motif_background.rds'))
cores<-min(as.integer(Sys.getenv('GRN_MOTIF_CORES','4')),length(regions))
chunks<-split(seq_along(regions),ceiling(seq_along(regions)/ceiling(length(regions)/cores)))
# Fixed full-candidate background makes region chunks independent without
# changing motif score thresholds or matches.
parts<-parallel::mclapply(chunks,function(i) {
 Signac::CreateMotifMatrix(features=regions[i],pwm=pfm,genome=BSgenome.Hsapiens.UCSC.hg38,
  use.counts=FALSE,bg=background$frequencies,p.cutoff=background$p_cutoff)
},mc.cores=cores,mc.preschedule=TRUE)
failed<-vapply(parts,inherits,logical(1),'try-error')
if(any(failed)) stop('Motif worker failed: ',paste(parts[failed],collapse='; '))
motifs<-do.call(rbind,parts)
stopifnot(identical(rownames(motifs),ids))
rownames(motifs)<-ids
saveRDS(list(regions=regions,peak_index=peak_index,motifs=motifs,motif2tf=map,
             peaks=peaks,targets=z$targets),file.path(out,'paper_motifs.rds'),compress=FALSE)
write.csv(data.frame(region=ids,peak=rownames(z$peak_probability)[peak_index]),file.path(out,'region_to_peak.csv'),row.names=FALSE)
cat('MOTIFS_COMPLETE regions=',length(regions),' motifs=',ncol(motifs),' matches=',sum(motifs),'\n')
