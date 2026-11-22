options(error=function(){traceback(15);quit(status=1)})
suppressPackageStartupMessages({library(Signac);library(GenomicRanges);library(motifmatchr);library(BSgenome.Hsapiens.UCSC.hg38)})
root<-Sys.getenv('PANDO_DATADIR','/home/wwd/py_code/mhr/final');work<-file.path(root,'new_grn')
peaks<-StringToGRanges(readRDS(file.path(root,'atacrowname.rds')),sep=c(':','-'))
genome(peaks)<-'hg38'
prior<-readRDS(file.path(work,'resources','paper_candidate_prior_regions.hg38.rds'))
hits<-findOverlaps(peaks,prior,ignore.strand=TRUE)
regions<-GenomicRanges::pintersect(peaks[queryHits(hits)],prior[subjectHits(hits)],ignore.strand=TRUE)
reference<-BSgenome.Hsapiens.UCSC.hg38
valid<-as.character(seqnames(regions))%in%seqlevels(reference)
seqs<-getSeq(reference,regions[valid])
# Match the installed motifmatchr's own default subject-background calculation,
# applied before the downstream accessibility/domain filters.
bg<-get('get_bg',asNamespace('motifmatchr'))('subject',seqs,reference)
stopifnot(length(bg)==4,all(is.finite(bg)),all(bg>0),abs(sum(bg)-1)<1e-8)
saveRDS(list(frequencies=bg,all_candidate_regions=length(regions),
 known_reference_regions=sum(valid),p_cutoff=5e-5,
 definition='motifmatchr subject background over all peak-intersected candidate regions before model filters'),
 file.path(work,'resources','paper_motif_background.rds'))
write.table(data.frame(base=names(bg),frequency=as.numeric(bg)),
 file.path(work,'resources','paper_motif_background.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
print(bg);cat('MOTIF_BACKGROUND_COMPLETE candidates=',length(regions),'\n')
