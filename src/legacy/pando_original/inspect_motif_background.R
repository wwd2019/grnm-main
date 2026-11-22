suppressPackageStartupMessages({library(Signac);library(motifmatchr)})
print(Signac::CreateMotifMatrix)
print(methods::selectMethod('matchMotifs',signature=c('PFMatrixList','GenomicRanges')))
print(methods::selectMethod('matchMotifs',signature=c('PWMatrixList','GenomicRanges')))
