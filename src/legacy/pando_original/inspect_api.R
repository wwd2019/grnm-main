suppressPackageStartupMessages({library(Pando);library(Signac)})
out <- '/home/wwd/py_code/mhr/final/new_grn/audit/api.txt'
sink(out,split=TRUE)
for(n in c('infer_grn.GRNData','find_motifs.GRNData','find_modules.GRNData','fit_glm','select_regions','select_genes','get_model_data')) {
 cat('\nPANDO',n,'\n'); if(exists(n,asNamespace('Pando'),inherits=FALSE)) print(get(n,asNamespace('Pando')))
}
cat('\nSIGNAC GeneActivity\n'); print(Signac::GeneActivity)
cat('\nSIGNAC CollapseToLongestTranscript\n'); print(Signac:::CollapseToLongestTranscript)
e<-new.env();data('motif2tf',package='Pando',envir=e);print(head(e$motif2tf)); print(colnames(e$motif2tf))
sink()
