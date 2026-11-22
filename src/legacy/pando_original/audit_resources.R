options(error=function(){traceback(15);quit(status=1)})
out <- '/home/wwd/py_code/mhr/final/new_grn/audit'
dir.create(out,showWarnings=FALSE,recursive=TRUE)
sink(file.path(out,'resources.txt'),split=TRUE)
suppressPackageStartupMessages(library(Pando))
print(packageDescription('Pando')[c('Version','RemoteSha','RemoteRepo','RemoteRef')])
print(data(package='Pando')$results)
for (f in c('initiate_grn','find_motifs','infer_grn','find_modules')) {
 cat('\nFUNCTION',f,'\n'); print(get(f,asNamespace('Pando')))
 method <- paste0(f,'.Seurat')
 if(exists(method,asNamespace('Pando'),inherits=FALSE)) print(get(method,asNamespace('Pando')))
}
for(f in c('ATAC_all_merged_srt.rds.metadata.rds','RNA_all_merged_srt.rds.metadata.rds')) {
 path <- file.path(out,f)
 if(!file.exists(path)) next
 m <- readRDS(path); cat('\nMETADATA',f,'\n')
 for(n in intersect(c('age','age_integrate','line','modality','batch','organoid'),names(m))) print(table(m[[n]],useNA='ifany'))
 if(all(c('age','line')%in%names(m))) print(table(m$age,m$line,useNA='ifany'))
}
sink()
