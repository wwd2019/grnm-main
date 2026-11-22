options(error=function(){traceback(10);quit(status=1)})
root <- '/home/wwd/py_code/mhr/final/new_grn'
path <- file.path(root,'reference','supplementary','41586_2022_5279_MOESM4_ESM.xlsx')
d <- readxl::read_excel(path,col_names=FALSE)
for(i in seq_len(nrow(d))) {
 label<-as.character(d[[1]][i]);features<-unique(trimws(strsplit(as.character(d[[2]][i]),',',fixed=TRUE)[[1]]))
 cat(label,length(features),'\n')
 saveRDS(features,file.path(root,'resources',paste0(gsub('[^a-z0-9]+','_',tolower(label)),'.rds')))
}
ref <- data.table::fread(file.path(root,'resources','nowakowski','exprMatrix.tsv.gz'),nrows=2)
cat('Reference columns',ncol(ref),'\n');print(names(ref)[1:5]);print(ref[,1:5])
