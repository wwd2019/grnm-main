root<-'/home/wwd/py_code/mhr/final/new_grn'
for(g in c('GLI3','FOXG1','HES5')) {
 files<-list.files(file.path(root,'fits'),pattern=paste0('_',g,'\\.rds$'),full.names=TRUE)
 if(!length(files)) next
 x<-readRDS(files[1]);cat('\n',g,x$status,'\n')
 if(x$status=='fit') {print(x$qc);cat('max_abs_beta',max(abs(x$coefs$estimate)),'\n')}
}
