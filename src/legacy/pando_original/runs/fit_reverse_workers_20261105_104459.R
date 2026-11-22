options(error=function(){traceback(20);quit(status=1)})
parent_source<-'/home/wwd/py_code/mhr/final/new_grn/runs/fit_grn_20261105_102736.R'
code<-parse(parent_source)
# Reuse the exact running model implementation, stopping before its scheduling.
found<-FALSE
for(expression in code) {
 if(is.call(expression)&&identical(expression[[1]],as.name('<-'))&&identical(expression[[2]],as.name('cores'))) {
  found<-TRUE;break
 }
 eval(expression,envir=.GlobalEnv)
}
stopifnot(found,exists('fit_one'),exists('targets'))
publish_dir<-out
out<-file.path(work,'fits_reverse');dir.create(out,recursive=TRUE,showWarnings=FALSE)
base_fit_one<-fit_one
run_one<-function(i) {
 name<-sprintf('%05d_%s.rds',i,gsub('[^A-Za-z0-9_.-]','_',targets[i]))
 published<-file.path(publish_dir,name)
 if(file.exists(published)&&readRDS(published)$status!='error') return(invisible(NULL))
 base_fit_one(i)
 local<-file.path(out,name)
 if(file.exists(local)&&!file.exists(published)) {
  temporary<-paste0(published,'.reverse_',Sys.getpid(),'.partial')
  stopifnot(file.copy(local,temporary,overwrite=TRUE))
  stopifnot(file.rename(temporary,published))
 }
 invisible(NULL)
}
if(requireNamespace('RhpcBLASctl',quietly=TRUE)) RhpcBLASctl::blas_set_num_threads(1)
cores_reverse<-as.integer(Sys.getenv('GRN_REVERSE_CORES','16'))
message('Same model source: ',parent_source,'; reverse-order independent target workers=',cores_reverse)
invisible(parallel::mclapply(rev(seq_along(targets)),run_one,
 mc.cores=cores_reverse,mc.preschedule=FALSE,mc.set.seed=TRUE))
files<-list.files(publish_dir,pattern='^[0-9]+_.*\\.rds$',full.names=TRUE)
stopifnot(length(files)==length(targets))
statuses<-vapply(files,function(f)readRDS(f)$status,character(1))
print(table(statuses))
stopifnot(!any(statuses=='error'))
cat('REVERSE_WORKERS_COMPLETE\n')
