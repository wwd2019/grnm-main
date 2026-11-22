root <- '/home/wwd/py_code/mhr/final/new_grn'
x <- readRDS(file.path(root,'audit','integrated_features.rds'))
r <- readRDS(file.path(root,'audit','RNA_all_merged_srt.rds.metadata.rds'))
a <- readRDS(file.path(root,'audit','ATAC_all_merged_srt.rds.metadata.rds'))
cat('RNA overlap',length(intersect(rownames(r),x$cells)),'ATAC overlap',length(intersect(rownames(a),x$cells)),'\n')
print(table(x$metadata$nowakowski_prediction,useNA='ifany'))
print(table(x$metadata$glyc_cluster,useNA='ifany'))
print(tail(x$cells))
if(file.exists(file.path(root,'reference','paper_supplementary.zip'))) {
 z <- unzip(file.path(root,'reference','paper_supplementary.zip'),list=TRUE)
 print(z)
 f <- z$Name[grepl('MOESM4_ESM.xlsx$',z$Name)]
 if(length(f)==1) {
  unzip(file.path(root,'reference','paper_supplementary.zip'),files=f,exdir=file.path(root,'reference','supplementary'))
  path<-file.path(root,'reference','supplementary',f)
  print(readxl::excel_sheets(path))
  for(s in readxl::excel_sheets(path)) print(head(readxl::read_excel(path,sheet=s),10))
 }
}
