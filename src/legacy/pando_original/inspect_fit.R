suppressPackageStartupMessages(library(Pando))
root<-'/home/wwd/py_code/mhr/final/new_grn'
sink(file.path(root,'audit','fit_api.txt'),split=TRUE)
for(n in ls(asNamespace('Pando'),all.names=TRUE)) {
 if(grepl('fit_grn_models|prepare_model|model_input|fit_single|find_modules.Network|find_regions',n)) {
  cat('\n',n,'\n');print(get(n,asNamespace('Pando')))
 }
}
sink()
