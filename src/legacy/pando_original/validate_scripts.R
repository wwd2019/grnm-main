root<-'/home/wwd/py_code/mhr/final/new_grn'
files<-file.path(root,c('prepare_inputs.R','prepare_cca.R','train_glycolysis_qc.R',
 'coarse_grain.R','scan_motifs.R','fit_grn.R','export_grn.R'))
for(f in files) {parse(file=f);cat('PARSE_OK',basename(f),'\n')}
stopifnot(all(file.exists(file.path(root,'resources',c('grn_features.rds',
 'full_timecourse_highly_variable_genes.rds','paper_candidate_prior_regions.hg38.rds',
 'EnsDb.Hsapiens.v86.genes.hg38.rds','motifs.rds','motif2tf.rds')))))
cat('RESOURCE_PATHS_OK\n')
