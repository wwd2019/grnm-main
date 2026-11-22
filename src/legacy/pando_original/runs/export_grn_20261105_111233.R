options(error=function(){traceback(20);quit(status=1)})
suppressPackageStartupMessages({library(data.table);library(GenomicRanges)})
root<-'/home/wwd/py_code/mhr/final/new_grn';out<-file.path(root,'results')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
paper_features<-readRDS(file.path(root,'resources','grn_features.rds'))
annotation<-readRDS(file.path(root,'resources','EnsDb.Hsapiens.v86.genes.hg38.rds'))
unannotated<-setdiff(paper_features,annotation$gene_name)
fwrite(data.table(gene=unannotated,reason='not_in_EnsDb_Hsapiens_v86_gene_name'),
 file.path(out,'features_without_v86_annotation.csv'))
regulator_name_audit<-fread(file.path(root,'audit','TFs_without_v86_annotation.csv'))
fwrite(regulator_name_audit,file.path(out,'regulators_without_v86_target_names.csv'))
files<-list.files(file.path(root,'fits'),pattern='^[0-9]+_.*\\.rds$',full.names=TRUE)
fit<-lapply(files,readRDS)
status<-data.table(target=vapply(fit,`[[`,character(1),'target'),
 status=vapply(fit,`[[`,character(1),'status'),
 warnings=vapply(fit,function(x)paste(x$warnings,collapse='; '),character(1)))
fwrite(status,file.path(out,'target_status.csv'))
stopifnot(!any(status$status=='error'))
good<-fit[vapply(fit,function(x)x$status=='fit',logical(1))]
qc<-rbindlist(lapply(good,`[[`,'qc'),fill=TRUE)
co<-rbindlist(lapply(good,`[[`,'coefs'),fill=TRUE)
co[,inference_valid:=qc$inference_valid[match(target,qc$target)]]
co[,padj:=NA_real_]
valid<-co$inference_valid & is.finite(co$pval)&co$pval>=0&co$pval<=1&is.finite(co$estimate)
# One BH family across all estimable TF-region coefficients from every target.
co$padj[valid]<-p.adjust(co$pval[valid],method='BH')
con<-gzfile(file.path(out,'all_TF_region_gene_coefficients.tsv.gz'),'wt')
write.table(co,con,sep='\t',quote=FALSE,row.names=FALSE);close(con)
fwrite(qc,file.path(out,'model_QC.csv'))
sig<-co[!is.na(padj)&padj<0.05]
stopifnot(nrow(sig)>0,all(sig$inference_valid),all(sig$padj<0.05))
con<-gzfile(file.path(out,'significant_TF_region_gene.tsv.gz'),'wt')
write.table(sig,con,sep='\t',quote=FALSE,row.names=FALSE);close(con)
edges<-sig[,.(estimate=sum(estimate),n_regions=uniqueN(region),
 regions=paste(unique(region),collapse=';'),pval_min=min(pval),padj_min=min(padj),
 mixed_region_signs=any(estimate>0)&any(estimate<0)),by=.(tf,target)]
edges[,n_genes:=uniqueN(target),by=tf];edges[,n_tfs:=uniqueN(tf),by=target]
edges[,direction:=fifelse(estimate>0,'positive',fifelse(estimate<0,'negative','zero'))]
setorder(edges,tf,target)
fwrite(edges,file.path(out,'TF_gene_network.csv'))
module_edges<-edges[direction!='zero']
modules<-split(module_edges$target,paste(module_edges$tf,module_edges$direction,sep='__'))
saveRDS(modules,file.path(out,'gene_modules.rds'))
saveRDS(list(region_coefficients=co,TF_gene_edges=edges,model_QC=qc,gene_modules=modules),
 file.path(out,'GRN_results.rds'))
writeLines(vapply(names(modules),function(n)paste(c(n,'Pando_paper_GLM_FDR005',unique(modules[[n]])),collapse='\t'),character(1)),
 file.path(out,'gene_modules.gmt'))
region_modules<-split(sig$region,paste(sig$tf,ifelse(sig$estimate>0,'positive','negative'),sep='__'))
saveRDS(lapply(region_modules,unique),file.path(out,'regulatory_region_modules.rds'))
degree<-edges[,.(targets=uniqueN(target),positive=sum(direction=='positive'),negative=sum(direction=='negative')),by=tf][order(-targets)]
fwrite(degree,file.path(out,'TF_degree.csv'))
png(file.path(out,'QC_overview.png'),width=2200,height=1600,res=180)
par(mfrow=c(2,2),mar=c(4.5,4.5,2.5,1))
hist(qc$rsq,breaks=40,col='#3886a8',border='white',main='Gaussian GLM explained variance',xlab='R-squared')
plot(qc$nvariables,qc$rsq,pch=16,cex=.45,col=adjustcolor('#284f69',.3),xlab='Candidate interaction terms',ylab='R-squared',main='Model complexity')
top<-head(degree,20);barplot(rev(top$targets),names.arg=rev(top$tf),horiz=TRUE,las=1,cex.names=.65,col='#3c9a87',border=NA,xlab='Significant target genes',main='TF regulon sizes')
hist(sig$estimate,breaks=60,col='#c78051',border='white',main='Significant interaction coefficients',xlab='Unscaled Gaussian GLM coefficient')
dev.off()
summary<-list(attempted_targets=nrow(status),fitted_targets=nrow(qc),
 invalid_inference_models=sum(!qc$inference_valid),coefficient_tests=sum(valid),
 significant_region_edges=nrow(sig),TF_gene_edges=nrow(edges),TFs=uniqueN(edges$tf),
 target_genes=uniqueN(edges$target),gene_modules=length(modules),
 method='Original Gaussian identity GLM; unscaled TF-expression x accessibility interactions',
 FDR='BH across all estimable coefficients; threshold 0.05',
 TF_gene_pvalues='pval_min and padj_min summarize significant region-level tests, not joint TF-gene tests')
summary$paper_features<-length(paper_features)
summary$features_without_v86_annotation<-length(unannotated)
summary$models_with_aliased_terms<-sum(qc$aliased>0)
summary$regulators_without_v86_target_names<-nrow(regulator_name_audit)
summary$upstream_QC_label_field<-'WGCNAcluster'
summary$method_deviation<-'User-approved public WGCNAcluster labels replace unavailable author MainCellType labels in upstream rank-expression multinomial LASSO QC.'
jsonlite::write_json(summary,file.path(out,'summary.json'),pretty=TRUE,auto_unbox=TRUE)
print(summary)
stopifnot(!anyDuplicated(edges[,.(tf,target)]),all(edges$tf!=edges$target),
 all(co$padj[valid]+1e-15>=co$pval[valid]),sum(degree$targets)==nrow(edges),
 all(qc$nobs-qc$rank==qc$residual_df))
metacell_qc<-fread(file.path(root,'coarse','metacell_QC.csv'))
report<-c('# GRN 推断结果',
 '',
 '本次使用论文 Pando 框架，在用户提供的数据上完成推断。',
 '**经用户接受的方法差异：上游 glycolysis 质控使用公开 WGCNAcluster 标签替代原作者 MainCellType。**',
 '',
 sprintf('- 匹配聚合 metacells：%s；预测为 glycolysis 并排除：%s；保留：%s。',
 nrow(metacell_qc),sum(metacell_qc$glycolysis_excluded),sum(!metacell_qc$glycolysis_excluded)),
 sprintf('- 目标基因尝试：%s；成功拟合：%s；统计推断不可用模型：%s。',nrow(status),nrow(qc),sum(!qc$inference_valid)),
 sprintf('- 原文 GRN features：%s；EnsDb v86 无对应基因名的 features：%s（另表记录）。',length(paper_features),length(unannotated)),
 sprintf('- 含别名项的模型：%s；按 R 默认 QR 处理，仅保留 summary(glm()) 返回的系数，诊断记录在 model_QC.csv。',sum(qc$aliased>0)),
 sprintf('- 可估计区域系数检验：%s；FDR<0.05 的 TF-region-gene 关系：%s。',sum(valid),nrow(sig)),
 sprintf('- TF-gene 关系：%s；TF：%s；靶基因：%s。',nrow(edges),uniqueN(edges$tf),uniqueN(edges$target)),
 '',
 'Gaussian identity GLM，预测变量为 TF 表达×区域可及性，保留截距、不缩放交互项。所有可估计系数统一进行 BH 校正。秩亏及不可估计项保留在模型诊断中。',
 '',
 'TF_gene_network.csv 的 estimate 为该 TF-target 显著区域系数之和；pval_min/padj_min 是区域检验的摘要，不是新的联合检验。mixed_region_signs 标记区域效应方向不同的 TF-target 组合。',
 '',
 sprintf('靶基因建模要求 v86 基因名与 RNA 名称匹配；调控因子依据 RNA 表达和 motif 映射选择，其自身坐标不用于建立预测变量。%s 个候选调控因子的名称未在 v86 靶基因名称中匹配，另表列出。',nrow(regulator_name_audit)),
 '',
 '完整原始系数：all_TF_region_gene_coefficients.tsv.gz；显著关系：significant_TF_region_gene.tsv.gz；模型诊断：model_QC.csv 和 target_status.csv；正负 gene modules：gene_modules.gmt；R 对象：GRN_results.rds；总览图：QC_overview.png。',
 '',
 '输入、特征、软件环境及与论文的其他已记录差异见项目 METHODS.md；运行日志、退出码及代码快照保存在 logs/ 和 runs/。')
writeLines(report,file.path(out,'REPORT_zh.md'))
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
cat('EXPORT_COMPLETE\n')
