options(error=function(){traceback(15);quit(status=1)})
suppressPackageStartupMessages(library(data.table))
out<-'/home/wwd/py_code/mhr/final/new_grn/results'
x<-readRDS(file.path(out,'GRN_results.rds'))
qc<-x$model_QC;sig<-x$region_coefficients[!is.na(padj)&padj<0.05]
degree<-fread(file.path(out,'TF_degree.csv'))
png(file.path(out,'QC_overview.updated.png'),width=2200,height=1600,res=180)
par(mfrow=c(2,2),mar=c(4.5,4.5,2.5,1))
hist(qc$rsq,breaks=40,col='#3886a8',border='white',main='Gaussian GLM explained variance',xlab='R-squared')
plot(qc$nvariables,qc$rsq,pch=16,cex=.45,col=adjustcolor('#284f69',.3),xlab='Candidate interaction terms',ylab='R-squared',main='Model complexity')
top<-head(degree,20);barplot(rev(top$targets),names.arg=rev(top$tf),horiz=TRUE,las=1,cex.names=.65,col='#3c9a87',border=NA,xlab='Significant target genes',main='TF regulon sizes')
hist(sign(sig$estimate)*log10(1+abs(sig$estimate)),breaks=60,col='#c78051',border='white',main='Significant interaction coefficients',xlab='Signed log10(1 + |coefficient|)')
dev.off()
stopifnot(file.rename(file.path(out,'QC_overview.updated.png'),file.path(out,'QC_overview.png')))
report<-file.path(out,'REPORT_zh.md')
text<-readLines(report,warn=FALSE)
if(!any(grepl('signed log10',text,fixed=TRUE))) writeLines(c(text,'',
 '总览图中系数直方图的横轴使用 signed log10(1+|coefficient|)，包含全部显著系数；原始未缩放系数保存在结果表中。'),report)
cat('QC_PLOT_COMPLETE\n')
