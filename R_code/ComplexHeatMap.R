# library(ComplexHeatmap)
# library(stringr)
# library(Seurat)
# library(ggplot2)
#
# smallmatrix = function(matrix_counts,num,ordercellName,careGene){
#   rna_counts = matrix_counts[rownames(matrix_counts) %in% caregene,]
#   matrix_rownum = length(colnames(rna_counts))%/% num + 1
#   namematrix <- matrix(rep(NA,time = num * matrix_rownum), ncol = num)
#   start = seq(from = 1,to = dim(rna_counts)[2],by = num)
#   rownum <- 1
#   for(st in start){
#     end <- st + num -1
#     namematrix[rownum,] <-ordercellName[st:end]
#     rownum <- rownum + 1
#   }
#   result <- matrix(rep(0,time =dim(rna_counts)[1] * dim(namematrix)[1]),ncol = dim(namematrix)[1])
#   for(i in 1:dim(namematrix)[1]){
#     result[,i] = apply(rna_counts[,colnames(rna_counts)%in% namematrix[i,]],1,mean)
#   }
#   rownames(result) <- rownames(rna_counts)
#   return(result)
# }
# setColnames = function(matrix_counts){
#   name = c()
#   for ( i in 1:dim(matrix_counts)[2]){
#     name =  c(name,str_c(c("pseudotime"),i))
#   }
#   colnames(matrix_counts) <- name
#   return(matrix_counts)
# }
# orderGene = function(matrix_counts){
#   maxresult =c()
#   for (i in 1:dim(matrix_counts)[1]){
#     row = matrix_counts[i,]
#     maxresult =c(maxresult,as.numeric(which(row == max(row))))
#   }
#   return(matrix_counts[order(maxresult),])
# }
#
#
#
# pseudotime <- readRDS("D:/GRN/final/data/newpseudotime.rds")
# rna_counts <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
# ordercellName <- pseudotime$cellName[order(pseudotime$pseudotime)]
# caregene <- rownames(rna_counts)[!str_starts(rownames(rna_counts),"SMESG")]
# smallrnaMatrix <- smallmatrix(matrix_counts = rna_counts,num = 100,ordercellName = ordercellName,careGene = caregene)
# smallrnaMatrix <- setColnames(smallrnaMatrix)
# pbmc <- CreateSeuratObject(counts = smallrnaMatrix)
# pbmc <- NormalizeData(pbmc)
# pbmc <- ScaleData(pbmc)
# scaleresult <- pbmc@assays[["RNA"]]@scale.data
# scaleresult <- orderGene(scaleresult)
#
# ?colorRamp2
# library(circlize)
# col_fun = colorRamp2(c(-4, 0, 4), c("#440454", "#23898D", "#FFEF24"))
#
# pdf('D:/GRN/final/heatmap/picture/genematrix_tfgene.pdf')
# Heatmap(scaleresult,
#         name = "tfExpression",
#         cluster_rows = FALSE,
#         cluster_columns = FALSE,
#         show_column_names = FALSE,
#         show_row_names = FALSE,
#         col = col_fun)
# dev.off()
#
# ggsave("D:/GRN/final/heatmap/picture/1.png",plot=p)
# write.csv(scaleresult,file="D:/GRN/final/heatmap/data/genematrix_tfgene.csv")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# num = 15
# matrix_rownum = 14322%/% 15 + 1
# matrix <- matrix(rep(NA,time = num * matrix_rownum), ncol = num)
# start = seq(from = 1,to = 14322,by = num)
# rownum <- 1
# for(st in start){
#   end <- st + num -1
#   matrix[rownum,] <-ordercellName[st:end]
#   rownum <- rownum + 1
# }
#
# result <- matrix(rep(0,time =dim(rna_counts)[1] * dim(matrix)[1]),ncol = dim(matrix)[1])
#
#
# for(i in 1:dim(matrix)[1]){
#   result[,i] = apply(rna_counts[,colnames(rna_counts)%in% matrix[i,]],1,mean)
# }
# rownames(result) <- rownames(rna_counts)
#
# test  = rna_counts[,colnames(rna_counts)%in% matrix[955,]]
# dim(arr)
# result[,1] <- arr
#
#
#
#
# tfresult <- result[!str_starts(rownames(result),"SMESG"),]
#
# scaleresult <- scale(tfresult)
#
#
#
#
#
# =========================
# 加载必要的 R 包
# =========================
library(ComplexHeatmap)  # 绘制复杂热图
library(stringr)         # 字符串处理
library(Seurat)          # 单细胞分析
library(ggplot2)         # 绘图

# =========================
# 函数：按细胞顺序聚合 RNA 矩阵
# =========================
smallmatrix = function(matrix_counts, num, ordercellName, careGene){
  # 只保留关注基因（careGene）
  rna_counts = matrix_counts[rownames(matrix_counts) %in% careGene, ]

  # 计算每行矩阵需要多少列
  matrix_rownum = length(colnames(rna_counts)) %/% num + 1

  # 创建空矩阵，用于存放每行的细胞名
  namematrix <- matrix(rep(NA, time = num * matrix_rownum), ncol = num)

  # 按 num 分组生成每组起始位置
  start = seq(from = 1, to = dim(rna_counts)[2], by = num)
  rownum <- 1
  for (st in start){
    end <- st + num - 1
    namematrix[rownum, ] <- ordercellName[st:end]  # 每行存储一组细胞名
    rownum <- rownum + 1
  }

  # 创建结果矩阵，用于存放每组的平均表达
  result <- matrix(rep(0, time = dim(rna_counts)[1] * dim(namematrix)[1]), ncol = dim(namematrix)[1])

  # 对每组细胞按行计算平均表达
  for (i in 1:dim(namematrix)[1]){
    result[, i] = apply(rna_counts[, colnames(rna_counts) %in% namematrix[i, ]], 1, mean)
  }

  # 设置行名为基因名
  rownames(result) <- rownames(rna_counts)

  return(result)
}

# =========================
# 函数：设置列名为 pseudotime1, pseudotime2...
# =========================
setColnames = function(matrix_counts){
  name = c()
  for (i in 1:dim(matrix_counts)[2]){
    name = c(name, str_c("pseudotime", i))
  }
  colnames(matrix_counts) <- name
  return(matrix_counts)
}

# =========================
# 函数：按基因在 pseudotime 上的峰值排序
# =========================
orderGene = function(matrix_counts){
  maxresult = c()
  for (i in 1:dim(matrix_counts)[1]){
    row = matrix_counts[i, ]
    maxresult = c(maxresult, as.numeric(which(row == max(row))))  # 找到每行最大值的位置
  }
  # 按最大值位置对基因排序
  return(matrix_counts[order(maxresult), ])
}

# =========================
# 加载数据
# =========================
pseudotime <- readRDS("/home/wwd/data/newpseudotime.rds")  # 包含细胞 pseudotime 排序
rna_counts <- readRDS("/home/wwd/data/rnaMatrix.rds")     # RNA-seq counts

# 按 pseudotime 排序细胞名
ordercellName <- pseudotime$cellName[order(pseudotime$pseudotime)]

# 只关注非 SMESG 的基因
caregene <- rownames(rna_counts)[!str_starts(rownames(rna_counts), "SMESG")]

# =========================
# 聚合 RNA 表达矩阵
# =========================
smallrnaMatrix <- smallmatrix(
  matrix_counts = rna_counts,
  num = 100,
  ordercellName = ordercellName,
  careGene = caregene
)

# 设置列名为 pseudotime1, pseudotime2 ...
smallrnaMatrix <- setColnames(smallrnaMatrix)

# 创建 Seurat 对象并标准化
pbmc <- CreateSeuratObject(counts = smallrnaMatrix)
pbmc <- NormalizeData(pbmc)
pbmc <- ScaleData(pbmc)

# 获取标准化后的表达矩阵
scaleresult <- pbmc@assays[["RNA"]]@scale.data

# 按 pseudotime 峰值排序基因
scaleresult <- orderGene(scaleresult)

# =========================
# 绘制热图
# =========================
library(circlize)  # 提供 colorRamp2

# 定义颜色映射
col_fun = colorRamp2(c(-4, 0, 4), c("#440454", "#23898D", "#FFEF24"))

# 保存 PDF 热图
pdf('/home/wwd/data/heatmap/picture/genematrix_tfgene.pdf')
Heatmap(
  scaleresult,
  name = "tfExpression",
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_column_names = FALSE,
  show_row_names = FALSE,
  col = col_fun
)
dev.off()

# =========================
# 保存 CSV 文件
# =========================
write.csv(scaleresult, file="/home/wwd/data/heatmap/data/genematrix_tfgene.csv")
