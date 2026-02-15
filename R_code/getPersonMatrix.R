# library(uwot)
# library(vizier)
# devtools::install_github("jlmelville/vizier")
# rna_counts <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
# tfname <- read.csv("D:/code/pycharm_workspace/drawPicture/data/table/tablename.csv",header = FALSE)
# result = rna_counts[rownames(rna_counts) %in% tfname$V1,]
# rna_counts[c(TRUE),]
#
# saveRDS(result,"D:/GRN/final/data/tfMatrix")
# result <- readRDS("D:/GRN/final/data/tfMatrix")
# result ['ASCL1',]
#
# cor(x = result ['ASCL1',],
#     y = result ['ALX4',],
#     method = "pearson")
#
# personMatrix = matrix(data = 0, nrow = 199, ncol = 199,dimnames = c(tfname,tfname))
#
# for(name1 in tfname$V1){
#   for(name2 in tfname$V1){
#       personMatrix[name1,name2] = cor(x = result [name1,],
#                                       y = result [name2,],
#                                       method = "pearson")
#   }
# }
#
# saveRDS(personMatrix,"D:/GRN/final/data/personMatrix.rds")
#
# personMatrix <- readRDS("D:/GRN/final/data/personMatrix.rds")
# estimateMatrix <- read.csv(row.names = 1,header=T,file = "D:/code/pycharm_workspace/drawPicture/data/table/tftable.csv")
# estimate_rowname <- rownames(estimateMatrix)
# personMatrix <- personMatrix[rownames(personMatrix) %in% estimate_rowname,]
# abs_estimateMatrix <- abs(estimateMatrix)
# abs_estimateMatrix <- abs_estimateMatrix + 1
# abs_estimateMatrix <- sqrt(abs_estimateMatrix)
# sMatrix = personMatrix * abs_estimateMatrix
#
#
# pca <- prcomp(sMatrix,scale = TRUE)
# pcaresult <- pca$x[,c(1:20)]
# s1k_umap <- umap(pcaresult)
#
# umap <- as.data.frame(s1k_umap)
# colnames(umap)<- c("umap1","umap2")
# umap["tfname"] <-  estimate_rowname
# embed_plotly(pcaresult, s1k_umap, pc_axes = TRUE, equal_axes = TRUE, alpha_scale = 0.2, cex = 1,color_scheme = "RColorBrewer::Dark2")
# pseudotime <- readRDS("D:/GRN/final/data/pseudotime.rds")
# changesymbolname <- read.csv("D:/GRN/data/changesymbolname.csv")
# umap$gene_id <- changesymbolname$gene_id[changesymbolname$SYMBOL %in% estimate_rowname]
# write.csv(umap,file = "D:/GRN/final/data/umap_tf.csv")
library(uwot)        # 用于 UMAP 降维
library(vizier)      # 可视化工具
devtools::install_github("jlmelville/vizier")  # 安装最新版本

# -------- 1. 读取数据 --------
rna_counts <- readRDS("/home/wwd/data/rnaMatrix.rds")  # 原始 RNA-seq 矩阵 [genes x cells]
tfname <- read.csv("/home/wwd/data/tfname.csv", header = FALSE) # TF 列表

# 只保留 TF 的基因表达矩阵
result = rna_counts[rownames(rna_counts) %in% tfname$V1,]

# 查看矩阵示例
rna_counts[c(TRUE),]  # 仅简单示例，选择所有行

# 保存 TF 矩阵
saveRDS(result, "/home/wwd/data/tfMatrix")

# 读取 TF 矩阵
result <- readRDS("/home/wwd/data/tfMatrix")

# 单个 TF 的表达量示例
result['ASCL1',]

# 计算两个 TF 之间的 Pearson 相关系数
cor(x = result['ASCL1',], y = result['ALX4',], method = "pearson")

# -------- 2. 构建 TF 相关性矩阵 --------
# 初始化矩阵 [TF x TF]
personMatrix = matrix(
  data = 0,
  nrow = 199,
  ncol = 199,
  dimnames = list(tfname$V1, tfname$V1)
)

# 双重循环，计算所有 TF 两两的 Pearson 相关系数
for(name1 in tfname$V1){
  for(name2 in tfname$V1){
      personMatrix[name1,name2] = cor(
        x = result[name1,],
        y = result[name2,],
        method = "pearson"
      )
  }
}

# 保存 Pearson 相关矩阵
saveRDS(personMatrix, "/home/wwd/data/personMatrix.rds")

# -------- 3. 加权处理相关矩阵 --------
personMatrix <- readRDS("/home/wwd/data/personMatrix.rds")

# 读取估计矩阵（例如 TF 活性或调控强度）
estimateMatrix <- read.csv(
  row.names = 1,
  header = TRUE,
  file = "/home/wwd/data/umapMatrix.csv"
)

estimate_rowname <- rownames(estimateMatrix)

# 保留 personMatrix 中存在于 estimateMatrix 的 TF
personMatrix <- personMatrix[rownames(personMatrix) %in% estimate_rowname,]

# 对 estimateMatrix 取绝对值并加 1，之后开根号
abs_estimateMatrix <- abs(estimateMatrix)
abs_estimateMatrix <- abs_estimateMatrix + 1
abs_estimateMatrix <- sqrt(abs_estimateMatrix)

# 将 Pearson 相关矩阵与加权矩阵相乘
sMatrix = personMatrix * abs_estimateMatrix

# -------- 4. PCA 降维 --------
pca <- prcomp(sMatrix, scale = TRUE)   # 标准化后 PCA
pcaresult <- pca$x[, c(1:20)]          # 取前 20 个主成分

# -------- 5. UMAP 降维 --------
s1k_umap <- umap(pcaresult)            # 使用 uwot UMAP

# 构建 UMAP 数据框
umap <- as.data.frame(s1k_umap)
colnames(umap) <- c("umap1","umap2")
umap["tfname"] <- estimate_rowname     # 保存 TF 名称

# 可视化 PCA + UMAP
embed_plotly(
  pcaresult,
  s1k_umap,
  pc_axes = TRUE,
  equal_axes = TRUE,
  alpha_scale = 0.2,
  cex = 1,
  color_scheme = "RColorBrewer::Dark2"
)

# -------- 6. 添加基因 ID --------
pseudotime <- readRDS("/home/wwd/data/pseudotime.rds")  # 可选，和 pseudotime 结合
changesymbolname <- read.csv("/home/wwd/data/changesymbolname.csv")  # SYMBOL->gene_id 对应表

umap$gene_id <- changesymbolname$gene_id[
  changesymbolname$SYMBOL %in% estimate_rowname
]

# 保存最终 UMAP 数据
write.csv(umap, file = "/home/wwd/data/umap_tf.csv")
