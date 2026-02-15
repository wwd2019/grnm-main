# library(dyno)
# library(tidyverse)
# library(Matrix)
# library(Seurat)
# library(hdf5r)
# library(rhdf5)
# rnaMatrix <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
# pbmc <- CreateSeuratObject(counts = rnaMatrix)
#
# dataset <- wrap_expression(
#   counts = t(pbmc@assays$RNA@counts),
#   expression = t(pbmc@assays$RNA@data)
# )
#
# saveRDS(Age,"D:/GRN/final/data/cellNameByLineAge.rds")
# clusterMessage <- readRDS("D:/GRN/final/data/clusterMessage.rds")
# celltypeMessage <- readRDS("D:/GRN/final/data/celltypeMessage.rds")
# rna_counts <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
#
#
# name <- rna_counts@Dimnames[[1]]
# "SMESG000036375" %in% name
# cellname <- celltypeMessage$cellNames[ celltypeMessage$cellType == "Nb2"]
# care <- rna_counts[rna_counts@Dimnames[[1]] %in% "SMESG000036375",rna_counts@Dimnames[[2]] %in% cellname]
#
# care <- care[care@Dimnames[[1]] == "SMESG000036375",]
# which (care == max(care))
#
# rna_counts["SMESG000036375","atac_36hpa#GGTCATACAGACTAAA-1"]
# which(celltypeMessage$cellNames == "atac_36hpa#GGTCATACAGACTAAA-1")
#
# ?add_dimred
#
# model_paga <- readRDS("D:/GRN/final/data/dimred_model_paga.rds")
# plot_dimred(
#   model_paga,
#   expression_source = t(pbmc@assays$RNA@data),
#   feature_oi = "SMESG000036375"
# )
#
#
# model_paga_tree <- readRDS("D:/GRN/final/data/modelpagatree.rds")
# plot_dimred(
#   model_paga_tree,
#   expression_source = t(pbmc@assays$RNA@data),
#   feature_oi = "SMESG000036375"
# )
#
# model_paga <- readRDS("D:/GRN/final/data/model_paga.rds")
# model_paga_tree <- model_paga_tree %>% add_dimred(dyndimred::dimred_mds, expression_source = t(pbmc@assays$RNA@data))
#
# plot_dimred(
#   model_paga_tree,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = celltypeMessage$cellType
# )
#
# saveRDS(model_paga_tree,"D:/GRN/final/data/dimred_model_paga_tree.rds")
#
# rna <- as.matrix(rnaMatrix)
#
# write.csv(rnaMatrix,"D:/GRN/final/data/rnaMatrix.csv",row.names = TRUE)
#
#
#
# ###############
# rnaMatrix <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
# clusterMessage <- readRDS("D:/GRN/final/data/clusterMessage.rds")
# celltypeMessage <- readRDS("D:/GRN/final/data/celltypeMessage.rds")
# ###############
#
# cell <- celltypeMessage$cellNames[celltypeMessage$cellType %in% c("Nb2","Intestines")]
# getMatrix <- rnaMatrix[,rnaMatrix@Dimnames[[2]] %in% cell]
# cluster <- clusterMessage[clusterMessage$cellNames %in% cell,]
# pbmc <- CreateSeuratObject(counts = getMatrix)
# Intestines_model_slingshot <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_slingshot.rds")
# Intestines_model_pagatree <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_pagatree.rds")
# Intestines_model_paga <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_paga.rds")
# Intestines_model_mst <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_mst.rds")
#
# typemessage <- readRDS("D:/GRN/final/InferTrajocry/typeMessage.rds")
# colnames(typemessage) <-c("cellNames","type")
# type <- typemessage[typemessage$cellNames %in% cell,]
#
# plot_dimred(
#    Intestines_model_slingshot,
#    expression_source = t(pbmc@assays$RNA@data),
#    grouping = type$type
# )
#
# plot_dimred(
#   Intestines_model_pagatree,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
#
#
# plot_dimred(
#   Intestines_model_paga,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
#
# plot_dimred(
#   Intestines_model_mst,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
#
# Dimred_Intestines_model_slingshot <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_slingshot.rds")
# Dimred_Intestines_model_pagatree <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_pagatree.rds")
# Dimred_Intestines_model_paga <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_paga.rds")
# Dimred_Intestines_model_mst <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_mst.rds")
#
# plot_dimred(
#   Dimred_Intestines_model_slingshot,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
#
# plot_dimred(
#   Dimred_Intestines_model_pagatree,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
#
#
# plot_dimred(
#   Dimred_Intestines_model_paga,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
#
# plot_dimred(
#   Dimred_Intestines_model_mst,
#   expression_source = t(pbmc@assays$RNA@data),
#   grouping = type$type
# )
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
# ==========================
# 1. 加载所需 R 包
# ==========================
library(dyno)        # 单细胞轨迹推断框架
library(tidyverse)   # 数据处理与可视化
library(Matrix)      # 稀疏矩阵操作
library(Seurat)      # 单细胞分析
library(hdf5r)       # HDF5 文件读取
library(rhdf5)       # HDF5 文件读取（dyno 依赖）

# ==========================
# 2. 创建 Seurat 对象
# ==========================
rnaMatrix <- readRDS("/home/wwd/data/rnaMatrix.rds")
pbmc <- CreateSeuratObject(counts = rnaMatrix)  # 将 RNA counts 转为 Seurat 对象

# ==========================
# 3. 构建 dyno 数据集
# ==========================
dataset <- wrap_expression(
  counts = t(pbmc@assays$RNA@counts),   # counts 矩阵：转置为 cells × genes
  expression = t(pbmc@assays$RNA@data)  # normalized expression 矩阵：转置
)

# ==========================
# 4. 保存细胞对应的年龄信息
# ==========================
saveRDS(Age, "/home/wwd/data/cellNameByLineAge.rds")

# ==========================
# 5. 读取聚类和细胞类型信息
# ==========================
clusterMessage <- readRDS("/home/wwd/data/clusterMessage.rds")  # 聚类信息
celltypeMessage <- readRDS("/home/wwd/data/celltypeMessage.rds")  # 细胞类型信息
rna_counts <- readRDS("/home/wwd/data/rnaMatrix.rds")  # RNA counts

# ==========================
# 6. 筛选特定基因和细胞
# ==========================
name <- rna_counts@Dimnames[[1]]  # 基因名
"SMESG000036375" %in% name       # 检查目标基因是否在矩阵中

# 选择 Nb2 类型的细胞
cellname <- celltypeMessage$cellNames[celltypeMessage$cellType == "Nb2"]

# 获取特定基因在特定细胞的表达矩阵
care <- rna_counts[
  rna_counts@Dimnames[[1]] %in% "SMESG000036375",
  rna_counts@Dimnames[[2]] %in% cellname
]

# 取最大表达量对应的索引
care <- care[care@Dimnames[[1]] == "SMESG000036375",]
which(care == max(care))

# 检查某个细胞名对应的列索引
rna_counts["SMESG000036375", "atac_36hpa#GGTCATACAGACTAAA-1"]
which(celltypeMessage$cellNames == "atac_36hpa#GGTCATACAGACTAAA-1")

# ==========================
# 7. 加载 PAGA 模型并可视化
# ==========================
model_paga <- readRDS("/home/wwd/data/dimred_model_paga.rds")
plot_dimred(
  model_paga,
  expression_source = t(pbmc@assays$RNA@data),
  feature_oi = "SMESG000036375"  # 绘制特定基因在轨迹上的表达
)

model_paga_tree <- readRDS("/home/wwd/data/modelpagatree.rds")
plot_dimred(
  model_paga_tree,
  expression_source = t(pbmc@assays$RNA@data),
  feature_oi = "SMESG000036375"
)

# 添加新的降维信息（MDS）到 PAGA tree
model_paga_tree <- model_paga_tree %>% add_dimred(
  dyndimred::dimred_mds,
  expression_source = t(pbmc@assays$RNA@data)
)

# 按细胞类型绘制轨迹
plot_dimred(
  model_paga_tree,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = celltypeMessage$cellType
)

# 保存处理好的模型
saveRDS(model_paga_tree,"/home/wwd/data/dimred_model_paga_tree.rds")

# ==========================
# 8. 将 RNA matrix 保存为 CSV
# ==========================
rna <- as.matrix(rnaMatrix)
write.csv(rnaMatrix, "/home/wwd/data/rnaMatrix.csv", row.names = TRUE)

# ==========================
# 9. 针对 Nb2 和 Intestines 细胞进行子集分析
# ==========================
rnaMatrix <- readRDS("/home/wwd/data/rnaMatrix.rds")
clusterMessage <- readRDS("/home/wwd/data/clusterMessage.rds")
celltypeMessage <- readRDS("/home/wwd/data/celltypeMessage.rds")

# 选择 Nb2 和 Intestines 细胞
cell <- celltypeMessage$cellNames[celltypeMessage$cellType %in% c("Nb2","Intestines")]
getMatrix <- rnaMatrix[, rnaMatrix@Dimnames[[2]] %in% cell]
cluster <- clusterMessage[clusterMessage$cellNames %in% cell,]

pbmc <- CreateSeuratObject(counts = getMatrix)

# ==========================
# 10. 读取不同轨迹推断模型
# ==========================
Intestines_model_slingshot <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_slingshot.rds")
Intestines_model_pagatree <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_pagatree.rds")
Intestines_model_paga <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_paga.rds")
Intestines_model_mst <- readRDS("D:/GRN/final/InferTrajocry/Intestines_model_mst.rds")

# 读取类型信息
typemessage <- readRDS("D:/GRN/final/InferTrajocry/typeMessage.rds")
colnames(typemessage) <- c("cellNames","type")
type <- typemessage[typemessage$cellNames %in% cell,]

# ==========================
# 11. 绘制不同轨迹模型的降维可视化
# ==========================
plot_dimred(
   Intestines_model_slingshot,
   expression_source = t(pbmc@assays$RNA@data),
   grouping = type$type
)

plot_dimred(
  Intestines_model_pagatree,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)

plot_dimred(
  Intestines_model_paga,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)

plot_dimred(
  Intestines_model_mst,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)

# ==========================
# 12. 读取已保存的 DimRed 结果并绘图
# ==========================
Dimred_Intestines_model_slingshot <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_slingshot.rds")
Dimred_Intestines_model_pagatree <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_pagatree.rds")
Dimred_Intestines_model_paga <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_paga.rds")
Dimred_Intestines_model_mst <- readRDS("D:/GRN/final/InferTrajocry/Dimred_Intestines_model_mst.rds")

plot_dimred(
  Dimred_Intestines_model_slingshot,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)

plot_dimred(
  Dimred_Intestines_model_pagatree,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)

plot_dimred(
  Dimred_Intestines_model_paga,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)

plot_dimred(
  Dimred_Intestines_model_mst,
  expression_source = t(pbmc@assays$RNA@data),
  grouping = type$type
)
