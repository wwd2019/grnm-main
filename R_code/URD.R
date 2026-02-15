# library(URD)
# library(Seurat)
# rnaMatrix <- readRDS("D:/GRN/final/data/rnaRealMatrix.rds")
# pbmc <- CreateSeuratObject(counts = rnaMatrix)
# memory.limit(60000)
# testsample <- createURD(count.data = pbmc@assays$RNA@counts,
#                         meta = pbmc@meta.data)
#
# saveRDS(testsample,"D:/GRN/final/data/urd.rds")
# testsample <- calcPCA(testsample, mp.factor = 2)
# pcSDPlot(testsample)
# set.seed(19)
# testsample <- calcTsne(object = testsample)
#

# ==========================
# 1. 加载所需 R 包
# ==========================
library(URD)     # 用于单细胞发育轨迹推断
library(Seurat)  # 单细胞数据处理与分析

# ==========================
# 2. 读取 RNA 表达矩阵并构建 Seurat 对象
# ==========================
rnaMatrix <- readRDS("D:/GRN/final/data/rnaRealMatrix.rds")
pbmc <- CreateSeuratObject(counts = rnaMatrix)  # 将 RNA counts 转为 Seurat 对象

# 增加内存限制，防止大数据操作时内存不足
memory.limit(60000)

# ==========================
# 3. 构建 URD 对象
# ==========================
testsample <- createURD(
  count.data = pbmc@assays$RNA@counts,  # RNA counts 矩阵
  meta = pbmc@meta.data                  # Seurat 对象的 meta data，包含细胞信息
)

# 保存 URD 对象，便于后续分析
saveRDS(testsample, "D:/GRN/final/data/urd.rds")

# ==========================
# 4. 计算 PCA
# ==========================
testsample <- calcPCA(
  testsample,
  mp.factor = 2  # 多重投影因子，控制 PCA 计算的精度和速度
)

# 可视化主成分标准差，帮助选择主成分数量
pcSDPlot(testsample)

# ==========================
# 5. 计算 t-SNE 降维
# ==========================
set.seed(19)  # 设置随机种子，保证 t-SNE 可重复
testsample <- calcTsne(
  object = testsample  # 基于 PCA 结果计算 t-SNE
)

