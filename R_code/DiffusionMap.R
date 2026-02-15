# library(destiny)
# library(ggplot2)
# library(ggthemes)
# library(plotly)
# memory.limit(60960)
# rnaMatrix <- readRDS("D:/GRN/final/data/rnaRealMatrix.rds")
# eset <- Biobase::ExpressionSet(rnaMatrix)
# rm(rnaMatrix)
#
# lognorm <- t(read.table('C:/Users/zym20/Desktop/data/nestorowa_corrected_log2_transformed_counts.txt', sep=" ", header=TRUE))
#
# dmap <- DiffusionMap(eset)
#
# palette(cube_helix(36)) #��cube_helix������������ɫ
#
# plot.DiffusionMap(dmap, dims=c(1,2))
#
# plot.DiffusionMap(dmap)
#
# plot.DiffusionMap(dmap, dims=c(1,3))
#
# plot(dmap, pch = 20, # pch for prettier points
#      col_by = 'num_cells', # or ��col�� with a vector or one color
#      legend_main = 'Cell stage')
#
# saveRDS(dmap,"D:/GRN/final/data/dmap.rds")
#
# dmap <- readRDS("D:/GRN/final/data/dmap.rds")
# plot.DiffusionMap(dmap)
# plot.DiffusionMap(dmap, dims=c(1,2))
#
#
# celltypeMessage <- readRDS("D:/GRN/final/data/celltypeMessage.rds")
# tmp <- data.frame(DC1 = eigenvectors(dmap)[,1],
#                   DC2 = eigenvectors(dmap)[,2],
#                   CellType = celltypeMessage$cellType)
#
# ?scale_color_tableau
# ggplot(tmp, aes(x = DC1, y = DC2, colour = CellType)) +
#   geom_point() + scale_color_tableau() +
#   xlab("Diffusion component 1") +
#   ylab("Diffusion component 2") +
#   theme_classic()
#
# anno_table <- read.table('C:/Users/zym20/Desktop/data/nestorowa_corrected_population_annotation.txt')
#
#
# tmp <- data.frame(DC1 = eigenvectors(dmap)[,1],
#                   DC2 = eigenvectors(dmap)[,2],
#                   DC3 = eigenvectors(dmap)[,3],
#                   CellType = celltypeMessage$cellType)
# fig <- plot_ly(tmp, x = ~DC1,
#                y = ~DC2,
#                z = ~DC3,
#                color = ~CellType,
#                colors = c('#5078A4','#F18E2F','#E15759','#7DB3AE','#5F9A56','#E5CA5D','#A87C9C','#EEA0A7','#9A7560'))
#
# fig

# =========================
# 加载必要的包
# =========================
library(destiny)     # 用于 Diffusion Map 计算
library(ggplot2)     # 绘图
library(ggthemes)    # 提供一些主题和调色盘
library(plotly)      # 交互式 3D 可视化

# =========================
# 设置内存上限（Windows 专用）
# =========================
memory.limit(60960)  # 设置内存上限为 60GB

# =========================
# 读取 RNA 矩阵并转换为 ExpressionSet 对象
# =========================
rnaMatrix <- readRDS("/home/wwd/data/rnaRealMatrix.rds")  # 读取原始 RNA counts
eset <- Biobase::ExpressionSet(rnaMatrix)                     # 转为 ExpressionSet 格式
rm(rnaMatrix)  # 释放原始矩阵内存

# =========================
# 读取 log2 标准化矩阵（如果需要后续比较）
# =========================
#lognorm <- t(read.table('C:/Users/zym20/Desktop/data/nestorowa_corrected_log2_transformed_counts.txt',
#                        sep=" ", header=TRUE))

# =========================
# 计算 Diffusion Map
# =========================
dmap <- DiffusionMap(eset)  # destiny 包函数，进行非线性降维

# =========================
# 设置绘图调色盘（cube helix）
# =========================
palette(cube_helix(36))  # cube_helix 颜色方案，36 种颜色

# =========================
# 绘制 Diffusion Map
# =========================
plot.DiffusionMap(dmap, dims=c(1,2))  # 绘制 DC1 vs DC2
plot.DiffusionMap(dmap)               # 绘制默认前两个 Diffusion component
plot.DiffusionMap(dmap, dims=c(1,3))  # 绘制 DC1 vs DC3

# =========================
# 另一种绘制方式，可以指定点的颜色和图例
# =========================
plot(dmap,
     pch = 20,          # 设置点的形状
     col_by = 'num_cells',  # 按细胞数量或其他向量着色
     legend_main = 'Cell stage')  # 图例标题

# =========================
# 保存 Diffusion Map 对象，方便后续使用
# =========================
saveRDS(dmap,"/home/wwd/data/dmap.rds")

# =========================
# 读取保存的 Diffusion Map 对象
# =========================
dmap <- readRDS("/home/wwd/data/dmap.rds")
plot.DiffusionMap(dmap)
plot.DiffusionMap(dmap, dims=c(1,2))

# =========================
# 读取细胞类型信息
# =========================
celltypeMessage <- readRDS("/home/wwd/data/celltypeMessage.rds")

# =========================
# 创建数据框用于 ggplot2 绘图
# =========================
tmp <- data.frame(
  DC1 = eigenvectors(dmap)[,1],  # 第一个 diffusion component
  DC2 = eigenvectors(dmap)[,2],  # 第二个 diffusion component
  CellType = celltypeMessage$cellType  # 细胞类型标签
)

# =========================
# ggplot2 绘制 2D diffusion map
# =========================
ggplot(tmp, aes(x = DC1, y = DC2, colour = CellType)) +
  geom_point() +
  scale_color_tableau() +  # 使用 Tableau 调色盘
  xlab("Diffusion component 1") +
  ylab("Diffusion component 2") +
  theme_classic()           # 使用简洁主题

# =========================
# 如果有额外注释文件，可读取
# =========================
#anno_table <- read.table('C:/Users/zym20/Desktop/data/nestorowa_corrected_population_annotation.txt')

# =========================
# 3D diffusion map 可视化
# =========================
tmp <- data.frame(
  DC1 = eigenvectors(dmap)[,1],
  DC2 = eigenvectors(dmap)[,2],
  DC3 = eigenvectors(dmap)[,3],
  CellType = celltypeMessage$cellType
)

# 使用 plotly 绘制 3D 散点图
fig <- plot_ly(tmp,
               x = ~DC1,
               y = ~DC2,
               z = ~DC3,
               color = ~CellType,
               colors = c('#5078A4','#F18E2F','#E15759','#7DB3AE',
                          '#5F9A56','#E5CA5D','#A87C9C','#EEA0A7','#9A7560'))
fig  # 显示交互式 3D 图
