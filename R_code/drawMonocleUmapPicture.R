# library(monocle3)
# library(ggplot2)
# library(plotly)
# cds <- readRDS("D:/GRN/final/data/cds.rds")
#
# umap <- readRDS("D:/GRN/final/data/umap.rds")
# meta_umap <- cds@int_colData$reducedDims$UMAP
#
# umapMatrix <- as.matrix(
#   umap$df)
#
# colnames(umapMatrix) <- c("UMAP_1","UMAP_2")
#
# cds@int_colData$reducedDims$UMAP <- umapMatrix
#
# pseudotime <- readRDS("D:/GRN/final/data/pseudotime.rds")
#
# pseudotimeTo100=function(pseudotime){
#   min <- min(pseudotime$pseudotime)
#   max <- max(pseudotime$pseudotime)
#   result <- pseudotime$pseudotime *100/(max-min)
#   return(result)
# }
#
# pseudotime$normalpseudotime = result
# saveRDS(pseudotime,"D:/GRN/final/data/pseudotime.rds")
#
# result <- pseudotimeTo100(pseudotime)
#
#
# cds <- learn_graph(cds)
# cds[["ps"]] <- pseudotime$pseudotime
#
#
# pseudotime <- readRDS("D:/GRN/final/data/newpseudotime.rds")
#
# ?plot_cells
# p <- plot_cells(cds,
#            color_cells_by = "ps", label_cell_groups = FALSE,
#            label_leaves = FALSE, label_branch_points = FALSE,
#            show_trajectory_graph = FALSE
# )
# ggsave("D:/GRN/final/picture/Pseudotime.pdf", plot = p)
#
# p<- plot_cells(cds,
#            color_cells_by = "cellType", label_groups_by_cluster = FALSE,
#            label_leaves = FALSE, label_branch_points = FALSE
# )
# ggsave("D:/GRN/final/picture/Trajectory.png", plot = p)
#
# order_cell <- order_cells(cds)
#
# p <- plot_cells(
#             order_cell,
#            color_cells_by = "pseudotime", label_cell_groups = FALSE,
#            label_leaves = FALSE, label_branch_points = FALSE
# )
# ggsave("D:/GRN/final/picture/pseudotime.png", plot = p)
#
#
# ?geom_point

# =========================
# 加载必要的 R 包
# =========================
library(monocle3)   # 用于单细胞轨迹推断（pseudotime / trajectory）
library(ggplot2)    # 绘图
library(plotly)     # 交互式 3D 图（可选）

# =========================
# 读取单细胞数据集 CDS 对象
# =========================
cds <- readRDS("/home/wwd/data/cds.rds")

# =========================
# 读取 UMAP 降维结果
# =========================
umap <- readRDS("/home/wwd/data/umap.rds")

# 提取原本 CDS 中的 UMAP 矩阵（用于核对 / 更新）
meta_umap <- cds@int_colData$reducedDims$UMAP

# 将外部 UMAP 数据转换为矩阵格式
umapMatrix <- as.matrix(umap$df)
colnames(umapMatrix) <- c("UMAP_1","UMAP_2")  # 设置列名

# 更新 CDS 内部的 UMAP 结果
cds@int_colData$reducedDims$UMAP <- umapMatrix

# =========================
# 读取 pseudotime 信息
# =========================
pseudotime <- readRDS("/home/wwd/data/pseudotime.rds")

# =========================
# 定义函数，将 pseudotime 归一化到 0-100
# =========================
pseudotimeTo100 = function(pseudotime){
  min <- min(pseudotime$pseudotime)
  max <- max(pseudotime$pseudotime)
  result <- pseudotime$pseudotime * 100 / (max - min)
  return(result)
}

# 归一化 pseudotime
result <- pseudotimeTo100(pseudotime)
pseudotime$normalpseudotime <- result

# 保存归一化后的 pseudotime
saveRDS(pseudotime,"/home/wwd/data/pseudotime.rds")

# =========================
# 构建单细胞轨迹图
# =========================
cds <- learn_graph(cds)  # monocle3 内置函数，构建细胞轨迹图

# 将 pseudotime 添加到 CDS 对象中
cds[["ps"]] <- pseudotime$pseudotime

# =========================
# 读取新的 pseudotime 数据（如果更新了）
# =========================
pseudotime <- readRDS("/home/wwd/data/newpseudotime.rds")

# =========================
# 绘制 pseudotime 渐变图
# =========================
p <- plot_cells(
      cds,
      color_cells_by = "ps",          # 用 pseudotime 上色
      label_cell_groups = FALSE,      # 不显示细胞分组标签
      label_leaves = FALSE,           # 不标注叶节点
      label_branch_points = FALSE,    # 不标注分支点
      show_trajectory_graph = FALSE   # 不显示轨迹图
)
ggsave("/home/wwd/data/picture/Pseudotime.pdf", plot = p)  # 保存 PDF

# =========================
# 按 cellType 上色绘制轨迹图
# =========================
p <- plot_cells(
      cds,
      color_cells_by = "cellType",    # 用细胞类型上色
      label_groups_by_cluster = FALSE,# 不显示分组标签
      label_leaves = FALSE,
      label_branch_points = FALSE
)
ggsave("/home/wwd/data/picture/Trajectory.png", plot = p)

# =========================
# 对细胞按轨迹顺序排序
# =========================
order_cell <- order_cells(cds)  # monocle3 内置函数，输出排序后的 CDS 对象

# =========================
# 绘制排序后的 pseudotime 图
# =========================
p <- plot_cells(
      order_cell,
      color_cells_by = "pseudotime", # 用 pseudotime 上色
      label_cell_groups = FALSE,
      label_leaves = FALSE,
      label_branch_points = FALSE
)
ggsave("/home/wwd/data/picture/pseudotime.png", plot = p)

# =========================
# 注：?geom_point
# geom_point 是 ggplot2 的函数，用于绘制散点图
# 在 plot_cells 内部已经调用了 geom_point 来显示每个细胞
# =========================


