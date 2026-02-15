# library(stringr)
# treenode <- readRDS("D:/GRN/final/data/treenode.rds")
# pseudotime <- readRDS("D:/GRN/final/data/newpseudotime.rds")
#
# ?str_starts
# #
# length(treenode[is.na(treenode)])
#
# cathepsin = c("13-","12-","11-","9-","4-")
# muscle = c("13-","12-","11-","9-","8-")
# epidermal = c("13-","12-","11-","1-")
# intestines = c("13-","12-","11-","6-")
# protonephridia = c("13-","12-","5-")
# pharynx = c("13-","7-")
# neural = c("14-","2-")
# parenchymal = c("14-","3-")
#
# x = treenode
# x[str_starts(treenode,"12-")] <- NA
# unique(x)
#
#
# getTreeNode <- function(treenode,care,pseudotime){
#   x = treenode
#   result = rep(FALSE, times = length(treenode))
#   for(item in care){
#     result = result | str_starts(treenode,item)
#   }
#   x[!result] <- NA
#   pseudotime$pseudotime[is.na(x)] <- NA
#   return(pseudotime$pseudotime)
# }
#
# cds <- readRDS("D:/GRN/final/data/cds.rds")
# umap <- readRDS("D:/GRN/final/data/umap.rds")
# carepseudo = getTreeNode(treenode,parenchymal,pseudotime)
# p <- drawUMAP(cds,umap,carepseudo)
# ggsave("D:/GRN/final/umap/trajectory/parenchymal.pdf",plot = p)
# print(p)
#
#
# drawUMAP <- function(cds,umap,value){
#   umapMatrix <- as.matrix(
#     umap$df)
#   cds@int_colData$reducedDims$UMAP <- umapMatrix
#   cds[["ps"]] <- value
#   p <- plot_cells(cds,
#                   color_cells_by = "ps", label_cell_groups = FALSE,
#                   label_leaves = FALSE, label_branch_points = FALSE,
#                   show_trajectory_graph = FALSE
#   )
#   return(p)
# }
#
# carepseudo = getTreeNode(treenode,cathepsin,pseudotime)
#
# pseudotimeTo100=function(pseudotime){
#   min <- min(pseudotime$pseudotime)
#   max <- max(pseudotime$pseudotime)
#   min <- 0
#   max <- 1
#   result <- pseudotime$pseudotime *100/(max-min)
#   return(result)
# }
#
# saveRDS(carepseudo,"D:/GRN/final/data/cathepsin_pseudo.rds")

library(stringr)

# 读取树状节点信息和 pseudotime 数据
treenode <- readRDS("/home/wwd/data/treenode.rds")
pseudotime <- readRDS("/home/wwd/data/newpseudotime.rds")

?str_starts  # 查看 stringr::str_starts 的用法

# 查看 treenode 中 NA 的数量
length(treenode[is.na(treenode)])

# 定义各类细胞对应的节点前缀
cathepsin = c("13-","12-","11-","9-","4-")
muscle = c("13-","12-","11-","9-","8-")
epidermal = c("13-","12-","11-","1-")
intestines = c("13-","12-","11-","6-")
protonephridia = c("13-","12-","5-")
pharynx = c("13-","7-")
neural = c("14-","2-")
parenchymal = c("14-","3-")

# 将 treenode 中以 "12-" 开头的节点置为 NA
x = treenode
x[str_starts(treenode,"12-")] <- NA
unique(x)  # 查看剩余唯一节点

# --------
# 定义函数：根据关心的节点筛选 pseudotime
# treenode: 每个细胞对应的树节点
# care: 关心的节点前缀列表
# pseudotime: 包含 pseudotime 信息的 data.frame
# 返回值：只保留 care 节点对应细胞的 pseudotime，其余置为 NA
# --------
getTreeNode <- function(treenode,care,pseudotime){
  x = treenode
  result = rep(FALSE, times = length(treenode))  # 初始化逻辑向量
  for(item in care){
    result = result | str_starts(treenode,item)  # 如果节点以 care 中任意前缀开头则置 TRUE
  }
  x[!result] <- NA  # 非关心节点置 NA
  pseudotime$pseudotime[is.na(x)] <- NA  # 对应 pseudotime 也置 NA
  return(pseudotime$pseudotime)
}

# 读取 Monocle3 的 cds 对象和 umap 结果
cds <- readRDS("/home/wwd/data/cds.rds")
umap <- readRDS("/home/wwd/data/umap.rds")

# 筛选 parenchymal 相关的 pseudotime
carepseudo = getTreeNode(treenode,parenchymal,pseudotime)

# 绘制 parenchymal 细胞的 UMAP，按 pseudotime 上色
p <- drawUMAP(cds,umap,carepseudo)
ggsave("/home/wwd/data/umap/trajectory/parenchymal.pdf",plot = p)
print(p)

# --------
# 定义函数：根据 pseudotime 绘制 UMAP
# cds: Monocle3 的 cds 对象
# umap: umap 对象
# value: pseudotime 或其他数值向量，用于上色
# 返回值：ggplot 对象
# --------
drawUMAP <- function(cds,umap,value){
  umapMatrix <- as.matrix(umap$df)  # 提取 UMAP 坐标
  cds@int_colData$reducedDims$UMAP <- umapMatrix  # 更新 cds 内部 UMAP
  cds[["ps"]] <- value  # 将 value 赋给 cds，用于 plot_cells 上色
  p <- plot_cells(
    cds,
    color_cells_by = "ps",           # 按 pseudotime 上色
    label_cell_groups = FALSE,       # 不显示群组标签
    label_leaves = FALSE,            # 不标记叶子节点
    label_branch_points = FALSE,     # 不标记分支点
    show_trajectory_graph = FALSE    # 不显示轨迹线
  )
  return(p)
}

# 筛选 cathepsin 相关的 pseudotime
carepseudo = getTreeNode(treenode,cathepsin,pseudotime)

# --------
# 定义函数：将 pseudotime 映射到 0~100 的区间
# --------
pseudotimeTo100=function(pseudotime){
  min <- min(pseudotime$pseudotime)
  max <- max(pseudotime$pseudotime)
  # 这里硬编码 0~1 也可以使用 min/max
  min <- 0
  max <- 1
  result <- pseudotime$pseudotime *100/(max-min)
  return(result)
}

# 保存 cathepsin 的 pseudotime
saveRDS(carepseudo,"/home/wwd/data/cathepsin_pseudo.rds")
