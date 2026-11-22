library(dplyr)
library(ggraph)
library(igraph)
library(stringi)

# 从 zenodo CSV 读取数据
meta_data <- read.csv("../raw_data/zenodo5242913_GLI3_KO_multiome_18d__GRN_meta_format(2).csv",
                      stringsAsFactors = FALSE)

meta_data <- meta_data %>%
  select(tf, target, estimate) %>%
  filter(!is.na(tf), !is.na(target), nzchar(trimws(tf)), nzchar(trimws(target)))

root <- 'GLI3'

# 构建边表：两层 GRN，控制各层数量
# 第一层：GLI3 → 直接靶基因，保留 |estimate| 最大的 80 个
layer1_raw <- meta_data %>%
  filter(tf == root) %>%
  mutate(abs_est = abs(estimate)) %>%
  slice_max(abs_est, n = 80)

# 从第一层靶基因中取也是 TF 的，|estimate| 最大的 3 个作为第二层中心
all_tfs <- unique(meta_data$tf)
layer2_hubs <- layer1_raw %>%
  filter(target %in% all_tfs) %>%
  slice_max(abs_est, n = 3) %>%
  pull(target)

layer1 <- layer1_raw %>%
  transmute(
    from = tf,
    to   = target,
    attr = if_else(estimate > 0, "possitive", "negtaive")
  )

# 第二层：3 个中心基因各自保留 |estimate| 最大的 50 个靶基因
layer2 <- meta_data %>%
  filter(tf %in% layer2_hubs) %>%
  group_by(tf) %>%
  mutate(abs_est = abs(estimate)) %>%
  slice_max(abs_est, n = 50) %>%
  ungroup() %>%
  transmute(
    from = tf,
    to   = target,
    abs_est = abs(estimate),
    attr = if_else(estimate > 0, "possitive", "negtaive")
  )

# 去重：每个基因只保留在一个中心基因下（|estimate| 最强的那个），
#        且不在第一层中出现过
layer1_targets_all <- unique(layer1$to)
layer2 <- layer2 %>%
  filter(!to %in% layer1_targets_all) %>%           # 不在第一层
  group_by(to) %>%
  slice_max(abs_est, n = 1) %>%                     # 每个基因只归属一个 hub
  ungroup() %>%
  select(-abs_est)

edges <- bind_rows(layer1, layer2)

# 构建节点表
name <- unique(c(as.character(edges$from), as.character(edges$to)))
showname <- sapply(name, function(i) {
  if (startsWith(i, "SMESG")) {
    return("")
  } else {
    return(i)
  }
})

vertices <- data.frame(
  name     = name,
  showname = showname,
  stringsAsFactors = FALSE
)

# 构建图 & 圆形树布局
mygraph <- graph_from_data_frame(edges, vertices = vertices)
layout <- create_layout(mygraph, layout = "tree", circular = TRUE)

layout$angle <- atan2(layout$y, layout$x) * 180 / pi
layout$hjust <- ifelse(layout$angle < -90 | layout$angle > 90, 1, 0)
layout$angle <- ifelse(layout$angle < -90 | layout$angle > 90,
                       layout$angle + 180, layout$angle)

# 绘图
p <- ggraph(layout) +
  geom_edge_diagonal(aes(colour = factor(attr))) +
  scale_edge_color_manual(values = c("cornflowerblue", "salmon")) +
  geom_node_point(shape = 21, fill = "grey", color = "black", size = 1) +
  geom_node_text(aes(label = showname, angle = angle, hjust = hjust), size = 3) +
  scale_x_continuous(expand = expansion(mult = 0.12)) +
  scale_y_continuous(expand = expansion(mult = 0.12))

# 保存到 figures/ 目录
dir.create("../figures", showWarnings = FALSE)
file_name <- paste("../figures/", root, "_tree_pic.pdf", sep = "")
ggsave(p, filename = file_name, width = 12, height = 10)
