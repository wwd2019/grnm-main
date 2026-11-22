# 此文件用于从 Zenodo 的 GRN 元信息文件读取数据并绘图
library(dplyr)
library(ggraph)
library(igraph)
library(stringi)

input_path <- file.path(getwd(), "../raw_data/zenodo5242913_GLI3_KO_multiome_18d__GRN_meta_format(2).csv")
out_dir <- file.path(getwd(), "../figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

meta_data <- read.csv(input_path, stringsAsFactors = FALSE)

meta_data <- meta_data %>%
  select(tf, target, estimate) %>%
  filter(!is.na(tf), !is.na(target), nzchar(trimws(tf)), nzchar(trimws(target)))

root <- "GLI3"

edges <- meta_data %>%
  transmute(
    from = tf,
    to = target,
    attr = if_else(estimate > 0, "possitive", "negtaive")
  )

# 只保留与中心基因相连的唯一节点，避免重复
all_nodes <- unique(c(root, edges$from, edges$to))
name <- all_nodes
showname <- sapply(name, function(i) {
  if (startsWith(i, "SMESG")) {
    ""
  } else {
    i
  }
})

# 只在中心基因处显示红色名称，其他节点只显示一次且去重
showname[name != root] <- ""
showname[name == root] <- root

vertices <- data.frame(
  name = name,
  showname = showname,
  stringsAsFactors = FALSE
)

mygraph <- graph_from_data_frame(edges, vertices = vertices, directed = TRUE)
layout <- create_layout(mygraph, layout = "tree", circular = TRUE)

layout$angle <- atan2(layout$y, layout$x) * 180 / pi
layout$hjust <- ifelse(layout$angle < -90 | layout$angle > 90, 1, 0)
layout$angle <- ifelse(layout$angle < -90 | layout$angle > 90, layout$angle + 180, layout$angle)

p <- ggraph(layout) +
  geom_edge_diagonal(aes(colour = factor(attr))) +
  scale_edge_color_manual(values = c("cornflowerblue", "salmon")) +
  geom_node_point(shape = 21, fill = "grey", color = "black", size = 1) +
  geom_node_text(
    aes(label = showname, angle = angle, hjust = hjust),
    size = ifelse(name == root, 3.8, 2.2),
    fontface = ifelse(name == root, "bold", "plain"),
    color = ifelse(name == root, "red", "black")
  )

file_name <- file.path(out_dir, paste0(root, "_tree_pic.pdf"))
ggsave(p, filename = file_name, width = 13, height = 10)
message("Saved: ", file_name)


