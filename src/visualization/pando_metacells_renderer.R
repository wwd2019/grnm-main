# draw_grn_tree.R — 参数化圆形 GRN 树图
# 用法: Rscript draw_grn_tree.R --edges <csv> --root <gene> --output <png/pdf>
# 风格: 完全复用 tree_building + dendrogram_zenodo.R 的 igraph circular tree

library(dplyr, warn.conflicts = FALSE)
library(ggraph, warn.conflicts = FALSE)
library(igraph, warn.conflicts = FALSE)
library(ggplot2, warn.conflicts = FALSE)
library(ggrepel, warn.conflicts = FALSE)

# ── 解析命令行参数 ──
args <- commandArgs(trailingOnly = TRUE)
edges_file <- NULL
root_gene  <- "GLI3"
output_file <- "grn_tree.pdf"
n_layer1    <- 80
n_hubs      <- 3
n_layer2    <- 50
highlight_genes <- c("HES4", "HES5", "CREB5", "PAX6")
spread_highlight_labels <- FALSE
label_all_nodes <- FALSE

i <- 1
while (i <= length(args)) {
  if (args[i] == "--edges")      { edges_file  <- args[i+1]; i <- i+2 }
  else if (args[i] == "--root")  { root_gene   <- args[i+1]; i <- i+2 }
  else if (args[i] == "--output"){ output_file <- args[i+1]; i <- i+2 }
  else if (args[i] == "--n1")    { n_layer1    <- as.integer(args[i+1]); i <- i+2 }
  else if (args[i] == "--nhubs") { n_hubs      <- as.integer(args[i+1]); i <- i+2 }
  else if (args[i] == "--n2")    { n_layer2    <- as.integer(args[i+1]); i <- i+2 }
  else if (args[i] == "--highlight") {
    highlight_genes <- unlist(strsplit(args[i+1], ","))
    highlight_genes <- trimws(highlight_genes)
    highlight_genes <- highlight_genes[nzchar(highlight_genes)]
    i <- i+2
  }
  else if (args[i] == "--spread-labels") {
    spread_highlight_labels <- TRUE
    i <- i+1
  }
  else if (args[i] == "--label-all") {
    label_all_nodes <- TRUE
    i <- i+1
  }
  else { i <- i+1 }
}

if (is.null(edges_file)) {
  stop("Usage: Rscript draw_grn_tree.R --edges <csv> --root <gene> --output <png>")
}

cat(sprintf("Input:  %s\n", edges_file))
cat(sprintf("Root:   %s\n", root_gene))
cat(sprintf("Output: %s\n", output_file))
cat(sprintf("Params: n1=%d hubs=%d n2=%d\n", n_layer1, n_hubs, n_layer2))

# ── 读取边数据 ──
meta_data <- read.csv(edges_file, stringsAsFactors = FALSE)

# 期望列: tf, target, estimate (或 from, to, weight/estimate)
if ("tf" %in% names(meta_data) && "target" %in% names(meta_data)) {
  meta_data <- meta_data %>%
    select(tf, target, estimate = matches("estimate|weight")) %>%
    filter(!is.na(tf), !is.na(target), nzchar(trimws(tf)), nzchar(trimws(target)))
  names(meta_data)[1:2] <- c("tf", "target")
} else if ("from" %in% names(meta_data) && "to" %in% names(meta_data)) {
  meta_data <- meta_data %>%
    select(tf = from, target = to, estimate = matches("estimate|weight")) %>%
    filter(!is.na(tf), !is.na(target), nzchar(trimws(tf)), nzchar(trimws(target)))
} else {
  stop("Edge CSV must have columns: tf,target,estimate or from,to,weight")
}

cat(sprintf("Total edges loaded: %d\n", nrow(meta_data)))

# ── 确保 estimate 列存在 ──
if (!"estimate" %in% names(meta_data)) {
  names(meta_data)[3] <- "estimate"
}
meta_data$estimate <- as.numeric(meta_data$estimate)

# ── BFS 两层树构建 ──
# 重点基因的优先级高于按权重截断：
#   1. GLI3 有直接边时，始终画为一级（直接调控）；
#   2. 没有直接边时，寻找 GLI3 -> 中介 TF -> 重点基因的最可信两跳路径，
#      并强制把该中介 TF 和二级边加入树（间接调控）。
all_tfs <- unique(meta_data$tf)

# Layer 1: root → 直接靶基因
layer1_raw <- meta_data %>%
  filter(tf == root_gene) %>%
  mutate(abs_est = abs(estimate)) %>%
  slice_max(abs_est, n = n_layer1)

# 重点基因若为 root 的真实靶基因，强制入选 Layer 1（直接调控）。
layer1_direct_highlights <- meta_data[0, ]
if (length(highlight_genes) > 0) {
  layer1_direct_highlights <- meta_data %>%
    filter(tf == root_gene, target %in% highlight_genes) %>%
    mutate(abs_est = abs(estimate))
  layer1_raw <- bind_rows(layer1_raw, layer1_direct_highlights) %>%
    distinct(tf, target, .keep_all = TRUE)
}

# 对没有 GLI3 直连的重点基因，选一条两跳调控路径。
# 路径得分用两条边绝对权重的较小值，避免一强一弱的伪高分路径。
direct_highlight_names <- unique(layer1_direct_highlights$target)
indirect_highlights <- setdiff(highlight_genes, direct_highlight_names)
forced_indirect <- data.frame(from=character(), to=character(), attr=character())
if (length(indirect_highlights) > 0) {
  root_edges <- meta_data %>%
    filter(tf == root_gene, target %in% all_tfs) %>%
    transmute(intermediate = target, root_estimate = estimate,
              root_abs = abs(estimate))
  downstream_edges <- meta_data %>%
    filter(tf %in% root_edges$intermediate, target %in% indirect_highlights) %>%
    transmute(intermediate = tf, target, downstream_estimate = estimate,
              downstream_abs = abs(estimate))
  indirect_paths <- inner_join(root_edges, downstream_edges, by = "intermediate") %>%
    mutate(path_score = pmin(root_abs, downstream_abs)) %>%
    group_by(target) %>%
    slice_max(path_score, n = 1, with_ties = FALSE) %>%
    ungroup()

  if (nrow(indirect_paths) > 0) {
    # 中介 TF 的 GLI3 边必须在一级树中，才能明确表达间接调控关系。
    forced_root_edges <- indirect_paths %>%
      transmute(tf = root_gene, target = intermediate,
                estimate = root_estimate, abs_est = root_abs)
    layer1_raw <- bind_rows(layer1_raw, forced_root_edges) %>%
      distinct(tf, target, .keep_all = TRUE)
    forced_indirect <- indirect_paths %>%
      transmute(from = intermediate, to = target,
                attr = if_else(downstream_estimate > 0, "positive", "negative"))
  }
}

# 选 Layer 2 的 hub TFs
layer2_hubs <- layer1_raw %>%
  filter(target %in% all_tfs) %>%
  slice_max(abs_est, n = n_hubs) %>%
  pull(target)
# 间接重点基因使用的中介 TF 无论权重排名如何都必须作为 hub。
layer2_hubs <- unique(c(layer2_hubs, forced_indirect$from))
# 重点基因只要存在于一级层且是 TF, 就强制作为 hub (存在即展示其调控边)
watch_hubs <- intersect(unique(layer1_raw$target), highlight_genes)
watch_hubs <- intersect(watch_hubs, all_tfs)
layer2_hubs <- unique(c(layer2_hubs, watch_hubs))

# Layer 1 边
layer1 <- layer1_raw %>%
  transmute(from = tf, to = target,
            attr = if_else(estimate > 0, "positive", "negative"))

# Layer 2: hub TFs → 下游靶基因
layer1_targets_all <- unique(layer1$to)

if (length(layer2_hubs) > 0) {
  layer2 <- meta_data %>%
    filter(tf %in% layer2_hubs) %>%
    group_by(tf) %>%
    mutate(abs_est = abs(estimate)) %>%
    slice_max(abs_est, n = n_layer2) %>%
    ungroup() %>%
    filter(!target %in% layer1_targets_all) %>%
    group_by(target) %>%
    slice_max(abs_est, n = 1) %>%
    ungroup() %>%
    transmute(from = tf, to = target,
              attr = if_else(estimate > 0, "positive", "negative"))
  # 重点基因若为 hub 的真实靶基因, 强制入选 Layer 2
  if (length(highlight_genes) > 0) {
    layer2_extra <- meta_data %>%
      filter(tf %in% layer2_hubs, target %in% highlight_genes,
             !target %in% layer1_targets_all) %>%
      mutate(abs_est = abs(estimate)) %>%
      group_by(target) %>%
      slice_max(abs_est, n = 1) %>%
      ungroup() %>%
      transmute(from = tf, to = target,
                attr = if_else(estimate > 0, "positive", "negative"))
    layer2 <- bind_rows(layer2, layer2_extra) %>%
      distinct(from, to, .keep_all = TRUE)
  }
  # 合并上面为"无直接边"的重点基因专门找到的两跳路径。
  layer2 <- bind_rows(layer2, forced_indirect) %>%
    distinct(from, to, .keep_all = TRUE)
} else {
  layer2 <- forced_indirect
}

# ── 第 3 层: 一级节点作为 TF 的下游边 (GLI3 的直接靶 → 更多一级靶)
# 让 GLI3 树图不只有"根+一层叶子", 而是有真实两层的调控网络。
# 仅当一级靶中存在 TF, 且仍想保留 hub tree 时; 这里始终启用 (hub 优先)。
layer1_tf_set <- intersect(unique(layer1_raw$target), all_tfs)
if (length(layer1_tf_set) > 0) {
  layer3 <- meta_data %>%
    filter(tf %in% layer1_tf_set) %>%
    group_by(tf) %>%
    mutate(abs_est = abs(estimate)) %>%
    slice_max(abs_est, n = n_layer2) %>%
    ungroup() %>%
    filter(!target %in% layer1_targets_all) %>%
    group_by(target) %>%
    slice_max(abs_est, n = 1) %>%
    ungroup() %>%
    transmute(from = tf, to = target,
              attr = if_else(estimate > 0, "positive", "negative"))
  # 重点基因无论在第2层还是第3层都被强制覆盖
  if (length(highlight_genes) > 0) {
    layer3_extra <- meta_data %>%
      filter(tf %in% layer1_tf_set, target %in% highlight_genes,
             !target %in% layer1_targets_all) %>%
      mutate(abs_est = abs(estimate)) %>%
      group_by(target) %>%
      slice_max(abs_est, n = 1) %>%
      ungroup() %>%
      transmute(from = tf, to = target,
                attr = if_else(estimate > 0, "positive", "negative"))
    layer3 <- bind_rows(layer3, layer3_extra) %>%
      distinct(from, to, .keep_all = TRUE)
  }
} else {
  layer3 <- data.frame(from=character(), to=character(), attr=character())
}

edges <- bind_rows(layer1, layer2, layer3) %>% distinct(from, to, .keep_all = TRUE)
cat(sprintf("Tree: L1=%d edges, L2=%d edges, L3=%d edges, total=%d\n",
            nrow(layer1), nrow(layer2), nrow(layer3), nrow(edges)))

cat(sprintf("Highlights: %s\n", paste(highlight_genes, collapse = ", ")))
cat(sprintf("Direct GLI3 highlights: %s\n",
            ifelse(length(direct_highlight_names) > 0,
                   paste(direct_highlight_names, collapse = ", "), "none")))
cat(sprintf("Forced indirect highlights: %s\n",
            ifelse(nrow(forced_indirect) > 0,
                   paste(unique(forced_indirect$to), collapse = ", "), "none")))
unresolved_highlights <- setdiff(indirect_highlights, unique(forced_indirect$to))
if (length(unresolved_highlights) > 0) {
  cat(sprintf("No two-hop GLI3 path for: %s\n",
              paste(unresolved_highlights, collapse = ", ")))
}

if (nrow(edges) == 0) {
  # root 在数据集中没有边: 只画 root 单节点 + 说明文字
  cat(sprintf("Warning: root %s has no edges in this dataset, drawing placeholder\n", root_gene))
  p <- ggplot(data.frame(x = 0, y = 0), aes(x, y)) +
    geom_point(shape = 21, fill = "#2C3E50", color = "black", size = 8) +
    geom_text(aes(x = 0, y = 0.3, label = root_gene),
              size = 4.5, fontface = "bold", color = "#1a1a2e") +
    annotate("text", x = 0, y = -0.45,
             label = "no edges for this root in this dataset",
             size = 4, color = "#888888") +
    xlim(-1.5, 1.5) + ylim(-1.5, 1.5) +
    theme_void()
} else {
  # ── 构建节点表 ──
  name <- unique(c(as.character(edges$from), as.character(edges$to)))
  showname <- sapply(name, function(nm) {
    if (grepl("^SMESG", nm) || grepl("^SMED", nm)) "" else nm
  })

  vertices <- data.frame(name = name, showname = showname, stringsAsFactors = FALSE)

  # ── 标记重点基因 (红色标签) ──
  vertices$is_highlight <- vertices$name %in% highlight_genes | vertices$name == root_gene

  # ── 构建图 & 圆形树布局 ──
  mygraph <- graph_from_data_frame(edges, vertices = vertices)
  layout <- create_layout(mygraph, layout = "tree", circular = TRUE)

  layout$angle <- atan2(layout$y, layout$x) * 180 / pi
  layout$hjust <- ifelse(layout$angle < -90 | layout$angle > 90, 1, 0)
  layout$angle <- ifelse(layout$angle < -90 | layout$angle > 90,
                         layout$angle + 180, layout$angle)

  # ── 合并标签信息到 layout ──
  layout$showname    <- vertices$showname[match(layout$name, vertices$name)]
  layout$is_highlight <- vertices$is_highlight[match(layout$name, vertices$name)]
  # 默认只显示重点基因和 root 的标签; --label-all 时显示所有节点的名字
  if (!label_all_nodes) {
    layout$showname[!layout$is_highlight] <- ""
  }
  # 将重点基因标签沿圆形树的径向外移，给多个红色标签更多间隔。
  # root 位于圆心，保留原位；二级叶节点自然会比一级节点更靠外。
  # 全量标签时不需要大幅外移，否则文字会离节点太远。
  label_distance <- if (label_all_nodes) {
    1.04
  } else if (spread_highlight_labels) {
    1.65
  } else {
    1.22
  }
  layout$label_x <- ifelse(layout$name == root_gene, layout$x, layout$x * label_distance)
  layout$label_y <- ifelse(layout$name == root_gene, layout$y, layout$y * label_distance)
  # ggrepel 的线必须以真实节点坐标为锚点；label_* 仅用于标签的初始外移量。
  layout$label_nudge_x <- layout$label_x - layout$x
  layout$label_nudge_y <- layout$label_y - layout$y

  # ── 绘图 ──
  p <- ggraph(layout) +
    geom_edge_diagonal(aes(colour = factor(attr)), alpha = 0.5, linewidth = 0.25) +
    scale_edge_color_manual(
      values = c("positive" = "#6495ED", "negative" = "#FA8072"),
      labels = c("positive" = "positive coefficient", "negative" = "negative coefficient"),
      name = NULL
    ) +
    geom_node_point(aes(filter = !is_highlight),
                    shape = 21, fill = "#BEBEBE", color = "black", size = 1.5) +
    geom_node_point(aes(filter = is_highlight & name != root_gene),
                    shape = 21, fill = "#E74C3C", color = "#333333", size = 3.5) +
    geom_node_point(aes(filter = name == root_gene),
                    shape = 21, fill = "#2C3E50", color = "black", size = 5)

  # Phase 3 的重点节点更密集：用 ggrepel 在径向外移后再自动避让，
  # 并保留所有标签（max.overlaps = Inf）。
  if (label_all_nodes) {
    # ── 全量标签: 普通节点深灰小字, 重点基因仍保持红色加粗 ──
    p <- p +
      geom_node_text(
        aes(x = label_x, y = label_y, label = showname, angle = angle, hjust = hjust,
            filter = !is_highlight & name != root_gene),
        size = 2.6, color = "#333333")
    p <- p +
      geom_node_text(
        aes(x = label_x, y = label_y, label = showname, angle = angle, hjust = hjust,
            filter = is_highlight & name != root_gene),
        size = 3.4, fontface = "bold", color = "#D4380D")
  } else if (spread_highlight_labels) {
    highlight_layout <- subset(layout, is_highlight & name != root_gene)
    p <- p +
      ggrepel::geom_text_repel(
        data = highlight_layout,
        # x/y 是实际节点位置，因此每条引导线都准确连接到相应结点。
        aes(x = x, y = y, label = showname),
        inherit.aes = FALSE, size = 3.8, fontface = "bold", color = "#D4380D",
        nudge_x = highlight_layout$label_nudge_x,
        nudge_y = highlight_layout$label_nudge_y,
        box.padding = 0.8, point.padding = 0.5, force = 4, force_pull = 0.15,
        max.iter = 20000, max.overlaps = Inf, min.segment.length = 0,
        seed = 20260816
      )
  } else {
    p <- p +
      geom_node_text(aes(x = label_x, y = label_y, label = showname, angle = angle, hjust = hjust,
                         filter = is_highlight & name != root_gene),
                     size = 3.8, fontface = "bold", color = "#D4380D")
  }

  p <- p +
    geom_node_text(aes(x = label_x, y = label_y, label = showname, angle = angle, hjust = hjust,
                       filter = name == root_gene),
                   size = 4.5, fontface = "bold", color = "#1a1a2e") +
    scale_x_continuous(expand = expansion(mult = 0.18)) +
    scale_y_continuous(expand = expansion(mult = 0.18))
}

p <- p + labs(title = "Pando baseline") + theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))

# ── 游离重点基因: 存在于数据但与 root 无直接/两跳路径 ──
# 尽可能展示: 单独小图画出其自身调控边, 不虚构与 root 的连接
in_tree <- unique(c(edges$from, edges$to))
orphan_genes <- intersect(highlight_genes, unique(c(meta_data$tf, meta_data$target)))
orphan_genes <- setdiff(orphan_genes, in_tree)
p_orphans <- list()
for (og in orphan_genes) {
  oe <- meta_data %>%
    filter(tf == og) %>%
    mutate(abs_est = abs(estimate)) %>%
    slice_max(abs_est, n = 10) %>%
    transmute(from = tf, to = target,
              attr = if_else(estimate > 0, "positive", "negative"))
  if (nrow(oe) == 0) next
  g2 <- graph_from_data_frame(oe, directed = TRUE)
  lay2 <- create_layout(g2, layout = "tree", circular = TRUE,
                        root = intersect(og, V(g2)$name))
  lay2$is_hl <- lay2$name == og
  lay2$label <- ifelse(lay2$is_hl, lay2$name, "")
  p_orphans[[length(p_orphans) + 1]] <- ggraph(lay2) +
    geom_edge_diagonal(aes(colour = factor(attr)), alpha = 0.5, linewidth = 0.25) +
    scale_edge_color_manual(values = c("positive" = "#6495ED", "negative" = "#FA8072"),
                            labels = c("positive" = "positive coefficient", "negative" = "negative coefficient"),
                            name = NULL, guide = "none") +
    geom_node_point(aes(filter = !is_hl), shape = 21, fill = "#BEBEBE",
                    color = "black", size = 2) +
    geom_node_point(aes(filter = is_hl), shape = 21, fill = "#E74C3C",
                    color = "#333333", size = 4) +
    geom_node_text(aes(label = label, filter = is_hl), size = 3.8, fontface = "bold",
                   color = "#D4380D", nudge_y = 0.2) +
    labs(title = sprintf("游离重点基因 %s (存在但无 GLI3 路径)", og)) +
    theme_void() +
    theme(plot.title = element_text(size = 11, hjust = 0.5, color = "#444444"))
}
cat(sprintf("Orphan highlights (存在但不可达, 单独小图): %s\n",
            ifelse(length(orphan_genes) > 0, paste(orphan_genes, collapse = ", "), "none")))

# ── 保存 ──
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)

if (length(p_orphans) > 0) {
  library(patchwork, warn.conflicts = FALSE)
  p_final <- p | wrap_plots(p_orphans, nrow = 1)
  save_width <- 18
} else {
  p_final <- p
  save_width <- 12
}

if (grepl("\\.png$", output_file, ignore.case = TRUE)) {
  ggsave(p_final, filename = output_file, width = save_width, height = 12, dpi = 200,
         bg = "white")
} else {
  ggsave(p_final, filename = output_file, width = save_width, height = 12,
         bg = "white")
}

cat(sprintf("Done → %s\n", output_file))
