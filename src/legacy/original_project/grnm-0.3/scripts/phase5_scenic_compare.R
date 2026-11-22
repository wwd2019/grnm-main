# phase5_scenic_compare.R — SCENIC 式方法与 GRNM 方法的对比
# ================================================================
# SCENIC 工作流: 共表达排序 (GENIE3/GRNBoost2) → motif 富集 (cisTarget) → AUCell
#
# 数据约束 (诚实声明):
#   - 仓库中没有原始表达计数矩阵, 只有 RNA PCA (50,106 基因 × 64 PCs)。
#     因此共表达排序在 RNA PCA 空间做 (Pearson 相关, GENIE3 的原理替代)。
#   - 没有涡虫 cisTarget motif 数据库, 且 tf_gene_motif.csv 实际是
#     TF→自身SMESG 的 ID 表 (331 个 TF), 不是 TF→靶基因的 motif 表,
#     所以 cisTarget motif 过滤这一步无法执行, 如实省略。
#
# 对比内容:
#   1. 金标准 (example_data/edges.csv) 上两个方法的 AUROC / AUPRC
#      - SCENIC式: TF→靶基因共表达分数
#      - GRNM方法: new_edge.txt 的 |weight| (无预测边记 0)
#   2. 每个 TF 的 top-50 靶基因集合的 Jaccard 重叠
#   3. GLI3 regulon 详细对比 (含重点基因 PAX6/HES5 的排名)
#
# 用法: Rscript scripts/phase5_scenic_compare.R
# ================================================================

library(data.table, warn.conflicts = FALSE)
library(ggplot2, warn.conflicts = FALSE)
library(patchwork, warn.conflicts = FALSE)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
if (length(script_path) > 0) {
  ROOT_DIR <- dirname(dirname(normalizePath(script_path)))
} else {
  ROOT_DIR <- getwd()
}
DATA_DIR  <- file.path(ROOT_DIR, "data")
EDGE_DIR  <- file.path(ROOT_DIR, "example_data")
PHASE3_DIR<- file.path(ROOT_DIR, "outputs", "phase3_output")
OUT_DIR   <- file.path(ROOT_DIR, "outputs", "phase5_output")
PNG_DIR   <- file.path(ROOT_DIR, "figures")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PNG_DIR, showWarnings = FALSE, recursive = TRUE)

# 重点基因 (用户 watchlist; 涡虫数据里只存在 PAX6 / HES5)
WATCH <- c("PAX6" = "SMED30024586,SMED30017839", "HES5" = "SMED30005790")
GLI3_SMED <- "SMED30015613"

set.seed(42)

# ────────────────────────────────────────────────────────────────
# 1. 加载 RNA PCA 与 TF 列表
# ────────────────────────────────────────────────────────────────
cat("1. 加载 RNA PCA ...\n")
rna <- fread(file.path(DATA_DIR, "rna_pca_new.csv"), data.table = FALSE)
rownames(rna) <- rna[[1]]
rna <- as.matrix(rna[, -1])                    # (50106, 64)
cat(sprintf("   RNA PCA: %d 基因 × %d PCs\n", nrow(rna), ncol(rna)))

map_id <- fread(file.path(DATA_DIR, "smed_smesg_mapping.csv"), data.table = FALSE)
smesg2smed <- setNames(map_id$smed_id, map_id$smesg_id)

tf_tab <- fread(file.path(EDGE_DIR, "tf_gene_motif.csv"), data.table = FALSE)
tf_smed <- smesg2smed[tf_tab$smesg]
tf_keep <- !is.na(tf_smed) & tf_smed %in% rownames(rna)
tf_names <- tf_tab$tfname[tf_keep]
tf_smed  <- unname(tf_smed[tf_keep])
cat(sprintf("   TF: %d 个 (motif 表 %d 个中可映射到 RNA PCA)\n",
            length(tf_smed), nrow(tf_tab)))

# ────────────────────────────────────────────────────────────────
# 2. 共表达分数: TF × 全部基因 的 Pearson 相关 (PCA 空间)
#    SCENIC 式 GRN 推断 (GENIE3 的共表达排序原理替代)
# ────────────────────────────────────────────────────────────────
cat("2. 计算 TF × 基因 共表达分数 ...\n")
rna_s <- scale(rna)                             # 行标准化 (基因 × 64)
X <- rna_s[tf_smed, , drop = FALSE]             # (nTF, 64)
Y <- t(rna_s)                                   # (64, nGene)
coexp <- X %*% Y / (ncol(rna_s) - 1)            # Pearson r: (nTF, nGene)
rownames(coexp) <- tf_smed                      # 行名用 SMED, 便于与新GRN配对对齐
colnames(coexp) <- rownames(rna)
cat(sprintf("   共表达矩阵: %d × %d\n", nrow(coexp), ncol(coexp)))

# ────────────────────────────────────────────────────────────────
# 3. 金标准: example_data/edges.csv → SMED 配对
# ────────────────────────────────────────────────────────────────
cat("3. 构建金标准配对 (edges.csv) ...\n")
edges_gt <- fread(file.path(EDGE_DIR, "edges.csv"), data.table = FALSE)
s_gt <- smesg2smed[as.character(edges_gt$source)]
t_gt <- smesg2smed[as.character(edges_gt$target)]
pos_pairs <- data.frame(tf = s_gt, target = t_gt, stringsAsFactors = FALSE)
pos_pairs <- pos_pairs[complete.cases(pos_pairs), ]
pos_pairs <- pos_pairs[pos_pairs$tf %in% tf_smed &
                         pos_pairs$target %in% rownames(rna), ]
pos_pairs <- unique(pos_pairs)
cat(sprintf("   金标准正配对: %d 条 (映射后)\n", nrow(pos_pairs)))

# 负配对: 随机采样同等数量 (不在金标准中)
neg_pairs <- data.frame(tf = character(0), target = character(0))
pos_key <- paste(pos_pairs$tf, pos_pairs$target)
attempts <- 0
while (nrow(neg_pairs) < nrow(pos_pairs) && attempts < nrow(pos_pairs) * 200) {
  cand <- data.frame(tf = sample(tf_smed, nrow(pos_pairs), replace = TRUE),
                     target = sample(rownames(rna), nrow(pos_pairs), replace = TRUE))
  cand <- unique(cand)
  cand <- cand[!paste(cand$tf, cand$target) %in% pos_key, ]
  cand <- cand[!paste(cand$tf, cand$target) %in% paste(neg_pairs$tf, neg_pairs$target), ]
  need <- nrow(pos_pairs) - nrow(neg_pairs)
  neg_pairs <- rbind(neg_pairs, head(cand, need))
  attempts <- attempts + 1
}
cat(sprintf("   负配对: %d 条 (随机采样)\n", nrow(neg_pairs)))

# ────────────────────────────────────────────────────────────────
# 4. GRNM 方法分数: new_edge.txt |weight| (无预测边记 0)
#    target 混合 SMED/SMESG.1 → 统一为 SMED
# ────────────────────────────────────────────────────────────────
cat("4. 读取 new_edge.txt (GRNM 方法) ...\n")
ne <- fread(file.path(PHASE3_DIR, "new_edge.txt"), data.table = FALSE)
cat(sprintf("   new_edge: %s 条\n", format(nrow(ne), big.mark = ",")))
ne$tgt_base <- sub("\\.[0-9]+$", "", as.character(ne$target))
tgt_smed <- ifelse(startsWith(ne$tgt_base, "SMESG"),
                   smesg2smed[ne$tgt_base], ne$tgt_base)
keep <- !is.na(tgt_smed)
ne <- data.frame(tf = ne$source[keep],
                 target = tgt_smed[keep],
                 w = abs(ne$weight[keep]),
                 stringsAsFactors = FALSE)
# 同 (tf,target) 多行 → 取 |weight| 最大
ne <- aggregate(w ~ tf + target, data = ne, FUN = max)
ne_key <- paste(ne$tf, ne$target)
ne_map <- setNames(ne$w, ne_key)
cat(sprintf("   映射为 SMED 配对后: %d 条\n", nrow(ne)))

score_phase3 <- function(tf, target) {
  w <- ne_map[paste(tf, target)]
  w[is.na(w)] <- 0
  w
}
score_scenic <- function(tf, target) {
  idx <- match(tf, rownames(coexp))
  coexp[cbind(idx, match(target, colnames(coexp)))]
}

# ────────────────────────────────────────────────────────────────
# 5. 金标准评估: AUROC / AUPRC
# ────────────────────────────────────────────────────────────────
cat("5. 金标准评估 ...\n")
s_sc <- c(score_scenic(pos_pairs$tf, pos_pairs$target),
          score_scenic(neg_pairs$tf, neg_pairs$target))
s_p3 <- c(score_phase3(pos_pairs$tf, pos_pairs$target),
          score_phase3(neg_pairs$tf, neg_pairs$target))
lab  <- c(rep(1, nrow(pos_pairs)), rep(0, nrow(neg_pairs)))

rank_auc <- function(s, l) {
  r <- rank(s, ties.method = "average")
  n1 <- sum(l); n0 <- sum(!l)
  (sum(r[l == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
pr_auc <- function(s, l) {
  o <- order(s, decreasing = TRUE)
  s <- s[o]; l <- l[o]
  n_pos <- sum(l)
  tp <- cumsum(l); fp <- cumsum(!l)
  prec <- tp / (tp + fp); rec <- tp / n_pos
  # 梯形积分 (按阈值去重)
  keep <- !duplicated(s)
  prec <- prec[keep]; rec <- rec[keep]
  prec <- c(1, prec); rec <- c(0, rec)
  sum(diff(rec) * (prec[-1] + prec[-length(prec)]) / 2)
}
roc_curve <- function(s, l) {
  o <- order(s, decreasing = TRUE)
  l <- l[o]
  fpr <- cumsum(!l) / sum(!l); tpr <- cumsum(l) / sum(l)
  data.frame(fpr = c(0, fpr), tpr = c(0, tpr))
}

auc_sc <- rank_auc(s_sc, lab); auc_p3 <- rank_auc(s_p3, lab)
pra_sc <- pr_auc(s_sc, lab);  pra_p3 <- pr_auc(s_p3, lab)
cat(sprintf("   SCENIC式: AUROC=%.3f  AUPRC=%.3f\n", auc_sc, pra_sc))
cat(sprintf("   GRNM方法: AUROC=%.3f  AUPRC=%.3f\n", auc_p3, pra_p3))

roc_both <- rbind(
  data.frame(roc_curve(s_sc, lab), method = sprintf("SCENIC式 (AUROC=%.3f)", auc_sc)),
  data.frame(roc_curve(s_p3, lab), method = sprintf("GRNM (AUROC=%.3f)", auc_p3)))
roc_cols <- setNames(c("#55A868", "#4C72B0"), unique(roc_both$method))
p_roc <- ggplot(roc_both, aes(fpr, tpr, color = method)) +
  geom_line(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey60") +
  scale_color_manual(values = roc_cols, name = NULL) +
  labs(x = "False positive rate", y = "True positive rate",
       title = "金标准 edges.csv 上两方法 ROC") +
  theme_bw() + theme(legend.position = "bottom")

pr_both <- function(s, l, nm) {
  o <- order(s, decreasing = TRUE)
  s <- s[o]; l <- l[o]
  n_pos <- sum(l)
  tp <- cumsum(l); fp <- cumsum(!l)
  d <- data.frame(rec = c(0, tp / n_pos), prec = c(1, tp / (tp + fp)))
  keep <- !duplicated(s)
  d <- d[c(TRUE, keep), ]
  d$method <- nm
  d
}
prd <- rbind(pr_both(s_sc, lab, sprintf("SCENIC式 (AUPRC=%.3f)", pra_sc)),
             pr_both(s_p3, lab, sprintf("GRNM (AUPRC=%.3f)", pra_p3)))
pr_cols <- setNames(c("#55A868", "#4C72B0"), unique(prd$method))
p_pr <- ggplot(prd, aes(rec, prec, color = method)) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = pr_cols, name = NULL) +
  labs(x = "Recall", y = "Precision",
       title = "金标准 edges.csv 上两方法 PR 曲线") +
  theme_bw() + theme(legend.position = "bottom")

ggsave(p_roc + p_pr, filename = file.path(PNG_DIR, "phase5_scenic_roc.png"),
       width = 12, height = 5.5, dpi = 150, bg = "white")
cat("   -> figures/phase5_scenic_roc.png\n")

# ────────────────────────────────────────────────────────────────
# 6. 每个 TF 的 top-50 靶基因 Jaccard 重叠
# ────────────────────────────────────────────────────────────────
cat("6. top-50 靶基因集合重叠 (Jaccard) ...\n")
TOP_K <- 50
jacs <- numeric(length(tf_smed))
for (i in seq_along(tf_smed)) {
  tf <- tf_smed[i]
  s1 <- sort(coexp[i, ], decreasing = TRUE)[seq_len(TOP_K)]
  g1 <- names(s1)
  w2 <- ne$w[ne$tf == tf]
  g2 <- head(ne$target[ne$tf == tf][order(-w2)], TOP_K)
  jacs[i] <- length(intersect(g1, g2)) / length(union(g1, g2))
}
jaccard_mean <- mean(jacs, na.rm = TRUE)
cat(sprintf("   平均 Jaccard (top-%d): %.4f\n", TOP_K, jaccard_mean))

# 保存 SCENIC 式 top-50 边表
scenic_top <- lapply(seq_along(tf_smed), function(i) {
  s <- sort(coexp[i, ], decreasing = TRUE)[seq_len(TOP_K)]
  data.frame(tf = tf_names[i], tf_smed = tf_smed[i],
             target = names(s), coexp_score = as.numeric(s))
})
scenic_top <- do.call(rbind, scenic_top)
fwrite(scenic_top, file.path(OUT_DIR, "scenic_top50_edges.csv"))
cat(sprintf("   -> outputs/phase5_output/scenic_top50_edges.csv (%d 条)\n",
            nrow(scenic_top)))

# ────────────────────────────────────────────────────────────────
# 7. GLI3 regulon 对比 (含重点基因排名)
# ────────────────────────────────────────────────────────────────
cat("7. GLI3 regulon 对比 ...\n")
if (GLI3_SMED %in% tf_smed) {
  gli3_row <- which(tf_smed == GLI3_SMED)
  s3 <- sort(coexp[gli3_row, ], decreasing = TRUE)
  rank_sc <- match(names(s3), colnames(coexp))          # 全基因排名
  rank_sc_all <- setNames(seq_along(s3), names(s3))

  g3_ne <- ne[ne$tf == GLI3_SMED, ]
  g3_ne <- g3_ne[order(-g3_ne$w), ]
  rank_p3_all <- setNames(seq_len(nrow(g3_ne)), g3_ne$target)

  # 与金标准 GLI3 靶重叠
  gt_gli3 <- unique(pos_pairs$target[pos_pairs$tf == GLI3_SMED])
  sc_top100 <- head(names(s3), 100)
  p3_top100 <- head(g3_ne$target, 100)
  ov_sc_gt <- length(intersect(sc_top100, gt_gli3))
  ov_p3_gt <- length(intersect(p3_top100, gt_gli3))
  cat(sprintf("   GLI3 top100 与金标准靶重叠: SCENIC式 %d / GRNM %d (金标准靶 %d)\n",
              ov_sc_gt, ov_p3_gt, length(gt_gli3)))

  # 重点基因在两种方法中的 GLI3 排名
  watch_rank <- do.call(rbind, lapply(names(WATCH), function(gn) {
    ids <- strsplit(WATCH[[gn]], ",")[[1]]
    r_sc <- min(rank_sc_all[ids], na.rm = TRUE)
    r_p3 <- min(rank_p3_all[ids], na.rm = TRUE)
    data.frame(gene = gn, sc_rank = r_sc, p3_rank = r_p3)
  }))
  watch_rank$sc_rank <- ifelse(is.infinite(watch_rank$sc_rank), NA, watch_rank$sc_rank)
  watch_rank$p3_rank <- ifelse(is.infinite(watch_rank$p3_rank), NA, watch_rank$p3_rank)
  print(watch_rank)

  # GLI3 靶基因的排名对比散点 (两种方法)
  common <- intersect(names(rank_sc_all), g3_ne$target)
  rk <- data.frame(gene = common,
                   sc = rank_sc_all[common],
                   p3 = rank_p3_all[common],
                   is_watch = common %in% unlist(strsplit(WATCH, ",")))
  p_gli3 <- ggplot(rk, aes(sc, p3)) +
    geom_point(alpha = 0.25, size = 1.2, color = "#9EB3C7") +
    geom_point(data = subset(rk, is_watch), color = "#E74C3C", size = 3) +
    ggrepel::geom_text_repel(data = subset(rk, is_watch),
                             aes(label = gene), color = "#D4380D",
                             size = 4, fontface = "bold", max.overlaps = Inf) +
    labs(x = "SCENIC式 GLI3 靶排名", y = "GRNM GLI3 靶排名",
         title = sprintf("GLI3 靶基因两方法排名对比 (共同靶 %d 个)", nrow(rk))) +
    theme_bw()
  ggsave(p_gli3, filename = file.path(PNG_DIR, "phase5_scenic_gli3.png"),
         width = 6.5, height = 6, dpi = 150, bg = "white")
  cat("   -> figures/phase5_scenic_gli3.png\n")
} else {
  cat("   GLI3 不在 TF 列表中, 跳过\n")
  watch_rank <- data.frame(gene = names(WATCH), sc_rank = NA, p3_rank = NA)
}

# ────────────────────────────────────────────────────────────────
# 8. 汇总报告
# ────────────────────────────────────────────────────────────────
lines <- c(
  "================================================================",
  "Phase5 SCENIC 式方法对比报告",
  paste("  SCENIC式 = TF×基因共表达排序 (RNA PCA 空间, Pearson r)",
        "- GENIE3 的共表达原理替代"),
  paste("  注: 无原始表达矩阵(仅PCA)且无涡虫cisTarget数据库,"),
  paste("      motif 富集与 AUCell 步骤无法执行, 如实省略"),
  "",
  sprintf("1. 金标准 edges.csv 评估 (正 %d / 负 %d)",
          nrow(pos_pairs), nrow(neg_pairs)),
  sprintf("   SCENIC式: AUROC=%.3f  AUPRC=%.3f", auc_sc, pra_sc),
  sprintf("   GRNM方法: AUROC=%.3f  AUPRC=%.3f", auc_p3, pra_p3),
  sprintf("2. 每TF top-50 靶集合平均 Jaccard: %.4f", jaccard_mean),
  "3. GLI3 regulon:",
  sprintf("   GLI3 top100 与金标准靶重叠: SCENIC式 %d / GRNM %d (金标准靶 %d)",
          ov_sc_gt, ov_p3_gt, length(gt_gli3)),
  "   重点基因 GLI3 排名:",
  paste(capture.output(print(watch_rank, row.names = FALSE)), collapse = "\n")
)
writeLines(lines, file.path(OUT_DIR, "phase5_scenic_summary.txt"))
cat("   -> outputs/phase5_output/phase5_scenic_summary.txt\n")
cat("Done!\n")
