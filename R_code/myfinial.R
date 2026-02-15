# library(Signac)
# library(Seurat)
# library(EnsDb.Hsapiens.v86)
# library(dplyr)
# library(ggplot2)
# library(Pando)
# library(ArchR)
# ArchR::installExtraPackages()
# rna_counts <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
# atac_counts <- readRDS("D:/GRN/final/data/peakMatrix.rds")
# atac_rowname <- readRDS("D:/GRN/final/data/atacrowname.rds")
# # rnanames <- readRDS("D:/GRN/final/data/rnaname.rds")
# # rna_counts@Dimnames[[1]] <- rnanames
# # saveRDS(rna_counts,"D:/GRN/final/data/rnaMatrix.rds")
# memory.limit(60000)
# pbmc <- CreateSeuratObject(counts = rna_counts)
# pbmc <- Seurat::FindVariableFeatures(pbmc, assay='RNA')
#
# pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
# annotations <- readRDS("D:/GRN/final/data/geneGr.rds")
# dgcmatrix <- readRDS("D:/GRN/final/data/dgcmatrix.rds")
# atac_rowname2 <- atac_rowname[[1]]
# dgcmatrix@Dimnames[[1]] <- atac_rowname2
#
# dgcmatrix@Dimnames[[2]] <- atac_counts@Dimnames[[2]]
# saveRDS(dgcmatrix,"D:/GRN/final/data/finalpeakMatrix.rds")
#
#
#
# #####################################################################
# rna_counts <- readRDS("D:/GRN/final/data/rnaMatrix.rds")
# atac_counts <- readRDS("D:/GRN/final/data/finalpeakMatrix.rds")
# pbmc <- CreateSeuratObject(counts = rna_counts)
# pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
# # annotations <- readRDS("D:/GRN/final/data/geneGr.rds")
# annotations <- readRDS("F:/GRN/final/data/final_geneGr.rds")
# atac_rowname <- atac_counts@Dimnames[[1]]
# chrom_assay <- CreateChromatinAssay(counts = atac_counts,  sep = c(":", "-"), annotation = annotations)
# chrom_assay@data@Dimnames[[1]] <- atac_rowname
# pbmc[['peaks']] <- chrom_assay
# pbmc <- Seurat::FindVariableFeatures(pbmc, assay='RNA')
#
# gene_annot <- Signac::Annotation(pbmc[['peaks']])
# if (is.null(gene_annot)) {
#   stop("Please provide a gene annotation for the ChromatinAssay.")
# }
# peak_ranges <- StringToGRanges(rownames(GetAssay(pbmc,
#                                                  assay = 'peaks')),sep = c(":","-"))
# cand_ranges <- peak_ranges
# exon_ranges <- gene_annot[gene_annot$type == "exon", ]
# names(exon_ranges@ranges) <- NULL
# exon_ranges <- IRanges::intersect(exon_ranges, exon_ranges)
# exon_ranges <- GenomicRanges::GRanges(seqnames = exon_ranges@seqnames,
#                                       ranges = exon_ranges@ranges)
# cand_ranges <- IRanges::setdiff(cand_ranges, exon_ranges,
#                                 ignore.strand = TRUE)
# peak_overlaps <- findOverlaps(cand_ranges, peak_ranges)
# peak_matches <- subjectHits(peak_overlaps)
# regions_obj <- new(Class = "Regions", ranges = cand_ranges,
#                    peaks = peak_matches, motifs = NULL)
# params <- list(peak_assay = "peaks", rna_assay = "RNA",
#                exclude_exons = TRUE)
# grn_obj <- new(Class = "RegulatoryNetwork", regions = regions_obj,
#                params = params)
# test_srt <- as(pbmc, "SeuratPlus")
# test_srt@grn <- grn_obj
# saveRDS(test_srt,"D:/GRN/final/data/final_test_srt.rds")
#
# finialFindMotifs<- readRDS("D:/GRN/final/data/final_find_motifs.rds")
# Seurat::FindVariableFeatures(finialFindMotifs, assay='RNA')
# grn_test_srt <- infer_grn(finialFindMotifs)
#
# memory.limit(60000)
# ##################################################
# findmotifs <- readRDS("D:/GRN/final/data/final_mydeal_find_motifs.rds")
# modules <- NetworkModules(findmotifs)
#
#
# saveRDS(modules,"D:/GRN/final/data/final_mydeal_modules.rds")
# meta <- modules@meta
# features <- modules@features
# meta <- as.matrix(meta)
# write.csv(meta,file = "D:/GRN/final/data/meta.csv")#
library(Signac) # ATAC-seq分析
library(Seurat) # scRNA-seq分析
library(EnsDb.Hsapiens.v86) # 人类基因注释
library(dplyr)
library(ggplot2)
library(Pando) # GRN分析
library(ArchR) # ATAC-seq高级分析
library(GenomicRanges)
library(SeuratData)
library(TFBSTools) # 提供getMatrixSet函数
library(JASPAR2020)
library(Biostrings)
# 安装 ArchR 额外依赖（如果还没装）
# ArchR::installExtraPackages()

# -----------------------------
# Step 1: 读取 RNA 和 ATAC count 矩阵
# -----------------------------
rna_counts <- readRDS("/home/wwd/data/rnaMatrix.rds")
atac_counts <- readRDS("/home/wwd/data/peakMatrix.rds")
atac_rowname <- readRDS("/home/wwd/data/atacrowname.rds")


# -----------------------------
# Step 2: 创建 Seurat 对象（RNA）
# -----------------------------
pbmc <- CreateSeuratObject(counts = rna_counts)
pbmc <- Seurat::FindVariableFeatures(pbmc, assay = "RNA")
# 添加线粒体比例
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")


# -----------------------------
# Step 3: 处理 ATAC 注释和行名
# -----------------------------
annotations <- readRDS("/home/wwd/data/final_geneGr.rds") # 基因注释
dgcmatrix <- readRDS("/home/wwd/data/dgcmatrix.rds")
# 给 ATAC matrix 行列名对齐
atac_rowname2 <- atac_rowname[[1]]
dgcmatrix@Dimnames[[1]] <- atac_rowname2
dgcmatrix@Dimnames[[2]] <- atac_counts@Dimnames[[2]]
# 保存对齐后的 ATAC matrix
saveRDS(dgcmatrix, "/home/wwd/data/finalpeakMatrix.rds")


# -----------------------------
# Step 4: 创建 ChromatinAssay 并加入 Seurat 对象
# -----------------------------
atac_counts <- readRDS("/home/wwd/data/finalpeakMatrix.rds")
chrom_assay <- CreateChromatinAssay(
  counts = atac_counts,
  sep = c(":", "-"), # peak 名称分隔符
  annotation = annotations # 基因注释
)
# 规范格式
# 1. 定义标准特征名（以counts矩阵为基准，替换下划线为短横线，消除格式问题）
# 若counts矩阵行名已无下划线，此步骤仅做统一提取，不修改内容
standard_features <- gsub(
  pattern = "_",
  replacement = "-",
  x = rownames(chrom_assay@counts)
)
# 2. 同步更新counts矩阵行名（核心基准，必须优先设置）
rownames(chrom_assay@counts) <- standard_features
# 3. 同步更新data矩阵行名（若存在则必须匹配，无则跳过）
if (!is.null(chrom_assay@data)) {
  rownames(chrom_assay@data) <- standard_features
}
# 4. 同步更新meta.features行名（最易遗漏！必须与前两者完全一致）
rownames(chrom_assay@meta.features) <- standard_features
# 5. 验证修复后的对象有效性（无报错则修复成功）
validObject(chrom_assay)
cat("ChromatinAssay对象修复成功，核心部分特征名完全匹配！\n")
# chrom_assay@data@Dimnames[[1]] <- atac_counts@Dimnames[[1]]  # 行名对齐
pbmc[["peaks"]] <- chrom_assay
pbmc <- Seurat::FindVariableFeatures(pbmc, assay = "RNA")

# -----------------------------
# Step 5: 构建 GRN 对象
# -----------------------------
gene_annot <- Signac::Annotation(pbmc[["peaks"]])
if (is.null(gene_annot)) {
  stop("请提供 ChromatinAssay 的基因注释")
}
peaks_assay <- pbmc[["peaks"]]
peaks_rownames <- rownames(peaks_assay)
# 规范格式
# 步骤1：按-拆分字符串，提取各部分
split_str <- strsplit(peaks_rownames, split = "-")
# 步骤2：提取seqnames（除最后2个部分外的所有部分，重新拼接）
seqnames_vec <- sapply(split_str, function(x) paste(x[1:(length(x) - 2)], collapse = "-"))
# 步骤3：提取start和end（最后2个部分，转为数值）
start_vec <- as.numeric(sapply(split_str, function(x) x[length(x) - 1]))
end_vec <- as.numeric(sapply(split_str, function(x) x[length(x)]))
# 创建 GRanges 对象
peak_ranges <- GRanges(
  seqnames = seqnames_vec,
  ranges = IRanges(start = start_vec, end = end_vec)
)
# 构建候选调控区域
cand_ranges <- peak_ranges
exon_ranges <- gene_annot[gene_annot$type == "exon", ]
names(exon_ranges@ranges) <- NULL
exon_ranges <- IRanges::intersect(exon_ranges, exon_ranges)
exon_ranges <- GenomicRanges::GRanges(
  seqnames = exon_ranges@seqnames,
  ranges = exon_ranges@ranges
)
# 排除外显子
cand_ranges <- IRanges::setdiff(cand_ranges, exon_ranges, ignore.strand = TRUE)
# 匹配 peak
peak_overlaps <- findOverlaps(cand_ranges, peak_ranges)
peak_matches <- subjectHits(peak_overlaps)
# 构建 Regions 对象
regions_obj <- new(
  Class = "Regions",
  ranges = cand_ranges,
  peaks = peak_matches,
  motifs = NULL
)
# 设置参数
params <- list(peak_assay = "peaks", rna_assay = "RNA", exclude_exons = TRUE)
# 创建 RegulatoryNetwork 对象
grn_obj <- new(Class = "RegulatoryNetwork", regions = regions_obj, params = params)

# -----------------------------
# Step 6: 转换为 SeuratPlus 并加入 GRN
# -----------------------------
# 创建一个不依赖继承的SeuratPlus类
# 首先检查pbmc对象
cat("检查pbmc对象状态...\n")
cat("类:", class(pbmc), "\n")
cat("Assays:", names(pbmc@assays), "\n")
cat("细胞数:", ncol(pbmc), "\n")
# 确保assays是命名列表
if (is.null(names(pbmc@assays))) {
  cat("警告: assays没有名称，正在修复...\n")
  names(pbmc@assays) <- paste0("Assay", seq_along(pbmc@assays))
}
# 创建SeuratPlus构造函数
SeuratPlus <- function(seurat_obj) {
  # 创建一个新的环境来存储所有数据
  env <- new.env(parent = emptyenv())

  # 复制所有Seurat数据到环境
  env$assays <- seurat_obj@assays
  env$meta.data <- seurat_obj@meta.data
  env$active.assay <- seurat_obj@active.assay
  env$active.ident <- seurat_obj@active.ident
  env$reductions <- seurat_obj@reductions
  env$graphs <- seurat_obj@graphs
  env$neighbors <- seurat_obj@neighbors
  env$images <- seurat_obj@images
  env$project.name <- seurat_obj@project.name
  env$version <- seurat_obj@version
  env$commands <- seurat_obj@commands
  env$tools <- seurat_obj@tools
  env$misc <- seurat_obj@misc

  # 添加SeuratPlus特有数据
  env$grn <- list()
  env$trajectories <- list()
  env$networks <- list()
  env$metadata <- list(
    created = Sys.time(),
    original_class = class(seurat_obj)[1]
  )

  # 创建类
  class(env) <- c("SeuratPlus", "environment")

  # 添加一些实用方法
  env$ncol <- function() ncol(env$assays[[env$active.assay]])
  env$nrow <- function() nrow(env$assays[[env$active.assay]])
  env$colnames <- function() colnames(env$assays[[env$active.assay]])
  env$rownames <- function() rownames(env$assays[[env$active.assay]])

  return(env)
}

# 使用方法
test_srt <- SeuratPlus(pbmc)
cat("\nSeuratPlus对象创建成功！\n")
cat("类:", class(test_srt), "\n")
cat("Assays:", names(test_srt$assays), "\n")
cat("细胞数:", test_srt$ncol(), "\n")
cat("基因数:", test_srt$nrow(), "\n")
# 检查结构
cat("\n对象结构:\n")
ls(test_srt)
# 添加GRN的方法
addGRN <- function(seuratplus_obj, grn_obj, name = "grn") {
  cat("添加GRN到SeuratPlus对象...\n")
  # 处理不同类型的GRN对象
  if (inherits(grn_obj, "list")) {
    seuratplus_obj$grn[[name]] <- grn_obj
    cat("添加列表格式的GRN\n")
  } else if (inherits(grn_obj, "igraph")) {
    library(igraph)
    seuratplus_obj$grn[[name]] <- list(
      network = grn_obj,
      edges = as_data_frame(grn_obj),
      vertices = as_data_frame(grn_obj, what = "vertices"),
      metadata = list(
        type = "igraph",
        n_nodes = vcount(grn_obj),
        n_edges = ecount(grn_obj),
        is_directed = is.directed(grn_obj),
        added_date = Sys.time()
      )
    )
    cat("添加igraph格式的GRN\n")
  } else if (inherits(grn_obj, "data.frame")) {
    seuratplus_obj$grn[[name]] <- list(
      edges = grn_obj,
      metadata = list(
        type = "edgelist",
        n_edges = nrow(grn_obj),
        added_date = Sys.time()
      )
    )
    cat("添加数据框格式的GRN\n")
  } else {
    seuratplus_obj$grn[[name]] <- list(
      data = grn_obj,
      metadata = list(
        type = class(grn_obj)[1],
        added_date = Sys.time()
      )
    )
    cat(sprintf("添加%s格式的GRN\n", class(grn_obj)[1]))
  }

  # 更新metadata
  seuratplus_obj$metadata$grn_names <- names(seuratplus_obj$grn)
  seuratplus_obj$metadata$last_updated <- Sys.time()
  cat("GRN添加完成！当前GRN:", paste(names(seuratplus_obj$grn), collapse = ", "), "\n")
  return(seuratplus_obj)
}
# 添加GRN
test_srt <- addGRN(test_srt, grn_obj, "regulatory_network")
# 检查添加结果
cat("\nGRN添加结果检查:\n")
cat("GRN列表:", names(test_srt$grn), "\n")
grn_info <- test_srt$grn$regulatory_network$metadata
if (!is.null(grn_info)) {
  cat("GRN类型:", grn_info$type, "\n")
  cat("节点数:", grn_info$n_nodes, "\n")
  cat("边数:", grn_info$n_edges, "\n")
}
# test_srt <- as(pbmc, "SeuratPlus")
# test_srt@grn <- grn_obj
# 保存中间对象
saveRDS(test_srt, "/home/wwd/data/final_test_srt.rds")
test_srt <- readRDS("/home/wwd/data/final_test_srt.rds")


# -----------------------------
# Step 7: motif 扫描与 GRN 推断
# -----------------------------
finialFindMotifs <- readRDS("/home/wwd/data/final_find_motifs.rds")
class(finialFindMotifs) <- "Seurat"
Seurat::FindVariableFeatures(finialFindMotifs, assay = "RNA")
# 规范格式
convert_colon_to_dash_in_peaks <- function(seurat_obj) {
  cat("=== 将 peaks 中的冒号改为连字符 ===\n")

  if (!"peaks" %in% names(seurat_obj@assays)) {
    cat("未找到 peaks assay\n")
    return(seurat_obj)
  }
  peaks_assay <- seurat_obj[["peaks"]]
  original_names <- rownames(peaks_assay)
  cat("原始peaks数量:", length(original_names), "\n")
  cat("原始格式示例:\n")
  print(head(original_names, 5))
  # 将冒号替换为连字符
  new_names <- gsub(":", "-", original_names)
  cat("\n新格式示例:\n")
  print(head(new_names, 5))
  # 检查修改了多少
  changed <- sum(original_names != new_names)
  cat("修改了", changed, "个peaks的名称\n")
  # 更新所有相关矩阵的行名
  # 1. counts 矩阵
  if (all(dim(peaks_assay@counts) > 0)) {
    rownames(peaks_assay@counts) <- new_names
  }
  # 2. data 矩阵
  if (all(dim(peaks_assay@data) > 0)) {
    rownames(peaks_assay@data) <- new_names
  }
  # 3. scale.data 矩阵（如果有）
  if (.hasSlot(peaks_assay, "scale.data") && all(dim(peaks_assay@scale.data) > 0)) {
    rownames(peaks_assay@scale.data) <- new_names
  }
  # 4. meta.features（如果有）
  if (!is.null(peaks_assay@meta.features) && nrow(peaks_assay@meta.features) > 0) {
    rownames(peaks_assay@meta.features) <- new_names
  }
  # 5. var.features（如果有）
  if (length(peaks_assay@var.features) > 0) {
    # 转换 var.features
    var_idx <- match(peaks_assay@var.features, original_names)
    peaks_assay@var.features <- new_names[var_idx[!is.na(var_idx)]]
  }
  # 更新对象
  seurat_obj[["peaks"]] <- peaks_assay
  cat("转换完成！\n")
  return(seurat_obj)
}
# 应用转换
finialFindMotifs <- convert_colon_to_dash_in_peaks(finialFindMotifs)

# 准备pfm和gegene数据
# 步骤1：加载涡虫基因组，构建专属BSgenome对象（替换错误的 小鼠基因组）
genome_fa_path <- "/home/wwd/data/smed_dd_g4.fa" # 你的涡虫FASTA 路径
planarian_genome <- readDNAStringSet(genome_fa_path)
# 步骤2：获取所有脊椎动物motif（最常用）
cat("\n=== 获取脊椎动物motif ===\n")
pfm <- getMatrixSet(
  x = JASPAR2020,
  opts = list(
    collection = "CORE",
    tax_group = "vertebrates"
  )
)
cat("获取到", length(pfm), "个脊椎动物motif\n")

# GRN
cat("=== 开始Pando GRN分析工作流程 ===\n")
# 步骤1: 检查对象状态
cat("\n[1/4] 检查对象状态\n")
cat("对象类:", class(finialFindMotifs), "\n")
cat("Assays:", names(finialFindMotifs@assays), "\n")
cat("是否有motif数据:", "motifs" %in% names(finialFindMotifs@assays), "\n")
# 步骤2: 准备数据
cat("\n[2/4] 准备数据\n")
# 确保有必要的assay名称
if (!"ATAC" %in% names(finialFindMotifs@assays) && "peaks" %in% names(finialFindMotifs@assays)) {
  cat("重命名peaks assay为ATAC...\n")
  finialFindMotifs@assays$ATAC <- finialFindMotifs@assays$peaks
  DefaultAssay(finialFindMotifs) <- "ATAC"
}

# 确保有基因表达数据
if (!"RNA" %in% names(finialFindMotifs@assays)) {
  cat("警告: 没有找到RNA assay\n")
  # 如果有其他assay包含基因表达，重命名为RNA
  possible_rna_assays <- c("SCT", "integrated", "logcounts")
  for (assay_name in possible_rna_assays) {
    if (assay_name %in% names(finialFindMotifs@assays)) {
      cat(sprintf("重命名%s assay为RNA...\n", assay_name))
      finialFindMotifs@assays$RNA <- finialFindMotifs@assays[[assay_name]]
      break
    }
  }
}
# 步骤3
# 寻找motif（关键步骤！）
cat("\n[3/4] 寻找motif\n")
finialFindMotifs <- initiate_grn(finialFindMotifs)
# find_motifs
if (exists("pfm") && length(pfm) > 0) {
  cat("使用已有的PFM motif...\n")
  tryCatch(
    {
      finialFindMotifs <- find_motifs(
        finialFindMotifs,
        pfm = pfm,
        genome = planarian_genome
      )
      cat("✓ find_motifs成功\n")
    },
    error = function(e) {
      cat("find_motifs失败:", e$message, "\n")
    }
  )
}

# 步骤4：infer_grn
cat("\n[3/4] infer_grn\n")
finialFindMotifs <- infer_grn(
  object = finialFindMotifs,
  peak_to_gene_method = "Signac",
  method = "glm",
  verbose = TRUE
)
# 查看结果
# 获取Network对象
grn_raw <- finialFindMotifs@grn@networks$glm_network
# 1：从coefs提取完整的调控关系数据（推荐）
grn_df <- grn_raw@coefs
cat("✓ 从@coefs提取数据\n")
cat("维度:", dim(grn_df), "\n")
cat("列名:", paste(colnames(grn_df), collapse = ", "), "\n")
cat("调控关系数:", nrow(grn_df), "\n\n")
# 查看前几行
print(head(grn_df, 5))
# 2：从fit提取模型拟合信息
grn_fit <- grn_raw@fit
cat("\n✓ 从@fit提取模型信息\n")
cat("维度:", dim(grn_fit), "\n")
print(head(grn_fit))
# 3: 提取模块信息
finialFindMotifs <- find_modules(
  object = finialFindMotifs, # 直接传GRNData对象
  p_thresh = 0.05 # 唯一可用的参数
)
modules <- finialFindMotifs@grn@networks$glm_network@modules
# 保存结果
# 1. 保存完整的调控关系数据
write.csv(grn_df, "/home/wwd/data/Planarian_GRN_all_regulations.csv", row.names = FALSE)
# 2. 保存整个GRNData对象
saveRDS(finialFindMotifs, "/home/wwd/data/Planarian_GRNData_complete.rds")
# 3. 保存Network对象单独备份
saveRDS(grn_raw, "/home/wwd/data/Planarian_glm_network_object.rds")
# 4. 保存模块
saveRDS(modules, "/home/wwd/data/Planarian_Modules_object.rds")
cat("\n✓ 所有结果已保存！\n")


# -----------------------------
# Step 8: 模块分析
# -----------------------------
# 读取你的GRNData对象
findmotifs <- readRDS("/home/wwd/data/Planarian_GRNData_complete.rds")
findmotifs <- find_modules(
  object = findmotifs, # 直接传GRNData对象
  p_thresh = 0.05 # 唯一可用的参数
)
# 提取模块
modules <- findmotifs@grn@networks$glm_network@modules
# 输出 meta 和 features
meta <- modules@meta
features <- modules@features
meta <- as.matrix(meta)
write.csv(meta, file = "/home/wwd/data/meta_final.csv")
