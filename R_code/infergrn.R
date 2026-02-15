# library(Signac)
# library(Seurat)
# library(EnsDb.Hsapiens.v86)
# library(dplyr)
# library(ggplot2)
# library(Pando)
# object <- readRDS("D:/GRN/final/data/final_find_motifs.rds")
# method <-"glm"
# peak_to_gene_method <- "Signac"
# params <- Params(object)
# motif2tf <- NetworkTFs(object)
# genes <- VariableFeatures(object, assay = params$rna_assay)
#
# gene_data <- Matrix::t(Seurat::GetAssayData(object,
#                                             assay = params$rna_assay))
# gene_groups <- TRUE
# peak_data <- Matrix::t(Seurat::GetAssayData(object,
#                                             assay = params$peak_assay))
# peak_groups <- TRUE
#
#
#
# gene_annot <- Signac::Annotation(object[[params$peak_assay]])
# features <- intersect(gene_annot$gene_name, genes) %>% intersect(rownames(GetAssay(object,
#                                                                                    params$rna_assay)))
# gene_annot <- gene_annot[gene_annot$gene_name %in% features, ]
#
# regions <- NetworkRegions(object)
# peak_data <- peak_data[, regions@peaks]
# colnames(peak_data) <- rownames(regions@motifs@data)
# peaks2motif <- regions@motifs@data
#
# regions_ranges <- regions@ranges
# anot <- gene_annot
#
# peaks_near_gene <- find_peaks_near_genes(peaks = regions@ranges,
#                                          method = peak_to_gene_method, genes = gene_annot, upstream = 1e+05,
#                                          downstream = 0, only_tss = FALSE)
# colnames(peaks_near_gene)
#
# peaks2gene <- aggregate_matrix(t(peaks_near_gene), groups = colnames(peaks_near_gene),
#                                fun = "sum")
#
#
#
library(Signac)                # 用于 ATAC-seq 数据处理
library(Seurat)                # 单细胞分析
library(EnsDb.Hsapiens.v86)    # 人类基因注释数据库
library(dplyr)                 # 数据处理
library(ggplot2)               # 可视化
library(Pando)                 # 基因调控网络构建工具

# -------- 1. 读取已经计算好的 motif 数据对象 --------
object <- readRDS("/home/wwd/data/final_find_motifs.rds")

# 分析方法
method <- "glm"                    # GLM 网络方法
peak_to_gene_method <- "Signac"    # 使用 Signac 方法将 peaks 关联到基因

# -------- 2. 提取对象参数 --------
params <- Params(object)           # Pando 对象参数
motif2tf <- NetworkTFs(object)     # 获取网络中 TF 信息
genes <- VariableFeatures(object, assay = params$rna_assay)  # RNA 高变基因

# -------- 3. 构建 RNA / peak 数据矩阵 --------
gene_data <- Matrix::t(Seurat::GetAssayData(object, assay = params$rna_assay))  # RNA 矩阵 [cells x genes]
gene_groups <- TRUE  # 是否按组聚合 RNA

peak_data <- Matrix::t(Seurat::GetAssayData(object, assay = params$peak_assay))  # Peak 矩阵 [cells x peaks]
peak_groups <- TRUE  # 是否按组聚合 peak

# -------- 4. 基因注释 --------
gene_annot <- Signac::Annotation(object[[params$peak_assay]])  # 从 peak assay 中提取注释信息

# 只保留在 RNA 高变基因中的基因
features <- intersect(gene_annot$gene_name, genes) %>%
            intersect(rownames(GetAssay(object, params$rna_assay)))

gene_annot <- gene_annot[gene_annot$gene_name %in% features, ]

# -------- 5. 提取网络中峰区域和 motif 信息 --------
regions <- NetworkRegions(object)         # 获取 Pando 网络中定义的 peak 区域
peak_data <- peak_data[, regions@peaks]   # 筛选只包含网络中的 peak
colnames(peak_data) <- rownames(regions@motifs@data)  # 将列名改为 motif 名
peaks2motif <- regions@motifs@data        # motif 数据矩阵

regions_ranges <- regions@ranges          # peak 的 GenomicRanges 对象
anot <- gene_annot                         # 保留基因注释副本

# -------- 6. 找到每个 peak 附近的基因 --------
peaks_near_gene <- find_peaks_near_genes(
  peaks = regions@ranges,        # 网络中的 peak
  method = peak_to_gene_method,  # 使用 Signac 方法
  genes = gene_annot,            # 基因注释
  upstream = 1e+05,              # 上游 100kb
  downstream = 0,                # 下游 0bp
  only_tss = FALSE               # 是否只考虑 TSS
)

colnames(peaks_near_gene)  # 查看 peak-gene 对应矩阵的列名

# -------- 7. 将 peak-gene 信息聚合成矩阵 --------
# 将 peaks 附近的基因关联信息按列求和，得到每个 gene 对应的 peak 活性矩阵
peaks2gene <- aggregate_matrix(
  t(peaks_near_gene),                # 转置矩阵 [genes x peaks]
  groups = colnames(peaks_near_gene), # 分组信息
  fun = "sum"                         # 聚合函数
)

