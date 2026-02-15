# example_test_srt <- readRDS("D:/GRN/data/example_test_srt.rds")
# method <-"glm"
# peak_to_gene_method <- "Signac"
# example_params <- Params(example_test_srt)
# example_motif2tf <- NetworkTFs(example_test_srt)
# example_gene_annot <- Signac::Annotation(example_test_srt[[example_params$peak_assay]])
# example_genes <- VariableFeatures(example_test_srt,assay = example_params$rna_assay)
# example_gene_data <- Matrix::t(Seurat::GetAssayData(example_test_srt,
#                                             assay = example_params$rna_assay))
# gene_groups <- TRUE
# example_peak_data <- Matrix::t(Seurat::GetAssayData(example_test_srt,
#                                               assay = example_params$peak_assay))
# peak_groups <- TRUE
# example_features <- intersect(example_gene_annot$gene_name, example_genes) %>% intersect(rownames(GetAssay(example_test_srt,
#                                                                                    example_params$rna_assay)))
# example_gene_annot <- example_gene_annot[example_gene_annot$gene_name %in% example_features, ]
# example_regions <- NetworkRegions(example_test_srt)
# example_peak_data <- example_peak_data[, example_regions@peaks]
# colnames(example_peak_data) <- rownames(example_regions@motifs@data)
# example_peaks2motif <- example_regions@motifs@data
#
# example_regions_ranges <- example_regions@ranges
# example_anot <- example_gene_annot
# example_peaks_near_gene <- find_peaks_near_genes(peaks = example_regions@ranges,
#                                          method = peak_to_gene_method, genes = example_gene_annot, upstream = 1e+05,
#                                          downstream = 0, only_tss = FALSE)
#
# colnames(example_peaks_near_gene)
# example_peaks2gene <- aggregate_matrix(t(example_peaks_near_gene), groups = colnames(example_peaks_near_gene),
#                                fun = "sum")

# -----------------------------
# 1. 读取示例 SeuratPlus 对象
# -----------------------------
example_test_srt <- readRDS("D:/GRN/data/example_test_srt.rds")

# -----------------------------
# 2. 设置 GRN 推断参数
# -----------------------------
method <- "glm"  # 推断基因调控网络使用的模型方法
peak_to_gene_method <- "Signac"  # 将 peak 映射到基因的方法

# -----------------------------
# 3. 提取 SeuratPlus 对象中的 GRN 参数和信息
# -----------------------------
example_params <- Params(example_test_srt)          # 获取对象参数，包括 RNA 和 peak assay 名称
example_motif2tf <- NetworkTFs(example_test_srt)   # 获取转录因子（TF）列表
example_gene_annot <- Signac::Annotation(example_test_srt[[example_params$peak_assay]])  # 获取 peak 对应基因注释
example_genes <- VariableFeatures(example_test_srt, assay = example_params$rna_assay)     # 获取 RNA 高变基因

# -----------------------------
# 4. 构建 RNA 和 peak 数据矩阵
# -----------------------------
example_gene_data <- Matrix::t(Seurat::GetAssayData(
    example_test_srt, assay = example_params$rna_assay
))  # 转置 RNA counts 矩阵（行 = 细胞，列 = 基因）
gene_groups <- TRUE  # 可选：是否对基因分组（这里保留为 TRUE）

example_peak_data <- Matrix::t(Seurat::GetAssayData(
    example_test_srt, assay = example_params$peak_assay
))  # 转置 ATAC-seq peak counts 矩阵（行 = 细胞，列 = peak）
peak_groups <- TRUE  # 可选：是否对 peak 分组

# -----------------------------
# 5. 筛选 gene 注释
# -----------------------------
# 只保留 RNA 高变基因且在 peak 注释中出现的基因
example_features <- intersect(example_gene_annot$gene_name, example_genes) %>%
                    intersect(rownames(GetAssay(example_test_srt, example_params$rna_assay)))
example_gene_annot <- example_gene_annot[example_gene_annot$gene_name %in% example_features, ]

# -----------------------------
# 6. 提取 GRN 的网络信息
# -----------------------------
example_regions <- NetworkRegions(example_test_srt)  # 获取 Regions 对象（包含 peak 范围和 motif）
example_peak_data <- example_peak_data[, example_regions@peaks]  # 只保留 GRN 中的 peaks
colnames(example_peak_data) <- rownames(example_regions@motifs@data)  # 将列名改为 motif 名称
example_peaks2motif <- example_regions@motifs@data  # peak 对应 motif 的数据

example_regions_ranges <- example_regions@ranges  # GRanges 对象，包含 peak 的位置信息
example_anot <- example_gene_annot                 # 方便后续使用的基因注释

# -----------------------------
# 7. 找每个 peak 附近的基因
# -----------------------------
example_peaks_near_gene <- find_peaks_near_genes(
    peaks = example_regions@ranges,
    method = peak_to_gene_method,
    genes = example_gene_annot,
    upstream = 1e+05,      # 向上游 100kb
    downstream = 0,        # 下游 0
    only_tss = FALSE       # 是否只考虑 TSS
)

# 查看矩阵列名（基因名）
colnames(example_peaks_near_gene)

# -----------------------------
# 8. 聚合 peak 到基因
# -----------------------------
example_peaks2gene <- aggregate_matrix(
    t(example_peaks_near_gene),          # 转置矩阵，使行 = peak，列 = 基因
    groups = colnames(example_peaks_near_gene),  # 以基因名分组
    fun = "sum"                          # 聚合方法：求和
)
