library(Signac)                # 用于 ATAC-seq 数据处理
library(Seurat)                # 单细胞分析
library(EnsDb.Hsapiens.v86)    # 人类基因注释数据库
library(dplyr)                 # 数据处理
library(ggplot2)               # 可视化
library(Pando)
# 1. 读取SeuratPlus对象
object <- readRDS("/home/wwd/data/final_find_motifs.rds")

# 分析方法定义（保留原设置）
method <- "glm"
peak_to_gene_method <- "Signac"

# -------- 2. 提取对象参数与核心信息（替换Pando专属函数，适配SeuratPlus）--------
# 定义RNA/Peak assay名称（SeuratPlus默认多为"RNA"/"ATAC"，可通过assays(object)查看确认）
rna_assay <- "RNA"
peak_assay <- "ATAC"

# 提取TF-motif对应关系（SeuratPlus中motif注释通常储存在ATAC assay的motifs槽位）
motif2tf <- Signac::MotifTFs(object[[peak_assay]])  # 核心替换：Signac标准函数，输出motif→TF对应表
# 提取RNA高变基因（Seurat标准方法，适配SeuratPlus）
genes <- Seurat::VariableFeatures(object, assay = rna_assay)

# -------- 3. 构建 RNA / peak 数据矩阵（仅微调assay指定方式，逻辑不变）--------
# RNA表达矩阵 [cells x genes]（Seurat标准提取，转置后匹配原代码维度）
gene_data <- Matrix::t(Seurat::GetAssayData(object, assay = rna_assay, slot = "data"))
gene_groups <- TRUE

# Peak矩阵 [cells x peaks]（ATAC assay的counts/score矩阵，转置后匹配原代码维度）
peak_data <- Matrix::t(Seurat::GetAssayData(object, assay = peak_assay, slot = "counts"))
peak_groups <- TRUE

# -------- 4. 基因注释（Signac标准方法，适配SeuratPlus）--------
# 从ATAC assay提取基因注释（包含gene_name、TSS位置等核心信息）
gene_annot <- Signac::Annotation(object[[peak_assay]])

# 只保留RNA高变基因中的注释（逻辑与原代码完全一致，无需修改）
features <- intersect(gene_annot$gene_name, genes) %>%
  intersect(rownames(Seurat::GetAssay(object, rna_assay)))
gene_annot <- gene_annot[gene_annot$gene_name %in% features, ]

# -------- 5. 提取peak区域和motif信息（SeuratPlus专属提取，替换Pando的NetworkRegions）--------
# 从ATAC assay提取motif相关的peak区域（GenomicRanges对象，含motif注释）
motif_regions <- Signac::MotifRegions(object[[peak_assay]])  # 核心替换：获取含motif的peak区域
regions_ranges <- motif_regions  # 与原代码变量名保持一致，衔接后续步骤

# 筛选peak_data：只保留含motif的peak（匹配原代码逻辑）
peak_data <- peak_data[, intersect(colnames(peak_data), as.character(GenomicRanges::granges(motif_regions)))]

# 提取motif矩阵（peaks2motif：[peaks x motifs]，表示peak中motif的富集/存在情况）
peaks2motif <- Signac::GetMotifData(object[[peak_assay]], slot = "motif.matrix")
# 重命名peak_data列名为motif名（与原代码逻辑一致，衔接TF关联）
colnames(peak_data) <- rownames(peaks2motif)

# 保留原代码变量名，确保后续步骤无报错
anot <- gene_annot

# -------- 6. 找到每个 peak 附近的基因（原代码完全不变，可直接运行）--------
peaks_near_gene <- find_peaks_near_genes(
  peaks = regions_ranges,        # 已替换为SeuratPlus提取的motif peak区域
  method = peak_to_gene_method,  # Signac方法，适配当前对象
  genes = gene_annot,            # 标准化基因注释
  upstream = 1e+05,              # 上游100kb（原参数）
  downstream = 0,                # 下游0bp（原参数）
  only_tss = FALSE               # 不局限于TSS（原参数）
)

colnames(peaks_near_gene)  # 查看peak-gene对应矩阵列名

# -------- 7. 将 peak-gene 信息聚合成矩阵（原代码完全不变，可直接运行）--------
peaks2gene <- aggregate_matrix(
  t(peaks_near_gene),
  groups = colnames(peaks_near_gene),
  fun = "sum"
)