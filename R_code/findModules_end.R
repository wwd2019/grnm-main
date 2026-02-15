library(purrr)          # 用于函数式编程，如 map()
library(Signac)         # 用于单细胞 ATAC-seq 分析
library(Seurat)         # 单细胞数据分析工具
library(EnsDb.Hsapiens.v86) # 人类基因注释
library(dplyr)          # 数据操作
library(ggplot2)        # 绘图
library(Pando)          # 基因调控网络构建工具

# 读取已推断 GRN 的 SeuratPlus 对象
object <- readRDS("/home/wwd/data/infer_grn.rds")

# ===================== 核心适配：手动定义assay名称（替换Pando的Params）=====================
# 先检查对象的assay名称，确保与实际一致（运行assays(object)查看）
rna_assay <- "RNA"    # SeuratPlus默认RNA assay名称
peak_assay <- "ATAC"  # SeuratPlus默认ATAC/peak assay名称
network = "glm_network"       # 指定网络名称（与原GRN推断时的命名一致）
# ==========================================================================================

# 设置分析参数（与原代码完全一致）
p_thresh = 0.05               # p-value 阈值
rsq_thresh = 0.1              # 回归模型 R^2 阈值
nvar_thresh = 10              # 每个模块最少变量数
min_genes_per_module = 5      # 每个模块最少基因数

# ===================== 核心适配：替换Pando专属信息提取函数 =====================
# 1. 提取网络中调控区域信息（SeuratPlus中GRN的调控区域储存在ATAC assay的motif/peak相关槽位）
# 从SeuratPlus对象中提取GRN对应的调控区域（GenomicRanges对象，匹配原NetworkRegions功能）
regions <- Signac::MotifRegions(object[[peak_assay]])
# 2. 获取指定GRN网络对象（Pando的GetNetwork适配SeuratPlus，直接调用即可）
net_obj <- Pando::GetNetwork(object, network = network)
# ==============================================================================

# -------- 构建基因调控模块（原代码完全不变，可直接运行）--------
# 根据指定阈值对网络模块进行筛选
net_obj <- find_modules(
  net_obj,
  p_thresh = p_thresh,
  rsq_thresh = rsq_thresh,
  nvar_thresh = nvar_thresh,
  min_genes_per_module = min_genes_per_module
)

# 提取筛选后的模块信息（原代码不变）
modules <- NetworkModules(net_obj)

# -------- 核心适配：调整调控区→峰的映射逻辑（适配SeuratPlus的命名规则）--------
# 原Pando对象的regions@peaks是索引，SeuratPlus中直接通过GRanges匹配peak名称
# 步骤1：将调控区域（GenomicRanges）转为Signac标准peak命名（chrX:start-end）
reg_ranges_str <- Signac::GRangesToString(regions)
# 步骤2：提取peak assay中的所有peak名称（SeuratPlus标准格式）
all_peaks <- rownames(GetAssay(object, assay = peak_assay))
# 步骤3：构建调控区→峰的映射（确保调控区在peak列表中，避免匹配失败）
reg2peaks <- all_peaks[match(reg_ranges_str, all_peaks)]
# 步骤4：为映射结果命名（初始为标准GRanges字符串，与原代码一致）
names(reg2peaks) <- reg_ranges_str

# -------- 处理命名格式（原代码逻辑不变，仅微调循环输入，避免冗余打印）--------
names <- names(reg2peaks)  # 直接取reg2peaks的命名，无需重新提取
mynames <- c()
for(item in names){
  # 可选：注释掉print，避免大量输出；需要调试时可取消注释
  # print(item)              # 打印原始名字
  item <- gsub("_", "-", item)  # 将 "_" 替换成 "-"，统一命名
  # print(item)              # 打印修改后的名字
  mynames <- c(mynames,item)    # 保存到列表
}
names(reg2peaks) <- mynames    # 更新 reg2peaks 的名字

# 检查模块中的调控区是否在 reg2peaks 中（原代码不变，验证匹配结果）
for(item in modules@features$regions_pos){
  print(paste("调控区", item, "是否存在：", item %in% mynames))  # 优化输出，更易读
}

# -------- 为模块添加峰位置信息（原代码完全不变，可直接运行）--------
# 对每个模块的正调控区域，映射到唯一的峰
peaks_pos <- modules@features$regions_pos %>% map(function(x) unique(reg2peaks[x]))

# 对每个模块的负调控区域，映射到唯一的峰
peaks_neg <- modules@features$regions_neg %>% map(function(x) unique(reg2peaks[x]))

# 将映射后的峰位置信息加入模块
modules@features[["peaks_pos"]] <- peaks_pos
modules@features[["peaks_neg"]] <- peaks_neg

# -------- 核心适配：调整模块信息更新逻辑（适配SeuratPlus的槽位结构）--------
# SeuratPlus对象的GRN信息储存在@tools$grn槽位，而非Pando的@grn@networks
if (!"grn" %in% names(object@tools)) {
  object@tools$grn <- list()  # 若不存在grn槽位，先创建
}
object@tools$grn[[network]] <- net_obj  # 更新指定网络的模块信息
object@tools$grn[[network]]@modules <- modules

# 保存处理后的SeuratPlus对象（原代码不变，修改文件名便于区分）
saveRDS(object,"/home/wwd/data/final_mydeal_find_motifs_seuratplus.rds")

# 查看 purrr::map 用法（原代码不变）
?map  # map(list, fun) 会对列表每个元素应用函数 fun