# symbolname <- read.csv("D:/GRN/final/data/changesymbolname.csv")
# genename <- symbolname[["gene_id"]]
# symbolname <- symbolname[["SYMBOL"]]
#
# annotations <- readRDS("D:/GRN/final/data/geneGr.rds")
# symbol <- annotations@elementMetadata@listData[["symbol"]]
#
#
# i<- 1
# for(i in c(1 : length(genename))){
#   symbol[symbol == genename[i]] = symbolname[i]
#   # rnaname[num] <- symbolname[i]
# }
#
# a = "1"
# b <-c("1","1","3")
#
# summary(genename[i] == symbol)
#
# annotations@elementMetadata@listData[["gene_name"]] <- symbol
# annotations
#
# saveRDS(annotations,"D:/GRN/final/data/final_geneGr.rds")
# ------------------------------
# 1️⃣ 读取基因 ID 与 SYMBOL 对照表
# ------------------------------
symbolname <- read.csv("/home/wwd/data/changesymbolname.csv")

# 提取原始 gene_id 列
genename <- symbolname[["gene_id"]]

# 提取对应的 SYMBOL 列
symbolname <- symbolname[["SYMBOL"]]

# ------------------------------
# 2️⃣ 读取已有的 gene graph 对象（Bioconductor 的 GRanges 对象）
# ------------------------------
annotations <- readRDS("/home/wwd/data/geneGr.rds")

# 提取 GRanges 对象中元数据的 symbol 列（当前基因名）
symbol <- annotations@elementMetadata@listData[["symbol"]]

# ------------------------------
# 3️⃣ 循环替换 symbol
# ------------------------------
# 将 genename 对应的 symbol 替换为新的 SYMBOL
for(i in 1:length(genename)){
  symbol[symbol == genename[i]] <- symbolname[i]
}

# ------------------------------
# 4️⃣ 可选调试：检查最后一个替换是否成功
# ------------------------------
# summary(genename[i] == symbol)  # i 此时为 length(genename)
# 用于检查最后一个基因 ID 是否已经替换成功

# ------------------------------
# 5️⃣ 更新 annotations 对象中的 gene_name 字段
# ------------------------------
annotations@elementMetadata@listData[["gene_name"]] <- symbol
if ("gene_name" %in% colnames(mcols(annotations))) {
  mcols(annotations)$gene_id <- paste0(mcols(annotations)$gene_name, "_gene", 1:length(annotations))
}
if ("gene_name" %in% colnames(mcols(annotations))) {
  mcols(annotations)$tx_id <- paste0(mcols(annotations)$gene_name, "_tx", 1:length(annotations))
}
mcols(annotations)$gene_biotype <- "protein_coding" # 基因生物类型
mcols(annotations)$type <- "gene" # 固定值，不可修改

# 查看更新后的 annotations
annotations

# ------------------------------
# 6️⃣ 保存修改后的 GRanges 对象
# ------------------------------
saveRDS(annotations, "/home/wwd/data/final_geneGr.rds")
