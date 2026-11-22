# convert_atac_rds.R
# 将 scATAC 基因级 PCA 数据从 RDS 格式转换为 CSV
# 输入: C:\Users\25439\Desktop\GSE262682_RAW\scATAC\
# 输出: 项目 data/ 目录下的 CSV 文件
#
# 用法: Rscript convert_atac_rds.R

# --- 配置路径 ---
rds_dir <- "C:/Users/25439/Desktop/GSE262682_RAW/scATAC"
out_dir <- "F:/pycharm_professional/py_code/dongyifei_final/grnm-0.3/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- 读取 RDS 文件 ---
# scATAC_genes_pca_64d.rds: scATAC 基因级 PCA (64维)
# scATAC_genes_pca_64d2.rds: scATAC 基因级 PCA v2

cat("读取 ATAC PCA RDS 文件...\n")

atac_pca <- readRDS(file.path(rds_dir, "scATAC_genes_pca_64d.rds"))
cat("scATAC_genes_pca_64d.rds 加载完成\n")
cat("  类型:", class(atac_pca), "\n")
cat("  维度:", dim(atac_pca), "\n")
cat("  前几行:\n")
print(head(atac_pca)[, 1:min(5, ncol(atac_pca))])

atac_pca2 <- tryCatch({
    readRDS(file.path(rds_dir, "scATAC_genes_pca_64d2.rds"))
}, error = function(e) {
    cat("scATAC_genes_pca_64d2.rds 读取失败:", e$message, "\n")
    NULL
})

# --- 处理并保存 ---
# 根据 RDS 对象类型转换为 data.frame
save_rds_as_csv <- function(obj, out_path, label) {
    if (is.null(obj)) return()

    if (is.matrix(obj)) {
        df <- as.data.frame(obj)
    } else if (inherits(obj, "dgCMatrix") || inherits(obj, "Matrix")) {
        df <- as.data.frame(as.matrix(obj))
    } else if (is.data.frame(obj)) {
        df <- obj
    } else if (is.list(obj)) {
        # 可能是 Seurat 对象或 list，尝试提取
        cat("  ", label, "是 list, 元素名:", names(obj), "\n")
        # 尝试常见的元素名
        for (el in names(obj)) {
            if (is.matrix(obj[[el]]) || is.data.frame(obj[[el]])) {
                cat("    发现矩阵/数据框元素:", el, " 维度:", dim(obj[[el]]), "\n")
            }
        }
        return()
    } else {
        cat("  ", label, "未知类型:", class(obj), "\n")
        return()
    }

    # 将行名作为 gene_id 列
    df <- cbind(gene_id = rownames(df), df)
    rownames(df) <- NULL

    # 列名处理
    if (ncol(df) > 2) {
        pc_cols <- grep("^PC|^PC_|^pc", colnames(df), value = TRUE)
        if (length(pc_cols) == 0) {
            # 如果没有 PC 前缀列，假设第一列是 gene_id, 其余是 PC
            colnames(df)[-1] <- paste0("PC", seq_len(ncol(df) - 1))
        }
    }

    write.csv(df, out_path, row.names = FALSE)
    cat("  已保存:", out_path, " (", nrow(df), "genes ×", ncol(df)-1, "PCs)\n")
}

cat("\n保存 CSV 文件...\n")
save_rds_as_csv(atac_pca, file.path(out_dir, "atac_pca.csv"), "atac_pca")
if (!is.null(atac_pca2)) {
    save_rds_as_csv(atac_pca2, file.path(out_dir, "atac_pca_v2.csv"), "atac_pca2")
}

cat("\n转换完成!\n")
