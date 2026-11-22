# 08 显著性检验与筛选

输入：全部逐基因 fits，以及注释审计 `audit/TFs_without_v86_annotation.csv`。运行前先确认拟合无 error。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh export'
```

正式实现 `export_grn.R` 从 summary(glm()) 提取系数、标准误、t 统计量和双侧 P 值。对于单个可估计系数，该 t 检验等价于一自由度 partial F 检验。秩亏项没有有效系数检验，不当作零效应填入。

核心筛选逻辑：

```r
# co 是全部模型合并后的区域系数表；valid 标记有效推断
co$padj[valid] <- p.adjust(co$pval[valid], method = "BH")
sig <- co[!is.na(co$padj) & co$padj < 0.05, ]
```

BH 的检验族是全部可估计 TF–region–gene 系数，不是按基因单独校正。本次 1,141,828 次检验 → 62,354 条显著区域关系 → 49,764 条 TF–gene 关系，涉及 738 个 TF、3,383 个靶基因。

TF–gene 汇总将同一组合的显著区域系数相加；n_regions 记录区域数，mixed_region_signs 标记区域正负混合。pval_min/padj_min 仅为区域检验摘要，不能称作 TF–gene 联合检验，也不能据其进行新的独立边级 FDR 声明。系数方向是模型条件关联，不单独构成因果证据。

输出：`results/GRN_results.rds`、完整/显著区域表、`TF_gene_network.csv`、`model_QC.csv`、`target_status.csv`、gene modules 和 QC 图。重新 export 会更新结果目录；默认沿用 FDR<0.05。

验收：无 error 状态；模型收敛且残差自由度>0；有效系数的 P 值、标准误、统计量有限；TF–gene 无重复和 self edge；详见结果报告。
