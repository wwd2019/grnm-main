# 01 候选区域与特征资源

输入：`resources/phastConsElements30way.txt.gz`、Pando 附带的扩展 motifs/motif2tf、cCRE、EnsDb.Hsapiens.v86、用户 final_geneGr，以及 `reference/supplementary/41586_2022_5279_MOESM4_ESM.xlsx`。这些资源已存在于服务器；来源及版本边界见 METHODS.md 和资源清单。

步骤：将 UCSC 的零起点区间转换为 GRanges 的一起点区间，从 30way PhastCons 中减去外显子，再与 ENCODE cCRE 合并；保存 v86 基因坐标；解析补充表 2 的 GRN features。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh resources'
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh features'
```

输出：`resources/paper_candidate_prior_regions.hg38.rds`、`EnsDb.Hsapiens.v86.genes.hg38.rds`、`motifs.rds`、`motif2tf.rds`、`grn_features.rds`。

验收：GRN features 4,342 个；最终 496 个名称无法匹配 v86 靶基因名称，必须保留审计而非默默换注释。不要直接用当前 Pando 的默认保守区和默认基因注释替代论文指定资源。资源历史版本不完全一致的问题保留在 provenance 中。
