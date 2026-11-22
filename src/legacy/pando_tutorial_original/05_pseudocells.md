# 05 metacell、CSS 与 pseudocell

输入：匹配对、实测 RNA、原 ATAC counts、质控分类器和作者 CSS/pseudocell 源码资源。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh pseudocells'
```

`coarse_grain.R` 依次完成：

1. 在每组按较小模态细胞聚合配对，成员的 log-normalized 实测 RNA 取算术均值，保存 matched_metacells 检查点。
2. 用参考分类器的相同表达排名预测 metacell 类别，排除 Glyc。
3. 作者 CSS 整合细胞系；前 20 PCs 构图，Bernoulli(0.3) 选种子、k=10 构建 territory；重叠成员随机分配，未覆盖者保留 singleton。
4. RNA 均值；ATAC 二值化后均值成为概率；TF-IDF/LSI 前 20 维、Louvain resolution=100 得到 accessibility clusters，将簇均值概率回填各 pseudocell。

输出：`coarse/metacell_membership.tsv`、`metacell_QC.csv`、`pseudocell_membership.tsv`、`accessibility_clusters.csv`、`paper_model_inputs.rds`。

本次 27,427 个 matched metacells → 排除 5,421 → 保留 22,006 → 6,612 pseudocells、676 accessibility clusters。

验收：模型输入的 RNA 与 peak_probability 列名、顺序完全相同；RNA 是测得的表达，ATAC 是概率。metacell 与 pseudocell 是先后两级聚合，不能互换术语。原脚本包含缓存，改变上游匹配后不能继续复用旧 matched_metacells。
