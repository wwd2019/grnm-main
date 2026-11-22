# 06 motif 背景与扫描

输入：候选先验、全部原 peaks、hg38 序列、pseudocell 模型输入、扩展 motif 与 TF 映射。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh background'
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh motifs'
```

先将全部 peaks 与先验相交，按 motifmatchr subject 定义计算共同背景频率。后续按模型保留区域扫描时继续使用同一个背景；并行分块不能各自估计背景。阈值 p.cutoff=5e-5，这是 motif 匹配阈值，不是 GRN 的 FDR。

正式实现：`prepare_motif_background.R` 和 `scan_motifs.R`。扫描保留 region→原 peak 的对应关系：序列扫描发生在相交区域，而交互项可及性来自对应原 peak。

输出：`resources/paper_motif_background.rds`、`motifs/paper_motifs.rds`、`motifs/region_to_peak.csv`。

本次背景由 749,658 个候选区域计算；最终扫描 245,710 个区域、1,122 个 motifs、805 个 TF，得到 7,600,744 个命中。验收区域顺序与矩阵行名一致，motif 名称可映射到 TF，映射 TF 出现在 RNA 中。
