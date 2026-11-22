# 03 CCA 与 MCMF 匹配

输入：实测 RNA counts、ATAC GeneActivity、两模态的 age/line 元数据。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh cca'
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh matching'
```

`prepare_cca.R` 在相同 age × line 内分别 LogNormalize(scale.factor=10000)，选择 2,000 个整合特征，ScaleData 后 RunCCA，默认 30 个分量。day16 Hoik1 极小组仅有 20 个 RNA 细胞，维度限制为 19，不能强行求 30 个分量。

`match_cells.py` 使用保存的作者 MCMF 源码：k=10、null_cost_percentile=99、uniform capacity。唯一接口适配是 SciPy 的 n_jobs 改为 workers。较大模态流向较小模态，空匹配不进入最终配对。

输出：`cca/<group>.tsv`、`cca/excluded_cells.csv`、`matches/<group>.tsv`、`matches/all_pairs.tsv`。本次 36 组，47,731 对；缺少 line 的 9,501 个 RNA 和 75 个 ATAC 细胞排除。

验收：每对的 age 和 line 一致；配对细胞均属于原始模态。一个细胞可能有多个匹配成员，不能把配对行数当成独立 metacell 数。旧分组 TSV 会被复用，改分组规则后需重新建立受影响的匹配。
