# 04 表达排名与多项式 LASSO 质控

输入：`resources/nowakowski/exprMatrix.tsv.gz`、`meta.tsv` 和实测 RNA 基因名。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh qc'
```

正式代码 `train_glycolysis_qc.R`：取共同基因，对每个细胞内部的表达做平均并列排名；glmnet multinomial LASSO，alpha=1，seed=42，固定 10 折，使用 lambda.1se。折分配导出，便于复核。参考为 4,261 细胞、20,477 个共同基因，本次 lambda.1se=0.002854229。

输出：`qc/nowakowski_rank_lasso.rds`、`qc/reference_CV_folds.csv`、`qc/WGCNAcluster_authorization.txt`。

这一步训练参考分类器；下一步在 metacells 上预测并排除 Glyc 类。预测必须用相同共同基因顺序和细胞内排名，不能直接输入原始 counts。

**已接受的方法差异：**公开 WGCNAcluster 与作者 MainCellType 不是同一层级；本教程保持排名与多项式 LASSO 方法，但不声称质控标签与原作者完全相同。输出模型携带实际标签字段与许可标记。
