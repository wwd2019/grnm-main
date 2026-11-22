# 02 fragments 与 GeneActivity

目标：从真实 ATAC fragments 提取跨模态共享的基因特征。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh fragments'
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh activity'
```

fragments 步骤读取大体积归档，通常无需重跑。解压脚本拒绝越界路径及链接，并复用同大小文件；同大小并非内容校验，资源身份仍应结合已有哈希和审计。

正式代码 `prepare_inputs.R` 按 batch 建立命名条形码到原始条形码的映射，通过 CreateFragmentObject 验证 fragments，构建带注释的 ChromatinAssay，调用 Signac::GeneActivity 默认参数。逐 batch 保存检查点，再按原 ATAC 列顺序合并。

输入：审计元数据、恢复的 peak counts、final_geneGr、每批 `fragments.tsv.gz` 和 `.tbi`。

输出：`inputs/<batch>.GeneActivity.rds`、`inputs/ATAC_gene_activity.rds`、`inputs/GeneActivity_QC.csv`。本次合并 19,607 基因 × 38,957 细胞。

验收：全部细胞活动总量大于零；合并列名与原始 ATAC 身份完全相同。此矩阵只进入 CCA，不进入最终模型作为 RNA 响应。
