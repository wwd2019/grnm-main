# 00 环境与输入

目标：确认计算地点、真实模态和文件身份。

项目为 `/home/wwd/py_code/mhr/final/new_grn`。默认数据根为 `/home/wwd/py_code/mhr/final`；部分历史脚本固定使用该路径，因此本教程也固定此项目，不承诺仅改 PANDO_DATADIR 即可迁移整个流程。

| 输入 | 用途 |
|---|---|
| ATAC_all_merged_srt.rds | 原始 ATAC 元数据与细胞身份 |
| RNA_all_merged_srt.rds、rnaMatrix.rds | 实测 RNA 元数据和 counts |
| final_geneGr.rds | GeneActivity 注释、去除保守区域中的外显子 |
| atacrowname.rds | peaks 顺序核对 |
| pando_zenodo15371701_from_atac_obj/peakMatrix_recovered.rds | 已恢复并完成全行列身份核对的 ATAC counts |
| /home/wwd/data/timecourse_fragments.tar.gz | 原始 fragments 和索引 |

已审计 RNA 50,000 个细胞、ATAC 38,957 个细胞；两者不是配对 multiome，不能直接按条形码拼接。GeneActivity 仅用于匹配，模型的表达来自实测 RNA。

执行独立结果检查：

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh inspect'
```

环境记录见 `results/software_versions.csv`、`results/execution_environment.txt` 和 `results/sessionInfo.txt`。spatialz 不具备本项目所需 R 包，入口激活 spatialz 后使用 `/home/wwd/miniconda3/envs/seurat4_env_restored/bin/Rscript`。所有计算都发生在 l40。

验收：输入细胞顺序和 peaks 顺序一致；参考为 hg38；不能将输入中的未知来源 RNA assay 或历史表达迁移结果替代实测 RNA。
