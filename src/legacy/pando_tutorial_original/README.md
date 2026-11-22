# Pando GRN 分步教程

这套教程对应本项目已运行成功的论文方法实现。计算脚本仍在项目根目录，教程负责解释输入、算法、运行顺序和验收；不复制一套容易与正式代码分叉的拟合实现。

**方法边界：**在用户数据上复现论文框架；上游参考标签经用户同意用 WGCNAcluster 替代不可获得的 MainCellType。当前软件版本、注释匹配和秩亏情况见 `../METHODS.md`、`../results/sessionInfo.txt`。最终 `GRN_results.rds` 是普通 R list，不是原生 Pando Seurat/GRN 容器。拟合阶段显式实现论文的 Gaussian GLM，而非调用当前版本的一键默认流程。

| 顺序 | 教程 | 正式实现 |
|---|---|---|
| 00 | [环境、输入与运行约定](00_setup.md) | audit_inputs.R、record_environment.R |
| 01 | [候选区域与特征资源](01_resources.md) | prepare_resources.R、prepare_feature_sets.R |
| 02 | [fragments 与 GeneActivity](02_gene_activity.md) | extract_fragments.py、prepare_inputs.R |
| 03 | [CCA 与 MCMF 跨模态匹配](03_matching.md) | prepare_cca.R、match_cells.py |
| 04 | [表达排名与多项式 LASSO 质控](04_quality_control.md) | train_glycolysis_qc.R |
| 05 | [metacell、CSS 与 pseudocell](05_pseudocells.md) | coarse_grain.R |
| 06 | [motif 背景与区域扫描](06_motifs.md) | prepare_motif_background.R、scan_motifs.R |
| 07 | [逐基因 Gaussian GLM](07_glm.md) | fit_grn.R |
| 08 | [显著性检验、BH 与网络筛选](08_significance.md) | export_grn.R |
| 09 | [读取 RDS 与使用 regulon](09_use_results.md) | tutorials/read_results.R |

## 如何使用

阅读教程无需计算。所有 Linux、R、Python 命令都通过 Windows PowerShell 中的 SSH 执行：

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh list'
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh inspect'
```

`list` 列出步骤，`inspect` 只读取现有结果并验证网络结构，不重跑模型。只有显式指定步骤才启动该计算，例如：

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh motifs'
```

入口激活 spatialz，R 显式使用服务器已有依赖环境的 Rscript。每次执行产生独立的 `logs/tutorial_<步骤>_<时间>_<PID>/`，包含 stdout、stderr、exitcode、代码快照；SSH 返回相同退出码。失败先读 stderr/traceback，修复对应根目录脚本后重跑同一步。入口不自动重跑上游。

输入文件和大型中间矩阵保留在服务器。本教程依赖已审计的项目资源，不是从零下载全部数据的一键安装器。已有检查点会被原脚本复用；改变参数或输入后必须评估并重建受影响的下游，不能把旧检查点当成新结果。历史 `continue_paper.sh` 用于原运行续跑；本教程入口使用独立日志，不改变历史成功状态。
