# GRN 推断结果

本次使用论文 Pando 框架，在用户提供的数据上完成推断。
**经用户接受的方法差异：上游 glycolysis 质控使用公开 WGCNAcluster 标签替代原作者 MainCellType。**

- 匹配聚合 metacells：27427；预测为 glycolysis 并排除：5421；保留：22006。
- 目标基因尝试：3846；成功拟合：3446；统计推断不可用模型：0。
- 原文 GRN features：4342；EnsDb v86 无对应基因名的 features：496（另表记录）。
- 含别名项的模型：3125；按 R 默认 QR 处理，仅保留 summary(glm()) 返回的系数，诊断记录在 model_QC.csv。
- 可估计区域系数检验：1141828；FDR<0.05 的 TF-region-gene 关系：62354。
- TF-gene 关系：49764；TF：738；靶基因：3383。

Gaussian identity GLM，预测变量为 TF 表达×区域可及性，保留截距、不缩放交互项。所有可估计系数统一进行 BH 校正。秩亏及不可估计项保留在模型诊断中。

TF_gene_network.csv 的 estimate 为该 TF-target 显著区域系数之和；pval_min/padj_min 是区域检验的摘要，不是新的联合检验。mixed_region_signs 标记区域效应方向不同的 TF-target 组合。

靶基因建模要求 v86 基因名与 RNA 名称匹配；调控因子依据 RNA 表达和 motif 映射选择，其自身坐标不用于建立预测变量。4 个候选调控因子的名称未在 v86 靶基因名称中匹配，另表列出。

完整原始系数：all_TF_region_gene_coefficients.tsv.gz；显著关系：significant_TF_region_gene.tsv.gz；模型诊断：model_QC.csv 和 target_status.csv；正负 gene modules：gene_modules.gmt；R 对象：GRN_results.rds；总览图：QC_overview.png。

输入、特征、软件环境及与论文的其他已记录差异见项目 METHODS.md；运行日志、退出码及代码快照保存在 logs/ 和 runs/。

总览图中系数直方图的横轴使用 signed log10(1+|coefficient|)，包含全部显著系数；原始未缩放系数保存在结果表中。
