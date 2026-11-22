# 分析方法及可复现记录

本项目在用户提供的数据上复现 Fleck 等论文的 Pando GRN 推断框架。

**已获用户明确接受的差异：**上游 glycolysis 质控使用公开 Nowakowski 数据的 WGCNAcluster 标签，替代作者未公开参考对象中的 MainCellType；不能称为全部历史输入完全相同的复现。

## 输入与匹配

- 项目根目录 `/home/wwd/py_code/mhr/final/new_grn/`；输入根目录由 PANDO_DATADIR 指定，默认 `/home/wwd/py_code/mhr/final`。
- 原始 ATAC 对象 38,957 个细胞；实测 RNA 矩阵 50,000 个细胞。已恢复的 peak 矩阵与输入对象的全部细胞名、peak 行名核对。
- 使用用户提供的 timecourse_fragments.tar.gz 和对应索引，按样本通过 Signac::GeneActivity 默认参数计算基因活性；仅作为跨模态共享特征，不作为实测 RNA。
- RNA 与基因活性分别 LogNormalize(scale.factor=10000)，在相同 age 和 line 内，通过 SelectIntegrationFeatures 选择 2,000 个特征，ScaleData 后 RunCCA。
- 原作者 MCMF：k=10、null_cost_percentile=99、capacity_method=uniform。仅适配 SciPy 将 n_jobs 改名为 workers 的接口；核心匹配代码保持不变。
- 按较小模态的细胞聚合匹配关系，各模态存在多个成员时取算术均值。缺少 line 的细胞单独记录并排除。极小组按可计算秩限制 CCA 维度。

## 质控与降噪

- 参考来源：https://cells.ucsc.edu/cortex-dev/ 。表达矩阵和 WGCNAcluster 注释在 resources/nowakowski/。
- 对共同基因计算每个细胞内的表达排名；glmnet 多项式 LASSO，alpha=1；种子 42，10 折交叉验证，lambda.1se。并行仅改变计算调度；折划分写入 qc/reference_CV_folds.csv。
- 在 metacells 上使用同一基因顺序和排名定义预测类别，排除预测为 Glyc 的 metacells。预测标签、使用的标签字段和排除标记均导出。
- 作者 CSS 源码整合细胞系；前 20 PCs 构建邻域。按作者 pseudocells.R 的 Bernoulli(0.3) 选择种子，k=10；多个 territory 的细胞随机选择一个，未覆盖细胞保留单独 territory。
- RNA 取算术均值；ATAC 二值化后取均值，得到概率。对 territory peak counts 做 TF-IDF/LSI，前 20 LSI、Louvain resolution=100；再将 ATAC 概率按簇平均，回填到各 pseudocell。

## 候选区域与回归

- 补充表 2 的 4,342 个 GRN features；使用 Pando 附带的作者扩展 motifs/motif2tf，而非仅用 JASPAR。
- 靶基因须与 EnsDb v86 的 gene_name 匹配；调控因子采用 RNA+motif 证据选择，不需要其自身基因坐标。AC023509.3、TBXT、AC073111.5、NSD2 四个候选 TF 的名称未匹配 v86 靶基因名称，审计结果单独导出。
- 将 UCSC hg38 30way PhastCons 非外显子区域与 ENCODE cCRE 合并，并与 ATAC peaks 相交。GeneActivity 注释使用用户的 final_geneGr.rds；gene body 使用论文指定的 EnsDb.Hsapiens.v86。
- motif 背景频率按全部候选区域计算（motifmatchr 的 subject 定义），不随入模区域筛选或并行分块改变；p.cutoff=5e-5。区域分块仅用于并行扫描。
- Gene body 加按链方向的 TSS 上游 100 kb；peak 最大簇可及性>0.1（作者拟合脚本规则）；绝对 peak–target 相关>0，绝对 TF–target 相关>0.1，排除自身 TF。
- 以 log-normalized 靶基因表达为响应，以 TF 表达×区域可及性为交互预测变量，保留截距。使用 stats::glm(family=gaussian(link=identity))，不进行自定义 QR、正则化或预测变量缩放。
- 使用作者 summary(glm()) 的系数 t 检验。对单个可估计系数，这与对应一自由度 partial F 检验等价。秩亏、不可估计系数、残差自由度和失败模型保存到诊断。
- 合并所有可估计区域系数的 P 值后统一 BH 校正，阈值 FDR<0.05。TF–gene 效应为显著区域系数之和；pval_min/padj_min 是区域检验摘要，不是新的 TF–gene 联合检验。

## 执行与输出

所有 R/Python、数据处理和建模命令仅通过 SSH alias 在 l40 执行。激活 spatialz；R 显式使用已有 seurat4_env_restored/bin/Rscript。软件版本与论文历史版本不同，具体见审计/sessionInfo。阶段 stdout、stderr、exitcode、代码快照均保存。

入口为 continue_paper.sh；结果目标目录为 results/。只有 export_grn 的退出码为 0 且最终文件通过检查后，才视为完成。

来源：

- https://www.nature.com/articles/s41586-022-05279-8
- https://github.com/quadbio/organoid_regulomes
- https://github.com/quadbio/simspec
- https://github.com/quadbio/Pando
