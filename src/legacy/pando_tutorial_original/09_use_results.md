# 09 读取 GRN RDS 与 regulon

服务器文件：`/home/wwd/py_code/mhr/final/new_grn/results/GRN_results.rds`。本地副本：`E:/main/new_grn/results/GRN_results.rds`。所有实际 R 操作仍通过 SSH 在服务器执行。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh inspect'
```

服务器 R 会话中的读取示例：

```r
grn <- readRDS("/home/wwd/py_code/mhr/final/new_grn/results/GRN_results.rds")
names(grn)
edges <- as.data.frame(grn$TF_gene_edges)
foxg1 <- edges[edges$tf == "FOXG1", ]
foxg1 <- foxg1[order(-abs(foxg1$estimate)), ]
head(foxg1)
positive_targets <- unique(foxg1$target[foxg1$direction == "positive"])
negative_targets <- unique(foxg1$target[foxg1$direction == "negative"])
```

| 成员 | 含义 |
|---|---|
| region_coefficients | 全部返回的区域系数，含 pval/padj 和 inference_valid；并非全显著 |
| TF_gene_edges | 筛选后按 TF–target 汇总的网络 |
| model_QC | 拟合模型的 rank、自由度、别名项、收敛与推断诊断 |
| gene_modules | TF__positive / TF__negative 对应的靶基因列表 |

这是普通结果 list，不能直接假定支持 Pando 对象的 GetGRN/Network 等接口。原始逐基因失败或跳过原因见 results/target_status.csv；不在显著边表中的基因不能直接解释为没有调控。

基因模块可用于后续富集或活性评分；正负模块分别处理。不要在不同靶基因间直接将原始系数绝对值视为统一强度，因为模型变量未缩放、各基因模型条件不同。后续扰动或命运验证需要额外分析。
