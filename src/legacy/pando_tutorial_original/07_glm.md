# 07 逐基因 Gaussian GLM

输入：`coarse/paper_model_inputs.rds`、`motifs/paper_motifs.rds`、EnsDb v86 坐标。

```powershell
ssh alias 'bash /home/wwd/py_code/mhr/final/new_grn/tutorials/run_step.sh glm'
```

对每个靶基因，候选区域为 gene body 加按链方向的上游 100 kb。模型输入已筛选最大簇可及性>0.1 的 peaks；入模再要求绝对 peak–target 相关>0、绝对 TF–target 相关>0.1，排除自身 TF，并要求对应 motif 命中。

模型为：

```r
# 示意公式；完整变量构建和检查点逻辑见 ../fit_grn.R
fit <- stats::glm(
  response ~ peak1:tf1 + peak2:tf1 + peak2:tf2,
  data = data, family = gaussian(link = "identity")
)
summary(fit)$coefficients
```

每个预测变量是 TF 表达 × 区域可及性概率，保留截距，不另加单独 TF/peak 主效应，不缩放交互项，不做 ridge/LASSO。peak1/tf1 是安全变量别名，导出时还原真实 TF 和区域。

输出：`fits/<编号>_<gene>.rds` 和 `fits/attempted_targets.txt`。单基因原子保存，已有非 error 检查点跳过；error 可重试。不要在改变模型后混用旧 fits。

本次 3,846 靶基因尝试，3,446 拟合；386 无相关 TF、9 无 motif 区域、4 恒定响应、1 全部交互项不可估计。3,125 个成功模型含别名项，按 R 默认 QR 保留可估计系数并记录 rank、aliased、residual_df、converged。0 个推断不可用模型。

运行可能耗时且内存占用高。入口默认沿用 GRN_FIT_CORES=8；历史反向辅助 worker 仅是调度加速，不是教程必需步骤，不应同时重复启动。
