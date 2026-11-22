import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class tf_moudle:
    def __init__(self,tf_list,tf_mou,tf_ratio):
        self.tf_list = tf_list
        self.tf_mou = tf_mou
        self.tf_ratio = tf_ratio
    def plot_scatter(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.tf_mou, self.tf_ratio, s=100, alpha=0.5)
        plt.xlabel('TF Modules')
        plt.ylabel('Ratio of Target Genes in the Same Module')
        plt.title('Transcription Factor Regulation of Target Genes')
        plt.xticks(rotation=45)  # 旋转x轴标签，以避免重叠
        plt.show()
# 创建测试数据
tf_gene = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5']  # TF gene 列表
tf_mou = ['Module1', 'Module2', 'Module1', 'Module4', 'Module5']  # TF 模块
tf_ratio = [0.2, 0.4, 0.6, 0.8, 1.0]  # 每个 TF 对应的 ratio 值

# 创建 tf_module 实例并绘制散点图
tf_module_instance = tf_moudle(tf_gene, tf_mou, tf_ratio)
tf_module_instance.plot_scatter()
