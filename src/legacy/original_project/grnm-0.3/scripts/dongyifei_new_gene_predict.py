# 此文件用于完成基于已拥有的GRN 数据预测新的基因之间调控关系
# nfyc和mdh1/mdh2/idh2
# nfyc SMESG000077821
# mdh1b SMESG000070444
# mdh1 SMESG000027913 SMESG000056995
# mdh2 SMESG000008924 SMESG000035456 SMESG000008151
# idh2 没找到
import itertools
from collections import Counter
import os
from torch.nn import init
from tqdm import tqdm
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
GPU_NUMBER = [2]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
def getData(gene_target):
    with open('../map_data/genes_PCA_64.csv', 'r')as file:
        rna_lists = file.readlines()
    print('rna_lists', len(rna_lists))
    rna_dict = {}
    for line in rna_lists[1:]:
        line = line.strip('\n').split(',')
        smed_name = line[0].replace('"','')
        rna_dict[smed_name] = [float(i) for i in line[1:]]
    print('rna_dict', len(rna_dict))
    # print(rna_dict.keys())
    gene_id_map = {}
    with open('../map_data/adata_var_data.csv', 'r')as file:
        adata_lists = file.readlines()
    for line in adata_lists[1:]:
        line = line.strip('\n').split(',')
        gene_id_map[line[2]] = line[1].split('-')[0]
    # 手动增加进去
    gene_id_map['SMED30023774'] = 'SMESG000042152'
    gene_id_map['SMED30019773'] = 'SMESG000034865'
    gene_id_map['SMED30034381'] = 'SMESG000056995'
    for gene in ['SMED30023774', 'SMED30019773', 'SMED30034381', 'SMED30023249','SMED30027148']:
        if(gene not in rna_dict):
            print(gene)
    # print(gene_id_map)

    smesg_rna = {}
    for gene in rna_dict:
        if(gene in gene_id_map):
            smesg_rna[gene_id_map[gene]] = rna_dict[gene]
        if(gene.startswith("SMESG")):
            smesg_rna[gene] = rna_dict[gene]
    print(len(smesg_rna))
    # 'SMESG000077821','SMESG000035181'
    smesg_rna['SMESG000077821'] = rna_dict['SMED30023249']
    smesg_rna['SMESG000035181'] = rna_dict['SMED30027148']
    # print([gene for gene in gene_target if gene not in smesg_rna]) # ['SMESG000056995']
    with open('../example_data/edges.csv', 'r')as file:
        edge_list = file.readlines()
    edge_scores = []
    for line in edge_list[1:]:
        line = line.strip('\n')
        line = line.split(',')
        edge_scores.append([line[1], line[2], float(line[3])])
    print(edge_scores[0])
    return edge_scores,smesg_rna
def makeDataSplit(edge_scores):
    # 提取原始分数
    scores = np.array([x[2] for x in edge_scores])
    # 计算 7 个等级的分位点（6个分割点）
    quantiles = np.quantile(scores, np.linspace(0, 1, 6))  # 分成5段 => 6个端点
    print(quantiles)
    # 按分布划分等级
    def get_level(score):
        for i in range(5):
            if quantiles[i] <= score <= quantiles[i+1]:
                return i   # 等级0~4
        return 4  # 默认最高

    # 应用映射
    edge_scores_new = [[g1, g2, get_level(s)] for g1, g2, s in edge_scores]
    return edge_scores_new
# 构建模型
class GenePairClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        # 调用初始化函数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化更适合分类任务
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
def makeModel(edge_scores_new, smesg_rna):
    # 构造训练样本
    X, y = [], []
    for g1, g2, s in edge_scores_new:
        if g1 in smesg_rna and g2 in smesg_rna:
            emb1 = np.array(smesg_rna[g1], dtype=np.float32)
            emb2 = np.array(smesg_rna[g2], dtype=np.float32)
            x = np.concatenate([emb1, emb2 ]) # ,np.abs(emb1 - emb2), emb1 * emb2
            X.append(x)
            y.append(int(s))  # 确保是整数类别标签

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)  # 分类任务必须是int64

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # 转成 TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))
    print(len(train_dataset))
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 初始化模型
    model = GenePairClassifier(X_train.shape[1], num_classes=5).to('cuda')
    # 在定义 criterion 时加权
    class_counts = torch.bincount(torch.tensor(y_train))
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to('cuda'))

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # 训练循环
    for epoch in tqdm(range(2000)):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        # 验证
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
                val_logits = model(X_batch)
                val_loss += criterion(val_logits, y_batch).item() * len(X_batch)
                val_pred = val_logits.argmax(dim=1)
                correct += (val_pred == y_batch).sum().item()
                total += len(y_batch)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {total_loss/len(train_dataset):.4f} | "
                  f"Val Loss: {val_loss/len(val_dataset):.4f} | Val Acc: {correct/total:.3f}")
            if (epoch + 1) % 1000 == 0:
                torch.save(model.state_dict(), f"../models/gene_pair_classifier_{epoch + 1}.pth")
            g1 = 'SMESG000077821'
            g2 = 'SMESG000035181'
            x1 = torch.tensor(smesg_rna[g1], dtype=torch.float32).unsqueeze(0).to('cuda')
            x2 = torch.tensor(smesg_rna[g2], dtype=torch.float32).unsqueeze(0).to('cuda')
            # ✅ 拼接向量（按列拼接）
            x = torch.cat([x1, x2], dim=1)
            # 前向预测
            with torch.no_grad():
                pred = model(x)  # 模型输出 logits
                prob = torch.softmax(pred, dim=1)  # 转成概率分布
                score, label = torch.max(prob, dim=1)  # 最大概率及对应类别
                print(score, label)
        # 构造伪样本
        g1, g2 = 'SMESG000077821', 'SMESG000035181'
        x1 = torch.tensor(smesg_rna[g1], dtype=torch.float32).unsqueeze(0).to('cuda')
        x2 = torch.tensor(smesg_rna[g2], dtype=torch.float32).unsqueeze(0).to('cuda')
        x = torch.cat([x1, x2], dim=1)

        # 假设类别=4
        y_pseudo = torch.tensor([4], dtype=torch.long).to('cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        # 微调几步
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y_pseudo)
        loss.backward()
        optimizer.step()


def loadModel(model_path):
    # 1. 定义模型结构（要和训练时完全一致）
    model = GenePairClassifier(input_dim = 128)
    # 2. 加载模型参数
    model_state = torch.load(model_path, map_location='cpu')  # 若无GPU，可加 map_location='cpu'
    model.load_state_dict(model_state)
    print(model)
    return model
if __name__ == "__main__":
    gene_target = ['SMESG000042152', 'SMESG000027913', 'SMESG000035456', 'SMESG000056995',
                   'SMESG000073623', 'SMESG000008924', 'SMESG000070444', 'SMESG000008151', 'SMESG000034865','SMESG000077821','SMESG000035181']
    edge_scores, smesg_rna = getData(gene_target)
    edge_scores_new = makeDataSplit(edge_scores)
    print(Counter([line[2] for line in edge_scores_new]))
    print(edge_scores_new[0])
    # ['SMESG000013010', 'SMESG000014588', -3.414662847]
    # [-3.39276085e+06 -7.83852330e-01  1.22438682e+00  4.53952444e+00
    #   1.72684505e+01  2.65275858e+06]
    # Counter({0: 2241, 1: 2241, 4: 2241, 3: 2241, 2: 2240}) # 0为负调控，4为正调控， 中间的可忽略
    # makeModel(edge_scores_new, smesg_rna)
    model = loadModel(model_path = '../models/gene_pair_classifier_2000.pth')
    gene_pairs = list(itertools.combinations(gene_target, 2))  # 所有两两组合

    results = []
    miss_genes = []
    for g1, g2 in gene_pairs:
        # 获取每个基因的表达向量（smesg_rna[g] 应该返回该基因的向量）
        if g1 not in smesg_rna or g2 not in smesg_rna:
            print(f"⚠️ 跳过 {g1}, {g2}，未在数据中找到。")
            if(g1 not in smesg_rna):
                miss_genes.append(g1)
            if(g2 not in smesg_rna):
                miss_genes.append(g2)
            continue

        x1 = torch.tensor(smesg_rna[g1], dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(smesg_rna[g2], dtype=torch.float32).unsqueeze(0)
        # ✅ 拼接向量（按列拼接）
        x = torch.cat([x1, x2], dim=1)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        # 前向预测
        with torch.no_grad():
            pred = model(x)  # 模型输出 logits
            prob = torch.softmax(pred, dim=1)  # 转成概率分布
            score, label = torch.max(prob, dim=1)  # 最大概率及对应类别

        # 保存为 Python 标量
        label = label.item()
        score = score.item()

        results.append((g1, g2, label, score))

    # =========================
    # 5️⃣ 打印预测结果
    # =========================
    for g1, g2, label, score in results:
        edge_scores_new.append([g1, g2, label])
        print(f"{g1} — {g2} : 类别={label}, 置信度={score:.4f}")
    miss_genes = list(set(miss_genes))
    print('miss_genes:', miss_genes)
    with open('../example_data/new_edges.csv', 'w') as file:
        for data in edge_scores_new:
            file.write(data[0]+','+data[1]+','+str(data[2])+'\n')
