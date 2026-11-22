import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler


class Louvain:
    def __init__(self,data, q):
        self.data = data
        self.q = q
    def louvain(self):
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        distances = euclidean_distances(self.data)
        threshold = np.percentile(distances,self.q)
        self.G = nx.Graph()
        # print(distances)
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                # if distances[i, j] < threshold:
                if distances[i, j] != 0:
                    self.G.add_edge(i, j)
        # 应用Louvain算法进行社区检测
        # print(self.G)
        self.partition = community_louvain.best_partition(self.G)
        print(len(self.partition))

