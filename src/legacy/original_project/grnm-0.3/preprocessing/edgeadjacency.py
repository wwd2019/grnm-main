import numpy as np
class edgeadjacency():
    def __init__(self,source,target,edges):
        self.source = source
        self.target = target
        self.edges = edges
    def edge_adjacency_generation(self):
        # 创建节点到编号的映射
        node_names = list(set(self.source) | set(self.target))

        node_to_idx = {node_name: idx for idx, node_name in enumerate(node_names)}
        # 初始化邻接矩阵为全零
        num_nodes = len(node_names)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        # 遍历每条边，更新邻接矩阵
        for s, t, e in zip(self.source, self.target, self.edges):
            # 将节点名转换为编号
            s_idx = node_to_idx[s]
            t_idx = node_to_idx[t]
            adjacency_matrix[s_idx, t_idx] = e
            # 如果是无向图，也设置反向边
            # adjacency_matrix[t_idx, s_idx] = e
        self.adjacency_matrix = adjacency_matrix
        self.node_to_idx = node_to_idx
