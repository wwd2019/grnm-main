import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
class grn_net:
    def __init__(self,df,edges,gene_idx,color_maps=[]):
        self.df = df
        self.edges = edges
        if (len(color_maps) == 0):
            color_maps = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]

        self.color_maps = color_maps
        self.gene_idx = gene_idx
    def grn_plot(self,label_dis,gene_list_dis):
        pos_dic = {}
        for idx, row in self.df.iterrows():
            xx = row['x']
            yy = row['y']
            pos_dic[idx] = np.array([xx, yy])

        src = [self.gene_idx[uu] for uu in self.edges['source']]
        tar = [self.gene_idx[vv] for vv in self.edges['target']]

        # node_color = [self.color_maps[c] for c in self.pos['module']]

        node_label = {}
        for idx, row in self.df.iterrows():
            name = ''
            if (row['tf_name'] != '-'):
                name = row['tf_name']
            node_label[idx] = name

        g_edges = list(zip(src, tar))

        nd_list = {}
        for mod in set(self.df['module']):
            nd_list[mod] = []
        for idx, row in self.df.iterrows():
            mod = row['module']
            nd_list[mod].append(idx)

        g = nx.Graph()
        # Assuming you have edges
        # Add nodes and edges to the graph
        g.add_nodes_from(pos_dic.keys())
        g.add_edges_from(g_edges)

        plt.figure(figsize=(20, 20))
        # node_size = [node for node in g.nodes()]
        node_size_dic = {}
        for node in g.nodes:
            node_size_dic[node] = g.degree(node)
        for mod in set(self.df['module']):
            node_size_list = []
            for node in nd_list[mod]:
                node_size_list.append(node_size_dic[node])
            nx.draw_networkx_nodes(g, pos_dic, nodelist=nd_list[mod], node_size=node_size_list, node_color=self.color_maps[mod], label=mod)

        if(label_dis == 'y'):
            node_label_list = []
            for node in self.df['tf_name']:
                if node in gene_list_dis:
                    node_label_list.append(node)
                else:
                    node_label_list.append('')
            node_label_dict = dict(zip(g.nodes(), node_label_list))
            nx.draw_networkx_labels(g, pos_dic, node_label_dict, font_size=10)
        nx.draw_networkx_edges(g, pos_dic, g_edges, edge_color='grey', width=0.01)
        plt.legend(loc='best')
        self.plt = plt
        plt.show()
    def s_grn_net_plot(self, nodes,label_dis,gene_list_dis):

        g = nx.Graph()
        pos_dic = {}
        for idx, row in self.df.iterrows():
            xx = row['x']
            yy = row['y']
            pos_dic[idx] = np.array([xx, yy])

        # Assuming you have source and target lists for edges
        src = [self.gene_idx[uu] for uu in self.edges['source']]
        tar = [self.gene_idx[vv] for vv in self.edges['target']]

        # Assuming you have node labels
        node_label = {}
        for idx, row in self.df.iterrows():
            name = ''
            if (row['tf_name'] != '-'):
                name = row['tf_name']
            node_label[idx] = name

        # Assuming you have edges
        edges = list(zip(src, tar))
        # Add nodes and edges to the graph
        g.add_nodes_from(pos_dic.keys())
        g.add_edges_from(edges)
        subgraph = nx.Graph()
        # 将节点添加到子图中
        subgraph.add_node(self.gene_idx[nodes])
        # 获取当前节点的邻居
        neighbors = list(g.neighbors(self.gene_idx[nodes]))
        neighbors.append(self.gene_idx[nodes])
        # 将邻居节点添加到子图中
        subgraph.add_nodes_from(neighbors)
        edges_sub = []
        # 添加原始图中连接节点的边到子图中
        for neighbor1, neighbor2 in itertools.combinations(neighbors, 2):
            if g.has_edge(neighbor1, neighbor2):
                subgraph.add_edge(neighbor1, neighbor2)
                edges_sub.append((neighbor1, neighbor2))
        # Draw the subgraph with edges

        node_size_dic = {}
        for node in subgraph.nodes:
            node_size_dic[node] = subgraph.degree(node)

        plt.figure(figsize=(10, 10))


        mou = {}
        for node in subgraph.nodes():
            mou[node] = self.df['module'][node]
        nd_list = {}
        for mod in set(mou.values()):
            nd_list[mod] = []
        for idx in mou.keys():
            mod = mou[idx]
            nd_list[mod].append(idx)
        for mod in set(mou.values()):
            node_size_list = []
            for node in nd_list[mod]:
                node_size_list.append(node_size_dic[node])
            nx.draw_networkx_nodes(subgraph, pos_dic, nodelist=nd_list[mod], node_size=node_size_list, node_color=self.color_maps[mod], label=mod)
        if (label_dis == 'y'):
            node_label_list = []
            for node in [self.df['tf_name'][n] for n in subgraph.nodes()]:
                if node in gene_list_dis:
                    node_label_list.append(node)
                else:
                    node_label_list.append('')
            node_label_dict = dict(zip(subgraph.nodes(), node_label_list))
            nx.draw_networkx_labels(subgraph, pos_dic, node_label_dict, font_size=10)

        nx.draw_networkx_edges(subgraph, pos_dic,edges_sub, edge_color='grey', width=0.05)
        self.plt = plt
        plt.legend(loc='best')
        plt.show()

        nd_list_ratio = {}
        num_all = 0
        for k in nd_list.keys():
            num_all += len(nd_list[k])
        for k in nd_list.keys():
            nd_list_ratio[k] = len(nd_list[k]) / num_all
        print(nd_list_ratio)


