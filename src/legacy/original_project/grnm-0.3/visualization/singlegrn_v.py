import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



class s_grn_net:
    def __init__(self,pos,edges,gene_idx,color_maps=[]):
        self.pos = pos
        self.edges = edges
        if(len(color_maps)==0):
            color_maps = ["#fe420f", "#a6cee3", "#b2df8a", "#cab2d6", "#fb9a99", "#6a3d9a", "#fdbf6f", "#1f78b4",
                          "#019529", "#00ffff", "#ff00ff",
                          "#ffff00"]
        self.color_maps = color_maps
        self.gene_idx = gene_idx
    def s_grn_net_plot(self,nodes,labels_display='y'):
        pos = {}
        for idx, row in self.pos.iterrows():
            xx = row['x']
            yy = row['y']
            pos[idx] = np.array([xx, yy])

        src = [self.gene_idx[uu] for uu in self.edges['source']]
        tar = [self.gene_idx[vv] for vv in self.edges['target']]

        # node_color = [self.color_maps[c] for c in self.pos['module']]

        node_label = {}
        for idx, row in self.pos.iterrows():
            name = ''
            if (row['tf_name'] != '-'):
                name = row['tf_name']
            node_label[idx] = name

        edges = list(zip(src, tar))

        nd_list = {}
        for mod in set(self.pos['module']):
            nd_list[mod] = []
        for idx, row in self.pos.iterrows():
            mod = row['module']
            nd_list[mod].append(idx)

        g = nx.Graph()
        plt.figure(figsize=(20, 20))
        nodes_to_extract = nodes
        subgraph = g.subgraph(nodes_to_extract)
        nx.draw(subgraph, with_labels=True, font_weight='bold')
        plt.show()

