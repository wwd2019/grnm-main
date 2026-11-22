
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class grn_net:
    def __init__(self,pos,edges,gene_idx,color_maps=[]):
        self.pos = pos
        self.edges = edges
        if(len(color_maps)==0):
            color_maps = ["#fe420f", "#a6cee3", "#b2df8a", "#cab2d6", "#fb9a99", "#6a3d9a", "#fdbf6f", "#1f78b4",
                          "#019529", "#00ffff", "#ff00ff",
                          "#ffff00"]
        self.color_maps = color_maps
        self.gene_idx = gene_idx
    def grn_plot(self,labels_display='y'):

        # x = np.array(self.pos['x'])
        # y = np.array(self.pos['y'])
        # label = np.array(self.pos['module'])
        # # show_cluster = np.array(embedding_res['show_cluster'])
        # name = np.array(self.pos['tf_name'])

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

        cnt = 0
        for mod in set(self.pos['module']):
            nx.draw_networkx_nodes(g, pos, nodelist=nd_list[mod], node_size=15, node_color=self.color_maps[cnt], label=mod)
            cnt += 1

        # nx.draw_networkx_nodes(g, pos, nodelist=range(len(embedding_res)), node_size=15, node_color=node_color,
        #                        )
        if(labels_display == 'y'):
            nx.draw_networkx_labels(g, pos, node_label, font_size=10)
        else:
            pass
        nx.draw_networkx_edges(g, pos, edges, edge_color='grey', width=0.01)
        nx.draw_networkx(g)
        plt.legend(loc='best')
        # plt.savefig('/home/lmy/data/scATAC/community_detection/nw_show/time_all_labeled.png', dpi=300)
        # plt.savefig('/home/lmy/data/scATAC/community_detection/nw_cons_1/time_all_test.svg', dpi=300)
        # plt.savefig('/home/lmy/data/scATAC/community_detection/nw_cons_1/time_all.svg')

        # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7'])
        plt.show()
        # nx.write_gml(g, 'test.gml')
        # plt.savefig('/home/lmy/data/scATAC/community_detection/nw_cons_1/time_all.pdf')
        # plt.show()

