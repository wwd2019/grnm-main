
from dimensionality_reduction.pca_preprocessing import PCA
from dimensionality_reduction.umap_preprocessing import UMAPReduction
from dimensionality_reduction.graph_preprocessing import GraphGeneration
from dimensionality_reduction.graph_autoencoder import GraphAutoencoderTrainer
from clustering.louvain_clustering import *
from plotting.generate_heatmap import *
from plotting.grn_visualization import grn_net
from plotting.module_target_visualization import tf_moudle
import numpy as np
class Grnm():
    def __init__(self, rnadata, atacdata, edges, tf_gene_motif):
        # Initialize with input data
        self.rnadata = rnadata
        self.atacdata = atacdata
        self.edges = edges
        self.tf_gene_motif = tf_gene_motif

        # Get gene names and create gene index dictionary
        self.gene_name = rnadata.index.to_list()
        self.gene_idx = {self.gene_name[i]: i for i in range(len(self.gene_name))}

        self.index_togene =  {i:self.gene_name[i] for i in range(len(self.gene_name))}
    def pca(self, d1):
        # Perform PCA on RNA and ATAC data
        PCA_r = PCA(n_components=d1)
        PCA_r.fit_transform(self.rnadata)
        self.PCA_r_data = PCA_r.PCA_data
        PCA_a = PCA(n_components=d1)
        PCA_a.fit_transform(self.atacdata)
        self.PCA_a_data = PCA_a.PCA_data

    def graph(self):
        # Generate graph using RNA and ATAC data, edges, and gene index
        gn = GraphGeneration(self.PCA_r_data, self.PCA_a_data, self.edges, self.gene_idx, self.tf_gene_motif)
        gn.graph_generation()
        self.g = gn.g

    def gae(self, d2=8, epoch_n=1000, alpha_n=0.01):
        # Perform GAE on graph
        gae_ = GraphAutoencoderTrainer(self.g, epoch_n, alpha_n)
        gae_.train(n_components=d2)
        self.z = gae_.z

    def umap(self):
        # Perform UMAP on GAE embedding
        UMAP = UMAPReduction()
        UMAP.fit_transform(self.z)
        self.umap_data = UMAP.umapdata

    def louvain(self,q):
        # Apply Louvain algorithm for community detection

        Louvain_ = Louvain(self.g, q)
        Louvain_.run_louvain()
        self.partition = Louvain_.get_partition()

    def match_heat(self, celltypes_marker):
        # Calculate overlap between modules and cell types markers
        moudels_dict = {}
        for moudel in self.partition.items():
            if moudel[1] in moudels_dict:
                moudels_dict[moudel[1]].append(self.index_togene[moudel[0]])
            else:
                moudels_dict[moudel[1]] = [self.index_togene[moudel[0]]]
        self.moudles = moudels_dict
        match = HeatmapGenerator(moudels_dict, celltypes_marker)
        match.generate_overlap_heatmap()

    def grn_plot(self,label_dis='y',gene_list_dis=[]):
        # Visualize gene regulatory network
        df = pd.DataFrame(self.umap_data, columns=['x', 'y'])
        df['tf_name'] = self.gene_name
        df['module'] = self.partition
        grnm_p = grn_net(df, self.edges, self.gene_idx)
        grnm_p.grn_plot(label_dis,gene_list_dis)

    def single_grn_plot(self, nodes,label_dis='y',gene_list_dis=[]):
        # Visualize single gene regulatory network
        df = pd.DataFrame(self.umap_data, columns=['x', 'y'])
        df['tf_name'] = self.gene_name
        df['module'] = self.partition
        grnm_sp = grn_net(df, self.edges, self.gene_idx)
        grnm_sp.s_grn_net_plot(nodes,label_dis,gene_list_dis)

    def tf_target_scatterplot(self,tf_list,tf_module):
        # Plot scatterplot for TF target genes
        print(tf_list)
        tf_ratio = []
        # for tf in tf_list:
        #     num = 0
        #     num_all = 0
        #     tf_id = self.gene_idx[tf]
        #     connected_nodes = self.g[tf_id]
        #     for node in connected_nodes:
        #         par_value = self.partition[node]
        #         if par_value is not None:
        #             if (self.partition[tf_id] == par_value):
        #                 num += 1
        #         num_all += 1
        #     tf_ratio.append(num / num_all)
        # tf_m = tf_moudle(tf_list, tf_module, tf_ratio)
        # tf_m.plot_scatter()
# # #
# import pandas as pd
# time = '72'
# rna_features = pd.read_csv('D:/work/lmy/rna_pca_'+time+'.csv', index_col=0)
# atac_features = pd.read_csv('D:/work/lmy/atac_pca_'+time+'.csv', index_col=0)
# edges = pd.read_csv('D:/work/lmy/edges_'+time+'.csv', index_col=0)
#
# tf_gene_motif = pd.read_csv('D:/work/lmy/tf_gene_motif.csv', index_col=0)
#
# gcn = Grnm(rna_features,atac_features,edges,tf_gene_motif)
# gcn.g_pca(10)
# gcn.g_graph()
# gcn.g_gae(8,100,0.01)
# #
# gcn.g_louvain(0.5)
# gcn.g_umap()
# # print(gcn.partition)
#
# gcn.g_match_heat(celltype_markers)
# gcn.g_grn_plot('y',['SMESG000000077', 'SMESG000000110', 'SMESG000000959', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774'])
# gcn.g_single_grn_plot('SMESG000001287','y',['SMESG000001287','SMESG000000077', 'SMESG000000110', 'SMESG000000959', 'SMESG000001202', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774', 'SMESG000002907'])
# # print(gcn.partition)
# gcn.g_tf_target_scatterplot()





