import dgl
import torch
import pandas as pd
import numpy as np

class graph_generation():
    def __init__(self,rna_features,atac_features,edges,gene_idx,tf_gene_motif):
        self.rna_features = rna_features
        self.atac_features = atac_features
        self.edges = edges
        self.gene_idx = gene_idx
        self.tf_gene_motif = tf_gene_motif
    def graph_generation_main(self):
        gene_smesg = list(self.gene_idx.keys())
        # gene_module = pd.DataFrame(index=gene_smesg)
        # gene_module['modulue'] = module_pred
        # tf_list = self.tf_gene_motif.index.to_list()
        tf_smeg = self.tf_gene_motif[['smesg']]
        tf_gene_idx = []
        for idx, row in tf_smeg.iterrows():
            smegs = row['smesg']
            if (smegs in self.gene_idx.keys()):
                tf_gene_idx.append(str(self.gene_idx[smegs]))
            else:
                tf_gene_idx.append('-')

        tf_smeg['node_idx'] = tf_gene_idx

        node_tfname = {}

        for idx, row in tf_smeg.iterrows():
            node_idx = row['node_idx']
            node_tfname[node_idx] = idx
        # 节点数目
        # N = len(self.rna_features)
        # rna_dim = len(self.rna_features.columns)
        # atac_dim = len(self.atac_features.columns)
        src_node = [self.gene_idx[key] for key in list(self.edges['source'])]
        tar_node = [self.gene_idx[key] for key in list(self.edges['target'])]

        g = dgl.graph((src_node, tar_node))
        feature_all = np.hstack((np.array(self.rna_features), np.array(self.atac_features)))

        g.ndata['all'] = torch.from_numpy(feature_all)
        g.ndata['rna'] = torch.from_numpy(np.array(self.rna_features))
        g.ndata['atac'] = torch.from_numpy(np.array(self.atac_features))
        g.edata['weight'] = torch.from_numpy(np.array(self.edges['weight']))
        TF = []
        for i in range(len(g.nodes())):
            if (str(i) in node_tfname.keys()):
                TF.append(True)
            else:
                TF.append(False)

        TF = np.array(TF)
        g.ndata['TF'] = torch.from_numpy(TF)
        self.g = g