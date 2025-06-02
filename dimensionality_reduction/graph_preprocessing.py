import dgl
import torch
import numpy as np
import dgl
import numpy as np
import torch

class GraphGeneration:
    """
    A class to generate a graph from biological datasets, incorporating both RNA and ATAC-seq features,
    edges, and transcription factor motifs. This graph is typically used for further analysis in
    genomics and systems biology.

    Attributes:
        rna_features (DataFrame): A dataset containing features derived from RNA sequencing.
        atac_features (DataFrame): A dataset containing features from ATAC sequencing.
        edges (DataFrame): A dataset specifying the edges between nodes in the graph, typically representing
                           biological interactions.
        gene_idx (dict): A dictionary mapping gene identifiers to their respective indices in the graph.
        tf_gene_motif (DataFrame): A dataset containing information about transcription factors and their motifs.
    """

    def __init__(self, rna_features, atac_features, edges, gene_idx, tf_gene_motif):
        """
        Initializes the GraphGeneration instance with datasets required to build the graph.

        Parameters:
            rna_features (DataFrame): RNA sequencing data.
            atac_features (DataFrame): ATAC sequencing data.
            edges (DataFrame): Data about the connections between nodes.
            gene_idx (dict): Mapping from gene identifiers to indices.
            tf_gene_motif (DataFrame): Data about transcription factors and their associated motifs.
        """
        self.rna_features = rna_features
        self.atac_features = atac_features
        self.edges = edges
        self.gene_idx = gene_idx
        self.tf_gene_motif = tf_gene_motif

    def graph_generation(self):
        """
        Generates a graph using the initialized datasets. This involves creating nodes for each gene and
        edges based on the provided edges dataset, along with annotating nodes with RNA and ATAC features
        and identifying transcription factor nodes.
        """
        # Process transcription factors to find corresponding gene indices
        tf_smeg = self.tf_gene_motif[['smesg']]
        tf_gene_idx = []
        for idx, row in tf_smeg.iterrows():
            smegs = row['smesg']
            if smegs in self.gene_idx:
                tf_gene_idx.append(str(self.gene_idx[smegs]))
            else:
                tf_gene_idx.append('-')
        tf_smeg['node_idx'] = tf_gene_idx

        # Map each node index to its corresponding transcription factor index
        node_tfname = {row['node_idx']: idx for idx, row in tf_smeg.iterrows()}

        # Obtain indices for graph construction
        src_node, tar_node = self._get_node_indices()

        # Create the graph
        g = dgl.graph((src_node, tar_node))
        feature_all = np.hstack((np.array(self.rna_features), np.array(self.atac_features)))
        g.ndata['all'] = torch.from_numpy(feature_all)
        g.ndata['rna'] = torch.from_numpy(np.array(self.rna_features))
        g.ndata['atac'] = torch.from_numpy(np.array(self.atac_features))
        g.edata['weight'] = torch.from_numpy(np.array(self.edges['weight']))

        # Identify transcription factor nodes
        TF = [str(i) in node_tfname for i in range(len(g.nodes()))]
        g.ndata['TF'] = torch.from_numpy(np.array(TF))

        # Store the generated graph
        self.g = g

    def _get_node_indices(self):
        """
        Helper function to obtain source and target node indices for graph construction.

        Returns:
            tuple: Lists of source and target indices used for constructing the graph.
        """
        src_node = [self.gene_idx[key] for key in self.edges['source']]
        tar_node = [self.gene_idx[key] for key in self.edges['target']]
        return src_node, tar_node

#
# class GraphGeneration():
#     def __init__(self,rna_features,atac_features,edges,gene_idx,tf_gene_motif):
#         self.rna_features = rna_features
#         self.atac_features = atac_features
#         self.edges = edges
#         self.gene_idx = gene_idx
#         self.tf_gene_motif = tf_gene_motif
#     def graph_generation(self):
#         tf_smeg = self.tf_gene_motif[['smesg']]
#         tf_gene_idx = []
#         for idx, row in tf_smeg.iterrows():
#             smegs = row['smesg']
#             if (smegs in self.gene_idx.keys()):
#                 tf_gene_idx.append(str(self.gene_idx[smegs]))
#             else:
#                 tf_gene_idx.append('-')
#
#         tf_smeg['node_idx'] = tf_gene_idx
#
#         node_tfname = {}
#
#         for idx, row in tf_smeg.iterrows():
#             node_idx = row['node_idx']
#             node_tfname[node_idx] = idx
#         src_node, tar_node = self._get_node_indices()
#
#         g = dgl.graph((src_node, tar_node))
#         feature_all = np.hstack((np.array(self.rna_features), np.array(self.atac_features)))
#
#         g.ndata['all'] = torch.from_numpy(feature_all)
#         g.ndata['rna'] = torch.from_numpy(np.array(self.rna_features))
#         g.ndata['atac'] = torch.from_numpy(np.array(self.atac_features))
#         g.edata['weight'] = torch.from_numpy(np.array(self.edges['weight']))
#         TF = []
#         for i in range(len(g.nodes())):
#             if (str(i) in node_tfname.keys()):
#                 TF.append(True)
#             else:
#                 TF.append(False)
#
#         TF = np.array(TF)
#         g.ndata['TF'] = torch.from_numpy(TF)
#         self.g = g
#
#     def _get_node_indices(self):
#         """Helper function to obtain source and target node indices for graph construction."""
#         src_node = [self.gene_idx[key] for key in self.edges['source']]
#         tar_node = [self.gene_idx[key] for key in self.edges['target']]
#         return src_node, tar_node
