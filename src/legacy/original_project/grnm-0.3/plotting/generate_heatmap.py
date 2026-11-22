import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class HeatmapGenerator:
    """
    A class for generating heatmaps to visualize the overlap between gene modules and cell type markers.

    Attributes:
        modules (dict): A dictionary where keys are module names and values are lists of gene names in the module.
        celltypes_marker (dict): A dictionary where keys are cell type names and values are lists of marker genes.
    """
    def __init__(self, modules, celltypes_marker):
        """
        Initializes the HeatmapGenerator with gene modules and cell type markers.
        Parameters:
            modules (dict): A dictionary containing gene modules.
            celltypes_marker (dict): A dictionary containing cell type markers.
        """
        self.modules = modules
        self.celltypes_marker = celltypes_marker

    def generate_overlap_heatmap(self):
        """
        Calculates the overlap between gene modules and cell type markers and generates a heatmap.
        """
        overlap_matrix = np.zeros((len(self.celltypes_marker), len(self.modules)))
        for i, (module_name, module_genes) in enumerate(self.modules.items()):
            for j, (cell_type, markers) in enumerate(self.celltypes_marker.items()):
                overlap = len(set(module_genes) & set(markers))
                module_gene_count = len(set(module_genes))
                overlap_matrix[j, i] = overlap / module_gene_count

        # 生成热图 — figure size scales with data dimensions to avoid memory issues
        n_rows = len(self.celltypes_marker)
        n_cols = len(self.modules)
        # Cap dimensions to a reasonable range; each cell gets ~0.5 inches
        fig_w = min(max(n_cols * 0.5, 8), 40)
        fig_h = min(max(n_rows * 0.5, 6), 30)
        # Use a lower DPI for very large heatmaps to avoid RendererAgg OOM
        dpi = 150 if n_rows * n_cols > 200 else 300

        sns.set()
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(overlap_matrix, annot=(n_rows * n_cols <= 200),
                    xticklabels=list(self.modules.keys()),
                    yticklabels=list(self.celltypes_marker.keys()))
        plt.xlabel("Modules")
        plt.ylabel("Cell Types")
        plt.title("Overlap Heatmap between Modules and Cell Type Markers")
        plt.savefig('./png/test.png', dpi=dpi, bbox_inches='tight')
        plt.close()
