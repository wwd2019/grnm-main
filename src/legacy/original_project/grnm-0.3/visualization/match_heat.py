import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class heat:
    def __init__(self,moudles,celltypes_marker):
        self.moudles = moudles
        self.celltypes_marker = celltypes_marker
    def overlap(self):
        overlap_matrix = np.zeros((len(self.moudles), len(self.celltypes_marker)))
        for i, (module_name, module_genes) in enumerate(self.moudles.items()):
            for j, (cell_type, markers) in enumerate(self.celltypes_marker.items()):
                overlap = len(set(module_genes) & set(markers))
                overlap_matrix[i, j] = overlap

        # 生成热图
        sns.set()
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, annot=True, xticklabels=list(self.celltypes_marker.keys()),
                    yticklabels=list(self.moudles.keys()))
        plt.xlabel("Cell Types")
        plt.ylabel("Modules")
        plt.title("Overlap Heatmap between Modules and Cell Type Markers")
        plt.show()
