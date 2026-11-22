# umap_reduction.py
from umap import UMAP

class UMAPReduction:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        """
        初始化UMAP降维类。
        
        :param n_components: int
            目标维度数。
        :param n_neighbors: int
            用于UMAP算法的邻居数。
        :param min_dist: float
            UMAP算法中的最小距离参数，控制嵌入空间中点的紧密程度。
        :param metric: str
            用于原始数据的距离度量。
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric=self.metric
        )

    def fit_transform(self, X):
        """
        对数据X进行UMAP降维。
        
        :param X: array-like, shape (n_samples, n_features)
            待降维的数据。
        :return: array-like, shape (n_samples, n_components)
            降维后的数据。
        """
        self.umapdata = self.reducer.fit_transform(X)

