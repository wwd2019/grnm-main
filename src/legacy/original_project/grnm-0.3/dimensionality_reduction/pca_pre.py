from sklearn.decomposition import PCA as SKLearnPCA

class PCA:
    def __init__(self, n_components):
        """
        初始化
        :param n_components: int
        降维后的维数
        """
        self.n_components = n_components
        self.model = SKLearnPCA(n_components = n_components)
    def fit_transform(self, X):
        """
        对数据X进行PCA降维。
        
        :param X: array-like, shape (n_samples, n_features)
            待降维的数据。
        :return: array-like, shape (n_samples, n_components)
            降维后的数据。
        """
        self.PCA_data = self.model.fit_transform(X)