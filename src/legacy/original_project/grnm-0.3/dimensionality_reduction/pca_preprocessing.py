from sklearn.decomposition import PCA as SKLearnPCA


class PCA:
    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=None):
        """
        Principal component analysis (PCA).

        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space.

        :param n_components: int, float, None or str
            The number of components to keep. If n_components is not set all components
            are kept. If n_components is an integer, it selects the number of components
            to keep. If n_components is a float between 0 and 1, it is used as the
            fraction of variance that has to be explained by the selected components.
            If svd_solver is 'arpack', the number of components must be strictly
            less than the minimum of n_features and n_samples.

        :param copy: bool (default True)
            If False, data passed to fit are overwritten and running fit(X).transform(X)
            will not yield the expected results, use fit_transform(X) instead.

        :param whiten: bool (default False)
            When True, the components_ vectors are multiplied by the square root of n_samples
            and then divided by the singular values to ensure uncorrelated outputs with
            unit component-wise variances.

        :param svd_solver: str {'auto', 'full', 'arpack', 'randomized'}
            The solver used to compute the eigenvectors and eigenvalues of the covariance matrix.

        :param tol: float >= 0, optional (default .0)
            Tolerance for singular values computed by svd_solver 'arpack'.

        :param iterated_power: int >= 0, or 'auto', (default 'auto')
            The number of iterations for the power method computed by svd_solver 'randomized'.

        :param random_state: int, RandomState instance or None, optional (default None)
            If int, random_state is the seed used by the random number generator. If
            RandomState instance, random_state is the random number generator. If None,
            the random number generator is the RandomState instance used by np.random.
        """
        self.model = SKLearnPCA(n_components=n_components, copy=copy, whiten=whiten,
                                svd_solver=svd_solver, tol=tol,
                                iterated_power=iterated_power, random_state=random_state)

    def fit_transform(self, X):
        """
        Apply PCA by fitting the model with X and projecting the data onto the
        principal components.

        :param X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        :return: array-like, shape (n_samples, n_components)
            The array of projected data.
        """
        self.PCA_data = self.model.fit_transform(X)

        def get_pca(self):
            """
            Retrieve the PCA model after fitting it with data.

            This method returns the `SKLearnPCA` instance with all attributes filled after fitting.
            It can be used to access the principal components, explained variance, singular values,
            and other properties of the fitted PCA model.

            :return: SKLearnPCA
                The underlying PCA model.
            """
            return self.model

        def explained_variance(self):
            """
            Retrieve the amount of variance explained by each of the selected components.

            This attribute is available only after fitting the model with data.

            :return: array-like, shape (n_components,)
                The amount of variance explained by each of the selected components.
            """
            return self.model.explained_variance_

        def components(self):
            """
            Retrieve the principal components.

            This attribute is available only after fitting the model with data.

            :return: array-like, shape (n_components, n_features)
                Principal axes in feature space, representing the directions of maximum variance in the data.
            """
            return self.model.components_

        def explained_variance_ratio(self):
            """
            Retrieve the percentage of variance explained by each of the selected components.

            This attribute is available only after fitting the model with data.

            :return: array-like, shape (n_components,)
                The percentage of variance explained by each of the selected components.
            """
            return self.model.explained_variance_ratio_

    # Example usage:
    # pca = PCA(n_components=2, svd_solver='randomized', random_state=42)
    # transformed_data = pca.fit_transform(data)
    # pca_model = pca.get_pca()
    # components = pca.components()
    # explained_variance = pca.explained_variance()
    # explained_variance_ratio = pca.explained_variance_ratio()