from umap import UMAP

class UMAPReduction:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean',
                 learning_rate=1.0, n_epochs=None, spread=1.0, set_op_mix_ratio=1.0,
                 local_connectivity=1, repulsion_strength=1, negative_sample_rate=5,
                 transform_queue_size=4.0, a=None, b=None, random_state=None, metric_kwds=None,
                 angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical',
                 target_metric_kwds=None, target_weight=0.5):
        """
        Initializes the UMAP dimensionality reduction class.

        Parameters:
            n_components (int): The number of dimensions in the target space.
            n_neighbors (int): The number of neighboring points used in the UMAP algorithm.
            min_dist (float): The minimum distance between embedded points, controls how tightly points are packed.
            metric (str): The metric to use for computing distances in the original space.
            learning_rate (float): The learning rate for the optimization.
            n_epochs (int): The number of epochs to use in optimization. If None, automatically determined.
            spread (float): The effective scale of embedded points. Inversely related to min_dist.
            set_op_mix_ratio (float): The mix ratio between union and intersection computations in the embedding optimization.
            local_connectivity (int): The local connectivity of the manifold. One means the nearest neighbor, higher values indicate higher connectivity.
            repulsion_strength (float): The strength of the repulsion term in the UMAP loss function.
            negative_sample_rate (int): The number of negative samples to use per positive sample.
            transform_queue_size (float): The size of the queue for storing computed transformations for out-of-sample data.
            a (float), b (float): Parameters of the embedding process, generally not set manually.
            random_state (int, RandomState instance, or None): Controls the randomness of the algorithm.
            metric_kwds (dict): Additional keyword arguments for the metric function.
            angular (bool): Whether to use angular distance instead of Euclidean.
            target_n_neighbors (int): The number of neighbors to use when constructing the target simplifying manifold.
            target_metric (str): The metric used to measure distance in the target space.
            target_metric_kwds (dict): Additional keyword arguments for the target metric.
            target_weight (float): The weight given to the preservation of the target configuration over the input configuration.

        Note: For more advanced settings, refer to the UMAP documentation.
        """
        self.reducer = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            spread=spread,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            transform_queue_size=transform_queue_size,
            a=a,
            b=b,
            random_state=random_state,
            metric_kwds=metric_kwds,
            angular_rp_forest=angular_rp_forest,
            target_n_neighbors=target_n_neighbors,
            target_metric=target_metric,
            target_metric_kwds=target_metric_kwds,
            target_weight=target_weight
        )

    def fit_transform(self, X):
        """
        Fits the UMAP model and transforms the data X into the lower-dimensional space.

        Parameters:
            X (array-like, shape (n_samples, n_features)): The data to reduce the dimensionality of.

        Returns:
            array-like, shape (n_samples, n_components): The data transformed into the lower-dimensional space.
        """
        self.umapdata = self.reducer.fit_transform(X)
        return self.umapdata

