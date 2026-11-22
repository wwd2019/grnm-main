import networkx as nx
import community  # 从python-louvain库导入


class Louvain:
    """
    A class to apply the Louvain method for community detection within a graph.

    Attributes:
        g (networkx.Graph): A NetworkX graph or an object that can be converted to one.
        q (float): The resolution parameter for modularity optimization.
                   Higher values lead to more communities.
        partition (dict): The result of the community detection, mapping each node
                          to the community number it belongs to.
    """

    def __init__(self, g, q=1.):
        """
        Initializes the Louvain community detection algorithm with a given graph.

        Args:
            g: A NetworkX graph or any object with a `to_networkx` method.
            q (float, optional): The modularity resolution parameter.
        """
        self.g = g
        self.q = q
        self.partition = None

    def dgl_to_weighted_networkx(self):
        """
        Convert a DGL graph with edge weights into a weighted NetworkX graph.
        """
        g_dgl = self.g
        g_nx = nx.Graph()
        src, dst = g_dgl.edges()
        weights = g_dgl.edata['weight'].numpy().flatten()
        for s, t, w in zip(src.numpy(), dst.numpy(), weights):
            g_nx.add_edge(int(s), int(t), weight=float(w))
        return g_nx
    def run_louvain(self):
        """
        Applies the Louvain community detection algorithm on the graph.

        This method will modify the graph stored in the instance by applying
        the Louvain community detection algorithm to identify communities.
        The method will set the `partition` attribute with a dictionary
        mapping each node to the community it belongs to.
        """
        # Convert the graph to a NetworkX graph if necessary
        # if not isinstance(self.g, nx.Graph):
        # Apply the Louvain algorithm to the graph
        # self.partition = community.best_partition(nx.Graph(self.g.to_networkx()))
        # g_nx = nx.Graph(self.g.to_networkx())
        g_nx = self.dgl_to_weighted_networkx()
        print(f"节点数: {g_nx.number_of_nodes()}")
        print(f"边数: {g_nx.number_of_edges()}")
        # g_nx = nx.Graph(self.g.to_networkx())

        # 如果你希望用 'id' 作为权重（假设它可反映连接强度），改成 weight
        for u, v, data in g_nx.edges(data=True):
            data['weight'] = data.get('id', 1.0)  # 如果没有id，就默认1.0

        self.partition = community.best_partition(
            g_nx,
            resolution=self.q,
            weight='weight'  # 显式指定权重字段
        )

    def get_partition(self):
        """
        Retrieves the partition of the graph nodes into communities.

        Returns:
            A dictionary where keys are node identifiers and values are the
            community numbers assigned to those nodes.
        """
        if self.partition is None:
            self.run_louvain()
        return self.partition

# Usage example:
# g is a graph object, either a NetworkX graph or an object with a `to_networkx` method.
# q is the resolution parameter for the Louvain algorithm.
# louvain_instance = Louvain(g, q)
# louvain_instance.run_louvain()
# partition = louvain_instance.get_partition()
# print(partition)