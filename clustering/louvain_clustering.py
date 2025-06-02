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
        self.partition = community.best_partition(nx.Graph(self.g.to_networkx()))

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