import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

class GNNGraph:
    """
    A wrapper class for NetworkX graphs that provides functionality specific to GNNs.
    This includes methods for storing and manipulating node/edge features and labels.
    """
    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialize a GNN graph.
        
        Args:
            graph: An existing NetworkX graph (optional)
        """
        self.graph = graph if graph is not None else nx.Graph()
        self.node_features = {}  # Dictionary to store node features
        self.edge_features = {}  # Dictionary to store edge features
        self.node_labels = {}    # Dictionary to store node labels
        self.graph_label = None  # Graph-level label
    
    def add_node(self, node_id: Any, features: Optional[np.ndarray] = None, 
                 label: Optional[Any] = None) -> None:
        """
        Add a node to the graph with optional features and label.
        
        Args:
            node_id: Unique identifier for the node
            features: Node feature vector (numpy array)
            label: Node label (any type)
        """
        self.graph.add_node(node_id)
        if features is not None:
            self.node_features[node_id] = features
        if label is not None:
            self.node_labels[node_id] = label
    
    def add_edge(self, source: Any, target: Any, 
                 features: Optional[np.ndarray] = None) -> None:
        """
        Add an edge to the graph with optional features.
        
        Args:
            source: Source node ID
            target: Target node ID
            features: Edge feature vector (numpy array)
        """
        self.graph.add_edge(source, target)
        if features is not None:
            self.edge_features[(source, target)] = features
            # For undirected graphs, add the reverse edge features too
            if not self.graph.is_directed():
                self.edge_features[(target, source)] = features
    
    def set_node_features(self, node_id: Any, features: np.ndarray) -> None:
        """
        Set features for a specific node.
        
        Args:
            node_id: Node identifier
            features: Node feature vector
        """
        if node_id in self.graph:
            self.node_features[node_id] = features
        else:
            raise ValueError(f"Node {node_id} does not exist in the graph")
    
    def set_node_label(self, node_id: Any, label: Any) -> None:
        """
        Set label for a specific node.
        
        Args:
            node_id: Node identifier
            label: Node label
        """
        if node_id in self.graph:
            self.node_labels[node_id] = label
        else:
            raise ValueError(f"Node {node_id} does not exist in the graph")
    
    def set_edge_features(self, source: Any, target: Any, features: np.ndarray) -> None:
        """
        Set features for a specific edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            features: Edge feature vector
        """
        if self.graph.has_edge(source, target):
            self.edge_features[(source, target)] = features
            if not self.graph.is_directed():
                self.edge_features[(target, source)] = features
        else:
            raise ValueError(f"Edge ({source}, {target}) does not exist in the graph")
    
    def set_graph_label(self, label: Any) -> None:
        """
        Set a label for the entire graph.
        
        Args:
            label: Graph-level label
        """
        self.graph_label = label
    
    def get_node_features(self, node_id: Any) -> np.ndarray:
        """
        Get features for a specific node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node feature vector
        """
        return self.node_features.get(node_id, None)
    
    def get_node_label(self, node_id: Any) -> Any:
        """
        Get label for a specific node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node label
        """
        return self.node_labels.get(node_id, None)
    
    def get_edge_features(self, source: Any, target: Any) -> np.ndarray:
        """
        Get features for a specific edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Edge feature vector
        """
        return self.edge_features.get((source, target), None)
    
    def get_graph_label(self) -> Any:
        """
        Get the graph-level label.
        
        Returns:
            Graph label
        """
        return self.graph_label
    
    def get_neighbors(self, node_id: Any) -> List[Any]:
        """
        Get the neighbors of a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of neighbor node IDs
        """
        return list(self.graph.neighbors(node_id))
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix representation of the graph.
        
        Returns:
            Adjacency matrix as numpy array
        """
        return nx.to_numpy_array(self.graph)
    
    def get_node_feature_matrix(self) -> Tuple[List[Any], np.ndarray]:
        """
        Get node features as a matrix.
        
        Returns:
            Tuple of (node_ids, feature_matrix)
        """
        node_ids = list(self.graph.nodes())
        # Initialize feature matrix with zeros
        feature_dim = len(next(iter(self.node_features.values()))) if self.node_features else 0
        feature_matrix = np.zeros((len(node_ids), feature_dim))
        
        # Fill in the feature matrix
        for i, node_id in enumerate(node_ids):
            if node_id in self.node_features:
                feature_matrix[i] = self.node_features[node_id]
        
        return node_ids, feature_matrix
    
    def get_nodes_count(self) -> int:
        """
        Get the number of nodes in the graph.
        
        Returns:
            Number of nodes
        """
        return self.graph.number_of_nodes()
    
    def get_edges_count(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges
        """
        return self.graph.number_of_edges()
    
    def get_all_nodes(self) -> List[Any]:
        """
        Get all node IDs in the graph.
        
        Returns:
            List of all node IDs
        """
        return list(self.graph.nodes())
    
    def get_all_edges(self) -> List[Tuple[Any, Any]]:
        """
        Get all edges in the graph.
        
        Returns:
            List of (source, target) tuples representing edges
        """
        return list(self.graph.edges())


def create_random_graph(num_nodes: int, 
                        edge_probability: float = 0.3, 
                        feature_dim: int = 2, 
                        directed: bool = False,
                        seed: Optional[int] = None) -> GNNGraph:
    """
    Create a random graph with given parameters.
    
    Args:
        num_nodes: Number of nodes
        edge_probability: Probability of creating an edge between any two nodes
        feature_dim: Dimension of node feature vectors
        directed: Whether the graph should be directed
        seed: Random seed for reproducibility
        
    Returns:
        GNNGraph object
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create random graph
    G = nx.gnp_random_graph(num_nodes, edge_probability, directed=directed, seed=seed)
    gnn_graph = GNNGraph(G)
    
    # Add random features to nodes
    for node in G.nodes():
        features = np.random.randn(feature_dim)
        gnn_graph.set_node_features(node, features)
    
    return gnn_graph


def create_grid_graph(rows: int, 
                      cols: int, 
                      feature_dim: int = 2,
                      seed: Optional[int] = None) -> GNNGraph:
    """
    Create a grid graph with given dimensions.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        feature_dim: Dimension of node feature vectors
        seed: Random seed for reproducibility
        
    Returns:
        GNNGraph object
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create grid graph
    G = nx.grid_2d_graph(rows, cols)
    gnn_graph = GNNGraph(G)
    
    # Add random features to nodes
    for node in G.nodes():
        features = np.random.randn(feature_dim)
        gnn_graph.set_node_features(node, features)
    
    return gnn_graph


def create_web_graph(num_pages: int, 
                     link_probability: float = 0.1,
                     feature_dim: int = 2,
                     seed: Optional[int] = None) -> GNNGraph:
    """
    Create a directed graph representing web pages and links between them.
    
    Args:
        num_pages: Number of web pages (nodes)
        link_probability: Probability of a link between any two pages
        feature_dim: Dimension of node feature vectors
        seed: Random seed for reproducibility
        
    Returns:
        GNNGraph object with directed edges
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create random directed graph
    G = nx.gnp_random_graph(num_pages, link_probability, directed=True, seed=seed)
    gnn_graph = GNNGraph(G)
    
    # Add random features to nodes, simulating page features
    for node in G.nodes():
        features = np.random.randn(feature_dim)
        gnn_graph.set_node_features(node, features)
    
    return gnn_graph
