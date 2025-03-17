import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
import urllib.request

def load_data(dataset="cora"):
    """
    Load citation network dataset (cora, citeseer, or pubmed).
    This is a simplified version focusing on Cora dataset.
    
    Parameters
    ----------
    dataset : str
        The name of the dataset ("cora", "citeseer", or "pubmed")
        
    Returns
    -------
    adj : scipy.sparse.csr_matrix
        The adjacency matrix of the graph
    features : scipy.sparse.csr_matrix
        The feature vectors for each node
    labels : numpy.ndarray
        The labels for each node
    idx_train : numpy.ndarray
        The indices of training nodes
    idx_val : numpy.ndarray
        The indices of validation nodes
    idx_test : numpy.ndarray
        The indices of test nodes
    """
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset == "cora":
        # URLs for Cora dataset
        url_data = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        data_path = os.path.join(data_dir, 'cora.tgz')
        extracted_dir = os.path.join(data_dir, 'cora')
        
        # Download the dataset if it doesn't exist
        if not os.path.exists(extracted_dir):
            print(f"Downloading {dataset} dataset...")
            urllib.request.urlretrieve(url_data, data_path)
            # Extract the dataset
            import tarfile
            with tarfile.open(data_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("Download completed!")
        
        # Load the data from the extracted files
        content_file = os.path.join(extracted_dir, "cora.content")
        cite_file = os.path.join(extracted_dir, "cora.cites")
        
        # Load content (features and labels)
        content = np.genfromtxt(content_file, dtype=np.dtype(str))
        idx = np.array(content[:, 0], dtype=np.int32)
        features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
        labels = np.array([np.where(l == np.unique(content[:, -1]))[0][0] 
                        for l in content[:, -1]])
        
        # Map from paper ID to index in feature matrix
        idx_map = {j: i for i, j in enumerate(idx)}
        
        # Load citations (edges)
        edges = np.genfromtxt(cite_file, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges.flatten())), 
                       dtype=np.int32).reshape(edges.shape)
        
        # Create adjacency matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(labels.shape[0], labels.shape[0]),
                         dtype=np.float32)
        
        # Make the adjacency matrix symmetric (undirected graph)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        # Split the data into train/val/test
        idx_train = range(140)
        idx_val = range(140, 640)
        idx_test = range(1708, 2708)
        
        return adj, features, labels, idx_train, idx_val, idx_test
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented yet")

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        The adjacency matrix of the graph
        
    Returns
    -------
    scipy.sparse.csr_matrix
        The normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """
    Preprocessing of adjacency matrix for GCN model:
    - Add self-loops
    - Apply symmetric normalization
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        The adjacency matrix of the graph
        
    Returns
    -------
    scipy.sparse.coo_matrix
        The preprocessed adjacency matrix
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    
    Parameters
    ----------
    sparse_mx : scipy.sparse.csr_matrix
        The sparse matrix
        
    Returns
    -------
    tuple
        (coords, values, shape) representation
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    
    Parameters
    ----------
    sparse_mx : scipy.sparse.csr_matrix
        The sparse matrix
        
    Returns
    -------
    torch.sparse.FloatTensor
        The torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    """
    Calculate classification accuracy.
    
    Parameters
    ----------
    output : torch.Tensor
        The model output
    labels : torch.Tensor
        The ground-truth labels
        
    Returns
    -------
    float
        The accuracy
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def visualize_graph(adj_matrix, node_labels=None, node_colors=None, layout="spring", figsize=(10, 8)):
    """
    Visualize a graph from its adjacency matrix.
    
    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix or numpy.ndarray
        The adjacency matrix of the graph
    node_labels : list, optional
        Labels for each node
    node_colors : list, optional
        Colors for each node
    layout : str, optional
        The layout algorithm to use ('spring', 'circular', 'random', etc.)
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Convert adjacency matrix to NetworkX graph
    if isinstance(adj_matrix, np.ndarray):
        G = nx.from_numpy_array(adj_matrix)
    else:  # sparse matrix
        G = nx.from_scipy_sparse_matrix(adj_matrix)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)  # default to spring layout
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors if node_colors else 'skyblue',
                          node_size=100, 
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    # Add node labels if provided
    if node_labels:
        labels = {i: str(label) for i, label in enumerate(node_labels)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f"Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    plt.axis('off')
    
    return plt.gcf()

def plot_training_curve(train_losses, val_losses, train_accs, val_accs, figsize=(12, 5)):
    """
    Plot training and validation curves.
    
    Parameters
    ----------
    train_losses : list
        Training losses
    val_losses : list
        Validation losses
    train_accs : list
        Training accuracies
    val_accs : list
        Validation accuracies
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'bo-', label='Training loss')
    ax1.plot(epochs, val_losses, 'ro-', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'bo-', label='Training accuracy')
    ax2.plot(epochs, val_accs, 'ro-', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig
