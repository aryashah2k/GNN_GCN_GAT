import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def visualize_graph(G: nx.Graph, 
                    node_states: Optional[Dict[Any, np.ndarray]] = None, 
                    node_colors: Optional[List] = None,
                    title: str = "Graph Visualization",
                    figsize: Tuple[int, int] = (10, 8),
                    node_size: int = 500,
                    layout: str = "spring") -> None:
    """
    Visualize a graph with optional node states.
    
    Args:
        G: NetworkX graph object
        node_states: Dictionary mapping node IDs to state vectors
        node_colors: List of colors for nodes
        title: Plot title
        figsize: Figure size (width, height)
        node_size: Size of nodes in visualization
        layout: Graph layout algorithm ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    
    # Choose layout algorithm
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G, seed=42)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    if node_colors is None:
        node_colors = ['skyblue' for _ in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        if node_states is not None and node in node_states:
            # Format state vector for display
            state_str = np.array2string(node_states[node], precision=2, separator=',')
            labels[node] = f"{node}\n{state_str}"
        else:
            labels[node] = f"{node}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_states_over_iterations(states_history: List[Dict[Any, np.ndarray]], 
                                     G: nx.Graph,
                                     max_iterations: Optional[int] = None,
                                     figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualize how node states evolve over iterations.
    
    Args:
        states_history: List of dictionaries, each mapping node IDs to state vectors at iteration t
        G: NetworkX graph object
        max_iterations: Maximum number of iterations to display (None for all)
        figsize: Figure size (width, height)
    """
    # Limit iterations if needed
    if max_iterations is not None:
        states_history = states_history[:max_iterations]
    
    n_iterations = len(states_history)
    
    if n_iterations == 0:
        print("No state history to visualize")
        return
    
    # Get a sample state to determine dimensionality
    sample_state = next(iter(states_history[0].values()))
    state_dim = len(sample_state)
    
    # Create plot
    fig, axs = plt.subplots(n_iterations, state_dim, figsize=figsize, squeeze=False)
    
    # Set global title
    fig.suptitle("Evolution of Node States Over Iterations", fontsize=16)
    
    # Get all nodes
    nodes = list(G.nodes())
    
    # For each iteration and state dimension
    for t in range(n_iterations):
        states = states_history[t]
        
        for dim in range(state_dim):
            ax = axs[t, dim]
            
            # Get node colors based on state values for this dimension
            node_colors = []
            for node in nodes:
                if node in states:
                    # Normalize value to [0, 1] for colormap
                    val = states[node][dim]
                    # Use different colormaps for positive and negative values
                    if val >= 0:
                        node_colors.append(plt.cm.Reds(min(val, 1.0)))
                    else:
                        node_colors.append(plt.cm.Blues(min(abs(val), 1.0)))
                else:
                    node_colors.append('gray')  # Nodes without states
            
            # Draw graph with these colors
            pos = nx.spring_layout(G, seed=42)  # Use same layout for all
            nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_colors, 
                             node_size=300, with_labels=True, font_size=8)
            
            ax.set_title(f"Iteration {t+1}, Dimension {dim+1}")
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
    plt.show()


def plot_loss_history(loss_history: List[float], 
                      title: str = "Training Loss Over Epochs",
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the training loss history.
    
    Args:
        loss_history: List of loss values over training
        title: Plot title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add points at each loss value
    plt.scatter(range(len(loss_history)), loss_history, c='b', s=50, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
