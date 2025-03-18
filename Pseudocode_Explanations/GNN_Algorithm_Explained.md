# Graph Neural Network (GNN) Algorithm Explained

## Algorithm/Pseudocode

```
Algorithm: Message Passing Graph Neural Network (MPNN)
Input: Graph G = (V, E) with node features X ∈ ℝⁿˣᵈ, edge features eij (optional)
Output: Node representations Z ∈ ℝⁿˣᵏ for downstream tasks

1. Initialize node representations: H⁽⁰⁾ = X

2. For each layer l from 1 to L:
   a. Message Passing Phase:
      For each node i ∈ V:
         i. Compute messages from all neighbors j ∈ N(i):
            m_ij^(l) = MESSAGE(h_i^(l-1), h_j^(l-1), e_ij)
         
         ii. Aggregate incoming messages:
             a_i^(l) = AGGREGATE({m_ij^(l) | j ∈ N(i)})
   
   b. Update Phase:
      For each node i ∈ V:
         h_i^(l) = UPDATE(h_i^(l-1), a_i^(l))

3. For each node i ∈ V:
   z_i = READOUT(h_i^(L))

4. Return Z = {z_i | i ∈ V}
```

Where:
- **MESSAGE** function computes a message based on the source node, target node, and edge features
- **AGGREGATE** function combines messages from all neighbors (e.g., sum, mean, max)
- **UPDATE** function updates the node representation based on its previous state and aggregated messages
- **READOUT** function transforms the final node representation for the target task

## ELI5: Graph Neural Networks

Imagine you and your friends are playing a game where everyone stands in a circle. Each person (including you) has a card with some information written on it. This information could be your age, favorite color, or hobbies - these are your "node features."

The game has several rounds, and in each round:

1. **Message Creation**: Everyone creates a special message to share with their neighbors. The message depends on their own card and possibly the relationship with their neighbors.

2. **Message Passing**: Everyone passes these messages to their neighbors. You only get messages from people directly connected to you.

3. **Information Gathering**: You collect all the messages sent to you.

4. **Update Your Knowledge**: You update your card based on what was originally written on it and the messages you received.

After playing several rounds:

5. **Final Answer**: Everyone creates a final answer based on the information they've collected over all rounds.

This game is essentially how a Graph Neural Network works! The people are "nodes" in a graph, the connections between people are "edges," and the information on the cards are "features." The GNN learns how to create useful messages, how to combine them, and how to update information in a way that helps solve problems like classifying nodes, predicting links between nodes, or understanding the entire graph.

## Line-by-Line Explanation

### 1. Initialize node representations:
```
H⁽⁰⁾ = X
```
We start with the original node features as our initial node representations.

### 2. For each layer l from 1 to L:

#### a. Message Passing Phase:
```
For each node i ∈ V:
   i. Compute messages from all neighbors j ∈ N(i):
      m_ij^(l) = MESSAGE(h_i^(l-1), h_j^(l-1), e_ij)
```
- Each node i computes a "message" for each of its neighbors j.
- The message function takes as input the current representations of both nodes and potentially the edge features.
- Different GNN variants define different message functions (for example, in GCNs, the message is simply the neighbor's representation multiplied by a normalized factor).

```
   ii. Aggregate incoming messages:
       a_i^(l) = AGGREGATE({m_ij^(l) | j ∈ N(i)})
```
- Node i collects all messages from its neighbors and aggregates them.
- Common aggregation functions include sum, mean, max, or more complex functions like attention-weighted sum (as in GAT).
- This step ensures that the model can handle varying numbers of neighbors.

#### b. Update Phase:
```
For each node i ∈ V:
   h_i^(l) = UPDATE(h_i^(l-1), a_i^(l))
```
- Each node updates its representation based on its previous representation and the aggregated messages.
- This is typically implemented as a neural network that combines the previous state and the aggregated message.
- The update function introduces non-linearity and allows the node to selectively incorporate new information.

### 3. Readout Phase:
```
For each node i ∈ V:
   z_i = READOUT(h_i^(L))
```
- After L layers of message passing, each node has a final representation h_i^(L).
- The readout function transforms this representation into the final output z_i.
- This step is task-specific. For node classification, it might be a simple linear layer followed by softmax. For graph-level tasks, it might involve pooling across all nodes.

### 4. Return Z:
```
Return Z = {z_i | i ∈ V}
```
- The algorithm returns the final representations for all nodes, which can be used for downstream tasks.

## By-Hand Calculation Example

Let's consider a tiny graph with 3 nodes and the following adjacency matrix:

```
A = [0, 1, 0]
    [1, 0, 1]
    [0, 1, 0]
```

And let's say each node has 2-dimensional features:

```
X = [1, 2]
    [3, 4]
    [5, 6]
```

Let's go through a simple GNN algorithm step by step using the following functions:

- **MESSAGE**: m_ij^(l) = W_m · h_j^(l-1) (where W_m is a weight matrix)
- **AGGREGATE**: a_i^(l) = Σ_j∈N(i) m_ij^(l) (sum aggregation)
- **UPDATE**: h_i^(l) = ReLU(W_u · [h_i^(l-1) || a_i^(l)]) (concatenation followed by a linear layer and ReLU)

Let's assume the following weight matrices:
```
W_m = [0.1, 0.2]
      [0.3, 0.4]

W_u = [0.5, 0.6, 0.7, 0.8]
      [0.9, 1.0, 1.1, 1.2]
```

### Step 1: Initialize node representations
```
H⁽⁰⁾ = X = [1, 2]
           [3, 4]
           [5, 6]
```

### Step 2: Apply the first layer

#### a. Message Passing Phase:

For node 1, compute messages from its neighbors (only node 2):
```
m_12^(1) = W_m · h_2^(0) = [0.1, 0.2] · [3, 4]^T = 0.1*3 + 0.2*4 = [0.3 + 0.8] = [1.1]
                           [0.3, 0.4]             0.3*3 + 0.4*4 = [0.9 + 1.6] = [2.5]
```

Aggregate messages for node 1:
```
a_1^(1) = Σ_j∈N(1) m_1j^(1) = m_12^(1) = [1.1, 2.5]
```

For node 2, compute messages from its neighbors (nodes 1 and 3):
```
m_21^(1) = W_m · h_1^(0) = [0.1, 0.2] · [1, 2]^T = 0.1*1 + 0.2*2 = [0.1 + 0.4] = [0.5]
                           [0.3, 0.4]             0.3*1 + 0.4*2 = [0.3 + 0.8] = [1.1]

m_23^(1) = W_m · h_3^(0) = [0.1, 0.2] · [5, 6]^T = 0.1*5 + 0.2*6 = [0.5 + 1.2] = [1.7]
                           [0.3, 0.4]             0.3*5 + 0.4*6 = [1.5 + 2.4] = [3.9]
```

Aggregate messages for node 2:
```
a_2^(1) = Σ_j∈N(2) m_2j^(1) = m_21^(1) + m_23^(1) = [0.5, 1.1] + [1.7, 3.9] = [2.2, 5.0]
```

For node 3, compute messages from its neighbors (only node 2):
```
m_32^(1) = W_m · h_2^(0) = [0.1, 0.2] · [3, 4]^T = [1.1, 2.5] (same as m_12^(1))
```

Aggregate messages for node 3:
```
a_3^(1) = Σ_j∈N(3) m_3j^(1) = m_32^(1) = [1.1, 2.5]
```

#### b. Update Phase:

For node 1, update its representation:
```
h_1^(1) = ReLU(W_u · [h_1^(0) || a_1^(1)])
        = ReLU(W_u · [1, 2, 1.1, 2.5])
        = ReLU([0.5, 0.6, 0.7, 0.8] · [1, 2, 1.1, 2.5]^T)
                [0.9, 1.0, 1.1, 1.2]
        = ReLU([0.5*1 + 0.6*2 + 0.7*1.1 + 0.8*2.5, 0.9*1 + 1.0*2 + 1.1*1.1 + 1.2*2.5])
        = ReLU([0.5 + 1.2 + 0.77 + 2.0, 0.9 + 2.0 + 1.21 + 3.0])
        = ReLU([4.47, 7.11])
        = [4.47, 7.11] (since all values are positive)
```

For node 2, update its representation:
```
h_2^(1) = ReLU(W_u · [h_2^(0) || a_2^(1)])
        = ReLU(W_u · [3, 4, 2.2, 5.0])
        = ReLU([0.5*3 + 0.6*4 + 0.7*2.2 + 0.8*5.0, 0.9*3 + 1.0*4 + 1.1*2.2 + 1.2*5.0])
        = ReLU([1.5 + 2.4 + 1.54 + 4.0, 2.7 + 4.0 + 2.42 + 6.0])
        = ReLU([9.44, 15.12])
        = [9.44, 15.12]
```

For node 3, update its representation:
```
h_3^(1) = ReLU(W_u · [h_3^(0) || a_3^(1)])
        = ReLU(W_u · [5, 6, 1.1, 2.5])
        = ReLU([0.5*5 + 0.6*6 + 0.7*1.1 + 0.8*2.5, 0.9*5 + 1.0*6 + 1.1*1.1 + 1.2*2.5])
        = ReLU([2.5 + 3.6 + 0.77 + 2.0, 4.5 + 6.0 + 1.21 + 3.0])
        = ReLU([8.87, 14.71])
        = [8.87, 14.71]
```

So after the first layer, our node representations are:
```
H⁽¹⁾ = [4.47, 7.11]
       [9.44, 15.12]
       [8.87, 14.71]
```

In a typical GNN, we would continue this process for multiple layers to capture higher-order neighborhood information. After the final layer, we would apply a readout function to get the final node representations for our downstream task (e.g., node classification, link prediction, or graph classification).

What makes GNNs powerful is their ability to learn from both the feature information and the graph structure simultaneously. By iteratively passing messages between connected nodes, GNNs allow each node to incorporate information from its local neighborhood, and with deeper layers, from more distant parts of the graph as well.

## GNN Variants

The general MPNN framework described above encompasses many popular GNN variants, which differ in how they define the message, aggregate, and update functions:

1. **Graph Convolutional Networks (GCN)**: Uses a simple message function where each neighbor contributes equally (weighted by degree normalization).

2. **Graph Attention Networks (GAT)**: Uses attention mechanisms to weigh the importance of different neighbors dynamically.

3. **GraphSAGE**: Uses various aggregators (mean, max, LSTM) and samples a fixed number of neighbors to handle large graphs.

4. **Gated Graph Neural Networks (GGNN)**: Uses GRU cells for the update function to better capture long-term dependencies.

5. **Graph Isomorphism Network (GIN)**: Designed to maximize the discriminative power of GNNs for graph-level tasks.

Each variant has its strengths and is suited for different types of problems and graph structures.
