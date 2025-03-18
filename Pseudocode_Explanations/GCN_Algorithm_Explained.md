# Graph Convolutional Network (GCN) Algorithm Explained

## Algorithm/Pseudocode

```
Algorithm: Graph Convolutional Network (GCN)
Input: Graph G = (V, E) with features X ∈ ℝⁿˣᵈ and adjacency matrix A ∈ ℝⁿˣⁿ
Output: Node representations Z ∈ ℝⁿˣᵏ for downstream tasks

1. Compute normalized adjacency matrix with self-loops:
   Ã = A + I_n (Add self-loops)
   D̃ = diag(∑j Ãij) (Compute degree matrix)
   Â = D̃^(-1/2) Ã D̃^(-1/2) (Normalize)

2. For each layer l from 1 to L:
   a. If l = 1:
      H⁽¹⁾ = ReLU(Â · X · W⁽⁰⁾)
   b. If 1 < l < L:
      H⁽ᵏ⁾ = ReLU(Â · H⁽ᵏ⁻¹⁾ · W⁽ᵏ⁻¹⁾)
   c. If l = L (output layer):
      Z = H⁽ᴸ⁾ = Â · H⁽ᴸ⁻¹⁾ · W⁽ᴸ⁻¹⁾

3. Return Z
```

## ELI5: Graph Convolutional Networks

Imagine you and your friends are passing notes to each other in a classroom. Each person (you and your friends) is a node in a graph, and the connections between you (who passes notes to whom) are the edges.

Now, let's say each person has some information (like their favorite color, age, etc.) - these are your node features. A Graph Convolutional Network (GCN) is like a system for sharing and combining this information across the classroom.

Here's how it works:
1. Everyone decides to also pass a note to themselves (self-loops).
2. Everyone prepares to receive notes from their friends (and themselves).
3. When you get notes from your friends, you don't want to be influenced too much by friends who send lots of notes to many people. So you divide the importance of each note by how many friends that person has (normalization).
4. You combine all the information you received with your own information using a special formula (weights and activation function).
5. Everyone does this at the same time, and you might repeat this process a few times (multiple layers).
6. At the end, everyone has a new piece of information that combines their original information with what they learned from their friends.

This new information helps you understand more about your place in the classroom and can be used to answer questions like "which group of friends do you belong to?" or "are you more like person A or person B?"

## Line-by-Line Explanation

### 1. Compute normalized adjacency matrix with self-loops:

```
Ã = A + I_n (Add self-loops)
D̃ = diag(∑j Ãij) (Compute degree matrix)
Â = D̃^(-1/2) Ã D̃^(-1/2) (Normalize)
```

- **Ã = A + I_n**: We add an identity matrix I_n to the adjacency matrix A. This means we're adding self-loops to each node in the graph, allowing nodes to retain their own information.
- **D̃ = diag(∑j Ãij)**: We compute the degree matrix D̃, which is a diagonal matrix where each diagonal element i is the sum of the i-th row of Ã (the number of connections for node i, including the self-loop).
- **Â = D̃^(-1/2) Ã D̃^(-1/2)**: We normalize the adjacency matrix using the degree matrix. This symmetric normalization ensures that nodes with many connections don't overly influence their neighbors.

### 2. For each layer l from 1 to L:

```
a. If l = 1:
   H⁽¹⁾ = ReLU(Â · X · W⁽⁰⁾)
b. If 1 < l < L:
   H⁽ᵏ⁾ = ReLU(Â · H⁽ᵏ⁻¹⁾ · W⁽ᵏ⁻¹⁾)
c. If l = L (output layer):
   Z = H⁽ᴸ⁾ = Â · H⁽ᴸ⁻¹⁾ · W⁽ᴸ⁻¹⁾
```

- For the first layer, we multiply the normalized adjacency matrix Â with the input features X and a weight matrix W⁽⁰⁾, then apply a ReLU activation function.
- For the hidden layers, we multiply the normalized adjacency matrix Â with the previous layer's output H⁽ᵏ⁻¹⁾ and a weight matrix W⁽ᵏ⁻¹⁾, then apply a ReLU activation function.
- For the final layer, we perform the same operation but typically don't apply an activation function (or use a different one suitable for the task, like softmax for classification).

### 3. Return Z

The final output Z contains the learned node representations, which can be used for node classification, graph classification, link prediction, etc.

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

Let's go through the GCN algorithm step by step:

### Step 1: Compute normalized adjacency matrix with self-loops

Add self-loops to the adjacency matrix:
```
Ã = A + I_3 = [0, 1, 0]   [1, 0, 0]   [1, 1, 0]
               [1, 0, 1] + [0, 1, 0] = [1, 1, 1]
               [0, 1, 0]   [0, 0, 1]   [0, 1, 1]
```

Compute the degree matrix (sum of each row):
```
D̃ = diag([2, 3, 2]) = [2, 0, 0]
                       [0, 3, 0]
                       [0, 0, 2]
```

Compute D̃^(-1/2):
```
D̃^(-1/2) = [1/√2,   0,   0]
            [  0, 1/√3,   0]
            [  0,   0, 1/√2]
```

Normalize the adjacency matrix:
```
Â = D̃^(-1/2) Ã D̃^(-1/2)
```

Let's compute this by hand:
```
Â = [1/√2,   0,   0] [1, 1, 0] [1/√2,   0,   0]
    [  0, 1/√3,   0] [1, 1, 1] [  0, 1/√3,   0]
    [  0,   0, 1/√2] [0, 1, 1] [  0,   0, 1/√2]
```

Working through the matrix multiplication:
```
Â = [1/√2, 1/√2,    0]
    [1/√3, 1/√3, 1/√3]
    [   0, 1/√2, 1/√2]
```

Continuing with the multiplication:
```
Â = [1/2, 1/(√2·√3), 0]
    [1/(√3·√2), 1/3, 1/(√3·√2)]
    [0, 1/(√2·√3), 1/2]
```

Simplifying:
```
Â = [1/2, 1/√6, 0]
    [1/√6, 1/3, 1/√6]
    [0, 1/√6, 1/2]
```

### Step 2: Apply the first layer

Let's assume we want to transform the 2-dimensional features into 3-dimensional features. So our weight matrix W⁽⁰⁾ is of shape 2×3:

```
W⁽⁰⁾ = [0.1, 0.2, 0.3]
       [0.4, 0.5, 0.6]
```

First, let's compute X · W⁽⁰⁾:
```
X · W⁽⁰⁾ = [1, 2] · [0.1, 0.2, 0.3] = [1·0.1 + 2·0.4, 1·0.2 + 2·0.5, 1·0.3 + 2·0.6] = [0.9, 1.2, 1.5]
           [3, 4]   [0.4, 0.5, 0.6]   [3·0.1 + 4·0.4, 3·0.2 + 4·0.5, 3·0.3 + 4·0.6]   [1.9, 2.6, 3.3]
           [5, 6]                      [5·0.1 + 6·0.4, 5·0.2 + 6·0.5, 5·0.3 + 6·0.6]   [2.9, 4.0, 5.1]
```

Now, let's compute Â · (X · W⁽⁰⁾):
```
Â · (X · W⁽⁰⁾) = [1/2, 1/√6, 0] · [0.9, 1.2, 1.5] = [(0.9·1/2 + 1.9·1/√6), (1.2·1/2 + 2.6·1/√6), (1.5·1/2 + 3.3·1/√6)]
                 [1/√6, 1/3, 1/√6] [1.9, 2.6, 3.3]   [(0.9·1/√6 + 1.9·1/3 + 2.9·1/√6), (1.2·1/√6 + 2.6·1/3 + 4.0·1/√6), (1.5·1/√6 + 3.3·1/3 + 5.1·1/√6)]
                 [0, 1/√6, 1/2]     [2.9, 4.0, 5.1]   [(1.9·1/√6 + 2.9·1/2), (2.6·1/√6 + 4.0·1/2), (3.3·1/√6 + 5.1·1/2)]
```

Computing the first row:
```
(0.9·1/2 + 1.9·1/√6) = 0.45 + 1.9/√6 ≈ 0.45 + 0.78 = 1.23
(1.2·1/2 + 2.6·1/√6) = 0.6 + 2.6/√6 ≈ 0.6 + 1.06 = 1.66
(1.5·1/2 + 3.3·1/√6) = 0.75 + 3.3/√6 ≈ 0.75 + 1.35 = 2.10
```

Similarly for the other rows, we get:
```
H⁽¹⁾ (before ReLU) ≈ [1.23, 1.66, 2.10]
                      [1.60, 2.17, 2.75]
                      [1.72, 2.36, 3.01]
```

After applying ReLU (which keeps positive values and sets negative values to 0), since all values are positive, H⁽¹⁾ remains unchanged:
```
H⁽¹⁾ = ReLU([1.23, 1.66, 2.10]) = [1.23, 1.66, 2.10]
                                   [1.60, 2.17, 2.75]
                                   [1.72, 2.36, 3.01]
```

This is the output of the first layer. If we had more layers, we would continue this process using H⁽¹⁾ as input to the next layer.

In a typical GCN, we might have 2-3 layers, with the final layer producing outputs used for downstream tasks like node classification or link prediction.
