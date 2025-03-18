# Graph Attention Network (GAT) Algorithm Explained

## Algorithm/Pseudocode

```
Algorithm: Graph Attention Network (GAT)
Input: Graph G = (V, E) with features X ∈ ℝⁿˣᵈ and adjacency matrix A ∈ ℝⁿˣⁿ
Output: Node representations Z ∈ ℝⁿˣᵏ for downstream tasks

1. Initialize node representations: H⁽⁰⁾ = X

2. For each layer l from 1 to L:
   a. For each node i ∈ V:
      i. For each neighbor j ∈ N(i) ∪ {i} (neighborhood of i including self):
         - Compute attention coefficient:
           e_ij = LeakyReLU(a^T · [W · h_i || W · h_j])
           where a is an attention vector, W is the weight matrix, and || is concatenation
      
      ii. Normalize attention coefficients using softmax:
          α_ij = softmax_j(e_ij) = exp(e_ij) / ∑_k∈N(i)∪{i} exp(e_ik)
      
      iii. Compute the new representation for node i:
           h_i^(l+1) = σ(∑_j∈N(i)∪{i} α_ij · W · h_j^(l))
           where σ is an activation function (typically ELU)
   
   b. Optionally, use multi-head attention:
      i. Compute K different attention mechanisms in parallel
      ii. For hidden layers, concatenate the K attention heads:
          h_i^(l+1) = ||_k=1^K σ(∑_j∈N(i)∪{i} α_ij^k · W^k · h_j^(l))
      iii. For the output layer, average the K attention heads:
           h_i^(L) = σ(1/K · ∑_k=1^K ∑_j∈N(i)∪{i} α_ij^k · W^k · h_j^(L-1))

3. Return Z = H⁽ᴸ⁾
```

## ELI5: Graph Attention Networks

Imagine you're in a classroom again with your friends, and everyone has their own information (like favorite colors, hobbies, etc.). But this time, instead of treating all your friends equally, you pay more attention to some friends than others.

Here's how a Graph Attention Network (GAT) works:

1. Everyone starts with their own information (node features).
2. When you want to update your information, you look at what all your friends know.
3. But here's the special part: you decide how much to listen to each friend. Maybe you pay more attention to your best friend and less to someone you don't know well.
4. To decide who to pay attention to, you look at both your information and your friend's information and use a special formula to calculate an "attention score".
5. You make sure all these attention scores add up to 100% (that's the softmax part).
6. You update your information by taking a weighted average of everyone's information, where the weights are your attention scores.
7. To make things even better, you might have multiple "attention mechanisms" (like looking at your friends in terms of who shares your favorite color, and separately in terms of who shares your hobbies). Then you combine all these different perspectives.

By the end, your updated information reflects what you learned from your friends, but with more influence from the friends you decided were more important to listen to for this particular topic.

## Line-by-Line Explanation

### 1. Initialize node representations:
```
H⁽⁰⁾ = X
```
We start with the original node features as our initial node representations.

### 2. For each layer l from 1 to L:

#### a. For each node i ∈ V:

##### i. Compute attention coefficient:
```
e_ij = LeakyReLU(a^T · [W · h_i || W · h_j])
```
- We apply a linear transformation `W` to both the node `i` and its neighbor `j`.
- We concatenate these transformed features (`||` represents concatenation).
- We then compute a dot product with an attention vector `a`.
- Finally, we apply LeakyReLU activation to introduce non-linearity.
- This gives us an unnormalized attention coefficient `e_ij`, which tells us how much node `i` should pay attention to node `j`.

##### ii. Normalize attention coefficients:
```
α_ij = softmax_j(e_ij) = exp(e_ij) / ∑_k∈N(i)∪{i} exp(e_ik)
```
- We normalize the attention coefficients using softmax, so they sum to 1 across all neighbors.
- This ensures that the attention weights form a valid probability distribution.

##### iii. Compute new representation:
```
h_i^(l+1) = σ(∑_j∈N(i)∪{i} α_ij · W · h_j^(l))
```
- We compute a weighted sum of the transformed neighbor features, where the weights are the attention coefficients.
- We apply an activation function `σ` (typically ELU or ReLU) to introduce non-linearity.

#### b. Multi-head attention:
```
h_i^(l+1) = ||_k=1^K σ(∑_j∈N(i)∪{i} α_ij^k · W^k · h_j^(l))
```
- To stabilize learning and capture different aspects of the graph, we use multiple attention heads.
- Each head has its own set of parameters (W^k and a^k).
- For hidden layers, we concatenate the outputs from all heads.
- For the output layer, we typically average the outputs from all heads.

### 3. Return Z = H⁽ᴸ⁾

The final output Z contains the learned node representations, which can be used for downstream tasks.

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

Let's go through the GAT algorithm step by step, using a simplified version with a single attention head:

### Step 1: Initialize node representations
```
H⁽⁰⁾ = X = [1, 2]
           [3, 4]
           [5, 6]
```

### Step 2: Apply the first layer

Let's use a weight matrix W to transform the 2-dimensional features into 3-dimensional features:
```
W = [0.1, 0.2, 0.3]
    [0.4, 0.5, 0.6]
```

First, let's compute W · h_i for each node:
```
W · h_1 = [0.1, 0.2, 0.3] · [1, 2]^T = 0.1*1 + 0.4*2 = [0.9, 1.2, 1.5]
W · h_2 = [0.1, 0.2, 0.3] · [3, 4]^T = 0.1*3 + 0.4*4 = [1.9, 2.6, 3.3]
W · h_3 = [0.1, 0.2, 0.3] · [5, 6]^T = 0.1*5 + 0.4*6 = [2.9, 4.0, 5.1]
```

Now, let's compute the attention coefficients. For simplicity, let's assume our attention vector `a` is [0.1, 0.2, 0.1, 0.2, 0.1, 0.2], designed to compute attention over concatenated 3-dimensional vectors.

For node 1, its neighbors are node 2 and itself (node 1):
```
e_11 = LeakyReLU(a^T · [W · h_1 || W · h_1])
     = LeakyReLU([0.1, 0.2, 0.1, 0.2, 0.1, 0.2] · [0.9, 1.2, 1.5, 0.9, 1.2, 1.5]^T)
     = LeakyReLU(0.1*0.9 + 0.2*1.2 + 0.1*1.5 + 0.2*0.9 + 0.1*1.2 + 0.2*1.5)
     = LeakyReLU(0.09 + 0.24 + 0.15 + 0.18 + 0.12 + 0.3)
     = LeakyReLU(1.08)
     = 1.08 (since LeakyReLU(x) = x for x > 0)

e_12 = LeakyReLU(a^T · [W · h_1 || W · h_2])
     = LeakyReLU([0.1, 0.2, 0.1, 0.2, 0.1, 0.2] · [0.9, 1.2, 1.5, 1.9, 2.6, 3.3]^T)
     = LeakyReLU(0.1*0.9 + 0.2*1.2 + 0.1*1.5 + 0.2*1.9 + 0.1*2.6 + 0.2*3.3)
     = LeakyReLU(0.09 + 0.24 + 0.15 + 0.38 + 0.26 + 0.66)
     = LeakyReLU(1.78)
     = 1.78
```

Normalize using softmax:
```
α_11 = exp(e_11) / (exp(e_11) + exp(e_12))
     = exp(1.08) / (exp(1.08) + exp(1.78))
     = 2.94 / (2.94 + 5.93)
     = 2.94 / 8.87
     ≈ 0.33

α_12 = exp(e_12) / (exp(e_11) + exp(e_12))
     = exp(1.78) / (exp(1.08) + exp(1.78))
     = 5.93 / (2.94 + 5.93)
     = 5.93 / 8.87
     ≈ 0.67
```

Now, compute the new representation for node 1:
```
h_1^(1) = σ(α_11 · W · h_1 + α_12 · W · h_2)
        = σ(0.33 * [0.9, 1.2, 1.5] + 0.67 * [1.9, 2.6, 3.3])
        = σ([0.33*0.9 + 0.67*1.9, 0.33*1.2 + 0.67*2.6, 0.33*1.5 + 0.67*3.3])
        = σ([0.297 + 1.273, 0.396 + 1.742, 0.495 + 2.211])
        = σ([1.57, 2.14, 2.71])
        = [1.57, 2.14, 2.71] (assuming ELU or ReLU as activation, which keeps positive values unchanged)
```

Similarly, we can compute h_2^(1) and h_3^(1) by considering their respective neighborhoods:

For node 2, its neighbors are nodes 1, 3, and itself (node 2). Let's compute the attention coefficients:
```
e_21 = LeakyReLU(a^T · [W · h_2 || W · h_1]) ≈ 1.78 (same as e_12 by symmetry)
e_22 = LeakyReLU(a^T · [W · h_2 || W · h_2]) ≈ 2.33 (similar calculation as e_11)
e_23 = LeakyReLU(a^T · [W · h_2 || W · h_3]) ≈ 2.87
```

Normalize using softmax:
```
α_21 ≈ 0.22
α_22 ≈ 0.37
α_23 ≈ 0.41
```

Compute the new representation for node 2:
```
h_2^(1) = σ(α_21 · W · h_1 + α_22 · W · h_2 + α_23 · W · h_3)
        ≈ σ(0.22 * [0.9, 1.2, 1.5] + 0.37 * [1.9, 2.6, 3.3] + 0.41 * [2.9, 4.0, 5.1])
        ≈ σ([0.198 + 0.703 + 1.189, 0.264 + 0.962 + 1.64, 0.33 + 1.221 + 2.091])
        ≈ σ([2.09, 2.87, 3.64])
        ≈ [2.09, 2.87, 3.64]
```

For node 3, similar calculations would give us h_3^(1).

This is the output of the first layer. In a typical GAT, we might have 2-3 layers, with multi-head attention at each layer to stabilize learning and capture different aspects of the graph information.

Note that the GAT algorithm provides a more flexible way to aggregate neighborhood information compared to GCN, as it allows the model to assign different importance to different neighbors, even if the graph structure is fixed.
