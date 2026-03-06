"""
Graph Neural Network Fundamentals
===================================
Covers: Graph representations, message passing, GCN, GraphSAGE, GAT.
All implemented from scratch in PyTorch (no PyG dependency for core code).
"""

import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. Graph Data Structures
# ---------------------------------------------------------------------------

class GraphData:
    """
    Simple graph container.
    
    Attributes:
        x:          Node feature matrix          (num_nodes, feature_dim)
        edge_index: Edge list in COO format       (2, num_edges)
        y:          Node or graph labels           (num_nodes,) or scalar
        adj:        Dense adjacency matrix         (num_nodes, num_nodes)
    """

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor,
                 y: Optional[torch.Tensor] = None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.num_nodes = x.size(0)

    @property
    def adj(self) -> torch.Tensor:
        """Build dense adjacency matrix from edge_index."""
        A = torch.zeros(self.num_nodes, self.num_nodes)
        src, dst = self.edge_index
        A[src, dst] = 1.0
        return A

    @staticmethod
    def from_adjacency(adj: np.ndarray, features: np.ndarray,
                       labels: Optional[np.ndarray] = None) -> "GraphData":
        """Create GraphData from a dense adjacency matrix."""
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        return GraphData(x, edge_index, y)


# ---------------------------------------------------------------------------
# 2. Graph Convolutional Network (GCN) — Kipf & Welling 2017
# ---------------------------------------------------------------------------

class GCNConv(nn.Module):
    """
    Single GCN layer.
    
    Message passing rule (with self-loops and symmetric normalization):
        H' = D̃^{-1/2} Ã D̃^{-1/2} H W
    
    where Ã = A + I (adjacency + self-loops), D̃ = degree matrix of Ã.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Add self-loops
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = adj_hat.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt.unsqueeze(0) * adj_hat * deg_inv_sqrt.unsqueeze(1)
        # Message passing
        out = norm @ x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """Two-layer GCN for node classification."""

    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x  # raw logits


# ---------------------------------------------------------------------------
# 3. GraphSAGE — Hamilton et al. 2017
# ---------------------------------------------------------------------------

class SAGEConv(nn.Module):
    """
    GraphSAGE layer with mean aggregation.
    
    For each node, aggregate neighbor features, concatenate with
    the node's own features, then project:
    
        h_N = MEAN({h_j : j ∈ N(i)})
        h_i' = W · [h_i || h_N]
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Mean aggregation of neighbors
        adj_norm = adj / (adj.sum(dim=1, keepdim=True).clamp(min=1))
        neighbor_agg = adj_norm @ x
        # Concatenate self + neighbor
        out = torch.cat([x, neighbor_agg], dim=-1)
        out = self.linear(out)
        return out


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE for node classification."""

    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x


# ---------------------------------------------------------------------------
# 4. Graph Attention Network (GAT) — Veličković et al. 2018
# ---------------------------------------------------------------------------

class GATConv(nn.Module):
    """
    Single-head Graph Attention layer.
    
    Instead of fixed normalization (GCN) or uniform aggregation (SAGE),
    GAT learns attention coefficients between connected nodes:
    
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        α_ij = softmax_j(e_ij)
        h_i' = Σ_j α_ij W h_j
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 1,
                 dropout: float = 0.6, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        # x: (N, in_features) → Wh: (n_heads, N, out_features)
        Wh = torch.einsum("nf,hfo->hno", x, self.W)

        # Compute attention coefficients
        # For each head, compute e_ij = a^T [Wh_i || Wh_j]
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # (H, N, N, F)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (H, N, N, F)
        e = torch.cat([Wh_i, Wh_j], dim=-1)            # (H, N, N, 2F)
        e = self.leaky_relu(torch.einsum("hijk,hkl->hijl", e, self.a).squeeze(-1))  # (H, N, N)

        # Mask non-edges (set to -inf before softmax)
        mask = (adj == 0).unsqueeze(0)
        e = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=-1)  # (H, N, N)
        alpha = self.dropout(alpha)

        # Aggregate
        out = torch.einsum("hnm,hmo->hno", alpha, Wh)  # (H, N, out_features)

        if self.concat:
            return out.permute(1, 0, 2).reshape(N, -1)  # (N, H*out_features)
        else:
            return out.mean(dim=0)  # (N, out_features)


class GAT(nn.Module):
    """Two-layer GAT for node classification."""

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 n_heads: int = 4, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden, n_heads=n_heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * n_heads, num_classes, n_heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x


# ---------------------------------------------------------------------------
# 5. Synthetic Cora-like Dataset
# ---------------------------------------------------------------------------

def make_synthetic_citation_graph(n_nodes: int = 200, n_features: int = 50,
                                   n_classes: int = 4, edge_prob: float = 0.05,
                                   seed: int = 42) -> Tuple[GraphData, torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic citation-like graph for node classification.
    
    Returns:
        graph: GraphData object
        train_mask: Boolean mask for training nodes
        test_mask: Boolean mask for test nodes
    """
    rng = np.random.RandomState(seed)

    # Generate class-correlated features
    labels = rng.randint(0, n_classes, n_nodes)
    features = rng.randn(n_nodes, n_features).astype(np.float32)
    for c in range(n_classes):
        mask = labels == c
        features[mask] += rng.randn(n_features) * 0.5  # class-specific shift

    # Generate edges (higher probability within same class)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = edge_prob * 3 if labels[i] == labels[j] else edge_prob
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    graph = GraphData.from_adjacency(adj, features, labels)

    # Train/test split (60/40)
    perm = rng.permutation(n_nodes)
    split = int(0.6 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm[:split]] = True
    test_mask[perm[split:]] = True

    return graph, train_mask, test_mask


# ---------------------------------------------------------------------------
# 6. Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(model_name: str = "GCN", epochs: int = 200, lr: float = 0.01):
    """Train a GNN on the synthetic citation graph."""
    graph, train_mask, test_mask = make_synthetic_citation_graph()
    adj = graph.adj
    x, y = graph.x, graph.y
    n_classes = int(y.max().item()) + 1

    # Select model
    models = {
        "GCN": GCN(x.size(1), 64, n_classes),
        "GraphSAGE": GraphSAGE(x.size(1), 64, n_classes),
        "GAT": GAT(x.size(1), 16, n_classes, n_heads=4),
    }
    model = models[model_name]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x, adj)
                train_acc = (logits[train_mask].argmax(1) == y[train_mask]).float().mean()
                test_acc = (logits[test_mask].argmax(1) == y[test_mask]).float().mean()
            print(f"[{model_name}] Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

    return model


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name in ["GCN", "GraphSAGE", "GAT"]:
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        train_and_evaluate(name, epochs=200)
