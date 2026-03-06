"""
Advanced GNN Architectures
============================
Covers: Graph Isomorphism Network (GIN), graph-level classification
with pooling, link prediction, and heterogeneous graphs.
"""

import numpy as np
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import importlib
import sys
import os

# Python modules can't start with a digit, so we use importlib to load it.
_fund_path = os.path.join(os.path.dirname(__file__), "01_gnn_fundamentals.py")
_spec = importlib.util.spec_from_file_location("gnn_fundamentals", _fund_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gnn_fundamentals"] = _mod
_spec.loader.exec_module(_mod)
GraphData = _mod.GraphData


# ---------------------------------------------------------------------------
# 1. Graph Isomorphism Network (GIN) — Xu et al. 2019
# ---------------------------------------------------------------------------

class GINConv(nn.Module):
    """
    GIN layer — provably as powerful as the Weisfeiler-Leman graph
    isomorphism test.
    
    h_i' = MLP((1 + ε) · h_i + Σ_{j∈N(i)} h_j)
    
    ε is a learnable scalar.
    """

    def __init__(self, in_features: int, out_features: int, hidden: int = 64):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neighbor_sum = adj @ x
        out = (1 + self.eps) * x + neighbor_sum
        return self.mlp(out)


# ---------------------------------------------------------------------------
# 2. Graph-Level Classification with Readout Pooling
# ---------------------------------------------------------------------------

class GraphClassifier(nn.Module):
    """
    Graph-level classifier using GIN layers + global pooling.
    
    Pipeline:
        1. Multiple GIN layers produce node embeddings
        2. Global readout (sum/mean) aggregates node embeddings → graph embedding
        3. MLP classifies the graph embedding
    
    Use case: molecule property prediction, social network classification.
    """

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 n_layers: int = 3, pooling: str = "sum"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(in_features, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GINConv(hidden, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )
        self.pooling = pooling

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (total_nodes, features)
            adj: (total_nodes, total_nodes) — block-diagonal for batched graphs
            batch: (total_nodes,) — graph index for each node (for batched graphs)
        """
        for conv in self.convs:
            x = F.relu(conv(x, adj))

        # Global readout
        if batch is not None:
            # Scatter pooling per graph
            num_graphs = batch.max().item() + 1
            graph_emb = torch.zeros(num_graphs, x.size(1), device=x.device)
            if self.pooling == "sum":
                graph_emb.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            else:
                graph_emb.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
                counts = torch.zeros(num_graphs, 1, device=x.device)
                counts.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch.unsqueeze(1).float()))
                graph_emb = graph_emb / counts.clamp(min=1)
        else:
            # Single graph
            if self.pooling == "sum":
                graph_emb = x.sum(dim=0, keepdim=True)
            else:
                graph_emb = x.mean(dim=0, keepdim=True)

        return self.classifier(graph_emb)


# ---------------------------------------------------------------------------
# 3. Link Prediction
# ---------------------------------------------------------------------------

class LinkPredictor(nn.Module):
    """
    Link prediction using GCN encoder + dot-product decoder.
    
    Encoder: GCN produces node embeddings z_i
    Decoder: score(i,j) = σ(z_i · z_j)
    
    Training: positive edges (existing) vs negative edges (sampled non-edges).
    """

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden)
        self.conv2 = nn.Linear(hidden, hidden)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj_hat.sum(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt.unsqueeze(0) * adj_hat * deg_inv_sqrt.unsqueeze(1)

        z = F.relu(norm @ x @ self.conv1.weight.T + self.conv1.bias)
        z = norm @ z @ self.conv2.weight.T + self.conv2.bias
        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, adj, pos_edges, neg_edges):
        z = self.encode(x, adj)
        pos_scores = self.decode(z, pos_edges)
        neg_scores = self.decode(z, neg_edges)
        return pos_scores, neg_scores


def sample_negative_edges(adj: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Sample non-existing edges as negative examples."""
    n = adj.size(0)
    neg_src, neg_dst = [], []
    while len(neg_src) < n_samples:
        i, j = np.random.randint(0, n, 2)
        if i != j and adj[i, j] == 0:
            neg_src.append(i)
            neg_dst.append(j)
    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def train_link_prediction(epochs: int = 200):
    """Train link prediction on a synthetic graph."""
    # Generate graph
    n_nodes, n_features = 100, 32
    rng = np.random.RandomState(42)
    features = rng.randn(n_nodes, n_features).astype(np.float32)
    adj_np = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.08:
                adj_np[i, j] = adj_np[j, i] = 1.0

    x = torch.tensor(features)
    adj = torch.tensor(adj_np)
    edge_index = torch.tensor(np.array(np.nonzero(adj_np)), dtype=torch.long)

    # Hold out 20% of edges for testing
    n_edges = edge_index.size(1)
    perm = torch.randperm(n_edges)
    n_test = n_edges // 5
    test_edges = edge_index[:, perm[:n_test]]
    train_edges = edge_index[:, perm[n_test:]]

    # Remove test edges from adjacency
    train_adj = adj.clone()
    for i in range(n_test):
        s, d = test_edges[0, i].item(), test_edges[1, i].item()
        train_adj[s, d] = 0
        train_adj[d, s] = 0

    model = LinkPredictor(n_features, 64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        neg_edges = sample_negative_edges(train_adj, train_edges.size(1))
        pos_scores, neg_scores = model(x, train_adj, train_edges, neg_edges)

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                neg_test = sample_negative_edges(adj, test_edges.size(1))
                pos_s, neg_s = model(x, train_adj, test_edges, neg_test)
                preds = torch.cat([pos_s, neg_s]).sigmoid() > 0.5
                labels = torch.cat([torch.ones(test_edges.size(1)), torch.zeros(neg_test.size(1))])
                acc = (preds == labels).float().mean()
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Test Link Pred Acc: {acc:.3f}")


# ---------------------------------------------------------------------------
# 4. Heterogeneous Graph (Simplified)
# ---------------------------------------------------------------------------

class HeteroGNNLayer(nn.Module):
    """
    Simplified heterogeneous GNN layer.
    
    In heterogeneous graphs, different edge types have different
    transformation matrices. The final node embedding is the
    mean of messages from all edge types.
    
    Example: a citation graph with "cites" and "co-authored" edges.
    """

    def __init__(self, in_features: int, out_features: int, edge_types: List[str]):
        super().__init__()
        self.transforms = nn.ModuleDict({
            etype: nn.Linear(in_features, out_features) for etype in edge_types
        })

    def forward(self, x: torch.Tensor, adj_dict: dict) -> torch.Tensor:
        """
        Args:
            x: (N, in_features)
            adj_dict: {edge_type: (N, N) adjacency matrix}
        """
        messages = []
        for etype, adj in adj_dict.items():
            adj_norm = adj / adj.sum(dim=1, keepdim=True).clamp(min=1)
            msg = adj_norm @ self.transforms[etype](x)
            messages.append(msg)
        # Aggregate across edge types
        return torch.stack(messages).mean(dim=0)


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Graph Classification with GIN ===")
    # Create a few small synthetic graphs
    graphs = []
    labels = []
    for i in range(50):
        n = np.random.randint(10, 30)
        feat = np.random.randn(n, 16).astype(np.float32)
        adj = (np.random.rand(n, n) < 0.2).astype(np.float32)
        adj = np.maximum(adj, adj.T)  # symmetric
        np.fill_diagonal(adj, 0)
        graphs.append((torch.tensor(feat), torch.tensor(adj)))
        labels.append(i % 3)

    model = GraphClassifier(16, 32, num_classes=3, n_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        total_loss = 0
        for (x, adj), label in zip(graphs, labels):
            logits = model(x, adj)
            loss = F.cross_entropy(logits, torch.tensor([label]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(graphs):.4f}")

    print("\n=== Link Prediction ===")
    train_link_prediction(epochs=200)
