"""
Recommendation Systems — Classical & Neural Methods
=====================================================
Covers: Collaborative filtering, matrix factorization, Neural Collaborative
Filtering (NCF), and evaluation metrics.
All self-contained with synthetic data.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Synthetic Dataset
# ---------------------------------------------------------------------------

def generate_movielens_like(n_users: int = 500, n_items: int = 200,
                            n_interactions: int = 10_000,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic user-item interaction dataset.
    
    Returns:
        interactions: (n_interactions, 3) — [user_id, item_id, rating]
        user_item_matrix: (n_users, n_items) — sparse rating matrix
    """
    rng = np.random.RandomState(seed)

    # Latent factors (ground truth)
    user_factors = rng.randn(n_users, 5).astype(np.float32)
    item_factors = rng.randn(n_items, 5).astype(np.float32)

    interactions = []
    seen = set()
    while len(interactions) < n_interactions:
        u = rng.randint(n_users)
        i = rng.randint(n_items)
        if (u, i) in seen:
            continue
        seen.add((u, i))
        # Rating = dot product + noise, clipped to [1, 5]
        rating = np.clip(user_factors[u] @ item_factors[i] + rng.randn() * 0.5, 1, 5)
        interactions.append([u, i, round(float(rating), 1)])

    interactions = np.array(interactions, dtype=np.float32)

    # Build sparse matrix
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in interactions:
        matrix[int(u), int(i)] = r

    return interactions, matrix


# ---------------------------------------------------------------------------
# 2. User-Based Collaborative Filtering
# ---------------------------------------------------------------------------

class UserBasedCF:
    """
    Memory-based collaborative filtering using user-user cosine similarity.
    
    Prediction for user u on item i:
        r̂(u,i) = r̄_u + Σ_{v∈N(u)} sim(u,v) * (r(v,i) - r̄_v) / Σ|sim|
    """

    def __init__(self, k_neighbors: int = 20):
        self.k = k_neighbors
        self.matrix = None
        self.user_means = None
        self.sim_matrix = None

    def fit(self, user_item_matrix: np.ndarray):
        self.matrix = user_item_matrix.copy()
        self.user_means = np.zeros(self.matrix.shape[0])
        for u in range(self.matrix.shape[0]):
            rated = self.matrix[u] > 0
            if rated.any():
                self.user_means[u] = self.matrix[u, rated].mean()

        # Cosine similarity between users
        # Center ratings first
        centered = self.matrix.copy()
        for u in range(centered.shape[0]):
            rated = centered[u] > 0
            centered[u, rated] -= self.user_means[u]
            centered[u, ~rated] = 0

        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = centered / norms
        self.sim_matrix = normalized @ normalized.T

    def predict(self, user: int, item: int) -> float:
        if self.matrix[user, item] > 0:
            return self.matrix[user, item]

        sims = self.sim_matrix[user].copy()
        sims[user] = -np.inf  # exclude self

        # Find k most similar users who rated this item
        rated_mask = self.matrix[:, item] > 0
        sims[~rated_mask] = -np.inf
        top_k = np.argsort(sims)[-self.k:]
        top_k = top_k[sims[top_k] > -np.inf]

        if len(top_k) == 0:
            return self.user_means[user]

        weights = self.sim_matrix[user, top_k]
        diffs = self.matrix[top_k, item] - self.user_means[top_k]
        denom = np.abs(weights).sum()
        if denom == 0:
            return self.user_means[user]

        return self.user_means[user] + (weights * diffs).sum() / denom


# ---------------------------------------------------------------------------
# 3. Matrix Factorization (SVD-style)
# ---------------------------------------------------------------------------

class MatrixFactorization(nn.Module):
    """
    Learns user and item embeddings such that:
        r̂(u,i) = μ + b_u + b_i + p_u · q_i
    
    Trained with MSE loss on observed ratings.
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize small
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        dot = (u * i).sum(dim=-1)
        return self.global_bias + self.user_bias(user_ids).squeeze() + \
               self.item_bias(item_ids).squeeze() + dot


# ---------------------------------------------------------------------------
# 4. Neural Collaborative Filtering (NCF) — He et al. 2017
# ---------------------------------------------------------------------------

class NeuralCF(nn.Module):
    """
    Combines Generalized Matrix Factorization (GMF) and a Multi-Layer
    Perceptron (MLP) for learning user-item interactions.
    
    GMF path:  element-wise product of user/item embeddings
    MLP path:  concatenation → hidden layers
    Final:     concatenate both paths → output layer
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32,
                 mlp_layers: List[int] = [64, 32, 16]):
        super().__init__()
        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)

        # MLP embeddings (separate from GMF)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # MLP layers
        layers = []
        input_dim = emb_dim * 2
        for hidden in mlp_layers:
            layers.extend([nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = hidden
        self.mlp = nn.Sequential(*layers)

        # Final prediction
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)

        # MLP path
        mlp_input = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Combine
        combined = torch.cat([gmf, mlp_out], dim=-1)
        return self.output(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# 5. Dataset & DataLoader
# ---------------------------------------------------------------------------

class RatingDataset(Dataset):
    def __init__(self, interactions: np.ndarray):
        self.users = torch.LongTensor(interactions[:, 0].astype(int))
        self.items = torch.LongTensor(interactions[:, 1].astype(int))
        self.ratings = torch.FloatTensor(interactions[:, 2])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# ---------------------------------------------------------------------------
# 6. Evaluation Metrics
# ---------------------------------------------------------------------------

def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Precision@K: fraction of top-K recommendations that are relevant."""
    return len(set(recommended[:k]) & relevant) / k


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Hit Rate@K: 1 if any relevant item is in top-K, else 0."""
    return float(len(set(recommended[:k]) & relevant) > 0)


# ---------------------------------------------------------------------------
# 7. Training & Comparison
# ---------------------------------------------------------------------------

def train_and_compare():
    """Train MF and NCF, compare with user-based CF."""
    interactions, matrix = generate_movielens_like()
    n_users, n_items = matrix.shape

    # Train/test split (80/20 by interaction)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(interactions))
    split = int(0.8 * len(interactions))
    train_data = interactions[perm[:split]]
    test_data = interactions[perm[split:]]

    print(f"Dataset: {n_users} users, {n_items} items, {len(interactions)} ratings")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # --- User-Based CF ---
    print("\n--- User-Based Collaborative Filtering ---")
    train_matrix = np.zeros_like(matrix)
    for u, i, r in train_data:
        train_matrix[int(u), int(i)] = r

    cf = UserBasedCF(k_neighbors=20)
    cf.fit(train_matrix)
    cf_preds = [cf.predict(int(u), int(i)) for u, i, _ in test_data]
    print(f"RMSE: {rmse(np.array(cf_preds), test_data[:, 2]):.4f}")

    # --- Matrix Factorization ---
    print("\n--- Matrix Factorization ---")
    mf = MatrixFactorization(n_users, n_items, n_factors=32)
    _train_model(mf, train_data, test_data, epochs=20, lr=0.005)

    # --- Neural Collaborative Filtering ---
    print("\n--- Neural Collaborative Filtering ---")
    ncf = NeuralCF(n_users, n_items, emb_dim=32, mlp_layers=[64, 32, 16])
    _train_model(ncf, train_data, test_data, epochs=20, lr=0.001)

    # --- Ranking Metrics Demo ---
    print("\n--- Ranking Metrics (NCF, sample user) ---")
    _evaluate_ranking(ncf, train_matrix, test_data, n_items, k=10)


def _train_model(model, train_data, test_data, epochs=20, lr=0.001, batch_size=256):
    dataset = RatingDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items, ratings in loader:
            preds = model(users, items)
            loss = F.mse_loss(preds, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ratings)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_users = torch.LongTensor(test_data[:, 0].astype(int))
                test_items = torch.LongTensor(test_data[:, 1].astype(int))
                test_preds = model(test_users, test_items).numpy()
            test_rmse = rmse(test_preds, test_data[:, 2])
            print(f"  Epoch {epoch+1:2d} | Train Loss: {total_loss/len(train_data):.4f} | "
                  f"Test RMSE: {test_rmse:.4f}")


def _evaluate_ranking(model, train_matrix, test_data, n_items, k=10):
    """Evaluate ranking metrics for a sample of users."""
    model.eval()
    # Build test relevance sets
    test_relevant = defaultdict(set)
    for u, i, r in test_data:
        if r >= 3.5:  # consider "relevant" if rating >= 3.5
            test_relevant[int(u)].add(int(i))

    precisions, ndcgs, hits = [], [], []
    sample_users = list(test_relevant.keys())[:50]

    for user in sample_users:
        # Items not seen in training
        seen = set(np.nonzero(train_matrix[user])[0])
        candidates = [i for i in range(n_items) if i not in seen]
        if not candidates:
            continue

        with torch.no_grad():
            user_t = torch.LongTensor([user] * len(candidates))
            item_t = torch.LongTensor(candidates)
            scores = model(user_t, item_t).numpy()

        # Top-K recommendations
        top_k_idx = np.argsort(scores)[-k:][::-1]
        recommended = [candidates[i] for i in top_k_idx]
        relevant = test_relevant[user]

        precisions.append(precision_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        hits.append(hit_rate_at_k(recommended, relevant, k))

    print(f"  Precision@{k}: {np.mean(precisions):.4f}")
    print(f"  NDCG@{k}:      {np.mean(ndcgs):.4f}")
    print(f"  Hit Rate@{k}:  {np.mean(hits):.4f}")


if __name__ == "__main__":
    train_and_compare()
