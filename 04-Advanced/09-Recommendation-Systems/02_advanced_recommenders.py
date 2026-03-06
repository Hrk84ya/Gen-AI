"""
Advanced Recommendation Systems
=================================
Covers: Two-tower retrieval model, sequence-aware recommendations,
hybrid content+collaborative model, and cold-start strategies.
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Two-Tower Retrieval Model
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """Encodes user features into a dense embedding."""

    def __init__(self, n_users: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.user_emb(user_ids)), dim=-1)


class ItemTower(nn.Module):
    """Encodes item features into a dense embedding."""

    def __init__(self, n_items: int, emb_dim: int = 64, n_item_features: int = 0,
                 hidden: int = 128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        input_dim = emb_dim + n_item_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.has_features = n_item_features > 0

    def forward(self, item_ids: torch.Tensor,
                item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.item_emb(item_ids)
        if self.has_features and item_features is not None:
            x = torch.cat([x, item_features], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-tower (dual encoder) retrieval model.
    
    Used in production at YouTube, Google, Pinterest, etc.
    
    Architecture:
        User tower → user embedding (d-dim)
        Item tower → item embedding (d-dim)
        Score = dot product (or cosine similarity)
    
    At serving time, item embeddings are precomputed and indexed
    for fast approximate nearest neighbor (ANN) retrieval.
    
    Training uses in-batch negatives: for each (user, positive_item) pair,
    all other items in the batch serve as negatives.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 n_item_features: int = 0, temperature: float = 0.1):
        super().__init__()
        self.user_tower = UserTower(n_users, emb_dim)
        self.item_tower = ItemTower(n_items, emb_dim, n_item_features)
        self.temperature = temperature

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                item_features: Optional[torch.Tensor] = None):
        user_emb = self.user_tower(user_ids)       # (B, D)
        item_emb = self.item_tower(item_ids, item_features)  # (B, D)
        return user_emb, item_emb

    def compute_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """In-batch softmax cross-entropy loss."""
        # Similarity matrix: (B, B)
        logits = (user_emb @ item_emb.T) / self.temperature
        # Each row i should match column i (positive pair)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# 2. Sequence-Aware Recommender (Session-Based)
# ---------------------------------------------------------------------------

class SequenceRecommender(nn.Module):
    """
    GRU-based session/sequence recommender.
    
    Given a sequence of item interactions [i_1, i_2, ..., i_t],
    predicts the next item i_{t+1}.
    
    Used for: session-based recommendations, "what to watch next",
    playlist continuation.
    """

    def __init__(self, n_items: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq: (batch, seq_len) — sequence of item IDs
        Returns:
            logits: (batch, n_items) — scores for next item
        """
        x = self.item_emb(item_seq)
        _, hidden = self.gru(x)
        logits = self.output(hidden.squeeze(0))
        return logits


class SequenceDataset(Dataset):
    """Create (sequence, target) pairs from user interaction histories."""

    def __init__(self, user_histories: dict, max_len: int = 20):
        self.samples = []
        for user, items in user_histories.items():
            if len(items) < 2:
                continue
            for i in range(1, len(items)):
                seq = items[max(0, i - max_len):i]
                target = items[i]
                # Pad sequence
                padded = [0] * (max_len - len(seq)) + seq
                self.samples.append((padded, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.LongTensor(seq), torch.tensor(target, dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. Hybrid Content + Collaborative Model
# ---------------------------------------------------------------------------

class HybridRecommender(nn.Module):
    """
    Combines collaborative filtering signals (user/item IDs) with
    content features (item attributes like genre, description embedding).
    
    This addresses the cold-start problem: new items with no interactions
    can still get reasonable recommendations from their content features.
    """

    def __init__(self, n_users: int, n_items: int, n_content_features: int,
                 emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        # Collaborative path
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # Content path
        self.content_net = nn.Sequential(
            nn.Linear(n_content_features, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                content_features: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        c = self.content_net(content_features)

        # Combine: user emb, item emb, content emb
        combined = torch.cat([u, i, c], dim=-1)
        return self.fusion(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. Cold-Start Strategies
# ---------------------------------------------------------------------------

class PopularityFallback:
    """
    Simple popularity-based fallback for cold-start users/items.
    
    When we have no interaction data for a user or item, recommend
    the most popular items (or use content-based similarity).
    """

    def __init__(self):
        self.item_popularity = None

    def fit(self, interactions: np.ndarray, n_items: int):
        counts = np.zeros(n_items)
        for _, item, _ in interactions:
            counts[int(item)] += 1
        self.item_popularity = np.argsort(counts)[::-1]

    def recommend(self, k: int = 10, exclude: set = None) -> List[int]:
        exclude = exclude or set()
        recs = [i for i in self.item_popularity if i not in exclude]
        return recs[:k]


class ContentBasedColdStart:
    """
    For new items: find similar items using content features
    and transfer their collaborative signals.
    """

    def __init__(self, item_features: np.ndarray):
        # Normalize features for cosine similarity
        norms = np.linalg.norm(item_features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.features_norm = item_features / norms

    def find_similar(self, item_id: int, k: int = 5) -> List[int]:
        sims = self.features_norm @ self.features_norm[item_id]
        sims[item_id] = -1  # exclude self
        return list(np.argsort(sims)[-k:][::-1])


# ---------------------------------------------------------------------------
# 5. Training Demos
# ---------------------------------------------------------------------------

def demo_two_tower():
    """Train a two-tower retrieval model."""
    print("=== Two-Tower Retrieval Model ===")
    n_users, n_items = 200, 100
    rng = np.random.RandomState(42)

    # Generate positive pairs
    pairs = []
    for _ in range(5000):
        u = rng.randint(n_users)
        i = rng.randint(n_items)
        pairs.append((u, i))

    model = TwoTowerModel(n_users, n_items, emb_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        rng.shuffle(pairs)
        total_loss = 0
        batch_size = 128
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            users = torch.LongTensor([p[0] for p in batch])
            items = torch.LongTensor([p[1] for p in batch])
            user_emb, item_emb = model(users, items)
            loss = model.compute_loss(user_emb, item_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {total_loss / (len(pairs) / batch_size):.4f}")


def demo_sequence_recommender():
    """Train a sequence-aware recommender."""
    print("\n=== Sequence-Aware Recommender ===")
    n_items = 50
    rng = np.random.RandomState(42)

    # Generate synthetic user histories
    histories = {}
    for u in range(100):
        length = rng.randint(5, 20)
        histories[u] = [rng.randint(1, n_items) for _ in range(length)]

    dataset = SequenceDataset(histories, max_len=10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SequenceRecommender(n_items, emb_dim=32, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for seqs, targets in loader:
            logits = model(seqs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")


def demo_hybrid():
    """Train a hybrid content+collaborative model."""
    print("\n=== Hybrid Recommender ===")
    n_users, n_items, n_features = 200, 100, 16
    rng = np.random.RandomState(42)

    # Synthetic data
    item_content = rng.randn(n_items, n_features).astype(np.float32)
    interactions = []
    for _ in range(5000):
        u = rng.randint(n_users)
        i = rng.randint(n_items)
        r = np.clip(rng.randn() + 3, 1, 5)
        interactions.append((u, i, r))

    model = HybridRecommender(n_users, n_items, n_features)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        rng.shuffle(interactions)
        total_loss = 0
        bs = 128
        for start in range(0, len(interactions), bs):
            batch = interactions[start:start + bs]
            users = torch.LongTensor([b[0] for b in batch])
            items = torch.LongTensor([b[1] for b in batch])
            ratings = torch.FloatTensor([b[2] for b in batch])
            content = torch.FloatTensor(item_content[[b[1] for b in batch]])

            preds = model(users, items, content)
            loss = F.mse_loss(preds, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {total_loss / (len(interactions) / bs):.4f}")


if __name__ == "__main__":
    demo_two_tower()
    demo_sequence_recommender()
    demo_hybrid()
