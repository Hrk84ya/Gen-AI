"""
Multimodal Fusion Techniques Implementation

This module demonstrates various fusion strategies for combining
multiple modalities in deep learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class EarlyFusion(nn.Module):
    """Early fusion: concatenate features at input level"""
    
    def __init__(self, modality_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.total_input_dim = sum(modality_dims)
        self.fusion_net = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        # Concatenate all modalities
        fused_features = torch.cat(modalities, dim=-1)
        return self.fusion_net(fused_features)


class LateFusion(nn.Module):
    """Late fusion: process modalities separately then combine predictions"""
    
    def __init__(self, modality_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.modality_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim)
            ) for dim in modality_dims
        ])
        self.fusion_weights = nn.Parameter(torch.ones(len(modality_dims)))
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        # Process each modality separately
        predictions = []
        for i, (modality, net) in enumerate(zip(modalities, self.modality_nets)):
            pred = net(modality)
            predictions.append(pred)
        
        # Weighted combination of predictions
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_pred = sum(w * pred for w, pred in zip(weights, predictions))
        return fused_pred


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism"""
    
    def __init__(self, modality_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.modality_encoders = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in modality_dims
        ])
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        # Encode each modality
        encoded_modalities = []
        for modality, encoder in zip(modalities, self.modality_encoders):
            encoded = encoder(modality)
            encoded_modalities.append(encoded.unsqueeze(1))  # Add sequence dimension
        
        # Stack modalities as sequence
        modality_sequence = torch.cat(encoded_modalities, dim=1)
        
        # Apply self-attention across modalities
        attended_features, _ = self.attention(
            modality_sequence, modality_sequence, modality_sequence
        )
        
        # Global average pooling across modalities
        fused_features = attended_features.mean(dim=1)
        
        return self.output_net(fused_features)


class BilinearFusion(nn.Module):
    """Bilinear fusion for two modalities"""
    
    def __init__(self, dim1: int, dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, hidden_dim)
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, modality1: torch.Tensor, modality2: torch.Tensor) -> torch.Tensor:
        fused = self.bilinear(modality1, modality2)
        return self.output_net(fused)


class TensorFusion(nn.Module):
    """Tensor fusion network for multiple modalities"""
    
    def __init__(self, modality_dims: List[int], output_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        
        # Add bias dimension to each modality
        self.extended_dims = [dim + 1 for dim in modality_dims]
        
        # Calculate tensor product dimension
        tensor_dim = np.prod(self.extended_dims)
        
        self.output_net = nn.Sequential(
            nn.Linear(tensor_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].shape[0]
        
        # Add bias term to each modality
        extended_modalities = []
        for modality in modalities:
            bias = torch.ones(batch_size, 1, device=modality.device)
            extended = torch.cat([modality, bias], dim=1)
            extended_modalities.append(extended)
        
        # Compute tensor product
        tensor_product = extended_modalities[0]
        for modality in extended_modalities[1:]:
            tensor_product = torch.bmm(
                tensor_product.unsqueeze(2),
                modality.unsqueeze(1)
            ).view(batch_size, -1)
        
        return self.output_net(tensor_product)


class CrossModalTransformer(nn.Module):
    """Cross-modal transformer for multimodal fusion"""
    
    def __init__(self, modality_dims: List[int], d_model: int, nhead: int, 
                 num_layers: int, output_dim: int):
        super().__init__()
        
        # Project each modality to common dimension
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in modality_dims
        ])
        
        # Positional embeddings for modalities
        self.modality_embeddings = nn.Parameter(
            torch.randn(len(modality_dims), d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].shape[0]
        
        # Project modalities to common space
        projected_modalities = []
        for i, (modality, projection) in enumerate(zip(modalities, self.modality_projections)):
            projected = projection(modality)
            # Add modality-specific positional embedding
            projected = projected + self.modality_embeddings[i].unsqueeze(0)
            projected_modalities.append(projected.unsqueeze(1))
        
        # Stack modalities as sequence
        modality_sequence = torch.cat(projected_modalities, dim=1)
        
        # Apply transformer
        transformed = self.transformer(modality_sequence)
        
        # Global average pooling
        pooled = transformed.mean(dim=1)
        
        return self.output_projection(pooled)


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns to weight different fusion strategies"""
    
    def __init__(self, modality_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Different fusion strategies
        self.early_fusion = EarlyFusion(modality_dims, hidden_dim, output_dim)
        self.late_fusion = LateFusion(modality_dims, hidden_dim, output_dim)
        self.attention_fusion = AttentionFusion(modality_dims, hidden_dim, output_dim)
        
        # Gating network to choose fusion strategy
        total_input_dim = sum(modality_dims)
        self.gating_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 fusion strategies
            nn.Softmax(dim=-1)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        # Compute outputs from different fusion strategies
        early_out = self.early_fusion(modalities)
        late_out = self.late_fusion(modalities)
        attention_out = self.attention_fusion(modalities)
        
        # Compute gating weights
        concatenated = torch.cat(modalities, dim=-1)
        gates = self.gating_net(concatenated)
        
        # Weighted combination
        fused_output = (gates[:, 0:1] * early_out + 
                       gates[:, 1:2] * late_out + 
                       gates[:, 2:3] * attention_out)
        
        return fused_output


class MultimodalVAE(nn.Module):
    """Multimodal Variational Autoencoder"""
    
    def __init__(self, modality_dims: List[int], latent_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.latent_dim = latent_dim
        
        # Encoders for each modality
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            ) for dim in modality_dims
        ])
        
        # Shared latent space
        total_encoded_dim = len(modality_dims) * 128
        self.mu_net = nn.Linear(total_encoded_dim, latent_dim)
        self.logvar_net = nn.Linear(total_encoded_dim, latent_dim)
        
        # Decoders for each modality
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, dim)
            ) for dim in modality_dims
        ])
    
    def encode(self, modalities: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode each modality
        encoded = []
        for modality, encoder in zip(modalities, self.encoders):
            encoded.append(encoder(modality))
        
        # Concatenate encoded features
        concatenated = torch.cat(encoded, dim=-1)
        
        # Compute mean and log variance
        mu = self.mu_net(concatenated)
        logvar = self.logvar_net(concatenated)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> List[torch.Tensor]:
        # Decode to each modality
        reconstructions = []
        for decoder in self.decoders:
            recon = decoder(z)
            reconstructions.append(recon)
        
        return reconstructions
    
    def forward(self, modalities: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(modalities)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        
        return reconstructions, mu, logvar


def multimodal_vae_loss(reconstructions: List[torch.Tensor], 
                       targets: List[torch.Tensor],
                       mu: torch.Tensor, 
                       logvar: torch.Tensor,
                       beta: float = 1.0) -> torch.Tensor:
    """Compute loss for multimodal VAE"""
    
    # Reconstruction loss for each modality
    recon_loss = 0
    for recon, target in zip(reconstructions, targets):
        recon_loss += F.mse_loss(recon, target, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


# Example usage and testing
if __name__ == "__main__":
    # Example data
    batch_size = 32
    modality_dims = [512, 256, 128]  # Three modalities
    output_dim = 10
    
    # Generate sample data
    modalities = [torch.randn(batch_size, dim) for dim in modality_dims]
    
    # Test different fusion methods
    print("Testing fusion methods...")
    
    # Early fusion
    early_model = EarlyFusion(modality_dims, 256, output_dim)
    early_out = early_model(modalities)
    print(f"Early fusion output shape: {early_out.shape}")
    
    # Late fusion
    late_model = LateFusion(modality_dims, 256, output_dim)
    late_out = late_model(modalities)
    print(f"Late fusion output shape: {late_out.shape}")
    
    # Attention fusion
    attention_model = AttentionFusion(modality_dims, 256, output_dim)
    attention_out = attention_model(modalities)
    print(f"Attention fusion output shape: {attention_out.shape}")
    
    # Tensor fusion
    tensor_model = TensorFusion(modality_dims, output_dim)
    tensor_out = tensor_model(modalities)
    print(f"Tensor fusion output shape: {tensor_out.shape}")
    
    # Cross-modal transformer
    transformer_model = CrossModalTransformer(modality_dims, 256, 8, 2, output_dim)
    transformer_out = transformer_model(modalities)
    print(f"Transformer fusion output shape: {transformer_out.shape}")
    
    # Multimodal VAE
    vae_model = MultimodalVAE(modality_dims, 64)
    reconstructions, mu, logvar = vae_model(modalities)
    print(f"VAE reconstructions: {[r.shape for r in reconstructions]}")
    print(f"VAE latent: mu={mu.shape}, logvar={logvar.shape}")
    
    print("All fusion methods tested successfully!")