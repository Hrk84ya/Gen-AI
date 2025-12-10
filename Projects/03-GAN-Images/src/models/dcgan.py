"""
Deep Convolutional GAN (DCGAN) Implementation

This module implements the DCGAN architecture following the guidelines from
"Unsupervised Representation Learning with Deep Convolutional GANs" paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator Network
    
    Generates images from random noise using transposed convolutions.
    Follows DCGAN architectural guidelines:
    - Replace pooling layers with strided convolutions
    - Use batch normalization in both generator and discriminator
    - Remove fully connected hidden layers
    - Use ReLU activation in generator for all layers except output (Tanh)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        num_channels: int = 3,
        image_size: int = 64
    ):
        """
        Initialize DCGAN Generator
        
        Args:
            latent_dim: Dimension of input noise vector
            feature_maps: Number of feature maps in the first layer
            num_channels: Number of output channels (3 for RGB, 1 for grayscale)
            image_size: Size of output images (must be power of 2)
        """
        super(DCGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.num_channels = num_channels
        self.image_size = image_size
        
        # Calculate the number of layers needed
        self.num_layers = int(np.log2(image_size)) - 2  # -2 because we start from 4x4
        
        # Initial layer: latent_dim -> (feature_maps * 8) * 4 * 4
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, feature_maps * 8, 
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True)
        )
        
        # Progressive upsampling layers
        layers = []
        in_channels = feature_maps * 8
        
        for i in range(self.num_layers):
            out_channels = in_channels // 2
            
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            
            if i < self.num_layers - 1:  # No batch norm on the last layer
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(True))
            
            in_channels = out_channels
        
        # Final layer to get the desired number of channels
        layers.append(nn.ConvTranspose2d(
            in_channels, num_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize network weights according to DCGAN paper"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator
        
        Args:
            noise: Random noise tensor of shape (batch_size, latent_dim, 1, 1)
        
        Returns:
            Generated images of shape (batch_size, num_channels, image_size, image_size)
        """
        # Reshape noise if needed
        if noise.dim() == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        
        x = self.initial(noise)
        x = self.main(x)
        
        return x


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator Network
    
    Classifies images as real or fake using convolutional layers.
    Follows DCGAN architectural guidelines:
    - Use LeakyReLU activation in discriminator for all layers
    - Use strided convolutions rather than pooling
    - Use batch normalization (except first layer)
    """
    
    def __init__(
        self,
        num_channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
        use_sigmoid: bool = True
    ):
        """
        Initialize DCGAN Discriminator
        
        Args:
            num_channels: Number of input channels (3 for RGB, 1 for grayscale)
            feature_maps: Number of feature maps in the first layer
            image_size: Size of input images (must be power of 2)
            use_sigmoid: Whether to use sigmoid activation (False for WGAN)
        """
        super(DCGANDiscriminator, self).__init__()
        
        self.num_channels = num_channels
        self.feature_maps = feature_maps
        self.image_size = image_size
        self.use_sigmoid = use_sigmoid
        
        # Calculate the number of layers needed
        self.num_layers = int(np.log2(image_size)) - 2
        
        layers = []
        in_channels = num_channels
        out_channels = feature_maps
        
        # First layer (no batch normalization)
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        in_channels = out_channels
        
        # Progressive downsampling layers
        for i in range(self.num_layers):
            out_channels = in_channels * 2
            
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            
            if i < self.num_layers - 1:  # No batch norm on the last conv layer
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.Conv2d(
            in_channels, 1,
            kernel_size=4, stride=1, padding=0, bias=False
        ))
        
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize network weights according to DCGAN paper"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator
        
        Args:
            images: Input images of shape (batch_size, num_channels, image_size, image_size)
        
        Returns:
            Discriminator output of shape (batch_size, 1, 1, 1)
        """
        return self.main(images)


class ConditionalDCGAN(nn.Module):
    """
    Conditional DCGAN that can generate images conditioned on class labels
    """
    
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 100,
        feature_maps: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        embed_dim: int = 50
    ):
        super(ConditionalDCGAN, self).__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Generator with concatenated noise and label embedding
        self.generator = DCGANGenerator(
            latent_dim=latent_dim + embed_dim,
            feature_maps=feature_maps,
            num_channels=num_channels,
            image_size=image_size
        )
        
        # Discriminator with additional input channel for label
        self.discriminator = DCGANDiscriminator(
            num_channels=num_channels + 1,  # +1 for label channel
            feature_maps=feature_maps,
            image_size=image_size
        )
    
    def generate(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate images conditioned on labels"""
        # Embed labels
        label_embed = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        if noise.dim() == 2:
            combined_input = torch.cat([noise, label_embed], dim=1)
            combined_input = combined_input.view(combined_input.size(0), -1, 1, 1)
        else:
            label_embed = label_embed.view(label_embed.size(0), -1, 1, 1)
            combined_input = torch.cat([noise, label_embed], dim=1)
        
        return self.generator(combined_input)
    
    def discriminate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Discriminate images with label information"""
        batch_size = images.size(0)
        
        # Create label maps
        label_maps = labels.view(batch_size, 1, 1, 1).expand(
            batch_size, 1, images.size(2), images.size(3)
        ).float()
        
        # Concatenate images and label maps
        combined_input = torch.cat([images, label_maps], dim=1)
        
        return self.discriminator(combined_input)


class SpectralNormDCGAN(nn.Module):
    """
    DCGAN with Spectral Normalization for improved training stability
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        num_channels: int = 3,
        image_size: int = 64
    ):
        super(SpectralNormDCGAN, self).__init__()
        
        self.generator = self._build_generator(latent_dim, feature_maps, num_channels, image_size)
        self.discriminator = self._build_discriminator(num_channels, feature_maps, image_size)
    
    def _build_generator(self, latent_dim, feature_maps, num_channels, image_size):
        """Build generator with spectral normalization"""
        num_layers = int(np.log2(image_size)) - 2
        
        # Initial layer
        layers = [
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                latent_dim, feature_maps * 8,
                kernel_size=4, stride=1, padding=0, bias=False
            )),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True)
        ]
        
        # Progressive layers
        in_channels = feature_maps * 8
        for i in range(num_layers):
            out_channels = in_channels // 2
            
            layers.append(nn.utils.spectral_norm(nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )))
            
            if i < num_layers - 1:
                layers.extend([
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                ])
            
            in_channels = out_channels
        
        # Final layer
        layers.extend([
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                in_channels, num_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )),
            nn.Tanh()
        ])
        
        return nn.Sequential(*layers)
    
    def _build_discriminator(self, num_channels, feature_maps, image_size):
        """Build discriminator with spectral normalization"""
        num_layers = int(np.log2(image_size)) - 2
        
        layers = []
        in_channels = num_channels
        out_channels = feature_maps
        
        # First layer
        layers.extend([
            nn.utils.spectral_norm(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        in_channels = out_channels
        
        # Progressive layers
        for i in range(num_layers):
            out_channels = in_channels * 2
            
            layers.append(nn.utils.spectral_norm(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.utils.spectral_norm(nn.Conv2d(
            in_channels, 1,
            kernel_size=4, stride=1, padding=0, bias=False
        )))
        
        return nn.Sequential(*layers)
    
    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        """Generate images from noise"""
        if noise.dim() == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        return self.generator(noise)
    
    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake images"""
        return self.discriminator(images)


class SelfAttentionDCGAN(nn.Module):
    """
    DCGAN with Self-Attention mechanism for better global coherence
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        attention_size: int = 32
    ):
        super(SelfAttentionDCGAN, self).__init__()
        
        self.attention_size = attention_size
        
        # Build generator with self-attention
        self.generator = self._build_generator_with_attention(
            latent_dim, feature_maps, num_channels, image_size
        )
        
        # Build discriminator with self-attention
        self.discriminator = self._build_discriminator_with_attention(
            num_channels, feature_maps, image_size
        )
    
    def _build_generator_with_attention(self, latent_dim, feature_maps, num_channels, image_size):
        """Build generator with self-attention at specified resolution"""
        layers = []
        
        # Initial layer
        layers.extend([
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True)
        ])
        
        current_size = 4
        in_channels = feature_maps * 8
        
        while current_size < image_size:
            out_channels = max(in_channels // 2, feature_maps)
            
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ])
            
            current_size *= 2
            
            # Add self-attention at specified resolution
            if current_size == self.attention_size:
                layers.append(SelfAttention(out_channels))
            
            in_channels = out_channels
        
        # Final layer
        layers.extend([
            nn.ConvTranspose2d(in_channels, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        
        return nn.Sequential(*layers)
    
    def _build_discriminator_with_attention(self, num_channels, feature_maps, image_size):
        """Build discriminator with self-attention"""
        layers = []
        
        # First layer
        layers.extend([
            nn.Conv2d(num_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        current_size = image_size // 2
        in_channels = feature_maps
        
        while current_size > 4:
            out_channels = min(in_channels * 2, feature_maps * 8)
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            # Add self-attention at specified resolution
            if current_size == self.attention_size:
                layers.append(SelfAttention(out_channels))
            
            current_size //= 2
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False))
        
        return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    """Self-Attention module for GANs"""
    
    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to input feature maps
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor with self-attention applied
        """
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Compute attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


# Utility functions for DCGAN
def create_dcgan(
    model_type: str = 'standard',
    latent_dim: int = 100,
    feature_maps: int = 64,
    num_channels: int = 3,
    image_size: int = 64,
    **kwargs
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create different DCGAN variants
    
    Args:
        model_type: Type of DCGAN ('standard', 'conditional', 'spectral', 'self_attention')
        latent_dim: Dimension of noise vector
        feature_maps: Base number of feature maps
        num_channels: Number of image channels
        image_size: Size of generated images
        **kwargs: Additional arguments for specific model types
    
    Returns:
        Tuple of (generator, discriminator)
    """
    if model_type == 'standard':
        generator = DCGANGenerator(latent_dim, feature_maps, num_channels, image_size)
        discriminator = DCGANDiscriminator(num_channels, feature_maps, image_size)
    
    elif model_type == 'conditional':
        num_classes = kwargs.get('num_classes', 10)
        model = ConditionalDCGAN(
            num_classes, latent_dim, feature_maps, num_channels, image_size
        )
        return model.generator, model.discriminator
    
    elif model_type == 'spectral':
        model = SpectralNormDCGAN(latent_dim, feature_maps, num_channels, image_size)
        return model.generator, model.discriminator
    
    elif model_type == 'self_attention':
        attention_size = kwargs.get('attention_size', 32)
        model = SelfAttentionDCGAN(
            latent_dim, feature_maps, num_channels, image_size, attention_size
        )
        return model.generator, model.discriminator
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return generator, discriminator


def test_dcgan_models():
    """Test different DCGAN variants"""
    batch_size = 4
    latent_dim = 100
    image_size = 64
    num_channels = 3
    
    print("Testing DCGAN models...")
    
    # Test standard DCGAN
    print("\n1. Standard DCGAN")
    generator, discriminator = create_dcgan('standard', latent_dim=latent_dim)
    
    noise = torch.randn(batch_size, latent_dim, 1, 1)
    fake_images = generator(noise)
    print(f"Generated images shape: {fake_images.shape}")
    
    real_images = torch.randn(batch_size, num_channels, image_size, image_size)
    d_real = discriminator(real_images)
    d_fake = discriminator(fake_images.detach())
    print(f"Discriminator output shapes: real={d_real.shape}, fake={d_fake.shape}")
    
    # Test conditional DCGAN
    print("\n2. Conditional DCGAN")
    cond_model = ConditionalDCGAN(num_classes=10)
    
    labels = torch.randint(0, 10, (batch_size,))
    fake_images = cond_model.generate(noise[:, :, 0, 0], labels)
    print(f"Conditional generated images shape: {fake_images.shape}")
    
    # Test self-attention DCGAN
    print("\n3. Self-Attention DCGAN")
    sa_model = SelfAttentionDCGAN()
    
    fake_images = sa_model.generator(noise)
    print(f"Self-attention generated images shape: {fake_images.shape}")
    
    print("\n✅ All DCGAN models tested successfully!")


if __name__ == "__main__":
    test_dcgan_models()