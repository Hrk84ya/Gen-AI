# Diffusion Models

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the mathematical foundations of diffusion models
- Implement denoising diffusion probabilistic models (DDPMs)
- Master the forward and reverse diffusion processes
- Build and train diffusion models for image generation
- Explore advanced techniques like classifier-free guidance
- Apply diffusion models to various domains (images, audio, 3D)

## 🌊 What are Diffusion Models?

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image generation, surpassing GANs in quality and diversity.

### Key Concepts:
- **Forward Process**: Gradually add noise to data until it becomes pure noise
- **Reverse Process**: Learn to denoise and recover original data
- **Score Matching**: Learn the gradient of the data distribution
- **Sampling**: Generate new data by starting from noise and denoising

### Applications:
- **Image Generation**: DALL-E 2, Midjourney, Stable Diffusion
- **Image Editing**: Inpainting, outpainting, style transfer
- **Audio Synthesis**: WaveGrad, DiffWave
- **3D Generation**: DreamFusion, Point-E
- **Video Generation**: Video diffusion models

## 📐 Mathematical Foundation

### Forward Diffusion Process
The forward process gradually adds Gaussian noise to data:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Where β_t is the noise schedule. The full forward process:
```
q(x_{1:T} | x_0) = ∏_{t=1}^T q(x_t | x_{t-1})
```

### Reverse Process
The reverse process learns to denoise:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Training Objective
The variational lower bound leads to:
```
L = E_t,x_0,ε [||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]
```

Where ε_θ is the learned denoising network.

## 📚 Module Contents

### 1. [Diffusion Fundamentals](./01_diffusion_fundamentals.ipynb)
- Mathematical derivation of diffusion processes
- Forward and reverse process implementation
- Noise schedules and their effects
- Basic DDPM implementation

### 2. [Denoising Networks](./02_denoising_networks.ipynb)
- U-Net architecture for diffusion
- Time embedding and conditioning
- Attention mechanisms in diffusion models
- Architecture design principles

### 3. [Training Diffusion Models](./03_training_diffusion.ipynb)
- Loss function implementation
- Training strategies and optimization
- Noise schedule design
- Evaluation metrics

### 4. [Sampling and Generation](./04_sampling_generation.ipynb)
- DDPM sampling algorithm
- DDIM deterministic sampling
- Accelerated sampling methods
- Quality vs speed trade-offs

### 5. [Conditional Generation](./05_conditional_generation.ipynb)
- Class-conditional diffusion
- Text-to-image generation
- Classifier-free guidance
- Inpainting and editing

### 6. [Advanced Techniques](./06_advanced_techniques.ipynb)
- Latent diffusion models
- Score-based generative models
- Continuous-time diffusion
- Diffusion for other modalities

### 7. [Applications and Deployment](./07_applications_deployment.ipynb)
- Stable Diffusion implementation
- Real-world applications
- Optimization for inference
- Ethical considerations

## 🏗️ Core Implementation

### 1. **Basic DDPM Implementation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DDPM:
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, timesteps=1000, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        self.betas = self.linear_beta_schedule(beta_start, beta_end, timesteps)
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Linear beta schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def extract(self, a, t, x_shape):
        """Extract values from a 1-D tensor for a batch of indices"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def p_losses(self, x_start, t, noise=None, loss_type="l2"):
        """Compute training losses"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion (add noise)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # Compute loss
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss
    
    def p_mean_variance(self, x, t, clip_denoised=True):
        """Compute mean and variance of p(x_{t-1} | x_t)"""
        # Predict noise
        model_output = self.model(x, t)
        
        # Predict x_0
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        
        pred_x_start = (
            x - sqrt_one_minus_alphas_cumprod_t * model_output
        ) / sqrt_alphas_cumprod_t
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1., 1.)
        
        # Compute posterior mean
        posterior_mean_coef1_t = self.extract(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2_t = self.extract(self.posterior_mean_coef2, t, x.shape)
        
        posterior_mean = (
            posterior_mean_coef1_t * pred_x_start + posterior_mean_coef2_t * x
        )
        
        # Compute posterior variance
        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped_t = self.extract(
            self.posterior_log_variance_clipped, t, x.shape
        )
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t, pred_x_start
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        """Sample from p(x_{t-1} | x_t)"""
        mean, _, log_variance, _ = self.p_mean_variance(x, t, clip_denoised=clip_denoised)
        
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x.shape) - 1)))
        
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
    
    @torch.no_grad()
    def sample(self, shape, return_all_timesteps=False):
        """Generate samples using DDPM"""
        device = next(self.model.parameters()).device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            
            if return_all_timesteps:
                imgs.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        else:
            return img
```

### 2. **U-Net Architecture for Diffusion**
```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        
        # Attention
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(C // self.num_heads)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        h = h.reshape(B, C, H, W)
        
        return x + self.proj(h)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], 
                 time_emb_dim=256, num_classes=None):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Class embedding (optional)
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        for i in range(len(features)):
            in_ch = features[i-1] if i > 0 else features[0]
            out_ch = features[i]
            
            self.encoder.append(nn.ModuleList([
                ResnetBlock(in_ch, out_ch, time_emb_dim),
                ResnetBlock(out_ch, out_ch, time_emb_dim),
                AttentionBlock(out_ch) if out_ch >= 256 else nn.Identity()
            ]))
            
            if i < len(features) - 1:
                self.pool.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResnetBlock(features[-1], features[-1], time_emb_dim),
            AttentionBlock(features[-1]),
            ResnetBlock(features[-1], features[-1], time_emb_dim)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        for i in reversed(range(len(features))):
            in_ch = features[i] * 2  # Skip connection
            out_ch = features[i-1] if i > 0 else features[0]
            
            if i < len(features) - 1:
                self.upconv.append(nn.ConvTranspose2d(features[i+1], features[i], 2, stride=2))
            
            self.decoder.append(nn.ModuleList([
                ResnetBlock(in_ch, out_ch, time_emb_dim),
                ResnetBlock(out_ch, out_ch, time_emb_dim),
                AttentionBlock(out_ch) if out_ch >= 256 else nn.Identity()
            ]))
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU(),
            nn.Conv2d(features[0], out_channels, 3, padding=1)
        )
    
    def forward(self, x, timestep, y=None):
        # Time embedding
        t_emb = self.time_embedding(timestep)
        t_emb = self.time_mlp(t_emb)
        
        # Class embedding (if provided)
        if y is not None and self.num_classes is not None:
            t_emb = t_emb + self.class_emb(y)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        skip_connections = []
        for i, (resnet1, resnet2, attn) in enumerate(self.encoder):
            x = resnet1(x, t_emb)
            x = resnet2(x, t_emb)
            x = attn(x)
            skip_connections.append(x)
            
            if i < len(self.encoder) - 1:
                x = self.pool[i](x)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResnetBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for i, (resnet1, resnet2, attn) in enumerate(self.decoder):
            if i > 0:
                x = self.upconv[i-1](x)
            
            # Skip connection
            x = torch.cat([x, skip_connections[i]], dim=1)
            
            x = resnet1(x, t_emb)
            x = resnet2(x, t_emb)
            x = attn(x)
        
        # Final convolution
        return self.final_conv(x)
```

### 3. **DDIM Sampling (Faster Inference)**
```python
class DDIM:
    def __init__(self, ddpm_model, eta=0.0):
        self.ddpm = ddpm_model
        self.eta = eta  # 0 = deterministic, 1 = DDPM
    
    @torch.no_grad()
    def sample(self, shape, timesteps=50, return_all_timesteps=False):
        """DDIM sampling with fewer timesteps"""
        device = next(self.ddpm.model.parameters()).device
        
        # Create subset of timesteps
        step_size = self.ddpm.timesteps // timesteps
        timesteps_subset = torch.arange(0, self.ddpm.timesteps, step_size).long()
        timesteps_subset = torch.cat([timesteps_subset, torch.tensor([self.ddpm.timesteps - 1])])
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(len(timesteps_subset))):
            t = timesteps_subset[i]
            t_prev = timesteps_subset[i-1] if i > 0 else 0
            
            img = self.ddim_step(img, t, t_prev)
            
            if return_all_timesteps:
                imgs.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        else:
            return img
    
    def ddim_step(self, x_t, t, t_prev):
        """Single DDIM step"""
        # Get model prediction
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        pred_noise = self.ddpm.model(x_t, t_tensor)
        
        # Get alpha values
        alpha_t = self.ddpm.alphas_cumprod[t]
        alpha_t_prev = self.ddpm.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        # Clip x_0
        pred_x0 = torch.clamp(pred_x0, -1., 1.)
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev - self.eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * pred_noise
        
        # Compute x_{t-1}
        noise = torch.randn_like(x_t) if self.eta > 0 else 0
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + self.eta * torch.sqrt((1 - alpha_t_prev) - dir_xt**2) * noise
        
        return x_prev
```

### 4. **Classifier-Free Guidance**
```python
class ClassifierFreeGuidance:
    def __init__(self, model, guidance_scale=7.5, unconditional_prob=0.1):
        self.model = model
        self.guidance_scale = guidance_scale
        self.unconditional_prob = unconditional_prob
    
    def forward(self, x, t, y=None):
        """Forward pass with classifier-free guidance"""
        if y is None or self.training:
            # During training, randomly drop conditioning
            if self.training and torch.rand(1) < self.unconditional_prob:
                y = None
            return self.model(x, t, y)
        
        # During inference, compute both conditional and unconditional predictions
        # Unconditional prediction
        uncond_pred = self.model(x, t, y=None)
        
        # Conditional prediction
        cond_pred = self.model(x, t, y=y)
        
        # Apply guidance
        guided_pred = uncond_pred + self.guidance_scale * (cond_pred - uncond_pred)
        
        return guided_pred

class ConditionalUNet(UNet):
    def __init__(self, *args, text_emb_dim=512, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Text conditioning
        self.text_emb_dim = text_emb_dim
        self.text_proj = nn.Linear(text_emb_dim, self.time_emb_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(feat, text_emb_dim) 
            for feat in [64, 128, 256, 512]
        ])
    
    def forward(self, x, timestep, text_emb=None, y=None):
        # Time embedding
        t_emb = self.time_embedding(timestep)
        t_emb = self.time_mlp(t_emb)
        
        # Text conditioning
        if text_emb is not None:
            text_emb_proj = self.text_proj(text_emb.mean(dim=1))  # Pool text embeddings
            t_emb = t_emb + text_emb_proj
        
        # Class embedding
        if y is not None and self.num_classes is not None:
            t_emb = t_emb + self.class_emb(y)
        
        # Rest of forward pass with cross-attention
        # ... (similar to UNet but with cross-attention to text)
        
        return self.final_conv(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x, context):
        B, C, H, W = x.shape
        
        # Normalize input
        h = self.norm(x)
        
        # Compute queries from image features
        q = self.to_q(h).reshape(B, self.num_heads, self.head_dim, H * W)
        
        # Compute keys and values from text context
        k = self.to_k(context).reshape(B, self.num_heads, self.head_dim, -1)
        v = self.to_v(context).reshape(B, self.num_heads, self.head_dim, -1)
        
        # Attention
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.to_out(out)
```

## 🎯 Training Strategies

### 1. **Training Loop**
```python
def train_diffusion_model(model, dataloader, optimizer, ddpm, num_epochs, device):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device) if labels is not None else None
            
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, ddpm.timesteps, (data.shape[0],), device=device).long()
            
            # Compute loss
            loss = ddpm.p_losses(data, t)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
        
        # Generate samples for monitoring
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                samples = ddpm.sample((16, 3, 32, 32))
                save_samples(samples, f'samples_epoch_{epoch}.png')
            model.train()

def save_samples(samples, filename):
    """Save generated samples as image grid"""
    import torchvision.utils as vutils
    
    # Denormalize samples from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = vutils.make_grid(samples, nrow=4, padding=2)
    
    # Save
    vutils.save_image(grid, filename)
```

### 2. **Advanced Training Techniques**
```python
class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class DiffusionTrainer:
    def __init__(self, model, ddpm, optimizer, device, use_ema=True):
        self.model = model
        self.ddpm = ddpm
        self.optimizer = optimizer
        self.device = device
        
        # EMA model
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(model)
    
    def train_step(self, batch):
        self.model.train()
        
        data, labels = batch
        data = data.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        
        # Sample timesteps
        t = torch.randint(0, self.ddpm.timesteps, (data.shape[0],), device=self.device).long()
        
        # Forward pass
        loss = self.ddpm.p_losses(data, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update EMA
        if self.use_ema:
            self.ema_model.update(self.model)
        
        return loss.item()
    
    def sample(self, shape, use_ema=True):
        """Generate samples using EMA model if available"""
        if use_ema and self.use_ema:
            self.ema_model.apply_shadow(self.model)
        
        self.model.eval()
        with torch.no_grad():
            samples = self.ddpm.sample(shape)
        
        if use_ema and self.use_ema:
            self.ema_model.restore(self.model)
        
        return samples
```

## 📊 Evaluation Metrics

### 1. **Image Quality Metrics**
```python
def compute_fid(real_images, generated_images, device='cuda'):
    """Compute Fréchet Inception Distance"""
    from torchvision.models import inception_v3
    import scipy.linalg
    
    # Load Inception model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    
    def get_activations(images):
        activations = []
        with torch.no_grad():
            for i in range(0, len(images), 32):  # Process in batches
                batch = images[i:i+32].to(device)
                # Resize to 299x299 for Inception
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                
                pred = inception(batch)[0]
                
                # Global average pooling
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                
                activations.append(pred.cpu().numpy().reshape(pred.size(0), -1))
        
        return np.concatenate(activations, axis=0)
    
    # Get activations
    real_activations = get_activations(real_images)
    fake_activations = get_activations(generated_images)
    
    # Compute statistics
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    
    # Compute FID
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_is(images, splits=10):
    """Compute Inception Score"""
    from torchvision.models import inception_v3
    
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    
    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    # Get predictions
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            pred = get_pred(batch)
            preds.append(pred)
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def compute_lpips(real_images, generated_images):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity)"""
    import lpips
    
    loss_fn = lpips.LPIPS(net='alex')
    
    distances = []
    for real, fake in zip(real_images, generated_images):
        # Normalize to [-1, 1]
        real = 2 * real - 1
        fake = 2 * fake - 1
        
        distance = loss_fn(real.unsqueeze(0), fake.unsqueeze(0))
        distances.append(distance.item())
    
    return np.mean(distances)
```

### 2. **Sampling Quality Analysis**
```python
def analyze_sampling_quality(ddpm, num_samples=1000, timesteps_list=[50, 100, 250, 1000]):
    """Analyze quality vs sampling speed trade-off"""
    results = {}
    
    for timesteps in timesteps_list:
        print(f"Sampling with {timesteps} timesteps...")
        
        # Create DDIM sampler
        ddim = DDIM(ddpm, eta=0.0)
        
        # Generate samples
        samples = []
        for i in range(0, num_samples, 16):
            batch_size = min(16, num_samples - i)
            batch_samples = ddim.sample((batch_size, 3, 32, 32), timesteps=timesteps)
            samples.append(batch_samples)
        
        samples = torch.cat(samples, dim=0)
        
        # Compute metrics (placeholder - would need real implementation)
        fid_score = compute_fid_placeholder(samples)
        is_score = compute_is_placeholder(samples)
        
        results[timesteps] = {
            'fid': fid_score,
            'is': is_score,
            'samples': samples
        }
    
    return results

def plot_sampling_analysis(results):
    """Plot sampling quality vs speed"""
    import matplotlib.pyplot as plt
    
    timesteps = list(results.keys())
    fid_scores = [results[t]['fid'] for t in timesteps]
    is_scores = [results[t]['is'] for t in timesteps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FID vs timesteps
    ax1.plot(timesteps, fid_scores, 'o-')
    ax1.set_xlabel('Number of Timesteps')
    ax1.set_ylabel('FID Score (lower is better)')
    ax1.set_title('FID vs Sampling Steps')
    ax1.grid(True)
    
    # IS vs timesteps
    ax2.plot(timesteps, is_scores, 'o-', color='orange')
    ax2.set_xlabel('Number of Timesteps')
    ax2.set_ylabel('IS Score (higher is better)')
    ax2.set_title('Inception Score vs Sampling Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## 🎨 Advanced Applications

### 1. **Image Inpainting**
```python
class DiffusionInpainting:
    def __init__(self, ddpm):
        self.ddpm = ddpm
    
    def inpaint(self, image, mask, num_steps=1000):
        """Inpaint masked regions using diffusion"""
        device = image.device
        
        # Start from noise in masked region, keep original in unmasked region
        x = torch.randn_like(image)
        x = image * (1 - mask) + x * mask
        
        for i in reversed(range(num_steps)):
            t = torch.full((image.shape[0],), i, device=device, dtype=torch.long)
            
            # Denoise step
            x = self.ddpm.p_sample(x, t)
            
            # Replace unmasked regions with noisy version of original
            if i > 0:
                noise = torch.randn_like(image)
                x_t_original = self.ddpm.q_sample(image, t, noise)
                x = x_t_original * (1 - mask) + x * mask
        
        return x

def create_mask(image_size, mask_type='center', mask_ratio=0.5):
    """Create different types of masks for inpainting"""
    H, W = image_size
    mask = torch.zeros(1, 1, H, W)
    
    if mask_type == 'center':
        # Center square mask
        h_start = int(H * (1 - mask_ratio) / 2)
        h_end = int(H * (1 + mask_ratio) / 2)
        w_start = int(W * (1 - mask_ratio) / 2)
        w_end = int(W * (1 + mask_ratio) / 2)
        mask[:, :, h_start:h_end, w_start:w_end] = 1
        
    elif mask_type == 'random':
        # Random mask
        mask = torch.rand(1, 1, H, W) < mask_ratio
        mask = mask.float()
        
    elif mask_type == 'half':
        # Half image mask
        mask[:, :, :, W//2:] = 1
    
    return mask
```

### 2. **Style Transfer with Diffusion**
```python
class DiffusionStyleTransfer:
    def __init__(self, ddpm, style_encoder):
        self.ddpm = ddpm
        self.style_encoder = style_encoder
    
    def transfer_style(self, content_image, style_image, strength=0.7):
        """Transfer style using diffusion process"""
        
        # Encode style
        style_features = self.style_encoder(style_image)
        
        # Add noise to content image
        t_start = int(self.ddpm.timesteps * strength)
        t = torch.full((content_image.shape[0],), t_start, device=content_image.device, dtype=torch.long)
        
        noise = torch.randn_like(content_image)
        x_t = self.ddpm.q_sample(content_image, t, noise)
        
        # Denoise with style conditioning
        for i in reversed(range(t_start)):
            t_step = torch.full((content_image.shape[0],), i, device=content_image.device, dtype=torch.long)
            
            # Modified denoising with style conditioning
            x_t = self.style_conditioned_step(x_t, t_step, style_features)
        
        return x_t
    
    def style_conditioned_step(self, x, t, style_features):
        """Denoising step conditioned on style features"""
        # This would require a modified U-Net that accepts style conditioning
        # For now, this is a placeholder
        return self.ddpm.p_sample(x, t)
```

## 🎓 Learning Path

### Week 1: Fundamentals
- Understand diffusion process mathematics
- Implement basic DDPM from scratch
- Train on simple datasets (MNIST, CIFAR-10)

### Week 2: Architecture and Training
- Master U-Net architecture for diffusion
- Learn advanced training techniques
- Implement different noise schedules

### Week 3: Sampling and Acceleration
- Implement DDIM for faster sampling
- Learn about other acceleration methods
- Optimize inference speed

### Week 4: Advanced Applications
- Implement conditional generation
- Build text-to-image models
- Explore inpainting and editing

## 📚 Additional Resources

### Foundational Papers
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
- "Denoising Diffusion Implicit Models" (Song et al., 2020)
- "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021)

### Advanced Papers
- "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
- "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2)
- "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen)

### Practical Resources
- [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Denoising Diffusion PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

---

**Next Module**: [Advanced Applications](../../04-Advanced/) →

*Ready to master the cutting-edge of generative AI? Let's build diffusion models that create stunning images! 🌊✨*