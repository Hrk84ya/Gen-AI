# Variational Autoencoders (VAEs)

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the mathematical foundations of VAEs
- Implement VAEs from scratch using modern frameworks
- Master the reparameterization trick and ELBO optimization
- Apply VAEs to image generation and latent space manipulation
- Explore advanced VAE variants and their applications
- Compare VAEs with other generative models (GANs, Flow-based models)

## 🧠 What are Variational Autoencoders?

Variational Autoencoders (VAEs) are probabilistic generative models that learn to encode data into a latent space and decode it back to the original space. Unlike traditional autoencoders, VAEs learn a probability distribution over the latent space, enabling generation of new samples.

### Key Concepts:
- **Probabilistic Encoding**: Maps input to a distribution in latent space
- **Reparameterization Trick**: Enables backpropagation through stochastic nodes
- **ELBO Optimization**: Maximizes Evidence Lower BOund
- **Latent Space Interpolation**: Smooth transitions between data points
- **Disentangled Representations**: Separable latent factors

## 🔬 Mathematical Foundation

### Probabilistic Framework
VAEs model the data generation process as:
```
z ~ p(z)           # Prior distribution (usually N(0,I))
x ~ p(x|z)         # Likelihood (decoder)
```

The goal is to maximize the log-likelihood:
```
log p(x) = log ∫ p(x|z)p(z)dz
```

Since this integral is intractable, VAEs use variational inference with an approximate posterior q(z|x) (encoder).

### Evidence Lower Bound (ELBO)
```
log p(x) ≥ E[log p(x|z)] - KL(q(z|x) || p(z))
         = Reconstruction Loss - KL Divergence
```

## 📚 Module Contents

### 1. [VAE Fundamentals](./01_vae_fundamentals.ipynb)
- Mathematical derivation of ELBO
- Reparameterization trick explained
- Basic VAE implementation
- Training dynamics and loss analysis

### 2. [VAE Architecture Design](./02_vae_architectures.ipynb)
- Encoder and decoder design principles
- Convolutional VAEs for images
- Fully connected VAEs for tabular data
- Architecture best practices

### 3. [Advanced VAE Variants](./03_advanced_vae_variants.ipynb)
- β-VAE for disentangled representations
- WAE (Wasserstein Autoencoders)
- VQ-VAE (Vector Quantized VAE)
- Conditional VAEs (CVAE)

### 4. [Latent Space Analysis](./04_latent_space_analysis.ipynb)
- Latent space visualization
- Interpolation and manipulation
- Disentanglement metrics
- Latent space arithmetic

### 5. [VAE Applications](./05_vae_applications.ipynb)
- Image generation and reconstruction
- Anomaly detection
- Data augmentation
- Semi-supervised learning

### 6. [VAE vs Other Models](./06_vae_comparisons.ipynb)
- VAE vs GAN comparison
- VAE vs Flow-based models
- Hybrid approaches (VAE-GAN)
- When to use each model type

## 🏗️ VAE Architecture Components

### 1. **Basic VAE Implementation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For normalized inputs [0,1]
        )
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate new samples"""
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function"""
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

### 2. **Convolutional VAE for Images**
```python
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(ConvVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(input_channels, 32, 4, 2, 1),  # 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),             # 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),            # 128 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),           # 256 x 4 x 4
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),  # 3 x 64 x 64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 4, 4)  # Reshape for conv layers
        return self.decoder(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
```

### 3. **β-VAE for Disentanglement**
```python
class BetaVAE(ConvVAE):
    def __init__(self, input_channels=3, latent_dim=128, beta=4.0):
        super(BetaVAE, self).__init__(input_channels, latent_dim)
        self.beta = beta
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence with beta weighting
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

class ConditionalVAE(nn.Module):
    """Conditional VAE with class labels"""
    def __init__(self, input_dim=784, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        # Encoder (input + class label)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder (latent + class label)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
        self.num_classes = num_classes
    
    def encode(self, x, c):
        # Concatenate input with one-hot encoded class
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z, c):
        # Concatenate latent code with class label
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar
```

## 🎯 Training Strategies

### 1. **Standard VAE Training**
```python
def train_vae(model, dataloader, optimizer, device, beta=1.0):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
    return total_loss / len(dataloader.dataset)

# Training loop with annealing
def train_with_annealing(model, dataloader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        # Beta annealing for stable training
        beta = min(1.0, epoch / 50.0)  # Gradually increase beta
        
        avg_loss = train_vae(model, dataloader, optimizer, device, beta)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}, Beta: {beta:.3f}')
            
            # Generate samples
            with torch.no_grad():
                samples = model.sample(16, device)
                save_samples(samples, f'samples_epoch_{epoch}.png')
```

### 2. **Advanced Training Techniques**
```python
class VAETrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    def train_epoch(self, dataloader, optimizer, beta=1.0, warmup=False):
        self.model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            
            # KL warmup
            if warmup and batch_idx < len(dataloader) // 4:
                current_beta = beta * (batch_idx / (len(dataloader) // 4))
            else:
                current_beta = beta
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'num_classes'):  # Conditional VAE
                labels_onehot = F.one_hot(labels, self.model.num_classes).float().to(self.device)
                recon_batch, mu, logvar = self.model(data, labels_onehot)
            else:
                recon_batch, mu, logvar = self.model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, current_beta)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        # Store history
        self.history['loss'].append(epoch_loss / len(dataloader))
        self.history['recon_loss'].append(epoch_recon / len(dataloader))
        self.history['kl_loss'].append(epoch_kl / len(dataloader))
        
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, _, _ = vae_loss(recon_batch, data, mu, logvar)
                total_loss += loss.item()
        
        return total_loss / len(dataloader.dataset)
```

## 📊 Evaluation and Analysis

### 1. **Reconstruction Quality**
```python
def evaluate_reconstruction(model, test_loader, device, num_samples=8):
    model.eval()
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            
            # Plot original vs reconstruction
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
            
            for i in range(num_samples):
                # Original
                axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstruction
                axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()
            break

def compute_reconstruction_error(model, test_loader, device):
    model.eval()
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, _, _ = model(data)
            
            mse = F.mse_loss(recon, data, reduction='sum')
            total_mse += mse.item()
            total_samples += data.size(0)
    
    return total_mse / total_samples
```

### 2. **Latent Space Analysis**
```python
def analyze_latent_space(model, dataloader, device, num_samples=1000):
    model.eval()
    
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            if len(latent_codes) * data.size(0) >= num_samples:
                break
                
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            
            latent_codes.append(z.cpu())
            labels.append(label)
    
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    return latent_codes.numpy(), labels.numpy()

def plot_latent_space_2d(latent_codes, labels, title="Latent Space"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()

def interpolate_in_latent_space(model, z1, z2, num_steps=10):
    """Interpolate between two latent codes"""
    model.eval()
    
    alphas = torch.linspace(0, 1, num_steps)
    interpolations = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = model.decode(z_interp.unsqueeze(0))
            interpolations.append(x_interp.squeeze(0))
    
    return torch.stack(interpolations)
```

### 3. **Disentanglement Metrics**
```python
def compute_beta_vae_metric(model, dataset, device, num_samples=10000):
    """Compute β-VAE disentanglement metric"""
    model.eval()
    
    # Sample latent codes
    z_samples = []
    
    with torch.no_grad():
        for i in range(num_samples // 64 + 1):
            batch_size = min(64, num_samples - i * 64)
            if batch_size <= 0:
                break
                
            # Sample from dataset
            indices = torch.randint(0, len(dataset), (batch_size,))
            batch = torch.stack([dataset[i][0] for i in indices]).to(device)
            
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            z_samples.append(z.cpu())
    
    z_samples = torch.cat(z_samples, dim=0)
    
    # Compute variance for each latent dimension
    latent_variances = torch.var(z_samples, dim=0)
    
    # Disentanglement score (higher variance = more disentangled)
    disentanglement_score = torch.mean(latent_variances)
    
    return disentanglement_score.item(), latent_variances.numpy()

def mutual_information_gap(model, dataset, device):
    """Compute MIG (Mutual Information Gap) metric"""
    # This is a simplified version - full implementation requires
    # ground truth factors of variation
    pass
```

## 🎨 Applications and Use Cases

### 1. **Anomaly Detection**
```python
class VAEAnomalyDetector:
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold = None
        self.threshold_percentile = threshold_percentile
    
    def fit_threshold(self, normal_data_loader, device):
        """Fit threshold on normal data"""
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for data, _ in normal_data_loader:
                data = data.to(device)
                recon, _, _ = self.model(data)
                
                # Compute reconstruction error per sample
                error = F.mse_loss(recon, data, reduction='none')
                error = error.view(error.size(0), -1).mean(dim=1)
                reconstruction_errors.extend(error.cpu().numpy())
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
    def detect_anomalies(self, data, device):
        """Detect anomalies in new data"""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(device)
            recon, _, _ = self.model(data)
            
            # Compute reconstruction error
            error = F.mse_loss(recon, data, reduction='none')
            error = error.view(error.size(0), -1).mean(dim=1)
            
            # Classify as anomaly if error > threshold
            anomalies = error.cpu().numpy() > self.threshold
            
        return anomalies, error.cpu().numpy()
```

### 2. **Data Augmentation**
```python
def generate_augmented_data(model, original_data, device, num_augmentations=5):
    """Generate augmented data using VAE latent space"""
    model.eval()
    augmented_samples = []
    
    with torch.no_grad():
        for data in original_data:
            data = data.unsqueeze(0).to(device)
            
            # Encode to latent space
            mu, logvar = model.encode(data)
            
            # Generate multiple samples from the latent distribution
            for _ in range(num_augmentations):
                z = model.reparameterize(mu, logvar)
                augmented = model.decode(z)
                augmented_samples.append(augmented.squeeze(0).cpu())
    
    return torch.stack(augmented_samples)

def semantic_interpolation(model, class1_samples, class2_samples, device, num_steps=10):
    """Interpolate between different classes in latent space"""
    model.eval()
    
    with torch.no_grad():
        # Encode samples from both classes
        mu1, _ = model.encode(class1_samples.to(device))
        mu2, _ = model.encode(class2_samples.to(device))
        
        # Average latent codes for each class
        z1_mean = torch.mean(mu1, dim=0, keepdim=True)
        z2_mean = torch.mean(mu2, dim=0, keepdim=True)
        
        # Interpolate
        interpolations = []
        for alpha in torch.linspace(0, 1, num_steps):
            z_interp = (1 - alpha) * z1_mean + alpha * z2_mean
            x_interp = model.decode(z_interp)
            interpolations.append(x_interp.squeeze(0))
    
    return torch.stack(interpolations)
```

## 🔍 Debugging and Troubleshooting

### 1. **Common Issues and Solutions**
```python
def diagnose_vae_training(model, dataloader, device):
    """Diagnose common VAE training issues"""
    model.eval()
    
    issues = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            
            # Check for posterior collapse
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            active_dims = (kl_per_dim.mean(0) > 0.1).sum().item()
            
            if active_dims < mu.size(1) * 0.5:
                issues.append(f"Posterior collapse: only {active_dims}/{mu.size(1)} dimensions active")
            
            # Check reconstruction quality
            recon_error = F.mse_loss(recon, data)
            if recon_error > 0.1:
                issues.append(f"Poor reconstruction: MSE = {recon_error:.4f}")
            
            # Check for mode collapse in latent space
            latent_std = torch.std(mu, dim=0).mean()
            if latent_std < 0.1:
                issues.append(f"Low latent diversity: std = {latent_std:.4f}")
            
            break
    
    return issues

def monitor_vae_training(losses, kl_losses, recon_losses):
    """Monitor VAE training progress"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label='Reconstruction')
    plt.plot(kl_losses, label='KL Divergence')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(np.array(kl_losses) / np.array(recon_losses))
    plt.title('KL/Reconstruction Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    
    plt.tight_layout()
    plt.show()
```

## 🎓 Learning Path

### Week 1: Foundations
- Understand VAE mathematical framework
- Implement basic VAE from scratch
- Learn reparameterization trick

### Week 2: Architecture Design
- Build convolutional VAEs
- Experiment with different architectures
- Understand encoder-decoder design

### Week 3: Advanced Variants
- Implement β-VAE for disentanglement
- Try conditional VAEs
- Explore VQ-VAE concepts

### Week 4: Applications
- Build anomaly detection system
- Create data augmentation pipeline
- Analyze latent space properties

## 📚 Additional Resources

### Research Papers
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- "Neural Discrete Representation Learning" (VQ-VAE)
- "Wasserstein Auto-Encoders" (WAE)

### Practical Resources
- [VAE Tutorial by Kingma](https://arxiv.org/abs/1906.02691)
- [Disentangled Representation Learning](https://github.com/google-research/disentanglement_lib)
- [PyTorch VAE Examples](https://github.com/pytorch/examples/tree/master/vae)

---

**Next Module**: [Transformers](../04-Transformers/) →

*Ready to explore the probabilistic world of VAEs? Let's learn to generate with uncertainty! 🎲✨*