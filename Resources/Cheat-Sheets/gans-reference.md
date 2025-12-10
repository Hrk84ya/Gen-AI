# 🎭 GANs Quick Reference

## 🎯 Core Concept

**Generative Adversarial Networks (GANs)** consist of two neural networks competing in a zero-sum game:
- **Generator (G)**: Creates fake data to fool the discriminator
- **Discriminator (D)**: Distinguishes between real and fake data

## 📐 Mathematical Foundation

### Objective Function
```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

Where:
- `x` ~ real data distribution
- `z` ~ noise distribution (usually Gaussian)
- `G(z)` = generated fake data
- `D(x)` = probability that x is real

### Training Process
1. **Train Discriminator**: Maximize ability to distinguish real vs fake
2. **Train Generator**: Minimize discriminator's ability to detect fakes
3. **Alternate**: Repeat until Nash equilibrium

## 🏗️ Architecture Components

### Generator Network
```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, np.prod(img_shape)),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)
```

### Discriminator Network
```python
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)
```

## 🔄 Training Loop

### Basic Training Step
```python
def train_step(generator, discriminator, real_data, optimizer_G, optimizer_D, device):
    batch_size = real_data.size(0)
    
    # Labels
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    
    # ---------------------
    #  Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    
    # Real data
    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output, real_labels)
    
    # Fake data
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_data = generator(z)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, fake_labels)
    
    # Total discriminator loss
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()
    
    # -----------------
    #  Train Generator
    # -----------------
    optimizer_G.zero_grad()
    
    # Generate fake data and get discriminator's opinion
    fake_output = discriminator(fake_data)
    g_loss = criterion(fake_output, real_labels)  # Want D to think it's real
    
    g_loss.backward()
    optimizer_G.step()
    
    return d_loss.item(), g_loss.item()
```

## 🎛️ Hyperparameters

### Common Settings
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning Rate | 0.0002 | Often same for G and D |
| Beta1 (Adam) | 0.5 | Lower than default 0.9 |
| Beta2 (Adam) | 0.999 | Standard value |
| Batch Size | 64-128 | Larger often better |
| Latent Dim | 100 | Can vary 64-512 |

### Architecture Guidelines
```python
# Recommended activation functions
generator_activations = {
    'hidden': nn.LeakyReLU(0.2),
    'output': nn.Tanh()  # For [-1, 1] range
}

discriminator_activations = {
    'hidden': nn.LeakyReLU(0.2),
    'output': nn.Sigmoid()  # For probability
}

# Normalization
generator_norm = nn.BatchNorm1d  # or nn.BatchNorm2d for conv
discriminator_norm = None  # Often no normalization
```

## 🚨 Common Issues & Solutions

### Mode Collapse
**Problem**: Generator produces limited variety of samples

**Solutions**:
```python
# 1. Unrolled GANs
# 2. Minibatch discrimination
# 3. Feature matching loss
def feature_matching_loss(real_features, fake_features):
    return F.mse_loss(fake_features.mean(0), real_features.mean(0))

# 4. Different learning rates
optimizer_G = Adam(generator.parameters(), lr=0.0001)
optimizer_D = Adam(discriminator.parameters(), lr=0.0004)
```

### Training Instability
**Problem**: Loss oscillations, non-convergence

**Solutions**:
```python
# 1. Spectral normalization
from torch.nn.utils import spectral_norm
conv_layer = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))

# 2. Gradient penalty (WGAN-GP)
def gradient_penalty(discriminator, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# 3. Label smoothing
real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Instead of 1.0
fake_labels = torch.ones(batch_size, 1).to(device) * 0.1  # Instead of 0.0
```

### Discriminator Overpowering
**Problem**: Discriminator becomes too good, generator can't learn

**Solutions**:
```python
# 1. Train generator more frequently
for epoch in range(num_epochs):
    for i, real_data in enumerate(dataloader):
        # Train discriminator every iteration
        d_loss = train_discriminator(...)
        
        # Train generator multiple times
        for _ in range(2):  # Train G twice per D update
            g_loss = train_generator(...)

# 2. Add noise to discriminator inputs
def add_noise(data, noise_factor=0.1):
    noise = torch.randn_like(data) * noise_factor
    return data + noise

real_data_noisy = add_noise(real_data)
fake_data_noisy = add_noise(fake_data)
```

## 📊 Evaluation Metrics

### Inception Score (IS)
```python
def inception_score(images, splits=10):
    # Higher is better (typical range: 1-10+)
    # Measures quality and diversity
    pass
```

### Fréchet Inception Distance (FID)
```python
def fid_score(real_images, fake_images):
    # Lower is better (0 = identical distributions)
    # Measures similarity to real data distribution
    pass
```

### Visual Inspection Checklist
- [ ] **Diversity**: Are samples varied?
- [ ] **Quality**: Are images sharp and realistic?
- [ ] **Mode Coverage**: All classes/modes represented?
- [ ] **Artifacts**: Any obvious generation artifacts?

## 🎨 GAN Variants

### DCGAN (Deep Convolutional GAN)
```python
# Key principles:
# 1. Replace pooling with strided convolutions
# 2. Use batch normalization (except G output and D input)
# 3. Remove fully connected layers
# 4. Use ReLU in G (except output: Tanh)
# 5. Use LeakyReLU in D

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # channels x 32 x 32
        )
```

### WGAN (Wasserstein GAN)
```python
# Key changes:
# 1. Remove sigmoid from discriminator (now "critic")
# 2. Use Wasserstein loss
# 3. Clip critic weights

def wasserstein_loss(critic_real, critic_fake):
    return -torch.mean(critic_real) + torch.mean(critic_fake)

# Weight clipping
def clip_weights(model, clip_value=0.01):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)
```

### StyleGAN Concepts
```python
# Key innovations:
# 1. Style-based generator
# 2. Adaptive instance normalization (AdaIN)
# 3. Progressive growing
# 4. Style mixing regularization

def adain(content_features, style_features):
    content_mean = content_features.mean(dim=[2, 3], keepdim=True)
    content_std = content_features.std(dim=[2, 3], keepdim=True)
    style_mean = style_features.mean(dim=[2, 3], keepdim=True)
    style_std = style_features.std(dim=[2, 3], keepdim=True)
    
    normalized = (content_features - content_mean) / content_std
    return normalized * style_std + style_mean
```

## 🛠️ Implementation Tips

### Data Preprocessing
```python
# Normalize to [-1, 1] for Tanh output
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Denormalize for visualization
def denormalize(tensor):
    return (tensor + 1) / 2
```

### Weight Initialization
```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)
```

### Monitoring Training
```python
# Track key metrics
metrics = {
    'g_loss': [],
    'd_loss': [],
    'd_real_acc': [],  # Accuracy on real data
    'd_fake_acc': [],  # Accuracy on fake data
    'gradient_norm_g': [],
    'gradient_norm_d': []
}

# Early stopping conditions
if d_real_acc < 0.6 or d_fake_acc > 0.4:
    print("Discriminator struggling - adjust learning rates")

if abs(g_loss - d_loss) > 5.0:
    print("Loss imbalance detected")
```

## 🎯 Quick Debugging Checklist

### Generator Issues
- [ ] Output range matches data range ([-1,1] with Tanh)
- [ ] Sufficient capacity (not too small)
- [ ] Proper activation functions
- [ ] Batch normalization in hidden layers

### Discriminator Issues
- [ ] Not too powerful (add dropout, reduce capacity)
- [ ] Proper loss function (BCE for vanilla GAN)
- [ ] No batch norm on input layer
- [ ] LeakyReLU activations

### Training Issues
- [ ] Balanced learning rates
- [ ] Proper label smoothing
- [ ] Gradient clipping if needed
- [ ] Monitor loss curves and sample quality

---

**Pro Tip**: Start with a simple GAN on MNIST, then gradually increase complexity! 🚀