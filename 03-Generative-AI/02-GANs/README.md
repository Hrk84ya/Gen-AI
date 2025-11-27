# 🥊 Generative Adversarial Networks (GANs)

## 🎯 Learning Objectives
- Understand adversarial training dynamics
- Build Generator and Discriminator networks
- Train stable GANs
- Generate realistic images

## 🏗️ GAN Architecture

### Core Components
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
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
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

## ⚔️ Adversarial Training

### Training Loop
```python
# Initialize models
generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
discriminator = Discriminator(img_shape=(1, 28, 28))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real images
        real_imgs = imgs
        real_labels = torch.ones(imgs.size(0), 1)
        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
        
        # Fake images
        z = torch.randn(imgs.size(0), latent_dim)
        fake_imgs = generator(z)
        fake_labels = torch.zeros(imgs.size(0), 1)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
        
        g_loss.backward()
        optimizer_G.step()
```

## 🎨 Advanced GAN Variants

### DCGAN (Deep Convolutional GAN)
```python
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        
        self.init_size = 7  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
```

## 📊 Training Tips & Tricks

### Stabilizing Training
```python
# 1. Use different learning rates
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0004)

# 2. Label smoothing
real_labels = torch.ones(batch_size, 1) * 0.9  # Instead of 1.0
fake_labels = torch.zeros(batch_size, 1) + 0.1  # Instead of 0.0

# 3. Feature matching loss
def feature_matching_loss(real_features, fake_features):
    return nn.MSELoss()(fake_features.mean(0), real_features.mean(0))
```

## 🔍 Evaluation Metrics

### Inception Score (IS)
```python
from torchvision.models import inception_v3
import torch.nn.functional as F

def inception_score(imgs, batch_size=32, splits=1):
    N = len(imgs)
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    
    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    preds = np.zeros((N, 1000))
    for i in range(0, N, batch_size):
        batch = imgs[i:i+batch_size]
        preds[i:i+batch_size] = get_pred(batch)
    
    # Calculate IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)
```

## 🎯 Applications

### Style Transfer
```python
# CycleGAN for unpaired image-to-image translation
class CycleGAN:
    def __init__(self):
        self.G_AB = Generator()  # A to B
        self.G_BA = Generator()  # B to A
        self.D_A = Discriminator()
        self.D_B = Discriminator()
    
    def cycle_consistency_loss(self, real_A, real_B):
        # A -> B -> A
        fake_B = self.G_AB(real_A)
        recovered_A = self.G_BA(fake_B)
        loss_A = nn.L1Loss()(recovered_A, real_A)
        
        # B -> A -> B
        fake_A = self.G_BA(real_B)
        recovered_B = self.G_AB(fake_A)
        loss_B = nn.L1Loss()(recovered_B, real_B)
        
        return loss_A + loss_B
```

## 🐛 Common Issues & Solutions

### Mode Collapse
```python
# Solution: Use different architectures, learning rates, or losses
# Add diversity loss
def diversity_loss(fake_imgs):
    batch_size = fake_imgs.size(0)
    diversity = 0
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            diversity += torch.norm(fake_imgs[i] - fake_imgs[j])
    return -diversity / (batch_size * (batch_size - 1) / 2)
```

### Training Instability
```python
# Solution: Spectral normalization
from torch.nn.utils import spectral_norm

class StableDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        # ... rest of the network
```

## 📚 Key Papers
- **Generative Adversarial Networks** (Goodfellow et al., 2014)
- **DCGAN** (Radford et al., 2015)
- **Progressive GAN** (Karras et al., 2017)
- **StyleGAN** (Karras et al., 2019)

---
*Continue to [VAEs](../03-VAEs/) or [Transformers](../04-Transformers/) →*