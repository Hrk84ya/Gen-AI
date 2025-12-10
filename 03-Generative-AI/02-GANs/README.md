# Generative Adversarial Networks (GANs)

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the GAN framework and adversarial training
- Implement various GAN architectures from scratch
- Master training techniques and stability methods
- Apply GANs to image generation, style transfer, and data augmentation
- Understand advanced GAN variants and their applications
- Evaluate and debug GAN training issues

## 🎭 What are GANs?

Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks compete in a zero-sum game. The **Generator** creates fake data to fool the **Discriminator**, while the **Discriminator** tries to distinguish between real and fake data.

### Core Concept:
```
Generator: "I'll create fake images so realistic you can't tell them apart!"
Discriminator: "I'll catch every fake image you create!"
Result: Through competition, both networks improve, leading to highly realistic generated data.
```

## 🎨 Applications of GANs

### **Image Generation**
- High-resolution face generation (StyleGAN)
- Artwork and creative content creation
- Fashion and product design
- Synthetic dataset creation

### **Image-to-Image Translation**
- Style transfer (artistic styles)
- Colorization of black & white images
- Super-resolution (enhancing image quality)
- Domain adaptation (day to night, summer to winter)

### **Data Augmentation**
- Generating training data for rare classes
- Medical image synthesis
- Balancing imbalanced datasets

### **Creative Applications**
- DeepFakes and face swapping
- Virtual try-on systems
- Game asset generation
- Music and audio synthesis

## 📚 Module Contents

### 1. [GAN Fundamentals](./01_gan_fundamentals.ipynb)
- Mathematical foundation and game theory
- Generator and discriminator architectures
- Loss functions and training dynamics
- Implementation from scratch

### 2. [DCGAN - Deep Convolutional GANs](./02_dcgan_implementation.ipynb)
- Convolutional generator and discriminator
- Best practices for stable training
- Image generation on CIFAR-10 and CelebA
- Architecture guidelines and tips

### 3. [Advanced GAN Variants](./03_advanced_gan_variants.ipynb)
- WGAN and Wasserstein distance
- WGAN-GP with gradient penalty
- LSGAN and least squares loss
- Progressive GAN for high-resolution images

### 4. [Conditional GANs](./04_conditional_gans.ipynb)
- Class-conditional generation
- Pix2Pix for image-to-image translation
- CycleGAN for unpaired translation
- Text-to-image generation

### 5. [StyleGAN and Advanced Architectures](./05_stylegan_architecture.ipynb)
- Style-based generator design
- Adaptive instance normalization (AdaIN)
- Style mixing and latent space control
- Progressive growing techniques

### 6. [GAN Training Techniques](./06_training_techniques.ipynb)
- Stabilizing GAN training
- Mode collapse prevention
- Spectral normalization
- Self-attention mechanisms

### 7. [Evaluation and Applications](./07_evaluation_applications.ipynb)
- Inception Score (IS) and FID
- Qualitative evaluation methods
- Real-world deployment considerations
- Ethical implications and deepfakes

## 🏗️ GAN Architecture Evolution

### Historical Timeline
```
2014: Original GAN (Goodfellow et al.)
2015: DCGAN (Deep Convolutional GAN)
2016: InfoGAN (Information Maximizing GAN)
2017: WGAN (Wasserstein GAN)
2017: Pix2Pix (Image-to-Image Translation)
2017: CycleGAN (Unpaired Image Translation)
2018: Progressive GAN (High-Resolution Generation)
2018: StyleGAN (Style-Based Generator)
2019: StyleGAN2 (Improved Quality and Control)
2020: StyleGAN3 (Alias-Free Generation)
```

## 🔧 Core Mathematical Framework

### 1. **Minimax Game Formulation**
```
min_G max_D V(D,G) = E_{x~p_data(x)}[log D(x)] + E_{z~p_z(z)}[log(1 - D(G(z)))]
```

Where:
- **G**: Generator network
- **D**: Discriminator network  
- **x**: Real data samples
- **z**: Random noise (latent code)
- **p_data(x)**: Real data distribution
- **p_z(z)**: Prior noise distribution (usually Gaussian)

### 2. **Training Algorithm**
```python
def train_gan(generator, discriminator, dataloader, num_epochs):
    for epoch in range(num_epochs):
        for real_batch in dataloader:
            
            # Train Discriminator
            # Maximize log(D(x)) + log(1 - D(G(z)))
            
            # Real data
            real_output = discriminator(real_batch)
            d_loss_real = -torch.mean(torch.log(real_output + 1e-8))
            
            # Fake data
            noise = torch.randn(batch_size, latent_dim)
            fake_batch = generator(noise)
            fake_output = discriminator(fake_batch.detach())
            d_loss_fake = -torch.mean(torch.log(1 - fake_output + 1e-8))
            
            d_loss = d_loss_real + d_loss_fake
            
            # Train Generator
            # Minimize log(1 - D(G(z))) ≡ Maximize log(D(G(z)))
            fake_output = discriminator(fake_batch)
            g_loss = -torch.mean(torch.log(fake_output + 1e-8))
```

## 🏛️ Architecture Implementations

### 1. **Basic GAN (Fully Connected)**
```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

### 2. **DCGAN (Deep Convolutional GAN)**
```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, img_size=64):
        super().__init__()
        
        self.init_size = img_size // 4
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
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=3, img_size=64):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                    nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate size after convolutions
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
```

### 3. **Wasserstein GAN (WGAN)**
```python
class WGANCritic(nn.Module):
    """WGAN Critic (no sigmoid output)"""
    def __init__(self, channels=3, img_size=64):
        super().__init__()
        
        def critic_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                    nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block
        
        self.model = nn.Sequential(
            *critic_block(channels, 16, bn=False),
            *critic_block(16, 32),
            *critic_block(32, 64),
            *critic_block(64, 128),
        )
        
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def wgan_loss(critic_real, critic_fake):
    """Wasserstein loss"""
    return -torch.mean(critic_real) + torch.mean(critic_fake)

def gradient_penalty(critic, real_samples, fake_samples, device):
    """Gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    
    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
```

### 4. **Conditional GAN**
```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
```

## 🎯 Training Techniques and Stabilization

### 1. **Training Loop Best Practices**
```python
def train_gan_stable(generator, discriminator, dataloader, device, 
                    lr_g=0.0002, lr_d=0.0002, beta1=0.5, beta2=0.999):
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            # Configure input
            real_imgs = imgs.to(device)
            batch_size = real_imgs.shape[0]
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
            fake = torch.zeros(batch_size, 1, device=device, requires_grad=False)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            # Loss on real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)
            
            # Loss on fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            
            optimizer_G.zero_grad()
            
            # Generate fake images
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs)
            
            # Generator loss
            g_loss = adversarial_loss(fake_pred, valid)
            g_loss.backward()
            optimizer_G.step()
            
            # Logging and monitoring
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
```

### 2. **Advanced Training Techniques**
```python
# Label smoothing
def get_smooth_labels(batch_size, device, real=True, smooth_factor=0.1):
    if real:
        return torch.ones(batch_size, 1, device=device) - smooth_factor * torch.rand(batch_size, 1, device=device)
    else:
        return torch.zeros(batch_size, 1, device=device) + smooth_factor * torch.rand(batch_size, 1, device=device)

# Feature matching loss
def feature_matching_loss(real_features, fake_features):
    return F.mse_loss(fake_features.mean(0), real_features.mean(0))

# Spectral normalization
from torch.nn.utils import spectral_norm

class SpectralNormDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(512, 1))
        )
    
    def forward(self, x):
        return self.model(x)

# Self-attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        # Attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        return out
```

## 📊 Evaluation Metrics

### 1. **Inception Score (IS)**
```python
def inception_score(images, splits=10):
    """Calculate Inception Score"""
    # Load pre-trained Inception model
    inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    
    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    # Get predictions
    preds = np.zeros((len(images), 1000))
    for i in range(len(images)):
        batch = images[i:i+1]
        preds[i] = get_pred(batch)
    
    # Calculate IS
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
```

### 2. **Fréchet Inception Distance (FID)**
```python
def calculate_fid(real_images, fake_images):
    """Calculate Fréchet Inception Distance"""
    
    def get_activations(images, model, batch_size=50):
        model.eval()
        activations = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            with torch.no_grad():
                pred = model(batch)[0]
                
            # If model output is not scalar, apply global spatial average pooling
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            activations.append(pred.cpu().numpy().reshape(pred.size(0), -1))
        
        return np.concatenate(activations, axis=0)
    
    # Load Inception model
    inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()  # Remove final layer
    
    # Get activations
    real_activations = get_activations(real_images, inception)
    fake_activations = get_activations(fake_images, inception)
    
    # Calculate statistics
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
```

## 🛠️ Debugging and Troubleshooting

### 1. **Common Issues and Solutions**
```python
def diagnose_gan_training(g_losses, d_losses, generated_samples):
    """Diagnose common GAN training issues"""
    
    issues = []
    
    # Check for mode collapse
    if len(np.unique(generated_samples.reshape(len(generated_samples), -1), axis=0)) < len(generated_samples) * 0.8:
        issues.append("Possible mode collapse detected")
    
    # Check loss patterns
    recent_g_loss = np.mean(g_losses[-100:])
    recent_d_loss = np.mean(d_losses[-100:])
    
    if recent_d_loss < 0.1:
        issues.append("Discriminator too strong - generator can't learn")
    
    if recent_g_loss > 5.0:
        issues.append("Generator struggling - discriminator might be too strong")
    
    if abs(recent_g_loss - recent_d_loss) > 3.0:
        issues.append("Severe loss imbalance detected")
    
    # Check for vanishing gradients
    if recent_g_loss < 0.01 and recent_d_loss < 0.01:
        issues.append("Possible vanishing gradients")
    
    return issues

# Monitor gradient norms
def monitor_gradients(model, name="Model"):
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    print(f"{name} - Gradient norm: {total_norm:.4f}, Params with gradients: {param_count}")
    
    return total_norm
```

### 2. **Training Monitoring**
```python
class GANTrainingMonitor:
    def __init__(self, save_interval=100):
        self.save_interval = save_interval
        self.g_losses = []
        self.d_losses = []
        self.iteration = 0
    
    def update(self, g_loss, d_loss, generator, fixed_noise):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        self.iteration += 1
        
        if self.iteration % self.save_interval == 0:
            self.save_samples(generator, fixed_noise)
            self.plot_losses()
    
    def save_samples(self, generator, fixed_noise):
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_noise)
            
        # Save grid of images
        grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
        torchvision.utils.save_image(grid, f'samples/iteration_{self.iteration}.png')
        
        generator.train()
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.savefig(f'losses/losses_iteration_{self.iteration}.png')
        plt.close()
```

## 🎓 Learning Path

### Week 1: Fundamentals
- Understand GAN theory and mathematics
- Implement basic GAN from scratch
- Train on simple datasets (MNIST)

### Week 2: Convolutional GANs
- Learn DCGAN architecture
- Implement stable training techniques
- Generate images on CIFAR-10

### Week 3: Advanced Variants
- Explore WGAN and WGAN-GP
- Implement conditional GANs
- Study mode collapse solutions

### Week 4: Applications
- Build image-to-image translation systems
- Explore StyleGAN concepts
- Deploy GAN applications

## 📚 Additional Resources

### Research Papers
- "Generative Adversarial Networks" (Goodfellow et al., 2014)
- "Unsupervised Representation Learning with Deep Convolutional GANs" (DCGAN)
- "Wasserstein GAN" (Arjovsky et al., 2017)
- "Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2)

### Practical Resources
- [PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)
- [Papers With Code - GANs](https://paperswithcode.com/methods/category/generative-adversarial-networks)
- [GAN Lab - Interactive Visualization](https://poloclub.github.io/ganlab/)

---

**Next Module**: [Variational Autoencoders (VAEs)](../03-VAEs/) →

*Ready to generate some amazing content? Let's master the art of adversarial training! 🎭✨*