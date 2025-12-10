# 🎨 GAN Image Generation Project

## Project Overview
Build and train Generative Adversarial Networks (GANs) to generate high-quality synthetic images. This project covers various GAN architectures from basic GANs to advanced models like StyleGAN and Progressive GANs, with applications in art generation, data augmentation, and creative AI.

## 🎯 Learning Objectives
- Understand GAN theory and training dynamics
- Implement different GAN architectures (DCGAN, WGAN, StyleGAN)
- Master techniques for stable GAN training
- Generate high-quality synthetic images
- Apply GANs for practical applications

## 🏗️ Project Structure
```
03-GAN-Images/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── generated/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── basic_gan.py
│   │   ├── dcgan.py
│   │   ├── wgan.py
│   │   ├── stylegan.py
│   │   └── progressive_gan.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── visualization.py
│   │   └── image_utils.py
│   └── evaluation/
│       ├── fid_score.py
│       ├── inception_score.py
│       └── quality_metrics.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_basic_gan.ipynb
│   ├── 03_dcgan_training.ipynb
│   ├── 04_advanced_gans.ipynb
│   └── 05_evaluation_metrics.ipynb
├── configs/
│   ├── basic_gan.yaml
│   ├── dcgan.yaml
│   └── stylegan.yaml
├── experiments/
│   ├── basic_gan/
│   ├── dcgan/
│   └── stylegan/
├── web_app/
│   ├── app.py
│   ├── templates/
│   └── static/
└── deployment/
    ├── Dockerfile
    └── model_server.py
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Navigate to project
cd Projects/03-GAN-Images

# Create virtual environment
python -m venv gan_env
source gan_env/bin/activate  # On Windows: gan_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets
python src/utils/data_loader.py --download --dataset celeba
```

### 2. Train Your First GAN
```bash
# Basic GAN training
python src/training/trainer.py --config configs/basic_gan.yaml --dataset mnist

# DCGAN training
python src/training/trainer.py --config configs/dcgan.yaml --dataset celeba

# Monitor training with TensorBoard
tensorboard --logdir experiments/
```

### 3. Generate Images
```bash
# Generate images with trained model
python src/generate.py --model experiments/dcgan/checkpoints/best_model.pth --num_images 100

# Interactive generation web app
python web_app/app.py
```

## 📊 Supported Datasets

### Image Datasets
- **MNIST**: Handwritten digits (28x28, grayscale)
- **CIFAR-10**: Natural images (32x32, RGB)
- **CelebA**: Celebrity faces (64x64, RGB)
- **FFHQ**: High-quality faces (1024x1024, RGB)
- **Custom**: Your own image datasets

### Dataset Statistics
| Dataset | Images | Resolution | Classes | Size |
|---------|--------|------------|---------|------|
| MNIST | 70K | 28x28 | 10 | 11MB |
| CIFAR-10 | 60K | 32x32 | 10 | 163MB |
| CelebA | 200K | 64x64 | - | 1.3GB |
| FFHQ | 70K | 1024x1024 | - | 89GB |

## 🧠 GAN Architectures

### 1. Basic GAN
**Architecture**: Simple fully connected networks
```python
# Generator: noise -> image
Generator: Linear(100) -> Linear(256) -> Linear(512) -> Linear(784) -> Tanh

# Discriminator: image -> real/fake
Discriminator: Linear(784) -> Linear(512) -> Linear(256) -> Linear(1) -> Sigmoid
```

**Use Cases**: 
- Learning GAN fundamentals
- Simple datasets (MNIST)
- Proof of concept

### 2. Deep Convolutional GAN (DCGAN)
**Architecture**: Convolutional layers with specific design principles
```python
# Generator: Uses transposed convolutions
noise(100) -> ConvTranspose2d -> BatchNorm -> ReLU -> ... -> Tanh

# Discriminator: Uses standard convolutions
image -> Conv2d -> LeakyReLU -> ... -> Conv2d -> Sigmoid
```

**Features**:
- Stable training with architectural guidelines
- High-quality image generation
- Good for 64x64 and 128x128 images

### 3. Wasserstein GAN (WGAN)
**Architecture**: DCGAN with Wasserstein loss
```python
# Uses Earth Mover's Distance instead of JS divergence
# Critic (not discriminator) outputs real numbers
# Weight clipping for Lipschitz constraint
```

**Advantages**:
- More stable training
- Meaningful loss curves
- Reduced mode collapse

### 4. StyleGAN
**Architecture**: Style-based generator with adaptive instance normalization
```python
# Mapping Network: z -> w
# Synthesis Network: w -> image with style injection at each layer
# Progressive growing for high-resolution generation
```

**Features**:
- State-of-the-art quality
- Style control and mixing
- High-resolution generation (1024x1024+)

### 5. Progressive GAN
**Architecture**: Gradually growing generator and discriminator
```python
# Start with 4x4, progressively add layers to reach 1024x1024
# Smooth transition between resolutions
# Minibatch standard deviation for diversity
```

**Benefits**:
- Stable high-resolution training
- Faster convergence
- Better quality at high resolutions

## 🔧 Training Techniques

### Stabilization Methods
```python
# 1. Spectral Normalization
def spectral_norm(module):
    return torch.nn.utils.spectral_norm(module)

# 2. Gradient Penalty (WGAN-GP)
def gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(d_interpolates, interpolates)[0]
    
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# 3. Feature Matching
def feature_matching_loss(real_features, fake_features):
    return F.mse_loss(fake_features.mean(0), real_features.mean(0))
```

### Advanced Training Strategies
```python
# 1. Two Time-Scale Update Rule (TTUR)
g_optimizer = Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_optimizer = Adam(discriminator.parameters(), lr=0.0004, betas=(0.0, 0.9))

# 2. Exponential Moving Average (EMA)
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    
    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
            else:
                self.shadow[name] = param.data.clone()

# 3. Progressive Growing
class ProgressiveTrainer:
    def __init__(self):
        self.current_resolution = 4
        self.transition_phase = False
        
    def should_grow(self, iteration):
        return iteration % self.growth_interval == 0
    
    def grow_networks(self):
        self.current_resolution *= 2
        # Add new layers to generator and discriminator
```

## 📈 Evaluation Metrics

### 1. Fréchet Inception Distance (FID)
```python
def calculate_fid(real_images, generated_images):
    """
    Calculate FID score between real and generated images
    Lower is better (0 = identical distributions)
    """
    # Extract features using pre-trained Inception network
    real_features = extract_inception_features(real_images)
    fake_features = extract_inception_features(generated_images)
    
    # Calculate statistics
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid
```

### 2. Inception Score (IS)
```python
def inception_score(images, splits=10):
    """
    Calculate Inception Score
    Higher is better (measures quality and diversity)
    """
    # Get predictions from Inception network
    preds = get_inception_predictions(images)
    
    # Calculate IS
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits:(i + 1) * len(preds) // splits]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        scores.append(np.exp(np.mean(np.sum(kl_div, 1))))
    
    return np.mean(scores), np.std(scores)
```

### 3. Perceptual Path Length (PPL)
```python
def perceptual_path_length(generator, latent_codes, epsilon=1e-4):
    """
    Measure smoothness of generator's latent space
    Lower is better (smoother interpolations)
    """
    # Generate images from latent codes
    images1 = generator(latent_codes)
    
    # Perturb latent codes slightly
    noise = torch.randn_like(latent_codes) * epsilon
    images2 = generator(latent_codes + noise)
    
    # Calculate perceptual distance
    lpips_distance = calculate_lpips_distance(images1, images2)
    
    # Normalize by latent space distance
    ppl = lpips_distance / (epsilon * np.sqrt(latent_codes.shape[1]))
    
    return ppl.mean()
```

## 🎨 Applications and Use Cases

### 1. Art and Creative Generation
```python
class ArtisticGAN:
    def __init__(self, style='impressionist'):
        self.generator = load_pretrained_stylegan()
        self.style_vectors = load_artistic_styles()
    
    def generate_artwork(self, content_image=None, style='random'):
        if style == 'random':
            style_vector = random.choice(self.style_vectors)
        else:
            style_vector = self.style_vectors[style]
        
        if content_image:
            # Style transfer mode
            latent_code = self.encode_image(content_image)
            artwork = self.generator(latent_code, style_vector)
        else:
            # Pure generation mode
            latent_code = torch.randn(1, 512)
            artwork = self.generator(latent_code, style_vector)
        
        return artwork
```

### 2. Data Augmentation
```python
class DataAugmentationGAN:
    def __init__(self, trained_gan_path):
        self.generator = load_model(trained_gan_path)
        self.generator.eval()
    
    def augment_dataset(self, original_dataset, augmentation_factor=2):
        augmented_data = []
        
        for _ in range(len(original_dataset) * augmentation_factor):
            # Generate synthetic sample
            noise = torch.randn(1, 100)
            synthetic_image = self.generator(noise)
            
            # Add to augmented dataset
            augmented_data.append(synthetic_image)
        
        return augmented_data
    
    def class_balanced_augmentation(self, dataset, target_samples_per_class):
        """Generate samples to balance class distribution"""
        class_counts = count_samples_per_class(dataset)
        
        for class_id, current_count in class_counts.items():
            needed_samples = target_samples_per_class - current_count
            
            if needed_samples > 0:
                # Generate samples for this class
                class_samples = self.generate_class_samples(class_id, needed_samples)
                dataset.extend(class_samples)
        
        return dataset
```

### 3. Image Editing and Manipulation
```python
class ImageEditor:
    def __init__(self, stylegan_model):
        self.generator = stylegan_model
        self.encoder = load_image_encoder()
    
    def edit_image(self, image, edit_direction, strength=1.0):
        # Encode image to latent space
        latent_code = self.encoder(image)
        
        # Apply edit direction
        edited_latent = latent_code + strength * edit_direction
        
        # Generate edited image
        edited_image = self.generator(edited_latent)
        
        return edited_image
    
    def interpolate_images(self, image1, image2, steps=10):
        # Encode both images
        latent1 = self.encoder(image1)
        latent2 = self.encoder(image2)
        
        # Create interpolation
        interpolated_images = []
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
            interpolated_image = self.generator(interpolated_latent)
            interpolated_images.append(interpolated_image)
        
        return interpolated_images
```

### 4. Super-Resolution
```python
class SuperResolutionGAN:
    def __init__(self):
        self.generator = self._build_srgan_generator()
        self.discriminator = self._build_srgan_discriminator()
    
    def _build_srgan_generator(self):
        # Generator with residual blocks and sub-pixel convolution
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 64, 9, padding=4),
            nn.PReLU(),
            
            # Residual blocks
            *[ResidualBlock(64) for _ in range(16)],
            
            # Upsampling blocks
            UpsampleBlock(64, 256),
            UpsampleBlock(64, 256),
            
            # Final convolution
            nn.Conv2d(64, 3, 9, padding=4),
            nn.Tanh()
        )
    
    def enhance_image(self, low_res_image):
        with torch.no_grad():
            high_res_image = self.generator(low_res_image)
        return high_res_image
```

## 🌐 Web Application

### Interactive Generation Interface
```python
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)
gan_model = load_trained_model('models/stylegan.pth')

@app.route('/')
def index():
    return render_template('generator.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    # Get parameters from request
    seed = request.json.get('seed', None)
    style = request.json.get('style', 'random')
    
    # Generate image
    if seed:
        torch.manual_seed(seed)
    
    noise = torch.randn(1, 512)
    generated_image = gan_model(noise)
    
    # Convert to base64 for web display
    image_b64 = tensor_to_base64(generated_image)
    
    return jsonify({
        'image': image_b64,
        'seed': seed,
        'style': style
    })

@app.route('/interpolate', methods=['POST'])
def interpolate_images():
    seed1 = request.json.get('seed1')
    seed2 = request.json.get('seed2')
    steps = request.json.get('steps', 10)
    
    # Generate interpolation
    torch.manual_seed(seed1)
    noise1 = torch.randn(1, 512)
    
    torch.manual_seed(seed2)
    noise2 = torch.randn(1, 512)
    
    interpolated_images = []
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated_noise = (1 - alpha) * noise1 + alpha * noise2
        image = gan_model(interpolated_noise)
        interpolated_images.append(tensor_to_base64(image))
    
    return jsonify({'images': interpolated_images})
```

## 📊 Training Monitoring

### Real-time Metrics Dashboard
```python
import wandb
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    def __init__(self, experiment_name):
        self.writer = SummaryWriter(f'runs/{experiment_name}')
        wandb.init(project='gan-images', name=experiment_name)
        
        self.metrics = {
            'g_loss': [],
            'd_loss': [],
            'fid_score': [],
            'is_score': []
        }
    
    def log_iteration(self, iteration, g_loss, d_loss, real_images, fake_images):
        # Log losses
        self.writer.add_scalar('Loss/Generator', g_loss, iteration)
        self.writer.add_scalar('Loss/Discriminator', d_loss, iteration)
        
        wandb.log({
            'iteration': iteration,
            'g_loss': g_loss,
            'd_loss': d_loss
        })
        
        # Log images every N iterations
        if iteration % 100 == 0:
            self.writer.add_images('Real Images', real_images[:8], iteration)
            self.writer.add_images('Generated Images', fake_images[:8], iteration)
    
    def log_epoch(self, epoch, fid_score, is_score):
        self.writer.add_scalar('Metrics/FID', fid_score, epoch)
        self.writer.add_scalar('Metrics/IS', is_score, epoch)
        
        wandb.log({
            'epoch': epoch,
            'fid_score': fid_score,
            'is_score': is_score
        })
        
        # Save metrics
        self.metrics['fid_score'].append(fid_score)
        self.metrics['is_score'].append(is_score)
    
    def plot_training_curves(self):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['g_loss'], label='Generator')
        axes[0, 0].plot(self.metrics['d_loss'], label='Discriminator')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        
        # FID score
        axes[0, 1].plot(self.metrics['fid_score'])
        axes[0, 1].set_title('FID Score (Lower is Better)')
        
        # IS score
        axes[1, 0].plot(self.metrics['is_score'])
        axes[1, 0].set_title('Inception Score (Higher is Better)')
        
        # Generated samples grid
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
```

## 🔧 Advanced Features

### 1. Conditional Generation
```python
class ConditionalGAN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.generator = self._build_conditional_generator()
        self.discriminator = self._build_conditional_discriminator()
    
    def _build_conditional_generator(self):
        # Generator that takes both noise and class label
        return ConditionalGenerator(
            noise_dim=100,
            num_classes=self.num_classes,
            embed_dim=50
        )
    
    def generate_class_samples(self, class_label, num_samples=1):
        noise = torch.randn(num_samples, 100)
        labels = torch.full((num_samples,), class_label, dtype=torch.long)
        
        generated_images = self.generator(noise, labels)
        return generated_images
```

### 2. Style Transfer Integration
```python
class StyleTransferGAN:
    def __init__(self):
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.generator = StyleGenerator()
    
    def transfer_style(self, content_image, style_image):
        # Extract content and style features
        content_features = self.content_encoder(content_image)
        style_features = self.style_encoder(style_image)
        
        # Generate stylized image
        stylized_image = self.generator(content_features, style_features)
        
        return stylized_image
    
    def multi_style_transfer(self, content_image, style_images, weights):
        content_features = self.content_encoder(content_image)
        
        # Weighted combination of style features
        combined_style = torch.zeros_like(self.style_encoder(style_images[0]))
        
        for style_image, weight in zip(style_images, weights):
            style_features = self.style_encoder(style_image)
            combined_style += weight * style_features
        
        stylized_image = self.generator(content_features, combined_style)
        return stylized_image
```

### 3. Latent Space Exploration
```python
class LatentSpaceExplorer:
    def __init__(self, generator):
        self.generator = generator
        self.pca = None
        self.latent_directions = {}
    
    def find_semantic_directions(self, attribute_classifier, num_samples=10000):
        """Find directions in latent space corresponding to semantic attributes"""
        # Generate random samples
        latent_codes = torch.randn(num_samples, 512)
        generated_images = self.generator(latent_codes)
        
        # Classify attributes
        attributes = attribute_classifier(generated_images)
        
        # Find directions using linear separation
        for attr_name in attributes.columns:
            positive_samples = latent_codes[attributes[attr_name] > 0.5]
            negative_samples = latent_codes[attributes[attr_name] < 0.5]
            
            # Calculate direction as difference of means
            direction = positive_samples.mean(0) - negative_samples.mean(0)
            self.latent_directions[attr_name] = direction
    
    def edit_attribute(self, latent_code, attribute, strength=1.0):
        """Edit specific attribute in generated image"""
        if attribute not in self.latent_directions:
            raise ValueError(f"Direction for {attribute} not found")
        
        direction = self.latent_directions[attribute]
        edited_latent = latent_code + strength * direction
        
        return self.generator(edited_latent)
    
    def interpolate_attributes(self, latent_code, attributes, strengths):
        """Apply multiple attribute edits simultaneously"""
        edited_latent = latent_code.clone()
        
        for attribute, strength in zip(attributes, strengths):
            if attribute in self.latent_directions:
                direction = self.latent_directions[attribute]
                edited_latent += strength * direction
        
        return self.generator(edited_latent)
```

## 📚 Learning Resources

### Research Papers
- **Original GAN**: "Generative Adversarial Networks" (Goodfellow et al., 2014)
- **DCGAN**: "Unsupervised Representation Learning with Deep Convolutional GANs"
- **WGAN**: "Wasserstein GAN" (Arjovsky et al., 2017)
- **StyleGAN**: "A Style-Based Generator Architecture for GANs"
- **Progressive GAN**: "Progressive Growing of GANs for Improved Quality"

### Books and Courses
- "Generative Deep Learning" by David Foster
- "Deep Learning" by Ian Goodfellow (Chapter 20: Deep Generative Models)
- CS231n: Convolutional Neural Networks (Stanford)
- Deep Learning Specialization (Coursera)

### Online Resources
- [GAN Lab](https://poloclub.github.io/ganlab/): Interactive GAN visualization
- [This Person Does Not Exist](https://thispersondoesnotexist.com/): StyleGAN demo
- [Papers With Code - GANs](https://paperswithcode.com/methods/category/generative-adversarial-networks)

## 🚀 Future Enhancements

### Planned Features
1. **3D GAN**: Generate 3D objects and scenes
2. **Video GAN**: Temporal consistency for video generation
3. **Text-to-Image**: Generate images from text descriptions
4. **Few-Shot Learning**: Adapt to new domains with minimal data
5. **Controllable Generation**: Fine-grained attribute control

### Research Directions
- **Improved Training Stability**: Better loss functions and training procedures
- **Disentangled Representations**: Separate control over different attributes
- **Efficiency**: Faster training and inference
- **Evaluation Metrics**: Better measures of generation quality
- **Ethical Considerations**: Addressing deepfake concerns and bias

---

**Next Steps**: After mastering GAN image generation, explore the [Text Generation Project](../04-Text-Generation/) to learn about language models and text synthesis.