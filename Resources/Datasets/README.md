# 📊 Datasets for Generative AI

A comprehensive collection of datasets for learning and experimenting with generative AI models.

## 🎯 Dataset Categories

### 🖼️ **Computer Vision Datasets**
- **Image Classification**: Standard benchmarks for training classifiers
- **Object Detection**: Datasets with bounding box annotations
- **Segmentation**: Pixel-level labeled images
- **Face Recognition**: Facial image datasets
- **Style Transfer**: Artistic and photographic image pairs
- **Medical Imaging**: Healthcare-specific image data

### 📝 **Natural Language Processing**
- **Text Generation**: Large text corpora for language modeling
- **Machine Translation**: Parallel text in multiple languages
- **Sentiment Analysis**: Labeled text with emotional annotations
- **Question Answering**: Q&A pairs for comprehension tasks
- **Dialogue Systems**: Conversational data for chatbots

### 🎵 **Audio and Music**
- **Speech Recognition**: Audio recordings with transcriptions
- **Music Generation**: MIDI files and audio samples
- **Sound Classification**: Labeled audio clips
- **Voice Synthesis**: Speaker-specific audio data

### 🎬 **Multimodal Datasets**
- **Video Understanding**: Video clips with descriptions
- **Image Captioning**: Images paired with text descriptions
- **Visual Question Answering**: Images with Q&A pairs

## 📚 Beginner-Friendly Datasets

### 1. **MNIST** 📊
- **Description**: Handwritten digits (0-9)
- **Size**: 70,000 images (28×28 grayscale)
- **Use Cases**: Basic classification, simple GANs, autoencoders
- **Download**: Built into TensorFlow/PyTorch

```python
# Load MNIST
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PyTorch version
from torchvision import datasets, transforms
dataset = datasets.MNIST(root='./data', train=True, download=True, 
                        transform=transforms.ToTensor())
```

### 2. **CIFAR-10** 🖼️
- **Description**: 10 classes of natural images
- **Size**: 60,000 images (32×32 color)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Use Cases**: Image classification, CNN training, transfer learning

```python
# Load CIFAR-10
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# PyTorch version
dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                          transform=transforms.ToTensor())
```

### 3. **Fashion-MNIST** 👕
- **Description**: Fashion items (clothing, shoes, bags)
- **Size**: 70,000 images (28×28 grayscale)
- **Classes**: 10 fashion categories
- **Use Cases**: Alternative to MNIST, more challenging classification

```python
# Load Fashion-MNIST
dataset = datasets.FashionMNIST(root='./data', train=True, download=True,
                               transform=transforms.ToTensor())
```

## 🎨 Image Generation Datasets

### 1. **CelebA** 👤
- **Description**: Celebrity faces with attributes
- **Size**: 200,000+ images
- **Features**: 40 binary attributes (gender, age, hair color, etc.)
- **Use Cases**: Face generation, attribute manipulation, StyleGAN training

```python
# Download CelebA
import torchvision.datasets as datasets
dataset = datasets.CelebA(root='./data', split='train', download=True,
                         transform=transforms.Compose([
                             transforms.Resize(64),
                             transforms.CenterCrop(64),
                             transforms.ToTensor(),
                         ]))
```

### 2. **FFHQ (Flickr-Faces-HQ)** 🎭
- **Description**: High-quality human faces
- **Size**: 70,000 images (1024×1024)
- **Use Cases**: High-resolution face generation, StyleGAN experiments
- **Download**: [NVIDIA Research](https://github.com/NVlabs/ffhq-dataset)

### 3. **LSUN** 🏠
- **Description**: Large-scale scene understanding
- **Categories**: Bedrooms, churches, conference rooms, dining rooms, etc.
- **Size**: Millions of images
- **Use Cases**: Scene generation, large-scale GAN training

### 4. **ImageNet** 🌍
- **Description**: Large-scale object recognition dataset
- **Size**: 14+ million images, 1000+ classes
- **Use Cases**: Transfer learning, pre-trained model fine-tuning
- **Note**: Requires registration for download

## 📝 Text Generation Datasets

### 1. **WikiText** 📖
- **Description**: Wikipedia articles for language modeling
- **Variants**: WikiText-2, WikiText-103
- **Size**: 2M to 103M tokens
- **Use Cases**: Language model training, text generation

```python
# Load WikiText
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

### 2. **OpenWebText** 🌐
- **Description**: Web pages used to train GPT-2
- **Size**: 40GB of text
- **Use Cases**: Large language model training
- **Download**: Available through Hugging Face

```python
# Load OpenWebText
dataset = load_dataset("openwebtext")
```

### 3. **Common Crawl** 🕷️
- **Description**: Web crawl data
- **Size**: Petabytes of web data
- **Use Cases**: Large-scale language model training
- **Note**: Requires significant preprocessing

### 4. **BookCorpus** 📚
- **Description**: Collection of over 11,000 books
- **Size**: ~1 billion words
- **Use Cases**: Long-form text generation, story writing
- **Access**: Through research agreements

## 🎵 Audio Datasets

### 1. **LibriSpeech** 🎤
- **Description**: English speech recognition corpus
- **Size**: 1000 hours of speech
- **Use Cases**: Speech recognition, voice synthesis
- **Download**: [OpenSLR](http://www.openslr.org/12/)

### 2. **MAESTRO** 🎹
- **Description**: Piano performances with MIDI
- **Size**: 200 hours of piano music
- **Use Cases**: Music generation, audio synthesis
- **Download**: [Magenta](https://magenta.tensorflow.org/datasets/maestro)

### 3. **NSynth** 🎶
- **Description**: Musical note dataset
- **Size**: 300,000+ musical notes
- **Use Cases**: Audio synthesis, timbre transfer
- **Download**: [Magenta](https://magenta.tensorflow.org/datasets/nsynth)

## 🎬 Multimodal Datasets

### 1. **MS COCO** 📷
- **Description**: Images with captions and object annotations
- **Size**: 330K images, 2.5M labeled instances
- **Use Cases**: Image captioning, object detection, VQA
- **Download**: [COCO Dataset](https://cocodataset.org/)

```python
# Load COCO captions
from pycocotools.coco import COCO
coco = COCO('annotations/captions_train2017.json')
```

### 2. **Flickr30k** 🖼️
- **Description**: Images with multiple captions
- **Size**: 31,000 images, 158,000 captions
- **Use Cases**: Image captioning, multimodal learning
- **Download**: [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/)

### 3. **Visual Genome** 👁️
- **Description**: Dense annotations of images
- **Size**: 108K images with scene graphs
- **Use Cases**: Visual reasoning, scene understanding
- **Download**: [Visual Genome](https://visualgenome.org/)

## 🏥 Specialized Datasets

### Medical Imaging
```python
# Medical datasets (require special access)
datasets = {
    'ChestX-ray14': 'Chest X-ray images with disease labels',
    'MIMIC-CXR': 'Chest radiographs with reports',
    'BraTS': 'Brain tumor segmentation',
    'ISIC': 'Skin lesion images',
    'NIH Clinical Center': 'Various medical imaging datasets'
}
```

### Scientific Data
```python
# Scientific datasets
datasets = {
    'Materials Project': 'Crystal structure data',
    'QM9': 'Molecular properties dataset',
    'ZINC': 'Chemical compound database',
    'Protein Data Bank': '3D protein structures'
}
```

## 🛠️ Dataset Utilities

### 1. **Data Loading Template**
```python
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Usage
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomImageDataset('path/to/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2. **Data Preprocessing Pipeline**
```python
def preprocess_images(input_dir, output_dir, target_size=(256, 256)):
    """Preprocess images for training"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            img = img.resize(target_size, Image.LANCZOS)
            
            # Save
            output_path = os.path.join(output_dir, filename)
            img.save(output_path, quality=95)

def create_train_val_split(dataset_dir, train_ratio=0.8):
    """Split dataset into train/validation sets"""
    import random
    
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Create directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    
    # Move files
    for file in train_files:
        shutil.copy(os.path.join(dataset_dir, file), 'data/train/')
    
    for file in val_files:
        shutil.copy(os.path.join(dataset_dir, file), 'data/val/')
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
```

### 3. **Data Augmentation**
```python
# Comprehensive augmentation pipeline
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1)
])

# Validation transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 📊 Dataset Statistics and Analysis

### 1. **Dataset Analysis Tools**
```python
def analyze_dataset(dataset_path):
    """Analyze image dataset statistics"""
    import numpy as np
    from collections import Counter
    
    stats = {
        'total_images': 0,
        'image_sizes': [],
        'file_formats': Counter(),
        'color_modes': Counter()
    }
    
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, filename)
            img = Image.open(img_path)
            
            stats['total_images'] += 1
            stats['image_sizes'].append(img.size)
            stats['file_formats'][img.format] += 1
            stats['color_modes'][img.mode] += 1
    
    # Calculate size statistics
    widths = [size[0] for size in stats['image_sizes']]
    heights = [size[1] for size in stats['image_sizes']]
    
    print(f"Total images: {stats['total_images']}")
    print(f"Average size: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
    print(f"Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    print(f"File formats: {dict(stats['file_formats'])}")
    print(f"Color modes: {dict(stats['color_modes'])}")
    
    return stats

def visualize_dataset_samples(dataset, num_samples=16):
    """Visualize random samples from dataset"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if isinstance(sample, tuple):
            image = sample[0]  # If dataset returns (image, label)
        else:
            image = sample
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()
            image = (image + 1) / 2  # Denormalize if needed
        
        axes[i].imshow(image)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## 🔗 Dataset Download Scripts

### 1. **Automated Download Script**
```python
import requests
import zipfile
import os
from tqdm import tqdm

def download_dataset(url, dataset_name, extract=True):
    """Download and extract dataset"""
    
    # Create directory
    os.makedirs(f'datasets/{dataset_name}', exist_ok=True)
    
    # Download
    print(f"Downloading {dataset_name}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    filename = f'datasets/{dataset_name}/{dataset_name}.zip'
    
    with open(filename, 'wb') as file, tqdm(
        desc=dataset_name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)
    
    # Extract
    if extract and filename.endswith('.zip'):
        print(f"Extracting {dataset_name}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(f'datasets/{dataset_name}')
        
        # Remove zip file
        os.remove(filename)
    
    print(f"Dataset {dataset_name} ready!")

# Example usage
datasets_urls = {
    'flowers': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
    'food101': 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
}

for name, url in datasets_urls.items():
    download_dataset(url, name)
```

## 📋 Dataset Checklist

### Before Starting a Project:
- [ ] **Dataset Size**: Sufficient for your model complexity
- [ ] **Data Quality**: Clean, consistent, well-labeled
- [ ] **Legal Compliance**: Proper licensing and usage rights
- [ ] **Preprocessing Needs**: Understand required transformations
- [ ] **Hardware Requirements**: Storage and memory considerations
- [ ] **Baseline Performance**: Know expected results on the dataset

### During Training:
- [ ] **Data Leakage**: Ensure proper train/val/test splits
- [ ] **Augmentation**: Apply appropriate data augmentation
- [ ] **Monitoring**: Track data loading performance
- [ ] **Validation**: Regular evaluation on held-out data

## 🎯 Recommended Learning Path

### Beginner (Weeks 1-2)
1. Start with MNIST for basic concepts
2. Move to CIFAR-10 for color images
3. Try Fashion-MNIST for variety

### Intermediate (Weeks 3-4)
1. Work with CelebA for face generation
2. Experiment with text datasets (WikiText)
3. Try multimodal datasets (COCO)

### Advanced (Weeks 5+)
1. Use large-scale datasets (ImageNet)
2. Work with domain-specific data (medical, scientific)
3. Create custom datasets for specific applications

## 🤝 Contributing Datasets

### How to Add New Datasets:
1. **Document thoroughly**: Description, size, use cases
2. **Provide code examples**: Loading and preprocessing scripts
3. **Include statistics**: Dataset analysis and visualizations
4. **Ensure accessibility**: Clear download instructions
5. **Legal compliance**: Verify licensing and permissions

### Dataset Submission Template:
```markdown
## Dataset Name
- **Description**: Brief description of the dataset
- **Size**: Number of samples, file sizes
- **Format**: File formats and structure
- **Use Cases**: Recommended applications
- **Download**: Instructions or links
- **License**: Usage rights and restrictions
- **Citation**: How to cite the dataset
```

---

**Ready to find the perfect dataset for your project?** Start with the beginner-friendly options and work your way up! 📊🚀

*Remember: Great models start with great data. Choose wisely and preprocess carefully!*