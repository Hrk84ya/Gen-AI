# 📊 Datasets for Learning

## 🎯 Curated Datasets by Category

### 🖼️ Computer Vision

#### Beginner-Friendly
| Dataset | Size | Classes | Description | Use Case |
|---------|------|---------|-------------|----------|
| **MNIST** | 70K images | 10 digits | Handwritten digits (28x28) | Classification, first CNN |
| **Fashion-MNIST** | 70K images | 10 clothing | Fashion items (28x28) | Classification alternative to MNIST |
| **CIFAR-10** | 60K images | 10 objects | Natural images (32x32) | Object classification |

#### Intermediate
| Dataset | Size | Classes | Description | Use Case |
|---------|------|---------|-------------|----------|
| **CIFAR-100** | 60K images | 100 objects | Natural images with more classes | Fine-grained classification |
| **STL-10** | 13K images | 10 objects | Higher resolution (96x96) | Semi-supervised learning |
| **SVHN** | 600K images | 10 digits | Street View House Numbers | Real-world digit recognition |

#### Advanced
| Dataset | Size | Classes | Description | Use Case |
|---------|------|---------|-------------|----------|
| **ImageNet** | 14M images | 1000+ classes | Large-scale object recognition | Transfer learning, benchmarking |
| **COCO** | 330K images | 80 objects | Object detection and segmentation | Object detection, captioning |
| **CelebA** | 200K images | 40 attributes | Celebrity faces with attributes | Face recognition, GANs |

### 📝 Natural Language Processing

#### Text Classification
| Dataset | Size | Classes | Description | Use Case |
|---------|------|---------|-------------|----------|
| **IMDB Reviews** | 50K reviews | 2 (pos/neg) | Movie review sentiment | Sentiment analysis |
| **20 Newsgroups** | 20K posts | 20 topics | Newsgroup posts | Topic classification |
| **AG News** | 120K articles | 4 categories | News categorization | Text classification |

#### Language Modeling
| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **WikiText-103** | 267M tokens | Wikipedia articles | Language modeling |
| **BookCorpus** | 11K books | Book texts | Pre-training language models |
| **Common Crawl** | Billions of pages | Web crawl data | Large-scale language modeling |

#### Question Answering
| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **SQuAD 2.0** | 150K questions | Reading comprehension | Question answering |
| **Natural Questions** | 300K questions | Real Google queries | Open-domain QA |
| **MS MARCO** | 1M queries | Web search queries | Information retrieval |

### 🎵 Audio & Speech

| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **LibriSpeech** | 1000 hours | English speech | Speech recognition |
| **Common Voice** | 9000+ hours | Multilingual speech | Speech recognition |
| **UrbanSound8K** | 8732 clips | Urban sounds | Audio classification |
| **GTZAN** | 1000 clips | Music genres | Music genre classification |

### 🏢 Structured Data

#### Tabular Data
| Dataset | Size | Features | Description | Use Case |
|---------|------|----------|-------------|----------|
| **Titanic** | 891 rows | 12 features | Passenger survival | Binary classification |
| **Boston Housing** | 506 rows | 13 features | House prices | Regression |
| **Wine Quality** | 6497 rows | 11 features | Wine ratings | Multi-class classification |
| **Adult Income** | 48K rows | 14 features | Income prediction | Binary classification |

#### Time Series
| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **Stock Prices** | Daily data | Financial time series | Price prediction |
| **Weather Data** | Hourly data | Meteorological data | Weather forecasting |
| **Energy Consumption** | Hourly data | Power usage | Demand forecasting |

## 🔗 Dataset Sources

### Official Repositories
- **[Kaggle Datasets](https://www.kaggle.com/datasets)** - Largest collection of datasets
- **[UCI ML Repository](https://archive.ics.uci.edu/ml/)** - Classic ML datasets
- **[Google Dataset Search](https://datasetsearch.research.google.com/)** - Search engine for datasets
- **[Papers With Code](https://paperswithcode.com/datasets)** - Research datasets

### Domain-Specific
- **[Hugging Face Datasets](https://huggingface.co/datasets)** - NLP datasets
- **[TensorFlow Datasets](https://www.tensorflow.org/datasets)** - Ready-to-use datasets
- **[PyTorch Datasets](https://pytorch.org/vision/stable/datasets.html)** - Computer vision datasets
- **[OpenML](https://www.openml.org/)** - Machine learning datasets

## 📥 Loading Datasets

### Using Built-in Libraries
```python
# TensorFlow/Keras
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Scikit-learn
from sklearn.datasets import load_iris, fetch_20newsgroups
iris = load_iris()
newsgroups = fetch_20newsgroups(subset='train')

# PyTorch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
```

### Using Hugging Face
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Access data
print(train_data[0])  # First example
```

### Using Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle datasets download -d username/dataset-name

# Unzip
unzip dataset-name.zip
```

## 🛠️ Dataset Preparation

### Data Splitting
```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Time series split
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Data Augmentation
```python
# Image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Text augmentation
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(text)
```

## 📊 Dataset Analysis

### Exploratory Data Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('dataset.csv')

# Basic info
print(df.info())
print(df.describe())
print(df.head())

# Missing values
print(df.isnull().sum())

# Visualizations
plt.figure(figsize=(12, 8))

# Distribution of target variable
plt.subplot(2, 2, 1)
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution')

# Correlation heatmap
plt.subplot(2, 2, 2)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')

# Feature distributions
plt.subplot(2, 2, 3)
df.hist(bins=20, figsize=(12, 8))
plt.title('Feature Distributions')

plt.tight_layout()
plt.show()
```

## 🎯 Dataset Selection Guide

### For Beginners
1. **Start Small**: Use datasets like MNIST, Iris, or Titanic
2. **Clear Objectives**: Choose datasets with well-defined problems
3. **Good Documentation**: Ensure datasets have clear descriptions
4. **Balanced Classes**: Avoid highly imbalanced datasets initially

### For Intermediate Learners
1. **Real-world Data**: Use datasets with noise and missing values
2. **Multiple Modalities**: Try combining text, images, or audio
3. **Larger Scale**: Work with datasets that require optimization
4. **Domain-specific**: Choose datasets from your field of interest

### For Advanced Practitioners
1. **Research Datasets**: Use datasets from recent papers
2. **Custom Collection**: Create your own datasets
3. **Multi-task Learning**: Datasets supporting multiple objectives
4. **Streaming Data**: Real-time or continuously updating datasets

## 🔍 Data Quality Checklist

### Before Using Any Dataset
- [ ] **Size**: Is it large enough for your model?
- [ ] **Quality**: Are there missing values or errors?
- [ ] **Bias**: Does it represent your target population?
- [ ] **Licensing**: Can you use it for your purpose?
- [ ] **Documentation**: Is it well-documented?
- [ ] **Splits**: Are train/test splits provided or needed?
- [ ] **Baseline**: Are there benchmark results available?

### Red Flags to Avoid
- ❌ **Data Leakage**: Future information in training data
- ❌ **Sampling Bias**: Non-representative samples
- ❌ **Label Noise**: Incorrect or inconsistent labels
- ❌ **Temporal Issues**: Time-based data mixed incorrectly
- ❌ **Privacy Concerns**: Sensitive personal information

## 📚 Creating Custom Datasets

### Web Scraping
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract data based on HTML structure
    data = []
    for item in soup.find_all('div', class_='item'):
        data.append({
            'title': item.find('h2').text,
            'description': item.find('p').text
        })
    
    return pd.DataFrame(data)
```

### API Data Collection
```python
import requests
import json

def collect_api_data(api_url, params):
    response = requests.get(api_url, params=params)
    data = response.json()
    
    # Process and structure data
    processed_data = []
    for item in data['results']:
        processed_data.append({
            'id': item['id'],
            'text': item['text'],
            'label': item['category']
        })
    
    return processed_data
```

### Data Annotation
```python
# Simple annotation interface
def annotate_data(data_samples):
    annotations = []
    
    for sample in data_samples:
        print(f"Sample: {sample}")
        label = input("Enter label (0/1): ")
        annotations.append({
            'sample': sample,
            'label': int(label)
        })
    
    return annotations
```

## 🚀 Best Practices

### Data Management
1. **Version Control**: Track dataset versions
2. **Documentation**: Maintain clear metadata
3. **Backup**: Keep multiple copies of important datasets
4. **Organization**: Use consistent folder structures
5. **Validation**: Regularly check data integrity

### Ethical Considerations
1. **Privacy**: Respect individual privacy rights
2. **Consent**: Ensure proper data collection consent
3. **Bias**: Actively work to reduce dataset bias
4. **Transparency**: Document data collection methods
5. **Compliance**: Follow relevant regulations (GDPR, etc.)

---

*Choose the right dataset for your learning level and project goals! 🎯*